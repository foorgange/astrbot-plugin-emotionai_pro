# image_format_guard.py
"""图片格式守卫：把不支持的图片格式拦在 LLM 请求之外。

背景（v4.1.6 线上日志排查）：
    QQ 群里有用户发送 SVG 等非通用格式图片。框架的图片管线
    （MediaResolver）只做「模态」粒度处理——provider 声明支持
    image 模态时原样放行，格式是否被模型认识并不校验。于是
    data:image/svg+xml;base64,... 会带着进请求：

        step-5-preview  400 Cannot decode visions
        商汤/deepseek   400 invalid image base64 content
        商汤/glm        400 invalid image base64 content
        商汤/sensenova  400 invalid image base64 content

    一条带 SVG 图的消息要连切三四个 provider，直到 fallback
    链尾的模型兜住，耗时 30 秒以上——用户侧看到的就是
    「回复出奇地慢」，而且每切一个 provider 都白烧一次配额
    （step 的 rpm 配额还被这些无效请求挤占，连带情感分析
    LLM 链一起踩 429）。

    框架自身在 provider 不声明 image 模态时会丢弃图片并替换
    为 [Image] 文本占位，这里借用同样的占位风格，在插件侧
    补齐格式粒度的过滤。

设计约束：
    1. 只影响本轮 LLM 请求，不改数据库里的对话记录——与
       v4.1.0 上下文保鲜同一哲学：命中的消息整体拷贝出新
       dict，传入的 contexts/image_urls 不被原地修改；
    2. 保守放行：判不出格式的（http URL、无后缀临时文件、
       白名单内格式）一律放过——误杀一张正常图片比多绕一个
       provider 更糟；
    3. 调用方必须 try 包裹：本模块任何异常都不应阻断 LLM
       请求，过滤失败时保持框架默认行为。
"""
from urllib.parse import unquote, urlparse

# 主流模型实际支持的图片 mime（与 step-5-preview 报错信息中列出的
# 白名单对齐）。application/octet-stream 是部分渠道传图的兜底
# 形态，保守放行。
SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/gif",
        "application/octet-stream",
    }
)

# 框架在丢弃图片时使用的占位文本（见 tool_loop_agent_runner
# _assemble_request_context_for_provider），保持风格一致。
UNSUPPORTED_IMAGE_PLACEHOLDER = "[Image]"

# 已知的图片文件后缀 → mime。只收录「能明确判定格式」的后缀：
# 白名单内的走 SUPPORTED_IMAGE_MIME_TYPES，其余已知图片格式
# 视为不支持。
_KNOWN_SUFFIX_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".svgz": "image/svg+xml",
    ".bmp": "image/bmp",
    ".dib": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".ico": "image/x-icon",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
}

# 分类结果三态：
#   supported  —— 白名单内，放行
#   unsupported—— 确定不被支持，拦截
#   unknown    —— 判不出格式，保守放行
CLASSIFY_SUPPORTED = "supported"
CLASSIFY_UNSUPPORTED = "unsupported"
CLASSIFY_UNKNOWN = "unknown"


def extract_data_uri_mime(url):
    """从 data URI 中提取 mime 类型。

    "data:image/png;base64,...." -> "image/png"
    不是 data URI、或 mime 段为空/畸形时返回 None。
    """
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header = url[len("data:"):].split(",", 1)[0]
    mime = header.split(";", 1)[0].strip().lower()
    # 合法 mime 必为 type/subtype 形态；"base64"（无 mime 的简写
    # data:base64,xxx）与空段都判为解析失败
    return mime if mime and "/" in mime else None


def guess_mime_from_ref(ref):
    """从本地路径 / http URL / file:// URI 的文件名猜 mime。

    只认后缀，猜不出返回 None（调用方按 unknown 放行）。
    query 串（?xxx=1）与 fragment 会先剥掉。
    """
    if not isinstance(ref, str) or not ref:
        return None
    candidate = ref
    if candidate.startswith("file://"):
        try:
            candidate = urlparse(candidate).path
        except ValueError:
            return None
    elif "://" in candidate:
        try:
            candidate = urlparse(candidate).path
        except ValueError:
            return None
    candidate = unquote(candidate)
    # 取最后一段路径的后缀；query 已在 urlparse 时剥掉（file://
    # 与裸路径没有 query，无需再处理）
    name = candidate.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if "." not in name:
        return None
    suffix = "." + name.rsplit(".", 1)[-1].lower()
    return _KNOWN_SUFFIX_MIME_TYPES.get(suffix)


def classify_image_url(url, supported_mimes=None):
    """把单个图片引用分类为 supported / unsupported / unknown。

    data URI：白名单内 supported；解析出 mime 但不在白名单
    unsupported；畸形（解析不出 mime）unsupported——已经是
    data URI 形态还解析不出 mime，发给任何 provider 都解不了。

    路径 / URL：后缀能猜出且是已知图片格式时，按白名单判定；
    猜不出后缀 unknown。
    """
    supported = (
        SUPPORTED_IMAGE_MIME_TYPES
        if supported_mimes is None
        else frozenset(supported_mimes)
    )
    if not isinstance(url, str) or not url:
        return CLASSIFY_UNKNOWN
    if url.startswith("data:"):
        mime = extract_data_uri_mime(url)
        if mime is None:
            return CLASSIFY_UNSUPPORTED
        return (
            CLASSIFY_SUPPORTED if mime in supported else CLASSIFY_UNSUPPORTED
        )
    mime = guess_mime_from_ref(url)
    if mime is None:
        return CLASSIFY_UNKNOWN
    return CLASSIFY_SUPPORTED if mime in supported else CLASSIFY_UNSUPPORTED


def _extract_part_url(part):
    """从 image_url part 中取出 URL 字符串，结构异常返回 None。

    兼容两种形态：
      框架真实形态  {"image_url": {"url": "...", "id": ...}}
      简化/手写形态 {"image_url": "http://..."}
    """
    if not isinstance(part, dict):
        return None
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return image_url or None
    if isinstance(image_url, dict):
        url = image_url.get("url")
        return url if isinstance(url, str) and url else None
    return None


def filter_unsupported_image_parts(
    contexts,
    supported_mimes=None,
    placeholder_text=UNSUPPORTED_IMAGE_PLACEHOLDER,
):
    """过滤 OpenAI 格式消息列表中不支持的图片 part。

    Args:
        contexts: req.contexts，OpenAI 格式 dict 消息列表。
        supported_mimes: 覆盖默认白名单（测试/调试用）。
        placeholder_text: 替换用的文本占位。

    Returns:
        (new_contexts, replaced_count)。new_contexts 是新 list；
        未命中的消息原样引用，命中的消息是拷贝出的新 dict
        （原始数据不被修改，数据库记录不受影响）。
    """
    if not contexts:
        return contexts, 0

    new_contexts = []
    replaced = 0
    for msg in contexts:
        if not isinstance(msg, dict):
            new_contexts.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            new_contexts.append(msg)
            continue

        new_content = None
        for idx, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = _extract_part_url(part)
            if url is None:
                continue
            if classify_image_url(url, supported_mimes) != CLASSIFY_UNSUPPORTED:
                continue
            if new_content is None:
                new_content = list(content)
            new_content[idx] = {"type": "text", "text": placeholder_text}
            replaced += 1

        if new_content is None:
            new_contexts.append(msg)
        else:
            new_msg = dict(msg)
            new_msg["content"] = new_content
            new_contexts.append(new_msg)

    return new_contexts, replaced


def filter_image_urls(image_urls, supported_mimes=None):
    """过滤本回合图片 URL 列表（req.image_urls）。

    只移除 unsupported 的项；unknown（判不出格式）一律保留。
    返回新 list，不修改传入列表。
    """
    if not image_urls:
        return list(image_urls) if image_urls else []
    return [
        url
        for url in image_urls
        if classify_image_url(url, supported_mimes) != CLASSIFY_UNSUPPORTED
    ]
