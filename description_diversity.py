# description_diversity.py
"""关系 / 态度描述的差异化工具（v4.1.15）

问题背景（线上数据，2026-10-19）：
261 个用户的「关系 + 态度」描述里，`傲娇` 出现 104 次、`撒娇` 63 次、
`俏皮` 55 次、`互怼` 43 次、`调侃` 41 次。大量用户拿到的只是同一组
人设套话的排列组合，例如「傲娇虚拟主播与主人的互怼日常」「互怼互关心
的损友」。根因有三条：

1. 提示词只要求「生动有趣」，没给本轮从哪个角度写，模型默认回到
   人设词库；
2. 提示词不告诉模型上一版写了什么、最近写过什么，模型不知道自己在
   重复；
3. 大模型没返回字段、JSON 解析失败或本地兜底时，文案写死成
   「正常关系 / 友好交流」。

本模块提供三件事（emotion_expert 调用，不做任何 I/O）：

- `DESCRIPTION_ANGLES` + `pick_angle()`：每轮轮换一个具体描写角度；
- `is_cliche()` / `is_repeat()` / `ensure_diverse()`：套话与近重复检测，
  命中后换成按关系远近分档的兜底描述；
- `RELATIONSHIP_POOLS` / `ATTITUDE_POOLS`：分档的多套兜底文案，同一档
  内也有多条可选，不再写死一句。

设计约束：
- 纯函数、无框架依赖，可直接单测；
- 所有产出都清洗到 <= 20 字，且不含 `TextDescriptions` 不接受的特殊
  符号（`/`、`…`、`|` 等会被替换）；
- 兜底池刻意避开线上高频套话（傲娇/撒娇/互怼/损友…），保证兜底本身
  不会继续同质化。
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, List, Tuple

# 与提示词、解析层保持同一上限（原逻辑一直是 20 字）
MAX_DESCRIPTION_LENGTH = 20

# 每个用户保留最近几次描述，用于提示词「别重复」和相似度检测
RECENT_DESCRIPTION_LIMIT = 5

# 每轮换一个角度，逼模型从具体细节切入，而不是从人设形容词切入
DESCRIPTION_ANGLES: Tuple[str, ...] = (
    "从你们最近一起做过的一件具体小事切入",
    "从对方的身份、职业或群里的角色切入",
    "从最近一次意见不合、互相拆台或互相帮忙切入",
    "从谁先找谁、回消息快慢这类小细节切入",
    "从你们共用过的梗、共同爱好或共同熟人切入",
    "从这段关系的距离感切入：是客气、熟络还是没大没小",
    "从{bot_name}观察到的对方一个小习惯或口头禅切入",
    "从最近一次被夸、被关心或被怼的感受切入",
    "从群聊和私聊里不同的相处方式切入",
    "用一句像聊天记录的短句来写，不要写成总结陈词",
    "从这段关系最近的变化切入：更近了，还是有点疏远",
    "从对方给{bot_name}带来的具体影响切入，比如心情、作息或说话方式",
)

# 线上统计里的真正高频词：一个就够可疑，两个基本就是套话堆砌
STRONG_CLICHE_TERMS: Tuple[str, ...] = (
    "傲娇", "撒娇", "俏皮", "互怼", "损友", "欢喜冤家", "嘴硬",
    "宠溺", "互宠", "互撩", "互损", "斗嘴", "互闹", "娇嗔", "卖萌",
)

# 单独出现不算套话，和强词叠加时才计分
WEAK_CLICHE_TERMS: Tuple[str, ...] = (
    "调侃", "日常", "玩闹", "打趣", "冤家", "互夸",
)

# 纯空话 / 默认占位，命中直接判为套话
EMPTY_PHRASES: Tuple[str, ...] = (
    "友好交流", "正常关系", "关系融洽", "相处愉快", "关系良好", "普通关系",
    "陌生人", "中立",
)

# 提示词里的「写法示例」如果被模型原样照抄，会变成下一批套话。
# 命中按纯空话同分处理，逼模型回到本轮对话的细节。
PROMPT_EXAMPLE_ECHOES: Tuple[str, ...] = (
    "一起蹲开播的老观众",
    "刚吵完还没台阶下的熟人",
    "互相投喂零食的饭搭子",
    "嘴上说不管还是把链接发来了",
    "嘴上说不管，还是把链接发来了",
    "没接话，只发了个问号",
    "认真解释了三遍才放心",
)

# 兜底池按「关系远近」分档。分档只用 favor / intimacy，不依赖可自定义
# 的阶段名，避免用户改名后兜底失配。
RELATIONSHIP_POOLS = {
    "negative": (
        "还没台阶下的旧识",
        "正在冷处理的聊天对象",
        "话不投机的群友",
        "需要重新破冰的熟人",
        "刚吵完架的两个人",
        "互相躲着走的网友",
        "只剩客套的旧朋友",
        "关系降到冰点的熟人",
        "说话开始带刺的旧友",
        "还没消气的聊天对象",
        "见面只当没看见的熟人",
        "聊天记录停在吵架那天的网友",
        "谁也不肯先低头的旧搭档",
        "徘徊在拉黑边缘的熟人",
        "只剩节日问候的熟人",
        "一提旧事就冷场的朋友",
    ),
    "distant": (
        "刚加好友的礼貌网友",
        "只在群里打过照面",
        "偶尔冒泡的潜水群友",
        "还没混熟的聊天对象",
        "正在互相试探的网友",
        "点头之交的群友",
        "刚被圈粉的路人观众",
        "说不上熟的私聊对象",
        "只在评论区见过的群友",
        "偶尔互相点赞的网友",
        "刚关注不久的新观众",
        "还没说过几句话的群友",
        "只在同一个群的熟人",
        "偶尔接一句话的网友",
        "刚认识没多久的同好",
        "还没找到共同话题的网友",
        "刚进群的陌生网友",
        "只在活动里见过一面",
        "偶尔被艾特的群友",
        "还没加好友的群友",
        "只在同好群里聊过",
        "刚互关没多久的网友",
        "偶尔在群里搭句话",
        "还记不住名字的群友",
        "刚打过一次招呼的网友",
        "只聊过一两句的网友",
        "还没熟到能开玩笑的群友",
        "偶尔在评论区互动的网友",
        "刚加群不久的新人",
        "只在群活动里碰过面",
    ),
    "casual": (
        "会互相分享琐事的网友",
        "聊得来的固定搭子",
        "一起追直播的同好",
        "互相接梗的群友",
        "偶尔私聊吐槽的熟人",
        "一起蹲更新的同好",
        "能借到作业的学习搭子",
        "会互相递链接的网友",
        "一起打游戏的搭子",
        "会分享歌单的网友",
        "偶尔拼单的饭搭子",
        "互相推荐视频的群友",
        "一起赶末班车的同路人",
        "会讨论剧情的剧搭子",
        "偶尔互相安利的网友",
        "一起摸鱼的群友",
    ),
    "close": (
        "随叫随到的深夜搭子",
        "彼此兜底的老熟人",
        "无话不谈的树洞",
        "一起立过 flag 的共犯",
        "互相投喂的饭搭子",
        "会互相报备行程的朋友",
        "共用一套梗的老网友",
        "失恋先找对方哭的朋友",
        "会互相代签到的同学",
        "一起熬过期末的战友",
        "看过彼此狼狈样的熟人",
        "会帮对方打掩护的搭档",
        "深夜也能打电话的朋友",
        "互相寄存小秘密的搭子",
        "会记得对方忌口的朋友",
        "一言不合就开玩的熟人",
    ),
    "intimate": (
        "不用多说就懂的搭档",
        "把对方写进日程的人",
        "互相留一盏灯的老朋友",
        "共享同一套默契的搭子",
        "一个眼神就懂的共犯",
        "习惯了彼此存在的人",
        "会替对方兜底的老搭档",
        "隔着屏幕也能接住情绪的人",
        "把对方算进未来计划的人",
        "不用寒暄也能接上话的人",
        "知道对方所有黑历史的熟人",
        "会替对方留最后一口的人",
        "分开多久都能接上话的朋友",
        "互相把对方放心上的人",
        "一句在吗就能聊到半夜的人",
        "见过彼此最真实样子的搭档",
    ),
}

ATTITUDE_POOLS = {
    "negative": (
        "简短回绝不想多说",
        "礼貌但明显疏远",
        "把话题岔开",
        "压着火气回应",
        "冷处理不接茬",
        "只回必要的那几个字",
        "语气硬得没留余地",
        "假装没看见",
        "一句话把天聊死",
        "不接话也不解释",
        "回得很慢又很敷衍",
        "带着刺反问一句",
        "直接把话题截断",
        "只回一个句号",
        "明显不耐烦地应一声",
        "把情绪写在字缝里",
    ),
    "distant": (
        "客客气气地接话",
        "保持距离地回两句",
        "只回答不主动延伸",
        "谨慎地找着话题",
        "礼貌回应但不热络",
        "边观察边慢慢搭话",
        "把话说得很客气",
        "回得简短又得体",
        "客气地点头应一声",
        "有一搭没一搭地寒暄",
        "只接话不主动问",
        "生疏地回答两句",
        "不冷不热地问一句",
        "慢慢把话题接住",
        "礼貌地笑一下",
        "保持客气地回应",
        "客气地回一句就结束",
        "简短地应答一声",
        "礼貌地找借口结束",
        "保持距离地寒暄两句",
        "只点头不主动开口",
        "谨慎地回一个表情",
        "客气地反问一句",
        "不咸不淡地应着",
        "慢慢地才接一句话",
        "礼貌地避开私人话题",
        "简短地说明一下就完",
        "生疏地叫了声称呼",
        "只答问题不闲聊",
        "客气地附和一声",
        "有一句没一句地应着",
        "礼貌地等对方先说",
    ),
    "casual": (
        "顺手接梗再抛回去",
        "边聊边找共同话题",
        "认真听完再回应",
        "偶尔主动搭一句话",
        "耐心解释自己的看法",
        "被逗笑后顺着聊下去",
        "想起什么就说什么",
        "半开玩笑地回一句",
        "随口接一句再继续",
        "顺着话头聊下去",
        "好奇地追问一句",
        "边笑边回一句",
        "自然地开个玩笑",
        "分享一件刚发生的小事",
        "顺着对方的话接梗",
        "不紧不慢地聊着",
    ),
    "close": (
        "先损两句再认真哄",
        "嘴上嫌弃手上照做",
        "主动分享今天发生的事",
        "把对方的话记在心上",
        "一边吐槽一边帮忙",
        "不留情面地关心",
        "对方没说完就接上了",
        "半夜也爬起来回消息",
        "先问清楚再给建议",
        "顺手替对方把事情办了",
        "嘴上说不管还是发了链接",
        "记得对方上次提过的小事",
        "忙到一半也会回一句",
        "直接说出不太好听的实话",
        "主动问一句到家没",
        "把对方爱吃的留一份",
    ),
    "intimate": (
        "不用问就知道对方要什么",
        "说话随意但事事有回应",
        "先安抚情绪再讲道理",
        "把对方的事当成自己的事",
        "一边忙一边抽空回一句",
        "隔着屏幕也能听出不对劲",
        "习惯了先问一句吃没吃",
        "不客套，直接说重点",
        "不用铺垫就聊到心里去",
        "对方一开口就知道下文",
        "把好听的话留给对方",
        "再忙也会先回个表情",
        "先顺着对方再慢慢劝",
        "不用解释也能互相明白",
        "说话不客气但从不敷衍",
        "记得对方所有重要的日子",
    ),
}

_POOLS = {
    "relationship": RELATIONSHIP_POOLS,
    "attitude": ATTITUDE_POOLS,
}

_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:关系描述|态度描述|关系|态度)\s*[:：]\s*"
)
_TRIM_CHARS = "\"'`[]【】「」《》 \t\r\n"
# TextDescriptions 的合法字符集不接受这些符号：统一换成中文顿号 / 去除，
# 避免 LLM 的「抽卡/追番」这类写法让 update_* 抛 ValueError。
_SLASH_RE = re.compile(r"[/|｜]+")
# 与 models.TextDescriptions 的字符集保持同一份白名单：清洗后一定可写回。
_UNSAFE_RE = re.compile(
    r"[^\w\-\s\u4e00-\u9fa5\uff0c\u3002\uff01\uff1f\u3001\uff1b\uff1a"
    r"\u201c\u201d\u2018\u2019\u300a\u300b\u00b7]",
    re.UNICODE,
)
_COMPACT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _as_text(value: Any) -> str:
    """把任意输入安全转成去首尾空白的字符串"""
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def sanitize_description(value: Any, limit: int = MAX_DESCRIPTION_LENGTH) -> str:
    """清洗 LLM / 兜底产出的描述

    - 压掉换行与连续空白（提示词偶尔会带换行）；
    - 剥掉「关系：」「态度描述：」前缀和引号/方括号；
    - `/`、`|`、`…` 换成模型状态层能接受的字符；
    - 去掉句末标点，再截到 20 字。

    注意：这里**不**做「是否为空」的判断，调用方需要自己决定空值走哪条
    兜底路径。
    """
    text = " ".join(_as_text(value).split())
    text = text.strip(_TRIM_CHARS)
    text = _LEADING_LABEL_RE.sub("", text)
    text = text.strip(_TRIM_CHARS)
    text = _SLASH_RE.sub("、", text)
    text = text.replace("…", "").replace("~", "").replace("～", "")
    text = _UNSAFE_RE.sub("", text)
    text = text.strip(_TRIM_CHARS)
    text = text.rstrip("。！？!?，,、；;：:")
    if limit is not None and limit > 0 and len(text) > limit:
        text = text[:limit]
    return text.strip()


def cliche_score(text: str) -> int:
    """套话分：强词 2 分、弱词 1 分、纯空话 3 分"""
    value = _as_text(text)
    if not value:
        return 0
    score = 0
    for term in STRONG_CLICHE_TERMS:
        if term in value:
            score += 2
    for term in WEAK_CLICHE_TERMS:
        if term in value:
            score += 1
    for phrase in EMPTY_PHRASES:
        if phrase in value:
            score += 3
    for phrase in PROMPT_EXAMPLE_ECHOES:
        if phrase in value:
            score += 3
    return score


def is_cliche(text: str, threshold: int = 3) -> bool:
    """是否属于套话堆砌

    阈值 3 的含义：两个强词（如「傲娇 + 互怼」）、一个强词加一个弱词
    （如「傲娇 + 日常」）或一句纯空话才算；单个「傲娇」不判——那可能
    确实是本轮对话里的具体表现。
    """
    return cliche_score(text) >= threshold


def _compact(text: str) -> str:
    return _COMPACT_RE.sub("", _as_text(text))


def _bigrams(text: str) -> set:
    compact = _compact(text)
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def description_similarity(a: Any, b: Any) -> float:
    """字符二元的 Jaccard 相似度（0~1），用于识别「换汤不换药」"""
    left, right = _bigrams(a), _bigrams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def is_repeat(candidate: Any, previous: Any = "", recent: Iterable[Any] = (),
              threshold: float = 0.6) -> bool:
    """候选描述是否与上一版/最近几次重复或高度相似

    阈值定在 0.6 而不是 0.8：线上常见的重复不是逐字复制，而是
    「一起熬夜改稿的搭档」→「一起熬夜改稿的伙伴」这种换一个词的改写
    （字符二元 Jaccard = 0.6）。0.8 会漏掉这类情况。
    """
    cand = sanitize_description(candidate)
    if not cand:
        return False
    values: List[str] = [previous]
    values.extend(recent or ())
    for value in values:
        other = sanitize_description(value)
        if not other:
            continue
        if cand == other:
            return True
        if description_similarity(cand, other) >= threshold:
            return True
    return False


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def relationship_band(favor: Any, intimacy: Any) -> str:
    """按好感/亲密把关系分为五档（与阶段名解耦）"""
    f = _number(favor)
    i = _number(intimacy, 0.0)
    if f < 0:
        return "negative"
    score = f * 0.6 + i * 0.4
    if score < 20:
        return "distant"
    if score < 50:
        return "casual"
    if score < 100:
        return "close"
    return "intimate"


def _seed_number(seed: Any) -> int:
    try:
        return abs(int(seed))
    except (TypeError, ValueError):
        return 0


def _avoid_set(avoid: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in avoid or ():
        value = sanitize_description(item)
        if value and value not in result:
            result.append(value)
    return result


def pick_fallback(kind: str, favor: Any = 0, intimacy: Any = 0, seed: Any = 0,
                  avoid: Iterable[Any] = ()) -> str:
    """按关系档位与种子挑一条兜底描述

    - 同一档内按 seed 轮换，避免所有用户/所有轮次看到同一句；
    - `avoid` 里的上一版/最近描述会被跳过；若整档都被跳过（最多 5 条
      最近记录，池子有 8 条，正常不会发生），退回种子位置那条。
    """
    pools = _POOLS.get(kind) or RELATIONSHIP_POOLS
    band = relationship_band(favor, intimacy)
    candidates = pools.get(band) or pools.get("casual") or ()
    if not candidates:
        return ""
    avoided = _avoid_set(avoid)
    start = _seed_number(seed) % len(candidates)
    for offset in range(len(candidates)):
        candidate = candidates[(start + offset) % len(candidates)]
        if candidate in avoided:
            continue
        if any(description_similarity(candidate, value) >= 0.6 for value in avoided):
            continue
        return candidate
    return candidates[start]


def _descriptions_of(state: Any) -> Any:
    """兼容 EnhancedEmotionalState / TextDescriptions / SimpleNamespace"""
    if state is None:
        return None
    return getattr(state, "descriptions", state)


def recent_descriptions(state: Any, kind: str,
                        limit: int = RECENT_DESCRIPTION_LIMIT) -> List[str]:
    """取最近记录（不含当前值），供提示词「别重复」与兜底避开使用"""
    descriptions = _descriptions_of(state)
    values = getattr(descriptions, f"recent_{kind}", None)
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: List[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        value = sanitize_description(item)
        if value and value not in cleaned:
            cleaned.append(value)
    current = getattr(descriptions, kind, "")
    if isinstance(current, str):
        current = sanitize_description(current)
        cleaned = [value for value in cleaned if value != current]
    return cleaned[-limit:]


def previous_description(state: Any, kind: str) -> str:
    descriptions = _descriptions_of(state)
    return sanitize_description(getattr(descriptions, kind, ""))


def stable_description_seed(state_or_key: Any, salt: int = 0) -> int:
    """只按用户生成稳定种子（不含 update_count）

    兜底替换用这个种子：同一个用户在同一关系档位下，即使大模型连续
    返回套话/重复，也稳定落到同一条兜底文案，不会每轮随机换一句；
    不同用户则分散到池子的不同位置，避免跨用户同质化。
    """
    state = state_or_key
    key = getattr(state, "user_key", "") if state is not None else ""
    if not key:
        key = _as_text(state_or_key)
    digest = hashlib.md5(_as_text(key).encode("utf-8")).hexdigest()[:8]
    base = int(digest, 16)
    return (base + 13 * _seed_number(salt)) % 1000003


def description_seed(state_or_key: Any, salt: int = 0) -> int:
    """按用户与已更新次数生成稳定种子

    只依赖 user_key 与 `update_count`，同一轮提示词、解析、兜底得到同一个
    角度/位置；下次成功更新后 `update_count` 增加，角度自然轮换。
    """
    state = state_or_key
    descriptions = _descriptions_of(state)
    count = getattr(descriptions, "update_count", 0)
    try:
        count = max(0, int(count))
    except (TypeError, ValueError):
        count = 0
    return (stable_description_seed(state_or_key, salt) + 7 * count) % 1000003


def pick_angle(state_or_key: Any, salt: int = 0) -> str:
    """按用户与更新次数轮换本轮描写角度"""
    return DESCRIPTION_ANGLES[description_seed(state_or_key, salt) % len(DESCRIPTION_ANGLES)]


def ensure_diverse(kind: str, candidate: Any, state: Any,
                   seed: Any = None) -> str:
    """返回一条「不套话、不重复」的描述；不合格时换兜底池

    判定顺序：
    1. 空值 / 清洗后为空 → 兜底；
    2. 套话分 >= 3 → 兜底；
    3. 与上一版重复/相似度 >= 0.6：
       - 上一版本身具体（非套话）→ 沿用上一版（关系是慢变量，
         模型按提示词保留原说法时不应该被硬改）；
       - 上一版是套话 → 兜底；
    4. 与更早的最近记录重复/相似 → 兜底（防止绕回旧说法）；
    5. 否则保留 LLM 原文（它最了解本轮对话）。
    """
    value = sanitize_description(candidate)
    previous = previous_description(state, kind)
    recent = recent_descriptions(state, kind)
    if value and not is_cliche(value):
        if (previous and not is_cliche(previous)
                and description_similarity(value, previous) >= 0.6):
            return previous
        if not any(description_similarity(value, item) >= 0.6 for item in recent):
            return value

    # 上一版如果已经是本档位的兜底文案，就沿用，不每轮换一句。
    # 关系描述是慢变量；大模型连续返回套话时，稳定的一句比随机轮换更可信。
    favor = getattr(state, "favor", 0) if state is not None else 0
    intimacy = getattr(state, "intimacy", 0) if state is not None else 0
    pools = _POOLS.get(kind) or RELATIONSHIP_POOLS
    band_candidates = pools.get(relationship_band(favor, intimacy)) or ()
    if previous and previous in band_candidates:
        return previous

    if seed is None:
        seed = stable_description_seed(state, 1 if kind == "attitude" else 0)
    return pick_fallback(
        kind,
        favor,
        intimacy,
        seed,
        avoid=[previous] + recent,
    )


def push_recent(values: Iterable[Any], text: Any,
                limit: int = RECENT_DESCRIPTION_LIMIT) -> List[str]:
    """把新描述压入最近记录：去重、保序、截到最近 N 条"""
    value = sanitize_description(text)
    result: List[str] = []
    for item in values or ():
        if not isinstance(item, str):
            continue
        cleaned = sanitize_description(item)
        if cleaned and cleaned != value and cleaned not in result:
            result.append(cleaned)
    if value:
        result.append(value)
    return result[-limit:]
