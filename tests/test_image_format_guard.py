# tests/test_image_format_guard.py
"""图片格式守卫（v4.1.7）离线单测

覆盖三层：
  1. mime 解析与分类（data URI / URL 后缀 / 本地路径 / 畸形输入）
  2. 过滤函数（contexts 历史图 + 本回合 image_urls，不可变保证）
  3. 插件接入行为（main._filter_unsupported_images：开关、异常不炸、
     与上下文保鲜的协同——快照永远是过滤前的原始上下文）
"""
import sys
import os
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401  (注册 astrbot stub)
import tests.bootstrap  # noqa: F401  (注册 emotionai_pro 包别名)

from emotionai_pro.image_format_guard import (
    SUPPORTED_IMAGE_MIME_TYPES,
    UNSUPPORTED_IMAGE_PLACEHOLDER,
    extract_data_uri_mime,
    guess_mime_from_ref,
    classify_image_url,
    filter_unsupported_image_parts,
    filter_image_urls,
)
from emotionai_pro.config import PluginConfig
from emotionai_pro.main import EmotionAIProPlugin


def _img_part(url):
    """构造框架真实形态的 image_url part。"""
    return {"type": "image_url", "image_url": {"url": url, "id": None}}


class TestExtractDataUriMime(unittest.TestCase):
    def test_plain_png(self):
        self.assertEqual(
            extract_data_uri_mime("data:image/png;base64,iVBORw0KGgo="),
            "image/png",
        )

    def test_with_extra_params(self):
        self.assertEqual(
            extract_data_uri_mime("data:image/svg+xml;charset=utf-8;base64,PHN2Zz4="),
            "image/svg+xml",
        )

    def test_uppercase_normalized(self):
        self.assertEqual(
            extract_data_uri_mime("data:IMAGE/PNG;base64,iVBORw0KGgo="),
            "image/png",
        )

    def test_non_data_uri(self):
        self.assertIsNone(extract_data_uri_mime("https://example.com/a.png"))

    def test_malformed_no_mime(self):
        self.assertIsNone(extract_data_uri_mime("data:;base64,iVBORw0KGgo="))
        self.assertIsNone(extract_data_uri_mime("data:base64,iVBORw0KGgo="))
        self.assertIsNone(extract_data_uri_mime("data:,iVBORw0KGgo="))

    def test_non_string(self):
        self.assertIsNone(extract_data_uri_mime(None))
        self.assertIsNone(extract_data_uri_mime(123))


class TestGuessMimeFromRef(unittest.TestCase):
    def test_local_path_suffix(self):
        self.assertEqual(guess_mime_from_ref("/tmp/a/b.svg"), "image/svg+xml")
        self.assertEqual(guess_mime_from_ref("C:\\pic\\x.PNG"), "image/png")

    def test_http_url_with_query(self):
        self.assertEqual(
            guess_mime_from_ref("https://gxh.vip.qq.com/x/raw123.png?token=abc"),
            "image/png",
        )

    def test_file_uri(self):
        self.assertEqual(guess_mime_from_ref("file:///tmp/a.gif"), "image/gif")

    def test_no_suffix(self):
        self.assertIsNone(guess_mime_from_ref("/tmp/download_9f3a2b"))

    def test_unknown_suffix(self):
        self.assertIsNone(guess_mime_from_ref("/tmp/a.xyz"))

    def test_non_string(self):
        self.assertIsNone(guess_mime_from_ref(None))
        self.assertIsNone(guess_mime_from_ref(""))


class TestClassifyImageUrl(unittest.TestCase):
    def test_supported_data_uris(self):
        for url in (
            "data:image/png;base64,iVBORw0KGgo=",
            "data:image/jpeg;base64,/9j/4AAQ",
            "data:image/jpg;base64,/9j/4AAQ",
            "data:image/webp;base64,UklGRg==",
            "data:image/gif;base64,R0lGOD==",
            "data:application/octet-stream;base64,AAAA",
        ):
            self.assertEqual(
                classify_image_url(url), "supported", f"应放行: {url[:40]}"
            )

    def test_unsupported_data_uris(self):
        for url in (
            "data:image/svg+xml;base64,PHN2Zz4=",
            "data:image/bmp;base64,Qk0=",
            "data:image/tiff;base64,SUk=",
            "data:image/heic;base64,AAAA",
            "data:;base64,iVBORw0KGgo=",
        ):
            self.assertEqual(
                classify_image_url(url), "unsupported", f"应拦截: {url[:40]}"
            )

    def test_path_and_url_classification(self):
        self.assertEqual(classify_image_url("/tmp/x.svg"), "unsupported")
        self.assertEqual(
            classify_image_url("https://q.q.com/raw1.svg"), "unsupported"
        )
        self.assertEqual(classify_image_url("/tmp/x.png"), "supported")
        # 判不出后缀的一律 unknown（放行）
        self.assertEqual(classify_image_url("/tmp/download_9f3a"), "unknown")
        self.assertEqual(
            classify_image_url("https://q.q.com/media?id=7"), "unknown"
        )

    def test_bad_inputs_unknown(self):
        for bad in (None, "", 123, [], {}):
            self.assertEqual(classify_image_url(bad), "unknown")

    def test_custom_whitelist(self):
        # 调试/测试可通过参数覆盖白名单：把 png 也判为不支持
        self.assertEqual(
            classify_image_url("data:image/png;base64,iVBORw0KGgo=",
                               supported_mimes=frozenset({"image/gif"})),
            "unsupported",
        )

    def test_default_whitelist_matches_step(self):
        # 与 step-5-preview 报错信息中列出的支持格式对齐
        self.assertIn("image/png", SUPPORTED_IMAGE_MIME_TYPES)
        self.assertIn("image/jpeg", SUPPORTED_IMAGE_MIME_TYPES)
        self.assertIn("image/webp", SUPPORTED_IMAGE_MIME_TYPES)
        self.assertIn("image/gif", SUPPORTED_IMAGE_MIME_TYPES)
        self.assertNotIn("image/svg+xml", SUPPORTED_IMAGE_MIME_TYPES)


class TestFilterUnsupportedImageParts(unittest.TestCase):
    def test_svg_replaced_with_placeholder(self):
        contexts = [
            {"role": "user", "content": [
                {"type": "text", "text": "看这张图"},
                _img_part("data:image/svg+xml;base64,PHN2Zz4="),
            ]},
        ]
        new, replaced = filter_unsupported_image_parts(contexts)
        self.assertEqual(replaced, 1)
        content = new[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "看这张图"})
        self.assertEqual(
            content[1],
            {"type": "text", "text": UNSUPPORTED_IMAGE_PLACEHOLDER},
        )

    def test_supported_images_untouched(self):
        contexts = [
            {"role": "user", "content": [
                {"type": "text", "text": "看这张图"},
                _img_part("data:image/png;base64,iVBORw0KGgo="),
            ]},
        ]
        new, replaced = filter_unsupported_image_parts(contexts)
        self.assertEqual(replaced, 0)
        self.assertIs(new[0], contexts[0])

    def test_input_not_mutated(self):
        """不可变保证：传入的 dict/list 一个字节都不改。"""
        contexts = [
            {"role": "user", "content": [
                {"type": "text", "text": "看这张图"},
                _img_part("data:image/svg+xml;base64,PHN2Zz4="),
            ]},
        ]
        original = [
            {"role": "user", "content": [
                {"type": "text", "text": "看这张图"},
                _img_part("data:image/svg+xml;base64,PHN2Zz4="),
            ]},
        ]
        new, replaced = filter_unsupported_image_parts(contexts)
        self.assertEqual(replaced, 1)
        self.assertEqual(contexts, original)
        self.assertIsNot(new[0], contexts[0])
        self.assertIsNot(new[0]["content"], contexts[0]["content"])
        # 原 part 对象也没被碰
        self.assertEqual(contexts[0]["content"][1]["type"], "image_url")

    def test_only_poisoned_messages_copied(self):
        clean = {"role": "assistant", "content": "纯文本"}
        poisoned = {"role": "user", "content": [
            _img_part("/tmp/x.svg"),
        ]}
        new, replaced = filter_unsupported_image_parts([clean, poisoned])
        self.assertEqual(replaced, 1)
        self.assertIs(new[0], clean)
        self.assertIsNot(new[1], poisoned)

    def test_string_content_and_bad_messages_skipped(self):
        contexts = [
            {"role": "user", "content": "纯文本消息"},
            {"role": "assistant"},                      # 无 content
            {"role": "user", "content": None},          # content=None
            "不是dict",                                  # 坏消息
            {"role": "user", "content": [               # 坏 part 结构
                {"type": "image_url", "image_url": 123},
                {"type": "image_url"},
                {"type": "text", "text": "正常"},
            ]},
        ]
        new, replaced = filter_unsupported_image_parts(contexts)
        self.assertEqual(replaced, 0)
        self.assertEqual(new, contexts)

    def test_str_form_image_url_compat(self):
        """兼容简化/手写形态 {"image_url": "http://..."}。"""
        contexts = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": "https://q.q.com/x.svg"},
            ]},
        ]
        new, replaced = filter_unsupported_image_parts(contexts)
        self.assertEqual(replaced, 1)
        self.assertEqual(new[0]["content"][0]["text"],
                         UNSUPPORTED_IMAGE_PLACEHOLDER)

    def test_custom_placeholder(self):
        contexts = [{"role": "user", "content": [_img_part("data:image/bmp;base64,Qk0=")]}]
        new, replaced = filter_unsupported_image_parts(
            contexts, placeholder_text="[一张无法显示的图片]"
        )
        self.assertEqual(replaced, 1)
        self.assertEqual(new[0]["content"][0]["text"], "[一张无法显示的图片]")

    def test_empty_contexts(self):
        new, replaced = filter_unsupported_image_parts([])
        self.assertEqual(new, [])
        self.assertEqual(replaced, 0)
        new, replaced = filter_unsupported_image_parts(None)
        self.assertIsNone(new)
        self.assertEqual(replaced, 0)


class TestFilterImageUrls(unittest.TestCase):
    def test_mixed_list(self):
        urls = [
            "https://gxh.vip.qq.com/x/raw1.svg",
            "https://gxh.vip.qq.com/x/raw2.png",
            "/tmp/local_no_suffix",
            "data:image/gif;base64,R0lGOD==",
        ]
        new = filter_image_urls(urls)
        self.assertEqual(new, [
            "https://gxh.vip.qq.com/x/raw2.png",
            "/tmp/local_no_suffix",
            "data:image/gif;base64,R0lGOD==",
        ])
        # 原列表不被修改
        self.assertEqual(urls[0], "https://gxh.vip.qq.com/x/raw1.svg")

    def test_empty(self):
        self.assertEqual(filter_image_urls([]), [])
        self.assertEqual(filter_image_urls(None), [])


def _make_req(contexts=None, image_urls=None):
    return types.SimpleNamespace(
        contexts=contexts if contexts is not None else [],
        image_urls=image_urls if image_urls is not None else [],
    )


def _run_filter(plugin, req):
    """把 main._filter_unsupported_images 绑到简易 plugin 对象上执行。"""
    method = EmotionAIProPlugin._filter_unsupported_images.__get__(plugin)
    method(None, req)
    return req


class TestPluginFilterIntegration(unittest.TestCase):
    def _plugin(self, **cfg):
        return types.SimpleNamespace(config=PluginConfig(**cfg))

    def test_default_enabled_filters_contexts(self):
        plugin = self._plugin()
        req = _make_req(contexts=[
            {"role": "user", "content": [
                {"type": "text", "text": "看"},
                _img_part("data:image/svg+xml;base64,PHN2Zz4="),
            ]},
        ])
        _run_filter(plugin, req)
        self.assertEqual(
            req.contexts[0]["content"][1]["text"],
            UNSUPPORTED_IMAGE_PLACEHOLDER,
        )

    def test_default_enabled_filters_image_urls(self):
        plugin = self._plugin()
        req = _make_req(image_urls=["/tmp/x.svg", "/tmp/y.png"])
        _run_filter(plugin, req)
        self.assertEqual(req.image_urls, ["/tmp/y.png"])

    def test_switch_off_is_noop(self):
        """开关关闭分支行为测试：不过滤、不抛异常。"""
        plugin = self._plugin(image_format_filter_enabled=False)
        req = _make_req(
            contexts=[{"role": "user", "content": [
                _img_part("data:image/svg+xml;base64,PHN2Zz4="),
            ]}],
            image_urls=["/tmp/x.svg"],
        )
        _run_filter(plugin, req)
        self.assertEqual(req.contexts[0]["content"][0]["type"], "image_url")
        self.assertEqual(req.image_urls, ["/tmp/x.svg"])

    def test_exception_never_breaks_request(self):
        """过滤内部异常只能 warning，绝不能扩散阻断 LLM 请求。"""
        plugin = self._plugin()
        req = _make_req(image_urls=["/tmp/x.svg"])
        with mock.patch(
            "emotionai_pro.main.filter_image_urls",
            side_effect=RuntimeError("boom"),
        ):
            _run_filter(plugin, req)
        # 异常被吞，请求原样放行
        self.assertEqual(req.image_urls, ["/tmp/x.svg"])

    def test_freshness_snapshot_survives_filter(self):
        """与上下文保鲜协同：过滤后快照键里的仍是过滤前的原始上下文。

        模拟顺序：保鲜先跑（快照原始 contexts 并清空）→ 过滤跑
        （此时 contexts 为空，无副作用）→ 断言 extras 中的快照
        未被过滤污染。
        """
        plugin = self._plugin()
        original = [{"role": "user", "content": [
            {"type": "text", "text": "旧话题"},
            _img_part("data:image/svg+xml;base64,PHN2Zz4="),
        ]}]
        req = _make_req(contexts=list(original))
        # 模拟保鲜：快照 + 清空
        snapshot = list(req.contexts)
        req.contexts = []
        _run_filter(plugin, req)
        self.assertEqual(snapshot[0]["content"][1]["type"], "image_url")
        self.assertEqual(req.contexts, [])


if __name__ == "__main__":
    unittest.main()
