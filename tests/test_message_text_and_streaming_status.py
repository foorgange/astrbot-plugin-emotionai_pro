# tests/test_message_text_and_streaming_status.py
"""v4.1.13 回归：非文本消息提取 + 流式模式状态面板投递

背景（线上实锤 2026-10-15）：

  A1. 用户发纯图片/卡片等非文本消息时 `event.message_str` 为空，而框架的
     AstrBotMessage 并没有 extract_plain_text / get_plain_text 方法，旧
     `_get_message_text` 落到 `str(message_obj)`，把整个消息对象（含
     raw_message、原始图片 URL、发送者信息）的 dict repr 当作用户消息：
       · /心情 的「最近三次变化」显示「听到『{'timestamp'…』」；
       · 长期记忆里存进几千字符的垃圾；
       · 情感分析提示词被垃圾撑大（线上实测 3283 字符）；
       · INFO 日志刷屏并泄漏原始图片 URL。

  A2. 流式输出开启时（线上 `streaming_response: true`），on_llm_response
     里追加到 `resp.completion_text` 的状态面板会被框架 respond stage
     丢弃（STREAMING_FINISH 明确不发送），/状态显示 功能完全不可见。
     修复：流式路径改走「流式净化器尾部追加」。

反向验证要求：把 A1/A2 的修复换回旧代码时，本文件对应测试必须失败。
"""
import asyncio
import os
import re
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.main import (  # noqa: E402
    EmotionAIProPlugin,
    _STATUS_PANEL_EXTRA_KEY,
)

# ---------------------------------------------------------------------------
# 流式净化器内部会 `from astrbot.core.message.components import Plain`、
# `from astrbot.core.message.message_event_result import MessageChain`，
# 离线桩里没有这两个模块，这里注入最小实现（只供本文件使用）。
# ---------------------------------------------------------------------------
if "astrbot.core.message" not in sys.modules:
    _core_msg = types.ModuleType("astrbot.core.message")
    _comps = types.ModuleType("astrbot.core.message.components")

    class _Plain:
        def __init__(self, text=""):
            self.text = text
            self.type = "Plain"

    _comps.Plain = _Plain
    _mer = types.ModuleType("astrbot.core.message.message_event_result")

    class _MessageChain:
        def __init__(self):
            self.chain = []

        def message(self, text):
            self.chain.append(_Plain(text))
            return self

    _mer.MessageChain = _MessageChain
    sys.modules["astrbot.core.message"] = _core_msg
    sys.modules["astrbot.core.message.components"] = _comps
    sys.modules["astrbot.core.message.message_event_result"] = _mer


# ---------------------------------------------------------------------------
# 假件
# ---------------------------------------------------------------------------
class FakeSegment:
    """消息段替身。type 接受 "Plain"/"Image" 或枚举形态（str() 带点号）。"""

    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text


class FakeMessageObj:
    """消息对象替身。

    __str__ 忠实复刻线上垃圾形态：框架 AstrBotMessage 的 str() 会输出
    整个消息对象的 dict repr（含 timestamp/raw_message/图片 URL）。
    旧 _get_message_text 的 `str(message_obj)` 兜底正是把它当用户消息。
    """

    _GARBAGE_REPR = (
        "{'timestamp': 1792050576, 'group': Group(group_id='100200300'), "
        "'self_id': '10000', 'sender': MessageMember(user_id='20002'), "
        "'message': [Image(file='/AstrBot/data/temp/media_image_xxx.jpg', "
        "'url': 'https://multimedia.nt.qq.com.cn/download?...')], "
        "'message_str': '', 'raw_message': <Event, {...}>}"
    )

    def __init__(self, segments):
        self.message = segments

    def __str__(self):
        return self._GARBAGE_REPR


class FakeEvent:
    """最小事件替身：message_str / message_obj / extras"""

    def __init__(self, message_str="", segments=None):
        self.message_str = message_str
        self.message_obj = FakeMessageObj(segments or [])
        self._extras = {}

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value


class FakeResult:
    def __init__(self, name):
        self.result_content_type = types.SimpleNamespace(name=name)


class FakeState:
    def __init__(self, show_status=True):
        self.force_update_counter = 0
        self.favor = 50
        self.intimacy = 50
        self.show_status = show_status
        self.descriptions = types.SimpleNamespace(attitude="中立", relationship="普通")

    def should_force_update(self, interval):
        return False


class FakeUserManager:
    def __init__(self, state):
        self._state = state
        self.saved = 0

    async def get_user_state(self, user_key):
        return self._state

    async def update_user_state(self, user_key, state):
        self.saved += 1


class FakeUpdateManager:
    def should_update_emotion(self, state, user_message, original_text):
        return (False, "", 0)


def _make_plugin(state, *, streaming_result=None):
    """装配只保留被测逻辑的插件实例（复用后台任务测试的装配模式）"""
    plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
    plugin.config = PluginConfig()
    plugin.need_assessment_pattern = re.compile(r"\[需要情感评估\]")
    plugin._emotion_update_tasks = {}
    plugin.user_manager = FakeUserManager(state)
    plugin.update_manager = FakeUpdateManager()
    plugin._get_user_key = lambda ev: "user1"
    plugin._get_message_text = EmotionAIProPlugin._get_message_text.__get__(plugin)
    plugin._spawn_emotion_update = lambda *a, **k: None
    plugin._update_global_mood = lambda *a, **k: None
    plugin._format_emotional_state = lambda st: "【情感状态面板】"
    plugin._sanitize_ai_text = lambda t: t
    plugin._result_for_streaming = streaming_result
    return plugin


# ---------------------------------------------------------------------------
# A1：非文本消息提取
# ---------------------------------------------------------------------------
class TestMessageTextExtraction(unittest.TestCase):
    """_get_message_text 不再把消息对象 repr 当文本（v4.1.13）"""

    def _plugin(self):
        return EmotionAIProPlugin.__new__(EmotionAIProPlugin)

    def test_plain_text_returned_as_is(self):
        ev = FakeEvent(message_str="  你好喵  ")
        self.assertEqual(self._plugin()._get_message_text(ev), "你好喵")

    def test_image_only_returns_placeholder(self):
        ev = FakeEvent(message_str="", segments=[FakeSegment("Image")])
        self.assertEqual(self._plugin()._get_message_text(ev), "[图片]")

    def test_image_only_never_returns_object_repr(self):
        """核心回归：结果里不得出现消息对象 dict repr 的任何特征"""
        ev = FakeEvent(message_str="", segments=[FakeSegment("Image")])
        text = self._plugin()._get_message_text(ev)
        self.assertNotIn("{", text)
        self.assertNotIn("timestamp", text)
        self.assertNotIn("raw_message", text)
        self.assertNotIn("http", text)

    def test_mixed_text_and_image(self):
        ev = FakeEvent(
            message_str="",
            segments=[FakeSegment("Plain", "看这个"), FakeSegment("Image")],
        )
        self.assertEqual(self._plugin()._get_message_text(ev), "看这个 [图片]")

    def test_enum_like_segment_type(self):
        """框架真实段类型是枚举，str() 形如 ComponentType.Image"""
        from enum import Enum

        class _Comp(Enum):
            Image = "Image"

        ev = FakeEvent(message_str="", segments=[FakeSegment(_Comp.Image)])
        self.assertEqual(self._plugin()._get_message_text(ev), "[图片]")

    def test_multiple_images_each_marked(self):
        ev = FakeEvent(message_str="", segments=[FakeSegment("Image"), FakeSegment("Image")])
        self.assertEqual(self._plugin()._get_message_text(ev), "[图片] [图片]")

    def test_common_placeholders(self):
        cases = {
            "Face": "[表情]",
            "At": "[At]",
            "AtAll": "[At全体成员]",
            "Reply": "[回复]",
            "Forward": "[转发消息]",
            "Record": "[语音]",
            "Video": "[视频]",
            "File": "[文件]",
            "Json": "[卡片]",
            "Poke": "[戳一戳]",
        }
        for seg_type, expected in cases.items():
            ev = FakeEvent(message_str="", segments=[FakeSegment(seg_type)])
            self.assertEqual(
                self._plugin()._get_message_text(ev), expected,
                f"{seg_type} 应映射为 {expected}",
            )

    def test_unknown_segment_uses_type_name(self):
        ev = FakeEvent(message_str="", segments=[FakeSegment("Mystery")])
        self.assertEqual(self._plugin()._get_message_text(ev), "[mystery]")

    def test_empty_everything_returns_empty(self):
        ev = FakeEvent(message_str="", segments=[])
        self.assertEqual(self._plugin()._get_message_text(ev), "")

    def test_no_message_obj_returns_empty(self):
        ev = FakeEvent(message_str="", segments=[])
        ev.message_obj = None
        self.assertEqual(self._plugin()._get_message_text(ev), "")

    def test_broken_segment_does_not_raise(self):
        """段对象取不到类型时给出通用占位，绝不炸断消息链路、绝不回退 repr"""
        ev = FakeEvent(message_str="", segments=[object()])
        text = self._plugin()._get_message_text(ev)
        self.assertEqual(text, "[未知段]")
        self.assertNotIn("{", text)

    def test_get_message_str_fallback_still_works(self):
        """message_str 为空且无消息段时，仍兼容 get_message_str 取数"""
        plugin = self._plugin()

        class Ev(FakeEvent):
            def get_message_str(self):
                return "兜底文本"

        ev = Ev(message_str="", segments=[])
        self.assertEqual(plugin._get_message_text(ev), "兜底文本")


class TestMoodInputIsClean(unittest.TestCase):
    """修复后心情链路拿到的必须是干净文本（/心情 展示不再出现 dict 垃圾）"""

    def test_mood_signal_empty_for_image_placeholder(self):
        from emotionai_pro.global_mood import compute_mood_signal, mood_trigger_excerpt

        text = "[图片]"
        self.assertEqual(compute_mood_signal(text), {})
        excerpt = mood_trigger_excerpt(text)
        self.assertNotIn("{", excerpt)
        self.assertNotIn("timestamp", excerpt)

    def test_trigger_excerpt_bounded_for_long_garbage_like_input(self):
        from emotionai_pro.global_mood import mood_trigger_excerpt

        excerpt = mood_trigger_excerpt("{'timestamp': 1792050576, 'group': " + "x" * 500)
        self.assertLessEqual(len(excerpt), 13)  # 12 字符 + 省略号


class TestLogExcerpt(unittest.TestCase):
    """日志摘要：截断 + 压缩空白（v4.1.13）"""

    def test_short_text_unchanged(self):
        self.assertEqual(EmotionAIProPlugin._log_excerpt("你好"), "你好")

    def test_long_text_truncated(self):
        out = EmotionAIProPlugin._log_excerpt("啊" * 100)
        self.assertEqual(out, "啊" * 50 + "…")

    def test_whitespace_squashed(self):
        self.assertEqual(EmotionAIProPlugin._log_excerpt("a\n\nb   c"), "a b c")

    def test_none_safe(self):
        self.assertEqual(EmotionAIProPlugin._log_excerpt(None), "")


# ---------------------------------------------------------------------------
# A2：流式模式状态面板投递
# ---------------------------------------------------------------------------
class TestStatusPanelRouting(unittest.TestCase):
    """流式路径状态面板进 extras；非流式路径保持追加 completion_text"""

    def _run(self, *, streaming, show_status=True, marker=True):
        state = FakeState(show_status=show_status)
        result = FakeResult("STREAMING_RESULT") if streaming else None
        plugin = _make_plugin(state)

        class Ev(FakeEvent):
            def __init__(self):
                super().__init__(message_str="你好喵")
                self.unified_msg_origin = "g1"

            def get_result(self):
                return result

        ev = Ev()
        resp = types.SimpleNamespace(
            completion_text="回复内容" + ("[需要情感评估]" if marker else "")
        )
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            plugin.process_smart_update(ev, resp)
        )
        return plugin, ev, resp

    def test_streaming_puts_panel_in_extra(self):
        _, ev, resp = self._run(streaming=True)
        self.assertEqual(ev.get_extra(_STATUS_PANEL_EXTRA_KEY), "【情感状态面板】")
        self.assertNotIn("情感状态面板", resp.completion_text)

    def test_non_streaming_appends_to_completion_text(self):
        _, ev, resp = self._run(streaming=False)
        self.assertIn("\n\n【情感状态面板】", resp.completion_text)
        self.assertIsNone(ev.get_extra(_STATUS_PANEL_EXTRA_KEY))

    def test_no_panel_when_show_status_off(self):
        _, ev, resp = self._run(streaming=True, show_status=False)
        self.assertIsNone(ev.get_extra(_STATUS_PANEL_EXTRA_KEY))

    def test_no_panel_when_not_needing_update(self):
        """未触发更新时不展示面板（marker=False 且智能判断不触发）"""
        _, ev, resp = self._run(streaming=True, marker=False)
        self.assertIsNone(ev.get_extra(_STATUS_PANEL_EXTRA_KEY))

    def test_marker_stripped_in_both_modes(self):
        _, _, resp = self._run(streaming=True)
        self.assertNotIn("[需要情感评估]", resp.completion_text)
        _, _, resp2 = self._run(streaming=False)
        self.assertNotIn("[需要情感评估]", resp2.completion_text)


class TestStreamingFilterAppendsPanel(unittest.TestCase):
    """流式净化器在流的尾部追加状态面板（一次性）"""

    def _install(self, plugin, event, panel="【情感状态面板】"):
        received = []

        async def fake_original(generator, use_fallback=False):
            async for chain in generator:
                received.append(chain)
            return None

        event.send_streaming = fake_original
        if panel is not None:
            event.set_extra(_STATUS_PANEL_EXTRA_KEY, panel)
        plugin._install_streaming_filter(event)
        return received

    @staticmethod
    def _chain(text):
        return types.SimpleNamespace(type=None, chain=[sys.modules["astrbot.core.message.components"].Plain(text)])

    def test_panel_appended_after_marker_stripped(self):
        plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
        event = FakeEvent()
        received = self._install(plugin, event)

        async def gen():
            yield self._chain("你好[需要情感评估]")

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            event.send_streaming(gen(), False)
        )
        texts = [c.chain[0].text for c in received if c and c.chain]
        # 标记被剥掉 → "你好" 先出；面板在流尾部单独成块（无前导文本时
        # 不加 "\n\n" 前缀，避免消息以空行开头）
        self.assertEqual(texts, ["你好", "【情感状态面板】"])
        # 一次性：flush 后暂存键必须被清掉
        self.assertIsNone(event.get_extra(_STATUS_PANEL_EXTRA_KEY))

    def test_panel_joined_with_tail_when_tail_exists(self):
        """流尾部还有暂扣残余时，面板与残余用空行连接后一起发出"""
        plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
        event = FakeEvent()
        received = self._install(plugin, event)

        async def gen():
            # "[需要" 是标记前缀，会被暂扣到 flush
            yield self._chain("你好[需要")

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            event.send_streaming(gen(), False)
        )
        texts = [c.chain[0].text for c in received if c and c.chain]
        self.assertEqual(texts, ["你好", "[需要\n\n【情感状态面板】"])

    def test_panel_alone_when_no_text(self):
        plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
        event = FakeEvent()
        received = self._install(plugin, event)

        async def gen():
            yield self._chain("[需要情感评估]")

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            event.send_streaming(gen(), False)
        )
        texts = [c.chain[0].text for c in received if c and c.chain]
        self.assertEqual(texts, ["【情感状态面板】"])

    def test_no_panel_extra_means_plain_tail(self):
        """没有暂存面板时行为与旧版完全一致（只放flush残余）"""
        plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
        event = FakeEvent()
        received = self._install(plugin, event, panel=None)

        async def gen():
            yield self._chain("你好[需要")

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            event.send_streaming(gen(), False)
        )
        texts = [c.chain[0].text for c in received if c and c.chain]
        self.assertEqual(texts, ["你好", "[需要"])


if __name__ == "__main__":
    unittest.main()
