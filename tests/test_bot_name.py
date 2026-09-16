# tests/test_bot_name.py
"""bot_name 配置项 + _sanitize_ai_text 离线单测

main.py 顶层 import astrbot，故先注入 stub。
"""
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401  (注入 astrbot stub)
import tests.bootstrap  # noqa: F401

from emotionai_pro.main import EmotionAIProPlugin  # noqa: E402
from emotionai_pro.config import PluginConfig  # noqa: E402


def make_plugin(bot_name=None):
    """构造最小可用插件实例（绕过 __init__，直接手工装配）"""
    plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
    plugin.config = PluginConfig(bot_name=bot_name)
    plugin._resolved_bot_name = None
    plugin._bot_name_resolved = False
    plugin._raw_config = AsyncMock()
    plugin.context = MagicMock()
    return plugin


class TestExtractNameFromPrompt(unittest.TestCase):
    def setUp(self):
        self.plugin = make_plugin()

    def test_extract_tail_wind(self):
        """「# 永雏塔菲 — QQ群机器人人设提示词」→ 永雏塔菲"""
        name = self.plugin._extract_name_from_prompt("# 永雏塔菲 — QQ群机器人人设提示词")
        self.assertEqual(name, "永雏塔菲")

    def test_extract_simple(self):
        """「# 塔菲」→ 塔菲"""
        name = self.plugin._extract_name_from_prompt("# 塔菲")
        self.assertEqual(name, "塔菲")

    def test_extract_english(self):
        """「# Taffy」→ Taffy"""
        name = self.plugin._extract_name_from_prompt("# Taffy")
        self.assertEqual(name, "Taffy")

    def test_extract_no_hash(self):
        """不以 # 开头 → None"""
        name = self.plugin._extract_name_from_prompt("永雏塔菲")
        self.assertIsNone(name)

    def test_extract_empty(self):
        """空提示词 → None"""
        self.assertIsNone(self.plugin._extract_name_from_prompt(""))
        self.assertIsNone(self.plugin._extract_name_from_prompt(None))


class TestGetBotName(unittest.TestCase):
    def test_config_priority(self):
        """配置项优先于运行时解析值"""
        plugin = make_plugin(bot_name="塔菲")
        plugin._resolved_bot_name = "旧值"
        self.assertEqual(plugin._get_bot_name(), "塔菲")

    def test_resolved_fallback(self):
        """配置为空时用运行时解析值"""
        plugin = make_plugin()
        plugin._resolved_bot_name = "永雏塔菲"
        self.assertEqual(plugin._get_bot_name(), "永雏塔菲")

    def test_none_when_unresolved(self):
        """配置与解析值均为空 → None（不替换）"""
        plugin = make_plugin()
        self.assertIsNone(plugin._get_bot_name())


class TestSanitizeAiText(unittest.TestCase):
    def setUp(self):
        self.plugin = make_plugin(bot_name="塔菲")

    def test_standalone_ai(self):
        """独立 AI → 塔菲"""
        self.assertEqual(self.plugin._sanitize_ai_text("AI 说你好"), "塔菲 说你好")

    def test_chinese_embedded_ai(self):
        """中文夹住的 ai → 塔菲（核心场景）"""
        self.assertEqual(self.plugin._sanitize_ai_text("亲密玩闹的ai伙伴"), "亲密玩闹的塔菲伙伴")

    def test_upper_lower(self):
        """大小写均替换"""
        self.assertEqual(self.plugin._sanitize_ai_text("ai 你好"), "塔菲 你好")

    def test_chinese_not_false_positive(self):
        """纯中文无 ASCII "AI" 字形 → 不误伤（“人工智能”由汉字组成，不是字母 AI）"""
        self.assertEqual(self.plugin._sanitize_ai_text("人工智能"), "人工智能")

    def test_alphanumeric_boundary_keeps_FAI(self):
        """FAI级别 中 AI 前后有字母 → 不替换"""
        self.assertEqual(self.plugin._sanitize_ai_text("FAI级别"), "FAI级别")

    def test_empty_text(self):
        """空文本原样返回"""
        self.assertEqual(self.plugin._sanitize_ai_text(""), "")
        self.assertEqual(self.plugin._sanitize_ai_text(None), None)

    def test_no_bot_name_returns_original(self):
        """bot_name 为空时返回原文，不引入新 AI"""
        plugin = make_plugin()
        self.assertEqual(plugin._sanitize_ai_text("AI 说你好"), "AI 说你好")


class _AsyncPersonaResolver:
    """按 AstrBot **真实签名**实现的 persona 解析桩。

    真实实现（`astrbot/core/persona_mgr.py::PersonaManager.resolve_selected_persona`）：
        async def resolve_selected_persona(self, *, umo, conversation_persona_id,
                                           platform_name, provider_settings=None)
            -> tuple[str | None, Personality | None, str | None, bool]

    为什么不能再用同步 MagicMock：
        v4.0.11 引入 bot_name 自动提取时，本文件用的是
        `MagicMock(return_value=(...))`（同步桩），于是插件里**漏掉的 await**
        在测试里完全看不出来（同步桩返回的就是元组，解包当然成功）。
        结果该 bug 静默存活到 v4.0.14：线上每次都是
        `TypeError: cannot unpack non-iterable coroutine object`
        → 被 except 吞成 WARN → bot_name 永远为空
        → `_sanitize_ai_text()` 退化为空操作（人设一致性修复形同未生效）。

    所以桩必须和真身一样是 async，且参数必须是 keyword-only，
    这样「忘了 await」和「调用约定写错」都会被测出来。
    """

    def __init__(self, persona_name="永雏塔菲", prompt="# 永雏塔菲 — 人设",
                 default_persona=False, raises=None):
        if default_persona:
            self._persona = {"name": "default", "prompt": prompt}
        else:
            self._persona = {"name": persona_name, "prompt": prompt}
        self._raises = raises
        self.calls = []

    async def resolve_selected_persona(self, *, umo, conversation_persona_id,
                                       platform_name, provider_settings=None):
        self.calls.append({
            "umo": umo,
            "conversation_persona_id": conversation_persona_id,
            "platform_name": platform_name,
            "provider_settings": provider_settings,
        })
        if self._raises is not None:
            raise self._raises
        return ("永雏塔菲", self._persona, None, False)


class TestEnsureBotName(unittest.TestCase):
    async def _run_ensure(self, plugin, persona_name="永雏塔菲", prompt="# 永雏塔菲 — 人设", default_persona=False):
        from astrbot.api.event import AstrMessageEvent
        from astrbot.api.provider import ProviderRequest

        event = AstrMessageEvent()
        req = ProviderRequest()

        plugin.context.get_config = MagicMock(return_value={"provider_settings": {}})
        plugin.context.persona_manager = MagicMock()
        plugin.context.persona_manager.resolve_selected_persona = _AsyncPersonaResolver(
            persona_name=persona_name, prompt=prompt, default_persona=default_persona
        ).resolve_selected_persona

        await plugin._ensure_bot_name(event, req)
        return plugin

    def test_extracts_from_persona_name(self):
        """persona name 非 default → 直接用 persona 名"""
        import asyncio
        plugin = make_plugin()
        result = asyncio.run(self._run_ensure(plugin, persona_name="永雏塔菲"))
        self.assertEqual(result._resolved_bot_name, "永雏塔菲")
        self.assertEqual(result.config.bot_name, "永雏塔菲")
        # 落盘被调用
        result._raw_config.save_config_async.assert_awaited()

    def test_default_persona_extracts_from_prompt(self):
        """default persona → 从提示词第一行提取"""
        import asyncio
        plugin = make_plugin()
        result = asyncio.run(self._run_ensure(plugin, default_persona=True, prompt="# 塔菲 — 人设提示词"))
        self.assertEqual(result._resolved_bot_name, "塔菲")

    def test_async_resolver_is_awaited(self):
        """回归锁定：async 的 persona 解析必须被 await（v4.0.15 修的静默失效）

        这是上面那个 bug 的**直接**锁定项。用真身同款签名（async + keyword-only）
        的桩：如果插件漏掉 await，解包到的就是 coroutine 对象 →
        TypeError 被 except 吞掉 → `_resolved_bot_name` 保持 None → 本测试失败。
        """
        import asyncio
        from astrbot.api.event import AstrMessageEvent
        from astrbot.api.provider import ProviderRequest

        plugin = make_plugin()
        resolver = _AsyncPersonaResolver(persona_name="永雏塔菲")
        plugin.context.get_config = MagicMock(return_value={"provider_settings": {}})
        plugin.context.persona_manager = MagicMock()
        plugin.context.persona_manager.resolve_selected_persona = resolver.resolve_selected_persona

        async def go():
            await plugin._ensure_bot_name(AstrMessageEvent(), ProviderRequest())

        asyncio.run(go())

        self.assertEqual(len(resolver.calls), 1, "解析函数应被调用一次")
        self.assertEqual(plugin._resolved_bot_name, "永雏塔菲",
                         "async 解析函数未被 await（解包 coroutine 会抛 TypeError 被吞掉）")

    def test_real_signature_is_keyword_only_async(self):
        """守住「桩必须与真身同构」这条前提

        如果将来有人把 _AsyncPersonaResolver 改回同步、或改成位置参数，
        本测试会失败，提醒他：桩一旦与真身不同构，这类 bug 就又会溜过去。
        """
        import inspect
        self.assertTrue(inspect.iscoroutinefunction(_AsyncPersonaResolver.resolve_selected_persona))
        params = inspect.signature(_AsyncPersonaResolver.resolve_selected_persona).parameters
        for name in ("umo", "conversation_persona_id", "platform_name", "provider_settings"):
            self.assertIn(name, params)
            self.assertEqual(params[name].kind, inspect.Parameter.KEYWORD_ONLY,
                             f"{name} 必须是 keyword-only（与真身一致）")

    def test_idempotent(self):
        """只尝试一次：解析失败后不再重试"""
        import asyncio
        from astrbot.api.event import AstrMessageEvent
        from astrbot.api.provider import ProviderRequest

        plugin = make_plugin()
        resolver = _AsyncPersonaResolver(raises=Exception("boom"))
        plugin.context.persona_manager = MagicMock()
        plugin.context.persona_manager.resolve_selected_persona = resolver.resolve_selected_persona

        async def go():
            event = AstrMessageEvent()
            req = ProviderRequest()
            await plugin._ensure_bot_name(event, req)  # 第一次失败
            plugin._bot_name_resolved = True
            await plugin._ensure_bot_name(event, req)  # 第二次不应再调用

        asyncio.run(go())
        self.assertEqual(len(resolver.calls), 1)
        self.assertIsNone(plugin._resolved_bot_name)

    def test_skips_when_config_set(self):
        """config.bot_name 已设置 → 不执行解析"""
        import asyncio
        from astrbot.api.event import AstrMessageEvent
        from astrbot.api.provider import ProviderRequest

        plugin = make_plugin(bot_name="手动设置")
        resolver = _AsyncPersonaResolver()
        plugin.context.persona_manager = MagicMock()
        plugin.context.persona_manager.resolve_selected_persona = resolver.resolve_selected_persona
        plugin.context.get_config = MagicMock(return_value={})

        async def go():
            event = AstrMessageEvent()
            req = ProviderRequest()
            await plugin._ensure_bot_name(event, req)

        asyncio.run(go())
        self.assertEqual(len(resolver.calls), 0)


if __name__ == "__main__":
    unittest.main()
