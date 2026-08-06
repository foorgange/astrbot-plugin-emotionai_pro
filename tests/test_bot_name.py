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


class TestEnsureBotName(unittest.TestCase):
    async def _run_ensure(self, plugin, persona_name="永雏塔菲", prompt="# 永雏塔菲 — 人设", default_persona=False):
        from astrbot.api.event import AstrMessageEvent
        from astrbot.api.provider import ProviderRequest

        event = AstrMessageEvent()
        req = ProviderRequest()

        if default_persona:
            persona = {"name": "default", "prompt": prompt}
        else:
            persona = {"name": persona_name, "prompt": prompt}

        plugin.context.get_config = MagicMock(return_value={"provider_settings": {}})
        plugin.context.persona_manager = MagicMock()
        plugin.context.persona_manager.resolve_selected_persona = MagicMock(
            return_value=("永雏塔菲", persona, False, False)
        )

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

    def test_idempotent(self):
        """只尝试一次：解析失败后不再重试"""
        import asyncio
        from astrbot.api.event import AstrMessageEvent
        from astrbot.api.provider import ProviderRequest

        plugin = make_plugin()
        plugin.context.persona_manager = MagicMock()
        plugin.context.persona_manager.resolve_selected_persona = MagicMock(side_effect=Exception("boom"))

        async def go():
            event = AstrMessageEvent()
            req = ProviderRequest()
            await plugin._ensure_bot_name(event, req)  # 第一次失败
            plugin._bot_name_resolved = True
            await plugin._ensure_bot_name(event, req)  # 第二次不应再调用

        asyncio.run(go())
        self.assertEqual(plugin.context.persona_manager.resolve_selected_persona.call_count, 1)
        self.assertIsNone(plugin._resolved_bot_name)

    def test_skips_when_config_set(self):
        """config.bot_name 已设置 → 不执行解析"""
        import asyncio
        from astrbot.api.event import AstrMessageEvent
        from astrbot.api.provider import ProviderRequest

        plugin = make_plugin(bot_name="手动设置")
        plugin.context.persona_manager = MagicMock()
        plugin.context.get_config = MagicMock(return_value={})

        async def go():
            event = AstrMessageEvent()
            req = ProviderRequest()
            await plugin._ensure_bot_name(event, req)

        asyncio.run(go())
        plugin.context.persona_manager.resolve_selected_persona.assert_not_called()


if __name__ == "__main__":
    unittest.main()
