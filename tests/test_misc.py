# tests/test_misc.py
"""杂项：cache 无 xxhash 导入、config bot_name、_format_emotional_state 显示"""
import sys
import os
import asyncio
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

import emotionai_pro.cache as cache_mod  # noqa: E402
from emotionai_pro.cache import ShardedTTLCache  # noqa: E402
from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402
from emotionai_pro.main import EmotionAIProPlugin  # noqa: E402
from emotionai_pro.global_mood import GlobalMood  # noqa: E402


def _default_mood():
    return GlobalMood.default()


class TestCacheNoBareXxhash(unittest.TestCase):
    """cache.py 不再裸导入 xxhash（本机无 xxhash 也能 import + fallback）"""

    def test_import_works_without_xxhash(self):
        import sys as _sys
        if "xxhash" in _sys.modules:
            _sys.modules.pop("xxhash")
        import importlib
        cache_mod_clean = importlib.reload(cache_mod)
        self.assertFalse(cache_mod_clean.XXHASH_AVAILABLE)
        # 仍能通过 hashlib fallback 正常工作（缓存启动需在事件循环内）
        async def go():
            c = ShardedTTLCache(max_size=10, shard_count=2, default_ttl=60)
            self.assertTrue(hasattr(c, "_get_shard"))
            await c.close()
        asyncio.run(go())

    def test_set_get_roundtrip(self):
        async def go():
            c = ShardedTTLCache(max_size=10, shard_count=2, default_ttl=60)
            await c.set("k", {"v": 1})
            val = await c.get("k")
            await c.close()
            return val
        result = asyncio.run(go())
        self.assertEqual(result, {"v": 1})


class TestConfigBotName(unittest.TestCase):
    def test_default_none(self):
        """bot_name 默认 None"""
        cfg = PluginConfig()
        self.assertIsNone(cfg.bot_name)

    def test_set_value(self):
        cfg = PluginConfig(bot_name="塔菲")
        self.assertEqual(cfg.bot_name, "塔菲")


class TestFormatEmotionalState(unittest.TestCase):
    def _make_plugin(self, bot_name="塔菲", privacy=2):
        plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
        plugin.config = PluginConfig(bot_name=bot_name, global_privacy_level=privacy)
        from emotionai_pro.relationship_manager import DynamicWeightManager
        from emotionai_pro.attitude_manager import AttitudeRelationshipManager
        from emotionai_pro.managers import EmotionAnalyzer
        plugin.weight_manager = DynamicWeightManager()
        plugin.analyzer = EmotionAnalyzer()
        plugin.attitude_manager = AttitudeRelationshipManager()
        plugin._resolved_bot_name = None
        plugin._mood_cache = None
        plugin._mood_cache_time = 0.0
        plugin._get_mood_label = EmotionAIProPlugin._get_mood_label.__get__(plugin)
        plugin._sanitize_ai_text = EmotionAIProPlugin._sanitize_ai_text.__get__(plugin)
        plugin.get_mood_sync = lambda: _default_mood()
        return plugin

    def test_detailed_has_next_stage_true(self):
        """DETAILED 分支显示真·下一阶段阈值"""
        from emotionai_pro.global_mood import GlobalMood
        from emotionai_pro.models import EmotionalMetrics
        plugin = self._make_plugin()
        mood = GlobalMood(emotions=EmotionalMetrics(joy=60, trust=40))
        mood._recompute()
        plugin.get_mood_sync = lambda: mood

        state = EnhancedEmotionalState(user_key="u1", favor=20, intimacy=10)
        text = plugin._format_emotional_state(state)
        # 初识期 → 下一阶段 深化期 (55)
        self.assertIn("下一阶段", text)
        self.assertIn("55", text)
        self.assertIn("深化期", text)

    def test_basic_has_next_stage(self):
        """BASIC 分支包含下一阶段字段"""
        plugin = self._make_plugin(privacy=1)
        state = EnhancedEmotionalState(user_key="u1", favor=20, intimacy=10)
        text = plugin._format_emotional_state(state)
        self.assertIn("下一阶段", text)
        self.assertIn("55", text)

    def test_mood_uses_global(self):
        """心情显示来自全局心情而非 per-user"""
        from emotionai_pro.global_mood import GlobalMood
        from emotionai_pro.models import EmotionalMetrics
        plugin = self._make_plugin()
        mood = GlobalMood(emotions=EmotionalMetrics(joy=100, trust=0))
        mood._recompute()
        plugin.get_mood_sync = lambda: mood

        # 用户的 per-user 情绪全零
        state = EnhancedEmotionalState(user_key="u1", favor=20, intimacy=10)
        text = plugin._format_emotional_state(state)
        self.assertIn("喜悦", text)  # 全局心情主导情感
        self.assertIn("心情微动", text)  # intensity=50 → 心情微动

    def test_ai_sanitized_in_output(self):
        """描述中的 AI 被替换为 bot 人设名"""
        plugin = self._make_plugin()
        state = EnhancedEmotionalState(user_key="u1", favor=20, intimacy=10)
        state.descriptions.relationship = "亲密玩闹的ai伙伴"
        state.descriptions.attitude = "对AI温柔"
        text = plugin._format_emotional_state(state)
        self.assertNotIn("AI", text)
        self.assertIn("亲密玩闹的塔菲伙伴", text)

    def test_symbiosis_no_next_threshold(self):
        """共生期 → 显示已达最高阶段"""
        plugin = self._make_plugin()
        state = EnhancedEmotionalState(user_key="u1", favor=98, intimacy=99)
        text = plugin._format_emotional_state(state)
        self.assertIn("已达最高阶段", text)


if __name__ == "__main__":
    unittest.main()
