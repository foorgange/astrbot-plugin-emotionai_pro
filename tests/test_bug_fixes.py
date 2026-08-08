# tests/test_bug_fixes.py
"""崩溃 bug 修复离线单测

- DebugCommandHandler 具备 _is_admin（不再 AttributeError）
- fix_interaction_stats: positive + negative == total
- _apply_expert_updates 互动计数不重复
- stability_score 无 -inf
- 版本号统一 4.0.6
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.command_handlers import DebugCommandHandler, AdminCommandHandler, UserCommandHandler  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402
from emotionai_pro.managers import EmotionAnalyzer  # noqa: E402


async def _async_val(val):
    return val


class TestDebugCommandHandlerIsAdmin(unittest.TestCase):
    """DebugCommandHandler 崩溃修复：_is_admin 上移到 BaseCommandHandler"""

    def test_all_handlers_have_is_admin(self):
        plugin = type("P", (), {"config": type("C", (), {"admin_qq_list": []})()})()
        for cls in (UserCommandHandler, AdminCommandHandler, DebugCommandHandler):
            handler = cls.__new__(cls)
            handler.config = plugin.config
            self.assertTrue(hasattr(handler, "_is_admin"),
                            f"{cls.__name__} 缺少 _is_admin（会导致 AttributeError 崩溃）")
            self.assertTrue(hasattr(handler, "_resolve_user_key"),
                            f"{cls.__name__} 缺少 _resolve_user_key")


class TestFixInteractionStats(unittest.TestCase):
    """fix_interaction_stats: negative_count = total - positive"""

    async def _run_fix(self, state, total):
        state.stats.total_count = total
        state.stats.positive_count = 0
        state.stats.negative_count = 0

        plugin = type("P", (), {
            "_get_user_key": lambda self, e: "u1",
            "user_manager": type("UM", (), {
                "get_user_state": lambda self, k: _async_val(state),
                "update_user_state": lambda self, k, s: _async_val(None),
            })(),
        })()
        handler = DebugCommandHandler.__new__(DebugCommandHandler)
        handler.plugin = plugin
        handler.user_manager = plugin.user_manager

        from astrbot.api.event import AstrMessageEvent
        event = AstrMessageEvent(role="admin")
        # 迭代生成器
        results = []
        async for r in handler.fix_interaction_stats(event):
            results.append(r)
        return state

    def test_positive_plus_negative_equals_total(self):
        import asyncio
        state = EnhancedEmotionalState(user_key="u1")
        result = asyncio.run(self._run_fix(state, 10))
        self.assertEqual(result.stats.positive_count + result.stats.negative_count,
                         result.stats.total_count)
        self.assertEqual(result.stats.total_count, 10)

    def test_zero_total_skips(self):
        """total=0 时不应修改任何值"""
        import asyncio
        state = EnhancedEmotionalState(user_key="u1")
        result = asyncio.run(self._run_fix(state, 0))
        self.assertEqual(result.stats.total_count, 0)


class TestApplyExpertUpdatesNoDoubleCount(unittest.TestCase):
    """互动计数只在 _apply_expert_updates 内记录一次"""

    def test_record_interaction_three_state(self):
        """正面>负面 → positive 计数 +1"""
        plugin = type("P", (), {
            "_sanitize_ai_text": lambda self, t: t,
            "config": type("C", (), {"favour_min": -100, "favour_max": 100,
                                     "intimacy_min": 0, "intimacy_max": 100})(),
            "weight_manager": type("WM", (), {"apply_transition_benefits": lambda self, s, u: u})(),
        })()
        from emotionai_pro.main import EmotionAIProPlugin
        # 直接调用 unbound 方法
        state = EnhancedEmotionalState(user_key="u1")
        updates = {"joy": 5, "trust": 5, "favor": 3, "source": "llm_analysis", "llm_available": True}
        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)
        self.assertEqual(state.stats.total_count, 1)
        self.assertEqual(state.stats.positive_count, 1)
        self.assertEqual(state.stats.negative_count, 0)

    def test_negative_interaction(self):
        """负面>正面 → negative 计数 +1"""
        plugin = type("P", (), {
            "_sanitize_ai_text": lambda self, t: t,
            "config": type("C", (), {"favour_min": -100, "favour_max": 100,
                                     "intimacy_min": 0, "intimacy_max": 100})(),
            "weight_manager": type("WM", (), {"apply_transition_benefits": lambda self, s, u: u})(),
        })()
        from emotionai_pro.main import EmotionAIProPlugin
        state = EnhancedEmotionalState(user_key="u1")
        updates = {"joy": -5, "favor": -3, "source": "llm_analysis", "llm_available": True}
        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)
        self.assertEqual(state.stats.total_count, 1)
        self.assertEqual(state.stats.negative_count, 1)
        self.assertEqual(state.stats.positive_count, 0)

    def test_old_double_count_method_removed(self):
        """_update_interaction_stats 方法已删除"""
        from emotionai_pro.main import EmotionAIProPlugin
        self.assertFalse(hasattr(EmotionAIProPlugin, "_update_interaction_stats"))


class TestStabilityScoreNoInf(unittest.TestCase):
    """managers.get_emotional_profile 不再返回 -inf"""

    def test_never_interacted_gives_finite_stability(self):
        state = EnhancedEmotionalState(user_key="u1", favor=50, intimacy=50)
        analyzer = EmotionAnalyzer()
        profile = analyzer.get_emotional_profile(state, 0.5, 0.5)
        self.assertTrue(profile["stability_score"] >= 0)
        self.assertTrue(profile["stability_score"] <= 100)
        import math
        self.assertTrue(math.isfinite(profile["stability_score"]))


class TestVersionConsistency(unittest.TestCase):
    """版本号统一为 4.0.7"""

    def test_init_version(self):
        init_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__init__.py")
        src = open(init_path, encoding="utf-8").read()
        self.assertIn('__version__ = "4.0.7"', src)

    def test_main_register_version(self):
        """@register 装饰器版本为 4.0.7"""
        main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        src = open(main_path, encoding="utf-8").read()
        self.assertIn('"4.0.7"', src)
        self.assertNotIn('"4.0.0"', src)

    def test_storage_version(self):
        import emotionai_pro.storage as storage_mod
        src = open(storage_mod.__file__, encoding="utf-8").read()
        self.assertIn("'4.0.7'", src)


if __name__ == "__main__":
    unittest.main()
