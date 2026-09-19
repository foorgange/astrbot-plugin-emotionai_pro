# tests/test_bug_fixes.py
"""崩溃 bug 修复离线单测

- DebugCommandHandler 具备 _is_admin（不再 AttributeError）
- fix_interaction_stats: positive + negative == total
- _apply_expert_updates 互动计数不重复
- stability_score 无 -inf
- 版本号在 __init__.py / main.py / storage.py / metadata.yaml / README.md 五处一致
"""
import sys
import os
import re
import types
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
            # 里程碑加成不在本测试范围（由 tests/test_intimacy_gate.py 覆盖），
            # 这里给空操作桩；桩必须与真身同构，否则 _apply_expert_updates
            # 一接入新逻辑就会 AttributeError。
            "_apply_intimacy_milestones": lambda self, s, u: None,
            "config": type("C", (), {"favour_min": -100, "favour_max": 100,
                                     "intimacy_min": 0, "intimacy_max": 100,
                                     "transition_intimacy_pct": 50,
                                     "intimacy_first_deep_bonus": 3,
                                     "intimacy_streak_days": 3,
                                     "intimacy_streak_bonus": 1})(),
            "weight_manager": type("WM", (), {"apply_transition_benefits": lambda self, s, u: u,
                                              "is_favor_frozen": lambda self, s: False})(),
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
            # 里程碑加成不在本测试范围（由 tests/test_intimacy_gate.py 覆盖）
            "_apply_intimacy_milestones": lambda self, s, u: None,
            "config": type("C", (), {"favour_min": -100, "favour_max": 100,
                                     "intimacy_min": 0, "intimacy_max": 100,
                                     "transition_intimacy_pct": 50,
                                     "intimacy_first_deep_bonus": 3,
                                     "intimacy_streak_days": 3,
                                     "intimacy_streak_bonus": 1})(),
            "weight_manager": type("WM", (), {"apply_transition_benefits": lambda self, s, u: u,
                                              "is_favor_frozen": lambda self, s: False})(),
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
    """版本号在 __init__.py / main.py @register / storage.py 三处保持一致

    以 __init__.py 的 __version__ 为唯一真源，避免每次发版都要改测试。
    """

    @staticmethod
    def _expected() -> str:
        """从 __init__.py 源码解析 __version__（唯一真源）

        注意：不能 `import emotionai_pro` 取 __version__ —— tests/bootstrap.py
        注册的是**合成包**（只设 __path__，不执行 __init__.py），拿不到该属性。
        """
        init_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__init__.py")
        src = open(init_path, encoding="utf-8").read()
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', src, re.M)
        if not m:
            raise AssertionError("__init__.py 中未找到 __version__ 定义")
        return m.group(1)

    def test_semver_shape(self):
        """版本号形如 x.y.z"""
        self.assertRegex(self._expected(), r"^\d+\.\d+\.\d+$")

    def test_init_version(self):
        init_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__init__.py")
        src = open(init_path, encoding="utf-8").read()
        self.assertIn(f'__version__ = "{self._expected()}"', src)

    def test_main_register_version(self):
        """@register 装饰器版本与 __version__ 一致"""
        main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        src = open(main_path, encoding="utf-8").read()
        self.assertIn(f'"{self._expected()}"', src)
        self.assertNotIn('"4.0.0"', src)

    def test_storage_version(self):
        import emotionai_pro.storage as storage_mod
        src = open(storage_mod.__file__, encoding="utf-8").read()
        self.assertIn(f"'{self._expected()}'", src)

    def test_metadata_yaml_version(self):
        """metadata.yaml 的 version 与 __version__ 一致（插件市场读这里）"""
        meta_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadata.yaml")
        src = open(meta_path, encoding="utf-8").read()
        self.assertIn(f"version: v{self._expected()}", src)

    def test_readme_title_version(self):
        """README 标题里的版本号与 __version__ 一致"""
        readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
        src = open(readme_path, encoding="utf-8").read()
        self.assertIn(f"v{self._expected()}", src.splitlines()[0])


class TestViewFavorHeaderNoDuplicateUser(unittest.TestCase):
    """管理员 `/查看好感 <id>` 的输出头部不得重复出现「用户」字样。

    根因：`RankingManager._format_user_display()` 的契约是**自带** `用户` 前缀
    （排行榜里 `{rank}. {display_name}` 是独立使用的，models.py 的
    `RankingEntry.display_name` 默认值也带前缀），而头部模板又写了一次「用户」，
    于是输出成「【用户 用户3418451176 完整情感状态】」。
    修法：模板去掉多余的「用户」，保留 `_format_user_display` 的前缀。
    """

    USER_INPUT = "3418451176"

    def _run_view_favor(self, user_input=None):
        from emotionai_pro.managers import RankingManager
        from astrbot.api.event import AstrMessageEvent

        user_input = user_input or self.USER_INPUT

        state = types.SimpleNamespace(
            descriptions=types.SimpleNamespace(
                attitude="用卖萌化解你的忧心", relationship="你和AI的超亲密日常"
            ),
            favor=100,
            intimacy=100,
            show_status=False,
            stats=types.SimpleNamespace(
                total_count=601, positive_count=596, negative_count=5,
                last_interaction_time=0.0,
            ),
            emotions=types.SimpleNamespace(
                joy=100, trust=100, fear=0, surprise=100,
                sadness=21, disgust=17, anger=16, anticipation=100,
            ),
        )

        plugin = types.SimpleNamespace(
            get_mood_sync=lambda: types.SimpleNamespace(
                dominant_emotion="喜悦", intensity=1.0
            ),
            _sanitize_ai_text=lambda t: t,
            _get_mood_label=lambda i: "情绪高涨",
        )

        handler = AdminCommandHandler.__new__(AdminCommandHandler)
        handler.plugin = plugin
        handler.config = types.SimpleNamespace(admin_qq_list=[], session_based=False)
        handler.user_manager = types.SimpleNamespace(
            get_user_state=lambda k: _async_val(state),
            resolve_user_key=lambda s, sb: s,
        )
        handler.ranking_manager = types.SimpleNamespace(
            # 用真实实现 —— 它才是加「用户」前缀的源头，stub 掉就测不出这个 bug
            _format_user_display=lambda k: RankingManager._format_user_display(None, k)
        )
        handler.weight_manager = types.SimpleNamespace(
            get_stage_info=lambda st: {
                "stage_name": "共生期",
                "progress_to_next": 100.0,
                "favor_weight": 0.5,
                "intimacy_weight": 0.5,
                "is_transitioning": True,
                "intimacy_boost_active": False,
            }
        )
        handler.analyzer = types.SimpleNamespace(
            get_emotional_profile=lambda st, fw, iw: {"composite_score": 100.0}
        )

        import asyncio

        async def go():
            event = AstrMessageEvent(role="admin")
            chunks = []
            async for r in handler.view_favor(event, user_input):
                chunks.append(r)
            return "\n".join(chunks)

        return asyncio.run(go())

    def test_no_duplicate_user_prefix(self):
        """输出中不得出现「用户 用户」这种重复前缀"""
        text = self._run_view_favor()
        self.assertNotIn("用户 用户", text)

    def test_header_uses_single_prefix(self):
        """头部应为「【用户3418451176 完整情感状态】」"""
        text = self._run_view_favor()
        self.assertIn("【用户3418451176 完整情感状态】", text)

    def test_header_is_first_line(self):
        """头部仍是第一行，其余行不受影响"""
        text = self._run_view_favor()
        lines = text.splitlines()
        self.assertEqual(lines[0], "【用户3418451176 完整情感状态】")
        self.assertEqual(lines[1], f"用户标识: {self.USER_INPUT}")

    def test_no_user_prefix_when_unknown(self):
        """空输入走保护分支，不会崩"""
        text = self._run_view_favor(user_input="x")
        self.assertIn("【用户x 完整情感状态】", text)


if __name__ == "__main__":
    unittest.main()
