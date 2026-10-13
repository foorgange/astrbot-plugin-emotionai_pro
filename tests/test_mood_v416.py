# tests/test_mood_v416.py
"""v4.1.6 心情系统优化回归测试

三项改动的锁定：
1. 主导心情描述不再自相矛盾（「愤怒（心情平静）」→「愤怒（有点生气）」）
2. 全局心情注入主对话回复逻辑，且注入只在尾部（不影响前缀缓存命中）
3. /心情 命令展示最近三次变化（GlobalMood.mood_history 环形缓冲）
"""
import asyncio
import os
import sys
import time
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401  (注入 astrbot stub)
import tests.bootstrap  # noqa: F401

from emotionai_pro.global_mood import (  # noqa: E402
    GlobalMood, describe_mood, MOOD_HISTORY_LIMIT,
    _MOOD_EMOTION_GRADES, _MOOD_NEUTRAL_GRADES,
    preview_time_decayed,
)
from emotionai_pro.models import EmotionalMetrics, EnhancedEmotionalState  # noqa: E402
from emotionai_pro.main import EmotionAIProPlugin, _AI_STANDALONE_RE  # noqa: E402
from emotionai_pro.config import PluginConfig, PrivacyLevel  # noqa: E402
from emotionai_pro.command_handlers import UserCommandHandler  # noqa: E402


def _mood_with(**dims):
    """按 8 维情绪值构造心情（自动重算强度/主导）"""
    mood = GlobalMood(emotions=EmotionalMetrics(**dims))
    mood._recompute()
    return mood


def _ctx_plugin(mood, **cfg_over):
    """构造最小插件实例：只够跑 _build_enhanced_context / _format_emotional_state"""
    cfg = {"bot_name": "塔菲"}
    cfg.update(cfg_over)
    plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
    plugin.config = PluginConfig(**cfg)
    plugin._resolved_bot_name = "塔菲"
    plugin._bot_name_resolved = True
    plugin._raw_config = None
    plugin.attitude_manager = types.SimpleNamespace(
        get_tone_instruction=lambda st: "用撒娇的语气回应，可以带一点傲娇"
    )
    plugin.memory_system = types.SimpleNamespace(
        get_relationship_context=lambda k: "【长期关系发展轨迹】\n深度互动次数: 5"
    )
    plugin.analyzer = types.SimpleNamespace(get_dominant_emotion=lambda st: "喜悦")
    plugin.get_mood_sync = lambda: mood
    state = EnhancedEmotionalState(user_key="u1", favor=20, intimacy=10)
    return plugin, state


def _make_hook_plugin(mood, **cfg_over):
    """构造可跑 inject_enhanced_context 钩子的插件实例（协作者全部桩化）"""
    cfg = {"bot_name": "塔菲", "session_gap_minutes": 60, "session_based": False}
    cfg.update(cfg_over)
    plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
    plugin.config = PluginConfig(**cfg)
    plugin._resolved_bot_name = "塔菲"
    plugin._bot_name_resolved = True
    plugin._raw_config = None
    plugin.context = types.SimpleNamespace(get_config=lambda umo=None: None)
    plugin._mood_cache = mood
    plugin._mood_cache_time = time.time()

    state = EnhancedEmotionalState(user_key="u1", favor=20, intimacy=10)

    class _Cache:
        def __init__(self):
            self.store = {}

        async def get(self, key):
            return self.store.get(key)

        async def set(self, key, value, ttl=None):
            self.store[key] = value

    plugin.cache = _Cache()

    async def _get_state(key):
        return state

    plugin.user_manager = types.SimpleNamespace(get_user_state=_get_state)
    plugin.attitude_manager = types.SimpleNamespace(
        get_tone_instruction=lambda st: "用撒娇的语气回应，可以带一点傲娇"
    )
    plugin.memory_system = types.SimpleNamespace(
        get_relationship_context=lambda k: "【长期关系发展轨迹】\n深度互动次数: 5"
    )
    plugin.analyzer = types.SimpleNamespace(get_dominant_emotion=lambda st: "喜悦")
    return plugin, state


async def _noop_send_streaming(generator, use_fallback=False):
    return None


def _run_hook(plugin):
    """跑一次 on_llm_request 钩子，返回 (event, req) 供断言"""
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import ProviderRequest

    event = AstrMessageEvent()
    event.send_streaming = _noop_send_streaming  # stub 事件默认没有，补一个
    req = ProviderRequest()

    async def go():
        await plugin.inject_enhanced_context(event, req)

    asyncio.run(go())
    return event, req


class TestDescribeMood(unittest.TestCase):
    """任务1：主导情绪 + 强度 → 分级描述，杜绝「愤怒（心情平静）」"""

    def test_anger_low_not_calm(self):
        """用户报的场景原样复刻：愤怒 0.11"""
        label = describe_mood("愤怒", 0.11)
        self.assertEqual(label, "有点生气")
        self.assertNotIn("平静", label)

    def test_all_emotions_never_use_neutral_words(self):
        """8 个具体情绪 × 各强度档，都必须是该情绪的分级描述"""
        for name, grades in _MOOD_EMOTION_GRADES.items():
            for intensity in (0.05, 0.2, 0.4, 0.6, 0.9):
                label = describe_mood(name, intensity)
                self.assertIn(label, grades, f"{name}@{intensity} → {label}")
                for neutral in _MOOD_NEUTRAL_GRADES:
                    self.assertNotEqual(
                        label, neutral,
                        f"{name}@{intensity} 不该用整体心境词 {neutral}",
                    )

    def test_neutral_keeps_generic_five_grades(self):
        """中立主导保持原有 5 档整体心境词（行为不变）"""
        self.assertEqual(describe_mood("中立", 0.1), "心情平静")
        self.assertEqual(describe_mood("中立", 0.2), "心情平稳")
        self.assertEqual(describe_mood("中立", 0.4), "心情微动")
        self.assertEqual(describe_mood("中立", 0.6), "心情波动")
        self.assertEqual(describe_mood("中立", 0.9), "情绪高涨")

    def test_grade_boundaries(self):
        """强度分档阈值 0.15/0.35/0.55/0.75"""
        self.assertEqual(describe_mood("愤怒", 0.149), "有点生气")
        self.assertEqual(describe_mood("愤怒", 0.15), "挺恼火")
        self.assertEqual(describe_mood("愤怒", 0.35), "很生气")
        self.assertEqual(describe_mood("愤怒", 0.55), "非常生气")
        self.assertEqual(describe_mood("愤怒", 0.75), "气炸了")

    def test_composite_and_unknown_fall_back(self):
        """并列复合/未知/空主导 → 回退整体心境词，不抛异常"""
        self.assertEqual(describe_mood("复合(喜悦+信任)", 0.5), "心情微动")
        self.assertEqual(describe_mood("", 0.5), "心情微动")
        self.assertEqual(describe_mood(None, 0.5), "心情微动")
        self.assertEqual(describe_mood("喜怒哀乐", 0.9), "情绪高涨")

    def test_get_mood_label_delegates(self):
        """_get_mood_label 委托 describe_mood，第二参数可缺省（中立）"""
        plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
        bound = EmotionAIProPlugin._get_mood_label.__get__(plugin)
        self.assertEqual(bound(0.11, "愤怒"), "有点生气")
        self.assertEqual(bound(0.1), "心情平静")
        self.assertEqual(bound(0.9, "喜悦"), "兴高采烈")

    def test_panel_no_longer_contradictory(self):
        """情感状态面板（/查看好感）同样不再出现矛盾描述"""
        from emotionai_pro.managers import EmotionAnalyzer
        from emotionai_pro.relationship_manager import DynamicWeightManager

        plugin, state = _ctx_plugin(_mood_with(anger=11))
        plugin.config = PluginConfig(
            bot_name="塔菲", global_privacy_level=PrivacyLevel.DETAILED
        )
        plugin.weight_manager = DynamicWeightManager()
        plugin.analyzer = EmotionAnalyzer()
        text = plugin._format_emotional_state(state)
        self.assertIn("愤怒 (有点生气)", text)
        self.assertNotIn("心情平静", text)


class TestMoodInjectionContent(unittest.TestCase):
    """任务2：全局心情进入 _build_enhanced_context（回复逻辑）"""

    def test_mood_section_present(self):
        plugin, state = _ctx_plugin(_mood_with(anger=11))
        text = plugin._build_enhanced_context(state)
        self.assertIn("【当前心情】", text)
        self.assertIn("主导情绪：愤怒（有点生气）", text)
        self.assertIn("强度：0.11/1", text)
        self.assertIn("说明：", text)
        # 原有结构完好
        self.assertIn("【当前情感状态】", text)
        self.assertIn("[需要情感评估]", text)

    def test_switch_off_removes_only_mood_section(self):
        """行为测试（非源码断言）：关闭开关后心情段落消失，其余注入不动"""
        plugin, state = _ctx_plugin(_mood_with(anger=11), enable_mood_influence=False)
        text = plugin._build_enhanced_context(state)
        self.assertNotIn("【当前心情】", text)
        self.assertNotIn("主导情绪", text)
        self.assertIn("【当前情感状态】", text)
        self.assertIn("[需要情感评估]", text)
        self.assertIn("好感度", text)

    def test_neutral_mood_section_natural(self):
        plugin, state = _ctx_plugin(GlobalMood.default())
        text = plugin._build_enhanced_context(state)
        self.assertIn("主导情绪：中立（心情平静）", text)

    def test_no_standalone_ai_in_injection(self):
        plugin, state = _ctx_plugin(_mood_with(joy=80))
        text = plugin._build_enhanced_context(state)
        self.assertIsNone(_AI_STANDALONE_RE.search(text))


class TestMoodInjectionCacheFriendly(unittest.TestCase):
    """任务2：注入只在尾部发生，前缀（system_prompt + 历史）一动不动"""

    def test_injection_goes_to_tail_only(self):
        mood = _mood_with(anger=11)
        plugin, _ = _make_hook_plugin(mood)
        _event, req = _run_hook(plugin)

        # 前缀缓存三要素：一个都没被碰
        self.assertEqual(req.system_prompt, "")
        self.assertEqual(req.contexts, [])
        # 注入只追加到用户消息之后
        self.assertEqual(len(req.extra_user_content_parts), 1)
        text = req.extra_user_content_parts[0].text
        self.assertIn("【当前心情】", text)
        self.assertIn("主导情绪：愤怒（有点生气）", text)

    def test_prefix_stable_when_mood_changes(self):
        """心情变了也只变尾部：system_prompt/contexts 连续两次保持一致"""
        plugin1, _ = _make_hook_plugin(_mood_with(anger=11))
        _e1, req1 = _run_hook(plugin1)
        plugin2, _ = _make_hook_plugin(_mood_with(joy=90))
        _e2, req2 = _run_hook(plugin2)

        self.assertEqual(req1.system_prompt, req2.system_prompt)
        self.assertEqual(req1.contexts, req2.contexts)
        self.assertNotEqual(
            req1.extra_user_content_parts[0].text,
            req2.extra_user_content_parts[0].text,
        )
        self.assertIn("有点生气", req1.extra_user_content_parts[0].text)
        self.assertIn("兴高采烈", req2.extra_user_content_parts[0].text)

    def test_switch_off_hook_no_mood_section(self):
        """钩子级行为测试：关闭开关后尾部注入不含心情段落"""
        plugin, _ = _make_hook_plugin(_mood_with(anger=11), enable_mood_influence=False)
        _event, req = _run_hook(plugin)
        text = req.extra_user_content_parts[0].text
        self.assertNotIn("【当前心情】", text)
        self.assertIn("【当前情感状态】", text)
        # 前缀依旧没被碰
        self.assertEqual(req.system_prompt, "")
        self.assertEqual(req.contexts, [])


class TestMoodHistory(unittest.TestCase):
    """任务3：mood_history 环形缓冲（最近三次，新的在前）"""

    def test_push_keeps_three_newest_first(self):
        mood = GlobalMood()
        mood.push_change_reason("第一次", 100.0)
        mood.push_change_reason("第二次", 200.0)
        mood.push_change_reason("第三次", 300.0)
        mood.push_change_reason("第四次", 400.0)
        self.assertEqual(len(mood.mood_history), MOOD_HISTORY_LIMIT)
        self.assertEqual(
            [e["reason"] for e in mood.mood_history],
            ["第四次", "第三次", "第二次"],
        )
        # 「最近一次」镜像恒等于 history[0]
        self.assertEqual(mood.last_change_reason, "第四次")
        self.assertEqual(mood.last_change_at, 400.0)

    def test_same_text_refreshes_timestamp_only(self):
        """同一句话的连续波动不重复压栈（避免三次列表出现两条一样记录）"""
        mood = GlobalMood()
        mood.push_change_reason("因为听到「哈哈」，永雏塔菲变得开心起来", 100.0)
        mood.push_change_reason("因为听到「哈哈」，永雏塔菲变得开心起来", 200.0)
        self.assertEqual(len(mood.mood_history), 1)
        self.assertEqual(mood.mood_history[0]["at"], 200.0)
        self.assertEqual(mood.last_change_at, 200.0)

    def test_push_ignores_empty_and_non_str(self):
        mood = GlobalMood()
        mood.push_change_reason("", 100.0)
        mood.push_change_reason(None, 100.0)
        mood.push_change_reason(123, 100.0)
        self.assertEqual(mood.mood_history, [])
        self.assertEqual(mood.last_change_reason, "")

    def test_roundtrip_to_from_dict(self):
        mood = GlobalMood()
        mood.push_change_reason("原因甲", 111.0)
        mood.push_change_reason("原因乙", 222.0)
        restored = GlobalMood.from_dict(mood.to_dict())
        self.assertEqual(
            restored.mood_history,
            [
                {"reason": "原因乙", "at": 222.0},
                {"reason": "原因甲", "at": 111.0},
            ],
        )
        self.assertEqual(restored.last_change_reason, "原因乙")

    def test_legacy_archive_migrates_single_entry(self):
        """v4.1.5 及之前的旧存档只有单条 last_change_reason → 迁移成历史首条"""
        mood = GlobalMood.from_dict({
            "emotions": {"joy": 10},
            "updated_at": 1.0,
            "last_change_reason": "因为听到「哈哈」，永雏塔菲变得开心起来",
            "last_change_at": 123.0,
        })
        self.assertEqual(
            mood.mood_history,
            [{"reason": "因为听到「哈哈」，永雏塔菲变得开心起来", "at": 123.0}],
        )

    def test_corrupted_history_tolerated(self):
        """历史字段各种花式损坏 → 容错跳过，不抛异常"""
        mood = GlobalMood.from_dict({"emotions": {}, "mood_history": "不是列表"})
        self.assertEqual(mood.mood_history, [])
        mood2 = GlobalMood.from_dict({
            "emotions": {},
            "mood_history": [
                {"reason": "好的", "at": "坏时间"}, 1, "x", {"reason": None},
            ],
        })
        self.assertEqual(mood2.mood_history, [{"reason": "好的", "at": 0.0}])

    def test_preview_carries_history_without_mutation(self):
        """衰减预览把历史带出去，且原存档分毫未动"""
        mood = _mood_with(joy=80)
        mood.updated_at = time.time() - 2 * 3600
        mood.push_change_reason("原因甲", 111.0)
        preview = preview_time_decayed(mood)
        self.assertEqual(preview.mood_history, mood.mood_history)
        self.assertIsNot(preview.mood_history, mood.mood_history)
        self.assertEqual(mood.emotions.joy, 80)
        self.assertEqual(len(mood.mood_history), 1)

    def test_to_dict_history_is_copy(self):
        """to_dict 返回的条目是拷贝，调用方改不到原对象"""
        mood = GlobalMood()
        mood.push_change_reason("原因甲", 111.0)
        data = mood.to_dict()
        data["mood_history"][0]["reason"] = "被篡改"
        self.assertEqual(mood.mood_history[0]["reason"], "原因甲")


class TestMoodCommandThreeChanges(unittest.TestCase):
    """任务3：/心情 命令展示最近三次变化"""

    def _run_mood_command(self, mood, privacy=PrivacyLevel.BASIC):
        handler = UserCommandHandler.__new__(UserCommandHandler)
        handler.config = types.SimpleNamespace(global_privacy_level=privacy)
        handler.plugin = types.SimpleNamespace(
            get_mood_sync=lambda: mood,
            _get_mood_label=EmotionAIProPlugin._get_mood_label.__get__(
                types.SimpleNamespace()
            ),
        )
        from astrbot.api.event import AstrMessageEvent

        event = AstrMessageEvent()

        async def go():
            chunks = []
            async for r in handler.show_global_mood(event):
                chunks.append(r)
            return "\n".join(chunks)

        return asyncio.run(go())

    def _mood_with_history(self, n):
        mood = _mood_with(anger=11)
        mood.updated_at = time.time()
        now = time.time()
        for i in range(n):
            mood.push_change_reason(f"第{i + 1}次变化", now - (n - i) * 60)
        return mood

    def test_shows_three_changes_newest_first(self):
        text = self._run_mood_command(self._mood_with_history(5))
        self.assertIn("【最近三次变化】", text)
        self.assertIn("1. 第5次变化", text)
        self.assertIn("2. 第4次变化", text)
        self.assertIn("3. 第3次变化", text)
        self.assertNotIn("第2次变化", text)
        self.assertNotIn("第1次变化", text)

    def test_label_not_contradictory(self):
        text = self._run_mood_command(self._mood_with_history(2))
        self.assertIn("愤怒（有点生气）", text)
        self.assertNotIn("心情平静", text)

    def test_fewer_than_three_shows_what_exists(self):
        text = self._run_mood_command(self._mood_with_history(1))
        self.assertIn("1. 第1次变化", text)
        self.assertIn("【最近三次变化】", text)

    def test_empty_history_message(self):
        text = self._run_mood_command(_mood_with(anger=11))
        self.assertIn("还没有值得记录的心情变化", text)

    def test_full_secret_hides_all(self):
        text = self._run_mood_command(
            self._mood_with_history(3), privacy=PrivacyLevel.FULL_SECRET
        )
        self.assertIn("【当前心情】*保密*", text)
        self.assertNotIn("最近三次变化", text)

    def test_legacy_single_reason_displayed(self):
        """旧存档（无 mood_history、只有 last_change_reason）也能展示"""
        mood = _mood_with(anger=11)
        mood.updated_at = time.time()
        mood.last_change_reason = "因为听到「气死」，永雏塔菲变得生气起来"
        mood.last_change_at = time.time() - 120
        text = self._run_mood_command(mood)
        self.assertIn("1. 因为听到「气死」，永雏塔菲变得生气起来", text)


if __name__ == "__main__":
    unittest.main()
