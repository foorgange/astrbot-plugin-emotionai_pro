# tests/test_intimacy_gate.py
"""v4.0.22 亲密度门槛 / 好感度冻结 / 亲密度里程碑 回归测试

需求（用户原话整理）：
    ① 阶段过渡时期，亲密度达不到「最大亲密度的百分比数值」就无法顺利
       完成过渡；
    ② 亲密度不达标时，好感度也不变化（冻结）；
    ③ 亲密度保留日常变化（幅度由用户自调低），但「首次深度交流、
       连续多日互动」两类里程碑让亲密度变化更频繁；
    ④ /好感度 命令在过渡期额外展示亲密度与到达下一阶段的差值。

本文件锁死以上行为，并覆盖关闭分支（阈值 0 / 奖励 0）与配置三处同步。
"""
import sys
import os
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.constants import EmotionConstants  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402
from emotionai_pro.relationship_manager import DynamicWeightManager  # noqa: E402
from emotionai_pro.main import EmotionAIProPlugin  # noqa: E402
from emotionai_pro.global_mood import GlobalMood  # noqa: E402


def _state(favor: int, intimacy: int, key: str = "u") -> EnhancedEmotionalState:
    return EnhancedEmotionalState(user_key=key, favor=favor, intimacy=intimacy)


def _days_ago(n: int) -> str:
    """n 天前的本地日期字符串（YYYY-MM-DD）"""
    return time.strftime("%Y-%m-%d", time.localtime(time.time() - n * 86400))


def _make_plugin(**cfg):
    """造一个够 _apply_expert_updates / _format_emotional_state 使用的插件实例"""
    from emotionai_pro.config import PluginConfig
    from emotionai_pro.managers import EmotionAnalyzer

    plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
    plugin.config = PluginConfig(**cfg)
    plugin.weight_manager = DynamicWeightManager()
    plugin.analyzer = EmotionAnalyzer()
    plugin._resolved_bot_name = "塔菲"
    plugin._mood_cache = None
    plugin._mood_cache_time = 0.0
    plugin._get_mood_label = EmotionAIProPlugin._get_mood_label.__get__(plugin)
    plugin._sanitize_ai_text = EmotionAIProPlugin._sanitize_ai_text.__get__(plugin)
    plugin.get_mood_sync = lambda: GlobalMood.default()
    return plugin


class _GateTestCase(unittest.TestCase):
    """每个用例前后都还原进程级单例，避免类属性/常量泄漏"""

    def setUp(self):
        DynamicWeightManager.reset()
        EmotionConstants.reset()

    def tearDown(self):
        DynamicWeightManager.reset()
        EmotionConstants.reset()


class TestIntimacyGateBlocksTransition(_GateTestCase):
    """门槛：分数够升级、亲密度没达标 → 过渡卡住且持续"""

    def test_blocked_when_intimacy_below_threshold(self):
        # favor 70 / intimacy 40：复合评分 55 刚好够升深化期，但门槛要 50
        state = _state(70, 40)
        info = DynamicWeightManager.get_stage_info(state)

        self.assertTrue(info["is_transitioning"], "亲密度未达标却放行了过渡")
        self.assertEqual(info["stage"], "INITIAL", "未达标时不应提前进入新阶段")

        gate = info["intimacy_gate"]
        self.assertIsNotNone(gate)
        self.assertEqual(gate["required"], 50)
        self.assertEqual(gate["current"], 40)
        self.assertEqual(gate["gap"], 10)
        self.assertFalse(gate["met"])
        self.assertEqual(gate["to_stage"], "深化期")

    def test_block_persists_until_gate_met(self):
        """阻断期间 _previous_stage 不推进，重复查询仍是过渡中"""
        state = _state(70, 40)
        DynamicWeightManager.get_stage_info(state)
        self.assertIsNone(state._previous_stage, "阻断时不应落盘新基线")

        info2 = DynamicWeightManager.get_stage_info(state)
        self.assertTrue(info2["is_transitioning"], "第二次查询阻断状态丢了")
        self.assertEqual(info2["intimacy_gate"]["gap"], 10)

    def test_gate_met_completes_transition(self):
        """亲密度达标 → 正常一次性过渡，之后稳定"""
        state = _state(70, 50)
        info = DynamicWeightManager.get_stage_info(state)

        self.assertTrue(info["is_transitioning"])
        self.assertEqual(info["stage"], "DEEPENING")
        self.assertIsNone(info["intimacy_gate"], "达标时不应带门槛信息")
        self.assertEqual(state._previous_stage, "DEEPENING", "达标后应落盘新基线")

        info2 = DynamicWeightManager.get_stage_info(state)
        self.assertFalse(info2["is_transitioning"], "过渡不应反复触发")

    def test_gate_met_after_intimacy_grows(self):
        """被门槛卡住后，亲密度涨到达标线即放行（冻结结束）"""
        state = _state(70, 40)
        DynamicWeightManager.get_stage_info(state)
        self.assertTrue(state._previous_stage is None)

        state.intimacy = 52  # 模拟里程碑/常规变化把亲密度推过门槛
        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage"], "DEEPENING")
        self.assertTrue(info["is_transitioning"])

    def test_transition_boost_uses_blocked_stage_config(self):
        """增益系数必须按「被挡住的阶段」取，不能按压回的当前阶段

        门槛阻断时 target_stage 被压回当前阶段，若
        apply_transition_benefits 也用它取 intimacy_boost_factor，
        「显示的还差多少」与「实际生效的倍数」就是两套配置。
        """
        state = _state(70, 40)
        plugin = _make_plugin()
        out = plugin.weight_manager.apply_transition_benefits(
            state, {"intimacy": 2})

        # 深化期(intimacy_boost_factor=3.6) 的系数，而非初识期(4.0)
        self.assertEqual(out["intimacy"], int(2 * 3.6),
                         "增益系数应按被挡住的深化期配置算")

    def test_zero_pct_disables_gate(self):
        """关闭分支：阈值 0 → 永不门槛（旧行为）"""
        DynamicWeightManager.configure(transition_intimacy_pct=0)
        # favor 79 / intimacy 0：复合评分 55.3 单靠好感度即可升级
        state = _state(79, 0)
        info = DynamicWeightManager.get_stage_info(state)

        self.assertEqual(info["stage"], "DEEPENING")
        self.assertTrue(info["is_transitioning"])
        self.assertIsNone(info["intimacy_gate"])
        self.assertFalse(DynamicWeightManager.is_favor_frozen(state))

    def test_required_follows_intimacy_max_config(self):
        """门槛占的是「最大亲密度」的百分比，上限改了口径必须跟着变"""
        EmotionConstants.configure(intimacy_min=0, intimacy_max=200)
        DynamicWeightManager.configure(transition_intimacy_pct=50)

        gate = DynamicWeightManager.get_intimacy_gate(_state(70, 99))
        self.assertEqual(gate["required"], 100)
        self.assertEqual(gate["gap"], 1)
        self.assertFalse(gate["met"])

        gate_ok = DynamicWeightManager.get_intimacy_gate(_state(70, 100))
        self.assertTrue(gate_ok["met"])

    def test_negative_favor_not_gated(self):
        """负好感不走阶段门禁"""
        state = _state(-50, 0)
        info = DynamicWeightManager.get_stage_info(state)
        self.assertFalse(info["is_transitioning"])
        self.assertFalse(info["intimacy_gate_blocked"])
        self.assertFalse(DynamicWeightManager.is_favor_frozen(state))

    def test_no_upgrade_no_gate(self):
        """分数不够升级时根本不咨询门槛"""
        state = _state(20, 0)
        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage"], "INITIAL")
        self.assertFalse(info["is_transitioning"])
        self.assertIsNone(info["intimacy_gate"])


class TestFavorFreeze(_GateTestCase):
    """未达标期间好感度冻结（亲密度照常变化）"""

    def test_is_favor_frozen_states(self):
        self.assertTrue(DynamicWeightManager.is_favor_frozen(_state(70, 40)))
        self.assertFalse(DynamicWeightManager.is_favor_frozen(_state(70, 55)))
        self.assertFalse(DynamicWeightManager.is_favor_frozen(_state(20, 0)))

    def test_apply_updates_freezes_favor_keeps_intimacy(self):
        plugin = _make_plugin(
            transition_intimacy_pct=50,
            intimacy_first_deep_bonus=0,
            intimacy_streak_bonus=0,
        )
        state = _state(70, 40)  # 门槛阻断中
        updates = {"favor": 3, "intimacy": 1, "joy": 1, "source": "llm_analysis",
                   "llm_available": True}

        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)

        self.assertEqual(state.favor, 70, "未达标期间好感度不应变化")
        self.assertGreater(state.intimacy, 40, "亲密度的变化不应被冻结")
        # 调用方 dict 不被污染（之后还要参与意义计算/全局心情）
        self.assertEqual(updates["favor"], 3)

    def test_apply_updates_not_frozen_when_gate_met(self):
        plugin = _make_plugin(
            transition_intimacy_pct=50,
            intimacy_first_deep_bonus=0,
            intimacy_streak_bonus=0,
        )
        state = _state(70, 60)
        updates = {"favor": 3, "intimacy": 1, "joy": 1, "source": "llm_analysis",
                   "llm_available": True}

        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)

        self.assertEqual(state.favor, 73)
        self.assertEqual(state.intimacy, 61)


class TestIntimacyMilestones(_GateTestCase):
    """里程碑：首次深度交流 + 连续多日互动"""

    def _plugin(self, **cfg):
        base = dict(intimacy_first_deep_bonus=3,
                    intimacy_streak_days=3,
                    intimacy_streak_bonus=1)
        base.update(cfg)
        return _make_plugin(**base)

    def test_first_deep_conversation_bonus_once(self):
        plugin = self._plugin()
        state = _state(0, 0)
        updates = {"favor": 3, "intimacy": 1, "joy": 2, "trust": 2}

        plugin._apply_intimacy_milestones(state, updates)  # 意义分 8 ≥ 5
        self.assertEqual(state.intimacy, 3)
        self.assertTrue(state.stats.deep_conversation_achieved)

        plugin._apply_intimacy_milestones(state, updates)  # 一次性，不重复
        self.assertEqual(state.intimacy, 3, "首次深度交流奖励不应重复发放")

    def test_small_talk_is_not_deep(self):
        plugin = self._plugin()
        state = _state(0, 0)
        plugin._apply_intimacy_milestones(state, {"favor": 1})  # 意义分 1
        self.assertEqual(state.intimacy, 0)
        self.assertFalse(state.stats.deep_conversation_achieved)

    def test_streak_bonus_requires_threshold_days(self):
        plugin = self._plugin()

        # 中断（3 天前）→ 连击重置为 1，不够 3 天
        state = _state(0, 0)
        state.stats.last_active_date = _days_ago(3)
        state.stats.interaction_streak = 5
        plugin._apply_intimacy_milestones(state, {"favor": 0})
        self.assertEqual(state.stats.interaction_streak, 1)
        self.assertEqual(state.intimacy, 0, "连击 1 天不够阈值，不应发奖")

        # 差一天（2 天前）→ 同样重置为 1，仍不够
        state = _state(0, 0)
        state.stats.last_active_date = _days_ago(2)
        state.stats.interaction_streak = 5
        plugin._apply_intimacy_milestones(state, {"favor": 0})
        self.assertEqual(state.stats.interaction_streak, 1)
        self.assertEqual(state.intimacy, 0, "连击不够阈值，不应发奖")

        # 昨天互动过、连击 2 → 今天第 3 天，达到阈值
        state = _state(0, 0)
        state.stats.last_active_date = _days_ago(1)
        state.stats.interaction_streak = 2
        plugin._apply_intimacy_milestones(state, {"favor": 0})
        self.assertEqual(state.stats.interaction_streak, 3)
        self.assertEqual(state.intimacy, 1, "连续 3 天应获得 +1 加成")

    def test_same_day_counts_once(self):
        plugin = self._plugin()
        state = _state(0, 0)
        state.stats.last_active_date = _days_ago(0)
        state.stats.interaction_streak = 3

        streak, is_new = plugin._advance_interaction_streak(state)
        self.assertFalse(is_new)
        self.assertEqual(streak, 3)

        plugin._apply_intimacy_milestones(state, {"favor": 0})
        self.assertEqual(state.intimacy, 0, "同一天多次互动不重复发奖")

    def test_streak_days_two_config(self):
        plugin = self._plugin(intimacy_streak_days=2)
        state = _state(0, 0)
        state.stats.last_active_date = _days_ago(1)
        state.stats.interaction_streak = 1

        plugin._apply_intimacy_milestones(state, {"favor": 0})
        self.assertEqual(state.intimacy, 1, "阈值配 2 天时连击 2 天即发放")

    def test_zero_bonus_disables_milestones(self):
        plugin = self._plugin(intimacy_first_deep_bonus=0, intimacy_streak_bonus=0)
        state = _state(0, 0)
        state.stats.last_active_date = _days_ago(1)
        state.stats.interaction_streak = 5

        plugin._apply_intimacy_milestones(
            state, {"favor": 3, "intimacy": 1, "joy": 2, "trust": 2}
        )
        self.assertEqual(state.intimacy, 0)
        self.assertFalse(state.stats.deep_conversation_achieved)

    def test_bonus_clamped_at_intimacy_max(self):
        plugin = self._plugin()
        state = _state(0, 100)
        plugin._apply_intimacy_milestones(
            state, {"favor": 3, "intimacy": 1, "joy": 2, "trust": 2}
        )
        self.assertEqual(state.intimacy, 100, "加成不得超出亲密度上限")

    def test_milestone_fires_through_apply_expert_updates(self):
        """端到端：常规更新 + 里程碑一起走 _apply_expert_updates"""
        plugin = _make_plugin()
        state = _state(0, 0)
        updates = {"favor": 3, "intimacy": 1, "joy": 2, "trust": 2,
                   "source": "llm_analysis", "llm_available": True}

        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)

        # 常规 +1，首次深度交流 +3
        self.assertEqual(state.intimacy, 4)
        self.assertTrue(state.stats.deep_conversation_achieved)

    def test_new_stats_fields_roundtrip(self):
        """里程碑字段必须落盘，否则重启后重复发奖"""
        state = _state(0, 0)
        state.stats.deep_conversation_achieved = True
        state.stats.interaction_streak = 4
        state.stats.last_active_date = "2026-09-19"

        reloaded = EnhancedEmotionalState.from_dict(state.to_dict())

        self.assertTrue(reloaded.stats.deep_conversation_achieved)
        self.assertEqual(reloaded.stats.interaction_streak, 4)
        self.assertEqual(reloaded.stats.last_active_date, "2026-09-19")


class TestGateDisplay(_GateTestCase):
    """/好感度 命令：过渡期展示亲密度与差值"""

    def test_basic_shows_gap_when_blocked(self):
        plugin = _make_plugin(transition_intimacy_pct=50, global_privacy_level=1)
        state = _state(70, 40)
        text = plugin._format_emotional_state(state)

        self.assertIn("亲密度 40/50", text)
        self.assertIn("还差 10 点", text)
        self.assertIn("未达标期间好感度不变化", text)

    def test_detailed_shows_gap_when_blocked(self):
        plugin = _make_plugin(transition_intimacy_pct=50, global_privacy_level=2)
        state = _state(70, 40)
        text = plugin._format_emotional_state(state)

        self.assertIn("亲密度未达标", text)
        self.assertIn("亲密度: 40/50（还差 10 点）", text)
        self.assertIn("未达标期间好感度不会变化", text)

    def test_no_gate_line_when_not_transitioning(self):
        plugin = _make_plugin(transition_intimacy_pct=50, global_privacy_level=1)
        text = plugin._format_emotional_state(_state(20, 10))
        self.assertNotIn("未达标", text)

    def test_advice_mentions_gate(self):
        state = _state(70, 40)
        advice = DynamicWeightManager.get_stage_progression_advice(state)
        self.assertIn("亲密度还未达标", advice)
        self.assertIn("40/50", advice)
        self.assertIn("还差 10 点", advice)
        self.assertIn("好感度不会变化", advice)


class TestConfigThreePlaceSync(unittest.TestCase):
    """新配置字段必须三处同步（config.py / schema_validator.py / _conf_schema.json），
    且在 main.py / config_manager.py 两处注入，否则又是死配置"""

    def _schema(self):
        import json
        root = Path(__file__).resolve().parent.parent
        return json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))

    def test_schema_defaults_match_pydantic(self):
        from emotionai_pro.config import PluginConfig
        cfg = PluginConfig()
        schema = self._schema()
        for field in ("transition_intimacy_pct", "intimacy_first_deep_bonus",
                      "intimacy_streak_days", "intimacy_streak_bonus"):
            node = schema[field]
            self.assertEqual(node["default"], getattr(cfg, field),
                             f"{field} 的 schema 默认值与 PluginConfig 不一致")
            self.assertIn("slider", node, f"{field} 缺少 slider，无法图形化配置")
            slider = node["slider"]
            self.assertLessEqual(slider["min"], node["default"])
            self.assertGreaterEqual(slider["max"], node["default"])

    def test_schema_validator_knows_new_fields(self):
        from emotionai_pro.schema_validator import ConfigValidator
        props = ConfigValidator.CONFIG_SCHEMA["properties"]
        self.assertEqual(props["transition_intimacy_pct"],
                         {"type": "integer", "minimum": 0, "maximum": 100})
        self.assertEqual(props["intimacy_streak_days"],
                         {"type": "integer", "minimum": 2, "maximum": 30})
        # create_default_config 会写文件，不便直接调；改为源码级断言默认值存在
        root = Path(__file__).resolve().parent.parent
        src = (root / "schema_validator.py").read_text(encoding="utf-8")
        for field, default in (("transition_intimacy_pct", 50),
                               ("intimacy_first_deep_bonus", 3),
                               ("intimacy_streak_days", 3),
                               ("intimacy_streak_bonus", 1)):
            self.assertIn(f'"{field}": {default},', src,
                          f"create_default_config 漏了 {field}")

    def test_base_mapping_whitelist(self):
        root = Path(__file__).resolve().parent.parent
        src = (root / "main.py").read_text(encoding="utf-8")
        for field in ("transition_intimacy_pct", "intimacy_first_deep_bonus",
                      "intimacy_streak_days", "intimacy_streak_bonus"):
            self.assertIn(f'"{field}": "{field}"', src,
                          f"base_mapping 白名单漏了 {field}")

    def test_two_injection_points(self):
        root = Path(__file__).resolve().parent.parent
        main_src = (root / "main.py").read_text(encoding="utf-8")
        cm_src = (root / "config_manager.py").read_text(encoding="utf-8")
        self.assertIn(
            "DynamicWeightManager.configure(\n"
            "            transition_intimacy_pct=self.config.transition_intimacy_pct,\n"
            "        )",
            main_src, "main.py 启动时未注入门槛百分比",
        )
        self.assertIn(
            "DynamicWeightManager.configure(\n"
            "                transition_intimacy_pct=config.transition_intimacy_pct,\n"
            "            )",
            cm_src, "config_manager 热重载时未注入门槛百分比",
        )

    def test_configure_rejects_out_of_range(self):
        DynamicWeightManager.configure(transition_intimacy_pct=150)
        self.assertEqual(DynamicWeightManager.TRANSITION_INTIMACY_PCT, 50)
        DynamicWeightManager.configure(transition_intimacy_pct="abc")
        self.assertEqual(DynamicWeightManager.TRANSITION_INTIMACY_PCT, 50)
        DynamicWeightManager.configure(transition_intimacy_pct=True)
        self.assertEqual(DynamicWeightManager.TRANSITION_INTIMACY_PCT, 50)
        DynamicWeightManager.configure(transition_intimacy_pct=70)
        self.assertEqual(DynamicWeightManager.TRANSITION_INTIMACY_PCT, 70)
        DynamicWeightManager.reset()


if __name__ == "__main__":
    unittest.main()
