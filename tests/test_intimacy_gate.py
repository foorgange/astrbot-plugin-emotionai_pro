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
import types
import asyncio
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
    """门槛：分数够升级、亲密度没达标 → 过渡卡住且持续

    v4.0.23 起门槛按目标阶段分档，深化期默认 20%（= 出厂默认上限 100 × 20%）。
    """

    def test_blocked_when_intimacy_below_threshold(self):
        # favor 79 / intimacy 0：复合评分 55.3 刚好够升深化期，但深化期门槛要 20
        state = _state(79, 0)
        info = DynamicWeightManager.get_stage_info(state)

        self.assertTrue(info["is_transitioning"], "亲密度未达标却放行了过渡")
        self.assertEqual(info["stage"], "INITIAL", "未达标时不应提前进入新阶段")

        gate = info["intimacy_gate"]
        self.assertIsNotNone(gate)
        self.assertEqual(gate["required"], 20)
        self.assertEqual(gate["current"], 0)
        self.assertEqual(gate["gap"], 20)
        self.assertFalse(gate["met"])
        self.assertEqual(gate["to_stage"], "深化期")

    def test_block_persists_until_gate_met(self):
        """阻断期间 _previous_stage 不推进，重复查询仍是过渡中"""
        state = _state(79, 0)
        DynamicWeightManager.get_stage_info(state)
        self.assertIsNone(state._previous_stage, "阻断时不应落盘新基线")

        info2 = DynamicWeightManager.get_stage_info(state)
        self.assertTrue(info2["is_transitioning"], "第二次查询阻断状态丢了")
        self.assertEqual(info2["intimacy_gate"]["gap"], 20)

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
        state = _state(79, 0)
        DynamicWeightManager.get_stage_info(state)
        self.assertTrue(state._previous_stage is None)

        state.intimacy = 35  # 模拟里程碑/常规变化把亲密度推过门槛
        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage"], "DEEPENING")
        self.assertTrue(info["is_transitioning"])

    def test_transition_boost_uses_blocked_stage_config(self):
        """增益系数必须按「被挡住的阶段」取，不能按压回的当前阶段

        门槛阻断时 target_stage 被压回当前阶段，若
        apply_transition_benefits 也用它取 intimacy_boost_factor，
        「显示的还差多少」与「实际生效的倍数」就是两套配置。
        """
        state = _state(79, 0)
        plugin = _make_plugin()
        out = plugin.weight_manager.apply_transition_benefits(
            state, {"intimacy": 2})

        # 深化期(intimacy_boost_factor=3.6) 的系数，而非初识期(4.0)
        self.assertEqual(out["intimacy"], int(2 * 3.6),
                         "增益系数应按被挡住的深化期配置算")

    def test_zero_pct_disables_gate(self):
        """关闭分支：统一阈值 0 且分档表清空 → 永不门槛（旧行为）"""
        DynamicWeightManager.configure(transition_intimacy_pct=0,
                                       stage_intimacy_pcts={})
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
        self.assertTrue(DynamicWeightManager.is_favor_frozen(_state(79, 0)))
        self.assertFalse(DynamicWeightManager.is_favor_frozen(_state(70, 60)))
        self.assertFalse(DynamicWeightManager.is_favor_frozen(_state(20, 0)))

    def test_apply_updates_freezes_favor_keeps_intimacy(self):
        plugin = _make_plugin(
            transition_intimacy_pct=50,
            intimacy_first_deep_bonus=0,
            intimacy_streak_bonus=0,
        )
        state = _state(79, 0)  # 深化期门槛（20%）阻断中
        updates = {"favor": 3, "intimacy": 1, "joy": 1, "source": "llm_analysis",
                   "llm_available": True}

        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)

        self.assertEqual(state.favor, 79, "未达标期间好感度不应变化")
        self.assertGreater(state.intimacy, 0, "亲密度的变化不应被冻结")
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
        state = _state(79, 0)
        text = plugin._format_emotional_state(state)

        self.assertIn("亲密度 0/20", text)
        self.assertIn("还差 20 点", text)
        self.assertIn("未达标期间好感度不变化", text)

    def test_detailed_shows_gap_when_blocked(self):
        plugin = _make_plugin(transition_intimacy_pct=50, global_privacy_level=2)
        state = _state(79, 0)
        text = plugin._format_emotional_state(state)

        self.assertIn("亲密度未达标", text)
        self.assertIn("亲密度：0/20（还差 20 点）", text)
        self.assertIn("未达标期间好感度不会变化", text)

    def test_no_gate_line_when_not_transitioning(self):
        plugin = _make_plugin(transition_intimacy_pct=50, global_privacy_level=1)
        text = plugin._format_emotional_state(_state(20, 10))
        self.assertNotIn("未达标", text)

    def test_advice_mentions_gate(self):
        state = _state(79, 0)
        advice = DynamicWeightManager.get_stage_progression_advice(state)
        self.assertIn("亲密度还未达标", advice)
        self.assertIn("0/20", advice)
        self.assertIn("还差 20 点", advice)
        self.assertIn("好感度不会变化", advice)


class TestStageIntimacyGates(_GateTestCase):
    """v4.0.23 分阶段门槛：每次过渡按**目标阶段**各自取百分比

    动机：统一门槛只有第一次过渡有牙齿。承诺期权重 favor 0.3 /
    intimacy 0.7，复合分上 80 本身就要求亲密度 ≥72（favor 100 时），
    40 的统一门槛在第二、三次过渡永远拦不到东西。
    """

    def _cfg(self, **kw):
        EmotionConstants.configure(favour_min=-100, favour_max=200,
                                  intimacy_min=-100, intimacy_max=kw.pop("imax", 200))
        DynamicWeightManager.configure(**kw)

    def test_each_stage_uses_own_threshold(self):
        """默认 20/40/60，随最大亲密度换算"""
        self._cfg(transition_intimacy_pct=20,
                  stage_intimacy_pcts={"DEEPENING": 20, "COMMITMENT": 40,
                                       "SYMBIOSIS": 60})
        blank = EnhancedEmotionalState(user_key="x")
        for stage, expected in (("DEEPENING", 40), ("COMMITMENT", 80),
                                ("SYMBIOSIS", 120)):
            gate = DynamicWeightManager.get_intimacy_gate(blank, stage)
            self.assertEqual(gate["required"], expected,
                             f"{stage} 应按自己的百分比算门槛")
            self.assertEqual(gate["target_stage"], stage)

    def test_second_transition_not_redundant(self):
        """回归：第二次过渡不再形同虚设

        favor 100 / intimacy 72：复合评分 80.4 已达承诺期阈值 80。
        统一门槛 40 时会直接放行（72 > 40，门槛无意义）；分档后
        承诺期要 80，必须卡住并提示还差 8 点。
        """
        self._cfg(transition_intimacy_pct=20,
                  stage_intimacy_pcts={"DEEPENING": 20, "COMMITMENT": 40,
                                       "SYMBIOSIS": 60})
        state = _state(100, 72)
        state._previous_stage = "DEEPENING"
        info = DynamicWeightManager.get_stage_info(state)

        self.assertTrue(info["intimacy_gate_blocked"],
                        "承诺期亲密度未达标却放行了过渡")
        self.assertEqual(info["stage"], "DEEPENING", "未达标不应提前进入承诺期")
        gate = info["intimacy_gate"]
        self.assertEqual(gate["required"], 80)
        self.assertEqual(gate["gap"], 8)
        self.assertEqual(gate["to_stage"], "承诺期")
        self.assertEqual(gate["to_stage_key"], "COMMITMENT")
        self.assertTrue(DynamicWeightManager.is_favor_frozen(state))

    def test_same_state_passes_under_unified_gate(self):
        """对照：清空分档表、只用统一门槛 20% 时，同一状态不再被挡
        （证明上面的阻断确实来自分档，而不是分数本身不够）"""
        self._cfg(transition_intimacy_pct=20, stage_intimacy_pcts={})
        state = _state(100, 72)
        state._previous_stage = "DEEPENING"
        info = DynamicWeightManager.get_stage_info(state)

        self.assertFalse(info["intimacy_gate_blocked"],
                         "统一门槛 40 低于 72，本应放行")
        self.assertEqual(info["stage"], "COMMITMENT")

    def test_third_transition_gate(self):
        """第三次过渡（承诺期 → 共生期）也按自己的档位生效"""
        self._cfg(transition_intimacy_pct=20,
                  stage_intimacy_pcts={"DEEPENING": 20, "COMMITMENT": 40,
                                       "SYMBIOSIS": 60})
        state = _state(100, 100)
        state._previous_stage = "COMMITMENT"
        info = DynamicWeightManager.get_stage_info(state)

        self.assertTrue(info["intimacy_gate_blocked"])
        self.assertEqual(info["stage"], "COMMITMENT")
        self.assertEqual(info["intimacy_gate"]["required"], 120)
        self.assertEqual(info["intimacy_gate"]["gap"], 20)

    def test_unlisted_stage_falls_back_to_global(self):
        """分档表里没有的阶段回退到统一门槛"""
        self._cfg(transition_intimacy_pct=50, stage_intimacy_pcts={})
        blank = EnhancedEmotionalState(user_key="x")
        # 上限 200 × 统一 50% = 100（证明取的是统一档，不是分档默认值）
        self.assertEqual(
            DynamicWeightManager.get_intimacy_gate(blank, "DEEPENING")["required"], 100)
        self.assertEqual(DynamicWeightManager.get_stage_intimacy_pct("DEEPENING"), 50)

    def test_stage_zero_disables_only_that_stage(self):
        """某一档填 0 只关闭该段，不影响其它档"""
        self._cfg(transition_intimacy_pct=20,
                  stage_intimacy_pcts={"DEEPENING": 0, "COMMITMENT": 40,
                                       "SYMBIOSIS": 60})
        # 第一段关闭：亲密 0 也能升深化期
        first = _state(79, 0)
        info = DynamicWeightManager.get_stage_info(first)
        self.assertEqual(info["stage"], "DEEPENING")
        self.assertFalse(info["intimacy_gate_blocked"])

        # 第二段仍然生效
        second = _state(100, 72)
        second._previous_stage = "DEEPENING"
        info2 = DynamicWeightManager.get_stage_info(second)
        self.assertTrue(info2["intimacy_gate_blocked"])
        self.assertEqual(info2["intimacy_gate"]["required"], 80)

    def test_required_follows_max_intimacy_per_stage(self):
        """最大亲密度改了，各档门槛按同一比例缩放"""
        self._cfg(transition_intimacy_pct=20, imax=400,
                  stage_intimacy_pcts={"DEEPENING": 20, "COMMITMENT": 40,
                                       "SYMBIOSIS": 60})
        blank = EnhancedEmotionalState(user_key="x")
        self.assertEqual(
            DynamicWeightManager.get_intimacy_gate(blank, "COMMITMENT")["required"], 160)
        self.assertEqual(
            DynamicWeightManager.get_intimacy_gate(blank, "SYMBIOSIS")["required"], 240)

    def test_configure_sanitizes_stage_table(self):
        """非法 key / 越界值 / bool 一律丢弃，合法值保留"""
        DynamicWeightManager.configure(
            stage_intimacy_pcts={"DEEPENING": 15, "COMMITMENT": 150,
                                 "SYMBIOSIS": True, "INITIAL": 30,
                                 "NOT_A_STAGE": 40, "COMMITMENT2": 10})
        self.assertEqual(DynamicWeightManager.STAGE_INTIMACY_PCT, {"DEEPENING": 15})

    def test_configure_stage_table_must_be_dict(self):
        DynamicWeightManager.configure(stage_intimacy_pcts="20")
        DynamicWeightManager.configure(stage_intimacy_pcts=True)
        self.assertEqual(DynamicWeightManager.STAGE_INTIMACY_PCT,
                         {"DEEPENING": 20, "COMMITMENT": 40, "SYMBIOSIS": 60})

    def test_reset_restores_stage_table(self):
        DynamicWeightManager.configure(stage_intimacy_pcts={"DEEPENING": 99})
        self.assertEqual(DynamicWeightManager.STAGE_INTIMACY_PCT, {"DEEPENING": 99})
        DynamicWeightManager.reset()
        self.assertEqual(DynamicWeightManager.STAGE_INTIMACY_PCT,
                         {"DEEPENING": 20, "COMMITMENT": 40, "SYMBIOSIS": 60})
        self.assertEqual(DynamicWeightManager.TRANSITION_INTIMACY_PCT, 50)

    def test_hot_reload_replaces_table(self):
        """热重载是整体替换：删掉某个 key 后该段回退统一门槛"""
        DynamicWeightManager.configure(
            stage_intimacy_pcts={"DEEPENING": 20, "COMMITMENT": 40})
        self.assertEqual(DynamicWeightManager.STAGE_INTIMACY_PCT["COMMITMENT"], 40)
        DynamicWeightManager.configure(stage_intimacy_pcts={"DEEPENING": 20})
        self.assertNotIn("COMMITMENT", DynamicWeightManager.STAGE_INTIMACY_PCT)
        self.assertEqual(DynamicWeightManager.get_stage_intimacy_pct("COMMITMENT"),
                         DynamicWeightManager.TRANSITION_INTIMACY_PCT)
        DynamicWeightManager.reset()


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

        # v4.0.23：分阶段门槛是 object 节点，逐档比对默认值
        gates_node = schema["stage_intimacy_gates"]
        self.assertEqual(gates_node["type"], "object")
        self.assertEqual(gates_node["items"].keys(), cfg.stage_intimacy_gates.keys())
        for stage_key, default in cfg.stage_intimacy_gates.items():
            item = gates_node["items"][stage_key]
            self.assertEqual(item["default"], default,
                             f"{stage_key} 档 schema 默认值与 PluginConfig 不一致")
            self.assertIn("slider", item, f"{stage_key} 档缺少 slider")
            self.assertLessEqual(item["slider"]["min"], default)
            self.assertGreaterEqual(item["slider"]["max"], default)

    def test_schema_validator_knows_new_fields(self):
        from emotionai_pro.schema_validator import ConfigValidator
        props = ConfigValidator.CONFIG_SCHEMA["properties"]
        self.assertEqual(props["transition_intimacy_pct"],
                         {"type": "integer", "minimum": 0, "maximum": 100})
        self.assertEqual(props["intimacy_streak_days"],
                         {"type": "integer", "minimum": 2, "maximum": 30})
        # v4.0.23：分阶段门槛必须进 schema（顶层 additionalProperties=False，
        # 漏了它 AstrBot 把该字段写进配置后整份校验会失败）
        self.assertEqual(props["stage_intimacy_gates"],
                         {"type": "object",
                          "additionalProperties": {"type": "integer",
                                                   "minimum": 0, "maximum": 100}})
        # create_default_config 会写文件，不便直接调；改为源码级断言默认值存在
        root = Path(__file__).resolve().parent.parent
        src = (root / "schema_validator.py").read_text(encoding="utf-8")
        for field, default in (("transition_intimacy_pct", 50),
                               ("intimacy_first_deep_bonus", 3),
                               ("intimacy_streak_days", 3),
                               ("intimacy_streak_bonus", 1)):
            self.assertIn(f'"{field}": {default},', src,
                          f"create_default_config 漏了 {field}")
        self.assertIn(
            '"stage_intimacy_gates": {"DEEPENING": 20, "COMMITMENT": 40, "SYMBIOSIS": 60},',
            src, "create_default_config 漏了分阶段门槛")

    def test_base_mapping_whitelist(self):
        root = Path(__file__).resolve().parent.parent
        src = (root / "main.py").read_text(encoding="utf-8")
        for field in ("transition_intimacy_pct", "intimacy_first_deep_bonus",
                      "intimacy_streak_days", "intimacy_streak_bonus",
                      "stage_intimacy_gates"):
            self.assertIn(f'"{field}": "{field}"', src,
                          f"base_mapping 白名单漏了 {field}")

    def test_two_injection_points(self):
        root = Path(__file__).resolve().parent.parent
        main_src = (root / "main.py").read_text(encoding="utf-8")
        cm_src = (root / "config_manager.py").read_text(encoding="utf-8")
        self.assertIn(
            "DynamicWeightManager.configure(\n"
            "            transition_intimacy_pct=self.config.transition_intimacy_pct,\n"
            "            stage_intimacy_pcts=self.config.stage_intimacy_gates,\n"
            "        )",
            main_src, "main.py 启动时未注入门槛（含分阶段表）",
        )
        self.assertIn(
            "DynamicWeightManager.configure(\n"
            "                transition_intimacy_pct=config.transition_intimacy_pct,\n"
            "                stage_intimacy_pcts=config.stage_intimacy_gates,\n"
            "            )",
            cm_src, "config_manager 热重载时未注入门槛（含分阶段表）",
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

    def test_stage_table_untouched_when_config_omits_it(self):
        """老配置没有 stage_intimacy_gates 字段 → 分档表保持原样"""
        DynamicWeightManager.configure(
            stage_intimacy_pcts={"DEEPENING": 15, "COMMITMENT": 25, "SYMBIOSIS": 35})
        DynamicWeightManager.configure(transition_intimacy_pct=30)
        self.assertEqual(DynamicWeightManager.STAGE_INTIMACY_PCT["COMMITMENT"], 25,
                         "只传统一门槛时不应清空分档表")
        DynamicWeightManager.reset()


class TestRelationshipStageDisplay(_GateTestCase):
    """/关系阶段 命令：门槛卡住时不能显示「正在适应新阶段」

    阻断时 intimacy_boost_active 恒为 True，旧文案因此落进适应分支，
    但过渡实际被挡住了，那句话有误导。
    """

    def _run_stage_command(self, state) -> str:
        from emotionai_pro.command_handlers import UserCommandHandler
        from astrbot.api.event import AstrMessageEvent

        handler = UserCommandHandler.__new__(UserCommandHandler)
        handler.plugin = types.SimpleNamespace(
            _get_user_key=lambda event: state.user_key
        )

        async def _get_state(key):
            return state

        handler.user_manager = types.SimpleNamespace(get_user_state=_get_state)
        handler.weight_manager = DynamicWeightManager()
        event = AstrMessageEvent(sender_id="123")

        chunks = []

        async def go():
            async for chunk in handler.show_relationship_stage(event):
                chunks.append(chunk)

        asyncio.run(go())
        return "\n".join(str(c) for c in chunks)

    def test_blocked_shows_gate_not_adapting(self):
        EmotionConstants.configure(
            favour_min=-100, favour_max=200,
            intimacy_min=-100, intimacy_max=200,
        )
        DynamicWeightManager.configure(
            transition_intimacy_pct=20,
            stage_intimacy_pcts={"DEEPENING": 20, "COMMITMENT": 40, "SYMBIOSIS": 60},
        )
        state = _state(100, 72)
        state._previous_stage = "DEEPENING"

        text = self._run_stage_command(state)

        self.assertIn("状态：亲密度未达标，过渡暂时卡住", text)
        self.assertIn("目标阶段：承诺期", text)
        self.assertIn("亲密度：72 / 80（还差 8 点）", text)
        self.assertNotIn("正在适应新阶段", text,
                         "卡住时不能显示正在适应新阶段（误导）")

    def test_not_blocked_hides_gate_line(self):
        EmotionConstants.configure(
            favour_min=-100, favour_max=200,
            intimacy_min=-100, intimacy_max=200,
        )
        DynamicWeightManager.configure(
            transition_intimacy_pct=20,
            stage_intimacy_pcts={"DEEPENING": 20, "COMMITMENT": 40, "SYMBIOSIS": 60},
        )
        state = _state(100, 85)
        state._previous_stage = "DEEPENING"

        text = self._run_stage_command(state)

        self.assertNotIn("亲密度未达标", text, "达标时不应出现门槛文案")


if __name__ == "__main__":
    unittest.main()
