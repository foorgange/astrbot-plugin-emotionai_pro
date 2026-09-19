# tests/test_stage_baseline_recovery.py
"""阶段过渡基线（_previous_stage）丢失后的恢复行为（v4.1.1）

线上实例（用户 2961113185）复现的 bug：
    favor=108 / intimacy=118，实际早已在承诺期，面板却显示
    「关注塔菲喵（初识期）+ 下一阶段：雏草姬（55+）+ 亲密度 118/120」。

根因链：
  1. `_previous_stage` 只在 get_stage_info（展示路径）里推进，消息流
     只读不写；被修复逻辑重建过的存档 / 老存档 / 没看过面板的用户，
     基线恒为 None；
  2. calculate_stage 旧实现 `state._previous_stage or "INITIAL"` 把它猜成
     初识期 → 复合分够共生期的老用户被打回初识期；
  3. v4.0.22 起门槛阻断期间**故意不推进基线**（保证过渡持续），于是这个
     错基线自愈不了，用户被最高阶段的门槛（120）永久卡住。

本文件锁死：
  A. 基线缺失/非法 → 按当前复合分归档到「分数对应阶段的下一级」
     （显示回到真正的上一阶段，门槛按目标阶段正常生效）；
  B. 低分新用户、有有效基线的老路径行为不变（零回归）；
  C. storage._try_repair_user_data 必须抢救这两个字段，且修复后
     get_user_state 返回修复态而不是 None（None 会让上层造 favor=0
     的默认状态并可能覆盖存档）。
"""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.constants import EmotionConstants  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402
from emotionai_pro.relationship_manager import (  # noqa: E402
    DynamicWeightManager,
)
from emotionai_pro.stage_names import (  # noqa: E402
    configure_stage_names, reset_stage_names,
)
from emotionai_pro.storage import UserStateRepository  # noqa: E402


def _state(favor: int, intimacy: int, key: str = "u") -> EnhancedEmotionalState:
    """无过渡基线的裸状态（模拟刚加载 / 被修复过的存档）"""
    return EnhancedEmotionalState(user_key=key, favor=favor, intimacy=intimacy)


class _RecoveryTestCase(unittest.TestCase):
    """每个用例前后还原进程级单例"""

    def setUp(self):
        DynamicWeightManager.reset()
        EmotionConstants.reset()
        reset_stage_names()

    def tearDown(self):
        DynamicWeightManager.reset()
        EmotionConstants.reset()
        reset_stage_names()


class TestMissingBaselineReconstructed(_RecoveryTestCase):
    """基线缺失时按分数归档，而不是猜 INITIAL"""

    def test_reported_case_shows_commitment_not_initial(self):
        """用户 2961113185 的实例：应显示承诺期，而不是初识期"""
        # 线上配置：亲密度上限 200、分档门槛 20/40/60、自定义阶段名
        EmotionConstants.configure(favour_min=-100, favour_max=200,
                              intimacy_min=-100, intimacy_max=200)
        configure_stage_names({
            "初识期": "关注塔菲喵", "深化期": "雏草姬",
            "承诺期": "永雏结晶", "共生期": "塔不灭",
        })

        state = _state(108, 118)
        info = DynamicWeightManager.get_stage_info(state)

        self.assertEqual(
            info["stage"], "COMMITMENT",
            "老用户被打回初识期了（基线缺失被猜成 INITIAL）")
        self.assertEqual(info["stage_name"], "永雏结晶")
        self.assertEqual(info["next_stage_name"], "塔不灭")
        self.assertTrue(info["is_transitioning"])

        gate = info["intimacy_gate"]
        self.assertIsNotNone(gate, "被共生期门槛卡住时必须带门槛信息")
        self.assertEqual(gate["required"], 120)
        self.assertEqual(gate["current"], 118)
        self.assertEqual(gate["gap"], 2)
        self.assertFalse(gate["met"])
        self.assertEqual(gate["to_stage_key"], "SYMBIOSIS")
        self.assertTrue(
            DynamicWeightManager.is_favor_frozen(state),
            "未达标期间好感度应冻结")

    def test_deepening_band_missing_baseline_shows_initial(self):
        """分数只在深化期档：上一级确实是初识期，不能过度修复"""
        state = _state(79, 0)
        info = DynamicWeightManager.get_stage_info(state)

        self.assertEqual(info["stage"], "INITIAL")
        gate = info["intimacy_gate"]
        self.assertEqual(gate["required"], 20)
        self.assertEqual(gate["to_stage_key"], "DEEPENING")

    def test_invalid_baseline_string_treated_as_missing(self):
        """脏字符串基线等同缺失（from_dict 会忽略，这里防御内存态）"""
        EmotionConstants.configure(favour_min=-100, favour_max=200,
                              intimacy_min=-100, intimacy_max=200)
        state = _state(108, 118)
        state._previous_stage = "NOT_A_STAGE"

        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage"], "COMMITMENT")

    def test_gate_met_self_heals_baseline(self):
        """亲密度涨过门槛 → 正常过渡并把恢复出的基线落盘（自愈）"""
        EmotionConstants.configure(favour_min=-100, favour_max=200,
                              intimacy_min=-100, intimacy_max=200)
        state = _state(108, 125)

        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage"], "SYMBIOSIS")
        self.assertIsNone(info["intimacy_gate"])
        self.assertEqual(state._previous_stage, "SYMBIOSIS",
                         "达标后应落盘新基线（自愈完成）")

        info2 = DynamicWeightManager.get_stage_info(state)
        self.assertFalse(info2["is_transitioning"], "过渡不应反复触发")

    def test_low_score_new_user_unaffected(self):
        """低分新用户：基线重建结果仍是初识期，行为零变化"""
        state = _state(20, 0)
        info = DynamicWeightManager.get_stage_info(state)

        self.assertEqual(info["stage"], "INITIAL")
        self.assertFalse(info["is_transitioning"])
        self.assertIsNone(info["intimacy_gate"])
        self.assertEqual(state._previous_stage, "INITIAL")


class TestValidBaselineUnchanged(_RecoveryTestCase):
    """有有效基线时走原有逻辑（v4.0.22/v4.0.23 语义不回归）"""

    def test_saved_baseline_is_respected(self):
        """已存档 COMMITMENT 的用户：仍按承诺期→共生期的档位卡门槛"""
        EmotionConstants.configure(favour_min=-100, favour_max=200,
                              intimacy_min=-100, intimacy_max=200)
        DynamicWeightManager.configure(
            transition_intimacy_pct=20,
            stage_intimacy_pcts={"DEEPENING": 20, "COMMITMENT": 40, "SYMBIOSIS": 60})

        state = _state(100, 72)
        state._previous_stage = "DEEPENING"
        state._previous_composite = 60.0

        info = DynamicWeightManager.get_stage_info(state)
        self.assertTrue(info["intimacy_gate_blocked"])
        self.assertEqual(info["stage"], "DEEPENING")
        self.assertEqual(info["intimacy_gate"]["required"], 80)
        self.assertEqual(info["intimacy_gate"]["gap"], 8)

    def test_stage_order_helper(self):
        """_raw_stage_by_score 的阈值梯形"""
        raw = DynamicWeightManager._raw_stage_by_score
        self.assertEqual(raw(10), "INITIAL")
        self.assertEqual(raw(54.9), "INITIAL")
        self.assertEqual(raw(55), "DEEPENING")
        self.assertEqual(raw(79.9), "DEEPENING")
        self.assertEqual(raw(80), "COMMITMENT")
        self.assertEqual(raw(94.9), "COMMITMENT")
        self.assertEqual(raw(95), "SYMBIOSIS")
        self.assertEqual(raw(200), "SYMBIOSIS")


class TestRepairSalvagesBaseline(unittest.TestCase):
    """数据修复必须连过渡基线一起抢救"""

    def setUp(self):
        EmotionConstants.reset()
        DynamicWeightManager.reset()
        self._tmp = TemporaryDirectory()
        self.repo = UserStateRepository(Path(self._tmp.name))

    def tearDown(self):
        EmotionConstants.reset()
        DynamicWeightManager.reset()
        self._tmp.cleanup()

    def _seed(self, user_key, state, **corrupt):
        payload = state.to_dict()
        payload.update(corrupt)
        self.repo._user_data = {user_key: payload}
        self.repo._loaded = True

    def _load(self, user_key):
        return asyncio.run(self.repo.get_user_state(user_key))

    def test_corrupted_record_keeps_previous_fields(self):
        """emotions 损坏 → 修复后 _previous_stage/_previous_composite 必须还在"""
        state = _state(88, 77)
        state._previous_stage = "COMMITMENT"
        state._previous_composite = 82.5
        self._seed("u1", state, emotions={"joy": 50, "不存在的字段": 1})

        self._load("u1")
        repaired = self.repo._user_data["u1"]

        self.assertEqual(repaired["favor"], 88)
        self.assertEqual(repaired["intimacy"], 77)
        self.assertEqual(repaired["_previous_stage"], "COMMITMENT",
                         "修复把过渡基线丢了（老用户会被打回初识期）")
        self.assertAlmostEqual(repaired["_previous_composite"], 82.5, places=3)

    def test_repair_returns_repaired_state_not_none(self):
        """修复后必须返回修复态；返回 None 会让上层造 favor=0 默认态"""
        state = _state(88, 77)
        state._previous_stage = "COMMITMENT"
        state._previous_composite = 82.5
        self._seed("u2", state, stats=None)

        loaded = self._load("u2")

        self.assertIsNotNone(loaded, "返回 None → 上层按新用户造 favor=0 状态")
        self.assertEqual(loaded.favor, 88)
        self.assertEqual(loaded._previous_stage, "COMMITMENT")

    def test_repair_does_not_write_dirty_baseline(self):
        """脏值基线不能被修进行存档（None 表示缺失，按分数重建）"""
        state = _state(50, 50)
        self._seed("u3", state, emotions={"joy": 50, "不存在的字段": 1},
                   _previous_stage="NOT_A_STAGE", _previous_composite="abc")

        self._load("u3")
        repaired = self.repo._user_data["u3"]

        self.assertIsNone(repaired["_previous_stage"],
                          "脏基线被写回存档了")
        self.assertEqual(repaired["_previous_composite"], 0.0)


if __name__ == "__main__":
    unittest.main()
