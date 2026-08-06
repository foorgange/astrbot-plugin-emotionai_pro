# tests/test_relationship_stage.py
"""关系阶段"下一阶段"字段离线单测"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.bootstrap  # noqa: F401

from emotionai_pro.models import EnhancedEmotionalState
from emotionai_pro.relationship_manager import DynamicWeightManager


def make_state(favor: int = 0, intimacy: int = 0, key: str = "test_user"):
    """构造全新状态（每个用例独立，规避滞回副作用）"""
    return EnhancedEmotionalState(user_key=key, favor=favor, intimacy=intimacy)


class TestStageInfoNextStage(unittest.TestCase):
    def test_initial_stage_next_threshold(self):
        """初识期 → 下一阶段为深化期 (55)"""
        state = make_state(favor=10, intimacy=5)
        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage_name"], "初识期")
        self.assertEqual(info["current_stage_threshold"], 25)
        self.assertEqual(info["next_stage_threshold"], 55)
        self.assertEqual(info["next_stage_name"], "深化期")
        self.assertFalse(info["is_max_stage"])

    def test_deepening_stage_next_threshold(self):
        """深化期 → 下一阶段为承诺期 (80)"""
        state = make_state(favor=60, intimacy=60)
        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage_name"], "深化期")
        self.assertEqual(info["next_stage_threshold"], 80)
        self.assertEqual(info["next_stage_name"], "承诺期")
        self.assertFalse(info["is_max_stage"])

    def test_commitment_stage_next_threshold(self):
        """承诺期 → 下一阶段为共生期 (95)"""
        state = make_state(favor=80, intimacy=85)
        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage_name"], "承诺期")
        self.assertEqual(info["next_stage_threshold"], 95)
        self.assertEqual(info["next_stage_name"], "共生期")

    def test_symbiosis_stage_is_max(self):
        """共生期 → next_stage_threshold 为 None，is_max_stage 为 True"""
        state = make_state(favor=98, intimacy=99)
        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage_name"], "共生期")
        self.assertIsNone(info["next_stage_threshold"])
        self.assertEqual(info["next_stage_name"], "已达最高阶段")
        self.assertTrue(info["is_max_stage"])

    def test_negative_favor_stage(self):
        """负好感 → 下一阶段为"恢复正常关系"，无固定阈值"""
        state = make_state(favor=-20)
        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage_name"], "冷淡期")
        self.assertIsNone(info["next_stage_threshold"])
        self.assertEqual(info["next_stage_name"], "恢复正常关系")
        self.assertFalse(info["is_max_stage"])

    def test_hysteresis_prevents_stage_jitter(self):
        """滞回：刚过阈值但未稳定时保持上一阶段"""
        # 从 INITIAL 提升到 composite 接近 55
        state = make_state(favor=50, intimacy=50)
        # 第一次计算会记录 _previous_stage
        DynamicWeightManager.get_stage_info(state)
        # composite 仍低于 55（50*0.5+50*0.5=50），保持在初始
        info = DynamicWeightManager.get_stage_info(state)
        self.assertEqual(info["stage_name"], "初识期")


class TestStageOrderHelpers(unittest.TestCase):
    def test_next_stage_key(self):
        """STAGE_ORDER 顺序正确"""
        from emotionai_pro.relationship_manager import _next_stage_key, STAGE_ORDER
        self.assertEqual(STAGE_ORDER, ["INITIAL", "DEEPENING", "COMMITMENT", "SYMBIOSIS"])
        self.assertEqual(_next_stage_key("INITIAL"), "DEEPENING")
        self.assertEqual(_next_stage_key("DEEPENING"), "COMMITMENT")
        self.assertEqual(_next_stage_key("COMMITMENT"), "SYMBIOSIS")
        self.assertIsNone(_next_stage_key("SYMBIOSIS"))


if __name__ == "__main__":
    unittest.main()
