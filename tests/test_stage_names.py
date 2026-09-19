# tests/test_stage_names.py
"""关系阶段名自定义的回归测试（v4.0.21）

覆盖三类风险：
  A. 配置 → 生效名 的链路（中文键 / 英文键 / 坏值逐项忽略 / 重名跳过）
  B. **旧存档不被重置**：改名前存的中文名必须归一化成新名，
     而不是被 models._validate_relationship_stage 判无效后按 favor 修复
  C. 全部显示出口（面板阶段名 / 下一阶段名 / 负向三档 / 建议文案 /
     managers 的初始态判断 / 配置热重载）都跟随当前生效名
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro import stage_names  # noqa: E402
from emotionai_pro.stage_names import (  # noqa: E402
    DEFAULT_STAGE_NAMES, configure_stage_names, get_stage_name,
    normalize_stage_name, reset_stage_names, valid_stage_names,
)
from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402
from emotionai_pro.relationship_manager import DynamicWeightManager  # noqa: E402


class _StageNamesTestCase(unittest.TestCase):
    """每个用例前后都Reset模块级生效名，避免污染同进程其它测试"""

    def setUp(self):
        reset_stage_names()

    def tearDown(self):
        reset_stage_names()


class TestDefaultNames(_StageNamesTestCase):
    """不配置时全部用出厂默认名"""

    def test_all_seven_keys_have_defaults(self):
        self.assertEqual(len(DEFAULT_STAGE_NAMES), 7)
        for key, name in DEFAULT_STAGE_NAMES.items():
            self.assertEqual(get_stage_name(key), name)

    def test_valid_names_are_the_seven_defaults(self):
        self.assertEqual(set(valid_stage_names()), set(DEFAULT_STAGE_NAMES.values()))

    def test_unknown_key_falls_back_safely(self):
        self.assertEqual(get_stage_name("NOT_A_STAGE"), "NOT_A_STAGE")


class TestConfigureStageNames(_StageNamesTestCase):
    """配置应用：中文键 / 英文键 / 坏值逐项忽略 / 重名跳过"""

    def test_chinese_keys_work(self):
        applied = configure_stage_names({"初识期": "初见", "共生期": "羁绊"})
        self.assertEqual(len(applied), 2)
        self.assertEqual(get_stage_name("INITIAL"), "初见")
        self.assertEqual(get_stage_name("SYMBIOSIS"), "羁绊")
        # 未配置的阶段保持默认
        self.assertEqual(get_stage_name("DEEPENING"), "深化期")

    def test_english_keys_also_work(self):
        configure_stage_names({"INITIAL": "初见", "COLD": "冰点"})
        self.assertEqual(get_stage_name("INITIAL"), "初见")
        self.assertEqual(get_stage_name("COLD"), "冰点")

    def test_negative_stage_keys(self):
        configure_stage_names({"冷淡期": "降温", "反感期": "嫌恶", "敌对期": "决裂"})
        self.assertEqual(get_stage_name("COLD"), "降温")
        self.assertEqual(get_stage_name("AVERSION"), "嫌恶")
        self.assertEqual(get_stage_name("HOSTILITY"), "决裂")

    def test_bad_values_are_ignored_individually(self):
        applied = configure_stage_names({
            "初识期": "",                 # 空串
            "深化期": "   ",              # 纯空白
            "承诺期": "x" * 21,           # 超长
            "共生期": 123,                # 非字符串
            "冷淡期": None,               # None
            "不存在的阶段": "随便",       # 未知键
            "反感期": "疏远",             # 唯一合法项
        })
        self.assertEqual(applied, ["反感期→疏远"])
        self.assertEqual(get_stage_name("INITIAL"), "初识期")
        self.assertEqual(get_stage_name("AVERSION"), "疏远")

    def test_duplicate_names_are_skipped(self):
        """两个阶段不能同名：后者被跳过，前者（含默认名）保留"""
        applied = configure_stage_names({"初识期": "同一個名字", "深化期": "同一個名字"})
        self.assertEqual(applied, ["初识期→同一個名字"])
        self.assertEqual(get_stage_name("DEEPENING"), "深化期")

    def test_duplicate_with_default_name_is_skipped(self):
        """把某阶段改成另一个阶段的默认名，同样判重名"""
        applied = configure_stage_names({"INITIAL": "深化期"})
        self.assertEqual(applied, [])
        self.assertEqual(get_stage_name("INITIAL"), "初识期")

    def test_non_dict_config_is_safe(self):
        for bad in (None, "x", 123, ["初识期"], object()):
            self.assertEqual(configure_stage_names(bad), [])
            self.assertEqual(get_stage_name("INITIAL"), "初识期")

    def test_reset_restores_defaults(self):
        configure_stage_names({"初识期": "初见"})
        reset_stage_names()
        self.assertEqual(get_stage_name("INITIAL"), "初识期")

    def test_config_default_factory_is_empty(self):
        """PluginConfig 默认不改任何阶段名"""
        self.assertEqual(PluginConfig().stage_names, {})


class TestNormalizeStageName(_StageNamesTestCase):
    """normalize_stage_name：旧默认名 → 当前生效名"""

    def test_current_name_passes_through(self):
        configure_stage_names({"初识期": "初见"})
        self.assertEqual(normalize_stage_name("初见"), "初见")

    def test_legacy_name_maps_to_current(self):
        """改名后，存档里的旧默认名要映射到新名（核心防重置逻辑）"""
        configure_stage_names({"初识期": "初见"})
        self.assertEqual(normalize_stage_name("初识期"), "初见")

    def test_unconfigured_legacy_name_is_identity(self):
        self.assertEqual(normalize_stage_name("初识期"), "初识期")
        self.assertEqual(normalize_stage_name("敌对期"), "敌对期")

    def test_garbage_returns_none(self):
        for bad in (None, "", "   ", 123, "乱七八糟", "INITIAL"):
            self.assertIsNone(normalize_stage_name(bad), f"{bad!r} 不应被当成有效阶段名")


class TestOldSavesSurviveRename(_StageNamesTestCase):
    """改名后旧存档必须跟随新名，而不是被修复重置"""

    def test_old_initial_stage_name_is_renamed_not_reset(self):
        configure_stage_names({"初识期": "初见"})
        state = EnhancedEmotionalState(user_key="u", relationship_stage="初识期")
        self.assertEqual(
            state.relationship_stage, "初见",
            "旧存档的「初识期」被当成无效值重置了（改名功能会毁存档）",
        )

    def test_new_name_survives_reload(self):
        configure_stage_names({"共生期": "羁绊"})
        state = EnhancedEmotionalState(user_key="u", relationship_stage="羁绊")
        reloaded = EnhancedEmotionalState.from_dict(state.to_dict())
        self.assertEqual(reloaded.relationship_stage, "羁绊")

    def test_invalid_stage_is_repaired_to_current_name(self):
        configure_stage_names({"初识期": "初见"})
        state = EnhancedEmotionalState(user_key="u", relationship_stage="乱七八糟")
        self.assertEqual(state.relationship_stage, "初见")

    def test_negative_favor_repair_uses_current_names(self):
        configure_stage_names({"冷淡期": "降温"})
        state = EnhancedEmotionalState(
            user_key="u", favor=-10, relationship_stage="乱七八糟"
        )
        self.assertEqual(state.relationship_stage, "降温")

    def test_old_negative_name_is_renamed(self):
        configure_stage_names({"敌对期": "决裂"})
        state = EnhancedEmotionalState(
            user_key="u", favor=-99, relationship_stage="敌对期"
        )
        self.assertEqual(state.relationship_stage, "决裂")


class TestDisplaySurfacesFollowConfig(_StageNamesTestCase):
    """面板 / 建议 / 负向三档全部跟随生效名"""

    @staticmethod
    def _state(favor, intimacy):
        return EnhancedEmotionalState(user_key="u", favor=favor, intimacy=intimacy)

    def test_stage_info_uses_custom_name(self):
        configure_stage_names({"共生期": "羁绊"})
        info = DynamicWeightManager.get_stage_info(self._state(100, 100))
        self.assertEqual(info["stage"], "SYMBIOSIS")
        self.assertEqual(info["stage_name"], "羁绊")

    def test_next_stage_name_uses_custom_name(self):
        configure_stage_names({"承诺期": "约定"})
        # DEEPENING 区间：composite ≈ 55+
        info = DynamicWeightManager.get_stage_info(self._state(60, 60))
        self.assertEqual(info["stage"], "DEEPENING")
        self.assertEqual(info["next_stage_name"], "约定")

    def test_negative_stage_info_uses_custom_names(self):
        configure_stage_names({
            "冷淡期": "降温", "反感期": "嫌恶", "敌对期": "决裂",
        })
        self.assertEqual(
            DynamicWeightManager.get_stage_info(self._state(-10, 0))["stage_name"], "降温")
        self.assertEqual(
            DynamicWeightManager.get_stage_info(self._state(-50, 0))["stage_name"], "嫌恶")
        self.assertEqual(
            DynamicWeightManager.get_stage_info(self._state(-99, 0))["stage_name"], "决裂")

    def test_advice_uses_custom_names(self):
        configure_stage_names({"冷淡期": "降温", "反感期": "嫌恶", "敌对期": "决裂"})
        self.assertTrue(
            DynamicWeightManager.get_stage_progression_advice(self._state(-10, 0))
            .startswith("降温："))
        self.assertTrue(
            DynamicWeightManager.get_stage_progression_advice(self._state(-50, 0))
            .startswith("嫌恶："))
        self.assertTrue(
            DynamicWeightManager.get_stage_progression_advice(self._state(-99, 0))
            .startswith("决裂："))

    def test_positive_advice_prefix_uses_custom_name(self):
        configure_stage_names({"初识期": "初见"})
        advice = DynamicWeightManager.get_stage_progression_advice(self._state(0, 0))
        self.assertTrue(advice.startswith("初见："), f"建议文案未跟随改名: {advice[:40]}")

    def test_transition_text_uses_custom_name(self):
        """过渡中/过渡完成文案里的阶段名同样来自配置"""
        configure_stage_names({"共生期": "羁绊"})
        state = self._state(100, 100)
        state._previous_stage = "INITIAL"   # 伪造一次刚跃迁
        info = DynamicWeightManager.get_stage_info(state)
        self.assertTrue(info["is_transitioning"])
        advice = DynamicWeightManager.get_stage_progression_advice(state)
        self.assertIn("羁绊", advice)


class TestManagersInitialCheckFollowsConfig(_StageNamesTestCase):
    """managers._is_initial_state_user 的「初识期」判断必须跟随当前生效名"""

    def _mk_manager_with_state(self, state):
        from emotionai_pro.managers import UserStateManager

        mgr = UserStateManager.__new__(UserStateManager)

        class _Cache:
            async def get(self, key):
                return None

        class _Repo:
            async def get_user_state(self, user_key):
                return state

        mgr.cache = _Cache()
        mgr.repository = _Repo()
        mgr.stats = {"errors": 0}
        return mgr

    def test_renamed_initial_stage_still_matches(self):
        """改名后，处在「新名字」的初始用户必须仍被判为初始状态"""
        import asyncio

        configure_stage_names({"初识期": "初见"})
        state = EnhancedEmotionalState(user_key="u")
        self.assertEqual(state.relationship_stage, "初见")
        mgr = self._mk_manager_with_state(state)
        self.assertTrue(asyncio.run(mgr._is_initial_state_user("u")))

    def test_default_name_still_matches_without_config(self):
        import asyncio

        state = EnhancedEmotionalState(user_key="u")
        mgr = self._mk_manager_with_state(state)
        self.assertTrue(asyncio.run(mgr._is_initial_state_user("u")))


class TestHotReloadAppliesStageNames(_StageNamesTestCase):
    """配置热重载必须同步阶段名（与数值边界同一注入点）"""

    def test_apply_numeric_bounds_also_applies_stage_names(self):
        import asyncio

        from emotionai_pro.config_manager import ConfigManager

        cm = ConfigManager.__new__(ConfigManager)
        cfg = PluginConfig(stage_names={"初识期": "初见"})
        cm._apply_numeric_bounds(cfg)
        self.assertEqual(get_stage_name("INITIAL"), "初见")

    def test_main_applies_stage_names_on_startup(self):
        """main.py 启动时也要应用（与 EmotionConstants.configure 同一处）"""
        root = Path(__file__).resolve().parent.parent
        src = (root / "main.py").read_text(encoding="utf-8")
        self.assertIn("configure_stage_names(self.config.stage_names)", src)
        self.assertIn('"stage_names": "stage_names"', src,
                      "base_mapping 白名单漏了 stage_names")

    def test_schema_matches_default_stage_names(self):
        """_conf_schema.json 的 7 个子项默认值必须与 DEFAULT_STAGE_NAMES 一致"""
        import json

        root = Path(__file__).resolve().parent.parent
        schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
        node = schema.get("stage_names")
        self.assertIsNotNone(node, "schema 缺少 stage_names")
        self.assertEqual(node.get("type"), "object")
        items = node.get("items", {})
        self.assertEqual(len(items), 7)
        for name in DEFAULT_STAGE_NAMES.values():
            self.assertIn(name, items, f"schema 子项缺少 {name}")
            self.assertEqual(items[name]["default"], name)
            self.assertEqual(items[name]["type"], "string")


class TestSchemaValidatorAcceptsStageNames(_StageNamesTestCase):
    """ConfigValidator（jsonschema 副本）必须接受新字段"""

    def _base(self):
        """一份满足 schema required 的最小合法配置"""
        return {
            "session_based": False,
            "favour_min": -100, "favour_max": 100,
            "intimacy_min": 0, "intimacy_max": 100,
            "change_min": -10, "change_max": 5,
            "admin_qq_list": [], "plugin_priority": 100000,
        }

    def test_stage_names_object_is_valid(self):
        from emotionai_pro.schema_validator import ConfigValidator

        import jsonschema

        cfg = self._base()
        cfg["stage_names"] = {"初识期": "初见"}
        jsonschema.validate(cfg, ConfigValidator.CONFIG_SCHEMA)

        cfg2 = self._base()
        cfg2["stage_names"] = {"INITIAL": "初见", "敌对期": "决裂"}
        jsonschema.validate(cfg2, ConfigValidator.CONFIG_SCHEMA)

    def test_intimacy_change_fields_are_valid(self):
        """v4.0.21 新增的两个数值字段也要过 schema 校验"""
        from emotionai_pro.schema_validator import ConfigValidator

        import jsonschema

        cfg = self._base()
        cfg["intimacy_change_min"] = -3
        cfg["intimacy_change_max"] = 3
        jsonschema.validate(cfg, ConfigValidator.CONFIG_SCHEMA)


if __name__ == "__main__":
    unittest.main()
