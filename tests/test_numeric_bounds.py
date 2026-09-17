# tests/test_numeric_bounds.py
"""数值上限「配置生效」回归测试（v4.0.19）

背景（用户实际报障）：
    把插件配置里的「好感度最大值」设为 200，然后在 QQ 里执行
        /设置好感 3418451176 200
    插件回显「【成功】…已设置为 200」，可随后的状态面板却是
        好感度: 100 | 亲密度: 100
        复合评分: 100.0

根因链：
    1. 配置界面（_conf_schema.json）与 PluginConfig 都允许 favour_max=200，
       所以配置本身保存成功、命令回显也是 200；
    2. 但 constants.EmotionConstants 里**另有一份硬编码的** MAX_FAVOR=100；
    3. models.EnhancedEmotionalState._validate_core_values 用这份硬编码值钳制，
       200 被静默削成 100；
    4. 复合评分同步被压低：200*0.5+100*0.5=150 变成 100*0.5+100*0.5=100。

附带修掉的两个真缺陷（同一数据链路上）：
    · storage.py 的 except 元组漏了 AttributeError ——
      `EmotionalMetrics(**emotions_data)` 遇到旧版本存档会抛它，
      异常冒到 managers.get_user_state 的兜底后**整个状态被重置为 0**；
    · storage._try_repair_user_data 直接拿默认状态覆盖，
      名为"修复"实则清零；现改为逐字段抢救。
    · models.from_dict 不恢复 _previous_stage/_previous_composite ——
      每次从磁盘加载都被当成"刚从初识期跃迁"，面板恒显假的「过渡完成」。

本文件锁死这些行为，任何一层回退都会让测试失败。
"""
import sys
import os
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.constants import EmotionConstants  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402
from emotionai_pro.relationship_manager import DynamicWeightManager  # noqa: E402
from emotionai_pro.storage import UserStateRepository  # noqa: E402


def _make_state(user_key="u", favor=0, intimacy=0):
    state = EnhancedEmotionalState(user_key=user_key)
    state.favor = favor
    state.intimacy = intimacy
    return state


class TestBoundsFollowConfig(unittest.TestCase):
    """上限必须跟随配置，而不是写死在代码里"""

    def setUp(self):
        EmotionConstants.reset()

    tearDown = setUp

    def test_default_bounds_unchanged(self):
        """出厂默认仍是 -100/100、0/100（未注入配置时行为与旧版一致）"""
        EmotionConstants.reset()
        self.assertEqual(EmotionConstants.MIN_FAVOR, -100)
        self.assertEqual(EmotionConstants.MAX_FAVOR, 100)
        self.assertEqual(EmotionConstants.MIN_INTIMACY, 0)
        self.assertEqual(EmotionConstants.MAX_INTIMACY, 100)

    def test_config_200_is_accepted_and_kept(self):
        """配置 favour_max=200 后，写入 200 必须原样保留（核心回归点）"""
        EmotionConstants.configure(
            favour_min=-100, favour_max=200, intimacy_min=0, intimacy_max=200
        )

        state = _make_state(favor=200, intimacy=200)
        # 走一遍落盘/读盘往返，确保不只是内存里对
        roundtrip = EnhancedEmotionalState.from_dict(state.to_dict())

        self.assertEqual(roundtrip.favor, 200, "好感度 200 被钳制了")
        self.assertEqual(roundtrip.intimacy, 200, "亲密度 200 被钳制了")

    def test_composite_score_not_depressed(self):
        """复合评分不能因钳制而被压低（150 不能变成 100）"""
        EmotionConstants.configure(
            favour_min=-100, favour_max=200, intimacy_min=0, intimacy_max=200
        )

        state = _make_state(favor=200, intimacy=100)
        info = DynamicWeightManager.get_stage_info(state)

        # 共生期权重各 0.5：200*0.5 + 100*0.5 = 150
        self.assertAlmostEqual(
            info["composite_score"], 150.0, places=2,
            msg=f"复合评分应为 150，实际 {info['composite_score']}",
        )

    def test_old_behaviour_would_fail(self):
        """反向验证：不注入配置时，200 确实会被削成 100

        这条测试用来证明上面的回归测试是"有牙"的 ——
        如果哪天有人把钳制逻辑彻底删掉，这里会失败并提醒修正预期。
        """
        EmotionConstants.reset()
        state = _make_state(favor=200)
        roundtrip = EnhancedEmotionalState.from_dict(state.to_dict())
        self.assertEqual(roundtrip.favor, 100, "默认边界下应被钳到 100")

    def test_negative_favor_still_clamped(self):
        """负向边界同样跟随配置（下限设 -500 时 -200 应保留）"""
        EmotionConstants.configure(
            favour_min=-500, favour_max=500, intimacy_min=0, intimacy_max=500
        )
        state = _make_state(favor=-200)
        roundtrip = EnhancedEmotionalState.from_dict(state.to_dict())
        self.assertEqual(roundtrip.favor, -200)


class TestBoundsSafety(unittest.TestCase):
    """坏配置不得把边界改坏"""

    BASE = dict(favour_min=-100, favour_max=200, intimacy_min=0, intimacy_max=200)

    def setUp(self):
        EmotionConstants.reset()
        EmotionConstants.configure(**self.BASE)

    tearDown = setUp

    def _assert_kept(self, **bad_kwargs):
        EmotionConstants.configure(**bad_kwargs)
        self.assertEqual(EmotionConstants.MIN_FAVOR, -100,
                         f"坏配置 {bad_kwargs} 改坏了 MIN_FAVOR")
        self.assertEqual(EmotionConstants.MAX_FAVOR, 200,
                         f"坏配置 {bad_kwargs} 改坏了 MAX_FAVOR")
        self.assertEqual(EmotionConstants.MAX_INTIMACY, 200,
                         f"坏配置 {bad_kwargs} 改坏了 MAX_INTIMACY")

    def test_none_keeps_bounds(self):
        self._assert_kept(favour_max=None)

    def test_non_numeric_keeps_bounds(self):
        self._assert_kept(favour_max="abc")

    def test_over_magnitude_keeps_bounds(self):
        self._assert_kept(favour_max=99999)

    def test_max_below_min_keeps_bounds(self):
        """上限小于下限 —— 曾因只判 number>lower_bound 而放行了 -5"""
        self._assert_kept(favour_max=-5)

    def test_max_negative_keeps_bounds(self):
        """上限为负：-100 < -5 虽成立，但会让整个区间变负数，必须拒绝

        这是"整对采用"策略的最后一道护栏。缺了它，好感度区间会变成
        [-100, -5]，所有用户被判为负好感/敌对期。
        """
        EmotionConstants.configure(**self.BASE)
        EmotionConstants.configure(favour_max=-5)
        self.assertEqual(EmotionConstants.MAX_FAVOR, 200,
                         "上限为负时未拒绝，整个区间会变成负数")

    def test_max_zero_keeps_bounds(self):
        """上限为 0 同样拒绝（那会把所有好感度钳到 ≤0）"""
        EmotionConstants.configure(**self.BASE)
        EmotionConstants.configure(favour_max=0)
        self.assertEqual(EmotionConstants.MAX_FAVOR, 200)

    def test_intimacy_max_negative_keeps_bounds(self):
        """亲密度上限为负同样拒绝"""
        EmotionConstants.configure(**self.BASE)
        EmotionConstants.configure(intimacy_max=-1)
        self.assertEqual(EmotionConstants.MAX_INTIMACY, 200)

    def test_max_equal_min_keeps_bounds(self):
        self._assert_kept(favour_max=-100)

    def test_min_above_max_keeps_bounds(self):
        self._assert_kept(favour_min=300)

    def test_both_non_numeric_keeps_bounds(self):
        self._assert_kept(favour_min="x", favour_max="y")

    def test_bool_is_rejected(self):
        """bool 是 int 的子类，不能被当成合法数值"""
        self._assert_kept(favour_max=True)


class TestTransitionStatePersistence(unittest.TestCase):
    """_previous_stage / _previous_composite 必须能跨读写保留"""

    def setUp(self):
        EmotionConstants.reset()

    tearDown = setUp

    def test_previous_stage_survives_roundtrip(self):
        state = _make_state(favor=85, intimacy=80)
        state._previous_stage = "COMMITMENT"
        state._previous_composite = 82.5

        roundtrip = EnhancedEmotionalState.from_dict(state.to_dict())

        self.assertEqual(roundtrip._previous_stage, "COMMITMENT")
        self.assertAlmostEqual(roundtrip._previous_composite, 82.5, places=3)

    def test_previous_fields_are_serialized(self):
        """字段必须真的进存档，否则重启后必然丢失"""
        state = _make_state(favor=60)
        state._previous_stage = "DEEPENING"
        state._previous_composite = 58.0

        dumped = state.to_dict()

        self.assertIn("_previous_stage", dumped)
        self.assertIn("_previous_composite", dumped)

    def test_no_false_transition_after_reload(self):
        """阶段未变时，重载后不得误判为「正在过渡」"""
        state = _make_state(favor=52, intimacy=55)
        # 先跑一次，让 _previous_* 被正常写入
        DynamicWeightManager.get_stage_info(state)
        stage_before = state._previous_stage

        # 模拟落盘 + 重启后加载
        reloaded = EnhancedEmotionalState.from_dict(state.to_dict())
        self.assertEqual(reloaded._previous_stage, stage_before)

        info = DynamicWeightManager.get_stage_info(reloaded)
        self.assertFalse(
            info["is_transitioning"],
            "阶段未变化，却被判定为过渡中（说明过渡基线丢了）",
        )

    def test_invalid_previous_stage_ignored(self):
        """存档里是脏值时保持默认，不强行采用"""
        payload = _make_state(favor=10).to_dict()
        payload["_previous_stage"] = "NOT_A_STAGE"
        payload["_previous_composite"] = "abc"

        roundtrip = EnhancedEmotionalState.from_dict(payload)

        self.assertIsNone(roundtrip._previous_stage)
        self.assertEqual(roundtrip._previous_composite, 0.0)

    def test_old_save_without_previous_fields_still_loads(self):
        """旧存档没有这两个键时不能崩，按默认处理"""
        payload = _make_state(favor=30).to_dict()
        payload.pop("_previous_stage", None)
        payload.pop("_previous_composite", None)

        roundtrip = EnhancedEmotionalState.from_dict(payload)

        self.assertEqual(roundtrip.favor, 30)


class TestRepairKeepsSalvageableData(unittest.TestCase):
    """数据修复不得把还能用的字段一起清零"""

    def setUp(self):
        EmotionConstants.reset()
        self._tmp = TemporaryDirectory()
        self.repo = UserStateRepository(Path(self._tmp.name))

    def tearDown(self):
        EmotionConstants.reset()
        self._tmp.cleanup()

    def _load(self, user_key):
        return asyncio.run(self.repo.get_user_state(user_key))

    def _seed(self, user_key, state, **corrupt):
        payload = state.to_dict()
        payload.update(corrupt)
        self.repo._user_data = {user_key: payload}
        self.repo._loaded = True

    def test_corrupted_emotions_keeps_favor(self):
        """emotions 损坏时，好感度/亲密度必须保住（旧实现会清零）"""
        self._seed(
            "u1",
            _make_state("u1", favor=88, intimacy=77),
            emotions={"joy": 50, "不存在的字段": 1},
        )

        self._load("u1")
        repaired = self.repo._user_data["u1"]

        self.assertEqual(repaired["favor"], 88, "好感度被清零了")
        self.assertEqual(repaired["intimacy"], 77, "亲密度被清零了")

    def test_attribute_error_is_caught_in_place(self):
        """AttributeError 必须被 storage 就地兜住，而不是冒到外层重置状态

        直接调用 from_dict 会抛 AttributeError；storage.get_user_state
        必须捕获它并走修复分支，而不是让 managers 的兜底把数据抹掉。
        """
        broken = _make_state("u2", favor=42).to_dict()
        # 构造一个会让 EmotionalMetrics(**...) 抛 AttributeError 的输入
        broken["emotions"] = {"joy": 1}
        broken["stats"] = None
        self.repo._user_data = {"u2": broken}
        self.repo._loaded = True

        result = self._load("u2")  # 不应抛异常

        repaired = self.repo._user_data["u2"]
        self.assertEqual(repaired.get("favor"), 42, "可保留的好感度丢了")
        self.assertIsNotNone(repaired)

    def test_non_dict_entry_rebuilt(self):
        """整条记录非字典时才允许整体重建"""
        self.repo._user_data = {"u3": ["not", "a", "dict"]}
        self.repo._loaded = True

        self._load("u3")

        self.assertIsInstance(self.repo._user_data["u3"], dict)
        self.assertEqual(self.repo._user_data["u3"]["favor"], 0)

    def test_healthy_data_untouched(self):
        """完全正常的数据不应进入修复路径"""
        state = _make_state("u4", favor=66, intimacy=55)
        self.repo._user_data = {"u4": state.to_dict()}
        self.repo._loaded = True

        loaded = self._load("u4")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.favor, 66)
        self.assertEqual(loaded.intimacy, 55)


class TestPluginInjectsBounds(unittest.TestCase):
    """插件启动时必须把配置边界注入状态模型"""

    def tearDown(self):
        EmotionConstants.reset()

    def test_main_module_calls_configure(self):
        """main.py 的 __init__ 里必须有注入调用（防止有人删掉）"""
        import inspect
        from emotionai_pro.main import EmotionAIProPlugin

        source = inspect.getsource(EmotionAIProPlugin.__init__)
        self.assertIn("EmotionConstants.configure", source,
                      "__init__ 未注入数值边界，配置的上限会失效")

    def test_config_manager_syncs_on_reload(self):
        """配置热重载路径也必须同步边界"""
        import inspect
        from emotionai_pro.config_manager import ConfigManager

        self.assertTrue(
            hasattr(ConfigManager, "_apply_numeric_bounds"),
            "ConfigManager 缺少 _apply_numeric_bounds",
        )
        reload_src = inspect.getsource(ConfigManager._reload_config)
        self.assertIn("_apply_numeric_bounds", reload_src,
                      "_reload_config 未同步数值边界")
        update_src = inspect.getsource(ConfigManager.update_config)
        self.assertIn("_apply_numeric_bounds", update_src,
                      "update_config 未同步数值边界")


if __name__ == "__main__":
    unittest.main(verbosity=2)
