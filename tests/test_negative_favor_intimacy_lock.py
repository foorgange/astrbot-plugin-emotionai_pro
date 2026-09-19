# tests/test_negative_favor_intimacy_lock.py
"""v4.1.3 负好感阶段「亲密度固定为 0」回归测试

需求（用户 2026-09-19 原话）：
    「负数好感阶段,亲密度应该不变化,亲密度如果好感为正数才变化,
      然后如果好感为负数了,亲密度也降低为0.,
      也就是只要负好感阶段,亲密度固定为零
      （任何其它增加亲密度的机制你都检查下,规则都应该比这个低）」

拆成可验证的规则：
    ① 好感度为负 → 亲密度强制为 0（含读档、构造、管理员设置、存档修复）；
    ② 好感度由正转负的那一轮，亲密度同步清零（不允许「先结算完再转负」）；
    ③ 好感度为正时，亲密度一切机制照旧（常规变化 / 过渡增益 / 里程碑）；
    ④ 好感度回到 0 以上后，亲密度从 0 重新开始积累。

本文件锁死以上行为，并覆盖每条写入入口（模型层、LLM 打分、里程碑、
过渡增益、管理员命令、存档修复），以及「规则高于一切机制」这条优先级。
"""
import sys
import os
import copy
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401
from emotionai_pro.constants import EmotionConstants  # noqa: E402
from emotionai_pro.models import (  # noqa: E402
    EnhancedEmotionalState,
    clamp_intimacy_for_favor,
    is_negative_favor,
)
from emotionai_pro.relationship_manager import DynamicWeightManager  # noqa: E402
from emotionai_pro.main import EmotionAIProPlugin  # noqa: E402
from emotionai_pro.storage import UserStateRepository  # noqa: E402


def _state(favor: int, intimacy: int, key: str = "u") -> EnhancedEmotionalState:
    return EnhancedEmotionalState(user_key=key, favor=favor, intimacy=intimacy)


def _make_plugin(**cfg):
    """造一个够 _apply_expert_updates / _apply_intimacy_milestones 使用的插件实例"""
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
    return plugin


class _FakeEvent:
    """最小 event 桩：只需要 _is_admin / plain_result / stop_event"""

    def __init__(self, sender_id="1", role="member"):
        self.role = role
        self._sender_id = sender_id
        self.stopped = False

    def get_sender_id(self):
        return self._sender_id

    def plain_result(self, text):
        return text

    def stop_event(self):
        self.stopped = True


class _FakeUserManager:
    def __init__(self, state):
        self._state = state
        self.updated = None

    async def get_user_state(self, user_key):
        return self._state

    async def update_user_state(self, user_key, state):
        self.updated = (user_key, state)

    def resolve_user_key(self, user_input, session_based):
        return user_input


class _FakePlugin:
    def __init__(self, config, user_manager):
        self.config = config
        self.user_manager = user_manager
        self.invalidated = []

    async def invalidate_state_cache(self, user_key):
        self.invalidated.append(user_key)


class _ResetTestCase(unittest.TestCase):
    """每个用例前后都还原进程级单例，避免类属性/常量泄漏"""

    def setUp(self):
        DynamicWeightManager.reset()
        EmotionConstants.reset()

    def tearDown(self):
        DynamicWeightManager.reset()
        EmotionConstants.reset()


class TestSharedRule(_ResetTestCase):
    """共享判定函数：规则的唯一真源"""

    def test_is_negative_favor(self):
        self.assertTrue(is_negative_favor(-1))
        self.assertTrue(is_negative_favor(-200))
        self.assertFalse(is_negative_favor(0))
        self.assertFalse(is_negative_favor(1))
        # 类型异常按非负处理，交给后续校验逻辑报错
        self.assertFalse(is_negative_favor("abc"))
        self.assertFalse(is_negative_favor(None))
        self.assertFalse(is_negative_favor(True))

    def test_clamp_intimacy_for_favor(self):
        self.assertEqual(clamp_intimacy_for_favor(-4, 88), 0)
        self.assertEqual(clamp_intimacy_for_favor(-4, 0), 0)
        self.assertEqual(clamp_intimacy_for_favor(0, 88), 88)
        self.assertEqual(clamp_intimacy_for_favor(50, 88), 88)
        self.assertEqual(clamp_intimacy_for_favor("abc", 88), 88)


class TestModelLevelLock(_ResetTestCase):
    """① 模型写入口统一约束：负好感 → 亲密度只能是 0"""

    def test_construct_with_negative_favor_zeroes_intimacy(self):
        """读档/构造时 favor<0 + intimacy>0 → 亲密度被夹成 0"""
        state = _state(-4, 4)
        self.assertEqual(state.favor, -4)
        self.assertEqual(state.intimacy, 0, "负好感状态下亲密度必须为 0")

    def test_positive_favor_writes_intimacy_normally(self):
        """好感度为正时亲密度照常写入（不被误伤）"""
        state = _state(50, 10)
        state.intimacy = 60
        self.assertEqual(state.intimacy, 60)

    def test_favor_turning_negative_zeroes_intimacy(self):
        """② 好感度由正转负 → 亲密度同步清零"""
        state = _state(50, 60)
        state.favor = -1
        self.assertEqual(state.favor, -1)
        self.assertEqual(state.intimacy, 0, "好感度转负时亲密度必须同步清零")

    def test_recovery_reaccumulates_from_zero(self):
        """④ 好感度回到正 → 亲密度从 0 重新积累"""
        state = _state(50, 60)
        state.favor = -3
        self.assertEqual(state.intimacy, 0)
        state.favor = 5
        state.intimacy += 3
        self.assertEqual(state.intimacy, 3, "恢复后亲密度应从 0 重新开始积累")

    def test_dict_roundtrip_carries_zero(self):
        """负好感存档 to_dict/from_dict 闭环后亲密度仍是 0"""
        state = _state(-30, 80)
        payload = state.to_dict()
        self.assertEqual(payload["intimacy"], 0)
        self.assertEqual(
            EnhancedEmotionalState.from_dict(payload).intimacy, 0,
            "反序列化后亲密度必须仍为 0",
        )

    def test_healthy_positive_roundtrip_keeps_intimacy(self):
        """正好感存档 roundtrip 亲密度不被误清零"""
        state = _state(88, 77)
        self.assertEqual(
            EnhancedEmotionalState.from_dict(state.to_dict()).intimacy, 77
        )

    def test_favor_zero_is_not_negative(self):
        """favor=0 是中性而非负好感，亲密度照常（边界语义）"""
        state = _state(0, 0)
        state.intimacy = 5
        self.assertEqual(state.intimacy, 5)

    def test_deepcopy_preserves_lock(self):
        """深拷贝不会把锁后的状态复制回正值"""
        state = _state(-9, 99)
        cloned = copy.deepcopy(state)
        self.assertEqual((cloned.favor, cloned.intimacy), (-9, 0))

    def test_bad_favor_type_not_silently_corrected(self):
        """坏 favor 类型交给既有校验报错，不被本规则静默纠正"""
        with self.assertRaises((TypeError, ValueError)):
            EnhancedEmotionalState(user_key="u", favor="abc", intimacy=5)


class TestExpertUpdateFlow(_ResetTestCase):
    """②③ LLM 打分路径：先 favor 后 intimacy，负好感跳过亲密度"""

    # 关闭两类里程碑，隔离出「常规亲密度变化」这条路径。
    # ⚠️ intimacy_streak_days 有 pydantic 下限 2，不能填 0 关闭；
    # 关连击走 streak_bonus=0（代码先判它再判天数）。
    PLUGIN = dict(intimacy_first_deep_bonus=0, intimacy_streak_bonus=0)

    def test_favor_turning_negative_this_round_zeroes_intimacy(self):
        """本轮 favor 转负 → 亲密度必须先冻结清零，不被先结算完"""
        plugin = _make_plugin(**self.PLUGIN)
        state = _state(1, 40)
        updates = {"favor": -5, "intimacy": 2, "joy": 1, "trust": 1,
                   "source": "llm_analysis", "llm_available": True}

        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)

        self.assertEqual(state.favor, -4)
        self.assertEqual(state.intimacy, 0,
                         "好感度本轮已转负，亲密度必须为 0（不允许先结算再转负）")

    def test_intimacy_skipped_while_still_negative(self):
        """仍处负好感 → 本轮亲密度变化被忽略（哪怕好感度部分回升）"""
        plugin = _make_plugin(**self.PLUGIN)
        state = _state(-4, 0)
        updates = {"favor": 1, "intimacy": 3, "joy": 1}

        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)

        self.assertEqual(state.favor, -3)
        self.assertEqual(state.intimacy, 0,
                         "好感度仍为负，亲密度不应开始累积")

    def test_caller_dict_not_polluted(self):
        """调用方 dict 不被污染（之后还要参与意义计算/全局心情）"""
        plugin = _make_plugin(**self.PLUGIN)
        state = _state(1, 40)
        updates = {"favor": -5, "intimacy": 2, "joy": 1}

        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)

        self.assertEqual(updates["favor"], -5)
        self.assertEqual(updates["intimacy"], 2)

    def test_positive_flow_unaffected(self):
        """好感度为正时整条链路与改动前一致"""
        plugin = _make_plugin(**self.PLUGIN)
        state = _state(50, 40)
        updates = {"favor": 3, "intimacy": 5, "joy": 1,
                   "source": "llm_analysis", "llm_available": True}

        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)

        self.assertEqual(state.favor, 53)
        self.assertEqual(state.intimacy, 45)

    def test_favor_drop_without_intimacy_key_still_zeroes(self):
        """updates 里没有 intimacy 键也要清零（转负即清零）"""
        plugin = _make_plugin(**self.PLUGIN)
        state = _state(1, 40)
        updates = {"favor": -5, "joy": 1}

        EmotionAIProPlugin._apply_expert_updates(plugin, state, updates)

        self.assertEqual(state.favor, -4)
        self.assertEqual(state.intimacy, 0)


class TestMilestonesBlockedWhenNegative(_ResetTestCase):
    """③ 里程碑加成也受同一条规则约束"""

    def _plugin(self, **cfg):
        base = dict(intimacy_first_deep_bonus=3,
                    intimacy_streak_days=3,
                    intimacy_streak_bonus=1)
        base.update(cfg)
        return _make_plugin(**base)

    def test_no_bonus_when_favor_negative(self):
        """负好感 → 深度交流/连击里程碑都不发"""
        plugin = self._plugin()
        state = _state(-10, 0)
        # 意义分 8 ≥ DEEP_CONVERSATION(5)，本来足以触发首次深度交流
        updates = {"favor": 3, "intimacy": 1, "joy": 2, "trust": 2}

        plugin._apply_intimacy_milestones(state, updates)

        self.assertEqual(state.intimacy, 0, "负好感阶段不应发里程碑")
        self.assertFalse(state.stats.deep_conversation_achieved,
                         "负好感阶段不应标记首次深度交流")

    def test_bonus_still_works_when_favor_positive(self):
        """好感度为正 → 里程碑照常发放"""
        plugin = self._plugin()
        state = _state(10, 0)
        updates = {"favor": 3, "intimacy": 1, "joy": 2, "trust": 2}

        plugin._apply_intimacy_milestones(state, updates)

        self.assertEqual(state.intimacy, 3)
        self.assertTrue(state.stats.deep_conversation_achieved)


class TestTransitionBenefitsBlockedWhenNegative(_ResetTestCase):
    """③ 过渡期亲密度增益也受同一条规则约束"""

    def test_no_boost_when_favor_negative(self):
        """负好感 → 过渡增益原样返回，不放大亲密度"""
        state = _state(-10, 0)
        updates = {"intimacy": 1, "joy": 1}
        self.assertEqual(
            DynamicWeightManager.apply_transition_benefits(state, updates),
            updates,
        )

    def test_boost_still_applies_when_favor_positive(self):
        """正好感且门槛阻断 → 增益照常（深化期 3.6 倍，int(2×3.6)=7）"""
        DynamicWeightManager.configure(transition_intimacy_pct=50)
        state = _state(79, 0)
        updates = {"intimacy": 2}

        result = DynamicWeightManager.apply_transition_benefits(state, updates)

        self.assertEqual(result["intimacy"], 7, "正好感的过渡增益不应被影响")

    def test_auto_intimacy_when_no_intimacy_key(self):
        """positive: 深度交流轮次没有 intimacy 键时自动补亲密度"""
        DynamicWeightManager.configure(transition_intimacy_pct=50)
        state = _state(79, 0)
        updates = {"joy": 2, "trust": 2}

        result = DynamicWeightManager.apply_transition_benefits(state, updates)

        self.assertGreater(result.get("intimacy", 0), 0,
                           "正好感的自动亲密度加成不应被影响")


class TestAdminCommands(_ResetTestCase):
    """① 管理员手动设置同样服从规则"""

    @staticmethod
    def _handler(state, favor_value=None, intimacy_value=None):
        from emotionai_pro.config import PluginConfig
        from emotionai_pro.command_handlers import AdminCommandHandler

        config = PluginConfig()
        plugin = _FakePlugin(config, _FakeUserManager(state))
        handler = AdminCommandHandler.__new__(AdminCommandHandler)
        handler.plugin = plugin
        handler.config = config
        handler.user_manager = plugin.user_manager
        return handler

    @staticmethod
    def _run(coro):
        async def _collect(c):
            return [item async for item in c]

        return asyncio.run(_collect(coro))

    def test_set_intimacy_refused_when_favor_negative(self):
        """负好感 → /设置亲密度 拒绝执行并说明原因"""
        state = _state(-4, 0)
        handler = self._handler(state)
        event = _FakeEvent(role="admin")

        results = self._run(handler.set_intimacy(event, "u1", "50"))

        self.assertTrue(any("负好感" in r for r in results),
                        f"应提示负好感阶段无法设置，实际: {results}")
        self.assertIsNone(handler.user_manager.updated, "负好感时不应写入")

    def test_set_intimacy_allowed_when_favor_positive(self):
        """好感度为正 → /设置亲密度 正常执行"""
        state = _state(50, 0)
        handler = self._handler(state)
        event = _FakeEvent(role="admin")

        results = self._run(handler.set_intimacy(event, "u1", "30"))

        self.assertTrue(any("30" in r for r in results))
        self.assertEqual(handler.user_manager.updated[1].intimacy, 30)

    def test_set_favor_announces_intimacy_reset(self):
        """/设置好感度 调到负数 → 提示亲密度已同步清零"""
        state = _state(50, 60)
        handler = self._handler(state)
        event = _FakeEvent(role="admin")

        results = self._run(handler.set_favor(event, "u1", "-5"))

        self.assertTrue(any("清零" in r for r in results),
                        f"应提示亲密度清零，实际: {results}")
        self.assertEqual(handler.user_manager.updated[1].intimacy, 0)

    def test_set_favor_positive_no_reset_notice(self):
        """/设置好感度 调到正数 → 不提清零"""
        state = _state(-5, 0)
        handler = self._handler(state)
        event = _FakeEvent(role="admin")

        results = self._run(handler.set_favor(event, "u1", "10"))

        self.assertFalse(any("清零" in r for r in results))


class TestRepairNormalization(unittest.TestCase):
    """① 存档修复落盘必须与内存口径一致"""

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

    def test_repair_zeroes_intimacy_for_negative_favor(self):
        """负好感 + 损坏子对象 → 修复后落盘的亲密度也是 0

        否则会出现「内存里 0、磁盘里正值」，下次读档又被夹一次，
        两边长期不一致。
        """
        self._seed(
            "u1",
            _state(-30, 80),
            emotions={"joy": 50, "不存在的字段": 1},
        )

        self._load("u1")
        repaired = self.repo._user_data["u1"]

        self.assertEqual(repaired["favor"], -30)
        self.assertEqual(repaired["intimacy"], 0,
                         "负好感用户修复后落盘的亲密度必须为 0")

    def test_repair_keeps_intimacy_for_positive_favor(self):
        """正好感 → 修复不动亲密度"""
        self._seed(
            "u2",
            _state(88, 77),
            emotions={"joy": 50, "不存在的字段": 1},
        )

        self._load("u2")
        repaired = self.repo._user_data["u2"]

        self.assertEqual(repaired["favor"], 88)
        self.assertEqual(repaired["intimacy"], 77)


if __name__ == "__main__":
    unittest.main()
