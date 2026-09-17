# tests/test_live_config_fields.py
"""「死配置」回归测试（v4.0.20）

背景（用户提问引发的一次全量审计）：
    用户问「单次好感度减少的最大数值我设的 3，是不是得设成 -3 才正常运行」。
    顺着这条线索做了一次**所有配置字段的全量审计**，结果发现 6 个字段是
    「死配置」——配置界面能填、pydantic 能收、config_manager 会校验，
    但**没有任何代码读取它们**：

    | 字段 | 曾经的真实行为 |
    |---|---|
    | change_min / change_max | 单次幅度写死 `max(-5, min(5, v))`（emotion_expert.py） |
    | emotional_significance_threshold | 写死用 `UpdateThresholds.EMOTIONAL_SIGNIFICANCE`(=5) |
    | plugin_priority | 两个 LLM 钩子写死 `priority=100000` |
    | enable_attitude_system | 关闭后态度描述照样注入主 LLM |
    | enable_ai_text_generation | 关闭后 AI 照样改写态度/关系描述 |

    另外 `change_min >= change_max` 的校验只在**热重载**路径生效，
    用户把符号填反（服务器上真实存在 3 / 2）会导致「改别的配置也保存不上」，
    且只有一行含糊的「新配置验证失败」。

本文件锁死修复后的行为：这些字段**改了必须真的起作用**。
"""
import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.config_manager import ConfigManager  # noqa: E402
from emotionai_pro.emotion_expert import EmotionAnalysisExpert  # noqa: E402
from emotionai_pro.memory import EnhancedMemorySystem  # noqa: E402


class _FakeCache:
    """给 EmotionAnalysisExpert 用的最小缓存桩"""

    async def get(self, key):
        return None

    async def set(self, key, value, ttl=None):
        return True


def _make_expert(**kwargs):
    return EmotionAnalysisExpert(_FakeCache(), None, **kwargs)


def _parse_with_ranges(expert):
    """走真实的 JSON 解析分支，返回被钳制后的 emotion_updates

    直接调用内部解析入口，避免依赖 LLM 网络调用。
    """
    from emotionai_pro.models import EnhancedEmotionalState

    payload = (
        '{"emotion_updates": {"favor": 99, "intimacy": -99},'
        ' "relationship": "测试关系", "attitude": "测试态度"}'
    )
    state = EnhancedEmotionalState(user_key="u")
    return expert._parse_emotion_analysis(payload, state)


class TestChangeRangeFollowsConfig(unittest.TestCase):
    """change_min / change_max 必须真的限制单次好感度变化幅度

    注意作用范围：这两个字段的定义是「**好感度**单次变化幅度」，
    原作者（v3.30）也只对 favor 做此钳制，所以只有 favor 受它们约束；
    intimacy 沿用原先固定的 ±5。
    """

    def test_config_range_is_applied_to_favor(self):
        """设成 -3/3 后，LLM 返回的 +99 应被削到 +3"""
        expert = _make_expert(change_min=-3, change_max=3)
        self.assertEqual(expert.change_min, -3)
        self.assertEqual(expert.change_max, 3)

        updates = _parse_with_ranges(expert)
        self.assertEqual(updates["favor"], 3, "favor 未被 change_max=3 限制")

    def test_negative_side_of_range_is_applied(self):
        """负侧同样受 change_min 约束"""
        expert = _make_expert(change_min=-3, change_max=3)
        updates = _parse_with_ranges(expert)
        self.assertEqual(updates["favor"], 3)
        # 用负向载荷再验一次
        from emotionai_pro.models import EnhancedEmotionalState
        payload = '{"emotion_updates": {"favor": -99}, "relationship": "r", "attitude": "a"}'
        out = expert._parse_emotion_analysis(payload, EnhancedEmotionalState(user_key="u"))
        self.assertEqual(out["favor"], -3, "favor 未被 change_min=-3 限制")

    def test_intimacy_keeps_original_fixed_range(self):
        """亲密度不受 change_min/max 影响，保持原固定 ±5（不擅自改变语义）"""
        expert = _make_expert(change_min=-3, change_max=3)
        updates = _parse_with_ranges(expert)
        self.assertEqual(updates["intimacy"], -5,
                         "亲密度不该被 change_min=-3 改变（那是好感度的字段）")

    def test_default_range_is_used_when_not_configured(self):
        """不传参时退回默认 -10/5"""
        expert = _make_expert()
        self.assertEqual(expert.change_min, -10)
        self.assertEqual(expert.change_max, 5)

        updates = _parse_with_ranges(expert)
        self.assertEqual(updates["favor"], 5)

    def test_illegal_pair_falls_back_to_default(self):
        """非法组合（含符号填反的 3/2）整对退回默认，绝不半套生效"""
        for bad in (
            dict(change_min=3, change_max=2),     # 符号填反 —— 用户服务器上的真实值
            dict(change_min=5, change_max=5),     # 相等
            dict(change_min=3, change_max=3),     # 相等且都为正
            dict(change_min=-3, change_max=-1),   # 都为负
            dict(change_min=None, change_max=5),
            dict(change_min="x", change_max=5),
            dict(change_min=True, change_max=5),
            dict(change_min=-99999, change_max=5),  # 量级离谱
        ):
            expert = _make_expert(**bad)
            self.assertEqual(
                (expert.change_min, expert.change_max), (-10, 5),
                f"坏配置 {bad} 未被整对退回默认",
            )

    def test_reverse_proof_old_behaviour_would_fail(self):
        """反向证明：若幅度仍写死 ±5，配置成 ±3 时这里就会失败"""
        expert = _make_expert(change_min=-3, change_max=3)
        updates = _parse_with_ranges(expert)
        # 写死 ±5 的实现会返回 5，与配置的 3 不符
        self.assertNotEqual(updates["favor"], 5)
        self.assertEqual(updates["favor"], 3)


class TestSignificanceThresholdFollowsConfig(unittest.TestCase):
    """emotional_significance_threshold 必须决定「多重要才记进长期记忆」"""

    @staticmethod
    def _mk_mem(threshold=None):
        """构造一个只关心阈值的实例

        `EnhancedMemorySystem.__init__` 会建 `ShardedTTLCache`，而后者在构造时
        就 `asyncio.create_task` —— 没有事件循环会直接 RuntimeError。
        这里用 asyncio.run 起一个临时循环来构造，构造完即关闭
        （缓存的后台清理任务在该循环里，本测试不依赖它）。
        """
        import asyncio

        async def _build():
            if threshold is None:
                return EnhancedMemorySystem(None)
            return EnhancedMemorySystem(None, significance_threshold=threshold)

        return asyncio.run(_build())

    def test_threshold_is_used(self):
        mem = self._mk_mem(8)
        self.assertEqual(mem.significance_threshold, 8)

    def test_default_when_not_given(self):
        mem = self._mk_mem()
        self.assertEqual(mem.significance_threshold, 5)

    def test_illegal_value_falls_back(self):
        for bad in (0, -1, 11, 999, None, "x", True):
            mem = self._mk_mem(bad)
            self.assertEqual(
                mem.significance_threshold, 5,
                f"非法阈值 {bad!r} 未被退回默认 5",
            )

    def test_threshold_actually_gates_long_term_memory(self):
        """阈值调高到 8 时，意义=5 的互动**不应**进长期记忆"""
        import asyncio

        def _run():
            async def _inner():
                mem = EnhancedMemorySystem(None, significance_threshold=8)
                called = {"long": 0}

                async def _fake_long(user_key, interaction):
                    called["long"] += 1

                async def _fake_short(user_key, interaction):
                    return None

                async def _fake_save():
                    return None

                async def _ensure():
                    return None

                mem._ensure_memory_loaded = _ensure
                mem._add_to_short_term_memory = _fake_short
                mem._add_to_long_term_memory = _fake_long
                mem._save_long_term_memory = _fake_save

                # significance=5 < 阈值 8 → 不该写长期记忆
                await mem.add_interaction("u1", "hi", "hello", 5, {})
                low = called["long"]
                # significance=8 >= 阈值 8 → 应写长期记忆
                await mem.add_interaction("u1", "hi", "hello", 8, {})
                return low, called["long"]

            return asyncio.run(_inner())

        low, high = _run()
        self.assertEqual(low, 0, "阈值=8 时，意义=5 的互动不应进长期记忆")
        self.assertEqual(high, 1, "阈值=8 时，意义=8 的互动应进长期记忆")


class TestConfigRepairInsteadOfReject(unittest.TestCase):
    """填反符号时只修复那一对，不再整份拒绝保存"""

    def _mk_manager(self):
        # 不跑 __init__（会碰文件系统），只测纯逻辑方法
        return ConfigManager.__new__(ConfigManager)

    def test_reversed_pair_is_repaired_not_rejected(self):
        cm = self._mk_manager()
        cfg = PluginConfig(change_min=3, change_max=2)
        ok = cm._validate_config(cfg)
        self.assertTrue(ok, "填反符号不应导致整份配置被拒")
        self.assertEqual(
            (cfg.change_min, cfg.change_max), (-10, 5),
            "填反的那一对应被重置为默认值",
        )

    def test_other_fields_survive_the_repair(self):
        """修复坏字段时，其它配置（尤其 admin_qq_list）必须原样保留"""
        cm = self._mk_manager()
        cfg = PluginConfig(
            change_min=3, change_max=2,
            admin_qq_list=["3418451176"],
            favour_max=200,
        )
        cm._validate_config(cfg)
        self.assertEqual(cfg.admin_qq_list, ["3418451176"],
                         "修复坏字段时误伤了管理员列表")
        self.assertEqual(cfg.favour_max, 200)

    def test_valid_config_is_untouched(self):
        cm = self._mk_manager()
        cfg = PluginConfig(change_min=-3, change_max=3)
        self.assertTrue(cm._validate_config(cfg))
        self.assertEqual((cfg.change_min, cfg.change_max), (-3, 3))

    def test_bad_admin_entries_are_dropped_not_rejected(self):
        cm = self._mk_manager()
        cfg = PluginConfig(admin_qq_list=["123456", "abc", "789"])
        self.assertTrue(cm._validate_config(cfg))
        self.assertEqual(cfg.admin_qq_list, ["123456", "789"],
                         "非法 QQ 应被逐个剔除而非整份拒绝")

    def test_nonpositive_force_interval_is_repaired(self):
        """force_update_interval <= 0 时修复为默认值

        注意：pydantic 已声明 ge=1，所以 0/负数在**构造阶段**就会被拦下
        并走 main.py 的逐字段回退。这里直接构造非法对象来覆盖 ConfigManager
        的兜底分支（例如配置来自旧版本文件、绕过 pydantic 的场景）。
        """
        cm = self._mk_manager()
        cfg = PluginConfig()
        cfg.force_update_interval = 0  # 绕过 pydantic，模拟陈旧数据
        self.assertTrue(cm._validate_config(cfg))
        self.assertEqual(cfg.force_update_interval, 5)


class TestPluginWiresTheFields(unittest.TestCase):
    """源码级断言：这些字段必须真的被传下去/读取，防止将来又被架空"""

    def _src(self, module_name):
        root = Path(__file__).resolve().parent.parent
        return (root / module_name).read_text(encoding="utf-8")

    def test_main_passes_change_range_to_expert(self):
        src = self._src("main.py")
        self.assertIn("change_min=self.config.change_min", src)
        self.assertIn("change_max=self.config.change_max", src)

    def test_main_passes_text_generation_switch(self):
        src = self._src("main.py")
        self.assertIn("enable_ai_text_generation=self.config.enable_ai_text_generation", src)

    def test_main_passes_significance_threshold(self):
        src = self._src("main.py")
        self.assertIn("significance_threshold=self.config.emotional_significance_threshold", src)

    def test_main_reads_plugin_priority(self):
        src = self._src("main.py")
        self.assertIn("self.config.plugin_priority", src)

    def test_main_reads_attitude_switch(self):
        src = self._src("main.py")
        self.assertIn('getattr(self.config, "enable_attitude_system"', src)

    def test_memory_uses_instance_threshold(self):
        src = self._src("memory.py")
        self.assertIn("self.significance_threshold", src)

    def test_expert_uses_configured_range(self):
        src = self._src("emotion_expert.py")
        self.assertIn("self.change_min", src)
        self.assertIn("self.change_max", src)
        # 旧的写死写法不应再出现在 favor 分支上
        self.assertNotIn(
            "if emotion in ['favor', 'intimacy']:\n"
            "                                    int_value = max(-5, min(5, int_value))",
            src,
            "favor 仍然被写死的 ±5 钳制（配置未接通）",
        )


class TestAiTextGenerationSwitchBehaviour(unittest.TestCase):
    """「启用 AI 自主生成文本描述」开关必须有**真实行为**差异

    ⚠️ 这里刻意做成行为测试而不是源码断言：
    该开关的关闭分支曾经写成 `state.descriptions.attitude`，而方法参数名是
    `current_state` —— 于是抛 NameError，被方法外层宽泛的 `except Exception`
    吞掉，静默退化到「文本提取」兜底返回写死的「正常关系/友好交流」。
    源码里 `self.enable_ai_text_generation` 看起来完全正确，
    只有真正跑一遍才能发现开关其实没生效。
    """

    def _payload(self, favor=1):
        return (
            '{"emotion_updates": {"favor": %d},'
            ' "relationship": "AI生成的关系", "attitude": "AI生成的态度"}' % favor
        )

    def test_off_keeps_user_set_descriptions(self):
        """关闭时必须保留用户手工设置的描述，而不是写死的兜底文案"""
        from emotionai_pro.models import EnhancedEmotionalState

        expert = _make_expert(enable_ai_text_generation=False)
        state = EnhancedEmotionalState(user_key="u")
        # 先设成非默认值，才能区分「保留原值」与「被覆盖成兜底文案」
        state.descriptions.attitude = "手工态度"
        state.descriptions.relationship = "手工关系"

        out = expert._parse_emotion_analysis(self._payload(), state)

        self.assertEqual(out["attitude_text"], "手工态度",
                         "关闭开关后态度描述被改写（开关未生效）")
        self.assertEqual(out["relationship_text"], "手工关系",
                         "关闭开关后关系描述被改写（开关未生效）")
        # 明确排除「静默退化成兜底文案」这一种失败形态
        self.assertNotIn(out["attitude_text"], ("友好交流",),
                         "落到了写死的兜底文案 —— 极可能是关闭分支抛异常被吞")
        self.assertNotIn(out["relationship_text"], ("正常关系",),
                         "落到了写死的兜底文案 —— 极可能是关闭分支抛异常被吞")

    def test_on_uses_ai_generated_descriptions(self):
        """开启时必须采用 LLM 返回的描述"""
        from emotionai_pro.models import EnhancedEmotionalState

        expert = _make_expert(enable_ai_text_generation=True)
        state = EnhancedEmotionalState(user_key="u")
        state.descriptions.attitude = "手工态度"
        state.descriptions.relationship = "手工关系"

        out = expert._parse_emotion_analysis(self._payload(), state)

        self.assertEqual(out["attitude_text"], "AI生成的态度")
        self.assertEqual(out["relationship_text"], "AI生成的关系")

    def test_off_does_not_trigger_fallback(self):
        """关闭时不能让解析整体失败（否则连 favor 更新都会丢）"""
        from emotionai_pro.models import EnhancedEmotionalState

        expert = _make_expert(change_min=-3, change_max=3,
                              enable_ai_text_generation=False)
        state = EnhancedEmotionalState(user_key="u")
        out = expert._parse_emotion_analysis(self._payload(favor=99), state)
        self.assertEqual(out["favor"], 3,
                         "识别失败：favor 未按配置钳制到 3，可能落到了文本提取兜底")

    def test_description_branch_uses_correct_parameter_name(self):
        """关闭分支必须引用 current_state（而非不存在的 state）

        这是上面 NameError 的直接回归防线：AST 层面确认
        `_parse_emotion_analysis` 方法体内没有裸露的 `state` 名字。
        """
        import ast

        src = (Path(__file__).resolve().parent.parent / "emotion_expert.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_parse_emotion_analysis":
                params = [a.arg for a in node.args.args]
                self.assertIn("current_state", params)
                self.assertNotIn("state", params)
                bare = [n.lineno for n in ast.walk(node)
                        if isinstance(n, ast.Name) and n.id == "state"]
                self.assertEqual(bare, [],
                                 f"方法内出现未定义的 `state`（行号 {bare}）")
                return
        self.fail("未找到 _parse_emotion_analysis 方法")


if __name__ == "__main__":
    unittest.main()
