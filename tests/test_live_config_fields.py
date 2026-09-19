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

    def test_intimacy_does_not_follow_favor_range(self):
        """亲密度有**自己的**配置对（v4.0.21），不再跟 change_min/max 走

        v4.0.20 时这里断言「亲密度保持固定 ±5」；v4.0.21 起亲密度改由
        intimacy_change_min/max 驱动，默认 ±3。
        """
        expert = _make_expert(change_min=-10, change_max=5)
        self.assertEqual(
            (expert.intimacy_change_min, expert.intimacy_change_max), (-3, 3),
            "不显式配置时亲密度应退回出厂默认 ±3",
        )
        updates = _parse_with_ranges(expert)
        self.assertEqual(updates["intimacy"], -3,
                         "亲密度应按自己的默认 ±3 钳制（既不是老的 ±5，也不跟好感度配置走）")
        self.assertEqual(updates["favor"], 5,
                         "好感度仍应由 change_min/max 决定")

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


class TestIntimacyChangeRangeFollowsConfig(unittest.TestCase):
    """v4.0.21：亲密度的单次变化幅度必须由 intimacy_change_min/max 决定

    背景：v4.0.21 之前这里是写死的 `max(-5, min(5, v))` —— LLM 一条消息
    就能让亲密度跳 5 点，而用户没有任何配置项能约束它（和 v4.0.20 修掉的
    change_min/change_max 死配置是同一类问题）。
    现在由配置驱动，出厂默认 ±3（比原来的 ±5 更保守）。

    注意作用范围：这一对**只**约束亲密度，绝不碰好感度 ——
    好感度仍由 change_min/max 管（见 TestChangeRangeFollowsConfig）。
    """

    @staticmethod
    def _payload(intimacy):
        return ('{"emotion_updates": {"favor": 0, "intimacy": %d},'
                ' "relationship": "测试关系", "attitude": "测试态度"}' % intimacy)

    def _parse(self, expert, intimacy):
        from emotionai_pro.models import EnhancedEmotionalState
        return expert._parse_emotion_analysis(
            self._payload(intimacy), EnhancedEmotionalState(user_key="u")
        )

    def test_config_range_is_applied_to_intimacy(self):
        """设成 -3/3 后，LLM 返回的 ±99 应被削到 ±3"""
        expert = _make_expert(intimacy_change_min=-3, intimacy_change_max=3)
        self.assertEqual(expert.intimacy_change_min, -3)
        self.assertEqual(expert.intimacy_change_max, 3)
        self.assertEqual(self._parse(expert, 99)["intimacy"], 3)
        self.assertEqual(self._parse(expert, -99)["intimacy"], -3)

    def test_default_is_pm3(self):
        """不传参时退回出厂默认 ±3（v4.0.21 起不再是 ±5）"""
        expert = _make_expert()
        self.assertEqual(
            (expert.intimacy_change_min, expert.intimacy_change_max), (-3, 3)
        )

    def test_favor_is_not_affected_by_intimacy_range(self):
        """intimacy_change_* 绝不改变好感度的钳制结果"""
        expert = _make_expert(intimacy_change_min=-1, intimacy_change_max=1)
        out = self._parse(expert, 99)
        self.assertEqual(out["intimacy"], 1)
        self.assertEqual(out["favor"], 0, "好感度被亲密度配置误伤了")

    def test_illegal_pair_falls_back_to_default(self):
        """非法组合整对退回默认 ±3，绝不半套生效"""
        for bad in (
            dict(intimacy_change_min=3, intimacy_change_max=2),   # 符号填反
            dict(intimacy_change_min=2, intimacy_change_max=2),   # 相等
            dict(intimacy_change_min=-1, intimacy_change_max=-5), # 都为负
            dict(intimacy_change_min=None, intimacy_change_max=5),
            dict(intimacy_change_min="x", intimacy_change_max=5),
            dict(intimacy_change_min=True, intimacy_change_max=5),
            dict(intimacy_change_min=-99999, intimacy_change_max=5),  # 量级离谱
        ):
            expert = _make_expert(**bad)
            self.assertEqual(
                (expert.intimacy_change_min, expert.intimacy_change_max), (-3, 3),
                f"坏配置 {bad} 未被整对退回默认",
            )

    def test_local_fallback_path_is_clamped_too(self):
        """本地兜底路径（不走 LLM）也必须受配置约束

        `_generate_smart_fallback` 的「亲密互动」分支会写死产出 intimacy=3。
        若用户把上限配成 1，这个值必须被削到 1 —— 否则就是又一起
        「配置没生效」。三条路径在 analyze_and_update_emotion 里都汇到
        `_ensure_updates_completeness`，钳制就做在那里。
        """
        from emotionai_pro.models import EnhancedEmotionalState

        expert = _make_expert(intimacy_change_min=-1, intimacy_change_max=1)
        state = EnhancedEmotionalState(user_key="u")

        raw = expert._generate_smart_fallback("宝贝", "", state)
        self.assertEqual(raw["intimacy"], 3, "兜底路径原始值应为写死的 3")

        out = expert._ensure_updates_completeness(raw, state)
        self.assertEqual(out["intimacy"], 1, "漏斗处未按 intimacy_change_max=1 钳制")

    def test_text_extraction_path_is_clamped_too(self):
        """「从文本提取」兜底路径同样受约束"""
        from emotionai_pro.models import EnhancedEmotionalState

        expert = _make_expert(intimacy_change_min=-1, intimacy_change_max=1)
        state = EnhancedEmotionalState(user_key="u")

        raw = expert._extract_updates_from_text("非常亲密的关系")
        self.assertEqual(raw["intimacy"], 2, "文本提取的原始值应为 2")

        out = expert._ensure_updates_completeness(raw, state)
        self.assertEqual(out["intimacy"], 1, "漏斗处未按配置钳制文本提取路径")

    def test_funnel_does_not_touch_favor(self):
        """本次只钳亲密度：好感度行为必须与 v4.0.20 保持一致（防回归）"""
        from emotionai_pro.models import EnhancedEmotionalState

        expert = _make_expert()
        state = EnhancedEmotionalState(user_key="u")
        raw = expert._generate_smart_fallback("宝贝", "", state)
        out = expert._ensure_updates_completeness(raw, state)
        self.assertEqual(out["favor"], raw["favor"], "好感度被漏斗改动了")
        self.assertEqual(out["intimacy"], 3, "默认 ±3 下亲密度 3 不应被削")

    def test_default_clamp_is_idempotent_on_llm_path(self):
        """LLM 路径已经在区间内的值，过漏斗后不应再变"""
        from emotionai_pro.models import EnhancedEmotionalState

        expert = _make_expert(intimacy_change_min=-2, intimacy_change_max=4)
        state = EnhancedEmotionalState(user_key="u")
        llm_out = self._parse(expert, 3)
        self.assertEqual(llm_out["intimacy"], 3)
        funnelled = expert._ensure_updates_completeness(dict(llm_out), state)
        self.assertEqual(funnelled["intimacy"], 3, "幂等性被破坏")

    def test_reverse_proof_old_hardcoded_pm5_would_fail(self):
        """反向证明：若仍写死 ±5，配成 ±3 时这里就会失败"""
        expert = _make_expert(intimacy_change_min=-3, intimacy_change_max=3)
        out = self._parse(expert, 99)
        self.assertNotEqual(out["intimacy"], 5, "亲密度仍是写死的 ±5（配置未接通）")
        self.assertEqual(out["intimacy"], 3)

    def test_schema_has_slider_and_matches_pydantic(self):
        """_conf_schema.json 必须给这两个字段配 slider（图形化配置），
        且默认值与 PluginConfig 一致 —— 两边不一致就会出现
        「界面显示一个值、实际生效另一个值」"""
        import json

        root = Path(__file__).resolve().parent.parent
        schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
        cfg = PluginConfig()

        for field, default in (
            ("intimacy_change_min", cfg.intimacy_change_min),
            ("intimacy_change_max", cfg.intimacy_change_max),
        ):
            node = schema[field]
            self.assertIn("slider", node, f"{field} 缺少 slider，无法图形化配置")
            slider = node["slider"]
            self.assertLessEqual(slider["min"], default)
            self.assertGreaterEqual(slider["max"], default)
            self.assertEqual(node["default"], default,
                             f"{field} 的 schema 默认值与 PluginConfig 不一致")


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

    def test_reversed_intimacy_pair_is_repaired_not_rejected(self):
        """v4.0.21：亲密度的变化幅度对填反时也只修那一对"""
        cm = self._mk_manager()
        cfg = PluginConfig(intimacy_change_min=3, intimacy_change_max=2)
        self.assertTrue(cm._validate_config(cfg))
        self.assertEqual(
            (cfg.intimacy_change_min, cfg.intimacy_change_max), (-3, 3),
            "填反的亲密度幅度对应被重置为默认 ±3",
        )


class TestPluginWiresTheFields(unittest.TestCase):
    """源码级断言：这些字段必须真的被传下去/读取，防止将来又被架空"""

    def _src(self, module_name):
        root = Path(__file__).resolve().parent.parent
        return (root / module_name).read_text(encoding="utf-8")

    def test_main_passes_change_range_to_expert(self):
        src = self._src("main.py")
        self.assertIn("change_min=self.config.change_min", src)
        self.assertIn("change_max=self.config.change_max", src)

    def test_main_passes_intimacy_change_range_to_expert(self):
        """v4.0.21：亲密度的变化幅度也必须从配置传进 Expert"""
        src = self._src("main.py")
        self.assertIn("intimacy_change_min=self.config.intimacy_change_min", src)
        self.assertIn("intimacy_change_max=self.config.intimacy_change_max", src)
        base_mapping = self._src("main.py")
        self.assertIn('"intimacy_change_min": "intimacy_change_min"', base_mapping,
                      "base_mapping 白名单漏了 intimacy_change_min")
        self.assertIn('"intimacy_change_max": "intimacy_change_max"', base_mapping,
                      "base_mapping 白名单漏了 intimacy_change_max")

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
