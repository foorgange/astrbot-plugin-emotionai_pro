# tests/test_mood_v418.py
"""v4.1.8 心情变化原因：大模型执笔 + 历史能累积 + 多套兜底文案

线上 bug（用户实测）：
1. /心情 的「最近三次变化」永远只显示一条——push_change_reason 对
   同文本永久合并，而专家通道触发语固定、回落模板也固定，同一句话
   被无限合并；
2. 变化文本单一——「回味刚才的对话之后，永雏塔菲的心情渐渐平复」
   反复出现。

本版修复：
1. 同文本去重改为时间窗口判定（MOOD_HISTORY_DEDUP_WINDOW，见
   test_mood_v416.py 的 TestMoodHistory）：窗口内的重复是同一事件的
   双报，只刷时间戳；窗口外的重复照常进历史；
2. 变化原因优先由情感分析大模型生成（enable_llm_mood_reason 可关），
   拿不到时走内置多套模板轮换 + 专家触发语多套同义表达轮换。
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401  (注入 astrbot stub)
import tests.bootstrap  # noqa: F401

from emotionai_pro.global_mood import (  # noqa: E402
    GlobalMood, MOOD_HISTORY_LIMIT,
    apply_mood_update, build_mood_change_reason,
    expert_mood_trigger, mood_variant_seed,
    _EXPERT_MOOD_TRIGGERS,
)
from emotionai_pro.models import EmotionalMetrics  # noqa: E402
from emotionai_pro.emotion_expert import EmotionAnalysisExpert  # noqa: E402
from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.main import EmotionAIProPlugin  # noqa: E402
from emotionai_pro.schema_validator import ConfigValidator  # noqa: E402


def _mood_with(**dims):
    """按 8 维情绪值构造心情（自动重算强度/主导）"""
    mood = GlobalMood(emotions=EmotionalMetrics(**dims))
    mood._recompute()
    return mood


def _expert(**kw):
    """最小专家实例：只够跑提示词构建与结果解析"""
    return EmotionAnalysisExpert(types.SimpleNamespace(), None, **kw)


def _state():
    return types.SimpleNamespace(
        favor=10,
        intimacy=5,
        stats=types.SimpleNamespace(total_count=3, positive_ratio=60.0),
        descriptions=types.SimpleNamespace(attitude="友好", relationship="正常"),
    )


def _llm_json(**extra):
    """一份合法的最小情感分析 JSON（mood_reason 等字段按需附加）"""
    payload = {
        "emotion_updates": {
            "favor": 1, "intimacy": 0, "joy": 1, "trust": 0, "fear": 0,
            "surprise": 0, "sadness": 0, "disgust": 0, "anger": 0,
            "anticipation": 0,
        },
        "relationship": "老朋友",
        "attitude": "热情",
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _mood_plugin(mood):
    """可跑 _update_global_mood 的最小插件实例（协作者全部桩化）"""
    plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
    plugin.config = PluginConfig()
    plugin._mood_cache = mood
    plugin._mood_cache_time = time.time()
    plugin._get_bot_name = lambda: "永雏塔菲"
    plugin.get_mood_sync = lambda: mood
    saved = []

    class _Store:
        async def save(self, m):
            saved.append(m)

    plugin.global_mood_store = _Store()
    return plugin, saved


class TestBuildReasonLlmAndTemplates(unittest.TestCase):
    """build_mood_change_reason：LLM 原因优先，模板多套轮换"""

    def test_llm_reason_used_when_visible(self):
        mood = _mood_with(joy=30)
        reason = build_mood_change_reason(
            "中立", 0.0, mood, "回味刚才的对话", subject="永雏塔菲",
            llm_reason="被夸了一句，有点开心",
        )
        self.assertEqual(reason, "被夸了一句，有点开心")

    def test_llm_reason_stripped(self):
        mood = _mood_with(joy=30)
        reason = build_mood_change_reason(
            "中立", 0.0, mood, "回味刚才的对话", subject="永雏塔菲",
            llm_reason="  被凶了一句，火气上来了  ",
        )
        self.assertEqual(reason, "被凶了一句，火气上来了")

    def test_llm_reason_gated_by_visibility(self):
        """不可见变化：LLM 原因也不记录（防弱波动把历史刷成废话）"""
        mood = _mood_with(joy=20)
        reason = build_mood_change_reason(
            "喜悦", 0.20, mood, "回味刚才的对话", subject="永雏塔菲",
            llm_reason="没什么变化",
        )
        self.assertIsNone(reason)

    def test_blank_llm_reason_falls_back_to_template(self):
        mood = _mood_with(joy=30)
        reason = build_mood_change_reason(
            "中立", 0.0, mood, "听到「哈哈」", subject="永雏塔菲",
            llm_reason="   ",
        )
        self.assertIsNotNone(reason)
        self.assertIn("听到「哈哈」", reason)
        self.assertIn("永雏塔菲", reason)

    def test_rise_templates_rotate_by_seed(self):
        """上升/翻转分支三套模板，按种子轮换且互异"""
        mood = _mood_with(joy=30)
        texts = [
            build_mood_change_reason(
                "中立", 0.0, mood, "听到「哈哈」", subject="永雏塔菲",
                variant_seed=s,
            )
            for s in range(3)
        ]
        self.assertEqual(len(set(texts)), 3)
        for t in texts:
            self.assertIn("听到「哈哈」", t)
            self.assertIn("永雏塔菲", t)
            self.assertIn("开心起来", t)

    def test_fallback_templates_include_calm_phrase(self):
        """回落分支：两句通用 + 一句按主导情绪的具体描述"""
        mood = _mood_with(anger=40)
        texts = [
            build_mood_change_reason(
                "愤怒", 0.60, mood, "回味刚才的对话", subject="永雏塔菲",
                variant_seed=s,
            )
            for s in range(3)
        ]
        self.assertEqual(len(set(texts)), 3)
        self.assertIn("回味刚才的对话之后，永雏塔菲的心情渐渐平复", texts)
        self.assertIn("回味刚才的对话之后，永雏塔菲的情绪慢慢缓和下来", texts)
        self.assertIn("回味刚才的对话之后，永雏塔菲火气慢慢降了下来", texts)
        for t in texts:
            self.assertNotIn("因为", t)  # 不把回落算到触发消息头上

    def test_default_seed_follows_emotion_sum(self):
        """不传种子时取 8 维情绪之和：心情变、模板就跟着换"""
        mood_a = _mood_with(anger=80)
        mood_b = _mood_with(anger=40)
        text_a = build_mood_change_reason("愤怒", 0.90, mood_a, "回味刚才的对话", subject="永雏塔菲")
        text_b = build_mood_change_reason("愤怒", 0.50, mood_b, "回味刚才的对话", subject="永雏塔菲")
        # 80 % 3 = 2（具体描述），40 % 3 = 1（缓和），两句必然不同
        self.assertNotEqual(text_a, text_b)
        self.assertIn("火气慢慢降了下来", text_a)
        self.assertIn("情绪慢慢缓和下来", text_b)


class TestVariantSeedAndExpertTrigger(unittest.TestCase):
    """轮换种子与专家通道触发语"""

    def test_seed_is_emotion_sum(self):
        mood = GlobalMood(emotions=EmotionalMetrics(joy=10, anger=5, trust=3))
        self.assertEqual(mood_variant_seed(mood), 18)

    def test_seed_changes_with_emotions(self):
        self.assertNotEqual(
            mood_variant_seed(_mood_with(anger=80)),
            mood_variant_seed(_mood_with(anger=73)),
        )

    def test_trigger_has_multiple_distinct_variants(self):
        variants = {expert_mood_trigger(s) for s in range(8)}
        self.assertEqual(len(variants), len(_EXPERT_MOOD_TRIGGERS))
        self.assertGreaterEqual(len(variants), 3)
        for v in variants:
            self.assertTrue(v.strip())

    def test_trigger_deterministic_by_seed(self):
        self.assertEqual(expert_mood_trigger(100.0), expert_mood_trigger(100.0))
        self.assertEqual(
            expert_mood_trigger(100.0),
            _EXPERT_MOOD_TRIGGERS[100 % len(_EXPERT_MOOD_TRIGGERS)],
        )

    def test_trigger_bad_seed_falls_back_to_first(self):
        self.assertEqual(expert_mood_trigger(None), _EXPERT_MOOD_TRIGGERS[0])
        self.assertEqual(expert_mood_trigger("坏种子"), _EXPERT_MOOD_TRIGGERS[0])


class TestExpertParsesMoodReason(unittest.TestCase):
    """专家 LLM：提示词要、解析收、开关能关"""

    def test_mood_reason_parsed_and_sanitized(self):
        e = _expert()
        updates = e._parse_emotion_analysis(
            _llm_json(mood_reason="  被夸了一句，有点开心  "), _state(),
        )
        self.assertEqual(updates["mood_reason"], "被夸了一句，有点开心")

    def test_mood_reason_truncated_when_too_long(self):
        e = _expert()
        updates = e._parse_emotion_analysis(
            _llm_json(mood_reason="啊" * 40), _state(),
        )
        self.assertEqual(len(updates["mood_reason"]), 25)  # 24 字 + 省略号
        self.assertTrue(updates["mood_reason"].endswith("…"))

    def test_mood_reason_absent_when_switch_off(self):
        """开关关闭：不写这个键，上游走内置模板兜底"""
        e = _expert(enable_llm_mood_reason=False)
        updates = e._parse_emotion_analysis(
            _llm_json(mood_reason="被夸了一句，有点开心"), _state(),
        )
        self.assertNotIn("mood_reason", updates)

    def test_mood_reason_absent_when_field_missing(self):
        e = _expert()
        updates = e._parse_emotion_analysis(_llm_json(), _state())
        self.assertNotIn("mood_reason", updates)

    def test_mood_reason_absent_when_not_string(self):
        e = _expert()
        updates = e._parse_emotion_analysis(_llm_json(mood_reason=123), _state())
        self.assertNotIn("mood_reason", updates)

    def test_mood_reason_absent_when_blank(self):
        e = _expert()
        updates = e._parse_emotion_analysis(_llm_json(mood_reason="   "), _state())
        self.assertNotIn("mood_reason", updates)

    def test_prompt_asks_for_mood_reason(self):
        e = _expert()
        prompt = e._build_emotion_analysis_prompt("你好", "你好呀", _state())
        self.assertIn("mood_reason", prompt)
        self.assertIn("心情变化描述要求", prompt)

    def test_text_fallback_path_has_no_mood_reason(self):
        """JSON 解析失败的文本提取兜底也不产生 mood_reason（模板兜底）"""
        e = _expert()
        updates = e._parse_emotion_analysis("这不是JSON，就是一段闲聊", _state())
        self.assertNotIn("mood_reason", updates)


class TestUpdateGlobalMoodReasonSource(unittest.TestCase):
    """_update_global_mood：LLM 原因 / 模板兜底 / 可见性门槛"""

    def _plugin(self, mood):
        plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
        plugin.config = PluginConfig()
        plugin._mood_cache = mood
        plugin._mood_cache_time = time.time()
        plugin._get_bot_name = lambda: "永雏塔菲"
        plugin.get_mood_sync = lambda: mood
        saved = []

        class _Store:
            async def save(self, m):
                saved.append(m)

        plugin.global_mood_store = _Store()
        return plugin, saved

    def _run(self, plugin, **kw):
        async def go():
            plugin._update_global_mood(**kw)
            await asyncio.sleep(0)  # 让 create_task 的 save 跑掉

        asyncio.run(go())

    def test_llm_reason_recorded_on_visible_change(self):
        mood = _mood_with(anger=95)  # 95→87，强度跌 0.08，可见
        plugin, saved = self._plugin(mood)
        self._run(
            plugin, updates={}, allow_spike=False, trigger="回味刚才的对话",
            llm_reason="被凶了一句，火气上来了",
        )
        self.assertEqual(mood.mood_history[0]["reason"], "被凶了一句，火气上来了")
        self.assertEqual(mood.last_change_reason, "被凶了一句，火气上来了")
        self.assertEqual(len(saved), 1)

    def test_llm_reason_not_recorded_on_invisible_change(self):
        mood = _mood_with(anger=8)  # 8→7，强度跌 0.01，不可见
        plugin, _ = self._plugin(mood)
        self._run(
            plugin, updates={}, allow_spike=False, trigger="回味刚才的对话",
            llm_reason="没什么变化",
        )
        self.assertEqual(mood.mood_history, [])
        self.assertEqual(mood.last_change_reason, "")

    def test_template_used_when_no_llm_reason(self):
        mood = _mood_with(anger=95)
        plugin, _ = self._plugin(mood)
        self._run(plugin, updates={}, allow_spike=False, trigger="回味刚才的对话")
        reason = mood.mood_history[0]["reason"]
        self.assertIn("回味刚才的对话", reason)
        self.assertIn("永雏塔菲", reason)

    def test_non_string_llm_reason_ignored(self):
        """防御：上游传来非字符串 llm_reason 不炸，退回模板"""
        mood = _mood_with(anger=95)
        plugin, _ = self._plugin(mood)
        self._run(
            plugin, updates={}, allow_spike=False, trigger="回味刚才的对话",
            llm_reason=123,
        )
        self.assertIn("回味刚才的对话", mood.mood_history[0]["reason"])


class TestProductionScenarioAccumulates(unittest.TestCase):
    """线上场景复刻与修复验证

    专家通道连续多轮（无 LLM 原因、同触发语池）：旧实现把同一句话
    无限合并成一条历史；现在历史应攒满 3 条且文本互异。
    """

    def test_expert_rounds_accumulate_distinct_history(self):
        mood = _mood_with(anger=95)
        plugin, _ = _mood_plugin(mood)
        clock = {"t": 1_000_000.0}
        fake_time = types.SimpleNamespace(time=lambda: clock["t"])

        async def go():
            for _ in range(5):
                # 每轮间隔 10 分 1 秒：跨过去重窗口，又不够一个时间衰减单位
                clock["t"] += 601
                plugin._update_global_mood(
                    {}, allow_spike=False,
                    trigger=expert_mood_trigger(clock["t"]),
                )
                await asyncio.sleep(0)

        with mock.patch("emotionai_pro.global_mood.time", fake_time):
            asyncio.run(go())

        self.assertEqual(len(mood.mood_history), MOOD_HISTORY_LIMIT)
        reasons = [e["reason"] for e in mood.mood_history]
        self.assertEqual(
            len(set(reasons)), MOOD_HISTORY_LIMIT,
            f"历史文本应互异，实际: {reasons}",
        )
        ats = [e["at"] for e in mood.mood_history]
        self.assertEqual(ats, sorted(ats, reverse=True))  # 新的在前
        self.assertEqual(mood.last_change_reason, reasons[0])
        self.assertEqual(mood.last_change_at, ats[0])


class TestConfigWiring(unittest.TestCase):
    """新配置字段三处同步 + 主程序接线（防死配置）"""

    def test_pydantic_default_on(self):
        self.assertTrue(PluginConfig().enable_llm_mood_reason)

    def test_schema_entry_matches(self):
        root = Path(__file__).resolve().parent.parent
        schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
        entry = schema.get("enable_llm_mood_reason")
        self.assertIsInstance(entry, dict)
        self.assertEqual(entry.get("type"), "bool")
        self.assertTrue(entry.get("default"))
        self.assertTrue(str(entry.get("hint", "")).strip())

    def test_validator_schema_and_default_config(self):
        props = ConfigValidator.CONFIG_SCHEMA["properties"]
        self.assertIn("enable_llm_mood_reason", props)
        self.assertEqual(props["enable_llm_mood_reason"], {"type": "boolean"})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            self.assertTrue(ConfigValidator.create_default_config(path))
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(data["enable_llm_mood_reason"])
            # 默认配置必须能过校验（schema 顶层 additionalProperties=False）
            self.assertTrue(ConfigValidator.validate_config_file(path))

    def test_main_wires_field_to_expert(self):
        root = Path(__file__).resolve().parent.parent
        src = (root / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            "enable_llm_mood_reason=self.config.enable_llm_mood_reason", src,
            "main.py 没把开关传给情感分析专家（死配置）",
        )
        self.assertIn(
            '"enable_llm_mood_reason": "enable_llm_mood_reason"', src,
            "base_mapping 白名单漏了 enable_llm_mood_reason",
        )

    def test_expert_ctor_accepts_switch(self):
        e = _expert(enable_llm_mood_reason=False)
        self.assertFalse(e.enable_llm_mood_reason)
        self.assertTrue(_expert().enable_llm_mood_reason)


if __name__ == "__main__":
    unittest.main()
