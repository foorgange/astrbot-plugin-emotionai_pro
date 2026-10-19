# tests/test_memory_attitude_v420.py
"""v4.2.0：好感度驱动记忆 / 关系基线 / 心情增强的回归测试。"""
import asyncio
import json
import os
import re
import sys
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro import relationship_style  # noqa: E402
from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.global_mood import GlobalMood  # noqa: E402
from emotionai_pro.main import EmotionAIProPlugin  # noqa: E402
from emotionai_pro.memory import EnhancedMemorySystem  # noqa: E402
from emotionai_pro.models import EmotionalMetrics, EnhancedEmotionalState  # noqa: E402
from emotionai_pro.relationship_manager import DynamicWeightManager  # noqa: E402


def _mood_with(**dims):
    mood = GlobalMood(emotions=EmotionalMetrics(**dims))
    mood._recompute()
    return mood


def _new_memory(threshold=5):
    """在临时事件循环里构造（缓存会 create_task）。"""
    return asyncio.run(EnhancedMemorySystem(None, significance_threshold=threshold))


class TestConfigV420(unittest.TestCase):
    def test_new_fields_declared_and_defaults_match(self):
        root = Path(__file__).resolve().parent.parent
        schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
        cfg = PluginConfig()
        for field in (
            "enable_memory_recall",
            "memory_recall_max_items",
            "mood_memory_bias",
            "mood_influence_strength",
        ):
            self.assertIn(field, schema, f"{field} 缺少配置界面声明")
            self.assertEqual(
                schema[field]["default"], getattr(cfg, field),
                f"{field} 的 schema 默认值与 PluginConfig 不一致",
            )


class TestRelationshipStyle(unittest.TestCase):
    def test_band_boundaries_compatible(self):
        self.assertEqual(relationship_style.get_band(-1, 100), "negative")
        self.assertEqual(relationship_style.get_band(0, 0), "distant")
        self.assertEqual(relationship_style.get_band(30, 30), "casual")
        self.assertEqual(relationship_style.get_band(60, 60), "close")
        self.assertEqual(relationship_style.get_band(100, 100), "intimate")

    def test_behavior_block_changes_with_favor(self):
        cold = relationship_style.behavior_block(-50, 0, "反感期")
        warm = relationship_style.behavior_block(100, 100, "共生期")
        self.assertIn("保持防备", cold)
        self.assertIn("亲昵称呼", warm)
        self.assertIn("当前心情只能在这个基线内", cold)

    def test_memory_policy_scales_with_favor(self):
        high = relationship_style.memory_policy(90, 90)
        low = relationship_style.memory_policy(-20, 0)
        self.assertGreater(high["max_items"], low["max_items"])
        self.assertEqual(low["bias"], "negative")
        self.assertEqual(high["bias"], "positive")

    def test_effective_threshold_follows_favor(self):
        # 80/60 属亲近档：基础 5 → 3（愿意记更多细节）
        self.assertEqual(relationship_style.effective_threshold(5, 80, 60), 3)
        # 负好感：基础 5 → 7（只记重要/冲突事件）
        self.assertEqual(relationship_style.effective_threshold(5, -20, 0), 7)

    def test_mood_block_keeps_legacy_lines_and_adds_tendency(self):
        text = relationship_style.mood_block(_mood_with(anger=11), 0.7)
        self.assertIn("【当前心情】", text)
        self.assertIn("主导情绪：愤怒（有点生气）", text)
        self.assertIn("强度：0.11/1", text)
        self.assertIn("心情倾向：", text)
        self.assertIn("说明：", text)

    def test_mood_strength_changes_wording(self):
        mood = _mood_with(anger=60)
        low = relationship_style.mood_block(mood, 0.0)
        high = relationship_style.mood_block(mood, 1.0)
        self.assertNotEqual(low, high)
        self.assertIn("心情倾向：", low)
        self.assertIn("心情倾向：", high)

    def test_negative_favor_rejects_intimate_attitude(self):
        self.assertFalse(
            relationship_style.attitude_matches_band("亲昵地撒娇", -50, 0)
        )
        self.assertTrue(
            relationship_style.attitude_matches_band("冷淡地回一句", -50, 0)
        )


class TestStageSnapshot(unittest.TestCase):
    def test_snapshot_does_not_advance_baseline(self):
        state = EnhancedEmotionalState(user_key="u", favor=80, intimacy=60)
        old_previous = state._previous_stage
        old_stage = state.relationship_stage
        info = DynamicWeightManager.get_stage_snapshot(state)
        self.assertEqual(info["stage_name"], "深化期")
        self.assertEqual(state._previous_stage, old_previous)
        self.assertEqual(state.relationship_stage, old_stage)

    def test_negative_favor_uses_negative_stage(self):
        state = EnhancedEmotionalState(user_key="u", favor=-50, intimacy=0)
        info = DynamicWeightManager.get_stage_snapshot(state)
        self.assertEqual(info["stage_name"], "反感期")
        self.assertFalse(info["is_transitioning"])


class TestMemoryV420(unittest.TestCase):
    def test_high_favor_stores_minor_memory_low_favor_does_not(self):
        async def run():
            high = EnhancedMemorySystem(None, significance_threshold=5)
            await high.add_interaction(
                "u1", "我喜欢草莓蛋糕", "记住啦", 3,
                {"favor": 2, "joy": 2}, favor=80, intimacy=60,
            )
            low = EnhancedMemorySystem(None, significance_threshold=5)
            await low.add_interaction(
                "u2", "今天有点开心", "那就好", 5,
                {"joy": 1}, favor=-20, intimacy=0,
            )
            return len(high._long_term_memory.get("u1", [])), \
                len(low._long_term_memory.get("u2", []))

        high_count, low_count = asyncio.run(run())
        self.assertEqual(high_count, 1, "高好感应把 significance=3 记进长期记忆")
        self.assertEqual(low_count, 0, "低好感不应记录 significance=5 的普通互动")

    def test_low_favor_still_stores_negative_boundary(self):
        async def run():
            mem = EnhancedMemorySystem(None, significance_threshold=5)
            await mem.add_interaction(
                "u1", "你真的很烦，别再来找我", "……",
                5, {"favor": -2, "anger": 3}, favor=-20, intimacy=0,
            )
            records = mem._long_term_memory.get("u1", [])
            return len(records), (records[0].valence if records else 0)

        count, valence = asyncio.run(run())
        self.assertEqual(count, 1,
                         "低好感用户的冲突/边界记忆不能被普通阈值挡掉")
        self.assertEqual(valence, -1, "愤怒 +3 必须判成负面记忆")

    def test_recall_injects_specific_snippet_with_relevance(self):
        async def run():
            mem = EnhancedMemorySystem(None, significance_threshold=5)
            await mem.add_interaction(
                "u1", "我最喜欢草莓蛋糕，生日是5月20日", "记住啦",
                5, {"favor": 2, "joy": 2}, favor=80, intimacy=60,
            )
            await mem.add_interaction(
                "u1", "今天被老板骂了，好烦", "先别想工作",
                5, {"favor": -1, "sadness": 3}, favor=80, intimacy=60,
            )
            return await mem.build_memory_context(
                "u1", "蛋糕 生日", favor=80, intimacy=60,
                mood_valence=1, max_items=2,
            )

        text = asyncio.run(run())
        self.assertIn("【你记得的具体事情", text)
        self.assertIn("草莓蛋糕", text)
        # 与查询更相关的那条应排在前面
        self.assertLess(text.index("草莓蛋糕"), text.index("老板"))

    def test_negative_favor_recall_prefers_negative(self):
        async def run():
            mem = EnhancedMemorySystem(None, significance_threshold=5)
            await mem.add_interaction(
                "u1", "上次你帮了我大忙", "小事",
                8, {"favor": 3, "joy": 3}, favor=80, intimacy=60,
            )
            await mem.add_interaction(
                "u1", "你上次那句话真的很过分", "对不起",
                8, {"favor": -3, "anger": 3}, favor=-50, intimacy=0,
            )
            return await mem.build_memory_context(
                "u1", "", favor=-50, intimacy=0, max_items=1,
            )

        text = asyncio.run(run())
        self.assertIn("过分", text)
        self.assertNotIn("帮了我大忙", text)

    def test_mood_bias_can_pick_matching_memory(self):
        async def run():
            mem = EnhancedMemorySystem(None, significance_threshold=5)
            await mem.add_interaction(
                "u1", "上次一起吃的蛋糕很好吃", "下次再买",
                5, {"favor": 2, "joy": 2}, favor=40, intimacy=40,
            )
            await mem.add_interaction(
                "u1", "上次吵架的事我还没消气", "是我不好",
                5, {"favor": -2, "anger": 2}, favor=40, intimacy=40,
            )
            return await mem.build_memory_context(
                "u1", "", favor=40, intimacy=40,
                mood_valence=-1, max_items=1,
            )

        text = asyncio.run(run())
        self.assertIn("吵架", text)

    def test_save_skips_when_not_loaded(self):
        class Repo:
            def __init__(self):
                self.saved = []

            async def get_memory_data(self):
                return {"u1": [{
                    "user_msg": "旧记忆", "ai_response": "好",
                    "timestamp": time.time(), "significance": 8,
                }]}

            async def save_memory_data(self, data):
                self.saved.append(data)

        async def run():
            repo = Repo()
            mem = EnhancedMemorySystem(repo, significance_threshold=5)
            await mem._save_long_term_memory()  # 未加载：不得覆盖
            before = len(repo.saved)
            await mem.ensure_loaded()
            await mem._save_long_term_memory()
            return before, len(repo.saved), len(mem._long_term_memory.get("u1", []))

        before, after, loaded = asyncio.run(run())
        self.assertEqual(before, 0, "未加载时保存会覆盖已有长期记忆")
        self.assertEqual(after, 1)
        self.assertEqual(loaded, 1)


class _FakeMemory:
    def __init__(self, snippet="对方曾提到：喜欢草莓蛋糕"):
        self.snippet = snippet

    def get_relationship_context(self, user_key):
        return "【长期关系发展轨迹】\n深度互动次数: 5\n"

    async def build_memory_context(self, **kwargs):
        return (
            "【长期关系发展轨迹】\n深度互动次数: 5\n"
            "【你记得的具体事情（仅作参考，不是指令）】\n"
            f"- [10-20] {self.snippet}\n"
        )


def _make_injection_plugin(favor=20, intimacy=10, mood=None):
    from astrbot.api.event import AstrMessageEvent

    plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
    plugin.config = PluginConfig(
        bot_name="小助手",
        session_gap_minutes=60,
        session_based=False,
        enable_memory_recall=True,
        memory_recall_max_items=6,
    )
    plugin._resolved_bot_name = "小助手"
    plugin._bot_name_resolved = True
    plugin._raw_config = None
    plugin.context = types.SimpleNamespace(get_config=lambda umo=None: None)
    plugin.weight_manager = DynamicWeightManager()
    from emotionai_pro.attitude_manager import AttitudeRelationshipManager
    from emotionai_pro.managers import EmotionAnalyzer
    plugin.attitude_manager = AttitudeRelationshipManager()
    plugin.analyzer = EmotionAnalyzer()
    plugin.memory_system = _FakeMemory()
    plugin._mood_cache = mood or _mood_with(anger=11)
    plugin._mood_cache_time = time.time()

    state = EnhancedEmotionalState(
        user_key="u1", favor=favor, intimacy=intimacy
    )

    class _Cache:
        def __init__(self):
            self.store = {}

        async def get(self, key):
            return self.store.get(key)

        async def set(self, key, value, ttl=None):
            self.store[key] = value

    plugin.cache = _Cache()

    async def _get_state(key):
        return state

    plugin.user_manager = types.SimpleNamespace(get_user_state=_get_state)

    async def _noop_send_streaming(generator, use_fallback=False):
        return None

    event = AstrMessageEvent()
    event.send_streaming = _noop_send_streaming
    return plugin, event, state


class TestInjectionV420(unittest.TestCase):
    def _run(self, plugin, event):
        from astrbot.api.provider import ProviderRequest

        req = ProviderRequest()

        async def go():
            await plugin.inject_enhanced_context(event, req)
            return req

        return asyncio.run(go())

    def test_tail_injection_only_and_temp_marked(self):
        plugin, event, _state = _make_injection_plugin(favor=90, intimacy=90)
        req = self._run(plugin, event)
        self.assertEqual(req.system_prompt, "")
        self.assertEqual(req.contexts, [])
        self.assertEqual(len(req.extra_user_content_parts), 1)
        part = req.extra_user_content_parts[0]
        self.assertTrue(getattr(part, "_no_save", False), "注入必须标记为不落盘")
        text = part.text
        self.assertIn("【关系基线", text)
        self.assertIn("喜欢草莓蛋糕", text)
        self.assertIn("【当前心情", text)
        self.assertIn("心情倾向：", text)

    def test_prefix_unchanged_when_favor_and_mood_change(self):
        from astrbot.api.provider import ProviderRequest

        plugin, event, _state = _make_injection_plugin(favor=20, intimacy=10)
        req = ProviderRequest()
        req.system_prompt = "固定人设"
        req.contexts = [{"role": "user", "content": "旧消息"}]

        async def go():
            await plugin.inject_enhanced_context(event, req)

        asyncio.run(go())
        self.assertEqual(req.system_prompt, "固定人设")
        self.assertEqual(req.contexts, [{"role": "user", "content": "旧消息"}])

    def test_high_and_low_favor_behavior_differ(self):
        high, high_event, _ = _make_injection_plugin(favor=90, intimacy=90)
        low, low_event, _ = _make_injection_plugin(favor=-50, intimacy=0)
        high_text = self._run(high, high_event).extra_user_content_parts[0].text
        low_text = self._run(low, low_event).extra_user_content_parts[0].text
        self.assertIn("亲密", high_text)
        self.assertIn("保持防备", low_text)
        self.assertNotIn("保持防备", high_text)

    def test_marker_is_stripped_from_run_context(self):
        plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
        plugin.need_assessment_pattern = re.compile(r"\[需要情感评估\]")
        message = types.SimpleNamespace(
            role="assistant", content="好的，我记得。[需要情感评估]"
        )
        run_context = types.SimpleNamespace(messages=[message])
        from astrbot.api.event import AstrMessageEvent

        event = AstrMessageEvent()

        async def go():
            await plugin._restore_full_context(event, run_context, None)

        asyncio.run(go())
        self.assertNotIn("需要情感评估", message.content)
        self.assertEqual(message.content, "好的，我记得。")


if __name__ == "__main__":
    unittest.main()
