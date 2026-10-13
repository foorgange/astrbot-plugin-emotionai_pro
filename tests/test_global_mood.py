# tests/test_global_mood.py
"""全局心情（共享字段）离线单测"""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.bootstrap  # noqa: F401  (注册 emotionai_pro 包别名)

from emotionai_pro.global_mood import (
    GlobalMood, GlobalMoodStore, apply_mood_update, compute_mood_signal,
    MOOD_SIGNAL_CAP, MOOD_SPIKE_CAP,
    build_mood_change_reason, preview_time_decayed, compute_time_decay_factor,
    mood_trigger_excerpt,
    MOOD_TIME_DECAY, MOOD_TIME_DECAY_MAX_UNITS,
)
from emotionai_pro.models import EmotionalMetrics


class TestGlobalMoodUpdate(unittest.TestCase):
    def test_apply_update_positive(self):
        """正向更新：joy 上升，负面被对冲压制"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=20, trust=10))
        mood._recompute()

        apply_mood_update(mood, {"joy": 3, "trust": 2})
        # 信号强度 5 ≥ MOOD_SPIKE_MIN → 冲高：正面赢，joy 大幅上升，负面被压制
        self.assertGreater(mood.emotions.joy, 20)
        self.assertLess(mood.emotions.anger, 10)
        # 强度在 (0,1] 之间
        self.assertGreater(mood.intensity, 0)
        self.assertLessEqual(mood.intensity, 1.0)

        # 再应用大幅正向更新，验证 clamp
        apply_mood_update(mood, {"joy": 3, "trust": 3, "anticipation": 3})
        self.assertLessEqual(mood.emotions.joy, 100)
        self.assertLessEqual(mood.emotions.trust, 100)

        # 验证 updated_at 被更新
        self.assertGreater(mood.updated_at, 0)

    def test_apply_update_negative_flips_mood(self):
        """负向更新：joy 被对冲压制，最强负面维度冲高 → 主导翻转"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=50, trust=40))
        mood._recompute()
        self.assertEqual(mood.dominant_emotion, "喜悦")

        apply_mood_update(mood, {"disgust": 3, "anger": 3})
        # 衰减后 joy 46, trust 36；负面赢：最强维度(disgust)冲高到 35（单次上限），
        # 正面维度 ×0.2 → joy 9 / trust 7，主导翻到厌恶
        self.assertGreaterEqual(mood.emotions.disgust, MOOD_SPIKE_CAP)
        self.assertLess(mood.emotions.joy, 20)
        self.assertGreater(mood.emotions.disgust, mood.emotions.joy)
        self.assertNotEqual(mood.dominant_emotion, "喜悦")
        self.assertGreater(mood.intensity, 0.3)

    def test_dominant_single_dimension(self):
        """冲高只作用于单一最强维度，不产生复合情绪"""
        mood = GlobalMood()
        mood._recompute()
        apply_mood_update(mood, {"joy": 3, "trust": 2, "anticipation": 2})
        # 正面赢：只冲高 joy（最强），trust/anticipation 不冲高
        self.assertGreater(mood.emotions.joy, mood.emotions.trust)
        self.assertEqual(mood.dominant_emotion, "喜悦")

    def test_weak_signal_gentle(self):
        """弱信号（<MOOD_SPIKE_MIN）：温和叠加，不触发冲高"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=20, anger=5))
        mood._recompute()
        apply_mood_update(mood, {"joy": 1})
        # 先衰减 joy 20→18（MOOD_DECAY=0.92），弱信号 +1 → 19（不冲高）
        self.assertEqual(mood.emotions.joy, 19)
        self.assertLessEqual(mood.emotions.anger, 5)
        self.assertGreater(mood.intensity, 0)

    def test_dominant_and_intensity_0to1(self):
        """强度为 0~1，主导情感正确"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=100, trust=50))
        mood._recompute()
        self.assertEqual(mood.intensity, 1.0)
        self.assertEqual(mood.dominant_emotion, "喜悦")

        mood2 = GlobalMood(emotions=EmotionalMetrics(joy=30, anger=60))
        mood2._recompute()
        self.assertEqual(mood2.intensity, 0.6)
        self.assertEqual(mood2.dominant_emotion, "愤怒")

    def test_zero_emotions_gives_default_dominant(self):
        """全零心情 → 中立"""
        mood = GlobalMood()
        mood._recompute()
        self.assertEqual(mood.dominant_emotion, "中立")
        self.assertEqual(mood.intensity, 0.0)


class TestGlobalMoodStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmpdir)

    def test_save_load_roundtrip(self):
        """save → load 往返一致"""
        store = GlobalMoodStore(self.data_dir)
        mood = GlobalMood(emotions=EmotionalMetrics(joy=66, trust=44))
        mood._recompute()

        asyncio.run(store.save(mood))
        loaded = asyncio.run(store.load())

        self.assertEqual(loaded.emotions.joy, 66)
        self.assertEqual(loaded.emotions.trust, 44)
        self.assertEqual(loaded.dominant_emotion, mood.dominant_emotion)
        self.assertEqual(loaded.intensity, mood.intensity)

    def test_load_corrupted_returns_default(self):
        """损坏文件 → 默认心情，不抛异常"""
        from emotionai_pro.constants import PathConstants
        file = self.data_dir / PathConstants.GLOBAL_MOOD_FILE
        file.write_text("{invalid json!!", encoding="utf-8")

        store = GlobalMoodStore(self.data_dir)
        mood = asyncio.run(store.load())
        self.assertEqual(mood.dominant_emotion, "中立")
        self.assertEqual(mood.intensity, 0)

    def test_load_missing_returns_default(self):
        """文件不存在 → 默认心情"""
        store = GlobalMoodStore(self.data_dir)
        mood = asyncio.run(store.load())
        self.assertEqual(mood.dominant_emotion, "中立")


class TestComputeMoodSignal(unittest.TestCase):
    def test_positive_keywords(self):
        sig = compute_mood_signal("我好喜欢你，真开心！")
        self.assertGreaterEqual(sig.get("joy", 0), 1)
        self.assertGreaterEqual(sig.get("trust", 0), 1)

    def test_negative_keywords(self):
        sig = compute_mood_signal("我讨厌你，真烦！")
        self.assertGreaterEqual(sig.get("disgust", 0), 1)
        self.assertGreaterEqual(sig.get("anger", 0), 1)

    def test_question_adds_surprise(self):
        sig = compute_mood_signal("真的吗？")
        self.assertGreaterEqual(sig.get("surprise", 0), 1)

    def test_exclamation_amplifies_negative(self):
        # "生气" 命中 anger；感叹号在负面语境下不再给 joy，而是加剧
        sig = compute_mood_signal("我生气了！")
        self.assertEqual(sig.get("joy", 0), 0)
        self.assertGreaterEqual(sig.get("anger", 0), 1)

    def test_exclamation_positive(self):
        sig = compute_mood_signal("好棒啊！")
        self.assertGreaterEqual(sig.get("joy", 0), 1)

    def test_emoji(self):
        self.assertGreaterEqual(compute_mood_signal("哈哈😄").get("joy", 0), 1)
        self.assertGreaterEqual(compute_mood_signal("呜呜😭").get("sadness", 0), 1)

    def test_empty_or_no_signal_returns_empty(self):
        self.assertEqual(compute_mood_signal(""), {})
        self.assertEqual(compute_mood_signal("随便聊聊"), {})
        self.assertEqual(compute_mood_signal(None), {})

    def test_cap_on_strong_negative(self):
        # 多个负面词叠加，单维净增量被 cap
        sig = compute_mood_signal("讨厌！滚！垃圾！恨你！")
        self.assertLessEqual(sig.get("disgust", 0), MOOD_SIGNAL_CAP)
        self.assertLessEqual(sig.get("anger", 0), MOOD_SIGNAL_CAP)

    def test_signal_keeps_mood_changeable_over_time(self):
        """多次轻量信号叠加，心情逐步变化"""
        mood = GlobalMood()
        mood._recompute()
        self.assertEqual(mood.dominant_emotion, "中立")
        for _ in range(3):
            apply_mood_update(mood, compute_mood_signal("好开心！"))
        self.assertNotEqual(mood.dominant_emotion, "中立")


class TestMoodAmplitudeV414(unittest.TestCase):
    """v4.1.4 心情幅度重调：实时可见、单条有界、随时间平复

    背景：心情是高频波动量（同一天可变几次），但又不是开关——
    单条消息不允许打满，冲上高强度档位需要持续性互动。
    """

    def test_mild_message_visible(self):
        """普通一句话（"谢谢"级，信号强度 2）也应把强度推到看得见的档位"""
        mood = GlobalMood()
        apply_mood_update(mood, compute_mood_signal("谢谢塔菲"))
        self.assertEqual(compute_mood_signal("谢谢塔菲"), {"trust": 1, "joy": 1})
        # 强度 2 ≥ MOOD_SPIKE_MIN → 冲高 20 → 强度 0.20（超出"心情平静"档）
        self.assertGreaterEqual(mood.intensity, 0.15)

    def test_single_message_bounded(self):
        """单条消息再激烈也不打满：强度 ≤ MOOD_SPIKE_CAP/100"""
        mood = GlobalMood()
        apply_mood_update(mood, {"anger": 4, "disgust": 4, "sadness": 4})
        self.assertLessEqual(mood.intensity, MOOD_SPIKE_CAP / 100.0)

    def test_sustained_interaction_reaches_high(self):
        """持续性互动才能冲上"情绪高涨"（不是一条消息）"""
        mood = GlobalMood()
        for _ in range(4):
            apply_mood_update(mood, compute_mood_signal("塔菲好可爱！"))
        self.assertGreater(mood.intensity, 0.7)

    def test_expert_channel_no_spike(self):
        """专家评估走温和通道（allow_spike=False）：总分再高也不冲高"""
        mood = GlobalMood()
        apply_mood_update(mood, {"joy": 3, "trust": 3}, allow_spike=False)
        self.assertLessEqual(mood.emotions.joy, 5)
        # 同一信号若允许冲高（实时通道）则显著更大
        mood2 = GlobalMood()
        apply_mood_update(mood2, {"joy": 3, "trust": 3}, allow_spike=True)
        self.assertGreater(mood2.emotions.joy, mood.emotions.joy + 20)

    def test_time_decay_regresses_to_neutral(self):
        """久无消息 → 心情按时间衰减回归中性（早上冲的高不滞留到深夜）"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=80, anger=60))
        mood._recompute()
        mood.updated_at = time.time() - 3 * 3600  # 3 小时前（= 6 个衰减单位）
        apply_mood_update(mood, {})  # 空信号也要衰减
        self.assertLess(mood.emotions.joy, 5)
        self.assertLess(mood.emotions.anger, 5)
        self.assertLess(mood.intensity, 0.05)

    def test_no_time_decay_without_timestamp(self):
        """updated_at=0（从未更新过）不做无依据的时间衰减"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=80))
        mood._recompute()
        apply_mood_update(mood, {})
        # 只剩常规消息衰减 0.92
        self.assertGreaterEqual(mood.emotions.joy, 70)

    def test_mood_swings_multiple_times_a_day(self):
        """核心场景：同一天内心情因几句话变好几次，且段间自行平复"""
        mood = GlobalMood()
        # 早上：被夸 → 喜悦冲高
        mood.updated_at = time.time() - 60
        apply_mood_update(mood, compute_mood_signal("塔菲太可爱了！"))
        first = mood.dominant_emotion
        self.assertGreater(mood.intensity, 0.2)

        # 中午：2 小时沉默 → 平复到接近中立
        mood.updated_at = time.time() - 2 * 3600
        apply_mood_update(mood, {})
        self.assertLess(mood.intensity, 0.05)

        # 下午：被骂 → 主导翻到负面
        apply_mood_update(mood, compute_mood_signal("讨厌死了滚啊"))
        second = mood.dominant_emotion
        self.assertNotEqual(second, first)
        self.assertGreater(mood.intensity, 0.2)

        # 深夜：又哄开心 → 再翻回正面
        mood.updated_at = time.time() - 3 * 3600
        apply_mood_update(mood, {})
        apply_mood_update(mood, compute_mood_signal("给你贴贴么么"))
        third = mood.dominant_emotion
        self.assertNotEqual(third, second)

    def test_new_keywords_covered(self):
        """群聊常见表达进入关键词表（普通闲聊也能推动心情）"""
        self.assertGreaterEqual(compute_mood_signal("哈哈哈笑死我了").get("joy", 0), 4)
        self.assertGreaterEqual(compute_mood_signal("蚌埠住了").get("joy", 0), 2)
        self.assertGreaterEqual(compute_mood_signal("无语了").get("sadness", 0), 1)
        self.assertGreaterEqual(compute_mood_signal("今晚开播吗").get("anticipation", 0), 2)

    def test_negated_keyword_no_positive(self):
        """被否定的关键词不产生反向信号（"不喜欢"不是喜悦）"""
        sig = compute_mood_signal("我不喜欢这样")
        self.assertEqual(sig.get("joy", 0), 0)
        self.assertEqual(sig.get("trust", 0), 0)
        # "别生气"同样不计愤怒
        self.assertEqual(compute_mood_signal("别生气嘛").get("anger", 0), 0)


class TestMoodChangeReasonV415(unittest.TestCase):
    """「/心情」命令支撑：变化原因、衰减预览、消息摘录（v4.1.5）"""

    def test_reason_recorded_on_flip(self):
        """主导翻转 → 生成带人设名与触发来源的变化原因"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=50, trust=40))
        mood._recompute()
        before_dom, before_int = mood.dominant_emotion, mood.intensity
        apply_mood_update(mood, {"anger": 3})
        reason = build_mood_change_reason(
            before_dom, before_int, mood, "听到「气死」", subject="永雏塔菲",
        )
        self.assertIsNotNone(reason)
        self.assertIn("永雏塔菲", reason)
        self.assertIn("听到「气死」", reason)
        self.assertIn("生气起来", reason)  # 翻到愤怒

    def test_reason_without_subject(self):
        """无人设名时用「心情」作主语，句子仍完整"""
        mood = GlobalMood()
        mood._recompute()
        before_dom, before_int = mood.dominant_emotion, mood.intensity
        apply_mood_update(mood, {"joy": 3})
        reason = build_mood_change_reason(before_dom, before_int, mood, "听到「哈哈」")
        self.assertIn("心情变得", reason)
        self.assertNotIn("None", reason)

    def test_no_reason_on_invisible_change(self):
        """不可见波动（未翻转且强度变化 <0.05）→ 不生成新原因"""
        # 场景 1：弱信号 + 逐轮衰减，强度仅跌 0.01
        mood = GlobalMood(emotions=EmotionalMetrics(joy=20))
        mood._recompute()
        before_dom, before_int = mood.dominant_emotion, mood.intensity
        apply_mood_update(mood, {"joy": 1})  # joy 20→衰减 18→+1=19
        self.assertLess(abs(mood.intensity - before_int), 0.05)
        self.assertIsNone(
            build_mood_change_reason(before_dom, before_int, mood, "听到「好」")
        )

        # 场景 2：空信号（只有逐轮衰减），强度跌 0.03——弱消息的常规
        # 衰减带，同样不该覆盖已有原因
        mood2 = GlobalMood(emotions=EmotionalMetrics(joy=50))
        mood2._recompute()
        before_dom2, before_int2 = mood2.dominant_emotion, mood2.intensity
        apply_mood_update(mood2, {})  # joy 50→46，强度 0.50→0.46
        self.assertAlmostEqual(mood2.intensity, before_int2 - 0.04, places=2)
        self.assertIsNone(
            build_mood_change_reason(before_dom2, before_int2, mood2, "听到「在吗」")
        )

    def test_reason_on_intensity_drop(self):
        """强度大幅回落（未翻转）→ 心情渐渐平复描述（不归因于触发消息）"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=80))
        mood._recompute()
        before_dom, before_int = mood.dominant_emotion, mood.intensity
        mood.updated_at = time.time() - 3 * 3600  # 时间衰减主导：强度大降
        apply_mood_update(mood, {"joy": 1})
        self.assertLess(mood.intensity, before_int - 0.05)
        reason = build_mood_change_reason(
            before_dom, before_int, mood, "听到「喵」", subject="永雏塔菲",
        )
        self.assertIn("心情渐渐平复", reason)
        self.assertNotIn("因为", reason)  # 不把回落算到触发消息头上

    def test_reason_persistence_roundtrip(self):
        """last_change_reason/last_change_at 随 to_dict/from_dict 往返"""
        mood = GlobalMood()
        mood.last_change_reason = "因为听到「哈哈」，永雏塔菲变得开心起来"
        mood.last_change_at = 1234567890.0
        data = mood.to_dict()
        self.assertEqual(data["last_change_reason"], mood.last_change_reason)
        self.assertEqual(data["last_change_at"], 1234567890.0)
        restored = GlobalMood.from_dict(data)
        self.assertEqual(restored.last_change_reason, mood.last_change_reason)
        self.assertEqual(restored.last_change_at, 1234567890.0)

    def test_from_dict_missing_reason_fields(self):
        """旧存档（v4.1.5 之前）没有新字段 → 默认空，不抛异常"""
        mood = GlobalMood.from_dict({"emotions": {"joy": 10}, "updated_at": 1.0})
        self.assertEqual(mood.last_change_reason, "")
        self.assertEqual(mood.last_change_at, 0)

    def test_from_dict_corrupted_reason_fields(self):
        """新字段类型损坏 → 回退默认值，不抛异常"""
        mood = GlobalMood.from_dict({
            "emotions": {"joy": 10},
            "last_change_reason": 12345,
            "last_change_at": "不是时间",
        })
        self.assertEqual(mood.last_change_reason, "")
        self.assertEqual(mood.last_change_at, 0)

    def test_compute_time_decay_factor(self):
        """衰减系数：从未更新/刚更新=1，久未更新按单位衰减且有上限"""
        self.assertEqual(compute_time_decay_factor(0), 1.0)
        self.assertEqual(compute_time_decay_factor(time.time()), 1.0)
        # 1 小时 → 2 个单位 → 0.5^2
        self.assertAlmostEqual(compute_time_decay_factor(time.time() - 3600), 0.25)
        # 10 小时 → 超过 6 个单位上限 → 0.5^6
        self.assertAlmostEqual(
            compute_time_decay_factor(time.time() - 10 * 3600),
            MOOD_TIME_DECAY ** MOOD_TIME_DECAY_MAX_UNITS,
        )

    def test_preview_time_decayed_copies_without_mutation(self):
        """预览：按衰减复制心情，原存档不被修改，原因字段随预览带出"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=80, trust=40))
        mood._recompute()
        mood.updated_at = time.time() - 2 * 3600  # 4 个单位 → ×0.5^4=0.0625
        mood.last_change_reason = "因为听到「哈哈」，永雏塔菲变得开心起来"
        preview = preview_time_decayed(mood)
        self.assertIsNot(preview, mood)
        self.assertEqual(preview.emotions.joy, int(80 * 0.0625))   # 5
        self.assertEqual(preview.emotions.trust, int(40 * 0.0625))  # 2
        self.assertEqual(preview.last_change_reason, mood.last_change_reason)
        # 原存档分毫未动
        self.assertEqual(mood.emotions.joy, 80)
        self.assertEqual(mood.emotions.trust, 40)
        self.assertAlmostEqual(preview.intensity, 0.05)

    def test_preview_no_decay_returns_same_object(self):
        """刚更新过 / 从未更新 → 无衰减，直接返回原对象（调用方只读）"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=50))
        mood._recompute()
        mood.updated_at = time.time()
        self.assertIs(preview_time_decayed(mood), mood)
        mood.updated_at = 0
        self.assertIs(preview_time_decayed(mood), mood)

    def test_mood_trigger_excerpt(self):
        """消息摘录：去换行压缩空白，超长截断加省略号"""
        self.assertEqual(mood_trigger_excerpt("哈哈笑死"), "哈哈笑死")
        self.assertEqual(mood_trigger_excerpt("  哈哈\n  笑死  "), "哈哈 笑死")
        self.assertEqual(mood_trigger_excerpt("", ), "")
        long_text = "这是一条特别特别特别特别特别长的消息"
        excerpt = mood_trigger_excerpt(long_text)
        self.assertTrue(excerpt.endswith("…"))
        self.assertLessEqual(len(excerpt), 13)  # 12 字 + 省略号


if __name__ == "__main__":
    unittest.main()
