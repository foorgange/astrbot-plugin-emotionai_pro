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


if __name__ == "__main__":
    unittest.main()
