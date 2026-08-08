# tests/test_global_mood.py
"""全局心情（共享字段）离线单测"""
import asyncio
import tempfile
import unittest
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.bootstrap  # noqa: F401  (注册 emotionai_pro 包别名)

from emotionai_pro.global_mood import (
    GlobalMood, GlobalMoodStore, apply_mood_update, compute_mood_signal,
    MOOD_SIGNAL_CAP,
)
from emotionai_pro.models import EmotionalMetrics


class TestGlobalMoodUpdate(unittest.TestCase):
    def test_apply_update_positive(self):
        """正向更新：joy 上升，负面被对冲压制"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=20, trust=10))
        mood._recompute()

        apply_mood_update(mood, {"joy": 3, "trust": 2})
        # 信号强度 5 ≥ 3 → 冲高：正面赢，joy 大幅上升，负面被压制
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
        # 衰减后 joy 47, trust 38；负面赢：最强维度(disgust)冲高到 40，joy ×0.2 → 9
        self.assertGreaterEqual(mood.emotions.disgust, 40)
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
        """弱信号（<3）：温和叠加，不触发冲高"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=20, anger=5))
        mood._recompute()
        apply_mood_update(mood, {"joy": 1})
        # 先衰减 joy 20→19，弱信号 +1 → 20（不冲高）
        self.assertEqual(mood.emotions.joy, 20)
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


if __name__ == "__main__":
    unittest.main()
