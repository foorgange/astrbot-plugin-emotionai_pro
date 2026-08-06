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

from emotionai_pro.global_mood import GlobalMood, GlobalMoodStore, apply_mood_update
from emotionai_pro.models import EmotionalMetrics


class TestGlobalMoodUpdate(unittest.TestCase):
    def test_apply_update_decay_and_clamp(self):
        """衰减 + 温和叠加 + clamp 到 [0,100]"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=50, trust=40))
        mood._recompute()

        # 应用正向更新（应被温和化）
        apply_mood_update(mood, {"joy": 10, "trust": 8})

        # 衰减: joy 50→47, trust 40→38；叠加 min(10,2)=+2 → 49 / 40
        self.assertEqual(mood.emotions.joy, 49)
        self.assertEqual(mood.emotions.trust, 40)

        # 再应用大幅正向更新，验证 clamp
        apply_mood_update(mood, {"joy": 100, "trust": 100})
        self.assertLessEqual(mood.emotions.joy, 100)
        self.assertLessEqual(mood.emotions.trust, 100)

        # 验证 updated_at 被更新
        self.assertGreater(mood.updated_at, 0)

    def test_apply_update_negative(self):
        """负向更新降低心情"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=50, trust=50))
        apply_mood_update(mood, {"joy": -20})
        # 衰减: 50→47; -min(20,2)=-2 → 45
        self.assertEqual(mood.emotions.joy, 45)

    def test_dominant_and_intensity(self):
        """强度与主导情感正确重算"""
        mood = GlobalMood(emotions=EmotionalMetrics(joy=100, trust=50))
        apply_mood_update(mood, {})
        # 衰减后: joy 95, trust 47; intensity = min(100, (95+47)//2) = 71
        self.assertEqual(mood.intensity, min(100, (95 + 47) // 2))
        self.assertEqual(mood.dominant_emotion, "喜悦")

    def test_zero_emotions_gives_default_dominant(self):
        """全零心情 → 中立"""
        mood = GlobalMood()
        mood._recompute()
        self.assertEqual(mood.dominant_emotion, "中立")
        self.assertEqual(mood.intensity, 0)


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


if __name__ == "__main__":
    unittest.main()
