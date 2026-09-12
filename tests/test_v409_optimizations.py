# tests/test_v409_optimizations.py
"""v4.0.9 性能/代码优化回归测试

锁定以下优化的**行为等价性**，防止后续改动把语义改回去：
- cache: _estimate_size 复用、cleanup_expired 单遍历、get_stats 用 nlargest
- models: get_dominant 单遍历、预编译正则、apply_update、get_summary
- global_mood: 关键词「担心」不再被覆盖、_recompute 与 8 维口径一致
- storage: 保存路径 bytes 写入 + 校验和一致性
"""
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.bootstrap  # noqa: F401

from emotionai_pro.models import EmotionalMetrics, TextDescriptions
from emotionai_pro.global_mood import (
    GlobalMood, apply_mood_update, compute_mood_signal, _MOOD_KEYWORD_MAP,
)
from emotionai_pro.cache import LRUCacheShard, ShardedTTLCache
from emotionai_pro.storage import AtomicJSONStorage


class TestBedEmotionMetricsOptimized(unittest.TestCase):
    """models.EmotionalMetrics 优化后的行为等价性"""

    def test_get_dominant_single(self):
        self.assertEqual(EmotionalMetrics(joy=50).get_dominant(), "喜悦")
        self.assertEqual(EmotionalMetrics(anger=80).get_dominant(), "愤怒")
        self.assertEqual(EmotionalMetrics().get_dominant(), "中立")

    def test_get_dominant_tie_is_composite(self):
        """并列最高值返回复合描述（与旧实现一致）"""
        m = EmotionalMetrics(joy=30, trust=30)
        self.assertEqual(m.get_dominant(), "复合(喜悦+信任)")

    def test_get_dominant_category_not_key(self):
        """回归：max 必须取 values 而非 dict 键（曾误用 max(dict) 返回键名）"""
        m = EmotionalMetrics(trust=5, joy=90)
        self.assertEqual(m.get_dominant(), "喜悦")

    def test_emotion_values_mapping(self):
        m = EmotionalMetrics(joy=1, trust=2, fear=3, surprise=4,
                             sadness=5, disgust=6, anger=7, anticipation=8)
        self.assertEqual(
            m.emotion_values(),
            {"joy": 1, "trust": 2, "fear": 3, "surprise": 4,
             "sadness": 5, "disgust": 6, "anger": 7, "anticipation": 8},
        )

    def test_emotion_fields_match_names(self):
        """EMOTION_FIELDS 必须与 EMOTION_NAMES 键集合一致"""
        self.assertEqual(set(EmotionalMetrics.EMOTION_FIELDS),
                         set(EmotionalMetrics.EMOTION_NAMES.keys()))

    def test_get_summary_consistent(self):
        m = EmotionalMetrics(joy=10, trust=20, anger=5)
        s = m.get_summary()
        self.assertEqual(s['total_intensity'], 35)
        self.assertEqual(s['positive_balance'], (10 + 20) - 5)
        self.assertEqual(s['dominant'], "信任")

    def test_apply_update_valid(self):
        m = EmotionalMetrics(joy=10)
        m.apply_update({"joy": 20, "anger": 30})
        self.assertEqual(m.joy, 30)
        self.assertEqual(m.anger, 30)

    def test_apply_update_clamps(self):
        m = EmotionalMetrics(joy=95)
        m.apply_update({"joy": 50})
        self.assertEqual(m.joy, 100)
        m.apply_update({"anger": -20})
        self.assertEqual(m.anger, 0)

    def test_apply_update_unknown_dimension_ignored(self):
        m = EmotionalMetrics(joy=10)
        m.apply_update({"not_a_dim": 5, "joy": 5})
        self.assertEqual(m.joy, 15)
        self.assertFalse(hasattr(m, "not_a_dim"))


class TestPrecompiledTextPatterns(unittest.TestCase):
    """预编译正则后校验行为不变"""

    def test_valid_chinese_punctuation_accepted(self):
        self.assertTrue(TextDescriptions.is_valid_attitude("俏皮调侃带点宠溺"))
        self.assertTrue(TextDescriptions.is_valid_attitude("温柔，体贴。"))
        self.assertTrue(TextDescriptions.is_valid_attitude("亲昵！"))

    def test_over_length_rejected(self):
        self.assertFalse(TextDescriptions.is_valid_attitude("好" * 51))
        self.assertTrue(TextDescriptions.is_valid_attitude("好" * 50))

    def test_relationship_limit_is_80(self):
        self.assertTrue(TextDescriptions.is_valid_relationship("好" * 80))
        self.assertFalse(TextDescriptions.is_valid_relationship("好" * 81))

    def test_update_attitude_raises_on_invalid(self):
        td = TextDescriptions()
        with self.assertRaises(ValueError):
            td.update_attitude("a" * 100)
        td.update_attitude("温和")
        self.assertEqual(td.attitude, "温和")

    def test_invalid_restored_to_default_on_init(self):
        td = TextDescriptions(attitude="x" * 100, relationship="y" * 200)
        self.assertEqual(td.attitude, "中立")
        self.assertEqual(td.relationship, "陌生人")


class TestGlobalMoodOptimized(unittest.TestCase):

    def test_recompute_matches_manual(self):
        m = GlobalMood(emotions=EmotionalMetrics(anger=80))
        m._recompute()
        self.assertEqual(m.intensity, 0.8)
        self.assertEqual(m.dominant_emotion, "愤怒")
        e = m.emotions
        expected = round(max(e.joy, e.trust, e.fear, e.surprise,
                             e.sadness, e.disgust, e.anger, e.anticipation) / 100.0, 2)
        self.assertEqual(m.intensity, expected)

    def test_dont_worry_keyword_not_overwritten(self):
        """回归：'担心' 曾被重复定义导致 trust 信号丢失"""
        delta = _MOOD_KEYWORD_MAP["担心"]
        self.assertIn("trust", delta, "'担心' 应保留 trust 信号")
        self.assertIn("fear", delta)

    def test_compute_signal_dont_worry(self):
        sig = compute_mood_signal("我很担心你")
        self.assertGreater(sig.get("trust", 0), 0)
        self.assertGreater(sig.get("fear", 0), 0)

    def test_compute_signal_empty(self):
        self.assertEqual(compute_mood_signal(""), {})
        self.assertEqual(compute_mood_signal("   "), {})

    def test_compute_signal_clamped(self):
        sig = compute_mood_signal("讨厌讨厌讨厌讨厌讨厌讨厌")
        self.assertLessEqual(abs(sig.get("disgust", 0)), 3)

    def test_apply_update_shifts_mood(self):
        m = GlobalMood(emotions=EmotionalMetrics(joy=20))
        apply_mood_update(m, {"anger": 3})
        self.assertGreater(m.emotions.anger, 0)
        self.assertLess(m.emotions.joy, 20)


class TestCacheOptimized(unittest.TestCase):

    def test_estimate_size_reuse_equivalent(self):
        """set 复用一个估算值，结果应与逐个估算一致"""
        async def run():
            shard = LRUCacheShard(max_size=10)
            await shard.set("k1", "v" * 100)
            await shard.set("k2", {"a": "b"})
            expected = shard._estimate_size("k1", "v" * 100) + shard._estimate_size("k2", {"a": "b"})
            self.assertEqual(shard.total_size, expected)
        asyncio.run(run())

    def test_overwrite_keeps_size_consistent(self):
        async def run():
            shard = LRUCacheShard(max_size=10)
            await shard.set("k", "x" * 100)
            after_first = shard.total_size
            await shard.set("k", "y" * 10)  # 覆盖为更小值
            self.assertLess(shard.total_size, after_first)
            self.assertEqual(shard.total_size, shard._estimate_size("k", "y" * 10))
        asyncio.run(run())

    def test_cleanup_expired_frees_size(self):
        async def run():
            shard = LRUCacheShard(max_size=10)
            await shard.set("alive", "a" * 50, ttl=1000)
            await shard.set("dead", "d" * 50, ttl=-1)  # 立即过期
            cleaned, freed = await shard.cleanup_expired()
            self.assertEqual(cleaned, 1)
            self.assertGreater(freed, 0)
            self.assertEqual(shard.total_size, shard._estimate_size("alive", "a" * 50))
        asyncio.run(run())

    def test_total_size_never_negative(self):
        async def run():
            shard = LRUCacheShard(max_size=10)
            await shard.set("k", "v", ttl=-1)
            await shard.cleanup_expired()
            self.assertGreaterEqual(shard.total_size, 0)
        asyncio.run(run())

    def test_get_stats_hot_keys_sorted_desc(self):
        async def run():
            cache = ShardedTTLCache(max_size=64, shard_count=4)
            try:
                for _ in range(5):
                    await cache.get("hot")
                await cache.get("cold")
                stats = await cache.get_stats()
                counts = [h["access_count"] for h in stats["hot_keys"]]
                self.assertEqual(counts, sorted(counts, reverse=True))
                self.assertEqual(stats["hot_keys"][0]["key"], "hot")
            finally:
                await cache.close()
        asyncio.run(run())


class TestStorageOptimized(unittest.TestCase):

    def test_save_writes_valid_utf8_and_checksum(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "data.json"
                st = AtomicJSONStorage(path)
                data = {"名字": "塔菲", "值": 42}
                await st.save(data)

                raw = path.read_bytes()
                # 校验和文件应与落盘字节的 md5 一致
                checksum_file = path.with_suffix('.checksum')
                if checksum_file.exists():
                    stored = checksum_file.read_text(encoding='utf-8').strip()
                    self.assertIn(hashlib.md5(raw).hexdigest(), stored)

                # 内容是合法 JSON 且能往返
                loaded = json.loads(raw.decode('utf-8'))
                self.assertEqual(loaded, data)
        asyncio.run(run())

    def test_backup_created_on_second_save(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "data.json"
                st = AtomicJSONStorage(path)
                await st.save({"v": 1})
                await st.save({"v": 2})
                self.assertTrue(path.with_suffix('.bak').exists())
                self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {"v": 2})
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
