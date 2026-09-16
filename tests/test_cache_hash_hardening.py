# tests/test_cache_hash_hardening.py
"""分片缓存哈希的「异常兜底」回归测试（v4.0.15）

背景：缓存的分片哈希只是「选哪个分片」的内部细节，却曾经把整条 pipeline 打挂——
xxhash>=4.0 不再接受 str，`_get_shard` 直接抛
`TypeError: Strings must be encoded before hashing`，异常沿着
`cache.get()` → `managers.get_user_state()` → `main.process_smart_update()`
一路冒到 AstrBot 的 event hook（= GitHub Issue #1，v4.0.10 修）。

v4.0.10 只修了「未 encode」这一个具体原因。本文件锁定的是**更根本的性质**：
**无论哈希实现出什么问题，缓存都不允许把异常抛给调用方。**

因此每个用例都用 `patch.object(cache_mod.xxhash, "xxh64", side_effect=...)`
强制让哈希炸掉 —— 如果哪天有人把 try/except 去掉，这些用例会立刻失败。
"""
import asyncio
import hashlib
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

import emotionai_pro.cache as cache_mod  # noqa: E402
from emotionai_pro.cache import ShardedTTLCache  # noqa: E402

WARN_MARK = "已降级到备用哈希实现"


def _run(coro):
    return asyncio.run(coro)


class _CacheCase(unittest.TestCase):
    """统一的缓存构造 / 关闭样板（必须在事件循环内创建与关闭）"""

    shard_count = 8

    def _with_cache(self, body):
        async def go():
            c = ShardedTTLCache(
                max_size=64, shard_count=self.shard_count, default_ttl=60
            )
            try:
                return await body(c)
            finally:
                await c.close()

        return _run(go())

    @staticmethod
    def _broken_xxhash():
        """模拟「哈希实现抛异常」——这正是 Issue #1 的形态"""
        return patch.object(
            cache_mod.xxhash,
            "xxh64",
            side_effect=TypeError("Strings must be encoded before hashing"),
        )


class TestHashFailureDoesNotEscape(_CacheCase):
    """核心：哈希炸了，缓存也不能炸"""

    def test_get_shard_does_not_raise(self):
        async def body(c):
            with self._broken_xxhash():
                shard = c._get_shard("state_3418451176")
            self.assertIn(shard, c.shards)

        self._with_cache(body)

    def test_falls_back_to_md5_shard(self):
        """降级后应落到 md5 算出来的那个分片，而不是随便一个"""
        async def body(c):
            key = "state_3418451176"
            with self._broken_xxhash():
                shard = c._get_shard(key)
            expected = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % (
                self.shard_count
            )
            self.assertIs(shard, c.shards[expected])

        self._with_cache(body)

    def test_set_get_delete_roundtrip_survives_broken_hash(self):
        """完整读写链路：正是 Issue #1 里炸掉的那条路径"""
        async def body(c):
            with self._broken_xxhash():
                await c.set("state_u1", {"favor": 50})
                self.assertEqual(await c.get("state_u1"), {"favor": 50})
                self.assertTrue(await c.delete("state_u1"))
                self.assertIsNone(await c.get("state_u1"))

        self._with_cache(body)

    def test_shard_is_stable_across_calls_after_fallback(self):
        """set 与 get 必须落到同一分片，否则写入等于丢失"""
        async def body(c):
            with self._broken_xxhash():
                first = c._get_shard("同一个键")
                for _ in range(5):
                    self.assertIs(first, c._get_shard("同一个键"))

        self._with_cache(body)

    def test_md5_failure_falls_back_to_builtin_hash(self):
        """极端情况（如 FIPS 禁用 md5）也不能抛"""
        async def body(c):
            with self._broken_xxhash():
                with patch.object(
                    cache_mod.hashlib, "md5", side_effect=ValueError("md5 disabled")
                ):
                    shard = c._get_shard("key")
                    self.assertIn(shard, c.shards)
                    self.assertIs(shard, c._get_shard("key"))

        self._with_cache(body)

    def test_hash_returning_garbage_does_not_raise(self):
        """哈希返回了没有 intdigest 的对象（将来改 API 的形态）"""

        class NoDigest:
            pass

        async def body(c):
            with patch.object(cache_mod.xxhash, "xxh64", return_value=NoDigest()):
                shard = c._get_shard("key")
            self.assertIn(shard, c.shards)

        self._with_cache(body)


class TestWarningIsNotSpammy(_CacheCase):
    """降级提示只打一次，避免每条缓存访问刷屏

    ⚠️ v4.0.17 起断言对象从 stdout 换成了 logger。
    原因：上架规则要求插件不得用 `print()` / 内置 `logging` 打日志，
    必须走 `from astrbot.api import logger`。原先这里用
    `contextlib.redirect_stdout` 抓 `print` 的输出，改完就抓不到任何东西了
    —— 那种情况下 `assertEqual(count, 1)` 会退化成 `0 == 1` 恒失败，
    或者反过来如果有人把断言放宽，就会变成**永远通过的假测试**。
    所以这里直接 patch 模块级 logger，断言"warning 恰好被调用一次"。
    """

    def test_warning_logged_only_once(self):
        async def body(c):
            with patch.object(cache_mod, "logger") as fake_logger:
                with self._broken_xxhash():
                    for i in range(10):
                        c._get_shard(f"k{i}")
            self.assertEqual(fake_logger.warning.call_count, 1)
            args, _kwargs = fake_logger.warning.call_args
            self.assertIn(WARN_MARK, str(args[0]))

        self._with_cache(body)

    def test_no_warning_on_healthy_path(self):
        if not cache_mod.XXHASH_AVAILABLE:
            self.skipTest("本环境无 xxhash")

        async def body(c):
            with patch.object(cache_mod, "logger") as fake_logger:
                for i in range(10):
                    c._get_shard(f"k{i}")
            self.assertEqual(fake_logger.warning.call_count, 0)

        self._with_cache(body)


class TestNormalPathUnchanged(_CacheCase):
    """回归守卫：哈希正常时，分片结果必须与改动前逐位一致"""

    def test_uses_xxhash_when_available(self):
        if not cache_mod.XXHASH_AVAILABLE:
            self.skipTest("本环境无 xxhash")

        async def body(c):
            key = "state_3418451176"
            expected = cache_mod.xxhash.xxh64(key.encode("utf-8")).intdigest() % (
                self.shard_count
            )
            self.assertIs(c._get_shard(key), c.shards[expected])

        self._with_cache(body)

    def test_distribution_spreads_over_all_shards(self):
        """分片哈希的意义就是把键摊开；降级后也要能摊开"""
        async def body(c):
            keys = [f"state_{i}" for i in range(400)]
            with self._broken_xxhash():
                used = {c._get_shard(k).shard_id for k in keys}
            self.assertEqual(len(used), self.shard_count)

        self._with_cache(body)


class TestKeyCoercion(_CacheCase):
    """键统一成 bytes 的过程本身也不能抛"""

    def test_to_bytes_is_deterministic(self):
        for key in ["a", b"a", 1, 4.5, None, ("x", 1), {"a": 1}]:
            self.assertEqual(
                ShardedTTLCache._to_bytes(key), ShardedTTLCache._to_bytes(key)
            )

    def test_to_bytes_passes_bytes_through(self):
        self.assertEqual(ShardedTTLCache._to_bytes(b"\x00\xff"), b"\x00\xff")

    def test_to_bytes_encodes_str_as_utf8(self):
        self.assertEqual(ShardedTTLCache._to_bytes("键"), "键".encode("utf-8"))

    def test_various_key_types_do_not_raise(self):
        async def body(c):
            for key in ["str", b"bytes", 123, 4.5, None, ("t", 1), {"a": 1}]:
                self.assertIn(c._get_shard(key), c.shards)

        self._with_cache(body)

    def test_unencodable_key_falls_back_to_placeholder(self):
        class Nasty:
            def __str__(self):
                raise RuntimeError("拒绝转字符串")

        self.assertEqual(
            ShardedTTLCache._to_bytes(Nasty()), b"<unencodable-key>"
        )


if __name__ == "__main__":
    unittest.main()
