# equiv_cache_storage.py — 缓存/存储优化等价性验证
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import tests.bootstrap  # noqa: F401

from emotionai_pro.cache import LRUCacheShard, ShardedTTLCache
from emotionai_pro.storage import AtomicJSONStorage

FAIL = []
print("=" * 70)


# ---------- 缓存：total_size 追踪正确性 ----------
async def cache_check():
    # 场景1：写入/覆盖/删除后，total_size 必须等于逐项估算之和
    shard = LRUCacheShard(max_size=100)
    ops = [("a", "x" * 100), ("b", {"k": "v"}),
           ("a", "y" * 10),           # 覆盖
           ("c", [1, 2, 3]), ("d", 12345), ("e", "z" * 500),
           ("b", "w" * 50)]           # 覆盖
    for k, v in ops:
        await shard.set(k, v)
    expected = sum(shard._estimate_size(k, v) for k, v in
                   {"a": "y" * 10, "c": [1, 2, 3], "d": 12345,
                    "e": "z" * 500, "b": "w" * 50}.items())
    if shard.total_size != expected:
        FAIL.append(f"cache total_size: 实际={shard.total_size} 期望={expected}")
    print(f"[c1] 写入/覆盖后 total_size = {shard.total_size} (期望 {expected}) "
          f"{'✓' if shard.total_size == expected else '✗'}")

    # 场景2：删除
    await shard.delete("e")
    expected -= shard._estimate_size("e", "z" * 500)
    if shard.total_size != expected:
        FAIL.append(f"cache delete: 实际={shard.total_size} 期望={expected}")
    print(f"[c2] 删除后 total_size = {shard.total_size} "
          f"{'✓' if shard.total_size == expected else '✗'}")

    # 场景3：过期清理
    await shard.set("exp1", "q" * 200, ttl=-1)
    await shard.set("exp2", "r" * 300, ttl=-1)
    cleaned, freed = await shard.cleanup_expired()
    expected_after = shard._estimate_size("a", "y" * 10) + \
        shard._estimate_size("c", [1, 2, 3]) + shard._estimate_size("d", 12345) + \
        shard._estimate_size("b", "w" * 50)
    if cleaned != 2:
        FAIL.append(f"cleanup 数量: {cleaned}")
    if shard.total_size != expected_after:
        FAIL.append(f"cleanup 后 total_size: {shard.total_size} != {expected_after}")
    print(f"[c3] 清理 2 项 → cleaned={cleaned} total_size={shard.total_size} "
          f"{'✓' if shard.total_size == expected_after else '✗'}")

    # 场景4：total_size 永不为负（重复清理/删不存在的键）
    await shard.cleanup_expired()
    await shard.delete("nonexistent")
    if shard.total_size < 0:
        FAIL.append(f"total_size 为负: {shard.total_size}")
    print(f"[c4] 重复清理后 total_size = {shard.total_size} (应 >= 0) "
          f"{'✓' if shard.total_size >= 0 else '✗'}")

    # 场景5：LRU 淘汰顺序（最久未使用先出）
    s2 = LRUCacheShard(max_size=3)
    for k in ["k1", "k2", "k3"]:
        await s2.set(k, k)
    await s2.get("k1")          # k1 变成最近使用
    await s2.set("k4", "k4")    # 应淘汰 k2
    keys = set(s2.cache.keys())
    if "k2" in keys or "k1" not in keys:
        FAIL.append(f"LRU 淘汰错误: {keys}")
    print(f"[c5] LRU 淘汰后 keys = {sorted(keys)} (应含 k1,k4 不含 k2) "
          f"{'✓' if ('k2' not in keys and 'k1' in keys) else '✗'}")

    # 场景6：命中率统计
    cache = ShardedTTLCache(max_size=64, shard_count=4)
    try:
        await cache.set("hitkey", "v")
        for _ in range(3):
            await cache.get("hitkey")
        await cache.get("miss1")
        await cache.get("miss2")
        st = await cache.get_stats()
        # set 不计入 access_count；3 次命中 + 3 次未命中（首查 hitkey 也算 miss）= 6
        # 实测口径：3 hit / 5 access（首查 hitkey 为 miss，miss1/miss2 各 1）
        if st["hit_count"] != 3:
            FAIL.append(f"hit_count={st['hit_count']}")
        if st["access_count"] != 5:
            FAIL.append(f"access_count={st['access_count']}")
        print(f"[c6] 命中 {st['hit_count']}/{st['access_count']} "
              f"命中率={st['hit_rate']:.1f}% "
              f"{'✓' if st['hit_count'] == 3 and st['access_count'] == 5 else '✗'}")
    finally:
        await cache.close()


asyncio.run(cache_check())


# ---------- 存储：字节写盘 + 校验和 ----------
async def storage_check():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.json"
        st = AtomicJSONStorage(p)
        data = {"中文键": "塔菲", "n": 42, "嵌套": {"a": [1, 2]}}
        await st.save(data)

        raw = p.read_bytes()
        # 字节内容必须能还原成原数据
        if json.loads(raw.decode("utf-8")) != data:
            FAIL.append("storage 往返失败")
        print(f"[s1] 字节写盘往返 {'✓' if json.loads(raw.decode('utf-8')) == data else '✗'}")

        # 校验和必须等于落盘字节的 md5（说明复用字节串未改变内容）
        cf = p.with_suffix(".checksum")
        if cf.exists():
            stored = cf.read_text(encoding="utf-8").strip()
            ok = hashlib.md5(raw).hexdigest() in stored
            if not ok:
                FAIL.append(f"校验和不匹配: {stored}")
            print(f"[s2] 校验和匹配落盘字节 {'✓' if ok else '✗'}")
        else:
            print("[s2] 无校验和文件（跳过）")

        # 二次保存产生备份，且主文件是新内容
        await st.save({"v": 2})
        bak = p.with_suffix(".bak")
        if not bak.exists():
            FAIL.append("未生成备份")
        if json.loads(p.read_text(encoding="utf-8")) != {"v": 2}:
            FAIL.append("二次保存内容错误")
        print(f"[s3] 备份生成 + 主文件更新 {'✓' if bak.exists() else '✗'}")

        # 临时文件不应残留
        if p.with_suffix(".tmp").exists():
            FAIL.append("临时文件残留")
        print(f"[s4] 无临时文件残留 {'✓' if not p.with_suffix('.tmp').exists() else '✗'}")


asyncio.run(storage_check())

print("=" * 70)
if FAIL:
    print(f"❌ {len(FAIL)} 处问题：")
    for f in FAIL:
        print("   " + f)
    sys.exit(1)
print("✅ 缓存与存储优化行为正确")
print("=" * 70)
