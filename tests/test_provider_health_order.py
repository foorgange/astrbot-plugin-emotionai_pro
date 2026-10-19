# tests/test_provider_health_order.py
"""v4.1.14 回归：备选链按 provider 近期健康分动态排序

背景：备选链是静态顺序（辅助 LLM → 主 LLM → 档案退避链 → 其余），
而免费模型可用性随时间波动（429 限流 / 额度耗尽 / 高负载慢响应）。
静态顺序的后果：「配置靠前但当前不可用」的 provider 先烧掉时间片，
链尾存活的候选可能永远轮不到（max_providers 只取前 N 个）。
线上实锤：step 系 + 商汤系同时不可用时，链尾存活的 agnes 系排第 7+，
从不被尝试，整轮分析直接降级本地兜底。

修复：进程内记录每个 provider 近期尝试结果，备选链按健康分排序后再
取前 N 个——稳定的往前放，不稳定的沉底；无历史记录取中性分。

反向验证要求：把排序逻辑换回「静态顺序」时，本文件排序相关测试必须失败。
"""
import asyncio
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.emotion_expert import (  # noqa: E402
    EmotionAnalysisExpert,
    ProviderHealthTracker,
)
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402

_OK_TEXT = "情感分析结果-" + "x" * 40


class _Meta:
    def __init__(self, pid, model=None):
        self.id = pid
        self.model = model or pid
        self.type = "openai_chat_completion"


class FakeProvider:
    """可控行为的 provider 桩。behavior: ok / empty / raise / hang"""

    def __init__(self, pid, behavior="ok", delay=0.0):
        self._meta = _Meta(pid)
        self.behavior = behavior
        self.delay = delay
        self.calls = 0

    def meta(self):
        return self._meta

    async def text_chat(self, prompt=None, model=None, **kwargs):
        self.calls += 1
        if self.behavior == "raise":
            raise RuntimeError(f"{self._meta.id} 故意失败")
        if self.behavior == "hang":
            await asyncio.sleep(self.delay or 60)
            return _OK_TEXT
        if self.behavior == "empty":
            return ""
        if self.delay:
            await asyncio.sleep(self.delay)
        return _OK_TEXT


class FakeContext:
    def __init__(self, providers=(), main=None, fallback_ids=(), by_id=None):
        self._providers = list(providers)
        self._main = main
        self._by_id = dict(by_id or {})
        self.provider_settings = {"fallback_chat_models": list(fallback_ids)}

    def get_all_providers(self):
        return list(self._providers)

    def get_using_provider(self, umo=None):
        return self._main

    def get_provider_by_id(self, pid):
        return self._by_id.get(pid)

    def get_config(self, umo=None):
        return {"provider_settings": self.provider_settings}


class FakeCache:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value


def _expert(context=None, provider=None, model=None, budget=45.0,
            max_providers=3, cache=None):
    return EmotionAnalysisExpert(
        cache=cache,
        context=context,
        secondary_llm_provider=provider,
        secondary_llm_model=model,
        time_budget=budget,
        max_providers=max_providers,
    )


def _state():
    return EnhancedEmotionalState(user_key="u1", favor=50, intimacy=50)


def _call(e, umo=None):
    return asyncio.run(
        e._call_real_llm_with_retry("你好呀", "你好", _state(), umo)
    )


# ---------------------------------------------------------------------------
# 单元：健康分与排序
# ---------------------------------------------------------------------------
class TestHealthScore(unittest.TestCase):
    def setUp(self):
        self.t = ProviderHealthTracker()

    def test_no_history_is_neutral(self):
        self.assertEqual(self.t.score("vendor/x"), 0.5)

    def test_single_success_above_neutral(self):
        self.t.record("vendor/x", True)
        score = self.t.score("vendor/x")
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_single_failure_below_neutral(self):
        self.t.record("vendor/x", False)
        score = self.t.score("vendor/x")
        self.assertLess(score, 0.5)
        self.assertGreaterEqual(score, 0.0)

    def test_repeated_failures_sink_further(self):
        t = self.t
        t.record("vendor/x", False)
        first = t.score("vendor/x")
        for _ in range(3):
            t.record("vendor/x", False)
        self.assertLess(t.score("vendor/x"), first)

    def test_recovery_raises_score(self):
        t = self.t
        for _ in range(3):
            t.record("vendor/x", False)
        low = t.score("vendor/x")
        for _ in range(3):
            t.record("vendor/x", True)
        self.assertGreater(t.score("vendor/x"), low)

    def test_old_samples_pruned_by_window(self):
        now = time.time()
        # 窗口外的一次成功 + 窗口内的一次失败 → 只算失败
        self.t.record("vendor/x", True, now=now - ProviderHealthTracker.WINDOW_SECONDS - 10)
        self.t.record("vendor/x", False, now=now)
        self.assertEqual(self.t.score("vendor/x", now=now), self.t.score("vendor/x", now=now))
        fresh_only = (0 + ProviderHealthTracker.PRIOR_SUCCESS) / (1 + ProviderHealthTracker.PRIOR_TOTAL)
        self.assertAlmostEqual(self.t.score("vendor/x", now=now), fresh_only)

    def test_all_samples_expired_returns_neutral(self):
        old = time.time() - ProviderHealthTracker.WINDOW_SECONDS - 10
        self.t.record("vendor/x", False, now=old)
        self.assertEqual(self.t.score("vendor/x"), 0.5)

    def test_max_samples_capped(self):
        for i in range(ProviderHealthTracker.MAX_SAMPLES + 10):
            self.t.record("vendor/x", i % 2 == 0)
        samples = self.t._samples.get("vendor/x", [])
        self.assertLessEqual(len(samples), ProviderHealthTracker.MAX_SAMPLES)

    def test_reset_clears_everything(self):
        self.t.record("vendor/x", True)
        self.t.reset()
        self.assertEqual(self.t.score("vendor/x"), 0.5)


class TestSortOrder(unittest.TestCase):
    def setUp(self):
        self.t = ProviderHealthTracker()

    def test_no_history_keeps_original_order(self):
        """无历史记录时稳定排序保持构造顺序（配置优先级仍是次级依据）"""
        ps = [FakeProvider(f"vendor/{c}") for c in "abc"]
        self.assertEqual(self.t.sort_providers(ps), ps)

    def test_stable_provider_floats_to_front(self):
        ps = [FakeProvider(f"vendor/{c}") for c in "abc"]
        # b 稳定成功，a/c 连续失败
        for _ in range(3):
            self.t.record("vendor/b", True)
            self.t.record("vendor/a", False)
            self.t.record("vendor/c", False)
        ordered = self.t.sort_providers(ps)
        self.assertEqual(ordered[0].meta().id, "vendor/b")

    def test_unstable_sinks_but_stays_in_chain(self):
        """沉底不等于移除：恢复后还能被尝试"""
        ps = [FakeProvider(f"vendor/{c}") for c in "abc"]
        for _ in range(3):
            self.t.record("vendor/a", False)
            self.t.record("vendor/b", True)
            self.t.record("vendor/c", True)
        ordered = self.t.sort_providers(ps)
        self.assertEqual(ordered[-1].meta().id, "vendor/a")
        self.assertEqual(len(ordered), 3)

    def test_equal_scores_keep_relative_order(self):
        ps = [FakeProvider(f"vendor/{c}") for c in "abc"]
        for p in ps:
            self.t.record(p.meta().id, True)
        self.assertEqual(self.t.sort_providers(ps), ps)

    def test_provider_key_fallbacks(self):
        """没有 meta 的对象退化为对象 id，不炸"""
        obj = object()
        key = ProviderHealthTracker.provider_key(obj)
        self.assertTrue(key.startswith("obj:"))

        class NamedNoMeta:
            name = "weird"

        self.assertEqual(ProviderHealthTracker.provider_key(NamedNoMeta()), "weird")


# ---------------------------------------------------------------------------
# 集成：排序接入备选链
# ---------------------------------------------------------------------------
class TestChainOrdering(unittest.TestCase):
    def setUp(self):
        self._orig_slice = EmotionAnalysisExpert.MIN_SLICE
        EmotionAnalysisExpert.MIN_SLICE = 0.3

    def tearDown(self):
        EmotionAnalysisExpert.MIN_SLICE = self._orig_slice

    def _ctx_with(self, *providers):
        return FakeContext(providers=list(providers), main=None)

    def test_healthy_provider_tried_first_next_round(self):
        """核心回归：第一轮 a 挂 b 成功；第二轮必须优先尝试 b"""
        a = FakeProvider("vendor/a", behavior="raise")
        b = FakeProvider("vendor/b")
        e = _expert(context=self._ctx_with(a, b), max_providers=2)

        # 第一轮：静态顺序 a → b
        self.assertEqual(_call(e), _OK_TEXT)
        self.assertEqual((a.calls, b.calls), (1, 1))

        # 第二轮：健康分排序后 b 应当排在 a 前面，a 不再被尝试
        self.assertEqual(_call(e), _OK_TEXT)
        self.assertEqual((a.calls, b.calls), (1, 2))

    def test_all_fail_records_failure_for_each(self):
        ps = [FakeProvider(f"vendor/{c}", behavior="raise") for c in "abc"]
        e = _expert(context=self._ctx_with(*ps), max_providers=3)
        self.assertIsNone(_call(e))
        for p in ps:
            key = e._provider_health.provider_key(p)
            self.assertLess(e._provider_health.score(key), 0.5)

    def test_success_recorded_for_winner(self):
        a = FakeProvider("vendor/a", behavior="raise")
        b = FakeProvider("vendor/b")
        e = _expert(context=self._ctx_with(a, b), max_providers=2)
        self.assertEqual(_call(e), _OK_TEXT)
        self.assertGreater(
            e._provider_health.score(e._provider_health.provider_key(b)), 0.5
        )

    def test_max_providers_caps_after_sorting(self):
        """排序发生在取前 N 个之前：被挤出去的挂掉 provider 不再烧时间片"""
        dead = [FakeProvider(f"vendor/dead{i}", behavior="raise") for i in range(2)]
        alive = [FakeProvider(f"vendor/alive{i}") for i in range(2)]
        e = _expert(context=self._ctx_with(*(dead + alive)), max_providers=3)

        # 第一轮：无历史 → 静态顺序，前 3 个 = dead0, dead1, alive0
        self.assertEqual(_call(e), _OK_TEXT)
        self.assertEqual([p.calls for p in dead], [1, 1])
        self.assertEqual(alive[0].calls, 1)
        self.assertEqual(alive[1].calls, 0)

        # 第二轮：挂掉的沉底，alive0/alive1 排前面，dead 不再被尝试
        self.assertEqual(_call(e), _OK_TEXT)
        self.assertEqual([p.calls for p in dead], [1, 1])
        self.assertEqual(alive[0].calls, 2)

    def test_single_provider_retry_semantics_unchanged(self):
        """只有一个候选时保留同一 provider 重试的旧语义"""
        p = FakeProvider("vendor/a", behavior="raise")
        e = _expert(context=self._ctx_with(p), max_providers=3)
        e.llm_retry_delay = 0
        self.assertIsNone(_call(e))
        self.assertEqual(p.calls, 3)

    def test_sorting_is_stable_sort_source_constraint(self):
        """源码级约束：必须先排序再切片（防止回归成先切片后排序）"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "emotion_expert.py"), "r", encoding="utf-8") as f:
            src = f.read()
        sort_pos = src.find("self._provider_health.sort_providers(chain)")
        slice_pos = src.find("ordered_chain[: self.max_providers]")
        self.assertGreater(sort_pos, 0, "备选链必须经过健康分排序")
        self.assertGreater(slice_pos, 0, "必须存在取前 N 个的切片")
        self.assertLess(sort_pos, slice_pos, "必须先排序再切片")


class TestExcludedProviders(unittest.TestCase):
    """v4.1.17：不合适的视觉模型不进入情感分析专家链"""

    VL_ID = "siliconflow/Qwen/Qwen3-VL-30B-A3B-Instruct"

    def _ctx(self):
        vl = FakeProvider(self.VL_ID)
        main = FakeProvider("step/step-5-preview")
        agnes = FakeProvider("agnes/agnes-2.5-flash")
        providers = [vl, main, agnes]
        by_id = {p.meta().id: p for p in providers}
        return FakeContext(providers=providers, main=main, by_id=by_id), vl, main, agnes

    def test_vision_provider_excluded_from_chain(self):
        ctx, vl, main, agnes = self._ctx()
        e = _expert(context=ctx, provider="agnes/agnes-2.5-flash")
        chain = e._build_provider_chain()
        ids = [e._get_provider_meta(p)[0] for p in chain]
        self.assertNotIn(self.VL_ID, ids)
        self.assertIn("agnes/agnes-2.5-flash", ids)
        self.assertIn("step/step-5-preview", ids)
        # provider 本身还在 AstrBot 列表里，只是不进专家链
        self.assertIn(vl, ctx.get_all_providers())

    def test_vision_provider_excluded_even_as_explicit_secondary(self):
        """即使有人把辅助 LLM 配成这个视觉模型，也不能进入专家链"""
        ctx, vl, main, agnes = self._ctx()
        e = _expert(context=ctx, provider=self.VL_ID, max_providers=2)
        chain = e._build_provider_chain()
        ids = [e._get_provider_meta(p)[0] for p in chain]
        self.assertNotIn(self.VL_ID, ids)
        self.assertIn("step/step-5-preview", ids)

    def test_vision_model_excluded_by_model_name(self):
        """按 model 名也能排除（防止同一模型换一个 provider id）"""
        class _OtherVlProvider(FakeProvider):
            def __init__(self):
                super().__init__("custom/vision")
                self._meta.model = "Qwen/Qwen3-VL-30B-A3B-Instruct"

        vl = _OtherVlProvider()
        main = FakeProvider("step/step-5-preview")
        ctx = FakeContext(providers=[vl, main], main=main,
                          by_id={p.meta().id: p for p in (vl, main)})
        e = _expert(context=ctx, provider="custom/vision")
        ids = [e._get_provider_meta(p)[0] for p in e._build_provider_chain()]
        self.assertNotIn("custom/vision", ids)
        self.assertIn("step/step-5-preview", ids)


if __name__ == "__main__":
    unittest.main()
