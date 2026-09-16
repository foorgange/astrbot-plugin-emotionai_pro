# tests/test_llm_fallback_budget.py
"""情感分析「带总时间预算的备选链」回归测试（问题 #8）

改动前的行为：
    只选 1 个 provider，然后对**同一个** provider 重试 llm_retry_count(3) 次 ×
    llm_timeout(30s)，最坏约 90s 才降级到 smart_fallback；主 provider 一旦
    不可用就完全没有冗余。

改动后：
    在 time_budget 秒的总预算内，沿备选链
    （辅助LLM → 会话主LLM → 档案里的 fallback_chat_models → 其它 provider）
    最多尝试 max_providers 次，任一成功即返回；预算耗尽即降级。

本文件同时锁定一条硬约束：**不得影响 AstrBot 主对话自身的退避重试机制**。
插件对 provider / provider_settings 全程只读，不会写回 fallback_chat_models。
"""
import asyncio
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.emotion_expert import EmotionAnalysisExpert  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OK_TEXT = "情感分析结果-" + "x" * 40


class _Meta:
    """对应 astrbot.core.provider.entities.ProviderMeta"""

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
    """完整 Context 桩（含 provider 解析与配置读取）"""

    def __init__(self, providers=(), main=None, fallback_ids=(), by_id=None):
        self._providers = list(providers)
        self._main = main
        self._by_id = dict(by_id or {})
        self.provider_settings = {"fallback_chat_models": list(fallback_ids)}
        self.received_umos = []
        self.config_reads = 0

    def get_all_providers(self):
        return list(self._providers)

    def get_using_provider(self, umo=None):
        self.received_umos.append(umo)
        return self._main

    def get_provider_by_id(self, pid):
        return self._by_id.get(pid)

    def get_config(self, umo=None):
        self.config_reads += 1
        return {"provider_settings": self.provider_settings}


class BareContext:
    """最小桩：只有 get_using_provider（模拟旧版 / 非常规 Context）"""

    def __init__(self, main=None):
        self._main = main

    def get_using_provider(self, umo=None):
        return self._main


class RaisingContext:
    """所有读取接口都抛异常，用于验证防御式降级"""

    def get_all_providers(self):
        raise RuntimeError("provider manager down")

    def get_using_provider(self, umo=None):
        raise RuntimeError("provider manager down")

    def get_provider_by_id(self, pid):
        raise RuntimeError("provider manager down")

    def get_config(self, umo=None):
        raise RuntimeError("config manager down")


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


def _read_source(filename):
    with open(os.path.join(_ROOT, filename), "r", encoding="utf-8") as f:
        return f.read()


# ==================== 配置项 ====================


class TestBudgetConfig(unittest.TestCase):
    """新增两个配置项及其默认值 / 边界"""

    def test_plugin_config_defaults(self):
        cfg = PluginConfig()
        self.assertEqual(cfg.emotion_llm_time_budget, 45.0)
        self.assertEqual(cfg.emotion_llm_max_providers, 3)

    def test_budget_range_enforced(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            PluginConfig(emotion_llm_time_budget=5.0)   # < 10
        with self.assertRaises(ValidationError):
            PluginConfig(emotion_llm_time_budget=999.0)  # > 300
        with self.assertRaises(ValidationError):
            PluginConfig(emotion_llm_max_providers=0)   # < 1
        with self.assertRaises(ValidationError):
            PluginConfig(emotion_llm_max_providers=11)  # > 10

    def test_schema_exposes_both_keys(self):
        with open(os.path.join(_ROOT, "_conf_schema.json"), "r",
                  encoding="utf-8") as f:
            schema = json.load(f)
        self.assertIn("emotion_llm_time_budget", schema)
        self.assertIn("emotion_llm_max_providers", schema)
        self.assertEqual(schema["emotion_llm_time_budget"]["default"], 45.0)
        self.assertEqual(schema["emotion_llm_max_providers"]["default"], 3)

    def test_main_maps_both_keys_from_raw_config(self):
        src = _read_source("main.py")
        self.assertIn('"emotion_llm_time_budget": "emotion_llm_time_budget"', src)
        self.assertIn('"emotion_llm_max_providers": "emotion_llm_max_providers"', src)

    def test_main_forwards_budget_to_expert(self):
        src = _read_source("main.py")
        self.assertIn("time_budget=self.config.emotion_llm_time_budget", src)
        self.assertIn("max_providers=self.config.emotion_llm_max_providers", src)

    def test_expert_sanitizes_bad_budget_values(self):
        """非法输入不得让专家构造失败"""
        e = _expert(budget=None, max_providers=None)
        self.assertEqual(e.time_budget, 45.0)
        self.assertEqual(e.max_providers, 3)

        e = _expert(budget="abc", max_providers="abc")
        self.assertEqual(e.time_budget, 45.0)
        self.assertEqual(e.max_providers, 3)

    def test_expert_clamps_budget_to_min_slice(self):
        """预算不得低于 MIN_SLICE，否则一次尝试都发不出去"""
        e = _expert(budget=0.1)
        self.assertEqual(e.time_budget, EmotionAnalysisExpert.MIN_SLICE)
        e = _expert(max_providers=0)
        self.assertEqual(e.max_providers, 1)


# ==================== 备选链构造 ====================


class TestProviderChain(unittest.TestCase):
    """链的顺序、去重与容错"""

    def test_chain_order_is_secondary_main_archive_rest(self):
        secondary = FakeProvider("vendor/secondary")
        main = FakeProvider("商汤/deepseek-v4-flash")
        arch1 = FakeProvider("vendor/arch1")
        arch2 = FakeProvider("vendor/arch2")
        rest = FakeProvider("vendor/rest")

        ctx = FakeContext(
            providers=[rest, secondary, main, arch1, arch2],
            main=main,
            fallback_ids=["vendor/arch1", "vendor/arch2"],
            by_id={"vendor/arch1": arch1, "vendor/arch2": arch2},
        )
        e = _expert(context=ctx, provider="vendor/secondary")
        chain = e._build_provider_chain("umo-x")

        self.assertEqual(
            [e._get_provider_name(p) for p in chain],
            ["vendor/secondary", "商汤/deepseek-v4-flash",
             "vendor/arch1", "vendor/arch2", "vendor/rest"],
        )

    def test_chain_dedups_by_provider_id(self):
        main = FakeProvider("商汤/deepseek-v4-flash")
        ctx = FakeContext(
            providers=[main],
            main=main,
            fallback_ids=["商汤/deepseek-v4-flash"],
            by_id={"商汤/deepseek-v4-flash": main},
        )
        e = _expert(context=ctx)
        chain = e._build_provider_chain("umo-x")
        self.assertEqual(len(chain), 1)

    def test_chain_skips_unresolvable_fallback_ids(self):
        """档案里配了但解析不到的 id 必须被跳过，而不是崩掉"""
        main = FakeProvider("商汤/deepseek-v4-flash")
        ctx = FakeContext(
            providers=[main],
            main=main,
            fallback_ids=["vendor/not-exist", ""],
            by_id={},
        )
        e = _expert(context=ctx)
        chain = e._build_provider_chain("umo-x")
        self.assertEqual([e._get_provider_name(p) for p in chain],
                         ["商汤/deepseek-v4-flash"])

    def test_chain_empty_when_no_providers(self):
        e = _expert(context=FakeContext(providers=[]))
        self.assertEqual(e._build_provider_chain("umo-x"), [])

    def test_chain_survives_bare_context(self):
        """Context 缺 get_all_providers 时返回空链，不抛异常"""
        e = _expert(context=BareContext())
        self.assertEqual(e._build_provider_chain("umo-x"), [])

    def test_chain_survives_raising_context(self):
        e = _expert(context=RaisingContext())
        self.assertEqual(e._build_provider_chain("umo-x"), [])

    def test_chain_handles_non_dict_provider_settings(self):
        ctx = FakeContext(providers=[FakeProvider("vendor/a")], main=None)
        ctx.provider_settings = "not-a-dict"
        e = _expert(context=ctx)
        chain = e._build_provider_chain("umo-x")
        self.assertEqual(len(chain), 1)

    def test_chain_handles_non_list_fallback_models(self):
        ctx = FakeContext(providers=[FakeProvider("vendor/a")], main=None)
        ctx.provider_settings = {"fallback_chat_models": "vendor/x"}
        e = _expert(context=ctx)
        self.assertEqual(len(e._build_provider_chain("umo-x")), 1)

    def test_resolve_fallback_ids_filters_non_strings(self):
        ctx = FakeContext(providers=[FakeProvider("vendor/a")])
        ctx.provider_settings = {
            "fallback_chat_models": ["vendor/ok", 123, None, "  ", "vendor/ok2"]
        }
        e = _expert(context=ctx)
        self.assertEqual(e._resolve_fallback_provider_ids("umo-x"),
                         ["vendor/ok", "vendor/ok2"])


# ==================== 预算与切换行为 ====================


class TestBudgetedFallback(unittest.TestCase):
    """核心：失败切下一个、成功即停、预算封顶"""

    def setUp(self):
        self._orig_slice = EmotionAnalysisExpert.MIN_SLICE
        EmotionAnalysisExpert.MIN_SLICE = 0.3

    def tearDown(self):
        EmotionAnalysisExpert.MIN_SLICE = self._orig_slice

    def _ctx_with(self, *providers):
        return FakeContext(providers=list(providers), main=None)

    def test_success_on_first_provider_does_not_try_second(self):
        p1 = FakeProvider("vendor/a")
        p2 = FakeProvider("vendor/b")
        e = _expert(context=self._ctx_with(p1, p2))
        self.assertEqual(_call(e), _OK_TEXT)
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 0)

    def test_first_fails_then_second_succeeds(self):
        p1 = FakeProvider("vendor/a", behavior="raise")
        p2 = FakeProvider("vendor/b")
        p3 = FakeProvider("vendor/c")
        e = _expert(context=self._ctx_with(p1, p2, p3))
        self.assertEqual(_call(e), _OK_TEXT)
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 1)
        self.assertEqual(p3.calls, 0)

    def test_empty_response_counts_as_failure_and_switches(self):
        p1 = FakeProvider("vendor/a", behavior="empty")
        p2 = FakeProvider("vendor/b")
        e = _expert(context=self._ctx_with(p1, p2))
        self.assertEqual(_call(e), _OK_TEXT)
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 1)

    def test_all_fail_returns_none(self):
        ps = [FakeProvider(f"vendor/{c}", behavior="raise") for c in "abc"]
        e = _expert(context=self._ctx_with(*ps))
        self.assertIsNone(_call(e))
        for p in ps:
            self.assertEqual(p.calls, 1)

    def test_max_providers_caps_attempts(self):
        ps = [FakeProvider(f"vendor/{c}", behavior="raise") for c in "abcde"]
        e = _expert(context=self._ctx_with(*ps), max_providers=2)
        self.assertIsNone(_call(e))
        self.assertEqual([p.calls for p in ps], [1, 1, 0, 0, 0])

    def test_single_provider_keeps_retry_semantics(self):
        """只有一个候选时，保留旧的「同一 provider 重试」行为"""
        p = FakeProvider("vendor/a", behavior="raise")
        e = _expert(context=self._ctx_with(p), max_providers=3)
        e.llm_retry_delay = 0  # 去掉退避，保持测试快速
        self.assertIsNone(_call(e))
        self.assertEqual(p.calls, 3)

    def test_single_provider_retry_succeeds_on_second_try(self):
        p = FakeProvider("vendor/a")

        async def flaky(prompt=None, model=None, **kwargs):
            p.calls += 1
            if p.calls == 1:
                raise RuntimeError("第一次抖动")
            return _OK_TEXT

        p.text_chat = flaky
        e = _expert(context=self._ctx_with(p), max_providers=3)
        e.llm_retry_delay = 0
        self.assertEqual(_call(e), _OK_TEXT)
        self.assertEqual(p.calls, 2)

    def test_budget_cuts_off_hanging_provider(self):
        """一个挂住的 provider 不得吃掉全部预算，更不得让总耗时失控"""
        p1 = FakeProvider("vendor/a", behavior="hang")
        p2 = FakeProvider("vendor/b", behavior="hang")
        p3 = FakeProvider("vendor/c", behavior="hang")
        e = _expert(context=self._ctx_with(p1, p2, p3), budget=0.6,
                    max_providers=3)

        start = time.monotonic()
        self.assertIsNone(_call(e))
        elapsed = time.monotonic() - start

        # 旧行为会是 3 × 30s；现在应被预算截断在 1s 内
        self.assertLess(elapsed, 2.0, f"耗时 {elapsed:.2f}s 超出预算约束")
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 0)
        self.assertEqual(p3.calls, 0)

    def test_budget_allows_second_attempt_when_it_fits(self):
        p1 = FakeProvider("vendor/a", behavior="hang")
        p2 = FakeProvider("vendor/b", behavior="hang")
        p3 = FakeProvider("vendor/c")
        e = _expert(context=self._ctx_with(p1, p2, p3), budget=0.9,
                    max_providers=3)

        start = time.monotonic()
        self.assertIsNone(_call(e))
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 2.0)
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 1)
        self.assertEqual(p3.calls, 0)

    def test_umo_is_passed_to_config_and_provider_lookup(self):
        p1 = FakeProvider("vendor/a", behavior="raise")
        p2 = FakeProvider("vendor/b")
        ctx = FakeContext(providers=[p1, p2], main=None,
                          fallback_ids=["vendor/b"],
                          by_id={"vendor/b": p2})
        e = _expert(context=ctx)
        self.assertEqual(_call(e, "aiocqhttp:GroupMessage:123"), _OK_TEXT)
        self.assertGreaterEqual(ctx.config_reads, 1)
        self.assertIn("aiocqhttp:GroupMessage:123", ctx.received_umos)

    def test_no_providers_returns_none_without_raising(self):
        e = _expert(context=FakeContext(providers=[]))
        self.assertIsNone(_call(e))


# ==================== 端到端：降级到本地兜底 ====================


class TestAnalyzeEndToEnd(unittest.TestCase):
    """整条 analyze_and_update_emotion 链路仍能正常降级 / 升级"""

    def test_all_providers_fail_falls_back_to_smart_fallback(self):
        p = FakeProvider("vendor/a", behavior="raise")
        cache = FakeCache()
        e = _expert(context=FakeContext(providers=[p]), cache=cache)
        e.llm_retry_delay = 0

        updates = asyncio.run(
            e.analyze_and_update_emotion("u1", "你好", "你好呀", _state(), "umo-x")
        )
        self.assertEqual(updates["source"], "smart_fallback")
        self.assertFalse(updates["llm_available"])

    def test_successful_call_reports_llm_analysis(self):
        p = FakeProvider("vendor/a")
        cache = FakeCache()
        e = _expert(context=FakeContext(providers=[p]), cache=cache)

        updates = asyncio.run(
            e.analyze_and_update_emotion("u1", "你好", "你好呀", _state(), "umo-x")
        )
        self.assertEqual(updates["source"], "llm_analysis")
        self.assertTrue(updates["llm_available"])

    def test_second_provider_rescues_after_first_fails(self):
        p1 = FakeProvider("vendor/a", behavior="raise")
        p2 = FakeProvider("vendor/b")
        cache = FakeCache()
        e = _expert(context=FakeContext(providers=[p1, p2]), cache=cache)

        updates = asyncio.run(
            e.analyze_and_update_emotion("u1", "你好", "你好呀", _state(), "umo-x")
        )
        self.assertEqual(updates["source"], "llm_analysis")
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 1)


# ==================== 硬约束：不动 AstrBot 的退避机制 ====================


class TestDoesNotTouchAstrBotFallback(unittest.TestCase):
    """插件对 provider 配置只读，绝不影响主对话自身的退避重试"""

    FORBIDDEN_CALLS = (
        "set_provider", "update_provider", "create_provider",
        "terminate_provider",
    )

    def test_no_provider_write_calls_in_sources(self):
        for filename in ("emotion_expert.py", "main.py", "config_manager.py",
                         "managers.py"):
            src = _read_source(filename)
            for token in self.FORBIDDEN_CALLS:
                self.assertNotIn(
                    token, src,
                    f"{filename} 不应出现 provider 写入调用 {token}()",
                )

    def test_fallback_chat_models_is_only_read(self):
        """fallback_chat_models 只能以 .get() 读取，不得赋值"""
        src = _read_source("emotion_expert.py")
        self.assertIn('provider_settings.get("fallback_chat_models"', src)
        self.assertNotIn('["fallback_chat_models"] =', src)
        self.assertNotIn("['fallback_chat_models'] =", src)
        self.assertNotIn("fallback_chat_models\"].append", src)

    def test_reading_chain_does_not_mutate_provider_settings(self):
        p1 = FakeProvider("vendor/a")
        p2 = FakeProvider("vendor/b")
        ctx = FakeContext(providers=[p1, p2], main=None,
                          fallback_ids=["vendor/b"], by_id={"vendor/b": p2})
        before = json.dumps(ctx.provider_settings, sort_keys=True)

        e = _expert(context=ctx)
        e._build_provider_chain("umo-x")
        e._resolve_fallback_provider_ids("umo-x")
        _call(e, "umo-x")

        after = json.dumps(ctx.provider_settings, sort_keys=True)
        self.assertEqual(before, after, "provider_settings 被意外修改了")

    def test_legacy_method_still_available(self):
        """_execute_llm_call 的旧签名（不传 timeout）必须继续可用"""
        provider = FakeProvider("vendor/a")
        e = _expert()
        result = asyncio.run(e._execute_llm_call(provider, "分析这段对话"))
        self.assertEqual(result, _OK_TEXT)
        self.assertEqual(e.llm_timeout, 30.0)


if __name__ == "__main__":
    unittest.main()
