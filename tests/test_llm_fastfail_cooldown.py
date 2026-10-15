# tests/test_llm_fastfail_cooldown.py
"""情感分析专家链「429 快速失败」与「断路器冷却复位」回归测试

2026-10-13 线上实锤的两个问题：

1. **429 烧满时间片**：备选链给每个 provider 的时间片是 15s，而框架
   `retry_provider_request` 对 429 有 5 次 tenacity 指数退避（累计 ~15s），
   rpm 窗口又不可能在秒级恢复 —— 结果每次尝试都恰好耗满 15s 后失败，
   3 个候选 = 45s 预算全部浪费，最终降级 smart_fallback。
   修复：插件侧调用 text_chat 时传 `request_max_retries=1`，429 时约 3s
   （SDK 自带少量快速重试 + 框架 1s 键轮换等待）内失败并切换下一个候选。
   ⚠️ 主对话链路不传该参数，框架退避重试不受影响。

2. **断路器永久禁用**：`_llm_failures >= 3` 时 `_llm_available = False`，
   而 `reset_llm_availability()` 没有任何调用方 —— 一旦禁用，后续所有
   情感分析直接降级 smart_fallback，即使上游 429 早已消退
   （23:45:00 实锤触发后再无一条 llm_analysis）。
   修复：禁用改为「冷却」，记录 `_llm_disabled_at`，超过
   `LLM_CIRCUIT_COOLDOWN_SECONDS` 自动恢复一次探测。

3. **冷却被消息流「续期」（2026-10-14 深夜实锤，v4.1.12 修复）**：禁用期间每条消息
   仍会走到失败分支，`_llm_failures` 只增不减，导致每次都刷新 `_llm_disabled_at`
   —— 持续聊天时 600s 冷却永远到期不了（线上 14:46 禁用直到 15:02 已 809s 仍不恢复），
   只有群静默超过 10 分钟才自愈。修复：失败计数仅在可用态累计；禁用分支加
   `and self._llm_available` 守卫，只在刚进入禁用时记时间戳 + 打日志。
"""
import asyncio
import os
import sys
import time
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.emotion_expert import EmotionAnalysisExpert  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OK_TEXT = "情感分析结果-" + "x" * 40


class _Meta:
    def __init__(self, pid, model=None):
        self.id = pid
        self.model = model or pid
        self.type = "openai_chat_completion"


class KwargsProvider:
    """text_chat 只带 **kwargs（最常见形态，如线上 FakeProvider/真实 provider）"""

    def __init__(self, pid):
        self._meta = _Meta(pid)
        self.calls = 0
        self.kwargs_seen = None

    def meta(self):
        return self._meta

    async def text_chat(self, prompt=None, model=None, **kwargs):
        self.calls += 1
        self.kwargs_seen = kwargs
        return _OK_TEXT


class ExplicitProvider:
    """text_chat 显式声明 request_max_retries"""

    def __init__(self, pid):
        self._meta = _Meta(pid)
        self.calls = 0
        self.kwargs_seen = None

    def meta(self):
        return self._meta

    async def text_chat(self, prompt=None, model=None, request_max_retries=None):
        self.calls += 1
        self.kwargs_seen = {"request_max_retries": request_max_retries}
        return _OK_TEXT


class StrictProvider:
    """text_chat 既不接受 request_max_retries 也没有 **kwargs"""

    def __init__(self, pid):
        self._meta = _Meta(pid)
        self.calls = 0

    def meta(self):
        return self._meta

    async def text_chat(self, prompt=None, model=None):
        self.calls += 1
        return _OK_TEXT


class RaisingProvider:
    def __init__(self, pid):
        self._meta = _Meta(pid)
        self.calls = 0

    def meta(self):
        return self._meta

    async def text_chat(self, prompt=None, model=None, **kwargs):
        self.calls += 1
        raise RuntimeError("429 inference exceeds tpm/rpm limit")


class FakeContext:
    def __init__(self, providers=()):
        self._providers = list(providers)

    def get_all_providers(self):
        return list(self._providers)

    def get_using_provider(self, umo=None):
        return None

    def get_provider_by_id(self, pid):
        return None

    def get_config(self, umo=None):
        return {"provider_settings": {"fallback_chat_models": []}}


class FakeCache:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value


def _expert(providers=(), cache=None):
    return EmotionAnalysisExpert(
        cache=cache,
        context=FakeContext(providers),
        secondary_llm_provider=None,
        secondary_llm_model=None,
        time_budget=45.0,
        max_providers=3,
    )


def _state():
    return EnhancedEmotionalState(user_key="u1", favor=50, intimacy=50)


# ==================== 429 快速失败 ====================


class TestFastFailOn429(unittest.TestCase):
    """插件侧必须传 request_max_retries=1，避免 tenacity 5 次退避烧满时间片"""

    def test_supported_provider_receives_request_max_retries_1(self):
        p = KwargsProvider("vendor/a")
        e = _expert([p])
        result = asyncio.run(e._execute_llm_call(p, "分析这段对话"))
        self.assertEqual(result, _OK_TEXT)
        self.assertEqual(p.kwargs_seen, {"request_max_retries": 1})

    def test_explicit_signature_provider_receives_value_1(self):
        p = ExplicitProvider("vendor/a")
        e = _expert([p])
        self.assertTrue(e._text_chat_supports_fast_fail(p))
        result = asyncio.run(e._execute_llm_call(p, "分析这段对话"))
        self.assertEqual(result, _OK_TEXT)
        self.assertEqual(p.kwargs_seen, {"request_max_retries": 1})

    def test_unsupported_provider_called_without_the_param(self):
        """不接受该参数的 provider 必须退回普通调用，行为与改动前一致"""
        p = StrictProvider("vendor/a")
        e = _expert([p])
        self.assertFalse(e._text_chat_supports_fast_fail(p))
        result = asyncio.run(e._execute_llm_call(p, "分析这段对话"))
        self.assertEqual(result, _OK_TEXT)
        self.assertEqual(p.calls, 1)

    def test_support_detection_is_cached_per_class(self):
        p1 = KwargsProvider("vendor/a")
        p2 = KwargsProvider("vendor/b")
        e = _expert([p1, p2])
        self.assertTrue(e._text_chat_supports_fast_fail(p1))
        self.assertIn(KwargsProvider, e._fast_fail_support_cache)
        # 同一类第二个实例直接命中缓存（不重新 inspect）
        self.assertTrue(e._text_chat_supports_fast_fail(p2))

    def test_strict_provider_signature_without_meta_callable(self):
        """签名检测对奇怪对象也必须返回 False 而不是抛异常"""
        class Weird:
            pass

        e = _expert()
        self.assertFalse(e._text_chat_supports_fast_fail(Weird()))

    def test_raising_provider_still_switches_quickly(self):
        """429 型异常下备选链照常切换下一个候选（快速失败不破坏链逻辑）"""
        p1 = RaisingProvider("vendor/a")
        p2 = KwargsProvider("vendor/b")
        e = _expert([p1, p2])
        result = asyncio.run(
            e._call_real_llm_with_retry("你好", "你好呀", _state(), "umo-x")
        )
        self.assertEqual(result, _OK_TEXT)
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 1)


# ==================== 断路器冷却复位 ====================


class TestCircuitCooldown(unittest.TestCase):
    """禁用必须变成「冷却」，期满自动恢复探测"""

    def test_constant_is_reasonable(self):
        self.assertGreaterEqual(
            EmotionAnalysisExpert.LLM_CIRCUIT_COOLDOWN_SECONDS, 60
        )
        self.assertLessEqual(
            EmotionAnalysisExpert.LLM_CIRCUIT_COOLDOWN_SECONDS, 3600
        )

    def test_noop_when_available(self):
        e = _expert()
        e._maybe_reenable_llm()
        self.assertTrue(e._llm_available)
        self.assertEqual(e._llm_failures, 0)

    def test_noop_when_never_disabled(self):
        """_llm_disabled_at 为 0（未禁用）时不得误恢复"""
        e = _expert()
        e._llm_available = False
        e._llm_disabled_at = 0.0
        e._maybe_reenable_llm()
        self.assertFalse(e._llm_available)

    def test_stays_disabled_within_cooldown(self):
        e = _expert()
        e._llm_available = False
        e._llm_failures = 3
        e._llm_disabled_at = time.time() - 10
        e._maybe_reenable_llm()
        self.assertFalse(e._llm_available)
        self.assertEqual(e._llm_failures, 3)

    def test_reenables_after_cooldown(self):
        e = _expert()
        e._llm_available = False
        e._llm_failures = 3
        e._llm_disabled_at = (
            time.time() - EmotionAnalysisExpert.LLM_CIRCUIT_COOLDOWN_SECONDS - 1
        )
        e._maybe_reenable_llm()
        self.assertTrue(e._llm_available)
        self.assertEqual(e._llm_failures, 0)

    def test_reset_clears_timestamp(self):
        e = _expert()
        e._llm_available = False
        e._llm_failures = 5
        e._llm_disabled_at = time.time()
        e.reset_llm_availability()
        self.assertTrue(e._llm_available)
        self.assertEqual(e._llm_failures, 0)
        self.assertEqual(e._llm_disabled_at, 0.0)


class TestCircuitEndToEnd(unittest.TestCase):
    """整条 analyze 链路上的禁用 / 冷却 / 恢复"""

    def _round(self, e, msg):
        return asyncio.run(
            e.analyze_and_update_emotion("u1", msg, "回复", _state(), "umo-x")
        )

    def test_three_failures_disable_and_record_timestamp(self):
        p = RaisingProvider("vendor/a")
        cache = FakeCache()
        e = _expert([p], cache=cache)
        e.llm_retry_delay = 0

        for i in range(3):
            updates = self._round(e, f"消息{i}")
            self.assertEqual(updates["source"], "smart_fallback")

        self.assertFalse(e._llm_available)
        self.assertGreater(e._llm_disabled_at, 0.0)

    def test_disabled_round_skips_llm_entirely(self):
        """冷却期内的轮次完全不调 LLM（provider.calls 不变）"""
        p = RaisingProvider("vendor/a")
        cache = FakeCache()
        e = _expert([p], cache=cache)
        e.llm_retry_delay = 0
        for i in range(3):
            self._round(e, f"消息{i}")
        calls_after_disable = p.calls

        updates = self._round(e, "冷却期内的消息")
        self.assertEqual(updates["source"], "smart_fallback")
        self.assertEqual(p.calls, calls_after_disable)

    def test_recovers_after_cooldown_and_succeeds(self):
        """冷却期满后恢复探测，且 provider 可用时真的走 llm_analysis"""
        p = RaisingProvider("vendor/a")
        cache = FakeCache()
        e = _expert([p], cache=cache)
        e.llm_retry_delay = 0
        for i in range(3):
            self._round(e, f"消息{i}")
        self.assertFalse(e._llm_available)

        # 模拟冷却期满（把禁用时刻拨回过去）
        e._llm_disabled_at = (
            time.time() - EmotionAnalysisExpert.LLM_CIRCUIT_COOLDOWN_SECONDS - 1
        )

        # 上游恢复：把 provider 换成可用的
        ok = KwargsProvider("vendor/a")
        e.context = FakeContext([ok])
        updates = self._round(e, "冷却后的消息")
        self.assertEqual(updates["source"], "llm_analysis")
        self.assertEqual(ok.calls, 1)
        self.assertTrue(e._llm_available)

    def test_success_resets_failure_count(self):
        """旧行为保持：成功即清零失败计数"""
        ok = KwargsProvider("vendor/a")
        cache = FakeCache()
        e = _expert([ok], cache=cache)
        e._llm_failures = 2
        updates = self._round(e, "一条消息")
        self.assertEqual(updates["source"], "llm_analysis")
        self.assertEqual(e._llm_failures, 0)


# ==================== 冷却不被消息流续期（v4.1.12） ====================


class TestCooldownNotRefreshedByTraffic(unittest.TestCase):
    """禁用期间的消息流不得刷新冷却计时器

    线上实锤（2026-10-14）：14:46:47 禁用后，14:47:49→15:02:37 每条消息都重打
    禁用 ERRO；15:00:16 距首次禁用已 809s（远超 600s 冷却）仍不恢复，且期间
    没有任何 LLM 尝试。根因：禁用期间每条消息仍走失败分支，`_llm_failures`
    只增不减，导致每次都执行 `_llm_disabled_at = time.time()`——冷却被不断
    续期，只有群静默超过 10 分钟才自愈。
    """

    def _round(self, e, msg):
        return asyncio.run(
            e.analyze_and_update_emotion("u1", msg, "回复", _state(), "umo-x")
        )

    def _disabled_expert(self):
        p = RaisingProvider("vendor/a")
        cache = FakeCache()
        e = _expert([p], cache=cache)
        e.llm_retry_delay = 0
        for i in range(3):
            self._round(e, f"消息{i}")
        self.assertFalse(e._llm_available)
        return e

    def test_traffic_during_cooldown_does_not_refresh_timestamp(self):
        """冷却期内的每条消息都不得刷新 _llm_disabled_at"""
        e = self._disabled_expert()
        disabled_at = e._llm_disabled_at
        for i in range(5):
            self._round(e, f"冷却期消息{i}")
        self.assertEqual(e._llm_disabled_at, disabled_at)

    def test_failure_counter_frozen_while_disabled(self):
        """禁用期间不累加失败计数（计数只反映可用状态下的连续失败）"""
        e = self._disabled_expert()
        for i in range(5):
            self._round(e, f"冷却期消息{i}")
        self.assertEqual(e._llm_failures, 3)

    def test_disable_error_not_repeated_during_cooldown(self):
        """每次进入禁用只打一条 ERRO；冷却期内的消息不得重复刷屏"""
        e = self._disabled_expert()
        with mock.patch("emotionai_pro.emotion_expert.logger") as m:
            for i in range(4):
                self._round(e, f"冷却期消息{i}")
        repeated = [
            c for c in m.error.call_args_list if "暂时禁用LLM分析" in str(c)
        ]
        self.assertEqual(repeated, [])

    def test_steady_traffic_still_recovers_after_cooldown(self):
        """持续消息流（条间隔 < 冷却时长）下，冷却仍从首次禁用起算

        旧代码在此场景永远恢复不了：每条消息都把 _llm_disabled_at 刷成当前时刻。
        用可拨动的时钟模拟 7 条消息、每条间隔 90s（总跨度 630s > 600s，任何
        相邻两条间隔都 < 600s）——只有最后一条应触发恢复探测。
        """
        e = self._disabled_expert()
        first_disable = e._llm_disabled_at
        ok = KwargsProvider("vendor/a")
        e.context = FakeContext([ok])

        clock = {"t": first_disable}
        fake_time = mock.Mock()
        fake_time.time.side_effect = lambda: clock["t"]
        updates = None
        with mock.patch("emotionai_pro.emotion_expert.time", fake_time):
            for i in range(7):
                clock["t"] = first_disable + 90 * (i + 1)
                updates = self._round(e, f"持续消息{i}")
        self.assertEqual(updates["source"], "llm_analysis")
        self.assertEqual(ok.calls, 1)


# ==================== 源码级硬约束 ====================


class TestSourceConstraints(unittest.TestCase):
    """锁定修复的形态，防止回归"""

    def _read_source(self, filename):
        with open(os.path.join(_ROOT, filename), "r", encoding="utf-8") as f:
            return f.read()

    def test_request_max_retries_only_in_expert_chain_call(self):
        """request_max_retries 只允许出现在情感分析备选链的调用点

        4 处 = __init__ 注释 + 实际传参 + 检测方法 docstring + 检测逻辑；
        除第 2 处外都只是说明文字，真正的传值只有 call_kwargs 那一行。
        """
        src = self._read_source("emotion_expert.py")
        self.assertEqual(src.count("request_max_retries"), 4)
        self.assertIn('call_kwargs["request_max_retries"] = 1', src)

    def test_main_py_does_not_pass_request_max_retries(self):
        """主对话链路不得受本修复影响"""
        src = self._read_source("main.py")
        self.assertNotIn("request_max_retries", src)

    def test_cooldown_hook_called_before_gate(self):
        src = self._read_source("emotion_expert.py")
        self.assertIn("self._maybe_reenable_llm()", src)
        # 恢复探测必须发生在 `if self._llm_available:` 之前
        self.assertLess(
            src.index("self._maybe_reenable_llm()"),
            src.index("if self._llm_available:"),
        )

    def test_disable_branch_records_timestamp(self):
        src = self._read_source("emotion_expert.py")
        self.assertIn("self._llm_disabled_at = time.time()", src)

    def test_disable_branch_guarded_by_availability(self):
        """禁用分支必须带 `and self._llm_available` 守卫（v4.1.12）

        没有这个守卫，禁用期间每条消息都会刷新 _llm_disabled_at，
        600s 冷却在持续消息流下永远到期不了（线上实锤 809s 仍不恢复）。
        """
        src = self._read_source("emotion_expert.py")
        self.assertIn(
            "if self._llm_failures >= 3 and self._llm_available:", src
        )


if __name__ == "__main__":
    unittest.main()
