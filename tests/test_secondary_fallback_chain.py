# tests/test_secondary_fallback_chain.py
"""v4.2.3：辅助 LLM 退避链（有序多选配置 + 专家链显式优先）

背景：v4.2.2 把「辅助 LLM 提供商」改成从已配置模型下拉选择，但只能选一个；
用户要求失败时还能按自己排的顺序退避，因此新增
`secondary_llm_fallback_providers`（多选有序 provider id 列表，WebUI 用与
ds-flash「回退对话模型列表」相同的 ProviderSelector 多选）。

本文件锁定的行为契约：
1. 配置了退避链时，专家链 = [辅助 LLM（若显式配置且匹配）] + [退避链（按配置
   顺序）] + 自动链（主 LLM → 档案 fallback_chat_models → 其余 provider）。
2. **显式段保持用户顺序**：不参与自动健康分排序、不会被健康的自动候选挤到后面。
3. 未配置退避链时，链构造与 v4.1.14 完全一致（整链交给健康分排序）。
4. 未知/非法 id 一律跳过并告警，不崩。
5. 配置五处联动：config.py / main.py base_mapping+构造参数 / schema_validator /
   _conf_schema.json / 真实消费方（专家链）。
"""
import asyncio
import json
import os
import sys
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
    def __init__(self, pid, model=None):
        self.id = pid
        self.model = model or pid
        self.type = "openai_chat_completion"


class FakeProvider:
    """可控行为的 provider 桩。behavior: ok / empty / raise"""

    def __init__(self, pid, behavior="ok"):
        self._meta = _Meta(pid)
        self.behavior = behavior
        self.calls = 0

    def meta(self):
        return self._meta

    async def text_chat(self, prompt=None, model=None, **kwargs):
        self.calls += 1
        if self.behavior == "raise":
            raise RuntimeError(f"{self._meta.id} 故意失败")
        if self.behavior == "empty":
            return ""
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


def _expert(context=None, provider=None, fallback_ids=None, budget=60.0,
            max_providers=3):
    return EmotionAnalysisExpert(
        cache=FakeCache(),
        context=context,
        secondary_llm_provider=provider,
        secondary_llm_model=None,
        fallback_provider_ids=fallback_ids,
        time_budget=budget,
        max_providers=max_providers,
    )


def _names(e, providers):
    return [e._get_provider_name(p) for p in providers]


def _state():
    return EnhancedEmotionalState(user_key="u1", favor=50, intimacy=50)


def _read_source(filename):
    with open(os.path.join(_ROOT, filename), "r", encoding="utf-8") as f:
        return f.read()


class TestChainOrder(unittest.TestCase):
    """显式退避链在链中的位置与顺序"""

    def _ctx(self):
        secondary = FakeProvider("vendor/secondary")
        fb1 = FakeProvider("vendor/fb1")
        fb2 = FakeProvider("vendor/fb2")
        main = FakeProvider("vendor/main")
        rest = FakeProvider("vendor/rest")
        providers = [rest, secondary, fb1, fb2, main]
        ctx = FakeContext(
            providers=providers,
            main=main,
            by_id={p.meta().id: p for p in providers},
        )
        return ctx, secondary, fb1, fb2, main, rest

    def test_configured_fallback_chain_in_user_order_after_secondary(self):
        ctx, secondary, fb1, fb2, main, rest = self._ctx()
        e = _expert(context=ctx, provider="vendor/secondary",
                    fallback_ids=["vendor/fb1", "vendor/fb2"])
        self.assertEqual(
            _names(e, e._build_provider_chain()),
            ["vendor/secondary", "vendor/fb1", "vendor/fb2",
             "vendor/main", "vendor/rest"],
        )

    def test_no_configured_chain_keeps_legacy_order(self):
        """留空 → 与 v4.1.14 相同：辅助 LLM → 主 LLM → 其余（含全部 provider）"""
        ctx, secondary, fb1, fb2, main, rest = self._ctx()
        e = _expert(context=ctx, provider="vendor/secondary", fallback_ids=None)
        self.assertEqual(
            _names(e, e._build_provider_chain()),
            ["vendor/secondary", "vendor/main", "vendor/rest",
             "vendor/fb1", "vendor/fb2"],
        )

    def test_unknown_and_duplicate_ids_are_skipped(self):
        ctx, secondary, fb1, fb2, main, rest = self._ctx()
        e = _expert(
            context=ctx,
            provider="vendor/secondary",
            fallback_ids=["vendor/not-exist", "vendor/fb2", "vendor/fb2",
                          "", 123, "  "],
        )
        self.assertEqual(
            _names(e, e._build_provider_chain()),
            ["vendor/secondary", "vendor/fb2", "vendor/main", "vendor/rest",
             "vendor/fb1"],
        )

    def test_configured_chain_without_secondary_still_used_first(self):
        """只配了退避链、没配首选时：退避链仍在自动链之前被尝试"""
        ctx, secondary, fb1, fb2, main, rest = self._ctx()
        e = _expert(context=ctx, provider=None,
                    fallback_ids=["vendor/fb2"])
        self.assertEqual(
            _names(e, e._build_provider_chain()),
            ["vendor/fb2", "vendor/main", "vendor/rest",
             "vendor/secondary", "vendor/fb1"],
        )


class TestExplicitOrderSurvivesHealthSorting(unittest.TestCase):
    """显式段不参与自动健康分排序（用户配置的顺序就是优先级）"""

    def _ctx(self):
        secondary = FakeProvider("vendor/secondary", behavior="raise")
        fb1 = FakeProvider("vendor/fb1", behavior="raise")
        fb2 = FakeProvider("vendor/fb2", behavior="ok")
        main = FakeProvider("vendor/main", behavior="ok")
        providers = [secondary, fb1, fb2, main]
        return FakeContext(
            providers=providers,
            main=main,
            by_id={p.meta().id: p for p in providers},
        ), secondary, fb1, fb2, main

    def test_explicit_order_not_reordered_by_health(self):
        ctx, secondary, fb1, fb2, main = self._ctx()
        e = _expert(context=ctx, provider="vendor/secondary",
                    fallback_ids=["vendor/fb1", "vendor/fb2"],
                    max_providers=4)
        # 人为制造健康分鸿沟：自动段 main 全成功、显式段 fb1 全失败
        for _ in range(5):
            e._provider_health.record(e._provider_health.provider_key(main), True)
        for _ in range(5):
            e._provider_health.record(e._provider_health.provider_key(fb1), False)

        order = []

        orig = e._execute_llm_call

        async def tracking(provider, prompt, timeout=None):
            order.append(provider.meta().id)
            return await orig(provider, prompt, timeout=timeout)

        e._execute_llm_call = tracking

        result = asyncio.run(
            e._call_real_llm_with_retry("你好", "你好", _state(), "umo-x")
        )
        self.assertEqual(result, _OK_TEXT)
        # 失败的首选/退避链第一项依次尝试，然后 fb2 成功（main 健康分再高也不能插队）
        self.assertEqual(order[:3],
                         ["vendor/secondary", "vendor/fb1", "vendor/fb2"])

    def test_auto_segment_still_health_sorted(self):
        """未配置退避链时，整条自动链仍按健康分排序（v4.1.14 不回归）"""
        keep = FakeProvider("vendor/keep", behavior="ok")
        dead = FakeProvider("vendor/dead", behavior="raise")
        main = FakeProvider("vendor/main", behavior="ok")
        ctx = FakeContext(
            providers=[dead, keep, main],
            main=main,
            by_id={p.meta().id: p for p in (dead, keep, main)},
        )
        e = _expert(context=ctx, provider=None, max_providers=3)
        for _ in range(5):
            e._provider_health.record(e._provider_health.provider_key(keep), True)
        for _ in range(5):
            e._provider_health.record(e._provider_health.provider_key(dead), False)

        order = []
        orig = e._execute_llm_call

        async def tracking(provider, prompt, timeout=None):
            order.append(provider.meta().id)
            return await orig(provider, prompt, timeout=timeout)

        e._execute_llm_call = tracking
        result = asyncio.run(
            e._call_real_llm_with_retry("你好", "你好", _state(), "umo-x")
        )
        self.assertEqual(result, _OK_TEXT)
        # 未配置显式链 → 整条自动链按健康分排序：稳定的 keep 升到最前并直接成功
        self.assertEqual(order, ["vendor/keep"])


class TestConfigPlumbing(unittest.TestCase):
    """新增配置字段必须五处联动 + 真实消费"""

    def test_pydantic_default_and_type(self):
        cfg = PluginConfig()
        self.assertEqual(cfg.secondary_llm_fallback_providers, [])
        cfg = PluginConfig(
            secondary_llm_fallback_providers=["agnes/agnes-2.5-flash"],
        )
        self.assertEqual(cfg.secondary_llm_fallback_providers,
                         ["agnes/agnes-2.5-flash"])

    def test_webui_values_reach_plugin_config(self):
        from emotionai_pro.main import EmotionAIProPlugin

        plugin = type("P", (), {"_raw_config": None})()
        cfg = EmotionAIProPlugin._load_and_validate_config(
            plugin,
            {"secondary_llm_fallback_providers": ["vendor/a", "vendor/b"]},
        )
        self.assertEqual(cfg.secondary_llm_fallback_providers,
                         ["vendor/a", "vendor/b"])

    def test_base_mapping_and_expert_forwarding(self):
        src = _read_source("main.py")
        self.assertIn(
            '"secondary_llm_fallback_providers": "secondary_llm_fallback_providers"',
            src,
        )
        self.assertIn(
            "fallback_provider_ids=self.config.secondary_llm_fallback_providers",
            src,
        )

    def test_schema_validator_covers_field(self):
        from emotionai_pro.schema_validator import ConfigValidator

        props = ConfigValidator.CONFIG_SCHEMA["properties"]
        self.assertIn("secondary_llm_fallback_providers", props)
        node = props["secondary_llm_fallback_providers"]
        self.assertEqual(node["type"], "array")
        self.assertEqual(node["items"]["type"], "string")

    def test_conf_schema_uses_multi_provider_selector(self):
        with open(os.path.join(_ROOT, "_conf_schema.json"), "r",
                  encoding="utf-8") as f:
            schema = json.load(f)
        node = schema["secondary_llm_fallback_providers"]
        self.assertEqual(node["type"], "list")
        self.assertEqual(node["_special"], "select_providers")
        self.assertEqual(node["default"], [])

    def test_expert_normalizes_bad_input(self):
        e = _expert(fallback_ids="not-a-list")
        self.assertEqual(e.secondary_llm_fallback_providers, [])
        e = _expert(fallback_ids=[" a/b ", "a/b", None, 3, "c/d"])
        self.assertEqual(e.secondary_llm_fallback_providers, ["a/b", "c/d"])


if __name__ == "__main__":
    unittest.main()
