# tests/test_provider_exclusion.py
"""v4.2.4：不合适的视觉模型不进入情感分析专家链

背景：v4.1.14 起的「备选链按 provider 健康分动态排序」随 v4.2.4 自动段
移除而整体删除（情感分析只走用户配置的「首选辅助 LLM + 退避链」，严格
按配置顺序，不做动态调整；ProviderHealthTracker 一并移除）。本文件
（原 test_provider_health_order.py）保留的约束：siliconflow 的 Qwen3-VL
等视觉模型不得进入纯文本情感分析的候选链——provider 本身仍保留在
AstrBot 配置里，meme_manager 等视觉功能可按需使用。
"""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.emotion_expert import EmotionAnalysisExpert  # noqa: E402

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
        # v4.2.4 起链只走配置：未配置的 step-5-preview 不进链
        self.assertNotIn("step/step-5-preview", ids)
        # provider 本身还在 AstrBot 列表里，只是不进专家链
        self.assertIn(vl, ctx.get_all_providers())

    def test_vision_provider_excluded_even_as_explicit_secondary(self):
        """即使有人把辅助 LLM 配成这个视觉模型，也不能进入专家链"""
        ctx, vl, main, agnes = self._ctx()
        e = _expert(context=ctx, provider=self.VL_ID, max_providers=2)
        chain = e._build_provider_chain()
        ids = [e._get_provider_meta(p)[0] for p in chain]
        self.assertNotIn(self.VL_ID, ids)
        # 首选被排除且没配退避链 → 空链（不再自动补候选）
        self.assertEqual(ids, [])

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
        self.assertEqual(ids, [])


if __name__ == "__main__":
    unittest.main()
