# tests/test_provider_resolution.py
"""EmotionAnalysisExpert 的 provider 解析回归测试。

覆盖的 bug：早期实现读取并不存在的 `provider.name` 属性，
元信息退化成类名（如 `ProviderOpenAIOfficial`），导致

1. `secondary_llm_provider` 配置项完全失效；
2. 名称匹配永远不命中，最终落到 `providers[0]`——线上实测是视觉模型
   `Qwen/Qwen3-VL-30B-A3B-Instruct`，而不是配置里声明的
   `商汤/deepseek-v4-flash`。

同时覆盖 `secondary_llm_model` 此前从未被使用的问题。
"""
import os
import sys
import asyncio
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.emotion_expert import EmotionAnalysisExpert  # noqa: E402


class _Meta:
    """对应 astrbot.core.provider.entities.ProviderMeta"""

    def __init__(self, pid, model):
        self.id = pid
        self.model = model
        self.type = "openai_chat_completion"


class FakeProvider:
    """模拟 AstrBot Provider：只有 meta()，**没有** name 属性"""

    def __init__(self, pid, model=None):
        self._meta = _Meta(pid, model)
        self.text_chat_models = []

    def meta(self):
        return self._meta

    async def text_chat(self, prompt=None, model=None, **kwargs):
        self.text_chat_models.append(model)
        return f"mock-response-from-{self._meta.id}"


class FakeContext:
    def __init__(self, main=None):
        self._main = main
        self.received_umos = []

    def get_using_provider(self, umo=None):
        self.received_umos.append(umo)
        return self._main


class LegacyContext:
    """不接受 umo 参数的旧版签名"""

    def __init__(self, main=None):
        self._main = main

    def get_using_provider(self):
        return self._main


def _expert(context=None, provider=None, model=None):
    return EmotionAnalysisExpert(
        cache=None,
        context=context,
        secondary_llm_provider=provider,
        secondary_llm_model=model,
    )


def _default_providers():
    """还原线上顺序：视觉模型排在前面，deepseek 在后"""
    return [
        FakeProvider(
            "siliconflow/Qwen/Qwen3-VL-30B-A3B-Instruct",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        ),
        FakeProvider("商汤/deepseek-v4-flash", "deepseek-v4-flash"),
    ]


class TestProviderMeta(unittest.TestCase):
    """元信息读取必须走 meta()，而不是不存在的 name"""

    def test_reads_id_and_model_from_meta(self):
        e = _expert()
        pid, model = e._get_provider_meta(
            FakeProvider("商汤/deepseek-v4-flash", "deepseek-v4-flash")
        )
        self.assertEqual(pid, "商汤/deepseek-v4-flash")
        self.assertEqual(model, "deepseek-v4-flash")

    def test_provider_name_is_id_not_class_name(self):
        """回归点：修复前这里返回的是 'FakeProvider' 这类类名"""
        e = _expert()
        self.assertEqual(
            e._get_provider_name(FakeProvider("siliconflow/Qwen3-VL")),
            "siliconflow/Qwen3-VL",
        )

    def test_broken_meta_does_not_raise(self):
        class Broken:
            def meta(self):
                raise RuntimeError("boom")

        e = _expert()
        pid, model = e._get_provider_meta(Broken())
        self.assertEqual(pid, "")
        self.assertEqual(model, "")
        self.assertEqual(e._get_provider_name(Broken()), "Broken")


class TestFindTargetProvider(unittest.TestCase):
    def test_explicit_provider_matched_by_id(self):
        providers = _default_providers()
        e = _expert(context=FakeContext(main=providers[0]), provider="商汤/deepseek")
        self.assertIs(e._find_target_provider(providers), providers[1])

    def test_explicit_provider_matched_by_model_name(self):
        providers = _default_providers()
        e = _expert(context=FakeContext(main=None), provider="Qwen3-VL")
        self.assertIs(e._find_target_provider(providers), providers[0])

    def test_empty_config_uses_main_provider(self):
        """留空 → 使用主 LLM（与 _conf_schema 的 hint 一致）"""
        providers = _default_providers()
        main = providers[1]
        e = _expert(context=FakeContext(main=main))
        self.assertIs(e._find_target_provider(providers), main)

    def test_unmatched_explicit_config_falls_back_to_main(self):
        providers = _default_providers()
        main = providers[1]
        e = _expert(context=FakeContext(main=main), provider="不存在的提供商")
        self.assertIs(e._find_target_provider(providers), main)

    def test_falls_back_to_deepseek_when_no_main(self):
        providers = _default_providers()
        e = _expert(context=FakeContext(main=None))
        self.assertIs(e._find_target_provider(providers), providers[1])

    def test_falls_back_to_first_when_nothing_matches(self):
        providers = [FakeProvider("vendor/vl-model", "vl-model")]
        e = _expert(context=FakeContext(main=None))
        self.assertIs(e._find_target_provider(providers), providers[0])

    def test_empty_provider_list_returns_none(self):
        e = _expert(context=FakeContext(main=None))
        self.assertIsNone(e._find_target_provider([]))

    def test_context_without_api_does_not_raise(self):
        """context 没有 get_using_provider 时不能崩，应继续走名称匹配"""
        providers = _default_providers()
        e = _expert(context=object())
        self.assertIs(e._find_target_provider(providers), providers[1])

    def test_raising_context_does_not_raise(self):
        class BadContext:
            def get_using_provider(self, umo=None):
                raise RuntimeError("provider manager down")

        providers = _default_providers()
        e = _expert(context=BadContext())
        self.assertIs(e._find_target_provider(providers), providers[1])


class TestUmoThreading(unittest.TestCase):
    """umo 必须一路传到 get_using_provider。

    回归点：不传 umo 时 AstrBot 会读**全局** cmd_config.json 的
    default_provider_id，而实际生效的是 WebUI 配置档案（ds-flash），
    两者可能指向不同 provider——线上就因此拿到了一个已失效的 provider。
    """

    def test_umo_forwarded_to_get_using_provider(self):
        providers = _default_providers()
        ctx = FakeContext(main=providers[1])
        e = _expert(context=ctx)
        e._find_target_provider(providers, "aiocqhttp:GroupMessage:123456")
        self.assertEqual(ctx.received_umos, ["aiocqhttp:GroupMessage:123456"])

    def test_umo_none_when_not_passed(self):
        providers = _default_providers()
        ctx = FakeContext(main=providers[1])
        e = _expert(context=ctx)
        e._find_target_provider(providers)
        self.assertEqual(ctx.received_umos, [None])

    def test_legacy_signature_without_umo_still_works(self):
        providers = _default_providers()
        ctx = LegacyContext(main=providers[1])
        e = _expert(context=ctx)
        self.assertIs(e._find_target_provider(providers, "umo-x"), providers[1])

    def test_explicit_provider_skips_main_lookup(self):
        providers = _default_providers()
        ctx = FakeContext(main=providers[0])
        e = _expert(context=ctx, provider="商汤/deepseek")
        self.assertIs(e._find_target_provider(providers, "umo-x"), providers[1])
        self.assertEqual(ctx.received_umos, [])


class TestSecondaryLlmModel(unittest.TestCase):
    """secondary_llm_model 必须真正生效（此前是死配置）"""

    def test_target_model_none_when_unset(self):
        self.assertIsNone(_expert()._target_model())
        self.assertIsNone(_expert(model="")._target_model())
        self.assertIsNone(_expert(model="   ")._target_model())

    def test_target_model_strips_and_returns(self):
        self.assertEqual(_expert(model=" glm-5.2 ")._target_model(), "glm-5.2")

    def test_model_forwarded_to_text_chat(self):
        provider = FakeProvider("商汤/deepseek-v4-flash", "deepseek-v4-flash")
        e = _expert(model="glm-5.2")
        asyncio.run(e._execute_llm_call(provider, "分析这段对话"))
        self.assertEqual(provider.text_chat_models, ["glm-5.2"])

    def test_model_none_when_not_configured(self):
        provider = FakeProvider("商汤/deepseek-v4-flash", "deepseek-v4-flash")
        e = _expert()
        asyncio.run(e._execute_llm_call(provider, "分析这段对话"))
        self.assertEqual(provider.text_chat_models, [None])


if __name__ == "__main__":
    unittest.main()
