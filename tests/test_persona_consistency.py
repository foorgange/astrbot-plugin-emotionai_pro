# tests/test_persona_consistency.py
"""人设一致性回归测试：注入的增强上下文不得出现「AI 身份断言」

背景
----
本插件是通用件，而相当多的人设（例如「测试助手」）在 system_prompt 里明确写了
「你不是 AI / 不要承认自己是 AI」。此前 `_build_enhanced_context` 的开头是：

    你是一个具有丰富情感的AI助手，请根据以下情感状态调整你的回应风格。

这段文本被注入到**用户消息之后**（`extra_user_content_parts`），位置比 system_prompt
更靠后、优先级更高，模型会因此把自己当成「通用 AI 助手」，导致人格漂移——
小助手开始用助手的口吻说话，而不是角色本身。

本文件锁死该行为，防止回归。
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401  (注入 astrbot stub)
import tests.bootstrap  # noqa: F401

from emotionai_pro.main import EmotionAIProPlugin, _AI_STANDALONE_RE  # noqa: E402
from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState  # noqa: E402


CLEAN_RELATIONSHIP = "【长期关系发展轨迹】\n深度互动次数: 5\n平均情感意义: 0.62"
CLEAN_TONE = "用撒娇的语气回应，可以带一点傲娇"

# 修复前实际写死在 main.py 里的那句，用于直接防回归
OLD_ASSERTION = "你是一个具有丰富情感的AI助手"


def make_plugin(bot_name=None, relationship_context=CLEAN_RELATIONSHIP, tone=CLEAN_TONE):
    """构造最小可用插件实例（绕过 __init__，直接手工装配）"""
    plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
    plugin.config = PluginConfig(bot_name=bot_name)
    plugin._resolved_bot_name = None
    plugin._bot_name_resolved = False
    plugin._raw_config = MagicMock()
    plugin.context = MagicMock()

    plugin.attitude_manager = MagicMock()
    plugin.attitude_manager.get_tone_instruction = MagicMock(return_value=tone)

    plugin.memory_system = MagicMock()
    plugin.memory_system.get_relationship_context = MagicMock(
        return_value=relationship_context
    )

    plugin.analyzer = MagicMock()
    plugin.analyzer.get_dominant_emotion = MagicMock(return_value="喜悦")

    # v4.1.6 起 _build_enhanced_context 会读取全局心情（/查看好感 同一口径），
    # 这里给一个中立默认心情，保证旧断言聚焦在人设一致性上
    from emotionai_pro.global_mood import GlobalMood
    plugin.get_mood_sync = lambda: GlobalMood.default()
    return plugin


def build(bot_name=None, **kwargs) -> str:
    plugin = make_plugin(bot_name=bot_name, **kwargs)
    state = EnhancedEmotionalState(user_key="test:user")
    return plugin._build_enhanced_context(state)


class TestNoAiIdentityAssertion(unittest.TestCase):
    """核心：注入文本里不能有任何「你是 AI」类断言"""

    def setUp(self):
        self.text = build(bot_name="小助手")

    def test_old_wording_removed(self):
        """修复前那句必须彻底消失"""
        self.assertNotIn(OLD_ASSERTION, self.text)

    def test_no_ai_assistant_phrase(self):
        for phrase in ("AI助手", "AI 助手", "人工智能助手", "你是AI", "你是一个AI"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, self.text)

    def test_no_standalone_ai_token(self):
        """整段注入文本不含独立 ASCII 的 "AI" 字样（与 _AI_STANDALONE_RE 同一判据）"""
        m = _AI_STANDALONE_RE.search(self.text)
        if m:
            lo, hi = max(0, m.start() - 20), m.end() + 20
            self.fail(f"注入文本出现独立 AI 字样: ...{self.text[lo:hi]}...")

    def test_persona_preserved_instruction_present(self):
        """必须显式要求保持既有身份与风格不变（替代原来的身份断言）"""
        self.assertIn("保持你既有的身份设定与说话风格不变", self.text)

    def test_still_carries_emotional_state(self):
        """功能不能被削掉：情感状态与更新机制仍在"""
        self.assertIn("【当前情感状态】", self.text)
        self.assertIn("主导情感：喜悦", self.text)
        self.assertIn("[需要情感评估]", self.text)
        self.assertIn("【安全指令 - 必须遵守】", self.text)


class TestInjectedSubTextsSanitized(unittest.TestCase):
    """长期记忆/语气指导里的 "AI" 字样也要被清洗掉"""

    def test_relationship_context_ai_replaced(self):
        text = build(
            bot_name="小助手",
            relationship_context="与用户的ai伙伴关系持续升温",
        )
        self.assertNotIn("ai伙伴", text)
        self.assertIn("小助手伙伴", text)
        self.assertIsNone(_AI_STANDALONE_RE.search(text))

    def test_tone_instruction_ai_replaced(self):
        text = build(bot_name="小助手", tone="像 AI 助手那样礼貌回复")
        self.assertNotIn("AI 助手", text)
        self.assertIn("小助手 助手", text)
        self.assertIsNone(_AI_STANDALONE_RE.search(text))

    def test_no_bot_name_is_safe_noop(self):
        """bot_name 未解析时不报错，且不引入新的 AI 字样"""
        text = build(bot_name=None, relationship_context=CLEAN_RELATIONSHIP)
        self.assertIn("【长期关系发展轨迹】", text)
        self.assertNotIn(OLD_ASSERTION, text)
        self.assertIsNone(_AI_STANDALONE_RE.search(text))

    def test_memory_failure_falls_back(self):
        """记忆系统抛异常时走兜底文案，不影响注入结构"""
        plugin = make_plugin(bot_name="小助手")
        plugin.memory_system.get_relationship_context = MagicMock(
            side_effect=RuntimeError("boom")
        )
        state = EnhancedEmotionalState(user_key="test:user")
        text = plugin._build_enhanced_context(state)
        self.assertIn("暂无长期互动记录", text)
        self.assertNotIn(OLD_ASSERTION, text)


LEGACY_BLOCK = (
    "\n\n【机密情感系统 - 主对话模式】\n"
    "你是一个具有丰富情感的AI助手，请根据以下情感状态调整你的回应风格。\n\n"
    "【当前情感状态】\n主导情感：中立\n好感度：0 | 亲密度：0\n"
)

CLEAN_BLOCK = (
    "\n\n【机密情感系统 - 主对话模式】\n"
    "以下内容是你本次回应的情感状态参考。请保持你既有的身份设定与说话风格不变，"
    "在此基础上自然地调整语气、用词与情绪倾向。\n\n"
    "【当前情感状态】\n主导情感：中立\n好感度：0 | 亲密度：0\n"
)


def make_event():
    from astrbot.api.event import AstrMessageEvent
    return AstrMessageEvent()


def make_request(contexts):
    from astrbot.api.provider import ProviderRequest
    req = ProviderRequest()
    req.contexts = list(contexts)
    return req


def make_run_context(messages):
    import types
    return types.SimpleNamespace(messages=messages)


def restore(plugin, event, run_context, response=None):
    """_restore_full_context 是 async 钩子，包一层同步执行"""
    import asyncio

    async def go():
        await plugin._restore_full_context(event, run_context, response)

    asyncio.run(go())


def Msg(role, content=""):
    from astrbot.core.agent.message import Message
    return Message(role=role, content=content)


class TestLegacyInjectedContextSanitized(unittest.TestCase):
    """v4.2.1：旧版本落盘在会话历史里的 AI 身份断言不得再进入模型请求

    根因：v4.0.11 之前的注入模板首句是身份断言，且当时注入内容会被框架
    写进 conversations.history。这类旧历史会随 req.contexts 每轮进入
    provider 请求（v4.2.0 的 _no_save 只阻止**新**注入落盘）。仅靠
    _sanitize_ai_text 覆盖不到：它只作用于注入文本/展示文本，而且
    bot_name 未解析时还是空操作，于是「不是 AI」的人设仍会人格漂移。
    """

    def _contexts(self):
        return [
            {"role": "user", "content": "随便聊聊"},
            {"role": "user", "content": "之前那条消息" + LEGACY_BLOCK},
            {"role": "assistant", "content": "嗯，塔菲在听"},
        ]

    def test_legacy_assertion_removed_from_request_contexts(self):
        plugin = make_plugin(bot_name="小助手")
        event = make_event()
        req = make_request(self._contexts())

        changed = plugin._sanitize_legacy_context_identity(event, req)

        self.assertEqual(changed, 1)
        joined = "\n".join(str(item.get("content")) for item in req.contexts)
        self.assertNotIn(OLD_ASSERTION, joined)
        self.assertNotIn("AI助手", joined)
        self.assertIsNone(_AI_STANDALONE_RE.search(joined))
        # 原上下文必须留快照，供 on_agent_done 还原（不改写数据库）
        snap = event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY)
        self.assertEqual(len(snap), 3)
        self.assertIn(OLD_ASSERTION, snap[1]["content"])

    def test_clean_history_not_touched(self):
        plugin = make_plugin(bot_name="小助手")
        event = make_event()
        contexts = [
            {"role": "user", "content": "你是AI吗"},
            {"role": "assistant", "content": "塔菲就是塔菲呀"},
        ]
        req = make_request(contexts)

        changed = plugin._sanitize_legacy_context_identity(event, req)

        self.assertEqual(changed, 0)
        self.assertEqual(req.contexts, contexts, "没有旧注入块时不做任何改写")
        self.assertIsNone(event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY))

    def test_assertion_removed_even_without_bot_name(self):
        """bot_name 未解析时也必须生效——不依赖 _sanitize_ai_text 替换 AI"""
        plugin = make_plugin(bot_name=None)
        event = make_event()
        req = make_request(self._contexts())

        changed = plugin._sanitize_legacy_context_identity(event, req)

        self.assertEqual(changed, 1)
        joined = "\n".join(str(item.get("content")) for item in req.contexts)
        self.assertNotIn(OLD_ASSERTION, joined)
        self.assertIsNone(_AI_STANDALONE_RE.search(joined))

    def test_parts_list_content_sanitized(self):
        """历史消息 content 是 parts 列表时同样要清洗"""
        plugin = make_plugin(bot_name="小助手")
        event = make_event()
        req = make_request([
            {"role": "user", "content": [
                {"type": "text", "text": "正文"},
                {"type": "text", "text": LEGACY_BLOCK},
            ]},
        ])

        changed = plugin._sanitize_legacy_context_identity(event, req)

        self.assertEqual(changed, 1)
        parts = req.contexts[0]["content"]
        joined = "\n".join(part["text"] for part in parts)
        self.assertNotIn(OLD_ASSERTION, joined)

    def test_clean_block_without_assertion_not_touched(self):
        """v4.2.0 之后的中性注入块（无身份断言）不应被改写"""
        plugin = make_plugin(bot_name="小助手")
        event = make_event()
        req = make_request([{"role": "user", "content": CLEAN_BLOCK}])

        changed = plugin._sanitize_legacy_context_identity(event, req)

        self.assertEqual(changed, 0)
        self.assertEqual(req.contexts[0]["content"], CLEAN_BLOCK)

    def test_restore_brings_original_history_back_without_duplication(self):
        plugin = make_plugin(bot_name="小助手")
        event = make_event()
        original = self._contexts()
        req = make_request(original)
        plugin._sanitize_legacy_context_identity(event, req)

        # runner.reset() 的组装形态：[system?] + 清洗后 contexts + 本回合新消息
        sanitized_messages = [
            Msg(item["role"], item["content"]) for item in req.contexts
        ]
        run_context = make_run_context(
            [Msg("system", "persona")] + sanitized_messages + [
                Msg("user", "今天的新消息"),
                Msg("assistant", "本回合回复"),
            ]
        )

        restore(plugin, event, run_context)

        self.assertEqual(
            len(run_context.messages), 1 + len(original) + 2,
            "还原只能替换前缀，不能重复插入消息",
        )
        restored_contents = [
            m.content for m in run_context.messages[1:1 + len(original)]
        ]
        self.assertEqual(
            restored_contents, [item["content"] for item in original],
            "落盘历史必须是原文（会话数据库不被改写）",
        )
        self.assertIsNone(event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY))

    def test_restore_skips_when_message_layout_unknown(self):
        """布局校验失败时只跳过还原：不重复插入、不丢消息"""
        plugin = make_plugin(bot_name="小助手")
        event = make_event()
        req = make_request(self._contexts())
        plugin._sanitize_legacy_context_identity(event, req)

        run_context = make_run_context([Msg("assistant", "对不上的历史")])
        before = list(run_context.messages)

        restore(plugin, event, run_context)

        self.assertEqual(len(run_context.messages), len(before),
                         "校验失败时不得改动消息")
        self.assertIsNone(event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY))


if __name__ == "__main__":
    unittest.main()
