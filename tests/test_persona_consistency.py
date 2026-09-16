# tests/test_persona_consistency.py
"""人设一致性回归测试：注入的增强上下文不得出现「AI 身份断言」

背景
----
本插件是通用件，而相当多的人设（例如「永雏塔菲」）在 system_prompt 里明确写了
「你不是 AI / 不要承认自己是 AI」。此前 `_build_enhanced_context` 的开头是：

    你是一个具有丰富情感的AI助手，请根据以下情感状态调整你的回应风格。

这段文本被注入到**用户消息之后**（`extra_user_content_parts`），位置比 system_prompt
更靠后、优先级更高，模型会因此把自己当成「通用 AI 助手」，导致人格漂移——
塔菲开始用助手的口吻说话，而不是角色本身。

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
    return plugin


def build(bot_name=None, **kwargs) -> str:
    plugin = make_plugin(bot_name=bot_name, **kwargs)
    state = EnhancedEmotionalState(user_key="test:user")
    return plugin._build_enhanced_context(state)


class TestNoAiIdentityAssertion(unittest.TestCase):
    """核心：注入文本里不能有任何「你是 AI」类断言"""

    def setUp(self):
        self.text = build(bot_name="塔菲")

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
            bot_name="塔菲",
            relationship_context="与用户的ai伙伴关系持续升温",
        )
        self.assertNotIn("ai伙伴", text)
        self.assertIn("塔菲伙伴", text)
        self.assertIsNone(_AI_STANDALONE_RE.search(text))

    def test_tone_instruction_ai_replaced(self):
        text = build(bot_name="塔菲", tone="像 AI 助手那样礼貌回复")
        self.assertNotIn("AI 助手", text)
        self.assertIn("塔菲 助手", text)
        self.assertIsNone(_AI_STANDALONE_RE.search(text))

    def test_no_bot_name_is_safe_noop(self):
        """bot_name 未解析时不报错，且不引入新的 AI 字样"""
        text = build(bot_name=None, relationship_context=CLEAN_RELATIONSHIP)
        self.assertIn("【长期关系发展轨迹】", text)
        self.assertNotIn(OLD_ASSERTION, text)
        self.assertIsNone(_AI_STANDALONE_RE.search(text))

    def test_memory_failure_falls_back(self):
        """记忆系统抛异常时走兜底文案，不影响注入结构"""
        plugin = make_plugin(bot_name="塔菲")
        plugin.memory_system.get_relationship_context = MagicMock(
            side_effect=RuntimeError("boom")
        )
        state = EnhancedEmotionalState(user_key="test:user")
        text = plugin._build_enhanced_context(state)
        self.assertIn("暂无长期互动记录", text)
        self.assertNotIn(OLD_ASSERTION, text)


if __name__ == "__main__":
    unittest.main()
