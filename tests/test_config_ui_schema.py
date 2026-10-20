# tests/test_config_ui_schema.py
"""v4.2.2：辅助 LLM 配置界面契约（从已配置模型下拉选择）

锁定 `_conf_schema.json` 的 UI 元数据，防止以后改动把界面退回「手动输入提供商/模型名」：

1. `secondary_llm_provider` 必须带 `_special: "select_provider"` —— AstrBot WebUI 会把它
   渲染成 ProviderSelector（与「回退对话模型列表」同一套选择器，列出全部已配置对话模型，
   含退避链里的模型）；该选择器 emit 的是 provider id（形如 `agnes/agnes-2.5-flash`），
   插件侧 `_find_target_provider` 本来就按 id/model 子串匹配，语义兼容。
2. `secondary_llm_model` 保留为高级覆盖项但 `invisible: true`，避免用户被迫手打模型名；
   旧配置里手填的值仍会被读取（`_target_model()` 逻辑未变）。
3. 仍是 `type: string`、`default: ""`，不改变配置校验与回退行为。
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "_conf_schema.json"


def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


class TestSecondaryLlmUiSchema(unittest.TestCase):
    """辅助 LLM 配置项必须由已配置模型选择器呈现"""

    def test_provider_uses_select_provider_special(self):
        schema = load_schema()
        node = schema["secondary_llm_provider"]
        self.assertEqual(node.get("_special"), "select_provider")
        self.assertEqual(node.get("type"), "string")
        self.assertEqual(node.get("default"), "")

    def test_model_override_is_hidden_advanced(self):
        schema = load_schema()
        node = schema["secondary_llm_model"]
        self.assertIs(node.get("invisible"), True)
        self.assertEqual(node.get("type"), "string")
        self.assertEqual(node.get("default"), "")

    def test_other_fields_do_not_get_special_renderer(self):
        """避免误给别的配置项加 _special（会改变整个配置页渲染）"""
        schema = load_schema()
        special = {
            key: node.get("_special")
            for key, node in schema.items()
            if isinstance(node, dict) and node.get("_special")
        }
        self.assertEqual(special, {"secondary_llm_provider": "select_provider"})

    def test_schema_keeps_crlf_and_no_bom(self):
        """仓库 JSON 约定：无 BOM、统一 CRLF（防止改 UI 时被编辑器混入 LF）"""
        raw = SCHEMA_PATH.read_bytes()
        self.assertNotEqual(raw[:3], b"\xef\xbb\xbf")
        self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"))

    def test_python_side_still_accepts_the_fields(self):
        """UI 元数据变化不能影响 Python 侧配置模型/校验器"""
        from emotionai_pro.config import PluginConfig
        from emotionai_pro.schema_validator import ConfigValidator

        self.assertIn("secondary_llm_provider", PluginConfig.model_fields)
        self.assertIn("secondary_llm_model", PluginConfig.model_fields)
        props = ConfigValidator.CONFIG_SCHEMA["properties"]
        self.assertIn("secondary_llm_provider", props)
        self.assertIn("secondary_llm_model", props)


if __name__ == "__main__":
    unittest.main()
