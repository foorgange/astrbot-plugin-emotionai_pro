# tests/test_admin_config_resilience.py
"""管理员权限失效根因回归测试（v4.0.18）

背景（用户实际报障）：
    在插件配置界面把 QQ 3418451176 填进「管理员QQ号列表」并保存，
    但使用 /设置好感、/查看好感 等管理员命令时始终提示「权限不足」。

根因链：
    1. 配置界面（_conf_schema.json）对这些数值项**没有任何取值约束**，
       用户完全可以填出 intimacy_min=-100、change_min=3 这类组合；
    2. 而 config.py 的 PluginConfig 把它们写成 ge=0 / le=0，
       pydantic 直接抛 ValidationError；
    3. main.py::_load_and_validate_config 的兜底是 `return PluginConfig()`，
       **整份配置被默认值覆盖**（默认 admin_qq_list=[]）；
    4. 于是 _is_admin 永远为 False —— 用户看到的就是「权限不足」。

本文件锁死三层防线，任何一层回退都会让测试失败：
    A. PluginConfig 必须接受这些"看起来不常规但合法"的取值；
    B. 即使出现真·非法字段，也只能回退**那一个字段**，
       admin_qq_list 等其余配置必须保留；
    C. 权限不足的回复必须带可自助排查的信息（不能是一句干巴巴的提示）。
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from pydantic import ValidationError  # noqa: E402

from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.main import EmotionAIProPlugin  # noqa: E402
from emotionai_pro.command_handlers import (  # noqa: E402
    AdminCommandHandler, UserCommandHandler, DebugCommandHandler,
)
from astrbot.api.event import AstrMessageEvent  # noqa: E402


# 服务器上真实的配置值（用户实际填写的）
SERVER_LIKE_RAW = {
    "session_based": False,
    "favour_min": -100,
    "favour_max": 200,
    "intimacy_min": -100,      # 旧版 ge=0 → 报错
    "intimacy_max": 200,
    "change_min": 3,           # 旧版 le=0 → 报错
    "change_max": 2,
    "admin_qq_list": ["3418451176"],
    "plugin_priority": 100000,
    "enable_attitude_system": True,
    "enable_ai_text_generation": True,
    "global_privacy_level": 1,
    "enable_smart_update": True,
    "force_update_interval": 5,
    "emotional_significance_threshold": 5,
    "enable_secondary_llm": True,
    "secondary_llm_provider": "",
    "secondary_llm_model": "",
    "emotion_llm_time_budget": 45.0,
    "emotion_llm_max_providers": 3,
    "bot_name": "永雏塔菲",
}

TARGET_QQ = "3418451176"


def _plugin_stub():
    """不需要完整 __init__ 的插件 stub，仅用于调用配置加载方法"""
    return type("P", (), {"_raw_config": None})()


class TestServerConfigAccepted(unittest.TestCase):
    """A. 服务器上那份配置必须能被 PluginConfig 接受"""

    def test_real_server_config_constructs(self):
        cfg = PluginConfig(**SERVER_LIKE_RAW)
        self.assertEqual(cfg.admin_qq_list, [TARGET_QQ],
                         "服务器配置被拒绝或管理员列表丢失")

    def test_per_field_no_validation_error(self):
        """逐字段校验，定位到底是哪个字段越界（回归时能一眼看出）"""
        offenders = []
        for key, value in SERVER_LIKE_RAW.items():
            try:
                PluginConfig(**{key: value})
            except ValidationError as e:
                offenders.append((key, value, e.errors()[0].get("type")))
        self.assertEqual(offenders, [],
                         f"以下字段被 pydantic 判为非法，会导致整份配置失效: {offenders}")

    def test_bounds_are_magnitude_only(self):
        """区间约束只做量级限制，不再用符号猜用户意图"""
        cfg = PluginConfig(intimacy_min=-100, change_min=3, change_max=2,
                           favour_min=-100, favour_max=200)
        self.assertEqual(cfg.intimacy_min, -100)
        self.assertEqual(cfg.change_min, 3)
        self.assertEqual(cfg.change_max, 2)
        # 仍然要拦住明显不合理的量级
        with self.assertRaises(ValidationError):
            PluginConfig(plugin_priority=99_999_999)
        with self.assertRaises(ValidationError):
            PluginConfig(favour_max=10_000_000)


class TestTolerantConfigLoad(unittest.TestCase):
    """B. 出现非法字段时只能回退该字段，其余配置必须保留"""

    def _load(self, raw_dict):
        return EmotionAIProPlugin._load_and_validate_config(
            _plugin_stub(), dict(raw_dict)
        )

    def test_valid_server_config_keeps_admin_list(self):
        cfg = self._load(SERVER_LIKE_RAW)
        self.assertIn(TARGET_QQ, cfg.admin_qq_list)
        self.assertEqual(cfg.intimacy_min, -100)
        self.assertEqual(cfg.bot_name, "永雏塔菲")

    def test_bad_field_falls_back_alone(self):
        """一个字段真非法时，管理员列表不能被连坐清空"""
        raw = dict(SERVER_LIKE_RAW)
        raw["plugin_priority"] = 99_999_999      # 真·越界
        cfg = self._load(raw)
        self.assertEqual(cfg.plugin_priority, 100000, "非法字段应回退为默认值")
        self.assertIn(TARGET_QQ, cfg.admin_qq_list,
                      "非法字段不应导致整份配置被丢弃（这正是权限不足的根因）")
        self.assertEqual(cfg.bot_name, "永雏塔菲")

    def test_unknown_key_is_ignored(self):
        """配置里多出插件不认识的键不应触发校验失败"""
        raw = dict(SERVER_LIKE_RAW)
        raw["some_future_option"] = 123
        cfg = self._load(raw)
        self.assertIn(TARGET_QQ, cfg.admin_qq_list)

    def test_all_bad_returns_defaults_without_crash(self):
        """极端情况：多个字段同时非法，仍要返回可用配置（不抛异常）"""
        raw = dict(SERVER_LIKE_RAW)
        raw["plugin_priority"] = 99_999_999
        raw["favour_max"] = 10_000_000
        cfg = self._load(raw)
        self.assertIsInstance(cfg, PluginConfig)
        self.assertIn(TARGET_QQ, cfg.admin_qq_list)
        self.assertEqual(cfg.plugin_priority, 100000)


class TestAdminGateWorks(unittest.TestCase):
    """端到端：管理员列表生效后，_is_admin 必须放行"""

    def _handler(self, admin_list):
        plugin = type("P", (), {
            "config": PluginConfig(admin_qq_list=admin_list),
        })()
        handler = AdminCommandHandler.__new__(AdminCommandHandler)
        handler.plugin = plugin
        handler.config = plugin.config
        return handler

    def test_configured_qq_is_admin(self):
        handler = self._handler([TARGET_QQ])
        event = AstrMessageEvent(role="user", sender_id=TARGET_QQ)
        self.assertTrue(handler._is_admin(event),
                        "配置里的管理员 QQ 必须能通过 _is_admin")

    def test_other_qq_is_not_admin(self):
        handler = self._handler([TARGET_QQ])
        event = AstrMessageEvent(role="user", sender_id="10000")
        self.assertFalse(handler._is_admin(event))

    def test_event_role_admin_still_works(self):
        handler = self._handler([])
        event = AstrMessageEvent(role="admin", sender_id="10000")
        self.assertTrue(handler._is_admin(event))


class TestDeniedMessageIsDiagnosable(unittest.TestCase):
    """C. 权限不足回复必须带可自助排查的信息"""

    def _handler(self, admin_list):
        plugin = type("P", (), {
            "config": PluginConfig(admin_qq_list=admin_list),
        })()
        handler = AdminCommandHandler.__new__(AdminCommandHandler)
        handler.plugin = plugin
        handler.config = plugin.config
        return handler

    def test_denied_message_shows_sender_and_list(self):
        handler = self._handler([TARGET_QQ])
        event = AstrMessageEvent(role="user", sender_id="10000")
        msg = handler._admin_denied_message(event)
        self.assertIn("10000", msg, "应显示当前发送者 ID")
        self.assertIn(TARGET_QQ, msg, "应显示当前生效的管理员列表")

    def test_denied_message_handles_empty_list(self):
        """管理员列表为空 = 配置大概率回退了，提示要指向这一点"""
        handler = self._handler([])
        event = AstrMessageEvent(role="user", sender_id="10000")
        msg = handler._admin_denied_message(event)
        self.assertIn("为空", msg)
        self.assertIn("默认配置", msg)

    def test_all_admin_handlers_use_diagnosable_message(self):
        """所有管理员拦截点都必须用新提示（不能有漏网的干巴巴文案）"""
        import inspect
        import emotionai_pro.command_handlers as ch_mod
        src = inspect.getsource(ch_mod)
        self.assertNotIn('plain_result("【错误】需要管理员权限")', src,
                         "仍存在未升级的权限不足提示")
        self.assertIn("_admin_denied_message", src)


class TestAllHandlersHaveDeniedHelper(unittest.TestCase):
    """DebugCommandHandler / UserCommandHandler 也能用上该提示（继承自基类）"""

    def test_inherited(self):
        for cls in (UserCommandHandler, AdminCommandHandler, DebugCommandHandler):
            self.assertTrue(hasattr(cls, "_admin_denied_message"),
                            f"{cls.__name__} 缺少 _admin_denied_message")


if __name__ == "__main__":
    unittest.main(verbosity=2)
