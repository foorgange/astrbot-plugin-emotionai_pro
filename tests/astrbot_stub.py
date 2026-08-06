# tests/astrbot_stub.py
"""AstrBot 模块 stub，用于离线单测 main.py / command_handlers.py"""
import sys
import types
from pathlib import Path


def _install():
    """向 sys.modules 注入 astrbot 相关 stub 模块"""
    if "astrbot" in sys.modules:
        return

    # ---- astrbot.api ----
    api = types.ModuleType("astrbot.api")
    api.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )

    class AstrBotConfig(dict):
        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)

        async def save_config_async(self, *args, **kwargs):
            pass

        async def save_config(self, *args, **kwargs):
            pass

    api.AstrBotConfig = AstrBotConfig

    # ---- astrbot.api.event ----
    event_mod = types.ModuleType("astrbot.api.event")

    def _identity_decorator(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap

    class filter:
        command = staticmethod(_identity_decorator)
        on_llm_request = staticmethod(_identity_decorator)
        on_llm_response = staticmethod(_identity_decorator)
        on_decorating_result = staticmethod(_identity_decorator)
        command_group = staticmethod(_identity_decorator)

    event_mod.filter = filter

    class AstrMessageEvent:
        """最小事件 stub（供 handler 方法签名与内部使用）"""
        def __init__(self, role="user", sender_id="123", message_str="", unified_msg_origin="g1"):
            self.role = role
            self._sender_id = sender_id
            self.message_str = message_str
            self.unified_msg_origin = unified_msg_origin
            self.message_obj = None
            self.persona_id = None
            self.stop_called = False

        def get_sender_id(self):
            return self._sender_id

        def get_message_str(self):
            return self.message_str

        def get_platform_name(self):
            return "qq"

        def plain_result(self, text):
            return text

        def stop_event(self):
            self.stop_called = True

    event_mod.AstrMessageEvent = AstrMessageEvent

    # ---- astrbot.api.star ----
    star_mod = types.ModuleType("astrbot.api.star")

    class Star:
        def __init__(self, context=None):
            self.context = context

    class StarTools:
        @classmethod
        def get_data_dir(cls) -> Path:
            return Path("/tmp/astrbot_stub_data")

    def register(*args, **kwargs):
        def wrap(cls):
            return cls
        return wrap

    star_mod.Star = Star
    star_mod.StarTools = StarTools
    star_mod.register = register
    star_mod.Context = object

    # ---- astrbot.api.provider ----
    provider_mod = types.ModuleType("astrbot.api.provider")

    class LLMResponse:
        def __init__(self, completion_text=""):
            self.completion_text = completion_text

    class ProviderRequest:
        def __init__(self):
            self.extra_user_content_parts = []
            self.system_prompt = ""
            self.conversation = types.SimpleNamespace(persona_id=None)
            self.llm_response = None

    provider_mod.LLMResponse = LLMResponse
    provider_mod.ProviderRequest = ProviderRequest

    # ---- astrbot.core.agent.message ----
    core_mod = types.ModuleType("astrbot.core")
    agent_mod = types.ModuleType("astrbot.core.agent")

    class TextPart:
        def __init__(self, text=""):
            self.text = text

    agent_mod.message = types.ModuleType("astrbot.core.agent.message")
    agent_mod.message.TextPart = TextPart

    # ---- 组装 ----
    star_mod.message = agent_mod.message

    api.event = event_mod
    api.star = star_mod
    api.provider = provider_mod

    sys.modules["astrbot"] = types.ModuleType("astrbot")
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.star"] = star_mod
    sys.modules["astrbot.api.provider"] = provider_mod
    sys.modules["astrbot.core"] = core_mod
    sys.modules["astrbot.core.agent"] = agent_mod
    sys.modules["astrbot.core.agent.message"] = agent_mod.message


_install()
