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
    class _RecordingLogger:
        """记录型 logger 桩。

        为什么不是 `SimpleNamespace(info=lambda *a: None, ...)` 这种纯 no-op：
            插件原先大量用 `print()` 打日志，测试靠
            `contextlib.redirect_stdout` 断言"打了几次"。
            上架规则要求 print 必须改成 `from astrbot.api import logger`
            （内置 logging 与 print 都禁止），于是那些测试再也抓不到东西。
            给桩加一个 records 列表，测试可以直接断言"打了什么、打了几次"，
            同时 no-op 语义不变（不会往真实 stdout 写）。

        注意：`api.logger` 是所有插件模块共享的**同一个**对象，
        所以断言前要么先 `clear()`，要么用 before/after 切片比较。
        """

        def __init__(self, name="astrbot"):
            self.name = name
            self.records = []

        def _add(self, level, msg, *args, **kwargs):
            self.records.append((level, msg))

        def debug(self, msg, *a, **k):
            self._add("debug", msg)

        def info(self, msg, *a, **k):
            self._add("info", msg)

        def warning(self, msg, *a, **k):
            self._add("warning", msg)

        def error(self, msg, *a, **k):
            self._add("error", msg)

        def critical(self, msg, *a, **k):
            self._add("critical", msg)

        def exception(self, msg, *a, **k):
            self._add("exception", msg)

        # ---- 测试辅助 ----
        def clear(self):
            self.records.clear()

        def count(self, mark, level=None):
            """包含 mark 的记录条数（可按级别过滤）"""
            return sum(
                1 for lv, m in self.records
                if mark in str(m) and (level is None or lv == level)
            )

        def texts(self, level=None):
            return [str(m) for lv, m in self.records
                    if level is None or lv == level]

    api = types.ModuleType("astrbot.api")
    api.logger = _RecordingLogger()

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
        on_agent_done = staticmethod(_identity_decorator)
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
            self._extras = {}

        def get_sender_id(self):
            return self._sender_id

        def get_message_str(self):
            return self.message_str

        def get_platform_name(self):
            return "qq"

        def get_extra(self, key, default=None):
            return self._extras.get(key, default)

        def set_extra(self, key, value):
            self._extras[key] = value

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
            self.contexts = []
            self.prompt = ""
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

    # 忠实复刻框架侧的最小行为：checkpoint 段（role=_checkpoint）不产生
    # 独立消息，而是绑定到前一条消息的 _checkpoint_after 上；
    # _no_save 标记随消息保留。供上下文保鲜的还原逻辑测试使用。
    class Message:
        def __init__(self, role="user", content=""):
            self.role = role
            self.content = content
            self._no_save = False
            self._checkpoint_after = None

        @classmethod
        def model_validate(cls, data):
            msg = cls(role=data.get("role", "user"), content=data.get("content", ""))
            msg._no_save = bool(data.get("_no_save"))
            return msg

    def is_checkpoint_message(message):
        if isinstance(message, Message):
            return message.role == "_checkpoint"
        return isinstance(message, dict) and message.get("role") == "_checkpoint"

    def bind_checkpoint_messages(history):
        messages = []
        for item in history:
            if is_checkpoint_message(item):
                if messages:
                    messages[-1]._checkpoint_after = item.get("content")
                continue
            msg = Message.model_validate(item)
            messages.append(msg)
        return messages

    agent_mod.message = types.ModuleType("astrbot.core.agent.message")
    agent_mod.message.TextPart = TextPart
    agent_mod.message.Message = Message
    agent_mod.message.is_checkpoint_message = is_checkpoint_message
    agent_mod.message.bind_checkpoint_messages = bind_checkpoint_messages

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
