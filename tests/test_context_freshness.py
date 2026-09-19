# tests/test_context_freshness.py
"""v4.1.0 会话边界上下文保鲜 回归测试

需求（用户原话整理）：
    bot 聊天时会「忽然接上很久之前的话题」，而那个话题用户可能已经不聊了。

根因：框架每轮携带整个对话历史（max_context_length=50），群聊场景下
几天前的话题仍在上下文窗口内。方案：距上次活跃超过
`session_gap_minutes` 时，本轮请求清空 provider 可见历史（req.contexts），
让 bot 从零开始接话；同时通过 on_agent_done 钩子把历史还原回
run_context.messages，保证框架落盘的会话记录完整、不丢数据。

本文件锁死：
    ① 间隔超阈值才裁剪（边界值不裁）；
    ② 时间信号取会话 updated_at 与插件存档 last_interaction_time 中较新者；
    ③ session_gap_minutes <= 0 关闭保鲜；配置非法时安全跳过；
    ④ 裁剪后 on_agent_done 必须把旧历史完整还原（含 checkpoint 段）；
    ⑤ 新配置字段 schema / _conf_schema.json / pydantic 模型三处同步。
"""
import sys
import os
import json
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.main import EmotionAIProPlugin  # noqa: E402


def _make_plugin(**cfg):
    """造一个够 _apply_context_freshness / _restore_full_context 使用的实例"""
    from emotionai_pro.config import PluginConfig
    from emotionai_pro.managers import EmotionAnalyzer

    plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
    plugin.config = PluginConfig(**cfg)
    plugin.analyzer = EmotionAnalyzer()
    return plugin


def _make_event():
    from astrbot.api.event import AstrMessageEvent
    return AstrMessageEvent()


def _make_request(contexts, updated_at=None):
    """造 ProviderRequest；conversation.updated_at 为 None 表示无时间信号"""
    from astrbot.api.provider import ProviderRequest
    import types

    req = ProviderRequest()
    req.contexts = list(contexts)
    req.conversation = types.SimpleNamespace(
        persona_id=None, updated_at=updated_at
    )
    return req


def _make_state(last_interaction_time=None):
    from emotionai_pro.models import EnhancedEmotionalState
    import types

    state = EnhancedEmotionalState(user_key="u", favor=0, intimacy=0)
    if last_interaction_time is not None:
        state.stats = types.SimpleNamespace(
            last_interaction_time=last_interaction_time
        )
    else:
        state.stats = types.SimpleNamespace(last_interaction_time=None)
    return state


def _make_run_context(messages):
    import types
    return types.SimpleNamespace(messages=messages)


def _restore(plugin, event, run_context, response=None):
    """_restore_full_context 是 async 钩子，包一层同步执行"""
    import asyncio

    async def go():
        await plugin._restore_full_context(event, run_context, response)

    asyncio.run(go())


def _Msg(role, content=""):
    from astrbot.core.agent.message import Message
    return Message(role=role, content=content)


class TestTrimDecision(unittest.TestCase):
    """裁剪判定：间隔、时间信号、开关"""

    def test_trim_when_gap_exceeds_threshold(self):
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        # 3 小时前的会话
        req = _make_request(
            [{"role": "user", "content": "旧话题"}],
            updated_at=time.time() - 3 * 3600,
        )
        state = _make_state()

        plugin._apply_context_freshness(event, req, state)

        self.assertEqual(req.contexts, [], "超过阈值必须清空本轮上下文")
        snap = event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY)
        self.assertEqual(snap, [{"role": "user", "content": "旧话题"}],
                         "快照必须保留原始上下文供还原")

    def test_no_trim_when_recent(self):
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        req = _make_request(
            [{"role": "user", "content": "刚才"}],
            updated_at=time.time() - 60,
        )
        state = _make_state()

        plugin._apply_context_freshness(event, req, state)

        self.assertEqual(len(req.contexts), 1)
        self.assertIsNone(event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY))

    def test_boundary_exactly_threshold_not_trimmed(self):
        """恰好等于阈值不裁（<= 返回），只裁严格超过的"""
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        req = _make_request(
            [{"role": "user", "content": "x"}],
            updated_at=time.time() - 60 * 60 + 2,
        )
        plugin._apply_context_freshness(event, req, _make_state())
        self.assertEqual(len(req.contexts), 1)

    def test_just_over_threshold_trimmed(self):
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        req = _make_request(
            [{"role": "user", "content": "x"}],
            updated_at=time.time() - 60 * 60 - 5,
        )
        plugin._apply_context_freshness(event, req, _make_state())
        self.assertEqual(req.contexts, [])

    def test_disabled_when_zero(self):
        plugin = _make_plugin(session_gap_minutes=0)
        event = _make_event()
        req = _make_request(
            [{"role": "user", "content": "x"}],
            updated_at=time.time() - 48 * 3600,
        )
        plugin._apply_context_freshness(event, req, _make_state())
        self.assertEqual(len(req.contexts), 1, "0 = 关闭保鲜")
        self.assertIsNone(event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY))

    def test_uses_newer_of_two_signals(self):
        """会话旧但插件存档新 → 不裁（少裁不多裁的安全方向）"""
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        req = _make_request(
            [{"role": "user", "content": "x"}],
            updated_at=time.time() - 5 * 3600,
        )
        state = _make_state(last_interaction_time=time.time() - 30)
        plugin._apply_context_freshness(event, req, state)
        self.assertEqual(len(req.contexts), 1)

    def test_stats_older_than_conversation_uses_conversation(self):
        """插件存档旧、会话新 → 同样不裁"""
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        req = _make_request(
            [{"role": "user", "content": "x"}],
            updated_at=time.time() - 30,
        )
        state = _make_state(last_interaction_time=time.time() - 5 * 3600)
        plugin._apply_context_freshness(event, req, state)
        self.assertEqual(len(req.contexts), 1)

    def test_no_time_signal_no_trim(self):
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        req = _make_request([{"role": "user", "content": "x"}], updated_at=None)
        plugin._apply_context_freshness(event, req, _make_state())
        self.assertEqual(len(req.contexts), 1, "无时间信号时保持框架默认行为")

    def test_snapshot_taken_once(self):
        """同一事件二次触发请求时不覆盖已有快照"""
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        req = _make_request(
            [{"role": "user", "content": "第一轮"}],
            updated_at=time.time() - 3 * 3600,
        )
        plugin._apply_context_freshness(event, req, _make_state())
        first_snap = event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY)

        # 第二次：全新 request（可能已被其它插件改过），快照已存在
        req2 = _make_request(
            [{"role": "user", "content": "别的"}],
            updated_at=time.time() - 3 * 3600,
        )
        plugin._apply_context_freshness(event, req2, _make_state())

        self.assertEqual(event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY), first_snap)
        self.assertEqual(len(req2.contexts), 1, "快照已存在时不再裁剪")

    def test_bad_config_value_ignored(self):
        plugin = _make_plugin()
        plugin.config.session_gap_minutes = "abc"
        event = _make_event()
        req = _make_request(
            [{"role": "user", "content": "x"}],
            updated_at=time.time() - 99 * 3600,
        )
        plugin._apply_context_freshness(event, req, _make_state())
        self.assertEqual(len(req.contexts), 1, "配置类型非法时安全跳过")

    def test_bool_config_value_ignored(self):
        plugin = _make_plugin()
        plugin.config.session_gap_minutes = True
        event = _make_event()
        req = _make_request(
            [{"role": "user", "content": "x"}],
            updated_at=time.time() - 99 * 3600,
        )
        plugin._apply_context_freshness(event, req, _make_state())
        self.assertEqual(len(req.contexts), 1, "bool 不是合法间隔值")

    def test_empty_contexts_no_snapshot(self):
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        req = _make_request([], updated_at=time.time() - 99 * 3600)
        plugin._apply_context_freshness(event, req, _make_state())
        self.assertIsNone(event.get_extra(plugin._FRESHNESS_SNAPSHOT_KEY),
                          "没有旧历史可裁时不留快照")

    def test_iso_string_timestamp_supported(self):
        """updated_at 在库里可能是 ISO 字符串（历史数据兼容）"""
        from datetime import datetime
        plugin = _make_plugin(session_gap_minutes=60)
        event = _make_event()
        stale = datetime.fromtimestamp(time.time() - 3 * 3600).isoformat()
        req = _make_request([{"role": "user", "content": "x"}], updated_at=stale)
        plugin._apply_context_freshness(event, req, _make_state())
        self.assertEqual(req.contexts, [], "ISO 字符串时间戳也要能识别")


class TestRestoreAfterDone(unittest.TestCase):
    """还原：on_agent_done 把旧历史补回 run_context.messages"""

    KEY = EmotionAIProPlugin._FRESHNESS_SNAPSHOT_KEY

    def test_restore_reinserts_before_new_messages(self):
        plugin = _make_plugin()
        event = _make_event()
        old_history = [
            {"role": "user", "content": "三天前的话题"},
            {"role": "assistant", "content": "当时的回复"},
        ]
        event.set_extra(self.KEY, old_history)
        # reset() 的组装形态：[system?] + 裁剪后 contexts + 本回合新消息
        run_context = _make_run_context([
            _Msg("system", "persona"),
            _Msg("user", "今天的新消息"),
            _Msg("assistant", "本回合回复"),
        ])

        _restore(plugin, event, run_context)

        roles = [(m.role, m.content) for m in run_context.messages]
        self.assertEqual(roles, [
            ("system", "persona"),
            ("user", "三天前的话题"),
            ("assistant", "当时的回复"),
            ("user", "今天的新消息"),
            ("assistant", "本回合回复"),
        ], "旧历史必须插在本回合新消息之前，保持原始顺序")
        self.assertIsNone(event.get_extra(self.KEY), "快照一次性，用完即清")

    def test_restore_without_system_message(self):
        plugin = _make_plugin()
        event = _make_event()
        event.set_extra(self.KEY, [{"role": "user", "content": "旧的"}])
        run_context = _make_run_context([_Msg("user", "新的")])

        _restore(plugin, event, run_context)

        self.assertEqual([m.content for m in run_context.messages], ["旧的", "新的"])

    def test_restore_noop_without_snapshot(self):
        plugin = _make_plugin()
        event = _make_event()
        run_context = _make_run_context([_Msg("user", "新的")])

        _restore(plugin, event, run_context)

        self.assertEqual(len(run_context.messages), 1)

    def test_restore_binds_checkpoint(self):
        """checkpoint 段不产生独立消息，绑定到前一条消息"""
        plugin = _make_plugin()
        event = _make_event()
        event.set_extra(self.KEY, [
            {"role": "user", "content": "老消息"},
            {"role": "_checkpoint", "content": {"id": "cp1"}},
            {"role": "assistant", "content": "老回复"},
        ])
        run_context = _make_run_context([_Msg("user", "新的")])

        _restore(plugin, event, run_context)

        msgs = run_context.messages
        self.assertEqual(len(msgs), 3, "checkpoint 段不应成为独立消息")
        self.assertEqual(msgs[0]._checkpoint_after, {"id": "cp1"},
                         "checkpoint 必须绑定回前一条消息")

    def test_restore_tolerates_missing_messages_attr(self):
        plugin = _make_plugin()
        event = _make_event()
        event.set_extra(self.KEY, [{"role": "user", "content": "老的"}])

        # 不应抛异常（异常会被吞成 warning，但这里直接验证不炸）
        _restore(plugin, event, object())
        self.assertIsNone(event.get_extra(self.KEY), "异常路径也要清快照")

    def test_restore_ignores_empty_snapshot(self):
        plugin = _make_plugin()
        event = _make_event()
        event.set_extra(self.KEY, [])
        run_context = _make_run_context([_Msg("user", "新的")])

        _restore(plugin, event, run_context)

        self.assertEqual(len(run_context.messages), 1)


class TestFreshnessConfigSync(unittest.TestCase):
    """session_gap_minutes 必须三处同步，且关闭分支可用"""

    def test_pydantic_field(self):
        from emotionai_pro.config import PluginConfig
        cfg = PluginConfig()
        self.assertEqual(cfg.session_gap_minutes, 60)
        # 边界由 pydantic 把守：0（关闭）合法、-1 非法
        self.assertEqual(PluginConfig(session_gap_minutes=0).session_gap_minutes, 0)
        with self.assertRaises(Exception):
            PluginConfig(session_gap_minutes=-1)

    def test_schema_validator_knows_field(self):
        import inspect
        from emotionai_pro.schema_validator import ConfigValidator
        props = ConfigValidator.CONFIG_SCHEMA["properties"]
        self.assertEqual(props["session_gap_minutes"],
                         {"type": "integer", "minimum": 0, "maximum": 1440})
        # create_default_config 会写文件，改用源码断言默认值存在
        src = inspect.getsource(ConfigValidator.create_default_config)
        self.assertIn('"session_gap_minutes": 60', src,
                      "默认配置缺少 session_gap_minutes")

    def test_conf_schema_node(self):
        root = Path(__file__).resolve().parent.parent
        schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
        node = schema["session_gap_minutes"]
        self.assertEqual(node["default"], 60)
        self.assertIn("slider", node)
        self.assertLessEqual(node["slider"]["min"], 0)
        self.assertGreaterEqual(node["slider"]["max"], 60)

    def test_base_mapping_whitelist(self):
        """main.py 的 base_mapping 白名单必须认这个字段，否则永远是默认值"""
        src = Path(__file__).resolve().parent.parent / "main.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn('"session_gap_minutes": "session_gap_minutes"', text)


if __name__ == "__main__":
    unittest.main()
