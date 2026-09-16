# tests/test_background_emotion_update.py
"""情感分析转后台任务的回归测试（v4.0.12）

背景：情感分析（辅助 LLM 调用）实测单次 7~11s，原先在 on_llm_response 钩子里
`await` 执行（该钩子由 astr_agent_hooks.on_agent_done 通过 await 触发），
会拖慢回复收尾与后续消息处理。改为 asyncio.create_task 后台执行后必须保证：

  1) 钩子本身不再等待分析完成（非阻塞）—— 这是本次改动的核心收益
  2) 同一用户同时只有一个后台分析（get_user_state 返回缓存里的同一对象，
     并发会互相覆盖 force_update_counter 与数值更新）
  3) 后台任务异常被吞掉，不产生 unhandled task exception
  4) CancelledError 正常向上传播（供 terminate() 取消）
  5) 持有 task 强引用，完成后自动从登记表清理
  6) resp.completion_text 的改动（标记剥离、状态展示）仍发生在回复发出前
"""
import asyncio
import os
import re
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.config import PluginConfig  # noqa: E402
from emotionai_pro.main import EmotionAIProPlugin  # noqa: E402

MARKER = "[需要情感评估]"


class FakeState:
    """最小 EmotionalState 替身"""

    def __init__(self, show_status=False):
        self.force_update_counter = 0
        self.favor = 50
        self.intimacy = 50
        self.show_status = show_status
        self.descriptions = types.SimpleNamespace(attitude="中立", relationship="普通")
        self.reset_calls = 0

    def should_force_update(self, interval):
        return False

    def reset_force_update_counter(self):
        self.reset_calls += 1
        self.force_update_counter = 0


class FakeExpert:
    """情感分析专家替身。behavior: ok / none / raise / hang"""

    def __init__(self, behavior="ok"):
        self.behavior = behavior
        self.calls = 0
        self.last_umo = "unset"

    async def analyze_and_update_emotion(self, user_key, user_message, original_text, state, umo):
        self.calls += 1
        self.last_umo = umo
        if self.behavior == "hang":
            await asyncio.sleep(3600)
        if self.behavior == "raise":
            raise RuntimeError("分析炸了")
        if self.behavior == "none":
            return None
        return {"joy": 10, "source": "llm_analysis"}


class FakeUserManager:
    def __init__(self, state):
        self._state = state
        self.saved = 0

    async def get_user_state(self, user_key):
        return self._state

    async def update_user_state(self, user_key, state):
        self.saved += 1


class FakeMemory:
    def __init__(self):
        self.records = []

    async def add_interaction(self, *args, **kwargs):
        self.records.append((args, kwargs))


class FakeUpdateManager:
    def should_update_emotion(self, state, user_message, original_text):
        return (False, "", 0)


class Harness:
    """装配一个只保留被测逻辑的插件实例"""

    def __init__(self, state=None, behavior="ok"):
        self.state = state or FakeState()
        self.expert = FakeExpert(behavior)
        self.user_manager = FakeUserManager(self.state)
        self.memory = FakeMemory()
        self.applied = []
        self.mood_updates = []

        plugin = EmotionAIProPlugin.__new__(EmotionAIProPlugin)
        plugin.config = PluginConfig()
        plugin.need_assessment_pattern = re.compile(r"\[需要情感评估\]")
        plugin._emotion_update_tasks = {}
        plugin.update_manager = FakeUpdateManager()
        plugin.emotion_expert = self.expert
        plugin.user_manager = self.user_manager
        plugin.memory_system = self.memory

        # 隔离与本次改动无关的实现细节
        plugin._get_user_key = lambda ev: getattr(ev, "user_key", "user1")
        plugin._get_message_text = lambda ev: "你好喵"
        plugin._apply_expert_updates = lambda st, up: self.applied.append(up)
        plugin._calculate_emotional_significance = lambda up: 5
        plugin._update_global_mood = lambda updates: self.mood_updates.append(updates)
        plugin._format_emotional_state = lambda st: "【状态】"
        plugin._sanitize_ai_text = lambda t: t

        self.plugin = plugin

    def event(self, user_key="user1"):
        return types.SimpleNamespace(user_key=user_key, unified_msg_origin="g1")

    def resp(self, text="你好呀"):
        return types.SimpleNamespace(completion_text=text)

    async def drain(self):
        """等所有后台任务跑完，并让 done_callback 有机会执行"""
        tasks = list(self.plugin._emotion_update_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for _ in range(3):
            await asyncio.sleep(0)

    async def tick(self, n=1):
        """让出事件循环 n 次（create_task 后协程需一次调度才会真正开始执行）"""
        for _ in range(n):
            await asyncio.sleep(0)

    async def cleanup(self):
        for t in list(self.plugin._emotion_update_tasks.values()):
            t.cancel()
        tasks = list(self.plugin._emotion_update_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class TestNonBlocking(unittest.TestCase):
    def test_hook_returns_while_analysis_still_running(self):
        """核心：分析挂住不返回，钩子也必须立即返回（证明非阻塞）"""
        h = Harness(behavior="hang")

        async def go():
            resp = h.resp(MARKER + "你好呀")
            # 若钩子仍 await 分析，这里会超时失败
            await asyncio.wait_for(
                h.plugin.process_smart_update(h.event(), resp), timeout=1.0
            )
            self.assertEqual(len(h.plugin._emotion_update_tasks), 1)
            task = next(iter(h.plugin._emotion_update_tasks.values()))
            self.assertFalse(task.done(), "分析应仍在后台运行")
            await h.cleanup()

        asyncio.run(go())

    def test_marker_stripped_before_return(self):
        """标记剥离仍在回复发出前完成（原有行为不变）"""
        h = Harness(behavior="hang")

        async def go():
            resp = h.resp(MARKER + "你好呀")
            await asyncio.wait_for(
                h.plugin.process_smart_update(h.event(), resp), timeout=1.0
            )
            self.assertEqual(resp.completion_text, "你好呀")
            await h.cleanup()

        asyncio.run(go())

    def test_status_appended_before_return(self):
        """状态展示仍在回复发出前追加（用分析前的状态）"""
        h = Harness(state=FakeState(show_status=True), behavior="hang")

        async def go():
            resp = h.resp(MARKER + "你好呀")
            await asyncio.wait_for(
                h.plugin.process_smart_update(h.event(), resp), timeout=1.0
            )
            self.assertTrue(resp.completion_text.startswith("你好呀"))
            self.assertIn("【状态】", resp.completion_text)
            await h.cleanup()

        asyncio.run(go())

    def test_umo_passed_through_to_background(self):
        """umo 必须透传进后台任务（问题 #2b 的修复不能因改后台而丢失）"""
        h = Harness(behavior="ok")

        async def go():
            await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "a"))
            await h.drain()
            self.assertEqual(h.expert.last_umo, "g1")  # stub 事件默认 umo

        asyncio.run(go())


class TestConcurrencyGuard(unittest.TestCase):
    def test_same_user_second_call_skipped(self):
        """同一用户上一轮仍在跑 → 本轮跳过，不并发"""
        h = Harness(behavior="hang")

        async def go():
            await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "a"))
            await h.tick()  # 让第一轮后台分析真正启动
            await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "b"))
            self.assertEqual(h.expert.calls, 1, "第二次不应再发起分析")
            self.assertEqual(len(h.plugin._emotion_update_tasks), 1)
            await h.cleanup()

        asyncio.run(go())

    def test_different_users_run_concurrently(self):
        """不同用户互不阻塞"""
        h = Harness(behavior="hang")

        async def go():
            await h.plugin.process_smart_update(h.event("u1"), h.resp(MARKER + "a"))
            await h.plugin.process_smart_update(h.event("u2"), h.resp(MARKER + "b"))
            await h.tick()  # 让两个后台分析都启动
            self.assertEqual(h.expert.calls, 2)
            self.assertEqual(len(h.plugin._emotion_update_tasks), 2)
            await h.cleanup()

        asyncio.run(go())

    def test_respawn_after_previous_finished(self):
        """上一轮跑完后可以再次触发，且登记表已自动清理"""
        h = Harness(behavior="ok")

        async def go():
            await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "a"))
            await h.drain()
            self.assertEqual(h.plugin._emotion_update_tasks, {}, "完成后应自动出表")

            await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "b"))
            await h.drain()
            self.assertEqual(h.expert.calls, 2)

        asyncio.run(go())


class TestErrorHandling(unittest.TestCase):
    def test_exception_swallowed(self):
        """分析抛异常 → 后台任务正常结束，不产生 unhandled exception"""
        h = Harness(behavior="raise")

        async def go():
            await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "a"))
            task = next(iter(h.plugin._emotion_update_tasks.values()))
            await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(task.done())
            self.assertIsNone(task.exception(), "异常应已被内部捕获")
            self.assertFalse(task.cancelled())
            await h.drain()

        asyncio.run(go())

    def test_cancel_propagates(self):
        """CancelledError 必须向上传播（terminate 取消依赖它）"""
        h = Harness(behavior="hang")

        async def go():
            await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "a"))
            task = next(iter(h.plugin._emotion_update_tasks.values()))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(task.cancelled())

        asyncio.run(go())

    def test_no_spawn_when_not_needed(self):
        """不需要更新时完全不建任务"""
        h = Harness(behavior="hang")

        async def go():
            await h.plugin.process_smart_update(h.event(), h.resp("普通回复"))
            self.assertEqual(h.plugin._emotion_update_tasks, {})
            self.assertEqual(h.expert.calls, 0)

        asyncio.run(go())


class TestSuccessPath(unittest.TestCase):
    def test_applies_state_and_persists(self):
        """成功路径：应用更新、重置计数、写记忆、落盘"""
        h = Harness(behavior="ok")

        async def go():
            await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "a"))
            await h.drain()

            self.assertEqual(len(h.applied), 1, "应应用一次专家更新")
            self.assertEqual(h.state.reset_calls, 1, "应重置一次强制更新计数")
            self.assertEqual(len(h.memory.records), 1, "应写一条记忆")
            self.assertGreaterEqual(h.user_manager.saved, 2, "同步一次 + 后台一次落盘")
            # 专家更新也要叠加到全局心情演进
            self.assertIn({"joy": 10, "source": "llm_analysis"}, h.mood_updates)

        asyncio.run(go())

    def test_empty_result_does_not_apply(self):
        """分析返回空 → 不应用更新、不重置计数"""
        h = Harness(behavior="none")

        async def go():
            await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "a"))
            await h.drain()
            self.assertEqual(h.applied, [])
            self.assertEqual(h.state.reset_calls, 0)
            self.assertEqual(h.memory.records, [])

        asyncio.run(go())


class TestTerminate(unittest.TestCase):
    """插件关闭时必须收拾掉在跑的后台任务，且自身不能抛异常"""

    def test_terminate_cancels_pending_tasks(self):
        import emotionai_pro.main as main_mod

        h = Harness(behavior="hang")

        async def _anoop(*args, **kwargs):
            return None

        class _Closer:
            async def close(self):
                return None

        h.user_manager.close = _anoop
        h.plugin.cache = _Closer()
        h.plugin.global_mood_store = _Closer()
        h.plugin.memory_system._save_long_term_memory = _anoop
        h.plugin.smart_cleanup_task = None

        old_grace = main_mod._EMOTION_SHUTDOWN_GRACE
        main_mod._EMOTION_SHUTDOWN_GRACE = 0.05  # 缩短宽限期，避免测试真等 3 秒
        try:

            async def go():
                await h.plugin.process_smart_update(h.event(), h.resp(MARKER + "a"))
                await h.tick()
                self.assertEqual(len(h.plugin._emotion_update_tasks), 1)

                await h.plugin.terminate()  # 不应抛异常

                self.assertEqual(h.plugin._emotion_update_tasks, {})

            asyncio.run(go())
        finally:
            main_mod._EMOTION_SHUTDOWN_GRACE = old_grace

    def test_terminate_with_no_tasks(self):
        """没有在跑的任务时也要能正常关闭"""
        import emotionai_pro.main as main_mod

        h = Harness(behavior="ok")

        async def _anoop(*args, **kwargs):
            return None

        class _Closer:
            async def close(self):
                return None

        h.user_manager.close = _anoop
        h.plugin.cache = _Closer()
        h.plugin.global_mood_store = _Closer()
        h.plugin.memory_system._save_long_term_memory = _anoop
        h.plugin.smart_cleanup_task = None

        async def go():
            await h.plugin.terminate()
            self.assertEqual(h.plugin._emotion_update_tasks, {})

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
