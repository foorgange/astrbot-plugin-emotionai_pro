# stream_filter.py
"""流式输出净化器（v2）

背景
----
AstrBot 流式模式下 `ResultDecorateStage` 会被整体跳过：

    # result_decorate/stage.py
    if result.result_content_type == ResultContentType.STREAMING_RESULT:
        return

因此依赖 `on_llm_response` / `on_decorating_result` 的后置过滤在流式
开启时不会执行，模型输出的控制标记（如 `[需要情感评估]`）会被直接
推送到聊天窗口。

本模块把过滤下沉到「流式生成器」层：包装 run_agent 产出的 MessageChain
序列，在文本交给平台适配器之前完成净化。

设计要点
--------
1. 标记可能被切分到相邻 chunk（如 `[需要` + `情感评估` + `]`），
   因此不能逐块做正则替换，必须维护滑动窗口，暂扣末尾的
   「疑似标记前缀」。
2. 暂扣窗口最大长度 = 最长标记长度 - 1。仅在末尾字符确实构成了
   某个标记的前缀时才暂扣，正常文本不会被延迟，流式节奏几乎无损。
3. 流结束时 flush 释放残余。此时残余已经过「非标记前缀」判定，
   属于应当正常发送的普通文本。
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# 需要在流式输出中屏蔽的控制标记
_DEFAULT_MARKERS: tuple[str, ...] = (
    "[需要情感评估]",
    "【需要情感评估】",
    "[NEED_ASSESSMENT]",
)


class StreamingMarkerFilter:
    """有状态的流式文本净化器。

    逐段喂入增量文本，返回「当前可安全发送」的部分。
    内部保留一个尾部窗口，仅在末尾构成控制标记前缀时才暂扣字符。

    用法::

        f = StreamingMarkerFilter()
        for chunk in stream:
            safe = f.feed(chunk)
            if safe:
                send(safe)
        tail = f.flush()      # 流结束，释放残余
        if tail:
            send(tail)
    """

    def __init__(self, markers: Optional[Iterable[str]] = None):
        self._markers: tuple[str, ...] = (
            tuple(markers) if markers is not None else _DEFAULT_MARKERS
        )
        if not self._markers:
            raise ValueError("markers 不能为空")
        self._pattern = re.compile("|".join(re.escape(m) for m in self._markers))
        self._max_marker_len = max(len(m) for m in self._markers)
        self._buffer = ""
        # 记录「本段是否发生过删除」，用于顺手吞掉标记残留的空白
        self._removed_marker = False

    # ---------- 内部工具 ----------

    def _held_suffix_len(self, text: str) -> int:
        """返回末尾必须暂扣的字符数（若它是某标记的真前缀）。"""
        best = 0
        upper = min(self._max_marker_len - 1, len(text))
        for marker in self._markers:
            limit = min(len(marker) - 1, len(text))
            for k in range(limit, best, -1):
                if text.endswith(marker[:k]):
                    best = k
                    break
        del upper
        return best

    def _strip_leading_ws_if_after_removal(self, text: str) -> str:
        """标记被删除后，紧随其后的空白应一并吞掉，避免留下空行。"""
        if self._removed_marker:
            stripped = text.lstrip(" \t\r\n\u3000")
            if stripped != text:
                self._removed_marker = bool(stripped)
                return stripped
            self._removed_marker = False
        return text

    # ---------- 对外接口 ----------

    def feed(self, text: str) -> str:
        """喂入一段增量文本，返回本次可安全发送的部分。"""
        if not text:
            return ""

        self._buffer += text
        out = ""

        # 1) 清除缓冲区中已完整出现的标记
        while True:
            m = self._pattern.search(self._buffer)
            if not m:
                break
            out += self._buffer[: m.start()]
            self._buffer = self._buffer[m.end():]
            self._removed_marker = True

        # 2) 标记后残留的空白一并吞掉
        out = self._strip_leading_ws_if_after_removal(out)

        # 3) 暂扣末尾可能的标记前缀，其余立即放行
        hold = self._held_suffix_len(self._buffer)
        if hold:
            cut = len(self._buffer) - hold
            out += self._buffer[:cut]
            self._buffer = self._buffer[cut:]
        else:
            out += self._buffer
            self._buffer = ""

        return out

    def flush(self) -> str:
        """流结束调用，释放残余内容。"""
        rest = self._pattern.sub("", self._buffer)
        self._buffer = ""
        self._removed_marker = False
        return rest.lstrip(" \t\r\n\u3000") if rest else rest


def sanitize_text(text: str, markers: Optional[Iterable[str]] = None) -> str:
    """一次性净化（非流式场景的便捷函数）。"""
    f = StreamingMarkerFilter(markers)
    return f.feed(text) + f.flush()
