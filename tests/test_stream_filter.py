# tests/test_stream_filter.py
"""流式输出净化：控制标记的跨 chunk 过滤与内容完整性。

覆盖两类风险：
1. 标记被切分到相邻 chunk（如 `[需要` + `情感评估` + `]`）时必须仍能拦截；
2. 正常文本不得被延迟、截断或误删。
"""
import sys
import os
import random
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401
import tests.bootstrap  # noqa: F401

from emotionai_pro.stream_filter import (  # noqa: E402
    StreamingMarkerFilter,
    sanitize_text,
)

MARKER = "[需要情感评估]"


def _drain(chunks, markers=None):
    """按 chunk 序列喂入，返回 (逐次输出列表, 最终文本)。"""
    f = StreamingMarkerFilter(markers)
    outs = []
    for c in chunks:
        outs.append(f.feed(c))
    outs.append(f.flush())
    return outs, "".join(outs)


class TestMarkerRemoval(unittest.TestCase):
    """标记必须被移除，无论它落在哪个位置、是否被切碎。"""

    def test_marker_whole_in_one_chunk(self):
        outs, final = _drain(["今天天气不错呀", MARKER])
        self.assertNotIn("情感评估", final)
        self.assertEqual(final, "今天天气不错呀")

    def test_marker_split_into_three_chunks(self):
        _, final = _drain(["你好呀", "[需要", "情感评估", "]"])
        self.assertNotIn("情感评估", final)
        self.assertEqual(final, "你好呀")

    def test_marker_split_character_by_character(self):
        chunks = ["你好"] + list(MARKER)
        _, final = _drain(chunks)
        self.assertNotIn("情感评估", final)
        self.assertEqual(final, "你好")

    def test_marker_at_start(self):
        _, final = _drain([MARKER + "开场白"])
        self.assertEqual(final, "开场白")

    def test_marker_in_middle(self):
        _, final = _drain(["中间" + MARKER + "夹心"])
        self.assertEqual(final, "中间夹心")

    def test_marker_repeated(self):
        _, final = _drain(["这里有" + MARKER + "和" + MARKER + "两个"])
        self.assertEqual(final, "这里有和两个")

    def test_fullwidth_variant(self):
        _, final = _drain(["用【需要情感评估】变体"])
        self.assertNotIn("情感评估", final)

    def test_marker_followed_by_text(self):
        _, final = _drain(["聊天内容[需要", "情感评估]结束"])
        self.assertEqual(final, "聊天内容结束")


class TestContentIntegrity(unittest.TestCase):
    """正常文本不得被影响。"""

    def test_normal_chunks_pass_through_untouched(self):
        f = StreamingMarkerFilter()
        texts = ["你好啊", "今天过得", "怎么样？"]
        outs = [f.feed(t) for t in texts]
        self.assertEqual(outs, texts, "正常文本不应被延迟或改写")
        self.assertEqual(f.flush(), "")

    def test_plain_brackets_not_removed(self):
        self.assertEqual(sanitize_text("价格是[100]元"), "价格是[100]元")

    def test_unclosed_marker_released_on_flush(self):
        f = StreamingMarkerFilter()
        f.feed("结尾是半截[需要")
        self.assertEqual(f.flush(), "[需要", "未闭合前缀应在 flush 时释放")

    def test_long_stream_lossless(self):
        random.seed(42)
        base = "塔菲今天心情很好，想和你聊聊最近的事情。要不要一起出去玩呢？" * 20
        chunks = []
        i = 0
        while i < len(base):
            n = random.randint(1, 7)
            chunks.append(base[i:i + n])
            i += n
        _, final = _drain(chunks)
        self.assertEqual(final, base, "长文本流式拼接必须无损")

    def test_long_stream_with_marker_lossless(self):
        random.seed(7)
        payload = "开头很正常。" + "聊天内容" * 30 + MARKER + "结尾也要正常"
        chunks = []
        i = 0
        while i < len(payload):
            n = random.randint(1, 5)
            chunks.append(payload[i:i + n])
            i += n
        _, final = _drain(chunks)
        self.assertEqual(final, payload.replace(MARKER, ""))
        self.assertNotIn("情感评估", final)

    def test_custom_markers(self):
        _, final = _drain(["内容[[CTRL]]尾部"], markers=["[[CTRL]]"])
        self.assertEqual(final, "内容尾部")

    def test_empty_markers_rejected(self):
        with self.assertRaises(ValueError):
            StreamingMarkerFilter([])


if __name__ == "__main__":
    unittest.main()
