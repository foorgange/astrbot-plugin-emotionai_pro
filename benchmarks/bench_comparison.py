# benchmarks/bench_comparison.py
"""v4.0.9 优化收益量化：直接对比「旧实现」与「新实现」的等价逻辑。

不依赖 git stash，把两版实现写在同一文件内，保证公平对比。
"""
import os
import re
import sys
import time
import timeit
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.bootstrap  # noqa: F401

from emotionai_pro.models import EmotionalMetrics
from emotionai_pro.global_mood import compute_mood_signal

N = 30000


# --------------------------------------------------------------------------
# get_dominant：旧实现（每次构造中文键字典 + 两次遍历）
# --------------------------------------------------------------------------
def old_get_dominant(m: EmotionalMetrics) -> str:
    emotions = {
        "喜悦": m.joy, "信任": m.trust, "恐惧": m.fear, "惊讶": m.surprise,
        "悲伤": m.sadness, "厌恶": m.disgust, "愤怒": m.anger, "期待": m.anticipation,
    }
    max_value = max(emotions.values())
    if max_value == 0:
        return "中立"
    dominant = [n for n, v in emotions.items() if v == max_value]
    if len(dominant) == 1:
        return dominant[0]
    return f"复合({'+'.join(dominant)})"


# --------------------------------------------------------------------------
# get_summary：旧实现（重复列举 8 个字段）
# --------------------------------------------------------------------------
def old_get_summary(m: EmotionalMetrics) -> Dict:
    return {
        'dominant': old_get_dominant(m),
        'total_intensity': sum([m.joy, m.trust, m.fear, m.surprise,
                                m.sadness, m.disgust, m.anger, m.anticipation]),
        'positive_balance': (m.joy + m.trust + m.anticipation) -
                            (m.fear + m.sadness + m.disgust + m.anger),
        'details': m.to_dict(),
    }


# --------------------------------------------------------------------------
# compute_mood_signal：旧实现（闭包 _add，每次调用重建函数对象）
# --------------------------------------------------------------------------
_KEYWORD_MAP = {
    "喜欢": {"joy": 2, "trust": 1}, "爱": {"joy": 2, "trust": 1},
    "开心": {"joy": 2}, "高兴": {"joy": 2}, "谢谢": {"trust": 1, "joy": 1},
    "感谢": {"trust": 1, "joy": 1}, "感动": {"joy": 2, "trust": 1},
    "温暖": {"joy": 2, "trust": 1}, "棒": {"joy": 1}, "好": {"joy": 1},
    "不错": {"joy": 1}, "可爱": {"joy": 2, "anticipation": 1},
    "漂亮": {"joy": 1}, "美丽": {"joy": 1}, "夸": {"joy": 2, "trust": 1},
    "厉害": {"joy": 1, "trust": 1}, "讨厌": {"disgust": 3, "anger": 2},
    "恨": {"disgust": 3, "anger": 2}, "生气": {"anger": 3}, "愤怒": {"anger": 3},
    "伤心": {"sadness": 3}, "难过": {"sadness": 3}, "失望": {"sadness": 3, "trust": -2},
    "烦": {"anger": 2, "disgust": 2}, "滚": {"anger": 3, "disgust": 2},
    "傻": {"anger": 2}, "笨": {"anger": 2}, "蠢": {"anger": 2},
    "煞笔": {"disgust": 3, "anger": 2}, "傻逼": {"disgust": 3, "anger": 2},
    "沙币": {"disgust": 3, "anger": 2}, "垃圾": {"disgust": 3, "anger": 1},
    "神经病": {"disgust": 2, "anger": 2, "fear": 1}, "不愿意": {"sadness": 2},
    "想你": {"trust": 2, "joy": 1, "anticipation": 1}, "想念": {"trust": 2, "joy": 1},
    "关心": {"trust": 2}, "担心": {"trust": 1}, "在乎": {"trust": 2},
    "重要": {"trust": 1}, "宝贝": {"joy": 2, "trust": 1},
    "亲爱的": {"joy": 2, "trust": 1}, "拥抱": {"trust": 2, "joy": 1},
    "吻": {"trust": 2, "joy": 1}, "吵架": {"anger": 2, "sadness": 1},
    "争执": {"anger": 2}, "不满": {"anger": 1, "disgust": 1},
    "抱怨": {"sadness": 1, "anger": 1}, "批评": {"sadness": 1, "disgust": 1},
    "指责": {"anger": 1, "sadness": 1}, "反对": {"anger": 1}, "不同意": {"anger": 1},
    "哇": {"surprise": 2, "joy": 1}, "天啊": {"surprise": 2, "fear": 1},
    "真的吗": {"surprise": 2}, "竟然": {"surprise": 2}, "没想到": {"surprise": 2},
    "害怕": {"fear": 2}, "恐怖": {"fear": 2}, "担心": {"fear": 1},
    "期待": {"anticipation": 2}, "希望": {"anticipation": 1},
    "加油": {"anticipation": 1, "trust": 1},
}
_STRONG_POS = re.compile(r"(非常|特别|极其|真的|太)(好|开心|高兴|可爱|喜欢)")
_STRONG_NEG = re.compile(r"(非常|特别|极其|真的|太)(讨厌|生气|愤怒|烦|难过|伤心|恨)")
_QUESTION = re.compile(r"[？?]")
_EXCLAIM = re.compile(r"[！!]")
_EMOJI_POS = re.compile(r"[:：][)）]|😊|😄|😍|🥰|🤗")
_EMOJI_NEG = re.compile(r"[:：][(（]|😠|😡|😢|😭|😤")
_CAP = 3


def old_compute_mood_signal(user_message: str) -> Dict[str, int]:
    if not user_message or not user_message.strip():
        return {}
    text = user_message.strip()
    signals: Dict[str, int] = {}

    def _add(delta):
        for k, v in delta.items():
            signals[k] = signals.get(k, 0) + v

    for word, delta in _KEYWORD_MAP.items():
        if word in text:
            _add(delta)
    if _STRONG_POS.search(text):
        _add({"joy": 1})
    if _STRONG_NEG.search(text):
        _add({"anger": 1})
    if _EXCLAIM.search(text):
        if signals.get("anger", 0) > 0 or signals.get("sadness", 0) > 0 or signals.get("disgust", 0) > 0:
            _add({"sadness": 1})
        else:
            _add({"joy": 1})
    if _QUESTION.search(text):
        _add({"surprise": 1})
    if _EMOJI_POS.search(text):
        _add({"joy": 1})
    if _EMOJI_NEG.search(text):
        _add({"sadness": 1})
    if not signals:
        return {}
    return {k: max(-_CAP, min(_CAP, v)) for k, v in signals.items()}


def run(label, old_fn, new_fn, n=N):
    old_t = timeit.timeit(old_fn, number=n) / n * 1e6
    new_t = timeit.timeit(new_fn, number=n) / n * 1e6
    delta = (new_t - old_t) / old_t * 100
    flag = "↓" if delta < 0 else "↑"
    print(f"{label:<34} 旧 {old_t:7.3f} µs   新 {new_t:7.3f} µs   {flag}{abs(delta):5.1f}%")


if __name__ == "__main__":
    m = EmotionalMetrics(joy=50, trust=30, anger=10)
    gm = EmotionalMetrics(joy=40, anger=60)

    print("=" * 88)
    print("v4.0.9 优化收益（micro-benchmark, 越低越好）")
    print("=" * 88)
    run("get_dominant()",
        lambda: old_get_dominant(m),
        lambda: m.get_dominant())
    run("get_summary()",
        lambda: old_get_summary(m),
        lambda: m.get_summary())
    long_msg = "我今天真的很开心，谢谢你一直关心我！"
    run(f"compute_mood_signal(long)",
        lambda: old_compute_mood_signal(long_msg),
        lambda: compute_mood_signal(long_msg))
    run("compute_mood_signal(short)",
        lambda: old_compute_mood_signal("你好"),
        lambda: compute_mood_signal("你好"))
    print("=" * 88)
