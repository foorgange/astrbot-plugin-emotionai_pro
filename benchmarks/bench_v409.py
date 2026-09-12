# bench_v409.py — 量化 v4.0.9 优化的实际收益
import os
import sys
import timeit

sys.path.insert(0, os.path.abspath('.'))
import tests.bootstrap  # noqa: F401

from emotionai_pro.models import EmotionalMetrics
from emotionai_pro.global_mood import compute_mood_signal, GlobalMood


def bench(label, stmt, number=20000):
    t = timeit.timeit(stmt, number=number, globals=globals())
    print(f"{label:<42} {t/number*1e6:8.3f} µs/次   ({number} 次共 {t:.3f}s)")
    return t / number


print("=" * 78)
print("v4.0.9 优化基准（数值越小越好）")
print("=" * 78)

m = EmotionalMetrics(joy=50, trust=30, anger=10)
bench("get_dominant()", "m.get_dominant()")
bench("emotion_values()", "m.emotion_values()")
bench("get_summary()", "m.get_summary()")

msg = "我今天真的很开心，谢谢你一直关心我！"
bench("compute_mood_signal(中文长句)", f"compute_mood_signal({msg!r})")
bench("compute_mood_signal(短句)", "compute_mood_signal('你好')")

gm = GlobalMood(emotions=EmotionalMetrics(joy=40, anger=60))
bench("GlobalMood._recompute()", "gm._recompute()")

print("=" * 78)
