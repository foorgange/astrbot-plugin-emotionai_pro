# global_mood.py
"""全局心情管理 - bot 基于事件/对话变化的共享心情字段

所有用户看到同一个心情。每个用户的 8 维情绪变化经衰减+对冲+温和叠加
汇总到这里，形成 bot 的全局心情与强度（强度范围 0~1）。
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .models import EmotionalMetrics
from .storage import AtomicJSONStorage
from .constants import PathConstants

# 全局心情缓存 TTL（秒）
MOOD_CACHE_TTL = 60

# 心情也随每条对话实时变化的轻量信号强度上限
MOOD_SIGNAL_CAP = 3

# 全局心情演进参数
MOOD_DECAY = 0.95          # 每轮向中性回归的比例（心情不会永远停留在高点）
MOOD_SPIKE_MIN = 3         # 信号总强度 ≥ 该值时触发"情绪冲高"
MOOD_SPIKE_PER_UNIT = 8    # 每单位信号强度对应的冲高值
MOOD_SPIKE_CAP = 40        # 单次冲高上限（避免一条消息直接封顶 100）
MOOD_LOSE_SUPPRESS = 0.2   # 输掉一极的情绪抑制比例（如被骂时 joy ×0.2）

MOOD_POSITIVE_DIMS = ("joy", "trust", "anticipation", "surprise")
MOOD_NEGATIVE_DIMS = ("sadness", "disgust", "anger", "fear")


@dataclass
class GlobalMood:
    """全局心情状态"""
    emotions: EmotionalMetrics = field(default_factory=EmotionalMetrics)
    intensity: float = 0.0  # 0~1，主导情绪强度
    dominant_emotion: str = "中立"
    updated_at: float = 0

    def _recompute(self):
        """根据 8 维情绪重算强度与主导情感

        强度 = 主导情绪值 / 100（0~1）。相比旧的「总和/2」，主导值对
        情绪转移敏感：喜悦 21 被压到 6、愤怒涨到 5，强度立即变化。
        """
        values = [
            self.emotions.joy, self.emotions.trust, self.emotions.fear,
            self.emotions.surprise, self.emotions.sadness, self.emotions.disgust,
            self.emotions.anger, self.emotions.anticipation,
        ]
        self.intensity = round(max(values) / 100.0, 2)
        self.dominant_emotion = self.emotions.get_dominant()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotions": self.emotions.to_dict(),
            "intensity": self.intensity,
            "dominant_emotion": self.dominant_emotion,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalMood":
        """容错恢复：缺字段/损坏时返回默认全零心情，绝不抛异常

        强度/主导情感始终从 8 维情绪重算，不信任旧文件里的强度值
        （旧版是 0~100 整数，新版是 0~1 小数，直接读会溢出）。
        """
        try:
            emotions = EmotionalMetrics(**data.get("emotions", {}))
        except (TypeError, ValueError, KeyError):
            emotions = EmotionalMetrics()
        mood = cls(emotions=emotions)
        mood._recompute()
        mood.updated_at = data.get("updated_at", 0)
        return mood

    @staticmethod
    def default() -> "GlobalMood":
        return GlobalMood()


class GlobalMoodStore:
    """全局心情持久化存储"""

    def __init__(self, data_dir):
        self.storage = AtomicJSONStorage(data_dir / PathConstants.GLOBAL_MOOD_FILE)

    async def load(self) -> GlobalMood:
        """加载全局心情，损坏/失败时返回默认值，绝不中断主流程"""
        try:
            data = await self.storage.load()
            return GlobalMood.from_dict(data) if data else GlobalMood.default()
        except Exception:
            return GlobalMood.default()

    async def save(self, mood: GlobalMood) -> None:
        """保存全局心情，失败仅记录，绝不抛出"""
        try:
            await self.storage.save(mood.to_dict())
        except Exception as e:
            print(f"全局心情保存失败: {e}")

    async def close(self) -> None:
        """关闭存储（无共享资源需释放，保留以统一接口）"""
        pass


def apply_mood_update(mood: GlobalMood, updates: Dict[str, Any]) -> GlobalMood:
    """应用一次对话信号到全局心情（每条对话调用一次）

    设计目标：
    - 实时响应：一条强烈消息即可显著改变心情与强度
    - 情绪翻转：被骂时 joy 大幅下降 + anger/disgust 上升 → 主导情绪切换
    - 防刷爆：各维度绝对值上限 [0,100]，冲高有上限，任何用户无法刷穿

    算法：
    1. 轻微衰减（×MOOD_DECAY 朝 0 回归），避免长时间无对话后心情恒高
    2. 计算信号总强度（所有维度绝对值之和）
    3. 若信号够强（≥MOOD_SPIKE_MIN）：触发「情绪冲高」——
       - 受信号维度先大幅升降（×冲高系数，但不超过该维绝对值上限）
       - 对立维度受对冲影响下降
       - 主导情绪强度冲到高强度档位，确保单条消息就"看得出变化"
    4. 弱信号：仅温和叠加（用于长期累积与细微信号）
    5. clamp 到 [0,100]，重算强度(0~1)与主导情感
    """
    emotion_keys = [
        "joy", "trust", "fear", "surprise",
        "sadness", "disgust", "anger", "anticipation",
    ]

    # 1. 轻微衰减
    decayed = {
        k: max(0, int(getattr(mood.emotions, k) * MOOD_DECAY))
        for k in emotion_keys
    }
    mood.emotions = EmotionalMetrics(**decayed)

    # 2. 归一化信号
    change_map = {
        k: updates[k]
        for k in emotion_keys
        if k in updates and isinstance(updates[k], (int, float))
    }
    normalized = {
        k: (1 if v > 0 else -1 if v < 0 else 0) * min(abs(v), MOOD_SIGNAL_CAP)
        for k, v in change_map.items()
    }

    if normalized:
        # 信号按情绪极性分边：
        #   积极信号 = 正面维度(joy/trust/anticipation/surprise)上升 + 负面维度下降
        #   消极信号 = 负面维度(anger/disgust/sadness/fear)上升 + 正面维度下降
        pos_signal = sum(max(0, normalized.get(dim, 0)) for dim in MOOD_POSITIVE_DIMS) \
                     + sum(max(0, -normalized.get(dim, 0)) for dim in MOOD_NEGATIVE_DIMS)
        neg_signal = sum(max(0, normalized.get(dim, 0)) for dim in MOOD_NEGATIVE_DIMS) \
                     + sum(max(0, -normalized.get(dim, 0)) for dim in MOOD_POSITIVE_DIMS)
        total_signal = pos_signal + neg_signal
        final = dict(decayed)

        if total_signal >= MOOD_SPIKE_MIN:
            # 3. 情绪冲高：赢家通吃，但只冲高「信号最强」的单一维度
            #    （人不会同时有好几种心情，主导情绪单一化）
            winner_is_positive = pos_signal > neg_signal
            spike = min(MOOD_SPIKE_CAP, int(total_signal * MOOD_SPIKE_PER_UNIT))
            if winner_is_positive:
                # 正面赢：冲高正向信号最强的维度，负面被压制
                pos_dims = [d for d in MOOD_POSITIVE_DIMS if normalized.get(d, 0) > 0]
                if pos_dims:
                    top = max(pos_dims, key=lambda d: normalized[d])
                    final[top] = min(100, final[top] + spike)
                for dim in MOOD_NEGATIVE_DIMS:
                    final[dim] = int(final[dim] * MOOD_LOSE_SUPPRESS)
            else:
                # 负面赢：冲高负向信号最强的维度，正面被压制
                neg_dims = [d for d in MOOD_NEGATIVE_DIMS if normalized.get(d, 0) > 0]
                if neg_dims:
                    top = max(neg_dims, key=lambda d: normalized[d])
                    final[top] = min(100, final[top] + spike)
                for dim in MOOD_POSITIVE_DIMS:
                    final[dim] = int(final[dim] * MOOD_LOSE_SUPPRESS)
        else:
            # 4. 弱信号：温和叠加 + 轻微对冲
            for k, delta in normalized.items():
                final[k] = max(0, min(100, final[k] + delta))

        for k in emotion_keys:
            final[k] = max(0, min(100, int(final[k])))
        mood.emotions = EmotionalMetrics(**final)

    # 5. 重算强度与主导情感
    mood._recompute()
    mood.updated_at = time.time()
    return mood


# ---------------------------------------------------------------------------
# 轻量实时心情信号：bot 的心情随每条对话实时变化。
# 仅对用户消息做关键词/语气分析，返回 8 维情感增量，供每次对话后演进全局心情。
# ---------------------------------------------------------------------------

# 每个词命中时对应维度的增量（词与用户消息逐一匹配）
_MOOD_KEYWORD_MAP: Dict[str, Dict[str, int]] = {
    # 积极
    "喜欢": {"joy": 2, "trust": 1},
    "爱": {"joy": 2, "trust": 1},
    "开心": {"joy": 2},
    "高兴": {"joy": 2},
    "谢谢": {"trust": 1, "joy": 1},
    "感谢": {"trust": 1, "joy": 1},
    "感动": {"joy": 2, "trust": 1},
    "温暖": {"joy": 2, "trust": 1},
    "棒": {"joy": 1},
    "好": {"joy": 1},
    "不错": {"joy": 1},
    "可爱": {"joy": 2, "anticipation": 1},
    "漂亮": {"joy": 1},
    "美丽": {"joy": 1},
    "夸": {"joy": 2, "trust": 1},
    "厉害": {"joy": 1, "trust": 1},
    # 消极
    "讨厌": {"disgust": 3, "anger": 2},
    "恨": {"disgust": 3, "anger": 2},
    "生气": {"anger": 3},
    "愤怒": {"anger": 3},
    "伤心": {"sadness": 3},
    "难过": {"sadness": 3},
    "失望": {"sadness": 3, "trust": -2},
    "烦": {"anger": 2, "disgust": 2},
    "滚": {"anger": 3, "disgust": 2},
    "傻": {"anger": 2},
    "笨": {"anger": 2},
    "蠢": {"anger": 2},
    "煞笔": {"disgust": 3, "anger": 2},
    "傻逼": {"disgust": 3, "anger": 2},
    "沙币": {"disgust": 3, "anger": 2},
    "垃圾": {"disgust": 3, "anger": 1},
    "神经病": {"disgust": 2, "anger": 2, "fear": 1},
    "不愿意": {"sadness": 2},
    # 亲密
    "想你": {"trust": 2, "joy": 1, "anticipation": 1},
    "想念": {"trust": 2, "joy": 1},
    "关心": {"trust": 2},
    "担心": {"trust": 1},
    "在乎": {"trust": 2},
    "重要": {"trust": 1},
    "宝贝": {"joy": 2, "trust": 1},
    "亲爱的": {"joy": 2, "trust": 1},
    "拥抱": {"trust": 2, "joy": 1},
    "吻": {"trust": 2, "joy": 1},
    # 冲突
    "吵架": {"anger": 2, "sadness": 1},
    "争执": {"anger": 2},
    "不满": {"anger": 1, "disgust": 1},
    "抱怨": {"sadness": 1, "anger": 1},
    "批评": {"sadness": 1, "disgust": 1},
    "指责": {"anger": 1, "sadness": 1},
    "反对": {"anger": 1},
    "不同意": {"anger": 1},
    # 惊讶/恐惧/期待
    "哇": {"surprise": 2, "joy": 1},
    "天啊": {"surprise": 2, "fear": 1},
    "真的吗": {"surprise": 2},
    "竟然": {"surprise": 2},
    "没想到": {"surprise": 2},
    "害怕": {"fear": 2},
    "恐怖": {"fear": 2},
    "担心": {"fear": 1},
    "期待": {"anticipation": 2},
    "希望": {"anticipation": 1},
    "加油": {"anticipation": 1, "trust": 1},
}

# 语气符号带来的统一信号
_MOOD_STRONG_POSITIVE = re.compile(r"(非常|特别|极其|真的|太)(好|开心|高兴|可爱|喜欢)")
_MOOD_STRONG_NEGATIVE = re.compile(r"(非常|特别|极其|真的|太)(讨厌|生气|愤怒|烦|难过|伤心|恨)")
_MOOD_QUESTION = re.compile(r"[？?]")
_MOOD_EXCLAMATION = re.compile(r"[！!]")
_MOOD_EMOJI_POSITIVE = re.compile(r"[:：][)）]|😊|😄|😍|🥰|🤗")
_MOOD_EMOJI_NEGATIVE = re.compile(r"[:：][(（]|😠|😡|😢|😭|😤")


def compute_mood_signal(user_message: str) -> Dict[str, int]:
    """从用户消息提取心情信号（8 维情感增量）

    规则：
    - 关键词命中 → 累加对应维度增量（每词最多一次，防刷屏）
    - 强烈语气（非常/太/真的+情绪词）→ joy/anger +1
    - 感叹号 → joy +1；疑问号 → surprise +1
    - 正面颜文字 → joy +1；负面颜文字 → sadness +1
    - 最终每个维度的净增量 clamp 到 [-MOOD_SIGNAL_CAP, MOOD_SIGNAL_CAP]
    - 无任何信号时返回空 dict（心情自然衰减，不额外变化）
    """
    if not user_message or not user_message.strip():
        return {}

    text = user_message.strip()
    signals: Dict[str, int] = {}

    def _add(delta: Dict[str, int]):
        for k, v in delta.items():
            signals[k] = signals.get(k, 0) + v

    # 关键词（去重：每词命中一次）
    for word, delta in _MOOD_KEYWORD_MAP.items():
        if word in text:
            _add(delta)

    # 语气
    if _MOOD_STRONG_POSITIVE.search(text):
        _add({"joy": 1})
    if _MOOD_STRONG_NEGATIVE.search(text):
        _add({"anger": 1})
    if _MOOD_EXCLAMATION.search(text):
        # 感叹号放大小情绪：负面语境加剧负面，否则视为兴奋/喜悦
        if signals.get("anger", 0) > 0 or signals.get("sadness", 0) > 0 or signals.get("disgust", 0) > 0:
            _add({"sadness": 1})
        else:
            _add({"joy": 1})
    if _MOOD_QUESTION.search(text):
        _add({"surprise": 1})
    if _MOOD_EMOJI_POSITIVE.search(text):
        _add({"joy": 1})
    if _MOOD_EMOJI_NEGATIVE.search(text):
        _add({"sadness": 1})

    if not signals:
        return {}

    # clamp 到 [-MOOD_SIGNAL_CAP, MOOD_SIGNAL_CAP]
    return {k: max(-MOOD_SIGNAL_CAP, min(MOOD_SIGNAL_CAP, v)) for k, v in signals.items()}
