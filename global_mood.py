# global_mood.py
"""全局心情管理 - bot 基于事件/对话变化的共享心情字段

所有用户看到同一个心情。每个用户的 8 维情绪变化经衰减+温和叠加
汇总到这里，形成 bot 的全局心情与强度。
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .models import EmotionalMetrics
from .storage import AtomicJSONStorage
from .constants import PathConstants

# 全局心情缓存 TTL（秒）
MOOD_CACHE_TTL = 60


@dataclass
class GlobalMood:
    """全局心情状态"""
    emotions: EmotionalMetrics = field(default_factory=EmotionalMetrics)
    intensity: int = 0
    dominant_emotion: str = "中立"
    updated_at: float = 0

    def _recompute(self):
        """根据 8 维情绪重算强度与主导情感（与 _get_emotion_intensity 同一口径）"""
        self.intensity = min(100, sum([
            self.emotions.joy, self.emotions.trust, self.emotions.fear,
            self.emotions.surprise, self.emotions.sadness, self.emotions.disgust,
            self.emotions.anger, self.emotions.anticipation,
        ]) // 2)
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
        """容错恢复：缺字段/损坏时返回默认全零心情，绝不抛异常"""
        try:
            emotions = EmotionalMetrics(**data.get("emotions", {}))
        except (TypeError, ValueError, KeyError):
            emotions = EmotionalMetrics()
        return cls(
            emotions=emotions,
            intensity=data.get("intensity", 0),
            dominant_emotion=data.get("dominant_emotion", "中立"),
            updated_at=data.get("updated_at", 0),
        )

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
    """应用专家更新到全局心情

    算法：先各维度向中性衰减（×0.95 朝 0 回归），再叠加
    sign(change) × min(abs(change), 2) 温和影响，clamp 到 [0,100]。
    防止单一用户快速刷爆，保持全局缓慢演进。
    """
    emotion_keys = [
        "joy", "trust", "fear", "surprise",
        "sadness", "disgust", "anger", "anticipation",
    ]

    # 1. 各维度衰减
    decayed = {
        k: int(getattr(mood.emotions, k) * 0.95)
        for k in emotion_keys
    }
    mood.emotions = EmotionalMetrics(**decayed)

    # 2. 叠加温和影响（仅取 updates 中存在的 8 维键）
    change_map = {
        k: updates[k]
        for k in emotion_keys
        if k in updates and isinstance(updates[k], (int, float))
    }
    if change_map:
        # 归一化到 [-2, 2]：sign × min(abs, 2)
        normalized = {
            k: int((1 if v > 0 else -1 if v < 0 else 0) * min(abs(v), 2))
            for k, v in change_map.items()
        }
        mood.emotions.apply_update(normalized)

    # 3. 重算强度与主导情感
    mood._recompute()
    mood.updated_at = time.time()
    return mood
