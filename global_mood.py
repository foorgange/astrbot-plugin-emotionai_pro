# global_mood.py
"""全局心情管理 - bot 基于事件/对话变化的共享心情字段

所有用户看到同一个心情。每个用户的 8 维情绪变化经衰减+对冲+温和叠加
汇总到这里，形成 bot 的全局心情与强度（强度范围 0~1）。
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from astrbot.api import logger

from .models import EmotionalMetrics
from .storage import AtomicJSONStorage
from .constants import PathConstants

# 全局心情缓存 TTL（秒）
MOOD_CACHE_TTL = 60

# 「最近心情变化」历史保留条数（/心情 命令展示用，新的在前）
MOOD_HISTORY_LIMIT = 3

# 心情也随每条对话实时变化的轻量信号强度上限
# （单维度净增量上限；多条消息叠加时由冲高机制放大，这里只限"一条消息"）
MOOD_SIGNAL_CAP = 4

# 全局心情演进参数
#
# 设计取向：心情是高频波动量——同一天内因几句话变好几次很正常，
# 所以"单条消息即可看见变化"；但又不是开关——单条消息不允许打满，
# 冲上高强度档位需要持续性互动，久无消息还会自行回归中性。
MOOD_DECAY = 0.92          # 每轮向中性回归的比例（心情不会永远停留在高点）
MOOD_SPIKE_MIN = 2         # 信号总强度 ≥ 该值时触发"情绪冲高"（一句话即可触发）
MOOD_SPIKE_PER_UNIT = 10   # 每单位信号强度对应的冲高值
MOOD_SPIKE_CAP = 35        # 单次冲高上限（一条消息最多把强度推到 0.35，留足余量）
MOOD_LOSE_SUPPRESS = 0.2   # 输掉一极的情绪抑制比例（如被骂时 joy ×0.2）

# 时间衰减：心情不随消息流动也会自行平复（早上冲的高不会滞留到深夜）。
# 距上次更新每满一个 UNIT，整体再乘 TIME_DECAY；最多累计 MAX_UNITS 个
# 单位（约 3 小时后已接近归零，再久也按 3 小计算，避免无消息时无限衰减）。
MOOD_TIME_DECAY_UNIT = 1800        # 秒（30 分钟）
MOOD_TIME_DECAY = 0.5              # 每个时间单位的衰减系数
MOOD_TIME_DECAY_MAX_UNITS = 6      # 最多累计 6 个单位（3 小时）

MOOD_POSITIVE_DIMS = ("joy", "trust", "anticipation", "surprise")
MOOD_NEGATIVE_DIMS = ("sadness", "disgust", "anger", "fear")

# 8 维情绪键（演进/衰减/预览共用同一份顺序，避免多处各写一份）
MOOD_EMOTION_KEYS = (
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
)


@dataclass
class GlobalMood:
    """全局心情状态"""
    emotions: EmotionalMetrics = field(default_factory=EmotionalMetrics)
    intensity: float = 0.0  # 0~1，主导情绪强度
    dominant_emotion: str = "中立"
    updated_at: float = 0
    # 最近一次「可见」心情变化的原因描述与发生时间（/心情 命令展示用）。
    # 只有主导翻转或强度波动 ≥0.02 的更新才会刷新，避免「波动不大」
    # 这类废话把真正的变化原因覆盖掉；旧存档没有这两个字段，默认空。
    last_change_reason: str = ""
    last_change_at: float = 0
    # 最近若干次「可见」心情变化（/心情 命令展示最近三次，新的在前）。
    # 每项为 {"reason": str, "at": float}；last_change_reason/last_change_at
    # 恒等于 history[0]，作为「最近一次」的镜像保留（旧代码与旧存档依赖）。
    mood_history: list = field(default_factory=list)

    def _recompute(self):
        """根据 8 维情绪重算强度与主导情感

        强度 = 主导情绪值 / 100（0~1）。相比旧的「总和/2」，主导值对
        情绪转移敏感：喜悦 21 被压到 6、愤怒涨到 5，强度立即变化。
        """
        e = self.emotions
        self.intensity = round(
            max(e.joy, e.trust, e.fear, e.surprise,
                e.sadness, e.disgust, e.anger, e.anticipation) / 100.0,
            2,
        )
        self.dominant_emotion = e.get_dominant()

    def push_change_reason(self, reason: str, at: float) -> None:
        """记录一条「可见」的心情变化（刷新最近一次 + 压入历史）

        reason 为空/非字符串时忽略。历史最多保留 MOOD_HISTORY_LIMIT 条，
        最新的在最前（/心情 命令按序展示）。与最新一条文本完全相同的
        原因不重复压栈（同一句话引发的连续波动只留一条，只刷新时间），
        避免「最近三次变化」里出现两条一模一样的记录。
        """
        if not isinstance(reason, str) or not reason.strip():
            return
        ts = (
            float(at)
            if isinstance(at, (int, float)) and not isinstance(at, bool)
            else 0.0
        )
        self.last_change_reason = reason
        self.last_change_at = ts
        history = [e for e in self.mood_history if isinstance(e, dict)]
        if history and history[0].get("reason") == reason:
            history[0] = {"reason": reason, "at": ts}
        else:
            history.insert(0, {"reason": reason, "at": ts})
        self.mood_history = history[:MOOD_HISTORY_LIMIT]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotions": self.emotions.to_dict(),
            "intensity": self.intensity,
            "dominant_emotion": self.dominant_emotion,
            "updated_at": self.updated_at,
            "last_change_reason": self.last_change_reason,
            "last_change_at": self.last_change_at,
            # 深拷贝条目，避免调用方持有同一 dict 引用后互相影响
            "mood_history": [dict(e) for e in self.mood_history if isinstance(e, dict)],
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
        # 变化原因两个字段是 v4.1.5 新增：旧存档没有，类型损坏时回退默认值
        reason = data.get("last_change_reason", "")
        changed_at = data.get("last_change_at", 0)
        mood.last_change_reason = reason if isinstance(reason, str) else ""
        mood.last_change_at = (
            float(changed_at)
            if isinstance(changed_at, (int, float)) and not isinstance(changed_at, bool)
            else 0
        )
        # v4.1.5 之前的旧存档只有单条 last_change_reason，没有 mood_history：
        # 把那条迁移成历史首条，/心情 的「最近三次变化」对老用户同样有内容
        history = data.get("mood_history", [])
        if isinstance(history, list):
            parsed = []
            for item in history[:MOOD_HISTORY_LIMIT]:
                if not isinstance(item, dict):
                    continue
                r = item.get("reason", "")
                if not isinstance(r, str) or not r.strip():
                    continue
                t = item.get("at", 0)
                parsed.append({
                    "reason": r,
                    "at": float(t)
                    if isinstance(t, (int, float)) and not isinstance(t, bool)
                    else 0.0,
                })
            mood.mood_history = parsed
        if not mood.mood_history and mood.last_change_reason:
            mood.mood_history = [
                {"reason": mood.last_change_reason, "at": mood.last_change_at}
            ]
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
            logger.error(f"全局心情保存失败: {e}")

    async def close(self) -> None:
        """关闭存储（无共享资源需释放，保留以统一接口）"""
        pass


def compute_time_decay_factor(updated_at: float) -> float:
    """距上次更新 updated_at 的时间衰减系数（1.0 = 不衰减）

    与 apply_mood_update 第 0 步同一套规则：每满一个
    MOOD_TIME_DECAY_UNIT 再乘 MOOD_TIME_DECAY，最多累计
    MOOD_TIME_DECAY_MAX_UNITS 个单位。updated_at<=0 表示从未更新过，
    不做无依据的猜测，返回 1.0。
    """
    if updated_at <= 0:
        return 1.0
    elapsed = max(0.0, time.time() - updated_at)
    units = min(int(elapsed // MOOD_TIME_DECAY_UNIT), MOOD_TIME_DECAY_MAX_UNITS)
    if units <= 0:
        return 1.0
    return MOOD_TIME_DECAY ** units


def apply_mood_update(mood: GlobalMood, updates: Dict[str, Any], allow_spike: bool = True) -> GlobalMood:
    """应用一次对话信号到全局心情（每条对话调用一次）

    两条输入通道，幅度刻意区分：

    - **关键词/语气信号**（实时反射，allow_spike=True）：可触发「情绪冲高」，
      一条消息即可显著改变主导情绪与强度——心情本来就是高频波动的，
      同一天变几次很正常；
    - **专家评估更新**（LLM 慢思考，allow_spike=False）：只做温和叠加。
      它与实时信号描述的是同一轮对话，若也走冲高就是重复计分，
      会把"一条消息打满"的空间让出来——那种强度只应来自持续性互动。

    设计目标：
    - 实时响应：普通一句话（"谢谢"/"哈哈"）也能把强度推到看得见的档位
    - 情绪翻转：被骂时 joy 大幅下降 + anger/disgust 上升 → 主导情绪切换
    - 防刷爆：单次冲高有上限（MOOD_SPIKE_CAP），任何用户无法一条封顶
    - 自行平复：久无消息时按时间衰减回归中性，心情不会从早上滞留到深夜

    算法：
    0. 时间衰减：距上次更新每满 MOOD_TIME_DECAY_UNIT 秒，整体再乘
       MOOD_TIME_DECAY（最多累计 MOOD_TIME_DECAY_MAX_UNITS 个单位；
       updated_at=0 表示从未更新过，不做无依据的猜测）
    1. 轻微衰减（×MOOD_DECAY 朝 0 回归），避免长时间无对话后心情恒高
    2. 计算信号总强度（所有维度绝对值之和）
    3. 若允许冲高且信号够强（≥MOOD_SPIKE_MIN）：触发「情绪冲高」——
       - 受信号维度先大幅升降（×冲高系数，但不超过该维绝对值上限）
       - 对立维度受对冲影响下降
       - 主导情绪强度冲到高强度档位，确保单条消息就"看得出变化"
    4. 弱信号 / 专家通道：仅温和叠加（用于长期累积与细微信号）
    5. clamp 到 [0,100]，重算强度(0~1)与主导情感
    """
    emotion_keys = MOOD_EMOTION_KEYS

    # 0. 时间衰减：久无消息时心情自行回归中性
    #    （updated_at=0 表示从未更新过，此时不做无依据的猜测）
    time_factor = compute_time_decay_factor(mood.updated_at)

    # 1. 轻微衰减（叠加时间衰减，朝 0 回归）
    decayed = {
        k: max(0, int(getattr(mood.emotions, k) * MOOD_DECAY * time_factor))
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

        if allow_spike and total_signal >= MOOD_SPIKE_MIN:
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
            # 4. 弱信号 / 专家通道（allow_spike=False）：温和叠加 + 轻微对冲
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
    "担心": {"trust": 1, "fear": 1},
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
    "期待": {"anticipation": 2},
    "希望": {"anticipation": 1},
    "加油": {"anticipation": 1, "trust": 1},
    # ---- 群聊常见表达（v4.1.4 扩充：让普通闲聊也能推动心情）----
    # 积极
    "哈哈": {"joy": 2},
    "笑死": {"joy": 2},
    "好耶": {"joy": 2},
    "蚌埠": {"joy": 2},
    "笑不活": {"joy": 2},
    "贴贴": {"joy": 2, "trust": 1},
    "抱抱": {"trust": 2, "joy": 1},
    "么么": {"trust": 2},
    "亲亲": {"trust": 2, "joy": 1},
    "恭喜": {"joy": 2},
    "生日快乐": {"joy": 3, "trust": 1},
    "早安": {"joy": 1},
    "晚安": {"joy": 1, "trust": 1},
    "辛苦了": {"trust": 2},
    "牛逼": {"joy": 2, "trust": 1},
    "太强": {"joy": 2},
    "靠谱": {"trust": 2},
    "信任": {"trust": 2},
    "安慰": {"trust": 1, "joy": 1},
    "陪伴": {"trust": 2},
    "赞": {"joy": 1, "trust": 1},
    "喵": {"joy": 1},
    # 消极
    "无语": {"sadness": 1, "anger": 1},
    "离谱": {"anger": 1, "surprise": 1},
    "逆天": {"anger": 1, "disgust": 1},
    "麻了": {"sadness": 1},
    "摆烂": {"sadness": 1},
    "破防": {"sadness": 2},
    "泪目": {"sadness": 2},
    "呜呜": {"sadness": 2},
    "服了": {"sadness": 1, "anger": 1},
    "气死": {"anger": 3},
    "恶心": {"disgust": 3},
    "膈应": {"disgust": 2},
    "取关": {"anger": 1, "sadness": 1},
    "拉黑": {"anger": 2},
    "爪巴": {"anger": 2},
    "别叫": {"anger": 1},
    "闭嘴": {"anger": 1},
    "吵": {"anger": 1},
    "焯": {"anger": 1},
    "对不起": {"sadness": 1},
    "抱歉": {"sadness": 1},
    "不好意思": {"sadness": 1},
    "不理你": {"sadness": 1, "anger": 1},
    "慌": {"fear": 1},
    # 直播 / 期待（塔菲是主播，这类词在她直播间极高频率）
    "直播": {"anticipation": 1},
    "开播": {"anticipation": 2},
    "下播": {"sadness": 1},
    "生日": {"anticipation": 1, "joy": 1},
    "礼物": {"joy": 1, "anticipation": 1},
    "催更": {"anticipation": 1},
    "想看": {"anticipation": 1, "joy": 1},
    "等着": {"anticipation": 1},
}

# 关键词表的扁平化副本：(词, ((维度, 增量), ...))——
# 内层用 tuple 而非 dict，迭代更快且避免哈希查找
_MOOD_KEYWORD_MAP_ITEMS = tuple(
    (word, tuple(delta.items())) for word, delta in _MOOD_KEYWORD_MAP.items()
)

# 语气符号带来的统一信号
_MOOD_STRONG_POSITIVE = re.compile(r"(非常|特别|极其|真的|太)(好|开心|高兴|可爱|喜欢)")
_MOOD_STRONG_NEGATIVE = re.compile(r"(非常|特别|极其|真的|太)(讨厌|生气|愤怒|烦|难过|伤心|恨)")
_MOOD_QUESTION = re.compile(r"[？?]")
_MOOD_EXCLAMATION = re.compile(r"[！!]")
_MOOD_EMOJI_POSITIVE = re.compile(r"[:：][)）]|😊|😄|😍|🥰|🤗")
_MOOD_EMOJI_NEGATIVE = re.compile(r"[:：][(（]|😠|😡|😢|😭|😤")


# 否定词：出现在关键词前 1~2 字内时该词不计入
# （"不喜欢"不是喜悦、"别生气"不是愤怒、"并不感动"不是感动）
_MOOD_NEGATIONS = ("不", "别", "没", "无")


def _is_negated(text: str, idx: int) -> bool:
    """关键词（起始于 idx）前 1~2 字是否定词"""
    prefix = text[max(0, idx - 2):idx]
    return any(n in prefix for n in _MOOD_NEGATIONS)


def compute_mood_signal(user_message: str) -> Dict[str, int]:
    """从用户消息提取心情信号（8 维情感增量）

    规则：
    - 关键词命中 → 累加对应维度增量（每词最多一次，防刷屏）
    - 被"不/别/没/无"否定的关键词不计（"不喜欢"不产生喜悦信号）
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
    get = signals.get

    # 关键词（去重：每词命中一次；带否定前缀的词跳过）
    for word, delta in _MOOD_KEYWORD_MAP_ITEMS:
        idx = text.find(word)
        if idx == -1 or _is_negated(text, idx):
            continue
        for k, v in delta:
            signals[k] = get(k, 0) + v

    # 语气
    if _MOOD_STRONG_POSITIVE.search(text):
        signals["joy"] = get("joy", 0) + 1
    if _MOOD_STRONG_NEGATIVE.search(text):
        signals["anger"] = get("anger", 0) + 1
    if _MOOD_EXCLAMATION.search(text):
        # 感叹号放大小情绪：负面语境加剧负面，否则视为兴奋/喜悦
        if get("anger", 0) > 0 or get("sadness", 0) > 0 or get("disgust", 0) > 0:
            signals["sadness"] = get("sadness", 0) + 1
        else:
            signals["joy"] = get("joy", 0) + 1
    if _MOOD_QUESTION.search(text):
        signals["surprise"] = get("surprise", 0) + 1
    if _MOOD_EMOJI_POSITIVE.search(text):
        signals["joy"] = get("joy", 0) + 1
    if _MOOD_EMOJI_NEGATIVE.search(text):
        signals["sadness"] = get("sadness", 0) + 1

    if not signals:
        return {}

    # clamp 到 [-MOOD_SIGNAL_CAP, MOOD_SIGNAL_CAP]
    return {k: max(-MOOD_SIGNAL_CAP, min(MOOD_SIGNAL_CAP, v)) for k, v in signals.items()}


# ---------------------------------------------------------------------------
# 「/心情」命令支持：衰减预览、变化原因生成、消息摘录
# ---------------------------------------------------------------------------

def preview_time_decayed(mood: GlobalMood) -> GlobalMood:
    """返回按时间衰减预览的心情副本（只读：不修改传入对象、不落盘）

    时间衰减是惰性的——只在下次 apply_mood_update 时生效。命令查询时
    若距上次更新已久，直接展示存档值会高估当前强度，故展示前先按
    同一套规则预览。无衰减（系数 1.0）时直接返回原对象，调用方须
    视为只读。
    """
    factor = compute_time_decay_factor(mood.updated_at)
    if factor >= 1.0:
        return mood
    decayed = {
        k: max(0, int(getattr(mood.emotions, k) * factor))
        for k in MOOD_EMOTION_KEYS
    }
    preview = GlobalMood(emotions=EmotionalMetrics(**decayed))
    preview._recompute()
    preview.updated_at = mood.updated_at
    preview.last_change_reason = mood.last_change_reason
    preview.last_change_at = mood.last_change_at
    preview.mood_history = [
        dict(e) for e in mood.mood_history if isinstance(e, dict)
    ]
    return preview


# 心情变化原因描述：主导情绪 → 「变得__」的口语短语
_MOOD_CHANGE_PHRASES = {
    "喜悦": "开心起来",
    "信任": "安心起来",
    "期待": "期待起来",
    "惊讶": "惊讶起来",
    "悲伤": "难过起来",
    "厌恶": "嫌弃起来",
    "愤怒": "生气起来",
    "恐惧": "慌张起来",
    "中立": "平静起来",
}

# 主导情绪 → 强度分级口语描述（v4.1.6）
#
# 旧展示只有强度一维：「愤怒（心情平静）」——主导情绪与标签自相矛盾
# （都愤怒了还说什么心情平静）。现在具体情绪主导时，标签就是该情绪
# 本身按强度的分级描述；只有「中立」主导（8 维全零、没有任何情绪
# 冒头）或复合/未知名称，才用「心情平静」这类整体心境词。
_MOOD_EMOTION_GRADES: Dict[str, tuple] = {
    #                <0.15     <0.35     <0.55     <0.75      ≥0.75
    "喜悦": ("有点开心", "挺开心", "很开心", "特别开心", "兴高采烈"),
    "信任": ("有点安心", "挺安心", "很安心", "特别安心", "全心信赖"),
    "期待": ("有点期待", "挺期待", "很期待", "特别期待", "期待不已"),
    "惊讶": ("有点惊讶", "挺惊讶", "很惊讶", "非常惊讶", "大吃一惊"),
    "悲伤": ("有点低落", "挺难过", "很难过", "特别难过", "伤心透了"),
    "厌恶": ("有点嫌弃", "挺嫌弃", "很嫌弃", "非常嫌弃", "嫌弃至极"),
    "愤怒": ("有点生气", "挺恼火", "很生气", "非常生气", "气炸了"),
    "恐惧": ("有点慌", "挺不安", "很慌张", "非常慌张", "害怕极了"),
}

# 中立/复合/未知主导时的整体心境词（强度分档阈值与分级表一致）
_MOOD_NEUTRAL_GRADES = ("心情平静", "心情平稳", "心情微动", "心情波动", "情绪高涨")


def describe_mood(dominant_emotion: str, intensity: float) -> str:
    """主导情绪 + 强度（0~1）→ 中文心情描述

    展示（/心情、/查看好感、情感状态面板）与 LLM 注入共用这一份真源，
    保证用户看到的描述和模型收到的心情指引是同一套措辞。
    """
    grades = _MOOD_EMOTION_GRADES.get(dominant_emotion or "") or _MOOD_NEUTRAL_GRADES
    try:
        value = float(intensity)
    except (TypeError, ValueError):
        value = 0.0
    if value < 0.15:
        return grades[0]
    if value < 0.35:
        return grades[1]
    if value < 0.55:
        return grades[2]
    if value < 0.75:
        return grades[3]
    return grades[4]


def build_mood_change_reason(
    before_dominant: str,
    before_intensity: float,
    after: GlobalMood,
    trigger: str,
    subject: str = "",
) -> Optional[str]:
    """根据一次心情更新生成「最近一次变化原因」描述

    返回 None 表示这次更新没有可见变化（主导未翻转且强度波动
    <0.05）——此时调用方应保留上一条有记录的原因。阈值取 0.05
    （半个心情档位的 1/3）：弱消息引发的逐轮衰减（×0.92）通常只
    带来 0.01~0.03 的跌落，够不上「可见」；而情绪冲高（≥0.1）与
    久无消息的时间衰减（动辄 0.2+）都远超它。

    trigger：触发来源（如 听到「哈哈」）；subject：bot 人设名，
    为空时用「心情」作主语，保证句子完整。
    """
    flipped = after.dominant_emotion != before_dominant
    delta = after.intensity - before_intensity
    if not flipped and abs(delta) < 0.05:
        return None

    who = subject if subject else "心情"
    if delta < 0 and not flipped:
        # 强度回落多半是逐轮衰减/时间衰减主导，别把账算到触发消息头上
        return f"{trigger}之后，{who}的心情渐渐平复"
    phrase = _MOOD_CHANGE_PHRASES.get(after.dominant_emotion, "有所波动")
    return f"因为{trigger}，{who}变得{phrase}"


def mood_trigger_excerpt(text: str, limit: int = 12) -> str:
    """心情变化原因里的消息摘录：去换行/压缩空白，超长截断加省略号

    原因描述会随心情存档持久化并可能被 /心情 命令展示，摘录必须
    短——既避免刷屏，也避免把整条长消息塞进存档。
    """
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "…"
