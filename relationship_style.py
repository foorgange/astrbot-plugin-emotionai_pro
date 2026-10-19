# relationship_style.py
"""关系档位、态度基线与心情表达（v4.2.0）

设计目标
--------
好感度不应该只影响面板数字，而应该同时影响三件事：

1. **态度基线**：关系越近，越愿意主动、关心、开玩笑；关系越冷，
   越简短、疏远、不套近乎。
2. **记忆策略**：高好感愿意记住更多细节，低好感只记边界/冲突。
3. **心情修饰**：当前心情只能在关系基线内调整语气，不能把冷淡
   演成亲昵，也不能把亲密演成敌对。

本模块是纯函数，不读写文件、不依赖 AstrBot，方便单测。所有档位
统一从 ``description_diversity.relationship_band`` 派生，避免插件里
再出现第三套阈值。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .description_diversity import pick_fallback, relationship_band

# 关系档位配置。label 只用于提示词里的“关系档位”，不复用可自定义的
# 阶段显示名（阶段名由用户配置，这里保持稳定）。
BAND_PROFILES: Dict[str, Dict[str, Any]] = {
    "negative": {
        "label": "敌对/防备",
        "behavior": (
            "保持防备和距离。句子短，不主动延展话题，不主动关心私事；"
            "不使用昵称、亲昵称呼或撒娇语气；不因为对方示好就立刻回暖。"
        ),
        "initiative": "很低",
        "address": "只用普通称呼",
        "memory_bias": "negative",
        "k_base": 2,
        "per_user_limit": 30,
    },
    "distant": {
        "label": "初识/客气",
        "behavior": (
            "礼貌、友好但保持距离。可以正常回应，不主动套近乎；"
            "不主动提私密细节，不承诺超出关系的亲密。"
        ),
        "initiative": "低",
        "address": "普通称呼，不使用昵称",
        "memory_bias": "neutral",
        "k_base": 2,
        "per_user_limit": 40,
    },
    "casual": {
        "label": "友好/随意",
        "behavior": (
            "可以轻松聊天、接梗、偶尔主动；不要表现得过分亲密，"
            "不主动提及私密细节，不替对方做决定。"
        ),
        "initiative": "中",
        "address": "普通称呼或对方允许的称呼",
        "memory_bias": "neutral",
        "k_base": 3,
        "per_user_limit": 50,
    },
    "close": {
        "label": "亲近",
        "behavior": (
            "可以主动关心、提起共同经历、开适度玩笑；亲昵程度要符合"
            "当前关系，不要越界，不要替对方做决定。"
        ),
        "initiative": "较高",
        "address": "可使用对方接受的昵称",
        "memory_bias": "positive",
        "k_base": 4,
        "per_user_limit": 65,
    },
    "intimate": {
        "label": "亲密",
        "behavior": (
            "可以自然使用亲昵称呼、主动关心、提起共同回忆；仍要尊重"
            "对方边界，不替对方做决定，不把亲密当成理所当然。"
        ),
        "initiative": "高",
        "address": "可使用亲昵称呼（人设允许时）",
        "memory_bias": "positive",
        "k_base": 5,
        "per_user_limit": 80,
    },
}

# 心情主导情绪 -> 记忆召回偏置。惊讶/中立/复合不偏置。
_MOOD_VALENCE = {
    "喜悦": 1,
    "信任": 1,
    "期待": 1,
    "惊讶": 0,
    "中立": 0,
    "悲伤": -1,
    "恐惧": -1,
    "愤怒": -1,
    "厌恶": -1,
}

# 心情 -> 表达倾向。只影响“怎么说”，不改变事实和关系立场。
_MOOD_STYLE = {
    "喜悦": "语气轻快，愿意接梗和分享，句子可以更有活力",
    "信任": "语气温和、耐心，愿意解释和安抚",
    "期待": "更主动追问后续，愿意把话题往前推",
    "惊讶": "反应更快，容易用反问或感叹，但不要夸张",
    "悲伤": "语速放慢，句子变短，少开玩笑，多一些停顿感",
    "恐惧": "谨慎、犹豫，避免冒险和绝对承诺",
    "愤怒": "句子短、直接，耐心下降，但不要人身攻击",
    "厌恶": "疏离、不想接话，避免主动延展",
    "中立": "保持平稳，不额外放大情绪",
}

# 负好感时明确禁止的“越界亲密词”。只拦自动生成的描述，
# 管理员用 /设置态度 手工写的文案不受影响。
_NEGATIVE_INTIMATE_WORDS = (
    "宝贝", "亲爱的", "抱抱", "贴贴", "撒娇", "亲昵", "亲亲",
    "好想你", "想你啦", "宝宝", "小可爱",
)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _write_delta(favor: int) -> int:
    """好感度直接决定“多愿意记”：高好感更容易记细节，低好感只记大事。

    这里刻意以 favor 为主轴，不要求 intimacy 也高——一个人可以“很有好感
    但还不熟”，仍然会愿意记住对方说过的小事；能不能使用亲昵称呼由
    态度/行为基线另行约束。
    """
    if favor >= 60:
        return -2
    if favor >= 30:
        return -1
    if favor < 0:
        return 2
    return 0


def _per_user_limit(favor: int, intimacy: int) -> int:
    """长期记忆保留上限：高好感多留，低好感少留。

    favor=0 时保留 50 条，兼容 v4.1 旧存档（旧记录没有 favor_at_time，
    反序列化后是 0；不能因为升级把老用户的历史从 50 砍到 40）。
    """
    if favor >= 80:
        return 80
    if favor >= 60:
        return 65
    if favor < 0:
        return 30
    if favor == 0:
        return 50
    return int(BAND_PROFILES[relationship_band(favor, intimacy)]["per_user_limit"])


def get_band(favor: Any, intimacy: Any = 0) -> str:
    """返回统一关系档位 key（negative/distant/casual/close/intimate）"""
    return relationship_band(_coerce_int(favor), _coerce_int(intimacy))


def get_profile(favor: Any, intimacy: Any = 0) -> Dict[str, Any]:
    return BAND_PROFILES[get_band(favor, intimacy)]


def get_band_label(favor: Any, intimacy: Any = 0) -> str:
    """关系档位显示名。负好感按强度细分为冷淡/反感/敌对。"""
    f = _coerce_int(favor)
    if f < 0:
        if f >= -30:
            return "冷淡"
        if f >= -70:
            return "反感"
        return "敌对"
    return str(get_profile(f, intimacy)["label"])


def behavior_block(favor: Any, intimacy: Any = 0,
                   stage_name: Optional[str] = None) -> str:
    """主模型提示词里的“关系基线”段落。

    这段是硬约束，优先级高于当前心情；心情只能在其范围内修饰语气。
    """
    f = _coerce_int(favor)
    i = _coerce_int(intimacy)
    profile = get_profile(f, i)
    stage_line = f"阶段：{stage_name}；" if stage_name else ""
    return (
        "【关系基线（优先级高于当前心情）】\n"
        f"{stage_line}关系档位：{get_band_label(f, i)}"
        f"（好感 {f} / 亲密 {i}）\n"
        f"行为基线：{profile['behavior']}\n"
        f"主动程度：{profile['initiative']}；称呼：{profile['address']}。\n"
        "当前心情只能在这个基线内调整语气，不能把冷淡变成亲昵，"
        "也不能把亲密变成敌对。\n"
    )


def memory_policy(favor: Any, intimacy: Any = 0,
                  max_items: Optional[int] = None) -> Dict[str, Any]:
    """好感度驱动的记忆策略。

    返回：
    - ``max_items``：本轮最多召回几条具体记忆；
    - ``bias``：positive/neutral/negative，决定召回时的情绪偏置；
    - ``write_delta``：写入阈值调整量（正数=更难记，负数=更容易记）；
    - ``per_user_limit``：该用户最多保留多少条长期记忆。
    """
    f = _coerce_int(favor)
    i = _coerce_int(intimacy)
    profile = get_profile(f, i)
    k = int(profile["k_base"]) + int(max(f, 0) / 20)
    k = max(1, min(10, k))
    if max_items is not None:
        try:
            k = max(1, min(k, int(max_items)))
        except (TypeError, ValueError):
            pass
    # 好感度是记忆的主轴：高好感偏正面、低好感偏负面；中间档位沿用
    # 关系档位的偏置。
    if f >= 60:
        bias = "positive"
    elif f < 0:
        bias = "negative"
    else:
        bias = str(profile["memory_bias"])
    return {
        "max_items": k,
        "bias": bias,
        "write_delta": _write_delta(f),
        "per_user_limit": _per_user_limit(f, i),
    }


def effective_threshold(base_threshold: int, favor: Any,
                        intimacy: Any = 0) -> int:
    """把关系档位的写入偏移叠加到用户配置的基础阈值上。"""
    f = _coerce_int(favor)
    try:
        base = int(base_threshold)
    except (TypeError, ValueError):
        base = 5
    return max(1, min(10, base + _write_delta(f)))


def mood_valence(mood: Any) -> int:
    """当前心情的情绪价：1 正 / 0 中 / -1 负。"""
    if mood is None:
        return 0
    dominant = str(getattr(mood, "dominant_emotion", "") or "")
    if dominant in _MOOD_VALENCE:
        return _MOOD_VALENCE[dominant]
    # 复合情绪（“复合(喜悦+信任)”）按中性处理，避免偏置过强
    return 0


def mood_block(mood: Any, strength: float = 0.7) -> str:
    """心情注入段落。保留 v4.1.6 的字段格式，额外增加“心情倾向”。

    ``strength`` 是用户配置的影响强度（0~1），只影响措辞强弱，
    不改变主导情绪/强度字段本身。
    """
    if mood is None:
        return ""
    dominant = str(getattr(mood, "dominant_emotion", "中立") or "中立")
    try:
        intensity = float(getattr(mood, "intensity", 0.0) or 0.0)
    except (TypeError, ValueError):
        intensity = 0.0
    intensity = max(0.0, min(1.0, intensity))
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        strength = 0.7
    strength = max(0.0, min(1.0, strength))

    from .global_mood import describe_mood

    label = describe_mood(dominant, intensity)
    style = _MOOD_STYLE.get(dominant, "保持平稳，不额外放大情绪")
    effective = intensity * strength
    if strength <= 0.05:
        level = "极轻"
    elif effective < 0.2:
        level = "轻微"
    elif effective < 0.5:
        level = "明显"
    else:
        level = "强烈"

    return (
        "【当前心情】\n"
        f"主导情绪：{dominant}（{label}）| 强度：{intensity:.2f}/1\n"
        f"心情倾向：{style}（{level}）；它只影响表达方式，"
        "不改变事实、关系基线与你的身份设定。\n"
        "说明：这是你此刻真实的心情。请让它在措辞、句长、标点与语气词上"
        "自然可感知，但不要主动向用户提起它，也不要拿它当借口改变关系基线。\n"
    )


def attitude_matches_band(text: str, favor: Any, intimacy: Any = 0) -> bool:
    """自动生成的态度描述是否越过当前关系档位。

    只做保守拦截：负好感时禁止明显的亲昵称呼/撒娇词；其他档位不拦，
    避免把模型正常的态度变化误判成越界。管理员手工设置不经过这里。
    """
    if not isinstance(text, str) or not text.strip():
        return True
    f = _coerce_int(favor)
    if f >= 0:
        return True
    return not any(word in text for word in _NEGATIVE_INTIMATE_WORDS)


def attitude_fallback(favor: Any, intimacy: Any = 0, seed: Any = 0) -> str:
    """按当前关系档位挑一条态度兜底文案。"""
    return pick_fallback("attitude", favor, intimacy, seed=seed)
