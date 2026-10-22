# attitude_manager.py
from .models import EnhancedEmotionalState
from . import relationship_style

class AttitudeRelationshipManager:
    """态度关系管理器 - 完整的原有实现"""
    
    TONE_INSTRUCTIONS = {
        "joy": "你的语气应该是愉快、充满热情和活力的。多使用积极的词汇和表情符号。",
        "trust": "你的语气应该是平和、真诚且令人安心的。展现出你的可靠和支持。",
        "fear": "你的语气应该显得有些紧张、谨慎或不安。表现出犹豫或退缩。",
        "surprise": "你的语气应该是震惊、难以置信或充满好奇的。多使用感叹号和反问。",
        "sadness": "你的语气应该是低落、消沉和遗憾的。句子要简短，带有一种无力感。",
        "disgust": "你的语气应该是厌烦、抗拒甚至带有生理性不适的。表现出想回避的态度。",
        "anger": "你的语气应该是愤怒、急躁和具有攻击性的。使用简短有力的句子，表现出不耐烦。",
        "anticipation": "你的语气应该是期待、急切和向往的。关注未来的可能性。"
    }
    
    @classmethod
    def get_tone_instruction(cls, state: EnhancedEmotionalState) -> str:
        """根据主导情感 + 关系基线获取语气指导。

        v4.2.0：心情/情绪只能修饰，不能压过关系基线。旧实现先看情绪，
        低好感用户只要 joy 高一点就会拿到“愉快热情”的语气，和冷淡的
        关系设定打架。现在按档位加一层约束。
        """
        emotions = {
            "joy": state.emotions.joy,
            "trust": state.emotions.trust,
            "fear": state.emotions.fear,
            "surprise": state.emotions.surprise,
            "sadness": state.emotions.sadness,
            "disgust": state.emotions.disgust,
            "anger": state.emotions.anger,
            "anticipation": state.emotions.anticipation
        }
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        band = relationship_style.get_band(state.favor, state.intimacy)
        label = relationship_style.get_band_label(state.favor, state.intimacy)

        if dominant_emotion[1] > 30:
            tone = cls.TONE_INSTRUCTIONS.get(dominant_emotion[0], "保持自然友好的语气。")
        elif band in ("negative", "distant"):
            tone = (
                "保持礼貌、简洁、不过分热络的语气。"
                if band == "distant"
                else "保持简短、冷淡、带防备的语气。"
            )
        elif band == "casual":
            tone = "保持轻松、友好但不过分亲密的语气。"
        elif band == "close":
            tone = "保持温和、亲近、愿意主动关心的语气。"
        else:
            tone = "保持自然、亲密、有默契的语气。"

        if band == "negative":
            return (
                f"{tone} 当前关系档位是“{label}”：不要撒娇、不要使用亲昵"
                "称呼，不要因为一时开心就表现得关系很好。"
            )
        if band == "distant":
            return (
                f"{tone} 当前关系档位是“{label}”：可以友好，但不要主动"
                "套近乎，不要使用昵称或承诺超出关系的亲密。"
            )
        if band == "casual":
            return (
                f"{tone} 当前关系档位是“{label}”：可以接梗，但不要主动"
                "提及私密细节或使用过度亲密的称呼。"
            )
        return tone
