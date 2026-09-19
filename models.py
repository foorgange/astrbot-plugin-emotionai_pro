# models.py
import time
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List, Deque, ClassVar
from enum import Enum
from collections import deque
from pathlib import Path
import re

from astrbot.api import logger

from .constants import EmotionConstants, TimeConstants
from .config import AttitudeType, RelationshipStage, PrivacyLevel
from .stage_names import get_stage_name, normalize_stage_name

@dataclass
class EmotionalMetrics:
    """情感指标 - 专门处理8维情感"""
    joy: int = 0
    trust: int = 0
    fear: int = 0
    surprise: int = 0
    sadness: int = 0
    disgust: int = 0
    anger: int = 0
    anticipation: int = 0
    
    # 情感名称映射
    EMOTION_NAMES: ClassVar[Dict[str, str]] = {
        'joy': '喜悦',
        'trust': '信任',
        'fear': '恐惧',
        'surprise': '惊讶',
        'sadness': '悲伤',
        'disgust': '厌恶',
        'anger': '愤怒',
        'anticipation': '期待'
    }

    # 8 维情感字段顺序（与 EMOTION_NAMES 的键一致，避免各处重复罗列）
    EMOTION_FIELDS: ClassVar[tuple] = (
        'joy', 'trust', 'fear', 'surprise',
        'sadness', 'disgust', 'anger', 'anticipation',
    )
    
    def __post_init__(self):
        """初始化后验证"""
        self._validate_emotions()
    
    def _validate_emotions(self):
        """验证情感值范围"""
        min_v = EmotionConstants.MIN_EMOTION
        max_v = EmotionConstants.MAX_EMOTION
        for name, value in zip(
            self.EMOTION_FIELDS,
            (self.joy, self.trust, self.fear, self.surprise,
             self.sadness, self.disgust, self.anger, self.anticipation),
        ):
            if not min_v <= value <= max_v:
                raise ValueError(
                    f"情感 {self.EMOTION_NAMES.get(name, name)} 值 {value} "
                    f"超出范围 [{min_v}, {max_v}]"
                )
    
    def apply_update(self, updates: Dict[str, int]):
        """应用情感更新"""
        min_v = EmotionConstants.MIN_EMOTION
        max_v = EmotionConstants.MAX_EMOTION
        valid = self.EMOTION_NAMES  # 仅接受已知维度

        for emotion, change in updates.items():
            if emotion in valid:
                current = getattr(self, emotion)
                setattr(self, emotion, max(min_v, min(max_v, current + change)))
            else:
                # 记录警告但不抛出异常
                logger.warning(f"未知的情感类型 '{emotion}'")

        # 更新后重新验证
        self._validate_emotions()
    
    def get_dominant(self) -> str:
        """获取主导情感"""
        # 直接属性访问（比 getattr 循环快得多），单次遍历求最大值与并列项
        joy, trust, fear, surprise = self.joy, self.trust, self.fear, self.surprise
        sadness, disgust, anger, anticipation = (
            self.sadness, self.disgust, self.anger, self.anticipation
        )

        max_value = max(joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
        if max_value == 0:
            return "中立"

        names = self.EMOTION_NAMES
        dominant = []
        if joy == max_value:
            dominant.append(names['joy'])
        if trust == max_value:
            dominant.append(names['trust'])
        if fear == max_value:
            dominant.append(names['fear'])
        if surprise == max_value:
            dominant.append(names['surprise'])
        if sadness == max_value:
            dominant.append(names['sadness'])
        if disgust == max_value:
            dominant.append(names['disgust'])
        if anger == max_value:
            dominant.append(names['anger'])
        if anticipation == max_value:
            dominant.append(names['anticipation'])

        if len(dominant) == 1:
            return dominant[0]
        # 多个情感并列，返回复合描述
        return "复合(" + "+".join(dominant) + ")"

    def emotion_values(self) -> Dict[str, int]:
        """返回 {字段名: 情感值} 映射（供调试/统计用，非热路径）"""
        return {
            name: getattr(self, name)
            for name in self.EMOTION_FIELDS
        }
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取情感摘要"""
        joy, trust, fear, surprise = self.joy, self.trust, self.fear, self.surprise
        sadness, disgust, anger, anticipation = (
            self.sadness, self.disgust, self.anger, self.anticipation
        )
        return {
            'dominant': self.get_dominant(),
            'total_intensity': joy + trust + fear + surprise + sadness + disgust + anger + anticipation,
            'positive_balance': (joy + trust + anticipation) - (fear + sadness + disgust + anger),
            'details': self.to_dict()
        }

@dataclass  
class InteractionStats:
    """互动统计"""
    total_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    last_interaction_time: float = 0
    # v4.0.22：亲密度里程碑的存档字段（旧存档没有这些键时按默认值加载）
    # - deep_conversation_achieved：「首次深度交流」奖励是否已发（一次性防重复）
    # - interaction_streak / last_active_date：连续互动天数与最后互动日期
    #   （"YYYY-MM-DD"，本地时区；用于「连续多日互动」里程碑）
    deep_conversation_achieved: bool = False
    interaction_streak: int = 0
    last_active_date: str = ""
    
    def __post_init__(self):
        """初始化后验证"""
        self._validate_stats()
    
    def _validate_stats(self):
        """验证统计数据的有效性"""
        if self.total_count < 0:
            raise ValueError(f"total_count 不能为负数: {self.total_count}")
        
        if self.positive_count < 0:
            raise ValueError(f"positive_count 不能为负数: {self.positive_count}")
        
        if self.negative_count < 0:
            raise ValueError(f"negative_count 不能为负数: {self.negative_count}")
        
        if self.positive_count + self.negative_count > self.total_count:
            # 自动修复不一致
            self.total_count = self.positive_count + self.negative_count
            logger.warning(f"修复互动统计不一致: total_count调整为{self.total_count}")
        
        if self.last_interaction_time < 0:
            self.last_interaction_time = 0
            logger.warning("修复无效的last_interaction_time")
    
    def record_interaction(self, is_positive: bool = True):
        """记录互动"""
        self.total_count += 1
        if is_positive:
            self.positive_count += 1
        else:
            self.negative_count += 1
        self.last_interaction_time = time.time()
        
        # 记录后验证
        self._validate_stats()
    
    @property
    def positive_ratio(self) -> float:
        """正面互动比例"""
        if self.total_count == 0:
            return 0.0
        return (self.positive_count / self.total_count) * 100
    
    @property
    def negative_ratio(self) -> float:
        """负面互动比例"""
        if self.total_count == 0:
            return 0.0
        return (self.negative_count / self.total_count) * 100
    
    @property
    def neutral_count(self) -> int:
        """中性互动数量"""
        return self.total_count - self.positive_count - self.negative_count
    
    @property
    def days_since_last(self) -> float:
        """距离上次互动的天数"""
        if self.last_interaction_time == 0:
            return float('inf')
        return (time.time() - self.last_interaction_time) / 86400
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'total': self.total_count,
            'positive': self.positive_count,
            'negative': self.negative_count,
            'neutral': self.neutral_count,
            'positive_ratio': self.positive_ratio,
            'negative_ratio': self.negative_ratio,
            'days_since_last': self.days_since_last
        }

@dataclass
class TextDescriptions:
    """文本描述"""
    attitude: str = "中立"
    relationship: str = "陌生人"
    last_attitude_update: float = 0
    last_relationship_update: float = 0
    update_count: int = 0
    
    # 有效的态度和关系模式
    # 允许汉字、字母数字、空格、-，以及常见中文标点（，。！？、；：""''《》·）
    # 注意：AI 生成的自然中文描述几乎必带标点，正则必须放行，否则会被误拒
    VALID_ATTITUDE_PATTERN: ClassVar[str] = r'^[\w\-\s\u4e00-\u9fa5\uff0c\u3002\uff01\uff1f\u3001\uff1b\uff1a""''\u300a\u300b\u00b7]{1,50}$'
    VALID_RELATIONSHIP_PATTERN: ClassVar[str] = r'^[\w\-\s\u4e00-\u9fa5\uff0c\u3002\uff01\uff1f\u3001\uff1b\uff1a""''\u300a\u300b\u00b7]{1,80}$'

    # 预编译正则，避免每次校验都重新解析模式
    _ATTITUDE_RE: ClassVar[re.Pattern] = re.compile(VALID_ATTITUDE_PATTERN)
    _RELATIONSHIP_RE: ClassVar[re.Pattern] = re.compile(VALID_RELATIONSHIP_PATTERN)

    @classmethod
    def is_valid_attitude(cls, text: str) -> bool:
        """校验态度描述是否合法"""
        return bool(cls._ATTITUDE_RE.match(text or ""))

    @classmethod
    def is_valid_relationship(cls, text: str) -> bool:
        """校验关系描述是否合法"""
        return bool(cls._RELATIONSHIP_RE.match(text or ""))
    
    def __post_init__(self):
        """初始化后验证"""
        self._validate_descriptions()
    
    def _validate_descriptions(self):
        """验证描述文本"""
        # 验证态度
        if not self.is_valid_attitude(self.attitude):
            self.attitude = "中立"
            logger.warning(f"修复无效的态度描述: {self.attitude}")

        # 验证关系
        if not self.is_valid_relationship(self.relationship):
            self.relationship = "陌生人"
            logger.warning(f"修复无效的关系描述: {self.relationship}")
        
        # 验证时间戳
        current_time = time.time()
        if self.last_attitude_update > current_time:
            self.last_attitude_update = current_time
        
        if self.last_relationship_update > current_time:
            self.last_relationship_update = current_time
        
        if self.update_count < 0:
            self.update_count = 0
    
    def update_attitude(self, new_attitude: str):
        """更新态度描述"""
        if self.is_valid_attitude(new_attitude):
            self.attitude = new_attitude
            self.last_attitude_update = time.time()
            self.update_count += 1
        else:
            raise ValueError(f"无效的态度描述格式: {new_attitude}")
    
    def update_relationship(self, new_relationship: str):
        """更新关系描述"""
        if self.is_valid_relationship(new_relationship):
            self.relationship = new_relationship
            self.last_relationship_update = time.time()
            self.update_count += 1
        else:
            raise ValueError(f"无效的关系描述格式: {new_relationship}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取描述摘要"""
        return {
            'attitude': self.attitude,
            'relationship': self.relationship,
            'last_attitude_update': self.last_attitude_update,
            'last_relationship_update': self.last_relationship_update,
            'update_count': self.update_count
        }

@dataclass
class EnhancedEmotionalState:
    """优化的情感状态 - 增强验证版本"""
    
    # 核心标识
    user_key: str = ""
    
    # 核心数值状态
    favor: int = 0
    intimacy: int = 0
    
    # 分离的职责模块
    emotions: EmotionalMetrics = field(default_factory=EmotionalMetrics)
    stats: InteractionStats = field(default_factory=InteractionStats)
    descriptions: TextDescriptions = field(default_factory=TextDescriptions)
    
    # 关系阶段
    relationship_stage: str = "初识期"
    stage_composite_score: float = 0.0
    stage_progress: float = 0.0
    
    # 更新追踪
    force_update_counter: int = 0
    last_force_update: float = 0
    
    # 用户设置
    show_status: bool = False
    privacy_level: Optional[int] = None
    
    # 内部状态（不序列化）
    _previous_stage: Optional[str] = None
    _previous_composite: float = 0.0
    
    # 有效的关系阶段**显示名**列表。
    #
    # ⚠️ v4.0.21 起阶段名支持用户自定义（见 stage_names.py），校验不再看
    # 这个出厂默认常量，改看 `stage_names.valid_stage_names()` ——
    # 即「当前生效名」。留在这里只为兼容可能引用它的旧代码。
    VALID_STAGES: ClassVar[List[str]] = ["初识期", "深化期", "承诺期", "共生期", "冷淡期", "反感期", "敌对期"]

    # 有效的阶段「内部 key」列表。
    # 注意与 VALID_STAGES 的区别：VALID_STAGES 是给用户看的中文名，
    # 而 `_previous_stage` 存的是 relationship_manager 用的英文 key
    # （INITIAL / DEEPENING / COMMITMENT / SYMBIOSIS），两者不可混用。
    VALID_STAGE_KEYS: ClassVar[List[str]] = ["INITIAL", "DEEPENING", "COMMITMENT", "SYMBIOSIS"]
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.user_key:
            raise ValueError("user_key is required")
        
        # 验证核心值
        self._validate_core_values()
        
        # 验证关系阶段
        self._validate_relationship_stage()
        
        # 验证进度值
        self._validate_progress_values()
    
    def _validate_core_values(self):
        """验证核心值"""
        # 验证好感度
        if not EmotionConstants.MIN_FAVOR <= self.favor <= EmotionConstants.MAX_FAVOR:
            self.favor = max(EmotionConstants.MIN_FAVOR, 
                           min(EmotionConstants.MAX_FAVOR, self.favor))
            logger.info(f"调整好感度到有效范围: {self.favor}")
        
        # 验证亲密度
        if not EmotionConstants.MIN_INTIMACY <= self.intimacy <= EmotionConstants.MAX_INTIMACY:
            self.intimacy = max(EmotionConstants.MIN_INTIMACY, 
                              min(EmotionConstants.MAX_INTIMACY, self.intimacy))
            logger.info(f"调整亲密度到有效范围: {self.intimacy}")
        
        # 验证强制更新计数器
        if self.force_update_counter < 0:
            self.force_update_counter = 0
        
        # 验证时间戳
        current_time = time.time()
        if self.last_force_update > current_time:
            self.last_force_update = current_time
    
    def _validate_relationship_stage(self):
        """验证关系阶段

        v4.0.21：阶段名可自定义后，校验逻辑改为两步：

        ① **先归一化**：存档里可能是改名前的老名字（出厂默认名）。
           `normalize_stage_name()` 会把它映射到该阶段的当前生效名 ——
           于是用户改名后旧存档立即跟随新名，**不会**被判无效而重置。
        ② 仍非法的才按 favor 修复，且修复目标用当前生效名
           （负向三档同理），不再写死「初识期」。
        """
        normalized = normalize_stage_name(self.relationship_stage)
        if normalized is None:
            # 尝试修复无效的阶段
            if self.favor < 0:
                if self.favor >= -30:
                    self.relationship_stage = get_stage_name("COLD")
                elif self.favor >= -70:
                    self.relationship_stage = get_stage_name("AVERSION")
                else:
                    self.relationship_stage = get_stage_name("HOSTILITY")
            else:
                self.relationship_stage = get_stage_name("INITIAL")
            logger.warning(f"修复无效的关系阶段: {self.relationship_stage}")
        elif normalized != self.relationship_stage:
            # 旧默认名 → 当前生效名（用户改过阶段名）
            logger.info(
                f"关系阶段名同步为新名称: {self.relationship_stage} → {normalized}"
            )
            self.relationship_stage = normalized
    
    def _validate_progress_values(self):
        """验证进度值"""
        # 验证复合评分
        if self.stage_composite_score < -100 or self.stage_composite_score > 200:
            self.stage_composite_score = max(-100, min(200, self.stage_composite_score))
        
        # 验证进度百分比
        if not 0 <= self.stage_progress <= 100:
            self.stage_progress = max(0.0, min(100.0, self.stage_progress))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 - 只序列化必要字段"""
        data = {
            'user_key': self.user_key,
            'favor': self.favor,
            'intimacy': self.intimacy,
            'emotions': self.emotions.to_dict(),
            'stats': asdict(self.stats),
            'descriptions': asdict(self.descriptions),
            'relationship_stage': self.relationship_stage,
            'stage_composite_score': self.stage_composite_score,
            'stage_progress': self.stage_progress,
            'force_update_counter': self.force_update_counter,
            'last_force_update': self.last_force_update,
            'show_status': self.show_status,
            'privacy_level': self.privacy_level,
            # 过渡状态：不落盘的话，重启后第一次判定会被当成"阶段刚跃迁"，
            # 面板恒显一条假的「过渡完成」（见 from_dict 里的详细说明）
            '_previous_stage': self._previous_stage,
            '_previous_composite': self._previous_composite
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedEmotionalState':
        """从字典创建实例 - 增强版本

        ⚠️ v4.1.1：**不要在内部吞异常返回默认状态**。

        旧实现里这一整段包着 try/except，异常时
        `return cls(user_key=...)` —— favor/intimacy/统计全部归零。
        后果是 storage.get_user_state 里 v4.0.19 精心加的修复分支
        （_try_repair_user_data）**永远走不到**：它等的是异常，
        from_dict 却自己把异常咽了。用户看到数值莫名清零，
        日志里只有一行笼统的「从字典创建失败」。

        现在异常原样抛出，由各调用方按自己的策略处理：
          · storage.get_user_state：就地捕获 → 数据修复 → 返回修复态；
          · storage.get_all_user_states：跳过该条并记日志；
          · managers.get_user_state：极端情况下才退默认态。
        """
        # 提取基础字段
        base_data = {
            'user_key': data.get('user_key', ''),
            'favor': data.get('favor', 0),
            'intimacy': data.get('intimacy', 0),
            'relationship_stage': data.get('relationship_stage', get_stage_name('INITIAL')),
            'stage_composite_score': data.get('stage_composite_score', 0.0),
            'stage_progress': data.get('stage_progress', 0.0),
            'force_update_counter': data.get('force_update_counter', 0),
            'last_force_update': data.get('last_force_update', 0),
            'show_status': data.get('show_status', False),
            'privacy_level': data.get('privacy_level')
        }
        
        # 构建子对象
        emotions_data = data.get('emotions', {})
        emotions = EmotionalMetrics(**emotions_data)
        
        stats_data = data.get('stats', {})
        stats = InteractionStats(**stats_data)
        
        descriptions_data = data.get('descriptions', {})
        descriptions = TextDescriptions(**descriptions_data)
        
        state = cls(
            **base_data,
            emotions=emotions,
            stats=stats,
            descriptions=descriptions
        )

        # ⚠️ 必须恢复「过渡状态」的两个内部字段（v4.0.19 修正）
        #
        # `_previous_stage` / `_previous_composite` 是
        # `DynamicWeightManager` 判定阶段升降与过渡保护的输入。旧版
        # `from_dict` 从不恢复它们，于是每次从磁盘加载出来的状态都是
        # `_previous_stage=None -> "INITIAL"`、`_previous_composite=0.0`：
        #
        #   previous_stage = state._previous_stage or "INITIAL"   # 恒为初识期
        #   previous_stage != target_stage  ->  误判为"发生阶段跃迁"
        #   -> 触发过渡保护，面板恒显「过渡完成」
        #
        # 结果就是：每次重启后第一次查询 /好感度，都会看到一条假的
        # 「过渡完成」，且复合评分被 max(current, previous) 保护逻辑
        # 用错误的基线参与计算。
        #
        # 存档里没有这两个键时（旧数据）保持默认，不强行猜测。
        prev_stage = data.get('_previous_stage')
        if isinstance(prev_stage, str) and prev_stage in cls.VALID_STAGE_KEYS:
            state._previous_stage = prev_stage

        prev_composite = data.get('_previous_composite')
        if isinstance(prev_composite, (int, float)) and not isinstance(prev_composite, bool):
            state._previous_composite = float(prev_composite)

        return state
    
    def should_force_update(self, force_update_interval: int) -> bool:
        """判断是否需要强制更新"""
        current_time = time.time()
        
        # 检查对话计数
        if self.force_update_counter >= force_update_interval:
            return True
            
        # 检查时间间隔
        if current_time - self.last_force_update > TimeConstants.THIRTY_MINUTES:
            return True
            
        return False
    
    def reset_force_update_counter(self):
        """重置强制更新计数器"""
        self.force_update_counter = 0
        self.last_force_update = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            'user_key': self.user_key,
            'favor': self.favor,
            'intimacy': self.intimacy,
            'composite_score': self.favor * 0.6 + self.intimacy * 0.4,
            'relationship_stage': self.relationship_stage,
            'stage_progress': self.stage_progress,
            'emotion_summary': self.emotions.get_summary(),
            'interaction_summary': self.stats.get_summary(),
            'description_summary': self.descriptions.get_summary(),
            'show_status': self.show_status,
            'privacy_level': self.privacy_level
        }
    
    def is_valid(self) -> bool:
        """检查状态是否有效"""
        try:
            self._validate_core_values()
            self._validate_relationship_stage()
            self._validate_progress_values()
            return True
        except Exception as e:
            logger.error(f"状态验证失败: {e}")
            return False
    
    def repair(self):
        """尝试修复状态"""
        try:
            self._validate_core_values()
            self._validate_relationship_stage()
            self._validate_progress_values()
            logger.info(f"状态修复完成: {self.user_key}")
        except Exception as e:
            logger.error(f"状态修复失败: {e}")

@dataclass  
class RankingEntry:
    """排行榜条目"""
    rank: int
    user_key: str
    average_score: float
    favor: int
    intimacy: int
    attitude: str
    relationship: str
    display_name: str
    
    def __post_init__(self):
        """初始化后验证"""
        if self.rank < 1:
            raise ValueError(f"排名不能小于1: {self.rank}")
        
        if not self.user_key:
            raise ValueError("user_key不能为空")
        
        if not self.display_name:
            self.display_name = f"用户{self.user_key}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rank': self.rank,
            'user_key': self.user_key,
            'average_score': self.average_score,
            'favor': self.favor,
            'intimacy': self.intimacy,
            'attitude': self.attitude,
            'relationship': self.relationship,
            'display_name': self.display_name
        }

@dataclass
class CacheStats:
    """缓存统计"""
    total_entries: int = 0
    access_count: int = 0
    hit_count: int = 0
    eviction_count: int = 0
    
    def __post_init__(self):
        """初始化后验证"""
        if self.total_entries < 0:
            self.total_entries = 0
        
        if self.access_count < 0:
            self.access_count = 0
        
        if self.hit_count < 0:
            self.hit_count = 0
        
        if self.eviction_count < 0:
            self.eviction_count = 0
        
        if self.hit_count > self.access_count:
            self.hit_count = self.access_count
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        return (self.hit_count / self.access_count * 100) if self.access_count > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """未命中率"""
        return 100 - self.hit_rate
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'total_entries': self.total_entries,
            'access_count': self.access_count,
            'hit_count': self.hit_count,
            'eviction_count': self.eviction_count,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate
        }