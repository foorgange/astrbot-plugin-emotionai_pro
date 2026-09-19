# relationship_manager.py
import time
from typing import Dict, Any, Optional, Tuple

from .models import EnhancedEmotionalState
from .constants import EmotionConstants
from .stage_names import get_stage_name, stage_key_from_name

# 阶段顺序（用于计算"下一阶段"）
# 内部逻辑一律用英文 key；显示名统一走 stage_names.get_stage_name(key)，
# 用户在配置界面改阶段名时只影响显示，不动这里的任何判定逻辑。
STAGE_ORDER = ["INITIAL", "DEEPENING", "COMMITMENT", "SYMBIOSIS"]


def _next_stage_key(stage: str) -> Optional[str]:
    """返回给定阶段的下一阶段 key；已是最高阶段返回 None"""
    if stage in STAGE_ORDER:
        idx = STAGE_ORDER.index(stage)
        if idx + 1 < len(STAGE_ORDER):
            return STAGE_ORDER[idx + 1]
    return None


class DynamicWeightManager:
    """动态权重管理器 - 完整的原有实现"""
    
    # 关系阶段配置
    #
    # ⚠️ 这里的 "name" 只是**出厂默认显示名**。用户可在配置界面自定义
    # 阶段名（v4.0.21），生效名以 stage_names.get_stage_name(key) 为准，
    # 所有对外输出（面板 / 建议文案 / 下一阶段名）都走那个函数取。
    STAGE_CONFIGS = {
        "INITIAL": {
            "name": "初识期",
            "description": "好感驱动，建立吸引",
            "favor_weight": 0.7,
            "intimacy_weight": 0.3,
            "favor_range": (0, 40),
            "intimacy_range": (0, 30),
            "composite_threshold": 25,
            "transition_buffer": 3,
            "intimacy_boost_factor": 4.0
        },
        "DEEPENING": {
            "name": "深化期", 
            "description": "互动平衡，共同成长",
            "favor_weight": 0.5,
            "intimacy_weight": 0.5,
            "favor_range": (40, 70),
            "intimacy_range": (30, 60),
            "composite_threshold": 55,
            "transition_buffer": 5,
            "intimacy_boost_factor": 3.6
        },
        "COMMITMENT": {
            "name": "承诺期",
            "description": "亲密主导，根基稳固", 
            "favor_weight": 0.3,
            "intimacy_weight": 0.7,
            "favor_range": (70, 90),
            "intimacy_range": (60, 85),
            "composite_threshold": 80,
            "transition_buffer": 7,
            "intimacy_boost_factor": 3.0
        },
        "SYMBIOSIS": {
            "name": "共生期",
            "description": "完全融合，不分彼此",
            "favor_weight": 0.5,
            "intimacy_weight": 0.5,
            "favor_range": (90, 100),
            "intimacy_range": (85, 100),
            "composite_threshold": 95,
            "transition_buffer": 10,
            "intimacy_boost_factor": 1.0
        }
    }
    
    # 阶段过渡的亲密度门槛（v4.0.22）
    #
    # 复合评分达到下一阶段阈值只是**必要条件**；亲密度还必须达到
    # 「亲密度上限 × 该百分比」才能完成过渡。0 = 关闭门槛（旧行为）。
    #
    # ⚠️ 与 EmotionConstants 同样的理由做成类属性 + configure() 注入：
    # 本类是纯 classmethod 工具类，拿不到插件实例，而门槛百分比来自
    # 用户配置。注入点两处缺一不可：
    #   main.py::__init__ 与 config_manager.py::_apply_numeric_bounds
    TRANSITION_INTIMACY_PCT = 50

    # 分阶段门槛（v4.0.23）：key 是**目标阶段**的英文 key，值是占最大亲密度的百分比。
    #
    # 为什么不能只有一个统一值：承诺期权重 favor 0.3 / intimacy 0.7，复合分
    # 冲到 80 本身就要求亲密度很高（favor 100 时需 ≥72），统一门槛 40 在
    # 第二、三次过渡永远拦不到东西 —— 只有第一次有牙齿。
    # 查表顺序：STAGE_INTIMACY_PCT[目标阶段] → 未列出的阶段回退到
    # TRANSITION_INTIMACY_PCT（含 INITIAL，它不作为升级目标出现）。
    STAGE_INTIMACY_PCT = {
        "DEEPENING": 20,
        "COMMITMENT": 40,
        "SYMBIOSIS": 60,
    }

    @classmethod
    def configure(cls, transition_intimacy_pct: int = None,
                  stage_intimacy_pcts: dict = None) -> None:
        """用插件配置注入亲密度门槛百分比（0-100，其余值忽略）

        transition_intimacy_pct：统一门槛（未按阶段配置时的回退值）
        stage_intimacy_pcts：{目标阶段key: 百分比}，**整体替换**语义
            （配置里删掉某个 key = 该阶段回退到统一门槛，热重载立即生效）
        """
        if transition_intimacy_pct is None or isinstance(transition_intimacy_pct, bool):
            pass
        else:
            try:
                pct = int(transition_intimacy_pct)
            except (TypeError, ValueError):
                pass
            else:
                if 0 <= pct <= 100:
                    cls.TRANSITION_INTIMACY_PCT = pct

        if stage_intimacy_pcts is None or isinstance(stage_intimacy_pcts, bool):
            return
        if not isinstance(stage_intimacy_pcts, dict):
            return
        sanitized = {}
        for stage_key, value in stage_intimacy_pcts.items():
            if stage_key not in cls.STAGE_CONFIGS or stage_key == "INITIAL":
                continue
            if isinstance(value, bool):
                continue
            try:
                pct = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= pct <= 100:
                sanitized[stage_key] = pct
        cls.STAGE_INTIMACY_PCT = sanitized

    @classmethod
    def reset(cls) -> None:
        """恢复出厂门槛（测试用）"""
        cls.TRANSITION_INTIMACY_PCT = 50
        cls.STAGE_INTIMACY_PCT = {
            "DEEPENING": 20,
            "COMMITMENT": 40,
            "SYMBIOSIS": 60,
        }

    @classmethod
    def get_stage_intimacy_pct(cls, target_stage: str = None) -> int:
        """取某个升级目标阶段适用的亲密度门槛百分比

        未按阶段配置（或该阶段不在表里）时回退到统一门槛
        TRANSITION_INTIMACY_PCT —— 老配置只改过统一值也不会失效。
        """
        if target_stage and target_stage in cls.STAGE_INTIMACY_PCT:
            return cls.STAGE_INTIMACY_PCT[target_stage]
        return cls.TRANSITION_INTIMACY_PCT

    @classmethod
    def get_intimacy_gate(cls, state: EnhancedEmotionalState,
                          target_stage: str = None) -> Dict[str, Any]:
        """计算阶段过渡的亲密度门槛（v4.0.22，v4.0.23 起按目标阶段分档）

        门槛 = 亲密度上限（EmotionConstants.MAX_INTIMACY，跟随用户配置）
        × 目标阶段对应的百分比。target_stage 为 None 时用统一门槛
        TRANSITION_INTIMACY_PCT（保持旧调用方行为）。返回 required /
        current / gap / met，目标阶段显示名由调用方补 to_stage。
        """
        pct = cls.get_stage_intimacy_pct(target_stage)
        max_intimacy = EmotionConstants.MAX_INTIMACY
        required = int(max_intimacy * pct / 100)
        current = state.intimacy
        gap = max(0, required - current)
        return {
            "required": required,
            "current": current,
            "gap": gap,
            "met": gap <= 0,
            "threshold_pct": pct,
            "target_stage": target_stage,
        }

    @classmethod
    def is_favor_frozen(cls, state: EnhancedEmotionalState) -> bool:
        """过渡期亲密度未达标 → 好感度冻结（v4.0.22）

        只在**阶段过渡期**且门槛未达标时成立；负好感不走阶段门禁。
        calculate_stage 是纯读（不写 _previous_*），可在应用更新前安全调用。
        """
        if state.favor < 0:
            return False
        _, transition_info = cls.calculate_stage(state)
        return bool(transition_info.get("intimacy_gate_blocked"))

    @classmethod
    def calculate_stage(cls, state: EnhancedEmotionalState) -> Tuple[str, Dict[str, Any]]:
        """计算当前关系阶段和过渡状态 - 保持原有逻辑"""
        current_composite = cls._calculate_raw_composite(state)
        # v4.1.1：基线缺失/非法时按分数归档，绝不猜 INITIAL（见
        # _resolve_previous_stage 的完整说明）
        previous_stage = cls._resolve_previous_stage(state, current_composite)
        previous_composite = state._previous_composite
        
        # 判断当前阶段
        target_stage = cls._get_stage_by_score(current_composite, state)

        # v4.0.22：亲密度门槛。分数够升级、但亲密度没达到「上限×百分比」时，
        # 把目标阶段压回当前阶段（过渡卡住）；_check_transition_status 会把
        # 它标成「门槛阻断的过渡」，持续到亲密度达标为止。
        gate_blocked = False
        gate_target = None
        if STAGE_ORDER.index(target_stage) > STAGE_ORDER.index(previous_stage):
            # v4.0.23：按**目标阶段**取门槛（深化期 20% / 承诺期 40% / 共生期 60%），
            # 未分档配置的阶段回退到统一门槛
            if not cls.get_intimacy_gate(state, target_stage)["met"]:
                gate_blocked = True
                gate_target = target_stage
                target_stage = previous_stage

        # 检查是否处于阶段过渡期
        transition_info = cls._check_transition_status(
            state, previous_stage, target_stage, previous_composite, current_composite,
            gate_blocked=gate_blocked, gate_target=gate_target
        )
        
        return target_stage, transition_info
    
    @classmethod
    def _calculate_raw_composite(cls, state: EnhancedEmotionalState) -> float:
        """计算原始复合评分"""
        current_stage = cls._get_stage_by_score(
            state.favor * 0.6 + state.intimacy * 0.4,
            state
        )
        stage_config = cls.STAGE_CONFIGS[current_stage]
        return state.favor * stage_config["favor_weight"] + state.intimacy * stage_config["intimacy_weight"]
    
    @classmethod
    def _raw_stage_by_score(cls, composite_score: float) -> str:
        """无滞后的裸阶段判定（只看复合分阈值，不看历史基线）"""
        if composite_score >= cls.STAGE_CONFIGS["SYMBIOSIS"]["composite_threshold"]:
            return "SYMBIOSIS"
        if composite_score >= cls.STAGE_CONFIGS["COMMITMENT"]["composite_threshold"]:
            return "COMMITMENT"
        if composite_score >= cls.STAGE_CONFIGS["DEEPENING"]["composite_threshold"]:
            return "DEEPENING"
        return "INITIAL"

    @classmethod
    def _resolve_previous_stage(cls, state: EnhancedEmotionalState,
                                current_composite: float) -> str:
        """取过渡基线阶段；基线缺失/非法时按当前复合分重建

        ⚠️ v4.1.1：绝不能默认 "INITIAL"。

        `_previous_stage` 只在 `get_stage_info`（展示路径）里推进，消息流
        只读不写。于是以下用户的基线恒为空：
          · 被 `_try_repair_user_data` 重建过的存档（v4.1.1 前修复清单
            漏了这两个字段，键在但值是 null）；
          · v4.0.19 之前的老存档（键不存在）；
          · 从来没看过面板/没触发过展示路径的用户。

        空基线猜 INITIAL 的后果（线上真实实例，用户 2961113185）：
          复合分早已越过承诺期/共生期的老用户，面板被打回「初识期」，
          并且被**最高阶段**的亲密度门槛卡住——v4.0.22 起阻断期间不推进
          基线，错基线便永远自愈不了，直到亲密度硬撑到最高档才一次性
          跳级。该用户 favor=108 / intimacy=118 时面板显示
          「关注塔菲喵 + 亲密度 118/120」，而他实际早在承诺期。

        正确做法：按当前复合分归档到「分数对应阶段的下一级」，即视为
        即将过渡——
          · 显示回到真正的上一阶段（承诺期），门槛按目标阶段正常生效；
          · 亲和度达标后 `get_stage_info` 把恢复出的基线落盘，自愈完成；
          · 分数本就在初识期的新用户不受影响（下一级仍是初识期）。

        ⚠️ v4.1.2 追加：重建值不得低于**存档里已达成的阶段**。

        亲密度门槛是 v4.0.22 才引入的，此前阶段晋升只看复合分。于是有
        一批老用户「阶段早就到位、亲密度其实没达标」，他们的存档
        `relationship_stage` 里写着真实阶段（承诺期/共生期），而基线
        因为存档修复或老格式被清空。若只按分数归档，这些人会被门槛
        压回下一级——相当于因为「当年没有的规则」被降级。

        修法：重建时取「分数归档」与「存档阶段」中较高的那个。
          · 老用户保持当前阶段，不再被门槛踩下去；
          · 等复合分涨到下一阶段时，门槛按**目标阶段**正常生效
            （亲密度在「过渡到下一阶段」时该起作用的时候起作用）；
          · 分数驱动的降级不受影响（滞后判定仍会把阶段降回去）。
        """
        saved = state._previous_stage
        if isinstance(saved, str) and saved in STAGE_ORDER:
            return saved
        raw_stage = cls._raw_stage_by_score(current_composite)
        reconstructed = STAGE_ORDER[max(0, STAGE_ORDER.index(raw_stage) - 1)]

        # 存档阶段（显示名 → key）：负向三档与脏值都不在 STAGE_ORDER 里，
        # 自然被跳过，不会影响正向阶段的归档
        archived_key = stage_key_from_name(state.relationship_stage)
        if archived_key in STAGE_ORDER and \
                STAGE_ORDER.index(archived_key) > STAGE_ORDER.index(reconstructed):
            return archived_key
        return reconstructed

    @classmethod
    def _get_stage_by_score(cls, composite_score: float, state: EnhancedEmotionalState) -> str:
        """滞后版阶段判定：上升阈值 > 下降阈值，防止抖动

        v4.1.1：历史基线统一走 `_resolve_previous_stage`。旧实现直接
        `state._previous_stage or "INITIAL"`，除了把缺失基线猜成初识期，
        还会在存档里混进脏字符串时让 `STAGE_ORDER.index(prev_stage)`
        抛 ValueError —— 而脏存档恰恰是会走到这里的场景。
        """
        prev_stage = cls._resolve_previous_stage(state, composite_score)

        # 计算当前"裸"阶段
        raw_target = cls._raw_stage_by_score(composite_score)

        # 滞后逻辑
        UP_THRESHOLD = cls.STAGE_CONFIGS[raw_target]["composite_threshold"]
        DOWN_THRESHOLD = UP_THRESHOLD - 5  # 5 点滞后带

        # 如果比上一阶段高，用上升阈值；否则用下降阈值
        use_threshold = UP_THRESHOLD if STAGE_ORDER.index(raw_target) > STAGE_ORDER.index(prev_stage) else DOWN_THRESHOLD

        if composite_score < use_threshold:
            return prev_stage

        return raw_target
    
    @classmethod
    def _check_transition_status(cls, state: EnhancedEmotionalState, previous_stage: str, 
                               target_stage: str, previous_composite: float, 
                               current_composite: float, gate_blocked: bool = False,
                               gate_target: Optional[str] = None) -> Dict[str, Any]:
        """检查过渡状态并应用保护机制"""
        transition_info = {
            "is_transitioning": False,
            "from_stage": previous_stage,
            "to_stage": target_stage,
            "protected_composite": current_composite,
            "intimacy_boost_active": False,
            "transition_progress": 0.0,
            "needed_intimacy_boost": 0,
            # v4.0.22：亲密度门槛阻断的过渡（分数够、亲密度没够）
            "intimacy_gate_blocked": gate_blocked,
            "intimacy_gate": None
        }

        if previous_stage != target_stage or gate_blocked:
            transition_info["is_transitioning"] = True
            
            # 应用复合评分保护：不低于前一阶段的最高评分
            protected_score = max(current_composite, previous_composite)
            transition_info["protected_composite"] = protected_score
            
            # 计算需要的亲密度提升。
            # v4.0.22：门槛阻断时 target_stage 已被压回当前阶段，增益必须按
            # 「被挡住的阶段」配置算，否则过渡加成会按错的权重生效。
            boost_target_key = gate_target or target_stage
            target_config = cls.STAGE_CONFIGS[boost_target_key]
            needed_intimacy = cls._calculate_needed_intimacy(state, target_config, protected_score)
            transition_info["needed_intimacy_boost"] = needed_intimacy
            transition_info["intimacy_boost_active"] = needed_intimacy > 0
            
            # 计算过渡进度
            transition_info["transition_progress"] = cls._calculate_transition_progress(
                state, target_config, needed_intimacy
            )

            if gate_blocked:
                # 门槛阻断：把「还差多少亲密度」并入过渡信息，
                # 供面板 / 建议文案 / 命令展示使用
                # v4.0.23：required 必须按**被挡住的目标阶段**那档算
                gate = dict(cls.get_intimacy_gate(state, gate_target))
                gate["to_stage"] = get_stage_name(gate_target) if gate_target else None
                # 被挡住阶段的英文 key：apply_transition_benefits 取增益系数时
                # 必须用它，不能用在压回当前阶段的 target_stage
                gate["to_stage_key"] = gate_target
                transition_info["intimacy_gate"] = gate
                transition_info["needed_intimacy_boost"] = max(
                    needed_intimacy, gate["gap"]
                )
                transition_info["intimacy_boost_active"] = True

        return transition_info
    
    @classmethod
    def _calculate_needed_intimacy(cls, state: EnhancedEmotionalState, target_config: Dict[str, Any], 
                                 protected_score: float) -> int:
        """计算达到目标阶段所需的最小亲密度"""
        fav_weight = target_config["favor_weight"]
        int_weight = target_config["intimacy_weight"]
        
        if int_weight == 0:
            return 0
            
        needed_intimacy = (protected_score - state.favor * fav_weight) / int_weight
        needed_intimacy = max(0, needed_intimacy)
        needed_intimacy = min(100, needed_intimacy)
        
        needed_intimacy = int(needed_intimacy)
        current_intimacy = state.intimacy
        
        return max(0, needed_intimacy - current_intimacy)
    
    @classmethod
    def _calculate_transition_progress(cls, state: EnhancedEmotionalState, target_config: Dict[str, Any],
                                    needed_intimacy: int) -> float:
        """计算过渡进度"""
        if needed_intimacy <= 0:
            return 100.0
            
        target_intimacy = state.intimacy + needed_intimacy
        current_progress = (state.intimacy / target_intimacy) * 100 if target_intimacy > 0 else 0
        return min(100.0, current_progress)
    
    @classmethod
    def get_stage_weights(cls, state: EnhancedEmotionalState) -> Tuple[float, float]:
        """获取当前阶段的权重（考虑过渡期）"""
        if state.favor < 0:
            return 1.0, 0.0
        
        target_stage, transition_info = cls.calculate_stage(state)
        stage_config = cls.STAGE_CONFIGS[target_stage]
        
        if transition_info["intimacy_boost_active"]:
            boost_factor = stage_config["intimacy_boost_factor"]
            base_favor = stage_config["favor_weight"]
            base_intimacy = stage_config["intimacy_weight"]
            
            total = base_favor + base_intimacy * boost_factor
            adjusted_favor = base_favor / total
            adjusted_intimacy = (base_intimacy * boost_factor) / total
            
            return adjusted_favor, adjusted_intimacy
        
        return stage_config["favor_weight"], stage_config["intimacy_weight"]
    
    @classmethod
    def calculate_composite_score(cls, state: EnhancedEmotionalState) -> float:
        """计算当前阶段的复合评分（应用过渡保护）"""
        if state.favor < 0:
            return state.favor
        
        target_stage, transition_info = cls.calculate_stage(state)
        return transition_info["protected_composite"]
    
    @classmethod
    def get_stage_info(cls, state: EnhancedEmotionalState) -> Dict[str, Any]:
        """获取完整的阶段信息（包含过渡状态）"""
        if state.favor < 0:
            return cls._get_negative_favor_stage_info(state)
        
        target_stage, transition_info = cls.calculate_stage(state)
        stage_config = cls.STAGE_CONFIGS[target_stage]
        
        favor_weight, intimacy_weight = cls.get_stage_weights(state)
        composite_score = cls.calculate_composite_score(state)
        
        progress = (composite_score / stage_config["composite_threshold"]) * 100
        progress_to_next = max(0, min(100, progress))

        # 计算"下一阶段"信息
        next_key = _next_stage_key(target_stage)
        if next_key is not None:
            next_stage_threshold = cls.STAGE_CONFIGS[next_key]["composite_threshold"]
            next_stage_name = get_stage_name(next_key)
        else:
            next_stage_threshold = None
            next_stage_name = "已达最高阶段"
        is_max_stage = target_stage == "SYMBIOSIS"

        info = {
            "stage": target_stage,
            "stage_name": get_stage_name(target_stage),
            "description": stage_config["description"],
            "favor_weight": favor_weight,
            "intimacy_weight": intimacy_weight,
            "composite_score": composite_score,
            "current_stage_threshold": stage_config["composite_threshold"],
            "next_stage_threshold": next_stage_threshold,
            "next_stage_name": next_stage_name,
            "is_max_stage": is_max_stage,
            "progress_to_next": progress_to_next,
            "is_transitioning": transition_info["is_transitioning"],
            "transition_progress": transition_info["transition_progress"],
            "intimacy_boost_active": transition_info["intimacy_boost_active"],
            "needed_intimacy_boost": transition_info["needed_intimacy_boost"],
            # v4.0.22：亲密度门槛（面板展示「还差多少」用）
            "intimacy_gate_blocked": transition_info["intimacy_gate_blocked"],
            "intimacy_gate": transition_info["intimacy_gate"]
        }
        
        # 保存当前状态用于下一次计算
        #
        # v4.0.22：门槛阻断时**不推进基线**。_previous_stage 保持旧阶段，
        # 下一轮 calculate_stage 会再次算出同一处阻断，过渡状态持续到
        # 亲密度达标；一旦达标即走正常的一次性过渡并落盘新基线。
        if not transition_info["intimacy_gate_blocked"]:
            state._previous_stage = target_stage
            state._previous_composite = composite_score

        return info
    
    @classmethod
    def _get_negative_favor_stage_info(cls, state: EnhancedEmotionalState) -> Dict[str, Any]:
        """获取负好感时的阶段信息

        v4.0.21：负向三档补齐英文 key（COLD / AVERSION / HOSTILITY），
        显示名走 stage_names.get_stage_name()，与正向阶段同一套自定义机制。
        """
        composite_score = state.favor

        if state.favor >= -30:
            stage_key = "COLD"
            description = "关系冷淡，需要修复"
            progress = max(0, (state.favor + 30) / 30 * 100)
        elif state.favor >= -70:
            stage_key = "AVERSION"
            description = "存在反感情绪"
            progress = max(0, (state.favor + 70) / 40 * 100)
        else:
            stage_key = "HOSTILITY"
            description = "关系敌对"
            progress = 0

        return {
            "stage": None,
            "stage_name": get_stage_name(stage_key),
            "description": description,
            "favor_weight": 1.0,
            "intimacy_weight": 0.0,
            "composite_score": composite_score,
            "current_stage_threshold": 0,
            "next_stage_threshold": None,
            "next_stage_name": "恢复正常关系",
            "is_max_stage": False,
            "progress_to_next": progress,
            "is_transitioning": False,
            "transition_progress": 0.0,
            "intimacy_boost_active": False,
            "needed_intimacy_boost": 0,
            # v4.0.22：负好感不走阶段门禁，两个键给默认值保持下游一致
            "intimacy_gate_blocked": False,
            "intimacy_gate": None
        }
    
    @classmethod
    def apply_transition_benefits(cls, state: EnhancedEmotionalState, updates: Dict[str, Any]) -> Dict[str, Any]:
        """应用过渡期增益效果"""
        target_stage, transition_info = cls.calculate_stage(state)
        
        if transition_info["intimacy_boost_active"]:
            # v4.0.22：门槛阻断时 target_stage 已被压回当前阶段，增益系数必须按
            # 「被挡住的阶段」取（与 _check_transition_status 里的
            # boost_target_key 同一口径），否则显示的「需要提升多少」与实际
            # 生效的倍数不是同一套配置。
            gate = transition_info.get("intimacy_gate") or {}
            boost_key = gate.get("to_stage_key") or target_stage
            stage_config = cls.STAGE_CONFIGS[boost_key]
            boost_factor = stage_config["intimacy_boost_factor"]
            
            if 'intimacy' in updates:
                original_boost = updates['intimacy']
                boosted_boost = int(original_boost * boost_factor)
                updates['intimacy'] = boosted_boost
            
            if ('joy' in updates or 'trust' in updates or 'anticipation' in updates) and 'intimacy' not in updates:
                auto_intimacy = max(1, int(2 * boost_factor))
                updates['intimacy'] = updates.get('intimacy', 0) + auto_intimacy
        
        return updates
    
    @classmethod
    def get_stage_progression_advice(cls, state: EnhancedEmotionalState) -> str:
        """获取阶段进阶建议（包含过渡期建议）"""
        stage_info = cls.get_stage_info(state)
    
        if state.favor < 0:
            if state.favor >= -30:
                return (f"{get_stage_name('COLD')}：需要真诚道歉和积极行动来修复关系，"
                        f"避免进一步恶化。")
            elif state.favor >= -70:
                return (f"{get_stage_name('AVERSION')}：需要时间和耐心来缓解负面情绪，"
                        f"避免直接冲突。")
            else:
                return (f"{get_stage_name('HOSTILITY')}：关系极度紧张，"
                        f"需要保持距离或寻求第三方调解。")
    
        if stage_info["is_transitioning"]:
            gate = stage_info.get("intimacy_gate")
            if gate and not gate["met"]:
                target_display = gate.get("to_stage") or "下一阶段"
                return (f"【阶段过渡中】{stage_info['stage_name']} → {target_display}\n"
                        f"   亲密度还未达标：{gate['current']}/{gate['required']}"
                        f"（还差 {gate['gap']} 点）\n"
                        f"   达标前好感度不会变化；多进行深度交流、保持连续互动可以提升亲密度")
            if stage_info["intimacy_boost_active"]:
                return (f"【阶段过渡中】{stage_info['stage_name']}\n"
                        f"   当前需要提升亲密度 {stage_info['needed_intimacy_boost']} 点来适应新阶段\n"
                       f"   过渡进度: {stage_info['transition_progress']:.1f}%\n"
                       f"   建议: 多进行深度交流，分享个人经历和情感")
            else:
                return (f"【阶段过渡完成】{stage_info['stage_name']}\n"
                       f"   已成功进入新阶段，关系正在稳定发展")
    
        # 建议文案以「当前生效的阶段名」开头（用户在配置界面改过名时同步变化）
        advice_map = {
            "INITIAL":
                f"{get_stage_name('INITIAL')}：多展示个人魅力，建立良好第一印象。通过有趣的话题和积极的互动提升好感度。",
            "DEEPENING":
                f"{get_stage_name('DEEPENING')}：分享更多个人经历和情感，建立信任基础。共同经历和深度交流是关键。",
            "COMMITMENT":
                f"{get_stage_name('COMMITMENT')}：巩固信任和默契，在困难时刻相互支持。关系的深度比广度更重要。",
            "SYMBIOSIS":
                f"{get_stage_name('SYMBIOSIS')}：维持情感的深度连接，共同成长和创造美好回忆。"
        }
    
        return advice_map.get(stage_info["stage"], "继续培养这段关系吧！")