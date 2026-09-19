# config.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class PrivacyLevel(int, Enum):
    FULL_SECRET = 0
    BASIC = 1  
    DETAILED = 2

class AttitudeType(str, Enum):
    HOSTILE = "敌对"
    COLD = "冷淡" 
    NEUTRAL = "中立"
    FRIENDLY = "友好"
    INTIMATE = "热情"

class RelationshipStage(str, Enum):
    INITIAL = "初识期"
    DEEPENING = "深化期"
    COMMITMENT = "承诺期" 
    SYMBIOSIS = "共生期"

class PluginConfig(BaseModel):
    """插件配置模型"""
    #
    # ⚠️ 数值区间的设计原则（v4.0.18 修正）
    #
    # 这四个「上下限」是**运行时钳制边界**，不是"上限必须为正 / 下限必须为负"。
    # 配置界面（_conf_schema.json）对它们没有任何取值约束，用户完全可以填出
    # intimacy_min=-100（亲密度允许为负）、change_min=3 / change_max=2
    # 这类组合。旧版本把它们写成 ge=0 / le=0，于是 pydantic 抛 ValidationError，
    # 而 main.py 的兜底是 `return PluginConfig()` —— **静默丢掉整份配置**
    # （连带 admin_qq_list 变空，管理员命令全部提示「权限不足」）。
    #
    # 所以这里的约束只保证：① 取值在合理量级内 ② min < max 由
    # ConfigManager._validate_config / ConfigValidator 单独检查，
    # 不再由 pydantic 用"符号"去猜用户意图。
    session_based: bool = Field(default=False, description="是否启用会话独立的情感系统")
    favour_min: int = Field(default=-100, ge=-1000, le=1000, description="好感度最小值")
    favour_max: int = Field(default=100, ge=-1000, le=1000, description="好感度最大值")
    intimacy_min: int = Field(default=0, ge=-1000, le=1000, description="亲密度最小值")
    intimacy_max: int = Field(default=100, ge=-1000, le=1000, description="亲密度最大值")
    change_min: int = Field(default=-10, ge=-1000, le=1000, description="好感度单次变化最小值")
    change_max: int = Field(default=5, ge=-1000, le=1000, description="好感度单次变化最大值")
    # 亲密度的单次变化幅度（v4.0.21 新增）
    #
    # v4.0.21 之前这两个值是 emotion_expert.py 里写死的 ±5，用户无法约束。
    # 默认 ±3（比原来的 ±5 更保守）：一次性亲密关系跳跃变小，
    # 关系推进更自然，也和「好感度单次变化」的默认量级一致。
    intimacy_change_min: int = Field(default=-3, ge=-1000, le=1000, description="亲密度单次减少的最大幅度")
    intimacy_change_max: int = Field(default=3, ge=-1000, le=1000, description="亲密度单次增加的最大幅度")
    # 阶段过渡亲密度门槛 + 亲密度里程碑（v4.0.22）
    #
    # 亲密度从「每轮小幅波动」升级为「阶段过渡的硬门槛 + 里程碑加成」：
    # ① transition_intimacy_pct：复合评分达到下一阶段阈值只是必要条件，
    #    亲密度还必须达到「亲密度上限 × 该百分比」才能完成过渡；未达标
    #    期间过渡卡住、好感度一并冻结（见 relationship_manager）。
    # ② 首次深度交流 / 连续多日互动两类里程碑额外加成亲密度，让它不必
    #    只靠 LLM 每轮打分缓慢积累。填 0 即关闭对应加成。
    transition_intimacy_pct: int = Field(default=50, ge=0, le=100, description="阶段过渡所需亲密度占最大亲密度的百分比(0=关闭门槛)")
    # 分阶段亲密度门槛（v4.0.23）
    #
    # key 是**目标阶段英文 key**（DEEPENING/COMMITMENT/SYMBIOSIS），值是
    # 占最大亲密度的百分比。只有一个统一值时，第二、三次过渡的门槛形同虚设
    # （承诺期权重 favor 0.3/intimacy 0.7，复合分上 80 本身就要求亲密度
    # 很高），所以按目标阶段分别设，且仍跟随「最大亲密度」换算。
    # 查表顺序：本表 → 未列出的阶段回退 transition_intimacy_pct。
    stage_intimacy_gates: Dict[str, int] = Field(
        default_factory=lambda: {"DEEPENING": 20, "COMMITMENT": 40, "SYMBIOSIS": 60},
        description="分阶段过渡亲密度门槛(占最大亲密度的百分比，按目标阶段分别设置)",
    )
    intimacy_first_deep_bonus: int = Field(default=3, ge=0, le=100, description="首次深度交流的亲密度奖励")
    intimacy_streak_days: int = Field(default=3, ge=2, le=30, description="连续互动多少天开始获得亲密度加成")
    intimacy_streak_bonus: int = Field(default=1, ge=0, le=100, description="连续互动达标后每日首次互动的亲密度加成")
    # 会话边界上下文保鲜（v4.1.0）
    #
    # 框架默认每轮携带最近 50 条对话历史（max_context_length）。间隔较长的
    # 两次聊天之间，旧话题会一直留在上下文里，导致 bot「忽然接上很久之前
    # 的话题」。这里按会话间隔裁剪：距上次活跃超过该分钟数时，本轮请求
    # 丢弃 provider 可见的历史消息（对话记录仍由框架完整保留在数据库中，
    # 只是本轮不注入），bot 只基于当前会话 + 情感状态回应。
    # 填 0 关闭保鲜，恢复框架默认行为。
    session_gap_minutes: int = Field(default=60, ge=0, le=1440, description="会话间隔超过多少分钟则丢弃旧对话历史(0=关闭)")
    # 关系阶段显示名自定义（v4.0.21）
    #
    # 键是**出厂默认阶段名**（初识期/深化期/承诺期/共生期/冷淡期/反感期/敌对期，
    # 代码层同时兼容英文 key INITIAL/DEEPENING/...），值是用户想要的名字。
    # 空 dict = 全部用默认名。阶段判定逻辑与阈值完全不受影响 —— 只改显示。
    stage_names: Dict[str, str] = Field(default_factory=dict, description="关系阶段名称自定义")
    admin_qq_list: List[str] = Field(default_factory=list, description="管理员QQ号列表")
    plugin_priority: int = Field(default=100000, ge=1, le=1000000, description="插件处理优先级")
    enable_attitude_system: bool = Field(default=True, description="启用态度关系系统")
    enable_ai_text_generation: bool = Field(default=True, description="启用AI自主生成文本描述")
    global_privacy_level: PrivacyLevel = Field(default=PrivacyLevel.BASIC, description="全局隐私级别")
    enable_smart_update: bool = Field(default=True, description="启用智能更新机制")
    force_update_interval: int = Field(default=5, ge=1, le=100, description="强制更新间隔（对话次数）")
    emotional_significance_threshold: int = Field(default=5, ge=1, le=10, description="情感意义阈值")
    enable_secondary_llm: bool = Field(default=True, description="启用辅助LLM进行情感分析")
    secondary_llm_provider: Optional[str] = Field(default=None, description="辅助LLM提供商")
    secondary_llm_model: Optional[str] = Field(default=None, description="辅助LLM模型名称")
    emotion_llm_time_budget: float = Field(default=45.0, ge=10.0, le=300.0, description="情感分析总时间预算(秒)，预算耗尽即放弃LLM分析，降级为本地兜底")
    emotion_llm_max_providers: int = Field(default=3, ge=1, le=10, description="情感分析预算内最多尝试的provider个数")
    bot_name: Optional[str] = Field(default=None, description="机器人人设名称，留空则首次启动自动从 AstrBot persona 提取")
    
    # 性能配置
    cache_ttl: int = Field(default=300, description="缓存默认TTL(秒)")
    cache_max_size: int = Field(default=1000, description="缓存最大条目数")
    auto_save_interval: int = Field(default=60, description="自动保存间隔(秒)")
    max_dirty_keys: int = Field(default=1000, description="最大脏键数量")
    backup_retention_days: int = Field(default=7, description="备份保留天数")

    class Config:
        use_enum_values = True