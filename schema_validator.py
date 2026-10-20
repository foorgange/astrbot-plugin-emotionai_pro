# schema_validator.py
"""
配置文件验证器
"""
import json
from pathlib import Path
from typing import Dict, Any
from jsonschema import validate, ValidationError

from astrbot.api import logger

class ConfigValidator:
    """配置文件验证器"""
    
    # 配置架构定义
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "session_based": {"type": "boolean"},
            # 数值区间只做量级约束（与 config.py 的 PluginConfig 保持一致）。
            # 不要写成"下限必须 ≤0 / 上限必须 ≥0"——那会把用户合理填出的
            # intimacy_min=-100、change_min=3 判为非法，进而整份配置被丢弃。
            # min < max 的关系由下方 extra 校验单独负责。
            "favour_min": {"type": "integer", "minimum": -1000, "maximum": 1000},
            "favour_max": {"type": "integer", "minimum": -1000, "maximum": 1000},
            "intimacy_min": {"type": "integer", "minimum": -1000, "maximum": 1000},
            "intimacy_max": {"type": "integer", "minimum": -1000, "maximum": 1000},
            "change_min": {"type": "integer", "minimum": -1000, "maximum": 1000},
            "change_max": {"type": "integer", "minimum": -1000, "maximum": 1000},
            # v4.0.21：亲密度的单次变化幅度（此前写死 ±5）
            "intimacy_change_min": {"type": "integer", "minimum": -1000, "maximum": 1000},
            "intimacy_change_max": {"type": "integer", "minimum": -1000, "maximum": 1000},
            # v4.0.22：阶段过渡亲密度门槛 + 亲密度里程碑
            "transition_intimacy_pct": {"type": "integer", "minimum": 0, "maximum": 100},
            # v4.0.23：分阶段亲密度门槛（key=目标阶段英文 key，值=百分比）
            "stage_intimacy_gates": {
                "type": "object",
                "additionalProperties": {"type": "integer", "minimum": 0, "maximum": 100}
            },
            "intimacy_first_deep_bonus": {"type": "integer", "minimum": 0, "maximum": 100},
            "intimacy_streak_days": {"type": "integer", "minimum": 2, "maximum": 30},
            "intimacy_streak_bonus": {"type": "integer", "minimum": 0, "maximum": 100},
            # v4.1.0：会话边界上下文保鲜（分钟，0=关闭）
            "session_gap_minutes": {"type": "integer", "minimum": 0, "maximum": 1440},
            # v4.0.21：关系阶段显示名自定义（嵌套 object，键为出厂默认名/英文 key）
            "stage_names": {
                "type": "object",
                "additionalProperties": {"type": "string"}
            },
            "admin_qq_list": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[0-9]{5,12}$"}
            },
            "plugin_priority": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "enable_attitude_system": {"type": "boolean"},
            # v4.1.6：全局心情是否参与主对话回复生成
            "enable_mood_influence": {"type": "boolean"},
            # v4.2.0：心情影响强度 + 心情一致记忆召回 + 好感度驱动记忆召回
            "mood_influence_strength": {"type": "number", "minimum": 0, "maximum": 1},
            "mood_memory_bias": {"type": "boolean"},
            "enable_memory_recall": {"type": "boolean"},
            "memory_recall_max_items": {"type": "integer", "minimum": 1, "maximum": 10},
            # v4.1.8：心情变化原因是否由大模型生成（关闭走内置文案兜底）
            "enable_llm_mood_reason": {"type": "boolean"},
            # v4.1.7：图片格式守卫（拦截 SVG 等不支持的格式，避免
            # provider 连环报错导致回复变慢）
            "image_format_filter_enabled": {"type": "boolean"},
            "enable_ai_text_generation": {"type": "boolean"},
            "global_privacy_level": {"type": "integer", "minimum": 0, "maximum": 2},
            "enable_smart_update": {"type": "boolean"},
            "force_update_interval": {"type": "integer", "minimum": 1, "maximum": 100},
            "emotional_significance_threshold": {"type": "integer", "minimum": 1, "maximum": 10},
            "enable_secondary_llm": {"type": "boolean"},
            "secondary_llm_provider": {"type": "string"},
            "secondary_llm_model": {"type": "string"},
            # v4.2.3：辅助 LLM 退避链（有序 provider id 列表）。
            # 顶层 additionalProperties=False，漏了它会让带该键的
            # 配置文件在严格校验下被判非法。
            "secondary_llm_fallback_providers": {
                "type": "array",
                "items": {"type": "string"}
            },
            # v4.1.10：情感分析 LLM 的时间预算与备选个数（补漏）。
            # 顶层是 additionalProperties=False，漏掉这两个键会让
            # 含有它们的配置文件在严格校验下被判非法。
            "emotion_llm_time_budget": {"type": "number", "minimum": 10, "maximum": 300},
            "emotion_llm_max_providers": {"type": "integer", "minimum": 1, "maximum": 10},
            "bot_name": {"type": "string"},
            "cache_ttl": {"type": "integer", "minimum": 1},
            "cache_max_size": {"type": "integer", "minimum": 10},
            "auto_save_interval": {"type": "integer", "minimum": 1},
            "max_dirty_keys": {"type": "integer", "minimum": 10},
            "backup_retention_days": {"type": "integer", "minimum": 1}
        },
        "required": [
            "session_based", "favour_min", "favour_max", "intimacy_min", "intimacy_max",
            "change_min", "change_max", "admin_qq_list", "plugin_priority"
        ],
        "additionalProperties": False
    }
    
    @classmethod
    def validate_config_file(cls, config_path: Path) -> bool:
        """验证配置文件"""
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 验证架构
            validate(instance=config_data, schema=cls.CONFIG_SCHEMA)
            
            # 额外验证：确保最小值小于最大值
            if config_data['favour_min'] >= config_data['favour_max']:
                logger.error(f"favour_min ({config_data['favour_min']}) 必须小于 favour_max ({config_data['favour_max']})")
                return False
            
            if config_data['intimacy_min'] >= config_data['intimacy_max']:
                logger.error(f"intimacy_min ({config_data['intimacy_min']}) 必须小于 intimacy_max ({config_data['intimacy_max']})")
                return False
            
            if config_data['change_min'] >= config_data['change_max']:
                logger.error(f"change_min ({config_data['change_min']}) 必须小于 change_max ({config_data['change_max']})")
                return False

            # 亲密度的单次变化幅度对。用 .get 兜默认值：这两个字段是 v4.0.21
            # 新增的可选键，旧配置文件里可能还没有（AstrBot 重启时会按 schema
            # 默认值补写），不能因为键缺失就把整份配置判为非法。
            _ic_min = config_data.get('intimacy_change_min', -3)
            _ic_max = config_data.get('intimacy_change_max', 3)
            if _ic_min >= _ic_max:
                logger.error(f"intimacy_change_min ({_ic_min}) 必须小于 intimacy_change_max ({_ic_max})")
                return False
            
            logger.info("配置文件验证成功")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {e}")
            return False
        except ValidationError as e:
            logger.error(f"配置文件验证失败: {e.message}")
            return False
        except Exception as e:
            logger.error(f"配置文件验证过程中出错: {e}")
            return False
    
    @classmethod
    def create_default_config(cls, config_path: Path):
        """创建默认配置文件"""
        default_config = {
            "session_based": False,
            "favour_min": -100,
            "favour_max": 100,
            "intimacy_min": 0,
            "intimacy_max": 100,
            "change_min": -10,
            "change_max": 5,
            "intimacy_change_min": -3,
            "intimacy_change_max": 3,
            "transition_intimacy_pct": 50,
            "stage_intimacy_gates": {"DEEPENING": 20, "COMMITMENT": 40, "SYMBIOSIS": 60},
            "intimacy_first_deep_bonus": 3,
            "intimacy_streak_days": 3,
            "intimacy_streak_bonus": 1,
            "session_gap_minutes": 60,
            "stage_names": {},
            "admin_qq_list": [],
            "plugin_priority": 100000,
            "enable_attitude_system": True,
            "enable_mood_influence": True,
            "mood_influence_strength": 0.7,
            "mood_memory_bias": True,
            "enable_memory_recall": True,
            "memory_recall_max_items": 6,
            "enable_llm_mood_reason": True,
            "image_format_filter_enabled": True,
            "enable_ai_text_generation": True,
            "global_privacy_level": 1,
            "enable_smart_update": True,
            "force_update_interval": 5,
            "emotional_significance_threshold": 5,
            "enable_secondary_llm": True,
            "secondary_llm_provider": "",
            "secondary_llm_model": "",
            "secondary_llm_fallback_providers": [],
            "emotion_llm_time_budget": 45.0,
            "emotion_llm_max_providers": 3,
            "bot_name": "",
            "cache_ttl": 300,
            "cache_max_size": 1000,
            "auto_save_interval": 60,
            "max_dirty_keys": 1000,
            "backup_retention_days": 7
        }
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已创建默认配置文件: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建默认配置文件失败: {e}")
            return False