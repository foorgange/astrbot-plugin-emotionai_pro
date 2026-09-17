# constants.py
"""常量定义"""
from enum import Enum

# 时间常量（秒）
class TimeConstants:
    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    THIRTY_MINUTES = 1800
    ONE_HOUR = 3600
    ONE_DAY = 86400

# 缓存常量
class CacheConstants:
    DEFAULT_TTL = 300
    MAX_SIZE = 1000
    SHARD_COUNT = 8  # 缓存分片数

# 情感常量
class EmotionConstants:
    """情感数值边界

    ⚠️ 为什么这四个值要写成「可被配置覆盖」而不是直接写死（v4.0.19 修正）

    它们在 `_conf_schema.json` / `PluginConfig` 里都有对应的用户配置项
    （favour_min / favour_max / intimacy_min / intimacy_max）。但旧版本在这里
    又硬编码了一份 100，于是 `models.EnhancedEmotionalState._validate_core_values`
    会把用户设的 200 直接削成 100：

        好感度设置成功 -> 回显 200 -> 实际存档/显示 100

    更隐蔽的是复合评分一起被压低：`200*0.5 + 100*0.5 = 150` 变成
    `100*0.5 + 100*0.5 = 100`，用户看到「复合评分：100.0」却找不到原因
    （阶段进度那个 100% 是另一回事，它本来就是百分比，天然封顶）。

    所以这里只保留「出厂默认值」，真正的边界由 `configure()` 从插件配置注入。
    默认值与旧版完全一致，未调用 configure() 时行为不变。
    """
    # 出厂默认（与 PluginConfig 的字段默认值保持同步）
    MIN_FAVOR = -100
    MAX_FAVOR = 100
    MIN_INTIMACY = 0
    MAX_INTIMACY = 100
    MIN_EMOTION = 0
    MAX_EMOTION = 100

    # 量级保护：即便用户配置缺失/异常，也绝不接受超出此范围的值，
    # 避免数值被无限拉大后把复合评分、排行榜等下游计算带偏。
    ABSOLUTE_LIMIT = 1000

    @classmethod
    def configure(
        cls,
        favour_min: int = None,
        favour_max: int = None,
        intimacy_min: int = None,
        intimacy_max: int = None,
    ) -> None:
        """用插件配置覆盖数值边界（插件启动/配置热重载时调用）

        策略：**整对一起用，或整对一起弃**。
        下限和上限互为约束（必须 min < max），单独判其中一个都可能出现
        「下限合法、上限非法」这种半生效状态。所以先各自做基础合法性检查
        （是数字、在量级内），只有两个都合法且 min < max 时才整体采用；
        否则这一对全部保持原值。

        ⚠️ 为什么不逐个判：曾写成「上限只要求 > 下限」，
        当上限=-5、下限=-100 时 `-5 > -100` 成立，-5 就被当成合法上限，
        于是所有好感度被钳到 -5。整对采用可以杜绝这类半生效。
        """
        def _clean(value, fallback: int):
            """基础合法性检查：返回 (是否可用, 数值)"""
            if value is None or isinstance(value, bool):
                return False, fallback
            try:
                number = int(value)
            except (TypeError, ValueError):
                return False, fallback
            if abs(number) > cls.ABSOLUTE_LIMIT:
                return False, fallback
            return True, number

        f_min_ok, f_min = _clean(favour_min, cls.MIN_FAVOR)
        f_max_ok, f_max = _clean(favour_max, cls.MAX_FAVOR)
        if f_min_ok and f_max_ok and f_min < f_max:
            cls.MIN_FAVOR, cls.MAX_FAVOR = f_min, f_max

        i_min_ok, i_min = _clean(intimacy_min, cls.MIN_INTIMACY)
        i_max_ok, i_max = _clean(intimacy_max, cls.MAX_INTIMACY)
        if i_min_ok and i_max_ok and i_min < i_max:
            cls.MIN_INTIMACY, cls.MAX_INTIMACY = i_min, i_max

    @classmethod
    def reset(cls) -> None:
        """恢复出厂默认边界（测试用）"""
        cls.MIN_FAVOR = -100
        cls.MAX_FAVOR = 100
        cls.MIN_INTIMACY = 0
        cls.MAX_INTIMACY = 100

# 更新阈值
class UpdateThresholds:
    MINOR_CHANGE = 3
    MAJOR_CHANGE = 8
    FORCE_UPDATE = 5
    EMOTIONAL_SIGNIFICANCE = 5

# 文件路径
class PathConstants:
    USER_DATA_FILE = "user_emotion_data.json"
    LONG_TERM_MEMORY_FILE = "long_term_memory.json"
    GLOBAL_MOOD_FILE = "global_mood.json"
    BACKUP_DIR = "backups"
    TEMP_DIR = "temp"