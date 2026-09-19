# stage_names.py
# -*- coding: utf-8 -*-
"""关系阶段显示名的单一来源（v4.0.21：支持用户自定义阶段名）

## 为什么需要这个模块

阶段系统有两套标识，历史上显示名散布在五六个文件里：

- **内部逻辑**永远用英文 key（INITIAL / DEEPENING / COMMITMENT / SYMBIOSIS），
  惰性判定、权重、阈值全部按 key 索引；
- **存档与界面**用中文显示名（`models.relationship_stage` 存的就是它）。

负向三档（冷淡期 / 反感期 / 敌对期）原来没有 key，是按 favor 阈值现场取名的。
v4.0.21 起把它们补齐为 COLD / AVERSION / HOSTILITY，与正向阶段共用同一套
「key → 显示名」机制，于是**所有阶段名都可以在配置界面自定义**。

## 两个必须小心的坑

1. **改名前存下的旧存档**：用户把「初识期」改成「初见」后，存档里仍是旧名
   「初识期」。若直接拿旧名去比对生效名单，会被判定无效 → 按 favor 修复
   → 用户的关系阶段被静默重置。所以 `normalize_stage_name()` 负责把
   **出厂默认名**映射回当前生效名，改名对旧存档立即生效而不是清零。
2. **重名**：两个阶段配成同一个名字会让面板与建议文案自相矛盾。
   `configure_stage_names()` 发现重名时**跳过后者、保留前者**并记日志，
   绝不生成歧义状态。
"""
from typing import Any, Dict, List, Optional

# 全部阶段 key（正向四个 + 负向三档）
STAGE_KEYS = (
    "INITIAL", "DEEPENING", "COMMITMENT", "SYMBIOSIS",
    "COLD", "AVERSION", "HOSTILITY",
)

# 出厂默认显示名（同时也是「旧存档识别表」——改名前的历史名）
DEFAULT_STAGE_NAMES: Dict[str, str] = {
    "INITIAL": "初识期",
    "DEEPENING": "深化期",
    "COMMITMENT": "承诺期",
    "SYMBIOSIS": "共生期",
    "COLD": "冷淡期",
    "AVERSION": "反感期",
    "HOSTILITY": "敌对期",
}

# 出厂默认名 → key（用于把旧存档的名字归一到当前生效名）
LEGACY_NAME_TO_KEY: Dict[str, str] = {
    name: key for key, name in DEFAULT_STAGE_NAMES.items()
}

# 当前生效的显示名（进程内单例；启动与热重载时由 configure_stage_names 刷新）
_current_names: Dict[str, str] = dict(DEFAULT_STAGE_NAMES)

# 阶段名长度上限（面板显示用途，超长直接忽略该配置项）
MAX_STAGE_NAME_LEN = 20


def _resolve_key(raw_key: Any) -> Optional[str]:
    """把配置里的键解析成阶段 key

    接受两种写法：英文 key（`INITIAL`）或出厂默认中文名（`初识期`）——
    `_conf_schema.json` 的嵌套对象用**默认中文名**作键（对用户最直观），
    同时在代码层面兼容英文 key，两者都能用。
    """
    if isinstance(raw_key, str):
        key = raw_key.strip()
        if key in DEFAULT_STAGE_NAMES:
            return key
        if key in LEGACY_NAME_TO_KEY:
            return LEGACY_NAME_TO_KEY[key]
    return None


def configure_stage_names(raw: Any) -> List[str]:
    """应用配置里的自定义阶段名，返回实际发生的改名列表

    `raw` 是 `_conf_schema.json` 中 `stage_names`（type=object 嵌套配置）
    对应的 dict。残缺/非法项**逐项忽略**，绝不让一个坏值影响其它阶段。
    """
    global _current_names

    mapping = dict(DEFAULT_STAGE_NAMES)
    used = set(mapping.values())
    applied: List[str] = []

    if isinstance(raw, dict):
        for raw_key, value in raw.items():
            key = _resolve_key(raw_key)
            if key is None:
                continue
            if not isinstance(value, str):
                continue
            name = value.strip()
            if not name or len(name) > MAX_STAGE_NAME_LEN:
                continue
            if name in used:
                # 与其它阶段重名：跳过，保留先到的那个（含出厂默认名）
                continue
            if mapping[key] != name:
                mapping[key] = name
                used.add(name)
                applied.append(f"{DEFAULT_STAGE_NAMES[key]}→{name}")

    _current_names = mapping
    return applied


def reset_stage_names() -> None:
    """恢复出厂默认名（测试用）"""
    global _current_names
    _current_names = dict(DEFAULT_STAGE_NAMES)


def get_stage_name(key: str) -> str:
    """取某个阶段的当前生效显示名（未知 key 安全回退）"""
    return _current_names.get(key, DEFAULT_STAGE_NAMES.get(key, key))


def stage_name_map() -> Dict[str, str]:
    """当前生效的 key→显示名 副本"""
    return dict(_current_names)


def valid_stage_names() -> List[str]:
    """当前生效的全部合法阶段显示名（供存档校验）"""
    return list(_current_names.values())


def normalize_stage_name(name: Any) -> Optional[str]:
    """把存档里的阶段名归一到「当前生效名」

    - 已是当前生效名 → 原样返回
    - 是出厂默认名（用户改过名，存档还是旧名）→ 返回该 key 的当前生效名
    - 其它（含 None / 非字符串 / 空串）→ None，由调用方按无效值处理
    """
    if not isinstance(name, str) or not name:
        return None
    name = name.strip()
    if not name:
        return None
    if name in _current_names.values():
        return name
    key = LEGACY_NAME_TO_KEY.get(name)
    if key is not None:
        return get_stage_name(key)
    return None
