# config_manager.py
import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import hashlib

from pydantic import ValidationError

from astrbot.api import logger

from .config import PluginConfig
from .constants import EmotionConstants

class ConfigManager:
    """配置管理器 - 增强的热重载支持"""
    
    def __init__(self, config_path: Path, initial_config: PluginConfig):
        self.config_path = config_path
        self.current_config = initial_config
        self._listeners: List[Callable] = []
        self._watch_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._last_hash: Optional[str] = None
        self._last_modified: float = 0
        self._error_count = 0
        self._last_error_time = 0
        
        # 确保配置文件存在
        self._ensure_config_file()
        
        logger.info(f"配置管理器初始化完成，配置文件: {config_path}")
    
    def _ensure_config_file(self):
        """确保配置文件存在"""
        if not self.config_path.exists():
            logger.info(f"配置文件不存在，创建默认配置: {self.config_path}")
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_config(self.current_config)
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """计算配置哈希值"""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_config(self, config: PluginConfig):
        """保存配置到文件"""
        try:
            config_dict = config.dict()
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
            # 更新哈希值
            self._last_hash = self._calculate_config_hash(config_dict)
            self._last_modified = time.time()
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def add_change_listener(self, callback: Callable):
        """添加配置变更监听器"""
        if callback not in self._listeners:
            self._listeners.append(callback)
            logger.info(f"添加配置变更监听器: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def remove_change_listener(self, callback: Callable):
        """移除配置变更监听器"""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    async def start_watching(self, interval: float = 2.0):
        """开始监控配置文件变化"""
        if self._watch_task:
            logger.info("配置监控任务已在运行")
            return
        
        logger.info(f"开始监控配置文件变化，检查间隔: {interval}秒")
        
        async def watch_loop():
            while True:
                try:
                    await self._check_for_changes()
                    await asyncio.sleep(interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"配置监控循环错误: {e}")
                    self._error_count += 1
                    self._last_error_time = time.time()
                    
                    # 错误过多时暂停
                    if self._error_count > 10 and time.time() - self._last_error_time < 60:
                        logger.error("配置监控错误过多，暂停60秒")
                        await asyncio.sleep(60)
                    else:
                        await asyncio.sleep(5)
        
        self._watch_task = asyncio.create_task(watch_loop())
    
    async def _check_for_changes(self):
        """检查配置变化"""
        if not self.config_path.exists():
            logger.warning(f"配置文件不存在: {self.config_path}")
            return
        
        try:
            # 检查文件修改时间
            current_mtime = self.config_path.stat().st_mtime
            if current_mtime <= self._last_modified:
                return
            
            # 读取配置文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    logger.warning("配置文件为空")
                    return
                
                new_config_data = json.loads(content)
            
            # 计算新哈希
            new_hash = self._calculate_config_hash(new_config_data)
            
            # 如果哈希值相同，只是时间戳变化
            if new_hash == self._last_hash:
                self._last_modified = current_mtime
                return
            
            logger.info(f"检测到配置变化，重新加载配置")
            
            # 重新加载配置
            await self._reload_config(new_config_data)
            
            # 更新状态
            self._last_hash = new_hash
            self._last_modified = current_mtime
            self._error_count = 0
            
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {e}")
        except Exception as e:
            logger.error(f"检查配置变化失败: {e}")
            self._error_count += 1
    
    def _apply_numeric_bounds(self, config: PluginConfig) -> None:
        """把配置里的数值边界同步到状态模型

        与 main.py::__init__ 里的注入是同一件事，区别在于这里是热重载路径。
        不同步的后果：用户在配置界面把「好感度最大值」从 100 改成 200，
        配置确实更新了，但 `models._validate_core_values` 仍按旧的 100 钳制，
        表现就是"配置生效了、数值却写不上去"。
        """
        try:
            EmotionConstants.configure(
                favour_min=config.favour_min,
                favour_max=config.favour_max,
                intimacy_min=config.intimacy_min,
                intimacy_max=config.intimacy_max,
            )
        except Exception as e:  # noqa: BLE001
            # 边界同步失败不该影响配置更新本身，保持旧边界即可
            logger.warning(f"同步数值边界失败，保持原边界: {e}")

    async def _reload_config(self, new_config_data: Dict[str, Any]):
        """重新加载配置"""
        try:
            async with self._lock:
                old_config = self.current_config

                # 创建新配置实例
                # 与 main.py::_load_and_validate_config 同一策略：单个坏字段
                # 只回退该字段，不让整份配置（尤其是 admin_qq_list）失效。
                try:
                    new_config = PluginConfig(**new_config_data)
                except ValidationError as e:
                    bad = [
                        f"{(err.get('loc') or ('?',))[0]}="
                        f"{new_config_data.get((err.get('loc') or ('?',))[0], '<缺失>')!r}"
                        f"({err.get('type', 'unknown')})"
                        for err in e.errors()
                    ]
                    logger.warning(
                        "热重载配置校验失败，以下字段回退为默认值（其余照常生效）: "
                        + ", ".join(bad)
                    )
                    sanitized = dict(new_config_data)
                    for err in e.errors():
                        loc = err.get("loc") or ()
                        if loc and loc[0] in sanitized:
                            field = str(loc[0])
                            sanitized[field] = PluginConfig.model_fields[field].default
                    try:
                        new_config = PluginConfig(**sanitized)
                    except Exception as retry_error:  # noqa: BLE001
                        logger.error(f"配置热重载失败，保持当前配置: {retry_error}")
                        return
                
                # 验证新配置
                if not self._validate_config(new_config):
                    logger.error("新配置验证失败，保持当前配置")
                    return

                # 数值边界要跟着热重载一起变，否则用户改了「好感度最大值」
                # 后，状态模型仍按旧边界钳制，出现"配置已更新但数值不生效"。
                self._apply_numeric_bounds(new_config)

                # 通知监听器（在更新当前配置之前）
                logger.info(f"通知 {len(self._listeners)} 个监听器配置变更")
                
                listener_tasks = []
                for listener in self._listeners:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            task = asyncio.create_task(listener(old_config, new_config))
                            listener_tasks.append(task)
                        else:
                            # 同步函数在线程池中执行
                            loop = asyncio.get_event_loop()
                            task = loop.run_in_executor(None, listener, old_config, new_config)
                            listener_tasks.append(task)
                    except Exception as e:
                        logger.error(f"配置变更监听器调用失败: {e}")
                
                # 等待所有监听器完成
                if listener_tasks:
                    try:
                        await asyncio.gather(*listener_tasks, return_exceptions=True)
                    except Exception as e:
                        logger.error(f"等待监听器完成时出错: {e}")
                
                # 更新当前配置
                self.current_config = new_config
                
                logger.info("配置热重载完成")
                
                # 记录配置差异
                self._log_config_changes(old_config, new_config)
                
        except Exception as e:
            logger.error(f"配置重载失败: {e}")
            raise
    
    def _repair_invalid_pairs(self, config: PluginConfig) -> List[str]:
        """修复「上下限填反」的字段对，返回被修复的字段说明

        ⚠️ 为什么不再直接 `return False` 整份拒绝（v4.0.20 修正）：
        旧做法是 `_validate_config` 一旦发现 `change_min >= change_max` 就返回
        False，调用方随即「保持当前配置」或抛 ValueError —— 后果是用户**改了
        别的任何配置都保存不上**，而提示只有一行含糊的「新配置验证失败」。
        用户服务器上就真实存在 `change_min=3 / change_max=2` 这种组合
        （填反了符号），导致他的配置保存长期静默失败。

        现在的策略与 v4.0.18「逐字段回退」一致：**只把出问题的那一对退回默认
        值**，其余配置照常保存生效，并把修复内容明确打出来。

        返回被修复项的说明列表（空列表表示没有任何问题）。
        """
        repaired: List[str] = []

        def _fix_pair(low_field: str, high_field: str) -> None:
            low = getattr(config, low_field)
            high = getattr(config, high_field)
            if low < high:
                return
            default_low = PluginConfig.model_fields[low_field].default
            default_high = PluginConfig.model_fields[high_field].default
            setattr(config, low_field, default_low)
            setattr(config, high_field, default_high)
            repaired.append(
                f"{low_field}={low} >= {high_field}={high} "
                f"→ 已重置为 {default_low} / {default_high}"
            )

        _fix_pair("favour_min", "favour_max")
        _fix_pair("intimacy_min", "intimacy_max")
        _fix_pair("change_min", "change_max")

        if config.force_update_interval <= 0:
            default = PluginConfig.model_fields["force_update_interval"].default
            repaired.append(
                f"force_update_interval={config.force_update_interval} "
                f"→ 已重置为 {default}"
            )
            config.force_update_interval = default

        # 管理员列表里的非法项**逐个剔除**（而不是整份拒绝）
        bad_admins = [
            qq for qq in config.admin_qq_list
            if not isinstance(qq, str) or not qq.isdigit()
        ]
        if bad_admins:
            config.admin_qq_list = [
                qq for qq in config.admin_qq_list
                if isinstance(qq, str) and qq.isdigit()
            ]
            repaired.append(
                f"admin_qq_list 剔除非法项 {bad_admins}"
                f"（保留 {config.admin_qq_list}）"
            )

        return repaired

    def _validate_config(self, config: PluginConfig) -> bool:
        """验证配置的有效性

        v4.0.20 起：先**就地修复**非法的字段对，再复核。
        仍然返回 bool 以兼容既有调用方，但正常路径下几乎不会返回 False ——
        因为不该让一个填错的字段把整份配置（连带管理员列表）拖下水。
        """
        try:
            repaired = self._repair_invalid_pairs(config)
            if repaired:
                logger.warning(
                    "配置中存在非法取值，已按字段修复（其余配置照常生效）: "
                    + "; ".join(repaired)
                )
            return True

        except Exception as e:
            logger.error(f"配置验证过程中出错: {e}")
            return False

    def _log_config_changes(self, old_config: PluginConfig, new_config: PluginConfig):
        """记录配置变化"""
        changes = []
        
        old_dict = old_config.dict()
        new_dict = new_config.dict()
        
        for key in old_dict.keys():
            if key in new_dict and old_dict[key] != new_dict[key]:
                changes.append({
                    'key': key,
                    'old': old_dict[key],
                    'new': new_dict[key]
                })
        
        if changes:
            logger.info("配置变化详情:")
            for change in changes:
                logger.info(f"  {change['key']}: {change['old']} -> {change['new']}")
        else:
            logger.info("没有检测到配置值变化")
    
    async def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        async with self._lock:
            try:
                old_config = self.current_config
                config_dict = old_config.dict()
                
                # 应用更新
                config_dict.update(updates)
                
                # 创建新配置实例（同样逐字段回退，避免整体失败）
                try:
                    new_config = PluginConfig(**config_dict)
                except ValidationError as e:
                    bad = [
                        f"{(err.get('loc') or ('?',))[0]}"
                        f"({err.get('type', 'unknown')})"
                        for err in e.errors()
                    ]
                    logger.warning(
                        "更新后的配置校验失败，以下字段回退为默认值: " + ", ".join(bad)
                    )
                    sanitized = dict(config_dict)
                    for err in e.errors():
                        loc = err.get("loc") or ()
                        if loc and loc[0] in sanitized:
                            field = str(loc[0])
                            sanitized[field] = PluginConfig.model_fields[field].default
                    new_config = PluginConfig(**sanitized)
                
                # 验证新配置
                if not self._validate_config(new_config):
                    raise ValueError("新配置验证失败")
                
                # 与热重载同理：边界变更必须同步到状态模型
                self._apply_numeric_bounds(new_config)
                
                # 保存到文件
                self._save_config(new_config)
                
                # 通知监听器
                listener_tasks = []
                for listener in self._listeners:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            task = asyncio.create_task(listener(old_config, new_config))
                            listener_tasks.append(task)
                        else:
                            loop = asyncio.get_event_loop()
                            task = loop.run_in_executor(None, listener, old_config, new_config)
                            listener_tasks.append(task)
                    except Exception as e:
                        logger.error(f"配置变更监听器调用失败: {e}")
                
                # 等待所有监听器完成
                if listener_tasks:
                    await asyncio.gather(*listener_tasks, return_exceptions=True)
                
                # 更新当前配置
                self.current_config = new_config
                
                logger.info(f"配置更新成功: {len(updates)} 个字段已更新")
                
                return True
                
            except Exception as e:
                logger.error(f"更新配置失败: {e}")
                return False
    
    async def get_config_snapshot(self) -> Dict[str, Any]:
        """获取配置快照"""
        async with self._lock:
            config_dict = self.current_config.dict()
            
            return {
                'config': config_dict,
                'metadata': {
                    'config_path': str(self.config_path),
                    'last_modified': self._last_modified,
                    'config_hash': self._last_hash,
                    'listener_count': len(self._listeners),
                    'error_count': self._error_count
                }
            }
    
    async def stop_watching(self):
        """停止监控"""
        if self._watch_task:
            logger.info("停止配置监控任务...")
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None
            logger.info("配置监控任务已停止")
    
    async def refresh(self):
        """手动刷新配置"""
        logger.info("手动刷新配置...")
        await self._check_for_changes()
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            'watching': self._watch_task is not None,
            'listener_count': len(self._listeners),
            'error_count': self._error_count,
            'last_modified': self._last_modified,
            'config_file_exists': self.config_path.exists()
        }