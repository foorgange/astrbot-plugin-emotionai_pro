# managers.py
import time
import asyncio
import math
from typing import Dict, Any, Optional, List, Set, Tuple, Deque
from pathlib import Path
from collections import defaultdict, deque
import re
from dataclasses import asdict
import hashlib

from astrbot.api import logger

from .models import EnhancedEmotionalState, RankingEntry, InteractionStats
from .storage import UserStateRepository, BackupManager
from .cache import ShardedTTLCache
from .constants import UpdateThresholds, TimeConstants, EmotionConstants
from .config import PluginConfig

class UserStateManager:
    """用户状态管理器 - 完全优化版本"""
    
    def __init__(self, repository: UserStateRepository, config: PluginConfig):
        self.repository = repository
        self.config = config
        
        # 内存缓存
        self.cache = ShardedTTLCache(
            max_size=config.cache_max_size,
            default_ttl=config.cache_ttl
        )
        
        # 脏键管理
        self.dirty_keys: Set[str] = set()
        self.dirty_lock = asyncio.Lock()
        self.last_save_time = time.time()
        self.save_count = 0
        
        # 用户ID反向索引
        self.user_id_index: Dict[str, str] = {}
        
        # 性能统计
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'state_loads': 0,
            'state_saves': 0,
            'avg_save_time': 0.0,
            'errors': 0
        }
        
        # 自动保存任务
        self.auto_save_task: Optional[asyncio.Task] = None
        self._start_auto_save()
        
        # 状态监控
        self.monitor_task: Optional[asyncio.Task] = None
        self._start_monitoring()
        
        logger.info(f"用户状态管理器初始化完成，缓存大小: {config.cache_max_size}")

    async def smart_cache_cleanup(self) -> int:
        """智能缓存清理 - 优化版本"""
        try:
            cleaned_count = 0
            current_time = time.time()
            
            # 获取所有活跃用户（最近7天有互动的）
            all_states = await self.repository.get_all_user_states()
            active_users = set()
            
            for user_key, state in all_states.items():
                if state.stats.last_interaction_time > 0:
                    days_since_last = (current_time - state.stats.last_interaction_time) / 86400
                    if days_since_last <= 7:
                        active_users.add(user_key)
            
            logger.info(f"活跃用户数量: {len(active_users)}")
            
            # 清理不活跃或初始状态的缓存
            cache_stats = await self.cache.get_stats()
            total_entries = cache_stats.get('total_entries', 0)
            
            if total_entries > self.config.cache_max_size * 0.8:  # 超过80%容量
                # 获取所有缓存键（通过已知的用户键）
                for user_key in list(all_states.keys()):
                    cache_key = f"state_{user_key}"
                    
                    # 检查是否为活跃用户
                    if user_key in active_users:
                        continue
                    
                    # 检查是否为初始状态
                    if await self._is_initial_state_user(user_key):
                        # 清理这个初始状态用户的缓存
                        await self.cache.delete(cache_key)
                        cleaned_count += 1
                        
                        if cleaned_count % 10 == 0:
                            logger.info(f"已清理 {cleaned_count} 个初始状态用户缓存")
                    
                    # 如果清理了足够多的条目，停止
                    if cleaned_count >= 50:  # 每次最多清理50个
                        break
            
            if cleaned_count > 0:
                new_stats = await self.cache.get_stats()
                logger.info(f"智能缓存清理完成: 清理了 {cleaned_count} 个初始状态用户, "
                      f"缓存条目从 {total_entries} 减少到 {new_stats.get('total_entries', 0)}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"智能缓存清理失败: {e}")
            self.stats['errors'] += 1
            return 0

    async def _is_initial_state_user(self, user_key: str) -> bool:
        """判断用户是否为初始状态 - 优化版本"""
        try:
            # 尝试从缓存获取（如果存在，说明最近活跃）
            cached_state = await self.cache.get(f"state_{user_key}")
            if cached_state is not None:
                # 如果还在缓存中，说明最近使用过，不是初始状态
                return False
            
            # 从持久化存储获取用户状态
            state = await self.repository.get_user_state(user_key)
            if state is None:
                return True  # 用户不存在
            
            current_time = time.time()
            
            # 检查活跃度
            if state.stats.last_interaction_time > 0:
                days_since_last = (current_time - state.stats.last_interaction_time) / 86400
                if days_since_last < 30:  # 30天内活跃的用户
                    return False
            
            # 严格的初始状态判断标准
            is_initial = (
                state.favor == 0 and
                state.intimacy == 0 and
                state.descriptions.relationship == "陌生人" and
                state.descriptions.attitude == "中立" and
                state.stats.total_count == 0 and
                state.stats.positive_count == 0 and
                state.stats.negative_count == 0 and
                state.relationship_stage == "初识期" and
                state.stage_composite_score == 0.0 and
                state.force_update_counter == 0
            )
            
            return is_initial
            
        except Exception as e:
            logger.error(f"判断用户初始状态失败 {user_key}: {e}")
            self.stats['errors'] += 1
            return False  # 出错时保守处理，不清理

    def _start_auto_save(self):
        """启动自动保存任务"""
        async def auto_save_loop():
            last_save_report = time.time()
            
            while True:
                try:
                    await asyncio.sleep(self.config.auto_save_interval)
                    
                    # 检查是否有脏数据需要保存
                    async with self.dirty_lock:
                        has_dirty_data = len(self.dirty_keys) > 0
                    
                    if has_dirty_data:
                        start_time = time.time()
                        await self.force_save()
                        save_time = time.time() - start_time
                        
                        # 更新统计
                        self.stats['avg_save_time'] = (
                            self.stats['avg_save_time'] * 0.9 + save_time * 0.1
                        )
                    
                    # 每小时报告一次
                    current_time = time.time()
                    if current_time - last_save_report > 3600:
                        logger.info(f"自动保存统计: {self.stats['state_saves']}次保存, "
                              f"平均耗时 {self.stats['avg_save_time']:.3f}s")
                        last_save_report = current_time
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"自动保存失败: {e}")
                    self.stats['errors'] += 1
                    await asyncio.sleep(TimeConstants.ONE_MINUTE)
        
        self.auto_save_task = asyncio.create_task(auto_save_loop())
    
    def _start_monitoring(self):
        """启动状态监控任务"""
        async def monitor_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # 5分钟检查一次
                    
                    # 检查缓存命中率
                    cache_stats = await self.cache.get_stats()
                    hit_rate = cache_stats.get('hit_rate', 0)
                    
                    if hit_rate < 30:
                        logger.warning(f"缓存警告: 命中率较低 ({hit_rate:.1f}%)")
                    
                    # 检查脏键数量
                    async with self.dirty_lock:
                        dirty_count = len(self.dirty_keys)
                    
                    if dirty_count > self.config.max_dirty_keys * 0.8:
                        logger.warning(f"脏键警告: {dirty_count}个待保存键")
                    
                    # 检查内存使用
                    memory_info = cache_stats.get('memory_usage', {})
                    if memory_info.get('usage_percent', 0) > 80:
                        logger.warning(f"内存警告: 使用率 {memory_info['usage_percent']:.1f}%")
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"监控任务错误: {e}")
                    await asyncio.sleep(60)
        
        self.monitor_task = asyncio.create_task(monitor_loop())
    
    async def get_user_state(self, user_key: str) -> EnhancedEmotionalState:
        """获取用户状态（带缓存）"""
        cache_key = f"state_{user_key}"
        
        # 首先尝试缓存
        cached_state = await self.cache.get(cache_key)
        if cached_state is not None:
            self.stats['cache_hits'] += 1
            return cached_state
        
        # 缓存未命中
        self.stats['cache_misses'] += 1
        
        # 从存储加载
        try:
            state = await self.repository.get_user_state(user_key)
            if state is None:
                # 创建新状态
                state = EnhancedEmotionalState(user_key=user_key)
            
            # 放入缓存
            await self.cache.set(cache_key, state)
            self.stats['state_loads'] += 1
            
            return state
            
        except Exception as e:
            logger.error(f"加载用户状态失败 {user_key}: {e}")
            self.stats['errors'] += 1
            # 返回一个默认状态
            return EnhancedEmotionalState(user_key=user_key)
    
    async def update_user_state(self, user_key: str, state: EnhancedEmotionalState):
        """更新用户状态"""
        try:
            # 验证状态
            if not state.is_valid():
                logger.warning(f"用户 {user_key} 的状态无效，尝试修复")
                state.repair()
            
            # 更新缓存
            cache_key = f"state_{user_key}"
            await self.cache.set(cache_key, state)
            
            # 标记为脏
            async with self.dirty_lock:
                self.dirty_keys.add(user_key)
                
                # 检查脏键数量限制
                if len(self.dirty_keys) >= self.config.max_dirty_keys:
                    logger.info(f"脏键达到限制 ({len(self.dirty_keys)})，触发强制保存")
                    await self.force_save()
            
            # 更新反向索引
            if '_' in user_key:
                try:
                    _, user_id = user_key.split('_', 1)
                    self.user_id_index[user_id] = user_key
                except ValueError:
                    pass
            
        except Exception as e:
            logger.error(f"更新用户状态失败 {user_key}: {e}")
            self.stats['errors'] += 1
    
    async def force_save(self):
        """强制保存所有脏数据"""
        if not self.dirty_keys:
            return
        
        async with self.dirty_lock:
            dirty_keys = self.dirty_keys.copy()
            self.dirty_keys.clear()
        
        if not dirty_keys:
            return
        
        # 收集需要保存的状态
        states_to_save = {}
        failed_keys = []
        
        for user_key in dirty_keys:
            try:
                state = await self.cache.get(f"state_{user_key}")
                if state is not None:
                    states_to_save[user_key] = state
                else:
                    logger.warning(f"脏键 {user_key} 不在缓存中")
            except Exception as e:
                logger.error(f"获取脏键状态失败 {user_key}: {e}")
                failed_keys.append(user_key)
        
        if states_to_save:
            try:
                start_time = time.time()
                await self.repository.save_updated_user_states_only(states_to_save)
                save_time = time.time() - start_time
                
                self.last_save_time = time.time()
                self.save_count += 1
                self.stats['state_saves'] += len(states_to_save)
                
                logger.info(f"保存了 {len(states_to_save)} 个更新的用户状态，耗时 {save_time:.3f}s")
                
            except Exception as e:
                logger.error(f"保存用户状态失败: {e}")
                self.stats['errors'] += 1
                # 把失败的键加回脏键集合
                async with self.dirty_lock:
                    self.dirty_keys.update(dirty_keys)
        
        if failed_keys:
            logger.error(f"{len(failed_keys)} 个键保存失败")
    
    def resolve_user_key(self, user_input: str, session_based: bool) -> str:
        """解析用户标识符"""
        if not user_input or not isinstance(user_input, str):
            return ""
        
        if session_based:
            if '_' in user_input:
                return user_input
            else:
                # 尝试从反向索引查找
                return self.user_id_index.get(user_input, user_input)
        else:
            return user_input
    
    async def clear_all_data(self):
        """清空所有数据"""
        async with self.dirty_lock:
            self.dirty_keys.clear()
        
        await self.cache.clear()
        await self.repository.save_all_user_states({})
        self.user_id_index.clear()
        
        logger.info("已清空所有用户数据")
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        cache_stats = await self.cache.get_stats()
        
        return {
            'user_manager': {
                'dirty_keys': len(self.dirty_keys),
                'save_count': self.save_count,
                'last_save_time': self.last_save_time,
                'avg_save_time': self.stats['avg_save_time'],
                'state_loads': self.stats['state_loads'],
                'state_saves': self.stats['state_saves'],
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'cache_hit_rate': (
                    self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100
                    if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
                ),
                'errors': self.stats['errors'],
                'user_count': len(self.user_id_index)
            },
            'cache_stats': cache_stats
        }
    
    async def close(self):
        """关闭管理器"""
        logger.info("正在关闭用户状态管理器...")
        
        # 取消监控任务
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # 取消自动保存任务
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
        
        # 保存所有脏数据
        await self.force_save()
        
        # 关闭缓存
        await self.cache.close()
        
        # 打印统计信息
        stats = await self.get_stats()
        logger.info(f"用户状态管理器关闭完成，统计: {stats['user_manager']}")
        
        logger.info("用户状态管理器已关闭")

class RankingManager:
    """排行榜管理器 - 优化版本"""
    
    def __init__(self, user_manager: UserStateManager):
        self.user_manager = user_manager
        self.cache = ShardedTTLCache(max_size=20, default_ttl=120)  # 短缓存
        
        # 缓存预热标记
        self._last_cache_warm = 0
    
    async def get_enhanced_ranking(self, limit: int = 10, reverse: bool = True) -> List[RankingEntry]:
        """获取增强的排行榜 - 优化性能"""
        cache_key = f"ranking_{limit}_{reverse}"
        
        # 尝试缓存
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # 计算排行榜
        try:
            all_states = await self.user_manager.repository.get_all_user_states()
            averages = []
            
            for user_key, state in all_states.items():
                try:
                    # 计算加权平均分
                    avg = (state.favor * 0.6 + state.intimacy * 0.4)
                    
                    # 添加互动频率权重
                    if state.stats.total_count > 0:
                        interaction_weight = min(1.0, state.stats.total_count / 100.0)
                        avg *= (1 + interaction_weight * 0.1)  # 最多增加10%
                    
                    averages.append((
                        user_key, avg, state.favor, state.intimacy, 
                        state.descriptions.attitude, state.descriptions.relationship,
                        state.stats.total_count
                    ))
                except (AttributeError, TypeError) as e:
                    logger.error(f"用户 {user_key} 数据格式错误: {e}")
                    continue
            
            if not averages:
                return []
            
            # 排序
            averages.sort(key=lambda x: x[1], reverse=reverse)
            
            # 转换为条目
            entries = []
            for i, (user_key, avg, favor, intimacy, attitude, relationship, total_count) in enumerate(averages[:limit], 1):
                display_name = self._format_user_display(user_key)
                
                # 计算趋势
                trend = "↑" if avg > 0 else "↓" if avg < 0 else "→"
                
                entries.append(RankingEntry(
                    rank=i,
                    user_key=user_key,
                    average_score=avg,
                    favor=favor,
                    intimacy=intimacy,
                    attitude=attitude,
                    relationship=relationship,
                    display_name=display_name
                ))
            
            # 缓存结果
            await self.cache.set(cache_key, entries)
            return entries
            
        except Exception as e:
            logger.error(f"获取排行榜失败: {e}")
            return []
    
    def _format_user_display(self, user_key: str) -> str:
        """格式化用户显示名称"""
        if not user_key:
            return "未知用户"
        
        if '_' in user_key:
            try:
                session_id, user_id = user_key.split('_', 1)
                # 截断过长的ID
                if len(user_id) > 8:
                    user_id = user_id[:8] + "..."
                return f"用户{user_id}"
            except ValueError:
                pass
        
        # 截断过长的键
        if len(user_key) > 10:
            return f"用户{user_key[:8]}..."
        
        return f"用户{user_key}"
    
    async def warm_cache(self):
        """预热排行榜缓存"""
        current_time = time.time()
        if current_time - self._last_cache_warm < 300:  # 5分钟内不重复预热
            return
        
        try:
            # 预热常用排行榜
            common_limits = [5, 10, 20]
            for limit in common_limits:
                for reverse in [True, False]:
                    await self.get_enhanced_ranking(limit, reverse)
            
            self._last_cache_warm = current_time
            logger.info("排行榜缓存预热完成")
            
        except Exception as e:
            logger.error(f"缓存预热失败: {e}")
    
    async def get_ranking_stats(self) -> Dict[str, Any]:
        """获取排行榜统计信息"""
        try:
            all_states = await self.user_manager.repository.get_all_user_states()
            
            if not all_states:
                return {
                    'total_users': 0,
                    'average_favor': 0,
                    'average_intimacy': 0,
                    'top_users': []
                }
            
            total_favor = 0
            total_intimacy = 0
            user_count = len(all_states)
            
            for state in all_states.values():
                total_favor += state.favor
                total_intimacy += state.intimacy
            
            # 获取前5名用户
            top_users = []
            rankings = await self.get_enhanced_ranking(5, True)
            for entry in rankings:
                top_users.append({
                    'rank': entry.rank,
                    'display_name': entry.display_name,
                    'average_score': entry.average_score,
                    'favor': entry.favor,
                    'intimacy': entry.intimacy
                })
            
            return {
                'total_users': user_count,
                'average_favor': total_favor / user_count,
                'average_intimacy': total_intimacy / user_count,
                'top_users': top_users
            }
            
        except Exception as e:
            logger.error(f"获取排行榜统计失败: {e}")
            return {}

class SmartUpdateManager:
    """智能更新管理器 - 增强版本"""
    
    def __init__(self):
        # 情感关键词数据库
        self.emotional_keywords = {
            'positive': ['喜欢', '爱', '开心', '高兴', '谢谢', '感谢', '感动', '温暖', '棒', '好', '不错', '可爱', '漂亮', '美丽'],
            'negative': ['讨厌', '恨', '生气', '愤怒', '伤心', '难过', '失望', '烦', '滚', '傻', '笨', '蠢', '垃圾', '不愿意'],
            'intimate': ['想你', '想念', '关心', '担心', '在乎', '重要', '宝贝', '亲爱的', '搞好关系', '拥抱', '吻'],
            'conflict': ['吵架', '争执', '不满', '抱怨', '批评', '指责', '反对', '不同意']
        }
        
        # 情感强度分析器
        self.intensity_patterns = {
            'strong_positive': re.compile(r'(非常|特别|极其|太|真的)好|喜欢|爱|开心'),
            'strong_negative': re.compile(r'(非常|特别|极其|太|真的)讨厌|恨|生气|烦'),
            'question': re.compile(r'[？?]'),
            'exclamation': re.compile(r'[！!]'),
            'emoticon_positive': re.compile(r'[:：][)）]|😊|😄|😍|🥰|🤗'),
            'emoticon_negative': re.compile(r'[:：][(（]|😠|😡|😢|😭|😤')
        }
        
        # 缓存分析结果
        self.analysis_cache = {}
        self.cache_max_size = 1000
    
    def should_update_emotion(self, current_state: EnhancedEmotionalState, 
                            user_message: str, ai_response: str) -> Tuple[bool, str, int]:
        """判断是否需要情感更新 - 返回（是否需要，原因，情感强度）"""
        reasons = []
        emotional_intensity = 0
        
        # 1. 基于情感强度变化
        emotion_intensity = self._calculate_emotion_intensity(current_state)
        if emotion_intensity >= UpdateThresholds.MAJOR_CHANGE:
            reasons.append("情感强度重大变化")
            emotional_intensity += 3
        
        # 2. 基于对话内容关键词分析
        keyword_result = self._analyze_emotional_keywords(user_message, ai_response)
        if keyword_result['should_update']:
            reasons.append(keyword_result['reason'])
            emotional_intensity += keyword_result['intensity']
        
        # 3. 基于时间间隔
        if self._is_long_time_no_update(current_state):
            reasons.append("长时间未更新")
            emotional_intensity += 1
        
        # 4. 强制更新检查
        if current_state.should_force_update(UpdateThresholds.FORCE_UPDATE):
            reasons.append("强制更新机制")
            emotional_intensity += 2
        
        # 5. 基于互动频率
        if current_state.stats.total_count > 0:
            days_since_last = current_state.stats.days_since_last
            if days_since_last > 7:  # 超过7天
                reasons.append("久别重逢")
                emotional_intensity += 2
        
        # 判断是否需要更新
        should_update = len(reasons) > 0 or emotional_intensity >= 3
        
        reason_text = " | ".join(reasons) if reasons else "无明显情感变化"
        
        return should_update, reason_text, emotional_intensity
    
    def _calculate_emotion_intensity(self, state: EnhancedEmotionalState) -> int:
        """计算情感变化强度"""
        emotions = [
            state.emotions.joy, state.emotions.trust, state.emotions.fear, state.emotions.surprise,
            state.emotions.sadness, state.emotions.disgust, state.emotions.anger, state.emotions.anticipation
        ]
        return max(emotions) - min(emotions)
    
    def _analyze_emotional_keywords(self, user_message: str, ai_response: str) -> Dict[str, Any]:
        """分析情感关键词 - 增强版本"""
        message_lower = user_message.lower()
        response_lower = ai_response.lower()
        
        result = {
            'should_update': False,
            'reason': '',
            'intensity': 0,
            'category': 'neutral'
        }
        
        # 检查用户消息中的情感关键词
        intensity_score = 0
        detected_categories = set()
        
        for category, keywords in self.emotional_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    detected_categories.add(category)
                    
                    # 计算强度
                    if category == 'positive':
                        intensity_score += 2
                    elif category == 'negative':
                        intensity_score += 3  # 负面情感权重更高
                    elif category == 'intimate':
                        intensity_score += 2
                    elif category == 'conflict':
                        intensity_score += 3
        
        # 检查AI回应中的情感关键词
        for category, keywords in self.emotional_keywords.items():
            for keyword in keywords:
                if keyword in response_lower:
                    detected_categories.add(category)
                    intensity_score += 1  # AI回应的权重较低
        
        # 检查情感强度模式
        for pattern_name, pattern in self.intensity_patterns.items():
            if pattern.search(user_message) or pattern.search(ai_response):
                if 'strong' in pattern_name:
                    intensity_score += 2
                elif 'emoticon' in pattern_name:
                    intensity_score += 1
                elif 'question' in pattern_name:
                    intensity_score += 0.5
                elif 'exclamation' in pattern_name:
                    intensity_score += 1
        
        # 判断是否需要更新
        if intensity_score >= 2:
            result['should_update'] = True
            result['intensity'] = min(5, int(intensity_score))
            
            # 生成原因描述
            if 'negative' in detected_categories and 'conflict' in detected_categories:
                result['reason'] = "用户表达强烈负面情感和冲突"
                result['category'] = 'negative_conflict'
            elif 'negative' in detected_categories:
                result['reason'] = "用户表达负面情感"
                result['category'] = 'negative'
            elif 'positive' in detected_categories and 'intimate' in detected_categories:
                result['reason'] = "用户表达积极亲密情感"
                result['category'] = 'positive_intimate'
            elif 'positive' in detected_categories:
                result['reason'] = "用户表达积极情感"
                result['category'] = 'positive'
            elif 'intimate' in detected_categories:
                result['reason'] = "用户表达亲密情感"
                result['category'] = 'intimate'
            else:
                result['reason'] = "对话包含情感关键词"
                result['category'] = 'emotional'
        
        return result
    
    def _is_long_time_no_update(self, state: EnhancedEmotionalState) -> bool:
        """检查是否长时间未更新"""
        current_time = time.time()
        
        # 检查态度更新
        attitude_update_time = state.descriptions.last_attitude_update
        if current_time - attitude_update_time > TimeConstants.ONE_DAY:
            return True
        
        # 检查关系更新
        relationship_update_time = state.descriptions.last_relationship_update
        if current_time - relationship_update_time > TimeConstants.ONE_DAY:
            return True
        
        # 检查强制更新
        force_update_time = state.last_force_update
        if current_time - force_update_time > TimeConstants.THIRTY_MINUTES * 2:
            return True
        
        return False
    
    def get_conversation_analysis(self, user_message: str, ai_response: str) -> Dict[str, Any]:
        """获取对话情感分析"""
        # 生成缓存键
        cache_key = hashlib.md5(f"{user_message}_{ai_response}".encode()).hexdigest()[:16]
        
        # 检查缓存
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        analysis = self._analyze_emotional_keywords(user_message, ai_response)
        
        # 添加更多分析维度
        analysis['message_length'] = len(user_message)
        analysis['response_length'] = len(ai_response)
        
        # 简单的情感倾向分析
        if analysis['category'] in ['positive', 'positive_intimate']:
            analysis['sentiment'] = 'positive'
        elif analysis['category'] in ['negative', 'negative_conflict']:
            analysis['sentiment'] = 'negative'
        else:
            analysis['sentiment'] = 'neutral'
        
        # 缓存结果
        if len(self.analysis_cache) >= self.cache_max_size:
            # 删除最旧的条目
            oldest_key = next(iter(self.analysis_cache))
            del self.analysis_cache[oldest_key]
        
        self.analysis_cache[cache_key] = analysis
        
        return analysis

class EmotionAnalyzer:
    """情感分析器 - 增强版本"""
    
    @classmethod
    def get_dominant_emotion(cls, state: EnhancedEmotionalState) -> str:
        """获取主导情感"""
        return state.emotions.get_dominant()
    
    @classmethod
    def get_emotional_profile(cls, state: EnhancedEmotionalState, 
                            favor_weight: float = 0.6, intimacy_weight: float = 0.4) -> Dict[str, Any]:
        """获取完整的情感档案"""
        dominant_emotion = cls.get_dominant_emotion(state)
        
        # 计算情感强度
        emotion_summary = state.emotions.get_summary()
        
        # 计算复合评分
        composite_score = state.favor * favor_weight + state.intimacy * intimacy_weight
        
        # 判断关系趋势
        favor_contribution = state.favor * favor_weight
        intimacy_contribution = state.intimacy * intimacy_weight
        
        if favor_contribution > intimacy_contribution * 1.2:
            relationship_trend = "好感领先"
        elif intimacy_contribution > favor_contribution * 1.2:
            relationship_trend = "亲密度领先" 
        else:
            relationship_trend = "平衡发展"
        
        # 计算稳定性得分
        interaction_stats = state.stats.get_summary()
        days_since_last = interaction_stats['days_since_last']
        # days_since_last 可能为 inf（从未互动），clamp 到 365 天，避免 100-inf 出现 -inf
        if not math.isfinite(days_since_last):
            days_since_last = 365
        stability_score = max(0.0, min(100.0,
            interaction_stats['positive_ratio'] * 0.8 +
            (100 - days_since_last) * 0.2))
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": emotion_summary['total_intensity'],
            "positive_balance": emotion_summary['positive_balance'],
            "relationship_trend": relationship_trend,
            "positive_ratio": state.stats.positive_ratio,
            "composite_score": composite_score,
            "favor_weight": favor_weight,
            "intimacy_weight": intimacy_weight,
            "stability_score": stability_score,
            "interaction_summary": interaction_stats,
            "emotion_details": emotion_summary['details']
        }
    
    @classmethod
    def analyze_emotional_change(cls, old_state: EnhancedEmotionalState, 
                               new_state: EnhancedEmotionalState) -> Dict[str, Any]:
        """分析情感变化"""
        changes = {
            'favor': new_state.favor - old_state.favor,
            'intimacy': new_state.intimacy - old_state.intimacy,
            'emotions': {},
            'relationship_stage_changed': old_state.relationship_stage != new_state.relationship_stage,
            'attitude_changed': old_state.descriptions.attitude != new_state.descriptions.attitude,
            'relationship_changed': old_state.descriptions.relationship != new_state.descriptions.relationship
        }
        
        # 计算情感变化
        emotion_fields = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
        for field in emotion_fields:
            old_value = getattr(old_state.emotions, field)
            new_value = getattr(new_state.emotions, field)
            changes['emotions'][field] = new_value - old_value
        
        # 计算总变化量
        total_change = abs(changes['favor']) + abs(changes['intimacy']) + \
                      sum(abs(v) for v in changes['emotions'].values())
        
        changes['total_change'] = total_change
        
        # 判断变化级别
        if total_change >= 8:
            changes['change_level'] = 'major'
        elif total_change >= 3:
            changes['change_level'] = 'moderate'
        else:
            changes['change_level'] = 'minor'
        
        return changes