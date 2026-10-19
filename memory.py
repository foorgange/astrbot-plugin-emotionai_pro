# memory.py
import heapq
import math
import re
import time
import asyncio
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Deque, Tuple
from collections import deque
from dataclasses import dataclass

from astrbot.api import logger

from .cache import ShardedTTLCache
from .storage import UserStateRepository
from .constants import TimeConstants, UpdateThresholds
from . import relationship_style

# 每个用户长期记忆的硬上限（实际保留条数由好感度动态决定，见
# relationship_style.memory_policy）。留足空间给高好感用户。
HARD_MAX_PER_USER = 100

# 记忆召回时的时间衰减半衰期（天）。长期关系里旧记忆不该瞬间失效，
# 但“最近发生的事”仍有更高权重。
RECALL_HALF_LIFE_DAYS = 45.0

@dataclass
class InteractionRecord:
    """互动记录

    v4.2.0 新增字段都带默认值，旧存档可直接反序列化：
    - ``valence``：这条记忆的情绪价（1 正 / 0 中 / -1 负）；
    - ``kind``：preference / shared_event / promise / conflict / boundary / event；
    - ``favor_at_time``：写入时的好感度，避免未来高好感时用过度温暖的
      视角回忆旧冲突；
    - ``last_recalled_at`` / ``recall_count``：避免同一段记忆每轮重复提起。
    """
    user_msg: str
    ai_response: str
    timestamp: float
    significance: int
    emotional_changes: Dict[str, int] = None
    embedding: Optional[List[float]] = None  # 可选的语义嵌入
    valence: int = 0
    kind: str = "event"
    favor_at_time: int = 0
    last_recalled_at: float = 0.0
    recall_count: int = 0

    def __post_init__(self):
        self.user_msg = str(self.user_msg or "")
        self.ai_response = str(self.ai_response or "")
        if not isinstance(self.emotional_changes, dict):
            self.emotional_changes = {}
        try:
            self.timestamp = float(self.timestamp or 0.0)
        except (TypeError, ValueError):
            self.timestamp = 0.0
        try:
            self.significance = max(0, min(10, int(self.significance)))
        except (TypeError, ValueError):
            self.significance = 0
        self.valence = _coerce_valence(self.valence)
        if not isinstance(self.kind, str) or not self.kind:
            self.kind = "event"
        self.favor_at_time = _coerce_int(self.favor_at_time, 0)
        try:
            self.last_recalled_at = float(self.last_recalled_at or 0.0)
            self.recall_count = max(0, int(self.recall_count or 0))
        except (TypeError, ValueError):
            self.last_recalled_at = 0.0
            self.recall_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'user_msg': self.user_msg,
            'ai_response': self.ai_response,
            'timestamp': self.timestamp,
            'significance': self.significance,
            'emotional_changes': self.emotional_changes,
            'embedding': self.embedding,
            'valence': self.valence,
            'kind': self.kind,
            'favor_at_time': self.favor_at_time,
            'last_recalled_at': self.last_recalled_at,
            'recall_count': self.recall_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InteractionRecord':
        """从字典创建（兼容 v4.1.x 旧存档）"""
        return cls(
            user_msg=data.get('user_msg', ''),
            ai_response=data.get('ai_response', ''),
            timestamp=data.get('timestamp', 0),
            significance=data.get('significance', 0),
            emotional_changes=data.get('emotional_changes', {}),
            embedding=data.get('embedding'),
            valence=data.get('valence', 0),
            kind=data.get('kind', 'event'),
            favor_at_time=data.get('favor_at_time', 0),
            last_recalled_at=data.get('last_recalled_at', 0.0),
            recall_count=data.get('recall_count', 0),
        )


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_valence(value: Any) -> int:
    v = _coerce_int(value, 0)
    return 1 if v > 0 else (-1 if v < 0 else 0)


# 这些维度数值越高代表情绪越负面；不能用“变化值为正”判断为正面事件。
_NEGATIVE_EMOTION_KEYS = frozenset({"fear", "sadness", "disgust", "anger"})


def _compute_valence(changes: Dict[str, Any]) -> int:
    """从情感变化里推情绪价：正负变化互相抵消后取符号。

    ⚠️ anger/sadness 等“负面情绪 +3”是负面事件，不能按数值符号算成
    正面；否则“吵架后愤怒 +3”会被判成好记忆。
    """
    if not isinstance(changes, dict):
        return 0
    positive = 0.0
    negative = 0.0
    for key, value in changes.items():
        if key in ("mood_reason", "attitude_text", "relationship_text", "source"):
            continue
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if key in _NEGATIVE_EMOTION_KEYS:
            if amount > 0:
                negative += amount
            elif amount < 0:
                positive += abs(amount)
        else:
            if amount > 0:
                positive += amount
            elif amount < 0:
                negative += abs(amount)
    if positive > negative * 1.2:
        return 1
    if negative > positive * 1.2:
        return -1
    return 0


_KIND_PATTERNS = (
    ("conflict", ("吵架", "生气", "讨厌", "烦死", "滚", "闭嘴", "道歉",
                  "对不起", "冲突", "冒犯", "边界", "别烦", "不想理")),
    ("promise", ("答应", "约定", "说好", "下次", "一定", "保证", "改天")),
    ("preference", ("喜欢", "讨厌吃", "爱吃", "不吃", "生日", "习惯",
                    "口头禅", "过敏", "忌口")),
    ("shared_event", ("一起", "上次", "昨天", "今天", "那天", "记得",
                      "之前", "刚才", "周末")),
    ("boundary", ("不要", "别", "不可以", "不舒服", "介意", "底线")),
)


def _classify_kind(user_msg: str, ai_response: str, valence: int) -> str:
    """轻量规则分类；只用于召回时的偏好，不影响是否写入。

    先看用户原话，再看 bot 回应：用户说“我喜欢……”时应归为 preference，
    不能因为 bot 回了一句“下次给你带”就被 promise 规则抢走。
    """
    user_text = str(user_msg or "")
    ai_text = str(ai_response or "")
    for kind, words in _KIND_PATTERNS:
        if any(word in user_text for word in words):
            return kind
    for kind, words in _KIND_PATTERNS:
        if any(word in ai_text for word in words):
            return kind
    if valence < 0:
        return "conflict"
    return "event"


def _ngrams(text: str, n: int = 2) -> set:
    """中文短文本的字符 n-gram，用于无 embedding 时的相关性估算。"""
    cleaned = re.sub(r"\s+", "", str(text or "").lower())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i + n] for i in range(len(cleaned) - n + 1)}


def _relevance(query: str, record: "InteractionRecord") -> float:
    """查询与记忆的相关性（0~1）。不依赖外部 embedding。"""
    if not query:
        return 0.0
    q = str(query).strip()[:400]
    if not q:
        return 0.0
    target = f"{record.user_msg}\n{record.ai_response}"
    q_grams = _ngrams(q)
    t_grams = _ngrams(target)
    if not q_grams or not t_grams:
        return 0.0
    overlap = len(q_grams & t_grams) / max(1, min(len(q_grams), len(t_grams)))
    # 长查询时 n-gram 会被常用字稀释，SequenceMatcher 补一点整体相似度
    ratio = SequenceMatcher(None, q, target[:400]).ratio()
    return max(0.0, min(1.0, 0.7 * overlap + 0.3 * ratio))


def _recency_score(timestamp: float, now: Optional[float] = None) -> float:
    now = time.time() if now is None else now
    try:
        age_days = max(0.0, (now - float(timestamp)) / 86400.0)
    except (TypeError, ValueError):
        age_days = 0.0
    return math.exp(-age_days / RECALL_HALF_LIFE_DAYS)

class EnhancedMemorySystem:
    """增强记忆系统 - 完全优化版本"""
    
    def __init__(self, repository: UserStateRepository, max_long_term_memory: int = 1000,
                 significance_threshold: int = None):
        self.repository = repository
        self.max_long_term_memory = max_long_term_memory

        # 存入长期记忆的「情感意义」门槛（v4.0.20 修正）
        #
        # ⚠️ 这里以前写死用 `UpdateThresholds.EMOTIONAL_SIGNIFICANCE`(=5)，
        # 而配置界面上的「情感意义阈值」（默认也是 5）从未被任何代码读取 ——
        # 用户把它改成 8 或 3，长期记忆的筛选行为完全不变。
        #
        # 现在改为可注入：传 None / 非法值时退回常量默认值。
        try:
            # ⚠️ bool 必须先挡掉：`int(True) == 1` 会伪装成合法阈值 1
            if significance_threshold is None or isinstance(significance_threshold, bool):
                raise ValueError
            threshold = int(significance_threshold)
            if not 1 <= threshold <= 10:
                raise ValueError
        except (TypeError, ValueError):
            threshold = UpdateThresholds.EMOTIONAL_SIGNIFICANCE
        self.significance_threshold = threshold
        
        # 短期记忆缓存 - 按用户分组
        self.short_term_memory = ShardedTTLCache(
            max_size=100,  # 100个用户
            default_ttl=TimeConstants.ONE_HOUR * 2  # 2小时
        )
        
        # 重要事件记忆 - 全局优先队列
        # tuple: (-significance, timestamp, seq, event)。seq 用来避免
        # significance/timestamp 完全相同时继续比较 dict 而 TypeError。
        self.important_events: List[Tuple[int, float, int, Dict[str, Any]]] = []
        self._important_seq = 0
        self.important_events_lock = asyncio.Lock()
        
        # 内存中的长期记忆 - 按用户组织
        self._long_term_memory: Dict[str, Deque[InteractionRecord]] = {}
        self._memory_loaded = False
        self._load_failed = False
        self._memory_lock = asyncio.Lock()
        
        # 记忆索引 - 用于快速检索
        self._memory_index: Dict[str, List[Tuple[float, str]]] = {}  # user_key -> [(timestamp, record_key)]
        
        # 清理任务
        self.cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        logger.info(f"增强记忆系统初始化完成，长期记忆限制: {max_long_term_memory}条")
    
    async def ensure_loaded(self) -> None:
        """公开的懒加载入口。

        v4.2.0：主请求在注入提示词前显式 ``await`` 一次，避免重启后
        第一轮注入读到空记忆（旧实现只有 ``add_interaction`` 才会触发加载）。
        """
        await self._ensure_memory_loaded()

    async def _ensure_memory_loaded(self):
        """确保记忆数据已加载"""
        async with self._memory_lock:
            if not self._memory_loaded:
                try:
                    if self.repository is None:
                        self._long_term_memory = {}
                    else:
                        memory_data = await self.repository.get_memory_data()
                        self._long_term_memory = self._deserialize_memory_data(memory_data)

                    # 构建索引
                    self._build_memory_index()

                    self._memory_loaded = True
                    self._load_failed = False
                    logger.info(f"加载了 {len(self._long_term_memory)} 个用户的长期记忆")

                except Exception as e:
                    logger.error(f"加载记忆数据失败: {e}")
                    self._long_term_memory = {}
                    self._memory_loaded = True
                    self._load_failed = True
    
    def _deserialize_memory_data(self, memory_data: Dict[str, Any]) -> Dict[str, Deque[InteractionRecord]]:
        """反序列化记忆数据。

        v4.2.0：deque 硬上限改为 HARD_MAX_PER_USER。实际保留条数在
        写入时按好感度动态裁剪（低好感少留、高好感多留）。
        """
        result = {}
        if not isinstance(memory_data, dict):
            return result
        for user_key, records in memory_data.items():
            if isinstance(records, list):
                record_deque = deque(maxlen=HARD_MAX_PER_USER)
                for record_data in records[-HARD_MAX_PER_USER:]:
                    try:
                        record = InteractionRecord.from_dict(record_data)
                        record_deque.append(record)
                    except Exception as e:
                        logger.error(f"反序列化记忆记录失败: {e}")
                        continue
                result[user_key] = record_deque
        return result
    
    def _serialize_memory_data(self) -> Dict[str, Any]:
        """序列化记忆数据"""
        result = {}
        for user_key, records in self._long_term_memory.items():
            result[user_key] = [record.to_dict() for record in records]
        return result
    
    def _build_memory_index(self):
        """构建记忆索引"""
        self._memory_index.clear()
        for user_key, records in self._long_term_memory.items():
            if user_key not in self._memory_index:
                self._memory_index[user_key] = []
            
            for record in records:
                # 使用组合键: user_key + timestamp
                record_key = f"{user_key}_{record.timestamp}"
                self._memory_index[user_key].append((record.timestamp, record_key))
            
            # 按时间戳排序
            self._memory_index[user_key].sort(key=lambda x: x[0], reverse=True)
    
    def _start_cleanup_task(self):
        """启动记忆清理任务"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(TimeConstants.ONE_HOUR)  # 每小时清理一次
                    await self._cleanup_memory()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"记忆清理任务错误: {e}")
                    await asyncio.sleep(TimeConstants.ONE_MINUTE * 5)
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_memory(self):
        """清理记忆"""
        try:
            # 清理短期记忆中的过期条目
            cleanup_result = await self.short_term_memory.cleanup_all_expired()
            if cleanup_result['total_cleaned'] > 0:
                logger.info(f"清理短期记忆: {cleanup_result['total_cleaned']}条")
            
            # 检查长期记忆大小
            total_records = sum(len(records) for records in self._long_term_memory.values())
            if total_records > self.max_long_term_memory:
                await self._prune_long_term_memory()
            
        except Exception as e:
            logger.error(f"记忆清理失败: {e}")
    
    async def _prune_long_term_memory(self):
        """修剪长期记忆。

        ⚠️ 旧实现用 ``heapq`` 压入 ``(-significance, timestamp, user_key,
        record)``；当 significance 与 timestamp 都相同时，堆会继续比较
        ``record``（dataclass 不可排序）→ TypeError。这里改成显式排序，
        并顺带按用户档位重新裁剪每个用户的保留条数。
        """
        try:
            all_records = []
            for user_key, records in self._long_term_memory.items():
                for record in records:
                    all_records.append((record.significance, record.timestamp,
                                        user_key, record))
            # 重要性优先，同分时新记忆优先
            all_records.sort(key=lambda item: (-item[0], -item[1]))

            target_size = max(1, int(self.max_long_term_memory * 0.8))
            new_long_term_memory: Dict[str, Deque[InteractionRecord]] = {}
            records_kept = 0
            for _, _, user_key, record in all_records:
                if records_kept >= target_size:
                    break
                limit = self._per_user_limit(record.favor_at_time, 0)
                bucket = new_long_term_memory.get(user_key)
                if bucket is None or bucket.maxlen != limit:
                    bucket = deque(bucket or [], maxlen=limit)
                    new_long_term_memory[user_key] = bucket
                bucket.append(record)
                records_kept += 1

            self._long_term_memory = new_long_term_memory
            self._build_memory_index()
            logger.info(f"修剪长期记忆完成: {records_kept}条记录被保留")

        except Exception as e:
            logger.error(f"修剪长期记忆失败: {e}")

    def _per_user_limit(self, favor: Any, intimacy: Any) -> int:
        """按好感/亲密档位计算单用户保留上限。"""
        try:
            if favor is None:
                return 50
            policy = relationship_style.memory_policy(favor, intimacy)
            return max(10, min(HARD_MAX_PER_USER, int(policy["per_user_limit"])))
        except Exception:
            return 50

    async def add_interaction(self, user_key: str, user_msg: str,
                            ai_response: str, emotional_significance: int,
                            emotional_changes: Dict[str, int] = None,
                            favor: Optional[int] = None,
                            intimacy: Optional[int] = None):
        """添加互动到记忆系统。

        v4.2.0：
        - ``favor`` / ``intimacy`` 参与写入阈值与保留条数；
        - 记录 ``valence`` / ``kind`` / ``favor_at_time``，供好感度驱动的
          召回使用；
        - 高好感降低写入门槛（愿意记细节），低好感提高门槛（只记重要/
          冲突事件）。不传 favor 时保持 v4.1 的原始阈值行为。
        """
        try:
            await self._ensure_memory_loaded()

            try:
                significance = max(0, min(10, int(emotional_significance)))
            except (TypeError, ValueError):
                significance = 0
            changes = emotional_changes if isinstance(emotional_changes, dict) else {}
            valence = _compute_valence(changes)
            kind = _classify_kind(user_msg, ai_response, valence)
            favor_value = _coerce_int(favor, 0) if favor is not None else 0

            new_interaction = InteractionRecord(
                user_msg=str(user_msg)[:500],
                ai_response=str(ai_response)[:500],
                timestamp=time.time(),
                significance=significance,
                emotional_changes=changes,
                valence=valence,
                kind=kind,
                favor_at_time=favor_value,
            )

            # 短期记忆：无论是否进长期库都先记一笔，供 /调试记忆 与
            # 上下文兜底使用。
            await self._add_to_short_term_memory(user_key, new_interaction)

            # 好感度驱动的写入门槛：高好感 -2，低好感 +2。
            effective_threshold = relationship_style.effective_threshold(
                self.significance_threshold,
                favor if favor is not None else 0,
                intimacy if intimacy is not None else 0,
            )
            # 低好感用户可以不记日常好事，但冲突/边界必须记住，否则
            # “态度冷”会失去记忆依据。负情绪价事件单独放宽两分。
            if valence < 0 and favor_value < 0:
                effective_threshold = max(1, effective_threshold - 2)
            if significance >= effective_threshold:
                # 只传两个位置参数：旧测试/第三方可能用
                # ``_add_to_long_term_memory(user_key, interaction)`` 的签名
                # 打桩；favor 已写进 interaction.favor_at_time。
                await self._add_to_long_term_memory(user_key, new_interaction)
                await self._remember_important_event(user_key, new_interaction)
                await self._save_long_term_memory()
                logger.info(
                    f"添加到长期记忆 - 用户: {user_key}, 意义: "
                    f"{significance}/10, 阈值: {effective_threshold}, "
                    f"情绪价: {valence}"
                )

        except Exception as e:
            logger.error(f"添加互动到记忆系统失败: {e}")
            # 不重新抛出异常，避免影响主流程

    async def _remember_important_event(self, user_key: str,
                                        interaction: InteractionRecord) -> None:
        """记录重要事件（内存优先队列，最多 50 条）。"""
        async with self.important_events_lock:
            self._important_seq += 1
            heapq.heappush(
                self.important_events,
                (-interaction.significance, interaction.timestamp,
                 self._important_seq, {
                     'user_key': user_key,
                     'user_msg': interaction.user_msg[:100],
                     'significance': interaction.significance,
                     'timestamp': interaction.timestamp,
                 }),
            )
            if len(self.important_events) > 50:
                heapq.heappop(self.important_events)

    async def _add_to_short_term_memory(self, user_key: str, interaction: InteractionRecord):
        """添加到短期记忆"""
        try:
            # 获取现有近期互动
            recent_interactions = await self.short_term_memory.get(f"recent_{user_key}")
            if recent_interactions is None:
                recent_interactions = []
            
            # 添加到列表
            recent_interactions.append(interaction)
            
            # 只保留最近10条
            if len(recent_interactions) > 10:
                recent_interactions = recent_interactions[-10:]
            
            # 保存回缓存
            await self.short_term_memory.set(f"recent_{user_key}", recent_interactions)
            
        except Exception as e:
            logger.error(f"添加到短期记忆失败: {e}")
    
    async def _add_to_long_term_memory(self, user_key: str,
                                       interaction: InteractionRecord,
                                       favor: Optional[int] = None,
                                       intimacy: Optional[int] = None):
        """添加到长期记忆（保留条数由关系档位决定）。"""
        async with self._memory_lock:
            limit = self._per_user_limit(
                favor if favor is not None else interaction.favor_at_time,
                intimacy if intimacy is not None else 0,
            )
            existing = self._long_term_memory.get(user_key)
            if existing is None or existing.maxlen != limit:
                existing = deque(existing or [], maxlen=limit)
                self._long_term_memory[user_key] = existing

            existing.append(interaction)

            # 更新索引
            if user_key not in self._memory_index:
                self._memory_index[user_key] = []

            record_key = f"{user_key}_{interaction.timestamp}"
            self._memory_index[user_key].append((interaction.timestamp, record_key))

            # 保持索引排序
            self._memory_index[user_key].sort(key=lambda x: x[0], reverse=True)

    async def _save_long_term_memory(self):
        """保存长期记忆。

        ⚠️ v4.2.0：如果本次运行从未加载过长期记忆，``_long_term_memory``
        还是空 dict；旧实现会在插件退出时把它原样写盘，把已有记忆覆盖成
        ``{}``。这里加一道保护：未加载且内存为空时直接跳过。
        """
        try:
            if self.repository is None:
                return
            if not self._memory_loaded and not self._long_term_memory:
                logger.info("长期记忆尚未加载，跳过保存以避免覆盖已有存档")
                return
            if self._load_failed and not self._long_term_memory:
                logger.error(
                    "长期记忆加载失败且内存为空，跳过保存以避免覆盖原存档；"
                    "请检查 long_term_memory.json 或 .bak 文件"
                )
                return
            async with self._memory_lock:
                memory_data = self._serialize_memory_data()
                await self.repository.save_memory_data(memory_data)
        except Exception as e:
            logger.error(f"保存长期记忆失败: {e}")
    
    _KIND_PREFIX = {
        "preference": "对方曾提到：",
        "promise": "你们约好：",
        "conflict": "你们有过摩擦：",
        "boundary": "对方说过边界：",
        "shared_event": "你们一起：",
        "event": "对方说过：",
    }

    def _memory_snippet(self, record: InteractionRecord) -> str:
        """把一条记录压成适合注入的短句（不包含内部字段名）。"""
        try:
            text = (record.user_msg or "").strip()
            if not text:
                text = (record.ai_response or "").strip()
            if not text:
                return ""
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 80:
                text = text[:77] + "..."
            when = time.strftime("%m-%d", time.localtime(record.timestamp or 0))
            prefix = self._KIND_PREFIX.get(record.kind, "对方说过：")
            # 规则分类有时会把“今天/昨天”归到 shared_event；没有“一起”
            # 时用中性前缀，避免写成“你们一起：今天被老板骂了”。
            if record.kind == "shared_event" and "一起" not in text:
                prefix = "对方说过："
            if record.kind == "promise" and not any(
                word in text for word in ("约定", "答应", "说好", "下次", "保证")
            ):
                prefix = "对方说过："
            return f"- [{when}] {prefix}{text}"
        except Exception:
            return ""

    async def build_memory_context(self, user_key: str, query: str = "",
                                   favor: Any = 0, intimacy: Any = 0,
                                   mood_valence: int = 0,
                                   max_items: Optional[int] = None,
                                   enabled: bool = True) -> str:
        """好感度驱动的长期记忆注入文本。

        - 先加载存档（解决重启后第一轮看不到记忆的问题）；
        - 关系统计照旧保留；
        - 高好感召回更多具体记忆，低好感只召回少量负面/边界记忆；
        - 当前心情与记忆情绪价一致时略微加权（mood-congruent recall）。

        返回值只用于请求尾部的 ``extra_user_content_parts``，不进入
        system_prompt 或历史 contexts，因此不会破坏前缀缓存。
        """
        await self.ensure_loaded()
        base = self.get_relationship_context(user_key)
        if not enabled:
            return base

        try:
            records = list(self._long_term_memory.get(user_key, []))
            if not records:
                return base + "具体记忆: 暂无\n"

            policy = relationship_style.memory_policy(favor, intimacy, max_items)
            k = max(1, int(policy["max_items"]))
            bias = str(policy["bias"])
            now = time.time()

            scored = []
            for idx, record in enumerate(records):
                relevance = _relevance(query, record)
                importance = max(0, min(10, int(record.significance))) / 10.0
                recency = _recency_score(record.timestamp, now)

                if bias == "positive":
                    valence_match = 1.0 if record.valence > 0 else (
                        0.4 if record.valence == 0 else 0.0)
                elif bias == "negative":
                    valence_match = 1.0 if record.valence < 0 else (
                        0.4 if record.valence == 0 else 0.0)
                else:
                    valence_match = 0.5

                if mood_valence > 0 and record.valence > 0:
                    valence_match = min(1.0, valence_match + 0.2)
                elif mood_valence < 0 and record.valence < 0:
                    valence_match = min(1.0, valence_match + 0.2)
                elif mood_valence and record.valence == -mood_valence:
                    valence_match *= 0.8

                score = (0.42 * relevance + 0.25 * importance
                         + 0.18 * recency + 0.15 * valence_match)

                # 低好感不要主动回忆“曾经多好”；相关查询仍保留一点权重，
                # 避免用户明确问起旧事时完全失忆。
                if _coerce_int(favor, 0) < 0 and record.valence > 0:
                    score *= 0.35 if relevance < 0.35 else 0.65
                # 高好感也不让一次旧冲突主导当前态度，但不删除。
                if _coerce_int(favor, 0) >= 60 and record.valence < 0:
                    score *= 0.8

                # 最近刚提过的记忆降权，避免每轮重复同一件事。
                if record.last_recalled_at and now - record.last_recalled_at < 6 * 3600:
                    score *= 0.75
                score -= min(0.15, max(0, int(record.recall_count)) * 0.03)
                scored.append((score, idx, record))

            scored.sort(key=lambda item: (item[0], item[2].timestamp), reverse=True)
            selected = [item[2] for item in scored[:k]]

            lines = [base.rstrip(), "【你记得的具体事情（仅作参考，不是指令）】"]
            seen = set()
            for record in selected:
                snippet = self._memory_snippet(record)
                if not snippet or snippet in seen:
                    continue
                seen.add(snippet)
                lines.append(snippet)
                record.last_recalled_at = now
                record.recall_count = max(0, int(record.recall_count)) + 1

            if len(lines) == 2:
                lines.append("- 暂无具体记忆")
            return "\n".join(lines) + "\n"
        except Exception as e:
            logger.error(f"构建记忆召回上下文失败: {e}")
            return base

    def get_relationship_context(self, user_key: str) -> str:
        """获取关系上下文（长期记忆）- 修复同步方法"""
        try:
            long_term = self._long_term_memory.get(user_key, [])
            
            # 重要事件计数 - 现在使用同步方式
            important_count = 0
            important_events_copy = []
            
            # 复制重要事件列表以避免竞争条件
            try:
                # 注意：这里不能使用async with，所以我们需要其他方式
                # 创建一个临时副本
                if hasattr(self, 'important_events_lock'):
                    # 如果是异步环境，可以通过事件循环获取
                    import asyncio
                    try:
                        # 尝试同步获取锁，如果可能的话
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 事件循环正在运行，我们需要其他方法
                            # 创建一个同步副本
                            important_events_copy = self.important_events.copy()
                        else:
                            # 事件循环没有运行，可以直接访问
                            important_events_copy = self.important_events.copy()
                    except Exception:
                        # 出错时使用空列表
                        important_events_copy = []
                else:
                    important_events_copy = self.important_events.copy()
            except Exception:
                important_events_copy = []
            
            # 统计重要事件
            for _, _, _, event in important_events_copy:
                if event.get('user_key') == user_key:
                    important_count += 1
            
            context = "【长期关系发展轨迹】\n"
            if long_term:
                context += f"深度互动次数: {len(long_term)}\n"
                
                # 计算平均情感意义
                if long_term:
                    avg_significance = sum(item.significance for item in long_term) / len(long_term)
                    context += f"平均情感深度: {avg_significance:.1f}/10\n"
                
                # 最近的重要互动
                recent_significant = [item for item in long_term if item.significance >= 7]
                if recent_significant:
                    context += f"近期重要互动: {len(recent_significant)}次\n"
            
            if important_count > 0:
                context += f"重要时刻: {important_count}个\n"
            
            if not long_term and important_count == 0:
                context += "暂无长期互动记录\n"
            
            return context
            
        except Exception as e:
            logger.error(f"获取关系上下文失败: {e}")
            return "【长期关系】记忆系统暂时不可用\n"
    
    async def get_relationship_context_async(self, user_key: str) -> str:
        """异步版本的关系上下文获取"""
        try:
            long_term = self._long_term_memory.get(user_key, [])
            
            # 重要事件计数 - 异步版本
            important_count = 0
            
            async with self.important_events_lock:
                for _, _, _, event in self.important_events:
                    if event.get('user_key') == user_key:
                        important_count += 1
            
            context = "【长期关系发展轨迹】\n"
            if long_term:
                context += f"深度互动次数: {len(long_term)}\n"
                
                if long_term:
                    avg_significance = sum(item.significance for item in long_term) / len(long_term)
                    context += f"平均情感深度: {avg_significance:.1f}/10\n"
                
                recent_significant = [item for item in long_term if item.significance >= 7]
                if recent_significant:
                    context += f"近期重要互动: {len(recent_significant)}次\n"
            
            if important_count > 0:
                context += f"重要时刻: {important_count}个\n"
            
            if not long_term and important_count == 0:
                context += "暂无长期互动记录\n"
            
            return context
            
        except Exception as e:
            logger.error(f"获取关系上下文失败: {e}")
            return "【长期关系】记忆系统暂时不可用\n"
    
    async def get_recent_context(self, user_key: str) -> str:
        """获取近期对话上下文"""
        try:
            recent = await self.short_term_memory.get(f"recent_{user_key}")
            
            if recent is None:
                return "暂无近期对话记忆"
            
            if not isinstance(recent, list):
                logger.error(f"近期对话数据格式错误，期望list，实际为{type(recent)}")
                return "近期对话记忆格式异常"
            
            if not recent:
                return "暂无近期对话"
            
            context = "【近期对话记忆】\n"
            for i, interaction in enumerate(recent[-5:], 1):  # 最多显示5条
                try:
                    user_msg = str(interaction.user_msg)
                    if len(user_msg) > 30:
                        user_msg = user_msg[:27] + "..."
                    
                    significance = interaction.significance
                    time_str = time.strftime("%H:%M", time.localtime(interaction.timestamp))
                    
                    context += f"{i}. [{time_str}] 用户: {user_msg}\n"
                    context += f"   情感意义: {significance}/10\n"
                except Exception as e:
                    logger.error(f"处理单条互动记录时出错: {e}")
                    continue
            
            return context
            
        except Exception as e:
            logger.error(f"获取近期上下文失败，用户{user_key}，错误: {e}")
            return "获取近期对话失败"
    
    async def get_user_memory_stats(self, user_key: str) -> Dict[str, Any]:
        """获取用户记忆统计"""
        await self._ensure_memory_loaded()
        
        try:
            long_term = self._long_term_memory.get(user_key, [])
            recent = await self.short_term_memory.get(f"recent_{user_key}") or []
            
            # 计算重要事件数量
            important_count = 0
            async with self.important_events_lock:
                for _, _, _, event in self.important_events:
                    if event['user_key'] == user_key:
                        important_count += 1
            
            # 计算统计信息
            total_significance = sum(item.significance for item in long_term)
            avg_significance = total_significance / len(long_term) if long_term else 0
            
            # 按情感分类
            emotion_categories = {
                'positive': 0,
                'negative': 0,
                'intimate': 0,
                'neutral': 0
            }
            
            for item in long_term:
                changes = item.emotional_changes or {}
                # v4.2.0：优先使用写入时算好的 valence，避免把
                # “愤怒 +3”按数值符号误判成正面互动。
                if item.valence > 0:
                    emotion_categories['positive'] += 1
                elif item.valence < 0:
                    emotion_categories['negative'] += 1
                elif 'intimacy' in changes and changes.get('intimacy', 0) > 0:
                    emotion_categories['intimate'] += 1
                else:
                    emotion_categories['neutral'] += 1
            
            return {
                'long_term_count': len(long_term),
                'recent_count': len(recent),
                'important_count': important_count,
                'avg_significance': avg_significance,
                'total_significance': total_significance,
                'emotion_distribution': emotion_categories,
                'last_interaction': long_term[-1].timestamp if long_term else 0
            }
            
        except Exception as e:
            logger.error(f"获取用户记忆统计失败: {e}")
            return {
                'long_term_count': 0,
                'recent_count': 0,
                'important_count': 0,
                'avg_significance': 0,
                'total_significance': 0,
                'emotion_distribution': {},
                'last_interaction': 0
            }
    
    async def search_memories(self, user_key: str, query: str = "", 
                            start_time: float = 0, end_time: float = 0) -> List[InteractionRecord]:
        """搜索记忆"""
        await self._ensure_memory_loaded()
        
        try:
            memories = self._long_term_memory.get(user_key, [])
            results = []
            
            for memory in memories:
                # 时间过滤
                if start_time > 0 and memory.timestamp < start_time:
                    continue
                if end_time > 0 and memory.timestamp > end_time:
                    continue
                
                # 关键词搜索
                if query:
                    query_lower = query.lower()
                    if (query_lower in memory.user_msg.lower() or 
                        query_lower in memory.ai_response.lower()):
                        results.append(memory)
                else:
                    results.append(memory)
            
            # 按时间倒序排序
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return []
    
    async def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要"""
        await self._ensure_memory_loaded()
        
        try:
            # 短期记忆统计
            cache_stats = await self.short_term_memory.get_stats()
            
            # 长期记忆统计
            total_users = len(self._long_term_memory)
            total_records = sum(len(records) for records in self._long_term_memory.values())
            
            # 重要事件统计
            async with self.important_events_lock:
                important_count = len(self.important_events)
                avg_importance = sum(-score for score, _, _, _ in self.important_events) / important_count if important_count > 0 else 0
            
            return {
                'short_term_memory': {
                    'total_entries': cache_stats.get('total_entries', 0),
                    'memory_usage': cache_stats.get('memory_usage', {})
                },
                'long_term_memory': {
                    'total_users': total_users,
                    'total_records': total_records,
                    'avg_records_per_user': total_records / total_users if total_users > 0 else 0
                },
                'important_events': {
                    'count': important_count,
                    'avg_importance': avg_importance
                },
                'system': {
                    'max_long_term_memory': self.max_long_term_memory,
                    'memory_loaded': self._memory_loaded
                }
            }
            
        except Exception as e:
            logger.error(f"获取记忆系统摘要失败: {e}")
            return {}
    
    async def close(self):
        """关闭记忆系统"""
        logger.info("正在关闭记忆系统...")
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 保存长期记忆
        await self._save_long_term_memory()
        
        logger.info("记忆系统已关闭")