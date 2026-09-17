# emotion_expert.py
import re
import json
import random
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import asdict
import hashlib

from astrbot.api import logger

from .cache import ShardedTTLCache
from .models import EnhancedEmotionalState
from .constants import TimeConstants

class EmotionAnalysisExpert:
    """情感分析专家 - 完全修复版本"""

    # 单次尝试的最低时间片：剩余预算低于此值就不再发起新的尝试
    MIN_SLICE = 3.0

    def __init__(self, cache: ShardedTTLCache, context=None,
                 secondary_llm_provider: str = None, secondary_llm_model: str = None,
                 bot_name_provider=None, time_budget: float = 45.0,
                 max_providers: int = 3, change_min: int = -10,
                 change_max: int = 5, enable_ai_text_generation: bool = True):
        self.cache = cache
        self.context = context
        self.secondary_llm_provider = secondary_llm_provider
        self.secondary_llm_model = secondary_llm_model
        self.bot_name_provider = bot_name_provider
        self.llm_timeout = 30.0
        self.llm_retry_count = 3
        self.llm_retry_delay = 1.0
        self._llm_available = True  # 跟踪LLM可用性
        self._llm_failures = 0  # 连续失败次数

        # 单次情感变化幅度（v4.0.20 修正）
        #
        # ⚠️ 这两个值以前是**死配置**：配置界面能填、pydantic 能收、
        # config_manager 会校验，但从来没有任何代码读它们 —— 真正生效的是
        # 本文件里写死的 `max(-5, min(5, v))`。用户把「单次变化最大值」设成 3，
        # 实际仍能一次涨 5 点，且完全看不出原因。
        #
        # 现在改为由配置驱动，并用下面的 _clamp_pair 做整对合法性检查：
        # 只有「两个都是数值 且 change_min < change_max」时才采用，
        # 否则整对退回默认值（和 EmotionConstants.configure 同一策略）。
        self.change_min, self.change_max = self._clamp_pair(
            change_min, change_max, -10, 5
        )
        # 是否允许 LLM 生成态度/关系描述文本；关闭时走本地兜底文案
        self.enable_ai_text_generation = bool(enable_ai_text_generation)

        # 情感分析的总时间预算（秒）：预算耗尽即放弃 LLM 分析，走本地 smart_fallback。
        # 预算内按备选链依次尝试，最多 max_providers 个 provider。
        # 这两个值只作用于「情感分析」这条链路，与 AstrBot 主对话的
        # fallback_chat_models 退避机制互不影响（本插件对 provider 配置纯只读）。
        try:
            self.time_budget = max(self.MIN_SLICE, float(time_budget))
        except (TypeError, ValueError):
            self.time_budget = 45.0
        try:
            self.max_providers = max(1, int(max_providers))
        except (TypeError, ValueError):
            self.max_providers = 3

    @staticmethod
    def _clamp_pair(low, high, default_low: int, default_high: int):
        """校验「变化幅度」数值对，非法则整对退回默认值

        策略与 `EmotionConstants.configure` 一致：**整对一起用或整对一起弃**。
        只判 `low < high` 是不够的，还要挡住：
          - 非数值（None / 字符串 / bool）
          - 量级离谱（|v| > 1000）
          - 两个都是正数或都是负数（如 3 / 2），这种组合语义上是错的
        """
        def _ok(v):
            if v is None or isinstance(v, bool):
                return False
            try:
                n = int(v)
            except (TypeError, ValueError):
                return False
            return abs(n) <= 1000

        if not (_ok(low) and _ok(high)):
            return default_low, default_high
        low, high = int(low), int(high)
        if low >= high:
            return default_low, default_high
        # 语义检查：减少幅度应为负、增加幅度应为正
        if not (low <= 0 <= high):
            return default_low, default_high
        return low, high

    async def analyze_and_update_emotion(self, user_key: str, user_message: str, ai_response: str,
                                       current_state: EnhancedEmotionalState,
                                       umo: str = None) -> Dict[str, Any]:
        """情感分析入口 - 增强异常处理

        umo（unified_msg_origin）用于解析该会话实际使用的 provider：
        不传时 AstrBot 会退回读**全局** cmd_config，可能拿到与当前档案
        （如 WebUI 里的 ds-flash）不一致的 provider。
        """
        logger.info(f"情感分析专家被调用: user_key={user_key}, message_length={len(user_message)}")
    
        # 生成更精确的缓存键，避免重复分析相同对话
        message_hash = hashlib.md5(f"{user_message}_{ai_response}".encode()).hexdigest()[:8]
        cache_key = f"emotion_analysis_{user_key}_{message_hash}"
        
        # 尝试从缓存获取
        cached = await self.cache.get(cache_key)
        if cached:
            logger.info("使用缓存的情感分析结果")
            return cached
        
        # 主要分析流程
        analysis_result = None
        try:
            # 首先尝试使用真正的LLM分析
            if self._llm_available:
                analysis_result = await self._call_real_llm_with_retry(user_message, ai_response, current_state, umo)
            
            # 如果LLM分析成功
            if analysis_result:
                updates = self._parse_emotion_analysis(analysis_result, current_state)
                updates['source'] = 'llm_analysis'
                updates['llm_available'] = True
                
                # 重置失败计数
                self._llm_failures = 0
                
            else:
                # LLM分析失败或不可用
                self._llm_failures += 1
                
                # 如果连续失败超过3次，暂时禁用LLM
                if self._llm_failures >= 3:
                    self._llm_available = False
                    logger.error("LLM连续失败3次，暂时禁用LLM分析")
                
                # 使用智能后备方案
                updates = self._generate_smart_fallback(user_message, ai_response, current_state)
                updates['source'] = 'smart_fallback'
                updates['llm_available'] = False
                
        except Exception as e:
            logger.error(f"情感分析过程发生异常: {e}")
            # 使用紧急后备方案
            updates = self._generate_emergency_fallback(user_message, ai_response, current_state)
            updates['source'] = 'emergency_fallback'
            updates['llm_available'] = False
        
        # 确保返回结果包含所有必要字段
        updates = self._ensure_updates_completeness(updates, current_state)
        
        # 缓存结果（短期缓存）
        await self.cache.set(cache_key, updates, ttl=TimeConstants.ONE_HOUR)
        logger.info(f"情感分析完成: {updates.get('source', 'unknown')}")
        
        return updates

    async def _call_real_llm_with_retry(self, user_message: str, ai_response: str, 
                                      state: EnhancedEmotionalState,
                                      umo: str = None) -> Optional[str]:
        """在总时间预算内，沿备选链依次尝试各 provider。

        旧实现是「同一个 provider 重试 llm_retry_count 次」，最坏
        llm_retry_count × llm_timeout ≈ 90s 才降级，而且主 provider 一旦
        不可用就完全没有冗余。现在改为：

        1. 构造备选链（辅助LLM → 会话主LLM → 档案里的 fallback_chat_models
           → 其它 provider，按 id 去重，见 `_build_provider_chain`）；
        2. 在 `time_budget` 秒的总预算内，最多尝试 `max_providers` 次；
        3. 单次时间片 = min(llm_timeout, 预算 / max_providers)，且不超过剩余预算；
        4. 任一成功即返回；全部失败或预算耗尽则返回 None，由调用方降级到
           smart_fallback。

        ⚠️ 这里只「读」provider 实例与 provider_settings，不修改任何配置，
        因此不会影响 AstrBot 主对话自身的退避重试机制。
        """
        chain = self._build_provider_chain(umo)
        if not chain:
            logger.warning("没有可用的LLM提供商")
            return None

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.time_budget

        attempts = list(chain[: self.max_providers])
        if len(attempts) == 1:
            # 只有一个候选时，保留原有的「同一 provider 重试」语义
            attempts = attempts * max(1, min(self.llm_retry_count, self.max_providers))

        # 单次时间片：预算均分给 max_providers 次尝试，且不超过 llm_timeout
        slice_ = min(
            self.llm_timeout,
            max(self.MIN_SLICE, self.time_budget / self.max_providers),
        )

        prompt = self._build_emotion_analysis_prompt(user_message, ai_response, state)
        logger.info(
            f"情感分析备选链: {[self._get_provider_name(p) for p in chain]} | "
            f"预算 {self.time_budget:.0f}s, 最多 {len(attempts)} 次尝试, 每次 {slice_:.1f}s"
        )

        previous = None
        for index, provider in enumerate(attempts, start=1):
            remaining = deadline - loop.time()
            if remaining <= self.MIN_SLICE:
                logger.warning(f"情感分析时间预算耗尽（剩余 {remaining:.1f}s），停止尝试")
                break

            # 同一 provider 连续重试时保留退避；换 provider 则无需等待
            if provider is previous and self.llm_retry_delay > 0:
                if remaining > self.llm_retry_delay + self.MIN_SLICE:
                    await asyncio.sleep(self.llm_retry_delay)
            previous = provider

            timeout = min(slice_, max(self.MIN_SLICE, deadline - loop.time()))
            name = self._get_provider_name(provider)
            logger.warning(f"情感分析尝试 [{index}/{len(attempts)}] provider={name} 超时={timeout:.1f}s")

            try:
                result = await self._execute_llm_call(provider, prompt, timeout=timeout)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"情感分析 provider [{name}] 调用异常: {e}")
                result = None

            if result and len(result) > 10:  # 确保有足够的返回内容
                logger.info(f"情感分析成功: provider={name}")
                return result

            logger.warning(f"provider [{name}] 未返回有效内容，切换下一个")

        logger.error("情感分析备选链全部失败，降级到本地兜底")
        return None

    def _get_all_providers(self) -> List[Any]:
        """读取全部 provider 实例；context 不可用或异常时返回空列表"""
        if self.context is None:
            return []
        getter = getattr(self.context, "get_all_providers", None)
        if not callable(getter):
            return []
        try:
            return list(getter() or [])
        except Exception as e:
            logger.error(f"读取 provider 列表失败: {e}")
            return []

    def _get_provider_by_id(self, provider_id: str) -> Optional[Any]:
        """按 provider id 取实例；取不到返回 None（不抛异常）

        与 AstrBot 自身解析 fallback_chat_models 的方式一致
        （astr_main_agent.py 里用的是 plugin_context.get_provider_by_id）。
        """
        if not provider_id or self.context is None:
            return None
        getter = getattr(self.context, "get_provider_by_id", None)
        if not callable(getter):
            return None
        try:
            return getter(provider_id)
        except Exception as e:
            logger.error(f"按 id 解析 provider 失败 [{provider_id}]: {e}")
            return None

    def _resolve_fallback_provider_ids(self, umo: str = None) -> List[str]:
        """读取当前会话档案里配置的 fallback_chat_models（只读，不修改）

        provider_settings 取自 umo 解析后的档案，因此不同会话可以有不同的链。
        任何异常都退化为空列表，绝不影响情感分析主流程。
        """
        if self.context is None:
            return []
        getter = getattr(self.context, "get_config", None)
        if not callable(getter):
            return []
        try:
            cfg = getter(umo)
            if cfg is None:
                return []
            provider_settings = cfg.get("provider_settings", {})
            if not isinstance(provider_settings, dict):
                return []
            ids = provider_settings.get("fallback_chat_models", [])
            if not isinstance(ids, (list, tuple)):
                return []
            return [str(i).strip() for i in ids if isinstance(i, str) and i.strip()]
        except Exception as e:
            logger.error(f"读取档案退避链失败: {e}")
            return []

    def _build_provider_chain(self, umo: str = None) -> List[Any]:
        """构造情感分析的备选 provider 链（有序、按 id 去重）

        顺序：
          1. `_find_target_provider` 的结果 —— 与改动前的首选完全一致
             （辅助LLM → 会话主LLM → deepseek/default → providers[0]）
          2. 当前会话的主 LLM —— 显式配置了辅助 LLM 时，第 1 步只会返回辅助
             LLM，这里补上主 LLM 作为第二候选（按 id 去重，重复时自动忽略）
          3. 当前档案配置的 fallback_chat_models（复用用户已有配置，只读）
          4. 其余全部 provider（保持 get_all_providers 的顺序）

        后两项保证即使档案没配退避链，也有兜底候选可选。
        """
        providers = self._get_all_providers()
        if not providers:
            return []

        chain: List[Any] = []
        seen = set()

        def add(provider) -> None:
            if provider is None:
                return
            pid, _ = self._get_provider_meta(provider)
            key = pid or f"obj:{id(provider)}"
            if key in seen:
                return
            seen.add(key)
            chain.append(provider)

        # 1. 首选（与旧行为一致）
        try:
            add(self._find_target_provider(providers, umo))
        except Exception as e:
            logger.error(f"首选 provider 解析失败: {e}")

        # 2. 会话主 LLM（辅助 LLM 已命中时的第二候选）
        add(self._get_main_provider(umo))

        # 3. 档案里配置的退避链
        for provider_id in self._resolve_fallback_provider_ids(umo):
            add(self._get_provider_by_id(provider_id))

        # 4. 其余全部
        for provider in providers:
            add(provider)

        return chain

    async def _call_real_llm(self, user_message: str, ai_response: str,
                             state: EnhancedEmotionalState,
                             umo: str = None) -> Optional[str]:
        """只调用「首选 provider」一次（调试/单发入口）

        正式路径是 `_call_real_llm_with_retry` 的备选链；这里不做跨 provider
        切换，方便单独验证首选 provider 是否可用。
        """
        if not self.context:
            logger.warning("没有context，无法调用LLM")
            return None
        
        providers = self._get_all_providers()
        if not providers:
            logger.warning("没有可用的LLM提供商")
            return None
        
        # 确定目标提供商
        target_provider = self._find_target_provider(providers, umo)
        if not target_provider:
            logger.warning("找不到目标LLM提供商")
            return None
        
        # 构建提示词
        prompt = self._build_emotion_analysis_prompt(user_message, ai_response, state)
        
        try:
            # 实际调用LLM
            result = await self._execute_llm_call(target_provider, prompt)
            return result
        except Exception as e:
            logger.error(f"LLM调用执行失败: {e}")
            return None

    def _get_provider_meta(self, provider) -> Tuple[str, str]:
        """读取 provider 的 (id, model)。

        AstrBot 的 Provider 并没有 `name` 属性，元信息要通过 meta() 获取。
        早期实现直接读 provider.name，取不到就退化成类名（如
        ProviderOpenAIOfficial），于是所有按名称匹配 provider 的逻辑全部失效，
        永远落到 providers[0]。这里做兼容读取，并保证任何情况下都不抛异常。
        """
        pid = ""
        model = ""
        meta = getattr(provider, "meta", None)
        try:
            if callable(meta):
                meta = meta()
            if meta is not None:
                pid = str(getattr(meta, "id", "") or "")
                model = str(getattr(meta, "model", "") or "")
        except Exception as e:
            logger.error(f"读取 provider 元信息失败: {e}")
        if not pid:
            # 兼容非标准对象：退化为 name 属性或类名，仅用于日志与兜底
            pid = str(getattr(provider, "name", "") or "")
        return pid, model

    def _get_provider_name(self, provider) -> str:
        """获取提供商名称（用于日志输出）"""
        pid, model = self._get_provider_meta(provider)
        if pid:
            return pid
        if model:
            return model
        return getattr(provider, "__class__", type(provider)).__name__ or "未知"

    def _get_main_provider(self, umo: str = None) -> Optional[Any]:
        """获取当前会话的主 LLM；取不到时返回 None（不抛异常）

        必须带上 umo：AstrBot 的 get_using_provider(umo=None) 会回退到读全局
        cmd_config.json 的 default_provider_id，而当前生效的是 WebUI 里的
        配置档案（如 ds-flash），两者可能指向完全不同的 provider。
        """
        if self.context is None:
            return None
        try:
            getter = getattr(self.context, "get_using_provider", None)
            if not callable(getter):
                return None
            try:
                return getter(umo)
            except TypeError:
                # 兼容不接受参数的旧版签名
                return getter()
        except Exception as e:
            logger.error(f"获取主LLM失败，回退到名称匹配: {e}")
            return None

    def _find_target_provider(self, providers: List, umo: str = None) -> Optional[Any]:
        """按优先级选择情感分析所使用的 provider

        1. 配置了 secondary_llm_provider → 按 id / model 匹配
        2. 留空 → 使用当前会话的主 LLM
           （与 _conf_schema.json 中「留空则使用主LLM」的说明保持一致）
        3. 主 LLM 不可用 → 名称含 deepseek / default 的 provider
        4. 兜底 → 第一个可用 provider
        """
        if not providers:
            return None

        # 1. 显式配置的辅助 LLM
        if self.secondary_llm_provider:
            target = self.secondary_llm_provider.strip().lower()
            if target:
                for provider in providers:
                    pid, model = self._get_provider_meta(provider)
                    if target in pid.lower() or target in model.lower():
                        logger.info(f"找到配置的辅助LLM提供商: {pid}")
                        return provider
                logger.warning(f"未匹配到辅助LLM提供商 [{self.secondary_llm_provider}]，回退到主LLM")

        # 2. 未配置 → 使用主 LLM
        main_provider = self._get_main_provider(umo)
        if main_provider is not None:
            pid, _ = self._get_provider_meta(main_provider)
            logger.info(f"使用主LLM进行情感分析: {pid}")
            return main_provider

        # 3. 名称含 deepseek / default
        for provider in providers:
            pid, model = self._get_provider_meta(provider)
            haystack = f"{pid} {model}".lower()
            if "deepseek" in haystack or "default" in haystack:
                logger.info(f"找到DeepSeek提供商: {pid}")
                return provider

        # 4. 兜底：第一个可用的
        pid, _ = self._get_provider_meta(providers[0])
        logger.info(f"使用第一个可用提供商: {pid}")
        return providers[0]

    def _target_model(self) -> Optional[str]:
        """返回情感分析要指定的模型名；未配置时返回 None（沿用 provider 自身模型）。

        早期实现完全没有使用 secondary_llm_model，该配置项形同虚设。
        """
        model = (self.secondary_llm_model or "").strip()
        return model or None

    async def _execute_llm_call(self, provider, prompt: str,
                                timeout: Optional[float] = None) -> Optional[str]:
        """执行LLM调用 - 完整实现

        timeout 为 None 时沿用 self.llm_timeout（保持旧调用方的行为不变）；
        备选链会传入按剩余预算裁剪过的时间片。
        """
        if timeout is None:
            timeout = self.llm_timeout
        provider_name = self._get_provider_name(provider)
        logger.info(f"使用LLM提供商 [{provider_name}] 进行情感分析")
        
        # 记录提示词长度（用于调试）
        logger.info(f"情感分析提示词长度: {len(prompt)} 字符")
        
        try:
            # 尝试调用text_chat方法（异步）
            if hasattr(provider, 'text_chat') and asyncio.iscoroutinefunction(provider.text_chat):
                logger.info(f"调用 {provider_name}.text_chat()")
                result = await asyncio.wait_for(
                    provider.text_chat(prompt, model=self._target_model()),
                    timeout=timeout
                )
                text = self._extract_response_text(result)
                if text and len(text) > 10:
                    logger.info(f"LLM情感分析成功，响应长度: {len(text)}")
                    return text
                else:
                    logger.warning("LLM返回空或过短的响应")
                    return None
            
            # 尝试调用chat_completion方法（异步）
            elif hasattr(provider, 'chat_completion') and asyncio.iscoroutinefunction(provider.chat_completion):
                logger.info(f"调用 {provider_name}.chat_completion()")
                
                # 构建消息
                messages = [{"role": "user", "content": prompt}]
                result = await asyncio.wait_for(
                    provider.chat_completion(messages=messages),
                    timeout=timeout
                )
                text = self._extract_response_text(result)
                if text and len(text) > 10:
                    logger.info(f"LLM情感分析成功，响应长度: {len(text)}")
                    return text
            
            # 尝试同步方法
            elif hasattr(provider, 'text_chat') and not asyncio.iscoroutinefunction(provider.text_chat):
                logger.info(f"调用同步方法 {provider_name}.text_chat()")
                result = provider.text_chat(prompt, model=self._target_model())
                text = self._extract_response_text(result)
                if text and len(text) > 10:
                    return text
            
            logger.warning(f"提供商 {provider_name} 不支持已知的调用方法")
            return None
            
        except asyncio.TimeoutError:
            logger.warning(f"LLM调用超时 ({timeout:.1f}秒)")
            return None
        except Exception as e:
            logger.error(f"LLM调用异常: {e}")
            return None

    def _extract_response_text(self, response_obj) -> str:
        """从LLM响应对象中提取文本"""
        if isinstance(response_obj, str):
            return response_obj.strip()
        
        # 尝试常见属性
        text_attrs = ['completion_text', 'text', 'content', 'response', 'result', 'message', 'choices']
        
        for attr in text_attrs:
            if hasattr(response_obj, attr):
                val = getattr(response_obj, attr)
                if isinstance(val, str):
                    return val.strip()
                elif isinstance(val, list) and len(val) > 0:
                    # 处理choices数组
                    first_choice = val[0]
                    if hasattr(first_choice, 'message'):
                        msg = first_choice.message
                        if hasattr(msg, 'content'):
                            return msg.content.strip()
        
        # 尝试字典访问
        if isinstance(response_obj, dict):
            for attr in text_attrs:
                if attr in response_obj:
                    val = response_obj[attr]
                    if isinstance(val, str):
                        return val.strip()
                    elif isinstance(val, list) and len(val) > 0:
                        if 'message' in val[0] and 'content' in val[0]['message']:
                            return val[0]['message']['content'].strip()
        
        # 最后尝试字符串转换
        try:
            text = str(response_obj).strip()
            if text and len(text) > 10:
                return text
        except:
            pass
        
        return ""

    def _generate_smart_fallback(self, user_message: str, ai_response: str, state: EnhancedEmotionalState) -> Dict[str, Any]:
        """智能后备方案 - 返回字典而非字符串"""
        user_lower = user_message.lower()
        resp_lower = ai_response.lower()
        
        # 情感关键词分析
        positive_words = ['好', '开心', '高兴', '谢谢', '感谢', '喜欢', '爱', '不错', '棒', '可爱', '漂亮', '美丽', '相信']
        negative_words = ['讨厌', '生气', '愤怒', '烦', '恨', '滚', '傻', '笨', '蠢', '垃圾', '不愿意']
        intimate_words = ['想你', '想念', '关心', '担心', '在乎', '重要', '宝贝', '亲爱的', '搞好关系']
        
        # 计算情感权重
        pos_weight = sum(3 for word in positive_words if word in user_lower) + \
                    sum(1 for word in positive_words if word in resp_lower)
        neg_weight = sum(3 for word in negative_words if word in user_lower) + \
                    sum(1 for word in negative_words if word in resp_lower)
        int_weight = sum(2 for word in intimate_words if word in user_lower) + \
                    sum(1 for word in intimate_words if word in resp_lower)
        
        # 基于权重生成响应
        emotion_updates = {}
        
        if neg_weight > pos_weight and neg_weight > 0:
            # 负面互动
            neg_strength = min(3, neg_weight)
            emotion_updates = {
                "favor": -neg_strength,
                "intimacy": -1,
                "sadness": 2,
                "anger": 1,
                "disgust": 1
            }
            relationship = "关系紧张"
            attitude = "谨慎回应"
            
        elif pos_weight > neg_weight and pos_weight > 0:
            # 正面互动
            pos_strength = min(3, pos_weight)
            intimacy_boost = 2 if int_weight > 0 else 1
            emotion_updates = {
                "favor": pos_strength,
                "intimacy": intimacy_boost,
                "joy": 2,
                "trust": 1,
                "anticipation": 1
            }
            relationship = "友好的对话伙伴"
            attitude = "愉快开放的交流"
            
        elif int_weight > 0:
            # 亲密互动
            emotion_updates = {
                "favor": 1,
                "intimacy": 3,
                "joy": 2,
                "trust": 2,
                "anticipation": 1
            }
            relationship = "亲密的朋友"
            attitude = "温暖关怀的交流"
            
        else:
            # 中性互动 - 小幅正面
            emotion_updates = {
                "favor": 0,
                "intimacy": 0,
                "anticipation": 1
            }
            relationship = "平常的交流对象"
            attitude = "标准回应"
        
        return {
            **emotion_updates,
            "relationship_text": relationship,
            "attitude_text": attitude
        }

    def _generate_emergency_fallback(self, user_message: str, ai_response: str, state: EnhancedEmotionalState) -> Dict[str, Any]:
        """紧急后备方案 - 只在完全失败时使用"""
        # 默认产生小幅正面情感变化
        return {
            'favor': 1,
            'intimacy': 1,
            'joy': 1,
            'trust': 0,
            'fear': 0,
            'surprise': 0,
            'sadness': 0,
            'disgust': 0,
            'anger': 0,
            'anticipation': 1,
            'relationship_text': "正常关系",
            'attitude_text': "友好交流"
        }

    def _build_emotion_analysis_prompt(self, user_msg: str, bot_msg: str, state: EnhancedEmotionalState) -> str:
        """构建生动的情感分析提示词"""
        # 使用 bot 人设名代替"AI"字样（未解析到时兜底"AI"，写入/显示层会再次清洗）
        bot_name = (self.bot_name_provider() if self.bot_name_provider else "AI") or "AI"
        return f"""你是一个情感分析专家，请分析以下对话的情感变化，输出JSON格式的分析结果。

对话内容：
用户：「{user_msg}」
{bot_name}：「{bot_msg}」

当前用户情感状态：
- 好感度：{state.favor}（范围：-100到100）
- 亲密度：{state.intimacy}（范围：0到100）
- 互动次数：{state.stats.total_count}次
- 正面互动比例：{state.stats.positive_ratio:.1f}%

【情感数值变化范围】
请为以下情感维度分配-2到+2之间的整数值：
- 好感度 (favor): 基于对话的情感倾向
- 亲密度 (intimacy): 基于关系的亲密程度
- 喜悦 (joy): 愉快、开心的程度
- 信任 (trust): 信任、可靠的程度
- 恐惧 (fear): 害怕、担忧的程度
- 惊讶 (surprise): 惊讶、意外的程度
- 悲伤 (sadness): 伤心、难过的程度
- 厌恶 (disgust): 厌恶、反感的程度
- 愤怒 (anger): 生气、愤怒的程度
- 期待 (anticipation): 期待、盼望的程度

【关系描述要求】
- 用不超过 20 个字概括双方的关系性质，保持生动有趣
- 考虑当前好感度、亲密度和互动历史
- 必须简短！禁止使用逗号连接的长句，禁止超过 20 字
- 保持自然、符合人类社交常识
- 若提到双方，用「{bot_name}」称呼 bot 一方，不要出现"AI"字样

【态度描述要求】
- 用不超过 20 个字描述 {bot_name} 对用户的回应态度或互动方式
- 体现情感倾向和互动风格
- 必须简短！禁止使用逗号连接的长句，禁止超过 20 字
- 若提到双方，用「{bot_name}」称呼 bot 一方，不要出现"AI"字样

【输出格式】
请输出严格的JSON格式：
{{
  "emotion_updates": {{
    "favor": 整数变化值,
    "intimacy": 整数变化值,
    "joy": 整数变化值,
    "trust": 整数变化值,
    "fear": 整数变化值,
    "surprise": 整数变化值,
    "sadness": 整数变化值,
    "disgust": 整数变化值,
    "anger": 整数变化值,
    "anticipation": 整数变化值
  }},
  "relationship": "关系描述（不超过20字）",
  "attitude": "态度描述（不超过20字）"
}}

注意：
- 如果对话情感不明显，可以设置部分值为0。
- relationship 和 attitude 必须简短（不超过20个字），禁止使用长句和标点堆砌。"""

    def _parse_emotion_analysis(self, analysis_text: str, current_state: EnhancedEmotionalState) -> Dict[str, Any]:
        """解析情感分析结果 - 增强版本"""
        updates = {}
        
        try:
            # 清理和提取JSON
            cleaned_text = self._clean_json_response(analysis_text)
            
            # 尝试解析JSON
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_str = self._fix_common_json_errors(json_str)
                
                data = json.loads(json_str)
                
                # 验证必需字段
                if 'emotion_updates' not in data:
                    raise ValueError("缺少emotion_updates字段")
                
                # 解析数值更新
                emotion_updates = data['emotion_updates']
                valid_emotions = ['favor', 'intimacy', 'joy', 'trust', 'fear', 
                                 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
                
                for emotion in valid_emotions:
                    if emotion in emotion_updates:
                        try:
                            value = emotion_updates[emotion]
                            # 确保是整数且在合理范围内
                            if isinstance(value, (int, float)):
                                int_value = int(value)
                                # 限制变化范围（v4.0.20：好感度改由配置驱动）
                                #
                                # ⚠️ 只有 favor 用配置的 change_min/change_max：
                                # 这两个字段的定义就是「**好感度**单次变化幅度」
                                # （原作者 v3.30 也只对 favor 做此钳制），
                                # 套到 intimacy 上会擅自改变亲密度的一次性变化幅度。
                                # intimacy 与其余情绪维度沿用原先的固定幅度。
                                if emotion == 'favor':
                                    int_value = max(self.change_min,
                                                    min(self.change_max, int_value))
                                elif emotion == 'intimacy':
                                    int_value = max(-5, min(5, int_value))
                                else:
                                    int_value = max(-3, min(3, int_value))
                                updates[emotion] = int_value
                        except (ValueError, TypeError):
                            updates[emotion] = 0
                    else:
                        updates[emotion] = 0  # 缺失的情感设为0
                
                # 解析文本描述
                #
                # ⚠️ 「启用 AI 自主生成文本描述」开关（v4.0.20 修正）
                # 这个开关以前是死配置：关掉它，态度/关系描述仍会由 LLM 生成。
                # 现在关闭时**不改写**这两个字段（留空即被下游替换为当前值），
                # 于是用户自己用 /设置态度、/设置关系 设的文案不会被 AI 覆盖。
                if self.enable_ai_text_generation:
                    if 'relationship' in data and data['relationship']:
                        updates['relationship_text'] = str(data['relationship']).strip()[:20]  # 限制长度
                    else:
                        updates['relationship_text'] = "正常关系"

                    if 'attitude' in data and data['attitude']:
                        updates['attitude_text'] = str(data['attitude']).strip()[:20]  # 限制长度
                    else:
                        updates['attitude_text'] = "友好交流"
                else:
                    # 保留当前描述，不覆盖。
                    # ⚠️ 参数名是 current_state（不是 state）—— 写成 state 会抛
                    # NameError，被本方法外层宽泛的 `except Exception` 吞掉，
                    # 静默退化到「文本提取」兜底，返回写死的「正常关系/友好交流」，
                    # 看起来就像开关完全没生效。
                    updates['relationship_text'] = current_state.descriptions.relationship
                    updates['attitude_text'] = current_state.descriptions.attitude
                
                logger.info(f"成功解析JSON情感分析结果，包含 {len(updates)} 个更新")
                return updates
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 文本: {analysis_text[:100]}")
        except KeyError as e:
            logger.warning(f"JSON字段缺失: {e}")
        except Exception as e:
            logger.error(f"解析情感分析结果失败: {e}")
        
        # JSON解析失败，尝试从文本提取
        logger.error("JSON解析失败，尝试文本提取")
        return self._extract_updates_from_text(analysis_text)

    def _clean_json_response(self, text: str) -> str:
        """清理JSON响应"""
        # 移除可能的代码块标记
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```\s*', '', text)
        
        # 移除JSON之前的说明文字
        text = re.sub(r'^[^{]*', '', text)
        text = re.sub(r'[^}]*$', '', text)
        
        return text.strip()

    def _fix_common_json_errors(self, json_str: str) -> str:
        """修复常见的JSON格式错误"""
        # 修复未加引号的键
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        
        # 修复单引号
        json_str = json_str.replace("'", '"')
        
        # 修复多余的逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保大括号匹配
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        return json_str

    def _extract_updates_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取更新信息（后备方法）"""
        updates = {
            'favor': 0,
            'intimacy': 0,
            'joy': 0,
            'trust': 0,
            'fear': 0,
            'surprise': 0,
            'sadness': 0,
            'disgust': 0,
            'anger': 0,
            'anticipation': 0,
            'relationship_text': "正常关系",
            'attitude_text': "友好交流"
        }
        
        text_lower = text.lower()
        
        # 简单的情感关键词映射
        emotion_patterns = [
            ('favor', ['好感', '喜欢', '欣赏', '满意', 'positive']),
            ('intimacy', ['亲密', '亲近', '密切', '亲密感', 'intimacy']),
            ('joy', ['开心', '高兴', '愉快', '欢乐', '微笑', 'joy', 'happy']),
            ('trust', ['信任', '相信', '可靠', '安心', 'trust']),
            ('fear', ['害怕', '恐惧', '担心', '紧张', 'fear', 'scared']),
            ('surprise', ['惊讶', '惊奇', '意外', '吃惊', 'surprise']),
            ('sadness', ['悲伤', '伤心', '难过', '沮丧', 'sadness', 'sad']),
            ('disgust', ['厌恶', '讨厌', '反感', '恶心', 'disgust']),
            ('anger', ['生气', '愤怒', '恼火', '气愤', 'anger', 'angry']),
            ('anticipation', ['期待', '期望', '盼望', 'anticipation'])
        ]
        
        for emotion, keywords in emotion_patterns:
            for keyword in keywords:
                if keyword in text_lower:
                    # 根据关键词强度调整值
                    if '非常' in text_lower or '特别' in text_lower or 'extremely' in text_lower:
                        updates[emotion] = 2 if emotion in ['favor', 'intimacy'] else 1
                    elif '有点' in text_lower or '稍微' in text_lower or 'slightly' in text_lower:
                        updates[emotion] = 1 if emotion in ['favor', 'intimacy'] else 0
                    else:
                        updates[emotion] = 1 if emotion in ['favor', 'intimacy'] else 0
                    break
        
        return updates

    def _ensure_updates_completeness(self, updates: Dict[str, Any], state: EnhancedEmotionalState) -> Dict[str, Any]:
        """确保更新结果包含所有必要字段"""
        required_fields = {
            'favor': 0,
            'intimacy': 0,
            'joy': 0,
            'trust': 0,
            'fear': 0,
            'surprise': 0,
            'sadness': 0,
            'disgust': 0,
            'anger': 0,
            'anticipation': 0
        }
        
        # 确保所有情感字段都存在
        for field, default_value in required_fields.items():
            if field not in updates:
                updates[field] = default_value
        
        # 确保有文本描述
        if 'relationship_text' not in updates:
            updates['relationship_text'] = state.descriptions.relationship
        
        if 'attitude_text' not in updates:
            updates['attitude_text'] = state.descriptions.attitude
        
        # 确保有来源信息
        if 'source' not in updates:
            updates['source'] = 'unknown'
        
        if 'llm_available' not in updates:
            updates['llm_available'] = False
        
        return updates
    
    def reset_llm_availability(self):
        """重置LLM可用性状态"""
        self._llm_available = True
        self._llm_failures = 0
        logger.info("已重置LLM可用性状态")