# tests/test_description_diversity.py
"""v4.1.15 关系 / 态度描述去同质化

线上现象（2026-10-19 生产数据）：261 个用户的描述里，`傲娇` 出现 104 次、
`撒娇` 63 次、`俏皮` 55 次、`互怼` 43 次，大量用户拿到的是同一组人设
套话的排列组合（「傲娇虚拟主播与主人的互怼日常」「互怼互关心的损友」）。

本版做三件事：
1. 提示词：给上一版/最近用过的描述，要求别重复；每轮轮换一个描写角度；
   显式列出高频套话与空话；
2. 解析：LLM 返回套话堆砌、近重复或空字段时，换成按关系档位挑选的兜底；
3. 兜底：`_extract_updates_from_text` / 智能兜底 / 紧急兜底不再写死
   「正常关系 / 友好交流」，统一走分档文案池。

本文件同时锁定「模型层最近描述历史」的写入、去重、上限与旧存档兼容。
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.astrbot_stub  # noqa: F401  (注入 astrbot stub)
import tests.bootstrap  # noqa: F401
from emotionai_pro.description_diversity import (  # noqa: E402
    ATTITUDE_POOLS,
    DESCRIPTION_ANGLES,
    MAX_DESCRIPTION_LENGTH,
    RECENT_DESCRIPTION_LIMIT,
    RELATIONSHIP_POOLS,
    cliche_score,
    description_seed,
    description_similarity,
    ensure_diverse,
    is_cliche,
    is_repeat,
    pick_angle,
    pick_fallback,
    recent_descriptions,
    relationship_band,
    sanitize_description,
)
from emotionai_pro.emotion_expert import EmotionAnalysisExpert  # noqa: E402
from emotionai_pro.models import EnhancedEmotionalState, TextDescriptions  # noqa: E402


def _expert(**kw):
    """最小专家实例：只够跑提示词构建、解析与本地兜底"""
    kw.setdefault("bot_name_provider", lambda: "永雏塔菲")
    return EmotionAnalysisExpert(types.SimpleNamespace(), None, **kw)


def _state(user_key="u1", favor=0, intimacy=0):
    return EnhancedEmotionalState(user_key=user_key, favor=favor, intimacy=intimacy)


def _payload(relationship="一起熬夜改稿的搭档", attitude="先损两句再认真哄",
             favor=1):
    return (
        '{"emotion_updates": {"favor": %d},'
        ' "relationship": "%s", "attitude": "%s"}'
    ) % (favor, relationship, attitude)


def _pool_texts(kind):
    pools = RELATIONSHIP_POOLS if kind == "relationship" else ATTITUDE_POOLS
    for values in pools.values():
        yield from values


class TestSanitizeAndCliche(unittest.TestCase):
    """清洗与套话检测"""

    def test_sanitize_strips_label_quotes_and_whitespace(self):
        self.assertEqual(
            sanitize_description(' 关系： "一起熬夜改稿的搭档"  '),
            "一起熬夜改稿的搭档",
        )
        self.assertEqual(
            sanitize_description("态度描述：【嘴上嫌弃手上照做】"),
            "嘴上嫌弃手上照做",
        )

    def test_sanitize_replaces_unsupported_symbols(self):
        text = sanitize_description("追番/打游戏|睡觉…")
        self.assertNotIn("/", text)
        self.assertNotIn("|", text)
        self.assertNotIn("…", text)
        self.assertIn("、", text)

    def test_sanitize_truncates_to_limit(self):
        text = sanitize_description("好" * 50)
        self.assertEqual(len(text), MAX_DESCRIPTION_LENGTH)

    def test_sanitize_output_is_always_writable(self):
        """清洗结果必须能通过 TextDescriptions 的字符校验，否则 update_* 会抛错"""
        samples = [
            '他说"好"',
            "追番/打游戏|睡觉…",
            "【标签】一起熬夜改稿的搭档",
            "A&B + C",
        ]
        for sample in samples:
            cleaned = sanitize_description(sample)
            self.assertTrue(
                TextDescriptions.is_valid_attitude(cleaned),
                f"清洗后仍不可写: {sample!r} -> {cleaned!r}",
            )

    def test_sanitize_handles_non_string(self):
        self.assertEqual(sanitize_description(None), "")
        self.assertEqual(sanitize_description(123), "123")

    def test_stacked_cliche_is_detected(self):
        self.assertTrue(is_cliche("傲娇虚拟主播与主人的互怼日常"))
        self.assertTrue(is_cliche("互怼互关心的损友"))
        self.assertTrue(is_cliche("傲娇卖萌故意气人"))

    def test_single_strong_term_is_not_cliche(self):
        """单个「傲娇」可能确实是本轮的具体表现，不能一棍子打死"""
        self.assertFalse(is_cliche("傲娇猫娘和她指定的铲屎官"))
        self.assertEqual(cliche_score("傲娇猫娘和她指定的铲屎官"), 2)

    def test_empty_phrases_are_cliche(self):
        self.assertTrue(is_cliche("友好交流"))
        self.assertTrue(is_cliche("正常关系"))
        self.assertTrue(is_cliche("相处愉快"))
        self.assertTrue(is_cliche("陌生人"))
        self.assertTrue(is_cliche("中立"))

    def test_specific_description_is_not_cliche(self):
        self.assertFalse(is_cliche("一起熬夜改稿的搭档"))
        self.assertFalse(is_cliche("刚和好还没完全消气的朋友"))

    def test_prompt_example_echo_is_cliche(self):
        """提示词示例被原样照抄时会变成下一批套话"""
        self.assertTrue(is_cliche("一起蹲开播的老观众"))
        self.assertTrue(is_cliche("认真解释了三遍才放心"))

    def test_similarity_detects_rephrase(self):
        self.assertEqual(description_similarity("一起熬夜改稿的搭档", "一起熬夜改稿的搭档"), 1.0)
        self.assertGreater(
            description_similarity("一起熬夜改稿的搭档", "一起熬夜改稿的伙伴"), 0.5,
        )
        self.assertLess(
            description_similarity("一起熬夜改稿的搭档", "刚认识的礼貌网友"), 0.5,
        )

    def test_is_repeat_uses_previous_and_recent(self):
        self.assertTrue(is_repeat("一起熬夜改稿的搭档", previous="一起熬夜改稿的搭档"))
        self.assertTrue(is_repeat("一起熬夜改稿的搭档", previous="",
                                  recent=["一起熬夜改稿的搭档"]))
        self.assertFalse(is_repeat("刚认识的礼貌网友", previous="一起熬夜改稿的搭档"))


class TestFallbackPools(unittest.TestCase):
    """分档兜底池的质量与选择"""

    def test_pools_cover_all_bands(self):
        expected = {"negative", "distant", "casual", "close", "intimate"}
        self.assertEqual(set(RELATIONSHIP_POOLS), expected)
        self.assertEqual(set(ATTITUDE_POOLS), expected)
        for kind in ("relationship", "attitude"):
            for band, values in (
                (RELATIONSHIP_POOLS if kind == "relationship" else ATTITUDE_POOLS).items()
            ):
                self.assertGreaterEqual(len(values), 6, f"{kind}/{band} 文案太少")

    def test_pool_entries_are_short_and_not_cliche(self):
        for kind in ("relationship", "attitude"):
            for text in _pool_texts(kind):
                self.assertLessEqual(len(text), MAX_DESCRIPTION_LENGTH, text)
                self.assertFalse(is_cliche(text), f"兜底池自己也在套话: {text}")

    def test_pool_entries_are_unique(self):
        for kind in ("relationship", "attitude"):
            texts = list(_pool_texts(kind))
            self.assertEqual(len(texts), len(set(texts)), f"{kind} 兜底池有重复")

    def test_band_boundaries(self):
        self.assertEqual(relationship_band(-1, 100), "negative")
        self.assertEqual(relationship_band(0, 0), "distant")
        self.assertEqual(relationship_band(30, 30), "casual")
        self.assertEqual(relationship_band(60, 60), "close")
        self.assertEqual(relationship_band(100, 100), "intimate")

    def test_pick_fallback_matches_band(self):
        for favor, intimacy, band in (
            (-10, 0, "negative"),
            (0, 0, "distant"),
            (30, 30, "casual"),
            (60, 60, "close"),
            (120, 120, "intimate"),
        ):
            text = pick_fallback("relationship", favor, intimacy, seed=3)
            self.assertIn(text, RELATIONSHIP_POOLS[band])
            text = pick_fallback("attitude", favor, intimacy, seed=4)
            self.assertIn(text, ATTITUDE_POOLS[band])

    def test_pick_fallback_rotates_with_seed(self):
        """同一档位、不同种子必须能挑出不同句子，不能又退回固定文案"""
        texts = {
            pick_fallback("relationship", 60, 60, seed=s) for s in range(8)
        }
        self.assertGreater(len(texts), 1)

    def test_pick_fallback_avoids_recent(self):
        recent = list(RELATIONSHIP_POOLS["distant"][:5])
        text = pick_fallback("relationship", 0, 0, seed=0, avoid=recent)
        self.assertNotIn(text, recent)

    def test_pick_fallback_unknown_kind_falls_back_to_relationship(self):
        text = pick_fallback("unknown", 60, 60, seed=0)
        self.assertIn(text, RELATIONSHIP_POOLS["close"])


class TestEnsureDiverse(unittest.TestCase):
    """套话 / 重复候选的替换逻辑"""

    def test_specific_candidate_is_kept(self):
        state = _state(favor=60, intimacy=60)
        self.assertEqual(
            ensure_diverse("relationship", "一起熬夜改稿的搭档", state),
            "一起熬夜改稿的搭档",
        )

    def test_cliche_candidate_is_replaced(self):
        state = _state(favor=60, intimacy=60)
        text = ensure_diverse("relationship", "傲娇虚拟主播与主人的互怼日常", state)
        self.assertFalse(is_cliche(text))
        self.assertNotEqual(text, "傲娇虚拟主播与主人的互怼日常")

    def test_duplicate_of_specific_previous_is_kept(self):
        """关系是慢变量：模型按提示词保留原说法时不应该被硬改"""
        state = _state(favor=60, intimacy=60)
        state.descriptions.relationship = "一起熬夜改稿的搭档"
        text = ensure_diverse("relationship", "一起熬夜改稿的搭档", state)
        self.assertEqual(text, "一起熬夜改稿的搭档")

    def test_duplicate_of_cliche_previous_is_replaced(self):
        state = _state(favor=60, intimacy=60)
        state.descriptions.relationship = "傲娇虚拟主播与主人的互怼日常"
        text = ensure_diverse("relationship", "傲娇虚拟主播与主人的互怼日常", state)
        self.assertNotEqual(text, "傲娇虚拟主播与主人的互怼日常")
        self.assertFalse(is_cliche(text))

    def test_near_duplicate_of_recent_is_replaced(self):
        state = _state(favor=60, intimacy=60)
        state.descriptions.recent_relationship = ["一起熬夜改稿的搭档"]
        text = ensure_diverse("relationship", "一起熬夜改稿的伙伴", state)
        self.assertNotIn(text, ("一起熬夜改稿的伙伴", "一起熬夜改稿的搭档"))

    def test_empty_candidate_uses_fallback(self):
        state = _state(favor=0, intimacy=0)
        text = ensure_diverse("attitude", "", state)
        self.assertIn(text, ATTITUDE_POOLS["distant"])

    def test_fallback_avoids_previous_and_recent(self):
        state = _state(favor=60, intimacy=60)
        state.descriptions.relationship = "一起熬夜改稿的搭档"
        state.descriptions.recent_relationship = ["互相投喂的饭搭子"]
        text = ensure_diverse("relationship", "傲娇虚拟主播与主人的互怼日常", state)
        self.assertNotIn(text, ("一起熬夜改稿的搭档", "互相投喂的饭搭子"))

    def test_relationship_and_attitude_pools_do_not_overlap(self):
        relationship = set(_pool_texts("relationship"))
        attitude = set(_pool_texts("attitude"))
        self.assertFalse(relationship & attitude, "关系/态度兜底池不应共用句子")

    def test_fallback_is_stable_for_same_user(self):
        """同一用户连续命中兜底时必须稳定，不能每轮随机换一句"""
        state = _state(user_key="stable-fallback-u1", favor=60, intimacy=60)
        first = ensure_diverse("relationship", "傲娇虚拟主播与主人的互怼日常", state)
        second = ensure_diverse("relationship", "傲娇虚拟主播与主人的互怼日常", state)
        self.assertEqual(first, second)

    def test_existing_pool_text_is_kept_stable(self):
        """上一版已经是本档兜底时，模型再返回套话也沿用原句"""
        state = _state(user_key="stable-fallback-u2", favor=60, intimacy=60)
        state.descriptions.relationship = RELATIONSHIP_POOLS["close"][3]
        text = ensure_diverse(
            "relationship", "傲娇虚拟主播与主人的互怼日常", state,
        )
        self.assertEqual(text, RELATIONSHIP_POOLS["close"][3])

    def test_pool_text_from_other_band_is_replaced(self):
        """关系档位变了，旧档位的兜底不能继续沿用"""
        state = _state(user_key="stable-fallback-u3", favor=120, intimacy=120)
        state.descriptions.relationship = RELATIONSHIP_POOLS["distant"][0]
        text = ensure_diverse(
            "relationship", "傲娇虚拟主播与主人的互怼日常", state,
        )
        self.assertIn(text, RELATIONSHIP_POOLS["intimate"])

    def test_fallback_differs_across_users(self):
        """不同用户的兜底应从池子的不同位置起步"""
        texts = set()
        for index in range(12):
            state = _state(user_key=f"fallback-user-{index}", favor=60, intimacy=60)
            texts.add(ensure_diverse("relationship", "傲娇虚拟主播与主人的互怼日常", state))
        self.assertGreater(len(texts), 1)


class TestPromptDiversity(unittest.TestCase):
    """提示词：上一版、最近记录、角度、去套话"""

    def test_prompt_includes_previous_and_recent(self):
        state = _state(user_key="prompt-u1", favor=60, intimacy=60)
        state.descriptions.relationship = "一起熬夜改稿的搭档"
        state.descriptions.attitude = "先损两句再认真哄"
        state.descriptions.recent_relationship = ["互相投喂的饭搭子", "一起蹲开播的老观众"]
        state.descriptions.recent_attitude = ["嘴上说不管，还是把链接发来了"]
        prompt = _expert()._build_emotion_analysis_prompt("你好", "你好呀", state)
        self.assertIn("上一版关系描述", prompt)
        self.assertIn("一起熬夜改稿的搭档", prompt)
        self.assertIn("最近用过的关系描述", prompt)
        self.assertIn("互相投喂的饭搭子", prompt)
        self.assertIn("嘴上说不管，还是把链接发来了", prompt)

    def test_prompt_includes_angle_and_anticliche_rules(self):
        state = _state(user_key="prompt-u2", favor=60, intimacy=60)
        prompt = _expert()._build_emotion_analysis_prompt("你好", "你好呀", state)
        self.assertIn("本轮描写角度", prompt)
        self.assertIn("去套话", prompt)
        self.assertIn("傲娇", prompt)
        self.assertIn("正常关系", prompt)
        self.assertIn("不要照抄上一版", prompt)
        self.assertIn(pick_angle(state), prompt)

    def test_prompt_separates_relationship_and_attitude(self):
        prompt = _expert()._build_emotion_analysis_prompt("你好", "你好呀", _state())
        self.assertIn("双方关系的具体样子", prompt)
        self.assertIn("这一轮具体怎么回应", prompt)
        self.assertIn("不能和 relationship 是同一个意思", prompt)

    def test_angle_rotates_with_update_count(self):
        angles = set()
        for count in range(len(DESCRIPTION_ANGLES)):
            state = _state(user_key="rotate-u1")
            state.descriptions.update_count = count
            angles.add(pick_angle(state))
        self.assertGreater(len(angles), 1)
        self.assertTrue(angles <= set(DESCRIPTION_ANGLES))

    def test_angle_is_stable_for_same_state(self):
        state = _state(user_key="stable-u1")
        state.descriptions.update_count = 7
        self.assertEqual(pick_angle(state), pick_angle(state))

    def test_different_users_get_different_angles(self):
        """至少不能所有用户第一轮都拿到同一个角度"""
        angles = set()
        for index in range(12):
            angles.add(pick_angle(_state(user_key=f"user-{index}")))
        self.assertGreater(len(angles), 1)


class TestParseDiversity(unittest.TestCase):
    """解析层：LLM 套话/空字段不能直接落到展示"""

    def test_cliche_llm_text_is_replaced(self):
        state = _state(favor=60, intimacy=60)
        out = _expert()._parse_emotion_analysis(
            _payload("傲娇虚拟主播与主人的互怼日常", "傲娇卖萌故意气人"), state,
        )
        self.assertFalse(is_cliche(out["relationship_text"]))
        self.assertFalse(is_cliche(out["attitude_text"]))

    def test_specific_llm_text_is_kept(self):
        state = _state(favor=60, intimacy=60)
        out = _expert()._parse_emotion_analysis(
            _payload("一起熬夜改稿的搭档", "先损两句再认真哄"), state,
        )
        self.assertEqual(out["relationship_text"], "一起熬夜改稿的搭档")
        self.assertEqual(out["attitude_text"], "先损两句再认真哄")

    def test_blank_fields_do_not_become_empty_phrases(self):
        state = _state(favor=0, intimacy=0)
        out = _expert()._parse_emotion_analysis(
            '{"emotion_updates": {"favor": 1}, "relationship": "", "attitude": ""}',
            state,
        )
        self.assertNotEqual(out["relationship_text"], "正常关系")
        self.assertNotEqual(out["attitude_text"], "友好交流")
        self.assertIn(out["relationship_text"], RELATIONSHIP_POOLS["distant"])
        self.assertIn(out["attitude_text"], ATTITUDE_POOLS["distant"])

    def test_negative_state_uses_negative_pool(self):
        state = _state(favor=-20, intimacy=0)
        out = _expert()._parse_emotion_analysis(
            '{"emotion_updates": {"favor": 0}, "relationship": "", "attitude": ""}',
            state,
        )
        self.assertIn(out["relationship_text"], RELATIONSHIP_POOLS["negative"])
        self.assertIn(out["attitude_text"], ATTITUDE_POOLS["negative"])

    def test_specific_previous_text_is_kept_stable(self):
        state = _state(favor=60, intimacy=60)
        state.descriptions.relationship = "一起熬夜改稿的搭档"
        state.descriptions.attitude = "边回边把话题记下来"
        out = _expert()._parse_emotion_analysis(
            _payload("一起熬夜改稿的搭档", "边回边把话题记下来"), state,
        )
        self.assertEqual(out["relationship_text"], "一起熬夜改稿的搭档")
        self.assertEqual(out["attitude_text"], "边回边把话题记下来")

    def test_ai_text_switch_off_still_preserves_manual_text(self):
        state = _state()
        state.descriptions.relationship = "手工关系"
        state.descriptions.attitude = "手工态度"
        out = _expert(enable_ai_text_generation=False)._parse_emotion_analysis(
            _payload("傲娇虚拟主播与主人的互怼日常", "傲娇卖萌故意气人"), state,
        )
        self.assertEqual(out["relationship_text"], "手工关系")
        self.assertEqual(out["attitude_text"], "手工态度")


class TestLocalFallbackDiversity(unittest.TestCase):
    """三条本地兜底路径都不能写死「正常关系 / 友好交流」"""

    def test_smart_fallback_uses_pool(self):
        expert = _expert()
        state = _state(favor=0, intimacy=0)
        raw = expert._generate_smart_fallback("你好", "", state)
        self.assertIn(raw["relationship_text"], RELATIONSHIP_POOLS["distant"])
        self.assertIn(raw["attitude_text"], ATTITUDE_POOLS["distant"])

    def test_smart_fallback_matches_relation_band(self):
        expert = _expert()
        state = _state(favor=120, intimacy=120)
        raw = expert._generate_smart_fallback("宝贝", "", state)
        self.assertIn(raw["relationship_text"], RELATIONSHIP_POOLS["intimate"])
        self.assertIn(raw["attitude_text"], ATTITUDE_POOLS["intimate"])

    def test_text_extraction_fallback_uses_pool(self):
        expert = _expert()
        state = _state(favor=60, intimacy=60)
        raw = expert._extract_updates_from_text("非常亲密的关系", state)
        self.assertEqual(raw["intimacy"], 2, "文本提取的数值逻辑不应被改动")
        self.assertIn(raw["relationship_text"], RELATIONSHIP_POOLS["close"])
        self.assertIn(raw["attitude_text"], ATTITUDE_POOLS["close"])

    def test_text_extraction_fallback_without_state_still_works(self):
        raw = _expert()._extract_updates_from_text("随便一段文本")
        self.assertTrue(raw["relationship_text"])
        self.assertTrue(raw["attitude_text"])
        self.assertNotEqual(raw["relationship_text"], "正常关系")

    def test_emergency_fallback_uses_pool(self):
        expert = _expert()
        state = _state(favor=0, intimacy=0)
        raw = expert._generate_emergency_fallback("你好", "", state)
        self.assertIn(raw["relationship_text"], RELATIONSHIP_POOLS["distant"])
        self.assertIn(raw["attitude_text"], ATTITUDE_POOLS["distant"])

    def test_emergency_fallback_respects_negative_band(self):
        expert = _expert()
        state = _state(favor=-20, intimacy=0)
        raw = expert._generate_emergency_fallback("滚", "", state)
        self.assertIn(raw["relationship_text"], RELATIONSHIP_POOLS["negative"])
        self.assertIn(raw["attitude_text"], ATTITUDE_POOLS["negative"])


class TestTextDescriptionsHistory(unittest.TestCase):
    """模型层最近描述历史：写入、去重、上限、旧存档兼容"""

    def test_update_records_history(self):
        descriptions = TextDescriptions()
        descriptions.update_relationship("一起熬夜改稿的搭档")
        descriptions.update_attitude("先损两句再认真哄")
        self.assertEqual(descriptions.recent_relationship, ["一起熬夜改稿的搭档"])
        self.assertEqual(descriptions.recent_attitude, ["先损两句再认真哄"])

    def test_history_dedupes_and_caps(self):
        descriptions = TextDescriptions()
        for text in ("甲", "乙", "丙", "丁", "戊", "己", "庚"):
            descriptions.update_relationship(text)
        self.assertEqual(len(descriptions.recent_relationship), RECENT_DESCRIPTION_LIMIT)
        self.assertEqual(
            descriptions.recent_relationship,
            ["丙", "丁", "戊", "己", "庚"],
        )
        # 重复写入只保留最后一次位置
        descriptions.update_relationship("丁")
        self.assertEqual(descriptions.recent_relationship[-1], "丁")
        self.assertEqual(descriptions.recent_relationship.count("丁"), 1)

    def test_old_archive_without_history_loads(self):
        old = {
            "user_key": "legacy-u1",
            "favor": 10,
            "intimacy": 5,
            "descriptions": {
                "attitude": "老的",
                "relationship": "旧的",
                "last_attitude_update": 0,
                "last_relationship_update": 0,
                "update_count": 2,
            },
        }
        state = EnhancedEmotionalState.from_dict(old)
        self.assertEqual(state.descriptions.attitude, "老的")
        self.assertEqual(state.descriptions.recent_attitude, [])
        self.assertEqual(state.descriptions.recent_relationship, [])

    def test_roundtrip_preserves_history(self):
        state = _state(user_key="roundtrip-u1", favor=10, intimacy=5)
        state.descriptions.update_relationship("一起熬夜改稿的搭档")
        state.descriptions.update_attitude("先损两句再认真哄")
        data = state.to_dict()
        self.assertEqual(data["descriptions"]["recent_relationship"], ["一起熬夜改稿的搭档"])
        restored = EnhancedEmotionalState.from_dict(data)
        self.assertEqual(
            restored.descriptions.recent_relationship, ["一起熬夜改稿的搭档"],
        )

    def test_invalid_history_items_are_sanitized(self):
        descriptions = TextDescriptions(
            recent_relationship=["好的", 123, "", "x" * 100],
            recent_attitude=[None, "还行"],
        )
        self.assertEqual(descriptions.recent_relationship[0], "好的")
        self.assertEqual(descriptions.recent_relationship[1], "x" * MAX_DESCRIPTION_LENGTH)
        self.assertEqual(descriptions.recent_attitude, ["还行"])

    def test_history_survives_direct_validation(self):
        descriptions = TextDescriptions(
            recent_relationship=["一起熬夜改稿的搭档", "一起熬夜改稿的搭档"],
        )
        self.assertEqual(
            descriptions.recent_relationship, ["一起熬夜改稿的搭档"],
        )

    def test_recent_descriptions_excludes_current(self):
        state = _state(user_key="recent-u1")
        state.descriptions.relationship = "当前"
        state.descriptions.recent_relationship = ["旧的", "当前"]
        self.assertEqual(recent_descriptions(state, "relationship"), ["旧的"])

    def test_description_seed_changes_with_update_count(self):
        state = _state(user_key="seed-u1")
        first = description_seed(state)
        state.descriptions.update_count = 1
        second = description_seed(state)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
