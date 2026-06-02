from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from opinion_agent.config import load_configs
from opinion_agent.schemas import Evidence, EvidenceScores, Intent, QueryUnderstanding, SearchResult, SearchRoundResult, SourceType
from opinion_agent.text_utils import (
    clamp,
    contains_any,
    domain_from_url,
    lexical_similarity,
    normalize_url,
    short_summary,
    stable_id,
    title_similarity,
)


INTENT_TERMS: dict[Intent, list[str]] = {
    Intent.OFFICIAL_RESPONSE: ["官方回应", "声明", "通报", "情况说明", "回应称", "发布公告", "品牌发布"],
    Intent.RUMOR_CHECK: ["辟谣", "澄清", "不实", "反转", "谣言", "已证实", "未证实", "核实"],
    Intent.RISK_IMPACT: ["处罚", "整改", "下架", "召回", "解约", "股价", "赔偿", "约谈", "影响"],
    Intent.PUBLIC_REACTION: ["网友", "评论", "热议", "吐槽", "支持", "质疑", "社交平台", "用户"],
    Intent.LEGAL_REGULATION: ["监管", "市监局", "法院", "律师", "违法", "处罚决定", "依法"],
    Intent.MEDIA_COVERAGE: ["报道", "记者", "媒体", "新闻", "跟进"],
    Intent.FACT: ["发生", "事件", "称", "反映", "曝光", "基本事实"],
    Intent.TIMELINE: ["时间线", "最新进展", "起初", "随后", "之后"],
}


class DeduplicationResult:
    def __init__(self) -> None:
        self.seen_urls: set[str] = set()
        self.kept_titles: list[str] = []
        self.duplicate_count = 0
        self.total_count = 0

    @property
    def duplicate_ratio(self) -> float:
        if self.total_count == 0:
            return 0
        return self.duplicate_count / self.total_count


class EvidenceDeduplicator:
    def __init__(self, title_threshold: float = 0.90) -> None:
        self.title_threshold = title_threshold
        self.result = DeduplicationResult()

    def is_duplicate(self, result: SearchResult) -> bool:
        self.result.total_count += 1
        url = normalize_url(result.url)
        if url in self.result.seen_urls:
            self.result.duplicate_count += 1
            return True
        if any(title_similarity(result.title, title) > self.title_threshold for title in self.result.kept_titles):
            self.result.duplicate_count += 1
            return True
        self.result.seen_urls.add(url)
        self.result.kept_titles.append(result.title)
        return False


class FreshnessScorer:
    def score(self, published_at: str | None, historical: bool = False) -> float:
        if historical:
            return 0.75
        if not published_at:
            return 0.45
        try:
            published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except ValueError:
            try:
                published = datetime.fromisoformat(f"{published_at}T00:00:00+00:00")
            except ValueError:
                return 0.45
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - published.astimezone(UTC)
        if delta.days < 1:
            return 1.0
        if delta.days <= 3:
            return 0.9
        if delta.days <= 7:
            return 0.75
        if delta.days <= 30:
            return 0.55
        return 0.35


class IntentMatcher:
    def score(self, intent: Intent, title: str, content: str) -> float:
        text = f"{title}\n{content}"
        terms = INTENT_TERMS.get(intent, [])
        if not terms:
            return 0.55
        count = sum(1 for term in terms if term in text)
        if count >= 2:
            return 0.9
        if count == 1:
            return 0.75
        if intent == Intent.FACT and contains_any(text, ["曝", "事件", "回应", "报道"]):
            return 0.65
        return 0.35


class RelevanceFilter:
    def hard_match(self, understanding: QueryUnderstanding, result: SearchResult) -> bool:
        text = f"{result.title}\n{result.snippet}\n{result.raw_content}"
        entities = [entity.name for entity in understanding.main_entities]
        keywords = understanding.event_keywords
        has_entity = contains_any(text, entities) if entities else True
        has_keyword = contains_any(text, keywords) if keywords else True
        return has_entity and has_keyword

    def score(self, understanding: QueryUnderstanding, result: SearchResult) -> float:
        query_text = f"{understanding.raw_input} {' '.join(understanding.event_keywords)}"
        candidate_text = f"{result.title} {result.snippet} {result.raw_content[:1000]}"
        score = lexical_similarity(query_text, candidate_text)
        if self.hard_match(understanding, result):
            score += 0.25
        return min(1.0, score)


class CredibilityScorer:
    def __init__(self) -> None:
        configs = load_configs()
        self.base = configs["source_credibility"].get("credibility_base", {})

    def classify_source(self, url: str, source: str, platform: str) -> SourceType:
        domain = domain_from_url(url)
        source_l = source.lower()
        if domain.endswith("gov.cn") or "市监" in source or "监管" in source or "法院" in source:
            return SourceType.GOVERNMENT
        if "官方" in source or "品牌官方" in source or platform == "official":
            return SourceType.OFFICIAL
        if any(d in domain for d in ["people.com.cn", "xinhuanet.com", "cctv.com", "chinanews.com", "thepaper.cn", "caixin.com"]):
            return SourceType.MAINSTREAM_MEDIA
        if any(w in source for w in ["澎湃", "新华", "央视", "中新", "财新", "人民日报"]):
            return SourceType.MAINSTREAM_MEDIA
        if any(d in domain for d in ["weibo.com", "zhihu.com", "xiaohongshu.com"]) or any(w in source for w in ["微博", "知乎", "小红书", "网友", "用户"]):
            return SourceType.ORDINARY_SOCIAL_POST
        if "地方" in source or "本地" in source:
            return SourceType.LOCAL_MEDIA
        if "财经" in source or "行业" in source:
            return SourceType.INDUSTRY_MEDIA
        if "unknown" in source_l:
            return SourceType.UNKNOWN
        return SourceType.UNKNOWN

    def score(self, source_type: SourceType, title: str, content: str, published_at: str | None) -> float:
        score = float(self.base.get(source_type.value, 0.4))
        if published_at:
            score += 0.05
        if any(k in content for k in ["图片", "视频", "文件", "公告", "通报", "处罚决定"]):
            score += 0.05
        if any(k in title for k in ["震惊", "炸了", "疯传", "内幕", "惊天"]):
            score -= 0.10
        if len(content) < 40:
            score -= 0.10
        if any(k in content for k in ["疑似搬运", "营销号", "网传截图"]):
            score -= 0.20
        if "没有明确来源" in content:
            score -= 0.15
        return clamp(score)


class EvidenceFilter:
    def __init__(self, relevance_threshold: float = 0.45) -> None:
        self.relevance_threshold = relevance_threshold
        self.relevance = RelevanceFilter()
        self.credibility = CredibilityScorer()
        self.freshness = FreshnessScorer()
        self.intent_matcher = IntentMatcher()
        self.last_filter_trace: list[dict[str, object]] = []

    def filter_rounds(
        self,
        understanding: QueryUnderstanding,
        rounds: list[SearchRoundResult],
        existing: list[Evidence] | None = None,
    ) -> tuple[list[Evidence], dict[str, float | int]]:
        existing = existing or []
        evidence: list[Evidence] = list(existing)
        dedup = EvidenceDeduplicator()
        self.last_filter_trace = []
        for ev in existing:
            dedup.result.seen_urls.add(normalize_url(ev.url))
            dedup.result.kept_titles.append(ev.title)

        source_types_seen = {ev.source_type for ev in existing}
        dropped = 0
        for search_round in rounds:
            for query_result in search_round.query_results:
                for result in query_result.results:
                    record: dict[str, object] = {
                        "round": search_round.round,
                        "query_id": query_result.query_id,
                        "query": query_result.query,
                        "intent": query_result.intent.value,
                        "platform": query_result.platform,
                        "result_id": result.result_id,
                        "rank": result.rank,
                        "title": result.title,
                        "url": normalize_url(result.url),
                        "source": result.source,
                        "published_at": result.published_at,
                    }
                    if dedup.is_duplicate(result):
                        dropped += 1
                        record.update({"decision": "dropped", "reason": "duplicate"})
                        self.last_filter_trace.append(record)
                        continue
                    hard_match = self.relevance.hard_match(understanding, result)
                    relevance_score = self.relevance.score(understanding, result)
                    if not hard_match or relevance_score < self.relevance_threshold:
                        dropped += 1
                        record.update(
                            {
                                "decision": "dropped",
                                "reason": "hard_match_failed" if not hard_match else "below_relevance_threshold",
                                "hard_match": hard_match,
                                "relevance": relevance_score,
                                "relevance_threshold": self.relevance_threshold,
                            }
                        )
                        self.last_filter_trace.append(record)
                        continue
                    source_type = self.credibility.classify_source(result.url, result.source, query_result.platform)
                    credibility_score = self.credibility.score(
                        source_type,
                        result.title,
                        result.raw_content or result.snippet,
                        result.published_at,
                    )
                    freshness_score = self.freshness.score(
                        result.published_at,
                        historical=understanding.time_sensitivity == "historical",
                    )
                    intent_match_score = self.intent_matcher.score(
                        query_result.intent,
                        result.title,
                        result.raw_content or result.snippet,
                    )
                    novelty = 0.85
                    diversity_bonus = 0.05 if source_type not in source_types_seen else 0.0
                    source_types_seen.add(source_type)
                    overall = clamp(
                        0.35 * relevance_score
                        + 0.20 * credibility_score
                        + 0.15 * freshness_score
                        + 0.15 * intent_match_score
                        + 0.10 * novelty
                        + 0.05 * diversity_bonus
                    )
                    summary = short_summary(result.raw_content or result.snippet or result.title)
                    evidence_id = stable_id("ev", result.result_id, result.url)
                    record.update(
                        {
                            "decision": "kept",
                            "reason": "passed_filter",
                            "evidence_id": evidence_id,
                            "hard_match": hard_match,
                            "source_type": source_type.value,
                            "scores": {
                                "relevance": relevance_score,
                                "credibility": credibility_score,
                                "freshness": freshness_score,
                                "intent_match": intent_match_score,
                                "novelty": novelty,
                                "overall": overall,
                                "source_diversity_bonus": diversity_bonus,
                            },
                            "summary": summary,
                        }
                    )
                    self.last_filter_trace.append(record)
                    evidence.append(
                        Evidence(
                            evidence_id=evidence_id,
                            source_result_id=result.result_id,
                            from_query_id=query_result.query_id,
                            round=search_round.round,
                            query=query_result.query,
                            intent=query_result.intent,
                            platform=query_result.platform,
                            title=result.title,
                            url=normalize_url(result.url),
                            source=result.source,
                            source_type=source_type,
                            published_at=result.published_at,
                            retrieved_at=result.retrieved_at,
                            content=result.raw_content or result.snippet,
                            summary=summary,
                            entities=[entity.name for entity in understanding.main_entities if entity.name in f"{result.title} {result.raw_content}"],
                            event_mentions=[kw for kw in understanding.event_keywords if kw in f"{result.title} {result.raw_content}"],
                            scores=EvidenceScores(
                                relevance=relevance_score,
                                credibility=credibility_score,
                                freshness=freshness_score,
                                intent_match=intent_match_score,
                                novelty=novelty,
                                overall=overall,
                                source_diversity_bonus=diversity_bonus,
                            ),
                            filter_reasons=self._filter_reasons(relevance_score, credibility_score, intent_match_score),
                        )
                    )

        stats = {
            "kept_count": len(evidence),
            "new_kept_count": max(0, len(evidence) - len(existing)),
            "dropped_count": dropped,
            "duplicate_count": dedup.result.duplicate_count,
            "duplicate_ratio": dedup.result.duplicate_ratio,
            "total_seen": dedup.result.total_count,
        }
        return evidence, stats

    def query_reward_inputs(self, evidences: list[Evidence]) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[Evidence]] = defaultdict(list)
        for ev in evidences:
            grouped[ev.from_query_id].append(ev)
        rewards: dict[str, dict[str, float | int]] = {}
        for query_id, items in grouped.items():
            rewards[query_id] = {
                "kept_evidence_count": len(items),
                "avg_relevance": sum(ev.scores.relevance for ev in items) / len(items),
                "avg_credibility": sum(ev.scores.credibility for ev in items) / len(items),
                "novelty_gain": sum(ev.scores.novelty for ev in items) / len(items),
            }
        return rewards

    def _filter_reasons(self, relevance: float, credibility: float, intent_match: float) -> list[str]:
        reasons: list[str] = []
        if relevance >= 0.65:
            reasons.append("high_relevance")
        if credibility >= 0.70:
            reasons.append("credible_source")
        if intent_match >= 0.70:
            reasons.append("matches_intent")
        if not reasons:
            reasons.append("kept_low_priority")
        return reasons
