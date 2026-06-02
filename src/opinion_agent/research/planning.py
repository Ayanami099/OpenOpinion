from __future__ import annotations

import re

from opinion_agent.schemas import (
    CoverageItem,
    CoverageMatrix,
    CoverageStatus,
    Evidence,
    Intent,
    QueryUnderstanding,
    SearchPlan,
    SearchQuery,
)
from opinion_agent.text_utils import stable_id


GENERIC_KEYWORDS = {"事件", "舆情", "风波", "争议", "最新", "进展"}
HIGH_QUALITY_THRESHOLD = 0.70


INTENT_TEMPLATES: dict[Intent, list[tuple[str, str, int, str, str]]] = {
    Intent.FACT: [
        ("{entity} {keywords}", "general_web", 1, "basic_fact", "确认事件是否真实发生以及基本事实"),
        ("{entity} {keywords} 事件 最新", "general_web", 2, "basic_fact", "补充最新事实信息"),
    ],
    Intent.OFFICIAL_RESPONSE: [
        ("{entity} {keywords} 官方回应", "general_web", 1, "official_statement", "获取企业或相关方正式回应"),
        ("{entity} {keywords} 情况说明 通报", "official", 1, "official_statement", "查找官方说明或监管通报"),
    ],
    Intent.MEDIA_COVERAGE: [
        ("{entity} {keywords} 媒体报道", "news", 1, "news_report", "寻找相对可信媒体报道"),
        ("{entity} {keywords} site:thepaper.cn", "news", 2, "news_report", "查找主流媒体跟进"),
    ],
    Intent.PUBLIC_REACTION: [
        ("{entity} {keywords} 网友评价 热议", "weibo", 2, "public_opinion", "观察公众情绪和讨论焦点"),
        ("{entity} {keywords} 小红书 知乎", "xiaohongshu", 3, "public_opinion", "补充社区平台讨论"),
    ],
    Intent.RUMOR_CHECK: [
        ("{entity} {keywords} 辟谣 澄清 反转", "general_web", 2, "rumor_check", "检查是否存在澄清、反转或谣言"),
    ],
    Intent.RISK_IMPACT: [
        ("{entity} {keywords} 处罚 整改 赔偿 影响", "general_web", 2, "impact_or_resolution", "获取后续处理和风险影响"),
    ],
    Intent.LEGAL_REGULATION: [
        ("{entity} {keywords} 监管 市监局 处罚决定", "official", 2, "legal_regulation", "获取监管或法律处理信息"),
    ],
    Intent.TIMELINE: [
        ("{entity} {keywords} 时间线 最新进展", "general_web", 3, "timeline", "查找事件发展节点"),
    ],
    Intent.STAKEHOLDER_VIEW: [
        ("{entity} {keywords} 消费者 商家 相关方", "general_web", 3, "stakeholder_view", "补充相关方观点"),
    ],
    Intent.COMPETITOR_CONTEXT: [
        ("{entity} {keywords} 同类事件 对比", "general_web", 4, "competitor_context", "补充同类事件参照"),
    ],
}


REPLENISH_TEMPLATES: dict[Intent, list[str]] = {
    Intent.OFFICIAL_RESPONSE: [
        "{entity} 官方回应 {keywords}",
        "{entity} 情况说明 {keywords}",
        "{entity} 发布声明 通报 {keywords}",
    ],
    Intent.RUMOR_CHECK: [
        "{entity} {keywords} 辟谣",
        "{entity} {keywords} 澄清",
        "{entity} {keywords} 反转 不实",
    ],
    Intent.PUBLIC_REACTION: [
        "{entity} {keywords} 网友评价",
        "{entity} {keywords} 热议 微博",
        "{entity} {keywords} 小红书 知乎",
    ],
    Intent.RISK_IMPACT: [
        "{entity} {keywords} 处罚 整改",
        "{entity} {keywords} 下架 召回 赔偿",
        "{entity} {keywords} 影响 后续",
    ],
    Intent.LEGAL_REGULATION: [
        "{entity} {keywords} 监管",
        "{entity} {keywords} 市监局 处罚",
        "{entity} {keywords} 法院 律师 违法",
    ],
}


def normalize_query_text(text: str) -> str:
    parts = [part.strip() for part in re.split(r"\s+", text) if part.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        unique.append(part)
    return " ".join(unique)


def build_query_context(understanding: QueryUnderstanding, max_entities: int = 3, max_keywords: int = 5) -> tuple[str, str]:
    entity_terms = _unique_terms(entity.name for entity in understanding.main_entities if entity.name)
    entity_terms = entity_terms[:max_entities] or [understanding.raw_input]
    keyword_terms = _keyword_terms(understanding.event_keywords, entity_terms, max_keywords=max_keywords)
    return normalize_query_text(" ".join(entity_terms)), normalize_query_text(" ".join(keyword_terms))


def render_query(template: str, entity: str, keywords: str) -> str:
    return normalize_query_text(template.format(entity=entity, keywords=keywords).strip())


class SearchPlanner:
    def create_initial_plan(self, understanding: QueryUnderstanding, query_budget: int = 12) -> SearchPlan:
        entity, keywords = build_query_context(understanding)
        queries: list[SearchQuery] = []
        seen: set[str] = set()

        for intent in understanding.required_intents:
            for template, platform, priority, evidence_type, reason in INTENT_TEMPLATES.get(intent, []):
                query = render_query(template, entity=entity, keywords=keywords)
                if query in seen:
                    continue
                seen.add(query)
                queries.append(
                    SearchQuery(
                        query_id=stable_id("q", 1, len(queries), query),
                        query=query,
                        intent=intent,
                        platform=platform,
                        priority=priority,
                        expected_evidence_type=evidence_type,
                        reason=reason,
                    )
                )
                if len(queries) >= query_budget:
                    return SearchPlan(round=1, queries=queries)

        if len(queries) < 8:
            for suffix, intent in [
                ("最新", Intent.FACT),
                ("官方回应", Intent.OFFICIAL_RESPONSE),
                ("媒体报道", Intent.MEDIA_COVERAGE),
                ("网友评价", Intent.PUBLIC_REACTION),
                ("辟谣 反转", Intent.RUMOR_CHECK),
                ("处罚 整改", Intent.RISK_IMPACT),
            ]:
                query = render_query("{entity} {keywords} " + suffix, entity=entity, keywords=keywords)
                if query in seen:
                    continue
                seen.add(query)
                queries.append(
                    SearchQuery(
                        query_id=stable_id("q", 1, len(queries), query),
                        query=query,
                        intent=intent,
                        platform="general_web",
                        priority=2,
                        expected_evidence_type="fallback",
                        reason="补齐第一轮搜索覆盖",
                    )
                )
                if len(queries) >= query_budget:
                    break

        return SearchPlan(round=1, queries=queries)


class IterativeSearchController:
    def build_coverage(self, understanding: QueryUnderstanding, evidences: list[Evidence]) -> CoverageMatrix:
        coverage: dict[Intent, CoverageItem] = {}
        for intent in understanding.required_intents:
            matched = [ev for ev in evidences if ev.intent == intent and ev.filter_status == "kept"]
            high_quality = [ev for ev in matched if ev.scores.overall >= HIGH_QUALITY_THRESHOLD]
            if not matched:
                status = CoverageStatus.MISSING
            elif len(matched) >= 2 and high_quality:
                status = CoverageStatus.SUFFICIENT
            else:
                status = CoverageStatus.PARTIAL
            coverage[intent] = CoverageItem(
                status=status,
                evidence_count=len(matched),
                high_quality_count=len(high_quality),
            )
        return CoverageMatrix(coverage=coverage)

    def should_stop(
        self,
        round_id: int,
        coverage: CoverageMatrix,
        new_evidence_count: int,
        duplicate_ratio: float,
    ) -> bool:
        if round_id >= 3:
            return True
        required = [
            Intent.FACT,
            Intent.OFFICIAL_RESPONSE,
            Intent.MEDIA_COVERAGE,
            Intent.PUBLIC_REACTION,
            Intent.RUMOR_CHECK,
        ]
        statuses = coverage.coverage
        if all(statuses.get(intent, CoverageItem(status=CoverageStatus.MISSING)).status in {CoverageStatus.SUFFICIENT, CoverageStatus.PARTIAL} for intent in required):
            return True
        if new_evidence_count <= 2:
            return True
        if duplicate_ratio > 0.75:
            return True
        return False

    def create_replenish_plan(
        self,
        understanding: QueryUnderstanding,
        coverage: CoverageMatrix,
        next_round: int,
        query_budget: int = 6,
    ) -> SearchPlan:
        entity, keywords = build_query_context(understanding)
        queries: list[SearchQuery] = []
        seen: set[str] = set()

        for intent, item in coverage.coverage.items():
            if item.status == CoverageStatus.SUFFICIENT:
                continue
            templates = REPLENISH_TEMPLATES.get(intent, [f"{{entity}} {{keywords}} {intent.value} 最新"])
            for template in templates:
                query = render_query(template, entity=entity, keywords=keywords)
                if query in seen:
                    continue
                seen.add(query)
                queries.append(
                    SearchQuery(
                        query_id=stable_id("q", next_round, len(queries), query),
                        query=query,
                        intent=intent,
                        platform="general_web",
                        priority=2,
                        expected_evidence_type=f"{intent.value.lower()}_replenish",
                        reason=f"补搜缺失或不足的 {intent.value} 信息",
                    )
                )
                if len(queries) >= query_budget:
                    return SearchPlan(round=next_round, queries=queries)

        return SearchPlan(round=next_round, queries=queries)


def _keyword_terms(keywords: list[str], entity_terms: list[str], max_keywords: int) -> list[str]:
    cleaned: list[str] = []
    for keyword in keywords:
        term = _remove_entity_terms(keyword, entity_terms)
        term = _strip_generic_edges(term)
        if not term or term in GENERIC_KEYWORDS:
            continue
        if _is_duplicate(term, cleaned):
            continue
        cleaned.append(term)
        if len(cleaned) >= max_keywords:
            break
    return cleaned


def _unique_terms(terms: object) -> list[str]:
    unique: list[str] = []
    for term in terms:
        if not isinstance(term, str):
            continue
        normalized = normalize_query_text(term)
        if normalized and not _is_duplicate(normalized, unique):
            unique.append(normalized)
    return unique


def _remove_entity_terms(keyword: str, entity_terms: list[str]) -> str:
    term = keyword.strip()
    for entity in entity_terms:
        term = term.replace(entity, " ")
    return normalize_query_text(term)


def _strip_generic_edges(term: str) -> str:
    cleaned = term.strip()
    cleaned = re.sub(r"^(关于|有关|网传|疑似)", "", cleaned)
    cleaned = re.sub(r"(事件|舆情|风波|争议)$", "", cleaned)
    return normalize_query_text(cleaned)


def _is_duplicate(term: str, existing: list[str]) -> bool:
    compact = _compact(term)
    return any(compact == _compact(item) for item in existing)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()
