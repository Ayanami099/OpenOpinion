from opinion_agent.research import EvidenceDeduplicator, EvidenceFilter
from opinion_agent.schemas import (
    Entity,
    EventType,
    Intent,
    QueryResult,
    QueryUnderstanding,
    RiskLevel,
    SearchResult,
    SearchRoundResult,
)
from opinion_agent.text_utils import normalize_url


def _understanding():
    return QueryUnderstanding(
        raw_input="某奶茶品牌被曝使用过期原料",
        main_entities=[Entity(name="某奶茶品牌", type="brand")],
        event_keywords=["过期原料", "食品安全"],
        event_type=EventType.FOOD_SAFETY,
        risk_level=RiskLevel.HIGH,
        required_intents=[Intent.FACT],
    )


def test_url_normalization_removes_tracking_params():
    assert normalize_url("https://example.com/a?utm_source=x&id=1#frag") == "https://example.com/a?id=1"


def test_deduplicator_detects_same_normalized_url():
    result = SearchResult(
        result_id="r1",
        query_id="q1",
        rank=1,
        title="某奶茶品牌被曝使用过期原料",
        url="https://example.com/a?utm_source=x",
        source="澎湃新闻",
        retrieved_at="2026-05-29T00:00:00Z",
    )
    duplicate = result.model_copy(update={"result_id": "r2", "url": "https://example.com/a"})
    dedup = EvidenceDeduplicator()
    assert dedup.is_duplicate(result) is False
    assert dedup.is_duplicate(duplicate) is True


def test_evidence_filter_scores_and_keeps_relevant_result():
    result = SearchResult(
        result_id="r1",
        query_id="q1",
        rank=1,
        title="某奶茶品牌被曝使用过期原料",
        url="https://thepaper.cn/news/1",
        source="澎湃新闻",
        snippet="某奶茶品牌 过期原料 食品安全 事件",
        published_at="2026-05-29",
        retrieved_at="2026-05-29T00:00:00Z",
        raw_content="某奶茶品牌被曝使用过期原料，多家媒体报道食品安全问题，监管部门关注。",
        content_length=40,
    )
    round_result = SearchRoundResult(
        round=1,
        query_results=[
            QueryResult(query_id="q1", query="某奶茶品牌 过期原料", intent=Intent.FACT, platform="news", results=[result])
        ],
    )
    evidences, stats = EvidenceFilter().filter_rounds(_understanding(), [round_result])
    assert stats["kept_count"] == 1
    assert evidences[0].scores.relevance >= 0.45
    assert evidences[0].scores.overall > 0
