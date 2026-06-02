from opinion_agent.analysis import RiskAssessor
from opinion_agent.research import IterativeSearchController
from opinion_agent.schemas import (
    Entity,
    EventType,
    Evidence,
    EvidenceScores,
    Intent,
    OpinionMiningResult,
    QueryUnderstanding,
    RiskLevel,
    SentimentResult,
    SourceType,
    Timeline,
    TimelineEvent,
)


def _understanding():
    return QueryUnderstanding(
        raw_input="某奶茶品牌被曝使用过期原料",
        main_entities=[Entity(name="某奶茶品牌", type="brand")],
        event_keywords=["过期原料", "食品安全"],
        event_type=EventType.FOOD_SAFETY,
        risk_level=RiskLevel.HIGH,
        required_intents=[Intent.FACT, Intent.OFFICIAL_RESPONSE, Intent.PUBLIC_REACTION],
    )


def _evidence(ev_id: str, intent: Intent, source_type: SourceType = SourceType.MAINSTREAM_MEDIA) -> Evidence:
    return Evidence(
        evidence_id=ev_id,
        source_result_id=f"r_{ev_id}",
        from_query_id=f"q_{ev_id}",
        round=1,
        query="某奶茶品牌 过期原料",
        intent=intent,
        platform="general_web",
        title="某奶茶品牌被曝使用过期原料",
        url=f"https://example.com/{ev_id}",
        source="澎湃新闻",
        source_type=source_type,
        published_at="2026-05-29",
        retrieved_at="2026-05-29T00:00:00Z",
        content="某奶茶品牌过期原料食品安全事件。",
        summary="事件摘要",
        scores=EvidenceScores(relevance=0.9, credibility=0.8, freshness=1.0, intent_match=0.8, novelty=0.8, overall=0.84),
    )


def test_controller_creates_replenish_plan_for_missing_intent():
    controller = IterativeSearchController()
    coverage = controller.build_coverage(_understanding(), [_evidence("1", Intent.FACT)])
    plan = controller.create_replenish_plan(_understanding(), coverage, next_round=2)
    intents = {query.intent for query in plan.queries}
    assert Intent.OFFICIAL_RESPONSE in intents
    assert Intent.PUBLIC_REACTION in intents


def test_risk_assessor_outputs_dimension_scores():
    evidences = [
        _evidence("1", Intent.FACT, SourceType.MAINSTREAM_MEDIA),
        _evidence("2", Intent.PUBLIC_REACTION, SourceType.ORDINARY_SOCIAL_POST),
        _evidence("3", Intent.OFFICIAL_RESPONSE, SourceType.OFFICIAL),
    ]
    opinion = OpinionMiningResult(
        sentiments=[
            SentimentResult(evidence_id="1", sentiment="neutral", sentiment_score=0, confidence=0.8),
            SentimentResult(evidence_id="2", sentiment="negative", sentiment_score=-0.8, confidence=0.9),
            SentimentResult(evidence_id="3", sentiment="neutral", sentiment_score=0, confidence=0.8),
        ]
    )
    timeline = Timeline(
        timeline=[
            TimelineEvent(time="2026-05-28", stage="initial_exposure", event="曝光", evidence_ids=["1"]),
            TimelineEvent(time="2026-05-29", stage="official_response", event="回应", evidence_ids=["3"]),
        ]
    )
    risk = RiskAssessor().assess(EventType.FOOD_SAFETY, evidences, opinion, timeline)
    assert 0 <= risk.overall_risk_score <= 1
    assert "heat" in risk.dimensions
    assert risk.overall_risk_level in {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM_HIGH, RiskLevel.HIGH}
