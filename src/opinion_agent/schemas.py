from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class EventType(StrEnum):
    PRODUCT_QUALITY_CRISIS = "product_quality_crisis"
    PRODUCT_SAFETY_CRISIS = "product_safety_crisis"
    FOOD_SAFETY = "food_safety"
    CELEBRITY_SCANDAL = "celebrity_scandal"
    CAMPUS_INCIDENT = "campus_incident"
    CORPORATE_PR_CRISIS = "corporate_pr_crisis"
    CONSUMER_COMPLAINT = "consumer_complaint"
    POLICY_DISPUTE = "policy_dispute"
    PUBLIC_SAFETY = "public_safety"
    FINANCIAL_RISK = "financial_risk"
    LABOR_DISPUTE = "labor_dispute"
    RUMOR_OR_MISINFORMATION = "rumor_or_misinformation"
    OTHER = "other"


class Intent(StrEnum):
    FACT = "FACT"
    TIMELINE = "TIMELINE"
    OFFICIAL_RESPONSE = "OFFICIAL_RESPONSE"
    MEDIA_COVERAGE = "MEDIA_COVERAGE"
    PUBLIC_REACTION = "PUBLIC_REACTION"
    STAKEHOLDER_VIEW = "STAKEHOLDER_VIEW"
    RUMOR_CHECK = "RUMOR_CHECK"
    RISK_IMPACT = "RISK_IMPACT"
    LEGAL_REGULATION = "LEGAL_REGULATION"
    COMPETITOR_CONTEXT = "COMPETITOR_CONTEXT"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"


class SourceType(StrEnum):
    OFFICIAL = "official"
    GOVERNMENT = "government"
    MAINSTREAM_MEDIA = "mainstream_media"
    LOCAL_MEDIA = "local_media"
    INDUSTRY_MEDIA = "industry_media"
    VERIFIED_SOCIAL_ACCOUNT = "verified_social_account"
    ORDINARY_SOCIAL_POST = "ordinary_social_post"
    FORUM_POST = "forum_post"
    ANONYMOUS_POST = "anonymous_post"
    CONTENT_FARM = "content_farm"
    UNKNOWN = "unknown"


class CoverageStatus(StrEnum):
    MISSING = "missing"
    PARTIAL = "partial"
    SUFFICIENT = "sufficient"
    CONFLICTING = "conflicting"


class Entity(BaseModel):
    name: str
    type: str = "unknown"
    confidence: float = Field(default=0.5, ge=0, le=1)


class QueryUnderstanding(BaseModel):
    raw_input: str
    main_entities: list[Entity] = Field(default_factory=list)
    event_keywords: list[str] = Field(default_factory=list)
    event_type: EventType = EventType.OTHER
    risk_level: RiskLevel = RiskLevel.MEDIUM
    time_sensitivity: str = "recent"
    geographic_scope: str = "unknown"
    possible_platforms: list[str] = Field(default_factory=lambda: ["general_web", "news"])
    required_intents: list[Intent] = Field(default_factory=list)
    optional_intents: list[Intent] = Field(default_factory=list)
    known_constraints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_required_intents(self) -> "QueryUnderstanding":
        if not self.required_intents:
            self.required_intents = [
                Intent.FACT,
                Intent.OFFICIAL_RESPONSE,
                Intent.MEDIA_COVERAGE,
                Intent.PUBLIC_REACTION,
                Intent.RUMOR_CHECK,
                Intent.RISK_IMPACT,
            ]
        return self


class SearchQuery(BaseModel):
    query_id: str
    query: str
    intent: Intent
    platform: str = "general_web"
    priority: int = Field(default=2, ge=1, le=5)
    expected_evidence_type: str = "general"
    reason: str = ""


class SearchPlan(BaseModel):
    round: int
    planner_type: str = "prompt_rule"
    queries: list[SearchQuery] = Field(default_factory=list)


class SearchResult(BaseModel):
    result_id: str
    query_id: str
    rank: int
    title: str
    url: str
    source: str = "unknown"
    snippet: str = ""
    published_at: str | None = None
    retrieved_at: str
    raw_content: str = ""
    content_length: int = 0


class QueryResult(BaseModel):
    query_id: str
    query: str
    intent: Intent
    platform: str
    results: list[SearchResult] = Field(default_factory=list)


class SearchRoundResult(BaseModel):
    round: int
    query_results: list[QueryResult] = Field(default_factory=list)


class EvidenceScores(BaseModel):
    relevance: float = Field(ge=0, le=1)
    credibility: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    intent_match: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)
    source_diversity_bonus: float = Field(default=0, ge=0, le=1)


class Evidence(BaseModel):
    evidence_id: str
    source_result_id: str
    from_query_id: str
    round: int
    query: str
    intent: Intent
    platform: str
    title: str
    url: str
    source: str
    source_type: SourceType
    published_at: str | None = None
    retrieved_at: str
    content: str = ""
    summary: str = ""
    entities: list[str] = Field(default_factory=list)
    event_mentions: list[str] = Field(default_factory=list)
    scores: EvidenceScores
    filter_status: Literal["kept", "dropped"] = "kept"
    filter_reasons: list[str] = Field(default_factory=list)


class CoverageItem(BaseModel):
    status: CoverageStatus
    evidence_count: int = 0
    high_quality_count: int = 0


class CoverageMatrix(BaseModel):
    coverage: dict[Intent, CoverageItem] = Field(default_factory=dict)


class SentimentResult(BaseModel):
    evidence_id: str
    sentiment: Literal["positive", "neutral", "negative", "mixed", "unknown"]
    sentiment_score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    reason: str = ""
    positive_hits: list[str] = Field(default_factory=list)
    negative_hits: list[str] = Field(default_factory=list)
    positive_score: float = Field(default=0, ge=0)
    negative_score: float = Field(default=0, ge=0)


class StanceResult(BaseModel):
    evidence_id: str
    target: str
    stance: Literal[
        "support_entity",
        "criticize_entity",
        "question_claim",
        "support_claim",
        "neutral_report",
        "official_defense",
        "regulatory_action",
        "unknown",
    ]
    confidence: float = Field(ge=0, le=1)
    reason: str = ""


class OpinionUnit(BaseModel):
    evidence_id: str
    aspect: str
    claim: str
    sentiment: str
    stance: str
    keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.6, ge=0, le=1)


class OpinionCluster(BaseModel):
    cluster_id: str
    cluster_name: str
    representative_claim: str
    sentiment_distribution: dict[str, float] = Field(default_factory=dict)
    stance_distribution: dict[str, float] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source_platforms: list[str] = Field(default_factory=list)
    cluster_size: int = 0
    importance_score: float = Field(default=0, ge=0, le=1)


class OpinionMiningResult(BaseModel):
    sentiments: list[SentimentResult] = Field(default_factory=list)
    stances: list[StanceResult] = Field(default_factory=list)
    opinion_units: list[OpinionUnit] = Field(default_factory=list)
    opinion_clusters: list[OpinionCluster] = Field(default_factory=list)
    keywords: dict[str, list[str]] = Field(default_factory=dict)


class TimelineEvent(BaseModel):
    time: str
    stage: str
    event: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.6, ge=0, le=1)


class Timeline(BaseModel):
    timeline: list[TimelineEvent] = Field(default_factory=list)


class RiskDimension(BaseModel):
    score: float = Field(ge=0, le=1)
    level: RiskLevel
    signals: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class RiskAssessment(BaseModel):
    overall_risk_level: RiskLevel
    overall_risk_score: float = Field(ge=0, le=1)
    dimensions: dict[str, RiskDimension] = Field(default_factory=dict)
    risk_summary: str = ""


class TextAnalysisResult(BaseModel):
    evidence_count: int = 0
    platform_distribution: dict[str, int] = Field(default_factory=dict)
    source_type_distribution: dict[str, int] = Field(default_factory=dict)
    intent_distribution: dict[str, int] = Field(default_factory=dict)
    date_distribution: dict[str, int] = Field(default_factory=dict)
    sentiment_counts: dict[str, int] = Field(default_factory=dict)
    sentiment_ratios: dict[str, float] = Field(default_factory=dict)
    average_sentiment_score: float = 0
    weighted_negative_ratio: float = 0
    source_sentiment_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    aspect_sentiment_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    stance_distribution: dict[str, int] = Field(default_factory=dict)
    top_keywords: list[dict[str, Any]] = Field(default_factory=list)
    sentiment_term_hits: dict[str, int] = Field(default_factory=dict)
    risk_term_hits: dict[str, int] = Field(default_factory=dict)
    quality_averages: dict[str, float] = Field(default_factory=dict)
    text_volume: dict[str, float] = Field(default_factory=dict)


class QueryRewardLog(BaseModel):
    query_id: str
    query: str
    intent: Intent
    platform: str
    round: int
    result_count: int = 0
    kept_evidence_count: int = 0
    avg_relevance: float = 0
    avg_credibility: float = 0
    novelty_gain: float = 0
    intent_coverage_gain: float = 0
    redundancy_rate: float = 0
    query_reward: float = 0


class TrainingViews(BaseModel):
    sft_query_planner_sample: dict[str, Any] = Field(default_factory=dict)
    query_reward_logs: list[QueryRewardLog] = Field(default_factory=list)
    search_plan_rewards: list[dict[str, Any]] = Field(default_factory=list)
    preference_samples: list[dict[str, Any]] = Field(default_factory=list)
    rl_trajectory: list[dict[str, Any]] = Field(default_factory=list)


class SessionLog(BaseModel):
    session_id: str
    created_at: str
    user_input: str
    system_version: str = "stage1_prompt_rule_v0.1"
    query_understanding: QueryUnderstanding
    search_plan_rounds: list[SearchPlan] = Field(default_factory=list)
    search_results: list[SearchRoundResult] = Field(default_factory=list)
    evidence_pack: list[Evidence] = Field(default_factory=list)
    evidence_filtering: dict[str, Any] = Field(default_factory=dict)
    iterative_decisions: list[dict[str, Any]] = Field(default_factory=list)
    opinion_mining: OpinionMiningResult
    timeline: Timeline
    risk_assessment: RiskAssessment
    text_analysis: TextAnalysisResult = Field(default_factory=TextAnalysisResult)
    final_report: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    agent_trace: dict[str, Any] = Field(default_factory=dict)
    training_views: TrainingViews
