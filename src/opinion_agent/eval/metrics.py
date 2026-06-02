from __future__ import annotations

from opinion_agent.schemas import CoverageMatrix, Evidence, OpinionMiningResult, QueryUnderstanding


def evaluate_session(
    understanding: QueryUnderstanding,
    evidences: list[Evidence],
    coverage: CoverageMatrix,
    opinion: OpinionMiningResult | None = None,
    total_results: int = 0,
    duplicate_ratio: float = 0,
) -> dict[str, float]:
    required = understanding.required_intents
    covered = sum(1 for intent in required if coverage.coverage.get(intent) and coverage.coverage[intent].evidence_count > 0)
    intent_coverage = covered / len(required) if required else 0
    useful_rate = len(evidences) / total_results if total_results else (1.0 if evidences else 0.0)
    avg_score = sum(ev.scores.overall for ev in evidences) / len(evidences) if evidences else 0
    source_diversity = len({ev.source_type for ev in evidences}) / 8 if evidences else 0
    platform_diversity = len({ev.platform for ev in evidences}) / 6 if evidences else 0
    viewpoint_coverage = len(opinion.opinion_clusters) / 6 if opinion else 0
    return {
        "intent_coverage": min(1.0, intent_coverage),
        "useful_evidence_rate": min(1.0, useful_rate),
        "redundancy_rate": duplicate_ratio,
        "source_diversity": min(1.0, source_diversity),
        "platform_diversity": min(1.0, platform_diversity),
        "viewpoint_coverage": min(1.0, viewpoint_coverage),
        "average_evidence_score": avg_score,
        "report_faithfulness": 0.8 if evidences else 0.0,
    }

