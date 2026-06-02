from __future__ import annotations

from dataclasses import dataclass

from opinion_agent.analysis.opinion import OpinionMiner
from opinion_agent.analysis.risk import RiskAssessor
from opinion_agent.analysis.text_stats import TextStatsAnalyzer
from opinion_agent.analysis.timeline import TimelineBuilder
from opinion_agent.eval.metrics import evaluate_session
from opinion_agent.schemas import (
    CoverageMatrix,
    Evidence,
    OpinionMiningResult,
    QueryUnderstanding,
    RiskAssessment,
    SearchRoundResult,
    TextAnalysisResult,
    Timeline,
)


@dataclass
class AnalysisResult:
    opinion: OpinionMiningResult
    timeline: Timeline
    risk: RiskAssessment
    text_analysis: TextAnalysisResult
    evaluation: dict[str, float]


class AnalysisStage:
    def __init__(
        self,
        opinion_miner: OpinionMiner | None = None,
        timeline_builder: TimelineBuilder | None = None,
        risk_assessor: RiskAssessor | None = None,
        text_stats_analyzer: TextStatsAnalyzer | None = None,
    ) -> None:
        self.opinion_miner = opinion_miner or OpinionMiner()
        self.timeline_builder = timeline_builder or TimelineBuilder()
        self.risk_assessor = risk_assessor or RiskAssessor()
        self.text_stats_analyzer = text_stats_analyzer or TextStatsAnalyzer()

    def run(
        self,
        understanding: QueryUnderstanding,
        evidences: list[Evidence],
        coverage: CoverageMatrix,
        search_results: list[SearchRoundResult] | None = None,
        duplicate_ratio: float = 0,
    ) -> AnalysisResult:
        target = understanding.main_entities[0].name if understanding.main_entities else understanding.raw_input
        opinion = self.opinion_miner.mine(evidences, target=target)
        timeline = self.timeline_builder.build(evidences)
        risk = self.risk_assessor.assess(understanding.event_type, evidences, opinion, timeline)
        text_analysis = self.text_stats_analyzer.analyze(evidences, opinion)
        total_results = sum(len(qr.results) for sr in search_results or [] for qr in sr.query_results)
        evaluation = evaluate_session(
            understanding,
            evidences,
            coverage,
            opinion=opinion,
            total_results=total_results,
            duplicate_ratio=duplicate_ratio,
        )
        return AnalysisResult(opinion=opinion, timeline=timeline, risk=risk, text_analysis=text_analysis, evaluation=evaluation)
