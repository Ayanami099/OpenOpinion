from __future__ import annotations

from opinion_agent.reporting.generator import ReportGenerator
from opinion_agent.schemas import (
    CoverageMatrix,
    Evidence,
    OpinionMiningResult,
    QueryUnderstanding,
    RiskAssessment,
    TextAnalysisResult,
    Timeline,
)


class ReportStage:
    def __init__(self, report_generator: ReportGenerator | None = None) -> None:
        self.report_generator = report_generator or ReportGenerator()
        self.last_trace: dict = {}

    def run(
        self,
        understanding: QueryUnderstanding,
        evidences: list[Evidence],
        coverage: CoverageMatrix,
        opinion: OpinionMiningResult,
        timeline: Timeline,
        risk: RiskAssessment,
        text_analysis: TextAnalysisResult | None = None,
    ) -> str:
        report = self.report_generator.generate(understanding, evidences, coverage, opinion, timeline, risk, text_analysis)
        self.last_trace = dict(self.report_generator.last_trace)
        return report
