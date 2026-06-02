from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opinion_agent.analysis import AnalysisStage, OpinionMiner, RiskAssessor, TextStatsAnalyzer, TimelineBuilder
from opinion_agent.logging import AgentTraceBuilder, SessionLogger, TrainingViewBuilder
from opinion_agent.reporting import ReportGenerator, ReportStage
from opinion_agent.research import (
    EvidenceFilter,
    IterativeSearchController,
    QueryUnderstandingEngine,
    ResearchStage,
    SearchExecutor,
    SearchPlanner,
)
from opinion_agent.schemas import (
    SessionLog,
)
from opinion_agent.text_utils import now_iso, stable_id


@dataclass
class PipelineResult:
    session: SessionLog
    report_markdown: str
    artifacts: dict[str, Any]


class OpinionAnalysisPipeline:
    def __init__(
        self,
        understanding: QueryUnderstandingEngine | None = None,
        planner: SearchPlanner | None = None,
        executor: SearchExecutor | None = None,
        evidence_filter: EvidenceFilter | None = None,
        controller: IterativeSearchController | None = None,
        opinion_miner: OpinionMiner | None = None,
        timeline_builder: TimelineBuilder | None = None,
        risk_assessor: RiskAssessor | None = None,
        text_stats_analyzer: TextStatsAnalyzer | None = None,
        report_generator: ReportGenerator | None = None,
        research_stage: ResearchStage | None = None,
        analysis_stage: AnalysisStage | None = None,
        report_stage: ReportStage | None = None,
        logger: SessionLogger | None = None,
    ) -> None:
        self.research_stage = research_stage or ResearchStage(
            understanding=understanding,
            planner=planner,
            executor=executor,
            evidence_filter=evidence_filter,
            controller=controller,
        )
        self.analysis_stage = analysis_stage or AnalysisStage(
            opinion_miner=opinion_miner,
            timeline_builder=timeline_builder,
            risk_assessor=risk_assessor,
            text_stats_analyzer=text_stats_analyzer,
        )
        self.report_stage = report_stage or ReportStage(report_generator=report_generator)
        self.training_builder = TrainingViewBuilder()
        self.trace_builder = AgentTraceBuilder()
        self.logger = logger

    def analyze(
        self,
        user_input: str,
        max_rounds: int = 3,
        top_k: int = 5,
        out_dir: str = "data/sessions",
        query_budget: int = 12,
    ) -> PipelineResult:
        created_at = now_iso()
        session_id = stable_id("case", created_at, user_input, length=12)
        research = self.research_stage.run(
            user_input,
            max_rounds=max_rounds,
            top_k=top_k,
            query_budget=query_budget,
        )
        analysis = self.analysis_stage.run(
            research.understanding,
            research.evidences,
            research.coverage,
            search_results=research.search_results,
            duplicate_ratio=float(research.filtering_stats.get("duplicate_ratio", 0)),
        )
        report = self.report_stage.run(
            research.understanding,
            research.evidences,
            research.coverage,
            analysis.opinion,
            analysis.timeline,
            analysis.risk,
            analysis.text_analysis,
        )
        report_trace = self.report_stage.last_trace
        run_config = {
            "max_rounds": max_rounds,
            "top_k": top_k,
            "query_budget": query_budget,
            "out_dir": out_dir,
        }
        agent_trace = self.trace_builder.build(
            session_id=session_id,
            created_at=created_at,
            user_input=user_input,
            run_config=run_config,
            research=research,
            analysis=analysis,
            report_trace=report_trace,
        )
        training_views = self.training_builder.build(
            user_input=user_input,
            query_understanding=research.understanding.model_dump(mode="json"),
            plans=research.plans,
            results=research.search_results,
            evidences=research.evidences,
            coverage=research.coverage,
            evaluation=analysis.evaluation,
        )
        session = SessionLog(
            session_id=session_id,
            created_at=created_at,
            user_input=user_input,
            query_understanding=research.understanding,
            search_plan_rounds=research.plans,
            search_results=research.search_results,
            evidence_pack=research.evidences,
            evidence_filtering=research.filtering_stats,
            iterative_decisions=research.decisions,
            opinion_mining=analysis.opinion,
            timeline=analysis.timeline,
            risk_assessment=analysis.risk,
            text_analysis=analysis.text_analysis,
            final_report={
                "format": "markdown",
                "content": report,
                "visual_evidence": report_trace.get("visual_evidence", []),
            },
            evaluation=analysis.evaluation,
            agent_trace=agent_trace,
            training_views=training_views,
        )
        logger = self.logger or SessionLogger(session_dir=out_dir)
        artifacts = logger.save(session, report)
        return PipelineResult(session=session, report_markdown=report, artifacts=artifacts)
