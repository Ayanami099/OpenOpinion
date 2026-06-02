from __future__ import annotations

from dataclasses import dataclass, field

from opinion_agent.research.evidence import EvidenceFilter
from opinion_agent.research.planning import IterativeSearchController, SearchPlanner
from opinion_agent.research.search import SearchExecutor
from opinion_agent.research.understanding import QueryUnderstandingEngine
from opinion_agent.schemas import (
    CoverageMatrix,
    Evidence,
    QueryUnderstanding,
    SearchPlan,
    SearchRoundResult,
)


@dataclass
class ResearchResult:
    understanding: QueryUnderstanding
    plans: list[SearchPlan]
    search_results: list[SearchRoundResult]
    evidences: list[Evidence]
    coverage: CoverageMatrix
    decisions: list[dict]
    search_trace: list[dict]
    filter_trace: list[dict]
    filtering_stats: dict[str, float | int] = field(default_factory=dict)


class ResearchStage:
    def __init__(
        self,
        understanding: QueryUnderstandingEngine | None = None,
        planner: SearchPlanner | None = None,
        executor: SearchExecutor | None = None,
        evidence_filter: EvidenceFilter | None = None,
        controller: IterativeSearchController | None = None,
    ) -> None:
        self.understanding = understanding or QueryUnderstandingEngine()
        self.planner = planner or SearchPlanner()
        self.executor = executor or SearchExecutor()
        self.evidence_filter = evidence_filter or EvidenceFilter()
        self.controller = controller or IterativeSearchController()

    def run(
        self,
        user_input: str,
        max_rounds: int = 3,
        top_k: int = 5,
        query_budget: int = 12,
    ) -> ResearchResult:
        understanding = self.understanding.understand(user_input)
        plans: list[SearchPlan] = []
        search_results: list[SearchRoundResult] = []
        evidences: list[Evidence] = []
        decisions: list[dict] = []
        search_trace: list[dict] = []
        filter_trace: list[dict] = []
        filtering_stats: dict[str, float | int] = {}

        initial_plan = self.planner.create_initial_plan(understanding, query_budget=query_budget)
        for round_id in range(1, max_rounds + 1):
            if round_id == 1:
                current_plan = initial_plan
            else:
                current_coverage = self.controller.build_coverage(understanding, evidences)
                current_plan = self.controller.create_replenish_plan(
                    understanding,
                    current_coverage,
                    next_round=round_id,
                    query_budget=max(2, min(8, query_budget // 2)),
                )
                if not current_plan.queries:
                    break

            plans.append(current_plan)
            round_result = self.executor.execute(current_plan, top_k=top_k)
            search_trace.append({"round": round_id, "queries": list(self.executor.last_trace)})
            search_results.append(round_result)
            before_count = len(evidences)
            evidences, stats = self.evidence_filter.filter_rounds(understanding, [round_result], existing=evidences)
            filter_trace.append({"round": round_id, "stats": stats, "results": list(self.evidence_filter.last_filter_trace)})
            filtering_stats = stats
            coverage = self.controller.build_coverage(understanding, evidences)
            new_count = len(evidences) - before_count
            stop = round_id >= max_rounds or self.controller.should_stop(
                round_id,
                coverage,
                new_evidence_count=new_count,
                duplicate_ratio=float(stats.get("duplicate_ratio", 0)),
            )
            decisions.append(
                {
                    "round": round_id,
                    "coverage": {intent.value: item.model_dump(mode="json") for intent, item in coverage.coverage.items()},
                    "new_evidence_count": new_count,
                    "duplicate_ratio": stats.get("duplicate_ratio", 0),
                    "stop": stop,
                }
            )
            if stop:
                break

        coverage = self.controller.build_coverage(understanding, evidences)
        return ResearchResult(
            understanding=understanding,
            plans=plans,
            search_results=search_results,
            evidences=evidences,
            coverage=coverage,
            decisions=decisions,
            search_trace=search_trace,
            filter_trace=filter_trace,
            filtering_stats=filtering_stats,
        )
