from __future__ import annotations

from opinion_agent.eval.metrics import evaluate_session
from opinion_agent.analysis import OpinionMiner
from opinion_agent.research import EvidenceFilter, IterativeSearchController, SearchExecutor, SearchPlanner
from opinion_agent.schemas import Intent, QueryUnderstanding, SearchPlan, SearchQuery
from opinion_agent.text_utils import stable_id


class BaselineEvaluator:
    def __init__(self, executor: SearchExecutor | None = None) -> None:
        self.executor = executor or SearchExecutor()
        self.evidence_filter = EvidenceFilter()
        self.controller = IterativeSearchController()
        self.opinion_miner = OpinionMiner()
        self.planner = SearchPlanner()

    def evaluate(self, understanding: QueryUnderstanding, top_k: int = 5) -> dict[str, dict[str, float]]:
        return {
            "baseline_1_raw_query_only": self._run_plan(understanding, self._raw_query_plan(understanding), top_k),
            "baseline_2_llm_5_queries_no_taxonomy": self._run_plan(understanding, self._generic_5_query_plan(understanding), top_k),
            "baseline_3_single_round_no_replenish": self._run_plan(understanding, self.planner.create_initial_plan(understanding), top_k),
        }

    def _run_plan(self, understanding: QueryUnderstanding, plan: SearchPlan, top_k: int) -> dict[str, float]:
        round_result = self.executor.execute(plan, top_k=top_k)
        evidences, stats = self.evidence_filter.filter_rounds(understanding, [round_result])
        coverage = self.controller.build_coverage(understanding, evidences)
        target = understanding.main_entities[0].name if understanding.main_entities else understanding.raw_input
        opinion = self.opinion_miner.mine(evidences, target)
        total_results = sum(len(qr.results) for qr in round_result.query_results)
        return evaluate_session(
            understanding,
            evidences,
            coverage,
            opinion=opinion,
            total_results=total_results,
            duplicate_ratio=float(stats.get("duplicate_ratio", 0)),
        )

    def _raw_query_plan(self, understanding: QueryUnderstanding) -> SearchPlan:
        return SearchPlan(
            round=1,
            planner_type="baseline_raw_query",
            queries=[
                SearchQuery(
                    query_id=stable_id("bq", "raw", understanding.raw_input),
                    query=understanding.raw_input,
                    intent=Intent.FACT,
                    platform="general_web",
                    priority=1,
                    expected_evidence_type="raw_fact",
                    reason="Baseline: 只搜索用户原句",
                )
            ],
        )

    def _generic_5_query_plan(self, understanding: QueryUnderstanding) -> SearchPlan:
        entity = understanding.main_entities[0].name if understanding.main_entities else understanding.raw_input
        keywords = " ".join(understanding.event_keywords[:2]) or understanding.raw_input
        suffixes = ["", "最新", "怎么回事", "媒体报道", "网友评价"]
        queries = []
        for idx, suffix in enumerate(suffixes):
            query = f"{entity} {keywords} {suffix}".strip()
            queries.append(
                SearchQuery(
                    query_id=stable_id("bq", "generic5", idx, query),
                    query=query,
                    intent=Intent.FACT,
                    platform="general_web",
                    priority=idx + 1,
                    expected_evidence_type="generic",
                    reason="Baseline: 无 intent taxonomy 的 5 条通用 query",
                )
            )
        return SearchPlan(round=1, planner_type="baseline_generic_5", queries=queries)
