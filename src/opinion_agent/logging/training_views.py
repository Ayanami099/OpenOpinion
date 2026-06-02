from __future__ import annotations

from collections import defaultdict
from typing import Any

from opinion_agent.schemas import (
    CoverageMatrix,
    Evidence,
    QueryRewardLog,
    SearchPlan,
    SearchRoundResult,
    TrainingViews,
)


class TrainingViewBuilder:
    def build(
        self,
        user_input: str,
        query_understanding: dict[str, Any],
        plans: list[SearchPlan],
        results: list[SearchRoundResult],
        evidences: list[Evidence],
        coverage: CoverageMatrix,
        evaluation: dict[str, Any],
    ) -> TrainingViews:
        rewards = self._query_rewards(plans, results, evidences, coverage)
        plan_rewards = self._plan_rewards(plans, evidences, evaluation)
        return TrainingViews(
            sft_query_planner_sample=self._sft_sample(user_input, query_understanding, plans[0], evaluation),
            query_reward_logs=rewards,
            search_plan_rewards=plan_rewards,
            preference_samples=self._preference_samples(user_input, query_understanding, plans, plan_rewards),
            rl_trajectory=self._rl_trajectory(user_input, query_understanding, plans, results, rewards, coverage),
        )

    def _sft_sample(
        self,
        user_input: str,
        query_understanding: dict[str, Any],
        first_plan: SearchPlan,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "input": {
                "user_input": user_input,
                "query_understanding": {
                    "main_entities": [e["name"] for e in query_understanding.get("main_entities", [])],
                    "event_keywords": query_understanding.get("event_keywords", []),
                    "event_type": query_understanding.get("event_type"),
                    "risk_level": query_understanding.get("risk_level"),
                    "required_intents": query_understanding.get("required_intents", []),
                },
            },
            "output": {
                "round": first_plan.round,
                "queries": [
                    {
                        "query": q.query,
                        "intent": q.intent.value,
                        "platform": q.platform,
                        "priority": q.priority,
                    }
                    for q in first_plan.queries
                ],
            },
            "quality": {
                "intent_coverage": evaluation.get("intent_coverage", 0),
                "search_utility": evaluation.get("useful_evidence_rate", 0),
                "average_evidence_score": evaluation.get("average_evidence_score", 0),
                "redundancy_rate": evaluation.get("redundancy_rate", 0),
                "final_report_faithfulness": evaluation.get("report_faithfulness", 0.8),
                "human_approved": None,
            },
        }

    def _query_rewards(
        self,
        plans: list[SearchPlan],
        results: list[SearchRoundResult],
        evidences: list[Evidence],
        coverage: CoverageMatrix,
    ) -> list[QueryRewardLog]:
        result_counts: dict[str, int] = defaultdict(int)
        for search_round in results:
            for query_result in search_round.query_results:
                result_counts[query_result.query_id] += len(query_result.results)

        ev_by_query: dict[str, list[Evidence]] = defaultdict(list)
        for ev in evidences:
            ev_by_query[ev.from_query_id].append(ev)

        logs: list[QueryRewardLog] = []
        for plan in plans:
            for query in plan.queries:
                evs = ev_by_query.get(query.query_id, [])
                result_count = result_counts.get(query.query_id, 0)
                kept = len(evs)
                avg_rel = sum(ev.scores.relevance for ev in evs) / kept if kept else 0
                avg_cred = sum(ev.scores.credibility for ev in evs) / kept if kept else 0
                novelty = sum(ev.scores.novelty for ev in evs) / kept if kept else 0
                redundancy = 1 - (kept / result_count) if result_count else 0
                coverage_gain = 0.2 if coverage.coverage.get(query.intent) and kept else 0
                reward = max(0.0, min(1.0, 0.35 * avg_rel + 0.25 * avg_cred + 0.2 * novelty + coverage_gain - 0.15 * redundancy))
                logs.append(
                    QueryRewardLog(
                        query_id=query.query_id,
                        query=query.query,
                        intent=query.intent,
                        platform=query.platform,
                        round=plan.round,
                        result_count=result_count,
                        kept_evidence_count=kept,
                        avg_relevance=avg_rel,
                        avg_credibility=avg_cred,
                        novelty_gain=novelty,
                        intent_coverage_gain=coverage_gain,
                        redundancy_rate=max(0, redundancy),
                        query_reward=reward,
                    )
                )
        return logs

    def _plan_rewards(self, plans: list[SearchPlan], evidences: list[Evidence], evaluation: dict[str, Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for plan in plans:
            query_ids = {q.query_id for q in plan.queries}
            plan_evs = [ev for ev in evidences if ev.from_query_id in query_ids]
            avg_score = sum(ev.scores.overall for ev in plan_evs) / len(plan_evs) if plan_evs else 0
            reward = max(0.0, min(1.0, 0.35 * evaluation.get("intent_coverage", 0) + 0.25 * evaluation.get("source_diversity", 0) + 0.25 * avg_score - 0.15 * evaluation.get("redundancy_rate", 0)))
            output.append(
                {
                    "round": plan.round,
                    "query_ids": list(query_ids),
                    "intent_coverage": evaluation.get("intent_coverage", 0),
                    "source_diversity": evaluation.get("source_diversity", 0),
                    "viewpoint_coverage": evaluation.get("viewpoint_coverage", 0),
                    "avg_evidence_score": avg_score,
                    "redundancy_rate": evaluation.get("redundancy_rate", 0),
                    "search_cost": min(1.0, len(plan.queries) / 20),
                    "plan_reward": reward,
                }
            )
        return output

    def _preference_samples(
        self,
        user_input: str,
        query_understanding: dict[str, Any],
        plans: list[SearchPlan],
        rewards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not plans or not rewards:
            return []
        chosen_plan = plans[0]
        chosen_reward = rewards[0].get("plan_reward", 0)
        entity = query_understanding.get("main_entities", [{}])[0].get("name", user_input)
        keywords = " ".join(query_understanding.get("event_keywords", [])[:3]) or user_input
        rejected_queries = [
            {"query": f"{entity} {keywords}", "intent": "FACT"},
            {"query": f"{entity} {keywords} 最新", "intent": "FACT"},
            {"query": f"{entity} {keywords} 怎么回事", "intent": "FACT"},
        ]
        return [
            {
                "input": {"user_input": user_input, "query_understanding": query_understanding},
                "chosen": {
                    "queries": [{"query": q.query, "intent": q.intent.value} for q in chosen_plan.queries],
                    "plan_reward": chosen_reward,
                },
                "rejected": {"queries": rejected_queries, "plan_reward": max(0, chosen_reward - 0.35)},
                "preference_reason": "chosen 覆盖多意图和多平台，rejected 仅重复事实搜索，信息增益较低。",
            }
        ]

    def _rl_trajectory(
        self,
        user_input: str,
        query_understanding: dict[str, Any],
        plans: list[SearchPlan],
        results: list[SearchRoundResult],
        rewards: list[QueryRewardLog],
        coverage: CoverageMatrix,
    ) -> list[dict[str, Any]]:
        reward_by_round: dict[int, list[QueryRewardLog]] = defaultdict(list)
        for reward in rewards:
            reward_by_round[reward.round].append(reward)
        result_by_round = {item.round: item for item in results}
        trajectory: list[dict[str, Any]] = []
        searched: list[str] = []
        for idx, plan in enumerate(plans):
            round_results = result_by_round.get(plan.round)
            round_rewards = reward_by_round.get(plan.round, [])
            avg_reward = sum(r.query_reward for r in round_rewards) / len(round_rewards) if round_rewards else 0
            new_results = sum(len(qr.results) for qr in round_results.query_results) if round_results else 0
            new_evidence = sum(r.kept_evidence_count for r in round_rewards)
            state = {
                "user_input": user_input,
                "query_understanding": query_understanding,
                "searched_queries": list(searched),
                "intent_coverage": {intent.value: item.status.value for intent, item in coverage.coverage.items()},
                "round": plan.round,
            }
            action = {
                "action_type": "generate_queries",
                "queries": [{"query": q.query, "intent": q.intent.value} for q in plan.queries],
            }
            searched.extend(q.query for q in plan.queries)
            trajectory.append(
                {
                    "step": idx,
                    "state": state,
                    "action": action,
                    "observation": {
                        "new_results_count": new_results,
                        "new_evidence_count": new_evidence,
                        "high_quality_evidence_count": sum(1 for r in round_rewards if r.query_reward >= 0.7),
                    },
                    "reward": {
                        "step_reward": avg_reward,
                        "search_cost": min(1, len(plan.queries) / 20),
                    },
                    "next_state": {
                        "intent_coverage": {intent.value: item.status.value for intent, item in coverage.coverage.items()},
                        "round": plan.round + 1,
                    },
                    "done": idx == len(plans) - 1,
                }
            )
        return trajectory

