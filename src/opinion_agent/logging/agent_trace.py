from __future__ import annotations

from collections import Counter
from typing import Any


class AgentTraceBuilder:
    def build(
        self,
        session_id: str,
        created_at: str,
        user_input: str,
        run_config: dict[str, Any],
        research: Any,
        analysis: Any,
        report_trace: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "trace_version": "agent_trace_v1",
            "session_id": session_id,
            "created_at": created_at,
            "user_input": user_input,
            "run_config": run_config,
            "summary": self._summary(research, analysis, report_trace),
            "steps": [
                self._input_step(user_input, run_config),
                self._understanding_step(research),
                self._planning_step(research),
                self._search_step(research),
                self._filtering_step(research),
                self._analysis_step(analysis),
                self._report_step(report_trace),
            ],
        }

    def _summary(self, research: Any, analysis: Any, report_trace: dict[str, Any]) -> dict[str, Any]:
        query_count = sum(len(plan.queries) for plan in research.plans)
        result_count = sum(len(qr.results) for sr in research.search_results for qr in sr.query_results)
        kept_count = len(research.evidences)
        return {
            "event_type": research.understanding.event_type.value,
            "risk_level": analysis.risk.overall_risk_level.value,
            "risk_score": analysis.risk.overall_risk_score,
            "query_count": query_count,
            "search_result_count": result_count,
            "kept_evidence_count": kept_count,
            "average_sentiment_score": analysis.text_analysis.average_sentiment_score,
            "weighted_negative_ratio": analysis.text_analysis.weighted_negative_ratio,
            "report_mode": report_trace.get("mode"),
            "report_used_llm": report_trace.get("used_llm"),
        }

    def _input_step(self, user_input: str, run_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": "input_received",
            "description": "接收用户输入和运行参数。",
            "data": {
                "user_input": user_input,
                "run_config": run_config,
            },
        }

    def _understanding_step(self, research: Any) -> dict[str, Any]:
        understanding = research.understanding
        return {
            "step": "query_understanding",
            "description": "解析输入，得到主体、事件关键词、事件类型、风险初判和检索意图。",
            "data": {
                "raw_input": understanding.raw_input,
                "main_entities": [entity.model_dump(mode="json") for entity in understanding.main_entities],
                "event_keywords": understanding.event_keywords,
                "event_type": understanding.event_type.value,
                "risk_level": understanding.risk_level.value,
                "possible_platforms": understanding.possible_platforms,
                "required_intents": [intent.value for intent in understanding.required_intents],
                "optional_intents": [intent.value for intent in understanding.optional_intents],
                "known_constraints": understanding.known_constraints,
            },
        }

    def _planning_step(self, research: Any) -> dict[str, Any]:
        return {
            "step": "search_planning",
            "description": "根据 query understanding 生成首轮与补搜轮次的多意图搜索计划。",
            "data": {
                "plans": [
                    {
                        "round": plan.round,
                        "planner_type": plan.planner_type,
                        "queries": [
                            {
                                "query_id": query.query_id,
                                "query": query.query,
                                "intent": query.intent.value,
                                "platform": query.platform,
                                "priority": query.priority,
                                "expected_evidence_type": query.expected_evidence_type,
                                "reason": query.reason,
                            }
                            for query in plan.queries
                        ],
                    }
                    for plan in research.plans
                ],
                "iterative_decisions": research.decisions,
            },
        }

    def _search_step(self, research: Any) -> dict[str, Any]:
        return {
            "step": "search_execution",
            "description": "执行每条搜索 query，记录 provider、fallback、返回结果和检索到的页面摘要。",
            "data": {
                "provider_trace": research.search_trace,
                "rounds": [
                    {
                        "round": search_round.round,
                        "query_results": [
                            {
                                "query_id": query_result.query_id,
                                "query": query_result.query,
                                "intent": query_result.intent.value,
                                "platform": query_result.platform,
                                "result_count": len(query_result.results),
                                "results": [
                                    {
                                        "result_id": result.result_id,
                                        "rank": result.rank,
                                        "title": result.title,
                                        "url": result.url,
                                        "source": result.source,
                                        "snippet": result.snippet,
                                        "published_at": result.published_at,
                                        "retrieved_at": result.retrieved_at,
                                        "content_length": result.content_length,
                                    }
                                    for result in query_result.results
                                ],
                            }
                            for query_result in search_round.query_results
                        ],
                    }
                    for search_round in research.search_results
                ],
            },
        }

    def _filtering_step(self, research: Any) -> dict[str, Any]:
        return {
            "step": "evidence_filtering",
            "description": "对搜索结果做去重、相关性、可信度、新鲜度和意图匹配打分，形成证据包。",
            "data": {
                "filtering_stats": research.filtering_stats,
                "filter_trace": research.filter_trace,
                "coverage": {
                    intent.value: item.model_dump(mode="json")
                    for intent, item in research.coverage.coverage.items()
                },
                "kept_evidences": [
                    {
                        "evidence_id": ev.evidence_id,
                        "source_result_id": ev.source_result_id,
                        "from_query_id": ev.from_query_id,
                        "intent": ev.intent.value,
                        "platform": ev.platform,
                        "title": ev.title,
                        "url": ev.url,
                        "source": ev.source,
                        "source_type": ev.source_type.value,
                        "summary": ev.summary,
                        "entities": ev.entities,
                        "event_mentions": ev.event_mentions,
                        "scores": ev.scores.model_dump(mode="json"),
                        "filter_reasons": ev.filter_reasons,
                    }
                    for ev in research.evidences
                ],
            },
        }

    def _analysis_step(self, analysis: Any) -> dict[str, Any]:
        sentiments = Counter(item.sentiment for item in analysis.opinion.sentiments)
        return {
            "step": "analysis",
            "description": "从证据中抽取情绪、立场、观点聚类、事件时间线和风险维度。",
            "data": {
                "sentiment_counts": dict(sentiments),
                "opinion_clusters": [cluster.model_dump(mode="json") for cluster in analysis.opinion.opinion_clusters],
                "stances": [stance.model_dump(mode="json") for stance in analysis.opinion.stances],
                "timeline": analysis.timeline.model_dump(mode="json"),
                "risk_assessment": analysis.risk.model_dump(mode="json"),
                "text_analysis": analysis.text_analysis.model_dump(mode="json"),
                "evaluation": analysis.evaluation,
            },
        }

    def _report_step(self, report_trace: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": "report_generation",
            "description": "记录报告生成方式、是否调用 LLM、使用的 brief 和提示词；LLM 失败时记录规则兜底。",
            "data": report_trace,
        }


__all__ = ["AgentTraceBuilder"]
