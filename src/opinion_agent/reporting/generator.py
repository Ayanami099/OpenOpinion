from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from typing import Any

from opinion_agent.llm import LLMError, OpenAICompatibleClient
from opinion_agent.reporting.prompts import NARRATIVE_REPORT_SYSTEM, build_narrative_report_user_prompt
from opinion_agent.schemas import (
    CoverageMatrix,
    Evidence,
    OpinionMiningResult,
    QueryUnderstanding,
    RiskAssessment,
    TextAnalysisResult,
    Timeline,
)


class ReportGenerator:
    def __init__(self, llm: OpenAICompatibleClient | None = None, use_llm: bool | None = None) -> None:
        self.llm = llm or OpenAICompatibleClient.from_env()
        self.use_llm = _env_flag("REPORT_USE_LLM", default=True) if use_llm is None else use_llm
        self.last_trace: dict[str, Any] = {}

    def generate(
        self,
        understanding: QueryUnderstanding,
        evidences: list[Evidence],
        coverage: CoverageMatrix,
        opinion: OpinionMiningResult,
        timeline: Timeline,
        risk: RiskAssessment,
        text_analysis: TextAnalysisResult | None = None,
    ) -> str:
        brief = self._build_brief(understanding, evidences, coverage, opinion, timeline, risk, text_analysis)
        fallback = self._generate_rule_report(understanding, evidences, coverage, opinion, timeline, risk, text_analysis)
        if not self.use_llm:
            self.last_trace = {
                "mode": "rule_template",
                "used_llm": False,
                "reason": "REPORT_USE_LLM disabled or ReportGenerator(use_llm=False)",
                "visual_evidence": brief.get("visual_evidence", []),
                "brief": brief,
                "fallback_chars": len(fallback),
            }
            return fallback
        user_prompt = build_narrative_report_user_prompt(brief)
        try:
            report = self.llm.chat_text(
                NARRATIVE_REPORT_SYSTEM,
                user_prompt,
                temperature=0.45,
            )
        except LLMError as exc:
            self.last_trace = {
                "mode": "llm_error",
                "used_llm": True,
                "reason": "llm_generation_failed",
                "error": str(exc),
                "llm_model": getattr(self.llm, "model", None),
                "system_prompt": NARRATIVE_REPORT_SYSTEM,
                "user_prompt": user_prompt,
                "visual_evidence": brief.get("visual_evidence", []),
                "brief": brief,
            }
            raise
        except (KeyError, IndexError, ValueError) as exc:
            self.last_trace = {
                "mode": "llm_error",
                "used_llm": True,
                "reason": "llm_generation_invalid_response",
                "error": str(exc),
                "llm_model": getattr(self.llm, "model", None),
                "system_prompt": NARRATIVE_REPORT_SYSTEM,
                "user_prompt": user_prompt,
                "visual_evidence": brief.get("visual_evidence", []),
                "brief": brief,
            }
            raise LLMError(f"Report model returned invalid output: {exc}") from exc
        cleaned = self._clean_llm_report(report)
        if not cleaned:
            self.last_trace = {
                "mode": "llm_error",
                "used_llm": True,
                "reason": "llm_returned_empty_report",
                "llm_model": getattr(self.llm, "model", None),
                "system_prompt": NARRATIVE_REPORT_SYSTEM,
                "user_prompt": user_prompt,
                "visual_evidence": brief.get("visual_evidence", []),
                "brief": brief,
            }
            raise LLMError("Report model returned an empty report")
        self.last_trace = {
            "mode": "llm_narrative",
            "used_llm": True,
            "llm_model": getattr(self.llm, "model", None),
            "system_prompt": NARRATIVE_REPORT_SYSTEM,
            "user_prompt": user_prompt,
            "visual_evidence": brief.get("visual_evidence", []),
            "brief": brief,
            "output_chars": len(cleaned),
            "fallback_chars": len(fallback),
        }
        return cleaned

    def _generate_rule_report(
        self,
        understanding: QueryUnderstanding,
        evidences: list[Evidence],
        coverage: CoverageMatrix,
        opinion: OpinionMiningResult,
        timeline: Timeline,
        risk: RiskAssessment,
        text_analysis: TextAnalysisResult | None = None,
    ) -> str:
        entity = understanding.main_entities[0].name if understanding.main_entities else understanding.raw_input
        text = text_analysis or TextAnalysisResult()
        top_evidence = self._top_evidence_ids(evidences)
        missing_intents = [intent.value for intent, item in coverage.coverage.items() if item.status.value == "missing"]
        lines: list[str] = []
        lines.append(f"# OpenOpinion 事件与舆论速读：{understanding.raw_input}")
        lines.append("")
        lines.append("## 核心判断")
        if evidences:
            lines.append(
                f"围绕“{entity}”的现有材料已形成初步事实线索和舆论样本。"
                f"{self._status(coverage)}；整体风险信号为 {risk.overall_risk_level.value}（{risk.overall_risk_score:.2f}）。"
                f"核心依据来自 {top_evidence}。"
            )
            if risk.risk_summary:
                lines.append(f"从当前信号看，{risk.risk_summary}")
        else:
            lines.append("现有材料不足以确认事件事实和舆论走向，需要先补充可靠来源后再做判断。")
        lines.append("")
        lines.append("## 事件脉络")
        if timeline.timeline:
            for item in timeline.timeline:
                evidence_ids = f"（证据：{', '.join(item.evidence_ids)}）" if item.evidence_ids else ""
                lines.append(f"- {item.time}，{item.stage}：{item.event}{evidence_ids}")
        else:
            lines.append("现有材料还不足以还原清晰时间线。")
        lines.append("")
        lines.append("## 已确认事实")
        if evidences:
            for intent, item in coverage.coverage.items():
                intent_evs = [ev for ev in evidences if ev.intent == intent]
                if intent_evs:
                    sample = intent_evs[0]
                    lines.append(f"- {intent.value}：{sample.summary or sample.title}（证据：{sample.evidence_id}）")
                elif item.status.value == "missing":
                    lines.append(f"- {intent.value}：现有材料不足以确认。")
        else:
            lines.append("现有材料不足以确认任何关键事实。")
        lines.append("")
        lines.append("## 舆论焦点")
        if text.top_keywords:
            top_terms = "、".join(str(item["term"]) for item in text.top_keywords[:8])
            lines.append(f"当前样本中的高频焦点集中在：{top_terms}。这些关键词可作为理解讨论重心的入口。")
        if opinion.opinion_clusters:
            for cluster in sorted(opinion.opinion_clusters, key=lambda item: item.importance_score, reverse=True)[:5]:
                lines.append(f"- {cluster.cluster_name}：{cluster.representative_claim}（证据：{', '.join(cluster.evidence_ids[:4])}）")
        elif not text.top_keywords:
            lines.append("现有材料不足以归纳稳定的舆论焦点。")
        lines.append("")
        lines.append("## 情绪与立场")
        if text.evidence_count:
            lines.append(
                f"统计样本共 {text.evidence_count} 条，平均情感分 {text.average_sentiment_score:.3f}，"
                f"加权负面率 {text.weighted_negative_ratio:.1%}。"
            )
            if text.sentiment_counts:
                lines.append(f"- 情感分布：{text.sentiment_counts}")
            if text.stance_distribution:
                lines.append(f"- 立场分布：{text.stance_distribution}")
            for stance in opinion.stances[:6]:
                lines.append(f"- {stance.target}：{stance.reason}（{stance.stance}，证据：{stance.evidence_id}）")
        else:
            lines.append("现有材料不足以判断公众情绪和主要立场。")
        lines.append("")
        lines.append("## 不确定信息")
        rumor_evs = [ev for ev in evidences if ev.intent.value == "RUMOR_CHECK"]
        if rumor_evs:
            lines.append(f"- 已检索到核查/澄清相关证据：{', '.join(ev.evidence_id for ev in rumor_evs[:5])}")
        else:
            lines.append("- 当前检索未发现可靠辟谣、反转或澄清证据。")
        if missing_intents:
            lines.append(f"- 信息缺口仍包括：{', '.join(missing_intents)}。")
        for key, dim in risk.dimensions.items():
            if dim.signals:
                lines.append(f"- {key} 信号：{dim.level.value}（{dim.score:.2f}），{'; '.join(dim.signals)}")
        lines.append("")
        lines.append("## 后续关注")
        lines.append("- 优先补齐官方通报、监管处理、涉事方说明和可核验的一手材料。")
        lines.append("- 继续观察舆论焦点是否从事实质疑转向责任追问、补偿诉求或制度性讨论。")
        lines.append("- 对社交平台二次传播、辟谣反转和高频关键词变化保持跟进。")
        if understanding.event_keywords:
            lines.append(f"- 可继续跟踪关键词：{', '.join(understanding.event_keywords[:8])}")
        lines.append("")
        return "\n".join(lines)

    def _build_brief(
        self,
        understanding: QueryUnderstanding,
        evidences: list[Evidence],
        coverage: CoverageMatrix,
        opinion: OpinionMiningResult,
        timeline: Timeline,
        risk: RiskAssessment,
        text_analysis: TextAnalysisResult | None = None,
    ) -> dict[str, Any]:
        top_evidences = self._rank_evidences(evidences)[:18]
        evidence_by_id = {ev.evidence_id: ev for ev in evidences}
        return {
            "task": "write_openopinion_event_opinion_speed_read",
            "report_style": "event_first_opinion_reading_markdown",
            "event": {
                "raw_input": understanding.raw_input,
                "entities": [entity.model_dump(mode="json") for entity in understanding.main_entities],
                "event_keywords": understanding.event_keywords,
                "event_type": understanding.event_type.value,
                "initial_risk_level": understanding.risk_level.value,
                "status": self._status(coverage),
            },
            "coverage": {
                intent.value: {
                    "status": item.status.value,
                    "evidence_count": item.evidence_count,
                    "high_quality_count": item.high_quality_count,
                }
                for intent, item in coverage.coverage.items()
            },
            "key_evidence": [self._evidence_note(ev) for ev in top_evidences],
            "evidence_by_intent": self._evidence_by_intent(top_evidences),
            "source_mix": self._source_mix(evidences),
            "timeline": [
                {
                    "time": item.time,
                    "stage": item.stage,
                    "event": item.event,
                    "evidence_ids": item.evidence_ids,
                    "confidence": item.confidence,
                }
                for item in timeline.timeline
            ],
            "opinion": {
                "sentiment_distribution": self._sentiment_distribution(opinion),
                "clusters": [
                    {
                        "name": cluster.cluster_name,
                        "representative_claim": self._clean_text(cluster.representative_claim, max_chars=220),
                        "sentiment_distribution": cluster.sentiment_distribution,
                        "stance_distribution": cluster.stance_distribution,
                        "keywords": cluster.keywords[:8],
                        "evidence_ids": cluster.evidence_ids[:8],
                        "source_platforms": cluster.source_platforms,
                        "importance_score": cluster.importance_score,
                    }
                    for cluster in sorted(opinion.opinion_clusters, key=lambda item: item.importance_score, reverse=True)
                ],
                "stance_examples": [
                    {
                        "target": stance.target,
                        "stance": stance.stance,
                        "reason": stance.reason,
                        "evidence_id": stance.evidence_id,
                        "evidence_title": evidence_by_id[stance.evidence_id].title if stance.evidence_id in evidence_by_id else "",
                    }
                    for stance in opinion.stances[:10]
                ],
            },
            "text_analysis": text_analysis.model_dump(mode="json") if text_analysis else {},
            "visual_evidence": self._build_visual_evidence(coverage, risk, text_analysis),
            "risk": {
                "overall_level": risk.overall_risk_level.value,
                "overall_score": risk.overall_risk_score,
                "summary": risk.risk_summary,
                "dimensions": {
                    name: {
                        "score": dim.score,
                        "level": dim.level.value,
                        "signals": dim.signals,
                        "extra": dim.extra,
                    }
                    for name, dim in risk.dimensions.items()
                },
            },
            "writing_constraints": [
                "以事件速读为主，不要写成舆情检测或风险监测报告。",
                "正文以自然段落为主，少用表格。",
                "把 visual_evidence 当作支撑段落的证据，不要单独堆统计区。",
                "事实、媒体报道、社交评论和网传推测必须分层表达。",
                "结尾给出可执行的后续关注点。",
            ],
        }

    def _build_visual_evidence(
        self,
        coverage: CoverageMatrix,
        risk: RiskAssessment,
        text_analysis: TextAnalysisResult | None,
    ) -> list[dict[str, Any]]:
        text = text_analysis or TextAnalysisResult()
        risk_dimensions = {
            name: {
                "score": dim.score,
                "level": dim.level.value,
                "signals": dim.signals,
                "extra": dim.extra,
            }
            for name, dim in risk.dimensions.items()
        }
        coverage_rows = {
            intent.value: {
                "status": item.status.value,
                "evidence_count": item.evidence_count,
                "high_quality_count": item.high_quality_count,
            }
            for intent, item in coverage.coverage.items()
        }
        return [
            {
                "chart_id": "overview_metrics",
                "table_id": "overview_metrics",
                "kind": "table",
                "title": "速读指标",
                "section": "核心判断",
                "raw_data": {
                    "evidence_count": text.evidence_count,
                    "average_sentiment_score": text.average_sentiment_score,
                    "weighted_negative_ratio": text.weighted_negative_ratio,
                    "overall_risk_level": risk.overall_risk_level.value,
                    "overall_risk_score": risk.overall_risk_score,
                },
                "supports": ["说明样本规模、情绪方向和风险信号强弱，避免只凭单条材料下结论。"],
            },
            {
                "chart_id": "date_trend",
                "kind": "chart",
                "title": "时间趋势",
                "section": "事件脉络",
                "raw_data": text.date_distribution,
                "supports": ["观察材料集中出现的时间点，辅助判断事件发酵或报道节奏。"],
            },
            {
                "chart_id": "coverage_table",
                "table_id": "coverage_table",
                "kind": "table",
                "title": "证据覆盖",
                "section": "已确认事实",
                "raw_data": coverage_rows,
                "supports": ["区分哪些事实已有证据覆盖，哪些仍是信息缺口。"],
            },
            {
                "chart_id": "top_keywords",
                "kind": "chart",
                "title": "Top 关键词",
                "section": "舆论焦点",
                "raw_data": text.top_keywords[:20],
                "supports": ["概括讨论中反复出现的对象、问题和情绪词。"],
            },
            {
                "chart_id": "aspect_sentiment",
                "kind": "chart",
                "title": "观点面向 × 情感",
                "section": "舆论焦点",
                "raw_data": text.aspect_sentiment_matrix,
                "supports": ["识别不同争议面向对应的情绪强度和分化方向。"],
            },
            {
                "chart_id": "sentiment_distribution",
                "kind": "chart",
                "title": "情感分布",
                "section": "情绪与立场",
                "raw_data": {
                    "counts": text.sentiment_counts,
                    "ratios": text.sentiment_ratios,
                    "average_sentiment_score": text.average_sentiment_score,
                    "weighted_negative_ratio": text.weighted_negative_ratio,
                },
                "supports": ["说明整体情绪是偏负面、中性、混合还是分散。"],
            },
            {
                "chart_id": "stance_distribution",
                "kind": "chart",
                "title": "立场分布",
                "section": "情绪与立场",
                "raw_data": text.stance_distribution,
                "supports": ["展示不同立场类型的数量，辅助判断舆论阵营。"],
            },
            {
                "chart_id": "source_sentiment",
                "kind": "chart",
                "title": "来源类型 × 情感",
                "section": "情绪与立场",
                "raw_data": text.source_sentiment_matrix,
                "supports": ["比较媒体、官方、社交平台等来源中的情绪差异。"],
            },
            {
                "chart_id": "risk_dimensions",
                "kind": "chart",
                "title": "风险维度",
                "section": "不确定信息",
                "raw_data": {
                    "overall_level": risk.overall_risk_level.value,
                    "overall_score": risk.overall_risk_score,
                    "dimensions": risk_dimensions,
                },
                "supports": ["把风险信号作为不确定性和后续关注优先级的依据，而不是报告主轴。"],
            },
            {
                "chart_id": "quality_scores",
                "kind": "chart",
                "title": "证据质量",
                "section": "不确定信息",
                "raw_data": text.quality_averages,
                "supports": ["说明当前材料的相关性、可信度和新鲜度，提示结论可靠边界。"],
            },
        ]

    def _status(self, coverage: CoverageMatrix) -> str:
        missing = [intent.value for intent, item in coverage.coverage.items() if item.status.value == "missing"]
        if missing:
            return f"信息仍有缺口：{', '.join(missing)}"
        return "核心信息已有证据覆盖"

    def _top_evidence_ids(self, evidences: list[Evidence]) -> str:
        top = sorted(evidences, key=lambda ev: ev.scores.overall, reverse=True)[:5]
        return ", ".join(ev.evidence_id for ev in top) or "当前检索未发现可靠证据"

    def _rank_evidences(self, evidences: list[Evidence]) -> list[Evidence]:
        return sorted(
            evidences,
            key=lambda ev: (
                ev.scores.overall,
                ev.scores.credibility,
                ev.scores.relevance,
                ev.source_type.value in {"official", "government", "mainstream_media"},
            ),
            reverse=True,
        )

    def _evidence_note(self, ev: Evidence) -> dict[str, Any]:
        return {
            "evidence_id": ev.evidence_id,
            "intent": ev.intent.value,
            "platform": ev.platform,
            "source": ev.source,
            "source_type": ev.source_type.value,
            "title": self._clean_text(ev.title, max_chars=120),
            "url": ev.url,
            "published_at": ev.published_at,
            "summary": self._clean_text(ev.summary or ev.content, max_chars=260),
            "excerpt": self._clean_text(ev.content, max_chars=360),
            "entities": ev.entities,
            "event_mentions": ev.event_mentions,
            "scores": {
                "overall": round(ev.scores.overall, 3),
                "credibility": round(ev.scores.credibility, 3),
                "relevance": round(ev.scores.relevance, 3),
            },
        }

    def _evidence_by_intent(self, evidences: list[Evidence]) -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for ev in evidences:
            grouped[ev.intent.value].append(
                {
                    "evidence_id": ev.evidence_id,
                    "source_type": ev.source_type.value,
                    "title": self._clean_text(ev.title, max_chars=100),
                    "summary": self._clean_text(ev.summary or ev.content, max_chars=180),
                }
            )
        return dict(grouped)

    def _source_mix(self, evidences: list[Evidence]) -> dict[str, int]:
        return dict(Counter(ev.source_type.value for ev in evidences))

    def _sentiment_distribution(self, opinion: OpinionMiningResult) -> dict[str, float]:
        counts = Counter(item.sentiment for item in opinion.sentiments)
        total = sum(counts.values())
        if not total:
            return {}
        return {key: round(value / total, 3) for key, value in counts.items()}

    def _clean_llm_report(self, report: str) -> str:
        cleaned = report.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:markdown|md)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        return cleaned

    def _clean_text(self, text: str, max_chars: int = 240) -> str:
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"https?://\S+", " ", cleaned)
        cleaned = re.sub(r"#+\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1] + "..."


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
