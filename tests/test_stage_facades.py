import importlib.util

import pytest

from opinion_agent.analysis import AnalysisStage, OpinionMiner, RiskAssessor, TextStatsAnalyzer, TimelineBuilder
from opinion_agent.llm import LLMError
from opinion_agent.logging import SessionLogger
from opinion_agent.reporting import PdfReportGenerator, ReportGenerator, ReportStage
from opinion_agent.research import (
    EvidenceFilter,
    MockSearchProvider,
    QueryUnderstandingEngine,
    ResearchStage,
    SearchExecutor,
    SearchPlanner,
)
from opinion_agent.schemas import CoverageMatrix, Entity, EventType, QueryUnderstanding, RiskLevel, SessionLog, TrainingViews


class FallbackLLM:
    def chat_json(self, system, user, fallback=None):
        return fallback


class ReportLLM:
    def __init__(self, content="# LLM 生成的事件与舆论速读\n\n这是结合材料写成的自然语言速读。") -> None:
        self.content = content
        self.calls = []

    def chat_text(self, system, user, fallback=None, temperature=0.35, max_tokens=None):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "fallback": fallback,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.content


class FailingReportLLM:
    model = "failing-model"

    def chat_text(self, system, user, fallback=None, temperature=0.35, max_tokens=None):
        raise LLMError("connection timed out")


def _research_result():
    stage = ResearchStage(
        understanding=QueryUnderstandingEngine(llm=FallbackLLM()),
        executor=SearchExecutor(provider=MockSearchProvider()),
    )
    return stage.run("某奶茶品牌被曝使用过期原料", max_rounds=1, top_k=3, query_budget=6)


def test_research_stage_runs_with_mock_search():
    result = _research_result()

    assert result.understanding.raw_input == "某奶茶品牌被曝使用过期原料"
    assert result.plans
    assert result.search_results
    assert result.evidences
    assert result.coverage.coverage
    assert result.decisions
    assert result.search_trace
    assert result.search_trace[0]["queries"][0]["requested_provider"] == "MockSearchProvider"
    assert result.filter_trace
    assert result.filter_trace[0]["results"]


def test_analysis_and_report_stages_run_from_research_output():
    research = _research_result()
    analysis = AnalysisStage().run(
        research.understanding,
        research.evidences,
        research.coverage,
        search_results=research.search_results,
        duplicate_ratio=float(research.filtering_stats.get("duplicate_ratio", 0)),
    )
    report = ReportStage(report_generator=ReportGenerator(use_llm=False)).run(
        research.understanding,
        research.evidences,
        research.coverage,
        analysis.opinion,
        analysis.timeline,
        analysis.risk,
    )

    assert analysis.opinion.sentiments
    assert analysis.timeline.timeline
    assert 0 <= analysis.risk.overall_risk_score <= 1
    assert analysis.text_analysis.evidence_count == len(research.evidences)
    assert analysis.text_analysis.sentiment_counts
    assert analysis.text_analysis.source_sentiment_matrix
    assert analysis.text_analysis.top_keywords
    assert "OpenOpinion 事件与舆论速读" in report
    assert "## 核心判断" in report


def test_report_generator_uses_llm_for_narrative_report():
    research = _research_result()
    analysis = AnalysisStage().run(research.understanding, research.evidences, research.coverage)
    llm = ReportLLM()
    generator = ReportGenerator(llm=llm)

    report = generator.generate(
        research.understanding,
        research.evidences,
        research.coverage,
        analysis.opinion,
        analysis.timeline,
        analysis.risk,
    )

    assert report == llm.content
    assert llm.calls
    assert "结构化材料" in llm.calls[0]["user"]
    assert "visual_evidence" in llm.calls[0]["user"]
    assert "chart_id" in llm.calls[0]["user"]
    assert "raw_data" in llm.calls[0]["user"]
    assert "不要输出 JSON" in llm.calls[0]["system"]
    assert "事件与舆论速读" in llm.calls[0]["system"]
    assert llm.calls[0]["fallback"] is None
    assert generator.last_trace["mode"] == "llm_narrative"
    assert generator.last_trace["used_llm"] is True
    assert any(item["chart_id"] == "sentiment_distribution" for item in generator.last_trace["visual_evidence"])
    assert "brief" in generator.last_trace
    assert "user_prompt" in generator.last_trace


def test_report_generator_falls_back_when_llm_is_disabled():
    research = _research_result()
    analysis = AnalysisStage().run(research.understanding, research.evidences, research.coverage)
    llm = ReportLLM()
    generator = ReportGenerator(llm=llm, use_llm=False)

    report = generator.generate(
        research.understanding,
        research.evidences,
        research.coverage,
        analysis.opinion,
        analysis.timeline,
        analysis.risk,
    )

    assert "OpenOpinion 事件与舆论速读" in report
    assert "## 核心判断" in report
    assert "## 事件脉络" in report
    assert "舆情分析报告" not in report
    assert llm.calls == []
    assert generator.last_trace["mode"] == "rule_template"
    assert generator.last_trace["used_llm"] is False
    assert any(item["chart_id"] == "top_keywords" for item in generator.last_trace["visual_evidence"])


def test_report_generator_raises_model_errors_instead_of_falling_back():
    research = _research_result()
    analysis = AnalysisStage().run(research.understanding, research.evidences, research.coverage)
    generator = ReportGenerator(llm=FailingReportLLM())

    with pytest.raises(LLMError, match="connection timed out"):
        generator.generate(
            research.understanding,
            research.evidences,
            research.coverage,
            analysis.opinion,
            analysis.timeline,
            analysis.risk,
        )

    assert generator.last_trace["mode"] == "llm_error"
    assert generator.last_trace["used_llm"] is True


def test_new_core_modules_export_expected_classes():
    assert QueryUnderstandingEngine.__name__ == "QueryUnderstandingEngine"
    assert SearchPlanner.__name__ == "SearchPlanner"
    assert SearchExecutor.__name__ == "SearchExecutor"
    assert MockSearchProvider.__name__ == "MockSearchProvider"
    assert EvidenceFilter.__name__ == "EvidenceFilter"
    assert OpinionMiner.__name__ == "OpinionMiner"
    assert RiskAssessor.__name__ == "RiskAssessor"
    assert TextStatsAnalyzer.__name__ == "TextStatsAnalyzer"
    assert TimelineBuilder.__name__ == "TimelineBuilder"
    assert PdfReportGenerator.__name__ == "PdfReportGenerator"
    assert ReportGenerator.__name__ == "ReportGenerator"


def test_empty_evidence_markdown_pdf_report_writes_pdf(tmp_path):
    understanding = QueryUnderstanding(
        raw_input="空证据测试",
        main_entities=[Entity(name="空证据测试")],
        event_type=EventType.OTHER,
        risk_level=RiskLevel.LOW,
    )
    analysis = AnalysisStage().run(understanding, [], CoverageMatrix())
    session = SessionLog(
        session_id="case_empty",
        created_at="2026-05-30T00:00:00Z",
        user_input="空证据测试",
        query_understanding=understanding,
        opinion_mining=analysis.opinion,
        timeline=analysis.timeline,
        risk_assessment=analysis.risk,
        text_analysis=analysis.text_analysis,
        evaluation=analysis.evaluation,
        training_views=TrainingViews(),
    )

    artifacts = SessionLogger(
        session_dir=tmp_path / "sessions",
        training_dir=tmp_path / "training",
        sqlite_path=tmp_path / "openopinion.sqlite3",
    ).save(session, "# 空证据测试\n\n## 核心判断\n\n暂无证据。")

    assert analysis.text_analysis.evidence_count == 0
    markdown_path = tmp_path / "sessions" / "case_empty.md"
    pdf_path = tmp_path / "sessions" / "case_empty.pdf"
    html_path = tmp_path / "sessions" / "case_empty.html"
    assert markdown_path.exists()
    assert pdf_path.exists()
    assert not html_path.exists()
    assert artifacts["report_markdown"] == str(markdown_path)
    assert artifacts["report_pdf"] == str(pdf_path)
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert "report_html" not in artifacts
    assert "chart_dir" not in artifacts
    assert "charts" not in artifacts


def test_old_stage_modules_are_removed():
    removed_modules = [
        "opinion_agent.understanding",
        "opinion_agent.planning",
        "opinion_agent.search",
        "opinion_agent.evidence",
        "opinion_agent.opinion",
        "opinion_agent.timeline",
        "opinion_agent.risk",
        "opinion_agent.report",
    ]
    assert all(importlib.util.find_spec(module) is None for module in removed_modules)
