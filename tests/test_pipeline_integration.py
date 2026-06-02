import json
import sqlite3

from opinion_agent.agent import OpinionAnalysisPipeline
from opinion_agent.logging import SessionLogger
from opinion_agent.reporting import ReportGenerator, ReportStage
from opinion_agent.research import MockSearchProvider, QueryUnderstandingEngine, SearchExecutor


class FallbackLLM:
    def chat_json(self, system, user, fallback=None):
        return fallback


def test_pipeline_runs_with_mock_provider_and_writes_artifacts(tmp_path):
    logger = SessionLogger(session_dir=tmp_path / "sessions", training_dir=tmp_path / "training", sqlite_path=tmp_path / "openopinion.sqlite3")
    pipeline = OpinionAnalysisPipeline(
        understanding=QueryUnderstandingEngine(llm=FallbackLLM()),
        executor=SearchExecutor(provider=MockSearchProvider()),
        report_stage=ReportStage(report_generator=ReportGenerator(use_llm=False)),
        logger=logger,
    )
    result = pipeline.analyze("某奶茶品牌被曝使用过期原料", max_rounds=2, top_k=3, out_dir=str(tmp_path / "sessions"))
    assert result.session.evidence_pack
    assert result.session.agent_trace
    assert result.session.training_views.sft_query_planner_sample
    assert "证据" in result.report_markdown
    assert "OpenOpinion 事件与舆论速读" in result.report_markdown
    assert "visual_evidence" in result.session.final_report
    assert any(item["chart_id"] == "sentiment_distribution" for item in result.session.final_report["visual_evidence"])
    session_path = tmp_path / "sessions" / f"{result.session.session_id}.json"
    trace_path = tmp_path / "sessions" / f"{result.session.session_id}.trace.json"
    markdown_path = tmp_path / "sessions" / f"{result.session.session_id}.md"
    pdf_path = tmp_path / "sessions" / f"{result.session.session_id}.pdf"
    html_path = tmp_path / "sessions" / f"{result.session.session_id}.html"
    assert session_path.exists()
    assert trace_path.exists()
    assert markdown_path.exists()
    assert pdf_path.exists()
    assert not html_path.exists()
    assert result.artifacts["agent_trace"] == str(trace_path)
    assert result.artifacts["report_markdown"] == str(markdown_path)
    assert result.artifacts["report_pdf"] == str(pdf_path)
    assert "report_html" not in result.artifacts
    assert "chart_dir" not in result.artifacts
    assert "charts" not in result.artifacts
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert not (tmp_path / "training" / "sft_query_planner.jsonl").exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["trace_version"] == "agent_trace_v1"
    assert trace["summary"]["kept_evidence_count"] == len(result.session.evidence_pack)
    assert "weighted_negative_ratio" in trace["summary"]
    assert [step["step"] for step in trace["steps"]] == [
        "input_received",
        "query_understanding",
        "search_planning",
        "search_execution",
        "evidence_filtering",
        "analysis",
        "report_generation",
    ]
    assert trace["steps"][3]["data"]["provider_trace"]
    assert trace["steps"][4]["data"]["filter_trace"]
    assert trace["steps"][6]["data"]["mode"] == "rule_template"
    assert any(item["chart_id"] == "top_keywords" for item in trace["steps"][6]["data"]["visual_evidence"])
    saved_session = json.loads(session_path.read_text(encoding="utf-8"))
    assert any(item["chart_id"] == "quality_scores" for item in saved_session["final_report"]["visual_evidence"])
    assert saved_session["final_report"]["markdown_report_path"] == str(markdown_path)
    assert saved_session["final_report"]["pdf_report_path"] == str(pdf_path)
    assert "html_report_path" not in saved_session["final_report"]
    assert "chart_dir" not in saved_session["final_report"]
    assert "charts" not in saved_session["final_report"]

    with sqlite3.connect(tmp_path / "openopinion.sqlite3") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        evidence_count = conn.execute("SELECT COUNT(*) FROM evidences").fetchone()[0]
    assert session_count == 1
    assert evidence_count >= 1
