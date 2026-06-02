from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from opinion_agent.config import project_path
from opinion_agent.reporting.pdf_report import PdfReportGenerator
from opinion_agent.schemas import SessionLog


class SessionLogger:
    def __init__(
        self,
        session_dir: str | Path = "data/sessions",
        training_dir: str | Path = "data/training",
        sqlite_path: str | Path = "data/openopinion.sqlite3",
        write_training: bool = False,
    ) -> None:
        self.session_dir = project_path(str(session_dir)) if not Path(session_dir).is_absolute() else Path(session_dir)
        self.training_dir = project_path(str(training_dir)) if not Path(training_dir).is_absolute() else Path(training_dir)
        self.sqlite_path = project_path(str(sqlite_path)) if not Path(sqlite_path).is_absolute() else Path(sqlite_path)
        self.write_training = write_training
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self.write_training:
            self.training_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def save(self, session: SessionLog, report_markdown: str) -> dict[str, Any]:
        session_path = self.session_dir / f"{session.session_id}.json"
        report_path = self.session_dir / f"{session.session_id}.md"
        pdf_path = self.session_dir / f"{session.session_id}.pdf"
        trace_path = self.session_dir / f"{session.session_id}.trace.json"
        PdfReportGenerator().generate(session, report_markdown, pdf_path)
        data = session.model_dump(mode="json")
        data.setdefault("final_report", {})
        data["final_report"]["markdown_report_path"] = str(report_path)
        data["final_report"]["pdf_report_path"] = str(pdf_path)
        session_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(report_markdown, encoding="utf-8")
        trace_path.write_text(json.dumps(data.get("agent_trace", {}), ensure_ascii=False, indent=2), encoding="utf-8")
        if self.write_training:
            self._append_jsonl(self.training_dir / "stage1_sessions.jsonl", data)
            self._append_training_views(session)
        self._write_sqlite(session, str(report_path))
        return {
            "session_json": str(session_path),
            "report_markdown": str(report_path),
            "report_pdf": str(pdf_path),
            "agent_trace": str(trace_path),
            "sqlite": str(self.sqlite_path),
        }

    def export_training_from_sessions(self, sessions_dir: str | Path | None = None) -> dict[str, int]:
        sessions_dir = Path(sessions_dir) if sessions_dir else self.session_dir
        counts = {"sft": 0, "preference": 0, "rl": 0}
        for path in sessions_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            views = data.get("training_views", {})
            if views.get("sft_query_planner_sample") and self._is_high_quality_sft(views["sft_query_planner_sample"]):
                self._append_jsonl(self.training_dir / "sft_query_planner.jsonl", views["sft_query_planner_sample"])
                counts["sft"] += 1
            for sample in views.get("preference_samples", []):
                self._append_jsonl(self.training_dir / "preference_pairs.jsonl", sample)
                counts["preference"] += 1
            if views.get("rl_trajectory"):
                self._append_jsonl(self.training_dir / "rl_trajectories.jsonl", {"session_id": data["session_id"], "trajectory": views["rl_trajectory"]})
                counts["rl"] += 1
        return counts

    def _append_training_views(self, session: SessionLog) -> None:
        views = session.training_views.model_dump(mode="json")
        if views.get("sft_query_planner_sample") and self._is_high_quality_sft(views["sft_query_planner_sample"]):
            self._append_jsonl(self.training_dir / "sft_query_planner.jsonl", views["sft_query_planner_sample"])
        for sample in views.get("preference_samples", []):
            self._append_jsonl(self.training_dir / "preference_pairs.jsonl", sample)
        if views.get("rl_trajectory"):
            self._append_jsonl(self.training_dir / "rl_trajectories.jsonl", {"session_id": session.session_id, "trajectory": views["rl_trajectory"]})

    def _is_high_quality_sft(self, sample: dict[str, Any]) -> bool:
        quality = sample.get("quality", {})
        return (
            quality.get("intent_coverage", 0) >= 0.75
            and quality.get("search_utility", 0) >= 0.60
            and quality.get("average_evidence_score", 0) >= 0.65
            and quality.get("redundancy_rate", 1) <= 0.40
            and quality.get("final_report_faithfulness", 0) >= 0.75
        )

    def _append_jsonl(self, path: Path, item: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _init_db(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    user_input TEXT,
                    event_type TEXT,
                    risk_level TEXT,
                    final_risk_score REAL,
                    final_report_path TEXT
                );
                CREATE TABLE IF NOT EXISTS queries (
                    query_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    round INTEGER,
                    query TEXT,
                    intent TEXT,
                    platform TEXT,
                    priority INTEGER,
                    planner_type TEXT
                );
                CREATE TABLE IF NOT EXISTS search_results (
                    result_id TEXT PRIMARY KEY,
                    query_id TEXT,
                    title TEXT,
                    url TEXT,
                    source TEXT,
                    snippet TEXT,
                    published_at TEXT,
                    retrieved_at TEXT,
                    raw_content TEXT
                );
                CREATE TABLE IF NOT EXISTS evidences (
                    evidence_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query_id TEXT,
                    result_id TEXT,
                    intent TEXT,
                    source_type TEXT,
                    platform TEXT,
                    summary TEXT,
                    relevance REAL,
                    credibility REAL,
                    freshness REAL,
                    intent_match REAL,
                    novelty REAL,
                    overall_score REAL,
                    filter_status TEXT
                );
                CREATE TABLE IF NOT EXISTS opinions (
                    opinion_id TEXT PRIMARY KEY,
                    evidence_id TEXT,
                    aspect TEXT,
                    claim TEXT,
                    sentiment TEXT,
                    stance TEXT,
                    confidence REAL,
                    cluster_id TEXT
                );
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    session_id TEXT PRIMARY KEY,
                    heat_score REAL,
                    negative_ratio REAL,
                    spread_speed_score REAL,
                    authority_involvement_score REAL,
                    overall_risk_score REAL
                );
                """
            )

    def _write_sqlite(self, session: SessionLog, report_path: str) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, created_at, user_input, event_type, risk_level, final_risk_score, final_report_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.created_at,
                    session.user_input,
                    session.query_understanding.event_type.value,
                    session.risk_assessment.overall_risk_level.value,
                    session.risk_assessment.overall_risk_score,
                    report_path,
                ),
            )
            for plan in session.search_plan_rounds:
                for query in plan.queries:
                    conn.execute(
                        "INSERT OR REPLACE INTO queries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (query.query_id, session.session_id, plan.round, query.query, query.intent.value, query.platform, query.priority, plan.planner_type),
                    )
            for search_round in session.search_results:
                for query_result in search_round.query_results:
                    for result in query_result.results:
                        conn.execute(
                            "INSERT OR REPLACE INTO search_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                result.result_id,
                                query_result.query_id,
                                result.title,
                                result.url,
                                result.source,
                                result.snippet,
                                result.published_at,
                                result.retrieved_at,
                                result.raw_content,
                            ),
                        )
            for ev in session.evidence_pack:
                conn.execute(
                    "INSERT OR REPLACE INTO evidences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ev.evidence_id,
                        session.session_id,
                        ev.from_query_id,
                        ev.source_result_id,
                        ev.intent.value,
                        ev.source_type.value,
                        ev.platform,
                        ev.summary,
                        ev.scores.relevance,
                        ev.scores.credibility,
                        ev.scores.freshness,
                        ev.scores.intent_match,
                        ev.scores.novelty,
                        ev.scores.overall,
                        ev.filter_status,
                    ),
                )
            cluster_by_ev = {}
            for cluster in session.opinion_mining.opinion_clusters:
                for ev_id in cluster.evidence_ids:
                    cluster_by_ev[ev_id] = cluster.cluster_id
            for idx, opinion in enumerate(session.opinion_mining.opinion_units, start=1):
                conn.execute(
                    "INSERT OR REPLACE INTO opinions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"{session.session_id}_op_{idx:04d}",
                        opinion.evidence_id,
                        opinion.aspect,
                        opinion.claim,
                        opinion.sentiment,
                        opinion.stance,
                        opinion.confidence,
                        cluster_by_ev.get(opinion.evidence_id),
                    ),
                )
            dims = session.risk_assessment.dimensions
            conn.execute(
                "INSERT OR REPLACE INTO risk_metrics VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session.session_id,
                    dims.get("heat").score if dims.get("heat") else 0,
                    dims.get("negative_ratio").score if dims.get("negative_ratio") else 0,
                    dims.get("spread_speed").score if dims.get("spread_speed") else 0,
                    dims.get("authority_involvement").score if dims.get("authority_involvement") else 0,
                    session.risk_assessment.overall_risk_score,
                ),
            )
