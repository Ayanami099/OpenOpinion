from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opinion_agent.agent import OpinionAnalysisPipeline
from opinion_agent.config import project_path
from opinion_agent.eval import BaselineEvaluator
from opinion_agent.llm import LLMError
from opinion_agent.logging import SessionLogger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="opinion-agent", description="OpenOpinion event and public-opinion speed-read agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Run one event and public-opinion speed-read session")
    analyze_parser.add_argument("user_input", help="事件关键词或句子")
    analyze_parser.add_argument("--max-rounds", type=int, default=3)
    analyze_parser.add_argument("--top-k", type=int, default=5)
    analyze_parser.add_argument("--out", default="data/sessions")
    analyze_parser.add_argument("--query-budget", type=int, default=12)

    eval_parser = subparsers.add_parser("eval", help="Run benchmark cases and baseline comparison")
    eval_parser.add_argument("--cases", default="data/benchmark_cases.json")
    eval_parser.add_argument("--out", default="data/eval_results.json")

    export_parser = subparsers.add_parser("export-training", help="Export training JSONL files from saved sessions")
    export_parser.add_argument("--sessions", default="data/sessions")

    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            _cmd_analyze(args)
        elif args.command == "eval":
            _cmd_eval(args)
        elif args.command == "export-training":
            _cmd_export_training(args)
    except LLMError as exc:
        print(f"Model error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _cmd_analyze(args: argparse.Namespace) -> None:
    pipeline = OpinionAnalysisPipeline()
    result = pipeline.analyze(
        args.user_input,
        max_rounds=args.max_rounds,
        top_k=args.top_k,
        out_dir=args.out,
        query_budget=args.query_budget,
    )
    print(json.dumps({"session_id": result.session.session_id, "artifacts": result.artifacts, "evaluation": result.session.evaluation}, ensure_ascii=False, indent=2))


def _cmd_eval(args: argparse.Namespace) -> None:
    cases_path = _resolve_path(args.cases)
    out_path = _resolve_path(args.out)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    pipeline = OpinionAnalysisPipeline()
    baselines = BaselineEvaluator()
    runs = []
    for case in cases:
        result = pipeline.analyze(case["user_input"], max_rounds=3, top_k=5, out_dir="data/sessions")
        ours = result.session.evaluation
        runs.append(
            {
                "case_id": case.get("case_id"),
                "user_input": case["user_input"],
                "ours": ours,
                "baselines": baselines.evaluate(result.session.query_understanding, top_k=5),
                "session_id": result.session.session_id,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"eval_results": str(out_path), "case_count": len(runs)}, ensure_ascii=False, indent=2))


def _cmd_export_training(args: argparse.Namespace) -> None:
    logger = SessionLogger()
    counts = logger.export_training_from_sessions(_resolve_path(args.sessions))
    print(json.dumps({"exported": counts}, ensure_ascii=False, indent=2))


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_path(path)


if __name__ == "__main__":
    main()
