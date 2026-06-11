from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys


INVALID_PATH_CHARS = r'[<>:"/\\|?*\r\n\t]'
SUPPORTED_PLATFORMS = {"tavily", "shuiyuan", "zhihu"}


def safe_topic_name(topic: str, max_len: int = 40) -> str:
    name = topic.strip()
    name = re.sub(INVALID_PATH_CHARS, "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._ ")
    if not name:
        name = "case"
    return name[:max_len]


def parse_platforms(raw: str) -> list[str]:
    raw = raw.strip().lower()
    if raw in {"all", "*"}:
        return ["tavily", "shuiyuan", "zhihu"]

    platforms = [p.strip().lower() for p in raw.split(",") if p.strip()]
    unknown = [p for p in platforms if p not in SUPPORTED_PLATFORMS]
    if unknown:
        raise ValueError(
            f"Unsupported platform(s): {unknown}. "
            f"Supported: {sorted(SUPPORTED_PLATFORMS)}"
        )
    if not platforms:
        raise ValueError("No platform selected.")
    return platforms


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_default_out_dir(topic: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = safe_topic_name(topic)
    return Path("data") / f"session_{topic_slug}_{timestamp}"


def run_and_log(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    print("Running:", " ".join(command))

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)

    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="openopinion-run-topic",
        description="Create one isolated session directory for one topic and selected platforms.",
    )
    parser.add_argument("topic", help="事件主题，例如：樊思睿")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="本次主题的保存目录；不填则自动生成 data/session_主题_时间戳",
    )
    parser.add_argument(
        "--platforms",
        default="tavily",
        help="要运行的平台，用逗号分隔，例如 tavily,shuiyuan,zhihu；也可以填 all",
    )
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--query-budget", type=int, default=4)

    parser.add_argument(
        "--shuiyuan-config",
        default="configs/campus_config.local.json",
        help="水源配置文件路径，默认 configs/campus_config.local.json",
    )
    parser.add_argument(
        "--shuiyuan-max-results",
        type=int,
        default=20,
        help="水源搜索最多返回条数，默认 20",
    )


    parser.add_argument(
        "--mediacrawler-python-exe",
        default=".venv_mediacrawler\\Scripts\\python.exe",
        help="?? MediaCrawler ? Python ????? .venv_mediacrawler\\Scripts\\python.exe",
    )
    parser.add_argument(
        "--media-crawler-repo",
        default="external\\MediaCrawler",
        help="MediaCrawler ??????? external\\MediaCrawler",
    )

    args = parser.parse_args()

    try:
        platforms = parse_platforms(args.platforms)
    except ValueError as exc:
        print(f"Platform error: {exc}", file=sys.stderr)
        return 2

    session_root = Path(args.out_dir) if args.out_dir else build_default_out_dir(args.topic)

    search_results_dir = session_root / "search_results"
    tavily_dir = search_results_dir / "tavily"
    multi_platform_dir = search_results_dir / "multi_platform"
    shuiyuan_dir = multi_platform_dir / "shuiyuan"
    zhihu_dir = multi_platform_dir / "zhihu"
    text_analysis_dir = session_root / "text_analysis"
    reports_dir = session_root / "reports"
    logs_dir = session_root / "logs"

    for directory in [
        tavily_dir,
        shuiyuan_dir,
        zhihu_dir,
        text_analysis_dir,
        reports_dir,
        logs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "topic": args.topic,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_platforms": platforms,
        "session_root": str(session_root),
        "search_results_dir": str(search_results_dir),
        "tavily_dir": str(tavily_dir),
        "multi_platform_dir": str(multi_platform_dir),
        "shuiyuan_dir": str(shuiyuan_dir),
        "zhihu_dir": str(zhihu_dir),
        "text_analysis_dir": str(text_analysis_dir),
        "reports_dir": str(reports_dir),
        "logs_dir": str(logs_dir),
        "steps": [],
    }

    manifest_path = session_root / "manifest.json"
    write_json(manifest_path, manifest)

    print(f"SESSION_ROOT = {session_root}")
    print(f"SELECTED_PLATFORMS = {platforms}")
    print(f"TAVILY_DIR = {tavily_dir}")
    print(f"SHUIYUAN_DIR = {shuiyuan_dir}")
    print(f"ZHIHU_DIR = {zhihu_dir}")
    print(f"TEXT_ANALYSIS_DIR = {text_analysis_dir}")
    print(f"REPORTS_DIR = {reports_dir}")
    print(f"MANIFEST = {manifest_path}")

    final_returncode = 0

    if "tavily" in platforms:
        command = [
            sys.executable,
            "-m",
            "opinion_agent.cli",
            "analyze",
            args.topic,
            "--max-rounds",
            str(args.max_rounds),
            "--top-k",
            str(args.top_k),
            "--query-budget",
            str(args.query_budget),
            "--out",
            str(tavily_dir),
        ]

        returncode = run_and_log(
            command,
            logs_dir / "opinion_agent.stdout.log",
            logs_dir / "opinion_agent.stderr.log",
        )

        manifest["steps"].append(
            {
                "name": "opinion_agent_analyze",
                "platform": "tavily",
                "command": command,
                "returncode": returncode,
                "stdout_log": str(logs_dir / "opinion_agent.stdout.log"),
                "stderr_log": str(logs_dir / "opinion_agent.stderr.log"),
                "output_dir": str(tavily_dir),
            }
        )
        write_json(manifest_path, manifest)

        if returncode != 0:
            final_returncode = returncode
            print(f"Tavily/OpenOpinion failed. Session directory was still created: {session_root}", file=sys.stderr)

    if "shuiyuan" in platforms:
        topic_slug = safe_topic_name(args.topic)
        shuiyuan_out = shuiyuan_dir / f"search_{topic_slug}.jsonl"

        command = [
            sys.executable,
            str(Path("scripts") / "fetch_shuiyuan.py"),
            "--config",
            args.shuiyuan_config,
            "search",
            args.topic,
            "--max-results",
            str(args.shuiyuan_max_results),
            "--out",
            str(shuiyuan_out),
        ]

        returncode = run_and_log(
            command,
            logs_dir / "shuiyuan.stdout.log",
            logs_dir / "shuiyuan.stderr.log",
        )

        manifest["steps"].append(
            {
                "name": "fetch_shuiyuan_search",
                "platform": "shuiyuan",
                "command": command,
                "returncode": returncode,
                "stdout_log": str(logs_dir / "shuiyuan.stdout.log"),
                "stderr_log": str(logs_dir / "shuiyuan.stderr.log"),
                "output_file": str(shuiyuan_out),
            }
        )
        write_json(manifest_path, manifest)

        if returncode != 0:
            final_returncode = returncode
            print(f"Shuiyuan search failed. Session directory was still created: {session_root}", file=sys.stderr)

    if "zhihu" in platforms:
        command = [
            sys.executable,
            str(Path("scripts") / "run_mediacrawler.py"),
            "--no-conda",
            "--python-exe",
            args.mediacrawler_python_exe,
            "--media-crawler-repo",
            args.media_crawler_repo,
            "--output-dir",
            str(zhihu_dir),
            "--platform",
            "zhihu",
            "--type",
            "search",
            "--keywords",
            args.topic,
            "--save_data_option",
            "jsonl",
        ]

        returncode = run_and_log(
            command,
            logs_dir / "zhihu.stdout.log",
            logs_dir / "zhihu.stderr.log",
        )

        manifest["steps"].append(
            {
                "name": "fetch_zhihu_search",
                "platform": "zhihu",
                "command": command,
                "returncode": returncode,
                "stdout_log": str(logs_dir / "zhihu.stdout.log"),
                "stderr_log": str(logs_dir / "zhihu.stderr.log"),
                "output_dir": str(zhihu_dir),
            }
        )
        write_json(manifest_path, manifest)

        if returncode != 0:
            final_returncode = returncode
            print(f"Zhihu search failed. Session directory was still created: {session_root}", file=sys.stderr)

    if final_returncode == 0:
        print(f"Session completed: {session_root}")
    else:
        print(f"Session finished with errors: {session_root}", file=sys.stderr)

    return final_returncode


if __name__ == "__main__":
    raise SystemExit(main())
