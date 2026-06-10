#!/usr/bin/env python3
"""Run bundled MediaCrawler and write outputs into this project.

This wrapper keeps MediaCrawler as an external tool under external/MediaCrawler,
while normalizing its output location to data/external/mediacrawler.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "external" / "MediaCrawler"
DEFAULT_OUTPUT = ROOT / "data" / "external" / "mediacrawler"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run external MediaCrawler with campus-opinion-agent defaults",
        add_help=True,
    )
    parser.add_argument("--conda-env", default="mediacrawler", help="Conda env name. Default: mediacrawler")
    parser.add_argument("--no-conda", action="store_true", help="Use current Python instead of conda run.")
    parser.add_argument("--media-crawler-repo", default=str(DEFAULT_REPO), help="Path to MediaCrawler repo.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Normalized MediaCrawler output directory.")
    args, passthrough = parser.parse_known_args()

    repo = Path(args.media_crawler_repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        raise SystemExit(f"MediaCrawler repo not found: {repo}")

    command = ["python", "main.py", *passthrough]
    if "--save_data_path" not in passthrough:
        command.extend(["--save_data_path", str(output_dir)])
    if "--save_data_option" not in passthrough:
        command.extend(["--save_data_option", "jsonl"])

    if not args.no_conda:
        command = ["conda", "run", "-n", args.conda_env, *command]

    print("Running:", " ".join(command), flush=True)
    print("MediaCrawler repo:", repo, flush=True)
    print("Output dir:", output_dir, flush=True)
    return subprocess.call(command, cwd=repo)


if __name__ == "__main__":
    raise SystemExit(main())

