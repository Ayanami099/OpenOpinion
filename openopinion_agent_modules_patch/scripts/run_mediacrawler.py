from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run external MediaCrawler with campus-opinion-agent defaults",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--conda-env",
        default="mediacrawler",
        help="Conda env name. Default: mediacrawler",
    )
    parser.add_argument(
        "--no-conda",
        action="store_true",
        help="Use current Python or --python-exe instead of conda run.",
    )
    parser.add_argument(
        "--python-exe",
        default=None,
        help="Explicit Python executable for running MediaCrawler.",
    )
    parser.add_argument(
        "--media-crawler-repo",
        default="external/MediaCrawler",
        help="Path to MediaCrawler repo.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/external/mediacrawler",
        help="Normalized MediaCrawler output directory.",
    )

    args, extra_args = parser.parse_known_args()

    project_root = Path(__file__).resolve().parents[1]
    repo = (project_root / args.media_crawler_repo).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not repo.exists():
        raise FileNotFoundError(f"MediaCrawler repo not found: {repo}")

    main_py = repo / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"MediaCrawler main.py not found: {main_py}")

    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_extra_args = list(extra_args)

    if "--save_data_path" not in normalized_extra_args:
        normalized_extra_args.extend(["--save_data_path", str(output_dir)])

    if "--save_data_option" not in normalized_extra_args:
        normalized_extra_args.extend(["--save_data_option", "jsonl"])

    if args.no_conda:
        python_exe = args.python_exe or sys.executable
        command = [
            str(Path(python_exe).resolve()),
            "main.py",
            *normalized_extra_args,
        ]
    else:
        command = [
            "conda",
            "run",
            "-n",
            args.conda_env,
            "python",
            "main.py",
            *normalized_extra_args,
        ]

    print("Running:", " ".join(command))
    print("MediaCrawler repo:", repo)
    print("Output dir:", output_dir)

    return subprocess.call(command, cwd=repo)


if __name__ == "__main__":
    raise SystemExit(main())