#!/usr/bin/env python3
"""Fetch Shuiyuan topics/posts and save local JSONL snapshots."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_opinion_agent.collectors.shuiyuan import ShuiyuanCollector, ShuiyuanConfig


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_collector(config_path: str) -> ShuiyuanCollector:
    return ShuiyuanCollector(ShuiyuanConfig.from_json(config_path))


def cmd_hot(args: argparse.Namespace) -> None:
    collector = _load_collector(args.config)
    topics = collector.fetch_monitor_candidates(
        hours=args.hours,
        min_views=args.min_views,
        min_likes=args.min_likes,
        min_posts=args.min_posts,
    )
    rows = [{"type": "topic", "captured_at": _now_id(), **asdict(topic)} for topic in topics]
    out_path = Path(args.out or f"data/raw/shuiyuan_hot_{_now_id()}.jsonl")
    _write_jsonl(out_path, rows)
    print(json.dumps({"count": len(rows), "out": str(out_path)}, ensure_ascii=False, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    collector = _load_collector(args.config)
    topics = collector.search_topics(args.query, max_results=args.max_results)
    rows = [{"type": "topic_search_result", "captured_at": _now_id(), "query": args.query, **asdict(topic)} for topic in topics]
    out_path = Path(args.out or f"data/raw/shuiyuan_search_{_now_id()}.jsonl")
    _write_jsonl(out_path, rows)
    print(json.dumps({"query": args.query, "count": len(rows), "out": str(out_path)}, ensure_ascii=False, indent=2))


def cmd_topic(args: argparse.Namespace) -> None:
    collector = _load_collector(args.config)
    max_posts = args.max_posts
    if args.all_posts:
        preview_topic, _ = collector.read_topic(args.topic, max_posts=1)
        max_posts = max(preview_topic.posts_count, 1)
        print(f"Detected {max_posts} posts in topic {preview_topic.topic_id}. Fetching all...", flush=True)

    last_printed = 0

    def progress(done: int, total: int) -> None:
        nonlocal last_printed
        if done == total or done - last_printed >= 200:
            print(f"Fetched {done}/{total} posts...", flush=True)
            last_printed = done

    topic, posts = collector.read_topic(
        args.topic,
        max_posts=max_posts,
        batch_size=args.batch_size,
        progress_callback=progress,
    )
    captured_at = _now_id()
    rows = [
        {"type": "topic", "captured_at": captured_at, **asdict(topic)},
        *[
            {"type": "post", "captured_at": captured_at, **asdict(post)}
            for post in posts
        ],
    ]
    out_path = Path(args.out or f"data/raw/shuiyuan_topic_{topic.topic_id}_{captured_at}.jsonl")
    _write_jsonl(out_path, rows)
    print(json.dumps({
        "topic_id": topic.topic_id,
        "title": topic.title,
        "post_count": len(posts),
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Shuiyuan topics and posts")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hot = subparsers.add_parser("hot", help="Fetch hot/latest monitor candidate topics")
    hot.add_argument("--hours", type=int, default=24)
    hot.add_argument("--min-views", type=int, default=50)
    hot.add_argument("--min-likes", type=int, default=3)
    hot.add_argument("--min-posts", type=int, default=5)
    hot.add_argument("--out", default="")
    hot.set_defaults(func=cmd_hot)

    search = subparsers.add_parser("search", help="Search Shuiyuan topics")
    search.add_argument("query")
    search.add_argument("--max-results", type=int, default=10)
    search.add_argument("--out", default="")
    search.set_defaults(func=cmd_search)

    topic = subparsers.add_parser("topic", help="Read one topic and its posts")
    topic.add_argument("topic", help="Topic id or Shuiyuan topic URL")
    topic.add_argument("--max-posts", type=int, default=100)
    topic.add_argument("--all-posts", action="store_true", help="Fetch all posts in the topic.")
    topic.add_argument("--batch-size", type=int, default=20, help="Discourse posts.json batch size. Default: 20.")
    topic.add_argument("--out", default="")
    topic.set_defaults(func=cmd_topic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
