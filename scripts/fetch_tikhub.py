#!/usr/bin/env python3
"""Fetch external platform data through TikHub and save JSONL snapshots."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_opinion_agent.collectors.tikhub_external import (  # noqa: E402
    TikHubExternalConfig,
    TikhubExternalCollector,
)


def _date_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _collector(config_path: str, output_dir: str | None = None) -> tuple[TikhubExternalCollector, Path]:
    config = TikHubExternalConfig.from_json(config_path)
    if output_dir:
        config.output_dir = output_dir
    return TikhubExternalCollector(config), Path(config.output_dir)


def cmd_zhihu_search(args: argparse.Namespace) -> None:
    collector, output_root = _collector(args.config, args.output_dir)
    raw_payloads: list[dict] = []
    contents = collector.search_zhihu(
        args.query,
        max_results=args.max_results,
        poll_attempts=args.poll_attempts,
        poll_interval=args.poll_interval,
        raw_payloads=raw_payloads if args.raw_out else None,
    )

    out_dir = output_root / "zhihu" / "jsonl"
    contents_path = Path(args.contents_out or out_dir / f"search_contents_{_date_id()}.jsonl")
    content_rows = [asdict(item) for item in contents]
    _write_jsonl(contents_path, content_rows)

    comments_path = ""
    comments_count = 0
    if args.comments:
        comments = collector.fetch_zhihu_comments_for_contents(
            contents,
            max_comments_per_content=args.max_comments_per_content,
            include_replies=args.include_replies,
        )
        comments_path_obj = Path(args.comments_out or out_dir / f"search_comments_{_date_id()}.jsonl")
        _write_jsonl(comments_path_obj, [asdict(item) for item in comments])
        comments_path = str(comments_path_obj)
        comments_count = len(comments)

    raw_path = ""
    if args.raw_out:
        raw_path_obj = Path(args.raw_out)
        raw_path_obj.parent.mkdir(parents=True, exist_ok=True)
        raw_path_obj.write_text(json.dumps(raw_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_path = str(raw_path_obj)

    print(json.dumps({
        "source": "tikhub",
        "platform": "zhihu",
        "query": args.query,
        "contents_count": len(content_rows),
        "contents_out": str(contents_path),
        "comments_count": comments_count,
        "comments_out": comments_path,
        "raw_out": raw_path,
    }, ensure_ascii=False, indent=2))


def cmd_zhihu_comments(args: argparse.Namespace) -> None:
    collector, output_root = _collector(args.config, args.output_dir)
    comments = collector.fetch_zhihu_comments(
        args.answer_id,
        max_comments=args.max_comments,
        include_replies=args.include_replies,
        order_by=args.order_by or None,
    )
    out_dir = output_root / "zhihu" / "jsonl"
    out_path = Path(args.out or out_dir / f"answer_{args.answer_id}_comments_{_date_id()}.jsonl")
    _write_jsonl(out_path, [asdict(item) for item in comments])
    print(json.dumps({
        "source": "tikhub",
        "platform": "zhihu",
        "answer_id": args.answer_id,
        "comments_count": len(comments),
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))


def cmd_xiaohongshu_search(args: argparse.Namespace) -> None:
    collector, output_root = _collector(args.config, args.output_dir)
    raw_payloads: list[dict] = []
    notes = collector.search_xiaohongshu(
        args.query,
        max_results=args.max_results,
        page=args.page,
        raw_payloads=raw_payloads if args.raw_out else None,
    )

    out_dir = output_root / "xiaohongshu" / "jsonl"
    notes_path = Path(args.notes_out or out_dir / f"search_notes_{_date_id()}.jsonl")
    note_rows = [asdict(item) for item in notes]
    _write_jsonl(notes_path, note_rows)

    comments_path = ""
    comments_count = 0
    if args.comments:
        comments = collector.fetch_xiaohongshu_comments_for_notes(
            notes,
            max_comments_per_note=args.max_comments_per_note,
            include_replies=args.include_replies,
            pages_per_note=args.pages_per_note,
        )
        comments_path_obj = Path(args.comments_out or out_dir / f"search_comments_{_date_id()}.jsonl")
        _write_jsonl(comments_path_obj, [asdict(item) for item in comments])
        comments_path = str(comments_path_obj)
        comments_count = len(comments)

    raw_path = ""
    if args.raw_out:
        raw_path_obj = Path(args.raw_out)
        raw_path_obj.parent.mkdir(parents=True, exist_ok=True)
        raw_path_obj.write_text(json.dumps(raw_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_path = str(raw_path_obj)

    print(json.dumps({
        "source": "tikhub",
        "platform": "xiaohongshu",
        "query": args.query,
        "notes_count": len(note_rows),
        "notes_out": str(notes_path),
        "comments_count": comments_count,
        "comments_out": comments_path,
        "raw_out": raw_path,
    }, ensure_ascii=False, indent=2))


def cmd_xiaohongshu_comments(args: argparse.Namespace) -> None:
    collector, output_root = _collector(args.config, args.output_dir)
    raw_payloads: list[dict] = []
    comments = collector.fetch_xiaohongshu_comments(
        args.note_id,
        max_comments=args.max_comments,
        include_replies=args.include_replies,
        cursor=args.cursor or None,
        pages=args.pages,
        raw_payloads=raw_payloads if args.raw_out else None,
    )
    out_dir = output_root / "xiaohongshu" / "jsonl"
    out_path = Path(args.out or out_dir / f"note_{args.note_id}_comments_{_date_id()}.jsonl")
    _write_jsonl(out_path, [asdict(item) for item in comments])

    raw_path = ""
    if args.raw_out:
        raw_path_obj = Path(args.raw_out)
        raw_path_obj.parent.mkdir(parents=True, exist_ok=True)
        raw_path_obj.write_text(json.dumps(raw_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_path = str(raw_path_obj)

    print(json.dumps({
        "source": "tikhub",
        "platform": "xiaohongshu",
        "note_id": args.note_id,
        "comments_count": len(comments),
        "out": str(out_path),
        "raw_out": raw_path,
    }, ensure_ascii=False, indent=2))


def cmd_weibo_search(args: argparse.Namespace) -> None:
    collector, output_root = _collector(args.config, args.output_dir)
    raw_payloads: list[dict] = []
    posts = collector.search_weibo(
        args.query,
        max_results=args.max_results,
        page=args.page,
        raw_payloads=raw_payloads if args.raw_out else None,
    )

    out_dir = output_root / "weibo" / "jsonl"
    posts_path = Path(args.posts_out or out_dir / f"search_posts_{_date_id()}.jsonl")
    post_rows = [asdict(item) for item in posts]
    _write_jsonl(posts_path, post_rows)

    comments_path = ""
    comments_count = 0
    if args.comments:
        comments = collector.fetch_weibo_comments_for_posts(
            posts,
            max_comments_per_post=args.max_comments_per_post,
            include_replies=args.include_replies,
        )
        comments_path_obj = Path(args.comments_out or out_dir / f"search_comments_{_date_id()}.jsonl")
        _write_jsonl(comments_path_obj, [asdict(item) for item in comments])
        comments_path = str(comments_path_obj)
        comments_count = len(comments)

    raw_path = ""
    if args.raw_out:
        raw_path_obj = Path(args.raw_out)
        raw_path_obj.parent.mkdir(parents=True, exist_ok=True)
        raw_path_obj.write_text(json.dumps(raw_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_path = str(raw_path_obj)

    print(json.dumps({
        "source": "tikhub",
        "platform": "weibo",
        "query": args.query,
        "posts_count": len(post_rows),
        "posts_out": str(posts_path),
        "comments_count": comments_count,
        "comments_out": comments_path,
        "raw_out": raw_path,
    }, ensure_ascii=False, indent=2))


def cmd_weibo_comments(args: argparse.Namespace) -> None:
    collector, output_root = _collector(args.config, args.output_dir)
    raw_payloads: list[dict] = []
    comments = collector.fetch_weibo_comments(
        args.post_id,
        max_comments=args.max_comments,
        include_replies=args.include_replies,
        raw_payloads=raw_payloads if args.raw_out else None,
    )
    out_dir = output_root / "weibo" / "jsonl"
    out_path = Path(args.out or out_dir / f"post_{args.post_id}_comments_{_date_id()}.jsonl")
    _write_jsonl(out_path, [asdict(item) for item in comments])

    raw_path = ""
    if args.raw_out:
        raw_path_obj = Path(args.raw_out)
        raw_path_obj.parent.mkdir(parents=True, exist_ok=True)
        raw_path_obj.write_text(json.dumps(raw_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_path = str(raw_path_obj)

    print(json.dumps({
        "source": "tikhub",
        "platform": "weibo",
        "post_id": args.post_id,
        "comments_count": len(comments),
        "out": str(out_path),
        "raw_out": raw_path,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch external data through TikHub")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--output-dir", default="", help="Override TikHub output dir.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    zhihu_search = subparsers.add_parser("zhihu-search", help="Search Zhihu through TikHub")
    zhihu_search.add_argument("query")
    zhihu_search.add_argument("--max-results", type=int, default=10)
    zhihu_search.add_argument("--poll-attempts", type=int, default=5)
    zhihu_search.add_argument("--poll-interval", type=float, default=2.0)
    zhihu_search.add_argument("--comments", action="store_true", help="Fetch comments for answer rows.")
    zhihu_search.add_argument("--max-comments-per-content", type=int, default=50)
    zhihu_search.add_argument("--include-replies", action="store_true")
    zhihu_search.add_argument("--contents-out", default="")
    zhihu_search.add_argument("--comments-out", default="")
    zhihu_search.add_argument("--raw-out", default="", help="Write raw TikHub responses to this JSON file.")
    zhihu_search.set_defaults(func=cmd_zhihu_search)

    zhihu_comments = subparsers.add_parser("zhihu-comments", help="Fetch comments for one Zhihu answer")
    zhihu_comments.add_argument("answer_id")
    zhihu_comments.add_argument("--max-comments", type=int, default=100)
    zhihu_comments.add_argument("--include-replies", action="store_true")
    zhihu_comments.add_argument("--order-by", default="")
    zhihu_comments.add_argument("--out", default="")
    zhihu_comments.set_defaults(func=cmd_zhihu_comments)

    xhs_search = subparsers.add_parser("xiaohongshu-search", help="Search Xiaohongshu notes through TikHub")
    xhs_search.add_argument("query")
    xhs_search.add_argument("--max-results", type=int, default=20)
    xhs_search.add_argument("--page", type=int, default=1)
    xhs_search.add_argument("--comments", action="store_true", help="Fetch comments for note rows.")
    xhs_search.add_argument("--max-comments-per-note", type=int, default=50)
    xhs_search.add_argument("--pages-per-note", type=int, default=1)
    xhs_search.add_argument("--include-replies", action="store_true")
    xhs_search.add_argument("--notes-out", default="")
    xhs_search.add_argument("--comments-out", default="")
    xhs_search.add_argument("--raw-out", default="", help="Write raw TikHub responses to this JSON file.")
    xhs_search.set_defaults(func=cmd_xiaohongshu_search)

    xhs_comments = subparsers.add_parser("xiaohongshu-comments", help="Fetch comments for one Xiaohongshu note")
    xhs_comments.add_argument("note_id")
    xhs_comments.add_argument("--max-comments", type=int, default=100)
    xhs_comments.add_argument("--pages", type=int, default=1)
    xhs_comments.add_argument("--cursor", default="", help="Cursor from the previous raw response for manual paging.")
    xhs_comments.add_argument("--include-replies", action="store_true")
    xhs_comments.add_argument("--out", default="")
    xhs_comments.add_argument("--raw-out", default="", help="Write raw TikHub responses to this JSON file.")
    xhs_comments.set_defaults(func=cmd_xiaohongshu_comments)

    weibo_search = subparsers.add_parser("weibo-search", help="Search Weibo posts through TikHub")
    weibo_search.add_argument("query")
    weibo_search.add_argument("--max-results", type=int, default=20)
    weibo_search.add_argument("--page", type=int, default=1)
    weibo_search.add_argument("--comments", action="store_true", help="Fetch comments for post rows.")
    weibo_search.add_argument("--max-comments-per-post", type=int, default=50)
    weibo_search.add_argument("--include-replies", action="store_true")
    weibo_search.add_argument("--posts-out", default="")
    weibo_search.add_argument("--comments-out", default="")
    weibo_search.add_argument("--raw-out", default="", help="Write raw TikHub responses to this JSON file.")
    weibo_search.set_defaults(func=cmd_weibo_search)

    weibo_comments = subparsers.add_parser("weibo-comments", help="Fetch comments for one Weibo post")
    weibo_comments.add_argument("post_id")
    weibo_comments.add_argument("--max-comments", type=int, default=100)
    weibo_comments.add_argument("--include-replies", action="store_true")
    weibo_comments.add_argument("--out", default="")
    weibo_comments.add_argument("--raw-out", default="", help="Write raw TikHub responses to this JSON file.")
    weibo_comments.set_defaults(func=cmd_weibo_comments)

    args = parser.parse_args()
    args.output_dir = args.output_dir or None
    args.func(args)


if __name__ == "__main__":
    main()
