from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def unix_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        ts = int(value)
    except (TypeError, ValueError):
        return None

    if ts <= 0:
        return None

    if ts > 10_000_000_000:
        ts = ts // 1000

    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return None


def make_record(
    *,
    platform: str,
    source_type: str,
    source_file: Path,
    line_no: int,
    item_id: str | None,
    title: str,
    text: str,
    url: str | None,
    created_at: str | None,
    author: str | None,
    metrics: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any] | None:
    title = clean_text(title)
    text = clean_text(text)

    if not text and title:
        text = title

    if not text:
        return None

    return {
        "platform": platform,
        "source_type": source_type,
        "source_file": str(source_file),
        "line_no": line_no,
        "item_id": item_id,
        "title": title,
        "text": text,
        "url": url,
        "created_at": created_at,
        "author": author,
        "metrics": metrics,
        "raw": raw,
    }


def normalize_tavily_session(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return records

    evidence_pack = data.get("evidence_pack") or []
    if not isinstance(evidence_pack, list):
        return records

    for idx, ev in enumerate(evidence_pack, start=1):
        if not isinstance(ev, dict):
            continue

        title = clean_text(ev.get("title"))
        content = clean_text(ev.get("content"))
        summary = clean_text(ev.get("summary"))

        text_parts = []
        if title:
            text_parts.append(title)
        if summary and summary != title:
            text_parts.append(summary)
        if content and content not in text_parts:
            text_parts.append(content)

        record = make_record(
            platform="tavily",
            source_type="evidence",
            source_file=path,
            line_no=idx,
            item_id=ev.get("evidence_id"),
            title=title,
            text="。".join(text_parts),
            url=ev.get("url"),
            created_at=ev.get("published_at") or ev.get("retrieved_at"),
            author=ev.get("source"),
            metrics={
                "session_id": data.get("session_id"),
                "query": ev.get("query"),
                "intent": ev.get("intent"),
                "original_platform": ev.get("platform"),
                "source": ev.get("source"),
                "source_type": ev.get("source_type"),
                "scores": ev.get("scores"),
                "filter_status": ev.get("filter_status"),
            },
            raw=ev,
        )

        if record:
            records.append(record)

    return records


def normalize_shuiyuan(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for line_no, obj in iter_jsonl(path):
        record_type = obj.get("type")

        if record_type == "post":
            record = make_record(
                platform="shuiyuan",
                source_type="post",
                source_file=path,
                line_no=line_no,
                item_id=str(obj.get("post_id")) if obj.get("post_id") is not None else None,
                title="",
                text=obj.get("content") or "",
                url=None,
                created_at=obj.get("created_at") or obj.get("captured_at"),
                author=obj.get("username"),
                metrics={
                    "topic_id": obj.get("topic_id"),
                    "post_number": obj.get("post_number"),
                    "like_count": obj.get("like_count"),
                },
                raw=obj,
            )
        else:
            title = clean_text(obj.get("title"))
            excerpt = clean_text(obj.get("excerpt"))
            text = f"{title}。{excerpt}" if title and excerpt else title or excerpt

            record = make_record(
                platform="shuiyuan",
                source_type=record_type or "topic",
                source_file=path,
                line_no=line_no,
                item_id=str(obj.get("topic_id")) if obj.get("topic_id") is not None else None,
                title=title,
                text=text,
                url=obj.get("url"),
                created_at=obj.get("created_at") or obj.get("captured_at"),
                author=obj.get("last_poster_username"),
                metrics={
                    "views": obj.get("views"),
                    "like_count": obj.get("like_count"),
                    "posts_count": obj.get("posts_count"),
                },
                raw=obj,
            )

        if record:
            records.append(record)

    return records


def normalize_zhihu_contents(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for line_no, obj in iter_jsonl(path):
        title = clean_text(obj.get("title"))
        desc = clean_text(obj.get("desc"))
        content_text = clean_text(obj.get("content_text"))

        parts = []
        if title:
            parts.append(title)
        if desc and desc != title:
            parts.append(desc)
        if content_text:
            parts.append(content_text)

        record = make_record(
            platform="zhihu",
            source_type=obj.get("content_type") or "content",
            source_file=path,
            line_no=line_no,
            item_id=str(obj.get("content_id")) if obj.get("content_id") is not None else None,
            title=title,
            text="。".join(parts),
            url=obj.get("content_url"),
            created_at=unix_to_iso(obj.get("created_time")),
            author=obj.get("user_nickname"),
            metrics={
                "voteup_count": obj.get("voteup_count"),
                "comment_count": obj.get("comment_count"),
            },
            raw=obj,
        )

        if record:
            records.append(record)

    return records


def normalize_zhihu_comments(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for line_no, obj in iter_jsonl(path):
        record = make_record(
            platform="zhihu",
            source_type="comment",
            source_file=path,
            line_no=line_no,
            item_id=str(obj.get("comment_id")) if obj.get("comment_id") is not None else None,
            title="",
            text=obj.get("content") or "",
            url=None,
            created_at=unix_to_iso(obj.get("publish_time")),
            author=obj.get("user_nickname"),
            metrics={
                "like_count": obj.get("like_count"),
                "dislike_count": obj.get("dislike_count"),
                "sub_comment_count": obj.get("sub_comment_count"),
                "content_id": obj.get("content_id"),
                "parent_comment_id": obj.get("parent_comment_id"),
            },
            raw=obj,
        )

        if record:
            records.append(record)

    return records


def build_text_input(session_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    text_analysis_dir = session_dir / "text_analysis"
    text_analysis_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    tavily_dir = session_dir / "search_results" / "tavily"
    for path in sorted(tavily_dir.glob("case_*.json")):
        records.extend(normalize_tavily_session(path))

    multi_dir = session_dir / "search_results" / "multi_platform"

    for path in sorted((multi_dir / "shuiyuan").glob("*.jsonl")):
        records.extend(normalize_shuiyuan(path))

    zhihu_jsonl_dir = multi_dir / "zhihu" / "zhihu" / "jsonl"

    for path in sorted(zhihu_jsonl_dir.glob("search_contents_*.jsonl")):
        records.extend(normalize_zhihu_contents(path))

    for path in sorted(zhihu_jsonl_dir.glob("search_comments_*.jsonl")):
        records.extend(normalize_zhihu_comments(path))

    output_path = text_analysis_dir / "input_texts.jsonl"
    summary_path = text_analysis_dir / "input_texts_summary.json"

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "session_dir": str(session_dir),
        "output_path": str(output_path),
        "total_records": len(records),
        "by_platform": {},
        "by_source_type": {},
    }

    for record in records:
        platform = record["platform"]
        source_type = record["source_type"]
        summary["by_platform"][platform] = summary["by_platform"].get(platform, 0) + 1
        summary["by_source_type"][source_type] = summary["by_source_type"].get(source_type, 0) + 1

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path, summary_path, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="openopinion-build-text-input",
        description="Normalize Tavily, Shuiyuan and Zhihu results into text_analysis/input_texts.jsonl.",
    )
    parser.add_argument("--session-dir", required=True)

    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    output_path, summary_path, summary = build_text_input(session_dir)

    print(json.dumps({
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "total_records": summary["total_records"],
        "by_platform": summary["by_platform"],
        "by_source_type": summary["by_source_type"],
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
