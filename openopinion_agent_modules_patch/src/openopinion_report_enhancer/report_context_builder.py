from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            obj["_input_line_no"] = line_no
            records.append(obj)

    return records


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def slim_item(item: dict[str, Any], max_text_len: int = 500) -> dict[str, Any]:
    text = clean_text(item.get("text"))

    return {
        "index": item.get("index"),
        "platform": item.get("platform"),
        "source_type": item.get("source_type"),
        "title": clean_text(item.get("title")),
        "text": text[:max_text_len],
        "url": item.get("url"),
        "author": item.get("author"),
        "created_at": item.get("created_at"),
        "text_length": item.get("text_length"),
        "is_low_value": item.get("is_low_value"),
        "engagement_score": item.get("engagement_score"),
        "sentiment": item.get("sentiment"),
        "sentiment_level": item.get("sentiment_level"),
        "sentiment_score": item.get("sentiment_score"),
        "sentiment_confidence": item.get("sentiment_confidence"),
    }


def build_representative_samples(
    results: list[dict[str, Any]],
    max_samples: int,
) -> list[dict[str, Any]]:
    valid = [
        item for item in results
        if isinstance(item, dict)
        and not item.get("is_low_value")
        and clean_text(item.get("text"))
    ]

    negative = sorted(
        valid,
        key=lambda x: (
            float(x.get("sentiment_score") or 0.0),
            -float(x.get("engagement_score") or 0.0),
        ),
    )[: max_samples // 3]

    hot = sorted(
        valid,
        key=lambda x: float(x.get("engagement_score") or 0.0),
        reverse=True,
    )[: max_samples // 3]

    by_platform: dict[str, list[dict[str, Any]]] = {}
    for item in valid:
        platform = str(item.get("platform") or "unknown")
        by_platform.setdefault(platform, []).append(item)

    platform_samples: list[dict[str, Any]] = []
    per_platform = max(3, max_samples // max(1, len(by_platform)) // 2)

    for platform_items in by_platform.values():
        platform_items = sorted(
            platform_items,
            key=lambda x: (
                abs(float(x.get("sentiment_score") or 0.0)),
                float(x.get("engagement_score") or 0.0),
            ),
            reverse=True,
        )
        platform_samples.extend(platform_items[:per_platform])

    merged: list[dict[str, Any]] = []
    seen = set()

    for item in negative + hot + platform_samples:
        key = (
            item.get("platform"),
            item.get("source_type"),
            item.get("item_id"),
            item.get("index"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    return [slim_item(item, max_text_len=600) for item in merged[:max_samples]]


def build_compact_corpus(
    results: list[dict[str, Any]],
    max_items: int,
    max_text_len: int,
) -> list[dict[str, Any]]:
    valid = [
        item for item in results
        if isinstance(item, dict)
        and not item.get("is_low_value")
        and clean_text(item.get("text"))
    ]

    ranked = sorted(
        valid,
        key=lambda x: (
            abs(float(x.get("sentiment_score") or 0.0)),
            float(x.get("engagement_score") or 0.0),
            int(x.get("text_length") or 0),
        ),
        reverse=True,
    )

    corpus = []

    for item in ranked[:max_items]:
        corpus.append({
            "index": item.get("index"),
            "platform": item.get("platform"),
            "source_type": item.get("source_type"),
            "sentiment": item.get("sentiment"),
            "sentiment_level": item.get("sentiment_level"),
            "engagement_score": item.get("engagement_score"),
            "text": clean_text(item.get("text"))[:max_text_len],
        })

    return corpus


def build_report_context(
    event_name: str,
    input_texts_path: Path,
    local_analysis_path: Path,
    max_samples: int,
    max_corpus_items: int,
) -> dict[str, Any]:
    input_records = load_jsonl(input_texts_path)
    local_analysis = load_json(local_analysis_path)

    summary = safe_dict(local_analysis.get("summary"))
    results = [
        item for item in safe_list(local_analysis.get("results"))
        if isinstance(item, dict)
    ]

    top_negative_items = [
        slim_item(item, max_text_len=600)
        for item in safe_list(summary.get("top_negative_items"))
        if isinstance(item, dict)
    ]

    top_hot_items = [
        slim_item(item, max_text_len=600)
        for item in safe_list(summary.get("top_hot_items"))
        if isinstance(item, dict)
    ]

    top_confident_items = [
        slim_item(item, max_text_len=500)
        for item in safe_list(summary.get("top_confident_items"))
        if isinstance(item, dict)
    ]

    representative_samples = build_representative_samples(
        results=results,
        max_samples=max_samples,
    )

    compact_corpus_for_llm = build_compact_corpus(
        results=results,
        max_items=max_corpus_items,
        max_text_len=350,
    )

    context = {
        "event_name": event_name,
        "context_type": "report_context",
        "pipeline_design": {
            "single_item_analysis": "local_transformers_sentiment_model",
            "llm_usage": "report_level_synthesis_only",
            "llm_used_in_single_item_analysis": False,
        },
        "input_files": {
            "input_texts": str(input_texts_path),
            "local_text_analysis": str(local_analysis_path),
        },
        "data_overview": {
            "input_text_count": len(input_records),
            "analyzed_text_count": summary.get("total_records"),
            "low_value_count": summary.get("low_value_count"),
            "model_name": local_analysis.get("model_name"),
            "sentiment_backend": local_analysis.get("sentiment_backend"),
            "average_sentiment_score": summary.get("average_sentiment_score"),
        },
        "chart_data": {
            "sentiment_counts": summary.get("sentiment_counts", {}),
            "sentiment_level_counts": summary.get("sentiment_level_counts", {}),
            "platform_counts": summary.get("platform_counts", {}),
            "source_type_counts": summary.get("source_type_counts", {}),
            "by_platform_sentiment": summary.get("by_platform_sentiment", {}),
            "by_source_type_sentiment": summary.get("by_source_type_sentiment", {}),
        },
        "sample_sets": {
            "top_negative_items": top_negative_items,
            "top_hot_items": top_hot_items,
            "top_confident_items": top_confident_items,
            "representative_samples": representative_samples,
        },
        "llm_input": {
            "task": (
                "Read the local sentiment statistics and representative text samples. "
                "Generate report-level insights, topic clusters, high-frequency keyword groups, "
                "platform comparison, and risk interpretation. Do not claim that local sentiment "
                "classification proves factual responsibility."
            ),
            "event_name": event_name,
            "summary": {
                "total_records": summary.get("total_records"),
                "low_value_count": summary.get("low_value_count"),
                "sentiment_counts": summary.get("sentiment_counts", {}),
                "sentiment_level_counts": summary.get("sentiment_level_counts", {}),
                "platform_counts": summary.get("platform_counts", {}),
                "source_type_counts": summary.get("source_type_counts", {}),
                "by_platform_sentiment": summary.get("by_platform_sentiment", {}),
                "average_sentiment_score": summary.get("average_sentiment_score"),
            },
            "representative_samples": representative_samples,
            "compact_corpus_for_keyword_extraction": compact_corpus_for_llm,
        },
    }

    return context


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="openopinion-build-report-context",
        description="Build report_context.json from local-only text analysis results.",
    )
    parser.add_argument("--event", required=True, help="事件名称")
    parser.add_argument("--input-texts", required=True, help="input_texts.jsonl 路径")
    parser.add_argument("--local-analysis", required=True, help="local_text_analysis.json 路径")
    parser.add_argument("--out", required=True, help="输出 report_context.json 路径")
    parser.add_argument("--max-samples", type=int, default=40, help="代表性样本数量")
    parser.add_argument("--max-corpus-items", type=int, default=120, help="给 LLM 读的紧凑语料条数")

    args = parser.parse_args()

    context = build_report_context(
        event_name=args.event,
        input_texts_path=Path(args.input_texts),
        local_analysis_path=Path(args.local_analysis),
        max_samples=args.max_samples,
        max_corpus_items=args.max_corpus_items,
    )

    output_path = Path(args.out)
    save_json(output_path, context)

    print("报告上下文生成完成：")
    print(output_path)
    print()
    print(json.dumps({
        "event_name": context["event_name"],
        "input_text_count": context["data_overview"]["input_text_count"],
        "analyzed_text_count": context["data_overview"]["analyzed_text_count"],
        "representative_samples": len(context["sample_sets"]["representative_samples"]),
        "compact_corpus_for_llm": len(context["llm_input"]["compact_corpus_for_keyword_extraction"]),
        "llm_used": False,
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
