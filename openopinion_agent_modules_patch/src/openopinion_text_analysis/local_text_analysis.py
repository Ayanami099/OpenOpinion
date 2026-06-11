from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openopinion_text_analysis.bettafish_sentiment_analyzer import BettaFishSentimentAnalyzer


LOW_VALUE_PATTERNS = [
    "查看图片",
    "[图片]",
    "图片",
    "cy",
    "mark",
    "蹲",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = clean_text(obj.get("text"))
            title = clean_text(obj.get("title"))

            if not text and not title:
                continue

            obj["_input_line_no"] = line_no
            records.append(obj)

    return records


def is_low_value_text(text: str) -> bool:
    text = clean_text(text)

    if len(text) <= 4:
        return True

    for pattern in LOW_VALUE_PATTERNS:
        if text == pattern:
            return True
        if pattern in text and len(text) <= 12:
            return True

    return False


def engagement_score(record: dict[str, Any]) -> float:
    metrics = record.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    weights = {
        "like_count": 1.0,
        "voteup_count": 1.0,
        "comment_count": 0.5,
        "sub_comment_count": 0.3,
        "posts_count": 0.2,
        "views": 0.01,
    }

    score = 0.0

    for key, weight in weights.items():
        try:
            score += float(metrics.get(key) or 0) * weight
        except (TypeError, ValueError):
            pass

    return round(score, 4)


def analyze_record(
    record: dict[str, Any],
    index: int,
    analyzer: BettaFishSentimentAnalyzer,
) -> dict[str, Any]:
    title = clean_text(record.get("title"))
    text = clean_text(record.get("text"))
    full_text = f"{title}。{text}" if title and title not in text else text

    sentiment_result = analyzer.analyze(full_text).to_dict()

    if not sentiment_result.get("is_success"):
        sentiment_level = "中性"
        sentiment = "neutral"
        sentiment_score = 0.0
        sentiment_confidence = 0.0
        probability_distribution = {}
        error_message = sentiment_result.get("error_message", "")
    else:
        sentiment_level = sentiment_result.get("sentiment_level", "中性")
        sentiment = sentiment_result.get("sentiment", "neutral")
        sentiment_score = sentiment_result.get("sentiment_score", 0.0)
        sentiment_confidence = sentiment_result.get("sentiment_confidence", 0.0)
        probability_distribution = sentiment_result.get("sentiment_probability_distribution", {})
        error_message = ""

    return {
        "index": index,
        "platform": record.get("platform"),
        "source_type": record.get("source_type"),
        "item_id": record.get("item_id"),
        "title": title,
        "text": text,
        "url": record.get("url"),
        "author": record.get("author"),
        "created_at": record.get("created_at"),
        "text_length": len(text),
        "is_low_value": is_low_value_text(text),
        "engagement_score": engagement_score(record),
        "sentiment": sentiment,
        "sentiment_level": sentiment_level,
        "sentiment_score": sentiment_score,
        "sentiment_confidence": sentiment_confidence,
        "probability_distribution": probability_distribution,
        "sentiment_backend": "local_model",
        "sentiment_error": error_message,
        "metrics": record.get("metrics") or {},
        "source_file": record.get("source_file"),
        "line_no": record.get("line_no"),
    }


def slim_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": item.get("index"),
        "platform": item.get("platform"),
        "source_type": item.get("source_type"),
        "title": item.get("title"),
        "text": str(item.get("text") or "")[:300],
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


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    sentiment_counts = Counter(item.get("sentiment") for item in results)
    sentiment_level_counts = Counter(item.get("sentiment_level") for item in results)
    platform_counts = Counter(item.get("platform") for item in results)
    source_type_counts = Counter(item.get("source_type") for item in results)

    by_platform_sentiment: dict[str, Counter[str]] = defaultdict(Counter)
    by_source_type_sentiment: dict[str, Counter[str]] = defaultdict(Counter)

    for item in results:
        platform = str(item.get("platform") or "unknown")
        source_type = str(item.get("source_type") or "unknown")
        sentiment = str(item.get("sentiment") or "unknown")

        by_platform_sentiment[platform][sentiment] += 1
        by_source_type_sentiment[source_type][sentiment] += 1

    valid_scores = [
        float(item.get("sentiment_score") or 0.0)
        for item in results
        if not item.get("is_low_value")
    ]

    average_sentiment_score = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

    negative_items = sorted(
        results,
        key=lambda x: (
            float(x.get("sentiment_score") or 0.0),
            -float(x.get("engagement_score") or 0.0),
        ),
    )[:20]

    hot_items = sorted(
        results,
        key=lambda x: float(x.get("engagement_score") or 0.0),
        reverse=True,
    )[:20]

    confident_items = sorted(
        results,
        key=lambda x: float(x.get("sentiment_confidence") or 0.0),
        reverse=True,
    )[:20]

    return {
        "total_records": len(results),
        "low_value_count": sum(1 for item in results if item.get("is_low_value")),
        "sentiment_counts": dict(sentiment_counts),
        "sentiment_level_counts": dict(sentiment_level_counts),
        "platform_counts": dict(platform_counts),
        "source_type_counts": dict(source_type_counts),
        "by_platform_sentiment": {
            platform: dict(counter)
            for platform, counter in by_platform_sentiment.items()
        },
        "by_source_type_sentiment": {
            source_type: dict(counter)
            for source_type, counter in by_source_type_sentiment.items()
        },
        "average_sentiment_score": average_sentiment_score,
        "top_negative_items": [slim_item(item) for item in negative_items],
        "top_hot_items": [slim_item(item) for item in hot_items],
        "top_confident_items": [slim_item(item) for item in confident_items],
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="openopinion-local-text-analysis",
        description="Local-only text sentiment analysis. No LLM call is made.",
    )
    parser.add_argument("--input", required=True, help="input_texts.jsonl 路径")
    parser.add_argument("--out", default="", help="输出 JSON 路径；不填则自动生成")
    parser.add_argument("--limit", type=int, default=0, help="最多分析多少条；0 表示全部")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="只从本地缓存加载模型")
    parser.add_argument("--allow-download", action="store_true", help="允许 transformers 下载模型")

    args = parser.parse_args()

    input_path = Path(args.input)
    records = load_jsonl(input_path)

    if args.limit > 0:
        records = records[: args.limit]

    analyzer = BettaFishSentimentAnalyzer(
        local_files_only=not args.allow_download,
    )

    results = []

    for index, record in enumerate(records, start=1):
        print(f"本地模型分析第 {index}/{len(records)} 条")
        results.append(analyze_record(record, index, analyzer))

    summary = build_summary(results)

    output = {
        "input_path": str(input_path),
        "analysis_type": "local_model_only",
        "llm_used": False,
        "sentiment_backend": "local_model",
        "model_name": analyzer.model_name,
        "summary": summary,
        "results": results,
    }

    if args.out:
        output_path = Path(args.out)
    else:
        output_path = input_path.with_name("local_text_analysis.json")

    save_json(output_path, output)

    print()
    print("本地模型文本分析完成：")
    print(output_path)
    print()
    print(json.dumps({
        "total_records": summary["total_records"],
        "low_value_count": summary["low_value_count"],
        "sentiment_counts": summary["sentiment_counts"],
        "sentiment_level_counts": summary["sentiment_level_counts"],
        "platform_counts": summary["platform_counts"],
        "source_type_counts": summary["source_type_counts"],
        "average_sentiment_score": summary["average_sentiment_score"],
        "llm_used": False,
        "sentiment_backend": "local_model",
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
