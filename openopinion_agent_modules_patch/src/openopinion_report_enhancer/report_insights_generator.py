from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from opinion_agent.config import load_dotenv
from opinion_agent.llm import LLMError, OpenAICompatibleClient


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def compact_text(text: Any, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "……"


def build_llm_client() -> OpenAICompatibleClient:
    load_dotenv()

    api_key = (
        os.getenv("REPORT_LLM_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    base_url = (
        os.getenv("REPORT_LLM_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )

    model = (
        os.getenv("REPORT_LLM_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
    )

    timeout = (
        os.getenv("REPORT_LLM_TIMEOUT")
        or os.getenv("LLM_TIMEOUT")
        or os.getenv("OPENAI_TIMEOUT")
        or "300"
    )

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=int(timeout),
    )

    if not client.enabled:
        raise RuntimeError(
            "没有找到 LLM API Key。请检查 .env 中的 LLM_API_KEY，"
            "或设置 REPORT_LLM_API_KEY / OPENAI_API_KEY。"
        )

    return client


def build_llm_material(
    context: dict[str, Any],
    max_corpus_items: int,
    max_sample_items: int,
) -> dict[str, Any]:
    llm_input = context.get("llm_input") or {}
    sample_sets = context.get("sample_sets") or {}

    corpus = llm_input.get("compact_corpus_for_keyword_extraction") or []
    samples = llm_input.get("representative_samples") or sample_sets.get("representative_samples") or []

    compact_corpus = []
    for item in corpus[:max_corpus_items]:
        if not isinstance(item, dict):
            continue
        compact_corpus.append({
            "index": item.get("index"),
            "platform": item.get("platform"),
            "source_type": item.get("source_type"),
            "sentiment": item.get("sentiment"),
            "sentiment_level": item.get("sentiment_level"),
            "engagement_score": item.get("engagement_score"),
            "text": compact_text(item.get("text"), 320),
        })

    compact_samples = []
    for item in samples[:max_sample_items]:
        if not isinstance(item, dict):
            continue
        compact_samples.append({
            "index": item.get("index"),
            "platform": item.get("platform"),
            "source_type": item.get("source_type"),
            "title": compact_text(item.get("title"), 120),
            "text": compact_text(item.get("text"), 450),
            "sentiment": item.get("sentiment"),
            "sentiment_level": item.get("sentiment_level"),
            "engagement_score": item.get("engagement_score"),
        })

    return {
        "event_name": context.get("event_name"),
        "pipeline_design": context.get("pipeline_design"),
        "data_overview": context.get("data_overview"),
        "chart_data": context.get("chart_data"),
        "representative_samples": compact_samples,
        "compact_corpus": compact_corpus,
    }


def system_prompt() -> str:
    return """
你是一个舆情报告生成 Agent，只负责报告阶段的综合归纳。

重要边界：
1. 单条文本情感已经由本地 Transformers 模型完成，你不要重新逐条打情感标签。
2. 你可以读取样本文本，归纳高频关键词、主题簇、争议焦点和风险信号。
3. 图表用的关键词组和主题簇可以由你归纳，但必须基于输入样本，不要编造材料外事实。
4. 不要把样本情绪当作事实定责。只能说“样本中呈现”“讨论集中于”“评论语气偏向”。
5. 不要生成 Markdown，只输出合法 JSON。
""".strip()


def user_prompt(material: dict[str, Any]) -> str:
    schema = {
        "event_name": "事件名称",
        "executive_summary": [
            "摘要段落1",
            "摘要段落2"
        ],
        "core_findings": [
            {
                "title": "核心发现标题",
                "description": "完整解释",
                "supporting_data": "引用输入中的统计或样本依据"
            }
        ],
        "keyword_groups": [
            {
                "name": "关键词组名称",
                "keywords": ["关键词1", "关键词2"],
                "weight": 85,
                "description": "这个关键词组代表的讨论含义",
                "supporting_sample_indices": [1, 2, 3]
            }
        ],
        "topic_clusters": [
            {
                "name": "主题簇名称",
                "summary": "主题解释",
                "dominant_sentiment": "negative/neutral/positive/mixed",
                "platforms": ["zhihu", "shuiyuan"],
                "weight": 80,
                "supporting_sample_indices": [1, 2, 3]
            }
        ],
        "platform_comparison": {
            "summary": "不同平台讨论差异",
            "items": [
                {
                    "platform": "平台名",
                    "observation": "该平台讨论特点",
                    "sentiment_pattern": "情绪特点"
                }
            ]
        },
        "risk_interpretation": {
            "risk_level": "low/medium/high",
            "risk_summary": "风险概括",
            "risk_signals": ["风险信号1", "风险信号2"],
            "uncertainties": ["待核验点1", "待核验点2"]
        },
        "chart_data": {
            "keyword_group_chart": [
                {"name": "关键词组名称", "value": 85}
            ],
            "topic_cluster_chart": [
                {"name": "主题簇名称", "value": 80}
            ]
        },
        "report_sections": {
            "event_overview": "事件概览段落",
            "public_opinion_focus": "舆论焦点段落",
            "emotion_structure": "情绪结构段落",
            "platform_spread": "平台传播段落",
            "risk_and_observation": "风险与观察段落"
        },
        "method_note": "方法说明"
    }

    return f"""
请根据输入材料生成报告洞察 JSON。

输出必须严格接近以下结构，字段名必须保留：
{json.dumps(schema, ensure_ascii=False, indent=2)}

要求：
- keyword_groups 建议 6 到 10 个。
- topic_clusters 建议 4 到 8 个。
- weight 是 0 到 100 的相对权重，用于图表展示，不是精确词频。
- supporting_sample_indices 必须尽量引用输入中存在的 index。
- executive_summary 和 report_sections 要能直接放进 HTML 报告。
- method_note 要说明：单条情感由本地模型完成，LLM 只用于报告层归纳和关键词/主题总结。

输入材料：
{json.dumps(material, ensure_ascii=False, indent=2)}
""".strip()


def normalize_insights(data: dict[str, Any], event_name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}

    data.setdefault("event_name", event_name)
    data.setdefault("executive_summary", [])
    data.setdefault("core_findings", [])
    data.setdefault("keyword_groups", [])
    data.setdefault("topic_clusters", [])
    data.setdefault("platform_comparison", {"summary": "", "items": []})
    data.setdefault("risk_interpretation", {
        "risk_level": "medium",
        "risk_summary": "",
        "risk_signals": [],
        "uncertainties": [],
    })
    data.setdefault("chart_data", {})
    data.setdefault("report_sections", {})
    data.setdefault(
        "method_note",
        "单条文本情感由本地模型完成，LLM 仅用于报告阶段的综合归纳。",
    )

    keyword_chart = []
    for item in data.get("keyword_groups") or []:
        if not isinstance(item, dict):
            continue
        keyword_chart.append({
            "name": clean_text(item.get("name")),
            "value": int(item.get("weight") or 0),
        })

    topic_chart = []
    for item in data.get("topic_clusters") or []:
        if not isinstance(item, dict):
            continue
        topic_chart.append({
            "name": clean_text(item.get("name")),
            "value": int(item.get("weight") or 0),
        })

    data["chart_data"] = {
        **(data.get("chart_data") or {}),
        "keyword_group_chart": keyword_chart,
        "topic_cluster_chart": topic_chart,
    }

    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="openopinion-generate-report-insights",
        description="Use LLM only at report stage to generate insights from report_context.json.",
    )
    parser.add_argument("--context", required=True, help="report_context.json 路径")
    parser.add_argument("--out", required=True, help="输出 report_insights.json 路径")
    parser.add_argument("--max-corpus-items", type=int, default=100, help="给 LLM 读取的紧凑语料条数")
    parser.add_argument("--max-sample-items", type=int, default=40, help="给 LLM 读取的代表样本条数")

    args = parser.parse_args()

    context_path = Path(args.context)
    context = load_json(context_path)

    material = build_llm_material(
        context=context,
        max_corpus_items=args.max_corpus_items,
        max_sample_items=args.max_sample_items,
    )

    client = build_llm_client()

    print("正在调用 LLM 生成报告洞察。注意：这一步只用于报告层归纳，不做单条文本分析。")

    try:
        insights = client.chat_json(
            system=system_prompt(),
            user=user_prompt(material),
        )
    except LLMError as exc:
        raise RuntimeError(f"报告洞察生成失败：{exc}") from exc

    event_name = clean_text(context.get("event_name")) or "未命名事件"
    insights = normalize_insights(insights, event_name=event_name)

    output = {
        "event_name": event_name,
        "insights_type": "report_level_llm_synthesis",
        "llm_used": True,
        "single_item_analysis_llm_used": False,
        "source_context": str(context_path),
        "insights": insights,
    }

    output_path = Path(args.out)
    save_json(output_path, output)

    print("报告洞察生成完成：")
    print(output_path)
    print()
    print(json.dumps({
        "event_name": event_name,
        "keyword_groups": len(insights.get("keyword_groups") or []),
        "topic_clusters": len(insights.get("topic_clusters") or []),
        "llm_used": True,
        "single_item_analysis_llm_used": False,
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
