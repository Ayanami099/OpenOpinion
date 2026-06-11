from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

SENTIMENT_LEVELS = [
    "非常负面",
    "负面",
    "中性",
    "正面",
    "非常正面",
]

INDEX_TO_ZH_LABEL = {
    0: "非常负面",
    1: "负面",
    2: "中性",
    3: "正面",
    4: "非常正面",
}

EN_TO_ZH_LABEL = {
    "very negative": "非常负面",
    "negative": "负面",
    "neutral": "中性",
    "positive": "正面",
    "very positive": "非常正面",
    "label_0": "非常负面",
    "label_1": "负面",
    "label_2": "中性",
    "label_3": "正面",
    "label_4": "非常正面",
}


@dataclass
class BettaFishSentimentResult:
    text: str
    sentiment_level: str = "中性"
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0
    sentiment_probability_distribution: dict[str, float] = field(default_factory=dict)
    is_success: bool = True
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BettaFishBatchSentimentResult:
    total_count: int
    success_count: int
    failed_count: int
    average_confidence: float
    sentiment_level_distribution: dict[str, int]
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BettaFishSentimentAnalyzer:
    """
    BettaFish-style local sentiment analyzer.

    It uses a local Hugging Face sequence classification model to output:
    - 5-level Chinese sentiment label
    - coarse sentiment label: positive / neutral / negative
    - confidence
    - probability distribution

    This module only handles sentiment classification.
    It does not handle stance, risk labels, topics, evidence quotes, or aspect opinions.
    Those should still be handled by LLMTextAnalyzer.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = 512,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.local_files_only = local_files_only

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"正在加载情感分类模型：{self.model_name}")
        print(f"运行设备：{self.device}")
        print(f"仅从本地缓存加载：{self.local_files_only}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )

        self.model.to(self.device)
        self.model.eval()

    def analyze(self, text: str) -> BettaFishSentimentResult:
        text = (text or "").strip()

        if not text:
            return BettaFishSentimentResult(
                text=text,
                is_success=False,
                error_message="Empty text.",
            )

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()

            distribution = self._build_probability_distribution(probabilities)

            best_index = int(
                max(
                    range(len(probabilities)),
                    key=lambda i: probabilities[i],
                )
            )

            sentiment_level = self._label_from_index(best_index)
            confidence = round(float(probabilities[best_index]), 4)

            return BettaFishSentimentResult(
                text=text,
                sentiment_level=sentiment_level,
                sentiment=self._coarse_sentiment(sentiment_level),
                sentiment_score=self._score_from_level(sentiment_level),
                sentiment_confidence=confidence,
                sentiment_probability_distribution=distribution,
                is_success=True,
                error_message="",
            )

        except Exception as exc:
            return BettaFishSentimentResult(
                text=text,
                is_success=False,
                error_message=str(exc),
            )

    def analyze_batch(
        self,
        texts: list[str],
        limit: int | None = None,
    ) -> BettaFishBatchSentimentResult:
        selected_texts = texts[:limit] if limit else texts

        results: list[dict[str, Any]] = []
        confidences: list[float] = []
        distribution_counter = {label: 0 for label in SENTIMENT_LEVELS}

        for index, text in enumerate(selected_texts, start=1):
            print(f"正在分析第 {index}/{len(selected_texts)} 条文本")

            result = self.analyze(text)
            result_dict = result.to_dict()
            result_dict["index"] = index
            results.append(result_dict)

            if result.is_success:
                distribution_counter[result.sentiment_level] += 1
                confidences.append(result.sentiment_confidence)

        success_count = sum(1 for item in results if item.get("is_success"))
        failed_count = len(results) - success_count

        return BettaFishBatchSentimentResult(
            total_count=len(selected_texts),
            success_count=success_count,
            failed_count=failed_count,
            average_confidence=round(mean(confidences), 4) if confidences else 0.0,
            sentiment_level_distribution=distribution_counter,
            results=results,
        )

    def _label_from_index(self, index: int) -> str:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        raw_label = id2label.get(index)

        if raw_label:
            normalized = str(raw_label).strip().lower()
            if normalized in EN_TO_ZH_LABEL:
                return EN_TO_ZH_LABEL[normalized]

        return INDEX_TO_ZH_LABEL.get(index, "中性")

    def _build_probability_distribution(
        self,
        probabilities: list[float],
    ) -> dict[str, float]:
        distribution = {label: 0.0 for label in SENTIMENT_LEVELS}

        for index, probability in enumerate(probabilities):
            label = self._label_from_index(index)
            if label in distribution:
                distribution[label] = round(float(probability), 4)

        return distribution

    def _coarse_sentiment(self, sentiment_level: str) -> str:
        if sentiment_level in {"非常负面", "负面"}:
            return "negative"

        if sentiment_level in {"正面", "非常正面"}:
            return "positive"

        return "neutral"

    def _score_from_level(self, sentiment_level: str) -> float:
        mapping = {
            "非常负面": -1.0,
            "负面": -0.6,
            "中性": 0.0,
            "正面": 0.6,
            "非常正面": 1.0,
        }
        return mapping.get(sentiment_level, 0.0)


def main() -> None:
    analyzer = BettaFishSentimentAnalyzer(
        local_files_only=True,
    )

    demo_texts = [
        "网传瑞幸咖啡门店存在使用过期原料的问题，不少网友质疑其食品安全管理和品牌诚信。",
        "瑞幸咖啡这次回应很快，希望后续能把品控继续做好。",
        "瑞幸咖啡今天发布了新品活动，部分门店参与优惠。",
    ]

    batch_result = analyzer.analyze_batch(demo_texts)

    print(json.dumps(batch_result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()