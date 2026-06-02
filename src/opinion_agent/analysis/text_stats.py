from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime

from opinion_agent.analysis.opinion import NEGATIVE_TERMS, POSITIVE_TERMS, RISK_TERMS
from opinion_agent.schemas import Evidence, OpinionMiningResult, SourceType, TextAnalysisResult
from opinion_agent.text_utils import text_tokens


SOCIAL_SOURCE_TYPES = {
    SourceType.ORDINARY_SOCIAL_POST,
    SourceType.VERIFIED_SOCIAL_ACCOUNT,
    SourceType.FORUM_POST,
    SourceType.ANONYMOUS_POST,
}

SOURCE_SENTIMENT_WEIGHTS = {
    SourceType.ORDINARY_SOCIAL_POST: 1.0,
    SourceType.VERIFIED_SOCIAL_ACCOUNT: 0.9,
    SourceType.FORUM_POST: 0.8,
    SourceType.ANONYMOUS_POST: 0.7,
    SourceType.MAINSTREAM_MEDIA: 0.4,
    SourceType.LOCAL_MEDIA: 0.4,
    SourceType.INDUSTRY_MEDIA: 0.4,
    SourceType.OFFICIAL: 0.2,
    SourceType.GOVERNMENT: 0.2,
    SourceType.CONTENT_FARM: 0.2,
    SourceType.UNKNOWN: 0.4,
}

KEYWORD_STOPWORDS = {
    "https",
    "http",
    "www",
    "com",
    "cn",
    "jpg",
    "png",
    "html",
    "image",
    "static",
    "assets",
    "新闻",
    "报道",
    "相关",
    "事件",
    "网页",
    "客户端",
}


class TextStatsAnalyzer:
    def analyze(self, evidences: list[Evidence], opinion: OpinionMiningResult) -> TextAnalysisResult:
        if not evidences:
            return TextAnalysisResult(
                quality_averages={"relevance": 0, "credibility": 0, "freshness": 0, "overall": 0},
                text_volume={"total_chars": 0, "average_chars": 0, "total_tokens": 0},
            )

        evidence_by_id = {ev.evidence_id: ev for ev in evidences}
        sentiment_by_id = {item.evidence_id: item for item in opinion.sentiments}
        sentiment_counts = Counter(item.sentiment for item in opinion.sentiments)
        total_sentiments = sum(sentiment_counts.values())

        source_sentiment: dict[str, Counter[str]] = defaultdict(Counter)
        for ev in evidences:
            sentiment = sentiment_by_id.get(ev.evidence_id)
            label = sentiment.sentiment if sentiment else "unknown"
            source_sentiment[ev.source_type.value][label] += 1

        aspect_sentiment: dict[str, Counter[str]] = defaultdict(Counter)
        for unit in opinion.opinion_units:
            aspect_sentiment[unit.aspect][unit.sentiment] += 1

        positive_terms = Counter(term for item in opinion.sentiments for term in item.positive_hits)
        negative_terms = Counter(term for item in opinion.sentiments for term in item.negative_hits)
        all_text = " ".join(self._evidence_text(ev) for ev in evidences)
        keyword_counts = self._keyword_counts(evidences, all_text)

        total_score = sum(item.sentiment_score for item in opinion.sentiments)
        weighted_negative_ratio = self._weighted_negative_ratio(evidences, sentiment_by_id)

        total_chars = sum(len(self._evidence_text(ev)) for ev in evidences)
        total_tokens = sum(len(text_tokens(self._evidence_text(ev))) for ev in evidences)

        return TextAnalysisResult(
            evidence_count=len(evidences),
            platform_distribution=dict(Counter(ev.platform for ev in evidences)),
            source_type_distribution=dict(Counter(ev.source_type.value for ev in evidences)),
            intent_distribution=dict(Counter(ev.intent.value for ev in evidences)),
            date_distribution=dict(Counter(self._date_key(ev) for ev in evidences)),
            sentiment_counts=dict(sentiment_counts),
            sentiment_ratios={key: round(value / total_sentiments, 4) for key, value in sentiment_counts.items()} if total_sentiments else {},
            average_sentiment_score=round(total_score / total_sentiments, 4) if total_sentiments else 0,
            weighted_negative_ratio=round(weighted_negative_ratio, 4),
            source_sentiment_matrix={key: dict(value) for key, value in sorted(source_sentiment.items())},
            aspect_sentiment_matrix={key: dict(value) for key, value in sorted(aspect_sentiment.items())},
            stance_distribution=dict(Counter(item.stance for item in opinion.stances)),
            top_keywords=[{"term": term, "count": count} for term, count in keyword_counts.most_common(20)],
            sentiment_term_hits=dict((positive_terms + negative_terms).most_common()),
            risk_term_hits=self._term_hits(all_text, RISK_TERMS),
            quality_averages=self._quality_averages(evidences),
            text_volume={
                "total_chars": float(total_chars),
                "average_chars": round(total_chars / len(evidences), 2),
                "total_tokens": float(total_tokens),
            },
        )

    def _weighted_negative_ratio(self, evidences: list[Evidence], sentiment_by_id: dict[str, object]) -> float:
        total = 0.0
        negative = 0.0
        for ev in evidences:
            sentiment = sentiment_by_id.get(ev.evidence_id)
            weight = SOURCE_SENTIMENT_WEIGHTS.get(ev.source_type, 0.4)
            total += weight
            label = getattr(sentiment, "sentiment", "unknown")
            if label == "negative":
                negative += weight
            elif label == "mixed":
                negative += weight * 0.5
        return negative / total if total else 0

    def _keyword_counts(self, evidences: list[Evidence], all_text: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for ev in evidences:
            for term in ev.entities + ev.event_mentions:
                if self._keep_keyword(term):
                    counts[term] += 3
            text = self._evidence_text(ev)
            for term in POSITIVE_TERMS + NEGATIVE_TERMS + RISK_TERMS:
                if term in text:
                    counts[term] += text.count(term)
            for hashtag in re.findall(r"#[^#\s]{2,30}#", text):
                counts[hashtag.strip("#")] += 2
        for token in text_tokens(all_text):
            if self._keep_keyword(token):
                counts[token] += 1
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,8}", all_text):
            if self._keep_keyword(chunk):
                counts[chunk] += 1
        return counts

    def _term_hits(self, text: str, terms: list[str]) -> dict[str, int]:
        return {term: text.count(term) for term in terms if term in text}

    def _quality_averages(self, evidences: list[Evidence]) -> dict[str, float]:
        return {
            "relevance": round(sum(ev.scores.relevance for ev in evidences) / len(evidences), 4),
            "credibility": round(sum(ev.scores.credibility for ev in evidences) / len(evidences), 4),
            "freshness": round(sum(ev.scores.freshness for ev in evidences) / len(evidences), 4),
            "overall": round(sum(ev.scores.overall for ev in evidences) / len(evidences), 4),
        }

    def _date_key(self, ev: Evidence) -> str:
        raw = ev.published_at or ev.retrieved_at
        if not raw:
            return "unknown"
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return raw[:10] if len(raw) >= 10 else raw

    def _evidence_text(self, ev: Evidence) -> str:
        return " ".join(part for part in [ev.title, ev.summary, ev.content] if part)

    def _keep_keyword(self, term: str) -> bool:
        cleaned = term.strip().lower()
        if len(cleaned) < 2 or cleaned in KEYWORD_STOPWORDS:
            return False
        if cleaned.isdigit():
            return False
        return True
