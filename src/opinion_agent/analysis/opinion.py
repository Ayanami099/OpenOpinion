from __future__ import annotations

from collections import Counter, defaultdict

from opinion_agent.schemas import (
    Evidence,
    OpinionCluster,
    OpinionMiningResult,
    OpinionUnit,
    SentimentResult,
    SourceType,
    StanceResult,
)
from opinion_agent.text_utils import clamp, text_tokens


NEGATIVE_TERM_WEIGHTS = {
    "担心": 1.0,
    "担忧": 1.0,
    "失望": 1.1,
    "抵制": 1.4,
    "质疑": 1.1,
    "恶心": 1.3,
    "愤怒": 1.4,
    "不满": 1.1,
    "漏洞": 0.8,
    "处罚": 0.7,
    "违法": 1.3,
    "危机": 1.0,
    "风险": 0.8,
    "欺骗": 1.5,
    "隐瞒": 1.3,
    "谎称": 1.3,
    "伪造": 1.6,
    "造假": 1.5,
    "诈骗": 1.6,
    "私吞": 1.6,
    "侵占": 1.5,
    "贪婪": 1.2,
    "暗箱": 1.4,
    "特权": 1.1,
    "不公": 1.2,
    "争议": 0.8,
    "谴责": 1.3,
    "问责": 1.0,
    "失信": 1.3,
    "不诚信": 1.4,
    "泄露": 1.0,
    "网暴": 1.0,
    "翻车": 1.0,
}
POSITIVE_TERM_WEIGHTS = {
    "支持": 1.1,
    "认可": 1.0,
    "及时": 0.8,
    "透明": 1.0,
    "负责": 1.0,
    "整改": 0.9,
    "澄清": 0.8,
    "辟谣": 0.8,
    "道歉": 0.8,
    "回应": 0.6,
    "改正": 0.9,
    "补偿": 0.8,
    "赔偿": 0.8,
    "核查": 0.7,
    "通报": 0.6,
    "公开": 0.8,
    "公正": 1.0,
    "严肃处理": 1.0,
    "维护权益": 1.0,
}
NEGATIVE_TERMS = list(NEGATIVE_TERM_WEIGHTS)
POSITIVE_TERMS = list(POSITIVE_TERM_WEIGHTS)
RISK_TERMS = ["监管", "处罚", "处分", "整改", "赔偿", "补偿", "下架", "召回", "约谈", "违法", "问责", "危机", "舆情"]
EVENT_TERMS = ["过期原料", "食品安全", "门店", "消费者", "官方回应", "辟谣", "澄清", "监管", "奖学金", "竞赛奖金", "私吞奖金"]


class OpinionMiner:
    def mine(self, evidences: list[Evidence], target: str) -> OpinionMiningResult:
        sentiments: list[SentimentResult] = []
        stances: list[StanceResult] = []
        units: list[OpinionUnit] = []

        for ev in evidences:
            sentiment, score, confidence, reason, pos_hits, neg_hits, pos_score, neg_score = self._sentiment(ev)
            stance, stance_reason = self._stance(ev, sentiment)
            sentiments.append(
                SentimentResult(
                    evidence_id=ev.evidence_id,
                    sentiment=sentiment,
                    sentiment_score=score,
                    confidence=confidence,
                    reason=reason,
                    positive_hits=pos_hits,
                    negative_hits=neg_hits,
                    positive_score=round(pos_score, 3),
                    negative_score=round(neg_score, 3),
                )
            )
            stances.append(
                StanceResult(
                    evidence_id=ev.evidence_id,
                    target=target,
                    stance=stance,
                    confidence=0.78,
                    reason=stance_reason,
                )
            )
            units.append(
                OpinionUnit(
                    evidence_id=ev.evidence_id,
                    aspect=self._aspect(ev),
                    claim=self._claim(ev, sentiment),
                    sentiment=sentiment,
                    stance=stance,
                    keywords=self._keywords(ev.content + ev.title),
                    confidence=0.74,
                )
            )

        clusters = self._cluster(units, evidences)
        return OpinionMiningResult(
            sentiments=sentiments,
            stances=stances,
            opinion_units=units,
            opinion_clusters=clusters,
            keywords=self._keyword_groups(evidences),
        )

    def _sentiment(self, ev: Evidence) -> tuple[str, float, float, str, list[str], list[str], float, float]:
        text = f"{ev.title} {ev.content}"
        pos_hits, pos_score = self._weighted_hits(text, POSITIVE_TERM_WEIGHTS)
        neg_hits, neg_score = self._weighted_hits(text, NEGATIVE_TERM_WEIGHTS)
        if ev.source_type in {
            SourceType.MAINSTREAM_MEDIA,
            SourceType.GOVERNMENT,
            SourceType.OFFICIAL,
            SourceType.LOCAL_MEDIA,
        } and neg_score <= pos_score + 1:
            return "neutral", 0.0, 0.78, self._reason("新闻、官方或监管材料以事实陈述为主", pos_hits, neg_hits), pos_hits, neg_hits, pos_score, neg_score
        if pos_score and neg_score and abs(pos_score - neg_score) <= 0.75:
            return "mixed", 0.0, 0.82, self._reason("文本同时包含负面担忧和积极处理信号", pos_hits, neg_hits), pos_hits, neg_hits, pos_score, neg_score
        if neg_score > pos_score and neg_score > 0:
            score = clamp(-0.25 - 0.12 * neg_score + 0.04 * pos_score, -1, 0)
            return "negative", score, self._confidence(pos_score, neg_score), self._reason("文本包含明显质疑、担忧或批评表达", pos_hits, neg_hits), pos_hits, neg_hits, pos_score, neg_score
        if pos_score > neg_score and pos_score > 0:
            score = clamp(0.25 + 0.12 * pos_score - 0.04 * neg_score)
            return "positive", score, self._confidence(pos_score, neg_score), self._reason("文本包含支持、认可或积极整改表达", pos_hits, neg_hits), pos_hits, neg_hits, pos_score, neg_score
        return "neutral", 0.0, 0.58, "未发现明显情绪表达", pos_hits, neg_hits, pos_score, neg_score

    def _weighted_hits(self, text: str, weights: dict[str, float]) -> tuple[list[str], float]:
        hits: list[str] = []
        score = 0.0
        for term, weight in weights.items():
            count = text.count(term)
            if count:
                hits.append(term)
                score += count * weight
        return hits, score

    def _confidence(self, pos_score: float, neg_score: float) -> float:
        total = pos_score + neg_score
        margin = abs(pos_score - neg_score)
        return round(clamp(0.58 + min(total, 6) * 0.045 + min(margin, 4) * 0.045, 0.45, 0.92), 3)

    def _reason(self, prefix: str, pos_hits: list[str], neg_hits: list[str]) -> str:
        parts = [prefix]
        if neg_hits:
            parts.append(f"负面命中：{', '.join(neg_hits[:6])}")
        if pos_hits:
            parts.append(f"正面命中：{', '.join(pos_hits[:6])}")
        return "；".join(parts)

    def _stance(self, ev: Evidence, sentiment: str) -> tuple[str, str]:
        text = f"{ev.title} {ev.content}"
        if any(term in text for term in ["官方回应", "情况说明", "品牌发布", "整改"]):
            return "official_defense", "文本包含主体官方解释或处理措施"
        if any(term in text for term in ["监管", "市监局", "依法", "处罚决定"]):
            return "regulatory_action", "文本包含监管或法律处理立场"
        if any(term in text for term in ["辟谣", "澄清", "不实", "未证实"]):
            return "question_claim", "文本关注爆料真实性或澄清"
        if sentiment == "negative":
            return "criticize_entity", "文本对事件主体表达批评或不信任"
        if sentiment == "positive":
            return "support_entity", "文本对主体处理表示支持"
        return "neutral_report", "文本主要为中立报道或信息汇总"

    def _aspect(self, ev: Evidence) -> str:
        text = f"{ev.title} {ev.content}"
        if any(term in text for term in ["官方回应", "情况说明", "声明", "公告"]):
            return "官方回应"
        if any(term in text for term in ["监管", "处罚", "市监局", "违法"]):
            return "监管处理"
        if any(term in text for term in ["网友", "评论", "热议", "质疑"]):
            return "公众反应"
        if any(term in text for term in ["辟谣", "澄清", "反转", "不实"]):
            return "辟谣核查"
        return "事件事实"

    def _claim(self, ev: Evidence, sentiment: str) -> str:
        if ev.summary:
            return ev.summary
        return f"该证据围绕{ev.intent.value}提供{sentiment}倾向的信息。"

    def _keywords(self, text: str) -> list[str]:
        keyword_hits = [term for term in EVENT_TERMS + NEGATIVE_TERMS + RISK_TERMS if term in text]
        if keyword_hits:
            return list(dict.fromkeys(keyword_hits))[:8]
        counts = Counter(tok for tok in text_tokens(text) if len(tok) > 1)
        return [tok for tok, _ in counts.most_common(5)]

    def _cluster(self, units: list[OpinionUnit], evidences: list[Evidence]) -> list[OpinionCluster]:
        by_aspect: dict[str, list[OpinionUnit]] = defaultdict(list)
        evidence_by_id = {ev.evidence_id: ev for ev in evidences}
        for unit in units:
            by_aspect[unit.aspect].append(unit)

        clusters: list[OpinionCluster] = []
        for idx, (aspect, items) in enumerate(sorted(by_aspect.items()), start=1):
            sentiment_counts = Counter(item.sentiment for item in items)
            stance_counts = Counter(item.stance for item in items)
            total = len(items)
            keywords = Counter(kw for item in items for kw in item.keywords)
            evs = [evidence_by_id[item.evidence_id] for item in items if item.evidence_id in evidence_by_id]
            clusters.append(
                OpinionCluster(
                    cluster_id=f"oc_{idx:03d}",
                    cluster_name=aspect,
                    representative_claim=items[0].claim,
                    sentiment_distribution={k: v / total for k, v in sentiment_counts.items()},
                    stance_distribution={k: v / total for k, v in stance_counts.items()},
                    keywords=[kw for kw, _ in keywords.most_common(8)],
                    evidence_ids=[item.evidence_id for item in items],
                    source_platforms=sorted({ev.platform for ev in evs}),
                    cluster_size=total,
                    importance_score=clamp(0.35 + min(total, 10) / 10 * 0.55),
                )
            )
        return clusters

    def _keyword_groups(self, evidences: list[Evidence]) -> dict[str, list[str]]:
        all_text = " ".join(f"{ev.title} {ev.content}" for ev in evidences)
        return {
            "event_keywords": [term for term in EVENT_TERMS if term in all_text][:8],
            "sentiment_keywords": [term for term in NEGATIVE_TERMS + POSITIVE_TERMS if term in all_text][:8],
            "risk_keywords": [term for term in RISK_TERMS if term in all_text][:8],
            "emerging_keywords": [term for term, _ in Counter(text_tokens(all_text)).most_common(8) if len(term) > 1],
        }
