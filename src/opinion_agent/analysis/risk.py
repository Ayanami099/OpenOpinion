from __future__ import annotations

from datetime import datetime

from opinion_agent.config import load_configs
from opinion_agent.schemas import (
    Evidence,
    EventType,
    OpinionMiningResult,
    RiskAssessment,
    RiskDimension,
    RiskLevel,
    SourceType,
    Timeline,
)
from opinion_agent.text_utils import clamp


SENTIMENT_WEIGHTS = {
    SourceType.ORDINARY_SOCIAL_POST: 1.0,
    SourceType.VERIFIED_SOCIAL_ACCOUNT: 0.9,
    SourceType.FORUM_POST: 0.8,
    SourceType.MAINSTREAM_MEDIA: 0.4,
    SourceType.LOCAL_MEDIA: 0.4,
    SourceType.OFFICIAL: 0.2,
    SourceType.GOVERNMENT: 0.2,
}


class RiskAssessor:
    def __init__(self) -> None:
        self.weights = load_configs()["risk_weights"].get("overall", {})
        self.adjustments = load_configs()["risk_weights"].get("event_type_adjustments", {})

    def assess(
        self,
        event_type: EventType,
        evidences: list[Evidence],
        opinion: OpinionMiningResult,
        timeline: Timeline,
    ) -> RiskAssessment:
        dimensions = {
            "heat": self._heat(evidences),
            "negative_ratio": self._negative_ratio(evidences, opinion),
            "spread_speed": self._spread_speed(timeline),
            "authority_involvement": self._authority(evidences),
            "secondary_spread_risk": self._secondary_spread(event_type, evidences),
            "uncertainty_risk": self._uncertainty(evidences),
        }
        score = sum(dimensions[key].score * float(self.weights.get(key, 0)) for key in dimensions)
        score = clamp(score + float(self.adjustments.get(event_type.value, 0)))
        level = self._level(score)
        return RiskAssessment(
            overall_risk_level=level,
            overall_risk_score=score,
            dimensions=dimensions,
            risk_summary=self._summary(level, dimensions),
        )

    def _heat(self, evidences: list[Evidence]) -> RiskDimension:
        count_score = clamp(len(evidences) / 30)
        platform_score = clamp(len({ev.platform for ev in evidences}) / 5)
        social_count = sum(1 for ev in evidences if ev.source_type in {SourceType.ORDINARY_SOCIAL_POST, SourceType.VERIFIED_SOCIAL_ACCOUNT, SourceType.FORUM_POST})
        media_count = sum(1 for ev in evidences if ev.source_type in {SourceType.MAINSTREAM_MEDIA, SourceType.LOCAL_MEDIA, SourceType.INDUSTRY_MEDIA})
        score = clamp(0.25 * count_score + 0.25 * platform_score + 0.20 * clamp(social_count / 10) + 0.20 * clamp(media_count / 8) + 0.10 * 0.6)
        signals = [f"保留证据 {len(evidences)} 条", f"覆盖平台 {len({ev.platform for ev in evidences})} 个"]
        if social_count:
            signals.append("存在社交平台讨论")
        if media_count:
            signals.append("存在媒体报道")
        return RiskDimension(score=score, level=self._level(score), signals=signals)

    def _negative_ratio(self, evidences: list[Evidence], opinion: OpinionMiningResult) -> RiskDimension:
        ev_by_id = {ev.evidence_id: ev for ev in evidences}
        total = 0.0
        negative = 0.0
        for sentiment in opinion.sentiments:
            ev = ev_by_id.get(sentiment.evidence_id)
            if not ev:
                continue
            weight = SENTIMENT_WEIGHTS.get(ev.source_type, 0.4)
            total += weight
            if sentiment.sentiment == "negative":
                negative += weight
            elif sentiment.sentiment == "mixed":
                negative += weight * 0.5
        ratio = negative / total if total else 0
        signals = [f"加权负面率约 {ratio:.2f}"]
        if ratio >= 0.6:
            signals.append("负面观点占比较高")
        return RiskDimension(score=clamp(ratio), level=self._level(ratio), signals=signals, extra={"estimated_ratio": ratio})

    def _spread_speed(self, timeline: Timeline) -> RiskDimension:
        if len(timeline.timeline) < 2:
            return RiskDimension(score=0.35, level=RiskLevel.LOW, signals=["时间线节点较少，扩散速度证据不足"])
        dates = []
        for item in timeline.timeline:
            try:
                dates.append(datetime.fromisoformat(item.time[:10]))
            except ValueError:
                continue
        if len(dates) < 2:
            score = 0.45
        else:
            days = max(1, (max(dates) - min(dates)).days + 1)
            score = clamp(len(timeline.timeline) / days / 3 + len(timeline.timeline) / 10)
        signals = [f"时间线节点 {len(timeline.timeline)} 个"]
        if score >= 0.65:
            signals.append("短时间内出现多个阶段信息")
        return RiskDimension(score=score, level=self._level(score), signals=signals)

    def _authority(self, evidences: list[Evidence]) -> RiskDimension:
        authority = [
            ev
            for ev in evidences
            if ev.source_type in {SourceType.GOVERNMENT, SourceType.OFFICIAL, SourceType.MAINSTREAM_MEDIA}
        ]
        score = len(authority) / len(evidences) if evidences else 0
        level_signal = "none"
        if any(ev.source_type == SourceType.GOVERNMENT for ev in authority):
            level_signal = "regulator_involved"
            score = max(score, 0.70)
        elif any(ev.source_type == SourceType.OFFICIAL for ev in authority):
            level_signal = "official_response"
            score = max(score, 0.60)
        elif authority:
            level_signal = "media_only"
            score = max(score, 0.45)
        return RiskDimension(
            score=clamp(score),
            level=self._level(score),
            signals=[f"权威参与状态：{level_signal}", f"权威证据 {len(authority)} 条"],
            extra={"authority_state": level_signal},
        )

    def _secondary_spread(self, event_type: EventType, evidences: list[Evidence]) -> RiskDimension:
        score = 0.45
        signals: list[str] = []
        if event_type in {EventType.FOOD_SAFETY, EventType.PUBLIC_SAFETY, EventType.PRODUCT_SAFETY_CRISIS}:
            score += 0.25
            signals.append("事件类型天然具备较高二次传播风险")
        if any(ev.source_type == SourceType.ORDINARY_SOCIAL_POST for ev in evidences):
            score += 0.10
            signals.append("社交平台情绪表达可能带来二次传播")
        return RiskDimension(score=clamp(score), level=self._level(score), signals=signals or ["暂未发现明显二次传播信号"])

    def _uncertainty(self, evidences: list[Evidence]) -> RiskDimension:
        low_cred = sum(1 for ev in evidences if ev.scores.credibility < 0.5)
        rumor = sum(1 for ev in evidences if ev.intent.value == "RUMOR_CHECK")
        score = clamp(low_cred / max(len(evidences), 1) + (0.15 if rumor == 0 else 0))
        signals = [f"低可信证据 {low_cred} 条"]
        if rumor == 0:
            signals.append("尚缺少辟谣或澄清证据")
        return RiskDimension(score=score, level=self._level(score), signals=signals)

    def _level(self, score: float) -> RiskLevel:
        if score >= 0.75:
            return RiskLevel.HIGH
        if score >= 0.60:
            return RiskLevel.MEDIUM_HIGH
        if score >= 0.35:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _summary(self, level: RiskLevel, dimensions: dict[str, RiskDimension]) -> str:
        top = sorted(dimensions.items(), key=lambda item: item[1].score, reverse=True)[:2]
        names = "、".join(name for name, _ in top)
        return f"综合判断当前风险等级为 {level.value}，主要由 {names} 维度驱动。"
