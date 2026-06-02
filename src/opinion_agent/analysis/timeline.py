from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

from opinion_agent.schemas import Evidence, Intent, SourceType, Timeline, TimelineEvent


class TimelineBuilder:
    def build(self, evidences: list[Evidence]) -> Timeline:
        grouped: dict[str, list[Evidence]] = defaultdict(list)
        for ev in evidences:
            time = self._event_time(ev)
            grouped[time].append(ev)

        events: list[TimelineEvent] = []
        for time in sorted(grouped):
            items = grouped[time]
            stage = self._stage(items)
            events.append(
                TimelineEvent(
                    time=time,
                    stage=stage,
                    event=self._describe(stage, items),
                    evidence_ids=[ev.evidence_id for ev in items[:5]],
                    source_types=sorted({ev.source_type.value for ev in items}),
                    confidence=min(0.95, 0.55 + 0.1 * len(items)),
                )
            )
        return Timeline(timeline=events)

    def _event_time(self, ev: Evidence) -> str:
        for text in [ev.content, ev.title]:
            match = re.search(r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})", text)
            if match:
                year, month, day = map(int, match.groups())
                return f"{year:04d}-{month:02d}-{day:02d}"
        if ev.published_at:
            try:
                return datetime.fromisoformat(ev.published_at.replace("Z", "+00:00")).date().isoformat()
            except ValueError:
                return ev.published_at[:10]
        return ev.retrieved_at[:10]

    def _stage(self, evidences: list[Evidence]) -> str:
        intents = {ev.intent for ev in evidences}
        source_types = {ev.source_type for ev in evidences}
        text = " ".join(f"{ev.title} {ev.content}" for ev in evidences)
        if Intent.OFFICIAL_RESPONSE in intents or SourceType.OFFICIAL in source_types:
            return "official_response"
        if Intent.LEGAL_REGULATION in intents or SourceType.GOVERNMENT in source_types:
            return "regulatory_involvement"
        if Intent.RUMOR_CHECK in intents:
            return "rumor_correction"
        if Intent.RISK_IMPACT in intents or any(k in text for k in ["处罚", "整改", "赔偿", "召回"]):
            return "resolution"
        if Intent.PUBLIC_REACTION in intents:
            return "public_backlash"
        if Intent.MEDIA_COVERAGE in intents or SourceType.MAINSTREAM_MEDIA in source_types:
            return "media_amplification"
        return "initial_exposure"

    def _describe(self, stage: str, evidences: list[Evidence]) -> str:
        first = evidences[0]
        descriptions = {
            "official_response": "主体或相关官方渠道发布回应、说明或处理措施。",
            "regulatory_involvement": "监管或权威机构介入，事件进入核查和处理阶段。",
            "rumor_correction": "出现辟谣、澄清、反转或真实性核查信息。",
            "resolution": "证据显示事件出现整改、处罚、赔偿或后续影响信息。",
            "public_backlash": "社交平台出现集中讨论和公众情绪反馈。",
            "media_amplification": "媒体跟进报道，事件关注度上升。",
            "initial_exposure": "事件相关信息首次或早期曝光。",
        }
        return descriptions.get(stage, first.summary or first.title)
