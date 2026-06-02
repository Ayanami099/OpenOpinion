from __future__ import annotations

import re

from pydantic import ValidationError

from opinion_agent.llm import LLMError, OpenAICompatibleClient
from opinion_agent.schemas import Entity, EventType, Intent, QueryUnderstanding, RiskLevel


QUERY_UNDERSTANDING_SYSTEM = """你是 OpenOpinion 的中文事件与舆论理解模块。
只输出 JSON，不要输出 Markdown。需要识别主体、事件关键词、事件类型、风险等级、平台和搜索意图。
event_type 必须使用以下枚举值之一：
product_quality_crisis, product_safety_crisis, food_safety, celebrity_scandal,
campus_incident, corporate_pr_crisis, consumer_complaint, policy_dispute,
public_safety, financial_risk, labor_dispute, rumor_or_misinformation, other。
risk_level 必须使用 low, medium, medium_high, high。
required_intents 和 optional_intents 必须使用以下大写枚举值：
FACT, TIMELINE, OFFICIAL_RESPONSE, MEDIA_COVERAGE, PUBLIC_REACTION,
STAKEHOLDER_VIEW, RUMOR_CHECK, RISK_IMPACT, LEGAL_REGULATION, COMPETITOR_CONTEXT。
main_entities 必须是对象数组，例如 [{"name":"特朗普","type":"person","confidence":0.8}]。"""


def query_understanding_user_prompt(user_input: str) -> str:
    return f"""请解析事件/舆论输入，并输出符合以下字段的 JSON：

raw_input, main_entities, event_keywords, event_type, risk_level,
time_sensitivity, geographic_scope, possible_platforms, required_intents,
optional_intents, known_constraints。

用户输入：{user_input}
"""


EVENT_TYPE_ALIASES = {
    "产品质量": EventType.PRODUCT_QUALITY_CRISIS,
    "产品质量危机": EventType.PRODUCT_QUALITY_CRISIS,
    "产品安全": EventType.PRODUCT_SAFETY_CRISIS,
    "产品安全危机": EventType.PRODUCT_SAFETY_CRISIS,
    "食品安全": EventType.FOOD_SAFETY,
    "明星丑闻": EventType.CELEBRITY_SCANDAL,
    "公众人物争议": EventType.CELEBRITY_SCANDAL,
    "校园事件": EventType.CAMPUS_INCIDENT,
    "高校事件": EventType.CAMPUS_INCIDENT,
    "企业公关危机": EventType.CORPORATE_PR_CRISIS,
    "公关危机": EventType.CORPORATE_PR_CRISIS,
    "消费者投诉": EventType.CONSUMER_COMPLAINT,
    "政策争议": EventType.POLICY_DISPUTE,
    "外交访问": EventType.POLICY_DISPUTE,
    "外交事件": EventType.POLICY_DISPUTE,
    "国际关系": EventType.POLICY_DISPUTE,
    "公共安全": EventType.PUBLIC_SAFETY,
    "金融风险": EventType.FINANCIAL_RISK,
    "劳资纠纷": EventType.LABOR_DISPUTE,
    "劳动争议": EventType.LABOR_DISPUTE,
    "谣言": EventType.RUMOR_OR_MISINFORMATION,
    "不实信息": EventType.RUMOR_OR_MISINFORMATION,
    "其他": EventType.OTHER,
}
RISK_LEVEL_ALIASES = {
    "低": RiskLevel.LOW,
    "低风险": RiskLevel.LOW,
    "中": RiskLevel.MEDIUM,
    "中等": RiskLevel.MEDIUM,
    "中风险": RiskLevel.MEDIUM,
    "中等风险": RiskLevel.MEDIUM,
    "中高": RiskLevel.MEDIUM_HIGH,
    "中高风险": RiskLevel.MEDIUM_HIGH,
    "较高": RiskLevel.MEDIUM_HIGH,
    "高": RiskLevel.HIGH,
    "高风险": RiskLevel.HIGH,
}
INTENT_ALIASES = {
    "事实": Intent.FACT,
    "基本事实": Intent.FACT,
    "事件详情": Intent.FACT,
    "了解事件详情": Intent.FACT,
    "详情": Intent.FACT,
    "时间线": Intent.TIMELINE,
    "相关历史": Intent.TIMELINE,
    "历史": Intent.TIMELINE,
    "官方回应": Intent.OFFICIAL_RESPONSE,
    "官方表态": Intent.OFFICIAL_RESPONSE,
    "媒体报道": Intent.MEDIA_COVERAGE,
    "报道": Intent.MEDIA_COVERAGE,
    "公众反应": Intent.PUBLIC_REACTION,
    "舆论反应": Intent.PUBLIC_REACTION,
    "网友评价": Intent.PUBLIC_REACTION,
    "背景分析": Intent.STAKEHOLDER_VIEW,
    "相关方观点": Intent.STAKEHOLDER_VIEW,
    "利益相关方": Intent.STAKEHOLDER_VIEW,
    "辟谣": Intent.RUMOR_CHECK,
    "澄清": Intent.RUMOR_CHECK,
    "事实核查": Intent.RUMOR_CHECK,
    "影响评估": Intent.RISK_IMPACT,
    "风险影响": Intent.RISK_IMPACT,
    "影响": Intent.RISK_IMPACT,
    "监管法律": Intent.LEGAL_REGULATION,
    "法律监管": Intent.LEGAL_REGULATION,
    "对比分析": Intent.COMPETITOR_CONTEXT,
    "竞品对比": Intent.COMPETITOR_CONTEXT,
}


HIGH_RISK_KEYWORDS = [
    "食品安全",
    "过期",
    "变质",
    "中毒",
    "死亡",
    "监管",
    "处罚",
    "公共安全",
    "爆炸",
    "坍塌",
    "召回",
    "举报",
    "处分",
    "私吞",
    "挪用",
]


ORG_SUFFIXES = [
    "公安局",
    "市监局",
    "教育局",
    "委员会",
    "大学",
    "学院",
    "高校",
    "学校",
    "中学",
    "小学",
    "医院",
    "公司",
    "集团",
    "品牌",
    "平台",
    "政府",
    "协会",
]
ORG_PATTERN = re.compile(rf"([\u4e00-\u9fffA-Za-z0-9]{{2,30}}?(?:{'|'.join(ORG_SUFFIXES)}))")
PERSON_EVENT_MARKERS = [
    "国奖",
    "国家奖学金",
    "奖学金",
    "竞赛奖金",
    "比赛奖金",
    "奖金",
    "私吞",
    "挪用",
    "处分",
    "通报",
    "举报",
    "事件",
    "争议",
    "舆情",
]
PERSON_MARKER_PATTERN = re.compile(rf"^([\u4e00-\u9fff]{{2,4}})(?=(?:{'|'.join(PERSON_EVENT_MARKERS)}))")
PERSON_STOPWORDS = {
    "国奖",
    "奖金",
    "事件",
    "舆情",
    "争议",
    "学生",
    "老师",
    "教师",
    "官方",
    "学校",
    "大学",
    "学院",
    "高校",
}
LOCATION_NAMES = {
    "中国",
    "美国",
    "北京",
    "华盛顿",
    "香港",
    "澳门",
    "台湾",
}
KNOWN_EVENT_KEYWORDS = [
    "私吞奖金",
    "竞赛奖金",
    "比赛奖金",
    "国家奖学金",
    "奖学金评选",
    "过期原料",
    "食品安全",
    "官方回应",
    "情况说明",
    "处罚决定",
    "饮水安全",
    "市监局",
    "教育局",
    "通报",
    "处分",
    "整改",
    "投诉",
    "举报",
    "辟谣",
    "澄清",
    "反转",
    "监管",
    "消费者",
    "宿舍",
    "国奖",
    "私吞",
    "挪用",
    "保研",
    "公示",
    "访华",
    "外交访问",
    "中美关系",
    "关税",
    "贸易谈判",
    "元首会晤",
]
CAMPUS_AWARD_TRIGGERS = {"国奖", "国家奖学金", "奖学金", "竞赛奖金", "比赛奖金", "奖金", "私吞"}
CAMPUS_AWARD_EXPANSIONS = ["国奖", "竞赛奖金", "私吞奖金", "处分", "国家奖学金"]


class QueryUnderstandingEngine:
    def __init__(self, llm: OpenAICompatibleClient | None = None) -> None:
        self.llm = llm or OpenAICompatibleClient.from_env()

    def understand(self, user_input: str) -> QueryUnderstanding:
        fallback = self._heuristic(user_input)
        try:
            raw = self.llm.chat_json(
                QUERY_UNDERSTANDING_SYSTEM,
                query_understanding_user_prompt(user_input),
                fallback=fallback.model_dump(mode="json"),
            )
            understanding = QueryUnderstanding.model_validate(self._normalize_model_output(raw, user_input))
            return self._post_process(understanding, fallback)
        except LLMError:
            raise
        except (ValidationError, ValueError, KeyError) as exc:
            raise LLMError(f"Query understanding model returned invalid output: {exc}") from exc

    def _normalize_model_output(self, raw: object, user_input: str) -> dict:
        if not isinstance(raw, dict):
            raise ValueError(f"expected JSON object, got {type(raw).__name__}")
        data = dict(raw)
        data["raw_input"] = str(data.get("raw_input") or user_input)
        data["main_entities"] = self._normalize_entities(data.get("main_entities", []))
        data["event_keywords"] = self._normalize_string_list(data.get("event_keywords", []))
        data["event_type"] = self._normalize_event_type(data.get("event_type"))
        data["risk_level"] = self._normalize_risk_level(data.get("risk_level"))
        data["geographic_scope"] = self._normalize_geographic_scope(data.get("geographic_scope", "unknown"))
        data["possible_platforms"] = self._normalize_string_list(data.get("possible_platforms", [])) or [
            "general_web",
            "news",
            "weibo",
            "xiaohongshu",
            "zhihu",
            "official",
        ]
        data["required_intents"] = self._normalize_intents(data.get("required_intents", []))
        data["optional_intents"] = self._normalize_intents(data.get("optional_intents", []))
        data["known_constraints"] = self._normalize_constraints(data.get("known_constraints", {}))
        return data

    def _normalize_entities(self, value: object) -> list[dict[str, object]]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        entities: list[dict[str, object]] = []
        for item in items:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    entities.append({"name": name, "type": self._entity_type(name), "confidence": 0.7})
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("entity") or "").strip()
                if not name:
                    continue
                entity_type = str(item.get("type") or self._entity_type(name))
                confidence = item.get("confidence", 0.7)
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.7
                entities.append({"name": name, "type": entity_type, "confidence": max(0.0, min(1.0, confidence))})
        return entities

    def _normalize_event_type(self, value: object) -> str:
        if isinstance(value, EventType):
            return value.value
        text = str(value or "").strip()
        if not text:
            return EventType.OTHER.value
        for item in EventType:
            if text == item.value:
                return item.value
        return EVENT_TYPE_ALIASES.get(text, EventType.OTHER).value

    def _normalize_risk_level(self, value: object) -> str:
        if isinstance(value, RiskLevel):
            return value.value
        text = str(value or "").strip()
        if not text:
            return RiskLevel.MEDIUM.value
        for item in RiskLevel:
            if text == item.value:
                return item.value
        return RISK_LEVEL_ALIASES.get(text, RiskLevel.MEDIUM).value

    def _normalize_intents(self, value: object) -> list[str]:
        items = value if isinstance(value, list) else [value] if value else []
        intents: list[str] = []
        for item in items:
            if isinstance(item, Intent):
                normalized = item.value
            else:
                text = str(item or "").strip()
                if not text:
                    continue
                normalized = text if text in {intent.value for intent in Intent} else INTENT_ALIASES.get(text)
                if isinstance(normalized, Intent):
                    normalized = normalized.value
            if normalized and normalized not in intents:
                intents.append(str(normalized))
        return intents

    def _normalize_geographic_scope(self, value: object) -> str:
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip()) or "unknown"
        if isinstance(value, dict):
            return ", ".join(str(item).strip() for item in value.values() if str(item).strip()) or "unknown"
        return str(value or "unknown").strip() or "unknown"

    def _normalize_constraints(self, value: object) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"notes": [str(item) for item in value]}
        if value in {None, ""}:
            return {}
        return {"note": str(value)}

    def _normalize_string_list(self, value: object) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [str(item).strip() for item in items if str(item).strip()]

    def _heuristic(self, user_input: str) -> QueryUnderstanding:
        event_type = self._event_type(user_input)
        risk = RiskLevel.HIGH if any(k in user_input for k in HIGH_RISK_KEYWORDS) else RiskLevel.MEDIUM
        entities = self._extract_entities(user_input)
        if not entities:
            entity = self._extract_entity(user_input)
            entities = [Entity(name=entity, type=self._entity_type(entity), confidence=0.6)]
        keywords = self._extract_keywords(user_input, event_type, [entity.name for entity in entities])
        platforms = ["general_web", "news", "weibo", "xiaohongshu", "zhihu", "official"]
        intents = [
            Intent.FACT,
            Intent.OFFICIAL_RESPONSE,
            Intent.MEDIA_COVERAGE,
            Intent.PUBLIC_REACTION,
            Intent.RUMOR_CHECK,
            Intent.RISK_IMPACT,
        ]
        if event_type in {EventType.FOOD_SAFETY, EventType.PUBLIC_SAFETY, EventType.PRODUCT_SAFETY_CRISIS}:
            intents.append(Intent.LEGAL_REGULATION)
        return QueryUnderstanding(
            raw_input=user_input,
            main_entities=entities,
            event_keywords=keywords,
            event_type=event_type,
            risk_level=risk,
            time_sensitivity="recent",
            possible_platforms=platforms,
            required_intents=intents,
            optional_intents=[Intent.STAKEHOLDER_VIEW, Intent.COMPETITOR_CONTEXT],
            known_constraints={"time_range": "recent", "language": "zh", "region": "cn"},
        )

    def _event_type(self, text: str) -> EventType:
        if any(k in text for k in ["奶茶", "餐饮", "食品", "过期", "变质", "食材"]):
            return EventType.FOOD_SAFETY
        if any(k in text for k in ["校园", "学校", "高校", "大学", "学院", "学生", "宿舍", "国奖", "奖学金", "竞赛奖金", "保研", "评奖"]):
            return EventType.CAMPUS_INCIDENT
        if any(k in text for k in ["股价", "金融", "暴雷", "兑付"]):
            return EventType.FINANCIAL_RISK
        if any(k in text for k in ["访华", "访美", "外交", "中美", "关税", "贸易谈判", "元首会晤"]):
            return EventType.POLICY_DISPUTE
        if any(k in text for k in ["员工", "裁员", "欠薪", "劳资"]):
            return EventType.LABOR_DISPUTE
        if any(k in text for k in ["谣言", "不实", "网传"]):
            return EventType.RUMOR_OR_MISINFORMATION
        if any(k in text for k in ["明星", "艺人", "公众人物"]):
            return EventType.CELEBRITY_SCANDAL
        if any(k in text for k in ["投诉", "维权", "消费者"]):
            return EventType.CONSUMER_COMPLAINT
        return EventType.OTHER

    def _post_process(self, understanding: QueryUnderstanding, fallback: QueryUnderstanding) -> QueryUnderstanding:
        raw_input = understanding.raw_input
        raw_compact = self._compact(raw_input)
        current_entities = [entity.name for entity in understanding.main_entities]
        current_is_underparsed = (
            not current_entities
            or any(self._compact(name) == raw_compact for name in current_entities)
            or (len(current_entities) == 1 and len(current_entities[0]) >= max(10, len(raw_input) - 2))
        )
        if current_is_underparsed and fallback.main_entities:
            understanding.main_entities = fallback.main_entities

        entity_names = [entity.name for entity in understanding.main_entities]
        current_keywords = self._clean_keywords(understanding.event_keywords, entity_names)
        fallback_keywords = self._clean_keywords(fallback.event_keywords, entity_names)
        if not current_keywords or any(self._compact(keyword) == raw_compact for keyword in current_keywords):
            understanding.event_keywords = fallback_keywords
        else:
            understanding.event_keywords = self._merge_unique(current_keywords, fallback_keywords)[:8]

        if understanding.event_type == EventType.OTHER and fallback.event_type != EventType.OTHER:
            understanding.event_type = fallback.event_type
        if understanding.risk_level == RiskLevel.MEDIUM and fallback.risk_level != RiskLevel.MEDIUM:
            understanding.risk_level = fallback.risk_level
        return understanding

    def _extract_entities(self, text: str) -> list[Entity]:
        candidates: list[tuple[int, str, str, float]] = []
        org_spans: list[tuple[int, int]] = []
        for match in ORG_PATTERN.finditer(text):
            name = match.group(1).strip()
            if self._valid_entity_name(name):
                org_spans.append(match.span(1))
                candidates.append((match.start(1), name, self._entity_type(name), 0.82))

        for _, end in org_spans:
            tail = text[end:]
            person_match = PERSON_MARKER_PATTERN.search(tail)
            if person_match:
                person = person_match.group(1)
                if self._valid_person_name(person):
                    candidates.append((end + person_match.start(1), person, "person", 0.72))

        leading_person = re.match(
            rf"([\u4e00-\u9fff]{{2,4}})(?=(?:{'|'.join(PERSON_EVENT_MARKERS)}))",
            text,
        )
        if leading_person and self._valid_person_name(leading_person.group(1)):
            candidates.append((leading_person.start(1), leading_person.group(1), "person", 0.68))

        entities: list[Entity] = []
        seen: set[str] = set()
        for _, name, entity_type, confidence in sorted(candidates, key=lambda item: item[0]):
            compact = self._compact(name)
            if compact in seen:
                continue
            seen.add(compact)
            entities.append(Entity(name=name, type=entity_type, confidence=confidence))
        return entities[:5]

    def _extract_entity(self, text: str) -> str:
        entities = self._extract_entities(text)
        if entities:
            return entities[0].name
        patterns = [
            r"([\u4e00-\u9fa5A-Za-z0-9]+品牌)",
            r"([\u4e00-\u9fa5A-Za-z0-9]+公司)",
            r"([\u4e00-\u9fa5A-Za-z0-9]+高校)",
            r"([\u4e00-\u9fa5A-Za-z0-9]+学校)",
            r"([\u4e00-\u9fa5A-Za-z0-9]+大学)",
            r"([\u4e00-\u9fa5A-Za-z0-9]+学院)",
            r"([\u4e00-\u9fa5A-Za-z0-9]+明星)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        cleaned = re.split(r"被曝|被指|涉嫌|发生|回应|通报|投诉|反映", text)[0].strip()
        return cleaned or text[:12]

    def _entity_type(self, entity: str) -> str:
        if entity in LOCATION_NAMES or entity.endswith(("国", "省", "市")):
            return "location"
        if "品牌" in entity or "公司" in entity:
            return "brand"
        if any(suffix in entity for suffix in ["高校", "学校", "大学", "学院", "医院", "集团", "平台", "政府", "局", "委员会", "协会"]):
            return "organization"
        if self._valid_person_name(entity):
            return "person"
        return "unknown"

    def _extract_keywords(self, text: str, event_type: EventType, entity_names: list[str] | None = None) -> list[str]:
        entity_names = entity_names or []
        keywords = [keyword for keyword in KNOWN_EVENT_KEYWORDS if keyword in text]
        if event_type == EventType.CAMPUS_INCIDENT and any(trigger in text for trigger in CAMPUS_AWARD_TRIGGERS):
            keywords = self._merge_unique(keywords, CAMPUS_AWARD_EXPANSIONS)
        if "过期" in text and "过期原料" not in keywords:
            keywords.append("过期")
        if event_type == EventType.FOOD_SAFETY and "食品安全" not in keywords:
            keywords.append("食品安全")
        keywords = self._clean_keywords(keywords, entity_names)
        if not keywords:
            keywords = self._fallback_keywords(text, entity_names)
        return keywords[:8]

    def _fallback_keywords(self, text: str, entity_names: list[str]) -> list[str]:
        cleaned = text
        for entity in entity_names:
            cleaned = cleaned.replace(entity, " ")
        cleaned = re.sub(r"(事件|舆情|风波|争议)$", "", cleaned.strip())
        pieces = [
            piece.strip()
            for piece in re.split(r"[，,。；;：:\s]+|被曝|被指|涉嫌|发生|回应|通报|投诉|反映|关于|有关|的", cleaned)
            if piece.strip()
        ]
        return self._clean_keywords(pieces, entity_names)[:4]

    def _clean_keywords(self, keywords: list[str], entity_names: list[str]) -> list[str]:
        cleaned: list[str] = []
        for keyword in keywords:
            item = keyword.strip()
            for entity in entity_names:
                item = item.replace(entity, " ")
            item = re.sub(r"\s+", " ", item).strip()
            item = re.sub(r"(事件|舆情|风波|争议)$", "", item)
            if not item or item in {"事件", "舆情", "风波", "争议"}:
                continue
            if not any(self._compact(item) == self._compact(existing) for existing in cleaned):
                cleaned.append(item)
        return cleaned

    def _merge_unique(self, first: list[str], second: list[str]) -> list[str]:
        merged: list[str] = []
        for item in [*first, *second]:
            if not any(self._compact(item) == self._compact(existing) for existing in merged):
                merged.append(item)
        return merged

    def _valid_entity_name(self, name: str) -> bool:
        compact = self._compact(name)
        return bool(compact) and compact not in {self._compact(word) for word in PERSON_STOPWORDS}

    def _valid_person_name(self, name: str) -> bool:
        if not re.fullmatch(r"[\u4e00-\u9fff]{2,4}", name):
            return False
        if name in PERSON_STOPWORDS:
            return False
        if name in LOCATION_NAMES:
            return False
        return not any(suffix in name for suffix in ORG_SUFFIXES)

    def _compact(self, text: str) -> str:
        return re.sub(r"\s+", "", text).lower()
