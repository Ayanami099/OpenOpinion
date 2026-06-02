from opinion_agent.research import QueryUnderstandingEngine, SearchPlanner
from opinion_agent.schemas import Entity, EventType, Intent, QueryUnderstanding, RiskLevel


def test_query_understanding_schema_defaults_required_intents():
    understanding = QueryUnderstanding(
        raw_input="某奶茶品牌被曝使用过期原料",
        main_entities=[Entity(name="某奶茶品牌", type="brand", confidence=0.9)],
        event_keywords=["过期原料", "食品安全"],
        event_type=EventType.FOOD_SAFETY,
        risk_level=RiskLevel.HIGH,
    )
    assert Intent.FACT in understanding.required_intents
    assert Intent.OFFICIAL_RESPONSE in understanding.required_intents


def test_search_planner_covers_required_intents():
    understanding = QueryUnderstanding(
        raw_input="某奶茶品牌被曝使用过期原料",
        main_entities=[Entity(name="某奶茶品牌", type="brand")],
        event_keywords=["过期原料", "食品安全"],
        event_type=EventType.FOOD_SAFETY,
        risk_level=RiskLevel.HIGH,
        required_intents=[
            Intent.FACT,
            Intent.OFFICIAL_RESPONSE,
            Intent.MEDIA_COVERAGE,
            Intent.PUBLIC_REACTION,
            Intent.RUMOR_CHECK,
            Intent.RISK_IMPACT,
        ],
    )
    plan = SearchPlanner().create_initial_plan(understanding)
    planned_intents = {query.intent for query in plan.queries}
    assert len(plan.queries) >= 8
    assert set(understanding.required_intents).issubset(planned_intents)
    assert any("官方回应" in query.query for query in plan.queries)


def test_chinese_campus_award_understanding_extracts_specific_terms():
    engine = QueryUnderstandingEngine(llm=object())
    understanding = engine._heuristic("上海交通大学樊思睿国奖事件")
    entity_names = [entity.name for entity in understanding.main_entities]

    assert entity_names[:2] == ["上海交通大学", "樊思睿"]
    assert understanding.event_type == EventType.CAMPUS_INCIDENT
    assert {"国奖", "竞赛奖金", "私吞奖金", "处分"}.issubset(set(understanding.event_keywords))
    assert understanding.raw_input not in understanding.event_keywords


def test_search_planner_deduplicates_entity_and_keyword_terms():
    raw = "上海交通大学樊思睿国奖事件"
    understanding = QueryUnderstanding(
        raw_input=raw,
        main_entities=[Entity(name=raw, type="unknown")],
        event_keywords=[raw],
        event_type=EventType.CAMPUS_INCIDENT,
        risk_level=RiskLevel.MEDIUM,
        required_intents=[Intent.FACT, Intent.OFFICIAL_RESPONSE],
    )

    plan = SearchPlanner().create_initial_plan(understanding)

    assert plan.queries[0].query == raw
    assert all(f"{raw} {raw}" not in query.query for query in plan.queries)


def test_query_understanding_normalizes_chinese_llm_labels():
    engine = QueryUnderstandingEngine(llm=object())
    raw = {
        "raw_input": "特朗普访华",
        "main_entities": ["特朗普", "中国"],
        "event_keywords": ["外交访问", "中美关系"],
        "event_type": "外交访问",
        "risk_level": "低风险",
        "time_sensitivity": "recent",
        "geographic_scope": ["中国", "美国"],
        "possible_platforms": ["general_web", "news"],
        "required_intents": ["了解事件详情", "背景分析", "影响评估"],
        "optional_intents": ["相关历史", "对比分析"],
        "known_constraints": [],
    }

    understanding = QueryUnderstanding.model_validate(engine._normalize_model_output(raw, "特朗普访华"))

    assert [entity.name for entity in understanding.main_entities] == ["特朗普", "中国"]
    assert [entity.type for entity in understanding.main_entities] == ["person", "location"]
    assert understanding.event_type == EventType.POLICY_DISPUTE
    assert understanding.risk_level == RiskLevel.LOW
    assert understanding.geographic_scope == "中国, 美国"
    assert understanding.required_intents == [Intent.FACT, Intent.STAKEHOLDER_VIEW, Intent.RISK_IMPACT]
    assert understanding.optional_intents == [Intent.TIMELINE, Intent.COMPETITOR_CONTEXT]
    assert understanding.known_constraints == {"notes": []}
