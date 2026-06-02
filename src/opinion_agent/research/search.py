from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Any

from opinion_agent.schemas import Intent, QueryResult, SearchPlan, SearchQuery, SearchResult, SearchRoundResult
from opinion_agent.text_utils import stable_id


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: SearchQuery, top_k: int) -> list[SearchResult]:
        raise NotImplementedError


class MockSearchProvider(SearchProvider):
    """Deterministic search provider used for tests and no-key local runs."""

    def search(self, query: SearchQuery, top_k: int) -> list[SearchResult]:
        now = datetime.now(UTC).replace(microsecond=0)
        templates = self._templates(query)
        results: list[SearchResult] = []
        for rank, (title_suffix, source, content, days_ago) in enumerate(templates[:top_k], start=1):
            url = f"https://example.com/{query.intent.value.lower()}/{stable_id('r', query.query_id, rank)}?utm_source=mock"
            results.append(
                SearchResult(
                    result_id=stable_id("r", query.query_id, rank),
                    query_id=query.query_id,
                    rank=rank,
                    title=f"{query.query} {title_suffix}",
                    url=url,
                    source=source,
                    snippet=content[:120],
                    published_at=(now - timedelta(days=days_ago)).date().isoformat(),
                    retrieved_at=now.isoformat().replace("+00:00", "Z"),
                    raw_content=content,
                    content_length=len(content),
                )
            )
        return results

    def _templates(self, query: SearchQuery) -> list[tuple[str, str, str, int]]:
        q = query.query
        base = [
            ("事件梳理", "澎湃新闻", f"{q}。多家媒体关注事件基本事实，消费者反映食品安全和处理进展问题。", 1),
            ("相关报道", "中国新闻网", f"{q}。报道称事件正在核实，相关门店和品牌方处理措施受到关注。", 2),
            ("讨论汇总", "微博网友", f"{q}。网友热议并表达担心、失望、质疑，也有人等待官方调查结果。", 1),
        ]
        if query.intent == Intent.OFFICIAL_RESPONSE:
            return [
                ("官方回应", "品牌官方账号", f"{q}。品牌发布情况说明，称已启动内部调查并将配合监管，涉事门店已整改。", 0),
                ("监管通报", "市场监管局", f"{q}。属地监管部门表示已关注相关情况，后续将依法处理。", 1),
            ] + base
        if query.intent == Intent.RUMOR_CHECK:
            return [
                ("澄清与反转核查", "澎湃新闻", f"{q}。目前未发现权威辟谣，但部分细节仍待官方调查确认。", 1),
                ("传言核实", "辟谣平台", f"{q}。网传信息存在夸大可能，需以监管和官方说明为准。", 2),
            ] + base
        if query.intent == Intent.PUBLIC_REACTION:
            return [
                ("网友评价", "微博网友", f"{q}。大量评论表达负面情绪，关键词包括担心、抵制、失望、食品安全。", 0),
                ("社区讨论", "知乎用户", f"{q}。部分用户认为应关注门店管理和监管处罚，也有人质疑爆料真实性。", 1),
                ("消费反馈", "小红书用户", f"{q}。消费者分享门店体验并要求品牌公开处理结果。", 1),
            ] + base
        if query.intent == Intent.RISK_IMPACT:
            return [
                ("处罚整改进展", "地方媒体", f"{q}。报道提到涉事门店整改、品牌致歉和可能面临的监管处罚。", 0),
                ("商业影响", "财经媒体", f"{q}。事件可能影响品牌口碑和短期门店客流。", 1),
            ] + base
        if query.intent == Intent.LEGAL_REGULATION:
            return [
                ("监管介入", "市场监管局", f"{q}。监管部门依法开展检查，若发现违法行为将作出处罚决定。", 0),
            ] + base
        return base


class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str | None = None, timeout: int = 60) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(self, query: SearchQuery, top_k: int) -> list[SearchResult]:
        if not self.enabled:
            raise RuntimeError("TAVILY_API_KEY is not configured")
        payload = {
            "api_key": self.api_key,
            "query": query.query,
            "search_depth": "advanced",
            "max_results": top_k,
            "include_answer": False,
            "include_raw_content": True,
        }
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return self._parse(query, data)

    def _parse(self, query: SearchQuery, data: dict[str, Any]) -> list[SearchResult]:
        now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        results: list[SearchResult] = []
        for rank, item in enumerate(data.get("results", []), start=1):
            url = item.get("url") or ""
            content = item.get("raw_content") or item.get("content") or ""
            results.append(
                SearchResult(
                    result_id=stable_id("r", query.query_id, rank, url),
                    query_id=query.query_id,
                    rank=rank,
                    title=item.get("title") or "",
                    url=url,
                    source=item.get("source") or item.get("title") or "unknown",
                    snippet=item.get("content") or "",
                    published_at=item.get("published_date"),
                    retrieved_at=now,
                    raw_content=content,
                    content_length=len(content),
                )
            )
        return results


class SearchExecutor:
    def __init__(self, provider: SearchProvider | None = None) -> None:
        self.provider = provider or self._default_provider()
        self.fallback_provider = MockSearchProvider()
        self.last_trace: list[dict[str, Any]] = []

    def execute(self, plan: SearchPlan, top_k: int = 5) -> SearchRoundResult:
        query_results: list[QueryResult] = []
        self.last_trace = []
        for query in sorted(plan.queries, key=lambda q: q.priority):
            provider_name = self.provider.__class__.__name__
            used_fallback = False
            error: str | None = None
            try:
                results = self.provider.search(query, top_k)
            except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                error = str(exc)
                used_fallback = True
                results = self.fallback_provider.search(query, top_k)
            self.last_trace.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "intent": query.intent.value,
                    "platform": query.platform,
                    "requested_provider": provider_name,
                    "used_fallback": used_fallback,
                    "fallback_provider": self.fallback_provider.__class__.__name__ if used_fallback else None,
                    "error": error,
                    "top_k": top_k,
                    "result_count": len(results),
                    "result_ids": [result.result_id for result in results],
                }
            )
            query_results.append(
                QueryResult(
                    query_id=query.query_id,
                    query=query.query,
                    intent=query.intent,
                    platform=query.platform,
                    results=results,
                )
            )
        return SearchRoundResult(round=plan.round, query_results=query_results)

    def _default_provider(self) -> SearchProvider:
        tavily = TavilySearchProvider()
        if tavily.enabled:
            return tavily
        return MockSearchProvider()
