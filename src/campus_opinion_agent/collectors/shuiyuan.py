"""Shuiyuan Community collector.

This module is adapted from kuan-er/sjtu-agent's Shuiyuan-related logic. It
uses Shuiyuan's Discourse JSON API instead of scraping rendered HTML pages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import html
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests


CST = timezone(timedelta(hours=8))
SHUIYUAN_BASE_URL = "https://shuiyuan.sjtu.edu.cn"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


@dataclass
class ShuiyuanConfig:
    """Authentication config for Shuiyuan.

    Prefer Discourse User API credentials. Cookie fallback is supported for
    local experiments.
    """

    user_api_key: str = ""
    user_api_client_id: str = ""
    cookies: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "ShuiyuanConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(
            user_api_key=str(data.get("shuiyuan_user_api_key") or ""),
            user_api_client_id=str(data.get("shuiyuan_user_api_client_id") or ""),
            cookies=data.get("shuiyuan_cookies") or {},
        )

    @property
    def has_auth(self) -> bool:
        return bool((self.user_api_key and self.user_api_client_id) or self.cookies)


@dataclass
class ShuiyuanTopic:
    topic_id: int
    title: str
    url: str
    category_id: int | None = None
    created_at: str | None = None
    last_posted_at: str | None = None
    last_poster_username: str = ""
    views: int = 0
    like_count: int = 0
    posts_count: int = 0
    excerpt: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShuiyuanPost:
    post_id: int
    topic_id: int
    post_number: int
    username: str
    created_at: str
    content: str
    like_count: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ShuiyuanCollector:
    """Collector for Shuiyuan topics and posts."""

    def __init__(self, config: ShuiyuanConfig | None = None, timeout: int = 20) -> None:
        self.config = config or ShuiyuanConfig()
        self.timeout = timeout

    def fetch_hot_topics(self, period: str = "daily", hours: int = 24) -> list[ShuiyuanTopic]:
        """Fetch top topics from Shuiyuan's Discourse top endpoint."""
        data = self._get_json(f"{SHUIYUAN_BASE_URL}/top.json", params={"period": period})
        topics = data.get("topic_list", {}).get("topics", [])
        return self._parse_topic_list(topics, hours=hours)

    def fetch_latest_topics(self, hours: int = 24) -> list[ShuiyuanTopic]:
        """Fetch latest created topics."""
        data = self._get_json(f"{SHUIYUAN_BASE_URL}/latest.json", params={"order": "created"})
        topics = data.get("topic_list", {}).get("topics", [])
        return self._parse_topic_list(topics, hours=hours)

    def fetch_monitor_candidates(
        self,
        hours: int = 24,
        min_views: int = 50,
        min_likes: int = 3,
        min_posts: int = 5,
    ) -> list[ShuiyuanTopic]:
        """Fetch and merge hot/latest topics for monitoring."""
        by_id: dict[int, ShuiyuanTopic] = {}
        for topic in [*self.fetch_hot_topics(hours=hours), *self.fetch_latest_topics(hours=hours)]:
            if topic.views < min_views and topic.like_count < min_likes and topic.posts_count < min_posts:
                continue
            by_id[topic.topic_id] = topic
        return sorted(by_id.values(), key=lambda item: (item.posts_count, item.views), reverse=True)

    def search_topics(self, query: str, max_results: int = 10) -> list[ShuiyuanTopic]:
        """Search Shuiyuan topics via Discourse search.json."""
        if not query.strip():
            return []
        data = self._get_json(
            f"{SHUIYUAN_BASE_URL}/search.json",
            params={"q": query.strip(), "page": 1},
            require_auth=True,
        )
        topics = data.get("topics") or []
        posts = data.get("posts") or []
        post_map = {post.get("topic_id"): post for post in posts}
        results: list[ShuiyuanTopic] = []
        for topic in topics[:max_results]:
            topic_id = int(topic.get("id") or 0)
            if not topic_id:
                continue
            slug = topic.get("slug") or str(topic_id)
            post = post_map.get(topic_id, {})
            results.append(
                ShuiyuanTopic(
                    topic_id=topic_id,
                    title=topic.get("fancy_title") or topic.get("title") or "",
                    url=f"{SHUIYUAN_BASE_URL}/t/{slug}/{topic_id}",
                    category_id=topic.get("category_id"),
                    views=int(topic.get("views") or 0),
                    posts_count=int(topic.get("posts_count") or 0),
                    excerpt=self._html_to_text(post.get("blurb") or ""),
                    raw=topic,
                )
            )
        return results

    def read_topic(
        self,
        topic: str | int,
        max_posts: int = 100,
        batch_size: int = 20,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[ShuiyuanTopic, list[ShuiyuanPost]]:
        """Read a topic and up to max_posts posts.

        `topic` may be a numeric topic id or a Shuiyuan topic URL.
        """
        topic_id = self._extract_topic_id(topic)
        data = self._get_json(
            f"{SHUIYUAN_BASE_URL}/t/{topic_id}.json",
            params={"include_raw": "false"},
            require_auth=True,
        )

        slug = data.get("slug") or "topic"
        topic_obj = ShuiyuanTopic(
            topic_id=topic_id,
            title=data.get("fancy_title") or data.get("title") or "",
            url=f"{SHUIYUAN_BASE_URL}/t/{slug}/{topic_id}",
            category_id=data.get("category_id"),
            created_at=data.get("created_at"),
            last_posted_at=data.get("last_posted_at"),
            views=int(data.get("views") or 0),
            like_count=int(data.get("like_count") or 0),
            posts_count=int(data.get("posts_count") or 0),
            raw=data,
        )

        post_stream = data.get("post_stream") or {}
        initial_posts = post_stream.get("posts") or []
        stream_ids = post_stream.get("stream") or []
        by_id: dict[int, dict[str, Any]] = {
            int(post["id"]): post for post in initial_posts if post.get("id") is not None
        }

        target = max(1, int(max_posts))
        if progress_callback:
            progress_callback(min(len(by_id), target), target)
        if target > len(initial_posts) and stream_ids:
            need_ids = [int(post_id) for post_id in stream_ids if int(post_id) not in by_id]
            need_ids = need_ids[: max(0, target - len(initial_posts))]
            batch_size = max(1, min(int(batch_size), 200))
            for index in range(0, len(need_ids), batch_size):
                chunk = need_ids[index:index + batch_size]
                params = [("post_ids[]", str(post_id)) for post_id in chunk]
                more_data = self._get_json(
                    f"{SHUIYUAN_BASE_URL}/t/{topic_id}/posts.json",
                    params=params,
                    require_auth=True,
                )
                for post in (more_data.get("post_stream") or {}).get("posts") or []:
                    if post.get("id") is not None:
                        by_id[int(post["id"])] = post
                if progress_callback:
                    progress_callback(min(len(by_id), target), target)

        ordered: list[dict[str, Any]] = []
        for post_id in stream_ids:
            post = by_id.get(int(post_id))
            if post:
                ordered.append(post)
            if len(ordered) >= target:
                break
        if not ordered:
            ordered = initial_posts[:target]

        posts = [self._parse_post(topic_id, post) for post in ordered]
        return topic_obj, posts

    def _get_json(
        self,
        url: str,
        params: dict[str, Any] | list[tuple[str, str]] | None = None,
        require_auth: bool = False,
        max_retry: int = 3,
    ) -> dict[str, Any]:
        if require_auth and not self.config.has_auth:
            raise RuntimeError("Shuiyuan auth is missing. Configure user API key or cookies first.")

        headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
        cookies = None
        if self.config.user_api_key and self.config.user_api_client_id:
            headers["User-Api-Key"] = self.config.user_api_key
            headers["User-Api-Client-Id"] = self.config.user_api_client_id
        elif self.config.cookies:
            cookies = self.config.cookies

        last_error: Exception | None = None
        for attempt in range(max_retry):
            try:
                response = requests.get(url, params=params, headers=headers, cookies=cookies, timeout=self.timeout)
                if response.status_code == 429:
                    time.sleep(30 * (attempt + 1))
                    continue
                if response.status_code in (401, 403) or "login" in response.url:
                    raise RuntimeError("Shuiyuan auth expired or permission denied.")
                if response.status_code == 404:
                    raise RuntimeError(f"Shuiyuan resource not found: {url}")
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"Shuiyuan request failed: {last_error}")

    def _parse_topic_list(self, topics: list[dict[str, Any]], hours: int) -> list[ShuiyuanTopic]:
        parsed: list[ShuiyuanTopic] = []
        for topic in topics:
            topic_id = int(topic.get("id") or 0)
            title = str(topic.get("title") or "").strip()
            if not topic_id or not title:
                continue
            created_at = self._parse_iso(topic.get("created_at") or "")
            if created_at and self._age_hours(created_at) > hours:
                continue
            slug = topic.get("slug") or str(topic_id)
            parsed.append(
                ShuiyuanTopic(
                    topic_id=topic_id,
                    title=title,
                    url=f"{SHUIYUAN_BASE_URL}/t/{slug}/{topic_id}",
                    category_id=topic.get("category_id"),
                    created_at=topic.get("created_at"),
                    last_posted_at=topic.get("last_posted_at"),
                    last_poster_username=topic.get("last_poster_username") or "",
                    views=int(topic.get("views") or 0),
                    like_count=int(topic.get("like_count") or 0),
                    posts_count=int(topic.get("posts_count") or 0),
                    excerpt=self._html_to_text(topic.get("excerpt") or ""),
                    raw=topic,
                )
            )
        return parsed

    def _parse_post(self, topic_id: int, post: dict[str, Any]) -> ShuiyuanPost:
        return ShuiyuanPost(
            post_id=int(post.get("id") or 0),
            topic_id=topic_id,
            post_number=int(post.get("post_number") or 0),
            username=post.get("username") or "",
            created_at=post.get("created_at") or "",
            like_count=self._post_like_count(post),
            content=self._html_to_text(post.get("cooked") or ""),
            raw=post,
        )

    def _extract_topic_id(self, topic: str | int) -> int:
        raw = str(topic).strip()
        if raw.isdigit():
            return int(raw)
        match = re.search(r"/t(?:/[^/]+)?/(\d+)", raw)
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot extract Shuiyuan topic id from: {topic}")

    def _parse_iso(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(CST)
        except ValueError:
            return None

    def _age_hours(self, value: datetime) -> float:
        return (datetime.now(CST) - value).total_seconds() / 3600

    def _html_to_text(self, value: str) -> str:
        text = re.sub(r"(?is)<script[^>]*>.*?</script>", "", value)
        text = re.sub(r"(?is)<style[^>]*>.*?</style>", "", text)
        text = re.sub(r"(?is)<aside[^>]*class=\"quote\"[^>]*>.*?</aside>", "", text)
        text = re.sub(r"(?is)<br\s*/?>", "\n", text)
        text = re.sub(r"(?is)</p\s*>", "\n", text)
        text = re.sub(r"(?is)<[^>]+>", "", text)
        text = html.unescape(text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _post_like_count(self, post: dict[str, Any]) -> int | None:
        actions = post.get("actions_summary") or []
        if not actions:
            return None
        first = actions[0] if isinstance(actions[0], dict) else {}
        count = first.get("count")
        return int(count) if count is not None else None
