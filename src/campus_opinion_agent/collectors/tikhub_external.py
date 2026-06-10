"""TikHub-backed external social media collectors.

The collector normalizes TikHub's API responses into JSONL-friendly rows for
external social media evidence. Supported platforms currently include Zhihu,
Xiaohongshu, and Weibo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

from tikhub import TikHub, TikHubError


ZHihu_ANSWER_URL_RE = re.compile(r"zhihu\.com/question/(\d+)/answer/(\d+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class TikHubExternalConfig:
    """Authentication and output config for TikHub external collectors."""

    api_key: str = ""
    output_dir: str = "data/external/tikhub"

    @classmethod
    def from_json(cls, path: str | Path) -> "TikHubExternalConfig":
        config_path = Path(path)
        data: dict[str, Any] = {}
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))

        tikhub_config = data.get("tikhub") or {}
        return cls(
            api_key=str(tikhub_config.get("api_key") or data.get("tikhub_api_key") or ""),
            output_dir=str(tikhub_config.get("output_dir") or "data/external/tikhub"),
        )

    @property
    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get("TIKHUB_API_KEY", "")


@dataclass
class ZhihuContent:
    content_id: str
    content_type: str
    content_text: str
    content_url: str
    question_id: str = ""
    title: str = ""
    desc: str = ""
    created_time: int | None = None
    updated_time: int | None = None
    voteup_count: int = 0
    comment_count: int = 0
    source_keyword: str = ""
    user_id: str = ""
    user_link: str = ""
    user_nickname: str = ""
    user_avatar: str = ""
    user_url_token: str = ""
    captured_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZhihuComment:
    comment_id: str
    parent_comment_id: str
    content: str
    publish_time: int | None = None
    ip_location: str = ""
    sub_comment_count: int = 0
    like_count: int = 0
    dislike_count: int = 0
    content_id: str = ""
    content_type: str = "answer"
    user_id: str = ""
    user_link: str = ""
    user_nickname: str = ""
    user_avatar: str = ""
    captured_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class XiaohongshuNote:
    note_id: str
    note_type: str
    title: str
    desc: str
    note_url: str
    liked_count: int = 0
    collected_count: int = 0
    comments_count: int = 0
    shared_count: int = 0
    timestamp: int | None = None
    source_keyword: str = ""
    user_id: str = ""
    user_nickname: str = ""
    user_red_id: str = ""
    user_avatar: str = ""
    xsec_token: str = ""
    captured_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class XiaohongshuComment:
    comment_id: str
    parent_comment_id: str
    note_id: str
    content: str
    publish_time: int | None = None
    like_count: int = 0
    sub_comment_count: int = 0
    user_id: str = ""
    user_nickname: str = ""
    user_red_id: str = ""
    user_avatar: str = ""
    captured_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeiboPost:
    post_id: str
    content: str
    post_url: str
    publish_time: str = ""
    source: str = ""
    weibo_type: str = ""
    repost_count: int = 0
    comment_count: int = 0
    like_count: int = 0
    source_keyword: str = ""
    user_id: str = ""
    user_name: str = ""
    user_url: str = ""
    user_avatar: str = ""
    captured_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeiboComment:
    comment_id: str
    parent_comment_id: str
    post_id: str
    content: str
    publish_time: str = ""
    source: str = ""
    like_count: int = 0
    user_id: str = ""
    user_name: str = ""
    user_url: str = ""
    user_avatar: str = ""
    captured_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class TikhubExternalCollector:
    """External platform collector backed by TikHub API."""

    def __init__(self, config: TikHubExternalConfig | None = None, timeout: int = 30) -> None:
        self.config = config or TikHubExternalConfig()
        self.timeout = timeout

    def search_zhihu(
        self,
        keyword: str,
        max_results: int = 10,
        poll_attempts: int = 5,
        poll_interval: float = 2.0,
        raw_payloads: list[Any] | None = None,
    ) -> list[ZhihuContent]:
        """Search Zhihu through TikHub and normalize content rows.

        TikHub's Zhihu answer-like search is exposed as AI search. Some
        responses include results immediately; others return a message id that
        needs polling. This method handles both shapes.
        """
        keyword = keyword.strip()
        if not keyword:
            return []

        with self._client() as client:
            raw = client.zhihu_web.fetch_ai_search(message_content=keyword)
            payloads = [raw]
            if raw_payloads is not None:
                raw_payloads.append(raw)
            message_id = _find_zhihu_ai_result_message_id(raw)
            contents = self._normalize_zhihu_contents(raw, keyword=keyword)

            for _ in range(max(0, poll_attempts)):
                if contents or not message_id:
                    break
                time.sleep(max(0, poll_interval))
                try:
                    result = client.zhihu_web.fetch_ai_search_result(message_id=str(message_id))
                except TikHubError:
                    break
                payloads.append(result)
                if raw_payloads is not None:
                    raw_payloads.append(result)
                contents = self._normalize_zhihu_contents(result, keyword=keyword)

        if not contents:
            contents = self._normalize_zhihu_contents({"responses": payloads}, keyword=keyword)
        return contents[: max(0, max_results)]

    def fetch_zhihu_comments(
        self,
        answer_id: str,
        max_comments: int = 100,
        include_replies: bool = False,
        order_by: str | None = None,
    ) -> list[ZhihuComment]:
        """Fetch comments for one Zhihu answer id."""
        answer_id = str(answer_id).strip()
        if not answer_id:
            return []

        limit = min(max(max_comments, 1), 100)
        with self._client() as client:
            raw = client.zhihu_web.fetch_comment_v5(
                answer_id=answer_id,
                order_by=order_by,
                limit=str(limit),
                offset="0",
            )
            comments = self._normalize_zhihu_comments(raw, answer_id=answer_id)
            if include_replies:
                comments.extend(self._fetch_zhihu_comment_replies(client, comments, max_comments, order_by))
        return comments[: max(0, max_comments)]

    def fetch_zhihu_comments_for_contents(
        self,
        contents: Iterable[ZhihuContent],
        max_comments_per_content: int = 50,
        include_replies: bool = False,
    ) -> list[ZhihuComment]:
        """Fetch comments for normalized Zhihu answer rows."""
        rows: list[ZhihuComment] = []
        for content in contents:
            if content.content_type != "answer" or not content.content_id:
                continue
            rows.extend(
                self.fetch_zhihu_comments(
                    answer_id=content.content_id,
                    max_comments=max_comments_per_content,
                    include_replies=include_replies,
                )
            )
        return rows

    def search_xiaohongshu(
        self,
        keyword: str,
        max_results: int = 20,
        page: int = 1,
        raw_payloads: list[Any] | None = None,
    ) -> list[XiaohongshuNote]:
        """Search Xiaohongshu notes through TikHub App V2 API."""
        keyword = keyword.strip()
        if not keyword:
            return []

        with self._client() as client:
            raw = client.xiaohongshu_app_v2.search_notes(keyword=keyword, page=page)
            if raw_payloads is not None:
                raw_payloads.append(raw)
        notes = self._normalize_xiaohongshu_notes(raw, keyword=keyword)
        return notes[: max(0, max_results)]

    def fetch_xiaohongshu_comments(
        self,
        note_id: str,
        max_comments: int = 100,
        include_replies: bool = False,
        cursor: str | None = None,
        pages: int = 1,
        raw_payloads: list[Any] | None = None,
    ) -> list[XiaohongshuComment]:
        """Fetch Xiaohongshu comments for one note id."""
        note_id = note_id.strip()
        if not note_id:
            return []

        rows: list[XiaohongshuComment] = []
        seen: set[str] = set()
        next_cursor = cursor or None
        with self._client() as client:
            for _ in range(max(1, pages)):
                raw = client.xiaohongshu_app_v2.get_note_comments(note_id=note_id, cursor=next_cursor)
                if raw_payloads is not None:
                    raw_payloads.append(raw)
                for comment in self._normalize_xiaohongshu_comments(
                    raw,
                    note_id=note_id,
                    include_replies=include_replies,
                ):
                    if comment.comment_id in seen:
                        continue
                    seen.add(comment.comment_id)
                    rows.append(comment)
                    if len(rows) >= max_comments:
                        return rows[: max(0, max_comments)]
                next_cursor = _find_xiaohongshu_comment_cursor(raw)
                if not next_cursor:
                    break
        return rows[: max(0, max_comments)]

    def fetch_xiaohongshu_comments_for_notes(
        self,
        notes: Iterable[XiaohongshuNote],
        max_comments_per_note: int = 50,
        include_replies: bool = False,
        pages_per_note: int = 1,
    ) -> list[XiaohongshuComment]:
        """Fetch comments for normalized Xiaohongshu note rows."""
        rows: list[XiaohongshuComment] = []
        for note in notes:
            if not note.note_id:
                continue
            rows.extend(
                self.fetch_xiaohongshu_comments(
                    note_id=note.note_id,
                    max_comments=max_comments_per_note,
                    include_replies=include_replies,
                    pages=pages_per_note,
                )
            )
        return rows

    def search_weibo(
        self,
        keyword: str,
        max_results: int = 20,
        page: int = 1,
        raw_payloads: list[Any] | None = None,
    ) -> list[WeiboPost]:
        """Search Weibo posts through TikHub Web V2 advanced search."""
        keyword = keyword.strip()
        if not keyword:
            return []

        with self._client() as client:
            raw = client.weibo_web_v2.fetch_advanced_search(q=keyword, page=page)
            if raw_payloads is not None:
                raw_payloads.append(raw)
        posts = self._normalize_weibo_posts(raw, keyword=keyword)
        return posts[: max(0, max_results)]

    def fetch_weibo_comments(
        self,
        post_id: str,
        max_comments: int = 100,
        include_replies: bool = False,
        raw_payloads: list[Any] | None = None,
    ) -> list[WeiboComment]:
        """Fetch Weibo comments for one post id."""
        post_id = post_id.strip()
        if not post_id:
            return []

        with self._client() as client:
            raw = client.weibo_web_v2.fetch_post_comments(id=post_id, count=min(max_comments, 50))
            if raw_payloads is not None:
                raw_payloads.append(raw)
        comments = self._normalize_weibo_comments(raw, post_id=post_id, include_replies=include_replies)
        return comments[: max(0, max_comments)]

    def fetch_weibo_comments_for_posts(
        self,
        posts: Iterable[WeiboPost],
        max_comments_per_post: int = 50,
        include_replies: bool = False,
    ) -> list[WeiboComment]:
        """Fetch comments for normalized Weibo post rows."""
        rows: list[WeiboComment] = []
        for post in posts:
            if not post.post_id:
                continue
            rows.extend(
                self.fetch_weibo_comments(
                    post_id=post.post_id,
                    max_comments=max_comments_per_post,
                    include_replies=include_replies,
                )
            )
        return rows

    def _fetch_zhihu_comment_replies(
        self,
        client: TikHub,
        comments: list[ZhihuComment],
        max_comments: int,
        order_by: str | None,
    ) -> list[ZhihuComment]:
        replies: list[ZhihuComment] = []
        remaining = max(0, max_comments - len(comments))
        if remaining <= 0:
            return replies

        for comment in comments:
            if remaining <= 0 or not comment.comment_id or comment.sub_comment_count <= 0:
                continue
            raw = client.zhihu_web.fetch_sub_comment_v5(
                comment_id=comment.comment_id,
                order_by=order_by,
                limit=str(min(remaining, 50)),
                offset="0",
            )
            normalized = self._normalize_zhihu_comments(
                raw,
                answer_id=comment.content_id,
                parent_comment_id=comment.comment_id,
            )
            replies.extend(normalized)
            remaining -= len(normalized)
        return replies

    def _client(self) -> TikHub:
        api_key = self.config.resolved_api_key
        if not api_key:
            raise RuntimeError("TikHub API key is missing. Set TIKHUB_API_KEY or config.json tikhub.api_key.")
        return TikHub(api_key=api_key, timeout=self.timeout)

    def _normalize_zhihu_contents(self, raw: Any, keyword: str) -> list[ZhihuContent]:
        captured_at = _now_id()
        rows: list[ZhihuContent] = []
        seen: set[str] = set()

        for item in _iter_dicts(raw):
            normalized = _normalize_content_item(item, keyword=keyword, captured_at=captured_at)
            if not normalized:
                continue
            key = normalized.content_id or normalized.content_url or normalized.content_text[:80]
            if key in seen:
                continue
            seen.add(key)
            rows.append(normalized)
        return rows

    def _normalize_zhihu_comments(
        self,
        raw: Any,
        answer_id: str,
        parent_comment_id: str = "0",
    ) -> list[ZhihuComment]:
        captured_at = _now_id()
        rows: list[ZhihuComment] = []
        seen: set[str] = set()

        for item in _iter_dicts(raw):
            normalized = _normalize_comment_item(
                item,
                answer_id=answer_id,
                parent_comment_id=parent_comment_id,
                captured_at=captured_at,
            )
            if not normalized or normalized.comment_id in seen:
                continue
            seen.add(normalized.comment_id)
            rows.append(normalized)
        return rows

    def _normalize_xiaohongshu_notes(self, raw: Any, keyword: str) -> list[XiaohongshuNote]:
        captured_at = _now_id()
        rows: list[XiaohongshuNote] = []
        seen: set[str] = set()

        for item in _iter_dicts(raw):
            note = item.get("note") if isinstance(item.get("note"), dict) else item
            normalized = _normalize_xiaohongshu_note_item(note, keyword=keyword, captured_at=captured_at)
            if not normalized or normalized.note_id in seen:
                continue
            seen.add(normalized.note_id)
            rows.append(normalized)
        return rows

    def _normalize_xiaohongshu_comments(
        self,
        raw: Any,
        note_id: str,
        include_replies: bool = False,
    ) -> list[XiaohongshuComment]:
        captured_at = _now_id()
        rows: list[XiaohongshuComment] = []
        seen: set[str] = set()
        comments = _find_list_by_key(raw, "comments")

        for comment in comments:
            normalized = _normalize_xiaohongshu_comment_item(
                comment,
                note_id=note_id,
                parent_comment_id="0",
                captured_at=captured_at,
            )
            if normalized and normalized.comment_id not in seen:
                seen.add(normalized.comment_id)
                rows.append(normalized)

            if include_replies:
                for reply in comment.get("sub_comments") or []:
                    reply_row = _normalize_xiaohongshu_comment_item(
                        reply,
                        note_id=note_id,
                        parent_comment_id=str(comment.get("id") or ""),
                        captured_at=captured_at,
                    )
                    if reply_row and reply_row.comment_id not in seen:
                        seen.add(reply_row.comment_id)
                        rows.append(reply_row)
        return rows

    def _normalize_weibo_posts(self, raw: Any, keyword: str) -> list[WeiboPost]:
        captured_at = _now_id()
        rows: list[WeiboPost] = []
        seen: set[str] = set()
        results = _find_path(raw, ["data", "parsed_data", "results"])
        if not isinstance(results, list):
            results = []

        for item in results:
            normalized = _normalize_weibo_post_item(item, keyword=keyword, captured_at=captured_at)
            if not normalized or normalized.post_id in seen:
                continue
            seen.add(normalized.post_id)
            rows.append(normalized)
        return rows

    def _normalize_weibo_comments(
        self,
        raw: Any,
        post_id: str,
        include_replies: bool = False,
    ) -> list[WeiboComment]:
        captured_at = _now_id()
        rows: list[WeiboComment] = []
        seen: set[str] = set()
        comments = _find_list_by_key(raw, "data")

        for comment in comments:
            normalized = _normalize_weibo_comment_item(
                comment,
                post_id=post_id,
                parent_comment_id="0",
                captured_at=captured_at,
            )
            if normalized and normalized.comment_id not in seen:
                seen.add(normalized.comment_id)
                rows.append(normalized)

            if include_replies:
                for reply in comment.get("comments") or []:
                    reply_row = _normalize_weibo_comment_item(
                        reply,
                        post_id=post_id,
                        parent_comment_id=str(comment.get("idstr") or comment.get("id") or ""),
                        captured_at=captured_at,
                    )
                    if reply_row and reply_row.comment_id not in seen:
                        seen.add(reply_row.comment_id)
                        rows.append(reply_row)
        return rows


def _normalize_content_item(item: dict[str, Any], keyword: str, captured_at: str) -> ZhihuContent | None:
    obj = _first_dict(item, "answer", "target", "object", "content") or item
    question = _first_dict(obj, "question") or _first_dict(item, "question") or {}
    author = _first_dict(obj, "author", "member", "user") or _first_dict(item, "author", "member", "user") or {}

    url = _first_str(obj, "content_url", "url", "link", "share_url") or _first_str(item, "url", "link")
    question_id = str(
        _first_value(obj, "question_id")
        or _first_value(question, "id", "question_id")
        or ""
    )
    content_id = str(_first_value(obj, "content_id", "answer_id", "id") or "")
    if url:
        match = ZHihu_ANSWER_URL_RE.search(url)
        if match:
            question_id = question_id or match.group(1)
            content_id = content_id or match.group(2)

    content_type = str(_first_value(obj, "content_type", "type") or "")
    if _looks_like_author_only(obj):
        return None
    if obj.get("message_type") == "ai_tab_summary":
        return None

    public_content_id = str(_first_value(obj, "content_token") or content_id)
    if content_type == "answer" and public_content_id:
        content_id = public_content_id
    if content_id and (question_id or "answer" in url or content_type == "answer"):
        content_type = "answer"
    elif "article" in content_type:
        content_type = "article"
    elif not content_type:
        content_type = "search_result"
    elif content_type not in {"answer", "article", "video", "search_result"}:
        return None

    title = _clean_text(
        _first_str(obj, "title", "question_title")
        or _first_str(question, "title", "name")
        or _first_str(item, "title")
    )
    text = _clean_text(
        _first_str(obj, "content_text", "content", "abstract", "excerpt", "description", "desc")
        or _first_str(item, "content", "abstract", "excerpt")
    )
    desc = _clean_text(_first_str(obj, "desc", "description") or _first_str(question, "excerpt", "desc") or "")

    if not (content_id or url or title or text):
        return None
    if not (url or text):
        return None
    if content_type == "search_result" and not text:
        return None
    if not (url or question_id or content_id or "zhihu" in json.dumps(item, ensure_ascii=False)[:2000]):
        return None
    if not url and question_id and content_id and content_type == "answer":
        url = f"https://www.zhihu.com/question/{question_id}/answer/{content_id}"

    return ZhihuContent(
        content_id=content_id,
        content_type=content_type,
        content_text=text,
        content_url=url,
        question_id=question_id,
        title=title,
        desc=desc,
        created_time=_to_int(_first_value(obj, "created_time", "created", "created_at")),
        updated_time=_to_int(_first_value(obj, "updated_time", "updated", "updated_at")),
        voteup_count=_to_int(_first_value(obj, "voteup_count", "vote_count", "like_count")) or _parse_metric_count(obj, "赞同") or 0,
        comment_count=_to_int(_first_value(obj, "comment_count", "comments_count")) or _parse_metric_count(obj, "评论") or 0,
        source_keyword=keyword,
        user_id=str(_first_value(author, "id", "user_id") or ""),
        user_link=_first_str(author, "url", "user_link", "link"),
        user_nickname=_first_str(author, "name", "nickname", "user_nickname"),
        user_avatar=_first_str(author, "avatar_url", "avatar", "user_avatar"),
        user_url_token=_first_str(author, "url_token", "user_url_token"),
        captured_at=captured_at,
        raw=item,
    )


def _normalize_comment_item(
    item: dict[str, Any],
    answer_id: str,
    parent_comment_id: str,
    captured_at: str,
) -> ZhihuComment | None:
    obj = _first_dict(item, "comment", "target", "object") or item
    author = _first_dict(obj, "author", "member", "user") or _first_dict(item, "author", "member", "user") or {}
    comment_id = str(_first_value(obj, "comment_id", "id") or "")
    content = _clean_text(_first_str(obj, "content", "content_text", "text") or "")

    if not comment_id or not content:
        return None

    return ZhihuComment(
        comment_id=comment_id,
        parent_comment_id=str(_first_value(obj, "parent_comment_id", "reply_to_comment_id") or parent_comment_id),
        content=content,
        publish_time=_to_int(_first_value(obj, "publish_time", "created_time", "created_at")),
        ip_location=_first_str(obj, "ip_location", "location"),
        sub_comment_count=_to_int(_first_value(obj, "sub_comment_count", "child_comment_count", "reply_count")) or 0,
        like_count=_to_int(_first_value(obj, "like_count", "vote_count")) or 0,
        dislike_count=_to_int(_first_value(obj, "dislike_count")) or 0,
        content_id=answer_id,
        user_id=str(_first_value(author, "id", "user_id") or ""),
        user_link=_first_str(author, "url", "user_link", "link"),
        user_nickname=_first_str(author, "name", "nickname", "user_nickname"),
        user_avatar=_first_str(author, "avatar_url", "avatar", "user_avatar"),
        captured_at=captured_at,
        raw=item,
    )


def _normalize_xiaohongshu_note_item(
    item: dict[str, Any],
    keyword: str,
    captured_at: str,
) -> XiaohongshuNote | None:
    if not isinstance(item, dict):
        return None
    note_id = str(_first_value(item, "note_id", "id") or "")
    title = _clean_text(_first_str(item, "title"))
    desc = _clean_text(_first_str(item, "desc", "description", "content"))
    if not note_id or not (title or desc):
        return None

    user = _first_dict(item, "user", "user_info", "author") or {}
    xsec_token = _first_str(item, "xsec_token")
    note_url = f"https://www.xiaohongshu.com/explore/{note_id}"
    if xsec_token:
        note_url = f"{note_url}?xsec_token={xsec_token}&xsec_source=pc_search"

    return XiaohongshuNote(
        note_id=note_id,
        note_type=_first_str(item, "type", "note_type"),
        title=title,
        desc=desc,
        note_url=note_url,
        liked_count=_to_int(_first_value(item, "liked_count", "like_count")) or 0,
        collected_count=_to_int(_first_value(item, "collected_count", "collect_count")) or 0,
        comments_count=_to_int(_first_value(item, "comments_count", "comment_count")) or 0,
        shared_count=_to_int(_first_value(item, "shared_count", "share_count")) or 0,
        timestamp=_to_int(_first_value(item, "timestamp", "time", "last_update_time", "update_time")),
        source_keyword=keyword,
        user_id=str(_first_value(user, "userid", "user_id", "id") or ""),
        user_nickname=_first_str(user, "nickname", "name"),
        user_red_id=_first_str(user, "red_id"),
        user_avatar=_first_str(user, "images", "avatar", "avatar_url"),
        xsec_token=xsec_token,
        captured_at=captured_at,
        raw=item,
    )


def _normalize_xiaohongshu_comment_item(
    item: dict[str, Any],
    note_id: str,
    parent_comment_id: str,
    captured_at: str,
) -> XiaohongshuComment | None:
    if not isinstance(item, dict):
        return None
    comment_id = str(_first_value(item, "comment_id", "id") or "")
    content = _clean_text(_first_str(item, "content", "text"))
    if not comment_id or not content:
        return None

    user = _first_dict(item, "user", "user_info", "author") or {}
    return XiaohongshuComment(
        comment_id=comment_id,
        parent_comment_id=parent_comment_id,
        note_id=str(_first_value(item, "note_id") or note_id),
        content=content,
        publish_time=_to_int(_first_value(item, "time", "publish_time", "created_time")),
        like_count=_to_int(_first_value(item, "like_count", "liked_count")) or 0,
        sub_comment_count=_to_int(_first_value(item, "sub_comment_count", "sub_comment_num")) or 0,
        user_id=str(_first_value(user, "userid", "user_id", "id") or ""),
        user_nickname=_first_str(user, "nickname", "name"),
        user_red_id=_first_str(user, "red_id"),
        user_avatar=_first_str(user, "images", "avatar", "avatar_url"),
        captured_at=captured_at,
        raw=item,
    )


def _normalize_weibo_post_item(
    item: dict[str, Any],
    keyword: str,
    captured_at: str,
) -> WeiboPost | None:
    if not isinstance(item, dict):
        return None
    post_id = str(_first_value(item, "weibo_id", "id", "mid") or "")
    content = _clean_text(_first_str(item, "content", "text", "text_raw"))
    if not post_id or not content:
        return None

    interaction = _first_dict(item, "interaction") or {}
    post_url = _first_str(item, "post_url", "url")
    if post_url.startswith("//"):
        post_url = f"https:{post_url}"

    user_url = _first_str(item, "user_url")
    if user_url.startswith("//"):
        user_url = f"https:{user_url}"

    return WeiboPost(
        post_id=post_id,
        content=content,
        post_url=post_url,
        publish_time=_first_str(item, "publish_time", "created_at"),
        source=_first_str(item, "source"),
        weibo_type=_first_str(item, "weibo_type", "type"),
        repost_count=_to_int(_first_value(interaction, "repost_count", "reposts_count")) or 0,
        comment_count=_to_int(_first_value(interaction, "comment_count", "comments_count")) or 0,
        like_count=_to_int(_first_value(interaction, "like_count", "attitudes_count")) or 0,
        source_keyword=keyword,
        user_id=_extract_weibo_uid(_first_str(item, "user_url")),
        user_name=_first_str(item, "user_name", "user_nick", "screen_name"),
        user_url=user_url,
        user_avatar=_first_str(item, "user_avatar", "profile_image_url"),
        captured_at=captured_at,
        raw=item,
    )


def _normalize_weibo_comment_item(
    item: dict[str, Any],
    post_id: str,
    parent_comment_id: str,
    captured_at: str,
) -> WeiboComment | None:
    if not isinstance(item, dict):
        return None
    comment_id = str(_first_value(item, "idstr", "comment_id", "id", "mid") or "")
    content = _clean_text(_first_str(item, "text", "content", "text_raw"))
    if not comment_id or not content:
        return None

    user = _first_dict(item, "user") or {}
    user_url = _first_str(user, "profile_url", "url")
    if user_url.startswith("/"):
        user_url = f"https://weibo.com{user_url}"

    return WeiboComment(
        comment_id=comment_id,
        parent_comment_id=parent_comment_id,
        post_id=post_id,
        content=content,
        publish_time=_first_str(item, "created_at", "publish_time"),
        source=_first_str(item, "source"),
        like_count=_to_int(_first_value(item, "like_count")) or 0,
        user_id=str(_first_value(user, "idstr", "id", "uid") or ""),
        user_name=_first_str(user, "screen_name", "name"),
        user_url=user_url,
        user_avatar=_first_str(user, "profile_image_url", "avatar_large", "avatar_hd"),
        captured_at=captured_at,
        raw=item,
    )


def _looks_like_author_only(item: dict[str, Any]) -> bool:
    author_keys = {"avatar_url", "url_token", "is_anonymous", "is_followed", "author_metrics"}
    content_keys = {"content", "abstract", "excerpt", "summary", "content_type", "content_token"}
    return bool(author_keys.intersection(item)) and not bool(content_keys.intersection(item))


def _parse_metric_count(item: dict[str, Any], label: str) -> int | None:
    metrics = _first_str(item, "content_metrics")
    if not metrics:
        return None
    match = re.search(rf"([\d.]+)\s*(万)?\s*({label})", metrics)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):
        value *= 10000
    return int(value)


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _find_list_by_key(value: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
        for child in value.values():
            found = _find_list_by_key(child, key)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_list_by_key(child, key)
            if found:
                return found
    return []


def _find_path(value: Any, path: list[str]) -> Any:
    current = value
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _extract_weibo_uid(value: str) -> str:
    match = re.search(r"weibo\.com/(\d+)", value or "")
    return match.group(1) if match else ""


def _first_dict(data: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return None


def _first_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _first_str(data: dict[str, Any], *keys: str) -> str:
    value = _first_value(data, *keys)
    if value is None:
        return ""
    return str(value)


def _find_first(value: Any, keys: set[str], prefer_key: str | None = None) -> Any:
    if isinstance(value, dict):
        if prefer_key and value.get(prefer_key):
            return value[prefer_key]
        for key in keys:
            if value.get(key):
                return value[key]
        for child in value.values():
            found = _find_first(child, keys, prefer_key=prefer_key)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_first(child, keys, prefer_key=prefer_key)
            if found:
                return found
    return None


def _find_zhihu_ai_result_message_id(value: Any) -> Any:
    """Return the result message id from Zhihu AI search's initial response."""
    if not isinstance(value, dict):
        return None

    data = value.get("data")
    if isinstance(data, dict):
        recv_message = data.get("recv_message")
        if isinstance(recv_message, dict):
            message_id = recv_message.get("message_id") or recv_message.get("messageId")
            if message_id:
                return message_id

    return _find_first(value, {"message_id", "messageId", "messageID"})


def _find_xiaohongshu_comment_cursor(value: Any) -> str:
    data = _find_path(value, ["data", "data"])
    if isinstance(data, dict):
        cursor = data.get("cursor")
        if cursor:
            return str(cursor)
    cursor = _find_first(value, {"cursor"})
    return str(cursor) if cursor else ""


def _clean_text(value: str) -> str:
    if not value:
        return ""
    text = HTML_TAG_RE.sub(" ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
