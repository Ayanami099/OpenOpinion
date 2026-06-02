from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "spm",
    "from",
    "share_token",
}


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object, length: int = 10) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in TRACKING_PARAMS
    ]
    normalized = parsed._replace(query=urlencode(query), fragment="")
    return urlunparse(normalized)


def domain_from_url(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def text_tokens(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    return [w for w in words if w.strip()]


def lexical_similarity(a: str, b: str) -> float:
    a_tokens = Counter(text_tokens(a))
    b_tokens = Counter(text_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    shared = set(a_tokens) & set(b_tokens)
    dot = sum(a_tokens[t] * b_tokens[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a_tokens.values()))
    norm_b = math.sqrt(sum(v * v for v in b_tokens.values()))
    return clamp(dot / (norm_a * norm_b))


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term and term in text for term in terms)


def short_summary(text: str, max_chars: int = 140) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "..."

