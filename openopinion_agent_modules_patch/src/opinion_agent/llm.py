from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


@dataclass
class OpenAICompatibleClient:
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    timeout: int = 60
    max_retries: int = 2
    retry_backoff: float = 2.0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        return cls(
            api_key=_first_env("LLM_API_KEY", "OPENAI_API_KEY"),
            base_url=_first_env("LLM_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1"),
            model=_first_env("LLM_MODEL", "OPENAI_MODEL", default="gpt-4.1-mini"),
            embedding_api_key=_first_env(
                "EMBEDDING_API_KEY",
                "OPENAI_EMBEDDING_API_KEY",
                "LLM_API_KEY",
                "OPENAI_API_KEY",
            ),
            embedding_base_url=_first_env(
                "EMBEDDING_BASE_URL",
                "OPENAI_EMBEDDING_BASE_URL",
                "LLM_BASE_URL",
                "OPENAI_BASE_URL",
                default="https://api.openai.com/v1",
            ),
            embedding_model=_first_env("EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small"),
            timeout=_env_int("LLM_TIMEOUT", "OPENAI_TIMEOUT", default=60),
            max_retries=_env_int("LLM_MAX_RETRIES", "OPENAI_MAX_RETRIES", default=2),
            retry_backoff=_env_float("LLM_RETRY_BACKOFF", "OPENAI_RETRY_BACKOFF", default=2.0),
        )

    def __post_init__(self) -> None:
        self.base_url = _normalize_base_url(self.base_url)
        self.embedding_api_key = self.embedding_api_key or self.api_key
        self.embedding_base_url = _normalize_base_url(self.embedding_base_url or self.base_url)
        self.timeout = max(1, int(self.timeout))
        self.max_retries = max(0, int(self.max_retries))
        self.retry_backoff = max(0.0, float(self.retry_backoff))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def embedding_enabled(self) -> bool:
        return bool(self.embedding_api_key)

    def chat_json(self, system: str, user: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise LLMError("LLM_API_KEY or OPENAI_API_KEY is not configured")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        try:
            data = self._post_json("/chat/completions", payload, api_key=self.api_key, base_url=self.base_url)
            content = data["choices"][0]["message"]["content"]
            return extract_json_object(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            if fallback is not None:
                return fallback
            raise LLMError(f"Invalid LLM JSON response: {exc}") from exc

    def chat_text(
        self,
        system: str,
        user: str,
        fallback: str | None = None,
        temperature: float = 0.35,
        max_tokens: int | None = None,
    ) -> str:
        if not self.enabled:
            raise LLMError("LLM_API_KEY or OPENAI_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            data = self._post_json("/chat/completions", payload, api_key=self.api_key, base_url=self.base_url)
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            if fallback is not None:
                return fallback
            raise LLMError(f"Invalid LLM text response: {exc}") from exc

    def embedding(self, text: str) -> list[float] | None:
        if not self.embedding_enabled:
            raise LLMError(
                "EMBEDDING_API_KEY, OPENAI_EMBEDDING_API_KEY, LLM_API_KEY, or OPENAI_API_KEY is not configured"
            )
        payload = {"model": self.embedding_model, "input": text[:8000]}
        try:
            data = self._post_json(
                "/embeddings",
                payload,
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
            )
            return data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Invalid embedding response: {exc}") from exc

    def _post_json(self, path: str, payload: dict[str, Any], api_key: str | None, base_url: str | None) -> dict[str, Any]:
        if not api_key or not base_url:
            raise LLMError("API key or base URL is not configured")

        url = f"{base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        total_attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(total_attempts):
            request = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))

            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= total_attempts - 1:
                    raise LLMError(f"HTTP {exc.code} from model endpoint {url}: {exc.reason}") from exc
                self._sleep_before_retry(attempt)

            except urllib.error.URLError as exc:
                last_error = exc
                if attempt >= total_attempts - 1:
                    raise LLMError(f"Model endpoint connection failed for {url}: {exc.reason}") from exc
                self._sleep_before_retry(attempt)

            except (TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= total_attempts - 1:
                    raise LLMError(f"Model endpoint read failed for {url}: {exc}") from exc
                self._sleep_before_retry(attempt)

            except json.JSONDecodeError as exc:
                raise LLMError(f"Model endpoint returned invalid JSON from {url}: {exc}") from exc

        raise LLMError(f"Model endpoint failed for {url}: {last_error}")

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff <= 0:
            return
        delay = self.retry_backoff * (attempt + 1)
        time.sleep(delay)


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _env_int(*names: str, default: int) -> int:
    value = _first_env(*names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(*names: str, default: float) -> float:
    value = _first_env(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _normalize_base_url(base_url: str | None) -> str | None:
    return base_url.rstrip("/") if base_url else None
