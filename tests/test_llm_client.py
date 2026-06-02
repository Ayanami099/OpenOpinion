import urllib.request

import pytest

from opinion_agent.llm import LLMError, OpenAICompatibleClient


def test_from_env_supports_separate_llm_and_embedding_endpoints(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1/")
    monkeypatch.setenv("LLM_MODEL", "chat-model")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embed.example.com/v1/")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-model")

    client = OpenAICompatibleClient.from_env()

    assert client.api_key == "llm-key"
    assert client.base_url == "https://llm.example.com/v1"
    assert client.model == "chat-model"
    assert client.embedding_api_key == "embed-key"
    assert client.embedding_base_url == "https://embed.example.com/v1"
    assert client.embedding_model == "embed-model"


def test_embedding_endpoint_falls_back_to_llm_endpoint_for_backward_compatibility():
    client = OpenAICompatibleClient(
        api_key="shared-key",
        base_url="https://shared.example.com/v1",
        model="chat-model",
        embedding_model="embed-model",
    )

    assert client.embedding_api_key == "shared-key"
    assert client.embedding_base_url == "https://shared.example.com/v1"


def test_chat_and_embedding_use_their_own_credentials():
    calls = []

    class RecordingClient(OpenAICompatibleClient):
        def _post_json(self, path, payload, api_key, base_url):  # type: ignore[override]
            calls.append((path, payload["model"], api_key, base_url))
            if path == "/embeddings":
                return {"data": [{"embedding": [0.1, 0.2]}]}
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    client = RecordingClient(
        api_key="llm-key",
        base_url="https://llm.example.com/v1",
        model="chat-model",
        embedding_api_key="embed-key",
        embedding_base_url="https://embed.example.com/v1",
        embedding_model="embed-model",
    )

    assert client.chat_json("system", "user") == {"ok": True}
    assert client.embedding("hello") == [0.1, 0.2]
    assert calls == [
        ("/chat/completions", "chat-model", "llm-key", "https://llm.example.com/v1"),
        ("/embeddings", "embed-model", "embed-key", "https://embed.example.com/v1"),
    ]


def test_chat_text_returns_plain_text_response():
    class RecordingClient(OpenAICompatibleClient):
        def _post_json(self, path, payload, api_key, base_url):  # type: ignore[override]
            assert path == "/chat/completions"
            assert payload["temperature"] == 0.45
            assert "response_format" not in payload
            return {"choices": [{"message": {"content": "  plain markdown report  "}}]}

    client = RecordingClient(
        api_key="llm-key",
        base_url="https://llm.example.com/v1",
        model="chat-model",
    )

    assert client.chat_text("system", "user", temperature=0.45) == "plain markdown report"


def test_chat_json_does_not_use_fallback_when_model_is_not_configured():
    client = OpenAICompatibleClient(api_key=None, base_url="https://llm.example.com/v1", model="chat-model")

    with pytest.raises(LLMError, match="LLM_API_KEY"):
        client.chat_json("system", "user", fallback={"ok": True})


def test_embedding_missing_configuration_raises_model_error():
    client = OpenAICompatibleClient(api_key=None, embedding_api_key=None, embedding_model="embed-model")

    with pytest.raises(LLMError, match="EMBEDDING_API_KEY"):
        client.embedding("hello")


def test_post_json_wraps_read_timeout_as_model_error(monkeypatch):
    class TimeoutResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            raise TimeoutError("read timed out")

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: TimeoutResponse())
    client = OpenAICompatibleClient(api_key="llm-key", base_url="https://llm.example.com/v1", model="chat-model")

    with pytest.raises(LLMError, match="read timed out"):
        client.chat_json("system", "user")
