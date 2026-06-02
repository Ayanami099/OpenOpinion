import os

from opinion_agent.config import load_dotenv


def test_load_dotenv_sets_missing_values(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "# comment",
                "TAVILY_API_KEY=abc123",
                "OPENAI_BASE_URL=\"https://example.com/v1\"",
                "export OPENAI_MODEL='test-model' # inline comment",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    loaded = load_dotenv(dotenv)

    assert loaded["TAVILY_API_KEY"] == "abc123"
    assert os.environ["OPENAI_BASE_URL"] == "https://example.com/v1"
    assert os.environ["OPENAI_MODEL"] == "test-model"


def test_load_dotenv_does_not_override_existing_values(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=from-file", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")

    loaded = load_dotenv(dotenv)

    assert "OPENAI_API_KEY" not in loaded
    assert os.environ["OPENAI_API_KEY"] == "from-shell"

