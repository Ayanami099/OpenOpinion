from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv(path: str | Path = DOTENV_PATH, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Existing environment variables win by default, which keeps shell/CI secrets
    higher priority than local developer defaults.
    """
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = _strip_inline_comment(value.strip())
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return key, value


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            prefix = value[:index]
            if not prefix or prefix[-1].isspace():
                return prefix.strip()
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


@lru_cache(maxsize=1)
def load_configs() -> dict[str, Any]:
    return {
        "event_types": load_yaml(CONFIG_DIR / "event_types.yaml"),
        "intents": load_yaml(CONFIG_DIR / "intents.yaml"),
        "platform_rules": load_yaml(CONFIG_DIR / "platform_rules.yaml"),
        "source_credibility": load_yaml(CONFIG_DIR / "source_credibility.yaml"),
        "risk_weights": load_yaml(CONFIG_DIR / "risk_weights.yaml"),
    }


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
