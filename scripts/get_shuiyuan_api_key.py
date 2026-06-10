#!/usr/bin/env python3
"""Request a Shuiyuan Discourse User API Key.

This script opens Shuiyuan's authorization page, asks you to authorize this
local app, then decrypts the response payload locally. The resulting key can be
used in config.json as:

{
  "shuiyuan_user_api_key": "...",
  "shuiyuan_user_api_client_id": "...",
  "shuiyuan_cookies": {}
}
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import secrets
from collections.abc import Iterable
import urllib.parse
import uuid
import webbrowser

import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


SITE_URL_BASE = "https://shuiyuan.sjtu.edu.cn"
ALL_SCOPES = [
    "read",
    "write",
    "message_bus",
    "push",
    "one_time_password",
    "notifications",
    "session_info",
    "bookmarks_calendar",
    "user_status",
]
DEFAULT_SCOPES = ["read"]


@dataclass
class UserApiKeyPayload:
    key: str
    nonce: str
    push: bool
    api: int


@dataclass
class UserApiKeyRequestResult:
    client_id: str
    payload: UserApiKeyPayload


def generate_user_api_key(
    application_name: str,
    *,
    client_id: str | None = None,
    scopes: Iterable[str] | None = None,
) -> UserApiKeyRequestResult:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    public_key = private_key.public_key()
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")

    client_id_to_use = str(uuid.uuid4()) if client_id is None else client_id
    nonce = secrets.token_urlsafe(32)

    scopes_list = DEFAULT_SCOPES if scopes is None else list(scopes)
    if not set(scopes_list) <= set(ALL_SCOPES):
        raise ValueError(f"Invalid scopes: {scopes_list}")

    params = {
        "application_name": application_name,
        "client_id": client_id_to_use,
        "scopes": ",".join(scopes_list),
        "public_key": public_key_pem,
        "nonce": nonce,
    }
    query = "&".join(f"{key}={urllib.parse.quote(value)}" for key, value in params.items())
    auth_url = f"{SITE_URL_BASE}/user-api-key/new?{query}"

    print("Opening Shuiyuan authorization page...")
    print(auth_url)
    webbrowser.open(auth_url)

    encrypted_payload = input("\nPaste the response payload here: ").strip()
    decrypted = private_key.decrypt(base64.b64decode(encrypted_payload), padding.PKCS1v15())
    payload = UserApiKeyPayload(**json.loads(decrypted))
    if payload.nonce != nonce:
        raise ValueError("Nonce mismatch. Refusing to use this payload.")

    return UserApiKeyRequestResult(client_id=client_id_to_use, payload=payload)


def test_user_api_key(key: str, client_id: str) -> dict:
    response = requests.get(
        f"{SITE_URL_BASE}/search.json",
        params={"q": "tags:水源开发者"},
        headers={
            "User-Api-Key": key,
            "User-Api-Client-Id": client_id,
            "Accept": "application/json",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def write_config(path: Path, api_key: str, client_id: str) -> None:
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data["shuiyuan_user_api_key"] = api_key
    data["shuiyuan_user_api_client_id"] = client_id
    data.setdefault("shuiyuan_cookies", {})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Request a Shuiyuan Discourse User API Key")
    parser.add_argument("--application-name", default="Campus Opinion Agent")
    parser.add_argument("--client-id", default="")
    parser.add_argument("--scope", action="append", choices=ALL_SCOPES, help="Discourse user API scope. Defaults to read.")
    parser.add_argument("--write-config", action="store_true", help="Write the key to ./config.json after success.")
    parser.add_argument("--config", default="config.json", help="Config path used with --write-config.")
    args = parser.parse_args()

    result = generate_user_api_key(
        args.application_name,
        client_id=args.client_id or None,
        scopes=args.scope or DEFAULT_SCOPES,
    )
    api_key = result.payload.key
    client_id = result.client_id

    print("\nShuiyuan API key received.")
    print("\nAdd these fields to config.json:")
    print(json.dumps({
        "shuiyuan_user_api_key": api_key,
        "shuiyuan_user_api_client_id": client_id,
        "shuiyuan_cookies": {},
    }, ensure_ascii=False, indent=2))

    try:
        data = test_user_api_key(api_key, client_id)
        topic_count = len(data.get("topics") or [])
        print(f"\nKey test succeeded. Search returned {topic_count} topics.")
    except Exception as exc:
        print(f"\nKey test failed: {exc}")

    if args.write_config:
        config_path = Path(args.config)
        write_config(config_path, api_key, client_id)
        print(f"\nWrote Shuiyuan credentials to {config_path}")


if __name__ == "__main__":
    main()

