from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Optional

import jwt
import requests

from .config import cfg

from scripts.get_private_key import get_private_key



class YandexGPTError(RuntimeError):
    pass


@dataclass
class _ChatCompletions:
    client: "YandexGPTClient"

    def create(
        self,
        *,
        model: str,
        messages: Iterable[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> SimpleNamespace:
        text = self.client._create_completion(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

class YandexGPTClient:
    def __init__(self) -> None:
        self._folder_id = cfg.yandex_folder_id
        self._service_account_id = cfg.yandex_service_account_id
        self._key_id = cfg.yandex_key_id
        self._private_key_path = cfg.yandex_private_key_path
        self._iam_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self.chat = SimpleNamespace(completions=_ChatCompletions(self))

        missing = [
            name
            for name, value in (
                ("FOLDER_ID", self._folder_id),
                ("SERVICE_ACCOUNT_ID", self._service_account_id),
                ("KEY_ID", self._key_id),
                ("PRIVATE_KEY", self._private_key_path),
            )
            if not value
        ]
        if missing:
            raise YandexGPTError(
                "Missing Yandex GPT credentials. Expected env variables: "
                + ", ".join(missing)
            )

        self._private_key = self._load_private_key()

    def _load_private_key(self) -> str:
        try:
            raw_key = get_private_key()
        except FileNotFoundError as exc:
            raise YandexGPTError(str(exc)) from exc
        except Exception as exc:
            raise YandexGPTError(f"Failed to load private key: {exc}") from exc

        if isinstance(raw_key, (bytes, bytearray)):
            return raw_key.decode("utf-8")
        return str(raw_key)

    def _get_iam_token(self) -> str:
        now = time.time()
        if self._iam_token and now < self._token_expires_at - 60:
            return self._iam_token

        payload = {
            "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            "iss": self._service_account_id,
            "iat": int(now),
            "exp": int(now) + 3600,
        }

        encoded = jwt.encode(
            payload,
            self._private_key,
            algorithm="PS256",
            headers={"kid": self._key_id},
        )

        try:
            resp = requests.post(
                "https://iam.api.cloud.yandex.net/iam/v1/tokens",
                json={"jwt": encoded},
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise YandexGPTError(f"IAM token request failed: {exc}") from exc

        data = resp.json()
        token = data.get("iamToken")
        if not token:
            raise YandexGPTError("IAM token response missing 'iamToken'")

        self._iam_token = token
        self._token_expires_at = now + 3500
        return token

    def _create_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        iam_token = self._get_iam_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {iam_token}",
            "x-folder-id": self._folder_id,
        }
        payload = {
            "modelUri": f"gpt://{self._folder_id}/{model}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": [
                {
                    "role": m.get("role", "user"),
                    "text": str(m.get("content", "")),
                }
                for m in messages
            ],
        }

        try:
            resp = requests.post(
                "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise YandexGPTError(f"Completion request failed: {exc}") from exc

        data = resp.json()
        try:
            return data["result"]["alternatives"][0]["message"]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise YandexGPTError("Unexpected response structure from Yandex GPT") from exc


_CLIENT: Optional[YandexGPTClient] = None


def get_yandex_gpt_client() -> YandexGPTClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = YandexGPTClient()
    return _CLIENT


def has_yandex_gpt_credentials() -> bool:
    return cfg.has_yandex_credentials
