from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from telethon import TelegramClient
from telethon.errors import RPCError, SessionPasswordNeededError
from telethon.sessions import StringSession

from .config import cfg


class TelegramSyncError(RuntimeError):
    """Raised when fetching Telegram data fails."""


@dataclass
class TelegramDatasetResult:
    dataset_path: Path
    dataset_name: str
    total_messages: int
    channels: list[str]
    start: datetime
    end: datetime
    fetched_at: datetime


def _normalize_channel(reference: str) -> str:
    reference = reference.strip()
    if reference.startswith("http"):
        reference = reference.split("//", 1)[-1]
        parts = reference.split("/", 1)
        reference = parts[1] if len(parts) > 1 else parts[0]
    reference = reference.lstrip("@")
    reference = reference.strip()
    if not reference:
        raise TelegramSyncError("Empty channel reference encountered")
    return reference


def _resolve_channels(channels: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in channels:
        name = _normalize_channel(item)
        if name and name not in seen:
            seen.add(name)
            normalized.append(name)
    if not normalized:
        raise TelegramSyncError("No Telegram channels configured")
    return normalized


def _build_client() -> TelegramClient:
    if not cfg.has_telegram_credentials:
        raise TelegramSyncError("Telegram API credentials are not configured")
    api_id = cfg.telegram_api_id
    api_hash = cfg.telegram_api_hash
    if api_id is None or not api_hash:
        raise TelegramSyncError("Telegram API credentials are invalid")
    session_string = cfg.telegram_session_string
    if session_string:
        return TelegramClient(StringSession(session_string), api_id, api_hash)
    session_path = cfg.telegram_session_path
    session_path.parent.mkdir(parents=True, exist_ok=True)
    return TelegramClient(str(session_path), api_id, api_hash)


def _make_record(channel: str, title: str, message) -> dict[str, object]:
    stamp = message.date.astimezone(timezone.utc)
    text = message.message or ""
    record: dict[str, object] = {
        "id": message.id,
        "type": "message",
        "date": stamp.replace(microsecond=0).isoformat(),
        "date_unixtime": str(int(stamp.timestamp())),
        "from": title or channel,
        "from_id": str(getattr(message.peer_id, "channel_id", channel)),
        "channel": channel,
        "text": text,
    }
    if channel:
        record["link"] = f"https://t.me/{channel}/{message.id}"
    return record


async def _collect_messages(
    client: TelegramClient,
    channels: Sequence[str],
    start: datetime,
    end: datetime,
    limit: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for channel in channels:
        remaining = limit - len(results)
        if remaining <= 0:
            break
        try:
            entity = await client.get_entity(channel)
        except RPCError as exc:
            raise TelegramSyncError(f"Failed to resolve channel '{channel}': {exc}") from exc
        title = getattr(entity, "title", None) or getattr(entity, "username", channel)
        async for message in client.iter_messages(entity, offset_date=end, limit=remaining):
            if message is None or message.date is None:
                continue
            stamp = message.date.astimezone(timezone.utc)
            if stamp < start:
                break
            if stamp > end:
                continue
            if not getattr(message, "message", None):
                continue
            if getattr(message, "action", None) is not None:
                continue
            results.append(_make_record(channel, title, message))
            if len(results) >= limit:
                break
    results.sort(key=lambda item: item.get("date", ""))
    return results


async def _fetch_dataset_async(
    channels: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    limit: int,
) -> list[dict[str, object]]:
    client = _build_client()
    try:
        async with client:
            return await _collect_messages(client, channels, start, end, limit)
    except SessionPasswordNeededError as exc:
        raise TelegramSyncError(
            "Telegram session requires 2FA password. Provide TELEGRAM_SESSION_STRING"
        ) from exc
    except RPCError as exc:
        raise TelegramSyncError(str(exc)) from exc


def refresh_telegram_dataset(
    start: datetime,
    end: datetime,
    *,
    max_messages: int | None = None,
    dataset_name: str | None = None,
    channels: Sequence[str] | None = None,
) -> TelegramDatasetResult:
    if start.tzinfo is None or end.tzinfo is None:
        raise TelegramSyncError("Start and end datetimes must be timezone-aware")
    if start > end:
        raise TelegramSyncError("Start date must not be after end date")
    channel_list = _resolve_channels(channels or cfg.telegram_channels)
    limit = max_messages or cfg.telegram_max_messages
    limit = min(limit, cfg.telegram_max_messages)
    messages = asyncio.run(
        _fetch_dataset_async(channel_list, start, end, limit=limit)
    )
    dataset_dir = cfg.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    start_tag = start.date().strftime("%Y%m%d")
    end_tag = end.date().strftime("%Y%m%d")
    name = dataset_name or f"telegram_{start_tag}_{end_tag}.json"
    path = dataset_dir / name
    payload = {
        "name": name,
        "type": "telegram_dataset",
        "id": name,
        "start_date": start.replace(microsecond=0).isoformat(),
        "end_date": end.replace(microsecond=0).isoformat(),
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "channels": channel_list,
        "message_limit": limit,
        "messages": messages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return TelegramDatasetResult(
        dataset_path=path,
        dataset_name=name,
        total_messages=len(messages),
        channels=channel_list,
        start=start,
        end=end,
        fetched_at=datetime.now(timezone.utc),
    )


__all__ = ["TelegramDatasetResult", "TelegramSyncError", "refresh_telegram_dataset"]
