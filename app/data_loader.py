from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import hashlib
import re
import pandas as pd

from .config import cfg


def _flatten_text(text_field: Any) -> str:
    if text_field is None:
        return ""
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts: list[str] = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Telegram exports may contain {"type": "text", "text": "..."}
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return " ".join(parts)
    return str(text_field)

PREFERRED_TEXT_FIELDS = (
    "investor_focus",
    "focus_text",
    "summary_text",
    "text",
)


def extract_message_text(rec: dict[str, Any]) -> str:
    for field in PREFERRED_TEXT_FIELDS:
        if field in rec:
            value = _flatten_text(rec.get(field))
            if value:
                return value
    return ""




def get_raw_message_text(rec: dict[str, Any]) -> str:
    return _flatten_text(rec.get("text"))


def _normalize_sender(rec: dict[str, Any]) -> str:
    for key in ["from", "sender", "author", "user"]:
        if key in rec and isinstance(rec[key], str) and rec[key].strip():
            return rec[key].strip()
    return "unknown"


def _normalize_date(rec: dict[str, Any]) -> str:
    # Keep as ISO-like string to avoid tz pitfalls; pandas can parse later
    for key in ["date", "timestamp", "time"]:
        if key in rec and isinstance(rec[key], str):
            return rec[key]
    return ""


def load_messages(path: Path | None = None) -> pd.DataFrame:
    path = path or cfg.dataset_file
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "messages" in raw:
        data = raw["messages"]
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError("Unsupported dataset/result.json format")

    records: list[dict[str, Any]] = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            continue
        text = extract_message_text(rec)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        message_id = str(rec.get("id", i))
        sender = _normalize_sender(rec)
        date = _normalize_date(rec)
        records.append(
            {
                "message_id": message_id,
                "sender": sender,
                "date": date,
                "text": text,
            }
        )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.drop_duplicates(subset=["message_id"], inplace=True)
    return df


def merge_datasets(dataset_names: Iterable[str], output_name: str = "combined_dataset.json") -> Path:
    """Merge multiple dataset JSON files into a single dataset file inside the dataset directory."""
    normalized: list[str] = []
    for name in dataset_names:
        if not name:
            continue
        cleaned = name.strip()
        if not cleaned:
            continue
        filename = cleaned if cleaned.lower().endswith(".json") else f"{cleaned}.json"
        if filename not in normalized:
            normalized.append(filename)
    if not normalized:
        raise ValueError("dataset_names must contain at least one entry")

    dataset_dir = cfg.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    available_files = list(dataset_dir.glob("*.json"))
    if not available_files:
        raise FileNotFoundError(f"No JSON datasets found in {dataset_dir}")

    def resolve_path(candidate_name: str) -> Path:
        direct = dataset_dir / candidate_name
        if direct.exists():
            return direct
        lower_candidate = candidate_name.lower()
        for file_path in available_files:
            if file_path.name.lower() == lower_candidate:
                return file_path
        simplified = re.sub(r"[^a-z0-9]", "", lower_candidate)
        for file_path in available_files:
            if re.sub(r"[^a-z0-9]", "", file_path.name.lower()) == simplified:
                return file_path
        raise FileNotFoundError(f"Dataset '{candidate_name}' not found in {dataset_dir}")

    merged_messages: list[dict[str, Any]] = []
    source_files: list[str] = []
    for filename in normalized:
        path = resolve_path(filename)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            messages = raw.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"Dataset '{path.name}' does not contain a 'messages' list")
        elif isinstance(raw, list):
            messages = raw
        else:
            raise ValueError(f"Unsupported dataset format in '{path.name}'")

        source_files.append(path.name)
        prefix = path.stem
        for index, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            message = dict(item)
            original_id = message.get("id", index)
            message["id"] = f"{prefix}-{original_id}"
            merged_messages.append(message)

    if not output_name.lower().endswith(".json"):
        output_name = f"{output_name}.json"

    combined = {
        "name": "Merged dataset",
        "type": "merged_dataset",
        "id": "merged_dataset",
        "source_files": source_files,
        "messages": merged_messages,
    }

    output_path = dataset_dir / output_name
    output_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
