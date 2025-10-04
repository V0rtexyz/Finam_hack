from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import json
from typing import Iterable, Optional, Tuple

import pandas as pd

from .config import cfg


@dataclass
class EmbeddingRecord:
    message_id: str
    text_hash: str
    vector: bytes  # stored as blob (numpy array serialized)
    model: str


class EmbeddingCache:
    def __init__(self, sqlite_path: Path | str | None = None):
        self.sqlite_path = Path(sqlite_path) if sqlite_path else cfg.embeddings_sqlite
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.sqlite_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    message_id TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (message_id, model)
                );
                """
            )

    def get_many(self, message_ids: Iterable[str], model: str) -> dict[str, bytes]:
        ids = list(message_ids)
        if not ids:
            return {}
        placeholders = ",".join(["?"] * len(ids))
        query = f"SELECT message_id, vector FROM embeddings WHERE model = ? AND message_id IN ({placeholders})"
        with sqlite3.connect(self.sqlite_path) as con:
            cur = con.execute(query, [model, *ids])
            return {row[0]: row[1] for row in cur.fetchall()}

    def put_many(self, records: Iterable[EmbeddingRecord]) -> None:
        rows = [(r.message_id, r.text_hash, r.model, r.vector) for r in records]
        with sqlite3.connect(self.sqlite_path) as con:
            con.executemany(
                "INSERT OR REPLACE INTO embeddings (message_id, text_hash, model, vector) VALUES (?, ?, ?, ?)",
                rows,
            )


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

