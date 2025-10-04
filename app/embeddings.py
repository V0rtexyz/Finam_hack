from __future__ import annotations

import hashlib
import os
import pickle
from typing import Iterable, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

from .storage import EmbeddingCache, EmbeddingRecord

load_dotenv()

_DOC_CACHE_MODEL = "ycloud:text-search-doc/latest"

_YC_CLIENT = None
_YC_MODELS: dict[str, object] = {}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _serialize_vector(vec: np.ndarray) -> bytes:
    return pickle.dumps(vec.astype(np.float32))


def _deserialize_vector(blob: bytes) -> np.ndarray:
    return pickle.loads(blob)


def _chunk_iter(items: list[str], chunk_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _get_ycloud_client():
    global _YC_CLIENT
    if _YC_CLIENT is not None:
        return _YC_CLIENT
    folder_id = os.getenv("FOLDER_ID")
    api_key = os.getenv("API_KEY_EMBEDDER")
    if not folder_id or not api_key:
        raise RuntimeError("FOLDER_ID and API_KEY_EMBEDDER must be set for Yandex Cloud embeddings")
    from yandex_cloud_ml_sdk import YCloudML

    _YC_CLIENT = YCloudML(folder_id=folder_id, auth=api_key)
    return _YC_CLIENT


def _get_ycloud_model(text_type: str):
    if text_type not in {"doc", "query"}:
        raise ValueError("text_type must be 'doc' or 'query'")
    if text_type in _YC_MODELS:
        return _YC_MODELS[text_type]
    client = _get_ycloud_client()
    model = client.models.text_embeddings(text_type)
    _YC_MODELS[text_type] = model
    return model


class YandexEmbeddings:
    def __init__(self, text_type: str = "doc") -> None:
        self.text_type = text_type

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(6))
    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = _get_ycloud_model(self.text_type)
        batches: list[np.ndarray] = []
        for chunk in _chunk_iter(texts, 32):
            vectors = [model.run(text) for text in chunk]
            if not vectors:
                continue
            batches.append(np.array(vectors, dtype=np.float32))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches)


def embed_texts(texts: list[str], text_type: str = "doc") -> np.ndarray:
    embedder = YandexEmbeddings(text_type=text_type)
    try:
        return embedder.embed(texts)
    except Exception as e:
        import traceback
        print("Embedding failed:", e)
        traceback.print_exc()
        raise


def compute_message_embeddings(df: pd.DataFrame, text_col: str = "text", id_col: str = "message_id") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[id_col])

    cache = EmbeddingCache()
    model_name = _DOC_CACHE_MODEL

    ids = df[id_col].astype(str).tolist()
    texts = df[text_col].astype(str).tolist()
    hashes = [_hash_text(t) for t in texts]

    existing = cache.get_many(ids, model=model_name)
    missing_ids: list[str] = []
    missing_texts: list[str] = []
    missing_hashes: list[str] = []
    for mid, text, text_hash in zip(ids, texts, hashes):
        if mid not in existing:
            missing_ids.append(mid)
            missing_texts.append(text)
            missing_hashes.append(text_hash)

    if missing_ids:
        vectors = embed_texts(missing_texts, text_type="doc")
        records = [
            EmbeddingRecord(message_id=m_id, text_hash=h, model=model_name, vector=_serialize_vector(vec))
            for m_id, h, vec in zip(missing_ids, missing_hashes, vectors)
        ]
        cache.put_many(records)

    all_vectors_blob = cache.get_many(ids, model=model_name)
    vectors_ordered = [_deserialize_vector(all_vectors_blob[mid]) for mid in ids]
    emb = np.vstack(vectors_ordered)
    emb_df = pd.DataFrame(emb)
    emb_df.columns = [f"e{i:04d}" for i in range(emb_df.shape[1])]
    emb_df[id_col] = ids
    return emb_df


def build_semantic_batches(
    emb_df: pd.DataFrame,
    id_col: str = "message_id",
    threshold: float = 0.2,
    min_batch: int = 2,
    max_batch: int = 25,
) -> pd.DataFrame:
    """
    Формируем батчи сообщений так, чтобы внутри окна семантическая
    связность сохранялась: cosine-dist между соседями < threshold.

    threshold — максимально допустимый "дрейф".
    min_batch — минимум сообщений в батче.
    max_batch — максимум (safety, чтобы не раздувалось).
    """
    if emb_df.empty:
        return pd.DataFrame(columns=[id_col, "text"])

    vec_cols = [c for c in emb_df.columns if c.startswith("e")]
    if not vec_cols:
        raise ValueError("Vector columns not found in embeddings DataFrame")

    vectors = emb_df[vec_cols].to_numpy()
    ids = emb_df[id_col].astype(str).tolist()
    if not ids:
        return pd.DataFrame(columns=[id_col, "text"])

    batches: List[List[str]] = []
    current: List[str] = [ids[0]]

    for i in range(1, len(ids)):
        v_prev, v_cur = vectors[i - 1], vectors[i]
        sim = cosine_similarity([v_prev], [v_cur])[0, 0]
        dist = 1 - sim

        if dist < threshold and len(current) < max_batch:
            current.append(ids[i])
        else:
            if len(current) >= min_batch:
                batches.append(current)
            current = [ids[i]]

    if current:
        batches.append(current)

    batch_records = []
    for batch_idx, batch_ids in enumerate(batches):
        if "text" in emb_df.columns:
            texts = emb_df.loc[emb_df[id_col].astype(str).isin(batch_ids), "text"].astype(str).tolist()
        else:
            texts = [str(mid) for mid in batch_ids]
        combined_text = " ".join(texts)
        batch_records.append({"message_id": str(batch_idx), "text": combined_text})
    return pd.DataFrame(batch_records)


def compute_batches_embeddings(
    batch_df: pd.DataFrame
):
    batch_emb_df = compute_message_embeddings(batch_df)
    return batch_emb_df
