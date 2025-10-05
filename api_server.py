from __future__ import annotations

import math
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import cfg
from app.data_loader import load_messages
from app.intents import (
    propose_intents_from_topics,
    suggest_relevant_intents,
    embed_intents,
    zero_shot_filter,
)

from app.embeddings import compute_message_embeddings
from app.storage import load_json, load_parquet, save_json, save_parquet
from app.topics import load_topic_artifacts, run_topic_modeling
from app.content_generation import DraftMode, generate_content_draft
from app.telegram_sync import TelegramSyncError, refresh_telegram_dataset

ARTIFACT_ACTIVE_DATASET = cfg.artifacts_dir / "active_dataset.json"
DEFAULT_DATASET_NAME = cfg.dataset_file.name


class DatasetInfo(BaseModel):
    name: str
    path: str
    size_bytes: int
    modified_at: datetime


class DatasetState(BaseModel):
    name: str
    path: Optional[str] = None
    total_messages: int = 0
    updated_at: Optional[datetime] = None
    params: Optional[dict[str, Any]] = None


class ClusterSummary(BaseModel):
    cluster: int
    label: str
    size: int
    share: float
    is_noise: bool
    top_terms: list[str] = Field(default_factory=list)
    priority_score: float = Field(default=0.0)


class TopicMapPoint(BaseModel):
    message_id: str
    cluster: int
    cluster_label: str
    is_noise: bool
    x: float
    y: float
    z: Optional[float] = None
    sender: Optional[str] = None
    date: Optional[str] = None
    text: Optional[str] = None


class TopicTimelinePoint(BaseModel):
    cluster: int
    cluster_label: str
    date: str
    count: int
    period_count: int


class IntentMatchRow(BaseModel):
    message_id: str
    text: str
    sender: Optional[str] = None
    date: Optional[str] = None
    intent_score: float


class IntentMatch(BaseModel):
    intent: str
    results: list[IntentMatchRow] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    n_components: int = 50
    eps: Optional[float] = None
    min_samples: Optional[int] = None
    metric: str = "cosine"
    normalize: bool = False
    force: bool = False


class AnalyzeResponse(BaseModel):
    dataset: DatasetState
    quality_metrics: dict[str, Any]


class TelegramSyncRequest(BaseModel):
    start_date: str
    end_date: str
    cluster_limit: int = Field(..., ge=1)
    max_messages: Optional[int] = Field(default=None, ge=1)
    dataset_name: Optional[str] = None


class TelegramSyncResponse(BaseModel):
    dataset: DatasetState
    quality_metrics: dict[str, Any]
    records_fetched: int
    channels: list[str]
    start_date: str
    end_date: str


class IntentSuggestRequest(BaseModel):
    primary_query: Optional[str] = None
    intents: list[str] = Field(default_factory=list)
    top_n: int = 5


class IntentSuggestResponse(BaseModel):
    suggested: list[str] = Field(default_factory=list)


class IntentApplyRequest(BaseModel):
    dataset: Optional[str] = None
    intents: list[str]
    top_k: int = 200
    min_score: float = 0.25


class IntentApplyResponse(BaseModel):
    dataset: DatasetState
    matches: list[IntentMatch] = Field(default_factory=list)


class ContentDraftRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    mode: DraftMode = DraftMode.ARTICLE


class ContentDraftResponse(BaseModel):
    topic: str
    mode: DraftMode
    draft: str
    used_llm: bool


class StateResponse(BaseModel):
    generated_at: datetime
    active_dataset: DatasetState
    datasets: list[DatasetInfo]
    quality_metrics: dict[str, Any] = Field(default_factory=dict)
    clusters: list[ClusterSummary] = Field(default_factory=list)
    topic_map: list[TopicMapPoint] = Field(default_factory=list)
    topic_timeline: list[TopicTimelinePoint] = Field(default_factory=list)
    intent_suggestions: list[str] = Field(default_factory=list)


app = FastAPI(title="Trend Monitor API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _dataset_dir() -> Path:
    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    return cfg.dataset_dir


def _relativize(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(cfg.project_root))
    except ValueError:
        return str(path)


def _list_dataset_infos() -> list[DatasetInfo]:
    infos: list[DatasetInfo] = []
    for file in sorted(_dataset_dir().glob("*.json")):
        stat = file.stat()
        infos.append(
            DatasetInfo(
                name=file.name,
                path=_relativize(file),
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return infos


def _resolve_dataset_path(dataset_name: str) -> Path:
    if not dataset_name:
        raise ValueError("Dataset name must not be empty")
    base = _dataset_dir().resolve()
    candidate = (base / dataset_name).resolve()
    if not candidate.exists():
        raise FileNotFoundError(dataset_name)
    if base not in candidate.parents:
        raise ValueError("Dataset path escapes the dataset directory")
    return candidate


def _load_active_metadata() -> dict[str, Any]:
    data = load_json(ARTIFACT_ACTIVE_DATASET)
    return data or {}


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _clean_nullable(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _shorten(text: Optional[str], limit: int = 160) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_input_date(value: str) -> date:
    if not value:
        raise ValueError('Empty date value')
    cleaned = value.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def _clamp_end_date(candidate: date) -> date:
    today = datetime.utcnow().date()
    return candidate if candidate <= today else today


def _combine_utc(dt_date: date, dt_time: time) -> datetime:
    return datetime.combine(dt_date, dt_time).replace(tzinfo=timezone.utc)


def _build_dataset_state(metadata: dict[str, Any], fallback_name: str) -> tuple[DatasetState, Optional[Path]]:
    candidate_name = metadata.get("name") or metadata.get("dataset_name") or fallback_name
    dataset_path: Optional[Path]
    try:
        dataset_path = _resolve_dataset_path(candidate_name)
    except (FileNotFoundError, ValueError):
        dataset_path = None
    updated_at = _parse_datetime(metadata.get("updated_at"))
    params = metadata.get("params")
    total_messages = int(metadata.get("total_messages", 0))
    path_value = _relativize(dataset_path) if dataset_path else metadata.get("path")
    state = DatasetState(
        name=candidate_name,
        path=path_value,
        total_messages=total_messages,
        updated_at=updated_at,
        params=params,
    )
    return state, dataset_path


def _summarize_clusters(
    clusters_df: pd.DataFrame,
    labels: dict[int, str],
    cluster_terms: dict[int, list[str]],
    total_messages: int,
    cluster_limit: Optional[int] = None,
) -> list[ClusterSummary]:
    if clusters_df is None or clusters_df.empty:
        return []
    total = total_messages or len(clusters_df)
    summaries: list[ClusterSummary] = []
    grouped = clusters_df.groupby("cluster", dropna=False)
    for cluster_value, group in grouped:
        cluster_id = int(cluster_value) if not pd.isna(cluster_value) else -1
        size = int(len(group))
        is_noise = bool(group["is_noise"].iloc[0]) if "is_noise" in group.columns else cluster_id == -1
        label = "Noise" if cluster_id == -1 else labels.get(cluster_id) or f"Cluster {cluster_id}"
        terms = cluster_terms.get(cluster_id, [])[:8] if cluster_terms else []
        share = size / total if total else 0.0
        priority_score = float(cluster_terms.get(f"priority_{cluster_id}", 0.0)) if cluster_terms else 0.0
        summaries.append(
            ClusterSummary(
                cluster=cluster_id,
                label=label,
                size=size,
                share=share,
                is_noise=is_noise or cluster_id == -1,
                top_terms=terms,
                priority_score=priority_score,
            )
        )
    summaries.sort(key=lambda item: (item.is_noise, -item.priority_score, -item.size))
    if cluster_limit is not None and cluster_limit > 0:
        focused = [item for item in summaries if not item.is_noise]
        kept = focused[:cluster_limit]
        noise_items = [item for item in summaries if item.is_noise]
        summaries = kept + noise_items
    return summaries


def _build_topic_timeline(
    clusters_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    labels: dict[int, str],
    limit: int = 10,
    allowed_clusters: Optional[set[int]] = None,
) -> list[TopicTimelinePoint]:
    if clusters_df is None or clusters_df.empty:
        return []
    if messages_df is None or messages_df.empty:
        return []
    if 'date' not in messages_df.columns:
        return []

    merged = clusters_df.merge(
        messages_df[['message_id', 'date']],
        on='message_id',
        how='left',
    )
    merged = merged[merged['cluster'] != -1]
    if allowed_clusters is not None:
        merged = merged[merged['cluster'].isin(allowed_clusters)]
    if merged.empty:
        return []

    merged['date_dt'] = pd.to_datetime(merged['date'], errors='coerce')
    merged.dropna(subset=['date_dt'], inplace=True)
    if merged.empty:
        return []

    merged['date_norm'] = merged['date_dt'].dt.date
    if allowed_clusters is not None and allowed_clusters:
        limit = min(limit, len(allowed_clusters))
    top_clusters = (
        merged.groupby('cluster')['message_id'].count().sort_values(ascending=False).head(limit).index
    )
    filtered = merged[merged['cluster'].isin(top_clusters)]
    if filtered.empty:
        return []

    counts = (
        filtered.groupby(['cluster', 'date_norm'])['message_id']
        .count()
        .reset_index(name='period_count')
        .sort_values(['cluster', 'date_norm'])
    )
    counts['cumulative_count'] = (
        counts.groupby('cluster')['period_count']
        .cumsum()
    )

    points: list[TopicTimelinePoint] = []
    for row in counts.itertuples():
        cluster_id = int(row.cluster)
        label = labels.get(cluster_id) if labels else None
        label_text = label or f"Cluster {cluster_id}"
        points.append(
            TopicTimelinePoint(
                cluster=cluster_id,
                cluster_label=label_text,
                date=row.date_norm.isoformat(),
                count=int(row.cumulative_count),
                period_count=int(row.period_count),
            )
        )
    return points


def _build_topic_map(
    proj_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    labels: dict[int, str],
    allowed_clusters: Optional[set[int]] = None,
) -> list[TopicMapPoint]:
    if proj_df is None or proj_df.empty:
        return []
    coord_cols = [c for c in proj_df.columns if c.startswith("p")]
    if len(coord_cols) < 2:
        raise HTTPException(status_code=500, detail="Projection data must contain at least two components")
    rename_map = {coord_cols[0]: "x", coord_cols[1]: "y"}
    if len(coord_cols) >= 3:
        rename_map[coord_cols[2]] = "z"
    columns = ["message_id", *rename_map.keys()]
    coords = (
        proj_df[columns]
        .rename(columns=rename_map)
    )
    merged = coords.merge(clusters_df, on="message_id", how="left")
    if allowed_clusters is not None:
        merged = merged[merged["cluster"].isin(allowed_clusters)]
    if messages_df is not None and not messages_df.empty:
        meta_cols = [c for c in ["message_id", "sender", "date", "text"] if c in messages_df.columns]
        if meta_cols:
            meta = messages_df[meta_cols].copy()
            meta["message_id"] = meta["message_id"].astype(str)
            merged = merged.merge(meta, on="message_id", how="left")
    points: list[TopicMapPoint] = []
    for row in merged.itertuples():
        cluster_val = getattr(row, "cluster", -1)
        cluster_id = int(cluster_val) if not pd.isna(cluster_val) else -1
        label = "Noise" if cluster_id == -1 else labels.get(cluster_id) or f"Cluster {cluster_id}"
        is_noise = bool(getattr(row, "is_noise", cluster_id == -1)) or cluster_id == -1
        sender = _clean_nullable(getattr(row, "sender", None))
        date = _clean_nullable(getattr(row, "date", None))
        text = _shorten(_clean_nullable(getattr(row, "text", None)))
        raw_z = getattr(row, "z", None)
        z_value = None
        if raw_z is not None and not pd.isna(raw_z):
            z_value = float(raw_z)
        points.append(
            TopicMapPoint(
                message_id=str(row.message_id),
                cluster=cluster_id,
                cluster_label=label,
                is_noise=is_noise,
                x=float(getattr(row, "x")),
                y=float(getattr(row, "y")),
                z=z_value,
                sender=sender,
                date=date,
                text=text,
            )
        )
    return points


def _analyze_dataset_sync(dataset_name: str, params: AnalyzeRequest) -> AnalyzeResponse:
    previous_metadata = _load_active_metadata()
    try:
        dataset_path = _resolve_dataset_path(dataset_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    messages_df = load_messages(dataset_path)
    if messages_df.empty:
        raise HTTPException(status_code=400, detail="Selected dataset contains no messages")
    messages_df["message_id"] = messages_df["message_id"].astype(str)
    current_ids = set(messages_df["message_id"])

    proj_df, clusters_df, labels, existing_quality, _ = load_topic_artifacts()
    artifacts_cover_dataset = False
    if not params.force and proj_df is not None and clusters_df is not None:
        cluster_ids = set(clusters_df["message_id"].astype(str))
        artifacts_cover_dataset = current_ids.issubset(cluster_ids)

    if not artifacts_cover_dataset or params.force:
        emb_df = compute_message_embeddings(messages_df)
        proj_df, clusters_df, labels, quality_metrics, _ = run_topic_modeling(
            messages_df,
            emb_df,
            n_components=params.n_components,
            eps=params.eps,
            min_samples=params.min_samples,
            metric=params.metric,
            normalize=params.normalize,
        )
    else:
        quality_metrics = existing_quality or {}

    save_parquet(messages_df, cfg.messages_parquet)

    params_payload = {k: v for k, v in params.model_dump(exclude={"force"}).items() if v is not None}
    previous_params = previous_metadata.get("params") if isinstance(previous_metadata, dict) else None
    if artifacts_cover_dataset and not params.force and previous_params:
        params_payload = previous_params

    previous_updated_at = None
    previous_name = None
    if isinstance(previous_metadata, dict):
        previous_updated_at = _parse_datetime(previous_metadata.get("updated_at"))
        previous_name = previous_metadata.get("name")

    if artifacts_cover_dataset and not params.force and previous_name == dataset_name and previous_updated_at:
        updated_at = previous_updated_at
    else:
        updated_at = datetime.utcnow()

    metadata = {
        "name": dataset_name,
        "path": _relativize(dataset_path),
        "total_messages": len(messages_df),
        "updated_at": updated_at.isoformat(),
        "params": params_payload,
    }
    save_json(metadata, ARTIFACT_ACTIVE_DATASET)

    dataset_state = DatasetState(
        name=dataset_name,
        path=_relativize(dataset_path),
        total_messages=len(messages_df),
        updated_at=updated_at,
        params=params_payload,
    )

    return AnalyzeResponse(dataset=dataset_state, quality_metrics=quality_metrics)


def _build_state_sync() -> StateResponse:
    datasets = _list_dataset_infos()
    metadata = _load_active_metadata()
    dataset_state, dataset_path = _build_dataset_state(metadata, DEFAULT_DATASET_NAME)

    proj_df, clusters_df, labels, quality_metrics, cluster_terms = load_topic_artifacts()

    messages_df = load_parquet(cfg.messages_parquet)
    if messages_df is None:
        if dataset_path and dataset_path.exists():
            messages_df = load_messages(dataset_path)
        else:
            messages_df = pd.DataFrame(columns=["message_id"])

    labels_map = labels or {}
    cluster_terms_map = cluster_terms or {}
    metrics = dict(quality_metrics or {})

    params_dict = dataset_state.params or {}
    cluster_limit = _coerce_int(params_dict.get("cluster_limit")) if isinstance(params_dict, dict) else None

    if dataset_path:
        dataset_state = dataset_state.model_copy(update={"path": _relativize(dataset_path)})
    if dataset_state.updated_at is None and metadata.get("updated_at"):
        dataset_state = dataset_state.model_copy(update={"updated_at": _parse_datetime(metadata.get("updated_at"))})
    if params_dict and dataset_state.params is None:
        dataset_state = dataset_state.model_copy(update={"params": params_dict})

    total_messages = 0
    if metrics.get("total_messages"):
        try:
            total_messages = int(metrics["total_messages"])
        except Exception:
            total_messages = 0
    if not total_messages and clusters_df is not None:
        total_messages = len(clusters_df)
    if not total_messages and messages_df is not None:
        total_messages = len(messages_df)

    if dataset_state.total_messages == 0 and total_messages:
        dataset_state = dataset_state.model_copy(update={"total_messages": total_messages})

    if proj_df is None or clusters_df is None:
        metrics["visible_clusters"] = 0
        metrics["unique_clusters"] = 0
        intent_suggestions = list(labels_map.values()) if labels_map else []
        return StateResponse(
            generated_at=datetime.utcnow(),
            active_dataset=dataset_state,
            datasets=datasets,
            quality_metrics=metrics,
            clusters=[],
            topic_map=[],
            topic_timeline=[],
            intent_suggestions=intent_suggestions,
        )

    cluster_summary = _summarize_clusters(
        clusters_df,
        labels_map,
        cluster_terms_map,
        total_messages,
        cluster_limit=cluster_limit,
    )
    allowed_clusters = {item.cluster for item in cluster_summary if not item.is_noise}

    metrics['visible_clusters'] = len(allowed_clusters)
    metrics['unique_clusters'] = len(allowed_clusters)

    topic_map = _build_topic_map(
        proj_df,
        clusters_df,
        messages_df,
        labels_map,
        allowed_clusters=allowed_clusters if allowed_clusters else None,
    )
    timeline_limit = max(len(allowed_clusters), 1) if allowed_clusters else (cluster_limit or 10)
    topic_timeline = _build_topic_timeline(
        clusters_df,
        messages_df,
        labels_map,
        limit=timeline_limit,
        allowed_clusters=allowed_clusters if allowed_clusters else None,
    )
    try:
        intent_suggestions = propose_intents_from_topics(labels_map) if labels_map else []
    except Exception:
        intent_suggestions = list(labels_map.values()) if labels_map else []

    return StateResponse(
        generated_at=datetime.utcnow(),
        active_dataset=dataset_state,
        datasets=datasets,
        quality_metrics=metrics,
        clusters=cluster_summary,
        topic_map=topic_map,
        topic_timeline=topic_timeline,
        intent_suggestions=intent_suggestions,
    )


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets", response_model=list[DatasetInfo])
async def list_datasets() -> list[DatasetInfo]:
    return await run_in_threadpool(_list_dataset_infos)


@app.post("/api/datasets/{dataset_name}/analyze", response_model=AnalyzeResponse)
async def analyze_dataset(dataset_name: str, body: AnalyzeRequest = Body(default_factory=AnalyzeRequest)) -> AnalyzeResponse:
    return await run_in_threadpool(_analyze_dataset_sync, dataset_name, body)


@app.post("/api/datasets/telegram-sync", response_model=TelegramSyncResponse)
async def telegram_sync(body: TelegramSyncRequest) -> TelegramSyncResponse:
    return await run_in_threadpool(_telegram_sync_sync, body)


@app.get("/api/state", response_model=StateResponse)
async def state() -> StateResponse:
    return await run_in_threadpool(_build_state_sync)

def _telegram_sync_sync(payload: TelegramSyncRequest) -> TelegramSyncResponse:
    try:
        start_date = _parse_input_date(payload.start_date)
        end_date = _parse_input_date(payload.end_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    end_date = _clamp_end_date(end_date)
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date must not be after end date")

    start_dt = _combine_utc(start_date, time(0, 0, 0))
    end_dt_candidate = _combine_utc(end_date, time(23, 59, 59))
    now_utc = datetime.now(timezone.utc)
    end_dt = end_dt_candidate if end_dt_candidate <= now_utc else now_utc
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="Date range does not contain any time interval")

    max_messages = payload.max_messages or cfg.telegram_max_messages
    max_messages = max(1, min(max_messages, cfg.telegram_max_messages))

    try:
        dataset_result = refresh_telegram_dataset(
            start=start_dt,
            end=end_dt,
            max_messages=max_messages,
            dataset_name=payload.dataset_name,
        )
    except TelegramSyncError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    analyze_request = AnalyzeRequest(
        force=True,
        cluster_limit=payload.cluster_limit,
        start_date=start_dt.date().isoformat(),
        end_date=end_dt.date().isoformat(),
        record_limit=max_messages,
        records_fetched=dataset_result.total_messages,
        channels=list(dataset_result.channels),
    )
    analyze_response = _analyze_dataset_sync(dataset_result.dataset_name, analyze_request)
    dataset_state = analyze_response.dataset
    params = dict(dataset_state.params or {})
    params.update(
        {
            "cluster_limit": payload.cluster_limit,
            "record_limit": max_messages,
            "records_fetched": dataset_result.total_messages,
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "channels": list(dataset_result.channels),
        }
    )
    dataset_state = dataset_state.model_copy(update={"params": params})

    return TelegramSyncResponse(
        dataset=dataset_state,
        quality_metrics=analyze_response.quality_metrics,
        records_fetched=dataset_result.total_messages,
        channels=list(dataset_result.channels),
        start_date=start_dt.date().isoformat(),
        end_date=end_dt.date().isoformat(),
    )


def _suggest_intents_sync(payload: IntentSuggestRequest) -> IntentSuggestResponse:
    intents = [intent.strip() for intent in payload.intents if intent and intent.strip()]
    if not intents:
        return IntentSuggestResponse(suggested=[])
    if payload.primary_query:
        suggested = suggest_relevant_intents(payload.primary_query, intents, top_n=payload.top_n)
    else:
        suggested = intents[: payload.top_n]
    return IntentSuggestResponse(suggested=suggested)


def _apply_intents_sync(payload: IntentApplyRequest) -> IntentApplyResponse:
    intents = [intent.strip() for intent in payload.intents if intent and intent.strip()]
    if not intents:
        raise HTTPException(status_code=400, detail="Intents list must not be empty")

    metadata = _load_active_metadata()
    fallback_name = metadata.get("name") if isinstance(metadata, dict) else None
    dataset_name = payload.dataset or fallback_name or DEFAULT_DATASET_NAME
    try:
        dataset_path = _resolve_dataset_path(dataset_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    messages_df = load_messages(dataset_path)
    if messages_df.empty:
        raise HTTPException(status_code=400, detail="Selected dataset contains no messages")
    messages_df["message_id"] = messages_df["message_id"].astype(str)

    emb_df = compute_message_embeddings(messages_df)
    vector_cols = [c for c in emb_df.columns if c.startswith("e")]
    if not vector_cols:
        raise HTTPException(status_code=500, detail="Embeddings not available for intent matching")

    merged = emb_df.merge(messages_df, on="message_id", how="left")
    message_matrix = merged[vector_cols].to_numpy(dtype=float)
    messages_for_filter = merged[["message_id", "text", "sender", "date"]].fillna("")

    if not cfg.openai_api_key:
        raise HTTPException(status_code=400, detail="Intent matching requires OPENAI_API_KEY")

    try:
        intent_embeddings = embed_intents(intents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results_map = zero_shot_filter(
        messages_for_filter,
        intents,
        message_matrix,
        intent_embeddings,
        top_k=payload.top_k,
        min_score=payload.min_score,
    )

    matches: list[IntentMatch] = []
    for intent in intents:
        df = results_map.get(intent)
        if df is None or df.empty:
            matches.append(IntentMatch(intent=intent, results=[]))
            continue
        subset = df.head(payload.top_k)
        rows = [
            IntentMatchRow(
                message_id=str(row.message_id),
                text=str(row.text),
                sender=_clean_nullable(getattr(row, "sender", None)),
                date=_clean_nullable(getattr(row, "date", None)),
                intent_score=float(row.intent_score),
            )
            for row in subset.itertuples(index=False)
        ]
        matches.append(IntentMatch(intent=intent, results=rows))

    dataset_state = DatasetState(
        name=dataset_name,
        path=_relativize(dataset_path),
        total_messages=len(messages_df),
        updated_at=datetime.utcnow(),
    )

    return IntentApplyResponse(dataset=dataset_state, matches=matches)


def _generate_content_draft_sync(payload: ContentDraftRequest) -> ContentDraftResponse:
    topic = (payload.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must not be empty")
    try:
        draft, used_llm = generate_content_draft(topic, payload.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to generate content draft") from exc
    return ContentDraftResponse(topic=topic, mode=payload.mode, draft=draft, used_llm=used_llm)


@app.post("/api/content/generate", response_model=ContentDraftResponse)
async def generate_content_draft_endpoint(body: ContentDraftRequest) -> ContentDraftResponse:
    return await run_in_threadpool(_generate_content_draft_sync, body)




@app.post("/api/intents/suggest", response_model=IntentSuggestResponse)
async def suggest_intents(body: IntentSuggestRequest) -> IntentSuggestResponse:
    return await run_in_threadpool(_suggest_intents_sync, body)


@app.post("/api/intents/apply", response_model=IntentApplyResponse)
async def apply_intents(body: IntentApplyRequest) -> IntentApplyResponse:
    return await run_in_threadpool(_apply_intents_sync, body)
