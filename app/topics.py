from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import json
import re

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import re

_LABEL_LINE_RE = re.compile(
    r"(?:cluster\s*)?(?P<id>\d+)[\)\s:\-]+(?P<label>.+)", 
    re.IGNORECASE
)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import cfg
from .storage import save_parquet, load_parquet, save_json, load_json
from .yandex_gpt import get_yandex_gpt_client, has_yandex_gpt_credentials


def _auto_dbscan_params(X: np.ndarray, metric: str = 'cosine') -> tuple[float, int]:
    n_samples = len(X)
    min_samples = max(3, min(10, int(np.log(n_samples))))
    k = min_samples
    try:
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
        distances, _ = nbrs.kneighbors(X)
        k_distances = distances[:, k - 1]
        k_distances = np.sort(k_distances)
        eps = np.percentile(k_distances, 85)
        if metric == 'cosine':
            if eps < 0.2:
                eps = 0.3
            elif eps > 0.8:
                eps = 0.6
        else:
            if eps < 1.0:
                eps = 1.5
            elif eps > 10.0:
                eps = 5.0
        print(f"Auto DBSCAN params: eps={eps:.3f}, min_samples={min_samples}, data_points={n_samples}")
    except Exception as e:
        print(f"???????????? ?? _auto_dbscan_params: {e}")
        eps, min_samples = 0.3, 5  # Fallback
    return eps, min_samples



@dataclass
class _DbscanAttempt:
    eps: float
    min_samples: int
    labels: np.ndarray
    cluster_count: int
    noise_ratio: float


def _summarize_labels(labels: np.ndarray) -> tuple[int, float]:
    if len(labels) == 0:
        return 0, 0.0
    noise_mask = labels == -1
    cluster_count = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_ratio = float(noise_mask.sum() / len(labels))
    return cluster_count, noise_ratio


def _fit_dbscan(X: np.ndarray, eps: float, min_samples: int, metric: str) -> _DbscanAttempt:
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)
    cluster_count, noise_ratio = _summarize_labels(labels)
    return _DbscanAttempt(
        eps=float(eps),
        min_samples=int(min_samples),
        labels=labels,
        cluster_count=cluster_count,
        noise_ratio=noise_ratio,
    )


def _adaptive_dbscan_search(
    X: np.ndarray,
    metric: str,
    base_eps: float,
    base_min_samples: int,
    *,
    allow_eps_variation: bool,
    allow_min_samples_variation: bool,
) -> _DbscanAttempt:
    eps_candidates: list[float] = [max(0.05, float(base_eps))]
    if allow_eps_variation:
        for scale in (0.9, 0.8, 0.7, 0.6, 0.5, 0.42, 0.36, 0.3, 0.25, 0.22, 0.18, 0.15, 0.12, 0.1):
            candidate = max(0.05, float(base_eps) * scale)
            if all(abs(candidate - existing) > 1e-6 for existing in eps_candidates):
                eps_candidates.append(candidate)
    eps_candidates = sorted(eps_candidates, reverse=True)

    min_samples_candidates: set[int] = {max(3, int(round(base_min_samples)))}
    if allow_min_samples_variation:
        min_samples_candidates.update(
            {
                max(3, int(round(base_min_samples)) - 1),
                max(3, int(round(base_min_samples)) - 2),
                max(3, int(np.floor(base_min_samples * 0.75))),
            }
        )
    min_samples_list = sorted(min_samples_candidates)

    attempts: list[_DbscanAttempt] = []
    seen: set[tuple[int, int]] = set()
    for min_samples_value in min_samples_list:
        for eps_value in eps_candidates:
            key = (int(round(eps_value * 1000)), min_samples_value)
            if key in seen:
                continue
            attempt = _fit_dbscan(X, eps_value, min_samples_value, metric)
            attempts.append(attempt)
            seen.add(key)

    if not attempts:
        raise RuntimeError('DBSCAN auto-tuning produced no attempts')

    acceptable = [
        attempt
        for attempt in attempts
        if attempt.cluster_count >= 2 and attempt.noise_ratio <= 0.95
    ]

    def _attempt_score(attempt: _DbscanAttempt) -> tuple[int, float, float]:
        target_noise = 0.4
        return (
            attempt.cluster_count,
            -abs(attempt.noise_ratio - target_noise),
            -attempt.noise_ratio,
        )

    ranking_pool = acceptable if acceptable else attempts
    ranking_pool.sort(key=_attempt_score, reverse=True)
    return ranking_pool[0]


def project_embeddings(emb_df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    feat_cols = [c for c in emb_df.columns if c.startswith("e")]
    X = emb_df[feat_cols].values
    n_components = min(n_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
    if n_components < 1:
        n_components = 1
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)
    proj = pd.DataFrame(Z, columns=[f"p{i:03d}" for i in range(Z.shape[1])])
    proj["message_id"] = emb_df["message_id"].values
    return proj


def cluster_embeddings_dbscan(
        proj_df: pd.DataFrame,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        metric: str = 'cosine',
        normalize: bool = False
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    feat_cols = [c for c in proj_df.columns if c.startswith("p")]
    if not feat_cols:
        raise ValueError("Projection DataFrame must contain columns starting with 'p'")
    X = proj_df[feat_cols].values
    labels = np.full(len(proj_df), -1, dtype=int)
    info: dict[str, float | int | bool] = {
        "eps": float(eps) if eps is not None else 0.0,
        "min_samples": int(min_samples) if min_samples is not None else 0,
        "cluster_count": 0,
        "noise_ratio": 0.0,
        "auto_tuned": False,
    }
    attempt: Optional[_DbscanAttempt] = None
    if len(proj_df) < 3:
        print("Too few points for DBSCAN; marking everything as noise.")
    else:
        X_proc = X
        if normalize and metric != 'cosine':
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X)
        if eps is not None and min_samples is not None:
            attempt = _fit_dbscan(X_proc, eps, min_samples, metric)
        else:
            auto_eps, auto_min_samples = _auto_dbscan_params(X_proc, metric=metric)
            base_eps = eps if eps is not None else auto_eps
            base_min_samples = min_samples if min_samples is not None else auto_min_samples
            print(
                f"DBSCAN auto base params: eps={auto_eps:.3f}, min_samples={auto_min_samples}, "
                f"data_points={len(X_proc)}"
            )
            attempt = _adaptive_dbscan_search(
                X_proc,
                metric,
                base_eps,
                base_min_samples,
                allow_eps_variation=eps is None,
                allow_min_samples_variation=min_samples is None,
            )
            info["auto_tuned"] = True
        if attempt is not None:
            labels = attempt.labels.copy()
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) > 0:
                label_map = {old: new for new, old in enumerate(sorted(unique_labels), start=1)}
                for idx in range(len(labels)):
                    if labels[idx] != -1:
                        labels[idx] = label_map[labels[idx]]
    cluster_count, noise_ratio = _summarize_labels(labels)
    info["cluster_count"] = cluster_count
    info["noise_ratio"] = noise_ratio
    if attempt is not None:
        info["eps"] = float(attempt.eps)
        info["min_samples"] = int(attempt.min_samples)
        print(
            f"DBSCAN using eps={attempt.eps:.3f}, min_samples={attempt.min_samples}, "
            f"clusters={cluster_count}, noise={noise_ratio:.1%}"
        )
    clusters = pd.DataFrame({
        "message_id": proj_df["message_id"],
        "cluster": labels,
        "is_noise": labels == -1
    })
    return clusters, info


def analyze_cluster_quality(clusters_df: pd.DataFrame) -> dict:
    """???????????? ???????????????? ?????????????????????????? ?????? ??????????????"""
    total_messages = len(clusters_df)
    noise_messages = (clusters_df["cluster"] == -1).sum()
    unique_clusters = clusters_df[clusters_df["cluster"] != -1]["cluster"].nunique()

    cluster_sizes = clusters_df[clusters_df["cluster"] != -1]["cluster"].value_counts()
    avg_cluster_size = cluster_sizes.mean() if len(cluster_sizes) > 0 else 0

    # ???????????????????????? numpy.int64 ?? int ?????? JSON
    cluster_size_distribution = {int(k): int(v) for k, v in cluster_sizes.to_dict().items()}

    quality_metrics = {
        "total_messages": int(total_messages),
        "noise_messages": int(noise_messages),
        "noise_ratio": float(noise_messages / total_messages if total_messages > 0 else 0),
        "unique_clusters": int(unique_clusters),
        "avg_cluster_size": float(avg_cluster_size),
        "largest_cluster_size": int(cluster_sizes.max() if len(cluster_sizes) > 0 else 0),
        "cluster_size_distribution": cluster_size_distribution
    }

    print(f"???????????????? ??????????????????????????: {unique_clusters} ??????????????????, "
          f"??????: {noise_messages}/{total_messages} ({quality_metrics['noise_ratio']:.1%})")

    return quality_metrics


def extract_cluster_terms(
        messages_df: pd.DataFrame,
        clusters_df: pd.DataFrame,
        top_k: int = 12,
        include_noise: bool = False
) -> dict[int, list[str]]:
    df = messages_df.merge(clusters_df, on="message_id", how="inner")
    if not include_noise:
        df = df[df["cluster"] != -1]
    if len(df) == 0:
        print("?????? ???????????? ?????????? merge ?????? ?????? ???????????????? ??? ??????")
        return {}
    df = df.reset_index(drop=True)
    message_id_to_idx = {mid: idx for idx, mid in enumerate(df["message_id"])}
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=None,
        min_df=2,
        max_df=0.95
    )
    try:
        tfidf = vectorizer.fit_transform(df["text"].values)
        vocab = np.array(vectorizer.get_feature_names_out())
    except ValueError as e:
        print(f"???????????? TF-IDF: {e}")
        return {}
    terms: dict[int, list[str]] = {}
    for cid, sub in df.groupby("cluster"):
        if len(sub) == 0:
            continue
        idx = [message_id_to_idx[mid] for mid in sub["message_id"]]
        cluster_tfidf = tfidf[idx]
        scores = cluster_tfidf.mean(axis=0).A1
        top_idx = np.argsort(scores)[-top_k:][::-1]
        cluster_terms = vocab[top_idx].tolist()
        cluster_terms = [term for term in cluster_terms if len(term) > 2]
        terms[int(cid)] = cluster_terms[:top_k]
    return terms


# def _llm_client():
#     from openai import OpenAI
#     client_kwargs = {}
#     if cfg.openai_base_url:
#         client_kwargs["base_url"] = cfg.openai_base_url
#     if cfg.openai_api_key:
#         client_kwargs["api_key"] = cfg.openai_api_key
#     return OpenAI(**client_kwargs)



_LABEL_SYSTEM_PROMPT = (
    "???? ???????????????????? ???????????????? ?? ???????????????????????? ???????????????? ???????????????? ?????? ?????????????????? ???????????????????? ????????????????. "
    "???????????? ???????????????? ???????????????? ?????????????????? ?????????????????? ???????????? ???????? ?? ?????????????? ????????????????."
)

_LABEL_GENERIC_SET = {
    "?????? ???????????????????????? ????????????",
    "?????? ???????????????????????? ????????????.",
    "?????????? ??????",
    "?????????? ?????? ???? ??????????",
    "?????????? ??????????",
    "???????????? ????????????????",
    "?????????? ??????????????",
    "???????????? ??????????",
}

_LABEL_GENERIC_PREFIXES = (
    "?????????? ??????",
    "?????????? ??????????",
    "?????????? ??????????????",
    "????????????",
    "?????????? ??????????",
)

_LABEL_LINE_RE = re.compile(r"(?:cluster\s*)?(?P<id>\d+)[\)\s:\-]+(?P<label>.+)", re.IGNORECASE)
_TICKER_RE = re.compile(r"^[A-Z]{2,6}$")
_LABEL_STRIP_CHARS = '"\'????`""???"'


def _fallback_cluster_label(terms: list[str]) -> str:
    cleaned = [term.strip() for term in terms if term and term.strip()]
    if not cleaned:
        return "?????? ????????????????"
    primary = cleaned[:3]
    return " ".join(primary)


def _pick_anchor_term(terms: list[str]) -> str:
    for term in terms:
        if _TICKER_RE.match(term):
            return term
    for term in terms:
        if term:
            return term.split()[0].title()
    return ""


def _normalize_label(label: str) -> str:
    if not label:
        return ""
    label = label.strip().strip('"????`????????????')
    label = re.sub(r"\s+", " ", label)
    return label.strip()


def _is_generic_label(label: str) -> bool:
    base = label.lower().strip()
    base = base.strip("()")
    if base in _LABEL_GENERIC_SET:
        return True
    return any(base.startswith(prefix) for prefix in _LABEL_GENERIC_PREFIXES)


def _ensure_unique_label(label: str, terms: list[str], cid: int, used: set[str]) -> str:
    candidate = label
    slug = candidate.lower()
    if slug not in used:
        used.add(slug)
        return candidate
    anchor = _pick_anchor_term(terms) or f"?????????????? {cid}"
    base = candidate
    attempt = f"{base} ?? {anchor}"
    slug = attempt.lower()
    counter = 2
    while slug in used:
        attempt = f"{base} ?? {anchor} #{counter}"
        slug = attempt.lower()
        counter += 1
    used.add(slug)
    return attempt


def _parse_llm_labels(raw: str) -> dict[int, str]:
    parsed: dict[int, str] = {}
    if not raw:
        return parsed
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        entries = data.get("labels") or data.get("clusters") or data.get("items")
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                cid = item.get("id") or item.get("cluster_id") or item.get("cluster")
                title = item.get("title") or item.get("label") or item.get("name")
                try:
                    cid_int = int(cid)
                except (TypeError, ValueError):
                    continue
                if isinstance(title, str) and title.strip():
                    parsed[cid_int] = title.strip()
    elif isinstance(data, list):
        for element in data:
            if isinstance(element, dict):
                cid = element.get("id") or element.get("cluster_id")
                title = element.get("title") or element.get("label") or element.get("name")
                try:
                    cid_int = int(cid)
                except (TypeError, ValueError):
                    continue
                if isinstance(title, str) and title.strip():
                    parsed[cid_int] = title.strip()
    if not parsed:
        for line in raw.splitlines():
            line = line.strip(" -*	")
            if not line:
                continue
            match = _LABEL_LINE_RE.match(line)
            if not match:
                continue
            cid = match.group("id")
            label = match.group("label").strip(" -'\"????`??????")
            try:
                cid_int = int(cid)
            except ValueError:
                continue
            if label:
                parsed[cid_int] = label
    return parsed


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def label_clusters_with_llm(cluster_terms: dict[int, list[str]]) -> dict[int, str]:
    cluster_ids = [cid for cid in sorted(cluster_terms) if cid != -1]
    if not cluster_ids:
        return {}
    if not has_yandex_gpt_credentials():
        return {cid: _fallback_cluster_label(cluster_terms.get(cid, [])) for cid in cluster_ids}
    client = get_yandex_gpt_client()
    payload = []
    for cid in cluster_ids:
        terms = cluster_terms.get(cid, [])
        payload.append({
            "id": cid,
            "keywords": terms[:10],
        })
    user_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    user_message = (
        "???????? ?????????????????????? ???????????????? ???????????????? ?? ?????????????????? ??????????????. "
        "?????? ?????????????? id ???????????????? ???????????????????? ?????????????????????????? ???????????????? ???? 3-6 ????????, "
        "?????????????? ???????????????? ?????????????? ?????? ???????? ?????? ??????????????????. ?????????????????? ???????????? ?????? ??????????????, ?????????? ?????? ??????????????. "
        "???? ?????????????????? ???????????????????????? ?????????? ?????????? '?????? ???????????????????????? ????????????' ?????? '?????????? ??????'. "
        "?????????? ?????????? ???????????? ?? JSON: {\"labels\": [{\"id\": int, \"title\": \"...\"}, ...]} ?????? ???????????????????????????? ????????????????????????."
        f"????????????????:{user_payload}"
    )
    try:
        resp = client.chat.completions.create(
            model=cfg.chat_model,
            messages=[
                {"role": "system", "content": _LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=480,
        )
        content = resp.choices[0].message.content.strip()
        llm_labels = _parse_llm_labels(content)
    except Exception as exc:
        print(f"???????????? LLM ?????? ???????????????? ??????????????????: {exc}")
        llm_labels = {}
    labels: dict[int, str] = {}
    used: set[str] = set()
    for cid in cluster_ids:
        terms = cluster_terms.get(cid, [])
        candidate = _normalize_label(llm_labels.get(cid, ""))
        if not candidate or _is_generic_label(candidate):
            candidate = _fallback_cluster_label(terms)
        candidate = candidate[:80]
        candidate = _ensure_unique_label(candidate, terms, cid, used)
        labels[cid] = candidate
    return labels

def run_topic_modeling(
        messages_df: pd.DataFrame,
        emb_df: pd.DataFrame,
        n_components: int = 50,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        metric: str = 'cosine',
        normalize: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str], dict, dict[int, list[str]]]:
    """
    ???????????? ???????????????? ?????????????????????????? ?????????????????????????? ?? DBSCAN
    """
    messages_df["message_id"] = messages_df["message_id"].astype(str)
    emb_df["message_id"] = emb_df["message_id"].astype(str)

    print("???????????????? ??????????????????????...")
    proj_df = project_embeddings(emb_df, n_components=n_components)
    save_parquet(proj_df, cfg.projections_parquet)

    print("DBSCAN ??????????????????????????...")
    clusters_df, clustering_info = cluster_embeddings_dbscan(
        proj_df, eps=eps, min_samples=min_samples, metric=metric, normalize=normalize
    )
    save_parquet(clusters_df, cfg.clusters_parquet)

    print("???????????? ???????????????? ??????????????????????????...")
    quality_metrics = analyze_cluster_quality(clusters_df)
    quality_metrics.update({
        "dbscan_eps": float(clustering_info.get("eps", 0.0)),
        "dbscan_min_samples": int(clustering_info.get("min_samples", 0)),
        "dbscan_cluster_count": int(clustering_info.get("cluster_count", 0)),
        "dbscan_noise_ratio": float(clustering_info.get("noise_ratio", 0.0)),
        "dbscan_auto_tuned": bool(clustering_info.get("auto_tuned", False)),
    })
    save_json(quality_metrics, cfg.artifacts_dir / "quality_metrics.json")

    print(f"?????????????? {quality_metrics['unique_clusters']} ??????????????????, "
          f"??????: {quality_metrics['noise_ratio']:.1%}")

    print("???????????????????? ????????????????...")
    terms = extract_cluster_terms(messages_df, clusters_df)
    save_json({str(k): v for k, v in terms.items()}, cfg.artifacts_dir / "cluster_terms.json")

    print("LLM ???????????????? ??????????????????...")
    labels = label_clusters_with_llm(terms)

    labels_with_noise = labels.copy()
    if -1 in terms:
        labels_with_noise[-1] = "?????????????????????????? ??????????????????"

    save_json({str(k): v for k, v in labels_with_noise.items()}, cfg.cluster_labels_json)

    return proj_df, clusters_df, labels, quality_metrics, terms


def load_topic_artifacts() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict[int, str]], Optional[dict], Optional[dict[int, list[str]]]]:
    """???????????????? ?????????????????????? ???????????????????? ?????????????????????????? ??????????????????????????"""
    proj = load_parquet(cfg.projections_parquet)
    clus = load_parquet(cfg.clusters_parquet)
    labels_raw = load_json(cfg.cluster_labels_json)
    labels = {int(k): v for k, v in labels_raw.items()} if labels_raw else None
    quality_metrics = load_json(cfg.artifacts_dir / "quality_metrics.json") or {}
    cluster_terms_raw = load_json(cfg.artifacts_dir / "cluster_terms.json")
    cluster_terms = {int(k): v for k, v in cluster_terms_raw.items()} if cluster_terms_raw else None
    return proj, clus, labels, quality_metrics, cluster_terms


def suggest_dbscan_params(emb_df: pd.DataFrame, n_components: int = 50, metric: str = 'cosine') -> dict:
    proj_df = project_embeddings(emb_df, n_components=n_components)
    feat_cols = [c for c in proj_df.columns if c.startswith("p")]
    if not feat_cols:
        raise ValueError("Projection DataFrame must contain columns starting with 'p'")
    X = proj_df[feat_cols].values
    if len(proj_df) < 3:
        dims = int(X.shape[1]) if X.ndim == 2 else 0
        return {
            "suggested_eps": 0.3,
            "suggested_min_samples": 3,
            "data_points": int(len(X)),
            "dimensions": dims,
            "auto_tuned": False,
        }
    auto_eps, auto_min_samples = _auto_dbscan_params(X, metric=metric)
    attempt = _adaptive_dbscan_search(
        X,
        metric,
        auto_eps,
        auto_min_samples,
        allow_eps_variation=True,
        allow_min_samples_variation=True,
    )
    return {
        "suggested_eps": float(attempt.eps),
        "suggested_min_samples": int(attempt.min_samples),
        "base_eps": float(auto_eps),
        "base_min_samples": int(auto_min_samples),
        "data_points": int(len(X)),
        "dimensions": int(X.shape[1]),
        "estimated_clusters": int(attempt.cluster_count),
        "estimated_noise_ratio": float(attempt.noise_ratio),
        "auto_tuned": True,
    }

