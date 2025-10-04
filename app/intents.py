from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import cfg
from .embeddings import embed_texts
from .yandex_gpt import get_yandex_gpt_client, has_yandex_gpt_credentials


def _llm_client():
    return get_yandex_gpt_client()


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def propose_intents_from_topics(topic_labels: dict[int, str]) -> list[str]:
    if not has_yandex_gpt_credentials():
        # Heuristic fallback: use labels themselves as intents
        return list(topic_labels.values())
    client = _llm_client()
    prompt = (
        """
Ты — эксперт по прикладной аналитике и формулировке пользовательских намерений для анализа тенденций.
Твоя задача — по списку тем сгенерировать набор коротких императивных намерений, каждое из которых задаёт конкретный «срез» данных для более глубокого исследования.

Входные данные:
ТЕМЫ:
{TOPIC_TITLES (по одному названию темы на строку)}

КОНТЕКСТ ПОЛЬЗОВАТЕЛЯ: {AUDIENCE/ROLE, например: инвестаналитик, PR-специалист}
ЦЕЛЬ ИССЛЕДОВАНИЯ: {GOAL, например: выявить ранние сигналы тренда для управленческих решений}
ГОРИЗОНТ: {TIME_HORIZON, например: 7/30/90 дней}
ЯЗЫК ВЫВОДА: {LANG, например: ru}

Что считать «намерением»

Короткая императивная команда (2–7 слов)

Определяет конкретный фильтр, срез или операцию над данными

Может включать: время, географию, канал, аудиторию, тональность, формат, метрику, источник, стадию воронки, уникальность/новизну, репост/оригинал, ценовой диапазон, категорию продукта, уровень вовлечённости, рост/спад и пр.

Правила вывода

5–12 строк

Каждое намерение на отдельной строке

Без нумерации, кавычек и точек

Пример вывода (для иллюстрации):
Отфильтровать упоминания за последние 7 дней
Сравнить позитивные и негативные отклики
Выделить уникальные новые формулировки
Показать темы с наибольшим ростом цитирования
Сгруппировать по регионам России
Отобрать посты лидеров мнений
        """
    )
    labels_text = "\n".join(f"- {t}" for _, t in sorted(topic_labels.items()))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": labels_text},
    ]
    resp = client.chat.completions.create(model=cfg.chat_model, messages=messages, temperature=0.3)
    text = resp.choices[0].message.content.strip()
    intents = [l.strip("- ") for l in text.splitlines() if l.strip()]
    return intents


def zero_shot_filter(
    messages_df: pd.DataFrame,
    intents: list[str],
    message_embeddings: np.ndarray,
    intent_embeddings: np.ndarray,
    top_k: int = 200,
    min_score: float = 0.25,
) -> dict[str, pd.DataFrame]:
    if not intents or intent_embeddings.size == 0 or message_embeddings.size == 0:
        return {intent: pd.DataFrame(columns=list(messages_df.columns) + ["intent_score"]) for intent in intents}
    scores = cosine_similarity(intent_embeddings, message_embeddings)
    results: dict[str, pd.DataFrame] = {}
    for idx, intent in enumerate(intents):
        s = scores[idx]
        top_idx = np.argsort(s)[-top_k:][::-1]
        keep = top_idx[s[top_idx] >= min_score]
        sub = messages_df.iloc[keep].copy()
        sub["intent_score"] = s[keep]
        results[intent] = sub.sort_values("intent_score", ascending=False)
    return results


def suggest_relevant_intents(primary_query: str, candidate_intents: list[str], top_n: int = 5) -> list[str]:
    if not candidate_intents:
        return []
    if not primary_query:
        return candidate_intents[:top_n]
    try:
        vecs = embed_texts([primary_query] + candidate_intents, text_type="query")
    except Exception:
        return candidate_intents[:top_n]
    if len(vecs) != len(candidate_intents) + 1:
        return candidate_intents[:top_n]
    q_vec = vecs[0:1]
    i_vecs = vecs[1:]
    sims = cosine_similarity(q_vec, i_vecs)[0]
    order = np.argsort(sims)[::-1][:top_n]
    return [candidate_intents[i] for i in order]


def embed_intents(intents: list[str]) -> np.ndarray:
    if not intents:
        return np.empty((0, 0), dtype=np.float32)
    return embed_texts(intents, text_type="query")
