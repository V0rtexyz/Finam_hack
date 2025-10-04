from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import cfg
from .yandex_gpt import (
    YandexGPTError,
    get_yandex_gpt_client,
    has_yandex_gpt_credentials,
)

__all__ = [
    "make_text_key",
    "generate_investor_focus",
]

_CACHE_PATH = cfg.artifacts_dir / "investor_focus_cache.json"
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, str] = {}
_CACHE_LOADED = False
_CACHE_DIRTY = False

_SYSTEM_PROMPT = (
    "You are an equity research associate summarizing Russian financial news for professional investors. "
    "Extract catalysts, quantify numbers, highlight expected market impact, and surface trending themes. "
    "Always answer in Russian and keep the tone analytical."
)

_USER_TEMPLATE = (
    "Сообщение из финансового канала:\n"
    "{content}\n\n"
    "Фокус — лаконичное резюме сути новости (до 25 слов).\n"
    "Влияние — ожидаемое направление и масштаб эффекта на активы или рынок (рост/падение/нейтрально).\n"
    "Перспективность — оцени срочность сигнала как высокая, средняя или низкая.\n"
    "Теги/тикеры — перечисли через запятую tickers, компании, сектора, хештеги или оставь \"-\".\n"
    "Если информации нет, оставь после двоеточия дефис. Не добавляй лишних секций. Ты должен стараться избегать общих слов, пиши по существу"
)

_SECTION_ORDER = [
    "Фокус",
    "Катализатор",
    "Влияние",
    "Горячесть",
    "Теги/тикеры",
]

_HEAT_KEYWORDS_HIGH = {
    "срочно",
    "экстренно",
    "немедленно",
    "внимание",
    "критично",
    "urgent",
    "обвал",
    "паника",
}

_HEAT_KEYWORDS_MEDIUM = {
    "сегодня",
    "в ближайшее время",
    "ожидается",
    "прогноз",
    "возможен",
    "получит",
    "ожидание",
    "сильный",
    "рост",
    "падение",
    "просадка",
    "jump",
    "surge",
}

_IMPACT_POSITIVE = re.compile(r"\b(рост|выраст|улучш|подорож|повыш|strength|upside)\w*", re.IGNORECASE)
_IMPACT_NEGATIVE = re.compile(r"\b(паден|сниж|ухудш|подешев|просроч|pressure|selloff)\w*", re.IGNORECASE)
_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")
_HASHTAG_PATTERN = re.compile(r"#([\w-]{2,30})")


def _ensure_cache_loaded() -> None:
    global _CACHE_LOADED, _CACHE
    if _CACHE_LOADED:
        return
    if _CACHE_PATH.exists():
        try:
            data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _CACHE = {str(k): str(v) for k, v in data.items()}
        except Exception:
            _CACHE = {}
    _CACHE_LOADED = True


def _mark_cache_dirty() -> None:
    global _CACHE_DIRTY
    _CACHE_DIRTY = True


def _save_cache() -> None:
    global _CACHE_DIRTY
    with _CACHE_LOCK:
        if not _CACHE_LOADED or not _CACHE_DIRTY:
            return
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _CACHE_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(_CACHE, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(_CACHE_PATH)
        _CACHE_DIRTY = False


def make_text_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _format_summary(
    *,
    focus: str,
    catalyst: str,
    impact: str,
    heat: str,
    tags: str,
) -> str:
    values = {
        "Фокус": focus or "-",
        "Катализатор": catalyst or "-",
        "Влияние": impact or "-",
        "Горячесть": heat or "-",
        "Теги/тикеры": tags or "-",
    }
    return "; ".join(f"{section}: {values[section].strip()}" for section in _SECTION_ORDER)


def _sanitize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


class _FocusLLM:
    def __init__(self) -> None:
        self._enabled = has_yandex_gpt_credentials()
        self._client = get_yandex_gpt_client() if self._enabled else None

    def _prepare_input(self, text: str) -> str:
        collapsed = _sanitize_text(text)
        if len(collapsed) > 2800:
            return collapsed[:2800] + "..."
        return collapsed

    def _detect_heat(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in _HEAT_KEYWORDS_HIGH):
            return "высокая"
        if any(word in lowered for word in _HEAT_KEYWORDS_MEDIUM):
            return "средняя"
        return "низкая"

    def _guess_impact_direction(self, text: str) -> str:
        if _IMPACT_POSITIVE.search(text):
            return "ожидается рост"
        if _IMPACT_NEGATIVE.search(text):
            return "ожидается падение"
        return "нейтрально"

    def _extract_tags(self, text: str) -> str:
        tickers = set(_TICKER_PATTERN.findall(text))
        hashtags = {tag for tag in _HASHTAG_PATTERN.findall(text)}
        combined = sorted({*tickers, *hashtags})
        return ", ".join(combined)

    def _fallback(self, text: str) -> str:
        stripped = re.sub(r"https?://\S+", "", text, flags=re.IGNORECASE)
        base = _sanitize_text(stripped)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", base) if s.strip()]
        focus = sentences[0][:220] if sentences else base[:220]
        catalyst = sentences[1][:180] if len(sentences) > 1 else "-"
        impact = self._guess_impact_direction(base)
        heat = self._detect_heat(base)
        tags = self._extract_tags(text)
        return _format_summary(
            focus=focus,
            catalyst=catalyst,
            impact=impact,
            heat=heat,
            tags=tags,
        )

    def _postprocess(self, summary: str) -> str:
        if not summary:
            return ""
        merged = _sanitize_text(summary.replace("\n", " "))
        sections: dict[str, str] = {section: "" for section in _SECTION_ORDER}
        pattern = re.compile(r"(Фокус|Катализатор|Влияние|Горячесть|Теги/тикеры)\s*:\s*([^;]+)")
        for match in pattern.finditer(merged):
            name, value = match.groups()
            sections[name] = value.strip()
        return _format_summary(
            focus=sections.get("Фокус", ""),
            catalyst=sections.get("Катализатор", ""),
            impact=sections.get("Влияние", ""),
            heat=sections.get("Горячесть", ""),
            tags=sections.get("Теги/тикеры", ""),
        )

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(4), reraise=True)
    def _invoke(self, prompt_text: str) -> str:
        if not self._client:
            raise YandexGPTError("Yandex GPT client is not available")
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(content=prompt_text)},
        ]
        response = self._client.chat.completions.create(
            model=cfg.chat_model,
            messages=messages,
            temperature=0.1,
            max_tokens=380,
        )
        content = response.choices[0].message.content.strip()
        if not content:
            raise YandexGPTError("Empty response from Yandex GPT")
        return content

    def process(self, text: str) -> str:
        prepared = self._prepare_input(text)
        if not prepared:
            return ""
        if not self._client:
            return self._fallback(prepared)
        try:
            raw = self._invoke(prepared)
        except Exception:
            return self._fallback(prepared)
        return self._postprocess(raw)

    def fallback_only(self, text: str) -> str:
        prepared = self._prepare_input(text)
        if not prepared:
            return ""
        return self._fallback(prepared)


def generate_investor_focus(text_map: Mapping[str, str], *, max_workers: int = 4) -> dict[str, str]:
    if not text_map:
        return {}
    _ensure_cache_loaded()
    focus_map: dict[str, str] = {}
    missing: dict[str, str] = {}

    with _CACHE_LOCK:
        for key, text in text_map.items():
            cached = _CACHE.get(key)
            if cached is not None:
                focus_map[key] = cached
            elif text:
                missing[key] = text
            else:
                focus_map[key] = ""

    if missing:
        processor = _FocusLLM()
        workers = max(1, int(max_workers))
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            for key, text in missing.items():
                futures[pool.submit(processor.process, text)] = key
            for future in as_completed(futures):
                key = futures[future]
                try:
                    summary = future.result()
                except Exception:
                    summary = processor.fallback_only(missing[key])
                focus_map[key] = summary
                with _CACHE_LOCK:
                    _CACHE[key] = summary
                    _mark_cache_dirty()
        _save_cache()

    ordered: dict[str, str] = {}
    for key in text_map:
        ordered[key] = focus_map.get(key, "")
    return ordered
