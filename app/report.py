from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import cfg
from .yandex_gpt import get_yandex_gpt_client, has_yandex_gpt_credentials


def _llm_client():
    return get_yandex_gpt_client()


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def generate_research_report(topic: str, df: pd.DataFrame, max_rows: int = 500) -> str:
    if df.empty:
        return f"No data available for: {topic}"
    sample = df.head(max_rows)
    # Prepare compact evidence
    evidence_lines = [f"- [{r.date}] {r.sender}: {r.text[:300]}" for r in sample.itertuples(index=False)]
    evidence = "\n".join(evidence_lines)

    if not has_yandex_gpt_credentials():
        # Fallback heuristic output
        return (
            f"Research Report: {topic}\n\n"
            f"Sample size: {len(sample)} messages.\n\n"
            f"Key evidence (first {len(sample)}):\n{evidence}\n\n"
            f"Note: Yandex GPT credentials missing; provide FOLDER_ID/KEY_ID/SERVICE_ACCOUNT_ID/PRIVATE_KEY to enable synthesis."
        )

    client = _llm_client()
    system = (
        "You are a senior analyst. Synthesize an analytical report with: Executive summary, "
        "Key insights, Evidence-backed findings (cite quotes), Risks/Gaps, and Actionable Recommendations."
    )
    user = (
        f"Topic: {topic}\n\nContext messages (summarize, do not copy verbatim; cite short quotes):\n{evidence}"
    )
    resp = client.chat.completions.create(
        model=cfg.chat_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def save_report(text: str, filename: str) -> Path:
    path = cfg.reports_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def generate_research_report_with_comments(topic: str, df: pd.DataFrame, extra_comments: str = "", max_rows: int = 500) -> str:
    if df.empty:
        return f"No data available for: {topic}"
    sample = df.head(max_rows)
    evidence_lines = [f"- [{r.date}] {r.sender}: {r.text[:300]}" for r in sample.itertuples(index=False)]
    evidence = "\n".join(evidence_lines)

    if not has_yandex_gpt_credentials():
        base = (
            f"Research Report: {topic}\n\n"
            f"Sample size: {len(sample)} messages.\n\n"
            f"Key evidence (first {len(sample)}):\n{evidence}\n\n"
        )
        if extra_comments:
            base += f"User comments: {extra_comments}\n(LLM disabled; synthesis not applied.)"
        else:
            base += "Note: Yandex GPT credentials missing; provide FOLDER_ID/KEY_ID/SERVICE_ACCOUNT_ID/PRIVATE_KEY to enable synthesis."
        return base

    client = _llm_client()
    system = (
        "You are a senior analyst. Synthesize an analytical report with: Executive summary, "
        "Key insights, Evidence-backed findings (cite quotes), Risks/Gaps, and Actionable Recommendations."
    )
    user = (
        f"Topic: {topic}\n\nExtra comments/instructions: {extra_comments}\n\n"
        f"Context messages (summarize, do not copy verbatim; cite short quotes):\n{evidence}"
    )
    resp = client.chat.completions.create(
        model=cfg.chat_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def chat_about_report(report_text: str, chat_history: list[dict[str, str]], question: str) -> str:
    if not has_yandex_gpt_credentials():
        return "LLM disabled. Provide Yandex GPT credentials to enable chat."
    client = _llm_client()
    system = (
        "You are a helpful analyst assistant. Answer based only on the provided report."
        " If you don't know, say you don't know."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": f"Report:\n{report_text}"}]
    for turn in chat_history:
        messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(model=cfg.chat_model, messages=messages, temperature=0.0)
    return resp.choices[0].message.content.strip()
