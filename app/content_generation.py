from __future__ import annotations

from enum import Enum

from textwrap import dedent

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import cfg
from .yandex_gpt import (
    YandexGPTError,
    get_yandex_gpt_client,
    has_yandex_gpt_credentials,
)


class DraftMode(str, Enum):
    ARTICLE = "article"
    SOCIAL = "social"
    MARKET = "market"


def _compose_prompt(topic: str, mode: DraftMode) -> tuple[str, str, float, int]:
    if mode == DraftMode.SOCIAL:
        system = (
            "You are a Russian-speaking social media strategist who writes engaging but accurate long-form posts."
        )
        user = dedent(
            f"""
            Cluster context:
            {topic}

            Task: Write an informal Russian post for LinkedIn/VK/Telegram audiences.
            Structure:
            1. Hook explaining why the topic matters now.
            2. Three to four concise insights or micro-stories rooted in the provided context; if data is missing, flag it openly.
            3. Practical takeaway for readers.
            4. Conversational question or call for feedback.
            Requirements: 150-220 words, light use of emojis, no invented statistics, finish with up to three relevant hashtags.
            Language: Russian.
            """
        ).strip()
        return system, user, 0.5, 450

    if mode == DraftMode.MARKET:
        system = (
            "You are a senior financial analyst delivering precise market impact briefings in Russian."
        )
        user = dedent(
            f"""
            Cluster context:
            {topic}

            Task: Prepare a market impact brief in Russian.
            Sections:
            1. Key signals and demand/supply drivers.
            2. Potential impact on the market (pricing, volume, affected segments).
            3. Opportunities for investors or companies.
            4. Constraints and risks (regulatory, competitive, data limitations).
            5. 3-6 month outlook with scenarios and recommended actions.
            Requirements: rely only on provided facts or clearly-stated industry patterns; if assumptions are required, label them explicitly.
            Tone must remain analytical. Length: 350-500 words.
            Language: Russian.
            """
        ).strip()
        return system, user, 0.25, 650

    system = (
        "You are a corporate communications expert who writes formal Russian letters summarizing analytical insights."
    )
    user = dedent(
        f"""
        Cluster context:
        {topic}

        Task: Draft a formal Russian letter addressed to business stakeholders.
        Structure:
        1. Greeting and concise statement of purpose.
        2. Situation overview summarising what the cluster signals.
        3. Detailed findings: business impact, validated data points, related trends.
        4. Recommendations and next steps, noting owners or data needs when possible.
        5. Closing paragraph with invitation for follow-up.
        Requirements: 450-600 words, strict professional tone, do not invent figures; if data is missing, explicitly say further research is required.
        Language: Russian.
        """
    ).strip()
    return system, user, 0.3, 700


@retry(wait=wait_exponential(min=1, max=15), stop=stop_after_attempt(3))
def _request_llm(topic: str, mode: DraftMode) -> str:
    client = get_yandex_gpt_client()
    system, user, temperature, max_tokens = _compose_prompt(topic, mode)
    try:
        response = client.chat.completions.create(
            model=cfg.chat_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise YandexGPTError(f"LLM draft request failed: {exc}") from exc
    return response.choices[0].message.content.strip()


def _fallback(topic: str, mode: DraftMode) -> str:
    if mode == DraftMode.SOCIAL:
        lines = [
            f"Topic: {topic}",
            "Post outline:",
            "- Hook that explains why the topic matters now.",
            "- Three concise insights grounded in available data.",
            "- Practical takeaway for the audience.",
            "- Question that invites comments.",
            "Hashtags: #trend #insight #discussion",
            "(Enable Yandex GPT credentials for a full draft.)",
        ]
        return "\n".join(lines)

    if mode == DraftMode.MARKET:
        lines = [
            f"Context: {topic}",
            "Market impact brief template:",
            "1) Signals and demand/supply drivers.",
            "2) Expected effects on pricing, volume, segments.",
            "3) Opportunities for investors or operators.",
            "4) Constraints, regulatory or competitive risks.",
            "5) 3-6 month scenarios and recommended next steps.",
            "(Enable Yandex GPT credentials for a detailed analysis.)",
        ]
        return "\n".join(lines)

    lines = [
        f"Subject: {topic}",
        "Letter outline:",
        "- Formal greeting and purpose.",
        "- Summary of current situation from the cluster signals.",
        "- Detailed findings and implications.",
        "- Recommended actions and owners.",
        "- Closing paragraph with follow-up request.",
        "(Enable Yandex GPT credentials for a complete letter.)",
    ]
    return "\n".join(lines)

def generate_content_draft(topic: str, mode: DraftMode = DraftMode.ARTICLE) -> tuple[str, bool]:
    cleaned = topic.strip()
    if not cleaned:
        raise ValueError("Topic must not be empty.")

    if not has_yandex_gpt_credentials():
        return _fallback(cleaned, mode), False

    try:
        text = _request_llm(cleaned, mode)
    except YandexGPTError:
        return _fallback(cleaned, mode), False

    return (text or _fallback(cleaned, mode)), True

