from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import httpx


@dataclass(frozen=True)
class LlmConfig:
    model: str
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 60.0


def load_llm_config() -> LlmConfig | None:
    model = os.getenv("PODCAST_RAG_LLM_MODEL") or os.getenv("OPENAI_MODEL")
    api_key = os.getenv("PODCAST_RAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("PODCAST_RAG_LLM_BASE_URL")
    if not model:
        return None
    if not base_url:
        base_url = "https://api.openai.com/v1" if api_key else ""
    if not base_url:
        return None
    return LlmConfig(model=model, base_url=base_url.rstrip("/"), api_key=api_key)


def synthesize_with_llm(question: str, local_brief: str, config: LlmConfig | None = None) -> str:
    resolved = config or load_llm_config()
    if resolved is None:
        raise RuntimeError(
            "LLM mode is not configured. Set PODCAST_RAG_LLM_MODEL and PODCAST_RAG_LLM_BASE_URL "
            "for an OpenAI-compatible endpoint; set PODCAST_RAG_LLM_API_KEY if the endpoint requires authentication."
        )

    headers = {"Content-Type": "application/json"}
    if resolved.api_key:
        headers["Authorization"] = f"Bearer {resolved.api_key}"

    payload: dict[str, Any] = {
        "model": resolved.model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful podcast research assistant. Answer only from the provided evidence. "
                    "Preserve timestamps and episode references when useful. If evidence is weak or noisy, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved evidence and entity connections:\n{local_brief}",
            },
        ],
    }
    response = httpx.post(
        f"{resolved.base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=resolved.timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM endpoint returned an unexpected chat completions response.") from exc
