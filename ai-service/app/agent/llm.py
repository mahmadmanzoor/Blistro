from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import get_settings


def chat_model(*, streaming: bool = False, model: str | None = None) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=model or s.chat_model,
        api_key=s.openai_api_key,
        temperature=0,
        timeout=s.llm_timeout_s,
        max_retries=1,  # two attempts in total, then the node's fallback answer
        streaming=streaming,
    )
