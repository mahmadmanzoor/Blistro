"""Langfuse wiring: tracing callbacks and prompt management, both optional.

With LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY set, every graph run is traced
(each node, LLM call and tool call) and prompts are pulled from Langfuse by
label. Without keys the service runs the same, using the local prompts."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_core.callbacks import BaseCallbackHandler

from app.config import get_settings

log = logging.getLogger(__name__)


@lru_cache
def _client():
    s = get_settings()
    if not s.langfuse_enabled:
        return None
    # The SDK reads these env vars; set them from our settings so .env works too.
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key or "")
    os.environ.setdefault("LANGFUSE_HOST", s.langfuse_host)
    from langfuse import get_client

    return get_client()


def callbacks() -> list[BaseCallbackHandler]:
    if _client() is None:
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def trace_metadata(user_id: str, thread_id: str, strategy: str) -> dict:
    """Run metadata the Langfuse LangChain handler maps onto the trace."""
    return {
        "langfuse_user_id": user_id,
        "langfuse_session_id": thread_id,
        "langfuse_tags": [f"strategy:{strategy}"],
        "chunk_strategy": strategy,
    }


def get_chat_prompt(name: str, fallback: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    """Return (messages, metadata). Pulls the `production` label from Langfuse,
    falling back to the local copy if Langfuse is off or unreachable."""
    client = _client()
    if client is None:
        return fallback, {"prompt_source": "local"}
    try:
        prompt = client.get_prompt(
            name, type="chat", label="production", fallback=fallback, cache_ttl_seconds=300
        )
        messages = [{"role": m["role"], "content": m["content"]} for m in prompt.prompt]
        return messages, {
            "prompt_source": "fallback" if getattr(prompt, "is_fallback", False) else "langfuse",
            "prompt_version": getattr(prompt, "version", None),
        }
    except Exception:  # noqa: BLE001
        log.warning("Langfuse prompt %s unavailable; using local copy", name, exc_info=True)
        return fallback, {"prompt_source": "local"}


def flush() -> None:
    client = _client()
    if client is not None:
        client.flush()
