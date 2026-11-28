"""FastAPI app: `uvicorn app.api:app`.

POST /documents        register a PDF and queue ingestion -> 202 {job_id}
GET  /jobs/{id}        ingestion progress
GET  /documents        the user's documents
GET  /documents/{id}   one document's ingestion status
POST /chat             SSE stream: progress / token / reset / sources / done / error
GET  /health
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field

from app import db, observability, retrieval
from app.agent.graph import ANSWER_TAG, build_graph
from app.auth import User, current_user
from app.config import get_settings
from app.ingest.pipeline import (
    IngestError,
    check_source_url,
    enqueue_job,
    get_job,
    upsert_document,
)

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    await db.open_pool()
    await db.migrate()
    saver = AsyncPostgresSaver(db.pool())
    await saver.setup()  # creates checkpoint tables; idempotent
    app.state.graph = build_graph(checkpointer=saver)
    yield
    observability.flush()
    await db.close_pool()


app = FastAPI(title="Blistro AI service", version="0.1.0", lifespan=lifespan)


class DocumentIn(BaseModel):
    document_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=8, max_length=2000)


class ChatIn(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=10)
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=100)


@app.get("/health")
async def health() -> dict:
    async with db.connection() as conn:
        await conn.execute("select 1")
    return {"ok": True}


@app.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def create_document(body: DocumentIn, user: User = Depends(current_user)) -> dict:
    try:
        check_source_url(body.url)
        await upsert_document(body.document_id, user.id, body.name, body.url)
    except IngestError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    job_id = await enqueue_job(body.document_id, user.id)
    return {"job_id": job_id, "document_id": body.document_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str, user: User = Depends(current_user)) -> dict:
    job = await get_job(job_id, user.id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job


@app.get("/documents")
async def documents(user: User = Depends(current_user)) -> list[dict]:
    return await retrieval.list_documents(user.id)


@app.get("/documents/{document_id}")
async def document(document_id: str, user: User = Depends(current_user)) -> dict:
    doc = await retrieval.get_document(user.id, document_id)
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return doc


async def chat_scope(body: ChatIn, user: User = Depends(current_user)) -> tuple[ChatIn, User]:
    """Runs before the stream opens, so an unknown document is a real 404,
    not an error event in a 200 stream."""
    owned = {d["id"] for d in await retrieval.list_documents(user.id)}
    missing = [d for d in body.document_ids if d not in owned]
    if missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown documents: {missing}")
    return body, user


@app.post("/chat", response_class=EventSourceResponse)
async def chat(
    request: Request, scope: tuple[ChatIn, User] = Depends(chat_scope)
) -> AsyncIterator[ServerSentEvent]:
    body, user = scope
    s = get_settings()
    # Threads are namespaced by user so one user can never resume another's thread.
    thread = f"{user.id}:{body.thread_id or ':'.join(sorted(body.document_ids))}"
    config = {
        "configurable": {
            "thread_id": thread,
            "user_id": user.id,
            "document_ids": body.document_ids,
        },
        "recursion_limit": s.agent_recursion_limit,
        "callbacks": observability.callbacks(),
        "metadata": observability.trace_metadata(user.id, thread, s.active_strategy),
        "run_name": "blistro-chat",
    }
    graph = request.app.state.graph
    started = time.perf_counter()
    first_token: float | None = None

    stream = graph.astream(
        {"messages": [HumanMessage(content=body.message)]},
        config,
        stream_mode=["messages", "custom", "updates"],
        version="v2",
    )
    try:
        async for part in stream:
            if await request.is_disconnected():
                log.info("client disconnected; cancelling run for %s", thread)
                break
            kind, data = part["type"], part["data"]
            if kind == "messages":
                chunk, meta = data
                if (
                    isinstance(chunk, AIMessageChunk)
                    and chunk.content
                    and ANSWER_TAG in (meta.get("tags") or [])
                ):
                    if first_token is None:
                        first_token = time.perf_counter() - started
                    yield ServerSentEvent(event="token", data={"text": chunk.content})
            elif kind == "custom":
                yield ServerSentEvent(event=data.get("type", "progress"), data=data)
            elif kind == "updates":
                for node, update in (data or {}).items():
                    if node == "finalize" and update:
                        yield ServerSentEvent(
                            event="sources", data={"sources": update.get("sources", [])}
                        )
                        yield ServerSentEvent(
                            event="done",
                            data={
                                "answer": update.get("answer", ""),
                                "ttft_ms": round(first_token * 1000) if first_token else None,
                                "total_ms": round((time.perf_counter() - started) * 1000),
                            },
                        )
                    elif node in {
                        "route",
                        "retrieve",
                        "grade",
                        "tool_agent",
                        "tools",
                        "check_grounding",
                        "generate",
                        "rewrite",
                    }:
                        yield ServerSentEvent(event="node", data={"node": node})
    except (asyncio.CancelledError, GeneratorExit):
        log.info("stream cancelled for %s", thread)
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean error event to the client
        log.exception("chat run failed")
        yield ServerSentEvent(
            event="error",
            data={"message": "The assistant failed. Try again.", "detail": type(exc).__name__},
        )
    finally:
        await stream.aclose()
