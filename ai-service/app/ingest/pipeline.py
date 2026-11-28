"""Ingestion: fetch PDF -> parse -> chunk (per strategy) -> embed -> store.

Runs in the worker process, never inside a request. Jobs live in rag_jobs and
are claimed with FOR UPDATE SKIP LOCKED, so several workers can run safely."""

from __future__ import annotations

import logging
import uuid
from urllib.parse import urlparse

import httpx

from app import db
from app.config import ChunkStrategy, get_settings
from app.embeddings import Embedder, get_embedder
from app.ingest.chunking import Chunk, chunk_document
from app.ingest.pdf import parse_pdf

log = logging.getLogger(__name__)


class IngestError(Exception):
    pass


def check_source_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed = get_settings().allowed_source_hosts
    if parsed.scheme != "https" or not any(host == h or host.endswith("." + h) for h in allowed):
        raise IngestError(f"source host not allowed: {host or url!r}")


async def fetch_pdf(url: str) -> bytes:
    check_source_url(url)
    limit = get_settings().max_pdf_bytes
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            buf = bytearray()
            async for part in resp.aiter_bytes():
                buf.extend(part)
                if len(buf) > limit:
                    raise IngestError(f"PDF larger than {limit} bytes")
    if not bytes(buf[:5]).startswith(b"%PDF"):
        raise IngestError("not a PDF")
    return bytes(buf)


async def store_chunks(
    document_id: str, user_id: str, strategy: str, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    async with db.transaction() as conn:
        await conn.execute(
            "delete from rag_chunks where document_id = %s and strategy = %s",
            (document_id, strategy),
        )
        async with conn.cursor() as cur:
            await cur.executemany(
                """insert into rag_chunks
                   (document_id, user_id, strategy, chunk_index, page_start, page_end,
                    heading, content, token_count, embedding)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)""",
                [
                    (
                        document_id,
                        user_id,
                        strategy,
                        i,
                        c.page_start,
                        c.page_end,
                        c.heading,
                        c.content,
                        c.token_count,
                        db.to_vector_literal(v),
                    )
                    for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )


async def ingest_bytes(
    document_id: str,
    user_id: str,
    data: bytes,
    strategies: list[ChunkStrategy] | None = None,
    embedder: Embedder | None = None,
) -> dict[str, int]:
    """Parse once, then chunk + embed + store for each strategy. Returns chunk counts."""
    s = get_settings()
    embedder = embedder or get_embedder()
    strategies = strategies or s.ingest_strategies
    parsed = parse_pdf(data)
    if not parsed.blocks:
        raise IngestError("no extractable text (scanned PDF?)")
    counts: dict[str, int] = {}
    for strategy in strategies:
        chunks = await chunk_document(parsed, strategy, embedder, s.chunk_tokens, s.chunk_overlap)
        vectors = await embedder.embed([c.embed_text for c in chunks])
        await store_chunks(document_id, user_id, strategy, chunks, vectors)
        counts[strategy] = len(chunks)
    async with db.connection() as conn:
        await conn.execute(
            "update rag_documents set page_count = %s, status = 'ready', updated_at = now()"
            " where id = %s",
            (parsed.page_count, document_id),
        )
    return counts


# ---------------------------------------------------------------- documents + jobs


async def upsert_document(
    document_id: str, user_id: str, name: str, source_url: str | None
) -> None:
    async with db.connection() as conn:
        cur = await conn.execute(
            """insert into rag_documents (id, user_id, name, source_url, status)
               values (%s, %s, %s, %s, 'pending')
               on conflict (id) do update
                 set name = excluded.name, source_url = excluded.source_url,
                     status = 'pending', updated_at = now()
                 where rag_documents.user_id = excluded.user_id
               returning id""",
            (document_id, user_id, name, source_url),
        )
        if await cur.fetchone() is None:
            # Same id owned by someone else: refuse rather than hijack it.
            raise PermissionError("document belongs to another user")


async def enqueue_job(document_id: str, user_id: str) -> str:
    job_id = str(uuid.uuid4())
    async with db.connection() as conn:
        await conn.execute(
            "insert into rag_jobs (id, document_id, user_id) values (%s, %s, %s)",
            (job_id, document_id, user_id),
        )
    return job_id


async def get_job(job_id: str, user_id: str) -> dict | None:
    async with db.connection() as conn:
        cur = await conn.execute(
            """select id::text, document_id, status, attempts, error, progress,
                      created_at, updated_at
               from rag_jobs where id = %s and user_id = %s""",
            (job_id, user_id),
        )
        return await cur.fetchone()


async def claim_job() -> dict | None:
    """Claim one queued job (or a 'running' one whose worker died 10+ min ago)."""
    async with db.transaction() as conn:
        cur = await conn.execute(
            """select j.id::text as id, j.document_id, j.user_id, j.attempts, d.source_url
               from rag_jobs j join rag_documents d on d.id = j.document_id
               where j.status = 'queued'
                  or (j.status = 'running' and j.locked_at < now() - interval '10 minutes')
               order by j.created_at
               for update of j skip locked
               limit 1"""
        )
        job = await cur.fetchone()
        if job:
            await conn.execute(
                """update rag_jobs set status = 'running', attempts = attempts + 1,
                          locked_at = now(), updated_at = now() where id = %s""",
                (job["id"],),
            )
            await conn.execute(
                "update rag_documents set status = 'processing', updated_at = now() where id = %s",
                (job["document_id"],),
            )
        return job


async def finish_job(job: dict, error: str | None) -> None:
    max_attempts = get_settings().job_max_attempts
    async with db.transaction() as conn:
        if error is None:
            await conn.execute(
                "update rag_jobs set status = 'done', error = null, updated_at = now()"
                " where id = %s",
                (job["id"],),
            )
            return
        retry = job["attempts"] + 1 < max_attempts and not error.startswith("IngestError")
        await conn.execute(
            "update rag_jobs set status = %s, error = %s, updated_at = now() where id = %s",
            ("queued" if retry else "failed", error[:2000], job["id"]),
        )
        if not retry:
            await conn.execute(
                "update rag_documents set status = 'failed', updated_at = now() where id = %s",
                (job["document_id"],),
            )


async def run_job(job: dict) -> None:
    try:
        if not job["source_url"]:
            raise IngestError("document has no source_url")
        data = await fetch_pdf(job["source_url"])
        counts = await ingest_bytes(job["document_id"], job["user_id"], data)
        log.info("ingested %s: %s", job["document_id"], counts)
        await finish_job(job, None)
    except Exception as exc:  # noqa: BLE001 - recorded on the job row
        log.exception("job %s failed", job["id"])
        await finish_job(job, f"{type(exc).__name__}: {exc}")
