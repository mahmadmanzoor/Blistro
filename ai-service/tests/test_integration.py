"""Postgres integration tests (pgvector). Need DATABASE_URL pointing at a
throwaway database; skipped when it isn't reachable."""

import httpx
import psycopg
import pytest
import pytest_asyncio

from app import db, retrieval
from app.api import app
from app.config import get_settings
from app.ingest.pipeline import (
    claim_job,
    enqueue_job,
    finish_job,
    get_job,
    ingest_bytes,
    upsert_document,
)


def _db_up() -> bool:
    try:
        psycopg.connect(get_settings().database_url, connect_timeout=2).close()
        return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="test Postgres not reachable")


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def api(sample_pdf_module):
    # The app's lifespan opens the pool, runs migrations and sets up the checkpointer.
    async with app.router.lifespan_context(app):
        async with db.connection() as conn:
            await conn.execute("truncate rag_jobs, rag_chunks, rag_documents cascade")
        await db.migrate()  # second run must be a no-op
        for doc, user in (("doc-a", "user-1"), ("doc-b", "user-2")):
            await upsert_document(doc, user, f"{doc}.pdf", None)
            await ingest_bytes(doc, user, sample_pdf_module, ["fixed", "recursive"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            yield client


@pytest.fixture(scope="module")
def sample_pdf_module():
    from tests.conftest import make_pdf

    body = "The XR-200 valve must be inspected every 90 days by a certified inspector. " * 6
    return make_pdf(
        [
            [("1 Scope", 18, True), ("This manual covers pumps and valves. " * 12, 10, False)],
            [("2 Inspection", 18, True), (body, 10, False)],
        ]
    )


pytestmark = [pytestmark, pytest.mark.asyncio(loop_scope="session")]


async def test_hybrid_finds_exact_term(api):
    hits = await retrieval.search("user-1", "XR-200 inspection interval", strategy="recursive")
    assert hits and hits[0].document_id == "doc-a"
    assert hits[0].page_start == 2
    assert "fulltext" in hits[0].sources


async def test_users_are_isolated(api):
    for mode in ("vector", "fulltext", "hybrid"):
        hits = await retrieval.search("user-1", "XR-200", strategy="fixed", mode=mode, k=20)
        assert hits and {h.document_id for h in hits} == {"doc-a"}, mode
    assert await retrieval.search("nobody", "XR-200", strategy="fixed") == []


async def test_page_and_document_filters(api):
    hits = await retrieval.search(
        "user-1", "pumps and valves", strategy="recursive", pages=(1, 1), k=10
    )
    assert hits and all(h.page_start <= 1 for h in hits)


async def test_hijack_refused(api):
    with pytest.raises(PermissionError):
        await upsert_document("doc-a", "user-2", "mine-now.pdf", None)


async def test_job_lifecycle(api):
    await upsert_document("doc-c", "user-1", "c.pdf", "https://utfs.io/f/c.pdf")
    job_id = await enqueue_job("doc-c", "user-1")
    job = await claim_job()
    assert job and job["id"] == job_id
    assert await claim_job() is None  # already running, not stale
    await finish_job(job, "IngestError: not a PDF")  # permanent: no retry
    row = await get_job(job_id, "user-1")
    assert row["status"] == "failed"
    assert await get_job(job_id, "user-2") is None


async def test_api_auth_and_scoping(api):
    assert (await api.get("/documents")).status_code == 401
    docs = (await api.get("/documents", headers={"X-Dev-User": "user-1"})).json()
    assert {d["id"] for d in docs} >= {"doc-a"} and "doc-b" not in {d["id"] for d in docs}

    bad = await api.post(
        "/documents",
        headers={"X-Dev-User": "user-1"},
        json={"document_id": "x", "name": "x.pdf", "url": "https://169.254.169.254/x"},
    )
    assert bad.status_code == 400

    hijack = await api.post(
        "/documents",
        headers={"X-Dev-User": "user-2"},
        json={"document_id": "doc-a", "name": "a.pdf", "url": "https://utfs.io/f/a.pdf"},
    )
    assert hijack.status_code == 403

    other = await api.post(
        "/chat",
        headers={"X-Dev-User": "user-1"},
        json={"document_ids": ["doc-b"], "message": "hi"},
    )
    assert other.status_code == 404

    mine = await api.get("/documents/doc-a", headers={"X-Dev-User": "user-1"})
    assert mine.status_code == 200 and mine.json()["status"] == "ready"
    theirs = await api.get("/documents/doc-a", headers={"X-Dev-User": "user-2"})
    assert theirs.status_code == 404
