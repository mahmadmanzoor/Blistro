"""Postgres integration tests (pgvector). Need DATABASE_URL pointing at a
throwaway database; skipped when it isn't reachable."""

import uuid

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


# ---------------------------------------------------------------- agent graph
# A scripted stand-in for ChatOpenAI, so the whole graph (routing, grading,
# tool loop, grounding, citations, Postgres checkpointer) runs without OpenAI.

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

import app.agent.graph as graph_mod  # noqa: E402
from app.agent.prompts import MODEL_FAILED, NOT_FOUND  # noqa: E402


class FakeLLM:
    def __init__(self, script: dict):
        self.script = script
        self.schema = None
        self.tools = False

    def with_structured_output(self, schema):
        clone = FakeLLM(self.script)
        clone.schema = schema
        return clone

    def bind_tools(self, tools):
        clone = FakeLLM(self.script)
        clone.tools = True
        return clone

    def with_config(self, **_):
        return self

    async def ainvoke(self, messages, config=None):
        if self.schema is not None:
            value = self.script[self.schema.__name__]
            value = value(messages) if callable(value) else value
            return self.schema(**value)
        if self.script.get("fail"):
            raise RuntimeError("model down")
        if self.tools:
            if not any(isinstance(m, ToolMessage) for m in messages):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "calculator", "args": {"expression": "90 * 4"}, "id": "c1"}
                    ],
                )
            return AIMessage(content="That's 360 days in total.")
        return AIMessage(content=self.script.get("answer", "Every 90 days [S1]."))


def use_fake(monkeypatch, **script):
    script.setdefault("Route", {"route": "retrieve", "standalone_question": "XR-200 inspection"})
    script.setdefault("Grade", {"relevant_ids": ["S1"]})
    script.setdefault("Grounding", {"grounded": True})
    monkeypatch.setattr(graph_mod, "chat_model", lambda **_: FakeLLM(script))


RUN = uuid.uuid4().hex[:8]  # fresh threads each run; checkpoints persist in the DB


def run_config(thread: str) -> dict:
    return {
        "configurable": {
            "thread_id": f"{thread}-{RUN}",
            "user_id": "user-1",
            "document_ids": ["doc-a"],
        },
        "recursion_limit": 25,
    }


async def ask(text: str, thread: str) -> tuple[dict, list[dict]]:
    events: list[dict] = []
    final: dict = {}
    async for part in app.state.graph.astream(
        {"messages": [HumanMessage(content=text)]},
        run_config(thread),
        stream_mode=["custom", "values"],
        version="v2",
    ):
        if part["type"] == "custom":
            events.append(part["data"])
        else:
            final = part["data"]
    return final, events


async def test_graph_retrieve_cites_pages_and_remembers(api, monkeypatch):
    use_fake(monkeypatch)
    final, events = await ask("How often is the XR-200 inspected?", "t-retrieve")
    assert final["answer"].startswith("Every 90 days [p. ")
    assert final["sources"] and final["sources"][0]["document_id"] == "doc-a"
    assert any(e.get("text") == "Searching your documents" for e in events)

    final, _ = await ask("And who does it?", "t-retrieve")
    # Checkpointer kept the first turn: 2 user + 2 assistant messages.
    assert len(final["messages"]) == 4


async def test_graph_not_found_after_one_rewrite(api, monkeypatch):
    use_fake(monkeypatch, Grade={"relevant_ids": []})
    final, events = await ask("What's the CEO's salary?", "t-notfound")
    assert final["answer"] == NOT_FOUND
    assert final["rewrites"] == 1
    assert any(e.get("text") == "Rephrasing the search" for e in events)


async def test_graph_tool_loop(api, monkeypatch):
    use_fake(monkeypatch, Route={"route": "tools", "standalone_question": "90 days x 4"})
    final, _ = await ask("How many days do four inspection cycles take?", "t-tools")
    assert final["answer"] == "That's 360 days in total."
    assert final["tool_steps"] == 2
    assert any(isinstance(m, ToolMessage) and "360" in m.content for m in final["scratch"])


async def test_graph_ungrounded_twice_gives_safe_fallback(api, monkeypatch):
    use_fake(monkeypatch, Grounding={"grounded": False, "unsupported_claims": ["x"]})
    final, events = await ask("How often is the XR-200 inspected?", "t-ungrounded")
    assert final["grounding_attempts"] == 2
    assert "couldn't verify" in final["answer"]
    assert {"type": "reset"} in events


async def test_graph_model_failure_gives_clear_answer(api, monkeypatch):
    use_fake(monkeypatch, fail=True)
    final, _ = await ask("How often is the XR-200 inspected?", "t-fail")
    assert final["answer"] == MODEL_FAILED


async def test_eval_pipeline_runs_end_to_end(api, monkeypatch):
    """Retrieval metrics, the agent run, every judge and the report, with fakes."""
    import evals.run_eval as run_eval
    from evals.build_dataset import Item

    monkeypatch.setattr(run_eval, "EVAL_USER", "user-1")
    use_fake(
        monkeypatch,
        Correctness={"score": 5, "reason": "matches"},
        Faithfulness={"total_claims": 1, "supported_claims": 1},
        GeneratedQuestions={"questions": ["xr-200 inspection interval"], "noncommittal": False},
        ChunkVerdicts={"useful": [True, False]},
        Attribution={"total_statements": 1, "attributed_statements": 1},
    )
    monkeypatch.setattr(run_eval, "chat_model", graph_mod.chat_model)

    item = Item(
        id="q000",
        question="XR-200 inspection interval",
        answer="Every 90 days.",
        quote="inspected every 90 days",
        kind="numeric",
        document_id="doc-a",
        document_name="doc-a.pdf",
        page=2,
        split="dev",
    )
    row = await run_eval.eval_retrieval(
        [item], run_eval.RetrievalConfig("recursive", "hybrid", "none"), k=5
    )
    assert row["recall"] == 1.0 and row["mrr"] > 0

    summary, rows = await run_eval.eval_e2e([item], concurrency=1, doc_ids=["doc-a"])
    assert summary["errors"] == 0, rows
    assert summary["citation_accuracy"] == 1.0 and summary["correct_rate"] == 1.0
    assert summary["context_precision"] == 1.0 and summary["context_recall"] == 1.0
    md = run_eval.report("dev", 1, [row], summary, k=5)
    assert "| recursive/hybrid/rerank=none | 1.000 |" in md and "context recall@5" in md
