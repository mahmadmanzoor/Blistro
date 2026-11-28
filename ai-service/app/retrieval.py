"""Hybrid retrieval: pgvector cosine + Postgres full-text, fused with reciprocal
rank fusion, then optionally reranked by Cohere.

Full-text ranking uses ts_rank_cd. It is not BM25 (no IDF, no length
saturation); it's there to catch exact terms such as model numbers, names and
codes that embeddings blur together.

Every query is filtered by user_id. The user id always comes from the verified
token, never from the model or the request body."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Literal

import cohere

from app import db
from app.config import ChunkStrategy, get_settings
from app.embeddings import Embedder, get_embedder

log = logging.getLogger(__name__)

Mode = Literal["vector", "fulltext", "hybrid"]


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    heading: str | None
    content: str
    score: float
    sources: tuple[str, ...] = field(default_factory=tuple)

    def cite(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"{self.document_name}, {pages}"


_WORD = re.compile(r"[A-Za-z0-9]+")

_COLUMNS = """c.id, c.document_id, d.name as document_name, c.page_start, c.page_end,
              c.heading, c.content"""


def _scope_sql(document_ids: list[str] | None, pages: tuple[int, int] | None) -> tuple[str, list]:
    sql, params = "", []
    if document_ids:
        sql += " and c.document_id = any(%s)"
        params.append(document_ids)
    if pages:
        sql += " and c.page_end >= %s and c.page_start <= %s"
        params.extend(pages)
    return sql, params


async def vector_search(
    user_id: str,
    query_vec: list[float],
    strategy: ChunkStrategy,
    limit: int,
    document_ids: list[str] | None = None,
    pages: tuple[int, int] | None = None,
) -> list[Hit]:
    scope, scope_params = _scope_sql(document_ids, pages)
    async with db.transaction() as conn:
        # With filters, a plain HNSW scan can return fewer than `limit` rows.
        # Iterative scans (pgvector >= 0.8) keep scanning until enough rows pass.
        await conn.execute("set local hnsw.iterative_scan = relaxed_order")
        await conn.execute("set local hnsw.ef_search = 100")
        vec = db.to_vector_literal(query_vec)
        cur = await conn.execute(
            f"""select {_COLUMNS}, 1 - (c.embedding <=> %s::vector) as score
                from rag_chunks c join rag_documents d on d.id = c.document_id
                where c.user_id = %s and c.strategy = %s {scope}
                order by c.embedding <=> %s::vector
                limit %s""",
            [vec, user_id, strategy, *scope_params, vec, limit],
        )
        rows = await cur.fetchall()
    return [_hit(r, "vector") for r in rows]


async def fulltext_search(
    user_id: str,
    query: str,
    strategy: ChunkStrategy,
    limit: int,
    document_ids: list[str] | None = None,
    pages: tuple[int, int] | None = None,
) -> list[Hit]:
    scope, scope_params = _scope_sql(document_ids, pages)
    # websearch_to_tsquery ANDs every term, which rarely matches a long question.
    # Rank by an OR of the terms, with a bonus for chunks matching all of them.
    words = _WORD.findall(query)
    if not words:
        return []
    loose = " | ".join(dict.fromkeys(w.lower() for w in words))
    async with db.connection() as conn:
        cur = await conn.execute(
            f"""with q as (
                  select websearch_to_tsquery('english', %s) as strict,
                         to_tsquery('english', %s) as loose
                )
                select {_COLUMNS},
                       ts_rank_cd(c.tsv, q.loose, 32) + 0.5 * ts_rank_cd(c.tsv, q.strict, 32)
                         as score
                from rag_chunks c join rag_documents d on d.id = c.document_id, q
                where c.user_id = %s and c.strategy = %s {scope}
                  and c.tsv @@ q.loose
                order by score desc
                limit %s""",
            [query, loose, user_id, strategy, *scope_params, limit],
        )
        rows = await cur.fetchall()
    return [_hit(r, "fulltext") for r in rows]


def _hit(row: dict, source: str) -> Hit:
    return Hit(
        chunk_id=row["id"],
        document_id=row["document_id"],
        document_name=row["document_name"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        heading=row["heading"],
        content=row["content"],
        score=float(row["score"]),
        sources=(source,),
    )


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """score(d) = sum over lists of 1 / (k + rank). Robust to the two scores
    living on different scales, which is why it's used instead of a weighted sum."""
    fused: dict[int, float] = {}
    best: dict[int, Hit] = {}
    sources: dict[int, set[str]] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(hit.chunk_id, hit)
            sources.setdefault(hit.chunk_id, set()).update(hit.sources)
    ranked = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [
        replace(best[cid], score=fused[cid], sources=tuple(sorted(sources[cid]))) for cid in ranked
    ]


_cohere: cohere.AsyncClientV2 | None = None
_cross_encoder = None


def _local_scores(query: str, docs: list[str]) -> list[float]:
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # optional extra: local-rerank

        _cross_encoder = CrossEncoder(get_settings().local_rerank_model, max_length=512)
    return [float(x) for x in _cross_encoder.predict([(query, d) for d in docs], batch_size=16)]


async def rerank(query: str, hits: list[Hit], top_n: int) -> list[Hit]:
    s = get_settings()
    global _cohere
    provider = s.rerank_provider
    if provider == "cohere" and not s.cohere_api_key:
        provider = "none"
    if not hits or provider == "none":
        return hits[:top_n]
    docs = [f"{h.heading}\n{h.content}" if h.heading else h.content for h in hits]
    try:
        if provider == "local":
            # CPU/GPU-bound: run off the event loop.
            scores = await asyncio.to_thread(_local_scores, query, docs)
            order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)[:top_n]
            return [replace(hits[i], score=scores[i]) for i in order]
        if _cohere is None:
            _cohere = cohere.AsyncClientV2(api_key=s.cohere_api_key, timeout=20)
        resp = await _cohere.rerank(model=s.rerank_model, query=query, documents=docs, top_n=top_n)
        return [replace(hits[r.index], score=r.relevance_score) for r in resp.results]
    except Exception:  # noqa: BLE001 - reranking is an optimisation; fall back to fused order
        log.warning("rerank failed; using fused order", exc_info=True)
        return hits[:top_n]


async def search(
    user_id: str,
    query: str,
    *,
    strategy: ChunkStrategy | None = None,
    mode: Mode = "hybrid",
    k: int | None = None,
    use_rerank: bool | None = None,
    document_ids: list[str] | None = None,
    pages: tuple[int, int] | None = None,
    embedder: Embedder | None = None,
) -> list[Hit]:
    s = get_settings()
    strategy = strategy or s.active_strategy
    k = k or s.retrieval_k
    use_rerank = s.rerank_enabled if use_rerank is None else use_rerank
    n = s.retrieval_candidates if (use_rerank or mode == "hybrid") else k
    embedder = embedder or get_embedder()

    async def vec() -> list[Hit]:
        [qv] = await embedder.embed([query])
        return await vector_search(user_id, qv, strategy, n, document_ids, pages)

    if mode == "vector":
        hits = await vec()
    elif mode == "fulltext":
        hits = await fulltext_search(user_id, query, strategy, n, document_ids, pages)
    else:
        v, f = await asyncio.gather(
            vec(), fulltext_search(user_id, query, strategy, n, document_ids, pages)
        )
        hits = reciprocal_rank_fusion([v, f], k=s.rrf_k)

    if use_rerank:
        return await rerank(query, hits, k)
    return hits[:k]


async def get_page_text(user_id: str, document_id: str, page: int) -> tuple[str, str] | None:
    """Full text of one page, rebuilt from the active strategy's chunks."""
    s = get_settings()
    async with db.connection() as conn:
        cur = await conn.execute(
            """select d.name, c.heading, c.content from rag_chunks c
               join rag_documents d on d.id = c.document_id
               where c.user_id = %s and c.document_id = %s and c.strategy = %s
                 and c.page_start <= %s and c.page_end >= %s
               order by c.chunk_index""",
            (user_id, document_id, s.active_strategy, page, page),
        )
        rows = await cur.fetchall()
    if not rows:
        return None
    parts = [(f"{r['heading']}\n" if r["heading"] else "") + r["content"] for r in rows]
    return rows[0]["name"], "\n\n".join(parts)


async def get_document_chunks(user_id: str, document_id: str, max_chunks: int = 400) -> list[dict]:
    s = get_settings()
    async with db.connection() as conn:
        cur = await conn.execute(
            """select d.name, c.page_start, c.page_end, c.heading, c.content
               from rag_chunks c join rag_documents d on d.id = c.document_id
               where c.user_id = %s and c.document_id = %s and c.strategy = %s
               order by c.chunk_index limit %s""",
            (user_id, document_id, s.active_strategy, max_chunks),
        )
        return await cur.fetchall()


async def list_documents(user_id: str) -> list[dict]:
    async with db.connection() as conn:
        cur = await conn.execute(
            """select id, name, page_count, status, created_at from rag_documents
               where user_id = %s order by created_at desc""",
            (user_id,),
        )
        return await cur.fetchall()


async def get_document(user_id: str, document_id: str) -> dict | None:
    async with db.connection() as conn:
        cur = await conn.execute(
            """select d.id, d.name, d.page_count, d.status, d.updated_at,
                      (select j.error from rag_jobs j where j.document_id = d.id
                       order by j.created_at desc limit 1) as error
               from rag_documents d where d.user_id = %s and d.id = %s""",
            (user_id, document_id),
        )
        return await cur.fetchone()
