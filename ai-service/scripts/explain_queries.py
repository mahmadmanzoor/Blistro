"""EXPLAIN ANALYZE the two retrieval queries, before and after their fixes, on a
synthetic multi-user table. Writes evals/results/explain.md.

    uv run python -m scripts.explain_queries --rows 60000 --users 200

Uses its own database (default: blistro_ai_bench on the dev server), never the
app's. Random unit vectors stand in for embeddings: the planner and the index
don't care what the numbers mean, so timings and plans carry over.

Findings it measures:
1. Vector search filtered by user. Without a vector index, every one of the user's
   chunks gets a distance computed and sorted. HNSW fixes the speed, but a plain
   HNSW scan applies the user filter *after* taking ef_search candidates, so it
   returns fewer than k rows (0 for a small user). pgvector 0.8 iterative scans
   (`hnsw.iterative_scan = relaxed_order`) keep scanning until k rows pass; with a
   btree on user_id as well, the planner uses the btree for small users.
2. Full-text search for a rare exact term: without the GIN index on the generated
   tsvector column, every row is re-read.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
import time
from pathlib import Path

import numpy as np
import psycopg
from psycopg.rows import dict_row

from app.config import get_settings

OUT = Path(__file__).resolve().parent.parent / "evals" / "results" / "explain.md"
DIMS = 1536
WORDS = (
    "valve pressure inspection deposit wage employer form schedule quarterly risk "
    "control incident response identity password authenticator travel vehicle home "
    "office expense threshold report audit policy section table federal state payment"
).split()

SCHEMA = """
create extension if not exists vector;
drop table if exists bench_chunks;
create table bench_chunks (
  id bigserial primary key,
  user_id text not null,
  content text not null,
  embedding vector(1536) not null,
  tsv tsvector generated always as (to_tsvector('english', content)) stored
);
"""

VECTOR_SQL = """select id, 1 - (embedding <=> %(v)s::vector) as score from bench_chunks
where user_id = %(u)s order by embedding <=> %(v)s::vector limit 5"""

FULLTEXT_SQL = """select id, ts_rank_cd(tsv, q, 32) as score
from bench_chunks, to_tsquery('english', %(q)s) q
where user_id = %(u)s and tsv @@ q order by score desc limit 30"""


def bench_url(base: str, name: str) -> str:
    return re.sub(r"/[^/?]+(\?|$)", f"/{name}\\1", base, count=1)


def unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, DIMS)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.5f}" for x in v) + "]"


async def seed(conn, rows: int, users: int, rng: np.random.Generator) -> str:
    await conn.execute(SCHEMA)
    # Skewed sizes: a few heavy users, many light ones (like real uploads).
    weights = np.array([1 / (i + 1) for i in range(users)])
    owners = rng.choice(users, size=rows, p=weights / weights.sum())
    pyrng = random.Random(1)
    batch = 2000
    async with conn.cursor() as cur:
        for start in range(0, rows, batch):
            n = min(batch, rows - start)
            vecs = unit_vectors(n, rng)
            await cur.executemany(
                "insert into bench_chunks (user_id, content, embedding) values (%s, %s, %s)",
                [
                    (
                        f"user-{owners[start + i]}",
                        " ".join(pyrng.choices(WORDS, k=60))
                        # A rare exact term (like a model number) in ~0.5% of chunks.
                        + (" XR-200" if pyrng.random() < 0.005 else ""),
                        vec_literal(vecs[i]),
                    )
                    for i in range(n)
                ],
            )
    await conn.execute("create index on bench_chunks (user_id)")
    await conn.execute("analyze bench_chunks")
    # The lightest user: where post-filtering hurts most.
    cur = await conn.execute(
        "select user_id, count(*) n from bench_chunks group by 1 order by 2 asc, 1 limit 1"
    )
    return (await cur.fetchone())["user_id"]


async def explain(conn, sql: str, params: dict, settings: list[str]) -> dict:
    async with conn.transaction():
        for stmt in settings:
            await conn.execute(stmt)
        cur = await conn.execute(f"explain (analyze, buffers, format json) {sql}", params)
        plan = (await cur.fetchone())["QUERY PLAN"][0]
        cur = await conn.execute(sql, params)
        n = len(await cur.fetchall())
    nodes = []

    def walk(node):
        index = f" on {node['Index Name']}" if "Index Name" in node else ""
        nodes.append(node["Node Type"] + index)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan["Plan"])
    return {"ms": plan["Execution Time"], "rows": n, "plan": " > ".join(nodes)}


async def best_of(conn, sql, params, settings, runs=5) -> dict:
    results = [await explain(conn, sql, params, settings) for _ in range(runs)]
    return min(results, key=lambda r: r["ms"])


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=60000)
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--db", default="blistro_ai_bench")
    args = p.parse_args()

    base = get_settings().database_url
    async with await psycopg.AsyncConnection.connect(base, autocommit=True) as admin:
        cur = await admin.execute("select 1 from pg_database where datname = %s", (args.db,))
        if not await cur.fetchone():
            await admin.execute(f'create database "{args.db}"')

    rng = np.random.default_rng(7)
    async with await psycopg.AsyncConnection.connect(
        bench_url(base, args.db), autocommit=True, row_factory=dict_row
    ) as conn:
        started = time.perf_counter()
        light_user = await seed(conn, args.rows, args.users, rng)
        print(f"seeded {args.rows} rows in {time.perf_counter() - started:.0f}s")
        cur = await conn.execute(
            "select count(*) n from bench_chunks where user_id = %s", (light_user,)
        )
        light_n = (await cur.fetchone())["n"]
        heavy_user = "user-0"
        qv = vec_literal(unit_vectors(1, rng)[0])
        lines = [
            "# EXPLAIN ANALYZE: retrieval queries",
            "",
            f"Synthetic table: {args.rows:,} chunks, {args.users} users (skewed sizes), "
            f"1536-dim vectors. Postgres + pgvector, best of 5 runs. Generated by "
            "`scripts/explain_queries.py`.",
            "",
            "## 1. Vector search filtered by user (top 5)",
            "",
            "| case | user (chunks) | time ms | rows returned | plan |",
            "|---|---|---|---|---|",
        ]

        def row(case, user, n, r):
            lines.append(f"| {case} | {user} ({n}) | {r['ms']:.1f} | {r['rows']} | {r['plan']} |")

        # Before: only the btree on user_id; every row of the user gets a distance.
        r = await best_of(conn, VECTOR_SQL, {"v": qv, "u": heavy_user}, [])
        cur = await conn.execute("select count(*) n from bench_chunks where user_id = 'user-0'")
        heavy_n = (await cur.fetchone())["n"]
        row("before: no vector index", heavy_user, heavy_n, r)

        await conn.execute(
            "create index on bench_chunks using hnsw (embedding vector_cosine_ops) "
            "with (m = 16, ef_construction = 64)"
        )
        # Drop the btree so the HNSW cases really use HNSW (with it, the planner
        # picks the btree for small users, shown in the last row).
        await conn.execute("drop index bench_chunks_user_id_idx")
        await conn.execute("analyze bench_chunks")
        plain = ["set local hnsw.ef_search = 40", "set local hnsw.iterative_scan = off"]
        iterative = [
            "set local hnsw.ef_search = 100",
            "set local hnsw.iterative_scan = relaxed_order",
        ]
        force_hnsw = ["set local enable_seqscan = off"]
        r = await best_of(conn, VECTOR_SQL, {"v": qv, "u": heavy_user}, plain + force_hnsw)
        row("HNSW, plain scan", heavy_user, heavy_n, r)
        r = await best_of(conn, VECTOR_SQL, {"v": qv, "u": light_user}, plain + force_hnsw)
        row("HNSW, plain scan", light_user, light_n, r)
        r = await best_of(conn, VECTOR_SQL, {"v": qv, "u": light_user}, iterative + force_hnsw)
        row("after: HNSW + iterative scan", light_user, light_n, r)
        r = await best_of(conn, VECTOR_SQL, {"v": qv, "u": heavy_user}, iterative + force_hnsw)
        row("after: HNSW + iterative scan", heavy_user, heavy_n, r)
        await conn.execute("create index on bench_chunks (user_id)")
        await conn.execute("analyze bench_chunks")
        for user, n in ((light_user, light_n), (heavy_user, heavy_n)):
            r = await best_of(conn, VECTOR_SQL, {"v": qv, "u": user}, iterative)
            row("final: both indexes, planner's choice", user, n, r)

        terms = "xr-200"
        lines += [
            "",
            "## 2. Full-text search filtered by user (top 30)",
            "",
            "| case | time ms | rows returned | plan |",
            "|---|---|---|---|",
        ]
        settings = ["set local max_parallel_workers_per_gather = 0"]
        await conn.execute("drop index if exists bench_chunks_user_id_idx")
        r = await best_of(conn, FULLTEXT_SQL, {"q": terms, "u": heavy_user}, settings)
        lines.append(f"| before: no indexes | {r['ms']:.1f} | {r['rows']} | {r['plan']} |")
        await conn.execute("create index on bench_chunks using gin (tsv)")
        await conn.execute("create index on bench_chunks (user_id)")
        await conn.execute("analyze bench_chunks")
        r = await best_of(conn, FULLTEXT_SQL, {"q": terms, "u": heavy_user}, settings)
        lines.append(
            f"| after: GIN on tsv + btree on user_id | {r['ms']:.1f} | {r['rows']} | {r['plan']} |"
        )
        await conn.execute("drop table bench_chunks")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
