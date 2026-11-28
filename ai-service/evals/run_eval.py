"""Run the evals and write a markdown report to evals/results/.

Two parts:

1. Retrieval (cheap, no LLM): every chunking strategy x search mode x reranker.
   A hit means a top-k chunk comes from the gold document and its page range
   covers the gold page. Reports recall@k, MRR@k and search latency.

2. End-to-end (LLM): the full LangGraph agent on the active config. An LLM judge
   scores faithfulness (share of answer claims supported by the retrieved context)
   and correctness against the reference answer. Citation accuracy is computed,
   not judged: does any cited source point at the gold page? Also records
   time-to-first-token.

    uv run python -m evals.run_eval                     # dev split, everything
    uv run python -m evals.run_eval --limit 15          # smoke run (CI)
    uv run python -m evals.run_eval --split test        # held-out numbers
    uv run python -m evals.run_eval --skip-e2e          # retrieval only
    uv run python -m evals.run_eval --langfuse          # also log a Langfuse experiment

Ingest the corpus first: `uv run python -m scripts.ingest_local evals/corpus/*.pdf`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel, Field

from app import db, retrieval
from app.agent.graph import ANSWER_TAG, build_graph
from app.agent.llm import chat_model
from app.agent.prompts import NOT_FOUND
from app.config import get_settings
from evals.build_dataset import Item, load

HERE = Path(__file__).parent
RESULTS = HERE / "results"
EVAL_USER = "eval"


# ---------------------------------------------------------------- retrieval


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: str
    mode: str
    reranker: str  # "none" | "local" | "cohere"

    @property
    def name(self) -> str:
        return f"{self.strategy}/{self.mode}/rerank={self.reranker}"


def page_match(hit_doc: str, start: int, end: int, item: Item) -> bool:
    return hit_doc == item.document_id and start <= item.page <= end


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


async def eval_retrieval(items: list[Item], cfg: RetrievalConfig, k: int) -> dict:
    s = get_settings()
    s.rerank_provider = cfg.reranker if cfg.reranker != "none" else "none"
    hits_at_k, doc_hits, rr, latencies = 0, 0, 0.0, []
    for item in items:
        started = time.perf_counter()
        hits = await retrieval.search(
            EVAL_USER,
            item.question,
            strategy=cfg.strategy,
            mode=cfg.mode,
            k=k,
            use_rerank=cfg.reranker != "none",
        )
        latencies.append((time.perf_counter() - started) * 1000)
        rank = next(
            (
                i
                for i, h in enumerate(hits, start=1)
                if page_match(h.document_id, h.page_start, h.page_end, item)
            ),
            None,
        )
        if rank:
            hits_at_k += 1
            rr += 1 / rank
        if any(h.document_id == item.document_id for h in hits):
            doc_hits += 1
    n = len(items) or 1
    return {
        "config": cfg.name,
        "recall": hits_at_k / n,
        "mrr": rr / n,
        "doc_recall": doc_hits / n,
        "p50_ms": pct(latencies, 0.5),
        "p95_ms": pct(latencies, 0.95),
    }


# ---------------------------------------------------------------- end-to-end


class Faithfulness(BaseModel):
    total_claims: int = Field(description="factual claims in the answer")
    supported_claims: int = Field(description="claims fully supported by the context")


class Correctness(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = same facts as the reference")
    reason: str


FAITH_PROMPT = (
    "Count the factual claims in the ANSWER, then count how many are fully supported by "
    "the CONTEXT. Ignore citations, greetings and statements that the answer wasn't found."
)
CORRECT_PROMPT = (
    "Score how well the ANSWER matches the REFERENCE for the QUESTION, from 1 (wrong or "
    "missing) to 5 (all key facts of the reference, nothing contradicting it). Extra "
    "correct detail is fine."
)


async def run_agent(graph, item: Item, doc_ids: list[str]) -> dict:
    config = {
        "configurable": {
            "thread_id": f"eval:{item.id}",
            "user_id": EVAL_USER,
            "document_ids": doc_ids,
        },
        "recursion_limit": get_settings().agent_recursion_limit,
    }
    started = time.perf_counter()
    ttft = None
    final: dict = {}
    async for part in graph.astream(
        {"messages": [HumanMessage(content=item.question)]},
        config,
        stream_mode=["messages", "values"],
        version="v2",
    ):
        if part["type"] == "messages":
            chunk, meta = part["data"]
            if (
                ttft is None
                and isinstance(chunk, AIMessageChunk)
                and chunk.content
                and ANSWER_TAG in (meta.get("tags") or [])
            ):
                ttft = (time.perf_counter() - started) * 1000
        else:
            final = part["data"]
    return {
        "answer": final.get("answer", ""),
        "context": final.get("context", ""),
        "sources": final.get("sources", []),
        "route": final.get("route"),
        "ttft_ms": ttft,
        "total_ms": (time.perf_counter() - started) * 1000,
    }


async def judge(item: Item, out: dict) -> dict:
    judge_llm = chat_model(model=get_settings().judge_model)
    not_found = out["answer"].strip().startswith(NOT_FOUND[:20])
    faith = None
    if out["context"] and not not_found:
        f = await judge_llm.with_structured_output(Faithfulness).ainvoke(
            [
                ("system", FAITH_PROMPT),
                ("human", f"CONTEXT:\n{out['context'][:20000]}\n\nANSWER:\n{out['answer']}"),
            ]
        )
        faith = f.supported_claims / f.total_claims if f.total_claims else 1.0
    c = await judge_llm.with_structured_output(Correctness).ainvoke(
        [
            ("system", CORRECT_PROMPT),
            (
                "human",
                f"QUESTION: {item.question}\nREFERENCE: {item.answer}\nANSWER: {out['answer']}",
            ),
        ]
    )
    cited_ok = any(
        page_match(s["document_id"], s["page_start"], s["page_end"], item) for s in out["sources"]
    )
    return {
        "faithfulness": faith,
        "correctness": c.score,
        "correct": c.score >= 4,
        "citation_ok": cited_ok,
        "not_found": not_found,
        "judge_reason": c.reason,
    }


async def eval_e2e(items: list[Item], concurrency: int) -> tuple[dict, list[dict]]:
    graph = build_graph()
    doc_ids = sorted({i.document_id for i in load(split="all")})
    sem = asyncio.Semaphore(concurrency)

    async def one(item: Item) -> dict:
        async with sem:
            try:
                out = await run_agent(graph, item, doc_ids)
                scores = await judge(item, out)
            except Exception as exc:  # noqa: BLE001 - count as a failure, keep going
                return {"id": item.id, "error": f"{type(exc).__name__}: {exc}"}
        return {"id": item.id, "question": item.question, **out, **scores}

    rows = await asyncio.gather(*(one(i) for i in items))
    ok = [r for r in rows if "error" not in r]
    faith = [r["faithfulness"] for r in ok if r["faithfulness"] is not None]
    ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"]]
    total = [r["total_ms"] for r in ok]
    n = len(ok) or 1
    summary = {
        "n": len(items),
        "errors": len(rows) - len(ok),
        "correct_rate": sum(r["correct"] for r in ok) / n,
        "mean_correctness": statistics.mean(r["correctness"] for r in ok) if ok else 0,
        "faithfulness": statistics.mean(faith) if faith else None,
        "citation_accuracy": sum(r["citation_ok"] for r in ok) / n,
        "not_found_rate": sum(r["not_found"] for r in ok) / n,
        "ttft_p50_ms": pct(ttft, 0.5),
        "ttft_p95_ms": pct(ttft, 0.95),
        "total_p50_ms": pct(total, 0.5),
        "total_p95_ms": pct(total, 0.95),
    }
    return summary, rows


# ---------------------------------------------------------------- langfuse


def log_langfuse_experiment(items: list[Item], rows: list[dict], run_name: str) -> None:
    """Record the end-to-end run as a Langfuse experiment (dataset + scores),
    so runs can be compared side by side in the UI."""
    s = get_settings()
    if not s.langfuse_enabled:
        print("Langfuse keys not set; skipping experiment upload.")
        return
    from langfuse import Evaluation, Langfuse

    client = Langfuse(
        public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host
    )
    by_id = {r["id"]: r for r in rows if "error" not in r}
    data = [
        {
            "input": {"id": i.id, "question": i.question},
            "expected_output": i.answer,
            "metadata": {"document_id": i.document_id, "page": i.page, "split": i.split},
        }
        for i in items
        if i.id in by_id
    ]

    def task(*, item, **_):
        return by_id[item["input"]["id"]]["answer"]

    def scores(*, input, output, expected_output, **_):
        r = by_id[input["id"]]
        out = [
            Evaluation(name="correctness", value=r["correctness"]),
            Evaluation(name="citation_ok", value=bool(r["citation_ok"]), data_type="BOOLEAN"),
        ]
        if r["faithfulness"] is not None:
            out.append(Evaluation(name="faithfulness", value=r["faithfulness"]))
        return out

    client.run_experiment(
        name="blistro-rag", run_name=run_name, data=data, task=task, evaluators=[scores]
    )
    client.flush()
    print(f"Langfuse experiment logged: blistro-rag / {run_name}")


# ---------------------------------------------------------------- report


def report(split: str, n: int, retrieval_rows: list[dict], e2e: dict | None, k: int) -> str:
    s = get_settings()
    lines = [
        f"# Blistro RAG eval: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        f"Split: `{split}`, {n} questions. Embeddings: `{s.embedding_model}`. "
        f"Chat/judge model: `{s.chat_model}` / `{s.judge_model}`.",
        "",
        f"## Retrieval (k={k})",
        "",
        "A hit = a top-k chunk from the gold document whose page range covers the gold page.",
        "",
        f"| config | recall@{k} | MRR@{k} | doc recall@{k} | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(retrieval_rows, key=lambda r: (-r["recall"], -r["mrr"])):
        lines.append(
            f"| {r['config']} | {r['recall']:.3f} | {r['mrr']:.3f} | {r['doc_recall']:.3f} "
            f"| {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |"
        )
    if e2e:
        f = e2e["faithfulness"]
        lines += [
            "",
            f"## End-to-end agent (active config: {s.active_strategy}/hybrid/"
            f"rerank={s.rerank_provider if s.rerank_enabled else 'none'})",
            "",
            "| metric | value |",
            "|---|---|",
            f"| questions (errors) | {e2e['n']} ({e2e['errors']}) |",
            f"| correct (judge score >= 4/5) | {e2e['correct_rate']:.1%} |",
            f"| mean correctness (1-5) | {e2e['mean_correctness']:.2f} |",
            f"| faithfulness | {'n/a' if f is None else f'{f:.1%}'} |",
            f"| citation points at gold page | {e2e['citation_accuracy']:.1%} |",
            f"| answered 'not found' | {e2e['not_found_rate']:.1%} |",
            f"| time to first token p50 / p95 | {e2e['ttft_p50_ms']:.0f} / "
            f"{e2e['ttft_p95_ms']:.0f} ms |",
            f"| total time p50 / p95 | {e2e['total_p50_ms']:.0f} / {e2e['total_p95_ms']:.0f} ms |",
        ]
    return "\n".join(lines) + "\n"


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--strategies", default="fixed,recursive,semantic")
    p.add_argument("--modes", default="vector,fulltext,hybrid")
    p.add_argument("--rerankers", default="none,local", help="for hybrid mode only")
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--langfuse", action="store_true")
    args = p.parse_args()

    items = load(split=args.split)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("No eval items. Run `python -m evals.build_dataset` first.")

    configs = [
        RetrievalConfig(st, mode, "none")
        for st in args.strategies.split(",")
        for mode in args.modes.split(",")
    ]
    if "hybrid" in args.modes:
        configs += [
            RetrievalConfig(st, "hybrid", rr)
            for st in args.strategies.split(",")
            for rr in args.rerankers.split(",")
            if rr != "none"
        ]

    await db.open_pool()
    try:
        s = get_settings()
        active_reranker = s.rerank_provider
        retrieval_rows = []
        for cfg in configs:
            row = await eval_retrieval(items, cfg, args.k)
            print(f"{row['config']:<40} recall={row['recall']:.3f} mrr={row['mrr']:.3f}")
            retrieval_rows.append(row)
        s.rerank_provider = active_reranker

        e2e_summary, e2e_rows = None, []
        if not args.skip_e2e:
            e2e_summary, e2e_rows = await eval_e2e(items, args.concurrency)
            print(json.dumps(e2e_summary, indent=2))
    finally:
        await db.close_pool()

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    tag = f"{args.split}-{stamp}" + (f"-n{args.limit}" if args.limit else "")
    md = report(args.split, len(items), retrieval_rows, e2e_summary, args.k)
    (RESULTS / f"{tag}.md").write_text(md)
    (RESULTS / f"{tag}.json").write_text(
        json.dumps(
            {"retrieval": retrieval_rows, "e2e": e2e_summary, "rows": e2e_rows},
            indent=2,
            default=str,
        )
    )
    print(md)
    print(f"saved evals/results/{tag}.md")
    if args.langfuse and e2e_rows:
        log_langfuse_experiment(items, e2e_rows, tag)


if __name__ == "__main__":
    asyncio.run(main())
