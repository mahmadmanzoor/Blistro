"""Agent tools. The user id and the documents in scope come from the run config
(set by the API from the verified token), never from tool arguments, so the
model can't widen its own access."""

from __future__ import annotations

import ast
import asyncio
import operator

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from app import retrieval
from app.config import get_settings


def _scope(config: RunnableConfig) -> tuple[str, list[str]]:
    conf = config.get("configurable", {})
    user_id = conf.get("user_id")
    if not user_id:
        raise PermissionError("no user in run config")
    return user_id, list(conf.get("document_ids") or [])


def _check_doc(document_id: str, allowed: list[str]) -> None:
    if allowed and document_id not in allowed:
        raise ValueError(f"document {document_id!r} is not in this chat's scope: {allowed}")


def _progress(text: str) -> None:
    try:
        get_stream_writer()({"type": "progress", "text": text})
    except Exception:  # noqa: BLE001 - not streaming (e.g. evals)
        pass


def _format_hits(hits: list[retrieval.Hit]) -> str:
    if not hits:
        return "No matching passages."
    return "\n\n".join(f"[{h.cite()}]\n{h.content}" for h in hits)


async def _with_timeout(coro):
    return await asyncio.wait_for(coro, timeout=get_settings().tool_timeout_s)


@tool
async def search_documents(
    query: str,
    document_id: str | None = None,
    *,
    config: RunnableConfig,
) -> str:
    """Search the user's documents for passages relevant to `query`. Optionally
    restrict to one document_id. Returns passages labelled with document and page."""
    user_id, allowed = _scope(config)
    if document_id:
        _check_doc(document_id, allowed)
    _progress(f"Searching for “{query}”")
    hits = await _with_timeout(
        retrieval.search(
            user_id, query, document_ids=[document_id] if document_id else allowed or None
        )
    )
    return _format_hits(hits)


@tool
async def get_page(
    document_id: str,
    page: int,
    *,
    config: RunnableConfig,
) -> str:
    """Return the full text of one page (1-based) of a document."""
    user_id, allowed = _scope(config)
    _check_doc(document_id, allowed)
    _progress(f"Reading page {page}")
    found = await _with_timeout(retrieval.get_page_text(user_id, document_id, page))
    if not found:
        return f"Page {page} not found in that document."
    name, text = found
    return f"[{name}, p. {page}]\n{text}"


@tool
async def summarize_document(
    document_id: str,
    focus: str | None = None,
    *,
    config: RunnableConfig,
) -> str:
    """Summarise a whole document (optionally focused on a topic). Use for
    'what is this document about' or 'summarise section X' requests."""
    from app.agent.llm import chat_model  # local import avoids a cycle

    user_id, allowed = _scope(config)
    _check_doc(document_id, allowed)
    _progress("Reading the whole document")
    chunks = await retrieval.get_document_chunks(user_id, document_id)
    if not chunks:
        return "That document has no indexed text."
    name = chunks[0]["name"]
    # Map: summarise ~6k-token groups in parallel. Reduce: merge the partials.
    groups: list[list[dict]] = []
    cur: list[dict] = []
    size = 0
    for c in chunks:
        cur.append(c)
        size += len(c["content"]) // 4
        if size > 6000:
            groups.append(cur)
            cur, size = [], 0
    if cur:
        groups.append(cur)

    llm = chat_model()
    focus_note = f" Focus on: {focus}." if focus else ""

    async def summarise(group: list[dict]) -> str:
        text = "\n\n".join(f"(p. {c['page_start']}) {c['content']}" for c in group)
        msg = await llm.ainvoke(
            [
                (
                    "system",
                    "Summarise the text in 5-8 bullet points. Keep page numbers "
                    f"as (p. N) after each point.{focus_note}",
                ),
                ("human", text),
            ]
        )
        return str(msg.content)

    partials = await asyncio.gather(*(summarise(g) for g in groups[:12]))
    if len(partials) == 1:
        return f"[{name}] summary:\n{partials[0]}"
    merged = await llm.ainvoke(
        [
            (
                "system",
                "Merge these partial summaries into one summary of 6-10 bullet points, "
                f"keeping (p. N) page references.{focus_note}",
            ),
            ("human", "\n\n".join(partials)),
        ]
    )
    return f"[{name}] summary:\n{merged.content}"


@tool
async def compare_documents(
    document_a: str,
    document_b: str,
    aspect: str,
    *,
    config: RunnableConfig,
) -> str:
    """Fetch passages about `aspect` from two documents side by side, so you can
    compare them."""
    user_id, allowed = _scope(config)
    _check_doc(document_a, allowed)
    _check_doc(document_b, allowed)
    _progress(f"Comparing documents on “{aspect}”")
    a, b = await _with_timeout(
        asyncio.gather(
            retrieval.search(user_id, aspect, document_ids=[document_a], k=4),
            retrieval.search(user_id, aspect, document_ids=[document_b], k=4),
        )
    )
    return f"=== Document A ===\n{_format_hits(a)}\n\n=== Document B ===\n{_format_hits(b)}"


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Arithmetic only: numbers and + - * / // % **. No names, calls or attributes."""
    if len(expression) > 200:
        raise ValueError("expression too long")

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return ev(ast.parse(expression.replace(",", ""), mode="eval"))


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(1250 - 980) / 980 * 100'."""
    try:
        result = safe_eval(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"error: {exc}"
    return f"{expression} = {round(result, 6)}"


TOOLS = [search_documents, get_page, summarize_document, compare_documents, calculator]
