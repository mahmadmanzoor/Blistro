"""The Blistro agent as a LangGraph StateGraph.

    START -> route --direct--> direct ------------------------------> finalize -> END
                   --retrieve-> retrieve -> grade --relevant--> generate --> check_grounding
                                  ^           |--none, 1st time--> rewrite --^      |
                                  |           '--none again--> not_found -> finalize
                   --tools----> tool_agent <-> tools (loop, step-limited) -> check_grounding

    check_grounding --grounded--> finalize
                    --not grounded, 1st time--> generate / tool_agent (strict mode)
                    --not grounded again------> finalize with a safe fallback

Memory: the conversation (`messages`) is checkpointed in Postgres per thread.
Per-turn working state (hits, tool scratchpad) is reset at the start of each turn.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app import retrieval
from app.agent import prompts
from app.agent.llm import chat_model
from app.agent.tools import TOOLS
from app.observability import get_chat_prompt

log = logging.getLogger(__name__)

MAX_TOOL_STEPS = 6
ANSWER_TAG = "answer"  # the API streams tokens only from LLM calls with this tag


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # persisted conversation
    scratch: Annotated[list[AnyMessage], add_messages]  # this turn's tool loop
    question: str
    route: Literal["direct", "retrieve", "tools"]
    query: str
    hits: list[dict]
    rewrites: int
    tool_steps: int
    answer: str
    context: str
    grounding_attempts: int
    strict: bool
    sources: list[dict]
    prompt_meta: dict
    model_failed: bool


# ---------------------------------------------------------------- structured outputs


class Route(BaseModel):
    route: Literal["direct", "retrieve", "tools"]
    standalone_question: str = Field(description="The latest message as a standalone question")


class Grade(BaseModel):
    relevant_ids: list[str] = Field(description="Labels (e.g. S1) of passages that help answer")


class Grounding(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- helpers


def _emit(event: dict) -> None:
    """Send a custom stream event; a no-op when the graph isn't being streamed."""
    try:
        get_stream_writer()(event)
    except Exception:  # noqa: BLE001
        pass


def _progress(text: str) -> None:
    _emit({"type": "progress", "text": text})


def _history(messages: list[AnyMessage], turns: int = 6) -> str:
    recent = [m for m in messages if isinstance(m, HumanMessage | AIMessage)][-turns - 1 : -1]
    if not recent:
        return "(none)"
    return "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {str(m.content)[:800]}"
        for m in recent
    )


def _label_passages(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[S{i}] ({h['document_name']}, p. {h['page_start']}"
        f"{'-' + str(h['page_end']) if h['page_end'] != h['page_start'] else ''})"
        f"{' — ' + h['heading'] if h.get('heading') else ''}\n{h['content']}"
        for i, h in enumerate(hits, start=1)
    )


def _prompt(name: str, **values: str) -> tuple[list[tuple[str, str]], dict]:
    messages, meta = get_chat_prompt(name, prompts.ALL[name])
    return prompts.render(messages, **values), {name: meta}


def _scope(config: RunnableConfig) -> tuple[str, list[str]]:
    conf = config.get("configurable", {})
    return conf["user_id"], list(conf.get("document_ids") or [])


# ---------------------------------------------------------------- nodes


async def route(state: AgentState, config: RunnableConfig) -> dict:
    question = str(state["messages"][-1].content)
    reset = {
        "scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        "hits": [],
        "rewrites": 0,
        "tool_steps": 0,
        "grounding_attempts": 0,
        "strict": False,
        "answer": "",
        "context": "",
        "model_failed": False,
        "sources": [],
        "prompt_meta": {},
    }
    msgs, meta = _prompt("blistro-router", history=_history(state["messages"]), question=question)
    try:
        decision = await chat_model().with_structured_output(Route).ainvoke(msgs, config)
        return {
            **reset,
            "route": decision.route,
            "question": decision.standalone_question,
            "query": decision.standalone_question,
            "prompt_meta": meta,
        }
    except Exception:  # noqa: BLE001 - router failure: retrieval is the safe default
        log.warning("router failed; defaulting to retrieve", exc_info=True)
        return {**reset, "route": "retrieve", "question": question, "query": question}


async def retrieve(state: AgentState, config: RunnableConfig) -> dict:
    user_id, doc_ids = _scope(config)
    _progress("Searching your documents")
    hits = await retrieval.search(user_id, state["query"], document_ids=doc_ids or None)
    return {"hits": [h.__dict__ | {"sources": list(h.sources)} for h in hits]}


async def grade(state: AgentState, config: RunnableConfig) -> dict:
    hits = state.get("hits") or []
    if not hits:
        return {"hits": []}
    _progress("Checking what's relevant")
    msgs, meta = _prompt(
        "blistro-grader", question=state["question"], passages=_label_passages(hits)
    )
    try:
        result = await chat_model().with_structured_output(Grade).ainvoke(msgs, config)
        keep = {int(x.lstrip("Ss")) for x in result.relevant_ids if x.lstrip("Ss").isdigit()}
        kept = [h for i, h in enumerate(hits, start=1) if i in keep]
    except Exception:  # noqa: BLE001 - grader failure: keep everything rather than drop answers
        log.warning("grader failed; keeping all hits", exc_info=True)
        kept = hits
    return {"hits": kept, "prompt_meta": state.get("prompt_meta", {}) | meta}


async def rewrite(state: AgentState, config: RunnableConfig) -> dict:
    _progress("Rephrasing the search")
    msgs, _ = _prompt("blistro-rewrite", question=state["question"])
    try:
        msg = await chat_model().ainvoke(msgs, config)
        query = str(msg.content).strip().strip('"') or state["question"]
    except Exception:  # noqa: BLE001 - search again with the original wording
        log.warning("rewrite failed; reusing the question", exc_info=True)
        query = state["question"]
    return {"query": query, "rewrites": state.get("rewrites", 0) + 1}


def _model_failed(exc: Exception) -> dict:
    """The model failed twice (one retry inside the client). Give a clear answer
    instead of an error, and leave no context so the grounding check skips it."""
    log.warning("model call failed after retry: %s", exc, exc_info=True)
    _emit({"type": "token", "text": prompts.MODEL_FAILED})
    return {"answer": prompts.MODEL_FAILED, "context": "", "model_failed": True}


async def generate(state: AgentState, config: RunnableConfig) -> dict:
    hits = state["hits"]
    strict_note = (
        "IMPORTANT: a previous draft made claims not in the passages. Only state what the "
        "passages say, word for word if needed."
        if state.get("strict")
        else ""
    )
    if state.get("grounding_attempts"):
        _emit({"type": "reset"})  # tell the client to discard the streamed draft
    msgs, meta = _prompt(
        "blistro-generate",
        passages=_label_passages(hits),
        history=_history(state["messages"]),
        question=state["question"],
        strict_note=strict_note,
    )
    llm = chat_model(streaming=True).with_config(tags=[ANSWER_TAG])
    try:
        msg = await llm.ainvoke(msgs, config)
    except Exception as exc:  # noqa: BLE001
        return _model_failed(exc)
    return {
        "answer": str(msg.content),
        "context": _label_passages(hits),
        "prompt_meta": state.get("prompt_meta", {}) | meta,
    }


async def tool_agent(state: AgentState, config: RunnableConfig) -> dict:
    user_id, doc_ids = _scope(config)
    docs = [d for d in await retrieval.list_documents(user_id) if not doc_ids or d["id"] in doc_ids]
    doc_list = (
        "\n".join(f"- {d['name']} (document_id: {d['id']}, {d['page_count']} pages)" for d in docs)
        or "(none)"
    )
    sys_msgs, meta = _prompt("blistro-tool-agent", documents=doc_list)
    system = [SystemMessage(content=text) for _, text in sys_msgs]
    if state.get("strict"):
        _emit({"type": "reset"})
    scratch = state.get("scratch") or []
    seed = [] if scratch else [HumanMessage(content=state["question"])]
    if state.get("strict") and scratch:
        seed = [
            HumanMessage(
                content="Your previous answer had claims the tool results don't "
                "support. Revise it using only the tool results above."
            )
        ]
    llm = chat_model(streaming=True).bind_tools(TOOLS).with_config(tags=[ANSWER_TAG])
    try:
        msg = await llm.ainvoke([*system, *_history_messages(state), *scratch, *seed], config)
    except Exception as exc:  # noqa: BLE001
        return _model_failed(exc) | {"tool_steps": state.get("tool_steps", 0) + 1}
    update: dict = {
        "scratch": [*seed, msg],
        "tool_steps": state.get("tool_steps", 0) + 1,
        "prompt_meta": state.get("prompt_meta", {}) | meta,
    }
    if not getattr(msg, "tool_calls", None):
        tool_text = "\n\n".join(str(m.content) for m in scratch if isinstance(m, ToolMessage))
        update |= {"answer": str(msg.content), "context": tool_text}
    return update


def _history_messages(state: AgentState) -> list[AnyMessage]:
    convo = [m for m in state["messages"] if isinstance(m, HumanMessage | AIMessage)]
    return convo[-7:-1]


tool_node = ToolNode(TOOLS, messages_key="scratch", handle_tool_errors=True)


async def check_grounding(state: AgentState, config: RunnableConfig) -> dict:
    answer = state.get("answer", "")
    context = state.get("context", "")
    if not context.strip():
        # Tool path with no lookups (e.g. pure arithmetic) has nothing to check against.
        return {"grounding_attempts": state.get("grounding_attempts", 0) + 1, "strict": False}
    _progress("Double-checking the answer against the sources")
    msgs, meta = _prompt("blistro-grounding", context=context[:24000], answer=answer)
    try:
        result = await chat_model().with_structured_output(Grounding).ainvoke(msgs, config)
        grounded = result.grounded
    except Exception:  # noqa: BLE001
        log.warning("grounding check failed; accepting answer", exc_info=True)
        grounded = True
    attempts = state.get("grounding_attempts", 0) + 1
    return {
        "grounding_attempts": attempts,
        "strict": not grounded,
        "prompt_meta": state.get("prompt_meta", {}) | meta,
    }


async def direct(state: AgentState, config: RunnableConfig) -> dict:
    msgs, _ = _prompt("blistro-direct")
    llm = chat_model(streaming=True).with_config(tags=[ANSWER_TAG])
    try:
        msg = await llm.ainvoke([*msgs, *_history_messages(state), state["messages"][-1]], config)
    except Exception as exc:  # noqa: BLE001
        return _model_failed(exc)
    return {"answer": str(msg.content)}


async def not_found(state: AgentState) -> dict:
    _emit({"type": "token", "text": prompts.NOT_FOUND})
    return {"answer": prompts.NOT_FOUND, "hits": []}


_LABEL = re.compile(r"\[S(\d+)\]")


async def finalize(state: AgentState) -> dict:
    answer = state.get("answer") or prompts.NOT_FOUND
    hits = state.get("hits") or []
    if state.get("strict") and state.get("grounding_attempts", 0) >= 2:
        answer = (
            "I found related passages but couldn't verify a precise answer from them. "
            "Try asking about a specific section or page."
        )
    used: dict[int, dict] = {}
    multi_doc = len({h["document_id"] for h in hits}) > 1

    def cite(m: re.Match) -> str:
        i = int(m.group(1))
        if not 1 <= i <= len(hits):
            return ""
        h = hits[i - 1]
        used[i] = h
        page = (
            f"p. {h['page_start']}"
            if h["page_start"] == h["page_end"]
            else f"pp. {h['page_start']}-{h['page_end']}"
        )
        return f"[{h['document_name']}, {page}]" if multi_doc else f"[{page}]"

    answer = _LABEL.sub(cite, answer)
    sources = [
        {
            "label": f"S{i}",
            "document_id": h["document_id"],
            "document_name": h["document_name"],
            "page_start": h["page_start"],
            "page_end": h["page_end"],
        }
        for i, h in sorted(used.items())
    ]
    return {"messages": [AIMessage(content=answer)], "answer": answer, "sources": sources}


# ---------------------------------------------------------------- edges


def after_route(state: AgentState) -> str:
    return {"direct": "direct", "tools": "tool_agent"}.get(state.get("route", ""), "retrieve")


def after_grade(state: AgentState) -> str:
    if state.get("hits"):
        return "generate"
    return "rewrite" if state.get("rewrites", 0) < 1 else "not_found"


def after_tool_agent(state: AgentState) -> str:
    if state.get("model_failed"):
        return "finalize"
    scratch = state.get("scratch") or []
    last = scratch[-1] if scratch else None
    if getattr(last, "tool_calls", None) and state.get("tool_steps", 0) < MAX_TOOL_STEPS:
        return "tools"
    if getattr(last, "tool_calls", None):  # step limit hit mid-loop
        return "finalize"
    return "check_grounding"


def after_grounding(state: AgentState) -> str:
    if not state.get("strict") or state.get("grounding_attempts", 0) >= 2:
        return "finalize"
    return "tool_agent" if state.get("route") == "tools" else "generate"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("tool_agent", tool_agent)
    g.add_node("tools", tool_node)
    g.add_node("check_grounding", check_grounding)
    g.add_node("direct", direct)
    g.add_node("not_found", not_found)
    g.add_node("finalize", finalize)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ["direct", "retrieve", "tool_agent"])
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "not_found"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check_grounding")
    g.add_conditional_edges(
        "tool_agent", after_tool_agent, ["tools", "check_grounding", "finalize"]
    )
    g.add_edge("tools", "tool_agent")
    g.add_conditional_edges(
        "check_grounding", after_grounding, ["finalize", "generate", "tool_agent"]
    )
    g.add_edge("direct", "finalize")
    g.add_edge("not_found", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
