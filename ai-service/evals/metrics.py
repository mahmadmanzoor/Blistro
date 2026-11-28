"""Answer and context metrics, following the RAGAS definitions, computed with an
LLM judge (and embeddings for response relevancy). Implemented here rather than
through the ragas package so every prompt and formula is visible and testable.

- faithfulness:       supported answer claims / all answer claims (vs. the context
                      the agent actually used)
- response relevancy: mean cosine similarity between the question and questions an
                      LLM reverse-engineers from the answer
- context precision:  average precision of the top-k retrieved chunks, where a chunk
                      counts if it helps produce the reference answer
- context recall:     reference-answer statements attributable to the retrieved
                      chunks / all reference statements
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.embeddings import Embedder, cosine


class Faithfulness(BaseModel):
    total_claims: int = Field(description="factual claims in the answer")
    supported_claims: int = Field(description="claims fully supported by the context")


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(description="questions this answer would answer")
    noncommittal: bool = Field(description="the answer is evasive or says it doesn't know")


class ChunkVerdicts(BaseModel):
    useful: list[bool] = Field(description="one verdict per chunk, in order")


class Attribution(BaseModel):
    total_statements: int
    attributed_statements: int = Field(description="statements supported by the chunks")


FAITH_PROMPT = (
    "Count the factual claims in the ANSWER, then count how many are fully supported by "
    "the CONTEXT. Ignore citations, greetings and statements that the answer wasn't found."
)
RELEVANCY_PROMPT = (
    "Write 3 different questions that the ANSWER answers. Also say whether the answer is "
    "noncommittal (evasive, 'I don't know', 'couldn't find it')."
)
PRECISION_PROMPT = (
    "For each numbered CHUNK, say whether it contains information that helps produce the "
    "REFERENCE answer to the QUESTION. Return one boolean per chunk, in order."
)
RECALL_PROMPT = (
    "Split the REFERENCE answer into its factual statements. Count how many can be "
    "attributed to the CHUNKS (supported by their text)."
)


def average_precision(relevant: list[bool]) -> float:
    """RAGAS context precision@k: mean of precision@i over the ranks i that are relevant."""
    hits, total = 0, 0.0
    for i, rel in enumerate(relevant, start=1):
        if rel:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


async def faithfulness(llm, answer: str, context: str) -> float | None:
    if not context.strip():
        return None
    f = await llm.with_structured_output(Faithfulness).ainvoke(
        [("system", FAITH_PROMPT), ("human", f"CONTEXT:\n{context[:20000]}\n\nANSWER:\n{answer}")]
    )
    return f.supported_claims / f.total_claims if f.total_claims else 1.0


async def response_relevancy(llm, embedder: Embedder, question: str, answer: str) -> float:
    g = await llm.with_structured_output(GeneratedQuestions).ainvoke(
        [("system", RELEVANCY_PROMPT), ("human", f"ANSWER:\n{answer}")]
    )
    if g.noncommittal or not g.questions:
        return 0.0
    vectors = await embedder.embed([question, *g.questions])
    sims = [cosine(vectors[0], v) for v in vectors[1:]]
    return sum(sims) / len(sims)


def _numbered(chunks: list[str]) -> str:
    return "\n\n".join(f"CHUNK {i}:\n{c[:3000]}" for i, c in enumerate(chunks, start=1))


async def context_precision(llm, question: str, reference: str, chunks: list[str]) -> float:
    if not chunks:
        return 0.0
    v = await llm.with_structured_output(ChunkVerdicts).ainvoke(
        [
            ("system", PRECISION_PROMPT),
            (
                "human",
                f"QUESTION: {question}\nREFERENCE: {reference}\n\n{_numbered(chunks)}",
            ),
        ]
    )
    verdicts = (list(v.useful) + [False] * len(chunks))[: len(chunks)]
    return average_precision(verdicts)


async def context_recall(llm, reference: str, chunks: list[str]) -> float:
    if not chunks:
        return 0.0
    a = await llm.with_structured_output(Attribution).ainvoke(
        [("system", RECALL_PROMPT), ("human", f"REFERENCE: {reference}\n\n{_numbered(chunks)}")]
    )
    if not a.total_statements:
        return 0.0
    return min(1.0, a.attributed_statements / a.total_statements)
