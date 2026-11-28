"""Build the eval set: questions with a known answer and a known source page.

For a sample of pages in each corpus PDF, an LLM writes one question that page
answers, the reference answer, and a verbatim supporting quote. Items whose
quote isn't actually on the page are dropped, so every gold label is checked
against the PDF text. About a quarter of items are marked `test` (held out:
don't tune chunking or prompts against them).

    uv run python -m evals.build_dataset --per-doc 12

Review evals/dataset.jsonl by hand before trusting the numbers: delete
ambiguous questions and ones answerable from many pages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.llm import chat_model
from app.ingest.pdf import parse_pdf
from scripts.ingest_local import doc_id_for

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
OUT = HERE / "dataset.jsonl"

KINDS = ["factual", "numeric", "exact_term", "definition", "procedure"]

SYSTEM = (
    "You write evaluation questions for a document search system. Given one page of a "
    "PDF, write ONE question that this page answers. Rules:\n"
    "- The question must be specific enough that it's answered by this page, not by "
    "general knowledge or by the document title. Name the concrete thing being asked "
    "about.\n"
    "- Don't mention 'this page', 'this section' or 'the document'. Ask it the way a user "
    "who has the document would.\n"
    "- quote: copy 1-2 sentences from the page VERBATIM that contain the answer.\n"
    "- answer: a short, complete reference answer.\n"
    "- If the page is a table of contents, index, references list, or has no "
    "substantive content, set usable=false."
)


class QA(BaseModel):
    usable: bool
    question: str = ""
    answer: str = ""
    quote: str = ""
    kind: Literal["factual", "numeric", "exact_term", "definition", "procedure"] = "factual"


class Item(BaseModel):
    id: str
    question: str
    answer: str
    quote: str
    kind: str
    document_id: str
    document_name: str
    page: int
    split: Literal["dev", "test"]
    tags: list[str] = Field(default_factory=list)


_NORM = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORM.sub(" ", text.lower()).strip()


def quote_on_page(quote: str, page_text: str) -> bool:
    q = _norm(quote)
    return len(q) >= 20 and q in _norm(page_text)


async def questions_for(path: Path, per_doc: int, rng: random.Random) -> list[Item]:
    parsed = parse_pdf(await asyncio.to_thread(path.read_bytes))
    pages = [p for p in range(1, parsed.page_count + 1) if len(parsed.page_text(p)) > 900]
    # Skip front matter: the first pages are covers, contents and forewords.
    pages = [p for p in pages if p > 3] or pages
    rng.shuffle(pages)
    llm = chat_model().with_structured_output(QA)
    items: list[Item] = []
    sem = asyncio.Semaphore(6)

    async def one(page: int, kind: str) -> Item | None:
        text = parsed.page_text(page)[:6000]
        async with sem:
            qa = await llm.ainvoke(
                [
                    ("system", SYSTEM),
                    ("human", f"Preferred question type: {kind}\n\nPage {page}:\n{text}"),
                ]
            )
        if not qa.usable or not qa.question or not quote_on_page(qa.quote, text):
            return None
        return Item(
            id="",
            question=qa.question.strip(),
            answer=qa.answer.strip(),
            quote=qa.quote.strip(),
            kind=qa.kind,
            document_id=doc_id_for(path),
            document_name=path.name,
            page=page,
            split="dev",
        )

    # Over-sample: some pages get rejected.
    candidates = pages[: per_doc * 2]
    results = await asyncio.gather(
        *(one(p, KINDS[i % len(KINDS)]) for i, p in enumerate(candidates))
    )
    for item in results:
        if item and len(items) < per_doc:
            items.append(item)
    return items


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-doc", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    rng = random.Random(args.seed)
    paths = sorted(CORPUS.glob("*.pdf"))
    if not paths:
        raise SystemExit(f"No PDFs in {CORPUS}. See evals/corpus/SOURCES.md.")
    all_items: list[Item] = []
    for path in paths:
        items = await questions_for(path, args.per_doc, rng)
        print(f"{path.name}: {len(items)} questions")
        all_items.extend(items)

    rng.shuffle(all_items)
    for i, item in enumerate(all_items):
        item.id = f"q{i:03d}"
        item.split = "test" if i % 4 == 3 else "dev"
    with args.out.open("w") as f:
        for item in sorted(all_items, key=lambda x: x.id):
            f.write(item.model_dump_json() + "\n")
    n_test = sum(i.split == "test" for i in all_items)
    print(f"wrote {len(all_items)} items ({n_test} test) to {args.out}")


def load(path: Path = OUT, split: str | None = None) -> list[Item]:
    items = [Item(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    return [i for i in items if split in (None, "all") or i.split == split]


if __name__ == "__main__":
    asyncio.run(main())
