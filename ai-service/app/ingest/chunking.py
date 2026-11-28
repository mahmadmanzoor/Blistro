"""Three chunking strategies behind one interface.

- fixed:     sliding window of N tokens with overlap, blind to structure.
- recursive: split by detected headings, then paragraphs, then sentences, then
             tokens, packing pieces up to N tokens. Keeps the section heading.
- semantic:  embed every sentence and cut where similarity between neighbours
             drops sharply (a high percentile of cosine distance), bounded by
             min/max size.

Every chunk keeps the page range it came from so answers can cite pages and
evals can check retrieval against a gold page."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import tiktoken

from app.config import ChunkStrategy
from app.embeddings import Embedder
from app.ingest.pdf import Block, ParsedPdf


@dataclass(frozen=True)
class Chunk:
    content: str
    page_start: int
    page_end: int
    token_count: int
    heading: str | None = None

    @property
    def embed_text(self) -> str:
        return f"{self.heading}\n{self.content}" if self.heading else self.content


@lru_cache
def _enc() -> tiktoken.Encoding:
    # cl100k_base is the tokenizer used by text-embedding-3-* models.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


_SENTENCE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    return [s.strip() for s in _SENTENCE.split(flat) if s.strip()]


# ---------------------------------------------------------------- fixed


def chunk_fixed(doc: ParsedPdf, size: int, overlap: int) -> list[Chunk]:
    enc = _enc()
    tokens: list[int] = []
    pages: list[int] = []
    for blk in doc.blocks:
        ids = enc.encode(blk.text + "\n", disallowed_special=())
        tokens.extend(ids)
        pages.extend([blk.page] * len(ids))
    if not tokens:
        return []
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        page_slice = pages[start : start + size]
        text = enc.decode(window).strip()
        if text:
            chunks.append(Chunk(text, min(page_slice), max(page_slice), len(window)))
        if start + size >= len(tokens):
            break
    return chunks


# ---------------------------------------------------------------- recursive


@dataclass
class _Piece:
    text: str
    page_start: int
    page_end: int
    tokens: int


def _split_piece(piece: _Piece, size: int) -> list[_Piece]:
    """Split one oversized piece by sentences, then by tokens."""
    if piece.tokens <= size:
        return [piece]
    out: list[_Piece] = []
    for sent in split_sentences(piece.text) or [piece.text]:
        n = count_tokens(sent)
        if n <= size:
            out.append(_Piece(sent, piece.page_start, piece.page_end, n))
            continue
        ids = _enc().encode(sent, disallowed_special=())
        for i in range(0, len(ids), size):
            part = _enc().decode(ids[i : i + size])
            out.append(_Piece(part, piece.page_start, piece.page_end, len(ids[i : i + size])))
    return out


def _pack(pieces: list[_Piece], size: int, heading: str | None) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[_Piece] = []
    cur_tokens = 0
    budget = size - (count_tokens(heading) + 1 if heading else 0)
    for p in pieces:
        if cur and cur_tokens + p.tokens > budget:
            chunks.append(_merge(cur, heading))
            cur, cur_tokens = [], 0
        cur.append(p)
        cur_tokens += p.tokens
    if cur:
        chunks.append(_merge(cur, heading))
    return chunks


def _merge(pieces: list[_Piece], heading: str | None) -> Chunk:
    text = "\n".join(p.text for p in pieces)
    return Chunk(
        content=text,
        page_start=min(p.page_start for p in pieces),
        page_end=max(p.page_end for p in pieces),
        token_count=sum(p.tokens for p in pieces),
        heading=heading,
    )


def chunk_recursive(doc: ParsedPdf, size: int, min_section_tokens: int = 40) -> list[Chunk]:
    # 1. sections: a heading plus the blocks under it
    sections: list[tuple[str | None, list[Block]]] = []
    heading: str | None = None
    body: list[Block] = []
    for blk in doc.blocks:
        if blk.is_heading:
            if body or heading:
                sections.append((heading, body))
            heading, body = blk.text, []
        else:
            body.append(blk)
    if body or heading:
        sections.append((heading, body))

    # 2. fold tiny sections (e.g. a heading straight after a heading) forward
    merged: list[tuple[str | None, list[Block]]] = []
    carry_heading: str | None = None
    carry_blocks: list[Block] = []
    for h, blocks in sections:
        h_full = " / ".join(x for x in (carry_heading, h) if x) or None
        all_blocks = carry_blocks + blocks
        if sum(count_tokens(b.text) for b in all_blocks) < min_section_tokens:
            carry_heading, carry_blocks = h_full, all_blocks
            continue
        merged.append((h_full, all_blocks))
        carry_heading, carry_blocks = None, []
    if carry_blocks or carry_heading:
        if merged and carry_blocks:
            h, blocks = merged[-1]
            merged[-1] = (h, blocks + carry_blocks)
        elif carry_blocks:
            merged.append((carry_heading, carry_blocks))

    # 3. paragraphs -> sentences -> tokens, packed up to `size`
    chunks: list[Chunk] = []
    for h, blocks in merged:
        pieces: list[_Piece] = []
        for b in blocks:
            p = _Piece(b.text, b.page, b.page, count_tokens(b.text))
            pieces.extend(_split_piece(p, size))
        chunks.extend(_pack(pieces, size, h))
    return chunks


# ---------------------------------------------------------------- semantic


async def chunk_semantic(
    doc: ParsedPdf,
    embedder: Embedder,
    max_tokens: int,
    min_tokens: int = 64,
    percentile: float = 90.0,
) -> list[Chunk]:
    sents: list[tuple[str, int]] = []
    for blk in doc.blocks:
        for s in split_sentences(blk.text):
            sents.append((s, blk.page))
    if not sents:
        return []
    vectors = np.array(await embedder.embed([s for s, _ in sents]), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1) if len(sents) > 1 else np.array([])
    threshold = float(np.percentile(distances, percentile)) if distances.size else 1.0

    chunks: list[Chunk] = []
    cur: list[tuple[str, int, int]] = []  # text, page, tokens
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            chunks.append(
                Chunk(
                    content=" ".join(t for t, _, _ in cur),
                    page_start=min(p for _, p, _ in cur),
                    page_end=max(p for _, p, _ in cur),
                    token_count=cur_tokens,
                )
            )
        cur, cur_tokens = [], 0

    for i, (sent, page) in enumerate(sents):
        n = count_tokens(sent)
        if cur and cur_tokens + n > max_tokens:
            flush()
        cur.append((sent, page, n))
        cur_tokens += n
        is_break = i < len(distances) and distances[i] >= threshold
        if is_break and cur_tokens >= min_tokens:
            flush()
    flush()
    return chunks


# ---------------------------------------------------------------- entry point


async def chunk_document(
    doc: ParsedPdf,
    strategy: ChunkStrategy,
    embedder: Embedder,
    size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(doc, size, overlap)
    if strategy == "recursive":
        return chunk_recursive(doc, size)
    if strategy == "semantic":
        return await chunk_semantic(doc, embedder, max_tokens=size)
    raise ValueError(f"unknown strategy {strategy!r}")
