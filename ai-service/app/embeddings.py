"""Embedding providers. OpenAI in real use; a deterministic hash embedding for
tests and offline runs (no network, stable vectors, shared words score higher)."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Protocol

from openai import AsyncOpenAI

from app.config import get_settings


class Embedder(Protocol):
    dims: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str, dims: int, api_key: str | None, batch_size: int = 128):
        self.model = model
        self.dims = dims
        self.batch_size = batch_size
        self._client = AsyncOpenAI(api_key=api_key, max_retries=4, timeout=60)
        self._sem = asyncio.Semaphore(4)

    async def _batch(self, batch: list[str]) -> list[list[float]]:
        async with self._sem:
            resp = await self._client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dims
            )
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clean = [t.replace("\n", " ")[:24000] or " " for t in texts]
        batches = [clean[i : i + self.batch_size] for i in range(0, len(clean), self.batch_size)]
        results = await asyncio.gather(*(self._batch(b) for b in batches))
        return [vec for batch in results for vec in batch]


_TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-words hashed into a fixed vector, L2-normalised."""

    def __init__(self, dims: int):
        self.dims = dims

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            vec[h % self.dims] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        s = get_settings()
        if s.embeddings_provider == "fake":
            _embedder = FakeEmbedder(s.embedding_dims)
        else:
            _embedder = OpenAIEmbedder(s.embedding_model, s.embedding_dims, s.openai_api_key)
    return _embedder


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
