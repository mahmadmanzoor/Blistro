"""Ingest local PDFs straight into Postgres (no queue). Used for the eval corpus.

    uv run python -m scripts.ingest_local evals/corpus/*.pdf --user eval \
        --strategies fixed,recursive,semantic
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from app import db
from app.ingest.pipeline import ingest_bytes, upsert_document


def doc_id_for(path: Path) -> str:
    return f"eval-{path.stem}"


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--user", default="eval")
    p.add_argument("--strategies", default="fixed,recursive,semantic")
    args = p.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    await db.open_pool()
    await db.migrate()
    try:
        for path in args.paths:
            started = time.perf_counter()
            doc_id = doc_id_for(path)
            await upsert_document(doc_id, args.user, path.name, None)
            counts = await ingest_bytes(doc_id, args.user, path.read_bytes(), strategies)
            print(f"{path.name}: {counts} in {time.perf_counter() - started:.1f}s")
    finally:
        await db.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
