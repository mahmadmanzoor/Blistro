"""Ingestion worker: `python -m app.worker`. Long-running process (Railway/Fly),
separate from the API so uploads never block or get killed mid-request."""

from __future__ import annotations

import asyncio
import logging
import signal

from app import db
from app.config import get_settings
from app.ingest.pipeline import claim_job, run_job

log = logging.getLogger("worker")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    await db.open_pool(max_size=4)
    await db.migrate()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    log.info("worker started")
    try:
        while not stop.is_set():
            job = await claim_job()
            if job is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=get_settings().worker_poll_s)
                except TimeoutError:
                    pass
                continue
            await run_job(job)
    finally:
        await db.close_pool()
        log.info("worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
