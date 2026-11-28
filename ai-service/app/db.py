"""Database access: one psycopg 3 async pool shared by app queries and the
LangGraph checkpointer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: AsyncConnectionPool | None = None


def _conn_kwargs() -> dict:
    # autocommit + dict_row are required by AsyncPostgresSaver. Behind a
    # transaction pooler, prepare_threshold=None turns off prepared statements.
    kwargs: dict = {"autocommit": True, "row_factory": dict_row}
    if get_settings().db_pooled:
        kwargs["prepare_threshold"] = None
    return kwargs


async def open_pool(min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=get_settings().database_url,
            min_size=min_size,
            max_size=max_size,
            kwargs=_conn_kwargs(),
            open=False,
        )
        await _pool.open(wait=True, timeout=30)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("DB pool not opened; call open_pool() at startup")
    return _pool


@asynccontextmanager
async def connection() -> AsyncIterator[AsyncConnection]:
    async with pool().connection() as conn:
        yield conn


@asynccontextmanager
async def transaction() -> AsyncIterator[AsyncConnection]:
    async with pool().connection() as conn, conn.transaction():
        yield conn


async def migrate() -> None:
    sql = SCHEMA_PATH.read_text()
    async with connection() as conn:
        await conn.execute(sql)


def to_vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
