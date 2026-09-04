"""Best-effort PostgreSQL connection-pool lifecycle for evaluation telemetry."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from psycopg import Connection
from psycopg_pool import ConnectionPool

from app import config
from app.logging_conf import get_logger

logger = get_logger("DATABASE")

_lock = threading.Lock()
_pool: ConnectionPool | None = None


class DatabaseUnavailable(RuntimeError):
    """Raised when evaluation storage has no usable PostgreSQL connection."""


def open_pool(*, wait: bool = True) -> bool:
    """Open the shared pool, logging and returning False when PostgreSQL is down."""
    global _pool
    if not config.DATABASE_URL:
        logger.warning("Evaluation database disabled | DATABASE_URL is not configured")
        return False
    with _lock:
        if _pool is None:
            _pool = ConnectionPool(
                conninfo=config.DATABASE_URL,
                min_size=0,
                max_size=5,
                timeout=3,
                open=False,
                kwargs={"connect_timeout": 3},
            )
        pool = _pool
    try:
        pool.open(wait=wait, timeout=3)
        return True
    except Exception as exc:
        logger.warning(
            "Evaluation database connection unavailable | error=%s: %s",
            type(exc).__name__,
            exc,
        )
        return False


def close_pool() -> None:
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.close()


@contextmanager
def connection() -> Iterator[Connection]:
    """Yield a pooled connection, retrying pool startup after an earlier outage."""
    if not open_pool(wait=False):
        raise DatabaseUnavailable("DATABASE_URL is not configured or PostgreSQL is unavailable")
    assert _pool is not None
    try:
        with _pool.connection(timeout=3) as conn:
            yield conn
    except Exception as exc:
        raise DatabaseUnavailable(str(exc)) from exc
