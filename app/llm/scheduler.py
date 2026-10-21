from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from app import config
from app.logging_conf import current_job_id, get_logger

logger = get_logger("LLM_SCHEDULER")


@dataclass
class _JobGate:
    semaphore: threading.BoundedSemaphore
    lock: threading.Lock
    active: int = 0
    peak: int = 0
    waiting: int = 0
    completed: int = 0


_gates_lock = threading.Lock()
_gates: dict[tuple[str, str], _JobGate] = {}
_last_request_metrics: ContextVar[dict[str, float] | None] = ContextVar(
    "llm_last_request_metrics", default=None
)


def _job_key() -> str:
    # Tests and synchronous utility calls without a job context share the
    # process-local gate. Real exam jobs receive an isolated ceiling.
    return current_job_id() or "__unscoped__"


def _gate(key: str, *, agent: str = "") -> _JobGate:
    scope = "assessment_catalog" if agent == "assessment_compiler" else "default"
    gate_key = (key, scope)
    limit = (
        config.MAX_PARALLEL_CATALOG_CALLS
        if scope == "assessment_catalog"
        else config.LLM_JOB_CONCURRENCY_LIMIT
    )
    with _gates_lock:
        gate = _gates.get(gate_key)
        if gate is None:
            gate = _JobGate(
                semaphore=threading.BoundedSemaphore(limit),
                lock=threading.Lock(),
            )
            _gates[gate_key] = gate
        return gate


@contextmanager
def llm_request_slot(*, agent: str, operation: str) -> Iterator[None]:
    """Enforce one shared LLM request ceiling for the current exam job."""
    key = _job_key()
    gate = _gate(key, agent=agent)
    limit = (
        config.MAX_PARALLEL_CATALOG_CALLS
        if agent == "assessment_compiler"
        else config.LLM_JOB_CONCURRENCY_LIMIT
    )
    queued_at = time.perf_counter()
    with gate.lock:
        gate.waiting += 1
    gate.semaphore.acquire()
    started_at = time.perf_counter()
    with gate.lock:
        gate.waiting -= 1
        gate.active += 1
        gate.peak = max(gate.peak, gate.active)
        active = gate.active
        waiting = gate.waiting
    logger.info(
        "LLM scheduler acquired | job=%s | agent=%s | operation=%s | "
        "queue_ms=%.1f | active=%d | waiting=%d | limit=%d",
        key, agent, operation, (started_at - queued_at) * 1000,
        active, waiting, limit,
    )
    try:
        yield
    finally:
        finished_at = time.perf_counter()
        _last_request_metrics.set({
            "queue_ms": round((started_at - queued_at) * 1000, 3),
            "run_ms": round((finished_at - started_at) * 1000, 3),
        })
        with gate.lock:
            gate.active -= 1
            gate.completed += 1
            active = gate.active
        gate.semaphore.release()
        logger.info(
            "LLM scheduler released | job=%s | agent=%s | operation=%s | "
            "run_ms=%.1f | active=%d",
            key, agent, operation, (finished_at - started_at) * 1000, active,
        )


def scheduler_snapshot(
    job_id: str | None = None, *, agent: str = "",
) -> dict[str, int]:
    key = job_id or _job_key()
    gate = _gate(key, agent=agent)
    limit = (
        config.MAX_PARALLEL_CATALOG_CALLS
        if agent == "assessment_compiler"
        else config.LLM_JOB_CONCURRENCY_LIMIT
    )
    with gate.lock:
        return {
            "active": gate.active,
            "peak": gate.peak,
            "waiting": gate.waiting,
            "completed": gate.completed,
            "limit": limit,
        }


def last_request_metrics() -> dict[str, float]:
    """Return scheduler timings for the latest request in this context."""
    return dict(_last_request_metrics.get() or {})


def reset_scheduler_for_tests() -> None:
    with _gates_lock:
        _gates.clear()
    _last_request_metrics.set(None)
