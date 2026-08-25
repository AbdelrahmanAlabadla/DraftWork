"""In-memory store for final validated exams.

Exams are saved after successful generation and addressed by ``exam_id`` for
all downstream exports (PDF / DOCX / Google Forms).

NOTE: this store is intentionally in-memory. Restarting the server clears
every stored exam. The public interface (``save_exam`` / ``get_exam``) is the
only contract exporters rely on, so it can be replaced by a persistent store
(PostgreSQL, Redis, ...) later without touching any exporter.

Tuning is explicit via the two constants below.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

# How long a stored exam stays retrievable (seconds).
EXAM_TTL_SECONDS = 3600

# Maximum number of exams kept at once; oldest are evicted first.
MAX_STORED_EXAMS = 20


class ExamNotFound(KeyError):
    """Raised when an exam_id does not exist (or has expired)."""


_lock = threading.Lock()
_exams: dict[str, dict[str, Any]] = {}


def _evict_expired(now: float) -> None:
    expired = [
        eid for eid, entry in _exams.items() if now - entry["created_at"] > EXAM_TTL_SECONDS
    ]
    for eid in expired:
        del _exams[eid]


def save_exam(exams: list[dict[str, Any]], warnings: list[str] | None = None,
              document_id: str | None = None,
              metadata: dict[str, Any] | None = None) -> str:
    """Store a final validated exam result and return its new exam_id."""
    exam_id = f"exam_{uuid.uuid4().hex[:12]}"
    with _lock:
        now = time.time()
        _evict_expired(now)
        while len(_exams) >= MAX_STORED_EXAMS:
            oldest = min(_exams, key=lambda k: _exams[k]["created_at"])
            del _exams[oldest]
        _exams[exam_id] = {
            "exam_id": exam_id,
            "document_id": document_id,
            "created_at": now,
            "exams": exams,
            "warnings": list(warnings or []),
            "metadata": dict(metadata or {}),
        }
    return exam_id


def get_exam(exam_id: str) -> dict[str, Any]:
    """Return the stored exam record; raise ExamNotFound for unknown/expired ids."""
    with _lock:
        _evict_expired(time.time())
        entry = _exams.get(exam_id)
    if entry is None:
        raise ExamNotFound(f"Unknown or expired exam_id: {exam_id}")
    return entry


def clear() -> None:
    """Test helper: drop every stored exam."""
    with _lock:
        _exams.clear()
