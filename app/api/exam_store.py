"""Compatibility store for the unversioned synchronous API.

The production ``/api/v1`` workflow stores exams in PostgreSQL. This small
file-backed adapter keeps existing local scripts and regression tests working
without process-global exam state while those callers migrate to queued jobs.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import LEGACY_EXAM_DIR


EXAM_TTL_SECONDS = 3600
MAX_STORED_EXAMS = 20


class ExamNotFound(KeyError):
    """Raised when an exam_id does not exist or has expired."""


_lock = threading.Lock()


def _directory() -> Path:
    path = Path(LEGACY_EXAM_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path_for(exam_id: str) -> Path:
    safe_id = str(exam_id)
    if not safe_id.startswith("exam_") or not safe_id.replace("_", "").isalnum():
        raise ExamNotFound(f"Unknown or expired exam_id: {exam_id}")
    return _directory() / f"{safe_id}.json"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _evict(now: float) -> None:
    records: list[tuple[Path, float]] = []
    for path in _directory().glob("exam_*.json"):
        entry = _read(path)
        if entry is None or now - float(entry.get("created_at") or 0) > EXAM_TTL_SECONDS:
            path.unlink(missing_ok=True)
            continue
        order = float(entry.get("created_at_ns") or float(entry["created_at"]) * 1_000_000_000)
        records.append((path, order))
    records.sort(key=lambda item: item[1])
    while len(records) >= MAX_STORED_EXAMS:
        path, _ = records.pop(0)
        path.unlink(missing_ok=True)


def save_exam(
    exams: list[dict[str, Any]],
    warnings: list[str] | None = None,
    document_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    exam_id = f"exam_{uuid.uuid4().hex[:12]}"
    with _lock:
        now = time.time()
        _evict(now)
        entry = {
            "exam_id": exam_id,
            "document_id": document_id,
            "created_at": now,
            "created_at_ns": time.time_ns(),
            "exams": exams,
            "warnings": list(warnings or []),
            "metadata": dict(metadata or {}),
        }
        path = _path_for(exam_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    return exam_id


def get_exam(exam_id: str) -> dict[str, Any]:
    with _lock:
        path = _path_for(exam_id)
        entry = _read(path)
        if entry is None or time.time() - float(entry.get("created_at") or 0) > EXAM_TTL_SECONDS:
            path.unlink(missing_ok=True)
            raise ExamNotFound(f"Unknown or expired exam_id: {exam_id}")
        return entry


def clear() -> None:
    """Test/local helper: remove compatibility exam records."""
    with _lock:
        for path in _directory().glob("exam_*.json"):
            path.unlink(missing_ok=True)
