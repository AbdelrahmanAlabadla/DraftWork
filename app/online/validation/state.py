"""Shared validation state helpers and cancellation checks."""
from __future__ import annotations

from typing import Any

from app.offline.pipeline import PipelineCancelled


def _ensure_generation_active(state: dict[str, Any]) -> None:
    """Stop graph scheduling once the owning job reports cancellation."""
    check = state.get("is_cancelled")
    if callable(check) and check():
        raise PipelineCancelled("Exam generation was cancelled before scheduling more work")
