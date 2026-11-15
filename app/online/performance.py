"""Small JSON-safe helpers for pipeline scheduling telemetry."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any


def start_timing(*, stage: str, model_number: int | None = None,
                 slot_id: str | None = None, barrier: str | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "model_number": model_number,
        "slot_id": slot_id,
        "barrier": barrier,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "_started_monotonic": time.perf_counter(),
    }


def finish_timing(value: dict[str, Any], **metadata: Any) -> dict[str, Any]:
    started = float(value.pop("_started_monotonic", time.perf_counter()))
    value.update({
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        **{key: item for key, item in metadata.items() if item is not None},
    })
    return value


def merge_telemetry(state: dict[str, Any], *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(state.get("performance_telemetry") or [])
    for group in groups:
        values.extend(group)
    return values[-5000:]
