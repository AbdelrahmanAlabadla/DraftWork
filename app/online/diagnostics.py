"""Best-effort persistence for nonfatal internal pipeline failures."""
from __future__ import annotations

import traceback
from typing import Any

from app.api import repositories
from app.logging_conf import current_job_id, get_logger

logger = get_logger("DIAGNOSTICS")


def record_optional_failure(
    exc: BaseException,
    *,
    stage: str,
    component: str,
    job_id: str | None = None,
    exam_id: str | None = None,
    continued: bool = True,
    payload: dict[str, Any] | None = None,
) -> None:
    """Persist one auxiliary failure; never raise into the exam pipeline.

    This writes a checkpoint-neutral internal_diagnostic row when possible.
    If the database is unavailable, the failure is logged and dropped rather
    than recursively retried.
    """
    resolved_job = job_id or current_job_id()
    resolved_exam = exam_id
    try:
        if not resolved_exam and resolved_job:
            resolved_exam = repositories.get_exam_id_for_job(resolved_job)
        if not resolved_exam:
            logger.warning(
                "optional failure has no exam record | component=%s | stage=%s | job_id=%s | %s: %s",
                component, stage, resolved_job, type(exc).__name__, exc,
                exc_info=True,
            )
            return
        trace = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        repositories.record_exam_diagnostic_event(
            resolved_exam,
            job_id=resolved_job,
            stage=stage,
            component=component,
            error_type=type(exc).__name__,
            message=str(exc),
            continued=continued,
            traceback_text=trace,
        )
    except Exception as log_exc:
        logger.warning(
            "optional failure diagnostic could not be persisted | component=%s | stage=%s | job_id=%s | %s: %s | diagnostic_error=%s: %s",
            component, stage, resolved_job, type(exc).__name__, exc,
            type(log_exc).__name__, log_exc,
            exc_info=True,
        )
