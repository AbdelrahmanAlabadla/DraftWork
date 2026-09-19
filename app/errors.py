from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_FILE = "INVALID_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_SELECTION = "INVALID_SELECTION"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_NOT_READY = "DOCUMENT_NOT_READY"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    EXAM_NOT_FOUND = "EXAM_NOT_FOUND"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    JOB_LIMIT_REACHED = "JOB_LIMIT_REACHED"
    RATE_LIMITED = "RATE_LIMITED"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    QUEUE_UNAVAILABLE = "QUEUE_UNAVAILABLE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    WORKER_LOST = "WORKER_LOST"
    GENERATION_FAILED = "GENERATION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(slots=True)
class AppError(Exception):
    code: ErrorCode
    message: str
    status_code: int = 400
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


def public_error_payload(
    code: ErrorCode | str,
    message: str,
    request_id: str,
    *,
    retryable: bool = False,
) -> dict[str, object]:
    value = code.value if isinstance(code, ErrorCode) else str(code)
    error = {
        "code": value,
        "message": message,
        "request_id": request_id,
        "retryable": retryable,
    }
    # Keep detail during the frontend migration to the stable error envelope.
    return {"detail": message, "error": error}
