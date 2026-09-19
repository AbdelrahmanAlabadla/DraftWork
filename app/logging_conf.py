from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Centralized logging configuration for the whole application.
#
# Log format:
#   2026-08-01 15:42:18 | INFO | Request=abc123 | PARSER | Starting LlamaParse...
#
# Every logger obtained via get_logger("COMPONENT") automatically emits the
# request/document id (from the context variable) and its component name.
# ---------------------------------------------------------------------------

_REQUEST_ID_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)
_JOB_ID_VAR: contextvars.ContextVar[str] = contextvars.ContextVar("job_id", default="-")
_DOCUMENT_ID_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "document_id", default="-"
)

class RequestContextFilter(logging.Filter):
    """Attach the request id and component name to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", None) or _REQUEST_ID_VAR.get()
        record.component = getattr(record, "component", None) or record.name
        record.job_id = getattr(record, "job_id", None) or _JOB_ID_VAR.get()
        record.document_id = (
            getattr(record, "document_id", None) or _DOCUMENT_ID_VAR.get()
        )
        return True


_SECRET_RE = re.compile(
    r"(?i)(authorization|api[_-]?key|password|secret|token|cookie)"
    r"(\s*[=:]\s*)(?:bearer\s+)?([^\s,;]+)"
)
_URL_PASSWORD_RE = re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^:/\s]+:)([^@/\s]+)(@)")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = _SECRET_RE.sub(r"\1\2[REDACTED]", record.getMessage())
        message = _URL_PASSWORD_RE.sub(r"\1[REDACTED]\3", message)
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "event": getattr(record, "event", "log"),
            "message": message,
            "request_id": getattr(record, "request_id", "-"),
        }
        job_id = getattr(record, "job_id", "-")
        document_id = getattr(record, "document_id", "-")
        if job_id != "-":
            payload["job_id"] = job_id
        if document_id != "-":
            payload["document_id"] = document_id
        for key in (
            "stage", "duration_ms", "provider", "model", "input_tokens",
            "output_tokens", "total_tokens", "status_code",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, default=str)


_configured = False


def configure_logging(level: str | int = "INFO") -> None:
    global _configured
    if _configured:
        return

    numeric_level = (
        getattr(logging, str(level).upper(), logging.INFO)
        if isinstance(level, str)
        else level
    )
    root = logging.getLogger()
    root.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestContextFilter())

    root.handlers.clear()
    root.addHandler(handler)
    _configured = True


def get_logger(component: str) -> logging.Logger:
    """Return a logger bound to a component name (e.g. 'PARSER', 'RETRIEVAL')."""
    return logging.getLogger(component)


def set_request_id(request_id: str) -> None:
    _REQUEST_ID_VAR.set(request_id)


def set_job_context(job_id: str, document_id: str = "-") -> None:
    _JOB_ID_VAR.set(job_id)
    _DOCUMENT_ID_VAR.set(document_id)


def current_job_id() -> str | None:
    value = _JOB_ID_VAR.get()
    return None if value == "-" else value
