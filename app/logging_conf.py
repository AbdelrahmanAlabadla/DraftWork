from __future__ import annotations

import contextvars
import logging
import sys

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

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | Request=%(request_id)s | %(component)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RequestContextFilter(logging.Filter):
    """Attach the request id and component name to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", None) or _REQUEST_ID_VAR.get()
        record.component = getattr(record, "component", None) or record.name
        return True


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
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(RequestContextFilter())

    root.handlers.clear()
    root.addHandler(handler)
    _configured = True


def get_logger(component: str) -> logging.Logger:
    """Return a logger bound to a component name (e.g. 'PARSER', 'RETRIEVAL')."""
    return logging.getLogger(component)


def set_request_id(request_id: str) -> None:
    _REQUEST_ID_VAR.set(request_id)
