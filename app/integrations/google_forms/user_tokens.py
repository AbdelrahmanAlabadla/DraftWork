"""Per-teacher Google credential storage (PROTOTYPE: in-memory).

LIMITATION — read this:
    Server restart -> every stored connection disappears and each teacher must
    reconnect Google. Do NOT treat this as production-ready storage. The
    public interface below is the only contract the exporter relies on, so it
    can later be replaced by encrypted database storage without changing any
    caller.

Credentials are keyed by Google's stable subject identifier (`sub`), NOT by
email, per the architecture decision. Raw tokens are never written to disk
and never returned to callers.
"""
from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_store: dict[str, tuple[Any, str]] = {}
# sub -> (credentials, email)


class NotConnected(KeyError):
    """Teacher has not connected a Google account."""


def save(sub: str, credentials: Any, email: str) -> None:
    with _lock:
        _store[sub] = (credentials, email)


def get(sub: str) -> Any:
    """Return stored Credentials for this google_sub or raise NotConnected."""
    with _lock:
        entry = _store.get(sub)
    if entry is None:
        raise NotConnected(f"No connected Google account for this teacher.")
    return entry[0]


def email_of(sub: str) -> str | None:
    with _lock:
        entry = _store.get(sub)
    return entry[1] if entry else None


def remove(sub: str) -> None:
    with _lock:
        _store.pop(sub, None)


def clear() -> None:
    """Test helper."""
    with _lock:
        _store.clear()
