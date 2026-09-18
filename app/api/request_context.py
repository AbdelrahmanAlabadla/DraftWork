from __future__ import annotations

import contextvars


_SESSION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)


def set_session_id(session_id: str | None) -> contextvars.Token:
    return _SESSION_ID.set(session_id)


def reset_session_id(token: contextvars.Token) -> None:
    _SESSION_ID.reset(token)


def current_session_id() -> str | None:
    return _SESSION_ID.get()
