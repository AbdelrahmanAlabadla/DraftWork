"""Teacher Google sign-in routes (authorization-code flow + session cookie).

Identity rules:
- the teacher identity comes ONLY from the verified Google ID token;
- OAuth credentials are stored server-side keyed by google_sub;
- the session cookie carries just {sub, email}, signed server-side.
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from app.api import session as session_mod
from app.integrations.google_forms import user_tokens
from app.integrations.google_forms.web_auth import (
    WebOAuthError,
    build_authorize_url,
    exchange_code,
    web_configured,
)
from app.logging_conf import get_logger

logger = get_logger("GOOGLE_AUTH")

router = APIRouter(prefix="/auth/google", tags=["auth"])

# Short-lived CSRF state store (in-memory): state -> unused marker.
_pending_states: dict[str, bool] = {}


def _new_state() -> str:
    state = secrets.token_urlsafe(32)
    # Bound memory: keep at most 20 pending flows.
    while len(_pending_states) >= 20:
        _pending_states.pop(next(iter(_pending_states)))
    _pending_states[state] = True
    return state


@router.get("/login")
def login():
    if not web_configured():
        raise HTTPException(
            status_code=503,
            detail="Google sign-in is not configured on this server "
                   "(missing Web OAuth client settings).",
        )
    state = _new_state()
    return RedirectResponse(url=build_authorize_url(state))


@router.get("/callback")
def callback(code: str | None = None, state: str | None = None):
    if not web_configured():
        raise HTTPException(status_code=503, detail="Google sign-in is not configured.")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state.")
    if not _pending_states.pop(state, False):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")
    try:
        creds, identity = exchange_code(code)
    except WebOAuthError as exc:
        logger.warning("Google connect failed | reason=%s", exc)
        raise HTTPException(status_code=401, detail=str(exc))

    user_tokens.save(identity["sub"], creds, identity["email"])
    cookie_value = session_mod.issue_cookie_value(identity["sub"], identity["email"])
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(value=cookie_value, **session_mod.cookie_kwargs())
    return response


@router.get("/me")
def me(request: Request):
    teacher = session_mod.read_cookie_value(request.cookies.get(session_mod.COOKIE_NAME))
    connected = teacher is not None and user_tokens.email_of(teacher["sub"]) is not None
    return {
        "connected": connected,
        "email": teacher["email"] if connected else None,
    }


@router.post("/logout")
def logout(request: Request, response: Response):
    teacher = session_mod.read_cookie_value(request.cookies.get(session_mod.COOKIE_NAME))
    if teacher is not None:
        user_tokens.remove(teacher["sub"])
    response.delete_cookie(key=session_mod.COOKIE_NAME, path="/")
    return {"connected": False}
