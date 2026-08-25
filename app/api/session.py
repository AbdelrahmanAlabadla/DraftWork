"""Signed HttpOnly session cookie for teacher identity.

The cookie payload contains ONLY the verified teacher identity
(google_sub + email). Google OAuth credentials never touch the frontend.
"""
from __future__ import annotations

import secrets

from app import config

_signer = None


def _get_signer():
    global _signer
    if _signer is None:
        from itsdangerous import TimestampSigner

        secret = config.SESSION_SECRET or secrets.token_hex(32)
        _signer = TimestampSigner(secret)
    return _signer


COOKIE_NAME = "dw_session"


def issue_cookie_value(sub: str, email: str) -> str:
    """Create a signed, expiring session token for the verified teacher."""
    return _get_signer().sign(f"{sub}|{email}").decode("utf-8")


def read_cookie_value(value: str | None) -> dict | None:
    """Verify + decode; returns {sub, email} or None when invalid/expired."""
    if not value:
        return None
    try:
        raw = _get_signer().unsign(
            value, max_age=config.SESSION_MAX_AGE_SECONDS
        ).decode("utf-8")
        sub, email = raw.split("|", 1)
        if not sub or not email:
            return None
        return {"sub": sub, "email": email}
    except Exception:
        return None


def cookie_kwargs(secure: bool | None = None) -> dict:
    """FastAPI set_cookie kwargs matching our security requirements."""
    if secure is None:
        # Local development runs on http://127.0.0.1; Secure cookies would not
        # be stored. Production should set Secure=True.
        secure = False
    return {
        "key": COOKIE_NAME,
        "httponly": True,
        "samesite": "lax",
        "secure": secure,
        "max_age": config.SESSION_MAX_AGE_SECONDS,
        "path": "/",
    }
