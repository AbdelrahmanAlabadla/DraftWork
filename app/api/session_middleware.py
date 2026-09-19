from __future__ import annotations

import secrets

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app import config, db
from app.api import repositories
from app.api.request_context import reset_session_id, set_session_id
from app.auth.clerk import authenticate, fetch_profile
from app.errors import ErrorCode, public_error_payload


def _needs_session(path: str) -> bool:
    # The versioned API is the durable, isolated contract.  Unversioned routes
    # remain temporarily available for existing local scripts and pipeline
    # regression tests while the browser migrates to /api/v1.
    return path.startswith("/api/v1/") and path != "/api/v1/auth/config"


def _create_session() -> tuple[dict, str]:
    token = secrets.token_urlsafe(config.SESSION_TOKEN_BYTES)
    return repositories.create_session(repositories.hash_session_token(token)), token


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if (
            request.url.path in {"/upload", "/generate", "/documents"}
            or request.url.path.startswith("/exams/")
        ) and not config.ENABLE_LEGACY_SYNC_API:
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        if not _needs_session(request.url.path):
            return await call_next(request)

        cookie_name = config.SESSION_COOKIE_NAME
        raw_token = request.cookies.get(cookie_name)
        new_token: str | None = None
        try:
            session = (
                repositories.resolve_session(repositories.hash_session_token(raw_token))
                if raw_token
                else None
            )
            if session is None:
                session, new_token = _create_session()

            try:
                identity = await authenticate(request)
            except ValueError:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired authentication token"},
                )
            except Exception:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Authentication service is temporarily unavailable"},
                )

            current_owner = str(session["user_id"]) if session.get("user_id") else None
            if identity is None and current_owner is not None:
                repositories.revoke_session(str(session["id"]))
                session, new_token = _create_session()
            elif identity is not None:
                user = repositories.get_user_by_clerk_id(identity.clerk_user_id)
                if user is None:
                    try:
                        profile = await fetch_profile(identity.clerk_user_id)
                    except Exception:
                        profile = {}
                    user = repositories.upsert_user(identity.clerk_user_id, **profile)
                user_id = str(user["id"])
                if current_owner is not None and current_owner != user_id:
                    repositories.revoke_session(str(session["id"]))
                    session, new_token = _create_session()
                try:
                    session = repositories.claim_session(str(session["id"]), user_id)
                except repositories.SessionOwnershipConflict:
                    # A concurrent login claimed the anonymous cookie first.
                    # Keep that claim intact and give this identity a new session.
                    session, new_token = _create_session()
                    session = repositories.claim_session(str(session["id"]), user_id)
                request.state.user_id = user_id
                request.state.clerk_user_id = identity.clerk_user_id
        except db.DatabaseUnavailable:
            request_id = getattr(request.state, "request_id", "-")
            return JSONResponse(
                status_code=503,
                content=public_error_payload(
                    ErrorCode.DATABASE_UNAVAILABLE,
                    "Application database is temporarily unavailable",
                    request_id,
                    retryable=True,
                ),
            )

        session_id = str(session["id"])
        request.state.session_id = session_id
        context_token = set_session_id(session_id)
        try:
            response = await call_next(request)
        finally:
            reset_session_id(context_token)

        if getattr(request.state, "rotate_session_after_response", False):
            try:
                repositories.revoke_session(session_id)
                _, new_token = _create_session()
            except db.DatabaseUnavailable:
                return JSONResponse(status_code=503, content={"detail": "Database unavailable"})

        if new_token is not None:
            response.set_cookie(
                key=cookie_name,
                value=new_token,
                max_age=config.SESSION_TTL_SECONDS,
                httponly=True,
                secure=config.SESSION_COOKIE_SECURE,
                samesite=config.SESSION_COOKIE_SAMESITE,
                path="/",
            )
        return response


def require_session_id(request: Request) -> str:
    session_id = getattr(request.state, "session_id", None)
    if not session_id:
        raise HTTPException(status_code=401, detail="Session is required")
    return str(session_id)


def optional_user_id(request: Request) -> str | None:
    user_id = getattr(request.state, "user_id", None)
    return str(user_id) if user_id else None


def require_user_id(request: Request) -> str:
    user_id = optional_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication is required")
    return user_id
