from __future__ import annotations

import secrets

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app import config, db
from app.api import repositories
from app.api.request_context import reset_session_id, set_session_id


def _needs_session(path: str) -> bool:
    # The versioned API is the durable, isolated contract.  Unversioned routes
    # remain temporarily available for existing local scripts and pipeline
    # regression tests while the browser migrates to /api/v1.
    return path.startswith("/api/v1/")


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
                new_token = secrets.token_urlsafe(config.SESSION_TOKEN_BYTES)
                session = repositories.create_session(
                    repositories.hash_session_token(new_token)
                )
        except db.DatabaseUnavailable:
            return JSONResponse(
                status_code=503,
                content={"detail": "Application database is temporarily unavailable"},
            )

        session_id = str(session["id"])
        request.state.session_id = session_id
        context_token = set_session_id(session_id)
        try:
            response = await call_next(request)
        finally:
            reset_session_id(context_token)

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
