"""Web-application OAuth (authorization-code flow) for teacher-owned Forms.

Separate from auth.py (desktop flow) which stays as the central-mode admin
path. Handles: authorize URL construction, server-side code exchange, and
Google ID-token verification (signature + audience + verified email).

Security rules enforced here:
- exact redirect URI validation is delegated to Google (registered URI);
- ID token audience must equal our web client_id;
- only email_verified addresses accepted;
- authorization codes / access tokens / refresh tokens / secrets are NEVER
  logged.
"""
from __future__ import annotations

from typing import Any

from app import config
from app.integrations.google_forms.auth import SCOPES

_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"


class WebOAuthError(Exception):
    """Raised for web OAuth configuration or protocol failures."""


def web_configured() -> bool:
    return config.web_oauth_configured()


def build_authorize_url(state: str) -> str:
    if not web_configured():
        raise WebOAuthError(
            "Web OAuth client is not configured "
            "(GOOGLE_WEB_OAUTH_CLIENT_ID / SECRET / REDIRECT_URI)."
        )
    import urllib.parse

    params = {
        "client_id": config.GOOGLE_WEB_OAUTH_CLIENT_ID,
        "redirect_uri": config.GOOGLE_WEB_OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{_AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str) -> tuple[Any, dict[str, Any]]:
    """Exchange an authorization code for credentials + verified identity."""
    if not web_configured():
        raise WebOAuthError("Web OAuth client is not configured.")
    try:
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2.credentials import Credentials
        from google.oauth2 import id_token as google_id_token
    except ImportError as exc:
        raise WebOAuthError("Google API client libraries not installed.") from exc

    import requests

    try:
        resp = requests.post(
            _TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": config.GOOGLE_WEB_OAUTH_CLIENT_ID,
                "client_secret": config.GOOGLE_WEB_OAUTH_CLIENT_SECRET,
                "redirect_uri": config.GOOGLE_WEB_OAUTH_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        # Never include the code or response body (may echo tokens).
        raise WebOAuthError("Google code exchange failed.") from exc

    payload = resp.json()
    creds = Credentials(
        token=payload.get("access_token"),
        refresh_token=payload.get("refresh_token"),
        token_uri=_TOKEN_ENDPOINT,
        client_id=config.GOOGLE_WEB_OAUTH_CLIENT_ID,
        client_secret=config.GOOGLE_WEB_OAUTH_CLIENT_SECRET,
        scopes=SCOPES,
    )

    id_token_value = payload.get("id_token")
    if not id_token_value:
        raise WebOAuthError("Google did not return an ID token.")

    request = GoogleRequest()
    try:
        claims = google_id_token.verify_oauth2_token(
            id_token_value, request, config.GOOGLE_WEB_OAUTH_CLIENT_ID
        )
    except Exception as exc:
        raise WebOAuthError(
            "Google ID token failed verification (audience/signature)."
        ) from exc

    if claims.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise WebOAuthError("Untrusted ID token issuer.")
    if not claims.get("email_verified"):
        raise WebOAuthError("Google account email is not verified.")
    return creds, {
        "sub": claims["sub"],
        "email": claims["email"],
    }


def refresh_credentials(creds: Any) -> Any:
    """Refresh an expired stored credential set in place."""
    from google.auth.transport.requests import Request

    if creds.valid:
        return creds
    if not creds.refresh_token:
        raise WebOAuthError(
            "Google connection expired without a refresh token — reconnect."
        )
    try:
        creds.refresh(Request())
    except Exception as exc:
        raise WebOAuthError(
            "Google connection expired and refresh failed — reconnect."
        ) from exc
    return creds
