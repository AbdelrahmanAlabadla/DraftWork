"""Google OAuth2 authentication for the Forms API (desktop flow).

Credentials live OUTSIDE version control:
- client secrets: data/google_forms/oauth_client.json (user provides)
- cached user token: data/google_forms/token.json (created on first consent)

No secret values are ever logged.
"""
from __future__ import annotations

from pathlib import Path

from app import config
from app.logging_conf import get_logger

logger = get_logger("GOOGLE_FORMS")

SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    # Narrow Drive scope: access only files created by this app (used to
    # share generated forms with teachers).
    "https://www.googleapis.com/auth/drive.file",
]


class GoogleAuthError(Exception):
    """Raised when OAuth credentials are missing or authorization fails."""


def _credentials_file() -> Path:
    return Path(config.GOOGLE_OAUTH_CLIENT_FILE)


def credentials_configured() -> bool:
    return _credentials_file().exists()


def get_credentials():
    """Return valid Google credentials, refreshing/re-authorizing as needed."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise GoogleAuthError(
            "Google API client libraries not installed. "
            "Run: pip install google-api-python-client google-auth-oauthlib"
        ) from exc

    token_path = Path(config.GOOGLE_OAUTH_TOKEN_FILE)
    creds = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as exc:
            logger.warning("Token cache unreadable; re-authorizing | reason=%s", type(exc).__name__)
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds, token_path)
            return creds
        except Exception as exc:
            raise GoogleAuthError(
                "Google token expired and refresh failed — re-authorize the app."
            ) from exc

    client_file = _credentials_file()
    if not client_file.exists():
        raise GoogleAuthError(
            f"OAuth client file not found at '{client_file}'. "
            "Place your Google OAuth desktop client JSON there and try again."
        )
    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
        creds = flow.run_local_server(port=0)
    except Exception as exc:
        raise GoogleAuthError(f"Google authorization failed: {exc}") from exc
    _save_token(creds, token_path)
    return creds


def _save_token(creds, token_path: Path) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
