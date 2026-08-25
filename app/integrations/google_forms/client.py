"""Thin wrapper around the Google Forms API.

Bounded retries apply ONLY to idempotent calls (batchUpdate item creation is
safe to retry per-chunk because a failed chunk leaves no partial items).
Form CREATION is never auto-retried: a timeout after a successful server-side
create would produce a duplicate form.
"""
from __future__ import annotations

import time
from typing import Any

from app import config
from app.integrations.google_forms.auth import get_credentials
from app.logging_conf import get_logger

logger = get_logger("GOOGLE_FORMS")

_MAX_BATCH_RETRIES = 3
_RETRY_DELAY_SECONDS = 1.0


class GoogleFormsError(Exception):
    """Raised when a Google Forms API operation fails."""


def get_service(creds=None):
    """Build a Forms service; explicit creds (teacher-owned) or desktop flow."""
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise GoogleFormsError(
            "Google API client libraries not installed."
        ) from exc
    if creds is None:
        creds = get_credentials()
    return build("forms", "v1", credentials=creds)


def get_drive_service(creds=None):
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise GoogleFormsError(
            "Google API client libraries not installed."
        ) from exc
    if creds is None:
        creds = get_credentials()
    return build("drive", "v3", credentials=creds)


def create_form(title: str, creds=None) -> dict[str, Any]:
    """Create an empty quiz-enabled form. NOT retried (duplicate protection)."""
    service = get_service(creds)
    t0 = time.perf_counter()
    try:
        form = service.forms().create(body={"info": {"title": title}}).execute()
    except Exception as exc:
        if type(exc).__name__ == "HttpError":
            status = getattr(getattr(exc, "resp", None), "status", "?")
            if status == 429:
                raise GoogleFormsError("Google Forms rate limit reached (429).") from exc
            raise GoogleFormsError(f"Form creation failed (HTTP {status}).") from exc
        raise GoogleFormsError(f"Form creation failed: {exc}") from exc
    form_id = form["formId"]
    logger.info("Form created | form_id=%s | time=%.2fs", form_id, time.perf_counter() - t0)

    try:
        service.forms().batchUpdate(
            formId=form_id,
            body={"requests": [{
                "updateSettings": {
                    "settings": {"quizSettings": {"isQuiz": True}},
                    "updateMask": "quizSettings",
                }
            }]},
        ).execute()
    except Exception as exc:
        # Quiz toggle failed but the form exists; surface it as partial success info.
        logger.warning("Quiz-mode enable failed | form_id=%s | error=%s", form_id, exc)

    return {
        "form_id": form_id,
        "title": title,
        "edit_url": f"https://docs.google.com/forms/d/{form_id}/edit",
        "view_url": form.get("responderUri")
        or f"https://docs.google.com/forms/d/{form_id}/viewform",
    }


def batch_update(form_id: str, requests_list: list[dict[str, Any]],
                 creds=None) -> int:
    """Apply requests in chunks with bounded retries. Returns applied request count."""
    if not requests_list:
        return 0
    service = get_service(creds)
    applied = 0
    for start in range(0, len(requests_list), 50):
        chunk = requests_list[start:start + 50]
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_BATCH_RETRIES + 1):
            try:
                result = service.forms().batchUpdate(
                    formId=form_id, body={"requests": chunk}
                ).execute()
                applied += len(result.get("replies", []))
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                status = getattr(getattr(exc, "resp", None), "status", None)
                # 4xx (except 429/500-class) are permanent — do not retry.
                if status is not None and status < 500 and status != 429:
                    raise GoogleFormsError(
                        f"batchUpdate failed permanently (HTTP {status})."
                    ) from exc
                time.sleep(_RETRY_DELAY_SECONDS * attempt)
        if last_exc is not None:
            raise GoogleFormsError(
                f"batchUpdate failed after {_MAX_BATCH_RETRIES} attempts: {last_exc}"
            ) from last_exc
    return applied


QUESTIONS_PER_PAGE = config.GOOGLE_FORMS_QUESTIONS_PER_PAGE


def share_form(form_id: str, email: str, role: str = "writer",
               message: str | None = None, creds=None) -> None:
    """Grant a user permission on a form via the Drive API.

    Every Form is also a Drive file (form_id == Drive file id), so sharing is
    a Drive permissions.create call. The teacher receives an invitation email
    from Google and gets editor access, including the Responses tab.
    Never retried: a failed share can simply be repeated manually and must not
    delay or fail the export.

    GUARD: only explicit user permissions are ever created — never `anyone`.
    """
    if email == "anyone" or role not in ("writer", "reader", "commenter"):
        raise GoogleFormsError("Refusing to create a non-user or unrestricted share.")
    drive = get_drive_service(creds)
    body: dict[str, Any] = {"type": "user", "role": role, "emailAddress": email}
    kwargs: dict[str, Any] = {
        "fileId": form_id,
        "body": body,
        "sendNotificationEmail": True,
    }
    if message:
        kwargs["emailMessage"] = message
    try:
        drive.permissions().create(**kwargs).execute()
    except Exception as exc:
        status = getattr(getattr(exc, "resp", None), "status", None)
        if status == 404:
            raise GoogleFormsError(f"Cannot share: form {form_id} not found.") from exc
        if status == 403:
            raise GoogleFormsError(
                f"Cannot share with {email}: permission denied by Google."
            ) from exc
        raise GoogleFormsError(f"Cannot share with {email}: {exc}") from exc
    logger.info("Form shared | form_id=%s | role=%s", form_id, role)
