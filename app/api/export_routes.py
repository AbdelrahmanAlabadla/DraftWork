"""Export endpoints: PDF / DOCX / Google Forms for a stored exam_id.

Export is downstream of generation; failures here never mutate or delete the
stored exam.
"""
from __future__ import annotations

import io

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr

from app.api import exam_store
from app.config import GOOGLE_FORMS_ENABLED
from app.exports.docx_exporter import render_exam_docx
from app.exports.pdf_exporter import render_exam_pdf
from app.integrations.google_forms.auth import GoogleAuthError, credentials_configured
from app.integrations.google_forms.client import GoogleFormsError
from app.integrations.google_forms.exporter import ExportError, export_exam
from app.integrations.google_forms.user_tokens import NotConnected
from app.logging_conf import get_logger, set_request_id

logger = get_logger("EXPORT")

router = APIRouter(prefix="/exams/{exam_id}/export", tags=["export"])


class FormsExportRequest(BaseModel):
    """Optional body for the Google Forms export: teacher emails to share with."""
    share_with: list[EmailStr] = []


def _load_record(exam_id: str) -> dict:
    try:
        return exam_store.get_exam(exam_id)
    except exam_store.ExamNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/pdf")
def export_pdf(exam_id: str) -> StreamingResponse:
    record = _load_record(exam_id)
    set_request_id(exam_id)
    try:
        pdf_bytes = render_exam_pdf(record)
    except Exception as exc:
        logger.error("PDF export failed | exam_id=%s | exc=%s", exam_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}")
    filename = f"exam_{exam_id}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/docx")
def export_docx(exam_id: str) -> StreamingResponse:
    record = _load_record(exam_id)
    set_request_id(exam_id)
    try:
        docx_bytes = render_exam_docx(record)
    except Exception as exc:
        logger.error("DOCX export failed | exam_id=%s | exc=%s", exam_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"DOCX export failed: {exc}")
    filename = f"exam_{exam_id}.docx"
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/google-forms")
def export_google_forms(exam_id: str, request: Request,
                        body: FormsExportRequest | None = None) -> dict:
    from app import config as app_config
    from app.api import session as session_mod
    from app.integrations.google_forms import user_tokens
    from app.integrations.google_forms.web_auth import (
        WebOAuthError, refresh_credentials, web_configured,
    )

    if not GOOGLE_FORMS_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Google Forms export is disabled. Set GOOGLE_FORMS_ENABLED=true in .env.",
        )

    mode = app_config.GOOGLE_FORMS_MODE
    if mode == "disabled":
        raise HTTPException(
            status_code=503,
            detail='Google Forms export is disabled (GOOGLE_FORMS_MODE="disabled").',
        )

    creds = None
    share_with = list(body.share_with) if body else []

    if mode == "teacher":
        if not web_configured():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Teacher-owned Google Forms require the Web OAuth client "
                    "configuration (GOOGLE_WEB_OAUTH_CLIENT_ID/SECRET/REDIRECT_URI) "
                    "or set GOOGLE_FORMS_MODE=central."
                ),
            )
        # Identity comes ONLY from the signed session — never the request body.
        teacher = session_mod.read_cookie_value(
            request.cookies.get(session_mod.COOKIE_NAME)
        )
        if teacher is None:
            raise HTTPException(
                status_code=401,
                detail="Connect Google Account first.",
            )
        try:
            creds = refresh_credentials(user_tokens.get(teacher["sub"]))
        except NotConnected:
            raise HTTPException(
                status_code=401,
                detail="Connect Google Account first.",
            )
        except WebOAuthError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        # Teacher owns the form; explicit share list is ignored in this mode.
        share_with = []
    else:  # central fallback
        if not credentials_configured():
            raise HTTPException(
                status_code=503,
                detail="Google Forms authentication is not configured. Place your OAuth client file at data/google_forms/oauth_client.json.",
            )
        # In central mode only an explicitly typed email is shared with.
        if share_with:
            pass  # intentional use of body-provided addresses (Feature 1)

    record = _load_record(exam_id)
    set_request_id(exam_id)
    try:
        result = export_exam(record, share_with=share_with, creds=creds)
    except (GoogleAuthError,) as exc:
        raise HTTPException(status_code=503, detail=f"Google Forms export failed: {exc}")
    except GoogleFormsError as exc:
        raise HTTPException(status_code=502, detail=f"Google Forms export failed: {exc}")
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=f"Google Forms export failed: {exc}")

    # Partial failures are reported per model without failing the whole request.
    status = 200 if result["exports"] else 502
    if status != 200:
        raise HTTPException(status_code=status, detail={
            "message": "No models could be exported to Google Forms",
            "errors": result["errors"],
        })
    return {"exam_id": exam_id, **result}
