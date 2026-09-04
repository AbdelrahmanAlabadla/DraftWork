"""Export endpoints: PDF / DOCX / Google Forms for a stored exam_id.

Export is downstream of generation; failures here never mutate or delete the
stored exam.
"""
from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr

from app.api import exam_store
from app.config import GOOGLE_FORMS_ENABLED
from app.exports.common import document_export_filenames
from app.exports.docx_exporter import render_answers_docx, render_exam_docx
from app.exports.pdf_exporter import render_answers_pdf, render_exam_pdf
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


class DocumentExportRequest(BaseModel):
    """Optional model selection for downloadable document archives."""
    model_numbers: list[int] | None = None


def _load_record(exam_id: str) -> dict:
    try:
        return exam_store.get_exam(exam_id)
    except exam_store.ExamNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def _selected_models(
    record: dict[str, Any], body: DocumentExportRequest | None
) -> list[dict[str, Any]]:
    exams = [exam for exam in (record.get("exams") or []) if isinstance(exam, dict)]
    available = {
        int(exam.get("model_number") or 1): exam
        for exam in exams
    }
    requested = (
        list(available)
        if body is None or body.model_numbers is None
        else list(body.model_numbers)
    )
    if not requested:
        raise HTTPException(status_code=400, detail="Select at least one exam model to export.")
    if len(requested) != len(set(requested)):
        raise HTTPException(status_code=400, detail="Duplicate exam model selections are not allowed.")
    unknown = [number for number in requested if number not in available]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown exam model selection: {', '.join(map(str, unknown))}",
        )
    selected = set(requested)
    return [exam for number, exam in available.items() if number in selected]


def _document_archive(
    record: dict[str, Any],
    selected: list[dict[str, Any]],
    extension: str,
    render_student: Callable[[dict[str, Any], dict[str, Any]], bytes],
    render_answers: Callable[[dict[str, Any], dict[str, Any]], bytes],
) -> bytes:
    metadata = dict(record.get("metadata") or {})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for exam in selected:
            model_number = int(exam.get("model_number") or 1)
            exam_name, answers_name = document_export_filenames(
                metadata, model_number, extension
            )
            archive.writestr(exam_name, render_student(exam, metadata))
            archive.writestr(answers_name, render_answers(exam, metadata))
    return output.getvalue()


def _zip_response(content: bytes) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="SmartExam_Export.zip"'
        },
    )


@router.post("/pdf")
def export_pdf(
    exam_id: str, body: DocumentExportRequest | None = None
) -> StreamingResponse:
    record = _load_record(exam_id)
    selected = _selected_models(record, body)
    set_request_id(exam_id)
    try:
        archive = _document_archive(
            record, selected, "pdf", render_exam_pdf, render_answers_pdf
        )
    except Exception as exc:
        logger.error("PDF export failed | exam_id=%s | exc=%s", exam_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}")
    return _zip_response(archive)


@router.post("/docx")
def export_docx(
    exam_id: str, body: DocumentExportRequest | None = None
) -> StreamingResponse:
    record = _load_record(exam_id)
    selected = _selected_models(record, body)
    set_request_id(exam_id)
    try:
        archive = _document_archive(
            record, selected, "docx", render_exam_docx, render_answers_docx
        )
    except Exception as exc:
        logger.error("DOCX export failed | exam_id=%s | exc=%s", exam_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"DOCX export failed: {exc}")
    return _zip_response(archive)


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
