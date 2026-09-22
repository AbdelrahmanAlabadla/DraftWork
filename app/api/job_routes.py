from __future__ import annotations

import hashlib
import base64
import binascii
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app import config, db
from app.api import repositories
from app.api.routes import GenerateRequest
from app.api.export_routes import (
    DocumentExportRequest,
    _document_archive,
    _selected_models,
    _zip_response,
)
from app.api.session_middleware import optional_user_id, require_session_id
from app.file_storage import get_file_storage
from app.jobs import service as job_service
from app.exports.docx_exporter import render_answers_docx, render_exam_docx
from app.exports.pdf_exporter import render_answers_pdf, render_exam_pdf


router = APIRouter(prefix="/api/v1", tags=["jobs"])


def _document_for_actor(document_id: str, session_id: str, user_id: str | None):
    if user_id is None:
        return repositories.get_document_for_session(document_id, session_id)
    return repositories.get_document_for_session(document_id, session_id, user_id)


def _job_for_actor(job_id: str, session_id: str, user_id: str | None):
    if user_id is None:
        return repositories.get_job_for_session(job_id, session_id)
    return repositories.get_job_for_session(job_id, session_id, user_id)


def _exam_for_actor(exam_id: str, session_id: str, user_id: str | None):
    if user_id is None:
        return repositories.get_exam_for_session(exam_id, session_id)
    return repositories.get_exam_for_session(exam_id, session_id, user_id)

_SUPPORTED_COUNTS = {
    "mcq": "mcq",
    "tf": "true_false",
    "fitb": "fill_in_the_blank",
    "definition": "definition",
    "why": "short_answer",
    "equation": "equation",
    "word_problem": "word_problem",
    "essay": "essay",
}
_VALID_DIFFICULTIES = {"easy", "medium", "hard", "mix"}
_LOGO_PREFIXES = {
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/webp;base64,",
}


def _http_not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc).strip("'"))


def _validate_idempotency_key(value: str | None) -> str | None:
    if value is None:
        if config.REQUIRE_IDEMPOTENCY_KEY:
            raise HTTPException(status_code=400, detail="Idempotency-Key is required")
        return None
    key = value.strip()
    if not key or len(key) > config.IDEMPOTENCY_KEY_MAX_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid Idempotency-Key")
    if any(ord(character) < 33 or ord(character) > 126 for character in key):
        raise HTTPException(status_code=400, detail="Invalid Idempotency-Key")
    return key


def _request_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_logo(value: str, field_name: str) -> None:
    prefix = next((item for item in _LOGO_PREFIXES if value.startswith(item)), None)
    if prefix is None:
        raise HTTPException(status_code=400, detail=f"{field_name} must be PNG, JPEG, or WebP")
    try:
        decoded = base64.b64decode(value[len(prefix):], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not valid base64") from exc
    if not decoded or len(decoded) > config.MAX_LOGO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be no larger than {config.MAX_LOGO_BYTES} bytes",
        )
    valid_signature = (
        (prefix == "data:image/png;base64," and decoded.startswith(b"\x89PNG\r\n\x1a\n"))
        or (prefix == "data:image/jpeg;base64," and decoded.startswith(b"\xff\xd8\xff"))
        or (
            prefix == "data:image/webp;base64,"
            and len(decoded) >= 12
            and decoded.startswith(b"RIFF")
            and decoded[8:12] == b"WEBP"
        )
    )
    if not valid_signature:
        raise HTTPException(status_code=400, detail=f"{field_name} content does not match its image type")


def _validate_child_selection(document: dict[str, Any], child_ids: list[str]) -> None:
    structure_key = document.get("structure_storage_key")
    if not structure_key:
        raise HTTPException(status_code=409, detail="Document structure is unavailable")
    try:
        structure = get_file_storage().read_json(str(structure_key))
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=409, detail="Document structure is unavailable") from exc
    if not isinstance(structure, dict) or not isinstance(structure.get("sections"), list):
        raise HTTPException(status_code=409, detail="Document structure is unavailable")
    available = {
        str(child_id)
        for section in structure.get("sections", [])
        if isinstance(section, dict)
        for child_id in section.get("child_ids", [])
        if child_id
    }
    missing = [child_id for child_id in child_ids if child_id not in available]
    if missing:
        raise HTTPException(status_code=400, detail="One or more selected sections are invalid")


def _generation_payload(body: GenerateRequest) -> dict[str, Any]:
    tasks: list[tuple[str, int]] = []
    if body.question_type is not None:
        if body.question_type not in set(_SUPPORTED_COUNTS.values()):
            raise HTTPException(status_code=400, detail="Unsupported question type")
        if not body.number_of_questions:
            raise HTTPException(status_code=400, detail="number_of_questions is required")
        tasks.append((body.question_type, body.number_of_questions))
    else:
        for key, question_type in _SUPPORTED_COUNTS.items():
            count = int(getattr(body, f"{key}_count") or 0)
            if count:
                tasks.append((question_type, count))

    question_count = sum(count for _, count in tasks)
    if not tasks:
        raise HTTPException(status_code=400, detail="Request at least one question")
    if question_count > config.MAX_QUESTIONS_PER_GENERATION:
        raise HTTPException(
            status_code=400,
            detail=f"A generation job may request at most {config.MAX_QUESTIONS_PER_GENERATION} questions",
        )
    num_models = int(body.num_models or 1)
    if not 1 <= num_models <= config.MAX_MODELS_PER_GENERATION:
        raise HTTPException(
            status_code=400,
            detail=f"num_models must be between 1 and {config.MAX_MODELS_PER_GENERATION}",
        )
    difficulty = str(body.difficulty or "easy").lower()
    if difficulty not in _VALID_DIFFICULTIES:
        raise HTTPException(status_code=400, detail="Unsupported difficulty")
    if not body.child_ids:
        raise HTTPException(status_code=400, detail="Choose at least one section topic")
    if len(body.child_ids) > config.MAX_CHILD_IDS_PER_GENERATION:
        raise HTTPException(
            status_code=400,
            detail=f"Choose at most {config.MAX_CHILD_IDS_PER_GENERATION} section topics",
        )

    metadata = {
        key: value.strip() if isinstance(value, str) else value
        for key, value in {
            "exam_title": body.exam_title,
            "class_name": body.class_name,
            "duration": body.duration,
            "exam_date": body.exam_date,
            "teacher_name": body.teacher_name,
            "footer_message": body.footer_message,
            "left_logo_data": body.left_logo_data,
            "right_logo_data": body.right_logo_data,
        }.items()
        if value
    }
    for field_name in ("left_logo_data", "right_logo_data"):
        value = metadata.get(field_name)
        if value:
            _validate_logo(str(value), field_name)
    return {
        "tasks": tasks,
        "num_models": num_models,
        "difficulty": difficulty,
        "child_ids": list(dict.fromkeys(body.child_ids)),
        "metadata": metadata,
    }


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def create_document(
    file: UploadFile = File(...),
    session_id: str = Depends(require_session_id),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    idempotency_key = _validate_idempotency_key(idempotency_key)
    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    document_id = str(uuid.uuid4())
    storage_key = f"sessions/{session_id}/documents/{document_id}/source.pdf"
    digest = hashlib.sha256()
    size = 0
    storage = get_file_storage()

    with tempfile.TemporaryDirectory(prefix="genexam-upload-") as directory:
        temporary = Path(directory) / "source.pdf"
        with temporary.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                digest.update(chunk)
                destination.write(chunk)
        with temporary.open("rb") as source:
            signature = source.read(5)
        if size == 0 or signature != b"%PDF-":
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF")
        storage.put_file(storage_key, temporary)

    try:
        if idempotency_key:
            document, job, reused = repositories.create_document_with_job_idempotent(
                session_id=session_id,
                filename=filename,
                media_type="application/pdf",
                size_bytes=size,
                sha256=digest.hexdigest(),
                source_storage_key=storage_key,
                document_id=document_id,
                idempotency_key=idempotency_key,
                request_hash=_request_hash(
                    {"filename": filename, "size": size, "sha256": digest.hexdigest()}
                ),
            )
        else:
            document, job = repositories.create_document_with_job(
                session_id=session_id,
                filename=filename,
                media_type="application/pdf",
                size_bytes=size,
                sha256=digest.hexdigest(),
                source_storage_key=storage_key,
                document_id=document_id,
            )
            reused = False
        if reused:
            storage.delete(storage_key)
        else:
            job_service.enqueue_ingestion(str(job["id"]))
            repositories.mark_job_dispatched(str(job["id"]))
    except repositories.IdempotencyConflict as exc:
        storage.delete(storage_key)
        raise HTTPException(status_code=409, detail=str(exc))
    except db.DatabaseUnavailable:
        storage.delete(storage_key)
        raise HTTPException(status_code=503, detail="Application database is unavailable")
    except Exception as exc:
        if 'job' not in locals():
            storage.delete(storage_key)
        raise HTTPException(status_code=503, detail="Background job queue is unavailable")

    return {
        "document_id": str(document["id"]),
        "job_id": str(job["id"]),
        "status": str(job.get("status") or "queued"),
        "idempotency_replayed": reused,
        "message": f"File '{filename}' was accepted for processing.",
    }


@router.get("/documents")
def list_documents(
    session_id: str = Depends(require_session_id),
    user_id: str | None = Depends(optional_user_id),
) -> dict[str, Any]:
    return {"documents": repositories.list_documents_for_session(session_id, user_id)}


@router.get("/documents/{document_id}")
def get_document(
    document_id: uuid.UUID,
    session_id: str = Depends(require_session_id),
    user_id: str | None = Depends(optional_user_id),
) -> dict[str, Any]:
    try:
        document = _document_for_actor(str(document_id), session_id, user_id)
    except repositories.ResourceNotFound as exc:
        raise _http_not_found(exc)
    structure = {}
    key = document.get("structure_storage_key")
    if document["status"] == "ready" and key:
        structure = get_file_storage().read_json(str(key))
    return {
        "document_id": str(document["id"]),
        "status": document["status"],
        "filename": document["original_filename"],
        "stats": document.get("stats") or {},
        "structure": structure,
        "error": (
            {"code": document.get("error_code"), "message": document.get("error_message")}
            if document.get("error_code") else None
        ),
    }


@router.post(
    "/documents/{document_id}/exam-jobs",
    status_code=status.HTTP_202_ACCEPTED,
)
def create_exam_job(
    document_id: uuid.UUID,
    body: GenerateRequest,
    session_id: str = Depends(require_session_id),
    user_id: str | None = Depends(optional_user_id),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    idempotency_key = _validate_idempotency_key(idempotency_key)
    payload = _generation_payload(body)
    try:
        document = _document_for_actor(str(document_id), session_id, user_id)
        if document["status"] != "ready":
            raise repositories.InvalidState("Document is not ready for exam generation")
        _validate_child_selection(document, payload["child_ids"])
        job = repositories.create_generation_job(
            session_id=session_id,
            document_id=str(document_id),
            request_data=payload,
            user_id=user_id,
            idempotency_key=idempotency_key,
            request_hash=_request_hash(payload) if idempotency_key else None,
        )
    except repositories.ResourceNotFound as exc:
        raise _http_not_found(exc)
    except repositories.InvalidState as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except repositories.JobLimitExceeded as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except repositories.GenerationRateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="Only two generation requests per minute are allowed",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )
    except repositories.IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    reused = bool(job.pop("_idempotency_reused", False))
    if not reused:
        try:
            job_service.enqueue_generation(str(job["id"]))
            repositories.mark_job_dispatched(str(job["id"]))
        except Exception:
            raise HTTPException(status_code=503, detail="Background job queue is unavailable")
    return {
        "job_id": str(job["id"]),
        "document_id": str(document_id),
        "status": str(job.get("status") or "queued"),
        "idempotency_replayed": reused,
    }


@router.get("/jobs/{job_id}")
def get_job(
    job_id: uuid.UUID,
    session_id: str = Depends(require_session_id),
    user_id: str | None = Depends(optional_user_id),
) -> dict[str, Any]:
    try:
        job = _job_for_actor(str(job_id), session_id, user_id)
    except repositories.ResourceNotFound as exc:
        raise _http_not_found(exc)
    return {
        "job_id": str(job["id"]),
        "document_id": str(job["document_id"]) if job.get("document_id") else None,
        "type": job["type"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "result": job.get("result_data") or None,
        "error": (
            {"code": job.get("error_code"), "message": job.get("error_message")}
            if job.get("error_code") else None
        ),
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }


@router.delete("/jobs/{job_id}")
def cancel_job(
    job_id: uuid.UUID,
    session_id: str = Depends(require_session_id),
    user_id: str | None = Depends(optional_user_id),
) -> dict[str, Any]:
    try:
        owned_job = _job_for_actor(str(job_id), session_id, user_id)
        job = repositories.cancel_job(str(owned_job["id"]))
    except repositories.ResourceNotFound as exc:
        raise _http_not_found(exc)

    if job["status"] == "cancelled":
        # With a process-based Celery pool this terminates the active task child.
        # A queued/redelivered message is also harmless because claim_job only
        # accepts queued or retrying database jobs.
        job_service.terminate_job(
            str(job_id),
            document_id=str(job["document_id"]) if job.get("document_id") else None,
            session_id=str(job["session_id"]),
            cleanup_document=job["type"] == "document_ingestion",
        )
    return {
        "job_id": str(job["id"]),
        "document_id": str(job["document_id"]) if job.get("document_id") else None,
        "status": job["status"],
        "stage": job["stage"],
    }


@router.get("/exams/{exam_id}")
def get_exam(
    exam_id: str,
    session_id: str = Depends(require_session_id),
    user_id: str | None = Depends(optional_user_id),
) -> dict[str, Any]:
    try:
        record = _exam_for_actor(exam_id, session_id, user_id)
    except repositories.ResourceNotFound as exc:
        raise _http_not_found(exc)
    return record


def _owned_exam(
    exam_id: str, session_id: str, user_id: str | None = None
) -> dict[str, Any]:
    try:
        return _exam_for_actor(exam_id, session_id, user_id)
    except repositories.ResourceNotFound as exc:
        raise _http_not_found(exc)


@router.post("/exams/{exam_id}/export/pdf")
def export_pdf(
    exam_id: str,
    body: DocumentExportRequest | None = None,
    session_id: str = Depends(require_session_id),
    user_id: str | None = Depends(optional_user_id),
) -> StreamingResponse:
    record = _owned_exam(exam_id, session_id, user_id)
    selected = _selected_models(record, body)
    archive = _document_archive(record, selected, "pdf", render_exam_pdf, render_answers_pdf)
    return _zip_response(archive)


@router.post("/exams/{exam_id}/export/docx")
def export_docx(
    exam_id: str,
    body: DocumentExportRequest | None = None,
    session_id: str = Depends(require_session_id),
    user_id: str | None = Depends(optional_user_id),
) -> StreamingResponse:
    record = _owned_exam(exam_id, session_id, user_id)
    selected = _selected_models(record, body)
    archive = _document_archive(record, selected, "docx", render_exam_docx, render_answers_docx)
    return _zip_response(archive)
