from __future__ import annotations

import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
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
from app.api.session_middleware import require_session_id
from app.file_storage import get_file_storage
from app.jobs import service as job_service
from app.exports.docx_exporter import render_answers_docx, render_exam_docx
from app.exports.pdf_exporter import render_answers_pdf, render_exam_pdf


router = APIRouter(prefix="/api/v1", tags=["jobs"])

_SUPPORTED_COUNTS = {
    "mcq": "mcq",
    "tf": "true_false",
    "fitb": "fill_in_the_blank",
    "why": "short_answer",
    "essay": "essay",
}
_VALID_DIFFICULTIES = {"easy", "medium", "hard", "mix"}


def _http_not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc).strip("'"))


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
) -> dict[str, Any]:
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
        document, job = repositories.create_document_with_job(
            session_id=session_id,
            filename=filename,
            media_type=file.content_type or "application/pdf",
            size_bytes=size,
            sha256=digest.hexdigest(),
            source_storage_key=storage_key,
            document_id=document_id,
        )
        job_service.enqueue_ingestion(str(job["id"]))
        repositories.mark_job_dispatched(str(job["id"]))
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
        "status": "queued",
        "message": f"File '{filename}' was accepted for processing.",
    }


@router.get("/documents")
def list_documents(
    session_id: str = Depends(require_session_id),
) -> dict[str, Any]:
    return {"documents": repositories.list_documents_for_session(session_id)}


@router.get("/documents/{document_id}")
def get_document(
    document_id: uuid.UUID,
    session_id: str = Depends(require_session_id),
) -> dict[str, Any]:
    try:
        document = repositories.get_document_for_session(str(document_id), session_id)
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
) -> dict[str, Any]:
    payload = _generation_payload(body)
    try:
        job = repositories.create_generation_job(
            session_id=session_id,
            document_id=str(document_id),
            request_data=payload,
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

    try:
        job_service.enqueue_generation(str(job["id"]))
        repositories.mark_job_dispatched(str(job["id"]))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Background job queue is unavailable")
    return {"job_id": str(job["id"]), "document_id": str(document_id), "status": "queued"}


@router.get("/jobs/{job_id}")
def get_job(
    job_id: uuid.UUID,
    session_id: str = Depends(require_session_id),
) -> dict[str, Any]:
    try:
        job = repositories.get_job_for_session(str(job_id), session_id)
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


@router.get("/exams/{exam_id}")
def get_exam(
    exam_id: str,
    session_id: str = Depends(require_session_id),
) -> dict[str, Any]:
    try:
        record = repositories.get_exam_for_session(exam_id, session_id)
    except repositories.ResourceNotFound as exc:
        raise _http_not_found(exc)
    return record


def _owned_exam(exam_id: str, session_id: str) -> dict[str, Any]:
    try:
        return repositories.get_exam_for_session(exam_id, session_id)
    except repositories.ResourceNotFound as exc:
        raise _http_not_found(exc)


def _persist_export(
    *, session_id: str, exam_id: str, extension: str,
    selected: list[dict[str, Any]], content: bytes,
) -> None:
    model_numbers = "-".join(
        str(int(exam.get("model_number") or 1)) for exam in selected
    )
    key = (
        f"sessions/{session_id}/exams/{exam_id}/exports/"
        f"{extension}-models-{model_numbers}.zip"
    )
    get_file_storage().put_bytes(key, content)


@router.post("/exams/{exam_id}/export/pdf")
def export_pdf(
    exam_id: str,
    body: DocumentExportRequest | None = None,
    session_id: str = Depends(require_session_id),
) -> StreamingResponse:
    record = _owned_exam(exam_id, session_id)
    selected = _selected_models(record, body)
    archive = _document_archive(record, selected, "pdf", render_exam_pdf, render_answers_pdf)
    _persist_export(
        session_id=session_id, exam_id=exam_id, extension="pdf",
        selected=selected, content=archive,
    )
    return _zip_response(archive)


@router.post("/exams/{exam_id}/export/docx")
def export_docx(
    exam_id: str,
    body: DocumentExportRequest | None = None,
    session_id: str = Depends(require_session_id),
) -> StreamingResponse:
    record = _owned_exam(exam_id, session_id)
    selected = _selected_models(record, body)
    archive = _document_archive(record, selected, "docx", render_exam_docx, render_answers_docx)
    _persist_export(
        session_id=session_id, exam_id=exam_id, extension="docx",
        selected=selected, content=archive,
    )
    return _zip_response(archive)
