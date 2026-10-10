from __future__ import annotations

import time
import uuid
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api import storage as registry
from app.api import evaluation_store, repositories
from app import db
from app.config import (
    CELERY_BROKER_URL,
    MAX_UPLOAD_BYTES,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_VERIFY_TLS,
    QDRANT_TIMEOUT_SECONDS,
    UPLOAD_DIR,
)
from app.logging_conf import get_logger, set_request_id
from app.offline.pipeline import PipelineError, run_pipeline
from app.offline.structure_store import load_structure
from app.file_storage import get_file_storage
from app.metrics import (
    collect_runtime_metrics,
    gauge,
    increment,
    render_prometheus,
    replace_gauges,
)
from app import config

logger = get_logger("API")

router = APIRouter()

@router.get("/api/v1/eval-summary")
@router.get("/api/v1/api/eval-summary", include_in_schema=False)
@router.get("/api/eval-summary", include_in_schema=False)
def eval_summary() -> dict[str, Any]:
    """Return a read-only aggregate of persisted evaluation telemetry."""
    try:
        return evaluation_store.load_eval_summary()
    except db.DatabaseUnavailable as exc:
        logger.warning("Evaluation dashboard database unavailable | error=%s", exc)
        raise HTTPException(status_code=503, detail="Evaluation telemetry is temporarily unavailable")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    # --- Frontend / HTML payload (all optional) -------------------------
    document_id: Optional[str] = None
    num_models: Optional[int] = None
    difficulty: Optional[str] = None
    mcq_count: Optional[int] = Field(default=None, ge=0)
    tf_count: Optional[int] = Field(default=None, ge=0)
    fitb_count: Optional[int] = Field(default=None, ge=0)
    definition_count: Optional[int] = Field(default=None, ge=0)
    why_count: Optional[int] = Field(default=None, ge=0)
    equation_count: Optional[int] = Field(default=None, ge=0)
    word_problem_count: Optional[int] = Field(default=None, ge=0)
    essay_count: Optional[int] = Field(default=None, ge=0)
    # --- Optional print/export metadata ----------------------------------
    exam_title: Optional[str] = Field(default=None, max_length=120)
    class_name: Optional[str] = Field(default=None, max_length=80)
    duration: Optional[str] = Field(default=None, max_length=40)
    exam_date: Optional[str] = Field(default=None, max_length=40)
    teacher_name: Optional[str] = Field(default=None, max_length=100)
    footer_message: Optional[str] = Field(default=None, max_length=120)
    left_logo_data: Optional[str] = Field(default=None, max_length=3_500_000)
    right_logo_data: Optional[str] = Field(default=None, max_length=3_500_000)
    # --- V1 single-type payload -------------------------------------------
    question_type: Optional[str] = None
    number_of_questions: Optional[int] = Field(default=None, ge=1, le=100)
    # --- Selected subsections (exam content scope) -------------------------
    child_ids: Optional[
        list[Annotated[str, Field(min_length=1, max_length=100)]]
    ] = Field(default=None, max_length=config.MAX_CHILD_IDS_PER_GENERATION)


@router.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics", include_in_schema=False)
def metrics(authorization: str | None = Header(default=None)) -> PlainTextResponse:
    if config.METRICS_TOKEN:
        expected = f"Bearer {config.METRICS_TOKEN}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=404, detail="Not found")
    collect_runtime_metrics()
    try:
        snapshot = repositories.monitoring_snapshot()
        replace_gauges(
            "genexam_jobs_current",
            (
                (
                    float(row["count"]),
                    {"type": row["type"], "status": row["status"]},
                )
                for row in snapshot["jobs"]
            ),
        )
        for metric, field in (
            ("genexam_job_duration_seconds_sum", "duration_sum"),
            ("genexam_job_duration_seconds_count", "duration_count"),
        ):
            replace_gauges(
                metric,
                (
                    (
                        float(row[field]),
                        {"type": row["type"], "status": row["status"]},
                    )
                    for row in snapshot["jobs"]
                ),
            )
        replace_gauges(
            "genexam_cleanup_runs",
            ((float(row["count"]), {"status": row["status"]}) for row in snapshot["cleanup_runs"]),
        )
        for metric, field in (
            ("genexam_cleanup_duration_seconds_sum", "duration_sum"),
            ("genexam_cleanup_duration_seconds_count", "duration_count"),
        ):
            replace_gauges(
                metric,
                ((float(row[field]), {"status": row["status"]}) for row in snapshot["cleanup_runs"]),
            )
        llm_fields = (
            ("genexam_llm_calls", "calls"),
            ("genexam_llm_input_tokens", "input_tokens"),
            ("genexam_llm_output_tokens", "output_tokens"),
            ("genexam_llm_cached_tokens", "cached_tokens"),
            ("genexam_llm_reasoning_tokens", "reasoning_tokens"),
            ("genexam_llm_estimated_cost_usd", "estimated_cost_usd"),
            ("genexam_llm_duration_seconds_sum", "duration_sum"),
            ("genexam_llm_duration_seconds_count", "duration_count"),
        )
        for metric, field in llm_fields:
            replace_gauges(
                metric,
                (
                    (
                        float(row[field]),
                        {
                            "provider": row["provider"],
                            "model": row["model"],
                            "operation": row["operation"],
                        },
                    )
                    for row in snapshot["llm_usage"]
                ),
            )
    except Exception:
        increment("genexam_monitoring_collection_failures", source="postgres")
    try:
        from redis import Redis

        queue_depth = Redis.from_url(CELERY_BROKER_URL, socket_timeout=2).llen("celery")
        gauge("genexam_queue_depth", int(queue_depth), queue="celery")
    except Exception:
        increment("genexam_monitoring_collection_failures", source="redis")
    return PlainTextResponse(
        render_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@router.get("/health/ready")
@router.get("/health")
def health() -> JSONResponse:
    statuses: dict[str, str] = {}
    try:
        with db.connection() as conn:
            conn.execute("SELECT 1")
        statuses["postgres"] = "ok"
    except Exception:
        statuses["postgres"] = "error"
        increment("genexam_health_dependency_failures", dependency="postgres")

    try:
        from redis import Redis

        Redis.from_url(CELERY_BROKER_URL, socket_timeout=3).ping()
        statuses["redis"] = "ok"
    except Exception:
        statuses["redis"] = "error"
        increment("genexam_health_dependency_failures", dependency="redis")

    try:
        from qdrant_client import QdrantClient

        QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT_SECONDS,
            verify=QDRANT_VERIFY_TLS,
        ).get_collections()
        statuses["qdrant"] = "ok"
    except Exception:
        statuses["qdrant"] = "error"
        increment("genexam_health_dependency_failures", dependency="qdrant")

    try:
        storage = get_file_storage()
        key = ".health/ready"
        storage.put_bytes(key, b"ok")
        storage.delete(key)
        statuses["storage"] = "ok"
    except Exception:
        statuses["storage"] = "error"
        increment("genexam_health_dependency_failures", dependency="storage")

    statuses["status"] = "ok" if all(v == "ok" for v in statuses.values()) else "degraded"
    return JSONResponse(status_code=200 if statuses["status"] == "ok" else 503, content=statuses)


@router.get("/documents")
def documents() -> dict[str, Any]:
    return {"documents": registry.list_documents()}


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    document_id = str(uuid.uuid4())
    set_request_id(document_id)

    filename = Path(file.filename or "upload").name
    logger.info(
        "Upload started | document_id=%s | file=%s | size=%d",
        document_id,
        filename,
        file.size or 0,
    )
    t0 = time.perf_counter()

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported in V1")
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 100 MB limit")

    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{document_id}.pdf"

    size_bytes = 0
    with dest.open("wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > MAX_UPLOAD_BYTES:
                output.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="PDF exceeds the 100 MB limit")
            output.write(chunk)

    try:
        summary = run_pipeline(dest, document_id)
    except PipelineError as exc:
        logger.warning(
            "Upload indexing failed | document_id=%s | error=%s",
            document_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail="Document indexing failed")
    except Exception as exc:
        logger.error(
            "Upload failed | document_id=%s | exc=%s: %s",
            document_id,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Document indexing failed")

    elapsed = time.perf_counter() - t0
    metadata = {
        "document_id": document_id,
        "filename": filename,
        "size_bytes": file.size or size_bytes,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "stats": summary,
    }
    registry.register_document(document_id, metadata)

    logger.info(
        "Upload completed | document_id=%s | file=%s | time=%.2fs",
        document_id,
        filename,
        elapsed,
    )
    return {
        "document_id": document_id,
        "message": f"File '{filename}' indexed successfully.",
        "stats": summary,
        "structure": load_structure(document_id),
    }


@router.post("/generate", include_in_schema=False)
def generate(_body: GenerateRequest) -> dict[str, Any]:
    """Retired synchronous endpoint; online generation runs through exam jobs."""
    raise HTTPException(
        status_code=410,
        detail="Synchronous generation was retired. Use POST /documents/{document_id}/exam-jobs.",
    )
