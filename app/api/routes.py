from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.api import storage as registry
from app.config import UPLOAD_DIR
from app.logging_conf import get_logger, set_request_id
from app.offline.pipeline import PipelineError, run_pipeline
from app.offline.structure_store import load_structure
from app.online.exam_builder import generate_exams

logger = get_logger("API")

router = APIRouter()

# Question types supported.
_SUPPORTED_COUNTS = {
    "mcq": "mcq",
    "tf": "true_false",
    "fitb": "fill_in_the_blank",
    "why": "short_answer",
    "essay": "essay",
}
_VALID_QTYPES = frozenset(_SUPPORTED_COUNTS.values())

_VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard", "mix"})
_NUM_MODELS_MIN = 1
_NUM_MODELS_MAX = 4


class GenerateRequest(BaseModel):
    # --- Frontend / HTML payload (all optional) -------------------------
    document_id: Optional[str] = None
    num_models: Optional[int] = None
    difficulty: Optional[str] = None
    mcq_count: Optional[int] = Field(default=None, ge=0)
    tf_count: Optional[int] = Field(default=None, ge=0)
    why_count: Optional[int] = Field(default=None, ge=0)
    fitb_count: Optional[int] = Field(default=None, ge=0)
    essay_count: Optional[int] = Field(default=None, ge=0)
    # --- V1 single-type payload -------------------------------------------
    question_type: Optional[str] = None
    number_of_questions: Optional[int] = Field(default=None, ge=1, le=100)
    # --- Selected subsections (exam content scope) -------------------------
    child_ids: Optional[list[str]] = None


@router.get("/health")
def health() -> dict[str, str]:
    statuses: dict[str, str] = {}
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url="http://localhost:6333").get_collections()
        statuses["qdrant"] = "ok"
    except Exception:
        statuses["qdrant"] = "error"

    try:
        import requests

        requests.get("http://127.0.0.1:1234/api/v1/models", timeout=3)
        statuses["lm_studio"] = "ok"
    except Exception:
        statuses["lm_studio"] = "error"

    statuses["status"] = "ok" if all(v == "ok" for v in statuses.values()) else "degraded"
    return statuses


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

    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{document_id}.pdf"

    content = await file.read()
    dest.write_bytes(content)

    try:
        summary = run_pipeline(dest, document_id)
    except PipelineError as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")
    except Exception as exc:
        logger.error(
            "Upload failed | document_id=%s | exc=%s: %s",
            document_id,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")

    elapsed = time.perf_counter() - t0
    metadata = {
        "document_id": document_id,
        "filename": filename,
        "size_bytes": file.size or len(content),
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


@router.post("/generate")
def generate(body: GenerateRequest) -> dict[str, Any]:
    document_id = body.document_id or registry.get_current_document()
    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="No document_id provided and no document uploaded yet. Upload a PDF first.",
        )
    if not registry.get_document(document_id):
        raise HTTPException(status_code=404, detail=f"Unknown document_id: {document_id}")

    if not body.child_ids:
        raise HTTPException(
            status_code=400,
            detail="Please choose at least one section topic to generate the exam.",
        )

    set_request_id(document_id)

    tasks: list[tuple[str, int]] = []
    if body.question_type is not None:
        qtype = body.question_type
        count = body.number_of_questions
        if qtype not in _VALID_QTYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported question_type '{qtype}'. Supported: "
                    f"{', '.join(sorted(_VALID_QTYPES))}"
                ),
            )
        if not count:
            raise HTTPException(status_code=400, detail="number_of_questions is required with question_type")
        tasks.append((qtype, count))
    else:
        for key, qtype in _SUPPORTED_COUNTS.items():
            count = getattr(body, f"{key}_count") or 0
            if count > 0:
                tasks.append((qtype, count))

    if not tasks:
        raise HTTPException(
            status_code=400,
            detail="No supported question types requested. Supported: MCQ, True/False, Fill in the Blank, Short Answer, and Essay.",
        )

    num_models = body.num_models if body.num_models is not None else 1
    if not isinstance(num_models, int) or not (
        _NUM_MODELS_MIN <= num_models <= _NUM_MODELS_MAX
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"num_models must be an integer between {_NUM_MODELS_MIN} and "
                f"{_NUM_MODELS_MAX}."
            ),
        )

    difficulty = body.difficulty or "easy"
    difficulty = difficulty.lower() if isinstance(difficulty, str) else difficulty
    if difficulty not in _VALID_DIFFICULTIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported difficulty '{difficulty}'. Supported: "
                f"{sorted(_VALID_DIFFICULTIES)}"
            ),
        )

    t_total = time.perf_counter()
    result = generate_exams(
        document_id, tasks, num_models, body.child_ids, difficulty
    )
    exams = result["exams"]
    warnings = result["warnings"]

    total_elapsed = time.perf_counter() - t_total

    if not any(exam["questions"] for exam in exams):
        detail = "; ".join(warnings) or "No questions could be generated."
        raise HTTPException(status_code=500, detail=detail)

    logger.info(
        "Exam generation completed | document_id=%s | models=%d | difficulty=%s | types=%s | "
        "total_questions=%d | total_time=%.2fs | success=True",
        document_id,
        num_models,
        difficulty,
        list(exams[0]["questions"].keys()),
        sum(len(q) for e in exams for q in e["questions"].values()),
        total_elapsed,
    )

    return {
        "document_id": document_id,
        "num_models": num_models,
        "difficulty": difficulty,
        "exams": exams,
        "warnings": warnings,
    }
