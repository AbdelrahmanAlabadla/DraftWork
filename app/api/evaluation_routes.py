"""Authenticated API for the Evaluation dashboard.

Read endpoints are available to any authenticated operator (a Clerk session, or
the optional ``X-Evaluation-Token`` machine credential when configured).
Benchmark creation and cancellation always require a signed-in user and a
browser session because they create real generation jobs.
"""
from __future__ import annotations

import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app import config, db
from app.api import benchmarks, evaluation_store, repositories
from app.api.session_middleware import require_session_id, require_user_id

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])

MAX_WINDOW_DAYS = 365
DEFAULT_WINDOW_DAYS = 30


def require_evaluation_access(
    request: Request,
    x_evaluation_token: Annotated[str | None, Header(alias="X-Evaluation-Token")] = None,
) -> None:
    """Allow a signed-in operator, or the configured machine token."""
    token = config.EVALUATION_DASHBOARD_TOKEN
    if token and x_evaluation_token and secrets.compare_digest(x_evaluation_token, token):
        return
    require_user_id(request)


def require_evaluation_operator(request: Request) -> str:
    """Benchmark execution requires a real signed-in user session."""
    user_id = require_user_id(request)
    require_session_id(request)
    return user_id


def _window_days(days: int | None) -> int | None:
    if days is None:
        return DEFAULT_WINDOW_DAYS
    if days <= 0:
        return None
    return min(int(days), MAX_WINDOW_DAYS)


def _handle_database_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503, detail="Evaluation telemetry is temporarily unavailable"
    )


@router.get("/overview")
def evaluation_overview(
    days: int | None = Query(default=None, ge=0, le=MAX_WINDOW_DAYS),
    include_benchmarks: bool = Query(default=False),
    _access: None = Depends(require_evaluation_access),
) -> dict[str, Any]:
    try:
        return evaluation_store.overview_summary(
            days=_window_days(days), include_benchmarks=include_benchmarks
        )
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)


@router.get("/exams")
def evaluation_exams(
    search: str = Query(default="", max_length=80),
    status: str = Query(default=""),
    benchmark: str = Query(default="exclude", pattern="^(exclude|include|only)$"),
    days: int | None = Query(default=None, ge=0, le=MAX_WINDOW_DAYS),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _access: None = Depends(require_evaluation_access),
) -> dict[str, Any]:
    try:
        return evaluation_store.list_exam_reports(
            search=search,
            status=status,
            benchmark_mode=benchmark,
            days=_window_days(days),
            limit=limit,
            offset=offset,
        )
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)


@router.get("/exams/{exam_id}")
def evaluation_exam_detail(
    exam_id: str,
    _access: None = Depends(require_evaluation_access),
) -> dict[str, Any]:
    try:
        report = evaluation_store.exam_report(exam_id)
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)
    if report is None:
        raise HTTPException(status_code=404, detail="Evaluation report not found")
    return report


@router.get("/pipeline")
def evaluation_pipeline(
    days: int | None = Query(default=None, ge=0, le=MAX_WINDOW_DAYS),
    include_benchmarks: bool = Query(default=False),
    _access: None = Depends(require_evaluation_access),
) -> dict[str, Any]:
    try:
        return evaluation_store.pipeline_analytics(
            days=_window_days(days), include_benchmarks=include_benchmarks
        )
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str | None = Field(default=None, max_length=120)
    document_id: str = Field(min_length=1, max_length=64)
    child_ids: list[str] = Field(min_length=1, max_length=config.MAX_CHILD_IDS_PER_GENERATION)
    tasks: dict[str, int] = Field(min_length=1)
    num_models: int = Field(default=1, ge=1, le=config.MAX_MODELS_PER_GENERATION)
    difficulty: str = Field(default="mix", max_length=20)
    repeats: int = Field(default=1, ge=1, le=config.BENCHMARK_MAX_REPEATS)
    stop_on_failure: bool = True


@router.get("/benchmarks/options")
def benchmark_options(
    session_id: str = Depends(require_session_id),
    user_id: str = Depends(require_user_id),
) -> dict[str, Any]:
    documents = repositories.list_documents_for_session(session_id, user_id)
    return {
        "documents": [
            {
                "document_id": str(document["document_id"]),
                "filename": document.get("filename"),
                "status": document.get("status"),
                "subject": (document.get("stats") or {}).get("subject"),
                "domain_type": (document.get("stats") or {}).get("domain_type"),
                "created_at": document["created_at"].isoformat()
                if document.get("created_at") else None,
            }
            for document in documents
            if document.get("status") == "ready"
        ],
        "question_types": list(benchmarks.QUESTION_TYPES),
        "difficulties": sorted(benchmarks.DIFFICULTIES),
        "limits": {
            "max_questions": config.MAX_QUESTIONS_PER_GENERATION,
            "max_models": config.MAX_MODELS_PER_GENERATION,
            "max_repeats": config.BENCHMARK_MAX_REPEATS,
            "max_child_ids": config.MAX_CHILD_IDS_PER_GENERATION,
        },
    }


@router.post("/benchmarks/estimate")
def benchmark_estimate(
    body: BenchmarkConfig,
    session_id: str = Depends(require_session_id),
    user_id: str = Depends(require_user_id),
) -> dict[str, Any]:
    try:
        return benchmarks.estimate(
            body.model_dump(), session_id=session_id, user_id=user_id
        )
    except benchmarks.BenchmarkError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)


@router.get("/benchmarks")
def benchmark_list(
    limit: int = Query(default=50, ge=1, le=200),
    session_id: str = Depends(require_session_id),
    user_id: str = Depends(require_user_id),
) -> dict[str, Any]:
    try:
        return benchmarks.list_benchmarks(
            session_id=session_id, user_id=user_id, limit=limit
        )
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)


@router.get("/benchmarks/compare")
def benchmark_compare(
    ids: str = Query(min_length=1, max_length=500),
    session_id: str = Depends(require_session_id),
    user_id: str = Depends(require_user_id),
) -> dict[str, Any]:
    benchmark_ids = [value.strip() for value in ids.split(",") if value.strip()]
    if len(benchmark_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least two benchmarks")
    if len(benchmark_ids) > 5:
        raise HTTPException(status_code=400, detail="Compare at most five benchmarks")
    try:
        return benchmarks.compare_benchmarks(
            benchmark_ids, session_id=session_id, user_id=user_id
        )
    except benchmarks.BenchmarkError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)


@router.post("/benchmarks", status_code=201)
def benchmark_create(
    body: BenchmarkConfig,
    idempotency_key: Annotated[
        str | None, Header(alias="Idempotency-Key", max_length=100)
    ] = None,
    session_id: str = Depends(require_session_id),
    user_id: str = Depends(require_user_id),
) -> dict[str, Any]:
    try:
        result = benchmarks.create_benchmark(
            body.model_dump(),
            session_id=session_id,
            user_id=user_id,
            idempotency_key=idempotency_key,
        )
    except benchmarks.BenchmarkError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except repositories.ResourceNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'"))
    except repositories.InvalidState as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except repositories.JobLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)
    return result


@router.get("/benchmarks/{benchmark_id}")
def benchmark_detail(
    benchmark_id: str,
    session_id: str = Depends(require_session_id),
    user_id: str = Depends(require_user_id),
) -> dict[str, Any]:
    try:
        result = benchmarks.get_benchmark(
            benchmark_id, session_id=session_id, user_id=user_id
        )
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return result


@router.post("/benchmarks/{benchmark_id}/cancel")
def benchmark_cancel(
    benchmark_id: str,
    session_id: str = Depends(require_session_id),
    user_id: str = Depends(require_user_id),
) -> dict[str, Any]:
    try:
        result = benchmarks.cancel_benchmark(
            benchmark_id, session_id=session_id, user_id=user_id
        )
    except db.DatabaseUnavailable as exc:
        raise _handle_database_unavailable(exc)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return result
