from __future__ import annotations

import time
import uuid
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse

from app import config
from app import db
from app.api.export_routes import router as export_router
from app.api.job_routes import router as job_router
from app.api.routes import router
from app.api.session_middleware import SessionMiddleware
from app.logging_conf import configure_logging, get_logger, set_request_id
from app.errors import AppError, ErrorCode, public_error_payload
from app.metrics import increment, observe

configure_logging(config.LOG_LEVEL)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The API stays lightweight; Celery workers own model warmup and GPU use.
    config.validate_runtime_config()
    db.open_pool(wait=True)
    try:
        yield
    finally:
        db.close_pool()

app = FastAPI(
    title="ExamGen AI",
    description="Exam Generator V1 — API-first backend. Upload a PDF, then generate exams "
    "(MCQ / True-False / Short Answer) grounded in the uploaded document.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ALLOW_ORIGINS),
    allow_credentials=True,
    allow_methods=list(config.CORS_ALLOW_METHODS),
    allow_headers=list(config.CORS_ALLOW_HEADERS),
)
app.add_middleware(SessionMiddleware)
if config.IS_PRODUCTION:
    app.add_middleware(HTTPSRedirectMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.ALLOWED_HOSTS))

app.include_router(router)
app.include_router(export_router)
app.include_router(job_router)

# Serve the frontend (ES modules require http://, not file://).
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "FrontEnd"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    supplied_request_id = request.headers.get("X-Request-ID", "")
    request_id = (
        supplied_request_id
        if _REQUEST_ID_RE.fullmatch(supplied_request_id)
        else str(uuid.uuid4())
    )
    set_request_id(request_id)
    request.state.request_id = request_id
    start = time.perf_counter()
    t_start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        if config.IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        get_logger("API").info(
            "Request completed | method=%s | path=%s | status=%d | time=%.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        route_path = getattr(request.scope.get("route"), "path", request.url.path)
        increment(
            "genexam_http_requests",
            method=request.method,
            path=route_path,
            status=response.status_code,
        )
        observe(
            "genexam_http_request_duration_seconds",
            elapsed,
            method=request.method,
            path=route_path,
            status=response.status_code,
        )
        return response
    except Exception as exc:
        get_logger("ERROR").error(
            "Request failed | method=%s | path=%s | exc=%s: %s | time=%.3fs",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc,
            time.perf_counter() - t_start,
            exc_info=True,
            extra={
                "event": "http_request_failed",
                "duration_ms": int((time.perf_counter() - t_start) * 1000),
                "status_code": 500,
            },
        )
        increment(
            "genexam_http_requests",
            method=request.method,
            path=request.url.path,
            status=500,
        )
        headers = {"X-Request-ID": request_id}
        if config.IS_PRODUCTION:
            headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return JSONResponse(
            status_code=500,
            content=public_error_payload(
                ErrorCode.INTERNAL_ERROR,
                "Internal server error",
                request_id,
                retryable=True,
            ),
            headers=headers,
        )
    finally:
        set_request_id("-")


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=exc.status_code,
        content=public_error_payload(
            exc.code, exc.message, request_id, retryable=exc.retryable
        ),
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    message = str(exc.detail)
    path = request.url.path
    lowered = message.lower()
    if exc.status_code == 404 and "/jobs/" in path:
        code = ErrorCode.JOB_NOT_FOUND
    elif exc.status_code == 404 and "/exams/" in path:
        code = ErrorCode.EXAM_NOT_FOUND
    elif exc.status_code == 404:
        code = ErrorCode.DOCUMENT_NOT_FOUND
    elif exc.status_code == 409 and "idempotency" in lowered:
        code = ErrorCode.IDEMPOTENCY_CONFLICT
    elif exc.status_code == 409 and "active generation" in lowered:
        code = ErrorCode.JOB_LIMIT_REACHED
    elif exc.status_code == 409:
        code = ErrorCode.DOCUMENT_NOT_READY
    elif exc.status_code == 413:
        code = ErrorCode.FILE_TOO_LARGE
    elif exc.status_code == 429:
        code = ErrorCode.RATE_LIMITED
    elif exc.status_code == 503 and "database" in lowered:
        code = ErrorCode.DATABASE_UNAVAILABLE
    elif exc.status_code == 503 and "queue" in lowered:
        code = ErrorCode.QUEUE_UNAVAILABLE
    elif exc.status_code == 503:
        code = ErrorCode.PROVIDER_UNAVAILABLE
    elif exc.status_code >= 500:
        code = ErrorCode.INTERNAL_ERROR
    elif "pdf" in lowered or "file" in lowered:
        code = ErrorCode.INVALID_FILE
    elif "section" in lowered or "selection" in lowered:
        code = ErrorCode.INVALID_SELECTION
    else:
        code = ErrorCode.INVALID_REQUEST
    return JSONResponse(
        status_code=exc.status_code,
        content=public_error_payload(code, message, request_id, retryable=exc.status_code >= 500),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=422,
        content={
            **public_error_payload(
                ErrorCode.INVALID_REQUEST,
                "Request validation failed",
                request_id,
            ),
            "validation": jsonable_encoder(exc.errors()),
        },
    )


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "ExamGen AI",
        "docs": "/docs",
        "endpoints": [
            "POST /api/v1/documents",
            "POST /api/v1/documents/{document_id}/exam-jobs",
            "GET /api/v1/jobs/{job_id}",
            "GET /api/v1/exams/{exam_id}",
            "POST /api/v1/exams/{exam_id}/export/{pdf|docx}",
            "GET /health",
        ],
    }
