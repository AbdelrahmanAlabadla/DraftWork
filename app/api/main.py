from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from app import config
from app import db
from app.api.export_routes import router as export_router
from app.api.job_routes import router as job_router
from app.api.routes import router
from app.api.session_middleware import SessionMiddleware
from app.logging_conf import configure_logging, get_logger, set_request_id

configure_logging(config.LOG_LEVEL)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The API stays lightweight; Celery workers own model warmup and GPU use.
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
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware)

app.include_router(router)
app.include_router(export_router)
app.include_router(job_router)

# Serve the frontend (ES modules require http://, not file://).
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "FrontEnd"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    set_request_id(request_id)
    start = time.perf_counter()
    t_start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        from app.logging_conf import get_logger

        get_logger("API").info(
            "Request completed | method=%s | path=%s | status=%d | time=%.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
    except Exception as exc:
        from app.logging_conf import get_logger

        get_logger("ERROR").error(
            "Request failed | method=%s | path=%s | exc=%s: %s | time=%.3fs",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc,
            time.perf_counter() - t_start,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
    finally:
        set_request_id("-")


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
