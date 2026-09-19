from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from billiard.exceptions import SoftTimeLimitExceeded
from celery import Task

from app import config
from app.api import evaluation_store, repositories
from app.deadlines import DeadlineExceeded, deadline
from app.file_storage import get_file_storage
from app.file_validation import InvalidPdfError, validate_pdf
from app.jobs.celery_app import celery_app
from app.cleanup import run_cleanup
from app.logging_conf import get_logger, set_job_context, set_request_id
from app.offline.pipeline import run_pipeline
from app.online.exam_builder import generate_exams
from app.metrics import increment, observe


logger = get_logger("WORKER")


class TrackedJobTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        if args:
            try:
                job = repositories.get_job(str(args[0]))
                if job and job["status"] == "running":
                    repositories.fail_job(
                        str(args[0]),
                        _error_code(exc),
                        _public_error_message(exc, "Background job"),
                    )
            except Exception:
                logger.exception("Could not persist task failure | task_id=%s", task_id)
        super().on_failure(exc, task_id, args, kwargs, einfo)


def _worker_id(task: Task) -> str:
    return str(getattr(task.request, "hostname", None) or socket.gethostname())


def _error_code(exc: BaseException) -> str:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, SoftTimeLimitExceeded)):
            return "TIMEOUT"
        name = type(current).__name__.lower()
        if name.startswith("deepseek") or "connection" in name or "network" in name:
            return "PROVIDER_UNAVAILABLE"
        current = current.__cause__ or current.__context__
    return "GENERATION_FAILED"


def _public_error_message(exc: BaseException, operation: str) -> str:
    code = _error_code(exc)
    if code == "TIMEOUT":
        return f"{operation} timed out"
    if code == "PROVIDER_UNAVAILABLE":
        return "The configured AI provider is temporarily unavailable"
    return f"{operation} failed"


@contextmanager
def _heartbeat(job_id: str):
    stop = threading.Event()

    def pulse() -> None:
        while not stop.wait(config.JOB_HEARTBEAT_SECONDS):
            try:
                repositories.heartbeat_job(job_id)
            except Exception:
                logger.exception("Job heartbeat failed | job_id=%s", job_id)

    thread = threading.Thread(target=pulse, daemon=True, name=f"heartbeat-{job_id}")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


@celery_app.task(
    bind=True,
    base=TrackedJobTask,
    name="genexam.ingest_document",
    max_retries=2,
    default_retry_delay=15,
)
def ingest_document(self: Task, job_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    set_request_id(job_id)
    job = repositories.claim_job(job_id, _worker_id(self))
    if job is None:
        existing = repositories.get_job(job_id)
        return {"job_id": job_id, "status": (existing or {}).get("status", "missing")}

    document = repositories.get_document(str(job["document_id"]))
    if document is None:
        repositories.fail_job(job_id, "DOCUMENT_NOT_FOUND", "Document record is missing")
        return {"job_id": job_id, "status": "failed"}
    set_job_context(job_id, str(document["id"]))

    storage = get_file_storage()
    try:
        repositories.update_job_progress(job_id, "parsing_and_indexing", 10)
        with _heartbeat(job_id):
            with storage.materialize(str(document["source_storage_key"])) as source_path:
                validate_pdf(source_path)
                summary = run_pipeline(
                    source_path,
                    str(document["id"]),
                    session_id=str(document["session_id"]),
                    job_id=job_id,
                )

        structure_path = Path(str(summary["structure_file"]))
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        structure_key = (
            f"sessions/{document['session_id']}/documents/{document['id']}/"
            f"jobs/{job_id}/structure.json"
        )
        storage.put_json(structure_key, structure)
        repositories.complete_ingestion_job(
            job_id, structure_storage_key=structure_key, stats=summary
        )
        increment("genexam_jobs", type="document_ingestion", status="completed")
        observe("genexam_job_duration_seconds", time.perf_counter() - started, type="document_ingestion")
        logger.info(
            "Ingestion job completed | job_id=%s",
            job_id,
            extra={
                "event": "job_completed",
                "stage": "completed",
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        return {"job_id": job_id, "status": "completed"}
    except InvalidPdfError as exc:
        repositories.fail_job(job_id, "INVALID_FILE", str(exc))
        increment("genexam_jobs", type="document_ingestion", status="failed")
        return {"job_id": job_id, "status": "failed"}
    except SoftTimeLimitExceeded as exc:
        repositories.fail_job(job_id, "TIMEOUT", "Document processing timed out")
        raise exc
    except Exception as exc:
        logger.error(
            "Ingestion job failed | job_id=%s | error=%s", job_id, exc,
            exc_info=True,
            extra={"event": "job_failed", "stage": "failed"},
        )
        if self.request.retries < self.max_retries:
            repositories.retry_job(job_id, _error_code(exc), _public_error_message(exc, "Document processing"))
            increment("genexam_jobs", type="document_ingestion", status="retrying")
            raise self.retry(exc=exc, countdown=15 * (2 ** self.request.retries))
        repositories.fail_job(job_id, _error_code(exc), _public_error_message(exc, "Document processing"))
        increment("genexam_jobs", type="document_ingestion", status="failed")
        return {"job_id": job_id, "status": "failed"}


@celery_app.task(
    bind=True,
    base=TrackedJobTask,
    name="genexam.generate_exam",
    max_retries=1,
    default_retry_delay=15,
    soft_time_limit=max(1, config.GENERATION_TIMEOUT_SECONDS - 10),
    time_limit=config.GENERATION_TIMEOUT_SECONDS,
)
def generate_exam(self: Task, job_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    set_request_id(job_id)
    job = repositories.claim_job(job_id, _worker_id(self))
    if job is None:
        existing = repositories.get_job(job_id)
        return {"job_id": job_id, "status": (existing or {}).get("status", "missing")}

    request_data = dict(job.get("request_data") or {})
    document = repositories.get_document(str(job["document_id"]))
    if document is None:
        repositories.fail_job(job_id, "DOCUMENT_NOT_FOUND", "Document record is missing")
        return {"job_id": job_id, "status": "failed"}
    set_job_context(job_id, str(document["id"]))

    try:
        repositories.update_job_progress(job_id, "generating", 10)
        tasks = [tuple(item) for item in request_data["tasks"]]
        with _heartbeat(job_id):
            with deadline(config.GENERATION_TIMEOUT_SECONDS, "Exam generation"):
                result = generate_exams(
                    str(document["id"]),
                    tasks,
                    int(request_data["num_models"]),
                    list(request_data["child_ids"]),
                    str(request_data["difficulty"]),
                    session_id=str(job["session_id"]),
                    index_job_id=str(document["active_index_job_id"]),
                )
        exams = result.get("exams") or []
        if not any(exam.get("questions") for exam in exams):
            raise RuntimeError("; ".join(result.get("warnings") or []) or "No questions generated")

        exam_id = f"exam_{uuid.uuid4().hex[:12]}"
        metadata = dict(request_data.get("metadata") or {})
        metadata["document_language"] = result.get("document_language") or "en"
        status = str(result.get("status") or "partial")
        repositories.complete_generation_job(
            job_id,
            exam_id=exam_id,
            document_id=str(document["id"]),
            session_id=str(job["session_id"]),
            status="complete" if status == "complete" else "partial",
            exams=exams,
            warnings=list(result.get("warnings") or []),
            metadata=metadata,
            evaluation=dict(result.get("eval") or {}),
        )
        try:
            evaluation_store.save_evaluation(
                exam_id=exam_id,
                eval_stats=dict(result.get("eval") or {}),
                models_count=int(request_data["num_models"]),
                questions_requested_per_model=sum(int(count) for _, count in tasks),
            )
        except Exception as exc:
            logger.warning("Evaluation telemetry was not saved | exam_id=%s | error=%s", exam_id, exc)
        increment("genexam_jobs", type="exam_generation", status="completed")
        observe("genexam_job_duration_seconds", time.perf_counter() - started, type="exam_generation")
        logger.info(
            "Generation job completed | job_id=%s | exam_id=%s",
            job_id,
            exam_id,
            extra={
                "event": "job_completed",
                "stage": "completed",
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        return {"job_id": job_id, "status": "completed", "exam_id": exam_id}
    except (DeadlineExceeded, SoftTimeLimitExceeded) as exc:
        repositories.fail_job(job_id, "TIMEOUT", "Generation exceeded its configured time limit")
        increment("genexam_jobs", type="exam_generation", status="failed")
        raise exc
    except Exception as exc:
        logger.error(
            "Generation job failed | job_id=%s | error=%s", job_id, exc,
            exc_info=True,
            extra={"event": "job_failed", "stage": "failed"},
        )
        if self.request.retries < self.max_retries:
            repositories.retry_job(job_id, _error_code(exc), _public_error_message(exc, "Exam generation"))
            increment("genexam_jobs", type="exam_generation", status="retrying")
            raise self.retry(exc=exc, countdown=15)
        repositories.fail_job(job_id, _error_code(exc), _public_error_message(exc, "Exam generation"))
        increment("genexam_jobs", type="exam_generation", status="failed")
        return {"job_id": job_id, "status": "failed"}


@celery_app.task(name="genexam.recover_stale_jobs")
def recover_stale_jobs() -> dict[str, int]:
    stale = repositories.recover_stale_jobs(config.JOB_STALE_AFTER_SECONDS)
    for job in stale:
        if job["type"] == "document_ingestion":
            ingest_document.delay(str(job["id"]))
        elif job["type"] == "exam_generation":
            generate_exam.delay(str(job["id"]))
    return {"requeued": len(stale)}


@celery_app.task(name="genexam.dispatch_undispatched_jobs")
def dispatch_undispatched_jobs() -> dict[str, int]:
    queued = repositories.undispatched_jobs()
    dispatched = 0
    for job in queued:
        if job["type"] == "document_ingestion":
            ingest_document.delay(str(job["id"]))
        elif job["type"] == "exam_generation":
            generate_exam.delay(str(job["id"]))
        else:
            continue
        repositories.mark_job_dispatched(str(job["id"]))
        dispatched += 1
    return {"dispatched": dispatched}


@celery_app.task(name="genexam.cleanup_expired_data")
def cleanup_expired_data() -> dict[str, Any]:
    return run_cleanup()
