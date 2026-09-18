from __future__ import annotations

import json
import socket
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from billiard.exceptions import SoftTimeLimitExceeded
from celery import Task

from app import config
from app.api import evaluation_store, repositories
from app.file_storage import get_file_storage
from app.jobs.celery_app import celery_app
from app.logging_conf import get_logger, set_request_id
from app.offline.pipeline import run_pipeline
from app.online.exam_builder import generate_exams


logger = get_logger("WORKER")


class TrackedJobTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        if args:
            try:
                job = repositories.get_job(str(args[0]))
                if job and job["status"] == "running":
                    repositories.fail_job(str(args[0]), _error_code(exc), str(exc))
            except Exception:
                logger.exception("Could not persist task failure | task_id=%s", task_id)
        super().on_failure(exc, task_id, args, kwargs, einfo)


def _worker_id(task: Task) -> str:
    return str(getattr(task.request, "hostname", None) or socket.gethostname())


def _error_code(exc: BaseException) -> str:
    return type(exc).__name__.lower()


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
    set_request_id(job_id)
    job = repositories.claim_job(job_id, _worker_id(self))
    if job is None:
        existing = repositories.get_job(job_id)
        return {"job_id": job_id, "status": (existing or {}).get("status", "missing")}

    document = repositories.get_document(str(job["document_id"]))
    if document is None:
        repositories.fail_job(job_id, "document_missing", "Document record is missing")
        return {"job_id": job_id, "status": "failed"}

    storage = get_file_storage()
    try:
        repositories.update_job_progress(job_id, "parsing_and_indexing", 10)
        with _heartbeat(job_id):
            with storage.materialize(str(document["source_storage_key"])) as source_path:
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
        return {"job_id": job_id, "status": "completed"}
    except Exception as exc:
        logger.error("Ingestion job failed | job_id=%s | error=%s", job_id, exc, exc_info=True)
        if self.request.retries < self.max_retries:
            repositories.retry_job(job_id, _error_code(exc), str(exc))
            raise self.retry(exc=exc, countdown=15 * (2 ** self.request.retries))
        repositories.fail_job(job_id, _error_code(exc), str(exc))
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
    set_request_id(job_id)
    job = repositories.claim_job(job_id, _worker_id(self))
    if job is None:
        existing = repositories.get_job(job_id)
        return {"job_id": job_id, "status": (existing or {}).get("status", "missing")}

    request_data = dict(job.get("request_data") or {})
    document = repositories.get_document(str(job["document_id"]))
    if document is None:
        repositories.fail_job(job_id, "document_missing", "Document record is missing")
        return {"job_id": job_id, "status": "failed"}

    try:
        repositories.update_job_progress(job_id, "generating", 10)
        tasks = [tuple(item) for item in request_data["tasks"]]
        with _heartbeat(job_id):
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
        return {"job_id": job_id, "status": "completed", "exam_id": exam_id}
    except SoftTimeLimitExceeded as exc:
        repositories.fail_job(job_id, "generation_timeout", "Generation exceeded seven minutes")
        raise exc
    except Exception as exc:
        logger.error("Generation job failed | job_id=%s | error=%s", job_id, exc, exc_info=True)
        if self.request.retries < self.max_retries:
            repositories.retry_job(job_id, _error_code(exc), str(exc))
            raise self.retry(exc=exc, countdown=15)
        repositories.fail_job(job_id, _error_code(exc), str(exc))
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
