from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from billiard.exceptions import SoftTimeLimitExceeded
from celery import Task

from app import config
from app.cleanup import _delete_pipeline_artifacts
from app.api import evaluation_store, repositories
from app.deadlines import DeadlineExceeded, deadline
from app.file_storage import get_file_storage
from app.file_validation import InvalidPdfError, validate_pdf
from app.jobs.celery_app import celery_app
from app.cleanup import run_cleanup
from app.logging_conf import get_logger, set_job_context, set_request_id
from app.offline.pipeline import PipelineCancelled, run_pipeline
from app.offline.parser import ParserError
from app.offline.vector_store import VectorStore
from app.online.exam_builder import generate_exams
from app.metrics import increment, observe


logger = get_logger("WORKER")


@celery_app.task(name="genexam.warmup_embeddings", ignore_result=True)
def warmup_embeddings() -> dict[str, str]:
    """Load BGE-M3 inside a ready Celery child, never in the prefork parent."""
    from app.offline.embeddings import device_name, warmup

    warmup()
    device = device_name()
    logger.info("Embedding worker warmup task completed | device=%s", device)
    return {"status": "ready", "device": device}


def _section_question_count(section: Any) -> int:
    if isinstance(section, dict):
        return len(section.get("items") or [])
    return len(section or []) if isinstance(section, list) else 0


def _question_shortage_warnings(
    result: dict[str, Any], tasks: list[tuple[str, int]], num_models: int
) -> list[dict[str, Any]]:
    lifecycle = result.get("slot_lifecycle") or {}
    generated: dict[str, int] = {}
    for record in lifecycle.values():
        if record.get("terminal_status") == "finalized":
            qtype = str(record.get("question_type") or "")
            generated[qtype] = generated.get(qtype, 0) + 1
    if not lifecycle:
        for exam in result.get("exams") or []:
            for qtype, section in (exam.get("questions") or {}).items():
                generated[qtype] = generated.get(qtype, 0) + _section_question_count(section)
    warnings: list[dict[str, Any]] = []
    for qtype, count_per_model in tasks:
        requested = int(count_per_model) * int(num_models)
        produced = int(generated.get(qtype, 0))
        if produced >= requested:
            continue
        missing = requested - produced
        warnings.append({
            "type": "question_shortage",
            "question_type": qtype,
            "requested_count": requested,
            "generated_count": produced,
            "missing_count": missing,
            "requested": requested,
            "generated": produced,
            "missing": missing,
            "reason": "missing_questions",
        })
    return warnings


def _difficulty_fallback_warnings(result: dict[str, Any]) -> list[dict[str, Any]]:
    identities = result.get("semantic_identities") or []
    lifecycle = result.get("slot_lifecycle") or {}
    finalized_slots = {
        str(slot_id) for slot_id, record in lifecycle.items()
        if record.get("terminal_status") == "finalized"
    }
    finalized = [
        row for row in identities
        if not finalized_slots or str(row.get("slot_id") or "") in finalized_slots
    ]
    fallbacks = [row for row in finalized if row.get("difficulty_fallback_reason")]
    if not fallbacks:
        return []
    requested_levels = {
        str(row.get("requested_difficulty") or "") for row in fallbacks
        if row.get("requested_difficulty") in {"easy", "medium", "hard"}
    }
    if len(requested_levels) != 1:
        return []
    requested_level = next(iter(requested_levels))
    counts: dict[str, int] = {}
    for row in finalized:
        level = str(row.get("actual_difficulty") or row.get("difficulty") or "unknown")
        counts[level] = counts.get(level, 0) + 1
    produced = sum(counts.values())
    matched = counts.get(requested_level, 0)
    substitutions = {level: count for level, count in counts.items()
                     if level != requested_level and count}
    return [{
        "type": "difficulty_fallback",
        "requested_difficulty": requested_level,
        "requested_count": produced,
        f"{requested_level}_count": matched,
        "actual_difficulty_counts": counts,
        "fallback_counts": substitutions,
        "message": (
            f"Your exam was generated with {produced} of {produced} requested questions. "
            f"It contains {matched} {requested_level.title()} and "
            + ", ".join(f"{count} {level.title()}" for level, count in sorted(substitutions.items()))
            + f" questions because the selected material did not support enough distinct "
              f"{requested_level.title()} questions."
        ),
    }]


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
        if isinstance(current, ParserError) and str(current).startswith(
            "Invalid LlamaParse JSON:"
        ):
            return "PARSER_RESPONSE_INVALID"
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
    if code == "PARSER_RESPONSE_INVALID":
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            message = str(current)
            prefix = "Invalid LlamaParse JSON:"
            if message.startswith(prefix):
                reason = message[len(prefix):].strip()
                return f"Document parser response was invalid: {reason}"[:500]
            current = current.__cause__ or current.__context__
        return "Document parser response was invalid"
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
        repositories.update_job_progress(job_id, "validating", 5)
        with _heartbeat(job_id):
            with storage.materialize(str(document["source_storage_key"])) as source_path:
                validate_pdf(source_path)
                if repositories.job_is_cancelled(job_id):
                    raise PipelineCancelled("Document processing was cancelled")
                summary = run_pipeline(
                    source_path,
                    str(document["id"]),
                    session_id=str(document["session_id"]),
                    job_id=job_id,
                    file_sha256_value=str(document.get("sha256") or "") or None,
                    on_stage=lambda stage: repositories.update_job_progress(
                        job_id, stage, {
                            "extracting": 15,
                            "detecting_language": 30,
                            "detecting_headings": 40,
                            "chunking": 50,
                            "generating_titles": 75,
                        }.get(stage, 10),
                    ),
                    is_cancelled=lambda: repositories.job_is_cancelled(job_id),
                )

        structure_path = Path(str(summary["structure_file"]))
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        structure_key = (
            f"sessions/{document['session_id']}/documents/{document['id']}/"
            f"jobs/{job_id}/structure.json"
        )
        storage.put_json(structure_key, structure)
        if repositories.job_is_cancelled(job_id):
            raise PipelineCancelled("Document processing was cancelled")
        completed = repositories.complete_ingestion_job(
            job_id, structure_storage_key=structure_key, stats=summary
        )
        if completed is False:
            return {"job_id": job_id, "status": "cancelled"}
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
    except PipelineCancelled:
        increment("genexam_jobs", type="document_ingestion", status="cancelled")
        return {"job_id": job_id, "status": "cancelled"}
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
    exam_id = str(
        request_data.get("exam_id")
        or repositories.get_exam_id_for_job(job_id)
        or ""
    )
    if not exam_id:
        repositories.fail_job(job_id, "EXAM_NOT_FOUND", "Pre-created exam record is missing")
        return {"job_id": job_id, "status": "failed"}
    document = repositories.get_document(str(job["document_id"]))
    if document is None:
        repositories.fail_job(job_id, "DOCUMENT_NOT_FOUND", "Document record is missing")
        return {"job_id": job_id, "status": "failed"}
    set_job_context(job_id, str(document["id"]))

    try:
        repositories.update_job_progress(job_id, "analyzing_request", 5)
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
                    processed_document_key=str(
                        (document.get("stats") or {}).get("processed_document_key") or ""
                    ) or None,
                    domain_type=str(
                        (document.get("stats") or {}).get("domain_type") or ""
                    ) or None,
                    subject=str(
                        (document.get("stats") or {}).get("subject") or ""
                    ) or None,
                    progress_callback=lambda stage, progress: repositories.update_job_progress(
                        job_id, stage, progress
                    ),
                    audit_callback=lambda stage, event_type, payload: (
                        repositories.append_exam_audit_event(
                            exam_id,
                            stage=stage,
                            event_type=event_type,
                            payload=payload,
                        )
                    ),
                )
        if repositories.job_is_cancelled(job_id):
            increment("genexam_jobs", type="exam_generation", status="cancelled")
            return {"job_id": job_id, "status": "cancelled"}
        exams = result.get("exams") or []
        if not any(exam.get("questions") for exam in exams):
            raise RuntimeError("; ".join(result.get("warnings") or []) or "No questions generated")

        shortage_warnings = _question_shortage_warnings(
            result, tasks, int(request_data["num_models"])
        )
        difficulty_warnings = _difficulty_fallback_warnings(result)
        persisted_warnings: list[Any] = [
            *list(result.get("warnings") or []),
            *shortage_warnings,
            *difficulty_warnings,
        ]
        if shortage_warnings:
            repositories.append_exam_audit_event(
                exam_id,
                stage="finalization",
                event_type="question_shortage",
                payload={"warnings": shortage_warnings},
            )
        metadata = dict(request_data.get("metadata") or {})
        metadata["document_language"] = result.get("document_language") or "en"
        metadata["quality_summary"] = dict(result.get("quality_summary") or {})
        metadata["question_shortages"] = shortage_warnings
        metadata["difficulty_fallbacks"] = difficulty_warnings
        metadata["generation_audit"] = {
            "invalid_questions": list(result.get("invalid_questions") or []),
            "unrepairable_plan_slots": list(result.get("unrepairable_plan_slots") or []),
            "semantic_identities": list(result.get("semantic_identities") or []),
            "duplicate_repair_history": list(result.get("duplicate_repair_history") or []),
            "plan_review_history": list(result.get("plan_review_history") or []),
            "plan_review_attempts": int(result.get("plan_review_attempts") or 0),
            "plan_duplicate_repair_attempts": dict(
                result.get("plan_duplicate_repair_attempts") or {}
            ),
            "unresolved_plan_duplicate_conflicts": list(
                result.get("unresolved_plan_duplicate_conflicts") or []
            ),
            "difficulty_adjustment_history": list(result.get("difficulty_adjustment_history") or []),
            "difficulty_adjustment_attempts": dict(result.get("difficulty_adjustment_attempts") or {}),
            "duplicate_repair_rounds": int(result.get("duplicate_repair_rounds") or 0),
            "duplicate_replacement_attempts": dict(
                result.get("duplicate_replacement_attempts") or {}
            ),
            "validation_repair_history": list(result.get("validation_repair_history") or []),
            "repair_manifest": list(result.get("repair_manifest") or []),
            "repair_replacement_memory": dict(result.get("repair_replacement_memory") or {}),
            "final_unresolved_slots": list(result.get("final_unresolved_slots") or []),
            "initial_global_reviews": list(result.get("initial_global_reviews") or []),
            "final_global_reviews": list(result.get("final_global_reviews") or []),
            "exam_quality_findings": list(result.get("exam_quality_findings") or []),
            "duplicate_reports": list(result.get("duplicate_reports") or []),
            "duplicate_similarity_distributions": list(
                result.get("duplicate_similarity_distributions") or []
            ),
            "assessment_catalog_key": result.get("assessment_catalog_key"),
            "assessment_catalog_cache_hit": result.get("assessment_catalog_cache_hit"),
            "assessment_compiler_cache_hit": result.get("assessment_compiler_cache_hit"),
            "assessment_catalog_usage": result.get("assessment_catalog_usage") or {},
            "slot_lifecycle": dict(result.get("slot_lifecycle") or {}),
            "developer_diagnostics": list(result.get("developer_diagnostics") or []),
            "mcq_task_reassignments": list(result.get("mcq_task_reassignments") or []),
        }
        for invalid in result.get("invalid_questions") or []:
            logger.warning(
                "Question omitted from public exam | slot_id=%s | failure_code=%s | reason=%s",
                invalid.get("slot_id"), invalid.get("failure_code"), invalid.get("reason"),
            )
        for missing in result.get("unrepairable_plan_slots") or []:
            logger.warning(
                "Plan slot omitted from public exam | slot_id=%s | failure_code=%s | reason=%s",
                missing.get("slot_id"), missing.get("failure_code"), missing.get("reason"),
            )
        status = "partial" if shortage_warnings or not result.get("complete") else "complete"
        completed = repositories.complete_generation_job(
            job_id,
            exam_id=exam_id,
            document_id=str(document["id"]),
            session_id=str(job["session_id"]),
            status="complete" if status == "complete" else "partial",
            exams=exams,
            warnings=persisted_warnings,
            metadata=metadata,
            evaluation=dict(result.get("eval") or {}),
        )
        if completed is False:
            increment("genexam_jobs", type="exam_generation", status="cancelled")
            return {"job_id": job_id, "status": "cancelled"}
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
        return {
            "job_id": job_id,
            "status": "completed",
            "exam_id": exam_id,
            "exam_status": status,
        }
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


@celery_app.task(name="genexam.cleanup_cancelled_document", ignore_result=True)
def cleanup_cancelled_document(document_id: str, session_id: str) -> dict[str, int]:
    """Remove regenerable artifacts left by a cancelled ingestion job."""
    deleted_vectors = VectorStore().delete_document(
        document_id, session_id=session_id, raise_errors=True
    )
    deleted_files = get_file_storage().delete_tree(
        f"sessions/{session_id}/documents/{document_id}"
    )
    deleted_files += _delete_pipeline_artifacts(document_id)
    logger.info(
        "Cancelled ingestion artifacts removed | document_id=%s | files=%d | vectors=%d",
        document_id,
        deleted_files,
        deleted_vectors,
    )
    return {"files": deleted_files, "vectors": deleted_vectors}


@celery_app.task(name="genexam.recover_stale_jobs")
def recover_stale_jobs() -> dict[str, int]:
    stale = repositories.recover_stale_jobs(config.JOB_STALE_AFTER_SECONDS)
    for job in stale:
        if job["type"] == "document_ingestion":
            ingest_document.apply_async(args=[str(job["id"])], task_id=str(job["id"]))
        elif job["type"] == "exam_generation":
            generate_exam.apply_async(args=[str(job["id"])], task_id=str(job["id"]))
    return {"requeued": len(stale)}


@celery_app.task(name="genexam.dispatch_undispatched_jobs")
def dispatch_undispatched_jobs() -> dict[str, int]:
    queued = repositories.undispatched_jobs()
    dispatched = 0
    for job in queued:
        if job["type"] == "document_ingestion":
            ingest_document.apply_async(args=[str(job["id"])], task_id=str(job["id"]))
        elif job["type"] == "exam_generation":
            generate_exam.apply_async(args=[str(job["id"])], task_id=str(job["id"]))
        else:
            continue
        repositories.mark_job_dispatched(str(job["id"]))
        dispatched += 1
    return {"dispatched": dispatched}


@celery_app.task(name="genexam.cleanup_expired_data")
def cleanup_expired_data() -> dict[str, Any]:
    return run_cleanup()
