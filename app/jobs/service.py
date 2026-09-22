from __future__ import annotations


def enqueue_ingestion(job_id: str) -> None:
    from app.jobs.tasks import ingest_document

    ingest_document.apply_async(args=[job_id], task_id=job_id)


def enqueue_generation(job_id: str) -> None:
    from app.jobs.tasks import generate_exam

    generate_exam.apply_async(args=[job_id], task_id=job_id)


def terminate_job(
    job_id: str,
    *,
    document_id: str | None = None,
    session_id: str | None = None,
    cleanup_document: bool = False,
) -> None:
    from app.jobs.celery_app import celery_app

    celery_app.control.revoke(job_id, terminate=True, signal="SIGTERM")
    if cleanup_document and document_id and session_id:
        from app.jobs.tasks import cleanup_cancelled_document

        cleanup_cancelled_document.apply_async(
            args=[document_id, session_id], countdown=2
        )
