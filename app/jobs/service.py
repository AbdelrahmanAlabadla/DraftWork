from __future__ import annotations


def enqueue_ingestion(job_id: str) -> None:
    from app.jobs.tasks import ingest_document

    ingest_document.delay(job_id)


def enqueue_generation(job_id: str) -> None:
    from app.jobs.tasks import generate_exam

    generate_exam.delay(job_id)
