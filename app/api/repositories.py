from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app import config, db


class ResourceNotFound(KeyError):
    pass


class InvalidState(RuntimeError):
    pass


class JobLimitExceeded(RuntimeError):
    pass


class GenerationRateLimitExceeded(RuntimeError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("Generation rate limit exceeded")
        self.retry_after_seconds = max(1, retry_after_seconds)


def hash_session_token(token: str) -> str:
    value = f"{config.SESSION_HASH_PEPPER}:{token}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def create_session(token_hash: str) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=config.SESSION_TTL_SECONDS)
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """INSERT INTO sessions (id, token_hash, expires_at)
                   VALUES (%s, %s, %s)
                   RETURNING *""",
                (session_id, token_hash, expires_at),
            )
            row = cursor.fetchone()
        conn.commit()
    assert row is not None
    return dict(row)


def resolve_session(token_hash: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """UPDATE sessions
                   SET last_seen_at = CURRENT_TIMESTAMP
                   WHERE token_hash = %s
                     AND revoked_at IS NULL
                     AND expires_at > CURRENT_TIMESTAMP
                   RETURNING *""",
                (token_hash,),
            )
            row = cursor.fetchone()
        conn.commit()
    return dict(row) if row else None


def create_document_with_job(
    *, session_id: str, filename: str, media_type: str, size_bytes: int,
    sha256: str, source_storage_key: str, document_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document_id = document_id or str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """INSERT INTO documents (
                       id, session_id, status, original_filename, media_type,
                       size_bytes, sha256, source_storage_key
                   ) VALUES (%s, %s, 'queued', %s, %s, %s, %s, %s)
                   RETURNING *""",
                (
                    document_id, session_id, filename, media_type, size_bytes,
                    sha256, source_storage_key,
                ),
            )
            document = cursor.fetchone()
            cursor.execute(
                """INSERT INTO jobs (
                       id, session_id, document_id, type, status, stage,
                       progress, max_attempts
                   ) VALUES (%s, %s, %s, 'document_ingestion', 'queued',
                             'queued', 0, 3)
                   RETURNING *""",
                (job_id, session_id, document_id),
            )
            job = cursor.fetchone()
        conn.commit()
    assert document is not None and job is not None
    return dict(document), dict(job)


def list_documents_for_session(session_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT id AS document_id, status, original_filename AS filename,
                          size_bytes, stats, error_code, error_message, created_at,
                          updated_at
                   FROM documents
                   WHERE session_id = %s AND status <> 'deleted'
                   ORDER BY created_at DESC""",
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]


def get_document_for_session(document_id: str, session_id: str) -> dict[str, Any]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT * FROM documents
                   WHERE id = %s AND session_id = %s AND status <> 'deleted'""",
                (document_id, session_id),
            )
            row = cursor.fetchone()
    if row is None:
        raise ResourceNotFound("Document not found")
    return dict(row)


def get_document(document_id: str) -> dict[str, Any] | None:
    """Worker-only lookup by id; API routes must use get_document_for_session."""
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM documents WHERE id = %s", (document_id,))
            row = cursor.fetchone()
    return dict(row) if row else None


def create_generation_job(
    *, session_id: str, document_id: str, request_data: dict[str, Any],
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            # Serialize limit decisions for this session across API replicas.
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))
            cursor.execute(
                """SELECT status FROM documents
                   WHERE id = %s AND session_id = %s AND status <> 'deleted'""",
                (document_id, session_id),
            )
            document = cursor.fetchone()
            if document is None:
                raise ResourceNotFound("Document not found")
            if document["status"] != "ready":
                raise InvalidState("Document is not ready for exam generation")

            cursor.execute(
                """SELECT COUNT(*) AS count FROM jobs
                   WHERE session_id = %s AND type = 'exam_generation'
                     AND status IN ('queued', 'running', 'retrying')""",
                (session_id,),
            )
            if int(cursor.fetchone()["count"]) >= config.MAX_ACTIVE_GENERATION_JOBS_PER_SESSION:
                raise JobLimitExceeded("Too many active generation jobs")

            cursor.execute(
                """SELECT created_at FROM jobs
                   WHERE session_id = %s AND type = 'exam_generation'
                     AND created_at > CURRENT_TIMESTAMP - INTERVAL '1 minute'
                   ORDER BY created_at ASC""",
                (session_id,),
            )
            recent = cursor.fetchall()
            if len(recent) >= config.GENERATION_REQUESTS_PER_MINUTE:
                oldest = recent[0]["created_at"]
                retry_after = int(max(1, 60 - (now - oldest).total_seconds()))
                raise GenerationRateLimitExceeded(retry_after)

            cursor.execute(
                """INSERT INTO jobs (
                       id, session_id, document_id, type, status, stage,
                       progress, request_data, max_attempts
                   ) VALUES (%s, %s, %s, 'exam_generation', 'queued',
                             'queued', 0, %s, 2)
                   RETURNING *""",
                (job_id, session_id, document_id, Jsonb(request_data)),
            )
            job = cursor.fetchone()
        conn.commit()
    assert job is not None
    return dict(job)


def get_job_for_session(job_id: str, session_id: str) -> dict[str, Any]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM jobs WHERE id = %s AND session_id = %s",
                (job_id, session_id),
            )
            row = cursor.fetchone()
    if row is None:
        raise ResourceNotFound("Job not found")
    return dict(row)


def get_job(job_id: str) -> dict[str, Any] | None:
    """Worker-only lookup by id; API routes must use get_job_for_session."""
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
            row = cursor.fetchone()
    return dict(row) if row else None


def mark_job_dispatched(job_id: str) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE jobs SET stage = 'dispatched', updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s AND status = 'queued' AND stage = 'queued'""",
                (job_id,),
            )
        conn.commit()


def undispatched_jobs() -> list[dict[str, Any]]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT id, type FROM jobs
                   WHERE status = 'queued' AND stage = 'queued'
                     AND created_at < CURRENT_TIMESTAMP - INTERVAL '15 seconds'
                   ORDER BY created_at
                   LIMIT 100"""
            )
            return [dict(row) for row in cursor.fetchall()]


def claim_job(job_id: str, worker_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """UPDATE jobs
                   SET status = 'running', stage = 'starting', progress = 1,
                       worker_id = %s, attempt_count = attempt_count + 1,
                       started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                       heartbeat_at = CURRENT_TIMESTAMP,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s AND status IN ('queued', 'retrying')
                   RETURNING *""",
                (worker_id, job_id),
            )
            row = cursor.fetchone()
        conn.commit()
    return dict(row) if row else None


def update_job_progress(job_id: str, stage: str, progress: int) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE jobs SET stage = %s, progress = %s,
                          heartbeat_at = CURRENT_TIMESTAMP,
                          updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s AND status = 'running'""",
                (stage, max(0, min(100, progress)), job_id),
            )
        conn.commit()


def heartbeat_job(job_id: str) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE jobs SET heartbeat_at = CURRENT_TIMESTAMP,
                          updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s AND status = 'running'""",
                (job_id,),
            )
        conn.commit()


def recover_stale_jobs(stale_after_seconds: int) -> list[dict[str, Any]]:
    """Move abandoned jobs to retrying/failed and return jobs to requeue."""
    requeue: list[dict[str, Any]] = []
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT id, type, attempt_count, max_attempts FROM jobs
                   WHERE status = 'running'
                     AND heartbeat_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                   FOR UPDATE SKIP LOCKED""",
                (stale_after_seconds,),
            )
            stale = cursor.fetchall()
            for row in stale:
                if int(row["attempt_count"]) < int(row["max_attempts"]):
                    cursor.execute(
                        """UPDATE jobs SET status = 'retrying', stage = 'retrying',
                                  worker_id = NULL, error_code = 'worker_lost',
                                  error_message = 'Worker heartbeat expired',
                                  updated_at = CURRENT_TIMESTAMP
                           WHERE id = %s""",
                        (row["id"],),
                    )
                    requeue.append(dict(row))
                else:
                    cursor.execute(
                        """UPDATE jobs SET status = 'failed', stage = 'failed',
                                  error_code = 'worker_lost',
                                  error_message = 'Worker heartbeat expired',
                                  finished_at = CURRENT_TIMESTAMP,
                                  updated_at = CURRENT_TIMESTAMP
                           WHERE id = %s""",
                        (row["id"],),
                    )
                    if row["type"] == "document_ingestion":
                        cursor.execute(
                            """UPDATE documents d SET status = 'failed',
                                      error_code = 'worker_lost',
                                      error_message = 'Worker heartbeat expired',
                                      updated_at = CURRENT_TIMESTAMP
                               FROM jobs j
                               WHERE j.id = %s AND d.id = j.document_id""",
                            (row["id"],),
                        )
        conn.commit()
    return requeue


def retry_job(job_id: str, error_code: str, error_message: str) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE jobs SET status = 'retrying', stage = 'retrying',
                          error_code = %s, error_message = %s,
                          heartbeat_at = CURRENT_TIMESTAMP,
                          updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s AND status = 'running'""",
                (error_code, str(error_message)[:1000], job_id),
            )
        conn.commit()


def complete_ingestion_job(
    job_id: str, *, structure_storage_key: str, stats: dict[str, Any]
) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE documents d
                   SET status = 'ready', structure_storage_key = %s,
                       active_index_job_id = %s, stats = %s,
                       error_code = NULL, error_message = NULL,
                       updated_at = CURRENT_TIMESTAMP
                   FROM jobs j
                   WHERE j.id = %s AND d.id = j.document_id""",
                (structure_storage_key, job_id, Jsonb(stats), job_id),
            )
            cursor.execute(
                """UPDATE jobs SET status = 'completed', stage = 'completed',
                          progress = 100, result_data = %s,
                          finished_at = CURRENT_TIMESTAMP,
                          updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (Jsonb({"document_id": stats.get("document_id")}), job_id),
            )
        conn.commit()


def complete_generation_job(
    job_id: str, *, exam_id: str, document_id: str, session_id: str,
    status: str, exams: list[dict[str, Any]], warnings: list[str],
    metadata: dict[str, Any], evaluation: dict[str, Any],
) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO exams (
                       id, session_id, document_id, job_id, status, metadata,
                       exams, warnings, evaluation
                   ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    exam_id, session_id, document_id, job_id, status,
                    Jsonb(metadata), Jsonb(exams), Jsonb(warnings), Jsonb(evaluation),
                ),
            )
            cursor.execute(
                """UPDATE jobs SET status = 'completed', stage = 'completed',
                          progress = 100, result_data = %s,
                          finished_at = CURRENT_TIMESTAMP,
                          updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (Jsonb({"exam_id": exam_id}), job_id),
            )
        conn.commit()


def fail_job(job_id: str, error_code: str, error_message: str) -> None:
    safe_message = str(error_message)[:1000]
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE jobs SET status = 'failed', stage = 'failed',
                          error_code = %s, error_message = %s,
                          finished_at = CURRENT_TIMESTAMP,
                          updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (error_code, safe_message, job_id),
            )
            cursor.execute(
                """UPDATE documents d SET status = 'failed', error_code = %s,
                          error_message = %s, updated_at = CURRENT_TIMESTAMP
                   FROM jobs j
                   WHERE j.id = %s AND j.type = 'document_ingestion'
                     AND d.id = j.document_id""",
                (error_code, safe_message, job_id),
            )
        conn.commit()


def get_exam_for_session(exam_id: str, session_id: str) -> dict[str, Any]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT id AS exam_id, document_id, created_at, exams,
                          warnings, metadata, evaluation, status
                   FROM exams WHERE id = %s AND session_id = %s""",
                (exam_id, session_id),
            )
            row = cursor.fetchone()
    if row is None:
        raise ResourceNotFound("Exam not found")
    return dict(row)
