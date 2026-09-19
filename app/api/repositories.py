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


class IdempotencyConflict(RuntimeError):
    pass


class SessionOwnershipConflict(RuntimeError):
    pass


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


def revoke_session(session_id: str) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE sessions SET revoked_at = CURRENT_TIMESTAMP,
                          expires_at = CURRENT_TIMESTAMP
                   WHERE id = %s AND revoked_at IS NULL""",
                (session_id,),
            )
        conn.commit()


def get_user_by_clerk_id(clerk_user_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM users WHERE clerk_user_id = %s",
                (clerk_user_id,),
            )
            row = cursor.fetchone()
    return dict(row) if row else None


def upsert_user(
    clerk_user_id: str, *, primary_email: str | None = None,
    display_name: str | None = None, image_url: str | None = None,
) -> dict[str, Any]:
    user_id = str(uuid.uuid4())
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """INSERT INTO users (
                       id, clerk_user_id, primary_email, display_name, image_url,
                       last_login_at
                   ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                   ON CONFLICT (clerk_user_id) DO UPDATE SET
                       primary_email = COALESCE(EXCLUDED.primary_email, users.primary_email),
                       display_name = COALESCE(EXCLUDED.display_name, users.display_name),
                       image_url = COALESCE(EXCLUDED.image_url, users.image_url),
                       last_login_at = CURRENT_TIMESTAMP,
                       updated_at = CURRENT_TIMESTAMP
                   RETURNING *""",
                (user_id, clerk_user_id, primary_email, display_name, image_url),
            )
            row = cursor.fetchone()
        conn.commit()
    assert row is not None
    return dict(row)


def claim_session(session_id: str, user_id: str) -> dict[str, Any]:
    """Attach one anonymous session and its exams, without moving stored data."""
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
            session = cursor.fetchone()
            if session is None:
                raise ResourceNotFound("Session not found")
            owner = str(session["user_id"]) if session.get("user_id") else None
            if owner is not None and owner != user_id:
                raise SessionOwnershipConflict("Session already belongs to another user")
            cursor.execute(
                """UPDATE sessions SET user_id = %s,
                          claimed_at = COALESCE(claimed_at, CURRENT_TIMESTAMP),
                          last_seen_at = CURRENT_TIMESTAMP
                   WHERE id = %s RETURNING *""",
                (user_id, session_id),
            )
            claimed = cursor.fetchone()
            cursor.execute(
                """UPDATE exams SET user_id = %s
                   WHERE session_id = %s AND user_id IS NULL""",
                (user_id, session_id),
            )
            cursor.execute(
                """UPDATE users SET last_login_at = CURRENT_TIMESTAMP,
                          updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (user_id,),
            )
        conn.commit()
    assert claimed is not None
    return dict(claimed)


def get_user(user_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cursor.fetchone()
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


def create_document_with_job_idempotent(
    *, session_id: str, filename: str, media_type: str, size_bytes: int,
    sha256: str, source_storage_key: str, document_id: str,
    idempotency_key: str, request_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    endpoint = "create_document"
    job_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=config.IDEMPOTENCY_TTL_SECONDS
    )
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"{session_id}:{endpoint}:{idempotency_key}",),
            )
            cursor.execute(
                """DELETE FROM idempotency_records
                   WHERE session_id = %s AND endpoint = %s
                     AND idempotency_key = %s
                     AND expires_at <= CURRENT_TIMESTAMP""",
                (session_id, endpoint, idempotency_key),
            )
            cursor.execute(
                """SELECT i.request_hash, d.*, j.id AS existing_job_id
                   FROM idempotency_records i
                   JOIN documents d ON d.id = i.document_id
                   JOIN jobs j ON j.id = i.job_id
                   WHERE i.session_id = %s AND i.endpoint = %s
                     AND i.idempotency_key = %s
                     AND i.expires_at > CURRENT_TIMESTAMP""",
                (session_id, endpoint, idempotency_key),
            )
            existing = cursor.fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflict(
                        "The idempotency key was already used with different upload data"
                    )
                document = dict(existing)
                cursor.execute("SELECT * FROM jobs WHERE id = %s", (existing["existing_job_id"],))
                job = cursor.fetchone()
                assert job is not None
                return document, dict(job), True

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
            cursor.execute(
                """INSERT INTO idempotency_records (
                       session_id, endpoint, idempotency_key, request_hash,
                       job_id, document_id, expires_at
                   ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    session_id, endpoint, idempotency_key, request_hash,
                    job_id, document_id, expires_at,
                ),
            )
        conn.commit()
    assert document is not None and job is not None
    return dict(document), dict(job), False


def list_documents_for_session(
    session_id: str, user_id: str | None = None
) -> list[dict[str, Any]]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            if user_id is None:
                cursor.execute(
                    """SELECT id AS document_id, status,
                              original_filename AS filename, size_bytes, stats,
                              error_code, error_message, created_at, updated_at
                       FROM documents
                       WHERE session_id = %s AND status <> 'deleted'
                       ORDER BY created_at DESC""",
                    (session_id,),
                )
                return [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT id AS document_id, status, original_filename AS filename,
                          size_bytes, stats, error_code, error_message, created_at,
                          updated_at
                   FROM documents d
                   WHERE d.status <> 'deleted' AND (
                       d.session_id = %s OR EXISTS (
                           SELECT 1 FROM sessions s
                           WHERE s.id = d.session_id AND s.user_id = %s
                       )
                   ) ORDER BY d.created_at DESC""",
                (session_id, user_id),
            )
            return [dict(row) for row in cursor.fetchall()]


def get_document_for_session(
    document_id: str, session_id: str, user_id: str | None = None
) -> dict[str, Any]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            if user_id is None:
                cursor.execute(
                    """SELECT * FROM documents
                       WHERE id = %s AND session_id = %s AND status <> 'deleted'""",
                    (document_id, session_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ResourceNotFound("Document not found")
                return dict(row)
            cursor.execute(
                """SELECT d.* FROM documents d
                   WHERE d.id = %s AND d.status <> 'deleted' AND (
                       d.session_id = %s OR EXISTS (
                           SELECT 1 FROM sessions s
                           WHERE s.id = d.session_id AND s.user_id = %s
                       )
                   )""",
                (document_id, session_id, user_id),
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
    idempotency_key: str | None = None, request_hash: str | None = None,
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            # Serialize limit decisions for this session across API replicas.
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))
            if idempotency_key:
                cursor.execute(
                    """DELETE FROM idempotency_records
                       WHERE session_id = %s AND endpoint = 'create_exam_job'
                         AND idempotency_key = %s
                         AND expires_at <= CURRENT_TIMESTAMP""",
                    (session_id, idempotency_key),
                )
                cursor.execute(
                    """SELECT i.request_hash, j.*
                       FROM idempotency_records i
                       JOIN jobs j ON j.id = i.job_id
                       WHERE i.session_id = %s AND i.endpoint = 'create_exam_job'
                         AND i.idempotency_key = %s
                         AND i.expires_at > CURRENT_TIMESTAMP""",
                    (session_id, idempotency_key),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing["request_hash"] != request_hash:
                        raise IdempotencyConflict(
                            "The idempotency key was already used with different generation data"
                        )
                    result = dict(existing)
                    result["_idempotency_reused"] = True
                    return result
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
            if idempotency_key:
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=config.IDEMPOTENCY_TTL_SECONDS
                )
                cursor.execute(
                    """INSERT INTO idempotency_records (
                           session_id, endpoint, idempotency_key, request_hash,
                           job_id, document_id, expires_at
                       ) VALUES (%s, 'create_exam_job', %s, %s, %s, %s, %s)""",
                    (
                        session_id, idempotency_key, request_hash, job_id,
                        document_id, expires_at,
                    ),
                )
        conn.commit()
    assert job is not None
    result = dict(job)
    result["_idempotency_reused"] = False
    return result


def get_job_for_session(
    job_id: str, session_id: str, user_id: str | None = None
) -> dict[str, Any]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            if user_id is None:
                cursor.execute(
                    "SELECT * FROM jobs WHERE id = %s AND session_id = %s",
                    (job_id, session_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ResourceNotFound("Job not found")
                return dict(row)
            cursor.execute(
                """SELECT j.* FROM jobs j
                   WHERE j.id = %s AND (
                       j.session_id = %s OR EXISTS (
                           SELECT 1 FROM sessions s
                           WHERE s.id = j.session_id AND s.user_id = %s
                       )
                   )""",
                (job_id, session_id, user_id),
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
                                  worker_id = NULL, error_code = 'WORKER_LOST',
                                  error_message = 'Worker heartbeat expired',
                                  updated_at = CURRENT_TIMESTAMP
                           WHERE id = %s""",
                        (row["id"],),
                    )
                    requeue.append(dict(row))
                else:
                    cursor.execute(
                        """UPDATE jobs SET status = 'failed', stage = 'failed',
                                  error_code = 'WORKER_LOST',
                                  error_message = 'Worker heartbeat expired',
                                  finished_at = CURRENT_TIMESTAMP,
                                  updated_at = CURRENT_TIMESTAMP
                           WHERE id = %s""",
                        (row["id"],),
                    )
                    if row["type"] == "document_ingestion":
                        cursor.execute(
                            """UPDATE documents d SET status = 'failed',
                                      error_code = 'WORKER_LOST',
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
                       id, session_id, user_id, document_id, job_id, status,
                       metadata, exams, warnings, evaluation
                   ) SELECT %s, %s, s.user_id, %s, %s, %s, %s, %s, %s, %s
                     FROM sessions s WHERE s.id = %s""",
                (
                    exam_id, session_id, document_id, job_id, status,
                    Jsonb(metadata), Jsonb(exams), Jsonb(warnings), Jsonb(evaluation),
                    session_id,
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


def get_exam_for_session(
    exam_id: str, session_id: str, user_id: str | None = None
) -> dict[str, Any]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            if user_id is None:
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
            cursor.execute(
                """SELECT id AS exam_id, document_id, created_at, exams,
                          warnings, metadata, evaluation, status
                   FROM exams WHERE id = %s AND (
                       session_id = %s OR user_id = %s
                   )""",
                (exam_id, session_id, user_id),
            )
            row = cursor.fetchone()
    if row is None:
        raise ResourceNotFound("Exam not found")
    return dict(row)


def list_exams_for_user(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT id AS exam_id, document_id, created_at, status,
                          metadata, warnings,
                          jsonb_array_length(exams) AS model_count
                   FROM exams
                   WHERE user_id = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (user_id, max(1, min(limit, 200))),
            )
            return [dict(row) for row in cursor.fetchall()]


def cleanup_document_candidates(
    *, abandoned_after_seconds: int, limit: int
) -> list[dict[str, Any]]:
    """Return expired-session or abandoned documents without deleting exams."""
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT d.id, d.session_id, d.source_storage_key,
                          d.structure_storage_key, d.active_index_job_id,
                          EXISTS(SELECT 1 FROM exams e WHERE e.document_id = d.id)
                              AS has_exam
                   FROM documents d
                   JOIN sessions s ON s.id = d.session_id
                   WHERE d.status <> 'deleted'
                     AND (
                         s.expires_at <= CURRENT_TIMESTAMP
                         OR (
                             d.status IN ('uploading', 'queued', 'processing', 'failed')
                             AND d.updated_at < CURRENT_TIMESTAMP
                                 - (%s * INTERVAL '1 second')
                         )
                     )
                   ORDER BY d.updated_at
                   FOR UPDATE OF d SKIP LOCKED
                   LIMIT %s""",
                (abandoned_after_seconds, limit),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        conn.commit()
    return rows


def mark_document_assets_deleted(document_id: str) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE documents
                   SET status = 'deleted', updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (document_id,),
            )
        conn.commit()


def cleanup_database_records(*, terminal_job_days: int) -> dict[str, int]:
    """Remove regenerable metadata while explicitly preserving every exam row."""
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "DELETE FROM idempotency_records WHERE expires_at <= CURRENT_TIMESTAMP"
            )
            idempotency = cursor.rowcount
            cursor.execute(
                """DELETE FROM jobs j
                   WHERE j.status IN ('completed', 'failed', 'cancelled')
                     AND j.finished_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')""",
                (terminal_job_days,),
            )
            jobs = cursor.rowcount
            cursor.execute(
                """DELETE FROM sessions s
                   WHERE s.expires_at <= CURRENT_TIMESTAMP"""
            )
            sessions = cursor.rowcount
        conn.commit()
    return {
        "idempotency_records": max(0, idempotency),
        "jobs": max(0, jobs),
        "sessions": max(0, sessions),
    }


def expired_session_ids(*, limit: int) -> list[str]:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT s.id FROM sessions s
                   WHERE s.expires_at <= CURRENT_TIMESTAMP
                   ORDER BY s.expires_at
                   LIMIT %s""",
                (limit,),
            )
            return [str(row[0]) for row in cursor.fetchall()]


def active_document_scopes() -> set[tuple[str, str]]:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT session_id, id FROM documents
                   WHERE status <> 'deleted'"""
            )
            return {(str(session_id), str(document_id)) for session_id, document_id in cursor.fetchall()}


def start_cleanup_run() -> int:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE cleanup_runs
                   SET status = 'failed', finished_at = CURRENT_TIMESTAMP,
                       error_message = 'Cleanup worker stopped before completion'
                   WHERE status = 'running'
                     AND started_at < CURRENT_TIMESTAMP - INTERVAL '2 hours'"""
            )
            cursor.execute(
                "INSERT INTO cleanup_runs (status) VALUES ('running') RETURNING id"
            )
            row = cursor.fetchone()
        conn.commit()
    assert row is not None
    return int(row[0])


def finish_cleanup_run(
    cleanup_id: int, *, stats: dict[str, Any] | None = None, error: str | None = None
) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE cleanup_runs
                   SET status = %s, stats = %s, error_message = %s,
                       finished_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (
                    "failed" if error else "completed",
                    Jsonb(stats or {}),
                    str(error)[:1000] if error else None,
                    cleanup_id,
                ),
            )
        conn.commit()


def record_llm_usage(
    *, job_id: str | None, provider: str, model: str, operation: str,
    input_tokens: int, output_tokens: int, cached_tokens: int,
    reasoning_tokens: int, estimated_cost_usd: float,
    duration_ms: int | None = None,
) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO llm_usage (
                       job_id, provider, model, operation, input_tokens,
                       output_tokens, cached_tokens, reasoning_tokens,
                       estimated_cost_usd, duration_ms
                   ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    job_id, provider, model, operation, input_tokens,
                    output_tokens, cached_tokens, reasoning_tokens,
                    estimated_cost_usd, duration_ms,
                ),
            )
        conn.commit()


def job_status_counts() -> dict[str, int]:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
            return {str(status): int(count) for status, count in cursor.fetchall()}


def monitoring_snapshot() -> dict[str, list[dict[str, Any]]]:
    """Return durable worker metrics for the API process to expose."""
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT type, status, COUNT(*) AS count,
                          COALESCE(SUM(EXTRACT(EPOCH FROM (finished_at - started_at)))
                              FILTER (WHERE finished_at IS NOT NULL
                                      AND started_at IS NOT NULL), 0) AS duration_sum,
                          COUNT(*) FILTER (WHERE finished_at IS NOT NULL
                                           AND started_at IS NOT NULL) AS duration_count
                   FROM jobs
                   GROUP BY type, status"""
            )
            jobs = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT status, COUNT(*) AS count,
                          COALESCE(SUM(EXTRACT(EPOCH FROM (finished_at - started_at)))
                              FILTER (WHERE finished_at IS NOT NULL), 0) AS duration_sum,
                          COUNT(*) FILTER (WHERE finished_at IS NOT NULL) AS duration_count
                   FROM cleanup_runs
                   GROUP BY status"""
            )
            cleanup_runs = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT provider, model, operation, COUNT(*) AS calls,
                          COALESCE(SUM(input_tokens), 0) AS input_tokens,
                          COALESCE(SUM(output_tokens), 0) AS output_tokens,
                          COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
                          COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                          COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd,
                          COALESCE(SUM(duration_ms), 0) / 1000.0 AS duration_sum,
                          COUNT(duration_ms) AS duration_count
                   FROM llm_usage
                   GROUP BY provider, model, operation"""
            )
            llm_usage = [dict(row) for row in cursor.fetchall()]
    return {"jobs": jobs, "cleanup_runs": cleanup_runs, "llm_usage": llm_usage}
