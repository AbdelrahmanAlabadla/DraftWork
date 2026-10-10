from __future__ import annotations

import os
import secrets

import pytest

from app import db
from app.api import repositories


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="set RUN_POSTGRES_TESTS=1 to exercise the configured PostgreSQL database",
)


def test_core_repository_ownership_and_active_job_limit():
    session = repositories.create_session(
        repositories.hash_session_token(secrets.token_urlsafe(32))
    )
    session_id = str(session["id"])
    try:
        document, _ = repositories.create_document_with_job(
            session_id=session_id,
            filename="integration.pdf",
            media_type="application/pdf",
            size_bytes=10,
            sha256="0" * 64,
            source_storage_key="integration/source.pdf",
        )
        document_id = str(document["id"])
        with db.connection() as connection:
            connection.execute(
                "UPDATE documents SET status = 'ready' WHERE id = %s",
                (document_id,),
            )
            connection.commit()

        request_data = {
            "tasks": [["mcq", 1]],
            "num_models": 1,
            "difficulty": "easy",
            "child_ids": ["child"],
            "metadata": {},
        }
        first = repositories.create_generation_job(
            session_id=session_id,
            document_id=document_id,
            request_data=request_data,
        )
        second = repositories.create_generation_job(
            session_id=session_id,
            document_id=document_id,
            request_data=request_data,
        )
        assert first["session_id"] == second["session_id"]
        with pytest.raises(repositories.JobLimitExceeded):
            repositories.create_generation_job(
                session_id=session_id,
                document_id=document_id,
                request_data=request_data,
            )
        with db.connection() as connection:
            connection.execute(
                """UPDATE jobs SET status = 'completed', stage = 'completed',
                          finished_at = CURRENT_TIMESTAMP
                   WHERE id = ANY(%s)""",
                ([first["id"], second["id"]],),
            )
            connection.commit()
        with pytest.raises(repositories.GenerationRateLimitExceeded):
            repositories.create_generation_job(
                session_id=session_id,
                document_id=document_id,
                request_data=request_data,
            )
        with pytest.raises(repositories.ResourceNotFound):
            repositories.get_document_for_session(
                document_id, "00000000-0000-4000-8000-000000000000"
            )
    finally:
        with db.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            connection.commit()


def test_clerk_session_claim_preserves_exam_across_session_deletion():
    session = repositories.create_session(
        repositories.hash_session_token(secrets.token_urlsafe(32))
    )
    session_id = str(session["id"])
    user = repositories.upsert_user(
        "user_integration_claim", primary_email="claim@example.test"
    )
    user_id = str(user["id"])
    exam_id = None
    try:
        document, job = repositories.create_document_with_job(
            session_id=session_id,
            filename="claimed.pdf",
            media_type="application/pdf",
            size_bytes=10,
            sha256="1" * 64,
            source_storage_key="integration/claimed.pdf",
        )
        with db.connection() as connection:
            connection.execute(
                "UPDATE documents SET status = 'ready' WHERE id = %s",
                (document["id"],),
            )
            connection.execute(
                """UPDATE jobs SET status = 'completed', stage = 'completed',
                          finished_at = CURRENT_TIMESTAMP WHERE id = %s""",
                (job["id"],),
            )
            connection.commit()
        generation_job = repositories.create_generation_job(
            session_id=session_id,
            document_id=str(document["id"]),
            request_data={
                "tasks": [["mcq", 1]], "num_models": 1,
                "difficulty": "easy", "child_ids": ["child"], "metadata": {},
            },
        )
        exam_id = str(generation_job["exam_id"])
        repositories.claim_job(str(generation_job["id"]), "integration-worker")
        repositories.complete_generation_job(
            str(generation_job["id"]),
            exam_id=exam_id,
            document_id=str(document["id"]),
            session_id=session_id,
            status="complete",
            exams=[{"model_number": 1}],
            warnings=[],
            metadata={"exam_title": "Claimed"},
            evaluation={},
        )
        repositories.claim_session(session_id, user_id)

        second = repositories.create_session(
            repositories.hash_session_token(secrets.token_urlsafe(32))
        )
        second_id = str(second["id"])
        repositories.claim_session(second_id, user_id)
        saved_exam = repositories.get_exam_for_session(exam_id, second_id, user_id)
        assert saved_exam["exam_id"] == exam_id
        assert saved_exam["status"] == "complete"

        with db.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            row = connection.execute(
                "SELECT session_id, user_id FROM exams WHERE id = %s", (exam_id,)
            ).fetchone()
            connection.commit()
        assert row[0] is None
        assert str(row[1]) == user_id
    finally:
        with db.connection() as connection:
            if exam_id:
                connection.execute("DELETE FROM exams WHERE id = %s", (exam_id,))
            connection.execute(
                "DELETE FROM sessions WHERE user_id = %s OR id = %s",
                (user_id, session_id),
            )
            connection.execute("DELETE FROM users WHERE id = %s", (user_id,))
            connection.commit()


def test_failed_exam_keeps_canonical_id_and_stage_audit_history():
    session = repositories.create_session(
        repositories.hash_session_token(secrets.token_urlsafe(32))
    )
    session_id = str(session["id"])
    document_id = None
    try:
        document, ingestion_job = repositories.create_document_with_job(
            session_id=session_id,
            filename="audit.pdf",
            media_type="application/pdf",
            size_bytes=10,
            sha256="2" * 64,
            source_storage_key="integration/audit.pdf",
        )
        document_id = str(document["id"])
        with db.connection() as connection:
            connection.execute(
                "UPDATE documents SET status = 'ready' WHERE id = %s", (document_id,)
            )
            connection.execute(
                """UPDATE jobs SET status = 'completed', stage = 'completed',
                          finished_at = CURRENT_TIMESTAMP WHERE id = %s""",
                (ingestion_job["id"],),
            )
            connection.commit()
        generation_job = repositories.create_generation_job(
            session_id=session_id,
            document_id=document_id,
            request_data={
                "tasks": [["mcq", 1]], "num_models": 1,
                "difficulty": "easy", "child_ids": ["child"], "metadata": {},
            },
        )
        exam_id = str(generation_job["exam_id"])
        assert generation_job["request_data"]["exam_id"] == exam_id
        assert repositories.get_exam_for_session(exam_id, session_id)["status"] == "queued"

        repositories.claim_job(str(generation_job["id"]), "integration-worker")
        evidence = {
            "generated_questions": [{"question_id": "model1_mcq_1", "question": "Q?"}],
            "validator_verdicts": [{"question_id": "model1_mcq_1", "action": "FIX_ANSWER"}],
            "repair": {"status": "accepted"},
            "duplicate": {"candidate_pairs": [], "fact_cache_hits": 0},
        }
        repositories.append_exam_audit_event(
            exam_id, stage="generation", event_type="questions_generated", payload=evidence
        )
        repositories.fail_job(
            str(generation_job["id"]), "GENERATION_FAILED", "controlled failure"
        )

        record = repositories.get_exam_for_session(exam_id, session_id)
        events = repositories.list_exam_audit_events(exam_id, session_id)
        assert record["exam_id"] == exam_id
        assert record["status"] == "failed"
        assert record["stage"] == "failed"
        generated = next(event for event in events if event["event_type"] == "questions_generated")
        assert generated["payload"] == evidence
        assert events[-1]["event_type"] == "job_failed"
    finally:
        if document_id:
            with db.connection() as connection:
                connection.execute("DELETE FROM exams WHERE document_id = %s", (document_id,))
                connection.execute("DELETE FROM jobs WHERE document_id = %s", (document_id,))
                connection.execute("DELETE FROM documents WHERE id = %s", (document_id,))
                connection.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
                connection.commit()
