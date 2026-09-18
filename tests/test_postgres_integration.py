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
