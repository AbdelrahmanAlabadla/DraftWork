"""Deduplicate durable provider-attempt records.

Revision ID: 0008_llm_request_event_idempotency
Revises: 0007_llm_request_events
"""
from alembic import op

revision = "0008_llm_request_event_idempotency"
down_revision = "0007_llm_request_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Revision identifiers are longer than Alembic's default VARCHAR(32).
    op.execute(
        "ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(128)"
    )
    op.execute(
        """CREATE UNIQUE INDEX llm_request_events_job_request_unique
             ON llm_request_events (job_id, request_id)
             WHERE job_id IS NOT NULL AND request_id IS NOT NULL"""
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS llm_request_events_job_request_unique")
