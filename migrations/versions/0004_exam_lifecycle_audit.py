"""Persist the complete website exam lifecycle and stage audit events.

Revision ID: 0004_exam_lifecycle_audit
Revises: 0003_cancelled_documents
"""

from alembic import op


revision = "0004_exam_lifecycle_audit"
down_revision = "0003_cancelled_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE exams DROP CONSTRAINT exams_status_check")
    op.execute(
        """ALTER TABLE exams ADD CONSTRAINT exams_status_check
           CHECK (status IN (
               'queued', 'running', 'retrying', 'complete', 'partial',
               'failed', 'cancelled'
           ))"""
    )
    op.execute("ALTER TABLE exams ADD COLUMN stage TEXT NOT NULL DEFAULT 'queued'")
    op.execute(
        "ALTER TABLE exams ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
    )
    op.execute(
        """CREATE TABLE exam_audit_events (
               id BIGSERIAL PRIMARY KEY,
               exam_id TEXT NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
               stage TEXT NOT NULL,
               event_type TEXT NOT NULL,
               payload JSONB NOT NULL DEFAULT '{}'::jsonb,
               created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    op.execute(
        """CREATE INDEX exam_audit_events_exam_created_idx
           ON exam_audit_events (exam_id, created_at, id)"""
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS exam_audit_events")
    op.execute("ALTER TABLE exams DROP COLUMN IF EXISTS updated_at")
    op.execute("ALTER TABLE exams DROP COLUMN IF EXISTS stage")
    op.execute("UPDATE exams SET status = 'partial' WHERE status NOT IN ('complete', 'partial')")
    op.execute("ALTER TABLE exams DROP CONSTRAINT exams_status_check")
    op.execute(
        """ALTER TABLE exams ADD CONSTRAINT exams_status_check
           CHECK (status IN ('complete', 'partial'))"""
    )
