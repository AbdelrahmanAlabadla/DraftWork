"""Fence worker executions and order durable generation checkpoints.

Revision ID: 0006_job_execution_fencing
Revises: 0005_postgres_document_chunks
"""

from alembic import op


revision = "0006_job_execution_fencing"
down_revision = "0005_postgres_document_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE jobs ADD COLUMN claim_generation BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE jobs ADD COLUMN checkpoint_revision BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE jobs ADD COLUMN request_fingerprint TEXT"
    )
    op.execute("ALTER TABLE exam_audit_events ADD COLUMN job_id UUID")
    op.execute("ALTER TABLE exam_audit_events ADD COLUMN claim_generation BIGINT")
    op.execute("ALTER TABLE exam_audit_events ADD COLUMN checkpoint_revision BIGINT")
    op.execute("ALTER TABLE exam_audit_events ADD COLUMN request_fingerprint TEXT")
    op.execute(
        """UPDATE exam_audit_events e SET job_id = x.job_id
             FROM exams x WHERE x.id = e.exam_id AND e.job_id IS NULL"""
    )
    op.execute(
        """ALTER TABLE exam_audit_events
             ADD CONSTRAINT exam_audit_events_job_fk
             FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE"""
    )
    op.execute(
        """CREATE UNIQUE INDEX exam_audit_events_job_checkpoint_revision_uidx
             ON exam_audit_events (job_id, checkpoint_revision)
             WHERE checkpoint_revision IS NOT NULL"""
    )
    op.execute(
        """CREATE INDEX exam_audit_events_job_revision_idx
             ON exam_audit_events (job_id, checkpoint_revision, id)"""
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS exam_audit_events_job_revision_idx")
    op.execute("DROP INDEX IF EXISTS exam_audit_events_job_checkpoint_revision_uidx")
    op.execute("ALTER TABLE exam_audit_events DROP CONSTRAINT IF EXISTS exam_audit_events_job_fk")
    op.execute("ALTER TABLE exam_audit_events DROP COLUMN IF EXISTS request_fingerprint")
    op.execute("ALTER TABLE exam_audit_events DROP COLUMN IF EXISTS checkpoint_revision")
    op.execute("ALTER TABLE exam_audit_events DROP COLUMN IF EXISTS claim_generation")
    op.execute("ALTER TABLE exam_audit_events DROP COLUMN IF EXISTS job_id")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS request_fingerprint")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS checkpoint_revision")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS claim_generation")
