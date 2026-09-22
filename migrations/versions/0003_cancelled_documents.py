"""Represent user-cancelled document processing explicitly.

Revision ID: 0003_cancelled_documents
Revises: 0002_clerk_accounts
"""

from alembic import op


revision = "0003_cancelled_documents"
down_revision = "0002_clerk_accounts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents DROP CONSTRAINT documents_status_check")
    op.execute(
        """ALTER TABLE documents ADD CONSTRAINT documents_status_check
           CHECK (status IN (
               'uploading', 'queued', 'processing', 'ready',
               'failed', 'cancelled', 'deleted'
           ))"""
    )


def downgrade() -> None:
    op.execute("UPDATE documents SET status = 'failed' WHERE status = 'cancelled'")
    op.execute("ALTER TABLE documents DROP CONSTRAINT documents_status_check")
    op.execute(
        """ALTER TABLE documents ADD CONSTRAINT documents_status_check
           CHECK (status IN (
               'uploading', 'queued', 'processing', 'ready', 'failed', 'deleted'
           ))"""
    )
