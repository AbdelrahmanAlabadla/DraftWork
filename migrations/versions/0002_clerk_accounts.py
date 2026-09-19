"""Add Clerk-backed user ownership.

Revision ID: 0002_clerk_accounts
Revises: 0001_baseline
"""

from alembic import op

revision = "0002_clerk_accounts"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE users (
            id UUID PRIMARY KEY,
            clerk_user_id TEXT NOT NULL UNIQUE,
            primary_email TEXT,
            display_name TEXT,
            image_url TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_login_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        """ALTER TABLE sessions
           ADD COLUMN user_id UUID REFERENCES users(id) ON DELETE SET NULL,
           ADD COLUMN claimed_at TIMESTAMPTZ"""
    )
    op.execute(
        """ALTER TABLE exams
           ADD COLUMN user_id UUID REFERENCES users(id) ON DELETE SET NULL"""
    )
    op.execute(
        """CREATE INDEX idx_sessions_user_created
           ON sessions (user_id, created_at DESC) WHERE user_id IS NOT NULL"""
    )
    op.execute(
        """CREATE INDEX idx_exams_user_created
           ON exams (user_id, created_at DESC) WHERE user_id IS NOT NULL"""
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_exams_user_created")
    op.execute("DROP INDEX IF EXISTS idx_sessions_user_created")
    op.execute("ALTER TABLE exams DROP COLUMN IF EXISTS user_id")
    op.execute(
        """ALTER TABLE sessions
           DROP COLUMN IF EXISTS claimed_at,
           DROP COLUMN IF EXISTS user_id"""
    )
    op.execute("DROP TABLE IF EXISTS users")
