"""Add per-run outcome telemetry and durable benchmark execution tables.

Revision ID: 0009_evaluation_analytics_benchmarks
Revises: 0008_llm_request_event_idempotency
"""
from alembic import op


revision = "0009_evaluation_analytics_benchmarks"
down_revision = "0008_llm_request_event_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    statements = (
        "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS job_id UUID",
        "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS duration_ms INTEGER",
        """ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS
               validation_failure_actions JSONB NOT NULL DEFAULT '{}'::jsonb""",
        """ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS
               recovery_outcomes JSONB NOT NULL DEFAULT '{}'::jsonb""",
        """ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS
               duplicate_outcomes JSONB NOT NULL DEFAULT '{}'::jsonb""",
        """ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS
               catalog_cache JSONB NOT NULL DEFAULT '{}'::jsonb""",
        "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS benchmark_id UUID",
        "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS failure_code TEXT",
        "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS failure_message TEXT",
        """ALTER TABLE evaluation_runs
               DROP CONSTRAINT IF EXISTS evaluation_runs_status_check""",
        """ALTER TABLE evaluation_runs
               ADD CONSTRAINT evaluation_runs_status_check
               CHECK (status IN ('healthy', 'needs_attention', 'failed', 'cancelled'))""",
        """CREATE INDEX IF NOT EXISTS evaluation_runs_job_idx
               ON evaluation_runs (job_id)""",
        """CREATE INDEX IF NOT EXISTS evaluation_runs_benchmark_idx
               ON evaluation_runs (benchmark_id)""",
        """CREATE TABLE IF NOT EXISTS evaluation_benchmarks (
               id UUID PRIMARY KEY,
               created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
               updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
               started_at TIMESTAMPTZ,
               finished_at TIMESTAMPTZ,
               status TEXT NOT NULL CHECK (
                   status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
               ),
               label TEXT NOT NULL DEFAULT '',
               config JSONB NOT NULL DEFAULT '{}'::jsonb,
               reproducibility JSONB NOT NULL DEFAULT '{}'::jsonb,
               summary JSONB NOT NULL DEFAULT '{}'::jsonb,
               error_message TEXT,
               owner_session_id UUID,
               owner_user_id UUID,
               idempotency_key TEXT
           )""",
        """CREATE UNIQUE INDEX IF NOT EXISTS evaluation_benchmarks_idempotency_uidx
               ON evaluation_benchmarks (owner_session_id, idempotency_key)
               WHERE idempotency_key IS NOT NULL""",
        """CREATE INDEX IF NOT EXISTS evaluation_benchmarks_owner_created_idx
               ON evaluation_benchmarks (owner_user_id, created_at DESC)""",
        """CREATE TABLE IF NOT EXISTS evaluation_benchmark_runs (
               id UUID PRIMARY KEY,
               benchmark_id UUID NOT NULL
                   REFERENCES evaluation_benchmarks(id) ON DELETE CASCADE,
               run_index INTEGER NOT NULL CHECK (run_index >= 1),
               status TEXT NOT NULL CHECK (
                   status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
               ),
               job_id UUID,
               exam_id TEXT,
               created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
               finished_at TIMESTAMPTZ,
               error_message TEXT,
               UNIQUE (benchmark_id, run_index)
           )""",
        """CREATE INDEX IF NOT EXISTS evaluation_benchmark_runs_job_idx
               ON evaluation_benchmark_runs (job_id)""",
    )
    for statement in statements:
        op.execute(statement)
    # The benchmark foreign key is added after the table exists so this migration
    # also succeeds on databases where only the run table was partially created.
    op.execute(
        """ALTER TABLE evaluation_runs
               DROP CONSTRAINT IF EXISTS evaluation_runs_benchmark_fk"""
    )
    op.execute(
        """ALTER TABLE evaluation_runs
               ADD CONSTRAINT evaluation_runs_benchmark_fk
               FOREIGN KEY (benchmark_id) REFERENCES evaluation_benchmarks(id)
               ON DELETE SET NULL"""
    )


def downgrade() -> None:
    statements = (
        "ALTER TABLE evaluation_runs DROP CONSTRAINT IF EXISTS evaluation_runs_benchmark_fk",
        "DROP TABLE IF EXISTS evaluation_benchmark_runs",
        "DROP TABLE IF EXISTS evaluation_benchmarks",
        "DROP INDEX IF EXISTS evaluation_runs_benchmark_idx",
        "DROP INDEX IF EXISTS evaluation_runs_job_idx",
        "UPDATE evaluation_runs SET status = 'failed' WHERE status = 'cancelled'",
        "ALTER TABLE evaluation_runs DROP CONSTRAINT IF EXISTS evaluation_runs_status_check",
        """ALTER TABLE evaluation_runs
               ADD CONSTRAINT evaluation_runs_status_check
               CHECK (status IN ('healthy', 'needs_attention', 'failed'))""",
        "ALTER TABLE evaluation_runs DROP COLUMN IF EXISTS failure_message",
        "ALTER TABLE evaluation_runs DROP COLUMN IF EXISTS failure_code",
        "ALTER TABLE evaluation_runs DROP COLUMN IF EXISTS benchmark_id",
        "ALTER TABLE evaluation_runs DROP COLUMN IF EXISTS catalog_cache",
        "ALTER TABLE evaluation_runs DROP COLUMN IF EXISTS duplicate_outcomes",
        "ALTER TABLE evaluation_runs DROP COLUMN IF EXISTS recovery_outcomes",
        "ALTER TABLE evaluation_runs DROP COLUMN IF EXISTS validation_failure_actions",
        "ALTER TABLE evaluation_runs DROP COLUMN IF EXISTS duration_ms",
        "ALTER TABLE evaluation_runs DROP COLUMN IF EXISTS job_id",
    )
    for statement in statements:
        op.execute(statement)
