"""Persist compact provider request outcomes for diagnostics.

Revision ID: 0007_llm_request_events
Revises: 0006_job_execution_fencing
"""

from alembic import op


revision = "0007_llm_request_events"
down_revision = "0006_job_execution_fencing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE llm_request_events (
               id BIGSERIAL PRIMARY KEY,
               job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
               provider TEXT NOT NULL,
               model TEXT NOT NULL,
               operation TEXT NOT NULL,
               agent_role TEXT,
               request_id TEXT,
               provider_request_id TEXT,
               status TEXT NOT NULL,
               error_category TEXT,
               provider_error_code TEXT,
               sanitized_error_message TEXT,
               retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
               retry_after_seconds DOUBLE PRECISION,
               usage_available BOOLEAN NOT NULL DEFAULT FALSE,
               input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
               output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
               reasoning_tokens INTEGER CHECK (reasoning_tokens IS NULL OR reasoning_tokens >= 0),
               estimated_cost_usd NUMERIC(18, 10),
               duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
               metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
               created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    op.execute(
        "CREATE INDEX llm_request_events_job_created_idx "
        "ON llm_request_events (job_id, created_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS llm_request_events")
