CREATE TABLE IF NOT EXISTS idempotency_records (
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    endpoint TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (session_id, endpoint, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idempotency_records_expires_idx
    ON idempotency_records (expires_at);

CREATE TABLE IF NOT EXISTS llm_usage (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    operation TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    cached_tokens INTEGER NOT NULL DEFAULT 0 CHECK (cached_tokens >= 0),
    reasoning_tokens INTEGER NOT NULL DEFAULT 0 CHECK (reasoning_tokens >= 0),
    duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
    estimated_cost_usd NUMERIC(12, 6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS llm_usage_job_created_idx
    ON llm_usage (job_id, created_at DESC);

CREATE TABLE IF NOT EXISTS cleanup_runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS cleanup_single_running_idx
    ON cleanup_runs ((status)) WHERE status = 'running';

-- Generated exams are permanent records. They must survive cleanup or future
-- ownership migration even if their anonymous session, source document, or
-- originating job is later removed.
ALTER TABLE exams ALTER COLUMN session_id DROP NOT NULL;
ALTER TABLE exams ALTER COLUMN document_id DROP NOT NULL;
ALTER TABLE exams ALTER COLUMN job_id DROP NOT NULL;

ALTER TABLE exams DROP CONSTRAINT IF EXISTS exams_session_id_fkey;
ALTER TABLE exams
    ADD CONSTRAINT exams_session_id_fkey
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL;

ALTER TABLE exams DROP CONSTRAINT IF EXISTS exams_document_id_fkey;
ALTER TABLE exams
    ADD CONSTRAINT exams_document_id_fkey
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL;

ALTER TABLE exams DROP CONSTRAINT IF EXISTS exams_job_id_fkey;
ALTER TABLE exams
    ADD CONSTRAINT exams_job_id_fkey
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE SET NULL;
