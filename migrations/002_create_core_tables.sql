CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS sessions_expires_at_idx
    ON sessions (expires_at);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (
        status IN ('uploading', 'queued', 'processing', 'ready', 'failed', 'deleted')
    ),
    original_filename TEXT NOT NULL,
    media_type TEXT NOT NULL DEFAULT 'application/pdf',
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL,
    source_storage_key TEXT NOT NULL,
    structure_storage_key TEXT,
    active_index_job_id UUID,
    stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_code TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS documents_session_created_idx
    ON documents (session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK (type IN ('document_ingestion', 'exam_generation')),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'retrying', 'completed', 'failed', 'cancelled')
    ),
    stage TEXT NOT NULL DEFAULT 'queued',
    progress SMALLINT NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    request_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 1 CHECK (max_attempts >= 1),
    worker_id TEXT,
    heartbeat_at TIMESTAMPTZ,
    error_code TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS jobs_session_created_idx
    ON jobs (session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS jobs_document_idx
    ON jobs (document_id, created_at DESC);

CREATE INDEX IF NOT EXISTS jobs_active_generation_idx
    ON jobs (session_id, created_at DESC)
    WHERE type = 'exam_generation' AND status IN ('queued', 'running', 'retrying');

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'documents_active_index_job_fk'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT documents_active_index_job_fk
            FOREIGN KEY (active_index_job_id) REFERENCES jobs(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS exams (
    id TEXT PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    job_id UUID NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('complete', 'partial')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    exams JSONB NOT NULL,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    evaluation JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS exams_session_created_idx
    ON exams (session_id, created_at DESC);
