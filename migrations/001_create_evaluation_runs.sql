CREATE TABLE IF NOT EXISTS evaluation_runs (
    exam_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    models_count SMALLINT NOT NULL CHECK (models_count BETWEEN 1 AND 4),
    questions_requested_per_model INTEGER NOT NULL CHECK (questions_requested_per_model >= 0),
    questions_requested INTEGER NOT NULL CHECK (questions_requested >= 0),
    generated_first INTEGER NOT NULL DEFAULT 0 CHECK (generated_first >= 0),
    missing_first INTEGER NOT NULL DEFAULT 0 CHECK (missing_first >= 0),
    generation_rejected INTEGER NOT NULL DEFAULT 0 CHECK (generation_rejected >= 0),
    shortfall_generated INTEGER NOT NULL DEFAULT 0 CHECK (shortfall_generated >= 0),
    shortfall_still_missing INTEGER NOT NULL DEFAULT 0 CHECK (shortfall_still_missing >= 0),
    validation_total_first INTEGER NOT NULL DEFAULT 0 CHECK (validation_total_first >= 0),
    validation_passed_first INTEGER NOT NULL DEFAULT 0 CHECK (validation_passed_first >= 0),
    validation_failed_first INTEGER NOT NULL DEFAULT 0 CHECK (validation_failed_first >= 0),
    validation_unvalidated_first INTEGER NOT NULL DEFAULT 0 CHECK (validation_unvalidated_first >= 0),
    repair_sent INTEGER NOT NULL DEFAULT 0 CHECK (repair_sent >= 0),
    repair_succeeded INTEGER NOT NULL DEFAULT 0 CHECK (repair_succeeded >= 0),
    repair_failed INTEGER NOT NULL DEFAULT 0 CHECK (repair_failed >= 0),
    final_valid INTEGER NOT NULL DEFAULT 0 CHECK (final_valid >= 0),
    final_invalid INTEGER NOT NULL DEFAULT 0 CHECK (final_invalid >= 0),
    final_unvalidated INTEGER NOT NULL DEFAULT 0 CHECK (final_unvalidated >= 0),
    final_missing INTEGER NOT NULL DEFAULT 0 CHECK (final_missing >= 0),
    status TEXT NOT NULL CHECK (status IN ('healthy', 'needs_attention', 'failed')),
    generation_rejection_reasons JSONB NOT NULL DEFAULT '{}'::jsonb,
    validation_failure_reasons JSONB NOT NULL DEFAULT '{}'::jsonb,
    validator_operational_failures JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_performance JSONB NOT NULL DEFAULT '{}'::jsonb,
    question_type_performance JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS evaluation_runs_created_at_desc_idx
    ON evaluation_runs (created_at DESC);
