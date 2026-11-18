"""PostgreSQL persistence and aggregation for exam evaluation telemetry.

The evaluation dashboard reads one durable record per generation run from
``evaluation_runs`` and joins optional provider telemetry from
``llm_request_events`` / ``jobs`` through the run's job id.  All ratios are
computed from stored counters; nothing is estimated except the explicit LLM cost
estimate recorded by the provider client.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app import db
from app.api.repositories import sanitize_jsonb
from app.online.eval_stats import (
    COUNT_FIELDS,
    REASON_FIELDS,
    add_derived_rates,
    derived_rates,
)


RAW_COUNT_FIELDS = tuple(
    field for field in COUNT_FIELDS if field != "final_missing_or_invalid"
)
DB_COUNT_COLUMNS = {
    "requested_questions": "questions_requested",
    **{field: field for field in RAW_COUNT_FIELDS if field != "requested_questions"},
}
REASON_COLUMN_MAP = {
    "generation_rejection_reasons": "r.generation_rejection_reasons",
    "validation_failure_reasons": "r.validation_failure_reasons",
    "validation_failure_actions": "(to_jsonb(r)->'validation_failure_actions')",
    "validator_failure_reasons": "r.validator_operational_failures",
}
REASON_ROW_KEYS = {
    "generation_rejection_reasons": "generation_rejection_reasons",
    "validation_failure_reasons": "validation_failure_reasons",
    "validation_failure_actions": "validation_failure_actions",
    "validator_failure_reasons": "validator_operational_failures",
}

BASE_COLUMNS = (
    "exam_id",
    "models_count",
    "questions_requested_per_model",
    "questions_requested",
    "generated_first",
    "missing_first",
    "generation_rejected",
    "shortfall_generated",
    "shortfall_still_missing",
    "validation_total_first",
    "validation_passed_first",
    "validation_failed_first",
    "validation_unvalidated_first",
    "repair_sent",
    "repair_succeeded",
    "repair_failed",
    "final_valid",
    "final_invalid",
    "final_unvalidated",
    "final_missing",
    "status",
    "generation_rejection_reasons",
    "validation_failure_reasons",
    "validator_operational_failures",
    "model_performance",
    "question_type_performance",
)
EXTENDED_COLUMNS = (
    "job_id",
    "duration_ms",
    "validation_failure_actions",
    "recovery_outcomes",
    "duplicate_outcomes",
    "catalog_cache",
    "benchmark_id",
    "failure_code",
    "failure_message",
)
JSON_COLUMNS = {
    "generation_rejection_reasons",
    "validation_failure_reasons",
    "validator_operational_failures",
    "model_performance",
    "question_type_performance",
    "validation_failure_actions",
    "recovery_outcomes",
    "duplicate_outcomes",
    "catalog_cache",
}
REQUIRED_EXTENDED_COLUMNS = (
    "job_id",
    "recovery_outcomes",
    "duplicate_outcomes",
    "catalog_cache",
    "benchmark_id",
    "validation_failure_actions",
)

_extended_schema: bool | None = None

STATUS_LABELS = {
    "healthy": "Healthy",
    "needs_attention": "Needs attention",
    "failed": "Failed",
    "cancelled": "Cancelled",
}

STAGE_LABELS = {
    "assessment_catalog": "catalog",
    "catalog": "catalog",
    "compiling_assessment_catalog": "catalog",
    "plan": "planning",
    "planning": "planning",
    "planning_exam": "planning",
    "analyzing_request": "request",
    "retrieving_context": "retrieval",
    "difficulty_adjustment": "difficulty_adjustment",
    "adjusting_difficulty": "difficulty_adjustment",
    "model_generation": "generation",
    "generation": "generation",
    "generating_exam": "generation",
    "general_validation": "validation",
    "quantitative_validation": "validation",
    "validation": "validation",
    "validating_exam": "validation",
    "revalidation": "revalidation",
    "validating_updates": "revalidation",
    "repair": "repair",
    "repairing_exam": "repair",
    "duplicate_review": "duplicate",
    "duplicate": "duplicate",
    "retrieval": "retrieval",
    "finalization": "finalization",
    "finalizing_exam": "finalization",
    "queued": "queue",
    "replay": "replay",
}
STAGE_ORDER = (
    "queue",
    "request",
    "retrieval",
    "catalog",
    "planning",
    "difficulty_adjustment",
    "generation",
    "validation",
    "repair",
    "revalidation",
    "duplicate",
    "finalization",
    "replay",
    "unknown",
)


# ---------------------------------------------------------------------------
# Schema compatibility
# ---------------------------------------------------------------------------
def extended_schema_available() -> bool:
    """Return True when migration 0009 columns are present.

    Legacy deployments must keep recording telemetry, so writes fall back to the
    original column set until the migration is applied.  The check is cached but
    re-evaluated while the answer is still negative.
    """
    global _extended_schema
    if _extended_schema:
        return True
    try:
        with db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT COUNT(*) FROM information_schema.columns
                       WHERE table_name = 'evaluation_runs'
                         AND column_name = ANY(%s)""",
                    (list(REQUIRED_EXTENDED_COLUMNS),),
                )
                found = int(cursor.fetchone()[0])
    except Exception:
        return False
    _extended_schema = found == len(REQUIRED_EXTENDED_COLUMNS)
    return _extended_schema


def _run_job_expression() -> str:
    return "COALESCE(r.job_id, e.job_id)" if extended_schema_available() else "e.job_id"


def _benchmark_column() -> str:
    return "r.benchmark_id" if extended_schema_available() else "NULL"


def _nonnegative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _row_get(row: Any, key: str, index: int) -> Any:
    """Read one column from a dict-row or tuple-row cursor result."""
    if isinstance(row, dict):
        return row.get(key)
    return row[index]


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_divide(numerator: Any, denominator: Any, digits: int = 4) -> float | None:
    try:
        numerator = float(numerator or 0)
        denominator = float(denominator or 0)
    except (TypeError, ValueError):
        return None
    if denominator <= 0:
        return None
    return round(numerator / denominator, digits)


def status_label(status: Any) -> str:
    return STATUS_LABELS.get(str(status or "").lower(), str(status or "Unknown"))


def normalize_stage(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return STAGE_LABELS.get(raw, raw or "unknown")


def _raw_json_bucket(bucket: Any) -> dict[str, Any]:
    """Return raw JSON telemetry only (no rates or derived totals)."""
    if not isinstance(bucket, dict):
        return {}
    result: dict[str, Any] = {}
    for field in RAW_COUNT_FIELDS:
        result[field] = _nonnegative_int(bucket.get(field))
    for field in REASON_FIELDS:
        reasons = bucket.get(field) or {}
        result[field] = {
            str(reason): _nonnegative_int(count)
            for reason, count in reasons.items()
            if _nonnegative_int(count)
        } if isinstance(reasons, dict) else {}
    qtypes = bucket.get("question_types") or {}
    if isinstance(qtypes, dict) and qtypes:
        result["question_types"] = {
            str(qtype): _raw_json_bucket(value)
            for qtype, value in qtypes.items()
            if isinstance(value, dict)
        }
    return result


def _empty_bucket() -> dict[str, Any]:
    bucket: dict[str, Any] = {field: 0 for field in COUNT_FIELDS}
    for field in REASON_FIELDS:
        bucket[field] = {}
    bucket["question_types"] = {}
    return bucket


def _bucket_from_row(row: dict[str, Any]) -> dict[str, Any]:
    bucket: dict[str, Any] = {}
    for field in COUNT_FIELDS:
        column = DB_COUNT_COLUMNS.get(field, field)
        bucket[field] = _nonnegative_int(row.get(column))
    for api_field, row_key in REASON_ROW_KEYS.items():
        raw = row.get(row_key) or {}
        bucket[api_field] = {
            str(reason): _nonnegative_int(count)
            for reason, count in raw.items()
            if _nonnegative_int(count)
        } if isinstance(raw, dict) else {}
    bucket["final_missing_or_invalid"] = sum(
        bucket[field] for field in ("final_invalid", "final_unvalidated", "final_missing")
    )
    bucket["question_types"] = {}
    raw_qtypes = row.get("question_type_performance") or {}
    if isinstance(raw_qtypes, dict):
        bucket["question_types"] = {
            str(qtype): _bucket_from_raw(value)
            for qtype, value in raw_qtypes.items()
            if isinstance(value, dict)
        }
    return bucket


def _bucket_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    bucket = {field: _nonnegative_int(raw.get(field)) for field in COUNT_FIELDS}
    for field in REASON_FIELDS:
        reasons = raw.get(field) or {}
        bucket[field] = {
            str(reason): _nonnegative_int(count)
            for reason, count in reasons.items()
            if _nonnegative_int(count)
        } if isinstance(reasons, dict) else {}
    bucket["final_missing_or_invalid"] = sum(
        bucket[field] for field in ("final_invalid", "final_unvalidated", "final_missing")
    )
    bucket["question_types"] = {}
    return bucket


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def evaluation_row(
    exam_id: str,
    eval_stats: dict[str, Any],
    models_count: int,
    questions_requested_per_model: int,
    *,
    job_id: str | None = None,
    duration_ms: int | None = None,
    outcomes: dict[str, Any] | None = None,
    benchmark_id: str | None = None,
    status_override: str | None = None,
    failure_code: str | None = None,
    failure_message: str | None = None,
) -> dict[str, Any]:
    """Transform one completed generation request into one database record."""
    if not 1 <= int(models_count) <= 4:
        raise ValueError("models_count must be between 1 and 4")
    overall = (eval_stats or {}).get("overall") or {}
    requested = int(models_count) * _nonnegative_int(questions_requested_per_model)
    raw = {field: _nonnegative_int(overall.get(field)) for field in RAW_COUNT_FIELDS}
    final_non_valid = sum(
        raw[field] for field in ("final_invalid", "final_unvalidated", "final_missing")
    )
    if status_override:
        status = str(status_override)
    elif requested and raw["final_valid"] == 0 and final_non_valid >= requested:
        status = "failed"
    elif final_non_valid:
        status = "needs_attention"
    else:
        status = "healthy"

    models = (eval_stats or {}).get("models") or {}
    model_json = {
        str(key): _raw_json_bucket(value)
        for key, value in models.items()
        if isinstance(value, dict)
    } if isinstance(models, dict) else {}
    question_types = overall.get("question_types") or {}
    qtype_json = {
        str(key): _raw_json_bucket(value)
        for key, value in question_types.items()
        if isinstance(value, dict)
    } if isinstance(question_types, dict) else {}
    outcomes = outcomes if isinstance(outcomes, dict) else {}

    def _outcome(name: str) -> dict[str, Any]:
        value = outcomes.get(name)
        return value if isinstance(value, dict) else {}

    return {
        "exam_id": str(exam_id),
        "models_count": int(models_count),
        "questions_requested_per_model": _nonnegative_int(questions_requested_per_model),
        "questions_requested": requested,
        **{field: raw[field] for field in RAW_COUNT_FIELDS if field != "requested_questions"},
        "status": status,
        "generation_rejection_reasons": _raw_json_bucket(overall)["generation_rejection_reasons"],
        "validation_failure_reasons": _raw_json_bucket(overall)["validation_failure_reasons"],
        "validator_operational_failures": _raw_json_bucket(overall)["validator_failure_reasons"],
        "model_performance": model_json,
        "question_type_performance": qtype_json,
        "job_id": str(job_id) if job_id else None,
        "duration_ms": max(int(duration_ms), 0) if duration_ms is not None else None,
        "validation_failure_actions": _raw_json_bucket(overall)["validation_failure_actions"],
        "recovery_outcomes": _outcome("recovery"),
        "duplicate_outcomes": _outcome("duplicates"),
        "catalog_cache": _outcome("catalog_cache"),
        "benchmark_id": str(benchmark_id) if benchmark_id else None,
        "failure_code": str(failure_code)[:120] if failure_code else None,
        "failure_message": str(failure_message)[:1000] if failure_message else None,
    }


def insert_evaluation_run(row: dict[str, Any]) -> None:
    """Insert or update one run record.

    ``exam_id`` is the primary key, so a retried or resumed job that re-records
    telemetry updates the existing row instead of failing a duplicate-key insert.
    """
    extended = extended_schema_available()
    if row.get("benchmark_id") and not extended:
        raise RuntimeError(
            "Benchmark telemetry requires the evaluation analytics migration"
        )
    columns = list(BASE_COLUMNS)
    if extended:
        columns.extend(EXTENDED_COLUMNS)
    values = [
        Jsonb(sanitize_jsonb(row.get(column) or {}))
        if column in JSON_COLUMNS
        else row.get(column)
        for column in columns
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    assignments = ", ".join(
        f"{column} = EXCLUDED.{column}" for column in columns if column != "exam_id"
    )
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""INSERT INTO evaluation_runs ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT (exam_id) DO UPDATE SET {assignments}""",
                values,
            )
        conn.commit()


def save_evaluation(
    exam_id: str,
    eval_stats: dict[str, Any],
    models_count: int,
    questions_requested_per_model: int,
    *,
    job_id: str | None = None,
    duration_ms: int | None = None,
    outcomes: dict[str, Any] | None = None,
    benchmark_id: str | None = None,
) -> None:
    insert_evaluation_run(
        evaluation_row(
            exam_id,
            eval_stats,
            models_count,
            questions_requested_per_model,
            job_id=job_id,
            duration_ms=duration_ms,
            outcomes=outcomes,
            benchmark_id=benchmark_id,
        )
    )


def save_terminal_failure(
    exam_id: str,
    *,
    models_count: int,
    questions_requested_per_model: int,
    status: str,
    job_id: str | None = None,
    duration_ms: int | None = None,
    failure_code: str | None = None,
    failure_message: str | None = None,
    benchmark_id: str | None = None,
) -> None:
    """Record a failed or cancelled run so reliability is measurable per run.

    Every requested question is counted as missing because the run produced no
    accepted output.  This row is replaced by the normal evaluation record if the
    same exam later completes successfully.
    """
    requested = int(models_count) * _nonnegative_int(questions_requested_per_model)
    stats = {
        "overall": {
            "requested_questions": requested,
            "final_missing": requested,
        },
        "models": {},
    }
    insert_evaluation_run(
        evaluation_row(
            exam_id,
            stats,
            models_count,
            questions_requested_per_model,
            job_id=job_id,
            duration_ms=duration_ms,
            status_override=status,
            failure_code=failure_code,
            failure_message=failure_message,
            benchmark_id=benchmark_id,
        )
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _sum_select(prefix: str = "", *, json_bucket: bool = False) -> str:
    return ", ".join(
        (
            f"COALESCE(SUM(({prefix}'{field}')::bigint), 0)::bigint AS {field}"
            if json_bucket else
            f"COALESCE(SUM(({column})::bigint), 0)::bigint AS {field}"
        )
        for field, column in DB_COUNT_COLUMNS.items()
    )


def _jsonb_int_sum(path: str) -> str:
    return f"COALESCE(SUM(NULLIF((to_jsonb(r){path}), '')::bigint), 0)::bigint"


def _run_filter(
    days: int | None,
    include_benchmarks: bool,
    *,
    alias: str = "r",
    extra: str | None = None,
    extra_params: Iterable[Any] = (),
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if days:
        clauses.append(
            f"{alias}.created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')"
        )
        params.append(int(days))
    if not include_benchmarks:
        clauses.append(f"{_benchmark_column()} IS NULL")
    if extra:
        clauses.append(extra)
        params.extend(extra_params)
    return ((" WHERE " + " AND ".join(clauses)) if clauses else ""), params


def _fetch_reason_counts(
    cursor: Any, expression: str, where: str = "", params: Iterable[Any] = ()
) -> dict[str, int]:
    cursor.execute(
        f"""SELECT reason.key, SUM((reason.value)::bigint)::bigint AS count
            FROM evaluation_runs r
            CROSS JOIN LATERAL jsonb_each_text(
                COALESCE({expression}, '{{}}'::jsonb)
            ) AS reason(key, value)
            {where}
            GROUP BY reason.key ORDER BY count DESC, reason.key""",
        tuple(params),
    )
    return {
        str(_row_get(row, "key", 0)): int(_row_get(row, "count", 1))
        for row in cursor.fetchall()
    }


def _fetch_jsonb_map_totals(
    cursor: Any, path: str, where: str = "", params: Iterable[Any] = ()
) -> dict[str, int]:
    cursor.execute(
        f"""SELECT entry.key, SUM((entry.value)::bigint)::bigint AS count
            FROM evaluation_runs r
            CROSS JOIN LATERAL jsonb_each_text(
                COALESCE((to_jsonb(r){path}), '{{}}'::jsonb)
            ) AS entry(key, value)
            {where}
            GROUP BY entry.key ORDER BY count DESC, entry.key""",
        tuple(params),
    )
    return {
        str(_row_get(row, "key", 0)): int(_row_get(row, "count", 1))
        for row in cursor.fetchall()
    }


def _fetch_jsonb_nested_totals(
    cursor: Any, path: str, where: str = "", params: Iterable[Any] = ()
) -> dict[str, dict[str, int]]:
    cursor.execute(
        f"""SELECT outer_entry.key AS outer_key, inner_entry.key AS inner_key,
                   SUM((inner_entry.value)::bigint)::bigint AS count
            FROM evaluation_runs r
            CROSS JOIN LATERAL jsonb_each(
                COALESCE((to_jsonb(r){path}), '{{}}'::jsonb)
            ) AS outer_entry(key, value)
            CROSS JOIN LATERAL jsonb_each_text(
                COALESCE(outer_entry.value, '{{}}'::jsonb)
            ) AS inner_entry(key, value)
            {where}
            GROUP BY outer_entry.key, inner_entry.key""",
        tuple(params),
    )
    totals: dict[str, dict[str, int]] = {}
    for row in cursor.fetchall():
        outer_key = str(_row_get(row, "outer_key", 0))
        inner_key = str(_row_get(row, "inner_key", 1))
        totals.setdefault(outer_key, {})[inner_key] = int(
            _row_get(row, "count", 2) or 0
        )
    return totals


def _fetch_dimension(
    cursor: Any, expression: str, where: str = "", params: Iterable[Any] = ()
) -> dict[str, dict[str, Any]]:
    cursor.execute(
        f"""SELECT dimension.key, {_sum_select("dimension.value->>", json_bucket=True)}
            FROM evaluation_runs r
            CROSS JOIN LATERAL jsonb_each({expression}) AS dimension(key, value)
            {where}
            GROUP BY dimension.key ORDER BY dimension.key""",
        tuple(params),
    )
    buckets: dict[str, dict[str, Any]] = {}
    for row in cursor.fetchall():
        bucket = _empty_bucket()
        for index, field in enumerate(DB_COUNT_COLUMNS, start=1):
            bucket[field] = int(_row_get(row, field, index) or 0)
        buckets[str(_row_get(row, "key", 0))] = bucket
    return buckets


def _add_dimension_reasons(
    cursor: Any,
    buckets: dict[str, dict[str, Any]],
    expression: str,
    where: str = "",
    params: Iterable[Any] = (),
) -> None:
    for api_field in REASON_FIELDS:
        cursor.execute(
            f"""SELECT dimension.key AS dimension_key, reason.key AS reason_key,
                       SUM((reason.value)::bigint)::bigint AS count
                FROM evaluation_runs r
                CROSS JOIN LATERAL jsonb_each({expression}) AS dimension(key, value)
                CROSS JOIN LATERAL jsonb_each_text(
                    COALESCE(dimension.value->'{api_field}', '{{}}'::jsonb)
                ) AS reason(key, value)
                {where}
                GROUP BY dimension.key, reason.key""",
            tuple(params),
        )
        for row in cursor.fetchall():
            dimension_key = str(_row_get(row, "dimension_key", 0))
            reason = str(_row_get(row, "reason_key", 1))
            count = int(_row_get(row, "count", 2) or 0)
            if dimension_key in buckets:
                buckets[dimension_key][api_field][reason] = count


def _add_model_question_types(cursor: Any, models: dict[str, dict[str, Any]]) -> None:
    cursor.execute(
        f"""SELECT model.key AS model_key, qtype.key AS question_type,
                   {_sum_select("qtype.value->>", json_bucket=True)}
            FROM evaluation_runs r
            CROSS JOIN LATERAL jsonb_each(r.model_performance) AS model(key, value)
            CROSS JOIN LATERAL jsonb_each(
                COALESCE(model.value->'question_types', '{{}}'::jsonb)
            ) AS qtype(key, value)
            GROUP BY model.key, qtype.key ORDER BY model.key, qtype.key"""
    )
    for row in cursor.fetchall():
        bucket = _empty_bucket()
        for index, field in enumerate(DB_COUNT_COLUMNS, start=2):
            bucket[field] = int(_row_get(row, field, index) or 0)
        model_key = str(_row_get(row, "model_key", 0))
        if model_key in models:
            models[model_key]["question_types"][
                str(_row_get(row, "question_type", 1))
            ] = bucket

    for api_field in REASON_FIELDS:
        cursor.execute(
            f"""SELECT model.key AS model_key, qtype.key AS question_type,
                       reason.key AS reason_key, SUM((reason.value)::bigint)::bigint AS count
                FROM evaluation_runs r
                CROSS JOIN LATERAL jsonb_each(r.model_performance) AS model(key, value)
                CROSS JOIN LATERAL jsonb_each(
                    COALESCE(model.value->'question_types', '{{}}'::jsonb)
                ) AS qtype(key, value)
                CROSS JOIN LATERAL jsonb_each_text(
                    COALESCE(qtype.value->'{api_field}', '{{}}'::jsonb)
                ) AS reason(key, value)
                GROUP BY model.key, qtype.key, reason.key"""
        )
        for row in cursor.fetchall():
            model_key = str(_row_get(row, "model_key", 0))
            if model_key not in models:
                continue
            models[model_key]["question_types"].setdefault(
                str(_row_get(row, "question_type", 1)), _empty_bucket()
            )[api_field][str(_row_get(row, "reason_key", 2))] = int(
                _row_get(row, "count", 3) or 0
            )


def _llm_aggregate_columns(prefix: str = "e.") -> str:
    return f"""
        COUNT(*)::bigint AS calls,
        COUNT(*) FILTER (WHERE {prefix}status = 'success')::bigint AS successful,
        COUNT(*) FILTER (WHERE {prefix}status <> 'success')::bigint AS failed,
        COALESCE(SUM({prefix}retry_count), 0)::bigint AS retries,
        COUNT({prefix}input_tokens)::bigint AS usage_calls,
        COALESCE(SUM({prefix}input_tokens), 0)::bigint AS input_tokens,
        COALESCE(SUM({prefix}output_tokens), 0)::bigint AS output_tokens,
        COALESCE(SUM({prefix}reasoning_tokens), 0)::bigint AS reasoning_tokens,
        COALESCE(SUM(NULLIF({prefix}metadata->>'cached_tokens', '')::bigint), 0)::bigint
            AS cached_tokens,
        COUNT({prefix}estimated_cost_usd)::bigint AS cost_calls,
        COALESCE(SUM({prefix}estimated_cost_usd), 0) AS estimated_cost_usd,
        COUNT({prefix}duration_ms)::bigint AS latency_calls,
        COALESCE(SUM({prefix}duration_ms), 0) AS call_time_ms,
        COALESCE(AVG({prefix}duration_ms), 0) AS avg_latency_ms,
        COALESCE(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {prefix}duration_ms), 0)
            AS p50_latency_ms,
        COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {prefix}duration_ms), 0)
            AS p95_latency_ms
    """


def _llm_where(
    days: int | None, include_benchmarks: bool
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if days:
        clauses.append("e.created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')")
        params.append(int(days))
    if not include_benchmarks:
        clauses.append("(j.id IS NULL OR j.request_data->>'benchmark_id' IS NULL)")
    return ((" WHERE " + " AND ".join(clauses)) if clauses else ""), params


def _llm_from() -> str:
    return "FROM llm_request_events e LEFT JOIN jobs j ON j.id = e.job_id"


def _llm_totals(cursor: Any, where: str, params: Iterable[Any]) -> dict[str, Any]:
    cursor.execute(
        f"""SELECT {_llm_aggregate_columns()} {_llm_from()} {where}""",
        tuple(params),
    )
    row = cursor.fetchone() or {}
    return _llm_payload(row)


def _llm_payload(row: dict[str, Any]) -> dict[str, Any]:
    calls = int(row.get("calls") or 0)
    latency_calls = int(row.get("latency_calls") or 0)
    usage_calls = int(row.get("usage_calls") or 0)
    cost_calls = int(row.get("cost_calls") or 0)
    return {
        "calls": calls,
        "successful": int(row.get("successful") or 0),
        "failed": int(row.get("failed") or 0),
        "failure_rate": _safe_divide(row.get("failed"), calls),
        "retries": int(row.get("retries") or 0),
        "usage_calls": usage_calls,
        "usage_coverage": _safe_divide(usage_calls, calls),
        "input_tokens": int(row.get("input_tokens") or 0),
        "output_tokens": int(row.get("output_tokens") or 0),
        "reasoning_tokens": int(row.get("reasoning_tokens") or 0),
        "cached_tokens": int(row.get("cached_tokens") or 0),
        "cost_calls": cost_calls,
        "estimated_cost_usd": (
            _number(row.get("estimated_cost_usd")) if cost_calls else None
        ),
        "call_time_ms": (
            round(_number(row.get("call_time_ms")) or 0, 1)
            if latency_calls else None
        ),
        "latency": {
            "calls": latency_calls,
            "avg_ms": round(_number(row.get("avg_latency_ms")) or 0, 1)
            if latency_calls else None,
            "p50_ms": round(_number(row.get("p50_latency_ms")) or 0, 1)
            if latency_calls else None,
            "p95_ms": round(_number(row.get("p95_latency_ms")) or 0, 1)
            if latency_calls else None,
        },
    }


def _llm_groups(
    cursor: Any,
    key_expression: str,
    where: str,
    params: Iterable[Any],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    cursor.execute(
        f"""SELECT {key_expression} AS key, {_llm_aggregate_columns()}
            {_llm_from()} {where}
            GROUP BY 1 ORDER BY calls DESC, key
            {f'LIMIT {int(limit)}' if limit else ''}""",
        tuple(params),
    )
    groups = []
    for row in cursor.fetchall():
        payload = _llm_payload(row)
        payload["key"] = str(row.get("key") or "unknown")
        groups.append(payload)
    return groups


def _llm_scoped_totals(
    cursor: Any, from_sql: str, where: str, params: Iterable[Any]
) -> dict[str, Any]:
    cursor.execute(
        f"""SELECT {_llm_aggregate_columns()} {from_sql} {where}""",
        tuple(params),
    )
    return _llm_payload(cursor.fetchone() or {})


def _llm_population(
    cursor: Any, days: int | None, include_benchmarks: bool
) -> dict[str, Any]:
    """Break provider telemetry into attribution scopes.

    ``evaluated`` covers only jobs that have an evaluation record;
    it is the only scope used for per-run averages.  ``other_generation`` covers
    failed/cancelled jobs that still spent tokens, ``unattributed`` covers events
    without a resolvable job (legacy or removed jobs), and ``all`` is everything
    recorded.  These scopes
    never overlap, so a cost is counted exactly once.
    """
    all_where, all_params = _llm_where(days, include_benchmarks)
    all_totals = _llm_totals(cursor, all_where, all_params)

    event_clause = (
        "e.created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')"
        if days else "TRUE"
    )
    event_params = [int(days)] if days else []

    run_clauses: list[str] = []
    run_params: list[Any] = []
    if days:
        run_clauses.append(
            "r.created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')"
        )
        run_params.append(int(days))
    if not include_benchmarks:
        run_clauses.append(f"{_benchmark_column()} IS NULL")
    run_where = " AND ".join(run_clauses) or "TRUE"
    # Attribution scopes follow the event window.  A run whose job emitted
    # requests in the window belongs to the evaluated scope even if the run
    # record itself was created a moment earlier.
    scope_clauses = [] if include_benchmarks else [f"{_benchmark_column()} IS NULL"]
    scope_where = " AND ".join(scope_clauses) or "TRUE"

    evaluated = _llm_scoped_totals(
        cursor,
        """FROM llm_request_events e
           JOIN exams ex ON ex.job_id = e.job_id
           JOIN evaluation_runs r ON r.exam_id = ex.id""",
        f" WHERE {event_clause} AND {scope_where}",
        event_params,
    )

    cursor.execute(
        f"""SELECT COUNT(*)::bigint AS linked_runs,
                   COUNT(*) FILTER (WHERE ex.job_id IN (
                       SELECT DISTINCT e.job_id FROM llm_request_events e
                       WHERE e.job_id IS NOT NULL AND {event_clause}
                   ))::bigint AS runs_with_calls,
                   COALESCE(SUM(r.models_count) FILTER (WHERE ex.job_id IN (
                       SELECT DISTINCT e.job_id FROM llm_request_events e
                       WHERE e.job_id IS NOT NULL AND {event_clause}
                   )), 0)::bigint AS models_with_calls,
                   COALESCE(SUM(r.final_valid) FILTER (WHERE ex.job_id IN (
                       SELECT DISTINCT e.job_id FROM llm_request_events e
                       WHERE e.job_id IS NOT NULL AND {event_clause}
                   )), 0)::bigint AS final_valid_with_calls
            FROM evaluation_runs r
            JOIN exams ex ON ex.id = r.exam_id
            WHERE {scope_where} AND ex.job_id IS NOT NULL""",
        [*event_params, *event_params, *event_params],
    )
    population = cursor.fetchone() or {}
    cursor.execute(
        f"SELECT COUNT(*)::bigint AS records FROM evaluation_runs r WHERE {run_where}",
        run_params,
    )
    records = int((cursor.fetchone() or {}).get("records") or 0)

    other_clauses = [
        event_clause,
        "j.type = 'exam_generation'",
        """NOT EXISTS (
               SELECT 1 FROM exams ex2
               JOIN evaluation_runs r2 ON r2.exam_id = ex2.id
               WHERE ex2.job_id = j.id
           )""",
    ]
    if not include_benchmarks:
        other_clauses.append("(j.request_data->>'benchmark_id' IS NULL)")
    other_generation = _llm_scoped_totals(
        cursor,
        "FROM llm_request_events e JOIN jobs j ON j.id = e.job_id",
        " WHERE " + " AND ".join(other_clauses),
        event_params,
    )

    other_jobs = _llm_scoped_totals(
        cursor,
        "FROM llm_request_events e JOIN jobs j ON j.id = e.job_id",
        f" WHERE {event_clause} AND j.type <> 'exam_generation'",
        event_params,
    )
    unattributed = _llm_scoped_totals(
        cursor,
        "FROM llm_request_events e LEFT JOIN jobs j ON j.id = e.job_id",
        f" WHERE {event_clause} AND j.id IS NULL",
        event_params,
    )

    return {
        "all": all_totals,
        "evaluated": evaluated,
        "other_generation": other_generation,
        "other_jobs": other_jobs,
        "unattributed": unattributed,
        "population": {
            "evaluation_records": records,
            "linked_runs": int(population.get("linked_runs") or 0),
            "runs_with_calls": int(population.get("runs_with_calls") or 0),
            "models_with_calls": int(population.get("models_with_calls") or 0),
            "final_valid_with_calls": int(
                population.get("final_valid_with_calls") or 0
            ),
        },
    }


def _duration_expression(alias: str = "r", job_alias: str = "j") -> str:
    """Recorded wall time: stored duration, job timestamps, or audit span."""
    parts = [f"NULLIF(to_jsonb({alias})->>'duration_ms', '')::bigint"]
    if job_alias:
        parts.append(
            f"CASE WHEN {job_alias}.started_at IS NOT NULL "
            f"AND {job_alias}.finished_at IS NOT NULL "
            f"THEN (EXTRACT(EPOCH FROM ({job_alias}.finished_at - "
            f"{job_alias}.started_at)) * 1000)::bigint END"
        )
    parts.append(
        f"""(SELECT (EXTRACT(EPOCH FROM (MAX(a.created_at) - MIN(a.created_at))) * 1000)::bigint
             FROM exam_audit_events a
             WHERE a.exam_id = {alias}.exam_id
             HAVING COUNT(*) >= 2)"""
    )
    return "COALESCE(" + ", ".join(parts) + ")"


def _request_statistics(
    cursor: Any,
    days: int | None,
    include_benchmarks: bool,
    jobs: dict[str, Any],
) -> dict[str, Any]:
    """Count known generation requests, model exams and question shortfall.

    The population is the union of evaluation records and tracked generation
    jobs that never produced an evaluation record (failed/cancelled jobs).  Each
    request is counted once: a job whose exam has an evaluation row is represented
    by that row, not by the job again.
    """
    eval_where, eval_params = _run_filter(days, include_benchmarks)
    cursor.execute(
        f"""SELECT COUNT(*)::bigint AS records,
                   COUNT(*) FILTER (WHERE r.status IN ('healthy', 'needs_attention'))
                       AS completed,
                   COUNT(*) FILTER (WHERE r.status = 'failed') AS failed,
                   COUNT(*) FILTER (WHERE r.status = 'cancelled') AS cancelled,
                   COALESCE(SUM(r.models_count), 0)::bigint AS models_requested,
                   COUNT(*) FILTER (WHERE e.id IS NULL)::bigint AS records_without_exam,
                   COALESCE(SUM(r.models_count) FILTER (WHERE e.id IS NULL), 0)::bigint
                       AS models_unavailable,
                   COALESCE(SUM(
                       CASE WHEN e.id IS NOT NULL
                                  AND jsonb_typeof(e.exams) = 'array'
                            THEN jsonb_array_length(e.exams)
                            ELSE 0 END
                   ), 0)::bigint AS models_produced,
                   COALESCE(SUM(r.questions_requested), 0)::bigint AS questions_requested,
                   COALESCE(SUM(r.final_valid), 0)::bigint AS questions_finalized,
                   COUNT(*) FILTER (WHERE e.job_id IS NULL)::bigint AS historical_without_job
            FROM evaluation_runs r
            LEFT JOIN exams e ON e.id = r.exam_id
            {eval_where}""",
        eval_params,
    )
    eval_row = cursor.fetchone() or {}

    job_clauses = [
        "j.type = 'exam_generation'",
        """NOT EXISTS (
               SELECT 1 FROM exams e2
               JOIN evaluation_runs r2 ON r2.exam_id = e2.id
               WHERE e2.job_id = j.id
           )""",
    ]
    job_params: list[Any] = []
    if days:
        job_clauses.append(
            "j.created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')"
        )
        job_params.append(int(days))
    if not include_benchmarks:
        job_clauses.append("(j.request_data->>'benchmark_id' IS NULL)")
    job_where = " WHERE " + " AND ".join(job_clauses)
    cursor.execute(
        f"""SELECT j.status,
                   COALESCE((j.request_data->>'num_models')::int, 1) AS models,
                   COALESCE((
                       SELECT SUM((task->>1)::int)
                       FROM jsonb_array_elements(
                           COALESCE(j.request_data->'tasks', '[]'::jsonb)
                       ) AS task
                   ), 0)::bigint AS questions_per_model,
                   CASE WHEN e.id IS NULL THEN NULL
                        WHEN jsonb_typeof(e.exams) = 'array'
                        THEN jsonb_array_length(e.exams)
                        ELSE 0 END AS produced,
                   e.id IS NULL AS no_exam
            FROM jobs j
            LEFT JOIN exams e ON e.job_id = j.id
            {job_where}""",
        job_params,
    )
    unlinked_rows = cursor.fetchall()

    job_completed = job_failed = job_cancelled = job_active = 0
    job_models = job_models_produced = job_models_unavailable = 0
    job_questions_terminal = job_questions_active = 0
    job_questions_failed = job_questions_unknown = 0
    for row in unlinked_rows:
        status = str(row.get("status") or "")
        models = _nonnegative_int(row.get("models")) or 1
        questions = _nonnegative_int(row.get("questions_per_model")) * models
        produced = row.get("produced")
        if status == "completed":
            job_completed += 1
        elif status == "failed":
            job_failed += 1
        elif status == "cancelled":
            job_cancelled += 1
        else:
            job_active += 1
        if status not in {"completed", "failed", "cancelled"}:
            job_questions_active += questions
            continue
        job_models += models
        job_questions_terminal += questions
        if status in {"failed", "cancelled"}:
            job_questions_failed += questions
        else:
            # No evaluation record means the finalized-question count is not
            # recorded even when a partial exam payload exists.
            job_questions_unknown += questions
        if produced is None:
            job_models_unavailable += models
        else:
            job_models_produced += int(produced or 0)

    completed = int(eval_row.get("completed") or 0) + job_completed
    failed = int(eval_row.get("failed") or 0) + job_failed
    cancelled = int(eval_row.get("cancelled") or 0) + job_cancelled
    active = job_active
    finished = completed + failed + cancelled
    questions_requested = (
        int(eval_row.get("questions_requested") or 0) + job_questions_terminal
    )
    questions_finalized = int(eval_row.get("questions_finalized") or 0)
    return {
        "total": completed + failed + cancelled + active,
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "active": active,
        "finished": finished,
        "completion_rate": _safe_divide(completed, finished),
        "evaluation_records": int(eval_row.get("records") or 0),
        "historical_without_job": int(eval_row.get("historical_without_job") or 0),
        "jobs_without_evaluation": len(unlinked_rows),
        "tracked_jobs": int(jobs.get("total") or 0),
        "tracked_job_completed": int(jobs.get("completed") or 0),
        "tracked_job_failed": int(jobs.get("failed") or 0),
        "tracked_job_cancelled": int(jobs.get("cancelled") or 0),
        "tracked_job_active": int(jobs.get("active") or 0),
        "model_exams_requested": int(eval_row.get("models_requested") or 0) + job_models,
        "model_exams_produced": int(eval_row.get("models_produced") or 0)
        + job_models_produced,
        "model_exams_unavailable": int(eval_row.get("models_unavailable") or 0)
        + job_models_unavailable,
        "model_exams_not_produced": max(
            int(eval_row.get("models_requested") or 0)
            + job_models
            - int(eval_row.get("models_produced") or 0)
            - job_models_produced
            - int(eval_row.get("models_unavailable") or 0)
            - job_models_unavailable,
            0,
        ),
        "evaluation_records_without_exam": int(
            eval_row.get("records_without_exam") or 0
        ),
        "questions_requested": questions_requested,
        "questions_requested_evaluated": int(
            eval_row.get("questions_requested") or 0
        ),
        "questions_requested_failed_jobs": job_questions_failed,
        "questions_finalized": questions_finalized,
        "questions_unavailable": job_questions_unknown,
        "question_shortfall": max(
            questions_requested - questions_finalized - job_questions_unknown, 0
        ),
        "active_questions": job_questions_active,
    }


def _recent_run_rows(
    cursor: Any, limit: int = 20, where: str = "", params: Iterable[Any] = ()
) -> list[dict[str, Any]]:
    job_expression = _run_job_expression()
    cursor.execute(
        f"""SELECT r.exam_id, r.created_at, r.status, r.models_count,
                   r.questions_requested, r.validation_total_first,
                   r.validation_passed_first, r.repair_sent, r.repair_succeeded,
                   r.final_valid, r.final_invalid, r.final_unvalidated, r.final_missing,
                   {_duration_expression()} AS duration_ms,
                   to_jsonb(r)->>'failure_code' AS failure_code,
                   {job_expression} AS job_id,
                   j.status AS job_status,
                   COALESCE((to_jsonb(r)->'recovery_outcomes'->>'entered_recovery')::bigint, 0)
                       AS recovery_entered,
                   COALESCE((to_jsonb(r)->'recovery_outcomes'->>'recovered')::bigint, 0)
                       AS recovery_recovered,
                   COALESCE((to_jsonb(r)->'recovery_outcomes'->>'unresolved')::bigint, 0)
                       AS recovery_unresolved,
                   COALESCE(to_jsonb(r)->'catalog_cache', '{{}}'::jsonb) <> '{{}}'::jsonb
                       AS has_cache_telemetry,
                   COALESCE(to_jsonb(r)->'recovery_outcomes', '{{}}'::jsonb) <> '{{}}'::jsonb
                       AS has_recovery_telemetry,
                   u.calls, u.failed_calls, u.input_tokens, u.output_tokens,
                   u.reasoning_tokens, u.cached_tokens, u.cost_usd, u.avg_latency_ms
            FROM evaluation_runs r
            LEFT JOIN exams e ON e.id = r.exam_id
            LEFT JOIN jobs j ON j.id = {job_expression}
            LEFT JOIN LATERAL (
                SELECT COUNT(*)::bigint AS calls,
                       COUNT(*) FILTER (WHERE l.status <> 'success')::bigint AS failed_calls,
                       COALESCE(SUM(l.input_tokens), 0)::bigint AS input_tokens,
                       COALESCE(SUM(l.output_tokens), 0)::bigint AS output_tokens,
                       COALESCE(SUM(l.reasoning_tokens), 0)::bigint AS reasoning_tokens,
                       COALESCE(SUM(NULLIF(l.metadata->>'cached_tokens', '')::bigint), 0)::bigint
                           AS cached_tokens,
                       SUM(l.estimated_cost_usd) AS cost_usd,
                       AVG(l.duration_ms) AS avg_latency_ms
                FROM llm_request_events l
                WHERE l.job_id = {job_expression}
            ) u ON TRUE
            {where}
            ORDER BY r.created_at DESC, r.exam_id DESC
            LIMIT %s""",
        (*tuple(params), int(limit)),
    )
    rows = []
    for row in cursor.fetchall():
        requested = _nonnegative_int(row.get("questions_requested"))
        final_valid = _nonnegative_int(row.get("final_valid"))
        first_passed = _nonnegative_int(row.get("validation_passed_first"))
        first_total = _nonnegative_int(row.get("validation_total_first"))
        repair_sent = _nonnegative_int(row.get("repair_sent"))
        repair_succeeded = _nonnegative_int(row.get("repair_succeeded"))
        cost = _number(row.get("cost_usd"))
        calls = int(row.get("calls") or 0)
        created = row.get("created_at")
        rows.append({
            "exam_id": str(row["exam_id"]),
            "created_at": created.isoformat() if created else None,
            "job_id": str(row["job_id"]) if row.get("job_id") else None,
            "status": status_label(row.get("status")),
            "status_key": str(row.get("status") or ""),
            "job_status": str(row.get("job_status") or "") or None,
            "models_count": _nonnegative_int(row.get("models_count")),
            "requested": requested,
            "final_valid": final_valid,
            "first_pass_rate": _safe_divide(first_passed, first_total),
            "validation_coverage": _safe_divide(first_total, requested),
            "final_acceptance_rate": _safe_divide(final_valid, requested),
            "repairs_sent": repair_sent,
            "repairs_succeeded": repair_succeeded,
            "repair_success_rate": _safe_divide(repair_succeeded, repair_sent),
            "recovery_entered": _nonnegative_int(row.get("recovery_entered")),
            "recovery_recovered": _nonnegative_int(row.get("recovery_recovered")),
            "recovery_unresolved": _nonnegative_int(row.get("recovery_unresolved")),
            "duration_ms": (
                int(row["duration_ms"]) if row.get("duration_ms") is not None else None
            ),
            "failure_code": row.get("failure_code"),
            "has_llm_telemetry": calls > 0,
            "has_recovery_telemetry": bool(row.get("has_recovery_telemetry")),
            "llm": {
                "calls": calls,
                "failed_calls": int(row.get("failed_calls") or 0),
                "input_tokens": int(row.get("input_tokens") or 0),
                "output_tokens": int(row.get("output_tokens") or 0),
                "reasoning_tokens": int(row.get("reasoning_tokens") or 0),
                "cached_tokens": int(row.get("cached_tokens") or 0),
                "estimated_cost_usd": cost if calls else None,
                "avg_latency_ms": (
                    round(_number(row.get("avg_latency_ms")) or 0, 1)
                    if row.get("avg_latency_ms") is not None else None
                ),
            },
            "cost_per_finalized_question_usd": (
                round(cost / final_valid, 6) if cost is not None and final_valid else None
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Legacy dashboard contract (/api/v1/eval-summary)
# ---------------------------------------------------------------------------
def load_eval_summary() -> dict[str, Any]:
    """Aggregate persisted runs while retaining the original API contract."""
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                f"""SELECT COUNT(*)::bigint AS generation_runs,
                           COALESCE(SUM(r.models_count), 0)::bigint AS model_exams,
                           {_sum_select()}
                    FROM evaluation_runs r"""
            )
            totals = cursor.fetchone() or {}
            overall = _empty_bucket()
            generation_runs = int(totals.get("generation_runs") or 0)
            model_exams = int(totals.get("model_exams") or 0)
            for field in DB_COUNT_COLUMNS:
                overall[field] = int(totals.get(field) or 0)
            overall["final_missing_or_invalid"] = sum(
                overall[field]
                for field in ("final_invalid", "final_unvalidated", "final_missing")
            )
            for api_field, expression in REASON_COLUMN_MAP.items():
                overall[api_field] = _fetch_reason_counts(cursor, expression)

            overall["question_types"] = _fetch_dimension(
                cursor, "r.question_type_performance"
            )
            _add_dimension_reasons(
                cursor, overall["question_types"], "r.question_type_performance"
            )
            for qtype in overall["question_types"].values():
                qtype["final_missing_or_invalid"] = sum(
                    qtype[field]
                    for field in ("final_invalid", "final_unvalidated", "final_missing")
                )

            models = _fetch_dimension(cursor, "r.model_performance")
            _add_dimension_reasons(cursor, models, "r.model_performance")
            _add_model_question_types(cursor, models)
            for model in models.values():
                model["final_missing_or_invalid"] = sum(
                    model[field] for field in ("final_invalid", "final_unvalidated", "final_missing")
                )
                for qtype in model["question_types"].values():
                    qtype["final_missing_or_invalid"] = sum(
                        qtype[field] for field in ("final_invalid", "final_unvalidated", "final_missing")
                    )

            cursor.execute(
                "SELECT MAX(created_at) AS latest FROM evaluation_runs"
            )
            latest = cursor.fetchone() or {}
            recent_rows = _recent_run_rows(cursor, 20)

    overall_with_rates = add_derived_rates(overall)
    return {
        "total_exam_runs": generation_runs,
        "total_generation_runs": generation_runs,
        "total_model_exams": model_exams,
        "latest_run_at": latest["latest"].isoformat() if latest.get("latest") else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall_with_rates,
        "rates": overall_with_rates["rates"],
        "question_types": overall_with_rates["question_types"],
        "generation_rejection_reasons": overall["generation_rejection_reasons"],
        "validation_failure_reasons": overall["validation_failure_reasons"],
        "validation_failure_actions": overall["validation_failure_actions"],
        "validator_failure_reasons": overall["validator_failure_reasons"],
        "models": {key: add_derived_rates(value) for key, value in models.items()},
        "recent_exam_runs": recent_rows,
    }


# ---------------------------------------------------------------------------
# Overview analytics
# ---------------------------------------------------------------------------
def overview_summary(
    *, days: int | None = None, include_benchmarks: bool = False
) -> dict[str, Any]:
    """Return headline metrics, request statistics, averages and daily trends."""
    where, run_params = _run_filter(days, include_benchmarks)
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                f"""SELECT COUNT(*)::bigint AS generation_runs,
                           COALESCE(SUM(r.models_count), 0)::bigint AS model_exams,
                           MIN(r.created_at) AS first_run_at,
                           MAX(r.created_at) AS latest_run_at,
                           AVG({_duration_expression()}) AS avg_run_duration_ms,
                           COUNT({_duration_expression()}) AS duration_runs,
                           {_sum_select()},
                           {_jsonb_int_sum("->'recovery_outcomes'->>'entered_recovery'")}
                               AS recovery_entered,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'recovered'")}
                               AS recovery_recovered,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'unresolved'")}
                               AS recovery_unresolved,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'attempts'")}
                               AS recovery_attempts,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'accepted'")}
                               AS recovery_attempts_accepted,
                           {_jsonb_int_sum("->'duplicate_outcomes'->>'repair_rounds'")}
                               AS duplicate_rounds,
                           {_jsonb_int_sum("->'duplicate_outcomes'->>'replacement_attempts'")}
                               AS duplicate_replacements,
                           {_jsonb_int_sum("->'duplicate_outcomes'->>'unresolved_conflicts'")}
                               AS duplicate_unresolved,
                           COUNT(*) FILTER (
                               WHERE COALESCE(to_jsonb(r)->'catalog_cache', '{{}}'::jsonb)
                                     <> '{{}}'::jsonb
                           )::bigint AS cache_runs,
                           COUNT(*) FILTER (
                               WHERE COALESCE(
                                   (to_jsonb(r)->'catalog_cache'->>'catalog_cache_hit')::boolean,
                                   false)
                           )::bigint AS catalog_hits,
                           COUNT(*) FILTER (
                               WHERE COALESCE(
                                   (to_jsonb(r)->'catalog_cache'->>'compiler_cache_hit')::boolean,
                                   false)
                           )::bigint AS compiler_hits,
                           COUNT(*) FILTER (
                               WHERE COALESCE(
                                   (to_jsonb(r)->'catalog_cache'->>'compatibility_cache_hit')::boolean,
                                   false)
                           )::bigint AS compatibility_hits,
                           COUNT({_run_job_expression()})::bigint AS job_linked_runs
                    FROM evaluation_runs r
                    LEFT JOIN exams e ON e.id = r.exam_id
                    LEFT JOIN jobs j ON j.id = {_run_job_expression()}
                    {where}""",
                run_params,
            )
            totals = cursor.fetchone() or {}
            overall = _empty_bucket()
            for field in COUNT_FIELDS:
                if field in DB_COUNT_COLUMNS:
                    overall[field] = int(totals.get(field) or 0)
            overall["final_missing_or_invalid"] = sum(
                overall[field]
                for field in ("final_invalid", "final_unvalidated", "final_missing")
            )
            for api_field, expression in REASON_COLUMN_MAP.items():
                overall[api_field] = _fetch_reason_counts(
                    cursor, expression, where, run_params
                )
            overall_rates = derived_rates(overall)

            jobs = _job_telemetry(cursor, days, include_benchmarks)
            llm = _llm_population(cursor, days, include_benchmarks)
            requests = _request_statistics(
                cursor, days, include_benchmarks, jobs
            )
            trends = _trend_rows(cursor, days, include_benchmarks)
            recent = _recent_run_rows(cursor, 5)
            terminal_status = _fetch_jsonb_map_totals(
                cursor, "->'recovery_outcomes'->'terminal_status'", where, run_params
            )

    generation_runs = int(totals.get("generation_runs") or 0)
    model_exams = int(totals.get("model_exams") or 0)
    recovery_entered = int(totals.get("recovery_entered") or 0)
    recovery_recovered = int(totals.get("recovery_recovered") or 0)
    requested = overall["requested_questions"]
    final_valid = overall["final_valid"]

    population = llm["population"]
    averages_per_run = _averages(
        llm["evaluated"], population["runs_with_calls"]
    )
    averages_per_model_exam = _averages(
        llm["evaluated"], population["models_with_calls"]
    )
    latest = totals.get("latest_run_at")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "days": days,
            "include_benchmarks": bool(include_benchmarks),
        },
        "runs": {
            "generation_runs": generation_runs,
            "model_exams": model_exams,
            "first_run_at": totals.get("first_run_at").isoformat()
            if totals.get("first_run_at") else None,
            "latest_run_at": latest.isoformat() if latest else None,
            "job_linked_runs": int(totals.get("job_linked_runs") or 0),
            "duration_runs": int(totals.get("duration_runs") or 0),
            "avg_run_duration_ms": round(
                _number(totals.get("avg_run_duration_ms")) or 0, 1
            ) if totals.get("duration_runs") else None,
            "runs_with_llm_telemetry": population["runs_with_calls"],
        },
        "requests": requests,
        "quality": {
            "requested": requested,
            "generated_first": overall["generated_first"],
            "generation_completion_rate": overall_rates["generation_completion_rate"],
            "generation_rejected": overall["generation_rejected"],
            "generation_rejection_rate": overall_rates["generation_rejection_rate"],
            "shortfall_generated": overall["shortfall_generated"],
            "shortfall_still_missing": overall["shortfall_still_missing"],
            "validation_total_first": overall["validation_total_first"],
            "validation_passed_first": overall["validation_passed_first"],
            "first_pass_rate": overall_rates["first_validation_pass_rate"],
            "validation_coverage": overall_rates["validation_coverage_rate"],
            "validation_unvalidated_first": overall["validation_unvalidated_first"],
            "repair_sent": overall["repair_sent"],
            "repair_succeeded": overall["repair_succeeded"],
            "repair_success_rate": overall_rates["repair_success_rate"],
            "recovery_entered": recovery_entered,
            "recovery_recovered": recovery_recovered,
            "recovery_unresolved": int(totals.get("recovery_unresolved") or 0),
            "recovery_success_rate": _safe_divide(recovery_recovered, recovery_entered),
            "recovery_attempts": int(totals.get("recovery_attempts") or 0),
            "recovery_attempts_accepted": int(
                totals.get("recovery_attempts_accepted") or 0
            ),
            "recovery_attempt_success_rate": _safe_divide(
                totals.get("recovery_attempts_accepted"), totals.get("recovery_attempts")
            ),
            "final_valid": final_valid,
            "final_acceptance_rate": overall_rates["final_success_rate"],
            "final_invalid": overall["final_invalid"],
            "final_unvalidated": overall["final_unvalidated"],
            "final_missing": overall["final_missing"],
            "final_non_valid": overall["final_missing_or_invalid"],
            "cost_per_finalized_question_usd": (
                round(
                    llm["evaluated"]["estimated_cost_usd"]
                    / population["final_valid_with_calls"],
                    6,
                )
                if llm["evaluated"]["estimated_cost_usd"] is not None
                and population["final_valid_with_calls"] else None
            ),
        },
        "coverage": {
            "evaluation_records": population["evaluation_records"],
            "job_linked_runs": population["linked_runs"],
            "runs_with_telemetry": population["runs_with_calls"],
            "models_with_telemetry": population["models_with_calls"],
            "final_valid_with_calls": population["final_valid_with_calls"],
        },
        "jobs": jobs,
        "llm": {
            **llm["all"],
            "scopes": {
                "evaluated": llm["evaluated"],
                "other_generation": llm["other_generation"],
                "other_jobs": llm["other_jobs"],
                "unattributed": llm["unattributed"],
            },
            "population": population,
            "runs_with_calls": population["runs_with_calls"],
            "averages_per_generation_run": averages_per_run,
            "averages_per_model_exam": averages_per_model_exam,
            "totals": {
                "input_tokens": llm["all"]["input_tokens"],
                "output_tokens": llm["all"]["output_tokens"],
                "reasoning_tokens": llm["all"]["reasoning_tokens"],
                "cached_tokens": llm["all"]["cached_tokens"],
                "estimated_cost_usd": llm["all"]["estimated_cost_usd"],
            },
        },
        "trends": trends,
        "pipeline": {
            "question_stages": [
                {"label": "Requested", "value": requested, "unit": "questions"},
                {"label": "Generated", "value": overall["generated_first"], "unit": "questions"},
                {
                    "label": "Validated",
                    "value": overall["validation_total_first"],
                    "unit": "questions",
                },
                {
                    "label": "First-pass accepted",
                    "value": overall["validation_passed_first"],
                    "unit": "questions",
                },
                {"label": "Entered repair", "value": overall["repair_sent"], "unit": "questions"},
                {
                    "label": "Repair accepted",
                    "value": overall["repair_succeeded"],
                    "unit": "questions",
                },
                {"label": "Final accepted", "value": final_valid, "unit": "questions"},
            ],
            "slot_outcomes": [
                {
                    "label": "Finalized slots",
                    "value": int(terminal_status.get("finalized", 0)),
                    "unit": "slots",
                },
                {
                    "label": "Failed slots",
                    "value": sum(
                        count
                        for status, count in terminal_status.items()
                        if status != "finalized"
                    ),
                    "unit": "slots",
                },
            ],
        },
        "recovery": {
            "entered": recovery_entered,
            "recovered": recovery_recovered,
            "unresolved": int(totals.get("recovery_unresolved") or 0),
            "terminal_status": terminal_status,
        },
        "duplicates": {
            "repair_rounds": int(totals.get("duplicate_rounds") or 0),
            "replacement_attempts": int(totals.get("duplicate_replacements") or 0),
            "unresolved_conflicts": int(totals.get("duplicate_unresolved") or 0),
        },
        "catalog_cache": {
            "runs": int(totals.get("cache_runs") or 0),
            "catalog_hits": int(totals.get("catalog_hits") or 0),
            "compiler_hits": int(totals.get("compiler_hits") or 0),
            "compatibility_hits": int(totals.get("compatibility_hits") or 0),
            "catalog_hit_rate": _safe_divide(
                totals.get("catalog_hits"), totals.get("cache_runs")
            ),
            "compiler_hit_rate": _safe_divide(
                totals.get("compiler_hits"), totals.get("cache_runs")
            ),
            "compatibility_hit_rate": _safe_divide(
                totals.get("compatibility_hits"), totals.get("cache_runs")
            ),
        },
        "recent_runs": recent,
    }


def _averages(llm: dict[str, Any], denominator: int | None) -> dict[str, Any]:
    if not denominator:
        return {
            "denominator": None,
            "cost_usd": None,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
            "calls": None,
            "successful_calls": None,
            "failed_calls": None,
            "retries": None,
            "latency_ms": None,
        }
    divisor = float(denominator)
    cost = llm.get("estimated_cost_usd")
    return {
        "denominator": int(denominator),
        "cost_usd": round(cost / divisor, 6) if cost is not None else None,
        "input_tokens": round(llm["input_tokens"] / divisor, 1),
        "output_tokens": round(llm["output_tokens"] / divisor, 1),
        "reasoning_tokens": round(llm["reasoning_tokens"] / divisor, 1),
        "cached_tokens": round(llm["cached_tokens"] / divisor, 1),
        "calls": round(llm["calls"] / divisor, 2),
        "successful_calls": round(llm["successful"] / divisor, 2),
        "failed_calls": round(llm["failed"] / divisor, 2),
        "retries": round(llm["retries"] / divisor, 2),
        "latency_ms": llm["latency"]["avg_ms"],
    }


def _llm_run_coverage(
    cursor: Any, where: str, params: Iterable[Any]
) -> dict[str, int]:
    cursor.execute(
        f"""SELECT COUNT(DISTINCT e.job_id)::bigint AS jobs_with_calls
            {_llm_from()} {where}""",
        tuple(params),
    )
    row = cursor.fetchone() or (0,)
    return {"jobs_with_calls": int(_row_get(row, "jobs_with_calls", 0) or 0)}


def _job_telemetry(
    cursor: Any, days: int | None, include_benchmarks: bool
) -> dict[str, Any]:
    clauses = ["type = 'exam_generation'"]
    params: list[Any] = []
    if days:
        clauses.append("created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')")
        params.append(int(days))
    if not include_benchmarks:
        clauses.append("(request_data->>'benchmark_id' IS NULL)")
    where = " WHERE " + " AND ".join(clauses)
    cursor.execute(
        f"""SELECT status, COUNT(*)::bigint AS count,
                   COALESCE(SUM(attempt_count), 0)::bigint AS attempts
            FROM jobs {where} GROUP BY status""",
        params,
    )
    by_status = {
        str(_row_get(row, "status", 0)): int(_row_get(row, "count", 1))
        for row in cursor.fetchall()
    }
    cursor.execute(
        f"""SELECT COALESCE(SUM(attempt_count), 0)::bigint AS attempts
            FROM jobs {where}""",
        params,
    )
    attempts = int(_row_get(cursor.fetchone(), "attempts", 0) or 0)
    finished = (
        by_status.get("completed", 0)
        + by_status.get("failed", 0)
        + by_status.get("cancelled", 0)
    )
    cursor.execute(
        f"""SELECT COUNT(*)::bigint AS samples,
                   AVG(EXTRACT(EPOCH FROM (finished_at - started_at)) * 1000) AS avg_ms,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (
                       ORDER BY EXTRACT(EPOCH FROM (finished_at - started_at)) * 1000
                   ) AS p50_ms,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (
                       ORDER BY EXTRACT(EPOCH FROM (finished_at - started_at)) * 1000
                   ) AS p95_ms
            FROM jobs {where}
              AND started_at IS NOT NULL AND finished_at IS NOT NULL""",
        params,
    )
    duration = cursor.fetchone()
    cursor.execute(
        f"""SELECT COUNT(*)::bigint AS samples,
                   AVG(EXTRACT(EPOCH FROM (started_at - created_at)) * 1000) AS avg_ms,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (
                       ORDER BY EXTRACT(EPOCH FROM (started_at - created_at)) * 1000
                   ) AS p95_ms
            FROM jobs {where} AND started_at IS NOT NULL""",
        params,
    )
    queue = cursor.fetchone()
    cursor.execute(
        f"""SELECT COALESCE(error_code, 'UNKNOWN') AS code, COUNT(*)::bigint AS count
            FROM jobs {where} AND status = 'failed'
            GROUP BY 1 ORDER BY count DESC""",
        params,
    )
    error_codes = {
        str(_row_get(row, "code", 0)): int(_row_get(row, "count", 1))
        for row in cursor.fetchall()
    }
    duration_samples = int(_row_get(duration, "samples", 0) or 0) if duration else 0
    queue_samples = int(_row_get(queue, "samples", 0) or 0) if queue else 0
    return {
        "total": sum(by_status.values()),
        "by_status": by_status,
        "completed": by_status.get("completed", 0),
        "failed": by_status.get("failed", 0),
        "cancelled": by_status.get("cancelled", 0),
        "active": (
            by_status.get("queued", 0)
            + by_status.get("running", 0)
            + by_status.get("retrying", 0)
        ),
        "finished": finished,
        "success_rate": _safe_divide(by_status.get("completed", 0), finished),
        "attempts": attempts,
        "duration": {
            "samples": duration_samples,
            "avg_ms": round(_number(_row_get(duration, "avg_ms", 1)) or 0, 1)
            if duration_samples else None,
            "p50_ms": round(_number(_row_get(duration, "p50_ms", 2)) or 0, 1)
            if duration_samples else None,
            "p95_ms": round(_number(_row_get(duration, "p95_ms", 3)) or 0, 1)
            if duration_samples else None,
        },
        "queue": {
            "samples": queue_samples,
            "avg_ms": round(_number(_row_get(queue, "avg_ms", 1)) or 0, 1)
            if queue_samples else None,
            "p95_ms": round(_number(_row_get(queue, "p95_ms", 2)) or 0, 1)
            if queue_samples else None,
        },
        "error_codes": error_codes,
    }


def _trend_rows(
    cursor: Any, days: int | None, include_benchmarks: bool
) -> list[dict[str, Any]]:
    """Daily cohorts with correct denominators and continuous dates.

    Cost/token averages are per evaluation run: the numerator is the provider
    telemetry of that day's evaluation-linked jobs, and the denominator is the
    number of those runs that recorded provider calls.  Provider reliability is
    computed from all recorded requests by event date.
    """
    where, params = _run_filter(days, include_benchmarks)
    today = datetime.now(timezone.utc).date()
    if days:
        window_start = today - timedelta(days=int(days) - 1)
        cursor.execute(
            f"""SELECT MIN(r.created_at)::date AS first
                FROM evaluation_runs r {where}""",
            params,
        )
        run_first = (cursor.fetchone() or {}).get("first")
        cursor.execute(
            """SELECT MIN(e.created_at)::date AS first
               FROM llm_request_events e
               WHERE e.created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')""",
            (int(days),),
        )
        event_first = (cursor.fetchone() or {}).get("first")
        candidates = [value for value in (run_first, event_first) if value]
        start = max(window_start, min(candidates)) if candidates else window_start
        end = today
    else:
        cursor.execute(
            f"""SELECT MIN(r.created_at)::date AS first,
                       MAX(r.created_at)::date AS last
                FROM evaluation_runs r {where}""",
            params,
        )
        row = cursor.fetchone() or {}
        start, end = row.get("first"), row.get("last")
        if start is None:
            cursor.execute(
                """SELECT MIN(e.created_at)::date AS first,
                          MAX(e.created_at)::date AS last
                   FROM llm_request_events e"""
            )
            row = cursor.fetchone() or {}
            start, end = row.get("first"), row.get("last")
        start = start or today
        end = end or today
        if end < start:
            start = end
        if (end - start).days > 365:
            start = end - timedelta(days=365)
    dates = [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    ]

    cursor.execute(
        f"""WITH run_jobs AS (
                SELECT r.exam_id,
                       r.created_at::date AS day,
                       r.models_count,
                       r.validation_total_first,
                       r.validation_passed_first,
                       r.repair_sent,
                       r.repair_succeeded,
                       r.final_valid,
                       r.questions_requested,
                       COALESCE((to_jsonb(r)->'recovery_outcomes'->>'entered_recovery')::bigint, 0)
                           AS recovery_entered,
                       COALESCE((to_jsonb(r)->'recovery_outcomes'->>'recovered')::bigint, 0)
                           AS recovery_recovered,
                       {_duration_expression()} AS duration_ms,
                       {_run_job_expression()} AS job_id
                FROM evaluation_runs r
                LEFT JOIN exams e ON e.id = r.exam_id
                LEFT JOIN jobs j ON j.id = {_run_job_expression()}
                {where}
            ),
            job_usage AS (
                SELECT l.job_id,
                       COUNT(*)::bigint AS calls,
                       COUNT(l.input_tokens)::bigint AS usage_calls,
                       COALESCE(SUM(l.input_tokens), 0)::bigint AS input_tokens,
                       COALESCE(SUM(l.output_tokens), 0)::bigint AS output_tokens,
                       COALESCE(SUM(l.reasoning_tokens), 0)::bigint AS reasoning_tokens,
                       COALESCE(SUM(NULLIF(l.metadata->>'cached_tokens', '')::bigint), 0)::bigint
                           AS cached_tokens,
                       SUM(l.estimated_cost_usd) AS cost_usd,
                       COUNT(l.estimated_cost_usd)::bigint AS cost_calls,
                       SUM(COALESCE(l.duration_ms, 0))::bigint AS call_time_ms,
                       COUNT(l.duration_ms)::bigint AS latency_calls
                FROM llm_request_events l
                WHERE l.job_id IN (
                    SELECT job_id FROM run_jobs WHERE job_id IS NOT NULL
                )
                GROUP BY l.job_id
            )
            SELECT run_jobs.day,
                   COUNT(*)::bigint AS runs,
                   COUNT(job_usage.job_id)::bigint AS runs_with_telemetry,
                   COALESCE(SUM(run_jobs.models_count)
                            FILTER (WHERE job_usage.job_id IS NOT NULL), 0)::bigint
                       AS models_with_telemetry,
                   COALESCE(SUM(run_jobs.validation_total_first), 0)::bigint
                       AS validation_total,
                   COALESCE(SUM(run_jobs.validation_passed_first), 0)::bigint
                       AS validation_passed,
                   COALESCE(SUM(run_jobs.repair_sent), 0)::bigint AS repair_sent,
                   COALESCE(SUM(run_jobs.repair_succeeded), 0)::bigint AS repair_succeeded,
                   COALESCE(SUM(run_jobs.final_valid), 0)::bigint AS final_valid,
                   COALESCE(SUM(run_jobs.questions_requested), 0)::bigint AS requested,
                   COALESCE(SUM(run_jobs.recovery_entered), 0)::bigint AS recovery_entered,
                   COALESCE(SUM(run_jobs.recovery_recovered), 0)::bigint AS recovery_recovered,
                   AVG(run_jobs.duration_ms) AS avg_duration_ms,
                   COUNT(run_jobs.duration_ms)::bigint AS duration_samples,
                   COALESCE(SUM(job_usage.calls), 0)::bigint AS calls,
                   COALESCE(SUM(job_usage.usage_calls), 0)::bigint AS usage_calls,
                   COALESCE(SUM(job_usage.input_tokens), 0)::bigint AS input_tokens,
                   COALESCE(SUM(job_usage.output_tokens), 0)::bigint AS output_tokens,
                   COALESCE(SUM(job_usage.reasoning_tokens), 0)::bigint AS reasoning_tokens,
                   COALESCE(SUM(job_usage.cached_tokens), 0)::bigint AS cached_tokens,
                   SUM(job_usage.cost_usd) AS cost_usd,
                   COALESCE(SUM(job_usage.cost_calls), 0)::bigint AS cost_calls,
                   COALESCE(SUM(job_usage.call_time_ms), 0)::bigint AS call_time_ms,
                   COALESCE(SUM(job_usage.latency_calls), 0)::bigint AS latency_calls
            FROM run_jobs
            LEFT JOIN job_usage ON job_usage.job_id = run_jobs.job_id
            GROUP BY run_jobs.day
            ORDER BY run_jobs.day""",
        params,
    )
    run_rows = {str(row["day"]): dict(row) for row in cursor.fetchall()}

    llm_where, llm_params = _llm_where(days, include_benchmarks)
    cursor.execute(
        f"""SELECT e.created_at::date AS day,
                   COUNT(*)::bigint AS calls,
                   COUNT(*) FILTER (WHERE e.status <> 'success')::bigint AS failed
            {_llm_from()} {llm_where}
            GROUP BY 1 ORDER BY 1""",
        llm_params,
    )
    provider_rows = {str(row["day"]): dict(row) for row in cursor.fetchall()}

    trends: list[dict[str, Any]] = []
    for current in dates:
        key = current.isoformat()
        runs = run_rows.get(key) or {}
        provider = provider_rows.get(key) or {}
        with_telemetry = int(runs.get("runs_with_telemetry") or 0)
        cost = _number(runs.get("cost_usd"))
        duration_samples = int(runs.get("duration_samples") or 0)
        latency_calls = int(runs.get("latency_calls") or 0)
        provider_calls = int(provider.get("calls") or 0)
        trends.append({
            "day": key,
            "runs": int(runs.get("runs") or 0),
            "runs_with_telemetry": with_telemetry,
            "models_with_telemetry": int(runs.get("models_with_telemetry") or 0),
            "first_pass_rate": _safe_divide(
                runs.get("validation_passed"), runs.get("validation_total")
            ),
            "final_acceptance_rate": _safe_divide(
                runs.get("final_valid"), runs.get("requested")
            ),
            "repair_success_rate": _safe_divide(
                runs.get("repair_succeeded"), runs.get("repair_sent")
            ),
            "recovery_success_rate": _safe_divide(
                runs.get("recovery_recovered"), runs.get("recovery_entered")
            ),
            "recovery_samples": int(runs.get("recovery_entered") or 0),
            "avg_cost_per_run": (
                round(cost / with_telemetry, 6)
                if cost is not None and with_telemetry else None
            ),
            "avg_input_tokens_per_run": (
                round(int(runs.get("input_tokens") or 0) / with_telemetry, 1)
                if with_telemetry else None
            ),
            "avg_output_tokens_per_run": (
                round(int(runs.get("output_tokens") or 0) / with_telemetry, 1)
                if with_telemetry else None
            ),
            "avg_reasoning_tokens_per_run": (
                round(int(runs.get("reasoning_tokens") or 0) / with_telemetry, 1)
                if with_telemetry else None
            ),
            "avg_cached_tokens_per_run": (
                round(int(runs.get("cached_tokens") or 0) / with_telemetry, 1)
                if with_telemetry else None
            ),
            "avg_duration_ms": (
                round(_number(runs.get("avg_duration_ms")) or 0, 1)
                if duration_samples else None
            ),
            "duration_samples": duration_samples,
            "avg_call_latency_ms": (
                round(int(runs.get("call_time_ms") or 0) / latency_calls, 1)
                if latency_calls else None
            ),
            "latency_calls": latency_calls,
            "provider_calls": provider_calls,
            "provider_failed": int(provider.get("failed") or 0),
            "provider_failure_rate": _safe_divide(
                provider.get("failed"), provider_calls
            ),
            "usage_calls": int(runs.get("usage_calls") or 0),
        })
    return trends


# ---------------------------------------------------------------------------
# Exam reports
# ---------------------------------------------------------------------------
def _normalize_benchmark_mode(value: str | None) -> str:
    mode = str(value or "exclude").strip().lower()
    return mode if mode in {"exclude", "include", "only"} else "exclude"


def _benchmark_mode_clause(mode: str) -> str | None:
    column = _benchmark_column()
    if not extended_schema_available():
        return None
    if mode == "only":
        return f"{column} IS NOT NULL"
    if mode == "include":
        return None
    return f"{column} IS NULL"


def list_exam_reports(
    *,
    search: str = "",
    status: str = "",
    benchmark_mode: str = "exclude",
    days: int | None = None,
    limit: int = 25,
    offset: int = 0,
) -> dict[str, Any]:
    """Return a paginated list of per-exam report summaries."""
    mode = _normalize_benchmark_mode(benchmark_mode)
    clauses: list[str] = []
    params: list[Any] = []
    if days:
        clauses.append("r.created_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')")
        params.append(int(days))
    benchmark_clause = _benchmark_mode_clause(mode)
    if benchmark_clause:
        clauses.append(benchmark_clause)
    cleaned_search = str(search or "").strip()[:80]
    if cleaned_search:
        clauses.append("r.exam_id ILIKE %s")
        params.append(f"%{cleaned_search}%")
    if status:
        clauses.append("r.status = %s")
        params.append(str(status))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) FROM evaluation_runs r {where}", params
            )
            total = int(cursor.fetchone()[0])
        with conn.cursor(row_factory=dict_row) as page_cursor:
            rows = _page_rows(page_cursor, where, params, limit, offset)
    return {"items": rows, "total": total, "limit": limit, "offset": offset}


def _page_rows(
    cursor: Any, where: str, params: list[Any], limit: int, offset: int
) -> list[dict[str, Any]]:
    job_expression = _run_job_expression()
    cursor.execute(
        f"""SELECT r.exam_id, r.created_at, r.status, r.models_count,
                   r.questions_requested, r.validation_total_first,
                   r.validation_passed_first, r.repair_sent, r.repair_succeeded,
                   r.final_valid, r.final_invalid, r.final_unvalidated, r.final_missing,
                   {_duration_expression()} AS duration_ms,
                   to_jsonb(r)->>'failure_code' AS failure_code,
                   to_jsonb(r)->>'failure_message' AS failure_message,
                   {job_expression} AS job_id,
                   j.status AS job_status, j.stage AS job_stage,
                   COALESCE((to_jsonb(r)->'recovery_outcomes'->>'entered_recovery')::bigint, 0)
                       AS recovery_entered,
                   COALESCE((to_jsonb(r)->'recovery_outcomes'->>'recovered')::bigint, 0)
                       AS recovery_recovered,
                   COALESCE((to_jsonb(r)->'recovery_outcomes'->>'unresolved')::bigint, 0)
                       AS recovery_unresolved,
                   COALESCE(to_jsonb(r)->'catalog_cache', '{{}}'::jsonb) <> '{{}}'::jsonb
                       AS has_cache_telemetry,
                   COALESCE(to_jsonb(r)->'recovery_outcomes', '{{}}'::jsonb) <> '{{}}'::jsonb
                       AS has_recovery_telemetry,
                   u.calls, u.failed_calls, u.input_tokens, u.output_tokens,
                   u.reasoning_tokens, u.cached_tokens, u.cost_usd, u.avg_latency_ms
            FROM evaluation_runs r
            LEFT JOIN exams e ON e.id = r.exam_id
            LEFT JOIN jobs j ON j.id = {job_expression}
            LEFT JOIN LATERAL (
                SELECT COUNT(*)::bigint AS calls,
                       COUNT(*) FILTER (WHERE l.status <> 'success')::bigint AS failed_calls,
                       COALESCE(SUM(l.input_tokens), 0)::bigint AS input_tokens,
                       COALESCE(SUM(l.output_tokens), 0)::bigint AS output_tokens,
                       COALESCE(SUM(l.reasoning_tokens), 0)::bigint AS reasoning_tokens,
                       COALESCE(SUM(NULLIF(l.metadata->>'cached_tokens', '')::bigint), 0)::bigint
                           AS cached_tokens,
                       SUM(l.estimated_cost_usd) AS cost_usd,
                       AVG(l.duration_ms) AS avg_latency_ms
                FROM llm_request_events l
                WHERE l.job_id = {job_expression}
            ) u ON TRUE
            {where}
            ORDER BY r.created_at DESC, r.exam_id DESC
            LIMIT %s OFFSET %s""",
        (*params, int(limit), int(offset)),
    )
    items = []
    for row in cursor.fetchall():
        requested = _nonnegative_int(row.get("questions_requested"))
        final_valid = _nonnegative_int(row.get("final_valid"))
        cost = _number(row.get("cost_usd"))
        calls = int(row.get("calls") or 0)
        created = row.get("created_at")
        items.append({
            "exam_id": str(row["exam_id"]),
            "created_at": created.isoformat() if created else None,
            "status": str(row.get("status") or ""),
            "status_label": status_label(row.get("status")),
            "job_status": str(row.get("job_status") or "") or None,
            "job_stage": str(row.get("job_stage") or "") or None,
            "job_id": str(row["job_id"]) if row.get("job_id") else None,
            "models_count": _nonnegative_int(row.get("models_count")),
            "requested": requested,
            "final_valid": final_valid,
            "final_invalid": _nonnegative_int(row.get("final_invalid")),
            "final_unvalidated": _nonnegative_int(row.get("final_unvalidated")),
            "final_missing": _nonnegative_int(row.get("final_missing")),
            "first_pass_rate": _safe_divide(
                row.get("validation_passed_first"), row.get("validation_total_first")
            ),
            "final_acceptance_rate": _safe_divide(final_valid, requested),
            "repair_sent": _nonnegative_int(row.get("repair_sent")),
            "repair_succeeded": _nonnegative_int(row.get("repair_succeeded")),
            "repair_success_rate": _safe_divide(
                row.get("repair_succeeded"), row.get("repair_sent")
            ),
            "recovery_entered": _nonnegative_int(row.get("recovery_entered")),
            "recovery_recovered": _nonnegative_int(row.get("recovery_recovered")),
            "recovery_unresolved": _nonnegative_int(row.get("recovery_unresolved")),
            "duration_ms": (
                int(row["duration_ms"]) if row.get("duration_ms") is not None else None
            ),
            "failure_code": row.get("failure_code"),
            "failure_message": row.get("failure_message"),
            "has_llm_telemetry": calls > 0,
            "has_recovery_telemetry": bool(row.get("has_recovery_telemetry")),
            "llm": {
                "calls": calls,
                "failed_calls": int(row.get("failed_calls") or 0),
                "input_tokens": int(row.get("input_tokens") or 0),
                "output_tokens": int(row.get("output_tokens") or 0),
                "reasoning_tokens": int(row.get("reasoning_tokens") or 0),
                "cached_tokens": int(row.get("cached_tokens") or 0),
                "estimated_cost_usd": cost if calls else None,
                "avg_latency_ms": (
                    round(_number(row.get("avg_latency_ms")) or 0, 1)
                    if row.get("avg_latency_ms") is not None else None
                ),
            },
            "cost_per_finalized_question_usd": (
                round(cost / final_valid, 6) if cost is not None and final_valid else None
            ),
        })
    return items


def exam_report(exam_id: str) -> dict[str, Any] | None:
    """Return the complete report for one generation run."""
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                f"""SELECT r.*, {_run_job_expression()} AS linked_job_id,
                           e.job_id AS exam_job_id, e.metadata AS exam_metadata,
                           e.status AS exam_status
                    FROM evaluation_runs r
                    LEFT JOIN exams e ON e.id = r.exam_id
                    WHERE r.exam_id = %s""",
                (str(exam_id),),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            bucket = _bucket_from_row(row)
            bucket_with_rates = add_derived_rates(bucket)
            question_types = bucket_with_rates.pop("question_types", {})

            models_raw = row.get("model_performance") or {}
            models = {
                str(key): add_derived_rates(_bucket_from_raw(value))
                for key, value in models_raw.items()
                if isinstance(value, dict)
            } if isinstance(models_raw, dict) else {}

            job = _job_detail(cursor, row.get("linked_job_id"))
            llm = _exam_llm(cursor, row.get("linked_job_id"))
            stage_rows = _stage_timeline(cursor, str(exam_id), llm["by_stage"])
            recovery = row.get("recovery_outcomes")
            duplicates = row.get("duplicate_outcomes")
            cache = row.get("catalog_cache")
            audit_duration = _audit_duration_ms(cursor, str(exam_id))

    requested = bucket["requested_questions"]
    final_valid = bucket["final_valid"]
    cost = llm["totals"]["estimated_cost_usd"]
    created = row.get("created_at")
    duration_ms = row.get("duration_ms")
    if duration_ms is None and job and job.get("duration_ms") is not None:
        duration_ms = round(job["duration_ms"])
    if duration_ms is None:
        duration_ms = audit_duration
    return {
        "exam_id": str(row["exam_id"]),
        "job_id": str(row["linked_job_id"]) if row.get("linked_job_id") else None,
        "created_at": created.isoformat() if created else None,
        "status": str(row.get("status") or ""),
        "status_label": status_label(row.get("status")),
        "benchmark_id": str(row["benchmark_id"]) if row.get("benchmark_id") else None,
        "models_count": _nonnegative_int(row.get("models_count")),
        "failure_code": row.get("failure_code"),
        "failure_message": row.get("failure_message"),
        "summary": {
            "requested": requested,
            "generated_first": bucket["generated_first"],
            "generation_rejected": bucket["generation_rejected"],
            "validation_total_first": bucket["validation_total_first"],
            "validation_passed_first": bucket["validation_passed_first"],
            "validation_unvalidated_first": bucket["validation_unvalidated_first"],
            "first_pass_rate": bucket_with_rates["rates"]["first_validation_pass_rate"],
            "validation_coverage": bucket_with_rates["rates"]["validation_coverage_rate"],
            "repair_sent": bucket["repair_sent"],
            "repair_succeeded": bucket["repair_succeeded"],
            "repair_success_rate": bucket_with_rates["rates"]["repair_success_rate"],
            "final_valid": final_valid,
            "final_invalid": bucket["final_invalid"],
            "final_unvalidated": bucket["final_unvalidated"],
            "final_missing": bucket["final_missing"],
            "final_acceptance_rate": bucket_with_rates["rates"]["final_success_rate"],
            "duration_ms": int(duration_ms) if duration_ms is not None else None,
            "estimated_cost_usd": cost,
            "cost_per_finalized_question_usd": (
                round(cost / final_valid, 6) if cost is not None and final_valid else None
            ),
        },
        "evaluation": {
            "counters": {
                field: bucket[field]
                for field in COUNT_FIELDS
            },
            "rates": bucket_with_rates["rates"],
            "validation_failure_reasons": bucket["validation_failure_reasons"],
            "validation_failure_actions": bucket["validation_failure_actions"],
            "validator_failure_reasons": bucket["validator_failure_reasons"],
            "generation_rejection_reasons": bucket["generation_rejection_reasons"],
        },
        "question_types": question_types,
        "models": models,
        "job": job,
        "llm": llm,
        "stages": stage_rows,
        "recovery": recovery if isinstance(recovery, dict) else {},
        "duplicates": duplicates if isinstance(duplicates, dict) else {},
        "catalog_cache": cache if isinstance(cache, dict) else {},
        "missing": {
            "job": job is None,
            "llm": llm["totals"]["calls"] == 0,
            "stages": not stage_rows,
            "recovery": not isinstance(recovery, dict) or not recovery,
            "duplicates": not isinstance(duplicates, dict) or not duplicates,
            "catalog_cache": not isinstance(cache, dict) or not cache,
            "duration": duration_ms is None,
        },
    }


def _audit_duration_ms(cursor: Any, exam_id: str) -> int | None:
    cursor.execute(
        """SELECT (EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) * 1000)
                      AS span_ms,
                  COUNT(*) AS samples
           FROM exam_audit_events
           WHERE exam_id = %s
           HAVING COUNT(*) >= 2""",
        (exam_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return int(round(_number(row.get("span_ms")) or 0)) or None


def _job_detail(cursor: Any, job_id: Any) -> dict[str, Any] | None:
    if not job_id:
        return None
    cursor.execute(
        """SELECT id, status, stage, attempt_count, max_attempts, error_code,
                  error_message, created_at, started_at, finished_at,
                  EXTRACT(EPOCH FROM (finished_at - started_at)) * 1000 AS duration_ms,
                  EXTRACT(EPOCH FROM (started_at - created_at)) * 1000 AS queue_ms
           FROM jobs WHERE id = %s""",
        (str(job_id),),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "job_id": str(row["id"]),
        "status": str(row.get("status") or ""),
        "stage": str(row.get("stage") or ""),
        "attempt_count": int(row.get("attempt_count") or 0),
        "max_attempts": int(row.get("max_attempts") or 0),
        "error_code": row.get("error_code"),
        "error_message": row.get("error_message"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "started_at": row["started_at"].isoformat() if row.get("started_at") else None,
        "finished_at": row["finished_at"].isoformat() if row.get("finished_at") else None,
        "duration_ms": (
            round(_number(row.get("duration_ms")) or 0, 1)
            if row.get("duration_ms") is not None else None
        ),
        "queue_ms": (
            round(_number(row.get("queue_ms")) or 0, 1)
            if row.get("queue_ms") is not None else None
        ),
    }


def _exam_llm(cursor: Any, job_id: Any) -> dict[str, Any]:
    if not job_id:
        return {
            "totals": _llm_payload({}),
            "by_stage": [],
            "by_agent": [],
            "by_provider_model": [],
        }
    where = " WHERE e.job_id = %s"
    params = (str(job_id),)
    totals = _llm_totals(cursor, where, params)
    by_stage = _llm_groups(
        cursor,
        "COALESCE(NULLIF(e.metadata->>'stage', ''), 'unknown')",
        where,
        params,
    )
    by_agent = _llm_groups(
        cursor,
        "COALESCE(e.agent_role, e.operation, 'unknown')",
        where,
        params,
    )
    by_provider_model = _llm_groups(
        cursor,
        "e.provider || ' / ' || e.model",
        where,
        params,
    )
    for group in by_stage:
        group["stage"] = normalize_stage(group["key"])
    return {
        "totals": totals,
        "by_stage": by_stage,
        "by_agent": by_agent,
        "by_provider_model": by_provider_model,
    }


def _stage_timeline(
    cursor: Any, exam_id: str, llm_by_stage: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reconstruct stage wall time from the durable audit timeline.

    ``pipeline_stage`` events mark the start of a job stage; every later event
    (including stage-completion events) closes the interval attributed to that
    stage.  Summing those intervals yields real recorded wall time per stage,
    including repeated validation/repair cycles.  This is deliberately not
    derived from provider call durations, which run in parallel.
    """
    cursor.execute(
        """SELECT id, stage, event_type, created_at
           FROM exam_audit_events WHERE exam_id = %s
           ORDER BY created_at, id""",
        (exam_id,),
    )
    events = cursor.fetchall()
    wall: dict[str, float] = {}
    samples: dict[str, int] = {}
    previous = None
    active = "queue"
    for event in events:
        created = event.get("created_at")
        event_type = str(event.get("event_type") or "")
        if previous is not None and created is not None:
            delta = (created - previous).total_seconds() * 1000
            if delta >= 0:
                wall[active] = wall.get(active, 0.0) + delta
                samples[active] = samples.get(active, 0) + 1
        if event_type == "pipeline_stage":
            active = normalize_stage(event.get("stage"))
        elif event_type == "exam_created":
            active = "queue"
        previous = created
    llm_by_key = {
        normalize_stage(group.get("stage") or group.get("key")): group
        for group in llm_by_stage
    }
    keys = set(wall) | set(llm_by_key)
    ordered = [key for key in STAGE_ORDER if key in keys] + sorted(
        key for key in keys if key not in STAGE_ORDER
    )
    rows = []
    for key in ordered:
        group = llm_by_key.get(key) or {}
        rows.append({
            "stage": key,
            "wall_ms": round(wall[key]) if key in wall else None,
            "wall_samples": samples.get(key, 0),
            "llm": group or None,
        })
    return rows


# ---------------------------------------------------------------------------
# Pipeline analytics
# ---------------------------------------------------------------------------
def pipeline_analytics(
    *, days: int | None = None, include_benchmarks: bool = False
) -> dict[str, Any]:
    where, params = _run_filter(days, include_benchmarks)
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                f"""SELECT COUNT(*)::bigint AS runs,
                           MAX(r.created_at) AS latest_run_at,
                           COUNT({_run_job_expression()})::bigint AS job_linked_runs,
                           {_sum_select()},
                           COUNT(*) FILTER (
                               WHERE COALESCE(to_jsonb(r)->'recovery_outcomes', '{{}}'::jsonb)
                                     <> '{{}}'::jsonb
                           )::bigint AS recovery_runs,
                           COUNT(*) FILTER (
                               WHERE COALESCE(to_jsonb(r)->'catalog_cache', '{{}}'::jsonb)
                                     <> '{{}}'::jsonb
                           )::bigint AS cache_runs,
                           COUNT(*) FILTER (
                               WHERE COALESCE(to_jsonb(r)->'duplicate_outcomes', '{{}}'::jsonb)
                                     <> '{{}}'::jsonb
                           )::bigint AS duplicate_runs,
                           COUNT(*) FILTER (
                               WHERE COALESCE(
                                   (to_jsonb(r)->'catalog_cache'->>'catalog_cache_hit')::boolean,
                                   false)
                           )::bigint AS catalog_hits,
                           COUNT(*) FILTER (
                               WHERE COALESCE(
                                   (to_jsonb(r)->'catalog_cache'->>'compiler_cache_hit')::boolean,
                                   false)
                           )::bigint AS compiler_hits,
                           COUNT(*) FILTER (
                               WHERE COALESCE(
                                   (to_jsonb(r)->'catalog_cache'->>'compatibility_cache_hit')::boolean,
                                   false)
                           )::bigint AS compatibility_hits,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'slots'")} AS slots,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'entered_recovery'")} AS entered,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'recovered'")} AS recovered,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'unresolved'")} AS unresolved,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'attempts'")} AS attempts,
                           {_jsonb_int_sum("->'recovery_outcomes'->>'accepted'")} AS accepted,
                           {_jsonb_int_sum("->'duplicate_outcomes'->>'repair_rounds'")} AS duplicate_rounds,
                           {_jsonb_int_sum("->'duplicate_outcomes'->>'replacement_attempts'")}
                               AS duplicate_replacements,
                           {_jsonb_int_sum("->'duplicate_outcomes'->>'slots_replaced'")}
                               AS duplicate_slots,
                           {_jsonb_int_sum("->'duplicate_outcomes'->>'replacement_failures'")}
                               AS duplicate_failures,
                           {_jsonb_int_sum("->'duplicate_outcomes'->>'unresolved_conflicts'")}
                               AS duplicate_unresolved
                    FROM evaluation_runs r
                    LEFT JOIN exams e ON e.id = r.exam_id
                    {where}""",
                params,
            )
            coverage = cursor.fetchone() or {}
            reasons = {
                api_field: _fetch_reason_counts(cursor, expression, where, params)
                for api_field, expression in REASON_COLUMN_MAP.items()
            }
            question_types = _fetch_dimension(
                cursor, "r.question_type_performance", where, params
            )
            _add_dimension_reasons(
                cursor, question_types, "r.question_type_performance", where, params
            )
            for qtype in question_types.values():
                qtype["final_missing_or_invalid"] = sum(
                    qtype[field]
                    for field in ("final_invalid", "final_unvalidated", "final_missing")
                )
            question_types = {
                key: add_derived_rates(value)
                for key, value in sorted(question_types.items())
            }
            terminal_status = _fetch_jsonb_map_totals(
                cursor, "->'recovery_outcomes'->'terminal_status'", where, params
            )
            failure_stages = _fetch_jsonb_map_totals(
                cursor, "->'recovery_outcomes'->'slot_failure_stages'", where, params
            )
            failure_codes = _fetch_jsonb_map_totals(
                cursor, "->'recovery_outcomes'->'slot_failure_codes'", where, params
            )
            attempt_outcomes = _fetch_jsonb_map_totals(
                cursor, "->'recovery_outcomes'->'attempt_outcomes'", where, params
            )
            attempt_strategies = _fetch_jsonb_map_totals(
                cursor, "->'recovery_outcomes'->'attempt_strategies'", where, params
            )
            attempt_strategy_details = _fetch_jsonb_map_totals(
                cursor, "->'recovery_outcomes'->'attempt_strategy_details'", where, params
            )
            attempt_strategy_outcomes = _fetch_jsonb_nested_totals(
                cursor, "->'recovery_outcomes'->'attempt_strategy_outcomes'", where, params
            )
            attempt_strategy_detail_outcomes = _fetch_jsonb_nested_totals(
                cursor,
                "->'recovery_outcomes'->'attempt_strategy_detail_outcomes'",
                where,
                params,
            )
            attempt_actions = _fetch_jsonb_map_totals(
                cursor, "->'recovery_outcomes'->'attempt_actions'", where, params
            )
            attempt_failure_codes = _fetch_jsonb_map_totals(
                cursor, "->'recovery_outcomes'->'attempt_failure_codes'", where, params
            )

            llm_where, llm_params = _llm_where(days, include_benchmarks)
            llm_totals = _llm_totals(cursor, llm_where, llm_params)
            llm_by_stage = _llm_groups(
                cursor,
                "COALESCE(NULLIF(e.metadata->>'stage', ''), 'unknown')",
                llm_where,
                llm_params,
            )
            for group in llm_by_stage:
                group["stage"] = normalize_stage(group["key"])
            llm_by_agent = _llm_groups(
                cursor,
                "COALESCE(e.agent_role, e.operation, 'unknown')",
                llm_where,
                llm_params,
            )
            llm_by_provider_model = _llm_groups(
                cursor,
                "e.provider || ' / ' || e.model",
                llm_where,
                llm_params,
            )

    slots = int(coverage.get("slots") or 0)
    entered = int(coverage.get("entered") or 0)
    recovered = int(coverage.get("recovered") or 0)
    attempts = int(coverage.get("attempts") or 0)
    accepted = int(coverage.get("accepted") or 0)
    cache_runs = int(coverage.get("cache_runs") or 0)
    return {
        "window": {"days": days, "include_benchmarks": bool(include_benchmarks)},
        "latest_run_at": (
            coverage["latest_run_at"].isoformat()
            if coverage.get("latest_run_at") else None
        ),
        "coverage": {
            "runs": int(coverage.get("runs") or 0),
            "job_linked_runs": int(coverage.get("job_linked_runs") or 0),
            "recovery_runs": int(coverage.get("recovery_runs") or 0),
            "duplicate_runs": int(coverage.get("duplicate_runs") or 0),
            "cache_runs": cache_runs,
        },
        "question_types": question_types,
        "validation": {
            "validated_questions": int(coverage.get("validation_total_first") or 0),
            "passed_questions": int(coverage.get("validation_passed_first") or 0),
            "first_pass_rate": _safe_divide(
                coverage.get("validation_passed_first"),
                coverage.get("validation_total_first"),
            ),
            "rejection_reasons": reasons["validation_failure_reasons"],
            "repair_actions": reasons["validation_failure_actions"],
            "operational_failures": reasons["validator_failure_reasons"],
            "generation_rejections": reasons["generation_rejection_reasons"],
        },
        "repair": {
            "question_repairs_sent": int(coverage.get("repair_sent") or 0),
            "question_repairs_succeeded": int(coverage.get("repair_succeeded") or 0),
            "question_repair_success_rate": _safe_divide(
                coverage.get("repair_succeeded"), coverage.get("repair_sent")
            ),
            "attempts": attempts,
            "accepted": accepted,
            "success_rate": _safe_divide(accepted, attempts),
            "outcomes": attempt_outcomes,
            "strategies": attempt_strategies,
            "strategy_details": attempt_strategy_details,
            "strategy_outcomes": attempt_strategy_outcomes,
            "strategy_detail_outcomes": attempt_strategy_detail_outcomes,
            "actions": attempt_actions,
            "failure_codes": attempt_failure_codes,
        },
        "recovery": {
            "slots": slots,
            "entered": entered,
            "recovered": recovered,
            "unresolved": int(coverage.get("unresolved") or 0),
            "success_rate": _safe_divide(recovered, entered),
            "terminal_status": terminal_status,
            "failure_stages": failure_stages,
            "failure_codes": failure_codes,
        },
        "duplicates": {
            "repair_rounds": int(coverage.get("duplicate_rounds") or 0),
            "replacement_attempts": int(coverage.get("duplicate_replacements") or 0),
            "slots_replaced": int(coverage.get("duplicate_slots") or 0),
            "replacement_failures": int(coverage.get("duplicate_failures") or 0),
            "unresolved_conflicts": int(coverage.get("duplicate_unresolved") or 0),
        },
        "catalog_cache": {
            "runs": cache_runs,
            "catalog_hits": int(coverage.get("catalog_hits") or 0),
            "compiler_hits": int(coverage.get("compiler_hits") or 0),
            "compatibility_hits": int(coverage.get("compatibility_hits") or 0),
            "catalog_hit_rate": _safe_divide(coverage.get("catalog_hits"), cache_runs),
            "compiler_hit_rate": _safe_divide(coverage.get("compiler_hits"), cache_runs),
            "compatibility_hit_rate": _safe_divide(
                coverage.get("compatibility_hits"), cache_runs
            ),
        },
        "llm": {
            "totals": llm_totals,
            "by_stage": llm_by_stage,
            "by_agent": llm_by_agent,
            "by_provider_model": llm_by_provider_model,
        },
    }
