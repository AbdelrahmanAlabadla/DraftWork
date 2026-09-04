"""PostgreSQL persistence and aggregation for exam evaluation telemetry."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb

from app import db
from app.online.eval_stats import COUNT_FIELDS, REASON_FIELDS, add_derived_rates, derived_rates


RAW_COUNT_FIELDS = tuple(
    field for field in COUNT_FIELDS if field != "final_missing_or_invalid"
)
DB_COUNT_COLUMNS = {
    "requested_questions": "questions_requested",
    **{field: field for field in RAW_COUNT_FIELDS if field != "requested_questions"},
}
REASON_COLUMN_MAP = {
    "generation_rejection_reasons": "generation_rejection_reasons",
    "validation_failure_reasons": "validation_failure_reasons",
    "validator_failure_reasons": "validator_operational_failures",
}


def _nonnegative_int(value: Any) -> int:
    return max(int(value or 0), 0)


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
    if isinstance(qtypes, dict):
        result["question_types"] = {
            str(qtype): _raw_json_bucket(value)
            for qtype, value in qtypes.items()
            if isinstance(value, dict)
        }
    return result


def evaluation_row(
    exam_id: str,
    eval_stats: dict[str, Any],
    models_count: int,
    questions_requested_per_model: int,
) -> dict[str, Any]:
    """Transform one completed generation request into one database record."""
    if not 1 <= int(models_count) <= 4:
        raise ValueError("models_count must be between 1 and 4")
    overall = eval_stats.get("overall") or {}
    requested = int(models_count) * _nonnegative_int(questions_requested_per_model)
    raw = {field: _nonnegative_int(overall.get(field)) for field in RAW_COUNT_FIELDS}
    final_non_valid = sum(
        raw[field] for field in ("final_invalid", "final_unvalidated", "final_missing")
    )
    if requested and raw["final_valid"] == 0 and final_non_valid >= requested:
        status = "failed"
    elif final_non_valid:
        status = "needs_attention"
    else:
        status = "healthy"

    models = eval_stats.get("models") or {}
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
    }


def insert_evaluation_run(row: dict[str, Any]) -> None:
    columns = (
        "exam_id", "models_count", "questions_requested_per_model",
        "questions_requested", "generated_first", "missing_first",
        "generation_rejected", "shortfall_generated", "shortfall_still_missing",
        "validation_total_first", "validation_passed_first",
        "validation_failed_first", "validation_unvalidated_first", "repair_sent",
        "repair_succeeded", "repair_failed", "final_valid", "final_invalid",
        "final_unvalidated", "final_missing", "status",
        "generation_rejection_reasons", "validation_failure_reasons",
        "validator_operational_failures", "model_performance",
        "question_type_performance",
    )
    json_columns = {
        "generation_rejection_reasons", "validation_failure_reasons",
        "validator_operational_failures", "model_performance",
        "question_type_performance",
    }
    values = [Jsonb(row[column]) if column in json_columns else row[column] for column in columns]
    placeholders = ", ".join(["%s"] * len(columns))
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO evaluation_runs ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
        conn.commit()


def save_evaluation(
    exam_id: str,
    eval_stats: dict[str, Any],
    models_count: int,
    questions_requested_per_model: int,
) -> None:
    insert_evaluation_run(
        evaluation_row(
            exam_id,
            eval_stats,
            models_count,
            questions_requested_per_model,
        )
    )


def _empty_bucket() -> dict[str, Any]:
    bucket: dict[str, Any] = {field: 0 for field in COUNT_FIELDS}
    for field in REASON_FIELDS:
        bucket[field] = {}
    bucket["question_types"] = {}
    return bucket


def _sum_select(prefix: str = "", *, json_bucket: bool = False) -> str:
    return ", ".join(
        (
            f"COALESCE(SUM(({prefix}'{field}')::bigint), 0)::bigint AS {field}"
            if json_bucket else
            f"COALESCE(SUM(({column})::bigint), 0)::bigint AS {field}"
        )
        for field, column in DB_COUNT_COLUMNS.items()
    )


def _fetch_reason_counts(cursor: Any, column: str) -> dict[str, int]:
    cursor.execute(
        f"""SELECT reason.key, SUM((reason.value)::bigint)::bigint
            FROM evaluation_runs r
            CROSS JOIN LATERAL jsonb_each_text(r.{column}) AS reason(key, value)
            GROUP BY reason.key ORDER BY reason.key"""
    )
    return {str(reason): int(count) for reason, count in cursor.fetchall()}


def _fetch_dimension(cursor: Any, expression: str) -> dict[str, dict[str, Any]]:
    cursor.execute(
        f"""SELECT dimension.key, {_sum_select("dimension.value->>", json_bucket=True)}
            FROM evaluation_runs r
            CROSS JOIN LATERAL jsonb_each({expression}) AS dimension(key, value)
            GROUP BY dimension.key ORDER BY dimension.key"""
    )
    buckets: dict[str, dict[str, Any]] = {}
    for row in cursor.fetchall():
        bucket = _empty_bucket()
        for index, field in enumerate(DB_COUNT_COLUMNS, start=1):
            bucket[field] = int(row[index])
        buckets[str(row[0])] = bucket
    return buckets


def _add_dimension_reasons(
    cursor: Any, buckets: dict[str, dict[str, Any]], expression: str
) -> None:
    for api_field in REASON_FIELDS:
        cursor.execute(
            f"""SELECT dimension.key, reason.key,
                       SUM((reason.value)::bigint)::bigint
                FROM evaluation_runs r
                CROSS JOIN LATERAL jsonb_each({expression}) AS dimension(key, value)
                CROSS JOIN LATERAL jsonb_each_text(
                    COALESCE(dimension.value->'{api_field}', '{{}}'::jsonb)
                ) AS reason(key, value)
                GROUP BY dimension.key, reason.key"""
        )
        for dimension_key, reason, count in cursor.fetchall():
            buckets[str(dimension_key)][api_field][str(reason)] = int(count)


def _add_model_question_types(cursor: Any, models: dict[str, dict[str, Any]]) -> None:
    cursor.execute(
        f"""SELECT model.key, qtype.key, {_sum_select("qtype.value->>", json_bucket=True)}
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
            bucket[field] = int(row[index])
        models[str(row[0])]["question_types"][str(row[1])] = bucket

    for api_field in REASON_FIELDS:
        cursor.execute(
            f"""SELECT model.key, qtype.key, reason.key,
                       SUM((reason.value)::bigint)::bigint
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
        for model_key, qtype, reason, count in cursor.fetchall():
            models[str(model_key)]["question_types"][str(qtype)][api_field][str(reason)] = int(count)


def load_eval_summary() -> dict[str, Any]:
    """Aggregate all persisted runs while retaining the current API contract."""
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT COALESCE(SUM(models_count), 0)::bigint, {_sum_select()} FROM evaluation_runs"
            )
            totals = cursor.fetchone()
            overall = _empty_bucket()
            total_exam_runs = int(totals[0])
            for index, field in enumerate(DB_COUNT_COLUMNS, start=1):
                overall[field] = int(totals[index])
            overall["final_missing_or_invalid"] = sum(
                overall[field]
                for field in ("final_invalid", "final_unvalidated", "final_missing")
            )
            for api_field, column in REASON_COLUMN_MAP.items():
                overall[api_field] = _fetch_reason_counts(cursor, column)

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
                """SELECT exam_id, created_at, questions_requested,
                          validation_passed_first, validation_total_first,
                          repair_sent, final_valid, status
                   FROM evaluation_runs
                   ORDER BY created_at DESC, exam_id DESC LIMIT 20"""
            )
            recent_rows = cursor.fetchall()

    overall_with_rates = add_derived_rates(overall)
    recent = []
    for exam_id, created_at, requested, passed, validation_total, repair_sent, final_valid, status in recent_rows:
        rates = derived_rates({
            "requested_questions": requested,
            "validation_passed_first": passed,
            "validation_total_first": validation_total,
            "repair_sent": repair_sent,
            "final_valid": final_valid,
        })
        recent.append({
            "exam_id": exam_id,
            "generated_at": created_at.isoformat(),
            "requested": int(requested),
            "first_pass_rate": rates["first_validation_pass_rate"],
            "repairs_sent": int(repair_sent),
            "final_success_rate": rates["final_success_rate"],
            "status": "Healthy" if status == "healthy" else "Needs attention",
        })
    return {
        "total_exam_runs": total_exam_runs,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall_with_rates,
        "rates": overall_with_rates["rates"],
        "generation_rejection_reasons": overall["generation_rejection_reasons"],
        "validation_failure_reasons": overall["validation_failure_reasons"],
        "validator_failure_reasons": overall["validator_failure_reasons"],
        "models": {key: add_derived_rates(value) for key, value in models.items()},
        "recent_exam_runs": recent,
    }
