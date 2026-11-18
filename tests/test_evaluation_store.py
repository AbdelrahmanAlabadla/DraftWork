from __future__ import annotations

from app.api.evaluation_store import evaluation_row


def _bucket(requested: int, final_valid: int) -> dict:
    return {
        "requested_questions": requested,
        "generated_first": requested,
        "missing_first": 0,
        "generation_rejected": 1,
        "shortfall_generated": 0,
        "shortfall_still_missing": 0,
        "validation_total_first": requested,
        "validation_passed_first": final_valid,
        "validation_failed_first": requested - final_valid,
        "validation_unvalidated_first": 0,
        "repair_sent": requested - final_valid,
        "repair_succeeded": 0,
        "repair_failed": requested - final_valid,
        "final_valid": final_valid,
        "final_invalid": requested - final_valid,
        "final_unvalidated": 0,
        "final_missing": 0,
        "final_missing_or_invalid": requested - final_valid,
        "generation_rejection_reasons": {"duplicate": 1},
        "validation_failure_reasons": {"wrong_answer": requested - final_valid},
        "validator_failure_reasons": {},
        "rates": {"final_success_rate": final_valid / requested},
    }


def test_evaluation_row_combines_models_and_preserves_question_types():
    model_1 = {**_bucket(20, 20), "question_types": {"mcq": _bucket(20, 20)}}
    model_2 = {**_bucket(20, 18), "question_types": {"mcq": _bucket(20, 18)}}
    overall = _bucket(40, 38)
    overall["generation_rejection_reasons"] = {"duplicate": 2}
    overall["validation_failure_reasons"] = {"wrong_answer": 2}
    overall["question_types"] = {"mcq": _bucket(40, 38)}

    row = evaluation_row(
        "exam_test", {"overall": overall, "models": {"1": model_1, "2": model_2}}, 2, 20
    )

    assert row["models_count"] == 2
    assert row["questions_requested_per_model"] == 20
    assert row["questions_requested"] == 40
    assert row["final_valid"] == 38
    assert row["status"] == "needs_attention"
    assert set(row["model_performance"]) == {"1", "2"}
    assert row["model_performance"]["2"]["question_types"]["mcq"]["final_valid"] == 18
    assert row["question_type_performance"]["mcq"]["requested_questions"] == 40
    assert row["generation_rejection_reasons"] == {"duplicate": 2}
    assert "rates" not in row["model_performance"]["1"]
    assert "final_missing_or_invalid" not in row["model_performance"]["1"]
    assert "final_missing_or_invalid" not in row


def test_evaluation_row_records_extended_outcome_telemetry():
    stats = _bucket(10, 9)
    stats["question_types"] = {"mcq": _bucket(10, 9)}
    outcomes = {
        "recovery": {"slots": 10, "entered_recovery": 2, "recovered": 1, "unresolved": 1},
        "duplicates": {"repair_rounds": 1, "unresolved_conflicts": 0},
        "catalog_cache": {"catalog_cache_hit": True, "compiler_cache_hit": False,
                          "compatibility_cache_hit": True},
    }
    row = evaluation_row(
        "exam_extended",
        {"overall": stats, "models": {}},
        1,
        10,
        job_id="11111111-1111-1111-1111-111111111111",
        duration_ms=12345,
        outcomes=outcomes,
        benchmark_id=None,
    )
    assert row["job_id"] == "11111111-1111-1111-1111-111111111111"
    assert row["duration_ms"] == 12345
    assert row["recovery_outcomes"] == outcomes["recovery"]
    assert row["duplicate_outcomes"] == outcomes["duplicates"]
    assert row["catalog_cache"] == outcomes["catalog_cache"]
    assert row["benchmark_id"] is None
    assert row["validation_failure_actions"] == {}
    assert row["status"] in {"healthy", "needs_attention"}


def test_evaluation_row_supports_terminal_failure_status_override():
    empty = {
        "overall": {"requested_questions": 12, "final_missing": 12},
        "models": {},
    }
    row = evaluation_row(
        "exam_failed",
        empty,
        1,
        12,
        status_override="failed",
        failure_code="TIMEOUT",
        failure_message="Generation exceeded its configured time limit",
        duration_ms=9000,
    )
    assert row["status"] == "failed"
    assert row["failure_code"] == "TIMEOUT"
    assert row["final_missing"] == 12
    assert row["final_valid"] == 0
    assert "final_missing_or_invalid" not in row

    cancelled = evaluation_row(
        "exam_cancelled", empty, 1, 12, status_override="cancelled",
    )
    assert cancelled["status"] == "cancelled"


def test_evaluation_row_keeps_legacy_payloads_backwards_compatible():
    row = evaluation_row("exam_legacy", {"overall": _bucket(20, 20)}, 1, 20)
    assert row["job_id"] is None
    assert row["duration_ms"] is None
    assert row["recovery_outcomes"] == {}
    assert row["duplicate_outcomes"] == {}
    assert row["catalog_cache"] == {}
    assert row["failure_code"] is None
    assert row["status"] == "healthy"
