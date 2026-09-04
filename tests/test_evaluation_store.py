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
