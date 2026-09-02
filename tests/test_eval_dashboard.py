from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from app.api.main import app
from app.api import exam_store


client = TestClient(app)


def _eval(requested: int, generated: int, passed: int, final: int, model: str = "1"):
    return {
        "overall": {
            "requested_questions": requested,
            "generated_first": generated,
            "missing_first": max(requested - generated, 0),
            "shortfall_generated": max(requested - generated, 0),
            "validation_total_first": requested,
            "validation_passed_first": passed,
            "validation_failed_first": requested - passed,
            "validation_unvalidated_first": 0,
            "repair_sent": requested - passed,
            "repair_succeeded": final - passed,
            "repair_failed": max(requested - final, 0),
            "final_valid": final,
            "final_invalid": requested - final,
            "final_unvalidated": 0,
            "final_missing": 0,
            "final_missing_or_invalid": requested - final,
            "generation_rejection_reasons": {"duplicate": 1},
            "validation_failure_reasons": {"wrong_answer": 1},
            "validator_failure_reasons": {},
            "question_types": {},
        },
        "models": {
            model: {
                "requested_questions": requested,
                "generated_first": generated,
                "missing_first": max(requested - generated, 0),
                "shortfall_generated": max(requested - generated, 0),
                "validation_total_first": requested,
                "validation_passed_first": passed,
                "validation_failed_first": requested - passed,
                "validation_unvalidated_first": 0,
                "repair_sent": requested - passed,
                "repair_succeeded": final - passed,
                "repair_failed": max(requested - final, 0),
                "final_valid": final,
                "final_invalid": requested - final,
                "final_unvalidated": 0,
                "final_missing": 0,
                "final_missing_or_invalid": requested - final,
                "generation_rejection_reasons": {"duplicate": 1},
                "validation_failure_reasons": {"wrong_answer": 1},
                "validator_failure_reasons": {},
                "question_types": {},
            }
        },
    }


def setup_function():
    exam_store.clear()


def teardown_function():
    exam_store.clear()


def test_eval_summary_aggregates_rates_reasons_models_and_recent_order():
    old_id = exam_store.save_exam([], eval_stats=_eval(10, 9, 8, 10))
    new_id = exam_store.save_exam([], eval_stats=_eval(20, 18, 16, 19))
    exam_store._exams[old_id]["created_at"] -= 20

    response = client.get("/api/eval-summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_exam_runs"] == 2
    assert data["overall"]["requested_questions"] == 30
    assert data["overall"]["final_valid"] == 29
    assert data["rates"]["generation_completion_rate"] == 0.9
    assert data["rates"]["first_validation_pass_rate"] == 0.8
    assert data["generation_rejection_reasons"] == {"duplicate": 2}
    assert data["validation_failure_reasons"] == {"wrong_answer": 2}
    assert data["validator_failure_reasons"] == {}
    assert list(data["models"]) == ["1"]
    assert [row["exam_id"] for row in data["recent_exam_runs"]] == [new_id, old_id]
    assert data["recent_exam_runs"][0]["status"] == "Needs attention"


def test_eval_summary_handles_zero_denominators():
    exam_store.save_exam([], eval_stats={"overall": {}, "models": {}})
    data = client.get("/api/eval-summary").json()
    assert data["rates"] == {
        "generation_completion_rate": None,
        "shortfall_recovery_rate": None,
        "first_validation_pass_rate": None,
        "repair_success_rate": None,
        "final_success_rate": None,
    }
    assert data["recent_exam_runs"][0]["status"] == "No telemetry"


def test_eval_dashboard_page_loads_successfully():
    response = client.get("/eval.html")
    assert response.status_code == 200
    assert "Eval Dashboard" in response.text
    assert "js/eval.js" in response.text
    assert "Final Unvalidated" in response.text
    assert "Validator Operational Failures" in response.text
