"""PostgreSQL integration checks for evaluation persistence and analytics.

Run with ``RUN_POSTGRES_TESTS=1``. The tests create clearly labelled synthetic
records and remove them afterwards; user telemetry is never modified.
"""
from __future__ import annotations

import os
import uuid

import pytest
from psycopg.types.json import Jsonb

from app import db
from app.api import benchmarks, evaluation_store

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="set RUN_POSTGRES_TESTS=1 to exercise the configured PostgreSQL database",
)


def _stats(final_valid: int = 7) -> dict:
    mcq = {
        "requested_questions": 6,
        "generated_first": 3,
        "validation_total_first": 6,
        "validation_passed_first": final_valid - 4,
        "validation_failed_first": 1,
        "validation_unvalidated_first": 1,
        "repair_sent": 2,
        "repair_succeeded": 1,
        "final_valid": final_valid - 4,
        "final_invalid": 1,
        "final_unvalidated": 1,
        "final_missing": 0,
        "generation_rejection_reasons": {"duplicate": 2},
        "validation_failure_reasons": {"incorrect_result": 1},
        "validation_failure_actions": {"fix_answer": 1},
        "validator_failure_reasons": {"missing_verdict": 1},
    }
    return {
        "overall": {
            "requested_questions": 10,
            "generated_first": 9,
            "validation_total_first": 8,
            "validation_passed_first": 5,
            "validation_failed_first": 2,
            "validation_unvalidated_first": 1,
            "repair_sent": 3,
            "repair_succeeded": 2,
            "final_valid": final_valid,
            "final_invalid": 1,
            "final_unvalidated": 1,
            "final_missing": 1,
            "generation_rejection_reasons": {"duplicate": 2},
            "validation_failure_reasons": {"incorrect_result": 2},
            "validation_failure_actions": {"fix_answer": 2},
            "validator_failure_reasons": {"missing_verdict": 1},
            "question_types": {"mcq": mcq},
        },
        "models": {},
    }


OUTCOMES = {
    "recovery": {
        "slots": 10,
        "entered_recovery": 2,
        "recovered": 1,
        "unresolved": 1,
        "terminal_status": {"finalized": 9, "repair_failed": 1},
        "slot_failure_stages": {"repair": 1},
        "slot_failure_codes": {"recovery_v3_plan_exhausted": 1},
        "attempts": 3,
        "accepted": 1,
        "attempt_outcomes": {"accepted": 1, "rejected": 2},
        "attempt_strategies": {"REPAIR": 2, "SWITCH_VARIATION": 1},
        "attempt_strategy_details": {"recovery_v3_targeted_repair": 2,
                                     "recovery_v3_switch_variation": 1},
        "attempt_strategy_outcomes": {
            "REPAIR": {"accepted": 1, "rejected": 1},
            "SWITCH_VARIATION": {"rejected": 1},
        },
        "attempt_actions": {"repair_question": 2, "switch_variation": 1},
        "attempt_failure_codes": {"contradicted": 2},
    },
    "duplicates": {
        "repair_rounds": 1,
        "replacement_attempts": 2,
        "slots_replaced": 1,
        "replacement_failures": 0,
        "unresolved_conflicts": 0,
    },
    "catalog_cache": {
        "catalog_cache_hit": True,
        "compiler_cache_hit": False,
        "compatibility_cache_hit": True,
    },
}


def _clear(exam_id: str) -> None:
    with db.connection() as conn:
        conn.execute("DELETE FROM evaluation_runs WHERE exam_id = %s", (exam_id,))
        conn.commit()


def test_persistence_is_idempotent_and_detail_uses_recorded_values():
    exam_id = f"exam_eval_test_{uuid.uuid4().hex[:8]}"
    try:
        evaluation_store.save_evaluation(
            exam_id,
            _stats(7),
            1,
            10,
            duration_ms=1234,
            outcomes=OUTCOMES,
        )
        # A retried job rewrites the same exam row instead of failing on the PK.
        evaluation_store.save_evaluation(
            exam_id,
            _stats(8),
            1,
            10,
            duration_ms=4321,
            outcomes=OUTCOMES,
        )
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT count(*), max(final_valid), max(duration_ms) "
                "FROM evaluation_runs WHERE exam_id = %s",
                (exam_id,),
            ).fetchone()
        assert rows[0] == 1
        assert rows[1] == 8
        assert rows[2] == 4321

        report = evaluation_store.exam_report(exam_id)
        assert report is not None
        assert report["summary"]["requested"] == 10
        assert report["summary"]["final_valid"] == 8
        assert report["summary"]["first_pass_rate"] == round(5 / 8, 4)
        assert report["summary"]["duration_ms"] == 4321
        assert report["question_types"]["mcq"]["final_valid"] == 4
        assert report["recovery"]["terminal_status"]["finalized"] == 9
        assert report["recovery"]["attempt_strategies"]["REPAIR"] == 2
        assert report["duplicates"]["repair_rounds"] == 1
        assert report["catalog_cache"]["catalog_cache_hit"] is True
        assert report["missing"]["job"] is True
        assert report["missing"]["llm"] is True

        page = evaluation_store.list_exam_reports(search=exam_id, days=0)
        assert page["total"] == 1
        item = page["items"][0]
        assert item["has_recovery_telemetry"] is True
        assert item["llm"]["calls"] == 0

        paginated = evaluation_store.list_exam_reports(search=exam_id, days=0, limit=1)
        assert paginated["total"] == 1
        assert len(paginated["items"]) == 1
    finally:
        _clear(exam_id)


def test_benchmark_records_are_linked_and_isolated():
    suffix = uuid.uuid4().hex[:8]
    benchmark_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    owner_session = str(uuid.uuid4())
    exam_id = f"exam_eval_bench_{suffix}"
    try:
        with db.connection() as conn:
            conn.execute(
                """INSERT INTO evaluation_benchmarks
                       (id, status, label, config, reproducibility, summary,
                        owner_session_id, owner_user_id)
                   VALUES (%s, 'completed', 'synthetic benchmark', %s, %s, %s, %s, NULL)""",
                (
                    benchmark_id,
                    Jsonb({"repeats": 1, "tasks": {"mcq": 10}}),
                    Jsonb({"git_commit": "test"}),
                    Jsonb({}),
                    owner_session,
                ),
            )
            conn.execute(
                """INSERT INTO evaluation_benchmark_runs
                       (id, benchmark_id, run_index, status, exam_id)
                   VALUES (%s, %s, 1, 'completed', %s)""",
                (run_id, benchmark_id, exam_id),
            )
            conn.commit()
        evaluation_store.save_evaluation(
            exam_id,
            _stats(7),
            1,
            10,
            outcomes=OUTCOMES,
            benchmark_id=benchmark_id,
        )
        report = evaluation_store.exam_report(exam_id)
        assert report["benchmark_id"] == benchmark_id
        payload = benchmarks.get_benchmark(
            benchmark_id,
            session_id=owner_session,
            user_id=str(uuid.uuid4()),
            advance=False,
        )
        assert payload is not None
        assert payload["summary"]["runs"]["completed"] == 1
        assert payload["summary"]["quality"]["final_valid"] == 7
        assert payload["runs"][0]["exam_id"] == exam_id
        comparison = benchmarks.compare_benchmarks(
            [benchmark_id, benchmark_id],
            session_id=owner_session,
            user_id=str(uuid.uuid4()),
        )
        assert comparison["comparisons"][0]["comparable"] is True
        assert comparison["comparisons"][0]["metrics"][0]["delta"] == 0
        with db.connection() as conn:
            isolated = conn.execute(
                "SELECT count(*) FROM evaluation_runs "
                "WHERE exam_id = %s AND benchmark_id = %s",
                (exam_id, benchmark_id),
            ).fetchone()[0]
        assert isolated == 1
    finally:
        with db.connection() as conn:
            conn.execute(
                "DELETE FROM evaluation_benchmarks WHERE id = %s", (benchmark_id,)
            )
            conn.commit()
        _clear(exam_id)


def test_overview_and_pipeline_analytics_execute_against_live_schema():
    exam_id = f"exam_eval_test_{uuid.uuid4().hex[:8]}"
    try:
        evaluation_store.save_evaluation(
            exam_id,
            _stats(7),
            1,
            10,
            duration_ms=1000,
            outcomes=OUTCOMES,
        )
        overview = evaluation_store.overview_summary(days=0)
        assert overview["runs"]["generation_runs"] >= 1
        assert overview["quality"]["requested"] >= 10
        assert overview["llm"]["calls"] >= 0
        pipeline = evaluation_store.pipeline_analytics(days=0)
        assert pipeline["coverage"]["runs"] >= 1
        assert "mcq" in pipeline["question_types"]
        assert pipeline["recovery"]["terminal_status"]["finalized"] >= 9
        assert pipeline["repair"]["strategy_outcomes"]["REPAIR"]["accepted"] >= 1
        assert pipeline["catalog_cache"]["runs"] >= 1
        legacy = evaluation_store.load_eval_summary()
        assert legacy["total_generation_runs"] >= 1
        assert legacy["total_model_exams"] >= 1
        assert legacy["latest_run_at"]
    finally:
        _clear(exam_id)


def test_terminal_failure_rows_are_listed_and_filterable():
    exam_id = f"exam_eval_failed_{uuid.uuid4().hex[:8]}"
    try:
        evaluation_store.save_terminal_failure(
            exam_id,
            models_count=2,
            questions_requested_per_model=10,
            status="failed",
            failure_code="TIMEOUT",
            failure_message="Generation exceeded its configured time limit",
            duration_ms=1000,
        )
        report = evaluation_store.exam_report(exam_id)
        assert report is not None
        assert report["status"] == "failed"
        assert report["status_label"] == "Failed"
        assert report["failure_code"] == "TIMEOUT"
        assert report["summary"]["requested"] == 20
        assert report["summary"]["final_valid"] == 0
        assert report["summary"]["final_missing"] == 20

        page = evaluation_store.list_exam_reports(
            search=exam_id, days=0, status="failed"
        )
        assert page["total"] == 1
        assert page["items"][0]["status_label"] == "Failed"
        assert page["items"][0]["failure_code"] == "TIMEOUT"
        assert evaluation_store.list_exam_reports(
            search=exam_id, days=0, status="healthy"
        )["total"] == 0
    finally:
        _clear(exam_id)

def test_overview_scopes_and_request_population_are_consistent():
    overview = evaluation_store.overview_summary(days=0)
    requests = overview["requests"]
    assert requests["total"] == (
        requests["completed"] + requests["failed"]
        + requests["cancelled"] + requests["active"]
    )
    assert requests["questions_finalized"] <= requests["questions_requested"]
    assert requests["model_exams_produced"] <= requests["model_exams_requested"]
    assert requests["model_exams_not_produced"] >= 0

    llm = overview["llm"]
    scopes = llm["scopes"]
    scope_calls = sum(scope["calls"] for scope in scopes.values())
    assert scope_calls == llm["calls"]
    assert llm["averages_per_generation_run"]["denominator"] == llm["population"]["runs_with_calls"]
    if llm["population"]["runs_with_calls"]:
        assert llm["averages_per_generation_run"]["calls"] is not None
        assert llm["averages_per_model_exam"]["denominator"] == llm["population"]["models_with_calls"]

    pipeline = evaluation_store.pipeline_analytics(days=0)
    assert "strategy_detail_outcomes" in pipeline["repair"]
    assert pipeline["repair"]["attempts"] == sum(
        pipeline["repair"]["outcomes"].values()
    )
    assert pipeline["coverage"]["duplicate_runs"] <= pipeline["coverage"]["runs"]

    assert overview["coverage"]["final_valid_with_calls"] >= 0
    if overview["llm"]["scopes"]["evaluated"]["estimated_cost_usd"] is not None and overview["coverage"]["final_valid_with_calls"]:
        expected = round(
            overview["llm"]["scopes"]["evaluated"]["estimated_cost_usd"]
            / overview["coverage"]["final_valid_with_calls"],
            6,
        )
        assert overview["quality"]["cost_per_finalized_question_usd"] == expected
