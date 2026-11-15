from __future__ import annotations

import threading
import time

import pytest

from app.llm import scheduler
from app.online import duplicate_pipeline, validator


def _pass_report(model: int, qid: str) -> dict:
    return {
        "model_number": model, "all_pass": True, "role_results": {},
        "role_fingerprints": {}, "validation_call_logs": [],
        "verdicts": [{
            "question_id": qid, "action": "PASS", "validation_status": "PASS",
            "question_valid": True, "answer_valid": True, "failure_code": "pass",
            "fields_to_fix": [], "required_fixes": [], "role_statuses": {},
            "reason": "", "expected_fix": "",
        }],
    }


@pytest.mark.parametrize("model_count", [1, 2, 3])
def test_one_two_and_three_model_initial_validation(monkeypatch, model_count):
    from app.online import multi_validator

    barrier = threading.Barrier(model_count)
    lock = threading.Lock()
    active = peak = 0

    def fake_validate(exam, **_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=3)
        time.sleep(0.02)
        with lock:
            active -= 1
        mn = int(exam["model_number"])
        return _pass_report(mn, f"q{mn}")

    monkeypatch.setattr(multi_validator, "validate_exam_multi", fake_validate)
    model_numbers = tuple(range(1, model_count + 1))
    exams = [{
        "model_number": mn,
        "questions": {"why": [{
            "slot_id": f"s{mn}", "question_id": f"q{mn}",
            "question": f"Why {mn}?", "reference_answer": "Because.",
        }]},
    } for mn in model_numbers]
    plans = [{
        "model_number": mn, "items": {"why": [{
            "slot_id": f"s{mn}", "task_id": f"t{mn}",
            "source_chunk_id": f"c{mn}", "support_chunk_ids": [],
        }]},
    } for mn in model_numbers]
    result = validator.validate_generated_questions({
        "num_models": model_count, "generated_exams": exams, "plans": plans,
        "retrieved_chunks": [
            {"child_id": f"c{mn}", "content": "Evidence"} for mn in model_numbers
        ],
        "assessment_profile": {"domain": "STEM"},
        "validation_reports": [], "validated_models": [],
        "repair_manifest": [], "repair_candidates": [],
        "pending_revalidation_ids": {}, "warnings": [],
    })
    assert peak == model_count
    assert set(result["validated_models"]) == set(model_numbers)
    timings = [
        value for value in result["performance_telemetry"]
        if value.get("stage") == "model_validation"
    ]
    if model_count > 1:
        assert {value["model_number"] for value in timings} == set(model_numbers)


def test_within_model_duplicate_scans_overlap_and_merge_deterministically(monkeypatch):
    barrier = threading.Barrier(2)
    active = peak = 0
    lock = threading.Lock()

    def fake_impl(local_state):
        nonlocal active, peak
        model = int(local_state["generated_exams"][0]["model_number"])
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=3)
        with lock:
            active -= 1
        return {
            "duplicate_reports": [{"model_number": model, "candidate_pairs": []}],
            "duplicate_similarity_distributions": [], "pending_duplicate_groups": [],
            "duplicate_scan_count": 1, "invalid_questions": [], "warnings": [],
            "generated_exams": local_state["generated_exams"],
            "duplicate_fact_cache": {},
            "duplicate_screen_fingerprints": {str(model): {}},
            "duplicate_candidate_blacklist": {}, "duplicate_replacement_failures": {},
            "duplicate_repair_history": [], "duplicate_repair_rounds": 0,
            "duplicate_verification_pending": False, "duplicate_verification_attempt": 0,
            "unresolved_duplicate_conflicts": [],
        }

    monkeypatch.setattr(duplicate_pipeline, "_scan_exam_duplicates_impl", fake_impl)
    result = duplicate_pipeline.scan_exam_duplicates({
        "generated_exams": [
            {"model_number": 1, "questions": {}},
            {"model_number": 2, "questions": {}},
        ],
    })
    assert peak == 2
    assert [value["model_number"] for value in result["duplicate_reports"]] == [1, 2]
    assert [value["model_number"] for value in result["generated_exams"]] == [1, 2]


def test_scheduler_telemetry_distinguishes_queue_and_provider_time(monkeypatch):
    monkeypatch.setattr(scheduler.config, "LLM_JOB_CONCURRENCY_LIMIT", 1)
    scheduler.reset_scheduler_for_tests()

    def work():
        with scheduler.llm_request_slot(agent="test", operation="telemetry"):
            time.sleep(0.02)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    snapshot = scheduler.scheduler_snapshot("__unscoped__")
    events = scheduler.scheduler_request_events("__unscoped__")
    assert snapshot["peak"] == 1
    assert snapshot["total_queue_ms"] > 0
    assert len(events) == 2
    assert all(value["provider_ms"] >= 15 for value in events)
    assert max(value["queue_ms"] for value in events) >= 10
