from __future__ import annotations

import pytest

from app.api.repositories import sanitize_jsonb
from app.jobs import tasks as job_tasks
from app.online import assessment_catalog as catalog
from app.online import exam_builder, graph, validator
from app.online.exam_builder import immutable_exam_request, request_fingerprint
from app.online.receive import receive_request


def _request_state(**updates):
    state = {
        "document_id": "doc",
        "tasks": [("mcq", 1)],
        "num_models": 1,
        "difficulty": "mix",
        "selected_child_ids": ["c1"],
    }
    state.update(updates)
    return state


def test_receive_accepts_zero_per_type_when_another_type_is_requested():
    result = receive_request(_request_state(tasks=[("mcq", 0), ("equation", 2)]))
    assert result == {"tasks": [("equation", 2)]}


def test_receive_rejects_negative_all_zero_and_invalid_model_counts():
    assert "between 0" in receive_request(_request_state(tasks=[("mcq", -1)]))["error"]
    assert "greater than zero" in receive_request(
        _request_state(tasks=[("mcq", 0), ("equation", 0)])
    )["error"]
    assert "num_models" in receive_request(_request_state(num_models=0))["error"]


def test_graph_never_routes_plan_or_difficulty_errors_to_generation():
    invalid = {"plan_errors": ["Slot s1 is invalid"], "plan_attempts": 3}
    assert graph._route_after_plan(invalid) == "stop_with_error"
    assert graph._route_after_difficulty_adjustment(invalid) == "stop_with_error"
    recoverable = {"plan_errors": ["Slot s1 is invalid"], "plan_attempts": 0}
    assert graph._route_after_plan(recoverable) == "repair_plan"
    assert graph._route_after_difficulty_adjustment(recoverable) == "repair_plan"


def test_immutable_request_fingerprint_changes_for_generation_settings():
    base = immutable_exam_request(
        job_id="job", document_id="doc", tasks=[("mcq", 1)], num_models=1,
        selected_child_ids=["c1"], difficulty="medium",
        processed_document_key="processed", domain_type="STEM", subject="math",
    )
    for field, changed in (
        ("difficulty", "hard"), ("num_models", 2), ("subject", "physics"),
        ("selected_child_ids", ["c2"]), ("tasks", [["mcq", 2]]),
    ):
        candidate = {**base, field: changed}
        assert request_fingerprint(candidate) != request_fingerprint(base)


def test_question_ids_are_derived_from_stable_planner_slots():
    section = [
        {"slot_id": "m1_mcq_10", "question_id": "model1_mcq_1"},
        {"slot_id": "m1_mcq_2"},
    ]
    exam_builder._assign_ids(1, "mcq", section)
    assert [item["question_id"] for item in section] == [
        "model1_mcq_10", "model1_mcq_2",
    ]


def test_duplicate_question_ids_are_rejected_before_validation():
    with pytest.raises(ValueError, match="Duplicate question_id"):
        exam_builder._assert_unique_question_ids({"mcq": [
            {"slot_id": "m1_mcq_1", "question_id": "model1_mcq_1"},
            {"slot_id": "m1_mcq_2", "question_id": "model1_mcq_1"},
        ]})


def test_terminal_repair_failure_does_not_reenter_repair():
    state = {
        "repair_manifest": [{
            "slot_id": "m1_mcq_3", "status": "exhausted",
            "failure_code": "replacement_inventory_exhausted",
        }],
        "question_repair_attempts": {},
    }
    assert graph._route_after_validation(state) == "scan_exam_duplicates"


def test_jsonb_sanitizer_replaces_only_nul_recursively():
    payload = {"formula": "x" + chr(0) + "y", "nested": ["س", "a\\nb"]}
    assert sanitize_jsonb(payload) == {
        "formula": "x\ufffdy", "nested": ["س", "a\\nb"],
    }
    assert payload["formula"] == "x" + chr(0) + "y"


def test_deterministic_data_error_is_not_a_full_job_retry():
    from psycopg.errors import UntranslatableCharacter

    assert not job_tasks._is_retryable_generation_error(
        UntranslatableCharacter("NUL cannot be converted")
    )


def test_compatible_catalog_ignores_historical_chunk_order(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog.config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(catalog.config, "ASSESSMENT_CATALOG_VERSION", "v-test")
    path = tmp_path / "aa" / "catalog.json"
    path.parent.mkdir(parents=True)
    task = {
        "task_id": "task_1", "base_task_id": "task_1", "knowledge_anchor": "k",
        "concept": "c", "relationship": None, "task_operation": "identify",
        "student_work_product": "answer", "target": "t", "reasoning_mode": "recall",
        "allowed_question_types": ["mcq"], "evidence_chunk_ids": ["c1"],
        "required_givens": [], "allowed_variations": [], "variable_givens": [],
        "complexity_boundaries": [], "exact_source_instance": False,
        "scenario_family": None, "text_answerable": True, "native_difficulty": "easy",
        "prerequisites": [], "supported_difficulties": ["easy"],
    }
    path.write_text(__import__("json").dumps({
        "catalog_version": "v-test", "processed_document_key": "processed",
        "selected_chunk_ids": ["c2", "c1"],
        "assessment_profile": {"subject": "mathematics", "domain": "STEM"},
        "tasks": [task], "fitb_bank": [],
    }), encoding="utf-8")
    assert catalog._compatible_compiler_seed("processed", ["c1", "c2"]) is not None


def test_catalog_batch_rejects_cross_batch_variation_evidence():
    value = {
        "tasks": [{
            "anchor": "k", "concept": "c", "relation": None,
            "operation": "identify", "work_product": "answer", "target": "t",
            "mode": "conceptual_recall", "question_types": ["mcq"], "evidence_ids": ["c1"],
            "required_givens": [], "variations": [{
                "variation_group_id": None, "variation_id": None,
                "target": "t2", "operation": "identify", "reasoning_mode": "conceptual_recall",
                "student_work_product": "answer", "form": None, "context": None,
                "givens": [], "vary_numbers": False,
                "evidence_chunk_ids": ["outside"], "question_types": ["mcq"],
            }], "variable_givens": [], "boundaries": [], "exact_source_instance": False,
            "scenario_family": None, "answerable": True, "difficulty": "easy", "prereqs": [],
        }],
        "fitb_bank": [], "batch_complete": True,
    }
    with pytest.raises(ValueError, match="variation evidence IDs"):
        catalog._validate_completed_batch(
            value, response_model=catalog.CompactCatalogBatchResponse,
            valid_chunk_ids={"c1"},
        )
