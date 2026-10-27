from __future__ import annotations

from copy import deepcopy

from app.online import assessment_catalog, duplicate_pipeline
from app.online.graph import _route_after_validation
from app.online.lifecycle import (
    assessment_instance_fingerprint,
    initialize_slot_lifecycle,
    reconcile_slot_lifecycle,
)


def _plan(task_id: str, givens: list[str]) -> dict:
    return {
        "model_number": 1,
        "items": {"mcq": [{
            "slot_id": "m1_mcq_1", "task_id": task_id,
            "task_operation": "calculate_total_with_tip",
            "target": "total including tip", "student_work_product": "total",
            "effective_givens": givens, "actual_difficulty": "easy",
        }]},
    }


def test_slot_lifecycle_preserves_original_assignment_and_revisions_current_state():
    records = initialize_slot_lifecycle([("mcq", 1)], 1)
    first = reconcile_slot_lifecycle(records, {"plans": [_plan("tip", ["60", "17%"])]}, stage="planning")
    original = deepcopy(first["m1_mcq_1"]["original_assignment"])
    second = reconcile_slot_lifecycle(first, {"plans": [_plan("systems", ["two equations"])]}, stage="repair")

    slot = second["m1_mcq_1"]
    assert slot["original_assignment"] == original
    assert slot["current_assignment"]["task_id"] == "systems"
    assert slot["assignment_revision"] == 2
    assert slot["history"][-1]["event"] == "assignment_changed"


def test_question_content_revision_changes_only_when_content_changes():
    records = initialize_slot_lifecycle([("mcq", 1)], 1)
    state = {
        "plans": [_plan("tip", ["60", "17%"])],
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [{
            "slot_id": "m1_mcq_1", "question_id": "model1_mcq_1",
            "question": "60 and 17%", "options": {"A": "70.2"}, "correct_answer": "A",
        }]}}],
    }
    first = reconcile_slot_lifecycle(records, state, stage="generation")
    unchanged = reconcile_slot_lifecycle(first, state, stage="validation")
    assert unchanged["m1_mcq_1"]["question_revision"] == 1
    changed_state = deepcopy(state)
    changed_state["generated_exams"][0]["questions"]["mcq"][0]["question"] = "80 and 15%"
    changed = reconcile_slot_lifecycle(unchanged, changed_state, stage="repair")
    assert changed["m1_mcq_1"]["question_revision"] == 2


def test_dinner_tip_contract_is_safely_normalized_without_changing_evidence():
    task = {
        "task_id": "task_b76c4d06b6beb27a",
        "task_operation": "calculate_total_with_tip",
        "target": "calculate total including tip",
        "required_givens": ["قيمة الفاتورة 60 درهما", "نسبة الإكرامية 17%"],
        "variable_givens": ["قيمة الفاتورة", "نسبة الإكرامية"],
        "exact_source_instance": True,
        "evidence_chunk_ids": ["source-tip"],
        "allowed_variations": [],
    }
    repaired, diagnostics = assessment_catalog._repair_catalog_numerical_contract(task)
    assert repaired["required_givens"] == []
    assert repaired["source_example_givens"] == task["required_givens"]
    assert repaired["variable_givens"] == task["variable_givens"]
    assert repaired["evidence_chunk_ids"] == ["source-tip"]
    assert repaired["exact_source_instance"] is False
    assert any(item["code"] == "safe_numeric_parameter_normalization" for item in diagnostics)


def test_legitimate_exact_numeric_target_is_preserved():
    task = {
        "task_operation": "construct_inequalities",
        "target": "construct a region of area 20 in quadrant IV",
        "required_givens": ["area is exactly 20 square units", "quadrant IV"],
        "variable_givens": [], "exact_source_instance": True,
        "allowed_variations": [],
    }
    repaired, diagnostics = assessment_catalog._repair_catalog_numerical_contract(task)
    assert repaired == task
    assert diagnostics == []


def test_assessment_instance_identity_rejects_identical_not_distinct_values():
    assignment = {"task_id": "tip", "task_operation": "tip", "target": "total"}
    one = assessment_instance_fingerprint(
        {**assignment, "effective_givens": ["60", "17%"]},
        {"question": "Bill 60, tip 17%", "correct_answer": "70.2"},
    )
    identical = assessment_instance_fingerprint(
        {**assignment, "effective_givens": ["60", "17%"]},
        {"question": "Bill 60, tip 17%", "correct_answer": "70.2"},
    )
    distinct = assessment_instance_fingerprint(
        {**assignment, "effective_givens": ["80", "15%"]},
        {"question": "Bill 80, tip 15%", "correct_answer": "92"},
    )
    assert one == identical
    assert one != distinct


def test_partial_duplicate_group_keeps_only_unchanged_edge():
    q1 = {"question_id": "q1", "slot_id": "s1", "question": "changed"}
    q2 = {"question_id": "q2", "slot_id": "s2", "question": "same tip"}
    q8 = {"question_id": "q8", "slot_id": "s8", "question": "same tip"}
    exam = {1: {"model_number": 1, "questions": {"mcq": [q1, q2, q8]}}}
    from app.online.lifecycle import question_fingerprint
    group = {
        "model_number": 1, "group_id": "tips",
        "question_ids": ["q1", "q2", "q8"],
        "keep_question_ids": ["q8"], "replace_question_ids": ["q1", "q2"],
        "question_fingerprints": {
            "q1": "old-fingerprint", "q2": question_fingerprint(q2),
            "q8": question_fingerprint(q8),
        },
        "edges": [
            {"left": "q1", "right": "q2", "classification": "duplicate"},
            {"left": "q1", "right": "q8", "classification": "duplicate"},
            {"left": "q2", "right": "q8", "classification": "duplicate"},
        ],
    }
    current = duplicate_pipeline._current_conflict_groups([group], exam)
    assert current[0]["question_ids"] == ["q2", "q8"]
    assert current[0]["edges"] == [group["edges"][2]]


def test_finalization_pending_routes_through_required_fingerprint_verification():
    assert _route_after_validation({
        "repair_manifest": [], "duplicate_repair_finalization_pending": True,
        "duplicate_verification_pending": True,
    }) == "scan_exam_duplicates"
    assert _route_after_validation({
        "repair_manifest": [], "duplicate_repair_finalization_pending": True,
        "duplicate_verification_pending": False,
    }) == "assemble_exams"
