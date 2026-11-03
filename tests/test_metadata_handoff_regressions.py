from __future__ import annotations

from app.llm.schemas import TrueFalseQuestion
from app.online.assessment_catalog import materialize_plan_item, permitted_task_variations
from app.online.duplicate_pipeline import _replace_plan_item
from app.online.models import _normalize_true_false, normalize_true_false_answer, tf_answer_label
from app.online.multi_repair import RepairStep, _apply_step
from app.online.multi_validator import protect_question
from app.online.validator import _canonical_changes, _plan_evidence_consistency_error


def test_true_false_normalization_is_strict_boolean_and_legacy_compatible():
    assert normalize_true_false_answer("True") is True
    assert normalize_true_false_answer("False") is False
    assert normalize_true_false_answer(True) is True
    assert normalize_true_false_answer(False) is False
    assert normalize_true_false_answer("yes") is None
    assert normalize_true_false_answer(1) is None
    assert _normalize_true_false({"statement": "S", "answer": "False"})["answer"] is False

    parsed = TrueFalseQuestion.model_validate({
        "slot_id": "m1_true_false_1", "statement": "S", "answer": "False",
    })
    assert _normalize_true_false(parsed.model_dump())["answer"] is False


def test_true_false_validation_repair_and_arabic_rendering_preserve_false():
    protected = protect_question(
        "true_false",
        {"question_id": "q1", "slot_id": "s1", "statement": "S", "answer": "False"},
        {"slot_id": "s1", "source_chunk_id": "c1"},
        {"c1": "S is false."},
    )
    assert protected.errors == []
    assert protected.question["answer"] is False

    issue = {"issue_id": "i1", "failure_code": "incorrect_answer"}
    step = RepairStep(agent="answer", issues=(issue,), fields=("answer",))
    repaired, metadata, error = _apply_step(
        "true_false",
        {"question_id": "q1", "slot_id": "s1", "statement": "S", "answer": True},
        step,
        {
            "patch": {"answer": "False"},
            "repair_metadata": {
                "modified_fields": ["answer"],
                "changes": [{
                    "field": "answer", "previous_value": "true", "new_value": "false",
                    "reason": "Correct the key.",
                }],
                "targeted_failure_codes": ["incorrect_answer"],
            },
        },
    )
    assert error is None
    assert repaired["answer"] is False
    assert metadata["changes"][0]["previous_value"] is True
    assert metadata["changes"][0]["new_value"] is False
    assert tf_answer_label(False, "en") == "False"
    assert tf_answer_label(False, "ar") != tf_answer_label(True, "ar")


def test_replacement_materialization_transfers_variation_evidence_atomically():
    task = {
        "task_id": "task-new", "task_family_id": "family-new", "coverage_id": "coverage-new",
        "evidence_chunk_ids": ["chunk-parent"], "allowed_question_types": ["equation"],
        "primary_question_types": ["equation"], "text_answerable": True,
        "task_operation": "calculate", "target": "velocity", "student_work_product": "result",
        "reasoning_mode": "calculation", "required_givens": ["u", "a", "t"],
        "supported_difficulties": ["medium"], "native_difficulty": "medium",
        "allowed_variations": [{
            "variation_id": "variation-new", "variation_group_id": "group-new",
            "target": "final velocity", "operation": "calculate", "givens": ["u", "a", "t"],
            "question_types": ["equation"], "evidence_chunk_ids": ["chunk-variation"],
            "reasoning_mode": "calculation", "student_work_product": "result",
        }],
    }
    selected_variation_id = next(
        value["variation_id"] for value in permitted_task_variations(task, "equation")
        if value.get("evidence_chunk_ids") == ["chunk-variation"]
    )
    replacement = materialize_plan_item(
        task, "m1_equation_1", "medium", selected_variation_id=selected_variation_id,
    )
    assert replacement["source_chunk_id"] == "chunk-variation"
    assert replacement["support_chunk_ids"] == []
    assert replacement["source_evidence_ids"] == ["chunk-variation"]
    assert _plan_evidence_consistency_error(
        replacement, {"chunk-variation": "v = u + at"},
    ) is None

    plans = [{"model_number": 1, "items": {"equation": [{
        "slot_id": "m1_equation_1", "task_id": "task-old",
        "source_chunk_id": "chunk-old", "support_chunk_ids": [],
    }]}}]
    assert _replace_plan_item(plans, 1, "m1_equation_1", replacement)
    committed = plans[0]["items"]["equation"][0]
    assert committed["task_id"] == "task-new"
    assert committed["source_evidence_ids"] == ["chunk-variation"]
    assert "chunk-old" not in committed.values()


def test_replacement_preflight_rejects_stale_or_unavailable_evidence():
    plan = {
        "selected_variation_id": "v1",
        "source_chunk_id": "old", "support_chunk_ids": [],
        "source_evidence_ids": ["old"],
        "permitted_variations": [{"variation_id": "v1", "evidence_chunk_ids": ["new"]}],
    }
    error = _plan_evidence_consistency_error(plan, {"old": "old", "new": "new"})
    assert "does not match" in error

    plan.update(source_chunk_id="new", source_evidence_ids=["new"])
    assert "unavailable" in _plan_evidence_consistency_error(plan, {})


def test_repair_content_diff_ignores_orchestrator_assignment_enrichment():
    before = {
        "question": "Original", "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "correct_answer": "B", "task_id": "task-1",
    }
    after = {
        **before, "question": "Repaired", "assignment_fingerprint": "fp",
        "assignment_revision": 2, "selected_variation_id": "variation-1",
        "effective_evidence_chunk_ids": ["chunk-1"],
    }
    assert [change["field"] for change in _canonical_changes("mcq", before, after)] == [
        "question"
    ]
