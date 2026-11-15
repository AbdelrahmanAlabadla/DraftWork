"""Group 4 recovery-context and replacement-state regressions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.online import multi_repair
from app.online import graph as pipeline_graph
from app.online.assessment_catalog import (
    catalog_exhaustion_replacement,
    eligible_task_variation_candidates,
    eligible_task_variations,
)
from app.online.difficulty_adjustment import (
    PlanDifficultyAdjustmentDecision,
    _acceptance_diagnostics,
)
from app.online.eval_stats import (
    create_pipeline_eval,
    finalize_pipeline_eval,
    record_repair_outcome,
)
from app.online.validator import (
    _apply_replacement_difficulty_design,
    _defect_signature,
    _mark_replacement_difficulty_blocked,
    _normalize_replacement_difficulty_state,
    _replacement_difficulty_unresolved,
    _safe_record_repair_outcome,
)

CATALOG = Path(
    r"C:\Users\abdul\PycharmProjects\GenExam\data\assessment_catalogs\ed"
    r"\edd2a703d9b6712ab2cdce92c593657042a43d704fa2cdeb845fe71d6249a439.json"
)


def _exam() -> dict:
    return {"questions": {"mcq": [{
        "question_id": "q1", "slot_id": "s1",
        "question": "A tree is 48 ft tall. Which band?",
        "options": {"A": "short", "B": "medium"}, "correct_answer": "B",
        "solution_steps": [], "solution_certificate": None,
    }]}}


def _plan() -> dict:
    return {
        "slot_id": "s1", "task_id": "t1", "selected_variation_id": "v1",
        "task_operation": "classify", "target": "band",
        "required_givens": ["height in ft"], "effective_givens": ["48 ft"],
        "subject": "mathematics", "exact_source_instance": True,
        "variable_givens": [], "source_evidence_ids": ["c1"],
        "permitted_variations": [{
            "variation_id": "v1", "numeric_values_may_change": False,
        }],
    }


def test_repair_payload_contains_source_constraints_and_repeated_defect(monkeypatch):
    captured: list[dict] = []

    def fake_batch(requests):
        for request in requests:
            captured.append(request.payload)
            request.result = (None, "blocked", {})
            request.done.set()

    monkeypatch.setattr(multi_repair, "_execute_batch", fake_batch)
    verdict = {
        "question_id": "q1", "validation_role": "answer_correctness",
        "failure_code": "answer_inconsistent",
        "fields_to_fix": ["correct_answer"],
        "reason": "The source height is 48 ft but the certificate uses 48 m.",
        "expected_fix": "Use 48 ft and recompute the band.",
        "required_fixes": [{
            "validator": "answer_correctness", "failure_code": "answer_inconsistent",
            "fields_to_fix": ["correct_answer"],
            "reason": "The source height is 48 ft but the certificate uses 48 m.",
            "expected_fix": "Use 48 ft and recompute the band.",
        }],
    }
    multi_repair.repair_question_specialized(
        exam=_exam(), verdict=verdict, qtype="mcq", plan_item=_plan(),
        source_chunks={"c1": "The tree is 48 ft tall."}, domain="STEM",
        repeated_defect={"failure_code": "unsupported", "failure_reason": "same source defect"},
    )
    payload = captured[0]
    assert payload["findings"][0]["reason"] == verdict["reason"]
    assert payload["repeated_defect"]["failure_code"] == "unsupported"
    assert payload["source_constraint_summary"]["exact_source_instance"] is True
    assert payload["source_constraint_summary"]["numeric_values_may_change"] is False
    assert payload["unit_policy"] == "essential"
    assert "unit_policy_instruction" in payload


def test_defect_signature_ignores_wording_but_not_defect_family():
    unsupported = {
        "validation_role": "grounding", "failure_code": "unsupported",
        "fields_to_fix": ["question"],
    }
    contradicted = {
        "validation_role": "grounding", "failure_code": "contradicted",
        "fields_to_fix": ["question"],
    }
    answer = {
        "validation_role": "answer_correctness", "failure_code": "answer_inconsistent",
        "fields_to_fix": ["question"],
    }
    assert _defect_signature(unsupported) == _defect_signature(contradicted)
    assert _defect_signature(unsupported) != _defect_signature(answer)


def test_replacement_difficulty_is_pending_until_design_runs():
    replacement = {
        "difficulty": "easy", "native_difficulty_prior": "easy",
        "requested_difficulty": "hard",
        "difficulty_design": {"status": "pending", "native_prior": "easy"},
        "difficulty_adaptation": {"status": "pending"},
    }
    _normalize_replacement_difficulty_state(replacement, "hard")
    assert replacement["requested_difficulty"] == "hard"
    assert replacement["actual_difficulty"] == "easy"
    assert replacement["difficulty_design"]["status"] == "pending"
    assert replacement["difficulty_status"] == "adaptation_pending"
    assert replacement["difficulty_adaptation"] is None


def test_replacement_design_can_explicitly_fall_back_to_source_constraint(monkeypatch):
    item = {
        "slot_id": "s1", "task_id": "t1", "difficulty": "easy",
        "native_difficulty_prior": "easy", "requested_difficulty": "hard",
        "difficulty_design": {"status": "pending", "native_prior": "easy"},
    }
    plans = [{"model_number": 1, "items": {"mcq": [item]}}]

    def fake_adjust(state):
        adjusted = deepcopy(state["plans"])
        target = adjusted[0]["items"]["mcq"][0]
        target["difficulty_design"] = {
            "status": "accepted_source_constraint", "native_prior": "easy",
            "requested": "hard", "reason": "source teaches only direct recall",
        }
        target["difficulty_status"] = "accepted_source_constraint"
        return {"plans": adjusted, "difficulty_adjustment_history": []}

    monkeypatch.setattr(
        "app.online.difficulty_adjustment.adjust_plan_difficulty", fake_adjust,
    )
    result = _apply_replacement_difficulty_design(
        plans, model_number=1, qtype="mcq", slot_id="s1",
        state={"difficulty": "hard"}, requested="hard",
    )
    assert result["difficulty_design"]["status"] == "accepted_source_constraint"
    assert result["difficulty_status"] == "accepted_source_constraint"
    assert result["requested_difficulty"] == "hard"


def test_replacement_design_runs_and_can_design_hard(monkeypatch):
    item = {
        "slot_id": "s1", "task_id": "t1", "difficulty": "easy",
        "native_difficulty_prior": "easy", "requested_difficulty": "hard",
        "difficulty_design": {"status": "pending", "native_prior": "easy"},
    }
    plans = [{"model_number": 1, "items": {"mcq": [item]}}]

    def fake_adjust(state):
        adjusted = deepcopy(state["plans"])
        target = adjusted[0]["items"]["mcq"][0]
        target["difficulty_design"] = {
            "status": "designed", "native_prior": "easy", "requested": "hard",
            "policy_version": "difficulty-design-v1",
        }
        target["difficulty_status"] = "designed"
        target["actual_difficulty"] = "hard"
        return {"plans": adjusted, "difficulty_adjustment_history": []}

    monkeypatch.setattr(
        "app.online.difficulty_adjustment.adjust_plan_difficulty", fake_adjust,
    )
    result = _apply_replacement_difficulty_design(
        plans, model_number=1, qtype="mcq", slot_id="s1",
        state={"difficulty": "hard"}, requested="hard",
    )
    assert result["difficulty_design"]["status"] == "designed"
    assert result["difficulty_status"] == "designed"
    assert result["actual_difficulty"] == "hard"



def test_replacement_adjuster_exception_is_not_source_constrained(monkeypatch):
    item = {
        "slot_id": "s1", "task_id": "t1", "difficulty": "easy",
        "native_difficulty_prior": "easy", "requested_difficulty": "hard",
        "difficulty_design": {"status": "pending", "native_prior": "easy"},
    }
    plans = [{"model_number": 1, "items": {"mcq": [item]}}]
    monkeypatch.setattr(
        "app.online.difficulty_adjustment.adjust_plan_difficulty",
        lambda _state: (_ for _ in ()).throw(TimeoutError("provider timeout")),
    )
    result = _apply_replacement_difficulty_design(
        plans, model_number=1, qtype="mcq", slot_id="s1",
        state={"difficulty": "hard"}, requested="hard",
    )
    assert result["difficulty_design"]["status"] == "pending"
    assert result["difficulty_status"] == "adaptation_pending"
    assert result["difficulty_status"] != "accepted_source_constraint"
    assert "TimeoutError" in result["difficulty_design_operational_error"]
    assert _replacement_difficulty_unresolved(result) is True
    target = {}
    _mark_replacement_difficulty_blocked(target, result)
    assert target["status"] == "validation_blocked"
    assert target["failure_code"] == "replacement_difficulty_operational_error"


def test_replacement_adjuster_resume_retries_pending_design(monkeypatch):
    item = {
        "slot_id": "s1", "task_id": "t1", "difficulty": "easy",
        "native_difficulty_prior": "easy", "requested_difficulty": "hard",
        "difficulty_design": {"status": "pending", "native_prior": "easy"},
        "difficulty_status": "adaptation_pending",
    }
    plans = [{"model_number": 1, "items": {"mcq": [item]}}]
    calls = []

    def fake_adjust(state):
        calls.append(1)
        adjusted = deepcopy(state["plans"])
        adjusted[0]["items"]["mcq"][0]["difficulty_design"] = {
            "status": "designed", "native_prior": "easy", "requested": "hard",
        }
        adjusted[0]["items"]["mcq"][0]["difficulty_status"] = "designed"
        return {"plans": adjusted, "difficulty_adjustment_history": []}

    monkeypatch.setattr(
        "app.online.difficulty_adjustment.adjust_plan_difficulty", fake_adjust,
    )
    result = _apply_replacement_difficulty_design(
        plans, model_number=1, qtype="mcq", slot_id="s1",
        state={"difficulty": "hard"}, requested="hard",
    )
    assert calls == [1]
    assert result["difficulty_design"]["status"] == "designed"
    assert result["difficulty_status"] == "designed"
    assert result["difficulty_status"] != "accepted_source_constraint"


def test_valid_noop_and_adjustment_kind_are_accepted():
    noop = PlanDifficultyAdjustmentDecision.model_validate({
        "slot_id": "m1_mcq_1",
        "adjustment_succeeded": True,
        "achieved_difficulty": "easy",
        "estimated_conceptual_difficulty": "easy",
        "estimated_numerical_difficulty": "easy",
        "further_adjustment_possible": False,
        "attempted_fields": [],
        "adjusted_fields": None,
        "evidence_support": ["c1"],
        "evidence_justification": "Already at the requested level.",
        "reason": "No change required.",
    })
    assert noop.no_change_required is True

    changed = PlanDifficultyAdjustmentDecision.model_validate({
        "slot_id": "m1_mcq_1",
        "adjustment_succeeded": False,
        "achieved_difficulty": "medium",
        "further_adjustment_possible": False,
        "attempted_fields": ["numeric_values", "adjustment_kind"],
        "adjusted_fields": None,
        "evidence_support": ["c1"],
        "evidence_justification": "Numeric variation was considered but not permitted.",
        "reason": "No safer adjustment.",
    })
    assert changed.attempted_fields == ["numeric_values", "adjustment_kind"]
    assert changed.no_change_required is False


def test_noop_decision_is_accepted_as_not_required():
    decision = PlanDifficultyAdjustmentDecision.model_validate({
        "slot_id": "m1_mcq_1",
        "adjustment_succeeded": True,
        "achieved_difficulty": "easy",
        "further_adjustment_possible": False,
        "attempted_fields": [],
        "adjusted_fields": None,
        "evidence_support": ["c1"],
        "evidence_justification": "Already at the requested level.",
        "reason": "No change required.",
    })
    checks = _acceptance_diagnostics(
        decision,
        {"slot_id": "m1_mcq_1", "difficulty_design": {"status": "pending"}},
        {"slot_id": "m1_mcq_1", "difficulty_design": {"status": "pending"}},
        qtype="mcq", requested="easy", evidence_ids=["c1"],
    )
    assert checks["no_change_required"] is True
    assert checks["meaningful_change_present"] is True


def test_safe_repair_telemetry_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        "app.online.validator.record_repair_outcome",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("telemetry down")),
    )
    _safe_record_repair_outcome(
        {}, {}, {}, repaired_ids=[], failed_ids=[], sent_ids=[],
    )


def test_progress_and_audit_callback_failures_do_not_abort_node():
    def node(_state):
        return {"ok": True}

    node.__name__ = "validate_generated_questions"
    wrapped = pipeline_graph._with_progress(node, "general_validation", 95)
    state = {
        "job_id": "callback-regression",
        "audit_callback": lambda *_a: (_ for _ in ()).throw(RuntimeError("audit down")),
        "progress_callback": lambda *_a: (_ for _ in ()).throw(RuntimeError("progress down")),
    }
    assert wrapped(state)["ok"] is True


def test_rejected_repair_and_resume_do_not_double_count():
    stats = create_pipeline_eval([("mcq", 2)], 1)
    exam = {"model_number": 1, "questions": {"mcq": [
        {"question_id": "q1", "slot_id": "s1", "question": "Q1?",
         "options": {"A": "a", "B": "b"}, "correct_answer": "A"},
        {"question_id": "q2", "slot_id": "s2", "question": "Q2?",
         "options": {"A": "a", "B": "b"}, "correct_answer": "A"},
    ]}}
    report = {"model_number": 1, "verdicts": [
        {"question_id": "q1", "action": "PASS"},
        {"question_id": "q2", "action": "PASS"},
    ]}
    record_repair_outcome(
        stats, exam, report,
        repaired_ids=["q1"], failed_ids=["q2"], sent_ids=["q1", "q2"],
    )
    # Checkpoint/resume may replay the same outcome; it must not add a second
    # entry for the same question IDs.
    record_repair_outcome(
        stats, exam, report,
        repaired_ids=["q1"], failed_ids=["q2"], sent_ids=["q1", "q2"],
    )
    finalize_pipeline_eval(stats, [exam], [report])
    assert stats["overall"]["repair_sent"] == 2
    assert stats["overall"]["repair_succeeded"] == 1
    assert stats["overall"]["repair_failed"] == 1


def test_repair_outcome_is_counted_in_existing_eval_stats():
    stats = create_pipeline_eval([("mcq", 1)], 1)
    exam = {"model_number": 1, "questions": {"mcq": [{
        "question_id": "q1", "slot_id": "s1", "question": "Q?",
        "options": {"A": "a", "B": "b"}, "correct_answer": "A",
    }]}}
    report = {"model_number": 1, "verdicts": [{
        "question_id": "q1", "action": "PASS",
    }]}
    record_repair_outcome(
        stats, exam, report, repaired_ids=["q1"], failed_ids=[], sent_ids=["q1"],
    )
    finalize_pipeline_eval(stats, [exam], [report])
    assert stats["overall"]["repair_sent"] == 1
    assert stats["overall"]["repair_succeeded"] == 1
    assert stats["overall"]["repair_failed"] == 0



@pytest.mark.skipif(not CATALOG.exists(), reason="persisted catalog unavailable")
def test_same_task_preference_cannot_bypass_shared_rank():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    task = next(
        value for value in catalog["tasks"]
        if len(eligible_task_variation_candidates(value, "mcq")) >= 2
    )
    variations = eligible_task_variation_candidates(task, "mcq")
    current_variation = str(variations[0]["variation_id"])
    alternative_variation = str(variations[1]["variation_id"])
    plans = [{"model_number": 1, "items": {"mcq": [{
        "slot_id": "m1_mcq_1", "task_id": task["task_id"],
        "selected_variation_id": current_variation,
        "question_type": "mcq",
    }]}}]
    replacement = catalog_exhaustion_replacement(
        catalog, plans, "m1_mcq_1", "mcq", "easy",
        active_slot_ids={"m1_mcq_1"},
        prefer_task_id=task["task_id"],
        exclude_variation_ids={current_variation},
    )
    assert replacement is not None
    assert replacement["task_id"] != task["task_id"]
    assert replacement["catalog_replacement_stage"].startswith("shared_rank_")

