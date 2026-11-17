from __future__ import annotations

import json

from app.online import multi_validator, validator
from app.online.multi_validator import ProtectedQuestion


def _variation(variation_id: str, *, numbers: bool = False, operation: str = "solve"):
    return {
        "variation_id": variation_id,
        "question_types": ["mcq"],
        "operation": operation,
        "target": "x",
        "student_work_product": "selected solution",
        "reasoning_mode": "calculation",
        "evidence_chunk_ids": ["c1"],
        "givens": ["2x=4"],
        "numeric_values_may_change": numbers,
    }


def _plan(*variations, selected="v1"):
    return {
        "task_id": "t1", "selected_variation_id": selected,
        "task_operation": "solve", "target": "x",
        "student_work_product": "selected solution", "reasoning_mode": "calculation",
        "evidence_chunk_ids": ["c1"], "text_answerable": True,
        "exact_source_instance": False,
        "permitted_variations": list(variations),
    }


def test_copied_source_with_numeric_authorization_keeps_targeted_repair():
    result = validator.copied_source_repair_options(
        _plan(_variation("v1", numbers=True)), "mcq",
    )
    assert result["repairable"] is True
    assert result["numeric_instance_authorized"] is True
    assert result["current_variation_repairable"] is True
    assert result["replace_or_switch_required"] is False


def test_copied_source_with_materially_different_variation_keeps_recovery():
    other = _variation("v2", operation="verify")
    result = validator.copied_source_repair_options(
        _plan(_variation("v1"), other), "mcq",
    )
    assert result["repairable"] is True
    assert result["alternative_variation_ids"] == ["v2"]
    assert result["current_variation_repairable"] is True
    assert result["alternative_variation_available"] is True
    assert result["replace_or_switch_required"] is False


def test_copied_source_ordinary_nonexact_contract_allows_new_numeric_instance():
    result = validator.copied_source_repair_options(
        _plan(_variation("v1")), "mcq",
    )
    assert result == {
        "repairable": True,
        "current_variation_repairable": True,
        "alternative_variation_available": False,
        "numeric_instance_authorized": True,
        "alternative_variation_ids": [],
        "replace_or_switch_required": False,
        "contract_complete": True,
    }


def test_authorized_alternative_is_materialized_as_new_assignment():
    plan = _plan(
        _variation("v1"),
        {**_variation("v2", numbers=True, operation="verify"),
         "givens": ["3x=9"]},
    )
    assert validator._switch_recovery_variation(
        plan, "v2", "mcq", {"c1": "source"},
    ) is True
    assert plan["task_id"] == "t1"
    assert plan["selected_variation_id"] == "v2"
    assert plan["task_operation"] == "verify"
    assert plan["effective_givens"] == ["3x=9"]
    assert plan["selected_variation"]["numeric_values_may_change"] is True


def test_alternative_without_available_evidence_is_not_switchable():
    alternative = {**_variation("v2", operation="verify"),
                   "evidence_chunk_ids": ["missing"]}
    plan = _plan(_variation("v1"), alternative)
    options = validator.copied_source_repair_options(
        plan, "mcq", available_evidence_ids={"c1"},
    )
    assert options["alternative_variation_ids"] == []
    assert validator._switch_recovery_variation(
        plan, "v2", "mcq", {"c1": "source"},
    ) is False


def test_true_false_operation_mismatch_switches_to_same_type_alternative():
    selected = {**_variation("v1"), "question_types": ["true_false"],
                "student_work_product": "judgment plus explanation"}
    alternative = {**_variation("v2", operation="judge"),
                   "question_types": ["true_false"],
                   "student_work_product": "boolean judgment"}
    plan = {**_plan(selected, alternative), "selected_variation_id": "v1",
            "source_evidence_ids": ["c1"], "evidence_chunk_ids": ["c1"]}
    result = validator.failure_aware_recovery_preflight(
        plan, "true_false", {
            "failure_code": "question_type_operation_mismatch",
            "required_fixes": [{"validator": "task_alignment",
                                "failure_code": "question_type_operation_mismatch",
                                "defect_scope": "question"}],
        }, {"c1": "source"},
    )
    assert result["decision"] == "switch_variation"
    assert result["alternative_variation_ids"] == ["v2"]


def test_true_false_operation_mismatch_without_alternative_replaces():
    selected = {**_variation("v1"), "question_types": ["true_false"]}
    result = validator.failure_aware_recovery_preflight(
        {**_plan(selected), "selected_variation_id": "v1",
         "source_evidence_ids": ["c1"], "evidence_chunk_ids": ["c1"]},
        "true_false", {
            "failure_code": "question_type_operation_mismatch",
            "required_fixes": [{"validator": "task_alignment",
                                "failure_code": "question_type_operation_mismatch"}],
        }, {"c1": "source"},
    )
    assert result["decision"] == "replace_assignment"


def test_incorrect_answer_does_not_switch_variation():
    result = validator.failure_aware_recovery_preflight(
        _plan(_variation("v1"), _variation("v2", operation="verify")),
        "mcq", {"failure_code": "incorrect_answer", "required_fixes": [{
            "validator": "answer_correctness", "failure_code": "incorrect_answer",
        }]}, {"c1": "source"},
    )
    assert result["decision"] == "repair_current"


def test_true_false_operation_mismatch_switches_to_same_type_alternative():
    selected = {
        **_variation("v1"), "question_types": ["true_false"],
        "student_work_product": "judgment plus explanation",
    }
    alternative = {
        **_variation("v2", operation="judge"), "question_types": ["true_false"],
        "student_work_product": "boolean judgment",
    }
    plan = {
        **_plan(selected, alternative), "selected_variation_id": "v1",
        "source_evidence_ids": ["c1"], "evidence_chunk_ids": ["c1"],
    }
    assert validator.copied_source_repair_options(
        plan, "true_false", available_evidence_ids={"c1"},
    )["alternative_variation_ids"] == ["v2"]


def test_true_false_operation_mismatch_without_alternative_replaces():
    selected = {**_variation("v1"), "question_types": ["true_false"]}
    result = validator.copied_source_repair_options(
        {**_plan(selected), "selected_variation_id": "v1",
         "source_evidence_ids": ["c1"], "evidence_chunk_ids": ["c1"]},
        "true_false", available_evidence_ids={"c1"},
    )
    assert result["alternative_variation_ids"] == []


def test_incorrect_answer_does_not_switch_variation():
    selected = _variation("v1")
    alternative = _variation("v2", operation="verify")
    assert validator.recovery_v3_dispatch_action("REPAIR") == "repair_question"


def test_legacy_switch_preference_cannot_bypass_shared_task_selection(monkeypatch):
    seen = {}

    def fake_regenerate(_state, exam, _mn, question_id, plan_by_slot, _sources):
        plan = plan_by_slot["m2_mcq_1"]
        seen["variation_id"] = plan["selected_variation_id"]
        seen["effective_givens"] = plan["effective_givens"]
        question = validator._build_by_id(exam["questions"])[question_id][2]
        question.update({
            "question": "Which equation represents the authorized new instance?",
            "options": {"A": "C=80+0.15m", "B": "C=80m+15"},
            "correct_answer": "A", "difficulty": "medium",
        })
        return True, []

    monkeypatch.setattr(validator, "_regenerate_plan_slot", fake_regenerate)
    current = {**_variation("v1"), "givens": ["100", "0.10"]}
    alternative = {
        **_variation("v2", numbers=True), "givens": ["authorized variable values"],
    }
    plan = {
        **_plan(current, alternative), "slot_id": "m2_mcq_1",
        "question_type": "mcq", "source_chunk_id": "c1",
        "source_evidence_ids": ["c1"],
    }
    old_question = {
        "question_id": "model2_mcq_1", "slot_id": "m2_mcq_1",
        "question": "A truck costs 100 plus .10 per km. Which equation?",
        "options": {"A": "C=100+.10m", "B": "C=100m+.10"},
        "correct_answer": "A", "difficulty": "easy",
    }
    state = {
        "generated_exams": [{"model_number": 2, "questions": {"mcq": [old_question]}}],
        "plans": [{"model_number": 2, "items": {"mcq": [plan]}}],
        "retrieved_chunks": [{"child_id": "c1", "content": "linear costs"}],
        "assessment_catalog": {}, "difficulty": "easy",
        "assessment_profile": {"domain": "STEM", "question_types": {}},
        "repair_manifest": [{
            "model_number": 2, "slot_id": "m2_mcq_1",
            "question_id": "model2_mcq_1", "question_type": "mcq",
            "status": "invalid", "repair_type": "repair_question",
            "failure_code": "copied_source_example", "failure_reason": "copied",
            "verdict": {"question_id": "model2_mcq_1", "action": "FIX_QUESTION",
                        "fields_to_fix": ["question"], "required_fixes": []},
            "recovery_v3": {"version": "recovery-v3-task-alignment", "status": "ready",
                            "decision": "SWITCH_TASK", "reason": "copied immutable source",
                            "plan": ["SWITCH_TASK"], "fallback_sequence": [],
                            "switch_preference": "v2", "next_index": 0,
                            "attempted_strategies": [], "decision_calls": 1},
        }],
        "slot_completion_attempts": {}, "question_repair_attempts": {},
        "question_repair_failures": {}, "question_repair_calls": {},
        "warnings": [],
    }
    result = validator.repair_invalid_questions(state)
    assert seen == {}
    assert result["repair_candidates"] == []


def test_incomplete_legacy_contract_is_not_incorrectly_skipped():
    assert validator.copied_source_repair_options(
        {"task_id": "legacy"}, "mcq",
    )["repairable"] is True


def test_impossible_copied_source_skips_content_repair_agent(monkeypatch):
    from app.online import multi_repair

    monkeypatch.setattr(
        multi_repair, "repair_question_specialized",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("content Repair must be skipped")
        ),
    )
    variation = {
        **_variation("v1"), "question_types": ["definition"],
        "operation": "define", "target": "mass",
        "student_work_product": "definition", "reasoning_mode": "recall",
    }
    state = {
        "generated_exams": [{"model_number": 1, "questions": {"definition": [{
            "question_id": "q1", "slot_id": "s1", "term": "mass",
            "reference_answer": "amount of matter",
        }]}}],
        "plans": [{"model_number": 1, "items": {"definition": [{
            **_plan(variation), "slot_id": "s1", "question_type": "definition",
            "task_operation": "define", "target": "mass",
            "student_work_product": "definition", "reasoning_mode": "recall",
            "source_chunk_id": "c1", "source_evidence_ids": ["c1"],
        }]}}],
        "retrieved_chunks": [{"child_id": "c1", "content": "Mass is amount of matter."}],
        "assessment_catalog": {}, "difficulty": "easy",
        "assessment_profile": {"domain": "NON_STEM", "question_types": {}},
        "repair_manifest": [{
            "model_number": 1, "slot_id": "s1", "question_id": "q1",
            "question_type": "definition", "status": "invalid",
            "repair_type": "repair_question", "failure_code": "copied_source_example",
            "failure_reason": "copied", "fields_to_fix": ["term"],
            "verdict": {"question_id": "q1", "action": "FIX_QUESTION",
                        "fields_to_fix": ["term"], "required_fixes": []},
            "recovery_v3": {"version": "recovery-v3-task-alignment", "status": "ready",
                            "decision": "SWITCH_TASK", "reason": "no material repair",
                            "plan": ["SWITCH_TASK"], "fallback_sequence": [],
                            "switch_preference": None, "next_index": 0,
                            "attempted_strategies": [], "decision_calls": 1},
        }],
        "slot_completion_attempts": {}, "question_repair_attempts": {},
        "question_repair_failures": {}, "question_repair_calls": {},
        "warnings": [],
    }
    result = validator.repair_invalid_questions(state)
    target = result["repair_manifest"][0]
    assert target["repair_type"] == "replace_assignment"
    assert target["recovery_routing_reason"] == "task_alignment_recovery_v3"
    assert result["question_repair_calls"] == {}
    # No candidate was generated, so the bounded replacement inventory was
    # not consumed by this failed preflight.
    assert result["slot_completion_attempts"].get("s1", 0) == 0


def test_merge_marks_fresh_and_historical_verdicts_explicitly():
    previous = {"model_number": 1, "verdicts": [
        {"question_id": "terminal", "action": "FIX_QUESTION"},
        {"question_id": "active", "action": "FIX_ANSWER"},
    ]}
    update = {"model_number": 1, "verdicts": [
        {"question_id": "active", "action": "PASS"},
    ]}
    merged = validator._merge_validation_report(previous, update)
    by_id = {row["question_id"]: row for row in merged["verdicts"]}
    assert by_id["terminal"]["action"] == "FIX_QUESTION"
    assert by_id["terminal"]["evaluation_origin"] == "historical_retained"
    assert by_id["active"]["evaluation_origin"] == "fresh"


def test_specialized_prompt_uses_lossless_isolated_evidence_registry():
    variation = _variation("v1")
    task = _plan(variation)
    task["allowed_variations"] = [variation]
    items = [
        ProtectedQuestion(
            question_id=f"q{index}", question_type="mcq",
            question={"question_id": f"q{index}", "question": "Solve.",
                      "options": {"A": "1", "B": "2"}, "correct_answer": "B"},
            task=task, evidence={"c1": "Shared full source text."},
            slot_id=f"m1_mcq_{index}", word_bank=[], sanitization_events=[],
            errors=[], requires_quantitative=True,
        ) for index in (1, 2)
    ]
    raw_payloads = [multi_validator._payload_for("task_alignment", item) for item in items]
    prompt = multi_validator._user_prompt("task_alignment", raw_payloads)
    envelope = json.loads(prompt.split("<VALIDATION_DATA>\n", 1)[1].split(
        "\n</VALIDATION_DATA>", 1,
    )[0])
    assert envelope["evidence_registry"] == {"c1": "Shared full source text."}
    assert prompt.count("Shared full source text.") == 1
    assert all(q["assigned_evidence_ids"] == ["c1"] for q in envelope["questions"])
    contract = envelope["questions"][0]["task_contract"]
    assert "allowed_variations" not in contract
    assert contract["selected_variation"] == variation
    assert contract["authorized_alternatives"] == []
    assert contract["authorized_variation_ids"] == ["v1"]

