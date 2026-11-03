from __future__ import annotations

import json
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.online import multi_repair
from app.online.multi_repair import RepairStep, _apply_step, build_repair_route, repair_question_specialized
from app.online.multi_validator import ROLE_FAILURE_CODES
from app.online.multi_validator import ProtectedQuestion, _compatibility_verdict
from app.online.validator import _manifest_from_report, _merge_repair_manifest
from app.online.validator import repair_invalid_questions, validate_generated_questions


def _issue(role, code, fields, expected="fix"):
    return {
        "validator": role, "validation_status": "FAIL", "failure_code": code,
        "fields_to_fix": fields, "reason": f"{code} detected", "expected_fix": expected,
        "grounding_outcome": None, "observed_difficulty": None,
        "calculation_evidence": [],
    }


def _verdict(*issues):
    return {
        "question_id": "q1", "validation_status": "FAIL",
        "action": "FIX_QUESTION_AND_ANSWER", "failure_code": issues[0]["failure_code"],
        "fields_to_fix": sorted({field for issue in issues for field in issue["fields_to_fix"]}),
        "required_fixes": list(issues), "repair_instruction_conflicts": [],
    }


def test_precise_mcq_option_target_changes_only_named_choice():
    issue = _issue("answer_correctness", "ambiguous_option", ["options.D"])
    issue["issue_id"] = "i-option-d"
    step = RepairStep(agent="options", issues=(issue,), fields=("options",))
    current = {"question_id": "q1", "slot_id": "s1", "question": "Choose.",
               "options": {"A": "a", "B": "b", "C": "c", "D": "b"},
               "correct_answer": "B"}
    new_options = {"A": "a", "B": "b", "C": "c", "D": "d"}
    repaired, metadata, error = _apply_step("mcq", current, step, {
        "patch": {"options": new_options},
        "repair_metadata": {"modified_fields": ["options"], "changes": [{
            "field": "options", "previous_value": current["options"],
            "new_value": new_options, "reason": "Make D distinct.",
        }], "targeted_failure_codes": ["ambiguous_option"]},
    })
    assert error is None
    assert repaired["options"] == new_options
    assert metadata["modified_fields"] == ["options.D"]


def test_precise_mcq_option_target_rejects_other_choice_changes():
    issue = _issue("answer_correctness", "ambiguous_option", ["options.D"])
    issue["issue_id"] = "i-option-d"
    step = RepairStep(agent="options", issues=(issue,), fields=("options",))
    current = {"question_id": "q1", "slot_id": "s1", "question": "Choose.",
               "options": {"A": "a", "B": "b", "C": "c", "D": "b"},
               "correct_answer": "B"}
    new_options = {"A": "changed", "B": "b", "C": "c", "D": "d"}
    _repaired, _metadata, error = _apply_step("mcq", current, step, {
        "patch": {"options": new_options},
        "repair_metadata": {"modified_fields": ["options"], "changes": [{
            "field": "options", "previous_value": current["options"],
            "new_value": new_options, "reason": "Changed too much.",
        }], "targeted_failure_codes": ["ambiguous_option"]},
    })
    assert "unauthorized paths" in error


def test_mcq9_full_options_object_accepts_only_authorized_d_leaf():
    issue = _issue("answer_correctness", "ambiguous_option", ["options.D"])
    step = RepairStep(agent="options", issues=(issue,), fields=("options",))
    current = {
        "question_id": "model1_mcq_9", "slot_id": "m1_mcq_9",
        "question": "Which system represents the situation?",
        "options": {"A": "A", "B": "correct", "C": "C", "D": "duplicate B"},
        "correct_answer": "B", "task_id": "task-9",
        "selected_variation_id": "variation-9", "assignment_revision": 2,
    }
    proposed = {**current["options"], "D": "distinct incorrect distractor"}
    repaired, metadata, error = _apply_step("mcq", current, step, {
        "patch": {"options": proposed},
        "repair_metadata": {
            "modified_fields": ["options.D"],
            "changes": [{
                "field": "options.D", "previous_value": "a description of old D",
                "new_value": "distinct incorrect distractor", "reason": "Remove ambiguity.",
            }],
            "targeted_failure_codes": ["ambiguous_option"],
        },
    })
    assert error is None
    assert repaired["options"] == proposed
    assert repaired["correct_answer"] == "B"
    assert repaired["task_id"] == "task-9"
    assert repaired["assignment_revision"] == 2
    assert metadata["modified_fields"] == ["options.D"]


def test_tf5_atomic_patch_uses_python_original_certificate_not_reported_metadata():
    issue = _issue("grounding", "unsupported", ["statement"])
    step = RepairStep(
        agent="task_alignment", issues=(issue,),
        fields=("answer", "solution_certificate", "statement"),
    )
    old_certificate = {
        "method": "old substitution", "important_givens": ["(1,2,3)"],
        "calculation_steps": ["old step"], "final_answer": "True", "units": "",
        "matching_correct_option": None, "verification_method": "old tuple",
    }
    new_certificate = {
        "method": "substitution", "important_givens": ["(-1,-5,7)"],
        "calculation_steps": ["first equation fails"], "final_answer": "False", "units": "",
        "matching_correct_option": None,
        "verification_method": "substitute (-1,-5,7) into all displayed equations",
    }
    current = {
        "question_id": "model1_true_false_5", "slot_id": "m1_true_false_5",
        "statement": "The tuple (1,2,3) satisfies the displayed system.",
        "answer": True, "solution_certificate": old_certificate,
        "task_id": "task-tf5", "selected_variation_id": "variation-tf5",
        "effective_evidence_chunk_ids": ["chunk-tf5"], "assignment_revision": 2,
    }
    repaired, metadata, error = _apply_step("true_false", current, step, {
        "patch": {
            "statement": "The tuple (-1,-5,7) satisfies the following displayed system: ...",
            "answer": False, "solution_certificate": new_certificate,
        },
        "repair_metadata": {
            "modified_fields": ["statement", "answer", "solution_certificate"],
            "changes": [
                {"field": "statement", "previous_value": "old statement", "new_value": "new", "reason": "grounding"},
                {"field": "answer", "previous_value": "true", "new_value": "false", "reason": "consistency"},
                {"field": "solution_certificate", "previous_value": "certificate for old tuple", "new_value": "new certificate", "reason": "consistency"},
            ],
            "targeted_failure_codes": ["unsupported"],
        },
    })
    assert error is None
    assert repaired["solution_certificate"] == new_certificate
    certificate_changes = metadata["changes"]
    assert any(
        change["field"].startswith("solution_certificate.")
        and change["previous_value"] == "old tuple"
        for change in certificate_changes
    )
    assert repaired["task_id"] == "task-tf5"
    assert repaired["effective_evidence_chunk_ids"] == ["chunk-tf5"]


def test_recursive_json_diff_handles_arrays_add_remove_null_and_missing():
    changes = multi_repair._recursive_json_changes(
        {"a": [1, {"x": None}], "removed": 2},
        {"a": [1, {"x": 3}, 4], "added": None},
    )
    assert set(changes) == {"a[1].x", "a[2]", "added", "removed"}
    assert changes["a[1].x"]["previous_value"] is None
    assert changes["a[1].x"]["previous_missing"] is False
    assert changes["added"]["previous_missing"] is True
    assert changes["added"]["new_value"] is None
    assert changes["removed"]["new_missing"] is True


def test_assignment_metadata_tampering_is_rejected_before_staging():
    issue = _issue("answer_correctness", "incorrect_answer", ["reference_answer"])
    step = RepairStep(agent="answer", issues=(issue,), fields=("reference_answer",))
    current = {
        "question_id": "q1", "slot_id": "s1", "term": "mass",
        "reference_answer": "wrong", "task_id": "task-1",
    }
    _repaired, normalized_metadata, error = _apply_step("definition", current, step, {
        "patch": {"reference_answer": "correct", "task_id": "task-evil"},
        "repair_metadata": _metadata("reference_answer", "wrong", "correct", "incorrect_answer"),
    })
    assert "patch fields did not exactly match authorized fields" in error


def test_precise_option_repair_cannot_change_correct_answer_without_authorization():
    issue = _issue("answer_correctness", "ambiguous_option", ["options.D"])
    step = RepairStep(agent="options", issues=(issue,), fields=("options",))
    current = {
        "question_id": "q1", "slot_id": "s1", "question": "Choose.",
        "options": {"A": "a", "B": "b", "C": "c", "D": "b"},
        "correct_answer": "B",
    }
    _repaired, _metadata_result, error = _apply_step("mcq", current, step, {
        "patch": {
            "options": {**current["options"], "D": "d"},
            "correct_answer": "D",
        },
        "repair_metadata": _metadata("options.D", "b", "d", "ambiguous_option"),
    })
    assert "patch fields did not exactly match authorized fields" in error


def test_noop_patch_is_rejected_for_existing_no_progress_recovery():
    issue = _issue("answer_correctness", "incorrect_answer", ["reference_answer"])
    step = RepairStep(agent="answer", issues=(issue,), fields=("reference_answer",))
    current = {
        "question_id": "q1", "slot_id": "s1", "term": "mass",
        "reference_answer": "amount of matter",
    }
    _repaired, _metadata_result, error = _apply_step("definition", current, step, {
        "patch": {"reference_answer": "amount of matter"},
        "repair_metadata": _metadata(
            "reference_answer", "amount of matter", "amount of matter", "incorrect_answer",
        ),
    })
    assert error == "patch made no canonical change"


def test_precise_nested_certificate_path_does_not_authorize_siblings():
    issue = _issue(
        "answer_correctness", "answer_inconsistent",
        ["solution_certificate.verification_method"],
    )
    step = RepairStep(
        agent="solution", issues=(issue,), fields=("solution_certificate",),
    )
    old = {
        "method": "substitution", "important_givens": ["x=1"],
        "calculation_steps": ["1+1=2"], "final_answer": "2", "units": "",
        "matching_correct_option": "A", "verification_method": "old option",
    }
    current = {
        "question_id": "q1", "slot_id": "s1", "question": "1+1?",
        "options": {"A": "2", "B": "3", "C": "4", "D": "5"},
        "correct_answer": "A", "solution_certificate": old,
    }
    invalid = {**old, "verification_method": "option A", "final_answer": "changed sibling"}
    _repaired, _metadata_result, error = _apply_step("mcq", current, step, {
        "patch": {"solution_certificate": invalid},
        "repair_metadata": _metadata(
            "solution_certificate.verification_method", "old option", "option A",
            "answer_inconsistent",
        ),
    })
    assert "solution_certificate.final_answer" in error


def _metadata(field, before, after, code):
    def encoded(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)
    return {
        "modified_fields": [field],
        "changes": [{"field": field, "previous_value": encoded(before),
                     "new_value": encoded(after), "reason": f"repairs {code}"}],
        "targeted_failure_codes": [code],
    }


@pytest.mark.parametrize("role,codes", ROLE_FAILURE_CODES.items())
def test_every_supported_failure_code_has_route_or_explicit_block(role, codes):
    fields_by_role = {
        "grounding": ["question"], "task_alignment": ["question"],
        "answer_correctness": ["reference_answer"],
        "difficulty": ["question"], "quantitative": ["solution_steps"],
        "duplicate": [],
    }
    for code in codes:
        qtype = "word_problem"
        fields = list(fields_by_role[role])
        if role == "answer_correctness" and code in {
            "multiple_correct_options", "no_correct_option", "weak_distractors",
            "ambiguous_option",
        }:
            qtype, fields = "mcq", ["options"]
        if code == "invalid_word_bank":
            qtype, fields = "fill_in_the_blank", ["distractors"]
        route = build_repair_route(qtype, _verdict(_issue(role, code, fields)))
        if role == "grounding" and code == "insufficient_evidence":
            assert route["status"] == "VALIDATION_BLOCKED"
        else:
            assert route["status"] == "READY", (role, code, route)
            assert route["steps"], (role, code)


def test_route_uses_all_findings_and_has_stable_dependency_order():
    findings = [
        _issue("answer_correctness", "incorrect_answer", ["correct_answer"]),
        _issue("answer_correctness", "multiple_correct_options", ["options"]),
        _issue("answer_correctness", "ambiguous_or_structural", ["question"]),
    ]
    first = build_repair_route("mcq", _verdict(*findings))
    second = build_repair_route("mcq", _verdict(*reversed(findings)))
    assert [step.agent for step in first["steps"]] == ["question", "options", "answer"]
    assert [step.agent for step in second["steps"]] == ["question", "options", "answer"]


def test_quantitative_option_mapping_orders_solution_options_answer():
    route = build_repair_route("mcq", _verdict(
        _issue("quantitative", "incorrect_option_mapping",
               ["solution_certificate", "options", "correct_answer"])
    ))
    assert [step.agent for step in route["steps"]] == ["options", "answer", "solution"]


def test_conflicts_block_automatic_repair():
    verdict = _verdict(_issue("answer_correctness", "incorrect_answer", ["correct_answer"]))
    verdict["repair_instruction_conflicts"] = [{
        "field": "correct_answer", "instructions": ["use A", "use B"],
    }]
    route = build_repair_route("mcq", verdict)
    assert route["status"] == "REPAIR_CONFLICT"
    assert route["steps"] == []


def test_compatible_certificate_findings_share_one_solution_route():
    verdict = _verdict(
        _issue(
            "answer_correctness", "answer_inconsistent", ["solution_certificate"],
            "Update the certificate to identify option C as -10.",
        ),
        _issue(
            "quantitative", "incorrect_option_mapping", ["solution_certificate"],
            "Make the verification method refer to option C.",
        ),
    )
    route = build_repair_route("mcq", verdict)
    assert route["status"] == "READY"
    assert [step.agent for step in route["steps"]] == ["solution"]
    assert route["steps"][0].fields == ("solution_certificate",)


def test_same_field_different_wording_is_merged_not_declared_a_conflict():
    item = ProtectedQuestion(
        question_id="q1", question_type="true_false", slot_id="s1",
        question={"question_id": "q1", "slot_id": "s1", "statement": "Claim.", "answer": True},
        task={"task_id": "t1"}, evidence={}, word_bank=[], sanitization_events=[],
        errors=[], requires_quantitative=True,
    )
    findings = {
        "answer_correctness": {
            "status": "FAIL", "issues": [
                _issue("answer_correctness", "answer_inconsistent", ["solution_certificate"],
                       "Replace the stale certificate with one supporting the current statement."),
            ],
        },
        "quantitative": {
            "status": "FAIL", "issues": [
                _issue("quantitative", "incorrect_result", ["solution_certificate"],
                       "Verify the tuple named in the statement and update the certificate."),
            ],
        },
    }
    verdict = _compatibility_verdict(item, findings, {"action": "PASS"})
    assert verdict["repair_instruction_conflicts"] == []
    assert len(verdict["required_fixes"]) == 2


def test_explicit_incompatible_required_values_remain_a_conflict():
    item = ProtectedQuestion(
        question_id="q1", question_type="true_false", slot_id="s1",
        question={"question_id": "q1", "slot_id": "s1", "statement": "Claim.", "answer": True},
        task={"task_id": "t1"}, evidence={}, word_bank=[], sanitization_events=[],
        errors=[], requires_quantitative=False,
    )
    require_true = _issue("answer_correctness", "incorrect_answer", ["answer"], "Set answer True.")
    require_true["required_answer"] = True
    require_false = _issue("quantitative", "incorrect_result", ["answer"], "Set answer False.")
    require_false["required_answer"] = False
    verdict = _compatibility_verdict(item, {
        "answer_correctness": {"status": "FAIL", "issues": [require_true]},
        "quantitative": {"status": "FAIL", "issues": [require_false]},
    }, {"action": "PASS"})
    assert verdict["repair_instruction_conflicts"]
    assert verdict["repair_instruction_conflicts"][0]["requirement_values"] == ["false", "true"]


def test_legacy_tf5_certificate_conflict_reaches_atomic_repair_route():
    verdict = _verdict(
        _issue("answer_correctness", "answer_inconsistent", ["solution_certificate"],
               "Replace the stale certificate with one supporting the current statement."),
        _issue("quantitative", "incorrect_result", ["solution_certificate"],
               "Verify the tuple named in the statement and update the certificate."),
        _issue("grounding", "unsupported", ["statement"],
               "Use the source-supported tuple and displayed equations."),
    )
    verdict["repair_instruction_conflicts"] = [{
        "field": "solution_certificate",
        "instructions": [
            "Replace the stale certificate with one supporting the current statement.",
            "Verify the tuple named in the statement and update the certificate.",
        ],
    }]
    reconciled, original_conflicts = multi_repair.reconcile_repair_conflicts(
        "true_false", verdict,
        {"statement": "Tuple (1, 2, 3) satisfies the displayed system.", "answer": True},
    )
    route = build_repair_route("true_false", reconciled)
    assert original_conflicts
    assert reconciled["repair_instruction_conflicts"] == []
    assert [step.agent for step in route["steps"]] == ["task_alignment"]
    assert route["steps"][0].fields == ("answer", "solution_certificate", "statement")


def test_legacy_explicit_answer_conflict_is_not_silently_merged():
    verdict = _verdict(_issue("answer_correctness", "incorrect_answer", ["correct_answer"]))
    verdict["repair_instruction_conflicts"] = [{
        "field": "correct_answer", "instructions": ["Use A", "Use B"],
    }]
    reconciled, _ = multi_repair.reconcile_repair_conflicts("mcq", verdict, {
        "correct_answer": "C", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
    })
    assert reconciled["repair_instruction_conflicts"]
    assert build_repair_route("mcq", reconciled)["status"] == "REPAIR_CONFLICT"


def test_repair_conflict_consumes_attempt_then_selects_reconciled_strategy(monkeypatch):
    state = _integration_state()
    strategies = []

    def conflict(**kwargs):
        strategies.append(kwargs["recovery_strategy"])
        if len(strategies) == 2:
            kwargs["exam"]["questions"]["definition"][0]["reference_answer"] = "Amount of matter."
            return {
                "sent_ids": ["q1"], "repaired_ids": ["q1"], "failed_ids": [],
                "warnings": [], "repair_route": ["answer"], "repair_traces": [],
                "route_status": "COMPLETE", "repair_metadata": {"q1": {
                    "modified_fields": ["reference_answer"], "changes": [],
                    "targeted_failure_codes": ["incorrect_answer"],
                    "route": ["answer"], "steps": [],
                }},
            }
        return {
            "sent_ids": [], "repaired_ids": [], "failed_ids": ["q1"],
            "warnings": ["incompatible requirements"], "repair_route": [],
            "repair_traces": [], "route_status": "REPAIR_CONFLICT",
            "routing_details": {"conflicts": [{"field": "reference_answer"}]},
        }

    monkeypatch.setattr(multi_repair, "repair_question_specialized", conflict)

    result = repair_invalid_questions(state)

    assert result["question_repair_attempts"] == {"s1": 1}
    assert result["repair_manifest"][0]["status"] == "invalid"
    assert result["repair_candidates"] == []
    second = repair_invalid_questions({**state, **result})
    assert second["question_repair_attempts"] == {"s1": 2}
    assert strategies == ["targeted_repair", "reconciled_targeted_repair"]
    assert len(second["repair_candidates"]) == 1

    monkeypatch.setattr("app.online.multi_validator.validate_exam_multi", lambda *_args, **_kwargs: {
        "model_number": 1, "all_pass": True, "verdicts": [{
            "question_id": "q1", "action": "PASS", "validation_status": "PASS",
            "failure_code": "pass", "fields_to_fix": [], "required_fixes": [],
        }], "role_results": {}, "role_fingerprints": {}, "validation_call_logs": [],
    })
    validated = validate_generated_questions({**state, **result, **second})
    assert not validated["repair_manifest"] or all(
        item["status"] == "resolved" for item in validated["repair_manifest"]
    )
    assert validated["generated_exams"][0]["questions"]["definition"][0][
        "reference_answer"
    ] == "Amount of matter."
    assert second["question_repair_attempts"] == {"s1": 2}


def test_reconciled_certificate_strategy_preserves_authoritative_answer():
    verdict = _verdict(
        _issue("answer_correctness", "answer_inconsistent", ["solution_certificate"],
               "Use option A."),
        _issue("quantitative", "incorrect_option_mapping", ["solution_certificate"],
               "Use option C."),
    )
    verdict["repair_instruction_conflicts"] = [{
        "field": "solution_certificate", "instructions": ["Use A", "Use C"],
    }]
    question = {
        "options": {"A": "2", "B": "-2", "C": "-10", "D": "-6"},
        "correct_answer": "C",
        "solution_certificate": {"matching_correct_option": "A"},
    }

    reconciled, conflicts = multi_repair.reconcile_repair_conflicts(
        "mcq", verdict, question,
    )

    assert conflicts
    assert reconciled["repair_instruction_conflicts"] == []
    assert reconciled["fields_to_fix"] == ["solution_certificate"]
    assert "option C" in reconciled["expected_fix"]
    assert question["correct_answer"] == "C"


def test_new_revalidation_finding_uses_targeted_followup_strategy(monkeypatch):
    state = _integration_state()
    state["question_repair_attempts"] = {"s1": 1}
    state["repair_manifest"][0]["history"] = [{
        "attempt": 1, "status": "rejected", "progress": "LATERAL",
        "new_failure_code": "missing_required_given",
        "new_failure_reason": "Variable c is still undefined.",
    }]
    state["repair_retry_seeds"] = {"s1": {
        "exam": deepcopy(state["generated_exams"][0]),
        "plans": deepcopy(state["plans"]),
    }}
    observed = []

    def followup(**kwargs):
        observed.append({
            "strategy": kwargs["recovery_strategy"],
            "previous_rejection": kwargs["previous_rejection"],
            "task_id": kwargs["plan_item"]["task_id"],
            "source_ids": sorted(kwargs["source_chunks"]),
        })
        return {
            "sent_ids": ["q1"], "repaired_ids": [], "failed_ids": ["q1"],
            "warnings": ["follow-up candidate unavailable"],
            "repair_route": ["answer"], "repair_traces": [],
            "route_status": "READY",
        }

    monkeypatch.setattr(multi_repair, "repair_question_specialized", followup)
    result = repair_invalid_questions(state)

    assert result["question_repair_attempts"] == {"s1": 2}
    assert observed == [{
        "strategy": "targeted_followup_repair",
        "previous_rejection": "Variable c is still undefined.",
        "task_id": "t1", "source_ids": ["c1"],
    }]


def test_identical_candidate_outcome_forces_different_supported_strategy(monkeypatch):
    state = _integration_state()
    state["question_repair_attempts"] = {"s1": 1}
    state["repair_replacement_memory"] = {
        "s1": {"force_alternative_strategy": True}
    }
    regeneration_calls = []
    specialist_calls = []
    monkeypatch.setattr(
        "app.online.validator._regenerate_plan_slot",
        lambda *_args, **_kwargs: (regeneration_calls.append(True) or False, ["no candidate"]),
    )
    monkeypatch.setattr(
        multi_repair, "repair_question_specialized",
        lambda **_kwargs: specialist_calls.append(True),
    )

    result = repair_invalid_questions(state)

    assert regeneration_calls == [True]
    assert specialist_calls == []
    assert result["question_repair_attempts"] == {"s1": 2}
    strategy = result["repair_replacement_memory"]["s1"]["strategy_history"][-1]
    assert strategy["strategy"] == "regenerate_original_assignment"


def test_terminal_manifest_survives_fresh_validator_finding():
    old = [{
        "model_number": 1, "slot_id": "s1", "status": "terminal_failure",
        "failure_code": "repair_no_progress", "failure_reason": "unchanged",
        "history": [{"status": "terminal_failure"}],
    }]
    fresh = [{
        "model_number": 1, "slot_id": "s1", "status": "invalid",
        "failure_code": "incorrect_answer", "failure_reason": "still wrong",
    }]

    merged = _merge_repair_manifest(old, fresh)

    assert merged[0]["status"] == "terminal_failure"
    assert merged[0]["failure_code"] == "repair_no_progress"
    assert merged[0]["history"] == [{"status": "terminal_failure"}]


def test_third_failed_content_attempt_is_immediately_terminal(monkeypatch):
    state = _integration_state()
    state["question_repair_attempts"] = {"s1": 2}
    monkeypatch.setattr(multi_repair, "repair_question_specialized", lambda **_kwargs: {
        "sent_ids": ["q1"], "repaired_ids": [], "failed_ids": ["q1"],
        "warnings": ["no usable candidate"], "repair_route": ["answer"],
        "repair_traces": [], "route_status": "READY",
    })

    result = repair_invalid_questions(state)

    assert result["question_repair_attempts"] == {"s1": 3}
    assert result["repair_manifest"][0]["status"] == "terminal_failure"
    assert result["repair_candidates"] == []


def test_operationally_blocked_validation_enters_bounded_completion_manifest():
    exam = {"model_number": 1, "questions": {"definition": [{
        "question_id": "q1", "slot_id": "s1", "term": "mass",
        "reference_answer": "amount of matter",
    }]}}
    report = {"model_number": 1, "verdicts": [{
        "question_id": "q1", "action": "UNVALIDATED",
        "validation_status": "VALIDATION_BLOCKED",
        "failure_code": "validation_blocked",
        "required_fixes": [], "fields_to_fix": [],
    }]}
    manifest = _manifest_from_report(report, exam, {"s1": {"task_id": "t1"}})
    assert len(manifest) == 1
    assert manifest[0]["repair_type"] == "replace_assignment"
    assert manifest[0]["repair_category"] == "operational_validation_block"


def test_role_specific_metadata_survives_merge_and_manifest():
    item = ProtectedQuestion(
        question_id="q1", question_type="word_problem", slot_id="s1",
        question={"question_id": "q1", "slot_id": "s1", "question": "Find x.",
                  "solution_steps": ["bad"], "final_answer": "2"},
        task={"task_id": "t1"}, evidence={"c1": "x = 3"}, word_bank=[],
        sanitization_events=[], errors=[], requires_quantitative=True,
    )
    findings = {
        "quantitative": {
            "question_id": "q1", "status": "FAIL",
            "issues": [_issue("quantitative", "incorrect_result", ["final_answer"])],
            "grounding_outcome": None, "observed_difficulty": "medium",
            "calculation_evidence": ["x = 3, not 2"],
        }
    }
    verdict = _compatibility_verdict(
        item, findings,
        {"action": "PASS", "question_valid": True, "answer_valid": True},
    )
    issue = verdict["required_fixes"][0]
    assert issue["validator"] == "quantitative"
    assert issue["validation_status"] == "FAIL"
    assert issue["observed_difficulty"] == "medium"
    assert issue["calculation_evidence"] == ["x = 3, not 2"]
    exam = {"model_number": 1, "questions": {"word_problem": [item.question]}}
    manifest = _manifest_from_report(verdict | {"model_number": 1, "verdicts": [verdict]}, exam, {"s1": item.task})
    assert manifest[0]["required_fixes"][0]["calculation_evidence"] == ["x = 3, not 2"]


def test_sequential_agents_receive_latest_staged_question_and_combined_metadata(monkeypatch):
    original_options = {"A": "15", "B": "15", "C": "10", "D": "20"}
    repaired_options = {"A": "15", "B": "8", "C": "10", "D": "20"}
    exam = {"model_number": 1, "questions": {"mcq": [{
        "question_id": "q1", "slot_id": "s1", "question": "5 x 3?",
        "options": original_options, "correct_answer": "B",
    }]}}
    verdict = _verdict(
        _issue("answer_correctness", "incorrect_answer", ["correct_answer"]),
        _issue("answer_correctness", "multiple_correct_options", ["options"]),
        _issue("answer_correctness", "ambiguous_or_structural", ["question"]),
    )
    calls = []

    def fake_request(_client, prompt, **kwargs):
        payload = json.loads(prompt.split("<REPAIR_DATA>\n", 1)[1].split("\n</REPAIR_DATA>", 1)[0])
        payload = payload["questions"][0]
        calls.append((kwargs["agent"], deepcopy(payload)))
        agent = kwargs["agent"]
        current = payload["current_question"]
        if agent == "question_repairer":
            assert current["question"] == "5 x 3?"
            patch = {"question": "What is 5 × 3?"}
            metadata = _metadata("question", "5 x 3?", patch["question"], "ambiguous_or_structural")
        elif agent == "options_repairer":
            assert current["question"] == "What is 5 × 3?"
            patch = {"options": repaired_options}
            metadata = _metadata("options", original_options, repaired_options, "multiple_correct_options")
        else:
            assert current["options"] == repaired_options
            patch = {"correct_answer": "A"}
            metadata = _metadata("correct_answer", "B", "A", "incorrect_answer")
            metadata["targeted_failure_codes"] = [
                "incorrect_answer", "multiple_correct_options",
            ]
        return {"repairs": [{"question_id": "q1", "patch": patch,
                             "repair_metadata": metadata}]}

    monkeypatch.setattr(multi_repair, "request_structured", fake_request)
    result = repair_question_specialized(
        exam=exam, verdict=verdict, qtype="mcq",
        plan_item={"task_id": "t1", "selected_variation_id": "v1",
                   "source_chunk_id": "c1", "difficulty": "medium"},
        source_chunks={"c1": "5 × 3 = 15."}, domain="NON_STEM",
        client_factory=lambda _agent: object(),
    )
    assert result["repaired_ids"] == ["q1"]
    assert result["repair_route"] == ["question", "options", "answer"]
    repaired = exam["questions"]["mcq"][0]
    assert repaired["question"] == "What is 5 × 3?"
    assert repaired["options"] == repaired_options
    assert repaired["correct_answer"] == "A"
    assert len(result["repair_metadata"]["q1"]["steps"]) == 3
    assert len(calls) == 3
    assert all(call[1]["assigned_child_chunks"] == [{"chunk_id": "c1", "content": "5 × 3 = 15."}]
               for call in calls)


def test_malformed_response_retries_only_operational_attempt_and_then_succeeds(monkeypatch):
    exam = {"model_number": 1, "questions": {"definition": [{
        "question_id": "q1", "slot_id": "s1", "term": "mass",
        "reference_answer": "wrong",
    }]}}
    verdict = _verdict(_issue("answer_correctness", "incorrect_answer", ["reference_answer"]))
    attempts = []

    def fake_request(*_args, **kwargs):
        attempts.append(kwargs["operation"])
        if len(attempts) == 1:
            return {"repairs": [{"question_id": "q1", "patch": None,
                                 "repair_metadata": None}]}
        return {"repairs": [{
            "question_id": "q1", "patch": {"reference_answer": "Matter in an object."},
            "repair_metadata": _metadata(
                "reference_answer", "wrong", "Matter in an object.", "incorrect_answer"
            ),
        }]}

    monkeypatch.setattr(multi_repair, "request_structured", fake_request)
    result = repair_question_specialized(
        exam=exam, verdict=verdict, qtype="definition",
        plan_item={"task_id": "t", "source_chunk_id": "c"},
        source_chunks={"c": "Mass is the amount of matter."}, domain="NON_STEM",
        client_factory=lambda _agent: object(),
    )
    assert result["repaired_ids"] == ["q1"]
    assert len(attempts) == 2
    assert exam["questions"]["definition"][0]["reference_answer"] == "Matter in an object."


def test_metadata_previous_value_is_diagnostic_and_python_applies_authoritative_diff(monkeypatch):
    question = {"question_id": "q1", "slot_id": "s1", "term": "mass",
                "reference_answer": "wrong"}
    exam = {"model_number": 1, "questions": {"definition": [deepcopy(question)]}}
    verdict = _verdict(_issue("answer_correctness", "incorrect_answer", ["reference_answer"]))
    monkeypatch.setattr(multi_repair, "request_structured", lambda *_a, **_k: {
        "repairs": [{
            "question_id": "q1", "patch": {"reference_answer": "correct"},
            "repair_metadata": _metadata("reference_answer", "not the old value", "correct", "incorrect_answer"),
        }]
    })
    result = repair_question_specialized(
        exam=exam, verdict=verdict, qtype="definition",
        plan_item={"task_id": "t", "source_chunk_id": "c"},
        source_chunks={"c": "evidence"}, domain="NON_STEM",
        client_factory=lambda _agent: object(),
    )
    assert result["repaired_ids"] == ["q1"]
    assert exam["questions"]["definition"][0]["reference_answer"] == "correct"
    change = result["repair_metadata"]["q1"]["final_structural_diff"][0]
    assert change["previous_value"] == "wrong"
    assert change["new_value"] == "correct"


def test_arabic_content_is_preserved_in_answer_repair(monkeypatch):
    exam = {"model_number": 1, "questions": {"definition": [{
        "question_id": "q1", "slot_id": "s1", "term": "الكتلة",
        "reference_answer": "إجابة خاطئة",
    }]}}
    verdict = _verdict(_issue("answer_correctness", "incorrect_answer", ["reference_answer"]))
    corrected = "مقدار المادة في الجسم"
    monkeypatch.setattr(multi_repair, "request_structured", lambda *_a, **_k: {
        "repairs": [{"question_id": "q1", "patch": {"reference_answer": corrected},
                     "repair_metadata": _metadata("reference_answer", "إجابة خاطئة", corrected, "incorrect_answer")}]})
    result = repair_question_specialized(
        exam=exam, verdict=verdict, qtype="definition",
        plan_item={"task_id": "t", "source_chunk_id": "c"},
        source_chunks={"c": "الكتلة هي مقدار المادة في الجسم."}, domain="NON_STEM",
        client_factory=lambda _agent: object(),
    )
    assert result["repaired_ids"] == ["q1"]
    assert exam["questions"]["definition"][0]["reference_answer"] == corrected


def test_concurrent_same_agent_repairs_batch_at_most_four(monkeypatch):
    lock = threading.Lock()
    batch_sizes = []

    def fake_request(_client, prompt, **_kwargs):
        envelope = json.loads(
            prompt.split("<REPAIR_DATA>\n", 1)[1].split("\n</REPAIR_DATA>", 1)[0]
        )
        questions = envelope["questions"]
        with lock:
            batch_sizes.append(len(questions))
        rows = []
        for payload in questions:
            current = payload["current_question"]
            qid = payload["question_id"]
            new_value = f"correct {qid}"
            rows.append({
                "question_id": qid,
                "patch": {"reference_answer": new_value},
                "repair_metadata": _metadata(
                    "reference_answer", current["reference_answer"], new_value,
                    "incorrect_answer",
                ),
            })
        return {"repairs": rows}

    monkeypatch.setattr(multi_repair, "request_structured", fake_request)

    def run(index):
        qid = f"q{index}"
        exam = {"model_number": 1, "questions": {"definition": [{
            "question_id": qid, "slot_id": f"s{index}", "term": "mass",
            "reference_answer": "wrong",
        }]}}
        verdict = _verdict(_issue(
            "answer_correctness", "incorrect_answer", ["reference_answer"]
        ))
        verdict["question_id"] = qid
        result = repair_question_specialized(
            exam=exam, verdict=verdict, qtype="definition",
            plan_item={"task_id": f"t{index}", "source_chunk_id": f"c{index}"},
            source_chunks={f"c{index}": "Mass is matter."}, domain="NON_STEM",
            client_factory=lambda _agent: object(),
        )
        return result

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run, range(8)))
    assert all(result["repaired_ids"] for result in results)
    assert max(batch_sizes) <= 4
    assert any(size > 1 for size in batch_sizes)


def test_partial_batch_preserves_valid_rows_and_retries_only_invalid_question(monkeypatch):
    lock = threading.Lock()
    attempts_by_id = {}

    def fake_request(_client, prompt, **_kwargs):
        envelope = json.loads(
            prompt.split("<REPAIR_DATA>\n", 1)[1].split("\n</REPAIR_DATA>", 1)[0]
        )
        rows = []
        for payload in envelope["questions"]:
            qid = payload["question_id"]
            with lock:
                attempts_by_id[qid] = attempts_by_id.get(qid, 0) + 1
                attempt = attempts_by_id[qid]
            if qid == "q2" and attempt == 1:
                rows.append({"question_id": qid, "patch": None, "repair_metadata": None})
                continue
            previous = payload["current_question"]["reference_answer"]
            replacement = f"correct {qid}"
            rows.append({
                "question_id": qid,
                "patch": {"reference_answer": replacement},
                "repair_metadata": _metadata(
                    "reference_answer", previous, replacement, "incorrect_answer"
                ),
            })
        return {"repairs": rows}

    monkeypatch.setattr(multi_repair, "request_structured", fake_request)

    def run(index):
        qid = f"q{index}"
        exam = {"model_number": 1, "questions": {"definition": [{
            "question_id": qid, "slot_id": f"s{index}", "term": "mass",
            "reference_answer": "wrong",
        }]}}
        verdict = _verdict(_issue(
            "answer_correctness", "incorrect_answer", ["reference_answer"]
        ))
        verdict["question_id"] = qid
        return repair_question_specialized(
            exam=exam, verdict=verdict, qtype="definition",
            plan_item={"task_id": f"t{index}", "source_chunk_id": f"c{index}"},
            source_chunks={f"c{index}": "Mass is matter."}, domain="NON_STEM",
            client_factory=lambda _agent: object(),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(4)))
    assert all(result["repaired_ids"] for result in results)
    assert attempts_by_id == {"q0": 1, "q1": 1, "q2": 2, "q3": 1}


def _integration_state():
    question = {"question_id": "q1", "slot_id": "s1", "term": "mass",
                "reference_answer": "wrong"}
    issue = _issue("answer_correctness", "incorrect_answer", ["reference_answer"])
    verdict = _verdict(issue)
    return {
        "generated_exams": [{"model_number": 1, "questions": {"definition": [question]}}],
        "plans": [{"model_number": 1, "items": {"definition": [{
            "slot_id": "s1", "question_type": "definition", "task_id": "t1",
            "source_chunk_id": "c1", "support_chunk_ids": [], "difficulty": "medium",
        }]}}],
        "retrieved_chunks": [{"child_id": "c1", "content": "Mass is matter."}],
        "assessment_profile": {"domain": "NON_STEM", "question_types": {}},
        "repair_manifest": [{
            "model_number": 1, "slot_id": "s1", "question_id": "q1",
            "question_type": "definition", "status": "invalid",
            "repair_type": "repair_question", "failure_code": "incorrect_answer",
            "fields_to_fix": ["reference_answer"], "verdict": verdict,
        }],
        "validation_reports": [], "validated_models": [],
        "question_repair_attempts": {}, "warnings": [],
    }


def test_specialized_candidate_is_staged_then_committed_only_after_pass(monkeypatch):
    state = _integration_state()

    def fake_repair(**kwargs):
        item = kwargs["exam"]["questions"]["definition"][0]
        item["reference_answer"] = "Amount of matter."
        return {
            "sent_ids": ["q1"], "repaired_ids": ["q1"], "failed_ids": [],
            "warnings": [], "repair_route": ["answer"], "repair_traces": [{"attempt": 1}],
            "repair_metadata": {"q1": {
                "modified_fields": ["reference_answer"], "changes": [],
                "targeted_failure_codes": ["incorrect_answer"],
                "route": ["answer"], "steps": [],
            }},
            "route_status": "COMPLETE",
        }

    monkeypatch.setattr(multi_repair, "repair_question_specialized", fake_repair)
    staged = repair_invalid_questions(state)
    assert state["generated_exams"][0]["questions"]["definition"][0]["reference_answer"] == "wrong"
    assert staged["repair_candidates"][0]["exam"]["questions"]["definition"][0]["reference_answer"] == "Amount of matter."

    def passing_revalidation(*_args, **_kwargs):
        return {"model_number": 1, "all_pass": True, "verdicts": [{
            "question_id": "q1", "action": "PASS", "validation_status": "PASS",
            "failure_code": "pass", "fields_to_fix": [], "required_fixes": [],
        }], "role_results": {}, "role_fingerprints": {}, "validation_call_logs": []}

    monkeypatch.setattr("app.online.multi_validator.validate_exam_multi", passing_revalidation)
    merged = {**state, **staged}
    validated = validate_generated_questions(merged)
    assert validated["generated_exams"][0]["questions"]["definition"][0]["reference_answer"] == "Amount of matter."
    assert any(log.get("revalidation_outcome") == "PASS" for log in validated["repair_call_logs"])


def test_failed_revalidation_preserves_original_active_question(monkeypatch):
    state = _integration_state()

    def fake_repair(**kwargs):
        kwargs["exam"]["questions"]["definition"][0]["reference_answer"] = "still wrong"
        return {
            "sent_ids": ["q1"], "repaired_ids": ["q1"], "failed_ids": [],
            "warnings": [], "repair_route": ["answer"], "repair_traces": [],
            "repair_metadata": {"q1": {"modified_fields": ["reference_answer"],
                "changes": [], "targeted_failure_codes": ["incorrect_answer"],
                "route": ["answer"], "steps": []}}, "route_status": "COMPLETE",
        }

    monkeypatch.setattr(multi_repair, "repair_question_specialized", fake_repair)
    staged = repair_invalid_questions(state)

    def failing_revalidation(*_args, **_kwargs):
        return {"model_number": 1, "all_pass": False, "verdicts": [{
            "question_id": "q1", "action": "FIX_ANSWER", "validation_status": "FAIL",
            "failure_code": "incorrect_answer", "fields_to_fix": ["reference_answer"],
            "reason": "still wrong", "expected_fix": "correct it",
            "required_fixes": [_issue("answer_correctness", "incorrect_answer", ["reference_answer"])],
        }], "role_results": {}, "role_fingerprints": {}, "validation_call_logs": []}

    monkeypatch.setattr("app.online.multi_validator.validate_exam_multi", failing_revalidation)
    validated = validate_generated_questions({**state, **staged})
    assert validated["generated_exams"][0]["questions"]["definition"][0]["reference_answer"] == "wrong"
    assert any(log.get("revalidation_outcome") == "FAIL" for log in validated["repair_call_logs"])


def test_operational_repair_failure_does_not_spend_content_attempt(monkeypatch):
    state = _integration_state()
    monkeypatch.setattr(multi_repair, "repair_question_specialized", lambda **_kwargs: {
        "sent_ids": ["q1"], "repaired_ids": [], "failed_ids": ["q1"],
        "warnings": ["provider timeout"], "repair_route": ["answer"],
        "repair_traces": [{"error": "timeout"}], "route_status": "READY",
        "operational_error": True,
    })
    result = repair_invalid_questions(state)
    assert result["question_repair_attempts"] == {}
    assert result["repair_candidates"] == []
    assert result["repair_replacement_memory"]["s1"]["operational_retry_failures"] == 1
    assert result["repair_manifest"][0]["status"] == "operational_retry_pending"


def test_exhausted_operational_retries_block_without_spending_content_attempts(
    monkeypatch,
):
    state = _integration_state()
    monkeypatch.setattr(multi_repair, "repair_question_specialized", lambda **_kwargs: {
        "sent_ids": ["q1"], "repaired_ids": [], "failed_ids": ["q1"],
        "warnings": ["provider unavailable"], "repair_route": ["answer"],
        "repair_traces": [{"error": "provider unavailable"}],
        "route_status": "READY", "operational_error": True,
    })

    current = state
    for _ in range(3):
        result = repair_invalid_questions(current)
        current = {**current, **result}

    assert current["question_repair_attempts"] == {}
    assert current["repair_candidates"] == []
    assert current["repair_manifest"][0]["status"] == "validation_blocked"
    assert current["repair_manifest"][0]["failure_code"] == "operational_retry_exhausted"
    assert current["repair_replacement_memory"]["s1"]["operational_retry_failures"] == 3


def test_operational_revalidation_retries_candidate_without_new_repair(monkeypatch):
    state = _integration_state()
    repair_calls = []

    def fake_repair(**kwargs):
        repair_calls.append(kwargs["verdict"]["question_id"])
        kwargs["exam"]["questions"]["definition"][0]["reference_answer"] = "Amount of matter."
        return {"sent_ids": ["q1"], "repaired_ids": ["q1"], "failed_ids": [],
                "warnings": [], "repair_route": ["answer"], "repair_traces": [],
                "repair_metadata": {"q1": {"modified_fields": ["reference_answer"],
                    "changes": [], "targeted_failure_codes": ["incorrect_answer"],
                    "route": ["answer"], "steps": []}}, "route_status": "COMPLETE"}

    monkeypatch.setattr(multi_repair, "repair_question_specialized", fake_repair)
    staged = repair_invalid_questions(state)
    content_attempts = dict(staged["question_repair_attempts"])
    candidate_calls = {"count": 0}

    def fake_validate(_exam, *, question_ids, revalidation_context=None, **_kwargs):
        qid = list(question_ids or ["q1"])[0]
        if revalidation_context:
            candidate_calls["count"] += 1
            if candidate_calls["count"] == 1:
                return {"model_number": 1, "all_pass": False, "verdicts": [{
                    "question_id": qid, "action": "UNVALIDATED",
                    "validation_status": "VALIDATION_BLOCKED", "failure_code": "validation_blocked",
                    "fields_to_fix": [], "required_fixes": [],
                    "role_statuses": {"difficulty": "ERROR"}, "reason": "timeout",
                    "expected_fix": "retry"}], "role_results": {qid: {
                        "grounding": {"status": "PASS"}, "difficulty": {"status": "ERROR"}}},
                    "role_fingerprints": {}, "validation_call_logs": []}
        return {"model_number": 1, "all_pass": True, "verdicts": [{
            "question_id": qid, "action": "PASS", "validation_status": "PASS",
            "failure_code": "pass", "fields_to_fix": [], "required_fixes": []}],
            "role_results": {qid: {"difficulty": {"status": "PASS"}}},
            "role_fingerprints": {}, "validation_call_logs": []}

    monkeypatch.setattr("app.online.multi_validator.validate_exam_multi", fake_validate)
    pending = validate_generated_questions({**state, **staged, "job_id": "job-retry"})
    assert pending["repair_candidates"]
    assert pending["repair_manifest"][0]["status"] == "validation_pending"
    assert pending["question_repair_attempts"] if "question_repair_attempts" in pending else content_attempts
    assert pending["generated_exams"][0]["questions"]["definition"][0]["reference_answer"] == "wrong"

    completed = validate_generated_questions({**state, **staged, **pending, "job_id": "job-retry"})
    assert completed["repair_candidates"] == []
    assert completed["generated_exams"][0]["questions"]["definition"][0]["reference_answer"] == "Amount of matter."
    assert repair_calls == ["q1"]
    assert staged["question_repair_attempts"] == content_attempts
