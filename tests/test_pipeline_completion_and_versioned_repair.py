from __future__ import annotations

import json

from app import config
from app.online import difficulty_adjustment, graph, multi_repair
from app.online.multi_repair import repair_question_specialized


def _issue(role: str, code: str, field: str, issue_id: str) -> dict:
    return {
        "validator": role,
        "validation_status": "FAIL",
        "failure_code": code,
        "fields_to_fix": [field],
        "reason": f"{field} is inconsistent",
        "expected_fix": f"correct {field}",
        "issue_id": issue_id,
        "calculation_evidence": [],
    }


def test_new_repair_and_completion_limits_are_independent():
    assert config.MAX_CONTENT_REPAIR_CALLS_PER_SLOT == 8
    assert config.MAX_FAILED_REPAIR_ATTEMPTS_PER_SLOT == 3
    assert config.MAX_COMPLETION_ATTEMPTS_PER_SLOT == 3


def test_terminal_content_slot_routes_to_separate_completion_budget():
    state = {
        "repair_candidates": [],
        "repair_manifest": [{
            "slot_id": "m1_mcq_1", "status": "terminal_failure",
            "failure_code": "content_repair_budget_exhausted",
        }],
        "question_repair_attempts": {"m1_mcq_1": 3},
        "question_repair_calls": {"m1_mcq_1": 8},
        "question_repair_failures": {"m1_mcq_1": 3},
        "slot_completion_attempts": {"m1_mcq_1": 0},
    }
    assert graph._route_after_validation(state) == "repair_invalid_questions"
    state["slot_completion_attempts"]["m1_mcq_1"] = 3
    assert graph._route_after_validation(state) == "assemble_exams"


def test_validation_block_routes_to_bounded_completion_and_cannot_loop_forever():
    state = {
        "repair_candidates": [],
        "repair_manifest": [{
            "slot_id": "m1_true_false_1", "status": "validation_blocked",
            "block_category": "operational", "repair_type": "replace_assignment",
        }],
        "question_repair_failures": {}, "question_repair_calls": {},
        "slot_completion_attempts": {"m1_true_false_1": 0},
    }
    assert graph._route_after_validation(state) == "repair_invalid_questions"
    state["slot_completion_attempts"]["m1_true_false_1"] = 3
    assert graph._route_after_validation(state) == "assemble_exams"


def test_inherently_medium_task_is_retained_and_honestly_labeled():
    plans = [{"model_number": 1, "items": {"equation": [{
        "slot_id": "m1_equation_1", "task_id": "task-graph",
        "question_type": "equation", "difficulty": "medium",
        "actual_difficulty": "medium", "requested_difficulty": "easy",
        "adjustment_attempts": 3,
        "difficulty_adaptation": {"status": "failed"},
        "difficulty_fallback_reason": (
            "Mandatory graphing and vertex analysis cannot genuinely be simplified."
        ),
    }]}}]
    difficulty_adjustment._apply_catalog_fallbacks(plans, {"tasks": []}, "easy")
    item = plans[0]["items"]["equation"][0]
    assert item["difficulty"] == "medium"
    assert item["actual_difficulty"] == "medium"
    assert item["requested_difficulty"] == "easy"
    assert item["difficulty_status"] == "accepted_source_constraint"
    assert item["difficulty_adaptation"]["status"] == "accepted_source_constraint"


def test_dependency_route_repairs_answer_before_derived_solution():
    verdict = {
        "question_id": "q1", "repair_instruction_conflicts": [],
        "required_fixes": [
            _issue("answer_correctness", "incorrect_answer", "correct_answer", "answer"),
            _issue("quantitative", "incorrect_option_mapping", "solution_certificate", "certificate"),
        ],
    }
    route = multi_repair.build_repair_route("mcq", verdict)
    assert [step.agent for step in route["steps"]] == ["answer", "solution"]


def test_successful_intermediate_revision_is_input_to_next_agent(monkeypatch):
    exam = {"model_number": 1, "questions": {"mcq": [{
        "question_id": "q1", "slot_id": "m1_mcq_1", "question": "2+2?",
        "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
        "correct_answer": "A",
        "solution_certificate": {
            "method": "addition", "important_givens": ["2+2"],
            "calculation_steps": ["2+2=4"], "final_answer": "A",
            "units": "", "matching_correct_option": "A",
            "verification_method": "4 is A",
        },
    }]}}
    seen = []

    def fake_call_step(*, payload, **_kwargs):
        seen.append((payload["repair_agent"], payload["current_question"]["correct_answer"]))
        current = payload["current_question"]
        if payload["repair_agent"] == "answer":
            patch = {"correct_answer": "B"}
        else:
            certificate = dict(current["solution_certificate"])
            certificate.update({"final_answer": "B", "matching_correct_option": "B",
                                "verification_method": "4 is B"})
            patch = {"solution_certificate": certificate}
        return ({
            "question_id": "q1", "patch": patch,
            "repair_metadata": {"modified_fields": list(patch), "changes": [],
                                "targeted_failure_codes": []},
        }, None, {})

    monkeypatch.setattr(multi_repair, "_call_step", fake_call_step)
    verdict = {
        "question_id": "q1", "action": "FIX_QUESTION_AND_ANSWER",
        "validation_status": "FAIL", "failure_code": "incorrect_answer",
        "fields_to_fix": ["correct_answer", "solution_certificate"],
        "repair_instruction_conflicts": [],
        "required_fixes": [
            _issue("answer_correctness", "incorrect_answer", "correct_answer", "answer"),
            _issue("quantitative", "incorrect_option_mapping", "solution_certificate", "certificate"),
        ],
    }
    result = repair_question_specialized(
        exam=exam, verdict=verdict, qtype="mcq", plan_item={"task_id": "t1"},
        source_chunks={}, domain="STEM", client_factory=lambda _agent: object(),
    )
    assert result["route_status"] == "COMPLETE"
    assert result["content_repair_calls"] == 2
    assert result["failed_repair_attempts"] == 0
    assert [row["status"] for row in result["candidate_revisions"]] == [
        "accepted", "accepted",
    ]
    assert seen == [("answer", "A"), ("solution", "B")]
    assert exam["questions"]["mcq"][0]["correct_answer"] == "B"
    assert exam["questions"]["mcq"][0]["solution_certificate"]["matching_correct_option"] == "B"


def test_rejected_later_patch_preserves_last_accepted_revision(monkeypatch):
    exam = {"model_number": 1, "questions": {"definition": [{
        "question_id": "q1", "slot_id": "m1_definition_1", "term": "mass",
        "reference_answer": "wrong",
        "solution_certificate": {"method": "old"},
    }]}}
    calls = {"count": 0}

    def fake_call_step(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return ({"question_id": "q1",
                "patch": {"reference_answer": "amount of matter"},
                "repair_metadata": {"modified_fields": ["reference_answer"],
                                    "changes": [], "targeted_failure_codes": []}}, None, {})
        # An unauthorized extra key is rejected; the accepted answer revision
        # remains the current candidate for the corrective call.
        return ({"question_id": "q1",
            "patch": {"solution_certificate": {"method": "new"}, "term": "changed"},
            "repair_metadata": {"modified_fields": ["solution_certificate", "term"],
                                "changes": [], "targeted_failure_codes": []}}, None, {})

    monkeypatch.setattr(multi_repair, "_call_step", fake_call_step)
    verdict = {
        "question_id": "q1", "action": "FIX_QUESTION_AND_ANSWER",
        "validation_status": "FAIL", "failure_code": "incorrect_answer",
        "fields_to_fix": ["reference_answer", "solution_certificate"],
        "repair_instruction_conflicts": [],
        "required_fixes": [
            _issue("answer_correctness", "incorrect_answer", "reference_answer", "answer"),
            _issue("quantitative", "incomplete_solution", "solution_certificate", "certificate"),
        ],
    }
    result = repair_question_specialized(
        exam=exam, verdict=verdict, qtype="definition", plan_item={"task_id": "t1"},
        source_chunks={}, domain="NON_STEM", client_factory=lambda _agent: object(),
        failed_repair_attempts_used=2,
    )
    assert result["failed_repair_attempts"] == 3
    assert result["candidate_revisions"][0]["status"] == "accepted"
    assert result["candidate_revisions"][1]["status"] == "rejected"
    assert result["candidate_revisions"][0]["candidate"]["reference_answer"] == "amount of matter"
    assert exam["questions"]["definition"][0]["reference_answer"] == "wrong"
