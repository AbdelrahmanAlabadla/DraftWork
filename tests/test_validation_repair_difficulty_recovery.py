from __future__ import annotations

import json
from copy import deepcopy

from app.online import difficulty_adjustment, multi_repair, multi_validator
def _certificate(tuple_text: str) -> dict:
    return {
        "method": "substitution",
        "important_givens": [tuple_text],
        "calculation_steps": [f"Substitute {tuple_text} in all equations."],
        "final_answer": "True",
        "units": None,
        "matching_correct_option": None,
        "verification_method": f"The tuple {tuple_text} satisfies all three equations.",
    }


def test_targeted_difficulty_adjustment_changes_only_requested_slot(monkeypatch):
    plans = [{
        "model_number": 1,
        "items": {"equation": [
            {
                "slot_id": "m1_equation_1", "question_type": "equation",
                "task_id": "t1", "base_catalog_task_id": "t1",
                "task_family_id": "family", "base_task_family_id": "family",
                "source_chunk_id": "c1", "support_chunk_ids": [],
                "selected_variation_id": "v1", "selected_variation_group_id": "g1",
                "permitted_variations": [{"variation_id": "v1", "numeric_values_may_change": False}],
                "concept": "systems", "task_operation": "graph_system",
                "target": "graph a supported system", "student_work_product": "graph",
                "reasoning_requirements": ["find every feasible vertex"],
                "given_conditions": ["x >= 0"], "required_givens": ["x >= 0"],
                "difficulty": "easy", "actual_difficulty": "easy",
                "requested_difficulty": "easy", "reasoning_depth": "easy",
            },
            {
                "slot_id": "m1_equation_2", "question_type": "equation",
                "task_id": "t2", "source_chunk_id": "c2",
                "difficulty": "easy", "reasoning_requirements": ["unchanged"],
            },
        ]},
    }]
    sibling_before = deepcopy(plans[0]["items"]["equation"][1])
    prompts = []

    def response(_client, prompt, **_kwargs):
        prompts.append(json.loads(prompt))
        return {
            "slot_id": "m1_equation_1", "adjustment_succeeded": True,
            "achieved_difficulty": "easy",
            "adjusted_fields": {
                "reasoning_requirements": ["draw the supplied boundary lines"],
                "prerequisites": None, "given_conditions": None,
                "required_givens": None, "idea_to_test": None,
                "strategy": "Remove unsupported vertex analysis while preserving graphing.",
            },
            "evidence_support": ["c1"],
            "evidence_justification": "The source supports graphing the boundaries.",
            "reason": "The revised work is direct representation.",
        }

    monkeypatch.setattr(difficulty_adjustment, "request_structured", response)
    ok, audit = difficulty_adjustment.adjust_failed_plan_slot(
        state={
            "difficulty": "easy", "assessment_profile": {"domain": "STEM"},
            "retrieved_chunks": [
                {"child_id": "c1", "content": "graph the inequalities"},
                {"child_id": "c2", "content": "other evidence"},
            ],
        },
        plans=plans, model_number=1, slot_id="m1_equation_1",
        question_type="equation",
        findings=[{
            "validator": "difficulty", "failure_code": "reasoning_complexity_mismatch",
            "requested_difficulty": "easy", "declared_difficulty": "easy",
            "observed_difficulty": "medium", "calculation_evidence": ["find all vertices"],
        }],
        attempt=1,
    )

    assert ok and audit["observed_difficulty"] == "medium"
    assert plans[0]["items"]["equation"][1] == sibling_before
    assert plans[0]["items"]["equation"][0]["reasoning_requirements"] == [
        "draw the supplied boundary lines"
    ]
    assert prompts[0]["difficulty_validator_findings"][0]["observed_difficulty"] == "medium"


def test_difficulty_failure_classifies_as_assignment_adjustment():
    from app.online.validator import _repair_classification

    assert _repair_classification("reasoning_complexity_mismatch") == (
        "difficulty_assignment_mismatch", "adjust_difficulty",
    )


def test_true_false_atomic_route_coordinates_statement_answer_and_certificate():
    verdict = {
        "question_id": "q1", "repair_instruction_conflicts": [],
        "required_fixes": [
            {
                "validator": "grounding", "failure_code": "unsupported",
                "fields_to_fix": ["statement"], "reason": "Unsupported tuple.",
                "expected_fix": "Use the source-supported tuple and displayed equations.",
                "issue_id": "i1",
            },
            {
                "validator": "answer_correctness", "failure_code": "answer_inconsistent",
                "fields_to_fix": ["solution_certificate"], "reason": "Stale tuple.",
                "expected_fix": "Verify the tuple in the statement.", "issue_id": "i2",
            },
        ],
    }
    route = multi_repair.build_repair_route("true_false", verdict)
    assert route["status"] == "READY"
    assert route["atomic_true_false"] is True
    assert len(route["steps"]) == 1
    assert route["steps"][0].agent == "task_alignment"
    assert set(route["steps"][0].fields) == {
        "statement", "answer", "solution_certificate",
    }


def test_true_false_compatible_certificate_findings_reconcile():
    verdict = {
        "repair_instruction_conflicts": [{
            "field": "solution_certificate",
            "instructions": ["Replace stale certificate", "Verify current tuple"],
        }],
        "required_fixes": [
            {"validator": "grounding", "failure_code": "contradicted",
             "fields_to_fix": ["solution_certificate"],
             "reason": "Old tuple", "expected_fix": "Replace stale certificate"},
            {"validator": "answer_correctness", "failure_code": "answer_inconsistent",
             "fields_to_fix": ["solution_certificate"],
             "reason": "Wrong verification", "expected_fix": "Verify current tuple"},
        ],
    }
    reconciled, conflicts = multi_repair.reconcile_repair_conflicts(
        "true_false", verdict,
        {"statement": "(-1,-5,7) is a solution", "answer": True},
    )
    assert conflicts
    assert reconciled["repair_instruction_conflicts"] == []
    assert "current" in reconciled["expected_fix"].lower()


def test_true_false_stale_tuple_certificate_is_rejected():
    current = {
        "question_id": "q1", "slot_id": "m1_true_false_1",
        "statement": "The tuple (1,2,3) satisfies the displayed system.",
        "answer": True, "solution_certificate": _certificate("(1,2,3)"),
    }
    step = multi_repair.RepairStep(
        agent="task_alignment",
        issues=({"failure_code": "unsupported", "issue_id": "i1",
                 "fields_to_fix": ["statement"]},),
        fields=("answer", "solution_certificate", "statement"),
    )
    changed_statement = "The tuple (-1,-5,7) satisfies x+y+z=1."
    row = {
        "patch": {
            "statement": changed_statement, "answer": True,
            "solution_certificate": _certificate("(1,2,3)"),
        },
        "repair_metadata": {
            "modified_fields": ["statement"],
            "changes": [{
                "field": "statement", "previous_value": current["statement"],
                "new_value": changed_statement, "reason": "Use supported tuple.",
            }],
            "targeted_failure_codes": ["unsupported"],
        },
    }
    repaired, _metadata, error = multi_repair._apply_step(
        "true_false", current, step, row,
    )
    assert repaired is None
    assert "stale tuple" in error


def test_answer_correctness_prompt_requires_all_mcq_options_and_precise_target():
    prompt = multi_validator._system_prompt("answer_correctness")
    assert "evaluate the keyed option and every distractor independently" in prompt
    assert "mathematically equivalent" in prompt
    assert "swapped variable definitions" in prompt
    assert "options.D" in prompt


def test_quantitative_revalidation_bypasses_prior_pass_cache_for_every_role(monkeypatch):
    multi_validator.clear_job_validation_cache("full-revalidation")
    question = {
        "question_id": "q1", "slot_id": "m1_equation_1",
        "equation": "Solve x+1=2", "solution_steps": ["x=1"],
        "final_answer": "x=1", "solution_certificate": _certificate("(1,1)"),
    }
    exam = {"model_number": 1, "questions": {"equation": [question]}}
    plan = {"m1_equation_1": {
        "task_id": "t1", "task_operation": "solve_equation", "target": "x",
        "required_givens": ["x+1=2"], "source_chunk_id": "c1",
        "difficulty": "easy", "requested_difficulty": "easy",
        "requires_quantitative_validation": True,
    }}
    calls = []

    def response(_client, prompt, **_kwargs):
        payload = json.loads(prompt.split("<VALIDATION_DATA>\n", 1)[1].split(
            "\n</VALIDATION_DATA>", 1,
        )[0])
        role = payload["validator"]
        calls.append(role)
        return {"validator": role, "verdicts": [{
            "question_id": "q1", "status": "PASS", "issues": [],
            "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
            "observed_difficulty": "easy" if role == "difficulty" else None,
            "requested_difficulty": "easy" if role == "difficulty" else None,
            "declared_difficulty": "easy" if role == "difficulty" else None,
            "calculation_evidence": [], "original_issue_checks": [],
        }]}

    monkeypatch.setattr(multi_validator, "request_structured", response)
    factory = lambda _role: type("Client", (), {"provider_name": "mock", "model": "mock"})()
    multi_validator.validate_exam_multi(
        exam, plan_by_slot=plan, source_by_chunk={"c1": "x+1=2"},
        domain="STEM", client_factory=factory, cache_scope="full-revalidation",
    )
    calls.clear()
    result = multi_validator.validate_exam_multi(
        exam, plan_by_slot=plan, source_by_chunk={"c1": "x+1=2"},
        domain="STEM", client_factory=factory, cache_scope="full-revalidation",
        revalidation_context={"q1": {
            "original_failure_findings": [], "force_full_revalidation": True,
        }},
    )
    assert result["all_pass"]
    assert set(calls) == {
        "grounding", "task_alignment", "answer_correctness", "difficulty", "quantitative",
    }
