from __future__ import annotations

import json
from copy import deepcopy

from app.online import multi_validator
from app.online.difficulty_adjustment import (
    PlanDifficultyAdjustmentDecision,
    _apply_adjustment_result,
)
from app.online.exam_builder import _generation_output_budget
from app.online.validator import _manifest_from_report


def test_generation_output_budgets_are_schema_and_bundle_aware():
    assert _generation_output_budget({"true_false": 1}) == 2048
    assert _generation_output_budget({"mcq": 10}) == 4096
    assert _generation_output_budget({"equation": 1}) == 9000
    assert _generation_output_budget({"equation": 4}) == 9000
    assert _generation_output_budget({"word_problem": 4}) == 9000
    assert _generation_output_budget({"mcq": 10, "true_false": 5}) == 6144
    assert _generation_output_budget(
        {"equation": 4}, provider_limit=5000
    ) == 5000


def test_terminal_difficulty_decision_records_honest_floor_without_retry():
    item = {
        "slot_id": "m1_equation_1", "task_id": "task-1",
        "difficulty": "medium", "actual_difficulty": "medium",
        "requested_difficulty": "easy", "difficulty_adaptation": {
            "status": "pending", "failed_strategies": [],
        },
    }
    decision = PlanDifficultyAdjustmentDecision.model_validate({
        "slot_id": "m1_equation_1", "adjustment_succeeded": False,
        "achieved_difficulty": "medium", "conceptual_difficulty": "medium",
        "numerical_difficulty": "easy", "further_adjustment_possible": False,
        "adjusted_fields": None, "evidence_support": [],
        "evidence_justification": "The arithmetic can be easy but the operation remains medium.",
        "reason": "The required three-variable elimination cannot be removed.",
    })
    history = []
    _apply_adjustment_result(item, {
        "slot_id": item["slot_id"], "original": deepcopy(item),
        "requested": "easy", "actual": "medium", "attempt": 1,
        "adaptation": deepcopy(item["difficulty_adaptation"]),
        "decision": decision, "proposed": None, "checks": {},
        "rejection_reasons": [], "failure": "",
    }, history)
    assert item["actual_difficulty"] == "medium"
    assert item["requested_difficulty"] == "easy"
    assert item["difficulty_design"]["status"] == "accepted_source_constraint"
    assert item["difficulty_design"]["native_prior"] == "medium"
    assert item["actual_difficulty"] == "medium"


def _grounding_manifest(scope: str):
    issue = {
        "validator": "grounding", "failure_code": "contradicted",
        "fields_to_fix": ["statement"], "reason": "scope mismatch",
        "expected_fix": "Use only the assigned evidence.", "defect_scope": scope,
    }
    report = {"model_number": 1, "verdicts": [{
        "question_id": "q1", "action": "FIX_QUESTION",
        "validation_status": "FAIL", "failure_code": "contradicted",
        "validation_role": "grounding", "fields_to_fix": ["statement"],
        "reason": "scope mismatch", "expected_fix": "recover",
        "required_fixes": [issue], "role_statuses": {"grounding": "FAIL"},
    }]}
    exam = {"questions": {"true_false": [{
        "question_id": "q1", "slot_id": "m1_true_false_1",
        "statement": "claim", "answer": True,
    }]}}
    plans = {"m1_true_false_1": {"task_id": "task-1"}}
    return _manifest_from_report(report, exam, plans)[0]


def test_grounding_question_defect_routes_to_same_assignment_regeneration():
    target = _grounding_manifest("question")
    assert target["repair_type"] == "recovery_decision_pending"
    assert target["repair_category"] == "awaiting_task_alignment_decision"


def test_grounding_assignment_defect_routes_to_catalog_replacement():
    target = _grounding_manifest("assignment")
    assert target["repair_type"] == "recovery_decision_pending"
    assert target["repair_category"] == "awaiting_task_alignment_decision"


def test_identical_failed_role_input_reuses_fail_without_provider_call(monkeypatch):
    multi_validator.clear_job_validation_cache("failed-role-cache")
    calls: list[str] = []

    def response(_client, prompt, **_kwargs):
        payload = json.loads(prompt.split("<VALIDATION_DATA>\n", 1)[1].split(
            "\n</VALIDATION_DATA>", 1,
        )[0])
        role = payload["validator"]
        calls.append(role)
        fail = role == "difficulty"
        return {"validator": role, "verdicts": [{
            "question_id": "q1", "status": "FAIL" if fail else "PASS",
            "issues": ([{
                "failure_code": "difficulty_mismatch",
                "fields_to_fix": ["equation"], "reason": "medium work",
                "expected_fix": "use an honest fallback",
            }] if fail else []),
            "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
            "observed_difficulty": "medium" if role == "difficulty" else None,
            "requested_difficulty": "easy" if role == "difficulty" else None,
            "declared_difficulty": "easy" if role == "difficulty" else None,
            "calculation_evidence": [], "original_issue_checks": [],
        }]}

    monkeypatch.setattr(multi_validator, "request_structured", response)
    factory = lambda _role: type(
        "Client", (), {"provider_name": "mock", "model": "mock"}
    )()
    exam = {"model_number": 1, "questions": {"equation": [{
        "question_id": "q1", "slot_id": "m1_equation_1",
        "equation": "Solve x+y+z=3", "solution_steps": ["eliminate"],
        "final_answer": "x=y=z=1",
    }]}}
    plan = {"m1_equation_1": {
        "task_id": "t1", "task_operation": "solve_system",
        "target": "three variables", "required_givens": ["x+y+z=3"],
        "source_chunk_id": "c1", "difficulty": "easy",
        "requested_difficulty": "easy", "requires_quantitative_validation": True,
    }}
    kwargs = dict(
        exam=exam, plan_by_slot=plan, source_by_chunk={"c1": "x+y+z=3"},
        domain="STEM", client_factory=factory, cache_scope="failed-role-cache",
    )
    first = multi_validator.validate_exam_multi(**kwargs)
    assert not first["all_pass"]
    initial_calls = len(calls)
    second = multi_validator.validate_exam_multi(**kwargs)
    assert not second["all_pass"]
    assert len(calls) == initial_calls


def test_persisted_failed_role_verdict_survives_memory_cache_reset(monkeypatch):
    scope = "persisted-failed-role-cache"
    multi_validator.clear_job_validation_cache(scope)
    calls: list[str] = []

    def response(_client, prompt, **_kwargs):
        payload = json.loads(prompt.split("<VALIDATION_DATA>\n", 1)[1].split(
            "\n</VALIDATION_DATA>", 1,
        )[0])
        role = payload["validator"]
        calls.append(role)
        return {"validator": role, "verdicts": [{
            "question_id": "q1", "status": "FAIL" if role == "difficulty" else "PASS",
            "issues": ([{
                "failure_code": "difficulty_mismatch",
                "fields_to_fix": ["equation"], "reason": "medium work",
                "expected_fix": "use an honest fallback",
            }] if role == "difficulty" else []),
            "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
            "observed_difficulty": "medium" if role == "difficulty" else None,
            "requested_difficulty": "easy" if role == "difficulty" else None,
            "declared_difficulty": "easy" if role == "difficulty" else None,
            "calculation_evidence": [], "original_issue_checks": [],
        }]}

    monkeypatch.setattr(multi_validator, "request_structured", response)
    factory = lambda _role: type(
        "Client", (), {"provider_name": "mock", "model": "mock"}
    )()
    exam = {"model_number": 1, "questions": {"equation": [{
        "question_id": "q1", "slot_id": "m1_equation_1",
        "equation": "Solve x+y+z=3", "solution_steps": ["eliminate"],
        "final_answer": "x=y=z=1",
    }]}}
    plan = {"m1_equation_1": {
        "task_id": "t1", "task_operation": "solve_system",
        "target": "three variables", "required_givens": ["x+y+z=3"],
        "source_chunk_id": "c1", "difficulty": "easy",
        "requested_difficulty": "easy", "requires_quantitative_validation": True,
    }}
    kwargs = dict(
        exam=exam, plan_by_slot=plan, source_by_chunk={"c1": "x+y+z=3"},
        domain="STEM", client_factory=factory, cache_scope=scope,
    )
    first = multi_validator.validate_exam_multi(**kwargs)
    persisted = {
        "q1": {
            role: {
                "fingerprint": fingerprint,
                "verdict": first["role_results"]["q1"][role],
            }
            for role, fingerprint in first["role_fingerprints"]["q1"].items()
        }
    }
    initial_calls = len(calls)
    multi_validator.clear_job_validation_cache(scope)
    second = multi_validator.validate_exam_multi(**kwargs, cached_roles=persisted)
    assert not second["all_pass"]
    assert len(calls) == initial_calls


def test_changed_assignment_invalidates_failed_role_verdict(monkeypatch):
    scope = "failed-role-assignment-change"
    multi_validator.clear_job_validation_cache(scope)
    calls: list[str] = []

    def response(_client, prompt, **_kwargs):
        payload = json.loads(prompt.split("<VALIDATION_DATA>\n", 1)[1].split(
            "\n</VALIDATION_DATA>", 1,
        )[0])
        role = payload["validator"]
        calls.append(role)
        return {"validator": role, "verdicts": [{
            "question_id": "q1", "status": "FAIL" if role == "difficulty" else "PASS",
            "issues": ([{
                "failure_code": "difficulty_mismatch",
                "fields_to_fix": ["equation"], "reason": "medium work",
                "expected_fix": "use an honest fallback",
            }] if role == "difficulty" else []),
            "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
            "observed_difficulty": "medium" if role == "difficulty" else None,
            "requested_difficulty": "easy" if role == "difficulty" else None,
            "declared_difficulty": "easy" if role == "difficulty" else None,
            "calculation_evidence": [], "original_issue_checks": [],
        }]}

    monkeypatch.setattr(multi_validator, "request_structured", response)
    factory = lambda _role: type(
        "Client", (), {"provider_name": "mock", "model": "mock"}
    )()
    exam = {"model_number": 1, "questions": {"equation": [{
        "question_id": "q1", "slot_id": "m1_equation_1",
        "equation": "Solve x+y+z=3", "solution_steps": ["eliminate"],
        "final_answer": "x=y=z=1",
    }]}}
    plan = {"m1_equation_1": {
        "task_id": "t1", "task_operation": "solve_system",
        "target": "three variables", "required_givens": ["x+y+z=3"],
        "source_chunk_id": "c1", "difficulty": "easy",
        "requested_difficulty": "easy", "requires_quantitative_validation": True,
    }}
    kwargs = dict(
        exam=exam, plan_by_slot=plan, source_by_chunk={"c1": "x+y+z=3"},
        domain="STEM", client_factory=factory, cache_scope=scope,
    )
    multi_validator.validate_exam_multi(**kwargs)
    first_call_count = len(calls)
    changed_plan = deepcopy(plan)
    changed_plan["m1_equation_1"]["assignment_revision"] = 2
    multi_validator.validate_exam_multi(**{**kwargs, "plan_by_slot": changed_plan})
    assert len(calls) > first_call_count

