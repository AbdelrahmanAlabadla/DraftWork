"""Group 2 regressions: difficulty design, no replacement, validator authority."""

from __future__ import annotations

import json
from copy import deepcopy

from app.online import assessment_catalog, difficulty_adjustment
from app.online.multi_validator import ProtectedQuestion, _compatibility_verdict, _payload_for
from app.online.planner import validate_plans


def _task(native: str = "medium", **overrides) -> dict:
    task = {
        "task_id": "task_linear",
        "task_family_id": "family_linear",
        "coverage_id": "coverage_linear",
        "concept_id": "concept_linear",
        "relationship_id": "relationship_linear",
        "knowledge_anchor": "linear equations",
        "concept": "linear equations",
        "relationship": "linear equality",
        "task_operation": "solve_equation",
        "student_work_product": "solve equation",
        "target": "value of x",
        "reasoning_mode": "symbolic_manipulation",
        "allowed_question_types": ["equation"],
        "evidence_chunk_ids": ["chunk-main"],
        "required_givens": ["linear equation"],
        "variable_givens": ["coefficient", "constant"],
        "scenario_family": None,
        "text_answerable": True,
        "native_difficulty": native,
        "reasoning_requirements": ["Solve a linear equation."],
        "prerequisites": [],
        "allowed_transformations": [{
            "transformation": "combine_like_terms",
            "evidence_chunk_ids": ["chunk-main"],
            "support": "Combine like terms before isolating the variable.",
        }],
        "allowed_variations": [{
            "target": "value of x",
            "operation": "solve_equation",
            "givens": ["ax + b = c"],
            "question_types": ["equation"],
            "reasoning_mode": "symbolic_manipulation",
            "student_work_product": "solve equation",
            "numeric_values_may_change": True,
        }],
    }
    task.update(overrides)
    return task


def _profile() -> dict:
    return {
        "domain": "STEM",
        "subject": "mathematics",
        "question_types": {
            "equation": {
                "support": "supported",
                "task_operations": ["solve_equation"],
                "form_guidance": "grounded",
                "evidence_chunk_ids": ["chunk-main"],
            },
        },
    }


def _state(task: dict | None = None, requested: str = "hard", slots: int = 1) -> dict:
    catalog = {"tasks": [task or _task()], "assessment_profile": _profile()}
    plans, issues = assessment_catalog.allocate_catalog_plan(
        catalog, [("equation", slots)], 1, requested,
    )
    return {
        "document_id": "doc",
        "tasks": [("equation", slots)],
        "num_models": 1,
        "difficulty": requested,
        "plans": plans,
        "assessment_catalog": catalog,
        "assessment_profile": _profile(),
        "planning_issues": issues,
        "retrieved_chunks": [{
            "child_id": "chunk-main",
            "content": "Combine like terms, then isolate the variable.",
        }],
        "difficulty_adjustment_history": [],
        "difficulty_adjustment_attempts": {},
    }


def _decision(slot_id: str, *, transformations=None, kind="cognitive") -> dict:
    return {
        "slot_id": slot_id,
        "adjustment_succeeded": True,
        "achieved_difficulty": "hard",
        "estimated_conceptual_difficulty": "medium",
        "estimated_numerical_difficulty": "medium",
        "adjusted_fields": {
            "adjustment_kind": kind,
            "reasoning_requirements": [
                "Combine like terms into one expression.",
                "Isolate the variable with inverse operations.",
            ],
            "cognitive_transformations": transformations or ["combine_like_terms"],
            "prerequisites": [],
        },
        "evidence_support": ["chunk-main"],
        "evidence_justification": "The chunk supports combining like terms.",
        "reason": "Adds one supported transformation without changing the operation.",
    }


def _decline(slot_id: str) -> dict:
    return {
        "slot_id": slot_id,
        "adjustment_succeeded": False,
        "achieved_difficulty": "medium",
        "adjusted_fields": None,
        "evidence_support": [],
        "evidence_justification": "No supported transformation reaches the request.",
        "reason": "The source does not support a harder version.",
    }


def test_all_compatible_tasks_are_eligible_regardless_of_native_level():
    catalog = {
        "tasks": [
            _task("hard", task_id="task_hard", task_family_id="family_h",
                  coverage_id="cov_h", concept_id="concept_h"),
            _task("medium", task_id="task_medium", task_family_id="family_m",
                  coverage_id="cov_m", concept_id="concept_m"),
        ],
        "assessment_profile": _profile(),
    }
    plans, issues = assessment_catalog.allocate_catalog_plan(
        catalog, [("equation", 2)], 1, "easy",
    )
    items = plans[0]["items"]["equation"]
    assert issues == []
    assert len(items) == 2
    assert {item["native_difficulty_prior"] for item in items} == {"hard", "medium"}
    assert all((item["difficulty_design"] or {}).get("status") == "pending" for item in items)


def test_numeric_only_design_is_accepted_without_requirement_change(monkeypatch):
    state = _state(requested="easy")
    original = state["plans"][0]["items"]["equation"][0]
    response = {
        "slot_id": original["slot_id"],
        "adjustment_succeeded": True,
        "achieved_difficulty": "easy",
        "estimated_conceptual_difficulty": "medium",
        "estimated_numerical_difficulty": "easy",
        "adjusted_fields": {
            "adjustment_kind": "numerical",
            "reasoning_requirements": list(original["reasoning_requirements"]),
            "given_conditions": ["2x + 3 = 11"],
        },
        "evidence_support": ["chunk-main"],
        "evidence_justification": "Smaller coefficients preserve the operation.",
        "reason": "Numerical simplification only.",
    }
    monkeypatch.setattr(difficulty_adjustment, "request_structured", lambda *_a, **_k: response)
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    item = result["plans"][0]["items"]["equation"][0]
    assert item["difficulty_design"]["status"] == "designed"
    assert item["difficulty_design"]["adjustment_kind"] == "numerical"
    assert item["actual_difficulty"] == "medium"  # prior; Luna owns actual
    assert item["effective_givens"] == ["2x + 3 = 11"]


def test_cognitive_transformation_requires_authorization(monkeypatch):
    state = _state()
    item = state["plans"][0]["items"]["equation"][0]
    response = _decision(item["slot_id"], transformations=["invented_quadratic_step"])
    monkeypatch.setattr(difficulty_adjustment, "request_structured", lambda *_a, **_k: response)
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    updated = result["plans"][0]["items"]["equation"][0]
    assert updated["difficulty_design"]["status"] == "accepted_source_constraint"
    assert "transformations_authorized=false" in updated["difficulty_fallback_reason"]


def test_authorized_cognitive_transformation_changes_reasoning_not_actual(monkeypatch):
    state = _state()
    item = state["plans"][0]["items"]["equation"][0]
    response = _decision(item["slot_id"])
    monkeypatch.setattr(difficulty_adjustment, "request_structured", lambda *_a, **_k: response)
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    updated = result["plans"][0]["items"]["equation"][0]
    assert updated["difficulty_design"]["status"] == "designed"
    assert updated["difficulty_design"]["cognitive_transformations"] == ["combine_like_terms"]
    assert updated["actual_difficulty"] == "medium"
    assert "Combine like terms" in " ".join(updated["reasoning_requirements"])


def test_unattainable_request_keeps_task_without_replacement(monkeypatch):
    state = _state()
    item = state["plans"][0]["items"]["equation"][0]
    monkeypatch.setattr(
        difficulty_adjustment, "request_structured",
        lambda *_a, **_k: _decline(item["slot_id"]),
    )
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    updated = result["plans"][0]["items"]["equation"][0]
    assert updated["base_catalog_task_id"] == "task_linear"
    assert updated.get("selected_replacement_task_id") is None
    assert updated["difficulty_design"]["status"] == "accepted_source_constraint"
    assert updated["actual_difficulty"] == "medium"
    assert result["difficulty_adjustment_attempts"][item["slot_id"]] == 1


def test_compatible_slots_each_receive_an_independent_difficulty_decision(monkeypatch):
    state = _state(slots=2)
    for item in state["plans"][0]["items"]["equation"]:
        for variation in item.get("permitted_variations") or []:
            variation["numeric_values_may_change"] = False
    calls: list[str] = []

    def fake(_client, prompt, **_kwargs):
        payload = json.loads(prompt)
        calls.append(str(len(payload.get("tasks") or [])))
        results = []
        for entry in payload.get("tasks") or []:
            slot_id = entry["task"]["immutable_identifiers"]["slot_id"]
            results.append(_decision(slot_id))
        return {"results": results}

    monkeypatch.setattr(difficulty_adjustment, "request_structured", fake)
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    items = result["plans"][0]["items"]["equation"]
    assert calls == ["2"]  # batching is allowed; both slots receive their own decision
    assert all(item["difficulty_design"]["status"] == "designed" for item in items)
    assert all(item["difficulty_design"].get("reused") is False for item in items)


def test_historical_design_cache_is_ignored_by_a_new_job(monkeypatch):
    state = _state()
    state["difficulty_design_cache"] = {
        "legacy-key": {"design": {"status": "designed", "reason": "stale"}}
    }
    calls = 0

    def fake(_client, prompt, **_kwargs):
        nonlocal calls
        calls += 1
        payload = json.loads(prompt)
        return {"results": [
            _decision(entry["task"]["immutable_identifiers"]["slot_id"])
            for entry in payload["tasks"]
        ]}

    monkeypatch.setattr(difficulty_adjustment, "request_structured", fake)
    result = difficulty_adjustment.adjust_plan_difficulty(state)

    assert calls == 1
    assert "difficulty_design_cache" not in result
    assert result["plans"][0]["items"]["equation"][0]["difficulty_design"]["reason"] != "stale"


def test_resumed_completed_adjustment_does_not_call_luna_again(monkeypatch):
    state = _state()
    calls = 0

    def fake(_client, prompt, **_kwargs):
        nonlocal calls
        calls += 1
        payload = json.loads(prompt)
        return {"results": [
            _decision(entry["task"]["immutable_identifiers"]["slot_id"])
            for entry in payload["tasks"]
        ]}

    monkeypatch.setattr(difficulty_adjustment, "request_structured", fake)
    completed = difficulty_adjustment.adjust_plan_difficulty(state)
    resumed_state = {
        **state,
        "plans": json.loads(json.dumps(completed["plans"])),
        "difficulty_adjustment_history": completed["difficulty_adjustment_history"],
        "difficulty_adjustment_attempts": completed["difficulty_adjustment_attempts"],
        "difficulty_adaptation_records": completed["difficulty_adaptation_records"],
    }
    resumed = difficulty_adjustment.adjust_plan_difficulty(resumed_state)

    assert calls == 1
    assert resumed["plans"] == resumed_state["plans"]


def test_main_terminal_decision_is_reused_by_targeted_recovery_after_checkpoint(monkeypatch):
    state = _state()
    slot_id = state["plans"][0]["items"]["equation"][0]["slot_id"]
    terminal = {**_decline(slot_id), "further_adjustment_possible": False}
    monkeypatch.setattr(
        difficulty_adjustment, "request_structured", lambda *_a, **_k: terminal,
    )
    completed = difficulty_adjustment.adjust_plan_difficulty(state)
    replayed_plans = json.loads(json.dumps(completed["plans"]))
    replayed_records = json.loads(json.dumps(completed["difficulty_adaptation_records"]))

    monkeypatch.setattr(
        difficulty_adjustment, "request_structured",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected provider call")),
    )
    accepted, audit = difficulty_adjustment.adjust_failed_plan_slot(
        state={**state, "difficulty_adaptation_records": replayed_records},
        plans=replayed_plans, model_number=1, slot_id=slot_id,
        question_type="equation", findings=[], attempt=2,
    )

    assert accepted is True
    assert audit["status"] == "reused_terminal_decision"
    assert audit["provider_call_made"] is False


def test_changed_assignment_invalidates_terminal_decision(monkeypatch):
    state = _state()
    slot_id = state["plans"][0]["items"]["equation"][0]["slot_id"]
    terminal = {**_decline(slot_id), "further_adjustment_possible": False}
    monkeypatch.setattr(
        difficulty_adjustment, "request_structured", lambda *_a, **_k: terminal,
    )
    completed = difficulty_adjustment.adjust_plan_difficulty(state)
    changed_plans = json.loads(json.dumps(completed["plans"]))
    changed = changed_plans[0]["items"]["equation"][0]
    changed["given_conditions"] = ["A materially changed assignment"]
    calls = 0

    def decline_again(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return terminal

    monkeypatch.setattr(difficulty_adjustment, "request_structured", decline_again)
    accepted, audit = difficulty_adjustment.adjust_failed_plan_slot(
        state={
            **state,
            "difficulty_adaptation_records": completed["difficulty_adaptation_records"],
        },
        plans=changed_plans, model_number=1, slot_id=slot_id,
        question_type="equation", findings=[], attempt=2,
    )

    assert accepted is True
    assert calls == 1
    assert audit["provider_call_made"] is True


def test_incompatible_evidence_does_not_reuse_design(monkeypatch):
    state = _state(slots=2)
    second = state["plans"][0]["items"]["equation"][1]
    second["support_chunk_ids"] = ["chunk-other"]
    calls: list[int] = []

    def fake(_client, prompt, **_kwargs):
        payload = json.loads(prompt)
        calls.append(len(payload.get("tasks") or []))
        return {"results": [
            _decision(entry["task"]["immutable_identifiers"]["slot_id"])
            for entry in payload.get("tasks") or []
        ]}

    monkeypatch.setattr(difficulty_adjustment, "request_structured", fake)
    difficulty_adjustment.adjust_plan_difficulty(state)
    assert calls == [2]


def test_difficulty_validator_payload_is_blind_and_observed():
    task = {
        "task_operation": "solve_equation",
        "target": "value of x",
        "requested_difficulty": "hard",
        "native_difficulty_prior": "medium",
        "actual_difficulty": "medium",
        "difficulty_design": {"estimated_actual": "hard", "status": "designed"},
        "selected_variation_id": "v1",
        "permitted_variations": [{
            "variation_id": "v1", "operation": "solve_equation", "target": "value of x",
            "givens": ["ax+b=c"], "numeric_values_may_change": True,
            "evidence_chunk_ids": ["c1"],
        }],
        "reasoning_requirements": ["Combine like terms.", "Isolate x."],
        "cognitive_transformations": ["combine_like_terms"],
        "difficulty_observation": None,
    }
    item = ProtectedQuestion(
        question_id="q1", question_type="equation", slot_id="m1_equation_1",
        question={
            "question_id": "q1", "slot_id": "m1_equation_1",
            "equation": "12x - 45 + 12x + 98 - 14x = 14",
            "solution_steps": ["10x + 53 = 14", "x = -3.9"],
            "final_answer": "-3.9", "solution_certificate": None,
        },
        task=task, evidence={"c1": "Combine like terms."},
        word_bank=[], sanitization_events=[], errors=[], requires_quantitative=True,
    )
    contract = _payload_for("difficulty", item)["difficulty_contract"]
    assert "actual_difficulty" not in contract
    assert "estimated_actual" not in json.dumps(contract)
    verdict = _compatibility_verdict(item, {
        "difficulty": {
            "status": "PASS", "issues": [],
            "actual_difficulty": "medium",
            "conceptual_difficulty": "medium",
            "numerical_difficulty": "medium",
            "difficulty_basis": "one added transformation, moderate arithmetic",
        },
    }, None)
    observation = verdict["difficulty_observation"]
    assert observation["actual_difficulty"] == "medium"
    assert observation["difficulty_basis"].startswith("one added")


def test_plan_validation_accepts_difficulty_mismatch_without_errors():
    state = _state(requested="hard")
    item = state["plans"][0]["items"]["equation"][0]
    assert item["difficulty"] == "medium"
    errors = validate_plans(
        state["plans"], [("equation", 1)], 1, {"chunk-main"},
        "hard", _profile(), [],
    )
    assert errors == []


def test_legacy_pending_adaptation_is_normalized_and_designed(monkeypatch):
    state = _state(requested="hard")
    item = state["plans"][0]["items"]["equation"][0]
    item.pop("difficulty_design", None)
    item["difficulty_adaptation"] = {
        "status": "pending", "from_difficulty": "medium", "to_difficulty": "hard",
    }
    response = _decision(item["slot_id"])
    monkeypatch.setattr(difficulty_adjustment, "request_structured", lambda *_a, **_k: response)
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    updated = result["plans"][0]["items"]["equation"][0]
    assert updated["difficulty_design"]["status"] == "designed"
    assert updated["difficulty_design"]["migrated_from_adaptation"] is True
    assert updated["base_catalog_task_id"] == "task_linear"


def test_legacy_accepted_adaptation_is_not_reprocessed(monkeypatch):
    state = _state(requested="hard")
    item = state["plans"][0]["items"]["equation"][0]
    item.pop("difficulty_design", None)
    item["difficulty_adaptation"] = {
        "status": "accepted_source_constraint",
        "from_difficulty": "medium", "to_difficulty": "hard",
    }
    calls = []
    monkeypatch.setattr(
        difficulty_adjustment, "request_structured",
        lambda *_a, **_k: calls.append(1) or _decision(item["slot_id"]),
    )
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    updated = result["plans"][0]["items"]["equation"][0]
    assert calls == []
    assert updated["difficulty_design"]["status"] == "accepted_source_constraint"


def test_non_catalog_mismatch_gets_pending_design():
    state = _state(requested="hard")
    item = state["plans"][0]["items"]["equation"][0]
    item.pop("difficulty_design", None)
    item.pop("difficulty_adaptation", None)
    difficulty_adjustment.normalize_difficulty_designs(state["plans"], "hard")
    assert item["difficulty_design"]["status"] == "pending"
    assert item["difficulty_design"]["native_prior"] == "medium"
