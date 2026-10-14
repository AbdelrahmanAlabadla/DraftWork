from __future__ import annotations

from copy import deepcopy
import json
import threading
import time

import pytest
from pydantic import ValidationError

from app.llm.schemas import strict_json_schema
from app.logging_conf import current_job_id, set_job_context
from app.online import assessment_catalog, difficulty_adjustment, graph, planner


def _catalog_task() -> dict:
    return {
        "task_id": "task_linear",
        "task_family_id": "family_linear",
        "coverage_id": "coverage_linear",
        "concept_id": "concept_linear",
        "relationship_id": "relationship_linear",
        "knowledge_anchor": "linear equations",
        "topic": "linear equations",
        "concept": "linear equations",
        "relationship": "linear equality",
        "task_operation": "solve_equation",
        "student_work_product": "solve equation",
        "target": "value of x",
        "reasoning_mode": "symbolic_manipulation",
        "allowed_question_types": ["equation"],
        "evidence_chunk_ids": ["chunk-main", "chunk-support"],
        "required_givens": [],
        "scenario_family": None,
        "text_answerable": True,
        "supported_difficulties": ["medium"],
        "reasoning_depth": "medium",
        "reasoning_requirements": ["Solve a linear equation."],
        "prerequisites": ["Apply inverse operations."],
    }


def _state() -> dict:
    task = _catalog_task()
    plans, issues = assessment_catalog.allocate_catalog_plan(
        {"tasks": [task]}, [("equation", 1)], 1, "hard"
    )
    return {
        "document_id": "doc",
        "tasks": [("equation", 1)],
        "num_models": 1,
        "difficulty": "hard",
        "plans": plans,
        "assessment_catalog": {"tasks": [task]},
        "assessment_profile": {
            "domain": "STEM",
            "question_types": {
                name: {
                    "support": "supported",
                    "task_operations": ["solve_equation"],
                    "form_guidance": "grounded",
                    "evidence_chunk_ids": ["chunk-main", "chunk-support"],
                }
                for name in planner.VALID_TYPES
            },
            "unsupported_visual_tasks": [],
        },
        "planning_issues": issues,
        "retrieved_chunks": [
            {"child_id": "chunk-main", "content": "Solve supported linear equations."},
            {"child_id": "chunk-support", "content": "Use distribution and combine like terms."},
        ],
        "difficulty_adjustment_history": [],
        "difficulty_adjustment_attempts": {},
    }


def _successful_response(state: dict) -> dict:
    original = state["plans"][0]["items"]["equation"][0]
    return {
        "slot_id": original["slot_id"],
        "adjustment_succeeded": True,
        "achieved_difficulty": "hard",
        "adjusted_fields": {
        "reasoning_requirements": [
            "Distribute across nested groups.",
            "Combine terms on both sides before isolating the unknown.",
        ],
            "prerequisites": ["Apply inverse operations."],
            "given_conditions": None,
            "required_givens": None,
            "idea_to_test": None,
            "strategy": "nested supported transformations",
        },
        "evidence_support": ["chunk-main", "chunk-support"],
        "evidence_justification": (
            "The assigned chunks explicitly support distribution and combining like terms."
        ),
        "reason": "The operation remains solving a linear equation with deeper supported work.",
    }


def _run(monkeypatch, response: object):
    state = _state()
    monkeypatch.setattr(
        difficulty_adjustment, "request_structured", lambda *_args, **_kwargs: response
    )
    return state, difficulty_adjustment.adjust_plan_difficulty(state)


def _multi_state(count: int) -> dict:
    state = _state()
    template = state["plans"][0]["items"]["equation"][0]
    items = []
    for index in range(1, count + 1):
        item = deepcopy(template)
        item["slot_id"] = f"m1_equation_{index}"
        items.append(item)
    state["tasks"] = [("equation", count)]
    state["plans"][0]["items"]["equation"] = items
    return state


def _response_from_prompt(prompt: str) -> dict:
    payload = json.loads(prompt)
    task = payload["task"]
    evidence_ids = list(payload["assigned_evidence"])
    return {
        "slot_id": task["immutable_identifiers"]["slot_id"],
        "adjustment_succeeded": True,
        "achieved_difficulty": task["requested_difficulty"],
        "adjusted_fields": {
            "reasoning_requirements": ["Apply two linked supported transformations."],
            "prerequisites": None, "given_conditions": None,
            "required_givens": None, "idea_to_test": None,
            "strategy": "linked transformations",
        },
        "evidence_support": evidence_ids,
        "evidence_justification": "The assigned chunks support both transformations.",
        "reason": "Supported adjustment.",
    }


def test_adjustment_schema_is_a_strict_patch_and_enumerates_assigned_evidence():
    state = _state()
    original = state["plans"][0]["items"]["equation"][0]
    model = difficulty_adjustment._response_model_for(
        original, "equation", "hard", ["chunk-main", "chunk-support"]
    )
    schema = strict_json_schema(model)
    assert schema["additionalProperties"] is False
    assert "evidence_justification" in schema["required"]
    assert schema["properties"]["evidence_support"]["items"]["enum"] == [
        "chunk-main", "chunk-support"
    ]
    patch_schema = next(
        branch for branch in schema["properties"]["adjusted_fields"]["anyOf"]
        if branch.get("type") == "object"
    )
    assert schema["properties"]["slot_id"]["enum"] == [original["slot_id"]]
    assert set(patch_schema["properties"]) == {
        "reasoning_requirements", "prerequisites", "given_conditions",
        "required_givens", "idea_to_test", "strategy",
    }
    assert "task_family_id" not in patch_schema["properties"]
    assert patch_schema["additionalProperties"] is False


def test_plan_adjustment_input_is_focused_and_excludes_pipeline_metadata(monkeypatch):
    state = _state()
    original = state["plans"][0]["items"]["equation"][0]
    original.update({
        "task_inventory": {"used": ["task-a"]},
        "previous_task_questions": [{"question": "unrelated"}],
        "validation_history": [{"status": "old"}],
    })
    captured = {}

    def fake_request(_client, prompt, **_kwargs):
        captured.update(json.loads(prompt))
        return _successful_response(state)

    monkeypatch.setattr(difficulty_adjustment, "request_structured", fake_request)
    difficulty_adjustment.adjust_plan_difficulty(state)
    task = captured["task"]
    assert set(task) == {
        "immutable_identifiers", "task_contract", "current_givens_and_constraints",
        "reasoning_profile", "actual_difficulty", "requested_difficulty",
    }
    rendered = json.dumps(captured)
    assert "task_inventory" not in rendered
    assert "previous_task_questions" not in rendered
    assert "validation_history" not in rendered
    assert captured["occupied_task_summaries"] == []


def test_valid_patch_changes_difficulty_details_but_preserves_owned_identity(monkeypatch):
    state = _state()
    original = deepcopy(state["plans"][0]["items"]["equation"][0])
    response = _successful_response(state)
    response["adjusted_fields"].update({
        "given_conditions": ["Use two supported nested linear transformations."],
        "required_givens": ["A source-grounded linear equation."],
        "idea_to_test": "Solve a linear equation requiring connected transformations.",
    })
    _, result = _run(monkeypatch, response)
    item = result["plans"][0]["items"]["equation"][0]
    for field in (
        "slot_id", "base_catalog_task_id", "task_family_id",
        "base_task_family_id", "concept", "task_operation", "target",
        "student_work_product", "source_chunk_id", "support_chunk_ids",
    ):
        assert item.get(field) == original.get(field)
    assert item["given_conditions"] == [
        "Use two supported nested linear transformations."
    ]
    assert item["actual_difficulty"] == "hard"


def test_no_op_reasoning_patch_is_rejected_and_original_is_retained(monkeypatch):
    state = _state()
    original = deepcopy(state["plans"][0]["items"]["equation"][0])
    response = _successful_response(state)
    response["adjusted_fields"]["reasoning_requirements"] = list(
        original["reasoning_requirements"]
    )
    _, result = _run(monkeypatch, response)
    item = result["plans"][0]["items"]["equation"][0]
    audit = result["difficulty_adjustment_history"][0]
    assert item["actual_difficulty"] == "medium"
    assert "reasoning_requirements_meaningfully_changed=false" in audit["rejection_reasons"]


def test_valid_grounded_adjustment_is_accepted_and_audited(monkeypatch):
    state = _state()
    response = _successful_response(state)
    _, result = _run(monkeypatch, response)
    item = result["plans"][0]["items"]["equation"][0]
    audit = result["difficulty_adjustment_history"][0]
    assert item["actual_difficulty"] == item["difficulty"] == "hard"
    assert item["difficulty_adaptation"]["status"] == "adapted"
    assert item["difficulty_adaptation"]["evidence_support"] == [
        "chunk-main", "chunk-support"
    ]
    assert audit["status"] == "succeeded"
    assert audit["rejection_reasons"] == []
    assert all(audit["acceptance_checks"].values())


@pytest.mark.parametrize(
    "slot_id",
    ["m1_mcq_1", "m1_equation_2", "m1_equation_3", "m1_word_problem_2"],
)
def test_old_full_task_contract_shapes_are_rejected_with_exact_schema_error(
    monkeypatch, slot_id,
):
    state = _state()
    response = _successful_response(state)
    response["adjusted_task"] = {"slot_id": slot_id}
    response["evidence_support"] = [
        f"Textual explanation from the assigned evidence for {slot_id}."
    ]
    _, result = _run(monkeypatch, response)
    item = result["plans"][0]["items"]["equation"][0]
    audit = result["difficulty_adjustment_history"][0]
    assert item["actual_difficulty"] == "medium"
    assert item["difficulty_adaptation"]["status"] == "failed"
    assert "structured_response_invalid" in item["adjustment_failure_reason"]
    assert "evidence_support" in audit["reason"]
    assert "adjusted_task" in audit["reason"]
    assert audit["model_decision"] == "invalid_response"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda value: value.update(slot_id="other-slot"), "slot_id"),
        (lambda value: value.update(evidence_support=["unknown-chunk"]), "evidence_support"),
        (lambda value: value.update(achieved_difficulty="medium"), "achieved_difficulty"),
        (
            lambda value: value["adjusted_fields"].update(task_family_id="different-family"),
            "task_family_id",
        ),
    ],
)
def test_invalid_identity_evidence_and_difficulty_retain_original(
    monkeypatch, mutate, expected,
):
    state = _state()
    response = _successful_response(state)
    mutate(response)
    _, result = _run(monkeypatch, response)
    item = result["plans"][0]["items"]["equation"][0]
    audit = result["difficulty_adjustment_history"][0]
    assert item["actual_difficulty"] == "medium"
    assert item["difficulty_adaptation"]["status"] == "failed"
    combined = " ".join([
        item["adjustment_failure_reason"], audit["reason"],
        *audit["rejection_reasons"],
    ])
    assert expected in combined


def test_patch_cannot_change_semantic_family_fields(monkeypatch):
    state = _state()
    response = _successful_response(state)
    response["adjusted_fields"].update({
        "task_operation": "derive_quadratic_formula",
        "target": "quadratic roots",
        "student_work_product": "derive a quadratic relationship",
    })
    _, result = _run(monkeypatch, response)
    audit = result["difficulty_adjustment_history"][0]
    assert audit["model_decision"] == "invalid_response"
    assert "task_operation" in audit["reason"]


def test_malformed_response_retains_original_with_schema_diagnostic(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        difficulty_adjustment,
        "request_structured",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValidationError.from_exception_data("malformed", [])
        ),
    )
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    item = result["plans"][0]["items"]["equation"][0]
    assert item["actual_difficulty"] == "medium"
    assert item["adjustment_failure_reason"].startswith("structured_response_invalid:")
    assert result["difficulty_adjustment_history"][0]["model_decision"] == "invalid_response"


def test_agent_decline_for_unsupported_transformation_retains_original(monkeypatch):
    state = _state()
    response = _successful_response(state)
    response.update({
        "adjustment_succeeded": False,
        "adjusted_fields": None,
        "evidence_support": [],
        "evidence_justification": "No assigned chunk supports the required prerequisite.",
        "reason": "A Hard transformation would introduce unsupported quadratic operations.",
    })
    _, result = _run(monkeypatch, response)
    item = result["plans"][0]["items"]["equation"][0]
    audit = result["difficulty_adjustment_history"][0]
    assert item["actual_difficulty"] == "medium"
    assert item["adjustment_failure_reason"].startswith("model_declined:")
    assert audit["model_decision"] == "declined"


def test_five_adjustments_run_concurrently_and_merge_in_slot_order(monkeypatch):
    state = _multi_state(7)
    lock = threading.Lock()
    active = 0
    maximum = 0
    calls: list[str] = []
    contexts: list[str | None] = []

    def fake_request(_client, prompt, **_kwargs):
        nonlocal active, maximum
        slot_id = json.loads(prompt)["task"]["immutable_identifiers"]["slot_id"]
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(slot_id)
            contexts.append(current_job_id())
        # Reverse completion order within each wave to prove merge order is stable.
        time.sleep(0.015 * (8 - int(slot_id.rsplit("_", 1)[1])))
        response = _response_from_prompt(prompt)
        with lock:
            active -= 1
        return response

    monkeypatch.setattr(difficulty_adjustment, "MAX_PARALLEL_DIFFICULTY_ADJUSTMENTS", 5)
    monkeypatch.setattr(difficulty_adjustment, "request_structured", fake_request)
    prior_job = current_job_id() or "-"
    set_job_context("job-concurrency-test", "doc")
    try:
        result = difficulty_adjustment.adjust_plan_difficulty(state)
    finally:
        set_job_context(prior_job)

    assert maximum == 5
    assert len(calls) == len(set(calls)) == 7
    assert contexts == ["job-concurrency-test"] * 7
    assert list(result["difficulty_adjustment_attempts"].values()) == [1] * 7
    assert [row["slot_id"] for row in result["difficulty_adjustment_history"]] == [
        f"m1_equation_{index}" for index in range(1, 8)
    ]
    assert all(row["status"] == "succeeded" for row in result["difficulty_adjustment_history"])


def test_parallel_rate_limit_failure_does_not_discard_other_results(monkeypatch):
    from app.llm.deepseek import DeepSeekRateLimitError

    state = _multi_state(6)

    def fake_request(_client, prompt, **_kwargs):
        slot_id = json.loads(prompt)["task"]["immutable_identifiers"]["slot_id"]
        if slot_id == "m1_equation_3":
            raise DeepSeekRateLimitError("provider capacity reached")
        return _response_from_prompt(prompt)

    monkeypatch.setattr(difficulty_adjustment, "MAX_PARALLEL_DIFFICULTY_ADJUSTMENTS", 5)
    monkeypatch.setattr(difficulty_adjustment, "request_structured", fake_request)
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    history = result["difficulty_adjustment_history"]
    assert [row["slot_id"] for row in history] == [
        f"m1_equation_{index}" for index in range(1, 7)
    ]
    assert [row["status"] for row in history].count("succeeded") == 5
    failed = next(row for row in history if row["slot_id"] == "m1_equation_3")
    assert failed["status"] == "failed"
    assert "DeepSeekRateLimitError" in failed["reason"]
    assert result["difficulty_adjustment_attempts"]["m1_equation_3"] == 1


@pytest.mark.parametrize(
    ("native", "requested"),
    [("medium", "easy"), ("easy", "medium"), ("easy", "hard"), ("hard", "easy")],
)
def test_fallback_preserves_native_level_and_routes_to_adjustment(native, requested):
    task = _catalog_task()
    task.update({
        "native_difficulty": native,
        "supported_difficulties": [native],
        "reasoning_depth": native,
    })
    plans, issues = assessment_catalog.allocate_catalog_plan(
        {"tasks": [task]}, [("equation", 1)], 1, requested
    )
    item = plans[0]["items"]["equation"][0]
    assert issues == []
    assert item["requested_difficulty"] == requested
    assert item["difficulty"] == item["actual_difficulty"] == native
    assert item["reasoning_depth"] == native
    assert item["difficulty_adaptation"] == {
        "status": "pending", "from_difficulty": native,
        "to_difficulty": requested, "base_task_id": "task_linear",
        "strategy": None, "failed_strategies": [],
    }
    assert planner.validate_plans(
        plans, [("equation", 1)], 1, {"chunk-main", "chunk-support"},
        requested, _state()["assessment_profile"], [],
    ) == []
    assert graph._route_after_plan({
        "plans": plans, "plan_errors": [], "planning_issues": [], "plan_attempts": 1,
    }) == "adjust_plan_difficulty"


def test_mixed_allocation_preserves_native_levels_without_pending_adjustment():
    tasks = []
    for level in ("easy", "medium", "hard"):
        task = deepcopy(_catalog_task())
        task.update({
            "task_id": f"task_{level}", "task_family_id": f"family_{level}",
            "coverage_id": f"coverage_{level}", "concept_id": f"concept_{level}",
            "concept": f"linear equations {level}", "target": f"value of x {level}",
            "native_difficulty": level, "supported_difficulties": [level],
            "reasoning_depth": level,
        })
        tasks.append(task)
    plans, issues = assessment_catalog.allocate_catalog_plan(
        {"tasks": tasks}, [("equation", 3)], 1, "mix"
    )
    items = plans[0]["items"]["equation"]
    assert issues == []
    assert {item["difficulty"] for item in items} == {"easy", "medium", "hard"}
    assert all(item["requested_difficulty"] == "mix" for item in items)
    assert all(item["difficulty_adaptation"] is None for item in items)


def test_successful_medium_to_easy_adjustment_updates_all_current_level_fields(monkeypatch):
    task = _catalog_task()
    task["native_difficulty"] = "medium"
    plans, issues = assessment_catalog.allocate_catalog_plan(
        {"tasks": [task]}, [("equation", 1)], 1, "easy"
    )
    state = _state()
    state.update({"difficulty": "easy", "plans": plans, "planning_issues": issues})
    original = plans[0]["items"]["equation"][0]
    monkeypatch.setattr(difficulty_adjustment, "request_structured", lambda *_a, **_k: {
        "slot_id": original["slot_id"], "adjustment_succeeded": True,
        "achieved_difficulty": "easy",
        "adjusted_fields": {
            "reasoning_requirements": [
                "Apply one immediately identifiable inverse operation."
            ],
            "prerequisites": [], "given_conditions": None,
            "required_givens": None, "idea_to_test": None,
            "strategy": "single supported inverse operation",
        },
        "evidence_support": ["chunk-main"],
        "evidence_justification": "The assigned chunk supports that operation.",
        "reason": "The task is simplified without changing its identity.",
    })
    result = difficulty_adjustment.adjust_plan_difficulty(state)
    item = result["plans"][0]["items"]["equation"][0]
    assert result["plan_errors"] == []
    assert item["requested_difficulty"] == "easy"
    assert item["difficulty"] == item["actual_difficulty"] == item["reasoning_depth"] == "easy"
    assert item["difficulty_adaptation"]["status"] == "adapted"


def test_failed_job_ten_slot_shape_is_pending_adjustment_not_plan_repair():
    slot_counts = [("mcq", 2), ("true_false", 1), ("equation", 4), ("word_problem", 3)]
    historical_ids = [
        "task_2734a0318fb0f5e9", "task_b324a752cb57cf8d",
        "task_290d0aefde0e106d", "task_0f171191d14a71bc",
        "task_4390c8ee1fff0149", "task_82d5f8d2cd3d093b",
        "task_7c8b1a6c0ebd574e", "task_44201eaf75b7960f",
        "task_c00c932d24f86496", "task_ea141744837c8b21",
    ]
    qtypes = [qtype for qtype, count in slot_counts for _ in range(count)]
    tasks = []
    for index, (task_id, qtype) in enumerate(zip(historical_ids, qtypes), start=1):
        task = deepcopy(_catalog_task())
        task.update({
            "task_id": task_id, "task_family_id": f"family_{index}",
            "coverage_id": f"coverage_{index}", "concept_id": f"concept_{index}",
            "concept": f"supported concept {index}", "target": f"result {index}",
            "student_work_product": f"produce result {index}",
            "allowed_question_types": [qtype], "native_difficulty": "medium",
            "supported_difficulties": ["medium"], "reasoning_depth": "medium",
        })
        tasks.append(task)
    plans, issues = assessment_catalog.allocate_catalog_plan(
        {"tasks": tasks}, slot_counts, 1, "easy"
    )
    items = [item for values in plans[0]["items"].values() for item in values]
    assert issues == []
    assert len(items) == 10
    assert {item["base_catalog_task_id"] for item in items} == set(historical_ids)
    assert all(item["difficulty"] == item["actual_difficulty"] == "medium" for item in items)
    assert all(item["reasoning_depth"] == "medium" for item in items)
    assert all((item["difficulty_adaptation"] or {}).get("status") == "pending" for item in items)
    assert graph._route_after_plan({
        "plans": plans, "plan_errors": [], "planning_issues": [], "plan_attempts": 1,
    }) == "adjust_plan_difficulty"


def _generated_equation_response(*, evidence_support=None) -> dict:
    return {
        "adjustment_succeeded": True,
        "question_type": "equation",
        "actual_difficulty": "hard",
        "direction": "medium_to_hard",
        "changed_reasoning_requirements": [
            "Distribute nested factors and combine terms on both sides."
        ],
        "evidence_support": evidence_support or ["chunk-main", "chunk-support"],
        "evidence_justification": "Both operations are taught in the assigned chunks.",
        "reason": "The equation now requires linked supported transformations.",
        "adjusted_question": {
            "slot_id": "m1_equation_1",
            "equation": "Solve:\n\\[3(2x-4)+5=2x+9\\]",
            "solution_steps": ["Expand both sides.", "Collect terms and solve."],
            "final_answer": "x=4",
            "solution_certificate": None,
        },
    }


def _generated_adjustment_state() -> tuple[dict, dict, dict]:
    state = {
        "difficulty": "hard",
        "assessment_profile": {"domain": "STEM"},
        "retrieved_chunks": [
            {"child_id": "chunk-main", "content": "Apply distribution."},
            {"child_id": "chunk-support", "content": "Combine like terms."},
        ],
    }
    plan = {
        "requested_difficulty": "hard", "actual_difficulty": "medium",
        "source_chunk_id": "chunk-main", "support_chunk_ids": ["chunk-support"],
    }
    question = {
        "slot_id": "m1_equation_1", "question_id": "q-original",
        "equation": "Solve:\n\\[2x+5=15\\]",
        "solution_steps": ["Subtract 5.", "Divide by 2."],
        "final_answer": "x=5", "solution_certificate": None,
    }
    return state, plan, question


def test_generated_adjustment_uses_strict_decision_and_preserves_question_id(monkeypatch):
    state, plan, question = _generated_adjustment_state()
    monkeypatch.setattr(
        difficulty_adjustment, "request_structured",
        lambda *_args, **_kwargs: _generated_equation_response(),
    )
    adjusted, audit = difficulty_adjustment.adjust_generated_question(
        state=state, qtype="equation", question=question, plan=plan,
        previous_questions=[],
    )
    assert adjusted["question_id"] == "q-original"
    assert adjusted["equation"] != question["equation"]
    assert audit["status"] == "succeeded"
    assert audit["verification_pending"] is True
    assert audit["evidence_support"] == ["chunk-main", "chunk-support"]
    assert audit["changed_reasoning_requirements"]
    assert all(audit["acceptance_checks"].values())


def test_generated_adjustment_uses_same_focused_task_contract(monkeypatch):
    state, plan, question = _generated_adjustment_state()
    plan.update({"task_inventory": {"unused": []}, "validation_history": ["old"]})
    captured = {}

    def fake_request(_client, prompt, **_kwargs):
        captured.update(json.loads(prompt))
        return _generated_equation_response()

    monkeypatch.setattr(difficulty_adjustment, "request_structured", fake_request)
    difficulty_adjustment.adjust_generated_question(
        state=state, qtype="equation", question=question, plan=plan,
        previous_questions=[],
    )
    assert set(captured["task"]) == {
        "immutable_identifiers", "task_contract", "current_givens_and_constraints",
        "reasoning_profile", "actual_difficulty", "requested_difficulty",
    }
    rendered = json.dumps(captured["task"])
    assert "task_inventory" not in rendered
    assert "validation_history" not in rendered


def test_generated_adjustment_rejects_unassigned_evidence_and_retains_candidate(monkeypatch):
    state, plan, question = _generated_adjustment_state()
    monkeypatch.setattr(
        difficulty_adjustment, "request_structured",
        lambda *_args, **_kwargs: _generated_equation_response(
            evidence_support=["not-assigned"]
        ),
    )
    adjusted, audit = difficulty_adjustment.adjust_generated_question(
        state=state, qtype="equation", question=question, plan=plan,
        previous_questions=[],
    )
    assert adjusted == question
    assert audit["status"] == "failed"
    assert "evidence_support" in audit["reason"]


@pytest.mark.parametrize("qtype", sorted(planner.VALID_TYPES))
def test_generated_adjustment_schema_wraps_every_question_type(qtype):
    model = difficulty_adjustment._generated_response_model_for(
        qtype, f"m1_{qtype}_1", "hard", "medium", ["chunk-1"]
    )
    schema = strict_json_schema(model)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["question_type"]["enum"] == [qtype]
    assert schema["properties"]["evidence_support"]["items"]["enum"] == [
        "chunk-1"
    ]
