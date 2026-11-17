from __future__ import annotations

from copy import deepcopy

import pytest

from app.online import assessment_catalog, validator


def _task(task_id: str, target: str) -> dict:
    task = {
        "knowledge_anchor": f"linear equation {task_id}",
        "concept": "linear equation", "relationship": "ax + b = c",
        "task_operation": "solve_equation", "student_work_product": "solve for x",
        "target": target, "reasoning_mode": "symbolic_manipulation",
        "allowed_question_types": ["mcq"], "evidence_chunk_ids": ["c1"],
        "required_givens": [], "scenario_family": None, "text_answerable": True,
        "native_difficulty": "easy", "variable_givens": [],
        "complexity_boundaries": [], "exact_source_instance": False,
        "prerequisites": [], "allowed_transformations": [],
        "allowed_variations": [{
            "target": target, "operation": "solve_equation",
            "givens": ["linear equation"], "question_types": ["mcq"],
            "evidence_chunk_ids": ["c1"], "reasoning_mode": "symbolic_manipulation",
            "student_work_product": "solve for x",
            "required_response_components": ["selection"],
        }],
    }
    task.update(assessment_catalog.canonical_task_ids(task) or {})
    task["task_id"] = task_id
    return task


def _state() -> tuple[dict, dict, dict]:
    tasks = [_task("task_A", "primary target"), _task("task_B", "backup one target"),
             _task("task_C", "backup two target")]
    catalog = {"tasks": tasks, "selected_chunk_ids": ["c1"]}
    task_a = tasks[0]
    variation_a = assessment_catalog.permitted_task_variations(task_a, "mcq")[0]
    plan_item = assessment_catalog.materialize_plan_item(
        task_a, "m1_mcq_1", "easy",
        selected_variation_id=variation_a["variation_id"], subject="mathematics",
    )
    plan_item["backups"] = [
        {
            "task_id": task["task_id"],
            "variation_id": assessment_catalog.permitted_task_variations(task, "mcq")[0]["variation_id"],
            "idea_to_test": task["target"], "operation": task["task_operation"],
            "target": task["target"], "difficulty": "easy", "evidence_ids": ["c1"],
        }
        for task in tasks[1:]
    ]
    question = {
        "slot_id": "m1_mcq_1", "question_id": "q1", "question": "Solve x + 1 = 2.",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "correct_answer": "A",
    }
    verdict = {
        "question_id": "q1", "action": "REPLACE", "validation_status": "FAIL",
        "failure_code": "insufficient_evidence", "fields_to_fix": [],
        "reason": "The assigned task cannot be repaired.", "required_fixes": [],
    }
    state = {
        "num_models": 1, "difficulty": "easy",
        "assessment_catalog": catalog,
        "assessment_profile": {"domain": "STEM", "subject": "mathematics"},
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [question]}}],
        "plans": [{"model_number": 1, "items": {"mcq": [plan_item]}}],
        "repair_manifest": [{
            "model_number": 1, "slot_id": "m1_mcq_1", "question_id": "q1",
            "question_type": "mcq", "status": "invalid", "repair_type": "replace_assignment",
            "failure_code": "insufficient_evidence",
            "failure_reason": verdict["reason"], "verdict": verdict,
            "fields_to_fix": [], "validation_role": "general",
            "recovery_v3": {"version": "recovery-v3-task-alignment", "status": "ready",
                            "decision": "SWITCH_TASK", "reason": "assignment unsupported",
                            "plan": ["SWITCH_TASK"], "fallback_sequence": [],
                            "next_index": 0, "attempted_strategies": [], "decision_calls": 1},
        }],
        "question_repair_attempts": {}, "question_repair_calls": {},
        "question_repair_failures": {}, "slot_completion_attempts": {"m1_mcq_1": 0},
        "repair_candidates": [], "repair_working_candidates": {},
        "repair_replacement_memory": {}, "repair_retry_seeds": {},
        "validation_reports": [], "validation_ledger": {}, "validated_models": [],
        "pending_revalidation_ids": {}, "warnings": [], "validation_repair_history": [],
        "repair_call_logs": [], "retrieved_chunks": [{"child_id": "c1", "content": "x + 1 = 2."}],
        "duplicate_reports": [], "duplicate_scan_count": 0,
    }
    return state, plan_item, tasks


def test_switch_task_uses_shared_rank_and_advances_after_blacklist(monkeypatch):
    state, plan_item, tasks = _state()
    requested: list[str] = []

    def regenerate(_state, _exam, _model_number, _qid, plan_by_slot, *args, **kwargs):
        requested.append(plan_by_slot["m1_mcq_1"]["task_id"])
        return True, []

    monkeypatch.setattr(validator, "_regenerate_plan_slot", regenerate)

    first = validator.repair_invalid_questions(deepcopy(state))
    first_candidate = first["repair_candidates"][0]
    first_plan = first_candidate["plans"][0]["items"]["mcq"][0]
    assert requested == ["task_B"]
    assert first_plan["task_id"] == "task_B"
    assert first_plan["assignment_revision"] == 1

    # The second attempt uses the next shared-ranked task, not a backup shortcut.
    second_state = deepcopy(state)
    second_state.update(first)
    second_state["plans"] = first_candidate["plans"]
    second_state["generated_exams"] = [first_candidate["exam"]]
    second_state["repair_manifest"] = [{
        **state["repair_manifest"][0], "status": "invalid",
        "candidate_task_id": "task_B",
    }]
    second_state["repair_replacement_memory"] = deepcopy(
        first["repair_replacement_memory"]
    )
    second_state["repair_replacement_memory"]["m1_mcq_1"][
        "blacklisted_candidate_ids"
    ].append("task_B")
    second = validator.repair_invalid_questions(second_state)
    second_candidate = second["repair_candidates"][0]
    second_plan = second_candidate["plans"][0]["items"]["mcq"][0]
    assert requested == ["task_B", "task_C"]
    assert second_plan["task_id"] == "task_C"
    assert second_plan["assignment_revision"] == 2


def test_switch_variation_keeps_task_and_applies_alternate_spec(monkeypatch):
    tasks = [_task("task_A", "primary target")]
    task = tasks[0]
    task["allowed_variations"].append({
        "target": "alternate target", "operation": "solve_alternate",
        "givens": ["linear equation"], "question_types": ["mcq"],
        "evidence_chunk_ids": ["c1"], "reasoning_mode": "symbolic_manipulation",
        "student_work_product": "solve for x",
        "required_response_components": ["selection"],
    })
    catalog = {"tasks": tasks, "selected_chunk_ids": ["c1"]}
    variations = assessment_catalog.permitted_task_variations(task, "mcq")
    current = variations[0]
    alternate = next(v for v in variations if v["target"] == "alternate target")
    plan_item = assessment_catalog.materialize_plan_item(
        task, "m1_mcq_1", "easy",
        selected_variation_id=current["variation_id"], subject="mathematics",
    )
    plan_item["permitted_variations"] = [current, alternate]
    plan_item["allowed_variations"] = [alternate]
    question = {
        "slot_id": "m1_mcq_1", "question_id": "q1", "question": "Solve x + 1 = 2.",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "correct_answer": "A",
    }
    verdict = {
        "question_id": "q1", "action": "FIX_QUESTION", "validation_status": "FAIL",
        "failure_code": "insufficient_evidence", "fields_to_fix": ["question"],
        "reason": "current variation unsupported", "required_fixes": [],
    }
    state = {
        "num_models": 1, "difficulty": "easy",
        "assessment_catalog": catalog,
        "assessment_profile": {"domain": "STEM", "subject": "mathematics"},
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [question]}}],
        "plans": [{"model_number": 1, "items": {"mcq": [plan_item]}}],
        "repair_manifest": [{
            "model_number": 1, "slot_id": "m1_mcq_1", "question_id": "q1",
            "question_type": "mcq", "status": "invalid",
            "repair_type": "recovery_decision_pending",
            "failure_code": "insufficient_evidence",
            "failure_reason": verdict["reason"], "verdict": verdict,
            "fields_to_fix": ["question"], "validation_role": "grounding",
            "recovery_v3": {
                "version": "recovery-v3-task-alignment", "status": "ready",
                "decision": "SWITCH_VARIATION", "strategy": "SWITCH_VARIATION",
                "reason": "alternate variation exists", "plan": ["SWITCH_VARIATION"],
                "fallback_sequence": [], "next_index": 0,
                "attempted_strategies": [], "decision_calls": 1,
            },
        }],
        "question_repair_attempts": {}, "question_repair_calls": {},
        "question_repair_failures": {}, "slot_completion_attempts": {"m1_mcq_1": 0},
        "repair_candidates": [], "repair_working_candidates": {},
        "repair_replacement_memory": {}, "repair_retry_seeds": {},
        "validation_reports": [], "validation_ledger": {}, "validated_models": [],
        "pending_revalidation_ids": {}, "warnings": [], "validation_repair_history": [],
        "repair_call_logs": [], "retrieved_chunks": [{"child_id": "c1", "content": "x + 1 = 2."}],
        "duplicate_reports": [], "duplicate_scan_count": 0,
    }
    requested: list[str] = []

    def regenerate(_state, _exam, _model_number, _qid, plan_by_slot, *args, **kwargs):
        requested.append(plan_by_slot["m1_mcq_1"]["selected_variation_id"])
        return True, []

    monkeypatch.setattr(validator, "_regenerate_plan_slot", regenerate)
    result = validator.repair_invalid_questions(deepcopy(state))
    candidate = result["repair_candidates"][0]
    selected = candidate["plans"][0]["items"]["mcq"][0]
    assert requested == [alternate["variation_id"]]
    assert selected["task_id"] == task["task_id"]
    assert selected["selected_variation_id"] == alternate["variation_id"]
    assert selected["target"] == "alternate target"
    assert selected["task_operation"] == "solve_alternate"
    assert candidate["switched_variation_id"] == alternate["variation_id"]


def test_rejected_variation_is_skipped_without_blacklisting_whole_task(monkeypatch):
    state, _plan_item, tasks = _state()
    task_b_variation = assessment_catalog.permitted_task_variations(tasks[1], "mcq")[0]["variation_id"]
    state["repair_replacement_memory"] = {"m1_mcq_1": {
        "original_candidate_id": "task_8a5e1d6ced0328ac",
        "attempted_candidate_ids": [], "blacklisted_candidate_ids": [],
        "candidate_failures": [], "replacement_attempt_count": 0,
        "assignment_validated_attempts": {}, "rejected_variation_ids": [task_b_variation],
    }}
    requested: list[tuple[str, str]] = []

    def regenerate(_state, _exam, _model, _qid, plan_by_slot, *args, **kwargs):
        item = plan_by_slot["m1_mcq_1"]
        requested.append((item["task_id"], item["selected_variation_id"]))
        return True, []

    monkeypatch.setattr(validator, "_regenerate_plan_slot", regenerate)
    validator.repair_invalid_questions(deepcopy(state))
    assert requested and requested[0][0] == "task_B"
    assert requested[0][1] != task_b_variation


def test_exhausted_repair_plan_is_reassessed_to_task_switch(monkeypatch):
    state, _plan_item, _tasks = _state()
    target = state["repair_manifest"][0]
    target["recovery_v3"] = {
        "version": "recovery-v3-task-alignment", "status": "ready",
        "decision": "REPAIR", "reason": "localized repair",
        "plan": ["REPAIR"], "fallback_sequence": [], "next_index": 1,
        "attempted_strategies": ["REPAIR"], "decision_calls": 1,
    }
    calls: list[list[str]] = []

    def decide(contexts, **_kwargs):
        calls.append([row["question_id"] for row in contexts])
        return {"q1": {
            "version": "recovery-v3-task-alignment", "status": "ready",
            "decision": "SWITCH_TASK", "strategy": "SWITCH_TASK",
            "reason": "repair failed; another task is required",
            "plan": ["SWITCH_TASK"], "fallback_sequence": [],
            "next_index": 0, "attempted_strategies": [], "decision_calls": 1,
        }}, {"calls": 1}

    monkeypatch.setattr("app.online.recovery_decision.decide_recovery", decide)
    requested: list[str] = []

    def regenerate(_state, _exam, _model, _qid, plan_by_slot, *args, **kwargs):
        requested.append(plan_by_slot["m1_mcq_1"]["task_id"])
        return True, []

    monkeypatch.setattr(validator, "_regenerate_plan_slot", regenerate)
    result = validator.repair_invalid_questions(deepcopy(state))
    assert calls == [["q1"]]
    assert result["repair_manifest"][0]["recovery_v3"]["decision_calls"] == 2
    assert requested == ["task_B"]
