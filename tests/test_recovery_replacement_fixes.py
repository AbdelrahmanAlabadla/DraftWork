from __future__ import annotations

from copy import deepcopy

from app.online import duplicate_pipeline, graph, validator


def test_m1_blocked_route_becomes_replacement_only_and_survives_merge():
    target = {
        "model_number": 1,
        "slot_id": "m1_mcq_2",
        "status": "invalid",
        "repair_type": "repair_question",
        "failure_code": "insufficient_evidence",
    }
    memory = {}
    validator._mark_blocked_repair_route(
        target, memory,
        route_status="VALIDATION_BLOCKED",
        input_fingerprint="same-input",
        reason="specialized repair blocked: VALIDATION_BLOCKED",
    )
    assert target["status"] == "validation_blocked"
    assert target["repair_type"] == "replace_assignment"
    assert memory["blocked_route_fingerprint"] == "same-input"

    fresh = [{
        "model_number": 1, "slot_id": "m1_mcq_2", "status": "invalid",
        "repair_type": "repair_question", "failure_code": "insufficient_evidence",
    }]
    merged = validator._merge_repair_manifest([target], fresh)[0]
    assert merged["status"] == "validation_blocked"
    assert merged["repair_type"] == "replace_assignment"
    decision = validator.recovery_eligibility(
        merged, failures=0, calls=0, completion_attempts=0,
        completion_enabled=True,
    )
    assert decision["selected_route"] is None
    assert decision["replacement_allowed"] is True


def test_completion_pending_does_not_reopen_blocked_content_repair():
    old = [{
        "model_number": 1, "slot_id": "m1_mcq_2",
        "status": "completion_pending", "repair_type": "replace_assignment",
        "failure_code": "completion_candidate_unavailable",
    }]
    fresh = [{
        "model_number": 1, "slot_id": "m1_mcq_2",
        "status": "invalid", "repair_type": "repair_question",
        "failure_code": "insufficient_evidence",
    }]
    merged = validator._merge_repair_manifest(old, fresh)[0]
    assert merged["status"] == "completion_pending"
    assert merged["repair_type"] == "replace_assignment"


def test_medium_numerical_replacement_is_retained_as_source_constraint():
    task = {
        "slot_id": "m2_true_false_3",
        "task_id": "task_d156ea069ffd7997",
        "selected_variation_id": "variation_394ba0b7db766665",
        "difficulty": "medium", "actual_difficulty": "medium",
        "requested_difficulty": "easy",
        "variable_givens": ["coefficients", "candidate triple"],
        "selected_variation": {"numeric_values_may_change": True},
    }
    assert validator._finalize_replacement_difficulty_fallback(
        task, requested="easy",
        adjustment_audit={"failure": "The operation cannot honestly become easy."},
    ) is True
    assert task["difficulty"] == "medium"
    assert task["actual_difficulty"] == "medium"
    assert task["difficulty_status"] == "accepted_source_constraint"
    assert task["difficulty_adaptation"]["further_adjustment_possible"] is False


def test_failed_instance_does_not_blacklist_another_instance_of_same_task(monkeypatch):
    candidates = [{
        "task_id": "shared-task", "selected_variation_id": "v1",
        "replacement_instance_sequence": 2,
        "task_operation": "check_candidate_solution_in_each_equation",
        "target": "judge the candidate triple", "source_evidence_ids": ["c1"],
        "evidence_chunk_ids": ["c1"],
        "allowed_question_types": ["true_false"], "text_answerable": True,
        "allowed_variations": [{
            "variation_id": "v1", "question_types": ["true_false"],
            "operation": "check_candidate_solution_in_each_equation",
            "target": "judge the candidate triple", "evidence_chunk_ids": ["c1"],
            "numeric_values_may_change": True,
        }],
        "requires_supplied_graph": False,
    }]
    failed = [{
        "candidate_id": "shared-task",
        "candidate_instance_key": "shared-task|v1||1",
        "reason": "prior numerical instance was ambiguous",
    }]
    monkeypatch.setattr(
        "app.llm.factory.create_duplicate_replacement_client", lambda: object(),
    )
    monkeypatch.setattr(
        "app.llm.client.request_structured",
        lambda *_a, **_k: {
            "slot_to_replace": "m2_true_false_3",
            "replacement_task_id": "shared-task",
            "replacement_fitb_entry_id": None,
            "checked_against_active_slots": True,
        },
    )
    chosen, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="m2_true_false_3", qtype="true_false",
        conflict_slots=[], conflict_edges=[], candidates=candidates,
        active_items=[], blacklist=set(), previous_failures=failed,
    )
    assert chosen is candidates[0]
    assert context["previously_failed_candidate_instances"] == [
        "shared-task|v1||1"
    ]


def test_pre_generation_feasibility_rejects_missing_evidence_and_visual():
    candidate = {
        "task_id": "curve-task", "selected_variation_id": "v1",
        "task_operation": "interpret_supplied_graph", "target": "judge the curve",
        "source_evidence_ids": [], "allowed_question_types": ["true_false"],
        "text_answerable": True, "requires_supplied_graph": True,
    }
    reasons = duplicate_pipeline.replacement_feasibility_reasons(
        candidate, "true_false",
    )
    assert "missing_source_evidence" in reasons
    assert "unsupported_supplied_graph" in reasons


def test_blocked_replacement_budget_remains_bounded():
    state = {
        "repair_candidates": [],
        "repair_manifest": [{
            "slot_id": "m1_mcq_2", "status": "validation_blocked",
            "repair_type": "replace_assignment",
        }],
        "question_repair_failures": {}, "question_repair_calls": {},
        "slot_completion_attempts": {"m1_mcq_2": 3},
    }
    assert graph._route_after_validation(deepcopy(state)) == "assemble_exams"


def test_duplicate_content_exhaustion_uses_remaining_completion_budget():
    decision = validator.recovery_eligibility(
        {
            "slot_id": "m2_mcq_3", "status": "invalid",
            "repair_type": "repair_question", "validation_role": "duplicate",
            "failure_code": "duplicate_question",
        },
        failures=validator.MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT,
        calls=validator.MAX_CONTENT_REPAIR_CALLS_PER_SLOT,
        completion_attempts=1,
        completion_enabled=True,
        content_attempt_limit=validator.MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT,
    )
    assert decision["content_exhausted"] is True
    assert decision["replacement_allowed"] is True
    assert decision["selected_route"] is None
    assert decision["replacement_allowed"] is True
