from __future__ import annotations

import json
from copy import deepcopy

import pytest

from app.online import prompts
from app.online import validator as val
from app.online.multi_validator import clear_job_validation_cache


class _OfflineClient:
    provider_name = "offline"
    model = "mock"


def _plan(*, task_id: str = "task-force", chunk_id: str = "assigned-1") -> dict:
    return {
        "model_number": 1,
        "items": {
            "word_problem": [{
                "slot_id": "m1_word_problem_1",
                "question_type": "word_problem",
                "task_id": task_id,
                "source_chunk_id": chunk_id,
                "support_chunk_ids": [],
                "selected_variation_id": "two-step-force",
                "permitted_variations": [{
                    "variation_id": "two-step-force",
                    "question_types": ["word_problem"],
                    "givens": ["mass", "initial velocity", "final velocity", "time"],
                    "evidence_chunk_ids": [chunk_id],
                }],
                "concept": "Newton's second law",
                "knowledge_focus": "derive acceleration before calculating force",
                "idea_to_test": "calculate acceleration, then force",
                "target": "two-step force calculation",
                "task_operation": "calculate_then_apply",
                "required_givens": ["mass", "initial velocity", "final velocity", "time"],
                "given_conditions": ["constant acceleration"],
                "reasoning_mode": "calculation",
                "reasoning_depth": "two_step",
                "reasoning_requirements": ["derive acceleration", "apply F = ma"],
                "difficulty": "medium",
                "requested_difficulty": "medium",
                "requires_quantitative_validation": True,
            }],
        },
    }


def _old_question() -> dict:
    return {
        "question_id": "model1_word_problem_1",
        "slot_id": "m1_word_problem_1",
        "selected_variation_id": "two-step-force",
        "question": "A 5 kg object accelerates at 3 m/s². Find the force.",
        "solution_steps": ["F = ma", "F = 5 × 3"],
        "final_answer": "15 N",
        "solution_certificate": _certificate(["F = ma", "F = 5 × 3 = 15 N"]),
    }


def _certificate(steps: list[str]) -> dict:
    return {
        "method": "Newton's second law",
        "important_givens": ["m = 5 kg"],
        "calculation_steps": steps,
        "final_answer": "15 N",
        "units": "N",
        "matching_correct_option": None,
        "verification_method": "Dimensional check: kg·m/s² = N.",
    }


def _replacement(*, valid: bool = True) -> dict:
    if valid:
        question = (
            "A 5 kg object accelerates from 0 m/s to 6 m/s in 2 s. "
            "Calculate its acceleration and then the force."
        )
        steps = ["a = (6 - 0) / 2 = 3 m/s²", "F = ma = 5 × 3 = 15 N"]
    else:
        question = "A 5 kg object accelerates at 3 m/s². Find the force."
        steps = ["F = ma = 5 × 3 = 15 N"]
    return {
        "slot_id": "m1_word_problem_1",
        "selected_variation_id": "two-step-force",
        "question": question,
        "solution_steps": steps,
        "final_answer": "15 N",
        "solution_certificate": _certificate(steps),
    }


def _state(*, task_id: str = "task-force", chunk_id: str = "assigned-1") -> dict:
    return {
        "job_id": f"offline-regeneration-{task_id}",
        "generated_exams": [{
            "model_number": 1,
            "questions": {"word_problem": [_old_question()]},
        }],
        "plans": [_plan(task_id=task_id, chunk_id=chunk_id)],
        "retrieved_chunks": [
            {"child_id": chunk_id, "content": "القوة تساوي الكتلة × التسارع. F = ma."},
            {"child_id": "unassigned-9", "content": "UNASSIGNED_SECRET_EVIDENCE"},
        ],
        "assessment_profile": {"domain": "STEM", "question_types": {}},
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "pending_revalidation_ids": {},
        "warnings": [],
    }


def _payload(prompt: str) -> dict:
    raw = prompt.split("<VALIDATION_DATA>\n", 1)[1].split(
        "\n</VALIDATION_DATA>", 1
    )[0]
    return json.loads(raw)


def _validator_response(
    prompt: str, *, revalidation_passes: bool = True, initial_failure: bool = True,
) -> dict:
    payload = _payload(prompt)
    role = payload["validator"]
    rows = []
    for item in payload["questions"]:
        context = item.get("revalidation") or {}
        checks = []
        for issue in context.get("original_failure_findings") or []:
            checks.append({
                "issue_id": issue["issue_id"],
                "resolved": revalidation_passes,
                "reason": (
                    "The final candidate now requires the planned two steps."
                    if revalidation_passes
                    else "The candidate still requires only one calculation."
                ),
                "remaining_defect": "" if revalidation_passes else "Still one-step.",
            })
        if role == "difficulty" and not context and initial_failure:
            status = "FAIL"
            issues = [{
                "failure_code": "reasoning_complexity_mismatch",
                "fields_to_fix": ["question", "solution_steps"],
                "reason": (
                    "The original question required only one calculation step, but the "
                    "selected task requires two-step reasoning."
                ),
                "expected_fix": "Require acceleration to be derived before force.",
            }]
        elif role == "difficulty" and not revalidation_passes:
            status = "FAIL"
            issues = [{
                "failure_code": "reasoning_complexity_mismatch",
                "fields_to_fix": ["question", "solution_steps"],
                "reason": "The regenerated question still requires only one calculation.",
                "expected_fix": "Generate a genuine two-step problem.",
            }]
        else:
            status, issues = "PASS", []
        rows.append({
            "question_id": item["question_id"],
            "status": status,
            "issues": issues,
            "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
            "observed_difficulty": (
                ("easy" if status == "FAIL" else "medium")
                if role == "difficulty" else None
            ),
            "requested_difficulty": "medium" if role == "difficulty" else None,
            "declared_difficulty": "medium" if role == "difficulty" else None,
            "calculation_evidence": [],
            "original_issue_checks": checks,
        })
    return {"validator": role, "verdicts": rows}


def _install_offline_validation(monkeypatch, responder):
    monkeypatch.setattr("app.llm.factory.create_review_client", lambda **_kwargs: _OfflineClient())
    monkeypatch.setattr("app.online.multi_validator.request_structured", responder)
    monkeypatch.setattr("app.online.multi_validator.time.sleep", lambda _seconds: None)


def _initial_failure(monkeypatch, state: dict) -> dict:
    clear_job_validation_cache(state["job_id"])
    def provider(_client, prompt, **_kwargs):
        return _validator_response(prompt)

    _install_offline_validation(monkeypatch, provider)
    initial = val.validate_generated_questions(state)
    merged = {**state, **initial}
    target = merged["repair_manifest"][0]
    assert target["failure_code"] == "reasoning_complexity_mismatch"
    # A prior content candidate was rejected, so existing bounded recovery
    # policy escalates the next attempt from specialized repair to regeneration.
    target["history"] = [{
        "attempt": 1,
        "status": "rejected",
        "repair_action": "repair_question",
        "new_failure_code": "reasoning_complexity_mismatch",
        "new_failure_reason": target["failure_reason"],
    }]
    # Initial-validation PASS rows are audit evidence, but these tests start the
    # candidate lifecycle with a fresh job-local cache just as a resumed worker
    # would. Revalidation then proves its own retry preservation below.
    clear_job_validation_cache(state["job_id"])
    return merged


def _stage_regeneration(monkeypatch, state: dict, *, valid: bool = True):
    prompts_seen: list[str] = []

    def generator(_client, prompt, **_kwargs):
        prompts_seen.append(prompt)
        return {"questions": [_replacement(valid=valid)]}

    monkeypatch.setattr("app.llm.factory.create_llm_client", lambda **_kwargs: _OfflineClient())
    monkeypatch.setattr("app.online.exam_builder.request_structured", generator)
    monkeypatch.setattr(
        "app.online.difficulty_adjustment.request_structured",
        lambda _client, _prompt, **_kwargs: {
            "slot_id": "m1_word_problem_1",
            "adjustment_succeeded": True,
            "achieved_difficulty": "medium",
            "adjusted_fields": {
                "reasoning_requirements": [
                    "derive acceleration from initial velocity, final velocity, and time",
                    "apply F = ma only after deriving acceleration",
                ],
                "prerequisites": ["constant acceleration"],
                "given_conditions": ["constant acceleration"],
                "required_givens": ["mass", "initial velocity", "final velocity", "time"],
                "idea_to_test": "calculate acceleration, then force",
                "strategy": "Require both supported calculation steps.",
            },
            "evidence_support": ["assigned-1"],
            "evidence_justification": "The assigned evidence supports F = ma.",
            "reason": "The revised assignment now requires the requested two-step reasoning.",
        },
    )
    staged = val.repair_invalid_questions(state)
    assert staged["repair_candidates"]
    return staged, prompts_seen


def test_regeneration_context_is_bounded_optional_and_preserves_unicode():
    item = _plan()["items"]["word_problem"][0]
    _, ordinary = prompts.build_prompt(
        "word_problem", 1, "source", planned_items=[item], language="ar",
    )
    assert "regeneration_context=" not in ordinary

    recovery = {
        **item,
        "recovery_failure_code": "reasoning_complexity_mismatch",
        "recovery_failure_reason": "السؤال يحتاج خطوتين: احسب a ثم استخدم F = ma. " + "x" * 2500,
        "recovery_attempt": 2,
        "assigned_sources": {"assigned-1": "نص عربي مع F = ma"},
    }
    _, regenerated = prompts.build_prompt(
        "word_problem", 1, "ignored", planned_items=[recovery], language="ar",
    )
    assert '"previous_failure_code": "reasoning_complexity_mismatch"' in regenerated
    assert "السؤال يحتاج خطوتين: احسب a ثم استخدم F = ma." in regenerated
    assert '"recovery_attempt": 2' in regenerated
    assert "untrusted diagnostic data" in regenerated
    rendered_context = prompts._regeneration_context(recovery)
    assert len(json.loads(rendered_context)["previous_failure_reason"]) == 2000


def test_validation_regeneration_revalidation_commits_and_prompt_is_isolated(monkeypatch):
    state = _initial_failure(monkeypatch, _state())
    original = deepcopy(state["generated_exams"][0]["questions"]["word_problem"][0])
    staged, prompts_seen = _stage_regeneration(monkeypatch, state)

    assert state["generated_exams"][0]["questions"]["word_problem"][0] == original
    final_prompt = prompts_seen[-1]
    assert "reasoning_complexity_mismatch" in final_prompt
    assert "selected task requires two-step reasoning" in final_prompt
    assert "selected_variation_id=two-step-force" in final_prompt
    assert "derive acceleration from initial velocity, final velocity, and time" in final_prompt
    assert "القوة تساوي الكتلة × التسارع. F = ma." in final_prompt
    assert "UNASSIGNED_SECRET_EVIDENCE" not in final_prompt

    def provider(_client, prompt, **_kwargs):
        return _validator_response(prompt, revalidation_passes=True)

    _install_offline_validation(monkeypatch, provider)
    completed = val.validate_generated_questions({**state, **staged})
    committed = completed["generated_exams"][0]["questions"]["word_problem"][0]
    assert committed["question"] == _replacement()["question"]
    assert committed["solution_steps"] == _replacement()["solution_steps"]
    assert completed["repair_candidates"] == []
    assert completed["repair_manifest"] == []


def test_regenerated_candidate_rejected_and_original_is_preserved(monkeypatch):
    state = _initial_failure(monkeypatch, _state())
    original = deepcopy(state["generated_exams"][0]["questions"]["word_problem"][0])
    staged, _ = _stage_regeneration(monkeypatch, state, valid=False)

    def provider(_client, prompt, **_kwargs):
        return _validator_response(prompt, revalidation_passes=False)

    _install_offline_validation(monkeypatch, provider)
    rejected = val.validate_generated_questions({**state, **staged})
    assert rejected["generated_exams"][0]["questions"]["word_problem"][0] == original
    target = rejected["repair_manifest"][0]
    assert target["failure_code"] == "reasoning_complexity_mismatch"
    assert "still requires only one calculation" in target["failure_reason"]
    assert target["history"][-1]["new_failure_code"] == "reasoning_complexity_mismatch"
    assert "still requires only one calculation" in target["history"][-1]["new_failure_reason"]


def test_operational_revalidation_retries_only_timeout_role_without_regeneration(monkeypatch):
    state = _initial_failure(monkeypatch, _state())
    staged, prompts_seen = _stage_regeneration(monkeypatch, state)
    generation_calls = len(prompts_seen)
    role_calls: list[str] = []
    difficulty_failures = {"remaining": 3}

    def timeout_provider(_client, prompt, **_kwargs):
        role = _payload(prompt)["validator"]
        role_calls.append(role)
        if role == "difficulty" and difficulty_failures["remaining"]:
            difficulty_failures["remaining"] -= 1
            raise TimeoutError("mock revalidation timeout")
        return _validator_response(prompt, revalidation_passes=True)

    _install_offline_validation(monkeypatch, timeout_provider)
    pending = val.validate_generated_questions({**state, **staged})
    assert pending["repair_manifest"][0]["status"] == "validation_pending"
    assert pending["repair_candidates"]
    pending_state = {**state, **staged, **pending}
    assert pending_state["question_repair_attempts"] == staged["question_repair_attempts"]
    assert len(prompts_seen) == generation_calls

    before_retry = len(role_calls)
    completed = val.validate_generated_questions(pending_state)
    retry_roles = role_calls[before_retry:]
    assert retry_roles == ["difficulty"]
    assert completed["repair_candidates"] == []
    assert completed["generated_exams"][0]["questions"]["word_problem"][0][
        "question"
    ] == _replacement()["question"]
    assert len(prompts_seen) == generation_calls


def test_operational_revalidation_exhaustion_blocks_without_content_recovery(monkeypatch):
    state = _initial_failure(monkeypatch, _state())
    staged, prompts_seen = _stage_regeneration(monkeypatch, state)
    generation_calls = len(prompts_seen)
    content_attempts = deepcopy(staged["question_repair_attempts"])

    def timeout_provider(_client, prompt, **_kwargs):
        if _payload(prompt)["validator"] == "difficulty":
            raise TimeoutError("persistent mock revalidation timeout")
        return _validator_response(prompt, revalidation_passes=True)

    _install_offline_validation(monkeypatch, timeout_provider)
    current = {**state, **staged}
    # Candidate-level retries remain independent of each call's internal
    # provider retries.
    for _ in range(val.REVALIDATION_OPERATIONAL_RETRIES + 1):
        delta = val.validate_generated_questions(current)
        current = {**current, **delta}

    assert current["repair_candidates"] == []
    assert current["repair_manifest"][0]["status"] == "validation_blocked"
    assert current["blocked_revalidation_candidates"][0][
        "candidate_status"
    ] == "VALIDATION_BLOCKED"
    assert current["question_repair_attempts"] == content_attempts
    assert current["generated_exams"][0]["questions"]["word_problem"][0] == _old_question()
    assert len(prompts_seen) == generation_calls


def test_catalog_replacement_plan_uses_new_assignment_evidence_and_invalidates_cache(monkeypatch):
    state = _state(task_id="replacement-task", chunk_id="replacement-child")
    state["job_id"] = "offline-catalog-replacement"
    state["plans"][0]["items"]["word_problem"][0].update({
        "recovery_failure_code": "evidence_mismatch",
        "recovery_failure_reason": "The prior assignment used unsuitable evidence.",
        "recovery_attempt": 2,
    })
    plan_by_slot, source_by_chunk = val._plan_context_for_model(state, 1)
    prompts_seen: list[str] = []

    def generator(_client, prompt, **_kwargs):
        prompts_seen.append(prompt)
        return {"questions": [_replacement()]}

    monkeypatch.setattr("app.llm.factory.create_llm_client", lambda **_kwargs: _OfflineClient())
    monkeypatch.setattr("app.online.exam_builder.request_structured", generator)
    staged_exam = deepcopy(state["generated_exams"][0])
    ok, warnings = val._regenerate_plan_slot(
        state, staged_exam, 1, "model1_word_problem_1", plan_by_slot, source_by_chunk,
    )
    assert ok is True
    assert warnings == []
    prompt = prompts_seen[-1]
    assert "task_id=replacement-task" in prompt
    assert "source_chunk_id=replacement-child" in prompt
    assert "evidence_mismatch" in prompt
    assert "replacement-child" in prompt
    assert "assigned-1" not in prompt
    assert "UNASSIGNED_SECRET_EVIDENCE" not in prompt

    # A new assignment produces new role dependency fingerprints and therefore
    # cannot reuse PASS rows cached for the old task/evidence.
    clear_job_validation_cache(state["job_id"])
    calls: list[str] = []

    def provider(_client, validation_prompt, **_kwargs):
        calls.append(_payload(validation_prompt)["validator"])
        return _validator_response(
            validation_prompt, revalidation_passes=True, initial_failure=False,
        )

    _install_offline_validation(monkeypatch, provider)
    from app.online.multi_validator import validate_exam_multi

    result = validate_exam_multi(
        staged_exam,
        plan_by_slot=plan_by_slot,
        source_by_chunk=source_by_chunk,
        domain="STEM",
        question_ids=["model1_word_problem_1"],
        cache_scope=state["job_id"],
    )
    assert result["all_pass"] is True
    assert result["cache_stats"]["hits"] == 0
    assert set(calls) == {
        "grounding", "task_alignment", "answer_correctness", "difficulty", "quantitative",
    }
