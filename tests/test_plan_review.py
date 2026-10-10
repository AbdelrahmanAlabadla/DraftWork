from __future__ import annotations

from copy import deepcopy

import pytest

from app.online import duplicate_pipeline, exam_builder, plan_review, prompts
from app.online.assessment_catalog import (
    canonical_fitb_entry,
    canonical_task_ids,
    materialize_fitb_plan_item,
    materialize_plan_item,
)


def _task(
    name: str,
    *,
    concept: str,
    operation: str,
    target: str,
    coverage: str,
    qtype: str = "mcq",
) -> dict:
    task = {
        "knowledge_anchor": coverage,
        "topic": "physics",
        "concept": concept,
        "relationship": None,
        "task_operation": operation,
        "student_work_product": f"{operation} {target}",
        "target": target,
        "reasoning_mode": "conceptual_reasoning",
        "supported_difficulties": ["medium"],
        "allowed_question_types": [qtype],
        "evidence_chunk_ids": ["c1"],
        "required_givens": [],
        "scenario_family": name,
        "text_answerable": True,
    }
    task.update(canonical_task_ids(task) or {})
    return task


def _state(catalog_tasks: list[dict], planned_tasks: list[dict]) -> dict:
    return {
        "plans": [{
            "model_number": 1,
            "items": {
                "mcq": [
                    materialize_plan_item(task, f"m1_mcq_{index}", "medium")
                    for index, task in enumerate(planned_tasks, 1)
                ]
            },
        }],
        "assessment_catalog": {"tasks": catalog_tasks},
        "tasks": [("mcq", len(planned_tasks))],
        "num_models": 1,
        "difficulty": "medium",
        "retrieved_chunks": [{"child_id": "c1", "content": "grounded source"}],
        "assessment_profile": {},
        "planning_issues": [],
    }


def _response(task_id: str, *, slot_id: str = "m1_mcq_2", selection_type: str = "unused_task") -> dict:
    reused = selection_type == "intentional_reuse"
    return {
        "verdicts": [{
            "pair_id": "plan_pair_1",
            "semantic_overlap": True,
            "reason": "Both slots assess the same operation and target.",
        }],
        "suggestions": [{
            "slot_id": slot_id,
            "replacement_task_id": task_id,
            "selection_type": selection_type,
            "reason": "This task assesses a different operation and target.",
            "suggested_assessment_angle": "Interpret the result instead of recalling it.",
            "reuse_justification": (
                "No unused compatible task remains." if reused else None
            ),
            "required_instance_distinctions": ([{
                "kind": "different_assessment_angle",
                "requirement": "Require interpretation rather than recall.",
                "reference_slot_ids": ["m1_mcq_1"],
            }] if reused else []),
        }],
    }


def _catalog_fixture() -> tuple[dict, dict, dict]:
    repeated = _task(
        "repeated", concept="energy", operation="identify",
        target="kinetic energy", coverage="kinetic energy definition",
    )
    interpret = _task(
        "interpret", concept="work", operation="interpret",
        target="sign of work", coverage="work sign from force direction",
    )
    compare = _task(
        "compare", concept="power", operation="compare",
        target="power for two processes", coverage="power comparison",
    )
    return repeated, interpret, compare


def test_agent_applies_valid_unused_task_and_preserves_canonical_identity(monkeypatch):
    repeated, interpret, compare = _catalog_fixture()
    state = _state([repeated, interpret, compare], [repeated, repeated])
    before_slots = [
        item["slot_id"] for item in state["plans"][0]["items"]["mcq"]
    ]
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda: object())
    monkeypatch.setattr(
        plan_review, "request_structured", lambda *_args, **_kwargs: _response(interpret["task_id"])
    )

    result = plan_review.review_plan(state)
    items = result["plans"][0]["items"]["mcq"]

    assert [item["slot_id"] for item in items] == before_slots
    assert items[1]["task_id"] == interpret["task_id"]
    assert items[1]["task_family_id"] == interpret["task_family_id"]
    assert items[1]["coverage_id"] == interpret["coverage_id"]
    assert items[1]["plan_review_source"] == "agent"
    assert items[1]["plan_review_status"] == "reassigned_unused"
    assert items[1]["suggested_assessment_angle"]
    assert items[1]["forced_duplicate_slot_ids"] == ["m1_mcq_1"]
    assert result["plan_errors"] == []

    generation_items = exam_builder._attach_plan_review_context({"mcq": deepcopy(items)})
    reviewed = generation_items["mcq"][1]
    assert reviewed["previous_task_plan_items"][0]["slot_id"] == "m1_mcq_1"
    rendered = prompts._format_planned_item(1, reviewed)
    assert "suggested_assessment_angle=Interpret the result instead of recalling it." in rendered
    assert "previous_task_plan_items=" in rendered

    exam = {"questions": {"mcq": [
        {"slot_id": "m1_mcq_1", "question_id": "model1_mcq_1"},
        {"slot_id": "m1_mcq_2", "question_id": "model1_mcq_2"},
    ]}}
    forced = duplicate_pipeline._forced_catalog_reuse_pairs(exam, result["plans"], 1)
    assert frozenset(("model1_mcq_1", "model1_mcq_2")) in forced


def test_invalid_suggestion_retries_once_with_exact_rejection_reason(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, repeated])
    prompts: list[str] = []
    responses = iter([
        _response("task_missing"),
        _response(interpret["task_id"]),
    ])

    def fake_request(_client, prompt, **_kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda: object())
    monkeypatch.setattr(plan_review, "request_structured", fake_request)

    result = plan_review.review_plan(state)

    assert len(prompts) == 2
    assert "Candidate task_missing is not allowed for m1_mcq_2." in prompts[1]
    assert result["plan_review_attempts"] == 2
    assert result["plans"][0]["items"]["mcq"][1]["task_id"] == interpret["task_id"]


def test_two_invalid_suggestions_use_ranked_deterministic_fallback(monkeypatch):
    repeated, interpret, compare = _catalog_fixture()
    state = _state([repeated, interpret, compare], [repeated, repeated])
    responses = iter([
        _response(interpret["task_id"], selection_type="intentional_reuse"),
        _response("task_missing"),
    ])
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda: object())
    monkeypatch.setattr(
        plan_review, "request_structured", lambda *_args, **_kwargs: next(responses)
    )

    result = plan_review.review_plan(state)
    changed = result["plans"][0]["items"]["mcq"][1]

    assert changed["task_id"] == compare["task_id"]
    assert changed["plan_review_source"] == "deterministic_fallback"
    assert changed["catalog_replacement_stage"] == "plan_review_unused_task"
    assert any(event["status"] == "fallback_applied" for event in result["plan_review_history"])


def test_fallback_does_not_reuse_the_directly_conflicting_task(monkeypatch):
    repeated, _interpret, _compare = _catalog_fixture()
    state = _state([repeated], [repeated, repeated])
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda: object())
    monkeypatch.setattr(
        plan_review, "request_structured", lambda *_args, **_kwargs: _response("task_missing")
    )

    with pytest.raises(RuntimeError, match="Plan Review produced an invalid plan"):
        plan_review.review_plan(deepcopy(state))


def test_intentional_reuse_uses_nonconflicting_least_used_task_and_reaches_generator(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, repeated, interpret])
    response = _response(
        interpret["task_id"], slot_id="m1_mcq_2", selection_type="intentional_reuse"
    )
    response["suggestions"][0]["required_instance_distinctions"][0][
        "reference_slot_ids"
    ] = ["m1_mcq_3"]
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda: object())
    monkeypatch.setattr(plan_review, "request_structured", lambda *_args, **_kwargs: response)

    result = plan_review.review_plan(state)
    changed = result["plans"][0]["items"]["mcq"][1]
    assert changed["task_id"] == interpret["task_id"]
    assert changed["task_reused"] is True
    assert changed["selection_rank"]["task_usage_count"] == 1
    assert changed["forced_duplicate_slot_ids"] == ["m1_mcq_1", "m1_mcq_3"]

    generation = exam_builder._attach_plan_review_context({
        "mcq": deepcopy(result["plans"][0]["items"]["mcq"])
    })
    reused = generation["mcq"][1]
    assert reused["previous_task_questions"][0]["slot_id"] == "m1_mcq_3"
    assert reused["previous_task_questions"][0]["context_role"] == "previous_planned_task_instance"
    rendered = prompts._format_planned_item(1, reused)
    assert "previous_task_questions=" in rendered


def test_plan_review_rejects_conflicting_slots_exact_task_before_apply(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, interpret])
    bad = _response(
        repeated["task_id"], slot_id="m1_mcq_2", selection_type="intentional_reuse"
    )
    responses = iter([bad, bad])
    prompts_seen = []
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda: object())
    monkeypatch.setattr(
        plan_review, "request_structured",
        lambda _client, prompt, **_kwargs: prompts_seen.append(prompt) or next(responses),
    )

    result = plan_review.review_plan(state)
    assert any(
        "directly conflicting slot" in event.get("reason", "")
        for event in result["plan_review_history"]
        if event["status"] == "agent_suggestion_rejected"
    )
    assert result["plans"][0]["items"]["mcq"][0]["task_id"] == repeated["task_id"]
    assert result["plans"][0]["items"]["mcq"][1]["task_id"] == interpret["task_id"]


def test_deterministic_reuse_fallback_selects_least_used_nonconflicting_task():
    repeated, interpret, compare = _catalog_fixture()
    state = _state(
        [repeated, interpret, compare],
        [repeated, repeated, interpret, compare, compare],
    )
    suggestion = plan_review._fallback_suggestion(
        state["assessment_catalog"], state["plans"][0],
        {"m1_mcq_1", "m1_mcq_2"}, "medium", {"c1"}, {},
    )
    assert suggestion is not None
    assert suggestion["selection_type"] == "intentional_reuse"
    assert suggestion["replacement_task_id"] == interpret["task_id"]


def test_plan_review_refuses_missing_requested_slot_before_model_call(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated])
    state["tasks"] = [("mcq", 2)]
    calls = []
    monkeypatch.setattr(
        plan_review, "request_structured", lambda *_args, **_kwargs: calls.append(True)
    )
    with pytest.raises(RuntimeError, match="refused an invalid Planner result"):
        plan_review.review_plan(state)
    assert calls == []


def test_generator_refuses_structurally_invalid_reviewed_plan():
    repeated, _interpret, _compare = _catalog_fixture()
    state = _state([repeated], [repeated])
    state.update({
        "tasks": [("mcq", 2)], "document_id": "doc", "context": "",
        "document_language": "en", "warnings": [], "eval_stats": {},
    })
    with pytest.raises(RuntimeError, match="Generator refused an invalid reviewed plan"):
        exam_builder.generate_exams_node(state)


def test_semantically_distinct_plan_is_still_reviewed_as_a_whole(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, interpret])

    calls = []
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda: object())
    monkeypatch.setattr(
        plan_review, "request_structured",
        lambda *_args, **_kwargs: calls.append(True) or {"verdicts": [], "suggestions": []},
    )
    result = plan_review.review_plan(state)

    assert calls == [True]
    assert result["plan_review_attempts"] == 1
    assert all(
        item["plan_review_status"] == "clear"
        for item in result["plans"][0]["items"]["mcq"]
    )


def test_fitb_cannot_reuse_the_directly_conflicting_entry(monkeypatch):
    entry = canonical_fitb_entry({
        "answer": "catalyst", "evidence_chunk_id": "c1",
        "clue_basis": "substance that speeds a reaction",
    }, {"c1"})
    plans = [{"model_number": 1, "items": {"fill_in_the_blank": [
        materialize_fitb_plan_item(entry, "m1_fill_in_the_blank_1", "medium"),
        materialize_fitb_plan_item(entry, "m1_fill_in_the_blank_2", "medium"),
    ]}}]
    state = {
        "plans": plans, "assessment_catalog": {"tasks": [], "fitb_bank": [entry]},
        "tasks": [("fill_in_the_blank", 2)], "num_models": 1, "difficulty": "medium",
        "retrieved_chunks": [{"child_id": "c1", "content": "A catalyst speeds a reaction."}],
        "assessment_profile": {"question_types": {"fill_in_the_blank": {
            "task_operations": ["identify_term_from_text_clue"],
        }}},
        "planning_issues": [],
    }
    response = {
        "verdicts": [{
            "pair_id": "plan_pair_1", "semantic_overlap": True,
            "reason": "Both slots would ask for the same term from the same clue.",
        }],
        "suggestions": [{
            "slot_id": "m1_fill_in_the_blank_2", "replacement_task_id": None,
            "replacement_fitb_entry_id": entry["fitb_entry_id"],
            "selection_type": "intentional_reuse",
            "reason": "No unused bank entry remains.",
            "suggested_assessment_angle": "Use a distinct textual property as the clue.",
            "reuse_justification": "No unused compatible FITB entry remains.",
            "required_instance_distinctions": [{
                "kind": "different_assessment_angle",
                "requirement": "Use a different source-supported textual clue.",
                "reference_slot_ids": ["m1_fill_in_the_blank_1"],
            }],
        }],
    }
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda: object())
    monkeypatch.setattr(plan_review, "request_structured", lambda *_args, **_kwargs: response)

    with pytest.raises(RuntimeError, match="Plan Review produced an invalid plan"):
        plan_review.review_plan(state)
