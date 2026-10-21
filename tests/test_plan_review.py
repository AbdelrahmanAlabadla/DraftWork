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


def _difficulty_verdicts(*slot_ids: str) -> list[dict]:
    return [{
        "slot_id": slot_id,
        "claimed_difficulty": "medium",
        "achieved_difficulty": "medium",
        "reasoning_profile_supported": True,
        "evidence_support_sufficient": True,
        "reason": "The assigned evidence supports the stated cognitive work.",
    } for slot_id in slot_ids]


def _response(
    task_id: str, *, slot_id: str = "m1_mcq_2", selection_type: str = "unused_task",
    reviewed_slot_ids: tuple[str, ...] = ("m1_mcq_1", "m1_mcq_2"),
) -> dict:
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
        "difficulty_verdicts": _difficulty_verdicts(*reviewed_slot_ids),
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


def test_discovery_ignores_model_candidate_and_preserves_canonical_identity(monkeypatch):
    repeated, interpret, compare = _catalog_fixture()
    state = _state([repeated, interpret, compare], [repeated, repeated])
    before_slots = [
        item["slot_id"] for item in state["plans"][0]["items"]["mcq"]
    ]
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        plan_review, "request_structured", lambda *_args, **_kwargs: _response(compare["task_id"])
    )

    result = plan_review.review_plan(state)
    items = result["plans"][0]["items"]["mcq"]

    assert [item["slot_id"] for item in items] == before_slots
    # Candidate selection belongs to deterministic repair, not discovery.
    assert items[1]["task_id"] == interpret["task_id"]
    assert items[1]["task_family_id"] == interpret["task_family_id"]
    assert items[1]["coverage_id"] == interpret["coverage_id"]
    assert items[1]["plan_review_source"] == "deterministic_repair_after_discovery"
    assert items[1]["plan_review_status"] == "reassigned_unused"
    assert items[1]["suggested_assessment_angle"]
    assert items[1]["forced_duplicate_slot_ids"] == ["m1_mcq_1"]
    assert result["plan_errors"] == []

    generation_items = exam_builder._attach_plan_review_context({"mcq": deepcopy(items)})
    reviewed = generation_items["mcq"][1]
    assert reviewed["previous_task_plan_items"][0]["slot_id"] == "m1_mcq_1"
    rendered = prompts._format_planned_item(1, reviewed)
    assert "suggested_assessment_angle=" in rendered
    assert "previous_task_plan_items=" in rendered

    exam = {"questions": {"mcq": [
        {"slot_id": "m1_mcq_1", "question_id": "model1_mcq_1"},
        {"slot_id": "m1_mcq_2", "question_id": "model1_mcq_2"},
    ]}}
    forced = duplicate_pipeline._forced_catalog_reuse_pairs(exam, result["plans"], 1)
    assert frozenset(("model1_mcq_1", "model1_mcq_2")) in forced


def test_valid_review_with_invalid_candidate_uses_fallback_without_full_rereview(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, repeated])
    prompts: list[str] = []
    responses = iter([_response("task_missing")])

    def fake_request(_client, prompt, **_kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(plan_review, "request_structured", fake_request)

    result = plan_review.review_plan(state)

    assert len(prompts) == 1
    assert result["plan_review_attempts"] == 1
    assert result["plans"][0]["items"]["mcq"][1]["task_id"] == interpret["task_id"]
    assert result["plan_review_pair_verdicts"]["1"]["plan_pair_1"]["semantic_overlap"] is True
    assert result["plan_review_telemetry"][0]["status"] == "findings_accepted"
    assert result["plan_review_telemetry"][0]["ignored_suggestion_count"] == 1


def test_incomplete_initial_review_gets_one_bounded_full_retry(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, repeated])
    prompts = []
    responses = iter([
        {"verdicts": [], "suggestions": []},
        _response(interpret["task_id"]),
    ])
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        plan_review, "request_structured",
        lambda _client, prompt, **_kwargs: prompts.append(prompt) or next(responses),
    )

    result = plan_review.review_plan(state)

    assert len(prompts) == 2
    assert result["plan_review_attempts"] == 2
    assert result["plan_review_telemetry"][0]["status"] == "findings_rejected"


def test_two_invalid_suggestions_use_ranked_deterministic_fallback(monkeypatch):
    repeated, interpret, compare = _catalog_fixture()
    state = _state([repeated, interpret, compare], [repeated, repeated])
    responses = iter([
        _response(interpret["task_id"], selection_type="intentional_reuse"),
        _response("task_missing"),
    ])
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        plan_review, "request_structured", lambda *_args, **_kwargs: next(responses)
    )

    result = plan_review.review_plan(state)
    changed = result["plans"][0]["items"]["mcq"][1]

    assert changed["task_id"] == interpret["task_id"]
    assert changed["plan_review_source"] == "deterministic_repair_after_discovery"
    assert changed["catalog_replacement_stage"] == "plan_review_unused_task"
    assert any(
        event["status"] == "targeted_repair_applied_after_discovery"
        for event in result["plan_review_history"]
    )


def test_fallback_does_not_reuse_the_directly_conflicting_task(monkeypatch):
    repeated, _interpret, _compare = _catalog_fixture()
    state = _state([repeated], [repeated, repeated])
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        plan_review, "request_structured", lambda *_args, **_kwargs: _response("task_missing")
    )

    result = plan_review.review_plan(deepcopy(state))
    repaired = result["plans"][0]["items"]["mcq"][1]
    assert repaired["task_id"] == repeated["task_id"]
    assert repaired["slot_id"] == "m1_mcq_2"
    assert repaired["plan_duplicate_repair_attempt"] == 1
    assert repaired["plan_duplicate_repair_mode"] == "targeted_semantic_repair"
    assert result["plan_review_retry_required"] is True


def test_intentional_reuse_uses_nonconflicting_least_used_task_and_reaches_generator(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, repeated, interpret])
    response = _response(
        interpret["task_id"], slot_id="m1_mcq_2", selection_type="intentional_reuse",
        reviewed_slot_ids=("m1_mcq_1", "m1_mcq_2", "m1_mcq_3"),
    )
    response["suggestions"][0]["required_instance_distinctions"][0][
        "reference_slot_ids"
    ] = ["m1_mcq_3"]
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
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


def test_plan_review_keeps_findings_when_legacy_candidate_conflicts(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, interpret])
    bad = _response(
        repeated["task_id"], slot_id="m1_mcq_2", selection_type="intentional_reuse"
    )
    prompts_seen = []
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        plan_review, "request_structured",
        lambda _client, prompt, **_kwargs: prompts_seen.append(prompt) or bad,
    )

    result = plan_review.review_plan(state)
    assert len(prompts_seen) == 1
    assert result["plan_review_pair_verdicts"]["1"]["plan_pair_1"]["semantic_overlap"] is True
    assert result["plan_review_telemetry"][0]["ignored_suggestion_count"] == 1
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
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        plan_review, "request_structured",
        lambda *_args, **_kwargs: calls.append(True) or {
            "verdicts": [], "suggestions": [],
            "difficulty_verdicts": _difficulty_verdicts("m1_mcq_1", "m1_mcq_2"),
        },
    )
    result = plan_review.review_plan(state)

    assert calls == [True]
    assert result["plan_review_attempts"] == 1
    assert all(
        item["plan_review_status"] == "clear"
        for item in result["plans"][0]["items"]["mcq"]
    )


def test_plan_review_passes_stem_domain_and_45000_output_limit(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, interpret])
    state["assessment_profile"] = {"domain": "STEM"}
    received: dict = {}

    def fake_client(*, domain=None):
        received["domain"] = domain
        return object()

    def fake_request(_client, _prompt, **kwargs):
        received["max_tokens"] = kwargs["max_tokens"]
        return {
            "verdicts": [], "suggestions": [],
            "difficulty_verdicts": _difficulty_verdicts("m1_mcq_1", "m1_mcq_2"),
        }

    monkeypatch.setattr(plan_review, "create_plan_review_client", fake_client)
    monkeypatch.setattr(plan_review, "request_structured", fake_request)

    result = plan_review.review_plan(state)

    assert result["plan_errors"] == []
    assert received == {"domain": "STEM", "max_tokens": 45000}


def test_plan_reviewer_routes_false_difficulty_to_adjustment(monkeypatch):
    repeated, interpret, _compare = _catalog_fixture()
    state = _state([repeated, interpret], [repeated, interpret])
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(plan_review, "request_structured", lambda *_args, **_kwargs: {
        "verdicts": [], "suggestions": [],
        "difficulty_verdicts": [
            {
                "slot_id": "m1_mcq_1", "claimed_difficulty": "medium",
                "achieved_difficulty": "easy", "reasoning_profile_supported": True,
                "evidence_support_sufficient": True,
                "reason": "The task requires only direct recall from the assigned source.",
            },
            {
                "slot_id": "m1_mcq_2", "claimed_difficulty": "medium",
                "achieved_difficulty": "medium", "reasoning_profile_supported": True,
                "evidence_support_sufficient": True,
                "reason": "The task requires source-supported interpretation.",
            },
        ],
    })
    result = plan_review.review_plan(state)
    first = result["plans"][0]["items"]["mcq"][0]
    assert first["actual_difficulty"] == "easy"
    assert first["difficulty_adaptation"]["status"] == "pending"
    assert any(event["status"] == "difficulty_mismatch" for event in result["plan_review_history"])


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
        "difficulty_verdicts": [
            {
                "slot_id": "m1_fill_in_the_blank_1", "claimed_difficulty": "medium",
                "achieved_difficulty": "medium", "reasoning_profile_supported": True,
                "evidence_support_sufficient": True, "reason": "Supported recall task.",
            },
            {
                "slot_id": "m1_fill_in_the_blank_2", "claimed_difficulty": "medium",
                "achieved_difficulty": "medium", "reasoning_profile_supported": True,
                "evidence_support_sufficient": True, "reason": "Supported recall task.",
            },
        ],
    }
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(plan_review, "request_structured", lambda *_args, **_kwargs: response)

    result = plan_review.review_plan(state)
    repaired = result["plans"][0]["items"]["fill_in_the_blank"][1]
    assert result["plan_review_retry_required"] is True
    assert result["plan_duplicate_repair_attempts"] == {
        "m1_fill_in_the_blank_2": 1
    }
    assert repaired["slot_id"] == "m1_fill_in_the_blank_2"
    assert repaired["plan_duplicate_repair_mode"] == "targeted_semantic_repair"
    assert repaired["numeric_diversification_allowed"] is False
    assert repaired["forced_duplicate_slot_ids"] == ["m1_fill_in_the_blank_1"]


def test_duplicate_plan_error_repairs_only_conflicting_word_problem_and_rechecks(monkeypatch):
    repeated = _task(
        "travel", concept="speed", operation="calculate_distance",
        target="total distance", coverage="distance from speed and time",
        qtype="word_problem",
    )
    alternative = _task(
        "comparison", concept="speed", operation="compare_average_speed",
        target="faster journey", coverage="compare two supported journeys",
        qtype="word_problem",
    )
    interval = _task(
        "interval", concept="speed", operation="calculate_time",
        target="travel time", coverage="time from supported speed and distance",
        qtype="word_problem",
    )
    units = _task(
        "units", concept="speed", operation="convert_speed",
        target="speed in alternate units", coverage="supported speed unit conversion",
        qtype="word_problem",
    )
    plans = [{"model_number": 1, "items": {"word_problem": [
        materialize_plan_item(repeated, "m1_word_problem_1", "medium"),
        materialize_plan_item(interval, "m1_word_problem_2", "medium"),
        materialize_plan_item(units, "m1_word_problem_3", "medium"),
        materialize_plan_item(repeated, "m1_word_problem_4", "medium"),
    ]}}]
    duplicate_errors = [
        "Slot m1_word_problem_4 duplicates the planned reasoning task in "
        "m1_word_problem_1; change the canonical skill, operation, grounded relationship, target, or givens.",
        "Slot m1_word_problem_4 repeats the same scenario/setup and conditions as "
        "m1_word_problem_1; use a meaningfully different scenario or task.",
    ]
    state = {
        "plans": plans, "assessment_catalog": {"tasks": [repeated, alternative, interval, units]},
        "tasks": [("word_problem", 4)], "num_models": 1, "difficulty": "medium",
        "retrieved_chunks": [{"child_id": "c1", "content": "speed distance time"}],
        "assessment_profile": {"question_types": {"word_problem": {
            "task_operations": [
                "calculate_distance", "compare_average_speed", "calculate_time", "convert_speed",
            ],
        }}},
        "planning_issues": [], "plan_errors": duplicate_errors,
    }
    prompts_seen = []
    first_response = _response(alternative["task_id"], slot_id="m1_word_problem_4",
        reviewed_slot_ids=("m1_word_problem_1", "m1_word_problem_2",
                           "m1_word_problem_3", "m1_word_problem_4"))
    first_response["verdicts"][0]["pair_id"] = "plan_pair_3"
    second_response = {
        "verdicts": [], "suggestions": [],
        "difficulty_verdicts": _difficulty_verdicts(
            "m1_word_problem_1", "m1_word_problem_2",
            "m1_word_problem_3", "m1_word_problem_4"
        ),
    }
    responses = iter([first_response, second_response])
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        plan_review, "request_structured",
        lambda _client, prompt, **_kwargs: prompts_seen.append(prompt) or next(responses),
    )

    first = plan_review.review_plan(state)
    original, _second, _third, repaired = first["plans"][0]["items"]["word_problem"]
    assert original["task_id"] == repeated["task_id"]
    assert repaired["slot_id"] == "m1_word_problem_4"
    assert repaired["task_id"] == alternative["task_id"]
    assert repaired["requested_difficulty"] == "medium"
    assert repaired["source_chunk_id"] == "c1"
    assert repaired["plan_duplicate_conflict_with"] == "m1_word_problem_1"
    assert repaired["plan_duplicate_repair_attempt"] == 1
    assert first["plan_review_retry_required"] is True
    assert all(value in prompts_seen[0] for value in (
        "m1_word_problem_1", "m1_word_problem_4", duplicate_errors[0],
        "calculate_distance",
    ))
    assert "compare_average_speed" not in prompts_seen[0]

    state.update(first)
    second = plan_review.review_plan(state)
    assert second["plan_review_retry_required"] is False
    assert second["plans"][0]["items"]["word_problem"][3][
        "plan_duplicate_unresolved"
    ] is False


def test_controlled_plan_diversification_allows_numeric_structure_only_for_quantitative_slots():
    quantitative = _task(
        "motion", concept="motion", operation="calculate_distance",
        target="distance", coverage="distance from speed and time", qtype="word_problem",
    )
    plan = {"model_number": 1, "items": {"word_problem": [
        materialize_plan_item(quantitative, "m1_word_problem_1", "medium"),
        materialize_plan_item(quantitative, "m1_word_problem_2", "medium"),
    ]}}
    original = deepcopy(plan["items"]["word_problem"][1])

    repaired = plan_review._controlled_plan_diversification(
        plan, {"m1_word_problem_1", "m1_word_problem_2"},
        "Same quantitative setup and reasoning path.",
        {"m1_word_problem_1": "word_problem", "m1_word_problem_2": "word_problem"},
    )

    assert repaired["numeric_diversification_allowed"] is True
    assert repaired["quantitative_structure_requirement"]
    for immutable in (
        "slot_id", "task_id", "task_family_id", "question_type", "difficulty",
        "source_chunk_id", "support_chunk_ids",
    ):
        assert repaired.get(immutable) == original.get(immutable)
    assert plan["items"]["word_problem"][0].get("plan_duplicate_repair_attempt") is None


def test_plan_duplicate_recovery_is_bounded_and_unresolved_conflict_can_generate(monkeypatch):
    repeated = _task(
        "definition", concept="energy", operation="define",
        target="energy", coverage="definition of energy", qtype="definition",
    )
    state = {
        "plans": [{"model_number": 1, "items": {"definition": [
            materialize_plan_item(repeated, "m1_definition_1", "medium"),
            materialize_plan_item(repeated, "m1_definition_2", "medium"),
        ]}}],
        "assessment_catalog": {"tasks": [repeated]},
        "tasks": [("definition", 2)], "num_models": 1, "difficulty": "medium",
        "retrieved_chunks": [{"child_id": "c1", "content": "Energy is the ability to do work."}],
        "assessment_profile": {"question_types": {"definition": {
            "task_operations": ["define"],
        }}}, "planning_issues": [],
    }
    overlap = _response(
        repeated["task_id"], slot_id="m1_definition_2",
        selection_type="intentional_reuse",
        reviewed_slot_ids=("m1_definition_1", "m1_definition_2"),
    )
    overlap["suggestions"][0]["required_instance_distinctions"][0][
        "reference_slot_ids"
    ] = ["m1_definition_1"]
    calls = []
    monkeypatch.setattr(plan_review, "create_plan_review_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        plan_review, "request_structured",
        lambda *_args, **_kwargs: calls.append(True) or deepcopy(overlap),
    )

    for expected_attempt in (1, 2, 3):
        result = plan_review.review_plan(state)
        state.update(result)
        repaired = state["plans"][0]["items"]["definition"][1]
        assert repaired["slot_id"] == "m1_definition_2"
        assert repaired["plan_duplicate_repair_attempt"] == expected_attempt
        assert repaired["numeric_diversification_allowed"] is False
        assert state["plan_review_retry_required"] is True

    result = plan_review.review_plan(state)
    state.update(result)
    repaired = state["plans"][0]["items"]["definition"][1]
    assert state["plan_review_retry_required"] is False
    assert repaired["plan_duplicate_unresolved"] is True
    assert repaired["plan_duplicate_repair_mode"] == "controlled_diversification"
    assert state["unresolved_plan_duplicate_conflicts"][0]["status"] == (
        "plan_duplicate_unresolved_kept"
    )
    assert len(calls) == 4  # one valid review per bounded repair/final verification cycle

    from app.online.graph import _route_after_plan_review
    assert _route_after_plan_review(state) == "generate_exams"
