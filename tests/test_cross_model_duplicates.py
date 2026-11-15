from __future__ import annotations

from copy import deepcopy
import threading
import time

import pytest

from app.online import assessment_catalog
from app.online import cross_model_duplicates as cross
from app.online import duplicate_judge, planner
from app.online import exam_builder


def _plan(model: int, *, task: str = "task_shared") -> dict:
    item = {
        "slot_id": f"m{model}_true_false_1",
        "question_type": "true_false",
        "task_id": task,
        "selected_variation_id": "variation_shared",
        "selected_variation_group_id": "variation_group_shared",
        "source_chunk_id": "chunk-1",
        "support_chunk_ids": [],
        "topic": "Algebra",
        "concept": "linear equations",
        "canonical_skill": "solve a linear equation",
        "knowledge_focus": "linear equation solution",
        "student_work_product": "determine whether the claimed solution is true",
        "relationship": None,
        "target": "solution of a linear equation",
        "task_operation": "verify_linear_equation_solution",
        "given_conditions": ["one linear equation"],
        "scenario_family": None,
        "required_givens": ["equation", "claimed solution"],
        "text_answerable": True,
        "reasoning_mode": "calculation",
        "difficulty": "easy",
        "reasoning_depth": "easy",
        "reasoning_requirements": ["substitute and compare"],
        "prerequisites": [],
        "idea_to_test": "verify a claimed linear equation solution",
        "concept_to_test": "linear equations",
        "permitted_variations": [{
            "variation_id": "variation_shared",
            "variation_group_id": "variation_group_shared",
            "question_types": ["true_false"],
            "numeric_values_may_change": True,
        }],
    }
    canonical_task_id = assessment_catalog.canonical_task_ids(item)["task_id"]
    item["task_id"] = canonical_task_id
    if model > 1:
        item.update({
            "task_reused": True,
            "reused_due_to_catalog_exhaustion": True,
            "catalog_replacement_stage": "reused_task_after_catalog_exhaustion",
            "reuse_reason": "no_unused_compatible_task",
            "reused_task_id": canonical_task_id,
            "forced_duplicate_slot_ids": ["m1_true_false_1"],
            "required_instance_distinctions": ["different numerical instance"],
        })
    return {"model_number": model, "items": {"true_false": [item]}}


def _question(model: int, statement: str, answer: bool) -> dict:
    return {
        "question_id": "true_false_1",
        "slot_id": f"m{model}_true_false_1",
        "statement": statement,
        "correct_answer": answer,
        "solution_certificate": {"verification_method": statement},
    }


def _state(statements: list[str]) -> tuple[dict, list[dict], dict[int, dict]]:
    plans = [_plan(index) for index in range(1, len(statements) + 1)]
    generated = [{
        "model_number": index,
        "questions": {"true_false": [_question(index, statement, True)]},
    } for index, statement in enumerate(statements, start=1)]
    reports = {
        index: {"model_number": index, "verdicts": [{
            "question_id": "true_false_1", "action": "PASS",
        }]}
        for index in range(1, len(statements) + 1)
    }
    return {"num_models": len(statements), "plans": plans}, generated, reports


def _mock_review(exact_duplicate: bool):
    def review(exam, **kwargs):
        ids = [
            item["question_id"]
            for values in exam["questions"].values() for item in values
        ]
        pairs = []
        verdicts = []
        for number, pair in enumerate(sorted(kwargs["forced_question_pairs"], key=sorted), 1):
            left, right = sorted(pair)
            pair_id = f"pair_{number}"
            pairs.append({
                "pair_id": pair_id,
                "left": {"question_id": left},
                "right": {"question_id": right},
                "signals": ["forced_catalog_reuse"],
            })
            verdicts.append({
                "pair_id": pair_id,
                "classification": (
                    "duplicate" if exact_duplicate
                    else "acceptable_practice_variation"
                ),
            })
        assert set(ids) == {value for pair in kwargs["forced_question_pairs"] for value in pair}
        return {
            "candidate_pairs": pairs,
            "pair_verdicts": verdicts,
            "fact_cache": {"cached": {}},
            "similarity_distribution": {"forced_only": kwargs["forced_only"]},
        }
    return review


def test_authorized_cross_model_reuse_passes_planner_validation():
    assert planner.validate_plans(
        [_plan(1), _plan(2)], [("true_false", 1)], 2, {"chunk-1"}
    ) == []


def test_unauthorized_cross_model_overlap_remains_rejected():
    second = _plan(2)
    for key in (
        "task_reused", "reused_due_to_catalog_exhaustion",
        "catalog_replacement_stage", "reuse_reason", "reused_task_id",
        "forced_duplicate_slot_ids", "required_instance_distinctions",
    ):
        second["items"]["true_false"][0].pop(key, None)
    errors = planner.validate_plans(
        [_plan(1), second], [("true_false", 1)], 2, {"chunk-1"}
    )
    assert any("repeats the same true_false idea" in error for error in errors)


def test_full_cross_model_scan_persists_findings_before_recovery(monkeypatch):
    state, generated, reports = _state(["x=2 solves x+1=3", "x=2 solves x+1=3"])
    monkeypatch.setattr(cross, "review_duplicate_candidates", _mock_review(True))
    manifest, validated, _warnings, result = cross.integrate_cross_model_duplicates(
        state, generated, reports, [], [1, 2], [],
    )
    assert result["cross_model_duplicate_scan_complete"] is True
    assert len(result["cross_model_duplicate_findings"]) == 1
    assert result["cross_model_duplicate_findings"][0]["reference"]["model_number"] == 1
    assert [target["model_number"] for target in manifest] == [2]
    assert validated == [1]
    assert manifest[0]["conflicting_question_ids"] == ["m1::true_false_1"]


def test_cross_model_duplicate_targets_later_recovery_owner_not_lower_model(monkeypatch):
    state, generated, reports = _state(["same", "same"])
    state["plans"][0]["items"]["true_false"][0]["assignment_revision"] = 4
    state["plans"][1]["items"]["true_false"][0]["assignment_revision"] = 1
    state["slot_lifecycle"] = {
        "m1_true_false_1": {"question_revision": 4},
        "m2_true_false_1": {"question_revision": 1},
    }
    monkeypatch.setattr(cross, "review_duplicate_candidates", _mock_review(True))
    manifest, validated, _warnings, _result = cross.integrate_cross_model_duplicates(
        state, generated, reports, [], [1, 2], [],
    )
    assert [(value["model_number"], value["slot_id"]) for value in manifest] == [
        (1, "m1_true_false_1")
    ]
    assert validated == [2]
    lifecycle = state["slot_lifecycle"]["m1_true_false_1"]
    assert lifecycle["local_validation_status"] == "PASS"
    assert lifecycle["cross_model_duplicate_status"] == "FAIL"
    assert lifecycle["final_acceptance_status"] == "PENDING_RECOVERY"


def test_permitted_distinct_instances_are_accepted(monkeypatch):
    state, generated, reports = _state(["x=2 solves x+1=3", "x=4 solves x+2=6"])
    monkeypatch.setattr(cross, "review_duplicate_candidates", _mock_review(False))
    manifest, validated, _warnings, result = cross.integrate_cross_model_duplicates(
        state, generated, reports, [], [1, 2], [],
    )
    assert manifest == []
    assert validated == [1, 2]
    assert result["cross_model_duplicate_findings"] == []


def test_four_model_conflict_keeps_earliest_and_combines_target(monkeypatch):
    state, generated, reports = _state(["same", "same", "same", "same"])
    monkeypatch.setattr(cross, "review_duplicate_candidates", _mock_review(True))
    manifest, validated, _warnings, result = cross.integrate_cross_model_duplicates(
        state, generated, reports, [], [1, 2, 3, 4], [],
    )
    assert [target["model_number"] for target in manifest] == [2, 3, 4]
    assert validated == [1]
    assert len(result["cross_model_duplicate_findings"]) == 1


def test_completed_unchanged_scan_is_not_repeated(monkeypatch):
    state, generated, reports = _state(["first", "second"])
    calls = []
    monkeypatch.setattr(
        cross, "review_duplicate_candidates",
        lambda *args, **kwargs: calls.append(1) or _mock_review(False)(*args, **kwargs),
    )
    _manifest, _validated, _warnings, result = cross.integrate_cross_model_duplicates(
        state, generated, reports, [], [1, 2], [],
    )
    resumed = {**state, **result}
    cross.integrate_cross_model_duplicates(resumed, generated, reports, [], [1, 2], [])
    assert len(calls) == 1


def test_forced_only_shortlist_does_not_compare_unrelated_pairs():
    exam = {"questions": {"true_false": [
        {**_question(1, "one", True), "question_id": "q1"},
        {**_question(2, "two", True), "question_id": "q2"},
        {**_question(3, "three", True), "question_id": "q3"},
    ]}}
    candidates, stats = duplicate_judge.shortlist_duplicate_pairs(
        exam, forced_question_pairs={frozenset(("q1", "q2"))}, forced_only=True,
    )
    assert len(candidates) == 1
    assert {candidates[0]["left"]["question_id"], candidates[0]["right"]["question_id"]} == {"q1", "q2"}
    assert stats["forced_only"] is True


def test_cross_model_judge_prompt_contains_content_not_planner_metadata():
    candidate = {
        "pair_id": "pair_1",
        "left": {
            "question_id": "m1::q1", "question_type": "equation",
            "stem": "Solve x+1=3", "answer": "x=2", "task_id": "secret-task",
            "reuse_context": {"authorized": True},
        },
        "right": {
            "question_id": "m2::q1", "question_type": "equation",
            "stem": "Solve x+2=6", "answer": "x=4", "task_id": "secret-task",
            "reuse_context": {"authorized": True},
        },
    }
    prompt = duplicate_judge._prompt([candidate], content_only=True)
    assert "Solve x+1=3" in prompt and "x=4" in prompt
    assert "secret-task" not in prompt
    assert '"reuse_context"' not in prompt


@pytest.mark.parametrize("model_count", [2, 4])
def test_model_generation_dispatch_overlaps_and_keeps_state_isolated(
    monkeypatch, model_count,
):
    barrier = threading.Barrier(model_count)
    intervals: dict[int, tuple[float, float]] = {}

    def fake_impl(state):
        model = state["_active_generation_models"][0]
        assert state["generated_exams"] == []
        started = time.perf_counter()
        barrier.wait(timeout=2)
        time.sleep(0.03)
        intervals[model] = (started, time.perf_counter())
        return {
            "generated_exams": [{"model_number": model, "questions": {}}],
            "plans": deepcopy(state["plans"]),
            "warnings": [f"model-{model}"],
            "mcq_task_reassignments": [],
            "fitb_recovery_history": [],
            "eval_stats": deepcopy(state.get("eval_stats")),
        }

    monkeypatch.setattr(exam_builder, "_generate_exams_node_impl", fake_impl)
    checkpoints = []
    result = exam_builder.generate_exams_node({
        "num_models": model_count,
        "plans": [_plan(model) for model in range(1, model_count + 1)],
        "generated_exams": [],
        "warnings": [],
        "eval_stats": None,
        "audit_callback": lambda stage, event, payload: checkpoints.append(
            (stage, event, payload)
        ),
    })
    assert [exam["model_number"] for exam in result["generated_exams"]] == list(
        range(1, model_count + 1)
    )
    assert max(value[0] for value in intervals.values()) < min(
        value[1] for value in intervals.values()
    )
    assert len(checkpoints) == model_count
    assert len(checkpoints[-1][2]["checkpoint_state"]["generated_exams"]) == model_count


def test_reuse_context_can_reference_another_models_plan_without_output():
    model_one = _plan(1)["items"]
    model_two = deepcopy(_plan(2)["items"])
    all_items = {
        item["slot_id"]: item
        for values in [*model_one.values(), *model_two.values()]
        for item in values
    }
    enriched = exam_builder._attach_reuse_context(model_two, all_items)
    prior = enriched["true_false"][0]["previous_task_plan_items"]
    assert prior[0]["slot_id"] == "m1_true_false_1"
    assert enriched["true_false"][0]["previous_task_questions"][0][
        "context_role"
    ] == "previous_planned_task_instance"
