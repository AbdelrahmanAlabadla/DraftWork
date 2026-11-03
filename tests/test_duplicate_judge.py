from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.online import duplicate_judge


def _question(qid: str, stem: str, answer: object, *, slot: str | None = None) -> dict:
    return {"question_id": qid, "slot_id": slot or f"s_{qid}", "statement": stem, "answer": answer}


def _facts(pair_id: str, *, duplicate: bool = False) -> dict:
    return {
        "pair_id": pair_id,
        "same_core_knowledge": "yes" if duplicate else "no",
        "same_student_operation_or_method": "yes" if duplicate else "no",
        "same_target_or_result": "yes" if duplicate else "no",
        "genuinely_new_reasoning": "no" if duplicate else "yes",
        "one_answer_substantially_gives_away_other": "no",
    }


def test_false_answer_is_preserved_in_semantic_payload():
    exam = {"questions": {"true_false": [_question("q1", "The claim is false.", False)]}}
    assert duplicate_judge.actual_question_registry(exam)[0]["answer"] == "False"


def test_metadata_screening_uses_stable_pair_id_without_embeddings():
    exam = {"questions": {"true_false": [
        _question("q2", "Second form", True, slot="s2"),
        _question("q1", "First form", False, slot="s1"),
    ]}}
    plans = {
        "s1": {"task_id": "task-a", "task_operation": "solve_system", "target": "find x"},
        "s2": {"task_id": "task-b", "task_operation": "solve_system", "target": "find x"},
    }
    candidates, stats = duplicate_judge.shortlist_duplicate_pairs(exam, plan_by_slot=plans)
    assert [row["pair_id"] for row in candidates] == ["q1::q2"]
    assert "matching_operation_and_target" in candidates[0]["signals"]
    assert stats["possible_pair_count"] == 1
    assert stats["embedding_calls"] == 0


def test_same_task_and_variation_are_candidates_not_automatic_failures():
    exam = {"questions": {"true_false": [
        _question("q1", "Apply the rule.", True, slot="s1"),
        _question("q2", "Explain the rule.", False, slot="s2"),
    ]}}
    plans = {
        "s1": {"task_id": "task-a", "selected_variation_group_id": "vg-a"},
        "s2": {"task_id": "task-a", "selected_variation_group_id": "vg-a"},
    }
    candidates, _ = duplicate_judge.shortlist_duplicate_pairs(exam, plan_by_slot=plans)
    assert candidates[0]["signals"] == ["shared_task_id", "shared_variation_group"]


def test_no_supported_signal_means_no_validator_call(monkeypatch):
    exam = {"questions": {"true_false": [
        _question("q1", "Alpha", True), _question("q2", "Beta", False),
    ]}}
    monkeypatch.setattr(duplicate_judge, "request_structured", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    result = duplicate_judge.review_duplicate_candidates(exam)
    assert result["candidate_pairs"] == []
    assert result["fact_cache_misses"] == 0


def test_python_classifies_and_selects_one_repair_target(monkeypatch):
    exam = {"questions": {"true_false": [
        _question("q1", "Equivalent form one", True, slot="s1"),
        _question("q2", "Equivalent form two", False, slot="s2"),
    ]}}
    plans = {"s1": {"task_id": "task-a"}, "s2": {"task_id": "task-a"}}
    monkeypatch.setattr(duplicate_judge, "create_llm_client", lambda: SimpleNamespace(last_usage={}))
    monkeypatch.setattr(duplicate_judge, "request_structured", lambda *_a, **k: {
        "verdicts": [_facts(pair_id, duplicate=True) for pair_id in k["expected_ids"]]
    })
    result = duplicate_judge.review_duplicate_candidates(exam, plan_by_slot=plans)
    assert result["pair_verdicts"][0]["classification"] == "duplicate"
    group = result["duplicate_groups"][0]
    assert len(group["replace_question_ids"]) == len(group["keep_question_ids"]) == 1


def test_incremental_screening_only_evaluates_changed_pairs():
    exam = {"questions": {"true_false": [
        _question("q1", "One", True, slot="s1"), _question("q2", "Two", False, slot="s2"),
        _question("q3", "Three", True, slot="s3"),
    ]}}
    plans = {slot: {"task_id": "shared"} for slot in ("s1", "s2", "s3")}
    candidates, stats = duplicate_judge.shortlist_duplicate_pairs(exam, plan_by_slot=plans, changed_question_ids={"q3"})
    assert {row["pair_id"] for row in candidates} == {"q1::q3", "q2::q3"}
    assert stats["possible_pair_count"] == 3
    assert stats["evaluated_pair_count"] == 2


def test_distinct_reasoning_and_authorized_practice_classifications():
    distinct = {"same_core_knowledge": "yes", "same_student_operation_or_method": "no",
                "same_target_or_result": "yes", "genuinely_new_reasoning": "yes",
                "one_answer_substantially_gives_away_other": "no"}
    practice = {**distinct, "same_student_operation_or_method": "yes", "genuinely_new_reasoning": "no"}
    assert duplicate_judge.deterministic_pair_classification(distinct) == "acceptable_reuse"
    assert duplicate_judge.deterministic_pair_classification(practice) == "duplicate"
    assert duplicate_judge.deterministic_pair_classification(practice, authorized_practice_variation=True) == "acceptable_practice_variation"


@pytest.mark.parametrize("rows", [
    [],
    [{"pair_id": "unknown", **_facts("unknown")}],
    [{**_facts("q1::q2"), "same_core_knowledge": "maybe"}],
    [{"pair_id": "q1::q2", "same_core_knowledge": "yes"}],
    [_facts("q1::q2"), _facts("q1::q2")],
])
def test_malformed_or_incomplete_judge_rows_never_mean_non_duplicate(monkeypatch, rows):
    exam = {"questions": {"true_false": [
        _question("q1", "First", True, slot="s1"),
        _question("q2", "Second", False, slot="s2"),
    ]}}
    plans = {"s1": {"task_id": "shared"}, "s2": {"task_id": "shared"}}
    monkeypatch.setattr(duplicate_judge, "create_llm_client", lambda: SimpleNamespace(last_usage={}))
    monkeypatch.setattr(duplicate_judge, "request_structured", lambda *_a, **_k: {"verdicts": rows})
    with pytest.raises(duplicate_judge.DuplicateValidatorResponseError):
        duplicate_judge.review_duplicate_candidates(exam, plan_by_slot=plans)


def test_large_conflict_group_uses_bounded_deterministic_cover():
    ids = [f"q{index:02d}" for index in range(13)]
    candidates = []
    verdicts = []
    for left, right in zip(ids, ids[1:]):
        pair_id = f"{left}::{right}"
        candidates.append({"pair_id": pair_id, "left": {"question_id": left, "question_type": "mcq", "answer": "x"},
                           "right": {"question_id": right, "question_type": "mcq", "answer": "x"}})
        verdicts.append({"pair_id": pair_id, **_facts(pair_id, duplicate=True), "classification": "duplicate"})
    first = duplicate_judge.build_duplicate_groups(candidates, verdicts)[0]
    second = duplicate_judge.build_duplicate_groups(candidates, verdicts)[0]
    selected = set(first["replace_question_ids"])
    assert first["replace_question_ids"] == second["replace_question_ids"]
    assert all(edge["left"] in selected or edge["right"] in selected for edge in first["edges"])
