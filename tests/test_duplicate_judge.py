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


def test_different_operations_and_targets_are_not_excessive_overlap():
    # Real false positive: point-inequality substitution vs interpreting 0=0
    # after elimination. Shared broad knowledge and operation labels are not
    # enough when the target differs and no answer is given away.
    distinct_target = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "yes",
        "same_target_or_result": "no",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "no",
    }
    assert duplicate_judge.deterministic_pair_classification(
        distinct_target
    ) == "acceptable_reuse"
    assert duplicate_judge.deterministic_pair_classification(
        distinct_target, authorized_practice_variation=True
    ) == "acceptable_practice_variation"


def test_same_target_and_operation_remain_a_duplicate():
    same = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "yes",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "no",
    }
    assert duplicate_judge.deterministic_pair_classification(same) == "duplicate"


def test_answer_giveaway_is_excessive_overlap_even_with_different_labels():
    giveaway = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "yes",
    }
    assert duplicate_judge.deterministic_pair_classification(giveaway) == "excessive_overlap"


def test_numeric_practice_variation_requires_authorization():
    numeric = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "yes",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "no",
    }
    assert duplicate_judge.deterministic_pair_classification(numeric) == "duplicate"
    assert duplicate_judge.deterministic_pair_classification(
        numeric, authorized_practice_variation=True
    ) == "acceptable_practice_variation"


def test_distinct_reasoning_and_authorized_practice_classifications():
    distinct = {"same_core_knowledge": "yes", "same_student_operation_or_method": "no",
                "same_target_or_result": "yes", "genuinely_new_reasoning": "yes",
                "one_answer_substantially_gives_away_other": "no"}
    practice = {**distinct, "same_student_operation_or_method": "yes", "genuinely_new_reasoning": "no"}
    assert duplicate_judge.deterministic_pair_classification(distinct) == "acceptable_reuse"
    assert duplicate_judge.deterministic_pair_classification(practice) == "duplicate"
    assert duplicate_judge.deterministic_pair_classification(practice, authorized_practice_variation=True) == "acceptable_practice_variation"
    # Authorized reuse never exempts exact content.
    assert duplicate_judge.deterministic_pair_classification(
        practice, authorized_practice_variation=True, exact_content=True,
    ) == "duplicate"


def test_mcq_options_are_not_numerical_givens():
    exam = {"questions": {
        "mcq": [{
            "question_id": "q_mcq", "slot_id": "s_mcq",
            "question": "Evaluate f(-3) for f(x)=2x+5.",
            "options": {"A": "-1", "B": "1", "C": "11", "D": "-11"},
            "correct_answer": "A", "answer": "A",
        }],
        "equation": [{
            "question_id": "q_eq", "slot_id": "s_eq",
            "equation": "Evaluate x+y+z and 2x-y+2z for x=-1, y=3, z=7.",
            "solution_steps": [], "final_answer": "-1",
        }],
    }}
    candidates, _ = duplicate_judge.shortlist_duplicate_pairs(exam)
    assert candidates == []


def test_exact_duplicate_is_never_exempted_by_authorized_reuse(monkeypatch):
    exam = {"questions": {"true_false": [
        {"question_id": "q1", "slot_id": "s1", "statement": "Same claim.", "answer": True},
        {"question_id": "q2", "slot_id": "s2", "statement": "Same claim.", "answer": True},
    ]}}
    for section in exam["questions"].values():
        for item in section:
            item["reuse_context"] = {"authorized": True}
    plans = {"s1": {"task_id": "task-a"}, "s2": {"task_id": "task-a"}}
    monkeypatch.setattr(duplicate_judge, "create_llm_client", lambda: SimpleNamespace(last_usage={}))
    monkeypatch.setattr(duplicate_judge, "request_structured", lambda *_a, **k: {
        "verdicts": [_facts(pair_id, duplicate=True) for pair_id in k["expected_ids"]]
    })
    result = duplicate_judge.review_duplicate_candidates(exam, plan_by_slot=plans)
    assert result["pair_verdicts"][0]["exact_content"] is True
    assert result["pair_verdicts"][0]["classification"] == "duplicate"


def test_exact_content_is_detected_despite_target_metadata_drift():
    exam = {"questions": {"mcq": [{
        "question_id": "q1", "slot_id": "s1", "question": "What is 2+2?",
        "options": {"A": "4", "B": "5"}, "correct_answer": "A",
    }, {
        "question_id": "q2", "slot_id": "s2", "question": "What is 2+2?",
        "options": {"A": "5", "B": "4"}, "correct_answer": "B",
    }]}}
    plans = {
        "s1": {"task_id": "t1", "task_operation": "solve", "target": "sum"},
        "s2": {"task_id": "t1", "task_operation": "calculate", "target": "total"},
    }
    left, right = duplicate_judge.actual_question_registry(exam, plans)
    assert duplicate_judge._exact_content_pair(left, right) is True


def test_exact_content_handles_formatting_mcq_order_and_targets():
    def registry(questions, plans):
        exam = {"questions": questions}
        return duplicate_judge.actual_question_registry(exam, plans)

    def pair(qtype, left, right, plans):
        items = registry({qtype: [left, right]}, plans)
        return items[0], items[1]

    plans = {"s1": {"task_id": "t1", "task_operation": "solve", "target": "value of x"},
             "s2": {"task_id": "t1", "task_operation": "solve", "target": "value of x"}}
    left, right = pair("equation", {
        "question_id": "q1", "slot_id": "s1", "equation": "2x + 3 = 11",
        "solution_steps": ["x=4"], "final_answer": "4",
    }, {
        "question_id": "q2", "slot_id": "s2", "equation": "2x+3=11",
        "solution_steps": ["divide"], "final_answer": "4",
    }, plans)
    assert duplicate_judge._exact_content_pair(left, right) is True

    # Same provable calculation with different surrounding wording.
    left, right = pair("equation", {
        "question_id": "q1", "slot_id": "s1", "equation": "Solve 2x + 3 = 11",
        "solution_steps": ["x=4"], "final_answer": "4",
    }, {
        "question_id": "q2", "slot_id": "s2", "equation": "Find x: 2x + 3 = 11",
        "solution_steps": ["x=4"], "final_answer": "4",
    }, plans)
    assert duplicate_judge._exact_content_pair(left, right) is False

    mcq_plans = {"s1": {"task_id": "t2", "task_operation": "identify", "target": "correct option"},
                 "s2": {"task_id": "t2", "task_operation": "identify", "target": "correct option"}}
    left, right = pair("mcq", {
        "question_id": "q1", "slot_id": "s1", "question": "What is 2+2?",
        "options": {"A": "4", "B": "5", "C": "6", "D": "7"}, "correct_answer": "A",
    }, {
        "question_id": "q2", "slot_id": "s2", "question": "What is 2+2?",
        "options": {"A": "7", "B": "6", "C": "5", "D": "4"}, "correct_answer": "D",
    }, mcq_plans)
    assert duplicate_judge._exact_content_pair(left, right) is True

    # Same numbers but different target/operation are not exact.
    left, right = pair("equation", {
        "question_id": "q1", "slot_id": "s1", "equation": "2x + 3 = 11",
        "solution_steps": ["x=4"], "final_answer": "4",
    }, {
        "question_id": "q2", "slot_id": "s2",
        "equation": "Verify whether x = 4 satisfies 2x + 3 = 11",
        "solution_steps": ["check"], "final_answer": "4",
    }, {
        "s1": {"task_id": "t1", "task_operation": "solve", "target": "value of x"},
        "s2": {"task_id": "t1", "task_operation": "verify", "target": "whether x=4"},
    })
    assert duplicate_judge._exact_content_pair(left, right) is False

    left, right = pair("equation", {
        "question_id": "q1", "slot_id": "s1", "equation": "2x + 3 = 11",
        "solution_steps": ["x=4"], "final_answer": "4",
    }, {
        "question_id": "q2", "slot_id": "s2", "equation": "3x + 4 = 19",
        "solution_steps": ["x=5"], "final_answer": "5",
    }, plans)
    assert duplicate_judge._exact_content_pair(left, right) is False


def test_python_exact_duplicate_overrides_luna_different_facts():
    practice = {"same_core_knowledge": "no", "same_student_operation_or_method": "no",
                "same_target_or_result": "no", "genuinely_new_reasoning": "yes",
                "one_answer_substantially_gives_away_other": "no"}
    assert duplicate_judge.deterministic_pair_classification(
        practice, exact_content=True,
    ) == "duplicate"


def test_corrected_judgment_is_reused_and_changed_question_invalidates_it(monkeypatch):
    client = SimpleNamespace(last_usage={}, model="judge-model")
    exam = {"questions": {"true_false": [
        {"question_id": "q1", "slot_id": "s1", "statement": "Gravity attracts masses.", "answer": True},
        {"question_id": "q2", "slot_id": "s2", "statement": "Inertia is a force.", "answer": False},
    ]}}
    plans = {"s1": {"task_id": "t1"}, "s2": {"task_id": "t1"}}
    candidates, _ = duplicate_judge.shortlist_duplicate_pairs(exam, plan_by_slot=plans)
    key = duplicate_judge._fact_cache_key(candidates[0], client)
    cached = {key: {
        "same_core_knowledge": "no",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "no",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "no",
    }}
    monkeypatch.setattr(
        duplicate_judge, "request_structured",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("provider called")),
    )
    result = duplicate_judge.review_duplicate_candidates(
        exam, plan_by_slot=plans, client=client, fact_cache=cached,
    )
    assert result["fact_cache_hits"] == 1
    assert result["pair_verdicts"][0]["classification"] == "acceptable_reuse"
    changed = {
        "questions": {"true_false": [
            {"question_id": "q1", "slot_id": "s1", "statement": "Changed claim.", "answer": True},
            {"question_id": "q2", "slot_id": "s2", "statement": "Inertia is a force.", "answer": False},
        ]},
    }
    changed_candidates, _ = duplicate_judge.shortlist_duplicate_pairs(
        changed, plan_by_slot=plans,
    )
    assert changed_candidates
    assert duplicate_judge._fact_cache_key(changed_candidates[0], client) != key


def test_shared_numbers_alone_are_not_a_candidate():
    exam = {"questions": {"true_false": [
        _question("q1", "A quadratic opens upward and has 3 roots, 4 intercepts, and 5 turning points.", True),
        _question("q2", "A chemical reaction uses 3 catalysts, 4 solvents, and 5 vessels.", False),
    ]}}
    candidates, _ = duplicate_judge.shortlist_duplicate_pairs(exam)
    assert candidates == []


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
