"""Unit tests for minimal-diff count repair (_repair_shortfalls) and structural repair.

These mock the LLM so no server call is made. Focus: missing -> add only the
missing amount, extra -> remove only the extra amount, correct -> untouched.
"""
from __future__ import annotations

import app.online.exam_builder as eb
from app.api.evaluation_store import evaluation_row
from app.online.eval_stats import (
    create_pipeline_eval,
    public_eval,
    record_generation_rejection,
    record_initial_generation_counts,
    record_shortfall_result,
)


def _mcq(text: str = "M?") -> dict:
    return {
        "question": text,
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "correct_answer": "A",
    }


def test_extra_removed_only():
    questions = {"mcq": [_mcq("m1"), _mcq("m2"), _mcq("m3"), _mcq("m4")]}
    tasks = [("mcq", 3)]
    warnings = eb._repair_shortfalls(
        questions, tasks, {}, "", "easy", 1, set(), [], [], max_passes=2
    )
    assert len(questions["mcq"]) == 3, questions["mcq"]
    assert any("extra" in w for w in warnings)
    # The first 3 (valid) questions are preserved unchanged.
    assert {q["question"] for q in questions["mcq"]} == {"m1", "m2", "m3"}


def test_correct_untouched_no_calls(monkeypatch):
    called = {"n": 0}

    def _fake_bundle(*a, **k):
        called["n"] += 1
        return {"mcq": [], "true_false": [], "fill_in_the_blank": None}, []

    monkeypatch.setattr(eb, "_generate_obj_bundle", _fake_bundle)
    questions = {"mcq": [_mcq("m1"), _mcq("m2")], "true_false": [{"statement": "t1", "answer": "True"}]}
    tasks = [("mcq", 2), ("true_false", 1)]
    eb._repair_shortfalls(questions, tasks, {}, "", "easy", 1, set(), [], [], max_passes=2)
    assert called["n"] == 0  # no generation needed
    assert len(questions["mcq"]) == 2
    assert len(questions["true_false"]) == 1


def test_missing_fills_exact_amount(monkeypatch):
    """why missing 1 of 2 -> generate ONLY 1 new question."""
    calls = []

    def _fake_gen(qtype, planned, *a, **k):
        calls.append((qtype, len(planned)))
        # produce exactly len(planned) questions
        return [{"question": f"new {qtype} {i}", "reference_answer": "ans"} for i in range(len(planned))], []

    monkeypatch.setattr(eb, "_generate_type_from_plan", _fake_gen)
    questions = {"why": [{"question": "orig sa", "reference_answer": "ans"}]}
    tasks = [("why", 2)]
    plan_items = {"why": [
        {"topic": "x", "concept_to_test": "c1"},
        {"topic": "x", "concept_to_test": "c2"},
    ]}
    appended = {}
    eb._repair_shortfalls(
        questions, tasks, plan_items, "", "easy", 1, set(), [], [],
        max_passes=2, appended_by_type=appended,
    )
    assert len(questions["why"]) == 2
    assert questions["why"][0]["question"] == "orig sa"  # original preserved
    assert calls == [("why", 1)]  # ONLY 1 generated
    assert appended == {"why": 1}


def test_shortfall_rejections_reach_model_overall_and_persistence(monkeypatch):
    stats = create_pipeline_eval([("essay", 1)], 1)
    record_initial_generation_counts(stats, 1, {"essay": 0})
    generated = {
        "question": "Explain static and dynamic data.",
        "reference_answer": "Static data is fixed; dynamic data changes.",
        "key_points": ["static", "dynamic"],
    }

    def _fake_gen(qtype, planned, *args, eval_stats=None, **kwargs):
        assert qtype == "essay"
        assert len(planned) == 1
        assert eval_stats is stats
        record_generation_rejection(
            eval_stats, 1, qtype, "near_duplicate"
        )
        record_generation_rejection(
            eval_stats, 1, qtype, "forbidden_content"
        )
        return [generated], []

    monkeypatch.setattr(eb, "_generate_type_from_plan", _fake_gen)
    questions = {"essay": []}
    appended = {}
    eb._repair_shortfalls(
        questions,
        [("essay", 1)],
        {"essay": [{"topic": "data", "concept_to_test": "data types"}]},
        "context",
        "easy",
        1,
        set(),
        [],
        [],
        max_passes=1,
        appended_by_type=appended,
        eval_stats=stats,
    )
    record_shortfall_result(stats, 1, questions, appended)

    assert questions == {"essay": [generated]}
    assert appended == {"essay": 1}
    result = public_eval(stats)
    expected_reasons = {"near_duplicate": 1, "forbidden_content": 1}
    assert result["models"]["1"]["generation_rejected"] == 2
    assert result["models"]["1"]["generation_rejection_reasons"] == expected_reasons
    assert result["overall"]["generation_rejected"] == 2
    assert result["overall"]["generation_rejection_reasons"] == expected_reasons

    row = evaluation_row("exam_shortfall", result, 1, 1)
    assert row["generation_rejected"] == 2
    assert row["generation_rejection_reasons"] == expected_reasons
    assert row["model_performance"]["1"]["generation_rejected"] == 2
    assert (
        row["question_type_performance"]["essay"]["generation_rejection_reasons"]
        == expected_reasons
    )


def test_empty_plan_is_not_fabricated_by_generator(monkeypatch):
    """A missing planner slot remains explicit instead of being invented."""
    captured = {"called": False}

    def _fake_gen(qtype, planned, *a, **k):
        captured["called"] = True
        return [], []

    monkeypatch.setattr(eb, "_generate_type_from_plan", _fake_gen)
    context = "First heading line\nSome body text about science."
    questions = {"essay": []}
    tasks = [("essay", 1)]
    warnings = eb._repair_shortfalls(
        questions, tasks, {}, context, "easy", 1, set(), [], [], max_passes=2
    )
    assert captured["called"] is False
    assert any("m1_essay_1" in warning for warning in warnings)


def test_fitb_missing_regenerated_full(monkeypatch):
    captured = {}
    stats = create_pipeline_eval([("fill_in_the_blank", 3)], 1)

    def _fake_fitb(count, planned, *a, **k):
        captured.setdefault("planned", []).extend(planned)
        captured["eval_stats"] = k.get("eval_stats")
        answers = ["alpha", "beta", "gamma"]
        distractors = [["delta", "epsilon"], ["zeta", "eta"], ["theta", "iota"]]
        index = int(planned[0]["slot_id"].rsplit("_", 1)[1]) - 1
        return {"word_bank": [answers[index], *distractors[index]], "items": [{
            "question": f"The term is ________ for item {index}.",
            "answers": [answers[index]], "distractors": distractors[index],
            "slot_id": planned[0]["slot_id"],
        }]}, []

    monkeypatch.setattr(eb, "_generate_fitb_type", _fake_fitb)
    questions = {}
    tasks = [("fill_in_the_blank", 3)]
    plans = {"fill_in_the_blank": [
        {"topic": "t", "concept_to_test": f"c{i}"} for i in range(3)
    ]}
    eb._repair_shortfalls(
        questions, tasks, plans, "", "easy", 1, set(), [], [],
        max_passes=2, eval_stats=stats,
    )
    assert len(questions["fill_in_the_blank"]["items"]) == 3
    assert len(questions["fill_in_the_blank"]["word_bank"]) == 5
    assert captured["eval_stats"] is stats


def test_fitb_shortfall_replaces_only_failed_slot_after_bounded_stem_retries(monkeypatch):
    existing = [
        {"slot_id": f"m1_fill_in_the_blank_{index}", "question": f"Clue word {word} ________.",
         "answers": [word], "distractors": ["theta", "lambda"]}
        for index, word in enumerate(["alpha", "beta", "gamma", "delta"], start=1)
    ]
    questions = {"fill_in_the_blank": {
        "items": existing,
        "word_bank": ["alpha", "beta", "gamma", "delta", "theta", "lambda"],
    }}
    plans = {"fill_in_the_blank": [
        {"slot_id": f"m1_fill_in_the_blank_{index}", "fitb_entry_id": f"fitb_{word}",
         "fitb_answer": word}
        for index, word in enumerate(["alpha", "beta", "gamma", "delta", "epsilon"], start=1)
    ]}
    calls = []

    def fake_generate(_count, planned, *args, **kwargs):
        calls.append(planned[0]["fitb_entry_id"])
        if planned[0]["fitb_entry_id"] != "fitb_zeta":
            return {"word_bank": [], "items": []}, ["invalid_structure"]
        return {"word_bank": ["zeta", "theta", "lambda"], "items": [{
            "slot_id": "m1_fill_in_the_blank_5", "fitb_entry_id": "fitb_zeta",
            "question": "Final grounded clue ________.", "answers": ["zeta"],
            "distractors": ["theta", "lambda"],
        }]}, []

    replaced = []
    def replace(slot_id, current, reason):
        replaced.append((slot_id, current["fitb_entry_id"], reason))
        return {**current, "fitb_entry_id": "fitb_zeta", "fitb_answer": "zeta"}

    monkeypatch.setattr(eb, "_generate_fitb_type", fake_generate)
    eb._repair_shortfalls(
        questions, [("fill_in_the_blank", 5)], plans, "", "easy", 1,
        set(), [], [], max_passes=2, fitb_entry_replacer=replace,
    )

    assert [item["slot_id"] for item in questions["fill_in_the_blank"]["items"]] == [
        f"m1_fill_in_the_blank_{index}" for index in range(1, 6)
    ]
    assert replaced[0][:2] == ("m1_fill_in_the_blank_5", "fitb_epsilon")
    assert calls == ["fitb_epsilon", "fitb_epsilon", "fitb_zeta"]
    assert len(questions["fill_in_the_blank"]["word_bank"]) == 7
    assert "zeta" in questions["fill_in_the_blank"]["word_bank"]
    assert "epsilon" not in questions["fill_in_the_blank"]["word_bank"]


def test_persistent_shortfall_warns(monkeypatch):
    def _fake_gen(*a, **k):
        return [], ["still failing"]

    monkeypatch.setattr(eb, "_generate_type_from_plan", _fake_gen)
    questions = {}
    tasks = [("why", 2)]
    warnings = eb._repair_shortfalls(questions, tasks, {}, "", "easy", 1, set(), [], [], max_passes=2)
    assert any("short" in w for w in warnings)
    assert len(questions.get("why", [])) == 0


def test_split_valid_invalid_detects_bad_key_points():
    """An essay whose key_points is a string must be flagged as invalid (rejected)."""
    from app.online.models import split_valid_invalid

    raw = {
        "questions": [
            {
                "question": "What is photosynthesis?",
                "reference_answer": "Plants use light to make food.",
                "key_points": "light, water, carbon dioxide",
            }
        ]
    }
    valid, invalid = split_valid_invalid("essay", raw)
    assert valid == []
    assert len(invalid) == 1
    assert invalid[0]["question"] == "What is photosynthesis?"


def test_structural_repair_preserves_valid_content(monkeypatch):
    """Invalid item is sent back for repair; repair fixes key_points to an array."""

    # Simulate the LLM repair: fix ONLY key_points, keep the question unchanged.
    def fake_chat_json(prompt, system_prompt=None, temperature=0.0, max_tokens=0, **kw):
        return {
            "questions": [
                {
                    "question": "What is photosynthesis?",
                    "reference_answer": "Plants use light to make food.",
                    "key_points": ["light", "water", "carbon dioxide"],
                }
            ]
        }

    class FakeClient:
        def chat_json(self, *a, **k):
            return fake_chat_json(*a, **k)

    monkeypatch.setattr("app.llm.factory.create_llm_client", FakeClient)

    invalid = [{
        "question": "What is photosynthesis?",
        "reference_answer": "Plants use light to make food.",
        "key_points": "light, water, carbon dioxide",
    }]
    from app.online.exam_builder import _repair_invalid_items

    repaired, warnings = _repair_invalid_items("essay", invalid, "ctx", "easy", 1)
    assert len(repaired) == 1
    q = repaired[0]
    assert q["question"] == "What is photosynthesis?"  # content preserved
    assert isinstance(q["key_points"], list)          # structure fixed
    assert q["key_points"] == ["light", "water", "carbon dioxide"]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
