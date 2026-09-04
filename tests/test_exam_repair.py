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
    """short_answer missing 1 of 2 -> generate ONLY 1 new question."""
    calls = []

    def _fake_gen(qtype, planned, *a, **k):
        calls.append((qtype, len(planned)))
        # produce exactly len(planned) questions
        return [{"question": f"new {qtype} {i}", "reference_answer": "ans"} for i in range(len(planned))], []

    monkeypatch.setattr(eb, "_generate_type_from_plan", _fake_gen)
    questions = {"short_answer": [{"question": "orig sa", "reference_answer": "ans"}]}
    tasks = [("short_answer", 2)]
    plan_items = {"short_answer": [{"topic": "x", "concept_to_test": "c"}]}
    appended = {}
    eb._repair_shortfalls(
        questions, tasks, plan_items, "", "easy", 1, set(), [], [],
        max_passes=2, appended_by_type=appended,
    )
    assert len(questions["short_answer"]) == 2
    assert questions["short_answer"][0]["question"] == "orig sa"  # original preserved
    assert calls == [("short_answer", 1)]  # ONLY 1 generated
    assert appended == {"short_answer": 1}


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


def test_grounded_fallback_not_blank(monkeypatch):
    """When the plan is empty, the repair must NOT send a blank concept."""
    captured = {}

    def _fake_gen(qtype, planned, *a, **k):
        captured["planned"] = planned
        return [{"question": "q", "reference_answer": "a"} for _ in planned], []

    monkeypatch.setattr(eb, "_generate_type_from_plan", _fake_gen)
    context = "First heading line\nSome body text about science."
    questions = {"essay": []}
    tasks = [("essay", 1)]
    eb._repair_shortfalls(questions, tasks, {}, context, "easy", 1, set(), [], [], max_passes=2)
    p = captured["planned"][0]
    assert p["topic"], "topic must not be blank"
    assert p["concept_to_test"], "concept must not be blank"
    assert "blank" not in p["topic"].lower() and "blank" not in p["concept_to_test"].lower()


def test_fitb_missing_regenerated_full(monkeypatch):
    captured = {}
    stats = create_pipeline_eval([("fill_in_the_blank", 3)], 1)

    def _fake_bundle(planned, *a, **k):
        captured["planned"] = planned
        captured["eval_stats"] = k.get("eval_stats")
        # produce a full valid fitb for the requested count
        return {"fill_in_the_blank": {"word_bank": [f"t{i}" for i in range(3)] + ["d1", "d2"], "items": [
            {"question": f"___ {i}", "answers": [f"t{i}"]} for i in range(3)]}, "mcq": [], "true_false": []}, []

    monkeypatch.setattr(eb, "_generate_obj_bundle", _fake_bundle)
    questions = {}
    tasks = [("fill_in_the_blank", 3)]
    eb._repair_shortfalls(
        questions, tasks, {}, "", "easy", 1, set(), [], [],
        max_passes=2, eval_stats=stats,
    )
    assert len(questions["fill_in_the_blank"]["items"]) == 3
    assert len(questions["fill_in_the_blank"]["word_bank"]) == 5
    assert captured["eval_stats"] is stats


def test_persistent_shortfall_warns(monkeypatch):
    def _fake_gen(*a, **k):
        return [], ["still failing"]

    monkeypatch.setattr(eb, "_generate_type_from_plan", _fake_gen)
    questions = {}
    tasks = [("short_answer", 2)]
    warnings = eb._repair_shortfalls(questions, tasks, {}, "", "easy", 1, set(), [], [], max_passes=2)
    assert any("short" in w for w in warnings)
    assert len(questions.get("short_answer", [])) == 0


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

    import app.llm.client as client_mod

    class FakeClient:
        def chat_json(self, *a, **k):
            return fake_chat_json(*a, **k)

    monkeypatch.setattr(client_mod, "LMStudioClient", lambda: FakeClient())

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
