from __future__ import annotations

import logging
from types import SimpleNamespace

from app.online import duplicate_judge


def _mcq(qid: str, stem: str, answer: str) -> dict:
    return {
        "question_id": qid, "slot_id": f"s_{qid}", "question": stem,
        "options": {"A": answer, "B": "wrong 1", "C": "wrong 2", "D": "wrong 3"},
        "correct_answer": "A",
    }


def _verdict(pair_id: str, classification: str) -> dict:
    facts = {
        "duplicate": ("yes", "yes", "yes", "no", "no"),
        "excessive_overlap": ("yes", "no", "yes", "yes", "yes"),
        "acceptable_reuse": ("no", "no", "no", "yes", "no"),
    }[classification]
    return dict(zip(("pair_id", *duplicate_judge.SEMANTIC_FACT_FIELDS), (pair_id, *facts)))


def test_actual_question_registry_expands_mcq_answer_and_complete_solution():
    exam = {"questions": {
        "mcq": [_mcq("q1", "Which formula?", "E = hν")],
        "word_problem": [{
            "question_id": "q2", "slot_id": "s2", "question": "Calculate energy.",
            "solution_steps": ["Use E = hν", "Substitute values"], "final_answer": "4 J",
        }],
    }}
    registry = {item["question_id"]: item for item in duplicate_judge.actual_question_registry(exam)}
    assert "A: E = hν" in registry["q1"]["answer"]
    assert "Use E = hν" in registry["q2"]["answer"]
    assert "Final: 4 J" in registry["q2"]["answer"]


def test_all_pairs_threshold_top_two_and_score_distribution(monkeypatch, caplog):
    exam = {"questions": {"mcq": [
        _mcq(f"q{i}", f"Question {i}", f"Answer {i}") for i in range(1, 13)
    ]}}
    calls = 0
    def fake_embed(texts, max_length=1024):
        nonlocal calls
        calls += 1
        # Adjacent questions are nearest; most pairs remain below 0.80.
        return [[1.0, i / 100.0] for i, _text in enumerate(texts)]
    monkeypatch.setattr(duplicate_judge, "embed_dense", fake_embed)

    with caplog.at_level(logging.INFO, logger="DUPLICATE_JUDGE"):
        candidates, distribution = duplicate_judge.shortlist_duplicate_pairs(exam)

    assert calls == 3
    assert distribution["question_count"] == 12
    assert distribution["pair_count"] == 66
    assert distribution["threshold"] == 0.80
    assert distribution["top_neighbours"] == 2
    assert "Duplicate similarity distribution" in caplog.text
    included = {(p["left"]["question_id"], p["right"]["question_id"]) for p in candidates}
    assert all(any(qid in pair for pair in included) for qid in [f"q{i}" for i in range(1, 13)])


def test_judge_uses_actual_content_and_builds_duplicate_groups(monkeypatch):
    exam = {"questions": {"mcq": [
        _mcq("q1", "What is mechanical work?", "Force times displacement along the force"),
        _mcq("q2", "Define work in mechanics.", "Force times displacement along the force"),
        _mcq("q3", "State the formula for power.", "P = W/t"),
    ]}}
    monkeypatch.setattr(duplicate_judge, "embed_dense", lambda texts, max_length=1024: [
        [1.0, 0.0], [0.99, 0.01], [0.0, 1.0]
    ])
    monkeypatch.setattr(duplicate_judge, "create_llm_client", lambda: SimpleNamespace(last_usage={"total_tokens": 50}))
    captured = {}
    def fake_request(_client, prompt, **kwargs):
        captured["prompt"] = prompt
        pair_ids = kwargs["expected_ids"]
        return {"verdicts": [
            _verdict(pair_id, "duplicate" if pair_id == "pair_1_2" else "acceptable_reuse")
            for pair_id in pair_ids
        ]}
    monkeypatch.setattr(duplicate_judge, "request_structured", fake_request)

    result = duplicate_judge.review_duplicate_candidates(exam)

    assert "Force times displacement along the force" in captured["prompt"]
    assert "knowledge_focus" not in captured["prompt"]
    assert result["duplicate_groups"][0]["question_ids"] == ["q1", "q2"]
    assert len(result["duplicate_groups"][0]["replace_question_ids"]) == 1


def test_28_questions_score_all_378_pairs_in_memory(monkeypatch):
    exam = {"questions": {"mcq": [
        _mcq(f"q{i}", f"Independent question {i}", f"Independent answer {i}")
        for i in range(1, 29)
    ]}}
    monkeypatch.setattr(duplicate_judge, "embed_dense", lambda texts, max_length=1024: [
        [1.0 if row == column else 0.0 for column in range(len(texts))]
        for row in range(len(texts))
    ])
    _candidates, distribution = duplicate_judge.shortlist_duplicate_pairs(exam)
    assert distribution["question_count"] == 28
    assert distribution["pair_count"] == 378


def test_excessive_overlap_edges_join_one_group():
    candidates = [
        {"pair_id": "p1", "left": {"question_id": "a", "question_type": "mcq", "answer": "x"},
         "right": {"question_id": "b", "question_type": "definition", "answer": "x"}},
        {"pair_id": "p2", "left": {"question_id": "b", "question_type": "definition", "answer": "x"},
         "right": {"question_id": "c", "question_type": "true_false", "answer": "x"}},
    ]
    verdicts = [
        {**_verdict("p1", "duplicate"), "classification": "duplicate"},
        {**_verdict("p2", "excessive_overlap"), "classification": "excessive_overlap"},
    ]
    groups = duplicate_judge.build_duplicate_groups(candidates, verdicts)
    assert len(groups) == 1
    assert groups[0]["question_ids"] == ["a", "b", "c"]
    assert set(groups[0]["replace_question_ids"]) in ({"b"}, {"a", "c"})


def test_deterministic_labels_keep_elimination_and_substitution_acceptable():
    facts = {
        "same_core_knowledge": "yes", "same_student_operation_or_method": "no",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "no",
    }
    assert duplicate_judge.deterministic_pair_classification(facts) == "acceptable_reuse"


def test_infer_then_derive_and_direct_translation_are_acceptable_reuse():
    facts = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "no",
    }
    assert duplicate_judge.deterministic_pair_classification(facts) == "acceptable_reuse"
    prompt = duplicate_judge._prompt([{
        "pair_id": "p_equations",
        "left": {"question_id": "equation_1", "question_type": "equation",
                 "stem": "Infer the initial balance, then derive B=500-25x.",
                 "answer": "The initial balance is 500, so B=500-25x."},
        "right": {"question_id": "equation_4", "question_type": "equation",
                  "stem": "Translate a fixed AED 100 fee plus 10% of m.",
                  "answer": "C=100+0.10m."},
        "signals": ["top_neighbour"], "combined_similarity": 0.579,
        "stem_similarity": 0.5, "answer_similarity": 0.4,
    }])
    assert "inferring an unknown starting quantity" in prompt
    assert '"signals"' not in prompt
    assert "combined_similarity" not in prompt


def test_model_supplied_classification_cannot_override_semantic_facts(monkeypatch):
    exam = {"questions": {"mcq": [
        _mcq("q1", "Translate a fixed charge.", "C=10+2x"),
        _mcq("q2", "Derive a balance after inferring its start.", "B=50-2x"),
    ]}}
    monkeypatch.setattr(duplicate_judge, "embed_dense", lambda texts, max_length=1024: [
        [1.0, 0.0], [0.99, 0.01]
    ])
    monkeypatch.setattr(duplicate_judge, "request_structured", lambda *_args, **kwargs: {
        "verdicts": [{
            **_verdict(kwargs["expected_ids"][0], "acceptable_reuse"),
            "classification": "duplicate",
        }]
    })
    result = duplicate_judge.review_duplicate_candidates(
        exam, client=SimpleNamespace(last_usage={}, model="controlled")
    )
    assert result["pair_verdicts"][0]["classification"] == "acceptable_reuse"


def test_substantial_answer_giveaway_cannot_be_cancelled_by_new_reasoning():
    facts = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "yes",
    }
    assert duplicate_judge.deterministic_pair_classification(facts) == "excessive_overlap"


def test_giveaway_with_different_target_and_operation_is_allowed():
    facts = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "no",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "yes",
    }
    assert duplicate_judge.deterministic_pair_classification(facts) == "acceptable_reuse"


def test_shared_stem_formula_with_different_assessment_angle_is_allowed():
    facts = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "no",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "yes",
    }
    assert duplicate_judge.deterministic_pair_classification(facts) == "acceptable_reuse"
    prompt = duplicate_judge._prompt([{
        "pair_id": "p_stem_formula",
        "left": {"question_id": "q1", "question_type": "equation",
                 "stem": "Explain why allowed energy levels produce discrete lines.",
                 "answer": "Delta E = h nu links each transition to one frequency."},
        "right": {"question_id": "q2", "question_type": "equation",
                  "stem": "Rearrange Delta E = h nu to solve for nu.",
                  "answer": "nu = Delta E / h."},
    }])
    assert "shared prerequisite equation" in prompt
    assert "exact requested conclusion" in prompt


def test_exact_answer_giveaway_with_shared_assessed_work_is_rejected():
    facts = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "yes",
    }
    assert duplicate_judge.deterministic_pair_classification(facts) == "excessive_overlap"


def test_intentional_reuse_is_judged_by_assessment_overlap():
    reused_task_different_angle = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "no",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "yes",
    }
    assert duplicate_judge.deterministic_pair_classification(reused_task_different_angle) == (
        "acceptable_reuse"
    )


def test_different_points_with_same_verification_procedure_are_not_new_reasoning():
    facts = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "yes",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "no",
    }
    assert duplicate_judge.deterministic_pair_classification(facts) == "duplicate"
    prompt = duplicate_judge._prompt([{
        "pair_id": "p_points",
        "left": {"question_id": "q1", "question_type": "why",
                 "stem": "Check whether (0,0) satisfies the system.", "answer": "It does."},
        "right": {"question_id": "q2", "question_type": "why",
                  "stem": "Check whether (-4,0) satisfies the same system.", "answer": "It does not."},
        "signals": ["same_normalized_equation"],
        "combined_similarity": 0.7, "stem_similarity": 0.7, "answer_similarity": 0.2,
    }])
    assert "Different test points and results do not create a new reasoning process" in prompt
    assert "same operation=yes, same target=yes, new reasoning=no" in prompt


def test_unchanged_pair_reuses_content_keyed_luna_facts(monkeypatch):
    exam = {"questions": {"equation": [
        {"question_id": "q_ineq", "slot_id": "s1",
         "equation": "Model at most 25 hours and at least AED 150 as inequalities.",
         "solution_steps": ["x+y<=25", "r1*x+r2*y>=150"],
         "final_answer": "x+y<=25; r1*x+r2*y>=150"},
        {"question_id": "q_eq", "slot_id": "s2",
         "equation": "Model exact total hours and wages as equations.",
         "solution_steps": ["x+y=350", "6.85x+11y=2978.50"],
         "final_answer": "x+y=350; 6.85x+11y=2978.50"},
    ]}}
    monkeypatch.setattr(duplicate_judge, "embed_dense", lambda texts, max_length=1024: [
        [1.0, 0.0], [0.9, 0.1]
    ])
    client = SimpleNamespace(last_usage={}, model="gpt-6-luna")
    calls = 0
    def fake_request(_client, _prompt, **kwargs):
        nonlocal calls
        calls += 1
        # A second call would deliberately flip new reasoning; the cache must
        # prevent that call for unchanged question+answer content.
        return {"verdicts": [{
            "pair_id": kwargs["expected_ids"][0],
            "same_core_knowledge": "yes",
            "same_student_operation_or_method": "yes",
            "same_target_or_result": "no",
            "genuinely_new_reasoning": "yes" if calls == 1 else "no",
            "one_answer_substantially_gives_away_other": "no",
        }]}
    monkeypatch.setattr(duplicate_judge, "request_structured", fake_request)

    first = duplicate_judge.review_duplicate_candidates(exam, client=client)
    second = duplicate_judge.review_duplicate_candidates(
        exam, client=client, fact_cache=first["fact_cache"]
    )

    assert calls == 1
    assert first["pair_verdicts"] == second["pair_verdicts"]
    assert second["fact_cache_hits"] == 1


def test_changed_answer_invalidates_only_affected_pair_fact(monkeypatch):
    exam = {"questions": {"mcq": [
        _mcq("q1", "Solve system A", "(1, 2)"),
        _mcq("q2", "Verify system A", "(1, 2)"),
        _mcq("q3", "Independent question", "7"),
    ]}}
    monkeypatch.setattr(duplicate_judge, "embed_dense", lambda texts, max_length=1024: [
        [1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 0.0, 1.0]
    ])
    client = SimpleNamespace(last_usage={}, model="gpt-6-luna")
    judged = []
    def fake_request(_client, _prompt, **kwargs):
        judged.append(list(kwargs["expected_ids"]))
        return {"verdicts": [{
            "pair_id": pair_id, "same_core_knowledge": "no",
            "same_student_operation_or_method": "no", "same_target_or_result": "no",
            "genuinely_new_reasoning": "yes",
            "one_answer_substantially_gives_away_other": "no",
        } for pair_id in kwargs["expected_ids"]]}
    monkeypatch.setattr(duplicate_judge, "request_structured", fake_request)
    first = duplicate_judge.review_duplicate_candidates(exam, client=client)
    exam["questions"]["mcq"][0]["options"]["A"] = "(3, 4)"
    second = duplicate_judge.review_duplicate_candidates(
        exam, client=client, fact_cache=first["fact_cache"]
    )
    assert second["fact_cache_hits"] >= 1
    assert second["fact_cache_misses"] >= 1


def test_catalog_reuse_pair_bypasses_cosine_shortlisting(monkeypatch):
    exam = {"questions": {"why": [
        {"question_id": "q1", "slot_id": "s1", "question": "Independent alpha task",
         "reference_answer": "alpha"},
        {"question_id": "q2", "slot_id": "s2", "question": "Independent beta task",
         "reference_answer": "beta"},
        {"question_id": "q3", "slot_id": "s3", "question": "Independent gamma task",
         "reference_answer": "gamma"},
        {"question_id": "q4", "slot_id": "s4", "question": "Independent delta task",
         "reference_answer": "delta"},
    ]}}
    monkeypatch.setattr(duplicate_judge, "embed_dense", lambda texts, max_length=1024: [
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ])
    candidates, _ = duplicate_judge.shortlist_duplicate_pairs(
        exam, {frozenset(("q1", "q4"))}
    )
    forced = next(pair for pair in candidates if {
        pair["left"]["question_id"], pair["right"]["question_id"]
    } == {"q1", "q4"})
    assert "forced_catalog_reuse" in forced["signals"]


def test_identical_math_equations_and_point_are_shortlisted_below_threshold(monkeypatch):
    exam = {"questions": {
        "mcq": [{
            "question_id": "q_verify", "slot_id": "s1",
            "question": "Verify that (-2,-3) solves 2x-y=-1 and 2y+5x=-16.",
            "options": {"A": "Both equations hold", "B": "Only first", "C": "Only second", "D": "Neither"},
            "correct_answer": "A",
        }],
        "why": [{
            "question_id": "q_graph", "slot_id": "s2",
            "question": "Graph 2x-y=-1 and 2y+5x=-16 and identify their intersection.",
            "reference_answer": "The lines intersect at (-2,-3).",
        }],
    }}
    monkeypatch.setattr(duplicate_judge, "embed_dense", lambda texts, max_length=1024: [[1.0, 0.0], [0.7, 0.714]])
    candidates, _distribution = duplicate_judge.shortlist_duplicate_pairs(exam)
    pair = candidates[0]
    assert pair["discovery_similarity"] < 0.80
    assert {pair["left"]["question_id"], pair["right"]["question_id"]} == {"q_verify", "q_graph"}
    assert any(signal in pair["signals"] for signal in (
        "same_normalized_equation", "important_repeated_numbers"
    ))


def test_authorized_distinct_numeric_instance_is_auditable_practice_variation():
    facts = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "yes",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "no",
    }
    assert duplicate_judge.deterministic_pair_classification(facts) == "duplicate"
    assert duplicate_judge.deterministic_pair_classification(
        facts, authorized_practice_variation=True,
    ) == "acceptable_practice_variation"


def test_authorized_wording_only_copy_remains_duplicate():
    facts = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "yes",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "yes",
    }
    assert duplicate_judge.deterministic_pair_classification(
        facts, authorized_practice_variation=True,
    ) == "duplicate"
