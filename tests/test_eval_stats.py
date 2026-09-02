from __future__ import annotations

import json

from app.online.eval_stats import (
    create_pipeline_eval,
    finalize_pipeline_eval,
    public_eval,
    record_first_validation,
    record_generation_rejection,
    record_initial_generation,
    record_repair_outcome,
    record_shortfall_result,
)


def _mcq(qid: str) -> dict:
    return {
        "question_id": qid,
        "question": "Q?",
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "correct_answer": "A",
    }


def _tf(qid: str) -> dict:
    return {"question_id": qid, "statement": "T?", "answer": "True"}


def _report(model: int, verdicts: list[dict]) -> dict:
    return {
        "model_number": model,
        "all_pass": all(v["action"] == "PASS" for v in verdicts),
        "verdicts": verdicts,
        "warnings": [],
    }


def _verdict(qid: str, action: str, reason: str = "") -> dict:
    return {"question_id": qid, "action": action, "reason": reason}


def test_requested_initial_shortfall_and_aggregation_are_per_type_and_model():
    stats = create_pipeline_eval([("mcq", 2), ("true_false", 1)], 2)

    # Surplus MCQ must not hide the missing True/False slot.
    initial1 = {"mcq": [_mcq("m1-m1"), _mcq("m1-m2"), _mcq("m1-extra")]}
    record_initial_generation(stats, 1, initial1)
    record_generation_rejection(stats, 1, "mcq", "duplicate", 2)
    after1 = {
        "mcq": [_mcq("m1-m1"), _mcq("m1-m2")],
        "true_false": [_tf("m1-t1")],
    }
    record_shortfall_result(stats, 1, after1, {"true_false": 1})

    initial2 = {
        "mcq": [_mcq("m2-m1"), _mcq("m2-m2")],
        "true_false": [_tf("m2-t1")],
    }
    record_initial_generation(stats, 2, initial2)
    record_shortfall_result(stats, 2, initial2, {})

    result = public_eval(stats)
    assert result["overall"]["requested_questions"] == 6
    assert result["models"]["1"]["generated_first"] == 2
    assert result["models"]["1"]["missing_first"] == 1
    assert result["models"]["1"]["shortfall_generated"] == 1
    assert result["models"]["1"]["shortfall_still_missing"] == 0
    assert result["models"]["1"]["generation_rejection_reasons"] == {"duplicate": 2}
    assert result["overall"]["generated_first"] == 5
    assert result["overall"]["shortfall_generated"] == 1


def test_first_validation_is_stable_and_includes_shortfall_questions():
    stats = create_pipeline_eval([("mcq", 2), ("true_false", 1)], 1)
    exam = {
        "model_number": 1,
        "questions": {
            "mcq": [_mcq("m1"), _mcq("m2")],
            "true_false": [_tf("t1")],
        },
    }
    first = _report(
        1,
        [
            _verdict("m1", "PASS"),
            _verdict("m2", "FIX_ANSWER", "wrong stored answer"),
            _verdict("t1", "FIX_QUESTION", "unclear"),
        ],
    )
    record_first_validation(stats, exam, first)
    # A later all-pass revalidation must not overwrite first-pass telemetry.
    record_first_validation(
        stats,
        exam,
        _report(1, [_verdict("m1", "PASS"), _verdict("m2", "PASS"), _verdict("t1", "PASS")]),
    )
    model = public_eval(stats)["models"]["1"]
    assert model["validation_total_first"] == 3
    assert model["validation_passed_first"] == 1
    assert model["validation_failed_first"] == 2
    assert model["validation_passed_first"] + model["validation_failed_first"] == 3
    assert model["validation_failure_reasons"] == {
        "wrong_answer": 1,
        "unclear_or_invalid_question": 1,
    }


def test_exact_mixed_generation_snapshot_and_no_shortfall():
    tasks = [
        ("mcq", 15),
        ("true_false", 5),
        ("fill_in_the_blank", 10),
        ("short_answer", 5),
    ]
    stats = create_pipeline_eval(tasks, 1)
    questions = {
        "mcq": [_mcq(f"model1_mcq_{i}") for i in range(1, 16)],
        "true_false": [_tf(f"model1_true_false_{i}") for i in range(1, 6)],
        "fill_in_the_blank": {
            "word_bank": [f"term-{i}" for i in range(1, 11)],
            "items": [
                {
                    "question_id": f"model1_fill_in_the_blank_{i}",
                    "question": "Complete ________.",
                    "answers": [f"term-{i}"],
                }
                for i in range(1, 11)
            ],
        },
        "short_answer": [
            {
                "question_id": f"model1_short_answer_{i}",
                "question": "Q?",
                "reference_answer": "A.",
            }
            for i in range(1, 6)
        ],
    }

    record_initial_generation(stats, 1, questions)
    record_shortfall_result(stats, 1, questions, {})

    model = public_eval(stats)["models"]["1"]
    assert model["requested_questions"] == 35
    assert model["generated_first"] == 35
    assert model["missing_first"] == 0
    assert model["shortfall_generated"] == 0
    assert model["shortfall_still_missing"] == 0
    assert model["question_types"]["fill_in_the_blank"]["generated_first"] == 10


def test_unique_repair_eventually_passes_once_across_two_attempts():
    stats = create_pipeline_eval([("mcq", 1)], 1)
    exam = {"model_number": 1, "questions": {"mcq": [_mcq("q1")]}}
    failed = _report(1, [_verdict("q1", "FIX_ANSWER")])
    record_repair_outcome(stats, exam, failed, [], ["q1"])
    record_repair_outcome(stats, exam, failed, ["q1"], [])
    passed = _report(1, [_verdict("q1", "PASS")])
    finalize_pipeline_eval(stats, [exam], [passed])
    model = public_eval(stats)["models"]["1"]
    assert (model["repair_sent"], model["repair_succeeded"], model["repair_failed"]) == (1, 1, 0)


def test_accepted_rewrite_that_still_fails_is_repair_failed():
    stats = create_pipeline_eval([("mcq", 1)], 1)
    exam = {"model_number": 1, "questions": {"mcq": [_mcq("q1")]}}
    failed = _report(1, [_verdict("q1", "FIX_ANSWER")])
    record_repair_outcome(stats, exam, failed, ["q1"], [])
    finalize_pipeline_eval(stats, [exam], [failed])
    model = public_eval(stats)["models"]["1"]
    assert (model["repair_sent"], model["repair_succeeded"], model["repair_failed"]) == (1, 0, 1)
    assert model["final_valid"] == 0
    assert model["final_missing_or_invalid"] == 1


def test_pass_without_applied_writeback_is_not_repair_success():
    stats = create_pipeline_eval([("mcq", 1)], 1)
    exam = {"model_number": 1, "questions": {"mcq": [_mcq("q1")]}}
    failed = _report(1, [_verdict("q1", "FIX_ANSWER")])
    record_repair_outcome(stats, exam, failed, [], ["q1"], ["q1"])
    finalize_pipeline_eval(stats, [exam], [_report(1, [_verdict("q1", "PASS")])])
    model = public_eval(stats)["models"]["1"]
    assert (model["repair_sent"], model["repair_succeeded"], model["repair_failed"]) == (1, 0, 1)


def test_final_validity_is_per_type_and_public_output_is_json_safe():
    stats = create_pipeline_eval([("mcq", 2), ("true_false", 1)], 1)
    # Three valid MCQs cannot compensate for a missing True/False question.
    exam = {
        "model_number": 1,
        "questions": {"mcq": [_mcq("q1"), _mcq("q2"), _mcq("q3")]},
    }
    report = _report(1, [_verdict("q1", "PASS"), _verdict("q2", "PASS"), _verdict("q3", "PASS")])
    finalize_pipeline_eval(stats, [exam], [report])
    result = public_eval(stats)
    assert result["models"]["1"]["question_types"]["mcq"]["final_valid"] == 2
    assert result["models"]["1"]["question_types"]["true_false"]["final_missing_or_invalid"] == 1
    assert result["overall"]["final_valid"] == 2
    assert result["overall"]["final_missing"] == 1
    assert result["overall"]["final_invalid"] == 0
    assert result["overall"]["final_unvalidated"] == 0
    assert result["overall"]["final_missing_or_invalid"] == 1
    assert "_runtime" not in result
    json.dumps(result)


def test_validation_and_final_metrics_separate_unvalidated_and_invalid():
    stats = create_pipeline_eval([("mcq", 3), ("true_false", 1)], 1)
    exam = {
        "model_number": 1,
        "questions": {
            "mcq": [_mcq("pass"), _mcq("bad"), _mcq("unknown")],
        },
    }
    report = _report(
        1,
        [
            _verdict("pass", "PASS"),
            _verdict("bad", "FIX_ANSWER"),
            _verdict("unknown", "UNVALIDATED", "No validator verdict was returned after retry."),
        ],
    )
    record_first_validation(stats, exam, report)
    finalize_pipeline_eval(stats, [exam], [report])
    model = public_eval(stats)["models"]["1"]
    assert model["validation_total_first"] == 3
    assert (
        model["validation_passed_first"]
        + model["validation_failed_first"]
        + model["validation_unvalidated_first"]
    ) == model["validation_total_first"]
    assert model["validation_failure_reasons"] == {"wrong_answer": 1}
    assert model["validator_failure_reasons"] == {"missing_verdict": 1}
    assert (
        model["final_valid"],
        model["final_invalid"],
        model["final_unvalidated"],
        model["final_missing"],
    ) == (1, 1, 1, 1)
