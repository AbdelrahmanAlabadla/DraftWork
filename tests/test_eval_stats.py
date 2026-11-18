from __future__ import annotations

import json

from app.online.eval_stats import (
    create_pipeline_eval,
    derived_rates,
    finalize_pipeline_eval,
    public_eval,
    record_first_validation,
    record_generation_rejection,
    record_initial_generation,
    record_repair_outcome,
    record_shortfall_result,
    summarize_catalog_cache,
    summarize_duplicate_outcomes,
    summarize_pipeline_outcomes,
    summarize_recovery_outcomes,
    summarize_repair_attempts,
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
        ("why", 5),
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
        "why": [
            {
                "question_id": f"model1_why_{i}",
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

def test_failure_code_is_preferred_over_generic_action_category():
    stats = create_pipeline_eval([("mcq", 2)], 1)
    exam = {"model_number": 1, "questions": {"mcq": [_mcq("q1"), _mcq("q2")]}}
    report = _report(1, [
        {**_verdict("q1", "FIX_ANSWER"), "failure_code": "incorrect_result"},
        {**_verdict("q2", "FIX_QUESTION_AND_ANSWER"), "failure_code": "contradicted"},
    ])
    record_first_validation(stats, exam, report)
    model = public_eval(stats)["models"]["1"]
    assert model["validation_failure_reasons"] == {
        "incorrect_result": 1,
        "contradicted": 1,
    }
    assert model["validation_failure_actions"] == {
        "fix_answer": 1,
        "fix_question_and_answer": 1,
    }


def test_operational_validation_failures_are_classified_separately():
    stats = create_pipeline_eval([("mcq", 4)], 1)
    exam = {"model_number": 1, "questions": {
        "mcq": [_mcq("blocked"), _mcq("input"), _mcq("error"), _mcq("missing")],
    }}
    report = _report(1, [
        {**_verdict("blocked", "UNVALIDATED"), "validation_status": "VALIDATION_BLOCKED",
         "block_category": "prerequisite"},
        {**_verdict("input", "UNVALIDATED"), "validation_status": "INPUT_INVALID"},
        {**_verdict("error", "UNVALIDATED"), "validation_status": "ERROR"},
        _verdict("missing", "UNVALIDATED"),
    ])
    record_first_validation(stats, exam, report)
    model = public_eval(stats)["models"]["1"]
    assert model["validation_unvalidated_first"] == 4
    assert model["validator_failure_reasons"] == {
        "prerequisite_validation_block": 1,
        "input_invalid": 1,
        "validator_error": 1,
        "missing_verdict": 1,
    }
    # Operational failures never appear as content rejection reasons.
    assert model["validation_failure_reasons"] == {}


def test_derived_rates_include_rejection_and_validation_coverage():
    rates = derived_rates({
        "requested_questions": 10,
        "generated_first": 8,
        "generation_rejected": 2,
        "validation_total_first": 6,
        "validation_passed_first": 3,
        "repair_sent": 2,
        "repair_succeeded": 1,
        "final_valid": 7,
    })
    assert rates["generation_rejection_rate"] == round(2 / 10, 4)
    assert rates["validation_coverage_rate"] == round(6 / 8, 4)
    assert rates["first_validation_pass_rate"] == 0.5
    assert rates["repair_success_rate"] == 0.5
    assert rates["final_success_rate"] == 0.7


def _lifecycle(**overrides):
    record = {
        "terminal_status": "finalized",
        "failure_stage": None,
        "failure_code": None,
        "repaired": False,
        "repair_attempts": [],
        "repair_manifest_status": None,
    }
    record.update(overrides)
    return record


def test_recovery_summary_uses_authoritative_slot_lifecycle():
    lifecycle = {
        "m1_mcq_1": _lifecycle(),
        "m1_mcq_2": _lifecycle(repaired=True, repair_manifest_status="candidate_generated"),
        "m1_mcq_3": _lifecycle(
            terminal_status="repair_failed",
            failure_stage="repair",
            failure_code="recovery_v3_plan_exhausted",
            repaired=True,
            repair_manifest_status="terminal_failure",
        ),
        "m1_mcq_4": _lifecycle(
            terminal_status="generation_failed",
            failure_stage="generation",
            failure_code="missing_requested_slot",
        ),
    }
    history = [
        {"slot_id": "m1_mcq_2", "attempt": 1, "status": "rejected",
         "repair_strategy": "recovery_v3_targeted_repair", "repair_action": "repair_question",
         "new_failure_code": "incorrect_result"},
        {"slot_id": "m1_mcq_2", "attempt": 1, "status": "accepted",
         "repair_strategy": "recovery_v3_targeted_repair", "repair_action": "repair_question"},
        {"slot_id": "m1_mcq_3", "attempt": 1, "status": "candidate_generation_failed",
         "repair_strategy": "recovery_v3_switch_assignment", "repair_action": "replace_task",
         "new_failure_code": "no_valid_source_grounded_replacement"},
        {"slot_id": "m1_mcq_3", "attempt": 2, "status": "rejected",
         "repair_strategy": "recovery_v3_switch_variation", "repair_action": "switch_variation",
         "new_failure_code": "duplicate_question"},
    ]
    summary = summarize_recovery_outcomes(lifecycle, history)
    assert summary["slots"] == 4
    assert summary["terminal_status"] == {
        "finalized": 2, "repair_failed": 1, "generation_failed": 1,
    }
    assert summary["entered_recovery"] == 2
    assert summary["recovered"] == 1
    assert summary["unresolved"] == 1
    assert summary["slot_failure_codes"] == {
        "recovery_v3_plan_exhausted": 1, "missing_requested_slot": 1,
    }
    assert summary["slot_failure_stages"] == {"repair": 1, "generation": 1}
    assert summary["attempts"] == 3
    assert summary["accepted"] == 1
    assert summary["attempt_outcomes"] == {
        "accepted": 1, "candidate_generation_failed": 1, "rejected": 1,
    }
    assert summary["attempt_strategies"] == {
        "REPAIR": 1, "SWITCH_TASK": 1, "SWITCH_VARIATION": 1,
    }
    assert summary["attempt_strategy_outcomes"]["SWITCH_TASK"] == {
        "candidate_generation_failed": 1,
    }
    assert summary["attempt_failure_codes"] == {
        "no_valid_source_grounded_replacement": 1, "duplicate_question": 1,
    }


def test_repair_attempt_grouping_prefers_strongest_outcome_per_attempt():
    summary = summarize_repair_attempts([
        {"slot_id": "s1", "attempt": 1, "status": "candidate_generated"},
        {"slot_id": "s1", "attempt": 1, "status": "rejected",
         "repair_strategy": "targeted_repair", "new_failure_code": "contradicted"},
    ])
    assert summary["attempts"] == 1
    assert summary["outcomes"] == {"rejected": 1}
    assert summary["strategies"] == {"REPAIR": 1}
    assert summary["failure_codes"] == {"contradicted": 1}


def test_duplicate_and_cache_summaries_only_use_recorded_values():
    duplicates = summarize_duplicate_outcomes({
        "duplicate_repair_rounds": 2,
        "duplicate_replacement_attempts": {"s1": 3, "s2": 1},
        "duplicate_replacement_failures": {"s1": [{"reason": "x"}]},
        "unresolved_duplicate_conflicts": [{"question_ids": ["q1", "q2"]}],
    })
    assert duplicates == {
        "repair_rounds": 2,
        "replacement_attempts": 4,
        "slots_replaced": 2,
        "replacement_failures": 1,
        "unresolved_conflicts": 1,
    }
    assert summarize_catalog_cache({}) == {
        "catalog_cache_hit": False,
        "compiler_cache_hit": False,
        "compatibility_cache_hit": False,
    }
    assert summarize_catalog_cache({
        "assessment_catalog_cache_hit": True,
        "assessment_compiler_cache_hit": False,
        "assessment_catalog_compatibility_cache_hit": True,
    }) == {
        "catalog_cache_hit": True,
        "compiler_cache_hit": False,
        "compatibility_cache_hit": True,
    }


def test_pipeline_outcome_payload_is_json_safe_and_separates_failure_maps():
    summary = summarize_pipeline_outcomes({
        "slot_lifecycle": {"m1_mcq_1": _lifecycle(repaired=True)},
        "validation_repair_history": [],
        "duplicate_repair_rounds": 1,
    })
    json.dumps(summary)
    assert summary["recovery"]["slot_failure_codes"] == {}
    assert summary["recovery"]["attempt_failure_codes"] == {}
    assert summary["duplicates"]["repair_rounds"] == 1

def test_repair_attempt_summary_tracks_raw_strategy_outcomes():
    summary = summarize_repair_attempts([
        {"slot_id": "s1", "attempt": 1, "status": "accepted",
         "repair_strategy": "targeted_followup_repair"},
        {"slot_id": "s2", "attempt": 1, "status": "rejected",
         "repair_strategy": "targeted_followup_repair", "new_failure_code": "contradicted"},
        {"slot_id": "s3", "attempt": 1, "status": "accepted",
         "repair_strategy": "catalog_replacement"},
    ])
    assert summary["strategy_outcomes"]["REPAIR"] == {"accepted": 1, "rejected": 1}
    assert summary["strategy_outcomes"]["SWITCH_TASK"] == {"accepted": 1}
    assert summary["strategy_detail_outcomes"]["targeted_followup_repair"] == {
        "accepted": 1, "rejected": 1,
    }
    assert summary["strategy_detail_outcomes"]["catalog_replacement"] == {
        "accepted": 1,
    }
