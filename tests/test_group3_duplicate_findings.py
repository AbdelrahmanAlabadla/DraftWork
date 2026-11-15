"""Group 3 structured findings and bounded reassessment tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.online import duplicate_pipeline
from app.online.duplicate_judge import shortlist_duplicate_pairs

FORENSIC = Path(r"C:\Users\abdul\AppData\Local\Temp\opencode\forensic")


def _find(value, key, depth=0):
    if depth > 7:
        return None
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find(child, key, depth + 1)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find(child, key, depth + 1)
            if found is not None:
                return found
    return None


def _report() -> dict:
    left = {
        "question_id": "q1", "slot_id": "s1", "question_type": "true_false",
        "task_id": "t1", "selected_variation_id": "v1",
        "stem": "A claim.", "answer": "True", "target": "claim one",
        "task_operation": "verify",
    }
    right = {
        "question_id": "q2", "slot_id": "s2", "question_type": "true_false",
        "task_id": "t1", "selected_variation_id": "v1",
        "stem": "A claim.", "answer": "True", "target": "claim one",
        "task_operation": "verify",
    }
    verdict = {
        "pair_id": "q1::q2",
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "yes",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "no",
        "classification": "duplicate",
        "exact_content": True,
        "authorized_practice_variation": False,
    }
    return {
        "candidate_pairs": [{
            "pair_id": "q1::q2", "left": left, "right": right,
            "signals": ["near_identical_stem"],
        }],
        "pair_verdicts": [verdict],
        "duplicate_groups": [{
            "group_id": "g1", "question_ids": ["q1", "q2"],
            "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
        }],
    }


def test_findings_capture_refs_status_and_retention():
    findings = duplicate_pipeline._build_duplicate_findings(_report(), 1, 2)
    assert len(findings) == 1
    finding = findings[0]
    assert finding["status"] == "confirmed"
    assert finding["requires_intervention"] is True
    assert finding["retain_question_id"] == "q1"
    assert finding["replace_question_id"] == "q2"
    assert finding["screening"]["signals"] == ["near_identical_stem"]
    assert finding["judgment"]["exact_content"] is True
    assert {ref["question_id"] for ref in finding["question_refs"]} == {"q1", "q2"}
    assert finding["question_types"] == ["true_false", "true_false"]


def test_acceptable_reuse_finding_has_no_intervention():
    report = _report()
    report["pair_verdicts"][0]["classification"] = "acceptable_reuse"
    report["duplicate_groups"] = []
    finding = duplicate_pipeline._build_duplicate_findings(report, 1, 1)[0]
    assert finding["status"] == "acceptable_reuse"
    assert finding["requires_intervention"] is False
    assert finding["replace_question_id"] is None


@pytest.mark.skipif(not FORENSIC.exists(), reason="historical duplicate stream unavailable")
def test_historical_false_positive_is_not_shortlisted():
    plan_event = json.loads(
        (FORENSIC / "0857_planning_plan_completed.json").read_text(encoding="utf-8")
    )
    plans = _find(plan_event, "plans") or []
    plan_by_slot = {
        item["slot_id"]: item
        for plan in plans if plan.get("model_number") == 1
        for values in (plan.get("items") or {}).values()
        for item in values
    }
    generated = json.loads(
        (FORENSIC / "0863_generation_questions_generated.json").read_text(encoding="utf-8")
    )
    exams = _find(generated, "generated_exams") or []
    exam = next(value for value in exams if value.get("model_number") == 1)
    candidates, stats = shortlist_duplicate_pairs(exam, plan_by_slot=plan_by_slot)
    assert stats["question_count"] >= 20
    assert not any(
        set(candidate["pair_id"].split("::"))
        == {"model1_equation_2", "model1_mcq_3"}
        for candidate in candidates
    )
    # The old number-only screen sent many weak pairs. The new screen must not.
    assert len(candidates) < 9


def test_same_task_with_different_numbers_is_still_shortlisted():
    exam = {"questions": {
        "equation": [
            {"question_id": "q1", "slot_id": "s1", "equation": "2x+3=11",
             "solution_steps": ["x=4"], "final_answer": "4"},
            {"question_id": "q2", "slot_id": "s2", "equation": "3x+4=19",
             "solution_steps": ["x=5"], "final_answer": "5"},
        ],
    }}
    plans = {
        "s1": {"task_id": "task-a", "task_operation": "solve", "target": "value of x"},
        "s2": {"task_id": "task-a", "task_operation": "solve", "target": "value of x"},
    }
    candidates, _ = shortlist_duplicate_pairs(exam, plan_by_slot=plans)
    assert [value["pair_id"] for value in candidates] == ["q1::q2"]


def test_finding_resolves_after_revalidation_rescan():
    exam = {"questions": {"true_false": [
        {"question_id": "q1", "slot_id": "s1", "statement": "A claim.", "answer": True},
        {"question_id": "q2", "slot_id": "s2", "statement": "Another claim.", "answer": False},
    ]}}
    existing = duplicate_pipeline._build_duplicate_findings(_report(), 1, 1)
    acceptable = _report()
    acceptable["pair_verdicts"][0]["classification"] = "acceptable_reuse"
    acceptable["pair_verdicts"][0]["exact_content"] = False
    acceptable["duplicate_groups"] = []
    current = duplicate_pipeline._build_duplicate_findings(acceptable, 1, 2)
    merged = duplicate_pipeline._merge_duplicate_findings(existing, current, exam, 2)
    assert len(merged) == 1
    assert merged[0]["status"] == "resolved"
    assert merged[0]["scan_history"][-1]["classification"] == "acceptable_reuse"


def test_repaired_finding_stays_unresolved_when_duplicate_remains():
    exam = {"questions": {"true_false": [
        {"question_id": "q1", "slot_id": "s1", "statement": "A claim.", "answer": True},
        {"question_id": "q2", "slot_id": "s2", "statement": "A claim.", "answer": True},
    ]}}
    existing = duplicate_pipeline._build_duplicate_findings(_report(), 1, 1)
    existing[0]["status"] = "repaired"
    current = duplicate_pipeline._build_duplicate_findings(_report(), 1, 2)
    merged = duplicate_pipeline._merge_duplicate_findings(existing, current, exam, 2)
    assert merged[0]["status"] == "unresolved"


def test_unchanged_pair_does_not_accumulate_findings():
    exam = {"questions": {"true_false": [
        {"question_id": "q1", "slot_id": "s1", "statement": "A claim.", "answer": True},
        {"question_id": "q2", "slot_id": "s2", "statement": "A claim.", "answer": True},
    ]}}
    findings = duplicate_pipeline._build_duplicate_findings(_report(), 1, 1)
    findings = duplicate_pipeline._merge_duplicate_findings(
        findings, duplicate_pipeline._build_duplicate_findings(_report(), 1, 2), exam, 2,
    )
    assert len(findings) == 1
    assert len(findings[0]["scan_history"]) == 2


def test_reassessment_corrects_contradictory_target(monkeypatch):
    report = _report()
    report["pair_verdicts"][0]["exact_content"] = False
    report["candidate_pairs"][0]["right"].update({
        "target": "an unrelated target", "task_operation": "explain",
        "question_type": "essay",
    })
    report["duplicate_groups"] = [{
        "group_id": "g1", "question_ids": ["q1", "q2"],
        "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
    }]

    def fake_review(_exam, **kwargs):
        pair_id = next(iter(kwargs["forced_question_pairs"]))
        left, right = sorted(pair_id)
        return {
            "candidate_pairs": report["candidate_pairs"],
            "pair_verdicts": [{
                "pair_id": "::".join((left, right)),
                "same_core_knowledge": "no",
                "same_student_operation_or_method": "no",
                "same_target_or_result": "no",
                "genuinely_new_reasoning": "yes",
                "one_answer_substantially_gives_away_other": "no",
                "classification": "acceptable_reuse",
                "exact_content": False,
            }],
            "fact_cache": {"corrected-key": {"same_core_knowledge": "no"}},
        }

    monkeypatch.setattr(duplicate_pipeline, "review_duplicate_candidates", fake_review)
    monkeypatch.setattr(
        duplicate_pipeline, "build_duplicate_groups",
        lambda candidates, verdicts: [],
    )
    corrected = duplicate_pipeline._reassess_contradictions(
        report, {"questions": {}}, {"s1": {}, "s2": {}}, None,
    )
    assert corrected["reassessment"]["pair_ids"] == ["q1::q2"]
    assert corrected["pair_verdicts"][0]["classification"] == "acceptable_reuse"
    assert corrected["reassessment_fact_cache"] == {"corrected-key": {"same_core_knowledge": "no"}}
    assert corrected["duplicate_groups"] == []
