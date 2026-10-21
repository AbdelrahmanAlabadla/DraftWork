from __future__ import annotations

from copy import deepcopy

from app.online import validator as val
from app.online.graph import _route_after_validation
from app.online.lifecycle import initialize_slot_lifecycle, reconcile_slot_lifecycle


def _tf(slot: str, *, statement: str = "A supported statement."):
    return {
        "slot_id": slot,
        "question_id": slot.replace("m1_", "model1_"),
        "statement": statement,
        "answer": True,
    }


def _plan(slots):
    return [{
        "model_number": 1,
        "items": {"true_false": [{
            "slot_id": slot,
            "question_type": "true_false",
            "task_id": f"task_{index}",
            "source_chunk_id": "chunk_1",
            "difficulty": "easy",
        } for index, slot in enumerate(slots, 1)]},
    }]


def _pass_report(exam, question_ids=None):
    ids = list(question_ids or [
        item["question_id"] for item in exam["questions"].get("true_false", [])
    ])
    return {
        "model_number": 1,
        "all_pass": True,
        "verdicts": [{
            "question_id": qid, "question_valid": True, "answer_valid": True,
            "action": "PASS", "fields_to_fix": [], "failure_code": None,
            "reason": "", "expected_fix": "",
        } for qid in ids],
        "warnings": [], "coverage": [],
    }


def test_full_validation_creates_missing_slot_target_for_fifth_tf(monkeypatch):
    slots = [f"m1_true_false_{index}" for index in range(1, 6)]
    exam = {"model_number": 1, "questions": {"true_false": [_tf(s) for s in slots[:4]]}}
    monkeypatch.setattr(val, "_validate_with_context",
                        lambda exam, ids=None, *_args: _pass_report(exam, ids))
    out = val.validate_generated_questions({
        "generated_exams": [exam], "plans": _plan(slots), "retrieved_chunks": [{"chunk_id": "chunk_1", "text": "source"}],
        "validation_reports": [], "validated_models": [], "question_repair_attempts": {},
        "pending_revalidation_ids": {}, "warnings": [],
    })
    assert out["validation_reports"][0]["missing_slots"][0]["slot_id"] == slots[-1]
    target = out["repair_manifest"][0]
    assert target["slot_id"] == slots[-1]
    assert target["repair_type"] == "regenerate_missing_slot"
    assert _route_after_validation(out) == "repair_invalid_questions"


def test_missing_whole_section_creates_one_target_per_expected_slot(monkeypatch):
    slots = ["m1_true_false_1", "m1_true_false_2"]
    exam = {"model_number": 1, "questions": {}}
    monkeypatch.setattr(val, "_validate_with_context",
                        lambda exam, ids=None, *_args: _pass_report(exam, ids))
    out = val.validate_generated_questions({
        "generated_exams": [exam], "plans": _plan(slots), "retrieved_chunks": [],
        "validation_reports": [], "validated_models": [], "question_repair_attempts": {},
        "pending_revalidation_ids": {}, "warnings": [],
    })
    assert {x["slot_id"] for x in out["repair_manifest"]} == set(slots)
    assert out["validation_reports"][0]["missing_sections"] == ["true_false"]


def test_unsupported_given_routes_to_assignment_replacement():
    assert val._repair_classification("unsupported_given") == (
        "task_or_evidence_invalid", "replace_assignment"
    )
    assert val._repair_classification("invalid_structure") == (
        "structure_invalid", "repair_question"
    )
    assert val._repair_classification(
        "incomplete_solution", validation_role="stem_quantitative"
    ) == ("quantitative_solution_invalid", "repair_question")


def test_repair_budget_is_per_slot(monkeypatch):
    calls = []
    def fake_repair(exam, verdicts, **_kwargs):
        qid = verdicts[0]["question_id"]
        calls.append(qid)
        return {"warnings": [], "sent_ids": [qid], "repaired_ids": [qid], "failed_ids": []}
    monkeypatch.setattr(val, "repair_exam", fake_repair)
    exam = {"model_number": 1, "questions": {"true_false": [
        _tf("m1_true_false_1"), _tf("m1_true_false_2")
    ]}}
    manifest = []
    for index in (1, 2):
        manifest.append({
            "model_number": 1, "slot_id": f"m1_true_false_{index}",
            "question_id": f"model1_true_false_{index}", "question_type": "true_false",
            "status": "invalid", "failure_code": "bad_wording", "failure_reason": "fix it",
            "repair_type": "repair_question", "validation_role": "general",
            "verdict": {"question_id": f"model1_true_false_{index}", "action": "FIX_QUESTION",
                        "fields_to_fix": ["statement"], "reason": "fix it"},
        })
    out = val.repair_invalid_questions({
        "generated_exams": [exam], "plans": [], "repair_manifest": manifest,
        "question_repair_attempts": {"m1_true_false_1": 3}, "warnings": [],
    })
    assert calls == ["model1_true_false_2"]
    assert out["question_repair_attempts"] == {
        "m1_true_false_1": 3, "m1_true_false_2": 1
    }


def test_failed_candidate_stays_outside_active_exam_and_records_new_reason(monkeypatch):
    active = {"model_number": 1, "questions": {"true_false": [_tf("m1_true_false_1", statement="old")]}}
    staged = deepcopy(active)
    staged["questions"]["true_false"][0]["statement"] = "candidate"
    def fail(exam, ids=None, *_args):
        return {"model_number": 1, "all_pass": False, "warnings": [], "coverage": [],
                "verdicts": [{"question_id": "model1_true_false_1", "question_valid": False,
                  "answer_valid": True, "action": "FIX_QUESTION", "fields_to_fix": ["statement"],
                  "failure_code": "invalid_structure", "reason": "new exact reason",
                  "expected_fix": "rewrite structure"}]}
    monkeypatch.setattr(val, "_validate_with_context", fail)
    state = {
        "generated_exams": [active], "plans": _plan(["m1_true_false_1"]),
        "validation_reports": [_pass_report(active)], "validated_models": [],
        "repair_manifest": [{"model_number": 1, "slot_id": "m1_true_false_1",
            "question_id": "model1_true_false_1", "status": "invalid",
            "failure_code": "old_failure", "repair_type": "repair_question"}],
        "repair_candidates": [{"model_number": 1, "slot_id": "m1_true_false_1",
            "question_id": "model1_true_false_1", "question_type": "true_false", "attempt": 1,
            "repair_action": "repair_question", "had_original": True, "exam": staged,
            "plans": _plan(["m1_true_false_1"]), "candidate_task_id": "task_1"}],
        "question_repair_attempts": {"m1_true_false_1": 1}, "pending_revalidation_ids": {}, "warnings": [],
    }
    out = val.validate_generated_questions(state)
    assert out["generated_exams"][0]["questions"]["true_false"][0]["statement"] == "old"
    assert out["repair_manifest"][0]["failure_reason"] == "new exact reason"
    assert out["repair_retry_seeds"]["m1_true_false_1"]["exam"]["questions"]["true_false"][0]["statement"] == "candidate"
    assert "task_1" in out["repair_replacement_memory"]["m1_true_false_1"]["blacklisted_candidate_ids"]


def test_passing_candidate_fills_missing_slot_and_resolves_manifest(monkeypatch):
    slots = ["m1_true_false_1", "m1_true_false_2"]
    active = {"model_number": 1, "questions": {"true_false": [_tf(slots[0])]}}
    staged = deepcopy(active)
    staged["questions"]["true_false"].append(_tf(slots[1]))
    monkeypatch.setattr(val, "_validate_with_context",
                        lambda exam, ids=None, *_args: _pass_report(exam, ids))
    out = val.validate_generated_questions({
        "generated_exams": [active], "plans": _plan(slots), "validation_reports": [],
        "validated_models": [], "repair_manifest": [{"model_number": 1, "slot_id": slots[1],
          "question_id": None, "status": "missing", "failure_code": "missing_requested_slot",
          "repair_type": "regenerate_missing_slot"}],
        "repair_candidates": [{"model_number": 1, "slot_id": slots[1],
          "question_id": "model1_true_false_2", "question_type": "true_false", "attempt": 1,
          "repair_action": "regenerate_missing_slot", "had_original": False, "exam": staged,
          "plans": _plan(slots), "candidate_task_id": "task_2"}],
        "question_repair_attempts": {slots[1]: 1}, "pending_revalidation_ids": {}, "warnings": [],
    })
    assert [x["slot_id"] for x in out["generated_exams"][0]["questions"]["true_false"]] == slots
    assert out["repair_manifest"] == []
    assert out["validation_reports"][0]["all_pass"]


def test_selective_manifest_merge_does_not_erase_other_unresolved_slots():
    old = [{"model_number": 1, "slot_id": "m1_true_false_5", "status": "missing",
            "failure_code": "missing_requested_slot"},
           {"model_number": 1, "slot_id": "m1_mcq_3", "status": "invalid",
            "failure_code": "bad_answer"}]
    fresh = [{"model_number": 1, "slot_id": "m1_mcq_3", "status": "invalid",
              "failure_code": "invalid_structure"}]
    merged = val._merge_repair_manifest(old, fresh)
    assert {item["slot_id"] for item in merged} == {"m1_true_false_5", "m1_mcq_3"}


def test_resolved_manifest_supersedes_stale_invalid_lifecycle_record():
    state = {
        "plans": _plan(["m1_true_false_1"]),
        "generated_exams": [{"model_number": 1, "questions": {"true_false": [_tf("m1_true_false_1")]}}],
        "validation_reports": [{"model_number": 1, "verdicts": [{
            "question_id": "model1_true_false_1", "action": "PASS", "fields_to_fix": []}]}],
        "repair_manifest": [{"slot_id": "m1_true_false_1", "status": "resolved"}],
        "invalid_questions": [{"slot_id": "m1_true_false_1", "question_id": "model1_true_false_1",
                               "failure_code": "old_failure", "failure_stage": "validation"}],
    }
    lifecycle = reconcile_slot_lifecycle(
        initialize_slot_lifecycle([("true_false", 1)], 1), state, stage="finalization"
    )
    assert lifecycle["m1_true_false_1"]["validated"] is True
    assert lifecycle["m1_true_false_1"]["failure_code"] is None
    assert lifecycle["m1_true_false_1"]["terminal_status"] == "finalized"
