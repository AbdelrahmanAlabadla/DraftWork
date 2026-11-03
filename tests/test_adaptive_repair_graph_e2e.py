"""Mocked production-node verification for bounded adaptive repair recovery."""
from __future__ import annotations

from copy import deepcopy
import json

from app.online import graph, multi_repair, multi_validator
from app.online import duplicate_pipeline
from app.online.exam_builder import assemble_exams_node
from app.online.lifecycle import initialize_slot_lifecycle, reconcile_slot_lifecycle
from app.online.validator import repair_invalid_questions, validate_generated_questions


def _certificate(option: str, text: str) -> dict:
    return {
        "method": "substitution",
        "important_givens": ["g(x)=3x-4", "x=-2"],
        "calculation_steps": ["g(-2) = 3(-2) - 4 = -10"],
        "final_answer": "-10",
        "units": "",
        "matching_correct_option": option,
        "verification_method": text,
    }


def _mcq(question_id: str = "model1_mcq_4", slot_id: str = "m1_mcq_4") -> dict:
    return {
        "question_id": question_id, "slot_id": slot_id,
        "question": "g(x)=3x-4. Find g(-2).",
        "options": {"A": "2", "B": "-2", "C": "-10", "D": "-6"},
        "correct_answer": "C",
        "solution_certificate": _certificate("A", "-10 matches option A."),
    }


def _plan(slot_id: str, qtype: str = "mcq") -> dict:
    return {
        "slot_id": slot_id, "question_type": qtype, "task_id": f"task-{slot_id}",
        "selected_variation_id": f"variation-{slot_id}", "source_chunk_id": "c1",
        "support_chunk_ids": [], "difficulty": "medium", "task_operation": "evaluate",
        "target": "result", "required_givens": [],
    }


def _issue(code: str, fields: list[str], reason: str, expected: str) -> dict:
    return {
        "validator": "answer_correctness", "validation_status": "FAIL",
        "failure_code": code, "fields_to_fix": fields, "reason": reason,
        "expected_fix": expected, "issue_id": f"I-{code}",
    }


def _state(question: dict, verdict: dict, *, attempts: int = 0) -> dict:
    slot_id, qid = question["slot_id"], question["question_id"]
    return {
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [deepcopy(question)]}}],
        "plans": [{"model_number": 1, "items": {"mcq": [_plan(slot_id)]}}],
        "retrieved_chunks": [{"child_id": "c1", "content": "Evaluate g(x)=3x-4."}],
        "assessment_profile": {"domain": "STEM"}, "difficulty": "medium",
        "repair_manifest": [{
            "model_number": 1, "slot_id": slot_id, "question_id": qid,
            "question_type": "mcq", "status": "invalid", "repair_type": "repair_question",
            "failure_code": verdict["failure_code"], "failure_reason": verdict["reason"],
            "fields_to_fix": list(verdict["fields_to_fix"]), "verdict": deepcopy(verdict),
        }],
        "question_repair_attempts": ({slot_id: attempts} if attempts else {}),
        "repair_replacement_memory": {}, "repair_retry_seeds": {},
        "repair_candidates": [], "validation_reports": [], "validation_ledger": {},
        "validated_models": [], "pending_revalidation_ids": {}, "warnings": [],
        "validation_repair_history": [], "repair_call_logs": [],
        "slot_lifecycle": initialize_slot_lifecycle([("mcq", 1)], 1),
        "duplicate_reports": [], "duplicate_scan_count": 0, "duplicate_fact_cache": {},
        "duplicate_screen_fingerprints": {}, "unresolved_duplicate_conflicts": [],
    }


def _pass_report(qid: str) -> dict:
    return {
        "model_number": 1, "all_pass": True,
        "verdicts": [{"question_id": qid, "action": "PASS", "validation_status": "PASS",
                      "failure_code": "pass", "fields_to_fix": [], "required_fixes": []}],
        "role_results": {}, "role_fingerprints": {}, "validation_call_logs": [],
    }


def test_mcq_conflict_reconciles_then_revalidates_with_production_nodes(monkeypatch):
    question = _mcq()
    answer_issue = _issue("answer_inconsistent", ["solution_certificate"],
                          "Certificate says option A.", "Use option C.")
    quantity_issue = {**_issue("incorrect_option_mapping", ["solution_certificate"],
                               "Certificate mapping is stale.", "Use option C."),
                      "validator": "quantitative", "issue_id": "I-map"}
    verdict = {
        "question_id": question["question_id"], "action": "FIX_ANSWER",
        "validation_status": "FAIL", "failure_code": "answer_inconsistent",
        "fields_to_fix": ["solution_certificate"], "reason": answer_issue["reason"],
        "required_fixes": [answer_issue, quantity_issue],
        "repair_instruction_conflicts": [{"field": "solution_certificate",
                                           "instructions": ["Use A", "Use C"]}],
    }
    state = _state(question, verdict)
    provider_calls: list[str] = []

    def fake_structured(_client, prompt, **_kwargs):
        provider_calls.append(prompt)
        payload = json.loads(prompt.split("<REPAIR_DATA>\n", 1)[1].split("\n</REPAIR_DATA>", 1)[0])
        current = payload["questions"][0]["current_question"]
        repaired = _certificate("C", "-10 matches option C.")
        return {"repairs": [{
            "question_id": question["question_id"],
            "patch": {"solution_certificate": repaired},
            "repair_metadata": {
                "modified_fields": ["solution_certificate"],
                "changes": [{"field": "solution_certificate",
                             "previous_value": json.dumps(current["solution_certificate"], ensure_ascii=False, sort_keys=True),
                             "new_value": json.dumps(repaired, ensure_ascii=False, sort_keys=True),
                             "reason": "Synchronizes the certificate mapping."}],
                "targeted_failure_codes": ["answer_inconsistent", "incorrect_option_mapping"],
            },
        }]}

    monkeypatch.setattr(multi_repair, "request_structured", fake_structured)
    first = repair_invalid_questions(state)
    assert first["question_repair_attempts"] == {"m1_mcq_4": 1}
    assert first["repair_manifest"][0]["status"] == "candidate_generated"
    assert len(provider_calls) == 1
    assert first["repair_candidates"][0]["repair_strategy"] == "targeted_repair"

    monkeypatch.setattr(multi_validator, "validate_exam_multi", lambda *_a, **_k: _pass_report(question["question_id"]))
    final = validate_generated_questions({**state, **first})
    item = final["generated_exams"][0]["questions"]["mcq"][0]
    assert item["correct_answer"] == "C"
    assert item["options"] == question["options"]
    assert item["solution_certificate"]["matching_correct_option"] == "C"
    assert "option C" in item["solution_certificate"]["verification_method"]
    assert first["question_repair_attempts"] == {"m1_mcq_4": 1}


def test_tf5_descriptive_metadata_stages_atomically_and_reaches_revalidation(monkeypatch):
    slot_id, qid = "m1_true_false_5", "model1_true_false_5"
    old_certificate = {
        "method": "old", "important_givens": ["(1,2,3)"],
        "calculation_steps": ["old"], "final_answer": "True", "units": "",
        "matching_correct_option": None, "verification_method": "old tuple",
    }
    question = {
        "question_id": qid, "slot_id": slot_id,
        "statement": "The tuple (1,2,3) satisfies the displayed system.",
        "answer": True, "solution_certificate": old_certificate,
    }
    issue = {
        "validator": "grounding", "validation_status": "FAIL",
        "failure_code": "unsupported", "fields_to_fix": ["statement"],
        "reason": "Use the source-backed tuple and equations.",
        "expected_fix": "Synchronize statement, answer and certificate.",
        "issue_id": "I-tf5-grounding",
    }
    verdict = {
        "question_id": qid, "action": "FIX_QUESTION", "validation_status": "FAIL",
        "failure_code": "unsupported", "fields_to_fix": ["statement"],
        "reason": issue["reason"], "required_fixes": [issue],
        "repair_instruction_conflicts": [],
    }
    state = {
        "job_id": "tf5-metadata-regression",
        "generated_exams": [{"model_number": 1, "questions": {"true_false": [deepcopy(question)]}}],
        "plans": [{"model_number": 1, "items": {"true_false": [_plan(slot_id, "true_false")]}}],
        "retrieved_chunks": [{"child_id": "c1", "content": "Use (-1,-5,7) in the displayed system."}],
        "assessment_profile": {"domain": "STEM"}, "difficulty": "easy",
        "repair_manifest": [{
            "model_number": 1, "slot_id": slot_id, "question_id": qid,
            "question_type": "true_false", "status": "invalid",
            "repair_type": "repair_question", "failure_code": "unsupported",
            "failure_reason": issue["reason"], "fields_to_fix": ["statement"],
            "verdict": verdict,
        }],
        "question_repair_attempts": {}, "repair_replacement_memory": {},
        "repair_retry_seeds": {}, "repair_candidates": [], "validation_reports": [],
        "validation_ledger": {}, "validated_models": [], "pending_revalidation_ids": {},
        "warnings": [], "validation_repair_history": [], "repair_call_logs": [],
        "slot_lifecycle": initialize_slot_lifecycle([("true_false", 1)], 1),
    }
    new_certificate = {
        "method": "substitution", "important_givens": ["(-1,-5,7)"],
        "calculation_steps": ["first equation fails"], "final_answer": "False", "units": "",
        "matching_correct_option": None,
        "verification_method": "substitute (-1,-5,7) into all displayed equations",
    }

    monkeypatch.setattr(multi_repair, "request_structured", lambda *_a, **_k: {
        "repairs": [{
            "question_id": qid,
            "patch": {
                "statement": "The tuple (-1,-5,7) satisfies this displayed system: ...",
                "answer": False, "solution_certificate": new_certificate,
            },
            "repair_metadata": {
                "modified_fields": ["statement", "answer", "solution_certificate"],
                "changes": [
                    {"field": "statement", "previous_value": "old statement", "new_value": "new", "reason": "grounding"},
                    {"field": "answer", "previous_value": "true", "new_value": "false", "reason": "consistency"},
                    {"field": "solution_certificate", "previous_value": "old certificate description", "new_value": "new certificate description", "reason": "consistency"},
                ],
                "targeted_failure_codes": ["unsupported"],
            },
        }]
    })
    staged = repair_invalid_questions(state)
    assert len(staged["repair_candidates"]) == 1
    called = []

    def pass_revalidation(*_args, **_kwargs):
        called.append(True)
        return _pass_report(qid)

    monkeypatch.setattr(multi_validator, "validate_exam_multi", pass_revalidation)
    completed = validate_generated_questions({**state, **staged})
    final = completed["generated_exams"][0]["questions"]["true_false"][0]
    assert called
    assert final["statement"].startswith("The tuple (-1,-5,7)")
    assert final["answer"] is False
    assert final["solution_certificate"] == new_certificate


def test_restart_uses_different_strategy_after_identical_candidate(monkeypatch):
    question = _mcq("q1", "m1_mcq_1")
    verdict = _issue("incorrect_answer", ["solution_certificate"], "Certificate is wrong.", "Fix it.")
    state = _state(question, verdict, attempts=1)
    state["repair_replacement_memory"] = {"m1_mcq_1": {"force_alternative_strategy": True}}
    calls: list[dict] = []

    def regenerate(_state, staged_exam, _mn, qid, plan_by_slot, _sources):
        calls.append(deepcopy(plan_by_slot["m1_mcq_1"]))
        item = staged_exam["questions"]["mcq"][0]
        item["question"] = "g(x)=2x-1. Find g(3)."
        item["options"] = {"A": "5", "B": "4", "C": "6", "D": "7"}
        item["correct_answer"] = "A"
        item["solution_certificate"] = _certificate("A", "5 matches option A.")
        return True, []

    monkeypatch.setattr("app.online.validator._regenerate_plan_slot", regenerate)
    resumed = repair_invalid_questions(deepcopy(state))
    candidate = resumed["repair_candidates"][0]
    assert candidate["repair_strategy"] == "regenerate_original_assignment"
    assert candidate["attempt"] == 2
    assert calls[0]["task_id"] == "task-m1_mcq_1"
    assert calls[0]["source_chunk_id"] == "c1"

    monkeypatch.setattr(multi_validator, "validate_exam_multi", lambda *_a, **_k: _pass_report("q1"))
    accepted = validate_generated_questions({**state, **resumed})
    assert accepted["generated_exams"][0]["questions"]["mcq"][0]["question"] == "g(x)=2x-1. Find g(3)."
    assert resumed["question_repair_attempts"] == {"m1_mcq_1": 2}


def test_terminal_slot_is_excluded_and_partial_assembly_keeps_three_questions(monkeypatch):
    questions = [_mcq(f"q{index}", f"m1_mcq_{index}") for index in range(1, 5)]
    plans = [{"model_number": 1, "items": {"mcq": [_plan(q["slot_id"]) for q in questions]}}]
    lifecycle = initialize_slot_lifecycle([("mcq", 4)], 1)
    for question in questions[:3]:
        lifecycle[question["slot_id"]].update({"generated": True, "validated": True,
                                                  "finalized": True, "terminal_status": "finalized"})
    lifecycle["m1_mcq_4"].update({"generated": True, "validated": False,
                                    "terminal_status": "repair_failed", "current_status": "FAILED_TO_REPAIR"})
    state = {
        "generated_exams": [{"model_number": 1, "questions": {"mcq": questions}}],
        "plans": plans, "slot_lifecycle": lifecycle,
        "validation_reports": [{"model_number": 1, "verdicts": [
            {"question_id": q["question_id"], "action": "PASS" if index < 4 else "FIX_ANSWER"}
            for index, q in enumerate(questions, start=1)
        ]}],
        "repair_manifest": [{"model_number": 1, "slot_id": "m1_mcq_4",
                             "status": "terminal_failure", "failure_code": "content_repair_budget_exhausted"}],
        "question_repair_attempts": {"m1_mcq_4": 3}, "invalid_questions": [],
        "validation_repair_history": [], "warnings": [],
    }
    assert graph._route_after_validation(state) == "assemble_exams"
    assembled = assemble_exams_node(state)
    public = assembled["generated_exams"][0]["questions"]["mcq"]
    assert [item["slot_id"] for item in public] == ["m1_mcq_1", "m1_mcq_2", "m1_mcq_3"]
    assert assembled["accepted_slot_ids"] == ["m1_mcq_1", "m1_mcq_2", "m1_mcq_3"]


def test_failed_content_recovery_moves_to_separate_completion_budget(monkeypatch):
    question = _mcq("q-fail", "m1_mcq_1")
    state = _state(question, _issue("incorrect_answer", ["solution_certificate"], "wrong", "fix"))
    provider_calls: list[str] = []
    monkeypatch.setattr(multi_repair, "repair_question_specialized", lambda **_kwargs: (
        provider_calls.append("targeted") or {
            "sent_ids": ["q-fail"], "repaired_ids": [], "failed_ids": ["q-fail"],
            "warnings": ["no candidate"], "repair_route": ["solution"],
            "repair_traces": [], "route_status": "READY",
        }
    ))
    monkeypatch.setattr("app.online.validator._regenerate_plan_slot", lambda *_a, **_k: (False, ["regeneration failed"]))
    monkeypatch.setattr(duplicate_pipeline, "_candidate_pool", lambda *_a, **_k: [])

    first = repair_invalid_questions(state)
    second = repair_invalid_questions({**state, **first})
    third = repair_invalid_questions({**state, **first, **second})
    resumed = repair_invalid_questions({**state, **first, **second, **third})

    assert first["question_repair_attempts"] == {"m1_mcq_1": 1}
    assert second["question_repair_attempts"] == {"m1_mcq_1": 2}
    assert third["question_repair_attempts"] == {"m1_mcq_1": 2}
    assert third["slot_completion_attempts"] == {"m1_mcq_1": 1}
    assert resumed["slot_completion_attempts"] == {"m1_mcq_1": 2}
    assert provider_calls == ["targeted"]
    assert graph._route_after_validation({**state, **third}) == "repair_invalid_questions"


def test_operational_failure_does_not_spend_content_attempt(monkeypatch):
    question = _mcq("q-op", "m1_mcq_1")
    state = _state(question, _issue("incorrect_answer", ["solution_certificate"], "wrong", "fix"))
    monkeypatch.setattr(multi_repair, "repair_question_specialized", lambda **_kwargs: {
        "sent_ids": ["q-op"], "repaired_ids": [], "failed_ids": ["q-op"],
        "warnings": ["provider timeout"], "repair_route": ["solution"],
        "repair_traces": [], "route_status": "READY", "operational_error": True,
    })
    result = repair_invalid_questions(state)
    assert result["question_repair_attempts"] == {}
    assert result["repair_manifest"][0]["status"] == "operational_retry_pending"
