from __future__ import annotations

import json
import threading

from app.online.multi_validator import clear_job_validation_cache, protect_question, validate_exam_multi
from app.online.validator import repair_exam


def _plan(slot_id: str, *, quantitative: bool = False) -> dict:
    return {
        "slot_id": slot_id,
        "source_chunk_id": f"chunk-{slot_id}",
        "support_chunk_ids": [],
        "task_id": f"task-{slot_id}",
        "concept": "gravity",
        "knowledge_focus": "gravity attracts masses",
        "idea_to_test": "explain gravity",
        "target": "state the cause",
        "task_operation": "explain_cause",
        "required_givens": [],
        "given_conditions": [],
        "difficulty": "medium",
        "reasoning_mode": "conceptual_reasoning",
        "requires_quantitative_validation": quantitative,
    }


def _definition(index: int) -> dict:
    return {
        "question_id": f"q{index}",
        "slot_id": f"s{index}",
        "term": f"Gravity {index}",
        "reference_answer": "The attraction between masses.",
    }


def _exam(count: int = 1) -> dict:
    return {"model_number": 1, "questions": {"definition": [_definition(i) for i in range(count)]}}


def _contexts(count: int = 1):
    plans = {f"s{i}": _plan(f"s{i}") for i in range(count)}
    sources = {f"chunk-s{i}": "Gravity is the attraction between masses." for i in range(count)}
    return plans, sources


class PassingClient:
    provider_name = "offline"
    model = "fake"

    def __init__(self, calls: list[dict], responder=None):
        self.calls = calls
        self.responder = responder

    def chat_structured(self, prompt, *, schema_name, **kwargs):
        payload = json.loads(prompt.split("<VALIDATION_DATA>\n", 1)[1].split("\n</VALIDATION_DATA>", 1)[0])
        with _CALL_LOCK:
            self.calls.append(payload)
            call_number = len(self.calls)
        if self.responder:
            response = self.responder(payload, call_number)
            if response is not None:
                return response
        role = payload["validator"]
        return {"validator": role, "verdicts": [
            {"question_id": item["question_id"], "status": "PASS", "issues": [],
             "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
             "observed_difficulty": "medium" if role == "difficulty" else None,
             "calculation_evidence": []}
            for item in payload["questions"]
        ]}


_CALL_LOCK = threading.Lock()


def test_batches_each_question_type_at_four_and_runs_all_roles():
    calls: list[dict] = []
    plans, sources = _contexts(5)
    result = validate_exam_multi(
        _exam(5), plan_by_slot=plans, source_by_chunk=sources, domain="NON_STEM",
        client_factory=lambda _role: PassingClient(calls),
    )

    assert result["all_pass"] is True
    assert len(calls) == 8  # four roles, each split into batches of 4 + 1
    assert max(len(call["questions"]) for call in calls) == 4
    assert {call["validator"] for call in calls} == {
        "grounding", "task_alignment", "answer_correctness", "difficulty",
    }


def test_invalid_row_retries_only_that_question_and_preserves_valid_rows(monkeypatch):
    monkeypatch.setattr("app.online.multi_validator.time.sleep", lambda _seconds: None)
    calls: list[dict] = []

    def responder(payload, _number):
        if payload["validator"] != "answer_correctness":
            return None
        ids = [item["question_id"] for item in payload["questions"]]
        if ids == ["q0", "q1"]:
            return {"validator": "answer_correctness", "verdicts": [
                {"question_id": "q0", "status": "PASS", "issues": [],
                 "grounding_outcome": None, "observed_difficulty": None, "calculation_evidence": []},
                {"question_id": "q1", "status": "BROKEN", "issues": []},
            ]}
        return None

    plans, sources = _contexts(2)
    result = validate_exam_multi(
        _exam(2), plan_by_slot=plans, source_by_chunk=sources, domain="NON_STEM",
        client_factory=lambda _role: PassingClient(calls, responder),
    )

    correctness_calls = [call for call in calls if call["validator"] == "answer_correctness"]
    assert [[item["question_id"] for item in call["questions"]] for call in correctness_calls] == [
        ["q0", "q1"], ["q1"],
    ]
    assert result["all_pass"] is True


def test_provider_errors_block_validation_without_content_repair(monkeypatch):
    monkeypatch.setattr("app.online.multi_validator.time.sleep", lambda _seconds: None)

    class BrokenClient(PassingClient):
        def chat_structured(self, *args, **kwargs):
            raise TimeoutError("offline timeout")

    plans, sources = _contexts()
    result = validate_exam_multi(
        _exam(), plan_by_slot=plans, source_by_chunk=sources, domain="NON_STEM",
        client_factory=lambda _role: BrokenClient([]),
    )
    verdict = result["verdicts"][0]
    assert verdict["validation_status"] == "VALIDATION_BLOCKED"
    assert verdict["action"] == "UNVALIDATED"
    assert verdict["fields_to_fix"] == []
    assert verdict["block_category"] == "operational"
    assert verdict["blocked_roles"]


def test_insufficient_evidence_is_content_failure_not_validation_block(monkeypatch):
    monkeypatch.setattr("app.online.multi_validator.time.sleep", lambda _seconds: None)
    calls: list[dict] = []

    def responder(payload, _number):
        if payload["validator"] != "grounding":
            return None
        return {"validator": "grounding", "verdicts": [{
            "question_id": "q0", "status": "FAIL", "issues": [{
                "failure_code": "insufficient_evidence",
                "fields_to_fix": ["term"],
                "reason": "The assigned source does not support this definition.",
                "expected_fix": "Use a source-supported definition or assignment.",
            }],
            "grounding_outcome": "INSUFFICIENT_EVIDENCE",
            "observed_difficulty": None, "calculation_evidence": [],
        }]}

    plans, sources = _contexts()
    result = validate_exam_multi(
        _exam(), plan_by_slot=plans, source_by_chunk=sources, domain="NON_STEM",
        client_factory=lambda _role: PassingClient(calls, responder),
    )

    verdict = result["verdicts"][0]
    assert verdict["validation_status"] == "FAIL"
    assert verdict["action"] != "UNVALIDATED"
    assert verdict["required_fixes"][0]["failure_code"] == "insufficient_evidence"


def test_missing_assigned_evidence_is_prerequisite_block():
    plans, _sources = _contexts()
    result = validate_exam_multi(
        _exam(), plan_by_slot=plans, source_by_chunk={}, domain="NON_STEM",
        client_factory=lambda _role: PassingClient([]),
    )
    verdict = result["verdicts"][0]
    assert verdict["validation_status"] == "VALIDATION_BLOCKED"
    assert verdict["block_category"] == "prerequisite"
    assert verdict["failure_code"] == "validation_prerequisite_unavailable"


def test_revalidation_requires_exact_original_issue_coverage(monkeypatch):
    monkeypatch.setattr("app.online.multi_validator.time.sleep", lambda _seconds: None)
    calls: list[dict] = []
    issue = {"issue_id": "q0-I-1", "validator": "answer_correctness",
             "failure_code": "incorrect_answer", "fields_to_fix": ["reference_answer"],
             "reason": "wrong", "expected_fix": "correct it"}

    def responder(payload, _number):
        role = payload["validator"]
        checks = ([{"issue_id": "q0-I-1", "resolved": True,
                    "reason": "The corrected definition is accurate.", "remaining_defect": ""}]
                  if role == "answer_correctness" else [])
        return {"validator": role, "verdicts": [{
            "question_id": "q0", "status": "PASS", "issues": [],
            "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
            "observed_difficulty": "medium" if role == "difficulty" else None,
            "calculation_evidence": [], "original_issue_checks": checks,
        }]}

    plans, sources = _contexts()
    result = validate_exam_multi(
        _exam(), plan_by_slot=plans, source_by_chunk=sources, domain="NON_STEM",
        client_factory=lambda _role: PassingClient(calls, responder),
        revalidation_context={"q0": {"original_question": _definition(0),
            "original_failure_findings": [issue], "repair_metadata": {}}},
    )
    assert result["all_pass"] is True
    check = result["role_results"]["q0"]["answer_correctness"]["original_issue_checks"]
    assert check[0]["issue_id"] == "q0-I-1"


def test_job_local_dependency_cache_reuses_only_eligible_passes():
    scope = "cache-test-job"
    clear_job_validation_cache(scope)
    calls: list[dict] = []
    exam = {"model_number": 1, "questions": {"mcq": [{
        "question_id": "q0", "slot_id": "s0", "question": "2 + 2?",
        "options": {"A": "4", "B": "3", "C": "5", "D": "6"},
        "correct_answer": "B",
    }]}}
    plans, sources = _contexts()
    validate_exam_multi(
        exam, plan_by_slot=plans, source_by_chunk=sources, domain="NON_STEM",
        client_factory=lambda _role: PassingClient(calls), cache_scope=scope,
    )
    initial = len(calls)
    repaired = json.loads(json.dumps(exam))
    repaired["questions"]["mcq"][0]["correct_answer"] = "A"
    issue = {"issue_id": "q0-I-1", "validator": "answer_correctness",
             "failure_code": "incorrect_answer", "fields_to_fix": ["correct_answer"],
             "reason": "wrong key", "expected_fix": "select A"}

    def responder(payload, _number):
        role = payload["validator"]
        return {"validator": role, "verdicts": [{
            "question_id": "q0", "status": "PASS", "issues": [],
            "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
            "observed_difficulty": "medium" if role == "difficulty" else None,
            "calculation_evidence": [],
            "original_issue_checks": ([{"issue_id": "q0-I-1", "resolved": True,
                "reason": "A now keys 4.", "remaining_defect": ""}]
                if role == "answer_correctness" else []),
        }]}

    validate_exam_multi(
        repaired, plan_by_slot=plans, source_by_chunk=sources, domain="NON_STEM",
        client_factory=lambda _role: PassingClient(calls, responder), cache_scope=scope,
        revalidation_context={"q0": {"original_question": exam["questions"]["mcq"][0],
            "original_failure_findings": [issue], "repair_metadata": {}}},
    )
    rerun_roles = [call["validator"] for call in calls[initial:]]
    assert set(rerun_roles) == {"grounding", "answer_correctness"}
    clear_job_validation_cache(scope)


def test_grounding_payload_enforces_per_question_evidence_boundaries():
    calls: list[dict] = []
    plans, sources = _contexts(2)
    validate_exam_multi(
        _exam(2), plan_by_slot=plans, source_by_chunk=sources, domain="NON_STEM",
        client_factory=lambda _role: PassingClient(calls),
    )
    grounding = next(call for call in calls if call["validator"] == "grounding")
    for item in grounding["questions"]:
        assert [chunk["chunk_id"] for chunk in item["assigned_child_chunks"]] == [
            f"chunk-s{item['question_id'][1:]}"
        ]
        assert "task_contract" not in item

    claimed = _definition(0) | {"effective_evidence_chunk_ids": ["chunk-s1"]}
    blocked = protect_question("definition", claimed, plans["s0"], sources)
    assert any("unassigned child chunk" in error for error in blocked.errors)


def test_input_protection_sanitizes_nul_and_preserves_arabic_unicode():
    item = {
        "question_id": "q-ar", "slot_id": "s-ar", "statement": "الجاذبية\x00 قوة؟",
        "answer": True,
    }
    task = _plan("s-ar")
    protected = protect_question(
        "true_false", item, task,
        {"chunk-s-ar": "الجاذبية قوة جذب بين الكتل ∑."},
    )
    assert not protected.errors
    assert "\x00" not in protected.question["statement"]
    assert "الجاذبية" in protected.question["statement"]
    assert "∑" in protected.evidence["chunk-s-ar"]
    assert protected.sanitization_events

    invalid = protect_question(
        "true_false", {**item, "statement": "broken \ud800 text"}, task,
        {"chunk-s-ar": "valid evidence"},
    )
    assert "invalid Unicode could not be preserved safely" in invalid.errors


def test_explicit_quantitative_metadata_routes_true_false_to_fifth_agent():
    calls: list[dict] = []
    exam = {"model_number": 1, "questions": {"true_false": [{
        "question_id": "q-tf", "slot_id": "s-tf", "statement": "The force is 15 N.",
        "answer": True,
    }]}}
    plan = _plan("s-tf", quantitative=True)
    validate_exam_multi(
        exam, plan_by_slot={"s-tf": plan},
        source_by_chunk={"chunk-s-tf": "The calculated force is 15 N."},
        domain="STEM", client_factory=lambda _role: PassingClient(calls),
    )
    assert "quantitative" in {call["validator"] for call in calls}


def test_findings_from_multiple_agents_are_merged_without_loss():
    calls: list[dict] = []

    def responder(payload, _number):
        role = payload["validator"]
        if role not in {"task_alignment", "answer_correctness"}:
            return None
        issue = (
            {"failure_code": "wrong_givens", "fields_to_fix": ["term"],
             "reason": "The requested given is absent.", "expected_fix": "Add the required given."}
            if role == "task_alignment" else
            {"failure_code": "incorrect_answer", "fields_to_fix": ["reference_answer"],
             "reason": "The answer is incorrect.", "expected_fix": "Correct the answer."}
        )
        return {"validator": role, "verdicts": [{
            "question_id": "q0", "status": "FAIL", "issues": [issue],
            "grounding_outcome": None, "observed_difficulty": None, "calculation_evidence": [],
        }]}

    plans, sources = _contexts()
    verdict = validate_exam_multi(
        _exam(), plan_by_slot=plans, source_by_chunk=sources, domain="NON_STEM",
        client_factory=lambda _role: PassingClient(calls, responder),
    )["verdicts"][0]
    assert verdict["action"] == "FIX_QUESTION_AND_ANSWER"
    assert verdict["fields_to_fix"] == ["reference_answer", "term"]
    assert {finding["validator"] for finding in verdict["required_fixes"]} == {
        "task_alignment", "answer_correctness",
    }


def test_certificate_findings_are_precisely_targeted_and_not_false_conflicts():
    calls: list[dict] = []
    exam = {"model_number": 1, "questions": {"mcq": [{
        "question_id": "q-cert", "slot_id": "s-cert",
        "question": "What is x?",
        "options": {"A": "2", "B": "4", "C": "-10", "D": "8"},
        "correct_answer": "C",
        "solution_certificate": {
            "matching_correct_option": "C",
            "verification_method": "The answer is option A.",
        },
    }]}}
    plan = _plan("s-cert", quantitative=True)

    def responder(payload, _number):
        role = payload["validator"]
        if role not in {"answer_correctness", "quantitative"}:
            return None
        return {"validator": role, "verdicts": [{
            "question_id": "q-cert", "status": "FAIL", "issues": [{
                "failure_code": (
                    "answer_inconsistent" if role == "answer_correctness"
                    else "incorrect_option_mapping"
                ),
                "fields_to_fix": ["question"],
                "reason": "The solution_certificate verification_method names stale option A.",
                "expected_fix": "Update the certificate to identify option C.",
            }],
            "grounding_outcome": None, "observed_difficulty": None,
            "calculation_evidence": [],
        }]}

    verdict = validate_exam_multi(
        exam, plan_by_slot={"s-cert": plan},
        source_by_chunk={"chunk-s-cert": "Solve for x."}, domain="STEM",
        client_factory=lambda _role: PassingClient(calls, responder),
    )["verdicts"][0]

    assert verdict["fields_to_fix"] == ["solution_certificate"]
    assert verdict["repair_instruction_conflicts"] == []
    assert {tuple(issue["fields_to_fix"]) for issue in verdict["required_fixes"]} == {
        ("solution_certificate",),
    }


def test_repair_metadata_is_verified_against_actual_object_diff(monkeypatch):
    exam = {"model_number": 1, "questions": {"definition": [{
        "question_id": "q1", "slot_id": "s1", "term": "gravity",
        "reference_answer": "Wrong answer.",
    }]}}
    verdict = {
        "question_id": "q1", "action": "FIX_ANSWER", "validation_status": "FAIL",
        "failure_code": "incorrect_answer", "fields_to_fix": ["reference_answer"],
        "reason": "incorrect", "expected_fix": "correct it",
        "required_fixes": [{"failure_code": "incorrect_answer"}],
    }

    def fake_request(*_args, **_kwargs):
        return {"repairs": [{
            "question_id": "q1", "patch": {"reference_answer": "Attraction between masses."},
            "repair_metadata": {
                "modified_fields": ["reference_answer"],
                "changes": [{
                    "field": "reference_answer", "previous_value": "Wrong answer.",
                    "new_value": "Attraction between masses.", "reason": "Corrects the definition.",
                }],
                "targeted_failure_codes": ["incorrect_answer"],
            },
        }]}

    monkeypatch.setattr("app.online.validator.request_structured", fake_request)
    result = repair_exam(exam, [verdict], client=object())
    assert result["repaired_ids"] == ["q1"]
    assert result["repair_metadata"]["q1"]["modified_fields"] == ["reference_answer"]
    assert exam["questions"]["definition"][0]["reference_answer"] == "Attraction between masses."


def test_validator_preserves_precise_mcq_option_path():
    from app.online.multi_validator import ProtectedQuestion, _compatibility_verdict

    item = ProtectedQuestion(
        question_id="q1", question_type="mcq", slot_id="s1", question={}, task={},
        evidence={}, word_bank=[], sanitization_events=[], errors=[],
        requires_quantitative=True,
    )
    findings = {role: {"status": "PASS", "issues": []} for role in (
        "grounding", "task_alignment", "answer_correctness", "difficulty", "quantitative",
    )}
    findings["answer_correctness"] = {"status": "FAIL", "issues": [{
        "failure_code": "ambiguous_option", "fields_to_fix": ["options.D"],
        "reason": "D duplicates B.", "expected_fix": "Replace D only.",
    }]}
    verdict = _compatibility_verdict(item, findings, None)
    assert verdict["fields_to_fix"] == ["options.D"]
    assert verdict["required_fixes"][0]["fields_to_fix"] == ["options.D"]


def test_graphing_policy_does_not_require_embedded_answer_image():
    from app.online.multi_validator import _system_prompt

    for role in ("task_alignment", "answer_correctness", "quantitative"):
        prompt = _system_prompt(role)
        assert "does not need an embedded graph image" in prompt
        assert "feasible-region/shading guidance" in prompt
