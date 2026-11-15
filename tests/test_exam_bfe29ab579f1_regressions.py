"""Regression coverage for the exam_bfe29ab579f1 failure classes.

The exam produced eight first-pass failures, 54 revalidation calls, 40 duplicate
validator calls, and a false ``contradicted`` grounding failure on ``m1_mcq_1``.
These offline tests pin the contract that prevents those classes:

* ordinary Catalog/source numbers are illustrative, not immutable;
* real operation/answer/constraint/immutable-data defects still fail;
* unchanged questions and unchanged duplicate pairs are not re-judged;
* a grounding rejection must name a concrete claim and cite assigned evidence.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from app.online import duplicate_judge, multi_validator
from app.online import duplicate_pipeline, validator as validator_module
from app.online.models import numerical_instance_contract
from app.online.multi_validator import (
    _payload_for,
    _system_prompt,
    _validate_grounding_evidence,
    protect_question,
    validate_exam_multi,
)

FACTS = {
    "same_core_knowledge": "no",
    "same_student_operation_or_method": "no",
    "same_target_or_result": "no",
    "genuinely_new_reasoning": "yes",
    "one_answer_substantially_gives_away_other": "no",
}


class _Client:
    provider_name = "offline"
    model = "fake"
    last_usage: dict = {}

    def __init__(self, calls: list[str], responder=None):
        self.calls = calls
        self.responder = responder

    def chat_structured(self, prompt, *, schema_name, **_kwargs):
        payload = json.loads(
            prompt.split("<VALIDATION_DATA>\n", 1)[1].split("\n</VALIDATION_DATA>", 1)[0]
        )
        role = payload["validator"]
        self.calls.append(role)
        if self.responder is not None:
            return self.responder(role, payload)
        return {
            "validator": role,
            "verdicts": [
                {
                    "question_id": item["question_id"],
                    "status": "PASS",
                    "issues": [],
                    "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
                    "observed_difficulty": "easy" if role == "difficulty" else None,
                    "calculation_evidence": [],
                }
                for item in payload["questions"]
            ],
        }


def _system_task(*, exact_source_instance: bool = False) -> dict:
    selected = {
        "variation_id": "variation_system",
        "variation_group_id": "variation_group_system",
        "question_types": ["equation"],
        "operation": "solve_linear_system",
        "target": "the values of x and y",
        "givens": ["x + y = 5", "x - y = 1"],
        "numeric_values_may_change": True,
        "evidence_chunk_ids": ["c1"],
    }
    return {
        "slot_id": "m1_equation_1",
        "task_id": "task_system",
        "concept": "systems of two linear equations",
        "knowledge_focus": "solve a two-variable linear system",
        "task_operation": "solve_linear_system",
        "target": "the values of x and y",
        "selected_variation_id": selected["variation_id"],
        "permitted_variations": [selected],
        "allowed_variations": [selected],
        "required_givens": ["x + y = A", "x - y = B"],
        "given_conditions": ["x + y = 5", "x - y = 1"],
        "source_example_givens": ["x + y = 5", "x - y = 1"],
        "complexity_boundaries": ["two variables, one linear system"],
        "exact_source_instance": exact_source_instance,
        "source_chunk_id": "c1",
        "support_chunk_ids": [],
        "subject": "Mathematics",
        "reasoning_mode": "calculation",
        "student_work_product": "worked_solution",
        "requires_quantitative_validation": True,
        "difficulty": "easy",
        "requested_difficulty": "easy",
        "actual_difficulty": "easy",
    }


def _system_question(*, first: str = "x + y = 12", second: str = "x - y = 4") -> dict:
    return {
        "question_id": "model1_equation_1",
        "slot_id": "m1_equation_1",
        "equation": f"Solve the system: {first}; {second}",
        "solution_steps": [f"Add: {first} + ({second})", "2x = 16", "x = 8", "y = 4"],
        "final_answer": "x = 8, y = 4",
        "solution_certificate": {
            "method": "elimination",
            "important_givens": [first, second],
            "calculation_steps": ["Add the equations.", "Substitute x = 8."],
            "final_answer": "x = 8, y = 4",
            "units": None,
            "matching_correct_option": None,
            "verification_method": "Substitute (8, 4) into both equations.",
        },
    }


def _exam(question: dict | None = None, *, model_number: int = 1) -> dict:
    return {"model_number": model_number, "questions": {"equation": [question or _system_question()]}}


def _plans(task: dict | None = None, *, model_number: int = 1) -> dict:
    return {f"m{model_number}_equation_1": dict(task or _system_task())}


def _run(
    question: dict | None = None,
    *,
    task: dict | None = None,
    responder=None,
    domain: str = "STEM",
    cache_scope: str = "bfe29ab579f1",
) -> tuple[dict, list[str]]:
    calls: list[str] = []
    multi_validator.clear_job_validation_cache(cache_scope)
    result = validate_exam_multi(
        _exam(question),
        plan_by_slot=_plans(task),
        source_by_chunk={"c1": "The source example solves x + y = 5 and x - y = 1."},
        domain=domain,
        client_factory=lambda _role: _Client(calls, responder),
        cache_scope=cache_scope,
    )
    return result, calls


def _fail_responder(role_to_code: dict[str, str], **issue_extra):
    def responder(role: str, payload: dict) -> dict:
        code = role_to_code.get(role)
        if code is None:
            return {
                "validator": role,
                "verdicts": [
                    {
                        "question_id": item["question_id"], "status": "PASS", "issues": [],
                        "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
                        "calculation_evidence": [],
                    }
                    for item in payload["questions"]
                ],
            }
        field = "equation" if role in {
            "grounding", "task_alignment", "difficulty",
        } else "final_answer"
        return {
            "validator": role,
            "verdicts": [{
                "question_id": payload["questions"][0]["question_id"],
                "status": "FAIL",
                "issues": [{
                    "failure_code": code,
                    "fields_to_fix": [field],
                    "reason": "A concrete defect was found in the generated content.",
                    "expected_fix": "Correct the generated content.",
                    **issue_extra,
                }],
                "grounding_outcome": (
                    "CONTRADICTED" if role == "grounding" and "contradicted" in code else None
                ),
                "calculation_evidence": [],
            }],
        }

    return responder


# ---------------------------------------------------------------------------
# Fix 1: new numerical instances pass; real defects still fail
# ---------------------------------------------------------------------------
def test_new_numerical_instance_is_illustrative_for_alignment_and_grounding():
    task = _system_task()
    question = _system_question()
    protected = protect_question(
        "equation", question, task, {"c1": "source example x+y=5, x-y=1"}
    )
    alignment = _payload_for("task_alignment", protected)["task_contract"]
    grounding = _payload_for("grounding", protected)["grounding_contract"]
    quantitative = _payload_for("quantitative", protected)["quantitative_contract"]

    for contract in (alignment, grounding, quantitative):
        assert contract["value_policy"] == "illustrative"
        assert contract["numeric_values_may_change"] is True
        assert contract["immutable_values"] == []
        assert "x + y = 5" in contract["illustrative_values"]
    # The generated instance is materially different from the source example.
    assert "x + y = 12" in question["equation"]

    prompt = _system_prompt("task_alignment")
    assert "ordinary Catalog/source example numbers" in prompt
    assert "not by itself wrong_givens" in prompt

    result, calls = _run(question, task=task)
    assert result["all_pass"] is True
    assert set(calls) == {
        "grounding", "task_alignment", "answer_correctness", "difficulty", "quantitative",
    }


def test_explicit_fixed_numerical_data_stays_immutable():
    task = _system_task()
    task["fixed_numerical_data"] = ["x + y = 5", "x - y = 1"]
    contract = numerical_instance_contract(task)
    assert contract["value_policy"] == "immutable"
    assert contract["numeric_values_may_change"] is False
    assert "x + y = 5" in contract["immutable_values"]
    assert contract["illustrative_values"] == []


def test_exact_source_instance_marks_values_immutable():
    task = _system_task(exact_source_instance=True)
    protected = protect_question(
        "equation", _system_question(), task, {"c1": "fixed source instance"}
    )
    contract = _payload_for("task_alignment", protected)["task_contract"]
    assert contract["value_policy"] == "immutable"
    assert contract["exact_source_instance"] is True
    assert "x + y = 5" in contract["immutable_values"]
    assert contract["illustrative_values"] == []

    result, _ = _run(
        _system_question(first="x + y = 6", second="x - y = 2"), task=task,
        responder=_fail_responder({"task_alignment": "wrong_givens"}),
    )
    assert result["all_pass"] is False
    assert result["verdicts"][0]["action"] == "FIX_QUESTION"
    assert result["verdicts"][0]["required_fixes"][0]["failure_code"] == "wrong_givens"


def test_incorrect_operation_still_fails():
    result, _ = _run(
        responder=_fail_responder({"task_alignment": "wrong_variation"}),
    )
    assert result["all_pass"] is False
    assert result["verdicts"][0]["required_fixes"][0]["failure_code"] == "wrong_variation"


def test_constraint_violation_still_fails():
    result, _ = _run(
        responder=_fail_responder({"task_alignment": "constraint_violation"}),
    )
    assert result["all_pass"] is False
    assert result["verdicts"][0]["required_fixes"][0]["failure_code"] == "constraint_violation"


def test_incorrect_answer_and_calculation_still_fail():
    answer_result, _ = _run(
        responder=_fail_responder({"answer_correctness": "incorrect_answer"}),
    )
    assert answer_result["all_pass"] is False
    assert answer_result["verdicts"][0]["action"] == "FIX_ANSWER"

    calculation_result, _ = _run(
        responder=_fail_responder({"quantitative": "incorrect_result"}),
    )
    assert calculation_result["all_pass"] is False
    assert calculation_result["verdicts"][0]["action"] == "FIX_ANSWER"


# ---------------------------------------------------------------------------
# Fix 2: revalidation is proportional to changed questions/roles
# ---------------------------------------------------------------------------
def _two_equation_exam() -> tuple[dict, dict]:
    task = _system_task()
    plans = {
        "m1_equation_1": {**task, "slot_id": "m1_equation_1", "task_id": "task_1"},
        "m1_equation_2": {
            **deepcopy(task), "slot_id": "m1_equation_2", "task_id": "task_2",
            "selected_variation_id": "variation_system",
        },
    }
    questions = [
        _system_question(),
        {
            **_system_question(first="x + y = 10", second="x - y = 2"),
            "question_id": "model1_equation_2", "slot_id": "m1_equation_2",
        },
    ]
    exam = {"model_number": 1, "questions": {"equation": questions}}
    return exam, plans


def test_only_the_changed_question_is_revalidated(monkeypatch):
    monkeypatch.setattr(multi_validator.time, "sleep", lambda _seconds: None)
    multi_validator.clear_job_validation_cache("proportional")
    exam, plans = _two_equation_exam()
    calls: list[tuple[str, list[str]]] = []

    def responder(role: str, payload: dict) -> dict:
        calls.append((role, [item["question_id"] for item in payload["questions"]]))
        return {
            "validator": role,
            "verdicts": [
                {
                    "question_id": item["question_id"], "status": "PASS", "issues": [],
                    "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
                    "observed_difficulty": "easy" if role == "difficulty" else None,
                    "calculation_evidence": [],
                }
                for item in payload["questions"]
            ],
        }

    def run(target_exam, question_ids=None):
        calls.clear()
        result = validate_exam_multi(
            target_exam, plan_by_slot=plans,
            source_by_chunk={"c1": "source example"},
            domain="STEM", client_factory=lambda _role: _Client([], responder),
            cache_scope="proportional", question_ids=question_ids,
        )
        assert result["all_pass"] is True
        return calls, result

    first_calls, _ = run(exam)
    assert {qid for _role, ids in first_calls for qid in ids} == {
        "model1_equation_1", "model1_equation_2",
    }

    # Only question 1 changed; question 2 must reuse its passing verdicts.
    changed = deepcopy(exam)
    changed["questions"]["equation"][0]["final_answer"] = "x = 8.5, y = 3.5"
    changed["questions"]["equation"][0]["solution_steps"][-1] = "y = 3.5"
    second_calls, _ = run(changed)
    assert {role for role, _ids in second_calls} <= {
        "grounding", "answer_correctness", "difficulty", "quantitative",
    }
    assert "task_alignment" not in {role for role, _ids in second_calls}
    assert {
        qid for _role, ids in second_calls for qid in ids
    } == {"model1_equation_1"}

    # Revalidating the unchanged revision a second time is entirely free.
    third_calls, _ = run(changed)
    assert third_calls == []


# ---------------------------------------------------------------------------
# Fix 3: duplicate facts are persisted and reused
# ---------------------------------------------------------------------------
def _duplicate_exam(stem: str = "Equivalent form one") -> tuple[dict, dict]:
    exam = {"model_number": 1, "questions": {"true_false": [
        {"question_id": "q1", "slot_id": "s1", "statement": stem, "answer": True},
        {"question_id": "q2", "slot_id": "s2", "statement": "Equivalent form two", "answer": False},
    ]}}
    plans = {"s1": {"task_id": "task-a"}, "s2": {"task_id": "task-a"}}
    return exam, plans


def test_unchanged_duplicate_pair_reuses_persisted_facts(monkeypatch):
    exam, plans = _duplicate_exam()
    judged: list[list[str]] = []

    class DuplicateClient:
        provider_name = "offline"
        model = "fake"
        last_usage: dict = {}

    monkeypatch.setattr(duplicate_judge, "create_llm_client", lambda: DuplicateClient())

    def request(_client, _prompt, **kwargs):
        judged.append(list(kwargs["expected_ids"]))
        return {"verdicts": [{"pair_id": pair_id, **FACTS} for pair_id in kwargs["expected_ids"]]}

    monkeypatch.setattr(duplicate_judge, "request_structured", request)

    first = duplicate_judge.review_duplicate_candidates(exam, plan_by_slot=plans)
    assert first["fact_cache_misses"] == 1
    assert len(judged) == 1

    # The same unchanged pair is served from the persisted content-keyed facts.
    second = duplicate_judge.review_duplicate_candidates(
        exam, plan_by_slot=plans, fact_cache=first["fact_cache"],
    )
    assert second["fact_cache_hits"] == 1
    assert second["fact_cache_misses"] == 0
    assert len(judged) == 1

    # A changed pair gets a new identity and is re-judged once.
    changed, _ = _duplicate_exam("Materially changed form")
    third = duplicate_judge.review_duplicate_candidates(
        changed, plan_by_slot=plans, fact_cache=first["fact_cache"],
    )
    assert third["fact_cache_misses"] == 1
    assert len(judged) == 2


def test_persisted_duplicate_fact_cache_is_content_keyed_not_pair_keyed():
    key_one = duplicate_judge._fact_cache_key(
        {
            "pair_id": "q1::q2",
            "left": {"question_type": "true_false", "stem": "one", "answer": "True"},
            "right": {"question_type": "true_false", "stem": "two", "answer": "False"},
        },
        type("C", (), {"model": "m"})(),
    )
    key_two = duplicate_judge._fact_cache_key(
        {
            "pair_id": "q1::q2",
            "left": {"question_type": "true_false", "stem": "changed", "answer": "True"},
            "right": {"question_type": "true_false", "stem": "two", "answer": "False"},
        },
        type("C", (), {"model": "m"})(),
    )
    assert key_one != key_two


# ---------------------------------------------------------------------------
# Fix 4: grounding failures need concrete claims and assigned evidence
# ---------------------------------------------------------------------------
def _mcq_boundary_item():
    task = {
        "slot_id": "m1_mcq_1",
        "task_id": "task_boundary",
        "concept": "graphing linear inequalities",
        "task_operation": "interpret_boundary",
        "target": "describe the boundary of a strict inequality",
        "source_chunk_id": "c1",
        "support_chunk_ids": [],
        "exact_source_instance": False,
        "difficulty": "easy",
        "requested_difficulty": "easy",
        "selected_variation": {
            "variation_id": "v1", "operation": "interpret_boundary",
            "target": "describe the boundary", "givens": ["y > 3x - 2"],
        },
        "selected_variation_id": "v1",
    }
    question = {
        "question_id": "model1_mcq_1",
        "slot_id": "m1_mcq_1",
        "question": "When graphing y > 3x - 2, which statement describes the boundary?",
        "options": {
            "A": "The boundary is included and is a solid line.",
            "B": "The boundary is not included and is a dashed line.",
            "C": "There is no boundary.",
            "D": "The boundary is the x-axis.",
        },
        "correct_answer": "B",
        "solution_certificate": {
            "method": "strict inequality boundary",
            "important_givens": ["y > 3x - 2"],
            "calculation_steps": ["A strict inequality excludes the boundary."],
            "final_answer": "B",
            "units": None,
            "matching_correct_option": "B",
            "verification_method": "Substitute a boundary point.",
        },
    }
    return task, question


def test_valid_strict_inequality_boundary_is_not_a_grounding_failure(monkeypatch):
    monkeypatch.setattr(multi_validator.time, "sleep", lambda _seconds: None)
    task, question = _mcq_boundary_item()
    prompt = _system_prompt("grounding")
    assert "strict inequality excludes its boundary and is therefore dashed" in prompt
    assert "do not invent an ambiguity or contradiction" in prompt

    result = validate_exam_multi(
        {"model_number": 1, "questions": {"mcq": [question]}},
        plan_by_slot={"m1_mcq_1": task},
        source_by_chunk={
            "c1": (
                "For a strict inequality such as y > 3x - 2, the boundary y = 3x - 2 "
                "is not part of the solution and is drawn as a dashed line."
            )
        },
        domain="STEM",
        client_factory=lambda _role: _Client([]),
        cache_scope="boundary-question",
    )
    assert result["all_pass"] is True


def test_grounding_contradiction_without_concrete_evidence_is_not_accepted(monkeypatch):
    monkeypatch.setattr(multi_validator.time, "sleep", lambda _seconds: None)
    task, question = _mcq_boundary_item()

    def responder(role: str, payload: dict) -> dict:
        if role != "grounding":
            return {
                "validator": role,
                "verdicts": [{
                    "question_id": item["question_id"], "status": "PASS", "issues": [],
                    "calculation_evidence": [],
                } for item in payload["questions"]],
            }
        return {"validator": "grounding", "verdicts": [{
            "question_id": "model1_mcq_1", "status": "FAIL",
            "issues": [{
                "failure_code": "contradicted",
                "fields_to_fix": ["question"],
                "reason": "The question contains an ambiguity about the boundary being dashed.",
                "expected_fix": "Remove the ambiguity.",
            }],
            "grounding_outcome": "CONTRADICTED",
            "calculation_evidence": [],
        }]}

    result = validate_exam_multi(
        {"model_number": 1, "questions": {"mcq": [question]}},
        plan_by_slot={"m1_mcq_1": task},
        source_by_chunk={"c1": "A strict inequality excludes the boundary."},
        domain="STEM",
        client_factory=lambda _role: _Client([], responder),
        cache_scope="invented-contradiction",
    )
    verdict = result["verdicts"][0]
    assert result["all_pass"] is False
    assert verdict["validation_status"] != "PASS"
    assert verdict["action"] != "PASS"


def test_real_grounding_failure_with_evidence_still_fails():
    task, question = _mcq_boundary_item()
    question = {
        **question,
        "question": "When graphing y > 3x - 2, the boundary is a solid line.",
        "options": {
            "A": "True", "B": "False", "C": "Sometimes", "D": "The boundary is vertical",
        },
        "correct_answer": "A",
    }
    def responder(role: str, payload: dict) -> dict:
        if role != "grounding":
            return {
                "validator": role,
                "verdicts": [{
                    "question_id": "model1_mcq_1", "status": "PASS", "issues": [],
                    "calculation_evidence": [],
                }],
            }
        return {"validator": "grounding", "verdicts": [{
            "question_id": "model1_mcq_1", "status": "FAIL",
            "issues": [{
                "failure_code": "contradicted",
                "fields_to_fix": ["question"],
                "challenged_claim": "The boundary of y > 3x - 2 is a solid line.",
                "source_evidence_ids": ["c1"],
                "reason": "The assigned source says a strict inequality excludes its boundary.",
                "expected_fix": "Describe the boundary as dashed and excluded.",
            }],
            "grounding_outcome": "CONTRADICTED",
            "calculation_evidence": [],
        }]}

    result = validate_exam_multi(
        {"model_number": 1, "questions": {"mcq": [question]}},
        plan_by_slot={"m1_mcq_1": task},
        source_by_chunk={"c1": "A strict inequality excludes the boundary."},
        domain="STEM",
        client_factory=lambda _role: _Client([], responder),
        cache_scope="real-contradiction",
    )
    verdict = result["verdicts"][0]
    assert result["all_pass"] is False
    assert verdict["required_fixes"][0]["failure_code"] == "contradicted"
    assert verdict["required_fixes"][0]["challenged_claim"]
    assert verdict["required_fixes"][0]["source_evidence_ids"] == ["c1"]


def test_grounding_evidence_validator_rejects_unassigned_evidence():
    task, question = _mcq_boundary_item()
    protected = protect_question("mcq", question, task, {"c1": "source"})
    verdict = {
        "status": "FAIL",
        "issues": [{
            "failure_code": "unsupported",
            "challenged_claim": "The boundary is dashed.",
            "source_evidence_ids": ["outside-chunk"],
        }],
    }
    error = _validate_grounding_evidence("grounding", protected, verdict)
    assert error and "unassigned evidence" in error


# ---------------------------------------------------------------------------
# Fix 3 (candidate gate): one content-keyed ledger across staged candidates
# ---------------------------------------------------------------------------
def _fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _candidate_mcq(qid: str, stem: str) -> dict:
    return {
        "question_id": qid,
        "slot_id": qid.replace("model", "m", 1),
        "question": stem,
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "correct_answer": "A",
    }


def _candidate_plan_item(qid: str, task_id: str) -> dict:
    return {
        "slot_id": qid.replace("model", "m", 1), "task_id": task_id,
        "source_chunk_id": "c1", "support_chunk_ids": [],
        "concept": "concept", "task_operation": "select_value", "target": "value",
        "question_type": "mcq", "difficulty": "easy", "requested_difficulty": "easy",
    }


def test_batched_candidate_gate_rejects_only_the_conflicting_candidate(monkeypatch):
    """A batched gate must not let a conflict block unrelated candidates."""
    q1, q2 = _candidate_mcq("model1_mcq_1", "first"), _candidate_mcq("model1_mcq_2", "second")
    staged1 = {**q1, "question": "repaired first"}
    staged2 = {**q2, "question": "repaired second"}
    plans = [{"model_number": 1, "items": {"mcq": [
        _candidate_plan_item("model1_mcq_1", "task1"),
        _candidate_plan_item("model1_mcq_2", "task2"),
    ]}}]

    def candidate(qid: str, staged: dict, original: dict) -> dict:
        return {
            "model_number": 1, "slot_id": qid.replace("model", "m", 1),
            "question_id": qid, "question_type": "mcq",
            "exam": {"model_number": 1, "questions": {"mcq": [staged]}},
            "plans": plans, "original_question": original,
            "original_failure_findings": [], "repair_metadata": {},
            "candidate_id": f"cand-{qid}", "candidate_fingerprint": _fingerprint(staged),
            "attempt": 1, "repair_action": "replace_task",
        }

    state = {
        "job_id": "batched-conflict", "assessment_profile": {"domain": "STEM"},
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [q1, q2]}}],
        "plans": plans, "retrieved_chunks": [{"child_id": "c1", "content": "evidence"}],
        "validation_reports": [], "validated_models": [],
        "question_repair_attempts": {}, "question_repair_failures": {},
        "slot_completion_attempts": {},
        "repair_candidates": [
            candidate("model1_mcq_1", staged1, q1),
            candidate("model1_mcq_2", staged2, q2),
        ],
        "duplicate_fact_cache": {}, "duplicate_screen_fingerprints": {}, "warnings": [],
    }

    def fake_validate(exam, **_kwargs):
        ids = [item["question_id"] for section in (exam.get("questions") or {}).values()
               for item in (section.get("items") if isinstance(section, dict) else section)]
        return {
            "model_number": 1, "all_pass": True,
            "verdicts": [{"question_id": qid, "action": "PASS", "question_valid": True,
                          "answer_valid": True, "failure_code": "pass", "fields_to_fix": [],
                          "required_fixes": [], "role_statuses": {"duplicate": "PASS"}}
                         for qid in ids],
            "role_results": {}, "role_fingerprints": {}, "validation_call_logs": [],
        }

    def fake_review(exam, **_kwargs):
        ids = [item["question_id"] for section in (exam.get("questions") or {}).values()
               for item in (section.get("items") if isinstance(section, dict) else section)]
        synthetic = next(
            value for value in ids
            if value.startswith("__candidate_") and value.endswith("model1_mcq_1")
        )
        active = "model1_mcq_2"
        pair_id = "::".join(sorted((synthetic, active)))
        return {
            "candidate_pairs": [{
                "pair_id": pair_id, "left": {"question_id": synthetic},
                "right": {"question_id": active},
            }],
            "pair_verdicts": [{"pair_id": pair_id, "classification": "duplicate"}],
            "duplicate_groups": [], "similarity_distribution": {}, "usage": {},
            "fact_cache": {}, "fact_cache_hits": 0, "fact_cache_misses": 1,
        }

    monkeypatch.setattr(multi_validator, "validate_exam_multi", fake_validate)
    monkeypatch.setattr(duplicate_judge, "review_duplicate_candidates", fake_review)
    monkeypatch.setattr(duplicate_pipeline, "scan_exam_duplicates", lambda _state: {
        "duplicate_reports": [], "pending_duplicate_groups": [], "warnings": [],
        "duplicate_fact_cache": {}, "duplicate_scan_count": 1,
        "duplicate_findings": [], "duplicate_similarity_distributions": [],
        "unresolved_duplicate_conflicts": [], "duplicate_candidate_blacklist": {},
        "duplicate_replacement_failures": {}, "duplicate_repair_history": [],
        "generated_exams": [],
    })
    result = validator_module.validate_generated_questions(state)

    questions = result["generated_exams"][0]["questions"]["mcq"]
    by_id = {value["question_id"]: value for value in questions}
    # Candidate 1 conflicted with active question 2 and was rejected; candidate
    # 2 is unrelated and still commits.
    assert by_id["model1_mcq_1"]["question"] == "first"
    assert by_id["model1_mcq_2"]["question"] == "repaired second"
    rejected = [
        value for value in result["validation_repair_history"]
        if value.get("question_id") == "model1_mcq_1"
    ]
    assert rejected
    assert rejected[-1]["status"] == "rejected"
    assert rejected[-1]["new_failure_code"] == "duplicate_question"


def test_role_fingerprint_does_not_rebuild_the_response_schema(monkeypatch):
    """The schema is hashed once; fingerprinting must not call Pydantic again."""
    from app.llm.schemas import SpecializedValidatorVerdict

    def _explode(*_args, **_kwargs):
        raise AssertionError("model_json_schema() must be cached")

    monkeypatch.setattr(SpecializedValidatorVerdict, "model_json_schema", _explode)
    task = _system_task()
    protected = protect_question("equation", _system_question(), task, {"c1": "source"})
    first = multi_validator._role_cache_fingerprint("task_alignment", protected)
    second = multi_validator._role_cache_fingerprint("task_alignment", protected)
    assert first == second


def test_candidate_duplicate_gate_shares_one_fact_ledger_across_candidates(monkeypatch):
    """The m2 candidate gate used to reset its duplicate facts per attempt.

    One pair was judged 15 times because every staged candidate started from the
    same empty ledger.  Every gate call must receive the facts learned by the
    previous calls in the same validation node, and the final scan must inherit
    them too.
    """
    q1, q2 = _candidate_mcq("model1_mcq_1", "first"), _candidate_mcq("model1_mcq_2", "second")
    staged1 = {**q1, "question": "repaired first"}
    staged2 = {**q2, "question": "repaired second"}
    plans = [{"model_number": 1, "items": {"mcq": [
        _candidate_plan_item("model1_mcq_1", "task1"),
        _candidate_plan_item("model1_mcq_2", "task2"),
    ]}}]

    def candidate(qid: str, staged: dict, original: dict) -> dict:
        return {
            "model_number": 1,
            "slot_id": qid.replace("model", "m", 1),
            "question_id": qid,
            "question_type": "mcq",
            "exam": {"model_number": 1, "questions": {"mcq": [staged]}},
            "plans": plans,
            "original_question": original,
            "original_failure_findings": [],
            "repair_metadata": {},
            "candidate_id": f"cand-{qid}",
            "candidate_fingerprint": _fingerprint(staged),
            "attempt": 1,
            "repair_action": "replace_task",
        }

    state = {
        "job_id": "candidate-ledger",
        "assessment_profile": {"domain": "STEM"},
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [q1, q2]}}],
        "plans": plans,
        "retrieved_chunks": [{"child_id": "c1", "content": "evidence"}],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "question_repair_failures": {},
        "slot_completion_attempts": {},
        "repair_candidates": [
            candidate("model1_mcq_1", staged1, q1),
            candidate("model1_mcq_2", staged2, q2),
        ],
        "duplicate_fact_cache": {"existing": {"same_core_knowledge": "no"}},
        "duplicate_screen_fingerprints": {},
        "warnings": [],
    }

    def fake_validate(exam, **_kwargs):
        ids = [
            item["question_id"]
            for section in (exam.get("questions") or {}).values()
            for item in (section.get("items") if isinstance(section, dict) else section)
            if item.get("question_id")
        ]
        return {
            "model_number": exam.get("model_number"), "all_pass": True,
            "verdicts": [{
                "question_id": qid, "action": "PASS", "question_valid": True,
                "answer_valid": True, "failure_code": "pass", "fields_to_fix": [],
                "required_fixes": [], "role_statuses": {"duplicate": "PASS"},
            } for qid in ids],
            "role_results": {}, "role_fingerprints": {}, "validation_call_logs": [],
        }

    monkeypatch.setattr(multi_validator, "validate_exam_multi", fake_validate)
    review_seen: list[dict] = []
    scan_seen: list[dict] = []

    def fake_review(_exam, **kwargs):
        review_seen.append(dict(kwargs.get("fact_cache") or {}))
        cache = dict(kwargs.get("fact_cache") or {})
        cache[f"review-{len(review_seen)}"] = {"same_core_knowledge": "no"}
        return {
            "candidate_pairs": [], "pair_verdicts": [], "duplicate_groups": [],
            "similarity_distribution": {}, "usage": {}, "fact_cache": cache,
            "fact_cache_hits": 0, "fact_cache_misses": 0,
        }

    def fake_scan(scan_state):
        scan_seen.append(dict(scan_state.get("duplicate_fact_cache") or {}))
        cache = dict(scan_state.get("duplicate_fact_cache") or {})
        cache[f"scan-{len(scan_seen)}"] = {"same_core_knowledge": "no"}
        return {
            "duplicate_reports": [], "pending_duplicate_groups": [], "warnings": [],
            "duplicate_fact_cache": cache, "duplicate_scan_count": len(scan_seen),
            "duplicate_findings": [], "duplicate_similarity_distributions": [],
            "unresolved_duplicate_conflicts": [],
            "duplicate_candidate_blacklist": {}, "duplicate_replacement_failures": {},
            "duplicate_repair_history": [], "generated_exams": [],
        }

    monkeypatch.setattr(duplicate_judge, "review_duplicate_candidates", fake_review)
    monkeypatch.setattr(duplicate_pipeline, "scan_exam_duplicates", fake_scan)
    result = validator_module.validate_generated_questions(state)

    # Both ready candidates are judged in one batched gate, then the final
    # authoritative scan inherits the facts learned by that gate.
    assert len(review_seen) == 1
    assert set(review_seen[0]) == {"existing"}
    assert len(scan_seen) == 1
    assert set(scan_seen[0]) == {"existing", "review-1"}
    assert {"existing", "review-1", "scan-1"} <= set(result["duplicate_fact_cache"])
