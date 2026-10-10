"""Tests for stable question IDs + the post-generation validator/repair stage.

The LLM is mocked (FakeClient), so no server call is made. Focus:
- stable, collision-free, model-isolated question IDs
- per-question validation (one call per exam) with PASS / FIX_ANSWER /
  FIX_OPTIONS / FIX_QUESTION / FIX_QUESTION_AND_ANSWER
- minimal-difference repair, enforced by code via fields_to_fix
- ID-keyed write-back (never by position)
- bounded, per-model repair attempts
"""
from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

import app.online.exam_builder as eb
from app.llm.schemas import validator_response_model
from app.online import validator as val
from app.online.validator import (
    MAX_QUESTION_REPAIR_ATTEMPTS,
    repair_exam,
    repair_invalid_questions,
    validate_exam,
    validate_generated_questions,
    review_exam_globally,
    build_global_review_prompt,
    GLOBAL_VALIDATOR_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_RETRY_SYSTEM_PROMPT,
)
from app.online.graph import _route_after_validation


class FakeClient:
    """chat_json mock; routes by system prompt and records calls."""

    def __init__(self, validator_result=None, repair_result=None, validator_retry_result=None,
                 global_result=None):
        self.validator_result = validator_result
        self.validator_retry_result = validator_retry_result
        self.repair_result = repair_result
        self.global_result = global_result
        self.validate_calls = 0
        self.retry_calls = 0
        self.repair_calls = 0
        self.last_validator_prompt = None
        self.last_repair_prompt = None
        self.last_max_tokens = None

    def chat_json(self, prompt, system_prompt=None, temperature=0.7,
                  max_tokens=4096, timeout=600, max_repair_attempts=2):
        self.last_max_tokens = max_tokens
        if system_prompt == GLOBAL_VALIDATOR_SYSTEM_PROMPT:
            if self.global_result is not None:
                return self.global_result
            registry = json.loads(prompt.split("## Complete exam registry\n", 1)[1].split(
                "\n\n## Candidate pairs", 1
            )[0])
            return _global_pass_result([item["question_id"] for item in registry])
        if system_prompt == VALIDATOR_SYSTEM_PROMPT:
            self.validate_calls += 1
            self.last_validator_prompt = prompt
            return (
                self.validator_result
                if isinstance(self.validator_result, dict)
                else {"verdicts": self.validator_result}
            )
        if system_prompt == VALIDATOR_RETRY_SYSTEM_PROMPT:
            self.retry_calls += 1
            return (
                self.validator_retry_result
                if isinstance(self.validator_retry_result, dict)
                else {"verdicts": self.validator_retry_result}
            )
        if system_prompt == REPAIR_SYSTEM_PROMPT:
            self.repair_calls += 1
            self.last_repair_prompt = prompt
            return (
                self.repair_result
                if isinstance(self.repair_result, dict)
                else {"repairs": self.repair_result}
            )
        raise AssertionError(f"unexpected system prompt: {system_prompt!r}")


def _validator_entries(prompt):
    block = prompt.split("## Questions to review\n", 1)[1].split(
        "\n\n## Verdict object format", 1
    )[0]
    return json.loads(block)


class EchoValidatorClient:
    """Return PASS for every ID and retain exact per-call batch membership."""

    def __init__(self):
        self.primary_batches = []
        self.retry_batches = []

    def chat_json(self, prompt, system_prompt=None, **kwargs):
        entries = _validator_entries(prompt)
        ids = [entry["question_id"] for entry in entries]
        if system_prompt == VALIDATOR_SYSTEM_PROMPT:
            self.primary_batches.append(ids)
        elif system_prompt == VALIDATOR_RETRY_SYSTEM_PROMPT:
            self.retry_batches.append(ids)
        else:
            raise AssertionError(f"unexpected system prompt: {system_prompt!r}")
        return {"verdicts": [_pass_verdict(qid) for qid in ids]}


def _mcq(qid, question="Which metric is the harmonic mean of precision and recall?",
         answer="B", options=None):
    return {
        "question_id": qid, "slot_id": qid.replace("model", "m", 1),
        "question": question,
        "options": options or {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
        "correct_answer": answer,
    }


def _tf(qid, statement="Cyanobacteria released oxygen through photosynthesis.", answer="True"):
    return {"question_id": qid, "slot_id": qid.replace("model", "m", 1),
            "statement": statement, "answer": answer}


def _sa(qid, question="Why is peer review important?",
        ref="Peer review allows other experts to evaluate the quality of research and identify errors."):
    return {"question_id": qid, "question": question, "reference_answer": ref}


def _definition(qid, term="Force", ref="A push or pull acting on an object."):
    return {"question_id": qid, "term": term, "reference_answer": ref}


def _equation(qid, equation="Solve 2x + 5 = 15.", steps=None, final="x = 5"):
    return {
        "question_id": qid,
        "equation": equation,
        "solution_steps": steps or ["2x = 10", "x = 5"],
        "final_answer": final,
    }


def _word_problem(qid, question="A car travels 120 km in 2 hours. Find its speed.",
                  steps=None, final="60 km/h"):
    return {
        "question_id": qid,
        "question": question,
        "solution_steps": steps or ["speed = distance ÷ time", "120 ÷ 2 = 60"],
        "final_answer": final,
    }


def _essay(qid, question="Compare precision and recall.", ref="A full reference answer.",
           key_points=None):
    return {"question_id": qid, "question": question, "reference_answer": ref,
            "key_points": key_points or ["a", "b"]}


def _fitb_item(qid, question="The measure of a shape's inside surface is called ________.",
               answers=None, distractors=None):
    return {
        "question_id": qid, "question": question, "answers": answers or ["area"],
        "distractors": distractors or ["perimeter", "volume"],
    }


def _fitb(items, bank=None):
    generated = list(dict.fromkeys(
        str(value)
        for item in items
        for value in [*(item.get("answers") or []), *(item.get("distractors") or [])]
    ))
    return {"word_bank": bank or generated, "items": items}


def _exam(questions, model_number=1):
    return {"model_number": model_number, "questions": questions}


def test_validator_reports_requested_missing_fitb_slots_and_whole_section(monkeypatch):
    monkeypatch.setattr(val, "_validate_with_context", lambda *args, **kwargs: {
        "model_number": 1, "all_pass": True, "verdicts": [], "warnings": []
    })
    state = {
        "generated_exams": [_exam({})],
        "plans": [{
            "model_number": 1,
            "items": {"fill_in_the_blank": [{
                "slot_id": "m1_fill_in_the_blank_1",
                "fitb_entry_id": "fitb_alpha",
                "fitb_answer": "alpha",
                "clue_basis": "source clue",
                "evidence_chunk_id": "chunk-a",
            }]},
        }],
    }

    result = validate_generated_questions(state)
    report = result["validation_reports"][0]

    assert report["all_pass"] is False
    assert report["missing_sections"] == ["fill_in_the_blank"]
    assert report["missing_slots"] == [{
        "slot_id": "m1_fill_in_the_blank_1",
        "question_type": "fill_in_the_blank",
        "task_id": None,
        "fitb_entry_id": "fitb_alpha",
        "fixed_answer": "alpha",
        "clue_basis": "source clue",
        "evidence_chunk_id": "chunk-a",
        "failure_code": "missing_requested_slot",
        "recovery_requirement": (
            "Regenerate this exact slot with its fixed FITB answer; replace only its "
            "fitb_entry_id from the candidate bank if the assigned entry is exhausted."
        ),
    }]


def test_missing_visual_repair_can_replace_complete_task_and_answer():
    original = _word_problem(
        "model1_word_problem_1",
        question="Use the unseen wage table to calculate the weekly income.",
        steps=["Read the values from the table."],
        final="Unknown",
    )
    original["slot_id"] = "m1_word_problem_1"
    exam = _exam({"word_problem": [original]})
    replacement = _word_problem(
        "model1_word_problem_1",
        question="A service earns 65 dirhams per customer. Find the income for 4 customers.",
        steps=["income = 65 × 4", "income = 260"],
        final="260 dirhams",
    )
    replacement["slot_id"] = "m1_word_problem_1"
    fake = FakeClient(repair_result={"repairs": [{
        "question_id": "model1_word_problem_1",
        "repaired_fields": ["question", "solution_steps", "final_answer"],
        "question": replacement,
    }]})
    verdict = _verdict(
        "model1_word_problem_1", "FIX_QUESTION_AND_ANSWER",
        qv=False, av=False, fields=["question", "solution_steps", "final_answer"],
        reason="Required values exist only in an unseen table.",
        expected_fix="Replace with another fully text-supported task.",
    )
    verdict["failure_code"] = "visual_dependency"
    outcome = repair_exam(exam, [verdict], client=fake)
    assert outcome["repaired_ids"] == ["model1_word_problem_1"]
    assert exam["questions"]["word_problem"][0]["question"].startswith("A service earns")
    assert '"replacement_mode": true' in fake.last_repair_prompt


def _verdict(qid, action="PASS", qv=True, av=True, fields=None, reason="", expected_fix=""):
    return {
        "question_id": qid,
        "question_valid": qv,
        "answer_valid": av,
        "action": action,
        "fields_to_fix": fields or [],
        "failure_code": "pass" if action == "PASS" else "other",
        "expected_results": [],
        "reason": reason,
        "expected_fix": expected_fix,
    }


def _pass_verdict(qid):
    return _verdict(qid)


def _complete_quantitative_verdict(qid):
    return {
        "question_id": qid,
        "question_valid": True,
        "answer_valid": True,
        "action": "PASS",
        "fields_to_fix": [],
        "failure_code": "pass",
        "expected_results": ["6.29 × 10^-19 J"],
        "reason": "The calculation is correct.",
        "expected_fix": "",
        "independent_solution": {
            "method": "Use E = hν.",
            "important_givens": ["h = 6.626 × 10^-34 J·s", "ν = 9.50 × 10^14 s^-1"],
            "calculation_steps": ["E = (6.626 × 10^-34)(9.50 × 10^14)"],
            "final_answer": "6.29 × 10^-19 J",
            "units": "J",
            "matching_correct_option": None,
            "verification_method": "Divide E by h to recover ν.",
        },
        "certificate_valid": True,
        "first_incorrect_step": None,
    }


def test_quantitative_response_contract_requires_independent_work_and_certificate():
    qid = "model1_word_problem_1"
    model = validator_response_model([qid], quantitative=True)
    complete = _complete_quantitative_verdict(qid)

    assert model.model_validate({"verdicts": [complete]}).verdicts[0].certificate_valid
    for missing in ("independent_solution", "certificate_valid"):
        malformed = copy.deepcopy(complete)
        malformed.pop(missing)
        with pytest.raises(ValidationError):
            model.model_validate({"verdicts": [malformed]})


def test_quantitative_retry_reuses_exact_required_contract(monkeypatch):
    qid = "model1_word_problem_1"
    item = _word_problem(
        qid,
        question="A photon has frequency 9.50 × 10^14 s^-1. Find its energy.",
        steps=["E = hν", "E = 6.29 × 10^-19 J"],
        final="6.29 × 10^-19 J",
    )
    item["slot_id"] = "m1_word_problem_1"
    calls = []

    def fake_request(_client, prompt, **kwargs):
        calls.append((prompt, kwargs))
        if len(calls) == 1:
            incomplete = _complete_quantitative_verdict(qid)
            incomplete.pop("certificate_valid")
            return {"verdicts": [incomplete]}
        return {"verdicts": [_complete_quantitative_verdict(qid)]}

    monkeypatch.setattr(val, "request_structured", fake_request)
    report = validate_exam(
        _exam({"word_problem": [item]}), client=object(),
        question_ids=[qid], plan_by_slot=_quant_plan("m1_word_problem_1"),
        source_by_chunk={"chunk-a": "E = hν"},
        validation_role="stem_quantitative",
    )

    assert report["verdicts"][0]["action"] == "PASS"
    assert len(calls) == 2
    first_prompt, first_kwargs = calls[0]
    retry_prompt, retry_kwargs = calls[1]
    assert first_kwargs["schema_name"] == "validator_verdicts"
    assert retry_kwargs["schema_name"] == "validator_missing_verdict"
    assert first_kwargs["response_model"].model_json_schema() == retry_kwargs[
        "response_model"
    ].model_json_schema()
    for prompt in (first_prompt, retry_prompt):
        assert val.QUANTITATIVE_VALIDATOR_OUTPUT_SCHEMA in prompt
        assert '"independent_solution"' in prompt
        assert '"certificate_valid"' in prompt


def test_incomplete_quantitative_verdict_remains_unvalidated_after_one_retry(monkeypatch):
    qid = "model1_word_problem_1"
    item = _word_problem(qid)
    item["slot_id"] = "m1_word_problem_1"
    calls = []

    def fake_request(_client, _prompt, **kwargs):
        calls.append(kwargs["schema_name"])
        incomplete = _complete_quantitative_verdict(qid)
        incomplete.pop("independent_solution")
        return {"verdicts": [incomplete]}

    monkeypatch.setattr(val, "request_structured", fake_request)
    report = validate_exam(
        _exam({"word_problem": [item]}), client=object(),
        question_ids=[qid], plan_by_slot=_quant_plan("m1_word_problem_1"),
        source_by_chunk={"chunk-a": "speed = distance / time"},
        validation_role="stem_quantitative",
    )

    assert calls == ["validator_verdicts", "validator_missing_verdict"]
    assert report["verdicts"][0]["action"] == "UNVALIDATED"
    assert report["coverage"][0]["missing_ids"] == [qid]


def test_certificate_only_retry_verdict_is_retained_without_incorrect_step(monkeypatch):
    qid = "model1_mcq_3"
    item = _mcq(qid, answer="A")
    item["slot_id"] = "m1_mcq_3"
    item["solution_certificate"] = None
    calls = []

    def fake_request(_client, _prompt, **kwargs):
        calls.append(kwargs["schema_name"])
        if len(calls) == 1:
            return {"verdicts": []}
        verdict = _complete_quantitative_verdict(qid)
        verdict.update({
            "question_valid": True,
            "answer_valid": False,
            "action": "FIX_ANSWER",
            "fields_to_fix": ["solution_certificate"],
            "failure_code": "incomplete_solution",
            "reason": (
                "previous solution certificate was invalidated after question repair "
                "and must be rebuilt for the current question"
            ),
            "expected_fix": "Rebuild only the solution certificate.",
            "certificate_valid": False,
            "first_incorrect_step": None,
            "expected_results": ["A"],
        })
        return {"verdicts": [verdict]}

    monkeypatch.setattr(val, "request_structured", fake_request)
    report = validate_exam(
        _exam({"mcq": [item]}), client=object(), question_ids=[qid],
        plan_by_slot=_quant_plan("m1_mcq_3"),
        source_by_chunk={"chunk-a": "grounded source"},
        validation_role="stem_quantitative",
    )

    assert calls == ["validator_verdicts", "validator_missing_verdict"]
    assert report["verdicts"][0]["action"] == "FIX_ANSWER"
    assert report["verdicts"][0]["fields_to_fix"] == ["solution_certificate"]
    assert report["coverage"][0]["missing_ids"] == []


def test_general_repair_explicitly_nulls_stale_certificate(monkeypatch):
    qid = "model1_mcq_certificate"
    question = _mcq(qid, question="Old wording?", answer="A")
    question["solution_certificate"] = {
        "method": "old method", "important_givens": ["old given"],
        "calculation_steps": ["old calculation"], "final_answer": "A",
        "units": "", "matching_correct_option": "A",
        "verification_method": "old verification",
    }
    exam = _exam({"mcq": [question]})
    monkeypatch.setattr(val, "request_structured", lambda *_args, **_kwargs: {
        "repairs": [{
            "question_id": qid,
            "repaired_fields": ["question"],
            "question": {
                "question": "Repaired wording?",
                "options": dict(question["options"]),
                "correct_answer": "A",
                "solution_certificate": question["solution_certificate"],
            },
        }]
    })

    outcome = repair_exam(
        exam, [_verdict(qid, "FIX_QUESTION", qv=False, fields=["question"])],
        client=object(), repair_role="general",
    )

    assert outcome["repaired_ids"] == [qid]
    assert "solution_certificate" in exam["questions"]["mcq"][0]
    assert exam["questions"]["mcq"][0]["solution_certificate"] is None


def test_repaired_word_problems_receive_complete_quantitative_verdicts(monkeypatch):
    ids = ["model1_word_problem_1", "model1_word_problem_2"]
    items = []
    plans = {}
    for index, qid in enumerate(ids, start=1):
        item = _word_problem(qid)
        item["slot_id"] = f"m1_word_problem_{index}"
        items.append(item)
        plans.update(_quant_plan(item["slot_id"]))

    monkeypatch.setattr(
        val, "request_structured",
        lambda *_args, **_kwargs: {
            "verdicts": [_complete_quantitative_verdict(qid) for qid in ids]
        },
    )
    report = validate_exam(
        _exam({"word_problem": items}), client=object(), question_ids=ids,
        plan_by_slot=plans, source_by_chunk={"chunk-a": "supported relationship"},
        validation_role="stem_quantitative",
    )

    assert report["all_pass"]
    assert [verdict["question_id"] for verdict in report["verdicts"]] == ids


def test_why_revalidation_uses_latest_authoritative_plan_task(monkeypatch):
    qid = "model1_why_3"
    slot_id = "m1_why_3"
    question = _sa(
        qid,
        "Why does photon energy increase as frequency increases?",
        "Because E = hν and h is positive.",
    )
    question["slot_id"] = slot_id
    old = {
        "slot_id": slot_id, "task_id": "task-stale",
        "concept": "energy-level transition", "task_operation": "explain_transition",
        "target": "emitted photon energy", "reasoning_mode": "conceptual_reasoning",
        "task_family_id": "family-stale", "coverage_id": "coverage-stale",
        "source_chunk_id": "chunk-stale", "support_chunk_ids": [],
    }
    current = {
        "slot_id": slot_id, "task_id": "task-current",
        "concept": "Frequency and energy", "task_operation": "explain_relationship",
        "target": "Explain how radiation energy changes as frequency increases.",
        "reasoning_mode": "conceptual_reasoning", "task_family_id": "family-current",
        "coverage_id": "coverage-current", "source_chunk_id": "chunk-current",
        "support_chunk_ids": ["chunk-support"],
    }
    captured = {}

    def fake_request(_client, prompt, **_kwargs):
        captured["entry"] = _validator_entries(prompt)[0]
        return {"verdicts": [_pass_verdict(qid)]}

    monkeypatch.setattr(val, "request_structured", fake_request)
    monkeypatch.setattr(val, "create_llm_client", lambda **_kwargs: object())
    state = {
        "generated_exams": [_exam({"why": [question]})],
        "plans": [
            {"model_number": 1, "items": {"why": [old]}},
            {"model_number": 1, "items": {"why": [current]}},
        ],
        "retrieved_chunks": [
            {"child_id": "chunk-stale", "content": "stale evidence"},
            {"child_id": "chunk-current", "content": "E = hν"},
            {"child_id": "chunk-support", "content": "h is Planck's constant"},
        ],
        "validation_reports": [{
            "model_number": 1, "all_pass": False,
            "verdicts": [_verdict(qid, "FIX_QUESTION", qv=False)],
            "warnings": [], "coverage": [],
        }],
        "validated_models": [], "question_repair_attempts": {1: 1},
        "pending_revalidation_ids": {1: [qid]}, "warnings": [],
    }

    result = validate_generated_questions(state)
    entry = captured["entry"]
    assert result["validation_reports"][0]["verdicts"][0]["action"] == "PASS"
    assert entry["plan"]["task_id"] == "task-current"
    assert entry["expected_catalog_task"] == {
        "slot_id": slot_id, "task_id": "task-current",
        "task_family_id": "family-current", "coverage_id": "coverage-current",
        "concept_id": None, "relationship_id": None,
        "concept": "Frequency and energy", "knowledge_target": None,
        "student_operation": "explain_relationship",
        "target_or_result": "Explain how radiation energy changes as frequency increases.",
        "relationship": None, "reasoning_mode": "conceptual_reasoning",
        "required_output": None, "required_givens": [], "scenario_family": None,
        "difficulty": None,
        "assigned_evidence_ids": ["chunk-current", "chunk-support"],
    }
    assert entry["assigned_sources"] == {
        "chunk-current": "E = hν",
        "chunk-support": "h is Planck's constant",
    }


def test_repair_exam_pins_deepseek_flash_without_pro_config(monkeypatch):
    qid = "model1_why_1"
    question = _sa(qid)
    created = []
    client = object()

    def fake_factory(**kwargs):
        created.append(kwargs)
        return client

    monkeypatch.setattr(val, "create_llm_client", fake_factory)
    monkeypatch.setattr(val, "request_structured", lambda used_client, *_args, **_kwargs: {
        "repairs": [{
            "question_id": qid,
            "repaired_fields": ["reference_answer"],
            "question": {
                "question": question["question"],
                "reference_answer": "Peer review identifies errors before publication.",
            },
        }]
    } if used_client is client else None)

    result = repair_exam(
        _exam({"why": [question]}),
        [_verdict(qid, "FIX_ANSWER", av=False, fields=["reference_answer"])],
    )

    assert result["repaired_ids"] == [qid]
    assert created == [{
        "local_reasoning": val.REPAIR_REASONING,
        "deepseek_model": "deepseek-flash",
    }]


def _global_pass_result(ids):
    return {
        "verdicts": [
            {
                "question_id": qid,
                "action": "PASS",
                "failure_code": "pass",
                "conflicting_question_ids": [],
                "fields_to_fix": [],
                "reason": "",
                "expected_fix": "",
            }
            for qid in ids
        ],
        "conflict_groups": [],
        "exam_quality_findings": [],
    }


def _global_pass_review(exam, _plan=None, **_kwargs):
    ids = [
        str(item.get("question_id"))
        for section in (exam.get("questions") or {}).values()
        for item in ((section.get("items") or []) if isinstance(section, dict) else section)
        if item.get("question_id")
    ]
    return {"phase": _kwargs.get("phase", "initial"), "complete": True,
            **_global_pass_result(ids), "warnings": []}


def _global_failure_result(ids, failures, *, conflicts=None, findings=None):
    result = _global_pass_result(ids)
    by_id = {item["question_id"]: item for item in result["verdicts"]}
    for qid, code, conflict_ids, reason in failures:
        by_id[qid] = {
            "question_id": qid,
            "action": "FIX_QUESTION_AND_ANSWER",
            "failure_code": code,
            "conflicting_question_ids": conflict_ids,
            "fields_to_fix": ["question", "answer"],
            "reason": reason,
            "expected_fix": "Replace the underlying task with a distinct grounded task.",
        }
    result["verdicts"] = [by_id[qid] for qid in ids]
    result["conflict_groups"] = conflicts or []
    result["exam_quality_findings"] = findings or []
    return result


def _quant_plan(slot_id="m1_equation_1"):
    return {
        slot_id: {
            "slot_id": slot_id,
            "source_chunk_id": "chunk-a",
            "support_chunk_ids": [],
            "canonical_skill": "solve a linear system",
            "concept": "systems of equations",
            "relationship": "linear system",
            "target": "ordered pair",
            "task_operation": "solve_system",
            "given_conditions": ["5x + 3y = -19", "8x + 3y = -25"],
            "required_givens": ["5x + 3y = -19", "8x + 3y = -25"],
            "text_answerable": True,
        }
    }


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------
def test_assign_ids_unique_per_type():
    mcq1, mcq2 = _mcq("x1"), _mcq("x2")
    del mcq1["question_id"], mcq2["question_id"]
    eb._assign_ids(1, "mcq", [mcq1, mcq2])
    assert [q["question_id"] for q in (mcq1, mcq2)] == ["model1_mcq_1", "model1_mcq_2"]


def test_assign_ids_different_types_no_collision():
    mcq = _mcq("x")
    del mcq["question_id"]
    tf = _tf("x")
    del tf["question_id"]
    eb._assign_ids(1, "mcq", [mcq])
    eb._assign_ids(1, "true_false", [tf])
    ids = {mcq["question_id"], tf["question_id"]}
    assert ids == {"model1_mcq_1", "model1_true_false_1"}


def test_assign_ids_model_isolation():
    mcq1, mcq2 = _mcq("x"), _mcq("y")
    del mcq1["question_id"], mcq2["question_id"]
    eb._assign_ids(1, "mcq", [mcq1])
    eb._assign_ids(2, "mcq", [mcq2])
    assert mcq1["question_id"] == "model1_mcq_1"
    assert mcq2["question_id"] == "model2_mcq_1"
    assert mcq1["question_id"] != mcq2["question_id"]


def test_assign_ids_continues_sequence():
    q1 = _mcq("x")
    del q1["question_id"]
    eb._assign_ids(1, "mcq", [q1])
    q2 = _mcq("y")
    del q2["question_id"]
    eb._assign_ids(1, "mcq", [q1, q2])  # q2 has no id -> continues
    assert q1["question_id"] == "model1_mcq_1"
    assert q2["question_id"] == "model1_mcq_2"


def test_assign_ids_fitb_items():
    item = _fitb_item("x")
    del item["question_id"]
    eb._assign_ids(1, "fill_in_the_blank", _fitb([item]))
    assert item["question_id"] == "model1_fill_in_the_blank_1"


def test_id_survives_repair():
    exam = _exam({"true_false": [_tf("model1_true_false_1", answer="False")]})
    fake = FakeClient(
        repair_result=[
            {"question_id": "model1_true_false_1", "repaired_fields": ["answer"],
             "question": {"statement": "Cyanobacteria released oxygen through photosynthesis.", "answer": "True"}}
        ]
    )
    verdicts = [_verdict("model1_true_false_1", "FIX_ANSWER", fields=["answer"],
                         reason="should be True")]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["true_false"][0]
    assert q["question_id"] == "model1_true_false_1"
    assert q["answer"] == "True"


def test_display_numbering_independent_of_id():
    questions = {
        "true_false": [_tf("model1_true_false_1")],
        "mcq": [_mcq("model1_mcq_1")],
    }
    _ = eb.assemble_exam(questions)
    assert questions["true_false"][0]["question_id"] == "model1_true_false_1"
    assert questions["mcq"][0]["question_id"] == "model1_mcq_1"


# ---------------------------------------------------------------------------
# Validation: objective types
# ---------------------------------------------------------------------------
def test_valid_mcq_remains_unchanged():
    q = _mcq("model1_mcq_1")
    exam = _exam({"mcq": [q]})
    fake = FakeClient(validator_result=[_pass_verdict("model1_mcq_1")])
    report = validate_exam(exam, client=fake)
    assert report["all_pass"] is True
    assert zero_repair(fake)  # no repair call needed
    assert exam["questions"]["mcq"][0] is q  # same object, untouched


def test_wrong_mcq_answer_repaired_answer_only():
    exam = _exam({"mcq": [_mcq("model1_mcq_2", answer="B")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_mcq_2", "FIX_ANSWER", av=False, fields=["correct_answer"])],
        repair_result=[
            {"question_id": "model1_mcq_2", "repaired_fields": ["correct_answer"],
             "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                          "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
                          "correct_answer": "C"}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["mcq"][0]
    assert q["question_id"] == "model1_mcq_2"
    assert q["correct_answer"] == "C"
    assert q["question"] == "Which metric is the harmonic mean of precision and recall?"
    assert q["options"] == {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"}


def test_polygon_area_quantitative_repair_replaces_wrong_certificate_and_answer(monkeypatch):
    question = _mcq(
        "model1_mcq_polygon", answer="B",
        question="Find the area of the polygon with vertices (-6,8), (-3,-4), (5,0), (2,6).",
        options={"A": "75", "B": "71.02", "C": "60.434", "D": "150"},
    )
    question["solution_certificate"] = {
        "method": "shoelace formula", "important_givens": ["four listed vertices"],
        "calculation_steps": ["Apply shoelace", "area = 71.02"],
        "final_answer": "71.02", "units": "square units",
        "matching_correct_option": "B", "verification_method": "recompute shoelace",
    }
    exam = _exam({"mcq": [question]})
    repaired_certificate = {
        "method": "shoelace formula",
        "important_givens": ["(-6,8)", "(-3,-4)", "(5,0)", "(2,6)"],
        "calculation_steps": [
            "forward sum = 70", "reverse sum = -80",
            "area = |70 - (-80)| / 2 = 75",
        ],
        "final_answer": "75", "units": "square units",
        "matching_correct_option": "A",
        "verification_method": "repeat shoelace after cyclically rotating the vertex list",
    }
    repair = {
        "question_id": "model1_mcq_polygon",
        "repaired_fields": ["options", "correct_answer", "solution_certificate"],
        "question": {
            "question": question["question"], "options": dict(question["options"]),
            "correct_answer": "A", "solution_certificate": repaired_certificate,
        },
    }
    monkeypatch.setattr(val, "request_structured", lambda *_args, **_kwargs: {
        "repairs": [repair]
    })
    verdict = _verdict(
        "model1_mcq_polygon", "FIX_OPTIONS", av=False,
        fields=["options", "correct_answer", "solution_certificate"],
    )
    verdict.update({
        "validation_role": "stem_quantitative", "expected_results": ["75", "A"],
        "first_incorrect_step": "area = 71.02",
    })

    outcome = repair_exam(
        exam, [verdict], client=object(), repair_role="stem_quantitative"
    )

    repaired = exam["questions"]["mcq"][0]
    assert outcome["repaired_ids"] == ["model1_mcq_polygon"], outcome
    assert repaired["correct_answer"] == "A"
    assert repaired["solution_certificate"]["final_answer"] == "75"
    assert repaired["solution_certificate"]["matching_correct_option"] == "A"
    assert "75" in repaired["solution_certificate"]["calculation_steps"][-1]


def test_mcq_missing_correct_option_detected_locally():
    bad = _mcq("model1_mcq_3", answer="E")
    exam = _exam({"mcq": [bad]})
    report = validate_exam(exam, client=FakeClient())
    v = report["verdicts"][0]
    assert v["action"] == "FIX_OPTIONS"
    assert set(v["fields_to_fix"]) & {"options", "answer"}


def test_mcq_duplicate_option_detected_locally():
    bad = _mcq("model1_mcq_4", options={"A": "same", "B": "same", "C": "c", "D": "d"})
    exam = _exam({"mcq": [bad]})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_OPTIONS"


def test_mcq_two_valid_options_repairs_options():
    exam = _exam({"mcq": [_mcq("model1_mcq_5", answer="B")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_mcq_5", "FIX_OPTIONS", av=False,
                                   fields=["options", "correct_answer"])],
        repair_result=[
            {"question_id": "model1_mcq_5", "repaired_fields": ["options", "correct_answer"],
             "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                          "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                          "correct_answer": "A"}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["mcq"][0]
    assert q["options"] == {"A": "a", "B": "b", "C": "c", "D": "d"}
    assert q["correct_answer"] == "A"


def test_valid_tf_unchanged():
    q = _tf("model1_true_false_1")
    exam = _exam({"true_false": [q]})
    fake = FakeClient(validator_result=[_pass_verdict("model1_true_false_1")])
    report = validate_exam(exam, client=fake)
    assert report["all_pass"]
    assert exam["questions"]["true_false"][0] is q


def test_tf_wrong_answer_repaired_answer_only():
    exam = _exam({"true_false": [_tf("model1_true_false_2", answer="False")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_true_false_2", "FIX_ANSWER", av=False, fields=["answer"])],
        repair_result=[
            {"question_id": "model1_true_false_2", "repaired_fields": ["answer"],
             "question": {"statement": "Cyanobacteria released oxygen through photosynthesis.", "answer": "True"}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["true_false"][0]
    assert q["answer"] == "True"
    assert q["statement"] == "Cyanobacteria released oxygen through photosynthesis."
    assert q["question_id"] == "model1_true_false_2"


def test_malformed_tf_statement_repaired():
    exam = _exam({"true_false": [{"question_id": "model1_true_false_3", "statement": "", "answer": "True"}]})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_QUESTION"


def test_fitb_repair_cannot_change_authoritative_answer():
    exam = _exam({"fill_in_the_blank": _fitb([_fitb_item(
        "model1_fill_in_the_blank_1", answers=["perimeter"],
        distractors=["length", "volume"],
    )])})
    exam["questions"]["fill_in_the_blank"]["items"][0]["fitb_entry_id"] = "fitb_perimeter"
    fake = FakeClient(
        repair_result=[{
            "question_id": "model1_fill_in_the_blank_1",
            "repaired_fields": ["question"],
            "question": {
                "stem": "The distance around a shape is its ________.",
                "distractor_1": "length",
                "distractor_2": "volume",
            },
        }],
    )
    verdict = _verdict(
        "model1_fill_in_the_blank_1", "FIX_QUESTION", qv=False,
        fields=["question"],
    )
    outcome = repair_exam(exam, [verdict], client=fake)
    items = exam["questions"]["fill_in_the_blank"]["items"]
    assert outcome["repaired_ids"] == ["model1_fill_in_the_blank_1"]
    assert items[0]["answers"] == ["perimeter"]
    assert items[0]["fitb_entry_id"] == "fitb_perimeter"


def test_fitb_is_never_routed_to_quantitative_validation():
    item = _fitb_item("model1_fill_in_the_blank_1", answers=["catalyst"])
    item["slot_id"] = "m1_fill_in_the_blank_1"
    exam = _exam({"fill_in_the_blank": _fitb([item])})
    plan = {
        item["slot_id"]: {
            "slot_id": item["slot_id"],
            "fitb_entry_id": "fitb_catalyst",
            "reasoning_mode": "calculation",
            "task_operation": "calculate",
        }
    }
    assert val._quantitative_question_ids(exam, plan) == []


def test_fitb_answer_missing_from_bank_detected_locally():
    exam = _exam({"fill_in_the_blank": _fitb(
        [_fitb_item("model1_fill_in_the_blank_2", answers=["nowhere"])],
        bank=["area", "perimeter", "volume"],
    )})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_QUESTION_AND_ANSWER"


def test_solve_system_requires_every_variable_and_ordered_pair():
    item = _equation(
        "model1_equation_1",
        equation="Solve 5x + 3y = -19 and 8x + 3y = -25.",
        steps=["Subtract to obtain x = -2"],
        final="x = -2",
    )
    item["slot_id"] = "m1_equation_1"
    report = validate_exam(
        _exam({"equation": [item]}), client=FakeClient(),
        plan_by_slot=_quant_plan(), source_by_chunk={"chunk-a": "source"},
    )
    verdict = report["verdicts"][0]
    assert verdict["failure_code"] == "incomplete_solution"
    assert "missing y" in verdict["reason"]


def test_no_solution_system_does_not_require_values_for_each_variable():
    item = _equation(
        "model1_equation_1",
        equation="Solve 5x + 3y = 52 and 15x + 9y = 54.",
        steps=["Multiplying the first equation by 3 gives 0 = 102."],
        final="There is no solution; no ordered pair satisfies both equations.",
    )
    item["slot_id"] = "m1_equation_1"
    assert val._solve_system_completion_verdict(item, _quant_plan()["m1_equation_1"]) is None


def test_parseable_linear_system_is_checked_independently():
    item = _equation(
        "model1_equation_1",
        equation="Solve 5x + 3y = -19 and 8x + 3y = -25.",
        steps=["x = -2", "y = 3"],
        final="(x, y) = (-2, 3)",
    )
    item["slot_id"] = "m1_equation_1"
    report = validate_exam(
        _exam({"equation": [item]}), client=FakeClient(),
        plan_by_slot=_quant_plan(), source_by_chunk={"chunk-a": "source"},
    )
    verdict = report["verdicts"][0]
    assert verdict["failure_code"] == "incorrect_result"
    assert verdict["expected_results"] == ["x = -2", "y = -3"]


def test_final_ordered_pair_overrides_intermediate_substitution_expression():
    item = _equation(
        "model1_equation_1",
        equation="Solve 5x + 3y = 23 and 2x + y = 7.",
        steps=["y = 7 - 2x", "x = -2", "Substitute into y = 7 - 2x"],
        final="(-2, 11)",
    )
    item["slot_id"] = "m1_equation_1"
    assert val._solve_system_completion_verdict(item, _quant_plan()["m1_equation_1"]) is None


def test_break_even_request_does_not_require_unrequested_system_variable():
    item = _word_problem(
        "model1_word_problem_1",
        question=(
            "A company has revenue y = 40x and cost y = 20x + 400. "
            "Find the break-even customer count and the profit at 60 customers."
        ),
        steps=["40x = 20x + 400, so x = 20", "Profit at 60 = 2400 - 1600 = 800"],
        final="Break-even: 20 customers; profit at 60: AED 800",
    )
    plan = _quant_plan("m1_word_problem_1")["m1_word_problem_1"]
    assert val._solve_system_completion_verdict(item, plan) is None
    assert "nothing beyond it" in val.STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT
    assert "does not also require an unrequested y value" in val.STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT


def test_final_duplicate_check_is_global_across_question_types():
    equation = _equation("model1_equation_1", equation="Calculate wavelength for 3.44 x 10^9 Hz.")
    equation["slot_id"] = "m1_equation_1"
    problem = _word_problem(
        "model1_word_problem_1",
        question="A microwave has frequency 3.44 x 10^9 Hz. Calculate its wavelength.",
    )
    problem["slot_id"] = "m1_word_problem_1"
    common = {
        "source_chunk_id": "chunk-a", "support_chunk_ids": [],
        "canonical_skill": "calculate wavelength", "concept": "waves",
        "relationship": "c = lambda nu", "target": "wavelength",
        "task_operation": "calculate_from_frequency",
        "given_conditions": ["frequency 3.44 x 10^9 Hz"],
        "scenario_family": "microwave wavelength", "required_givens": ["frequency"],
        "text_answerable": True,
    }
    plans = {
        "m1_equation_1": {**common, "slot_id": "m1_equation_1"},
        "m1_word_problem_1": {**common, "slot_id": "m1_word_problem_1"},
    }
    fake = EchoValidatorClient()
    report = validate_exam(
        _exam({"equation": [equation], "word_problem": [problem]}),
        client=fake, plan_by_slot=plans, source_by_chunk={"chunk-a": "source"},
    )
    verdicts = {item["question_id"]: item for item in report["verdicts"]}
    assert verdicts["model1_word_problem_1"]["action"] == "PASS"
    assert fake.primary_batches == [["model1_equation_1", "model1_word_problem_1"]]


def test_final_duplicate_check_rejects_same_specific_givens_across_operations():
    first = _equation("model1_equation_1", equation="Classify 5x+3y=52 and 15x+9y=54.")
    first["slot_id"] = "m1_equation_1"
    second = _word_problem(
        "model1_word_problem_1",
        question="Use elimination on 5x+3y=52 and 15x+9y=54. How many solutions are there?",
    )
    second["slot_id"] = "m1_word_problem_1"
    shared = {
        "source_chunk_id": "chunk-a", "support_chunk_ids": [],
        "concept": "linear systems", "scenario_family": None,
        "required_givens": [], "text_answerable": True,
        "given_conditions": ["5x+3y=52", "15x+9y=54"],
    }
    plans = {
        "m1_equation_1": {
            **shared, "slot_id": "m1_equation_1", "canonical_skill": "classify system",
            "relationship": "coefficient ratios", "target": "system classification",
            "task_operation": "classify_linear_system",
        },
        "m1_word_problem_1": {
            **shared, "slot_id": "m1_word_problem_1", "canonical_skill": "determine solutions",
            "relationship": "elimination", "target": "number of solutions",
            "task_operation": "determine_number_of_solutions",
        },
    }
    report = validate_exam(
        _exam({"equation": [first], "word_problem": [second]}),
        client=EchoValidatorClient(), plan_by_slot=plans,
        source_by_chunk={"chunk-a": "source"},
    )
    verdicts = {item["question_id"]: item for item in report["verdicts"]}
    assert verdicts["model1_word_problem_1"]["action"] == "PASS"


def test_final_duplicate_check_ignores_symbol_indices_in_same_snell_calculation():
    equation = _equation(
        "model1_equation_1",
        equation=(
            "Write n1 sin theta1 = n2 sin theta2, then calculate theta2 for "
            "n1 = 1.00, n2 = 1.52, theta1 = 30.0 degrees."
        ),
        final="theta2 = 19.2 degrees",
    )
    equation["slot_id"] = "m1_equation_1"
    problem = _word_problem(
        "model1_word_problem_1",
        question=(
            "A ray enters glass at 30.0 degrees. Air has index 1.00 and glass "
            "has index 1.52. Find the refracted angle."
        ),
    )
    problem["final_answer"] = "19.2 degrees"
    problem["slot_id"] = "m1_word_problem_1"
    base = _quant_plan()["m1_equation_1"]
    plans = {
        "m1_equation_1": {
            **base, "slot_id": "m1_equation_1",
            "canonical_skill": "rearrange snell law",
            "task_operation": "rearrange_relationship",
            "target": "symbolic relationship",
            "given_conditions": ["n1=1.00", "n2=1.52", "theta1=30.0 degrees"],
        },
        "m1_word_problem_1": {
            **base, "slot_id": "m1_word_problem_1",
            "canonical_skill": "calculate refraction angle",
            "task_operation": "calculate_from_scenario",
            "target": "angle in glass",
            # Vary metadata wording to reproduce the live escape.
            "given_conditions": ["air and polished glass", "incident ray data"],
        },
    }
    report = validate_exam(
        _exam({"equation": [equation], "word_problem": [problem]}),
        client=EchoValidatorClient(), plan_by_slot=plans,
        source_by_chunk={"chunk-a": "source"},
    )
    verdicts = {item["question_id"]: item for item in report["verdicts"]}
    assert verdicts["model1_word_problem_1"]["action"] == "PASS"


def test_duplicate_failure_uses_full_task_replacement_mode():
    assert "duplicate_or_repetition" not in val.TASK_REPLACEMENT_FAILURES


def test_final_duplicate_check_compares_mcq_answer_with_definition_answer():
    definition = "تغيّر اتجاه الضوء عند مروره من وسط شفاف إلى وسط شفاف آخر."
    mcq = _mcq(
        "model1_mcq_1",
        question="أي العبارات الآتية تعبّر بدقة عن ظاهرة الانكسار؟",
        answer="B",
        options={"A": "امتصاص الضوء", "B": definition, "C": "انعكاس الضوء", "D": "تشتت الضوء"},
    )
    definition_item = _definition(
        "model1_definition_1", term="الانكسار", ref=definition,
    )
    verdicts = val._global_duplicate_verdicts(
        {"mcq": [mcq], "definition": [definition_item]}, {},
    )
    assert verdicts["model1_definition_1"]["failure_code"] == "duplicate_or_repetition"


def test_final_duplicate_check_catches_reversed_true_false_restatement():
    mcq = _mcq(
        "model1_mcq_1",
        question="كيف يتغير مسار الضوء عندما ينتقل من الهواء إلى الزجاج عند السطح؟",
        answer="B",
        options={
            "A": "ينعكس الضوء بالكامل عند السطح",
            "B": "ينحرف الضوء نحو العمود المقام على السطح",
            "C": "يتوقف الضوء داخل الزجاج",
            "D": "لا يتغير مسار الضوء",
        },
    )
    statement = _tf(
        "model1_true_false_1",
        statement="عندما ينتقل الضوء من الهواء إلى الزجاج فإنه ينحرف بعيدًا عن العمود المقام على السطح.",
        answer="False",
    )
    verdicts = val._global_duplicate_verdicts(
        {"mcq": [mcq], "true_false": [statement]}, {},
    )
    assert verdicts["model1_true_false_1"]["failure_code"] == "duplicate_or_repetition"


def test_fitb_item_repair_rejects_unauthorized_shared_bank_rewrite():
    item = _fitb_item("model1_fill_in_the_blank_1", answers=["area"])
    exam = _exam({"fill_in_the_blank": _fitb([item])})
    fake = FakeClient(repair_result=[{
        "question_id": "model1_fill_in_the_blank_1",
        "repaired_fields": ["answers"],
        "question": {
            "question": item["question"],
            "answers": ["perimeter"],
            "word_bank": ["INVENTED", "BROKEN"],
        },
    }])
    outcome = repair_exam(
        exam,
        [_verdict(
            "model1_fill_in_the_blank_1", "FIX_ANSWER", av=False,
            fields=["answers"], reason="wrong term",
        )],
        client=fake,
    )
    section = exam["questions"]["fill_in_the_blank"]
    assert "INVENTED" not in section["word_bank"]
    assert section["items"][0]["answers"] == ["area"]
    assert section["word_bank"] == ["area", "perimeter", "volume"]
    assert outcome["repaired_ids"] == []
    assert outcome["failed_ids"] == ["model1_fill_in_the_blank_1"]


def test_fitb_ambiguous_bank_repair_scoped():
    item1 = _fitb_item("model1_fill_in_the_blank_3", answers=["area"])
    item2 = _fitb_item("model1_fill_in_the_blank_4", answers=["perimeter"])
    exam = _exam({"fill_in_the_blank": _fitb([item1, item2])})
    fake = FakeClient(
        validator_result=[
            _verdict("model1_fill_in_the_blank_3", "FIX_OPTIONS", av=False, fields=["word_bank", "answer"]),
            _pass_verdict("model1_fill_in_the_blank_4"),
        ],
        repair_result=[
            {"question_id": "model1_fill_in_the_blank_3", "repaired_fields": ["word_bank", "answer"],
             "question": {"question": "To calculate the ________ of a rectangle, multiply length by width.",
                          "answers": ["area"], "word_bank": ["area", "perimeter", "length", "width", "square"]}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    section = exam["questions"]["fill_in_the_blank"]
    assert section["items"][0]["answers"] == ["area"]  # item repaired
    assert section["items"][1] is item2  # unrelated item untouched
    assert section["items"][1]["answers"] == ["perimeter"]


# ---------------------------------------------------------------------------
# Validation: writing types
# ---------------------------------------------------------------------------
def test_why_irrelevant_reference_repaired_answer_only():
    exam = _exam({"why": [_sa("model1_why_1")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_why_1", "FIX_ANSWER", av=False, fields=["reference_answer"])],
        repair_result=[
            {"question_id": "model1_why_1", "repaired_fields": ["reference_answer"],
             "question": {"question": "Why is peer review important?",
                          "reference_answer": "Peer review lets experts catch errors."}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["why"][0]
    assert q["reference_answer"] == "Peer review lets experts catch errors."
    assert q["question"] == "Why is peer review important?"


def test_essay_unrelated_answer_repaired():
    exam = _exam({"essay": [_essay("model1_essay_1")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_essay_1", "FIX_ANSWER", av=False, fields=["reference_answer"])],
        repair_result=[
            {"question_id": "model1_essay_1", "repaired_fields": ["reference_answer"],
             "question": {"question": "Compare precision and recall.",
                          "reference_answer": "A fixed answer.", "key_points": ["a", "b"]}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    assert exam["questions"]["essay"][0]["reference_answer"] == "A fixed answer."


def test_essay_reference_answer_does_not_authorize_key_point_rewrite():
    exam = _exam({"essay": [_essay("model1_essay_1",
                                   ref="A wrong answer.",
                                   key_points=["old", "stale"])]})
    fake = FakeClient(
        validator_result=[_verdict("model1_essay_1", "FIX_ANSWER", av=False,
                                   fields=["reference_answer"])],
        repair_result=[
            {"question_id": "model1_essay_1", "repaired_fields": ["reference_answer", "key_points"],
             "question": {"question": "Compare precision and recall.",
                          "reference_answer": "The right answer.",
                          "key_points": ["new", "consistent"]}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    outcome = repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["essay"][0]
    assert outcome["failed_ids"] == ["model1_essay_1"]
    assert q["reference_answer"] == "A wrong answer."
    assert q["key_points"] == ["old", "stale"]


def test_valid_writing_question_unchanged():
    q = _sa("model1_why_2")
    exam = _exam({"why": [q]})
    fake = FakeClient(validator_result=[_pass_verdict("model1_why_2")])
    report = validate_exam(exam, client=fake)
    assert report["all_pass"]
    assert exam["questions"]["why"][0] is q


def test_definition_bare_term_is_the_canonical_contract():
    exam = _exam({"definition": [_definition("model1_definition_1", term="Force")]})
    report = validate_exam(
        exam, client=FakeClient(validator_result=[_pass_verdict("model1_definition_1")])
    )
    verdict = report["verdicts"][0]
    assert verdict["action"] == "PASS"


def test_equation_missing_steps_is_rejected_locally():
    question = _equation("model1_equation_1")
    question["solution_steps"] = []
    exam = _exam({"equation": [question]})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_ANSWER"


def test_complete_equation_question_is_allowed():
    question = _equation(
        "model1_equation_1",
        equation="A 4 kg object is raised 7 m. Using g = 10 N/kg, find the gravitational potential energy gained.",
    )
    fake = FakeClient(validator_result=[_pass_verdict(question["question_id"])])
    report = validate_exam(_exam({"equation": [question]}), client=fake)
    assert report["all_pass"] is True


def test_equation_form_is_deferred_to_profile_aware_validator():
    question = _equation(
        "model1_equation_1",
        equation="F = ma; m = 6 kg; a = 3 m/s²",
        steps=["F = 6 × 3", "F = 18 N"],
        final="18 N",
    )
    fake = FakeClient(validator_result=[{
        **_pass_verdict(question["question_id"]),
        "question_valid": False,
        "action": "FIX_QUESTION",
        "fields_to_fix": ["equation"],
        "reason": "The prompt does not state the planned task operation.",
        "expected_fix": "State the operation required by the plan.",
    }])
    report = validate_exam(_exam({"equation": [question]}), client=fake)
    verdict = report["verdicts"][0]
    assert verdict["action"] == "FIX_QUESTION"
    assert verdict["fields_to_fix"] == ["equation"]


def test_word_problem_is_sent_as_its_own_validator_type():
    question = _word_problem("model1_word_problem_1")
    fake = FakeClient(validator_result=[_pass_verdict(question["question_id"])])
    report = validate_exam(_exam({"word_problem": [question]}), client=fake)
    assert report["all_pass"] is True
    assert '"question_type": "word_problem"' in fake.last_validator_prompt


def test_validator_receives_internal_plan_and_assigned_source_only():
    question = _equation(
        "model1_equation_1",
        equation="Using v = u + at and s = (u + v)t/2, show that v² = u² + 2as.",
    )
    question["slot_id"] = "m1_equation_1"
    fake = FakeClient(validator_result=[_pass_verdict(question["question_id"])])
    plan = {
        "m1_equation_1": {
            "source_chunk_id": "chunk-kinematics",
            "relationship": "kinematic equations",
            "task_operation": "derive_relationship",
            "target": "v² = u² + 2as",
            "reasoning_mode": "derivation_relationship",
            "difficulty": "medium",
            "idea_to_test": "Eliminate time and derive the third kinematic equation.",
        }
    }
    report = validate_exam(
        _exam({"equation": [question]}),
        client=fake,
        plan_by_slot=plan,
        source_by_chunk={
            "chunk-kinematics": "v = u + at; s = (u + v)t/2",
            "unassigned": "This content must not be sent.",
        },
    )
    assert report["all_pass"] is True
    entries = _validator_entries(fake.last_validator_prompt)
    assert entries[0]["plan"]["reasoning_mode"] == "derivation_relationship"
    assert entries[0]["assigned_sources"] == {
        "chunk-kinematics": "v = u + at; s = (u + v)t/2"
    }
    assert "v = u + at; s = (u + v)t/2" in fake.last_validator_prompt
    assert "This content must not be sent" not in fake.last_validator_prompt


def test_missing_answer_detected_locally():
    exam = _exam({"essay": [{"question_id": "model1_essay_2", "question": "Q?", "reference_answer": ""}]})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_ANSWER"


def test_empty_question_detected_locally():
    exam = _exam({"essay": [{"question_id": "model1_essay_3", "question": "  ", "reference_answer": "ans"}]})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_QUESTION"


def test_question_answer_mismatch_flagged_by_llm():
    exam = _exam({"why": [_sa("model1_why_3")]})
    fake = FakeClient(validator_result=[_verdict("model1_why_3", "FIX_ANSWER", av=False,
                                                 fields=["reference_answer"], reason="answer belongs to another question")])
    report = validate_exam(exam, client=fake)
    assert report["verdicts"][0]["action"] == "FIX_ANSWER"


# ---------------------------------------------------------------------------
# Missing verdicts: retry once, then remain operationally unvalidated
# ---------------------------------------------------------------------------
def test_missing_verdict_retried_once():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(validator_result=[],
                      validator_retry_result=[_pass_verdict("model1_mcq_1")])
    report = validate_exam(exam, client=fake)
    assert fake.retry_calls == 1
    assert report["all_pass"]
    v = {x["question_id"]: x for x in report["verdicts"]}["model1_mcq_1"]
    assert v["action"] == "PASS"


def test_missing_verdict_never_passed():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(validator_result=[], validator_retry_result=[])
    report = validate_exam(exam, client=fake)
    assert fake.retry_calls == 1
    assert not report["all_pass"]
    v = {x["question_id"]: x for x in report["verdicts"]}["model1_mcq_1"]
    assert v["action"] == "UNVALIDATED"
    assert v["fields_to_fix"] == []
    assert "no validator verdict" in v["reason"].lower()


def test_validator_splits_large_input_into_configured_exact_id_batches(monkeypatch):
    monkeypatch.setattr(val, "VALIDATOR_BATCH_SIZE", 8)
    ids = [f"model1_mcq_{i}" for i in range(1, 35)]
    exam = _exam({"mcq": [_mcq(qid, question=f"Question {qid}?") for qid in ids]})
    fake = EchoValidatorClient()

    report = validate_exam(exam, client=fake)

    # Calls complete concurrently, so inspect their canonical input order rather
    # than provider completion order.
    ordered_batches = sorted(
        fake.primary_batches, key=lambda batch: ids.index(batch[0])
    )
    assert [len(batch) for batch in ordered_batches] == [8, 8, 8, 8, 2]
    assert [qid for batch in ordered_batches for qid in batch] == ids
    assert fake.retry_batches == []
    assert [v["question_id"] for v in report["verdicts"]] == ids
    assert all(v["action"] == "PASS" for v in report["verdicts"])


def test_validator_batches_run_concurrently_and_merge_by_question_id(monkeypatch):
    import threading
    import time

    monkeypatch.setattr(val, "VALIDATOR_BATCH_SIZE", 1)
    monkeypatch.setattr(val, "MAX_PARALLEL_VALIDATOR_CALLS", 5)
    ids = [f"model1_mcq_{i}" for i in range(1, 6)]
    exam = _exam({"mcq": [_mcq(qid, question=f"Question {qid}?") for qid in ids]})
    active = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(5)

    def fake_request(_client, _prompt, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=2)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"verdicts": [_pass_verdict(qid) for qid in kwargs["expected_ids"]]}

    monkeypatch.setattr(val, "request_structured", fake_request)
    report = validate_exam(exam, client=object())

    assert peak == 5
    assert [verdict["question_id"] for verdict in report["verdicts"]] == ids
    assert all(verdict["action"] == "PASS" for verdict in report["verdicts"])


def test_batch_coverage_retries_only_missing_and_tracks_protocol_issues(monkeypatch):
    monkeypatch.setattr(val, "VALIDATOR_BATCH_SIZE", 3)
    ids = ["model1_mcq_1", "model1_mcq_2", "model1_mcq_3"]
    exam = _exam({"mcq": [_mcq(qid, question=f"Question {qid}?") for qid in ids]})

    class PartialClient:
        def __init__(self):
            self.retry_ids = []

        def chat_json(self, prompt, system_prompt=None, **kwargs):
            sent = [entry["question_id"] for entry in _validator_entries(prompt)]
            if system_prompt == VALIDATOR_SYSTEM_PROMPT:
                first = _pass_verdict(sent[0])
                return {"verdicts": [first, dict(first), _pass_verdict("unexpected_id")]}
            self.retry_ids.append(sent)
            return {"verdicts": [_pass_verdict(sent[0])]}

    fake = PartialClient()
    report = validate_exam(exam, client=fake)
    actions = {v["question_id"]: v["action"] for v in report["verdicts"]}

    assert fake.retry_ids == [["model1_mcq_1"], ["model1_mcq_2"], ["model1_mcq_3"]]
    assert actions == {
        "model1_mcq_1": "PASS",
        "model1_mcq_2": "PASS",
        "model1_mcq_3": "PASS",
    }
    assert report["coverage"] == [
        {
            "sent_ids": ids,
            "returned_ids": [
                "model1_mcq_1",
                "model1_mcq_1",
                "unexpected_id",
                "model1_mcq_1",
                "model1_mcq_2",
                "model1_mcq_3",
            ],
            "matched_ids": ids,
            "missing_ids": [],
            "unexpected_ids": ["unexpected_id"],
            "duplicate_returned_ids": ["model1_mcq_1"],
        }
    ]


def test_malformed_batch_preserves_valid_rows_and_retries_unresolved_ids_individually():
    ids = ["model1_mcq_1", "model1_mcq_2", "model1_mcq_3"]
    exam = _exam({"mcq": [_mcq(qid, question=f"Question {qid}?") for qid in ids]})

    class MalformedBatchClient:
        def __init__(self):
            self.retry_prompts = []
            self.primary_calls = 0

        def chat_json(self, prompt, system_prompt=None, **kwargs):
            sent = [entry["question_id"] for entry in _validator_entries(prompt)]
            if system_prompt == VALIDATOR_SYSTEM_PROMPT:
                self.primary_calls += 1
                malformed = _pass_verdict(ids[1])
                malformed.pop("expected_results")
                return {"verdicts": [_pass_verdict(ids[0]), malformed]}
            self.retry_prompts.append(prompt)
            return {"verdicts": [_pass_verdict(sent[0])]}

    fake = MalformedBatchClient()
    report = validate_exam(exam, client=fake)

    assert fake.primary_calls == 1
    assert [
        [entry["question_id"] for entry in _validator_entries(prompt)]
        for prompt in fake.retry_prompts
    ] == [[ids[1]], [ids[2]]]
    assert all("previous structured response was rejected" in prompt for prompt in fake.retry_prompts)
    assert all("schema validation failed" in prompt for prompt in fake.retry_prompts)
    assert report["all_pass"] is True
    assert report["coverage"][0]["matched_ids"] == ids
    assert report["coverage"][0]["missing_ids"] == []


# ---------------------------------------------------------------------------
# Complete-exam AI review: cross-type semantic regressions
# ---------------------------------------------------------------------------
def test_legacy_global_review_delegates_to_focused_actual_question_judge(monkeypatch):
    work_mcq = _mcq(
        "model1_mcq_1", "ما التعريف العلمي الدقيق للشغل في الفيزياء؟", "A",
        {"A": "انتقال للطاقة عند تأثير قوة خلال إزاحة", "B": "معدل بذل الشغل",
         "C": "كتلة الجسم", "D": "زمن الحركة"},
    )
    work_mcq["slot_id"] = "m1_mcq_1"
    definition = _definition(
        "model1_definition_1", "الشغل", "انتقال للطاقة عند تأثير قوة خلال إزاحة."
    )
    definition["slot_id"] = "m1_definition_1"
    zero_mcq = _mcq(
        "model1_mcq_2", "متى يساوي الشغل صفراً رغم وجود قوة وإزاحة؟", "B",
        {"A": "عندما تتوازيان", "B": "عندما تتعامدان", "C": "عند زيادة القوة",
         "D": "عند زيادة الإزاحة"},
    )
    zero_mcq["slot_id"] = "m1_mcq_2"
    zero_short = _sa(
        "model1_why_1",
        "فسّر لماذا يساوي الشغل صفراً عندما تكون القوة عمودية على الإزاحة.",
        "لأن جتا 90 درجة يساوي صفراً في العلاقة W = Fd cos θ.",
    )
    zero_short["slot_id"] = "m1_why_1"
    energy_tf = _tf("model1_true_false_1", "الشغل الكلي يساوي التغير في الطاقة الحركية W = ΔE.", "True")
    energy_tf["slot_id"] = "m1_true_false_1"
    energy_eq = _equation(
        "model1_equation_1", "اكتب العلاقة التي تربط الشغل الكلي بالتغير في الطاقة الحركية.",
        ["ينص مبدأ الشغل والطاقة على العلاقة المطلوبة."], "W = ΔE",
    )
    energy_eq["slot_id"] = "m1_equation_1"
    snell_symbolic = _equation(
        "model1_equation_2", "أعد ترتيب قانون سنل لإيجاد زاوية الانكسار θ₂.",
        ["sin θ₂ = n₁ sin θ₁ / n₂"], "θ₂ = sin⁻¹(n₁ sin θ₁ / n₂)",
    )
    snell_symbolic["slot_id"] = "m1_equation_2"
    snell_numeric = _word_problem(
        "model1_word_problem_1",
        "ينتقل الضوء من الهواء إلى الزجاج بزاوية 30°. احسب زاوية الانكسار.",
        ["n₁ sin θ₁ = n₂ sin θ₂"], "19.2°",
    )
    snell_numeric["slot_id"] = "m1_word_problem_1"
    snell_valid = _word_problem(
        "model1_word_problem_2",
        "زاويتا السقوط والانكسار 30° و20°. احسب معامل انكسار الوسط المجهول.",
        ["n₂ = n₁ sin θ₁ / sin θ₂"], "1.46",
    )
    snell_valid["slot_id"] = "m1_word_problem_2"
    exam = _exam({
        "mcq": [work_mcq, zero_mcq], "definition": [definition],
        "why": [zero_short], "true_false": [energy_tf],
        "equation": [energy_eq, snell_symbolic],
        "word_problem": [snell_numeric, snell_valid],
    })
    ids = [
        item["question_id"]
        for section in exam["questions"].values()
        for item in section
    ]
    failures = [
        ("model1_definition_1", "duplicate_or_repetition", ["model1_mcq_1"], "Defines the same term recalled by the MCQ."),
        ("model1_why_1", "duplicate_or_repetition", ["model1_mcq_2"], "Explains the same perpendicular-force zero-work fact."),
        ("model1_equation_1", "duplicate_or_repetition", ["model1_true_false_1"], "Restates W = ΔE."),
        ("model1_word_problem_1", "duplicate_or_repetition", ["model1_equation_2"], "Uses the same Snell target."),
    ]
    conflicts = [
        {"question_ids": [qid, conflict[0]], "issue_type": "conceptual_duplicate", "reason": reason}
        for qid, _code, conflict, reason in failures
    ]
    focused = {
        "candidate_pairs": [{"pair_id": "pair_1_2"}],
        "pair_verdicts": [{"pair_id": "pair_1_2", "classification": "duplicate"}],
        "duplicate_groups": [{"group_id": "group_1"}],
        "similarity_distribution": {"pair_count": 36},
        "usage": {"total_tokens": 123},
    }
    calls = []
    monkeypatch.setattr(
        val, "review_duplicate_candidates",
        lambda actual_exam, client=None: calls.append((actual_exam, client)) or focused.copy(),
        raising=False,
    )
    # The import is local inside the compatibility wrapper, so patch the
    # focused module function that it resolves at call time.
    import app.online.duplicate_judge as duplicate_judge
    monkeypatch.setattr(
        duplicate_judge, "review_duplicate_candidates",
        lambda actual_exam, client=None: calls.append((actual_exam, client)) or focused.copy(),
    )
    fake = FakeClient(global_result=_global_failure_result(ids, failures, conflicts=conflicts))
    review = review_exam_globally(exam, client=fake)

    assert calls == [(exam, fake)]
    assert review["pair_verdicts"][0]["classification"] == "duplicate"
    assert review["phase"] == "initial"
    assert "verdicts" not in review


def test_legacy_global_review_no_longer_runs_language_or_answer_pattern_checks(monkeypatch):
    tf_items = [_tf(f"model1_true_false_{i}", f"عبارة فيزيائية رقم {i}.", "True") for i in range(1, 5)]
    definition = _definition("model1_definition_1", "الطاقة", "الطاقة هي الطاقة الموجودة في الجسم.")
    exam = _exam({"true_false": tf_items, "definition": [definition]})
    ids = [*(item["question_id"] for item in tf_items), definition["question_id"]]
    findings = [{
        "code": "true_false_answer_pattern", "severity": "error",
        "question_ids": [item["question_id"] for item in tf_items],
        "reason": "All True/False answers are True.",
        "expected_fix": "Replace one grounded statement without changing truth arbitrarily.",
    }, {
        "code": "arabic_language_quality", "severity": "error",
        "question_ids": [definition["question_id"]],
        "reason": "The definition is circular Arabic.",
        "expected_fix": "Give a non-circular definition.",
    }]
    failures = [
        ("model1_true_false_4", "answer_pattern_imbalance", [], "All answers have the same pattern."),
        ("model1_definition_1", "language_quality", [], "Circular definition."),
    ]
    import app.online.duplicate_judge as duplicate_judge
    monkeypatch.setattr(duplicate_judge, "review_duplicate_candidates", lambda *_args, **_kwargs: {
        "candidate_pairs": [], "pair_verdicts": [], "duplicate_groups": [],
        "similarity_distribution": {"pair_count": 10}, "usage": {},
    })
    review = review_exam_globally(
        exam, client=FakeClient(global_result=_global_failure_result(ids, failures, findings=findings))
    )
    assert review["pair_verdicts"] == []
    assert "exam_quality_findings" not in review
    assert "verdicts" not in review


def test_final_global_review_catches_repaired_question_creating_new_duplicate(monkeypatch):
    q1 = _sa("model1_why_1", "اشرح مفهوم الشغل.", "انتقال للطاقة بفعل قوة وإزاحة.")
    q2 = _definition("model1_definition_1", "الشغل", "انتقال للطاقة بفعل قوة وإزاحة.")
    q3 = _sa("model1_why_2", "اشرح القدرة.", "معدل بذل الشغل.")
    for q, sid in [(q1, "m1_why_1"), (q2, "m1_definition_1"), (q3, "m1_why_2")]:
        q["slot_id"] = sid
    exam = _exam({"why": [q1, q3], "definition": [q2]})

    def fake_bounded(_exam_value, client=None, question_ids=None, **_kwargs):
        selected = set(question_ids or [q1["question_id"], q2["question_id"], q3["question_id"]])
        verdicts = [_pass_verdict(qid) for qid in [q1["question_id"], q2["question_id"], q3["question_id"]] if qid in selected]
        return {"model_number": 1, "all_pass": True, "verdicts": verdicts, "warnings": [], "coverage": []}

    def fake_global(_exam_value, _plan=None, *, phase="initial", **_kwargs):
        ids = [q1["question_id"], q3["question_id"], q2["question_id"]]
        conflict = q1["question_id"] if phase == "initial" else q3["question_id"]
        result = _global_failure_result(ids, [
            (q2["question_id"], "duplicate_or_repetition", [conflict], "Duplicate after replacement.")
        ])
        return {"phase": phase, "complete": True, **result, "warnings": []}

    monkeypatch.setattr(val, "validate_exam", fake_bounded)
    monkeypatch.setattr(val, "review_exam_globally", fake_global)
    state = {
        "generated_exams": [exam], "validation_reports": [], "validated_models": [],
        "question_repair_attempts": {}, "pending_revalidation_ids": {}, "warnings": [],
    }
    state = {**state, **validate_generated_questions(state)}
    state["pending_revalidation_ids"] = {1: [q2["question_id"]]}
    state["question_repair_attempts"] = {1: 1}
    state = {**state, **validate_generated_questions(state)}
    verdict = {item["question_id"]: item for item in state["validation_reports"][0]["verdicts"]}[q2["question_id"]]
    assert verdict["action"] == "PASS"
    assert state["final_global_reviews"] == []


def test_global_ai_adjudicates_soft_similarity_but_not_hard_duplicate_identity():
    global_pass = {
        "phase": "initial", "complete": True,
        "verdicts": [{
            "question_id": "q2", "action": "PASS", "failure_code": "pass",
            "conflicting_question_ids": [], "fields_to_fix": [], "reason": "", "expected_fix": "",
        }],
        "conflict_groups": [], "exam_quality_findings": [], "warnings": [],
    }
    base = _verdict("q2", "FIX_QUESTION_AND_ANSWER", qv=False, av=False)
    base.update({
        "failure_code": "duplicate_or_repetition",
        "conflicting_question_id": "q1",
        "duplicate_bases": ["answer_aware_restatement"],
    })
    bounded = {"model_number": 1, "all_pass": False, "verdicts": [base], "warnings": []}

    soft = copy.deepcopy(bounded)
    soft["verdicts"][0]["duplicate_strength"] = "soft"
    soft_result = val._merge_global_review(soft, global_pass)
    assert soft_result["all_pass"] is True
    assert soft_result["verdicts"][0]["validation_sources"] == [
        "deterministic_advisory", "global_ai"
    ]

    hard = copy.deepcopy(bounded)
    hard["verdicts"][0]["duplicate_strength"] = "hard"
    hard_result = val._merge_global_review(hard, global_pass)
    assert hard_result["all_pass"] is False
    assert hard_result["verdicts"][0]["failure_code"] == "duplicate_or_repetition"


# ---------------------------------------------------------------------------
# Repair: strict ID-keyed mapping + field enforcement
# ---------------------------------------------------------------------------
def test_repair_batches_only_invalid_questions():
    exam = _exam({
        "mcq": [_mcq("model1_mcq_1", answer="B"), _mcq("model1_mcq_2", answer="B")],
        "true_false": [_tf("model1_true_false_1")],
    })
    fake = FakeClient(repair_result=[])
    verdicts = [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"]),
                _pass_verdict("model1_mcq_2"),
                _pass_verdict("model1_true_false_1")]
    repair_exam(exam, verdicts, client=fake)
    ids = json_ids(fake.last_repair_prompt)
    assert "model1_mcq_1" in ids  # the invalid question IS batched
    assert "model1_mcq_2" not in ids  # valid questions excluded
    assert "model1_true_false_1" not in ids


def test_unvalidated_is_never_sent_to_repair():
    exam = _exam({"mcq": [_mcq("missing"), _mcq("bad")]})
    fake = FakeClient(repair_result=[])
    repair_exam(
        exam,
        [
            _verdict("missing", "UNVALIDATED", qv=False, av=False),
            _verdict("bad", "FIX_ANSWER", av=False, fields=["answer"]),
        ],
        client=fake,
    )
    ids = json_ids(fake.last_repair_prompt)
    assert "missing" not in ids
    assert "bad" in ids


def test_unvalidated_only_report_does_not_enter_repair_node(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("repair_exam must not run for UNVALIDATED")

    monkeypatch.setattr(val, "repair_exam", should_not_run)
    state = {
        "generated_exams": [_exam({"mcq": [_mcq("missing")]})],
        "validation_reports": [
            {
                "model_number": 1,
                "all_pass": False,
                "verdicts": [_verdict("missing", "UNVALIDATED", qv=False, av=False)],
                "warnings": [],
            }
        ],
        "validated_models": [],
        "question_repair_attempts": {},
        "warnings": [],
    }

    out = repair_invalid_questions(state)
    assert out["question_repair_attempts"] == {}
    assert _route_after_validation(state) == "scan_exam_duplicates"


def test_real_content_failure_still_routes_to_repair():
    state = {
        "validation_reports": [
            {
                "model_number": 1,
                "all_pass": False,
                "verdicts": [_verdict("bad", "FIX_ANSWER", av=False)],
            }
        ],
        "question_repair_attempts": {},
    }
    assert _route_after_validation(state) == "repair_invalid_questions"


def test_missing_id_in_repair_response_rejected():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(repair_result=[{"repaired_fields": ["answer"],
                                      "question": {"question": "Q", "options": {}, "correct_answer": "A"}}])
    outcome = repair_exam(exam, [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert outcome["failed_ids"] == ["model1_mcq_1"]
    assert exam["questions"]["mcq"][0]["correct_answer"] == "B"  # original kept


def test_unknown_id_in_repair_response_rejected():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model9_mcq_9", "repaired_fields": ["answer"],
         "question": {"question": "Q", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct_answer": "A"}}
    ])
    outcome = repair_exam(exam, [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert "model1_mcq_1" in outcome["failed_ids"]
    assert exam["questions"]["mcq"][0]["correct_answer"] == "B"


def test_duplicate_id_in_repair_response_rejected():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    good = {"question_id": "model1_mcq_1", "repaired_fields": ["correct_answer"],
            "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                         "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
                         "correct_answer": "A"}}
    fake = FakeClient(repair_result=[good, good])
    outcome = repair_exam(exam, [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["correct_answer"])], client=fake)
    assert "model1_mcq_1" in outcome["repaired_ids"]  # first accepted
    assert not outcome["failed_ids"]


def test_disallowed_field_change_rejected():
    exam = _exam({"mcq": [_mcq("model1_mcq_2", answer="B", question="Original stem")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_2", "repaired_fields": ["answer"],
         "question": {"question": "CHANGED STEM", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                      "correct_answer": "C"}}
    ])
    outcome = repair_exam(exam, [_verdict("model1_mcq_2", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert outcome["failed_ids"] == ["model1_mcq_2"]  # stem touched -> rejected
    q = exam["questions"]["mcq"][0]
    assert q["question"] == "Original stem"
    assert q["correct_answer"] == "B"


def test_word_problem_wrapper_path_authorizes_fix_question_stem():
    original = _word_problem(
        "model1_word_problem_1",
        question="A shop has two cost rules. Find the requested quantity.",
    )
    repaired = _word_problem(
        "model1_word_problem_1",
        question="A shop charges AED 20 per item plus AED 40. Find the cost of 3 items.",
        steps=original["solution_steps"],
        final=original["final_answer"],
    )
    exam = _exam({"word_problem": [original]})
    fake = FakeClient(repair_result=[{
        "question_id": "model1_word_problem_1",
        "repaired_fields": ["question.question"],
        "question": repaired,
    }])
    verdict = _verdict(
        "model1_word_problem_1", "FIX_QUESTION", qv=False,
        fields=["question.question", "answer", "solution_certificate"],
    )
    outcome = repair_exam(exam, [verdict], client=fake)
    assert outcome["repaired_ids"] == ["model1_word_problem_1"]
    assert exam["questions"]["word_problem"][0]["question"] == repaired["question"]


def test_recognized_question_wrapper_paths_normalize_to_canonical_stems():
    cases = [
        ("word_problem", "question.question", "question"),
        ("true_false", "question.statement", "statement"),
        ("definition", "question.term", "term"),
        ("equation", "question.equation", "equation"),
    ]
    for qtype, path, expected in cases:
        assert expected in val._allowed_canonical(qtype, [path], "FIX_QUESTION")


def test_answer_only_action_blocks_stem_even_with_wrapper_path_requested():
    exam = _exam({"word_problem": [_word_problem(
        "model1_word_problem_1", question="Original student request.", final="wrong"
    )]})
    fake = FakeClient(repair_result=[{
        "question_id": "model1_word_problem_1",
        "repaired_fields": ["question.question", "final_answer"],
        "question": _word_problem(
            "model1_word_problem_1", question="Unauthorized changed request.", final="correct"
        ),
    }])
    verdict = _verdict(
        "model1_word_problem_1", "FIX_ANSWER", av=False,
        fields=["question.question", "final_answer"],
    )
    outcome = repair_exam(exam, [verdict], client=fake)
    assert outcome["failed_ids"] == ["model1_word_problem_1"]
    assert exam["questions"]["word_problem"][0]["question"] == "Original student request."


def test_general_validator_receives_task_contract_and_accepts_task_drift_code():
    q = _sa("model1_why_1", question="Why does checking one point fail to identify the common solution region?")
    q["slot_id"] = "m1_why_1"
    plan = {
        "m1_why_1": {
            "slot_id": "m1_why_1", "task_id": "task-region",
            "task_family_id": "family-region", "coverage_id": "coverage-region",
            "concept_id": "concept-region", "relationship_id": "relationship-region",
            "source_chunk_id": "chunk-a", "support_chunk_ids": [],
            "knowledge_focus": "common solution region",
            "concept": "solution region of a system of inequalities",
            "task_operation": "interpret_region",
            "target": "identify the common solution region",
            "reasoning_mode": "interpretation",
            "student_work_product": "description of the common region",
            "required_givens": ["two inequalities"],
        }
    }
    fake = FakeClient(validator_result=[{
        "question_id": "model1_why_1",
        "question_valid": False, "answer_valid": True,
        "action": "FIX_QUESTION", "fields_to_fix": ["question"],
        "failure_code": "task_drift", "expected_results": [],
        "reason": "Expected interpreting the region; actual task verifies one point.",
        "expected_fix": "Restore the assigned interpretation task.",
    }])
    report = validate_exam(
        _exam({"why": [q]}), client=fake, plan_by_slot=plan,
        source_by_chunk={"chunk-a": "Systems of inequalities have common solution regions."},
    )
    assert report["verdicts"][0]["failure_code"] == "task_drift"
    entry = _validator_entries(fake.last_validator_prompt)[0]
    assert entry["expected_catalog_task"] == {
        "slot_id": "m1_why_1",
        "task_id": "task-region",
        "task_family_id": "family-region",
        "coverage_id": "coverage-region",
        "concept_id": "concept-region",
        "relationship_id": "relationship-region",
        "concept": "solution region of a system of inequalities",
        "knowledge_target": "common solution region",
        "student_operation": "interpret_region",
        "target_or_result": "identify the common solution region",
        "relationship": None,
        "reasoning_mode": "interpretation",
        "required_output": "description of the common region",
        "required_givens": ["two inequalities"],
        "scenario_family": None,
        "difficulty": None,
        "assigned_evidence_ids": ["chunk-a"],
    }
    assert "Being on the same broad topic is not enough" in val.VALIDATOR_SYSTEM_PROMPT


def test_general_validator_uses_full_output_ceiling_without_cap_retry():
    fake = FakeClient(validator_result=[_pass_verdict("model1_why_1")])
    report = validate_exam(
        _exam({"why": [_sa("model1_why_1")]}), client=fake
    )
    assert report["all_pass"]
    assert fake.validate_calls == 1
    assert fake.retry_calls == 0
    assert fake.last_max_tokens == val.GENERAL_VALIDATOR_MAX_TOKENS == 100_000


def test_noop_repair_rejected():
    opts = {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"}
    exam = _exam({"mcq": [_mcq("model1_mcq_2", answer="B", question="Original stem", options=opts)]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_2", "repaired_fields": ["answer"],
         "question": {"question": "Original stem", "options": dict(opts),
                      "correct_answer": "B"}}  # byte-identical to the original
    ])
    outcome = repair_exam(exam, [_verdict("model1_mcq_2", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert outcome["failed_ids"] == ["model1_mcq_2"]
    assert any("no change" in w for w in outcome["warnings"])
    assert exam["questions"]["mcq"][0]["correct_answer"] == "B"


def test_sanitize_preserves_explicit_fix_options():
    raw = {"question_id": "model1_mcq_2", "question_valid": True, "answer_valid": True,
           "action": "FIX_OPTIONS", "fields_to_fix": ["options", "answer"],
           "reason": "Two options are correct.", "expected_fix": "Rewrite one distractor."}
    v = val._sanitize_verdict(raw, {"model1_mcq_2"})
    assert v["action"] == "FIX_OPTIONS"
    assert v["fields_to_fix"] == ["options", "answer"]


def test_sanitize_rejects_unknown_action_instead_of_turning_it_into_pass():
    raw = {
        "question_id": "model1_mcq_2",
        "question_valid": True,
        "answer_valid": True,
        "action": "NO_ACTION",
        "fields_to_fix": [],
    }
    assert val._sanitize_verdict(raw, {"model1_mcq_2"}) is None


def test_explicit_expected_tf_answer_resolves_self_contradictory_fix_verdict():
    verdict = {
        "question_id": "model1_true_false_1",
        "question_valid": True,
        "answer_valid": False,
        "action": "FIX_ANSWER",
        "fields_to_fix": ["answer"],
        "failure_code": "incorrect_result",
        "expected_results": ["False"],
        "reason": "The stored False answer is actually correct.",
        "expected_fix": "No repair is needed.",
    }
    entry = {
        "question_type": "true_false",
        "question": {"statement": "n can be below one.", "answer": "False"},
    }
    fixed = val._reconcile_simple_expected_answer(verdict, entry)
    assert fixed["action"] == "PASS"
    assert fixed["answer_valid"] is True
    assert fixed["fields_to_fix"] == []


def test_mcq_ambiguous_options_repair_accepted():
    exam = _exam({"mcq": [_mcq("model1_mcq_2",
                               question="Which of the following numbers is prime?",
                               options={"A": "2", "B": "4", "C": "7", "D": "9"},
                               answer="B")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_2", "repaired_fields": ["options", "correct_answer"],
         "question": {"question": "Which of the following numbers is prime?",
                      "options": {"A": "2", "B": "4", "C": "6", "D": "9"},
                      "correct_answer": "A"}}  # exactly one correct option remains
    ])
    verdicts = [_verdict("model1_mcq_2", "FIX_OPTIONS", av=False, fields=["options", "correct_answer"])]
    outcome = repair_exam(exam, verdicts, client=fake)
    assert outcome["repaired_ids"] == ["model1_mcq_2"]
    q = exam["questions"]["mcq"][0]
    assert q["options"]["C"] == "6"
    assert q["correct_answer"] == "A"


def test_mcq_missing_answer_option_repair_accepted():
    exam = _exam({"mcq": [_mcq("model1_mcq_4",
                               question="Which organelle produces most of a cell's ATP?",
                               options={"A": "Nucleus", "B": "Ribosome",
                                        "C": "Golgi apparatus", "D": "Lysosome"},
                               answer="A")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_4", "repaired_fields": ["options", "correct_answer"],
         "question": {"question": "Which organelle produces most of a cell's ATP?",
                      "options": {"A": "Nucleus", "B": "Mitochondria",
                                  "C": "Golgi apparatus", "D": "Lysosome"},
                      "correct_answer": "B"}}  # real answer added to the set
    ])
    verdicts = [_verdict("model1_mcq_4", "FIX_OPTIONS", av=False, fields=["options", "correct_answer"])]
    outcome = repair_exam(exam, verdicts, client=fake)
    assert outcome["repaired_ids"] == ["model1_mcq_4"]
    q = exam["questions"]["mcq"][0]
    assert q["options"]["B"] == "Mitochondria"
    assert q["correct_answer"] == "B"


def test_mcq_fix_options_rejects_reordering_known_bad_distractors():
    options = {
        "A": "Pressure waves in fluids",
        "B": "Mechanical sound waves",
        "C": "Electromagnetic radiation",
        "D": "Thermal vibrations of matter",
    }
    exam = _exam({"mcq": [_mcq(
        "m1_mcq_5", question="Which radiation includes visible light?",
        options=options, answer="C",
    )]})
    fake = FakeClient(repair_result=[{
        "question_id": "m1_mcq_5", "repaired_fields": ["options", "correct_answer"],
        "question": {
            "question": "Which radiation includes visible light?",
            "options": {"A": options["B"], "B": options["A"], "C": options["C"], "D": options["D"]},
            "correct_answer": "C",
        },
    }])
    verdict = _verdict(
        "m1_mcq_5", "FIX_OPTIONS", av=False,
        fields=["options", "correct_answer"],
    )
    verdict.update({
        "reason": "The distractors are unrelated and do not compete with the answer.",
        "expected_fix": "Use plausible same-topic misconceptions.",
    })

    outcome = repair_exam(exam, [verdict], client=fake)

    assert outcome["failed_ids"] == ["m1_mcq_5"]
    assert any("retained rejected distractor" in warning for warning in outcome["warnings"])
    payload = json.loads(fake.last_repair_prompt.split("## Items to repair\n", 1)[1].split(
        "\n\n## Repair object format", 1
    )[0])
    assert payload[0]["required_repair"] == "replace_all_rejected_distractors"
    assert set(payload[0]["rejected_distractors"]) == {options["A"], options["B"], options["D"]}


def test_mcq_fix_options_accepts_new_same_concept_distractors():
    options = {
        "A": "Pressure waves in fluids", "B": "Mechanical sound waves",
        "C": "Electromagnetic radiation", "D": "Thermal vibrations of matter",
    }
    exam = _exam({"mcq": [_mcq(
        "m1_mcq_5", question="Which radiation includes visible light?",
        options=options, answer="C",
    )]})
    fake = FakeClient(repair_result=[{
        "question_id": "m1_mcq_5", "repaired_fields": ["options", "correct_answer"],
        "question": {
            "question": "Which radiation includes visible light?",
            "options": {
                "A": "Infrared radiation only", "B": "Ultraviolet radiation only",
                "C": "Electromagnetic radiation", "D": "Ionizing radiation only",
            },
            "correct_answer": "C",
        },
    }])
    verdict = _verdict(
        "m1_mcq_5", "FIX_OPTIONS", av=False,
        fields=["options", "correct_answer"],
    )
    verdict.update({
        "reason": "The distractors are unrelated and do not compete with the answer.",
        "expected_fix": "Use plausible same-topic misconceptions.",
    })

    outcome = repair_exam(exam, [verdict], client=fake)

    assert outcome["repaired_ids"] == ["m1_mcq_5"]
    assert exam["questions"]["mcq"][0]["options"]["A"] == "Infrared radiation only"
    assert "rejected_distractors" in REPAIR_SYSTEM_PROMPT


def test_mcq_answer_verdict_does_not_authorize_option_rewrite():
    exam = _exam({"mcq": [_mcq("model1_mcq_4",
                               question="Which organelle produces most of a cell's ATP?",
                               options={"A": "Nucleus", "B": "Ribosome",
                                        "C": "Golgi apparatus", "D": "Lysosome"},
                               answer="A")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_4", "repaired_fields": ["correct_answer"],
         "question": {"question": "Which organelle produces most of a cell's ATP?",
                      "options": {"A": "Nucleus", "B": "Mitochondria",
                                  "C": "Golgi apparatus", "D": "Lysosome"},
                      "correct_answer": "B"}}  # options+answer coupled, stem untouched
    ])
    verdicts = [_verdict("model1_mcq_4", "FIX_ANSWER", av=False, fields=["correct_answer"])]
    outcome = repair_exam(exam, verdicts, client=fake)
    assert outcome["failed_ids"] == ["model1_mcq_4"]
    q = exam["questions"]["mcq"][0]
    assert q["options"]["B"] == "Ribosome"
    assert q["correct_answer"] == "A"
    assert q["question"] == "Which organelle produces most of a cell's ATP?"


def test_repair_normalizes_before_write_back():
    exam = _exam({"true_false": [_tf("model1_true_false_1", answer="False")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_true_false_1", "repaired_fields": ["answer"],
         "question": {"statement": "Cyanobacteria released oxygen through photosynthesis.", "answer": "true"}}
    ])
    repair_exam(exam, [_verdict("model1_true_false_1", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert exam["questions"]["true_false"][0]["answer"] == "True"  # normalized via normalizer


def test_valid_questions_not_touched_when_sibling_repairs():
    q_valid = _mcq("model1_mcq_3", question="Keep me")
    exam = _exam({"mcq": [_mcq("model1_mcq_2", answer="B"), q_valid]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_2", "repaired_fields": ["answer"],
         "question": {"question": "Q2", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                      "correct_answer": "C"}}
    ])
    before = copy.deepcopy(q_valid)
    repair_exam(exam, [_verdict("model1_mcq_2", "FIX_ANSWER", av=False, fields=["answer"]),
                       _pass_verdict("model1_mcq_3")], client=fake)
    assert exam["questions"]["mcq"][1] is q_valid
    assert q_valid == before


def test_repair_failure_keeps_original_for_revalidation():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(repair_result=None)  # unusable array
    outcome = repair_exam(exam, [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert outcome["failed_ids"] == ["model1_mcq_1"]
    assert exam["questions"]["mcq"][0]["correct_answer"] == "B"


# ---------------------------------------------------------------------------
# Graph nodes: revalidation, bounded and per-model attempts, isolation
# ---------------------------------------------------------------------------
def test_selective_revalidation_merges_untouched_passes(monkeypatch):
    all_ids = [f"model1_mcq_{index}" for index in range(1, 11)]
    invalid_ids = all_ids[-3:]
    exam = _exam({"mcq": [_mcq(qid) for qid in all_ids]})
    validation_scopes = []

    def fake_validate(_exam_value, client=None, question_ids=None):
        scope = None if question_ids is None else list(question_ids)
        validation_scopes.append(scope)
        ids = all_ids if scope is None else scope
        verdicts = [
            (
                _verdict(qid, "FIX_ANSWER", av=False, fields=["answer"])
                if scope is None and qid in invalid_ids
                else _pass_verdict(qid)
            )
            for qid in ids
        ]
        return {
            "model_number": 1,
            "all_pass": all(v["action"] == "PASS" for v in verdicts),
            "verdicts": verdicts,
            "warnings": [],
            "coverage": [],
        }

    monkeypatch.setattr(val, "validate_exam", fake_validate)
    monkeypatch.setattr(val, "review_exam_globally", _global_pass_review)
    monkeypatch.setattr(
        val,
        "repair_exam",
        lambda _exam_value, _verdicts: {
            "warnings": [],
            "sent_ids": invalid_ids,
            "repaired_ids": invalid_ids,
            "failed_ids": [],
        },
    )
    state = {
        "generated_exams": [exam],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "pending_revalidation_ids": {},
        "warnings": [],
    }
    state = {**state, **validate_generated_questions(state)}
    first_passes = {
        verdict["question_id"]: verdict
        for verdict in state["validation_reports"][0]["verdicts"]
        if verdict["action"] == "PASS"
    }
    state = {**state, **repair_invalid_questions(state)}
    state = {**state, **validate_generated_questions(state)}

    assert validation_scopes[0] is None
    assert {tuple(scope) for scope in validation_scopes[1:]} == {
        (qid,) for qid in invalid_ids
    }
    report = state["validation_reports"][0]
    assert report["all_pass"]
    assert len(report["verdicts"]) == 10
    assert all(verdict["action"] == "PASS" for verdict in report["verdicts"])
    for verdict in report["verdicts"][:7]:
        assert verdict == first_passes[verdict["question_id"]]


def test_mcq_balance_expands_revalidation_to_every_modified_item(monkeypatch):
    qids = ["model1_mcq_1", "model1_mcq_2"]
    exam = _exam({"mcq": [_mcq(qid) for qid in qids]})
    scopes = []
    repair_round = {"value": 0}

    def fake_validate(_exam_value, client=None, question_ids=None):
        scope = None if question_ids is None else list(question_ids)
        scopes.append(scope)
        if scope is None:
            verdicts = [
                _verdict(qid, "FIX_ANSWER", av=False, fields=["answer"])
                for qid in qids
            ]
        else:
            verdicts = [_pass_verdict(qid) for qid in scope]
        return {
            "model_number": 1,
            "all_pass": all(v["action"] == "PASS" for v in verdicts),
            "verdicts": verdicts,
            "warnings": [],
            "coverage": [],
        }

    def fake_repair(_exam_value, _verdicts):
        qid = qids[repair_round["value"]]
        repair_round["value"] += 1
        return {
            "warnings": [],
            "sent_ids": qids,
            "repaired_ids": [qid],
            "failed_ids": [other for other in qids if other != qid],
        }

    monkeypatch.setattr(val, "validate_exam", fake_validate)
    monkeypatch.setattr(val, "review_exam_globally", _global_pass_review)
    monkeypatch.setattr(val, "repair_exam", fake_repair)
    state = {
        "generated_exams": [exam],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "pending_revalidation_ids": {},
        "warnings": [],
    }
    state = {**state, **validate_generated_questions(state)}
    state = {**state, **repair_invalid_questions(state)}
    state = {**state, **validate_generated_questions(state)}
    state = {**state, **repair_invalid_questions(state)}
    state = {**state, **validate_generated_questions(state)}

    assert scopes[0] is None
    assert {tuple(scope) for scope in scopes[1:] if scope is not None} == {
        (qid,) for qid in qids
    }
    assert repair_round["value"] == 2
    assert state["validation_reports"][0]["all_pass"]


def test_repair_flow_revalidates_repaired_model(monkeypatch):
    fake = FakeClient(
        validator_result=[_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["correct_answer"])],
        repair_result=[
            {"question_id": "model1_mcq_1", "repaired_fields": ["correct_answer"],
             "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                          "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
                          "correct_answer": "C"}}
        ],
    )
    validate_calls = {"n": 0}

    def _chat_json(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, **kw):
        if system_prompt == GLOBAL_VALIDATOR_SYSTEM_PROMPT:
            registry = json.loads(prompt.split("## Complete exam registry\n", 1)[1].split(
                "\n\n## Candidate pairs", 1
            )[0])
            return _global_pass_result([item["question_id"] for item in registry])
        if system_prompt == REPAIR_SYSTEM_PROMPT:
            return {"repairs": fake.repair_result}
        validate_calls["n"] += 1
        if validate_calls["n"] >= 2:
            return {"verdicts": [_pass_verdict("model1_mcq_1")]}
        return {"verdicts": fake.validator_result}

    monkeypatch.setattr(FakeClient, "chat_json", _chat_json)
    monkeypatch.setattr(val, "create_llm_client", lambda **_kwargs: fake)

    state = {
        "generated_exams": [_exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "warnings": [],
    }
    state = {**state, **validate_generated_questions(state)}
    assert not state["validated_models"]  # invalid -> not validated yet
    state = {**state, **repair_invalid_questions(state)}
    assert state["question_repair_attempts"] == {"m1_mcq_1": 1}
    state = {**state, **validate_generated_questions(state)}
    assert 1 in state["validated_models"]
    assert state["validation_reports"][0]["all_pass"]


def test_repair_attempts_bounded_per_model(monkeypatch):
    fake = FakeClient(validator_result=[_verdict("model1_tf_1", "FIX_ANSWER", av=False, fields=["answer"])],
                      repair_result=[
                          {"question_id": "model1_tf_1", "repaired_fields": ["answer"],
                           "question": {"statement": "Stmt", "answer": "True"}}
                      ])
    monkeypatch.setattr(val, "create_llm_client", lambda **_kwargs: fake)

    state = {
        "generated_exams": [_exam({"true_false": [_tf("model1_tf_1", answer="False")]})],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {"m1_tf_1": MAX_QUESTION_REPAIR_ATTEMPTS},
        "warnings": [],
    }
    out = validate_generated_questions(state)
    assert any("max question repair attempts reached" in w for w in out["warnings"])
    assert 1 not in out["validated_models"]  # unresolved is never called validated


def test_model_repairs_never_mix(monkeypatch):
    q1 = _mcq("model1_mcq_1", answer="B")
    q2 = _mcq("model2_mcq_1", answer="B")
    fake = FakeClient(
        validator_result=[_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["correct_answer"])],
        validator_retry_result=[_pass_verdict("model2_mcq_1")],
        repair_result=[
            {"question_id": "model1_mcq_1", "repaired_fields": ["correct_answer"],
             "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                          "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
                          "correct_answer": "A"}}
        ],
    )
    monkeypatch.setattr(val, "create_llm_client", lambda **_kwargs: fake)
    state = {
        "generated_exams": [_exam({"mcq": [q1]}, model_number=1),
                            _exam({"mcq": [q2]}, model_number=2)],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "warnings": [],
    }
    state = {**state, **validate_generated_questions(state)}
    state = {**state, **repair_invalid_questions(state)}
    assert q1["correct_answer"] == "B"  # repaired answer was balanced before revalidation
    assert q1["options"]["B"] == "F1 Score"  # active item is unchanged before candidate PASS
    assert q2["correct_answer"] == "B"  # model2 untouched
    assert q2["question_id"] == "model2_mcq_1"
    assert state["question_repair_attempts"] == {"m1_mcq_1": 1}


def test_validated_model_not_revalidated(monkeypatch):
    fake = FakeClient(validator_result=[_pass_verdict("model1_mcq_1")])
    monkeypatch.setattr(val, "create_llm_client", lambda **_kwargs: fake)
    state = {
        "generated_exams": [_exam({"mcq": [_mcq("model1_mcq_1")]})],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "warnings": [],
    }
    out = validate_generated_questions(state)
    assert 1 in out["validated_models"]
    assert fake.validate_calls == 1


def test_latest_report_for_already_passed_model_is_preserved(monkeypatch):
    old_report = {
        "model_number": 1,
        "all_pass": True,
        "verdicts": [_pass_verdict("model1_mcq_1")],
        "warnings": [],
    }

    monkeypatch.setattr(
        val,
        "validate_exam",
        lambda exam: {
            "model_number": 2,
            "all_pass": True,
            "verdicts": [_pass_verdict("model2_mcq_1")],
            "warnings": [],
        },
    )
    state = {
        "generated_exams": [
            _exam({"mcq": [_mcq("model1_mcq_1")]}, model_number=1),
            _exam({"mcq": [_mcq("model2_mcq_1")]}, model_number=2),
        ],
        "validation_reports": [old_report],
        "validated_models": [1],
        "question_repair_attempts": {},
        "warnings": [],
    }
    out = validate_generated_questions(state)
    assert {report["model_number"] for report in out["validation_reports"]} == {1, 2}
    assert out["validation_reports"][0] is old_report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def json_ids(prompt):
    """Parse the repair payload JSON embedded in a prompt and return IDs order."""
    import re
    return re.findall(r'"question_id"\s*:\s*"([^"]+)"', prompt)


def zero_repair(fake):
    return fake.repair_calls == 0


def test_validator_prompt_contains_main_and_support_sources():
    prompt = val.build_validator_prompt([{
        "question_id": "q1",
        "question_type": "equation",
        "question": {"equation": "derive x"},
        "plan": {
            "source_chunk_id": "main",
            "support_chunk_ids": ["support"],
        },
        "_source_chunks": {"main": "first law", "support": "second law"},
    }])
    entries = _validator_entries(prompt)
    assert entries[0]["assigned_sources"] == {
        "main": "first law",
        "support": "second law",
    }


def test_local_validator_rejects_commutative_duplicate_mcq_options():
    item = {
        "question_id": "q1",
        "question": "Which relationship is correct?",
        "options": {
            "A": "sin θc = n2 / n1",
            "B": "sin θc = n1 / n2",
            "C": "sin θc = n1 × n2",
            "D": "sin θc = n2 × n1",
        },
        "correct_answer": "A",
    }
    verdict = val._local_verdict("mcq", item, [item])
    assert verdict is not None
    assert verdict["action"] == "FIX_OPTIONS"
    assert "duplicate options" in verdict["reason"]
