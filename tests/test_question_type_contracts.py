from __future__ import annotations

from app.online.prompts import build_prompt
from app.online.validator import _local_verdict


def _why(question: str, answer: str = "A concise grounded reason.") -> dict:
    return {"question": question, "reference_answer": answer}


def _definition(term: str, answer: str = "A concise grounded definition.") -> dict:
    return {"term": term, "reference_answer": answer}


def test_why_rejects_retrieval_and_accepts_causal_reasoning():
    assert _local_verdict("why", _why("Why does friction slow a moving object?"), []) is None
    assert _local_verdict("why", _why("Explain the mechanism by which friction slows motion."), []) is None
    for prompt in ("List three types of force.", "Define friction.", "State the unit of force."):
        verdict = _local_verdict("why", _why(prompt), [])
        assert verdict["failure_code"] == "question_type_operation_mismatch"


def test_definition_contract_uses_a_bare_term():
    item = _definition("linear equation", "An equation whose highest variable degree is one.")
    assert _local_verdict("definition", item, []) is None


def test_generator_prompts_preserve_why_and_definition_contracts():
    _system, why_prompt = build_prompt("why", 1, "source", planned_items=[{
        "slot_id": "m1_why_1", "task_operation": "explain_mechanism",
        "reasoning_mode": "conceptual_reasoning", "target": "a mechanism",
    }])
    _system, definition_prompt = build_prompt("definition", 1, "source")
    assert "Why / Give a Reason" in why_prompt
    assert "cause/effect reasoning" in why_prompt
    assert '\"term\": \"kinetic energy\"' in definition_prompt
    assert "only the source term or concept" in definition_prompt
