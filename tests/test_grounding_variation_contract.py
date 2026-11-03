from __future__ import annotations

from app.online.multi_validator import ProtectedQuestion, _payload_for, _system_prompt


def _item(
    question_type: str = "mcq",
    *,
    selected_variation: dict | None = None,
    exact_source_instance: bool = False,
) -> ProtectedQuestion:
    selected = selected_variation or {
        "variation_id": "variation_generic",
        "operation": "optimize_linear_function_at_vertices",
        "target": "Find the maximum or minimum of a linear function at feasible-region vertices",
        "givens": ["feasible-region vertices", "a linear objective to maximize or minimize"],
        "numeric_values_may_change": True,
        "evidence_chunk_ids": ["source-1"],
    }
    prompt_field = "statement" if question_type == "true_false" else "equation" if question_type == "equation" else "question"
    question = {
        "question_id": f"model1_{question_type}_1",
        "slot_id": f"m1_{question_type}_1",
        prompt_field: "Student-facing generated question",
        "solution_certificate": {"method": "test"},
    }
    if question_type == "mcq":
        question.update({"options": {"A": "1", "B": "2", "C": "3", "D": "4"}, "correct_answer": "A"})
    elif question_type == "true_false":
        question["answer"] = True
    else:
        question["final_answer"] = "answer"
    return ProtectedQuestion(
        question_id=question["question_id"], question_type=question_type, slot_id=question["slot_id"],
        question=question,
        task={
            "task_operation": selected["operation"], "target": selected["target"],
            "selected_variation_id": selected["variation_id"],
            "permitted_variations": [selected],
            # A restrictive source-instance alternative must not override the selected variation.
            "allowed_variations": [{
                "variation_id": "variation_source_minimum", "operation": "minimize_cost_at_vertices",
                "target": "Choose the minimum-cost vertex", "numeric_values_may_change": False,
            }],
            "exact_source_instance": exact_source_instance,
        },
        evidence={"source-1": "A source example minimizes 55a + 95p at feasible-region vertices."},
        word_bank=[], sanitization_events=[], errors=[], requires_quantitative=False,
    )


def test_grounding_payload_for_original_mcq_style_variation_allows_new_numbers_and_both_directions():
    payload = _payload_for("grounding", _item())

    assert payload["grounding_contract"] == {
        "task_operation": "optimize_linear_function_at_vertices",
        "target": "Find the maximum or minimum of a linear function at feasible-region vertices",
        "selected_variation_id": "variation_generic",
        "numeric_values_may_change": True,
        "exact_source_instance": False,
        "permitted_optimization_direction": "both",
        "selected_variation_givens": [
            "feasible-region vertices", "a linear objective to maximize or minimize",
        ],
        "selected_variation_evidence_chunk_ids": ["source-1"],
    }
    assert "allowed_variations" not in payload["grounding_contract"]


def test_grounding_contract_preserves_restrictive_minimum_variation():
    item = _item(selected_variation={
        "variation_id": "variation_minimum", "operation": "minimize_cost_at_vertices",
        "target": "Choose the minimum-cost vertex", "givens": ["f(a,p)=55a+95p"],
        "numeric_values_may_change": False, "evidence_chunk_ids": ["source-1"],
    }, exact_source_instance=True)

    contract = _payload_for("grounding", item)["grounding_contract"]
    assert contract["numeric_values_may_change"] is False
    assert contract["exact_source_instance"] is True
    assert contract["permitted_optimization_direction"] == "min"


def test_grounding_contract_applies_to_equation_and_word_problem_without_extra_request_path():
    for question_type in ("equation", "word_problem"):
        payload = _payload_for("grounding", _item(question_type))
        assert payload["question_type"] == question_type
        assert payload["grounding_contract"]["selected_variation_id"] == "variation_generic"
        assert len(payload["assigned_child_chunks"]) == 1


def test_grounding_prompt_describes_authorized_original_variations_without_weakening_strict_checks():
    prompt = _system_prompt("grounding")

    assert "grounding_contract" in prompt
    assert "numeric_values_may_change" in prompt
    assert "permitted_optimization_direction is both" in prompt
    assert "unsupported mathematical concept" in prompt
    assert "invented factual claim" in prompt
