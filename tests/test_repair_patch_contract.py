from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.llm.schemas import repair_response_model
from app.online import validator


CERTIFICATE = {
    "method": "substitution",
    "important_givens": ["x = 2"],
    "calculation_steps": ["2 + 3 = 5"],
    "final_answer": "5",
    "units": "",
    "matching_correct_option": "B",
    "verification_method": "reverse calculation",
}


def _mcq():
    return {
        "question_id": "model1_mcq_5",
        "slot_id": "m1_mcq_5",
        "question": "What is 2 + 3?",
        "options": {"A": "4", "B": "5", "C": "6", "D": "7"},
        "correct_answer": "B",
        "solution_certificate": None,
    }


def test_certificate_only_patch_preserves_mcq_content(monkeypatch):
    question = _mcq()
    exam = {"model_number": 1, "questions": {"mcq": [question]}}
    before = deepcopy(question)
    monkeypatch.setattr(validator, "request_structured", lambda *_a, **_k: {
        "repairs": [{
            "question_id": "model1_mcq_5",
            "patch": {"solution_certificate": CERTIFICATE},
        }]
    })
    verdict = {
        "question_id": "model1_mcq_5", "action": "FIX_ANSWER",
        "fields_to_fix": ["solution_certificate"],
        "failure_code": "incomplete_solution", "validation_role": "stem_quantitative",
    }

    result = validator.repair_exam(
        exam, [verdict], client=object(), repair_role="stem_quantitative"
    )

    assert result["repaired_ids"] == ["model1_mcq_5"]
    repaired = exam["questions"]["mcq"][0]
    for field in ("question_id", "slot_id", "question", "options", "correct_answer"):
        assert repaired[field] == before[field]
    assert repaired["solution_certificate"] == CERTIFICATE


def test_patch_schema_rejects_missing_and_unauthorized_fields():
    model = repair_response_model([
        ("model1_mcq_5", "mcq", ["solution_certificate"])
    ])
    with pytest.raises(ValidationError):
        model.model_validate({
            "repairs": [{"question_id": "model1_mcq_5", "patch": {}}]
        })
    with pytest.raises(ValidationError):
        model.model_validate({
            "repairs": [{
                "question_id": "model1_mcq_5",
                "patch": {
                    "solution_certificate": CERTIFICATE,
                    "correct_answer": "A",
                },
            }]
        })


def test_noop_patch_is_rejected_precisely(monkeypatch):
    question = _mcq()
    question["solution_certificate"] = deepcopy(CERTIFICATE)
    exam = {"model_number": 1, "questions": {"mcq": [question]}}
    monkeypatch.setattr(validator, "request_structured", lambda *_a, **_k: {
        "repairs": [{
            "question_id": "model1_mcq_5",
            "patch": {"solution_certificate": deepcopy(CERTIFICATE)},
        }]
    })
    result = validator.repair_exam(exam, [{
        "question_id": "model1_mcq_5", "action": "FIX_ANSWER",
        "fields_to_fix": ["solution_certificate"],
        "failure_code": "incomplete_solution",
    }], client=object(), repair_role="stem_quantitative")
    assert result["failed_ids"] == ["model1_mcq_5"]
    assert any("returned no change" in warning for warning in result["warnings"])


@pytest.mark.parametrize("failure_code", [
    "incomplete_solution", "incorrect_calculation", "incorrect_result",
    "invalid_solution_certificate",
])
def test_missing_validation_role_routes_quantitative_defects(failure_code):
    verdict = {
        "action": "FIX_ANSWER", "failure_code": failure_code,
        "fields_to_fix": ["solution_certificate"],
    }
    assert validator._repair_role_for("mcq", verdict, domain="STEM") == "stem_quantitative"


@pytest.mark.parametrize("qtype,field,value", [
    ("mcq", "question", "Updated stem"),
    ("true_false", "statement", "Updated statement"),
    ("fill_in_the_blank", "question", "A ________ clue."),
    ("definition", "term", "momentum"),
    ("why", "reference_answer", "Because the force opposes motion."),
    ("equation", "final_answer", "x=2"),
    ("word_problem", "final_answer", "12 m"),
    ("essay", "reference_answer", "A supported explanation."),
])
def test_all_question_types_have_strict_field_patch_contract(qtype, field, value):
    model = repair_response_model([("qid", qtype, [field])])
    parsed = model.model_validate({
        "repairs": [{"question_id": "qid", "patch": {field: value}}]
    })
    assert parsed.repairs[0].patch.model_dump() == {field: value}
