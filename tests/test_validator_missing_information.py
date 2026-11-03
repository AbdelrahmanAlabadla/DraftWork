from __future__ import annotations

import json
from copy import deepcopy

import pytest

from app.llm.schemas import ValidationIssue
from app.online import multi_repair, multi_validator
from app.online.multi_repair import build_repair_route, repair_question_specialized
from app.online.multi_validator import (
    ProtectedQuestion,
    _compatibility_verdict,
    _parse_rows,
    _payload_for,
    _normalize_missing_findings,
    _system_prompt,
    _validate_revalidation_coverage,
)


CONSTRAINTS = ["7 ≤ c ≤ 14", "8 ≤ p ≤ 12", "c + p ≤ 23"]
BROKEN_STATEMENT = (
    "وفق القيود، فإن تخصيص 12 ساعة للتفاضل و10 ساعات للفيزياء يحقق جميع القيود."
)
COMPLETE_STATEMENT = (
    "إذا كانت القيود 7 ≤ c ≤ 14، و8 ≤ p ≤ 12، وc + p ≤ 23، "
    "فإن تخصيص c = 12 ساعة للتفاضل وp = 10 ساعات للفيزياء يحقق جميع القيود."
)


def _task() -> dict:
    variation = {
        "variation_id": "variation_694f53999424bd42",
        "question_types": ["true_false"],
        "operation": "check_an_ordered_pair_against_all_study_time_inequalities",
        "target": "feasibility_against_all_constraints",
        "givens": CONSTRAINTS,
        "reasoning_mode": "law_or_theorem_application",
        "evidence_chunk_ids": ["ec91e86a-096e-5c4a-a6ad-390b56996b9c"],
    }
    return {
        "task_id": "task_e22b32e17c016bb2",
        "source_chunk_id": "ec91e86a-096e-5c4a-a6ad-390b56996b9c",
        "required_givens": CONSTRAINTS,
        "given_conditions": CONSTRAINTS,
        "variable_givens": ["c = calculus study hours", "p = physics study hours"],
        "task_operation": variation["operation"],
        "target": variation["target"],
        "student_work_product": "true_or_false_feasibility_judgment",
        "reasoning_mode": "law_or_theorem_application",
        "requires_quantitative_validation": True,
        "selected_variation_id": variation["variation_id"],
        "selected_variation": variation,
        "permitted_variations": [variation],
    }


def _protected(statement: str = BROKEN_STATEMENT) -> ProtectedQuestion:
    return ProtectedQuestion(
        question_id="model1_true_false_1",
        question_type="true_false",
        slot_id="m1_true_false_1",
        question={
            "question_id": "model1_true_false_1",
            "slot_id": "m1_true_false_1",
            "statement": statement,
            "answer": True,
            "solution_certificate": {
                "method": "constraint substitution",
                "important_givens": CONSTRAINTS,
                "calculation_steps": ["12 and 10 satisfy every inequality"],
                "final_answer": "True",
                "units": "hours",
                "matching_correct_option": None,
                "verification_method": "directly substitute both values",
            },
        },
        task=_task(),
        evidence={
            "ec91e86a-096e-5c4a-a6ad-390b56996b9c":
                "The feasible region is defined by 7 ≤ c ≤ 14, 8 ≤ p ≤ 12, c + p ≤ 23."
        },
        word_bank=[], sanitization_events=[], errors=[], requires_quantitative=True,
    )


def _missing_issue() -> dict:
    return {
        "failure_code": "missing_required_given",
        "fields_to_fix": ["statement"],
        "reason": (
            "The statement asks whether 12 and 10 satisfy constraints that are not "
            "shown to the student, so the claim cannot be evaluated independently."
        ),
        "expected_fix": (
            "State all three inequalities in statement and define c and p as the two "
            "study-hour variables."
        ),
        "category": "missing_constraint",
        "missing_information": CONSTRAINTS,
        "source_evidence_ids": ["ec91e86a-096e-5c4a-a6ad-390b56996b9c"],
    }


def _raw(role: str, *, status: str = "FAIL", statement_complete: bool = False) -> dict:
    issues = [] if status == "PASS" else [_missing_issue()]
    return {
        "validator": role,
        "verdicts": [{
            "question_id": "model1_true_false_1",
            "status": status,
            "issues": issues,
            "grounding_outcome": None,
            "observed_difficulty": None,
            "calculation_evidence": (
                ["The three constraints are absent from the student statement."]
                if role == "quantitative" and not statement_complete else []
            ),
            "original_issue_checks": [],
        }],
    }


def test_validator_prompts_define_student_visible_missing_information_contract():
    task_prompt = _system_prompt("task_alignment")
    quantitative_prompt = _system_prompt("quantitative")

    for prompt in (task_prompt, quantitative_prompt):
        assert "missing_information" in prompt
        assert "student-visible" in prompt
        assert "solution_certificate" in prompt
        assert "mathematically equivalent" in prompt
        assert "formula" in prompt and "derive" in prompt
    assert "missing_context" in task_prompt


def test_validator_prompt_lists_exact_top_level_repair_fields_for_every_question_type():
    prompt = _system_prompt("quantitative")

    for field_contract in (
        "MCQ=question", "True/False=statement", "Fill in the Blank=question",
        "Definition=term", "Equation=equation", "Why/Short Answer=question",
        "Word Problem=question", "Essay=question",
    ):
        assert field_contract in prompt
    assert "never a question_and_answer.* path" in prompt


def test_original_true_false_payload_exposes_contract_but_not_hidden_solution_as_given():
    item = _protected()
    task_payload = _payload_for("task_alignment", item)
    quantitative_payload = _payload_for("quantitative", item)

    assert task_payload["question_and_answer"]["statement"] == BROKEN_STATEMENT
    assert task_payload["task_contract"]["required_givens"] == CONSTRAINTS
    assert task_payload["assigned_child_chunks"][0]["chunk_id"] == (
        "ec91e86a-096e-5c4a-a6ad-390b56996b9c"
    )
    assert quantitative_payload["quantitative_contract"]["required_givens"] == CONSTRAINTS
    assert quantitative_payload["quantitative_contract"]["source_evidence_ids"] == [
        "ec91e86a-096e-5c4a-a6ad-390b56996b9c"
    ]
    # The certificate is visible for diagnosis, while the system instructions
    # explicitly prohibit treating it as student-available information.
    assert "solution_certificate" in quantitative_payload["question_and_answer"]


@pytest.mark.parametrize("role", ["task_alignment", "quantitative"])
def test_missing_constraint_finding_is_schema_valid_and_precise(role):
    valid, errors = _parse_rows(role, _raw(role), {"model1_true_false_1"})

    assert errors == {}
    issue = valid["model1_true_false_1"]["issues"][0]
    assert issue["failure_code"] == "missing_required_given"
    assert issue["category"] == "missing_constraint"
    assert issue["fields_to_fix"] == ["statement"]
    assert issue["missing_information"] == CONSTRAINTS
    assert issue["source_evidence_ids"] == [
        "ec91e86a-096e-5c4a-a6ad-390b56996b9c"
    ]


@pytest.mark.parametrize(
    "category",
    [
        "missing_given", "missing_constraint", "missing_formula",
        "missing_definition", "missing_context", "missing_variable_definition",
    ],
)
def test_all_missing_information_categories_are_schema_supported(category):
    issue = _missing_issue()
    issue["category"] = category
    assert ValidationIssue.model_validate(issue).category == category


def test_english_missing_context_uses_the_same_structured_contract():
    item = _protected(
        "According to the constraints, assigning 12 hours to calculus and 10 to "
        "physics satisfies every constraint."
    )
    payload = _payload_for("task_alignment", item)
    assert payload["question_and_answer"]["statement"].startswith(
        "According to the constraints"
    )
    raw = _raw("task_alignment")
    raw["verdicts"][0]["issues"][0].update({
        "category": "missing_context",
        "reason": "The phrase refers to constraints that are not supplied.",
    })
    valid, errors = _parse_rows(
        "task_alignment", raw, {item.question_id}
    )
    assert errors == {}
    assert valid[item.question_id]["issues"][0]["category"] == "missing_context"


@pytest.mark.parametrize(
    "removed_field",
    ["category", "missing_information", "expected_fix"],
)
def test_incomplete_missing_information_finding_is_rejected(removed_field):
    raw = _raw("task_alignment")
    issue = raw["verdicts"][0]["issues"][0]
    issue.pop(removed_field)
    valid, errors = _parse_rows(
        "task_alignment", raw, {"model1_true_false_1"}
    )

    assert valid == {}
    assert "model1_true_false_1" in errors


def test_findings_survive_aggregation_with_stable_issue_ids():
    item = _protected()
    task_rows, _ = _parse_rows(
        "task_alignment", _raw("task_alignment"), {item.question_id}
    )
    quantitative_rows, _ = _parse_rows(
        "quantitative", _raw("quantitative"), {item.question_id}
    )
    quantitative_rows[item.question_id]["issues"][0]["expected_fix"] = (
        "Define c and p and display each feasibility condition before asking for a verdict."
    )
    verdict = _compatibility_verdict(
        item,
        {
            "task_alignment": task_rows[item.question_id],
            "quantitative": quantitative_rows[item.question_id],
        },
        {"action": "PASS", "question_valid": True, "answer_valid": True},
    )

    assert verdict["action"] == "FIX_QUESTION"
    assert verdict["fields_to_fix"] == ["statement"]
    assert len(verdict["required_fixes"]) == 2
    assert verdict["repair_instruction_conflicts"] == []
    assert len({issue["issue_id"] for issue in verdict["required_fixes"]}) == 2
    assert all(issue["missing_information"] == CONSTRAINTS for issue in verdict["required_fixes"])


def test_missing_finding_target_and_evidence_are_code_verified():
    item = _protected()
    verdict = _raw("task_alignment")["verdicts"][0]
    verdict["issues"][0]["source_evidence_ids"] = []
    assert _normalize_missing_findings("task_alignment", item, verdict) is None
    assert verdict["issues"][0]["source_evidence_ids"] == [
        "ec91e86a-096e-5c4a-a6ad-390b56996b9c"
    ]

    wrong_field = deepcopy(verdict)
    wrong_field["issues"][0]["fields_to_fix"] = ["answer"]
    assert "student-facing field" in str(
        _normalize_missing_findings("task_alignment", item, wrong_field)
    )

    wrong_evidence = deepcopy(verdict)
    wrong_evidence["issues"][0]["source_evidence_ids"] = ["foreign-child"]
    assert "unassigned evidence" in str(
        _normalize_missing_findings("task_alignment", item, wrong_evidence)
    )


@pytest.mark.parametrize(
    ("question_type", "field"),
    [
        ("mcq", "question"),
        ("true_false", "statement"),
        ("fill_in_the_blank", "question"),
        ("definition", "term"),
        ("equation", "equation"),
        ("why", "question"),
        ("word_problem", "question"),
        ("essay", "question"),
    ],
)
def test_missing_information_targets_must_use_exact_top_level_field_names(question_type, field):
    item = _protected()
    item.question_type = question_type
    valid = _raw("quantitative")["verdicts"][0]
    valid["issues"][0]["fields_to_fix"] = [field]
    assert _normalize_missing_findings("quantitative", item, valid) is None

    invalid = deepcopy(valid)
    invalid["issues"][0]["fields_to_fix"] = [f"question_and_answer.{field}"]
    error = _normalize_missing_findings("quantitative", item, invalid)
    assert error is not None
    assert "student-facing field" in error


def test_repair_receives_exact_missing_requirements(monkeypatch):
    item = _protected()
    findings = []
    for role in ("task_alignment", "quantitative"):
        issue = {"validator": role, "validation_status": "FAIL", **_missing_issue()}
        issue["issue_id"] = f"{item.question_id}-{role}-missing"
        findings.append(issue)
    verdict = {
        "question_id": item.question_id,
        "validation_status": "FAIL",
        "action": "FIX_QUESTION",
        "failure_code": "missing_required_given",
        "fields_to_fix": ["statement"],
        "required_fixes": findings,
        "repair_instruction_conflicts": [],
    }
    route = build_repair_route("true_false", verdict)
    assert route["status"] == "READY"
    assert [step.agent for step in route["steps"]] == ["task_alignment"]

    captured = {}

    def fake_request(_client, prompt, **_kwargs):
        envelope = json.loads(
            prompt.split("<REPAIR_DATA>\n", 1)[1].split("\n</REPAIR_DATA>", 1)[0]
        )
        payload = envelope["questions"][0]
        captured.update(deepcopy(payload))
        return {"repairs": [{
            "question_id": item.question_id,
            "patch": {"statement": COMPLETE_STATEMENT},
            "repair_metadata": {
                "modified_fields": ["statement"],
                "changes": [{
                    "field": "statement",
                    "previous_value": BROKEN_STATEMENT,
                    "new_value": COMPLETE_STATEMENT,
                    "reason": "Adds all omitted constraints and defines c and p.",
                }],
                "targeted_failure_codes": ["missing_required_given"],
            },
        }]}

    monkeypatch.setattr(multi_repair, "request_structured", fake_request)
    exam = {"model_number": 1, "questions": {"true_false": [deepcopy(item.question)]}}
    result = repair_question_specialized(
        exam=exam, verdict=verdict, qtype="true_false", plan_item=item.task,
        source_chunks=item.evidence, domain="STEM",
        client_factory=lambda _agent: object(),
    )

    assert result["repaired_ids"] == [item.question_id], result
    assert captured["editable_fields"] == ["statement"]
    assert len(captured["findings"]) == 2
    assert all(finding["missing_information"] == CONSTRAINTS for finding in captured["findings"])
    assert all(finding["source_evidence_ids"] for finding in captured["findings"])
    assert exam["questions"]["true_false"][0]["statement"] == COMPLETE_STATEMENT


@pytest.mark.parametrize("role", ["task_alignment", "quantitative"])
def test_corrected_statement_can_pass_and_resolve_original_issue(role):
    item = _protected(COMPLETE_STATEMENT)
    issue_id = f"{item.question_id}-{role}-missing"
    item.revalidation = {
        "original_failure_findings": [{
            "issue_id": issue_id, "validator": role, **_missing_issue(),
        }]
    }
    revalidation_payload = _payload_for(role, item)["revalidation"]
    original = revalidation_payload["original_failure_findings"][0]
    assert original["category"] == "missing_constraint"
    assert original["missing_information"] == CONSTRAINTS
    raw = _raw(role, status="PASS", statement_complete=True)
    raw["verdicts"][0]["original_issue_checks"] = [{
        "issue_id": issue_id,
        "resolved": True,
        "reason": "All three constraints are now visible in the statement.",
        "remaining_defect": "",
    }]
    valid, errors = _parse_rows(role, raw, {item.question_id})

    assert errors == {}
    verdict = valid[item.question_id]
    assert verdict["status"] == "PASS"
    assert _validate_revalidation_coverage(role, item, verdict) is None


@pytest.mark.parametrize(
    "qtype,expected_field",
    [
        ("mcq", "question"),
        ("true_false", "statement"),
        ("equation", "equation"),
        ("word_problem", "question"),
        ("fill_in_the_blank", "question"),
        ("definition", "term"),
        ("why", "question"),
        ("essay", "question"),
    ],
)
def test_all_question_types_have_an_existing_student_facing_repair_field(
    qtype, expected_field,
):
    assert multi_validator._prompt_fields(qtype) == (expected_field,)


def test_quantitative_role_remains_eligibility_gated():
    assert multi_validator._requires_quantitative(
        "true_false", _task()
    ) is True
    assert multi_validator._requires_quantitative(
        "definition", {"reasoning_mode": "conceptual_recall", "task_operation": "define"}
    ) is False
