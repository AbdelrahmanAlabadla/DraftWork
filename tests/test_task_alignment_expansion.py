"""Task Alignment support for authorized cognitive difficulty expansions."""

from __future__ import annotations

from copy import deepcopy

from app.online.multi_validator import (
    ROLE_INSTRUCTIONS,
    ProtectedQuestion,
    _compatibility_verdict,
    _payload_for,
)


def _task(**overrides) -> dict:
    task = {
        "task_id": "task_linear",
        "concept": "linear equations",
        "knowledge_focus": "solve linear equations",
        "idea_to_test": "distribute and isolate",
        "target": "value of x",
        "task_operation": "distribute_then_isolate",
        "required_givens": ["linear equation"],
        "given_conditions": ["2(x+3)=14"],
        "effective_givens": ["2(x+3)=14"],
        "selected_variation_id": "v1",
        "selected_variation_group_id": "vg1",
        "permitted_variations": [{
            "variation_id": "v1",
            "operation": "distribute_then_isolate",
            "target": "value of x",
            "givens": ["ax+b=c"],
            "question_types": ["equation"],
            "numeric_values_may_change": True,
            "evidence_chunk_ids": ["c1"],
        }],
        "complexity_boundaries": ["linear equations only"],
        "exact_source_instance": False,
        "student_work_product": "solution",
        "reasoning_mode": "symbolic_manipulation",
        "allowed_transformations": [
            {
                "transformation": "distribute_then_isolate",
                "evidence_chunk_ids": ["c1"],
                "support": "Distribute then isolate.",
            },
            {
                "transformation": "collect_variables_one_side",
                "evidence_chunk_ids": ["c1"],
                "support": "Subtract 2x from both sides.",
            },
        ],
        "cognitive_transformations": [
            "distribute_then_isolate", "collect_variables_one_side",
        ],
        "reasoning_requirements": [
            "Distribute the factor.", "Collect variables on one side.",
            "Isolate the variable.",
        ],
        "subject": "mathematics",
    }
    task.update(overrides)
    return task


def _item(task: dict, *, equation: str = "3(x+2)=2x+11") -> ProtectedQuestion:
    return ProtectedQuestion(
        question_id="model1_equation_1",
        question_type="equation",
        slot_id="m1_equation_1",
        question={
            "question_id": "model1_equation_1",
            "slot_id": "m1_equation_1",
            "equation": equation,
            "solution_steps": ["3x+6=2x+11", "x+6=11", "x=5"],
            "final_answer": "5",
            "solution_certificate": None,
        },
        task=task,
        evidence={"c1": "Distribute; collect variables from both sides; solve."},
        word_bank=[],
        sanitization_events=[],
        errors=[],
        requires_quantitative=True,
    )


def test_task_alignment_payload_carries_authorized_expansion_context():
    payload = _payload_for("task_alignment", _item(_task()))
    contract = payload["task_contract"]
    assert contract["selected_variation"]["givens"] == ["ax+b=c"]
    assert contract["selected_variation"]["operation"] == "distribute_then_isolate"
    assert contract["cognitive_transformations"] == [
        "distribute_then_isolate", "collect_variables_one_side",
    ]
    assert [value["transformation"] for value in contract["authorized_transformations"]] == [
        "distribute_then_isolate", "collect_variables_one_side",
    ]
    assert contract["authorized_transformations"][1]["support"]
    assert contract["selected_transformations_authorized"] == [
        "distribute_then_isolate", "collect_variables_one_side",
    ]
    assert contract["selected_transformations_unauthorized"] == []
    assert contract["transformation_evidence_in_assignment"] is True
    assert contract["reasoning_requirements"]


def test_unauthorized_transformation_is_explicit_even_with_contradictory_givens():
    task = _task(
        allowed_transformations=[{
            "transformation": "distribute_then_isolate",
            "evidence_chunk_ids": ["c1"],
            "support": "Distribute then isolate.",
        }],
        # Contradictory one-sided base givens cannot authorize the two-sided work.
        effective_givens=["2(x+3)=14"],
        given_conditions=["2(x+3)=14"],
    )
    contract = _payload_for("task_alignment", _item(task))["task_contract"]
    assert contract["transformation_authorization_status"] == "unauthorized"
    assert contract["selected_transformations_authorized"] == ["distribute_then_isolate"]
    assert contract["selected_transformations_unauthorized"] == ["collect_variables_one_side"]


def test_unauthorized_transformation_cannot_pass_even_if_luna_says_pass():
    task = _task(allowed_transformations=[{
        "transformation": "distribute_then_isolate",
        "evidence_chunk_ids": ["c1"],
        "support": "Distribute then isolate.",
    }])
    verdict = _compatibility_verdict(
        _item(task),
        {"task_alignment": {"status": "PASS", "issues": []}},
        None,
    )
    assert verdict["action"] != "PASS"
    assert verdict["failure_code"] == "wrong_variation"
    assert any(
        "not compiler-authorized" in str(issue.get("reason"))
        for issue in verdict["required_fixes"]
    )


def test_transformation_evidence_outside_assignment_is_flagged():
    task = _task()
    task["allowed_transformations"][1]["evidence_chunk_ids"] = ["unassigned-9"]
    contract = _payload_for("task_alignment", _item(task))["task_contract"]
    assert contract["transformation_evidence_in_assignment"] is False
    assert contract["transformation_evidence_missing"] == [{
        "transformation": "collect_variables_one_side",
        "missing_chunk_ids": ["unassigned-9"],
    }]


def test_unassigned_transformation_evidence_cannot_pass_even_if_luna_says_pass():
    task = _task()
    task["allowed_transformations"][1]["evidence_chunk_ids"] = ["unassigned-9"]
    verdict = _compatibility_verdict(
        _item(task),
        {"task_alignment": {"status": "PASS", "issues": []}},
        None,
    )
    assert verdict["action"] != "PASS"
    assert verdict["failure_code"] == "constraint_violation"
    assert any(
        "not assigned" in str(issue.get("reason"))
        for issue in verdict["required_fixes"]
    )


def test_legacy_plan_without_transformations_keeps_empty_contract():
    task = _task()
    task.pop("allowed_transformations")
    task.pop("cognitive_transformations")
    task.pop("reasoning_requirements")
    contract = _payload_for("task_alignment", _item(task, equation="2(x+3)=14"))["task_contract"]
    assert contract["selected_transformations"] == []
    assert contract["authorized_transformations"] == []
    assert contract["selected_transformations_unauthorized"] == []


def test_task_alignment_prompt_requires_authorization_and_realization():
    prompt = ROLE_INSTRUCTIONS["task_alignment"]
    assert "authorized_transformations" in prompt
    assert "cognitive_transformations" in prompt
    assert "never proof of authorization" in prompt
    assert "not actually realized" in prompt
    assert "learning objective" in prompt
    assert "complexity_boundaries" in prompt
