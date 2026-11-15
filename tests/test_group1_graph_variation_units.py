"""Group 1 regressions: graph policy, variation/units, and repair dependency sync.

Offline only: no provider calls, no network, no persistence.
"""

from __future__ import annotations

import json

import pytest

from app.online.assessment_catalog import (
    _finalize_catalog,
    catalog_group1_diagnostics,
    eligible_task_variations,
    graph_dependency_class,
    materialize_plan_item,
    requires_unsupported_supplied_graph,
)
from app.online.lifecycle import question_fingerprint
from app.online.multi_repair import (
    RepairStep,
    _material_stem_change,
    _stale_dependent_error,
    _with_dependent_steps,
    build_repair_route,
    repair_question_specialized,
)
from app.online.multi_validator import (
    ROLE_FAILURE_CODES,
    ProtectedQuestion,
    _payload_for,
    _system_prompt,
)
from app.online.unit_policy import detect_unit_conflict, unit_policy_for


def _task(**overrides):
    task = {
        "concept": "solve linear equations",
        "knowledge_anchor": "linear equation",
        "task_operation": "solve_equation",
        "student_work_product": "solution",
        "target": "value of x",
        "reasoning_mode": "symbolic_manipulation",
        "allowed_question_types": ["equation"],
        "evidence_chunk_ids": ["c1"],
        "required_givens": ["linear equation in one variable"],
        "variable_givens": ["coefficient", "constant"],
        "complexity_boundaries": [],
        "allowed_variations": [{
            "target": "value of x",
            "operation": "solve_equation",
            "givens": ["ax + b = c"],
            "question_types": ["equation"],
            "reasoning_mode": "symbolic_manipulation",
            "student_work_product": "solution",
            "numeric_values_may_change": True,
        }],
        "exact_source_instance": False,
        "scenario_family": None,
        "text_answerable": True,
        "native_difficulty": "easy",
        "prerequisites": [],
    }
    task.update(overrides)
    return task


def _finalize(tasks, subject="mathematics"):
    return _finalize_catalog(
        {
            "assessment_profile": {"subject": subject, "domain": "STEM"},
            "tasks": tasks,
        },
        key="k",
        processed_document_key="d",
        selected_chunk_ids=["c1"],
        usage=None,
    )


# ---------------------------------------------------------------------------
# Graph classification
# ---------------------------------------------------------------------------

def test_graph_classifier_distinguishes_read_construct_and_conceptual():
    read_task = _task(
        task_operation="read_values_from_graph",
        target="read the y-intercept from the supplied graph",
        required_givens=["the graph shown below"],
    )
    assert graph_dependency_class(read_task) == "reads_supplied"
    assert not requires_unsupported_supplied_graph(read_task)
    assert requires_unsupported_supplied_graph(
        _task(
            task_operation="read_values_from_graph",
            target="read the y-intercept from the supplied graph",
            required_givens=["the graph shown below"],
            requires_supplied_graph=True,
        )
    )

    construct_task = _task(
        task_operation="graph_linear_equation",
        student_work_product="draw a graph",
        target="graph of a linear equation",
        required_givens=["y = 2x + 3"],
    )
    assert graph_dependency_class(construct_task) == "construct"
    assert not requires_unsupported_supplied_graph(construct_task)

    conceptual_task = _task(
        task_operation="explain_slope_meaning",
        target="meaning of slope on a graph",
        required_givens=[],
    )
    assert graph_dependency_class(conceptual_task) == "ambiguous"
    assert not requires_unsupported_supplied_graph(conceptual_task)


def test_plot_as_noun_is_not_student_construction():
    task = _task(
        task_operation="read_values_from_plot",
        target="read values from the plot",
        required_givens=["the plot below"],
    )
    assert graph_dependency_class(task) == "reads_supplied"
    assert not requires_unsupported_supplied_graph(task)


def test_arabic_graph_reading_is_blocked():
    task = _task(
        task_operation="identify_vertices",
        target="حدد رؤوس المنطقة من التمثيل البياني",
        required_givens=["التمثيل البياني لمنطقة المتباينات"],
    )
    assert graph_dependency_class(task) == "reads_supplied"
    assert not requires_unsupported_supplied_graph(task)


def test_arabic_graph_construction_is_allowed():
    task = _task(
        task_operation="ارسم التمثيل البياني",
        student_work_product="رسم المستقيم",
        target="تمثيل المستقيم بيانيا",
        required_givens=["y = 2x + 3"],
    )
    assert graph_dependency_class(task) == "construct"
    assert not requires_unsupported_supplied_graph(task)


def test_arabic_conceptual_graph_question_is_allowed():
    task = _task(
        task_operation="فسر معنى الميل",
        target="معنى الميل في التمثيل البياني",
        required_givens=[],
    )
    assert graph_dependency_class(task) == "ambiguous"
    assert not requires_unsupported_supplied_graph(task)


def test_defaulted_false_no_longer_masks_a_graph_reading_variation():
    task = _task(
        task_operation="identify_vertices_of_feasible_region",
        target="read the vertices from the graph",
        required_givens=["the graph of the feasible region"],
    )
    variation = {
        "variation_id": "variation_x",
        "operation": task["task_operation"],
        "target": task["target"],
        "givens": [],
        "question_types": ["equation"],
        # This is exactly what TaskVariation.model_validate used to synthesize.
        "requires_supplied_graph": False,
    }
    assert not requires_unsupported_supplied_graph(task, variation)
    explicit = {**variation, "requires_supplied_graph": True}
    assert requires_unsupported_supplied_graph(task, explicit)


def test_explicit_true_blocks_even_a_construction_angle():
    task = _task(
        task_operation="graph_linear_equation",
        target="graph of a linear equation",
        required_givens=["y = 2x + 3"],
        requires_supplied_graph=True,
    )
    assert requires_unsupported_supplied_graph(task)


def test_text_based_table_task_is_not_a_visual_dependency():
    task = _task(
        concept="interpret a data table",
        task_operation="interpret_table",
        target="interpret the table of observations",
        student_work_product="interpretation",
        required_givens=["year: 2001; time: 12.5; 2002: 13.1"],
    )
    assert not requires_unsupported_supplied_graph(task)
    assert eligible_task_variations(task, "equation")


def test_profile_operation_list_does_not_reject_catalog_variation():
    from app.online.assessment_catalog import eligible_task_variation_candidates

    task = _task(
        task_operation="verify_solution",
        target="verify whether x = 4",
        student_work_product="verification",
        allowed_variations=[{
            "target": "verify whether x = 4",
            "operation": "verify_solution",
            "givens": ["2x + 3 = 11"],
            "question_types": ["equation"],
            "reasoning_mode": "verification",
            "student_work_product": "verification",
        }],
    )
    candidates = eligible_task_variation_candidates(
        task, "equation",
        assessment_profile={"question_types": {"equation": {
            "task_operations": ["solve_equation"],
        }}},
        valid_chunk_ids={"c1"},
    )
    assert candidates


def test_graph_reading_with_stated_numeric_data_is_answerable():
    task = _task(
        task_operation="read_intersection_from_graph",
        target="intersection of two lines shown on a graph",
        required_givens=[
            "line 1 passes through (0, 3) and (4, 0)",
            "line 2 passes through (0, -1) and (2, 3)",
        ],
    )
    assert graph_dependency_class(task) == "reads_supplied"
    assert not requires_unsupported_supplied_graph(task)


def test_task_with_a_supported_alternative_keeps_that_alternative():
    task = _task(
        task_operation="interpret_graph",
        target="read the intercept from the supplied graph",
        required_givens=["the supplied graph"],
        allowed_question_types=["mcq"],
        allowed_variations=[
            {
                "target": "read the intercept from the supplied graph",
                "operation": "interpret_graph",
                "givens": ["the supplied graph"],
                "question_types": ["mcq"],
                "reasoning_mode": "representation_interpretation",
                "student_work_product": "intercept value",
            },
            {
                "target": "explain what the slope represents",
                "operation": "explain_slope_meaning",
                "givens": ["slope of a linear graph"],
                "question_types": ["mcq"],
                "reasoning_mode": "conceptual_reasoning",
                "student_work_product": "explanation",
            },
        ],
    )
    eligible = eligible_task_variations(task, "mcq")
    assert len(eligible) == 3
    assert {value["target"] for value in eligible} >= {
        "read the intercept from the supplied graph",
        "explain what the slope represents",
    }


# ---------------------------------------------------------------------------
# Catalog contract: variations, units and diagnostics
# ---------------------------------------------------------------------------

def test_exact_task_keeps_authorized_angles_but_disables_numeric_variation():
    catalog = _finalize([_task(exact_source_instance=True)])
    task = catalog["tasks"][0]
    assert task["exact_source_instance"] is True
    assert task["allowed_variations"]
    assert all(
        value["numeric_values_may_change"] is False
        for value in task["allowed_variations"]
    )
    assert "group1_diagnostics" in catalog


def test_replacement_feasibility_uses_the_classifier_when_the_flag_is_defaulted():
    from app.online.duplicate_pipeline import replacement_feasibility_reasons

    candidate = {
        "task_id": "t1",
        "concept": "read a graph",
        "task_operation": "read_values_from_graph",
        "target": "read values from the supplied graph",
        "required_givens": ["the graph shown below"],
        "text_answerable": True,
        "requires_supplied_graph": False,
        "allowed_question_types": ["mcq"],
    }
    reasons = replacement_feasibility_reasons(candidate, "mcq")
    assert "unsupported_supplied_graph" not in reasons
    explicit = {**candidate, "requires_supplied_graph": True}
    assert "unsupported_supplied_graph" in replacement_feasibility_reasons(
        explicit, "mcq"
    )


def test_exam_builder_swaps_a_blocked_angle_for_a_supported_alternative():
    from app.online.exam_builder import _reassign_supported_variation

    item = {
        "slot_id": "m1_mcq_1",
        "task_id": "t1",
        "task_operation": "read_values_from_graph",
        "target": "read values from the supplied graph",
        "required_givens": ["the supplied graph"],
        "selected_variation_id": "blocked",
        "permitted_variations": [
            {
                "variation_id": "blocked",
                "operation": "read_values_from_graph",
                "target": "read values from the supplied graph",
                "givens": ["the supplied graph"],
                "question_types": ["mcq"],
                "student_work_product": "values",
                "requires_supplied_graph": True,
                "numeric_values_may_change": False,
            },
            {
                "variation_id": "supported",
                "operation": "explain_slope_meaning",
                "target": "explain what the slope represents",
                "givens": ["slope of a linear graph"],
                "question_types": ["mcq"],
                "student_work_product": "explanation",
                "numeric_values_may_change": False,
            },
        ],
    }
    updated = _reassign_supported_variation(item, "mcq")
    assert updated is not None
    selected = next(
        value for value in updated["permitted_variations"]
        if value["variation_id"] == updated["selected_variation_id"]
    )
    assert selected["target"] == "explain what the slope represents"
    assert not requires_unsupported_supplied_graph(updated)
    # The original plan entry is never mutated in place.
    assert item.get("selected_variation_id") == "blocked"
    assert item.get("target") == "read values from the supplied graph"


def test_generator_payload_includes_subject_unit_policy_and_source_example():
    from app.online.prompts import _format_planned_item

    text = _format_planned_item(1, {
        "slot_id": "m1_equation_1",
        "subject": "mathematics",
        "exact_source_instance": False,
        "source_example_givens": ["2x + 5 = 15"],
        "task_operation": "solve_equation",
        "target": "value of x",
        "required_givens": ["linear equation in one variable"],
        "variable_givens": ["coefficient", "constant"],
        "permitted_variations": [],
        "allowed_variations": [],
    })
    assert "source_example_givens=" in text
    assert "subject=mathematics" in text
    assert "unit_policy=flexible" in text


def test_catalog_group1_diagnostics_report_graph_variation_and_units():
    catalog = {
        "assessment_profile": {"subject": "mathematics"},
        "tasks": [
            _task(),
            _task(
                concept="interpret a graph",
                knowledge_anchor="graph interpretation",
                task_operation="read_values_from_graph",
                target="read values from the supplied graph",
                student_work_product="values read from a graph",
                required_givens=["the graph shown below"],
                requires_supplied_graph=True,
                allowed_question_types=["mcq"],
                allowed_variations=[{
                    "target": "read values from the supplied graph",
                    "operation": "read_values_from_graph",
                    "givens": ["the graph shown below"],
                    "question_types": ["mcq"],
                    "reasoning_mode": "representation_interpretation",
                    "student_work_product": "values read from a graph",
                }],
            ),
        ],
    }
    diagnostics = catalog_group1_diagnostics(catalog)
    assert diagnostics["graph"]["excluded_variation_count"] >= 1
    assert diagnostics["variation"]["task_count"] == 2
    assert "policy_counts" in diagnostics["units"]


def test_unit_policy_is_subject_aware():
    assert unit_policy_for(
        "mathematics", exact_source_instance=False,
        operation="solve_equation", givens=["3x + 7 = 22"],
    ) == "flexible"
    assert unit_policy_for(
        "physics", exact_source_instance=False,
        operation="calculate_force", givens=["m = 5 kg"],
    ) == "constrained"
    assert unit_policy_for(
        "mathematics", exact_source_instance=True,
        operation="interpret_interval_table", givens=["tree height 48 ft"],
    ) == "essential"


def test_contradictory_source_units_are_detected():
    findings = detect_unit_conflict(
        "| الارتفاع (m) | أكبر من 72 | 56-72 |\n"
        "حدد فئة الجزء العلوي لشجرة يبلغ ارتفاعها 48 ft"
    )
    assert findings and findings[0]["code"] == "source_unit_conflict"
    assert findings[0]["systems"] == ["imperial", "metric"]
    assert detect_unit_conflict("The tree is 48 ft tall.") == []


# ---------------------------------------------------------------------------
# Validator wiring
# ---------------------------------------------------------------------------

def _protected_item(evidence_text: str, task_extra: dict | None = None) -> ProtectedQuestion:
    task = {
        "task_operation": "solve_equation",
        "target": "value of x",
        "selected_variation_id": "variation_generic",
        "permitted_variations": [{
            "variation_id": "variation_generic",
            "operation": "solve_equation",
            "target": "value of x",
            "givens": ["ax + b = c"],
            "numeric_values_may_change": True,
            "evidence_chunk_ids": ["c1"],
        }],
        "exact_source_instance": False,
        "subject": "mathematics",
        "source_example_givens": ["2x + 5 = 15"],
    }
    task.update(task_extra or {})
    return ProtectedQuestion(
        question_id="q1",
        question_type="equation",
        slot_id="m1_equation_1",
        question={
            "question_id": "q1",
            "slot_id": "m1_equation_1",
            "equation": "3x + 7 = 22",
            "solution_steps": ["3x = 15", "x = 5"],
            "final_answer": "x = 5",
            "solution_certificate": None,
        },
        task=task,
        evidence={"c1": evidence_text},
        word_bank=[],
        sanitization_events=[],
        errors=[],
        requires_quantitative=True,
    )


def test_copy_detection_is_registered_on_active_validator_roles():
    assert "copied_source_example" in ROLE_FAILURE_CODES["grounding"]
    assert "copied_source_example" in ROLE_FAILURE_CODES["task_alignment"]


def test_validator_prompts_state_the_copy_rule_and_unit_policy():
    grounding = _system_prompt("grounding")
    assert "copied_source_example" in grounding
    assert "unit_contract" in grounding
    assert "unit_contract" in _system_prompt("task_alignment")
    assert "unit_contract" in _system_prompt("quantitative")


def test_validator_payload_carries_source_example_subject_and_unit_context():
    item = _protected_item(
        "| الارتفاع (m) | 56-72 |\nitem: the tree is 48 ft tall."
    )
    payload = _payload_for("task_alignment", item)
    assert "source_example_givens" not in payload["task_contract"]
    assert "2x + 5 = 15" in payload["task_contract"]["illustrative_values"]
    assert payload["unit_contract"]["unit_policy"] == "flexible"
    assert payload["unit_contract"]["source_unit_conflict"]
    grounding = _payload_for("grounding", item)
    assert set(grounding["grounding_contract"]) == {
        "task_operation", "target", "selected_variation_id",
        "numeric_values_may_change", "exact_source_instance",
        "permitted_optimization_direction", "selected_variation_givens",
        "selected_variation_evidence_chunk_ids",
        "uses_effective_givens",
        "effective_instance_consistent", "numeric_adjustment_authorized",
        "value_policy", "illustrative_values", "immutable_values",
    }
    assert grounding["unit_contract"]["subject"] == "mathematics"


# ---------------------------------------------------------------------------
# Repair dependency synchronization
# ---------------------------------------------------------------------------

def test_material_stem_change_ignores_wording_only_edits():
    assert _material_stem_change(
        "equation", {"equation": "2x + 5 = 15"}, {"equation": "3x + 7 = 25"},
    )
    assert not _material_stem_change(
        "equation", {"equation": "2x + 5 = 15"}, {"equation": "Solve 2x + 5 = 15."},
    )


def test_dependent_steps_cover_every_qtype_contract():
    assert [step.agent for step in _with_dependent_steps("equation", [])] == ["solution"]
    assert [step.agent for step in _with_dependent_steps("mcq", [])] == [
        "options", "solution", "answer",
    ]
    assert [step.agent for step in _with_dependent_steps("true_false", [])] == [
        "solution", "answer",
    ]
    # An existing step that already owns a stem field is widened, not duplicated.
    widened = _with_dependent_steps(
        "equation", [RepairStep(agent="solution", issues=(), fields=("equation",))],
    )
    assert len(widened) == 1
    assert set(widened[0].fields) == {
        "equation", "solution_steps", "final_answer", "solution_certificate",
    }


def test_stale_certificate_is_rejected_and_recomputed_content_is_allowed():
    before = {
        "solution_steps": ["2x = 10", "x = 5"], "final_answer": "x = 5",
        "solution_certificate": {"final_answer": "x = 5"},
    }
    stale = {
        "solution_steps": ["3x = 18", "x = 6"], "final_answer": "x = 6",
        "solution_certificate": {"final_answer": "x = 5"},
    }
    fresh = {
        "solution_steps": ["3x = 18", "x = 6"], "final_answer": "x = 6",
        "solution_certificate": {"final_answer": "x = 6"},
    }
    assert _stale_dependent_error(
        "equation", {"equation": "2x + 5 = 15", **before},
        {"equation": "3x + 7 = 25", **stale},
    )
    assert _stale_dependent_error(
        "equation", {"equation": "2x + 5 = 15", **before},
        {"equation": "3x + 7 = 25", **fresh},
    ) is None


def test_wording_only_question_route_stays_single_step():
    route = build_repair_route("mcq", {
        "question_id": "q1",
        "required_fixes": [{
            "validator": "answer_correctness", "validation_status": "FAIL",
            "failure_code": "ambiguous_or_structural", "fields_to_fix": ["question"],
            "reason": "unclear", "expected_fix": "clarify wording",
        }],
        "fields_to_fix": ["question"],
    })
    assert [step.agent for step in route["steps"]] == ["question"]
    assert not any(step.dependent for step in route["steps"])


def _metadata(field: str, before, after, code: str) -> dict:
    return {
        "modified_fields": [field],
        "changes": [{"field": field, "previous_value": before, "new_value": after,
                     "reason": f"repairs {code}"}],
        "targeted_failure_codes": [code],
    }


def _equation_exam() -> dict:
    return {
        "model_number": 1,
        "questions": {"equation": [{
            "question_id": "q1",
            "slot_id": "m1_equation_1",
            "equation": "2x + 5 = 15",
            "solution_steps": ["2x = 10", "x = 5"],
            "final_answer": "x = 5",
            "solution_certificate": {
                "method": "isolate x",
                "important_givens": ["2x + 5 = 15"],
                "calculation_steps": ["2x = 10", "x = 5"],
                "final_answer": "x = 5",
                "units": "",
                "matching_correct_option": None,
                "verification_method": "substitute x = 5",
            },
        }]},
    }


def _copied_verdict() -> dict:
    return {
        "question_id": "q1",
        "validation_status": "FAIL",
        "action": "FIX_QUESTION_AND_ANSWER",
        "failure_code": "copied_source_example",
        "fields_to_fix": ["equation"],
        "required_fixes": [{
            "validator": "grounding",
            "validation_status": "FAIL",
            "failure_code": "copied_source_example",
            "fields_to_fix": ["equation"],
            "reason": "The question copies the source example.",
            "expected_fix": "Construct a new numerical instance.",
        }],
        "repair_instruction_conflicts": [],
    }


def test_material_question_repair_synchronizes_answer_and_solution(monkeypatch):
    from app.online import multi_repair

    calls: list[str] = []

    def fake_request(_client, _prompt, **kwargs):
        agent = kwargs["agent"]
        calls.append(agent)
        if agent == "question_repairer":
            return {"repairs": [{
                "question_id": "q1",
                "patch": {"equation": "3x + 7 = 25"},
                "repair_metadata": _metadata(
                    "equation", "2x + 5 = 15", "3x + 7 = 25", "copied_source_example",
                ),
            }]}
        if agent == "solution_repairer":
            certificate = {
                "method": "isolate x",
                "important_givens": ["3x + 7 = 25"],
                "calculation_steps": ["3x = 18", "x = 6"],
                "final_answer": "x = 6",
                "units": "",
                "matching_correct_option": None,
                "verification_method": "substitute x = 6",
            }
            return {"repairs": [{
                "question_id": "q1",
                "patch": {
                    "equation": "3x + 7 = 25",
                    "solution_steps": ["3x = 18", "x = 6"],
                    "final_answer": "x = 6",
                    "solution_certificate": certificate,
                },
                "repair_metadata": {
                    "modified_fields": [
                        "solution_steps", "final_answer", "solution_certificate",
                    ],
                    "changes": [
                        {"field": "final_answer", "previous_value": "x = 5",
                         "new_value": "x = 6", "reason": "recomputed"},
                    ],
                    "targeted_failure_codes": ["copied_source_example"],
                },
            }]}
        raise AssertionError(f"unexpected agent {agent}")

    monkeypatch.setattr(multi_repair, "request_structured", fake_request)
    exam = _equation_exam()
    result = repair_question_specialized(
        exam=exam,
        verdict=_copied_verdict(),
        qtype="equation",
        plan_item={
            "task_id": "t1", "selected_variation_id": "v1",
            "source_chunk_id": "c1", "difficulty": "easy",
        },
        source_chunks={"c1": "Solve 2x + 5 = 15."},
        domain="STEM",
        client_factory=lambda _agent: object(),
    )

    assert result["repaired_ids"] == ["q1"]
    assert "solution" in result["repair_route"]
    row = exam["questions"]["equation"][0]
    assert row["equation"] == "3x + 7 = 25"
    assert row["final_answer"] == "x = 6"
    assert row["solution_certificate"]["final_answer"] == "x = 6"
    metadata = result["repair_metadata"]["q1"]
    assert metadata["dependency_synchronized"] is True
    assert metadata["dependent_steps"]
    assert calls == ["question_repairer", "solution_repairer"]


def test_stale_certificate_is_rejected_before_commit(monkeypatch):
    from app.online import multi_repair

    original_certificate = _equation_exam()["questions"]["equation"][0][
        "solution_certificate"
    ]

    def fake_request(_client, _prompt, **kwargs):
        agent = kwargs["agent"]
        if agent == "question_repairer":
            return {"repairs": [{
                "question_id": "q1",
                "patch": {"equation": "3x + 7 = 25"},
                "repair_metadata": _metadata(
                    "equation", "2x + 5 = 15", "3x + 7 = 25", "copied_source_example",
                ),
            }]}
        return {"repairs": [{
            "question_id": "q1",
            "patch": {
                "equation": "3x + 7 = 25",
                "solution_steps": ["3x = 18", "x = 6"],
                "final_answer": "x = 6",
                "solution_certificate": original_certificate,
            },
            "repair_metadata": {
                "modified_fields": ["solution_steps", "final_answer"],
                "changes": [{
                    "field": "final_answer", "previous_value": "x = 5",
                    "new_value": "x = 6", "reason": "recomputed steps only",
                }],
                "targeted_failure_codes": ["copied_source_example"],
            },
        }]}

    monkeypatch.setattr(multi_repair, "request_structured", fake_request)
    exam = _equation_exam()
    result = repair_question_specialized(
        exam=exam,
        verdict=_copied_verdict(),
        qtype="equation",
        plan_item={"task_id": "t1", "selected_variation_id": "v1",
                   "source_chunk_id": "c1", "difficulty": "easy"},
        source_chunks={"c1": "Solve 2x + 5 = 15."},
        domain="STEM",
        client_factory=lambda _agent: object(),
    )

    assert result["repaired_ids"] == []
    assert result["route_status"] == "DEPENDENCY_SYNC_REJECTED"
    assert any("not synchronized" in warning for warning in result["warnings"])
    # Transactional: the staged exam keeps the original question.
    assert exam["questions"]["equation"][0]["equation"] == "2x + 5 = 15"


def test_actual_dependency_scheduler_uses_modified_values_not_authorized_fields():
    from app.online.multi_repair import _actual_dependent_steps

    before = _equation_exam()["questions"]["equation"][0]
    after = {**before, "equation": "3x + 7 = 25"}
    steps = _actual_dependent_steps("equation", before, after)
    assert [(step.agent, set(step.fields)) for step in steps] == [(
        "solution", {"solution_steps", "final_answer", "solution_certificate"},
    )]


def test_duplicate_fingerprint_tracks_repaired_content():
    before = _equation_exam()["questions"]["equation"][0]
    after = dict(before)
    after["equation"] = "3x + 7 = 25"
    assert question_fingerprint(before) != question_fingerprint(after)
