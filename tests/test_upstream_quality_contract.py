from __future__ import annotations

from app.online import assessment_catalog, prompts


def _variation(**overrides) -> dict:
    value = {
        "variation_id": "variation-linear",
        "variation_group_id": "variation-group-linear",
        "question_types": ["equation"],
        "target": "value of x",
        "operation": "solve_equation",
        "givens": ["2x + 6 = 18"],
        "reasoning_mode": "symbolic_manipulation",
        "student_work_product": "worked solution",
        "required_response_components": ["worked_solution"],
        "numeric_values_may_change": True,
        "evidence_chunk_ids": ["chunk-1"],
    }
    value.update(overrides)
    return value


def _plan() -> dict:
    variation = _variation()
    return {
        "slot_id": "m1_equation_1",
        "task_id": "task-linear",
        "concept": "linear equality",
        "task_operation": "solve_equation",
        "target": "value of x",
        "student_work_product": "worked solution",
        "required_givens": ["preserve equality"],
        "variable_givens": ["coefficients"],
        "complexity_boundaries": ["one variable; linear only"],
        "selected_variation_id": "variation-linear",
        "permitted_variations": [variation],
        "difficulty": "medium",
        "requested_difficulty": "medium",
        "exact_source_instance": False,
    }


def test_invalid_variation_is_recorded_without_discarding_valid_sibling():
    diagnostics: list[dict] = []

    normalized = assessment_catalog._normalize_variations(
        [_variation(), {"target": "missing operation"}],
        rejection_diagnostics=diagnostics,
    )

    assert len(normalized) == 1
    assert normalized[0]["operation"] == "solve_equation"
    assert diagnostics[0]["code"] == "invalid_variation_schema"
    assert "operation" in diagnostics[0]["reason"]


def test_adjusted_variation_overrides_only_the_effective_instance_givens():
    item = _plan()
    item.update({
        "effective_instance_id": "derived-linear",
        "effective_givens": ["3x + 7 = 22"],
        "difficulty_design": {
            "status": "designed",
            "effective_instance_id": "derived-linear",
        },
    })

    effective, origin = prompts._effective_variation_contract(item)

    assert origin == "difficulty_adjusted"
    assert effective["givens"] == ["3x + 7 = 22"]
    assert effective["operation"] == "solve_equation"
    assert item["permitted_variations"][0]["givens"] == ["2x + 6 = 18"]


def test_original_variation_is_used_when_adjustment_is_not_valid():
    item = _plan()
    item.update({
        "effective_instance_id": "derived-linear",
        "effective_givens": ["3x + 7 = 22"],
        "difficulty_design": {
            "status": "designed",
            "effective_instance_id": "different-instance",
        },
    })

    effective, origin = prompts._effective_variation_contract(item)

    assert origin == "catalog"
    assert effective["givens"] == ["2x + 6 = 18"]


def test_rendered_generator_contract_contains_only_effective_variation_values():
    item = _plan()
    item.update({
        "effective_instance_id": "derived-linear",
        "effective_givens": ["3x + 7 = 22"],
        "difficulty_design": {
            "status": "designed",
            "effective_instance_id": "derived-linear",
        },
    })

    rendered = prompts._format_planned_item(1, item)

    selected_contract = rendered.split("selected_variation=", 1)[1].split(" | ", 1)[0]
    assert "3x + 7 = 22" in selected_contract
    assert "2x + 6 = 18" not in selected_contract
    assert "effective_variation_origin=difficulty_adjusted" in rendered


def test_generator_prompt_requires_conceptual_novel_instances_and_exact_values():
    assert "conceptual blueprint" in prompts.SYSTEM_PROMPT
    assert "2x + 6 = 18" in prompts.SYSTEM_PROMPT
    assert "ax + b = c" in prompts.SYSTEM_PROMPT
    assert "exact_source_instance=true" in prompts.SYSTEM_PROMPT
    assert "Recompute the answer, all options" in prompts.SYSTEM_PROMPT


def test_variation_blueprint_is_preserved_and_reaches_generator_contract():
    variation = _variation(
        concept="Solve a one-variable linear equality",
        equation_structure="ax + b = c",
        variables=["a: nonzero coefficient", "b: constant", "c: total"],
        constraints=["a != 0", "generated values have a valid solution"],
    )
    normalized = assessment_catalog._normalize_variations([variation])[0]
    item = _plan()
    item["permitted_variations"] = [normalized]
    item["selected_variation_id"] = normalized["variation_id"]
    rendered = prompts._format_planned_item(1, item)
    assert "ax + b = c" in rendered
    assert "a: nonzero coefficient" in rendered
    assert "generated values have a valid solution" in rendered


def test_primary_variation_inherits_declared_task_blueprint_without_semantic_rewrite():
    task = {
        "concept": "total cost with a fixed fee", "target": "number of items",
        "task_operation": "solve_for_item_count", "allowed_question_types": ["word_problem"],
        "primary_question_types": ["word_problem"], "required_givens": [],
        "evidence_chunk_ids": ["chunk-1"], "reasoning_mode": "quantitative_reasoning",
        "student_work_product": "worked solution", "exact_source_instance": False,
        "equation_structure": "pn + f = T",
        "variable_parameters": ["p: price per item", "n: item count", "f: fixed fee", "T: total"],
        "validity_constraints": ["p > 0", "n is a positive integer"],
    }
    variation = assessment_catalog._primary_variations(task)[0]
    assert variation["equation_structure"] == "pn + f = T"
    assert variation["variables"] == task["variable_parameters"]
    assert variation["constraints"] == task["validity_constraints"]
