from __future__ import annotations

import json

from app.llm.schemas import ValidatorVerdict
from app.online import prompts
from app.online import validator
from app.online.assessment_catalog import _catalog_prompt
from app.online.duplicate_judge import deterministic_pair_classification


def _planned_item() -> dict:
    return {
        "slot_id": "m1_equation_1",
        "source_chunk_id": "chunk-1",
        "support_chunk_ids": [],
        "topic": "linear equations",
        "concept": "solving a linear equation",
        "task_id": "task-1",
        "task_operation": "solve_equation",
        "relationship": "linear equality",
        "target": "unknown value",
        "reasoning_mode": "calculation",
        "difficulty": "medium",
        "required_givens": [],
        "assessment_guidance": {"constructed_givens_allowed": True},
        "assigned_sources": {"chunk-1": "A linear equation preserves equality."},
    }


def test_generator_contract_has_current_role_and_quantitative_self_check():
    system = prompts.SYSTEM_PROMPT.casefold()
    assert "exam-wide" not in system
    assert "duplicate questions" not in system
    assert "full selected source" not in system
    assert "neutral constructed numbers and givens are allowed" in system
    assert "solve it completely before returning" in system
    assert "exactly one option" in system
    assert "solution_certificate" in system
    assert "verification_method" in system
    assert "preserve its concept/knowledge target" in system
    assert "being on the same broad topic is not sufficient" in system
    assert "interpreting a common solution region" in system


def test_question_type_boundaries_match_current_architecture():
    assert "mention/list" in prompts.WHY_RULES.casefold()
    assert "cause/effect reasoning" in prompts.WHY_RULES.casefold()
    assert "mechanism" in prompts.WHY_RULES.casefold()
    equation_rules = prompts.EQUATION_RULES.casefold()
    assert "forming, solving, manipulating" in equation_rules
    assert "deriving an equation" in equation_rules
    assert "do not state the final relationship" in equation_rules
    assert "universal rearrange-and-isolate template" in equation_rules
    word_problem_rules = prompts.WORD_PROBLEM_RULES.casefold()
    assert "superficial name or story" in word_problem_rules
    assert "same scenario + values + requested result" in word_problem_rules
    assert "selecting or forming the governing relationship" in word_problem_rules
    assert "exactly one blank" in prompts.FILL_IN_THE_BLANK_ITEMS_RULES.casefold()
    assert "only the source term or concept" in prompts.DEFINITION_RULES.casefold()
    assert "synthesis" in prompts.ESSAY_RULES.casefold()


def test_fitb_generation_uses_per_slot_evidence_without_full_source_assumption():
    planned = [_planned_item()]
    _system, user = prompts.build_fitb_items_prompt(
        1, {"m1_equation_1": "equality"},
        "unused full context", planned_items=planned
    )
    folded = user.casefold()
    assert "full selected source" not in folded
    assert "assigned evidence" in folded
    assert "unused full context" not in user


def test_correctness_validator_schema_cannot_emit_duplicate_verdicts():
    schema = json.dumps(ValidatorVerdict.model_json_schema())
    assert "duplicate_or_repetition" not in schema


def test_general_and_stem_validator_roles_are_disjoint():
    general = validator.VALIDATOR_SYSTEM_PROMPT.casefold()
    quantitative = validator.STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT.casefold()
    assert "do not perform duplicate detection" in general
    assert "leave missing numerical/symbolic givens and calculations" in general
    assert "review only numerical and symbolic correctness" in quantitative
    assert "first_incorrect_step" in quantitative
    assert "independent_solution" in quantitative
    assert "certificate_valid" in quantitative
    assert "second verification method" in quantitative
    assert "nothing beyond it" in quantitative
    assert "do not automatically require every variable" in quantitative
    assert "expected_catalog_task" in general
    assert "failure_code=task_drift" in general
    assert "must not perform the operation assigned to the student" in general
    assert "same scenario + values + requested result" in general
    for excluded in ("difficulty", "wording", "grammar", "duplication", "planner metadata"):
        assert excluded in quantitative
    assert validator.VALIDATOR_RETRY_SYSTEM_PROMPT.startswith(
        validator.VALIDATOR_SYSTEM_PROMPT
    )
    assert validator.STEM_QUANTITATIVE_VALIDATOR_RETRY_SYSTEM_PROMPT.startswith(
        validator.STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT
    )
    assert json.loads(validator.QUANTITATIVE_VALIDATOR_OUTPUT_SCHEMA) == json.loads(
        validator.build_validator_prompt([], quantitative=True).split(
            "## Verdict object format (use exactly these field names)\n", 1
        )[1].split("\n\nReturn ONLY", 1)[0]
    )


def test_general_and_quantitative_repair_contracts_are_disjoint():
    general = validator.REPAIR_SYSTEM_PROMPT.casefold()
    quantitative = validator.STEM_QUANTITATIVE_REPAIR_SYSTEM_PROMPT.casefold()
    assert "never repair calculations, equations, units" in general
    assert "independently solve the repaired problem from scratch" in quantitative
    assert "diagnostic hypotheses, not authority" in quantitative
    assert "never edit planner metadata" in quantitative
    assert "duplication" in quantitative
    assert "never add the final relationship" in quantitative


def test_compiler_does_not_encode_worked_examples_or_force_rearrangement_templates():
    prompt = _catalog_prompt(
        "selected evidence", "English", ["c1"], domain_type="stem", subject="Chemistry"
    ).casefold()
    assert "forcing every equation task into rearranging" in prompt
    assert "complete scenario + exact values + requested result" in prompt
    assert "transferable operation and target" in prompt


def test_physics_duplicate_fact_patterns_map_deterministically():
    perpendicular_zero_work = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "yes",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "yes",
    }
    power_definition_vs_formula = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "yes",
    }
    assert deterministic_pair_classification(perpendicular_zero_work) == "duplicate"
    assert deterministic_pair_classification(power_definition_vs_formula) == "excessive_overlap"


def test_math_different_methods_remain_acceptable_reuse():
    elimination_vs_substitution = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "no",
    }
    assert deterministic_pair_classification(elimination_vs_substitution) == "acceptable_reuse"
