from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.llm.schemas import generation_response_model, objective_bundle_response_model
from app.online import assessment_catalog as catalog
from app.online import exam_builder
from app.online import prompts
from app.online import validator


def _mcq(*, question="Which energy principle applies?", correct="Energy is conserved", **options):
    values = {
        "A": correct,
        "B": "Energy changes form while total energy remains constant",
        "C": "Energy transfer can increase one store while decreasing another",
        "D": "Measured energy may differ because some transfers are overlooked",
    }
    values.update(options)
    return {"question": question, "options": values, "correct_answer": "A"}


def _task(task_id: str, *, concept: str = "energy conservation") -> dict:
    return {
        "task_id": task_id,
        "task_family_id": f"family_{task_id}",
        "coverage_id": f"coverage_{task_id}",
        "concept_id": f"concept_{task_id}",
        "knowledge_anchor": concept,
        "topic": "energy",
        "concept": concept,
        "relationship": "energy changes form while total energy is conserved",
        "task_operation": "infer_relationship",
        "student_work_product": "select the best inference",
        "target": "conservation across an energy transfer",
        "reasoning_mode": "conceptual_inference",
        "supported_difficulties": ["medium"],
        "allowed_question_types": ["mcq"],
        "evidence_chunk_ids": ["c1"],
        "required_givens": [],
        "scenario_family": None,
        "text_answerable": True,
    }


def test_obviously_irrelevant_distractors_are_rejected():
    item = _mcq(
        B="Bananas grow in tropical climates",
        C="Paris contains several large museums",
        D="Copper conducts electric current well",
    )
    verdict = validator._local_verdict("mcq", item, [item])
    assert verdict and verdict["action"] == "FIX_OPTIONS"
    assert "unrelated" in verdict["reason"]


def test_plausible_same_topic_distractors_pass_local_quality_checks():
    assert validator._local_verdict("mcq", _mcq(), []) is None


def test_stem_that_reveals_complete_answer_is_rejected():
    item = _mcq(
        question="Energy is conserved in a closed system. Which energy principle applies?",
        correct="Energy is conserved in a closed system",
    )
    verdict = validator._local_verdict("mcq", item, [item])
    assert verdict and verdict["action"] == "FIX_QUESTION"
    assert verdict["fields_to_fix"] == ["question"]


def test_visibly_longer_correct_option_is_flagged():
    item = _mcq(
        correct="Energy remains constant because every transfer is fully accounted for across all stores in the closed system",
        B="It increases", C="It disappears", D="It is created",
    )
    verdict = validator._local_verdict("mcq", item, [item])
    assert verdict and verdict["action"] == "FIX_OPTIONS"
    assert "stands out" in verdict["reason"]


def test_two_equivalent_options_are_rejected_and_semantic_rule_is_explicit():
    item = _mcq(B="Energy is conserved")
    verdict = validator._local_verdict("mcq", item, [item])
    assert verdict and verdict["action"] == "FIX_OPTIONS"
    assert "multiple defensible answers" in validator.VALIDATOR_SYSTEM_PROMPT


def test_medium_and_hard_difficulty_come_from_reasoning():
    assert "Difficulty must come from the assigned reasoning" in prompts.MCQ_RULES
    assert "Never create difficulty through long wording" in prompts.MCQ_RULES
    assert "difficulty created only by long/confusing wording" in validator.VALIDATOR_SYSTEM_PROMPT


def test_quantitative_mcq_contract_solves_first_and_uses_realistic_errors():
    assert "independently solve the assigned task" in prompts.MCQ_RULES
    for error in ("sign", "unit conversion", "formula", "substitution", "intermediate result"):
        assert error in prompts.MCQ_RULES
    assert "solution certificate" in prompts.MCQ_RULES
    assert "solve first" in validator.STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT


def test_deterministic_answer_position_balancing_remains_intact():
    questions = {"mcq": [
        {**_mcq(), "question_id": f"q{i}"} for i in range(4)
    ]}
    exam_builder.balance_mcq_answer_positions(questions, 1)
    assert [item["correct_answer"] for item in questions["mcq"]] == ["B", "D", "A", "C"]


def test_generation_balances_before_graph_validation_and_repairs_rebalance_before_revalidation():
    generation_source = inspect.getsource(exam_builder.generate_exams_node)
    assert generation_source.index("balance_mcq_answer_positions") < generation_source.index(
        'generated_exams.append'
    )
    repair_source = inspect.getsource(validator.repair_invalid_questions)
    assert repair_source.index("balance_mcq_answer_positions") < repair_source.index(
        'pending_revalidation[mn] = all_repaired'
    )


def test_mcq_prompt_preserves_catalog_assessment_identity():
    task = catalog.materialize_plan_item(_task("t1"), "m1_mcq_1", "medium")
    task["assigned_sources"] = {"c1": "Energy changes form but is conserved."}
    _system, user = prompts.build_prompt(
        "mcq", 1, "fallback", difficulty="medium", model_number=1,
        planned_items=[task], language="en",
    )
    assert "concept=energy conservation" in user
    assert "task_operation=infer_relationship" in user
    assert "target=conservation across an energy transfer" in user
    assert "reasoning_mode=conceptual_inference" in user
    assert "Energy changes form but is conserved." in user


def test_mcq_schema_returns_structured_unsuitable_task_instead_of_forcing_question():
    model = generation_response_model("mcq", ["m1_mcq_1"])
    value = model.model_validate({
        "questions": [],
        "unsuitable_tasks": [{
            "slot_id": "m1_mcq_1",
            "failure_code": "mcq_task_unsuitable",
            "reason": "The task has no plausible competing alternatives.",
        }],
    }).model_dump()
    assert value["questions"] == []
    assert value["unsuitable_tasks"][0]["failure_code"] == "mcq_task_unsuitable"


def test_bundled_mcq_contract_nests_unsuitable_tasks_inside_mcq():
    model = objective_bundle_response_model(["m1_mcq_1"], ["m1_true_false_1"])
    value = model.model_validate({
        "mcq": {"questions": [], "unsuitable_tasks": []},
        "true_false": {"questions": [{
            "slot_id": "m1_true_false_1", "statement": "Energy is conserved.",
            "answer": "True",
        }]},
    }).model_dump()
    assert value["mcq"]["unsuitable_tasks"] == []
    with pytest.raises(ValidationError):
        model.model_validate({
            "mcq": {"questions": []},
            "unsuitable_tasks": [],
            "true_false": {"questions": [{
                "slot_id": "m1_true_false_1", "statement": "Energy is conserved.",
                "answer": "True",
            }]},
        })


def test_bundled_prompt_shows_exact_nested_contract_for_mcq_only():
    task = catalog.materialize_plan_item(_task("t1"), "m1_mcq_1", "medium")
    task["assigned_sources"] = {"c1": "Energy is conserved."}
    _system, user = prompts.build_obj_bundled_prompt(
        {"mcq": [task], "true_false": []}, "fallback", model_number=1,
    )
    required = user.split("## Required JSON Format (use exactly this nesting)\n", 1)[1]
    rendered = required.split("\n\nThe `unsuitable_tasks`", 1)[0]
    shape = json.loads(rendered)
    assert set(shape) == {"mcq"}
    assert shape["mcq"]["unsuitable_tasks"] == []
    assert "Never return `unsuitable_tasks` at the top level" in user
    assert "Even when this request contains only MCQs, keep the outer `mcq` object" in user
    task_instruction = user.split("## Task\n", 1)[1].split("\n\n", 1)[0]
    assert "`mcq` (1 questions)" in task_instruction
    assert "true_false" not in task_instruction


def test_unsuitable_task_is_reassigned_and_does_not_consume_slot(monkeypatch):
    old = catalog.materialize_plan_item(_task("old"), "m1_mcq_1", "medium")
    old["assigned_sources"] = {"c1": "evidence"}
    replacement = {**old, "task_id": "new", "concept": "energy transfer"}
    calls = []

    def fake_request(_client, prompt, **_kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return {"mcq": {"questions": [], "unsuitable_tasks": [{
                "slot_id": "m1_mcq_1", "failure_code": "mcq_task_unsuitable",
                "reason": "No plausible alternatives",
            }]}}
        return {"mcq": {"questions": [{
            "slot_id": "m1_mcq_1", "question": "Which transfer conserves total energy?",
            "options": {"A": "Transfer between stores", "B": "Creation from nothing", "C": "Loss from existence", "D": "Uncaused increase"},
            "correct_answer": "A", "solution_certificate": None,
        }], "unsuitable_tasks": []}}

    monkeypatch.setattr("app.llm.factory.create_llm_client", lambda: object())
    monkeypatch.setattr(exam_builder, "request_structured", fake_request)
    seen = set()
    bundle, warnings = exam_builder._generate_obj_bundle(
        {"mcq": [old]}, "evidence", "medium", 1, [], seen, [],
        global_plan_by_slot={"m1_mcq_1": old},
        mcq_task_replacer=lambda _sid, _old, _reason: replacement,
    )
    assert len(bundle["mcq"]) == 1
    assert bundle["mcq"][0]["slot_id"] == "m1_mcq_1"
    assert len(calls) == 2
    assert any("reassigned slot m1_mcq_1" in warning for warning in warnings)


def test_allocator_excludes_unsuitable_task_and_selects_another_compatible_task():
    old_task, new_task = _task("old"), _task("new", concept="energy transfer")
    plans = [{"model_number": 1, "items": {"mcq": [
        catalog.materialize_plan_item(old_task, "m1_mcq_1", "medium")
    ]}}]
    replacement = catalog.catalog_exhaustion_replacement(
        {"tasks": [old_task, new_task]}, plans, "m1_mcq_1", "mcq", "medium",
        excluded_task_ids={"old"},
    )
    assert replacement and replacement["task_id"] == "new"


def test_no_removed_question_type_contract_remains_in_active_pipeline():
    removed = "short" + "_answer"
    root = Path(__file__).parents[1]
    active = [root / "app", root / "FrontEnd", root / "scripts", root / "run.py"]
    matches = []
    for base in active:
        paths = [base] if base.is_file() else list(base.rglob("*"))
        for path in paths:
            if path.is_file() and path.suffix.lower() in {".py", ".js", ".jsx", ".ts", ".tsx"}:
                if removed in path.read_text(encoding="utf-8", errors="ignore").casefold():
                    matches.append(str(path.relative_to(root)))
    assert matches == []
