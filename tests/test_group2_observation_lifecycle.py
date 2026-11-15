"""Group 2 observation lifecycle regressions."""

from __future__ import annotations

from copy import deepcopy

from app.online.exam_builder import assemble_exams_node
from app.online.lifecycle import initialize_slot_lifecycle, question_fingerprint
from app.online.validator import persist_difficulty_observations


def _question(text: str = "2x + 3 = 11") -> dict:
    return {
        "question_id": "model1_equation_1",
        "slot_id": "m1_equation_1",
        "equation": text,
        "solution_steps": ["2x = 8", "x = 4"],
        "final_answer": "4",
        "solution_certificate": None,
    }


def _plan() -> dict:
    return {
        "slot_id": "m1_equation_1",
        "task_id": "task_1",
        "base_catalog_task_id": "task_1",
        "requested_difficulty": "easy",
        "actual_difficulty": "medium",
        "difficulty": "medium",
        "difficulty_design": {"status": "designed"},
    }


def _observation(actual: str = "easy") -> dict:
    return {
        "conceptual_difficulty": "easy",
        "numerical_difficulty": "easy",
        "actual_difficulty": actual,
        "difficulty_basis": "direct isolation",
        "validator": "difficulty",
    }


def _report(observation: dict | None, *, accepted: bool = False) -> dict:
    verdict = {
        "question_id": "model1_equation_1",
        "action": "PASS",
        "validation_status": "PASS",
        "failure_code": "pass",
        "difficulty_observation": observation,
    }
    if accepted:
        verdict["difficulty_deviation"] = {
            "status": "accepted_source_constraint",
            "actual_difficulty": (observation or {}).get("actual_difficulty"),
            "reason": "source ceiling",
        }
    return {"model_number": 1, "verdicts": [verdict]}


def test_normal_pass_persists_observation_with_fingerprint():
    plans = [{"model_number": 1, "items": {"equation": [_plan()]}}]
    exam = {"model_number": 1, "questions": {"equation": [_question()]}}
    persist_difficulty_observations(plans, _report(_observation("easy")), exam)
    stored = plans[0]["items"]["equation"][0]["difficulty_observation"]
    assert stored["actual_difficulty"] == "easy"
    assert stored["question_fingerprint"] == question_fingerprint(_question())
    assert plans[0]["items"]["equation"][0]["actual_difficulty"] == "easy"


def test_accepted_constraint_persists_observed_level_and_status():
    plans = [{"model_number": 1, "items": {"equation": [_plan()]}}]
    exam = {"model_number": 1, "questions": {"equation": [_question()]}}
    persist_difficulty_observations(
        plans, _report(_observation("medium"), accepted=True), exam,
    )
    item = plans[0]["items"]["equation"][0]
    assert item["actual_difficulty"] == "medium"
    assert item["requested_difficulty"] == "easy"
    assert item["difficulty_status"] == "accepted_source_constraint"
    assert item["difficulty_design"]["status"] == "accepted_source_constraint"


def test_stale_fingerprint_is_dropped_before_persisting_new_one():
    plans = [{"model_number": 1, "items": {"equation": [_plan()]}}]
    plans[0]["items"]["equation"][0]["difficulty_observation"] = {
        **_observation("easy"), "question_fingerprint": "old-fingerprint",
    }
    exam = {"model_number": 1, "questions": {"equation": [_question("3x + 4 = 19")]}}
    persist_difficulty_observations(plans, _report(_observation("medium")), exam)
    stored = plans[0]["items"]["equation"][0]["difficulty_observation"]
    assert stored["actual_difficulty"] == "medium"
    assert stored["question_fingerprint"] == question_fingerprint(_question("3x + 4 = 19"))


def test_repair_change_without_revalidation_leaves_no_stale_observation():
    plans = [{"model_number": 1, "items": {"equation": [_plan()]}}]
    exam = {"model_number": 1, "questions": {"equation": [_question()]}}
    persist_difficulty_observations(plans, _report(_observation("easy")), exam)
    # Simulate a later content change and a report with no difficulty observation.
    changed = {"model_number": 1, "questions": {"equation": [_question("5x + 1 = 21")]}}
    persist_difficulty_observations(plans, _report(None), changed)
    item = plans[0]["items"]["equation"][0]
    assert "difficulty_observation" not in item


def test_final_assembly_drops_stale_observation_and_reports_it():
    plan = _plan()
    plan["difficulty_observation"] = {
        **_observation("easy"), "question_fingerprint": "stale",
    }
    state = {
        "document_language": "en",
        "generated_exams": [{
            "model_number": 1,
            "questions": {"equation": [_question("7x + 2 = 30")]},
        }],
        "plans": [{"model_number": 1, "items": {"equation": [plan]}}],
        "validation_reports": [_report(_observation("easy"))],
        "slot_lifecycle": initialize_slot_lifecycle([("equation", 1)], 1),
        "repair_manifest": [],
        "invalid_questions": [],
        "validation_repair_history": [],
    }
    # Mark the slot accepted in the lifecycle.
    for record in state["slot_lifecycle"].values():
        record["terminal_status"] = "finalized"
        record["accepted"] = True
    result = assemble_exams_node(state)
    question = result["generated_exams"][0]["questions"]["equation"][0]
    assert "difficulty_observation" not in question
    assert result["difficulty_observation_warnings"]
    assert result["difficulty_observation_warnings"][0]["slot_id"] == "m1_equation_1"


def test_repeated_validation_is_idempotent_for_same_question():
    plans = [{"model_number": 1, "items": {"equation": [_plan()]}}]
    exam = {"model_number": 1, "questions": {"equation": [_question()]}}
    for _ in range(3):
        persist_difficulty_observations(plans, _report(_observation("easy")), exam)
    item = plans[0]["items"]["equation"][0]
    assert item["difficulty_observation"]["actual_difficulty"] == "easy"
    assert item["actual_difficulty"] == "easy"
