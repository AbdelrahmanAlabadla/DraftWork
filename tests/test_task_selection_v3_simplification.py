from __future__ import annotations

from app.online.assessment_catalog import rank_task_variation_candidates
from app.online.models import authoritative_givens, numeric_variation_allowed
from app.online.multi_repair import _stale_dependent_error


def _candidate(task_id: str, level: str, variation_id: str, qtype: str = "mcq"):
    return (
        {"task_id": task_id, "concept": "ignored by ranking"},
        level,
        {"variation_id": variation_id, "question_types": [qtype]},
    )


def test_unused_task_outranks_exact_difficulty_used_task():
    used_easy = _candidate("task_a", "easy", "v1")
    unused_medium = _candidate("task_z", "medium", "v2")
    ranked = rank_task_variation_candidates(
        [used_easy, unused_medium], question_type="mcq",
        requested_difficulty="easy", task_usage={"task_a": 1, "task_z": 0},
    )
    assert ranked == [unused_medium, used_easy]


def test_task_usage_is_shared_by_all_variations_and_difficulty_breaks_usage_ties():
    medium = _candidate("task_same", "medium", "v2")
    easy = _candidate("task_same", "easy", "v1")
    ranked = rank_task_variation_candidates(
        [medium, easy], question_type="mcq", requested_difficulty="easy",
        task_usage={"task_same": 3},
    )
    assert ranked == [easy, medium]


def test_question_type_is_an_eligibility_filter():
    equation = _candidate("task_a", "easy", "v1", "equation")
    mcq = _candidate("task_b", "hard", "v2", "mcq")
    assert rank_task_variation_candidates(
        [equation, mcq], question_type="mcq", requested_difficulty="easy",
        task_usage={},
    ) == [mcq]


def test_ordinary_numbers_are_flexible_but_exact_source_values_are_fixed():
    assert numeric_variation_allowed({"numeric_values_may_change": False}) is True
    assert numeric_variation_allowed({"exact_source_instance": True}) is False
    assert numeric_variation_allowed({"protected_source_data": True}) is False


def test_effective_instance_is_the_single_active_generation_validation_contract():
    task = {
        "selected_variation_id": "v1",
        "selected_variation": {"variation_id": "v1", "givens": ["x + 1 = 2"]},
        "effective_instance_id": "effective-1",
        "effective_givens": ["2x + 3 = 9"],
        "difficulty_design": {
            "status": "designed", "effective_instance_id": "effective-1",
            "adjustment_kind": "numerical",
        },
    }
    contract = authoritative_givens(task)
    assert contract["givens"] == ["2x + 3 = 9"]
    assert contract["source_givens"] == ["x + 1 = 2"]
    assert contract["numeric_adjustment_authorized"] is True


def test_equation_repair_rejects_each_stale_dependent_field():
    before = {
        "equation": "x + 1 = 2", "solution_steps": ["x=1"],
        "final_answer": "1", "solution_certificate": {"check": "1+1=2"},
    }
    after = {
        **before, "equation": "x + 2 = 4", "solution_steps": ["x=2"],
        "final_answer": "2",
    }
    error = _stale_dependent_error("equation", before, after)
    assert error and "solution_certificate" in error
