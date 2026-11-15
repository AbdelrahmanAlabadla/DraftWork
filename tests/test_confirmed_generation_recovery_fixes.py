from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.exports.docx_exporter import _answer_text as docx_answer_text
from app.exports.math_renderer import latex_to_plain_text, to_mathtext
from app.exports.pdf_exporter import _answer_text as pdf_answer_text
from app.llm.schemas import RepairTrueFalseQuestion, TrueFalseQuestion
from app.online import duplicate_pipeline
from app.online.assessment_catalog import (
    _finalize_catalog,
    apply_difficulty_fallback_metadata,
    materialize_plan_item,
)
from app.online.difficulty_adjustment import PlanDifficultyAdjustmentDecision
from app.online.exam_builder import _balanced_mcq_candidate
from app.online.models import _normalize_equation, _normalize_mcq


def _certificate(final_answer: str = "4") -> dict:
    return {
        "method": "addition",
        "important_givens": ["2", "2"],
        "calculation_steps": ["2+2=4"],
        "final_answer": final_answer,
        "units": "",
        "matching_correct_option": "A",
        "verification_method": "Subtract 2 from 4.",
    }


def _mcq(certificate: dict | None = None) -> dict:
    return {
        "question_id": "q1", "slot_id": "m1_mcq_1",
        "question": "What is 2+2?",
        "options": {"A": "4", "B": "5", "C": "6", "D": "7"},
        "correct_answer": "A", "solution_certificate": certificate or _certificate(),
    }


def test_mcq_balance_keeps_letter_independent_certificate_math_unchanged():
    original = _mcq()
    balanced = _balanced_mcq_candidate(original, "A", "D")
    assert balanced is not None
    assert balanced["correct_answer"] == "D"
    assert balanced["options"]["D"] == "4"
    assert balanced["solution_certificate"]["matching_correct_option"] == "D"
    assert balanced["solution_certificate"]["final_answer"] == "4"
    assert balanced["solution_certificate"]["calculation_steps"] == ["2+2=4"]


def test_mcq_balance_safely_remaps_legacy_structured_final_label():
    balanced = _balanced_mcq_candidate(_mcq(_certificate("A: 4")), "A", "D")
    assert balanced is not None
    assert balanced["solution_certificate"]["final_answer"] == "D: 4"


def test_mcq_balance_skips_ambiguous_legacy_references():
    certificate = _certificate("4")
    certificate["verification_method"] = "Option B is tempting but Option A is correct."
    assert _balanced_mcq_candidate(_mcq(certificate), "A", "D") is None

    certificate = _certificate("4")
    certificate["calculation_steps"] = ["A: use the first displayed equation"]
    assert _balanced_mcq_candidate(_mcq(certificate), "A", "D") is None


def _raw_task(*, exact: bool = False) -> dict:
    return {
        "knowledge_anchor": "Solve linear equations", "concept": "linear equations",
        "relationship": "ax+b=c", "task_operation": "solve_equation",
        "student_work_product": "solution", "target": "unknown value",
        "reasoning_mode": "symbolic_manipulation", "allowed_question_types": ["equation"],
        "evidence_chunk_ids": ["c1"], "required_givens": ["preserve linear structure"],
        "variable_givens": ["coefficient", "constant"], "complexity_boundaries": [],
        "allowed_variations": [{
            "target": "unknown value", "operation": "solve_equation", "givens": [],
            "question_types": ["equation"], "reasoning_mode": "symbolic_manipulation",
            "student_work_product": "solution", "numeric_values_may_change": True,
            "variation_group_id": None, "variation_id": None,
        }],
        "exact_source_instance": exact,
        "scenario_family": None, "text_answerable": True,
        "native_difficulty": "easy", "prerequisites": [],
    }


def test_catalog_policy_authorizes_numbers_for_generative_math_and_propagates_to_plan():
    catalog = _finalize_catalog(
        {"assessment_profile": {}, "tasks": [_raw_task()]},
        key="k", processed_document_key="d", selected_chunk_ids=["c1"], usage=None,
    )
    task = catalog["tasks"][0]
    assert task["allowed_variations"][0]["numeric_values_may_change"] is True
    plan = materialize_plan_item(
        task, "m1_equation_1", "easy",
        selected_variation_id=task["allowed_variations"][0]["variation_id"],
    )
    selected = next(v for v in plan["permitted_variations"]
                    if v["variation_id"] == plan["selected_variation_id"])
    assert selected["numeric_values_may_change"] is True


def test_exact_source_policy_keeps_numerical_values_protected():
    task = _raw_task(exact=True)
    task["allowed_variations"] = []
    catalog = _finalize_catalog(
        {"assessment_profile": {}, "tasks": [task]}, key="k", processed_document_key="d",
        selected_chunk_ids=["c1"], usage=None,
    )
    finalized = catalog["tasks"][0]
    assert finalized["exact_source_instance"] is True


def test_source_specific_dataset_without_mutable_parameters_stays_protected():
    task = _raw_task()
    task.update({
        "concept": "interpret the supplied race-time table",
        "target": "trend in the supplied observations",
        "task_operation": "interpret_dataset",
        "student_work_product": "interpretation of the supplied table",
        "required_givens": [
            "the supplied year/time table and its recorded observations",
            "table rows: 2001 -> 12.5, 2002 -> 13.1, 2003 -> 13.8",
        ],
        "variable_givens": [],
    })
    task["allowed_variations"][0].update({
        "target": task["target"], "operation": task["task_operation"],
        "student_work_product": task["student_work_product"],
        "numeric_values_may_change": True,
    })
    catalog = _finalize_catalog(
        {"assessment_profile": {}, "tasks": [task]}, key="k", processed_document_key="d",
        selected_chunk_ids=["c1"], usage=None,
    )
    assert catalog["tasks"][0]["allowed_variations"][0][
        "numeric_values_may_change"
    ] is False


def test_explicit_synthetic_dataset_parameters_can_vary_coherently():
    task = _raw_task()
    task.update({
        "concept": "construct and analyze a synthetic linear dataset",
        "target": "trend in a newly constructed coherent dataset",
        "task_operation": "construct_dataset",
        "variable_givens": ["starting value", "constant change", "observation count"],
    })
    task["allowed_variations"][0].update({
        "target": task["target"], "operation": task["task_operation"],
        "numeric_values_may_change": True,
    })
    catalog = _finalize_catalog(
        {"assessment_profile": {}, "tasks": [task]}, key="k", processed_document_key="d",
        selected_chunk_ids=["c1"], usage=None,
    )
    assert catalog["tasks"][0]["allowed_variations"][0][
        "numeric_values_may_change"
    ] is True


def test_true_false_provider_contract_is_native_boolean_but_legacy_normalizer_is_separate():
    assert TrueFalseQuestion.model_validate({
        "slot_id": "m1_true_false_1", "statement": "2+2=4", "answer": True,
    }).answer is True
    assert RepairTrueFalseQuestion.model_validate({
        "statement": "2+2=5", "answer": False,
    }).answer is False
    with pytest.raises(ValidationError):
        TrueFalseQuestion.model_validate({
            "slot_id": "m1_true_false_1", "statement": "2+2=4", "answer": "True",
        })


def test_false_answer_is_not_treated_as_missing_by_exports():
    item = {"qtype": "true_false", "answer": False}
    assert pdf_answer_text(item, "en") == "False"
    assert docx_answer_text(item, "en") == "False"


def test_one_fallback_builder_writes_complete_authoritative_state():
    item = {"slot_id": "m1_equation_1", "difficulty_adaptation": {}}
    apply_difficulty_fallback_metadata(item, "easy", "medium", "Conceptual floor.")
    assert item["difficulty_status"] == "accepted_source_constraint"
    assert item["difficulty_fallback"] is True
    assert item["requested_difficulty"] == "easy"
    assert item["actual_difficulty"] == "medium"
    assert item["difficulty_adaptation"]["further_adjustment_possible"] is False


def test_unsuccessful_difficulty_patch_becomes_attempt_diagnostic_not_applied_change():
    result = PlanDifficultyAdjustmentDecision.model_validate({
        "slot_id": "m1_equation_1", "adjustment_succeeded": False,
        "achieved_difficulty": "medium", "further_adjustment_possible": False,
        "adjusted_fields": {"given_conditions": ["3x+4=19"],
                            "reasoning_requirements": ["solve two-step equation"]},
        "evidence_support": [], "evidence_justification": "Still medium.",
        "reason": "Conceptual demand remains medium.",
    })
    assert result.adjusted_fields is None
    assert set(result.attempted_fields) == {"given_conditions", "reasoning_requirements"}


def test_selector_prompt_excludes_unrelated_exam_questions(monkeypatch):
    import app.llm.client as llm_client
    import app.llm.factory as llm_factory

    captured = {}
    monkeypatch.setattr(llm_factory, "create_duplicate_replacement_client", lambda: object())

    def fake_request(_client, prompt, **_kwargs):
        captured["context"] = json.loads(prompt.split("\n", 1)[1])
        captured["max_tokens"] = _kwargs["max_tokens"]
        return {"slot_to_replace": "m2_mcq_7", "conflicts_with": [],
                "conflict_reason": "failed candidate",
                "replacement_task_id": "task-new", "replacement_fitb_entry_id": None,
                "why_this_candidate_resolves_the_conflict": "best fit",
                "checked_against_active_slots": True, "rejected_candidates": []}

    monkeypatch.setattr(llm_client, "request_structured", fake_request)
    active = [{"slot_id": f"m{1 + (i % 2)}_mcq_{i}", "task_id": f"task-{i}",
               "task_family_id": f"family-{i}", "concept": f"concept-{i}"}
              for i in range(1, 47)]
    candidate = {"task_id": "task-new", "task_family_id": "family-new",
                 "concept": "new concept", "allowed_question_types": ["mcq"]}
    chosen, _ = duplicate_pipeline._select_replacement_candidate(
        slot_id="m2_mcq_7", qtype="mcq", conflict_slots=[], conflict_edges=[],
        candidates=[candidate], active_items=active, blacklist=set(), previous_failures=[],
        failed_slot_context={"slot_id": "m2_mcq_7", "question_type": "mcq",
                             "existing_question": {"question": "failed question"}},
        existing_questions=[{"slot_id": item["slot_id"], "task_id": item["task_id"],
                             "question": f"question {index}"}
                            for index, item in enumerate(active)],
        allow_semantic_overlap_candidates=True,
    )
    assert chosen["task_id"] == "task-new"
    assert len(captured["context"]["active_slots"]) <= 12
    assert len(captured["context"]["existing_questions"]) <= 12
    assert len(captured["context"]["active_slots"]) < len(active)
    old_payload = {
        "active_slots": active,
        "existing_questions": [
            {"slot_id": item["slot_id"], "task_id": item["task_id"],
             "question": f"question {index}"}
            for index, item in enumerate(active)
        ],
    }
    assert len(json.dumps(captured["context"], ensure_ascii=False)) < (
        len(json.dumps(old_payload, ensure_ascii=False)) / 2
    )


def test_easy_replacement_prefers_easy_but_medium_is_eligible(monkeypatch):
    import app.llm.client as llm_client
    import app.llm.factory as llm_factory

    captured = {}
    monkeypatch.setattr(llm_factory, "create_duplicate_replacement_client", lambda: object())

    def fake_request(_client, prompt, **_kwargs):
        captured["context"] = json.loads(prompt.split("\n", 1)[1])
        captured["max_tokens"] = _kwargs["max_tokens"]
        return {
            "slot_to_replace": "m2_mcq_1", "replacement_task_id": "easy",
            "replacement_fitb_entry_id": None, "checked_against_active_slots": True,
            "conflicts_with": [], "conflict_reason": "recovery",
            "why_this_candidate_resolves_the_conflict": "requested level",
            "rejected_candidates": [],
        }

    monkeypatch.setattr(llm_client, "request_structured", fake_request)
    base = {
        "allowed_question_types": ["mcq"], "task_operation": "construct",
        "target": "linear cost equation", "source_evidence_ids": ["c1"],
    }
    chosen, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="m2_mcq_1", qtype="mcq", conflict_slots=[], conflict_edges=[],
        candidates=[
            {**base, "task_id": "medium", "difficulty": "medium"},
            {**base, "task_id": "easy", "difficulty": "easy"},
        ], active_items=[], blacklist=set(), previous_failures=[],
        failed_slot_context={"required_difficulty": "easy"},
    )
    assert chosen["task_id"] == "easy"
    assert [row["task_id"] for row in captured["context"]["candidates"]] == [
        "easy", "medium",
    ]
    assert context["eligible_candidate_count"] == 2
    assert captured["max_tokens"] == 5000
    medium = next(row for row in captured["context"]["candidates"]
                  if row["task_id"] == "medium")
    assert medium == {**medium,
                      "requested_difficulty": "easy",
                      "candidate_difficulty": "medium",
                      "difficulty_tier": "medium_fallback",
                      "fallback_allowed": True}


def test_easy_replacement_allows_grounded_medium_without_prior_fallback(monkeypatch):
    import app.llm.client as llm_client
    import app.llm.factory as llm_factory

    monkeypatch.setattr(llm_factory, "create_duplicate_replacement_client", lambda: object())
    monkeypatch.setattr(llm_client, "request_structured", lambda *_args, **_kwargs: {
        "slot_to_replace": "m2_mcq_1", "replacement_task_id": "medium",
        "replacement_fitb_entry_id": None, "checked_against_active_slots": True,
        "conflicts_with": [], "conflict_reason": "no Easy candidate",
        "why_this_candidate_resolves_the_conflict": "grounded Medium fallback",
        "rejected_candidates": [],
    })
    chosen, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="m2_mcq_1", qtype="mcq", conflict_slots=[], conflict_edges=[],
        candidates=[{
            "task_id": "medium", "difficulty": "medium",
            "allowed_question_types": ["mcq"], "task_operation": "construct",
            "target": "linear cost equation", "source_evidence_ids": ["c1"],
        }], active_items=[], blacklist=set(), previous_failures=[],
        failed_slot_context={"required_difficulty": "easy"},
    )
    assert chosen["task_id"] == "medium"
    assert context["candidates"][0]["difficulty_candidate_status"] == (
        "source_constrained_medium_fallback_candidate"
    )


def test_easy_replacement_filters_hard_and_ungrounded_medium():
    base = {"allowed_question_types": ["mcq"], "task_operation": "construct",
            "target": "linear cost equation"}
    chosen, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="m2_mcq_1", qtype="mcq", conflict_slots=[], conflict_edges=[],
        candidates=[
            {**base, "task_id": "hard", "difficulty": "hard",
             "source_evidence_ids": ["c1"]},
            {**base, "task_id": "medium", "difficulty": "medium",
             "source_evidence_ids": []},
        ], active_items=[], blacklist=set(), previous_failures=[],
        failed_slot_context={"required_difficulty": "easy"},
    )
    assert chosen is None
    assert context["eligible_candidate_count"] == 0


def test_selector_shortlist_includes_semantic_equivalent_with_different_ids():
    candidate = {
        "slot_id": "candidate", "task_id": "task-new", "task_family_id": "family-new",
        "concept": "simultaneous linear equations", "concept_id": "linear-systems",
        "canonical_skill": "solve_linear_system", "task_operation": "find_common_solution",
        "target": "common solution of two equations", "student_work_product": "ordered pair",
    }
    equivalent = {
        "slot_id": "m1_mcq_1", "task_id": "different-id", "task_family_id": "other-family",
        "concept": "systems of linear equations", "concept_id": "linear-systems",
        "canonical_skill": "solve_linear_system", "task_operation": "solve_system",
        "target": "solution shared by both equations", "student_work_product": "ordered pair",
    }
    broad_but_different = {
        "slot_id": "m2_mcq_1", "task_id": "third-id", "task_family_id": "third-family",
        "concept": "linear functions", "concept_id": "linear-functions",
        "canonical_skill": "interpret_slope", "task_operation": "interpret_graph",
        "target": "rate of change", "student_work_product": "verbal interpretation",
    }
    shortlist = duplicate_pipeline._selector_relevant_active_items(
        [candidate], [broad_but_different, equivalent], set(), limit=1,
    )
    assert shortlist == [equivalent]
    # Shortlisting supplies context only; it does not classify or reject either task.
    assert duplicate_pipeline._selector_semantic_relation_score(candidate, equivalent) > (
        duplicate_pipeline._selector_semantic_relation_score(candidate, broad_but_different)
    )


def test_math_notation_normalizes_unambiguous_infinity_and_rejects_controls():
    normalized = _normalize_mcq({
        "question": "Select the interval.",
        "options": {"A": "(-infty, 2)", "B": "[2, 3]", "C": "[3, 4]", "D": "[4, 5]"},
        "correct_answer": "A",
    })
    assert normalized["options"]["A"] == r"(-\infty, 2)"
    assert _normalize_equation({
        "equation": "x\x08infty", "solution_steps": ["invalid"], "final_answer": "invalid",
    }) is None
    assert r"\infty" in to_mathtext("(-infty, 2)")
    assert "∞" in latex_to_plain_text(r"(-\infty, 2)")
