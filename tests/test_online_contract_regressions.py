from __future__ import annotations

import json
from typing import get_args

import pytest
from fastapi import HTTPException

from app.api.routes import GenerateRequest, generate
from app.llm.schemas import QuestionType
from app.online import assessment_catalog, duplicate_judge, prompts, validator
from app.online.exam_builder import balance_mcq_answer_positions
from app.online.lifecycle import developer_diagnostic, initialize_slot_lifecycle, reconcile_slot_lifecycle
from app.online.models import _normalize_definition


def _task(task_id: str, *, operation: str = "explain_cause", qtypes=None) -> dict:
    return {
        "task_id": task_id,
        "task_family_id": f"family-{task_id}",
        "coverage_id": f"coverage-{task_id}",
        "concept_id": f"concept-{task_id}",
        "relationship_id": None,
        "knowledge_anchor": f"knowledge {task_id}",
        "topic": "motion",
        "concept": "motion",
        "relationship": "force changes motion",
        "task_operation": operation,
        "student_work_product": "explain the mechanism",
        "target": "cause of acceleration",
        "reasoning_mode": "conceptual_reasoning",
        "allowed_question_types": qtypes or ["why"],
        "evidence_chunk_ids": ["c1"],
        "required_givens": [],
        "scenario_family": None,
        "text_answerable": True,
        "supported_difficulties": ["easy", "medium", "hard"],
    }


def test_validator_and_repair_prompts_match_their_root_wrapper_schemas():
    validator_shape = json.loads(validator.VALIDATOR_OUTPUT_SCHEMA)
    repair_shape = json.loads(validator.REPAIR_OUTPUT_SCHEMA)
    assert set(validator_shape) == {"verdicts"}
    assert isinstance(validator_shape["verdicts"], list)
    assert set(repair_shape) == {"repairs"}
    assert isinstance(repair_shape["repairs"], list)
    assert "never return a bare root array" in validator.build_validator_prompt([])
    assert "never return a bare root array" in validator.build_repair_prompt([])


def test_all_standalone_generation_examples_include_required_slot_id():
    for schema in (
        prompts.MCQ_SCHEMA,
        prompts.TRUE_FALSE_SCHEMA,
        prompts.WHY_SCHEMA,
        prompts.DEFINITION_SCHEMA,
        prompts.EQUATION_SCHEMA,
        prompts.WORD_PROBLEM_SCHEMA,
        prompts.ESSAY_SCHEMA,
    ):
        first = json.loads(schema)["questions"][0]
        assert first["slot_id"]
    assert json.loads(prompts.FILL_IN_THE_BLANK_ITEMS_SCHEMA)["items"][0]["slot_id"]


def test_dynamic_structured_prompts_name_their_canonical_root_wrappers():
    assert "`verdicts`" in duplicate_judge._prompt([])
    catalog_prompt = assessment_catalog._catalog_prompt(
        "source", "en", ["c1"], domain_type="stem", subject="chemistry",
    )
    assert "assessment_profile" in catalog_prompt
    assert "tasks: the array" in catalog_prompt


def test_repair_authorization_uses_exact_canonical_fields():
    assert validator._allowed_canonical("mcq", ["answer"], "FIX_ANSWER") == set()
    assert validator._allowed_canonical("definition", ["answer"], "FIX_ANSWER") == set()
    assert validator._allowed_canonical("equation", ["answer"], "FIX_ANSWER") == set()
    assert validator._allowed_canonical("mcq", ["correct_answer"], "FIX_ANSWER") == {
        "correct_answer"
    }
    assert "answer" not in validator._LOGICAL_TO_CANONICAL["definition"]


def test_fitb_repair_never_authorizes_bank_or_answer_fields():
    assert validator._allowed_canonical(
        "fill_in_the_blank", ["word_bank", "answers"], "FIX_OPTIONS"
    ) == set()
    # MCQ FIX_OPTIONS may update the label mapping when options move, but no
    # unrelated canonical fields are authorized.
    assert validator._allowed_canonical(
        "mcq", ["options", "correct_answer"], "FIX_OPTIONS"
    ) == {"options", "correct_answer"}


def test_generator_plan_text_preserves_catalog_and_replacement_identity():
    item = {
        **_task("task-1"),
        "slot_id": "m1_fill_in_the_blank_1",
        "difficulty": "medium",
        "source_chunk_id": "c1",
        "support_chunk_ids": ["c2"],
        "catalog_replacement_stage": "strict_unused_task",
        "fitb_answer_reused": True,
        "required_instance_distinctions": ["new target"],
    }
    rendered = prompts._format_planned_item(1, item)
    for expected in (
        "task_id=task-1",
        "task_family_id=family-task-1",
        "coverage_id=coverage-task-1",
        "concept_id=concept-task-1",
        "catalog_replacement_stage=strict_unused_task",
        "fitb_answer_reused=True",
        "required_instance_distinctions=['new target']",
    ):
        assert expected in rendered


def test_fixed_difficulty_generator_honors_fallback_slots_actual_difficulty():
    item = {
        **_task("task-medium"),
        "slot_id": "m1_why_1",
        "source_chunk_id": "c1",
        "support_chunk_ids": [],
        "requested_difficulty": "easy",
        "difficulty": "medium",
        "actual_difficulty": "medium",
        "reasoning_depth": "medium",
        "reasoning_requirements": ["connect cause and effect"],
        "prerequisites": ["identify the relevant relationship"],
        "difficulty_adaptation": {
            "status": "failed",
            "from_difficulty": "medium",
            "to_difficulty": "easy",
        },
        "difficulty_fallback_reason": "source_does_not_support_safe_simplification",
        "assigned_sources": {"primary": {"chunk_id": "c1", "text": "source"}},
    }

    _system, user = prompts.build_prompt(
        "why", 1, "unused global context", difficulty="easy", planned_items=[item]
    )

    assert "requested_difficulty=easy" in user
    assert "difficulty=medium" in user
    assert "actual_difficulty=medium" in user
    assert "reasoning_depth=medium" in user
    assert "reasoning_requirements=['connect cause and effect']" in user
    assert "including fixed-difficulty exams" in user
    assert "do not silently relabel" in user
    assert prompts._DIFFICULTY_EASY not in user


def test_definition_term_is_bare_concept_across_normalization_and_validation():
    item = {"term": "linear equation", "reference_answer": "An equation of degree one."}
    assert _normalize_definition(item) == item
    assert validator._local_verdict("definition", item, None) is None


def test_mcq_balance_recomputes_certificate_mapping_before_validation():
    question = {
        "question": "Which value is correct?",
        "options": {"A": "4", "B": "3", "C": "2", "D": "1"},
        "correct_answer": "A",
        "solution_certificate": {
            "method": "calculate", "important_givens": ["2+2"],
            "calculation_steps": ["2+2=4"], "final_answer": "4", "units": "",
            "matching_correct_option": "A", "verification_method": "substitution",
        },
    }
    questions = {"mcq": [question, *[
        {
            "question": f"Which value is correct for item {index}?",
            "options": {"A": "4", "B": "3", "C": "2", "D": "1"},
            "correct_answer": "A",
        }
        for index in range(2, 5)
    ]]}
    balance_mcq_answer_positions(questions)
    assert question["correct_answer"] == "B"
    assert question["options"]["B"] == "4"
    assert question["solution_certificate"]["matching_correct_option"] == "B"


def test_general_repair_invalidates_old_certificate_and_uses_canonical_writeback(monkeypatch):
    original = {
        "question_id": "q1", "slot_id": "m1_why_1",
        "question": "Why does friction slow motion?", "reference_answer": "old",
        "solution_certificate": {
            "method": "old", "important_givens": ["old"],
            "calculation_steps": ["old"], "final_answer": "old", "units": "",
            "matching_correct_option": None, "verification_method": "old",
        },
    }
    exam = {"model_number": 1, "questions": {"why": [original]}}
    monkeypatch.setattr(validator, "request_structured", lambda *_args, **_kwargs: {
        "repairs": [{
            "question_id": "q1", "repaired_fields": ["reference_answer"],
            "question": {
                "question": original["question"],
                "reference_answer": "Friction opposes relative motion.",
                "solution_certificate": original["solution_certificate"],
            },
        }]
    })
    result = validator.repair_exam(exam, [{
        "question_id": "q1", "action": "FIX_ANSWER",
        "fields_to_fix": ["reference_answer"], "reason": "incorrect",
        "expected_fix": "replace the answer", "failure_code": "incorrect_answer",
    }], client=object())
    assert result["repaired_ids"] == ["q1"]
    assert exam["questions"]["why"][0]["reference_answer"].startswith("Friction")
    assert exam["questions"]["why"][0]["solution_certificate"] is None


def test_question_type_contract_has_no_removed_type():
    assert set(get_args(QuestionType)) == {
        "mcq", "fill_in_the_blank", "true_false", "definition", "why",
        "equation", "word_problem", "essay",
    }


def test_why_catalog_compatibility_preserves_compiler_decision():
    raw = {
        "assessment_profile": {"question_types": {"why": {}}},
        "tasks": [
            {**_task("mechanism", operation="explain_mechanism"), "allowed_question_types": ["why"]},
            {**_task("relationship", operation="explain_relationship"), "allowed_question_types": ["why"]},
        ],
    }
    catalog = assessment_catalog._finalize_catalog(
        raw, key="k", processed_document_key="p", selected_chunk_ids=["c1"], usage=None
    )
    assert all("why" in task["allowed_question_types"] for task in catalog["tasks"])


def test_inventory_prefers_unused_then_enters_explicit_reuse_mode():
    catalog = {"tasks": [_task("t1"), _task("t2")]}
    plans, issues = assessment_catalog.allocate_catalog_plan(
        catalog, [("why", 3)], 1, "medium"
    )
    assert issues == []
    items = plans[0]["items"]["why"]
    assert [item["task_reused"] for item in items] == [False, False, True]
    inventory = items[-1]["task_inventory"]
    assert inventory["compatible_total_count"] == 2
    assert inventory["compatible_used_count"] == 2
    assert inventory["compatible_unused_count"] == 0
    assert inventory["reuse_mode"] is True
    assert items[-1]["reuse_reason"] == "no_unused_compatible_task"


def test_stale_catalog_cache_version_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(assessment_catalog.config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    key = "a" * 64
    path = assessment_catalog._path_for(key)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "cache_key": key, "catalog_version": "catalog-v1", "tasks": [_task("old")],
    }), encoding="utf-8")
    assert assessment_catalog._read_cache(key) is None


def test_fitb_stem_repair_preserves_authoritative_answer_and_entry_id(monkeypatch):
    section = {
        "word_bank": ["force", "mass", "energy"],
        "items": [{
            "question_id": "q1", "slot_id": "m1_fill_in_the_blank_1",
            "fitb_entry_id": "fitb_force",
            "question": "A vague ________.", "answers": ["force"],
            "distractors": ["mass", "energy"],
        }],
    }
    exam = {"model_number": 1, "questions": {"fill_in_the_blank": section}}
    monkeypatch.setattr(validator, "request_structured", lambda *_args, **_kwargs: {
        "repairs": [{
            "question_id": "q1", "repaired_fields": ["question"],
            "question": {
                "stem": "The push or pull is ________.",
                "distractor_1": "mass", "distractor_2": "energy",
            },
        }]
    })
    before = list(section["word_bank"])
    result = validator.repair_exam(exam, [{
        "question_id": "q1", "action": "FIX_QUESTION", "fields_to_fix": ["question"],
        "reason": "vague", "expected_fix": "clarify the clue", "failure_code": "other",
    }], client=object())
    assert result["repaired_ids"] == ["q1"]
    assert section["word_bank"] == before
    assert section["items"][0]["answers"] == ["force"]
    assert section["items"][0]["fitb_entry_id"] == "fitb_force"


def test_fitb_distractor_repair_rebuilds_section_bank_in_code(monkeypatch):
    section = {
        "word_bank": ["force", "mass", "energy"],
        "items": [{
            "question_id": "q1", "slot_id": "m1_fill_in_the_blank_1",
            "question": "The push or pull is ________.", "answers": ["force"],
            "distractors": ["mass", "energy"],
        }],
    }
    exam = {"model_number": 1, "questions": {"fill_in_the_blank": section}}
    monkeypatch.setattr(validator, "request_structured", lambda *_args, **_kwargs: {
        "repairs": [{
            "question_id": "q1", "repaired_fields": ["distractors"],
            "question": {
                "stem": "The push or pull is ________.",
                "distractor_1": "motion",
                "distractor_2": "matter",
            },
        }]
    })
    result = validator.repair_exam(exam, [{
        "question_id": "q1", "action": "FIX_OPTIONS",
        "fields_to_fix": ["distractors"], "reason": "weak distractors",
        "expected_fix": "replace bank distractors", "failure_code": "weak_options",
    }], client=object())
    assert result["failed_ids"] == []
    assert "q1" in result["repaired_ids"]
    assert section["word_bank"] == ["force", "motion", "matter"]


def test_unauthorized_fitb_word_bank_change_is_rejected(monkeypatch):
    section = {
        "word_bank": ["force", "mass", "energy"],
        "items": [{
            "question_id": "q1", "slot_id": "m1_fill_in_the_blank_1",
            "question": "The push or pull is ________.", "answers": ["wrong"],
        }],
    }
    exam = {"model_number": 1, "questions": {"fill_in_the_blank": section}}
    monkeypatch.setattr(validator, "request_structured", lambda *_args, **_kwargs: {
        "repairs": [{
            "question_id": "q1", "repaired_fields": ["answers", "word_bank"],
            "question": {
                "question": "The push or pull is ________.", "answers": ["force"],
                "word_bank": ["force", "motion", "matter"],
            },
        }]
    })
    result = validator.repair_exam(exam, [{
        "question_id": "q1", "action": "FIX_ANSWER", "fields_to_fix": ["answers"],
        "reason": "wrong answer", "expected_fix": "fix answer",
        "failure_code": "incorrect_answer",
    }], client=object())
    assert result["failed_ids"] == ["q1"]
    assert section["word_bank"] == ["force", "mass", "energy"]


def test_slot_lifecycle_has_one_terminal_status_and_deduplicated_history():
    records = initialize_slot_lifecycle([("why", 1)], 1)
    event = {
        "slot_id": "m1_why_1", "question_id": "q1", "status": "repaired", "attempt": 1,
    }
    state = {
        "plans": [{"model_number": 1, "items": {"why": [{
            "slot_id": "m1_why_1", "task_id": "t1",
        }]}}],
        "generated_exams": [{"model_number": 1, "questions": {"why": [{
            "slot_id": "m1_why_1", "question_id": "q1",
        }]}}],
        "validation_reports": [{"verdicts": [{"question_id": "q1", "action": "PASS"}]}],
        "validation_repair_history": [event],
    }
    once = reconcile_slot_lifecycle(records, state, stage="repair")
    twice = reconcile_slot_lifecycle(once, state, stage="finalization")
    record = twice["m1_why_1"]
    assert record["terminal_status"] == "finalized"
    assert len(record["repair_attempts"]) == 1


@pytest.mark.parametrize(
    ("failure_stage", "terminal_status"),
    [("validation", "validation_failed"), ("repair", "repair_failed")],
)
def test_slot_lifecycle_preserves_originating_failure_stage(
    failure_stage, terminal_status
):
    records = initialize_slot_lifecycle([("mcq", 1)], 1)
    state = {
        "plans": [{"model_number": 1, "items": {"mcq": [{
            "slot_id": "m1_mcq_1", "task_id": "t1",
        }]}}],
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [{
            "slot_id": "m1_mcq_1", "question_id": "q1",
        }]}}],
        "invalid_questions": [{
            "slot_id": "m1_mcq_1", "question_id": "q1",
            "failure_code": "invalid_options", "failure_stage": failure_stage,
        }],
    }
    result = reconcile_slot_lifecycle(records, state, stage="finalization")
    assert result["m1_mcq_1"]["failure_stage"] == failure_stage
    assert result["m1_mcq_1"]["terminal_status"] == terminal_status


def test_developer_diagnostic_preserves_original_exception_and_traceback():
    try:
        raise ValueError("original detail")
    except ValueError as exc:
        diagnostic = developer_diagnostic(exc, stage="generation", slot_id="m1_mcq_1")
    assert diagnostic["exception_type"] == "ValueError"
    assert diagnostic["exception_message"] == "original detail"
    assert "ValueError: original detail" in diagnostic["traceback"]


def test_legacy_synchronous_generation_endpoint_is_retired():
    with pytest.raises(HTTPException) as caught:
        generate(GenerateRequest(document_id="doc", child_ids=["c1"], mcq_count=1))
    assert caught.value.status_code == 410
    assert "exam-jobs" in str(caught.value.detail)
