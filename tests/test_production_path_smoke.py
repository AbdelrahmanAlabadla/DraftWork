"""Fast production-path smoke coverage for deployment gating.

These tests exercise the real production orchestration (graph facade, specialized
validator dispatch, independent recovery, duplicate scan, assembly, export) while
mocking only the external LLM boundary.  They are intentionally small; the
deployment gate also runs the focused existing regression tests for Repair
telemetry, single/two-model cross-model recovery, duplicate batching, and
validator request reconstruction.
"""
from __future__ import annotations

import pytest

from app.online import duplicate_judge, independent_recovery, multi_validator, validator
from app.online.exam_builder import assemble_exams_node
from app.online.validation import orchestrator, recovery
from app.online import graph as production_graph


def _client():
    return type("Client", (), {"provider_name": "smoke", "model": "smoke"})()


def _pass_validator_response(_client, prompt, **_kwargs):
    marker = "<VALIDATION_DATA>\n"
    payload = __import__("json").loads(
        prompt.split(marker, 1)[1].split("\n</VALIDATION_DATA>", 1)[0]
    )
    return {
        "validator": payload["validator"],
        "verdicts": [{
            "question_id": item["question_id"], "status": "PASS", "issues": [],
            "grounding_outcome": "SUPPORTED" if payload["validator"] == "grounding" else None,
            "observed_difficulty": "easy" if payload["validator"] == "difficulty" else None,
            "requested_difficulty": "easy" if payload["validator"] == "difficulty" else None,
            "declared_difficulty": "easy" if payload["validator"] == "difficulty" else None,
            "calculation_evidence": [], "original_issue_checks": [],
        } for item in payload["questions"]],
    }


def _no_duplicate(_client, _prompt, **kwargs):
    return {"verdicts": [{
        "pair_id": pair_id,
        "same_core_knowledge": "no",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "no",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "no",
    } for pair_id in kwargs.get("expected_ids") or []]}


def test_facade_resolves_to_production_orchestration_modules():
    assert validator.validate_generated_questions.__module__ == "app.online.validator"
    assert orchestrator.validate_generated_questions.__module__ == (
        "app.online.validation.orchestrator"
    )
    assert recovery.repair_invalid_questions.__module__ == (
        "app.online.validation.recovery"
    )
    nodes = set(production_graph.get_exam_graph().get_graph().nodes)
    assert {
        "validate_generated_questions", "repair_invalid_questions",
        "independent_model_recovery", "assemble_exams",
    } <= nodes


def test_two_model_production_recovery_assembly_and_export(monkeypatch):
    """46-question two-model path with real validators/recovery/assembly."""
    from tests.test_full_pipeline_mock import _question, _task
    from app.online import assessment_catalog, lifecycle

    monkeypatch.setattr(multi_validator, "request_structured", _pass_validator_response)
    monkeypatch.setattr(duplicate_judge, "request_structured", _no_duplicate)
    monkeypatch.setattr(
        duplicate_judge, "create_llm_client", lambda **_kwargs: _client(),
    )

    tasks = [_task(f"smoke_task_{index}") for index in range(1, 47)]
    catalog = {"tasks": tasks, "selected_chunk_ids": ["c1"]}
    plans, issues = assessment_catalog.allocate_catalog_plan(
        catalog,
        [("mcq", 10), ("true_false", 5), ("equation", 4), ("word_problem", 4)],
        2, "easy",
    )
    assert issues == []
    state = {
        "num_models": 2, "difficulty": "easy", "document_language": "en",
        "assessment_catalog": catalog,
        "assessment_profile": {"subject": "mathematics", "domain": "STEM"},
        "retrieved_chunks": [{"child_id": "c1", "content": "x+y=3; x-y=1"}],
        "plans": plans, "generated_exams": [], "validation_reports": [],
        "repair_manifest": [], "repair_candidates": [], "validated_models": [],
        "warnings": [], "validation_ledger": {},
        "slot_lifecycle": lifecycle.initialize_slot_lifecycle(
            [("mcq", 10), ("true_false", 5), ("equation", 4), ("word_problem", 4)], 2,
        ),
    }
    for plan in plans:
        exam = {"model_number": plan["model_number"], "questions": {}}
        for qtype, items in (plan.get("items") or {}).items():
            exam["questions"][qtype] = [
                _question(qtype, item["slot_id"], index)
                for index, item in enumerate(items, start=1)
            ]
        state["generated_exams"].append(exam)

    recovered = independent_recovery.run_independent_model_recovery(state)
    assert sorted(recovered["validated_models"]) == [1, 2]
    assert recovered["cross_model_duplicate_scan_after_local_completion"] is True
    assert "_recovery_task_coordinator" not in recovered

    assembled = assemble_exams_node(recovered)
    assert len(assembled["accepted_slot_ids"]) == 46

    from app.exports.docx_exporter import render_exam_docx
    from app.exports.pdf_exporter import render_exam_pdf
    model_one = next(
        exam for exam in assembled["generated_exams"]
        if int(exam.get("model_number") or 0) == 1
    )
    accepted = set(assembled["accepted_slot_ids"])
    public = {
        qtype: [value for value in section if value.get("slot_id") in accepted]
        for qtype, section in model_one["questions"].items()
    }
    pdf = render_exam_pdf({"model_number": 1, "questions": public},
                          {"exam_title": "Smoke"})
    docx = render_exam_docx({"model_number": 1, "questions": public},
                            {"exam_title": "Smoke"})
    assert pdf.startswith(b"%PDF") and docx.startswith(b"PK")
