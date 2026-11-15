from __future__ import annotations

import io
from copy import deepcopy

from app.online import assessment_catalog, independent_recovery
from app.online.exam_builder import assemble_exams_node
from app.online.lifecycle import initialize_slot_lifecycle
from app.exports.docx_exporter import render_exam_docx
from app.exports.pdf_exporter import render_exam_pdf


def _task(task_id: str) -> dict:
    task = {
        "knowledge_anchor": "linear system", "concept": "linear system",
        "relationship": "two equations determine one ordered pair",
        "task_operation": "solve_equation", "student_work_product": "solution",
        "target": "solve the system", "reasoning_mode": "symbolic_manipulation",
        "allowed_question_types": ["mcq", "true_false", "equation", "word_problem", "why"],
        "evidence_chunk_ids": ["c1"], "required_givens": [], "scenario_family": None,
        "text_answerable": True, "native_difficulty": "easy", "variable_givens": [],
        "complexity_boundaries": [], "exact_source_instance": False,
        "prerequisites": [], "allowed_transformations": [],
        "allowed_variations": [{
            "target": "solve the system", "operation": "solve_equation",
            "givens": ["x+y=3", "x-y=1"], "question_types": ["mcq"],
            "evidence_chunk_ids": ["c1"], "reasoning_mode": "symbolic_manipulation",
            "student_work_product": "solution",
            "required_response_components": ["selection"],
        }],
    }
    task.update(assessment_catalog.canonical_task_ids(task) or {})
    task["task_id"] = task_id
    return task


def _question(qtype: str, slot_id: str, index: int) -> dict:
    qid = f"model{slot_id.split('_', 1)[0][1:]}_{qtype}_{index}"
    base = {"slot_id": slot_id, "question_id": qid, "selected_variation_id": None}
    if qtype == "mcq":
        return {**base, "question": f"Solve unique system {slot_id}.",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_answer": "A"}
    if qtype == "true_false":
        return {**base, "statement": f"The system {slot_id} has one solution.", "answer": True}
    if qtype in {"equation", "word_problem"}:
        return {**base, "equation" if qtype == "equation" else "question": f"Solve {slot_id}.",
                "solution_steps": ["x=1", "y=2"], "final_answer": "(1,2)"}
    return {**base, "question": f"Explain {slot_id}.",
            "reference_answer": "Because the equations intersect once."}


def _pass_report(exam: dict) -> dict:
    model = int(exam["model_number"])
    qids = [q.get("question_id") for section in (exam.get("questions") or {}).values()
            for q in (section.get("items") if isinstance(section, dict) else section) or []]
    return {
        "model_number": model, "all_pass": True,
        "verdicts": [{"question_id": qid, "action": "PASS", "validation_status": "PASS",
                      "question_valid": True, "answer_valid": True, "failure_code": "pass",
                      "fields_to_fix": [], "required_fixes": [], "role_statuses": {},
                      "reason": "", "expected_fix": ""} for qid in qids],
        "role_results": {}, "role_fingerprints": {}, "validation_call_logs": [],
    }


def test_full_two_model_pipeline_real_validation_assembly_and_export(monkeypatch):
    from app.online import duplicate_judge, multi_validator

    def no_duplicate(_client, _prompt, **kwargs):
        return {"verdicts": [{
            "pair_id": pair_id,
            "same_core_knowledge": "no",
            "same_student_operation_or_method": "no",
            "same_target_or_result": "no",
            "genuinely_new_reasoning": "yes",
            "one_answer_substantially_gives_away_other": "no",
        } for pair_id in kwargs.get("expected_ids") or []]}

    monkeypatch.setattr(duplicate_judge, "request_structured", no_duplicate)

    tasks = [_task(f"task_{i}") for i in range(1, 47)]
    catalog = {"tasks": tasks, "selected_chunk_ids": ["c1"]}
    profile = {"subject": "mathematics", "domain": "STEM"}
    plans, issues = assessment_catalog.allocate_catalog_plan(
        catalog, [("mcq", 10), ("true_false", 5), ("equation", 4), ("word_problem", 4)],
        2, "easy",
    )
    assert issues == []
    state = {
        "num_models": 2, "difficulty": "easy", "document_language": "en",
        "assessment_catalog": catalog, "assessment_profile": profile,
        "retrieved_chunks": [{"child_id": "c1", "content": "x+y=3; x-y=1"}],
        "plans": plans,
        "generated_exams": [],
        "validation_reports": [], "repair_manifest": [], "repair_candidates": [],
        "validated_models": [], "warnings": [], "validation_ledger": {},
        "slot_lifecycle": initialize_slot_lifecycle(
            [("mcq", 10), ("true_false", 5), ("equation", 4), ("word_problem", 4)], 2
        ),
    }
    for plan in plans:
        exam = {"model_number": plan["model_number"], "questions": {}}
        for qtype, items in (plan.get("items") or {}).items():
            exam["questions"][qtype] = [
                _question(qtype, item["slot_id"], idx)
                for idx, item in enumerate(items, start=1)
            ]
            for item in items:
                item["selected_variation_id"] = item.get("selected_variation_id")
        state["generated_exams"].append(exam)

    monkeypatch.setattr(multi_validator, "validate_exam_multi", lambda exam, **_: _pass_report(exam))
    recovered = independent_recovery.run_independent_model_recovery(deepcopy(state))
    assert recovered["validated_models"] == [1, 2]
    assert recovered["cross_model_duplicate_scan_after_local_completion"] is True

    assembled = assemble_exams_node(recovered)
    assert len(assembled["accepted_slot_ids"]) == 46
    exams = []
    for exam in assembled["generated_exams"]:
        public_questions = {}
        for qtype, section in (exam.get("questions") or {}).items():
            public_questions[qtype] = [
                question for question in section
                if question.get("slot_id") in set(assembled["accepted_slot_ids"])
            ]
        exams.append({"model_number": exam["model_number"], "questions": public_questions})
    assert len(exams) == 2
    assert sum(len(section) for exam in exams for section in exam["questions"].values()) == 46
    assert all(q.get("answer") is True for exam in exams
               for q in exam["questions"].get("true_false", []))

    pdf = render_exam_pdf(exams[0], {"exam_title": "Controlled Full Pipeline"})
    docx = render_exam_docx(exams[0], {"exam_title": "Controlled Full Pipeline"})
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000
    assert docx.startswith(b"PK") and len(docx) > 500
