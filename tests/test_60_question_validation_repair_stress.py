"""Mocked 60-question stress coverage using DraftWork's production recovery nodes.

External LLM boundaries are supplied by the existing fail-closed 40-question
harness.  Validation, manifest construction, repair routing, candidate staging,
revalidation, replacement selection, and assembly remain the production code.
"""
from __future__ import annotations

from copy import deepcopy

from app.online import exam_builder
from app.online.exam_builder import assemble_exams_node
from app.online.validator import _integrate_duplicate_validation, repair_invalid_questions, validate_generated_questions
from tests import test_draftwork_40q_readonly_e2e as base


EXTRA_TYPES = {
    **{f"Q{i}": "mcq" for i in range(41, 46)},
    **{f"Q{i}": "true_false" for i in range(46, 49)},
    **{f"Q{i}": "fill_in_the_blank" for i in range(49, 51)},
    **{f"Q{i}": "definition" for i in range(51, 53)},
    **{f"Q{i}": "why" for i in range(53, 55)},
    **{f"Q{i}": "equation" for i in range(55, 57)},
    **{f"Q{i}": "word_problem" for i in range(57, 59)},
    "Q59": "essay", "Q60": "essay",
}


def _add_question(section, qtype: str, qid: str, slot_id: str) -> dict:
    base_item = {"question_id": qid, "slot_id": slot_id,
                 "selected_variation_id": f"var-{qid.lower()}"}
    if qtype == "mcq":
        return base_item | {"question": f"Which statement about {qid} is correct?",
                            "options": {"A": "Correct principle", "B": "Wrong unit", "C": "Wrong law", "D": "No relation"},
                            "correct_answer": "A"}
    if qtype == "true_false":
        return base_item | {"statement": f"The physical claim labelled {qid} is testable.", "answer": False if qid == "Q48" else True}
    if qtype == "fill_in_the_blank":
        answer = "velocity" if qid == "Q49" else "energy"
        return base_item | {"question": "The rate of change of displacement is ___." if qid == "Q49" else "The capacity to do work is ___.",
                            "answers": [answer], "distractors": ["mass", "time"]}
    if qtype == "definition":
        return base_item | {"term": f"term-{qid}", "reference_answer": f"A source-grounded definition for {qid}."}
    if qtype == "why":
        return base_item | {"question": f"Why does the relationship in {qid} occur?", "reference_answer": "Because the stated physical mechanism applies."}
    if qtype == "equation":
        return base_item | {"equation": "F = ma", "solution_steps": ["Substitute the givens.", "Compute 8 N."], "final_answer": "8 N"}
    if qtype == "word_problem":
        return base_item | {"question": f"A 4 kg object in {qid} accelerates at 2 m/s². Find its force.", "solution_steps": ["F = ma", "F = 4 × 2 = 8 N"], "final_answer": "8 N"}
    return base_item | {"question": f"Explain the evidence and reasoning for {qid}.", "reference_answer": "A source-grounded explanation.", "key_points": ["evidence", "reasoning"]}


def build_60_fixture():
    exam, plans, sources, plan_list = base.build_fixture()
    exam, plans, sources, plan_list = deepcopy(exam), deepcopy(plans), deepcopy(sources), deepcopy(plan_list)
    plan_sections = plan_list[0]["items"]
    for qid, qtype in EXTRA_TYPES.items():
        section = exam["questions"][qtype]
        items = section["items"] if isinstance(section, dict) else section
        slot_id = f"m1_{qtype}_{len(items) + 1}"
        item = _add_question(section, qtype, qid, slot_id)
        items.append(item)
        chunk_id = f"chunk-{qid}"
        sources[chunk_id] = f"Evidence for {qid}: the assigned physics relationship supports this assessment."
        variation = {"variation_id": f"var-{qid.lower()}", "question_types": [qtype],
                     "evidence_chunk_ids": [chunk_id], "operation": f"operation-{qid}",
                     "target": f"target-{qid}", "givens": [], "reasoning_mode": "conceptual_reasoning",
                     "student_work_product": f"product-{qid}"}
        plan = {"slot_id": slot_id, "question_type": qtype, "task_id": f"task-{qid.lower()}",
                "source_chunk_id": chunk_id, "support_chunk_ids": [], "evidence_chunk_ids": [chunk_id],
                "selected_variation_id": variation["variation_id"], "selected_variation_group_id": f"vg-{qid.lower()}",
                "permitted_variations": [variation], "concept": f"concept-{qid}", "idea_to_test": f"idea-{qid}",
                "target": variation["target"], "task_operation": variation["operation"],
                "student_work_product": variation["student_work_product"], "required_givens": [], "effective_givens": [],
                "difficulty": "medium", "requested_difficulty": "medium", "actual_difficulty": "medium",
                "reasoning_mode": "conceptual_reasoning", "reasoning_depth": "medium", "reasoning_requirements": [],
                "requires_quantitative_validation": qtype in {"equation", "word_problem"}}
        plans[slot_id] = plan
        plan_sections[qtype].append(plan)
    # The production FITB bank is global to its section.
    fitb = exam["questions"]["fill_in_the_blank"]
    fitb["word_bank"] = list(dict.fromkeys(value for item in fitb["items"] for value in [*item["answers"], *item["distractors"]]))
    return exam, plans, sources, plan_list


def test_60_question_mixed_defect_recovery_and_assembly(monkeypatch):
    harness = base.Harness(monkeypatch)
    harness.exam, harness.plans, harness.sources, harness.plan_list = build_60_fixture()
    counts = {
        qtype: len(section["items"] if isinstance(section, dict) else section)
        for qtype, section in harness.exam["questions"].items()
    }
    assert counts == {"mcq": 15, "true_false": 8, "fill_in_the_blank": 7,
                      "definition": 6, "why": 6, "equation": 6,
                      "word_problem": 6, "essay": 6}
    state = base._run_until_settled(harness.initial_state("stress-60-mixed"), harness, max_rounds=10)
    assembled = assemble_exams_node(state)
    final = base._final_questions(assembled)
    assert len(final) == 60, {
        "final": [item["question_id"] for item in final],
        "manifest": [(item.get("question_id"), item.get("status"), item.get("failure_code"))
                     for item in state.get("repair_manifest") or []],
        "invalid": [(item.get("question_id"), item.get("failure_code"))
                    for item in assembled.get("invalid_questions") or []],
    }
    final_report = next(report for report in state["validation_reports"] if report["model_number"] == 1)
    assert all(row["action"] == "PASS" for row in final_report["verdicts"])
    assert {call["validator"] for call in harness.validator_calls} >= {"grounding", "task_alignment", "answer_correctness", "difficulty", "quantitative"}
    assert {call["repair_agent"] for call in harness.repair_calls} >= {"answer", "question", "options", "solution", "task_alignment"}
    assert harness.difficulty_adjustment_prompts, (
        "difficulty mismatches must use targeted Difficulty Adjustment, not field Repair"
    )
    assert next(item for item in final if item["question_id"] == "Q48")["answer"] is False


def test_60_question_duplicate_v2_uses_pair_metadata_and_selects_one_target(monkeypatch):
    harness = base.Harness(monkeypatch)
    harness.exam, harness.plans, harness.sources, harness.plan_list = build_60_fixture()
    # Q1 and Q41 deliberately share a task/variation.  The mocked duplicate
    # validator supplies facts; Python retains Q1 and targets only Q41.
    q41 = next(item for item in harness.exam["questions"]["mcq"] if item["question_id"] == "Q41")
    p41 = harness.plans[q41["slot_id"]]
    p41["task_id"] = harness.plans[base._slot("Q1")]["task_id"]
    p41["selected_variation_id"] = harness.plans[base._slot("Q1")]["selected_variation_id"]
    p41["selected_variation_group_id"] = harness.plans[base._slot("Q1")]["selected_variation_group_id"]

    def duplicate_yes(_client, _prompt, **kwargs):
        rows = []
        for pair_id in kwargs["expected_ids"]:
            duplicate = pair_id == "Q1::Q41"
            rows.append({"pair_id": pair_id, "same_core_knowledge": "yes" if duplicate else "no",
                         "same_student_operation_or_method": "yes" if duplicate else "no",
                         "same_target_or_result": "yes" if duplicate else "no",
                         "genuinely_new_reasoning": "no" if duplicate else "yes",
                         "one_answer_substantially_gives_away_other": "no"})
        return {"verdicts": rows}
    monkeypatch.setattr(base.duplicate_judge, "request_structured", duplicate_yes)
    reports = {1: {"model_number": 1, "all_pass": True, "verdicts": [
        {"question_id": item["question_id"], "action": "PASS", "role_statuses": {}}
        for section in harness.exam["questions"].values()
        for item in (section["items"] if isinstance(section, dict) else section)
    ]}}
    manifest, validated, _warnings, result = _integrate_duplicate_validation(
        {"assessment_profile": {"domain": "STEM"}, "assessment_catalog": base.CATALOG,
         "retrieved_chunks": [{"child_id": key, "content": value} for key, value in harness.sources.items()]},
        [harness.exam], harness.plan_list, reports, [], [1], [],
    )
    duplicate_targets = [target for target in manifest if target["validation_role"] == "duplicate"]
    assert validated == []
    assert result["duplicate_reports"][-1]["similarity_distribution"]["embedding_calls"] == 0
    assert {target["question_id"] for target in duplicate_targets} in ({"Q1"}, {"Q41"})
    assert duplicate_targets[0]["repair_type"] == "replace_assignment"


def test_duplicate_conflict_runs_through_replacement_staging_revalidation_and_commit(monkeypatch):
    """Exercise the real transactional duplicate-repair path, not a direct commit."""
    harness = base.Harness(monkeypatch)
    state = base._run_until_settled(harness.initial_state("duplicate-transaction"), harness)
    # Make Q38/Q39 different catalog tasks with equivalent student work.
    q38 = next(item for item in state["generated_exams"][0]["questions"]["word_problem"] if item["question_id"] == "Q38")
    q39 = next(item for item in state["generated_exams"][0]["questions"]["word_problem"] if item["question_id"] == "Q39")
    before_question, before_plans = deepcopy(q39), deepcopy(state["plans"])
    q39.update(question=q38["question"], solution_steps=deepcopy(q38["solution_steps"]), final_answer=q38["final_answer"])
    p38, p39 = harness.plans[q38["slot_id"]], harness.plans[q39["slot_id"]]
    p39.update(task_id="task_linear_system_B", task_operation=p38["task_operation"], target=p38["target"],
               student_work_product=p38["student_work_product"], concept=p38["concept"])
    for plan in state["plans"]:
        for item in plan["items"]["word_problem"]:
            if item["slot_id"] == q39["slot_id"]:
                item.update(p39)

    calls: list[list[str]] = []
    def duplicate_provider(_client, _prompt, **kwargs):
        ids = list(kwargs["expected_ids"]); calls.append(ids)
        rows = []
        for pair_id in ids:
            duplicate = pair_id == "Q38::Q39"
            rows.append({"pair_id": pair_id, "same_core_knowledge": "yes" if duplicate else "no",
                         "same_student_operation_or_method": "yes" if duplicate else "no",
                         "same_target_or_result": "yes" if duplicate else "no",
                         "genuinely_new_reasoning": "no" if duplicate else "yes",
                         "one_answer_substantially_gives_away_other": "no"})
        return {"verdicts": rows}
    monkeypatch.setattr(base.duplicate_judge, "request_structured", duplicate_provider)
    # The production regeneration function still owns prompt construction,
    # evidence binding, duplicate preflight, and candidate staging.  Only its
    # external provider result is made distinct from the existing Q26 fixture.
    def replacement_generator(_client, _prompt, **kwargs):
        if str(kwargs.get("schema_name")) == "word_problem_questions":
            return {"questions": [{
                "slot_id": slot_id,
                "question": "A bicycle starts at 3 m/s and accelerates at 1 m/s² for 4 s. Find its final speed.",
                "solution_steps": ["v = u + at", "v = 3 + 1 × 4 = 7 m/s"],
                "final_answer": "7 m/s",
            } for slot_id in kwargs.get("expected_ids") or []]}
        return harness.generator_request(_client, _prompt, **kwargs)
    monkeypatch.setattr(exam_builder, "request_structured", replacement_generator)
    reports = {1: next(report for report in state["validation_reports"] if report["model_number"] == 1)}
    manifest, validated, warnings, result = _integrate_duplicate_validation(
        state, state["generated_exams"], state["plans"], reports,
        state["repair_manifest"], state["validated_models"], state["warnings"],
    )
    target = next(item for item in manifest if item.get("validation_role") == "duplicate")
    assert target["question_id"] in {"Q38", "Q39"}
    before_target = deepcopy(next(
        item for item in state["generated_exams"][0]["questions"]["word_problem"]
        if item["question_id"] == target["question_id"]
    ))
    state.update(repair_manifest=manifest, validated_models=validated, warnings=warnings,
                 validation_reports=list(reports.values()), duplicate_reports=result["duplicate_reports"],
                 duplicate_fact_cache=result.get("duplicate_fact_cache") or {})
    repaired = repair_invalid_questions(state)
    candidate = repaired["repair_candidates"][0]
    assert candidate["repair_action"] == "replace_task"
    assert next(item for item in state["generated_exams"][0]["questions"]["word_problem"]
                if item["question_id"] == target["question_id"]) == before_target
    assert candidate["plans"] != before_plans
    completed = validate_generated_questions({**state, **repaired})
    accepted = next(item for item in completed["generated_exams"][0]["questions"]["word_problem"]
                    if item["question_id"] == target["question_id"])
    assert accepted != before_target
    assert not completed["repair_candidates"]
    assert calls, "Duplicate Validator must receive production candidate pairs."
