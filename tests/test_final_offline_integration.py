from __future__ import annotations

import json
import threading
import time
import tracemalloc

from app.llm import scheduler
from app.online import assessment_catalog, duplicate_pipeline, validator


def _raw_task(index: int) -> dict:
    question_types = ["mcq", "true_false", "equation", "word_problem"]
    qtype = question_types[index % len(question_types)]
    concept = f"constant speed family {index // 4}"
    return {
        "knowledge_anchor": concept,
        "concept": concept,
        "relationship": "distance speed time",
        "task_operation": ("solve_for_distance" if index % 4 == 0 else
                           "solve_for_time" if index % 4 == 1 else
                           "solve_for_speed" if index % 4 == 2 else "verify_solution"),
        "student_work_product": "calculated quantity",
        "target": ("distance" if index % 4 == 0 else
                   "time" if index % 4 == 1 else
                   "speed" if index % 4 == 2 else "validity of a proposed solution"),
        "reasoning_mode": "quantitative_reasoning",
        "allowed_question_types": [qtype],
        "evidence_chunk_ids": [f"c{index % 12}"],
        "required_givens": ["constant-speed relationship"],
        "allowed_variations": [{
            "target": ("distance" if index % 4 == 0 else "time" if index % 4 == 1
                       else "speed" if index % 4 == 2 else "validity of a proposed solution"),
            "operation": ("solve_for_distance" if index % 4 == 0 else
                          "solve_for_time" if index % 4 == 1 else
                          "solve_for_speed" if index % 4 == 2 else "verify_solution"),
            "givens": ["distance", "speed", "time"], "representation": None,
            "application": "constant-speed scenario", "numeric_values_may_change": True,
            "question_types": [qtype],
        }],
        "variable_givens": ["distance", "speed", "time"],
        "complexity_boundaries": ["do not introduce acceleration"],
        "exact_source_instance": False,
        "scenario_family": f"motion-{index // 4}",
        "text_answerable": True,
        "native_difficulty": "easy",
        "prerequisites": ["constant speed"],
    }


def _finalize(count: int) -> dict:
    return assessment_catalog._finalize_catalog(
        {"assessment_profile": {}, "tasks": [_raw_task(index) for index in range(count)],
         "fitb_bank": []},
        key=f"catalog-{count}", processed_document_key="synthetic-doc",
        selected_chunk_ids=[f"c{index}" for index in range(12)], usage=None,
    )


def test_large_catalogs_finalize_allocate_and_stay_bounded(monkeypatch):
    metrics = []
    for count in (48, 60, 100, 200):
        tracemalloc.start()
        started = time.perf_counter()
        catalog = _finalize(count)
        plans, issues = assessment_catalog.allocate_catalog_plan(
            catalog,
            [("mcq", 10), ("true_false", 5), ("equation", 4), ("word_problem", 4)],
            1, "easy",
        )
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        items = [item for values in plans[0]["items"].values() for item in values]
        assert not issues
        assert len(catalog["tasks"]) == count
        assert len(items) == 23
        assert all(item["allowed_variations"][0]["numeric_values_may_change"] is True
                   for item in catalog["tasks"])
        assert all(item["complexity_boundaries"] == ["do not introduce acceleration"]
                   for item in catalog["tasks"])
        assert all(item["evidence_chunk_ids"] for item in catalog["tasks"])
        metrics.append((count, elapsed, peak, len(json.dumps(catalog))))
    assert metrics[-1][1] < 5
    assert metrics[-1][2] < 32 * 1024 * 1024
    print("catalog_metrics", metrics)


def _candidate(task_id: str, *, target: str, operation: str, concept: str = "motion") -> dict:
    return {
        "task_id": task_id, "concept": concept, "concept_id": concept,
        "target": target, "task_operation": operation,
        "reasoning_mode": "algebra" if operation == "rearrange" else "substitution",
        "student_work_product": target, "evidence_chunk_ids": ["c1"],
        "allowed_question_types": ["word_problem"],
    }


def test_large_deterministic_replacement_pool_prefers_suitable_unused_candidate():
    active = [{
        "slot_id": "keep", "task_id": "old", "concept": "motion", "concept_id": "motion",
        "target": "distance", "task_operation": "calculate", "reasoning_mode": "substitution",
        "student_work_product": "distance",
    }]
    candidates = [_candidate(f"same-{index}", target="distance", operation="calculate")
                  for index in range(100)]
    candidates.extend(_candidate(f"other-{index}", target="time", operation="rearrange")
                      for index in range(100))
    candidates.append({**_candidate("wrong-type", target="time", operation="rearrange"),
                       "allowed_question_types": ["mcq"]})
    started = time.perf_counter()
    selected, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="replace", qtype="word_problem", conflict_slots=["keep"],
        conflict_edges=[{"reason": "same target and operation"}], candidates=candidates,
        active_items=active, blacklist={"other-0"},
        previous_failures=[{"candidate_id": "other-1", "reason": "prior failure"}],
        allow_semantic_overlap_candidates=True,
    )
    elapsed = time.perf_counter() - started
    assert selected is not None
    assert selected["task_id"].startswith("other-")
    assert selected["task_id"] not in {"other-0", "other-1", "wrong-type"}
    assert context["selection_source"] == "deterministic_conflict_ranking"
    assert elapsed < 2
    assert "wrong-type" not in {item.get("task_id") for item in context["candidates"]}
    print("replacement_metrics", {"candidate_count": len(candidates), "seconds": elapsed})


def test_group_one_reuse_contract_covers_quantitative_and_biology_variations():
    speed_facts = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no",
        "same_target_or_result": "no",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "no",
    }
    wording_only = {
        "same_core_knowledge": "yes",
        "same_student_operation_or_method": "yes",
        "same_target_or_result": "yes",
        "genuinely_new_reasoning": "no",
        "one_answer_substantially_gives_away_other": "yes",
    }
    from app.online.duplicate_judge import deterministic_pair_classification
    assert deterministic_pair_classification(speed_facts, authorized_practice_variation=True) == (
        "acceptable_practice_variation"
    )
    assert deterministic_pair_classification(wording_only, authorized_practice_variation=True) == "duplicate"
    biology_plan = {
        "slot_id": "bio-1", "task_id": "task_food_chain", "concept": "food chain",
        "task_operation": "identify_producer", "target": "producer", "student_work_product": "term",
        "required_givens": ["organism roles in the selected chain"],
        "variable_givens": ["organism named in the question"],
        "allowed_variations": [{
            "target": "producer", "operation": "identify_producer", "givens": ["food chain"],
            "representation": None, "application": None, "numeric_values_may_change": False,
        }, {
            "target": "primary consumer", "operation": "explain_role", "givens": ["food chain"],
            "representation": None, "application": None, "numeric_values_may_change": False,
        }],
        "complexity_boundaries": ["do not introduce food webs or population modelling"],
        "source_chunk_id": "bio-c1", "support_chunk_ids": [],
    }
    contract = validator._catalog_task_contract(biology_plan)
    assert "food webs" in contract["complexity_boundaries"][0]
    assert contract["allowed_variations"][1]["operation"] == "explain_role"


def _question(index: int, text: str) -> dict:
    return {
        "slot_id": f"s{index}", "question_id": f"q{index}", "question": text,
        "solution_steps": ["Use d = vt."], "final_answer": str(index),
    }


def test_twelve_mixed_revalidations_are_bounded_isolated_and_parallel(monkeypatch):
    scheduler.reset_scheduler_for_tests()
    monkeypatch.setattr(scheduler.config, "LLM_JOB_CONCURRENCY_LIMIT", 10)
    lock = threading.Lock()
    active = 0
    peak = 0
    completion_order: list[tuple[str, str]] = []

    def fake_client(**_kwargs):
        return object()

    def fake_validate(exam, question_ids, _plans, _sources, _client=None, role="general"):
        nonlocal active, peak
        qid = list(question_ids)[0]
        # Reverse delays guarantee completion order differs from submission order.
        delay = 0.060 + (12 - int(qid[1:])) * 0.005
        with scheduler.llm_request_slot(agent=role, operation="stress_validation"):
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(delay)
            with lock:
                active -= 1
                completion_order.append((qid, role))
        if qid == "q5":
            return {"model_number": 1, "all_pass": False, "warnings": [], "coverage": [],
                    "verdicts": [{
                        "question_id": qid,
                        "action": "FIX_OPTIONS" if role == "general" else "FIX_ANSWER",
                        "failure_code": "options" if role == "general" else "incorrect_calculation",
                        "fields_to_fix": ["options"] if role == "general" else ["solution_steps"],
                        "reason": "synthetic failure", "expected_fix": "repair",
                        "validation_role": role,
                    }]}
        return {"model_number": 1, "all_pass": True, "warnings": [], "coverage": [],
                "verdicts": [{"question_id": qid, "action": "PASS",
                              "fields_to_fix": [], "validation_role": role}]}

    monkeypatch.setattr("app.llm.factory.create_review_client", fake_client)
    monkeypatch.setattr(validator, "_validate_with_context", fake_validate)
    active_exam = {"model_number": 1, "questions": {"word_problem": [
        _question(index, f"old {index}") for index in range(1, 13)
    ]}}
    plans = [{"model_number": 1, "items": {"word_problem": [{
        "slot_id": f"s{index}", "question_type": "word_problem", "task_id": "task_speed",
        "task_reused": True, "source_chunk_id": "c1", "support_chunk_ids": [],
    } for index in range(1, 13)]}}]
    candidates = [{
        "model_number": 1, "slot_id": f"s{index}", "question_id": f"q{index}",
        "question_type": "word_problem", "attempt": 1, "had_original": True,
        "exam": {"model_number": 1, "questions": {"word_problem": [
            _question(index, f"repaired {index}")
        ]}}, "plans": plans,
    } for index in range(1, 13)]
    state = {
        "generated_exams": [active_exam], "plans": plans, "repair_candidates": candidates,
        "repair_manifest": [{"model_number": 1, "slot_id": f"s{index}",
                             "question_id": f"q{index}", "status": "invalid"}
                            for index in range(1, 13)],
        "assessment_profile": {"domain": "STEM"}, "validated_models": [1],
        "validation_reports": [], "warnings": [], "question_repair_attempts": {"s12": 3},
    }
    started = time.perf_counter()
    result = validator.validate_generated_questions(state)
    elapsed = time.perf_counter() - started
    by_id = {item["question_id"]: item for item in result["generated_exams"][0]["questions"]["word_problem"]}
    assert by_id["q1"]["question"] == "repaired 1"
    assert by_id["q5"]["question"] == "old 5"
    failed = next(item for item in result["repair_manifest"] if item["slot_id"] == "s5")
    assert [item["validator"] for item in failed["verdict"]["required_fixes"]] == [
        "general", "stem_quantitative",
    ]
    assert peak == 10
    assert completion_order != sorted(completion_order)
    sequential_seconds = 2 * sum(0.060 + (12 - index) * 0.005 for index in range(1, 13))
    assert elapsed < sequential_seconds
    print("repair_metrics", {"parallel_seconds": elapsed, "sequential_seconds": sequential_seconds,
                              "speedup": sequential_seconds / elapsed, "peak": peak,
                              "completed": len(completion_order)})
    scheduler.reset_scheduler_for_tests()
