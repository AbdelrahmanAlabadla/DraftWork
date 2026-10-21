from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.llm import scheduler
from app.online import assessment_catalog, duplicate_judge, plan_review, validator
from app.online import duplicate_pipeline


def test_shared_scheduler_never_exceeds_configured_job_ceiling(monkeypatch):
    monkeypatch.setattr(scheduler.config, "LLM_JOB_CONCURRENCY_LIMIT", 3)
    scheduler.reset_scheduler_for_tests()


def test_catalog_scheduler_has_scoped_fifty_request_capacity(monkeypatch):
    monkeypatch.setattr(scheduler.config, "LLM_JOB_CONCURRENCY_LIMIT", 3)
    monkeypatch.setattr(scheduler.config, "MAX_PARALLEL_CATALOG_CALLS", 50)
    scheduler.reset_scheduler_for_tests()
    assert scheduler.scheduler_snapshot()["limit"] == 3
    assert scheduler.scheduler_snapshot(agent="assessment_compiler")["limit"] == 50
    scheduler.reset_scheduler_for_tests()
    active = 0
    peak = 0
    lock = threading.Lock()

    def work():
        nonlocal active, peak
        with scheduler.llm_request_slot(agent="test", operation="bounded"):
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(lambda _index: work(), range(12)))

    assert peak == 3
    assert scheduler.scheduler_snapshot("__unscoped__")["peak"] == 3
    assert scheduler.scheduler_snapshot("__unscoped__")["completed"] == 12
    scheduler.reset_scheduler_for_tests()


def _parent_chunks(parent_count: int) -> list[dict]:
    return [
        {
            "child_id": f"p{parent}-c{child}", "parent_id": f"p{parent}",
            "parent_title": f"Section {parent}", "chunk_title": f"Part {child}",
            "content": "x" * (parent + child + 1),
        }
        for parent in range(parent_count)
        for child in range((parent % 4) + 1)
    ]


@pytest.mark.parametrize(
    ("parent_count", "request_count"),
    [(2, 1), (4, 2), (5, 3), (20, 10), (100, 50)],
)
def test_catalog_batches_contain_two_complete_parent_groups(parent_count, request_count):
    chunks = _parent_chunks(parent_count)
    batches = assessment_catalog._balanced_catalog_batches(chunks, 1)
    assert len(batches) == request_count
    assert [row["child_id"] for batch in batches for row in batch] == [
        row["child_id"] for row in chunks
    ]
    assert all(len(dict.fromkeys(row["parent_id"] for row in batch)) <= 2 for batch in batches)
    locations = {}
    for batch_index, batch in enumerate(batches):
        for row in batch:
            locations.setdefault(row["parent_id"], set()).add(batch_index)
    assert all(len(indices) == 1 for indices in locations.values())


def test_catalog_batches_respect_partial_selection_and_skip_empty_parents():
    all_chunks = _parent_chunks(4)
    selected_ids = {"p0-c0", "p1-c1", "p2-c0", "p2-c1", "p2-c2"}
    selected = [row for row in all_chunks if row["child_id"] in selected_ids]
    batches = assessment_catalog._balanced_catalog_batches(selected)
    assert [[row["child_id"] for row in batch] for batch in batches] == [
        ["p0-c0", "p1-c1"], ["p2-c0", "p2-c1", "p2-c2"]
    ]
    assert all(row["parent_id"] != "p3" for batch in batches for row in batch)


def test_parent_relationship_changes_compiler_cache_identity():
    chunks = [
        {"child_id": "c1", "parent_id": "p1"},
        {"child_id": "c2", "parent_id": "p1"},
    ]
    regrouped = [
        {"child_id": "c1", "parent_id": "p1"},
        {"child_id": "c2", "parent_id": "p2"},
    ]
    membership_one = assessment_catalog._catalog_source_membership(chunks)
    membership_two = assessment_catalog._catalog_source_membership(regrouped)
    assert membership_one != membership_two
    base = {"batching": assessment_catalog.CATALOG_BATCHING_VERSION}
    first = assessment_catalog.compiler_cache_key("doc", ["c1", "c2"], {
        **base, "membership": str(membership_one),
    })
    second = assessment_catalog.compiler_cache_key("doc", ["c1", "c2"], {
        **base, "membership": str(membership_two),
    })
    legacy = assessment_catalog.compiler_cache_key("doc", ["c1", "c2"], {})
    assert first != second != legacy


def test_dynamic_usage_index_tracks_slots_and_questions():
    plans = [{"model_number": 1, "items": {"mcq": [
        {"slot_id": "s1", "task_id": "t1"},
        {"slot_id": "s2", "task_id": "t1"},
        {"slot_id": "s3", "task_id": "t2"},
    ]}}]
    exams = [{"model_number": 1, "questions": {"mcq": [
        {"slot_id": "s1", "question_id": "q1"},
        {"slot_id": "s2", "question_id": "q2"},
    ]}}]
    usage = assessment_catalog.build_task_usage_index(plans, exams)
    assert usage["t1"] == {
        "count": 2, "slot_ids": ["s1", "s2"], "question_ids": ["q1", "q2"]
    }
    assert usage["t2"]["count"] == 1


def test_duplicate_judge_runs_thirteen_independent_batches_with_ten_workers(monkeypatch):
    pairs = [{
        "pair_id": f"pair_{index}",
        "left": {"question_id": f"q{index}a", "question_type": "why",
                 "stem": f"Left {index}", "answer": f"Answer L{index}"},
        "right": {"question_id": f"q{index}b", "question_type": "why",
                  "stem": f"Right {index}", "answer": f"Answer R{index}"},
    } for index in range(13)]
    monkeypatch.setattr(
        duplicate_judge, "shortlist_duplicate_pairs",
        lambda _exam, _forced=None: (pairs, {"pair_count": 13}),
    )
    monkeypatch.setattr(duplicate_judge, "DUPLICATE_JUDGE_BATCH_SIZE", 1)
    monkeypatch.setattr(duplicate_judge, "MAX_PARALLEL_DUPLICATE_JUDGE_CALLS", 10)
    monkeypatch.setattr(duplicate_judge, "create_llm_client", lambda: SimpleNamespace(last_usage={}))
    active = 0
    peak = 0
    calls = 0
    lock = threading.Lock()

    def fake_request(_client, _prompt, **kwargs):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"verdicts": [{
            "pair_id": kwargs["expected_ids"][0],
            **{field: "no" for field in duplicate_judge.SEMANTIC_FACT_FIELDS},
        }]}

    monkeypatch.setattr(duplicate_judge, "request_structured", fake_request)
    result = duplicate_judge.review_duplicate_candidates({"questions": {}})
    assert calls == 13
    assert peak == 10
    assert [row["pair_id"] for row in result["pair_verdicts"]] == [
        pair["pair_id"] for pair in pairs
    ]


def test_validation_ledger_reuses_unchanged_failed_verdict(monkeypatch):
    calls: list[list[str] | None] = []

    def fake_validate(_exam, question_ids, _plans, _sources, *_args):
        ids = list(question_ids) if question_ids is not None else None
        calls.append(ids)
        return {
            "model_number": 1, "all_pass": False, "warnings": [], "coverage": [],
            "verdicts": [{
                "question_id": "q1", "question_valid": False,
                "answer_valid": True, "action": "FIX_QUESTION",
                "fields_to_fix": ["question"], "failure_code": "clarity",
                "expected_results": [], "reason": "unclear", "expected_fix": "clarify",
                "validation_role": "general",
            }],
        }

    monkeypatch.setattr(validator, "_validate_with_context", fake_validate)
    state = {
        "generated_exams": [{"model_number": 1, "questions": {"why": [{
            "slot_id": "s1", "question_id": "q1", "question": "Why?",
            "reference_answer": "Because.",
        }]}}],
        "plans": [{"model_number": 1, "items": {"why": [{
            "slot_id": "s1", "question_type": "why", "task_id": "t1",
            "source_chunk_id": "c1", "support_chunk_ids": [],
        }]}}],
        "retrieved_chunks": [{"child_id": "c1", "content": "Evidence"}],
        "validation_reports": [], "validated_models": [],
        "question_repair_attempts": {}, "pending_revalidation_ids": {},
        "repair_manifest": [], "repair_candidates": [], "warnings": [],
    }
    state.update(validator.validate_generated_questions(state))
    assert len(calls) == 1
    state.update(validator.validate_generated_questions(state))
    assert len(calls) == 1
    state["generated_exams"][0]["questions"]["why"][0]["question"] = "Why exactly?"
    state.update(validator.validate_generated_questions(state))
    assert len(calls) == 2
    assert calls[-1] == ["q1"]


def test_specialized_validator_findings_share_one_repair_record():
    general = {"model_number": 1, "coverage": [], "warnings": [], "verdicts": [{
        "question_id": "q4", "action": "FIX_OPTIONS", "failure_code": "options",
        "fields_to_fix": ["options"], "reason": "bad options", "expected_fix": "replace",
        "validation_role": "general",
    }]}
    quantitative = {"model_number": 1, "coverage": [], "warnings": [], "verdicts": [{
        "question_id": "q4", "action": "FIX_ANSWER", "failure_code": "incorrect_calculation",
        "fields_to_fix": ["solution_steps"], "reason": "bad step", "expected_fix": "recalculate",
        "validation_role": "stem_quantitative",
    }]}
    merged = validator._merge_role_reports(general, quantitative)
    verdict = merged["verdicts"][0]
    assert verdict["validation_role"] == "combined"
    assert verdict["fields_to_fix"] == ["options", "solution_steps"]
    assert [item["validator"] for item in verdict["required_fixes"]] == [
        "general", "stem_quantitative",
    ]


def test_independent_repair_jobs_overlap_and_merge_by_slot(monkeypatch):
    barrier = threading.Barrier(2)

    def fake_serial(local_state):
        target = dict(local_state["repair_manifest"][0])
        barrier.wait(timeout=2)
        target["status"] = "candidate_generated"
        slot = target["slot_id"]
        return {
            "repair_manifest": [target], "repair_candidates": [{"slot_id": slot}],
            "question_repair_attempts": {slot: 1}, "repair_replacement_memory": {},
            "repair_retry_seeds": {}, "validation_repair_history": [{"slot_id": slot}],
            "validated_models": [], "warnings": [], "eval_stats": None,
        }

    monkeypatch.setattr(validator, "_repair_invalid_questions_serial", fake_serial)
    state = {
        "repair_manifest": [
            {"model_number": 1, "slot_id": "s1", "question_type": "why", "status": "invalid"},
            {"model_number": 1, "slot_id": "s2", "question_type": "equation", "status": "invalid"},
        ],
        "question_repair_attempts": {}, "repair_replacement_memory": {},
        "repair_retry_seeds": {}, "validation_repair_history": [],
        "validated_models": [1], "warnings": [],
    }
    result = validator.repair_invalid_questions(state)
    assert [item["slot_id"] for item in result["repair_candidates"]] == ["s1", "s2"]
    assert all(item["status"] == "candidate_generated" for item in result["repair_manifest"])


def test_independent_candidate_revalidations_overlap_and_commit_separately(monkeypatch):
    barrier = threading.Barrier(2)

    def fake_validate(exam, question_ids, *_args):
        barrier.wait(timeout=2)
        qid = list(question_ids)[0]
        return {"model_number": 1, "all_pass": True, "warnings": [], "coverage": [],
                "verdicts": [{"question_id": qid, "action": "PASS",
                              "validation_role": "general", "fields_to_fix": []}]}

    monkeypatch.setattr(validator, "_validate_with_context", fake_validate)
    active_exam = {"model_number": 1, "questions": {"why": [
        {"slot_id": "s1", "question_id": "q1", "question": "old 1", "reference_answer": "a"},
        {"slot_id": "s2", "question_id": "q2", "question": "old 2", "reference_answer": "b"},
    ]}}
    plans = [{"model_number": 1, "items": {"why": [
        {"slot_id": "s1", "question_type": "why"},
        {"slot_id": "s2", "question_type": "why"},
    ]}}]
    candidates = []
    for index in (1, 2):
        staged = {"model_number": 1, "questions": {"why": [
            {"slot_id": f"s{index}", "question_id": f"q{index}",
             "question": f"new {index}", "reference_answer": "ok"},
        ]}}
        candidates.append({"model_number": 1, "slot_id": f"s{index}",
                           "question_id": f"q{index}", "question_type": "why",
                           "exam": staged, "plans": plans, "attempt": 1})
    state = {
        "generated_exams": [active_exam], "plans": plans, "repair_candidates": candidates,
        "repair_manifest": [
            {"model_number": 1, "slot_id": "s1", "status": "invalid"},
            {"model_number": 1, "slot_id": "s2", "status": "invalid"},
        ],
        "validated_models": [1], "validation_reports": [], "warnings": [],
    }
    result = validator.validate_generated_questions(state)
    assert [item["question"] for item in result["generated_exams"][0]["questions"]["why"]] == [
        "new 1", "new 2",
    ]
    assert all(item["status"] == "resolved" for item in result["repair_manifest"])


def test_deterministic_replacement_prefers_conflict_resolution_before_usage(monkeypatch):
    candidates = [
        {"task_id": "same", "concept": "motion", "target": "distance",
         "task_operation": "calculate", "reasoning_mode": "substitution"},
        {"task_id": "different", "concept": "motion", "target": "time",
         "task_operation": "rearrange", "reasoning_mode": "algebra"},
    ]
    active = [{"slot_id": "conflict", "task_id": "old", "concept": "motion",
               "target": "distance", "task_operation": "calculate",
               "reasoning_mode": "substitution"}]
    chosen, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="replace", qtype="word_problem", conflict_slots=["conflict"],
        conflict_edges=[{"reason": "same target and reasoning"}], candidates=candidates,
        active_items=active, blacklist=set(), previous_failures=[],
        allow_semantic_overlap_candidates=True,
    )
    assert chosen["task_id"] == "different"
    assert context["selection_source"] == "deterministic_conflict_ranking"


def test_catalog_cache_key_ignores_unrelated_module_bytes(monkeypatch):
    before = assessment_catalog.catalog_cache_key("doc", ["c1"])
    monkeypatch.setattr(assessment_catalog.Path, "read_bytes", lambda _self: b"unrelated logging")
    assert assessment_catalog.catalog_cache_key("doc", ["c1"]) == before


def test_catalog_prompt_has_no_task_quota_and_batching_ignores_child_count():
    prompt = assessment_catalog._catalog_prompt(
        "source", "en", ["c1"], domain_type="stem", subject="mathematics"
    )
    assert "no more than" not in prompt.casefold()
    assert "minimum, quota, or maximum" in prompt
    chunks = _parent_chunks(3)
    batches = assessment_catalog._balanced_catalog_batches(chunks, 99)
    assert [len(batch) for batch in batches] == [3, 3]


def test_catalog_finalizer_does_not_truncate_at_legacy_48_task_cap():
    tasks = [{
        "knowledge_anchor": f"knowledge {index}", "topic": "math",
        "concept": f"concept {index}", "relationship": None,
        "task_operation": "solve", "student_work_product": "answer",
        "target": f"target {index}", "reasoning_mode": "quantitative_reasoning",
        "allowed_question_types": ["equation"], "evidence_chunk_ids": ["c1"],
        "required_givens": [], "allowed_variations": [], "variable_givens": [],
        "complexity_boundaries": [], "exact_source_instance": False,
        "scenario_family": None, "text_answerable": True,
        "native_difficulty": "easy", "reasoning_depth": "easy",
        "reasoning_requirements": ["solve"], "prerequisites": [],
    } for index in range(60)]
    finalized = assessment_catalog._finalize_catalog(
        {"assessment_profile": {}, "tasks": tasks, "fitb_bank": []},
        key="k", processed_document_key="doc", selected_chunk_ids=["c1"], usage=None,
    )
    assert len(finalized["tasks"]) == 60


def _review_slot(index: int, **overrides):
    item = {
        "slot_id": f"s{index}", "question_type": "why", "task_id": f"t{index}",
        "task_family_id": f"f{index}", "coverage_id": f"c{index}",
        "concept": f"concept {index}", "task_operation": f"explain {index}",
        "target": f"target {index}", "reasoning_mode": "causal_reasoning",
        "reasoning_requirements": [f"reason {index}"],
        "student_work_product": "short explanation", "source_chunk_id": "chunk1",
        "support_chunk_ids": [], "difficulty": "medium", "actual_difficulty": "medium",
    }
    item.update(overrides)
    return item


def test_plan_pair_shortlist_keeps_known_conflict_below_threshold(monkeypatch):
    plan_review._PLAN_EMBEDDING_CACHE.clear()
    plan = {"model_number": 1, "items": {"why": [_review_slot(1), _review_slot(2)]}}
    monkeypatch.setattr(plan_review, "embed_dense", lambda *_args, **_kwargs: [[1, 0], [0, 1]])

    pairs, stats = plan_review._shortlist_plan_pairs(plan, [{
        "repair_slot_id": "s2", "conflict_slot_id": "s1", "reason": "known duplicate",
    }])

    assert len(pairs) == 1
    assert pairs[0]["embedding_similarity"] < plan_review.PLAN_PAIR_SIMILARITY_THRESHOLD
    assert "known_conflict" in pairs[0]["shortlist_reasons"]
    assert stats["possible_pair_count"] == 1


def test_plan_pair_shortlist_always_keeps_shared_identity(monkeypatch):
    plan_review._PLAN_EMBEDDING_CACHE.clear()
    left = _review_slot(1, task_family_id="shared")
    right = _review_slot(2, task_family_id="shared")
    plan = {"model_number": 1, "items": {"why": [left, right]}}
    monkeypatch.setattr(plan_review, "embed_dense", lambda *_args, **_kwargs: [[1, 0], [0, 1]])

    pairs, _stats = plan_review._shortlist_plan_pairs(plan)

    assert "shared_identity" in pairs[0]["shortlist_reasons"]


def test_plan_pair_shortlist_reduces_twenty_one_slot_review(monkeypatch):
    plan_review._PLAN_EMBEDDING_CACHE.clear()
    items = [_review_slot(index) for index in range(21)]
    plan = {"model_number": 1, "items": {"why": items}}
    vectors = [[1.0 if i == j else 0.0 for j in range(21)] for i in range(21)]
    monkeypatch.setattr(plan_review, "embed_dense", lambda *_args, **_kwargs: vectors)

    pairs, stats = plan_review._shortlist_plan_pairs(plan)

    assert stats["possible_pair_count"] == 210
    assert 0 < len(pairs) < 210
    assert stats["shortlisted_pair_count"] == len(pairs)
    assert all("top_neighbour" in pair["shortlist_reasons"] for pair in pairs)
