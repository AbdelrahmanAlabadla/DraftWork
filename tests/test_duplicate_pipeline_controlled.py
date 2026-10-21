from __future__ import annotations

from types import SimpleNamespace

from app.llm.deepseek import DeepSeekNetworkError
from app.online import assessment_catalog, duplicate_judge, duplicate_pipeline, validator


def _sa(qid: str, stem: str, answer: str) -> dict:
    return {"question_id": qid, "slot_id": f"s_{qid}", "question": stem, "reference_answer": answer}


def _mcq(qid: str, stem: str, answer: str) -> dict:
    return {"question_id": qid, "slot_id": f"s_{qid}", "question": stem,
            "options": {"A": answer, "B": "distractor 1", "C": "distractor 2", "D": "distractor 3"},
            "correct_answer": "A"}


def _word(qid: str, stem: str, steps: list[str], answer: str) -> dict:
    return {"question_id": qid, "slot_id": f"s_{qid}", "question": stem,
            "solution_steps": steps, "final_answer": answer}


def _task(qid: str, qtype: str) -> dict:
    return {
        "task_id": f"task_{qid}", "task_family_id": f"family_{qid}",
        "coverage_id": f"coverage_{qid}", "concept_id": f"concept_{qid}",
        "relationship_id": None, "knowledge_anchor": f"knowledge {qid}",
        "topic": "controlled", "concept": f"concept {qid}", "relationship": None,
        "task_operation": "explain", "student_work_product": "complete response",
        "target": f"target {qid}", "reasoning_mode": "conceptual_reasoning",
        "supported_difficulties": ["easy", "medium", "hard"],
        "allowed_question_types": [qtype], "evidence_chunk_ids": ["c1"],
        "required_givens": [], "scenario_family": None, "text_answerable": True,
    }


def _plan_item(task: dict, qid: str) -> dict:
    return {
        "slot_id": f"s_{qid}", "task_id": task["task_id"],
        "task_family_id": task["task_family_id"],
        "coverage_id": task["coverage_id"], "concept_id": task["concept_id"],
        "relationship_id": None, "source_chunk_id": "c1", "support_chunk_ids": [],
        "topic": task["topic"], "concept": task["concept"], "canonical_skill": "explain",
        "knowledge_focus": task["knowledge_anchor"], "student_work_product": "complete response",
        "relationship": None, "target": task["target"], "task_operation": "explain",
        "given_conditions": [], "required_givens": [], "scenario_family": None,
        "text_answerable": True, "reasoning_mode": "conceptual_reasoning",
        "difficulty": "medium", "idea_to_test": task["target"], "concept_to_test": task["target"],
    }


def test_controlled_12_question_correctness_then_grouped_duplicate_flow(monkeypatch):
    # This lifecycle test expects one Judge call per scan. Parallel Judge
    # batching has dedicated coverage in test_performance_optimizations.py.
    monkeypatch.setattr(duplicate_judge, "DUPLICATE_JUDGE_BATCH_SIZE", 1000)
    questions = {
        "why": [
            _sa("q1", "What does mechanical work mean?", "Energy transfer by force through displacement."),
            _sa("q2", "Explain the meaning of work in mechanics.", "Energy transfer by force through displacement."),
        ],
        "mcq": [
            _mcq("q3", "Which equation defines power?", "P = W/t"),
            _mcq("q4", "Select the formula for the rate of doing work.", "P = W/t"),
            *[_mcq(f"q{i}", f"Unique controlled concept {i}?", f"Unique answer {i}") for i in range(7, 13)],
        ],
        "word_problem": [
            _word("q5", "State the photon-energy relationship.", ["Recall the law."], "E = hν"),
            _word("q6", "A photon has frequency 3 Hz and h=2 J s. Calculate its energy.",
                  ["Use E = hν", "E = 2 x 3"], "6 J"),
        ],
    }
    # q12 is the planted correctness defect; duplicate adjudication must wait for its repair.
    questions["mcq"][-1]["correct_answer"] = "D"
    exam = {"model_number": 1, "questions": questions}

    type_by_qid = {**{"q1": "why", "q2": "why", "q3": "mcq", "q4": "mcq",
                      "q5": "word_problem", "q6": "word_problem"},
                   **{f"q{i}": "mcq" for i in range(7, 13)}}
    tasks = [_task(qid, qtype) for qid, qtype in type_by_qid.items()]
    tasks.extend([_task("replacement_sa", "why"), _task("replacement_mcq", "mcq")])
    plan_items = {"why": [], "mcq": [], "word_problem": []}
    by_task = {task["task_id"]: task for task in tasks}
    for qid, qtype in type_by_qid.items():
        plan_items[qtype].append(_plan_item(by_task[f"task_{qid}"], qid))

    validation_calls = []
    def fake_validate(exam_value, question_ids, _plans, _sources):
        selected = list(question_ids or [
            item["question_id"] for section in exam_value["questions"].values() for item in section
        ])
        validation_calls.append(selected)
        verdicts = []
        for qid in selected:
            broken = qid == "q12" and exam_value["questions"]["mcq"][-1]["correct_answer"] == "D"
            verdicts.append({
                "question_id": qid, "question_valid": True, "answer_valid": not broken,
                "action": "FIX_ANSWER" if broken else "PASS",
                "fields_to_fix": ["answer"] if broken else [],
                "failure_code": "incorrect_answer" if broken else "pass",
                "expected_results": ["A"] if broken else [], "reason": "planted" if broken else "",
                "expected_fix": "Use A" if broken else "",
            })
        return {"model_number": 1, "all_pass": not any(v["action"] != "PASS" for v in verdicts),
                "verdicts": verdicts, "warnings": [], "coverage": []}
    monkeypatch.setattr(validator, "_validate_with_context", fake_validate)

    def fake_correctness_repair(exam_value, verdicts, **_kwargs):
        exam_value["questions"]["mcq"][-1]["correct_answer"] = "A"
        return {"warnings": [], "sent_ids": ["q12"], "repaired_ids": ["q12"], "failed_ids": []}
    monkeypatch.setattr(validator, "repair_exam", fake_correctness_repair)

    # Registry order is q1,q2,q3,q4,q7..q12,q5,q6. Plant three close pairs.
    vectors = []
    for index in range(12):
        vector = [0.0] * 9
        if index in (0, 1): vector[0] = 1.0
        elif index in (2, 3): vector[1] = 1.0
        elif index in (10, 11): vector[2] = 1.0
        else: vector[index - 1] = 1.0
        vectors.append(vector)
    monkeypatch.setattr(duplicate_judge, "embed_dense", lambda texts, max_length=1024: vectors)
    monkeypatch.setattr(duplicate_judge, "create_llm_client", lambda: SimpleNamespace(last_usage={}))
    judge_calls = 0
    def fake_judge(_client, _prompt, **kwargs):
        nonlocal judge_calls
        judge_calls += 1
        classifications = {"pair_1_2": "duplicate", "pair_3_4": "excessive_overlap",
                           "pair_11_12": "acceptable_reuse"}
        fact_values = {
            "duplicate": ("yes", "yes", "yes", "no", "no"),
            "excessive_overlap": ("yes", "no", "yes", "yes", "yes"),
            "acceptable_reuse": ("no", "no", "no", "yes", "no"),
        }
        verdicts = []
        for pair_id in kwargs["expected_ids"]:
            label = (
                classifications.get(pair_id, "acceptable_reuse")
                if judge_calls == 1 else "acceptable_reuse"
            )
            verdicts.append(dict(zip(
                ("pair_id", *duplicate_judge.SEMANTIC_FACT_FIELDS),
                (pair_id, *fact_values[label]),
            )))
        return {"verdicts": verdicts}
    monkeypatch.setattr(duplicate_judge, "request_structured", fake_judge)
    monkeypatch.setattr(
        duplicate_pipeline, "_select_replacement_candidate",
        lambda **kwargs: (
            (kwargs.get("candidates") or [None])[0],
            {"selection_source": "controlled_test"},
        ),
    )

    def fake_generate(_state, plans, _model_number, qid, _exam):
        task_id = _plan_item_by_slot(plans, f"s_{qid}")["task_id"]
        if qid in {"q1", "q2"}:
            return qid, _sa(qid, "Explain a newly reserved concept.", f"Answer for {task_id}"), "why", []
        return qid, _mcq(qid, "Which newly reserved concept is correct?", f"Answer for {task_id}"), "mcq", []
    monkeypatch.setattr(duplicate_pipeline, "_generate_reserved", fake_generate)

    state = {
        "generated_exams": [exam], "plans": [{"model_number": 1, "items": plan_items}],
        "assessment_catalog": {"tasks": tasks}, "difficulty": "mix",
        "retrieved_chunks": [{"child_id": "c1", "content": "controlled evidence"}],
        "validation_reports": [], "validated_models": [], "question_repair_attempts": {},
        "pending_revalidation_ids": {}, "duplicate_reports": [],
        "duplicate_similarity_distributions": [], "pending_duplicate_groups": [],
        "duplicate_repair_rounds": 0, "duplicate_replacement_attempts": {},
        "duplicate_scan_count": 0,
        "duplicate_repair_history": [], "validation_repair_history": [],
        "invalid_questions": [], "warnings": [], "eval_stats": None,
    }

    state.update(validator.validate_generated_questions(state))
    assert state["validation_reports"][0]["all_pass"] is False
    assert judge_calls == 0
    state.update(validator.repair_invalid_questions(state))
    state.update(validator.validate_generated_questions(state))
    assert state["validation_reports"][0]["all_pass"] is True

    state.update(duplicate_pipeline.scan_exam_duplicates(state))
    assert len(state["pending_duplicate_groups"]) == 2
    assert state["duplicate_similarity_distributions"][0]["pair_count"] == 66
    answer_positions_before_repair = {
        item["question_id"]: item["correct_answer"]
        for item in state["generated_exams"][0]["questions"]["mcq"]
    }
    state.update(duplicate_pipeline.repair_duplicate_groups(state))
    reserved = [event["task_id"] for event in state["duplicate_repair_history"]
                if event.get("status") == "replaced"]
    replaced_ids = {event["question_id"] for event in state["duplicate_repair_history"]
                    if event.get("status") == "replaced"}
    assert len(reserved) == len(set(reserved)) == 2
    assert state["duplicate_repair_rounds"] == 0
    assert state["duplicate_verification_pending"] is True
    assert state["duplicate_verification_attempt"] == 1
    assert set(state["duplicate_replacement_attempts"].values()) == {1}
    answer_positions_after_repair = {
        item["question_id"]: item["correct_answer"]
        for item in state["generated_exams"][0]["questions"]["mcq"]
    }
    balanced_ids = {
        qid for qid, answer in answer_positions_after_repair.items()
        if answer_positions_before_repair.get(qid) != answer
    }
    affected_ids = replaced_ids | balanced_ids
    assert set(state["pending_revalidation_ids"][1]) == affected_ids

    state.update(validator.validate_generated_questions(state))
    assert set(validation_calls[-1]) == affected_ids
    state.update(duplicate_pipeline.scan_exam_duplicates(state))
    assert state["pending_duplicate_groups"] == []
    assert state["invalid_questions"] == []
    assert judge_calls == 2
    assert state["duplicate_repair_rounds"] == 1
    assert state["duplicate_verification_pending"] is False


def _plan_item_by_slot(plans: list[dict], slot_id: str) -> dict:
    return next(item for plan in plans for values in plan["items"].values()
                for item in values if item["slot_id"] == slot_id)


def test_authorized_reused_task_is_forced_into_duplicate_comparison():
    task = _task("shared", "why")
    first = _plan_item(task, "q1")
    reused = _plan_item(task, "q2")
    reused.update({
        "catalog_replacement_stage": "reused_task_after_catalog_exhaustion",
        "reused_due_to_catalog_exhaustion": True,
        "task_reused": True,
        "reuse_reason": "no_unused_compatible_task",
        "reused_task_id": task["task_id"],
        "required_instance_distinctions": ["different_required_method"],
        "forced_duplicate_slot_ids": ["s_q1"],
    })
    exam = {"model_number": 1, "questions": {"why": [
        _sa("q1", "Directly translate a stated relationship.", "y=2x+3"),
        _sa("q2", "Infer a missing value, then derive a relationship.", "b=20-2x"),
    ]}}
    plans = [{"model_number": 1, "items": {"why": [first, reused]}}]

    forced = duplicate_pipeline._forced_catalog_reuse_pairs(exam, plans, 1)

    assert frozenset(("q1", "q2")) in forced


def test_failed_replacement_call_does_not_discard_successful_parallel_replacement(monkeypatch):
    exam = {"model_number": 1, "questions": {"why": [
        _sa("q1", "Keep first", "A"),
        _sa("q2", "Replace first", "A"),
        _sa("q3", "Keep second", "B"),
        _sa("q4", "Replace second", "B"),
    ]}}
    current_tasks = [_task(qid, "why") for qid in ("q1", "q2", "q3", "q4")]
    replacement_tasks = [_task(qid, "why") for qid in ("replacement_1", "replacement_2")]
    plans = [{"model_number": 1, "items": {"why": [
        _plan_item(task, task["task_id"].removeprefix("task_"))
        for task in current_tasks
    ]}}]

    def generate_one(_state, _plans, _model_number, qid, _exam):
        if qid == "q2":
            raise DeepSeekNetworkError("temporary replacement failure")
        return qid, _sa(qid, "Distinct successful replacement", "new answer"), "why", []

    monkeypatch.setattr(duplicate_pipeline, "_generate_reserved", generate_one)
    state = {
        "generated_exams": [exam],
        "plans": plans,
        "assessment_catalog": {"tasks": [*current_tasks, *replacement_tasks]},
        "difficulty": "medium",
        "pending_duplicate_groups": [
            {
                "model_number": 1, "group_id": "g1",
                "question_ids": ["q1", "q2"], "edges": [],
                "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
            },
            {
                "model_number": 1, "group_id": "g2",
                "question_ids": ["q3", "q4"], "edges": [],
                "keep_question_ids": ["q3"], "replace_question_ids": ["q4"],
            },
        ],
        "duplicate_repair_rounds": 0,
        "duplicate_replacement_attempts": {},
        "pending_revalidation_ids": {},
        "validated_models": [1],
        "duplicate_repair_history": [],
        "warnings": [],
        "duplicate_fact_cache": {},
        "developer_diagnostics": [],
    }

    result = duplicate_pipeline.repair_duplicate_groups(state)

    by_id = {
        item["question_id"]: item for item in exam["questions"]["why"]
    }
    assert by_id["q2"]["question"] == "Replace first"
    assert by_id["q4"]["question"] == "Distinct successful replacement"
    statuses = {
        event["question_id"]: event["status"]
        for event in result["duplicate_repair_history"]
    }
    assert statuses == {"q2": "generation_failed", "q4": "replaced"}
    assert result["pending_revalidation_ids"] == {1: ["q4"]}


def test_fitb_duplicate_limit_stops_even_while_unused_inventory_exists(monkeypatch):
    fitb_question = {
        "question_id": "q_fitb", "slot_id": "s_q_fitb",
        "question": "The ultraviolet hydrogen series is ____.",
        "answers": ["Lyman series"],
        "distractors": ["Balmer series", "Paschen series"],
    }
    exam = {"model_number": 1, "questions": {
        "fill_in_the_blank": {"word_bank": ["Lyman series", "Balmer series", "Paschen series"],
                              "items": [fitb_question]},
    }}
    current = assessment_catalog.canonical_fitb_entry({
        "answer": "Lyman series", "evidence_chunk_id": "c1",
        "clue_basis": "hydrogen series ending at the ground level",
    }, {"c1"})
    alternate = assessment_catalog.canonical_fitb_entry({
        "answer": "Paschen series", "evidence_chunk_id": "c1",
        "clue_basis": "hydrogen series ending at the third level",
    }, {"c1"})
    plans = [{"model_number": 1, "items": {"fill_in_the_blank": [
        assessment_catalog.materialize_fitb_plan_item(current, "s_q_fitb", "medium")
    ]}}]
    monkeypatch.setattr(duplicate_pipeline.config, "MAX_DUPLICATE_REPAIR_ROUNDS", 2)
    monkeypatch.setattr(duplicate_pipeline.config, "MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT", 2)
    monkeypatch.setattr(duplicate_pipeline, "review_duplicate_candidates", lambda *_args, **_kwargs: {
        "duplicate_groups": [{
            "group_id": "g1", "question_ids": ["q_keep", "q_fitb"],
            "keep_question_ids": ["q_keep"], "replace_question_ids": ["q_fitb"],
            "edges": [],
        }],
        "pair_verdicts": [], "candidate_pairs": [], "similarity_distribution": {},
        "fact_cache": {},
    })
    state = {
        "generated_exams": [exam], "plans": plans,
        "assessment_catalog": {"tasks": [], "fitb_bank": [current, alternate]},
        "difficulty": "medium",
        "validation_reports": [{"model_number": 1, "verdicts": [{
            "question_id": "q_fitb", "action": "PASS",
        }]}],
        "duplicate_reports": [], "duplicate_similarity_distributions": [],
        "pending_duplicate_groups": [], "duplicate_scan_count": 1,
        "duplicate_repair_rounds": 2,
        "duplicate_replacement_attempts": {"1:s_q_fitb": 2},
        "duplicate_repair_history": [], "invalid_questions": [], "warnings": [],
        "duplicate_fact_cache": {},
    }

    result = duplicate_pipeline.scan_exam_duplicates(state)

    assert result["pending_duplicate_groups"] == []
    assert result["invalid_questions"] == []
    assert "stopped before another Duplicate Judge call" in result["warnings"][-1]


def test_exact_task_reuse_is_transactional_and_rolls_back_failed_precommit(monkeypatch):
    exam = {"model_number": 1, "questions": {"why": [
        _sa("q1", "Existing task A", "answer A"),
        _sa("q2", "Original task B", "answer B"),
    ]}}
    task_a = _task("q1", "why")
    task_b = _task("q2", "why")
    plans = [{"model_number": 1, "items": {"why": [
        _plan_item(task_a, "q1"), _plan_item(task_b, "q2"),
    ]}}]
    captured = {}
    def generate_reused(_state, generated_plans, *_args, **_kwargs):
        reused_plan = _plan_item_by_slot(generated_plans, "s_q2")
        captured["previous_task_questions"] = reused_plan.get("previous_task_questions")
        return "q2", _sa("q2", "Fallback candidate", "candidate answer"), "why", []
    monkeypatch.setattr(duplicate_pipeline, "_generate_reserved", generate_reused)
    monkeypatch.setattr(
        duplicate_pipeline, "_precommit_fallback_candidate",
        lambda *_args, **_kwargs: (False, ["forced duplicate conflict"], {}),
    )
    state = {
        "generated_exams": [exam], "plans": plans,
        # No alternative compatible catalog task remains, so this exercises the
        # final original-task fallback and its transactional rollback.
        "assessment_catalog": {"tasks": [task_b]}, "difficulty": "medium",
        "pending_duplicate_groups": [{
            "model_number": 1, "group_id": "g1", "question_ids": ["q1", "q2"],
            "edges": [{"left": "q1", "right": "q2", "classification": "duplicate"}],
            "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
        }],
        "duplicate_repair_rounds": 0, "duplicate_replacement_attempts": {},
        "pending_revalidation_ids": {}, "validated_models": [1],
        "duplicate_repair_history": [], "warnings": [], "duplicate_fact_cache": {},
    }

    result = duplicate_pipeline.repair_duplicate_groups(state)

    assert exam["questions"]["why"][1]["question"] == "Original task B"
    assert _plan_item_by_slot(result["plans"], "s_q2")["task_id"] == "task_q2"
    assert result["duplicate_repair_history"][-1]["status"] == "precommit_rejected"
    assert result["pending_revalidation_ids"] == {}
    assert captured["previous_task_questions"]
    roles = {
        value.get("context_role")
        for value in captured["previous_task_questions"]
    }
    assert "rejected_duplicate_question" in roles
    assert "duplicate_conflict" in roles


def test_failed_original_regeneration_advances_to_other_used_task(monkeypatch):
    exam = {"model_number": 1, "questions": {"why": [
        _sa("q1", "Existing task A", "answer A"),
        _sa("q2", "Rejected task B", "answer B"),
    ]}}
    task_a = _task("q1", "why")
    task_b = _task("q2", "why")
    plans = [{"model_number": 1, "items": {"why": [
        _plan_item(task_a, "q1"), _plan_item(task_b, "q2"),
    ]}}]
    captured = {}

    def generate_next(_state, generated_plans, *_args, **_kwargs):
        replacement = _plan_item_by_slot(generated_plans, "s_q2")
        captured.update(replacement)
        return "q2", _sa("q2", "Distinct fallback", "new answer"), "why", []

    monkeypatch.setattr(duplicate_pipeline, "_generate_reserved", generate_next)
    monkeypatch.setattr(
        duplicate_pipeline, "_precommit_fallback_candidate",
        lambda *_args, **_kwargs: (False, ["still duplicate"], {}),
    )
    state = {
        "generated_exams": [exam], "plans": plans,
        "assessment_catalog": {"tasks": [task_a, task_b]}, "difficulty": "medium",
        "pending_duplicate_groups": [{
            "model_number": 1, "group_id": "g1", "question_ids": ["q1", "q2"],
            "edges": [], "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
        }],
        "duplicate_repair_rounds": 1, "duplicate_replacement_attempts": {"1:s_q2": 1},
        "pending_revalidation_ids": {}, "validated_models": [1],
        "duplicate_repair_history": [{
            "model_number": 1, "slot_id": "s_q2", "question_id": "q2",
            "task_id": "task_q2", "status": "fallback_precommit_rejected",
        }],
        "warnings": [], "duplicate_fact_cache": {},
    }

    result = duplicate_pipeline.repair_duplicate_groups(state)

    assert captured["task_id"] == "task_q1"
    assert captured["reuse_mode"] == "intentional_practice_variation"
    assert captured["task_usage_count"] == 1
    assert result["duplicate_repair_history"][-1]["status"] == "replaced"
    assert result["pending_revalidation_ids"][1] == ["q2"]


def test_selector_keeps_authorized_same_skill_reuse_and_supplies_all_previous_questions(monkeypatch):
    reused = _plan_item(_task("reuse", "equation"), "candidate")
    reused.update({
        "task_reused": True,
        "reuse_mode": "intentional_practice_variation",
        "task_usage_count": 3,
    })
    active = []
    previous = []
    for index in range(3):
        active.append(dict(reused, slot_id=f"old_{index}"))
        previous.append({
            "question_id": f"q{index}", "slot_id": f"old_{index}",
            "task_id": reused["task_id"], "question": f"x + {index} = {index + 2}",
            "correct_answer": "x = 2", "solution_steps": ["subtract the constant"],
        })

    monkeypatch.setattr(
        "app.llm.client.request_structured",
        lambda *_args, **_kwargs: {
            "slot_to_replace": "target", "conflicts_with": [], "conflict_reason": "",
            "replacement_task_id": reused["task_id"], "replacement_fitb_entry_id": None,
            "why_this_candidate_resolves_the_conflict": "new numerical instance",
            "checked_against_active_slots": True, "rejected_candidates": [],
        },
    )
    chosen, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="target", qtype="equation", conflict_slots=[], conflict_edges=[],
        candidates=[reused], active_items=active, blacklist=set(), previous_failures=[],
        existing_questions=previous, allow_semantic_overlap_candidates=True,
    )

    assert chosen["task_id"] == reused["task_id"]
    assert len(context["candidates"][0]["previous_uses"]) == 3


def test_final_scan_exhaustion_marks_and_preserves_missing_slot(monkeypatch):
    from app.online.exam_builder import assemble_exams_node

    monkeypatch.setattr(duplicate_pipeline.config, "MAX_DUPLICATE_REPAIR_ROUNDS", 2)
    exam = {"model_number": 1, "questions": {"why": [
        _sa("q1", "Define A", "A"), _sa("q2", "Define A again", "A"),
    ]}}
    monkeypatch.setattr(duplicate_pipeline, "review_duplicate_candidates", lambda _exam: {
        "candidate_pairs": [], "pair_verdicts": [],
        "duplicate_groups": [{
            "group_id": "g1", "question_ids": ["q1", "q2"], "edges": [],
            "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
        }],
        "similarity_distribution": {"question_count": 2, "pair_count": 1}, "usage": {},
    })
    passes = [{"question_id": qid, "action": "PASS"} for qid in ("q1", "q2")]
    state = {
        "generated_exams": [exam], "validation_reports": [{
            "model_number": 1, "all_pass": True, "verdicts": passes,
        }],
        "duplicate_reports": [], "duplicate_similarity_distributions": [],
        "duplicate_repair_rounds": 1, "duplicate_replacement_attempts": {},
        "duplicate_scan_count": 1,
        "invalid_questions": [], "warnings": [], "document_language": "en",
        "eval_stats": None,
    }
    state.update(duplicate_pipeline.scan_exam_duplicates(state))
    assert state["pending_duplicate_groups"]
    assert state["invalid_questions"] == []
    state["duplicate_repair_rounds"] = 2
    state.update(duplicate_pipeline.scan_exam_duplicates(state))
    assert state["invalid_questions"] == []
    assert state["unresolved_duplicate_conflicts"]
    assert len(state["generated_exams"][0]["questions"]["why"]) == 2
    state.update(assemble_exams_node(state))
    assert state["invalid_questions"] == []


def test_slot_attempt_limit_is_independent_from_optional_third_global_round(monkeypatch):
    exam = {"model_number": 1, "questions": {"why": [
        _sa("q1", "Define A", "A"), _sa("q2", "Define A again", "A"),
    ]}}
    monkeypatch.setattr(duplicate_pipeline.config, "MAX_DUPLICATE_REPAIR_ROUNDS", 3)
    monkeypatch.setattr(duplicate_pipeline, "review_duplicate_candidates", lambda _exam: {
        "candidate_pairs": [], "pair_verdicts": [],
        "duplicate_groups": [{
            "group_id": "g1", "question_ids": ["q1", "q2"], "edges": [],
            "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
        }],
        "similarity_distribution": {"question_count": 2, "pair_count": 1}, "usage": {},
    })
    state = {
        "generated_exams": [exam],
        "validation_reports": [{"model_number": 1, "all_pass": True, "verdicts": [
            {"question_id": "q1", "action": "PASS"},
            {"question_id": "q2", "action": "PASS"},
        ]}],
        "duplicate_reports": [], "duplicate_similarity_distributions": [],
        "duplicate_repair_rounds": 2,
        "duplicate_replacement_attempts": {"1:s_q2": 3},
        "duplicate_scan_count": 2, "invalid_questions": [], "warnings": [],
    }

    state.update(duplicate_pipeline.scan_exam_duplicates(state))

    assert state["pending_duplicate_groups"] == []
    assert state["invalid_questions"] == []
    assert [item["question_id"] for item in exam["questions"]["why"]] == ["q1", "q2"]
    assert state["duplicate_repair_history"][-1]["status"] == "duplicate_repair_exhausted_kept"
    assert state["unresolved_duplicate_conflicts"]


def test_third_attempt_uses_numeric_variation_then_requires_duplicate_recheck(monkeypatch):
    from app.online.graph import _route_after_validation

    original = {
        "question_id": "q2", "slot_id": "s_q2",
        "equation": r"Solve: \[2x+5=15\]",
        "solution_steps": ["2x=10", "x=5"], "final_answer": "x=5",
    }
    conflict = {
        "question_id": "q1", "slot_id": "s_q1",
        "equation": r"Solve: \[2x+5=15\]",
        "solution_steps": ["2x=10", "x=5"], "final_answer": "x=5",
    }
    exam = {"model_number": 1, "questions": {"equation": [conflict, original]}}
    captured = {}

    def numeric(_state, _exam, model, qid, conflict_ids, edges):
        captured.update(model=model, qid=qid, conflict_ids=conflict_ids, edges=edges)
        return ({
            **original,
            "equation": r"Solve: \[4x+7=27\]",
            "solution_steps": ["4x=20", "x=5"], "final_answer": "x=5",
        }, [])

    monkeypatch.setattr(duplicate_pipeline, "_numeric_variation_fallback", numeric)
    monkeypatch.setattr(
        duplicate_pipeline, "_candidate_pool",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("attempt 3 selected a catalog task")),
    )
    edge = {"left": "q1", "right": "q2", "classification": "duplicate"}
    result = duplicate_pipeline.repair_duplicate_groups({
        "generated_exams": [exam], "plans": [{"model_number": 1, "items": {"equation": [{
            "slot_id": "s_q2", "task_id": "t2", "idea_to_test": "solve a linear equation",
            "reasoning_mode": "symbolic_manipulation", "difficulty": "easy",
            "variable_givens": ["coefficients and constants"],
                "allowed_variations": [{
                    "target": "solution", "operation": "solve_equation",
                    "givens": ["coefficients and constants"], "representation": None,
                    "application": None, "numeric_values_may_change": True,
                }],
        }]}}],
        "pending_duplicate_groups": [{
            "model_number": 1, "question_ids": ["q1", "q2"],
            "keep_question_ids": ["q1"], "replace_question_ids": ["q2"], "edges": [edge],
        }],
        "duplicate_repair_rounds": 2,
        "duplicate_replacement_attempts": {"1:s_q2": 2},
        "pending_revalidation_ids": {}, "validated_models": [1],
        "duplicate_repair_history": [], "warnings": [],
    })

    repaired = result["generated_exams"][0]["questions"]["equation"][1]
    assert captured == {"model": 1, "qid": "q2", "conflict_ids": ["q1"], "edges": [edge]}
    assert repaired["question_id"] == "q2"
    assert repaired["duplicate_repair_attempt"] == 3
    assert repaired["duplicate_repair_mode"] == "numeric_variation_fallback"
    assert repaired["duplicate_conflict_with"] == "q1"
    assert repaired["numeric_variant"] is True
    assert repaired["duplicate_recheck_required"] is True
    assert result["pending_revalidation_ids"] == {1: ["q2"]}
    assert result["duplicate_repair_rounds"] == 3
    # The final numeric variation is no longer exempt: it must receive an
    # independent Duplicate Judge scan rather than skipping straight to assembly.
    assert result["duplicate_verification_pending"] is True
    assert result["duplicate_judge_exempt_ids"] == []
    assert _route_after_validation({
        "repair_manifest": [],
        "duplicate_repair_finalization_pending": True,
    }) == "assemble_exams"


def test_numeric_variation_requires_task_authorization_and_respects_exact_source_instance():
    item = {
        "question_id": "q1", "slot_id": "s1",
        "equation": r"Solve: \[2x+5=15\]", "solution_steps": ["x=5"],
        "final_answer": "x=5",
    }
    exam = {"model_number": 1, "questions": {"equation": [item]}}
    plan = {
        "slot_id": "s1", "task_id": "t1", "question_type": "equation",
        "task_operation": "solve_equation", "reasoning_mode": "symbolic_manipulation",
        "difficulty": "easy",
    }
    state = {"plans": [{"model_number": 1, "items": {"equation": [plan]}}]}

    assert duplicate_pipeline._numeric_variation_applicable(
        state, exam, 1, "q1", "equation", item,
    ) is False

    plan["variable_givens"] = ["coefficients and constants"]
    plan["allowed_variations"] = [{
        "target": "solution", "operation": "solve_equation",
        "givens": ["coefficients and constants"], "representation": None,
        "application": None, "numeric_values_may_change": True,
    }]
    assert duplicate_pipeline._numeric_variation_applicable(
        state, exam, 1, "q1", "equation", item,
    ) is True

    plan["exact_source_instance"] = True
    assert duplicate_pipeline._numeric_variation_applicable(
        state, exam, 1, "q1", "equation", item,
    ) is False


def test_third_attempt_keeps_nonnumeric_definition_without_calling_numeric_agent(monkeypatch):
    from app.online.graph import _route_after_validation

    exam = {"model_number": 1, "questions": {"definition": [
        {"question_id": "q1", "slot_id": "s1", "term": "osmosis",
         "reference_answer": "Movement of water through a membrane."},
        {"question_id": "q2", "slot_id": "s2", "term": "osmosis",
         "reference_answer": "Movement of water through a membrane."},
    ]}}
    monkeypatch.setattr(
        duplicate_pipeline, "_numeric_variation_fallback",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("numeric agent was called")),
    )
    group = {
        "model_number": 1, "group_id": "definitions", "question_ids": ["q1", "q2"],
        "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
        "edges": [{"left": "q1", "right": "q2", "classification": "duplicate"}],
    }
    result = duplicate_pipeline.repair_duplicate_groups({
        "generated_exams": [exam],
        "plans": [{"model_number": 1, "items": {"definition": [
            {"slot_id": "s1", "task_id": "t1"}, {"slot_id": "s2", "task_id": "t2"},
        ]}}],
        "pending_duplicate_groups": [group], "duplicate_repair_rounds": 2,
        "duplicate_replacement_attempts": {"1:s2": 2},
        "pending_revalidation_ids": {}, "validated_models": [1],
        "duplicate_repair_history": [], "warnings": [],
    })

    assert len(result["generated_exams"][0]["questions"]["definition"]) == 2
    assert result["duplicate_replacement_attempts"]["1:s2"] == 3
    assert result["duplicate_repair_rounds"] == 3
    assert result["pending_revalidation_ids"] == {}
    assert result["duplicate_repair_history"][-1]["status"] == "numeric_variation_not_applicable_kept"
    assert result["unresolved_duplicate_conflicts"] == [group]
    assert result["duplicate_repair_finalization_pending"] is True
    assert _route_after_validation({
        "repair_manifest": [], "duplicate_repair_finalization_pending": True,
    }) == "assemble_exams"


def test_final_allowed_replacement_verification_clears_stale_conflict(monkeypatch):
    calls = []
    monkeypatch.setattr(duplicate_pipeline.config, "MAX_DUPLICATE_REPAIR_ROUNDS", 3)
    monkeypatch.setattr(
        duplicate_pipeline,
        "review_duplicate_candidates",
        lambda _exam: calls.append(True) or {
            "candidate_pairs": [], "pair_verdicts": [], "duplicate_groups": [],
            "similarity_distribution": {"question_count": 2, "pair_count": 1},
            "usage": {},
        },
    )
    exam = {"model_number": 1, "questions": {"why": [
        _sa("q1", "Explain energy conservation.", "Energy remains constant."),
        _sa("q2", "Explain power.", "Power is the rate of energy transfer."),
    ]}}
    stale = {"model_number": 1, "group_id": "old", "question_ids": ["q1", "q2"]}
    passes = [{"question_id": qid, "action": "PASS"} for qid in ("q1", "q2")]

    result = duplicate_pipeline.scan_exam_duplicates({
        "generated_exams": [exam],
        "validation_reports": [{"model_number": 1, "all_pass": True, "verdicts": passes}],
        "duplicate_reports": [{"model_number": 1, "duplicate_groups": [stale]}],
        "duplicate_similarity_distributions": [], "duplicate_scan_count": 3,
        "duplicate_repair_rounds": 2,
        "duplicate_verification_pending": True,
        "duplicate_verification_attempt": 3,
        "unresolved_duplicate_conflicts": [stale],
        "pending_duplicate_groups": [], "invalid_questions": [], "warnings": [],
    })

    assert calls == [True]
    assert result["duplicate_repair_rounds"] == 3
    assert result["duplicate_verification_pending"] is False
    assert result["pending_duplicate_groups"] == []
    assert result["unresolved_duplicate_conflicts"] == []


def test_final_allowed_replacement_verification_persists_exact_conflict(monkeypatch):
    calls = []
    monkeypatch.setattr(duplicate_pipeline.config, "MAX_DUPLICATE_REPAIR_ROUNDS", 3)
    exact_group = {
        "group_id": "final", "question_ids": ["q1", "q2"],
        "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
        "edges": [{"left": "q1", "right": "q2", "classification": "duplicate"}],
    }
    monkeypatch.setattr(
        duplicate_pipeline,
        "review_duplicate_candidates",
        lambda _exam: calls.append(True) or {
            "candidate_pairs": [], "pair_verdicts": [],
            "duplicate_groups": [exact_group],
            "similarity_distribution": {"question_count": 2, "pair_count": 1},
            "usage": {},
        },
    )
    exam = {"model_number": 1, "questions": {"why": [
        _sa("q1", "Explain energy conservation.", "Energy remains constant."),
        _sa("q2", "Restate energy conservation.", "Energy remains constant."),
    ]}}
    passes = [{"question_id": qid, "action": "PASS"} for qid in ("q1", "q2")]

    result = duplicate_pipeline.scan_exam_duplicates({
        "generated_exams": [exam],
        "validation_reports": [{"model_number": 1, "all_pass": True, "verdicts": passes}],
        "duplicate_reports": [], "duplicate_similarity_distributions": [],
        "duplicate_scan_count": 3, "duplicate_repair_rounds": 2,
        "duplicate_verification_pending": True,
        "duplicate_verification_attempt": 3,
        "pending_duplicate_groups": [], "invalid_questions": [], "warnings": [],
        "assessment_catalog": {"tasks": [_task("unused", "why")]},
    })

    assert calls == [True]
    assert result["duplicate_repair_rounds"] == 3
    assert result["pending_duplicate_groups"] == []
    assert result["unresolved_duplicate_conflicts"] == [
        {"model_number": 1, **exact_group}
    ]
    assert len(result["generated_exams"][0]["questions"]["why"]) == 2


def test_duplicate_verification_completes_exactly_three_attempts(monkeypatch):
    calls = []
    monkeypatch.setattr(duplicate_pipeline.config, "MAX_DUPLICATE_REPAIR_ROUNDS", 3)
    monkeypatch.setattr(duplicate_pipeline.config, "MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT", 3)
    conflict = {
        "group_id": "persistent", "question_ids": ["q1", "q2"],
        "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
        "edges": [{"left": "q1", "right": "q2", "classification": "duplicate"}],
    }
    monkeypatch.setattr(
        duplicate_pipeline,
        "review_duplicate_candidates",
        lambda _exam: calls.append(True) or {
            "candidate_pairs": [], "pair_verdicts": [], "duplicate_groups": [conflict],
            "similarity_distribution": {"question_count": 2, "pair_count": 1},
            "usage": {},
        },
    )
    exam = {"model_number": 1, "questions": {"why": [
        _sa("q1", "Explain energy conservation.", "Energy remains constant."),
        _sa("q2", "Restate energy conservation.", "Energy remains constant."),
    ]}}
    passes = [{"question_id": qid, "action": "PASS"} for qid in ("q1", "q2")]
    state = {
        "generated_exams": [exam],
        "validation_reports": [{"model_number": 1, "all_pass": True, "verdicts": passes}],
        "duplicate_reports": [], "duplicate_similarity_distributions": [],
        "duplicate_scan_count": 0, "duplicate_repair_rounds": 0,
        "duplicate_replacement_attempts": {}, "pending_duplicate_groups": [],
        "invalid_questions": [], "warnings": [],
    }

    for attempt in (1, 2, 3):
        state["duplicate_verification_pending"] = True
        state["duplicate_verification_attempt"] = attempt
        state.update(duplicate_pipeline.scan_exam_duplicates(state))
        assert state["duplicate_repair_rounds"] == attempt
        if attempt < 3:
            assert state["pending_duplicate_groups"]
            assert state["unresolved_duplicate_conflicts"] == []

    assert calls == [True, True, True]
    assert state["pending_duplicate_groups"] == []
    assert state["unresolved_duplicate_conflicts"] == [
        {"model_number": 1, **conflict}
    ]


def test_compact_selector_rejects_unused_semantic_collision(monkeypatch):
    from app.llm import client as llm_client
    monkeypatch.setattr(llm_client, "request_structured", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("offline")))
    active = [{"slot_id": "s1", "task_id": "active", "concept": "energy",
               "task_operation": "define", "target": "definition"}]
    unsafe = {"slot_id": "s2", "task_id": "unused_bad", "concept": "energy",
              "task_operation": "define", "target": "definition"}
    safe = {"slot_id": "s2", "task_id": "unused_good", "concept": "power",
            "task_operation": "calculate", "target": "rate of work"}
    selected, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="s2", qtype="mcq", conflict_slots=["s1"], conflict_edges=[],
        candidates=[unsafe, safe], active_items=active, blacklist=set(), previous_failures=[],
    )
    assert selected["task_id"] == "unused_good"
    assert [item["task_id"] for item in context["candidates"]] == ["unused_good"]
    assert all("question" not in item for item in context["active_slots"])


def test_compact_selector_prefers_less_used_lower_conflict_candidate(monkeypatch):
    from app.llm import client as llm_client
    monkeypatch.setattr(llm_client, "request_structured", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("offline")))
    active = [
        {"slot_id": "s1", "task_id": "often", "concept": "a", "task_operation": "explain", "target": "x"},
        {"slot_id": "s3", "task_id": "often", "concept": "b", "task_operation": "interpret", "target": "y"},
        {"slot_id": "s4", "task_id": "rare", "concept": "c", "task_operation": "compare", "target": "z"},
    ]
    candidates = [
        {"slot_id": "s2", "task_id": "often", "concept": "d", "task_operation": "derive", "target": "u"},
        {"slot_id": "s2", "task_id": "rare", "concept": "e", "task_operation": "classify", "target": "v"},
    ]
    selected, _ = duplicate_pipeline._select_replacement_candidate(
        slot_id="s2", qtype="why", conflict_slots=[], conflict_edges=[], candidates=candidates,
        active_items=active, blacklist=set(), previous_failures=[],
    )
    assert selected["task_id"] == "rare"


def test_compact_selector_blacklist_and_previous_failure_reach_second_attempt(monkeypatch):
    from app.llm import client as llm_client
    captured = {}
    def fail_request(_client, prompt, **_kwargs):
        captured["prompt"] = prompt
        raise RuntimeError("offline")
    monkeypatch.setattr(llm_client, "request_structured", fail_request)
    candidates = [
        {"slot_id": "s2", "task_id": "failed", "concept": "a", "task_operation": "derive", "target": "x"},
        {"slot_id": "s2", "task_id": "next", "concept": "b", "task_operation": "compare", "target": "y"},
    ]
    selected, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="s2", qtype="equation", conflict_slots=[], conflict_edges=[], candidates=candidates,
        active_items=[], blacklist={"failed"},
        previous_failures=[{"candidate_id": "failed", "reason": "duplicate_after_replacement"}],
    )
    assert selected["task_id"] == "next"
    assert context["previous_failures"][0]["candidate_id"] == "failed"
    assert "duplicate_after_replacement" in captured["prompt"]


def test_global_budget_prevents_extra_duplicate_judge_call(monkeypatch):
    calls = []
    monkeypatch.setattr(duplicate_pipeline.config, "MAX_DUPLICATE_REPAIR_ROUNDS", 2)
    monkeypatch.setattr(duplicate_pipeline, "review_duplicate_candidates",
                        lambda *_a, **_k: calls.append(True))
    result = duplicate_pipeline.scan_exam_duplicates({
        "generated_exams": [], "duplicate_reports": [{"duplicate_groups": [{"group_id": "g"}]}],
        "duplicate_similarity_distributions": [], "duplicate_repair_rounds": 2,
        "duplicate_scan_count": 2, "invalid_questions": [], "warnings": [],
    })
    assert calls == []
    assert result["unresolved_duplicate_conflicts"] == [{"group_id": "g"}]
