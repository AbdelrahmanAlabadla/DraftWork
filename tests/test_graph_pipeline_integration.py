from __future__ import annotations

from copy import deepcopy

from app.online import graph
from app.online.lifecycle import initialize_slot_lifecycle


def _plan_item(slot_id: str, task_id: str) -> dict:
    return {
        "slot_id": slot_id,
        "task_id": task_id,
        "base_catalog_task_id": task_id,
        "task_family_id": "linear-equations",
        "base_task_family_id": "linear-equations",
        "question_type": "mcq",
        "concept": "linear equations",
        "task_operation": "solve_equation",
        "target": "x",
        "student_work_product": "solve for x",
        "reasoning_mode": "symbolic_manipulation",
        "reasoning_requirements": ["Apply inverse operations."],
        "prerequisites": ["addition and subtraction"],
        "requested_difficulty": "easy",
        "difficulty": "medium",
        "actual_difficulty": "medium",
        "reasoning_depth": "medium",
        "source_chunk_id": "chunk-1",
        "support_chunk_ids": ["chunk-1"],
        "difficulty_adaptation": {
            "status": "pending", "from_difficulty": "medium", "to_difficulty": "easy",
        },
    }


def test_state_graph_preserves_contracts_through_recovery_lifecycles(monkeypatch):
    """Exercise the compiled StateGraph, including every bounded recovery route.

    The controlled nodes stand in for paid LLM calls; routing, state merging,
    lifecycle reconciliation, and every graph transition are production code.
    """
    calls: list[str] = []
    review_inputs: list[dict] = []
    generated_plan: list[dict] = []
    audit_events: list[tuple[str, str, dict]] = []
    validation_calls = 0
    duplicate_scan_calls = 0

    def receive_request(state):
        calls.append("receive")
        return {}

    def retrieve_context(_state):
        calls.append("retrieve")
        return {"retrieved_chunks": [{"child_id": "chunk-1", "content": "Solve linear equations."}]}

    def prepare_assessment_catalog(_state):
        calls.append("catalog")
        return {
            "assessment_catalog": {"tasks": []},
            "assessment_profile": {"domain": "STEM"},
            "planning_issues": [],
        }

    def plan_exams(_state):
        calls.append("plan")
        return {"plans": [{"model_number": 1, "items": {"mcq": [
            _plan_item("m1_mcq_1", "task-one"), _plan_item("m1_mcq_2", "task-two"),
        ]}}], "plan_errors": [], "plan_attempts": 1}

    def repair_plan(_state):  # pragma: no cover - the scenario must avoid structural repair.
        raise AssertionError("A pending difficulty adjustment is not a planner-repair error")

    def adjust_plan_difficulty(state):
        calls.append("adjust")
        adjusted = deepcopy(state["plans"])
        for item in adjusted[0]["items"]["mcq"]:
            item.update({
                "difficulty": "easy", "actual_difficulty": "easy", "reasoning_depth": "easy",
                "reasoning_requirements": ["Use one inverse operation."],
                "derived_task_id": f"derived:{item['base_catalog_task_id']}:easy",
                "difficulty_adaptation": {"status": "accepted", "from_difficulty": "medium", "to_difficulty": "easy"},
            })
        return {
            "plans": adjusted,
            "difficulty_adjustment_attempts": {"m1_mcq_1": 1, "m1_mcq_2": 1},
            "difficulty_adjustment_history": [{"status": "accepted", "slot_id": "m1_mcq_1"}],
        }

    def review_plan(state):
        attempt = len(review_inputs) + 1
        calls.append(f"review-{attempt}")
        items = deepcopy(state["plans"][0]["items"]["mcq"])
        review_inputs.append({"items": deepcopy(items), "attempt": attempt})
        repaired = items[1]
        repaired.update({
            "plan_duplicate_repair_attempt": min(attempt, 3),
            "plan_duplicate_conflict_with": "m1_mcq_1",
            "plan_duplicate_conflict_reason": "same operation and scenario",
        })
        if attempt <= 2:
            repaired.update({
                "plan_duplicate_repair_mode": "targeted_semantic_repair",
                "idea_to_test": f"distinct supported strategy {attempt}",
            })
            retry = True
            unresolved = []
        elif attempt == 3:
            repaired.update({
                "plan_duplicate_repair_mode": "controlled_diversification",
                "idea_to_test": "different supported equation representation",
            })
            retry = True
            unresolved = []
        else:
            repaired["plan_duplicate_unresolved"] = True
            retry = False
            unresolved = [{
                "slot_id": "m1_mcq_2", "conflict_with": "m1_mcq_1",
                "reason": "same operation and scenario", "status": "plan_duplicate_unresolved_kept",
            }]
        return {
            "plans": [{"model_number": 1, "items": {"mcq": items}}],
            "plan_review_retry_required": retry,
            "plan_duplicate_repair_attempts": {"m1_mcq_2": min(attempt, 3)},
            "unresolved_plan_duplicate_conflicts": unresolved,
        }

    def generate_exams_node(state):
        calls.append("generate")
        generated_plan.extend(deepcopy(state["plans"]))
        second = state["plans"][0]["items"]["mcq"][1]
        assert second["slot_id"] == "m1_mcq_2"
        assert second["task_family_id"] == "linear-equations"
        assert second["source_chunk_id"] == "chunk-1"
        assert second["derived_task_id"] == "derived:task-two:easy"
        assert second["plan_duplicate_unresolved"] is True
        return {"generated_exams": [{"model_number": 1, "questions": {"mcq": {"items": [
            {"slot_id": "m1_mcq_1", "question_id": "q-1", "question": "Solve x + 2 = 5."},
            {"slot_id": "m1_mcq_2", "question_id": "q-2", "question": "Solve x - 4 = 3."},
        ]}}}]}

    def validate_generated_questions(state):
        nonlocal validation_calls
        validation_calls += 1
        calls.append(f"validate-{validation_calls}")
        if validation_calls == 1:
            return {
                "validation_reports": [{"verdicts": [
                    {"question_id": "q-1", "action": "PASS"},
                    {"question_id": "q-2", "action": "REPAIR", "failure_code": "needs_precision"},
                ]}],
                "repair_manifest": [{"slot_id": "m1_mcq_2", "status": "pending", "failure_code": "needs_precision"}],
            }
        return {
            "validation_reports": [{"verdicts": [
                {"question_id": "q-1", "action": "PASS"}, {"question_id": "q-2", "action": "PASS"},
            ]}],
            "repair_manifest": [{"slot_id": "m1_mcq_2", "status": "resolved"}],
            "pending_revalidation_ids": {},
        }

    def repair_invalid_questions(state):
        calls.append("question-repair")
        assert state["generated_exams"][0]["questions"]["mcq"]["items"][1]["question_id"] == "q-2"
        return {
            "question_repair_attempts": {"m1_mcq_2": 1},
            "pending_revalidation_ids": {1: ["q-2"]},
            "validation_repair_history": [{"slot_id": "m1_mcq_2", "status": "repaired"}],
        }

    def scan_exam_duplicates(state):
        nonlocal duplicate_scan_calls
        duplicate_scan_calls += 1
        calls.append(f"duplicate-scan-{duplicate_scan_calls}")
        if duplicate_scan_calls == 1:
            return {"pending_duplicate_groups": [{"question_ids": ["q-1", "q-2"]}]}
        assert state["duplicate_verification_pending"] is True
        return {
            "pending_duplicate_groups": [], "duplicate_verification_pending": False,
            "duplicate_verification_attempt": 1,
        }

    def repair_duplicate_groups(state):
        calls.append("duplicate-repair")
        assert state["pending_duplicate_groups"][0]["question_ids"] == ["q-1", "q-2"]
        return {
            "pending_duplicate_groups": [], "pending_revalidation_ids": {1: ["q-2"]},
            "duplicate_verification_pending": True, "duplicate_verification_attempt": 1,
            "duplicate_replacement_attempts": {"1:m1_mcq_2": 1},
            "duplicate_repair_history": [{"slot_id": "m1_mcq_2", "status": "replaced"}],
        }

    def assemble_exams_node(state):
        calls.append("assemble")
        assert len(state["generated_exams"][0]["questions"]["mcq"]["items"]) == 2
        return {"warnings": ["controlled assembly completed with two valid questions"]}

    replacements = {
        "app.online.receive.receive_request": receive_request,
        "app.online.retrieval.retrieve_context": retrieve_context,
        "app.online.assessment_catalog.prepare_assessment_catalog": prepare_assessment_catalog,
        "app.online.planner.plan_exams": plan_exams,
        "app.online.planner.repair_plan": repair_plan,
        "app.online.difficulty_adjustment.adjust_plan_difficulty": adjust_plan_difficulty,
        "app.online.plan_review.review_plan": review_plan,
        "app.online.exam_builder.generate_exams_node": generate_exams_node,
        "app.online.exam_builder.assemble_exams_node": assemble_exams_node,
        "app.online.validator.validate_generated_questions": validate_generated_questions,
        "app.online.validator.repair_invalid_questions": repair_invalid_questions,
        "app.online.duplicate_pipeline.scan_exam_duplicates": scan_exam_duplicates,
        "app.online.duplicate_pipeline.repair_duplicate_groups": repair_duplicate_groups,
    }
    for target, replacement in replacements.items():
        monkeypatch.setattr(target, replacement)

    result = graph.build_exam_graph().invoke({
        "document_id": "controlled-doc", "tasks": [("mcq", 2)], "num_models": 1,
        "difficulty": "easy", "selected_child_ids": ["chunk-1"], "domain_type": "STEM",
        "subject": "Mathematics", "document_language": "en",
        "slot_lifecycle": initialize_slot_lifecycle([("mcq", 2)], 1),
        "audit_callback": lambda stage, event, payload: audit_events.append((stage, event, payload)),
    }, {"recursion_limit": 40})

    assert calls == [
        "receive", "retrieve", "catalog", "plan", "adjust",
        "review-1", "review-2", "review-3", "review-4", "generate",
        "validate-1", "question-repair", "validate-2", "duplicate-scan-1",
        "duplicate-repair", "validate-3", "duplicate-scan-2", "assemble",
    ]
    assert validation_calls == 3
    assert duplicate_scan_calls == 2
    assert len(review_inputs) == 4
    assert all(len(entry["items"]) == 2 for entry in review_inputs)
    assert result["warnings"] == ["controlled assembly completed with two valid questions"]
    finalization = next(payload for stage, event, payload in audit_events if event == "exam_assembled")
    final_slots = finalization["node_output"]["slot_lifecycle"]
    assert final_slots["m1_mcq_1"]["finalized"] is True
    assert final_slots["m1_mcq_2"]["finalized"] is True
    assert final_slots["m1_mcq_2"]["question_id"] == "q-2"
    final_second = finalization["plans_after_stage"][0]["items"]["mcq"][1]
    assert final_second["slot_id"] == "m1_mcq_2"
    assert final_second["derived_task_id"] == "derived:task-two:easy"
    assert final_second["actual_difficulty"] == "easy"
    assert final_second["reasoning_requirements"] == ["Use one inverse operation."]
    assert finalization["generated_exams_after_stage"][0]["questions"]["mcq"]["items"][1]["question_id"] == "q-2"
    assert generated_plan[0]["items"]["mcq"][0]["base_catalog_task_id"] == "task-one"
