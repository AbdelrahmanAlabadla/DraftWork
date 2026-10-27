from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

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

    def generate_exams_node(state):
        calls.append("generate")
        generated_plan.extend(deepcopy(state["plans"]))
        second = state["plans"][0]["items"]["mcq"][1]
        assert second["slot_id"] == "m1_mcq_2"
        assert second["task_family_id"] == "linear-equations"
        assert second["source_chunk_id"] == "chunk-1"
        assert second["derived_task_id"] == "derived:task-two:easy"
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
        "receive", "retrieve", "catalog", "plan",
        "adjust", "generate",
        "validate-1", "question-repair", "validate-2", "duplicate-scan-1",
        "duplicate-repair", "validate-3", "duplicate-scan-2", "assemble",
    ]
    assert validation_calls == 3
    assert duplicate_scan_calls == 2
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
def test_resume_router_keeps_durable_plan_and_partial_questions():
    plans = [{"model_number": 1, "items": {"mcq": [
        {"slot_id": "m1_mcq_1"}, {"slot_id": "m1_mcq_2"},
    ]}}]
    partial = [{"model_number": 1, "questions": {"mcq": [
        {"slot_id": "m1_mcq_1", "question_id": "q1"},
    ]}}]
    assert graph._route_after_receive({
        "plans": plans, "generated_exams": partial,
        "retrieved_chunks": [{"child_id": "c1"}],
    }) == "repair_plan"
    complete = deepcopy(partial)
    complete[0]["questions"]["mcq"].append(
        {"slot_id": "m1_mcq_2", "question_id": "q2"}
    )
    assert graph._route_after_receive({
        "plans": plans, "generated_exams": complete,
        "retrieved_chunks": [{"child_id": "c1"}],
    }) == "validate_generated_questions"


def test_new_job_without_checkpoint_always_enters_fresh_retrieval_and_planning():
    assert graph._route_after_receive({
        "plans": [], "assessment_catalog": {}, "retrieved_chunks": [],
    }) == "retrieve_context_progress"
    assert graph._route_after_retrieval({
        "plans": [], "retrieved_chunks": [{"child_id": "current-c1"}],
    }) == "prepare_assessment_catalog"


def test_resume_with_unresolved_plan_errors_routes_to_repair_before_generation():
    state = {
        "plans": [{"model_number": 1, "items": {"true_false": [{
            "slot_id": "m1_true_false_1",
        }]}}],
        "plan_errors": [
            "Slot m1_true_false_1 uses an operation outside the profile."
        ],
        "retrieved_chunks": [{"child_id": "c1"}],
    }
    assert graph._route_after_receive(state) == "repair_plan"


def test_legacy_same_job_checkpoint_retrieves_evidence_without_replanning():
    plans = [{"model_number": 1, "items": {"mcq": [
        {"slot_id": "m1_mcq_1", "selected_variation_id": "variation-1"},
    ]}}]
    assert graph._route_after_receive({"plans": plans}) == "retrieve_context_progress"
    assert graph._route_after_retrieval({
        "plans": plans, "retrieved_chunks": [{"child_id": "c1"}],
    }) == "repair_plan"


def test_saved_103_chunk_catalog_mocked_graph_reaches_final_assembly(
    monkeypatch, tmp_path,
):
    """Run the production graph and planner with every paid call mocked."""
    root = Path(__file__).parents[1]
    source_catalog = root / "data" / "assessment_catalogs" / "33" / (
        "339d652657343eec2e1a6f5af7dcee91b9523cdd66d9e86f95bceb47171ae3e7.json"
    )
    # The saved live artifact is intentionally optional for clean CI clones.
    if not source_catalog.exists():
        pytest.skip("saved 103-chunk mathematics Catalog is unavailable")
    processed_key = "772d2717699ca1dc2b015371744fb5d9df8f50751dc7a80b8eb63d716697f84e"
    # The filesystem copy under data/processed_documents predates the current
    # Docker index and has different chunk IDs despite sharing this legacy key.
    # Read the authoritative, persisted child rows used by the saved Catalog.
    query = """
        SELECT json_agg(metadata ORDER BY chunk_order)::text
          FROM document_chunks
         WHERE document_id = '175ef217-a754-4e58-8bae-2bbb4841f824'
           AND chunk_type = 'child'
    """
    try:
        docker_result = subprocess.run(
            [
                "docker", "exec", "genexam-postgres", "psql", "-U", "postgres",
                "-d", "draftwork", "-At", "-c", query,
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Inside the production worker there is intentionally no Docker socket.
        # Read the same authoritative PostgreSQL rows through its configured
        # connection instead of skipping the deployment-level verification.
        from app import db
        with db.connection() as connection:
            row = connection.execute(query).fetchone()
        direct_value = (row or [None])[0]
        children = list(
            json.loads(direct_value) if isinstance(direct_value, str) else (direct_value or [])
        )
    else:
        children = list(json.loads(docker_result.stdout.strip() or "[]"))
    assert len(children) == 103
    catalog_payload = json.loads(source_catalog.read_text(encoding="utf-8"))
    assert {str(row["child_id"]) for row in children} == set(
        catalog_payload["selected_chunk_ids"]
    )

    cache_copy = tmp_path / "33" / source_catalog.name
    cache_copy.parent.mkdir(parents=True)
    cache_copy.write_bytes(source_catalog.read_bytes())
    calls: list[str] = []

    from app.online import assessment_catalog, difficulty_adjustment
    monkeypatch.setattr(
        assessment_catalog.config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path)
    )
    monkeypatch.setattr(
        assessment_catalog, "request_structured",
        lambda *_args, **_kwargs: pytest.fail("Catalog cache reuse called a provider"),
    )
    monkeypatch.setattr(
        difficulty_adjustment, "create_difficulty_adjustment_client",
        lambda **_kwargs: object(),
    )

    def decline_adjustment(_client, prompt, **_kwargs):
        payload = json.loads(prompt)
        results = []
        for row in payload["tasks"]:
            task = row["task"]
            results.append({
                "slot_id": task["immutable_identifiers"]["slot_id"],
                "adjustment_succeeded": False,
                "achieved_difficulty": task["requested_difficulty"],
                "adjusted_fields": None,
                "evidence_support": [],
                "evidence_justification": "Mocked decline; deterministic fallback required.",
                "reason": "Mocked model cannot safely change the source task.",
            })
        return {"results": results}

    monkeypatch.setattr(difficulty_adjustment, "request_structured", decline_adjustment)
    monkeypatch.setattr(
        "app.online.receive.receive_request", lambda _state: calls.append("receive") or {}
    )
    monkeypatch.setattr(
        "app.online.retrieval.retrieve_context",
        lambda _state: calls.append("retrieve") or {
            "retrieved_chunks": children,
            "context": "",
            "planner_context": "",
        },
    )

    original_prepare = assessment_catalog.prepare_assessment_catalog
    monkeypatch.setattr(
        "app.online.assessment_catalog.prepare_assessment_catalog",
        lambda state: calls.append("catalog") or original_prepare(state),
    )

    def generate(state):
        calls.append("generate")
        sections = {}
        for qtype, values in state["plans"][0]["items"].items():
            sections[qtype] = [{
                "slot_id": item["slot_id"],
                "question_id": f"q_{item['slot_id']}",
                "question": f"Mocked grounded {qtype} question for {item['slot_id']}",
            } for item in values]
        assert sum(map(len, sections.values())) == 23
        return {"generated_exams": [{"model_number": 1, "questions": sections}]}

    def validate(state):
        calls.append("validate")
        verdicts = [{
            "question_id": item["question_id"], "action": "PASS",
        } for section in state["generated_exams"][0]["questions"].values()
          for item in section]
        return {"validation_reports": [{"verdicts": verdicts}], "repair_manifest": []}

    monkeypatch.setattr("app.online.exam_builder.generate_exams_node", generate)
    monkeypatch.setattr("app.online.validator.validate_generated_questions", validate)
    monkeypatch.setattr(
        "app.online.validator.repair_invalid_questions",
        lambda _state: pytest.fail("all mocked questions pass validation"),
    )
    monkeypatch.setattr(
        "app.online.duplicate_pipeline.scan_exam_duplicates",
        lambda _state: calls.append("duplicates") or {"pending_duplicate_groups": []},
    )
    monkeypatch.setattr(
        "app.online.duplicate_pipeline.repair_duplicate_groups",
        lambda _state: pytest.fail("mocked exam has no duplicate group"),
    )
    monkeypatch.setattr(
        "app.online.exam_builder.assemble_exams_node",
        lambda state: calls.append("assemble") or {
            "complete": True,
            "warnings": ["mocked final assembly persisted 23 of 23 questions"],
        },
    )

    tasks = [("mcq", 10), ("true_false", 5), ("equation", 4), ("word_problem", 4)]
    result = graph.build_exam_graph().invoke({
        "document_id": "saved-math-document", "processed_document_key": processed_key,
        "tasks": tasks, "num_models": 1, "difficulty": "easy",
        "selected_child_ids": [str(row["child_id"]) for row in reversed(children)],
        "domain_type": "stem", "subject": "mathematics", "document_language": "ar",
        "slot_lifecycle": initialize_slot_lifecycle(tasks, 1),
    }, {"recursion_limit": 60})

    assert result["assessment_catalog_cache_hit"] is True
    assert result["assessment_catalog_compatibility_cache_hit"] is True
    assert result["plan_errors"] == []
    assert result["warnings"] == ["mocked final assembly persisted 23 of 23 questions"]
    assert sum(
        len(section)
        for section in result["generated_exams"][0]["questions"].values()
    ) == 23
    assert calls[-4:] == ["generate", "validate", "duplicates", "assemble"]
