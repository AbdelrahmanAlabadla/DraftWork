from __future__ import annotations

from app.online import graph, validator
from app.online.multi_repair import REPAIR_AGENTS, build_repair_route
from app.online.multi_validator import VALIDATOR_ROLES


def _exam() -> dict:
    return {"model_number": 1, "questions": {"true_false": [
        {"question_id": "q1", "slot_id": "m1_true_false_1", "statement": "Alpha", "answer": True},
        {"question_id": "q2", "slot_id": "m1_true_false_2", "statement": "Beta", "answer": False},
    ]}}


def _plans() -> list[dict]:
    return [{"model_number": 1, "items": {"true_false": [
        {"slot_id": "m1_true_false_1", "question_type": "true_false", "task_id": "t1",
         "source_chunk_id": "c1", "support_chunk_ids": []},
        {"slot_id": "m1_true_false_2", "question_type": "true_false", "task_id": "t2",
         "source_chunk_id": "c2", "support_chunk_ids": []},
    ]}}]


def test_contracts_expose_six_validators_and_seven_repairs():
    assert VALIDATOR_ROLES == (
        "grounding", "task_alignment", "answer_correctness", "difficulty", "quantitative", "duplicate",
    )
    assert REPAIR_AGENTS[-1] == "duplicate"
    assert len(REPAIR_AGENTS) == 7


def test_duplicate_issue_routes_to_duplicate_repair():
    verdict = {"required_fixes": [{
        "issue_id": "q2-I-duplicate-1", "validator": "duplicate",
        "failure_code": "duplicate_question", "fields_to_fix": [],
        "reason": "same assessed work", "expected_fix": "replace assignment",
    }]}
    route = build_repair_route("true_false", verdict)
    assert route["status"] == "READY"
    assert [step.agent for step in route["steps"]] == ["duplicate"]


def test_duplicate_group_becomes_stable_transactional_manifest_issue(monkeypatch):
    monkeypatch.setattr("app.online.duplicate_pipeline.scan_exam_duplicates", lambda state: {
        "pending_duplicate_groups": [{
            "model_number": 1, "group_id": "g1", "question_ids": ["q1", "q2"],
            "keep_question_ids": ["q1"], "replace_question_ids": ["q2"],
            "edges": [{"left": "q1", "right": "q2", "classification": "duplicate"}],
        }],
        "duplicate_reports": [], "duplicate_similarity_distributions": [],
        "duplicate_fact_cache": {}, "duplicate_scan_count": 1, "warnings": [],
    })
    reports = {1: {"model_number": 1, "all_pass": True, "verdicts": [
        {"question_id": "q1", "action": "PASS", "role_statuses": {}},
        {"question_id": "q2", "action": "PASS", "role_statuses": {}},
    ]}}
    manifest, validated, _warnings, _result = validator._integrate_duplicate_validation(
        {"retrieved_chunks": []}, [_exam()], _plans(), reports, [], [1], [],
    )
    assert validated == []
    assert len(manifest) == 1
    target = manifest[0]
    assert target["repair_type"] == "replace_assignment"
    assert target["validation_role"] == "duplicate"
    assert target["conflicting_question_ids"] == ["q1"]
    assert target["required_fixes"][0]["issue_id"].startswith("q2-I-")
    assert reports[1]["all_pass"] is False


def test_duplicate_provider_failure_blocks_acceptance_without_content_repair(monkeypatch):
    def unavailable(_state):
        raise TimeoutError("mock duplicate provider timeout")

    monkeypatch.setattr("app.online.duplicate_pipeline.scan_exam_duplicates", unavailable)
    reports = {1: {"model_number": 1, "all_pass": True, "verdicts": [
        {"question_id": "q1", "action": "PASS", "role_statuses": {"grounding": "PASS"}},
        {"question_id": "q2", "action": "PASS", "role_statuses": {"grounding": "PASS"}},
    ]}}
    manifest, validated, warnings, result = validator._integrate_duplicate_validation(
        {"retrieved_chunks": []}, [_exam()], _plans(), reports, [], [1], [],
    )
    assert validated == []
    assert result["operational_error"].startswith("Duplicate Validator operational failure")
    assert len(manifest) == 2
    assert {target["status"] for target in manifest} == {"validation_blocked"}
    assert {target["repair_type"] for target in manifest} == {"none"}
    assert {row["action"] for row in reports[1]["verdicts"]} == {"UNVALIDATED"}
    assert any("Duplicate Validator operational failure" in warning for warning in warnings)


def test_compiled_graph_has_no_standalone_duplicate_nodes():
    compiled = graph.build_exam_graph()
    node_names = set(compiled.get_graph().nodes)
    assert "scan_exam_duplicates" not in node_names
    assert "repair_duplicate_groups" not in node_names
