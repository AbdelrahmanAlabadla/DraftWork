from __future__ import annotations

import importlib.util
import inspect
from contextlib import contextmanager
from pathlib import Path

import pytest


@pytest.fixture()
def runner_module():
    path = Path(__file__).parents[1] / "scripts" / "live_pipeline_stage_test.py"
    spec = importlib.util.spec_from_file_location("live_pipeline_stage_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_state(module):
    return {
        "run_id": "stage-test-unit",
        "document_id": module.DEFAULT_DOCUMENT_ID,
        "tasks": [("mcq", 1)],
        "num_models": 1,
        "difficulty": "mix",
        "selected_child_ids": ["child-1"],
        "immutable_request": {"document_id": module.DEFAULT_DOCUMENT_ID},
        "request_fingerprint": "fingerprint",
        "plans": [],
        "assessment_catalog": {},
        "assessment_catalog_key": "",
        "generated_exams": [],
        "warnings": [],
        "plan_errors": [],
        "planning_issues": [],
        "difficulty_adjustment_history": [],
        "difficulty_adjustment_attempts": {},
        "slot_lifecycle": {},
        "eval_stats": {},
        "is_cancelled": lambda: False,
    }


def test_runner_reuses_production_nodes_without_changing_production_graph(runner_module):
    from app.online import exam_builder, graph, retrieval

    assert runner_module.retrieve_context is retrieval.retrieve_context
    assert runner_module.generate_exams_node is exam_builder.generate_exams_node
    assert 'builder.add_edge("generate_exams", "validate_generated_questions")' in inspect.getsource(graph.build_exam_graph)


def test_generation_only_uses_production_node_hooks_and_stops_before_validation(monkeypatch, tmp_path, runner_module):
    module = runner_module
    calls = []
    state = _base_state(module)
    monkeypatch.setattr(module, "_document_state", lambda *args, **kwargs: state)

    def retrieval(current):
        calls.append("retrieval")
        return {"retrieved_chunks": [{"child_id": "child-1", "content": "x"}], "context": "x", "planner_context": "x", "document_language": "ar"}

    def catalog(current):
        calls.append("catalog")
        return {"assessment_catalog": {"cache_key": "catalog"}, "assessment_catalog_key": "catalog", "assessment_catalog_cache_hit": True, "assessment_compiler_cache_hit": True}

    def planner(current):
        calls.append("planning")
        return {"plans": [{"model_number": 1, "items": {"mcq": [{"slot_id": "m1_mcq_1", "task_id": "task-1"}]}}], "plan_errors": [], "assessment_profile": {}, "planning_issues": [], "plan_attempts": 1}

    def adjustment(current):
        calls.append("difficulty")
        return {"plans": current["plans"], "difficulty_adjustment_history": [], "difficulty_adjustment_attempts": {}}

    def generation(current):
        calls.append("generation")
        current["audit_callback"]("generation", "generation_checkpoint", {"checkpoint_state": {"generated_exams": [{"model_number": 1}]}})
        return {"generated_exams": [{"model_number": 1, "questions": {"mcq": {"items": [{"slot_id": "m1_mcq_1", "question": "سؤال", "correct_answer": "أ"}]}}}]}

    monkeypatch.setattr(module, "retrieve_context", retrieval)
    monkeypatch.setattr(module, "prepare_assessment_catalog", catalog)
    monkeypatch.setattr(module, "plan_exams", planner)
    monkeypatch.setattr(module, "adjust_plan_difficulty", adjustment)
    monkeypatch.setattr(module, "generate_exams_node", generation)
    monkeypatch.setattr(module, "repair_plan", lambda _state: pytest.fail("repair must not run for a valid plan"))

    checkpoint = module.run_generation_only(tasks=[("mcq", 1)], checkpoint_root=tmp_path)
    payload = module.load_generation_checkpoint(checkpoint)

    assert calls == ["retrieval", "catalog", "planning", "difficulty", "generation"]
    assert payload["status"] == "generation_complete_awaiting_validation"
    assert payload["state"]["generated_exams"][0]["questions"]["mcq"]["items"][0]["question"] == "سؤال"
    assert payload["generation_audit_events"][0]["event_type"] == "generation_checkpoint"


def test_generation_only_refuses_catalog_cache_miss_before_planning_or_generation(monkeypatch, tmp_path, runner_module):
    module = runner_module
    state = _base_state(module)
    monkeypatch.setattr(module, "_document_state", lambda *args, **kwargs: state)
    monkeypatch.setattr(module, "retrieve_context", lambda _state: {"retrieved_chunks": [{"child_id": "child-1"}]})
    monkeypatch.setattr(module, "prepare_assessment_catalog", lambda _state: {"assessment_catalog": {}, "assessment_catalog_cache_hit": False})
    monkeypatch.setattr(module, "plan_exams", lambda _state: pytest.fail("planner must not run"))
    monkeypatch.setattr(module, "generate_exams_node", lambda _state: pytest.fail("generator must not run"))

    with pytest.raises(module.StageTestError, match="catalog was not a cache hit"):
        module.run_generation_only(tasks=[("mcq", 1)], checkpoint_root=tmp_path)


def test_structured_request_capture_records_prompt_response_and_usage(monkeypatch, runner_module):
    module = runner_module
    import app.online.exam_builder as exam_builder

    class Client:
        provider_name = "test-provider"
        model = "test-model"
        last_usage = {"input_tokens": 12, "output_tokens": 4}
        last_transient_retries = 1
        last_structured_attempts = 2
        last_request_id = "request-123"

    monkeypatch.setattr(
        exam_builder,
        "request_structured",
        lambda _client, _prompt, **_kwargs: {"question_id": "q1", "question": "سؤال"},
    )
    records = []
    with module._capture_structured_requests(records):
        result = exam_builder.request_structured(
            Client(),
            "Generate one question",
            agent="generator",
            operation="question_generation",
            schema_name="Question",
            system_prompt="System instruction",
            max_tokens=100,
        )

    assert result["question_id"] == "q1"
    assert len(records) == 1
    assert records[0]["provider"] == "test-provider"
    assert records[0]["model"] == "test-model"
    assert records[0]["user_prompt"] == "Generate one question"
    assert records[0]["parsed_response"]["question"] == "سؤال"
    assert records[0]["usage"]["input_tokens"] == 12


def test_runner_applies_and_restores_production_domain_context(runner_module):
    from app.llm.factory import _domain

    assert _domain() == ""
    with runner_module._production_domain({"assessment_profile": {"domain": "STEM"}}):
        assert _domain() == "STEM"
    assert _domain() == ""


def test_runner_isolates_and_restores_llm_database_telemetry(monkeypatch, runner_module):
    from app.api import repositories

    calls = []
    monkeypatch.setattr(repositories, "record_llm_usage", lambda *args, **kwargs: calls.append((args, kwargs)))
    original = repositories.record_llm_usage
    with runner_module._isolate_llm_telemetry():
        assert repositories.record_llm_usage("would", "persist") is None
        assert repositories.record_llm_request_event("would", "persist") is None
    assert calls == []
    assert repositories.record_llm_usage is original


def test_validation_resume_uses_checkpoint_without_generation_repair_or_duplicate(monkeypatch, tmp_path, runner_module):
    module = runner_module
    state = _base_state(module)
    state.update({
        "plans": [{"model_number": 1, "items": {"mcq": [{"slot_id": "m1_mcq_1", "task_id": "task-1"}]}}],
        "generated_exams": [{"model_number": 1, "questions": {"mcq": {"items": [{"slot_id": "m1_mcq_1", "question": "سؤال"}]}}}],
    })
    checkpoint = module._save_generation_checkpoint(state, tmp_path, [], [])

    import app.online.multi_validator as multi_validator
    import app.online.validator as validator

    called = []
    monkeypatch.setattr(validator, "_plan_context_for_model", lambda _state, _model: ({"m1_mcq_1": {"slot_id": "m1_mcq_1"}}, {"child-1": "x"}))
    monkeypatch.setattr(validator, "_manifest_from_report", lambda _report, _exam, _plans: [])
    monkeypatch.setattr(multi_validator, "validate_exam_multi", lambda exam, **_kwargs: called.append(exam) or {"model_number": 1, "all_pass": False, "verdicts": [], "validation_call_logs": []})
    monkeypatch.setattr(module, "generate_exams_node", lambda _state: pytest.fail("generation must not run during validation resume"))

    output = module.run_validation_only(checkpoint, checkpoint_root=tmp_path)
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert output.exists()
    assert len(called) == 1
    assert called == state["generated_exams"]
    assert payload["status"] == "validation_complete_awaiting_repair"
    assert payload["state"]["duplicate_scan_count"] == 0


def test_validation_forensic_report_is_offline_and_preserves_request_evidence(tmp_path, runner_module):
    module = runner_module
    source = tmp_path / "generation_checkpoint.json"
    source.write_text("{}", encoding="utf-8")
    state = _base_state(module)
    state.update({
        "generated_exams": [{"model_number": 1, "questions": {"true_false": [{
            "question_id": "model1_true_false_1", "statement": "قيود غير معروضة", "answer": True,
        }]}}],
        "validation_reports": [{"model_number": 1, "verdicts": [{
            "question_id": "model1_true_false_1", "validation_status": "FAIL", "action": "FIX_QUESTION",
            "role_statuses": {"task_alignment": "FAIL"}, "reason": "missing",
            "required_fixes": [{
                "issue_id": "issue-1", "validator": "task_alignment", "failure_code": "missing_required_given",
                "category": "missing_constraint", "fields_to_fix": ["statement"],
                "missing_information": ["7 ≤ c ≤ 14"], "source_evidence_ids": ["child-1"],
                "expected_fix": "add constraint",
            }],
        }]}],
    })
    checkpoint_state = module._checkpoint_state(state)
    checkpoint = tmp_path / "validation_checkpoint.json"
    module._atomic_json_write(checkpoint, {
        "checkpoint_version": module.CHECKPOINT_VERSION,
        "mode": "validation_only",
        "status": "validation_complete_awaiting_repair",
        "source_generation_checkpoint": str(source),
        "source_generation_state_sha256": "source-state",
        "state_sha256": __import__("hashlib").sha256(
            __import__("json").dumps(checkpoint_state, ensure_ascii=False, sort_keys=True, default=module._json_default).encode("utf-8")
        ).hexdigest(),
        "state": checkpoint_state,
        "validation_requests": [{
            "agent": "task_alignment_validator", "operation": "validate_batch", "provider": "openai", "model": "test",
            "status": "success", "duration_seconds": 1.25, "system_prompt": "system", "user_prompt": "prompt",
            "parsed_response": {"validator": "task_alignment"},
            "usage": {"input_tokens": 10, "output_tokens": 2, "cached_tokens": 0, "reasoning_tokens": 1, "total_tokens": 12, "estimated_cost_usd": 0.000001},
            "structured_attempt_details": [],
        }],
    })

    markdown_path, json_path = module.write_validation_forensic_report(checkpoint)

    assert markdown_path.exists() and json_path.exists()
    report = __import__("json").loads(json_path.read_text(encoding="utf-8"))
    assert report["request_totals"]["request_count"] == 1
    assert report["request_totals"]["estimated_cost_usd"] == 0.000001
    assert report["validation_requests"][0]["user_prompt"] == "prompt"
    assert "missing_constraint" in markdown_path.read_text(encoding="utf-8")


def test_grounding_only_uses_one_grounding_call_and_preserves_source_checkpoints(monkeypatch, tmp_path, runner_module):
    module = runner_module
    state = _base_state(module)
    state.update({
        "assessment_profile": {"domain": "STEM"},
        "plans": [{"model_number": 1, "items": {"mcq": [{
            "slot_id": "m1_mcq_1", "task_id": "task_0f0a6376b5e3c24b",
            "task_operation": "optimize_linear_function_at_vertices",
            "target": "find maximum or minimum", "selected_variation_id": "variation_13007c8931b963a0",
            "permitted_variations": [{"variation_id": "variation_13007c8931b963a0",
                "operation": "optimize_linear_function_at_vertices", "target": "find maximum or minimum",
                "numeric_values_may_change": True}], "exact_source_instance": False,
            "source_chunk_id": "child-1",
        }]}}],
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [{
            "question_id": "model1_mcq_1", "slot_id": "m1_mcq_1", "question": "Q",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct_answer": "A",
        }]}}],
        "retrieved_chunks": [{"child_id": "child-1", "content": "source"}],
    })
    generation = module._save_generation_checkpoint(state, tmp_path, [], [])
    validation_state = module._checkpoint_state(state)
    validation = tmp_path / "validation_checkpoint.json"
    module._atomic_json_write(validation, {
        "mode": "validation_only", "state": validation_state,
        "state_sha256": __import__("hashlib").sha256(
            __import__("json").dumps(validation_state, ensure_ascii=False, sort_keys=True, default=module._json_default).encode("utf-8")
        ).hexdigest(),
    })

    import app.online.multi_validator as multi_validator
    import app.online.validator as validator
    calls = []
    monkeypatch.setattr(validator, "_plan_context_for_model", lambda _state, _model: (
        {"m1_mcq_1": state["plans"][0]["items"]["mcq"][0]}, {"child-1": "source"},
    ))
    monkeypatch.setattr(multi_validator, "_call_once", lambda role, items, _factory, **_kwargs: (
        calls.append((role, [item.question_id for item in items])) or ({"model1_mcq_1": {
            "question_id": "model1_mcq_1", "status": "PASS", "issues": [], "grounding_outcome": "SUPPORTED",
        }}, {}, {"validator": role})
    ))

    @contextmanager
    def capture(records):
        try:
            yield
        finally:
            records.append({"agent": "grounding_validator", "provider": "openai", "model": "test", "status": "success", "usage": {}})

    monkeypatch.setattr(module, "_capture_structured_requests", capture)
    output = module.run_grounding_only(generation, validation_checkpoint_path=validation, checkpoint_root=tmp_path)
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))

    assert calls == [("grounding", ["model1_mcq_1"])]
    assert payload["status"] == "grounding_complete_awaiting_review"
    assert payload["grounding_verdict"]["status"] == "PASS"
    assert payload["grounding_contract"]["permitted_optimization_direction"] == "both"
    assert generation.exists() and validation.exists()


def test_one_pass_repair_then_revalidation_hard_stops_without_repair_loop(monkeypatch, tmp_path, runner_module):
    module = runner_module
    state = _base_state(module)
    task = {
        "slot_id": "m1_true_false_1", "task_id": "task_e22b32e17c016bb2",
        "selected_variation_id": "variation_694f53999424bd42", "source_chunk_id": "child-1",
    }
    original = {
        "question_id": "model1_true_false_1", "slot_id": "m1_true_false_1",
        "statement": "وفق القيود", "answer": True,
    }
    state.update({
        "assessment_profile": {"domain": "STEM"},
        "plans": [{"model_number": 1, "items": {"true_false": [task]}}],
        "generated_exams": [{"model_number": 1, "questions": {"true_false": [original]}}],
        "retrieved_chunks": [{"child_id": "child-1", "content": "source"}],
    })
    generation = module._save_generation_checkpoint(state, tmp_path, [], [])
    findings = [{
        "issue_id": issue_id, "validator": validator, "failure_code": "missing_required_given",
        "fields_to_fix": ["statement"], "reason": "missing", "expected_fix": "add", "category": category,
        "missing_information": ["constraint"], "source_evidence_ids": ["child-1"],
    } for issue_id, validator, category in (
        ("model1_true_false_1-I-08d66e25f28d-1", "quantitative", "missing_constraint"),
        ("model1_true_false_1-I-44eb50c40f9c-1", "task_alignment", "missing_context"),
    )]
    validation_state = module._checkpoint_state(state)
    validation_state["validation_reports"] = [{"verdicts": [{
        "question_id": "model1_true_false_1", "validation_status": "FAIL", "required_fixes": findings,
    }]}]
    validation = tmp_path / "validation_checkpoint.json"
    module._atomic_json_write(validation, {
        "mode": "validation_only", "state": validation_state,
        "state_sha256": __import__("hashlib").sha256(
            __import__("json").dumps(validation_state, ensure_ascii=False, sort_keys=True, default=module._json_default).encode("utf-8")
        ).hexdigest(),
    })

    import app.online.multi_repair as multi_repair
    import app.online.multi_validator as multi_validator
    import app.online.validator as validator_module
    monkeypatch.setattr(validator_module, "_plan_context_for_model", lambda _state, _model: ({"m1_true_false_1": task}, {"child-1": "source"}))
    captured = []
    repair_calls = []
    revalidation_calls = []

    def fake_repair(*, exam, **_kwargs):
        repair_calls.append(True)
        exam["questions"]["true_false"][0]["statement"] = "القيود 7≤c≤14 و8≤p≤12 وc+p≤23"
        captured.append({"agent": "task_alignment_repairer", "usage": {"input_tokens": 1}})
        return {"repaired_ids": ["model1_true_false_1"], "failed_ids": [], "repair_route": ["task_alignment"],
                "route_status": "COMPLETE", "repair_metadata": {"model1_true_false_1": {"modified_fields": ["statement"]}}}

    def fake_revalidation(*_args, **kwargs):
        revalidation_calls.append(kwargs["question_ids"])
        captured.append({"agent": "task_alignment_validator", "usage": {"input_tokens": 1}})
        return {"verdicts": [{"question_id": "model1_true_false_1", "action": "PASS", "validation_status": "PASS"}]}

    @contextmanager
    def capture(records):
        records.extend(captured)
        before = len(records)
        try:
            yield
        finally:
            records.extend(captured[before:])

    monkeypatch.setattr(multi_repair, "repair_question_specialized", fake_repair)
    monkeypatch.setattr(multi_validator, "validate_exam_multi", fake_revalidation)
    monkeypatch.setattr(module, "_capture_structured_requests", capture)
    repair_path, revalidation_path = module.run_repair_revalidate_once(
        generation, validation_checkpoint_path=validation, checkpoint_root=tmp_path,
    )

    assert repair_calls == [True]
    assert revalidation_calls == [["model1_true_false_1"]]
    assert revalidation_path is not None and revalidation_path.exists()
    result = __import__("json").loads(revalidation_path.read_text(encoding="utf-8"))
    assert result["execution"]["hard_stop_after_revalidation"] is True
    assert result["repaired_question"]["statement"].startswith("القيود")
    assert generation.exists() and validation.exists() and repair_path.exists()
