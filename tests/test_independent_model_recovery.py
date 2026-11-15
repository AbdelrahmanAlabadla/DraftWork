from __future__ import annotations

import threading
import time

from app.online import independent_recovery


def _exam(model: int) -> dict:
    return {"model_number": model, "questions": {"mcq": [{
        "slot_id": f"m{model}_mcq_1", "question_id": f"m{model}_q1",
        "question": f"Question {model}?", "correct_answer": "A",
    }]}}


def _plan(model: int) -> dict:
    return {"model_number": model, "items": {"mcq": [{
        "slot_id": f"m{model}_mcq_1", "task_id": f"task_{model}",
        "selected_variation_id": f"var_{model}", "source_chunk_id": f"c{model}",
        "support_chunk_ids": [],
    }]}}


def test_models_run_isolated_local_lifecycles_and_cross_model_runs_after(monkeypatch):
    calls: list[tuple[str, int]] = []
    active = peak = 0
    lock = threading.Lock()

    def validate(state):
        nonlocal active, peak
        model = int(state["generated_exams"][0]["model_number"])
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append(("validate", model))
        time.sleep(0.05)
        with lock:
            active -= 1
        if model == 2 and not any(value[0] == "repair" and value[1] == 2 for value in calls):
            return {
                "generated_exams": state["generated_exams"],
                "plans": state["plans"],
                "validation_reports": [{"model_number": 2, "verdicts": []}],
                "repair_manifest": [{
                    "model_number": 2, "slot_id": "m2_mcq_1", "status": "invalid",
                    "failure_code": "needs_repair",
                }],
                "validated_models": [],
            }
        return {
            "generated_exams": state["generated_exams"],
            "plans": state["plans"],
            "validation_reports": [{"model_number": model, "verdicts": []}],
            "repair_manifest": [],
            "validated_models": [model],
        }

    def repair(state):
        model = int(state["generated_exams"][0]["model_number"])
        calls.append(("repair", model))
        return {
            "generated_exams": state["generated_exams"],
            "plans": state["plans"],
            "repair_manifest": [{
                "model_number": model, "slot_id": f"m{model}_mcq_1",
                "status": "candidate_generated",
            }],
        }

    cross_calls: list[int] = []

    def cross(state, generated, reports, manifest, validated, warnings):
        cross_calls.append(len(generated))
        return manifest, validated, warnings, {"cross_model_duplicate_scan_complete": True}

    monkeypatch.setattr("app.online.validator.validate_generated_questions", validate)
    monkeypatch.setattr("app.online.validator.repair_invalid_questions", repair)
    monkeypatch.setattr(
        "app.online.cross_model_duplicates.integrate_cross_model_duplicates", cross,
    )

    state = {
        "num_models": 2,
        "generated_exams": [_exam(1), _exam(2)],
        "plans": [_plan(1), _plan(2)],
        "validation_reports": [],
        "repair_manifest": [],
        "repair_candidates": [],
        "validated_models": [],
        "warnings": [],
    }
    result = independent_recovery.run_independent_model_recovery(state)

    assert peak == 2
    assert calls.count(("validate", 1)) == 1
    assert calls.count(("repair", 1)) == 0
    assert calls.count(("validate", 2)) == 2
    assert calls.count(("repair", 2)) == 1
    assert result["model_lifecycle_status"][1]["status"] == "locally_ready"
    assert result["model_lifecycle_status"][2]["status"] == "locally_ready"
    assert cross_calls == [2]
    assert result["cross_model_duplicate_scan_after_local_completion"] is True


def test_worker_threads_preserve_job_context(monkeypatch):
    from app.logging_conf import current_job_id, set_job_context

    seen: list[str | None] = []

    def validate(state):
        model = int(state["generated_exams"][0]["model_number"])
        seen.append(current_job_id())
        return {
            "generated_exams": state["generated_exams"],
            "plans": state["plans"],
            "validation_reports": [{"model_number": model, "verdicts": []}],
            "repair_manifest": [],
            "validated_models": [model],
        }

    def cross(state, generated, reports, manifest, validated, warnings):
        return manifest, validated, warnings, {}

    monkeypatch.setattr("app.online.validator.validate_generated_questions", validate)
    monkeypatch.setattr("app.online.validator.repair_invalid_questions", lambda state: {})
    monkeypatch.setattr(
        "app.online.cross_model_duplicates.integrate_cross_model_duplicates", cross,
    )
    set_job_context("job-context-test", "doc-context-test")
    independent_recovery.run_independent_model_recovery({
        "num_models": 2,
        "generated_exams": [_exam(1), _exam(2)],
        "plans": [_plan(1), _plan(2)],
        "validation_reports": [], "repair_manifest": [], "repair_candidates": [],
        "validated_models": [], "warnings": [],
    })
    assert seen == ["job-context-test", "job-context-test"]


def test_model_merge_preserves_other_models_committed_lifecycle_and_memory():
    state = {
        "generated_exams": [_exam(1), _exam(2)], "plans": [_plan(1), _plan(2)],
        "validation_reports": [], "repair_manifest": [], "repair_candidates": [],
        "validated_models": [],
        "slot_lifecycle": {
            "m1_mcq_1": {"current_status": "PASS", "assignment_revision": 3},
            "m2_mcq_1": {"current_status": "FAIL", "assignment_revision": 1},
        },
        "repair_replacement_memory": {
            "m1_mcq_1": {"committed": True}, "m2_mcq_1": {"committed": False},
        },
    }
    worker = independent_recovery._isolate(state, 2)
    worker["slot_lifecycle"]["m1_mcq_1"] = {"current_status": "STALE"}
    worker["slot_lifecycle"]["m2_mcq_1"] = {"current_status": "PASS", "assignment_revision": 2}
    worker["repair_replacement_memory"]["m1_mcq_1"] = {"committed": False}
    worker["repair_replacement_memory"]["m2_mcq_1"] = {"committed": True}
    merged = independent_recovery._merge_worker(state, worker, 2)
    assert merged["slot_lifecycle"]["m1_mcq_1"]["current_status"] == "PASS"
    assert merged["slot_lifecycle"]["m2_mcq_1"]["current_status"] == "PASS"
    assert merged["repair_replacement_memory"]["m1_mcq_1"]["committed"] is True
    assert merged["repair_replacement_memory"]["m2_mcq_1"]["committed"] is True


def test_model_merge_preserves_each_models_first_validation_statistics():
    from app.online.eval_stats import create_pipeline_eval

    stats = create_pipeline_eval([("mcq", 1)], 2)
    worker_one = {**stats, "models": {**stats["models"]}}
    worker_one = __import__("copy").deepcopy(worker_one)
    worker_one["models"]["1"]["question_types"]["mcq"]["validation_total_first"] = 1
    worker_one["_runtime"]["first_validation_models"] = [1]
    merged = independent_recovery._merge_eval_stats(stats, worker_one, 1)

    worker_two = __import__("copy").deepcopy(stats)
    worker_two["models"]["2"]["question_types"]["mcq"]["validation_total_first"] = 1
    worker_two["_runtime"]["first_validation_models"] = [2]
    merged = independent_recovery._merge_eval_stats(merged, worker_two, 2)

    assert merged["models"]["1"]["validation_total_first"] == 1
    assert merged["models"]["2"]["validation_total_first"] == 1
    assert merged["_runtime"]["first_validation_models"] == [1, 2]


def test_isolated_worker_keeps_global_usage_and_question_context():
    state = {
        "generated_exams": [_exam(1), _exam(2)],
        "plans": [_plan(1), _plan(2)],
        "validation_reports": [], "repair_manifest": [], "repair_candidates": [],
        "validated_models": [],
        "_recovery_task_coordinator": {
            "lock": threading.RLock(), "reservations": {"m1_mcq_1": "task_shared"},
        },
    }
    worker = independent_recovery._isolate(state, 2)
    assert [value["model_number"] for value in worker["plans"]] == [2]
    assert {value["model_number"] for value in worker["global_plans"]} == {1, 2}
    assert {value["model_number"] for value in worker["global_generated_exams"]} == {1, 2}
    assert worker["_recovery_task_coordinator"] is state["_recovery_task_coordinator"]


def test_runtime_callbacks_and_locks_are_not_deep_copied(monkeypatch):
    import threading

    class Cancellation:
        def __init__(self):
            self._lock = threading.Lock()

        def is_cancelled(self):
            with self._lock:
                return False

    cancellation = Cancellation()
    events: list[tuple] = []
    callback = lambda *args, **kwargs: events.append(args)
    calls: list[int] = []

    def validate(state):
        model = int(state["generated_exams"][0]["model_number"])
        calls.append(model)
        return {
            "generated_exams": state["generated_exams"],
            "plans": state["plans"],
            "validation_reports": [{"model_number": model, "verdicts": []}],
            "repair_manifest": [],
            "validated_models": [model],
        }

    def cross(state, generated, reports, manifest, validated, warnings):
        return manifest, validated, warnings, {"cross_model_duplicate_scan_complete": True}

    monkeypatch.setattr("app.online.validator.validate_generated_questions", validate)
    monkeypatch.setattr("app.online.validator.repair_invalid_questions", lambda state: {})
    monkeypatch.setattr(
        "app.online.cross_model_duplicates.integrate_cross_model_duplicates", cross,
    )
    state = {
        "num_models": 2,
        "generated_exams": [_exam(1), _exam(2)],
        "plans": [_plan(1), _plan(2)],
        "validation_reports": [], "repair_manifest": [], "repair_candidates": [],
        "validated_models": [], "warnings": [],
        "is_cancelled": cancellation.is_cancelled,
        "audit_callback": callback,
    }
    result = independent_recovery.run_independent_model_recovery(state)
    assert sorted(calls) == [1, 2]
    assert result["is_cancelled"].__self__ is cancellation
    assert result["is_cancelled"].__func__ is Cancellation.is_cancelled
    assert result["audit_callback"] is callback
    assert result["validated_models"] == [1, 2]
    phases = {
        args[2].get("phase") for args in events
        if len(args) >= 3 and isinstance(args[2], dict)
    }
    assert {"validating", "duplicate_check", "locally_ready"} <= phases
    assert "cross_model_duplicate" in phases


def test_graph_routes_multi_model_to_independent_recovery():
    from app.online import graph

    assert graph._route_after_generate({"num_models": 2}) == "independent_model_recovery"
    assert graph._route_after_generate({"num_models": 1}) == "validate_generated_questions"


def test_graph_executes_independent_node_before_assembly(monkeypatch):
    from app.online import graph

    calls: list[str] = []
    cross_calls: list[int] = []

    def receive(state):
        return {}

    def generate(state):
        calls.append("generate")
        return {"generated_exams": [_exam(1), _exam(2)]}

    def validate(state):
        model = int(state["generated_exams"][0]["model_number"])
        calls.append(f"validate-{model}")
        return {
            "generated_exams": state["generated_exams"],
            "plans": state["plans"],
            "validation_reports": [{"model_number": model, "verdicts": []}],
            "repair_manifest": [],
            "validated_models": [model],
        }

    def repair(state):
        raise AssertionError("no repair expected for passing models")

    def assemble(state):
        calls.append("assemble")
        assert state.get("cross_model_duplicate_scan_after_local_completion") is True
        return {"warnings": ["assembled"]}

    def cross(state, generated, reports, manifest, validated, warnings):
        cross_calls.append(len(generated))
        return manifest, validated, warnings, {"cross_model_duplicate_scan_complete": True}

    monkeypatch.setattr("app.online.receive.receive_request", receive)
    monkeypatch.setattr("app.online.exam_builder.generate_exams_node", generate)
    monkeypatch.setattr("app.online.exam_builder.assemble_exams_node", assemble)
    monkeypatch.setattr("app.online.validator.validate_generated_questions", validate)
    monkeypatch.setattr("app.online.validator.repair_invalid_questions", repair)
    monkeypatch.setattr(
        "app.online.cross_model_duplicates.integrate_cross_model_duplicates", cross,
    )
    monkeypatch.setattr(graph, "_route_after_receive", lambda state: "generate_exams")

    class Cancellation:
        def __init__(self):
            self._lock = threading.Lock()

        def is_cancelled(self):
            with self._lock:
                return False

    cancellation = Cancellation()
    result = graph.build_exam_graph().invoke({
        "num_models": 2, "tasks": [("mcq", 1)], "difficulty": "easy",
        "document_language": "en", "generated_exams": [], "plans": [_plan(1), _plan(2)],
        "validation_reports": [], "repair_manifest": [], "repair_candidates": [],
        "validated_models": [], "warnings": [],
        "is_cancelled": cancellation.is_cancelled,
        "audit_callback": lambda *args, **kwargs: None,
    }, {"recursion_limit": 20})

    assert calls == ["generate", "validate-1", "validate-2", "assemble"]
    assert cross_calls == [1] or cross_calls == [2]
    assert result["warnings"] == ["assembled"]



def test_single_model_cross_model_recovery_does_not_shadow_model_helper(monkeypatch):
    """Cross-model recovery with one affected model must not bind _model_number.

    The 2026-11-15 20:54 exam reached cross-model duplicate recovery with one
    affected model; ``_model_number, worker = recover_model(...)`` shadowed the
    module helper and the following report comprehension raised
    ``TypeError: 'int' object is not callable``.
    """
    state = {
        "generated_exams": [_exam(1), _exam(2)],
        "plans": [_plan(1), _plan(2)],
        "validation_reports": [
            {"model_number": 1, "verdicts": []},
            {"model_number": 2, "verdicts": []},
        ],
        "repair_manifest": [{
            "model_number": 1, "slot_id": "m1_mcq_1", "status": "invalid",
            "failure_code": "cross_model_duplicate",
        }],
        "validated_models": [2],
        "warnings": [],
    }
    manifest = [{
        "model_number": 1, "slot_id": "m1_mcq_1", "question_id": "m1_q1",
        "status": "invalid", "failure_code": "cross_model_duplicate",
    }]
    monkeypatch.setattr(
        "app.online.validator.repair_invalid_questions", lambda _worker: {},
    )
    monkeypatch.setattr(
        "app.online.validator.validate_generated_questions", lambda _worker: {},
    )
    monkeypatch.setattr(
        independent_recovery, "_merge_worker",
        lambda merged, worker, _model: {**merged, **worker},
    )
    monkeypatch.setattr(
        "app.online.cross_model_duplicates.integrate_cross_model_duplicates",
        lambda _state, _exams, _reports, _manifest, validated, warnings: (
            [], validated, warnings, {},
        ),
    )
    result = independent_recovery._targeted_cross_model_recovery(state, manifest)
    assert "validated_models" in result


def test_one_cross_model_target_completes_targeted_recovery_and_final_scan(monkeypatch):
    """End-to-end cross-model path with one affected model (the 20:54 crash)."""
    state = {
        "num_models": 2,
        "generated_exams": [_exam(1), _exam(2)],
        "plans": [_plan(1), _plan(2)],
        "validation_reports": [
            {"model_number": 1, "verdicts": []},
            {"model_number": 2, "verdicts": []},
        ],
        "repair_manifest": [], "repair_candidates": [],
        "validated_models": [1, 2], "warnings": [],
    }
    monkeypatch.setattr(
        independent_recovery, "_recover_model",
        lambda worker_state, model: ({
            **worker_state,
            "generated_exams": [
                value for value in worker_state["generated_exams"]
                if int(value.get("model_number") or 0) == model
            ],
            "validation_reports": [{
                "model_number": model, "verdicts": [],
            }],
            "validated_models": [model],
        }, 1),
    )
    monkeypatch.setattr(
        "app.online.validator.repair_invalid_questions", lambda _worker: {},
    )
    monkeypatch.setattr(
        "app.online.validator.validate_generated_questions", lambda _worker: {},
    )
    calls = {"n": 0}

    def cross(_state, _generated, _reports, manifest, validated, warnings):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([{
                "model_number": 1, "slot_id": "m1_mcq_1",
                "question_id": "m1_q1", "status": "invalid",
                "failure_code": "cross_model_duplicate",
            }], validated, warnings, {})
        return ([], validated, warnings, {})

    monkeypatch.setattr(
        "app.online.cross_model_duplicates.integrate_cross_model_duplicates", cross,
    )
    result = independent_recovery.run_independent_model_recovery(state)
    assert calls["n"] >= 2
    assert result["cross_model_duplicate_scan_after_local_completion"] is True
    assert "_recovery_task_coordinator" not in result


def test_two_model_cross_model_targets_run_concurrently_and_merge(monkeypatch):
    state = {
        "num_models": 2,
        "generated_exams": [_exam(1), _exam(2)],
        "plans": [_plan(1), _plan(2)],
        "validation_reports": [
            {"model_number": 1, "verdicts": []},
            {"model_number": 2, "verdicts": []},
        ],
        "repair_manifest": [], "repair_candidates": [],
        "validated_models": [], "warnings": [],
    }
    active_models: list[int] = []
    import threading
    lock = threading.Lock()

    def record_model(worker):
        with lock:
            active_models.append(int(worker["generated_exams"][0]["model_number"]))
        return {}

    monkeypatch.setattr(
        "app.online.validator.repair_invalid_questions", record_model,
    )
    monkeypatch.setattr(
        "app.online.validator.validate_generated_questions", record_model,
    )
    calls = {"n": 0}
    targets = [
        {"model_number": 1, "slot_id": "m1_mcq_1", "question_id": "m1_q1",
         "status": "invalid", "failure_code": "cross_model_duplicate"},
        {"model_number": 2, "slot_id": "m2_mcq_1", "question_id": "m2_q1",
         "status": "invalid", "failure_code": "cross_model_duplicate"},
    ]

    def cross(_state, _generated, _reports, manifest, validated, warnings):
        calls["n"] += 1
        return (targets if calls["n"] == 1 else []), validated, warnings, {}

    monkeypatch.setattr(
        "app.online.cross_model_duplicates.integrate_cross_model_duplicates", cross,
    )
    result = independent_recovery._targeted_cross_model_recovery(state, targets)
    assert set(active_models) == {1, 2}
    assert "validated_models" in result
