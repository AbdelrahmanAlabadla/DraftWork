"""Error persistence, checkpoint safety, and audit visibility regressions."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager

import pytest

from app.api import job_routes, repositories
from app.jobs import tasks
from app.online import diagnostics, graph, validator
from app.logging_conf import JsonFormatter


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.calls.append((str(sql), tuple(params)))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_diagnostic_insert_does_not_touch_checkpoint_revision(monkeypatch):
    connection = _FakeConnection()

    @contextmanager
    def fake_connection():
        yield connection

    monkeypatch.setattr(repositories.db, "connection", fake_connection)
    repositories.record_exam_diagnostic_event(
        "exam-1", job_id="job-1", stage="general_validation",
        component="telemetry", error_type="RuntimeError",
        message="telemetry down token=secret", continued=True,
        traceback_text="Traceback token=secret",
    )
    sql, params = connection.cursor_obj.calls[0]
    assert "internal_diagnostic" in sql
    assert "UPDATE jobs" not in sql
    assert "checkpoint_revision" not in sql.lower()
    assert params[0] == "exam-1"
    payload = params[3].obj if hasattr(params[3], "obj") else params[3]
    assert "secret" not in json.dumps(payload)


def test_optional_failure_falls_back_when_db_write_fails(monkeypatch):
    monkeypatch.setattr(
        repositories, "get_exam_id_for_job", lambda _job: "exam-1",
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(repositories, "record_exam_diagnostic_event", boom)
    diagnostics.record_optional_failure(
        RuntimeError("original"), stage="s", component="c", job_id="job-1",
    )


def test_safe_callback_does_not_swallow_checkpoint_control_errors():
    def stale(_job):
        raise repositories.StaleJobExecution("lost ownership")

    with pytest.raises(repositories.StaleJobExecution):
        graph._safe_optional_callback(stale, "job-1")


@pytest.mark.parametrize("error", [
    repositories.StaleJobExecution("lost ownership"),
    repositories.StaleCheckpoint("stale revision"),
    repositories.InvalidState("corrupt checkpoint"),
])
def test_safe_callback_propagates_all_checkpoint_states(error):
    def callback(_job):
        raise error

    with pytest.raises(type(error)):
        graph._safe_optional_callback(callback, "job-1")


def test_safe_callback_persists_optional_failure(monkeypatch):
    captured = []
    monkeypatch.setattr(
        graph, "record_optional_failure",
        lambda exc, **kwargs: captured.append((exc, kwargs)),
    )

    def optional():
        raise RuntimeError("progress db down")

    graph._safe_optional_callback(
        optional, job_id="job-1", stage="validating", component="progress",
    )
    assert captured and captured[0][1]["component"] == "progress"


def test_repair_telemetry_failure_is_persisted_and_isolated(monkeypatch):
    captured = []
    monkeypatch.setattr(
        validator, "record_repair_outcome",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("telemetry down")),
    )
    monkeypatch.setattr(
        validator, "record_optional_failure",
        lambda exc, **kwargs: captured.append((exc, kwargs)),
    )
    validator._safe_record_repair_outcome(
        {}, {}, {}, repaired_ids=[], failed_ids=[], sent_ids=[],
    )
    assert captured


def test_repair_coordinator_telemetry_path_calls_valid_recorder(monkeypatch):
    """Regression for the failed 2026-11-15 20:29 exam.

    Staging repaired candidates with eval_stats must call the telemetry wrapper,
    not a corrupted ``_safe__runtime`` name that crashes Recovery.
    """
    recorded = []
    monkeypatch.setattr(
        validator, "_safe_record_repair_outcome",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )
    monkeypatch.setattr(
        validator, "_repair_invalid_questions_serial",
        lambda _state: {
            "repair_manifest": [],
            "repair_candidates": [{
                "model_number": 1, "slot_id": "s1", "question_id": "q1",
                "question_type": "mcq",
            }],
            "question_repair_attempts": {}, "question_repair_calls": {},
            "question_repair_failures": {}, "slot_completion_attempts": {},
            "repair_working_candidates": {}, "repair_replacement_memory": {},
            "repair_retry_seeds": {}, "validation_repair_history": [],
            "repair_call_logs": [], "incompatible_task_ids": [],
            "validated_models": [1], "warnings": [], "eval_stats": {},
        },
    )
    state = {
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [{
            "question_id": "q1", "slot_id": "s1",
        }, {"question_id": "q2", "slot_id": "s2"}]}}],
        "validation_reports": [{"model_number": 1, "verdicts": []}],
        "repair_manifest": [
            {"model_number": 1, "slot_id": "s1", "status": "invalid"},
            {"model_number": 1, "slot_id": "s2", "status": "invalid"},
        ],
        "plans": [], "eval_stats": {}, "warnings": [],
    }
    result = validator.repair_invalid_questions(state)
    assert result["repair_candidates"]
    assert recorded


def test_optional_checkpoint_append_isolates_transient_errors(monkeypatch):
    class Writer:
        def append(self, **_kwargs):
            raise RuntimeError("database temporarily unavailable")

    captured = []
    monkeypatch.setattr(
        tasks, "record_optional_failure",
        lambda exc, **kwargs: captured.append((exc, kwargs)),
    )
    tasks._optional_checkpoint_append(
        Writer(), job_id="job-1", exam_id="exam-1", stage="replay",
        event_type="questions_generated", payload={"x": 1},
    )
    assert captured


def test_optional_checkpoint_append_propagates_stale_checkpoint():
    class Writer:
        def append(self, **_kwargs):
            raise repositories.StaleCheckpoint("stale revision")

    with pytest.raises(repositories.StaleCheckpoint):
        tasks._optional_checkpoint_append(
            Writer(), job_id="job-1", exam_id="exam-1", stage="replay",
            event_type="questions_generated", payload={"x": 1},
        )


def test_audit_endpoint_hides_internal_events(monkeypatch):
    monkeypatch.setattr(
        repositories, "list_exam_audit_events",
        lambda *_a, **_k: [
            {"event_type": "pipeline_stage", "payload": {"progress": 50}},
            {"event_type": "stage_exception", "payload": {"traceback": "secret"}},
            {"event_type": "internal_diagnostic", "payload": {"message": "x"}},
            {"event_type": "exam_saved", "payload": {}},
        ],
    )
    result = job_routes.get_exam_audit_events("exam-1", "session-1", None)
    assert [event["event_type"] for event in result["events"]] == [
        "pipeline_stage", "exam_saved",
    ]


def test_json_log_has_sanitized_exception_diagnostics():
    try:
        raise RuntimeError("provider token=secret failed")
    except RuntimeError:
        import sys

        exc_info = sys.exc_info()
    record = logging.LogRecord(
        "TEST", logging.ERROR, __file__, 1, "failed", (), exc_info,
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["exception_type"] == "RuntimeError"
    assert "provider" in payload["exception_message"]
    assert "secret" not in payload["exception_message"]
    assert "secret" not in payload["traceback"]
    assert "Traceback" in payload["traceback"]
