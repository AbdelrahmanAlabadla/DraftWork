from __future__ import annotations

import inspect
import io
import os
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import cleanup, config
from app.api import repositories
from app.api.main import app
from app.deadlines import DeadlineExceeded, supports_deadlines
from app.file_storage.local import LocalFileStorage
from app.file_validation import validate_pdf
from app.jobs import tasks as task_module
from app.offline.parser import ParserError
from app.offline.pipeline import PipelineError


def _session_fakes(monkeypatch):
    session = {"id": "11111111-1111-4111-8111-111111111111"}
    monkeypatch.setattr(
        "app.api.session_middleware.repositories.resolve_session",
        lambda token_hash: session,
    )
    monkeypatch.setattr(
        "app.api.session_middleware.repositories.create_session",
        lambda token_hash: session,
    )
    return session


def test_complete_pdf_validation_has_no_page_count_limit(tmp_path):
    path = tmp_path / "many-pages.pdf"
    writer = PdfWriter()
    for _ in range(250):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as target:
        writer.write(target)
    validate_pdf(path)


def test_old_exports_are_removed_without_touching_exam_sources(tmp_path):
    storage = LocalFileStorage(tmp_path)
    export_key = "sessions/s1/exams/e1/exports/result.zip"
    source_key = "sessions/s1/documents/d1/source.pdf"
    storage.put_bytes(export_key, b"zip")
    storage.put_bytes(source_key, b"pdf")
    old = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
    os.utime(storage._path(export_key), (old, old))
    os.utime(storage._path(source_key), (old, old))

    deleted = storage.delete_older_than(
        "sessions",
        datetime.now(timezone.utc) - timedelta(days=1),
        path_component="exports",
    )

    assert deleted == 1
    assert not storage._path(export_key).exists()
    assert storage._path(source_key).exists()


def test_cleanup_database_policy_explicitly_preserves_exams():
    source = inspect.getsource(repositories.cleanup_database_records)
    assert "DELETE FROM exams" not in source
    assert "NOT EXISTS (SELECT 1 FROM exams e WHERE e.session_id = s.id)" not in source
    migration = (
        Path(__file__).parents[1] / "migrations" / "003_deployment_hardening.sql"
    ).read_text(encoding="utf-8")
    assert "REFERENCES jobs(id) ON DELETE SET NULL" in migration
    assert "REFERENCES documents(id) ON DELETE SET NULL" in migration
    assert "REFERENCES sessions(id) ON DELETE SET NULL" in migration


def test_cleanup_removes_assets_but_does_not_delete_saved_exam(monkeypatch):
    calls: list[tuple[str, str]] = []

    class Storage:
        def delete_tree(self, prefix):
            calls.append(("storage", prefix))
            return 1

        def delete_older_than(self, prefix, cutoff, *, path_component=None):
            return 0

        def list_prefixes(self, prefix, *, levels):
            return []

    class Vectors:
        def delete_document(self, document_id, session_id=None, raise_errors=False):
            calls.append(("qdrant", document_id))
            return 1

        def delete_orphaned_session_points(self, valid_scopes, *, max_points):
            return 0

    monkeypatch.setattr(cleanup, "get_file_storage", lambda: Storage())
    monkeypatch.setattr(cleanup, "VectorStore", Vectors)
    monkeypatch.setattr(repositories, "start_cleanup_run", lambda: 1)
    monkeypatch.setattr(
        repositories,
        "cleanup_document_candidates",
        lambda **kwargs: [{
            "id": "d1", "session_id": "s1", "has_exam": True,
            "source_storage_key": "source", "structure_storage_key": "structure",
            "active_index_job_id": "j1",
        }],
    )
    monkeypatch.setattr(
        repositories,
        "mark_document_assets_deleted",
        lambda document_id: calls.append(("mark", document_id)),
    )
    monkeypatch.setattr(repositories, "expired_session_ids", lambda **kwargs: [])
    monkeypatch.setattr(repositories, "active_document_scopes", lambda: set())
    monkeypatch.setattr(repositories, "cleanup_database_records", lambda **kwargs: {})
    monkeypatch.setattr(repositories, "finish_cleanup_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(cleanup, "_delete_pipeline_artifacts", lambda document_id: 0)

    result = cleanup.run_cleanup()

    assert result["documents"] == 1
    assert ("mark", "d1") in calls
    assert all(kind != "exam" for kind, _ in calls)


def test_production_configuration_rejects_insecure_values(monkeypatch):
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    monkeypatch.setattr(config, "SESSION_COOKIE_SECURE", False)
    monkeypatch.setattr(config, "SESSION_HASH_PEPPER", "")
    monkeypatch.setattr(config, "DATABASE_URL", None)
    monkeypatch.setattr(config, "CORS_ALLOW_ORIGINS", ("http://localhost:8000",))
    monkeypatch.setattr(config, "METRICS_TOKEN", None)
    with pytest.raises(RuntimeError, match="Unsafe production configuration"):
        config.validate_runtime_config()


def test_upload_idempotency_reuses_existing_job_without_dispatch(monkeypatch, tmp_path):
    _session_fakes(monkeypatch)
    monkeypatch.setattr(config, "LOCAL_STORAGE_ROOT", str(tmp_path / "storage"))
    captured = {}

    def create_idempotent(**kwargs):
        captured.update(kwargs)
        return (
            {"id": "33333333-3333-4333-8333-333333333333"},
            {"id": "22222222-2222-4222-8222-222222222222"},
            True,
        )

    monkeypatch.setattr(repositories, "create_document_with_job_idempotent", create_idempotent)
    monkeypatch.setattr(
        "app.api.job_routes.job_service.enqueue_ingestion",
        lambda job_id: pytest.fail("a reused job must not be dispatched again"),
    )
    client = TestClient(app)
    response = client.post(
        "/api/v1/documents",
        headers={"Idempotency-Key": "upload-operation-123"},
        files={"file": ("notes.pdf", io.BytesIO(b"%PDF-1.4\ncontent"), "text/plain")},
    )

    assert response.status_code == 202
    assert captured["media_type"] == "application/pdf"
    assert captured["idempotency_key"] == "upload-operation-123"


def test_generation_idempotency_reuses_job_and_validates_selection(monkeypatch):
    _session_fakes(monkeypatch)
    captured = {}

    class Storage:
        def read_json(self, key):
            return {"sections": [{"child_ids": ["c1"]}]}

    monkeypatch.setattr(
        repositories,
        "get_document_for_session",
        lambda document_id, session_id: {
            "id": document_id,
            "status": "ready",
            "structure_storage_key": "structure.json",
        },
    )
    monkeypatch.setattr("app.api.job_routes.get_file_storage", lambda: Storage())

    def create_generation_job(**kwargs):
        captured.update(kwargs)
        return {
            "id": "22222222-2222-4222-8222-222222222222",
            "_idempotency_reused": True,
        }

    monkeypatch.setattr(repositories, "create_generation_job", create_generation_job)
    monkeypatch.setattr(
        "app.api.job_routes.job_service.enqueue_generation",
        lambda job_id: pytest.fail("a reused generation must not be dispatched again"),
    )
    client = TestClient(app)
    response = client.post(
        "/api/v1/documents/33333333-3333-4333-8333-333333333333/exam-jobs",
        headers={"Idempotency-Key": "generation-operation-123"},
        json={"mcq_count": 1, "child_ids": ["c1"]},
    )

    assert response.status_code == 202
    assert captured["idempotency_key"] == "generation-operation-123"
    assert captured["user_id"] is None
    assert len(captured["request_hash"]) == 64


def test_public_error_response_has_stable_code_and_request_id(monkeypatch):
    _session_fakes(monkeypatch)
    client = TestClient(app)
    response = client.post(
        "/api/v1/documents",
        headers={"X-Request-ID": "request-123"},
        files={"file": ("notes.txt", io.BytesIO(b"not a pdf"), "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "INVALID_FILE",
        "message": "Only PDF files are supported",
        "request_id": "request-123",
        "retryable": False,
    }


def test_wrapped_parser_timeout_maps_to_timeout():
    root = TimeoutError("provider timeout")
    parser_error = ParserError("LlamaParse failed")
    parser_error.__cause__ = root
    pipeline_error = PipelineError("LlamaParse failed")
    pipeline_error.__cause__ = parser_error

    assert task_module._error_code(pipeline_error) == "TIMEOUT"
    assert task_module._public_error_message(
        pipeline_error, "Document processing"
    ) == "Document processing timed out"


@pytest.mark.skipif(not supports_deadlines(), reason="SIGALRM deadline requires Linux main thread")
def test_generation_deadline_interrupts_running_task(monkeypatch):
    job = {
        "document_id": "document-1",
        "session_id": "session-1",
        "request_data": {
            "tasks": [["mcq", 1]],
            "num_models": 1,
            "child_ids": ["child-1"],
            "difficulty": "mix",
        },
    }
    document = {"id": "document-1", "active_index_job_id": "index-job-1"}
    failures: list[tuple[str, str, str]] = []
    monkeypatch.setattr(task_module.config, "GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(task_module.repositories, "claim_job", lambda *_: job)
    monkeypatch.setattr(task_module.repositories, "get_document", lambda *_: document)
    monkeypatch.setattr(task_module.repositories, "update_job_progress", lambda *_: None)
    monkeypatch.setattr(
        task_module.repositories,
        "fail_job",
        lambda job_id, code, message: failures.append((job_id, code, message)),
    )
    monkeypatch.setattr(task_module, "_heartbeat", lambda *_: nullcontext())
    monkeypatch.setattr(task_module, "generate_exams", lambda *args, **kwargs: time.sleep(2))

    started = time.monotonic()
    with pytest.raises(DeadlineExceeded):
        task_module.generate_exam.run("generation-job-1")

    assert time.monotonic() - started < 0.5
    assert failures == [
        (
            "generation-job-1",
            "TIMEOUT",
            "Generation exceeded its configured time limit",
        )
    ]


def test_metrics_uses_durable_worker_data_without_request_specific_labels(monkeypatch):
    monkeypatch.setattr(
        repositories,
        "monitoring_snapshot",
        lambda: {
            "jobs": [{
                "type": "exam_generation", "status": "completed", "count": 2,
                "duration_sum": 12.5, "duration_count": 2,
            }],
            "cleanup_runs": [{
                "status": "completed", "count": 1,
                "duration_sum": 3.0, "duration_count": 1,
            }],
            "llm_usage": [{
                "provider": "deepseek", "model": "deepseek-flash",
                "operation": "generator", "calls": 2,
                "input_tokens": 100, "output_tokens": 50,
                "cached_tokens": 10, "reasoning_tokens": 5,
                "estimated_cost_usd": 0.01,
                "duration_sum": 4.0, "duration_count": 2,
            }],
        },
    )
    monkeypatch.setattr(repositories, "job_status_counts", lambda: {})

    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert 'genexam_jobs_current{status="completed",type="exam_generation"} 2' in response.text
    assert "genexam_llm_input_tokens" in response.text
    assert "genexam_cleanup_duration_seconds_sum" in response.text
    assert "extra=" not in response.text
