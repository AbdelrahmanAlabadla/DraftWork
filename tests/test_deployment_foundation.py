from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import config
from app.api import repositories
from app.api.job_routes import _generation_payload
from app.api.main import app
from app.api.routes import GenerateRequest
from app.file_storage.local import LocalFileStorage
from app.jobs.tasks import generate_exam, ingest_document


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


def _valid_pdf_bytes() -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(output)
    return output.getvalue()


def test_versioned_upload_is_session_owned_and_queued(monkeypatch, tmp_path):
    session = _session_fakes(monkeypatch)
    monkeypatch.setattr(config, "LOCAL_STORAGE_ROOT", str(tmp_path / "storage"))
    captured = {}

    def create_document_with_job(**kwargs):
        captured.update(kwargs)
        return ({"id": kwargs["document_id"]}, {"id": "22222222-2222-4222-8222-222222222222"})

    monkeypatch.setattr(repositories, "create_document_with_job", create_document_with_job)
    monkeypatch.setattr("app.api.job_routes.job_service.enqueue_ingestion", lambda job_id: None)

    client = TestClient(app)
    pdf = _valid_pdf_bytes()
    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.pdf", io.BytesIO(pdf), "application/pdf")},
    )

    assert response.status_code == 202
    assert captured["session_id"] == session["id"]
    assert captured["size_bytes"] == len(pdf)
    assert response.json()["status"] == "queued"
    assert "HttpOnly" in response.headers["set-cookie"]


def test_upload_limit_is_enforced_while_streaming(monkeypatch, tmp_path):
    _session_fakes(monkeypatch)
    monkeypatch.setattr(config, "LOCAL_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 8)
    client = TestClient(app)
    response = client.post(
        "/api/v1/documents",
        files={"file": ("large.pdf", io.BytesIO(b"%PDF-1234"), "application/pdf")},
    )
    assert response.status_code == 413


def test_generation_limits_match_agreed_configuration():
    with pytest.raises(HTTPException) as exc:
        _generation_payload(
            GenerateRequest(
                document_id="doc",
                mcq_count=100,
                tf_count=51,
                child_ids=["child"],
            )
        )
    assert exc.value.status_code == 400
    assert "150" in str(exc.value.detail)
    assert config.MAX_MODELS_PER_GENERATION == 4
    assert config.MAX_ACTIVE_GENERATION_JOBS_PER_SESSION == 2
    assert config.GENERATION_REQUESTS_PER_MINUTE == 2


def test_generation_task_has_1300_second_hard_timeout():
    assert generate_exam.time_limit == 1300
    assert generate_exam.soft_time_limit == 1290


def test_online_generation_parallelism_has_fifty_call_ceiling():
    assert config.MAX_PARALLEL_GENERATION_CALLS == 50
    assert config.LLM_JOB_CONCURRENCY_LIMIT == 50
    assert config.MAX_PARALLEL_DIFFICULTY_ADJUSTMENTS == 10
    assert config.MAX_PARALLEL_VALIDATOR_CALLS == 10
    assert config.MAX_PARALLEL_CATALOG_CALLS == 50
    assert config.MAX_PARALLEL_DUPLICATE_JUDGE_CALLS == 10
    assert config.LLM_JOB_CONCURRENCY_LIMIT == 50
    assert config.DEEPSEEK_CONCURRENCY_LIMIT == 50


def test_ingestion_task_does_not_limit_downstream_pipeline_runtime():
    assert ingest_document.time_limit is None
    assert ingest_document.soft_time_limit is None


def test_gpu_worker_uses_terminable_process_pool_and_process_liveness_healthcheck():
    compose = (Path(__file__).parents[1] / "compose.yaml").read_text(encoding="utf-8")
    assert '"--pool=prefork", "--concurrency=1"' in compose
    assert "psutil.process_iter" in compose
    assert "inspect\", \"ping" not in compose


def test_cancel_job_is_owned_persisted_and_terminated(monkeypatch):
    _session_fakes(monkeypatch)
    job_id = "22222222-2222-4222-8222-222222222222"
    terminated = []
    monkeypatch.setattr(
        repositories,
        "get_job_for_session",
        lambda requested, session_id: {"id": requested},
    )
    monkeypatch.setattr(
        repositories,
        "cancel_job",
        lambda requested: {
            "id": requested,
            "document_id": "33333333-3333-4333-8333-333333333333",
            "session_id": "11111111-1111-4111-8111-111111111111",
            "type": "document_ingestion",
            "status": "cancelled",
            "stage": "cancelled",
        },
    )
    monkeypatch.setattr(
        "app.api.job_routes.job_service.terminate_job",
        lambda requested, **kwargs: terminated.append((requested, kwargs)),
    )

    response = TestClient(app).delete(f"/api/v1/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert terminated == [(job_id, {
        "document_id": "33333333-3333-4333-8333-333333333333",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "cleanup_document": True,
    })]


def test_local_storage_rejects_keys_outside_root(tmp_path):
    storage = LocalFileStorage(tmp_path / "root")
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    with pytest.raises(ValueError):
        storage.put_file("../escape.pdf", source)


def test_postgres_chunk_migration_has_document_isolation_and_lookup_indexes():
    migration = (
        Path(__file__).parents[1]
        / "migrations" / "versions" / "0005_postgres_document_chunks.py"
    ).read_text(encoding="utf-8")
    assert "document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE" in migration
    assert "document_chunks_document_order_idx" in migration
    assert "document_chunks_section_idx" in migration
    assert "document_chunks_subsection_idx" in migration
