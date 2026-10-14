from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.job_routes import _generation_payload
from app.api.main import app
from app.api.routes import GenerateRequest


client = TestClient(app)


def test_legacy_generate_endpoint_is_retired():
    response = client.post(
        "/generate",
        json={"document_id": "doc", "mcq_count": 1, "child_ids": ["c1"]},
    )
    assert response.status_code == 410
    assert "exam-jobs" in response.json()["detail"]


def test_async_exam_job_route_is_the_supported_entry_path():
    operation = app.openapi()["paths"][
        "/api/v1/documents/{document_id}/exam-jobs"
    ]["post"]
    assert operation["operationId"].startswith("create_exam_job_")


def test_async_payload_uses_canonical_question_types():
    payload = _generation_payload(GenerateRequest(
        document_id="doc", child_ids=["c1"], mcq_count=2, why_count=3,
        definition_count=1,
    ))
    assert payload["tasks"] == [("mcq", 2), ("definition", 1), ("why", 3)]


def test_removed_question_type_is_rejected_by_async_payload():
    with pytest.raises(HTTPException) as caught:
        _generation_payload(GenerateRequest(
            document_id="doc", child_ids=["c1"],
            question_type="short" + "_answer", number_of_questions=1,
        ))
    assert caught.value.status_code == 400
