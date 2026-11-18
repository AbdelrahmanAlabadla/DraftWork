from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app.api import benchmarks, evaluation_store, session_middleware
from app.api.evaluation_routes import require_evaluation_access
from app.api.main import app
from app.api.session_middleware import require_session_id, require_user_id, _needs_session

client = TestClient(app)
ROOT = Path(__file__).parents[1] / "FrontEnd"


def _stub_session(monkeypatch) -> None:
    async def _anonymous(_request):
        return None

    monkeypatch.setattr(session_middleware, "authenticate", _anonymous)
    monkeypatch.setattr(
        session_middleware.repositories, "resolve_session", lambda *a, **k: None
    )
    monkeypatch.setattr(
        session_middleware.repositories,
        "create_session",
        lambda *a, **k: {"id": "session-test", "user_id": None},
    )


@pytest.fixture
def authed(monkeypatch):
    _stub_session(monkeypatch)
    app.dependency_overrides[require_evaluation_access] = lambda: None
    app.dependency_overrides[require_session_id] = lambda: "session-test"
    app.dependency_overrides[require_user_id] = lambda: "user-test"
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def machine(monkeypatch):
    _stub_session(monkeypatch)
    monkeypatch.setattr(config, "EVALUATION_DASHBOARD_TOKEN", "local-verification-token")
    yield client
    app.dependency_overrides.clear()


def test_detailed_evaluation_endpoints_require_authentication(monkeypatch):
    _stub_session(monkeypatch)
    response = client.get("/api/v1/evaluation/overview")
    assert response.status_code == 401
    assert _needs_session("/api/v1/evaluation/overview")
    assert _needs_session("/api/v1/evaluation/exams/exam_1")


def test_machine_token_grants_read_only_access(machine, monkeypatch):
    monkeypatch.setattr(
        evaluation_store, "overview_summary", lambda **kwargs: {"window": kwargs}
    )
    denied = machine.get("/api/v1/evaluation/overview")
    assert denied.status_code == 401
    allowed = machine.get(
        "/api/v1/evaluation/overview",
        headers={"X-Evaluation-Token": "local-verification-token"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["window"]["days"] == 30


def test_overview_endpoint_forwards_filters(authed, monkeypatch):
    captured = {}

    def _overview(**kwargs):
        captured.update(kwargs)
        return {"runs": {"generation_runs": 3}, "quality": {}, "jobs": {}, "llm": {}}

    monkeypatch.setattr(evaluation_store, "overview_summary", _overview)
    response = authed.get(
        "/api/v1/evaluation/overview?days=7&include_benchmarks=true"
    )
    assert response.status_code == 200
    assert response.json()["runs"]["generation_runs"] == 3
    assert captured == {"days": 7, "include_benchmarks": True}


def test_exam_reports_endpoint_paginates_and_filters(authed, monkeypatch):
    captured = {}

    def _list(**kwargs):
        captured.update(kwargs)
        return {"items": [{"exam_id": "exam_1"}], "total": 1, "limit": 5, "offset": 10}

    monkeypatch.setattr(evaluation_store, "list_exam_reports", _list)
    response = authed.get(
        "/api/v1/evaluation/exams?search=exam&status=healthy&benchmark=only"
        "&days=90&limit=5&offset=10"
    )
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert captured["search"] == "exam"
    assert captured["status"] == "healthy"
    assert captured["benchmark_mode"] == "only"
    assert captured["days"] == 90
    assert captured["limit"] == 5
    assert captured["offset"] == 10


def test_exam_detail_returns_404_for_unknown_run(authed, monkeypatch):
    monkeypatch.setattr(evaluation_store, "exam_report", lambda exam_id: None)
    response = authed.get("/api/v1/evaluation/exams/exam_missing")
    assert response.status_code == 404


def test_pipeline_endpoint_contract(authed, monkeypatch):
    monkeypatch.setattr(
        evaluation_store,
        "pipeline_analytics",
        lambda **kwargs: {
            "coverage": {"runs": 2},
            "question_types": {},
            "recovery": {"slots": 0},
        },
    )
    response = authed.get("/api/v1/evaluation/pipeline?days=0")
    assert response.status_code == 200
    assert response.json()["coverage"]["runs"] == 2


def test_benchmark_estimate_requires_valid_configuration(authed, monkeypatch):
    monkeypatch.setattr(
        benchmarks,
        "estimate",
        lambda payload, **kwargs: {"available": True, "estimated": True, "estimate": {}},
    )
    body = {
        "label": "Smoke",
        "document_id": "doc-1",
        "child_ids": ["child-1"],
        "tasks": {"mcq": 4},
        "num_models": 1,
        "difficulty": "easy",
        "repeats": 1,
    }
    response = authed.post("/api/v1/evaluation/benchmarks/estimate", json=body)
    assert response.status_code == 200
    assert response.json()["estimated"] is True

    body["tasks"] = {}
    invalid = authed.post("/api/v1/evaluation/benchmarks/estimate", json=body)
    assert invalid.status_code == 422


def test_benchmark_creation_passes_idempotency_key(authed, monkeypatch):
    captured = {}

    def _create(payload, **kwargs):
        captured.update(kwargs)
        captured["payload"] = payload
        return {"benchmark_id": "bench-1", "status": "running"}

    monkeypatch.setattr(benchmarks, "create_benchmark", _create)
    body = {
        "label": "Physics baseline",
        "document_id": "doc-1",
        "child_ids": ["child-1", "child-2"],
        "tasks": {"mcq": 10, "true_false": 5},
        "num_models": 2,
        "difficulty": "mix",
        "repeats": 2,
        "stop_on_failure": True,
    }
    response = authed.post(
        "/api/v1/evaluation/benchmarks",
        json=body,
        headers={"Idempotency-Key": "bench-key-1"},
    )
    assert response.status_code == 201
    assert response.json()["benchmark_id"] == "bench-1"
    assert captured["idempotency_key"] == "bench-key-1"
    assert captured["session_id"] == "session-test"
    assert captured["user_id"] == "user-test"


def test_benchmark_comparison_requires_at_least_two_ids(authed):
    response = authed.get("/api/v1/evaluation/benchmarks/compare?ids=bench-1")
    assert response.status_code == 400


def test_legacy_eval_summary_contract_remains_available(monkeypatch):
    summary = {"total_exam_runs": 2, "overall": {}, "models": {}, "recent_exam_runs": []}
    monkeypatch.setattr(evaluation_store, "load_eval_summary", lambda: summary)
    response = client.get("/api/v1/eval-summary")
    assert response.status_code == 200
    assert response.json() == summary
    assert client.get("/api/v1/api/eval-summary").status_code == 200


def test_evaluation_dashboard_pages_are_served_with_shared_assets():
    redirect = client.get("/eval.html")
    assert redirect.status_code == 200
    assert "evaluation/overview.html" in redirect.text

    for page, module in (
        ("overview.html", "overview.js"),
        ("reports.html", "reports.js"),
        ("pipeline.html", "pipeline.js"),
        ("benchmarks.html", "benchmarks.js"),
    ):
        response = client.get(f"/evaluation/{page}")
        assert response.status_code == 200, page
        assert f"js/evaluation/{module}" in response.text
        assert "../css/evaluation/base.css" in response.text

    navigation = (ROOT / "js" / "evaluation" / "components.js").read_text(
        encoding="utf-8"
    )
    for link in ("overview.html", "reports.html", "pipeline.html", "benchmarks.html"):
        assert link in navigation

    index = client.get("/evaluation/index.html")
    assert index.status_code == 200
    assert "overview.html" in index.text

    for asset in (
        "/css/evaluation/base.css",
        "/js/evaluation/api.js",
        "/js/evaluation/components.js",
        "/js/evaluation/charts.js",
    ):
        assert client.get(asset).status_code == 200, asset

    assert not (ROOT / "js" / "eval.js").exists()


def test_frontend_uses_authenticated_same_origin_api():
    source = (ROOT / "js" / "evaluation" / "api.js").read_text(encoding="utf-8")
    assert "authHeaders" in source
    assert 'credentials: "same-origin"' in source
    assert "/api/v1/evaluation" in source
    assert 'credentials: "omit"' not in source
    assert "X-Evaluation-Token" not in source
