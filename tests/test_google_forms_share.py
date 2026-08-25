"""Teacher sharing: Drive permission granted per model x email; failures are
warnings only and never fail the export."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.integrations.google_forms import client, exporter


@pytest.fixture()
def fake_forms(monkeypatch):
    calls = {"share": [], "create": 0}

    def fake_create(title, creds=None):
        calls["create"] += 1
        return {"form_id": f"form_{calls['create']}", "title": title,
                "edit_url": "https://edit", "view_url": "https://view"}

    monkeypatch.setattr(client, "create_form", fake_create)
    monkeypatch.setattr(client, "batch_update",
                        lambda fid, reqs, creds=None: len(reqs))
    return calls


def _record(models=1) -> dict:
    exams = [{
        "model_number": m,
        "questions": {"mcq": [{"question_id": f"model{m}_mcq_1",
                               "question": f"Q{m}?",
                               "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                               "correct_answer": "A"}]},
        "markdown": "", "warnings": [],
    } for m in range(1, models + 1)]
    return {"exams": exams, "warnings": []}


def test_share_called_per_model_per_email(fake_forms, monkeypatch):
    shared = []
    monkeypatch.setattr(
        client, "share_form",
        lambda form_id, email, role="writer", message=None, creds=None:
        shared.append((form_id, email, role)))
    result = exporter.export_exam(_record(2), share_with=["t@school.com"])
    assert result["errors"] == []
    # 2 models x 1 teacher
    assert sorted(shared) == [("form_1", "t@school.com", "writer"),
                              ("form_2", "t@school.com", "writer")]
    for exp in result["exports"]:
        assert exp["shared_with"] == ["t@school.com"]
        assert not [w for w in exp["warnings"] if "share" in w.lower()]


def test_share_failure_is_warning_not_error(fake_forms, monkeypatch):
    def fail_share(form_id, email, role="writer", message=None, creds=None):
        raise client.GoogleFormsError(f"Cannot share with {email}: denied.")

    monkeypatch.setattr(client, "share_form", fail_share)
    result = exporter.export_exam(_record(1),
                                  share_with=["teacher@school.com"])
    assert len(result["exports"]) == 1
    exp = result["exports"][0]
    # The form itself is still returned normally.
    assert exp["form_id"] == "form_1"
    assert exp["edit_url"] and exp["view_url"]
    assert any("share with teacher@school.com failed" in w
               for w in exp["warnings"])
    assert "shared_with" not in exp


def test_no_emails_no_sharing(fake_forms, monkeypatch):
    called = []

    def spy(form_id, email, **kw):
        called.append((form_id, email))

    monkeypatch.setattr(client, "share_form", spy)
    result = exporter.export_exam(_record(1))
    assert called == []
    assert all("shared_with" not in e for e in result["exports"])


# ------------------------------------------------------------- API layer

@pytest.fixture()
def api_client(monkeypatch, fake_forms):
    from app.api import exam_store
    from app import config as app_config

    # Feature-1 email sharing lives in CENTRAL mode; teacher mode ignores it.
    monkeypatch.setattr(app_config, "GOOGLE_FORMS_MODE", "central")
    exam_store.clear()
    exam_id = exam_store.save_exam(_record(2)["exams"], warnings=[])
    yield TestClient(__import__("app.api.main", fromlist=["app"]).app), exam_id
    exam_store.clear()


def test_route_accepts_and_passes_emails(api_client, monkeypatch):
    client_http, exam_id = api_client
    seen = {}

    def spy(form_id, email, role="writer", message=None, creds=None):
        seen.setdefault(email, []).append(form_id)

    monkeypatch.setattr(client, "share_form", spy)
    resp = client_http.post(f"/exams/{exam_id}/export/google-forms",
                            json={"share_with": ["teacher@school.com"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["exports"][0]["shared_with"] == ["teacher@school.com"]
    assert seen == {"teacher@school.com": ["form_1", "form_2"]}
    # No credentials or tokens leaked into the response.
    assert "token" not in resp.text.lower()
    assert "secret" not in resp.text.lower()


def test_route_rejects_invalid_email(api_client):
    client_http, exam_id = api_client
    resp = client_http.post(f"/exams/{exam_id}/export/google-forms",
                            json={"share_with": ["not-an-email"]})
    assert resp.status_code == 422


def test_route_backward_compatible_empty_body(api_client, monkeypatch):
    client_http, exam_id = api_client
    monkeypatch.setattr(client, "share_form",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not share")))
    resp = client_http.post(f"/exams/{exam_id}/export/google-forms", json={})
    assert resp.status_code == 200
