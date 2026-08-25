"""Teacher-owned Google Forms: sign-in flow, sessions, isolation, modes."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import session as session_mod
from app.integrations.google_forms import client, exporter, user_tokens


@pytest.fixture(autouse=True)
def _clean():
    user_tokens.clear()
    yield
    user_tokens.clear()


@pytest.fixture()
def api(monkeypatch):
    from app.api.main import app

    monkeypatch.setattr("app.config.GOOGLE_FORMS_MODE", "teacher")
    return TestClient(app)


def _store_teacher(sub="sub-A", email="a@gmail.com"):
    class FakeCreds:
        valid = True

    user_tokens.save(sub, FakeCreds(), email)
    return sub, email


# ------------------------------------------------------- identity / session

def test_session_cookie_roundtrip():
    val = session_mod.issue_cookie_value("sub-1", "t@gmail.com")
    assert session_mod.read_cookie_value(val) == {"sub": "sub-1",
                                                  "email": "t@gmail.com"}


def test_forged_session_rejected():
    assert session_mod.read_cookie_value("garbage.sig") is None
    assert session_mod.read_cookie_value(None) is None
    # Tampered payload without a valid signature must not decode.
    assert session_mod.read_cookie_value("sub-1|attacker@evil.com") is None


def test_state_mismatch_rejected(api, monkeypatch):
    import app.api.auth_routes as ar
    monkeypatch.setattr(ar, "web_configured", lambda: True)
    resp = api.get("/auth/google/callback",
                   params={"code": "abc", "state": "not-issued"})
    assert resp.status_code == 400


def test_callback_verifies_identity_and_sets_session(api, monkeypatch):
    import app.api.auth_routes as ar

    state = ar._new_state()

    class FakeCreds:
        valid = True

    def fake_exchange(code):
        assert code == "auth-code-123"
        return FakeCreds(), {"sub": "sub-T1", "email": "teacher@gmail.com"}

    monkeypatch.setattr(ar, "exchange_code", fake_exchange)
    monkeypatch.setattr(ar, "web_configured", lambda: True)

    resp = api.get("/auth/google/callback",
                   params={"code": "auth-code-123", "state": state},
                   follow_redirects=False)
    assert resp.status_code == 303
    set_cookie = resp.headers["set-cookie"]
    assert session_mod.COOKIE_NAME in set_cookie
    assert "httponly" in set_cookie.lower()
    # Credentials stored under google_sub.
    assert user_tokens.email_of("sub-T1") == "teacher@gmail.com"
    from http.cookies import SimpleCookie
    sc = SimpleCookie()
    sc.load(resp.headers["set-cookie"])
    cookie_val = sc[session_mod.COOKIE_NAME].value
    assert session_mod.read_cookie_value(cookie_val)["sub"] == "sub-T1"


def test_logout_invalidates_session_and_credentials(api):
    sub, email = _store_teacher()
    cookie = session_mod.issue_cookie_value(sub, email)

    resp = api.post("/auth/google/logout",
                    cookies={session_mod.COOKIE_NAME: cookie})
    assert resp.status_code == 200 and resp.json() == {"connected": False}
    with pytest.raises(user_tokens.NotConnected):
        user_tokens.get(sub)
    me = api.get("/auth/google/me", cookies={session_mod.COOKIE_NAME: cookie})
    assert me.json()["connected"] is False


def test_me_reports_connection(api):
    sub, email = _store_teacher()
    cookie = session_mod.issue_cookie_value(sub, email)
    me = api.get("/auth/google/me", cookies={session_mod.COOKIE_NAME: cookie})
    assert me.json() == {"connected": True, "email": email}


def test_isolation_teacher_b_cannot_use_a_credentials():
    _store_teacher("sub-A", "a@gmail.com")
    with pytest.raises(user_tokens.NotConnected):
        user_tokens.get("sub-B")


# ------------------------------------------------------ export mode behavior

@pytest.fixture()
def stored_exam_id():
    from app.api import exam_store

    exam_store.clear()
    exams = [{
        "model_number": 1,
        "questions": {"mcq": [{"question_id": "model1_mcq_1",
                               "question": "Q?",
                               "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                               "correct_answer": "A"}]},
        "markdown": "", "warnings": [],
    }]
    return exam_store.save_exam(exams)


def _mock_forms(monkeypatch, capture):
    from app.integrations.google_forms import web_auth as _wa
    monkeypatch.setattr(_wa, "web_configured", lambda: True)
    def fake_create(title, creds=None):
        capture["creds"] = creds
        return {"form_id": "form_T1", "title": title,
                "edit_url": "https://edit", "view_url": "https://view"}

    def fake_batch(form_id, reqs, creds=None):
        capture["batch_creds"] = creds
        return len(reqs)

    def fail_share(*a, **k):
        raise AssertionError("permissions.create must not be called in teacher mode")

    monkeypatch.setattr(client, "create_form", fake_create)
    monkeypatch.setattr(client, "batch_update", fake_batch)
    monkeypatch.setattr(client, "share_form", fail_share)


def test_teacher_mode_uses_teacher_credentials_and_no_share(
        api, stored_exam_id, monkeypatch):
    sub, email = _store_teacher("sub-OWN", "owner@gmail.com")
    cookie = session_mod.issue_cookie_value(sub, email)

    class Marker:
        valid = True

    marker_creds = Marker()

    class SavedCreds:
        valid = True

    # Replace stored creds with a recognizable object.
    user_tokens.save(sub, marker_creds, email)

    capture = {}
    _mock_forms(monkeypatch, capture)

    resp = api.post(f"/exams/{stored_exam_id}/export/google-forms", json={},
                    cookies={session_mod.COOKIE_NAME: cookie})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["exports"][0]["owner"] == "teacher"
    # Exporter received the TEACHER's credentials.
    assert capture["creds"] is marker_creds
    assert capture["batch_creds"] is marker_creds


def test_not_signed_in_returns_401(api, stored_exam_id, monkeypatch):
    monkeypatch.setattr("app.config.GOOGLE_FORMS_MODE", "teacher")
    from app.integrations.google_forms import web_auth as wa

    monkeypatch.setattr(wa, "web_configured", lambda: True)
    resp = api.post(f"/exams/{stored_exam_id}/export/google-forms", json={})
    assert resp.status_code == 401
    assert "Connect Google Account first" in resp.json()["detail"]


def test_frontend_email_cannot_override_session_identity(
        api, stored_exam_id, monkeypatch):
    """In teacher mode any share_with in the body is ignored entirely."""
    sub, email = _store_teacher("sub-X", "real@gmail.com")
    cookie = session_mod.issue_cookie_value(sub, email)
    capture = {}
    _mock_forms(monkeypatch, capture)

    resp = api.post(f"/exams/{stored_exam_id}/export/google-forms",
                    json={"share_with": ["someone@else.com"]},
                    cookies={session_mod.COOKIE_NAME: cookie})
    assert resp.status_code == 200
    assert "shared_with" not in resp.json()["exports"][0]


def test_missing_web_config_distinct_error(api, stored_exam_id,
                                           monkeypatch):
    # Default environment has no Web OAuth client configured.
    from app.integrations.google_forms import web_auth as wa
    monkeypatch.setattr(wa, "web_configured", lambda: False)

    resp = api.post(f"/exams/{stored_exam_id}/export/google-forms", json={})
    assert resp.status_code == 503
    assert "Web OAuth client" in resp.json()["detail"]


def test_central_mode_still_shares_explicit_user_only(
        api, stored_exam_id, monkeypatch):
    from app import config as app_config

    monkeypatch.setattr(app_config, "GOOGLE_FORMS_MODE", "central")
    shared = []
    monkeypatch.setattr(client, "create_form",
                        lambda title, creds=None: {
                            "form_id": "f1", "title": title,
                            "edit_url": "e", "view_url": "v"})
    monkeypatch.setattr(client, "batch_update", lambda fid, r, creds=None: len(r))
    monkeypatch.setattr(client, "share_form",
                        lambda form_id, email, role="writer", message=None,
                        creds=None: shared.append((email, role)))

    resp = api.post(f"/exams/{stored_exam_id}/export/google-forms",
                    json={"share_with": ["teacher@gmail.com"]})
    assert resp.status_code == 200
    assert shared == [("teacher@gmail.com", "writer")]
