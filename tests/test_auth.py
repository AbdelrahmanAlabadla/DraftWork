from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from clerk_backend_api.security.types import AuthStatus
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.session_middleware import _needs_session
from app.auth import clerk as clerk_auth


def test_auth_config_exposes_only_browser_safe_values(monkeypatch):
    monkeypatch.setattr("app.config.CLERK_PUBLISHABLE_KEY", "pk_test_browser")
    monkeypatch.setattr("app.config.CLERK_SECRET_KEY", "server-test-secret")
    response = TestClient(app).get("/api/v1/auth/config")
    assert response.status_code == 200
    assert response.json()["publishable_key"] == "pk_test_browser"
    assert "secret" not in response.text.lower()
    assert "server-test-secret" not in response.text


def test_clerk_identity_uses_verified_subject(monkeypatch):
    class FakeClient:
        async def authenticate_request_async(self, request, options):
            return SimpleNamespace(
                status=AuthStatus.SIGNED_IN,
                payload={"sub": "user_clerk_123", "email": "untrusted@example.test"},
                reason=None,
            )

    monkeypatch.setattr(clerk_auth, "Clerk", lambda **kwargs: FakeClient())
    monkeypatch.setattr(clerk_auth.config, "CLERK_SECRET_KEY", "server-secret")
    request = SimpleNamespace(headers={"Authorization": "Bearer token"})
    identity = asyncio.run(clerk_auth.authenticate(request))
    assert identity is not None
    assert identity.clerk_user_id == "user_clerk_123"


def test_missing_bearer_token_remains_anonymous():
    request = SimpleNamespace(headers={})
    assert asyncio.run(clerk_auth.authenticate(request)) is None


def test_global_eval_dashboard_does_not_participate_in_user_sessions():
    assert not _needs_session("/api/v1/eval-summary")
    assert not _needs_session("/api/v1/api/eval-summary")
    assert _needs_session("/api/v1/documents")


def test_static_auth_pages_use_clerk_without_token_storage():
    auth_js = open("FrontEnd/js/auth.js", encoding="utf-8").read()
    assert "session.getToken()" in auth_js
    assert "localStorage" not in auth_js
    assert "sessionStorage" not in auth_js
    assert "CLERK_SECRET_KEY" not in auth_js


def test_auth_pages_and_account_controls_are_present():
    sign_in = open("FrontEnd/sign-in.html", encoding="utf-8").read()
    sign_up = open("FrontEnd/sign-up.html", encoding="utf-8").read()
    home = open("FrontEnd/index.html", encoding="utf-8").read()
    assert "Continue with Google" in sign_in
    assert "email and password" in sign_in
    assert "Use Google" in sign_up
    assert 'id="clerkAuth"' in sign_in and 'id="clerkAuth"' in sign_up
    assert 'id="clerkUserButton"' in home
    assert 'id="authSignedOut" hidden' in home
    assert 'id="authSignedIn" hidden' in home
    assert 'id="logoutButton"' not in home
    assert 'href="/my-exams.html"' not in home
    assert 'id="myExamsDialog"' in home
    assert 'id="myExamsList"' in home
    auth_js = open("FrontEnd/js/auth.js", encoding="utf-8").read()
    assert 'label: "manageAccount"' in auth_js
    assert 'label: "My Exams"' in auth_js
    assert 'CustomEvent("draftwork:open-my-exams")' in auth_js
    assert 'label: "signOut"' in auth_js
    dialog_js = open("FrontEnd/js/my-exams-dialog.js", encoding="utf-8").read()
    assert 'document.addEventListener("draftwork:open-my-exams"' in dialog_js
    assert "loadExamIntoPreview(exam.exam_id)" in dialog_js
    assert 't("saved.none")' in dialog_js
    i18n_js = open("FrontEnd/js/i18n.js", encoding="utf-8").read()
    assert '"saved.none": "There is no exam made yet."' in i18n_js
    assert '"saved.none": "لا توجد امتحانات منشأة بعد."' in i18n_js
    assert not Path("FrontEnd/my-exams.html").exists()
    assert not Path("FrontEnd/js/my-exams.js").exists()


def test_clerk_loading_does_not_block_main_ui_boot():
    main = open("FrontEnd/js/main.js", encoding="utf-8").read()
    assert "await initAuthNavigation()" not in main
    assert "initAuthNavigation().catch" in main
    assert "window.__dwBooted = true" in main
