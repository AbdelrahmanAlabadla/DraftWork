from __future__ import annotations

import os

import pytest

from tests.offline_provider_guard import install


# Existing API regression tests exercise the old synchronous compatibility
# routes. Product/browser traffic uses the queued /api/v1 contract.
os.environ.setdefault("ENABLE_LEGACY_SYNC_API", "true")


@pytest.fixture(autouse=True)
def block_unmocked_llm_providers(monkeypatch):
    """Offline pytest is fail-closed; individual mocks may override the guard."""
    install(monkeypatch)
