from __future__ import annotations

import asyncio
import socket

import pytest

from app.llm.client import LMStudioClient
from app.llm.deepseek import DeepSeekClient
from app.llm.openai_client import OpenAIClient
from tests.offline_provider_guard import UnmockedProviderCall


def _uninitialized(cls):
    return object.__new__(cls)


@pytest.mark.parametrize("client", [
    _uninitialized(DeepSeekClient),
    _uninitialized(OpenAIClient),
])
def test_unmocked_hosted_provider_clients_fail_closed(client):
    with pytest.raises(UnmockedProviderCall, match="UNMOCKED_PROVIDER_CALL"):
        asyncio.run(client._sdk_request(timeout=1))


def test_unmocked_async_provider_path_fails_closed():
    client = _uninitialized(DeepSeekClient)
    with pytest.raises(UnmockedProviderCall, match="No external request was sent"):
        asyncio.run(client._sdk_request(timeout=1))


def test_unmocked_local_provider_client_fails_closed():
    client = LMStudioClient(url="http://127.0.0.1:1234", model="offline-test")
    with pytest.raises(UnmockedProviderCall, match="UNMOCKED_PROVIDER_CALL"):
        client.chat("must not leave the process")


def test_low_level_provider_network_path_is_blocked():
    with pytest.raises(UnmockedProviderCall, match="LLM network endpoint"):
        socket.create_connection(("api.deepseek.com", 443), timeout=0.01)


def test_explicit_mock_still_overrides_guard(monkeypatch):
    client = _uninitialized(DeepSeekClient)
    async def mocked(**_kwargs):
        return "mocked"
    monkeypatch.setattr(client, "_sdk_request", mocked)
    assert asyncio.run(client._sdk_request(timeout=1)) == "mocked"
