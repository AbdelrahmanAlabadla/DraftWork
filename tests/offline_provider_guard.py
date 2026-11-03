from __future__ import annotations

import socket
import ipaddress
from urllib.parse import urlparse


class UnmockedProviderCall(BaseException):
    pass


def _blocked(provider: str, detail: str = "") -> UnmockedProviderCall:
    suffix = f" ({detail})" if detail else ""
    return UnmockedProviderCall(
        "UNMOCKED_PROVIDER_CALL: "
        f"Attempted to invoke {provider} during offline test execution{suffix}. "
        "No external request was sent."
    )


def install(monkeypatch) -> None:
    """Fail closed at concrete provider and network boundaries for every test."""
    from app import config
    from app.llm.client import LMStudioClient
    from app.llm.deepseek import DeepSeekClient

    def deny_async(provider: str):
        async def denied(*_args, **_kwargs):
            raise _blocked(provider)
        return denied

    # OpenAIClient inherits this actual SDK boundary from DeepSeekClient. Tests
    # that explicitly mock _sdk_request or _request_with_backoff keep working.
    monkeypatch.setattr(
        DeepSeekClient, "_sdk_request", deny_async("hosted LLM provider")
    )

    provider_hosts = {"api.deepseek.com", "api.openai.com"}
    provider_endpoints: set[tuple[str, int | None]] = set()
    for raw in (
        getattr(config, "DEEPSEEK_BASE_URL", ""),
        getattr(config, "OPENAI_BASE_URL", ""),
        getattr(config, "LMS_URL", ""),
    ):
        parsed = urlparse(str(raw or ""))
        if parsed.hostname:
            host = parsed.hostname.casefold()
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = host == "localhost"
            if not loopback:
                provider_hosts.add(host)
            provider_endpoints.add((host, parsed.port))

    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection
    import requests
    original_request = requests.sessions.Session.request

    def forbidden(address) -> bool:
        if not isinstance(address, tuple) or not address:
            return False
        host = str(address[0]).casefold()
        port = int(address[1]) if len(address) > 1 and address[1] is not None else None
        return host in provider_hosts or (host, port) in provider_endpoints

    def guarded_connect(sock, address):
        if forbidden(address):
            raise _blocked("LLM network endpoint", str(address[0]))
        return original_connect(sock, address)

    def guarded_create_connection(address, *args, **kwargs):
        if forbidden(address):
            raise _blocked("LLM network endpoint", str(address[0]))
        return original_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)

    def guarded_request(session, method, url, *args, **kwargs):
        parsed = urlparse(str(url))
        endpoint = (str(parsed.hostname or "").casefold(), parsed.port)
        if endpoint[0] in provider_hosts or endpoint in provider_endpoints:
            raise _blocked("LLM network endpoint", endpoint[0])
        return original_request(session, method, url, *args, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", guarded_request)
