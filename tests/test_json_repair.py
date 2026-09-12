from __future__ import annotations

import pytest

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.llm.client import LMStudioClient  # noqa: E402
from app.llm.json_utils import (  # noqa: E402
    JSONExtractionError,
    REPAIR_SYSTEM_PROMPT,
    build_repair_prompt,
    extract_json,
)

_BROKEN = '{"questions": [{"statement": "A", "answer": "True"} {"statement": "B", "answer": "False"}]}'
_VALID = '{"questions": [{"statement": "A", "answer": "True"}, {"statement": "B", "answer": "False"}]}'


class FakeChat:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str | None, float]] = []

    def __call__(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, timeout=600):
        self.calls.append((prompt, system_prompt, temperature))
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError("No more fake responses")


def _make_client(chat_fn) -> LMStudioClient:
    client = LMStudioClient.__new__(LMStudioClient)
    client.url = "http://fake"
    client.model = "fake-model"
    client.chat = chat_fn
    return client


def test_structured_output_uses_lm_studio_json_schema_endpoint(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"value":"ok"}'}}]}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, payload=json, headers=headers, timeout=timeout)
        return Response()

    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    client = LMStudioClient(url="http://localhost:1234", model="local-model")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    result = client.chat_structured(
        "return a value", json_schema=schema, schema_name="test schema"
    )
    assert result == {"value": "ok"}
    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    response_format = captured["payload"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"] == {
        "name": "test_schema",
        "strict": True,
        "schema": schema,
    }


def test_build_repair_prompt_contains_error_and_broken():
    prompt = build_repair_prompt("{}", "Expecting ',' delimiter")
    assert "Expecting ',' delimiter" in prompt
    assert "{}" in prompt
    assert "corrected raw JSON" in prompt


def test_chat_json_repairs_on_first_parse_failure():
    chat = FakeChat([_BROKEN, _VALID])
    client = _make_client(chat)

    result = client.chat_json("gen", system_prompt="gen-sys", temperature=0.7, max_tokens=512)

    assert result == {"questions": [
        {"statement": "A", "answer": "True"},
        {"statement": "B", "answer": "False"},
    ]}
    assert len(chat.calls) == 2
    # First call: generation prompt + caller temperature.
    assert chat.calls[0][0] == "gen"
    assert chat.calls[0][1] == "gen-sys"
    assert chat.calls[0][2] == 0.7
    # Second call: repair system prompt + temperature 0.0.
    assert "Parser error" in chat.calls[1][0]
    assert chat.calls[1][1] == REPAIR_SYSTEM_PROMPT
    assert chat.calls[1][2] == 0.0


def test_chat_json_returns_valid_json_without_repair():
    chat = FakeChat([_VALID])
    client = _make_client(chat)

    result = client.chat_json("gen", temperature=0.7, max_tokens=512)

    assert result == {"questions": [
        {"statement": "A", "answer": "True"},
        {"statement": "B", "answer": "False"},
    ]}
    assert len(chat.calls) == 1


def test_chat_json_raises_after_exhausting_repairs():
    chat = FakeChat([_BROKEN, _BROKEN, _BROKEN])
    client = _make_client(chat)

    with pytest.raises(JSONExtractionError):
        client.chat_json("gen", temperature=0.7, max_tokens=512)

    # 1 generation call + 2 repair attempts.
    assert len(chat.calls) == 3


def test_chat_json_returns_parsed_object_not_text():
    chat = FakeChat(['{"answer": 42}'])
    client = _make_client(chat)
    assert client.chat_json("gen", max_tokens=512) == {"answer": 42}


def test_extract_json_parses_valid():
    assert extract_json(_VALID) == {"questions": [
        {"statement": "A", "answer": "True"},
        {"statement": "B", "answer": "False"},
    ]}
