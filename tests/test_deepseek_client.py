from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

from app.llm import deepseek
from app.llm.schemas import generation_response_model


def _response(status: str, output_text: str, incomplete_reason: str | None = None):
    return SimpleNamespace(
        status=status,
        output_text=output_text,
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
        ),
        error=None,
        usage=SimpleNamespace(
            input_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
            output_tokens=5,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            total_tokens=15,
        ),
    )


def _client(monkeypatch) -> deepseek.DeepSeekClient:
    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    return deepseek.DeepSeekClient()


def test_structured_retry_keeps_task_schema_and_id(monkeypatch):
    client = _client(monkeypatch)
    calls = []
    responses = iter(
        [
            _response("incomplete", '{"questions":[', "max_output_tokens"),
            _response(
                "completed",
                '{"questions":[{"slot_id":"m1_mcq_1","question":"Q"}]}',
            ),
            _response(
                "completed",
                '{"questions":[{"slot_id":"m1_mcq_1","question":"Q",'
                '"options":{"A":"a","B":"b","C":"c","D":"d"},'
                '"correct_answer":"A"}],"unsuitable_tasks":[]}',
            ),
        ]
    )

    async def fake_request_with_backoff(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(client, "_request_with_backoff", fake_request_with_backoff)
    model = generation_response_model("mcq", ["m1_mcq_1"])
    result = client.chat_structured(
        "Generate the planned MCQ.",
        response_model=model,
        json_schema=model.model_json_schema(),
        schema_name="mcq_questions",
        agent="generator",
        item_count=1,
        expected_ids=["m1_mcq_1"],
        expected_id_field="slot_id",
    )

    assert result["questions"][0]["slot_id"] == "m1_mcq_1"
    assert len(calls) == 3
    assert [call["max_output_tokens"] for call in calls] == [2048, 4096, 4096]
    assert all(call["reasoning"] == {"effort": "none"} for call in calls)
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["text"]["format"]["schema"]["additionalProperties"] is False
    assert "Previous response failed because" in calls[1]["input"]
    assert "m1_mcq_1" in calls[1]["input"]
    assert "field required" in calls[2]["input"].lower()


def test_agent_reasoning_and_token_budgets_are_deepseek_specific(monkeypatch):
    client = _client(monkeypatch)

    assert client._reasoning_effort("planner") == "low"
    assert client._reasoning_effort("validator") == "high"
    assert client._reasoning_effort("duplicate_judge") == "none"
    assert client._reasoning_effort("global_validator_initial") == "none"
    assert client._reasoning_effort("generator") == "none"
    assert client._reasoning_effort("repairer") == "low"
    assert client._token_budget("planner", 200, "default") == (32000, 32000)
    assert client._token_budget("generator", 30, "default") == (8000, 8000)
    assert client._token_budget("validator", 30, "default") == (15360, 32000)
    assert client._token_budget("repairer", 30, "default") == (15360, 16000)
    assert client._token_budget("generator", 10, "fitb_word_bank") == (1024, 1024)
    assert client._token_budget("generator", 10, "fitb_items") == (3840, 8000)


def test_initial_planner_truncation_does_not_generic_retry(monkeypatch):
    client = _client(monkeypatch)
    calls = []

    async def fake_request_with_backoff(**kwargs):
        calls.append(kwargs)
        return _response("incomplete", '{"assessment_profile":{}}', "max_output_tokens")

    monkeypatch.setattr(client, "_request_with_backoff", fake_request_with_backoff)
    model = generation_response_model("mcq", ["m1_mcq_1"])
    with pytest.raises(deepseek.DeepSeekIncompleteError) as caught:
        client.chat_structured(
            "Plan once.", response_model=model, json_schema=model.model_json_schema(),
            schema_name="plan", agent="planner", operation="initial_planner",
            item_count=1, max_tokens=25000,
        )
    assert len(calls) == 1
    assert calls[0]["max_output_tokens"] == 25000
    assert caught.value.partial_output == '{"assessment_profile":{}}'


def test_initial_planner_malformed_completed_json_exposes_checkpoint(monkeypatch):
    client = _client(monkeypatch)
    partial = '{"questions":[{"slot_id":"m1_mcq_1"}'

    async def fake_request_with_backoff(**_kwargs):
        return _response("completed", partial)

    monkeypatch.setattr(client, "_request_with_backoff", fake_request_with_backoff)
    model = generation_response_model("mcq", ["m1_mcq_1"])
    with pytest.raises(deepseek.DeepSeekValidationError) as caught:
        client.chat_structured(
            "Plan once.", response_model=model, json_schema=model.model_json_schema(),
            schema_name="plan", agent="planner", operation="initial_planner",
            item_count=1, max_tokens=25000,
        )
    assert caught.value.partial_output == partial
    assert caught.value.reason == "schema_validation"


def test_transient_network_failure_retries_only_the_failed_call_twice(monkeypatch):
    client = _client(monkeypatch)
    calls = []
    request = httpx.Request("POST", "https://api.deepseek.com/responses")

    async def fail_request(**_kwargs):
        calls.append("failed")
        raise APITimeoutError(request=request)

    monkeypatch.setattr(client, "_sdk_request", fail_request)
    monkeypatch.setattr(deepseek, "DEEPSEEK_MAX_TRANSIENT_RETRIES", 2)
    monkeypatch.setattr(deepseek, "DEEPSEEK_RETRY_BASE_SECONDS", 0)

    with pytest.raises(deepseek.DeepSeekTimeoutError):
        asyncio.run(client._request_with_backoff(timeout=1, input="test"))

    assert calls == ["failed", "failed", "failed"]  # initial + two retries
