from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, BadRequestError, RateLimitError

from app import config
from app.llm import deepseek
from app.llm.openai_client import (
    OpenAIClient,
    OpenAINetworkError,
    OpenAIQuotaError,
    OpenAIRateLimitError,
    OpenAISchemaError,
    OpenAITimeoutError,
    OpenAIValidationError,
)
from app.llm.schemas import (
    SpecializedValidatorResponse,
    repair_response_model,
    strict_json_schema,
)
from app.online.duplicate_judge import duplicate_response_model


def _response(output: dict | str):
    text = output if isinstance(output, str) else json.dumps(output)
    return SimpleNamespace(
        status="completed", output_text=text, error=None,
        incomplete_details=None, usage=None,
    )


def _openai(monkeypatch) -> OpenAIClient:
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    return OpenAIClient(model="gpt-6-luna")


def _deepseek(monkeypatch) -> deepseek.DeepSeekClient:
    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    return deepseek.DeepSeekClient(model="deepseek-flash")


def _call(client, model, output, monkeypatch, *, agent="validator"):
    async def request(**_kwargs):
        return _response(output)

    monkeypatch.setattr(client, "_request_with_backoff", request)
    return client.chat_structured(
        "validate", response_model=model, json_schema=model.model_json_schema(),
        schema_name="production_contract", agent=agent, item_count=1,
    )


@pytest.mark.parametrize("factory", [_openai, _deepseek])
def test_valid_production_validator_contract_parses_for_each_ordinary_role(monkeypatch, factory):
    client = factory(monkeypatch)
    for role in (
        "grounding", "task_alignment", "answer_correctness", "difficulty", "quantitative",
    ):
        row = {
            "question_id": "q1", "status": "PASS", "issues": [],
            "grounding_outcome": "SUPPORTED" if role == "grounding" else None,
            "observed_difficulty": "medium" if role == "difficulty" else None,
            "calculation_evidence": ["2 + 2 = 4"] if role == "quantitative" else [],
            "original_issue_checks": [],
        }
        result = _call(
            client, SpecializedValidatorResponse,
            {"validator": role, "verdicts": [row]}, monkeypatch,
        )
        assert result["validator"] == role
        assert result["verdicts"][0]["question_id"] == "q1"


@pytest.mark.parametrize("factory", [_openai, _deepseek])
def test_duplicate_v2_uses_its_independent_production_contract(monkeypatch, factory):
    client = factory(monkeypatch)
    model = duplicate_response_model(["q1::q2"])
    result = _call(client, model, {"verdicts": [{
        "pair_id": "q1::q2", "same_core_knowledge": "yes",
        "same_student_operation_or_method": "no", "same_target_or_result": "no",
        "genuinely_new_reasoning": "yes",
        "one_answer_substantially_gives_away_other": "no",
    }]}, monkeypatch, agent="duplicate_validator")
    assert result["verdicts"][0]["pair_id"] == "q1::q2"


@pytest.mark.parametrize("factory", [_openai, _deepseek])
def test_typed_mixed_repair_batch_parses_at_each_provider_boundary(
    monkeypatch, factory,
):
    client = factory(monkeypatch)
    model = repair_response_model([
        ("mcq_q1", "mcq", ["correct_answer"]),
        ("equation_q1", "equation", ["final_answer"]),
    ], include_metadata=True)
    output = {"repairs": [
        {
            "question_id": "mcq_q1",
            "patch": {"correct_answer": "A"},
            "repair_metadata": {
                "modified_fields": ["correct_answer"],
                "changes": [{
                    "field": "correct_answer", "previous_value": "B",
                    "new_value": "A", "reason": "Corrects the keyed answer.",
                }],
                "targeted_failure_codes": ["incorrect_answer"],
            },
        },
        {
            "question_id": "equation_q1",
            "patch": {"final_answer": "x = 3"},
            "repair_metadata": {
                "modified_fields": ["final_answer"],
                "changes": [{
                    "field": "final_answer", "previous_value": "x = 2",
                    "new_value": "x = 3", "reason": "Matches the solution.",
                }],
                "targeted_failure_codes": ["incorrect_result"],
            },
        },
    ]}

    result = _call(client, model, output, monkeypatch, agent="answer_repairer")

    assert [row["question_id"] for row in result["repairs"]] == [
        "mcq_q1", "equation_q1",
    ]


def test_validator_schema_contains_no_untyped_properties():
    schema = strict_json_schema(SpecializedValidatorResponse)

    def untyped(node):
        if isinstance(node, dict):
            if node == {}:
                return True
            return any(untyped(value) for value in node.values())
        if isinstance(node, list):
            return any(untyped(value) for value in node)
        return False

    assert not untyped(schema)


@pytest.mark.parametrize(
    ("role", "qtype", "fields"),
    [
        ("task_alignment", "word_problem", ["question"]),
        ("question", "true_false", ["statement"]),
        ("options", "mcq", ["options"]),
        ("solution", "equation", ["solution_steps", "final_answer"]),
        ("answer", "mcq", ["correct_answer"]),
        ("difficulty", "why", ["question"]),
        ("duplicate", "definition", ["term"]),
    ],
)
def test_every_specialized_repair_role_has_typed_strict_schema(
    role, qtype, fields,
):
    model = repair_response_model(
        [(f"{role}_q1", qtype, fields)],
        include_metadata=True,
    )
    schema = strict_json_schema(model)

    def empty_nodes(node, path="$"):
        found = []
        if isinstance(node, dict):
            if node == {}:
                found.append(path)
            for key, value in node.items():
                found.extend(empty_nodes(value, f"{path}.{key}"))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                found.extend(empty_nodes(value, f"{path}[{index}]"))
        return found

    assert empty_nodes(schema) == []
    assert schema["properties"]["repairs"]["items"]


@pytest.mark.parametrize(
    ("factory", "expected"),
    [(_openai, OpenAIValidationError), (_deepseek, deepseek.DeepSeekValidationError)],
)
@pytest.mark.parametrize(
    "bad_output",
    [
        "not-json",
        {"validator": "grounding", "verdicts": [{"question_id": "q1"}]},
        {"validator": "grounding", "verdicts": [{
            "question_id": "q1", "status": 1, "issues": [],
            "grounding_outcome": None, "observed_difficulty": None,
            "calculation_evidence": [], "original_issue_checks": [],
        }]},
    ],
)
def test_invalid_json_missing_fields_and_wrong_types_are_provider_specific(
    monkeypatch, factory, expected, bad_output,
):
    client = factory(monkeypatch)
    with pytest.raises(expected):
        _call(client, SpecializedValidatorResponse, bad_output, monkeypatch)


def test_openai_schema_rejection_is_not_reported_as_deepseek(monkeypatch):
    client = _openai(monkeypatch)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(400, request=request)

    async def reject(**_kwargs):
        raise BadRequestError("Invalid schema for response_format", response=response, body={})

    monkeypatch.setattr(client, "_request_with_backoff", reject)
    with pytest.raises(OpenAISchemaError, match="OpenAI rejected"):
        client.chat_structured(
            "validate", response_model=SpecializedValidatorResponse,
            json_schema=SpecializedValidatorResponse.model_json_schema(),
            schema_name="validator", agent="validator", item_count=1,
        )


def test_deepseek_schema_rejection_keeps_deepseek_identity(monkeypatch):
    client = _deepseek(monkeypatch)
    request = httpx.Request("POST", "https://api.deepseek.com/responses")
    response = httpx.Response(400, request=request)

    async def reject(**_kwargs):
        raise BadRequestError("Invalid schema for response_format", response=response, body={})

    monkeypatch.setattr(client, "_request_with_backoff", reject)
    with pytest.raises(deepseek.DeepSeekSchemaError, match="DeepSeek rejected"):
        client.chat_structured(
            "validate", response_model=SpecializedValidatorResponse,
            json_schema=SpecializedValidatorResponse.model_json_schema(),
            schema_name="validator", agent="validator", item_count=1,
        )


@pytest.mark.parametrize(
    ("factory", "sdk_error", "expected"),
    [
        (_openai, APITimeoutError, OpenAITimeoutError),
        (_openai, APIConnectionError, OpenAINetworkError),
        (_deepseek, APITimeoutError, deepseek.DeepSeekTimeoutError),
        (_deepseek, APIConnectionError, deepseek.DeepSeekNetworkError),
    ],
)
def test_retryable_transport_errors_keep_provider_identity(
    monkeypatch, factory, sdk_error, expected,
):
    client = factory(monkeypatch)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def fail(**_kwargs):
        raise sdk_error(request=request)

    monkeypatch.setattr(client, "_sdk_request", fail)
    monkeypatch.setattr(deepseek, "DEEPSEEK_MAX_TRANSIENT_RETRIES", 0)
    with pytest.raises(expected):
        asyncio.run(client._request_with_backoff(timeout=1, input="test"))


@pytest.mark.parametrize(
    ("factory", "quota_type", "rate_type"),
    [
        (_openai, OpenAIQuotaError, OpenAIRateLimitError),
        (_deepseek, deepseek.DeepSeekQuotaError, deepseek.DeepSeekRateLimitError),
    ],
)
def test_429_quota_and_temporary_rate_limit_are_distinct(
    monkeypatch, factory, quota_type, rate_type,
):
    client = factory(monkeypatch)
    client._quota_blocked_accounts.clear()
    monkeypatch.setattr(deepseek, "DEEPSEEK_MAX_TRANSIENT_RETRIES", 0)
    request = httpx.Request("POST", "https://provider.test/v1/responses")

    async def insufficient(**_kwargs):
        response = httpx.Response(
            429, request=request,
            headers={"x-request-id": "req_quota"},
        )
        raise RateLimitError(
            "quota", response=response,
            body={"error": {
                "code": "insufficient_quota",
                "message": "You have no credits remaining.",
            }},
        )

    monkeypatch.setattr(client, "_sdk_request", insufficient)
    with pytest.raises(quota_type) as quota:
        asyncio.run(client._request_with_backoff(timeout=1, input="test"))
    assert quota.value.normalized_category == "INSUFFICIENT_QUOTA"
    assert quota.value.provider_code == "insufficient_quota"
    assert quota.value.provider_request_id == "req_quota"

    # The confirmed account circuit stops a second provider request.
    calls = {"count": 0}
    async def should_not_run(**_kwargs):
        calls["count"] += 1
    monkeypatch.setattr(client, "_sdk_request", should_not_run)
    with pytest.raises(quota_type):
        asyncio.run(client._request_with_backoff(timeout=1, input="test"))
    assert calls["count"] == 0

    client._quota_blocked_accounts.clear()
    async def throttled(**_kwargs):
        response = httpx.Response(
            429, request=request,
            headers={"retry-after": "0", "x-request-id": "req_rate"},
        )
        raise RateLimitError(
            "rate", response=response,
            body={"error": {"code": "rate_limit_exceeded", "message": "TPM exceeded"}},
        )

    monkeypatch.setattr(client, "_sdk_request", throttled)
    with pytest.raises(rate_type) as limited:
        asyncio.run(client._request_with_backoff(timeout=1, input="test"))
    assert limited.value.normalized_category == "RATE_LIMITED"
    assert limited.value.provider_code == "rate_limit_exceeded"
    assert limited.value.provider_request_id == "req_rate"
    client._quota_blocked_accounts.clear()
