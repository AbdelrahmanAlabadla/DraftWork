from __future__ import annotations

from types import SimpleNamespace

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
                '"correct_answer":"A"}]}',
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

    assert client._reasoning_effort("planner") == "high"
    assert client._reasoning_effort("validator") == "high"
    assert client._reasoning_effort("generator") == "none"
    assert client._reasoning_effort("repairer") == "none"
    assert client._token_budget("planner", 200, "default") == (32000, 32000)
    assert client._token_budget("generator", 30, "default") == (8000, 8000)
    assert client._token_budget("validator", 30, "default") == (15360, 32000)
    assert client._token_budget("repairer", 30, "default") == (15360, 16000)
    assert client._token_budget("generator", 10, "fitb_word_bank") == (1024, 1024)
    assert client._token_budget("generator", 10, "fitb_items") == (3840, 8000)
