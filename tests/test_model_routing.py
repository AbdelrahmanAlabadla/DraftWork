from __future__ import annotations

import pytest

from app import config
from app.llm.deepseek import DeepSeekClient
from app.llm.factory import (
    create_assessment_compiler_client,
    create_llm_client,
    create_plan_review_client,
    create_review_client,
)
from app.llm.openai_client import OpenAIClient


@pytest.mark.parametrize("stage", [
    "correctness_validator", "stem_quantitative_validator",
    "stem_quantitative_repairer", "duplicate_judge",
])
def test_stem_review_routes_to_luna_medium(monkeypatch, stage):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    client = create_review_client(domain="STEM", stage=stage)
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-6-luna"
    assert client._reasoning_effort(stage) == "medium"
    assert client.supports_temperature is False


@pytest.mark.parametrize("stage", ["correctness_validator", "duplicate_judge"])
def test_non_stem_review_stays_on_deepseek(monkeypatch, stage):
    from app.llm import deepseek

    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    client = create_review_client(domain="NON_STEM", stage=stage)
    assert isinstance(client, DeepSeekClient)


def test_review_routing_rejects_missing_stored_classification():
    with pytest.raises(ValueError, match="domain STEM or NON_STEM"):
        create_review_client(domain="", stage="correctness_validator")


def test_general_validator_always_uses_deepseek(monkeypatch):
    from app.llm import deepseek
    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    assert isinstance(create_review_client(domain="STEM", stage="general_validator"), DeepSeekClient)


def test_repairer_uses_default_deepseek_flash_with_low_reasoning(monkeypatch):
    from app.llm import deepseek
    import app.llm.factory as factory

    monkeypatch.setattr(factory, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    repairer = create_llm_client()

    assert isinstance(repairer, DeepSeekClient)
    assert repairer.model == "deepseek-flash"
    assert repairer._reasoning_effort("repairer") == "low"


def test_plan_reviewer_uses_deepseek_flash_with_low_reasoning(monkeypatch):
    from app.llm import deepseek

    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    reviewer = create_plan_review_client()

    assert isinstance(reviewer, DeepSeekClient)
    assert reviewer.model == "deepseek-flash"
    assert reviewer._reasoning_effort("plan_reviewer") == "low"


def test_stem_assessment_compiler_routes_to_luna(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    client = create_assessment_compiler_client(domain_type="stem")
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-6-luna"


def test_non_stem_assessment_compiler_routes_to_deepseek(monkeypatch):
    from app.llm import deepseek
    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    client = create_assessment_compiler_client(domain_type="non_stem")
    assert isinstance(client, DeepSeekClient)
