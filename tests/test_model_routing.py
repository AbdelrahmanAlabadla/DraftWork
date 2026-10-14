from __future__ import annotations

import pytest

from app import config
from app.llm.deepseek import DeepSeekClient
from app.llm.factory import (
    create_assessment_compiler_client,
    create_duplicate_replacement_client,
    create_llm_client,
    create_plan_review_client,
    create_review_client,
    reset_online_assessment_domain,
    set_online_assessment_domain,
)
from app.llm.openai_client import OpenAIClient


STEM_GENERAL_STAGES = [
    "planner", "plan_reviewer", "mcq_generator", "true_false_generator",
    "fill_in_the_blank_generator", "definition_generator", "why_generator",
    "equation_generator", "word_problem_generator", "essay_generator",
    "correctness_validator", "general_validator", "general_repairer",
    "duplicate_judge", "duplicate_replacement",
]


def _assert_luna_medium(client):
    assert type(client) is OpenAIClient
    assert client.provider_name == "openai"
    assert client.model == "gpt-6-luna"
    assert client.reasoning_effort == "medium"
    assert client.supports_temperature is False


@pytest.mark.parametrize("stage", STEM_GENERAL_STAGES)
def test_every_general_stem_stage_routes_to_luna_medium(monkeypatch, stage):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    _assert_luna_medium(create_llm_client(domain="STEM", stage=stage))


@pytest.mark.parametrize(
    "stage", ["stem_quantitative_validator", "stem_quantitative_repairer"]
)
def test_quantitative_stem_stages_default_to_luna_medium(monkeypatch, stage):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "STEM_QUANTITATIVE_REASONING_EFFORT", "medium")
    _assert_luna_medium(create_review_client(domain="STEM", stage=stage))


@pytest.mark.parametrize(
    "stage", ["stem_quantitative_validator", "stem_quantitative_repairer"]
)
def test_quantitative_stem_stages_support_configured_high(monkeypatch, stage):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "STEM_QUANTITATIVE_REASONING_EFFORT", "high")
    client = create_review_client(domain="STEM", stage=stage)
    assert type(client) is OpenAIClient
    assert client.model == "gpt-6-luna"
    assert client.reasoning_effort == "high"


def test_quantitative_high_does_not_change_other_stem_agents(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "STEM_QUANTITATIVE_REASONING_EFFORT", "high")
    _assert_luna_medium(create_llm_client(domain="STEM", stage="general_validator"))


def test_context_routes_shared_online_factories_to_stem(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    token = set_online_assessment_domain("STEM")
    try:
        for client in (
            create_llm_client(), create_plan_review_client(),
            create_duplicate_replacement_client(),
        ):
            _assert_luna_medium(client)
    finally:
        reset_online_assessment_domain(token)


@pytest.mark.parametrize("stage", STEM_GENERAL_STAGES)
def test_non_stem_general_stages_remain_deepseek_flash(monkeypatch, stage):
    from app.llm import deepseek

    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    client = create_llm_client(domain="NON_STEM", stage=stage)
    assert type(client) is DeepSeekClient
    assert client.provider_name == "deepseek"
    assert client.model == "deepseek-flash"


def test_non_stem_context_preserves_specialized_factories(monkeypatch):
    from app.llm import deepseek

    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    token = set_online_assessment_domain("NON_STEM")
    try:
        reviewer = create_plan_review_client()
        selector = create_duplicate_replacement_client()
    finally:
        reset_online_assessment_domain(token)
    assert type(reviewer) is DeepSeekClient
    assert reviewer.model == "deepseek-flash"
    assert reviewer._reasoning_effort("plan_reviewer") == "low"
    assert type(selector) is DeepSeekClient
    assert selector.model == "deepseek-flash"


def test_review_routing_rejects_missing_stored_classification():
    with pytest.raises(ValueError, match="domain STEM or NON_STEM"):
        create_review_client(domain="", stage="correctness_validator")


def test_stem_assessment_compiler_routes_to_luna(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    _assert_luna_medium(create_assessment_compiler_client(domain_type="stem"))


def test_non_stem_assessment_compiler_routes_to_deepseek(monkeypatch):
    from app.llm import deepseek

    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    client = create_assessment_compiler_client(domain_type="non_stem")
    assert type(client) is DeepSeekClient
    assert client.model == "deepseek-flash"
