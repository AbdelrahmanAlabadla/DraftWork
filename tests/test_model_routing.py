from __future__ import annotations

import pytest

from app import config
from app.llm.deepseek import DeepSeekClient
from app.llm.factory import create_assessment_compiler_client, create_generation_client, create_review_client
from app.llm.openai_client import OpenAIClient


GENERATION = {
    "fill_in_the_blank": "deepseek", "why": "deepseek", "definition": "deepseek",
    "mcq": "openai", "true_false": "openai", "equation": "openai",
    "word_problem": "openai", "essay": "openai",
}
REPAIRS = {
    "options_repairer": "deepseek", "answer_repairer": "deepseek",
    "question_repairer": "openai", "duplicate_repairer": "openai",
    "solution_repairer": "openai", "task_alignment_repairer": "openai",
    "difficulty_repairer": "openai", "stem_quantitative_repairer": "openai",
    "general_repairer": "openai",
}
VALIDATORS = [
    "grounding_validator", "task_alignment_validator", "answer_correctness_validator",
    "difficulty_validator", "quantitative_validator", "duplicate_validator",
    "general_validator", "correctness_validator", "stem_quantitative_validator",
]


def _assert_provider(client, provider):
    if provider == "openai":
        assert type(client) is OpenAIClient
        assert client.model == "gpt-6-luna"
    else:
        assert type(client) is DeepSeekClient
        assert client.model == "deepseek-flash"


@pytest.mark.parametrize("domain", ["STEM", "NON_STEM"])
@pytest.mark.parametrize("question_type,provider", GENERATION.items())
def test_question_generation_routing_is_explicit(monkeypatch, domain, question_type, provider):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("app.llm.deepseek.DEEPSEEK_API_KEY", "test-key")
    _assert_provider(create_generation_client(question_type=question_type, domain=domain), provider)


@pytest.mark.parametrize("domain", ["STEM", "NON_STEM"])
@pytest.mark.parametrize("stage,provider", REPAIRS.items())
def test_repair_routing_is_explicit(monkeypatch, domain, stage, provider):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("app.llm.deepseek.DEEPSEEK_API_KEY", "test-key")
    _assert_provider(create_review_client(domain=domain, stage=stage), provider)


@pytest.mark.parametrize("domain", ["STEM", "NON_STEM"])
@pytest.mark.parametrize("stage", VALIDATORS)
def test_every_validator_routes_to_luna(monkeypatch, domain, stage):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    _assert_provider(create_review_client(domain=domain, stage=stage), "openai")


def test_compiler_routing_is_unchanged(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("app.llm.deepseek.DEEPSEEK_API_KEY", "test-key")
    _assert_provider(create_assessment_compiler_client(domain_type="stem"), "openai")
    _assert_provider(create_assessment_compiler_client(domain_type="non_stem"), "deepseek")
