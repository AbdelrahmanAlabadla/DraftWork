from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

from app.config import LLM_PROVIDER
from app.llm.client import LMStudioClient

if TYPE_CHECKING:
    from app.llm.deepseek import DeepSeekClient


_ONLINE_ASSESSMENT_DOMAIN: ContextVar[str] = ContextVar(
    "online_assessment_domain", default=""
)


def set_online_assessment_domain(domain: str | None) -> Token:
    """Bind the persisted assessment domain for all LLM calls in one graph node."""
    return _ONLINE_ASSESSMENT_DOMAIN.set(str(domain or "").strip().upper())


def reset_online_assessment_domain(token: Token) -> None:
    _ONLINE_ASSESSMENT_DOMAIN.reset(token)


def _domain(domain: str | None = None) -> str:
    return str(domain or _ONLINE_ASSESSMENT_DOMAIN.get() or "").strip().upper()


def _stem_reasoning_effort(stage: str) -> str:
    from app import config

    if stage in {
        "stem_quantitative_validator", "stem_quantitative_repairer", "solution_repairer",
    }:
        return config.STEM_QUANTITATIVE_REASONING_EFFORT
    return config.STEM_REVIEW_REASONING_EFFORT


DEEPSEEK_GENERATION_TYPES = frozenset({
    "fill_in_the_blank", "why", "definition",
})
LUNA_GENERATION_TYPES = frozenset({
    "mcq", "true_false", "equation", "word_problem", "essay",
})
LUNA_VALIDATOR_STAGES = frozenset({
    "correctness_validator", "general_validator", "grounding_validator",
    "task_alignment_validator", "answer_correctness_validator",
    "difficulty_validator", "quantitative_validator", "duplicate_validator",
    "stem_quantitative_validator", "task_alignment_recovery_decision",
})
DEEPSEEK_REPAIR_STAGES = frozenset({"options_repairer", "answer_repairer"})
LUNA_REPAIR_STAGES = frozenset({
    "question_repairer", "duplicate_repairer", "solution_repairer",
    "task_alignment_repairer", "difficulty_repairer",
    "stem_quantitative_repairer", "general_repairer",
})


def _openai_client(stage: str):
    from app import config
    from app.llm.openai_client import OpenAIClient

    return OpenAIClient(
        model=config.STEM_REVIEW_MODEL,
        reasoning_effort=_stem_reasoning_effort(stage),
    )


def _deepseek_client(model: str | None = None):
    from app.llm.deepseek import DeepSeekClient

    return DeepSeekClient(model=model)


def create_llm_client(
    *,
    local_url: str | None = None,
    local_model: str | None = None,
    local_reasoning: str | None = None,
    deepseek_model: str | None = None,
    domain: str | None = None,
    stage: str = "online_default",
) -> LMStudioClient | DeepSeekClient:
    """Create a client using explicit stage policy, then legacy domain defaults."""
    normalized_domain = _domain(domain)
    if stage in LUNA_VALIDATOR_STAGES or stage in LUNA_REPAIR_STAGES:
        return _openai_client(stage)
    if stage in DEEPSEEK_REPAIR_STAGES:
        return _deepseek_client(deepseek_model)
    if stage.endswith("_generator"):
        question_type = stage[:-len("_generator")]
        if question_type in DEEPSEEK_GENERATION_TYPES:
            return _deepseek_client(deepseek_model)
        if question_type in LUNA_GENERATION_TYPES:
            return _openai_client(stage)
        raise ValueError(f"Unsupported generation stage: {stage}")
    if normalized_domain == "STEM":
        return _openai_client(stage)
    if normalized_domain == "NON_STEM":
        return _deepseek_client(deepseek_model)
    if LLM_PROVIDER == "deepseek":
        from app.llm.deepseek import DeepSeekClient

        return DeepSeekClient(model=deepseek_model)
    return LMStudioClient(
        url=local_url,
        model=local_model,
        reasoning=local_reasoning,
    )


def create_duplicate_replacement_client(*, domain: str | None = None):
    """Create the domain-routed duplicate replacement selector."""
    return create_llm_client(domain=domain, stage="duplicate_replacement")


def create_difficulty_adjustment_client(*, domain: str | None = None):
    """Create the on-demand, domain-routed difficulty adjustment agent."""
    return create_llm_client(domain=domain, stage="difficulty_adjuster")


def create_review_client(*, domain: str, stage: str):
    """Route review stages from the compiler's stored domain classification."""
    normalized_domain = str(domain or "").strip().upper()
    if stage not in {
        "correctness_validator", "general_validator", "stem_quantitative_validator",
        "stem_quantitative_repairer", "grounding_validator",
        "task_alignment_validator", "answer_correctness_validator",
        "task_alignment_recovery_decision",
        "difficulty_validator", "quantitative_validator", "duplicate_validator",
        "answer_repairer", "question_repairer", "options_repairer", "duplicate_repairer",
        "solution_repairer", "task_alignment_repairer", "difficulty_repairer",
        "general_repairer",
    }:
        raise ValueError(f"Unsupported review stage: {stage}")
    if normalized_domain not in {"STEM", "NON_STEM"}:
        raise ValueError("Assessment profile must contain domain STEM or NON_STEM")
    return create_llm_client(domain=normalized_domain, stage=stage)


def create_generation_client(*, question_type: str, domain: str | None = None):
    """Route a generator by its explicit question-type identifier."""
    normalized = str(question_type or "").strip().lower()
    if normalized not in DEEPSEEK_GENERATION_TYPES | LUNA_GENERATION_TYPES:
        raise ValueError(f"Unsupported generation question type: {question_type}")
    try:
        return create_llm_client(domain=domain, stage=f"{normalized}_generator")
    except TypeError as exc:
        # Keep the established zero-argument fake-factory seam used by offline
        # tests; the production factory always accepts explicit identifiers.
        message = str(exc)
        if "unexpected keyword argument" not in message and "takes no arguments" not in message:
            raise
        return create_llm_client()


def assessment_compiler_identity(*, domain_type: str) -> dict[str, str]:
    """Return the cache identity for the compiler selected by stored title metadata."""
    from app import config

    normalized = str(domain_type or "").strip().lower()
    if normalized == "stem":
        return {
            "provider": "openai",
            "model": config.STEM_REVIEW_MODEL,
            "version": config.ASSESSMENT_COMPILER_VERSION,
        }
    if normalized == "non_stem":
        return {
            "provider": "deepseek",
            "model": config.DEEPSEEK_MODEL,
            "version": config.ASSESSMENT_COMPILER_VERSION,
        }
    raise ValueError("Stored title metadata must contain domain_type stem or non_stem")


def create_assessment_compiler_client(*, domain_type: str):
    """Route the compiler solely from the persisted title-generation domain."""
    identity = assessment_compiler_identity(domain_type=domain_type)
    domain = "STEM" if identity["provider"] == "openai" else "NON_STEM"
    return create_llm_client(domain=domain, stage="assessment_compiler")
