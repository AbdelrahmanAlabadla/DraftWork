from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import LLM_PROVIDER
from app.llm.client import LMStudioClient

if TYPE_CHECKING:
    from app.llm.deepseek import DeepSeekClient


def create_llm_client(
    *,
    local_url: str | None = None,
    local_model: str | None = None,
    local_reasoning: str | None = None,
    deepseek_model: str | None = None,
) -> LMStudioClient | DeepSeekClient:
    """Create the selected LLM client while keeping LM Studio as the default."""
    if LLM_PROVIDER == "deepseek":
        from app.llm.deepseek import DeepSeekClient

        return DeepSeekClient(model=deepseek_model)
    return LMStudioClient(
        url=local_url,
        model=local_model,
        reasoning=local_reasoning,
    )


def create_plan_review_client():
    """Plan comparison always uses the small, bounded DeepSeek reviewer."""
    from app.llm.deepseek import DeepSeekClient

    return DeepSeekClient(model="deepseek-flash")


def create_duplicate_replacement_client():
    """Use the bounded comparison model for compact duplicate replacement selection."""
    from app.llm.deepseek import DeepSeekClient

    return DeepSeekClient(model="deepseek-flash")


def create_review_client(*, domain: str, stage: str):
    """Route review stages from the compiler's stored domain classification."""
    normalized_domain = str(domain or "").strip().upper()
    if stage not in {
        "correctness_validator", "general_validator", "stem_quantitative_validator",
        "stem_quantitative_repairer", "duplicate_judge",
    }:
        raise ValueError(f"Unsupported review stage: {stage}")
    if stage == "general_validator":
        from app.llm.deepseek import DeepSeekClient
        return DeepSeekClient()
    if normalized_domain == "STEM" and stage in {
        "correctness_validator", "stem_quantitative_validator",
        "stem_quantitative_repairer", "duplicate_judge",
    }:
        from app import config
        from app.llm.openai_client import OpenAIClient

        return OpenAIClient(
            model=config.STEM_REVIEW_MODEL,
            reasoning_effort=config.STEM_REVIEW_REASONING_EFFORT,
        )
    if normalized_domain == "NON_STEM":
        from app.llm.deepseek import DeepSeekClient

        return DeepSeekClient()
    raise ValueError("Assessment profile must contain domain STEM or NON_STEM")


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
    if identity["provider"] == "openai":
        from app import config
        from app.llm.openai_client import OpenAIClient

        return OpenAIClient(
            model=identity["model"],
            reasoning_effort=config.STEM_REVIEW_REASONING_EFFORT,
        )
    from app.llm.deepseek import DeepSeekClient

    return DeepSeekClient()
