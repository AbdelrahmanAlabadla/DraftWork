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
) -> LMStudioClient | DeepSeekClient:
    """Create the selected LLM client while keeping LM Studio as the default."""
    if LLM_PROVIDER == "deepseek":
        from app.llm.deepseek import DeepSeekClient

        return DeepSeekClient()
    return LMStudioClient(
        url=local_url,
        model=local_model,
        reasoning=local_reasoning,
    )
