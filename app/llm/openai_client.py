from __future__ import annotations

from typing import Any

from app import config
from app.llm.deepseek import DeepSeekClient, DeepSeekConfigurationError
from app.logging_conf import current_job_id, get_logger
from app.metrics import increment

logger = get_logger("LLM")


class OpenAIClient(DeepSeekClient):
    """OpenAI Responses client for the two routed STEM review stages."""

    provider_name = "openai"
    supports_temperature = False

    def __init__(self, *, model: str = "gpt-6-luna", reasoning_effort: str = "medium") -> None:
        api_key = str(config.OPENAI_API_KEY or "").strip()
        if not api_key:
            raise DeepSeekConfigurationError(
                "OPENAI_API_KEY is required for OpenAI model routing"
            )
        self.url = config.OPENAI_BASE_URL.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.last_usage: dict[str, int | str] | None = None
        self.usage_history: list[dict[str, int | str]] = []
        self.last_transient_retries = 0
        self.last_structured_attempts = 0

    def _reasoning_effort(self, agent: str) -> str:
        return self.reasoning_effort

    def _capture_usage(
        self, response: Any, agent: str, *, duration_ms: int | None = None
    ) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        values: dict[str, int | str | float] = {
            "agent": agent,
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
            "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        uncached = max(0, int(values["input_tokens"]) - int(values["cached_tokens"]))
        values["estimated_cost_usd"] = (
            uncached * config.OPENAI_INPUT_COST_PER_MTOK
            + int(values["cached_tokens"]) * config.OPENAI_CACHED_INPUT_COST_PER_MTOK
            + int(values["output_tokens"]) * config.OPENAI_OUTPUT_COST_PER_MTOK
        ) / 1_000_000
        self.last_usage = values  # type: ignore[assignment]
        increment(
            "genexam_llm_tokens", value=int(values["input_tokens"]),
            provider=self.provider_name, model=self.model, direction="input",
        )
        increment(
            "genexam_llm_tokens", value=int(values["output_tokens"]),
            provider=self.provider_name, model=self.model, direction="output",
        )
        increment(
            "genexam_llm_estimated_cost_usd", value=float(values["estimated_cost_usd"]),
            provider=self.provider_name, model=self.model,
        )
        try:
            from app.api import repositories

            repositories.record_llm_usage(
                job_id=current_job_id(), provider=self.provider_name, model=self.model,
                operation=agent, input_tokens=int(values["input_tokens"]),
                output_tokens=int(values["output_tokens"]),
                cached_tokens=int(values["cached_tokens"]),
                reasoning_tokens=int(values["reasoning_tokens"]),
                estimated_cost_usd=float(values["estimated_cost_usd"]),
                duration_ms=duration_ms,
            )
        except Exception as exc:
            logger.warning("LLM usage persistence failed | error=%s", type(exc).__name__)
        if self.last_usage:
            self.usage_history.append(dict(self.last_usage))
