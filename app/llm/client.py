from __future__ import annotations

import requests

from app.config import LMS_API_KEY, LMS_MODEL, LMS_REASONING, LMS_URL
from app.llm.json_utils import (
    JSONExtractionError,
    REPAIR_SYSTEM_PROMPT,
    build_repair_prompt,
    extract_json,
)
from app.logging_conf import get_logger

logger = get_logger("LLM")


class LMStudioClient:
    """Client for LM Studio's native REST API (POST {base}/api/v1/chat).

    The native API is used instead of the OpenAI-compatible endpoint because
    only it accepts a ``reasoning`` request field, letting us force reasoning
    off/on per call for models such as qwen3 that would otherwise spend their
    output budget on hidden thinking content. Accepts either the server base
    URL (http://127.0.0.1:1234) or its OpenAI-compatible form with a trailing
    /v1; both are normalized to the base.
    """

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        base = (url or LMS_URL).rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self.url = base
        self.model = model or LMS_MODEL
        self.reasoning = reasoning if reasoning is not None else LMS_REASONING
        # Kept for configuration compatibility; the local native API needs no
        # auth header and we never send credentials to it.
        self.api_key = LMS_API_KEY

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 600,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "reasoning": self.reasoning,
            "stream": False,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        logger.info(
            "LLM call | model=%s | url=%s/api/v1/chat | reasoning=%s | "
            "prompt_chars=%d | max_tokens=%d",
            self.model,
            self.url,
            self.reasoning,
            len(prompt),
            max_tokens,
        )

        response = requests.post(
            f"{self.url}/api/v1/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        parts: list[str] = []
        for item in data.get("output") or []:
            if isinstance(item, dict) and item.get("type") == "message":
                parts.append(str(item.get("content") or ""))
        text = "".join(parts).strip()

        stats = data.get("stats") or {}
        logger.info(
            "LLM response | output_chars=%d | in_tokens=%s | out_tokens=%s",
            len(text),
            stats.get("input_tokens"),
            stats.get("total_output_tokens"),
        )
        return text

    def chat_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 600,
        max_repair_attempts: int = 2,
    ) -> object:
        """Call the LLM and return a parsed JSON object.

        If the response is malformed JSON, the broken text and the parser error
        are sent back to the model (temperature 0.0) to repair, up to
        ``max_repair_attempts`` times. Raises JSONExtractionError if parsing
        still fails after all repair attempts.
        """
        raw = self.chat(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        for attempt in range(max_repair_attempts + 1):
            try:
                return extract_json(raw)
            except JSONExtractionError as exc:
                if attempt == max_repair_attempts:
                    raise
                logger.warning(
                    "JSON repair attempt %d/%d | err=%s | prompt_chars=%d",
                    attempt + 1,
                    max_repair_attempts,
                    exc,
                    len(raw),
                )
                repair_prompt = build_repair_prompt(raw, str(exc))
                raw = self.chat(
                    repair_prompt,
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=max_tokens * 2,
                    timeout=timeout,
                )
