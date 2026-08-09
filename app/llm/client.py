from __future__ import annotations

import requests

from app.config import LMS_MODEL, LMS_URL
from app.llm.json_utils import (
    JSONExtractionError,
    REPAIR_SYSTEM_PROMPT,
    build_repair_prompt,
    extract_json,
)
from app.logging_conf import get_logger

logger = get_logger("LLM")


class LMStudioClient:
    """Client for LM Studio's /api/v1/chat endpoint (stateless)."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        self.url = (url or LMS_URL).rstrip("/")
        self.model = model or LMS_MODEL
        # Optional reasoning override: "off"|"low"|"medium"|"high"|"on".
        # Only sent when set, so non-reasoning models never error.
        self.reasoning = reasoning

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
            "stream": False,
            "store": False,
        }
        if getattr(self, "reasoning", None):
            payload["reasoning"] = self.reasoning
        if system_prompt:
            payload["system_prompt"] = system_prompt

        logger.info(
            "LLM call | model=%s | prompt_chars=%d | max_tokens=%d",
            self.model,
            len(prompt),
            max_tokens,
        )

        response = requests.post(
            f"{self.url}/api/v1/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        output = data.get("output") or []
        text_parts = [
            item.get("content", "")
            for item in output
            if isinstance(item, dict) and item.get("type") == "message"
        ]
        text = "".join(text_parts).strip()

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
