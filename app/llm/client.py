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
    """Client for any OpenAI-compatible /chat/completions endpoint.

    Currently pointed at OpenCode Zen (https://opencode.ai/zen/v1) but works
    with LM Studio's OpenAI-compatible server too. Stateless; one call per
    request. The ``reasoning`` constructor argument is accepted for backward
    compatibility with existing call sites but is not sent on the wire.
    """

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        self.url = (url or LMS_URL).rstrip("/")
        self.model = model or LMS_MODEL
        self.reasoning = reasoning if reasoning is not None else LMS_REASONING
        self.api_key = LMS_API_KEY

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 600,
    ) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers: dict = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info(
            "LLM call | model=%s | url=%s | prompt_chars=%d | max_tokens=%d",
            self.model,
            self.url,
            len(prompt),
            max_tokens,
        )

        response = requests.post(
            f"{self.url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            text = str(message.get("content") or "").strip()

        usage = data.get("usage") or {}
        logger.info(
            "LLM response | output_chars=%d | in_tokens=%s | out_tokens=%s",
            len(text),
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
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
