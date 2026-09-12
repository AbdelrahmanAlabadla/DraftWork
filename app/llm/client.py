from __future__ import annotations

import json
import re

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

    provider_name = "lm_studio"

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

    def chat_structured(
        self,
        prompt: str,
        *,
        json_schema: dict,
        schema_name: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 600,
    ) -> object:
        """Return JSON constrained by LM Studio's JSON-Schema sampler.

        Structured output is exposed by LM Studio's OpenAI-compatible endpoint,
        while the native ``/api/v1/chat`` endpoint remains in use for calls that
        need its reasoning control.
        """
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", schema_name)[:64] or "response"
        payload: dict = {
            "model": self.model,
            "messages": [
                *(
                    [{"role": "system", "content": system_prompt}]
                    if system_prompt
                    else []
                ),
                {
                    "role": "user",
                    "content": (
                        prompt + "\n\n/no_think"
                        if "qwen3" in self.model.lower()
                        and self.reasoning.lower() in {"off", "none", "false", "0"}
                        else prompt
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": safe_name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info(
            "Structured LLM call | model=%s | schema=%s | prompt_chars=%d | max_tokens=%d",
            self.model,
            safe_name,
            len(prompt),
            max_tokens,
        )
        response = requests.post(
            f"{self.url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JSONExtractionError(
                "LM Studio structured response did not contain message content"
            ) from exc
        if isinstance(content, dict):
            return content
        try:
            return json.loads(str(content))
        except json.JSONDecodeError as exc:
            raise JSONExtractionError(
                f"LM Studio returned invalid structured JSON: {exc}"
            ) from exc

    def health(self, timeout: int = 3) -> None:
        response = requests.get(f"{self.url}/api/v1/models", timeout=timeout)
        response.raise_for_status()


def request_structured(
    client: object,
    prompt: str,
    *,
    json_schema: dict | None = None,
    response_model: type | None = None,
    schema_name: str,
    agent: str = "generator",
    item_count: int = 1,
    operation: str = "default",
    expected_ids: list[str] | None = None,
    expected_id_field: str | None = None,
    local_json: bool = False,
    system_prompt: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 600,
) -> object:
    """Use provider structured output while preserving the local call behavior."""
    if json_schema is None:
        if response_model is None:
            raise ValueError("response_model or json_schema is required")
        from app.llm.schemas import strict_json_schema

        json_schema = strict_json_schema(response_model)
    method = getattr(client, "chat_structured", None)
    if callable(method) and getattr(
        client, "supports_agent_structured_output", False
    ):
        if response_model is None:
            raise ValueError("DeepSeek structured output requires a Pydantic model")
        return method(
            prompt,
            response_model=response_model,
            json_schema=json_schema,
            schema_name=schema_name,
            agent=agent,
            item_count=item_count,
            operation=operation,
            expected_ids=expected_ids,
            expected_id_field=expected_id_field,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    if local_json:
        return client.chat_json(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    if callable(method):
        return method(
            prompt,
            json_schema=json_schema,
            schema_name=schema_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    return client.chat_json(
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
