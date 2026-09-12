from __future__ import annotations

import asyncio
import copy
import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from app.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CONCURRENCY_LIMIT,
    DEEPSEEK_MAX_STRUCTURED_ATTEMPTS,
    DEEPSEEK_MAX_TRANSIENT_RETRIES,
    DEEPSEEK_MODEL,
    DEEPSEEK_RETRY_BASE_SECONDS,
)
from app.llm.schemas import strict_json_schema
from app.logging_conf import get_logger

logger = get_logger("LLM")
T = TypeVar("T")


class DeepSeekError(RuntimeError):
    """Base error for the DeepSeek provider."""


class DeepSeekConfigurationError(DeepSeekError):
    pass


class DeepSeekAuthenticationError(DeepSeekError):
    pass


class DeepSeekRateLimitError(DeepSeekError):
    pass


class DeepSeekTimeoutError(DeepSeekError):
    pass


class DeepSeekNetworkError(DeepSeekError):
    pass


class DeepSeekServerError(DeepSeekError):
    pass


class DeepSeekSchemaError(DeepSeekError):
    pass


class DeepSeekValidationError(DeepSeekError):
    pass


class DeepSeekIncompleteError(DeepSeekError):
    pass


def _run_sync(awaitable: Awaitable[T]) -> T:
    """Run an async SDK call from DraftWork's existing synchronous pipeline."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, awaitable).result()


class DeepSeekClient:
    """DeepSeek Responses API client with schema validation and bounded retries."""

    supports_agent_structured_output = True
    provider_name = "deepseek"
    _concurrency = threading.BoundedSemaphore(DEEPSEEK_CONCURRENCY_LIMIT)
    _usage_lock = threading.Lock()
    _usage_by_agent: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
    )

    def __init__(self) -> None:
        if not DEEPSEEK_API_KEY:
            raise DeepSeekConfigurationError(
                "DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek"
            )
        self.url = DEEPSEEK_BASE_URL.rstrip("/")
        self.model = DEEPSEEK_MODEL
        self.api_key = DEEPSEEK_API_KEY
        self.last_usage: dict[str, int | str] | None = None

    @classmethod
    def usage_snapshot(cls) -> dict[str, dict[str, int]]:
        with cls._usage_lock:
            return copy.deepcopy(dict(cls._usage_by_agent))

    async def _sdk_request(self, *, timeout: int, **kwargs):
        await asyncio.to_thread(self._concurrency.acquire)
        try:
            async with AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.url,
                timeout=timeout,
                max_retries=0,
            ) as client:
                return await client.responses.create(**kwargs)
        finally:
            self._concurrency.release()

    async def _request_with_backoff(self, *, timeout: int, **kwargs):
        for transient_attempt in range(DEEPSEEK_MAX_TRANSIENT_RETRIES + 1):
            try:
                return await self._sdk_request(timeout=timeout, **kwargs)
            except AuthenticationError as exc:
                raise DeepSeekAuthenticationError(
                    "DeepSeek rejected DEEPSEEK_API_KEY"
                ) from exc
            except RateLimitError as exc:
                cause = exc
                mapped: DeepSeekError = DeepSeekRateLimitError(
                    "DeepSeek rate limit exceeded"
                )
            except APITimeoutError as exc:
                cause = exc
                mapped = DeepSeekTimeoutError("DeepSeek request timed out")
            except APIConnectionError as exc:
                cause = exc
                mapped = DeepSeekNetworkError("DeepSeek network connection failed")
            except BadRequestError:
                raise
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise DeepSeekError(
                        f"DeepSeek API rejected the request with status {exc.status_code}"
                    ) from exc
                cause = exc
                mapped = DeepSeekServerError(
                    f"DeepSeek server error ({exc.status_code})"
                )

            if transient_attempt >= DEEPSEEK_MAX_TRANSIENT_RETRIES:
                raise mapped from cause
            delay = DEEPSEEK_RETRY_BASE_SECONDS * (2 ** transient_attempt)
            logger.warning(
                "DeepSeek transient failure | type=%s | retry=%d/%d | delay=%.2fs",
                type(mapped).__name__,
                transient_attempt + 1,
                DEEPSEEK_MAX_TRANSIENT_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)

    @staticmethod
    def _reasoning_effort(agent: str) -> str:
        return "high" if agent in {"planner", "validator"} else "none"

    @staticmethod
    def _token_budget(agent: str, item_count: int, operation: str) -> tuple[int, int]:
        count = max(1, int(item_count))
        if operation == "fitb_word_bank":
            return 1024, 1024
        if operation == "fitb_items":
            return min(8000, max(2048, count * 384)), 8000
        if agent == "planner":
            return min(32000, max(8192, count * 256)), 32000
        if agent == "validator":
            return min(32000, max(8192, count * 512)), 32000
        if agent == "repairer":
            return min(16000, max(4096, count * 512)), 16000
        return min(8000, max(2048, count * 320)), 8000

    def _capture_usage(self, response: Any, agent: str) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        values: dict[str, int | str] = {
            "agent": agent,
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "reasoning_tokens": int(
                getattr(output_details, "reasoning_tokens", 0) or 0
            ),
            "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        self.last_usage = values
        with self._usage_lock:
            totals = self._usage_by_agent[agent]
            for key in (
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cached_tokens",
                "total_tokens",
            ):
                totals[key] += int(values[key])
            totals["requests"] += 1
        logger.info(
            "DeepSeek usage | agent=%s | input_tokens=%d | output_tokens=%d | "
            "reasoning_tokens=%d | cached_tokens=%d | total_tokens=%d",
            agent,
            values["input_tokens"],
            values["output_tokens"],
            values["reasoning_tokens"],
            values["cached_tokens"],
            values["total_tokens"],
        )

    @staticmethod
    def _output_text(response: Any) -> str:
        return str(getattr(response, "output_text", "") or "").strip()

    @staticmethod
    def _validate_expected_ids(
        value: Any, expected_ids: list[str] | None, id_field: str | None
    ) -> None:
        if not expected_ids or not id_field:
            return
        found: list[str] = []

        def collect(node: Any) -> None:
            if isinstance(node, dict):
                if id_field in node:
                    found.append(str(node[id_field]))
                for child in node.values():
                    collect(child)
            elif isinstance(node, list):
                for child in node:
                    collect(child)

        collect(value)
        if sorted(found) != sorted(str(item) for item in expected_ids):
            raise DeepSeekValidationError(
                f"{id_field} values must be exactly {expected_ids}; received {found}"
            )

    @staticmethod
    def _corrective_input(
        original_prompt: str,
        previous_output: str,
        failure_reason: str,
        schema: dict[str, Any],
        expected_ids: list[str] | None,
    ) -> str:
        return (
            f"{original_prompt}\n\n"
            "## Corrective structured-output retry\n"
            f"Previous response:\n{previous_output or '[no complete output returned]'}\n\n"
            f"Previous response failed because:\n{failure_reason}\n\n"
            "Correct only that specific failure while preserving every valid field, "
            "ID, concept, difficulty, question type, and other correct data. This is "
            "the same logical task and must not create an extra question or plan slot.\n"
            f"Expected IDs: {expected_ids or 'defined by the schema'}\n"
            f"Expected JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            "Return output matching the same JSON Schema."
        )

    async def achat_structured(
        self,
        prompt: str,
        *,
        response_model: type[BaseModel],
        json_schema: dict[str, Any],
        schema_name: str,
        agent: str,
        item_count: int,
        operation: str = "default",
        expected_ids: list[str] | None = None,
        expected_id_field: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        timeout: int = 600,
    ) -> object:
        try:
            schema = strict_json_schema(response_model)
        except ValueError as exc:
            raise DeepSeekSchemaError(str(exc)) from exc
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", schema_name)[:64] or "response"
        max_tokens, token_ceiling = self._token_budget(agent, item_count, operation)
        reasoning_effort = self._reasoning_effort(agent)
        previous_output = ""
        failure_reason = ""

        for attempt in range(1, DEEPSEEK_MAX_STRUCTURED_ATTEMPTS + 1):
            current_input = (
                prompt
                if attempt == 1
                else self._corrective_input(
                    prompt, previous_output, failure_reason, schema, expected_ids
                )
            )
            logger.info(
                "DeepSeek structured call | agent=%s | model=%s | schema=%s | "
                "reasoning=%s | attempt=%d/%d | max_output_tokens=%d",
                agent,
                self.model,
                safe_name,
                reasoning_effort,
                attempt,
                DEEPSEEK_MAX_STRUCTURED_ATTEMPTS,
                max_tokens,
            )
            try:
                structured_instructions = "\n\n".join(
                    part
                    for part in (
                        system_prompt,
                        "The supplied JSON Schema is authoritative. Return the exact "
                        "top-level object and fields required by that schema, even if "
                        "older wording in the task describes a bare JSON array.",
                    )
                    if part
                )
                response = await self._request_with_backoff(
                    timeout=timeout,
                    model=self.model,
                    instructions=structured_instructions,
                    input=current_input,
                    reasoning={"effort": reasoning_effort},
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": safe_name,
                            "schema": schema,
                            "strict": True,
                        }
                    },
                )
            except BadRequestError as exc:
                message = str(exc)
                if "schema" in message.lower() or "text.format" in message.lower():
                    raise DeepSeekSchemaError(
                        "DeepSeek rejected the normalized JSON Schema"
                    ) from exc
                raise DeepSeekError("DeepSeek rejected the structured request") from exc

            self._capture_usage(response, agent)
            previous_output = self._output_text(response)
            status = str(getattr(response, "status", "") or "")
            if status == "incomplete":
                details = getattr(response, "incomplete_details", None)
                reason = str(getattr(details, "reason", "unknown") or "unknown")
                failure_reason = f"response incomplete: {reason}"
                if reason == "max_output_tokens":
                    max_tokens = min(token_ceiling, max_tokens * 2)
                if attempt == DEEPSEEK_MAX_STRUCTURED_ATTEMPTS:
                    raise DeepSeekIncompleteError(failure_reason)
                continue
            if status == "failed":
                error = getattr(response, "error", None)
                code = str(getattr(error, "code", "unknown") or "unknown")
                failure_reason = f"response failed: {code}"
                if attempt == DEEPSEEK_MAX_STRUCTURED_ATTEMPTS:
                    raise DeepSeekError(failure_reason)
                continue
            if status and status != "completed":
                failure_reason = f"unexpected response status: {status}"
                if attempt == DEEPSEEK_MAX_STRUCTURED_ATTEMPTS:
                    raise DeepSeekIncompleteError(failure_reason)
                continue

            try:
                validated = response_model.model_validate_json(previous_output)
                value = validated.model_dump()
                self._validate_expected_ids(value, expected_ids, expected_id_field)
                return value
            except (ValidationError, json.JSONDecodeError, DeepSeekValidationError) as exc:
                failure_reason = str(exc)
                if attempt == DEEPSEEK_MAX_STRUCTURED_ATTEMPTS:
                    raise DeepSeekValidationError(
                        "DeepSeek structured output failed local Pydantic validation "
                        f"after {DEEPSEEK_MAX_STRUCTURED_ATTEMPTS} attempts: {failure_reason}"
                    ) from exc

        raise DeepSeekValidationError("DeepSeek structured output failed")

    def chat_structured(
        self,
        prompt: str,
        *,
        response_model: type[BaseModel],
        json_schema: dict[str, Any],
        schema_name: str,
        agent: str,
        item_count: int,
        operation: str = "default",
        expected_ids: list[str] | None = None,
        expected_id_field: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 600,
    ) -> object:
        del max_tokens
        return _run_sync(
            self.achat_structured(
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
                timeout=timeout,
            )
        )

    async def achat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 600,
    ) -> str:
        response = await self._request_with_backoff(
            timeout=timeout,
            model=self.model,
            instructions=system_prompt,
            input=prompt,
            reasoning={"effort": "none"},
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        self._capture_usage(response, "title")
        status = str(getattr(response, "status", "") or "")
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            raise DeepSeekIncompleteError(
                f"DeepSeek text response incomplete: {getattr(details, 'reason', 'unknown')}"
            )
        if status != "completed":
            raise DeepSeekError(f"DeepSeek text response status was {status or 'unknown'}")
        return self._output_text(response)

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 600,
    ) -> str:
        return _run_sync(
            self.achat(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        )

    def chat_json(self, *args, **kwargs) -> object:
        raise DeepSeekConfigurationError(
            "DeepSeek JSON operations require chat_structured with a Pydantic schema"
        )

    async def ahealth(self, timeout: int = 3) -> None:
        await asyncio.to_thread(self._concurrency.acquire)
        try:
            async with AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.url,
                timeout=timeout,
                max_retries=0,
            ) as client:
                await client.models.list()
        except AuthenticationError as exc:
            raise DeepSeekAuthenticationError(
                "DeepSeek rejected DEEPSEEK_API_KEY"
            ) from exc
        finally:
            self._concurrency.release()

    def health(self, timeout: int = 3) -> None:
        _run_sync(self.ahealth(timeout=timeout))
