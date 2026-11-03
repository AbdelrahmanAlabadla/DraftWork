from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
import uuid
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

from app import config
from app.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CONCURRENCY_LIMIT,
    DEEPSEEK_INPUT_COST_PER_MTOK,
    DEEPSEEK_CACHED_INPUT_COST_PER_MTOK,
    DEEPSEEK_MAX_STRUCTURED_ATTEMPTS,
    DEEPSEEK_MAX_TRANSIENT_RETRIES,
    DEEPSEEK_MODEL,
    DEEPSEEK_OUTPUT_COST_PER_MTOK,
    DEEPSEEK_RETRY_BASE_SECONDS,
)
from app.llm.schemas import strict_json_schema
from app.logging_conf import current_job_id, get_logger
from app.metrics import increment

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


class DeepSeekQuotaError(DeepSeekError):
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
    def __init__(self, message: str, *, partial_output: str = "", reason: str = ""):
        super().__init__(message)
        self.partial_output = partial_output
        self.reason = reason


class DeepSeekIncompleteError(DeepSeekError):
    def __init__(self, message: str, *, partial_output: str = "", reason: str = ""):
        super().__init__(message)
        self.partial_output = partial_output
        self.reason = reason


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
    supports_temperature = True
    provider_name = "deepseek"
    provider_label = "DeepSeek"
    error_type = DeepSeekError
    configuration_error_type = DeepSeekConfigurationError
    authentication_error_type = DeepSeekAuthenticationError
    rate_limit_error_type = DeepSeekRateLimitError
    quota_error_type = DeepSeekQuotaError
    timeout_error_type = DeepSeekTimeoutError
    network_error_type = DeepSeekNetworkError
    server_error_type = DeepSeekServerError
    schema_error_type = DeepSeekSchemaError
    validation_error_type = DeepSeekValidationError
    incomplete_error_type = DeepSeekIncompleteError
    _concurrency = threading.BoundedSemaphore(DEEPSEEK_CONCURRENCY_LIMIT)
    _catalog_concurrency = threading.BoundedSemaphore(
        config.MAX_PARALLEL_CATALOG_CALLS
    )
    _quota_lock = threading.Lock()
    _quota_blocked_accounts: set[str] = set()

    def __init__(self, *, model: str | None = None) -> None:
        if not DEEPSEEK_API_KEY:
            raise DeepSeekConfigurationError(
                "DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek"
            )
        self.url = DEEPSEEK_BASE_URL.rstrip("/")
        self.model = str(model or DEEPSEEK_MODEL)
        self.api_key = DEEPSEEK_API_KEY
        self.last_usage: dict[str, int | str] | None = None
        self.last_transient_retries = 0
        self.last_structured_attempts = 0
        self.last_structured_attempt_details: list[dict[str, Any]] = []

    def _account_scope(self) -> str:
        digest = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:16]
        return f"{self.provider_name}:{digest}"

    def _record_request_event(self, **values: Any) -> None:
        try:
            from app.api import repositories
            repositories.record_llm_request_event(
                job_id=current_job_id(), provider=self.provider_name,
                model=self.model, **values,
            )
        except Exception as exc:
            logger.warning(
                "LLM request-event persistence failed | error=%s",
                type(exc).__name__,
            )

    @staticmethod
    def _provider_error_details(exc: Exception) -> dict[str, Any]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        body = getattr(exc, "body", None)
        error = body.get("error") if isinstance(body, dict) else None
        error = error if isinstance(error, dict) else (body if isinstance(body, dict) else {})
        message = str(error.get("message") or exc)
        code = str(error.get("code") or error.get("type") or "")
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        request_id = (
            headers.get("x-request-id") or headers.get("X-Request-Id")
            or getattr(exc, "request_id", None)
        )
        selected_headers = {
            str(key).lower(): str(value)
            for key, value in headers.items()
            if str(key).lower() in {
                "retry-after", "x-request-id", "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
                "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
                "x-ratelimit-reset-tokens",
            }
        }
        return {
            "message": message[:1000], "code": code[:200],
            "retry_after": retry_after, "request_id": request_id,
            "headers": selected_headers,
            "status_code": getattr(response, "status_code", 429),
        }

    @staticmethod
    def _is_insufficient_quota(details: dict[str, Any]) -> bool:
        text = f"{details.get('code', '')} {details.get('message', '')}".lower()
        return any(marker in text for marker in (
            "insufficient_quota", "insufficient quota", "no credits remaining",
            "credit balance", "exceeded your current quota", "quota exhausted",
        ))

    @staticmethod
    def _attach_error_details(error: Exception, details: dict[str, Any], category: str) -> Exception:
        error.normalized_category = category  # type: ignore[attr-defined]
        error.provider_code = details.get("code")  # type: ignore[attr-defined]
        error.provider_request_id = details.get("request_id")  # type: ignore[attr-defined]
        error.retry_after = details.get("retry_after")  # type: ignore[attr-defined]
        error.rate_limit_headers = details.get("headers")  # type: ignore[attr-defined]
        error.status_code = details.get("status_code")  # type: ignore[attr-defined]
        return error

    async def _sdk_request(self, *, timeout: int, agent: str = "", **kwargs):
        concurrency = (
            self._catalog_concurrency
            if agent == "assessment_compiler"
            else self._concurrency
        )
        await asyncio.to_thread(concurrency.acquire)
        try:
            async with AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.url,
                timeout=timeout,
                max_retries=0,
            ) as client:
                return await client.responses.create(**kwargs)
        finally:
            concurrency.release()

    async def _request_with_backoff(
        self, *, timeout: int, agent: str = "", **kwargs
    ):
        self.last_transient_retries = 0
        account_scope = self._account_scope()
        with self._quota_lock:
            quota_blocked = account_scope in self._quota_blocked_accounts
        if quota_blocked:
            raise self._attach_error_details(
                self.quota_error_type(
                    f"{self.provider_label} account has confirmed insufficient quota"
                ), {}, "INSUFFICIENT_QUOTA",
            )
        for transient_attempt in range(DEEPSEEK_MAX_TRANSIENT_RETRIES + 1):
            request_event_id = str(uuid.uuid4())
            request_started = time.perf_counter()
            try:
                return await self._sdk_request(
                    timeout=timeout, agent=agent, **kwargs
                )
            except AuthenticationError as exc:
                raise self.authentication_error_type(
                    f"{self.provider_label} rejected its configured API key"
                ) from exc
            except RateLimitError as exc:
                details = self._provider_error_details(exc)
                if self._is_insufficient_quota(details):
                    with self._quota_lock:
                        self._quota_blocked_accounts.add(account_scope)
                    mapped_quota = self._attach_error_details(
                        self.quota_error_type(
                            f"{self.provider_label} reported insufficient quota"
                        ), details, "INSUFFICIENT_QUOTA",
                    )
                    self._record_request_event(
                        operation=agent or "provider_request", agent_role=agent,
                        request_id=request_event_id,
                        provider_request_id=details.get("request_id"),
                        status="failed", error_category="INSUFFICIENT_QUOTA",
                        provider_error_code=details.get("code"),
                        sanitized_error_message=details.get("message"),
                        retry_count=transient_attempt,
                        usage_available=False,
                        duration_ms=int((time.perf_counter() - request_started) * 1000),
                        metadata={"rate_limit_headers": details.get("headers") or {}},
                    )
                    raise mapped_quota from exc
                cause = exc
                mapped: Exception = self._attach_error_details(
                    self.rate_limit_error_type(
                        f"{self.provider_label} rate limit exceeded"
                    ), details, "RATE_LIMITED",
                )
                try:
                    from app.llm.scheduler import register_provider_cooldown
                    register_provider_cooldown(
                        self.provider_name, float(details.get("retry_after") or 0)
                    )
                except (TypeError, ValueError):
                    pass
            except APITimeoutError as exc:
                cause = exc
                mapped = self._attach_error_details(
                    self.timeout_error_type(f"{self.provider_label} request timed out"),
                    {}, "PROVIDER_UNAVAILABLE",
                )
            except APIConnectionError as exc:
                cause = exc
                mapped = self._attach_error_details(
                    self.network_error_type(
                        f"{self.provider_label} network connection failed"
                    ), {}, "PROVIDER_UNAVAILABLE",
                )
            except BadRequestError:
                raise
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise self.error_type(
                        f"{self.provider_label} API rejected the request with status {exc.status_code}"
                    ) from exc
                cause = exc
                mapped = self._attach_error_details(
                    self.server_error_type(
                        f"{self.provider_label} server error ({exc.status_code})"
                    ), {"status_code": exc.status_code}, "PROVIDER_UNAVAILABLE",
                )

            self._record_request_event(
                operation=agent or "provider_request", agent_role=agent,
                request_id=request_event_id,
                provider_request_id=getattr(mapped, "provider_request_id", None),
                status="failed",
                error_category=getattr(mapped, "normalized_category", type(mapped).__name__),
                provider_error_code=getattr(mapped, "provider_code", None),
                sanitized_error_message=str(mapped),
                retry_count=transient_attempt,
                retry_after_seconds=(
                    float(getattr(mapped, "retry_after"))
                    if str(getattr(mapped, "retry_after", "")).replace(".", "", 1).isdigit()
                    else None
                ),
                usage_available=False,
                duration_ms=int((time.perf_counter() - request_started) * 1000),
                metadata={
                    "rate_limit_headers": getattr(mapped, "rate_limit_headers", {}) or {},
                },
            )
            if transient_attempt >= DEEPSEEK_MAX_TRANSIENT_RETRIES:
                raise mapped from cause
            self.last_transient_retries = transient_attempt + 1
            delay = DEEPSEEK_RETRY_BASE_SECONDS * (2 ** transient_attempt)
            if isinstance(cause, RateLimitError):
                response = getattr(cause, "response", None)
                headers = getattr(response, "headers", {}) or {}
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                try:
                    delay = max(delay, min(60.0, float(retry_after)))
                except (TypeError, ValueError):
                    pass
            logger.warning(
                "%s transient failure | type=%s | retry=%d/%d | delay=%.2fs",
                self.provider_label,
                type(mapped).__name__,
                transient_attempt + 1,
                DEEPSEEK_MAX_TRANSIENT_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)

    @staticmethod
    def _reasoning_effort(agent: str) -> str:
        if agent == "planner":
            # The planner still receives deliberate reasoning, but "high" consumed most of
            # the 25k structured-output allowance before emitting the plan on large selections.
            return "low"
        if agent.startswith("global_validator"):
            # The full registry already externalizes the comparison task. Even
            # medium reasoning consumed the entire 12k allowance before emitting
            # structured JSON in a real 28-question run.
            return "none"
        if agent == "repairer":
            return "low"
        return "high" if agent == "validator" else "none"

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
        if agent.startswith("global_validator"):
            return min(12000, max(4096, count * 320)), 20000
        if agent == "repairer":
            return min(16000, max(4096, count * 512)), 16000
        return min(8000, max(2048, count * 320)), 8000

    def _capture_usage(
        self, response: Any, agent: str, *, duration_ms: int | None = None,
        record_request_event: bool = True,
    ) -> dict[str, Any] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
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
        uncached_input = max(0, int(values["input_tokens"]) - int(values["cached_tokens"]))
        estimated_cost = (
            uncached_input * DEEPSEEK_INPUT_COST_PER_MTOK
            + int(values["cached_tokens"]) * DEEPSEEK_CACHED_INPUT_COST_PER_MTOK
            + int(values["output_tokens"]) * DEEPSEEK_OUTPUT_COST_PER_MTOK
        ) / 1_000_000
        values["estimated_cost_usd"] = estimated_cost
        increment(
            "genexam_llm_tokens",
            value=int(values["input_tokens"]),
            provider=self.provider_name,
            model=self.model,
            direction="input",
        )
        increment(
            "genexam_llm_tokens",
            value=int(values["output_tokens"]),
            provider=self.provider_name,
            model=self.model,
            direction="output",
        )
        increment(
            "genexam_llm_estimated_cost_usd",
            value=estimated_cost,
            provider=self.provider_name,
            model=self.model,
        )
        try:
            from app.api import repositories

            repositories.record_llm_usage(
                job_id=current_job_id(),
                provider=self.provider_name,
                model=self.model,
                operation=agent,
                input_tokens=int(values["input_tokens"]),
                output_tokens=int(values["output_tokens"]),
                cached_tokens=int(values["cached_tokens"]),
                reasoning_tokens=int(values["reasoning_tokens"]),
                estimated_cost_usd=estimated_cost,
                duration_ms=duration_ms,
            )
            if record_request_event:
                repositories.record_llm_request_event(
                    job_id=current_job_id(), provider=self.provider_name,
                    model=self.model, operation=agent, agent_role=agent,
                    provider_request_id=str(getattr(response, "id", "") or "") or None,
                    status="success", usage_available=True,
                    input_tokens=int(values["input_tokens"]),
                    output_tokens=int(values["output_tokens"]),
                    reasoning_tokens=int(values["reasoning_tokens"]),
                    estimated_cost_usd=estimated_cost, duration_ms=duration_ms,
                    metadata={"cached_tokens": int(values["cached_tokens"])},
                )
        except Exception as exc:
            logger.warning("LLM usage persistence failed | error=%s", type(exc).__name__)
        logger.info(
            "DeepSeek usage | agent=%s | input_tokens=%d | output_tokens=%d | "
            "reasoning_tokens=%d | cached_tokens=%d | total_tokens=%d",
            agent,
            values["input_tokens"],
            values["output_tokens"],
            values["reasoning_tokens"],
            values["cached_tokens"],
            values["total_tokens"],
            extra={
                "event": "llm_call_completed",
                "provider": self.provider_name,
                "model": self.model,
                "input_tokens": int(values["input_tokens"]),
                "output_tokens": int(values["output_tokens"]),
                "total_tokens": int(values["total_tokens"]),
                "duration_ms": duration_ms,
            },
        )
        return dict(values)

    @staticmethod
    def _bounded_response_diagnostic(value: str) -> dict[str, Any]:
        text = " ".join(str(value or "").split())
        # Provider output should not contain credentials, but redact common
        # bearer/key forms before retaining a bounded debugging excerpt.
        text = re.sub(
            r"(?i)(bearer\s+|api[_ -]?key\s*[:=]\s*)[^\s,;]+",
            r"\1[REDACTED]",
            text,
        )
        return {
            "raw_response_excerpt": text[:2000],
            "raw_response_length": len(text),
            "raw_response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "raw_response_truncated": len(text) > 2000,
        }

    def _record_structured_attempt(
        self, *, operation: str, agent: str, request_id: str, attempt: int,
        schema_name: str, duration_ms: int, status: str,
        response: Any | None = None, response_status: str | None = None,
        provider_request_id: str | None = None,
        error_category: str | None = None, error_message: str | None = None,
        provider_error_code: str | None = None,
        usage: dict[str, Any] | None = None, raw_output: str = "",
    ) -> None:
        metadata: dict[str, Any] = {
            "event_kind": "structured_output_attempt",
            "attempt_number": attempt,
            "schema_name": schema_name,
            "response_status": response_status,
            "cached_tokens": int(usage["cached_tokens"]) if usage else None,
            **self._bounded_response_diagnostic(raw_output),
        }
        detail = {
            "provider": self.provider_name, "model": self.model,
            "request_id": request_id,
            "provider_request_id": (
                provider_request_id
                or str(getattr(response, "id", "") or "")
                or None
            ),
            "attempt_number": attempt, "status": status,
            "failure_classification": error_category,
            "error": str(error_message or "")[:1000] or None,
            "duration_ms": duration_ms, "usage_available": usage is not None,
            "input_tokens": int(usage["input_tokens"]) if usage else None,
            "output_tokens": int(usage["output_tokens"]) if usage else None,
            "cached_tokens": int(usage["cached_tokens"]) if usage else None,
            "reasoning_tokens": int(usage["reasoning_tokens"]) if usage else None,
            "estimated_cost_usd": float(usage["estimated_cost_usd"]) if usage else None,
            **metadata,
        }
        self.last_structured_attempt_details.append(detail)
        self._record_request_event(
            operation=operation, agent_role=agent, request_id=request_id,
            provider_request_id=detail["provider_request_id"], status=status,
            error_category=error_category, provider_error_code=provider_error_code,
            sanitized_error_message=error_message, retry_count=attempt - 1,
            usage_available=usage is not None,
            input_tokens=detail["input_tokens"], output_tokens=detail["output_tokens"],
            reasoning_tokens=detail["reasoning_tokens"],
            estimated_cost_usd=detail["estimated_cost_usd"], duration_ms=duration_ms,
            metadata=metadata,
        )

    @staticmethod
    def _output_text(response: Any) -> str:
        return str(getattr(response, "output_text", "") or "").strip()

    def _validate_expected_ids(
        self, value: Any, expected_ids: list[str] | None, id_field: str | None
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
            raise self.validation_error_type(
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
        # The provider already receives the authoritative JSON schema through
        # ``text.format`` on every attempt. Repeating that schema and an
        # unbounded prior response caused retry prompts to grow beyond the
        # original request. Preserve the task, but cap diagnostic material.
        task = original_prompt[:24000]
        prior = previous_output[:8000]
        reason = failure_reason[:2000]
        return (
            f"{task}\n\n"
            "## Corrective structured-output retry\n"
            f"Previous response (bounded excerpt):\n{prior or '[no complete output returned]'}\n\n"
            f"Previous response failed because:\n{reason}\n\n"
            "Correct only that specific failure while preserving every valid field, "
            "ID, concept, difficulty, question type, and other correct data. This is "
            "the same logical task and must not create an extra question or plan slot.\n"
            f"Expected IDs: {expected_ids or 'defined by the provider schema'}\n"
            "Return output matching the same provider-supplied JSON Schema."
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
        max_tokens: int | None = None,
        timeout: int = 600,
    ) -> object:
        try:
            schema = strict_json_schema(response_model)
        except ValueError as exc:
            raise self.schema_error_type(str(exc)) from exc
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", schema_name)[:64] or "response"
        default_tokens, token_ceiling = self._token_budget(agent, item_count, operation)
        max_tokens = max(1, int(max_tokens or default_tokens))
        token_ceiling = max(token_ceiling, max_tokens)
        reasoning_effort = self._reasoning_effort(agent)
        previous_output = ""
        failure_reason = ""
        self.last_structured_attempt_details = []

        attempt_limit = (
            1 if (
                (agent == "planner" and operation == "initial_planner")
                or agent in {"assessment_compiler", "duplicate_validator"}
            )
            else DEEPSEEK_MAX_STRUCTURED_ATTEMPTS
        )
        for attempt in range(1, attempt_limit + 1):
            self.last_structured_attempts = attempt
            attempt_request_id = str(uuid.uuid4())
            current_input = (
                prompt
                if attempt == 1
                else self._corrective_input(
                    prompt, previous_output, failure_reason, schema, expected_ids
                )
            )
            logger.info(
                "%s structured call | agent=%s | model=%s | schema=%s | "
                "reasoning=%s | attempt=%d/%d | max_output_tokens=%d",
                self.provider_label,
                agent,
                self.model,
                safe_name,
                reasoning_effort,
                attempt,
                attempt_limit,
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
                request_started = time.perf_counter()
                request_kwargs = dict(
                    timeout=timeout,
                    model=self.model,
                    instructions=structured_instructions,
                    input=current_input,
                    reasoning={"effort": reasoning_effort},
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
                if self.supports_temperature:
                    request_kwargs["temperature"] = temperature
                response = await self._request_with_backoff(
                    agent=agent, **request_kwargs
                )
            except BadRequestError as exc:
                message = str(exc)
                details = self._provider_error_details(exc)
                schema_rejected = "schema" in message.lower() or "text.format" in message.lower()
                self._record_structured_attempt(
                    operation=operation, agent=agent, request_id=attempt_request_id,
                    attempt=attempt, schema_name=safe_name,
                    duration_ms=int((time.perf_counter() - request_started) * 1000),
                    status="failed", response_status="request_rejected",
                    provider_request_id=details.get("request_id"),
                    error_category=("SCHEMA_REJECTED" if schema_rejected else "REQUEST_REJECTED"),
                    provider_error_code=details.get("code"),
                    error_message=details.get("message") or message,
                )
                if schema_rejected:
                    raise self.schema_error_type(
                        f"{self.provider_label} rejected the normalized JSON Schema"
                    ) from exc
                raise self.error_type(
                    f"{self.provider_label} rejected the structured request"
                ) from exc
            except Exception as exc:
                self._record_structured_attempt(
                    operation=operation, agent=agent, request_id=attempt_request_id,
                    attempt=attempt, schema_name=safe_name,
                    duration_ms=int((time.perf_counter() - request_started) * 1000),
                    status="failed", response_status="transport_error",
                    provider_request_id=getattr(exc, "provider_request_id", None),
                    error_category=str(
                        getattr(exc, "normalized_category", "PROVIDER_ERROR")
                    ),
                    provider_error_code=getattr(exc, "provider_code", None),
                    error_message=str(exc),
                )
                raise

            duration_ms = int((time.perf_counter() - request_started) * 1000)
            usage_values = self._capture_usage(
                response,
                agent,
                duration_ms=duration_ms,
                record_request_event=False,
            )
            previous_output = self._output_text(response)
            status = str(getattr(response, "status", "") or "")
            if status == "incomplete":
                details = getattr(response, "incomplete_details", None)
                reason = str(getattr(details, "reason", "unknown") or "unknown")
                failure_reason = f"response incomplete: {reason}"
                self._record_structured_attempt(
                    operation=operation, agent=agent, request_id=attempt_request_id,
                    attempt=attempt, schema_name=safe_name, duration_ms=duration_ms,
                    status="failed", response=response, response_status=status,
                    error_category="INCOMPLETE_RESPONSE", error_message=failure_reason,
                    usage=usage_values, raw_output=previous_output,
                )
                if reason == "max_output_tokens":
                    max_tokens = min(token_ceiling, max_tokens * 2)
                if attempt == attempt_limit:
                    raise self.incomplete_error_type(
                        failure_reason,
                        partial_output=previous_output,
                        reason=reason,
                    )
                continue
            if status == "failed":
                error = getattr(response, "error", None)
                code = str(getattr(error, "code", "unknown") or "unknown")
                failure_reason = f"response failed: {code}"
                self._record_structured_attempt(
                    operation=operation, agent=agent, request_id=attempt_request_id,
                    attempt=attempt, schema_name=safe_name, duration_ms=duration_ms,
                    status="failed", response=response, response_status=status,
                    error_category="PROVIDER_RESPONSE_FAILED", provider_error_code=code,
                    error_message=failure_reason, usage=usage_values,
                    raw_output=previous_output,
                )
                if attempt == attempt_limit:
                    raise self.error_type(failure_reason)
                continue
            if status and status != "completed":
                failure_reason = f"unexpected response status: {status}"
                self._record_structured_attempt(
                    operation=operation, agent=agent, request_id=attempt_request_id,
                    attempt=attempt, schema_name=safe_name, duration_ms=duration_ms,
                    status="failed", response=response, response_status=status,
                    error_category="UNEXPECTED_RESPONSE_STATUS",
                    error_message=failure_reason, usage=usage_values,
                    raw_output=previous_output,
                )
                if attempt == attempt_limit:
                    raise self.incomplete_error_type(
                        failure_reason,
                        partial_output=previous_output,
                        reason=status,
                    )
                continue

            try:
                validated = response_model.model_validate_json(previous_output)
                value = validated.model_dump()
                self._validate_expected_ids(value, expected_ids, expected_id_field)
                self._record_structured_attempt(
                    operation=operation, agent=agent, request_id=attempt_request_id,
                    attempt=attempt, schema_name=safe_name, duration_ms=duration_ms,
                    status="success", response=response,
                    response_status=status or "completed", usage=usage_values,
                    raw_output=previous_output,
                )
                return value
            except (ValidationError, json.JSONDecodeError, self.validation_error_type) as exc:
                failure_reason = str(exc)
                classification = (
                    "INVALID_JSON" if (
                        isinstance(exc, json.JSONDecodeError)
                        or (
                            isinstance(exc, ValidationError)
                            and any(
                                error.get("type") == "json_invalid"
                                for error in exc.errors()
                            )
                        )
                    )
                    else "SCHEMA_VALIDATION_FAILED"
                )
                self._record_structured_attempt(
                    operation=operation, agent=agent, request_id=attempt_request_id,
                    attempt=attempt, schema_name=safe_name, duration_ms=duration_ms,
                    status="failed", response=response,
                    response_status=status or "completed",
                    error_category=classification, error_message=failure_reason,
                    usage=usage_values, raw_output=previous_output,
                )
                if attempt == attempt_limit:
                    raise self.validation_error_type(
                        f"{self.provider_label} structured output failed local Pydantic validation "
                        f"after {attempt_limit} attempts: {failure_reason}",
                        partial_output=previous_output,
                        reason="schema_validation",
                    ) from exc

        raise self.validation_error_type(
            f"{self.provider_label} structured output failed"
        )

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
        max_tokens: int | None = None,
        timeout: int = 600,
    ) -> object:
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
                max_tokens=max_tokens,
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
        request_started = time.perf_counter()
        response = await self._request_with_backoff(
            timeout=timeout,
            model=self.model,
            instructions=system_prompt,
            input=prompt,
            reasoning={"effort": "none"},
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        self._capture_usage(
            response,
            "title",
            duration_ms=int((time.perf_counter() - request_started) * 1000),
        )
        status = str(getattr(response, "status", "") or "")
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            raise self.incomplete_error_type(
                f"{self.provider_label} text response incomplete: "
                f"{getattr(details, 'reason', 'unknown')}"
            )
        if status != "completed":
            raise self.error_type(
                f"{self.provider_label} text response status was {status or 'unknown'}"
            )
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
        raise self.configuration_error_type(
            f"{self.provider_label} JSON operations require chat_structured "
            "with a Pydantic schema"
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
            raise self.authentication_error_type(
                f"{self.provider_label} rejected its configured API key"
            ) from exc
        finally:
            self._concurrency.release()

    def health(self, timeout: int = 3) -> None:
        _run_sync(self.ahealth(timeout=timeout))
