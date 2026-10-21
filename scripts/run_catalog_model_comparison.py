"""Run one isolated, budget-bounded Assessment Catalog comparison.

This utility deliberately bypasses application cache and database persistence.  It
loads the immutable processed-document cache, compiles only Catalog batches, and
writes an auditable experiment directory.  It is not part of the exam pipeline.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# ``python scripts/...`` puts scripts/ on sys.path, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.llm.deepseek import (
    DeepSeekClient,
    DeepSeekIncompleteError,
    DeepSeekValidationError,
)
from app.llm.openai_client import OpenAIClient
from app.llm.schemas import strict_json_schema
from app.online.assessment_catalog import (
    CATALOG_CONTRACT_VERSION,
    CompactCatalogBatchResponse,
    CompactCatalogBatchWithProfileResponse,
    _balanced_catalog_batches,
    _catalog_prompt,
    _finalize_catalog,
    _merge_catalog_batches,
    _validate_completed_batch,
)
from app.online.retrieval import build_planner_context


DOCUMENT_KEY = "9b7c21316e78ec49cb5cf54b48b5e687183729aff44330392167772ad9e12115"
DOCUMENT_ID = "8928f477-3481-47ed-ac0d-1bb855f389b2"
BATCH_COUNT = 12
BUDGET_USD = 0.50
# This is an experiment-only cap. It is shared by both models and allows the
# three outer attempts to be reserved below $0.50 at published peak prices.
EXPERIMENT_MAX_OUTPUT_TOKENS = 6500
INPUT_SAFETY_FACTOR = 1.20

# Worst-case USD per 1M tokens.  DeepSeek uses its published peak rate because
# the provider does not expose a billable-price field in a response.  OpenAI's
# direct Standard short-context rate is used; both Catalog prompts are short
# context by the published definition.
PRICE = {
    "deepseek": {"input": 0.30, "cached": 0.006, "output": 1.20},
    "openai": {"input": 0.10, "cached": 0.01, "output": 0.50},
}


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+", r"\1[redacted]", text)
    return text[:1000]


def _json_default(value: Any) -> str:
    return str(value)


class _NoPersistenceUsageMixin:
    """Keep exact provider usage locally without touching the application DB."""

    def _capture_usage(self, response: Any, agent: str, *, duration_ms: int | None = None) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            self.last_usage = {}
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        self.last_usage = {
            "agent": agent,
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
            "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "provider_duration_ms": int(duration_ms or 0),
        }
        history = getattr(self, "usage_history", None)
        if isinstance(history, list):
            history.append(dict(self.last_usage))


class BenchmarkDeepSeekClient(_NoPersistenceUsageMixin, DeepSeekClient):
    _concurrency = threading.BoundedSemaphore(config.DEEPSEEK_CONCURRENCY_LIMIT)


class BenchmarkOpenAIClient(_NoPersistenceUsageMixin, OpenAIClient):
    # Do not make this benchmark's two providers contend on DeepSeekClient's
    # inherited class semaphore. Each retains the project limit of ten.
    _concurrency = threading.BoundedSemaphore(config.MAX_PARALLEL_CATALOG_CALLS)


@dataclass
class BatchOutcome:
    model_label: str
    provider: str
    model: str
    batch_id: int
    chunk_ids: list[str]
    started_at: str = ""
    finished_at: str = ""
    status: str = "pending"
    attempts: int = 0
    tasks_returned: int = 0
    fitb_entries: int = 0
    error: str = ""
    result: dict[str, Any] | None = None


@dataclass
class ModelOutcome:
    label: str
    provider: str
    model: str
    started_at: str = ""
    finished_at: str = ""
    status: str = "pending"
    catalog: dict[str, Any] | None = None
    batches: list[BatchOutcome] = field(default_factory=list)
    error: str = ""


class CallRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def add(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.rows.append(row)


def _load_source() -> tuple[dict[str, Any], list[dict[str, Any]], list[list[dict[str, Any]]]]:
    files = list(Path(config.PROCESSED_DOCUMENT_CACHE_DIR).rglob(f"{DOCUMENT_KEY}.json.gz"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one immutable processed cache for {DOCUMENT_KEY}; found {len(files)}")
    with gzip.open(files[0], "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("processed_document_key") != DOCUMENT_KEY:
        raise RuntimeError("Processed cache identity does not match the required historical key")
    chunks = list((payload.get("structure") or {}).get("children") or [])
    if len(chunks) != 89 or not all(chunk.get("child_id") and chunk.get("content") for chunk in chunks):
        raise RuntimeError("Required 89 complete child chunks are unavailable in the processed cache")
    batches = _balanced_catalog_batches(chunks, BATCH_COUNT)
    if len(batches) != BATCH_COUNT:
        raise RuntimeError(f"Expected {BATCH_COUNT} Catalog batches; constructed {len(batches)}")
    return payload, chunks, batches


def _prompt_token_estimate(prompts: list[str]) -> tuple[int, str]:
    try:
        import tiktoken  # type: ignore[import-not-found]

        encoder = tiktoken.get_encoding("o200k_base")
        return sum(len(encoder.encode(prompt)) for prompt in prompts), "o200k_base"
    except Exception:
        # Conservative fallback for an unavailable local tokenizer.
        return sum(len(prompt.encode("utf-8")) for prompt in prompts), "utf8_bytes_upper_bound"


def _reserved_budget(prompt_tokens: int) -> dict[str, float]:
    attempts = config.CATALOG_BATCH_MAX_ATTEMPTS
    estimated_input = prompt_tokens * INPUT_SAFETY_FACTOR * attempts
    output = BATCH_COUNT * EXPERIMENT_MAX_OUTPUT_TOKENS * attempts
    rows: dict[str, float] = {}
    for provider in ("deepseek", "openai"):
        rows[provider] = (
            estimated_input * PRICE[provider]["input"] + output * PRICE[provider]["output"]
        ) / 1_000_000
    rows["total"] = rows["deepseek"] + rows["openai"]
    return rows


def _client(provider: str):
    if provider == "deepseek":
        return BenchmarkDeepSeekClient(model="deepseek-flash")
    return BenchmarkOpenAIClient(
        model=config.STEM_REVIEW_MODEL,
        reasoning_effort=config.STEM_REVIEW_REASONING_EFFORT,
    )


def _cost(provider: str, usage: dict[str, Any]) -> float | None:
    if not usage:
        return None
    try:
        inp = int(usage.get("input_tokens") or 0)
        cached = int(usage.get("cached_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
    except (TypeError, ValueError):
        return None
    return ((max(0, inp - cached) * PRICE[provider]["input"])
            + (cached * PRICE[provider]["cached"])
            + (out * PRICE[provider]["output"])) / 1_000_000


def _compile_batch(
    *,
    label: str,
    provider: str,
    batch_index: int,
    batch: list[dict[str, Any]],
    language: str,
    subject: str,
    recorder: CallRecorder,
    queued_at: float,
) -> BatchOutcome:
    client = _client(provider)
    model = str(client.model)
    chunk_ids = [str(chunk["child_id"]) for chunk in batch]
    include_profile = batch_index == 0
    response_model = CompactCatalogBatchWithProfileResponse if include_profile else CompactCatalogBatchResponse
    prompt = _catalog_prompt(
        build_planner_context(batch), language, chunk_ids,
        domain_type="stem", subject=subject, include_profile=include_profile,
    )
    outcome = BatchOutcome(label, provider, model, batch_index + 1, chunk_ids)
    outcome.started_at = _utc()
    for attempt in range(1, config.CATALOG_BATCH_MAX_ATTEMPTS + 1):
        started_perf = time.perf_counter()
        call_started = _utc()
        client.last_usage = None
        error = ""
        retry_reason = ""
        value: dict[str, Any] | None = None
        try:
            result = client.chat_structured(
                prompt,
                response_model=response_model,
                json_schema=strict_json_schema(response_model),
                schema_name="assessment_catalog_batch",
                agent="assessment_compiler",
                operation="assessment_catalog_batch",
                item_count=max(1, len(batch)),
                system_prompt="Compile a complete, non-redundant assessment ontology grounded only in this evidence batch.",
                temperature=0.1,
                max_tokens=EXPERIMENT_MAX_OUTPUT_TOKENS,
            )
            value = _validate_completed_batch(
                result, response_model=response_model, valid_chunk_ids=set(chunk_ids),
            )
        except (DeepSeekIncompleteError, DeepSeekValidationError, ValueError, TypeError) as exc:
            error = _safe_error(exc)
            retry_reason = "incomplete_or_invalid_structured_output"
        except Exception as exc:  # provider failure is part of the comparison
            error = _safe_error(exc)
            retry_reason = "provider_or_transport_failure"
        call_finished = _utc()
        usage = dict(getattr(client, "last_usage", None) or {})
        row = {
            "model_label": label, "provider": provider, "model": model,
            "batch_id": batch_index + 1, "attempt": attempt, "chunk_count": len(batch),
            "chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),
            "queue_wait_ms": round((started_perf - queued_at) * 1000, 3),
            "start_timestamp": call_started, "end_timestamp": call_finished,
            "request_duration_ms": round((time.perf_counter() - started_perf) * 1000, 3),
            "input_tokens": usage.get("input_tokens", "not_available"),
            "cached_input_tokens": usage.get("cached_tokens", "not_available"),
            "output_tokens": usage.get("output_tokens", "not_available"),
            "reasoning_tokens": usage.get("reasoning_tokens", "not_available"),
            "total_tokens": usage.get("total_tokens", "not_available"),
            "provider_internal_structured_attempts": getattr(client, "last_structured_attempts", "not_available"),
            "provider_internal_transient_retries": getattr(client, "last_transient_retries", "not_available"),
            "schema_status": "valid" if value is not None else "invalid_or_unavailable",
            "completion_status": "complete" if value is not None else "incomplete_or_failed",
            "tasks_returned": len((value or {}).get("tasks") or []),
            "fitb_entries": len((value or {}).get("fitb_bank") or []),
            "tasks_accepted": "pending_finalization" if value is not None else 0,
            "error": error, "retry_reason": retry_reason,
            "api_cost_usd_estimated_peak": _cost(provider, usage),
        }
        recorder.add(row)
        outcome.attempts = attempt
        if value is not None:
            outcome.result = value
            outcome.tasks_returned = len(value.get("tasks") or [])
            outcome.fitb_entries = len(value.get("fitb_bank") or [])
            outcome.status = "complete"
            break
        outcome.error = error
    outcome.finished_at = _utc()
    if outcome.result is None:
        outcome.status = "failed"
    return outcome


def _compile_model(
    *, label: str, provider: str, batches: list[list[dict[str, Any]]],
    language: str, subject: str, recorder: CallRecorder, start_gate: threading.Barrier,
) -> ModelOutcome:
    test_client = _client(provider)
    model = str(test_client.model)
    outcome = ModelOutcome(label, provider, model)
    start_gate.wait()
    outcome.started_at = _utc()
    submitted_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_CATALOG_CALLS) as executor:
        futures = [executor.submit(
            _compile_batch, label=label, provider=provider, batch_index=index, batch=batch,
            language=language, subject=subject, recorder=recorder, queued_at=submitted_at,
        ) for index, batch in enumerate(batches)]
        outcome.batches = [future.result() for future in as_completed(futures)]
    outcome.batches.sort(key=lambda row: row.batch_id)
    failures = [row for row in outcome.batches if row.status != "complete"]
    if failures:
        outcome.status = "failed"
        outcome.error = "; ".join(f"batch {row.batch_id}: {row.error}" for row in failures)
        outcome.finished_at = _utc()
        return outcome
    raw = _merge_catalog_batches([row.result or {} for row in outcome.batches])
    if raw.get("catalog_compilation_complete") is not True:
        outcome.status = "failed"
        outcome.error = "merged catalog is incomplete"
        outcome.finished_at = _utc()
        return outcome
    selected_ids = [str(chunk["child_id"]) for batch in batches for chunk in batch]
    outcome.catalog = _finalize_catalog(
        raw, key=f"experiment-{provider}", processed_document_key=DOCUMENT_KEY,
        selected_chunk_ids=selected_ids, usage={"experiment_only": True},
    )
    outcome.status = "complete"
    outcome.finished_at = _utc()
    return outcome


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _duration_seconds(start: str, finish: str) -> float:
    return round((_parse_timestamp(finish) - _parse_timestamp(start)).total_seconds(), 3)


def _catalog_quality(outcome: ModelOutcome, valid_ids: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog = outcome.catalog or {}
    tasks = list(catalog.get("tasks") or [])
    findings: list[dict[str, Any]] = []
    concepts: set[str] = set()
    evidence: set[str] = set()
    type_counts: dict[str, int] = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        task_evidence = [str(value) for value in task.get("evidence_chunk_ids") or []]
        concepts.add(str(task.get("concept") or "").strip().casefold())
        evidence.update(task_evidence)
        for qtype in task.get("allowed_question_types") or []:
            type_counts[str(qtype)] = type_counts.get(str(qtype), 0) + 1
        if not task.get("text_answerable"):
            findings.append({"model_label": outcome.label, "task_id": task_id, "severity": "error", "category": "answerability", "finding": "Finalized task is not explicitly answerable."})
        if not task_evidence or not set(task_evidence).issubset(valid_ids):
            findings.append({"model_label": outcome.label, "task_id": task_id, "severity": "error", "category": "evidence", "finding": "Task cites no valid selected evidence."})
        if not task.get("allowed_question_types"):
            findings.append({"model_label": outcome.label, "task_id": task_id, "severity": "error", "category": "compatibility", "finding": "Task has no compatible question type."})
        for variation in task.get("allowed_variations") or []:
            if not isinstance(variation, dict) or not variation.get("target") or not variation.get("operation"):
                findings.append({"model_label": outcome.label, "task_id": task_id, "severity": "error", "category": "variation", "finding": "Malformed finalized variation."})
        verbose = [name for name in ("knowledge_anchor", "concept", "relationship", "student_work_product") if len(str(task.get(name) or "")) > 200]
        if verbose:
            findings.append({"model_label": outcome.label, "task_id": task_id, "severity": "warning", "category": "concision", "finding": f"Long metadata values: {', '.join(verbose)}."})
    return {
        "raw_tasks": sum(row.tasks_returned for row in outcome.batches),
        "accepted_tasks": len(tasks), "fitb_entries": len(catalog.get("fitb_bank") or []),
        "distinct_concepts": len([concept for concept in concepts if concept]),
        "distinct_evidence_chunks": len(evidence), "evidence_coverage_pct": round(100 * len(evidence) / len(valid_ids), 2),
        "word_problem_tasks": type_counts.get("word_problem", 0), "type_counts": type_counts,
    }, findings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader(); writer.writerows(rows)


def _git_status() -> dict[str, Any]:
    def call(*args: str) -> str:
        return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()
    return {"branch": call("git", "branch", "--show-current"), "status": call("git", "status", "--short")}


def _metrics(outcome: ModelOutcome, calls: list[dict[str, Any]], quality: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in calls if row["model_label"] == outcome.label]
    numbers = lambda key: sum(int(row.get(key) or 0) for row in rows if str(row.get(key)).isdigit())
    durations = [float(row["request_duration_ms"]) / 1000 for row in rows]
    costs = [float(row["api_cost_usd_estimated_peak"]) for row in rows if row.get("api_cost_usd_estimated_peak") is not None]
    return {
        "compiler_wall_seconds": _duration_seconds(outcome.started_at, outcome.finished_at),
        "provider_requests": len(rows), "successful_batches": sum(row.status == "complete" for row in outcome.batches),
        "failed_batches": sum(row.status != "complete" for row in outcome.batches),
        "outer_retry_attempts": max(0, len(rows) - len(outcome.batches)),
        "input_tokens": numbers("input_tokens"), "cached_input_tokens": numbers("cached_input_tokens"),
        "output_tokens": numbers("output_tokens"), "reasoning_tokens": numbers("reasoning_tokens"),
        "total_tokens": numbers("total_tokens"), "cost_usd_estimated_peak": round(sum(costs), 6),
        "average_successful_batch_seconds": round(sum(durations) / max(1, sum(row.status == "complete" for row in outcome.batches)), 3),
        "median_call_seconds": round(sorted(durations)[len(durations) // 2], 3) if durations else None,
        "maximum_call_seconds": round(max(durations), 3) if durations else None,
        **quality,
    }


def _report(config_data: dict[str, Any], outcomes: list[ModelOutcome], metrics: dict[str, dict[str, Any]], findings: list[dict[str, Any]], calls: list[dict[str, Any]]) -> str:
    lines = [
        "# Assessment Catalog model comparison",
        "", "## Executive summary", "",
        "This isolated experiment ran only the Assessment Catalog compiler against the immutable `math (2).pdf` processed cache. No planner, generation, validation, repair, duplicate, database, or application-cache operation ran.",
        "", "## Configuration", "",
        f"- Contract: `{CATALOG_CONTRACT_VERSION}`", f"- Outer Catalog batch attempts: {config.CATALOG_BATCH_MAX_ATTEMPTS} (provider-neutral)",
        f"- Batches/chunks: {BATCH_COUNT}/89; per-model concurrency: {config.MAX_PARALLEL_CATALOG_CALLS}",
        f"- Experiment-only output cap: {EXPERIMENT_MAX_OUTPUT_TOKENS} per request; the same cap was applied to both models to reserve the $0.50 limit.",
        f"- Reserved worst-case estimate: ${config_data['budget_reservation']['total']:.6f} at published peak rates.",
        "", "## Performance", "",
        "| Metric | DeepSeek V4.1 Flash | GPT-6 Luna |", "|---|---:|---:|",
    ]
    left, right = metrics["DeepSeek V4.1 Flash"], metrics["GPT-6 Luna"]
    for key, label in [
        ("compiler_wall_seconds", "Compiler wall time (s)"), ("provider_requests", "Observable provider requests"),
        ("successful_batches", "Successful batches"), ("failed_batches", "Failed batches"),
        ("outer_retry_attempts", "Outer retries"), ("input_tokens", "Input tokens"),
        ("cached_input_tokens", "Cached input tokens"), ("output_tokens", "Output tokens"),
        ("reasoning_tokens", "Reasoning tokens"), ("total_tokens", "Total tokens"),
        ("cost_usd_estimated_peak", "Estimated peak-rate cost (USD)"),
        ("raw_tasks", "Raw tasks"), ("accepted_tasks", "Accepted tasks"),
        ("distinct_concepts", "Distinct normalized concepts"), ("distinct_evidence_chunks", "Evidence chunks cited"),
        ("evidence_coverage_pct", "Evidence chunk coverage (%)"), ("word_problem_tasks", "Word-problem-compatible tasks"),
    ]:
        lines.append(f"| {label} | {left.get(key, 'not available')} | {right.get(key, 'not available')} |")
    lines += ["", "## Reliability and quality checks", "", "The deterministic review verifies schema, completion, final answerability, selected-evidence membership, variation structure, final IDs, and question-type compatibility. It does not prove every semantic claim against Arabic source text; task-level findings are in `quality_findings.csv` for source-aware review.", ""]
    if findings:
        lines += [f"- Deterministic findings: {len(findings)}. See `quality_findings.csv`."]
    else:
        lines += ["- No deterministic task-contract findings."]
    lines += ["", "## Chronology", ""]
    for row in sorted(calls, key=lambda item: item["start_timestamp"]):
        lines.append(f"- {row['start_timestamp']} — {row['model_label']} batch {row['batch_id']} attempt {row['attempt']} started; {row['completion_status']}; {row['request_duration_ms']} ms.")
    lines += ["", "## Limits", "", "Costs are provider-reported token usage priced with published peak rates where the API does not return billing. Provider-internal transient retries are listed per call; failed transport attempts may have no usage record. This is one run per model, so quality and latency differences are indicative rather than statistically conclusive."]
    return "\n".join(lines) + "\n"


def main() -> int:
    global EXPERIMENT_MAX_OUTPUT_TOKENS, BUDGET_USD
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", choices=("both", "luna"), default="both")
    parser.add_argument("--max-output-tokens", type=int, default=EXPERIMENT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--no-budget-cap", action="store_true")
    args = parser.parse_args()
    EXPERIMENT_MAX_OUTPUT_TOKENS = max(1, int(args.max_output_tokens))
    if args.no_budget_cap:
        BUDGET_USD = float("inf")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    payload, chunks, batches = _load_source()
    language = str((payload.get("metadata") or {}).get("document_language") or "ar")
    subject = str((payload.get("metadata") or {}).get("subject") or "mathematics")
    prompts = [_catalog_prompt(build_planner_context(batch), language, [str(c['child_id']) for c in batch], domain_type="stem", subject=subject, include_profile=index == 0) for index, batch in enumerate(batches)]
    prompt_tokens, tokenizer = _prompt_token_estimate(prompts)
    reservation = _reserved_budget(prompt_tokens)
    config_data = {
        "started_at": _utc(), "document_id": DOCUMENT_ID, "processed_document_key": DOCUMENT_KEY,
        "source_chunk_count": len(chunks), "batch_membership": [[str(c["child_id"]) for c in batch] for batch in batches],
        "contract": CATALOG_CONTRACT_VERSION, "compiler_version": config.ASSESSMENT_COMPILER_VERSION,
        "catalog_batch_max_attempts": config.CATALOG_BATCH_MAX_ATTEMPTS,
        "per_model_concurrency": config.MAX_PARALLEL_CATALOG_CALLS,
        "experiment_max_output_tokens": EXPERIMENT_MAX_OUTPUT_TOKENS,
        "prompt_token_estimate": prompt_tokens, "prompt_tokenizer": tokenizer,
        "budget_limit_usd": BUDGET_USD, "budget_reservation": reservation,
        "models": {"deepseek": "deepseek-flash (DeepSeek-V4.1-Flash-1017 alias)", "openai": config.STEM_REVIEW_MODEL},
        "reasoning": {"deepseek": "provider default for assessment_compiler (none)", "openai": config.STEM_REVIEW_REASONING_EFFORT},
        "git": _git_status(),
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(config_data, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_budget_cap and reservation["total"] > BUDGET_USD:
        (output_dir / "errors_and_retries.md").write_text(f"# Blocked\n\nReserved worst-case ${reservation['total']:.6f} exceeds ${BUDGET_USD:.2f}; no API requests were launched.\n", encoding="utf-8")
        return 2
    if not config.DEEPSEEK_API_KEY or not config.OPENAI_API_KEY:
        (output_dir / "errors_and_retries.md").write_text("# Blocked\n\nA required provider key is absent; no API requests were launched.\n", encoding="utf-8")
        return 2
    recorder = CallRecorder()
    selected = [("GPT-6 Luna", "openai")]
    if args.model == "both":
        selected.insert(0, ("DeepSeek V4.1 Flash", "deepseek"))
    gate = threading.Barrier(len(selected))
    started = _utc()
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = [executor.submit(
            _compile_model, label=label, provider=provider, batches=batches,
            language=language, subject=subject, recorder=recorder, start_gate=gate,
        ) for label, provider in selected]
        outcomes = [future.result() for future in futures]
    finished = _utc()
    config_data.update({"experiment_started_at": started, "experiment_finished_at": finished, "shared_wall_seconds": _duration_seconds(started, finished)})
    calls = sorted(recorder.rows, key=lambda row: row["start_timestamp"])
    _write_csv(output_dir / "model_calls.csv", calls)
    batch_rows = []
    all_findings: list[dict[str, Any]] = []
    metrics_by_label: dict[str, dict[str, Any]] = {}
    valid_ids = {str(chunk["child_id"]) for chunk in chunks}
    for outcome in outcomes:
        quality, findings = _catalog_quality(outcome, valid_ids)
        all_findings.extend(findings)
        metrics_by_label[outcome.label] = _metrics(outcome, calls, quality)
        for row in outcome.batches:
            batch_rows.append({"model_label": outcome.label, "provider": outcome.provider, "model": outcome.model, "batch_id": row.batch_id, "status": row.status, "attempts": row.attempts, "chunk_count": len(row.chunk_ids), "chunk_ids": json.dumps(row.chunk_ids, ensure_ascii=False), "tasks_returned": row.tasks_returned, "fitb_entries": row.fitb_entries, "error": row.error, "started_at": row.started_at, "finished_at": row.finished_at})
        if outcome.catalog is not None:
            filename = "deepseek_catalog.json" if outcome.provider == "deepseek" else "luna_catalog.json"
            (output_dir / filename).write_text(json.dumps(outcome.catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output_dir / "batch_results.csv", batch_rows)
    _write_csv(output_dir / "quality_findings.csv", all_findings)
    if len(outcomes) == 2:
        report = _report(config_data, outcomes, metrics_by_label, all_findings, calls)
    else:
        outcome = outcomes[0]
        report = "# Luna Assessment Catalog run\n\n" + json.dumps({
            "model": outcome.model, "provider": outcome.provider,
            "status": outcome.status, "metrics": metrics_by_label[outcome.label],
            "source_chunks": len(chunks), "batches": len(batches),
            "normal_production_max_output_tokens": EXPERIMENT_MAX_OUTPUT_TOKENS,
            "outer_batch_attempts": config.CATALOG_BATCH_MAX_ATTEMPTS,
        }, ensure_ascii=False, indent=2) + "\n"
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    error_lines = ["# Errors and retries", ""]
    for row in calls:
        if row["error"] or row["retry_reason"]:
            error_lines.append(f"- {row['start_timestamp']} | {row['model_label']} batch {row['batch_id']} attempt {row['attempt']} | {row['error'] or row['retry_reason']}")
    if len(error_lines) == 2:
        error_lines.append("No outer Catalog retry or observable provider failure occurred.")
    (output_dir / "errors_and_retries.md").write_text("\n".join(error_lines) + "\n", encoding="utf-8")
    config_data["outcomes"] = {outcome.label: {"status": outcome.status, "error": outcome.error, **metrics_by_label[outcome.label]} for outcome in outcomes}
    (output_dir / "experiment_config.json").write_text(json.dumps(config_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(outcome.status == "complete" for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
