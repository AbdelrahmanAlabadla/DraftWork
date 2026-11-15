"""Recovery V3 semantic strategy decisions using the Task Alignment reviewer.

This module decides strategy only.  Authorization, budgets, mutation, lifecycle
fencing, candidate validation, and Duplicate V2 remain Python responsibilities.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from typing import Any, Callable

from app import config
from app.llm.client import request_structured
from app.llm.factory import create_review_client
from app.llm.schemas import RecoveryDecisionResponse


SYSTEM_PROMPT = """You are the existing Task Alignment reviewer performing Recovery V3 decisions.
For every failed question choose exactly one strategy: REPAIR, REGENERATE, or SWITCH_TASK.
REPAIR: the assigned task/variation is valid and defects are localized and correctable.
REGENERATE: the assignment is valid but this generated question is fundamentally unsuitable for targeted repair.
SWITCH_TASK: the assignment/variation is contradictory, unsupported, response-incompatible, unusable, or cannot produce a distinct valid assessment.
The specialized Repair Router can repair only findings that map to executable field-owning
Question, Options, Solution, Answer, Task Alignment, or Difficulty steps. Do not choose
REPAIR when the supplied route feasibility says no executable route exists.
Cross-model duplication is not automatically content-repairable. For substantive overlap,
prefer SWITCH_TASK or an authorized alternative variation. Different numbers or wording alone
do not establish uniqueness. Choose REGENERATE only when the current valid assignment can
support a meaningfully different question under its authorized contract.
Use all validator and Duplicate V2 findings, not only the first finding. Never return PASS.
Give an ordered fallback_sequence containing only different remaining strategies. Always return
switch_preference=null; Python selects the concrete Task ID and Variation ID with the shared
question-type / global Task-ID usage / difficulty rank. Keep reasons concise.
Each question is isolated: never use another question's evidence or authorization as support."""


def _prompt(contexts: list[dict[str, Any]]) -> str:
    return (
        "Return one decision for every question_id in this batch. Python will enforce "
        "budgets and authorization; do not invent task or variation IDs.\n\n"
        + json.dumps({"failed_questions": contexts}, ensure_ascii=False, default=str,
                     separators=(",", ":"))
    )


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    decision = str(row["decision"]).upper()
    sequence: list[str] = []
    for value in row.get("fallback_sequence") or []:
        value = str(value).upper()
        if value != decision and value not in sequence:
            sequence.append(value)
    # A saved plan must remain useful without asking Task Alignment again.
    for value in ("REPAIR", "REGENERATE", "SWITCH_TASK"):
        if value != decision and value not in sequence:
            sequence.append(value)
    return {
        "version": "recovery-v3-task-alignment",
        "status": "ready",
        "decision": decision,
        "reason": str(row.get("reason") or ""),
        "fallback_sequence": sequence[:2],
        "plan": [decision, *sequence[:2]],
        "next_index": 0,
        "attempted_strategies": [],
        "switch_preference": row.get("switch_preference"),
        "decision_calls": 1,
    }


def decide_recovery(
    contexts: list[dict[str, Any]], *, domain: str,
    client_factory: Callable[[], object] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Decide unresolved contexts in batches of three with at most 50 calls."""
    if not contexts:
        return {}, {"calls": 0, "batch_sizes": [], "peak_concurrency": 0, "traces": []}
    size = config.RECOVERY_DECISION_BATCH_SIZE
    batches = [contexts[index:index + size] for index in range(0, len(contexts), size)]
    results: dict[str, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []
    concurrency_lock = threading.Lock()
    active_calls = 0
    observed_peak = 0

    def call(batch: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        nonlocal active_calls, observed_peak
        ids = [str(item["question_id"]) for item in batch]
        started = time.perf_counter()
        client = client_factory() if client_factory else create_review_client(
            domain=domain, stage="task_alignment_recovery_decision"
        )
        with concurrency_lock:
            active_calls += 1
            observed_peak = max(observed_peak, active_calls)
        try:
            raw = request_structured(
                client, _prompt(batch), response_model=RecoveryDecisionResponse,
                schema_name="task_alignment_recovery_decision_response",
                agent="task_alignment_validator", operation="recovery_decision",
                item_count=len(batch), expected_ids=ids, expected_id_field="question_id",
                system_prompt=SYSTEM_PROMPT, temperature=0.0,
                max_tokens=config.RECOVERY_DECISION_MAX_TOKENS,
                timeout=config.VALIDATOR_REQUEST_TIMEOUT_SECONDS,
            )
        finally:
            with concurrency_lock:
                active_calls -= 1
        payload = raw.model_dump() if hasattr(raw, "model_dump") else deepcopy(raw)
        rows = list((payload or {}).get("decisions") or [])
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            qid = str(row.get("question_id") or "")
            if qid in by_id or qid not in ids:
                raise ValueError(f"invalid or duplicate Recovery Decision id: {qid}")
            by_id[qid] = _normalize(row)
        missing = sorted(set(ids) - set(by_id))
        if missing:
            raise ValueError(f"Recovery Decision omitted question IDs: {missing}")
        return by_id, {
            "question_ids": ids,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "usage": deepcopy(getattr(client, "last_usage", None) or {}),
        }

    workers = min(config.MAX_PARALLEL_RECOVERY_DECISION_CALLS, len(batches))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(copy_context().run, call, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                values, trace = future.result()
                results.update(values)
                trace["status"] = "success"
                traces.append(trace)
            except Exception as exc:
                ids = [str(item.get("question_id") or "") for item in batch]
                traces.append({"question_ids": ids, "status": "blocked",
                               "error": f"{type(exc).__name__}: {exc}"})
                for qid in ids:
                    results[qid] = {
                        "version": "recovery-v3-task-alignment",
                        "status": "decision_blocked",
                        "error": f"{type(exc).__name__}: {exc}",
                        "decision_calls": 1,
                    }
    return results, {
        "calls": len(batches), "batch_sizes": [len(batch) for batch in batches],
        "configured_concurrency": config.MAX_PARALLEL_RECOVERY_DECISION_CALLS,
        "peak_concurrency": observed_peak, "traces": traces,
    }
