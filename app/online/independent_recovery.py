"""Independent per-model validation, duplicate, repair, and revalidation cycles.

The rest of the application still exposes one validator/repair function pair.
This module gives each model an isolated state partition, runs those existing
functions against that partition, and merges the completed partitions only
after each model reaches a local terminal state.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
import json
import threading
from typing import Any

from app.logging_conf import get_logger

logger = get_logger(__name__)

_MAX_LOCAL_ROUNDS = 12


def _model_number(value: Any, default: int = 1) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _target_model(target: dict[str, Any]) -> int:
    number = _model_number(target.get("model_number"), 0)
    if number:
        return number
    slot_id = str(target.get("slot_id") or "")
    if slot_id.startswith("m") and "_" in slot_id:
        try:
            return int(slot_id.split("_", 1)[0][1:])
        except ValueError:
            pass
    return 0


_RUNTIME_CALLBACK_KEYS = {
    "is_cancelled", "audit_callback", "progress_callback",
    "stage_callback", "cancel_event", "_recovery_task_coordinator",
}

_SLOT_MAP_KEYS = {
    "question_repair_attempts", "question_repair_calls",
    "question_repair_failures", "slot_completion_attempts",
    "repair_working_candidates", "repair_retry_seeds",
    "repair_replacement_memory",
}


def _copy_value(value: Any) -> Any:
    """Copy mutable workflow data while retaining runtime callables/locks."""
    if callable(value):
        return value
    if isinstance(value, dict):
        return {
            _copy_value(key): _copy_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    try:
        return deepcopy(value)
    except (TypeError, ValueError):
        # Provider clients/cancellation objects may own thread locks. They are
        # infrastructure, not model state, and must remain shared references.
        return value


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Copy the mutable workflow state without copying runtime infrastructure."""
    copied: dict[str, Any] = {}
    for key, value in state.items():
        copied[key] = value if key in _RUNTIME_CALLBACK_KEYS else _copy_value(value)
    return copied


def _isolate(state: dict[str, Any], model_number: int) -> dict[str, Any]:
    worker = _copy_state(state)
    exams = [
        deepcopy(exam) for exam in state.get("generated_exams") or []
        if _model_number(exam.get("model_number"), 0) == model_number
    ]
    plans = [
        deepcopy(plan) for plan in state.get("plans") or []
        if _model_number(plan.get("model_number"), 0) == model_number
    ]
    reports = [
        deepcopy(report) for report in state.get("validation_reports") or []
        if _model_number(report.get("model_number"), 0) == model_number
    ]
    manifest = [
        deepcopy(target) for target in state.get("repair_manifest") or []
        if _target_model(target) == model_number
    ]
    candidates = [
        deepcopy(candidate) for candidate in state.get("repair_candidates") or []
        if _target_model(candidate) == model_number
    ]
    validated = [
        value for value in state.get("validated_models") or []
        if _model_number(value, 0) == model_number
    ]
    worker.update({
        "generated_exams": exams,
        "plans": plans,
        "validation_reports": reports,
        "repair_manifest": manifest,
        "repair_candidates": candidates,
        "validated_models": validated,
        # Local validation remains partitioned, while replacement eligibility
        # and reuse context must see the authoritative whole-exam snapshot.
        "global_plans": deepcopy(state.get("plans") or []),
        "global_generated_exams": deepcopy(state.get("generated_exams") or []),
        "validation_ledger": {
            str(model_number): deepcopy(
                (state.get("validation_ledger") or {}).get(str(model_number)) or {}
            )
        },
    })
    return worker


def _dedupe_rows(rows: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _merge_list(key: str, current: list[Any], incoming: list[Any]) -> list[Any]:
    return _dedupe_rows([*current, *incoming])


def _merge_eval_stats(
    current: dict[str, Any], incoming: dict[str, Any], model_number: int,
) -> dict[str, Any]:
    """Merge only one worker's authoritative model counters and runtime IDs."""
    from app.online.eval_stats import aggregate

    merged = deepcopy(current or incoming)
    model_key = str(model_number)
    incoming_model = (incoming.get("models") or {}).get(model_key)
    if incoming_model is not None:
        merged.setdefault("models", {})[model_key] = deepcopy(incoming_model)
    runtime = merged.setdefault("_runtime", {})
    incoming_runtime = incoming.get("_runtime") or {}
    runtime["first_validation_models"] = sorted(set(
        runtime.get("first_validation_models") or []
    ) | set(incoming_runtime.get("first_validation_models") or []))
    repairs = dict(runtime.get("repair_questions") or {})
    for qid, value in (incoming_runtime.get("repair_questions") or {}).items():
        if int((value or {}).get("model_number") or 0) == model_number:
            repairs[str(qid)] = deepcopy(value)
    runtime["repair_questions"] = repairs
    initial = dict(runtime.get("initial_question_counts") or {})
    incoming_initial = incoming_runtime.get("initial_question_counts") or {}
    if model_key in incoming_initial:
        initial[model_key] = deepcopy(incoming_initial[model_key])
    runtime["initial_question_counts"] = initial
    aggregate(merged)
    return merged


def _merge_worker(
    state: dict[str, Any], worker: dict[str, Any], model_number: int,
) -> dict[str, Any]:
    """Replace only this model's rows; append shared event history."""
    merged = state
    if isinstance(worker.get("eval_stats"), dict):
        merged["eval_stats"] = _merge_eval_stats(
            dict(merged.get("eval_stats") or {}), worker["eval_stats"], model_number,
        )
    for key in ("generated_exams", "plans", "validation_reports"):
        current = [
            row for row in merged.get(key) or []
            if _model_number(row.get("model_number"), 0) != model_number
        ]
        incoming = [
            row for row in worker.get(key) or []
            if _model_number(row.get("model_number"), 0) == model_number
        ]
        merged[key] = [*current, *incoming]
    for key in ("repair_manifest", "repair_candidates"):
        current = [
            row for row in merged.get(key) or []
            if _target_model(row) != model_number
        ]
        incoming = [
            row for row in worker.get(key) or []
            if _target_model(row) == model_number
        ]
        merged[key] = [*current, *incoming]
    for key in (
        "validation_repair_history", "repair_call_logs", "warnings",
        "duplicate_reports", "duplicate_findings", "duplicate_repair_history",
        "invalid_questions", "performance_telemetry",
    ):
        if key in worker:
            merged[key] = _merge_list(key, list(merged.get(key) or []), list(worker.get(key) or []))
    if isinstance(worker.get("slot_lifecycle"), dict):
        lifecycle = dict(merged.get("slot_lifecycle") or {})
        prefix = f"m{model_number}_"
        for slot_id, value in worker["slot_lifecycle"].items():
            if str(slot_id).startswith(prefix):
                lifecycle[slot_id] = deepcopy(value)
        merged["slot_lifecycle"] = lifecycle
    prefix = f"m{model_number}_"
    for key in _SLOT_MAP_KEYS:
        if not isinstance(worker.get(key), dict):
            continue
        current = dict(merged.get(key) or {})
        for slot_id, value in worker[key].items():
            if str(slot_id).startswith(prefix):
                current[slot_id] = deepcopy(value)
        merged[key] = current
    if isinstance(worker.get("pending_revalidation_ids"), dict):
        pending = dict(merged.get("pending_revalidation_ids") or {})
        for lookup in (model_number, str(model_number)):
            if lookup in worker["pending_revalidation_ids"]:
                pending[lookup] = deepcopy(worker["pending_revalidation_ids"][lookup])
        merged["pending_revalidation_ids"] = pending
    for key, value in worker.items():
        if key in {
            "generated_exams", "plans", "validation_reports", "repair_manifest",
            "repair_candidates", "validation_repair_history", "repair_call_logs",
            "warnings", "duplicate_reports", "duplicate_findings",
            "duplicate_repair_history", "invalid_questions", "performance_telemetry",
            "slot_lifecycle", "pending_revalidation_ids", *_SLOT_MAP_KEYS,
            "eval_stats",
        }:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        elif isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = _merge_list(key, merged[key], value)
        else:
            merged[key] = value
    validated = set(merged.get("validated_models") or [])
    if model_number in set(worker.get("validated_models") or []):
        validated.add(model_number)
    merged["validated_models"] = sorted(validated)
    status = dict(merged.get("model_lifecycle_status") or {})
    status[model_number] = {
        "status": (
            "locally_ready" if model_number in set(worker.get("validated_models") or [])
            else "local_terminal"
        ),
        "rounds": int((worker.get("_local_recovery_rounds") or 0)),
    }
    merged["model_lifecycle_status"] = status
    return merged


def _has_pending(worker: dict[str, Any], model_number: int) -> bool:
    for target in worker.get("repair_manifest") or []:
        if _target_model(target) != model_number:
            continue
        if str(target.get("status") or "") in {"resolved", "terminal_failure", "exhausted", "failed_to_repair"}:
            continue
        return True
    return False


def _emit_progress(
    state: dict[str, Any], model_number: int, phase: str,
    progress: int, detail: str = "",
) -> None:
    """Publish real local-lifecycle progress without adding provider calls."""
    payload = {
        "model_number": model_number,
        "phase": phase,
        "detail": detail,
        "progress": progress,
    }
    callback = state.get("progress_callback")
    if callable(callback):
        try:
            callback(
                "validating_exam", progress,
                job_id=str(state.get("job_id") or ""),
                stage="validating_exam", component="independent_model_recovery",
                detail=detail, model_number=model_number,
            )
        except Exception:
            logger.debug("independent recovery progress callback failed", exc_info=True)
    audit = state.get("audit_callback")
    if callable(audit):
        try:
            audit("validating_exam", "model_recovery_progress", payload)
        except Exception:
            logger.debug("independent recovery audit callback failed", exc_info=True)


def _recover_model(state: dict[str, Any], model_number: int) -> tuple[dict[str, Any], int]:
    from app.online.validator import repair_invalid_questions, validate_generated_questions
    from app.logging_conf import set_llm_attribution

    set_llm_attribution(model_number=model_number, stage="validation")
    worker = _isolate(state, model_number)
    _emit_progress(worker, model_number, "validating", 76, "Model validation started")
    for round_index in range(1, _MAX_LOCAL_ROUNDS + 1):
        worker["_local_recovery_rounds"] = round_index
        validation = validate_generated_questions(worker)
        worker.update(validation)
        _emit_progress(worker, model_number, "duplicate_check", 82, "Within-model Duplicate V2 completed")
        if _has_pending(worker, model_number):
            _emit_progress(worker, model_number, "repairing", 86, "Targeted repair or backup recovery")
            worker.update(repair_invalid_questions(worker))
            _emit_progress(worker, model_number, "revalidating", 90, "Candidate revalidation")
            continue
        if model_number in set(worker.get("validated_models") or []):
            _emit_progress(worker, model_number, "locally_ready", 95, "Model locally ready")
            return worker, round_index
        _emit_progress(worker, model_number, "local_terminal", 95, "Model reached local terminal state")
        break
    return worker, int(worker.get("_local_recovery_rounds") or 0)


def _targeted_cross_model_recovery(
    state: dict[str, Any], manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute each target's saved bounded V3 plan without a new decision call."""
    from app.online.validator import repair_invalid_questions, validate_generated_questions
    from app.online.cross_model_duplicates import integrate_cross_model_duplicates

    merged = state
    active = deepcopy(manifest)
    # Three semantic strategies exist. Existing per-slot budgets can stop the
    # loop earlier; this cap cannot increase any Recovery budget.
    for _plan_step in range(3):
        affected = sorted({_target_model(value) for value in active if _target_model(value)})
        if not affected:
            break
        previous_by_slot = {
            (_target_model(value), str(value.get("slot_id") or "")): deepcopy(value)
            for value in active
        }

        def recover_model(
            snapshot: dict[str, Any], model_number: int,
        ) -> tuple[int, dict[str, Any]]:
            worker = _isolate(snapshot, model_number)
            worker["repair_manifest"] = [
                deepcopy(target) for target in active
                if _target_model(target) == model_number
            ]
            worker.update(repair_invalid_questions(worker))
            worker.update(validate_generated_questions(worker))
            return model_number, worker

        # Cross-model conflicts in different models are independent.  Run their
        # bounded repair/revalidation cycles concurrently; the subsequent
        # duplicate scan and merge stay deterministic and serial.
        if len(affected) > 1:
            completed: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=min(len(affected), 4)) as executor:
                futures = [
                    executor.submit(
                        copy_context().run, recover_model, _copy_state(merged), model_number,
                    )
                    for model_number in affected
                ]
                for future in as_completed(futures):
                    model_number, worker = future.result()
                    completed[model_number] = worker
            for model_number in affected:
                merged = _merge_worker(merged, completed[model_number], model_number)
        else:
            model_number = affected[0]
            _resolved_model, worker = recover_model(_copy_state(merged), model_number)
            merged = _merge_worker(merged, worker, model_number)

        reports = {
            _model_number(report.get("model_number"), 0): report
            for report in merged.get("validation_reports") or []
        }
        next_manifest, validated, warnings, cross = integrate_cross_model_duplicates(
            merged, list(merged.get("generated_exams") or []), reports,
            list(merged.get("repair_manifest") or []),
            list(merged.get("validated_models") or []), list(merged.get("warnings") or []),
        )
        merged["repair_manifest"] = next_manifest
        merged["validated_models"] = validated
        merged["warnings"] = warnings
        merged.update(cross)
        active = [
            value for value in next_manifest
            if str(value.get("failure_code") or value.get("original_failure_code") or "")
            == "cross_model_duplicate"
            and str(value.get("status") or "") not in {
                "resolved", "terminal_failure", "exhausted", "failed_to_repair",
            }
        ]
        for target in active:
            previous = previous_by_slot.get(
                (_target_model(target), str(target.get("slot_id") or "")), {}
            )
            if previous.get("recovery_v3") and not target.get("recovery_v3"):
                target["recovery_v3"] = deepcopy(previous["recovery_v3"])
                target["recovery_decision_audit"] = deepcopy(
                    previous.get("recovery_decision_audit") or {}
                )
            target["original_failure_code"] = "cross_model_duplicate"
    return merged


def _question_has_fresh_nonduplicate_pass(
    state: dict[str, Any], model_number: int, question_id: str,
) -> bool:
    report = next((
        value for value in state.get("validation_reports") or []
        if _model_number(value.get("model_number"), 0) == model_number
    ), {})
    verdict = next((
        value for value in report.get("verdicts") or []
        if str(value.get("question_id") or "") == question_id
    ), None)
    return bool(
        verdict and verdict.get("action") == "PASS"
        and str(verdict.get("evaluation_origin") or "fresh") != "historical_retained"
    )


def _restore_model_partition(
    state: dict[str, Any], snapshot: dict[str, Any], model_number: int,
) -> dict[str, Any]:
    """Restore the last fully validated model after invalid cross recovery."""
    restored = state
    for key in ("generated_exams", "plans", "validation_reports"):
        current = [
            row for row in restored.get(key) or []
            if _model_number(row.get("model_number"), 0) != model_number
        ]
        previous = [
            deepcopy(row) for row in snapshot.get(key) or []
            if _model_number(row.get("model_number"), 0) == model_number
        ]
        restored[key] = [*current, *previous]
    return restored


def _apply_cross_model_nonblocking_fallback(
    state: dict[str, Any], before_recovery: dict[str, Any],
) -> dict[str, Any]:
    """Keep/restore fully valid questions when only cross-model overlap remains."""
    targets = [
        value for value in state.get("repair_manifest") or []
        if str(value.get("failure_code") or "") == "cross_model_duplicate"
        or str(value.get("original_failure_code") or "") == "cross_model_duplicate"
    ]
    if not targets:
        return state
    audit = list(state.get("cross_model_duplicate_fallbacks") or [])
    for target in targets:
        model = _target_model(target)
        question_id = str(target.get("question_id") or "")
        kept_latest = _question_has_fresh_nonduplicate_pass(state, model, question_id)
        if not kept_latest:
            # A failed/ungrounded recovery candidate can never be accepted by
            # this policy. Restore the most recent model partition that had
            # already passed all non-duplicate validators.
            state = _restore_model_partition(state, before_recovery, model)
        audit.append({
            "model_number": model, "slot_id": target.get("slot_id"),
            "question_id": question_id,
            "resolution": "kept_latest_valid" if kept_latest else "restored_previous_valid",
            "reason": "cross_model_duplicate_nonblocking_fallback",
        })
        lifecycle = (state.get("slot_lifecycle") or {}).get(
            str(target.get("slot_id") or "")
        )
        if isinstance(lifecycle, dict):
            lifecycle["cross_model_duplicate_status"] = "ACCEPTED_NONBLOCKING"
            lifecycle["final_acceptance_status"] = "ACCEPTED"
    state["repair_manifest"] = [
        value for value in state.get("repair_manifest") or []
        if str(value.get("failure_code") or "") != "cross_model_duplicate"
        and str(value.get("original_failure_code") or "") != "cross_model_duplicate"
    ]
    state["validated_models"] = list(range(1, int(state.get("num_models") or 1) + 1))
    state["warnings"] = [
        value for value in state.get("warnings") or []
        if not (isinstance(value, dict) and value.get("type") == "cross_model_duplicates")
    ]
    findings = deepcopy(state.get("cross_model_duplicate_findings") or [])
    for finding in findings:
        finding["recovery_status"] = "accepted_nonblocking_cross_model_fallback"
    state["cross_model_duplicate_findings"] = findings
    state["cross_model_duplicate_fallbacks"] = audit
    state["cross_model_duplicates_user_visible"] = False
    return state


def run_independent_model_recovery(state: dict[str, Any]) -> dict[str, Any]:
    """Run all models through isolated local lifecycles, concurrently."""
    num_models = max(1, int(state.get("num_models") or 1))
    if num_models <= 1:
        return state
    from app.online.cross_model_duplicates import integrate_cross_model_duplicates

    merged = _copy_state(state)
    # The object is deliberately shared by isolated workers. It coordinates
    # task reservations only; model content and lifecycle data remain isolated.
    merged.setdefault("_recovery_task_coordinator", {
        "lock": threading.RLock(), "reservations": {},
    })
    state["_recovery_task_coordinator"] = merged["_recovery_task_coordinator"]
    previous_status = dict(state.get("model_lifecycle_status") or {})
    workers: dict[int, dict[str, Any]] = {}
    rounds: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=min(num_models, 10)) as executor:
        futures = {
            executor.submit(
                copy_context().run, _recover_model, _copy_state(state), model
            ): model
            for model in range(1, num_models + 1)
            if str((previous_status.get(model) or {}).get("status") or "") != "locally_ready"
        }
        for future in as_completed(futures):
            model = futures[future]
            worker, count = future.result()
            workers[model] = worker
            rounds[model] = count
        for model in range(1, num_models + 1):
            if str((previous_status.get(model) or {}).get("status") or "") == "locally_ready":
                rounds[model] = int((previous_status.get(model) or {}).get("rounds") or 0)
    for model in range(1, num_models + 1):
        if model not in workers:
            continue
        merged = _merge_worker(merged, workers[model], model)
    merged["_independent_model_rounds"] = rounds
    _emit_progress(merged, 0, "cross_model_duplicate", 96, "Cross-model Duplicate V2 started")

    # Cross-model comparison is deliberately after all local model states are
    # merged. It never runs inside a per-model worker.
    reports = {
        _model_number(report.get("model_number"), 0): report
        for report in merged.get("validation_reports") or []
    }
    manifest, validated, warnings, cross = integrate_cross_model_duplicates(
        merged, list(merged.get("generated_exams") or []), reports,
        list(merged.get("repair_manifest") or []),
        list(merged.get("validated_models") or []), list(merged.get("warnings") or []),
    )
    merged["repair_manifest"] = manifest
    merged["validated_models"] = validated
    merged["warnings"] = warnings
    merged.update(cross)
    merged["cross_model_duplicate_scan_after_local_completion"] = True

    cross_targets = [
        target for target in manifest
        if str(target.get("failure_code") or "") == "cross_model_duplicate"
        and str(target.get("status") or "") not in {"resolved", "terminal_failure"}
    ]
    if cross_targets:
        before_cross_recovery = _copy_state(merged)
        merged = _targeted_cross_model_recovery(merged, cross_targets)
        reports = {
            _model_number(report.get("model_number"), 0): report
            for report in merged.get("validation_reports") or []
        }
        manifest, validated, warnings, cross = integrate_cross_model_duplicates(
            merged, list(merged.get("generated_exams") or []), reports,
            list(merged.get("repair_manifest") or []),
            list(merged.get("validated_models") or []), list(merged.get("warnings") or []),
        )
        merged["repair_manifest"] = manifest
        merged["validated_models"] = validated
        merged["warnings"] = warnings
        merged.update(cross)
        merged = _apply_cross_model_nonblocking_fallback(
            merged, before_cross_recovery,
        )
    logger.info(
        "Independent model recovery complete | models=%d | rounds=%s | validated=%s",
        num_models, rounds, merged.get("validated_models"),
    )
    _emit_progress(merged, 0, "cross_model_complete", 98, "Cross-model duplicate check complete")
    # Coordination/snapshot values are runtime-only. Never send locks or a
    # duplicate whole-exam snapshot into checkpoint serialization.
    merged.pop("_recovery_task_coordinator", None)
    merged.pop("global_plans", None)
    merged.pop("global_generated_exams", None)
    state.pop("_recovery_task_coordinator", None)
    return merged
