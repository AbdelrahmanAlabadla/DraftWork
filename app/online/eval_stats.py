"""JSON-safe telemetry helpers for the existing exam-generation pipeline.

This module only records decisions already made by generation, validation, and
repair.  It deliberately contains no model calls and no pipeline control flow.
Per-question-type counters are the source of truth; model and overall totals are
recomputed from those counters before the eval object is returned or persisted.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


PipelineEvalStats = dict[str, Any]


COUNT_FIELDS = (
    "requested_questions",
    "generated_first",
    "missing_first",
    "generation_rejected",
    "shortfall_generated",
    "shortfall_still_missing",
    "validation_total_first",
    "validation_passed_first",
    "validation_failed_first",
    "validation_unvalidated_first",
    "repair_sent",
    "repair_succeeded",
    "repair_failed",
    "final_valid",
    "final_invalid",
    "final_unvalidated",
    "final_missing",
    "final_missing_or_invalid",
)
REASON_FIELDS = (
    "generation_rejection_reasons",
    "validation_failure_reasons",
    "validation_failure_actions",
    "validator_failure_reasons",
)


def _empty_counts(requested: int = 0) -> dict[str, Any]:
    values: dict[str, Any] = {field: 0 for field in COUNT_FIELDS}
    values["requested_questions"] = requested
    for field in REASON_FIELDS:
        values[field] = {}
    return values


def create_pipeline_eval(
    tasks: Iterable[tuple[str, int]], num_models: int
) -> dict[str, Any]:
    """Create the JSON-safe eval state, including runtime-only ID bookkeeping."""
    requested = {str(qtype): int(count) for qtype, count in tasks}
    models: dict[str, Any] = {}
    for model_number in range(1, num_models + 1):
        question_types = {
            qtype: _empty_counts(count) for qtype, count in requested.items()
        }
        models[str(model_number)] = {
            **_empty_counts(sum(requested.values())),
            "question_types": question_types,
            "slot_outcomes": {
                f"m{model_number}_{qtype}_{index}": {
                    "question_type": qtype,
                    "attempts": 0,
                    "status": "pending",
                }
                for qtype, count in requested.items()
                for index in range(1, count + 1)
            },
        }
    stats = {
        "overall": {**_empty_counts(), "question_types": {}},
        "models": models,
        # Lists/dicts keep LangGraph state and any future checkpoints JSON-safe.
        "_runtime": {
            "first_validation_models": [],
            "repair_questions": {},
        },
    }
    aggregate(stats)
    return stats


def _type_counts(
    stats: dict[str, Any], model_number: int, qtype: str
) -> dict[str, Any]:
    model = stats["models"][str(model_number)]
    return model["question_types"].setdefault(qtype, _empty_counts())


def section_items(section: Any) -> list[dict[str, Any]]:
    if isinstance(section, dict):
        return [item for item in (section.get("items") or []) if isinstance(item, dict)]
    if isinstance(section, list):
        return [item for item in section if isinstance(item, dict)]
    return []


def record_initial_generation(
    stats: dict[str, Any], model_number: int, questions: dict[str, Any]
) -> None:
    """Snapshot accepted questions before count/shortfall repair begins."""
    counts = {
        qtype: len(section_items(questions.get(qtype)))
        for qtype in stats["models"][str(model_number)]["question_types"]
    }
    record_initial_generation_counts(stats, model_number, counts)


def record_initial_generation_counts(
    stats: dict[str, Any], model_number: int, counts: dict[str, int]
) -> None:
    """Record an immutable per-type count snapshot taken by the generation node."""
    model = stats["models"][str(model_number)]
    for qtype, counters in model["question_types"].items():
        requested = counters["requested_questions"]
        generated = max(int(counts.get(qtype, 0)), 0)
        counters["generated_first"] = min(generated, requested)
        counters["missing_first"] = max(requested - generated, 0)
    stats["_runtime"].setdefault("initial_question_counts", {})[
        str(model_number)
    ] = {str(qtype): max(int(count), 0) for qtype, count in counts.items()}
    aggregate(stats)


def record_generation_rejection(
    stats: dict[str, Any] | None,
    model_number: int,
    qtype: str,
    reason: str,
    count: int = 1,
    slot_id: str | None = None,
) -> None:
    """Record raw generation candidates rejected by existing filters."""
    if stats is None or count <= 0:
        return
    counters = _type_counts(stats, model_number, qtype)
    counters["generation_rejected"] += count
    reasons = counters["generation_rejection_reasons"]
    reasons[reason] = reasons.get(reason, 0) + count
    if slot_id:
        outcomes = stats["models"][str(model_number)].setdefault("slot_outcomes", {})
        entry = outcomes.setdefault(
            slot_id,
            {"question_type": qtype, "attempts": 0, "status": "pending"},
        )
        entry["attempts"] += count
        entry["status"] = "rejected"
        entry["last_rejection_reason"] = reason


def record_slot_accepted(
    stats: dict[str, Any] | None,
    model_number: int,
    qtype: str,
    slot_id: str,
) -> None:
    if stats is None or not slot_id:
        return
    outcomes = stats["models"][str(model_number)].setdefault("slot_outcomes", {})
    entry = outcomes.setdefault(
        slot_id,
        {"question_type": qtype, "attempts": 0, "status": "pending"},
    )
    entry["attempts"] += 1
    entry["status"] = "accepted"
    entry.pop("last_rejection_reason", None)


def record_shortfall_result(
    stats: dict[str, Any],
    model_number: int,
    questions: dict[str, Any],
    appended_by_type: dict[str, int] | None = None,
) -> None:
    """Record actual shortfall appends plus any deficit that remains."""
    model = stats["models"][str(model_number)]
    appended_by_type = appended_by_type or {}
    for qtype, counters in model["question_types"].items():
        requested = counters["requested_questions"]
        final_count = min(len(section_items(questions.get(qtype))), requested)
        counters["shortfall_generated"] = max(int(appended_by_type.get(qtype, 0)), 0)
        counters["shortfall_still_missing"] = max(requested - final_count, 0)
    aggregate(stats)


# Repair-action categories are retained as a stable compatibility taxonomy for
# older runs. Current Validator verdicts also carry a diagnostic ``failure_code``
# (produced by the deterministic, general, quantitative, grounding and duplicate
# roles), which is the more precise rejection reason and is recorded when present.
def validation_failure_category(verdict: dict[str, Any]) -> str:
    return {
        "FIX_ANSWER": "wrong_answer",
        "FIX_OPTIONS": "bad_options",
        "FIX_QUESTION": "unclear_or_invalid_question",
        "FIX_QUESTION_AND_ANSWER": "question_and_answer",
    }.get(str(verdict.get("action") or "").upper(), "other")


def validation_rejection_reason(verdict: dict[str, Any]) -> str:
    """Return the diagnostic rejection code, falling back to the repair action."""
    code = str(verdict.get("failure_code") or "").strip()
    if code and code.lower() != "pass":
        return code
    return validation_failure_category(verdict)


def validator_operational_reason(verdict: dict[str, Any]) -> str:
    """Classify a question whose validation could not complete.

    Mirrors the operational states emitted by ``validate_exam_multi``:
    ``INPUT_INVALID``, ``VALIDATION_BLOCKED`` (operational or prerequisite), and
    malformed/missing provider output. These are validator outages, not content
    rejections, so they are counted separately from failure reasons.
    """
    status = str(verdict.get("validation_status") or "").upper()
    if status == "INPUT_INVALID":
        return "input_invalid"
    if status == "ERROR":
        return "validator_error"
    if status == "VALIDATION_BLOCKED":
        category = str(verdict.get("block_category") or "").strip().lower()
        if category in {"prerequisite", "operational"}:
            return f"{category}_validation_block"
        return "validation_blocked"
    return "missing_verdict"


def _question_type_by_id(exam: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for qtype, section in (exam.get("questions") or {}).items():
        for item in section_items(section):
            qid = str(item.get("question_id") or "")
            if qid:
                result[qid] = qtype
    return result


def record_first_validation(
    stats: dict[str, Any], exam: dict[str, Any], report: dict[str, Any]
) -> None:
    """Record a model's first validator pass exactly once."""
    model_number = int(report.get("model_number") or exam.get("model_number") or 0)
    runtime = stats["_runtime"]
    captured = runtime["first_validation_models"]
    if model_number in captured:
        return
    captured.append(model_number)
    type_by_id = _question_type_by_id(exam)
    for verdict in report.get("verdicts") or []:
        qtype = type_by_id.get(str(verdict.get("question_id") or ""))
        if not qtype:
            continue
        counters = _type_counts(stats, model_number, qtype)
        counters["validation_total_first"] += 1
        action = str(verdict.get("action") or "").upper()
        if action == "PASS":
            counters["validation_passed_first"] += 1
        elif action == "UNVALIDATED":
            counters["validation_unvalidated_first"] += 1
            reasons = counters["validator_failure_reasons"]
            reason = validator_operational_reason(verdict)
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            counters["validation_failed_first"] += 1
            reasons = counters["validation_failure_reasons"]
            reason = validation_rejection_reason(verdict)
            reasons[reason] = reasons.get(reason, 0) + 1
            actions = counters["validation_failure_actions"]
            action_key = action.lower() if action else "other"
            actions[action_key] = actions.get(action_key, 0) + 1
    aggregate(stats)


def record_repair_outcome(
    stats: dict[str, Any],
    exam: dict[str, Any],
    report: dict[str, Any],
    repaired_ids: Iterable[str],
    failed_ids: Iterable[str],
    sent_ids: Iterable[str] | None = None,
) -> None:
    """Preserve unique submitted repair IDs and whether write-back ever applied.

    Final success requires both an applied write-back and a later PASS verdict.
    """
    model_number = int(report.get("model_number") or exam.get("model_number") or 0)
    type_by_id = _question_type_by_id(exam)
    repaired = {str(qid) for qid in repaired_ids if str(qid)}
    failed = {str(qid) for qid in failed_ids if str(qid)}
    sent_source = [*repaired, *failed] if sent_ids is None else sent_ids
    sent = {str(qid) for qid in sent_source if str(qid)}
    repair_questions = stats["_runtime"]["repair_questions"]
    for verdict in report.get("verdicts") or []:
        qid = str(verdict.get("question_id") or "")
        qtype = type_by_id.get(qid)
        if qid not in sent or not qtype:
            continue
        entry = repair_questions.setdefault(
            qid, {"model_number": model_number, "question_type": qtype}
        )
        if qid in repaired:
            entry["writeback_applied"] = True
        elif qid in failed:
            entry.setdefault("writeback_applied", False)


def finalize_pipeline_eval(
    stats: dict[str, Any],
    exams: list[dict[str, Any]],
    latest_reports: list[dict[str, Any]],
) -> None:
    """Calculate final validity and eventual repair outcomes from latest reports."""
    reports = {str(r.get("model_number")): r for r in latest_reports}
    exams_by_model = {str(e.get("model_number")): e for e in exams}
    latest_action: dict[str, str] = {}
    for report in latest_reports:
        for verdict in report.get("verdicts") or []:
            qid = str(verdict.get("question_id") or "")
            if qid:
                latest_action[qid] = str(verdict.get("action") or "")

    for model_key, model in stats["models"].items():
        exam = exams_by_model.get(model_key) or {}
        report = reports.get(model_key) or {}
        action_by_id = {
            str(v.get("question_id") or ""): str(v.get("action") or "")
            for v in report.get("verdicts") or []
            if v.get("question_id")
        }
        questions = exam.get("questions") or {}
        for qtype, counters in model["question_types"].items():
            requested = counters["requested_questions"]
            items = section_items(questions.get(qtype))[:requested]
            actions = [
                action_by_id.get(str(item.get("question_id") or ""), "UNVALIDATED")
                for item in items
            ]
            counters["final_valid"] = sum(action == "PASS" for action in actions)
            counters["final_unvalidated"] = sum(
                action == "UNVALIDATED" for action in actions
            )
            counters["final_invalid"] = sum(
                action not in {"PASS", "UNVALIDATED"} for action in actions
            )
            counters["final_missing"] = max(requested - len(items), 0)
            counters["final_missing_or_invalid"] = (
                counters["final_invalid"]
                + counters["final_unvalidated"]
                + counters["final_missing"]
            )
            counters["repair_sent"] = 0
            counters["repair_succeeded"] = 0
            counters["repair_failed"] = 0

    for qid, info in stats["_runtime"]["repair_questions"].items():
        counters = _type_counts(
            stats, int(info["model_number"]), str(info["question_type"])
        )
        counters["repair_sent"] += 1
        if info.get("writeback_applied") and latest_action.get(qid) == "PASS":
            counters["repair_succeeded"] += 1
        else:
            counters["repair_failed"] += 1
    aggregate(stats)


def aggregate(stats: dict[str, Any]) -> None:
    """Recompute model totals and overall totals from per-type counters."""
    overall_types: dict[str, Any] = {}
    for model in stats.get("models", {}).values():
        for field in COUNT_FIELDS:
            model[field] = sum(
                int(counters.get(field, 0))
                for counters in model.get("question_types", {}).values()
            )
        for field in REASON_FIELDS:
            combined: dict[str, int] = {}
            for counters in model.get("question_types", {}).values():
                for reason, count in counters.get(field, {}).items():
                    combined[reason] = combined.get(reason, 0) + int(count)
            model[field] = combined

        for qtype, counters in model.get("question_types", {}).items():
            target = overall_types.setdefault(qtype, _empty_counts())
            for field in COUNT_FIELDS:
                target[field] += int(counters.get(field, 0))
            for field in REASON_FIELDS:
                for reason, count in counters.get(field, {}).items():
                    target[field][reason] = target[field].get(reason, 0) + int(count)

    overall = stats.setdefault("overall", {})
    overall.clear()
    overall.update(_empty_counts())
    overall["question_types"] = overall_types
    for field in COUNT_FIELDS:
        overall[field] = sum(int(model.get(field, 0)) for model in stats["models"].values())
    for field in REASON_FIELDS:
        combined: dict[str, int] = {}
        for model in stats["models"].values():
            for reason, count in model.get(field, {}).items():
                combined[reason] = combined.get(reason, 0) + int(count)
        overall[field] = combined


def public_eval(stats: dict[str, Any] | None) -> dict[str, Any]:
    """Return a detached JSON-safe eval object without runtime ID bookkeeping."""
    if not stats:
        return {"overall": {**_empty_counts(), "question_types": {}}, "models": {}}
    aggregate(stats)
    return {
        "overall": deepcopy(stats["overall"]),
        "models": deepcopy(stats["models"]),
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a rounded ratio, or None when the metric has no denominator."""
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 4)


def derived_rates(bucket: dict[str, Any]) -> dict[str, float | None]:
    generated_first = int(bucket.get("generated_first", 0))
    rejected = int(bucket.get("generation_rejected", 0))
    return {
        "generation_completion_rate": safe_rate(
            generated_first,
            int(bucket.get("requested_questions", 0)),
        ),
        "generation_rejection_rate": safe_rate(
            rejected,
            generated_first + rejected,
        ),
        "shortfall_recovery_rate": safe_rate(
            int(bucket.get("shortfall_generated", 0)),
            int(bucket.get("missing_first", 0)),
        ),
        "first_validation_pass_rate": safe_rate(
            int(bucket.get("validation_passed_first", 0)),
            int(bucket.get("validation_total_first", 0)),
        ),
        "validation_coverage_rate": safe_rate(
            int(bucket.get("validation_total_first", 0)),
            generated_first,
        ),
        "repair_success_rate": safe_rate(
            int(bucket.get("repair_succeeded", 0)),
            int(bucket.get("repair_sent", 0)),
        ),
        "final_success_rate": safe_rate(
            int(bucket.get("final_valid", 0)),
            int(bucket.get("requested_questions", 0)),
        ),
    }


def add_derived_rates(bucket: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(bucket)
    result["rates"] = derived_rates(result)
    for qtype, type_bucket in result.get("question_types", {}).items():
        result["question_types"][qtype] = add_derived_rates(type_bucket)
    return result


# ---------------------------------------------------------------------------
# Whole-pipeline outcome telemetry
#
# The counters above describe question acceptance.  The functions below derive
# additional outcome summaries from records the pipeline already produces: the
# authoritative per-slot lifecycle, the repair/recovery attempt history, the
# duplicate-repair bookkeeping, and the assessment-catalog cache flags.  They
# only count recorded decisions; nothing is inferred or estimated.
# ---------------------------------------------------------------------------
RECOVERY_ATTEMPT_SUCCESS = frozenset({"accepted", "repaired"})
_ATTEMPT_OUTCOME_PRIORITY = {
    "accepted": 6,
    "repaired": 6,
    "rejected": 5,
    "candidate_generation_failed": 4,
    "strategy_skipped": 3,
    "validation_pending": 2,
    "invalid": 1,
}
# Canonical Recovery V3 strategy identifiers, mapped from the descriptive
# strategy recorded in validation repair history.  Unknown identifiers are kept
# verbatim so no production value is hidden by the dashboard.
RECOVERY_STRATEGIES = ("REPAIR", "REGENERATE", "SWITCH_VARIATION", "SWITCH_TASK")
_STRATEGY_ALIASES = {
    "recovery_v3_targeted_repair": "REPAIR",
    "recovery_v3_content_repair": "REPAIR",
    "targeted_repair": "REPAIR",
    "repair_question": "REPAIR",
    "recovery_v3_regenerate_assignment": "REGENERATE",
    "regenerate_missing_slot": "REGENERATE",
    "recovery_v3_switch_variation": "SWITCH_VARIATION",
    "switch_variation": "SWITCH_VARIATION",
    "recovery_v3_switch_assignment": "SWITCH_TASK",
    "recovery_v3_switch_task": "SWITCH_TASK",
    "replace_task": "SWITCH_TASK",
    "switch_task": "SWITCH_TASK",
    # Historical strategy descriptors recorded before the Recovery V3 naming
    # was standardized.  They map to the same dispatch strategy.
    "targeted_followup_repair": "REPAIR",
    "reconciled_targeted_repair": "REPAIR",
    "accepted_source_constraint": "REPAIR",
    "targeted_difficulty_adjustment": "REPAIR",
    "catalog_replacement": "SWITCH_TASK",
    "catalog_replacement_after_difficulty": "SWITCH_TASK",
    "on_demand_completion_replacement": "SWITCH_TASK",
    "regenerate_original_assignment": "REGENERATE",
    "regenerate_authorized_alternative_variation": "SWITCH_VARIATION",
}


def recovery_strategy_name(value: Any) -> str:
    """Map a recorded repair strategy to its canonical Recovery V3 identifier."""
    raw = str(value or "").strip()
    if not raw:
        return "UNKNOWN"
    mapped = _STRATEGY_ALIASES.get(raw.lower())
    return mapped or raw


def _bump(mapping: dict[str, int], key: Any, count: int = 1) -> None:
    name = str(key or "unknown")
    mapping[name] = mapping.get(name, 0) + int(count)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def summarize_repair_attempts(history: Any) -> dict[str, Any]:
    """Collapse repair history events into one outcome per slot attempt.

    A single attempt can append several events (candidate generated, then
    rejected/accepted).  Counting events would overstate attempts, so events are
    grouped by ``(slot, attempt)`` and the strongest recorded outcome wins.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in history or []:
        if not isinstance(event, dict):
            continue
        slot_id = str(event.get("slot_id") or event.get("question_id") or "")
        if not slot_id:
            continue
        try:
            attempt = int(event.get("attempt") or 1)
        except (TypeError, ValueError):
            attempt = 1
        grouped.setdefault((slot_id, attempt), []).append(event)

    outcomes: dict[str, int] = {}
    strategies: dict[str, int] = {}
    strategy_details: dict[str, int] = {}
    strategy_outcomes: dict[str, dict[str, int]] = {}
    strategy_detail_outcomes: dict[str, dict[str, int]] = {}
    actions: dict[str, int] = {}
    failure_codes: dict[str, int] = {}
    accepted = 0
    for events in grouped.values():
        best = max(
            events,
            key=lambda item: _ATTEMPT_OUTCOME_PRIORITY.get(
                str(item.get("status") or "").strip().lower(), 0
            ),
        )
        status = str(best.get("status") or "unknown").strip().lower()
        if status == "repaired":
            status = "accepted"
        _bump(outcomes, status)
        if status in RECOVERY_ATTEMPT_SUCCESS:
            accepted += 1
        strategy_detail = str(
            best.get("repair_strategy")
            or next(
                (
                    event.get("repair_strategy")
                    for event in events
                    if event.get("repair_strategy")
                ),
                "",
            )
            or "unknown"
        )
        _bump(strategy_details, strategy_detail)
        strategy_name = recovery_strategy_name(strategy_detail)
        _bump(strategies, strategy_name)
        _bump(strategy_outcomes.setdefault(strategy_name, {}), status)
        _bump(
            strategy_detail_outcomes.setdefault(strategy_detail, {}), status
        )
        action = str(
            best.get("repair_action")
            or next(
                (event.get("repair_action") for event in events if event.get("repair_action")),
                "",
            )
            or "unknown"
        )
        _bump(actions, action)
        if status not in RECOVERY_ATTEMPT_SUCCESS:
            _bump(
                failure_codes,
                best.get("new_failure_code")
                or best.get("failure_code")
                or "unspecified",
            )
    return {
        "attempts": len(grouped),
        "accepted": accepted,
        "outcomes": outcomes,
        "strategies": strategies,
        "strategy_details": strategy_details,
        "strategy_outcomes": strategy_outcomes,
        "strategy_detail_outcomes": strategy_detail_outcomes,
        "actions": actions,
        "failure_codes": failure_codes,
    }


def summarize_recovery_outcomes(
    slot_lifecycle: Any, repair_history: Any = None
) -> dict[str, Any]:
    """Summarize authoritative slot terminal states and recovery attempts."""
    records = slot_lifecycle if isinstance(slot_lifecycle, dict) else {}
    terminal_status: dict[str, int] = {}
    failure_stages: dict[str, int] = {}
    failure_codes: dict[str, int] = {}
    slots = 0
    entered = 0
    recovered = 0
    for record in records.values():
        if not isinstance(record, dict):
            continue
        slots += 1
        status = str(record.get("terminal_status") or "unknown")
        _bump(terminal_status, status)
        finalized = status == "finalized"
        in_recovery = bool(
            record.get("repaired")
            or record.get("repair_attempts")
            or record.get("repair_manifest_status")
        )
        if in_recovery:
            entered += 1
            if finalized:
                recovered += 1
        if finalized:
            continue
        _bump(failure_stages, record.get("failure_stage") or "unknown")
        _bump(failure_codes, record.get("failure_code") or "unknown")
    repair = summarize_repair_attempts(repair_history)
    return {
        "slots": slots,
        "entered_recovery": entered,
        "recovered": recovered,
        "unresolved": max(entered - recovered, 0),
        "terminal_status": terminal_status,
        "slot_failure_stages": failure_stages,
        "slot_failure_codes": failure_codes,
        "attempts": repair["attempts"],
        "accepted": repair["accepted"],
        "attempt_outcomes": repair["outcomes"],
        "attempt_strategies": repair["strategies"],
        "attempt_strategy_details": repair["strategy_details"],
        "attempt_strategy_outcomes": repair["strategy_outcomes"],
        "attempt_strategy_detail_outcomes": repair["strategy_detail_outcomes"],
        "attempt_actions": repair["actions"],
        "attempt_failure_codes": repair["failure_codes"],
    }


def summarize_duplicate_outcomes(result: dict[str, Any]) -> dict[str, Any]:
    replacements = result.get("duplicate_replacement_attempts") or {}
    failures = result.get("duplicate_replacement_failures") or {}
    unresolved = result.get("unresolved_duplicate_conflicts") or []
    replacement_attempts = 0
    if isinstance(replacements, dict):
        replacement_attempts = sum(
            _nonnegative_int(value) for value in replacements.values()
        )
    replacement_failures = 0
    if isinstance(failures, dict):
        replacement_failures = sum(
            len(value) for value in failures.values() if isinstance(value, list)
        )
    return {
        "repair_rounds": _nonnegative_int(result.get("duplicate_repair_rounds")),
        "replacement_attempts": replacement_attempts,
        "slots_replaced": len(replacements) if isinstance(replacements, dict) else 0,
        "replacement_failures": replacement_failures,
        "unresolved_conflicts": len(unresolved) if isinstance(unresolved, list) else 0,
    }


def summarize_catalog_cache(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "catalog_cache_hit": bool(result.get("assessment_catalog_cache_hit")),
        "compiler_cache_hit": bool(result.get("assessment_compiler_cache_hit")),
        "compatibility_cache_hit": bool(
            result.get("assessment_catalog_compatibility_cache_hit")
        ),
    }


def summarize_pipeline_outcomes(result: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-safe outcome payload persisted with one evaluation run."""
    if not isinstance(result, dict):
        return {}
    return {
        "recovery": summarize_recovery_outcomes(
            result.get("slot_lifecycle"), result.get("validation_repair_history")
        ),
        "duplicates": summarize_duplicate_outcomes(result),
        "catalog_cache": summarize_catalog_cache(result),
    }
