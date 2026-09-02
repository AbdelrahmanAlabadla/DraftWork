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
) -> None:
    """Record raw initial-generation candidates rejected by existing filters."""
    if stats is None or count <= 0:
        return
    counters = _type_counts(stats, model_number, qtype)
    counters["generation_rejected"] += count
    reasons = counters["generation_rejection_reasons"]
    reasons[reason] = reasons.get(reason, 0) + count


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


def validation_failure_category(verdict: dict[str, Any]) -> str:
    return {
        "FIX_ANSWER": "wrong_answer",
        "FIX_OPTIONS": "bad_options",
        "FIX_QUESTION": "unclear_or_invalid_question",
        "FIX_QUESTION_AND_ANSWER": "question_and_answer",
    }.get(str(verdict.get("action") or "").upper(), "other")


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
        if verdict.get("action") == "PASS":
            counters["validation_passed_first"] += 1
        elif verdict.get("action") == "UNVALIDATED":
            counters["validation_unvalidated_first"] += 1
            reasons = counters["validator_failure_reasons"]
            reasons["missing_verdict"] = reasons.get("missing_verdict", 0) + 1
        else:
            counters["validation_failed_first"] += 1
            category = validation_failure_category(verdict)
            reasons = counters["validation_failure_reasons"]
            reasons[category] = reasons.get(category, 0) + 1
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


def _sum_bucket(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field in COUNT_FIELDS:
        target[field] += int(source.get(field, 0) or 0)
    for field in REASON_FIELDS:
        for reason, count in (source.get(field) or {}).items():
            target[field][reason] = target[field].get(reason, 0) + int(count or 0)


def aggregate_persisted_evals(eval_objects: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stored public eval objects without mutating any stored record."""
    overall = {**_empty_counts(), "question_types": {}}
    models: dict[str, Any] = {}
    for stats in eval_objects:
        if not isinstance(stats, dict):
            continue
        source_overall = stats.get("overall") or {}
        _sum_bucket(overall, source_overall)
        for qtype, source in (source_overall.get("question_types") or {}).items():
            target = overall["question_types"].setdefault(qtype, _empty_counts())
            _sum_bucket(target, source)
        for model_key, source_model in (stats.get("models") or {}).items():
            model = models.setdefault(
                str(model_key), {**_empty_counts(), "question_types": {}}
            )
            _sum_bucket(model, source_model)
            for qtype, source in (source_model.get("question_types") or {}).items():
                target = model["question_types"].setdefault(qtype, _empty_counts())
                _sum_bucket(target, source)
    return {"overall": overall, "models": models}


def safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a rounded ratio, or None when the metric has no denominator."""
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 4)


def derived_rates(bucket: dict[str, Any]) -> dict[str, float | None]:
    return {
        "generation_completion_rate": safe_rate(
            int(bucket.get("generated_first", 0)),
            int(bucket.get("requested_questions", 0)),
        ),
        "shortfall_recovery_rate": safe_rate(
            int(bucket.get("shortfall_generated", 0)),
            int(bucket.get("missing_first", 0)),
        ),
        "first_validation_pass_rate": safe_rate(
            int(bucket.get("validation_passed_first", 0)),
            int(bucket.get("validation_total_first", 0)),
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
