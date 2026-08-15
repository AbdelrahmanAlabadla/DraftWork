from __future__ import annotations

import json
from typing import Any, Optional

from app.config import PLANNER_SNIPPET_TOKENS
from app.llm.client import LMStudioClient
from app.logging_conf import get_logger
from app.online.graph import ExamState
from app.online.models import normalize_text
from app.online.prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_plan_repair_prompt,
    build_planner_prompt,
)

logger = get_logger("PLANNER")

MAX_PLAN_ATTEMPTS = 3
_TARGETED_FILL_ATTEMPTS = 2

VALID_TYPES = {"mcq", "true_false", "short_answer"}

# Token-overlap above which two planned concepts (topic + concept_to_test) are
# treated as the same concept across exam models. This is the planner-level
# guard against cross-model concept repetition; reworded-same-concept entries
# overlap highly while genuinely different concepts stay low.
_CROSS_MODEL_OVERLAP_THRESHOLD = 0.6


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_plans(raw: object, num_models: int) -> list[dict[str, Any]]:
    """Normalize one planner response into one plan per exam model.

    Returns ``[{"model_number": N, "items": {qtype: [{topic, concept_to_test}]}}]``
    ordered by model_number. Raw ``questions`` may be missing / malformed; the
    caller must validate counts afterwards.
    """
    raw_exams = raw.get("exams") if isinstance(raw, dict) else None
    if not isinstance(raw_exams, list):
        return []

    by_number: dict[int, dict[str, Any]] = {}
    for entry in raw_exams:
        if not isinstance(entry, dict):
            continue
        try:
            number = int(entry.get("model_number"))
        except (TypeError, ValueError):
            continue
        items: dict[str, list[dict[str, Any]]] = {}
        for q in entry.get("questions") or []:
            if not isinstance(q, dict):
                continue
            qtype = str(q.get("question_type") or "").strip()
            topic = str(q.get("topic") or "").strip()
            concept = str(q.get("concept_to_test") or "").strip()
            if qtype not in VALID_TYPES:
                continue
            items.setdefault(qtype, []).append(
                {"topic": topic, "concept_to_test": concept}
            )
        by_number[number] = {
            "model_number": number,
            "items": {t: items.get(t, []) for t in VALID_TYPES},
        }

    plans = [by_number.get(n) for n in range(1, num_models + 1)]
    return [p for p in plans if p is not None]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _concept_tokens(item: dict[str, Any]) -> set[str]:
    terms = set(normalize_text(item.get("topic", "")))
    terms |= set(normalize_text(item.get("concept_to_test", "")))
    # Drop low-signal tokens so short concept descriptors with shared function
    # words don't falsely collide across models.
    return {t for t in terms if t not in _LOW_SIGNAL_TOKENS}


_LOW_SIGNAL_TOKENS = frozenset(
    (
        "a an and are as at be but by for from has have how i in is it its of on "
        "or that the this to was we what when which who why with what does it's "
        "concept topic distinct about describe explain between using used and of "
        "how one value understand main key basic".split()
    )
)


def _concept_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def validate_plans(
    plans: list[dict[str, Any]],
    tasks: list[tuple[str, int]],
    num_models: int,
) -> list[str]:
    """Return human-readable validation errors for a parsed plan set."""
    errors: list[str] = []

    task_map = {t: c for t, c in tasks}
    if len(plans) != num_models:
        errors.append(f"Expected {num_models} exam models, got {len(plans)}.")

    for model in plans:
        number = model["model_number"]
        items = model["items"]
        for qtype, count in task_map.items():
            got = len(items.get(qtype) or [])
            if got != count:
                errors.append(
                    f"Model {number} has {got} {qtype} plan items; required {count}."
                )
            for entry in items.get(qtype) or []:
                if not entry.get("topic") or not entry.get("concept_to_test"):
                    errors.append(
                        f"Model {number} {qtype}: item is missing topic or "
                        "concept_to_test."
                    )

    # Cross-model concept-diversity check (primary anti-repetition guard).
    concepts_by_type: dict[str, list[tuple[int, set[str]]]] = {}
    for model in plans:
        number = model["model_number"]
        for qtype in VALID_TYPES:
            concepts_by_type.setdefault(qtype, [])
            for entry in model["items"].get(qtype) or []:
                concepts_by_type[qtype].append((number, _concept_tokens(entry)))

    for qtype, entries in concepts_by_type.items():
        for i in range(len(entries)):
            num_i, toks_i = entries[i]
            for j in range(i + 1, len(entries)):
                num_j, toks_j = entries[j]
                if num_i == num_j:
                    continue
                if _concept_overlap(toks_i, toks_j) >= _CROSS_MODEL_OVERLAP_THRESHOLD:
                    errors.append(
                        f"Cross-model concept repetition ({qtype}): Model {num_i} and "
                        f"Model {num_j} plan the same concept. Replace one with a "
                        "different concept."
                    )

    return errors


# --------------------------------------------------------------------------
# Deterministic finalize (trim extras from the end)
# --------------------------------------------------------------------------
def finalize_plans(
    plans: list[dict[str, Any]], tasks: list[tuple[str, int]]
) -> list[dict[str, Any]]:
    """Remove extra per-type plan items from the END of each type's list."""
    task_map = {t: c for t, c in tasks}
    finalized: list[dict[str, Any]] = []
    for model in plans:
        items = {
            qtype: (model["items"].get(qtype) or [])[: task_map.get(qtype, 0)]
            for qtype in VALID_TYPES
        }
        finalized.append({"model_number": model["model_number"], "items": items})
    return finalized


# --------------------------------------------------------------------------
# Planner nodes (LangGraph)
# --------------------------------------------------------------------------
def _call_planner(user_prompt: str) -> object:
    client = LMStudioClient()
    return client.chat_json(
        user_prompt,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        temperature=0.5,
        max_tokens=4096,
    )


def _store_plan_result(state: dict[str, Any], raw: object, tasks, num_models) -> dict:
    plans = parse_plans(raw, num_models)
    plans = finalize_plans(plans, tasks)
    errors = validate_plans(plans, tasks, num_models)
    return {
        "plans": plans,
        "plan_errors": errors,
        "last_plan_raw": _safe_dumps(raw),
    }


def _safe_dumps(raw: object) -> str:
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)


def plan_exams(state: ExamState) -> dict[str, Any]:
    """First planning call (all models at once). Shared across the workflow."""
    document_id = state.get("document_id")
    tasks = state.get("tasks") or []
    num_models = state.get("num_models") or 1
    planner_context = state.get("planner_context") or ""
    difficulty = state.get("difficulty") or "mix"

    logger.info(
        "Planning started | document_id=%s | models=%d | tasks=%s | attempt=1",
        document_id,
        num_models,
        tasks,
    )
    try:
        system_prompt, user_prompt = build_planner_prompt(
            num_models, tasks, planner_context
        )
        raw = _call_planner(user_prompt)
    except Exception as exc:
        logger.warning("Planning call failed | attempt=1 | exc=%s", exc)
        return {
            "plans": [],
            "plan_errors": [f"Planner call failed: {exc}"],
            "plan_attempts": 1,
            "last_plan_raw": "",
        }

    result = _store_plan_result(state, raw, tasks, num_models)
    result["plan_attempts"] = 1
    logger.info(
        "Planning completed | attempt=1 | errors=%d | plans=%d",
        len(result["plan_errors"]),
        len(result["plans"]),
    )
    return result


def repair_plan(state: ExamState) -> dict[str, Any]:
    """Repair the previous plan using its output + the exact validation errors."""
    document_id = state.get("document_id")
    tasks = state.get("tasks") or []
    num_models = state.get("num_models") or 1
    planner_context = state.get("planner_context") or ""
    previous = state.get("last_plan_raw") or ""
    errors = state.get("plan_errors") or []
    attempt = (state.get("plan_attempts") or 0) + 1

    logger.info(
        "Plan repair | document_id=%s | attempt=%d/%d | errors=%d",
        document_id,
        attempt,
        MAX_PLAN_ATTEMPTS,
        len(errors),
    )

    if not previous and not errors:
        return {"plan_errors": [], "plan_attempts": attempt}

    user_prompt = build_plan_repair_prompt(previous, errors, planner_context)
    try:
        raw = _call_planner(user_prompt)
    except Exception as exc:
        logger.warning("Plan repair failed | attempt=%d | exc=%s", attempt, exc)
        return {
            "plans": state.get("plans") or [],
            "plan_errors": [f"Planner repair failed: {exc}"],
            "plan_attempts": attempt,
            "last_plan_raw": previous,
        }

    result = _store_plan_result(
        {**state, "last_plan_raw": previous}, raw, tasks, num_models
    )
    result["plan_attempts"] = attempt
    if not result["plans"] and not result["plan_errors"]:
        result["plan_errors"] = ["Planner repair returned no usable plans."]
    logger.info(
        "Plan repair done | attempt=%d | errors=%d",
        attempt,
        len(result["plan_errors"]),
    )
    return result


# --------------------------------------------------------------------------
# Targeted fill for missing concepts (used by generation prep)
# --------------------------------------------------------------------------
def fill_missing_concepts(
    plans: list[dict[str, Any]],
    tasks: list[tuple[str, int]],
    planner_context: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fill any remaining per-type shortfall with TARGETED planner repairs.

    The generator is never allowed to invent unplanned concepts, so if a model is
    still short after the main repair loop, we ask the planner to add exactly the
    missing number of items for that model/type, inspecting existing entries to
    avoid duplicating concepts. Bounded per (model, type).
    """
    task_map = {t: c for t, c in tasks}
    current = [dict(p) for p in plans]
    warnings: list[str] = []

    for _ in range(_TARGETED_FILL_ATTEMPTS):
        deficits: list[tuple[int, str, int]] = []
        for model in current:
            number = model["model_number"]
            items = model["items"]
            for qtype, count in task_map.items():
                missing = count - len(items.get(qtype) or [])
                if missing > 0:
                    deficits.append((number, qtype, missing))

        if not deficits:
            break

        model_number, qtype, missing = deficits[0]
        existing_lines = "\n".join(
            f"- topic={e.get('topic', '')} | concept_to_test={e.get('concept_to_test', '')}"
            for e in current[model_number - 1]["items"].get(qtype) or []
        )
        prompt = (
            f"Exam Model {model_number} needs {missing} ADDITIONAL {qtype} plan items.\n"
            f"These already exist for that model (do NOT duplicate their concepts):\n"
            f"{existing_lines or '(none yet)'}\n\n"
            f"Add exactly {missing} NEW {qtype} items (topic + concept_to_test). "
            f"Do not reuse any concept already used by this model or by other models. "
            f"Return ONLY the raw JSON of the NEW items array, each element "
            f'{{\"question_type\": \"{qtype}\", \"topic\": ..., \"concept_to_test\": ...}}.\n'
            f"{planner_context}"
        )
        try:
            raw = _call_planner(prompt)
        except Exception as exc:
            warnings.append(
                f"Plan fill failed | model={model_number} type={qtype} | {exc}"
            )
            break

        entries = raw if isinstance(raw, list) else (raw.get("questions") if isinstance(raw, dict) else None)
        added = 0
        if isinstance(entries, list):
            for e in entries:
                if added >= missing:
                    break
                if not isinstance(e, dict):
                    continue
                qtype2 = str(e.get("question_type") or "").strip()
                topic = str(e.get("topic") or "").strip()
                concept = str(e.get("concept_to_test") or "").strip()
                if qtype2 != qtype or not topic or not concept:
                    continue
                current[model_number - 1]["items"][qtype].append(
                    {"topic": topic, "concept_to_test": concept}
                )
                added += 1

        if added == 0:
            warnings.append(
                f"Plan fill returned nothing usable | model={model_number} type={qtype}"
            )

    # Re-finalize to keep any over-shoot trimmed from the end.
    current = finalize_plans(current, tasks)
    for model in current:
        number = model["model_number"]
        for qtype, count in task_map.items():
            got = len(model["items"].get(qtype) or [])
            if got < count:
                warnings.append(
                    f"Model {number} {qtype}: plan still short ({got}/{count}) after "
                    "repairs; the exam will be short for that type."
                )
    return current, warnings


def build_planner_context_from_chunks(
    children: list[dict], snippet_tokens: int = PLANNER_SNIPPET_TOKENS
) -> str:
    """Convenience re-export used by tests/other modules if needed."""
    from app.online.retrieval import build_planner_context

    return build_planner_context(children, snippet_tokens)