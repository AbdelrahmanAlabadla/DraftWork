from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Optional

from app.config import PLANNER_SNIPPET_TOKENS
from app.llm.client import request_structured
from app.llm.factory import create_llm_client
from app.llm.schemas import (
    plan_fill_response_model,
    plan_repair_response_model,
    planner_response_model,
    strict_json_schema,
)
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

VALID_TYPES = {
    "mcq", "true_false", "fill_in_the_blank", "definition", "short_answer",
    "equation", "word_problem", "essay",
}

def slot_id(model_number: int, qtype: str, index: int) -> str:
    """Return the stable identity of one requested plan/generation slot."""
    return f"m{model_number}_{qtype}_{index}"


def build_plan_schema(
    tasks: list[tuple[str, int]], num_models: int, chunk_ids: list[str]
) -> dict[str, Any]:
    """Build the planner schema from its Pydantic response model."""
    return strict_json_schema(planner_response_model(tasks, num_models, chunk_ids))


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
        items: dict[str, list[dict[str, Any]]] = {t: [] for t in VALID_TYPES}
        grouped = entry.get("items")
        if isinstance(grouped, dict):
            source_entries = [
                (qtype, q)
                for qtype, values in grouped.items()
                if qtype in VALID_TYPES and isinstance(values, list)
                for q in values
            ]
        else:
            source_entries = [
                (str(q.get("question_type") or "").strip(), q)
                for q in (entry.get("questions") or [])
                if isinstance(q, dict)
            ]
        for qtype, q in source_entries:
            if qtype not in VALID_TYPES or not isinstance(q, dict):
                continue
            topic = str(q.get("topic") or "").strip()
            concept = str(q.get("concept") or topic).strip()
            idea = str(q.get("idea_to_test") or q.get("concept_to_test") or "").strip()
            items[qtype].append(
                {
                    "source_chunk_id": str(q.get("source_chunk_id") or "").strip(),
                    "topic": topic,
                    "concept": concept,
                    "idea_to_test": idea,
                    # Compatibility with existing generator prompt builders.
                    "concept_to_test": idea,
                }
            )
        by_number[number] = {
            "model_number": number,
            "items": items,
        }

    plans = [by_number.get(n) for n in range(1, num_models + 1)]
    return [p for p in plans if p is not None]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _concept_tokens(item: dict[str, Any]) -> set[str]:
    # Broad concepts may intentionally repeat across exam versions. Compare the
    # specific grounded idea only, so equivalent coverage remains possible.
    terms = set(
        normalize_text(item.get("idea_to_test") or item.get("concept_to_test", ""))
    )
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
    valid_chunk_ids: set[str] | None = None,
) -> list[str]:
    """Return human-readable validation errors for a parsed plan set."""
    errors: list[str] = []

    task_map = {t: c for t, c in tasks}
    if len(plans) != num_models:
        errors.append(f"Expected {num_models} exam models, got {len(plans)}.")
    present_models = {int(model["model_number"]) for model in plans}
    for number in range(1, num_models + 1):
        if number in present_models:
            continue
        for qtype, count in task_map.items():
            for index in range(1, count + 1):
                errors.append(
                    f"Slot {slot_id(number, qtype, index)} is missing from the plan."
                )

    for model in plans:
        number = model["model_number"]
        items = model["items"]
        for qtype, count in task_map.items():
            got = len(items.get(qtype) or [])
            if got < count:
                for index in range(got + 1, count + 1):
                    errors.append(
                        f"Slot {slot_id(number, qtype, index)} is missing from the plan."
                    )
            elif got > count:
                errors.append(
                    f"Model {number} has {got} {qtype} plan items; required {count}."
                )
            for index, entry in enumerate(items.get(qtype) or [], start=1):
                sid = slot_id(number, qtype, index)
                if not all(
                    entry.get(field)
                    for field in ("source_chunk_id", "topic", "concept", "idea_to_test")
                ):
                    errors.append(
                        f"Slot {sid} is missing source_chunk_id, topic, concept, or idea_to_test."
                    )
                elif valid_chunk_ids is not None and entry["source_chunk_id"] not in valid_chunk_ids:
                    errors.append(f"Slot {sid} references an unselected source chunk.")

    # Models may share a concept for equivalent coverage. Only an effectively
    # identical idea assignment is rejected across models.
    concepts_by_type: dict[str, list[tuple[int, str, set[str]]]] = {}
    for model in plans:
        number = model["model_number"]
        for qtype in VALID_TYPES:
            concepts_by_type.setdefault(qtype, [])
            for index, entry in enumerate(model["items"].get(qtype) or [], start=1):
                concepts_by_type[qtype].append(
                    (number, slot_id(number, qtype, index), _concept_tokens(entry))
                )

    for qtype, entries in concepts_by_type.items():
        for i in range(len(entries)):
            num_i, sid_i, toks_i = entries[i]
            for j in range(i + 1, len(entries)):
                num_j, sid_j, toks_j = entries[j]
                if num_i == num_j:
                    continue
                if _concept_overlap(toks_i, toks_j) >= 0.85:
                    errors.append(
                        f"Slot {sid_j} repeats the same {qtype} idea as {sid_i}. "
                        "Assign a different grounded idea."
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
        items: dict[str, list[dict[str, Any]]] = {}
        for qtype in VALID_TYPES:
            values = (model["items"].get(qtype) or [])[: task_map.get(qtype, 0)]
            for index, item in enumerate(values, start=1):
                item["slot_id"] = slot_id(model["model_number"], qtype, index)
            items[qtype] = values
        finalized.append({"model_number": model["model_number"], "items": items})
    return finalized


# --------------------------------------------------------------------------
# Planner nodes (LangGraph)
# --------------------------------------------------------------------------
def _call_planner(
    user_prompt: str,
    *,
    response_model: type,
    item_count: int,
    max_tokens: int,
    schema_name: str = "grounded_exam_plan",
    expected_ids: list[str] | None = None,
) -> object:
    client = create_llm_client()
    return request_structured(
        client,
        user_prompt,
        response_model=response_model,
        schema_name=schema_name,
        agent="planner",
        item_count=item_count,
        expected_ids=expected_ids,
        expected_id_field="slot_id" if expected_ids else None,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=max_tokens,
    )


def _store_plan_result(state: dict[str, Any], raw: object, tasks, num_models) -> dict:
    plans = parse_plans(raw, num_models)
    plans = finalize_plans(plans, tasks)
    chunk_ids = {
        str(chunk.get("child_id"))
        for chunk in state.get("retrieved_chunks") or []
        if chunk.get("child_id")
    }
    errors = validate_plans(plans, tasks, num_models, chunk_ids or None)
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
    language = state.get("document_language") or "en"

    logger.info(
        "Planning started | document_id=%s | models=%d | tasks=%s | attempt=1",
        document_id,
        num_models,
        tasks,
    )
    chunk_ids = [
        str(chunk.get("child_id"))
        for chunk in state.get("retrieved_chunks") or []
        if chunk.get("child_id")
    ]
    response_model = planner_response_model(tasks, num_models, chunk_ids)
    requested_slots = sum(count for _qtype, count in tasks) * num_models
    try:
        system_prompt, user_prompt = build_planner_prompt(
            num_models, tasks, planner_context, language=language
        )
        raw = _call_planner(
            user_prompt,
            response_model=response_model,
            item_count=requested_slots,
            max_tokens=min(16000, max(4096, requested_slots * 160)),
        )
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


_SLOT_RE = re.compile(
    r"\b(m\d+_(?:mcq|true_false|fill_in_the_blank|definition|short_answer|equation|word_problem|essay)_\d+)\b"
)


def _invalid_slot_ids(errors: list[str]) -> list[str]:
    """Return unique slot IDs named by plan validation errors."""
    result: list[str] = []
    for error in errors:
        for sid in _SLOT_RE.findall(error):
            if sid not in result:
                result.append(sid)
    return result


def _slot_location(sid: str) -> tuple[int, str, int] | None:
    match = re.fullmatch(
        r"m(\d+)_(mcq|true_false|fill_in_the_blank|definition|short_answer|equation|word_problem|essay)_(\d+)", sid
    )
    if not match:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def _merge_plan_replacements(
    plans: list[dict[str, Any]], replacements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = deepcopy(plans)
    by_number = {int(p["model_number"]): p for p in merged}
    for replacement in replacements:
        sid = str(replacement.get("slot_id") or "")
        location = _slot_location(sid)
        if not location:
            continue
        model_number, qtype, index = location
        model = by_number.get(model_number)
        if model is None:
            model = {"model_number": model_number, "items": {}}
            merged.append(model)
            by_number[model_number] = model
        values = model["items"].setdefault(qtype, [])
        normalized = {
            "slot_id": sid,
            "source_chunk_id": str(replacement.get("source_chunk_id") or "").strip(),
            "topic": str(replacement.get("topic") or "").strip(),
            "concept": str(replacement.get("concept") or "").strip(),
            "idea_to_test": str(replacement.get("idea_to_test") or "").strip(),
        }
        normalized["concept_to_test"] = normalized["idea_to_test"]
        if index <= len(values):
            values[index - 1] = normalized
        elif index == len(values) + 1:
            values.append(normalized)
    return sorted(merged, key=lambda plan: int(plan["model_number"]))


def repair_plan(state: ExamState) -> dict[str, Any]:
    """Repair only invalid slot IDs and retain the best complete plan so far."""
    document_id = state.get("document_id")
    tasks = state.get("tasks") or []
    num_models = state.get("num_models") or 1
    errors = list(state.get("plan_errors") or [])
    plans = deepcopy(state.get("plans") or [])
    attempt = (state.get("plan_attempts") or 0) + 1
    slot_ids = _invalid_slot_ids(errors)
    chunks = state.get("retrieved_chunks") or []
    chunk_ids = [str(c.get("child_id")) for c in chunks if c.get("child_id")]

    logger.info(
        "Targeted plan repair | document_id=%s | attempt=%d/%d | errors=%d | slots=%s",
        document_id,
        attempt,
        MAX_PLAN_ATTEMPTS,
        len(errors),
        slot_ids,
    )
    if not errors:
        return {"plan_errors": [], "plan_attempts": attempt}
    if not slot_ids:
        return {
            "plans": plans,
            "plan_errors": errors,
            "plan_attempts": attempt,
            "last_plan_raw": state.get("last_plan_raw") or "",
        }

    relevant_ids = {
        str(item.get("source_chunk_id"))
        for plan in plans
        for values in plan.get("items", {}).values()
        for item in values
        if item.get("slot_id") in slot_ids and item.get("source_chunk_id")
    }
    relevant_chunks = [c for c in chunks if not relevant_ids or str(c.get("child_id")) in relevant_ids]
    source = "\n\n".join(
        f"[source_chunk_id={c.get('child_id')}]\n{c.get('content', '')}"
        for c in relevant_chunks
    )
    prompt = (
        "Repair ONLY the listed plan slots. Return exactly one replacement per slot ID. "
        "Keep the same broad concept when it is valid, but choose a different grounded "
        "idea or angle when repetition is the error. Use only the supplied chunk IDs.\n\n"
        f"Invalid slots: {slot_ids}\n"
        f"Validation errors:\n" + "\n".join(f"- {e}" for e in errors) + "\n\n"
        f"Relevant selected source chunks:\n{source}"
    )
    try:
        client = create_llm_client()
        response_model = plan_repair_response_model(slot_ids, chunk_ids)
        raw = request_structured(
            client,
            prompt,
            response_model=response_model,
            schema_name="plan_slot_repair",
            agent="planner",
            item_count=len(slot_ids),
            expected_ids=slot_ids,
            expected_id_field="slot_id",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=max(2048, len(slot_ids) * 256),
        )
        replacements = raw.get("replacements") if isinstance(raw, dict) else []
        candidate = _merge_plan_replacements(
            plans, replacements if isinstance(replacements, list) else []
        )
        candidate = finalize_plans(candidate, tasks)
        candidate_errors = validate_plans(candidate, tasks, num_models, set(chunk_ids))
    except Exception as exc:
        logger.warning("Targeted plan repair failed | attempt=%d | exc=%s", attempt, exc)
        return {
            "plans": plans,
            "plan_errors": errors,
            "plan_attempts": attempt,
            "last_plan_raw": state.get("last_plan_raw") or "",
        }

    improved = len(candidate_errors) < len(errors)
    chosen_plans = candidate if improved else plans
    chosen_errors = candidate_errors if improved else errors
    logger.info(
        "Targeted plan repair done | attempt=%d | old_errors=%d | new_errors=%d | kept=%s",
        attempt,
        len(errors),
        len(candidate_errors),
        "candidate" if improved else "previous_best",
    )
    return {
        "plans": chosen_plans,
        "plan_errors": chosen_errors,
        "plan_attempts": attempt,
        "last_plan_raw": json.dumps(chosen_plans, ensure_ascii=False),
    }


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
            f"Add exactly {missing} NEW {qtype} items. "
            f"Do not reuse any concept already used by this model or by other models. "
            "For each item provide source_chunk_id, topic, concept, and idea_to_test. "
            "Use only source_chunk_id values present in the supplied context.\n"
            f"{planner_context}"
        )
        try:
            raw = _call_planner(
                prompt,
                response_model=plan_fill_response_model(missing),
                item_count=missing,
                schema_name="plan_fill",
                max_tokens=max(2048, missing * 192),
            )
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
                topic = str(e.get("topic") or "").strip()
                concept = str(e.get("idea_to_test") or "").strip()
                if not topic or not concept:
                    continue
                current[model_number - 1]["items"][qtype].append(
                    {
                        "source_chunk_id": str(e.get("source_chunk_id") or ""),
                        "topic": topic,
                        "concept": str(e.get("concept") or topic),
                        "idea_to_test": concept,
                        "concept_to_test": concept,
                    }
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
