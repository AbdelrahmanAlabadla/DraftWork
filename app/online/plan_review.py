from __future__ import annotations

import json
from copy import deepcopy
from itertools import combinations
from typing import Any, Literal

from pydantic import Field, ValidationError

from app.llm.client import request_structured
from app.llm.deepseek import DeepSeekIncompleteError, DeepSeekValidationError
from app.llm.factory import create_plan_review_client
from app.llm.schemas import NonEmptyString, StrictOutputModel
from app.online.assessment_catalog import (
    _difficulty_for,
    _inventory_after_reservation,
    _replacement_representation_rank,
    compatible_task_inventory,
    fitb_answer_keys,
    materialize_fitb_plan_item,
    materialize_plan_item,
    normalize_fitb_answer_key,
)
from app.online.planner import (
    _semantic_plan_conflict,
    finalize_plans,
    plan_structure_errors,
    validate_plans,
)


PLAN_REVIEW_TEMPERATURE = 0.1
PLAN_REVIEW_MAX_TOKENS = 12000


class PlanReviewPairVerdict(StrictOutputModel):
    pair_id: str = Field(min_length=1)
    semantic_overlap: bool
    reason: str = Field(min_length=1, max_length=500)


class PlanReuseDistinction(StrictOutputModel):
    kind: Literal[
        "different_assessment_angle",
        "different_required_method",
        "different_representation",
        "different_constraint_structure",
        "different_direction_of_reasoning",
        "different_application_requiring_new_reasoning",
        "avoid_answer_giveaway",
    ]
    requirement: str = Field(min_length=1, max_length=320)
    reference_slot_ids: list[NonEmptyString] = Field(min_length=1)


class PlanReviewSuggestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    replacement_task_id: str | None = None
    replacement_fitb_entry_id: str | None = None
    selection_type: Literal["unused_task", "intentional_reuse"]
    reason: str = Field(min_length=1, max_length=600)
    suggested_assessment_angle: str = Field(min_length=1, max_length=320)
    reuse_justification: str | None
    required_instance_distinctions: list[PlanReuseDistinction]


class PlanReviewResponse(StrictOutputModel):
    verdicts: list[PlanReviewPairVerdict]
    suggestions: list[PlanReviewSuggestion]


_SYSTEM_PROMPT = """You review the complete planned exam semantically before questions are generated.
Read the full plan as an assessment: determine the knowledge assessed, what the student must do,
the target/result, reasoning path, and observable work product. IDs and matching metadata fields are
signals only; never use them as an automatic overlap verdict. Different IDs may still produce the
same assessment. The same concept or formula may be distinct when the operation and reasoning are
meaningfully different. Changes only to wording, numbers, names, or surface scenario are overlap.
For each real overlap, suggest exactly one slot to change and choose only a candidate ID listed for
that slot (`replacement_task_id`, or `replacement_fitb_entry_id` for FITB).

Priority is strict:
1. If the slot has unused_task candidates, choose the semantically best UNUSED candidate.
2. Only when no unused candidate is provided may you choose intentional_reuse.
3. Never choose the exact task ID already used by a slot that this suggestion must separate from.
4. A replacement must resolve the reported overlap without creating a new overlap elsewhere.

Do not rewrite task metadata, question counts, slot IDs, evidence, or difficulty. Return a short
assessment angle, not a finished question. For intentional reuse, explain why reuse is necessary
and provide concrete instance distinctions from the referenced slots. Different wording, names,
numbers, or surface scenario alone is not a valid distinction. Reuse metadata is not approval of
the future question; the final Duplicate Judge remains independent.

Examples:
- Different task IDs can still overlap: `calculate wavelength from frequency` and `solve c=lambda*f
  for wavelength` have the same operation and target. Mark overlap.
- Same concept can be distinct: `derive P=Fv` and `calculate power from work/time` share power but
  require different operations and targets. Do not mark overlap solely for the shared concept.
- Good unused replacement: replace a second kinetic-energy calculation with an unused task asking
  students to interpret the sign of work from force direction.
- Bad replacement: changing only the object name, numbers, or story while retaining the same
  calculation is still overlap.
- Intentional reuse: when every compatible task is already represented, reuse the least represented
  task only with an explicit different method, representation, direction of reasoning, constraint
  structure, or application that requires genuinely new reasoning.
"""


def _items_for_model(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for values in (plan.get("items") or {}).values()
        for item in values or []
        if isinstance(item, dict)
    ]


def _question_type_by_slot(plan: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("slot_id") or ""): qtype
        for qtype, values in (plan.get("items") or {}).items()
        for item in values or []
        if item.get("slot_id")
    }


def _potential_pairs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose the whole exam to semantic review; code-side equality is only context."""
    items = _items_for_model(plan)
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(items, 2):
        pairs.append({
            "pair_id": f"plan_pair_{len(pairs) + 1}",
            "left_slot_id": str(left.get("slot_id") or ""),
            "right_slot_id": str(right.get("slot_id") or ""),
            "identity_signal": any(
                left.get(field) and left.get(field) == right.get(field)
                for field in ("task_id", "fitb_entry_id", "task_family_id", "coverage_id")
            ),
            "code_conflict_signal": bool(_semantic_plan_conflict(left, right)),
        })
    return pairs


def _active_model_plans(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"model_number": int(plan.get("model_number") or 1), "items": plan.get("items") or {}}]


def _valid_catalog_tasks(
    catalog: dict[str, Any], qtype: str, difficulty: str, valid_chunk_ids: set[str],
) -> list[dict[str, Any]]:
    result = []
    for task in catalog.get("tasks") or []:
        evidence = [str(value) for value in task.get("evidence_chunk_ids") or []]
        if (
            task.get("task_id")
            and qtype in (task.get("allowed_question_types") or [])
            and _difficulty_for(task, difficulty) is not None
            and task.get("text_answerable") is True
            and evidence
            and all(value in valid_chunk_ids for value in evidence)
        ):
            result.append(task)
    return result


def _representation_counts(task: dict[str, Any], represented: list[dict[str, Any]]) -> dict[str, int]:
    rank = _replacement_representation_rank(task, represented, include_task_usage=True)
    return {
        "task_usage_count": int(rank[0]),
        "coverage_count": int(rank[1]),
        "concept_count": int(rank[2]),
        "operation_count": int(rank[3]),
        "target_count": int(rank[4]),
        "family_count": int(rank[5]),
        "reasoning_mode_count": int(rank[6]),
    }


def _used_fitb_keys(items: list[dict[str, Any]], excluding_slot_id: str) -> set[str]:
    return {
        key
        for item in items
        if str(item.get("slot_id") or "") != excluding_slot_id
        for key in fitb_answer_keys(item)
        if normalize_fitb_answer_key(key)
    }


def _candidate_pool(
    catalog: dict[str, Any], plan: dict[str, Any], slot_id: str, qtype: str,
    difficulty: str, valid_chunk_ids: set[str], *, failed_task_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    failed = {str(value) for value in (failed_task_ids or set()) if value}
    items = _items_for_model(plan)
    current = next(item for item in items if str(item.get("slot_id") or "") == slot_id)
    if qtype == "fill_in_the_blank":
        current_id = str(current.get("fitb_entry_id") or "")
        represented = [item for item in items if str(item.get("slot_id") or "") != slot_id]
        used_ids = {str(item.get("fitb_entry_id") or "") for item in represented}
        entries = [
            entry for entry in catalog.get("fitb_bank") or []
            if entry.get("fitb_entry_id") and not entry.get("alias_of_fitb_entry_id")
            and str(entry.get("fitb_entry_id")) not in failed
            and str(entry.get("evidence_chunk_id") or "") in valid_chunk_ids
        ]
        unused = [
            entry for entry in entries
            if str(entry["fitb_entry_id"]) not in used_ids
            and str(entry["fitb_entry_id"]) != current_id
        ]
        used = [
            entry for entry in entries
            if str(entry["fitb_entry_id"]) in used_ids
            and str(entry["fitb_entry_id"]) != current_id
        ] or [entry for entry in entries if str(entry["fitb_entry_id"]) == current_id]
        mode, selected = ("unused_task", unused) if unused else ("intentional_reuse", used)
        rows: list[dict[str, Any]] = []
        for entry in selected:
            entry_id = str(entry["fitb_entry_id"])
            usage = sum(str(item.get("fitb_entry_id") or "") == entry_id for item in represented)
            answer_count = sum(
                normalize_fitb_answer_key(item.get("fitb_answer")) == entry.get("normalized_answer")
                for item in represented
            )
            rows.append({
                "candidate_id": entry_id,
                "fitb_entry_id": entry_id,
                "selection_type": mode,
                "answer": entry.get("answer"),
                "clue_basis": entry.get("clue_basis"),
                "task_operation": "identify_term_from_text_clue",
                "target": entry.get("answer"),
                "reasoning_mode": "conceptual_recall",
                "student_work_product": "complete one textual blank",
                "semantic_conflict_count": answer_count,
                "representation": {
                    "task_usage_count": usage, "coverage_count": 0,
                    "concept_count": answer_count, "operation_count": len(represented),
                    "target_count": answer_count, "family_count": 0,
                    "reasoning_mode_count": len(represented),
                },
                "used_by_slot_ids": sorted(
                    str(item.get("slot_id") or "") for item in represented
                    if str(item.get("fitb_entry_id") or "") == entry_id
                ),
            })
        rows.sort(key=lambda row: (
            row["representation"]["task_usage_count"] if mode == "intentional_reuse" else 0,
            row["semantic_conflict_count"], row["candidate_id"],
        ))
        return rows, mode, compatible_task_inventory(
            catalog, _active_model_plans(plan), qtype, difficulty,
            excluding_slot_id=slot_id, excluded_task_ids=failed,
        )
    current_task_id = str(current.get("task_id") or "")
    represented = [item for item in items if str(item.get("slot_id") or "") != slot_id]
    used_ids = {str(item.get("task_id") or "") for item in represented if item.get("task_id")}
    tasks = [
        task for task in _valid_catalog_tasks(catalog, qtype, difficulty, valid_chunk_ids)
        if str(task.get("task_id") or "") not in failed
    ]
    unused = [
        task for task in tasks
        if str(task.get("task_id") or "") not in used_ids
        and str(task.get("task_id") or "") != current_task_id
    ]
    fitb_keys = _used_fitb_keys(items, slot_id) if qtype == "fill_in_the_blank" else set()
    used = [task for task in tasks if str(task.get("task_id") or "") in used_ids]
    alternatives = [task for task in used if str(task.get("task_id") or "") != current_task_id]
    used = alternatives or [
        task for task in used if str(task.get("task_id") or "") == current_task_id
    ]
    if qtype == "fill_in_the_blank":
        unused_fresh = [task for task in unused if not (fitb_answer_keys(task) & fitb_keys)]
        used_fresh = [task for task in used if not (fitb_answer_keys(task) & fitb_keys)]
        # Preserve the existing FITB policy: a represented task with a fresh answer
        # is safer than an unused task that would repeat an answer. If neither tier
        # has a fresh answer, task-usage priority remains unchanged.
        if unused_fresh:
            unused = unused_fresh
        elif used_fresh:
            unused = []
            used = used_fresh
    if unused:
        mode, selected = "unused_task", unused
    else:
        selected = used
        mode = "intentional_reuse"
    inventory = compatible_task_inventory(
        catalog, _active_model_plans(plan), qtype, difficulty, excluding_slot_id=slot_id,
        excluded_task_ids=failed,
    )
    rows = []
    for task in selected:
        candidate_plan = materialize_plan_item(
            task, "candidate", _difficulty_for(task, difficulty) or "medium"
        )
        conflict_count = sum(
            _semantic_plan_conflict(candidate_plan, item) for item in represented
        )
        counts = _representation_counts(task, represented)
        used_by_slot_ids = sorted(
            str(item.get("slot_id") or "") for item in represented
            if str(item.get("task_id") or "") == str(task.get("task_id") or "")
        )
        rows.append({
            "candidate_id": str(task.get("task_id") or ""),
            "task_id": str(task.get("task_id") or ""),
            "selection_type": mode,
            "concept": task.get("concept"),
            "task_operation": task.get("task_operation"),
            "target": task.get("target"),
            "reasoning_mode": task.get("reasoning_mode"),
            "student_work_product": task.get("student_work_product"),
            "task_family_id": task.get("task_family_id"),
            "coverage_id": task.get("coverage_id"),
            "semantic_conflict_count": int(conflict_count),
            "representation": counts,
            "used_by_slot_ids": used_by_slot_ids,
        })
    rows.sort(key=lambda row: (
        row["representation"]["task_usage_count"] if mode == "intentional_reuse" else 0,
        row["semantic_conflict_count"],
        row["representation"]["coverage_count"],
        row["representation"]["concept_count"],
        row["representation"]["operation_count"],
        row["representation"]["target_count"],
        row["representation"]["family_count"],
        row["representation"]["reasoning_mode_count"],
        row["candidate_id"],
    ))
    return rows, mode, inventory


def _prompt_payload(
    catalog: dict[str, Any], plan: dict[str, Any], pairs: list[dict[str, Any]],
    difficulty: str, valid_chunk_ids: set[str], *, rejection_reason: str = "",
) -> tuple[str, dict[str, tuple[str, dict[str, Any]]]]:
    qtypes = _question_type_by_slot(plan)
    candidates: dict[str, list[dict[str, Any]]] = {}
    lookup: dict[str, tuple[str, dict[str, Any]]] = {}
    relevant_slots = sorted({
        value for pair in pairs for value in (pair["left_slot_id"], pair["right_slot_id"])
    })
    for slot_id in relevant_slots:
        rows, mode, _inventory = _candidate_pool(
            catalog, plan, slot_id, qtypes[slot_id], difficulty, valid_chunk_ids
        )
        candidates[slot_id] = rows
        for row in rows:
            lookup[f"{slot_id}:{row['candidate_id']}"] = (mode, row)
    registry = {
        str(item.get("slot_id") or ""): {
            key: item.get(key) for key in (
                "task_id", "fitb_entry_id", "task_family_id", "coverage_id", "concept", "knowledge_focus",
                "relationship", "task_operation", "target", "reasoning_mode",
                "student_work_product", "given_conditions", "scenario_family", "difficulty",
            )
        }
        for item in _items_for_model(plan)
    }
    retry = (
        "\n\nThe previous suggestion was rejected by code for this exact reason:\n"
        f"{rejection_reason}\nReturn a corrected complete response."
        if rejection_reason else ""
    )
    prompt = (
        "Review the complete plan registry and every candidate pair. Return verdict rows only for "
        "pairs that are genuine semantic overlaps; an empty verdicts array means the whole plan is distinct. "
        "Suggestions are required for every connected group containing a semantic_overlap=true "
        "verdict; one changed slot may resolve several pair verdicts. Choose only from the candidate "
        "list for that slot, preserving its listed selection_type. If all pairs are distinct, return "
        "an empty suggestions array.\n\n"
        f"Plan registry:\n{json.dumps(registry, ensure_ascii=False)}\n\n"
        f"Candidate pairs:\n{json.dumps(pairs, ensure_ascii=False)}\n\n"
        f"Allowed replacement candidates by slot:\n{json.dumps(candidates, ensure_ascii=False)}"
        f"{retry}"
    )
    return prompt, lookup


def _conflict_groups(pairs: list[dict[str, Any]], overlap_pair_ids: set[str]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {}
    for pair in pairs:
        if pair["pair_id"] not in overlap_pair_ids:
            continue
        left, right = pair["left_slot_id"], pair["right_slot_id"]
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    groups: list[set[str]] = []
    visited: set[str] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        pending, group = [start], set()
        while pending:
            slot = pending.pop()
            if slot in group:
                continue
            group.add(slot); visited.add(slot); pending.extend(adjacency.get(slot, set()) - group)
        groups.append(group)
    return groups


def _validate_response(
    response: PlanReviewResponse, catalog: dict[str, Any], plan: dict[str, Any],
    pairs: list[dict[str, Any]], difficulty: str, valid_chunk_ids: set[str],
    lookup: dict[str, tuple[str, dict[str, Any]]],
) -> list[str]:
    errors: list[str] = []
    expected_pair_ids = [pair["pair_id"] for pair in pairs]
    found_pair_ids = [verdict.pair_id for verdict in response.verdicts]
    if (
        any(pair_id not in set(expected_pair_ids) for pair_id in found_pair_ids)
        or len(found_pair_ids) != len(set(found_pair_ids))
    ):
        errors.append(f"pair_id values must be unique members of {expected_pair_ids}; received {found_pair_ids}")
        return errors
    pair_by_id = {pair["pair_id"]: pair for pair in pairs}
    overlap_ids = {verdict.pair_id for verdict in response.verdicts if verdict.semantic_overlap}
    groups = _conflict_groups(pairs, overlap_ids)
    suggestion_slots = [suggestion.slot_id for suggestion in response.suggestions]
    if len(suggestion_slots) != len(set(suggestion_slots)):
        errors.append("Each slot_id may have at most one Plan Review suggestion.")
    for group in groups:
        if not any(slot_id in group for slot_id in suggestion_slots):
            errors.append(f"No suggestion resolves overlapping slots {sorted(group)}.")
    overlap_slots = {
        slot for pair_id in overlap_ids for slot in (
            pair_by_id[pair_id]["left_slot_id"], pair_by_id[pair_id]["right_slot_id"]
        )
    }
    qtypes = _question_type_by_slot(plan)
    catalog_by_id = {str(task.get("task_id") or ""): task for task in catalog.get("tasks") or []}
    fitb_by_id = {
        str(entry.get("fitb_entry_id") or ""): entry
        for entry in catalog.get("fitb_bank") or [] if entry.get("fitb_entry_id")
    }
    for suggestion in response.suggestions:
        if suggestion.slot_id not in overlap_slots:
            errors.append(f"Slot {suggestion.slot_id} is not part of a reported overlap.")
            continue
        qtype = qtypes[suggestion.slot_id]
        replacement_id = (
            suggestion.replacement_fitb_entry_id
            if qtype == "fill_in_the_blank" else suggestion.replacement_task_id
        )
        if qtype == "fill_in_the_blank":
            if not replacement_id or suggestion.replacement_task_id is not None:
                errors.append(f"FITB suggestion for {suggestion.slot_id} must use replacement_fitb_entry_id only.")
                continue
        elif not replacement_id or suggestion.replacement_fitb_entry_id is not None:
            errors.append(f"Suggestion for {suggestion.slot_id} must use replacement_task_id only.")
            continue
        key = f"{suggestion.slot_id}:{replacement_id}"
        allowed = lookup.get(key)
        if allowed is None:
            errors.append(
                f"Candidate {replacement_id} is not allowed for {suggestion.slot_id}."
            )
            continue
        expected_mode, _row = allowed
        if suggestion.selection_type != expected_mode:
            errors.append(
                f"Candidate {replacement_id} for {suggestion.slot_id} must be marked "
                f"{expected_mode}, not {suggestion.selection_type}."
            )
        if qtype == "fill_in_the_blank":
            entry = fitb_by_id.get(str(replacement_id))
            if entry is None:
                errors.append(f"FITB entry {replacement_id} does not exist in the catalog.")
            elif str(entry.get("evidence_chunk_id") or "") not in valid_chunk_ids:
                errors.append(f"FITB entry {replacement_id} has invalid evidence.")
            task = None
        else:
            task = catalog_by_id.get(str(replacement_id))
        if task is None:
            if qtype != "fill_in_the_blank":
                errors.append(f"Task {replacement_id} does not exist in the catalog.")
                continue
        else:
            evidence = [str(value) for value in task.get("evidence_chunk_ids") or []]
            if qtype not in (task.get("allowed_question_types") or []):
                errors.append(f"Task {replacement_id} does not support {qtype}.")
            if _difficulty_for(task, difficulty) is None:
                errors.append(f"Task {replacement_id} does not support difficulty {difficulty}.")
            if not evidence or any(value not in valid_chunk_ids for value in evidence):
                errors.append(f"Task {replacement_id} has invalid evidence.")
        if suggestion.selection_type == "unused_task":
            if suggestion.reuse_justification is not None or suggestion.required_instance_distinctions:
                errors.append(f"Unused replacement for {suggestion.slot_id} must not carry reuse metadata.")
        elif not suggestion.reuse_justification or not suggestion.required_instance_distinctions:
            errors.append(f"Intentional reuse for {suggestion.slot_id} requires justification and distinctions.")
        plan_slot_ids = set(qtypes)
        for distinction in suggestion.required_instance_distinctions:
            if any(slot_id not in plan_slot_ids for slot_id in distinction.reference_slot_ids):
                errors.append(f"Reuse distinction for {suggestion.slot_id} references an unknown slot.")
        if suggestion.selection_type == "intentional_reuse":
            used_by = set(_row.get("used_by_slot_ids") or [])
            referenced = {
                slot_id for distinction in suggestion.required_instance_distinctions
                for slot_id in distinction.reference_slot_ids
            }
            if used_by and not (used_by & referenced):
                errors.append(
                    f"Intentional reuse for {suggestion.slot_id} must reference a slot already "
                    f"using candidate {replacement_id}: {sorted(used_by)}."
                )
        conflicting_slots = {
            slot for pair_id in overlap_ids
            for slot in (
                pair_by_id[pair_id]["left_slot_id"],
                pair_by_id[pair_id]["right_slot_id"],
            )
            if suggestion.slot_id in {
                pair_by_id[pair_id]["left_slot_id"],
                pair_by_id[pair_id]["right_slot_id"],
            } and slot != suggestion.slot_id
        }
        conflict_items = [
            item for item in _items_for_model(plan)
            if str(item.get("slot_id") or "") in conflicting_slots
        ]
        if any(
            str(item.get("fitb_entry_id") or item.get("task_id") or "")
            == str(replacement_id)
            for item in conflict_items
        ):
            errors.append(
                f"Candidate {replacement_id} for {suggestion.slot_id} is already used by "
                f"a directly conflicting slot {sorted(conflicting_slots)}."
            )
        if task is not None:
            candidate_plan = materialize_plan_item(
                task, suggestion.slot_id,
                _difficulty_for(task, difficulty) or "medium",
            )
            if any(_semantic_plan_conflict(candidate_plan, item) for item in conflict_items):
                errors.append(
                    f"Candidate {replacement_id} for {suggestion.slot_id} does not resolve "
                    f"the semantic conflict with {sorted(conflicting_slots)}."
                )
    return errors


def _apply_suggestion(
    plan: dict[str, Any], catalog: dict[str, Any], suggestion: dict[str, Any],
    difficulty: str, valid_chunk_ids: set[str], *, source: str,
) -> dict[str, Any]:
    slot_id = str(suggestion["slot_id"])
    qtypes = _question_type_by_slot(plan)
    qtype = qtypes[slot_id]
    items = _items_for_model(plan)
    current = next(item for item in items if str(item.get("slot_id") or "") == slot_id)
    if qtype == "fill_in_the_blank":
        task = next(
            entry for entry in catalog.get("fitb_bank") or []
            if str(entry.get("fitb_entry_id") or "") == str(suggestion["replacement_fitb_entry_id"])
        )
        candidate_id = str(task["fitb_entry_id"])
    else:
        task = next(
            task for task in catalog.get("tasks") or []
            if str(task.get("task_id") or "") == str(suggestion["replacement_task_id"])
        )
        candidate_id = str(task["task_id"])
    represented = [item for item in items if str(item.get("slot_id") or "") != slot_id]
    related_slots = sorted({
        str(item.get("slot_id") or "") for item in represented
        if (
            str(item.get("fitb_entry_id") or item.get("task_id") or "") == candidate_id
            or (
                task.get("task_family_id")
                and item.get("task_family_id") == task.get("task_family_id")
            )
        )
    })
    distinctions = list(suggestion.get("required_instance_distinctions") or [])
    if suggestion["selection_type"] == "intentional_reuse" and not distinctions:
        distinctions = [{
            "kind": "different_assessment_angle",
            "requirement": suggestion["suggested_assessment_angle"],
            "reference_slot_ids": related_slots or [str(current.get("slot_id") or "")],
        }]
    replacement = (
        materialize_fitb_plan_item(task, slot_id, str(current.get("difficulty") or "easy"))
        if qtype == "fill_in_the_blank"
        else materialize_plan_item(
            task, slot_id, _difficulty_for(task, difficulty) or str(current.get("difficulty") or "medium")
        )
    )
    inventory = compatible_task_inventory(
        catalog, _active_model_plans(plan), qtype, difficulty, excluding_slot_id=slot_id,
    )
    reused = suggestion["selection_type"] == "intentional_reuse"
    if qtype == "fill_in_the_blank":
        normalized_answer = normalize_fitb_answer_key(task.get("answer"))
        selection_rank = {
            "task_usage_count": sum(
                str(item.get("fitb_entry_id") or "") == candidate_id
                for item in represented
            ),
            "coverage_count": 0,
            "concept_count": sum(
                normalize_fitb_answer_key(item.get("fitb_answer")) == normalized_answer
                for item in represented
            ),
            "operation_count": sum(
                item.get("task_operation") == "identify_term_from_text_clue"
                for item in represented
            ),
            "target_count": sum(
                normalize_fitb_answer_key(item.get("target")) == normalized_answer
                for item in represented
            ),
            "family_count": 0,
            "reasoning_mode_count": sum(
                item.get("reasoning_mode") == "conceptual_recall"
                for item in represented
            ),
        }
    else:
        selection_rank = _representation_counts(task, represented)
    selection_rank["semantic_conflict_count"] = sum(
        _semantic_plan_conflict(replacement, item) for item in represented
    )
    replacement.update({
        "plan_review_status": "intentional_reuse" if reused else "reassigned_unused",
        "plan_review_reason": (
            "semantic_overlap_resolved_by_distinct_reuse" if reused
            else "semantic_overlap_resolved_by_unused_task"
        ),
        "plan_review_source": source,
        "catalog_replacement_stage": (
            "plan_review_intentional_reuse" if reused else "plan_review_unused_task"
        ),
        "suggested_assessment_angle": suggestion["suggested_assessment_angle"],
        "forced_duplicate_slot_ids": sorted(set(related_slots) | set(
            slot_id
            for distinction in distinctions
            for slot_id in distinction.get("reference_slot_ids") or []
        ) | {
            str(value) for value in suggestion.get("conflicting_slot_ids") or []
            if str(value) and str(value) != slot_id
        }),
        "required_instance_distinctions": distinctions,
        "task_inventory": _inventory_after_reservation(
            inventory, candidate_id, reused=reused
        ),
        "task_reused": bool(reused and qtype != "fill_in_the_blank"),
        "fitb_entry_reused": bool(reused and qtype == "fill_in_the_blank"),
        "selection_rank": selection_rank,
    })
    if reused:
        replacement.update({
            "reuse_reason": (
                "no_unused_compatible_fitb_entry"
                if qtype == "fill_in_the_blank" else "no_unused_compatible_task"
            ),
            "reuse_justification": suggestion.get("reuse_justification") or (
                "No unused compatible catalog task remained after hard constraints."
            ),
            "reused_due_to_catalog_exhaustion": True,
            "reused_task_id": task.get("task_id"),
            "reused_fitb_entry_id": task.get("fitb_entry_id"),
            "reuse_rule": (
                "Follow every required instance distinction. Wording, names, numbers, or surface "
                "scenario changes alone do not qualify."
            ),
        })
    for values in (plan.get("items") or {}).values():
        for index, item in enumerate(values or []):
            if str(item.get("slot_id") or "") == slot_id:
                values[index] = replacement
                return replacement
    raise ValueError(f"Unknown Plan Review slot {slot_id}")


def _simulate_suggestions(
    plan: dict[str, Any], catalog: dict[str, Any], response: PlanReviewResponse,
    pairs: list[dict[str, Any]], difficulty: str, valid_chunk_ids: set[str],
    tasks: list[tuple[str, int]], assessment_profile: dict[str, Any],
    planning_issues: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Apply all suggestions atomically and validate the resulting active plan."""
    trial = deepcopy(plan)
    overlap_ids = {v.pair_id for v in response.verdicts if v.semantic_overlap}
    neighbors: dict[str, set[str]] = {}
    for pair in pairs:
        if pair["pair_id"] not in overlap_ids:
            continue
        left, right = pair["left_slot_id"], pair["right_slot_id"]
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    for suggestion in response.suggestions:
        payload = suggestion.model_dump()
        payload["conflicting_slot_ids"] = sorted(neighbors.get(suggestion.slot_id, set()))
        _apply_suggestion(
            trial, catalog, payload, difficulty, valid_chunk_ids, source="agent"
        )
    trial_plans = finalize_plans([trial], tasks)
    errors = validate_plans(
        trial_plans, tasks, 1, valid_chunk_ids or None, difficulty,
        assessment_profile, planning_issues,
    )
    return trial_plans[0], errors


def _fallback_suggestion(
    catalog: dict[str, Any], plan: dict[str, Any], group: set[str], difficulty: str,
    valid_chunk_ids: set[str], failed_task_ids: dict[str, set[str]],
) -> dict[str, Any] | None:
    qtypes = _question_type_by_slot(plan)
    slot_positions = {
        str(item.get("slot_id") or ""): index
        for index, item in enumerate(_items_for_model(plan))
    }
    ranked: list[tuple[tuple[Any, ...], str, dict[str, Any], str]] = []
    for slot_id in sorted(group, reverse=True):
        rows, mode, _inventory = _candidate_pool(
            catalog, plan, slot_id, qtypes[slot_id], difficulty, valid_chunk_ids,
            failed_task_ids=failed_task_ids.get(slot_id) or set(),
        )
        current = next(
            item for item in _items_for_model(plan)
            if str(item.get("slot_id") or "") == slot_id
        )
        for row in rows:
            reference_slots = group - {slot_id}
            directly_conflicting_ids = {
                str(item.get("fitb_entry_id") or item.get("task_id") or "")
                for item in _items_for_model(plan)
                if str(item.get("slot_id") or "") in reference_slots
            }
            if str(row["candidate_id"]) in directly_conflicting_ids:
                continue
            counts = row["representation"]
            rank = (
                0 if mode == "unused_task" else 1,
                counts["task_usage_count"] if mode == "intentional_reuse" else 0,
                row["semantic_conflict_count"], counts["coverage_count"],
                counts["concept_count"], counts["operation_count"], counts["target_count"],
                counts["family_count"], counts["reasoning_mode_count"],
                int(str(row["candidate_id"]) == str(
                    current.get("fitb_entry_id") or current.get("task_id") or ""
                )),
                -slot_positions.get(slot_id, 0), row["candidate_id"], slot_id,
            )
            ranked.append((rank, slot_id, row, mode))
    if not ranked:
        return None
    _rank, slot_id, row, mode = min(ranked, key=lambda value: value[0])
    reference_slots = sorted(group - {slot_id})
    reused = mode == "intentional_reuse"
    return {
        "slot_id": slot_id,
        "replacement_task_id": (
            None if qtypes[slot_id] == "fill_in_the_blank" else row["candidate_id"]
        ),
        "replacement_fitb_entry_id": (
            row["candidate_id"] if qtypes[slot_id] == "fill_in_the_blank" else None
        ),
        "selection_type": mode,
        "reason": "Deterministic fallback selected the least-conflicting, least-represented valid task.",
        "suggested_assessment_angle": (
            f"Assess {row.get('task_operation')} toward {row.get('target')} through a materially "
            "distinct line of reasoning."
        ),
        "reuse_justification": (
            "No unused compatible task remained; this was the least-represented valid used task."
            if reused else None
        ),
        "required_instance_distinctions": ([{
            "kind": "different_assessment_angle",
            "requirement": (
                f"Assess {row.get('target')} without repeating the assessment work in "
                f"{', '.join(reference_slots)}."
            ),
            "reference_slot_ids": reference_slots or [slot_id],
        }] if reused else []),
        "conflicting_slot_ids": reference_slots,
    }


def review_plan(state: dict[str, Any]) -> dict[str, Any]:
    plans = deepcopy(state.get("plans") or [])
    catalog = state.get("assessment_catalog") or {}
    if not catalog or not plans:
        raise RuntimeError(
            "Plan Review refused missing catalog or planned exam state."
        )
    difficulty = str(state.get("difficulty") or "mix")
    valid_chunk_ids = {
        str(chunk.get("child_id")) for chunk in state.get("retrieved_chunks") or []
        if chunk.get("child_id")
    }
    input_errors = [
        *(state.get("plan_errors") or []),
        *plan_structure_errors(
            plans, state.get("tasks") or [], int(state.get("num_models") or 1)
        ),
    ]
    if input_errors:
        raise RuntimeError(
            "Plan Review refused an invalid Planner result: " + "; ".join(input_errors)
        )
    history: list[dict[str, Any]] = []
    total_attempts = 0
    expected_layout = {
        (int(plan.get("model_number") or 0), qtype): tuple(
            str(item.get("slot_id") or "") for item in values or []
        )
        for plan in plans
        for qtype, values in (plan.get("items") or {}).items()
        if values
    }
    for plan_index, plan in enumerate(plans):
        pairs = _potential_pairs(plan)
        if not pairs:
            for item in _items_for_model(plan):
                item.setdefault("plan_review_status", "clear")
                item.setdefault("plan_review_reason", "no_semantic_overlap")
            continue
        response: PlanReviewResponse | None = None
        reviewed_trial: dict[str, Any] | None = None
        last_overlap_ids: set[str] = set()
        rejection = ""
        failed_task_ids: dict[str, set[str]] = {}
        for attempt in (1, 2):
            total_attempts += 1
            prompt, lookup = _prompt_payload(
                catalog, plan, pairs, difficulty, valid_chunk_ids,
                rejection_reason=rejection,
            )
            try:
                raw = request_structured(
                    create_plan_review_client(), prompt,
                    response_model=PlanReviewResponse,
                    schema_name="plan_review",
                    agent="plan_reviewer",
                    item_count=len(pairs),
                    operation="plan_review",
                    system_prompt=_SYSTEM_PROMPT,
                    temperature=PLAN_REVIEW_TEMPERATURE,
                    max_tokens=PLAN_REVIEW_MAX_TOKENS,
                )
                parsed = (
                    raw if isinstance(raw, PlanReviewResponse)
                    else PlanReviewResponse.model_validate(raw)
                )
            except (ValidationError, DeepSeekValidationError, DeepSeekIncompleteError) as exc:
                rejection = f"Plan Review response failed schema validation: {exc}"
                history.append({
                    "model_number": plan.get("model_number"), "attempt": attempt,
                    "status": "agent_suggestion_rejected", "reason": rejection,
                })
                continue
            last_overlap_ids = {
                verdict.pair_id for verdict in parsed.verdicts if verdict.semantic_overlap
            }
            errors = _validate_response(
                parsed, catalog, plan, pairs, difficulty, valid_chunk_ids, lookup
            )
            if not errors:
                trial, trial_errors = _simulate_suggestions(
                    plan, catalog, parsed, pairs, difficulty, valid_chunk_ids,
                    state.get("tasks") or [], state.get("assessment_profile") or {},
                    state.get("planning_issues") or [],
                )
                if not trial_errors:
                    response = parsed
                    reviewed_trial = trial
                    break
                errors.extend(
                    f"Post-suggestion plan invalid: {error}" for error in trial_errors
                )
            rejection = "; ".join(errors)
            for suggestion in parsed.suggestions:
                rejected_id = (
                    suggestion.replacement_fitb_entry_id or suggestion.replacement_task_id
                )
                if rejected_id:
                    failed_task_ids.setdefault(suggestion.slot_id, set()).add(
                        rejected_id
                    )
            history.append({
                "model_number": plan.get("model_number"), "attempt": attempt,
                "status": "agent_suggestion_rejected", "reason": rejection,
            })
        if response is not None:
            overlap_ids = {v.pair_id for v in response.verdicts if v.semantic_overlap}
            overlap_neighbors: dict[str, set[str]] = {}
            for pair in pairs:
                if pair["pair_id"] not in overlap_ids:
                    continue
                left, right = pair["left_slot_id"], pair["right_slot_id"]
                overlap_neighbors.setdefault(left, set()).add(right)
                overlap_neighbors.setdefault(right, set()).add(left)
            plan = reviewed_trial or plan
            for suggestion in response.suggestions:
                applied = next(
                    item for item in _items_for_model(plan)
                    if str(item.get("slot_id") or "") == suggestion.slot_id
                )
                history.append({
                    "model_number": plan.get("model_number"), "status": "agent_suggestion_applied",
                    "slot_id": suggestion.slot_id,
                    "replacement_task_id": suggestion.replacement_task_id,
                    "replacement_fitb_entry_id": suggestion.replacement_fitb_entry_id,
                    "selection_type": suggestion.selection_type,
                    "suggested_assessment_angle": suggestion.suggested_assessment_angle,
                    "task_inventory": deepcopy(applied.get("task_inventory") or {}),
                })
            for item in _items_for_model(plan):
                item.setdefault("plan_review_status", "clear")
                item.setdefault("plan_review_reason", "no_semantic_overlap")
            history.append({
                "model_number": plan.get("model_number"), "status": "agent_review_completed",
                "overlap_pair_ids": sorted(overlap_ids),
            })
            plans[plan_index] = plan
            continue
        # Both suggestions were invalid. Resolve each potential conflict group using
        # the deterministic selector; if it has no alternative, retain the Planner task.
        groups = _conflict_groups(
            pairs, last_overlap_ids or {pair["pair_id"] for pair in pairs}
        )
        for group in groups:
            fallback = _fallback_suggestion(
                catalog, plan, group, difficulty, valid_chunk_ids, failed_task_ids
            )
            if fallback is None:
                history.append({
                    "model_number": plan.get("model_number"), "status": "original_assignment_retained",
                    "slot_ids": sorted(group), "reason": "No valid catalog alternative remained.",
                })
                continue
            applied = _apply_suggestion(
                plan, catalog, fallback, difficulty, valid_chunk_ids,
                source="deterministic_fallback",
            )
            history.append({
                "model_number": plan.get("model_number"), "status": "fallback_applied",
                "slot_id": fallback["slot_id"],
                "replacement_task_id": fallback["replacement_task_id"],
                "replacement_fitb_entry_id": fallback["replacement_fitb_entry_id"],
                "selection_type": fallback["selection_type"],
                "task_inventory": deepcopy(applied.get("task_inventory") or {}),
            })
        for item in _items_for_model(plan):
            item.setdefault("plan_review_status", "clear")
            item.setdefault("plan_review_reason", "no_semantic_overlap")

    plans = finalize_plans(plans, state.get("tasks") or [])
    actual_layout = {
        (int(plan.get("model_number") or 0), qtype): tuple(
            str(item.get("slot_id") or "") for item in values or []
        )
        for plan in plans
        for qtype, values in (plan.get("items") or {}).items()
        if values
    }
    if actual_layout != expected_layout:
        raise ValueError(
            "Plan Review changed the planned model/type/slot layout; "
            f"expected {expected_layout}, received {actual_layout}."
        )
    errors = validate_plans(
        plans, state.get("tasks") or [], int(state.get("num_models") or 1),
        valid_chunk_ids or None, difficulty, state.get("assessment_profile") or {},
        state.get("planning_issues") or [],
    )
    errors.extend(plan_structure_errors(
        plans, state.get("tasks") or [], int(state.get("num_models") or 1)
    ))
    if errors:
        raise RuntimeError(
            "Plan Review produced an invalid plan: " + "; ".join(dict.fromkeys(errors))
        )
    return {
        "plans": plans,
        "plan_errors": errors,
        "plan_review_history": history,
        "plan_review_attempts": total_attempts,
    }
