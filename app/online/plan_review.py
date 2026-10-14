from __future__ import annotations

import json
import re
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
PLAN_REVIEW_MAX_TOKENS = 45000
MAX_PLAN_DUPLICATE_REPAIR_ATTEMPTS = 3

_DUPLICATE_PLAN_ERROR_MARKERS = (
    "duplicates catalog task",
    "duplicates the planned reasoning task",
    "repeats the same scenario/setup and conditions",
    "repeats the same ",
)
_DUPLICATE_SLOT_PATTERNS = (
    re.compile(r"Slot\s+(\S+)\s+duplicates catalog task .*? already used by\s+(\S+?)(?:;|\.|$)", re.I),
    re.compile(r"Slot\s+(\S+)\s+duplicates .*? in\s+(\S+?)(?:;|\.|$)", re.I),
    re.compile(r"Slot\s+(\S+)\s+repeats .*? as\s+(\S+?)(?:;|\.|$)", re.I),
)


def _is_duplicate_plan_error(error: object) -> bool:
    value = str(error or "").casefold()
    return any(marker in value for marker in _DUPLICATE_PLAN_ERROR_MARKERS)


def _known_duplicate_conflicts(errors: list[str]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for error in errors:
        if not _is_duplicate_plan_error(error):
            continue
        for pattern in _DUPLICATE_SLOT_PATTERNS:
            match = pattern.search(error)
            if not match:
                continue
            repair_slot, conflict_slot = match.groups()
            record = {
                "repair_slot_id": repair_slot.rstrip(".;"),
                "conflict_slot_id": conflict_slot.rstrip(".;"),
                "reason": error,
            }
            if record not in conflicts:
                conflicts.append(record)
            break
    return conflicts


def _persisted_duplicate_conflicts(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover conflict context carried by a slot across bounded review cycles."""
    conflicts: list[dict[str, Any]] = []
    for plan in plans:
        for item in _items_for_model(plan):
            repair_slot = str(item.get("slot_id") or "")
            conflict_with = item.get("plan_duplicate_conflict_with")
            conflict_slots = conflict_with if isinstance(conflict_with, list) else [conflict_with]
            for conflict_slot in conflict_slots:
                conflict_slot = str(conflict_slot or "")
                if not repair_slot or not conflict_slot:
                    continue
                record = {
                    "repair_slot_id": repair_slot,
                    "conflict_slot_id": conflict_slot,
                    "reason": str(item.get("plan_duplicate_conflict_reason") or ""),
                }
                if record not in conflicts:
                    conflicts.append(record)
    return conflicts


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


class PlanDifficultyVerdict(StrictOutputModel):
    slot_id: NonEmptyString
    claimed_difficulty: Literal["easy", "medium", "hard"]
    achieved_difficulty: Literal["easy", "medium", "hard"]
    reasoning_profile_supported: bool
    evidence_support_sufficient: bool
    reason: str = Field(min_length=1, max_length=700)


class PlanReviewResponse(StrictOutputModel):
    verdicts: list[PlanReviewPairVerdict]
    suggestions: list[PlanReviewSuggestion]
    difficulty_verdicts: list[PlanDifficultyVerdict] = Field(default_factory=list)


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

Independently inspect any task whose difficulty_adaptation status is adapted. Its actual cognitive
work and reasoning requirements must match the claimed difficulty and remain supported by its exact
evidence boundary. Reasoning mode alone, longer wording, or larger numbers do not establish greater
difficulty. Treat a superficial or unsupported adaptation as invalid semantic planning.
Return exactly one difficulty_verdict for every planned slot. Determine achieved_difficulty from the
concrete cognitive work in task_operation, reasoning requirements, prerequisites, target, and work
product, using only assigned_sources. Do not infer it from the difficulty label, reasoning mode,
wording length, numerical size, or cue words in idea_to_test. If the claimed level or any required
operation/prerequisite is unsupported, report the genuine achieved level and mark support false.

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
            "idea_to_test": task.get("idea_to_test"),
            "relationship": task.get("relationship"),
            "task_operation": task.get("task_operation"),
            "target": task.get("target"),
            "reasoning_mode": task.get("reasoning_mode"),
            "reasoning_depth": task.get("reasoning_depth"),
            "reasoning_requirements": task.get("reasoning_requirements"),
            "prerequisites": task.get("prerequisites"),
            "student_work_product": task.get("student_work_product"),
            "required_givens": task.get("required_givens"),
            "scenario_family": task.get("scenario_family"),
            "evidence_chunk_ids": task.get("evidence_chunk_ids"),
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
    difficulty: str, valid_chunk_ids: set[str], *, evidence_by_id: dict[str, str] | None = None,
    rejection_reason: str = "", known_conflicts: list[dict[str, Any]] | None = None,
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
    sources = evidence_by_id or {}
    registry = {}
    for item in _items_for_model(plan):
        evidence_ids = list(dict.fromkeys(filter(None, [
            str(item.get("source_chunk_id") or ""),
            *map(str, item.get("support_chunk_ids") or []),
        ])))
        registry[str(item.get("slot_id") or "")] = {
            key: item.get(key) for key in (
                "task_id", "fitb_entry_id", "task_family_id", "coverage_id", "concept", "knowledge_focus",
                "idea_to_test", "relationship", "task_operation", "operation", "target", "reasoning_mode",
                "reasoning_depth", "reasoning_requirements", "prerequisites",
                "student_work_product", "required_givens", "given_conditions", "scenario_family", "difficulty",
                "requested_difficulty", "actual_difficulty", "difficulty_adaptation",
                "plan_duplicate_repair_attempt", "plan_duplicate_repair_mode",
                "plan_duplicate_conflict_with", "plan_duplicate_conflict_reason",
                "plan_duplicate_unresolved", "required_instance_distinctions",
                "forced_duplicate_slot_ids", "suggested_assessment_angle",
                "numeric_diversification_allowed", "quantitative_structure_requirement",
            )
        }
        registry[str(item.get("slot_id") or "")]["assigned_sources"] = {
            chunk_id: sources.get(chunk_id, "") for chunk_id in evidence_ids
        }
    retry = (
        "\n\nThe previous suggestion was rejected by code for this exact reason:\n"
        f"{rejection_reason}\nReturn a corrected complete response."
        if rejection_reason else ""
    )
    prompt = (
        "Review the complete plan registry and every candidate pair. Return exactly one "
        "difficulty_verdict per plan-registry slot. Return semantic verdict rows only for "
        "pairs that are genuine semantic overlaps; an empty verdicts array means the whole plan is distinct. "
        "Suggestions are required for every connected group containing a semantic_overlap=true "
        "verdict; one changed slot may resolve several pair verdicts. Choose only from the candidate "
        "list for that slot, preserving its listed selection_type. If all pairs are distinct, return "
        "an empty suggestions array. For every known deterministic conflict, compare BOTH named "
        "slots and use its exact explanation when choosing the repair. Deliberately change a "
        "meaningful conflicting dimension: idea, target, operation, relationship, reasoning path, "
        "givens, scenario/setup, task family, or quantitative structure. Do not randomly swap IDs "
        "or treat wording/numbers alone as semantic diversification.\n\n"
        f"Known deterministic conflicts:\n{json.dumps(known_conflicts or [], ensure_ascii=False)}\n\n"
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
    expected_slot_ids = sorted(_question_type_by_slot(plan))
    difficulty_slot_ids = [verdict.slot_id for verdict in response.difficulty_verdicts]
    if sorted(difficulty_slot_ids) != expected_slot_ids:
        errors.append(
            f"difficulty_verdict slot IDs must be exactly {expected_slot_ids}; "
            f"received {difficulty_slot_ids}."
        )
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


def _apply_difficulty_verdicts(
    plan: dict[str, Any], verdicts: list[PlanDifficultyVerdict], requested: str,
) -> list[dict[str, Any]]:
    """Record independent semantic review and queue genuine mismatches for adjustment."""
    by_slot = {str(item.get("slot_id") or ""): item for item in _items_for_model(plan)}
    events: list[dict[str, Any]] = []
    for verdict in verdicts:
        item = by_slot.get(verdict.slot_id)
        if item is None:
            continue
        claimed = str(item.get("difficulty") or verdict.claimed_difficulty)
        achieved = verdict.achieved_difficulty
        accepted = (
            verdict.claimed_difficulty == claimed
            and achieved == claimed
            and verdict.reasoning_profile_supported
            and verdict.evidence_support_sufficient
        )
        item["difficulty_review"] = {
            "status": "verified" if accepted else "mismatch",
            **verdict.model_dump(),
        }
        if not accepted:
            item["difficulty"] = achieved
            item["actual_difficulty"] = achieved
            item["reasoning_depth"] = achieved
            if (
                requested in {"easy", "medium", "hard"}
                and int(item.get("adjustment_attempts") or 0) < 3
            ):
                item["difficulty_adaptation"] = {
                    "status": "pending", "from_difficulty": achieved,
                    "to_difficulty": requested,
                    "base_task_id": item.get("base_catalog_task_id") or item.get("task_id"),
                    "strategy": None, "failed_strategies": [],
                }
            else:
                item["difficulty_adaptation"] = {
                    **(item.get("difficulty_adaptation") or {}), "status": "failed",
                }
                item["difficulty_fallback_reason"] = verdict.reason
        events.append({
            "slot_id": verdict.slot_id,
            "status": "difficulty_verified" if accepted else "difficulty_mismatch",
            "claimed_difficulty": claimed, "achieved_difficulty": achieved,
            "reason": verdict.reason,
        })
    return events


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


def _slot_item(plan: dict[str, Any], slot_id: str) -> dict[str, Any] | None:
    return next((
        item for item in _items_for_model(plan)
        if str(item.get("slot_id") or "") == slot_id
    ), None)


def _conflict_reason_for_slots(
    conflicts: list[dict[str, Any]], slot_id: str, conflict_slots: set[str],
    verdict_reasons: list[str] | None = None,
) -> str:
    reasons = [
        str(conflict.get("reason") or "") for conflict in conflicts
        if slot_id in {
            str(conflict.get("repair_slot_id") or ""),
            str(conflict.get("conflict_slot_id") or ""),
        } and (
            str(conflict.get("repair_slot_id") or "") in conflict_slots | {slot_id}
            and str(conflict.get("conflict_slot_id") or "") in conflict_slots | {slot_id}
        )
    ]
    reasons.extend(str(value) for value in verdict_reasons or [] if str(value).strip())
    return "; ".join(dict.fromkeys(filter(None, reasons))) or (
        "Plan Review found semantic overlap with " + ", ".join(sorted(conflict_slots))
    )


def _annotate_plan_duplicate_repair(
    item: dict[str, Any], *, attempt: int, mode: str,
    conflict_slots: set[str], reason: str, unresolved: bool,
) -> None:
    item.update({
        "plan_duplicate_repair_attempt": attempt,
        "plan_duplicate_repair_mode": mode,
        "plan_duplicate_conflict_with": (
            sorted(conflict_slots)[0] if len(conflict_slots) == 1
            else sorted(conflict_slots)
        ),
        "plan_duplicate_conflict_reason": reason,
        "plan_duplicate_unresolved": unresolved,
    })


def _controlled_plan_diversification(
    plan: dict[str, Any], group: set[str], reason: str, qtypes: dict[str, str],
    *, attempt: int = 3,
) -> dict[str, Any]:
    """Add a source-bounded diversification contract without changing slot identity."""
    slot_id = sorted(group)[-1]
    item = _slot_item(plan, slot_id)
    if item is None:
        raise ValueError(f"Unknown controlled-diversification slot {slot_id}")
    conflicts = set(group) - {slot_id}
    qtype = qtypes.get(slot_id, "")
    mode = str(item.get("reasoning_mode") or "").casefold()
    operation = str(item.get("task_operation") or item.get("operation") or "").casefold()
    quantitative = bool(
        qtype in {"equation", "word_problem"}
        or mode in {"calculation", "symbolic_manipulation", "optimization", "law_or_theorem_application"}
        or any(value in operation for value in ("calculate", "solve", "compute", "evaluate", "balance"))
    )
    distinction = (
        "Use a different source-supported quantitative setup, givens, constraints, and calculation "
        "path while preserving the assigned operation and difficulty."
        if quantitative else
        "Use a different source-supported idea, scenario, target, relationship, or reasoning path."
    )
    existing = list(item.get("required_instance_distinctions") or [])
    existing.append({
        "kind": "different_constraint_structure" if quantitative else "different_assessment_angle",
        "requirement": distinction,
        "reference_slot_ids": sorted(conflicts),
    })
    item.update({
        "plan_review_status": "intentional_reuse",
        "plan_review_reason": "controlled_plan_duplicate_diversification",
        "plan_review_source": (
            "controlled_diversification" if attempt >= 3 else "targeted_semantic_repair"
        ),
        "suggested_assessment_angle": distinction,
        "forced_duplicate_slot_ids": sorted(set(item.get("forced_duplicate_slot_ids") or []) | conflicts),
        "required_instance_distinctions": existing,
        "numeric_diversification_allowed": quantitative,
        "quantitative_structure_requirement": distinction if quantitative else None,
    })
    _annotate_plan_duplicate_repair(
        item, attempt=attempt,
        mode=("controlled_diversification" if attempt >= 3 else "targeted_semantic_repair"),
        conflict_slots=conflicts, reason=reason, unresolved=False,
    )
    return item


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
    evidence_by_id = {
        str(chunk.get("child_id")): str(chunk.get("content") or "")
        for chunk in state.get("retrieved_chunks") or [] if chunk.get("child_id")
    }
    input_errors = [
        *(state.get("plan_errors") or []),
        *plan_structure_errors(
            plans, state.get("tasks") or [], int(state.get("num_models") or 1)
        ),
    ]
    duplicate_input_errors = [error for error in input_errors if _is_duplicate_plan_error(error)]
    structural_input_errors = [error for error in input_errors if not _is_duplicate_plan_error(error)]
    if structural_input_errors:
        raise RuntimeError(
            "Plan Review refused an invalid Planner result: "
            + "; ".join(structural_input_errors)
        )
    known_conflicts = _known_duplicate_conflicts(duplicate_input_errors)
    for conflict in _persisted_duplicate_conflicts(plans):
        if conflict not in known_conflicts:
            known_conflicts.append(conflict)
    history: list[dict[str, Any]] = list(state.get("plan_review_history") or [])
    duplicate_attempts = dict(state.get("plan_duplicate_repair_attempts") or {})
    unresolved_conflicts = list(state.get("unresolved_plan_duplicate_conflicts") or [])
    total_attempts = 0
    retry_required = False
    prior_rechecks = int(state.get("plan_review_rechecks") or 0)
    rechecks = prior_rechecks
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
        response: PlanReviewResponse | None = None
        reviewed_trial: dict[str, Any] | None = None
        last_overlap_ids: set[str] = set()
        last_difficulty_verdicts: list[PlanDifficultyVerdict] = []
        rejection = ""
        failed_task_ids: dict[str, set[str]] = {}
        exhausted_overlap_groups: list[set[str]] = []
        for attempt in (1, 2):
            total_attempts += 1
            prompt, lookup = _prompt_payload(
                catalog, plan, pairs, difficulty, valid_chunk_ids,
                evidence_by_id=evidence_by_id,
                rejection_reason=rejection,
                known_conflicts=known_conflicts,
            )
            try:
                raw = request_structured(
                    create_plan_review_client(
                        domain=(state.get("assessment_profile") or {}).get("domain")
                    ), prompt,
                    response_model=PlanReviewResponse,
                    schema_name="plan_review",
                    agent="plan_reviewer",
                    item_count=max(len(pairs), len(_items_for_model(plan))),
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
            if sorted(verdict.slot_id for verdict in parsed.difficulty_verdicts) == sorted(
                _question_type_by_slot(plan)
            ):
                last_difficulty_verdicts = list(parsed.difficulty_verdicts)
            parsed_groups = _conflict_groups(pairs, last_overlap_ids)
            exhausted_overlap_groups = [
                group for group in parsed_groups
                if max((int(duplicate_attempts.get(slot_id) or 0) for slot_id in group), default=0)
                >= MAX_PLAN_DUPLICATE_REPAIR_ATTEMPTS
            ]
            if exhausted_overlap_groups and last_difficulty_verdicts:
                response = parsed
                reviewed_trial = deepcopy(plan)
                break
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
            history.extend({
                "model_number": plan.get("model_number"), **event,
            } for event in _apply_difficulty_verdicts(
                plan, response.difficulty_verdicts, difficulty
            ))
            if exhausted_overlap_groups:
                verdict_by_pair = {verdict.pair_id: verdict for verdict in response.verdicts}
                pair_by_id = {pair["pair_id"]: pair for pair in pairs}
                for group in exhausted_overlap_groups:
                    repair_slot = sorted(group)[-1]
                    item = _slot_item(plan, repair_slot)
                    if item is None:
                        continue
                    conflict_slots = set(group) - {repair_slot}
                    reasons = [
                        verdict_by_pair[pair_id].reason
                        for pair_id in overlap_ids
                        if pair_id in verdict_by_pair
                        and {
                            pair_by_id[pair_id]["left_slot_id"],
                            pair_by_id[pair_id]["right_slot_id"],
                        }.issubset(group)
                    ]
                    reason = _conflict_reason_for_slots(
                        known_conflicts, repair_slot, conflict_slots, reasons
                    )
                    _annotate_plan_duplicate_repair(
                        item, attempt=3, mode="controlled_diversification",
                        conflict_slots=conflict_slots, reason=reason, unresolved=True,
                    )
                    record = {
                        "model_number": plan.get("model_number"),
                        "slot_id": repair_slot,
                        "conflict_slot_ids": sorted(conflict_slots),
                        "reason": reason,
                        "attempts": 3,
                        "status": "plan_duplicate_unresolved_kept",
                    }
                    unresolved_conflicts.append(record)
                    history.append(record)
                for item in _items_for_model(plan):
                    item.setdefault("plan_review_status", "clear")
                    item.setdefault("plan_review_reason", "no_semantic_overlap")
                plans[plan_index] = plan
                continue
            for suggestion in response.suggestions:
                applied = next(
                    item for item in _items_for_model(plan)
                    if str(item.get("slot_id") or "") == suggestion.slot_id
                )
                conflict_slots = overlap_neighbors.get(suggestion.slot_id, set())
                repair_attempt = int(duplicate_attempts.get(suggestion.slot_id) or 0) + 1
                duplicate_attempts[suggestion.slot_id] = repair_attempt
                verdict_reasons = [
                    verdict.reason for verdict in response.verdicts
                    if verdict.semantic_overlap and suggestion.slot_id in {
                        next((pair["left_slot_id"] for pair in pairs if pair["pair_id"] == verdict.pair_id), ""),
                        next((pair["right_slot_id"] for pair in pairs if pair["pair_id"] == verdict.pair_id), ""),
                    }
                ]
                conflict_reason = _conflict_reason_for_slots(
                    known_conflicts, suggestion.slot_id, conflict_slots, verdict_reasons
                )
                _annotate_plan_duplicate_repair(
                    applied, attempt=repair_attempt,
                    mode=("controlled_diversification" if repair_attempt >= 3
                          else "targeted_semantic_repair"),
                    conflict_slots=conflict_slots, reason=conflict_reason, unresolved=False,
                )
                history.append({
                    "model_number": plan.get("model_number"), "status": "agent_suggestion_applied",
                    "slot_id": suggestion.slot_id,
                    "replacement_task_id": suggestion.replacement_task_id,
                    "replacement_fitb_entry_id": suggestion.replacement_fitb_entry_id,
                    "selection_type": suggestion.selection_type,
                    "suggested_assessment_angle": suggestion.suggested_assessment_angle,
                    "task_inventory": deepcopy(applied.get("task_inventory") or {}),
                    "plan_duplicate_repair_attempt": repair_attempt,
                    "plan_duplicate_conflict_reason": conflict_reason,
                })
            for item in _items_for_model(plan):
                item.setdefault("plan_review_status", "clear")
                item.setdefault("plan_review_reason", "no_semantic_overlap")
            history.append({
                "model_number": plan.get("model_number"), "status": "agent_review_completed",
                "overlap_pair_ids": sorted(overlap_ids),
            })
            if overlap_ids and response.suggestions:
                retry_required = True
                rechecks += 1
            elif not overlap_ids:
                for item in _items_for_model(plan):
                    if item.get("plan_duplicate_repair_attempt"):
                        item["plan_duplicate_unresolved"] = False
            plans[plan_index] = plan
            continue
        if not last_difficulty_verdicts:
            raise RuntimeError(
                "Plan Review did not return a complete independent difficulty assessment."
            )
        # Both suggestions were invalid. Resolve each potential conflict group using
        # the deterministic selector; if it has no alternative, retain the Planner task.
        groups = _conflict_groups(
            pairs, last_overlap_ids or {pair["pair_id"] for pair in pairs}
        )
        fallback_changed = False
        for group in groups:
            next_attempt = max(
                (int(duplicate_attempts.get(slot_id) or 0) for slot_id in group),
                default=0,
            ) + 1
            repair_slot = sorted(group)[-1]
            conflict_slots = set(group) - {repair_slot}
            conflict_reason = _conflict_reason_for_slots(
                known_conflicts, repair_slot, conflict_slots, [rejection]
            )
            if next_attempt >= MAX_PLAN_DUPLICATE_REPAIR_ATTEMPTS:
                applied = _controlled_plan_diversification(
                    plan, group, conflict_reason, _question_type_by_slot(plan), attempt=3
                )
                duplicate_attempts[repair_slot] = 3
                fallback_changed = True
                history.append({
                    "model_number": plan.get("model_number"),
                    "status": "controlled_diversification_applied",
                    "slot_id": repair_slot,
                    "conflict_slot_ids": sorted(conflict_slots),
                    "plan_duplicate_repair_attempt": 3,
                    "plan_duplicate_conflict_reason": conflict_reason,
                    "numeric_diversification_allowed": bool(
                        applied.get("numeric_diversification_allowed")
                    ),
                })
                continue
            fallback = _fallback_suggestion(
                catalog, plan, group, difficulty, valid_chunk_ids, failed_task_ids
            )
            if fallback is None:
                applied = _controlled_plan_diversification(
                    plan, group, conflict_reason, _question_type_by_slot(plan),
                    attempt=next_attempt,
                )
                duplicate_attempts[repair_slot] = next_attempt
                fallback_changed = True
                history.append({
                    "model_number": plan.get("model_number"),
                    "status": "targeted_contract_diversification_applied",
                    "slot_id": repair_slot,
                    "slot_ids": sorted(group),
                    "reason": conflict_reason,
                    "plan_duplicate_repair_attempt": next_attempt,
                    "numeric_diversification_allowed": bool(
                        applied.get("numeric_diversification_allowed")
                    ),
                })
                continue
            applied = _apply_suggestion(
                plan, catalog, fallback, difficulty, valid_chunk_ids,
                source="deterministic_fallback",
            )
            selected_slot = str(fallback["slot_id"])
            selected_conflicts = set(group) - {selected_slot}
            duplicate_attempts[selected_slot] = next_attempt
            _annotate_plan_duplicate_repair(
                applied, attempt=next_attempt, mode="targeted_semantic_repair",
                conflict_slots=selected_conflicts, reason=conflict_reason, unresolved=False,
            )
            fallback_changed = True
            history.append({
                "model_number": plan.get("model_number"), "status": "fallback_applied",
                "slot_id": fallback["slot_id"],
                "replacement_task_id": fallback["replacement_task_id"],
                "replacement_fitb_entry_id": fallback["replacement_fitb_entry_id"],
                "selection_type": fallback["selection_type"],
                "task_inventory": deepcopy(applied.get("task_inventory") or {}),
                "plan_duplicate_repair_attempt": next_attempt,
                "plan_duplicate_conflict_reason": conflict_reason,
            })
        for item in _items_for_model(plan):
            item.setdefault("plan_review_status", "clear")
            item.setdefault("plan_review_reason", "no_semantic_overlap")
        if fallback_changed:
            retry_required = True
            rechecks += 1
            history.append({
                "model_number": plan.get("model_number"),
                "status": "difficulty_recheck_required",
                "reason": "Deterministic overlap fallback changed a task after the agent review.",
            })
        else:
            history.extend({
                "model_number": plan.get("model_number"), **event,
            } for event in _apply_difficulty_verdicts(
                plan, last_difficulty_verdicts, difficulty
            ))

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
    # Duplicate-only findings are allowed while a bounded targeted repair is
    # awaiting re-review, or after the third attempt has been retained as an
    # auditable unresolved similarity. All other plan validation remains strict.
    if retry_required or unresolved_conflicts:
        errors = [error for error in errors if not _is_duplicate_plan_error(error)]
    if errors:
        raise RuntimeError(
            "Plan Review produced an invalid plan: " + "; ".join(dict.fromkeys(errors))
        )
    return {
        "plans": plans,
        "plan_errors": errors,
        "plan_review_history": history,
        "plan_review_attempts": total_attempts,
        "plan_review_retry_required": retry_required,
        "plan_review_rechecks": rechecks,
        "plan_duplicate_repair_attempts": duplicate_attempts,
        "unresolved_plan_duplicate_conflicts": unresolved_conflicts,
    }
