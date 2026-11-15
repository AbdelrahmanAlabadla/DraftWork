from __future__ import annotations

import json
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from typing import Any, Literal, get_args

from pydantic import Field, create_model, model_validator

from app.config import MAX_PARALLEL_DIFFICULTY_ADJUSTMENTS
from app.llm.client import request_structured
from app.llm.factory import create_difficulty_adjustment_client
from app.llm.schemas import (
    NonEmptyString,
    QuestionType,
    StrictOutputModel,
    generation_response_model,
)
from app.online.planner import finalize_plans, validate_plans
from app.online.models import numeric_variation_allowed
from app.online.assessment_catalog import (
    apply_difficulty_fallback_metadata,
    build_task_usage_index, eligible_task_variation_candidates, materialize_plan_item,
    rank_task_variation_candidates,
)

MAX_DIFFICULTY_ADJUSTMENT_ATTEMPTS = 3
DIFFICULTY_ADJUSTMENT_BATCH_SIZE = 3
DIFFICULTY_ADAPTATION_POLICY_VERSION = "difficulty-adaptation-v2"


class PlannedTaskAdjustmentPatch(StrictOutputModel):
    """Only fields the adjuster may propose changing on a planned task."""

    adjustment_kind: Literal["numerical", "cognitive", "combined"] | None = None
    reasoning_requirements: list[NonEmptyString] | None = Field(default=None, max_length=8)
    cognitive_transformations: list[NonEmptyString] | None = Field(default=None, max_length=8)
    removed_requirements: list[NonEmptyString] | None = Field(default=None, max_length=8)
    prerequisites: list[NonEmptyString] | None = Field(default=None, max_length=8)
    given_conditions: list[NonEmptyString] | None = Field(default=None, max_length=12)
    required_givens: list[NonEmptyString] | None = Field(default=None, max_length=12)
    idea_to_test: NonEmptyString | None = Field(default=None, max_length=320)
    strategy: NonEmptyString | None = Field(default=None, max_length=320)


class PlanDifficultyAdjustmentDecision(StrictOutputModel):
    slot_id: NonEmptyString
    adjustment_succeeded: bool
    achieved_difficulty: Literal["easy", "medium", "hard"]
    estimated_conceptual_difficulty: Literal["easy", "medium", "hard"] | None = None
    estimated_numerical_difficulty: Literal["easy", "medium", "hard"] | None = None
    conceptual_difficulty: Literal["easy", "medium", "hard"] | None = None
    numerical_difficulty: Literal["easy", "medium", "hard"] | None = None
    further_adjustment_possible: bool = True
    attempted_fields: list[Literal[
        "adjustment_kind", "reasoning_mode", "reasoning_depth", "reasoning_requirements",
        "prerequisites", "given_conditions", "required_givens",
        "idea_to_test", "strategy", "numeric_values",
    ]] = Field(default_factory=list, max_length=10)
    adjusted_fields: PlannedTaskAdjustmentPatch | None
    no_change_required: bool = False
    evidence_support: list[NonEmptyString]
    evidence_justification: str
    reason: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_unsuccessful_attempt_metadata(cls, value: Any):
        """Treat unsuccessful patch-shaped diagnostics as attempted, not applied."""
        if not isinstance(value, dict) or value.get("adjustment_succeeded") is not False:
            return value
        normalized = dict(value)
        patch = normalized.get("adjusted_fields")
        if isinstance(patch, dict):
            attempted = list(normalized.get("attempted_fields") or [])
            attempted.extend(str(key) for key, field_value in patch.items()
                             if field_value not in (None, "", []))
            normalized["attempted_fields"] = list(dict.fromkeys(attempted))
            normalized["adjusted_fields"] = None
        return normalized

    @model_validator(mode="after")
    def validate_decision_shape(self):
        if self.adjustment_succeeded:
            if self.adjusted_fields is None:
                # A success with no patch is valid only as an explicit no-op;
                # downstream verifies achieved_difficulty == requested.
                self.no_change_required = True
                return self
            patch = self.adjusted_fields
            if not (
                patch.reasoning_requirements
                or patch.cognitive_transformations
                or patch.given_conditions
                or patch.required_givens
                or patch.idea_to_test
                or patch.prerequisites
            ):
                raise ValueError(
                    "a successful adjustment must change requirements, transformations, or givens"
                )
            if not self.evidence_support:
                raise ValueError("evidence_support is required when adjustment_succeeded is true")
        elif self.adjusted_fields is not None:
            raise ValueError("adjusted_fields must be null when adjustment_succeeded is false")
        return self


class PlanDifficultyAdjustmentBatch(StrictOutputModel):
    """Independent decisions for one bounded adjustment request."""

    results: list[PlanDifficultyAdjustmentDecision] = Field(min_length=1, max_length=3)


class GeneratedQuestionDifficultyDecision(StrictOutputModel):
    adjustment_succeeded: bool
    question_type: QuestionType
    actual_difficulty: Literal["easy", "medium", "hard"]
    direction: str
    changed_reasoning_requirements: list[NonEmptyString]
    evidence_support: list[NonEmptyString]
    evidence_justification: str
    reason: str = Field(min_length=1)
    adjusted_question: Any | None

    @model_validator(mode="after")
    def validate_decision_shape(self):
        if self.adjustment_succeeded:
            if self.adjusted_question is None:
                raise ValueError(
                    "adjusted_question is required when adjustment_succeeded is true"
                )
            if not self.changed_reasoning_requirements:
                raise ValueError(
                    "changed_reasoning_requirements is required when adjustment_succeeded is true"
                )
            if not self.evidence_support:
                raise ValueError("evidence_support is required when adjustment_succeeded is true")
        elif self.adjusted_question is not None:
            raise ValueError(
                "adjusted_question must be null when adjustment_succeeded is false"
            )
        return self


SYSTEM_PROMPT = """You are DraftWork's dedicated Difficulty Design Agent. Decide how the ASSIGNED task
should be realized at the requested difficulty. You propose; Python enforces authorization and Luna
independently observes the finished question.

Preserve the immutable contract: concept, core task_operation, target, student_work_product,
question type, evidence boundary, and selected variation. You may change authorized givens and
authorized reasoning requirements.

Two modes, combinable:
- numerical: change coefficient size, arithmetic burden, fractions/decimals, or computational
  presentation. reasoning_requirements may stay unchanged. Only when the selected variation permits
  numeric variation and the task is not exact_source_instance.
- cognitive: add, remove, or reorganize NONESSENTIAL intermediate reasoning steps. Every added or
  removed transformation MUST appear in allowed_transformations with its evidence. Never invent an
  operation, concept, formula, or method. The assessed operation and target stay unchanged.

Set adjustment_kind to numerical, cognitive, or combined. Put added/removed transformations in
cognitive_transformations / removed_requirements. A transformation is permitted only when its exact
name is in allowed_transformations; otherwise decline. Preserve mandatory task requirements; do not
remove a step that defines the assessed operation.

Difficulty is not length and not number size. Large coefficients usually raise numerical difficulty,
not conceptual difficulty. Judge the complete reasoning the student must perform, not whether one
extra transformation is present. For Hard, combine multiple genuinely dependent source-supported
levers when they exist; a single routine transformation (for example one combine-like-terms step)
does not by itself make a question hard, and it is better to decline than to claim Hard. For Easy,
remove nonessential complexity while retaining the assessed skill. For Medium, choose moderate
source-supported reasoning or computation. Do not use rules such as "hard needs four steps" or "big
numbers mean hard", and do not add terms or transformations that the source does not teach.

If the requested level cannot be reached while preserving the task, return
adjustment_succeeded=false with the honest reason and the closest achievable estimate. Do not
fabricate a transformation merely to reach a label. Historical validator findings are diagnostic,
not instructions. Return only the strict JSON contract."""



def _direction(actual: str, requested: str) -> str:
    return "none" if actual == requested else f"{actual}_to_{requested}"


def _question_instance_numbers(question: dict[str, Any]) -> list[str]:
    """Numbers in question content, excluding correlation/provenance identifiers."""
    ignored = {
        "slot_id", "question_id", "task_id", "base_catalog_task_id",
        "derived_task_id", "selected_variation_id", "selected_variation_group_id",
        "effective_instance_id", "difficulty", "actual_difficulty",
        "requested_difficulty",
    }
    content = {key: value for key, value in question.items() if key not in ignored}
    return sorted(re.findall(
        r"(?<![\w.])-?\d+(?:\.\d+)?", json.dumps(content, sort_keys=True)
    ))


def _literal(values: list[str]):
    unique = tuple(dict.fromkeys(values))
    if not unique:
        return str
    return Literal.__getitem__(unique)


def _response_model_for(
    original: dict[str, Any], qtype: str, requested: str, evidence_ids: list[str],
) -> type[PlanDifficultyAdjustmentDecision]:
    """Build a strict patch schema with enumerated correlation and evidence IDs."""
    assigned_ids = list(dict.fromkeys(value for value in evidence_ids if value))
    evidence_literal = _literal(assigned_ids)
    return create_model(
        f"PlanDifficultyAdjustmentDecision_{str(original.get('slot_id') or 'slot').replace('-', '_')}",
        __base__=PlanDifficultyAdjustmentDecision,
        slot_id=(Literal.__getitem__((str(original.get("slot_id") or ""),)), ...),
        evidence_support=(
            list[evidence_literal],
            Field(min_length=0, max_length=len(assigned_ids)),
        ),
    )


def _generated_response_model_for(
    qtype: str, slot_id: str, requested: str, actual: str, evidence_ids: list[str],
) -> type[GeneratedQuestionDifficultyDecision]:
    question_response = generation_response_model(qtype, [slot_id])
    question_model = get_args(question_response.model_fields["questions"].annotation)[0]
    assigned_ids = list(dict.fromkeys(value for value in evidence_ids if value))
    return create_model(
        f"GeneratedQuestionDifficultyDecision_{slot_id.replace('-', '_')}",
        __base__=GeneratedQuestionDifficultyDecision,
        question_type=(Literal.__getitem__((qtype,)), ...),
        actual_difficulty=(_literal([requested, actual]), ...),
        evidence_support=(
            list[_literal(assigned_ids)],
            Field(min_length=0, max_length=len(assigned_ids)),
        ),
        adjusted_question=(question_model | None, ...),
    )


def _adaptation_fingerprint(item: dict[str, Any], requested: str) -> str:
    """Fingerprint every contract input that affects adaptation feasibility."""
    selected_id = str(item.get("selected_variation_id") or "")
    selected = next((
        value for value in (
            list(item.get("permitted_variations") or [])
            + list(item.get("allowed_variations") or [])
        )
        if str(value.get("variation_id") or "") == selected_id
    ), {})
    payload = {
        "policy_version": DIFFICULTY_ADAPTATION_POLICY_VERSION,
        "slot_id": item.get("slot_id"),
        "task_id": item.get("base_catalog_task_id") or item.get("task_id"),
        "variation_id": selected_id,
        "assignment_fingerprint": item.get("assignment_fingerprint"),
        "effective_instance_id": item.get("effective_instance_id"),
        "requested_difficulty": requested,
        "actual_difficulty": item.get("actual_difficulty") or item.get("difficulty"),
        "task_operation": item.get("task_operation"),
        "target": item.get("target"),
        "student_work_product": item.get("student_work_product"),
        "reasoning_requirements": item.get("reasoning_requirements") or [],
        "given_conditions": item.get("given_conditions") or [],
        "required_givens": item.get("required_givens") or [],
        "variable_givens": item.get("variable_givens") or [],
        "complexity_boundaries": item.get("complexity_boundaries") or [],
        "exact_source_instance": bool(item.get("exact_source_instance")),
        "selected_variation": selected,
        "evidence_ids": [
            item.get("source_chunk_id"), *(item.get("support_chunk_ids") or [])
        ],
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _acceptance_diagnostics(
    decision: PlanDifficultyAdjustmentDecision,
    original: dict[str, Any],
    proposed: dict[str, Any] | None,
    *,
    qtype: str,
    requested: str,
    evidence_ids: list[str],
) -> dict[str, bool]:
    assigned_evidence = set(filter(None, map(str, evidence_ids)))
    proposed = proposed or {}
    changed_requirements = {
        str(value).strip() for value in (
            (decision.adjusted_fields.reasoning_requirements or [])
            if decision.adjusted_fields else []
        ) if str(value).strip()
    }
    original_requirements = {
        str(value).strip() for value in (original.get("reasoning_requirements") or [])
        if str(value).strip()
    }
    proposed_requirements = {
        str(value).strip() for value in (proposed.get("reasoning_requirements") or [])
        if str(value).strip()
    }
    immutable_fields = (
        "slot_id", "base_catalog_task_id", "task_family_id",
        "base_task_family_id", "question_type", "concept", "task_operation",
        "target", "student_work_product", "source_chunk_id", "support_chunk_ids",
        "selected_variation_id", "selected_variation_group_id", "permitted_variations",
    )
    givens_changed = (
        list(proposed.get("given_conditions") or [])
        != list(original.get("given_conditions") or [])
        or list(proposed.get("required_givens") or [])
        != list(original.get("required_givens") or [])
    )
    selected = _selected_variation(original)
    numeric_change_authorized = numeric_variation_allowed(original)
    patch = decision.adjusted_fields
    allowed_transformations = {
        " ".join(str(value.get("transformation") or "").split()).casefold()
        for value in original.get("allowed_transformations") or []
        if isinstance(value, dict)
    }
    requested_transformations = {
        " ".join(str(value).split()).casefold()
        for value in (patch.cognitive_transformations if patch else []) or []
        if str(value).strip()
    }
    removed = {
        " ".join(str(value).split())
        for value in (patch.removed_requirements if patch else []) or []
        if str(value).strip()
    }
    transformation_mode = (
        (patch.adjustment_kind if patch else None)
        or ("cognitive" if requested_transformations else "numerical")
    )
    givens_changed_now = bool(givens_changed)
    no_change_valid = bool(
        decision.no_change_required
        and decision.adjustment_succeeded
        and patch is None
        and decision.achieved_difficulty == requested
        and not decision.attempted_fields
    )
    meaningful_change = bool(
        givens_changed_now
        or (changed_requirements and changed_requirements != original_requirements)
        or requested_transformations or removed
        or no_change_valid
    )
    checks = {
        "adjustment_succeeded": decision.adjustment_succeeded,
        "adjusted_fields_present": patch is not None or no_change_valid,
        "slot_id_matches": decision.slot_id == original.get("slot_id"),
        "immutable_contract_preserved": all(
            proposed.get(field) == original.get(field) for field in immutable_fields
        ),
        "effective_givens_authorized": not givens_changed_now or numeric_change_authorized,
        "transformations_authorized": requested_transformations.issubset(
            allowed_transformations
        ),
        "transformation_mode_consistent": (
            transformation_mode != "numerical" or not requested_transformations
        ) and (
            transformation_mode == "numerical"
            or bool(requested_transformations or removed)
        ),
        "meaningful_change_present": meaningful_change,
        "removed_requirements_present": removed.issubset(original_requirements),
        "changed_reasoning_requirements_applied": (
            not changed_requirements or changed_requirements == proposed_requirements
        ),
        "evidence_support_present": bool(decision.evidence_support),
        "evidence_support_assigned": set(map(str, decision.evidence_support)).issubset(
            assigned_evidence
        ),
    }
    if no_change_valid:
        checks["no_change_required"] = True
    return checks


def _rejection_reasons(checks: dict[str, bool]) -> list[str]:
    return [
        f"{name}=false" for name, passed in checks.items()
        if not passed and name != "no_change_required"
    ]


def _focused_task_input(
    task: dict[str, Any], qtype: str, actual: str, requested: str,
) -> dict[str, Any]:
    """Render the one compact task contract shared by both adjustment paths."""
    return {
        "immutable_identifiers": {
            "slot_id": task.get("slot_id"),
            "base_catalog_task_id": task.get("base_catalog_task_id") or task.get("task_id"),
            "task_family_id": task.get("task_family_id"),
        },
        "task_contract": {
            "question_type": qtype,
            "concept": task.get("concept"),
            "task_operation": task.get("task_operation"),
            "target": task.get("target"),
            "student_work_product": task.get("student_work_product"),
        },
        "current_givens_and_constraints": {
            "given_conditions": list(task.get("given_conditions") or []),
            "required_givens": list(task.get("required_givens") or []),
            "variable_givens": list(task.get("variable_givens") or []),
            "allowed_variations": list(task.get("allowed_variations") or []),
            "complexity_boundaries": list(task.get("complexity_boundaries") or []),
            "exact_source_instance": bool(task.get("exact_source_instance", False)),
            "numeric_variation_authorized": numeric_variation_allowed(task),
            "idea_to_test": task.get("idea_to_test"),
        },
        "allowed_transformations": [
            {
                "transformation": value.get("transformation"),
                "support": value.get("support"),
            }
            for value in task.get("allowed_transformations") or []
            if isinstance(value, dict)
        ],
        "reasoning_profile": {
            "reasoning_mode": task.get("reasoning_mode"),
            "reasoning_requirements": list(task.get("reasoning_requirements") or []),
            "prerequisites": list(task.get("prerequisites") or []),
        },
        "native_difficulty_prior": task.get("native_difficulty_prior") or task.get("native_difficulty"),
        "actual_difficulty": actual,
        "requested_difficulty": requested,
    }


def _selected_variation(task: dict[str, Any]) -> dict[str, Any]:
    selected_id = str(task.get("selected_variation_id") or "")
    return next((
        value for value in (task.get("permitted_variations") or [])
        if isinstance(value, dict) and str(value.get("variation_id") or "") == selected_id
    ), {})


def _execute_adjustment(work: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated adjustment call and return data for deterministic merging."""
    original = deepcopy(work["original"])
    qtype = str(work["qtype"])
    requested = str(work["requested"])
    actual = str(work["actual"])
    evidence_ids = list(work["evidence_ids"])
    prompt = json.dumps({
        "mode": "planned_task",
        "task": _focused_task_input(original, qtype, actual, requested),
        "assigned_evidence": deepcopy(work["evidence"]),
        "occupied_task_summaries": deepcopy(work["occupied_tasks"]),
        "difficulty_validator_findings": deepcopy(
            work.get("validator_findings") or []
        ),
        "recovery_instruction": work.get("recovery_instruction") or "",
    }, ensure_ascii=False)
    response_model = _response_model_for(original, qtype, requested, evidence_ids)
    decision: PlanDifficultyAdjustmentDecision | None = None
    failure = ""
    try:
        raw = request_structured(
            create_difficulty_adjustment_client(domain=str(work["domain"])), prompt,
            response_model=response_model,
            schema_name="difficulty_adjustment_plan", agent="difficulty_adjuster",
            operation="adjust_planned_task", item_count=1,
            system_prompt=SYSTEM_PROMPT, temperature=0.1, max_tokens=3500,
        )
        decision = response_model.model_validate(raw)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    proposed = None
    if decision and decision.adjustment_succeeded and decision.adjusted_fields:
        proposed = deepcopy(original)
        patch = decision.adjusted_fields.model_dump(exclude_none=True)
        patch.pop("strategy", None)
        adjustment_kind = patch.pop("adjustment_kind", None)
        cognitive = patch.pop("cognitive_transformations", None) or []
        removed = patch.pop("removed_requirements", None) or []
        proposed.update(patch)
        if cognitive:
            proposed["cognitive_transformations"] = list(cognitive)
        if removed:
            proposed["removed_requirements"] = list(removed)
        proposed["difficulty_design_proposal"] = {
            "adjustment_kind": adjustment_kind,
            "cognitive_transformations": list(cognitive),
            "removed_requirements": list(removed),
            "estimated_conceptual": decision.estimated_conceptual_difficulty
            or decision.conceptual_difficulty,
            "estimated_numerical": decision.estimated_numerical_difficulty
            or decision.numerical_difficulty,
            "estimated_actual": decision.achieved_difficulty,
            "requested": requested,
            "native_prior": original.get("native_difficulty_prior")
            or original.get("native_difficulty"),
            "reason": decision.reason,
            "strategy": (decision.adjusted_fields.strategy
                         if decision.adjusted_fields else None),
        }
    elif (
        decision and decision.adjustment_succeeded
        and decision.adjusted_fields is None
        and decision.no_change_required
        and decision.achieved_difficulty == requested
        and not decision.attempted_fields
    ):
        proposed = deepcopy(original)
    checks = (
        _acceptance_diagnostics(
            decision, original, proposed, qtype=qtype, requested=requested,
            evidence_ids=evidence_ids,
        )
        if decision else {"structured_response_valid": False}
    )
    return {
        **work,
        "decision": decision,
        "failure": failure,
        "proposed": proposed,
        "checks": checks,
        "rejection_reasons": _rejection_reasons(checks),
    }


def _execute_adjustment_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adjust up to three isolated tasks in one call and validate each independently."""
    if not batch or len(batch) > DIFFICULTY_ADJUSTMENT_BATCH_SIZE:
        raise ValueError("difficulty adjustment batches must contain one to three tasks")
    requested = str(batch[0]["requested"])
    payload = {
        "mode": "planned_task_batch",
        "instructions": (
            "Return one independent result for every supplied slot. A failure for one slot must "
            "not change or omit another slot's result. Use only that slot's assigned evidence IDs."
        ),
        "tasks": [{
            "task": _focused_task_input(
                work["original"], str(work["qtype"]), str(work["actual"]), requested,
            ),
            "assigned_evidence": deepcopy(work["evidence"]),
            "occupied_task_summaries": deepcopy(work["occupied_tasks"]),
        } for work in batch],
    }
    # Keep the legacy singular key for one-item batches and saved test harnesses.
    if len(batch) == 1:
        payload["task"] = payload["tasks"][0]["task"]
        payload["assigned_evidence"] = payload["tasks"][0]["assigned_evidence"]
        payload["occupied_task_summaries"] = payload["tasks"][0]["occupied_task_summaries"]
    prompt = json.dumps(payload, ensure_ascii=False)
    raw = request_structured(
        create_difficulty_adjustment_client(domain=str(batch[0]["domain"])), prompt,
        response_model=PlanDifficultyAdjustmentBatch,
        schema_name="difficulty_adjustment_plan_batch", agent="difficulty_adjuster",
        operation="adjust_planned_task_batch", item_count=len(batch),
        system_prompt=SYSTEM_PROMPT, temperature=0.1, max_tokens=3500 * len(batch),
    )
    if isinstance(raw, dict) and "results" not in raw and len(batch) == 1:
        raw = {"results": [raw]}
    parsed = PlanDifficultyAdjustmentBatch.model_validate(raw)
    by_slot: dict[str, PlanDifficultyAdjustmentDecision] = {}
    for decision in parsed.results:
        if decision.slot_id in by_slot:
            raise ValueError(f"duplicate adjustment result for slot {decision.slot_id}")
        by_slot[decision.slot_id] = decision
    expected = {str(work["slot_id"]) for work in batch}
    if set(by_slot) != expected:
        raise ValueError(
            f"adjustment result slot IDs must exactly match {sorted(expected)}; "
            f"received {sorted(by_slot)}"
        )
    results: list[dict[str, Any]] = []
    for work in batch:
        decision = by_slot[str(work["slot_id"])]
        original = deepcopy(work["original"])
        proposed = None
        if decision.adjustment_succeeded and decision.adjusted_fields:
            proposed = deepcopy(original)
            patch = decision.adjusted_fields.model_dump(exclude_none=True)
            patch.pop("strategy", None)
            adjustment_kind = patch.pop("adjustment_kind", None)
            cognitive = patch.pop("cognitive_transformations", None) or []
            removed = patch.pop("removed_requirements", None) or []
            proposed.update(patch)
            if cognitive:
                proposed["cognitive_transformations"] = list(cognitive)
            if removed:
                proposed["removed_requirements"] = list(removed)
            proposed["difficulty_design_proposal"] = {
                "adjustment_kind": adjustment_kind,
                "cognitive_transformations": list(cognitive),
                "removed_requirements": list(removed),
                "estimated_conceptual": decision.estimated_conceptual_difficulty
                or decision.conceptual_difficulty,
                "estimated_numerical": decision.estimated_numerical_difficulty
                or decision.numerical_difficulty,
                "estimated_actual": decision.achieved_difficulty,
                "requested": requested,
                "native_prior": original.get("native_difficulty_prior")
                or original.get("native_difficulty"),
                "reason": decision.reason,
                "strategy": (decision.adjusted_fields.strategy
                             if decision.adjusted_fields else None),
            }
        elif (
            decision.adjustment_succeeded
            and decision.adjusted_fields is None
            and decision.no_change_required
            and decision.achieved_difficulty == requested
            and not decision.attempted_fields
        ):
            proposed = deepcopy(original)
        checks = _acceptance_diagnostics(
            decision, original, proposed, qtype=str(work["qtype"]),
            requested=requested, evidence_ids=list(work["evidence_ids"]),
        )
        results.append({
            **work, "decision": decision, "failure": "", "proposed": proposed,
            "checks": checks, "rejection_reasons": _rejection_reasons(checks),
        })
    return results


def _apply_adjustment_result(
    item: dict[str, Any], result: dict[str, Any], history: list[dict[str, Any]],
) -> None:
    """Apply one completed call on the graph thread; workers never mutate shared state."""
    original = result["original"]
    requested = str(result["requested"])
    actual = str(result["actual"])
    attempt = int(result["attempt"])
    adaptation = deepcopy(result["adaptation"])
    decision = result.get("decision")
    proposed = result.get("proposed")
    checks = dict(result.get("checks") or {})
    rejection_reasons = list(result.get("rejection_reasons") or [])
    accepted = bool(
        decision and decision.adjustment_succeeded and proposed and not rejection_reasons
    )
    if accepted:
        if decision.no_change_required:
            # A no-op keeps the catalog task and its selected variation exactly as
            # assigned.  It is not a newly designed instance, so it must not gain
            # an effective-instance identity or design-only metadata.
            item.clear()
            item.update(deepcopy(original))
            item.update({
                "requested_difficulty": requested,
                "difficulty_fallback_reason": None,
                "adjustment_attempts": attempt,
                "adjustment_failure_reason": None,
                "difficulty_status": "not_required",
                "difficulty_design": {
                    **dict(original.get("difficulty_design") or {}),
                    "status": "not_required",
                    "policy_version": "difficulty-design-v1",
                    "requested": requested,
                    "native_prior": (
                        original.get("native_difficulty_prior")
                        or original.get("native_difficulty")
                        or actual
                    ),
                    "reason": decision.reason,
                    "evidence_support": list(decision.evidence_support),
                    "evidence_justification": decision.evidence_justification,
                    "reused": False,
                },
            })
            item.pop("effective_instance_id", None)
            item["derived_task_id"] = (
                original.get("base_catalog_task_id") or original.get("task_id")
            )
            item["effective_givens"] = []
            item["difficulty_design"].pop("effective_instance_id", None)
        else:
            proposed = deepcopy(proposed)
            proposal = deepcopy(proposed.get("difficulty_design_proposal") or {})
            effective_givens = list(proposed.get("given_conditions") or [])
            instance_payload = json.dumps({
                "task_id": original.get("base_catalog_task_id") or original.get("task_id"),
                "variation_id": original.get("selected_variation_id"),
                "givens": effective_givens,
                "requirements": list(proposed.get("reasoning_requirements") or []),
                "transformations": list(proposed.get("cognitive_transformations") or []),
                "requested": requested,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            effective_instance_id = "derived_" + hashlib.sha256(
                instance_payload.encode("utf-8")
            ).hexdigest()[:16]
            proposed["support_chunk_ids"] = list(original.get("support_chunk_ids") or [])
            proposed.update({
                "requested_difficulty": requested,
                "base_catalog_task_id": (
                    original.get("base_catalog_task_id") or original.get("task_id")
                ),
                "base_task_family_id": (
                    original.get("base_task_family_id") or original.get("task_family_id")
                ),
                "difficulty_fallback_reason": None,
                "adjustment_attempts": attempt,
                "adjustment_failure_reason": None,
                "derived_task_id": effective_instance_id,
                "effective_instance_id": effective_instance_id,
                "effective_givens": effective_givens,
                "difficulty_status": "designed",
                "difficulty_design": {
                    **dict(original.get("difficulty_design") or {}),
                    **proposal,
                    "status": "designed",
                    "policy_version": "difficulty-design-v1",
                    "effective_instance_id": effective_instance_id,
                    "evidence_support": list(decision.evidence_support),
                    "evidence_justification": decision.evidence_justification,
                    "reused": False,
                },
            })
            # actual_difficulty is intentionally NOT written here.  Luna observes it.
            item.clear()
            item.update(proposed)
    else:
        item.clear()
        item.update(deepcopy(original))
        exact_failure = (
            "; ".join(rejection_reasons)
            if decision else f"structured_response_invalid: {result.get('failure') or 'unknown error'}"
        )
        if decision and not decision.adjustment_succeeded:
            exact_failure = f"model_declined: {decision.reason}"
        item["requested_difficulty"] = requested
        item["adjustment_attempts"] = attempt
        item["adjustment_failure_reason"] = exact_failure
        item["difficulty_fallback_reason"] = exact_failure
        item["difficulty_status"] = "accepted_source_constraint"
        item["difficulty_design"] = {
            **dict(original.get("difficulty_design") or {}),
            "status": "accepted_source_constraint",
            "policy_version": "difficulty-design-v1",
            "requested": requested,
            "native_prior": (
                original.get("native_difficulty_prior")
                or original.get("native_difficulty")
                or actual
            ),
            "reason": exact_failure,
            "reused": False,
        }
        if decision and not decision.adjustment_succeeded:
            item["difficulty_design"].update({
                "achieved_difficulty": decision.achieved_difficulty,
                "further_adjustment_possible": decision.further_adjustment_possible,
                "adaptation_fingerprint": _adaptation_fingerprint(original, requested),
            })
    history.append({
        "slot_id": result["slot_id"],
        "attempt": attempt,
        "direction": _direction(actual, requested),
        "status": "succeeded" if accepted else "failed",
        "reason": decision.reason if decision else result.get("failure", ""),
        "model_decision": (
            "succeeded" if decision and decision.adjustment_succeeded
            else "declined" if decision else "invalid_response"
        ),
        "acceptance_checks": checks,
        "rejection_reasons": rejection_reasons,
        "achieved_difficulty": decision.achieved_difficulty if decision else actual,
        "conceptual_difficulty": decision.conceptual_difficulty if decision else None,
        "numerical_difficulty": decision.numerical_difficulty if decision else None,
        "further_adjustment_possible": (
            decision.further_adjustment_possible if decision else None
        ),
    })


def _apply_catalog_fallbacks(
    plans: list[dict[str, Any]], catalog: dict[str, Any], requested: str,
) -> None:
    """Deprecated no-op: difficulty never triggers task replacement in Group 2."""
    return None


def _legacy_apply_catalog_fallbacks(
    plans: list[dict[str, Any]], catalog: dict[str, Any], requested: str,
) -> None:
    """Replace exhausted adjustments deterministically, otherwise retain the genuine task."""
    if requested not in {"easy", "medium", "hard"}:
        return
    usage = build_task_usage_index(plans)
    tasks = list((catalog or {}).get("tasks") or [])
    profile = (catalog or {}).get("assessment_profile") or {}
    valid_chunks = set(map(str, (catalog or {}).get("selected_chunk_ids") or [])) or None
    for plan in plans:
        for qtype, values in (plan.get("items") or {}).items():
            for index, item in enumerate(values or []):
                adaptation = dict(item.get("difficulty_adaptation") or {})
                if adaptation.get("status") not in {"failed", "completed"}:
                    continue
                current_id = str(item.get("task_id") or "")
                compatible = [
                    (task, variation)
                    for task in tasks
                    if str(task.get("task_id") or "") != current_id
                    for variation in eligible_task_variation_candidates(
                        task, qtype, assessment_profile=profile,
                        valid_chunk_ids=valid_chunks,
                    )
                ]
                chosen = None
                chosen_variation = None
                chosen_level = None
                ranked = rank_task_variation_candidates(
                    [
                        (
                            task,
                            str(task.get("native_difficulty") or task.get("reasoning_depth") or "medium"),
                            variation,
                        )
                        for task, variation in compatible
                    ],
                    question_type=qtype,
                    requested_difficulty=requested,
                    task_usage=build_task_usage_index(
                        plans, excluding_slot_id=str(item.get("slot_id") or "")
                    ),
                )
                if ranked:
                    chosen, chosen_level, chosen_variation = ranked[0]
                if chosen is None:
                    apply_difficulty_fallback_metadata(
                        item, requested, str(item.get("actual_difficulty") or item.get("difficulty")),
                        str(item.get("difficulty_fallback_reason") or
                            "No further source-compatible adaptation is available."),
                        strategy="difficulty_adjustment_fallback",
                    )
                    item["requested_difficulty"] = requested
                    item["actual_difficulty"] = str(
                        item.get("actual_difficulty") or item.get("difficulty") or "medium"
                    )
                    item["difficulty"] = item["actual_difficulty"]
                    item["difficulty_fallback_reason"] = (
                        item.get("difficulty_fallback_reason")
                        or "No compatible catalog alternative remained after three adjustment attempts."
                    )
                    continue
                replacement = materialize_plan_item(
                    chosen, str(item.get("slot_id") or ""), chosen_level,
                    selected_variation_id=str(
                        chosen_variation.get("variation_id") or ""
                    ),
                )
                replacement.update({
                    "requested_difficulty": requested,
                    "selected_replacement_task_id": chosen.get("task_id"),
                    "adjustment_attempts": int(item.get("adjustment_attempts") or 0),
                    "adjustment_failure_reason": item.get("adjustment_failure_reason"),
                    "difficulty_adaptation": {
                        "status": "adapted", "from_difficulty": item.get("actual_difficulty"),
                        "to_difficulty": requested, "base_task_id": chosen.get("task_id"),
                        "strategy": "catalog_fallback", "failed_strategies": list(
                            adaptation.get("failed_strategies") or []
                        ),
                    },
                })
                if chosen_level == requested:
                    replacement["difficulty_status"] = "matched"
                    replacement["difficulty_fallback"] = False
                    replacement["difficulty_fallback_reason"] = None
                    replacement["difficulty_adaptation"]["status"] = "fallback"
                else:
                    apply_difficulty_fallback_metadata(
                        replacement, requested, str(chosen_level),
                        f"Adjustment exhausted; retained the closest supported {chosen_level} task.",
                        strategy="catalog_fallback",
                    )
                values[index] = replacement


def normalize_difficulty_designs(
    plans: list[dict[str, Any]], requested: str,
) -> None:
    """Migrate legacy adaptation state and synthesize the Group 2 contract.

    New jobs already carry difficulty_design. Legacy checkpoints and
    non-Catalog planner items do not; without this they would bypass design.
    """
    for plan in plans or []:
        for values in (plan.get("items") or {}).values():
            for item in values or []:
                if item.get("difficulty_design"):
                    continue
                adaptation = dict(item.get("difficulty_adaptation") or {})
                status = str(adaptation.get("status") or "")
                actual = str(
                    item.get("actual_difficulty")
                    or item.get("difficulty")
                    or adaptation.get("achieved_difficulty")
                    or "medium"
                )
                native = str(
                    adaptation.get("from_difficulty")
                    or item.get("native_difficulty_prior")
                    or item.get("native_difficulty")
                    or actual
                )
                if status in {"pending", "adapting"}:
                    design_status = "pending"
                elif status in {"adapted", "accepted_source_constraint", "completed"}:
                    design_status = "accepted_source_constraint"
                elif status in {"failed"}:
                    design_status = "accepted_source_constraint"
                else:
                    design_status = (
                        "pending"
                        if requested in {"easy", "medium", "hard"} and actual != requested
                        else "not_required"
                    )
                item["difficulty_design"] = {
                    "status": design_status,
                    "policy_version": "difficulty-design-v1",
                    "native_prior": native,
                    "requested": requested,
                    "reason": adaptation.get("failure_reason")
                    or item.get("difficulty_fallback_reason"),
                    "migrated_from_adaptation": bool(adaptation),
                }


def adjust_plan_difficulty(state: dict[str, Any]) -> dict[str, Any]:
    plans = deepcopy(state.get("plans") or [])
    requested = str(state.get("difficulty") or "mix")
    normalize_difficulty_designs(plans, requested)
    if requested == "mix":
        return {"plans": plans, "difficulty_adjustment_history": [],
                "difficulty_adjustment_attempts": {},
                "difficulty_adaptation_records": deepcopy(
                    state.get("difficulty_adaptation_records") or {}
                )}
    chunks = {str(c.get("child_id")): str(c.get("content") or "")
              for c in state.get("retrieved_chunks") or [] if c.get("child_id")}
    history = list(state.get("difficulty_adjustment_history") or [])
    attempts = dict(state.get("difficulty_adjustment_attempts") or {})
    records = deepcopy(state.get("difficulty_adaptation_records") or {})
    domain = str((state.get("assessment_profile") or {}).get("domain") or "")
    work_items: list[dict[str, Any]] = []
    for plan in plans:
        for qtype, values in (plan.get("items") or {}).items():
            for item in values or []:
                design = item.get("difficulty_design") or {}
                actual = str(item.get("actual_difficulty") or item.get("difficulty") or "medium")
                if design.get("status") != "pending":
                    continue
                sid = str(item.get("slot_id") or "")
                if int(attempts.get(sid) or 0) >= MAX_DIFFICULTY_ADJUSTMENT_ATTEMPTS:
                    item["difficulty_status"] = "accepted_source_constraint"
                    item["difficulty_design"] = {
                        **design,
                        "status": "accepted_source_constraint",
                        "reason": (
                            "Difficulty design exhausted after deterministic acceptance checks."
                        ),
                    }
                    continue
                attempt = int(attempts.get(sid) or 0) + 1
                # Reserve the attempt exactly once before dispatch. Concurrent workers never
                # read or mutate the shared counter.
                attempts[sid] = attempt
                evidence_ids = [str(item.get("source_chunk_id") or ""),
                                *map(str, item.get("support_chunk_ids") or [])]
                evidence = {key: chunks.get(key, "") for key in evidence_ids if key}
                original = deepcopy(item)
                work_items.append({
                    "slot_id": sid,
                    "target_item": item,
                    "original": original,
                    "adaptation": deepcopy(design),
                    "qtype": qtype,
                    "requested": requested,
                    "actual": actual,
                    "attempt": attempt,
                    "evidence_ids": evidence_ids,
                    "evidence": evidence,
                    "occupied_tasks": [{
                        "slot_id": other.get("slot_id"),
                        "task_family_id": other.get("task_family_id"),
                        "concept": other.get("concept"),
                        "task_operation": other.get("task_operation"),
                        "target": other.get("target"),
                        "actual_difficulty": (
                            other.get("actual_difficulty") or other.get("difficulty")
                        ),
                    } for other in values if (
                        other is not item and (
                            other.get("task_family_id") == item.get("task_family_id")
                            or other.get("concept") == item.get("concept")
                        )
                    )],
                    "domain": domain,
                })

    completed: list[dict[str, Any] | None] = [None] * len(work_items)
    if work_items:
        batches = [
            work_items[index:index + DIFFICULTY_ADJUSTMENT_BATCH_SIZE]
            for index in range(0, len(work_items), DIFFICULTY_ADJUSTMENT_BATCH_SIZE)
        ]
        worker_count = min(MAX_PARALLEL_DIFFICULTY_ADJUSTMENTS, len(batches), 10)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(copy_context().run, _execute_adjustment_batch, batch): batch_index
                for batch_index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_index = futures[future]
                start = batch_index * DIFFICULTY_ADJUSTMENT_BATCH_SIZE
                try:
                    for offset, result in enumerate(future.result()):
                        completed[start + offset] = result
                except Exception as exc:
                    for offset, work in enumerate(batches[batch_index]):
                        completed[start + offset] = {
                            **work, "decision": None,
                            "failure": f"{type(exc).__name__}: {exc}", "proposed": None,
                            "checks": {"structured_response_valid": False},
                            "rejection_reasons": ["structured_response_valid=false"],
                        }
    # Merge only on the graph thread and always in the original slot traversal order.
    for index, work in enumerate(work_items):
        result = completed[index]
        if result is None:
            result = {
                **work,
                "decision": None,
                "failure": "RuntimeError: adjustment did not return a result",
                "proposed": None,
                "checks": {"structured_response_valid": False},
                "rejection_reasons": ["structured_response_valid=false"],
            }
        _apply_adjustment_result(work["target_item"], result, history)
        adjusted = work["target_item"]
        design = deepcopy(adjusted.get("difficulty_design") or {})
        records[str(work["slot_id"])] = {
            "task_id": adjusted.get("base_catalog_task_id") or adjusted.get("task_id"),
            "slot_id": work["slot_id"],
            "variation_id": adjusted.get("selected_variation_id"),
            "assignment_fingerprint": adjusted.get("assignment_fingerprint"),
            "adaptation_fingerprint": _adaptation_fingerprint(work["original"], requested),
            "requested_difficulty": requested,
            "native_prior": design.get("native_prior") or work["actual"],
            "estimated_difficulty": design.get("estimated_actual"),
            "design_status": design.get("status"),
            "adjustment_kind": design.get("adjustment_kind"),
            "cognitive_transformations": list(
                design.get("cognitive_transformations") or []
            ),
            "adaptation_attempts": int(adjusted.get("adjustment_attempts") or 0),
            "reason": design.get("reason") or adjusted.get("difficulty_fallback_reason"),
            "achieved_difficulty": design.get("achieved_difficulty")
            or adjusted.get("actual_difficulty") or work["actual"],
            "further_adjustment_possible": design.get("further_adjustment_possible"),
            "difficulty_design": design,
            "policy_version": "difficulty-design-v1",
        }
    plans = finalize_plans(plans, state.get("tasks") or [])
    errors = validate_plans(plans, state.get("tasks") or [], int(state.get("num_models") or 1),
                            set(chunks), requested, state.get("assessment_profile") or {},
                            state.get("planning_issues") or [])
    verification_ids = [str(work["slot_id"]) for work in work_items]
    return {"plans": plans, "plan_errors": errors,
            "difficulty_adjustment_history": history,
            "difficulty_adjustment_attempts": attempts,
            "difficulty_adaptation_records": records,
            "difficulty_verification_slot_ids": verification_ids}


def adjust_failed_plan_slot(
    *,
    state: dict[str, Any],
    plans: list[dict[str, Any]],
    model_number: int,
    slot_id: str,
    question_type: str,
    findings: list[dict[str, Any]],
    attempt: int,
) -> tuple[bool, dict[str, Any]]:
    """Re-run Difficulty Adjustment for one validator-rejected assignment.

    This is deliberately assignment-level recovery.  It never edits the generated
    question and never traverses unrelated plan slots.  The regenerated question is
    independently checked by the normal full candidate-validation path.
    """
    target: dict[str, Any] | None = None
    siblings: list[dict[str, Any]] = []
    for plan in plans:
        if int(plan.get("model_number") or 0) != int(model_number):
            continue
        for qtype, values in (plan.get("items") or {}).items():
            if qtype == question_type:
                siblings = list(values or [])
            for item in values or []:
                if str(item.get("slot_id") or "") == slot_id:
                    target = item
                    question_type = str(qtype)
    if target is None:
        return False, {"status": "failed", "reason": "difficulty slot not found"}

    requested = str(
        target.get("requested_difficulty")
        or state.get("difficulty")
        or target.get("difficulty")
        or "medium"
    )
    current_fingerprint = _adaptation_fingerprint(target, requested)
    design = deepcopy(target.get("difficulty_design") or {})
    adaptation = deepcopy(target.get("difficulty_adaptation") or {})
    persisted = deepcopy(
        (state.get("difficulty_adaptation_records") or {}).get(slot_id) or {}
    )
    recorded_fingerprint = str(
        design.get("adaptation_fingerprint")
        or adaptation.get("adaptation_fingerprint")
        or persisted.get("adaptation_fingerprint")
        or ""
    )
    terminal_status = str(
        target.get("difficulty_status") or adaptation.get("status") or ""
    )
    terminal_decision = (
        design.get("further_adjustment_possible") is False
        or adaptation.get("further_adjustment_possible") is False
        or persisted.get("further_adjustment_possible") is False
    ) and terminal_status in {
        "accepted_source_constraint", "completed",
    }
    # Older checkpoints did not store the adaptation fingerprint.  An explicit
    # accepted fallback on the still-current plan item is authoritative; newer
    # records additionally require an exact dependency fingerprint match.
    if terminal_decision and (
        not recorded_fingerprint or recorded_fingerprint == current_fingerprint
    ):
        achieved = str(
            target.get("actual_difficulty")
            or persisted.get("achieved_difficulty")
            or target.get("difficulty")
            or "medium"
        )
        reason = str(
            target.get("difficulty_fallback_reason")
            or persisted.get("reason")
            or "The unchanged source-constrained assignment cannot be adapted further."
        )
        target["difficulty_status"] = "accepted_source_constraint"
        target["difficulty_design"] = {
            **design,
            "status": "accepted_source_constraint",
            "reason": reason,
            "native_prior": achieved,
            "achieved_difficulty": achieved,
            "further_adjustment_possible": False,
            "adaptation_fingerprint": current_fingerprint,
            "policy_version": "difficulty-design-v1",
        }
        return True, {
            "status": "reused_terminal_decision",
            "slot_id": slot_id,
            "attempt": attempt,
            "requested_difficulty": requested,
            "achieved_difficulty": achieved,
            "adaptation_fingerprint": current_fingerprint,
            "provider_call_made": False,
        }
    observed = next((
        str(value.get("observed_difficulty"))
        for value in findings
        if value.get("observed_difficulty")
    ), str(target.get("actual_difficulty") or target.get("difficulty") or "medium"))
    chunks = {
        str(chunk.get("child_id")): str(chunk.get("content") or "")
        for chunk in state.get("retrieved_chunks") or [] if chunk.get("child_id")
    }
    evidence_ids = [
        str(value) for value in (
            [target.get("source_chunk_id")]
            + list(target.get("support_chunk_ids") or [])
        ) if value
    ]
    original = deepcopy(target)
    adaptation.update({
        "status": "pending", "from_difficulty": observed,
        "to_difficulty": requested,
        "adaptation_fingerprint": current_fingerprint,
        "policy_version": DIFFICULTY_ADAPTATION_POLICY_VERSION,
    })
    work = {
        "slot_id": slot_id,
        "target_item": target,
        "original": original,
        "adaptation": adaptation,
        "qtype": question_type,
        "requested": requested,
        "actual": observed,
        "attempt": attempt,
        "evidence_ids": evidence_ids,
        "evidence": {key: chunks.get(key, "") for key in evidence_ids},
        "occupied_tasks": [
            {
                "slot_id": other.get("slot_id"),
                "task_family_id": other.get("task_family_id"),
                "concept": other.get("concept"),
                "task_operation": other.get("task_operation"),
                "target": other.get("target"),
                "actual_difficulty": other.get("actual_difficulty") or other.get("difficulty"),
            }
            for other in siblings if other is not target
        ],
        "domain": str((state.get("assessment_profile") or {}).get("domain") or ""),
        "validator_findings": deepcopy(findings),
        "recovery_instruction": (
            "Revise the real cognitive requirements for this one assignment. Preserve every "
            "mandatory operation and student work product. Do not merely change difficulty "
            "labels or shorten the answer. If the immutable task cannot meet the requested "
            "level, return adjustment_succeeded=false."
        ),
    }
    result = _execute_adjustment(work)
    audit: list[dict[str, Any]] = []
    _apply_adjustment_result(target, result, audit)
    accepted = (target.get("difficulty_design") or {}).get("status") in {
        "designed", "accepted_source_constraint",
    } or (target.get("difficulty_adaptation") or {}).get("status") == "adapted"
    decision = result.get("decision")
    if not accepted and decision is not None and decision.adjustment_succeeded is False:
        reason = str(
            target.get("difficulty_fallback_reason")
            or target.get("adjustment_failure_reason")
            or decision.reason
        )
        target["difficulty_status"] = "accepted_source_constraint"
        target["difficulty_design"] = {
            **dict(target.get("difficulty_design") or {}),
            "status": "accepted_source_constraint",
            "reason": reason,
            "native_prior": observed,
        }
        accepted = True
    return accepted, {
        "status": (
            "accepted_source_constraint"
            if target.get("difficulty_status") == "accepted_source_constraint"
            else "succeeded" if accepted else "failed"
        ),
        "slot_id": slot_id,
        "attempt": attempt,
        "requested_difficulty": requested,
        "observed_difficulty": observed,
        "findings": deepcopy(findings),
        "audit": audit,
        "rejection_reasons": list(result.get("rejection_reasons") or []),
        "failure": result.get("failure") or target.get("adjustment_failure_reason"),
        "adaptation_fingerprint": current_fingerprint,
        "provider_call_made": True,
    }


def adjust_generated_question(*, state: dict[str, Any], qtype: str, question: dict[str, Any],
                              plan: dict[str, Any], previous_questions: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    requested = str(plan.get("requested_difficulty") or state.get("difficulty") or "mix")
    actual = str(plan.get("actual_difficulty") or plan.get("difficulty") or requested)
    if requested == "mix" or actual == requested:
        return deepcopy(question), {"status": "not_needed", "actual_difficulty": actual}
    chunks = {str(c.get("child_id")): str(c.get("content") or "")
              for c in state.get("retrieved_chunks") or [] if c.get("child_id")}
    evidence_ids = [str(plan.get("source_chunk_id") or ""),
                    *map(str, plan.get("support_chunk_ids") or [])]
    evidence = {key: chunks.get(key, "") for key in evidence_ids if key and chunks.get(key)}
    slot_id = str(question.get("slot_id") or "")
    response_model = _generated_response_model_for(
        qtype, slot_id, requested, actual, list(evidence),
    )
    prompt = json.dumps({"mode": "generated_question", "question_type": qtype,
                         "requested_difficulty": requested, "actual_difficulty": actual,
                         "task": _focused_task_input(plan, qtype, actual, requested),
                         "question": question,
                         "assigned_evidence": evidence,
                         "negative_examples": previous_questions,
                         "required_output": {
                             "successful_adjustment": (
                                 "Return adjusted_question, the concrete changed reasoning requirements, "
                                 "and only exact assigned chunk IDs in evidence_support."
                             ),
                             "failed_adjustment": (
                                 "Return adjusted_question=null and explain why the original candidate "
                                 "cannot be adjusted safely."
                             ),
                         }}, ensure_ascii=False)
    try:
        raw = request_structured(
            create_difficulty_adjustment_client(domain=str((state.get("assessment_profile") or {}).get("domain") or "")),
            prompt, response_model=response_model,
            schema_name="difficulty_adjustment_question", agent="difficulty_adjuster",
            operation="adjust_generated_question", item_count=1,
            system_prompt=SYSTEM_PROMPT, temperature=0.1, max_tokens=3500,
        )
        decision = response_model.model_validate(raw)
        if not decision.adjustment_succeeded:
            return deepcopy(question), {
                "status": "failed", "actual_difficulty": actual,
                "from_difficulty": actual, "direction": _direction(actual, requested),
                "reason": f"model_declined_adjustment: {decision.reason}",
                "evidence_support": list(decision.evidence_support),
                "evidence_justification": decision.evidence_justification,
            }
        adjusted = decision.adjusted_question.model_dump()
        numbers_changed = (
            _question_instance_numbers(adjusted) != _question_instance_numbers(question)
        )
        selected = next((
            value for value in plan.get("permitted_variations") or []
            if value.get("variation_id") == plan.get("selected_variation_id")
        ), {})
        numeric_change_authorized = numeric_variation_allowed(plan)
        checks = {
            "slot_id_preserved": adjusted.get("slot_id") == slot_id,
            "actual_difficulty_matches": decision.actual_difficulty == requested,
            "changed_reasoning_requirements_present": bool(
                decision.changed_reasoning_requirements
            ),
            "evidence_support_present": bool(decision.evidence_support),
            "evidence_support_assigned": set(map(str, decision.evidence_support)).issubset(
                set(evidence)
            ),
            # Numerical instance changes are authorized and materialized at
            # plan adjustment time. A post-generation rewrite may deepen the
            # reasoning, but must retain that already-bound instance.
            "numeric_instance_authorized": not numbers_changed or numeric_change_authorized,
        }
        rejection_reasons = _rejection_reasons(checks)
        if rejection_reasons:
            return deepcopy(question), {
                "status": "failed", "actual_difficulty": actual,
                "from_difficulty": actual, "direction": _direction(actual, requested),
                "reason": "; ".join(rejection_reasons),
                "rejection_reasons": rejection_reasons,
                "acceptance_checks": checks,
            }
        adjusted["question_id"] = question.get("question_id")
        effective_givens = [str(adjusted.get(key)) for key in (
            "equation", "question", "statement", "context", "term"
        ) if adjusted.get(key)]
        effective_instance_id = None
        if numbers_changed:
            payload = json.dumps({
                "task_id": plan.get("base_catalog_task_id") or plan.get("task_id"),
                "variation_id": plan.get("selected_variation_id"),
                "question": adjusted,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            effective_instance_id = "derived_" + hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()[:16]
            adjusted["effective_instance_id"] = effective_instance_id
            adjusted["effective_givens"] = effective_givens
        for key in ("task_id", "base_catalog_task_id", "selected_variation_id",
                    "selected_variation_group_id"):
            if plan.get(key) is not None:
                adjusted[key] = plan.get(key)
        return adjusted, {
            "status": "succeeded", "actual_difficulty": requested,
            "from_difficulty": actual, "direction": _direction(actual, requested),
            "changed_reasoning_requirements": list(
                decision.changed_reasoning_requirements
            ),
            "evidence_support": list(decision.evidence_support),
            "evidence_justification": decision.evidence_justification,
            "reason": decision.reason,
            "acceptance_checks": checks,
            "effective_instance_id": effective_instance_id,
            "effective_givens": effective_givens if effective_instance_id else [],
            "verification_pending": True,
        }
    except Exception as exc:
        return deepcopy(question), {"status": "failed", "actual_difficulty": actual,
                                    "from_difficulty": actual,
                                    "direction": _direction(actual, requested),
                                    "reason": f"{type(exc).__name__}: {exc}"}
