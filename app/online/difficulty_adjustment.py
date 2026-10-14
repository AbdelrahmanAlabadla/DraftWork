from __future__ import annotations

import json
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

MAX_DIFFICULTY_ADJUSTMENT_ATTEMPTS = 3


class PlannedTaskAdjustmentPatch(StrictOutputModel):
    """Only fields the adjuster may propose changing on a planned task."""

    reasoning_requirements: list[NonEmptyString] | None = Field(default=None, max_length=8)
    prerequisites: list[NonEmptyString] | None = Field(default=None, max_length=8)
    given_conditions: list[NonEmptyString] | None = Field(default=None, max_length=12)
    required_givens: list[NonEmptyString] | None = Field(default=None, max_length=12)
    idea_to_test: NonEmptyString | None = Field(default=None, max_length=320)
    strategy: NonEmptyString | None = Field(default=None, max_length=320)


class PlanDifficultyAdjustmentDecision(StrictOutputModel):
    slot_id: NonEmptyString
    adjustment_succeeded: bool
    achieved_difficulty: Literal["easy", "medium", "hard"]
    adjusted_fields: PlannedTaskAdjustmentPatch | None
    evidence_support: list[NonEmptyString]
    evidence_justification: str
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision_shape(self):
        if self.adjustment_succeeded:
            if self.adjusted_fields is None:
                raise ValueError("adjusted_fields is required when adjustment_succeeded is true")
            if not self.adjusted_fields.reasoning_requirements:
                raise ValueError(
                    "adjusted_fields.reasoning_requirements is required when adjustment_succeeded is true"
                )
            if not self.evidence_support:
                raise ValueError("evidence_support is required when adjustment_succeeded is true")
        elif self.adjusted_fields is not None:
            raise ValueError("adjusted_fields must be null when adjustment_succeeded is false")
        return self


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


SYSTEM_PROMPT = """You are DraftWork's dedicated Difficulty Adjustment Agent. Adjust only difficulty.
Preserve concept, question type, base task family, evidence chunk IDs, and curriculum scope. Change
operational complexity and reasoning depth only when every added operation, relationship,
prerequisite, and reasoning requirement is supported by assigned evidence. reasoning_mode is the
kind of reasoning; reasoning_depth is its demand. Never use length, large numbers, or labels alone.
Preserve the question-type contract. Equation stays equation-centered valid LaTeX; Word Problem,
MCQ, True/False, Why, Definition, FITB, and Essay retain their forms. Return failure when adjustment
would require unsupported knowledge. You propose; Plan Reviewer and Validators independently decide.

For planned tasks, return only a controlled patch in adjusted_fields. Never reconstruct the task and
never return or modify IDs, question type, concept, core operation, target, work product, or source
boundaries. Use null for an allowed patch field that does not change. A successful patch must provide
the complete resulting reasoning_requirements, not merely a description of the change. Givens and
constraints may change only when the assigned evidence supports them and the core assessment remains
the same. `evidence_support` may contain only exact chunk-ID strings present as keys in
`assigned_evidence`; put prose support in `evidence_justification`. Do not add fields outside the schema.
"""


def _direction(actual: str, requested: str) -> str:
    return "none" if actual == requested else f"{actual}_to_{requested}"


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
        achieved_difficulty=(Literal.__getitem__((requested,)), ...),
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
    )
    return {
        "adjustment_succeeded": decision.adjustment_succeeded,
        "adjusted_fields_present": decision.adjusted_fields is not None,
        "slot_id_matches": decision.slot_id == original.get("slot_id"),
        "immutable_contract_preserved": all(
            proposed.get(field) == original.get(field) for field in immutable_fields
        ),
        "achieved_difficulty_matches": decision.achieved_difficulty == requested,
        "task_difficulty_matches": proposed.get("difficulty") == requested,
        "task_actual_difficulty_matches": proposed.get("actual_difficulty") == requested,
        "reasoning_depth_matches": proposed.get("reasoning_depth") == requested,
        "changed_reasoning_requirements_present": bool(changed_requirements),
        "reasoning_requirements_meaningfully_changed": bool(changed_requirements) and (
            changed_requirements != original_requirements
        ),
        "changed_reasoning_requirements_applied": bool(changed_requirements) and (
            changed_requirements == proposed_requirements
        ),
        "evidence_support_present": bool(decision.evidence_support),
        "evidence_support_assigned": set(map(str, decision.evidence_support)).issubset(
            assigned_evidence
        ),
    }


def _rejection_reasons(checks: dict[str, bool]) -> list[str]:
    return [f"{name}=false" for name, passed in checks.items() if not passed]


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
            "idea_to_test": task.get("idea_to_test"),
        },
        "reasoning_profile": {
            "reasoning_mode": task.get("reasoning_mode"),
            "reasoning_requirements": list(task.get("reasoning_requirements") or []),
            "prerequisites": list(task.get("prerequisites") or []),
        },
        "actual_difficulty": actual,
        "requested_difficulty": requested,
    }


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
        proposed.update(patch)
        proposed.update({
            "difficulty": requested,
            "actual_difficulty": requested,
            "reasoning_depth": requested,
        })
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
        proposed = deepcopy(proposed)
        proposed["support_chunk_ids"] = list(original.get("support_chunk_ids") or [])
        proposed.update({
            "requested_difficulty": requested,
            "actual_difficulty": requested,
            "difficulty": requested,
            "base_catalog_task_id": (
                original.get("base_catalog_task_id") or original.get("task_id")
            ),
            "base_task_family_id": (
                original.get("base_task_family_id") or original.get("task_family_id")
            ),
            "difficulty_fallback_reason": None,
            "adjustment_attempts": attempt,
            "adjustment_failure_reason": None,
            "difficulty_adaptation": {
                "status": "adapted",
                "from_difficulty": actual,
                "to_difficulty": requested,
                "base_task_id": (
                    original.get("base_catalog_task_id") or original.get("task_id")
                ),
                "strategy": (
                    decision.adjusted_fields.strategy
                    if decision.adjusted_fields else None
                ),
                "failed_strategies": [],
                "changed_reasoning_requirements": list(
                    decision.adjusted_fields.reasoning_requirements or []
                ) if decision.adjusted_fields else [],
                "evidence_support": list(decision.evidence_support),
                "evidence_justification": decision.evidence_justification,
            },
        })
        item.clear()
        item.update(proposed)
    else:
        item.clear()
        item.update(deepcopy(original))
        item["difficulty"] = actual
        item["actual_difficulty"] = actual
        item["requested_difficulty"] = requested
        item["adjustment_attempts"] = attempt
        exact_failure = (
            "; ".join(rejection_reasons)
            if decision else f"structured_response_invalid: {result.get('failure') or 'unknown error'}"
        )
        if decision and not decision.adjustment_succeeded:
            exact_failure = f"model_declined: {decision.reason}"
        item["adjustment_failure_reason"] = exact_failure
        item["difficulty_fallback_reason"] = exact_failure
        item["difficulty_adaptation"] = {**adaptation, "status": "failed"}
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
    })


def adjust_plan_difficulty(state: dict[str, Any]) -> dict[str, Any]:
    plans = deepcopy(state.get("plans") or [])
    requested = str(state.get("difficulty") or "mix")
    if requested == "mix":
        return {"plans": plans, "difficulty_adjustment_history": [],
                "difficulty_adjustment_attempts": {}}
    chunks = {str(c.get("child_id")): str(c.get("content") or "")
              for c in state.get("retrieved_chunks") or [] if c.get("child_id")}
    history = list(state.get("difficulty_adjustment_history") or [])
    attempts = dict(state.get("difficulty_adjustment_attempts") or {})
    domain = str((state.get("assessment_profile") or {}).get("domain") or "")
    work_items: list[dict[str, Any]] = []
    for plan in plans:
        for qtype, values in (plan.get("items") or {}).items():
            for item in values or []:
                adaptation = item.get("difficulty_adaptation") or {}
                actual = str(item.get("actual_difficulty") or item.get("difficulty") or "medium")
                if adaptation.get("status") != "pending":
                    continue
                sid = str(item.get("slot_id") or "")
                if int(attempts.get(sid) or 0) >= MAX_DIFFICULTY_ADJUSTMENT_ATTEMPTS:
                    item["difficulty_adaptation"] = {**adaptation, "status": "failed"}
                    item["difficulty_fallback_reason"] = (
                        "Difficulty adjustment exhausted after independent Plan Review."
                    )
                    item["adjustment_failure_reason"] = item["difficulty_fallback_reason"]
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
                    "adaptation": deepcopy(adaptation),
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
        worker_count = min(MAX_PARALLEL_DIFFICULTY_ADJUSTMENTS, len(work_items))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(copy_context().run, _execute_adjustment, work): index
                for index, work in enumerate(work_items)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    completed[index] = future.result()
                except Exception as exc:
                    completed[index] = {
                        **work_items[index],
                        "decision": None,
                        "failure": f"{type(exc).__name__}: {exc}",
                        "proposed": None,
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
    plans = finalize_plans(plans, state.get("tasks") or [])
    errors = validate_plans(plans, state.get("tasks") or [], int(state.get("num_models") or 1),
                            set(chunks), requested, state.get("assessment_profile") or {},
                            state.get("planning_issues") or [])
    return {"plans": plans, "plan_errors": errors,
            "difficulty_adjustment_history": history,
            "difficulty_adjustment_attempts": attempts}


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
            "verification_pending": True,
        }
    except Exception as exc:
        return deepcopy(question), {"status": "failed", "actual_difficulty": actual,
                                    "from_difficulty": actual,
                                    "direction": _direction(actual, requested),
                                    "reason": f"{type(exc).__name__}: {exc}"}
