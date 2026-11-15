from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.logging_conf import get_logger
from app.online.assessment_catalog import (
    catalog_exhaustion_replacement,
    compatible_task_inventory,
    fitb_answer_keys,
    is_authorized_catalog_exhaustion_reuse,
    eligible_task_variations,
    requires_unsupported_supplied_graph,
)
from app.online.duplicate_judge import (
    _jaccard,
    build_duplicate_groups,
    review_duplicate_candidates,
)
from app.online.diagnostics import record_optional_failure
from app.online.lifecycle import question_fingerprint, stable_fingerprint

logger = get_logger("DUPLICATE_PIPELINE")


def replacement_instance_key(candidate: dict[str, Any]) -> str:
    """Identify one replacement instance without condemning its whole task."""
    return "|".join((
        _identity(candidate),
        str(candidate.get("selected_variation_id") or ""),
        str(candidate.get("effective_instance_id") or ""),
        str(candidate.get("replacement_instance_sequence") or ""),
    ))


def replacement_feasibility_reasons(
    candidate: dict[str, Any], qtype: str, *,
    available_evidence_ids: set[str] | None = None,
    failed_instance_keys: set[str] | None = None,
) -> list[str]:
    """Return deterministic reasons a candidate must not reach Generation."""
    reasons: list[str] = []
    materialized = any(key in candidate for key in (
        "selected_variation_id", "catalog_replacement_stage",
        "source_evidence_ids", "source_chunk_id", "support_chunk_ids",
    ))
    if not _identity(candidate):
        reasons.append("missing_catalog_identity")
    if materialized and not str(
        candidate.get("task_operation") or candidate.get("operation") or ""
    ).strip():
        reasons.append("missing_task_operation")
    if materialized and not str(candidate.get("target") or "").strip():
        reasons.append("missing_target")
    if candidate.get("text_answerable") is False:
        reasons.append("not_text_answerable")
    if requires_unsupported_supplied_graph(candidate):
        reasons.append("unsupported_supplied_graph")
    declared = candidate.get("allowed_question_types") or candidate.get("primary_question_types")
    if declared and qtype not in declared:
        reasons.append("question_type_mismatch")
    evidence = {
        str(value) for value in (
            candidate.get("source_evidence_ids")
            or [candidate.get("source_chunk_id"), *(candidate.get("support_chunk_ids") or [])]
        ) if value
    }
    if materialized and not evidence:
        reasons.append("missing_source_evidence")
    elif available_evidence_ids is not None and not evidence.issubset(available_evidence_ids):
        reasons.append("source_evidence_unavailable")
    if replacement_instance_key(candidate) in (failed_instance_keys or set()):
        reasons.append("previously_rejected_instance")
    return reasons


def _build_question_fingerprint_index(exam: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for section in (exam.get("questions") or {}).values():
        values = section.get("items") if isinstance(section, dict) else section
        for item in values or []:
            qid = str(item.get("question_id") or "")
            if qid:
                result[qid] = question_fingerprint(item)
    return result


def _duplicate_fingerprint_index(
    exam: dict[str, Any], plan_by_slot: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Fingerprint content plus every assignment field used by screening."""
    result: dict[str, str] = {}
    for section in (exam.get("questions") or {}).values():
        values = section.get("items") if isinstance(section, dict) else section
        for item in values or []:
            qid = str(item.get("question_id") or "")
            if not qid:
                continue
            plan = plan_by_slot.get(str(item.get("slot_id") or "")) or {}
            result[qid] = stable_fingerprint({
                "question": question_fingerprint(item),
                "assignment": _compact_metadata(plan),
            })
    return result


def _current_conflict_groups(
    groups: list[dict[str, Any]], exams: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop stale endpoints and edges; groups are derived from current fingerprints."""
    current: list[dict[str, Any]] = []
    for raw in groups:
        group = deepcopy(raw)
        model_number = int(group.get("model_number") or 1)
        fingerprints = _build_question_fingerprint_index(exams.get(model_number) or {})
        recorded = group.get("question_fingerprints") or {}
        valid_ids = {
            str(qid) for qid in group.get("question_ids") or []
            if str(qid) in fingerprints
            and (not recorded or recorded.get(str(qid)) == fingerprints[str(qid)])
        }
        edges = [
            edge for edge in group.get("edges") or []
            if str(edge.get("left") or "") in valid_ids
            and str(edge.get("right") or "") in valid_ids
        ]
        connected_ids = {str(edge.get(side)) for edge in edges for side in ("left", "right")}
        if len(connected_ids) < 2:
            continue
        group["edges"] = edges
        group["question_ids"] = sorted(connected_ids)
        group["keep_question_ids"] = [
            qid for qid in group.get("keep_question_ids") or [] if qid in connected_ids
        ]
        group["replace_question_ids"] = [
            qid for qid in group.get("replace_question_ids") or [] if qid in connected_ids
        ]
        group["question_fingerprints"] = {
            qid: fingerprints[qid] for qid in connected_ids
        }
        current.append(group)
    return current


class ReplacementCandidateRejection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    reason: str = Field(max_length=240)


class DuplicateReplacementChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_to_replace: str
    conflicts_with: list[str]
    conflict_reason: str = Field(max_length=320)
    replacement_task_id: str | None = None
    replacement_fitb_entry_id: str | None = None
    why_this_candidate_resolves_the_conflict: str = Field(max_length=320)
    checked_against_active_slots: bool
    rejected_candidates: list[ReplacementCandidateRejection] = Field(default_factory=list)


def _exam_by_model(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(exam.get("model_number") or 1): exam for exam in state.get("generated_exams") or []}


def _correctness_pass_exam(state: dict[str, Any], exam: dict[str, Any]) -> dict[str, Any]:
    model_number = int(exam.get("model_number") or 1)
    report = next((
        value for value in state.get("validation_reports") or []
        if int(value.get("model_number") or 0) == model_number
    ), None)
    filtered = deepcopy(exam)
    terminal_slots = {
        str(item.get("slot_id") or "")
        for item in state.get("repair_manifest") or []
        if item.get("status") in {
            "exhausted", "terminal_failure", "validation_blocked",
        }
    }
    terminal_slots.update({
        str(slot_id)
        for slot_id, record in (state.get("slot_lifecycle") or {}).items()
        if record.get("current_status") in {"FAILED_TO_REPAIR", "VALIDATION_BLOCKED"}
    })
    if report is not None:
        passed = {
            str(verdict.get("question_id")) for verdict in report.get("verdicts") or []
            if verdict.get("action") == "PASS" and verdict.get("question_id")
        }
        for section in (filtered.get("questions") or {}).values():
            values = section.get("items") if isinstance(section, dict) else section
            if isinstance(values, list):
                values[:] = [
                    item for item in values
                    if str(item.get("question_id") or "") in passed
                    and str(item.get("slot_id") or "") not in terminal_slots
                ]
    _attach_reuse_context(filtered, state.get("plans") or [], model_number)
    return filtered


def _attach_reuse_context(
    exam: dict[str, Any], plans: list[dict[str, Any]], model_number: int,
) -> None:
    plan_by_slot = _plan_by_slot(plans, model_number)
    for section in (exam.get("questions") or {}).values():
        values = section.get("items") if isinstance(section, dict) else section
        for item in values or []:
            plan = plan_by_slot.get(str(item.get("slot_id") or "")) or {}
            authorized = bool(
                plan.get("reuse_mode") == "intentional_practice_variation"
                or is_authorized_catalog_exhaustion_reuse(plan)
            )
            if not authorized:
                continue
            item["reuse_context"] = {
                "status": "intentional_practice_variation",
                "authorized": True,
                "task_id": plan.get("task_id"),
                "task_family_id": plan.get("task_family_id"),
                "task_usage_count": plan.get("task_usage_count"),
                "reuse_mode": plan.get("reuse_mode"),
                "variation_reason": plan.get("variation_reason"),
                "intended_variation": plan.get("intended_variation"),
                "source_evidence_ids": list(plan.get("source_evidence_ids") or []),
                "suggested_assessment_angle": plan.get("suggested_assessment_angle"),
                "reuse_justification": plan.get("reuse_justification"),
                "required_instance_distinctions": deepcopy(
                    plan.get("required_instance_distinctions") or []
                ),
                "reference_slot_ids": list(plan.get("forced_duplicate_slot_ids") or []),
            }


def _plan_by_slot(plans: list[dict[str, Any]], model_number: int) -> dict[str, dict[str, Any]]:
    for plan in plans:
        if int(plan.get("model_number") or 0) == model_number:
            return {
                str(item.get("slot_id")): item
                for values in (plan.get("items") or {}).values()
                for item in values or [] if item.get("slot_id")
            }
    return {}


def _replace_plan_item(
    plans: list[dict[str, Any]], model_number: int, slot_id: str, replacement: dict[str, Any]
) -> bool:
    for plan in plans:
        if int(plan.get("model_number") or 0) != model_number:
            continue
        for values in (plan.get("items") or {}).values():
            for index, item in enumerate(values or []):
                if str(item.get("slot_id") or "") == slot_id:
                    replacement = deepcopy(replacement)
                    evidence = list(dict.fromkeys(str(value) for value in (
                        replacement.get("source_evidence_ids") or [
                            replacement.get("source_chunk_id"),
                            *(replacement.get("support_chunk_ids") or []),
                        ]
                    ) if value))
                    if evidence:
                        replacement["source_chunk_id"] = evidence[0]
                        replacement["support_chunk_ids"] = evidence[1:]
                        replacement["source_evidence_ids"] = evidence
                    values[index] = replacement
                    return True
    return False


def _remove_question(exam: dict[str, Any], question_id: str) -> dict[str, Any] | None:
    for qtype, section in (exam.get("questions") or {}).items():
        values = section.get("items") if isinstance(section, dict) else section
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if str(item.get("question_id") or "") == question_id:
                removed = values.pop(index)
                return {"question_type": qtype, "slot_id": removed.get("slot_id"), **removed}
    return None


def _question_slot(exam: dict[str, Any], question_id: str) -> str:
    for section in (exam.get("questions") or {}).values():
        values = section.get("items") if isinstance(section, dict) else section
        for item in values or []:
            if str(item.get("question_id") or "") == question_id:
                return str(item.get("slot_id") or "")
    return ""


_NUMERIC_VARIATION_FIELDS: dict[str, list[str]] = {
    "mcq": ["question", "options", "correct_answer", "solution_certificate"],
    "true_false": ["statement", "answer", "solution_certificate"],
    "equation": ["equation", "solution_steps", "final_answer", "solution_certificate"],
    "word_problem": ["question", "solution_steps", "final_answer", "solution_certificate"],
    "fill_in_the_blank": ["question", "distractors"],
    "definition": ["term", "reference_answer"],
    "why": ["question", "reference_answer"],
    "essay": ["question", "reference_answer", "key_points"],
}


_NUMERIC_CHARACTER_RE = re.compile(r"[0-9\u0660-\u0669\u06f0-\u06f9]")


def _numeric_variation_applicable(
    state: dict[str, Any], exam: dict[str, Any], model_number: int,
    question_id: str, qtype: str, item: dict[str, Any],
) -> bool:
    """Require both quantitative classification and mutable numeric content."""
    if qtype not in {"mcq", "true_false", "fill_in_the_blank", "equation", "word_problem"}:
        return False
    from app.online.validation.context import _plan_context_for_model, _quantitative_question_ids

    plan_by_slot, _source_by_chunk = _plan_context_for_model(state, model_number)
    plan = plan_by_slot.get(str(item.get("slot_id") or "")) or {}
    if plan.get("exact_source_instance") is True:
        return False
    selected_id = str(plan.get("selected_variation_id") or "")
    declared = [
        value for value in (
            plan.get("permitted_variations") or plan.get("allowed_variations") or []
        )
        if isinstance(value, dict)
    ]
    selected = next((
        value for value in declared
        if str(value.get("variation_id") or "") == selected_id
    ), None)
    if selected is None and not selected_id and len(declared) == 1:
        # Backward compatibility for persisted plans created before selected IDs
        # became mandatory; ambiguity still fails closed.
        selected = declared[0]
    from app.online.models import numeric_variation_allowed
    if not numeric_variation_allowed(plan):
        return False
    if question_id not in set(_quantitative_question_ids(exam, plan_by_slot)):
        return False
    relevant = {
        key: item.get(key) for key in (
            "question", "statement", "equation", "options", "correct_answer",
            "answer", "answers", "solution_steps", "final_answer", "distractors",
        ) if key in item
    }
    return bool(_NUMERIC_CHARACTER_RE.search(json.dumps(relevant, ensure_ascii=False)))


def _numeric_variation_fallback(
    state: dict[str, Any], exam: dict[str, Any], model_number: int,
    question_id: str, conflict_question_ids: list[str],
    conflict_edges: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Produce attempt three by changing dependent quantitative content only.

    The normal repair contract owns schema normalization, immutable question IDs,
    source evidence, and answer/certificate consistency.  This wrapper supplies
    both sides of the duplicate conflict and authorizes only question content.
    """
    from app.llm.factory import create_llm_client, create_review_client
    from app.online.validation.questions import _build_by_id
    from app.online.validation.context import (
        _plan_context_for_model, _quantitative_question_ids,
    )
    from app.online.validator import repair_exam

    isolated = deepcopy(exam)
    by_id = _build_by_id(isolated.get("questions") or {})
    entry = by_id.get(question_id)
    if entry is None:
        return None, [f"numeric fallback: question {question_id} was not found"]
    qtype, _container, item = entry
    conflicts = [
        deepcopy(by_id[qid][2]) for qid in conflict_question_ids if qid in by_id
    ]
    plan_by_slot, source_by_chunk = _plan_context_for_model(state, model_number)
    plan = plan_by_slot.get(str(item.get("slot_id") or "")) or {}
    fields = _NUMERIC_VARIATION_FIELDS.get(qtype, [])
    if not fields or not conflicts:
        return None, ["numeric fallback: no supported fields or conflict question"]
    details = {
        "mode": "numeric_variation_fallback",
        "duplicate_question": deepcopy(item),
        "conflicting_questions": conflicts,
        "previously_rejected_numerical_instance_fingerprints": [
            event.get("numerical_instance_fingerprint")
            for event in state.get("duplicate_repair_history") or []
            if event.get("slot_id") == item.get("slot_id")
            and event.get("numerical_instance_fingerprint")
        ],
        "duplicate_judge_conflict_details": deepcopy(conflict_edges),
        "preserve": {
            "question_id": question_id,
            "question_type": qtype,
            "difficulty": plan.get("actual_difficulty") or plan.get("difficulty"),
            "idea_to_test": plan.get("idea_to_test"),
            "reasoning_mode": plan.get("reasoning_mode"),
            "concept": plan.get("concept"),
            "task_operation": plan.get("task_operation"),
            "target": plan.get("target"),
            "required_givens": list(plan.get("required_givens") or []),
            "variable_givens": list(plan.get("variable_givens") or []),
            "allowed_variations": list(plan.get("allowed_variations") or []),
            "complexity_boundaries": list(plan.get("complexity_boundaries") or []),
            "expected_answer_type": plan.get("expected_work_product")
            or plan.get("student_work_product"),
            "evidence_chunk_ids": plan.get("support_chunk_ids")
            or plan.get("evidence_chunk_ids") or [plan.get("source_chunk_id")],
        },
    }
    verdict = {
        "question_id": question_id,
        "action": "FIX_QUESTION_AND_ANSWER",
        "failure_code": "duplicate_numeric_variation_required",
        "fields_to_fix": fields,
        "reason": json.dumps(details, ensure_ascii=False),
        "expected_fix": (
            "Compare both questions and change the values, coefficients, quantities, "
            "constants, or answer values that cause overlap. Recalculate every dependent "
            "answer, option, distractor, unit, explanation, solution step, and intermediate "
            "result. Preserve the listed semantic metadata and difficulty. Avoid blind "
            "number substitution, invalid or unrealistic values, unintended fractions, "
            "no/infinite-solution cases, and multiple-correct-answer MCQs."
            " Do not repeat a previously rejected numerical-instance fingerprint."
        ),
    }
    quantitative = question_id in set(_quantitative_question_ids(isolated, plan_by_slot))
    domain = str(
        (state.get("assessment_profile") or {}).get("domain")
        or state.get("domain_type") or "STEM"
    )
    client = (
        create_review_client(domain=domain, stage="stem_quantitative_repairer")
        if quantitative else create_llm_client(domain=domain, stage="general_repairer")
    )
    result = repair_exam(
        isolated, [verdict], client=client, plan_by_slot=plan_by_slot,
        source_by_chunk=source_by_chunk,
        repair_role="stem_quantitative" if quantitative else "general",
    )
    warnings = list(result.get("warnings") or [])
    if question_id not in set(result.get("repaired_ids") or []):
        return None, warnings or ["numeric fallback: repair was rejected"]
    repaired = (_build_by_id(isolated.get("questions") or {}).get(question_id) or (None, None, None))[2]
    return deepcopy(repaired) if isinstance(repaired, dict) else None, warnings


def _apply_effective_numeric_instance(
    plan: dict[str, Any], item: dict[str, Any], qtype: str,
) -> str:
    """Bind an authorized repaired instance without mutating Catalog identity."""
    content = {
        key: deepcopy(item.get(key))
        for key in _NUMERIC_VARIATION_FIELDS.get(qtype, []) if key in item
    }
    fingerprint = stable_fingerprint(content)
    effective_id = f"numeric_{fingerprint[:16]}"
    summary = json.dumps(content, ensure_ascii=False, sort_keys=True)
    effective_givens = list(plan.get("required_givens") or [])
    effective_givens.append(f"authorized numerical instance: {summary[:1200]}")
    plan.update({
        "effective_instance_id": effective_id,
        "effective_givens": effective_givens,
        "numerical_instance_fingerprint": fingerprint,
    })
    item.update({
        "effective_instance_id": effective_id,
        "numerical_instance_fingerprint": fingerprint,
    })
    return fingerprint


def _active_slot_ids(exam: dict[str, Any]) -> set[str]:
    return {
        str(item.get("slot_id"))
        for section in (exam.get("questions") or {}).values()
        for item in (section.get("items") if isinstance(section, dict) else section) or []
        if item.get("slot_id")
    }


def _slot_attempt_key(model_number: int, slot_id: str) -> str:
    return f"{model_number}:{slot_id}"


def _failed_task_exclusions(
    history: list[dict[str, Any]], model_number: int, slot_id: str,
) -> set[str]:
    """Exclude only tasks explicitly proven unusable, never one bad instance."""
    return {
        str(event.get("fitb_entry_id") or event.get("task_id") or "")
        for event in history
        if int(event.get("model_number") or 0) == model_number
        and str(event.get("slot_id") or "") == slot_id
        and event.get("task_wide") is True
        and event.get("status") in {
            "generation_failed", "fallback_precommit_rejected", "validation_failed",
            "quantitative_validation_failed", "duplicate_after_replacement",
        }
        and (event.get("fitb_entry_id") or event.get("task_id"))
    }


def _identity(item: dict[str, Any]) -> str:
    return str(item.get("fitb_entry_id") or item.get("task_id") or "")


def _compact_metadata(item: dict[str, Any]) -> dict[str, Any]:
    compact = {key: item.get(key) for key in (
        "slot_id", "task_id", "fitb_entry_id", "task_family_id", "coverage_id",
        "concept", "concept_id", "task_operation", "operation", "target",
        "reasoning_mode", "expected_work_product", "relationship_id",
        "normalized_answer", "clue_basis", "evidence_chunk_id", "source_chunk_id",
        "reasoning_requirements", "prerequisites", "student_work_product",
        "required_givens", "variable_givens", "allowed_variations",
        "complexity_boundaries", "exact_source_instance",
        "difficulty", "actual_difficulty", "requested_difficulty", "reuse_mode",
        "variation_reason", "intended_variation", "source_evidence_ids",
    ) if item.get(key) not in (None, "", [])}
    work_product = item.get("expected_work_product") or item.get("student_work_product")
    if work_product:
        compact["expected_work_product"] = work_product
    return compact


def _complete_question_context(
    question_id: str, item: dict[str, Any], plan: dict[str, Any], *, role: str,
) -> dict[str, Any]:
    return {
        "context_role": role,
        "question_id": question_id,
        "slot_id": item.get("slot_id") or plan.get("slot_id"),
        "task_id": plan.get("task_id"),
        "fitb_entry_id": plan.get("fitb_entry_id"),
        "task_family_id": plan.get("task_family_id"),
        "question_type": plan.get("question_type"),
        "question": item.get("question") or item.get("statement") or item.get("term")
        or item.get("equation") or item.get("prompt"),
        "options": deepcopy(item.get("options")),
        "correct_answer": (
            item.get("correct_answer") if item.get("correct_answer") is not None
            else item.get("answer") if item.get("answer") is not None
            else item.get("answers") or item.get("reference_answer") or item.get("final_answer")
        ),
        "solution_steps": deepcopy(item.get("solution_steps") or []),
        "solution_certificate": deepcopy(item.get("solution_certificate")),
        "idea_to_test": plan.get("idea_to_test"),
        "task_operation": plan.get("task_operation"),
        "target": plan.get("target"),
        "selected_variation_group_id": plan.get("selected_variation_group_id"),
        "effective_instance_id": plan.get("effective_instance_id"),
        "effective_givens": deepcopy(plan.get("effective_givens") or []),
        "reasoning_requirements": deepcopy(plan.get("reasoning_requirements") or []),
        "student_work_product": plan.get("student_work_product"),
        "required_givens": deepcopy(plan.get("required_givens") or []),
        "variable_givens": deepcopy(plan.get("variable_givens") or []),
        "allowed_variations": deepcopy(plan.get("allowed_variations") or []),
        "complexity_boundaries": deepcopy(plan.get("complexity_boundaries") or []),
        "exact_source_instance": bool(plan.get("exact_source_instance", False)),
        "source_evidence_ids": list(plan.get("source_evidence_ids") or [
            value for value in [plan.get("source_chunk_id"), *(plan.get("support_chunk_ids") or [])]
            if value
        ]),
    }


def _selector_question_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Keep only duplicate/reuse signals needed by replacement selection."""
    return {
        key: deepcopy(value.get(key))
        for key in (
            "context_role", "question_id", "slot_id", "task_id", "fitb_entry_id",
            "task_family_id", "question_type", "idea_to_test", "task_operation",
            "target", "selected_variation_group_id", "effective_instance_id",
            "effective_givens", "reasoning_requirements", "student_work_product",
            "required_givens", "variable_givens", "allowed_variations",
            "complexity_boundaries", "exact_source_instance", "source_evidence_ids",
            "question", "options", "correct_answer",
        )
        if value.get(key) not in (None, "", [])
    }


def _selector_failure_context(value: dict[str, Any]) -> dict[str, Any]:
    plan = value.get("plan") or {}
    compact = {
        key: deepcopy(value.get(key))
        for key in (
            "slot_id", "question_id", "question_type", "original_task_id",
            "failure_code", "failure_reason", "required_difficulty", "attempt",
        )
        if value.get(key) not in (None, "", [])
    }
    if plan:
        compact["current_assignment"] = _compact_metadata(plan)
    existing = value.get("existing_question")
    if isinstance(existing, dict):
        compact["current_question_metadata"] = _selector_question_metadata(existing)
    return compact


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _semantic_tokens(value: Any) -> set[str]:
    if isinstance(value, list):
        value = " ".join(map(str, value))
    return {
        token for token in re.findall(r"[^\W_]+", _norm(value), flags=re.UNICODE)
        if len(token) > 1
    }


def _token_overlap_score(left: Any, right: Any) -> float:
    left_tokens, right_tokens = _semantic_tokens(left), _semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _selector_semantic_relation_score(
    candidate: dict[str, Any], active: dict[str, Any],
) -> float:
    """Rank possible semantic conflicts using language-neutral structured signals.

    This score only selects compact LLM context. It never rejects a candidate or
    replaces Duplicate V2's actual-question judgment.
    """
    score = 0.0
    for field, weight in (
        ("task_family_id", 9.0), ("coverage_id", 8.0), ("concept_id", 8.0),
        ("relationship_id", 7.0), ("selected_variation_group_id", 7.0),
        ("canonical_skill", 6.0), ("task_operation", 6.0),
        ("reasoning_mode", 2.0),
    ):
        left, right = _norm(candidate.get(field)), _norm(active.get(field))
        if left and left == right:
            score += weight
    for field, weight in (
        ("concept", 4.0), ("target", 4.0), ("student_work_product", 3.0),
        ("idea_to_test", 3.0), ("relationship", 3.0),
        ("reasoning_requirements", 2.0),
    ):
        score += weight * _token_overlap_score(candidate.get(field), active.get(field))
    return score


def _selector_relevant_active_items(
    candidates: list[dict[str, Any]], active_items: list[dict[str, Any]],
    conflict_slots: set[str], *, limit: int = 12,
) -> list[dict[str, Any]]:
    """Return explicit conflicts plus a bounded semantic-context shortlist."""
    explicit = [
        item for item in active_items
        if str(item.get("slot_id") or "") in conflict_slots
    ]
    explicit_slots = {str(item.get("slot_id") or "") for item in explicit}
    ranked = sorted(
        (
            (max((_selector_semantic_relation_score(candidate, item)
                  for candidate in candidates), default=0.0), item)
            for item in active_items
            if str(item.get("slot_id") or "") not in explicit_slots
        ),
        key=lambda entry: (-entry[0], str(entry[1].get("slot_id") or "")),
    )
    positively_related = [item for score, item in ranked if score > 0]
    # If metadata has no usable relation signal, retain a small conservative
    # sample instead of silently claiming there can be no semantic conflict.
    fallback = [item for _score, item in ranked[: min(6, limit)]]
    related = positively_related or fallback
    room = max(0, limit - len(explicit))
    return [*explicit, *related[:room]]


def _obvious_semantic_collision(candidate: dict[str, Any], active: dict[str, Any]) -> bool:
    if candidate.get("fitb_entry_id"):
        return bool(_norm(candidate.get("normalized_answer") or candidate.get("fitb_answer"))
                    and _norm(candidate.get("normalized_answer") or candidate.get("fitb_answer")) ==
                    _norm(active.get("normalized_answer") or active.get("fitb_answer")))
    values = [
        (_norm(candidate.get("concept") or candidate.get("concept_id")),
         _norm(active.get("concept") or active.get("concept_id"))),
        (_norm(candidate.get("task_operation") or candidate.get("operation")),
         _norm(active.get("task_operation") or active.get("operation"))),
        (_norm(candidate.get("target")), _norm(active.get("target"))),
    ]
    return all(left and left == right for left, right in values)


def _conflict_resolution_score(
    candidate: dict[str, Any], conflicting: list[dict[str, Any]],
) -> tuple[bool, int]:
    """Return whether metadata proves a conflict is resolved and its strength.

    Missing semantic metadata is ambiguous and deliberately falls back to the
    existing LLM selector. A deterministic choice is made only when every
    conflicting slot has an explicit difference in assessed work.
    """
    if not conflicting:
        return False, 0
    fields = (
        ("concept", "concept_id"),
        ("target",),
        ("task_operation", "operation"),
        ("reasoning_mode",),
        ("expected_work_product", "student_work_product"),
    )
    total = 0
    for active in conflicting:
        differences = 0
        comparable = 0
        for aliases in fields:
            left = next((_norm(candidate.get(name)) for name in aliases
                         if _norm(candidate.get(name))), "")
            right = next((_norm(active.get(name)) for name in aliases
                          if _norm(active.get(name))), "")
            if left and right:
                comparable += 1
                differences += int(left != right)
        if comparable == 0 or differences == 0:
            return False, total
        total += differences
    return True, total


def _candidate_pool(
    catalog: dict[str, Any], plans: list[dict[str, Any]], slot_id: str, qtype: str,
    difficulty: str, active_slots: set[str], excluded: set[str], limit: int | None = None,
    *, prefer_task_id: str | None = None,
    exclude_variation_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    limit = max(1, int(limit or len(catalog.get("tasks") or []) or 1))
    candidates: list[dict[str, Any]] = []
    seen = set(excluded)
    for _ in range(limit):
        candidate = catalog_exhaustion_replacement(
            catalog, plans, slot_id, qtype, difficulty,
            excluded_task_ids=seen, active_slot_ids=active_slots,
            prefer_task_id=prefer_task_id,
            exclude_variation_ids=exclude_variation_ids,
        )
        if not candidate or not _identity(candidate) or _identity(candidate) in seen:
            break
        candidates.append(candidate)
        seen.add(_identity(candidate))
    # Replacement needs the next honest tier as a candidate inventory, even
    # when some requested-difficulty tasks exist but are later rejected by the
    # semantic selector. This does not affect initial Planner assignment.
    if difficulty in {"easy", "medium", "hard"} and len(candidates) < limit:
        for _ in range(limit - len(candidates)):
            candidate = catalog_exhaustion_replacement(
                catalog, plans, slot_id, qtype, "mix",
                excluded_task_ids=seen, active_slot_ids=active_slots,
                prefer_task_id=prefer_task_id,
                exclude_variation_ids=exclude_variation_ids,
            )
            if not candidate or not _identity(candidate) or _identity(candidate) in seen:
                break
            candidate["requested_difficulty"] = difficulty
            candidate["actual_difficulty"] = candidate.get("difficulty")
            candidate["selected_replacement_task_id"] = candidate.get("task_id")
            candidates.append(candidate)
            seen.add(_identity(candidate))
    return candidates


def select_preplanned_backup(
    *, slot_id: str, qtype: str, candidates: list[dict[str, Any]],
    backups: list[dict[str, Any]], blacklist: set[str] | None = None,
    active_items: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Compatibility shim: preplanned backups may no longer bypass shared ranking."""
    return None, {"selection_source": "preplanned_backup_disabled_shared_rank_required"}


def _select_replacement_candidate(
    *, slot_id: str, qtype: str, conflict_slots: list[str], conflict_edges: list[dict[str, Any]],
    candidates: list[dict[str, Any]], active_items: list[dict[str, Any]],
    blacklist: set[str], previous_failures: list[dict[str, Any]],
    failed_slot_context: dict[str, Any] | None = None,
    existing_questions: list[dict[str, Any]] | None = None,
    allow_semantic_overlap_candidates: bool = False,
    selector_operation: str = "catalog_replacement_selector",
    selector_page: int = 0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    active_without_slot = [item for item in active_items if str(item.get("slot_id")) != slot_id]
    usage: dict[str, int] = {}
    for item in active_without_slot:
        usage[_identity(item)] = usage.get(_identity(item), 0) + 1
    conflicting_ids = {
        _identity(item) for item in active_without_slot
        if str(item.get("slot_id") or "") in conflict_slots
    }
    failed_candidate_ids = {
        str(failure.get("candidate_id") or failure.get("task_id") or
            failure.get("fitb_entry_id") or "")
        for failure in previous_failures
        if str(failure.get("candidate_id") or failure.get("task_id") or
               failure.get("fitb_entry_id") or "")
    }
    failed_instance_keys = {
        str(failure.get("candidate_instance_key") or "")
        for failure in previous_failures if failure.get("candidate_instance_key")
    }

    def question_type_compatible(candidate: dict[str, Any]) -> bool:
        if qtype == "fill_in_the_blank":
            return bool(candidate.get("fitb_entry_id"))
        declared = candidate.get("allowed_question_types")
        # Legacy persisted plans may predate this compiler field. Keep their
        # existing candidate behavior, but never accept an explicit mismatch.
        if not declared:
            return True
        if qtype not in declared:
            return False
        if "text_answerable" not in candidate and "allowed_variations" not in candidate:
            return True
        return bool(eligible_task_variations(candidate, qtype))

    def authorized_reuse(candidate: dict[str, Any]) -> bool:
        return bool(
            candidate.get("task_reused") or candidate.get("fitb_entry_reused")
            or candidate.get("reuse_mode") == "intentional_practice_variation"
        )

    feasibility_rejections = {
        replacement_instance_key(candidate): replacement_feasibility_reasons(
            candidate, qtype, failed_instance_keys=failed_instance_keys,
        )
        for candidate in candidates
    }
    safe = [candidate for candidate in candidates
            if _identity(candidate) not in blacklist
            and not (
                _identity(candidate) in failed_candidate_ids
                and not any(
                    str(failure.get("candidate_id") or failure.get("task_id") or
                        failure.get("fitb_entry_id") or "") == _identity(candidate)
                    and failure.get("candidate_instance_key")
                    for failure in previous_failures
                )
            )
            and not feasibility_rejections[replacement_instance_key(candidate)]
            and question_type_compatible(candidate)
            and (_identity(candidate) not in conflicting_ids or authorized_reuse(candidate))]
    failure_code = _norm((failed_slot_context or {}).get("failure_code"))
    requested_difficulty = _norm(
        (failed_slot_context or {}).get("required_difficulty")
    )

    def difficulty_candidate_eligible(candidate: dict[str, Any]) -> bool:
        actual = _norm(
            candidate.get("actual_difficulty") or candidate.get("difficulty")
        )
        if requested_difficulty == "easy":
            return actual in {"", "easy", "medium"}
        return True

    safe = [candidate for candidate in safe if difficulty_candidate_eligible(candidate)]

    def difficulty_rank(candidate: dict[str, Any]) -> int:
        actual = _norm(
            candidate.get("actual_difficulty") or candidate.get("difficulty")
        )
        if not requested_difficulty or requested_difficulty == "mix":
            return 0
        if actual == requested_difficulty:
            return 0
        if requested_difficulty == "easy" and actual == "medium":
            return 1
        return 2

    def failure_relevance(candidate: dict[str, Any]) -> int:
        if "difficulty" in failure_code:
            requested = _norm((failed_slot_context or {}).get("required_difficulty"))
            actual = _norm(candidate.get("actual_difficulty") or candidate.get("difficulty"))
            return int(bool(requested) and actual != requested)
        if any(marker in failure_code for marker in ("ground", "evidence", "unsupported")):
            evidence = candidate.get("source_evidence_ids") or [
                candidate.get("source_chunk_id"), *(candidate.get("support_chunk_ids") or []),
            ]
            return int(not any(evidence))
        return 0

    def rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
        collision_count = sum(_obvious_semantic_collision(candidate, item) for item in active_without_slot)
        represented = lambda key: sum(_norm(candidate.get(key)) == _norm(item.get(key))
                                      for item in active_without_slot if candidate.get(key))
        return (difficulty_rank(candidate), failure_relevance(candidate), collision_count,
                int(usage.get(_identity(candidate), 0) > 0),
                usage.get(_identity(candidate), 0), represented("coverage_id"),
                represented("concept_id"), represented("task_operation"), represented("target"),
                represented("task_family_id"), represented("reasoning_mode"), _identity(candidate))
    safe.sort(key=rank)
    conflicting_items = [
        item for item in active_without_slot
        if str(item.get("slot_id") or "") in conflict_slots
    ]
    # Metadata differences are supporting evidence only. The selector or the
    # existing Duplicate Judge decides whether a conflict is actually resolved.
    deterministic: list[tuple[dict[str, Any], int]] = []
    deterministic.sort(key=lambda value: (
        int(usage.get(_identity(value[0]), 0) > 0),
        usage.get(_identity(value[0]), 0),
        -value[1],
        rank(value[0]),
    ))
    window = safe[:5]
    relevant_slots = set(conflict_slots)
    prompt_active_items = _selector_relevant_active_items(
        window, active_without_slot, relevant_slots,
    )
    prompt_slots = {
        str(item.get("slot_id") or "") for item in prompt_active_items
    }
    prompt_questions = [
        question for question in (existing_questions or [])
        if str(question.get("slot_id") or "") in prompt_slots
    ]
    compact_questions = [
        _selector_question_metadata(question)
        for question in prompt_questions
    ]
    compact_failures = [{
        key: deepcopy(failure.get(key))
        for key in ("candidate_id", "task_id", "fitb_entry_id", "reason", "stage", "attempt")
        if failure.get(key) not in (None, "", [])
    } for failure in previous_failures]
    context = {
        "slot_to_replace": slot_id, "conflicts_with": conflict_slots,
        "conflict": conflict_edges,
        "active_slots": [_compact_metadata(item) for item in prompt_active_items],
        "failed_slot": _selector_failure_context(failed_slot_context or {}),
        "existing_questions": compact_questions,
        "candidates": [{
            **_compact_metadata(item),
            "candidate_id": _identity(item),
            "requested_difficulty": requested_difficulty or "mix",
            "candidate_difficulty": _norm(
                item.get("actual_difficulty") or item.get("difficulty")
            ) or "unknown",
            "difficulty_tier": (
                "requested" if difficulty_rank(item) == 0 else "medium_fallback"
            ),
            "fallback_allowed": bool(
                requested_difficulty == "easy"
                and _norm(item.get("actual_difficulty") or item.get("difficulty"))
                == "medium"
            ),
            "difficulty_candidate_status": (
                "requested_match" if difficulty_rank(item) == 0
                else "source_constrained_medium_fallback_candidate"
            ),
            "used_count": usage.get(_identity(item), 0),
            "ranking": index + 1,
            "previous_uses": [
                _selector_question_metadata(question) for question in (existing_questions or [])
                if str(question.get("task_id") or question.get("fitb_entry_id") or "")
                == _identity(item)
            ][:4],
        } for index, item in enumerate(window)],
        "eligible_candidate_count": len(safe),
        "candidate_window_size": len(window),
        "candidate_page": selector_page + 1,
        "blacklisted_candidate_ids": sorted(blacklist),
        "previously_failed_candidate_ids": sorted(failed_candidate_ids),
        "previously_failed_candidate_instances": sorted(failed_instance_keys),
        "deterministic_feasibility_rejections": [
            {"candidate_instance_key": key, "reasons": reasons}
            for key, reasons in feasibility_rejections.items() if reasons
        ],
        "previous_failures": compact_failures,
    }
    component_chars = {
        key: len(json.dumps(value, ensure_ascii=False, default=str))
        for key, value in context.items()
    }
    logger.info(
        "catalog replacement selector context",
        extra={
            "event": "replacement_selector_context",
            "operation": selector_operation,
            "candidate_count": len(window),
            "context_component_chars": component_chars,
            "approx_prompt_tokens": max(1, sum(component_chars.values()) // 4),
        },
    )
    if not window:
        return None, context
    if deterministic:
        chosen, strength = deterministic[0]
        return chosen, {
            **context,
            "selection_source": "deterministic_conflict_ranking",
            "conflict_resolution_strength": strength,
        }
    try:
        from app.llm.factory import create_duplicate_replacement_client
        from app.llm.client import request_structured
        raw = request_structured(
            create_duplicate_replacement_client(),
            "Choose the first semantically suitable candidate from this Python-ranked batch. Prefer "
            "unused candidates, then least-used candidates. A previously used task is allowed only "
            "when its actual previous questions support a distinct source-grounded representation, "
            "procedure, or numerical instance. Wording-only variation is unsuitable. Compare the "
            "failed slot, failure reason, all previous uses, same-family context, and active slots. "
            "Reject candidates that cannot produce one unambiguous source-grounded answer, require "
            "invented givens, or cannot honestly represent its actual difficulty. Prefer an "
            "honest requested-difficulty candidate. For an Easy request, after no suitable Easy "
            "candidate remains, a source-grounded Medium candidate may proceed as a fallback "
            "candidate without a pre-existing fallback approval; it is still subject to full "
            "generation and validation. Treat fallback_allowed=true as explicit permission to "
            "select the candidate for Generation without prior accepted_source_constraint "
            "metadata; it is not final acceptance. Never substitute Hard for Easy. Keep the "
            "selection reason and rejection reasons to one concise sentence each. "
            "Return JSON matching the schema.\n" + json.dumps(context, ensure_ascii=False),
            response_model=DuplicateReplacementChoice, schema_name="duplicate_replacement_choice",
            agent=selector_operation, operation=selector_operation,
            item_count=len(window),
            temperature=0.0, max_tokens=5000,
        )
        chosen_id = str((raw or {}).get("replacement_fitb_entry_id") or
                        (raw or {}).get("replacement_task_id") or "")
        chosen = next((item for item in window if _identity(item) == chosen_id), None)
        if chosen and (raw or {}).get("slot_to_replace") == slot_id and (raw or {}).get("checked_against_active_slots") is True:
            return chosen, {**context, "agent_choice": raw}
        if (
            not chosen_id
            and (raw or {}).get("slot_to_replace") == slot_id
            and (raw or {}).get("checked_against_active_slots") is True
        ):
            # A valid structured response may deliberately reject every
            # candidate.  That semantic decision must not be converted into a
            # hidden first-candidate selection.
            if len(safe) > 5 and selector_page + 1 < config.REPLACEMENT_SELECTOR_MAX_PAGES:
                chosen, later = _select_replacement_candidate(
                    slot_id=slot_id, qtype=qtype, conflict_slots=conflict_slots,
                    conflict_edges=conflict_edges, candidates=safe[5:],
                    active_items=active_items, blacklist=blacklist,
                    previous_failures=previous_failures,
                    failed_slot_context=failed_slot_context,
                    existing_questions=existing_questions,
                    allow_semantic_overlap_candidates=allow_semantic_overlap_candidates,
                    selector_operation=selector_operation,
                    selector_page=selector_page + 1,
                )
                later["previous_candidate_page"] = {
                    "candidate_ids": [_identity(item) for item in window],
                    "agent_choice": raw,
                }
                return chosen, later
            return None, {**context, "agent_choice": raw,
                          "selection_source": "agent_no_candidate"}
        context["selector_error"] = "invalid_candidate_reference"
        context["agent_choice"] = raw
    except Exception as exc:
        logger.warning("Duplicate replacement agent failed; deterministic fallback used | %s", exc)
        context["agent_error"] = str(exc)
    # Operational/schema failure is distinct from a valid semantic rejection.
    # The window already passed deterministic eligibility and ranking, so the
    # existing bounded recovery path may use its first candidate.
    return window[0], {**context, "selection_source": "deterministic_fallback"}


def _used_fitb_answer_keys(
    exam: dict[str, Any], plans: list[dict[str, Any]], model_number: int,
    excluding_slot_id: str,
) -> set[str]:
    plan_by_slot = _plan_by_slot(plans, model_number)
    keys: set[str] = set()
    section = (exam.get("questions") or {}).get("fill_in_the_blank") or {}
    for item in section.get("items") or []:
        slot_id = str(item.get("slot_id") or "")
        if slot_id == excluding_slot_id:
            continue
        keys.update(fitb_answer_keys(item))
        keys.update(fitb_answer_keys(plan_by_slot.get(slot_id) or {}))
    return keys


def _forced_catalog_reuse_pairs(
    exam: dict[str, Any], plans: list[dict[str, Any]], model_number: int
) -> set[frozenset[str]]:
    """Pairs that must reach the Judge even when embeddings do not shortlist them."""
    slot_to_qid: dict[str, str] = {}
    for section in (exam.get("questions") or {}).values():
        values = section.get("items") if isinstance(section, dict) else section
        for item in values or []:
            if item.get("slot_id") and item.get("question_id"):
                slot_to_qid[str(item["slot_id"])] = str(item["question_id"])
    plan_by_slot = _plan_by_slot(plans, model_number)
    forced: set[frozenset[str]] = set()
    for slot_id, item in plan_by_slot.items():
        replacement_stage = str(item.get("catalog_replacement_stage") or "")
        if not (
            is_authorized_catalog_exhaustion_reuse(item)
            or item.get("fitb_entry_reused") is True
            or replacement_stage.startswith("unused_task_")
            or replacement_stage == "unused_fitb_entry"
            or replacement_stage == "deterministic_unused_task"
            or replacement_stage == "regenerated_original_task_after_duplicate"
            or replacement_stage == "regenerated_original_fitb_entry"
        ):
            continue
        qid = slot_to_qid.get(slot_id)
        if not qid:
            continue
        task_id = str(item.get("fitb_entry_id") or item.get("task_id") or "")
        family_id = str(item.get("task_family_id") or "")
        related_slots = {
            other_slot for other_slot, other in plan_by_slot.items()
            if other_slot != slot_id and (
                (task_id and str(
                    other.get("fitb_entry_id") or other.get("task_id") or ""
                ) == task_id)
                or (family_id and str(other.get("task_family_id") or "") == family_id)
            )
        }
        related_slots.update(str(value) for value in item.get("forced_duplicate_slot_ids") or [])
        related_qids = {
            slot_to_qid.get(other_slot) for other_slot in related_slots
            if slot_to_qid.get(other_slot)
        }
        related_qids.update(str(value) for value in item.get("forced_duplicate_question_ids") or [])
        for other_qid in related_qids:
            if other_qid and other_qid != qid:
                forced.add(frozenset((qid, other_qid)))
    return forced


def scan_exam_duplicates(state: dict[str, Any]) -> dict[str, Any]:
    """Scan independent model exams concurrently, then merge deterministically."""
    exams = sorted(_exam_by_model(state).items())
    if len(exams) <= 1 or state.get("_active_duplicate_model"):
        return _scan_exam_duplicates_impl(state)

    from app.online.performance import finish_timing, start_timing

    def scan_model(model_number: int, exam: dict[str, Any]):
        timing = start_timing(stage="within_model_duplicate_v2", model_number=model_number)
        worker = {
            **state,
            "_active_duplicate_model": model_number,
            "generated_exams": [deepcopy(exam)],
            "validation_reports": [
                deepcopy(value) for value in state.get("validation_reports") or []
                if int(value.get("model_number") or 0) == model_number
            ],
            "repair_candidates": [
                deepcopy(value) for value in state.get("repair_candidates") or []
                if int(value.get("model_number") or 0) == model_number
            ],
            "duplicate_reports": [
                deepcopy(value) for value in state.get("duplicate_reports") or []
                if int(value.get("model_number") or 0) == model_number
            ],
            "duplicate_similarity_distributions": [
                deepcopy(value) for value in state.get("duplicate_similarity_distributions") or []
                if int(value.get("model_number") or 0) == model_number
            ],
            "duplicate_repair_history": [
                deepcopy(value) for value in state.get("duplicate_repair_history") or []
                if int(value.get("model_number") or 0) == model_number
            ],
            "unresolved_duplicate_conflicts": [
                deepcopy(value) for value in state.get("unresolved_duplicate_conflicts") or []
                if int(value.get("model_number") or 0) == model_number
            ],
            "warnings": [],
        }
        result = _scan_exam_duplicates_impl(worker)
        return model_number, result, finish_timing(timing,
            candidate_pair_count=sum(
                len(value.get("candidate_pairs") or [])
                for value in result.get("duplicate_reports") or []
            ))

    completed: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(len(exams), 10)) as executor:
        futures = {
            executor.submit(copy_context().run, scan_model, model, exam): model
            for model, exam in exams
        }
        for future in as_completed(futures):
            model, result, timing = future.result()
            completed[model] = (result, timing)

    merged: dict[str, Any] = {
        "duplicate_reports": [], "duplicate_findings": [],
        "duplicate_similarity_distributions": [],
        "pending_duplicate_groups": [], "invalid_questions": list(state.get("invalid_questions") or []),
        "warnings": list(state.get("warnings") or []), "generated_exams": [],
        "duplicate_fact_cache": dict(state.get("duplicate_fact_cache") or {}),
        "duplicate_screen_fingerprints": {}, "duplicate_candidate_blacklist": {},
        "duplicate_replacement_failures": {}, "duplicate_repair_history": [],
        "unresolved_duplicate_conflicts": [],
    }
    telemetry = list(state.get("performance_telemetry") or [])
    for model, _exam in exams:
        result, timing = completed[model]
        for key in (
            "duplicate_reports", "duplicate_findings",
            "duplicate_similarity_distributions",
            "pending_duplicate_groups", "warnings", "generated_exams",
            "duplicate_repair_history", "unresolved_duplicate_conflicts",
        ):
            merged[key].extend(deepcopy(result.get(key) or []))
        for key in (
            "duplicate_fact_cache", "duplicate_screen_fingerprints",
            "duplicate_candidate_blacklist", "duplicate_replacement_failures",
        ):
            merged[key].update(deepcopy(result.get(key) or {}))
        telemetry.append(timing)
    exemplar = next(iter(completed.values()))[0]
    for key in (
        "duplicate_scan_count", "duplicate_repair_rounds",
        "duplicate_verification_pending", "duplicate_verification_attempt",
    ):
        merged[key] = exemplar.get(key)
    merged["performance_telemetry"] = telemetry[-5000:]
    return merged


def _reassess_contradictions(
    report: dict[str, Any], exam: dict[str, Any], plan_by_slot: dict[str, dict[str, Any]],
    client_factory: Any | None,
) -> dict[str, Any]:
    """One focused recheck for duplicate verdicts contradicted by metadata.

    Triggered only when the Judge says two items share a target/operation while
    their authoritative targets/operations clearly differ. Max three pairs per
    scan, one call per pair batch, no loop.
    """
    pair_by_id = {
        str(value.get("pair_id") or ""): value
        for value in report.get("candidate_pairs") or []
    }
    notes: dict[str, str] = {}
    for verdict in report.get("pair_verdicts") or []:
        pair_id = str(verdict.get("pair_id") or "")
        if str(verdict.get("classification") or "") not in {"duplicate", "excessive_overlap"}:
            continue
        if verdict.get("exact_content"):
            continue
        pair = pair_by_id.get(pair_id) or {}
        left, right = pair.get("left") or {}, pair.get("right") or {}
        contradictions = []
        if str(verdict.get("same_target_or_result") or "").casefold() == "yes":
            target_left = str(left.get("target") or "")
            target_right = str(right.get("target") or "")
            if target_left and target_right and _jaccard(target_left, target_right) < 0.34:
                contradictions.append("prior same_target=yes conflicts with the different authoritative targets")
        if str(verdict.get("same_student_operation_or_method") or "").casefold() == "yes":
            operation_left = str(left.get("task_operation") or "")
            operation_right = str(right.get("task_operation") or "")
            if operation_left and operation_right and operation_left.casefold() != operation_right.casefold():
                contradictions.append("prior same_operation=yes conflicts with the different task operations")
        if contradictions:
            notes[pair_id] = "; ".join(contradictions)
        if len(notes) >= 3:
            break
    if not notes:
        return report
    forced = {
        frozenset(pair_id.split("::", 1)) for pair_id in notes
        if "::" in pair_id
    }
    kwargs: dict[str, Any] = {
        "forced_question_pairs": forced,
        "plan_by_slot": plan_by_slot,
        "forced_only": True,
        "fact_cache": {},
        "reassessment_notes": notes,
    }
    if client_factory is not None:
        kwargs["client_factory"] = client_factory
    try:
        reviewed = review_duplicate_candidates(exam, **kwargs)
    except Exception as exc:
        record_optional_failure(
            exc, stage="duplicate_reassessment", component="duplicate_v2",
            continued=True,
        )
        logger.warning(
            "duplicate reassessment unavailable | error=%s: %s",
            type(exc).__name__, exc, exc_info=True,
        )
        report["reassessment_error"] = f"{type(exc).__name__}: {exc}"
        return report
    new_verdicts = {
        str(value.get("pair_id") or ""): value
        for value in reviewed.get("pair_verdicts") or []
    }
    for verdict in report.get("pair_verdicts") or []:
        replacement = new_verdicts.get(str(verdict.get("pair_id") or ""))
        if replacement is not None:
            verdict.update(replacement)
    report["duplicate_groups"] = build_duplicate_groups(
        report.get("candidate_pairs") or [], report.get("pair_verdicts") or [],
    )
    report["reassessment"] = {
        "pair_ids": sorted(notes),
        "notes": notes,
        "corrected_pairs": sorted(
            pair_id for pair_id, note in notes.items()
            if str(new_verdicts.get(pair_id, {}).get("classification") or "") not in {
                "duplicate", "excessive_overlap",
            }
        ),
    }
    # Persist the corrected facts under the same content/prompt cache keys so a
    # later unchanged scan reuses them instead of reassessing again.
    report["reassessment_fact_cache"] = reviewed.get("fact_cache") or {}
    return report


def _build_duplicate_findings(
    report: dict[str, Any], model_number: int, scan_index: int,
) -> list[dict[str, Any]]:
    """Turn one model scan into the structured, status-bearing findings list."""
    by_pair = {
        str(value.get("pair_id") or ""): value
        for value in report.get("candidate_pairs") or []
    }
    replacement_by_question: dict[str, str] = {}
    group_by_question: dict[str, str] = {}
    for group in report.get("duplicate_groups") or []:
        for qid in group.get("replace_question_ids") or []:
            replacement_by_question[str(qid)] = str(group.get("group_id") or "")
        for qid in group.get("question_ids") or []:
            group_by_question[str(qid)] = str(group.get("group_id") or "")
    findings: list[dict[str, Any]] = []
    for verdict in report.get("pair_verdicts") or []:
        pair = by_pair.get(str(verdict.get("pair_id") or "")) or {}
        classification = str(verdict.get("classification") or "acceptable_reuse")
        left, right = pair.get("left") or {}, pair.get("right") or {}
        refs = []
        for side, item in (("left", left), ("right", right)):
            refs.append({
                "side": side,
                "question_id": item.get("question_id"),
                "slot_id": item.get("slot_id"),
                "question_type": item.get("question_type"),
                "task_id": item.get("task_id"),
                "selected_variation_id": item.get("selected_variation_id"),
                "question_fingerprint": pair.get(f"{side}_question_fingerprint"),
                "content_excerpt": str(item.get("stem") or "")[:300],
            })
        replace_id = next((
            ref["question_id"] for ref in refs
            if ref["question_id"] in replacement_by_question
        ), None)
        retain_id = next((
            ref["question_id"] for ref in refs if ref["question_id"] != replace_id
        ), None)
        requires = classification in {"duplicate", "excessive_overlap"}
        findings.append({
            "finding_id": f"dup_m{model_number}_s{scan_index}_{len(findings) + 1}",
            "scan_index": scan_index,
            "model_number": model_number,
            "pair_id": verdict.get("pair_id"),
            "group_id": group_by_question.get(str(refs[0].get("question_id") or "")),
            "question_refs": refs,
            "question_ids": [ref["question_id"] for ref in refs],
            "question_types": [ref["question_type"] for ref in refs],
            "screening": {
                "signals": list(pair.get("signals") or []),
                "screening_version": "duplicate-screen-v3",
            },
            "judgment": {
                **{
                    key: verdict.get(key) for key in (
                        "same_core_knowledge", "same_student_operation_or_method",
                        "same_target_or_result", "genuinely_new_reasoning",
                        "one_answer_substantially_gives_away_other",
                    )
                },
                "exact_content": bool(verdict.get("exact_content")),
                "authorized_practice_variation": bool(
                    verdict.get("authorized_practice_variation")
                ),
            },
            "classification": classification,
            "requires_intervention": requires,
            "retain_question_id": retain_id,
            "replace_question_id": replace_id,
            "decision_reason": (
                "Exact/overlapping assessment requires a distinct instance."
                if requires else "Different target, operation, or reasoning; reuse accepted."
            ),
            "status": (
                "confirmed" if requires else
                "acceptable_reuse" if classification == "acceptable_reuse" else
                classification
            ),
        })
    return findings


def _merge_duplicate_findings(
    existing: list[dict[str, Any]], current: list[dict[str, Any]],
    exam: dict[str, Any], scan_index: int,
) -> list[dict[str, Any]]:
    """Merge one scan's findings into stable lifecycle records.

    A confirmed/repaired pair is resolved only when the current scan no longer
    requires intervention and the replacement fingerprint changed. A repaired
    pair that is still conflicting becomes unresolved. Unchanged pairs update
    one record instead of accumulating a new record per scan.
    """
    from app.online.lifecycle import question_fingerprint

    fingerprint_by_id = {
        str(item.get("question_id") or ""): question_fingerprint(item)
        for section in (exam.get("questions") or {}).values()
        for item in (section.get("items") if isinstance(section, dict) else section) or []
        if item.get("question_id")
    }
    merged = [deepcopy(value) for value in existing]
    open_by_pair = {
        str(value.get("pair_id") or ""): value
        for value in merged
        if str(value.get("status") or "") not in {"resolved", "acceptable_reuse"}
    }
    current_pairs = {str(value.get("pair_id") or "") for value in current}
    for new_finding in current:
        pair_id = str(new_finding.get("pair_id") or "")
        previous = open_by_pair.get(pair_id)
        if previous is None:
            new_finding = deepcopy(new_finding)
            new_finding["scan_history"] = [{
                "scan_index": new_finding.get("scan_index"),
                "status": new_finding.get("status"),
                "classification": new_finding.get("classification"),
            }]
            merged.append(new_finding)
            continue
        history = list(previous.get("scan_history") or [])
        if not history:
            history.append({
                "scan_index": previous.get("scan_index"),
                "status": previous.get("status"),
                "classification": previous.get("classification"),
            })
        history.append({
            "scan_index": new_finding.get("scan_index"),
            "status": new_finding.get("status"),
            "classification": new_finding.get("classification"),
        })
        previous["scan_history"] = history
        previous["last_scan_index"] = new_finding.get("scan_index")
        previous["judgment"] = deepcopy(new_finding.get("judgment") or {})
        previous["classification"] = new_finding.get("classification")
        previous["requires_intervention"] = bool(new_finding.get("requires_intervention"))
        previous["retain_question_id"] = new_finding.get("retain_question_id")
        previous["replace_question_id"] = new_finding.get("replace_question_id")
        previous["decision_reason"] = new_finding.get("decision_reason")
        if new_finding.get("requires_intervention"):
            previous["status"] = (
                "unresolved" if previous.get("status") == "repaired" else "confirmed"
            )
        else:
            previous["status"] = (
                "resolved" if previous.get("status") in {"confirmed", "repaired", "unresolved"}
                else "acceptable_reuse"
            )
            if previous["status"] == "resolved":
                previous["resolved_scan_index"] = new_finding.get("scan_index")
    for previous in merged:
        pair_id = str(previous.get("pair_id") or "")
        if pair_id in current_pairs:
            continue
        if str(previous.get("status") or "") not in {"confirmed", "repaired", "unresolved"}:
            continue
        replace_id = str(previous.get("replace_question_id") or "")
        original = next((
            ref for ref in previous.get("question_refs") or []
            if str(ref.get("question_id") or "") == replace_id
        ), {})
        if (
            replace_id
            and original.get("question_fingerprint")
            and fingerprint_by_id.get(replace_id) != original.get("question_fingerprint")
        ):
            previous["status"] = "resolved"
            previous["resolved_scan_index"] = scan_index
            previous.setdefault("scan_history", []).append({
                "scan_index": scan_index,
                "status": "resolved",
                "classification": "no_longer_screened",
            })
        else:
            previous["status"] = "unresolved"
    return merged


def _scan_exam_duplicates_impl(state: dict[str, Any]) -> dict[str, Any]:
    reports = list(state.get("duplicate_reports") or [])
    distributions = list(state.get("duplicate_similarity_distributions") or [])
    pending: list[dict[str, Any]] = []
    invalid = list(state.get("invalid_questions") or [])
    repair_rounds = int(state.get("duplicate_repair_rounds") or 0)
    slot_attempts = dict(state.get("duplicate_replacement_attempts") or {})
    warnings = list(state.get("warnings") or [])
    blacklists = {key: list(value) for key, value in
                  (state.get("duplicate_candidate_blacklist") or {}).items()}
    failure_memory = {key: list(value) for key, value in
                      (state.get("duplicate_replacement_failures") or {}).items()}
    history = list(state.get("duplicate_repair_history") or [])
    findings = list(state.get("duplicate_findings") or [])
    scan_index = int(state.get("duplicate_scan_count") or 0) + 1
    fact_cache = dict(state.get("duplicate_fact_cache") or {})
    previous_screen_fingerprints = deepcopy(
        state.get("duplicate_screen_fingerprints") or {}
    )
    next_screen_fingerprints: dict[str, dict[str, str]] = {}
    scanned_any = False
    verification_pending = bool(state.get("duplicate_verification_pending"))
    verification_attempt = int(
        state.get("duplicate_verification_attempt") or repair_rounds + 1
    )
    if repair_rounds >= config.MAX_DUPLICATE_REPAIR_ROUNDS and not verification_pending:
        unresolved = _current_conflict_groups(
            state.get("unresolved_duplicate_conflicts") or [], _exam_by_model(state)
        )
        unresolved.extend(deepcopy(state.get("pending_duplicate_groups") or []))
        if not unresolved and reports:
            unresolved.extend(deepcopy(reports[-1].get("duplicate_groups") or []))
        warnings.append(
            "Duplicate repair budget exhausted; stopped before another Duplicate Judge call."
        )
        for finding in findings:
            if str(finding.get("status") or "") in {"confirmed", "repaired", "pending"}:
                finding["status"] = "unresolved"
                finding["decision_reason"] = (
                    finding.get("decision_reason")
                    or "Duplicate recovery budget exhausted."
                )
        return {
            "duplicate_reports": reports,
            "duplicate_findings": findings,
            "duplicate_similarity_distributions": distributions,
            "pending_duplicate_groups": [],
            "duplicate_scan_count": int(state.get("duplicate_scan_count") or 0),
            "invalid_questions": invalid, "warnings": warnings,
            "generated_exams": list(_exam_by_model(state).values()),
            "duplicate_fact_cache": fact_cache,
            "unresolved_duplicate_conflicts": unresolved,
            "duplicate_verification_pending": False,
            "duplicate_verification_attempt": 0,
        }
    stored_domain = str((state.get("assessment_profile") or {}).get("domain") or "")
    review_client_factory = None
    if stored_domain:
        from app.llm.factory import create_review_client
        review_client_factory = lambda: create_review_client(
            domain=stored_domain, stage="duplicate_validator"
        )

    def run_review(
        filtered_exam: dict[str, Any], forced_pairs: set[frozenset[str]],
        plan_by_slot: dict[str, dict[str, Any]], changed_ids: set[str] | None,
    ):
        kwargs: dict[str, Any] = {
            "fact_cache": fact_cache,
            "forced_question_pairs": forced_pairs,
            "plan_by_slot": plan_by_slot,
            "changed_question_ids": changed_ids,
        }
        if review_client_factory is not None:
            kwargs["client_factory"] = review_client_factory
        try:
            return review_duplicate_candidates(filtered_exam, **kwargs)
        except TypeError as exc:
            # Compatibility for narrow test/dry-run substitutes implementing the
            # historical one-positional-argument call contract.
            if "unexpected keyword argument" not in str(exc):
                raise
            return review_duplicate_candidates(filtered_exam)

    unresolved: list[dict[str, Any]] = []
    final_verification = (
        verification_pending
        and verification_attempt >= config.MAX_DUPLICATE_REPAIR_ROUNDS
    )
    for model_number, exam in _exam_by_model(state).items():
        filtered_exam = _correctness_pass_exam(state, exam)
        # Legacy compatibility: persisted jobs may still carry exempt IDs from the
        # older contract where attempt-three numeric variants skipped this judge.
        # New numeric variants are no longer exempt, so the final variation always
        # receives an independent duplicate verification here.
        exempt_ids = set(state.get("duplicate_judge_exempt_ids") or [])
        if exempt_ids:
            for section in (filtered_exam.get("questions") or {}).values():
                values = section.get("items") if isinstance(section, dict) else section
                if isinstance(values, list):
                    values[:] = [
                        value for value in values
                        if str(value.get("question_id") or "") not in exempt_ids
                    ]
        forced_pairs = _forced_catalog_reuse_pairs(
            filtered_exam, state.get("plans") or [], model_number
        )
        plan_by_slot = _plan_by_slot(state.get("plans") or [], model_number)
        current_screen_fingerprints = _duplicate_fingerprint_index(
            filtered_exam, plan_by_slot,
        )
        previous_model_fingerprints = dict(
            previous_screen_fingerprints.get(str(model_number))
            or previous_screen_fingerprints.get(model_number) or {}
        )
        next_screen_fingerprints[str(model_number)] = current_screen_fingerprints
        pending_ids = set(map(str, (state.get("pending_revalidation_ids") or {}).get(model_number, []) or []))
        pending_ids.update(
            str(candidate.get("question_id") or "")
            for candidate in state.get("repair_candidates") or []
            if int(candidate.get("model_number") or 0) == model_number
        )
        changed_ids = {
            qid for qid, fingerprint in current_screen_fingerprints.items()
            if previous_model_fingerprints.get(qid) != fingerprint
        }
        changed_ids.update(pending_ids)
        baseline_exists = bool(previous_model_fingerprints)
        if baseline_exists and not changed_ids and not verification_pending:
            continue
        if baseline_exists and not current_screen_fingerprints and not verification_pending:
            continue
        review_scope = (
            None if verification_pending or not baseline_exists else changed_ids
        )
        report = run_review(filtered_exam, forced_pairs, plan_by_slot, review_scope)
        report = _reassess_contradictions(
            report, filtered_exam, plan_by_slot, review_client_factory,
        )
        fact_cache.update(deepcopy(report.pop("reassessment_fact_cache", {}) or {}))
        scanned_any = True
        fact_cache = dict(report.pop("fact_cache", fact_cache))
        by_id = _build_question_fingerprint_index(filtered_exam)
        for pair in report.get("candidate_pairs") or []:
            pair["left_question_fingerprint"] = by_id.get(
                str((pair.get("left") or {}).get("question_id") or "")
            )
            pair["right_question_fingerprint"] = by_id.get(
                str((pair.get("right") or {}).get("question_id") or "")
            )
            pair["screening_version"] = "duplicate-screen-v2-fingerprinted"
        for verdict in report.get("pair_verdicts") or []:
            pair = next((item for item in report.get("candidate_pairs") or []
                         if item.get("pair_id") == verdict.get("pair_id")), {})
            verdict["left_question_fingerprint"] = pair.get("left_question_fingerprint")
            verdict["right_question_fingerprint"] = pair.get("right_question_fingerprint")
            verdict["judge_version"] = "duplicate-judge-v2-fingerprinted"
            verdict.setdefault("resolution_status", "active")
        for group in report.get("duplicate_groups") or []:
            group["question_fingerprints"] = {
                str(qid): by_id.get(str(qid)) for qid in group.get("question_ids") or []
            }
        report_record = {"model_number": model_number, "scan_index": scan_index, **report}
        current_findings = _build_duplicate_findings(report_record, model_number, scan_index)
        findings = _merge_duplicate_findings(
            findings, current_findings, filtered_exam, scan_index,
        )
        reports.append(report_record)
        distributions.append({
            "model_number": model_number, "scan_index": scan_index,
            **report.get("similarity_distribution", {}),
        })
        groups = list(report.get("duplicate_groups") or [])
        if not groups:
            continue
        for group in groups:
            for qid in group.get("question_ids") or []:
                slot_id = _question_slot(exam, qid)
                latest = next((event for event in reversed(history)
                               if int(event.get("model_number") or 0) == model_number
                               and str(event.get("slot_id") or "") == slot_id
                               and event.get("status") == "replaced"), None)
                if latest:
                    failed_id = str(latest.get("fitb_entry_id") or latest.get("task_id") or "")
                    key = _slot_attempt_key(model_number, slot_id)
                    if failed_id:
                        blacklists.setdefault(key, []).append(failed_id)
                        failure_memory.setdefault(key, []).append({
                            "candidate_id": failed_id,
                            "reason": "duplicate_after_replacement",
                            "conflict_edges": deepcopy(group.get("edges") or []),
                        })
        if final_verification:
            unresolved.extend({"model_number": model_number, **deepcopy(group)} for group in groups)
            continue
        replace_ids = list(dict.fromkeys(
            qid for group in groups for qid in group.get("replace_question_ids") or []
        ))
        eligible_ids: set[str] = set()
        exhausted: dict[str, str] = {}
        for qid in replace_ids:
            slot_id = _question_slot(exam, qid)
            qtype = next((
                name for name, section in (exam.get("questions") or {}).items()
                if any(
                    str(value.get("question_id") or "") == qid
                    for value in (section.get("items") if isinstance(section, dict) else section) or []
                )
            ), "")
            exclusions = _failed_task_exclusions(
                list(state.get("duplicate_repair_history") or []), model_number, slot_id
            )
            inventory = compatible_task_inventory(
                state.get("assessment_catalog") or {}, state.get("plans") or [], qtype,
                str(state.get("difficulty") or "mix"), excluding_slot_id=slot_id,
                excluded_task_ids=exclusions,
                active_slot_ids=_active_slot_ids(exam),
            ) if qtype else {}
            current_plan = _plan_by_slot(
                state.get("plans") or [], model_number
            ).get(slot_id) or {}
            current_task_id = str(
                current_plan.get("fitb_entry_id") or current_plan.get("task_id") or ""
            )
            has_eligible_unused = any(
                str(task_id) != current_task_id
                for task_id in inventory.get("eligible_unused_after_exclusions_task_ids") or []
            )
            if int(slot_attempts.get(_slot_attempt_key(model_number, slot_id)) or 0) >= (
                config.MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT
            ):
                exhausted[qid] = "slot_attempt_limit"
            else:
                eligible_ids.add(qid)

        for group in groups:
            eligible = [
                qid for qid in group.get("replace_question_ids") or [] if qid in eligible_ids
            ]
            if eligible:
                pending.append({
                    "model_number": model_number,
                    **group,
                    "replace_question_ids": eligible,
                })

        for qid, exhausted_by in exhausted.items():
            slot_id = _question_slot(exam, qid)
            unresolved.extend(
                {"model_number": model_number, **deepcopy(group)}
                for group in groups if qid in (group.get("question_ids") or [])
            )
            history.append({
                "model_number": model_number, "question_id": qid,
                "slot_id": slot_id, "status": "duplicate_repair_exhausted_kept",
                "exhausted_by": exhausted_by,
            })
        fitb = exam.get("questions", {}).get("fill_in_the_blank")
        if isinstance(fitb, dict):
            from app.online.validation.questions import _fitb_word_bank_valid, _rebuild_fitb_word_bank
            if not _fitb_word_bank_valid(fitb):
                _rebuild_fitb_word_bank(fitb)
        if exhausted:
            warnings.append(
                f"Model {model_number}: retained {len(exhausted)} valid slot(s) with unresolved "
                "similarity after all three duplicate-repair attempts."
            )
    return {
        "duplicate_reports": reports,
        "duplicate_findings": findings,
        "duplicate_similarity_distributions": distributions,
        "pending_duplicate_groups": pending,
        "duplicate_scan_count": (
            scan_index if scanned_any else int(state.get("duplicate_scan_count") or 0)
        ),
        "invalid_questions": invalid,
        "warnings": warnings,
        "generated_exams": list(_exam_by_model(state).values()),
        "duplicate_fact_cache": fact_cache,
        "duplicate_screen_fingerprints": next_screen_fingerprints,
        "duplicate_candidate_blacklist": {
            key: list(dict.fromkeys(values)) for key, values in blacklists.items()
        },
        "duplicate_replacement_failures": failure_memory,
        "duplicate_repair_history": history,
        "duplicate_repair_rounds": (
            verification_attempt if verification_pending else repair_rounds
        ),
        "duplicate_verification_pending": False,
        "duplicate_verification_attempt": 0,
        "unresolved_duplicate_conflicts": unresolved,
    }


def _generate_reserved(
    state: dict[str, Any], plans: list[dict[str, Any]], model_number: int,
    question_id: str, exam: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, str | None, list[str]]:
    from app.online.validation.questions import _build_by_id
    from app.online.validation.context import _plan_context_for_model, _regenerate_plan_slot

    isolated_exam = deepcopy(exam)
    isolated_state = {**state, "plans": deepcopy(plans)}
    plan_by_slot, source_by_chunk = _plan_context_for_model(isolated_state, model_number)
    success, warnings = _regenerate_plan_slot(
        isolated_state, isolated_exam, model_number, question_id, plan_by_slot, source_by_chunk
    )
    if not success:
        return question_id, None, None, warnings
    entry = _build_by_id(isolated_exam.get("questions") or {}).get(question_id)
    if entry is None:
        return question_id, None, None, [*warnings, "replacement disappeared after generation"]
    qtype, _container, item = entry
    return question_id, deepcopy(item), qtype, warnings


def _generate_reserved_with_adjustment(
    state: dict[str, Any], plans: list[dict[str, Any]], model_number: int,
    reservation: dict[str, Any], exam: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, str | None, list[str], dict[str, Any]]:
    """Keep one replacement's generation→difficulty dependency in one worker."""
    qid, generated, qtype, warnings = _generate_reserved(
        state, plans, model_number, reservation["question_id"], exam,
    )
    if generated is None:
        return qid, None, qtype, warnings, {"status": "not_run"}
    from app.online.difficulty_adjustment import adjust_generated_question
    active_plan = deepcopy(
        _plan_by_slot(plans, model_number).get(reservation["slot_id"]) or {}
    )
    adjusted, adjustment = adjust_generated_question(
        state=state, qtype=qtype or reservation["qtype"],
        question=generated, plan=active_plan,
        previous_questions=reservation.get("previous_task_questions") or [],
    )
    return qid, adjusted, qtype, warnings, adjustment


def _precommit_fallback_candidate(
    state: dict[str, Any], plans: list[dict[str, Any]], model_number: int,
    question_id: str, exam: dict[str, Any], generated: dict[str, Any],
    fact_cache: dict[str, dict[str, str]],
) -> tuple[bool, list[str], dict[str, dict[str, str]]]:
    """Correctness-check and duplicate-check a fallback before mutating the live exam."""
    from app.llm.factory import create_review_client
    from app.online.validation.questions import _build_by_id
    from app.online.validation.context import (
        _merge_role_reports, _plan_context_for_model, _quantitative_question_ids,
    )
    from app.online.validator import _validate_with_context

    candidate_exam = deepcopy(exam)
    entry = _build_by_id(candidate_exam.get("questions") or {}).get(question_id)
    if entry is None:
        return False, ["fallback precommit: target question disappeared"], fact_cache
    entry[2].clear()
    entry[2].update(deepcopy(generated))
    context_state = {**state, "plans": plans}
    plan_by_slot, source_by_chunk = _plan_context_for_model(context_state, model_number)
    domain = str((state.get("assessment_profile") or {}).get("domain") or "").upper()
    try:
        general_client = create_review_client(domain=domain, stage="general_validator")
        general = _validate_with_context(
            candidate_exam, [question_id], plan_by_slot, source_by_chunk,
            general_client, "general",
        )
        correctness = general
        if domain == "STEM" and question_id in set(
            _quantitative_question_ids(candidate_exam, plan_by_slot)
        ):
            quantitative_client = create_review_client(
                domain=domain, stage="stem_quantitative_validator"
            )
            quantitative = _validate_with_context(
                candidate_exam, [question_id], plan_by_slot, source_by_chunk,
                quantitative_client, "stem_quantitative",
            )
            correctness = _merge_role_reports(general, quantitative)
    except Exception as exc:
        logger.exception(
            "Fallback correctness precommit failed | model=%d | question_id=%s",
            model_number, question_id,
        )
        return False, [f"fallback precommit correctness check failed: {exc}"], fact_cache
    if not correctness.get("all_pass"):
        reasons = [
            str(verdict.get("reason") or verdict.get("failure_code") or "invalid")
            for verdict in correctness.get("verdicts") or []
            if verdict.get("action") != "PASS"
        ]
        return False, [f"fallback precommit correctness rejected: {'; '.join(reasons)}"], fact_cache

    _attach_reuse_context(candidate_exam, plans, model_number)
    forced = _forced_catalog_reuse_pairs(candidate_exam, plans, model_number)
    try:
        duplicate_client = create_review_client(domain=domain, stage="duplicate_validator")
        duplicate_report = review_duplicate_candidates(
            candidate_exam, client=duplicate_client, fact_cache=fact_cache,
            forced_question_pairs=forced,
        )
    except Exception as exc:
        logger.exception(
            "Fallback duplicate precommit failed | model=%d | question_id=%s",
            model_number, question_id,
        )
        return False, [f"fallback precommit duplicate check failed: {exc}"], fact_cache
    updated_cache = dict(duplicate_report.get("fact_cache") or fact_cache)
    conflicts = [
        verdict for verdict in duplicate_report.get("pair_verdicts") or []
        if verdict.get("classification") in {"duplicate", "excessive_overlap"}
        and any(
            question_id == pair.get(side, {}).get("question_id")
            for pair in duplicate_report.get("candidate_pairs") or []
            if pair.get("pair_id") == verdict.get("pair_id")
            for side in ("left", "right")
        )
    ]
    if conflicts:
        labels = ", ".join(
            f"{value.get('pair_id')}={value.get('classification')}" for value in conflicts
        )
        return False, [f"fallback precommit duplicate rejected: {labels}"], updated_cache
    return True, [], updated_cache


def repair_duplicate_groups(state: dict[str, Any]) -> dict[str, Any]:
    from app.online.validation.questions import _build_by_id, _fitb_word_bank_valid, _rebuild_fitb_word_bank

    plans = deepcopy(state.get("plans") or [])
    exams = _exam_by_model(state)
    completed_rounds = int(state.get("duplicate_repair_rounds") or 0)
    repair_round = completed_rounds + 1
    slot_attempts = dict(state.get("duplicate_replacement_attempts") or {})
    pending_revalidation = dict(state.get("pending_revalidation_ids") or {})
    validated = list(state.get("validated_models") or [])
    history = list(state.get("duplicate_repair_history") or [])
    warnings = list(state.get("warnings") or [])
    catalog = state.get("assessment_catalog") or {}
    fact_cache = dict(state.get("duplicate_fact_cache") or {})
    diagnostics = list(state.get("developer_diagnostics") or [])
    blacklists = {key: list(value) for key, value in
                  (state.get("duplicate_candidate_blacklist") or {}).items()}
    failure_memory = {key: list(value) for key, value in
                      (state.get("duplicate_replacement_failures") or {}).items()}
    verification_required = False
    numeric_repaired_ids: list[str] = []
    judge_exempt_ids = list(state.get("duplicate_judge_exempt_ids") or [])
    numeric_fallback_attempted = False
    finalization_pending = False
    unresolved_conflicts = list(state.get("unresolved_duplicate_conflicts") or [])
    for report in state.get("validation_reports") or []:
        report_model = int(report.get("model_number") or 0)
        for verdict in report.get("verdicts") or []:
            if verdict.get("action") == "PASS":
                continue
            qid = str(verdict.get("question_id") or "")
            slot_id = _question_slot(exams.get(report_model) or {}, qid)
            latest = next((event for event in reversed(history)
                           if int(event.get("model_number") or 0) == report_model
                           and str(event.get("slot_id") or "") == slot_id
                           and event.get("status") == "replaced"), None)
            if latest:
                candidate_id = str(latest.get("fitb_entry_id") or latest.get("task_id") or "")
                key = _slot_attempt_key(report_model, slot_id)
                if candidate_id:
                    blacklists.setdefault(key, []).append(candidate_id)
                    failure_memory.setdefault(key, []).append({
                        "candidate_id": candidate_id,
                        "reason": str(verdict.get("failure_code") or verdict.get("reason") or "validation_failed"),
                    })
    groups_by_model: dict[int, list[dict[str, Any]]] = {}
    for group in state.get("pending_duplicate_groups") or []:
        groups_by_model.setdefault(int(group.get("model_number") or 1), []).append(group)

    for model_number, groups in groups_by_model.items():
        exam = exams.get(model_number)
        if exam is None:
            continue
        reserved: list[dict[str, Any]] = []
        by_id = _build_by_id(exam.get("questions") or {})
        replace_ids = list(dict.fromkeys(
            qid for group in groups for qid in group.get("replace_question_ids") or []
        ))
        # Reservation is deliberately sequential and atomic. Generation starts only
        # after every candidate has seen the tasks reserved before it.
        for qid in replace_ids:
            entry = by_id.get(qid)
            if entry is None:
                history.append({"model_number": model_number, "question_id": qid,
                                "repair_round": repair_round, "status": "question_not_found"})
                continue
            qtype, _container, item = entry
            slot_id = str(item.get("slot_id") or "")
            attempt_key = _slot_attempt_key(model_number, slot_id)
            if int(slot_attempts.get(attempt_key) or 0) >= config.MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT:
                history.append({"model_number": model_number, "question_id": qid,
                                "slot_id": slot_id, "repair_round": repair_round,
                                "status": "slot_attempt_limit"})
                continue
            slot_attempt = int(slot_attempts.get(attempt_key) or 0) + 1
            slot_attempts[attempt_key] = slot_attempt
            related_question_ids = list(dict.fromkeys(
                other_qid
                for group in groups if qid in (group.get("replace_question_ids") or [])
                for other_qid in (group.get("question_ids") or [])
                if str(other_qid) and str(other_qid) != qid
            ))
            related_slot_ids = list(dict.fromkeys(
                _question_slot(exam, other_qid) for other_qid in related_question_ids
                if _question_slot(exam, other_qid)
            ))
            old_plan_item = deepcopy(_plan_by_slot(plans, model_number).get(slot_id) or {})
            failed_replacement_task_ids = _failed_task_exclusions(
                history, model_number, slot_id
            )
            blacklist = set(blacklists.get(attempt_key) or []) | failed_replacement_task_ids
            active_plan_items = list(_plan_by_slot(plans, model_number).values())
            conflict_edges = [
                edge for group in groups if qid in (group.get("replace_question_ids") or [])
                for edge in group.get("edges") or []
                if qid in {edge.get("left"), edge.get("right")}
            ]
            numeric_applicable = _numeric_variation_applicable(
                state, exam, model_number, qid, qtype, item
            )
            if slot_attempt == 3 and not numeric_applicable:
                finalization_pending = True
                unresolved_conflicts.extend(
                    {"model_number": model_number, **deepcopy(group)}
                    for group in groups if qid in (group.get("question_ids") or [])
                )
                history.append({
                    "model_number": model_number, "question_id": qid,
                    "slot_id": slot_id, "repair_round": repair_round,
                    "slot_attempt": slot_attempt,
                    "status": "numeric_variation_not_applicable_kept",
                })
                continue
            if slot_attempt in {1, 3} and numeric_applicable:
                numeric_fallback_attempted = True
                repaired, numeric_warnings = _numeric_variation_fallback(
                    state, exam, model_number, qid, related_question_ids, conflict_edges
                )
                warnings.extend(numeric_warnings)
                if repaired is None:
                    failed_instance = stable_fingerprint({
                        key: item.get(key) for key in _NUMERIC_VARIATION_FIELDS.get(qtype, [])
                        if key in item
                    })
                    history.append({
                        "model_number": model_number, "question_id": qid,
                        "slot_id": slot_id, "repair_round": repair_round,
                        "slot_attempt": slot_attempt,
                        "duplicate_repair_mode": "numeric_variation_fallback",
                        "status": "numeric_variation_failed",
                        "failure_reason": "; ".join(numeric_warnings),
                        "numerical_instance_fingerprint": failed_instance,
                    })
                    # Fall through to Catalog replacement in the same bounded
                    # recovery turn; a failed numeric repair must not consume a
                    # whole round without trying the next authorized route.
                else:
                    item.clear()
                    item.update(repaired)
                    item.update({
                        "duplicate_repair_attempt": slot_attempt,
                        "duplicate_repair_mode": "numeric_variation_fallback",
                        "duplicate_conflict_with": (
                            related_question_ids[0] if len(related_question_ids) == 1
                            else related_question_ids
                        ),
                        "numeric_variant": True,
                        "duplicate_recheck_required": True,
                    })
                    live_plan = _plan_by_slot(plans, model_number).get(slot_id) or {}
                    _apply_effective_numeric_instance(live_plan, item, qtype)
                    numeric_repaired_ids.append(qid)
                    history.append({
                        "model_number": model_number, "question_id": qid,
                        "slot_id": slot_id, "repair_round": repair_round,
                        "slot_attempt": slot_attempt,
                        "duplicate_repair_mode": "numeric_variation_fallback",
                        "conflict_question_ids": related_question_ids,
                        "status": "numeric_variation_generated",
                    })
                    continue
            candidates = _candidate_pool(
                catalog, plans, slot_id, qtype, str(state.get("difficulty") or "mix"),
                _active_slot_ids(exam), blacklist,
            )
            replacement, selector_context = _select_replacement_candidate(
                slot_id=slot_id, qtype=qtype, conflict_slots=related_slot_ids,
                conflict_edges=conflict_edges, candidates=candidates,
                active_items=active_plan_items, blacklist=blacklist,
                previous_failures=failure_memory.get(attempt_key) or [],
                failed_slot_context={
                    **_compact_metadata(old_plan_item),
                    "question_id": qid,
                    "question": _complete_question_context(
                        qid, item, old_plan_item, role="failed_duplicate_question"
                    ),
                    "failure_code": "semantic_duplicate",
                    "failure_reason": conflict_edges,
                },
                existing_questions=[
                    _complete_question_context(
                        candidate_id, entry[2],
                        next((
                            active for active in active_plan_items
                            if str(active.get("slot_id") or "")
                            == str(entry[2].get("slot_id") or "")
                        ), {}),
                        role="active_exam_question",
                    )
                    for candidate_id, entry in by_id.items()
                ],
                selector_operation="duplicate_replacement_selector",
            )
            for rejected in (selector_context.get("agent_choice") or {}).get(
                "rejected_candidates", []
            ):
                rejected_id = str(rejected.get("candidate_id") or "")
                if not rejected_id:
                    continue
                blacklists.setdefault(attempt_key, []).append(rejected_id)
                failure_memory.setdefault(attempt_key, []).append({
                    "candidate_id": rejected_id,
                    "reason": str(rejected.get("reason") or "replacement_selector_rejected"),
                    "stage": "replacement_selector",
                })
            if replacement:
                replacement["selected_replacement_task_id"] = replacement.get("task_id")
                replacement["task_was_previously_used"] = bool(
                    replacement.get("task_reused") or replacement.get("fitb_entry_reused")
                )
                replacement["forced_duplicate_slot_ids"] = related_slot_ids
                replacement["forced_duplicate_question_ids"] = related_question_ids
                replacement["duplicate_conflict_context"] = conflict_edges
                agent_choice = selector_context.get("agent_choice") or {}
                replacement["suggested_assessment_angle"] = (
                    agent_choice.get("why_this_candidate_resolves_the_conflict")
                    or "Assess the selected task without repeating the conflicting operation, target, or work product."
                )
                required = list(replacement.get("required_instance_distinctions") or [])
                required.extend(
                    distinction for edge in conflict_edges
                    for distinction in edge.get("what_must_be_different") or []
                )
                replacement["required_instance_distinctions"] = list(dict.fromkeys(required))
            if replacement and (
                replacement.get("task_reused") or replacement.get("fitb_entry_reused")
            ):
                reused_task_id = str(
                    replacement.get("fitb_entry_id") or replacement.get("task_id") or ""
                )
                plan_items = _plan_by_slot(plans, model_number)
                previous_task_questions = []

                def add_context(
                    context_qid: str, context_slot: str, context_plan: dict[str, Any],
                    context_item: dict[str, Any], role: str,
                ) -> None:
                    if any(
                        str(existing.get("question_id") or "") == context_qid
                        for existing in previous_task_questions
                    ):
                        return
                    previous_task_questions.append(_complete_question_context(
                        context_qid, context_item, context_plan, role=role,
                    ))

                for previous_slot, previous_plan in plan_items.items():
                    if previous_slot == slot_id or str(
                        previous_plan.get("fitb_entry_id")
                        or previous_plan.get("task_id") or ""
                    ) != reused_task_id:
                        continue
                    previous_qid = next((
                        candidate_id for candidate_id, entry in by_id.items()
                        if str(entry[2].get("slot_id") or "") == previous_slot
                    ), "")
                    previous_item = (by_id.get(previous_qid) or (None, None, {}))[2]
                    add_context(
                        previous_qid, previous_slot, previous_plan, previous_item,
                        "previous_question_from_reused_task",
                    )
                if reused_task_id == str(
                    old_plan_item.get("fitb_entry_id") or old_plan_item.get("task_id") or ""
                ):
                    add_context(
                        qid, slot_id, old_plan_item, item,
                        "rejected_duplicate_question",
                    )
                    for conflict_qid in related_question_ids:
                        conflict_entry = by_id.get(conflict_qid) or (None, None, {})
                        conflict_item = conflict_entry[2]
                        conflict_slot = str(conflict_item.get("slot_id") or "")
                        add_context(
                            conflict_qid, conflict_slot,
                            plan_items.get(conflict_slot) or {}, conflict_item,
                            "duplicate_conflict",
                        )
                replacement["previous_task_questions"] = previous_task_questions
                replacement["previous_question_ids"] = [
                    str(value.get("question_id")) for value in previous_task_questions
                    if value.get("question_id")
                ]
                replacement["task_usage_count"] = len([
                    value for value in previous_task_questions
                    if value.get("context_role") == "previous_question_from_reused_task"
                ])
            if replacement is None or not _replace_plan_item(plans, model_number, slot_id, replacement):
                history.append({"model_number": model_number, "question_id": qid,
                                "slot_id": slot_id, "repair_round": repair_round,
                                "slot_attempt": slot_attempt,
                                "status": "no_compatible_catalog_replacement"})
                continue
            if catalog.get("catalog_version") == config.ASSESSMENT_CATALOG_VERSION:
                from app.online.assessment_catalog import validate_plan_catalog_bindings
                valid_chunk_ids = {
                    str(chunk.get("child_id") or "")
                    for chunk in state.get("retrieved_chunks") or []
                    if chunk.get("child_id")
                }
                binding_errors = validate_plan_catalog_bindings(
                    catalog, plans, valid_chunk_ids or None,
                )
                if binding_errors:
                    _replace_plan_item(plans, model_number, slot_id, old_plan_item)
                    blacklists.setdefault(attempt_key, []).append(_identity(replacement))
                    failure_memory.setdefault(attempt_key, []).append({
                        "candidate_id": _identity(replacement),
                        "reason": "; ".join(binding_errors),
                        "stage": "replacement_catalog_binding",
                        "attempt": slot_attempt,
                    })
                    history.append({
                        "model_number": model_number, "question_id": qid,
                        "slot_id": slot_id, "repair_round": repair_round,
                        "slot_attempt": slot_attempt,
                        "status": "replacement_catalog_binding_invalid",
                        "failure_reason": "; ".join(binding_errors),
                    })
                    continue
            reserved.append({"question_id": qid, "slot_id": slot_id, "qtype": qtype,
                             "task_id": replacement.get("task_id"),
                             "fitb_entry_id": replacement.get("fitb_entry_id"),
                             "catalog_replacement_stage": replacement.get("catalog_replacement_stage"),
                             "task_reused": bool(replacement.get("task_reused")),
                             "fitb_entry_reused": bool(replacement.get("fitb_entry_reused")),
                             "reuse_reason": replacement.get("reuse_reason"),
                             "task_inventory": deepcopy(replacement.get("task_inventory") or {}),
                             "fitb_answer_reused": bool(
                                 replacement.get("fitb_answer_reused")
                             ),
                             "previous_task_questions": deepcopy(
                                 replacement.get("previous_task_questions") or []
                             ),
                             "reused_due_to_catalog_exhaustion": bool(
                                 replacement.get("reused_due_to_catalog_exhaustion")
                             ),
                             "old_plan_item": old_plan_item,
                             "repair_round": repair_round, "slot_attempt": slot_attempt})

        repaired_ids: list[str] = []
        if reserved:
            with ThreadPoolExecutor(max_workers=min(4, len(reserved))) as executor:
                futures = {
                    executor.submit(
                        copy_context().run,
                        _generate_reserved_with_adjustment,
                        state,
                        deepcopy(plans),
                        model_number,
                        deepcopy(item),
                        exam,
                    ): item for item in reserved
                }
                for future in as_completed(futures):
                    reservation = futures[future]
                    history_reservation = {
                        key: value for key, value in reservation.items()
                        if key != "old_plan_item"
                    }
                    try:
                        qid, generated, qtype, generation_warnings, adjustment = future.result()
                    except Exception as exc:
                        from app.online.lifecycle import developer_diagnostic
                        diagnostics.append(developer_diagnostic(
                            exc, stage="duplicate_replacement",
                            slot_id=reservation.get("slot_id"),
                            question_id=reservation.get("question_id"),
                            task_id=reservation.get("task_id"),
                        ))
                        qid, generated, qtype, generation_warnings, adjustment = (
                            reservation["question_id"], None, reservation["qtype"], [str(exc)],
                            {"status": "not_run"},
                        )
                    warnings.extend(generation_warnings)
                    if generated is None:
                        _replace_plan_item(
                            plans, model_number, reservation["slot_id"],
                            reservation.get("old_plan_item") or {},
                        )
                        history.append({**history_reservation, "model_number": model_number,
                                        "status": "generation_failed"})
                        failed_id = str(reservation.get("fitb_entry_id") or reservation.get("task_id") or "")
                        if failed_id:
                            blacklists.setdefault(_slot_attempt_key(model_number, reservation["slot_id"]), []).append(failed_id)
                            failure_memory.setdefault(_slot_attempt_key(model_number, reservation["slot_id"]), []).append({
                                "candidate_id": failed_id, "reason": "generation_failed",
                            })
                        continue
                    active_plan = _plan_by_slot(plans, model_number).get(
                        reservation["slot_id"]
                    ) or {}
                    active_plan["adjustment_attempts"] = int(
                        active_plan.get("adjustment_attempts") or 0
                    ) + (0 if adjustment.get("status") == "not_needed" else 1)
                    if adjustment.get("status") == "succeeded":
                        target = str(active_plan.get("requested_difficulty") or "")
                        active_plan["difficulty"] = target
                        active_plan["actual_difficulty"] = target
                        active_plan["reasoning_depth"] = target
                        active_plan["reasoning_requirements"] = list(
                            adjustment.get("changed_reasoning_requirements") or []
                        )
                        active_plan["difficulty_fallback_reason"] = None
                        active_plan["adjustment_failure_reason"] = None
                        active_plan["difficulty_adaptation"] = {
                            "status": "adapted",
                            "from_difficulty": adjustment.get("from_difficulty")
                            or adjustment.get("direction", "").split("_to_")[0],
                            "to_difficulty": target,
                            "base_task_id": active_plan.get("base_catalog_task_id")
                            or active_plan.get("task_id"),
                            "strategy": "generated_question_adjustment",
                            "failed_strategies": [],
                            "changed_reasoning_requirements": list(
                                adjustment.get("changed_reasoning_requirements") or []
                            ),
                            "evidence_support": list(
                                adjustment.get("evidence_support") or []
                            ),
                            "evidence_justification": adjustment.get(
                                "evidence_justification"
                            ),
                            "base_variation_id": active_plan.get("selected_variation_id"),
                            "effective_instance_id": adjustment.get("effective_instance_id"),
                            "effective_givens": list(adjustment.get("effective_givens") or []),
                        }
                        if adjustment.get("effective_instance_id"):
                            active_plan["effective_instance_id"] = adjustment["effective_instance_id"]
                            active_plan["derived_task_id"] = adjustment["effective_instance_id"]
                            active_plan["effective_givens"] = list(
                                adjustment.get("effective_givens") or []
                            )
                    elif adjustment.get("status") == "failed":
                        active_plan["adjustment_failure_reason"] = adjustment.get("reason")
                        active_plan["difficulty_fallback_reason"] = adjustment.get("reason")
                    history.append({
                        **history_reservation, "model_number": model_number,
                        "status": "difficulty_adjustment_" + adjustment.get("status", "unknown"),
                        "difficulty_adjustment": adjustment,
                    })
                    old_identity = str(
                        (reservation.get("old_plan_item") or {}).get("fitb_entry_id")
                        or (reservation.get("old_plan_item") or {}).get("task_id") or ""
                    )
                    new_identity = str(
                        reservation.get("fitb_entry_id") or reservation.get("task_id") or ""
                    )
                    if old_identity and new_identity == old_identity:
                        accepted, precommit_warnings, fact_cache = _precommit_fallback_candidate(
                            state, plans, model_number, qid, exam, generated, fact_cache,
                        )
                        warnings.extend(precommit_warnings)
                        if not accepted:
                            _replace_plan_item(
                                plans, model_number, reservation["slot_id"],
                                reservation.get("old_plan_item") or {},
                            )
                            failed_id = str(
                                reservation.get("fitb_entry_id")
                                or reservation.get("task_id") or ""
                            )
                            if failed_id:
                                blacklists.setdefault(
                                    _slot_attempt_key(model_number, reservation["slot_id"]), []
                                ).append(failed_id)
                                failure_memory.setdefault(
                                    _slot_attempt_key(model_number, reservation["slot_id"]), []
                                ).append({
                                    "candidate_id": failed_id,
                                    "reason": "; ".join(precommit_warnings),
                                    "stage": "fallback_precommit",
                                })
                            history.append({
                                **history_reservation, "model_number": model_number,
                                "status": "precommit_rejected",
                                "failure_reason": "; ".join(precommit_warnings),
                            })
                            continue
                    current = _build_by_id(exam.get("questions") or {}).get(qid)
                    if current is None:
                        continue
                    current[2].clear(); current[2].update(generated)
                    repaired_ids.append(qid)
                    history.append({**history_reservation, "model_number": model_number,
                                    "status": "replaced"})
            fitb = exam.get("questions", {}).get("fill_in_the_blank")
            if isinstance(fitb, dict):
                if not _fitb_word_bank_valid(fitb):
                    _rebuild_fitb_word_bank(fitb)
        normal_repaired_ids = list(repaired_ids)
        repaired_ids.extend(numeric_repaired_ids)
        numeric_repaired_ids = []
        if repaired_ids:
            verification_required = verification_required or bool(normal_repaired_ids)
            from app.online.exam_builder import balance_mcq_answer_positions
            repaired_ids = list(dict.fromkeys([
                *repaired_ids,
                *balance_mcq_answer_positions(exam.get("questions") or {}, model_number),
            ]))
            pending_revalidation[model_number] = repaired_ids
            if model_number in validated:
                validated.remove(model_number)
        else:
            pending_revalidation.pop(model_number, None)

    return {
        "plans": plans, "generated_exams": list(exams.values()),
        "duplicate_repair_rounds": (
            max(completed_rounds, 3)
            if numeric_fallback_attempted or finalization_pending else completed_rounds
        ),
        "duplicate_verification_pending": (
            verification_required or numeric_fallback_attempted
        ),
        "duplicate_verification_attempt": (
            repair_round
            if (verification_required or numeric_fallback_attempted) else 0
        ),
        "duplicate_replacement_attempts": slot_attempts,
        "pending_revalidation_ids": pending_revalidation,
        "pending_duplicate_groups": [], "validated_models": validated,
        "duplicate_repair_history": history, "warnings": warnings,
        "duplicate_fact_cache": fact_cache,
        "duplicate_candidate_blacklist": {
            key: list(dict.fromkeys(values)) for key, values in blacklists.items()
        },
        "duplicate_replacement_failures": failure_memory,
        "duplicate_judge_exempt_ids": list(dict.fromkeys(judge_exempt_ids)),
        "duplicate_numeric_fallback_pending_validation": bool(judge_exempt_ids),
        "duplicate_repair_finalization_pending": (
            finalization_pending or bool(judge_exempt_ids)
        ),
        "unresolved_duplicate_conflicts": unresolved_conflicts,
        "developer_diagnostics": diagnostics,
    }


def has_pending_duplicate_repairs(state: dict[str, Any]) -> bool:
    return bool(state.get("pending_duplicate_groups"))
