"""Extracted from app.online.validator (mechanical move; behavior preserved)."""
from __future__ import annotations

import json
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from app.config import (
    MAX_PARALLEL_VALIDATOR_CALLS,
    VALIDATOR_BATCH_SIZE,
    VALIDATOR_TRACE_MAX_ATTEMPTS,
    REVALIDATION_OPERATIONAL_RETRIES,
    MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT,
    ASSESSMENT_CATALOG_VERSION,
    MAX_CONTENT_REPAIR_CALLS_PER_SLOT,
    MAX_FAILED_REPAIR_ATTEMPTS_PER_SLOT,
    MAX_COMPLETION_ATTEMPTS_PER_SLOT,
)
from app.llm.client import request_structured
from app.llm.factory import create_llm_client
from app.llm.schemas import (
    global_exam_review_response_model,
    repair_response_model,
    validator_response_model,
)
from app.logging_conf import get_logger
from app.online.eval_stats import record_first_validation, record_repair_outcome
from app.online.diagnostics import record_optional_failure
from app.online.assessment_catalog import (
    _apply_effective_variation,
    eligible_task_variation_candidates,
    is_pure_text_fitb_value,
    normalize_fitb_answer_key,
    variation_response_form_compatible,
)
from app.online.cross_model_duplicates import integrate_cross_model_duplicates
from app.online.models import (
    _normalize_definition,
    _normalize_equation,
    _normalize_essay,
    _normalize_mcq,
    _normalize_why,
    _normalize_true_false,
    _normalize_word_problem,
    authoritative_givens,
    normalize_fitb_item,
    normalize_text,
    numerical_instance_contract,
)
from app.offline.pipeline import PipelineCancelled

logger = get_logger("VALIDATOR")
from app.llm.math_strings import MATH_JSON_CONTRACT


from .questions import (
    _build_by_id, _iter_questions, _question_result_text, _question_task_text, _rebuild_fitb_word_bank,
)
from .contracts import (
    _assigned_source_chunks,
)
from .legacy.validation import (
    validate_exam,
)
from . import runtime as _runtime


def _plan_items_by_slot(plans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("slot_id") or ""): item
        for plan in plans or []
        for values in (plan.get("items") or {}).values()
        for item in values or []
        if item.get("slot_id")
    }


def persist_difficulty_observations(
    plans: list[dict[str, Any]], report: dict[str, Any], exam: dict[str, Any],
) -> list[dict[str, Any]]:
    """Persist Luna's observation keyed to the current question fingerprint.

    Also drops any prior observation whose fingerprint no longer matches, so a
    repaired or regenerated question can never inherit an old judgment.
    """
    from app.online.lifecycle import question_fingerprint

    plan_by_slot = _plan_items_by_slot(plans)
    by_id = _build_by_id(exam.get("questions") or {})
    for verdict in report.get("verdicts") or []:
        qid = str(verdict.get("question_id") or "")
        entry = by_id.get(qid)
        if entry is None:
            continue
        question = entry[2]
        slot_id = str(question.get("slot_id") or "")
        plan_item = plan_by_slot.get(slot_id)
        if plan_item is None:
            continue
        fingerprint = question_fingerprint(question)
        observation = dict(verdict.get("difficulty_observation") or {})
        previous = plan_item.get("difficulty_observation") or {}
        if previous and previous.get("question_fingerprint") != fingerprint:
            plan_item.pop("difficulty_observation", None)
            previous = {}
        if not observation:
            continue
        observation["question_fingerprint"] = fingerprint
        plan_item["difficulty_observation"] = observation
        actual = str(observation.get("actual_difficulty") or "")
        if actual in {"easy", "medium", "hard"}:
            plan_item["actual_difficulty"] = actual
            plan_item["difficulty"] = actual
        deviation = verdict.get("difficulty_deviation") or {}
        if deviation.get("status") == "accepted_source_constraint":
            plan_item["difficulty_status"] = "accepted_source_constraint"
            plan_item["difficulty_design"] = {
                **dict(plan_item.get("difficulty_design") or {}),
                "status": "accepted_source_constraint",
                "observed": actual,
                "reason": deviation.get("reason") or observation.get("reason") or "",
            }
    return plans


def _plan_context_for_model(
    state: Dict[str, Any], model_number: int
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    plan_by_slot: Dict[str, Dict[str, Any]] = {}
    assessment_profile = state.get("assessment_profile") or {}
    type_profiles = assessment_profile.get("question_types") or {}
    # Plans are canonical and ordered snapshots. Use only the current snapshot
    # for this model so stale task identities cannot leak into revalidation.
    model_plans = [
        plan for plan in (state.get("plans") or [])
        if int(plan.get("model_number") or 0) == int(model_number)
    ]
    if model_plans:
        plan = model_plans[-1]
        for qtype, values in (plan.get("items") or {}).items():
            for item in values or []:
                sid = str(item.get("slot_id") or "")
                if sid:
                    type_profile = type_profiles.get(qtype) or {}
                    plan_by_slot[sid] = {
                        **item,
                        "assessment_guidance": {
                            "domains": assessment_profile.get("domains") or [],
                            "educational_level": assessment_profile.get("educational_level") or "unspecified",
                            "assessment_style": assessment_profile.get("assessment_style") or "",
                            "question_type": qtype,
                            "support": type_profile.get("support") or "supported",
                            "task_operations": type_profile.get("task_operations") or [],
                            "form_guidance": type_profile.get("form_guidance") or "",
                            "unsupported_visual_tasks": assessment_profile.get("unsupported_visual_tasks") or [],
                        },
                    }
    source_by_chunk = {
        str(chunk.get("child_id")): str(chunk.get("content") or "")
        for chunk in state.get("retrieved_chunks") or []
        if chunk.get("child_id")
    }
    return plan_by_slot, source_by_chunk


def _validate_with_context(
    exam: Dict[str, Any],
    question_ids: Optional[Iterable[str]],
    plan_by_slot: Dict[str, Dict[str, Any]],
    source_by_chunk: Dict[str, str],
    client: Optional[Any] = None,
    validation_role: str = "general",
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if question_ids is not None:
        kwargs["question_ids"] = question_ids
    if plan_by_slot or source_by_chunk:
        kwargs["plan_by_slot"] = plan_by_slot
        kwargs["source_by_chunk"] = source_by_chunk
    if client is not None:
        kwargs["client"] = client
    # Keep the historical call shape for the default role.  This also lets
    # lightweight test/dry-run validators that implement the old signature
    # continue to stand in for the general validator.
    if validation_role != "general":
        kwargs["validation_role"] = validation_role
    return _runtime.validate_exam(exam, **kwargs)


def _quantitative_question_ids(exam: Dict[str, Any], plan_by_slot: Dict[str, Dict[str, Any]]) -> list[str]:
    modes = {"calculation", "symbolic_manipulation", "derivation_relationship",
             "law_or_theorem_application", "optimization"}
    operation_words = ("calculate", "compute", "solve", "derive", "balance", "optimize",
                       "evaluate", "convert", "determine_value", "write_equation")
    result: list[str] = []
    for qtype, _container, item in _iter_questions(exam.get("questions") or {}):
        if qtype == "fill_in_the_blank":
            continue
        qid = str(item.get("question_id") or "")
        plan = plan_by_slot.get(str(item.get("slot_id") or "")) or {}
        mode = str(plan.get("reasoning_mode") or "").casefold()
        operation = str(plan.get("task_operation") or "").casefold()
        quantitative = (qtype in {"equation", "word_problem"} or mode in modes
                        or any(word in operation for word in operation_words)
                        or (qtype == "mcq" and bool(re.search(r"[=<>≤≥]|\d", _question_task_text(qtype, item)))))
        if qid and quantitative:
            result.append(qid)
    return result


def _merge_role_reports(general: Dict[str, Any], quantitative: Dict[str, Any]) -> Dict[str, Any]:
    quant_by_id = {str(v.get("question_id")): v for v in quantitative.get("verdicts") or []}
    merged = []
    for verdict in general.get("verdicts") or []:
        quant = quant_by_id.get(str(verdict.get("question_id") or ""))
        findings = []
        for role, finding in (("general", verdict), ("stem_quantitative", quant)):
            if finding is not None and finding.get("action") != "PASS":
                findings.append({
                    "validator": role,
                    "type": finding.get("action"),
                    "failure_code": finding.get("failure_code"),
                    "reason": finding.get("reason") or "",
                    "expected_fix": finding.get("expected_fix") or "",
                    "fields_to_fix": list(finding.get("fields_to_fix") or []),
                    "verdict": deepcopy(finding),
                })
        if quant is None or quant.get("action") == "PASS":
            chosen = dict(verdict)
        elif verdict.get("action") == "PASS":
            chosen = dict(quant)
        else:
            chosen = dict(quant)
            chosen["additional_general_issue"] = {key: verdict.get(key) for key in
                ("action", "failure_code", "fields_to_fix", "reason", "expected_fix")}
        chosen["required_fixes"] = findings
        if len(findings) > 1:
            chosen["validation_role"] = "combined"
            chosen["fields_to_fix"] = sorted({
                field for finding in findings for field in finding["fields_to_fix"]
            })
        merged.append(chosen)
    return {**general, "verdicts": merged,
            "all_pass": all(v.get("action") == "PASS" for v in merged),
            "coverage": [*(general.get("coverage") or []), *(quantitative.get("coverage") or [])],
            "warnings": [*(general.get("warnings") or []), *(quantitative.get("warnings") or [])],
            "general_review": general,
            "quantitative_review": quantitative}


def _regenerate_plan_slot(
    state: Dict[str, Any],
    exam: Dict[str, Any],
    model_number: int,
    question_id: str,
    plan_by_slot: Dict[str, Dict[str, Any]],
    source_by_chunk: Dict[str, str],
) -> tuple[bool, list[str]]:
    """Regenerate one replaced plan through its normal type-specific generator."""
    _ensure_generation_active(state)
    from app.online.exam_builder import (
        _generate_fitb_type,
        _generate_obj_bundle,
        _generate_type_from_plan,
        _infer_question_type,
        _remember_question,
    )

    questions = exam.get("questions") or {}
    entry = _build_by_id(questions).get(question_id)
    if entry is None:
        return False, [f"repair: unknown question_id {question_id}"]
    qtype, _container, current = entry
    sid = str(current.get("slot_id") or "")
    plan = plan_by_slot.get(sid) or {}
    if not plan:
        return False, [f"repair: no plan found for duplicate slot {sid}"]

    sources = _assigned_source_chunks(plan, source_by_chunk)
    if not sources:
        return False, [f"repair: no assigned evidence found for duplicate slot {sid}"]
    context = "\n\n".join(
        f"[source_chunk_id={chunk_id}]\n{text}" for chunk_id, text in sources.items()
    )
    generation_plan = {**plan, "assigned_sources": sources}
    if plan.get("task_reused") and not generation_plan.get("previous_task_questions"):
        related_slots = set(str(value) for value in plan.get("forced_duplicate_slot_ids") or [])
        generation_plan["previous_task_questions"] = [
            {
                "context_role": "previous_question_from_reused_task",
                "question_id": other_item.get("question_id"),
                "slot_id": other_item.get("slot_id"),
                "task_id": (plan_by_slot.get(str(other_item.get("slot_id") or "")) or {}).get(
                    "task_id"
                ),
                "question_text": _question_task_text(other_type, other_item),
                "answer_pattern": _question_result_text(other_type, other_item),
            }
            for other_type, _other_container, other_item in _iter_questions(questions)
            if str(other_item.get("slot_id") or "") in related_slots
            and str(other_item.get("question_id") or "") != question_id
        ]
    within_model = [
        item for _other_type, _other_container, item in _iter_questions(questions)
        if str(item.get("question_id") or "") != question_id
    ]
    seen: set[tuple[str, str]] = set()
    for item in within_model:
        _remember_question(_infer_question_type(item), item, seen)
    difficulty = str(state.get("difficulty") or "mix")
    language = str(state.get("document_language") or "en")
    warnings: list[str] = []

    if qtype in {"mcq", "true_false"}:
        bundle, warnings = _generate_obj_bundle(
            {qtype: [generation_plan]}, context, difficulty, model_number,
            within_model, seen, [], language=language,
            global_plan_by_slot=plan_by_slot,
        )
        candidates = list(bundle.get(qtype) or [])
    elif qtype == "fill_in_the_blank":
        section, warnings = _generate_fitb_type(
            1, [generation_plan], context, difficulty, model_number,
            within_model, [], language=language,
            global_plan_by_slot=plan_by_slot,
        )
        candidates = list((section or {}).get("items") or [])
    else:
        candidates, warnings = _generate_type_from_plan(
            qtype, [generation_plan], context, difficulty, model_number,
            seen, within_model, [], language=language,
            global_plan_by_slot=plan_by_slot,
        )
    if len(candidates) != 1:
        return False, [*warnings, f"repair: normal generator did not replace slot {sid}"]

    replacement = candidates[0]
    replacement["question_id"] = question_id
    replacement["slot_id"] = sid
    current.clear()
    current.update(replacement)
    if qtype == "fill_in_the_blank":
        section = questions.get(qtype)
        if not isinstance(section, dict) or not _rebuild_fitb_word_bank(section):
            return False, [*warnings, f"repair: FITB bank rebuild failed for slot {sid}"]
    return True, warnings


def _question_type_from_slot(slot_id: str, plan_item: Dict[str, Any] | None = None) -> str:
    qtype = str((plan_item or {}).get("question_type") or "")
    if qtype:
        return qtype
    match = re.match(r"m\d+_(.+)_\d+$", slot_id)
    return match.group(1) if match else "unknown"

from .state import _ensure_generation_active

