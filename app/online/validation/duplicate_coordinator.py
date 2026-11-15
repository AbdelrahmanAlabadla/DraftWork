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
    _build_by_id,
)
from .context import (
    _plan_context_for_model,
)
from .contracts import (
    _plan_source_ids,
)
from .manifest import (
    _merge_repair_manifest,
)


def _integrate_duplicate_validation(
    state: Dict[str, Any], generated: list[Dict[str, Any]], plans: list[Dict[str, Any]],
    reports_by_model: Dict[Any, Dict[str, Any]], repair_manifest: list[Dict[str, Any]],
    validated: list[int], warnings: list[str],
) -> tuple[list[Dict[str, Any]], list[int], list[str], Dict[str, Any]]:
    """Convert pair findings into stable normal validation/repair issues."""
    from app.online.duplicate_pipeline import scan_exam_duplicates

    try:
        result = scan_exam_duplicates({
            **state, "generated_exams": generated, "plans": plans,
            "validation_reports": list(reports_by_model.values()),
            "repair_manifest": repair_manifest, "validated_models": validated,
            "warnings": warnings,
        })
    except Exception as exc:
        # Duplicate review is a required cross-question validation.  Treat an
        # operational failure exactly like an unavailable ordinary validator:
        # retain the accepted questions and plans, but prevent the model from
        # being represented as fully validated or assembled as complete.
        reason = f"Duplicate Validator operational failure: {type(exc).__name__}: {exc}"
        blocked: list[Dict[str, Any]] = []
        exams = {int(exam.get("model_number") or 0): exam for exam in generated}
        for mn, exam in exams.items():
            report = reports_by_model.get(mn)
            if report is None:
                continue
            plan_by_slot, _sources = _plan_context_for_model({**state, "plans": plans}, mn)
            for qid, (qtype, _container, item) in _build_by_id(exam.get("questions") or {}).items():
                slot_id = str(item.get("slot_id") or "")
                issue_id = f"{qid}-I-duplicate-operational-1"
                issue = {
                    "issue_id": issue_id, "validator": "duplicate",
                    "validation_status": "VALIDATION_BLOCKED",
                    "failure_code": "duplicate_validation_operational_error",
                    "fields_to_fix": [], "reason": reason,
                    "expected_fix": "Retry the Duplicate Validator; do not alter content.",
                }
                verdict = {
                    "question_id": qid, "action": "UNVALIDATED",
                    "validation_status": "VALIDATION_BLOCKED",
                    "validation_role": "duplicate",
                    "failure_code": issue["failure_code"], "fields_to_fix": [],
                    "reason": reason, "expected_fix": issue["expected_fix"],
                    "required_fixes": [issue], "role_statuses": {"duplicate": "ERROR"},
                }
                previous = next((value for value in report.get("verdicts") or []
                                 if str(value.get("question_id") or "") == qid), None)
                if previous is not None:
                    verdict["role_statuses"] = {
                        **deepcopy(previous.get("role_statuses") or {}), "duplicate": "ERROR",
                    }
                    report["verdicts"].remove(previous)
                report.setdefault("verdicts", []).append(verdict)
                plan_item = plan_by_slot.get(slot_id) or {}
                blocked.append({
                    "model_number": mn, "slot_id": slot_id, "question_id": qid,
                    "question_type": qtype, "status": "validation_blocked",
                    "task_id": plan_item.get("task_id"),
                    "fitb_entry_id": plan_item.get("fitb_entry_id"),
                    "evidence_ids": _plan_source_ids(plan_item),
                    "failure_code": issue["failure_code"], "failure_reason": reason,
                    "repair_category": "operational", "repair_type": "none",
                    "validation_role": "duplicate", "fields_to_fix": [],
                    "required_fixes": [issue], "verdict": verdict,
                })
            report["all_pass"] = False
            if mn in validated:
                validated.remove(mn)
        warnings.append(reason)
        return _merge_repair_manifest(repair_manifest, blocked), validated, warnings, {
            "operational_error": reason, "pending_duplicate_groups": [],
            "duplicate_reports": list(state.get("duplicate_reports") or []),
            "duplicate_findings": list(state.get("duplicate_findings") or []),
        }
    targets: list[Dict[str, Any]] = []
    exams = {int(exam.get("model_number") or 0): exam for exam in generated}
    for group in result.get("pending_duplicate_groups") or []:
        mn = int(group.get("model_number") or 1)
        exam, report = exams.get(mn) or {}, reports_by_model.get(mn)
        if report is None:
            continue
        plan_by_slot, _sources = _plan_context_for_model({**state, "plans": plans}, mn)
        for qid in group.get("replace_question_ids") or []:
            entry = _build_by_id(exam.get("questions") or {}).get(str(qid))
            if entry is None:
                continue
            qtype, _container, item = entry
            slot_id = str(item.get("slot_id") or "")
            conflicts = sorted(str(value) for value in group.get("question_ids") or []
                               if str(value) and str(value) != str(qid))
            pair_ids = sorted(
                f"{min(str(edge.get('left')), str(edge.get('right')))}::{max(str(edge.get('left')), str(edge.get('right')))}"
                for edge in group.get("edges") or []
                if str(qid) in {str(edge.get("left")), str(edge.get("right"))}
            )
            issue_id = f"{qid}-I-{hashlib.sha256((str(qid) + '|duplicate|' + '|'.join(pair_ids)).encode()).hexdigest()[:12]}-1"
            issue = {
                "issue_id": issue_id, "validator": "duplicate",
                "validation_status": "FAIL", "failure_code": "duplicate_question",
                "fields_to_fix": [],
                "reason": "The question duplicates or excessively overlaps another validated question.",
                "expected_fix": "Select an authorized distinct variation or catalog task and regenerate transactionally.",
                "conflicting_question_ids": conflicts, "duplicate_pair_ids": pair_ids,
                "duplicate_group_id": group.get("group_id"),
            }
            verdict = {
                "question_id": str(qid), "action": "FIX_QUESTION_AND_ANSWER",
                "validation_status": "FAIL", "validation_role": "duplicate",
                "failure_code": "duplicate_question", "fields_to_fix": [],
                "reason": issue["reason"], "expected_fix": issue["expected_fix"],
                "required_fixes": [issue], "conflicting_question_ids": conflicts,
                "duplicate_pair_ids": pair_ids, "role_statuses": {"duplicate": "FAIL"},
            }
            previous = next((value for value in report.get("verdicts") or []
                             if str(value.get("question_id") or "") == str(qid)), None)
            if previous is not None:
                verdict["role_statuses"] = {
                    **deepcopy(previous.get("role_statuses") or {}), "duplicate": "FAIL",
                }
                report["verdicts"].remove(previous)
            report.setdefault("verdicts", []).append(verdict)
            report["all_pass"] = False
            plan_item = plan_by_slot.get(slot_id) or {}
            targets.append({
                "model_number": mn, "slot_id": slot_id, "question_id": str(qid),
                "question_type": qtype, "status": "invalid",
                "task_id": plan_item.get("task_id"),
                "fitb_entry_id": plan_item.get("fitb_entry_id"),
                "evidence_ids": _plan_source_ids(plan_item),
                "failure_code": "duplicate_question", "failure_reason": issue["reason"],
                "repair_category": "duplicate_conflict", "repair_type": "replace_assignment",
                "preferred_recovery": "different authorized variation, then catalog replacement",
                "validation_role": "duplicate", "fields_to_fix": [],
                "required_fixes": [issue], "verdict": verdict,
                "conflicting_question_ids": conflicts, "duplicate_pair_ids": pair_ids,
                "duplicate_group_id": group.get("group_id"),
            })
            if mn in validated:
                validated.remove(mn)
    if targets:
        repair_manifest = _merge_repair_manifest(repair_manifest, targets)
    return repair_manifest, validated, list(result.get("warnings") or warnings), result
