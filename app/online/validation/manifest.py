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


from .constants import (
    RECOVERY_ACTIONS,
)
from .questions import (
    _build_by_id, _rebuild_fitb_word_bank,
)
from .context import _question_type_from_slot


def _manifest_from_report(
    report: Dict[str, Any], exam: Dict[str, Any],
    plan_by_slot: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Create one actionable target for every invalid or absent requested slot."""
    targets: list[Dict[str, Any]] = []
    by_id = _build_by_id(exam.get("questions") or {})
    for verdict in report.get("verdicts") or []:
        if verdict.get("action") not in RECOVERY_ACTIONS or (
            verdict.get("action") == "UNVALIDATED"
            and verdict.get("validation_status") != "VALIDATION_BLOCKED"
        ):
            continue
        qid = str(verdict.get("question_id") or "")
        entry = by_id.get(qid)
        if entry is None:
            continue
        qtype, _container, item = entry
        slot_id = str(item.get("slot_id") or "")
        plan_item = plan_by_slot.get(slot_id) or {}
        category, repair_type = (
            "awaiting_task_alignment_decision", "recovery_decision_pending"
        )
        if verdict.get("validation_status") == "VALIDATION_BLOCKED":
            block_category = str(verdict.get("block_category") or "operational")
            category = f"{block_category}_validation_block"
            repair_type = "replace_assignment"
        required_fixes = deepcopy(verdict.get("required_fixes") or [])
        seen_issue_bases: Dict[str, int] = {}
        required_fixes.sort(key=lambda issue: (
            str(issue.get("validator") or ""), str(issue.get("failure_code") or ""),
            tuple(sorted(map(str, issue.get("fields_to_fix") or []))),
            str(issue.get("reason") or ""),
        ))
        for issue in required_fixes:
            if issue.get("issue_id"):
                continue
            base = "|".join((
                qid, str(issue.get("validator") or "deterministic"),
                str(issue.get("failure_code") or "other"),
                ",".join(sorted(map(str, issue.get("fields_to_fix") or []))),
            ))
            seen_issue_bases[base] = seen_issue_bases.get(base, 0) + 1
            digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]
            issue["issue_id"] = f"{qid}-I-{digest}-{seen_issue_bases[base]}"
        verdict["required_fixes"] = deepcopy(required_fixes)
        grounding_scopes = {
            str(issue.get("defect_scope") or "uncertain")
            for issue in required_fixes
            if str(issue.get("validator") or "") == "grounding"
            and str(issue.get("failure_code") or "") in {
                "contradicted", "unsupported", "insufficient_evidence",
            }
        }
        if "assignment" in grounding_scopes:
            category, repair_type = "task_or_evidence_invalid", "replace_assignment"
        elif grounding_scopes == {"question"}:
            category, repair_type = "generated_content_not_grounded", "regenerate_missing_slot"
        if str(verdict.get("failure_code") or "") in {
            "missing_required_given", "missing_required_givens", "invalid_required_givens",
        }:
            diagnosis = " ".join((
                str(verdict.get("reason") or ""), str(verdict.get("expected_fix") or "")
            )).casefold()
            if any(marker in diagnosis for marker in (
                "unsupported", "not supported", "not in evidence", "not grounded", "impossible",
            )):
                category, repair_type = "task_or_evidence_invalid", "replace_assignment"
        category = "awaiting_task_alignment_decision"
        repair_type = "recovery_decision_pending"
        targets.append({
            "model_number": report.get("model_number"),
            "slot_id": slot_id,
            "question_id": qid,
            "question_type": qtype,
            "status": "invalid",
            "task_id": plan_item.get("task_id"),
            "fitb_entry_id": plan_item.get("fitb_entry_id"),
            "evidence_ids": list(plan_item.get("evidence_chunk_ids") or []) or [
                value for value in [plan_item.get("evidence_chunk_id") or plan_item.get("source_chunk_id")]
                if value
            ],
            "failure_code": verdict.get("failure_code") or "validation_failure",
            "failure_reason": verdict.get("reason") or verdict.get("expected_fix") or "",
            "repair_category": category,
            "repair_type": repair_type,
            "what_is_wrong": verdict.get("reason") or "",
            "what_must_change": verdict.get("expected_fix") or "",
            "preferred_recovery": "task_alignment_recovery_decision",
            "validation_role": verdict.get("validation_role") or "general",
            "fields_to_fix": list(verdict.get("fields_to_fix") or []),
            "required_fixes": required_fixes,
            "repair_instruction_conflicts": deepcopy(
                verdict.get("repair_instruction_conflicts") or []
            ),
            "role_statuses": deepcopy(verdict.get("role_statuses") or {}),
            "block_category": verdict.get("block_category"),
            "blocked_roles": deepcopy(verdict.get("blocked_roles") or []),
            "verdict": deepcopy(verdict),
        })
    for missing in report.get("missing_slots") or []:
        slot_id = str(missing.get("slot_id") or "")
        targets.append({
            "model_number": report.get("model_number"),
            "slot_id": slot_id,
            "question_id": None,
            "question_type": missing.get("question_type") or _question_type_from_slot(slot_id),
            "status": "missing",
            "task_id": missing.get("task_id"),
            "fitb_entry_id": missing.get("fitb_entry_id"),
            "evidence_ids": [missing.get("evidence_chunk_id")] if missing.get("evidence_chunk_id") else [],
            "failure_code": "missing_requested_slot",
            "failure_reason": missing.get("recovery_requirement") or "Requested slot was not generated.",
            "repair_category": "awaiting_task_alignment_decision",
            "repair_type": "recovery_decision_pending",
            "what_is_wrong": "The requested slot is absent from Generator output.",
            "what_must_change": "Generate and validate the exact assigned slot.",
            "preferred_recovery": "task_alignment_recovery_decision",
            "validation_role": "general",
            "verdict": None,
        })
    return targets


def _merge_repair_manifest(
    old: list[Dict[str, Any]], fresh: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Keep unresolved slots until a passing candidate explicitly resolves them."""
    old_by_key = {
        (int(item.get("model_number") or 0), str(item.get("slot_id") or "")): deepcopy(item)
        for item in old if item.get("slot_id")
    }
    fresh_by_key = {
        (int(item.get("model_number") or 0), str(item.get("slot_id") or "")): deepcopy(item)
        for item in fresh if item.get("slot_id")
    }
    merged: list[Dict[str, Any]] = []
    for key in sorted(set(old_by_key) | set(fresh_by_key)):
        if key in fresh_by_key:
            current = fresh_by_key[key]
            previous = old_by_key.get(key) or {}
            previous_status = str(previous.get("status") or "")
            if previous_status in {
                "validation_pending", "validation_blocked", "completion_pending", "exhausted",
                "terminal_failure",
            }:
                # Fresh Validator findings are audit evidence, not authority to
                # reopen a slot whose persisted lifecycle is pending or terminal.
                current["status"] = previous_status
                current["validation_candidate_id"] = previous.get("validation_candidate_id")
                for field in (
                    "failure_code", "failure_reason", "repair_category",
                    "repair_type", "verdict", "repair_conflict",
                ):
                    if field in previous:
                        current[field] = deepcopy(previous[field])
            current["history"] = list(previous.get("history") or [])
            if previous.get("recovery_v3"):
                current["recovery_v3"] = deepcopy(previous["recovery_v3"])
            if previous.get("recovery_decision_audit"):
                current["recovery_decision_audit"] = deepcopy(
                    previous["recovery_decision_audit"]
                )
            if current["history"] and current["history"][-1].get("new_failure_reason"):
                current["failure_code"] = previous.get("failure_code", current.get("failure_code"))
                current["failure_reason"] = previous.get("failure_reason", current.get("failure_reason"))
                current["repair_category"] = previous.get("repair_category", current.get("repair_category"))
                current["repair_type"] = previous.get("repair_type", current.get("repair_type"))
                current["verdict"] = previous.get("verdict", current.get("verdict"))
            merged.append(current)
        elif old_by_key[key].get("status") != "resolved":
            # A selective validation cannot silently erase another slot.
            merged.append(old_by_key[key])
    return merged


def _placeholder_question(qtype: str, slot_id: str, question_id: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {"slot_id": slot_id, "question_id": question_id}
    if qtype == "mcq":
        base.update(question="", options={}, correct_answer="")
    elif qtype == "true_false":
        base.update(statement="", answer=False)
    elif qtype == "fill_in_the_blank":
        base.update(question="_____", answers=[], distractors=[])
    elif qtype == "definition":
        base.update(term="", reference_answer="")
    elif qtype in {"why", "essay"}:
        base.update(question="", reference_answer="")
    elif qtype == "equation":
        base.update(equation="", solution_steps=[], final_answer="")
    else:
        base.update(question="", solution_steps=[], final_answer="")
    return base


def _put_question(exam: Dict[str, Any], qtype: str, item: Dict[str, Any]) -> None:
    """Insert/replace a slot while preserving its requested section and slot identity."""
    questions = exam.setdefault("questions", {})
    section = questions.get(qtype)
    if qtype == "fill_in_the_blank":
        if not isinstance(section, dict):
            section = {"items": [], "word_bank": []}
            questions[qtype] = section
        values = section.setdefault("items", [])
    else:
        if not isinstance(section, list):
            section = []
            questions[qtype] = section
        values = section
    slot_id = str(item.get("slot_id") or "")
    for index, current in enumerate(values):
        if str(current.get("slot_id") or "") == slot_id:
            values[index] = deepcopy(item)
            break
    else:
        values.append(deepcopy(item))
    values.sort(key=lambda value: str(value.get("slot_id") or ""))
    if qtype == "fill_in_the_blank":
        _rebuild_fitb_word_bank(section)
