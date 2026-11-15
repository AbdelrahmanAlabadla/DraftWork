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


from .. import runtime as _runtime
from ..constants import (
    REPAIR_ACTIONS,
    REPAIR_MAX_TOKENS,
    REPAIR_REASONING,
    REPAIR_TEMPERATURE,
    TASK_REPLACEMENT_FAILURES,
)
from ..questions import (
    _allowed_canonical, _build_by_id, _canonical_values, _fitb_word_bank_structure_valid, _normalize, _rebuild_fitb_word_bank, _simple_option_signature,
)
from ..contracts import (
    REPAIR_SYSTEM_PROMPT, STEM_QUANTITATIVE_REPAIR_SYSTEM_PROMPT, _assigned_source_chunks, build_repair_prompt,
)

logger = get_logger("VALIDATION_LEGACY_REPAIR")

def _fields_changed_disallowed(
        qtype: str,
        orig: Dict[str, Any],
        new: Dict[str, Any],
        allowed: Set[str],
) -> List[str]:
    """Return canonical field names that changed although they were not allowed."""
    orig_vals = _canonical_values(qtype, orig)
    new_vals = _canonical_values(qtype, new)
    violations: List[str] = []
    for key in sorted(set(orig_vals) | set(new_vals)):
        if key in allowed:
            continue
        if key in orig_vals and key not in new_vals:
            violations.append(key)
        elif orig_vals.get(key) != new_vals.get(key):
            violations.append(key)
    return violations


def _repair_fields_with_dependencies(
    qtype: str, fields: Set[str], repair_role: str,
) -> Set[str]:
    """Declare derived quantitative fields that must be rebuilt in one patch."""
    expanded = set(fields)
    if repair_role != "stem_quantitative" or expanded == {"solution_certificate"}:
        return expanded
    if qtype == "mcq":
        if expanded & {"question", "options", "correct_answer"}:
            expanded.add("solution_certificate")
            if expanded & {"question", "options"}:
                expanded.update({"options", "correct_answer"})
    elif qtype == "true_false" and expanded & {"statement", "answer"}:
        expanded.add("solution_certificate")
    elif qtype in {"equation", "word_problem"} and expanded & {
        "equation", "question", "solution_steps", "final_answer",
    }:
        expanded.update({"solution_steps", "final_answer", "solution_certificate"})
    return expanded


def _repair_role_for(
    qtype: str, verdict: Dict[str, Any], *, domain: str = "STEM",
) -> str:
    """Route mathematical defects even when validation_role was lost upstream."""
    if str(domain or "").upper() != "STEM":
        return "general"
    if str(verdict.get("validation_role") or "") == "stem_quantitative":
        return "stem_quantitative"
    code = str(verdict.get("failure_code") or "")
    fields = set(_allowed_canonical(
        qtype, verdict.get("fields_to_fix") or [], str(verdict.get("action") or "")
    ))
    if code in {
        "incomplete_solution", "incorrect_calculation", "incorrect_result",
        "invalid_solution_certificate", "invalid_or_extraneous_solution",
        "missing_units",
    }:
        return "stem_quantitative"
    if qtype in {"equation", "word_problem"} and fields & {
        "equation", "solution_steps", "final_answer", "solution_certificate",
    }:
        return "stem_quantitative"
    return "general"


def repair_exam(
        exam: Dict[str, Any], verdicts: List[Dict[str, Any]], client: Optional[Any] = None,
        plan_by_slot: Optional[Dict[str, Dict[str, Any]]] = None,
        source_by_chunk: Optional[Dict[str, str]] = None,
        repair_role: str = "general",
) -> Dict[str, Any]:
    """Repair the invalid questions of one exam with ONE batched LLM call.

    Write-back is strictly ID-keyed: an echoed ``question_id`` must exist, be
    unique in the response, and produce a normalized question whose changed
    canonical fields are all inside ``fields_to_fix``. Any violation rejects that
    repair (the original question is kept). Returns
    ``{"warnings", "repaired_ids", "failed_ids"}`` and mutates the exam in place.
    """
    warnings: List[str] = []
    repair_metadata_by_id: Dict[str, Dict[str, Any]] = {}
    questions = exam.get("questions") or {}
    invalid = [v for v in verdicts if v.get("action") in REPAIR_ACTIONS]
    if not invalid:
        return {
            "warnings": warnings,
            "sent_ids": [],
            "repaired_ids": [],
            "failed_ids": [],
        }

    by_id = _build_by_id(questions)
    verdict_by_id = {str(v.get("question_id")): v for v in invalid}

    payload: List[Dict[str, Any]] = []
    for v in invalid:
        qid = str(v.get("question_id") or "")
        entry = by_id.get(qid)
        if entry is None:
            warnings.append(f"repair: no question found for {qid}; skipped")
            continue
        qtype, container, item = entry
        p = dict(item)
        if qtype == "fill_in_the_blank" and isinstance(container, dict):
            p["word_bank"] = container.get("word_bank") or []
        plan_item = (plan_by_slot or {}).get(str(item.get("slot_id") or ""))
        source_chunks = _assigned_source_chunks(plan_item, source_by_chunk)
        rejected_distractors: list[str] = []
        required_repair: str | None = None
        if qtype == "mcq" and str(v.get("action") or "") == "FIX_OPTIONS":
            reason_text = " ".join((
                str(v.get("reason") or ""), str(v.get("expected_fix") or "")
            )).casefold()
            semantic_distractor_failure = any(marker in reason_text for marker in (
                "distractor", "implausible", "irrelevant", "unrelated", "same-topic",
                "misconception", "compete",
            ))
            if semantic_distractor_failure:
                correct_label = str(p.get("correct_answer") or "").strip().upper()
                rejected_distractors = [
                    str(text).strip() for label, text in (p.get("options") or {}).items()
                    if str(label).strip().upper() != correct_label and str(text).strip()
                ]
                required_repair = "replace_all_rejected_distractors"
        allowed = _allowed_canonical(
            qtype, v.get("fields_to_fix") or [], v.get("action") or "FIX_ANSWER"
        )
        allowed = _repair_fields_with_dependencies(qtype, allowed, repair_role)
        if not allowed:
            warnings.append(
                f"repair: {qid} has no authorized canonical patch fields; rejected"
            )
            continue
        # The manifest is the source of patch authority. Record deterministic
        # dependency expansion explicitly before constructing the JSON schema.
        v["fields_to_fix"] = sorted(allowed)
        payload.append(
            {
                "question_id": qid,
                "action": v.get("action", "FIX_ANSWER"),
                "fields_to_fix": v.get("fields_to_fix") or [],
                "reason": v.get("reason") or "",
                "expected_fix": v.get("expected_fix") or "",
                "required_fixes": deepcopy(v.get("required_fixes") or []),
                "expected_results": list(v.get("expected_results") or []),
                "independent_solution": v.get("independent_solution"),
                "first_incorrect_step": v.get("first_incorrect_step"),
                "replacement_mode": _requires_plan_replacement(v),
                "rejected_distractors": rejected_distractors,
                "required_repair": required_repair,
                "previous_repair_rejection": v.get("previous_repair_rejection") or "",
                "previous_attempts": list(v.get("repair_attempt_history") or [])[-3:],
                "authorized_patch_fields": sorted(allowed),
                "question": p,
                "plan": plan_item or {},
                "source_chunks": source_chunks,
            }
        )

    if not payload:
        return {
            "warnings": warnings,
            "sent_ids": [],
            "repaired_ids": [],
            "failed_ids": [str(v.get("question_id")) for v in invalid],
        }

    sent_ids = [item["question_id"] for item in payload]

    repair_stage = "stem_quantitative_repairer" if repair_role == "stem_quantitative" else (
        "options_repairer" if invalid and all(
            str(item.get("action") or "") == "FIX_OPTIONS" for item in invalid
        ) else "answer_repairer" if invalid and all(
            str(item.get("action") or "") == "FIX_ANSWER" for item in invalid
        ) else "question_repairer"
    )
    llm = client or _runtime.create_llm_client(
        local_reasoning=REPAIR_REASONING,
        deepseek_model="deepseek-flash",
        stage=repair_stage,
    )
    require_repair_metadata = all(
        str(verdict_by_id.get(item["question_id"], {}).get("validation_status") or "") == "FAIL"
        for item in payload
    )
    user_prompt = build_repair_prompt(payload, include_metadata=require_repair_metadata)
    try:
        repair_model_items = [
            (
                item["question_id"], by_id[item["question_id"]][0],
                list(item["authorized_patch_fields"]),
            )
            for item in payload
        ]
        raw = _runtime.request_structured(
            llm,
            user_prompt,
            response_model=repair_response_model(
                repair_model_items, include_metadata=require_repair_metadata,
            ),
            schema_name="question_repairs",
            agent="repairer",
            item_count=len(payload),
            expected_ids=sent_ids,
            expected_id_field="question_id",
            local_json=True,
            system_prompt=(STEM_QUANTITATIVE_REPAIR_SYSTEM_PROMPT
                           if repair_role == "stem_quantitative" else REPAIR_SYSTEM_PROMPT),
            temperature=REPAIR_TEMPERATURE,
            max_tokens=REPAIR_MAX_TOKENS,
        )
    except Exception as exc:
        logger.exception(
            "Question repair call failed | role=%s | question_ids=%s",
            repair_role, sent_ids,
        )
        warnings.append(f"repair: batched repair LLM call failed ({exc})")
        return {
            "warnings": warnings,
            "sent_ids": sent_ids,
            "repaired_ids": [],
            "failed_ids": [str(v.get("question_id")) for v in invalid],
        }

    repairs = raw.get("repairs") if isinstance(raw, dict) else None
    if not isinstance(repairs, list):
        warnings.append("repair: LLM returned no usable repair array")
        return {
            "warnings": warnings,
            "sent_ids": sent_ids,
            "repaired_ids": [],
            "failed_ids": [str(v.get("question_id")) for v in invalid],
        }

    accepted: Set[str] = set()
    seen_ids: Set[str] = set()
    for r in repairs:
        if not isinstance(r, dict):
            continue
        qid = str(r.get("question_id") or "").strip()
        if not qid:
            warnings.append("repair: returned item missing question_id; rejected")
            continue
        if qid in seen_ids:
            warnings.append(f"repair: duplicate question_id {qid} in response; rejected")
            continue
        seen_ids.add(qid)
        entry = by_id.get(qid)
        if entry is None:
            warnings.append(f"repair: unknown question_id {qid}; rejected")
            continue
        qtype, container, item = entry
        patch = r.get("patch")
        if not isinstance(patch, dict):
            warnings.append(f"repair: {qid} returned no patch payload; rejected")
            continue
        reported_metadata = r.get("repair_metadata")
        if require_repair_metadata and not isinstance(reported_metadata, dict):
            warnings.append(f"repair: {qid} omitted repair_metadata; rejected")
            continue

        verdict = verdict_by_id.get(qid) or {}
        request_item = next(
            (value for value in payload if value.get("question_id") == qid), {}
        )
        allowed = set(request_item.get("authorized_patch_fields") or [])
        patch_keys = set(map(str, patch))
        unauthorized = sorted(patch_keys - allowed)
        missing = sorted(allowed - patch_keys)
        if unauthorized:
            warnings.append(
                f"repair: {qid} patch contained unauthorized field(s) {unauthorized}; rejected"
            )
            continue
        if missing:
            warnings.append(
                f"repair: {qid} patch omitted required field(s) {missing}; rejected"
            )
            continue

        new_item = deepcopy(item)
        new_item.update(deepcopy(patch))
        proposed_fitb_bank = None

        stashed_id = item.get("question_id")
        stashed_slot_id = item.get("slot_id")
        stashed_fitb_entry_id = item.get("fitb_entry_id")
        normalized = _normalize(qtype, new_item)
        if normalized is None:
            warnings.append(f"repair: {qid} output failed normalization; rejected")
            continue
        if repair_role == "stem_quantitative" and not normalized.get("solution_certificate"):
            warnings.append(
                f"repair: {qid} quantitative repair omitted a valid solution_certificate; rejected"
            )
            continue
        normalized["question_id"] = stashed_id
        if stashed_slot_id:
            normalized["slot_id"] = stashed_slot_id
        if stashed_fitb_entry_id:
            normalized["fitb_entry_id"] = stashed_fitb_entry_id

        rejected_distractors = {
            _simple_option_signature(value)
            for value in request_item.get("rejected_distractors") or []
            if _simple_option_signature(value)
        }
        if qtype == "mcq" and rejected_distractors:
            new_correct = str(normalized.get("correct_answer") or "").strip().upper()
            retained = {
                _simple_option_signature(text)
                for label, text in (normalized.get("options") or {}).items()
                if str(label).strip().upper() != new_correct
                and _simple_option_signature(text) in rejected_distractors
            }
            if retained:
                warnings.append(
                    f"repair: {qid} retained rejected distractor content; rejected"
                )
                continue
        fitb_bank_changed = bool(
            qtype == "fill_in_the_blank"
            and isinstance(container, dict)
            and proposed_fitb_bank is not None
            and proposed_fitb_bank != list(container.get("word_bank") or [])
        )
        if fitb_bank_changed and "word_bank" not in allowed:
            warnings.append(
                f"repair: {qid} modified disallowed field(s) ['word_bank']; rejected"
            )
            continue
        if fitb_bank_changed:
            candidate_section = deepcopy(container)
            candidate_section["word_bank"] = proposed_fitb_bank
            if not _fitb_word_bank_structure_valid(candidate_section):
                warnings.append(f"repair: {qid} returned invalid word_bank; rejected")
                continue
        violations = _fields_changed_disallowed(qtype, item, normalized, allowed)
        if violations:
            warnings.append(
                f"repair: {qid} modified disallowed field(s) {violations}; rejected"
            )
            continue

        original_values = _canonical_values(qtype, item)
        repaired_values = _canonical_values(qtype, normalized)
        actual_changes = {
            field: {
                "previous_value": deepcopy(original_values.get(field)),
                "new_value": deepcopy(repaired_values.get(field)),
            }
            for field in allowed
            if original_values.get(field) != repaired_values.get(field)
        }
        if require_repair_metadata:
            reported_fields = set(map(str, reported_metadata.get("modified_fields") or []))
            change_fields = {
                str(change.get("field") or "")
                for change in (reported_metadata.get("changes") or [])
                if isinstance(change, dict)
            }
            # LLM metadata is advisory. Authorization above used Python's
            # canonical before/after diff, so an incomplete field list or a
            # descriptive previous_value cannot reject otherwise valid content.
            reported_metadata["llm_reported_modified_fields"] = sorted(reported_fields)
            reported_metadata["llm_reported_change_fields"] = sorted(change_fields)
            reported_metadata["modified_fields"] = sorted(actual_changes)
            reported_metadata["changes"] = [
                {"field": field, **values}
                for field, values in actual_changes.items()
            ]
            targeted = set(map(str, reported_metadata.get("targeted_failure_codes") or []))
            expected_codes = {
                str(finding.get("failure_code") or "")
                for finding in verdict.get("required_fixes") or []
                if finding.get("failure_code")
            } or {str(verdict.get("failure_code") or "")}
            if not expected_codes.issubset(targeted):
                warnings.append(
                    f"repair: {qid} repair_metadata omitted targeted failure codes; rejected"
                )
                continue

        # A repair that leaves every canonical field identical does not fix
        # anything; do not count it as repaired so the loop keeps trying. A FITB
        if (
            _canonical_values(qtype, item) == _canonical_values(qtype, normalized)
            and not fitb_bank_changed
        ):
            warnings.append(f"repair: {qid} returned no change; rejected")
            continue

        # Replace the question in place. Keeping the same dict object preserves
        # the stable question_id and any external references.
        item.clear()
        item.update(normalized)
        if fitb_bank_changed and isinstance(container, dict):
            container["word_bank"] = proposed_fitb_bank
        accepted.add(qid)
        repair_metadata_by_id[qid] = {
            "modified_fields": sorted(actual_changes),
            "changes": [
                {
                    "field": field,
                    **values,
                    "reason": next((
                        str(change.get("reason") or "")
                        for change in (reported_metadata or {}).get("changes") or []
                        if isinstance(change, dict)
                        and str(change.get("field") or "") == field
                    ), "repair patch changed this field"),
                }
                for field, values in actual_changes.items()
            ],
            "targeted_failure_codes": list(
                (reported_metadata or {}).get("targeted_failure_codes")
                or [verdict.get("failure_code")]
            ),
        }
        logger.info("Question repaired | id=%s | action=%s", qid, verdict.get("action"))

    fitb_section = questions.get("fill_in_the_blank")
    fitb_repaired = any(
        qid in accepted and (by_id.get(qid) or (None,))[0] == "fill_in_the_blank"
        for qid in accepted
    )
    revalidate_ids = set(accepted)
    if fitb_repaired and isinstance(fitb_section, dict):
        # The model never owns the shared bank. Any accepted stem/distractor
        # repair is followed by deterministic reconstruction from the fixed
        # catalog answers and the current per-item distractors.
        if _rebuild_fitb_word_bank(fitb_section, preferred_question_ids=accepted):
            revalidate_ids.update(
                str(item.get("question_id"))
                for item in fitb_section.get("items") or []
                if item.get("question_id")
            )
        else:
            warnings.append("repair: FITB Word Bank could not be rebuilt with two valid distractors")

    failed = [str(v.get("question_id")) for v in invalid if str(v.get("question_id")) not in accepted]
    if failed:
        warnings.append(f"repair: {len(failed)} question(s) not repaired: {sorted(failed)}")
    return {
        "warnings": warnings,
        "sent_ids": sent_ids,
        "repaired_ids": sorted(revalidate_ids),
        "failed_ids": failed,
        "repair_metadata": repair_metadata_by_id,
    }


def _requires_plan_replacement(verdict: Dict[str, Any]) -> bool:
    return (
        str(verdict.get("failure_code") or "") in TASK_REPLACEMENT_FAILURES
        or any(
            str(field).startswith(("semantic_identity.", "plan."))
            for field in (verdict.get("fields_to_fix") or [])
        )
    )

