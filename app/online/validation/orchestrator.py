"""Question validation orchestration: role dispatch, candidate gates, commit."""
from __future__ import annotations

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

from app.online.validation.state import _ensure_generation_active

from app.online.validation.constants import (
    ACTIONS,
    FUNDAMENTAL_ASSIGNMENT_FAILURES,
    GENERAL_VALIDATOR_MAX_TOKENS,
    MAX_ASSIGNMENT_REPAIR_ATTEMPTS,
    MAX_CANDIDATE_GENERATION_FAILURES,
    MAX_QUESTION_REPAIR_ATTEMPTS,
    RECOVERY_ACTIONS,
    REPAIR_ACTIONS,
    REPAIR_MAX_TOKENS,
    REPAIR_REASONING,
    REPAIR_TEMPERATURE,
    STEM_QUANTITATIVE_VALIDATOR_MAX_TOKENS,
    TASK_REPLACEMENT_FAILURES,
    VALIDATION_SCHEMA_VERSION,
    VALIDATOR_MAX_TOKENS,
    VALIDATOR_REASONING,
    VALIDATOR_TEMPERATURE,
)
from app.online.validation.questions import (
    _ACTION_DEFAULT_FIELDS,
    _ALL_CANONICAL,
    _BLANK_MARKERS,
    _LOGICAL_TO_CANONICAL,
    _WHY_PROMPT_RE,
    _allowed_canonical,
    _build_by_id,
    _canonical_changes,
    _canonical_values,
    _fitb_term_answer,
    _fitb_word_bank_structure_valid,
    _fitb_word_bank_valid,
    _global_duplicate_verdicts,
    _is_why_prompt,
    _iter_questions,
    _local_verdict,
    _meaningful_decimal_quantities,
    _normalize,
    _question_result_text,
    _question_task_text,
    _rebuild_fitb_word_bank,
    _simple_option_signature,
    _solve_system_completion_verdict,
    _verdict,
)
from app.online.validation.legacy.global_review import (
    GLOBAL_VALIDATOR_SYSTEM_PROMPT,
    _deterministic_exam_verdicts,
    _global_review_candidate_pairs,
    _global_review_entry,
    _merge_global_review,
    _operational_global_verdict,
    _standardize_global_verdict,
    build_global_review_prompt,
    review_exam_globally,
)
from app.online.validation.contracts import (
    QUANTITATIVE_VALIDATOR_OUTPUT_SCHEMA,
    REPAIR_SYSTEM_PROMPT,
    STEM_QUANTITATIVE_REPAIR_SYSTEM_PROMPT,
    STEM_QUANTITATIVE_VALIDATOR_RETRY_SYSTEM_PROMPT,
    STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_OUTPUT_SCHEMA,
    VALIDATOR_RETRY_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    _assigned_source_chunks,
    _catalog_task_contract,
    _plan_evidence_consistency_error,
    _plan_source_ids,
    build_repair_prompt,
    build_validator_prompt,
)
from app.online.validation.legacy.validation import (
    _merge_validation_report,
    _reconcile_simple_expected_answer,
    _sanitize_verdict,
    validate_exam,
)
from app.online.validation.legacy.repair import (
    _fields_changed_disallowed,
    _repair_fields_with_dependencies,
    _repair_role_for,
    _requires_plan_replacement,
    repair_exam,
)
from app.online.validation.context import (
    _merge_role_reports,
    _plan_context_for_model,
    _plan_items_by_slot,
    _quantitative_question_ids,
    _question_type_from_slot,
    _regenerate_plan_slot,
    _validate_with_context,
    persist_difficulty_observations,
)
from app.online.validation.manifest import (
    _manifest_from_report,
    _merge_repair_manifest,
    _placeholder_question,
    _put_question,
)
from app.online.validation.duplicate_coordinator import (
    _integrate_duplicate_validation,
)




from . import runtime as _runtime
from .recovery import (
    GENERAL_VALIDATION_RULE_VERSION,
    STEM_VALIDATION_RULE_VERSION,
    _defect_signature,
    _safe_record_repair_outcome,
)

def validate_generated_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run all six validators and transactionally revalidate staged candidates."""
    _ensure_generation_active(state)
    generated = deepcopy(state.get("generated_exams") or [])
    plans = deepcopy(state.get("plans") or [])
    validated = list(state.get("validated_models") or [])
    legacy_attempts = dict(state.get("question_repair_attempts") or {})
    failures = dict(
        state.get("question_repair_failures")
        or state.get("question_repair_attempts") or {}
    )
    attempts = failures  # compatibility name used by manifest diagnostics
    repair_calls = dict(state.get("question_repair_calls") or {})
    completion_attempts = dict(state.get("slot_completion_attempts") or {})
    completion_enabled = "slot_completion_attempts" in state
    working_candidates = deepcopy(state.get("repair_working_candidates") or {})
    pending_revalidation = dict(state.get("pending_revalidation_ids") or {})
    warnings = list(state.get("warnings") or [])
    stored_domain = str((state.get("assessment_profile") or {}).get("domain") or "")
    general_client = None
    quantitative_client = None
    if stored_domain:
        # Specialized clients are created per concurrent request by the new
        # orchestrator; sharing mutable provider clients across threads can
        # corrupt usage and response telemetry.
        general_client = quantitative_client = None
    # Preserve one latest report per model. Earlier passing models are skipped
    # by validation but their reports remain available for final telemetry.
    reports_by_model: Dict[Any, Dict[str, Any]] = {
        report.get("model_number"): report
        for report in (state.get("validation_reports") or [])
    }
    eval_stats = state.get("eval_stats")
    repair_manifest = deepcopy(state.get("repair_manifest") or [])
    replacement_memory = deepcopy(state.get("repair_replacement_memory") or {})
    retry_seeds = deepcopy(state.get("repair_retry_seeds") or {})
    validation_repair_history = list(state.get("validation_repair_history") or [])
    repair_call_logs = list(state.get("repair_call_logs") or [])
    validation_ledger = deepcopy(state.get("validation_ledger") or {})
    candidate_updates_by_model: Dict[int, Dict[str, Any]] = {}
    balanced_candidate_ids_by_model: Dict[int, list[str]] = {}
    # Candidate duplicate checks must share one content-keyed fact ledger for the
    # whole validation node.  Resetting it per candidate re-judged every
    # unchanged pair once per repair/replacement attempt.
    candidate_duplicate_fact_cache = dict(state.get("duplicate_fact_cache") or {})
    revalidation_progress = deepcopy(state.get("revalidation_progress") or {})
    pending_candidates: list[Dict[str, Any]] = []
    blocked_candidates = deepcopy(state.get("blocked_revalidation_candidates") or [])
    performance_events: list[dict[str, Any]] = []

    def accept_source_constrained_difficulty(
        report: Dict[str, Any], plan_by_slot: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Difficulty never fails a question; Luna's observation is authoritative.

        Narrow by design: every non-difficulty role must pass and every issue must
        be a difficulty mismatch. The observed level is persisted; the requested
        level stays separate. No repair, regeneration, or replacement is triggered.
        """
        result = deepcopy(report)
        for verdict in result.get("verdicts") or []:
            qid = str(verdict.get("question_id") or "")
            entry = _build_by_id(
                next((
                    exam.get("questions") or {} for exam in generated
                    if int(exam.get("model_number") or 0)
                    == int(result.get("model_number") or 0)
                ), {})
            ).get(qid)
            slot_id = str((entry or (None, None, {}))[2].get("slot_id") or "")
            plan_item = plan_by_slot.get(slot_id) or {}
            # Production plans always carry difficulty_design (allocation writes it).
            # Legacy/raw plans without it keep the historical strict behaviour.
            if not (
                plan_item.get("difficulty_design")
                or plan_item.get("difficulty_status")
                or (plan_item.get("difficulty_adaptation") or {}).get("status")
            ):
                continue
            issues = list(verdict.get("required_fixes") or [])
            if not issues or any(
                str(issue.get("failure_code") or "") not in {
                    "difficulty_mismatch", "reasoning_complexity_mismatch",
                }
                for issue in issues
            ):
                continue
            role_statuses = dict(verdict.get("role_statuses") or {})
            if any(
                status != "PASS" for role, status in role_statuses.items()
                if role != "difficulty"
            ):
                continue
            difficulty_role = (
                ((result.get("role_results") or {}).get(qid) or {}).get("difficulty")
                or {}
            )
            observation = deepcopy(verdict.get("difficulty_observation") or {}) or {
                "conceptual_difficulty": difficulty_role.get("conceptual_difficulty"),
                "numerical_difficulty": difficulty_role.get("numerical_difficulty"),
                "actual_difficulty": (
                    difficulty_role.get("actual_difficulty")
                    or difficulty_role.get("observed_difficulty")
                ),
                "difficulty_basis": difficulty_role.get("difficulty_basis"),
                "reason": str(difficulty_role.get("reason") or ""),
                "validator": "difficulty",
            }
            observed = str(
                observation.get("actual_difficulty")
                or plan_item.get("actual_difficulty")
                or plan_item.get("difficulty")
                or "medium"
            )
            question = (entry or (None, None, {}))[2]
            fingerprint = item_fingerprint(question) if question else ""
            observation.update({
                "actual_difficulty": observed,
                "question_fingerprint": fingerprint,
            })
            plan_item["difficulty_status"] = "accepted_source_constraint"
            plan_item["difficulty"] = observed
            plan_item["actual_difficulty"] = observed
            plan_item["difficulty_observation"] = deepcopy(observation)
            plan_item["difficulty_design"] = {
                **dict(plan_item.get("difficulty_design") or {}),
                "status": "accepted_source_constraint",
                "observed": observed,
                "reason": observation.get("reason") or "",
            }
            verdict.update({
                "action": "PASS", "validation_status": "PASS",
                "question_valid": True, "answer_valid": True,
                "failure_code": "pass", "fields_to_fix": [],
                "required_fixes": [], "expected_fix": "",
                "reason": "",
                "difficulty_deviation": {
                    "status": "accepted_source_constraint",
                    "requested_difficulty": plan_item.get("requested_difficulty"),
                    "actual_difficulty": observed,
                    "conceptual_difficulty": observation.get("conceptual_difficulty"),
                    "numerical_difficulty": observation.get("numerical_difficulty"),
                    "difficulty_basis": observation.get("difficulty_basis"),
                    "reason": observation.get("reason"),
                },
                "difficulty_observation": deepcopy(observation),
            })
            role_statuses["difficulty"] = "PASS"
            verdict["role_statuses"] = role_statuses
        result["all_pass"] = all(
            value.get("action") == "PASS" for value in result.get("verdicts") or []
        )
        return result

    def item_fingerprint(item: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(
            item, ensure_ascii=False, sort_keys=True, default=str,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    def final_integrity_error(
        candidate: Dict[str, Any], report: Dict[str, Any],
        active_exam: Dict[str, Any] | None, staged_entry: Any,
    ) -> str | None:
        if active_exam is None or staged_entry is None:
            return "active or staged question is missing"
        qid = str(candidate.get("question_id") or "")
        active_entry = _build_by_id(active_exam.get("questions") or {}).get(qid)
        if active_entry is None and candidate.get("had_original"):
            return "active original question is missing"
        expected_original = candidate.get("original_question") or {}
        if expected_original and active_entry is not None and item_fingerprint(active_entry[2]) != item_fingerprint(expected_original):
            return "active original question changed after candidate creation"
        staged_fingerprint = item_fingerprint(staged_entry[2])
        expected_candidate = str(candidate.get("candidate_fingerprint") or staged_fingerprint)
        if staged_fingerprint != expected_candidate:
            return "staged candidate changed after candidate creation"
        if str(report.get("validated_candidate_fingerprint") or staged_fingerprint) != staged_fingerprint:
            return "candidate selected for commit differs from validated candidate"
        current_lifecycle = (state.get("slot_lifecycle") or {}).get(str(candidate.get("slot_id") or "")) or {}
        expected_assignment = candidate.get("expected_assignment_fingerprint")
        expected_question = candidate.get("expected_question_fingerprint")
        if expected_assignment and current_lifecycle.get("assignment_fingerprint") != expected_assignment:
            return "assignment changed after revalidation"
        if expected_question and current_lifecycle.get("question_fingerprint") != expected_question:
            return "active question revision changed after revalidation"
        metadata = candidate.get("repair_metadata") or {}
        if candidate.get("repair_action") == "repair_question":
            actual = {
                change["field"] for change in _canonical_changes(
                    str(candidate.get("question_type") or staged_entry[0]),
                    expected_original,
                    staged_entry[2],
                )
            }
            def root(path: object) -> str:
                return str(path or "").split(".", 1)[0].split("[", 1)[0]
            # Specialized Repair metadata is Python-generated and records
            # recursive leaf paths. The graph integrity layer compares their
            # canonical roots with its independent top-level diff.
            reported = {root(path) for path in metadata.get("modified_fields") or []}
            if actual != reported:
                metadata.setdefault(
                    "advisory_modified_fields", sorted(reported)
                )
                metadata["modified_fields"] = sorted(actual)
            metadata["python_authoritative_modified_fields"] = sorted(actual)
        return None

    def run_specialized_reviews(
        target_exam: Dict[str, Any], ids: Optional[Iterable[str]],
        plan_context: Dict[str, Dict[str, Any]], sources: Dict[str, str],
        *, cached: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        revalidation: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        from app.online.multi_validator import validate_exam_multi
        return validate_exam_multi(
            target_exam, plan_by_slot=plan_context, source_by_chunk=sources,
            domain=stored_domain, question_ids=ids, cached_roles=cached,
            revalidation_context=revalidation,
            cache_scope=str(state.get("job_id") or ""),
        )

    # Validation of isolated staged candidates is independent. Workers only
    # produce reports; commits below remain deterministic and transactional.
    repair_candidates = []
    stale_candidate_events: list[dict[str, Any]] = []
    lifecycle = state.get("slot_lifecycle") or {}
    terminal_manifest_slots = {
        str(item.get("slot_id") or "")
        for item in repair_manifest
        if item.get("status") in {
            "exhausted", "terminal_failure", "validation_blocked",
        }
    }
    for candidate in state.get("repair_candidates") or []:
        slot_id = str(candidate.get("slot_id") or "")
        if slot_id in terminal_manifest_slots:
            continue
        current = lifecycle.get(slot_id) or {}
        expected_assignment = candidate.get("expected_assignment_fingerprint")
        expected_question = candidate.get("expected_question_fingerprint")
        stale = bool(
            (expected_assignment and expected_assignment != current.get("assignment_fingerprint"))
            or (expected_question and expected_question != current.get("question_fingerprint"))
        )
        if stale:
            stale_candidate_events.append({
                "model_number": candidate.get("model_number"), "slot_id": slot_id,
                "question_id": candidate.get("question_id"), "status": "stale_candidate_rejected",
                "failure_code": "stale_slot_revision",
                "expected_assignment_fingerprint": expected_assignment,
                "current_assignment_fingerprint": current.get("assignment_fingerprint"),
                "expected_question_fingerprint": expected_question,
                "current_question_fingerprint": current.get("question_fingerprint"),
            })
            continue
        repair_candidates.append(candidate)
    validation_repair_history.extend(stale_candidate_events)

    def blocked_report(candidate: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        qid = str(candidate.get("question_id") or "")
        return {"model_number": candidate.get("model_number"), "all_pass": False,
                "verdicts": [{"question_id": qid, "action": "UNVALIDATED",
                    "validation_status": "VALIDATION_BLOCKED",
                    "failure_code": "validation_blocked", "fields_to_fix": [],
                    "required_fixes": [], "role_statuses": {"system": "ERROR"},
                    "reason": f"{type(exc).__name__}: {exc}",
                    "expected_fix": "Retry candidate validation."}],
                "role_results": {qid: {"system": {"status": "ERROR",
                    "operational_error": f"{type(exc).__name__}: {exc}"}}},
                "role_fingerprints": {}, "validation_call_logs": []}

    def validate_candidate_group(indexes: list[int]) -> dict[int, Dict[str, Any]]:
        from app.logging_conf import set_llm_attribution
        first = repair_candidates[indexes[0]]
        mn = int(first.get("model_number") or 1)
        combined_exam: Dict[str, Any] = {"model_number": mn, "questions": {}}
        combined_plan: Dict[str, Dict[str, Any]] = {}
        combined_sources: Dict[str, str] = {}
        ids: list[str] = []
        contexts: Dict[str, Dict[str, Any]] = {}
        for index in indexes:
            candidate = repair_candidates[index]
            qid = str(candidate.get("question_id") or "")
            entry = _build_by_id((candidate.get("exam") or {}).get("questions") or {}).get(qid)
            if entry is None:
                raise ValueError(f"staged candidate question missing: {qid}")
            _put_question(combined_exam, entry[0], deepcopy(entry[2]))
            candidate_state = {**state, "plans": candidate.get("plans") or plans}
            local_plan, local_sources = _plan_context_for_model(candidate_state, mn)
            staged_assignment = local_plan.get(str(candidate.get("slot_id") or ""))
            if staged_assignment is None and local_plan:
                raise ValueError(f"staged candidate plan missing: {candidate.get('slot_id')}")
            if staged_assignment is not None:
                combined_plan[str(candidate.get("slot_id") or "")] = staged_assignment
                combined_sources.update(_assigned_source_chunks(
                    staged_assignment, local_sources,
                ))
            ids.append(qid)
            contexts[qid] = {
                "original_question": deepcopy(candidate.get("original_question") or {}),
                "original_failure_findings": deepcopy(candidate.get("original_failure_findings") or []),
                "repair_metadata": deepcopy(candidate.get("repair_metadata") or {}),
                "previous_rejections": deepcopy(candidate.get("previous_rejections") or []),
                "candidate_id": candidate.get("candidate_id"),
                "candidate_fingerprint": candidate.get("candidate_fingerprint"),
            }
        set_llm_attribution(
            model_number=mn, stage="revalidation",
            slot_ids=[str(repair_candidates[index].get("slot_id") or "") for index in indexes],
            question_ids=list(ids),
            recovery_attempts=[repair_candidates[index].get("attempt") for index in indexes],
        )
        if stored_domain:
            from app.online.multi_validator import validate_exam_multi
            shared = validate_exam_multi(
                combined_exam, plan_by_slot=combined_plan, source_by_chunk=combined_sources,
                domain=stored_domain, question_ids=ids,
                revalidation_context=contexts,
                cache_scope=str(state.get("job_id") or ""),
            )
        else:
            shared = _runtime._validate_with_context(combined_exam, ids, combined_plan, combined_sources)
        by_verdict = {str(value.get("question_id") or ""): value for value in shared.get("verdicts") or []}
        result: dict[int, Dict[str, Any]] = {}
        for index in indexes:
            qid = str(repair_candidates[index].get("question_id") or "")
            candidate_entry = _build_by_id(
                (repair_candidates[index].get("exam") or {}).get("questions") or {}
            ).get(qid)
            verdict = deepcopy(by_verdict.get(qid))
            if verdict is None:
                verdict = blocked_report(repair_candidates[index], RuntimeError("missing batched verdict"))["verdicts"][0]
            result[index] = {"model_number": mn, "verdicts": [verdict],
                "all_pass": verdict.get("action") == "PASS",
                "role_results": {qid: deepcopy((shared.get("role_results") or {}).get(qid) or {})},
                "role_fingerprints": {qid: deepcopy((shared.get("role_fingerprints") or {}).get(qid) or {})},
                "validation_call_logs": [deepcopy(trace) for trace in shared.get("validation_call_logs") or []
                    if qid in set(map(str, trace.get("question_ids") or []))],
                "input_protection": {qid: deepcopy((shared.get("input_protection") or {}).get(qid) or {})},
                "validated_candidate_fingerprint": item_fingerprint((candidate_entry or (None, None, {}))[2])}
        return result

    def candidate_duplicate_conflict(
        candidate: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Run the authoritative within-model Duplicate V2 gate before commit.

        Candidate selection/preflight operates on task metadata.  It cannot
        prove that the *generated question* is distinct, so every staged
        candidate (content repair, regeneration, and task replacement) must be
        checked against the current active model with its previous version
        replaced by the candidate.
        """
        nonlocal candidate_duplicate_fact_cache
        from app.online.duplicate_pipeline import scan_exam_duplicates
        qid = str(candidate.get("question_id") or "")
        mn = int(candidate.get("model_number") or 1)
        active_exam = next((
            value for value in generated
            if int(value.get("model_number") or 0) == mn
        ), None)
        candidate_exam = deepcopy(active_exam or candidate.get("exam") or {})
        staged_entry = _build_by_id(
            ((candidate.get("exam") or {}).get("questions") or {})
        ).get(qid)
        if staged_entry is None:
            return {
                "question_id": qid, "action": "UNVALIDATED",
                "validation_status": "VALIDATION_BLOCKED",
                "failure_code": "duplicate_recheck_candidate_missing",
                "role_statuses": {"duplicate": "ERROR"},
                "reason": "The staged candidate was unavailable for Duplicate V2.",
                "expected_fix": "Reconstruct the staged candidate.",
                "required_fixes": [], "fields_to_fix": [],
                "validation_role": "duplicate",
            }
        # This overwrites the active/old version and therefore excludes it from
        # self-comparison while preserving every other current model question.
        _put_question(candidate_exam, staged_entry[0], deepcopy(staged_entry[2]))
        duplicate_state = {
            **state,
            "generated_exams": [candidate_exam],
            "plans": deepcopy(candidate.get("plans") or plans),
            "validation_reports": [],
            "duplicate_reports": [],
            "cross_model_duplicate_reports": [],
            "repair_manifest": [],
            "repair_candidates": [],
            "validated_models": [],
            "warnings": [],
            # Duplicate facts are content-keyed.  A changed candidate produces
            # new keys for every pair that contains it, while facts for pairs
            # of unchanged questions remain valid and avoid repeated judging.
            "duplicate_fact_cache": dict(candidate_duplicate_fact_cache),
            # The current active-exam fingerprint baseline limits screening to
            # pairs that the candidate (or a just-committed revision) touched.
            "duplicate_screen_fingerprints": deepcopy(
                state.get("duplicate_screen_fingerprints") or {}
            ),
        }
        try:
            report = scan_exam_duplicates(duplicate_state)
        except Exception as exc:
            return {
                "question_id": qid, "action": "UNVALIDATED",
                "validation_status": "VALIDATION_BLOCKED",
                "failure_code": "duplicate_recheck_failed",
                "role_statuses": {"duplicate": "ERROR"},
                "reason": f"Duplicate V2 recheck failed: {type(exc).__name__}: {exc}",
                "expected_fix": "Retry the existing Duplicate V2 check.",
                "required_fixes": [], "fields_to_fix": [],
                "validation_role": "duplicate",
            }
        candidate_duplicate_fact_cache = dict(
            report.get("duplicate_fact_cache") or candidate_duplicate_fact_cache
        )
        for row in report.get("duplicate_reports") or []:
            for group in row.get("duplicate_groups") or []:
                if qid in set(map(str, group.get("question_ids") or [])):
                    return {
                        "question_id": qid, "action": "FIX_QUESTION_AND_ANSWER",
                        "validation_status": "FAIL",
                        "failure_code": "duplicate_question",
                        "role_statuses": {"duplicate": "FAIL"},
                        "reason": "The candidate still duplicates another validated question.",
                        "expected_fix": "Select a distinct authorized backup.",
                        "required_fixes": [{
                            "reason": "The repaired candidate still duplicates another validated question.",
                            "validator": "duplicate",
                            "failure_code": "duplicate_question",
                            "validation_status": "FAIL",
                            "fields_to_fix": [],
                            "duplicate_pair_ids": list(group.get("pair_ids") or []),
                            "conflicting_question_ids": [
                                value for value in group.get("question_ids") or []
                                if str(value) != qid
                            ],
                        }],
                        "fields_to_fix": [], "validation_role": "duplicate",
                        "duplicate_group_id": group.get("group_id"),
                    }
        return None

    def _candidate_conflict_verdict(
        qid: str, pair_ids: list[str], conflicting_ids: list[str],
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "question_id": qid, "action": "FIX_QUESTION_AND_ANSWER",
            "validation_status": "FAIL", "failure_code": "duplicate_question",
            "role_statuses": {"duplicate": "FAIL"}, "reason": reason,
            "expected_fix": "Select a distinct authorized backup or variation.",
            "required_fixes": [{
                "reason": reason, "validator": "duplicate",
                "failure_code": "duplicate_question", "validation_status": "FAIL",
                "fields_to_fix": [], "duplicate_pair_ids": list(pair_ids),
                "conflicting_question_ids": list(conflicting_ids),
            }],
            "fields_to_fix": [], "validation_role": "duplicate",
        }

    def batch_candidate_duplicate_conflicts(
        indexes: list[int], candidate_report_map: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Judge independent staged candidates in one batched pre-commit gate.

        Each ready candidate is inserted under a synthetic question/slot
        identity into a copy of the active model.  All candidate-involved pairs
        are screened and reviewed concurrently in provider-sized batches, using
        the shared content-keyed fact ledger.  Candidate-vs-active conflicts
        reject the candidate; candidate-vs-candidate conflicts keep one
        deterministic winner.  The final authoritative scan still runs after
        commits.
        """
        nonlocal candidate_duplicate_fact_cache
        from app.online.duplicate_judge import review_duplicate_candidates

        ready: list[tuple[int, Dict[str, Any], Dict[str, Any], str]] = []
        for index in indexes:
            candidate = repair_candidates[index]
            report = candidate_report_map.get(index) or {}
            verdicts = list(report.get("verdicts") or [])
            if not verdicts or any(value.get("action") != "PASS" for value in verdicts):
                continue
            mn = int(candidate.get("model_number") or 1)
            active_exam = next((
                value for value in generated
                if int(value.get("model_number") or 0) == mn
            ), None)
            qid = str(candidate.get("question_id") or "")
            staged_entry = _build_by_id(
                (candidate.get("exam") or {}).get("questions") or {}
            ).get(qid)
            if active_exam is None or staged_entry is None:
                continue
            if final_integrity_error(candidate, report, active_exam, staged_entry):
                continue
            ready.append((index, candidate, report, qid))
        if not ready:
            return {}
        # Every ready candidate is gated; None means explicitly checked and
        # clear, so the serial loop must not pay for another duplicate scan.
        gated: Dict[int, Dict[str, Any] | None] = {
            index: None for index, _candidate, _report, _qid in ready
        }

        gate_by_model: Dict[int, Dict[str, Any]] = {}
        plan_by_model: Dict[int, Dict[str, Dict[str, Any]]] = {}
        synthetic: Dict[str, Dict[str, Any]] = {}
        own_old_id: Dict[str, str] = {}
        for index, candidate, _report, qid in ready:
            mn = int(candidate.get("model_number") or 1)
            if mn not in gate_by_model:
                active_exam = next((
                    value for value in generated
                    if int(value.get("model_number") or 0) == mn
                ))
                gate_by_model[mn] = deepcopy(active_exam)
                plan_by_model[mn] = _plan_context_for_model(
                    {**state, "plans": plans}, mn
                )[0]
            qtype, _container, staged_item = _build_by_id(
                (candidate.get("exam") or {}).get("questions") or {}
            )[qid]
            synthetic_id = f"__candidate_{index}__{qid}"
            synthetic_slot = f"__candidate_slot_{index}__"
            item = deepcopy(staged_item)
            item["question_id"] = synthetic_id
            item["slot_id"] = synthetic_slot
            section = gate_by_model[mn].setdefault("questions", {}).setdefault(qtype, [])
            if isinstance(section, dict):
                section = section.setdefault("items", [])
            section.append(item)
            staged_plan = _plan_context_for_model(
                {**state, "plans": candidate.get("plans") or plans}, mn
            )[0].get(str(candidate.get("slot_id") or ""))
            if staged_plan:
                plan_by_model[mn][synthetic_slot] = staged_plan
            synthetic[synthetic_id] = {
                "index": index, "qid": qid,
                "slot_id": str(candidate.get("slot_id") or ""),
                "attempt": int(candidate.get("attempt") or 0),
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "model_number": mn,
            }
            own_old_id[synthetic_id] = qid

        rejected: Dict[int, Dict[str, Any]] = {}
        rejected_pairs: Dict[int, list[str]] = {}
        rejected_conflicts: Dict[int, list[str]] = {}
        for mn, gate_exam in gate_by_model.items():
            changed_ids = {
                value for value, meta in synthetic.items() if meta["model_number"] == mn
            }
            try:
                review = review_duplicate_candidates(
                    gate_exam, plan_by_slot=plan_by_model[mn],
                    changed_question_ids=changed_ids,
                    fact_cache=dict(candidate_duplicate_fact_cache),
                    excluded_question_pairs={
                        frozenset((value, own_old_id[value]))
                        for value in changed_ids if own_old_id.get(value)
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Batched candidate duplicate gate unavailable | model=%s | error=%s: %s",
                    mn, type(exc).__name__, exc,
                )
                for value in synthetic.values():
                    if value["model_number"] == mn:
                        gated.pop(int(value["index"]), None)
                continue
            candidate_duplicate_fact_cache = dict(
                review.get("fact_cache") or candidate_duplicate_fact_cache
            )
            pair_by_id = {
                str(value.get("pair_id")): value
                for value in review.get("candidate_pairs") or []
            }
            for verdict in review.get("pair_verdicts") or []:
                if str(verdict.get("classification") or "") not in {
                    "duplicate", "excessive_overlap",
                }:
                    continue
                pair_id = str(verdict.get("pair_id") or "")
                pair = pair_by_id.get(pair_id) or {}
                left_id = str((pair.get("left") or {}).get("question_id") or "")
                right_id = str((pair.get("right") or {}).get("question_id") or "")
                left = synthetic.get(left_id)
                right = synthetic.get(right_id)
                if left is None and right is None:
                    continue
                # A candidate is expected to replace its own old question; that
                # pair is not a duplicate conflict.
                if left and right is None and own_old_id.get(left_id) == right_id:
                    continue
                if right and left is None and own_old_id.get(right_id) == left_id:
                    continue
                if left and right:
                    # Keep the earlier attempt; deterministic tie-break on id.
                    loser = max(
                        (left, right),
                        key=lambda value: (value["attempt"], value["candidate_id"]),
                    )
                    winner = right if loser is left else left
                    if loser["index"] not in rejected:
                        rejected_pairs.setdefault(loser["index"], []).append(pair_id)
                        rejected_conflicts.setdefault(loser["index"], []).append(
                            str(winner["qid"])
                        )
                        rejected[loser["index"]] = _candidate_conflict_verdict(
                            str(loser["qid"]), rejected_pairs[loser["index"]],
                            rejected_conflicts[loser["index"]],
                            "The staged candidate duplicates another candidate "
                            "staged in the same recovery round.",
                        )
                else:
                    candidate_meta = left or right
                    active_id = right_id if left else left_id
                    index = int(candidate_meta["index"])
                    if index in rejected:
                        continue
                    rejected_pairs.setdefault(index, []).append(pair_id)
                    rejected_conflicts.setdefault(index, []).append(active_id)
                    rejected[index] = _candidate_conflict_verdict(
                        str(candidate_meta["qid"]), rejected_pairs[index],
                        rejected_conflicts[index],
                        "The candidate still duplicates another validated question.",
                    )
        gated.update(rejected)
        return gated

    candidate_reports: dict[int, Dict[str, Any]] = {}
    groups: list[list[int]] = []
    grouped: Dict[tuple[int, str], list[int]] = {}
    for index, candidate in enumerate(repair_candidates):
        qtype = str(candidate.get("question_type") or "")
        # FITB candidates retain serial section-level word-bank safety.
        key = (int(candidate.get("model_number") or 1), qtype if qtype == "fill_in_the_blank" else "parallel")
        grouped.setdefault(key, []).append(index)
    for (_mn, kind), indexes in grouped.items():
        size = 1 if kind == "fill_in_the_blank" else VALIDATOR_BATCH_SIZE
        groups.extend(indexes[start:start + size] for start in range(0, len(indexes), size))
    if groups:
        workers = min(MAX_PARALLEL_VALIDATOR_CALLS, len(groups), 10)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(copy_context().run, validate_candidate_group, group): group for group in groups}
            for future in as_completed(futures):
                group = futures[future]
                try:
                    candidate_reports.update(future.result())
                except Exception as exc:
                    for index in group:
                        candidate_reports[index] = blocked_report(repair_candidates[index], exc)

    # Accept source-constrained difficulty before gating so a difficulty-only
    # report is not treated as a duplicate-gate blocker.  The batched gate then
    # judges every ready candidate before any candidate is committed.
    for index, candidate in enumerate(repair_candidates):
        mn = int(candidate.get("model_number") or 1)
        candidate_state = {**state, "plans": candidate.get("plans") or plans}
        candidate_plan, _candidate_sources = _plan_context_for_model(candidate_state, mn)
        candidate_reports[index] = accept_source_constrained_difficulty(
            candidate_reports[index], candidate_plan,
        )
    batched_duplicate_conflicts = batch_candidate_duplicate_conflicts(
        list(range(len(repair_candidates))), candidate_reports,
    )

    # Repair writes are transactional: mutate the active exam/plan only after
    # every required validator for that candidate passes.
    for index, candidate in enumerate(repair_candidates):
        mn = int(candidate.get("model_number") or 1)
        slot_id = str(candidate.get("slot_id") or "")
        qid = str(candidate.get("question_id") or "")
        staged_exam = candidate.get("exam") or {}
        candidate_state = {**state, "plans": candidate.get("plans") or plans}
        candidate_plan, _candidate_sources = _plan_context_for_model(candidate_state, mn)
        candidate_report = candidate_reports[index]
        plans = persist_difficulty_observations(
            plans, candidate_report, staged_exam,
        )
        verdicts = list(candidate_report.get("verdicts") or [])
        passed = bool(verdicts) and all(v.get("action") == "PASS" for v in verdicts)
        failure = next((v for v in verdicts if v.get("action") != "PASS"), {})
        operational = str(failure.get("validation_status") or "") in {
            "VALIDATION_BLOCKED", "PENDING_RETRY",
        }
        candidate_id = str(candidate.get("candidate_id") or
                           candidate.get("candidate_fingerprint") or
                           f"{mn}:{slot_id}:{candidate.get('attempt')}")
        target = next((
            item for item in repair_manifest
            if int(item.get("model_number") or 0) == mn
            and str(item.get("slot_id") or "") == slot_id
        ), None)
        event = {
            "model_number": mn,
            "slot_id": slot_id,
            "question_id": qid,
            "attempt": candidate.get("attempt"),
            "repair_action": candidate.get("repair_action"),
            "repair_strategy": candidate.get("repair_strategy"),
            "candidate_task_id": candidate.get("candidate_task_id"),
            "candidate_fitb_entry_id": candidate.get("candidate_fitb_entry_id"),
            "revalidation_result": "PASS" if passed else "FAIL",
            "assignment_fingerprint": candidate.get("assignment_fingerprint"),
            "telemetry_category": (
                "cross_model_duplicate_replacement"
                if str((target or {}).get("original_failure_code") or
                       (target or {}).get("failure_code") or "")
                == "cross_model_duplicate"
                else {
                    "repair_question": "content_repair",
                    "regenerate_missing_slot": "regeneration",
                    "replace_fitb_entry": "assignment_replacement",
                    "replace_task": "assignment_replacement",
                    "replace_assignment": "assignment_replacement",
                }.get(str(candidate.get("repair_action") or ""), "revalidation")
            ),
        }
        active_exam = next((e for e in generated if int(e.get("model_number") or 0) == mn), None)
        staged_entry = _build_by_id(staged_exam.get("questions") or {}).get(qid)
        integrity_error = final_integrity_error(candidate, candidate_report, active_exam, staged_entry) if passed else None
        if integrity_error:
            passed = False
            operational = True
            failure = {
                "question_id": qid, "action": "UNVALIDATED",
                "validation_status": "VALIDATION_BLOCKED",
                "failure_code": "candidate_integrity_error", "role_statuses": {"integrity": "ERROR"},
                "reason": integrity_error, "expected_fix": "Discard or reconstruct the staged candidate.",
                "required_fixes": [], "fields_to_fix": [],
            }
            verdicts = [failure]
        if passed:
            duplicate_conflict = (
                batched_duplicate_conflicts[index]
                if index in batched_duplicate_conflicts
                else candidate_duplicate_conflict(candidate)
            )
            if duplicate_conflict:
                passed = False
                operational = False
                failure = duplicate_conflict
                verdicts = [duplicate_conflict]
            else:
                # Make the successful final gate explicit in the fresh report;
                # this supersedes, rather than merely hides, a historical
                # duplicate finding for the previous question revision.
                for verdict in verdicts:
                    verdict.setdefault("role_statuses", {})["duplicate"] = "PASS"
                    verdict["duplicate_recheck"] = {
                        "status": "PASS", "evaluation_origin": "fresh",
                        "candidate_fingerprint": candidate.get("candidate_fingerprint"),
                    }
        if passed:
            event["progress"] = "PASSED"
            if active_exam is not None and staged_entry is not None:
                _put_question(active_exam, staged_entry[0], staged_entry[2])
                if isinstance(state.get("eval_stats"), dict):
                    _runtime._safe_record_repair_outcome(
                        state.get("eval_stats"), active_exam, candidate_report,
                        repaired_ids=[qid], failed_ids=[], sent_ids=[qid],
                    )
                if mn in candidate_updates_by_model:
                    candidate_updates_by_model[mn] = _merge_validation_report(
                        candidate_updates_by_model[mn], candidate_report
                    )
                else:
                    candidate_updates_by_model[mn] = candidate_report
                staged_plan = candidate_plan.get(slot_id)
                if staged_plan:
                    from app.online.duplicate_pipeline import _replace_plan_item
                    _replace_plan_item(plans, mn, slot_id, deepcopy(staged_plan))
                    for plan in plans:
                        if int(plan.get("model_number") or 0) != mn:
                            continue
                        for values in (plan.get("items") or {}).values():
                            for item in values or []:
                                if str(item.get("slot_id") or "") == slot_id:
                                    item.pop("difficulty_observation", None)
                    plans = persist_difficulty_observations(
                        plans, candidate_report, staged_exam,
                    )
                if target is not None:
                    target["status"] = "resolved"
                    target["resolved_by_attempt"] = candidate.get("attempt")
                    target["resolved_candidate_fingerprint"] = candidate.get(
                        "candidate_fingerprint"
                    )
                    target["superseded_failure_code"] = target.get("failure_code")
                    recovery = target.get("recovery_v3") or {}
                    recovery["status"] = "completed"
                    recovery["completed_strategy"] = candidate.get("recovery_v3_strategy")
                    target["recovery_v3"] = recovery
                for finding in state.get("duplicate_findings") or []:
                    if (
                        str(finding.get("replace_question_id") or "") == qid
                        and str(finding.get("status") or "") in {"confirmed", "pending"}
                    ):
                        finding["status"] = "repaired"
                        finding["repair_attempt"] = candidate.get("attempt")
                event["status"] = "accepted"
                retry_seeds.pop(slot_id, None)
                revalidation_progress.pop(candidate_id, None)
                slot_memory = replacement_memory.get(slot_id) or {}
                if slot_memory.get("strategy_history"):
                    slot_memory["strategy_history"][-1].update(
                        outcome="PASS", candidate_fingerprint=candidate.get("candidate_fingerprint"),
                    )
        elif operational:
            role_statuses = deepcopy(failure.get("role_statuses") or {})
            pending_roles = sorted(
                role for role, status in role_statuses.items()
                if status in {"ERROR", "VALIDATION_BLOCKED", "PENDING_RETRY"}
            ) or ["system"]
            progress = revalidation_progress.setdefault(candidate_id, {
                "candidate_id": candidate_id, "question_id": qid,
                "candidate_fingerprint": candidate.get("candidate_fingerprint"),
                "role_statuses": {}, "operational_retry_counts": {},
            })
            for role, result in ((candidate_report.get("role_results") or {}).get(qid) or {}).items():
                status = str(result.get("status") or "ERROR")
                progress["role_statuses"][role] = status
            for role in pending_roles:
                progress["operational_retry_counts"][role] = int(
                    progress["operational_retry_counts"].get(role) or 0
                ) + 1
            exhausted = integrity_error is not None or any(
                progress["operational_retry_counts"].get(role, 0) > REVALIDATION_OPERATIONAL_RETRIES
                for role in pending_roles
            )
            progress.update(
                candidate_status="VALIDATION_BLOCKED" if exhausted else "PENDING_RETRY",
                pending_roles=[] if exhausted else pending_roles,
                last_operational_error=failure.get("reason") or failure.get("expected_fix") or "",
            )
            if target is not None:
                target["status"] = "validation_blocked" if exhausted else "validation_pending"
                target["validation_candidate_id"] = candidate_id
            event.update(
                status="validation_blocked" if exhausted else "pending_validation_retry",
                progress="BLOCKED" if exhausted else "PENDING_RETRY",
                revalidation_result="VALIDATION_BLOCKED" if exhausted else "PENDING_RETRY",
                operational=True,
            )
            if exhausted:
                blocked_candidates.append(deepcopy(candidate) | {
                    "candidate_status": "VALIDATION_BLOCKED",
                    "validation_error": progress["last_operational_error"],
                })
            else:
                pending_candidates.append(deepcopy(candidate))
        else:
            new_code = str(failure.get("failure_code") or "candidate_revalidation_failed")
            new_reason = str(failure.get("reason") or failure.get("expected_fix") or
                             "Candidate did not pass revalidation.")
            old_code = str(candidate.get("previous_failure_code") or "")
            severe = {"unsupported_given", "evidence_mismatch", "task_drift", "wrong_task_identity"}
            event["progress"] = (
                "UNCHANGED" if new_code == old_code
                else "REGRESSED" if new_code in severe and old_code not in severe
                else "LATERAL"
            )
            slot_memory = replacement_memory.setdefault(slot_id, {})
            signature = _defect_signature(failure or {})
            signature_history = list(slot_memory.get("defect_signature_history") or [])
            repeated_defect = signature in signature_history[-2:]
            signature_history.append(signature)
            slot_memory["defect_signature_history"] = signature_history[-6:]
            if repeated_defect:
                slot_memory["repeated_defect"] = {
                    "signature": signature,
                    "failure_code": new_code,
                    "failure_reason": new_reason,
                    "fields_to_fix": list(failure.get("fields_to_fix") or []),
                    "previous_attempts": list(slot_memory.get("strategy_history") or [])[-3:],
                }
            if active_exam is not None and isinstance(state.get("eval_stats"), dict):
                _runtime._safe_record_repair_outcome(
                    state.get("eval_stats"), active_exam, candidate_report,
                    repaired_ids=[], failed_ids=[qid], sent_ids=[qid],
                )
            if slot_memory.get("strategy_history"):
                slot_memory["strategy_history"][-1].update(
                    outcome="REVALIDATION_FAIL",
                    failure_code=new_code,
                    failure_reason=new_reason,
                    candidate_fingerprint=candidate.get("candidate_fingerprint"),
                    progress=event["progress"],
                )
            category = "recovery_v3_revalidation_failure"
            # A content candidate that fails full Revalidation spends one
            # failed strategy. A separately-budgeted replacement does not.
            if candidate.get("repair_action") == "repair_question":
                failures[slot_id] = min(
                    MAX_QUESTION_REPAIR_ATTEMPTS,
                    int(failures.get(slot_id) or 0) + 1,
                )
            if target is not None:
                recovery = target.get("recovery_v3") or {}
                strategy = str(candidate.get("recovery_v3_strategy") or "")
                if strategy and strategy not in recovery.setdefault("attempted_strategies", []):
                    recovery["attempted_strategies"].append(strategy)
                # SWITCH_TASK owns the bounded sequence of authorized
                # variations/backups/Catalog candidates. A failed replacement
                # advances that assignment inventory, not the semantic plan.
                if strategy != "SWITCH_TASK":
                    recovery["next_index"] = int(recovery.get("next_index") or 0) + 1
                recovery["last_failure_code"] = new_code
                recovery["last_failure_reason"] = new_reason
                next_plan = list(recovery.get("plan") or [])
                next_index = int(recovery.get("next_index") or 0)
                repair_type = {
                    "REPAIR": "repair_question",
                    "REGENERATE": "regenerate_missing_slot",
                    "SWITCH_TASK": "replace_assignment",
                }.get(str(next_plan[next_index]).upper(), "recovery_plan_exhausted") if next_index < len(next_plan) else "recovery_plan_exhausted"
                target.update(
                    status=(
                        "terminal_failure"
                        if (
                            int(failures.get(slot_id) or 0)
                            >= MAX_QUESTION_REPAIR_ATTEMPTS
                            and int(completion_attempts.get(slot_id) or 0)
                            >= MAX_COMPLETION_ATTEMPTS_PER_SLOT
                        )
                        else ("invalid" if candidate.get("had_original") else "missing")
                    ),
                    failure_code=new_code,
                    failure_reason=new_reason,
                    repair_category=category,
                    repair_type=repair_type,
                    verdict=deepcopy(failure) if failure else None,
                    recovery_v3=recovery,
                )
                target.setdefault("history", []).append(deepcopy(event) | {
                    "new_failure_code": new_code, "new_failure_reason": new_reason,
                })
            candidate_id = str(candidate.get("candidate_fitb_entry_id") or
                               candidate.get("candidate_task_id") or "")
            if candidate_id:
                from app.online.duplicate_pipeline import replacement_instance_key
                memory = replacement_memory.setdefault(slot_id, {
                    "attempted_candidate_ids": [], "blacklisted_candidate_ids": [],
                    "candidate_failures": [], "replacement_attempt_count": 0,
                })
                candidate_plan = next((
                    item for plan in candidate.get("plans") or []
                    if int(plan.get("model_number") or 0) == int(candidate.get("model_number") or 0)
                    for values in (plan.get("items") or {}).values()
                    for item in values or []
                    if str(item.get("slot_id") or "") == slot_id
                ), {})
                memory.setdefault("candidate_failures", []).append({
                    "candidate_id": candidate_id, "attempt": candidate.get("attempt"),
                    "candidate_instance_key": replacement_instance_key({
                        **candidate_plan,
                        "replacement_instance_sequence": candidate.get("attempt"),
                    }),
                    "reason": new_reason, "failure_code": new_code,
                })
                fingerprint = str(candidate.get("candidate_fingerprint") or "")
                if fingerprint and fingerprint not in memory.setdefault("rejected_candidate_fingerprints", []):
                    memory["rejected_candidate_fingerprints"].append(fingerprint)
            event.update(status="rejected", new_failure_code=new_code,
                         new_failure_reason=new_reason,
                         candidate_fingerprint=candidate.get("candidate_fingerprint"),
                         changed_fields=list((candidate.get("repair_metadata") or {}).get("modified_fields") or []))
            retry_seeds[slot_id] = deepcopy(candidate)
            revalidation_progress.pop(candidate_id, None)
        validation_repair_history.append(event)
        repair_call_logs.append({
            "job_id": None,
            "question_id": qid,
            "slot_id": slot_id,
            "repair_route": deepcopy(candidate.get("repair_route") or []),
            "attempt": candidate.get("attempt"),
            "phase": "revalidation",
            "revalidation_outcome": event.get("revalidation_result"),
            "failure_code": None if passed else event.get("new_failure_code"),
        })
    # Candidates are consumed exactly once, whether accepted or rejected.
    if candidate_updates_by_model:
        from app.online.exam_builder import balance_mcq_answer_positions
        for exam in generated:
            mn = int(exam.get("model_number") or 1)
            if mn not in candidate_updates_by_model:
                continue
            changed_ids = balance_mcq_answer_positions(
                exam.get("questions") or {}, mn
            )
            if changed_ids:
                balanced_candidate_ids_by_model[mn] = changed_ids

    # Initial model reviews are independent. Run their complete role queues at
    # the same time; cross-model duplicate coordination remains below, after
    # every local model report has been merged deterministically.
    prefetched_initial_reports: dict[int, Dict[str, Any]] = {}
    initial_models = [
        exam for exam in generated
        if stored_domain
        and exam.get("model_number") not in validated
        and reports_by_model.get(exam.get("model_number")) is None
        and int(exam.get("model_number") or 0) not in candidate_updates_by_model
        and not (pending_revalidation.get(exam.get("model_number")) or [])
    ]
    if len(initial_models) > 1:
        from app.online.performance import finish_timing, start_timing

        def initial_model_review(exam: Dict[str, Any]):
            mn = int(exam.get("model_number") or 1)
            timing = start_timing(stage="model_validation", model_number=mn)
            local_plan, local_sources = _plan_context_for_model(
                {**state, "plans": plans}, mn
            )
            report = run_specialized_reviews(exam, None, local_plan, local_sources)
            return mn, report, finish_timing(timing, question_count=len(
                _build_by_id(exam.get("questions") or {})
            ))

        workers = min(MAX_PARALLEL_VALIDATOR_CALLS, len(initial_models), 10)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(copy_context().run, initial_model_review, exam): exam
                for exam in initial_models
            }
            for future in as_completed(futures):
                mn, report, timing = future.result()
                prefetched_initial_reports[mn] = report
                performance_events.append(timing)

    for exam in generated:
        mn = exam.get("model_number")
        if mn in validated:
            continue
        previous = reports_by_model.get(mn)
        plan_by_slot, source_by_chunk = _plan_context_for_model(
            {**state, "plans": plans}, int(mn or 1)
        )
        terminal_slots_for_model = {
            str(entry.get("slot_id") or "")
            for entry in repair_manifest
            if int(entry.get("model_number") or 0) == int(mn or 0)
            and entry.get("status") in {
                "exhausted", "terminal_failure", "validation_blocked",
            }
        }
        def question_fingerprints() -> dict[str, str]:
            result: dict[str, str] = {}
            for qtype, container, item in _iter_questions(exam.get("questions") or {}):
                qid = str(item.get("question_id") or "")
                slot_id = str(item.get("slot_id") or "")
                if not qid or slot_id in terminal_slots_for_model:
                    continue
                payload = {
                    "question": item,
                    "plan": plan_by_slot.get(slot_id) or {},
                    "source_hashes": {
                        chunk_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
                        for chunk_id, text in _assigned_source_chunks(
                            plan_by_slot.get(slot_id), source_by_chunk,
                            list(item.get("effective_evidence_chunk_ids") or []) or None,
                        ).items()
                    },
                    "section_dependency": (
                        list(container.get("word_bank") or [])
                        if qtype == "fill_in_the_blank" and isinstance(container, dict)
                        else None
                    ),
                }
                result[qid] = hashlib.sha256(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
            return result

        current_fingerprints = question_fingerprints()
        general_identity = str(getattr(general_client, "model", "default-general"))
        stem_identity = str(getattr(quantitative_client, "model", "default-stem"))
        def role_fingerprint(common: str, role: str) -> str:
            version = (STEM_VALIDATION_RULE_VERSION if role == "stem_quantitative"
                       else GENERAL_VALIDATION_RULE_VERSION)
            model = stem_identity if role == "stem_quantitative" else general_identity
            return hashlib.sha256(
                f"{common}|{role}|{version}|{VALIDATION_SCHEMA_VERSION}|{model}".encode("utf-8")
            ).hexdigest()
        model_ledger = validation_ledger.setdefault(str(mn), {})

        def run_reviews(ids: Optional[Iterable[str]]) -> Dict[str, Any]:
            if ids is not None:
                ids = list(ids)
            elif terminal_slots_for_model:
                ids = sorted(current_fingerprints)
            if stored_domain:
                cached = {
                    qid: deepcopy((entry or {}).get("specialized_roles") or {})
                    for qid, entry in model_ledger.items()
                }
                return run_specialized_reviews(
                    exam, ids, plan_by_slot, source_by_chunk, cached=cached,
                )
            args = (exam, ids, plan_by_slot, source_by_chunk)
            if quantitative_client is None:
                return (_runtime._validate_with_context(*args, general_client, "general")
                        if general_client is not None else _runtime._validate_with_context(*args))
            quantitative_ids = set(_quantitative_question_ids(exam, plan_by_slot))
            selected = [qid for qid in (ids or quantitative_ids) if qid in quantitative_ids]
            if not selected:
                return (_runtime._validate_with_context(*args, general_client, "general")
                        if general_client is not None else _runtime._validate_with_context(*args))
            active_quantitative_client = quantitative_client
            if ids is not None:
                # A repaired answer is checked in a fresh stateless Luna request.
                # No previous expected_fix or repair explanation is included in
                # build_validator_prompt; only the repaired question/certificate,
                # its plan, and bounded source evidence are sent.
                from app.llm.factory import create_review_client
                active_quantitative_client = create_review_client(
                    domain="STEM", stage="stem_quantitative_validator"
                )
            with ThreadPoolExecutor(max_workers=2) as executor:
                general_future = executor.submit(
                    copy_context().run, _validate_with_context, *args,
                    general_client, "general"
                )
                quantitative_future = executor.submit(
                    copy_context().run, _validate_with_context,
                    exam, selected, plan_by_slot, source_by_chunk,
                    active_quantitative_client, "stem_quantitative",
                )
                general = general_future.result()
                quantitative = quantitative_future.result()
            return _merge_role_reports(general, quantitative)
        if int(mn or 0) in candidate_updates_by_model:
            update = candidate_updates_by_model[int(mn or 0)]
            report = _merge_validation_report(previous, update) if previous is not None else update
            balanced_ids = balanced_candidate_ids_by_model.get(int(mn or 0), [])
            if balanced_ids:
                logger.info(
                    "Revalidating MCQs changed by post-repair balancing | "
                    "model=%d | question_count=%d | question_ids=%s",
                    mn,
                    len(balanced_ids),
                    balanced_ids,
                )
                report = _merge_validation_report(report, run_reviews(balanced_ids))
        elif mn in pending_revalidation:
            repaired_ids = list(pending_revalidation.pop(mn) or [])
            if repaired_ids:
                logger.info(
                    "Selective revalidation | model=%d | question_count=%d | question_ids=%s",
                    mn,
                    len(repaired_ids),
                    repaired_ids,
                )
                update = run_reviews(repaired_ids)
                report = _merge_validation_report(previous, update) if previous is not None else update
            elif previous is not None:
                report = previous
            else:
                report = run_reviews(None)
        elif int(mn or 0) in prefetched_initial_reports:
            update = prefetched_initial_reports[int(mn or 0)]
            report = _merge_validation_report(previous, update) if previous is not None else update
        elif stored_domain:
            # The specialized orchestrator performs dependency-aware reuse per
            # role, so all current questions may be submitted here without
            # repeating unchanged calls.
            update = run_reviews(None)
            report = _merge_validation_report(previous, update) if previous is not None else update
        else:
            quantitative_required_ids = set(
                _quantitative_question_ids(exam, plan_by_slot)
                if quantitative_client is not None else []
            )
            stale_ids = [
                qid for qid, fingerprint in current_fingerprints.items()
                if (
                    ((model_ledger.get(qid) or {}).get("role_fingerprints") or {}).get("general")
                    != role_fingerprint(fingerprint, "general")
                    or "general" not in ((model_ledger.get(qid) or {}).get("roles") or {})
                    or (
                        qid in quantitative_required_ids
                        and (
                            "stem_quantitative" not in ((model_ledger.get(qid) or {}).get("roles") or {})
                            or ((model_ledger.get(qid) or {}).get("role_fingerprints") or {}).get("stem_quantitative")
                            != role_fingerprint(fingerprint, "stem_quantitative")
                        )
                    )
                )
            ]
            if previous is not None and not stale_ids:
                logger.info("Validation ledger reuse | model=%d | unchanged=%d", mn, len(current_fingerprints))
                report = previous
            elif previous is not None:
                logger.info(
                    "Validation fingerprint recheck | model=%d | question_count=%d | question_ids=%s",
                    mn, len(stale_ids), stale_ids,
                )
                report = _merge_validation_report(previous, run_reviews(stale_ids))
            else:
                report = run_reviews(None)
        verdict_by_id_for_ledger = {
            str(verdict.get("question_id") or ""): verdict
            for verdict in (
                (report.get("general_review") or {}).get("verdicts")
                or report.get("verdicts") or []
            )
            if verdict.get("question_id")
        }
        quantitative_by_id = {
            str(verdict.get("question_id") or ""): verdict
            for verdict in ((report.get("quantitative_review") or {}).get("verdicts") or [])
            if verdict.get("question_id")
        }
        for qid, fingerprint in current_fingerprints.items():
            verdict = verdict_by_id_for_ledger.get(qid)
            if verdict is None:
                continue
            roles = {"general": deepcopy(verdict)}
            if qid in quantitative_by_id:
                roles["stem_quantitative"] = deepcopy(quantitative_by_id[qid])
            model_ledger[qid] = {
                "fingerprint": fingerprint,
                "role_fingerprints": {role: role_fingerprint(fingerprint, role) for role in roles},
                "roles": roles,
                "specialized_roles": {
                    role: {
                        "fingerprint": ((report.get("role_fingerprints") or {}).get(qid) or {}).get(role),
                        "verdict": deepcopy(verdict_value),
                    }
                    for role, verdict_value in ((report.get("role_results") or {}).get(qid) or {}).items()
                    if ((report.get("role_fingerprints") or {}).get(qid) or {}).get(role)
                },
            }
        report = accept_source_constrained_difficulty(report, plan_by_slot)
        plans = persist_difficulty_observations(plans, report, exam)
        present_slots = {
            str(item.get("slot_id") or "")
            for _qtype, _container, item in _iter_questions(exam.get("questions") or {})
            if item.get("slot_id")
        }
        expected_slots = set(plan_by_slot)
        missing_slot_ids = sorted(expected_slots - present_slots)
        missing_slots = []
        for slot_id in missing_slot_ids:
            plan_item = plan_by_slot.get(slot_id) or {}
            qtype = str(plan_item.get("question_type") or "")
            if not qtype:
                match = re.match(r"m\d+_(.+)_\d+$", slot_id)
                qtype = match.group(1) if match else "unknown"
            missing_slots.append({
                "slot_id": slot_id,
                "question_type": qtype,
                "task_id": plan_item.get("task_id"),
                "fitb_entry_id": plan_item.get("fitb_entry_id"),
                "fixed_answer": plan_item.get("fitb_answer"),
                "clue_basis": plan_item.get("clue_basis"),
                "evidence_chunk_id": plan_item.get("evidence_chunk_id")
                or plan_item.get("source_chunk_id"),
                "failure_code": "missing_requested_slot",
                "recovery_requirement": (
                    "Regenerate this exact slot with its fixed FITB answer; replace only its "
                    "fitb_entry_id from the candidate bank if the assigned entry is exhausted."
                    if qtype == "fill_in_the_blank"
                    else "Regenerate this exact requested slot from its authoritative plan item."
                ),
            })
        report["missing_slots"] = missing_slots
        report["missing_sections"] = sorted({
            qtype for qtype, requested in (
                (str((plan_by_slot.get(slot_id) or {}).get("question_type") or
                     (re.match(r"m\d+_(.+)_\d+$", slot_id).group(1)
                      if re.match(r"m\d+_(.+)_\d+$", slot_id) else "unknown")), slot_id)
                for slot_id in missing_slot_ids
            )
            if qtype != "unknown" and not any(
                str(item.get("slot_id") or "").startswith(f"m{mn}_{qtype}_")
                for _kind, _container, item in _iter_questions(exam.get("questions") or {})
            )
        })
        if missing_slots:
            report["all_pass"] = False
        fresh_manifest = _manifest_from_report(report, exam, plan_by_slot)
        # Resolved staged candidates are absent from fresh problems. Preserve
        # their authoritative closure while the exact accepted candidate is
        # unchanged.  Other slots may keep the graph alive for several rounds;
        # that must not resurrect superseded findings for this slot.  A later
        # content change produces a different fingerprint and invalidates the
        # closure normally.
        resolved_old = [
            item for item in repair_manifest if item.get("status") == "resolved"
        ]
        resolved_fingerprints = {
            (int(item.get("model_number") or 0), str(item.get("slot_id") or "")):
            str(item.get("resolved_candidate_fingerprint") or "")
            for item in resolved_old
        }
        current_by_slot = {
            str(item.get("slot_id") or ""): hashlib.sha256(json.dumps(
                item, ensure_ascii=False, sort_keys=True, default=str,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            for _qtype, _container, item in _iter_questions(exam.get("questions") or {})
            if item.get("slot_id")
        }
        passing_ids = {
            str(verdict.get("question_id") or "")
            for verdict in report.get("verdicts") or []
            if verdict.get("action") == "PASS"
        }
        for item in resolved_old:
            if (
                int(item.get("model_number") or 0) == int(mn or 0)
                and str(item.get("question_id") or "") in passing_ids
            ):
                current_fingerprint = current_by_slot.get(
                    str(item.get("slot_id") or "")
                )
                if current_fingerprint:
                    item["resolved_candidate_fingerprint"] = current_fingerprint
                    resolved_fingerprints[(
                        int(item.get("model_number") or 0),
                        str(item.get("slot_id") or ""),
                    )] = current_fingerprint
        protected_resolved_keys = {
            key for key, fingerprint in resolved_fingerprints.items()
            if fingerprint and current_by_slot.get(key[1]) == fingerprint
        }
        fresh_manifest = [
            item for item in fresh_manifest
            if (int(item.get("model_number") or 0), str(item.get("slot_id") or ""))
            not in protected_resolved_keys
        ]
        unresolved_old = [
            item for item in repair_manifest if item.get("status") != "resolved"
        ]
        protected_resolved = [
            item for item in resolved_old
            if (int(item.get("model_number") or 0), str(item.get("slot_id") or ""))
            in protected_resolved_keys
        ]
        repair_manifest = _merge_repair_manifest(
            [*unresolved_old, *protected_resolved], fresh_manifest,
        )
        unresolved_for_model = [
            item for item in repair_manifest
            if int(item.get("model_number") or 0) == int(mn or 0)
            and item.get("status") != "resolved"
        ]
        report["all_pass"] = (
            all(verdict.get("action") == "PASS" for verdict in report.get("verdicts") or [])
            and not report.get("missing_slots") and not report.get("missing_sections")
            and not unresolved_for_model
        )
        reports_by_model[mn] = report
        if eval_stats is not None:
            record_first_validation(eval_stats, exam, report)
        if report["all_pass"]:
            validated.append(mn)
        elif any(
            int(item.get("model_number") or 0) == int(mn or 0)
            and item.get("status") != "resolved"
            for item in repair_manifest
        ) and all(
            (attempts.get(str(item.get("slot_id") or "")) or 0) >= MAX_QUESTION_REPAIR_ATTEMPTS
            for item in repair_manifest
            if int(item.get("model_number") or 0) == int(mn or 0)
            and item.get("status") != "resolved"
        ):
            warnings.append(
                f"Model {mn}: max question repair attempts reached; "
                "keeping best-effort questions."
            )

    if stored_domain:
        repair_manifest, validated, warnings, duplicate_result = _integrate_duplicate_validation(
            {**state, "duplicate_fact_cache": candidate_duplicate_fact_cache},
            generated, plans, reports_by_model, repair_manifest, validated, warnings,
        )
    else:
        duplicate_result = {
            "duplicate_reports": list(state.get("duplicate_reports") or []),
            "duplicate_similarity_distributions": list(
                state.get("duplicate_similarity_distributions") or []
            ),
            "duplicate_fact_cache": dict(candidate_duplicate_fact_cache),
            "duplicate_scan_count": int(state.get("duplicate_scan_count") or 0),
            "unresolved_duplicate_conflicts": list(
                state.get("unresolved_duplicate_conflicts") or []
            ),
        }

    # Every model first completes the unchanged within-model Duplicate V2
    # contract.  Only then compare finalized actual questions across models.
    # The coordinator persists the complete finding set before placing any
    # later-model question into the existing bounded recovery manifest.
    repair_manifest, validated, warnings, cross_model_result = (
        integrate_cross_model_duplicates(
            {**state, "plans": plans}, generated, reports_by_model,
            repair_manifest, validated, warnings,
        )
    )

    return {
        "generated_exams": generated,
        "plans": plans,
        "validation_reports": list(reports_by_model.values()),
        "model_lifecycle_status": {
            int(exam.get("model_number") or 0): {
                "status": "locally_ready" if int(exam.get("model_number") or 0) in set(validated)
                else "recovery_pending",
            }
            for exam in generated
        },
        "repair_manifest": repair_manifest,
        "repair_candidates": pending_candidates,
        "question_repair_attempts": legacy_attempts,
        "question_repair_calls": repair_calls,
        "question_repair_failures": failures,
        "slot_completion_attempts": completion_attempts,
        "repair_working_candidates": working_candidates,
        "revalidation_progress": revalidation_progress,
        "blocked_revalidation_candidates": blocked_candidates,
        "repair_replacement_memory": replacement_memory,
        "repair_retry_seeds": retry_seeds,
        "validated_models": validated,
        "pending_revalidation_ids": pending_revalidation,
        "warnings": warnings,
        "eval_stats": eval_stats,
        "initial_global_reviews": list(state.get("initial_global_reviews") or []),
        "final_global_reviews": list(state.get("final_global_reviews") or []),
        "exam_quality_findings": list(state.get("exam_quality_findings") or []),
        "validation_repair_history": validation_repair_history,
        "validation_ledger": validation_ledger,
        "validation_call_logs": [
            trace
            for report in reports_by_model.values()
            for trace in (report.get("validation_call_logs") or [])
        ],
        "repair_call_logs": repair_call_logs[-VALIDATOR_TRACE_MAX_ATTEMPTS:],
        "duplicate_reports": duplicate_result.get("duplicate_reports") or [],
        "duplicate_findings": duplicate_result.get("duplicate_findings") or [],
        "duplicate_similarity_distributions": duplicate_result.get("duplicate_similarity_distributions") or [],
        "duplicate_fact_cache": duplicate_result.get("duplicate_fact_cache") or {},
        "duplicate_scan_count": duplicate_result.get("duplicate_scan_count") or 0,
        "pending_duplicate_groups": [],
        "unresolved_duplicate_conflicts": duplicate_result.get("unresolved_duplicate_conflicts") or [],
        "duplicate_candidate_blacklist": duplicate_result.get("duplicate_candidate_blacklist") or {},
        "duplicate_replacement_failures": duplicate_result.get("duplicate_replacement_failures") or {},
        "duplicate_repair_history": duplicate_result.get("duplicate_repair_history") or [],
        **cross_model_result,
        "performance_telemetry": [
            *(state.get("performance_telemetry") or []), *performance_events,
        ][-5000:],
    }


