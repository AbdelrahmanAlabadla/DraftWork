"""Recovery policy, candidate staging, and repair coordination."""
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
from . import runtime as _runtime

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




def _safe_record_repair_outcome(
    stats: Any, exam: Any, report: Any,
    repaired_ids: Iterable[str], failed_ids: Iterable[str],
    sent_ids: Iterable[str] | None = None,
) -> None:
    """Record auxiliary Repair telemetry without aborting validation."""
    if not isinstance(stats, dict):
        return
    try:
        _runtime.record_repair_outcome(
            stats, exam, report, repaired_ids, failed_ids, sent_ids=sent_ids,
        )
    except Exception as exc:
        _runtime.record_optional_failure(
            exc, stage="general_validation", component="repair_telemetry",
            continued=True,
        )
        logger.warning(
            "repair telemetry failed | error=%s: %s",
            type(exc).__name__, exc, exc_info=True,
        )


def _defect_family(failure: dict[str, Any]) -> str:
    code = str(failure.get("failure_code") or "").casefold()
    role = str(failure.get("validation_role") or "").casefold()
    if (
        code in {
            "unsupported", "contradicted", "insufficient_evidence",
            "unsupported_given", "constraint_violation", "unit_inconsistency",
            "missing_units", "wrong_givens", "evidence_mismatch",
        }
        or "evidence" in code
        or role == "grounding"
    ):
        return "source_or_evidence"
    if role in {"answer_correctness", "quantitative"} or code in {
        "incorrect_answer", "incorrect_calculation", "incorrect_result",
        "answer_inconsistent", "incomplete_solution", "invalid_solution_certificate",
        "invalid_equation", "rounding_error", "significant_figures_error",
    }:
        return "answer_or_calculation"
    if role == "task_alignment" or code in {
        "task_drift", "wrong_variation", "wrong_task_identity",
        "question_type_operation_mismatch", "copied_source_example",
        "visual_dependency",
    }:
        return "task_alignment"
    if role == "duplicate" or code in {"duplicate_question", "excessive_overlap"}:
        return "duplicate"
    return "content"


def _defect_signature(failure: dict[str, Any]) -> str:
    """Objective signature of the unresolved defect, not its wording."""
    fields = sorted({
        str(field).split(".", 1)[0].split("[", 1)[0]
        for field in failure.get("fields_to_fix") or []
        if str(field).strip()
    })
    return "|".join((
        str(failure.get("validation_role") or "deterministic"),
        _defect_family(failure),
        ",".join(fields),
    ))


def recovery_eligibility(
    target: Dict[str, Any], *, failures: int, calls: int,
    completion_attempts: int, completion_enabled: bool,
    content_attempt_limit: int = MAX_QUESTION_REPAIR_ATTEMPTS,
) -> Dict[str, Any]:
    """Return the one authoritative, route-specific recovery budget decision.

    Content Repair and assignment replacement are independent budgets.  An
    exhausted replacement budget never becomes available merely because the
    content-Repair budget has capacity, and a replacement-only diagnosis never
    falls through to an unrelated content edit.
    """
    status = str(target.get("status") or "")
    terminal = status in {
        "resolved", "exhausted", "terminal_failure", "failed_to_repair",
    }
    if (
        status == "terminal_failure"
        and str(target.get("failure_code") or "")
        == "content_repair_budget_exhausted"
        and completion_enabled
        and int(completion_attempts) < MAX_COMPLETION_ATTEMPTS_PER_SLOT
    ):
        # Compatibility for checkpoints that used terminal_failure to mean
        # content-terminal before the independent completion budget existed.
        terminal = False
    content_allowed = (
        not terminal
        and status not in {"validation_blocked", "completion_pending"}
        and int(failures) < int(content_attempt_limit)
        and int(calls) < MAX_CONTENT_REPAIR_CALLS_PER_SLOT
    )
    replacement_allowed = (
        not terminal
        and (
            int(completion_attempts) < MAX_COMPLETION_ATTEMPTS_PER_SLOT
            if completion_enabled else content_allowed
        )
    )
    return {
        "eligible": content_allowed or replacement_allowed,
        "selected_route": None,
        "content_allowed": content_allowed,
        "replacement_allowed": replacement_allowed,
        "content_exhausted": not content_allowed,
        "completion_exhausted": not replacement_allowed,
        "terminal": terminal,
    }


def recovery_v3_dispatch_action(strategy: str) -> str:
    """Translate Task Alignment's semantic choice into an existing executor."""
    normalized = str(strategy or "").upper()
    try:
        return {
            "REPAIR": "repair_question",
            "SWITCH_VARIATION": "switch_variation",
            "REGENERATE": "regenerate_missing_slot",
            "SWITCH_TASK": "replace_assignment",
        }[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Recovery V3 strategy: {strategy}") from exc


def copied_source_repair_options(
    plan_item: Dict[str, Any], question_type: str,
    *, available_evidence_ids: Set[str] | None = None,
) -> Dict[str, Any]:
    """Objectively inventory authorized ways to create a different instance.

    This does not decide educational equivalence.  It only answers whether the
    Catalog exposes any materializable knob that Luna can semantically assess.
    Ambiguous legacy contracts therefore fail open to targeted Repair.
    """
    variations = [
        value for value in (
            plan_item.get("permitted_variations")
            or plan_item.get("allowed_variations") or []
        ) if isinstance(value, dict)
    ]
    selected_id = str(plan_item.get("selected_variation_id") or "")
    selected = next((
        value for value in variations
        if str(value.get("variation_id") or "") == selected_id
    ), None)
    if selected is None and len(variations) == 1 and not selected_id:
        selected = variations[0]
    from app.online.models import numeric_variation_allowed
    numerical = numeric_variation_allowed(plan_item)

    materializable = eligible_task_variation_candidates(
        {**plan_item, "text_answerable": plan_item.get("text_answerable", True)},
        question_type,
        variations=variations,
    )
    # A persisted permitted_variations list is already the Catalog's
    # authorization boundary. Include complete alternate operations even when
    # an older shared filter conservatively expects the parent operation.
    for variation in variations:
        evidence = variation.get("evidence_chunk_ids") or plan_item.get("evidence_chunk_ids")
        complete = all((
            variation.get("operation") or plan_item.get("task_operation"),
            variation.get("target") or plan_item.get("target"),
            variation.get("student_work_product") or plan_item.get("student_work_product"),
            variation.get("reasoning_mode") or plan_item.get("reasoning_mode"),
            evidence,
        ))
        types = variation.get("question_types") or []
        evidence_available = (
            available_evidence_ids is None
            or bool(evidence)
            and set(map(str, evidence)).issubset(available_evidence_ids)
        )
        if (complete and evidence_available
                and variation_response_form_compatible(variation, question_type)
                and (not types or question_type in types)
                and variation not in materializable):
            materializable.append(variation)
    selected_signature = (
        str((selected or {}).get("operation") or plan_item.get("task_operation") or ""),
        str((selected or {}).get("target") or plan_item.get("target") or ""),
        tuple(map(str, (selected or {}).get("givens") or [])),
        str((selected or {}).get("student_work_product") or plan_item.get("student_work_product") or ""),
        str((selected or {}).get("reasoning_mode") or plan_item.get("reasoning_mode") or ""),
    )
    alternatives = []
    for variation in materializable:
        if not variation_response_form_compatible(variation, question_type):
            continue
        variation_id = str(variation.get("variation_id") or "")
        signature = (
            str(variation.get("operation") or plan_item.get("task_operation") or ""),
            str(variation.get("target") or plan_item.get("target") or ""),
            tuple(map(str, variation.get("givens") or [])),
            str(variation.get("student_work_product") or plan_item.get("student_work_product") or ""),
            str(variation.get("reasoning_mode") or plan_item.get("reasoning_mode") or ""),
        )
        evidence = variation.get("evidence_chunk_ids") or plan_item.get("evidence_chunk_ids") or []
        if available_evidence_ids is not None and (
            not evidence or not set(map(str, evidence)).issubset(available_evidence_ids)
        ):
            continue
        if variation_id != selected_id and signature != selected_signature:
            alternatives.append(variation_id or f"variation_{len(alternatives) + 1}")
    # If the legacy contract is incomplete, avoid declaring Repair impossible.
    contract_complete = bool(selected and variations and selected_id)
    current_repairable = numerical or not contract_complete
    alternative_available = bool(alternatives)
    return {
        # Retained for checkpoint/test compatibility: this says some recovery
        # path exists, not that the immutable current variation may be edited.
        "repairable": current_repairable or alternative_available,
        "current_variation_repairable": current_repairable,
        "alternative_variation_available": alternative_available,
        "numeric_instance_authorized": numerical,
        "alternative_variation_ids": alternatives,
        "replace_or_switch_required": not current_repairable,
        "contract_complete": contract_complete,
    }


def _store_plan_item(
    plans: list[Dict[str, Any]], model_number: int, qtype: str,
    slot_id: str, item: Dict[str, Any],
) -> bool:
    """Write a switched plan item back to its canonical model partition."""
    clean = {
        key: deepcopy(value) for key, value in item.items()
        if key != "assessment_guidance"
    }
    for plan in plans:
        if int(plan.get("model_number") or 0) != int(model_number):
            continue
        values = (plan.get("items") or {}).get(qtype) or []
        for index, existing in enumerate(values):
            if str(existing.get("slot_id") or "") == slot_id:
                values[index] = clean
                return True
    return False


def _switch_recovery_variation(
    plan_item: Dict[str, Any], variation_id: str, question_type: str,
    source_by_chunk: Dict[str, str],
) -> bool:
    """Materialize one Catalog-authorized alternative on the existing task."""
    variations = [
        value for value in (
            plan_item.get("permitted_variations")
            or plan_item.get("allowed_variations") or []
        ) if isinstance(value, dict)
    ]
    options = copied_source_repair_options(
        plan_item, question_type,
        available_evidence_ids=set(map(str, source_by_chunk)),
    )
    if variation_id not in set(options["alternative_variation_ids"]):
        return False
    variation = next((
        value for value in variations
        if str(value.get("variation_id") or "") == variation_id
    ), None)
    if variation is None:
        return False
    previous_id = str(plan_item.get("selected_variation_id") or "")
    _apply_effective_variation(plan_item, plan_item, variation)
    plan_item["selected_variation"] = deepcopy(variation)
    plan_item["selected_variation_id"] = variation_id
    plan_item.pop("effective_instance_id", None)
    plan_item.pop("numerical_instance_fingerprint", None)
    plan_item["recovery_variation_switch"] = {
        "from_variation_id": previous_id,
        "to_variation_id": variation_id,
        "reason": "copied_source_example",
    }
    return True


GENERAL_VALIDATION_RULE_VERSION = "general-v3-illustrative-values"
STEM_VALIDATION_RULE_VERSION = "stem-v3-illustrative-values"


def _mark_blocked_repair_route(
    target: Dict[str, Any], memory: Dict[str, Any], *,
    route_status: str, input_fingerprint: str, reason: str,
) -> None:
    """Persist a blocked local route as replacement-only recovery."""
    target.update(
        status="validation_blocked",
        failure_code="validation_blocked",
        failure_reason=reason,
        repair_type="replace_assignment",
    )
    memory["blocked_route_fingerprint"] = input_fingerprint
    memory["blocked_route_status"] = route_status


def _normalize_replacement_difficulty_state(
    item: Dict[str, Any], requested: str,
) -> None:
    """Make a replacement's Group 2 difficulty state explicit and terminal.

    Replacement happens after the graph's design node. A pending design cannot
    be silently left behind, and legacy adaptation metadata must not leak into
    the new plan item. The native level is retained; Luna records the observed
    level during validation.
    """
    native = str(
        item.get("native_difficulty_prior")
        or item.get("difficulty")
        or item.get("actual_difficulty")
        or "medium"
    )
    requested_level = str(requested or item.get("requested_difficulty") or "mix")
    design = dict(item.get("difficulty_design") or {})
    if not design:
        design = {
            "status": (
                "pending" if native != requested_level
                and requested_level in {"easy", "medium", "hard"}
                else "not_required"
            ),
            "policy_version": "difficulty-design-v1",
            "native_prior": native,
            "requested": requested_level,
        }
    else:
        design.setdefault("native_prior", native)
        design.setdefault("requested", requested_level)
        if (
            str(design.get("status") or "") == "pending"
            and native == requested_level
        ):
            design["status"] = "not_required"
    item["requested_difficulty"] = requested_level
    item["native_difficulty_prior"] = native
    item["actual_difficulty"] = native
    item["difficulty_design"] = design
    item["difficulty_adaptation"] = None
    item["difficulty_status"] = (
        "adaptation_pending"
        if design.get("status") == "pending"
        else "not_required"
    )


def _apply_replacement_difficulty_design(
    staged_plans: list[Dict[str, Any]], *,
    model_number: int, qtype: str, slot_id: str,
    state: Dict[str, Any], requested: str,
) -> Dict[str, Any] | None:
    """Run the existing Group 2 design pass for a replacement before generation."""
    target = next((
        item
        for plan in staged_plans
        if int(plan.get("model_number") or 0) == int(model_number)
        for values in (plan.get("items") or {}).values()
        for item in values or []
        if str(item.get("slot_id") or "") == slot_id
    ), None)
    if target is None:
        return None
    design = dict(target.get("difficulty_design") or {})
    if str(design.get("status") or "") != "pending":
        return target
    from app.online.difficulty_adjustment import adjust_plan_difficulty

    single_plan = {
        "model_number": int(model_number),
        "items": {qtype: [deepcopy(target)]},
    }
    try:
        result = adjust_plan_difficulty({
            **state, "plans": [single_plan], "difficulty": requested,
        })
        adjusted = next((
            item
            for plan in result.get("plans") or []
            for values in (plan.get("items") or {}).values()
            for item in values or []
            if str(item.get("slot_id") or "") == slot_id
        ), None)
        if adjusted is not None:
            target.update(adjusted)
    except Exception as exc:
        _runtime.record_optional_failure(
            exc, stage="replacement_difficulty", component="replacement_design",
            job_id=str(state.get("job_id") or ""), continued=True,
        )
        target["difficulty_design"] = {
            **design,
            "status": "pending",
            "reason": f"Difficulty design unavailable for replacement: {type(exc).__name__}",
        }
        target["difficulty_status"] = "adaptation_pending"
        target["difficulty_design_operational_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return target
    if str((target.get("difficulty_design") or {}).get("status") or "") == "pending":
        # The design process did not establish a terminal result. Do not turn
        # an unresolved/operational state into a source-constrained claim.
        target["difficulty_status"] = "adaptation_pending"
    elif str((target.get("difficulty_design") or {}).get("status") or "") == "designed":
        target["difficulty_status"] = "designed"
        target.pop("difficulty_design_operational_error", None)
    return target


def _replacement_difficulty_unresolved(item: Dict[str, Any] | None) -> bool:
    if not item:
        return False
    design_status = str((item.get("difficulty_design") or {}).get("status") or "")
    return (
        design_status == "pending"
        or str(item.get("difficulty_status") or "") == "adaptation_pending"
    )


def _mark_replacement_difficulty_blocked(
    target: Dict[str, Any], item: Dict[str, Any],
) -> None:
    """Route an unavailable design step through bounded operational recovery."""
    reason = str(
        item.get("difficulty_design_operational_error")
        or (item.get("difficulty_design") or {}).get("reason")
        or "Replacement difficulty design did not reach a terminal state."
    )
    target.update(
        status="validation_blocked",
        repair_type="replace_assignment",
        failure_code="replacement_difficulty_operational_error",
        failure_reason=reason,
    )


def _finalize_replacement_difficulty_fallback(
    plan_item: Dict[str, Any] | None, *, requested: str,
    adjustment_audit: Dict[str, Any],
) -> bool:
    """Keep a grounded assignment at its honest source-constrained level."""
    if plan_item is None:
        return False
    from app.online.assessment_catalog import apply_difficulty_fallback_metadata
    achieved = str(
        plan_item.get("actual_difficulty") or plan_item.get("difficulty") or "medium"
    )
    failure = str(
        adjustment_audit.get("failure")
        or "The source-supported assignment cannot be adapted further."
    )
    apply_difficulty_fallback_metadata(
        plan_item, requested or achieved, achieved, failure,
        strategy="replacement_source_constrained_fallback",
    )
    return True







def _repair_invalid_questions_serial(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stage bounded, reason-aware candidates without mutating the active exam."""
    _ensure_generation_active(state)
    generated = state.get("generated_exams") or []
    plans = deepcopy(state.get("plans") or [])
    global_generated = state.get("global_generated_exams") or generated
    global_plans = state.get("global_plans") or plans
    manifest = deepcopy(state.get("repair_manifest") or [])
    # Legacy checkpoints stored one counter. Treat it as failed strategies;
    # new checkpoints persist calls and failures independently.
    legacy_attempts = dict(state.get("question_repair_attempts") or {})
    failures = dict(
        state.get("question_repair_failures")
        or state.get("question_repair_attempts") or {}
    )
    repair_calls = dict(state.get("question_repair_calls") or {})
    completion_attempts = dict(state.get("slot_completion_attempts") or {})
    completion_enabled = "slot_completion_attempts" in state
    working_candidates = deepcopy(state.get("repair_working_candidates") or {})
    memory_by_slot = deepcopy(state.get("repair_replacement_memory") or {})
    history = list(state.get("validation_repair_history") or [])
    repair_call_logs = list(state.get("repair_call_logs") or [])
    warnings = list(state.get("warnings") or [])
    incompatible_task_ids = {
        str(value) for value in state.get("incompatible_task_ids") or [] if value
    }
    candidates: list[Dict[str, Any]] = []
    retry_seeds = deepcopy(state.get("repair_retry_seeds") or {})
    validated_models = list(state.get("validated_models") or [])
    exam_by_model = {int(exam.get("model_number") or 0): exam for exam in generated}

    for target in manifest:
        _ensure_generation_active(state)
        if target.get("status") in {"resolved", "validation_pending"}:
            continue
        recovery_v3 = target.get("recovery_v3") or {}
        if recovery_v3.get("status") != "ready":
            target["status"] = "recovery_decision_blocked"
            continue
        slot_id = str(target.get("slot_id") or "")
        duplicate = str(target.get("validation_role") or "") == "duplicate"
        attempt_limit = (
            MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT
            if duplicate
            else MAX_QUESTION_REPAIR_ATTEMPTS
        )
        if not slot_id:
            continue
        eligibility = recovery_eligibility(
            target,
            failures=int(failures.get(slot_id) or 0),
            calls=int(repair_calls.get(slot_id) or 0),
            completion_attempts=int(completion_attempts.get(slot_id) or 0),
            completion_enabled=completion_enabled,
            content_attempt_limit=attempt_limit,
        )
        content_exhausted = bool(eligibility["content_exhausted"])
        completion_exhausted = bool(eligibility["completion_exhausted"])
        # Duplicate content attempts and assignment-completion attempts are
        # independent budgets. A duplicate slot with an unused backup or
        # replacement must escalate instead of becoming terminal here.
        if not eligibility["eligible"]:
            if slot_id:
                target.update(
                    status="terminal_failure",
                    failure_code="no_valid_source_grounded_replacement",
                    failure_reason=(
                        "Content repair and the bounded on-demand completion "
                        "budget were exhausted without a valid grounded candidate."
                    ),
                )
            continue
        mn = int(target.get("model_number") or 1)
        active_exam = exam_by_model.get(mn)
        if active_exam is None:
            continue
        previous_attempts = int(failures.get(slot_id) or 0)
        attempt = int(legacy_attempts.get(slot_id) or 0) + 1
        qtype = str(target.get("question_type") or _question_type_from_slot(slot_id))
        prefix = f"m{mn}_"
        qid = str(target.get("question_id") or f"model{mn}_{slot_id[len(prefix):] if slot_id.startswith(prefix) else slot_id}")
        retry_seed = retry_seeds.get(slot_id) or {}
        staged_exam = deepcopy(retry_seed.get("exam") or active_exam)
        staged_plans = deepcopy(retry_seed.get("plans") or plans)
        durable_working = deepcopy(
            (working_candidates.get(slot_id) or {}).get("candidate")
        )
        if durable_working:
            # Resume from the latest Python-accepted content revision, never
            # from the historical original or a rejected later proposal.
            _put_question(staged_exam, qtype, durable_working)
        staged_state = {**state, "plans": staged_plans}
        plan_by_slot, source_by_chunk = _plan_context_for_model(staged_state, mn)
        if slot_id in plan_by_slot:
            plan_by_slot[slot_id]["recovery_failure_code"] = target.get("failure_code")
            plan_by_slot[slot_id]["recovery_failure_reason"] = target.get("failure_reason")
            plan_by_slot[slot_id]["recovery_attempt"] = attempt
        original_entry = _build_by_id(active_exam.get("questions") or {}).get(qid)
        had_original = original_entry is not None
        original_question = deepcopy((original_entry or (None, None, {}))[2])
        memory = memory_by_slot.setdefault(slot_id, {
            "original_candidate_id": target.get("fitb_entry_id") or target.get("task_id"),
            "attempted_candidate_ids": [], "blacklisted_candidate_ids": [],
            "candidate_failures": [], "replacement_attempt_count": 0,
            "assignment_validated_attempts": {}, "rejected_variation_ids": [],
        })
        plan = list(recovery_v3.get("plan") or [])
        cursor = int(recovery_v3.get("next_index") or 0)
        while cursor < len(plan):
            strategy = str(plan[cursor] or "").upper()
            if strategy in {"REPAIR", "REGENERATE", "SWITCH_VARIATION"} and not eligibility["content_allowed"]:
                cursor += 1
                continue
            if strategy == "SWITCH_TASK" and not eligibility["replacement_allowed"]:
                cursor += 1
                continue
            break
        recovery_v3["next_index"] = cursor
        if cursor >= len(plan):
            target.update(
                status="terminal_failure", failure_code="recovery_v3_plan_exhausted",
                failure_reason="The saved Task Alignment recovery plan and authorized budgets are exhausted.",
            )
            continue
        selected_strategy = str(plan[cursor]).upper()
        repair_action = recovery_v3_dispatch_action(selected_strategy)
        target["repair_type"] = repair_action
        target["recovery_routing_reason"] = "task_alignment_recovery_v3"
        active_assignment = plan_by_slot.get(slot_id) or {}
        switched_variation_id = ""
        # SWITCH_TASK assignment identity is selected deterministically by
        # Python's shared Catalog ranker. Legacy switch_preference values are
        # audit-only and are never interpreted as variation IDs.
        if repair_action == "replace_assignment":
            target["status"] = "completion_pending"
        repair_strategy = {
            "REPAIR": "recovery_v3_targeted_repair",
            "SWITCH_VARIATION": "recovery_v3_switch_variation",
            "REGENERATE": "recovery_v3_regenerate_assignment",
            "SWITCH_TASK": ("recovery_v3_switch_variation" if switched_variation_id
                            else "recovery_v3_switch_assignment"),
        }[selected_strategy]
        difficulty_findings = (
            (target.get("verdict") or {}).get("required_fixes")
            or target.get("required_fixes") or []
        )
        if (
            selected_strategy == "REPAIR"
            and difficulty_findings
            and target.get("history")
            and all(
                str(issue.get("validator") or "") == "difficulty"
                for issue in difficulty_findings
            )
        ):
            # A difficulty-only mismatch is a policy observation, not a content
            # repair target. Preserve the existing terminal-accept behavior.
            repair_action = "accept_difficulty"
            repair_strategy = "accepted_source_constraint"
        if selected_strategy == "REPAIR" and repair_action == "repair_question":
            # Strategy remains Task Alignment's decision. This is a technical
            # executability check only: a blocked route must not consume calls
            # or hide the saved REGENERATE/SWITCH_TASK fallbacks.
            preflight_verdict = deepcopy(target.get("verdict") or {})
            route_metadata_available = bool(
                preflight_verdict.get("required_fixes")
                or str(target.get("failure_code") or "") == "cross_model_duplicate"
                or str(target.get("original_failure_code") or "") == "cross_model_duplicate"
            )
            route = {"status": "LEGACY_ROUTE", "steps": ["legacy"]}
            if route_metadata_available:
                from app.online.multi_repair import (
                    build_repair_route, reconcile_repair_conflicts,
                )
                if preflight_verdict.get("repair_instruction_conflicts") and original_question:
                    preflight_verdict, _ = reconcile_repair_conflicts(
                        qtype, preflight_verdict, original_question,
                    )
                route = build_repair_route(qtype, preflight_verdict)
            if route_metadata_available and (
                str(route.get("status") or "") != "READY" or not route.get("steps")
            ):
                target.setdefault(
                    "original_failure_code", target.get("failure_code")
                )
                recovery_v3.setdefault("attempted_strategies", []).append("REPAIR")
                recovery_v3["next_index"] = cursor + 1
                target["recovery_v3"] = recovery_v3
                target["status"] = "invalid"
                target["failure_code"] = "repair_route_unavailable"
                target["failure_reason"] = (
                    "The specialized Repair Router has no executable route; "
                    "Recovery advanced to the saved fallback without a provider call."
                )
                event = {
                    "model_number": mn, "slot_id": slot_id, "question_id": qid,
                    "attempt": attempt, "repair_action": "repair_route_preflight",
                    "repair_strategy": repair_strategy, "status": "strategy_skipped",
                    "new_failure_code": "repair_route_unavailable",
                    "original_failure_code": target.get("original_failure_code"),
                    "route_status": route.get("status"),
                }
                target.setdefault("history", []).append(deepcopy(event))
                history.append(event)
                continue
        if selected_strategy == "SWITCH_VARIATION":
            current_plan = plan_by_slot.get(slot_id) or {}
            options = copied_source_repair_options(
                current_plan, qtype,
                available_evidence_ids=set(source_by_chunk),
            )
            rejected_variations = set(memory.get("rejected_variation_ids") or [])
            alternatives = [
                str(variation_id)
                for variation_id in options.get("alternative_variation_ids") or []
                if str(variation_id) and str(variation_id) not in rejected_variations
            ]
            switched_variation_id = next(iter(alternatives), "")
            if not switched_variation_id or not _switch_recovery_variation(
                current_plan, switched_variation_id, qtype, source_by_chunk,
            ):
                switched_variation_id = ""
                recovery_v3.setdefault("attempted_strategies", []).append("SWITCH_VARIATION")
                recovery_v3["next_index"] = cursor + 1
                recovery_v3["last_failure_code"] = "variation_candidates_exhausted"
                recovery_v3["last_failure_reason"] = (
                    "No eligible non-rejected same-task variation remained."
                )
                target["recovery_v3"] = recovery_v3
                target.update(
                    status="invalid",
                    failure_code="variation_candidates_exhausted",
                    failure_reason=recovery_v3["last_failure_reason"],
                    repair_type="switch_variation",
                )
                event = {
                    "model_number": mn, "slot_id": slot_id, "question_id": qid,
                    "attempt": attempt, "repair_action": "switch_variation",
                    "repair_strategy": "recovery_v3_switch_variation",
                    "status": "strategy_skipped",
                    "new_failure_code": "variation_candidates_exhausted",
                    "new_failure_reason": recovery_v3["last_failure_reason"],
                }
                target.setdefault("history", []).append(deepcopy(event))
                history.append(event)
                continue
            if not _store_plan_item(
                staged_plans, mn, qtype, slot_id, current_plan,
            ):
                switched_variation_id = ""
                target.update(
                    status="invalid",
                    failure_code="variation_plan_write_failed",
                    failure_reason="Switched variation could not be written to the plan.",
                )
                continue
            target["status"] = "completion_pending"
            target["repair_type"] = "switch_variation"
            repair_action = "switch_variation"
            repair_strategy = "recovery_v3_switch_variation"
            staged_state = {**state, "plans": staged_plans}
            plan_by_slot, source_by_chunk = _plan_context_for_model(staged_state, mn)
        if (
            repair_action == "replace_assignment"
            and not eligibility["replacement_allowed"]
        ):
            target.update(
                status="terminal_failure",
                failure_code="replacement_budget_exhausted",
                failure_reason=(
                    "The bounded assignment-replacement budget is exhausted; "
                    "no additional replacement may be scheduled."
                ),
            )
            continue
        if (
            repair_action != "replace_assignment"
            and not eligibility["content_allowed"]
        ):
            target.update(
                status="terminal_failure",
                failure_code="content_repair_budget_exhausted",
                failure_reason="The bounded content-recovery budget is exhausted.",
            )
            continue
        use_replacement = repair_action == "replace_assignment"
        def repair_input_fingerprint() -> str:
            return hashlib.sha256(json.dumps({
                "question": staged_exam,
                "failure_code": target.get("failure_code"),
                "required_fixes": (target.get("verdict") or {}).get("required_fixes") or [],
                "fields_to_fix": (target.get("verdict") or {}).get("fields_to_fix") or [],
                "repair_action": repair_action,
                "repair_strategy": repair_strategy,
                "blacklisted_candidate_ids": sorted(
                    map(str, memory.get("blacklisted_candidate_ids") or [])
                ),
                "attempted_candidate_ids": sorted(
                    map(str, memory.get("attempted_candidate_ids") or [])
                ),
                "selected_variation_id": str(
                    (plan_by_slot.get(slot_id) or {}).get("selected_variation_id") or ""
                ),
            }, ensure_ascii=False, sort_keys=True, default=str,
                separators=(",", ":")).encode("utf-8")).hexdigest()

        attempted_inputs = set(memory.setdefault("attempted_repair_input_fingerprints", []))
        unresolved_fingerprint = repair_input_fingerprint()
        if unresolved_fingerprint in attempted_inputs:
            recovery_v3.setdefault("attempted_strategies", []).append(selected_strategy)
            recovery_v3["next_index"] = cursor + 1
            target["recovery_v3"] = recovery_v3
            target["status"] = "invalid"
            event = {
                "model_number": mn, "slot_id": slot_id, "question_id": qid,
                "attempt": attempt, "repair_action": "no_progress_guard",
                "repair_strategy": repair_strategy,
                "status": "strategy_skipped", "new_failure_code": "identical_recovery_input",
                "new_failure_reason": "Saved strategy input was already attempted unchanged.",
            }
            target.setdefault("history", []).append(deepcopy(event))
            history.append(event)
            warnings.append(f"repair: Recovery V3 advanced past identical input for {slot_id}")
            continue
        memory["last_content_repair_input_fingerprint"] = unresolved_fingerprint
        memory["attempted_repair_input_fingerprints"].append(unresolved_fingerprint)
        memory.setdefault("strategy_history", []).append({
            "attempt": attempt,
            "strategy": repair_strategy,
            "repair_action": repair_action,
            "input_fingerprint": unresolved_fingerprint,
            "previous_failure": target.get("failure_reason"),
        })
        selected_id = ""
        if use_replacement:
            from app.online.duplicate_pipeline import (
                _candidate_pool, _complete_question_context, _select_replacement_candidate,
            )
            excluded = {str(v) for v in [
                *(memory.get("blacklisted_candidate_ids") or []),
                *sorted(incompatible_task_ids),
            ] if v}
            current_assignment = plan_by_slot.get(slot_id) or {}
            current_task_id = str(
                current_assignment.get("base_catalog_task_id")
                or current_assignment.get("task_id") or ""
            )
            if current_task_id:
                excluded.add(current_task_id)
                if current_task_id not in memory["blacklisted_candidate_ids"]:
                    memory["blacklisted_candidate_ids"].append(current_task_id)
            used_answers = {
                normalize_fitb_answer_key(answer)
                for _kind, _container, item in _iter_questions(staged_exam.get("questions") or {})
                for answer in (item.get("answers") or []) if answer
            }
            active_slot_ids = {
                str(item.get("slot_id"))
                for exam_context in global_generated
                for _k, _c, item in _iter_questions(
                    exam_context.get("questions") or {}
                )
                if item.get("slot_id")
            } or {
                str(item.get("slot_id")) for _k, _c, item in
                _iter_questions(staged_exam.get("questions") or {}) if item.get("slot_id")
            }
            # Replace this model's stale global plan partition with its latest
            # staged partition, while retaining every other model for global
            # unused/least-used task ranking.
            staged_model_numbers = {
                int(value.get("model_number") or mn) for value in staged_plans
            }
            selection_plans = [
                deepcopy(value) for value in global_plans
                if int(value.get("model_number") or 0) not in staged_model_numbers
            ] + deepcopy(staged_plans)
            coordinator = state.get("_recovery_task_coordinator") or {}
            coordinator_lock = coordinator.get("lock")
            if coordinator_lock is not None:
                with coordinator_lock:
                    reserved_elsewhere = {
                        str(reserved_slot): str(task_id) for reserved_slot, task_id
                        in (coordinator.get("reservations") or {}).items()
                        if str(reserved_slot) != slot_id and task_id
                    }
                # Reservations participate in the same existing usage ranking:
                # they stop another worker from seeing the task as unused but
                # do not forbid authorized least-used reuse after exhaustion.
                if reserved_elsewhere:
                    selection_plans.append({
                        "model_number": 0,
                        "items": {"reserved": [
                            {"slot_id": reserved_slot, "task_id": task_id,
                             "question_type": qtype, "recovery_reservation": True}
                            for reserved_slot, task_id in reserved_elsewhere.items()
                        ]},
                    })
                    active_slot_ids.update(reserved_elsewhere)
            candidate_pool = _candidate_pool(
                state.get("assessment_catalog") or {}, selection_plans, slot_id, qtype,
                str(state.get("difficulty") or "mix"), active_slot_ids, excluded,
                prefer_task_id=None,
                exclude_variation_ids={
                    str((plan_by_slot.get(slot_id) or {}).get("selected_variation_id") or ""),
                    *map(str, memory.get("rejected_variation_ids") or []),
                },
            )
            for candidate in candidate_pool:
                # The sequence distinguishes bounded numerical re-instantiation
                # from an exact retry of a previously rejected instance.
                candidate["replacement_instance_sequence"] = int(
                    completion_attempts.get(slot_id) or 0
                )
            active_plan_items = [
                item for plan in selection_plans
                for values in (plan.get("items") or {}).values()
                for item in values or []
            ]
            existing_question_context = []
            global_plan_by_slot = {
                str(item.get("slot_id") or ""): item
                for item in active_plan_items if item.get("slot_id")
            }
            global_question_by_id: dict[str, dict[str, Any]] = {}
            staged_exam_models = {int(value.get("model_number") or mn) for value in generated}
            context_exams = [
                deepcopy(value) for value in global_generated
                if int(value.get("model_number") or 0) not in staged_exam_models
            ] + deepcopy(generated)
            for existing_exam in context_exams:
                existing_model = int(existing_exam.get("model_number") or 0)
                for existing_qtype, _container, existing_item in _iter_questions(
                    existing_exam.get("questions") or {}
                ):
                    existing_slot = str(existing_item.get("slot_id") or "")
                    existing_id = str(existing_item.get("question_id") or "")
                    existing_plan = global_plan_by_slot.get(existing_slot) or {
                        "slot_id": existing_slot, "question_type": existing_qtype,
                    }
                    context_item = _complete_question_context(
                        existing_id, existing_item, existing_plan,
                        role=(
                            "active_exam_question" if existing_model == mn
                            else "cross_model_reference_question"
                        ),
                    )
                    context_item["model_number"] = existing_model
                    existing_question_context.append(context_item)
                    if existing_id:
                        if existing_model == mn:
                            global_question_by_id[existing_id] = context_item
                        global_question_by_id[
                            f"m{existing_model}::{existing_id}"
                        ] = context_item
            failed_entry = _build_by_id(staged_exam.get("questions") or {}).get(qid)
            failed_item = (failed_entry or (None, None, {}))[2]
            replacement = deepcopy(candidate_pool[0]) if candidate_pool else None
            selector_context = {
                "selection_source": "shared_three_criterion_rank",
                "eligible_candidate_count": len(candidate_pool),
                "candidates": deepcopy(candidate_pool),
            }
            if replacement is None:
                explicit_no_candidate = (
                    selector_context.get("selection_source") == "agent_no_candidate"
                )
                rejected_rows = list(
                    (selector_context.get("agent_choice") or {}).get(
                        "rejected_candidates", []
                    )
                )
                rejected_ids = {
                    str(row.get("candidate_id") or "")
                    for row in rejected_rows if row.get("candidate_id")
                }
                if explicit_no_candidate:
                    # The Judge evaluated this whole bounded window. Persist it
                    # before retrying so the next attempt must inspect genuinely
                    # different candidates rather than repeat unchanged input.
                    rejected_ids.update(
                        str(row.get("task_id") or row.get("fitb_entry_id") or "")
                        for row in selector_context.get("candidates") or []
                    )
                    rejected_ids.discard("")
                    for rejected_id in sorted(rejected_ids):
                        if rejected_id not in memory["blacklisted_candidate_ids"]:
                            memory["blacklisted_candidate_ids"].append(rejected_id)
                        memory["candidate_failures"].append({
                            "candidate_id": rejected_id,
                            "reason": "replacement_selector_explicitly_rejected",
                            "attempt": attempt,
                            "stage": "replacement_selector",
                        })
                reason = "No unused or reusable grounded replacement remained for this slot."
                if not completion_enabled:
                    legacy_attempts[slot_id] = attempt
                    failures[slot_id] = min(
                        MAX_QUESTION_REPAIR_ATTEMPTS,
                        int(failures.get(slot_id) or 0) + 1,
                    )
                exhausted_completion = (
                    int(completion_attempts.get(slot_id) or 0)
                    >= MAX_COMPLETION_ATTEMPTS_PER_SLOT
                )
                remaining_ranked_candidates = max(
                    0,
                    int(selector_context.get("eligible_candidate_count") or 0)
                    - len(rejected_ids),
                )
                semantic_inventory_exhausted = (
                    explicit_no_candidate and remaining_ranked_candidates == 0
                )
                target.update(
                    status=(
                        "terminal_failure"
                        if exhausted_completion or semantic_inventory_exhausted
                        else "completion_pending"
                    ),
                    failure_code=(
                        "no_valid_source_grounded_replacement"
                        if exhausted_completion or semantic_inventory_exhausted
                        else "completion_candidate_unavailable"
                    ),
                    failure_reason=reason,
                )
                history.append({"model_number": mn, "slot_id": slot_id, "question_id": qid,
                                "attempt": attempt, "repair_action": repair_action,
                                "status": "candidate_generation_failed", "new_failure_reason": reason})
                continue
            proposed_task_id = str(
                replacement.get("fitb_entry_id") or replacement.get("task_id") or ""
            )
            if coordinator_lock is not None and proposed_task_id:
                with coordinator_lock:
                    reservations = coordinator.setdefault("reservations", {})
                    selected_candidate = None
                    conflicts_seen: list[tuple[str, str]] = []
                    for ranked_candidate in candidate_pool:
                        candidate_task_id = str(
                            ranked_candidate.get("fitb_entry_id")
                            or ranked_candidate.get("task_id") or ""
                        )
                        conflicting_slot = next((
                            str(reserved_slot) for reserved_slot, task_id in reservations.items()
                            if str(reserved_slot) != slot_id
                            and str(task_id) == candidate_task_id
                        ), None)
                        if conflicting_slot is not None:
                            conflicts_seen.append((candidate_task_id, conflicting_slot))
                            continue
                        selected_candidate = ranked_candidate
                        reservations[slot_id] = candidate_task_id
                        break
                for unavailable_id, conflicting_slot in conflicts_seen:
                    memory["candidate_failures"].append({
                        "candidate_id": unavailable_id,
                        "reason": "concurrent_global_task_reservation_conflict",
                        "conflicting_slot_id": conflicting_slot,
                        "attempt": attempt,
                    })
                if selected_candidate is None:
                    target["status"] = "completion_pending"
                    target["failure_code"] = "global_task_reservation_conflict"
                    target["failure_reason"] = (
                        "Every ranked candidate was concurrently unavailable; retry remains bounded."
                    )
                    continue
                replacement = deepcopy(selected_candidate)
                proposed_task_id = str(
                    replacement.get("fitb_entry_id") or replacement.get("task_id") or ""
                )
            for rejected in (selector_context.get("agent_choice") or {}).get(
                "rejected_candidates", []
            ):
                rejected_id = str(rejected.get("candidate_id") or "")
                if not rejected_id or rejected_id == str(
                    replacement.get("fitb_entry_id") or replacement.get("task_id") or ""
                ):
                    continue
                memory["blacklisted_candidate_ids"].append(rejected_id)
                memory["candidate_failures"].append({
                    "candidate_id": rejected_id,
                    "reason": str(rejected.get("reason") or "replacement_selector_rejected"),
                    "attempt": attempt,
                    "stage": "replacement_selector",
                })
            reused_identity = str(replacement.get("fitb_entry_id") or replacement.get("task_id") or "")
            previous_uses = [
                value for value in existing_question_context
                if str(value.get("fitb_entry_id") or value.get("task_id") or "") == reused_identity
            ]
            if previous_uses:
                replacement["previous_task_questions"] = previous_uses
                replacement["previous_question_ids"] = [
                    str(value.get("question_id")) for value in previous_uses if value.get("question_id")
                ]
                replacement["task_usage_count"] = len(previous_uses)
                replacement["reuse_mode"] = "intentional_practice_variation"
            from app.online.duplicate_pipeline import _replace_plan_item
            selected_id = str(replacement.get("fitb_entry_id") or replacement.get("task_id") or "")
            memory["replacement_attempt_count"] = int(memory.get("replacement_attempt_count") or 0) + 1
            memory["attempted_candidate_ids"].append(selected_id)
            replacement["assignment_revision"] = int(
                (plan_by_slot.get(slot_id) or {}).get("assignment_revision") or 0
            ) + 1
            _normalize_replacement_difficulty_state(
                replacement, str(state.get("difficulty") or "mix"),
            )
            if not _replace_plan_item(staged_plans, mn, slot_id, replacement):
                memory["blacklisted_candidate_ids"].append(selected_id)
                target.update(failure_code="replacement_plan_write_failed",
                              failure_reason="Replacement could not be applied to the plan slot.")
                continue
            applied_item = next((
                item for plan in staged_plans
                if int(plan.get("model_number") or 0) == mn
                for values in (plan.get("items") or {}).values()
                for item in values or []
                if str(item.get("slot_id") or "") == slot_id
            ), None)
            if applied_item is not None:
                for key in ("assignment_revision", "backup_activation", "backups"):
                    if key in replacement:
                        applied_item[key] = deepcopy(replacement[key])
            applied_replacement = _apply_replacement_difficulty_design(
                staged_plans, model_number=mn, qtype=qtype, slot_id=slot_id,
                state=state, requested=str(state.get("difficulty") or "mix"),
            )
            if _replacement_difficulty_unresolved(applied_replacement):
                _mark_replacement_difficulty_blocked(target, applied_replacement)
                history.append({
                    "model_number": mn, "slot_id": slot_id, "question_id": qid,
                    "attempt": attempt, "repair_action": "replace_assignment",
                    "candidate_task_id": selected_id,
                    "status": "validation_blocked",
                    "new_failure_code": target.get("failure_code"),
                    "new_failure_reason": target.get("failure_reason"),
                })
                continue
            catalog = state.get("assessment_catalog") or {}
            if catalog.get("catalog_version") == ASSESSMENT_CATALOG_VERSION:
                from app.online.assessment_catalog import validate_plan_catalog_bindings
                valid_chunk_ids = {
                    str(chunk.get("child_id") or "")
                    for chunk in state.get("retrieved_chunks") or []
                    if chunk.get("child_id")
                }
                binding_errors = validate_plan_catalog_bindings(
                    catalog, staged_plans, valid_chunk_ids or None,
                )
                if binding_errors:
                    memory["blacklisted_candidate_ids"].append(selected_id)
                    memory.setdefault("candidate_failures", []).append({
                        "candidate_id": selected_id,
                        "reason": "; ".join(binding_errors),
                        "attempt": attempt,
                        "stage": "replacement_catalog_binding",
                    })
                    target.update(
                        status="invalid",
                        failure_code="replacement_catalog_binding_invalid",
                        failure_reason="; ".join(binding_errors),
                    )
                    continue
            staged_state = {**state, "plans": staged_plans}
            plan_by_slot, source_by_chunk = _plan_context_for_model(staged_state, mn)
            if slot_id in plan_by_slot:
                plan_by_slot[slot_id]["recovery_failure_code"] = target.get("failure_code")
                plan_by_slot[slot_id]["recovery_failure_reason"] = target.get("failure_reason")
                plan_by_slot[slot_id]["recovery_attempt"] = attempt
            evidence_error = _plan_evidence_consistency_error(
                plan_by_slot.get(slot_id), source_by_chunk,
            )
            if evidence_error:
                memory["blacklisted_candidate_ids"].append(selected_id)
                memory.setdefault("candidate_failures", []).append({
                    "candidate_id": selected_id,
                    "reason": evidence_error,
                    "attempt": attempt,
                    "stage": "replacement_evidence_preflight",
                })
                target.update(
                    status="invalid",
                    failure_code="replacement_evidence_mismatch",
                    failure_reason=evidence_error,
                )
                history.append({
                    "model_number": mn, "slot_id": slot_id, "question_id": qid,
                    "attempt": attempt, "repair_action": "replace_assignment",
                    "candidate_task_id": selected_id,
                    "status": "candidate_generation_failed",
                    "new_failure_code": "replacement_evidence_mismatch",
                    "new_failure_reason": evidence_error,
                })
                continue
            replacement_plan = plan_by_slot.get(slot_id) or {}
            if (
                replacement_plan.get("difficulty_status") == "adaptation_pending"
                or (replacement_plan.get("difficulty_adaptation") or {}).get("status")
                == "pending"
            ):
                from app.online.difficulty_adjustment import adjust_failed_plan_slot
                findings = deepcopy(
                    (target.get("verdict") or {}).get("required_fixes")
                    or target.get("required_fixes")
                    or []
                )
                adjustment_attempt = int(
                    memory.get("replacement_difficulty_adjustment_attempts") or 0
                ) + 1
                memory["replacement_difficulty_adjustment_attempts"] = adjustment_attempt
                adjustment_ok, adjustment_audit = adjust_failed_plan_slot(
                    state={**staged_state, "plans": staged_plans},
                    plans=staged_plans,
                    model_number=mn,
                    slot_id=slot_id,
                    question_type=qtype,
                    findings=findings,
                    attempt=adjustment_attempt,
                )
                target.setdefault("difficulty_recovery_history", []).append(
                    deepcopy(adjustment_audit)
                )
                if not adjustment_ok:
                    fallback_plan = next((
                        item for plan in staged_plans
                        if int(plan.get("model_number") or 0) == mn
                        for values in (plan.get("items") or {}).values()
                        for item in values or []
                        if str(item.get("slot_id") or "") == slot_id
                    ), None)
                    failure = str(
                        adjustment_audit.get("failure")
                        or "The source-supported assignment cannot be adapted further."
                    )
                    _finalize_replacement_difficulty_fallback(
                        fallback_plan,
                        requested=str(state.get("difficulty") or ""),
                        adjustment_audit=adjustment_audit,
                    )
                    memory.setdefault("candidate_failures", []).append({
                        "candidate_id": selected_id,
                        "reason": failure,
                        "attempt": attempt,
                        "stage": "replacement_difficulty_fallback",
                    })
                    staged_state = {**state, "plans": staged_plans}
                    plan_by_slot, source_by_chunk = _plan_context_for_model(
                        staged_state, mn
                    )
                staged_state = {**state, "plans": staged_plans}
                plan_by_slot, source_by_chunk = _plan_context_for_model(staged_state, mn)
                finalized = plan_by_slot.get(slot_id) or {}
                if (
                    finalized.get("difficulty_status") == "adaptation_pending"
                    or (finalized.get("difficulty_adaptation") or {}).get("status")
                    == "pending"
                ):
                    target.update(
                        status="completion_pending",
                        failure_code="replacement_difficulty_unresolved",
                        failure_reason="Replacement remained adaptation_pending after preflight.",
                    )
                    continue
            repair_action = "replace_fitb_entry" if qtype == "fill_in_the_blank" else "replace_task"

        if _build_by_id(staged_exam.get("questions") or {}).get(qid) is None:
            _put_question(staged_exam, qtype, _placeholder_question(qtype, slot_id, qid))
        generated_ok = False
        outcome: Dict[str, Any] = {}
        outcome_warnings: list[str] = []
        selected_repair_role: str | None = None
        if repair_action == "accept_difficulty":
            # Luna's observation is authoritative; difficulty mismatch is terminal.
            verdict = target.get("verdict") or {}
            observation = deepcopy(verdict.get("difficulty_observation") or {})
            active_plan = plan_by_slot.get(slot_id) or {}
            observed = str(
                observation.get("actual_difficulty")
                or active_plan.get("actual_difficulty")
                or "medium"
            )
            staged_entry = _build_by_id(staged_exam.get("questions") or {}).get(qid)
            if staged_entry is not None:
                question = staged_entry[2]
                observation["question_fingerprint"] = hashlib.sha256(json.dumps(
                    question, ensure_ascii=False, sort_keys=True, default=str,
                ).encode("utf-8")).hexdigest()
                question["difficulty"] = observed
                question["actual_difficulty"] = observed
            observation["actual_difficulty"] = observed
            active_plan.update({
                "requested_difficulty": state.get("difficulty"),
                "actual_difficulty": observed,
                "difficulty": observed,
                "difficulty_status": "accepted_source_constraint",
                "difficulty_observation": deepcopy(observation),
                "difficulty_design": {
                    **dict(active_plan.get("difficulty_design") or {}),
                    "status": "accepted_source_constraint",
                    "observed": observed,
                    "reason": observation.get("reason") or "",
                },
            })
            target.update(
                status="resolved", failure_code="pass", failure_reason="",
            )
            generated_ok = staged_entry is not None
            outcome_warnings = []
            selected_repair_role = "accepted_source_constraint"
        elif repair_action == "adjust_difficulty":
            from app.online.difficulty_adjustment import adjust_failed_plan_slot
            difficulty_attempt = int(memory.get("difficulty_adjustment_attempts") or 0) + 1
            memory["difficulty_adjustment_attempts"] = difficulty_attempt
            findings = deepcopy(
                (target.get("verdict") or {}).get("required_fixes")
                or target.get("required_fixes")
                or []
            )
            adjusted, adjustment_audit = adjust_failed_plan_slot(
                state=state,
                plans=staged_plans,
                model_number=mn,
                slot_id=slot_id,
                question_type=qtype,
                findings=findings,
                attempt=difficulty_attempt,
            )
            target.setdefault("difficulty_recovery_history", []).append(
                deepcopy(adjustment_audit)
            )
            if adjusted:
                staged_state = {**state, "plans": staged_plans}
                plan_by_slot, source_by_chunk = _plan_context_for_model(staged_state, mn)
                if slot_id in plan_by_slot:
                    plan_by_slot[slot_id]["recovery_failure_code"] = target.get("failure_code")
                    plan_by_slot[slot_id]["recovery_failure_reason"] = target.get("failure_reason")
                    plan_by_slot[slot_id]["recovery_attempt"] = attempt
                generated_ok, outcome_warnings = _runtime._regenerate_plan_slot(
                    staged_state, staged_exam, mn, qid, plan_by_slot, source_by_chunk,
                )
            else:
                failure_reason = str(
                    adjustment_audit.get("failure") or "unsupported adaptation"
                )
                active_adaptation = dict(
                    (plan_by_slot.get(slot_id) or {}).get("difficulty_adaptation") or {}
                )
                adaptation_terminal = (
                    active_adaptation.get("further_adjustment_possible") is False
                    or active_adaptation.get("status") in {"completed", "failed"}
                )
                if difficulty_attempt >= 2 or adaptation_terminal:
                    # The source contract won: retain the honest observed
                    # level instead of repeatedly pretending the task is easy.
                    findings = list(findings or [])
                    observed = next((
                        str(row.get("observed_difficulty") or "")
                        for row in findings if row.get("observed_difficulty")
                    ), "") or str(
                        (plan_by_slot.get(slot_id) or {}).get("actual_difficulty")
                        or "medium"
                    )
                    active_plan = plan_by_slot.get(slot_id) or {}
                    active_plan.update({
                        "requested_difficulty": state.get("difficulty"),
                        "actual_difficulty": observed,
                        "difficulty": observed,
                        "difficulty_status": "accepted_source_constraint",
                        "difficulty_fallback_reason": failure_reason,
                        "difficulty_adaptation": {
                            **dict(active_plan.get("difficulty_adaptation") or {}),
                            "status": "accepted_source_constraint",
                            "reason": failure_reason,
                        },
                    })
                    staged_entry = _build_by_id(
                        staged_exam.get("questions") or {}
                    ).get(qid)
                    if staged_entry is not None:
                        staged_entry[2]["difficulty"] = observed
                        staged_entry[2]["actual_difficulty"] = observed
                    generated_ok = staged_entry is not None
                    outcome_warnings = []
                    repair_strategy = "accepted_source_constraint"
                else:
                    generated_ok = False
                    outcome_warnings = [
                        "Targeted Difficulty Adjustment did not produce an accepted assignment: "
                        + failure_reason
                    ]
        elif repair_action in {"regenerate_missing_slot", "replace_fitb_entry", "replace_task", "switch_variation"}:
            generated_ok, outcome_warnings = _runtime._regenerate_plan_slot(
                staged_state, staged_exam, mn, qid, plan_by_slot, source_by_chunk)
        else:
            verdict = deepcopy(target.get("verdict") or {})
            verdict.setdefault("question_id", qid)
            verdict.setdefault("action", "FIX_QUESTION")
            resolved_issue_ids = {
                str(issue_id)
                for revision in (working_candidates.get(slot_id) or {}).get("revisions") or []
                if revision.get("status") == "accepted"
                for issue_id in revision.get("resolved_issue_ids") or []
                if issue_id
            }
            if resolved_issue_ids:
                remaining = [
                    issue for issue in verdict.get("required_fixes") or []
                    if str(issue.get("issue_id") or "") not in resolved_issue_ids
                ]
                verdict["required_fixes"] = remaining
                verdict["fields_to_fix"] = sorted({
                    str(field) for issue in remaining
                    for field in issue.get("fields_to_fix") or []
                })
            if not resolved_issue_ids:
                verdict["fields_to_fix"] = list(
                    target.get("fields_to_fix") or verdict.get("fields_to_fix") or []
                )
            verdict["repair_attempt_history"] = [
                {key: item.get(key) for key in (
                    "attempt", "repair_action", "status", "new_failure_code",
                    "new_failure_reason", "progress",
                ) if item.get(key) is not None}
                for item in (target.get("history") or [])[-3:]
            ]
            repeated_defect = memory.get("repeated_defect")
            if repeated_defect:
                verdict["repeated_defect"] = deepcopy(repeated_defect)
            prior_rejection = next((
                str(item.get("new_failure_reason") or item.get("failure_reason") or "")
                for item in reversed(target.get("history") or [])
                if item.get("new_failure_reason") or item.get("failure_reason")
            ), "")
            if prior_rejection:
                verdict["previous_repair_rejection"] = prior_rejection
            stored_repair_domain = str(
                (state.get("assessment_profile") or {}).get("domain") or ""
            ).upper()
            if stored_repair_domain:
                from app.online.multi_repair import repair_question_specialized
                outcome = repair_question_specialized(
                    exam=staged_exam,
                    verdict=verdict,
                    qtype=qtype,
                    plan_item=plan_by_slot.get(slot_id) or {},
                    source_chunks=_assigned_source_chunks(
                        plan_by_slot.get(slot_id), source_by_chunk
                    ),
                    domain=stored_repair_domain,
                    previous_rejection=prior_rejection,
                    previous_attempts=verdict.get("repair_attempt_history") or [],
                    recovery_strategy=repair_strategy,
                    content_repair_calls_used=int(repair_calls.get(slot_id) or 0),
                    failed_repair_attempts_used=int(failures.get(slot_id) or 0),
                    repeated_defect=verdict.get("repeated_defect"),
                )
                repair_calls[slot_id] = int(
                    outcome.get("content_repair_calls", repair_calls.get(slot_id) or 0)
                )
                failures[slot_id] = int(
                    outcome.get("failed_repair_attempts", failures.get(slot_id) or 0)
                )
                if not failures[slot_id]:
                    failures.pop(slot_id, None)
                if not repair_calls[slot_id]:
                    repair_calls.pop(slot_id, None)
                revisions = list(outcome.get("candidate_revisions") or [])
                if revisions:
                    previous_revisions = list(
                        (working_candidates.get(slot_id) or {}).get("revisions") or []
                    )
                    base_revision = len(previous_revisions)
                    for offset, revision in enumerate(revisions, start=1):
                        revision["revision"] = base_revision + offset
                    combined_revisions = [*previous_revisions, *revisions]
                    working_candidates[slot_id] = {
                        "slot_id": slot_id,
                        "question_id": qid,
                        "assignment_fingerprint": (
                            ((state.get("slot_lifecycle") or {}).get(slot_id) or {}).get(
                                "assignment_fingerprint"
                            )
                        ),
                        "latest_revision": combined_revisions[-1].get("revision"),
                        "latest_fingerprint": revisions[-1].get("fingerprint"),
                        "candidate": deepcopy(
                            next((
                                row.get("candidate") for row in reversed(revisions)
                                if row.get("status") == "accepted" and row.get("candidate")
                            ), None)
                        ),
                        "revisions": deepcopy(combined_revisions),
                    }
                if "content_repair_calls" not in outcome:
                    # Compatibility for legacy/test repair adapters.
                    repair_calls[slot_id] = int(repair_calls.get(slot_id) or 0) + 1
                selected_repair_role = " -> ".join(
                    outcome.get("repair_route") or []
                )
            else:
                # Compatibility for persisted pre-domain jobs and narrow test doubles.
                repair_role = _repair_role_for(qtype, verdict, domain="STEM")
                selected_repair_role = repair_role
                if repair_role == "stem_quantitative":
                    from app.llm.factory import create_review_client
                    client = create_review_client(
                        domain="STEM", stage="stem_quantitative_repairer"
                    )
                    outcome = _runtime.repair_exam(
                        staged_exam, [verdict], client=client,
                        plan_by_slot=plan_by_slot, source_by_chunk=source_by_chunk,
                        repair_role="stem_quantitative",
                    )
                else:
                    outcome = (
                        _runtime.repair_exam(
                            staged_exam, [verdict], plan_by_slot=plan_by_slot,
                            source_by_chunk=source_by_chunk,
                        )
                        if plan_by_slot or source_by_chunk
                        else _runtime.repair_exam(staged_exam, [verdict])
                    )
                # Legacy repairers expose one logical content proposal.
                repair_calls[slot_id] = int(repair_calls.get(slot_id) or 0) + 1
            target["repair_role"] = selected_repair_role
            target["repair_route"] = list(outcome.get("repair_route") or [])
            target["repair_traces"] = list(outcome.get("repair_traces") or [])
            generated_ok = qid in set(outcome.get("repaired_ids") or [])
            outcome_warnings = list(outcome.get("warnings") or [])
            repair_call_logs.extend(outcome.get("repair_traces") or [])
        warnings.extend(outcome_warnings)
        if not outcome.get("operational_error"):
            legacy_attempts[slot_id] = attempt
        if not generated_ok:
            reason = "; ".join(outcome_warnings) or "Repairer did not produce a candidate."
            if outcome.get("operational_error"):
                memory.pop("last_content_repair_input_fingerprint", None)
                if unresolved_fingerprint in memory.get(
                    "attempted_repair_input_fingerprints", []
                ):
                    memory["attempted_repair_input_fingerprints"].remove(
                        unresolved_fingerprint
                    )
                if memory.get("strategy_history") and (
                    memory["strategy_history"][-1].get("input_fingerprint")
                    == unresolved_fingerprint
                ):
                    memory["strategy_history"].pop()
                operational_failures = int(
                    memory.get("operational_retry_failures") or 0
                ) + 1
                memory["operational_retry_failures"] = operational_failures
                exhausted = operational_failures >= MAX_CANDIDATE_GENERATION_FAILURES
                target.update(
                    status=("validation_blocked" if exhausted else "operational_retry_pending"),
                    failure_code=(
                        "operational_retry_exhausted"
                        if exhausted else "operational_retry_pending"
                    ),
                    failure_reason=reason,
                )
                event = {
                    "model_number": mn, "slot_id": slot_id,
                    "question_id": qid, "attempt": attempt,
                    "repair_action": repair_action,
                    "repair_role": target.get("repair_role"),
                    "status": target["status"], "operational": True,
                    "new_failure_reason": reason,
                }
                target.setdefault("history", []).append(deepcopy(event))
                history.append(event)
                continue
            if outcome.get("route_status") == "REPAIR_CONFLICT":
                failures[slot_id] = int(failures.get(slot_id) or 0) + 1
                exhausted = failures[slot_id] >= attempt_limit
                target.update(
                    status="terminal_failure" if exhausted else "invalid",
                    failure_code=(
                        "content_repair_budget_exhausted"
                        if exhausted else "repair_conflict"
                    ),
                    failure_reason=reason,
                    repair_conflict=deepcopy(
                        (outcome.get("routing_details") or {}).get("conflicts") or []
                    ),
                )
                event = {
                    "model_number": mn, "slot_id": slot_id, "question_id": qid,
                    "attempt": attempt, "repair_action": repair_action,
                    "repair_strategy": repair_strategy,
                    "repair_role": target.get("repair_role"),
                    "status": "repair_conflict",
                    "new_failure_code": "repair_conflict",
                    "new_failure_reason": reason,
                    "question_changed": False,
                    "reconciled_conflicts": deepcopy(outcome.get("reconciled_conflicts") or []),
                }
                target.setdefault("history", []).append(deepcopy(event))
                history.append(event)
                memory["strategy_history"][-1].update(
                    outcome="REPAIR_CONFLICT", failure_reason=reason,
                )
                recovery_v3.setdefault("attempted_strategies", []).append(selected_strategy)
                recovery_v3["next_index"] = cursor + 1
                target["recovery_v3"] = recovery_v3
                continue
            if outcome.get("route_status") in {"VALIDATION_BLOCKED", "ROUTING_BLOCKED"}:
                no_strategy = outcome.get("route_status") == "ROUTING_BLOCKED"
                if no_strategy:
                    recovery_v3.setdefault("attempted_strategies", []).append(selected_strategy)
                    recovery_v3["next_index"] = cursor + 1
                    target.update(
                        status="invalid",
                        failure_code="repair_router_blocked",
                        failure_reason=reason,
                        recovery_v3=recovery_v3,
                    )
                else:
                    target.update(
                        status="operational_retry_pending",
                        failure_code="repair_provider_blocked",
                        failure_reason=reason,
                    )
                    memory["blocked_route_fingerprint"] = unresolved_fingerprint
                    memory["blocked_route_status"] = str(outcome.get("route_status") or "")
                event = {
                    "model_number": mn, "slot_id": slot_id, "question_id": qid,
                    "attempt": attempt, "repair_action": repair_action,
                    "repair_strategy": repair_strategy,
                    "status": target["status"],
                    "new_failure_code": target["failure_code"],
                    "new_failure_reason": reason,
                }
                target.setdefault("history", []).append(deepcopy(event))
                history.append(event)
                continue
            if repair_action == "repair_question" and not outcome.get("operational_error"):
                # Invalid/no-op/unauthorized proposals are failed content
                # strategies. Specialized repair already reports these.
                if not stored_repair_domain:
                    failures[slot_id] = int(failures.get(slot_id) or 0) + 1
                elif "failed_repair_attempts" not in outcome:
                    failures[slot_id] = int(failures.get(slot_id) or 0) + 1
            elif not completion_enabled and not outcome.get("operational_error"):
                # Backward-compatible semantics for checkpoints created before
                # the separate completion budget existed.
                failures[slot_id] = int(failures.get(slot_id) or 0) + 1
            is_operational = bool(outcome.get("operational_error"))
            operational_failures = int(memory.get("candidate_generation_failures") or 0)
            if (
                not is_operational
                and repair_action != "repair_question"
                and not generated_ok
            ):
                operational_failures += 1
                memory["candidate_generation_failures"] = operational_failures
                memory["force_alternative_strategy"] = True
                # A failed generated/replacement candidate demonstrates an
                # unsuitable assignment. A no-op local content repair does
                # not: it must retain the established bounded repair budget.
                memory["force_backup"] = True
            elif (
                not is_operational
                and str(target.get("failure_code") or "")
                in FUNDAMENTAL_ASSIGNMENT_FAILURES
            ):
                memory["force_backup"] = True
            if selected_id and not is_operational:
                candidate_variation_id = str(
                    (plan_by_slot.get(slot_id) or {}).get("selected_variation_id") or ""
                )
                candidate_failure_code = str(target.get("failure_code") or "")
                if (
                    not generated_ok
                    or candidate_failure_code in {"mcq_task_unsuitable", "unusable_task"}
                    or (qtype == "fill_in_the_blank" and candidate_failure_code in {
                        "duplicate_question", "excessive_overlap", "insufficient_evidence",
                    })
                ):
                    if selected_id not in memory["blacklisted_candidate_ids"]:
                        memory["blacklisted_candidate_ids"].append(selected_id)
                elif candidate_variation_id:
                    rejected = memory.setdefault("rejected_variation_ids", [])
                    if candidate_variation_id not in rejected:
                        rejected.append(candidate_variation_id)
                memory["candidate_failures"].append({"candidate_id": selected_id,
                                                      "reason": reason, "attempt": attempt})
            target.update(
                status=(
                    "terminal_failure"
                    if (
                        int(failures.get(slot_id) or 0) >= attempt_limit
                        and (
                            not completion_enabled
                            or int(completion_attempts.get(slot_id) or 0)
                            >= MAX_COMPLETION_ATTEMPTS_PER_SLOT
                        )
                    )
                    or operational_failures >= MAX_CANDIDATE_GENERATION_FAILURES
                    else "invalid"
                ),
                failure_code=(
                    "candidate_generation_budget_exhausted"
                    if (
                        int(failures.get(slot_id) or 0) >= attempt_limit
                        and (
                            not completion_enabled
                            or int(completion_attempts.get(slot_id) or 0)
                            >= MAX_COMPLETION_ATTEMPTS_PER_SLOT
                        )
                    )
                    or operational_failures >= MAX_CANDIDATE_GENERATION_FAILURES
                    else "candidate_generation_failed"
                ),
                failure_reason=reason,
            )
            if not is_operational and selected_strategy != "SWITCH_TASK":
                recovery_v3.setdefault("attempted_strategies", []).append(selected_strategy)
                recovery_v3["next_index"] = cursor + 1
                recovery_v3["last_failure_code"] = target.get("failure_code")
                recovery_v3["last_failure_reason"] = reason
                target["recovery_v3"] = recovery_v3
            target.setdefault("history", []).append({"attempt": attempt,
                "repair_action": repair_action, "repair_role": target.get("repair_role"),
                "repair_strategy": repair_strategy,
                "requested_fields": list(target.get("fields_to_fix") or []),
                "status": "candidate_generation_failed",
                "new_failure_reason": reason})
            history.append({"model_number": mn, "slot_id": slot_id, "question_id": qid,
                            "attempt": attempt, "repair_action": repair_action,
                            "repair_strategy": repair_strategy,
                            "status": "candidate_generation_failed", "new_failure_reason": reason})
            continue
        candidate_plan = plan_by_slot.get(slot_id) or {}
        assignment_payload = {
            "task_id": candidate_plan.get("task_id"),
            "fitb_entry_id": candidate_plan.get("fitb_entry_id"),
            "variation_id": candidate_plan.get("selected_variation_id"),
            "effective_instance_id": candidate_plan.get("effective_instance_id"),
            "evidence_ids": _plan_source_ids(candidate_plan),
        }
        assignment_fingerprint = hashlib.sha256(json.dumps(
            assignment_payload, ensure_ascii=False, sort_keys=True, default=str,
        ).encode("utf-8")).hexdigest()
        assignment_attempts = memory.setdefault("assignment_validated_attempts", {})
        if int(assignment_attempts.get(assignment_fingerprint) or 0) >= MAX_ASSIGNMENT_REPAIR_ATTEMPTS:
            target.update(status=(
                              "terminal_failure"
                              if (
                                  not completion_enabled
                                  or int(completion_attempts.get(slot_id) or 0)
                                  >= MAX_COMPLETION_ATTEMPTS_PER_SLOT
                              ) else "invalid"
                          ),
                          failure_code="assignment_repair_budget_exhausted",
                          failure_reason="Assignment exhausted its validated-candidate budget.",
                          repair_type="replace_assignment")
            continue
        staged_entry = _build_by_id(staged_exam.get("questions") or {}).get(qid)
        staged_question = deepcopy((staged_entry or (None, None, {}))[2])
        candidate_fingerprint = hashlib.sha256(json.dumps(
            staged_question, ensure_ascii=False, sort_keys=True, default=str,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        rejected_fingerprints = set(memory.setdefault("rejected_candidate_fingerprints", []))
        if candidate_fingerprint in rejected_fingerprints:
            target.update(
                status=(
                    "terminal_failure"
                    if (
                        content_exhausted
                        and (
                            not completion_enabled
                            or int(completion_attempts.get(slot_id) or 0)
                            >= MAX_COMPLETION_ATTEMPTS_PER_SLOT
                        )
                    ) else "invalid"
                ),
                failure_code="identical_candidate_rejected",
                failure_reason="Generation or repair reproduced a previously rejected candidate.",
            )
            memory["force_alternative_strategy"] = True
            event = {
                "model_number": mn, "slot_id": slot_id, "question_id": qid,
                "attempt": attempt, "repair_action": repair_action,
                "repair_strategy": repair_strategy,
                "status": "identical_candidate_rejected",
                "new_failure_code": "identical_candidate_rejected",
                "new_failure_reason": target["failure_reason"],
                "candidate_fingerprint": candidate_fingerprint,
                "question_changed": False,
            }
            target.setdefault("history", []).append(deepcopy(event))
            history.append(event)
            warnings.append(f"repair: prevented identical candidate loop for {slot_id}")
            continue
        assignment_attempts[assignment_fingerprint] = int(
            assignment_attempts.get(assignment_fingerprint) or 0
        ) + 1
        actual_changes = _canonical_changes(qtype, original_question, staged_question)
        fallback_metadata = {
            "repair_method": repair_action,
            "modified_fields": [change["field"] for change in actual_changes],
            "changes": actual_changes,
            "final_structural_diff": deepcopy(actual_changes),
            "targeted_failure_codes": [target.get("failure_code")],
            "targeted_issue_ids": [
                issue.get("issue_id") for issue in
                ((target.get("verdict") or {}).get("required_fixes") or [])
                if issue.get("issue_id")
            ],
            "change_explanation": deepcopy(outcome.get("change_explanation") or {
                "source": "existing_validator_and_planner_metadata",
                "original_question_about": _question_task_text(qtype, original_question),
                "original_answer_or_solution": _question_result_text(qtype, original_question),
                "original_problem": target.get("failure_reason") or "",
                "replacement_question_about": _question_task_text(qtype, staged_question),
                "replacement_answer_or_solution": _question_result_text(qtype, staged_question),
                "why_changed": target.get("what_must_change") or target.get("failure_reason") or "",
                "intended_task": {
                    "task_id": candidate_plan.get("task_id"),
                    "idea_to_test": candidate_plan.get("idea_to_test"),
                    "target": candidate_plan.get("target"),
                    "task_operation": candidate_plan.get("task_operation"),
                    "selected_variation_id": candidate_plan.get("selected_variation_id"),
                },
            }),
            "replacement_selector_rationale": deepcopy(outcome.get("replacement_selector_rationale")),
        }
        candidate_id = f"{qid}:{attempt}:{candidate_fingerprint[:16]}"
        previous_rejections = [
            {"attempt": value.get("attempt"), "repair_method": value.get("repair_action"),
             "changed_fields": value.get("changed_fields") or [],
             "candidate_fingerprint": value.get("candidate_fingerprint"),
             "failure_code": value.get("new_failure_code"),
             "rejection_reason": value.get("new_failure_reason")}
            for value in (target.get("history") or [])[-3:]
            if value.get("status") == "rejected" and not value.get("operational")
        ]
        # The switch is a staged transaction. Fence it against the active
        # assignment; lifecycle reconciliation advances the assignment revision
        # only after successful revalidation commits candidate["plans"].
        lifecycle_record = ((state.get("slot_lifecycle") or {}).get(slot_id) or {})
        expected_assignment_fingerprint = lifecycle_record.get("assignment_fingerprint")
        expected_assignment_revision = lifecycle_record.get("assignment_revision")
        # Count only candidates that actually changed Task ID. Another
        # variation of the same task is inventory inside the current task, not
        # a replacement attempt against the task budget.
        if use_replacement and str(candidate_plan.get("task_id") or "") != current_task_id:
            completion_attempts[slot_id] = int(
                completion_attempts.get(slot_id) or 0
            ) + 1
        candidates.append({"model_number": mn, "slot_id": slot_id, "question_id": qid,
            "candidate_id": candidate_id, "candidate_fingerprint": candidate_fingerprint,
            "question_type": qtype, "attempt": attempt, "repair_action": repair_action,
            "repair_strategy": repair_strategy,
            "recovery_v3_strategy": selected_strategy,
            "repair_role": selected_repair_role,
            "repair_route": list(outcome.get("repair_route") or []),
            "repair_traces": list(outcome.get("repair_traces") or []),
            "requested_fields": list((target.get("verdict") or {}).get("fields_to_fix") or []),
            "had_original": had_original, "previous_failure_code": target.get("failure_code"),
            "previous_failure_reason": target.get("failure_reason"),
            "candidate_task_id": candidate_plan.get("task_id"),
            "candidate_variation_id": candidate_plan.get("selected_variation_id"),
            "candidate_fitb_entry_id": candidate_plan.get("fitb_entry_id"),
            "assignment_fingerprint": assignment_fingerprint,
            "expected_assignment_revision": expected_assignment_revision,
            "expected_assignment_fingerprint": expected_assignment_fingerprint,
            "expected_question_revision": ((state.get("slot_lifecycle") or {}).get(slot_id) or {}).get("question_revision"),
            "expected_question_fingerprint": ((state.get("slot_lifecycle") or {}).get(slot_id) or {}).get("question_fingerprint"),
            "original_question": original_question,
            "original_failure_findings": deepcopy(
                (target.get("verdict") or {}).get("required_fixes") or [target.get("verdict") or {}]
            ),
            "repair_metadata": deepcopy((outcome.get("repair_metadata") or {}).get(qid) or fallback_metadata),
            "previous_rejections": previous_rejections,
            "unaffected_validation_roles": {} if switched_variation_id else {
                qid: deepcopy(
                    (((state.get("validation_ledger") or {}).get(str(mn)) or {}).get(qid) or {}).get("specialized_roles") or {}
                )
            },
            "assignment_switched": bool(switched_variation_id),
            "switched_variation_id": switched_variation_id or None,
            "exam": staged_exam, "plans": staged_plans})
        if mn in validated_models:
            validated_models.remove(mn)
        target["status"] = "candidate_generated"
        target.setdefault("history", []).append({"attempt": attempt, "repair_action": repair_action,
            "repair_role": selected_repair_role,
            "requested_fields": list((target.get("verdict") or {}).get("fields_to_fix") or []),
            "candidate_id": selected_id or candidate_plan.get("fitb_entry_id") or candidate_plan.get("task_id"),
            "status": "candidate_generated", "failure_reason_received": target.get("failure_reason")})

    return {"repair_manifest": manifest, "repair_candidates": candidates,
            "model_lifecycle_status": {
                int(exam.get("model_number") or 0): {
                    "status": "locally_ready" if int(exam.get("model_number") or 0) in set(validated_models)
                    else "recovery_pending",
                }
                for exam in state.get("generated_exams") or []
            },
            "question_repair_attempts": legacy_attempts,
            "question_repair_calls": repair_calls,
            "question_repair_failures": failures,
            "slot_completion_attempts": completion_attempts,
            "repair_working_candidates": working_candidates,
            "repair_replacement_memory": memory_by_slot,
            "repair_retry_seeds": retry_seeds,
            "validation_repair_history": history, "pending_revalidation_ids": {},
            "validated_models": validated_models, "warnings": warnings,
            "repair_call_logs": repair_call_logs[-VALIDATOR_TRACE_MAX_ATTEMPTS:],
            "incompatible_task_ids": sorted(incompatible_task_ids),
            "eval_stats": state.get("eval_stats")}


def _ensure_recovery_v3_decisions(
    state: Dict[str, Any], manifest: list[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], Dict[str, Any] | None]:
    """Persist one Task Alignment decision for every newly failed slot."""
    pending = [item for item in manifest if item.get("status") not in {
        "resolved", "terminal_failure", "exhausted", "failed_to_repair",
        "validation_pending", "candidate_generated",
    } and not item.get("recovery_v3")]
    if not pending:
        return manifest, None
    contexts: list[dict[str, Any]] = []
    ids_to_target: dict[str, Dict[str, Any]] = {}
    exams = {int(exam.get("model_number") or 0): exam
             for exam in state.get("generated_exams") or []}
    for target in pending:
        mn = int(target.get("model_number") or 1)
        slot_id = str(target.get("slot_id") or "")
        decision_id = str(target.get("question_id") or slot_id)
        if not decision_id:
            continue
        plan_by_slot, sources = _plan_context_for_model(state, mn)
        assignment = deepcopy(plan_by_slot.get(slot_id) or {})
        entry = _build_by_id((exams.get(mn) or {}).get("questions") or {}).get(
            str(target.get("question_id") or "")
        )
        question = deepcopy((entry or (None, None, {}))[2])
        variations = list(assignment.get("permitted_variations") or [])
        variations.extend(assignment.get("allowed_variations") or [])
        selected_id = str(assignment.get("selected_variation_id") or "")
        alternatives = []
        seen: set[str] = set()
        for variation in variations:
            variation_id = str(variation.get("variation_id") or "")
            if not variation_id or variation_id == selected_id or variation_id in seen:
                continue
            seen.add(variation_id)
            alternatives.append({
                "variation_id": variation_id,
                "question_types": variation.get("question_types") or variation.get("question_type"),
                "task_operation": variation.get("task_operation") or variation.get("operation"),
                "target": variation.get("target"),
                "numeric_values_may_change": variation.get("numeric_values_may_change"),
                "evidence_chunk_ids": variation.get("evidence_chunk_ids") or [],
                "student_work_product": variation.get("student_work_product"),
            })
        conflicts: list[dict[str, Any]] = []
        conflict_ids = set(map(str, target.get("conflicting_question_ids") or []))
        for exam in state.get("generated_exams") or []:
            for _kind, _container, item in _iter_questions(exam.get("questions") or {}):
                if str(item.get("question_id") or "") in conflict_ids:
                    conflicts.append({
                        "model_number": exam.get("model_number"),
                        "question_id": item.get("question_id"), "question": deepcopy(item),
                    })
        catalog_alternatives: list[dict[str, Any]] = []
        try:
            from app.online.duplicate_pipeline import _candidate_pool
            active_slots = {
                str(item.get("slot_id") or "")
                for exam in state.get("generated_exams") or []
                for _kind, _container, item in _iter_questions(exam.get("questions") or {})
                if item.get("slot_id")
            }
            candidates = _candidate_pool(
                state.get("assessment_catalog") or {}, state.get("plans") or [],
                slot_id, str(target.get("question_type") or ""),
                str(state.get("difficulty") or "mix"), active_slots, set(),
                exclude_variation_ids={selected_id} if selected_id else set(),
            )
            catalog_alternatives = [{
                "task_id": item.get("task_id"),
                "variation_id": item.get("selected_variation_id") or item.get("variation_id"),
                "question_type": item.get("question_type"),
                "task_operation": item.get("task_operation"), "target": item.get("target"),
                "difficulty": item.get("difficulty"),
                "evidence_ids": _plan_source_ids(item),
            } for item in candidates[:20]]
        except (KeyError, TypeError, ValueError):
            # An absent/legacy Catalog does not make the decision call fail;
            # Python still authorizes any later switch against durable state.
            catalog_alternatives = []
        from app.online.multi_repair import build_repair_route, reconcile_repair_conflicts
        route_verdict = deepcopy(target.get("verdict") or {})
        if route_verdict.get("repair_instruction_conflicts") and question:
            route_verdict, _ = reconcile_repair_conflicts(
                str(target.get("question_type") or ""), route_verdict, question,
            )
        repair_route = build_repair_route(
            str(target.get("question_type") or ""), route_verdict,
        )
        contexts.append({
            "question_id": decision_id, "slot_id": slot_id, "model_number": mn,
            "question_type": target.get("question_type"), "question": question,
            "assignment": assignment,
            "authoritative_givens": authoritative_givens(assignment),
            "assigned_evidence": _assigned_source_chunks(assignment, sources),
            "validator_findings": deepcopy(
                (target.get("verdict") or {}).get("required_fixes")
                or target.get("required_fixes") or [target.get("verdict") or {
                    "failure_code": target.get("failure_code"),
                    "reason": target.get("failure_reason"),
                }]
            ),
            "duplicate_conflicts": conflicts,
            "previous_recovery": deepcopy(target.get("history") or []),
            "remaining_budgets": {
                "repair": max(0, MAX_QUESTION_REPAIR_ATTEMPTS - int(
                    (state.get("question_repair_failures") or {}).get(slot_id) or 0)),
                "replacement": max(0, MAX_COMPLETION_ATTEMPTS_PER_SLOT - int(
                    (state.get("slot_completion_attempts") or {}).get(slot_id) or 0)),
            },
            "authorized_alternative_variations": alternatives,
            "backups": deepcopy(assignment.get("backups") or []),
            "catalog_alternatives": catalog_alternatives,
            "repair_route_feasibility": {
                "status": repair_route.get("status"),
                "executable": bool(
                    repair_route.get("status") == "READY"
                    and repair_route.get("steps")
                ),
                "agents": [
                    getattr(step, "agent", None)
                    for step in repair_route.get("steps") or []
                ],
            },
        })
        ids_to_target[decision_id] = target
    if not contexts:
        return manifest, None
    from app.online.recovery_decision import decide_recovery
    domain = str((state.get("assessment_profile") or {}).get("domain") or "")
    decisions, audit = decide_recovery(contexts, domain=domain)
    for decision_id, target in ids_to_target.items():
        target["recovery_v3"] = deepcopy(decisions.get(decision_id) or {
            "version": "recovery-v3-task-alignment", "status": "decision_blocked",
            "error": "Recovery Decision response omitted this question.", "decision_calls": 1,
        })
        target["recovery_decision_audit"] = {
            "decision": target["recovery_v3"].get("decision"),
            "reason": target["recovery_v3"].get("reason"),
            "plan": deepcopy(target["recovery_v3"].get("plan") or []),
        }
    return manifest, audit


def _reassess_exhausted_recovery(
    state: Dict[str, Any], manifest: list[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], Dict[str, Any] | None]:
    """Ask the strategy agent again after a saved plan is exhausted.

    The first decision is still made by the Recovery Decision LLM. Reassessment
    is bounded by the same slot budgets and by a small decision cap; Python
    supplies the latest failure, attempted strategies, and remaining eligible
    strategies.
    """
    from app.online.recovery_decision import decide_recovery
    from app.online.duplicate_pipeline import _candidate_pool

    contexts: list[dict[str, Any]] = []
    by_id: dict[str, Dict[str, Any]] = {}
    exams = {
        int(exam.get("model_number") or 0): exam
        for exam in state.get("generated_exams") or []
    }
    for target in manifest:
        recovery = target.get("recovery_v3") or {}
        if recovery.get("status") != "ready":
            continue
        plan = list(recovery.get("plan") or [])
        if int(recovery.get("next_index") or 0) < len(plan):
            continue
        if target.get("status") in {
            "resolved", "terminal_failure", "exhausted", "failed_to_repair",
        }:
            continue
        if str(target.get("failure_code") or "") in {
            "difficulty_mismatch", "reasoning_complexity_mismatch",
        }:
            # Difficulty-only findings are terminal policy decisions, not a
            # failed semantic strategy that should trigger another repair loop.
            continue
        if any(
            str(event.get("status") or "") in {"accepted", "resolved", "PASSED"}
            for event in target.get("history") or []
        ):
            continue
        if int(recovery.get("decision_calls") or 1) >= 3:
            continue
        mn = int(target.get("model_number") or 1)
        slot_id = str(target.get("slot_id") or "")
        qid = str(target.get("question_id") or "")
        if not slot_id or not qid:
            continue
        plan_by_slot, sources = _plan_context_for_model(state, mn)
        assignment = deepcopy(plan_by_slot.get(slot_id) or {})
        entry = _build_by_id((exams.get(mn) or {}).get("questions") or {}).get(qid)
        question = deepcopy((entry or (None, None, {}))[2])
        if not question:
            continue
        options = copied_source_repair_options(
            assignment, str(target.get("question_type") or ""),
            available_evidence_ids=set(sources),
        )
        alternatives = [
            value for value in (
                assignment.get("permitted_variations")
                or assignment.get("allowed_variations") or []
            )
            if isinstance(value, dict)
            and str(value.get("variation_id") or "") in set(
                options.get("alternative_variation_ids") or []
            )
        ]
        attempted = {
            str(value).upper() for value in recovery.get("attempted_strategies") or []
        }
        content_budget = int(
            (state.get("question_repair_failures") or {}).get(slot_id) or 0
        )
        completion_budget = int(
            (state.get("slot_completion_attempts") or {}).get(slot_id) or 0
        )
        remaining: list[str] = []
        if content_budget < MAX_QUESTION_REPAIR_ATTEMPTS:
            remaining.extend(("REPAIR", "REGENERATE"))
        if alternatives:
            remaining.append("SWITCH_VARIATION")
        if completion_budget < MAX_COMPLETION_ATTEMPTS_PER_SLOT:
            remaining.append("SWITCH_TASK")
        remaining = [value for value in remaining if value not in attempted]
        if not remaining:
            continue
        active_slots = {
            str(item.get("slot_id") or "")
            for exam in state.get("generated_exams") or []
            for _kind, _container, item in _iter_questions(exam.get("questions") or {})
            if item.get("slot_id")
        }
        try:
            catalog_alternatives = [
                {
                    "task_id": item.get("task_id"),
                    "variation_id": item.get("selected_variation_id"),
                    "target": item.get("target"),
                    "task_operation": item.get("task_operation"),
                    "difficulty": item.get("difficulty"),
                    "evidence_ids": _plan_source_ids(item),
                }
                for item in _candidate_pool(
                    state.get("assessment_catalog") or {}, state.get("plans") or [],
                    slot_id, str(target.get("question_type") or ""),
                    str(state.get("difficulty") or "mix"), active_slots, set(),
                    limit=10,
                )
            ]
        except (KeyError, TypeError, ValueError):
            catalog_alternatives = []
        context = {
            "question_id": qid,
            "slot_id": slot_id,
            "model_number": mn,
            "question_type": target.get("question_type"),
            "question": question,
            "assignment": assignment,
            "authoritative_givens": authoritative_givens(assignment),
            "assigned_evidence": _assigned_source_chunks(assignment, sources),
            "validator_findings": deepcopy(
                (target.get("verdict") or {}).get("required_fixes")
                or target.get("required_fixes") or []
            ),
            "previous_recovery": deepcopy(target.get("history") or []),
            "previous_strategy": next((
                value for value in reversed(
                    recovery.get("attempted_strategies") or []
                ) if value
            ), None),
            "previous_result": "FAILED",
            "previous_failure_reason": target.get("failure_reason") or "",
            "remaining_strategies": remaining,
            "remaining_budgets": {
                "repair": max(0, MAX_QUESTION_REPAIR_ATTEMPTS - content_budget),
                "replacement": max(
                    0, MAX_COMPLETION_ATTEMPTS_PER_SLOT - completion_budget,
                ),
            },
            "authorized_alternative_variations": alternatives,
            "catalog_alternatives": catalog_alternatives,
            "backups": deepcopy(assignment.get("backups") or []),
        }
        contexts.append(context)
        by_id[qid] = target
    if not contexts:
        return manifest, None
    domain = str((state.get("assessment_profile") or {}).get("domain") or "")
    decisions, audit = decide_recovery(contexts, domain=domain)
    for qid, target in by_id.items():
        old = target.get("recovery_v3") or {}
        decision = decisions.get(qid) or {}
        chosen = str(decision.get("decision") or "").upper()
        attempted = {
            str(value).upper() for value in old.get("attempted_strategies") or []
        }
        allowed = {
            str(value).upper()
            for value in (next((
                context["remaining_strategies"] for context in contexts
                if context["question_id"] == qid
            ), []) or [])
        }
        if decision.get("status") != "ready" or chosen not in allowed:
            target.update(
                status="terminal_failure",
                failure_code="recovery_strategies_exhausted",
                failure_reason=(
                    "No remaining eligible Recovery strategy could be selected "
                    "after the previous strategy failed."
                ),
            )
            continue
        updated = deepcopy(decision)
        updated["attempted_strategies"] = sorted(attempted)
        updated["decision_calls"] = int(old.get("decision_calls") or 1) + 1
        updated["prior_plan"] = deepcopy(old.get("plan") or [])
        target["recovery_v3"] = updated
        target.setdefault("history", []).append({
            "attempt": 0,
            "status": "strategy_reassessed",
            "previous_strategy": next((
                value for value in reversed(old.get("attempted_strategies") or [])
                if value
            ), None),
            "new_strategy": chosen,
            "remaining_strategies": sorted(allowed),
            "reason": decision.get("reason") or "",
        })
    return manifest, audit


def repair_invalid_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build independent repair candidates concurrently, then merge by slot.

    FITB repairs stay in one ordered group because their shared word bank is a
    section-level dependency. Every other target receives an isolated snapshot.
    """
    manifest = deepcopy(state.get("repair_manifest") or [])
    manifest, reassess_audit = _reassess_exhausted_recovery(state, manifest)
    manifest, decision_audit = _runtime._ensure_recovery_v3_decisions(state, manifest)
    state = {**state, "repair_manifest": manifest}
    if reassess_audit:
        state.setdefault("recovery_decision_audit", []).append(reassess_audit)
    incompatible_task_ids = {
        str(value) for value in state.get("incompatible_task_ids") or [] if value
    }
    active = [item for item in manifest if item.get("status") not in {
        "resolved", "validation_pending",
    }]
    if len(active) < 2:
        result = _runtime._repair_invalid_questions_serial(state)
        if decision_audit:
            result.setdefault("recovery_decision_audit", []).append(decision_audit)
        return result

    jobs: list[list[Dict[str, Any]]] = [
        [item] for item in active
        if str(item.get("question_type") or "") != "fill_in_the_blank"
    ]
    fitb = [item for item in active
            if str(item.get("question_type") or "") == "fill_in_the_blank"]
    if fitb:
        jobs.append(fitb)

    def run_job(targets: list[Dict[str, Any]]) -> Dict[str, Any]:
        from app.online.performance import finish_timing, start_timing
        slot_ids = {str(item.get("slot_id") or "") for item in targets}
        timing = start_timing(
            stage="question_recovery",
            model_number=int(targets[0].get("model_number") or 1),
            slot_id=next(iter(slot_ids)) if len(slot_ids) == 1 else None,
        )
        isolated = {
            **state,
            "repair_manifest": deepcopy(targets),
            "warnings": [],
            "validation_repair_history": [],
            "repair_replacement_memory": {
                key: deepcopy(value)
                for key, value in (state.get("repair_replacement_memory") or {}).items()
                if key in slot_ids
            },
            "repair_retry_seeds": {
                key: deepcopy(value)
                for key, value in (state.get("repair_retry_seeds") or {}).items()
                if key in slot_ids
            },
        }
        result = _runtime._repair_invalid_questions_serial(isolated)
        result.setdefault("performance_telemetry", []).append(
            finish_timing(timing, slot_ids=sorted(slot_ids))
        )
        return result

    completed: dict[int, Dict[str, Any]] = {}
    workers = min(MAX_PARALLEL_VALIDATOR_CALLS, len(jobs), 10)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(copy_context().run, run_job, targets): index
            for index, targets in enumerate(jobs)
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()

    manifest_by_key = {
        (int(item.get("model_number") or 0), str(item.get("slot_id") or "")): item
        for item in manifest
    }
    attempts = dict(state.get("question_repair_attempts") or {})
    repair_calls = dict(state.get("question_repair_calls") or {})
    failures = dict(
        state.get("question_repair_failures")
        or state.get("question_repair_attempts") or {}
    )
    completion_attempts = dict(state.get("slot_completion_attempts") or {})
    working_candidates = deepcopy(state.get("repair_working_candidates") or {})
    memory = deepcopy(state.get("repair_replacement_memory") or {})
    retry_seeds = deepcopy(state.get("repair_retry_seeds") or {})
    candidates: list[Dict[str, Any]] = []
    warnings = list(state.get("warnings") or [])
    history = list(state.get("validation_repair_history") or [])
    repair_call_logs = list(state.get("repair_call_logs") or [])
    performance_telemetry = list(state.get("performance_telemetry") or [])
    validated = set(state.get("validated_models") or [])
    for index in range(len(jobs)):
        result = completed[index]
        incompatible_task_ids.update(result.get("incompatible_task_ids") or [])
        for item in result.get("repair_manifest") or []:
            manifest_by_key[(int(item.get("model_number") or 0),
                             str(item.get("slot_id") or ""))] = item
        for target in jobs[index]:
            slot_id = str(target.get("slot_id") or "")
            if slot_id in (result.get("question_repair_attempts") or {}):
                attempts[slot_id] = result["question_repair_attempts"][slot_id]
            if slot_id in (result.get("question_repair_calls") or {}):
                repair_calls[slot_id] = result["question_repair_calls"][slot_id]
            if slot_id in (result.get("question_repair_failures") or {}):
                failures[slot_id] = result["question_repair_failures"][slot_id]
            if slot_id in (result.get("slot_completion_attempts") or {}):
                completion_attempts[slot_id] = result["slot_completion_attempts"][slot_id]
            if slot_id in (result.get("repair_working_candidates") or {}):
                working_candidates[slot_id] = result["repair_working_candidates"][slot_id]
            if slot_id in (result.get("repair_replacement_memory") or {}):
                memory[slot_id] = result["repair_replacement_memory"][slot_id]
            if slot_id in (result.get("repair_retry_seeds") or {}):
                retry_seeds[slot_id] = result["repair_retry_seeds"][slot_id]
        candidates.extend(result.get("repair_candidates") or [])
        warnings.extend(result.get("warnings") or [])
        history.extend(result.get("validation_repair_history") or [])
        repair_call_logs.extend(result.get("repair_call_logs") or [])
        performance_telemetry.extend(result.get("performance_telemetry") or [])
        validated.intersection_update(result.get("validated_models") or [])

    # One coordinator owns assignment/content reservations. Parallel workers may
    # propose independently, but identical realized instances never proceed to
    # paid revalidation or overwrite an active sibling.
    from app.online.lifecycle import assessment_instance_fingerprint
    active_instances: dict[str, str] = {}
    plan_by_model = {
        int(plan.get("model_number") or 1): {
            str(item.get("slot_id") or ""): item
            for values in (plan.get("items") or {}).values() for item in values or []
        } for plan in state.get("plans") or []
    }
    for exam in state.get("generated_exams") or []:
        mn = int(exam.get("model_number") or 1)
        for section in (exam.get("questions") or {}).values():
            values = section.get("items") if isinstance(section, dict) else section
            for question in values or []:
                sid = str(question.get("slot_id") or "")
                lifecycle = (state.get("slot_lifecycle") or {}).get(sid) or {}
                # Permanently failed candidates do not retain a reservation;
                # active and finalized candidates still do.
                if sid and lifecycle.get("current_status") not in {
                    "FAILED_TO_REPAIR", "VALIDATION_BLOCKED",
                }:
                    active_instances[sid] = assessment_instance_fingerprint(
                        (plan_by_model.get(mn) or {}).get(sid) or {}, question
                    )
    accepted_candidates: list[Dict[str, Any]] = []
    reserved: dict[str, str] = {}
    for candidate in candidates:
        mn = int(candidate.get("model_number") or 1)
        sid = str(candidate.get("slot_id") or "")
        qid = str(candidate.get("question_id") or "")
        staged_plan, _ = _plan_context_for_model(
            {**state, "plans": candidate.get("plans") or []}, mn
        )
        entry = _build_by_id((candidate.get("exam") or {}).get("questions") or {}).get(qid)
        if entry is None or not staged_plan.get(sid):
            # Compatibility for legacy checkpoints and narrow test doubles that
            # predate staged assignment/question payloads.
            accepted_candidates.append(candidate)
            continue
        fingerprint = assessment_instance_fingerprint(
            staged_plan.get(sid) or {}, entry[2] if entry else {}
        )
        conflict_slot = next((
            other_sid for other_sid, other_fp in {**active_instances, **reserved}.items()
            if other_sid != sid and other_fp == fingerprint
        ), None)
        if conflict_slot:
            target = manifest_by_key.get((mn, sid))
            if target is not None:
                target["status"] = "invalid"
                target["failure_code"] = "assessment_instance_reservation_conflict"
                target["failure_reason"] = f"Candidate duplicates reserved/active slot {conflict_slot}."
            history.append({
                "model_number": mn, "slot_id": sid, "question_id": qid,
                "status": "candidate_reservation_rejected",
                "failure_code": "assessment_instance_reservation_conflict",
                "conflict_slot_id": conflict_slot,
                "assessment_instance_fingerprint": fingerprint,
            })
            continue
        reserved[sid] = fingerprint
        candidate["assessment_instance_fingerprint"] = fingerprint
        accepted_candidates.append(candidate)
    candidates = accepted_candidates

    ordered_manifest = [
        manifest_by_key[(int(item.get("model_number") or 0), str(item.get("slot_id") or ""))]
        for item in manifest
    ]
    if isinstance(state.get("eval_stats"), dict):
        exams_by_model = {
            int(value.get("model_number") or 0): value
            for value in state.get("generated_exams") or []
        }
        for report in state.get("validation_reports") or []:
            mn = int(report.get("model_number") or 0)
            exam = exams_by_model.get(mn)
            if exam is None:
                continue
            sent_ids = [
                str(candidate.get("question_id") or "")
                for candidate in candidates
                if int(candidate.get("model_number") or 0) == mn
                and candidate.get("question_id")
            ]
            if sent_ids:
                _runtime._safe_record_repair_outcome(
                    state.get("eval_stats"), exam, report, [], [], sent_ids=sent_ids,
                )
    return {
        "repair_manifest": ordered_manifest,
        "repair_candidates": candidates,
        "question_repair_attempts": attempts,
        "question_repair_calls": repair_calls,
        "question_repair_failures": failures,
        "slot_completion_attempts": completion_attempts,
        "repair_working_candidates": working_candidates,
        "repair_replacement_memory": memory,
        "repair_retry_seeds": retry_seeds,
        "validation_repair_history": history,
        "repair_call_logs": repair_call_logs[-VALIDATOR_TRACE_MAX_ATTEMPTS:],
        "incompatible_task_ids": sorted(incompatible_task_ids),
        "performance_telemetry": performance_telemetry[-5000:],
        "pending_revalidation_ids": {},
        "validated_models": sorted(validated),
        "warnings": warnings,
        "eval_stats": state.get("eval_stats"),
        "recovery_decision_audit": [
            *list(state.get("recovery_decision_audit") or []),
            *([decision_audit] if decision_audit else []),
        ],
    }
