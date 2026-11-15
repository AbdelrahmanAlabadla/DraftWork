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





def validate_generated_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility entry point (implementation in validation.orchestrator)."""
    from app.online.validation.orchestrator import validate_generated_questions as _impl

    return _impl(state)


def repair_invalid_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility entry point (implementation in validation.recovery)."""
    from app.online.validation.recovery import repair_invalid_questions as _impl

    return _impl(state)


def _repair_invalid_questions_serial(state: Dict[str, Any]) -> Dict[str, Any]:
    from app.online.validation.recovery import _repair_invalid_questions_serial as _impl

    return _impl(state)


def _ensure_recovery_v3_decisions(state: Dict[str, Any], manifest: list[Dict[str, Any]]):
    from app.online.validation.recovery import _ensure_recovery_v3_decisions as _impl

    return _impl(state, manifest)


def _safe_record_repair_outcome(*args: Any, **kwargs: Any) -> None:
    from app.online.validation.recovery import _safe_record_repair_outcome as _impl

    return _impl(*args, **kwargs)



from app.online.validation.recovery import (
    GENERAL_VALIDATION_RULE_VERSION,
    STEM_VALIDATION_RULE_VERSION,
    _apply_replacement_difficulty_design,
    _defect_family,
    _defect_signature,
    _finalize_replacement_difficulty_fallback,
    _mark_blocked_repair_route,
    _mark_replacement_difficulty_blocked,
    _normalize_replacement_difficulty_state,
    _replacement_difficulty_unresolved,
    _switch_recovery_variation,
    copied_source_repair_options,
    logger,
    recovery_eligibility,
    recovery_v3_dispatch_action,
)
