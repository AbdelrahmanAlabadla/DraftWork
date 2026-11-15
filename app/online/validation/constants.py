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



# Smallest possible repair actions produced by the validator and consumed by the
# repair stage. ``PASS`` means the question is left byte-identical.
ACTIONS = frozenset(
    {"PASS", "FIX_QUESTION", "FIX_ANSWER", "FIX_OPTIONS", "FIX_QUESTION_AND_ANSWER"}
)
REPAIR_ACTIONS = frozenset(
    {"FIX_QUESTION", "FIX_ANSWER", "FIX_OPTIONS", "FIX_QUESTION_AND_ANSWER"}
)
# Recovery also owns questions whose validation could not complete after the
# Validator's bounded role-specific retries.  They bypass content Repair and
# enter the existing bounded slot-completion/replacement path.
RECOVERY_ACTIONS = REPAIR_ACTIONS | {"UNVALIDATED"}

# Compatibility alias for callers/tests.  This now means failed content
# strategies, not total Repair-agent calls.
MAX_QUESTION_REPAIR_ATTEMPTS = MAX_FAILED_REPAIR_ATTEMPTS_PER_SLOT
MAX_ASSIGNMENT_REPAIR_ATTEMPTS = 3
MAX_CANDIDATE_GENERATION_FAILURES = 3

# Assignment-level failures are not repaired by repeatedly editing the same
# unsuitable task. Move to the next preplanned backup instead.
FUNDAMENTAL_ASSIGNMENT_FAILURES = frozenset({
    "duplicate_question", "duplicate", "excessive_overlap",
    "unsupported", "unsupported_given", "wrong_givens",
    "wrong_variation", "wrong_task_identity", "task_drift",
    "insufficient_evidence", "evidence_mismatch",
    "mcq_task_unsuitable", "unusable_task", "visual_dependency",
    "question_type_operation_mismatch", "response_form_incompatible",
    "constraint_violation", "missing_required_given",
    "generation_failed", "candidate_generation_failed",
})

VALIDATOR_TEMPERATURE = 0.1
# Some served models (e.g. qwen/qwen3-8b) only support reasoning 'off'/'on'.
# 'off' keeps the call fast and non-creative while still producing verdicts.
VALIDATOR_REASONING = "on"

VALIDATION_SCHEMA_VERSION = "validator-verdict-v3"
GENERAL_VALIDATOR_MAX_TOKENS = 100_000
STEM_QUANTITATIVE_VALIDATOR_MAX_TOKENS = 12_000
VALIDATOR_MAX_TOKENS = GENERAL_VALIDATOR_MAX_TOKENS

REPAIR_TEMPERATURE = 0.1
REPAIR_REASONING = "off"
REPAIR_MAX_TOKENS = 12000

TASK_REPLACEMENT_FAILURES: frozenset[str] = frozenset({
    "visual_dependency", "mcq_task_unsuitable", "evidence_mismatch",
    "bad_evidence_assignment", "wrong_task_identity", "bad_fitb_answer_evidence",
    "duplicate_question", "excessive_overlap",
})
