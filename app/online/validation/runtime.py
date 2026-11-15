"""Lazy compatibility bridge for provider entry points.

`validator.py` remains the patchable compatibility facade.  Extracted modules
resolve these three provider helpers through it at call time so existing tests
and integrations can keep monkeypatching `app.online.validator`.
"""
from __future__ import annotations

from typing import Any


def request_structured(*args: Any, **kwargs: Any):
    from app.online import validator

    return validator.request_structured(*args, **kwargs)


def create_llm_client(*args: Any, **kwargs: Any):
    from app.online import validator

    return validator.create_llm_client(*args, **kwargs)


def review_exam_globally(*args: Any, **kwargs: Any):
    from app.online import validator

    return validator.review_exam_globally(*args, **kwargs)


def validate_exam(*args, **kwargs):
    from app.online import validator
    return validator.validate_exam(*args, **kwargs)


def repair_exam(*args, **kwargs):
    from app.online import validator

    return validator.repair_exam(*args, **kwargs)


def validator_batch_size() -> int:
    from app.online import validator

    return int(validator.VALIDATOR_BATCH_SIZE)


def max_parallel_validator_calls() -> int:
    from app.online import validator

    return int(validator.MAX_PARALLEL_VALIDATOR_CALLS)


def _validate_with_context(*args, **kwargs):
    from app.online import validator
    return validator._validate_with_context(*args, **kwargs)


def _regenerate_plan_slot(*args, **kwargs):
    from app.online import validator

    return validator._regenerate_plan_slot(*args, **kwargs)


def _repair_invalid_questions_serial(*args, **kwargs):
    from app.online import validator

    return validator._repair_invalid_questions_serial(*args, **kwargs)


def _ensure_recovery_v3_decisions(*args, **kwargs):
    from app.online import validator

    return validator._ensure_recovery_v3_decisions(*args, **kwargs)


def _safe_record_repair_outcome(*args, **kwargs):
    from app.online import validator

    return validator._safe_record_repair_outcome(*args, **kwargs)


def record_repair_outcome(*args, **kwargs):
    from app.online import validator

    return validator.record_repair_outcome(*args, **kwargs)


def record_optional_failure(*args, **kwargs):
    from app.online import validator

    return validator.record_optional_failure(*args, **kwargs)
