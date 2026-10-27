from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, StrictBool, ValidationError

from app import config
from app.llm.client import request_structured
from app.llm.factory import (
    assessment_compiler_identity,
    create_assessment_compiler_client,
)
from app.llm.deepseek import DeepSeekIncompleteError, DeepSeekValidationError
from app.llm.schemas import (
    AssessmentProfile,
    NonEmptyString,
    QuestionType,
    ReasoningMode,
    StrictOutputModel,
    TaskVariation,
)
from app.logging_conf import get_logger
from app.online.semantic_keys import formula_semantic_key

logger = get_logger("ASSESSMENT_CATALOG")


class CatalogBatchIncompleteError(RuntimeError):
    """A compiler batch never produced an explicit complete, valid response."""

    def __init__(self, batch_index: int, attempts: int, reason: str):
        self.batch_index = batch_index
        self.attempts = attempts
        self.reason = reason
        super().__init__(
            f"Assessment Catalog batch {batch_index + 1} remained incomplete "
            f"after {attempts} attempts: {reason}"
        )


class CatalogCompilationCancelled(RuntimeError):
    """Catalog compilation stopped because its owning job was cancelled."""


_FITB_ANSWER_NOISE = {
    "a", "an", "the", "term", "name", "concept", "phenomenon", "series",
    "identify", "select", "recognize", "from", "definition", "description",
    "answer", "keyword", "phrase", "سلسلة", "مصطلح", "اسم", "مفهوم", "حدد",
    "اختر", "تعرف", "التعريف", "الوصف",
}
_FITB_TRANSLITERATION_ALIASES = {
    "ليمان": "lyman", "بالمر": "balmer", "باشان": "paschen",
}


def normalize_fitb_answer_key(value: Any) -> str:
    """Normalize a FITB answer identity without conflating its catalog task."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w\u0600-\u06ff]+", " ", text, flags=re.UNICODE)
    tokens = []
    for token in text.split():
        token = _FITB_TRANSLITERATION_ALIASES.get(token, token)
        if token not in _FITB_ANSWER_NOISE:
            tokens.append(token)
    return " ".join(tokens)


def fitb_answer_keys(item: dict[str, Any]) -> set[str]:
    """Return answer identities separately from task inventory identity."""
    explicit_values: list[Any] = [item.get("fitb_answer"), item.get("expected_answer")]
    explicit_values.extend(item.get("answers") or [])
    explicit = {
        normalize_fitb_answer_key(value) for value in explicit_values if value
    }
    if explicit:
        return {key for key in explicit if key}
    # Atomic catalog targets are the expected work/result. Broader concept and
    # knowledge labels are fallbacks only; combining all of them would make two
    # different answers under one concept look falsely identical.
    for field in ("target", "concept", "knowledge_anchor", "knowledge_focus"):
        key = normalize_fitb_answer_key(item.get(field))
        if key:
            return {key}
    return set()


def is_authorized_catalog_exhaustion_reuse(item: dict[str, Any]) -> bool:
    """Return whether a plan item carries the complete exact-task reuse contract."""
    task_id = str(item.get("task_id") or "")
    return bool(
        task_id
        and item.get("task_reused") is True
        and item.get("reuse_reason") == "no_unused_compatible_task"
        and item.get("reused_due_to_catalog_exhaustion") is True
        and item.get("catalog_replacement_stage") in {
            "reused_task_after_catalog_exhaustion", "deterministic_intentional_reuse",
            "semantic_plan_repair_intentional_reuse",
        }
        and str(item.get("reused_task_id") or "") == task_id
        and item.get("required_instance_distinctions")
        and item.get("forced_duplicate_slot_ids")
    )


def is_authorized_intentional_practice_reuse(item: dict[str, Any]) -> bool:
    """Return whether exact-task reuse has the complete practice-variation contract.

    This includes both least-used catalog reuse and regeneration of the original
    valid task after a generated duplicate.  The explicit metadata prevents a
    bare ``reuse_mode`` label from bypassing generator duplicate safeguards.
    """
    if is_authorized_catalog_exhaustion_reuse(item):
        return True
    task_id = str(item.get("task_id") or "")
    return bool(
        task_id
        and item.get("task_reused") is True
        and item.get("reuse_mode") == "intentional_practice_variation"
        and item.get("reuse_reason") == "regenerate_original_task_after_duplicate"
        and item.get("catalog_replacement_stage")
        == "regenerated_original_task_after_duplicate"
        and str(item.get("reused_task_id") or "") == task_id
        and int(item.get("task_usage_count") or 0) > 0
        and item.get("variation_reason")
        and item.get("intended_variation")
        and item.get("source_evidence_ids")
        and item.get("required_instance_distinctions")
        and item.get("forced_duplicate_slot_ids")
    )


def is_authorized_fitb_entry_reuse(item: dict[str, Any]) -> bool:
    """Return whether a FITB plan item carries the explicit bank-reuse contract."""
    entry_id = str(item.get("fitb_entry_id") or "")
    return bool(
        entry_id
        and item.get("fitb_entry_reused") is True
        and item.get("reuse_reason") == "no_unused_compatible_fitb_entry"
        and item.get("catalog_replacement_stage") in {
            "reused_fitb_entry_after_bank_exhaustion",
            "deterministic_intentional_reuse",
        }
        and str(item.get("reused_fitb_entry_id") or "") == entry_id
        and item.get("required_instance_distinctions")
        and item.get("forced_duplicate_slot_ids")
    )


def compatible_task_inventory(
    catalog: dict[str, Any], plans: list[dict[str, Any]], qtype: str,
    requested_difficulty: str, *, excluding_slot_id: str = "",
    excluded_task_ids: set[str] | None = None,
    active_slot_ids: set[str] | None = None,
) -> dict[str, Any]:
    if qtype == "fill_in_the_blank":
        entries = [
            entry for entry in catalog.get("fitb_bank") or []
            if entry.get("fitb_entry_id") and not entry.get("alias_of_fitb_entry_id")
        ]
        compatible_ids = {str(entry["fitb_entry_id"]) for entry in entries}
        used_ids = {
            str(item.get("fitb_entry_id"))
            for plan in plans
            for values in (plan.get("items") or {}).values()
            for item in values or []
            if item.get("fitb_entry_id")
            and str(item.get("slot_id") or "") != excluding_slot_id
            and (active_slot_ids is None or str(item.get("slot_id") or "") in active_slot_ids)
        } & compatible_ids
        excluded_ids = {
            str(value) for value in (excluded_task_ids or set()) if str(value) in compatible_ids
        }
        unused_ids = compatible_ids - used_ids
        eligible = unused_ids - excluded_ids
        return {
            "question_type": qtype,
            "identity_kind": "fitb_entry_id",
            "compatible_fitb_entry_ids": sorted(compatible_ids),
            "compatible_used_fitb_entry_ids": sorted(used_ids),
            "compatible_unused_fitb_entry_ids": sorted(unused_ids),
            # Generic aliases keep inventory/lifecycle consumers stable.
            "compatible_task_ids": sorted(compatible_ids),
            "compatible_used_task_ids": sorted(used_ids),
            "compatible_unused_task_ids": sorted(unused_ids),
            "compatible_total_count": len(compatible_ids),
            "compatible_used_count": len(used_ids),
            "compatible_unused_count": len(unused_ids),
            "all_unused_compatible_task_ids": sorted(unused_ids),
            "all_unused_compatible_count": len(unused_ids),
            "excluded_compatible_task_ids": sorted(excluded_ids),
            "eligible_unused_after_exclusions_task_ids": sorted(eligible),
            "eligible_unused_after_exclusions_count": len(eligible),
            "reuse_mode": not unused_ids and bool(compatible_ids),
            "reuse_reason": "no_unused_compatible_fitb_entry" if not unused_ids and compatible_ids else None,
        }
    indexed_ids = set((catalog.get("tasks_by_question_type") or {}).get(qtype) or [])
    profile = catalog.get("assessment_profile") or {}
    valid_chunks = set(map(str, catalog.get("selected_chunk_ids") or [])) or None
    compatible_ids = {
        str(task.get("task_id"))
        for task in catalog.get("tasks") or []
        if task.get("task_id")
        and (
            str(task.get("task_id")) in indexed_ids
            if indexed_ids else qtype in (task.get("allowed_question_types") or [])
        )
        and eligible_task_variation_candidates(
            task, qtype, assessment_profile=profile, valid_chunk_ids=valid_chunks,
        )
        and _difficulty_for(task, requested_difficulty) is not None
    }
    used_ids = {
        str(item.get("task_id"))
        for plan in plans
        for values in (plan.get("items") or {}).values()
        for item in values or []
        if item.get("task_id")
        and str(item.get("slot_id") or "") != excluding_slot_id
        and (
            active_slot_ids is None
            or str(item.get("slot_id") or "") in active_slot_ids
        )
        and str(item.get("task_id")) in compatible_ids
    }
    unused_ids = compatible_ids - used_ids
    excluded_ids = {
        str(task_id) for task_id in (excluded_task_ids or set())
        if str(task_id) in compatible_ids
    }
    eligible_unused_ids = unused_ids - excluded_ids
    return {
        "question_type": qtype,
        "compatible_task_ids": sorted(compatible_ids),
        "compatible_used_task_ids": sorted(used_ids),
        "compatible_unused_task_ids": sorted(unused_ids),
        "compatible_total_count": len(compatible_ids),
        "compatible_used_count": len(used_ids),
        "compatible_unused_count": len(unused_ids),
        "all_unused_compatible_task_ids": sorted(unused_ids),
        "all_unused_compatible_count": len(unused_ids),
        "excluded_compatible_task_ids": sorted(excluded_ids),
        "eligible_unused_after_exclusions_task_ids": sorted(eligible_unused_ids),
        "eligible_unused_after_exclusions_count": len(eligible_unused_ids),
        "reuse_mode": not unused_ids and bool(compatible_ids),
        "reuse_reason": "no_unused_compatible_task" if not unused_ids and compatible_ids else None,
    }


def _inventory_after_reservation(
    inventory: dict[str, Any], task_id: str, *, reused: bool
) -> dict[str, Any]:
    """Derive inventory counts from ID sets after one atomic reservation."""
    compatible = set(inventory.get("compatible_task_ids") or [])
    used = set(inventory.get("compatible_used_task_ids") or [])
    if task_id in compatible:
        used.add(task_id)
    unused = compatible - used
    excluded = set(inventory.get("excluded_compatible_task_ids") or [])
    eligible_unused = unused - excluded
    return {
        **inventory,
        "compatible_used_task_ids": sorted(used),
        "compatible_unused_task_ids": sorted(unused),
        "compatible_total_count": len(compatible),
        "compatible_used_count": len(used),
        "compatible_unused_count": len(unused),
        "remaining_unused_compatible_task_ids": sorted(unused),
        "remaining_unused_compatible_count": len(unused),
        "remaining_eligible_unused_after_exclusions_task_ids": sorted(eligible_unused),
        "remaining_eligible_unused_after_exclusions_count": len(eligible_unused),
        "reuse_mode": reused,
        "reuse_reason": (
            inventory.get("reuse_reason") or "no_unused_compatible_task"
        ) if reused else None,
    }


class CatalogTaskDraft(StrictOutputModel):
    knowledge_anchor: str = Field(min_length=1, max_length=240)
    concept: str = Field(min_length=1, max_length=200)
    relationship: str | None = Field(max_length=240)
    task_operation: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    student_work_product: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    reasoning_mode: ReasoningMode
    allowed_question_types: list[QuestionType] = Field(min_length=1)
    evidence_chunk_ids: list[NonEmptyString] = Field(min_length=1, max_length=4)
    required_givens: list[NonEmptyString] = Field(max_length=12)
    allowed_variations: list[TaskVariation] = Field(default_factory=list)
    variable_givens: list[NonEmptyString] = Field(default_factory=list, max_length=12)
    complexity_boundaries: list[NonEmptyString] = Field(default_factory=list, max_length=8)
    exact_source_instance: bool = False
    scenario_family: str | None = Field(max_length=160)
    text_answerable: StrictBool
    native_difficulty: Literal["easy", "medium", "hard"]
    prerequisites: list[NonEmptyString] = Field(default_factory=list, max_length=8)


class FITBEntryDraft(StrictOutputModel):
    """Compact source-grounded vocabulary candidate owned by the compiler."""

    answer: str = Field(min_length=1, max_length=80)
    evidence_chunk_id: NonEmptyString
    clue_basis: str = Field(min_length=1, max_length=320)


class AssessmentCatalogResponse(StrictOutputModel):
    assessment_profile: AssessmentProfile
    tasks: list[CatalogTaskDraft]
    fitb_bank: list[FITBEntryDraft] = Field(default_factory=list, max_length=128)


class CompactCatalogTask(StrictOutputModel):
    """LLM-facing task contract for one compiler batch.

    Readable wire aliases reduce schema/output size. ``model_dump`` normalizes
    them to the full internal field names before any downstream consumer sees
    the task.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    knowledge_anchor: str = Field(alias="anchor", min_length=1, max_length=240)
    concept: str = Field(min_length=1, max_length=200)
    relationship: str | None = Field(alias="relation", max_length=240)
    task_operation: str = Field(alias="operation", min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    student_work_product: str = Field(alias="work_product", min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    reasoning_mode: ReasoningMode = Field(alias="mode")
    allowed_question_types: list[QuestionType] = Field(alias="question_types", min_length=1)
    evidence_chunk_ids: list[NonEmptyString] = Field(alias="evidence_ids", min_length=1, max_length=4)
    required_givens: list[NonEmptyString] = Field(max_length=12)
    allowed_variations: list[TaskVariation] = Field(alias="variations", default_factory=list)
    variable_givens: list[NonEmptyString] = Field(default_factory=list, max_length=12)
    complexity_boundaries: list[NonEmptyString] = Field(alias="boundaries", default_factory=list, max_length=8)
    exact_source_instance: bool = False
    scenario_family: str | None = Field(max_length=160)
    text_answerable: StrictBool = Field(alias="answerable")
    native_difficulty: Literal["easy", "medium", "hard"] = Field(alias="difficulty")
    prerequisites: list[NonEmptyString] = Field(alias="prereqs", default_factory=list, max_length=8)


class CompactCatalogBatchResponse(StrictOutputModel):
    """Compact per-batch response for every batch except the first."""

    tasks: list[CompactCatalogTask]
    fitb_bank: list[FITBEntryDraft] = Field(default_factory=list, max_length=128)
    batch_complete: Literal[True]


class CompactCatalogBatchWithProfileResponse(StrictOutputModel):
    """First-batch response: carries the single catalog-level profile."""

    assessment_profile: AssessmentProfile
    tasks: list[CompactCatalogTask]
    fitb_bank: list[FITBEntryDraft] = Field(default_factory=list, max_length=128)
    batch_complete: Literal[True]


CATALOG_CONTRACT_VERSION = "catalog-v13-numerical-givens-contract"
CATALOG_BATCHING_VERSION = "parent-pairs-v1"


def _completed_json_objects(text: str) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    stack: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text or ""):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            try:
                value = json.loads(text[start:index + 1])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                completed.append(value)
    return completed


def _recover_catalog_checkpoint(
    text: str, *, require_profile: bool = True,
) -> dict[str, Any] | None:
    objects = _completed_json_objects(text)
    profile = next((
        value for value in objects
        if isinstance(value.get("question_types"), dict)
        and isinstance(value.get("domains"), list)
        and value.get("subject")
        and value.get("domain") in {"STEM", "NON_STEM"}
    ), None)
    if profile is not None:
        try:
            profile = AssessmentProfile.model_validate(profile).model_dump()
        except ValidationError:
            profile = None
    tasks: list[dict[str, Any]] = []
    for value in objects:
        if not (
            (value.get("anchor") or value.get("knowledge_anchor"))
            and (value.get("operation") or value.get("task_operation"))
            and isinstance(value.get("evidence_ids") or value.get("evidence_chunk_ids"), list)
        ):
            continue
        try:
            tasks.append(CompactCatalogTask.model_validate(value).model_dump())
        except ValidationError:
            continue
    fitb_bank: list[dict[str, Any]] = []
    for value in objects:
        if not (
            value.get("answer") and value.get("evidence_chunk_id")
            and value.get("clue_basis") and not value.get("task_operation")
        ):
            continue
        try:
            fitb_bank.append(FITBEntryDraft.model_validate(value).model_dump())
        except ValidationError:
            continue
    if require_profile and profile is None:
        return None
    if not tasks and not fitb_bank:
        return None
    logger.warning(
        "Recovered assessment catalog checkpoint | complete_tasks=%d | discarded_tail=true",
        len(tasks),
    )
    recovered: dict[str, Any] = {
        "tasks": tasks,
        "fitb_bank": fitb_bank,
        "batch_complete": False,
        "recovered_truncated_checkpoint": True,
    }
    if profile is not None:
        recovered["assessment_profile"] = profile
    return recovered


def _validate_completed_batch(
    value: Any, *, response_model: type[StrictOutputModel], valid_chunk_ids: set[str],
) -> dict[str, Any]:
    """Fail closed unless one batch is explicit, schema-valid, and source-bound."""
    parsed = response_model.model_validate(value)
    normalized = parsed.model_dump()
    if normalized.get("batch_complete") is not True:
        raise ValueError("batch_complete must be explicitly true")
    for task in normalized.get("tasks") or []:
        evidence = set(map(str, task.get("evidence_chunk_ids") or []))
        if not evidence or not evidence.issubset(valid_chunk_ids):
            raise ValueError("task evidence IDs are outside this compiler batch")
        for variation in task.get("allowed_variations") or []:
            variation_evidence = set(map(str, variation.get("evidence_chunk_ids") or []))
            if variation_evidence and not variation_evidence.issubset(valid_chunk_ids):
                raise ValueError("variation evidence IDs are outside this compiler batch")
    for entry in normalized.get("fitb_bank") or []:
        if str(entry.get("evidence_chunk_id") or "") not in valid_chunk_ids:
            raise ValueError("FITB evidence ID is outside this compiler batch")
    return normalized


def _normalized(value: object) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), flags=re.UNICODE))


def _stable_id(prefix: str, *values: object) -> str:
    digest = hashlib.sha256(
        "\x1f".join(_normalized(value) for value in values).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


_FITB_FORBIDDEN_SYMBOLS = re.compile(
    r"[\d=+*/^<>≤≥≈≠×÷√∑∫∆Δ→←↔⇌λν]|[₀-₉⁰-⁹]", re.UNICODE
)
_FITB_UNIT_PATTERN = re.compile(
    r"\b(?:kg|mol|mmol|cm|mm|km|m/s|m²|m³|hz|khz|mhz|ghz|kj|kw|pa|kpa|atm)\b",
    re.IGNORECASE,
)
_FITB_SINGLE_LETTER_UNIT_PATTERN = re.compile(r"(?:J|W|V|A|C)")
_FITB_FORMULA_TOKEN = re.compile(r"\b(?:[A-Z][a-z]?){2,}\b")
_FITB_CALCULATION_CUE_PATTERN = re.compile(
    r"\b(?:calculate|compute|solve|evaluate|formula|equation|law|unit)\b|"
    r"(?:احسب|أوجد|اوجد|معادلة|قانون|وحدة)",
    re.IGNORECASE,
)


def is_pure_text_fitb_value(value: Any, *, allow_blank: bool = False) -> bool:
    """Reject quantitative/symbolic FITB answers, clues, and generated stems."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if (
        not text or _FITB_FORBIDDEN_SYMBOLS.search(text)
        or _FITB_UNIT_PATTERN.search(text)
        or (
            not allow_blank
            and _FITB_SINGLE_LETTER_UNIT_PATTERN.fullmatch(text)
        )
    ):
        return False
    if _FITB_FORMULA_TOKEN.search(text):
        return False
    if allow_blank and _FITB_CALCULATION_CUE_PATTERN.search(text):
        return False
    blank_runs = re.findall(r"_{2,}", text)
    if allow_blank:
        if len(blank_runs) != 1:
            return False
        text = re.sub(r"_{2,}", " ", text)
    elif blank_runs or "_" in text:
        return False
    # A hyphen inside ordinary prose is acceptable; standalone arithmetic minus is not.
    if re.search(r"(?:^|\s)-(?:\s|$)", text):
        return False
    return bool(re.search(r"[^\W\d_]", text, re.UNICODE))


def canonical_fitb_entry(raw: dict[str, Any], valid_ids: set[str]) -> dict[str, Any] | None:
    answer = str(raw.get("answer") or "").strip()
    evidence_chunk_id = str(raw.get("evidence_chunk_id") or "").strip()
    clue_basis = str(raw.get("clue_basis") or "").strip()
    normalized_answer = normalize_fitb_answer_key(answer)
    if (
        evidence_chunk_id not in valid_ids
        or not normalized_answer
        or len(answer.split()) > 8
        or not is_pure_text_fitb_value(answer)
        or not is_pure_text_fitb_value(clue_basis)
    ):
        return None
    return {
        "fitb_entry_id": _stable_id(
            "fitb", normalized_answer, evidence_chunk_id, clue_basis
        ),
        "answer": answer,
        "normalized_answer": normalized_answer,
        "evidence_chunk_id": evidence_chunk_id,
        "clue_basis": clue_basis,
        "usage_count": 0,
        "used": False,
        "alias_of_fitb_entry_id": None,
        "duplicate_normalized_answer": False,
    }


CANONICAL_TASK_ID_FIELDS = (
    "task_id", "task_family_id", "coverage_id", "concept_id", "relationship_id",
)


def canonical_task_ids(item: dict[str, Any]) -> dict[str, str | None] | None:
    """Derive the shared canonical identity for a catalog task or plan item.

    Catalog tasks call their grounded knowledge field ``knowledge_anchor`` while
    plan items call the same semantic field ``knowledge_focus``.  The identity
    payload is otherwise identical, so fallback and repaired plans cannot create
    a second task-ID representation.
    """
    knowledge_anchor = item.get("knowledge_anchor") or item.get("knowledge_focus")
    required = {
        "knowledge_anchor": knowledge_anchor,
        "concept": item.get("concept"),
        "task_operation": item.get("task_operation"),
    }
    if any(not str(value or "").strip() for value in required.values()):
        return None

    coverage_id = _stable_id("knowledge", knowledge_anchor)
    concept_id = _stable_id("concept", item.get("concept"))
    relationship = item.get("relationship")
    relationship_basis = formula_semantic_key(relationship) or relationship
    relationship_id = (
        _stable_id("relationship", relationship_basis) if relationship else None
    )
    return {
        "task_id": _stable_id(
            "task", coverage_id, item.get("concept"), item.get("task_operation"),
            relationship_basis or "",
            sorted(_normalized(value) for value in item.get("required_givens") or []),
            item.get("scenario_family") or "",
            sorted(_normalized(value) for value in item.get("complexity_boundaries") or []),
        ),
        "task_family_id": _stable_id(
            "family", coverage_id, item.get("concept"),
            relationship_basis or "", item.get("scenario_family") or "",
        ),
        "coverage_id": coverage_id,
        "concept_id": concept_id,
        "relationship_id": relationship_id,
    }


def apply_canonical_task_ids(item: dict[str, Any]) -> bool:
    """Recompute canonical IDs in place; clear stale IDs if semantics are incomplete."""
    identity = canonical_task_ids(item)
    for field in CANONICAL_TASK_ID_FIELDS:
        item.pop(field, None)
    if identity is None:
        return False
    item.update(identity)
    return True


def _normalize_variations(values: Any) -> list[dict[str, Any]]:
    """Normalize legacy/multi-type angles into grouped, type-specific variations."""
    normalized: dict[str, dict[str, Any]] = {}
    for value in values or []:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if not isinstance(value, dict):
            continue
        try:
            variation = TaskVariation.model_validate(value).model_dump()
        except ValidationError:
            continue
        question_types = list(dict.fromkeys(variation.get("question_types") or []))
        supplied_group = str(variation.get("variation_group_id") or "")
        group_id = (
            supplied_group if supplied_group.startswith("variation_group_")
            else _stable_id("variation_group", supplied_group) if supplied_group
            else _stable_id(
            "variation_group", variation.get("target"), variation.get("operation"),
            sorted(_normalized(value) for value in variation.get("givens") or []),
            variation.get("reasoning_mode") or "",
            variation.get("student_work_product") or "",
            variation.get("representation") or "", variation.get("application") or "",
            variation.get("numeric_values_may_change", False),
            )
        )
        # Persisted pre-v12 catalogs did not bind variations to a question
        # type. Preserve those angles for display/migration, while excluding
        # them from typed eligibility until a primary typed contract is built.
        typed_question_types: list[str | None] = question_types or [None]
        for question_type in typed_question_types:
            typed = {**variation, "question_types": [question_type],
                     "variation_group_id": group_id}
            if question_type is None:
                typed["question_types"] = []
            variation_id = _stable_id("variation", group_id, question_type or "legacy")
            typed["variation_id"] = variation_id
            existing = normalized.get(variation_id)
            if existing is None:
                normalized[variation_id] = typed
                continue
            evidence = list(dict.fromkeys([
                *(existing.get("evidence_chunk_ids") or []),
                *(typed.get("evidence_chunk_ids") or []),
            ]))
            existing["evidence_chunk_ids"] = evidence or None
    return list(normalized.values())


def _primary_variations(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose one type-specific primary variation for every declared type."""
    primary = {
        "target": task.get("target"),
        "operation": task.get("task_operation"),
        "givens": list(task.get("required_givens") or []),
        "question_types": list(task.get("primary_question_types") or task.get("allowed_question_types") or []),
        "evidence_chunk_ids": list(task.get("evidence_chunk_ids") or []),
        "reasoning_mode": task.get("reasoning_mode"),
        "student_work_product": task.get("student_work_product"),
        "representation": None,
        "application": task.get("scenario_family"),
        "numeric_values_may_change": bool(task.get("variable_givens")),
    }
    normalized = _normalize_variations([primary])
    if normalized:
        return normalized
    # Legacy/test plans may carry a free-form reasoning label from the pre-contract era.
    primary["reasoning_mode"] = None
    normalized = _normalize_variations([primary])
    if normalized:
        return normalized
    primary["student_work_product"] = None
    return _normalize_variations([primary])


def _primary_variation(task: dict[str, Any]) -> dict[str, Any]:
    """Compatibility helper for callers that need the first primary angle."""
    return _primary_variations(task)[0]


def _scope_variations(
    task: dict[str, Any], variations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Make group and type-specific IDs stable within their owning task."""
    task_id = str(task.get("task_id") or "")
    if not task_id:
        return variations
    scoped: list[dict[str, Any]] = []
    for variation in variations:
        qtypes = list(variation.get("question_types") or [])
        if len(qtypes) > 1:
            continue
        # Recompute from semantic fields rather than wrapping a previously
        # generated group ID. This makes scoping idempotent across cache load,
        # allocation, repair, and resume while retaining cross-type grouping.
        group_id = _stable_id(
            "variation_group", task_id, variation.get("target"),
            variation.get("operation"),
            sorted(_normalized(value) for value in variation.get("givens") or []),
            variation.get("reasoning_mode") or "",
            variation.get("student_work_product") or "",
            variation.get("representation") or "", variation.get("application") or "",
            variation.get("numeric_values_may_change", False),
        )
        scoped.append({
            **variation,
            "variation_group_id": group_id,
            "variation_id": _stable_id(
                "variation", group_id, qtypes[0] if qtypes else "legacy"
            ),
        })
    return scoped


def permitted_task_variations(task: dict[str, Any], question_type: str | None = None) -> list[dict[str, Any]]:
    scoped = _scope_variations(task, [
        *_primary_variations(task), *_normalize_variations(task.get("allowed_variations"))
    ])
    # A finalized task may already contain its primary variation in
    # allowed_variations.  Keep this operation idempotent across cache loads.
    by_id: dict[str, dict[str, Any]] = {}
    for variation in scoped:
        variation_id = str(variation.get("variation_id") or "")
        existing = by_id.get(variation_id)
        if existing is None:
            by_id[variation_id] = variation
            continue
        evidence = list(dict.fromkeys([
            *(existing.get("evidence_chunk_ids") or []),
            *(variation.get("evidence_chunk_ids") or []),
        ]))
        existing["evidence_chunk_ids"] = evidence or None
    variations = list(by_id.values())
    if question_type:
        variations = [
            variation for variation in variations
            if question_type in (variation.get("question_types") or [])
        ]
    return variations


def eligible_task_variations(
    task: dict[str, Any], question_type: str, *,
    variations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the single shared eligibility contract for every planning path."""
    if question_type not in (task.get("allowed_question_types") or []):
        return []
    if task.get("text_answerable") is False:
        return []
    parent_evidence = [str(value) for value in task.get("evidence_chunk_ids") or [] if value]
    eligible: list[dict[str, Any]] = []
    permitted = variations if variations is not None else permitted_task_variations(task)
    for variation in permitted:
        if question_type not in (variation.get("question_types") or []):
            continue
        operation = str(variation.get("operation") or task.get("task_operation") or "").strip()
        target = str(variation.get("target") or task.get("target") or "").strip()
        mode = str(variation.get("reasoning_mode") or task.get("reasoning_mode") or "")
        givens = [str(value).strip() for value in variation.get("givens") or [] if str(value).strip()]
        relationship = str(task.get("relationship") or "").strip()
        evidence = variation.get("evidence_chunk_ids")
        effective_evidence = parent_evidence if evidence is None else [
            str(value) for value in evidence if value
        ]
        if not operation or not target or not effective_evidence:
            continue
        if mode in _RELATIONSHIP_VARIATION_MODES and not (relationship or givens):
            continue
        if _has_visual_dependency(_variation_idea(task, variation)) and not givens:
            continue
        eligible.append(variation)
    return eligible


PLANNER_SCORE_WEIGHTS = {
    "task_reuse": 30, "difficulty": 15, "scenario_diversity": 20,
    "target_diversity": 15, "variation_reuse": 20,
}
_TASK_REUSE_POINTS = (30, 20, 12, 6)
_SCENARIO_REUSE_POINTS = (20, 12, 6)
_TARGET_REUSE_POINTS = (15, 9, 4)
_VARIATION_REUSE_POINTS = (20, 12, 5)
_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


def normalize_planner_diversity_text(value: Any) -> str:
    """Conservatively normalize scenario/target labels for equality-only counts."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = text.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ـ": "",
    }))
    # Preserve arithmetic/comparison operators so normalization cannot merge
    # mathematically different targets such as x + 1 and x - 1.
    tokens = re.findall(r"[\w\u0600-\u06ff]+|[+\-*/=<>≤≥≠]", text, flags=re.UNICODE)
    normalized: list[str] = []
    for token in tokens:
        # A bounded English inflection fold keeps this dependency-free and
        # deliberately avoids semantic or fuzzy matching.
        if token.isascii() and token.isalpha():
            if token.endswith("ing") and len(token) > 5:
                stem = token[:-3]
                if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
                    stem = stem[:-1]
                token = stem + ("e" if stem.endswith("v") else "")
            elif token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
                token = token[:-1]
        normalized.append(token)
    return " ".join(normalized)


def _curve(points: tuple[int, ...], uses: int) -> int:
    return points[uses] if 0 <= uses < len(points) else 0


def score_task_variation_candidate(
    task: dict[str, Any], variation: dict[str, Any], *, requested_difficulty: str,
    task_uses: int = 0, scenario_uses: int = 0, target_uses: int = 0,
    variation_group_uses: int = 0,
) -> dict[str, Any]:
    """Return the central, explainable 0-100 Planner preference score."""
    native = str(_native_difficulty(task) or "medium")
    requested = str(requested_difficulty or "mix")
    difficulty_points = 0
    if requested in _DIFFICULTY_ORDER and native in _DIFFICULTY_ORDER:
        difficulty_points = {0: 15, 1: 8, 2: 2}[
            abs(_DIFFICULTY_ORDER[requested] - _DIFFICULTY_ORDER[native])
        ]
    scenario = variation.get("application")
    if scenario is None:
        scenario = task.get("scenario_family")
    target = variation.get("target") or task.get("target")
    parts = {
        "task_reuse": _curve(_TASK_REUSE_POINTS, int(task_uses)),
        "difficulty": difficulty_points,
        "scenario_diversity": (
            _curve(_SCENARIO_REUSE_POINTS, int(scenario_uses))
            if normalize_planner_diversity_text(scenario) else 10
        ),
        "target_diversity": (
            _curve(_TARGET_REUSE_POINTS, int(target_uses))
            if normalize_planner_diversity_text(target) else 7
        ),
        "variation_reuse": _curve(
            _VARIATION_REUSE_POINTS, int(variation_group_uses)
        ),
    }
    return {"score": sum(parts.values()), "components": parts}


def eligible_task_variation_candidates(
    task: dict[str, Any], question_type: str, *,
    assessment_profile: dict[str, Any] | None = None,
    valid_chunk_ids: set[str] | None = None,
    variations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Apply the shared hard filter and return materializable variations."""
    if task.get("text_answerable") is not True:
        return []
    type_profile = (
        ((assessment_profile or {}).get("question_types") or {}).get(question_type) or {}
    )
    allowed_operations = set(map(str, type_profile.get("task_operations") or []))
    parent_evidence = [str(value) for value in task.get("evidence_chunk_ids") or [] if value]
    candidates: list[dict[str, Any]] = []
    for variation in eligible_task_variations(
        task, question_type, variations=variations,
    ):
        operation = str(variation.get("operation") or task.get("task_operation") or "").strip()
        target = str(variation.get("target") or task.get("target") or "").strip()
        product = str(
            variation.get("student_work_product") or task.get("student_work_product") or ""
        ).strip()
        mode = str(variation.get("reasoning_mode") or task.get("reasoning_mode") or "").strip()
        evidence_value = variation.get("evidence_chunk_ids")
        evidence = parent_evidence if evidence_value is None else [
            str(value) for value in evidence_value if value
        ]
        if not all((operation, target, product, mode, evidence)):
            continue
        if allowed_operations and operation not in allowed_operations:
            continue
        if valid_chunk_ids is not None and any(value not in valid_chunk_ids for value in evidence):
            continue
        candidates.append(variation)
    return candidates


def selected_variation(task: dict[str, Any], variation_id: str, question_type: str) -> dict[str, Any] | None:
    permitted = task.get("permitted_variations")
    if isinstance(permitted, list) and permitted:
        return next((
            variation for variation in permitted
            if variation.get("variation_id") == variation_id
            and question_type in (variation.get("question_types") or [])
        ), None)
    return next((
        variation for variation in permitted_task_variations(task, question_type)
        if variation.get("variation_id") == variation_id
    ), None)


def _normalized_parameter(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("%", " ").split())


def catalog_numerical_contract_diagnostics(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Find deterministic cross-field contradictions without guessing semantics."""
    required = [str(value) for value in task.get("required_givens") or []]
    variables = [str(value) for value in task.get("variable_givens") or []]
    numeric_required = [value for value in required if any(ch.isdigit() for ch in value)]
    overlaps = [
        value for value in numeric_required
        if any(
            _normalized_parameter(variable)
            and _normalized_parameter(variable) in _normalized_parameter(value)
            for variable in variables
        )
    ]
    diagnostics: list[dict[str, Any]] = []
    if overlaps:
        diagnostics.append({
            "code": "required_variable_numeric_overlap",
            "fields": ["required_givens", "variable_givens"],
            "values": overlaps,
        })
    variations = _normalize_variations(task.get("allowed_variations"))
    numeric_freedom = any(value.get("numeric_values_may_change") for value in variations)
    if task.get("exact_source_instance") is True and (overlaps or numeric_freedom):
        diagnostics.append({
            "code": "exact_instance_numeric_variation_conflict",
            "fields": ["exact_source_instance", "variable_givens", "allowed_variations"],
        })
    return diagnostics


def _repair_catalog_numerical_contract(task: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Repair only the mechanically provable overlap; quarantine all other contradictions."""
    value = deepcopy(task)
    diagnostics = catalog_numerical_contract_diagnostics(value)
    overlap = next((item for item in diagnostics
                    if item["code"] == "required_variable_numeric_overlap"), None)
    if overlap:
        moved = list(overlap.get("values") or [])
        value["required_givens"] = [
            item for item in value.get("required_givens") or [] if str(item) not in set(moved)
        ]
        value["source_example_givens"] = list(dict.fromkeys([
            *(value.get("source_example_givens") or []), *moved,
        ]))
        # When every concrete numeric required given names an already-declared
        # variable parameter, exactness is contradicted by the task's own data.
        remaining_numeric = [
            item for item in value.get("required_givens") or []
            if any(ch.isdigit() for ch in str(item))
        ]
        if value.get("exact_source_instance") is True and not remaining_numeric:
            value["exact_source_instance"] = False
        diagnostics.append({
            "code": "safe_numeric_parameter_normalization",
            "disposition": "repaired",
            "moved_to_source_example_givens": moved,
        })
    # Exact-instance tasks cannot authorize changed numbers. Disabling that
    # permission is deterministic and preserves the exact source exercise.
    if value.get("exact_source_instance") is True:
        changed = False
        normalized_variations = []
        for variation in _normalize_variations(value.get("allowed_variations")):
            variation = deepcopy(variation)
            if variation.get("numeric_values_may_change"):
                variation["numeric_values_may_change"] = False
                changed = True
            normalized_variations.append(variation)
        if changed:
            value["allowed_variations"] = normalized_variations
            diagnostics.append({
                "code": "exact_instance_numeric_variation_disabled",
                "disposition": "repaired",
            })
    remaining = catalog_numerical_contract_diagnostics(value)
    return value, [*diagnostics, *(
        [{**item, "disposition": "quarantined"} for item in remaining]
        if remaining else []
    )]


def validate_plan_catalog_bindings(
    catalog: dict[str, Any], plans: list[dict[str, Any]],
    valid_chunk_ids: set[str] | None = None,
) -> list[str]:
    """Validate that every current plan identity belongs to this exact v12 Catalog."""
    errors: list[str] = []
    if catalog.get("catalog_version") != config.ASSESSMENT_CATALOG_VERSION:
        return ["Plan requires the current Assessment Catalog version."]
    tasks = {str(task.get("task_id") or ""): task for task in catalog.get("tasks") or []}
    fitb_ids = {
        str(entry.get("fitb_entry_id") or "") for entry in catalog.get("fitb_bank") or []
        if entry.get("fitb_entry_id") and not entry.get("alias_of_fitb_entry_id")
    }
    catalog_chunks = set(map(str, catalog.get("selected_chunk_ids") or []))
    if valid_chunk_ids is not None and catalog_chunks != set(valid_chunk_ids):
        errors.append("Assessment Catalog evidence IDs do not match the current retrieved evidence.")
    for plan in plans:
        for qtype, values in (plan.get("items") or {}).items():
            for item in values or []:
                sid = str(item.get("slot_id") or "unknown")
                if qtype == "fill_in_the_blank":
                    if str(item.get("fitb_entry_id") or "") not in fitb_ids:
                        errors.append(f"Slot {sid} references a FITB entry outside the current Catalog.")
                    continue
                base_id = str(item.get("base_catalog_task_id") or item.get("task_id") or "")
                task = tasks.get(base_id)
                if task is None:
                    errors.append(f"Slot {sid} references a task outside the current Catalog.")
                    continue
                variation_id = str(item.get("selected_variation_id") or "")
                group_id = str(item.get("selected_variation_group_id") or "")
                if not variation_id or not group_id or not item.get("permitted_variations"):
                    errors.append(f"Slot {sid} is missing its required typed variation binding.")
                    continue
                variation = selected_variation(task, variation_id, qtype)
                if variation is None:
                    errors.append(f"Slot {sid} references a variation outside its Catalog task.")
                    continue
                if str(variation.get("variation_group_id") or "") != group_id:
                    errors.append(f"Slot {sid} has an inconsistent variation-group identity.")
                bound = next((
                    candidate for candidate in item.get("permitted_variations") or []
                    if str(candidate.get("variation_id") or "") == variation_id
                ), None)
                if bound is None:
                    errors.append(f"Slot {sid} selected variation is absent from its bound variation set.")
                elif any(
                    bound.get(field) != variation.get(field)
                    for field in (
                        "variation_group_id", "question_types", "operation", "target",
                        "givens", "reasoning_mode", "student_work_product",
                        "evidence_chunk_ids",
                    )
                ):
                    errors.append(f"Slot {sid} bound variation differs from its Catalog contract.")
                operation = str(variation.get("operation") or task.get("task_operation") or "")
                target = str(variation.get("target") or task.get("target") or "")
                if str(item.get("task_operation") or "") != operation:
                    errors.append(f"Slot {sid} operation differs from its selected Catalog variation.")
                if str(item.get("target") or "") != target:
                    errors.append(f"Slot {sid} target differs from its selected Catalog variation.")
                evidence = {str(item.get("source_chunk_id") or ""), *map(
                    str, item.get("support_chunk_ids") or []
                )} - {""}
                allowed_evidence = set(map(str,
                    variation.get("evidence_chunk_ids")
                    if variation.get("evidence_chunk_ids") is not None
                    else task.get("evidence_chunk_ids") or []
                ))
                if not evidence or not evidence.issubset(allowed_evidence):
                    errors.append(f"Slot {sid} evidence differs from its selected Catalog variation.")
    return errors


_RELATIONSHIP_VARIATION_MODES = {
    "calculation",
    "symbolic_manipulation",
    "law_or_theorem_application",
    "derivation_relationship",
}
_VISUAL_VARIATION_RE = re.compile(
    r"\b(?:image|graph|table|diagram|figure|chart|visual)\b",
    re.IGNORECASE,
)


def _has_visual_dependency(text: str) -> bool:
    """Reuse the planner's multilingual visual cue detector when available."""
    try:
        from app.online.planner import _VISUAL_DEPENDENCY_RE
        return bool(_VISUAL_DEPENDENCY_RE.search(text))
    except (ImportError, AttributeError):
        return bool(_VISUAL_VARIATION_RE.search(text))


def _variation_idea(task: dict[str, Any], variation: dict[str, Any]) -> str:
    product = variation.get("student_work_product") or task.get("student_work_product")
    target = variation.get("target") or task.get("target")
    idea = f"{product}: {target}"
    mode = variation.get("reasoning_mode") or task.get("reasoning_mode")
    if mode == "derivation_relationship":
        idea = f"Derive and establish the relationship: {idea}"
    return idea


def _variation_is_self_contained(
    task: dict[str, Any], variation: dict[str, Any]
) -> bool:
    """Reject variations that would force the planner to invent missing data."""
    givens = list(variation.get("givens") or [])
    mode = variation.get("reasoning_mode") or task.get("reasoning_mode")
    if mode in _RELATIONSHIP_VARIATION_MODES and not (
        task.get("relationship") or givens
    ):
        return False
    if _has_visual_dependency(_variation_idea(task, variation)) and not givens:
        return False
    return True


def _effective_variation(
    task: dict[str, Any],
    question_type: str | None = None,
    selected_variation_id: str | None = None,
) -> dict[str, Any] | None:
    """Choose one self-contained variation, preferring the task's primary angle."""
    variations = (
        eligible_task_variations(task, question_type)
        if question_type else permitted_task_variations(task)
    )
    if not variations:
        return None
    if selected_variation_id:
        explicit = next((
            variation for variation in variations
            if variation.get("variation_id") == selected_variation_id
        ), None)
        if explicit is not None:
            return explicit
    return next((
        variation for variation in variations
        if _variation_is_self_contained(task, variation)
    ), variations[0])


def _apply_effective_variation(
    item: dict[str, Any], task: dict[str, Any], variation: dict[str, Any]
) -> None:
    """Bind the selected angle to the plan item without mutating the catalog task."""
    operation = variation.get("operation") or task.get("task_operation")
    target = variation.get("target") or task.get("target")
    product = variation.get("student_work_product") or task.get("student_work_product")
    mode = variation.get("reasoning_mode") or task.get("reasoning_mode")
    evidence = variation.get("evidence_chunk_ids")
    if evidence is None:
        if task.get("evidence_chunk_ids") is not None:
            evidence = task.get("evidence_chunk_ids") or []
        else:
            evidence = [
                task.get("source_chunk_id"),
                *(task.get("support_chunk_ids") or []),
            ]
    evidence = [str(value) for value in evidence if str(value)]
    givens = [
        str(value) for value in (variation.get("givens") or []) if str(value).strip()
    ]
    if not givens:
        givens = [
            str(value) for value in (task.get("required_givens") or [])
            if str(value).strip()
        ]
    idea = _variation_idea(task, variation)
    item.update({
        "source_chunk_id": evidence[0] if evidence else item.get("source_chunk_id"),
        "support_chunk_ids": list(dict.fromkeys(evidence[1:])),
        "canonical_skill": operation,
        "student_work_product": product,
        "target": target,
        "task_operation": operation,
        "given_conditions": list(dict.fromkeys(givens)),
        "scenario_family": (
            variation.get("application")
            if variation.get("application") is not None
            else task.get("scenario_family")
        ),
        "reasoning_mode": mode,
        "reasoning_requirements": [f"{operation}: {target}"],
        "idea_to_test": idea,
        "concept_to_test": idea,
        "selected_variation_id": variation.get("variation_id"),
        "selected_variation_group_id": variation.get("variation_group_id"),
        "effective_givens": list(dict.fromkeys(givens)),
    })


_METHOD_RESTRICTION_RE = re.compile(
    r"\b(?:must|required|only|without|rather\s+than|do\s+not|not)\b|"
    r"(?:يجب|فقط|دون|بدون|لا\s+تستخدم)", re.IGNORECASE,
)


def _migrate_legacy_method_restrictions(task: dict[str, Any]) -> list[str]:
    """Preserve only genuine method constraints from the removed legacy field."""
    boundaries = [
        str(value).strip() for value in task.get("complexity_boundaries") or []
        if str(value).strip()
    ]
    for value in task.get("reasoning_requirements") or []:
        text = str(value).strip()
        if text and _METHOD_RESTRICTION_RE.search(text) and text not in boundaries:
            boundaries.append(text)
    return boundaries


def _expand_compact_task(task: dict[str, Any]) -> dict[str, Any]:
    """Expand a compact compiler task into the full downstream catalog contract.

    Deterministic and idempotent: a full task from an earlier catalog, or one
    already expanded from a vocabulary code, passes through unchanged.
    """
    if not isinstance(task, dict):
        return task
    expanded = dict(task)
    expanded["complexity_boundaries"] = _migrate_legacy_method_restrictions(expanded)
    expanded["allowed_variations"] = _normalize_variations(
        expanded.get("allowed_variations") or expanded.get("variations")
    )
    # Removed Catalog fields must never survive as compatibility placeholders.
    for field in ("topic", "reasoning_depth", "reasoning_requirements"):
        expanded.pop(field, None)
    return expanded


def catalog_cache_key(
    processed_document_key: str,
    selected_chunk_ids: list[str],
    compiler_identity: dict[str, str] | None = None,
) -> str:
    finalizer_contract = "\n".join((
        inspect.getsource(canonical_task_ids),
        inspect.getsource(canonical_fitb_entry),
        CATALOG_CONTRACT_VERSION,
        inspect.getsource(_normalize_variations),
        inspect.getsource(_migrate_legacy_method_restrictions),
        inspect.getsource(_expand_compact_task),
        inspect.getsource(_finalize_catalog),
        inspect.getsource(_catalog_prompt),
        json.dumps(AssessmentCatalogResponse.model_json_schema(), sort_keys=True),
    ))
    payload = {
        "processed_document_key": processed_document_key,
        "selected_chunk_ids": sorted(set(map(str, selected_chunk_ids))),
        "catalog_version": config.ASSESSMENT_CATALOG_VERSION,
        # Deliberately exclude unrelated module text (logging, comments, imports).
        # Only behavior that can change a finalized catalog invalidates this cache.
        "catalog_contract_sha256": hashlib.sha256(
            finalizer_contract.encode("utf-8")
        ).hexdigest(),
        "compiler": compiler_identity or {},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compiler_cache_key(
    processed_document_key: str,
    selected_chunk_ids: list[str],
    compiler_identity: dict[str, str] | None = None,
) -> str:
    compiler_contract = "\n".join((
        CATALOG_CONTRACT_VERSION,
        inspect.getsource(_catalog_prompt),
        json.dumps(CompactCatalogBatchResponse.model_json_schema(), sort_keys=True),
        json.dumps(
            CompactCatalogBatchWithProfileResponse.model_json_schema(), sort_keys=True
        ),
    ))
    payload = {
        "processed_document_key": processed_document_key,
        "selected_chunk_ids": sorted(set(map(str, selected_chunk_ids))),
        "compiler_contract_sha256": hashlib.sha256(compiler_contract.encode("utf-8")).hexdigest(),
        "compiler": compiler_identity or {},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _path_for(key: str) -> Path:
    return Path(config.ASSESSMENT_CATALOG_CACHE_DIR) / key[:2] / f"{key}.json"


def _compiler_path(key: str) -> Path:
    return Path(config.ASSESSMENT_CATALOG_CACHE_DIR) / "compiler" / key[:2] / f"{key}.json"


def _read_cache(key: str) -> dict[str, Any] | None:
    path = _path_for(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        payload.get("cache_key") != key
        or payload.get("catalog_version") != config.ASSESSMENT_CATALOG_VERSION
        or not isinstance(payload.get("tasks"), list)
        or not isinstance(payload.get("fitb_bank"), list)
        or not all(_cached_task_contract_valid(task) for task in payload.get("tasks") or [])
    ):
        return None
    if not isinstance(payload.get("tasks_by_question_type"), dict):
        payload["tasks_by_question_type"] = {
            qtype: [
                str(task["task_id"]) for task in payload.get("tasks") or []
                if eligible_task_variations(task, qtype)
            ]
            for qtype in (
                "mcq", "true_false", "definition", "why", "equation",
                "word_problem", "essay",
            )
        }
        payload["tasks_by_question_type"]["fill_in_the_blank"] = [
            str(entry["fitb_entry_id"]) for entry in payload.get("fitb_bank") or []
            if entry.get("fitb_entry_id") and not entry.get("alias_of_fitb_entry_id")
        ]
    return payload


def _cached_catalog_matches_evidence(
    payload: dict[str, Any], processed_document_key: str, selected_chunk_ids: list[str],
) -> bool:
    """Fail closed when a cached catalog references another retrieval snapshot."""
    selected = set(map(str, selected_chunk_ids))
    if (
        payload.get("processed_document_key") != processed_document_key
        or set(map(str, payload.get("selected_chunk_ids") or [])) != selected
    ):
        return False
    for task in payload.get("tasks") or []:
        evidence = set(map(str, task.get("evidence_chunk_ids") or []))
        if not evidence or not evidence.issubset(selected):
            return False
        for variation in task.get("allowed_variations") or []:
            variation_evidence = variation.get("evidence_chunk_ids")
            if variation_evidence is not None and not set(
                map(str, variation_evidence)
            ).issubset(evidence):
                return False
    return all(
        str(entry.get("evidence_chunk_id") or "") in selected
        for entry in payload.get("fitb_bank") or []
    )


def _cached_task_contract_valid(task: Any) -> bool:
    """Reject caches outside the precise 18-field task contract."""
    if not isinstance(task, dict):
        return False
    native = str(task.get("native_difficulty") or "")
    prerequisites = task.get("prerequisites")
    return bool(
        native in {"easy", "medium", "hard"}
        and task.get("supported_difficulties") == [native]
        and "topic" not in task
        and "reasoning_depth" not in task
        and "reasoning_requirements" not in task
        and isinstance(task.get("text_answerable"), bool)
        and task.get("text_answerable") is True
        and _normalize_variations(task.get("allowed_variations"))
        == task.get("allowed_variations")
        and isinstance(prerequisites, list)
        and all(isinstance(value, str) and value.strip() for value in prerequisites)
        and not catalog_numerical_contract_diagnostics(task)
    )


def _write_cache(key: str, payload: dict[str, Any]) -> None:
    path = _path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_compiler_cache(key: str) -> dict[str, Any] | None:
    path = _compiler_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload.get("raw") if payload.get("compiler_cache_key") == key else None


def _write_compiler_cache(key: str, raw: dict[str, Any]) -> None:
    path = _compiler_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"compiler_cache_key": key, "raw": raw}
    handle, temporary_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _compatible_compiler_seed(
    processed_document_key: str, selected_chunk_ids: list[str]
) -> dict[str, Any] | None:
    """Return a same-version, same-evidence Catalog without an LLM call.

    Cache-key internals can change while the public Catalog contract remains
    compatible. Re-finalizing that payload under the current deterministic
    contract avoids needless recompilation. Incompatible contract changes must
    bump ``ASSESSMENT_CATALOG_VERSION``.
    """
    root = Path(config.ASSESSMENT_CATALOG_CACHE_DIR)
    selected = sorted(set(map(str, selected_chunk_ids)))
    for path in sorted(root.glob("[0-9a-f][0-9a-f]/*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        source_version = str(payload.get("catalog_version") or "")
        migratable_version = source_version in {
            config.ASSESSMENT_CATALOG_VERSION,
            "catalog-v12-grouped-type-variations",
        }
        migrated_tasks: list[dict[str, Any]] = []
        migration_failed = False
        for task in payload.get("tasks") or []:
            migrated, diagnostics = _repair_catalog_numerical_contract(task)
            if any(item.get("disposition") == "quarantined" for item in diagnostics):
                migration_failed = True
                break
            migrated["catalog_contract_version"] = config.ASSESSMENT_CATALOG_VERSION
            if diagnostics:
                migrated["contract_diagnostics"] = diagnostics
            migrated_tasks.append(migrated)
        if (
            migratable_version
            and payload.get("processed_document_key") == processed_document_key
            and sorted(set(map(str, payload.get("selected_chunk_ids") or []))) == selected
            and payload.get("assessment_profile")
            and payload["assessment_profile"].get("subject")
            and payload["assessment_profile"].get("domain") in {"STEM", "NON_STEM"}
            and bool(payload.get("tasks"))
            and isinstance(payload.get("fitb_bank"), list)
            and not migration_failed
            and all(_cached_task_contract_valid(task) for task in migrated_tasks)
        ):
            return {
                "assessment_profile": payload["assessment_profile"],
                "tasks": migrated_tasks,
                "fitb_bank": payload["fitb_bank"],
                "catalog_compilation_complete": True,
                "recovered_batch_count": int(payload.get("recovered_batch_count") or 0),
            }
    return None


def _catalog_prompt(
    context: str,
    language: str,
    valid_chunk_ids: list[str],
    *,
    domain_type: str,
    subject: str,
    include_profile: bool = True,
) -> str:
    if include_profile:
        output_contract = (
            "Return one root JSON object with exactly these top-level fields:\n"
            "- assessment_profile: the complete profile required by the schema;\n"
            "- tasks: the array of educational-objective task-family objects.\n"
            "- fitb_bank: the compact array of answer/evidence/clue objects.\n"
            "- batch_complete: true, written only after both arrays are complete.\n"
            "Never return either field outside that root object and never return a bare task array."
        )
    else:
        output_contract = (
            "Return one root JSON object with exactly these top-level fields:\n"
            "- tasks: the array of educational-objective task-family objects.\n"
            "- fitb_bank: the compact array of answer/evidence/clue objects.\n"
            "- batch_complete: true, written only after both arrays are complete.\n"
            "Do not return an assessment_profile; it is compiled once from the first batch.\n"
            "Never return either field outside that root object and never return a bare task array."
        )
    return f"""You are the one-time assessment compiler for selected textbook content.
Read the complete selection once and create a reusable semantic assessment catalog.

Title generation already classified this source. Copy these authoritative values exactly into the
assessment profile; do not reclassify or override them:
- assessment_profile.subject: {subject}
- assessment_profile.domain: {"STEM" if domain_type == "stem" else "NON_STEM"}

The catalog is an ontology, not an exam. Follow this opportunity-first workflow internally; return
only the strict Catalog JSON and never output an opportunity inventory or analysis transcript.

PHASE 1 — DISCOVER EDUCATIONAL OPPORTUNITIES
Step 1 — Read every source subsection. Systematically examine both parent sections and every selected
child chunk. Explicitly examine every selected child chunk. Inspect concepts, definitions, terminology, formulas, relationships, methods,
explanations, reasoning, worked examples, exercises and numbered subquestions, applications, tables,
graphs, diagrams, interpretations, comparisons, and classifications. One child may contain several
unrelated objectives. Do not stop after its first valid objective, and do not consider a chunk fully
covered merely because one task cites it.

Step 2 — Identify all meaningful assessment opportunities before constructing task objects. For each
source-supported opportunity, determine the objective or skill, student operation, target or expected
result, required reasoning, supporting chunk, and whether it is independent or another angle of an
existing objective. Treat each independent exercise, subquestion, application, or explanation as a
candidate during discovery. Do not immediately make one task per exercise, discard a similar-looking
opportunity, or treat different numerical instances as automatically distinct tasks or variations.

PHASE 2 — CONSTRUCT THE ASSESSMENT CATALOG
Step 3 — Group the discovered opportunities into coherent task families. A task represents one
coherent educational objective or task family, not one narrow question target. Group targets, operations, reasoning directions,
representations, and applications only when they genuinely assess that same objective, using
variations for meaningful alternative angles. Keep independently taught methods, skills, and objectives
separate. A shared chapter, section, chunk, formula, wording pattern, or final answer does not justify
merging. The goal is meaningful organization, not minimizing task IDs.

Step 4 — Generate complete Catalog metadata. Use the task's top-level target, operation, work_product,
and mode for one primary assessment angle and put every other meaningful, nonredundant angle under
variations. Preserve compatible question types, evidence, optional variation evidence and inheritance,
difficulty, boundaries, prerequisites, source restrictions, and explicit answerability. Do not create
question-type copies, superficial paraphrases, number-only copies, invented opportunities, filler, or
variations that repeat the primary angle. One evidence chunk may support multiple task families or
several variations.
Every task_operation used by a task must also appear under every corresponding question type in
assessment_profile.question_types.task_operations.
For every task, determine difficulty from the assessed concept, narrow target, student operation,
work product, prerequisites, and cognitive demand supported by the assigned evidence,
and the cognitive demand of that work. Assign one native cognitive-difficulty classification.

Use these unified classifications:
- easy: direct recall, recognition, straightforward substitution, or a simple operation with an
  immediately identifiable method;
- medium: selecting or organizing a method, connecting dependent operations, interpreting an
  intermediate result, or applying relationships through multiple cognitive decisions;
- hard: integrating relationships or constraints, evaluating alternatives, deriving a non-obvious
  result, analyzing errors or representations, or justifying a complex solution.

Record that single grounded cognitive-difficulty classification in difficulty. mode describes the
kind of reasoning and does not determine its level. Prereqs contain only knowledge supported by the
assigned evidence. A mandatory source-specific solution method belongs in operation or boundaries;
do not emit generic solution steps such as substitute values or apply the formula. Never infer difficulty
from wording length, numerical size, visible step count, reasoning_mode, or keyword cues such as
derive, compare, analyze, or justify.

Choose allowed_question_types as the union of types genuinely supported by the primary angle and
all variations; this field is authoritative for downstream allocation. Also give each variation its
own question_types. Do not decide suitability from the operation label or reasoning_mode alone.
Use the concept, target, relationship, evidence, and required givens together. A type is suitable
only when the task naturally supports its student-facing work:
- mcq: one defensible best answer plus three plausible same-concept distractors;
- true_false: one clear, source-grounded, unambiguous claim;
- definition: the meaning of a term or concept;
- why: cause, reason, justification, mechanism, or relationship reasoning;
- equation: symbolic manipulation, derivation, balancing, solving, or other equation-centred work
  in which the student works directly with a mathematical or
  discipline-specific formal representation: an equation, system, expression, inequality,
  scientific relationship, chemical reaction, or equivalent notation. A short instruction,
  necessary givens/units, multiple expressions, and a supporting table are allowed when they
  directly serve that symbolic operation. Do not assign equation to a prose-centred scenario,
  extended explanation, or task whose symbolic content appears only in the answer;
- word_problem: a grounded verbal quantitative scenario with supported givens and a numerical result;
- essay: bounded multi-point synthesis, explanation, comparison, or argument.
FITB is supplied only by the separate fitb_bank below. Never put fill_in_the_blank in an atomic
task's allowed_question_types. An identify_term_from_definition or identify_named_phenomenon task
may support MCQ when plausible same-concept distractors exist. A state_definition task should
normally support Definition and may support True/False when it yields one unambiguous claim. Do not
add a type merely because it can mention the same evidence, and do not remove a type merely because
its operation name is unfamiliar. Preserve the semantic decision in allowed_question_types.

Create a separate compact `fitb_bank`. FITB entries are vocabulary evidence, not assessment tasks,
and contain exactly `answer`, `evidence_chunk_id`, and `clue_basis`. Extract every distinct
source-supported word, term, name, concept, or short phrase recoverable from one clear textual clue.
One chunk may support several different answers. Do not merge entries merely because they share a
chunk; equivalent normalized answers are aliases. Never include a number, formula, equation,
calculated value, unit-bearing answer, mathematical notation, chemical symbolic expression, or
calculation. `clue_basis` is concise pure text describing what the future cloze may test without
giving away the answer.

Cover the selection broadly before adding depth to any concept. Do not target a fixed task count or
variation count, minimum, quota, or maximum. Return every source-supported educational objective and
all of its meaningful assessment opportunities; never remove an opportunity to save tokens and never
pad with filler. A task may validly have zero, one, or many alternative variations.
Do not satisfy symbolic coverage by forcing every Equation task into rearranging a supplied final
relationship. Preserve the evidence-supported operation: constructing, deriving, combining,
balancing, solving, proving, and rearranging are distinct only when the source supports that work.
Do not create several superficial rearrangements merely to meet coverage.

Worked examples demonstrate methods, relationships, constraints, and supported kinds of givens, and
may also contain independently assessable applications, targets, explanations, or subquestions.
Preserve those opportunities when supported; do not automatically reduce every worked example to
evidence for a broader method. Unless the exact instance is itself the assessment target, express its
transferable operation and target instead of encoding the complete scenario + exact values + requested result.
For a general application task, express the transferable operation and target, and include only
source-required constants or constraints in required_givens. Never invent an unsupported scientific
fact or context.
For every task, use the variation fields as follows:
- required_givens contains only invariant laws, scientific constants, factual truths, necessary
  constraints, or source-specific values that every valid instance must preserve;
- variable_givens describes illustrative quantities or parameters that a future question may choose
  and change while remaining valid (for example total matches, distance, speed, or elapsed time);
- variations is an array of bounded objects. Each object names one exact source-supported target,
  operation, givens, and exactly one question_types value; optionally use reasoning_mode, student_work_product, form for
  representation, context for application, and evidence_chunk_ids for a narrower evidence subset.
  Omit evidence_chunk_ids to inherit the task evidence. Never repeat the primary angle merely to fill
  the array. Set vary_numbers true only when the named givens are illustrative parameters and changing
  them preserves the relationship and constraints. Never emit broad labels such as different_target,
  different_operation, or different_application. Emit one type-specific variation for every declared
  allowed_question_types value. Variations that assess the same underlying concept, operation, and
  result share one variation_group_id; different assessment angles use different groups. Set both
  variation_group_id and variation_id to null; deterministic code assigns stable IDs after validation.
  Never invent an angle or context absent from the source. If evidence cannot support a declared type,
  remove that type rather than fabricating coverage;
- complexity_boundaries lists concepts, methods, or difficulty increases that the selected evidence
  does not authorize. Use short source-specific restrictions, not generic permission codes;
- exact_source_instance is true only when assessing the exact worked example is itself the task.

PHASE 3 — FINAL COVERAGE REVIEW
Step 5 — Recheck coverage and grounding in both directions before returning. Source → Catalog: revisit
every selected child, including exercises, subquestions, worked examples, explanations, and
applications. Ensure every meaningful discovered opportunity appears as a primary angle or variation;
add an overlooked valid angle to the appropriate family or create a separate task when justified.
A cited chunk may still contain unrelated uncovered opportunities. Catalog → Source: ensure every task
and variation is supported by its cited evidence and that no operation, target, numerical freedom, or
reasoning requirement exceeds the source. Return only the existing strict v11 Catalog JSON.
Ordinary textbook exercise numbers belong in variable_givens, not required_givens. A formula's
structure, a physical constant, an integrality/non-negativity constraint, or a factual source value
may remain required. Do not authorize a variation merely because it is generally true in the subject;
it must be supported by the selected evidence and educational scope.
Set answerable explicitly for every task. False tasks are filtered and must never be presented as
usable merely because related evidence exists. Every task, variation, and FITB evidence ID must be
copied exactly from the Valid source chunk IDs supplied for this compiler batch. Never cite a chunk
from another batch, even if it belongs to the same document. Every required numerical/symbolic given must
be present in those chunks. Equation tasks must ask for symbolic manipulation/derivation; word
problems must ask for a contextual numerical solution. Use information-dense values: short noun
phrases, compact formulas, canonical operation/mode identifiers, atomic givens/prereqs, and precise
boundaries. Omit filler such as "students should be able to". Preserve Arabic source-specific terms
when they carry meaning. A batch may legitimately return empty tasks and fitb_bank arrays.

{output_contract}

Document language: {language}
Valid source chunk IDs: {json.dumps(valid_chunk_ids, ensure_ascii=False)}

Complete selected source:
{context}"""


def _finalize_catalog(
    raw: dict[str, Any], *, key: str, processed_document_key: str,
    selected_chunk_ids: list[str], usage: dict[str, Any] | None,
) -> dict[str, Any]:
    valid_ids = set(map(str, selected_chunk_ids))
    finalized: list[dict[str, Any]] = []
    finalized_by_id: dict[str, dict[str, Any]] = {}
    for raw_task in raw.get("tasks") or []:
        if not isinstance(raw_task, dict):
            continue
        raw_task = _expand_compact_task(raw_task)
        raw_task, contract_diagnostics = _repair_catalog_numerical_contract(raw_task)
        if any(item.get("disposition") == "quarantined" for item in contract_diagnostics):
            continue
        if contract_diagnostics:
            raw_task["contract_diagnostics"] = contract_diagnostics
        evidence = [
            str(value) for value in raw_task.get("evidence_chunk_ids") or []
            if str(value) in valid_ids
        ]
        if not evidence or raw_task.get("text_answerable") is not True:
            continue
        parent_evidence = set(evidence)
        valid_variations: list[dict[str, Any]] = []
        for variation in _normalize_variations(raw_task.get("allowed_variations")):
            explicit = variation.get("evidence_chunk_ids")
            if explicit is not None:
                explicit_ids = {str(value) for value in explicit}
                # One bad optional angle must not discard an otherwise complete task.
                if not explicit_ids or not explicit_ids.issubset(parent_evidence & valid_ids):
                    continue
                variation["evidence_chunk_ids"] = list(dict.fromkeys(map(str, explicit)))
            valid_variations.append(variation)
        raw_task["allowed_variations"] = valid_variations
        identity = canonical_task_ids(raw_task)
        if identity is None:
            continue
        task_id = str(identity["task_id"])
        if task_id in finalized_by_id:
            existing = finalized_by_id[task_id]
            existing["evidence_chunk_ids"] = list(dict.fromkeys([
                *(existing.get("evidence_chunk_ids") or []), *evidence,
            ]))
            existing["allowed_question_types"] = list(dict.fromkeys([
                *(existing.get("allowed_question_types") or []),
                *(raw_task.get("allowed_question_types") or []),
                *(qtype for variation in valid_variations
                  for qtype in variation.get("question_types") or []),
            ]))
            for field in ("variable_givens", "complexity_boundaries"):
                existing[field] = list(dict.fromkeys([
                    *(existing.get(field) or []), *(raw_task.get(field) or []),
                ]))
            incoming_primaries = _primary_variations({
                **raw_task,
                "evidence_chunk_ids": evidence,
                "primary_question_types": list(raw_task.get("allowed_question_types") or []),
            })
            existing_primary_ids = {
                value.get("variation_id") for value in _primary_variations(existing)
            }
            existing["allowed_variations"] = _normalize_variations([
                *(existing.get("allowed_variations") or []),
                *(value for value in incoming_primaries
                  if value.get("variation_id") not in existing_primary_ids),
                *(raw_task.get("allowed_variations") or []),
            ])
            existing["exact_source_instance"] = bool(
                existing.get("exact_source_instance")
                and raw_task.get("exact_source_instance")
            )
            existing["allowed_question_types"] = [
                qtype for qtype in existing["allowed_question_types"]
                if eligible_task_variations(existing, qtype)
            ]
            existing["primary_question_types"] = [
                qtype for qtype in (existing.get("primary_question_types") or [])
                if qtype in existing["allowed_question_types"]
            ]
            existing["allowed_variations"] = _scope_variations(
                existing, existing.get("allowed_variations") or []
            )
            continue
        reasoning_mode = str(raw_task.get("reasoning_mode") or "")
        allowed = list(dict.fromkeys(raw_task.get("allowed_question_types") or []))
        primary_allowed = list(allowed)
        allowed = list(dict.fromkeys([
            *allowed,
            *(qtype for variation in valid_variations
              for qtype in variation.get("question_types") or []),
        ]))
        allowed = [
            qtype for qtype in allowed
            if qtype in {"mcq", "true_false", "definition",
                         "why", "equation", "word_problem", "essay"}
        ]
        if not allowed:
            # A task without a compiler-declared compatible representation is structurally
            # unusable; do not invent a semantic compatibility decision here.
            continue
        native_difficulty = str(raw_task.get("native_difficulty") or "medium")
        if native_difficulty not in {"easy", "medium", "hard"}:
            native_difficulty = "medium"
        supported_difficulties = [native_difficulty]
        exact_source_instance = bool(raw_task.get("exact_source_instance", False))
        finalized_task = {
            **raw_task,
            **identity,
            "evidence_chunk_ids": list(dict.fromkeys(evidence)),
            "allowed_question_types": allowed,
            "primary_question_types": primary_allowed,
            "supported_difficulties": supported_difficulties,
            "native_difficulty": native_difficulty,
            "prerequisites": list(raw_task.get("prerequisites") or []),
            "allowed_variations": (
                [] if exact_source_instance
                else valid_variations
            ),
            "variable_givens": list(raw_task.get("variable_givens") or []),
            "complexity_boundaries": list(raw_task.get("complexity_boundaries") or []),
            "exact_source_instance": exact_source_instance,
            "catalog_contract_version": config.ASSESSMENT_CATALOG_VERSION,
            "text_answerable": True,
        }
        finalized_task["allowed_variations"] = _scope_variations(
            finalized_task, finalized_task["allowed_variations"]
        )
        # The compiler may over-declare a representation. Keep only types with
        # a complete, source-grounded type-specific variation.
        finalized_variations = permitted_task_variations(finalized_task)
        finalized_task["allowed_question_types"] = [
            qtype for qtype in finalized_task["allowed_question_types"]
            if eligible_task_variations(
                finalized_task, qtype, variations=finalized_variations,
            )
        ]
        finalized_task["primary_question_types"] = [
            qtype for qtype in primary_allowed
            if qtype in finalized_task["allowed_question_types"]
        ]
        if not finalized_task["allowed_question_types"]:
            continue
        finalized.append(finalized_task)
        finalized_by_id[task_id] = finalized_task
    fitb_bank: list[dict[str, Any]] = []
    canonical_by_answer: dict[str, str] = {}
    seen_fitb_ids: set[str] = set()
    for raw_entry in raw.get("fitb_bank") or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = canonical_fitb_entry(raw_entry, valid_ids)
        if entry is None or entry["fitb_entry_id"] in seen_fitb_ids:
            continue
        seen_fitb_ids.add(str(entry["fitb_entry_id"]))
        normalized_answer = str(entry["normalized_answer"])
        canonical_id = canonical_by_answer.get(normalized_answer)
        if canonical_id is None:
            canonical_by_answer[normalized_answer] = str(entry["fitb_entry_id"])
        else:
            entry["alias_of_fitb_entry_id"] = canonical_id
            entry["duplicate_normalized_answer"] = True
        fitb_bank.append(entry)

    profile = raw.get("assessment_profile") or {}
    type_profiles = profile.get("question_types") if isinstance(profile, dict) else None
    if isinstance(type_profiles, dict):
        for qtype, type_profile in type_profiles.items():
            if not isinstance(type_profile, dict):
                continue
            supporting = [task for task in finalized if eligible_task_variations(task, qtype)]
            if qtype == "fill_in_the_blank" and fitb_bank:
                type_profile["support"] = "supported"
                type_profile["task_operations"] = ["identify_term_from_text_clue"]
                type_profile["evidence_chunk_ids"] = list(dict.fromkeys(
                    entry["evidence_chunk_id"] for entry in fitb_bank
                    if not entry.get("alias_of_fitb_entry_id")
                ))[:8]
                continue
            if supporting:
                type_profile["support"] = "supported"
                type_profile["task_operations"] = list(dict.fromkeys(
                    str(variation.get("operation") or task.get("task_operation"))
                    for task in supporting
                    for variation in eligible_task_variations(task, qtype)
                ))
                type_profile["evidence_chunk_ids"] = list(dict.fromkeys(
                    chunk_id for task in supporting for chunk_id in task["evidence_chunk_ids"]
                ))[:8]
            else:
                type_profile["support"] = "unsupported"
                type_profile["task_operations"] = []
                type_profile["evidence_chunk_ids"] = []
    family_counts: dict[str, int] = {}
    for task in finalized:
        family_id = str(task.get("task_family_id") or "")
        if family_id:
            family_counts[family_id] = family_counts.get(family_id, 0) + 1
    for task in finalized:
        family_id = str(task.get("task_family_id") or "")
        if family_id and family_counts.get(family_id, 0) > 1:
            task["semantic_overlap_group_id"] = family_id
    tasks_by_question_type = {
        qtype: [
            str(task["task_id"]) for task in finalized
            if eligible_task_variations(task, qtype)
        ]
        for qtype in (
            "mcq", "true_false", "definition", "why", "equation",
            "word_problem", "essay",
        )
    }
    tasks_by_question_type["fill_in_the_blank"] = [
        str(entry["fitb_entry_id"]) for entry in fitb_bank
        if not entry.get("alias_of_fitb_entry_id")
    ]
    return {
        "cache_key": key,
        "catalog_version": config.ASSESSMENT_CATALOG_VERSION,
        "processed_document_key": processed_document_key,
        "selected_chunk_ids": sorted(valid_ids),
        "assessment_profile": profile,
        "tasks": finalized,
        "fitb_bank": fitb_bank,
        "tasks_by_question_type": tasks_by_question_type,
        "compiler_usage": usage or {},
        "catalog_compilation_complete": raw.get("catalog_compilation_complete", True),
        "recovered_batch_count": int(raw.get("recovered_batch_count") or 0),
    }


def build_task_usage_index(
    plans: list[dict[str, Any]], generated_exams: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a current usage view from authoritative plan and question state."""
    usage: dict[str, dict[str, Any]] = {}
    slot_task: dict[str, str] = {}
    for plan in plans or []:
        for values in (plan.get("items") or {}).values():
            for item in values or []:
                task_id = str(item.get("fitb_entry_id") or item.get("task_id") or "")
                slot_id = str(item.get("slot_id") or "")
                if not task_id:
                    continue
                row = usage.setdefault(
                    task_id, {"count": 0, "slot_ids": [], "question_ids": []}
                )
                row["count"] += 1
                if slot_id and slot_id not in row["slot_ids"]:
                    row["slot_ids"].append(slot_id)
                    slot_task[slot_id] = task_id
    for exam in generated_exams or []:
        for section in (exam.get("questions") or {}).values():
            items = section.get("items") if isinstance(section, dict) else section
            for item in items or []:
                slot_id = str(item.get("slot_id") or "")
                task_id = slot_task.get(slot_id)
                question_id = str(item.get("question_id") or "")
                if task_id and question_id and question_id not in usage[task_id]["question_ids"]:
                    usage[task_id]["question_ids"].append(question_id)
    for row in usage.values():
        row["slot_ids"].sort()
        row["question_ids"].sort()
    return usage


def _balanced_catalog_batches(
    chunks: list[dict[str, Any]], limit: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Return ordered batches containing two complete selected parent groups.

    ``limit`` remains accepted for callers from the previous batching contract;
    parent pairs are intentionally independent of child count or text size.
    Chunks without stored parent metadata are conservatively treated as their
    own parent so unrelated source records are never merged into one section.
    """
    def source_order(indexed: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        index, chunk = indexed
        child_order = chunk.get("child_order")
        try:
            numeric_order = int(child_order)
        except (TypeError, ValueError):
            numeric_order = None
        return (
            0 if numeric_order is not None else 1,
            numeric_order if numeric_order is not None else index,
            str(chunk.get("parent_id") or ""),
            str(chunk.get("child_id") or ""),
        )

    # Retrieval preserves caller selection order, which differs between the UI,
    # resumed jobs, and workers. Canonicalize before deriving batches and the
    # cache identity so the same processed selection cannot trigger recompilation.
    ordered_chunks = [
        chunk for _index, chunk in sorted(enumerate(chunks), key=source_order)
    ]
    parent_groups: dict[str, list[dict[str, Any]]] = {}
    parent_order: list[str] = []
    for index, chunk in enumerate(ordered_chunks):
        parent_id = str(chunk.get("parent_id") or "").strip()
        parent_key = parent_id or f"__child_parent__:{chunk.get('child_id') or index}"
        if parent_key not in parent_groups:
            parent_groups[parent_key] = []
            parent_order.append(parent_key)
        parent_groups[parent_key].append(chunk)
    return [
        [
            chunk
            for parent_key in parent_order[offset:offset + 2]
            for chunk in parent_groups[parent_key]
        ]
        for offset in range(0, len(parent_order), 2)
    ]


def _catalog_source_membership(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe ordered selected parent/child membership for cache identity."""
    batches = _balanced_catalog_batches(chunks)
    groups: list[dict[str, Any]] = []
    for batch in batches:
        for chunk in batch:
            parent_id = str(chunk.get("parent_id") or "").strip()
            fallback = str(chunk.get("child_id") or "")
            key = parent_id or f"__child_parent__:{fallback}"
            if not groups or groups[-1]["parent_key"] != key:
                groups.append({"parent_key": key, "child_ids": []})
            groups[-1]["child_ids"].append(fallback)
    return groups


def _merge_catalog_batches(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    task_lists = [list(result.get("tasks") or []) for result in results]
    fitb_lists = [list(result.get("fitb_bank") or []) for result in results]
    tasks: list[dict[str, Any]] = []
    fitb_bank: list[dict[str, Any]] = []
    for index in range(max((len(values) for values in task_lists), default=0)):
        tasks.extend(deepcopy(values[index]) for values in task_lists if index < len(values))
    for index in range(max((len(values) for values in fitb_lists), default=0)):
        fitb_bank.extend(deepcopy(values[index]) for values in fitb_lists if index < len(values))
    return {
        "assessment_profile": deepcopy(results[0].get("assessment_profile") or {}),
        "tasks": tasks,
        "fitb_bank": fitb_bank,
        "catalog_compilation_complete": all(
            result.get("batch_complete") is True
            and not result.get("recovered_truncated_checkpoint")
            for result in results
        ),
        "recovered_batch_count": sum(
            int(bool(result.get("recovered_truncated_checkpoint"))) for result in results
        ),
    }


def prepare_assessment_catalog(state: dict[str, Any]) -> dict[str, Any]:
    # Legacy synchronous callers have no content/version fingerprint. Falling
    # back to the old planner is safer than creating a cache keyed only by the
    # mutable public document ID. API jobs always provide the processed key.
    if not state.get("processed_document_key"):
        return {
            "assessment_catalog": {},
            "assessment_catalog_key": "",
            "assessment_catalog_cache_hit": False,
            "assessment_compiler_cache_hit": False,
        }
    chunks = state.get("retrieved_chunks") or []
    domain_type = str(state.get("domain_type") or "").strip().lower()
    subject = str(state.get("subject") or "").strip()
    try:
        compiler_identity = {
            **assessment_compiler_identity(domain_type=domain_type),
            "domain_type": domain_type,
            "subject": subject,
        }
    except ValueError as exc:
        return {
            "assessment_catalog": {}, "assessment_catalog_key": "",
            "assessment_catalog_cache_hit": False,
            "assessment_compiler_cache_hit": False,
            "plan_errors": [str(exc)],
        }
    if not re.fullmatch(r"[a-z][a-z0-9_]*", subject):
        return {
            "assessment_catalog": {}, "assessment_catalog_key": "",
            "assessment_catalog_cache_hit": False,
            "assessment_compiler_cache_hit": False,
            "plan_errors": [
                "Stored title metadata subject must be a lowercase normalized identifier."
            ],
        }
    selected = [str(chunk.get("child_id")) for chunk in chunks if chunk.get("child_id")]
    membership = _catalog_source_membership(chunks)
    compiler_identity = {
        **compiler_identity,
        "batching": CATALOG_BATCHING_VERSION,
        "source_membership_sha256": hashlib.sha256(
            json.dumps(membership, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    processed_key = str(state.get("processed_document_key") or state.get("document_id") or "")
    key = catalog_cache_key(processed_key, selected, compiler_identity)
    cached = _read_cache(key)
    if cached is not None and not _cached_catalog_matches_evidence(
        cached, processed_key, selected
    ):
        logger.warning(
            "Assessment catalog cache rejected for retrieval mismatch | key=%s", key
        )
        cached = None
    if cached is not None:
        logger.info("Assessment catalog cache hit | key=%s | tasks=%d", key, len(cached["tasks"]))
        return {
            "assessment_catalog": cached,
            "assessment_catalog_key": key,
            "assessment_catalog_cache_hit": True,
            "assessment_compiler_cache_hit": True,
            "planner_context": "",
        }
    if not chunks:
        return {
            "assessment_catalog": {}, "assessment_catalog_key": key,
            "assessment_catalog_cache_hit": False,
            "assessment_compiler_cache_hit": False,
            "plan_errors": ["Assessment catalog cannot be built without selected source chunks."],
        }
    compiler_key = compiler_cache_key(processed_key, selected, compiler_identity)
    raw = _read_compiler_cache(compiler_key)
    if raw is not None and raw.get("catalog_compilation_complete") is not True:
        # Leave the file untouched for diagnostics. It is not a reusable result.
        raw = None
    compiler_cache_hit = raw is not None
    compatibility_cache_hit = False
    if raw is None:
        raw = _compatible_compiler_seed(processed_key, selected)
        if raw is not None:
            compatibility_cache_hit = True
            compiler_cache_hit = True
            logger.info(
                "Assessment catalog compatible cache migrated | key=%s", key
            )
    client = None
    compiler_usage: dict[str, Any] = (
        {"source": "compatible_finalized_catalog", "provider_calls": 0}
        if compatibility_cache_hit else {}
    )
    if raw is None:
        from app.online.retrieval import build_planner_context

        batches = _balanced_catalog_batches(chunks)

        def compile_batch(batch_index: int, batch: list[dict[str, Any]]):
            batch_started = time.perf_counter()
            # The catalog profile is catalog-level, but only the first batch
            # compiles it; every later batch omits it to avoid regenerating the
            # same ~3.4 kB profile across all twelve calls.
            include_profile = batch_index == 0
            response_model = (
                CompactCatalogBatchWithProfileResponse if include_profile
                else CompactCatalogBatchResponse
            )
            batch_ids = [
                str(chunk.get("child_id")) for chunk in batch if chunk.get("child_id")
            ]
            batch_key = compiler_cache_key(processed_key, batch_ids, {
                **compiler_identity, "batch_index": batch_index,
                "batch_count": len(batches), "partition": CATALOG_BATCHING_VERSION,
            })
            cached_batch = _read_compiler_cache(batch_key)
            if cached_batch is not None:
                try:
                    validated_cache = _validate_completed_batch(
                        cached_batch, response_model=response_model,
                        valid_chunk_ids=set(batch_ids),
                    )
                except (ValidationError, ValueError, TypeError):
                    validated_cache = None
                if validated_cache is not None:
                    return batch_index, validated_cache, {
                        "compiler_duration_ms": round(
                            (time.perf_counter() - batch_started) * 1000, 3
                        )
                    }, True
            batch_client = create_assessment_compiler_client(domain_type=domain_type)
            # This is the provider-neutral outer recovery budget for an
            # incomplete Catalog batch.  Provider clients retain their own
            # transport/structured-output retry policies.
            max_attempts = max(1, int(config.CATALOG_BATCH_MAX_ATTEMPTS))
            last_reason = "no response"
            for attempt in range(1, max_attempts + 1):
                is_cancelled = state.get("is_cancelled")
                if callable(is_cancelled) and is_cancelled():
                    raise CatalogCompilationCancelled(
                        f"Assessment Catalog compilation cancelled before retrying batch "
                        f"{batch_index + 1}"
                    )
                try:
                    result = request_structured(
                        batch_client,
                        _catalog_prompt(
                            build_planner_context(batch),
                            state.get("document_language") or "en", batch_ids,
                            domain_type=domain_type, subject=subject,
                            include_profile=include_profile,
                        ),
                        response_model=response_model,
                        schema_name="assessment_catalog_batch",
                        agent="assessment_compiler",
                        operation="assessment_catalog_batch",
                        item_count=max(1, len(batch)),
                        system_prompt=(
                            "Compile a complete, non-redundant assessment ontology "
                            "grounded only in this evidence batch."
                        ),
                        temperature=0.1, max_tokens=18000,
                    )
                    value = _validate_completed_batch(
                        result, response_model=response_model,
                        valid_chunk_ids=set(batch_ids),
                    )
                except (DeepSeekIncompleteError, DeepSeekValidationError) as exc:
                    checkpoint = _recover_catalog_checkpoint(
                        exc.partial_output, require_profile=include_profile
                    )
                    recovered_count = len((checkpoint or {}).get("tasks") or [])
                    last_reason = (
                        f"{type(exc).__name__}; recovered {recovered_count} complete tasks "
                        "from a truncated checkpoint"
                    )
                except (ValidationError, ValueError, TypeError) as exc:
                    last_reason = f"invalid completion contract: {exc}"
                else:
                    _write_compiler_cache(batch_key, value)
                    usage = dict(getattr(batch_client, "last_usage", None) or {})
                    usage["catalog_batch_attempts"] = attempt
                    usage["compiler_duration_ms"] = round(
                        (time.perf_counter() - batch_started) * 1000, 3
                    )
                    return batch_index, value, usage, False
                if callable(is_cancelled) and is_cancelled():
                    raise CatalogCompilationCancelled(
                        f"Assessment Catalog compilation cancelled after incomplete batch "
                        f"{batch_index + 1}"
                    )
                logger.warning(
                    "Retrying incomplete assessment catalog batch | batch=%d | attempt=%d/%d | reason=%s",
                    batch_index + 1, attempt, max_attempts, last_reason,
                )
            raise CatalogBatchIncompleteError(batch_index, max_attempts, last_reason)

        completed: dict[int, tuple[dict[str, Any], dict[str, Any], bool]] = {}
        workers = min(config.MAX_PARALLEL_CATALOG_CALLS, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(copy_context().run, compile_batch, index, batch): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                index, value, usage, cache_hit = future.result()
                completed[index] = (value, usage, cache_hit)

        raw = _merge_catalog_batches([
            completed[index][0] for index in range(len(batches))
        ])
        if raw.get("catalog_compilation_complete") is not True:
            incomplete = [
                str(index + 1) for index in range(len(batches))
                if completed[index][0].get("batch_complete") is not True
            ]
            raise CatalogBatchIncompleteError(
                int(incomplete[0]) - 1 if incomplete else 0,
                int(config.CATALOG_BATCH_MAX_ATTEMPTS),
                f"merged catalog contains incomplete batches: {', '.join(incomplete) or 'unknown'}",
            )
        compiler_usage = {
            "batch_count": len(batches),
            "batch_cache_hits": sum(
                int(completed[index][2]) for index in range(len(batches))
            ),
            "batches": [completed[index][1] for index in range(len(batches))],
        }
        if isinstance(raw, dict) and raw.get("catalog_compilation_complete") is True:
            _write_compiler_cache(compiler_key, raw)
    if isinstance(raw, dict):
        profile = raw.setdefault("assessment_profile", {})
        if isinstance(profile, dict):
            profile["subject"] = subject
            profile["domain"] = "STEM" if domain_type == "stem" else "NON_STEM"
    if not isinstance(raw, dict) or raw.get("catalog_compilation_complete") is not True:
        raise CatalogBatchIncompleteError(
            0, int(config.CATALOG_BATCH_MAX_ATTEMPTS),
            "catalog compilation did not produce a complete merged result",
        )
    payload = _finalize_catalog(
        raw if isinstance(raw, dict) else {}, key=key,
        processed_document_key=processed_key, selected_chunk_ids=selected,
        usage=compiler_usage or (
            getattr(client, "last_usage", None) if client is not None else {}
        ),
    )
    if not payload["tasks"] and not payload["fitb_bank"]:
        # Compatibility for legacy/local test doubles that still return the
        # former planner contract. Production structured output cannot omit
        # tasks because AssessmentCatalogResponse requires them.
        logger.warning("Assessment compiler returned the legacy planner contract; using planner fallback")
        return {
            "assessment_catalog": {},
            "assessment_catalog_key": key,
            "assessment_catalog_cache_hit": False,
            "assessment_compiler_cache_hit": compiler_cache_hit,
            "assessment_catalog_insufficient_content": True,
            "assessment_catalog_failure_reason": "no_source_supported_assessment_tasks",
        }
    _write_cache(key, payload)
    logger.info(
        "Assessment catalog %s | key=%s | tasks=%d",
        "migrated" if compatibility_cache_hit else "compiled",
        key,
        len(payload["tasks"]),
    )
    return {
        "assessment_catalog": payload,
        "assessment_catalog_key": key,
        "assessment_catalog_cache_hit": compatibility_cache_hit,
        "assessment_compiler_cache_hit": compiler_cache_hit,
        "assessment_catalog_compatibility_cache_hit": compatibility_cache_hit,
        "planner_context": "",
    }


def _difficulty_for(task: dict[str, Any], requested: str) -> str | None:
    supported = list(task.get("supported_difficulties") or [])
    if requested != "mix":
        native = _native_difficulty(task)
        return native if native == requested else None
    for level in ("medium", "easy", "hard"):
        if level in supported:
            return level
    return None


def _native_difficulty(task: dict[str, Any]) -> str | None:
    """Return one honest task level, including a deterministic legacy fallback."""
    for value in (task.get("native_difficulty"),):
        level = str(value or "")
        if level in {"easy", "medium", "hard"}:
            return level
    supported = {
        str(value) for value in (task.get("supported_difficulties") or [])
        if str(value) in {"easy", "medium", "hard"}
    }
    # Old catalogs used supported_difficulties as a broad capability list and did not
    # persist a native level. Preserve the old Mixed-mode preference as the migration
    # rule, while never deriving the level from the teacher's current request.
    for level in ("medium", "easy", "hard"):
        if level in supported:
            return level
    return None


def _slot_question_type(slot_id: str) -> str | None:
    match = re.fullmatch(
        r"m\d+_(mcq|true_false|fill_in_the_blank|definition|why|equation|"
        r"word_problem|essay)_\d+",
        slot_id,
    )
    return match.group(1) if match else None


def materialize_plan_item(
    task: dict[str, Any],
    slot_id: str,
    difficulty: str,
    *,
    selected_variation_id: str | None = None,
) -> dict[str, Any]:
    evidence = list(task.get("evidence_chunk_ids") or [])
    relationship = task.get("relationship")
    idea = f"{task.get('student_work_product')}: {task.get('target')}"
    if task.get("reasoning_mode") == "derivation_relationship":
        idea = f"Derive and establish the relationship: {idea}"
    item = {
        "slot_id": slot_id,
        "task_id": task["task_id"],
        "task_family_id": task.get("task_family_id") or task["task_id"],
        "coverage_id": task["coverage_id"],
        "concept_id": task.get("concept_id"),
        "relationship_id": task.get("relationship_id"),
        "source_chunk_id": evidence[0],
        "support_chunk_ids": evidence[1:],
        # PlanItem retains a presentation topic for existing UI/prompt contracts;
        # Catalog tasks no longer duplicate source section titles.
        "topic": task.get("concept") or task.get("knowledge_anchor"),
        "concept": task.get("concept") or task.get("knowledge_anchor"),
        "canonical_skill": task.get("task_operation"),
        "knowledge_focus": task.get("knowledge_anchor"),
        "student_work_product": task.get("student_work_product"),
        "relationship": relationship,
        "target": task.get("target"),
        "task_operation": task.get("task_operation"),
        "given_conditions": list(task.get("required_givens") or []),
        "scenario_family": task.get("scenario_family"),
        "required_givens": list(task.get("required_givens") or []),
        "allowed_variations": list(task.get("allowed_variations") or []),
        "permitted_variations": permitted_task_variations(task),
        "primary_question_types": list(
            task.get("primary_question_types") or task.get("allowed_question_types") or []
        ),
        "variable_givens": list(task.get("variable_givens") or []),
        "complexity_boundaries": list(task.get("complexity_boundaries") or []),
        "exact_source_instance": bool(task.get("exact_source_instance", False)),
        "text_answerable": task.get("text_answerable") is True,
        "reasoning_mode": task.get("reasoning_mode"),
        "difficulty": difficulty,
        "requested_difficulty": difficulty,
        "actual_difficulty": difficulty,
        "base_catalog_task_id": task["task_id"],
        "base_task_family_id": task.get("task_family_id") or task["task_id"],
        "difficulty_adaptation": None,
        "difficulty_fallback_reason": None,
        "reasoning_depth": difficulty,
        "reasoning_requirements": [
            f"{task.get('task_operation')}: {task.get('target')}"
        ],
        "prerequisites": list(task.get("prerequisites") or []),
        "derived_task_id": None,
        "selected_replacement_task_id": None,
        "task_was_previously_used": False,
        "adjustment_attempts": 0,
        "adjustment_failure_reason": None,
        "idea_to_test": idea,
        "concept_to_test": idea,
    }
    variation = _effective_variation(
        task, _slot_question_type(slot_id), selected_variation_id
    )
    if variation is not None:
        _apply_effective_variation(item, task, variation)
    return item


def materialize_fitb_plan_item(
    entry: dict[str, Any], slot_id: str, difficulty: str
) -> dict[str, Any]:
    """Build the FITB plan contract without inventing a catalog task identity."""
    return {
        "slot_id": slot_id,
        "fitb_entry_id": entry["fitb_entry_id"],
        "task_id": None,
        "task_family_id": None,
        "coverage_id": None,
        "concept_id": None,
        "relationship_id": None,
        "source_chunk_id": entry["evidence_chunk_id"],
        "support_chunk_ids": [],
        "topic": entry["answer"],
        "concept": entry["answer"],
        "canonical_skill": "identify_term_from_text_clue",
        "knowledge_focus": entry["clue_basis"],
        "student_work_product": "complete one textual blank with the source term",
        "relationship": None,
        "target": entry["answer"],
        "task_operation": "identify_term_from_text_clue",
        "given_conditions": [],
        "scenario_family": None,
        "required_givens": [],
        "text_answerable": True,
        "reasoning_mode": "conceptual_recall",
        "difficulty": difficulty,
        "requested_difficulty": difficulty,
        "actual_difficulty": difficulty,
        "reasoning_depth": difficulty,
        "reasoning_requirements": ["identify the supported term from one textual clue"],
        "prerequisites": [],
        "difficulty_adaptation": None,
        "difficulty_fallback_reason": None,
        "idea_to_test": entry["clue_basis"],
        "concept_to_test": entry["clue_basis"],
        "fitb_answer": entry["answer"],
        "normalized_fitb_answer": entry["normalized_answer"],
        "clue_basis": entry["clue_basis"],
        "fitb_entry_used": bool(entry.get("used")),
        "fitb_alias_of_entry_id": entry.get("alias_of_fitb_entry_id"),
    }


def allocate_catalog_plan(
    catalog: dict[str, Any], tasks: list[tuple[str, int]], num_models: int,
    difficulty: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        catalog.get("catalog_compilation_complete") is not True
        and (
            "catalog_compilation_complete" in catalog
            or "catalog_version" in catalog
            or "cache_key" in catalog
        )
    ):
        raise RuntimeError("Assessment Catalog is incomplete and cannot be planned")
    source_tasks = list(catalog.get("tasks") or [])
    plans: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    task_usage_count: dict[str, int] = {}
    coverage_usage_count: dict[str, int] = {}
    scenario_usage_count: dict[str, int] = {}
    target_usage_count: dict[str, int] = {}
    variation_group_usage_count: dict[str, int] = {}
    fitb_usage_count: dict[str, int] = {}
    for model_number in range(1, num_models + 1):
        items: dict[str, list[dict[str, Any]]] = {qtype: [] for qtype, _ in tasks}
        slots = [
            (qtype, index)
            for qtype, count in tasks if qtype != "fill_in_the_blank"
            for index in range(1, count + 1)
        ]
        profile = catalog.get("assessment_profile") or {}
        valid_chunks = set(map(str, catalog.get("selected_chunk_ids") or [])) or None
        permitted_by_position = {
            position: permitted_task_variations(candidate)
            for position, candidate in enumerate(source_tasks)
        }
        compatible: dict[
            tuple[str, int], list[tuple[int, dict[str, Any], str, dict[str, Any]]]
        ] = {}
        candidate_pool_by_type: dict[
            str, list[tuple[int, dict[str, Any], str, dict[str, Any]]]
        ] = {}
        for qtype in {slot[0] for slot in slots}:
            pool: list[tuple[int, dict[str, Any], str, dict[str, Any]]] = []
            for position, candidate in enumerate(source_tasks):
                variations = eligible_task_variation_candidates(
                    candidate, qtype, assessment_profile=profile,
                    valid_chunk_ids=valid_chunks,
                    variations=permitted_by_position[position],
                )
                if not variations:
                    continue
                native_level = _native_difficulty(candidate)
                selected_level = _difficulty_for(candidate, difficulty)
                level = selected_level or (
                    native_level if difficulty in {"easy", "medium", "hard"} else None
                )
                if level is not None:
                    pool.extend(
                        (position, candidate, level, variation)
                        for variation in variations
                    )
            candidate_pool_by_type[qtype] = pool
        for slot in slots:
            compatible[slot] = candidate_pool_by_type[slot[0]]
        ordered_slots = sorted(slots, key=lambda slot: (len(compatible[slot]), slot[0], slot[1]))
        assignment: dict[
            tuple[str, int], tuple[dict[str, Any], str, dict[str, Any]]
        ] = {}
        resource_owner: dict[str, tuple[str, int]] = {}
        source_scenario_frequency = {
            scenario: sum(
                _normalized(task.get("scenario_family")) == scenario for task in source_tasks
            )
            for scenario in {
                _normalized(task.get("scenario_family")) for task in source_tasks
                if _normalized(task.get("scenario_family"))
            }
        }

        def candidate_sort_key(
            entry: tuple[int, dict[str, Any], str, dict[str, Any]],
        ) -> tuple[Any, ...]:
            return (
                -score_task_variation_candidate(
                    entry[1], entry[3], requested_difficulty=difficulty,
                    task_uses=task_usage_count.get(str(entry[1].get("task_id") or ""), 0),
                    scenario_uses=scenario_usage_count.get(normalize_planner_diversity_text(
                        entry[3].get("application") if entry[3].get("application") is not None
                        else entry[1].get("scenario_family")
                    ), 0),
                    target_uses=target_usage_count.get(normalize_planner_diversity_text(
                        entry[3].get("target") or entry[1].get("target")
                    ), 0),
                    variation_group_uses=variation_group_usage_count.get(
                        str(entry[3].get("variation_group_id") or ""), 0
                    ),
                )["score"],
                len(entry[1].get("allowed_question_types") or []),
                source_scenario_frequency.get(_normalized(entry[1].get("scenario_family")), 0),
                len(entry[1].get("evidence_chunk_ids") or []), entry[0],
            )

        for qtype, pool in candidate_pool_by_type.items():
            candidate_pool_by_type[qtype] = sorted(pool, key=candidate_sort_key)
        for slot in slots:
            compatible[slot] = candidate_pool_by_type[slot[0]]

        def resource_id(candidate: dict[str, Any]) -> str:
            return str(
                candidate.get("coverage_id")
                or candidate.get("task_family_id")
                or candidate.get("task_id") or ""
            )

        def match(slot: tuple[str, int], seen: set[str]) -> bool:
            for _position, candidate, level, variation in compatible[slot]:
                resource = resource_id(candidate)
                if not resource or resource in seen:
                    continue
                seen.add(resource)
                previous_slot = resource_owner.get(resource)
                if previous_slot is not None and not match(previous_slot, seen):
                    continue
                resource_owner[resource] = slot
                assignment[slot] = (candidate, level, variation)
                return True
            return False

        for slot in ordered_slots:
            match(slot, set())

        # Exact task IDs are the inventory unit. Family/coverage uniqueness is
        # preferred by the matching pass above, but it must not create an
        # artificial shortage while a compatible task exists. Fill remaining
        # slots from unused task IDs first, then enter explicit reuse mode.
        assigned_ids = {
            str(task.get("task_id") or "") for task, _level, _variation in assignment.values()
        }
        assigned_coverage_ids = {
            str(task.get("coverage_id") or "")
            for task, _level, _variation in assignment.values()
            if task.get("coverage_id")
        }
        for slot in ordered_slots:
            if slot in assignment or not compatible[slot]:
                continue
            unused_concept_tasks = [
                entry for entry in compatible[slot]
                if str(entry[1].get("task_id") or "") not in assigned_ids
                and task_usage_count.get(str(entry[1].get("task_id") or ""), 0) == 0
                and str(entry[1].get("coverage_id") or "") not in assigned_coverage_ids
                and coverage_usage_count.get(str(entry[1].get("coverage_id") or ""), 0) == 0
            ]
            unused_tasks = [
                entry for entry in compatible[slot]
                if str(entry[1].get("task_id") or "") not in assigned_ids
                and task_usage_count.get(str(entry[1].get("task_id") or ""), 0) == 0
            ]
            pool = unused_concept_tasks or unused_tasks or compatible[slot]
            _position, chosen, level, variation = min(
                pool,
                key=lambda entry: (
                    -score_task_variation_candidate(
                        entry[1], entry[3], requested_difficulty=difficulty,
                        task_uses=task_usage_count.get(str(entry[1].get("task_id") or ""), 0),
                        scenario_uses=scenario_usage_count.get(normalize_planner_diversity_text(
                            entry[3].get("application") if entry[3].get("application") is not None
                            else entry[1].get("scenario_family")
                        ), 0),
                        target_uses=target_usage_count.get(normalize_planner_diversity_text(
                            entry[3].get("target") or entry[1].get("target")
                        ), 0),
                        variation_group_uses=variation_group_usage_count.get(
                            str(entry[3].get("variation_group_id") or ""), 0
                        ),
                    )["score"],
                    str(entry[1].get("task_id") or ""),
                ),
            )
            assignment[slot] = (chosen, level, variation)
            assigned_ids.add(str(chosen.get("task_id") or ""))
            if chosen.get("coverage_id"):
                assigned_coverage_ids.add(str(chosen["coverage_id"]))

        # Re-rank the complete feasible assignment in constrained-slot order.
        # Resources reserved for other slots remain protected, while each
        # committed choice updates diversity counts for the next slot.
        live_task_usage = dict(task_usage_count)
        live_scenario_usage = dict(scenario_usage_count)
        live_target_usage = dict(target_usage_count)
        live_group_usage = dict(variation_group_usage_count)
        resource_by_slot = {
            slot: resource_id(value[0]) for slot, value in assignment.items()
        }
        for slot in ordered_slots:
            current = assignment.get(slot)
            if current is None:
                continue
            reserved_elsewhere = {
                resource for other, resource in resource_by_slot.items()
                if other != slot and resource
            }
            feasible = [
                entry for entry in compatible[slot]
                if resource_id(entry[1]) not in reserved_elsewhere
            ] or [(0, current[0], current[1], current[2])]

            def live_rank(entry):
                task, variation = entry[1], entry[3]
                scenario_key = normalize_planner_diversity_text(
                    variation.get("application") if variation.get("application") is not None
                    else task.get("scenario_family")
                )
                target_key = normalize_planner_diversity_text(
                    variation.get("target") or task.get("target")
                )
                group_id = str(variation.get("variation_group_id") or "")
                return (
                    -score_task_variation_candidate(
                        task, variation, requested_difficulty=difficulty,
                        task_uses=live_task_usage.get(str(task.get("task_id") or ""), 0),
                        scenario_uses=live_scenario_usage.get(scenario_key, 0),
                        target_uses=live_target_usage.get(target_key, 0),
                        variation_group_uses=live_group_usage.get(group_id, 0),
                    )["score"],
                    str(task.get("task_id") or ""),
                    str(variation.get("variation_id") or ""),
                )

            _position, chosen, level, variation = min(feasible, key=live_rank)
            assignment[slot] = (chosen, level, variation)
            resource_by_slot[slot] = resource_id(chosen)
            task_id = str(chosen.get("task_id") or "")
            scenario_key = normalize_planner_diversity_text(
                variation.get("application") if variation.get("application") is not None
                else chosen.get("scenario_family")
            )
            target_key = normalize_planner_diversity_text(
                variation.get("target") or chosen.get("target")
            )
            group_id = str(variation.get("variation_group_id") or "")
            live_task_usage[task_id] = live_task_usage.get(task_id, 0) + 1
            if scenario_key:
                live_scenario_usage[scenario_key] = live_scenario_usage.get(scenario_key, 0) + 1
            if target_key:
                live_target_usage[target_key] = live_target_usage.get(target_key, 0) + 1
            if group_id:
                live_group_usage[group_id] = live_group_usage.get(group_id, 0) + 1

        for qtype, count in tasks:
            for index in range(1, count + 1):
                if qtype == "fill_in_the_blank":
                    entries = [
                        entry for entry in catalog.get("fitb_bank") or []
                        if entry.get("fitb_entry_id")
                        and not entry.get("alias_of_fitb_entry_id")
                    ]
                    if not entries:
                        shortages[qtype] = shortages.get(qtype, 0) + 1
                        continue
                    chosen_entry = min(
                        entries,
                        key=lambda entry: (
                            fitb_usage_count.get(str(entry["fitb_entry_id"]), 0),
                            str(entry.get("normalized_answer") or ""),
                            str(entry["fitb_entry_id"]),
                        ),
                    )
                    entry_id = str(chosen_entry["fitb_entry_id"])
                    reused = fitb_usage_count.get(entry_id, 0) > 0
                    item = materialize_fitb_plan_item(
                        chosen_entry, f"m{model_number}_{qtype}_{index}",
                        "easy" if difficulty == "mix" else difficulty,
                    )
                    fitb_usage_count[entry_id] = fitb_usage_count.get(entry_id, 0) + 1
                    used_ids = {
                        value for value, usage in fitb_usage_count.items() if usage > 0
                    }
                    all_ids = {str(entry["fitb_entry_id"]) for entry in entries}
                    item.update({
                        "task_reused": False,
                        "fitb_entry_reused": reused,
                        "reuse_reason": "no_unused_compatible_fitb_entry" if reused else None,
                        "task_inventory": {
                            "question_type": qtype,
                            "identity_kind": "fitb_entry_id",
                            "compatible_fitb_entry_ids": sorted(all_ids),
                            "compatible_used_fitb_entry_ids": sorted(used_ids),
                            "compatible_unused_fitb_entry_ids": sorted(all_ids - used_ids),
                            "compatible_task_ids": sorted(all_ids),
                            "compatible_used_task_ids": sorted(used_ids),
                            "compatible_unused_task_ids": sorted(all_ids - used_ids),
                            "compatible_total_count": len(all_ids),
                            "compatible_used_count": len(used_ids),
                            "compatible_unused_count": len(all_ids - used_ids),
                            "reuse_mode": reused,
                            "reuse_reason": "no_unused_compatible_fitb_entry" if reused else None,
                        },
                    })
                    if reused:
                        prior_slots = [
                            str(previous.get("slot_id") or "")
                            for prior in plans
                            for values in (prior.get("items") or {}).values()
                            for previous in values or []
                            if str(previous.get("fitb_entry_id") or "") == entry_id
                        ] + [
                            str(previous.get("slot_id") or "")
                            for previous in items.get(qtype) or []
                            if str(previous.get("fitb_entry_id") or "") == entry_id
                        ]
                        item.update({
                            "reused_due_to_catalog_exhaustion": True,
                            "reused_fitb_entry_id": entry_id,
                            "catalog_replacement_stage": "reused_fitb_entry_after_bank_exhaustion",
                            "forced_duplicate_slot_ids": list(dict.fromkeys(filter(None, prior_slots))),
                            "required_instance_distinctions": [
                                "different_textual_clue", "different_direction_of_recall",
                            ],
                        })
                    items[qtype].append(item)
                    continue
                selected = assignment.get((qtype, index))
                if selected is None:
                    shortages[qtype] = shortages.get(qtype, 0) + 1
                    continue
                chosen, level, variation = selected
                task_id = str(chosen.get("task_id") or "")
                reused = task_usage_count.get(task_id, 0) > 0
                item = materialize_plan_item(
                    chosen, f"m{model_number}_{qtype}_{index}", level,
                    selected_variation_id=variation.get("variation_id"),
                )
                scenario_key = normalize_planner_diversity_text(
                    variation.get("application") if variation.get("application") is not None
                    else chosen.get("scenario_family")
                )
                target_key = normalize_planner_diversity_text(
                    variation.get("target") or chosen.get("target")
                )
                group_id = str(variation.get("variation_group_id") or "")
                ranking = score_task_variation_candidate(
                    chosen, variation, requested_difficulty=difficulty,
                    task_uses=task_usage_count.get(task_id, 0),
                    scenario_uses=scenario_usage_count.get(scenario_key, 0),
                    target_uses=target_usage_count.get(target_key, 0),
                    variation_group_uses=variation_group_usage_count.get(group_id, 0),
                )
                item["planner_candidate_score"] = ranking["score"]
                item["planner_candidate_score_components"] = ranking["components"]
                item["requested_difficulty"] = difficulty
                item["actual_difficulty"] = level
                if difficulty in {"easy", "medium", "hard"} and level != difficulty:
                    item["difficulty_adaptation"] = {
                        "status": "pending",
                        "from_difficulty": level,
                        "to_difficulty": difficulty,
                        "base_task_id": task_id,
                        "strategy": None,
                        "failed_strategies": [],
                    }
                task_usage_count[task_id] = task_usage_count.get(task_id, 0) + 1
                if scenario_key:
                    scenario_usage_count[scenario_key] = scenario_usage_count.get(scenario_key, 0) + 1
                if target_key:
                    target_usage_count[target_key] = target_usage_count.get(target_key, 0) + 1
                if group_id:
                    variation_group_usage_count[group_id] = variation_group_usage_count.get(group_id, 0) + 1
                coverage_id = str(chosen.get("coverage_id") or "")
                if coverage_id:
                    coverage_usage_count[coverage_id] = coverage_usage_count.get(coverage_id, 0) + 1
                compatible_ids = {
                    str(entry[1].get("task_id") or "")
                    for entry in candidate_pool_by_type.get(qtype, [])
                    if entry[1].get("task_id")
                }
                used_compatible_ids = {
                    candidate_id for candidate_id in compatible_ids
                    if task_usage_count.get(candidate_id, 0) > 0
                }
                item["task_inventory"] = {
                    "question_type": qtype,
                    "compatible_task_ids": sorted(compatible_ids),
                    "compatible_used_task_ids": sorted(used_compatible_ids),
                    "compatible_unused_task_ids": sorted(compatible_ids - used_compatible_ids),
                    "compatible_total_count": len(compatible_ids),
                    "compatible_used_count": len(used_compatible_ids),
                    "compatible_unused_count": len(compatible_ids - used_compatible_ids),
                    "reuse_mode": reused,
                    "reuse_reason": "no_unused_compatible_task" if reused else None,
                }
                item["task_reused"] = reused
                if reused:
                    same_task_slots = [
                        str(previous.get("slot_id") or "")
                        for prior_plan in plans
                        for values in (prior_plan.get("items") or {}).values()
                        for previous in values or []
                        if str(previous.get("task_id") or "") == task_id
                    ] + [
                        str(previous.get("slot_id") or "")
                        for values in items.values()
                        for previous in values
                        if str(previous.get("task_id") or "") == task_id
                    ]
                    item.update({
                        "reuse_reason": "no_unused_compatible_task",
                        "reused_due_to_catalog_exhaustion": True,
                        "reused_task_id": task_id,
                        "catalog_replacement_stage": "reused_task_after_catalog_exhaustion",
                        "forced_duplicate_slot_ids": list(dict.fromkeys(filter(None, same_task_slots))),
                        "required_instance_distinctions": [
                            "different_target_or_unknown", "different_required_method",
                            "different_direction_of_reasoning",
                        ],
                    })
                items[qtype].append(item)
        plans.append({"model_number": model_number, "items": items})
    for qtype, count in tasks:
        supported = count - shortages.get(qtype, 0)
        if supported < count:
            issues.append({
                "question_type": qtype,
                "requested_count": count,
                "supported_count": max(0, supported),
                "reason": "The catalog has no compatible grounded task for this question type and difficulty.",
                "suggested_question_types": [],
            })
    return plans, issues


def _replacement_representation_rank(
    task: dict[str, Any], represented_items: list[dict[str, Any]], *,
    include_task_usage: bool,
) -> tuple[Any, ...]:
    """Rank one compatible task against the current active exam."""
    task_id = str(task.get("task_id") or "")

    def count(field: str, value: Any) -> int:
        key = _normalized(value)
        if not key:
            return 0
        return sum(_normalized(item.get(field)) == key for item in represented_items)

    task_usage = sum(
        str(item.get("task_id") or "") == task_id for item in represented_items
    )
    concept_count = (
        count("concept_id", task.get("concept_id"))
        if task.get("concept_id") else count("concept", task.get("concept"))
    )
    return (
        task_usage if include_task_usage else 0,
        count("coverage_id", task.get("coverage_id")),
        concept_count,
        count("task_operation", task.get("task_operation")),
        count("target", task.get("target")),
        count("task_family_id", task.get("task_family_id") or task_id),
        count("reasoning_mode", task.get("reasoning_mode")),
        task_id,
    )


def replacement_candidate_score(
    task: dict[str, Any], variation: dict[str, Any], plans: list[dict[str, Any]],
    *, excluding_slot_id: str, requested_difficulty: str,
) -> dict[str, Any]:
    """Score a replacement from the latest active plan state."""
    represented = [
        item for plan in plans for values in (plan.get("items") or {}).values()
        for item in values or []
        if str(item.get("slot_id") or "") != excluding_slot_id
    ]
    task_id = str(task.get("task_id") or "")
    scenario = variation.get("application")
    if scenario is None:
        scenario = task.get("scenario_family")
    target = variation.get("target") or task.get("target")
    scenario_key = normalize_planner_diversity_text(scenario)
    target_key = normalize_planner_diversity_text(target)
    group_id = str(variation.get("variation_group_id") or "")
    return score_task_variation_candidate(
        task, variation, requested_difficulty=requested_difficulty,
        task_uses=sum(
            str(item.get("base_catalog_task_id") or item.get("task_id") or "") == task_id
            for item in represented
        ),
        scenario_uses=sum(
            bool(scenario_key)
            and normalize_planner_diversity_text(item.get("scenario_family")) == scenario_key
            for item in represented
        ),
        target_uses=sum(
            bool(target_key)
            and normalize_planner_diversity_text(item.get("target")) == target_key
            for item in represented
        ),
        variation_group_uses=sum(
            bool(group_id)
            and str(item.get("selected_variation_group_id") or "") == group_id
            for item in represented
        ),
    )


def unused_catalog_replacement(
    catalog: dict[str, Any], plans: list[dict[str, Any]], slot_id: str,
    qtype: str, requested_difficulty: str, *,
    excluded_task_ids: set[str] | None = None,
    used_fitb_answer_keys: set[str] | None = None,
    active_slot_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    excluded = {str(value) for value in (excluded_task_ids or set()) if str(value)}
    current_items = [
        item
        for plan in plans
        for values in (plan.get("items") or {}).values()
        for item in values
        if str(item.get("slot_id") or "") == slot_id
    ]
    current_task_ids = {
        str(item.get("task_id") or "") for item in current_items
        if item.get("task_id")
    }
    current_family_ids = {
        str(item.get("task_family_id") or "") for item in current_items
        if item.get("task_family_id")
    }
    represented_items = [
        item
        for plan in plans
        for values in (plan.get("items") or {}).values()
        for item in values
        if str(item.get("slot_id") or "") != slot_id
        and (active_slot_ids is None or str(item.get("slot_id") or "") in active_slot_ids)
    ]
    used = {
        str(item.get("task_id") or "") for item in represented_items
        if item.get("task_id")
    }
    used_families = {
        str(item.get("task_family_id") or "") for item in represented_items
        if item.get("task_family_id")
    }
    represented_fitb_answers = {
        normalize_fitb_answer_key(value)
        for value in (used_fitb_answer_keys or set())
        if normalize_fitb_answer_key(value)
    }
    profile = catalog.get("assessment_profile") or {}
    valid_chunks = set(map(str, catalog.get("selected_chunk_ids") or [])) or None
    candidates: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for task in catalog.get("tasks") or []:
        level = _difficulty_for(task, requested_difficulty)
        variations = eligible_task_variation_candidates(
            task, qtype, assessment_profile=profile, valid_chunk_ids=valid_chunks,
        )
        if (
            level and variations
            and str(task.get("task_id") or "") not in excluded
            and str(task.get("task_id")) not in current_task_ids
            and str(task.get("task_family_id") or task.get("task_id")) not in current_family_ids
            and str(task.get("task_id")) not in used
            and str(task.get("task_family_id") or task.get("task_id")) not in used_families
        ):
            candidates.extend((task, level, variation) for variation in variations)
    if not candidates:
        return None
    task, level, variation = min(
        candidates,
        key=lambda entry: (
            -replacement_candidate_score(
                entry[0], entry[2], plans, excluding_slot_id=slot_id,
                requested_difficulty=requested_difficulty,
            )["score"],
            int(bool(
                qtype == "fill_in_the_blank"
                and fitb_answer_keys(entry[0]) & represented_fitb_answers
            )),
            *_replacement_representation_rank(
                entry[0], represented_items, include_task_usage=False,
            ),
        ),
    )
    return materialize_plan_item(
        task, slot_id, level,
        selected_variation_id=str(variation.get("variation_id") or ""),
    )


def catalog_exhaustion_replacement(
    catalog: dict[str, Any], plans: list[dict[str, Any]], slot_id: str,
    qtype: str, requested_difficulty: str, *,
    forced_duplicate_slot_ids: list[str] | None = None,
    forced_duplicate_question_ids: list[str] | None = None,
    excluded_task_ids: set[str] | None = None,
    used_fitb_answer_keys: set[str] | None = None,
    active_slot_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Reserve a bounded replacement, ranked against the active exam when supplied."""
    if qtype == "fill_in_the_blank":
        all_items = [
            item for plan in plans for values in (plan.get("items") or {}).values()
            for item in values or []
            if active_slot_ids is None or str(item.get("slot_id") or "") in active_slot_ids
        ]
        current = next((
            item for item in all_items if str(item.get("slot_id") or "") == slot_id
        ), {})
        current_id = str(current.get("fitb_entry_id") or "")
        excluded = {str(value) for value in (excluded_task_ids or set()) if str(value)}
        entries = [
            entry for entry in catalog.get("fitb_bank") or []
            if entry.get("fitb_entry_id") and not entry.get("alias_of_fitb_entry_id")
            and str(entry.get("fitb_entry_id")) not in excluded
        ]
        if not entries:
            return None
        represented = [
            item for item in all_items if str(item.get("slot_id") or "") != slot_id
        ]
        usage: dict[str, int] = {}
        for item in represented:
            entry_id = str(item.get("fitb_entry_id") or "")
            if entry_id:
                usage[entry_id] = usage.get(entry_id, 0) + 1
        unused = [
            entry for entry in entries
            if usage.get(str(entry["fitb_entry_id"]), 0) == 0
            and str(entry["fitb_entry_id"]) != current_id
        ]
        alternatives = [
            entry for entry in entries
            if usage.get(str(entry["fitb_entry_id"]), 0) > 0
            and str(entry["fitb_entry_id"]) != current_id
        ]
        original = [entry for entry in entries if str(entry["fitb_entry_id"]) == current_id]
        tier = unused or alternatives or original
        if not tier:
            return None
        selected = min(tier, key=lambda entry: (
            usage.get(str(entry["fitb_entry_id"]), 0),
            int(normalize_fitb_answer_key(entry.get("answer")) in {
                normalize_fitb_answer_key(value) for value in (used_fitb_answer_keys or set())
            }),
            str(entry.get("normalized_answer") or ""),
            str(entry["fitb_entry_id"]),
        ))
        selected_id = str(selected["fitb_entry_id"])
        reused = not bool(unused)
        replacement = materialize_fitb_plan_item(
            selected, slot_id, "easy" if requested_difficulty == "mix" else requested_difficulty
        )
        related_slots = sorted(
            str(item.get("slot_id") or "") for item in represented
            if str(item.get("fitb_entry_id") or "") == selected_id
        )
        inventory = compatible_task_inventory(
            catalog, plans, qtype, requested_difficulty, excluding_slot_id=slot_id,
            excluded_task_ids=excluded_task_ids, active_slot_ids=active_slot_ids,
        )
        replacement.update({
            "catalog_replacement_stage": (
                "unused_fitb_entry" if unused
                else "reused_fitb_entry_after_bank_exhaustion" if alternatives
                else "regenerated_original_fitb_entry"
            ),
            "fitb_entry_reused": reused,
            "task_reused": False,
            "reuse_reason": "no_unused_compatible_fitb_entry" if reused else None,
            "reused_fitb_entry_id": selected_id if reused else None,
            "reused_due_to_catalog_exhaustion": bool(reused and alternatives),
            "forced_duplicate_slot_ids": list(dict.fromkeys([
                *related_slots,
                *(str(value) for value in (forced_duplicate_slot_ids or []) if str(value)),
            ])),
            "forced_duplicate_question_ids": list(dict.fromkeys(
                str(value) for value in (forced_duplicate_question_ids or []) if str(value)
            )),
            "required_instance_distinctions": (
                ["different_textual_clue", "different_direction_of_recall"] if reused else []
            ),
            "task_inventory": _inventory_after_reservation(
                inventory, selected_id, reused=reused
            ),
        })
        return replacement
    inventory = compatible_task_inventory(
        catalog, plans, qtype, requested_difficulty, excluding_slot_id=slot_id,
        excluded_task_ids=excluded_task_ids,
        active_slot_ids=active_slot_ids,
    )
    all_items = [
        item for plan in plans for values in (plan.get("items") or {}).values()
        for item in values or []
        if active_slot_ids is None or str(item.get("slot_id") or "") in active_slot_ids
    ]
    represented_fitb_answers = {
        normalize_fitb_answer_key(value)
        for value in (used_fitb_answer_keys or set())
        if normalize_fitb_answer_key(value)
    }

    def attach_fitb_answer_inventory(
        replacement: dict[str, Any], task: dict[str, Any]
    ) -> dict[str, Any]:
        if qtype != "fill_in_the_blank":
            return replacement
        candidate_keys = sorted(fitb_answer_keys(task))
        answer_reused = bool(set(candidate_keys) & represented_fitb_answers)
        replacement["fitb_answer_reused"] = answer_reused
        task_inventory = dict(replacement.get("task_inventory") or {})
        task_inventory.update({
            "fitb_represented_answer_keys": sorted(represented_fitb_answers),
            "fitb_candidate_answer_keys": candidate_keys,
            "fitb_answer_reused": answer_reused,
        })
        replacement["task_inventory"] = task_inventory
        return replacement
    strict = unused_catalog_replacement(
        catalog, plans, slot_id, qtype, requested_difficulty,
        excluded_task_ids=excluded_task_ids,
        used_fitb_answer_keys=used_fitb_answer_keys,
        active_slot_ids=active_slot_ids,
    )
    used_task_ids_before_replacement = {
        str(item.get("task_id") or "") for item in all_items
        if str(item.get("slot_id") or "") != slot_id and item.get("task_id")
    }
    excluded_ids = {
        str(value) for value in (excluded_task_ids or set()) if str(value)
    }
    fitb_used_with_unused_answer = [
        task for task in catalog.get("tasks") or []
        if qtype == "fill_in_the_blank"
        and str(task.get("task_id") or "") in used_task_ids_before_replacement
        and str(task.get("task_id") or "") not in excluded_ids
        and eligible_task_variations(task, qtype)
        and _difficulty_for(task, requested_difficulty) is not None
        and not (fitb_answer_keys(task) & represented_fitb_answers)
    ]
    prefer_used_fresh_fitb_answer = bool(
        fitb_used_with_unused_answer
        and (
            strict is None
            or bool(fitb_answer_keys(strict) & represented_fitb_answers)
        )
    )
    if strict is not None and not prefer_used_fresh_fitb_answer:
        family_id = str(strict.get("task_family_id") or "")
        coverage_id = str(strict.get("coverage_id") or "")
        related_slots = [
            str(item.get("slot_id") or "") for item in all_items
            if str(item.get("slot_id") or "") != slot_id and (
                (family_id and str(item.get("task_family_id") or "") == family_id)
                or (coverage_id and str(item.get("coverage_id") or "") == coverage_id)
            )
        ]
        shared_identity = bool(related_slots)
        concept_id = str(strict.get("concept_id") or "")
        concept_was_used = bool(concept_id and any(
            str(item.get("slot_id") or "") != slot_id
            and str(item.get("concept_id") or "") == concept_id
            for item in all_items
        ))
        strict["catalog_replacement_stage"] = (
            "unused_task_used_concept" if concept_was_used
            else "unused_task_unused_concept"
        )
        if shared_identity:
            strict["forced_duplicate_slot_ids"] = list(dict.fromkeys(
                [*related_slots, *(
                    str(value) for value in (forced_duplicate_slot_ids or []) if str(value)
                )]
            ))
            strict["forced_duplicate_question_ids"] = list(dict.fromkeys(
                str(value) for value in (forced_duplicate_question_ids or []) if str(value)
            ))
        strict["task_inventory"] = _inventory_after_reservation(
            inventory, str(strict.get("task_id") or ""), reused=False
        )
        strict["task_reused"] = False
        return attach_fitb_answer_inventory(strict, strict)

    current = next(
        (item for item in all_items if str(item.get("slot_id") or "") == slot_id),
        {},
    )
    current_task_id = str(current.get("task_id") or "")
    used_task_ids = {
        str(item.get("task_id") or "") for item in all_items
        if str(item.get("slot_id") or "") != slot_id and item.get("task_id")
    }

    profile = catalog.get("assessment_profile") or {}
    valid_chunks = set(map(str, catalog.get("selected_chunk_ids") or [])) or None
    compatible: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for task in catalog.get("tasks") or []:
        level = _difficulty_for(task, requested_difficulty)
        variations = eligible_task_variation_candidates(
            task, qtype, assessment_profile=profile, valid_chunk_ids=valid_chunks,
        )
        if (
            level and variations
            and str(task.get("task_id") or "") not in {
                str(value) for value in (excluded_task_ids or set()) if str(value)
            }
        ):
            compatible.extend((task, level, variation) for variation in variations)

    represented_items = [
        item for item in all_items
        if str(item.get("slot_id") or "") != slot_id
    ]

    def coverage_rank(task: dict[str, Any], *, include_task_usage: bool) -> tuple[Any, ...]:
        return (
            int(bool(
                qtype == "fill_in_the_blank"
                and fitb_answer_keys(task) & represented_fitb_answers
            )),
            *_replacement_representation_rank(
                task, represented_items, include_task_usage=include_task_usage,
            ),
        )

    # Stage 2: retain exact task-ID uniqueness while relaxing only the broader
    # family/coverage uniqueness constraint.
    unused = [
        (task, level, variation) for task, level, variation in compatible
        if str(task.get("task_id") or "") != current_task_id
        and str(task.get("task_id") or "") not in used_task_ids
    ]
    if unused and not prefer_used_fresh_fitb_answer:
        task, level, variation = min(
            unused, key=lambda entry: (
                -replacement_candidate_score(
                    entry[0], entry[2], plans, excluding_slot_id=slot_id,
                    requested_difficulty=requested_difficulty,
                )["score"],
                *coverage_rank(entry[0], include_task_usage=False),
            )
        )
        replacement = materialize_plan_item(
            task, slot_id, level,
            selected_variation_id=str(variation.get("variation_id") or ""),
        )
        family_id = str(replacement.get("task_family_id") or "")
        coverage_id = str(replacement.get("coverage_id") or "")
        related_slots = [
            str(item.get("slot_id") or "") for item in all_items
            if str(item.get("slot_id") or "") != slot_id and (
                (family_id and str(item.get("task_family_id") or "") == family_id)
                or (coverage_id and str(item.get("coverage_id") or "") == coverage_id)
            )
        ]
        concept_id = str(replacement.get("concept_id") or "")
        concept_was_used = bool(concept_id and any(
            str(item.get("slot_id") or "") != slot_id
            and str(item.get("concept_id") or "") == concept_id
            for item in all_items
        ))
        replacement.update({
            "catalog_replacement_stage": (
                "unused_task_used_concept" if concept_was_used
                else "unused_task_unused_concept"
            ),
            "forced_duplicate_slot_ids": list(dict.fromkeys(
                [*related_slots, *(
                    str(value) for value in (forced_duplicate_slot_ids or []) if str(value)
                )]
            )),
            "forced_duplicate_question_ids": list(dict.fromkeys(
                str(value) for value in (forced_duplicate_question_ids or []) if str(value)
            )),
            "task_inventory": _inventory_after_reservation(
                inventory, str(replacement.get("task_id") or ""), reused=False
            ),
            "task_reused": False,
        })
        return attach_fitb_answer_inventory(replacement, task)

    # Stage 3: all compatible tasks are represented. Prefer the least-represented
    # alternative task before retrying the task that already produced a duplicate.
    excluded = {
        str(value) for value in (excluded_task_ids or set()) if str(value)
    }
    # Reuse a compatible task occupied by another active slot. Failed tasks and the
    # rejected original task remain excluded from this candidate tier.
    usage_count: dict[str, int] = {}
    for item in all_items:
        if str(item.get("slot_id") or "") == slot_id:
            continue
        task_id = str(item.get("task_id") or "")
        if task_id:
            usage_count[task_id] = usage_count.get(task_id, 0) + 1
    reused = [
        (usage_count.get(str(task.get("task_id") or ""), 0), task, level, variation)
        for task, level, variation in compatible
        if usage_count.get(str(task.get("task_id") or ""), 0) > 0
        and str(task.get("task_id") or "") != current_task_id
    ]
    if reused:
        _count, task, level, variation = min(
            reused, key=lambda value: (
                -replacement_candidate_score(
                    value[1], value[3], plans, excluding_slot_id=slot_id,
                    requested_difficulty=requested_difficulty,
                )["score"],
                *coverage_rank(value[1], include_task_usage=True),
            )
        )
        reused_task_id = str(task.get("task_id") or "")
        same_task_slots = [
            str(item.get("slot_id") or "") for item in all_items
            if str(item.get("slot_id") or "") != slot_id
            and str(item.get("task_id") or "") == reused_task_id
        ]
        replacement = materialize_plan_item(
            task, slot_id, level,
            selected_variation_id=str(variation.get("variation_id") or ""),
        )
        replacement.update({
            "catalog_replacement_stage": "reused_task_after_catalog_exhaustion",
            "reused_due_to_catalog_exhaustion": True,
            "task_reused": True,
            "reuse_reason": "no_unused_compatible_task",
            "reuse_mode": "intentional_practice_variation",
            "task_usage_count": _count,
            "previous_question_ids": [],
            "variation_reason": "catalog_exhausted_use_least_used_compatible_task",
            "intended_variation": (
                "Prefer a source-supported different representation or procedure; otherwise use a "
                "genuinely different quantitative instance with recalculated answers and solutions."
            ),
            "source_evidence_ids": list(dict.fromkeys([
                *([replacement.get("source_chunk_id")] if replacement.get("source_chunk_id") else []),
                *(replacement.get("support_chunk_ids") or []),
            ])),
            "reused_task_id": reused_task_id,
            "required_instance_distinctions": [
                "different_target_or_unknown",
                "different_required_method",
                "different_representation",
                "different_constraint_structure",
                "different_direction_of_reasoning",
                "different_application_requiring_new_reasoning",
                "different_numerical_instance_when_other_variations_are_unsupported",
            ],
            "reuse_rule": (
                "Prefer a source-supported different representation or procedure. If the source "
                "supports only the same method, a genuinely different numerical instance is allowed "
                "when all dependent answers and solution steps are recalculated. Wording alone never qualifies."
            ),
            "forced_duplicate_slot_ids": list(dict.fromkeys([
                *same_task_slots,
                *(str(value) for value in (forced_duplicate_slot_ids or []) if str(value)),
            ])),
            "forced_duplicate_question_ids": list(dict.fromkeys(
                str(value) for value in (forced_duplicate_question_ids or []) if str(value)
            )),
            "task_inventory": _inventory_after_reservation(
                inventory, reused_task_id, reused=True
            ),
        })
        return attach_fitb_answer_inventory(replacement, task)

    # Stage 4: a duplicate generated question does not invalidate its catalog task.
    # Retry it only after no unused or alternative represented task remains.
    original = next((
        (task, level, variation) for task, level, variation in compatible
        if str(task.get("task_id") or "") == current_task_id
        and current_task_id not in excluded
    ), None)
    if original is None:
        return None
    task, level, variation = original
    replacement = materialize_plan_item(
        task, slot_id, level,
        selected_variation_id=str(variation.get("variation_id") or ""),
    )
    replacement.update({
        "catalog_replacement_stage": "regenerated_original_task_after_duplicate",
        "reused_due_to_catalog_exhaustion": False,
        "task_reused": True,
        "reuse_reason": "regenerate_original_task_after_duplicate",
        "reuse_mode": "intentional_practice_variation",
        "task_usage_count": usage_count.get(current_task_id, 0),
        "previous_question_ids": [],
        "variation_reason": "regenerate_valid_catalog_task_after_duplicate",
        "intended_variation": (
            "Produce a source-supported different representation or procedure; if unavailable, a "
            "genuinely different numerical instance with independently recalculated results."
        ),
        "source_evidence_ids": list(dict.fromkeys([
            *([replacement.get("source_chunk_id")] if replacement.get("source_chunk_id") else []),
            *(replacement.get("support_chunk_ids") or []),
        ])),
        "reused_task_id": current_task_id,
        "forced_duplicate_slot_ids": list(dict.fromkeys(
            str(value) for value in (forced_duplicate_slot_ids or []) if str(value)
        )),
        "forced_duplicate_question_ids": list(dict.fromkeys(
            str(value) for value in (forced_duplicate_question_ids or []) if str(value)
        )),
        "required_instance_distinctions": [
            "different_assessment_angle",
            "different_target_or_unknown_when_supported",
            "different_required_method_when_supported",
            "different_direction_of_reasoning_when_supported",
            "different_numerical_instance_when_other_variations_are_unsupported",
        ],
        "reuse_rule": (
            "Regenerate this valid catalog task after a duplicate item. Use the rejected "
            "question and its conflicts as negative examples. The replacement must require "
            "materially different assessment work; a paraphrase does not qualify."
        ),
        "task_inventory": _inventory_after_reservation(
            {**inventory, "reuse_reason": "regenerate_original_task_after_duplicate"},
            current_task_id, reused=True,
        ),
    })
    return attach_fitb_answer_inventory(replacement, task)
