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

from pydantic import ConfigDict, Field, StrictBool, ValidationError, model_validator

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
    TransformationGrant,
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


class CatalogBatchValidationError(ValueError):
    """One schema-valid compiler response contains invalid evidence links."""

    def __init__(
        self, diagnostics: list[dict[str, Any]], preserved: dict[str, Any],
    ) -> None:
        self.diagnostics = diagnostics
        self.preserved = preserved
        fields = sorted({str(item.get("field") or "evidence") for item in diagnostics})
        compatibility_labels: list[str] = []
        if any(field == "task.evidence_chunk_ids" for field in fields):
            compatibility_labels.append("task evidence IDs")
        if any(field == "variation.evidence_chunk_ids" for field in fields):
            compatibility_labels.append("variation evidence IDs")
        category = (
            f" ({', '.join(compatibility_labels)})" if compatibility_labels else ""
        )
        super().__init__(
            "evidence validation failed for "
            f"{len(diagnostics)} reference(s) in {', '.join(fields)}{category}"
        )


class CatalogBatchAssignmentError(ValueError):
    """Selected child metadata cannot be assigned to safe compiler batches."""


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
    equation_structure: str | None = Field(default=None, max_length=240)
    variable_parameters: list[NonEmptyString] = Field(default_factory=list, max_length=12)
    validity_constraints: list[NonEmptyString] = Field(default_factory=list, max_length=12)
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
    allowed_transformations: list[TransformationGrant] = Field(default_factory=list, max_length=6)


class FITBEntryDraft(StrictOutputModel):
    """Compact source-grounded vocabulary candidate owned by the compiler."""

    answer: str = Field(min_length=1, max_length=80)
    evidence_chunk_id: NonEmptyString
    clue_basis: str = Field(min_length=1, max_length=320)


class AssessmentCatalogResponse(StrictOutputModel):
    assessment_profile: AssessmentProfile
    tasks: list[CatalogTaskDraft]
    fitb_bank: list[FITBEntryDraft] = Field(default_factory=list, max_length=128)


class CompactTaskVariation(StrictOutputModel):
    """LLM-facing variation contract.

    Full field names remain accepted for old catalogs and existing tests; the
    aliases are the compact wire form used for new compiler output.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    target: NonEmptyString
    operation: NonEmptyString = Field(alias="op")
    concept: NonEmptyString | None = None
    equation_structure: NonEmptyString | None = Field(alias="equation", default=None)
    variables: list[NonEmptyString] = Field(alias="vars", default_factory=list, max_length=12)
    constraints: list[NonEmptyString] = Field(default_factory=list, max_length=12)
    type: QuestionType | None = None
    givens: list[NonEmptyString] = Field(default_factory=list, max_length=12)
    evidence_chunk_ids: list[NonEmptyString] | None = Field(default=None, alias="ev", max_length=4)
    reasoning_mode: ReasoningMode | None = Field(default=None, alias="mode")
    student_work_product: NonEmptyString | None = Field(default=None, alias="wp")
    required_response_components: list[Literal[
        "selection", "boolean_judgment", "short_text", "written_justification",
        "worked_solution", "equation", "extended_response",
    ]] = Field(default_factory=list, alias="response", max_length=4)
    representation: NonEmptyString | None = Field(default=None, alias="form")
    application: NonEmptyString | None = Field(default=None, alias="context")
    numeric_values_may_change: StrictBool = Field(default=False, alias="vary")
    requires_supplied_graph: StrictBool | None = Field(default=None, alias="graph")
    variation_id: str | None = None
    variation_group_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_full_variation(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "vary_numbers" in data and "vary" not in data:
            data["vary"] = data.pop("vary_numbers")
        if "question_types" in data and "type" not in data:
            types = data.pop("question_types") or []
            if len(types) > 1:
                raise ValueError("each compact variation must declare exactly one question type")
            data["type"] = types[0] if types else None
        if "equation_structure" in data and "equation" not in data:
            data["equation"] = data.pop("equation_structure")
        if "variables" in data and "vars" not in data:
            data["vars"] = data.pop("variables")
        return data


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
    equation_structure: str | None = Field(alias="equation", default=None, max_length=240)
    variable_parameters: list[NonEmptyString] = Field(
        alias="parameters", default_factory=list, max_length=12,
    )
    validity_constraints: list[NonEmptyString] = Field(
        alias="constraints", default_factory=list, max_length=12,
    )
    task_operation: str = Field(alias="operation", min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    student_work_product: str = Field(alias="work_product", min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    reasoning_mode: ReasoningMode = Field(alias="mode")
    allowed_question_types: list[QuestionType] = Field(alias="question_types", min_length=1)
    evidence_chunk_ids: list[NonEmptyString] = Field(alias="evidence_ids", min_length=1, max_length=4)
    required_givens: list[NonEmptyString] = Field(max_length=12)
    allowed_variations: list[CompactTaskVariation] = Field(alias="variations", default_factory=list)
    variable_givens: list[NonEmptyString] = Field(default_factory=list, max_length=12)
    complexity_boundaries: list[NonEmptyString] = Field(alias="boundaries", default_factory=list, max_length=8)
    exact_source_instance: bool = False
    scenario_family: str | None = Field(max_length=160)
    text_answerable: StrictBool = Field(alias="answerable")
    native_difficulty: Literal["easy", "medium", "hard"] = Field(alias="difficulty")
    prerequisites: list[NonEmptyString] = Field(alias="prereqs", default_factory=list, max_length=8)
    allowed_transformations: list[TransformationGrant] = Field(alias="transformations", default_factory=list, max_length=6)


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
CATALOG_BATCHING_VERSION = "parent-isolated-budget-v2"


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
    batch_id: str = "batch_1",
) -> dict[str, Any]:
    """Fail closed while retaining independently valid compiler work."""
    parsed = response_model.model_validate(value)
    normalized = parsed.model_dump()
    if normalized.get("batch_complete") is not True:
        raise ValueError("batch_complete must be explicitly true")
    allowed = sorted(map(str, valid_chunk_ids))
    diagnostics: list[dict[str, Any]] = []
    preserved_tasks: list[dict[str, Any]] = []

    def task_identity(task: dict[str, Any], index: int) -> str:
        explicit = str(task.get("task_id") or "").strip()
        if explicit:
            return explicit
        digest = hashlib.sha256("\x1f".join(map(str, (
            task.get("knowledge_anchor"), task.get("concept"),
            task.get("task_operation"), task.get("target"),
        ))).encode("utf-8")).hexdigest()[:12]
        return f"draft_task_{digest or index + 1}"

    def variation_identity(variation: dict[str, Any], index: int) -> str:
        explicit = str(variation.get("variation_id") or "").strip()
        if explicit:
            return explicit
        digest = hashlib.sha256("\x1f".join(map(str, (
            variation.get("target"), variation.get("operation"),
            variation.get("type"), variation.get("student_work_product"),
        ))).encode("utf-8")).hexdigest()[:12]
        return f"draft_variation_{digest or index + 1}"

    def record(
        *, task_id: str | None, variation_id: str | None, field: str,
        invalid_id: str, reason: str,
    ) -> None:
        diagnostics.append({
            "batch_id": batch_id,
            "task_id": task_id,
            "variation_id": variation_id,
            "field": field,
            "invalid_evidence_id": invalid_id,
            "allowed_evidence_ids": allowed,
            "reason": reason,
        })

    profile_invalid = False
    for question_type, contract in (
        (normalized.get("assessment_profile") or {}).get("question_types") or {}
    ).items():
        for invalid in (
            evidence_id for evidence_id in map(
                str, (contract or {}).get("evidence_chunk_ids") or []
            ) if evidence_id not in valid_chunk_ids
        ):
            profile_invalid = True
            record(
                task_id="assessment_profile", variation_id=None,
                field=(
                    f"assessment_profile.question_types.{question_type}"
                    ".evidence_chunk_ids"
                ),
                invalid_id=invalid,
                reason="assessment profile evidence ID is outside this compiler batch",
            )

    for task_index, task in enumerate(normalized.get("tasks") or []):
        task_id = task_identity(task, task_index)
        task_invalid = [
            value for value in map(str, task.get("evidence_chunk_ids") or [])
            if value not in valid_chunk_ids
        ]
        for invalid in task_invalid:
            record(
                task_id=task_id, variation_id=None,
                field="task.evidence_chunk_ids", invalid_id=invalid,
                reason="task evidence ID is outside this compiler batch",
            )
        for transformation_index, transformation in enumerate(
            task.get("allowed_transformations") or []
        ):
            for invalid in (
                value for value in map(
                    str, transformation.get("evidence_chunk_ids") or []
                ) if value not in valid_chunk_ids
            ):
                record(
                    task_id=task_id, variation_id=None,
                    field=(
                        "task.allowed_transformations"
                        f"[{transformation_index}].evidence_chunk_ids"
                    ),
                    invalid_id=invalid,
                    reason="transformation evidence ID is outside this compiler batch",
                )
                task_invalid.append(invalid)
        if task_invalid:
            continue
        retained = deepcopy(task)
        retained_variations: list[dict[str, Any]] = []
        for variation_index, variation in enumerate(task.get("allowed_variations") or []):
            variation_id = variation_identity(variation, variation_index)
            invalid_values = [
                value for value in map(
                    str, variation.get("evidence_chunk_ids") or []
                ) if value not in valid_chunk_ids
            ]
            for invalid in invalid_values:
                record(
                    task_id=task_id, variation_id=variation_id,
                    field="variation.evidence_chunk_ids", invalid_id=invalid,
                    reason="variation evidence ID is outside this compiler batch",
                )
            if not invalid_values:
                retained_variations.append(deepcopy(variation))
        retained["allowed_variations"] = retained_variations
        preserved_tasks.append(retained)

    preserved_fitb: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(normalized.get("fitb_bank") or []):
        evidence_id = str(entry.get("evidence_chunk_id") or "")
        if evidence_id not in valid_chunk_ids:
            record(
                task_id=f"fitb_{entry_index + 1}", variation_id=None,
                field="fitb_bank.evidence_chunk_id", invalid_id=evidence_id,
                reason="FITB evidence ID is outside this compiler batch",
            )
        else:
            preserved_fitb.append(deepcopy(entry))
    if diagnostics:
        preserved = {
            "tasks": preserved_tasks,
            "fitb_bank": preserved_fitb,
            "batch_complete": False,
        }
        if normalized.get("assessment_profile") is not None and not profile_invalid:
            preserved["assessment_profile"] = deepcopy(
                normalized.get("assessment_profile")
            )
        raise CatalogBatchValidationError(diagnostics, preserved)
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


def _compact_variation_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Map compact wire aliases onto the authoritative variation fields."""
    data = dict(value)
    if "type" in data and "question_types" not in data:
        qtype = data.pop("type")
        data["question_types"] = [qtype] if qtype else []
    for compact, canonical in (
        ("op", "operation"), ("ev", "evidence_chunk_ids"),
        ("response", "required_response_components"),
        ("vary", "numeric_values_may_change"),
        ("graph", "requires_supplied_graph"),
        ("wp", "student_work_product"), ("mode", "reasoning_mode"),
        ("form", "representation"), ("context", "application"),
        ("equation", "equation_structure"), ("vars", "variables"),
    ):
        if compact in data and canonical not in data:
            data[canonical] = data.pop(compact)
    return data


RESPONSE_FORM_DEFAULTS: dict[str, tuple[str, ...]] = {
    "mcq": ("selection",),
    "true_false": ("boolean_judgment",),
    "fill_in_the_blank": ("short_text",),
    "definition": ("short_text",),
    "why": ("written_justification",),
    "equation": ("worked_solution",),
    "word_problem": ("worked_solution",),
    "essay": ("extended_response",),
}


def _normalize_variations(
    values: Any, *, rejection_diagnostics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize legacy/multi-type angles into grouped, type-specific variations."""
    normalized: dict[str, dict[str, Any]] = {}
    for value in values or []:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if not isinstance(value, dict):
            if rejection_diagnostics is not None:
                rejection_diagnostics.append({
                    "code": "invalid_variation_type",
                    "reason": "Variation must be an object.",
                })
            continue
        value = _compact_variation_payload(value)
        try:
            variation = TaskVariation.model_validate(value).model_dump()
        except ValidationError as exc:
            if rejection_diagnostics is not None:
                rejection_diagnostics.append({
                    "code": "invalid_variation_schema",
                    "reason": "; ".join(
                        f"{'.'.join(map(str, error.get('loc') or ())) or 'variation'}: "
                        f"{error.get('msg', 'invalid value')}"
                        for error in exc.errors()
                    ),
                })
            continue
        question_types = list(dict.fromkeys(variation.get("question_types") or []))
        supplied_group = str(variation.get("variation_group_id") or "")
        group_id = (
            supplied_group if supplied_group.startswith("variation_group_")
            else _stable_id("variation_group", supplied_group) if supplied_group
            else _stable_id(
            "variation_group", variation.get("target"), variation.get("operation"),
            variation.get("concept") or "", variation.get("equation_structure") or "",
            sorted(_normalized(value) for value in variation.get("variables") or []),
            sorted(_normalized(value) for value in variation.get("constraints") or []),
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
            elif not typed.get("required_response_components"):
                typed["required_response_components"] = list(
                    RESPONSE_FORM_DEFAULTS.get(question_type, ())
                )
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
        "concept": task.get("concept"),
        "equation_structure": task.get("equation_structure"),
        "variables": list(
            task.get("variable_parameters") or task.get("variable_givens") or []
        ),
        "constraints": list(
            task.get("validity_constraints") or task.get("complexity_boundaries") or []
        ),
        "givens": list(task.get("required_givens") or []),
        "question_types": list(task.get("primary_question_types") or task.get("allowed_question_types") or []),
        "evidence_chunk_ids": list(task.get("evidence_chunk_ids") or []),
        "reasoning_mode": task.get("reasoning_mode"),
        "student_work_product": task.get("student_work_product"),
        "required_response_components": list(
            task.get("required_response_components") or []
        ),
        "representation": None,
        "application": task.get("scenario_family"),
        "numeric_values_may_change": not bool(task.get("exact_source_instance")),
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


def _inherit_variation_blueprints(
    task: dict[str, Any], values: Any,
) -> list[Any]:
    """Backfill new structural fields from the owning task for old catalogs.

    This is metadata inheritance only.  Python does not interpret equations or
    invent concepts; new compiler output must provide the fields explicitly.
    """
    inherited: list[Any] = []
    for value in values or []:
        if not isinstance(value, dict):
            inherited.append(value)
            continue
        item = dict(value)
        item.setdefault("concept", task.get("concept"))
        if "equation_structure" not in item and "equation" not in item:
            item["equation_structure"] = task.get("equation_structure")
        if "variables" not in item and "vars" not in item:
            item["variables"] = list(
                task.get("variable_parameters") or task.get("variable_givens") or []
            )
        item.setdefault("constraints", list(
            task.get("validity_constraints") or task.get("complexity_boundaries") or []
        ))
        inherited.append(item)
    return inherited


RESPONSE_FORM_CAPABILITIES: dict[str, frozenset[str]] = {
    "mcq": frozenset({"selection"}),
    "true_false": frozenset({"boolean_judgment"}),
    "fill_in_the_blank": frozenset({"short_text"}),
    "definition": frozenset({"short_text"}),
    "why": frozenset({"short_text", "written_justification"}),
    "equation": frozenset({"equation", "worked_solution"}),
    "word_problem": frozenset({"worked_solution"}),
    "essay": frozenset({"extended_response", "written_justification"}),
}


def variation_response_form_compatible(
    variation: dict[str, Any], question_type: str,
) -> bool:
    """Objective response-schema check; semantic work remains Luna's decision.

    Empty component metadata is a backwards-compatible unknown, not permission
    for Python to infer meaning from multilingual work-product prose.
    """
    required = {
        str(value) for value in variation.get("required_response_components") or []
        if str(value)
    }
    if not required:
        return True
    return required.issubset(RESPONSE_FORM_CAPABILITIES.get(question_type, frozenset()))


def _enforce_numeric_variation_contract(
    task: dict[str, Any], variations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ordinary exercise numbers are flexible; exact source data remains fixed."""
    from app.online.models import numeric_variation_allowed
    may_vary = numeric_variation_allowed(task)
    return [
        {
            **variation,
            "numeric_values_may_change": may_vary,
        }
        for variation in variations
    ]


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
            variation.get("concept") or "", variation.get("equation_structure") or "",
            sorted(_normalized(value) for value in variation.get("variables") or []),
            sorted(_normalized(value) for value in variation.get("constraints") or []),
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
    explicit_variations = _normalize_variations(task.get("allowed_variations"))
    primary_variations = []
    for variation in _primary_variations(task):
        blocked = False
        for qtype in variation.get("question_types") or []:
            explicit_for_type = [
                explicit for explicit in explicit_variations
                if qtype in (explicit.get("question_types") or [])
            ]
            if explicit_for_type and not any(
                variation_response_form_compatible(explicit, qtype)
                for explicit in explicit_for_type
            ):
                blocked = True
                break
        if not blocked:
            primary_variations.append(variation)
    scoped = _scope_variations(task, [
        *primary_variations,
        *explicit_variations,
        *_normalize_variations(task.get("permitted_variations")),
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
        if requires_unsupported_supplied_graph(task, variation):
            continue
        if question_type not in (variation.get("question_types") or []):
            continue
        if not variation_response_form_compatible(variation, question_type):
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
        eligible.append(variation)
    return eligible


_GRAPH_EXHIBIT_RE = re.compile(
    r"\b(?:graph(?:s|ed|ing)?|chart|plot|figure|diagram|histogram|image)\b|"
    r"(?:التمثيل\s*(?:البياني|البيانى)|تمثيل\s*(?:بياني|نظام)|"
    r"الرسم\s*(?:البياني|البيانى)|رسم\s*بياني|"
    r"المنحنى|منحنى|المخطط|الصورة|صورة|"
    r"الشكل\s*(?:البياني|البيانى|التالي|الآتي|المرفق|الموضح|أعلاه|أدناه))",
    re.IGNORECASE,
)
# Construction is detected from VERB position only: "plot" and "graph" are also
# nouns, so a bare keyword hit must never mark a reading task as construction.
_GRAPH_CONSTRUCTION_RE = re.compile(
    r"^\s*(?:draw|plot|sketch|construct|create|graph|form|build)\b|"
    r"^\s*(?:ارسم|ارسمي|أنشئ|انشئ|مثّل|مثل|حوّل|حول|شكّل)\b",
    re.IGNORECASE,
)
_GRAPH_INSPECTION_RE = re.compile(
    r"\b(?:read|interpret|inspect|identify|determine|extract|decode|estimate)\b|"
    r"\b(?:find|locate)\b[^.;]{0,40}\b(?:from|on|in)\s+(?:the\s+)?"
    r"(?:graph|chart|plot|figure|diagram)\b|"
    r"(?:اقرأ|اقرا|استخرج|استنتج|"
    r"(?:حدد|فسر)[^.;]{0,30}(?:من|في|على|حسب)\s*(?:الرسم|التمثيل|الشكل)|"
    r"من\s*الرسم|من\s*التمثيل|بالنظر|الموضح(?:ة)?\s*في\s*الشكل|حسب\s*الشكل)",
    re.IGNORECASE,
)
# Conceptual angles talk ABOUT a graph without asking the student to read one.
_GRAPH_CONCEPT_CUE_RE = re.compile(
    r"\b(?:meaning|mean|represent|explain|describe|why|purpose)\b|"
    r"(?:^|[\s،.:؛])(?:معنى|اشرح|وضّح|وضح|علل|دلالة|سبب)(?=$|[\s،.:؛])",
    re.IGNORECASE,
)
_EXPLICIT_READ_RE = re.compile(
    r"\b(?:read|inspect|extract|identify|decode)\b|"
    r"(?:اقرأ|اقرا|استخرج|من\s*الرسم|من\s*التمثيل)",
    re.IGNORECASE,
)


def _graph_resolved_givens(
    task: dict[str, Any], variation: dict[str, Any] | None = None,
) -> list[str]:
    """Resolve variation givens with the same inheritance used when binding."""
    variation = variation or {}
    values = variation.get("givens")
    if not values:
        values = task.get("required_givens") or []
    return [str(value).strip() for value in values if str(value).strip()]


def _graph_givens_are_self_contained(givens: list[str]) -> bool:
    """A graph task is answerable when its data can be carried in the question."""
    return any(
        re.search(r"\d|=", value) for value in givens
    )


def graph_dependency_class(
    task: dict[str, Any], variation: dict[str, Any] | None = None,
) -> str:
    """Classify a task angle as none / construct / reads_supplied / ambiguous.

    ``construct`` and ``ambiguous`` are supported by DraftWork.  Only
    ``reads_supplied`` is ineligible, and only when the data needed to answer is
    not already available as student-visible text.
    """
    variation = variation or {}
    operation = str(
        variation.get("operation") or task.get("task_operation") or ""
    ).replace("_", " ")
    product = str(
        variation.get("student_work_product") or task.get("student_work_product") or ""
    ).replace("_", " ")
    target = str(variation.get("target") or task.get("target") or "")
    givens = " ".join(_graph_resolved_givens(task, variation))
    text = " ".join(part for part in (operation, product, target, givens) if part)
    if not _GRAPH_EXHIBIT_RE.search(text):
        return "none"
    if (
        _GRAPH_CONSTRUCTION_RE.search(operation)
        or _GRAPH_CONSTRUCTION_RE.search(product)
    ):
        return "construct"
    if _EXPLICIT_READ_RE.search(text):
        return "reads_supplied"
    if _GRAPH_CONCEPT_CUE_RE.search(text):
        # "Explain what the slope means on a graph" needs the concept, not the figure.
        return "ambiguous"
    if _GRAPH_INSPECTION_RE.search(text):
        return "reads_supplied"
    return "ambiguous"


def requires_unsupported_supplied_graph(
    task: dict[str, Any], variation: dict[str, Any] | None = None,
) -> bool:
    """Block only an explicitly unavailable supplied visual.

    Keyword classification remains available as a planning/telemetry signal.
    It must not permanently exclude a task; Task Alignment/Luna decides whether
    the assigned evidence actually supplies the needed information.
    """
    variation = variation or {}
    return bool(
        task.get("requires_supplied_graph") is True
        or variation.get("requires_supplied_graph") is True
    )


PLANNER_SCORE_WEIGHTS = {
    "task_reuse": 30, "difficulty": 15, "scenario_diversity": 20,
    "target_diversity": 15, "variation_reuse": 20,
}
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


def _difficulty_distance(level: str | None, requested: str | None) -> int:
    """Tie-break distance only; never an eligibility rule or score weight."""
    if requested not in _DIFFICULTY_ORDER or level not in _DIFFICULTY_ORDER:
        return 0
    return abs(_DIFFICULTY_ORDER[requested] - _DIFFICULTY_ORDER[level])


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
        if not variation_response_form_compatible(variation, question_type):
            continue
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


def catalog_group1_diagnostics(catalog: dict[str, Any]) -> dict[str, Any]:
    """Deterministic Group 1 telemetry: graph, variation and unit posture.

    Counts are computed from the persisted catalog contract only; no LLM input
    and no source text is required.  They are stored on the catalog so planning
    and reporting can explain why the eligible pool shrank.
    """
    from app.online.unit_policy import detect_unit_conflict, unit_policy_for

    graph = {"none": 0, "construct": 0, "ambiguous": 0, "reads_supplied": 0}
    excluded_variations = 0
    zero_variation_tasks = 0
    numeric_variation_tasks = 0
    exact_tasks = 0
    unit_conflict_tasks: list[str] = []
    policies: dict[str, int] = {}
    subject = (catalog.get("assessment_profile") or {}).get("subject")
    for task in catalog.get("tasks") or []:
        declared_count = len(task.get("allowed_variations") or [])
        variations = permitted_task_variations(task)
        if not declared_count:
            zero_variation_tasks += 1
        if task.get("exact_source_instance") is True:
            exact_tasks += 1
        if any(value.get("numeric_values_may_change") is True for value in variations):
            numeric_variation_tasks += 1
        seen: set[str] = set()
        for variation in variations:
            klass = graph_dependency_class(task, variation)
            if klass not in seen:
                graph[klass] += 1
                seen.add(klass)
            if requires_unsupported_supplied_graph(task, variation):
                excluded_variations += 1
        policy = unit_policy_for(
            subject,
            exact_source_instance=task.get("exact_source_instance") is True,
            operation=str(task.get("task_operation") or ""),
            target=str(task.get("target") or ""),
            givens=list(task.get("required_givens") or []) + list(
                task.get("variable_givens") or []
            ),
        )
        policies[policy] = policies.get(policy, 0) + 1
        task_text = " ".join([
            str(task.get("target") or ""),
            *map(str, task.get("required_givens") or []),
            *map(str, task.get("variable_givens") or []),
        ])
        if detect_unit_conflict(task_text):
            unit_conflict_tasks.append(str(task.get("task_id") or ""))
    return {
        "graph": {
            **graph,
            "excluded_variation_count": excluded_variations,
        },
        "variation": {
            "task_count": len(catalog.get("tasks") or []),
            "zero_variation_task_count": zero_variation_tasks,
            "numeric_variation_task_count": numeric_variation_tasks,
            "exact_source_instance_task_count": exact_tasks,
        },
        "units": {
            "subject": subject,
            "policy_counts": policies,
            "unit_conflict_task_ids": sorted(unit_conflict_tasks),
            "unit_conflict_task_count": len(unit_conflict_tasks),
        },
    }



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
                        "required_response_components", "evidence_chunk_ids",
                        "concept", "equation_structure", "variables", "constraints",
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
    givens = _graph_resolved_givens(task, variation)
    mode = variation.get("reasoning_mode") or task.get("reasoning_mode")
    if mode in _RELATIONSHIP_VARIATION_MODES and not (
        task.get("relationship") or givens
    ):
        return False
    if (
        graph_dependency_class(task, variation) == "reads_supplied"
        and not _graph_givens_are_self_contained(givens)
    ):
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
    ), None)


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
        "source_evidence_ids": list(dict.fromkeys(evidence)),
        "canonical_skill": operation,
        "student_work_product": product,
        "required_response_components": list(
            variation.get("required_response_components")
            or task.get("required_response_components") or []
        ),
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
        "requires_supplied_graph": requires_unsupported_supplied_graph(task, variation),
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


def _normalized_chunk_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _source_chunk_manifest(
    chunks: list[dict[str, Any]], source_document_hash: str,
) -> list[dict[str, Any]]:
    """Build document-row-independent identities for the selected evidence.

    UUIDs remain in the manifest only as remapping endpoints.  Identity uses
    normalized content, meaningful parent/section boundaries, provenance, and
    the original PDF hash. Arbitrary retrieval order is deliberately excluded.
    """
    parent_ordinals: dict[str, int] = {}
    manifest: list[dict[str, Any]] = []
    for ordinal, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("child_id") or "").strip()
        if not chunk_id:
            continue
        raw_parent = str(chunk.get("parent_id") or "__unparented__")
        parent_ordinal = parent_ordinals.setdefault(raw_parent, len(parent_ordinals))
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        provenance = {
            key: chunk.get(key, metadata.get(key))
            for key in (
                "page", "page_number", "page_start", "page_end", "section",
                "section_path", "start_char", "end_char", "child_order",
            )
            if chunk.get(key, metadata.get(key)) is not None
        }
        identity = {
            "source_document_hash": source_document_hash,
            "parent_content": _normalized_chunk_text(
                chunk.get("parent_content") or metadata.get("parent_content")
            ),
            "content": _normalized_chunk_text(chunk.get("content")),
            "provenance": provenance,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        manifest.append({
            "chunk_id": chunk_id,
            "fingerprint": fingerprint,
            "ordinal": ordinal,
            "parent_ordinal": parent_ordinal,
        })
    return manifest


def _manifest_id_mapping(
    cached_manifest: list[dict[str, Any]], current_manifest: list[dict[str, Any]],
) -> dict[str, str] | None:
    """Return a complete one-to-one evidence map, rejecting ambiguity."""
    def index(values: list[dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
        result: dict[str, dict[str, Any]] = {}
        for value in values:
            fingerprint = str(value.get("fingerprint") or "")
            chunk_id = str(value.get("chunk_id") or "")
            if not fingerprint or not chunk_id or fingerprint in result:
                return None
            result[fingerprint] = value
        return result

    cached = index(cached_manifest)
    current = index(current_manifest)
    if cached is None or current is None or set(cached) != set(current):
        return None
    return {
        str(cached[fp]["chunk_id"]): str(current[fp]["chunk_id"])
        for fp in cached
    }


_EVIDENCE_ID_KEYS = frozenset({
    "selected_chunk_ids", "evidence_chunk_ids", "source_chunk_ids",
    "supporting_chunk_ids", "effective_evidence_chunk_ids",
    "evidence_ids", "assigned_child_chunk_ids",
})
_EVIDENCE_ID_SCALAR_KEYS = frozenset({
    "evidence_chunk_id", "source_chunk_id", "supporting_chunk_id", "child_id",
})


def _remap_catalog_evidence(
    payload: dict[str, Any], current_manifest: list[dict[str, Any]],
) -> dict[str, Any] | None:
    cached_manifest = payload.get("source_chunk_manifest")
    if not isinstance(cached_manifest, list):
        # Safe lazy compatibility for historical entries is limited to exact
        # document-specific IDs; cross-document migration without source
        # fingerprints would be unverifiable.
        current_ids = {str(row["chunk_id"]) for row in current_manifest}
        return deepcopy(payload) if set(map(str, payload.get("selected_chunk_ids") or [])) == current_ids else None
    mapping = _manifest_id_mapping(cached_manifest, current_manifest)
    if mapping is None:
        return None

    unknown_reference = False

    def walk(value: Any, key: str = "") -> Any:
        nonlocal unknown_reference
        if isinstance(value, dict):
            return {child_key: walk(child, child_key) for child_key, child in value.items()}
        if isinstance(value, list):
            if key in _EVIDENCE_ID_KEYS:
                remapped = []
                for item in value:
                    source_id = str(item)
                    if source_id not in mapping:
                        unknown_reference = True
                        remapped.append(item)
                    else:
                        remapped.append(mapping[source_id])
                return remapped
            return [walk(item, key) for item in value]
        if key in _EVIDENCE_ID_SCALAR_KEYS and value is not None:
            source_id = str(value)
            if source_id not in mapping:
                unknown_reference = True
                return value
            return mapping[source_id]
        return value

    remapped = walk(deepcopy(payload))
    if unknown_reference:
        return None
    remapped["source_chunk_manifest"] = deepcopy(current_manifest)
    remapped["selected_chunk_ids"] = [str(row["chunk_id"]) for row in current_manifest]
    return remapped


def catalog_compiler_fingerprint() -> str:
    """Hash only the code, prompts, schemas, and config that build a Catalog."""
    generation_sources = (
        _catalog_prompt,
        _compact_variation_payload,
        _normalize_variations,
        _primary_variations,
        _expand_compact_task,
        _scope_variations,
        _enforce_numeric_variation_contract,
        _repair_catalog_numerical_contract,
        _normalize_transformations,
        _finalize_catalog,
        _migrate_legacy_method_restrictions,
        variation_response_form_compatible,
        eligible_task_variations,
        permitted_task_variations,
        canonical_task_ids,
        canonical_fitb_entry,
    )
    generation_models = (
        CompactTaskVariation,
        CompactCatalogTask,
        CompactCatalogBatchResponse,
        CompactCatalogBatchWithProfileResponse,
        AssessmentCatalogResponse,
        CatalogTaskDraft,
        FITBEntryDraft,
        TaskVariation,
        AssessmentProfile,
        TransformationGrant,
    )
    payload = {
        "catalog_contract_version": CATALOG_CONTRACT_VERSION,
        "catalog_batching_version": CATALOG_BATCHING_VERSION,
        "catalog_version": config.ASSESSMENT_CATALOG_VERSION,
        "compiler_version": config.ASSESSMENT_COMPILER_VERSION,
        "response_capabilities": {
            key: sorted(value) for key, value in RESPONSE_FORM_CAPABILITIES.items()
        },
        "response_defaults": RESPONSE_FORM_DEFAULTS,
        "compiler_models": {
            "stem": config.STEM_REVIEW_MODEL,
            "non_stem": config.DEEPSEEK_MODEL,
        },
        "sources": [
            inspect.getsource(value) for value in generation_sources
        ],
        "schemas": [
            json.dumps(value.model_json_schema(), sort_keys=True, separators=(",", ":"))
            for value in generation_models
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()


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
    ordered_membership = list(map(str, selected_chunk_ids))
    payload = {
        "processed_document_key": processed_document_key,
        "selected_source_membership": (
            ordered_membership if all(value.startswith("chunkfp:") for value in ordered_membership)
            else sorted(set(ordered_membership))
        ),
        "catalog_version": config.ASSESSMENT_CATALOG_VERSION,
        "catalog_compiler_fingerprint": catalog_compiler_fingerprint(),
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
        "catalog_compiler_fingerprint": catalog_compiler_fingerprint(),
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
        or payload.get("catalog_compiler_fingerprint") != catalog_compiler_fingerprint()
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
    source_document_hash: str | None = None,
) -> bool:
    """Fail closed when a cached catalog references another retrieval snapshot."""
    selected = set(map(str, selected_chunk_ids))
    if (
        payload.get("processed_document_key") != processed_document_key
        or (source_document_hash and payload.get("source_document_hash") != source_document_hash)
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
        and task.get("supported_difficulties") in ([], [native])
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
    return payload.get("raw") if (
        payload.get("compiler_cache_key") == key
        and payload.get("catalog_compiler_fingerprint") == catalog_compiler_fingerprint()
    ) else None


def _write_compiler_cache(key: str, raw: dict[str, Any]) -> None:
    path = _compiler_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "compiler_cache_key": key,
        "catalog_compiler_fingerprint": catalog_compiler_fingerprint(),
        "raw": raw,
    }
    handle, temporary_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _compatible_compiler_seed(
    processed_document_key: str, selected_chunk_ids: list[str],
    current_manifest: list[dict[str, Any]] | None = None,
    source_document_hash: str | None = None,
) -> dict[str, Any] | None:
    """Return a same-version, same-evidence Catalog without an LLM call.

    Cache-key internals can change while the public Catalog contract remains
    compatible. Re-finalizing that payload under the current deterministic
    contract avoids needless recompilation. Incompatible contract changes must
    bump ``ASSESSMENT_CATALOG_VERSION``.
    """
    root = Path(config.ASSESSMENT_CATALOG_CACHE_DIR)
    selected = sorted(set(map(str, selected_chunk_ids)))
    current_fingerprint = catalog_compiler_fingerprint()
    for path in sorted(root.glob("[0-9a-f][0-9a-f]/*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        # Compatibility migration is allowed only for the exact compiler
        # fingerprint. Legacy entries without the field must recompile once.
        if payload.get("catalog_compiler_fingerprint") != current_fingerprint:
            continue
        source_version = str(payload.get("catalog_version") or "")
        candidate = (
            _remap_catalog_evidence(payload, current_manifest)
            if current_manifest is not None else deepcopy(payload)
        )
        if candidate is None:
            continue
        migratable_version = source_version in {
            config.ASSESSMENT_CATALOG_VERSION,
            "catalog-v12-grouped-type-variations",
            CATALOG_CONTRACT_VERSION,
            "catalog-v13-numerical-givens-contract-live-20261030",
        }
        migrated_tasks: list[dict[str, Any]] = []
        migration_failed = False
        for task in candidate.get("tasks") or []:
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
            and candidate.get("processed_document_key") == processed_document_key
            # Historical compatible catalogs predate the explicit PDF hash.
            # Their exact processed key plus verified one-to-one evidence
            # manifest is the only safe migration path; a conflicting stored
            # PDF hash is never accepted.
            and (
                not source_document_hash
                or candidate.get("source_document_hash") in {None, "", source_document_hash}
            )
            and sorted(set(map(str, candidate.get("selected_chunk_ids") or []))) == selected
            and candidate.get("assessment_profile")
            and candidate["assessment_profile"].get("subject")
            and candidate["assessment_profile"].get("domain") in {"STEM", "NON_STEM"}
            and bool(candidate.get("tasks"))
            and isinstance(candidate.get("fitb_bank"), list)
            and not migration_failed
            and all(_cached_task_contract_valid(task) for task in migrated_tasks)
        ):
            return {
                "assessment_profile": candidate["assessment_profile"],
                "tasks": migrated_tasks,
                "fitb_bank": candidate["fitb_bank"],
                "catalog_compilation_complete": True,
                "recovered_batch_count": int(candidate.get("recovered_batch_count") or 0),
            }
    return None


def _catalog_evidence_registry(
    batch: list[dict[str, Any]], batch_id: str,
) -> str:
    """Render one unambiguous parent/child registry for the compiler."""
    lines = [f"CATALOG BATCH: {batch_id}", "", "ALLOWED CHILD CHUNKS:"]
    current_parent: str | None = None
    for index, chunk in enumerate(batch, 1):
        child_id = str(chunk.get("child_id") or "").strip()
        parent_id = str(chunk.get("parent_id") or "").strip()
        parent_key = parent_id or f"missing-parent-for:{child_id}"
        if parent_key != current_parent:
            current_parent = parent_key
            lines.extend([
                "",
                f"[PARENT SECTION: {parent_id or 'MISSING (child isolated)'}]",
                f"Parent title: {str(chunk.get('parent_title') or '').strip() or '(untitled)'}",
                "Parent IDs are context only and MUST NOT be used as evidence IDs.",
            ])
        lines.extend([
            "",
            f"[CHUNK {index}]",
            f"Evidence ID: {child_id}",
            f"Child title: {str(chunk.get('chunk_title') or chunk.get('title') or '').strip() or '(untitled)'}",
            "Content:",
            str(chunk.get("content") or "").strip(),
        ])
    return "\n".join(lines)


def _catalog_retry_feedback(
    diagnostics: list[dict[str, Any]], preserved: dict[str, Any],
) -> str:
    if not diagnostics:
        return ""
    return """\
COMPILER VALIDATION FAILURE — TARGETED CORRECTION REQUIRED

The JSON from the previous attempt was schema-valid but contained invalid evidence references.
The independently valid content below is already accepted and MUST NOT be regenerated, renamed, or
silently changed. Return only corrected/replacement tasks, variations, or FITB entries needed to
repair the rejected portions and complete coverage. If no allowed chunk genuinely supports a
rejected task or variation, discard it and produce a source-supported alternative only when needed
for coverage. Never replace an invalid ID with an arbitrary allowed ID.

Structured validation diagnostics:
{diagnostics}

Preserved valid compiler content:
{preserved}

Set batch_complete=true only when the returned corrections, together with the preserved content,
form a complete batch. Every corrected reference will be revalidated before acceptance.
""".format(
        diagnostics=json.dumps(diagnostics, ensure_ascii=False, indent=2),
        preserved=json.dumps(preserved, ensure_ascii=False, indent=2),
    )


def _catalog_task_merge_key(task: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_normalized(task.get(field)) for field in (
        "knowledge_anchor", "concept", "task_operation", "target",
        "student_work_product",
    ))


def _catalog_variation_merge_key(variation: dict[str, Any]) -> tuple[str, ...]:
    explicit = str(variation.get("variation_id") or "").strip()
    if explicit:
        return ("id", explicit)
    return ("semantic",) + tuple(_normalized(variation.get(field)) for field in (
        "target", "operation", "type", "student_work_product",
    ))


def _merge_catalog_batch_fragments(
    preserved: dict[str, Any] | None, candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge retry corrections without duplicating accepted task identities."""
    left = deepcopy(preserved or {})
    right = deepcopy(candidate or {})
    merged: dict[str, Any] = {
        "tasks": [], "fitb_bank": [],
        "batch_complete": bool(right.get("batch_complete")),
    }
    profile = right.get("assessment_profile") or left.get("assessment_profile")
    if profile is not None:
        merged["assessment_profile"] = deepcopy(profile)
    task_positions: dict[tuple[str, ...], int] = {}
    for task in [*(left.get("tasks") or []), *(right.get("tasks") or [])]:
        key = _catalog_task_merge_key(task)
        if key not in task_positions:
            task_positions[key] = len(merged["tasks"])
            merged["tasks"].append(deepcopy(task))
            continue
        existing = merged["tasks"][task_positions[key]]
        existing_variations = list(existing.get("allowed_variations") or [])
        variation_positions = {
            _catalog_variation_merge_key(value): index
            for index, value in enumerate(existing_variations)
        }
        for variation in task.get("allowed_variations") or []:
            variation_key = _catalog_variation_merge_key(variation)
            if variation_key in variation_positions:
                current = existing_variations[variation_positions[variation_key]]
                if current != variation:
                    # A correction may change evidence, but must not silently
                    # change an already accepted semantic variation contract.
                    semantic_fields = (
                        "target", "operation", "type", "student_work_product",
                        "reasoning_mode", "required_response_components",
                    )
                    if any(current.get(field) != variation.get(field) for field in semantic_fields):
                        raise ValueError(
                            "catalog variation identity collision during retry merge"
                        )
                    existing_variations[variation_positions[variation_key]] = deepcopy(variation)
            else:
                variation_positions[variation_key] = len(existing_variations)
                existing_variations.append(deepcopy(variation))
        existing["allowed_variations"] = existing_variations
    seen_fitb: set[tuple[str, str, str]] = set()
    for entry in [*(left.get("fitb_bank") or []), *(right.get("fitb_bank") or [])]:
        key = (
            _normalized(entry.get("answer")),
            str(entry.get("evidence_chunk_id") or ""),
            _normalized(entry.get("clue_basis")),
        )
        if key not in seen_fitb:
            seen_fitb.add(key)
            merged["fitb_bank"].append(deepcopy(entry))
    return merged


def _catalog_prompt(
    context: str,
    language: str,
    valid_chunk_ids: list[str],
    *,
    domain_type: str,
    subject: str,
    include_profile: bool = True,
    batch_id: str = "batch_1",
    retry_diagnostics: list[dict[str, Any]] | None = None,
    preserved_content: dict[str, Any] | None = None,
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
    retry_feedback = _catalog_retry_feedback(
        retry_diagnostics or [], preserved_content or {},
    )
    return f"""You are the one-time assessment compiler for selected textbook content.
Read the complete selection once and create a reusable semantic assessment catalog.

{retry_feedback}

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
For that primary angle, also emit equation, parameters, and constraints as its concise generation
blueprint. equation is a general symbolic mathematical relationship when one meaningfully exists in
the evidence; otherwise it is null. parameters identifies the replaceable variables and their roles.
constraints lists the validity, solvability, sign, domain, unit, integrality, or other rules the
Generator must preserve. Do not use explanatory prose as equation and do not insert source-example
numbers into the general structure. These primary fields have the same meaning as concept,
equation, vars, and constraints inside an alternative variation.
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

difficulty is an informational conceptual prior for this task, not a promise that the task can only
be assessed at that level. Do not compute difficulty floors, ceilings, or supported ranges.

allowed_transformations lists additional cognitive transformations the selected source explicitly
teaches and that may be inserted into, or removed from, a realization of this task without changing
its assessed operation. Treat these as difficulty levers: list every distinct lever the source
actually demonstrates, including combinations it teaches (for example distribute then combine like
terms, variables on both sides, substitution, fractions/decimals, or interpreting an intermediate
result). Emit one object per transformation with a short snake_case `transformation` name, the exact
`evidence_chunk_ids` that teach it, and a short `support` quote or paraphrase from those chunks.
Do not list a transformation merely because it is mathematically possible, and do not split one
operation into overlapping names to inflate the list. Use an empty array when the source teaches no
such transformation. Never list a generic, unsupported, or invented transformation.

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

For every variation, set requires_supplied_graph=true only when the student must inspect, read,
or interpret a graph/plot/chart/figure/table/image supplied by the exam and the data cannot be
carried in the question text. Set it false for student graph construction from complete
equations/data and for conceptual graph questions. DraftWork cannot display supplied visuals, so
downstream eligibility excludes true values. Distinguish the three cases rather than classifying
from the word "graph":
- student constructs the graph from complete equations/data (ارسم التمثيل البياني لـ y = 2x + 3):
  requires_supplied_graph=false, answerable=true; the equation/data belong in required_givens;
- conceptual graph question that needs no visual (ما معنى الميل؟ / find the slope of y = 3x + 5):
  requires_supplied_graph=false, answerable=true;
- student must read values off a supplied graph/table/figure (اقرأ الإحداثيات من التمثيل البياني /
  identify the vertices from the graph shown below): requires_supplied_graph=true unless this
  variation's givens already contain the exact numerical data the question will display.
When a concept is only supported through the third case, add a source-supported text-answerable
alternative angle for the same concept instead of omitting the concept.

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
  constraints, or source-specific values that every valid instance must preserve. Concrete exercise
  numbers and worked-example values do not belong here unless interpreting that exact data is the
  task; put them in source_example_givens (the replaceable source instance) instead;
- source_example_givens carries the literal values of the source example when the task is generative.
  It is reference material for the generator, never a requirement the student must reproduce;
- variable_givens describes illustrative quantities or parameters for provenance and prompt clarity
  (for example total matches, distance, speed, or elapsed time). Ordinary non-exact exercise numbers
  may change coherently even when this list is empty; it is not an authorization gate.
  A source table, graph, dataset, or measurement is not variable merely because it contains numbers.
  When the student must interpret that particular source data, keep its values or structure in
  required_givens and leave variable_givens empty. When the task supports constructing a coherent
  synthetic dataset, its values may vary while its mathematical and scientific constraints remain;
- variations is an array of bounded objects. Each object names one exact source-supported target,
  operation, givens, and exactly one question_types value. Every variation also contains a concise
  concept, variables, and constraints blueprint. concept is one sentence describing the knowledge or
  relationship assessed. variables names the parameters that may change and briefly states their
  roles. constraints states the rules needed to keep a generated instance valid and solvable. For a
  quantitative variation, equation is the concise general symbolic relationship when one is
  meaningful and source-supported (for example ax+b=c); otherwise equation is null. Derive this
  relationship from the supplied evidence rather than selecting from a hardcoded formula list. Do
  not put worked-example numbers into equation, variables, or constraints unless the exact source
  instance is immutable. For a non-quantitative variation, express a short conceptual/reasoning
  structure through concept, operation, target, and constraints instead. Optionally use
  reasoning_mode, student_work_product, form for
  representation, context for application, and evidence_chunk_ids for a narrower evidence subset.
  Declare required_response_components structurally for the student-visible response: selection,
  boolean_judgment, short_text, written_justification, worked_solution, equation, or
  extended_response. Do not encode this requirement only in free-form work_product prose. A
  true_false variation can require boolean_judgment but cannot require written_justification,
  because its student-visible schema has no justification field; use why or essay for that work.
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
- exact_source_instance is true only when assessing the exact worked example is itself the task. It
  constrains VALUES, never the number of authorized angles: still declare every source-supported
  alternative angle (interpretation, explanation, prediction, construction) as its own variation.
  For a generative ordinary exercise set exact_source_instance=false;
- Ordinary mathematical exercises are generative whenever exact_source_instance=false. Set
  exact_source_instance=true when the particular source data, table, measurements, constants, or
  exact worked instance are themselves what the student must interpret; those values cannot vary.
  Preserve invariant laws and constraints in required_givens;
- Units are not automatically immutable. Record the literal source value and unit (for example
  "48 ft") in source_example_givens when the task is generative. For mathematics, a coherent unit
  change is allowed when the mathematical concept is unchanged. For physics and other sciences,
  prefer the source unit where it carries the method or dimensional meaning; a change is allowed
  only when dimensionally valid and derivable from the source. Set exact_source_instance=true when
  the exact measured data (including its unit) is what the student must interpret.

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
Catalog batch ID: {batch_id}
Valid source chunk IDs: {json.dumps(valid_chunk_ids, ensure_ascii=False)}

Structured evidence registry (this is the complete selected source for this batch):
{context}"""


def _finalize_catalog(
    raw: dict[str, Any], *, key: str, processed_document_key: str,
    selected_chunk_ids: list[str], usage: dict[str, Any] | None,
    source_chunk_manifest: list[dict[str, Any]] | None = None,
    source_document_hash: str | None = None,
) -> dict[str, Any]:
    valid_ids = set(map(str, selected_chunk_ids))
    finalized: list[dict[str, Any]] = []
    finalized_by_id: dict[str, dict[str, Any]] = {}
    rejection_diagnostics: list[dict[str, Any]] = []
    for task_index, raw_task in enumerate(raw.get("tasks") or []):
        if not isinstance(raw_task, dict):
            rejection_diagnostics.append({
                "stage": "catalog_finalization", "task_index": task_index,
                "code": "invalid_task_type", "reason": "Task must be an object.",
            })
            continue
        raw_task = _expand_compact_task(raw_task)
        raw_task, contract_diagnostics = _repair_catalog_numerical_contract(raw_task)
        if any(item.get("disposition") == "quarantined" for item in contract_diagnostics):
            rejection_diagnostics.append({
                "stage": "catalog_finalization", "task_index": task_index,
                "code": "invalid_numerical_contract",
                "reason": "Task numerical contract was quarantined.",
                "details": contract_diagnostics,
            })
            continue
        if contract_diagnostics:
            raw_task["contract_diagnostics"] = contract_diagnostics
        evidence = [
            str(value) for value in raw_task.get("evidence_chunk_ids") or []
            if str(value) in valid_ids
        ]
        if not evidence or raw_task.get("text_answerable") is not True:
            rejection_diagnostics.append({
                "stage": "catalog_finalization", "task_index": task_index,
                "code": (
                    "invalid_evidence_reference" if not evidence
                    else "task_not_text_answerable"
                ),
                "reason": (
                    "Task has no evidence reference in the selected child chunks."
                    if not evidence else "Task is not explicitly text-answerable."
                ),
            })
            continue
        parent_evidence = set(evidence)
        valid_variations: list[dict[str, Any]] = []
        variation_rejections: list[dict[str, Any]] = []
        for variation in _normalize_variations(
            _inherit_variation_blueprints(
                raw_task, raw_task.get("allowed_variations"),
            ),
            rejection_diagnostics=variation_rejections,
        ):
            explicit = variation.get("evidence_chunk_ids")
            if explicit is not None:
                explicit_ids = {str(value) for value in explicit}
                # One bad optional angle must not discard an otherwise complete task.
                if not explicit_ids or not explicit_ids.issubset(parent_evidence & valid_ids):
                    variation_rejections.append({
                        "code": "invalid_variation_evidence_reference",
                        "variation_id": variation.get("variation_id"),
                        "reason": (
                            "Variation evidence must be a non-empty subset of the task's "
                            "selected child-chunk evidence."
                        ),
                    })
                    continue
                variation["evidence_chunk_ids"] = list(dict.fromkeys(map(str, explicit)))
            valid_variations.append(variation)
        rejection_diagnostics.extend({
            "stage": "catalog_finalization", "task_index": task_index,
            **diagnostic,
        } for diagnostic in variation_rejections)
        raw_task["allowed_variations"] = valid_variations
        identity = canonical_task_ids(raw_task)
        if identity is None:
            rejection_diagnostics.append({
                "stage": "catalog_finalization", "task_index": task_index,
                "code": "incomplete_task_identity",
                "reason": "Task identity requires knowledge_anchor, concept, and task_operation.",
            })
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
            existing["allowed_transformations"] = _normalize_transformations([
                *(existing.get("allowed_transformations") or []),
                *_normalize_transformations(
                    raw_task.get("allowed_transformations"), parent_evidence,
                ),
            ])
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
            # Conflicting compiler batches fail closed: one source-bound
            # declaration is enough to protect this shared task.
            existing["exact_source_instance"] = bool(
                existing.get("exact_source_instance")
                or raw_task.get("exact_source_instance")
            )
            existing["allowed_variations"] = _enforce_numeric_variation_contract(
                existing, existing.get("allowed_variations") or [],
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
            rejection_diagnostics.append({
                "stage": "catalog_finalization", "task_index": task_index,
                "code": "missing_supported_question_type",
                "reason": "Task declares no supported generator question type.",
            })
            continue
        native_difficulty = str(raw_task.get("native_difficulty") or "medium")
        if native_difficulty not in {"easy", "medium", "hard"}:
            native_difficulty = "medium"
        # Difficulty is a compiler prior only; it never implies a supported range.
        supported_difficulties: list[str] = []
        exact_source_instance = bool(raw_task.get("exact_source_instance", False))
        valid_variations = _enforce_numeric_variation_contract(
            {**raw_task, "exact_source_instance": exact_source_instance},
            valid_variations,
        )
        finalized_task = {
            **raw_task,
            **identity,
            "evidence_chunk_ids": list(dict.fromkeys(evidence)),
            "allowed_question_types": allowed,
            "primary_question_types": primary_allowed,
            "supported_difficulties": supported_difficulties,
            "native_difficulty": native_difficulty,
            "allowed_transformations": _normalize_transformations(
                raw_task.get("allowed_transformations"), parent_evidence,
            ),
            "prerequisites": list(raw_task.get("prerequisites") or []),
            # Exactness constrains VALUES, not the set of authorized angles.
            # Numeric variation is already disabled below for exact tasks, so
            # keeping alternative (non-numeric) angles preserves text-answerable
            # graph/interpretation alternatives instead of deleting them.
            "allowed_variations": valid_variations,
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
            rejection_diagnostics.append({
                "stage": "catalog_finalization", "task_index": task_index,
                "code": "no_complete_type_specific_variation",
                "reason": (
                    "No declared question type has a complete, source-grounded compatible variation."
                ),
            })
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
    result = {
        "cache_key": key,
        "catalog_version": config.ASSESSMENT_CATALOG_VERSION,
        "catalog_compiler_fingerprint": catalog_compiler_fingerprint(),
        "processed_document_key": processed_document_key,
        "source_document_hash": source_document_hash or processed_document_key,
        "selected_chunk_ids": sorted(valid_ids),
        "source_chunk_manifest": deepcopy(source_chunk_manifest or []),
        "assessment_profile": profile,
        "tasks": finalized,
        "fitb_bank": fitb_bank,
        "tasks_by_question_type": tasks_by_question_type,
        "compiler_usage": usage or {},
        "catalog_compilation_complete": raw.get("catalog_compilation_complete", True),
        "recovered_batch_count": int(raw.get("recovered_batch_count") or 0),
        "catalog_rejections": rejection_diagnostics,
    }
    result["group1_diagnostics"] = catalog_group1_diagnostics(result)
    return result


def build_task_usage_index(
    plans: list[dict[str, Any]], generated_exams: list[dict[str, Any]] | None = None,
    *, excluding_slot_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a current usage view from authoritative plan and question state."""
    usage: dict[str, dict[str, Any]] = {}
    slot_task: dict[str, str] = {}
    for plan in plans or []:
        for values in (plan.get("items") or {}).values():
            for item in values or []:
                task_id = str(item.get("fitb_entry_id") or item.get("task_id") or "")
                slot_id = str(item.get("slot_id") or "")
                if excluding_slot_id and slot_id == str(excluding_slot_id):
                    continue
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


def rank_task_variation_candidates(
    candidates: list[tuple[Any, ...]], *, question_type: str,
    requested_difficulty: str, task_usage: dict[str, int],
) -> list[tuple[Any, ...]]:
    """Apply the one task-selection order used by planning and Recovery.

    Callers must supply candidates that already passed source, response-form,
    graph, metadata, mathematical-constraint, and difficulty-policy eligibility.
    Tuple shapes may be ``(task, level, variation)`` or
    ``(position, task, level, variation)``.
    """
    def parts(entry: tuple[Any, ...]) -> tuple[dict[str, Any], str, dict[str, Any]]:
        if len(entry) == 4:
            return entry[1], str(entry[2]), entry[3]
        return entry[0], str(entry[1]), entry[2]

    def key(entry: tuple[Any, ...]) -> tuple[Any, ...]:
        task, level, variation = parts(entry)
        task_id = str(task.get("task_id") or "")
        variation_id = str(variation.get("variation_id") or "")
        return (
            int(
                (task_usage.get(task_id) or {}).get("count", 0)
                if isinstance(task_usage.get(task_id), dict)
                else task_usage.get(task_id, 0)
            ),
            _difficulty_distance(level, requested_difficulty),
            task_id,
            variation_id,
        )

    # Question type is an eligibility filter, never a score. Most callers have
    # already materialized eligible variations; keep this defensive check for
    # shared use by backup and replacement paths.
    eligible = []
    for entry in candidates:
        _task, _level, variation = parts(entry)
        supported = variation.get("question_types") or []
        if isinstance(supported, str):
            supported = [supported]
        if question_type and supported and question_type not in set(map(str, supported)):
            continue
        eligible.append(entry)
    return sorted(eligible, key=key)


def _balanced_catalog_batches(
    chunks: list[dict[str, Any]], limit: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Return deterministic, parent-isolated batches in original source order.

    A parent remains intact when it fits the child budget. Oversized parents
    are split deterministically by child order. Missing parent metadata is
    explicit and isolates that child; no textual parent inference is attempted.
    """
    if not chunks:
        return []
    max_children = max(
        1, int(limit if limit is not None else config.CATALOG_BATCH_MAX_CHILDREN)
    )

    def source_order(indexed: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        index, chunk = indexed
        child_order = chunk.get("child_order", chunk.get("chunk_order"))
        try:
            numeric_order = int(child_order)
        except (TypeError, ValueError):
            numeric_order = None
        return (
            0 if numeric_order is not None else 1,
            numeric_order if numeric_order is not None else index,
            index,
        )

    # Retrieval preserves caller selection order, which differs between the UI,
    # resumed jobs, and workers. Canonicalize before deriving batches and the
    # cache identity so the same processed selection cannot trigger recompilation.
    ordered_chunks = [chunk for _index, chunk in sorted(
        enumerate(chunks), key=source_order,
    )]
    parent_groups: dict[str, list[dict[str, Any]]] = {}
    parent_order: list[str] = []
    selected_ids: list[str] = []
    seen_ids: set[str] = set()
    for index, chunk in enumerate(ordered_chunks):
        child_id = str(chunk.get("child_id") or "").strip()
        if not child_id:
            raise CatalogBatchAssignmentError(
                f"selected chunk at source position {index} has no child_id"
            )
        if child_id in seen_ids:
            raise CatalogBatchAssignmentError(
                f"selected child ID appears more than once: {child_id}"
            )
        seen_ids.add(child_id)
        selected_ids.append(child_id)
        parent_id = str(chunk.get("parent_id") or "").strip()
        parent_key = parent_id or f"__missing_parent__:{child_id}"
        if not parent_id:
            logger.warning(
                "Catalog child has no parent metadata; isolating child | child_id=%s",
                child_id,
            )
        if parent_key not in parent_groups:
            parent_groups[parent_key] = []
            parent_order.append(parent_key)
        parent_groups[parent_key].append(chunk)
    batches: list[list[dict[str, Any]]] = []
    for parent_key in parent_order:
        group = parent_groups[parent_key]
        batches.extend(
            group[offset:offset + max_children]
            for offset in range(0, len(group), max_children)
        )
    assigned = [
        str(chunk.get("child_id") or "") for batch in batches for chunk in batch
    ]
    if set(assigned) != set(selected_ids) or len(assigned) != len(selected_ids):
        raise CatalogBatchAssignmentError(
            "catalog batch assignment omitted or added selected children"
        )
    if any(not batch for batch in batches) or len(set(assigned)) != len(assigned):
        raise CatalogBatchAssignmentError(
            "catalog batch assignment contains an empty batch or duplicate child"
        )
    # Parent isolation necessarily changes the global order when source rows for
    # different parents are interleaved. The required ordering invariant is the
    # original child order *within each parent*, not the flattened global order.
    assigned_by_parent: dict[str, list[str]] = {key: [] for key in parent_order}
    for batch in batches:
        batch_parent_keys = {
            (str(chunk.get("parent_id") or "").strip()
             or f"__missing_parent__:{str(chunk.get('child_id') or '').strip()}")
            for chunk in batch
        }
        if len(batch_parent_keys) != 1:
            raise CatalogBatchAssignmentError(
                "catalog batch contains children from more than one parent"
            )
        parent_key = next(iter(batch_parent_keys))
        assigned_by_parent.setdefault(parent_key, []).extend(
            str(chunk.get("child_id") or "") for chunk in batch
        )
    for parent_key in parent_order:
        expected = [
            str(chunk.get("child_id") or "") for chunk in parent_groups[parent_key]
        ]
        if assigned_by_parent.get(parent_key) != expected:
            raise CatalogBatchAssignmentError(
                f"catalog batch assignment changed child order for parent {parent_key}"
            )
    return batches


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
    processed_key = str(state.get("processed_document_key") or state.get("document_id") or "")
    source_document_hash = str(state.get("source_document_hash") or processed_key).lower()
    source_manifest = _source_chunk_manifest(chunks, source_document_hash)
    membership = [
        f"chunkfp:{row['fingerprint']}" for row in source_manifest
    ]
    compiler_identity = {
        **compiler_identity,
        "batching": CATALOG_BATCHING_VERSION,
        "source_membership_sha256": hashlib.sha256(
            json.dumps(sorted(membership), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    key = catalog_cache_key(source_document_hash, sorted(membership), compiler_identity)
    cached = _read_cache(key)
    if cached is not None:
        cached = _remap_catalog_evidence(cached, source_manifest)
    if cached is not None and not _cached_catalog_matches_evidence(
        cached, processed_key, selected, source_document_hash
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
    compiler_key = compiler_cache_key(source_document_hash, sorted(membership), compiler_identity)
    raw = _read_compiler_cache(compiler_key)
    if raw is not None and raw.get("catalog_compilation_complete") is not True:
        # Leave the file untouched for diagnostics. It is not a reusable result.
        raw = None
    compiler_cache_hit = raw is not None
    compatibility_cache_hit = False
    if raw is None:
        raw = _compatible_compiler_seed(
            processed_key, selected, source_manifest, source_document_hash
        )
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
            batch_id = f"batch_{batch_index + 1}"
            evidence_registry = _catalog_evidence_registry(batch, batch_id)
            if set(batch_ids) != {
                str(chunk.get("child_id") or "") for chunk in batch
            }:
                raise CatalogBatchAssignmentError(
                    f"{batch_id} allowed evidence IDs do not match supplied children"
                )
            batch_manifest = _source_chunk_manifest(batch, source_document_hash)
            batch_membership = sorted(
                f"chunkfp:{row['fingerprint']}" for row in batch_manifest
            )
            batch_key = compiler_cache_key(source_document_hash, batch_membership, {
                **compiler_identity, "batch_index": batch_index,
                "batch_count": len(batches), "partition": CATALOG_BATCHING_VERSION,
            })
            cached_batch = _read_compiler_cache(batch_key)
            if cached_batch is not None:
                try:
                    validated_cache = _validate_completed_batch(
                        cached_batch, response_model=response_model,
                        valid_chunk_ids=set(batch_ids),
                        batch_id=batch_id,
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
            retry_diagnostics: list[dict[str, Any]] = []
            preserved_content: dict[str, Any] = {}
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
                            evidence_registry,
                            state.get("document_language") or "en", batch_ids,
                            domain_type=domain_type, subject=subject,
                            include_profile=include_profile,
                            batch_id=batch_id,
                            retry_diagnostics=retry_diagnostics,
                            preserved_content=preserved_content,
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
                    normalized_result = response_model.model_validate(result).model_dump()
                    combined_result = _merge_catalog_batch_fragments(
                        preserved_content, normalized_result,
                    )
                    value = _validate_completed_batch(
                        combined_result, response_model=response_model,
                        valid_chunk_ids=set(batch_ids),
                        batch_id=batch_id,
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
                except CatalogBatchValidationError as exc:
                    preserved_content = _merge_catalog_batch_fragments(
                        preserved_content, exc.preserved,
                    )
                    retry_diagnostics = list(exc.diagnostics)
                    last_reason = (
                        "invalid completion contract: "
                        f"{exc}; invalid evidence IDs: "
                        + ", ".join(sorted({
                            str(item.get("invalid_evidence_id") or "")
                            for item in exc.diagnostics
                        }))
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
        source_chunk_manifest=source_manifest,
        source_document_hash=source_document_hash,
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


def _normalize_transformations(
    values: Any, valid_chunk_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Keep only source-grounded, deduplicated transformation grants."""
    valid = set(valid_chunk_ids) if valid_chunk_ids is not None else None
    normalized: dict[str, dict[str, Any]] = {}
    for value in values or []:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        try:
            grant = TransformationGrant.model_validate(value)
        except ValidationError:
            continue
        name = " ".join(grant.transformation.split()).casefold()
        if not name:
            continue
        evidence = [
            chunk_id for chunk_id in grant.evidence_chunk_ids
            if not valid or chunk_id in valid
        ]
        if not evidence:
            continue
        entry = normalized.setdefault(name, {
            "transformation": name,
            "evidence_chunk_ids": [],
            "support": grant.support,
        })
        entry["evidence_chunk_ids"] = list(dict.fromkeys([
            *entry["evidence_chunk_ids"], *evidence,
        ]))
    return list(normalized.values())


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


def activate_backup_assignment(
    item: dict[str, Any], catalog: dict[str, Any], *, slot_id: str,
    difficulty: str, subject: str | None = None,
) -> dict[str, Any] | None:
    """Materialize the next preplanned backup without generating it upfront."""
    backups = deepcopy(item.get("backups") or [])
    tasks = {str(task.get("task_id") or ""): task for task in catalog.get("tasks") or []}
    while backups:
        ref = backups.pop(0)
        task = tasks.get(str(ref.get("task_id") or ""))
        if task is None:
            continue
        qtype = _slot_question_type(slot_id)
        variation = next(
            (value for value in permitted_task_variations(task, qtype)
             if str(value.get("variation_id") or "") == str(ref.get("variation_id") or "")),
            None,
        )
        if variation is None:
            continue
        replacement = materialize_plan_item(
            task, slot_id, str(ref.get("difficulty") or difficulty),
            selected_variation_id=variation.get("variation_id"), subject=subject,
        )
        replacement["backups"] = backups
        replacement["backup_activation"] = {
            "task_id": ref.get("task_id"),
            "variation_id": ref.get("variation_id"),
            "purpose": ref.get("purpose"),
            "revision": int(item.get("assignment_revision") or 0) + 1,
        }
        replacement["assignment_revision"] = int(item.get("assignment_revision") or 0) + 1
        replacement["task_was_previously_used"] = True
        return replacement
    return None


def materialize_plan_item(
    task: dict[str, Any],
    slot_id: str,
    difficulty: str,
    *,
    selected_variation_id: str | None = None,
    subject: str | None = None,
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
        "source_example_givens": list(task.get("source_example_givens") or []),
        "allowed_transformations": deepcopy(task.get("allowed_transformations") or []),
        "complexity_boundaries": list(task.get("complexity_boundaries") or []),
        "exact_source_instance": bool(task.get("exact_source_instance", False)),
        "text_answerable": task.get("text_answerable") is True,
        "requires_supplied_graph": requires_unsupported_supplied_graph(task),
        "reasoning_mode": task.get("reasoning_mode"),
        "requires_quantitative_validation": bool(
            task.get("requires_quantitative_validation")
            or task.get("reasoning_mode") in {
                "calculation", "symbolic_manipulation", "derivation_relationship",
                "law_or_theorem_application", "estimation_reasonableness",
            }
            or any(word in str(task.get("task_operation") or "").casefold() for word in (
                "calculate", "compute", "solve", "derive", "balance", "evaluate", "convert",
            ))
        ),
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
        "subject": subject if subject is not None else task.get("subject"),
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


def materialize_backup_candidate(
    catalog: dict[str, Any], backup: dict[str, Any], *, slot_id: str,
    difficulty: str, subject: str | None = None, revision: int = 1,
) -> dict[str, Any] | None:
    """Materialize one backup ref into the existing replacement-candidate shape."""
    task = next((
        value for value in catalog.get("tasks") or []
        if str(value.get("task_id") or "") == str(backup.get("task_id") or "")
    ), None)
    if task is None:
        return None
    qtype = _slot_question_type(slot_id)
    variation = next((
        value for value in permitted_task_variations(task, qtype)
        if str(value.get("variation_id") or "") == str(backup.get("variation_id") or "")
    ), None)
    if variation is None:
        return None
    candidate = materialize_plan_item(
        task, slot_id, str(backup.get("difficulty") or difficulty),
        selected_variation_id=variation.get("variation_id"), subject=subject,
    )
    candidate["selected_replacement_task_id"] = task.get("task_id")
    candidate["requested_difficulty"] = difficulty
    candidate["actual_difficulty"] = str(backup.get("difficulty") or difficulty)
    candidate["assignment_revision"] = revision
    candidate["backup_activation"] = {
        "task_id": backup.get("task_id"),
        "variation_id": backup.get("variation_id"),
        "purpose": backup.get("purpose"),
        "revision": revision,
    }
    return candidate


def build_planner_index(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a compact, read-only Planner view without another model call."""
    index: list[dict[str, Any]] = []
    for task in catalog.get("tasks") or []:
        variations = permitted_task_variations(task)
        if not variations:
            continue
        index.append({
            "id": task.get("task_id"),
            "concept": task.get("concept"),
            "op": task.get("task_operation"),
            "target": task.get("target"),
            "diff": task.get("native_difficulty"),
            "v": [
                {
                    "id": variation.get("variation_id"),
                    "type": (variation.get("question_types") or [None])[0],
                    "op": variation.get("operation"),
                    "target": variation.get("target"),
                    "response": variation.get("required_response_components") or [],
                    "idea": variation.get("student_work_product"),
                    "ev": variation.get("evidence_chunk_ids") or [],
                    "vary": bool(variation.get("numeric_values_may_change")),
                    "graph": variation.get("requires_supplied_graph") is True,
                }
                for variation in variations
            ],
        })
    return index


def _backup_assignment_refs(
    candidates: list[tuple[int, dict[str, Any], str, dict[str, Any]]],
    primary_task_id: str, primary_variation_id: str, requested_difficulty: str,
    task_usage: dict[str, int], limit: int = 2,
    represented_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Choose backups with the same three-criterion order as primary tasks."""
    eligible = [
        entry for entry in candidates
        if not (
            str(entry[1].get("task_id") or "") == primary_task_id
            and str(entry[3].get("variation_id") or "") == primary_variation_id
        )
    ]
    qtype = str(
        next((
            value for entry in eligible
            for value in (entry[3].get("question_types") or [])
        ), "")
    )
    ranked = rank_task_variation_candidates(
        eligible, question_type=qtype, requested_difficulty=requested_difficulty,
        task_usage=task_usage,
    )
    backups: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _position, task, level, variation in ranked:
        key = (
            str(task.get("task_id") or ""),
            str(variation.get("variation_id") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        backups.append({
            "task_id": key[0],
            "variation_id": key[1],
            "idea_to_test": variation.get("student_work_product") or task.get("student_work_product"),
            "operation": variation.get("operation") or task.get("task_operation"),
            "target": variation.get("target") or task.get("target"),
            "difficulty": level,
            "evidence_ids": list(
                variation.get("evidence_chunk_ids") or task.get("evidence_chunk_ids") or []
            ),
        })
        if len(backups) >= limit:
            break
    return backups


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
                native_level = _native_difficulty(candidate) or "medium"
                pool.extend(
                    (position, candidate, native_level, variation)
                    for variation in variations
                )
            candidate_pool_by_type[qtype] = pool
        for slot in slots:
            compatible[slot] = candidate_pool_by_type[slot[0]]
        # Slot traversal is deterministic only. Candidate priority is exclusively
        # question-type eligibility, global Task-ID usage, then difficulty.
        ordered_slots = sorted(slots, key=lambda slot: (slot[0], slot[1]))
        assignment: dict[
            tuple[str, int], tuple[dict[str, Any], str, dict[str, Any]]
        ] = {}
        # One shared order only: type eligibility (already filtered), global
        # Task ID usage, difficulty distance, then deterministic IDs.
        live_task_usage = dict(task_usage_count)
        for slot in ordered_slots:
            qtype = slot[0]
            ranked = rank_task_variation_candidates(
                compatible[slot], question_type=qtype,
                requested_difficulty=difficulty, task_usage=live_task_usage,
            )
            if not ranked:
                continue
            _position, chosen, level, variation = ranked[0]
            assignment[slot] = (chosen, level, variation)
            task_id = str(chosen.get("task_id") or "")
            live_task_usage[task_id] = live_task_usage.get(task_id, 0) + 1

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
                    subject=profile.get("subject"),
                )
                item["backups"] = _backup_assignment_refs(
                    compatible.get((qtype, index)) or candidate_pool_by_type.get(qtype, []),
                    task_id, str(variation.get("variation_id") or ""), difficulty,
                    task_usage_count,
                    represented_items=[
                        previous for prior in plans
                        for values in (prior.get("items") or {}).values()
                        for previous in values or []
                    ] + [
                        previous for values in items.values()
                        for previous in values or []
                    ],
                )
                item["task_usage_count_at_selection"] = task_usage_count.get(task_id, 0)
                item["difficulty_distance_at_selection"] = _difficulty_distance(
                    level, difficulty
                )
                item["requested_difficulty"] = difficulty
                item["actual_difficulty"] = level
                item["native_difficulty_prior"] = level
                if difficulty in {"easy", "medium", "hard"} and level != difficulty:
                    item["difficulty_design"] = {
                        "status": "pending",
                        "policy_version": "difficulty-design-v1",
                        "native_prior": level,
                        "requested": difficulty,
                    }
                elif item.get("allowed_transformations"):
                    item["difficulty_design"] = {
                        "status": "pending",
                        "policy_version": "difficulty-design-v1",
                        "native_prior": level,
                        "requested": difficulty,
                    }
                else:
                    item["difficulty_design"] = {
                        "status": "not_required",
                        "policy_version": "difficulty-design-v1",
                        "native_prior": level,
                        "requested": difficulty,
                    }
                task_usage_count[task_id] = task_usage_count.get(task_id, 0) + 1
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
    represented_items = [
        item
        for plan in plans
        for values in (plan.get("items") or {}).values()
        for item in values
        if str(item.get("slot_id") or "") != slot_id
        and (active_slot_ids is None or str(item.get("slot_id") or "") in active_slot_ids)
    ]
    profile = catalog.get("assessment_profile") or {}
    valid_chunks = set(map(str, catalog.get("selected_chunk_ids") or [])) or None
    candidates: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for task in catalog.get("tasks") or []:
        level = _native_difficulty(task) or "medium"
        variations = eligible_task_variation_candidates(
            task, qtype, assessment_profile=profile, valid_chunk_ids=valid_chunks,
        )
        if (
            variations
            and str(task.get("task_id") or "") not in excluded
            and str(task.get("task_id")) not in current_task_ids
        ):
            candidates.extend((task, level, variation) for variation in variations)
    if not candidates:
        return None
    ranked = rank_task_variation_candidates(
        candidates, question_type=qtype, requested_difficulty=requested_difficulty,
        task_usage=build_task_usage_index(plans, excluding_slot_id=slot_id),
    )
    if not ranked:
        return None
    task, level, variation = ranked[0]
    replacement = materialize_plan_item(
        task, slot_id, level,
        selected_variation_id=str(variation.get("variation_id") or ""),
    )
    replacement["requested_difficulty"] = requested_difficulty
    replacement["actual_difficulty"] = level
    replacement["difficulty"] = level
    replacement["native_difficulty_prior"] = level
    replacement["difficulty_design"] = {
        "status": (
            "pending" if level != requested_difficulty
            and requested_difficulty in {"easy", "medium", "hard"}
            else "not_required"
        ),
        "policy_version": "difficulty-design-v1",
        "native_prior": level,
        "requested": requested_difficulty,
    }
    return replacement


def apply_difficulty_fallback_metadata(
    item: dict[str, Any], requested: str, achieved: str, reason: str,
    *, strategy: str = "source_constrained_fallback",
) -> dict[str, Any]:
    """Apply the one authoritative, checkpoint-safe fallback representation."""
    item.update({
        "requested_difficulty": requested,
        "actual_difficulty": achieved,
        "difficulty": achieved,
        "difficulty_status": "accepted_source_constraint",
        "difficulty_fallback": True,
        "difficulty_fallback_reason": reason,
        "difficulty_adaptation": {
            **dict(item.get("difficulty_adaptation") or {}),
            "status": "accepted_source_constraint",
            "from_difficulty": achieved,
            "to_difficulty": requested,
            "achieved_difficulty": achieved,
            "difficulty_floor": achieved,
            "further_adjustment_possible": False,
            "strategy": strategy,
            "policy_version": "difficulty-adaptation-v2",
        },
    })
    return item


def catalog_exhaustion_replacement(
    catalog: dict[str, Any], plans: list[dict[str, Any]], slot_id: str,
    qtype: str, requested_difficulty: str, *,
    forced_duplicate_slot_ids: list[str] | None = None,
    forced_duplicate_question_ids: list[str] | None = None,
    excluded_task_ids: set[str] | None = None,
    used_fitb_answer_keys: set[str] | None = None,
    active_slot_ids: set[str] | None = None,
    prefer_task_id: str | None = None,
    exclude_variation_ids: set[str] | None = None,
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
    # Legacy staged/backup selection is deliberately disabled. All non-FITB
    # replacements flow through the shared rank below.
    strict = None
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
        level = _native_difficulty(task) or "medium"
        variations = eligible_task_variation_candidates(
            task, qtype, assessment_profile=profile, valid_chunk_ids=valid_chunks,
        )
        if (
            variations
            and str(task.get("task_id") or "") not in {
                str(value) for value in (excluded_task_ids or set()) if str(value)
            }
        ):
            compatible.extend((task, level, variation) for variation in variations)

    # Shared selection contract for Recovery and Catalog exhaustion. Task and
    # variation preferences from an LLM or preplanned backup cannot bypass it.
    usage_count: dict[str, int] = {}
    for item in all_items:
        task_id = str(item.get("task_id") or "")
        if task_id:
            usage_count[task_id] = usage_count.get(task_id, 0) + 1
    ranked = rank_task_variation_candidates(
        compatible, question_type=qtype,
        requested_difficulty=requested_difficulty, task_usage=usage_count,
    )
    if not ranked:
        return None
    task, level, variation = ranked[0]
    selected_task_id = str(task.get("task_id") or "")
    reused = usage_count.get(selected_task_id, 0) > 0
    replacement = materialize_plan_item(
        task, slot_id, level,
        selected_variation_id=str(variation.get("variation_id") or ""),
    )
    same_task_slots = sorted(
        str(item.get("slot_id") or "") for item in all_items
        if str(item.get("slot_id") or "") != slot_id
        and str(item.get("task_id") or "") == selected_task_id
    )
    replacement.update({
        "requested_difficulty": requested_difficulty,
        "actual_difficulty": level,
        "difficulty": level,
        "native_difficulty_prior": level,
        "catalog_replacement_stage": (
            "shared_rank_unused_task" if not reused
            else "shared_rank_least_used_task"
        ),
        "task_reused": reused,
        "reused_due_to_catalog_exhaustion": reused,
        "reuse_reason": "no_unused_compatible_task" if reused else None,
        "reuse_mode": "intentional_practice_variation" if reused else False,
        "reused_task_id": selected_task_id if reused else None,
        "task_usage_count": usage_count.get(selected_task_id, 0),
        "forced_duplicate_slot_ids": list(dict.fromkeys([
            *same_task_slots,
            *(str(value) for value in (forced_duplicate_slot_ids or []) if str(value)),
        ])),
        "forced_duplicate_question_ids": list(dict.fromkeys(
            str(value) for value in (forced_duplicate_question_ids or []) if str(value)
        )),
        "task_inventory": _inventory_after_reservation(
            inventory, selected_task_id, reused=reused,
        ),
        "difficulty_design": {
            "status": (
                "pending" if level != requested_difficulty
                and requested_difficulty in {"easy", "medium", "hard"}
                else "not_required"
            ),
            "policy_version": "difficulty-design-v1",
            "native_prior": level,
            "requested": requested_difficulty,
        },
    })
    return attach_fitb_answer_inventory(replacement, task)
