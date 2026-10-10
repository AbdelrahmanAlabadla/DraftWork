from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

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
)
from app.logging_conf import get_logger
from app.online.semantic_keys import formula_semantic_key

logger = get_logger("ASSESSMENT_CATALOG")


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
            "reused_task_after_catalog_exhaustion", "plan_review_intentional_reuse",
            "semantic_plan_repair_intentional_reuse",
        }
        and str(item.get("reused_task_id") or "") == task_id
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
            "plan_review_intentional_reuse",
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
    compatible_ids = {
        str(task.get("task_id"))
        for task in catalog.get("tasks") or []
        if task.get("task_id")
        and qtype in (task.get("allowed_question_types") or [])
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
    topic: str = Field(min_length=1, max_length=160)
    concept: str = Field(min_length=1, max_length=200)
    relationship: str | None = Field(max_length=240)
    task_operation: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    student_work_product: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    reasoning_mode: ReasoningMode
    allowed_question_types: list[QuestionType] = Field(min_length=1)
    evidence_chunk_ids: list[NonEmptyString] = Field(min_length=1, max_length=4)
    required_givens: list[NonEmptyString] = Field(max_length=12)
    scenario_family: str | None = Field(max_length=160)
    text_answerable: bool


class FITBEntryDraft(StrictOutputModel):
    """Compact source-grounded vocabulary candidate owned by the compiler."""

    answer: str = Field(min_length=1, max_length=80)
    evidence_chunk_id: NonEmptyString
    clue_basis: str = Field(min_length=1, max_length=320)


class AssessmentCatalogResponse(StrictOutputModel):
    assessment_profile: AssessmentProfile
    tasks: list[CatalogTaskDraft] = Field(min_length=1)
    fitb_bank: list[FITBEntryDraft] = Field(default_factory=list, max_length=128)


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


def _recover_catalog_checkpoint(text: str) -> dict[str, Any] | None:
    objects = _completed_json_objects(text)
    profile = next((
        value for value in objects
        if isinstance(value.get("question_types"), dict)
        and isinstance(value.get("domains"), list)
        and value.get("subject")
        and value.get("domain") in {"STEM", "NON_STEM"}
    ), None)
    tasks = [
        value for value in objects
        if value.get("knowledge_anchor") and value.get("task_operation")
        and isinstance(value.get("evidence_chunk_ids"), list)
    ]
    fitb_bank = [
        value for value in objects
        if value.get("answer") and value.get("evidence_chunk_id")
        and value.get("clue_basis") and not value.get("task_operation")
    ]
    if profile is None or len(tasks) < 28:
        return None
    logger.warning(
        "Recovered assessment catalog checkpoint | complete_tasks=%d | discarded_tail=true",
        len(tasks),
    )
    return {"assessment_profile": profile, "tasks": tasks, "fitb_bank": fitb_bank}


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
        "student_work_product": item.get("student_work_product"),
        "target": item.get("target"),
        "reasoning_mode": item.get("reasoning_mode"),
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
            "task", coverage_id, item.get("task_operation"),
            item.get("student_work_product"), item.get("target"),
            item.get("reasoning_mode"), relationship_basis or "",
            item.get("required_givens") or [], item.get("scenario_family") or "",
        ),
        "task_family_id": _stable_id(
            "family", coverage_id, item.get("task_operation"),
            item.get("student_work_product"), item.get("target"),
            relationship_basis or item.get("concept"), item.get("reasoning_mode"),
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


def catalog_cache_key(
    processed_document_key: str,
    selected_chunk_ids: list[str],
    compiler_identity: dict[str, str] | None = None,
) -> str:
    payload = {
        "processed_document_key": processed_document_key,
        "selected_chunk_ids": sorted(set(map(str, selected_chunk_ids))),
        "catalog_version": config.ASSESSMENT_CATALOG_VERSION,
        "catalog_logic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        inspect.getsource(_catalog_prompt),
        json.dumps(AssessmentCatalogResponse.model_json_schema(), sort_keys=True),
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
    ):
        return None
    return payload


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
    """Migrate a prior derived V1 catalog into the new two-layer cache."""
    root = Path(config.ASSESSMENT_CATALOG_CACHE_DIR)
    selected = sorted(set(map(str, selected_chunk_ids)))
    for path in sorted(root.glob("[0-9a-f][0-9a-f]/*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if (
            payload.get("catalog_version") == config.ASSESSMENT_CATALOG_VERSION
            and payload.get("processed_document_key") == processed_document_key
            and payload.get("selected_chunk_ids") == selected
            and payload.get("assessment_profile")
            and payload["assessment_profile"].get("subject")
            and payload["assessment_profile"].get("domain") in {"STEM", "NON_STEM"}
            and len(payload.get("tasks") or []) >= 28
            and isinstance(payload.get("fitb_bank"), list)
        ):
            return {
                "assessment_profile": payload["assessment_profile"],
                "tasks": payload["tasks"],
                "fitb_bank": payload["fitb_bank"],
            }
    return None


def _catalog_prompt(
    context: str,
    language: str,
    valid_chunk_ids: list[str],
    *,
    domain_type: str,
    subject: str,
) -> str:
    return f"""You are the one-time assessment compiler for selected textbook content.
Read the complete selection once and create a reusable semantic assessment catalog.

Title generation already classified this source. Copy these authoritative values exactly into the
assessment profile; do not reclassify or override them:
- assessment_profile.subject: {subject}
- assessment_profile.domain: {"STEM" if domain_type == "stem" else "NON_STEM"}

The catalog is an ontology, not an exam. Extract distinct atomic assessment tasks supported by the
text. One task is exactly: concept + narrow target + student operation + expected work product.
knowledge_anchor identifies the underlying fact, concept, relationship, or skill and may be shared
by several tasks. Sharing a knowledge anchor, concept, or evidence chunk does not make two tasks
duplicates when the student operation, target, or expected work product is genuinely different.

Create separate tasks when the source supports materially different student work, including:
- identify a term from its description versus state its definition;
- identify a named phenomenon versus explain its cause or mechanism;
- identify a relationship versus explain that relationship;
- conceptual explanation versus symbolic manipulation or derivation;
- symbolic manipulation versus a numerical calculation with grounded givens.
For example, wavelength may support identify_term_from_definition -> term, state_definition ->
definition, explain_relationship -> explanation, and symbolic_rearrangement -> equation. These
tasks share a knowledge area but assess different work.

Merge only paraphrases or aliases of the same concept + target + operation + work product. Do not
create question-type copies of the same task, and do not create repeated identify/define/calculate
tasks with superficial wording, scenario, or numerical changes. Each target must be one narrow
assessable output rather than a bundled topic. One evidence chunk may support multiple atomic tasks
when it explicitly supports their distinct operations and targets.
Every task_operation used by a task must also appear under every corresponding question type in
assessment_profile.question_types.task_operations.

Choose allowed_question_types semantically for each atomic task; this field is authoritative for
downstream allocation. Do not decide suitability from the operation label or reasoning_mode alone.
Use the concept, target, relationship, evidence, and required givens together. A type is suitable
only when the task naturally supports its student-facing work:
- mcq: one defensible best answer plus three plausible same-concept distractors;
- true_false: one clear, source-grounded, unambiguous claim;
- definition: the meaning of a term or concept;
- why: cause, reason, justification, mechanism, or relationship reasoning;
- equation: symbolic manipulation, derivation, balancing, solving, or other equation-centred work;
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

Cover the selection broadly before adding depth to any concept. Do not target a fixed task count.
Return every meaningful atomic task needed for broad assessment coverage, up to the schema limit,
but never pad the catalog with paraphrases. Before returning, audit every important term, concept,
relationship, explanation, symbolic skill, and grounded calculation in the selection for a distinct
concept + target + operation + work-product combination. When supported, include at least
four definitional tasks, three symbolic equation tasks, and four contextual calculation tasks.
Do not satisfy symbolic coverage by forcing every Equation task into rearranging a supplied final
relationship. Preserve the evidence-supported operation: constructing, deriving, combining,
balancing, solving, proving, and rearranging are distinct only when the source supports that work.
Do not create several superficial rearrangements merely to meet coverage.

Treat worked examples as evidence for a method, relationship, constraints, and supported kinds of
givens. Unless reproducing that exact source instance is itself the narrow assessment target, do not
encode the worked example's complete scenario + exact values + requested result as the catalog task.
For a general application task, express the transferable operation and target, and include only
source-required constants or constraints in required_givens. Never invent an unsupported scientific
fact or context.
Include only tasks answerable from
text. Evidence IDs must come from the supplied list and every required numerical/symbolic given must
be present in those chunks. Equation tasks must ask for symbolic manipulation/derivation; word
problems must ask for a contextual numerical solution. Keep task metadata concise and language
independent enough for reuse. Return no more than {config.CATALOG_MAX_TASKS} tasks.

Return one root JSON object with exactly these top-level fields:
- assessment_profile: the complete profile required by the schema;
- tasks: the array of atomic assessment task objects.
- fitb_bank: the compact array of answer/evidence/clue objects.
Never return either field outside that root object and never return a bare task array.

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
    seen_task_ids: set[str] = set()
    for raw_task in raw.get("tasks") or []:
        if not isinstance(raw_task, dict):
            continue
        evidence = [
            str(value) for value in raw_task.get("evidence_chunk_ids") or []
            if str(value) in valid_ids
        ]
        if not evidence or not raw_task.get("text_answerable", True):
            continue
        identity = canonical_task_ids(raw_task)
        if identity is None:
            continue
        task_id = str(identity["task_id"])
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        reasoning_mode = str(raw_task.get("reasoning_mode") or "")
        allowed = list(dict.fromkeys(raw_task.get("allowed_question_types") or []))
        allowed = [
            qtype for qtype in allowed
            if qtype in {"mcq", "true_false", "definition",
                         "why", "equation", "word_problem", "essay"}
        ]
        if not allowed:
            # A task without a compiler-declared compatible representation is structurally
            # unusable; do not invent a semantic compatibility decision here.
            continue
        supported_difficulties = ["easy", "medium"]
        if reasoning_mode in {
            "conceptual_reasoning", "calculation", "symbolic_manipulation",
            "law_or_theorem_application", "derivation_relationship",
            "logical_proof_reasoning", "comparison_prediction", "error_analysis",
        }:
            supported_difficulties.append("hard")
        finalized.append({
            **raw_task,
            **identity,
            "evidence_chunk_ids": list(dict.fromkeys(evidence)),
            "allowed_question_types": allowed,
            "supported_difficulties": supported_difficulties,
        })
        if len(finalized) >= config.CATALOG_MAX_TASKS:
            break
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
            supporting = [task for task in finalized if qtype in task["allowed_question_types"]]
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
                    str(task["task_operation"]) for task in supporting
                ))
                type_profile["evidence_chunk_ids"] = list(dict.fromkeys(
                    chunk_id for task in supporting for chunk_id in task["evidence_chunk_ids"]
                ))[:8]
            else:
                type_profile["support"] = "unsupported"
                type_profile["task_operations"] = []
                type_profile["evidence_chunk_ids"] = []
    return {
        "cache_key": key,
        "catalog_version": config.ASSESSMENT_CATALOG_VERSION,
        "processed_document_key": processed_document_key,
        "selected_chunk_ids": sorted(valid_ids),
        "assessment_profile": profile,
        "tasks": finalized,
        "fitb_bank": fitb_bank,
        "compiler_usage": usage or {},
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
    key = catalog_cache_key(processed_key, selected, compiler_identity)
    cached = _read_cache(key)
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
    compiler_cache_hit = raw is not None
    client = None
    if raw is None:
        client = create_assessment_compiler_client(domain_type=domain_type)
        try:
            raw = request_structured(
            client,
            _catalog_prompt(
                state.get("planner_context") or state.get("context") or "",
                state.get("document_language") or "en",
                selected,
                domain_type=domain_type,
                subject=subject,
            ),
            response_model=AssessmentCatalogResponse,
            schema_name="assessment_catalog",
            agent="assessment_compiler",
            operation="assessment_catalog",
            item_count=min(config.CATALOG_MAX_TASKS, max(32, len(chunks) * 2)),
            system_prompt="Compile a complete, non-redundant assessment ontology grounded only in the source.",
            temperature=0.1,
            max_tokens=18000,
            )
        except (DeepSeekIncompleteError, DeepSeekValidationError) as exc:
            recovered = _recover_catalog_checkpoint(exc.partial_output)
            if recovered is None:
                raise
            raw = recovered
        _write_compiler_cache(compiler_key, raw if isinstance(raw, dict) else {})
    if isinstance(raw, dict):
        profile = raw.setdefault("assessment_profile", {})
        if isinstance(profile, dict):
            profile["subject"] = subject
            profile["domain"] = "STEM" if domain_type == "stem" else "NON_STEM"
    payload = _finalize_catalog(
        raw if isinstance(raw, dict) else {}, key=key,
        processed_document_key=processed_key, selected_chunk_ids=selected,
        usage=getattr(client, "last_usage", None) if client is not None else {},
    )
    if not payload["tasks"]:
        # Compatibility for legacy/local test doubles that still return the
        # former planner contract. Production structured output cannot omit
        # tasks because AssessmentCatalogResponse requires them.
        logger.warning("Assessment compiler returned the legacy planner contract; using planner fallback")
        return {
            "assessment_catalog": {},
            "assessment_catalog_key": key,
            "assessment_catalog_cache_hit": False,
            "assessment_compiler_cache_hit": compiler_cache_hit,
        }
    _write_cache(key, payload)
    logger.info("Assessment catalog compiled | key=%s | tasks=%d", key, len(payload["tasks"]))
    return {
        "assessment_catalog": payload,
        "assessment_catalog_key": key,
        "assessment_catalog_cache_hit": False,
        "assessment_compiler_cache_hit": compiler_cache_hit,
        "planner_context": "",
    }


def _difficulty_for(task: dict[str, Any], requested: str) -> str | None:
    supported = list(task.get("supported_difficulties") or [])
    if requested != "mix":
        # Difficulty is a writing constraint for direct questions. Any
        # grounded task can be simplified to easy; harder requests still need
        # explicit catalog support for deeper evidence.
        if requested == "easy" and supported:
            return "easy"
        return requested if requested in supported else None
    for level in ("medium", "easy", "hard"):
        if level in supported:
            return level
    return None


def materialize_plan_item(task: dict[str, Any], slot_id: str, difficulty: str) -> dict[str, Any]:
    evidence = list(task.get("evidence_chunk_ids") or [])
    relationship = task.get("relationship")
    idea = f"{task.get('student_work_product')}: {task.get('target')}"
    if task.get("reasoning_mode") == "derivation_relationship":
        idea = f"Derive and establish the relationship: {idea}"
    return {
        "slot_id": slot_id,
        "task_id": task["task_id"],
        "task_family_id": task.get("task_family_id") or task["task_id"],
        "coverage_id": task["coverage_id"],
        "concept_id": task.get("concept_id"),
        "relationship_id": task.get("relationship_id"),
        "source_chunk_id": evidence[0],
        "support_chunk_ids": evidence[1:],
        "topic": task.get("topic") or task.get("concept"),
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
        "text_answerable": True,
        "reasoning_mode": task.get("reasoning_mode"),
        "difficulty": difficulty,
        "idea_to_test": idea,
        "concept_to_test": idea,
    }


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
    source_tasks = list(catalog.get("tasks") or [])
    plans: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    task_usage_count: dict[str, int] = {}
    coverage_usage_count: dict[str, int] = {}
    fitb_usage_count: dict[str, int] = {}
    for model_number in range(1, num_models + 1):
        items: dict[str, list[dict[str, Any]]] = {qtype: [] for qtype, _ in tasks}
        slots = [
            (qtype, index)
            for qtype, count in tasks if qtype != "fill_in_the_blank"
            for index in range(1, count + 1)
        ]
        compatible: dict[tuple[str, int], list[tuple[int, dict[str, Any], str]]] = {}
        for slot in slots:
            qtype, _index = slot
            compatible[slot] = [
                (position, candidate, level)
                for position, candidate in enumerate(source_tasks)
                for level in [_difficulty_for(candidate, difficulty)]
                if level is not None and qtype in (candidate.get("allowed_question_types") or [])
            ]
        ordered_slots = sorted(slots, key=lambda slot: (len(compatible[slot]), slot[0], slot[1]))
        assignment: dict[tuple[str, int], tuple[dict[str, Any], str]] = {}
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

        def resource_id(candidate: dict[str, Any]) -> str:
            return str(
                candidate.get("coverage_id")
                or candidate.get("task_family_id")
                or candidate.get("task_id") or ""
            )

        def match(slot: tuple[str, int], seen: set[str]) -> bool:
            ranked = sorted(
                compatible[slot],
                key=lambda entry: (
                    coverage_usage_count.get(str(entry[1].get("coverage_id") or ""), 0) > 0,
                    task_usage_count.get(str(entry[1].get("task_id") or ""), 0) > 0,
                    len(entry[1].get("allowed_question_types") or []),
                    source_scenario_frequency.get(_normalized(entry[1].get("scenario_family")), 0),
                    len(entry[1].get("evidence_chunk_ids") or []),
                    entry[0],
                ),
            )
            for _position, candidate, level in ranked:
                resource = resource_id(candidate)
                if not resource or resource in seen:
                    continue
                seen.add(resource)
                previous_slot = resource_owner.get(resource)
                if previous_slot is not None and not match(previous_slot, seen):
                    continue
                resource_owner[resource] = slot
                assignment[slot] = (candidate, level)
                return True
            return False

        for slot in ordered_slots:
            match(slot, set())

        # Exact task IDs are the inventory unit. Family/coverage uniqueness is
        # preferred by the matching pass above, but it must not create an
        # artificial shortage while a compatible task exists. Fill remaining
        # slots from unused task IDs first, then enter explicit reuse mode.
        assigned_ids = {
            str(task.get("task_id") or "") for task, _level in assignment.values()
        }
        assigned_coverage_ids = {
            str(task.get("coverage_id") or "") for task, _level in assignment.values()
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
            _position, chosen, level = min(
                pool,
                key=lambda entry: (
                    coverage_usage_count.get(str(entry[1].get("coverage_id") or ""), 0),
                    task_usage_count.get(str(entry[1].get("task_id") or ""), 0),
                    str(entry[1].get("task_id") or ""),
                ),
            )
            assignment[slot] = (chosen, level)
            assigned_ids.add(str(chosen.get("task_id") or ""))
            if chosen.get("coverage_id"):
                assigned_coverage_ids.add(str(chosen["coverage_id"]))

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
                chosen, level = selected
                task_id = str(chosen.get("task_id") or "")
                reused = task_usage_count.get(task_id, 0) > 0
                item = materialize_plan_item(
                    chosen, f"m{model_number}_{qtype}_{index}", level
                )
                task_usage_count[task_id] = task_usage_count.get(task_id, 0) + 1
                coverage_id = str(chosen.get("coverage_id") or "")
                if coverage_id:
                    coverage_usage_count[coverage_id] = coverage_usage_count.get(coverage_id, 0) + 1
                compatible_ids = {
                    str(candidate.get("task_id") or "")
                    for candidate in source_tasks
                    if candidate.get("task_id")
                    and qtype in (candidate.get("allowed_question_types") or [])
                    and _difficulty_for(candidate, difficulty) is not None
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
    candidates: list[tuple[dict[str, Any], str]] = []
    for task in catalog.get("tasks") or []:
        level = _difficulty_for(task, requested_difficulty)
        if (
            level and qtype in (task.get("allowed_question_types") or [])
            and str(task.get("task_id") or "") not in excluded
            and str(task.get("task_id")) not in current_task_ids
            and str(task.get("task_family_id") or task.get("task_id")) not in current_family_ids
            and str(task.get("task_id")) not in used
            and str(task.get("task_family_id") or task.get("task_id")) not in used_families
        ):
            candidates.append((task, level))
    if not candidates:
        return None
    task, level = min(
        candidates,
        key=lambda entry: (
            int(bool(
                qtype == "fill_in_the_blank"
                and fitb_answer_keys(entry[0]) & represented_fitb_answers
            )),
            *_replacement_representation_rank(
                entry[0], represented_items, include_task_usage=False,
            ),
        ),
    )
    return materialize_plan_item(task, slot_id, level)


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
        and qtype in (task.get("allowed_question_types") or [])
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

    compatible: list[tuple[dict[str, Any], str]] = []
    for task in catalog.get("tasks") or []:
        level = _difficulty_for(task, requested_difficulty)
        if (
            level and qtype in (task.get("allowed_question_types") or [])
            and str(task.get("task_id") or "") not in {
                str(value) for value in (excluded_task_ids or set()) if str(value)
            }
        ):
            compatible.append((task, level))

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
        (task, level) for task, level in compatible
        if str(task.get("task_id") or "") != current_task_id
        and str(task.get("task_id") or "") not in used_task_ids
    ]
    if unused and not prefer_used_fresh_fitb_answer:
        task, level = min(
            unused, key=lambda entry: coverage_rank(entry[0], include_task_usage=False)
        )
        replacement = materialize_plan_item(task, slot_id, level)
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
        (usage_count.get(str(task.get("task_id") or ""), 0), task, level)
        for task, level in compatible
        if usage_count.get(str(task.get("task_id") or ""), 0) > 0
        and str(task.get("task_id") or "") != current_task_id
    ]
    if reused:
        _count, task, level = min(
            reused, key=lambda value: coverage_rank(value[1], include_task_usage=True)
        )
        reused_task_id = str(task.get("task_id") or "")
        same_task_slots = [
            str(item.get("slot_id") or "") for item in all_items
            if str(item.get("slot_id") or "") != slot_id
            and str(item.get("task_id") or "") == reused_task_id
        ]
        replacement = materialize_plan_item(task, slot_id, level)
        replacement.update({
            "catalog_replacement_stage": "reused_task_after_catalog_exhaustion",
            "reused_due_to_catalog_exhaustion": True,
            "task_reused": True,
            "reuse_reason": "no_unused_compatible_task",
            "reused_task_id": reused_task_id,
            "required_instance_distinctions": [
                "different_target_or_unknown",
                "different_required_method",
                "different_representation",
                "different_constraint_structure",
                "different_direction_of_reasoning",
                "different_application_requiring_new_reasoning",
            ],
            "reuse_rule": (
                "Use at least one meaningful distinction supported by this task. Different wording, "
                "names, scenarios, or numbers alone do not qualify."
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
        (task, level) for task, level in compatible
        if str(task.get("task_id") or "") == current_task_id
        and current_task_id not in excluded
    ), None)
    if original is None:
        return None
    task, level = original
    replacement = materialize_plan_item(task, slot_id, level)
    replacement.update({
        "catalog_replacement_stage": "regenerated_original_task_after_duplicate",
        "reused_due_to_catalog_exhaustion": False,
        "task_reused": True,
        "reuse_reason": "regenerate_original_task_after_duplicate",
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
