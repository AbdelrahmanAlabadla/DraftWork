from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Optional

from pydantic import Field

from app.config import PLANNER_SNIPPET_TOKENS
from app.llm.client import request_structured
from app.llm.factory import create_llm_client
from app.llm.deepseek import DeepSeekIncompleteError, DeepSeekValidationError
from app.llm.schemas import (
    NonEmptyString,
    StrictOutputModel,
    plan_repair_response_model,
    planner_response_model,
    strict_json_schema,
)
from app.logging_conf import get_logger
from app.online.assessment_catalog import (
    CANONICAL_TASK_ID_FIELDS,
    apply_canonical_task_ids,
    is_authorized_catalog_exhaustion_reuse,
    is_authorized_fitb_entry_reuse,
)
from app.online.graph import ExamState
from app.online.models import normalize_text
from app.online.semantic_keys import formula_semantic_key
from app.online.prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_prompt,
)

logger = get_logger("PLANNER")

MAX_PLAN_ATTEMPTS = 3


class SemanticPlanRepair(StrictOutputModel):
    slot_to_replace: NonEmptyString
    conflicts_with: list[NonEmptyString] = Field(min_length=1)
    conflict_reason: NonEmptyString
    what_must_be_different: list[NonEmptyString] = Field(min_length=1)
    replacement_task_id: NonEmptyString
    why_this_replacement_resolves_the_conflict: NonEmptyString


class SemanticPlanRepairResponse(StrictOutputModel):
    repair: SemanticPlanRepair

VALID_TYPES = {
    "mcq", "true_false", "fill_in_the_blank", "definition", "why",
    "equation", "word_problem", "essay",
}


def plan_structure_errors(
    plans: list[dict[str, Any]], tasks: list[tuple[str, int]], num_models: int,
) -> list[str]:
    """Strict requested-layout gate independent of catalog shortage allowances."""
    errors: list[str] = []
    by_model = {int(plan.get("model_number") or 0): plan for plan in plans}
    for model_number in range(1, num_models + 1):
        plan = by_model.get(model_number)
        if plan is None:
            errors.append(f"Missing requested exam model {model_number}.")
            continue
        for qtype, count in tasks:
            actual = [
                str(item.get("slot_id") or "")
                for item in (plan.get("items") or {}).get(qtype) or []
            ]
            expected = [slot_id(model_number, qtype, index) for index in range(1, count + 1)]
            if actual != expected:
                errors.append(
                    f"Requested slots for model {model_number} {qtype} are {expected}; received {actual}."
                )
    return errors


RELATIONSHIP_REQUIRED_MODES = {
    "calculation",
    "symbolic_manipulation",
    "law_or_theorem_application",
    "derivation_relationship",
}

_FITB_FORMULA_TEXT_RE = re.compile(
    r"(?:[+*/^]|\b(?:equation|formula|calculate|calculation|computed?|solve|derive)\b|"
    r"(?:معادل(?:ة|ات)|احسب|حساب|حل\s+المعادلة|اشتق))",
    re.IGNORECASE,
)


def _fitb_plan_requires_non_term_answer(entry: dict[str, Any]) -> bool:
    """Return True when a FITB plan asks for a formula or calculated result."""
    answer_shape = " ".join(
        str(entry.get(field) or "")
        for field in (
            "target", "canonical_skill", "idea_to_test",
        )
    )
    if _FITB_FORMULA_TEXT_RE.search(answer_shape):
        return True
    target_shape = " ".join(
        str(entry.get(field) or "")
        for field in ("target", "canonical_skill", "idea_to_test")
    )
    if re.search(
        r"\b(?:numeric(?:al)?\s+value|constant\s+value|calculated\s+result)\b|"
        r"(?:قيمة\s+(?:ثابت|عدد|ناتج)|ناتج\s+الحساب|العلاقة\s+الرياضية)",
        target_shape,
        re.IGNORECASE,
    ):
        return True
    relationship = str(entry.get("relationship") or "")
    quantitative_target = re.search(
        r"\b(?:numeric|number|value|result|level|endpoint|frequency|wavelength|energy|speed|mass|time|distance)\b",
        target_shape,
        re.IGNORECASE,
    )
    return bool(quantitative_target and re.search(r"(?:=|[+*/^])", relationship))
_DERIVATION_CUE_RE = re.compile(
    r"\b(?:derive|derivation|prove|show\s+that|establish(?:\s+a)?\s+relationship)\b|"
    r"(?:اشتق|يشتق|اشتقاق|أثبت|اثبت|برهن|برهنة|استنتج\s+العلاقة|إثبات)",
    flags=re.IGNORECASE,
)
_ARABIC_DERIVATION_CUE_RE = re.compile(
    r"(?:اشتق|يشتق|اشتقاق|أثبت|اثبت|برهن|برهنة|استنتج|استنتاج)",
    flags=re.IGNORECASE,
)
_HARD_DEPTH_CUE_RE = re.compile(
    r"\b(?:combine|multiple\s+(?:laws|relationships|ideas)|several\s+(?:meaningful\s+)?steps|"
    r"intermediate\s+result|then\s+(?:interpret|compare|conclude|apply)|compare\s+alternatives|"
    r"error\s+analysis|derive|derivation|prove|show\s+that|infer)\b|"
    r"(?:ادمج|دمج|علاقتين|عدة\s+خطوات|نتيجة\s+وسيطة|ثم\s+(?:فسر|قارن|استنتج|طبق)|"
    r"قارن\s+البدائل|تحليل\s+خطأ|اشتق|يشتق|اشتقاق|أثبت|اثبت|برهن|استنتج|استنتاج|"
    r"ثم\s+(?:فسر|قارن|استنتج|طبق|وظف|استخدم))",
    flags=re.IGNORECASE,
)
_ARABIC_HARD_DEPTH_CUE_RE = re.compile(
    r"(?:ادمج|دمج|عدة\s+خطوات|نتيجة\s+وسيطة|قارن|تحليل\s+خطأ|"
    r"اشتق|يشتق|اشتقاق|أثبت|اثبت|برهن|استنتج|استنتاج)",
    flags=re.IGNORECASE,
)
_VISUAL_DEPENDENCY_RE = re.compile(
    r"\b(?:image|graph|table|diagram|figure|chart|visual)\b|"
    r"(?:صورة|رسم|جدول|مخطط|بياني|تسميات)|"
    r"(?:(?:في|من|حسب)\s+الشكل|الشكل\s+(?:التالي|الآتي|المرفق|الموضح|أعلاه|أدناه)|"
    r"شكل\s+(?:بياني|توضيحي))",
    flags=re.IGNORECASE,
)

def slot_id(model_number: int, qtype: str, index: int) -> str:
    """Return the stable identity of one requested plan/generation slot."""
    return f"m{model_number}_{qtype}_{index}"


def build_plan_schema(
    tasks: list[tuple[str, int]], num_models: int, chunk_ids: list[str]
) -> dict[str, Any]:
    """Build the planner schema from its Pydantic response model."""
    return strict_json_schema(planner_response_model(tasks, num_models, chunk_ids))


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_plans(raw: object, num_models: int) -> list[dict[str, Any]]:
    """Normalize one planner response into one plan per exam model.

    Returns ``[{"model_number": N, "items": {qtype: [{topic, concept_to_test}]}}]``
    ordered by model_number. Raw ``questions`` may be missing / malformed; the
    caller must validate counts afterwards.
    """
    raw_exams = raw.get("exams") if isinstance(raw, dict) else None
    if not isinstance(raw_exams, list):
        return []

    by_number: dict[int, dict[str, Any]] = {}
    for entry in raw_exams:
        if not isinstance(entry, dict):
            continue
        try:
            number = int(entry.get("model_number"))
        except (TypeError, ValueError):
            continue
        items: dict[str, list[dict[str, Any]]] = {t: [] for t in VALID_TYPES}
        grouped = entry.get("items")
        if isinstance(grouped, dict):
            source_entries = [
                (qtype, q)
                for qtype, values in grouped.items()
                if qtype in VALID_TYPES and isinstance(values, list)
                for q in values
            ]
        else:
            source_entries = [
                (str(q.get("question_type") or "").strip(), q)
                for q in (entry.get("questions") or [])
                if isinstance(q, dict)
            ]
        for qtype, q in source_entries:
            if qtype not in VALID_TYPES or not isinstance(q, dict):
                continue
            topic = str(q.get("topic") or "").strip()
            concept = str(q.get("concept") or topic).strip()
            idea = str(q.get("idea_to_test") or q.get("concept_to_test") or "").strip()
            relationship_raw = q.get("relationship")
            relationship = (
                str(relationship_raw).strip() if relationship_raw is not None else None
            )
            items[qtype].append(
                {
                    "slot_id": str(q.get("slot_id") or "").strip(),
                    "source_chunk_id": str(q.get("source_chunk_id") or "").strip(),
                    "support_chunk_ids": list(dict.fromkeys(
                        str(chunk_id).strip()
                        for chunk_id in (q.get("support_chunk_ids") or [])
                        if str(chunk_id).strip()
                    )),
                    "topic": topic,
                    "concept": concept,
                    "canonical_skill": str(q.get("canonical_skill") or idea or concept).strip(),
                    "knowledge_focus": str(q.get("knowledge_focus") or "").strip(),
                    "student_work_product": str(q.get("student_work_product") or "").strip(),
                    "relationship": relationship or None,
                    "target": str(q.get("target") or "").strip(),
                    "task_operation": str(
                        q.get("task_operation") or q.get("reasoning_mode") or "conceptual_recall"
                    ).strip(),
                    "given_conditions": [
                        str(value).strip() for value in (q.get("given_conditions") or [])
                        if str(value).strip()
                    ],
                    "scenario_family": (
                        str(q.get("scenario_family")).strip()
                        if q.get("scenario_family") is not None else None
                    ),
                    "required_givens": [
                        str(value).strip() for value in (q.get("required_givens") or [])
                        if str(value).strip()
                    ],
                    "text_answerable": q.get("text_answerable", True),
                    "reasoning_mode": str(q.get("reasoning_mode") or "").strip(),
                    "difficulty": str(q.get("difficulty") or "").strip().lower(),
                    "idea_to_test": idea,
                    # Compatibility with existing generator prompt builders.
                    "concept_to_test": idea,
                }
            )
        by_number[number] = {
            "model_number": number,
            "items": items,
        }

    plans = [by_number.get(n) for n in range(1, num_models + 1)]
    return [p for p in plans if p is not None]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _concept_tokens(item: dict[str, Any]) -> set[str]:
    # Broad concepts may intentionally repeat across exam versions. Compare the
    # specific grounded idea only, so equivalent coverage remains possible.
    terms = set(
        normalize_text(item.get("idea_to_test") or item.get("concept_to_test", ""))
    )
    return {t for t in terms if t not in _LOW_SIGNAL_TOKENS}


_LOW_SIGNAL_TOKENS = frozenset(
    (
        "a an and are as at be but by for from has have how i in is it its of on "
        "or that the this to was we what when which who why with what does it's "
        "concept topic distinct about describe explain between using used and of "
        "how one value understand main key basic".split()
    )
)


def _concept_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _task_part(value: object) -> str:
    """Conservative comparison key for planner-owned canonical labels."""
    if value is None:
        return ""
    text = re.sub(r"(?i)(\w)[’']s\b", r"\1", str(value or "").strip())
    tokens = normalize_text(text)
    # Arabic planners inconsistently include the definite article in otherwise
    # identical canonical labels. Removing it from comparison tokens keeps the
    # duplicate key language-aware without attempting symbolic equivalence.
    tokens = [token[2:] if token.startswith("ال") and len(token) > 4 else token for token in tokens]
    return " ".join(tokens)


def _condition_key(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(sorted(filter(None, (_task_part(value) for value in values))))


def _specific_conditions(values: tuple[str, ...]) -> bool:
    """Return true when conditions identify a concrete numerical/symbolic setup."""
    return bool(values) and any(re.search(r"\d|[=<>≤≥]", value) for value in values)


def _condition_quantity_key(values: tuple[str, ...]) -> tuple[float, ...]:
    """Extract a conservative fingerprint of important numeric givens."""
    numbers: list[float] = []
    for value in values:
        for match in re.findall(r"(?<![\w.])[+\-]?\d+(?:\.\d+)?", value):
            try:
                numbers.append(float(match))
            except ValueError:
                continue
    return tuple(sorted(numbers))


def duplicate_task_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Structured task identity, deliberately independent of question type."""
    relationship = item.get("relationship")
    basis = formula_semantic_key(relationship) or relationship or item.get("concept")
    return (
        _task_part(item.get("knowledge_focus")),
        _task_part(item.get("student_work_product")),
        _task_part(item.get("canonical_skill")),
        _task_part(item.get("task_operation")),
        _task_part(basis),
        _task_part(item.get("target")),
        _condition_key(item.get("given_conditions")),
    )


def scenario_repetition_key(item: dict[str, Any]) -> tuple[Any, ...] | None:
    family = _task_part(item.get("scenario_family"))
    if not family:
        return None
    # Different names or values do not make reuse of the same setup a new
    # scenario. The detailed conditions remain in duplicate_task_key so the
    # repairer still receives them as context.
    return (family,)


_RECALL_MODES = frozenset({_task_part("conceptual_recall")})


def semantic_task_identity(item: dict[str, Any]) -> tuple[str, str]:
    """Primary, question-type-independent identity of assessed learning."""
    return (
        _task_part(item.get("knowledge_focus")),
        _task_part(item.get("student_work_product")),
    )


def _recall_level(item: dict[str, Any]) -> bool:
    return _task_part(item.get("reasoning_mode")) in _RECALL_MODES


def _semantic_plan_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Apply semantic identity before supporting task/scenario identity."""
    left_focus, left_product = semantic_task_identity(left)
    right_focus, right_product = semantic_task_identity(right)
    if not left_focus or not right_focus:
        return False
    if left_focus == right_focus:
        if left_product == right_product:
            return True
        left_target = _task_part(left.get("target"))
        right_target = _task_part(right.get("target"))
        if (
            _recall_level(left) and _recall_level(right)
            and _targets_equivalent_for_duplicate(left_target, right_target)
        ):
            return True
        left_secondary = (
            _task_part(left.get("task_operation")),
            _task_part(left.get("target")),
            _task_part(left.get("relationship") or left.get("concept")),
            _task_part(left.get("reasoning_mode")),
        )
        right_secondary = (
            _task_part(right.get("task_operation")),
            _task_part(right.get("target")),
            _task_part(right.get("relationship") or right.get("concept")),
            _task_part(right.get("reasoning_mode")),
        )
        return left_secondary == right_secondary
    return False


def _same_assessment_work(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Calibrate supporting overlap against the work asked of the student."""
    left_operation = _task_part(left.get("task_operation"))
    right_operation = _task_part(right.get("task_operation"))
    left_target = _task_part(left.get("target"))
    right_target = _task_part(right.get("target"))
    same_operation_target = bool(
        left_operation and left_operation == right_operation
        and left_target and right_target
        and _targets_equivalent_for_duplicate(left_target, right_target)
    )
    if same_operation_target:
        return True
    left_mode = _task_part(left.get("reasoning_mode"))
    right_mode = _task_part(right.get("reasoning_mode"))
    if not left_mode or left_mode != right_mode:
        return False
    if _targets_equivalent_for_duplicate(left_target, right_target):
        return True
    # Identical concrete setups can encode the same result under two labels
    # (for example, classify an inconsistent system vs determine its solution
    # count). This remains duplicate evidence only within the same work mode.
    left_conditions = _condition_key(left.get("given_conditions"))
    right_conditions = _condition_key(right.get("given_conditions"))
    return bool(
        left_conditions == right_conditions and _specific_conditions(left_conditions)
    )


def _authorized_reuse_pair(
    left: dict[str, Any], left_sid: str, right: dict[str, Any], right_sid: str,
) -> bool:
    """Authorize metadata-bearing reuse regardless of validation iteration order."""
    for reused, reused_sid, other_sid in (
        (left, left_sid, right_sid), (right, right_sid, left_sid),
    ):
        if (
            is_authorized_catalog_exhaustion_reuse(reused)
            or is_authorized_fitb_entry_reuse(reused)
        ) and other_sid in (reused.get("forced_duplicate_slot_ids") or []):
            return True
    return False


def _targets_equivalent_for_duplicate(left: str, right: str) -> bool:
    """Compare planner labels conservatively, allowing scenario suffixes."""
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return True
    # Ignore a material/entity suffix when the requested output is otherwise
    # the same (for example, an angle in glass versus in polyethylene).
    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return overlap >= 0.60


def validate_assessment_profile(
    profile: dict[str, Any],
    tasks: list[tuple[str, int]],
    valid_chunk_ids: set[str],
    planning_issues: list[dict[str, Any]],
) -> list[str]:
    """Validate the compact analysis contract without another LLM call."""
    errors: list[str] = []
    question_types = profile.get("question_types") if isinstance(profile, dict) else None
    if not isinstance(question_types, dict):
        return ["Assessment profile is missing question_types."]
    issue_by_type = {
        str(issue.get("question_type")): issue
        for issue in planning_issues
        if isinstance(issue, dict)
    }
    for qtype in VALID_TYPES:
        entry = question_types.get(qtype)
        if not isinstance(entry, dict):
            errors.append(f"Assessment profile is missing {qtype} guidance.")
            continue
        support = entry.get("support")
        operations = entry.get("task_operations")
        evidence_ids = entry.get("evidence_chunk_ids")
        if support not in {"supported", "limited", "unsupported"}:
            errors.append(f"Assessment profile has invalid support for {qtype}.")
        if not isinstance(operations, list) or any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", str(value or ""))
            for value in operations
        ):
            errors.append(f"Assessment profile has invalid task_operations for {qtype}.")
        if support != "unsupported" and not operations:
            errors.append(f"Assessment profile lists no task_operations for supported {qtype}.")
        if support == "unsupported" and operations:
            errors.append(f"Unsupported {qtype} must not list task_operations.")
        if not isinstance(evidence_ids, list) or any(
            str(chunk_id) not in valid_chunk_ids for chunk_id in evidence_ids
        ):
            errors.append(f"Assessment profile references invalid evidence for {qtype}.")
        if support != "unsupported" and not evidence_ids:
            errors.append(f"Assessment profile lists no evidence chunks for supported {qtype}.")

    task_map = dict(tasks)
    for qtype, requested_count in task_map.items():
        entry = question_types.get(qtype) or {}
        support = entry.get("support")
        issue = issue_by_type.get(qtype)
        if support == "unsupported" and issue is None:
            errors.append(f"Requested unsupported type {qtype} has no planning issue.")
        if issue is not None:
            if int(issue.get("requested_count") or 0) != requested_count:
                errors.append(f"Planning issue for {qtype} has the wrong requested_count.")
            supported_count = int(issue.get("supported_count") or 0)
            if supported_count < 0 or supported_count >= requested_count:
                errors.append(f"Planning issue for {qtype} has an invalid supported_count.")
    for qtype in issue_by_type:
        if qtype not in task_map:
            errors.append(f"Planning issue was returned for unrequested type {qtype}.")
    return errors


def validate_plans(
    plans: list[dict[str, Any]],
    tasks: list[tuple[str, int]],
    num_models: int,
    valid_chunk_ids: set[str] | None = None,
    requested_difficulty: str | None = None,
    assessment_profile: dict[str, Any] | None = None,
    planning_issues: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return human-readable validation errors for a parsed plan set."""
    errors: list[str] = []

    task_map = {t: c for t, c in tasks}
    issue_by_type = {
        str(issue.get("question_type")): issue
        for issue in (planning_issues or [])
        if isinstance(issue, dict)
    }
    type_profiles = (assessment_profile or {}).get("question_types") or {}
    if len(plans) != num_models:
        errors.append(f"Expected {num_models} exam models, got {len(plans)}.")
    present_models = {int(model["model_number"]) for model in plans}
    for number in range(1, num_models + 1):
        if number in present_models:
            continue
        for qtype, count in task_map.items():
            issue = issue_by_type.get(qtype)
            expected_count = int(issue.get("supported_count") or 0) if issue else count
            for index in range(1, expected_count + 1):
                errors.append(
                    f"Slot {slot_id(number, qtype, index)} is missing from the plan."
                )

    for model in plans:
        number = model["model_number"]
        items = model["items"]
        for qtype, count in task_map.items():
            issue = issue_by_type.get(qtype)
            expected_count = int(issue.get("supported_count") or 0) if issue else count
            values = items.get(qtype) or []
            expected_ids = {
                slot_id(number, qtype, index)
                for index in range(1, expected_count + 1)
            }
            present_ids = {
                str(entry.get("slot_id") or slot_id(number, qtype, index))
                for index, entry in enumerate(values, start=1)
                if isinstance(entry, dict)
            }
            for missing_sid in sorted(expected_ids - present_ids):
                errors.append(f"Slot {missing_sid} is missing from the plan.")
            unexpected_ids = sorted(present_ids - expected_ids)
            if len(values) > expected_count or unexpected_ids:
                errors.append(
                    f"Model {number} has invalid {qtype} slot IDs {unexpected_ids}; "
                    f"required {sorted(expected_ids)}."
                )
            for index, entry in enumerate(values, start=1):
                sid = str(entry.get("slot_id") or slot_id(number, qtype, index))
                is_fitb_entry = qtype == "fill_in_the_blank"
                if is_fitb_entry:
                    # FITB bank entries deliberately have their own identity and
                    # must never acquire a synthetic assessment-task identity.
                    for field in CANONICAL_TASK_ID_FIELDS:
                        entry[field] = None
                    identity_complete = bool(entry.get("fitb_entry_id"))
                else:
                    identity_complete = apply_canonical_task_ids(entry)
                metadata_complete = all(
                    entry.get(field)
                    for field in (
                        "source_chunk_id", "topic", "concept", "canonical_skill",
                        "knowledge_focus", "student_work_product",
                        "target", "task_operation",
                        "reasoning_mode", "difficulty", "idea_to_test",
                    )
                )
                if not metadata_complete:
                    errors.append(
                        f"Slot {sid} is missing required planning metadata."
                    )
                elif valid_chunk_ids is not None and entry["source_chunk_id"] not in valid_chunk_ids:
                    errors.append(f"Slot {sid} references an unselected source chunk.")
                if not identity_complete or (not is_fitb_entry and (
                    any(
                        not entry.get(field)
                        for field in CANONICAL_TASK_ID_FIELDS
                        if field != "relationship_id"
                    ) or (
                        entry.get("relationship") and not entry.get("relationship_id")
                    )
                )):
                    errors.append(
                        f"Slot {sid} is missing canonical "
                        f"{'FITB entry' if is_fitb_entry else 'task'} identity metadata."
                    )
                support_ids = entry.get("support_chunk_ids") or []
                if not isinstance(support_ids, list):
                    errors.append(f"Slot {sid} has invalid support_chunk_ids metadata.")
                else:
                    if entry.get("source_chunk_id") in support_ids:
                        errors.append(f"Slot {sid} repeats its main chunk as a support chunk.")
                    if len(support_ids) != len(set(support_ids)):
                        errors.append(f"Slot {sid} repeats a support chunk ID.")
                    if valid_chunk_ids is not None and any(
                        chunk_id not in valid_chunk_ids for chunk_id in support_ids
                    ):
                        errors.append(f"Slot {sid} references an unselected support chunk.")
                mode = str(entry.get("reasoning_mode") or "")
                operation = str(entry.get("task_operation") or "")
                allowed_operations = (
                    (type_profiles.get(qtype) or {}).get("task_operations") or []
                )
                if allowed_operations and operation not in allowed_operations:
                    errors.append(
                        f"Slot {sid} uses task_operation {operation!r}, which is not listed in "
                        f"the assessment profile for {qtype}."
                    )
                if qtype == "why" and mode in {
                    "calculation", "symbolic_manipulation"
                }:
                    errors.append(
                        f"Slot {sid} assigns {mode} to {qtype}; use a conceptual "
                        "explanation task or move the quantitative/symbolic task to equation."
                    )
                if qtype == "fill_in_the_blank" and _fitb_plan_requires_non_term_answer(entry):
                    errors.append(
                        f"Slot {sid} makes Fill-in-the-Blank require an equation, formula, "
                        "calculation, or multi-step result. Replace it with a grounded word, "
                        "term, short phrase, concept name, unit name, or symbol name."
                    )
                if (
                    mode in RELATIONSHIP_REQUIRED_MODES
                    and not entry.get("relationship")
                    and not (entry.get("required_givens") or entry.get("given_conditions"))
                ):
                    errors.append(
                        f"Slot {sid} requires a source-grounded relationship or explicit "
                        f"givens for {mode}."
                    )
                slot_difficulty = str(entry.get("difficulty") or "")
                if slot_difficulty not in {"easy", "medium", "hard"}:
                    errors.append(f"Slot {sid} has an invalid difficulty.")
                elif requested_difficulty in {"easy", "medium", "hard"} and (
                    slot_difficulty != requested_difficulty
                ):
                    errors.append(
                        f"Slot {sid} difficulty must be {requested_difficulty}."
                    )
                idea = str(entry.get("idea_to_test") or "")
                has_derivation_cue = bool(
                    _DERIVATION_CUE_RE.search(idea)
                    or _ARABIC_DERIVATION_CUE_RE.search(idea)
                )
                if mode == "symbolic_manipulation" and has_derivation_cue:
                    errors.append(
                        f"Slot {sid} asks to derive/prove a relationship, so reasoning_mode "
                        "must be derivation_relationship rather than symbolic_manipulation."
                    )
                elif mode == "derivation_relationship" and not has_derivation_cue:
                    errors.append(
                        f"Slot {sid} uses derivation_relationship but idea_to_test does not "
                        "ask the student to derive, prove, show, or establish a relationship."
                    )
                if slot_difficulty == "hard" and mode in {
                    "calculation", "law_or_theorem_application", "conceptual_recall",
                    "conceptual_reasoning", "comparison_prediction",
                } and not (
                    _HARD_DEPTH_CUE_RE.search(idea)
                    or _ARABIC_HARD_DEPTH_CUE_RE.search(idea)
                ):
                    errors.append(
                        f"Slot {sid} is labeled hard without a supported deeper-reasoning "
                        "operation such as combining ideas, meaningful multi-step work, inference, "
                        "comparison, derivation, proof, error analysis, or an intermediate conclusion."
                    )

        # Reject the same relationship/target/operation across question types.
        # A null relationship deliberately falls back to concept.
        occupied: dict[tuple[Any, ...], str] = {}
        occupied_parts: list[tuple[tuple[Any, ...], str]] = []
        occupied_items: list[tuple[dict[str, Any], str]] = []
        occupied_scenarios: dict[tuple[Any, ...], tuple[str, dict[str, Any]]] = {}
        occupied_catalog_tasks: dict[str, tuple[str, dict[str, Any]]] = {}
        for qtype in VALID_TYPES:
            for index, entry in enumerate(items.get(qtype) or [], start=1):
                sid = str(entry.get("slot_id") or slot_id(number, qtype, index))
                task_id = str(entry.get("task_id") or "")
                coverage_id = str(entry.get("coverage_id") or "")
                replacement_stage = str(entry.get("catalog_replacement_stage") or "")
                authorized_exact_reuse = is_authorized_catalog_exhaustion_reuse(entry)
                authorized_family_reuse = bool(
                    replacement_stage in {
                        "unused_task_used_concept",
                        "regenerated_original_task_after_duplicate",
                        "plan_review_unused_task",
                        "plan_review_intentional_reuse",
                    }
                    and entry.get("forced_duplicate_slot_ids")
                )
                authorized_catalog_fallback = (
                    authorized_exact_reuse
                    or is_authorized_fitb_entry_reuse(entry)
                    or authorized_family_reuse
                )
                if task_id:
                    prior_catalog_task = occupied_catalog_tasks.get(task_id)
                    prior_sid, prior_entry = prior_catalog_task or ("", {})
                    pair_authorized = bool(
                        prior_catalog_task
                        and _authorized_reuse_pair(prior_entry, prior_sid, entry, sid)
                    )
                    if prior_catalog_task and not pair_authorized:
                        errors.append(
                            f"Slot {sid} duplicates catalog task {task_id} already used by "
                            f"{prior_sid}."
                        )
                    else:
                        occupied_catalog_tasks[task_id] = (sid, entry)
                if entry.get("text_answerable") is not True:
                    errors.append(
                        f"Slot {sid} is not fully answerable from text; choose a task whose "
                        "required data can be included without an unavailable visual."
                    )
                required_givens = entry.get("required_givens")
                given_conditions = entry.get("given_conditions")
                if not isinstance(required_givens, list) or not isinstance(given_conditions, list):
                    errors.append(f"Slot {sid} has invalid givens metadata.")
                elif required_givens and not given_conditions:
                    errors.append(
                        f"Slot {sid} requires givens but does not provide grounded given_conditions."
                    )
                if _VISUAL_DEPENDENCY_RE.search(str(entry.get("idea_to_test") or "")) and not required_givens:
                    errors.append(
                        f"Slot {sid} appears to depend on a visual representation but does not "
                        "list the text givens needed to make it self-contained."
                    )
                key = duplicate_task_key(entry)
                if not all(key[:2]):
                    continue
                prior = next((
                    prior_sid for prior_item, prior_sid in occupied_items
                    if _semantic_plan_conflict(prior_item, entry)
                ), None)
                if prior is None:
                    prior = occupied.get(key)
                if prior is None:
                    prior = next((
                        prior_sid for (prior_key, prior_sid), (prior_item, _prior_item_sid)
                        in zip(occupied_parts, occupied_items)
                        if (
                            (
                                prior_key[3] == key[3]
                                and prior_key[4] == key[4]
                                and _targets_equivalent_for_duplicate(prior_key[5], key[5])
                            )
                            or (
                                prior_key[6] == key[6]
                                and _specific_conditions(key[6])
                                and _same_assessment_work(prior_item, entry)
                            )
                            or (
                                len(_condition_quantity_key(key[6])) >= 3
                                and _condition_quantity_key(prior_key[6])
                                == _condition_quantity_key(key[6])
                                and _same_assessment_work(prior_item, entry)
                            )
                        )
                    ), None)
                prior_entry = next((
                    item for item, item_sid in occupied_items if item_sid == prior
                ), {})
                pair_authorized = bool(
                    prior and _authorized_reuse_pair(prior_entry, prior, entry, sid)
                )
                if prior and not authorized_catalog_fallback and not pair_authorized:
                    errors.append(
                        f"Slot {sid} duplicates the planned reasoning task in {prior}; change the "
                        "canonical skill, operation, grounded relationship, target, or givens."
                    )
                else:
                    occupied[key] = sid
                    occupied_parts.append((key, sid))
                    occupied_items.append((entry, sid))
                scenario_key = scenario_repetition_key(entry)
                if scenario_key is not None:
                    scenario_prior = occupied_scenarios.get(scenario_key)
                    scenario_sid, scenario_entry = scenario_prior or ("", {})
                    scenario_pair_authorized = bool(
                        scenario_prior
                        and _authorized_reuse_pair(
                            scenario_entry, scenario_sid, entry, sid
                        )
                    )
                    if scenario_prior and not authorized_catalog_fallback and not scenario_pair_authorized:
                        errors.append(
                            f"Slot {sid} repeats the same scenario/setup and conditions as "
                            f"{scenario_sid}; use a meaningfully different scenario or task."
                        )
                    else:
                        occupied_scenarios[scenario_key] = (sid, entry)

        # A mixed request aims for balance only where the source supports real
        # differences in reasoning demand. Enforcing a quota here caused simple
        # tasks to be mislabeled as hard.

    # Models may share a concept for equivalent coverage. Only an effectively
    # identical idea assignment is rejected across models.
    concepts_by_type: dict[str, list[tuple[int, str, set[str]]]] = {}
    for model in plans:
        number = model["model_number"]
        for qtype in VALID_TYPES:
            if qtype == "fill_in_the_blank":
                # FITB answer reuse is tracked by normalized bank identity and
                # independently reviewed by the Plan Review and Duplicate Judge.
                continue
            concepts_by_type.setdefault(qtype, [])
            for index, entry in enumerate(model["items"].get(qtype) or [], start=1):
                concepts_by_type[qtype].append(
                    (number, slot_id(number, qtype, index), _concept_tokens(entry))
                )

    for qtype, entries in concepts_by_type.items():
        for i in range(len(entries)):
            num_i, sid_i, toks_i = entries[i]
            for j in range(i + 1, len(entries)):
                num_j, sid_j, toks_j = entries[j]
                if num_i == num_j:
                    continue
                if _concept_overlap(toks_i, toks_j) >= 0.85:
                    errors.append(
                        f"Slot {sid_j} repeats the same {qtype} idea as {sid_i}. "
                        "Assign a different grounded idea."
                    )

    return errors


# --------------------------------------------------------------------------
# Deterministic finalize (trim extras from the end)
# --------------------------------------------------------------------------
def finalize_plans(
    plans: list[dict[str, Any]], tasks: list[tuple[str, int]]
) -> list[dict[str, Any]]:
    """Remove extra per-type plan items from the END of each type's list."""
    task_map = {t: c for t, c in tasks}
    finalized: list[dict[str, Any]] = []
    for model in plans:
        items: dict[str, list[dict[str, Any]]] = {}
        for qtype in VALID_TYPES:
            values = (model["items"].get(qtype) or [])[: task_map.get(qtype, 0)]
            for index, item in enumerate(values, start=1):
                item.setdefault("slot_id", slot_id(model["model_number"], qtype, index))
                if not item.get("slot_id"):
                    item["slot_id"] = slot_id(model["model_number"], qtype, index)
                # IDs are derived state. Recompute them after every initial or
                # targeted planner result so changed semantics cannot retain a
                # stale catalog identity. FITB has a separate compiler-owned ID.
                if qtype == "fill_in_the_blank" and item.get("fitb_entry_id"):
                    for field in CANONICAL_TASK_ID_FIELDS:
                        item[field] = None
                else:
                    apply_canonical_task_ids(item)
            items[qtype] = values
        finalized.append({"model_number": model["model_number"], "items": items})
    return finalized


# --------------------------------------------------------------------------
# Planner nodes (LangGraph)
# --------------------------------------------------------------------------
def _call_planner(
    user_prompt: str,
    *,
    response_model: type,
    item_count: int,
    max_tokens: int,
    schema_name: str = "grounded_exam_plan",
    expected_ids: list[str] | None = None,
    operation: str = "initial_planner",
    agent: str = "planner",
) -> object:
    client = create_llm_client()
    return request_structured(
        client,
        user_prompt,
        response_model=response_model,
        schema_name=schema_name,
        agent=agent,
        item_count=item_count,
        operation=operation,
        expected_ids=expected_ids,
        expected_id_field="slot_id" if expected_ids else None,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=max_tokens,
    )


def _store_plan_result(state: dict[str, Any], raw: object, tasks, num_models) -> dict:
    plans = parse_plans(raw, num_models)
    plans = finalize_plans(plans, tasks)
    chunk_ids = {
        str(chunk.get("child_id"))
        for chunk in state.get("retrieved_chunks") or []
        if chunk.get("child_id")
    }
    raw_dict = raw if isinstance(raw, dict) else {}
    assessment_profile = raw_dict.get("assessment_profile")
    planning_issues = raw_dict.get("planning_issues")
    if not isinstance(planning_issues, list):
        planning_issues = []
    if not isinstance(assessment_profile, dict):
        # Compatibility for direct/legacy test doubles. Production structured
        # output requires the real profile and cannot take this path.
        operations_by_type = {
            qtype: sorted({
                str(item.get("task_operation") or "conceptual_recall")
                for plan in plans
                for item in (plan.get("items", {}).get(qtype) or [])
            })
            for qtype in VALID_TYPES
        }
        assessment_profile = {
            "domains": ["unspecified"],
            "educational_level": "unspecified",
            "assessment_style": "Source-grounded assessment",
            "question_types": {
                qtype: {
                    "support": "supported",
                    "task_operations": operations_by_type[qtype] or ["conceptual_recall"],
                    "form_guidance": "Follow the source-supported planned form.",
                    "evidence_chunk_ids": sorted(chunk_ids),
                }
                for qtype in VALID_TYPES
            },
            "unsupported_visual_tasks": [],
        }
    errors = validate_assessment_profile(
        assessment_profile, tasks, chunk_ids, planning_issues
    )
    errors.extend(validate_plans(
        plans, tasks, num_models, chunk_ids or None, state.get("difficulty") or "mix",
        assessment_profile, planning_issues,
    ))
    return {
        "plans": plans,
        "assessment_profile": assessment_profile,
        "planning_issues": planning_issues,
        "plan_errors": errors,
        "last_plan_raw": _safe_dumps(raw),
    }


def _completed_json_objects(text: str) -> list[dict[str, Any]]:
    """Recover only fully closed JSON objects from a truncated response."""
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


def recover_planner_checkpoint(
    partial_output: str,
    tasks: list[tuple[str, int]],
    num_models: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover a complete profile and complete slot objects; ignore trailing fragments."""
    objects = _completed_json_objects(partial_output)
    profile = next((
        obj for obj in objects
        if isinstance(obj.get("question_types"), dict)
        and isinstance(obj.get("domains"), list)
    ), None)
    issues = next((
        obj.get("planning_issues") for obj in reversed(objects)
        if isinstance(obj.get("planning_issues"), list)
    ), None)
    if issues is None:
        issues = [
            obj for obj in objects
            if obj.get("question_type") in VALID_TYPES
            and "requested_count" in obj and "supported_count" in obj
        ]
    by_slot: dict[str, dict[str, Any]] = {}
    for obj in objects:
        sid = str(obj.get("slot_id") or "")
        if _slot_location(sid) and obj.get("source_chunk_id"):
            by_slot[sid] = obj
    plans = [
        {"model_number": number, "items": {qtype: [] for qtype in VALID_TYPES}}
        for number in range(1, num_models + 1)
    ]
    for sid, item in by_slot.items():
        location = _slot_location(sid)
        if location is None:
            continue
        model_number, qtype, _index = location
        plans[model_number - 1]["items"][qtype].append(item)
    for plan in plans:
        for qtype, values in plan["items"].items():
            values.sort(key=lambda item: (_slot_location(str(item.get("slot_id"))) or (0, qtype, 0))[2])
    return profile, plans, issues if isinstance(issues, list) else []


def _expected_supported_slots(
    tasks: list[tuple[str, int]], num_models: int, planning_issues: list[dict[str, Any]]
) -> list[str]:
    issue_by_type = {
        str(issue.get("question_type")): issue
        for issue in planning_issues if isinstance(issue, dict)
    }
    return [
        slot_id(model_number, qtype, index)
        for model_number in range(1, num_models + 1)
        for qtype, count in tasks
        for index in range(
            1,
            int((issue_by_type.get(qtype) or {}).get("supported_count", count)) + 1,
        )
    ]


def _plan_items_by_slot(plans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("slot_id")): item
        for plan in plans
        for values in (plan.get("items") or {}).values()
        for item in values
        if item.get("slot_id")
    }


def _safe_dumps(raw: object) -> str:
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)


def plan_exams(state: ExamState) -> dict[str, Any]:
    """First planning call (all models at once). Shared across the workflow."""
    from app.online.assessment_catalog import allocate_catalog_plan

    document_id = state.get("document_id")
    tasks = state.get("tasks") or []
    num_models = state.get("num_models") or 1
    difficulty = state.get("difficulty") or "mix"
    catalog = state.get("assessment_catalog") or {}

    if catalog:
        plans, planning_issues = allocate_catalog_plan(
            catalog, tasks, num_models, difficulty
        )
        profile = dict(catalog.get("assessment_profile") or {})
        chunk_ids = {
            str(chunk.get("child_id"))
            for chunk in state.get("retrieved_chunks") or [] if chunk.get("child_id")
        }
        errors = validate_assessment_profile(profile, tasks, chunk_ids, planning_issues)
        errors.extend(validate_plans(
            plans, tasks, num_models, chunk_ids, difficulty, profile, planning_issues
        ))
        logger.info(
            "Catalog plan allocated | document_id=%s | slots=%d | issues=%d | errors=%d",
            document_id,
            sum(len(values) for plan in plans for values in plan["items"].values()),
            len(planning_issues), len(errors),
        )
        return {
            "plans": plans,
            "assessment_profile": profile,
            "planning_issues": planning_issues,
            "plan_errors": errors,
            "plan_attempts": 1,
            "last_plan_raw": json.dumps(plans, ensure_ascii=False),
        }

    planner_context = state.get("planner_context") or ""
    language = state.get("document_language") or "en"

    logger.info(
        "Planning started | document_id=%s | models=%d | tasks=%s | attempt=1",
        document_id,
        num_models,
        tasks,
    )
    chunk_ids = [
        str(chunk.get("child_id"))
        for chunk in state.get("retrieved_chunks") or []
        if chunk.get("child_id")
    ]
    response_model = planner_response_model(tasks, num_models, chunk_ids)
    requested_slots = sum(count for _qtype, count in tasks) * num_models
    try:
        system_prompt, user_prompt = build_planner_prompt(
            num_models, tasks, planner_context, difficulty=difficulty, language=language
        )
        raw = _call_planner(
            user_prompt,
            response_model=response_model,
            item_count=requested_slots,
            max_tokens=25000,
            operation="initial_planner",
        )
    except (DeepSeekIncompleteError, DeepSeekValidationError) as exc:
        profile, checkpoint_plans, checkpoint_issues = recover_planner_checkpoint(
            exc.partial_output, tasks, num_models
        )
        if profile is None:
            logger.warning(
                "Planner profile was truncated; performing one controlled full-context retry"
            )
            try:
                raw = _call_planner(
                    user_prompt,
                    response_model=response_model,
                    item_count=requested_slots,
                    max_tokens=25000,
                    operation="initial_planner",
                )
            except Exception as retry_exc:
                logger.warning("Controlled planner retry failed | exc=%s", retry_exc)
                return {
                    "plans": [], "assessment_profile": {}, "planning_issues": [],
                    "plan_errors": [f"Planner call failed: {retry_exc}"],
                    "plan_attempts": 2, "last_plan_raw": exc.partial_output,
                }
            result = _store_plan_result(state, raw, tasks, num_models)
            result["plan_attempts"] = 2
            return result

        checkpoint_raw = {
            "assessment_profile": profile,
            "planning_issues": checkpoint_issues,
            "exams": checkpoint_plans,
        }
        checkpoint = _store_plan_result(state, checkpoint_raw, tasks, num_models)
        expected_slots = _expected_supported_slots(tasks, num_models, checkpoint_issues)
        completed = _plan_items_by_slot(checkpoint["plans"])
        missing_slots = [sid for sid in expected_slots if sid not in completed]
        logger.warning(
            "Planner truncated after checkpoint | completed=%d | missing=%d",
            len(completed), len(missing_slots),
        )
        if not missing_slots:
            checkpoint["plan_attempts"] = 1
            checkpoint["last_plan_raw"] = exc.partial_output
            return checkpoint
        try:
            continuation = _continue_plan_slots(
                state, checkpoint["plans"], profile, checkpoint_issues, missing_slots
            )
            merged = _merge_plan_replacements(
                checkpoint["plans"], continuation
            )
            merged = finalize_plans(merged, tasks)
            errors = validate_assessment_profile(
                profile,
                tasks,
                {
                    str(chunk.get("child_id"))
                    for chunk in state.get("retrieved_chunks") or []
                    if chunk.get("child_id")
                },
                checkpoint_issues,
            )
            errors.extend(validate_plans(
                merged, tasks, num_models,
                {
                    str(chunk.get("child_id"))
                    for chunk in state.get("retrieved_chunks") or []
                    if chunk.get("child_id")
                },
                difficulty, profile, checkpoint_issues,
            ))
            return {
                "plans": merged,
                "assessment_profile": profile,
                "planning_issues": checkpoint_issues,
                "plan_errors": errors,
                "plan_attempts": 1,
                "last_plan_raw": exc.partial_output,
            }
        except Exception as continuation_exc:
            logger.exception("Bounded planner continuation failed")
            checkpoint["plan_attempts"] = 1
            checkpoint["last_plan_raw"] = exc.partial_output
            return checkpoint
    except Exception as exc:
        logger.exception("Planning call failed | attempt=1")
        return {
            "plans": [],
            "assessment_profile": {},
            "planning_issues": [],
            "plan_errors": [f"Planner call failed: {exc}"],
            "plan_attempts": 1,
            "last_plan_raw": "",
        }

    result = _store_plan_result(state, raw, tasks, num_models)
    result["plan_attempts"] = 1
    logger.info(
        "Planning completed | attempt=1 | errors=%d | plans=%d",
        len(result["plan_errors"]),
        len(result["plans"]),
    )
    return result


_SLOT_RE = re.compile(
    r"\b(m\d+_(?:mcq|true_false|fill_in_the_blank|definition|why|equation|word_problem|essay)_\d+)\b"
)


def _invalid_slot_ids(errors: list[str]) -> list[str]:
    """Return the repair target at the start of each slot validation error."""
    result: list[str] = []
    for error in errors:
        match = re.search(rf"\bSlot\s+({_SLOT_RE.pattern})", error)
        if match:
            sid = match.group(1)
            if sid not in result:
                result.append(sid)
    return result


def _continue_plan_slots(
    state: ExamState,
    plans: list[dict[str, Any]],
    assessment_profile: dict[str, Any],
    planning_issues: list[dict[str, Any]],
    missing_slots: list[str],
) -> list[dict[str, Any]]:
    """Continue a truncated plan using only profile-shortlisted evidence."""
    chunks = state.get("retrieved_chunks") or []
    types = {
        location[1] for sid in missing_slots
        if (location := _slot_location(sid)) is not None
    }
    type_profiles = assessment_profile.get("question_types") or {}
    relevant_ids = {
        str(chunk_id)
        for qtype in types
        for chunk_id in ((type_profiles.get(qtype) or {}).get("evidence_chunk_ids") or [])
    }
    relevant_chunks = [
        chunk for chunk in chunks if str(chunk.get("child_id")) in relevant_ids
    ]
    source = "\n\n".join(
        f"[source_chunk_id={chunk.get('child_id')}]\n{chunk.get('content', '')}"
        for chunk in relevant_chunks
    )
    occupied = [
        (str(item.get("slot_id") or ""), duplicate_task_key(item), scenario_repetition_key(item))
        for item in _plan_items_by_slot(plans).values()
    ]
    prompt = (
        "Continue a truncated exam plan. Return one root JSON object with a `replacements` array "
        "containing exactly one complete replacement object for every missing slot_id and no other "
        "slots. Preserve the assessment profile. Each replacement "
        "must include all required plan fields, be text-answerable, and must not duplicate any "
        "occupied task or scenario across question types. Use only the supplied relevant chunks.\n\n"
        f"Missing slot IDs: {missing_slots}\n"
        f"Assessment profile: {json.dumps(assessment_profile, ensure_ascii=False)}\n"
        f"Planning issues: {json.dumps(planning_issues, ensure_ascii=False)}\n"
        f"Occupied task/scenario fingerprints: {json.dumps(occupied, ensure_ascii=False)}\n"
        f"Requested difficulty: {state.get('difficulty') or 'mix'}\n\n"
        f"Relevant evidence only:\n{source}"
    )
    chunk_ids = [str(chunk.get("child_id")) for chunk in relevant_chunks]
    response_model = plan_repair_response_model(missing_slots, chunk_ids)
    raw = _call_planner(
        prompt,
        response_model=response_model,
        item_count=len(missing_slots),
        max_tokens=min(8000, max(2048, len(missing_slots) * 500)),
        schema_name="planner_continuation",
        expected_ids=missing_slots,
        operation="planner_continuation",
        agent="planner_continuation",
    )
    replacements = raw.get("replacements") if isinstance(raw, dict) else None
    if not isinstance(replacements, list):
        raise ValueError("Planner continuation returned no replacements")
    return replacements


def _slot_location(sid: str) -> tuple[int, str, int] | None:
    match = re.fullmatch(
        r"m(\d+)_(mcq|true_false|fill_in_the_blank|definition|why|equation|word_problem|essay)_(\d+)", sid
    )
    if not match:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def _merge_plan_replacements(
    plans: list[dict[str, Any]], replacements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = deepcopy(plans)
    by_number = {int(p["model_number"]): p for p in merged}
    for replacement in replacements:
        sid = str(replacement.get("slot_id") or "")
        location = _slot_location(sid)
        if not location:
            continue
        model_number, qtype, index = location
        model = by_number.get(model_number)
        if model is None:
            model = {"model_number": model_number, "items": {}}
            merged.append(model)
            by_number[model_number] = model
        values = model["items"].setdefault(qtype, [])
        normalized = {
            "slot_id": sid,
            "task_id": str(replacement.get("task_id") or ""),
            "task_family_id": str(replacement.get("task_family_id") or ""),
            "coverage_id": str(replacement.get("coverage_id") or ""),
            "concept_id": str(replacement.get("concept_id") or ""),
            "relationship_id": replacement.get("relationship_id"),
            "source_chunk_id": str(replacement.get("source_chunk_id") or "").strip(),
            "support_chunk_ids": list(dict.fromkeys(
                str(chunk_id).strip()
                for chunk_id in (replacement.get("support_chunk_ids") or [])
                if str(chunk_id).strip()
            )),
            "topic": str(replacement.get("topic") or "").strip(),
            "concept": str(replacement.get("concept") or "").strip(),
            "canonical_skill": str(
                replacement.get("canonical_skill")
                or replacement.get("idea_to_test")
                or replacement.get("concept") or ""
            ).strip(),
            "knowledge_focus": str(replacement.get("knowledge_focus") or "").strip(),
            "student_work_product": str(
                replacement.get("student_work_product") or ""
            ).strip(),
            "relationship": (
                str(replacement.get("relationship")).strip()
                if replacement.get("relationship") is not None else None
            ),
            "target": str(replacement.get("target") or "").strip(),
            "task_operation": str(replacement.get("task_operation") or "").strip(),
            "given_conditions": [
                str(value).strip() for value in (replacement.get("given_conditions") or [])
                if str(value).strip()
            ],
            "scenario_family": (
                str(replacement.get("scenario_family")).strip()
                if replacement.get("scenario_family") is not None else None
            ),
            "required_givens": [
                str(value).strip() for value in (replacement.get("required_givens") or [])
                if str(value).strip()
            ],
            "text_answerable": replacement.get("text_answerable", True),
            "reasoning_mode": str(replacement.get("reasoning_mode") or "").strip(),
            "difficulty": str(replacement.get("difficulty") or "").strip().lower(),
            "idea_to_test": str(replacement.get("idea_to_test") or "").strip(),
        }
        for field in (
            "plan_repair_conflicts_with", "plan_repair_conflict_reason",
            "plan_repair_required_differences", "plan_repair_resolution_reason",
            "task_inventory", "task_reused", "reuse_reason",
            "reused_due_to_catalog_exhaustion", "reused_task_id",
            "catalog_replacement_stage", "forced_duplicate_slot_ids",
            "required_instance_distinctions", "suggested_assessment_angle",
        ):
            if field in replacement:
                normalized[field] = deepcopy(replacement[field])
        normalized["concept_to_test"] = normalized["idea_to_test"]
        by_slot = {str(value.get("slot_id") or ""): value for value in values}
        by_slot[sid] = normalized
        model["items"][qtype] = sorted(
            by_slot.values(),
            key=lambda value: (_slot_location(str(value.get("slot_id") or "")) or (0, qtype, 0))[2],
        )
    return sorted(merged, key=lambda plan: int(plan["model_number"]))


def _slot_errors(errors: list[str], sid: str) -> list[str]:
    return [error for error in errors if sid in error]


def _conflicting_slot_ids(errors: list[str], sid: str) -> list[str]:
    """Extract the other concrete slots named in this slot's validation errors."""
    return sorted({
        found
        for error in _slot_errors(errors, sid)
        for found in re.findall(r"m\d+_[a-z_]+_\d+", error)
        if found != sid
    })


def _repair_semantic_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key) for key in (
            "slot_id", "task_id", "task_family_id", "coverage_id", "concept",
            "task_operation", "target", "reasoning_mode", "student_work_product",
            "relationship", "source_chunk_id", "support_chunk_ids",
        )
    }


def _candidate_resolves_conflicts(
    candidate: dict[str, Any], conflicts: list[dict[str, Any]],
) -> bool:
    """Hard gate: replacement must materially differ from every conflicting slot."""
    candidate_task_id = str(candidate.get("task_id") or "")
    semantic_fields = (
        "concept", "task_operation", "target", "reasoning_mode",
        "student_work_product", "relationship", "task_family_id",
    )
    for conflict in conflicts:
        if candidate_task_id and candidate_task_id == str(conflict.get("task_id") or ""):
            return False
        if _semantic_plan_conflict(candidate, conflict):
            return False
        differences = sum(
            _task_part(candidate.get(field)) != _task_part(conflict.get(field))
            for field in semantic_fields
        )
        if differences < 2:
            return False
    return True


def _semantic_catalog_replacement(
    state: ExamState, plans: list[dict[str, Any]], sid: str, qtype: str,
    errors: list[str],
) -> dict[str, Any] | None:
    """Ask for a grounded repair goal, then enforce that goal against catalog tasks."""
    from app.online.assessment_catalog import (
        _difficulty_for,
        _replacement_representation_rank,
        compatible_task_inventory,
        materialize_plan_item,
    )

    catalog = state.get("assessment_catalog") or {}
    items = _plan_items_by_slot(plans)
    current = items.get(sid) or {}
    conflict_ids = _conflicting_slot_ids(errors, sid)
    conflicts = [items[value] for value in conflict_ids if value in items]
    if not conflicts:
        return None
    valid_chunk_ids = {
        str(chunk.get("child_id")) for chunk in state.get("retrieved_chunks") or []
        if chunk.get("child_id")
    }
    difficulty = str(state.get("difficulty") or "mix")
    represented = [item for slot, item in items.items() if slot != sid]
    used_ids = {str(item.get("task_id") or "") for item in represented if item.get("task_id")}
    candidates: list[dict[str, Any]] = []
    for task in catalog.get("tasks") or []:
        task_id = str(task.get("task_id") or "")
        evidence = [str(value) for value in task.get("evidence_chunk_ids") or []]
        level = _difficulty_for(task, difficulty)
        if (
            not task_id or task_id == str(current.get("task_id") or "")
            or qtype not in (task.get("allowed_question_types") or [])
            or level is None or task.get("text_answerable") is not True
            or not evidence or any(value not in valid_chunk_ids for value in evidence)
        ):
            continue
        materialized = materialize_plan_item(task, sid, level)
        if not _candidate_resolves_conflicts(materialized, conflicts):
            continue
        rank = _replacement_representation_rank(
            task, represented, include_task_usage=True
        )
        candidates.append({
            **_repair_semantic_record(materialized),
            "selection_type": "intentional_reuse" if task_id in used_ids else "unused_task",
            "representation_rank": list(rank),
        })
    candidates.sort(key=lambda value: (
        value["selection_type"] != "unused_task",
        *value["representation_rank"], str(value.get("task_id") or ""),
    ))
    if not candidates:
        return None
    prompt = (
        "Repair exactly one invalid deterministic-plan slot. First explain the semantic conflict, "
        "then select only a replacement_task_id from the supplied candidates. The replacement must "
        "change the assessed operation, target/result, reasoning path, or student work product in a "
        "material way. Changing wording, numbers, or scenario alone is invalid. Prefer unused_task. "
        "Choose intentional_reuse only when no unused candidate resolves the conflict.\n\n"
        f"Broken slot:\n{json.dumps(_repair_semantic_record(current), ensure_ascii=False)}\n\n"
        f"Conflicting slots:\n{json.dumps([_repair_semantic_record(x) for x in conflicts], ensure_ascii=False)}\n\n"
        f"Validation errors:\n{json.dumps(_slot_errors(errors, sid), ensure_ascii=False)}\n\n"
        f"Allowed candidates in ranked priority order:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    try:
        raw = request_structured(
            create_llm_client(), prompt, response_model=SemanticPlanRepairResponse,
            schema_name="semantic_plan_slot_repair", agent="plan_repairer",
            item_count=1, expected_ids=[sid], expected_id_field="slot_to_replace",
            system_prompt=PLANNER_SYSTEM_PROMPT, temperature=0.1, max_tokens=2048,
        )
        parsed = (
            raw if isinstance(raw, SemanticPlanRepairResponse)
            else SemanticPlanRepairResponse.model_validate(raw)
        ).repair
    except Exception:
        logger.exception("Semantic targeted slot repair failed | slot=%s", sid)
        return None
    allowed = {str(candidate["task_id"]): candidate for candidate in candidates}
    chosen_row = allowed.get(str(parsed.replacement_task_id))
    if (
        parsed.slot_to_replace != sid
        or set(parsed.conflicts_with) != set(conflict_ids)
        or chosen_row is None
    ):
        return None
    unused_exists = any(row["selection_type"] == "unused_task" for row in candidates)
    if unused_exists and chosen_row["selection_type"] != "unused_task":
        return None
    task = next(
        task for task in catalog.get("tasks") or []
        if str(task.get("task_id") or "") == str(parsed.replacement_task_id)
    )
    replacement = materialize_plan_item(
        task, sid, _difficulty_for(task, difficulty) or str(current.get("difficulty") or "medium")
    )
    if not _candidate_resolves_conflicts(replacement, conflicts):
        return None
    reused = str(replacement.get("task_id") or "") in used_ids
    inventory = compatible_task_inventory(
        catalog, plans, qtype, difficulty, excluding_slot_id=sid
    )
    replacement.update({
        "plan_repair_conflicts_with": conflict_ids,
        "plan_repair_conflict_reason": parsed.conflict_reason,
        "plan_repair_required_differences": list(parsed.what_must_be_different),
        "plan_repair_resolution_reason": parsed.why_this_replacement_resolves_the_conflict,
        "task_inventory": inventory,
        "task_reused": reused,
    })
    if reused:
        previous_slots = sorted(
            slot for slot, item in items.items()
            if slot != sid and str(item.get("task_id") or "") == str(replacement.get("task_id") or "")
        )
        replacement.update({
            "reuse_reason": "no_unused_compatible_task",
            "reused_due_to_catalog_exhaustion": True,
            "reused_task_id": replacement.get("task_id"),
            "catalog_replacement_stage": "semantic_plan_repair_intentional_reuse",
            "forced_duplicate_slot_ids": previous_slots,
            "required_instance_distinctions": [{
                "kind": "different_assessment_angle",
                "requirement": value,
                "reference_slot_ids": conflict_ids,
            } for value in parsed.what_must_be_different],
            "suggested_assessment_angle": parsed.why_this_replacement_resolves_the_conflict,
        })
    else:
        replacement["catalog_replacement_stage"] = "semantic_plan_repair_unused_task"
    return replacement


def _repair_plan_slot(
    state: ExamState,
    plans: list[dict[str, Any]],
    sid: str,
    errors: list[str],
    conflict_slot_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Try one independent, bounded replacement and accept it only if globally valid."""
    location = _slot_location(sid)
    if location is None:
        return plans, errors, False
    _model_number, qtype, _index = location
    catalog = state.get("assessment_catalog") or {}
    if catalog:
        replacement = _semantic_catalog_replacement(
            state, plans, sid, qtype, errors
        )
        if replacement is None:
            return plans, errors, False
        candidate = _merge_plan_replacements(plans, [replacement])
        candidate = finalize_plans(candidate, state.get("tasks") or [])
        candidate_errors = validate_plans(
            candidate,
            state.get("tasks") or [],
            state.get("num_models") or 1,
            {
                str(chunk.get("child_id"))
                for chunk in state.get("retrieved_chunks") or [] if chunk.get("child_id")
            },
            state.get("difficulty") or "mix",
            state.get("assessment_profile") or {},
            state.get("planning_issues") or [],
        )
        other_errors = [error for error in candidate_errors if sid not in error]
        if any(sid in error for error in candidate_errors) or len(other_errors) > len(
            [error for error in errors if sid not in error]
        ):
            return plans, errors, False
        return candidate, candidate_errors, True
    assessment_profile = state.get("assessment_profile") or {}
    planning_issues = state.get("planning_issues") or []
    chunks = state.get("retrieved_chunks") or []
    all_chunk_ids = {
        str(chunk.get("child_id")) for chunk in chunks if chunk.get("child_id")
    }
    current = _plan_items_by_slot(plans).get(sid) or {}
    original_identity = (
        duplicate_task_key(current), scenario_repetition_key(current),
        tuple(current.get("required_givens") or []), current.get("source_chunk_id"),
    )
    type_profile = ((assessment_profile.get("question_types") or {}).get(qtype) or {})
    relevant_ids = {
        str(value) for value in (type_profile.get("evidence_chunk_ids") or [])
        if str(value) in all_chunk_ids
    }
    relevant_ids.update(
        str(value) for value in [
            current.get("source_chunk_id"), *(current.get("support_chunk_ids") or [])
        ] if str(value) in all_chunk_ids
    )
    relevant_chunks = [
        chunk for chunk in chunks if str(chunk.get("child_id")) in relevant_ids
    ]
    source = "\n\n".join(
        f"[source_chunk_id={chunk.get('child_id')}]\n{chunk.get('content', '')}"
        for chunk in relevant_chunks
    )
    occupied = [
        {
            "slot_id": other_sid,
            "knowledge_focus": item.get("knowledge_focus"),
            "student_work_product": item.get("student_work_product"),
            "canonical_skill": item.get("canonical_skill"),
            "task_operation": item.get("task_operation"),
            "relationship": item.get("relationship") or item.get("concept"),
            "target": item.get("target"),
            "reasoning_mode": item.get("reasoning_mode"),
            "given_conditions": item.get("given_conditions") or [],
            "scenario_family": item.get("scenario_family"),
            "idea_to_test": item.get("idea_to_test"),
        }
        for other_sid, item in _plan_items_by_slot(plans).items()
        if other_sid != sid
    ]
    conflict_fingerprint = next(
        (item for item in occupied if item.get("slot_id") == conflict_slot_id), None
    )
    compact_profile = {
        "domains": assessment_profile.get("domains") or [],
        "educational_level": assessment_profile.get("educational_level") or "unspecified",
        "assessment_style": assessment_profile.get("assessment_style") or "",
        "question_type": qtype,
        **type_profile,
        "unsupported_visual_tasks": assessment_profile.get("unsupported_visual_tasks") or [],
    }
    prompt = (
        f"Return one root JSON object with a `replacements` array containing exactly one replacement "
        f"for invalid plan slot {sid}. Preserve its question type ({qtype}) and "
        f"obey the requested difficulty ({state.get('difficulty') or 'mix'}). For a fixed easy, "
        "medium, or hard request, the replacement difficulty must match it exactly. For mix, choose "
        "a level the evidence genuinely supports. task_operation must exactly match one of the "
        "operations in the compact question-type profile. If difficulty is hard, idea_to_test must "
        "explicitly describe the deeper work (multi-step combination, inference, comparison, proof, "
        "error analysis, or an intermediate conclusion). The original idea is not sacred: "
        "when it is invalid, choose a different grounded skill, task_operation, relationship, "
        "target, scenario, or givens. Update the task and all metadata together. Never invent a "
        "value or depend on an unavailable image, table, graph, diagram, or label.\n\n"
        "For Fill-in-the-Blank, the target answer must be a word, term, short phrase, concept "
        "name, unit name, or short symbol/name; it must never be an equation, formula, numerical "
        "result, or multi-step result. For derive_relationship, the idea must explicitly require "
        "a real source-grounded derivation; otherwise choose a different supported operation and "
        "matching reasoning_mode/target.\n\n"
        f"Validation failure:\n" + "\n".join(f"- {e}" for e in _slot_errors(errors, sid)) +
        f"\n\nCurrent invalid slot:\n{json.dumps(current, ensure_ascii=False)}\n\n"
        f"Compact assessment guidance:\n{json.dumps(compact_profile, ensure_ascii=False)}\n\n"
        f"Conflicting slot fingerprint:\n{json.dumps(conflict_fingerprint, ensure_ascii=False)}\n\n"
        "For duplicate_or_repetition, changing only wording, question type, material names, or "
        "numbers is invalid. Change knowledge_focus first when evidence permits; otherwise change "
        "student_work_product together with operation and target so the student performs genuinely "
        "different work. Do not use any occupied knowledge_focus for another recall-level task.\n\n"
        f"Occupied global fingerprints and prohibited targets:\n{json.dumps(occupied, ensure_ascii=False)}\n\n"
        f"Relevant evidence only:\n{source}"
    )
    try:
        raw = request_structured(
            create_llm_client(),
            prompt,
            response_model=plan_repair_response_model([sid], sorted(relevant_ids)),
            schema_name="plan_slot_repair",
            agent="plan_repairer",
            item_count=1,
            expected_ids=[sid],
            expected_id_field="slot_id",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=2048,
        )
        replacements = raw.get("replacements") if isinstance(raw, dict) else None
        if not isinstance(replacements, list) or len(replacements) != 1:
            return plans, errors, False
        candidate = finalize_plans(_merge_plan_replacements(plans, replacements), state.get("tasks") or [])
        candidate_errors = validate_plans(
            candidate,
            state.get("tasks") or [],
            state.get("num_models") or 1,
            all_chunk_ids,
            state.get("difficulty") or "mix",
            assessment_profile,
            planning_issues,
        )
    except Exception as exc:
        logger.exception("Targeted slot repair failed | slot=%s", sid)
        return plans, errors, False

    intentionally_missing = {
        str(item.get("slot_id") or "")
        for item in (state.get("unrepairable_plan_slots") or [])
        if item.get("slot_id")
    }
    # A previously exhausted slot is an accepted shortfall. It must not make an
    # unrelated later replacement fail the full-plan count audit.
    candidate_errors = [
        error for error in candidate_errors
        if not any(missing_sid in error for missing_sid in intentionally_missing)
    ]
    baseline_bad = set(_invalid_slot_ids(errors))
    candidate_bad = set(_invalid_slot_ids(candidate_errors))
    introduced_bad = candidate_bad - (baseline_bad - {sid})
    replacement = _plan_items_by_slot(candidate).get(sid) or {}
    replacement_identity = (
        duplicate_task_key(replacement), scenario_repetition_key(replacement),
        tuple(replacement.get("required_givens") or []), replacement.get("source_chunk_id"),
    )
    requires_new_idea = any(
        marker in error.casefold()
        for error in _slot_errors(errors, sid)
        for marker in (
            "missing_required_given", "visual_dependency", "missing visual",
            "missing data", "original idea", "unavailable image", "unavailable table",
            "duplicate_or_repetition", "duplicates the planned", "repeats the underlying",
        )
    )
    fixed = (
        sid not in candidate_bad
        and not introduced_bad
        and (not requires_new_idea or replacement_identity != original_identity)
    )
    if not fixed:
        logger.info(
            "Targeted slot repair rejected | slot=%s | candidate_errors=%s",
            sid, candidate_errors,
        )
        return plans, errors, False
    logger.info("Targeted slot repair accepted | slot=%s", sid)
    return candidate, candidate_errors, True


def repair_plan(state: ExamState) -> dict[str, Any]:
    """Repair invalid slots independently; fail if requested structure cannot be preserved."""
    errors = list(state.get("plan_errors") or [])
    plans = deepcopy(state.get("plans") or [])
    attempt = (state.get("plan_attempts") or 0) + 1
    slot_ids = _invalid_slot_ids(errors)
    logger.info(
        "Targeted plan repair | document_id=%s | attempt=%d/%d | errors=%d | slots=%s",
        state.get("document_id"), attempt, MAX_PLAN_ATTEMPTS, len(errors), slot_ids,
    )
    if not errors:
        return {"plan_errors": [], "plan_attempts": attempt}
    if not slot_ids:
        return {
            "plans": plans,
            "plan_errors": errors,
            "plan_attempts": attempt,
            "last_plan_raw": state.get("last_plan_raw") or "",
        }

    for sid in slot_ids:
        if sid not in set(_invalid_slot_ids(errors)):
            continue
        plans, errors, _accepted = _repair_plan_slot(state, plans, sid, errors)

    unresolved = _invalid_slot_ids(errors)
    prior_unrepairable = list(state.get("unrepairable_plan_slots") or [])
    if unresolved and attempt >= MAX_PLAN_ATTEMPTS:
        details = "; ".join(
            f"{sid}: {' '.join(_slot_errors(errors, sid))}" for sid in unresolved
        )
        logger.error(
            "Plan repair exhausted; refusing structurally incomplete plan | slots=%s",
            unresolved,
        )
        raise RuntimeError(
            "Planner repair could not preserve every requested slot: " + details
        )

    return {
        "plans": plans,
        "plan_errors": errors,
        "plan_attempts": attempt,
        "unrepairable_plan_slots": prior_unrepairable,
        "last_plan_raw": json.dumps(plans, ensure_ascii=False),
    }

def build_planner_context_from_chunks(
    children: list[dict], snippet_tokens: int = PLANNER_SNIPPET_TOKENS
) -> str:
    """Convenience re-export used by tests/other modules if needed."""
    from app.online.retrieval import build_planner_context

    return build_planner_context(children, snippet_tokens)
