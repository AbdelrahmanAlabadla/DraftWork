from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from app.config import (
    MAX_PARALLEL_VALIDATOR_CALLS,
    VALIDATOR_BATCH_SIZE,
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
from app.online.assessment_catalog import is_pure_text_fitb_value, normalize_fitb_answer_key
from app.online.models import (
    _normalize_definition,
    _normalize_equation,
    _normalize_essay,
    _normalize_mcq,
    _normalize_why,
    _normalize_true_false,
    _normalize_word_problem,
    normalize_fitb_item,
    normalize_text,
)

logger = get_logger("VALIDATOR")

# Smallest possible repair actions produced by the validator and consumed by the
# repair stage. ``PASS`` means the question is left byte-identical.
ACTIONS = frozenset(
    {"PASS", "FIX_QUESTION", "FIX_ANSWER", "FIX_OPTIONS", "FIX_QUESTION_AND_ANSWER"}
)
REPAIR_ACTIONS = frozenset(
    {"FIX_QUESTION", "FIX_ANSWER", "FIX_OPTIONS", "FIX_QUESTION_AND_ANSWER"}
)

MAX_QUESTION_REPAIR_ATTEMPTS = 2

VALIDATOR_TEMPERATURE = 0.1
# Some served models (e.g. qwen/qwen3-8b) only support reasoning 'off'/'on'.
# 'off' keeps the call fast and non-creative while still producing verdicts.
VALIDATOR_REASONING = "on"
# DeepSeek can spend a large part of its response budget on hidden reasoning.
# Give the general validator its final budget up front instead of paying for an
# avoidable 12k -> 24k -> 32k output-limit retry chain.
GENERAL_VALIDATOR_MAX_TOKENS = 100_000
STEM_QUANTITATIVE_VALIDATOR_MAX_TOKENS = 12_000
# Backwards-compatible name for callers/tests that import the historical
# constant. It now describes the general validator ceiling.
VALIDATOR_MAX_TOKENS = GENERAL_VALIDATOR_MAX_TOKENS

GLOBAL_VALIDATOR_TEMPERATURE = 0.1
GLOBAL_VALIDATOR_REASONING = "on"
GLOBAL_VALIDATOR_MAX_TOKENS = 12000

REPAIR_TEMPERATURE = 0.1
# Repairer reasoning is deliberately disabled for both DeepSeek and local
# fallback clients; the dedicated model is evaluated on direct instruction
# following, not hidden chain-of-thought output.
REPAIR_REASONING = "off"
REPAIR_MAX_TOKENS = 12000

TASK_REPLACEMENT_FAILURES: frozenset[str] = frozenset({
    "task_drift", "question_type_operation_mismatch", "visual_dependency",
    "mcq_task_unsuitable", "unsupported_given", "evidence_mismatch",
    "bad_evidence_assignment", "wrong_task_identity",
    "bad_fitb_answer_evidence",
})

STRUCTURE_FAILURES: frozenset[str] = frozenset({
    "invalid_structure", "missing_required_field", "invalid_options",
    "invalid_word_bank", "malformed_question",
})


def _requires_plan_replacement(verdict: Dict[str, Any]) -> bool:
    return (
        str(verdict.get("failure_code") or "") in TASK_REPLACEMENT_FAILURES
        or any(
            str(field).startswith(("semantic_identity.", "plan."))
            for field in (verdict.get("fields_to_fix") or [])
        )
    )

# Canonical student-facing fields. Validator, repair prompt and writeback all
# use these exact names; there is no ambiguous generic ``answer`` alias.
_LOGICAL_TO_CANONICAL: Dict[str, Dict[str, Set[str]]] = {
    "mcq": {
        "question": {"question"},
        "options": {"options"},
        "correct_answer": {"correct_answer"},
        "solution_certificate": {"solution_certificate"},
    },
    "true_false": {
        "statement": {"statement"},
        "answer": {"answer"},
        "solution_certificate": {"solution_certificate"},
    },
    "fill_in_the_blank": {
        "question": {"question"},
        "distractors": {"distractors"},
    },
    "definition": {
        "term": {"term"},
        "reference_answer": {"reference_answer"},
        "solution_certificate": {"solution_certificate"},
    },
    "why": {
        "question": {"question"},
        "reference_answer": {"reference_answer"},
        "solution_certificate": {"solution_certificate"},
    },
    "equation": {
        "equation": {"equation"},
        "solution_steps": {"solution_steps"},
        "final_answer": {"final_answer"},
        "solution_certificate": {"solution_certificate"},
    },
    "word_problem": {
        "question": {"question"},
        "solution_steps": {"solution_steps"},
        "final_answer": {"final_answer"},
        "solution_certificate": {"solution_certificate"},
    },
    "essay": {
        "question": {"question"},
        "reference_answer": {"reference_answer"},
        "key_points": {"key_points"},
        "solution_certificate": {"solution_certificate"},
    },
}

# Default canonical fields per (qtype, action), used only when the validator
# omits ``fields_to_fix``. Explicit unknown names are rejected.
_ACTION_DEFAULT_FIELDS: Dict[str, Dict[str, List[str]]] = {
    "mcq": {
        "FIX_ANSWER": ["correct_answer"],
        "FIX_OPTIONS": ["options", "correct_answer"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "options", "correct_answer"],
    },
    "true_false": {
        "FIX_ANSWER": ["answer"],
        "FIX_OPTIONS": ["answer"],
        "FIX_QUESTION": ["statement"],
        "FIX_QUESTION_AND_ANSWER": ["statement", "answer"],
    },
    "fill_in_the_blank": {
        "FIX_ANSWER": ["question"],
        "FIX_OPTIONS": ["distractors"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "distractors"],
    },
    "definition": {
        "FIX_ANSWER": ["reference_answer"],
        "FIX_OPTIONS": ["reference_answer"],
        "FIX_QUESTION": ["term"],
        "FIX_QUESTION_AND_ANSWER": ["term", "reference_answer"],
    },
    "why": {
        "FIX_ANSWER": ["reference_answer"],
        "FIX_OPTIONS": ["reference_answer"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "reference_answer"],
    },
    "equation": {
        "FIX_ANSWER": ["solution_steps", "final_answer"],
        "FIX_OPTIONS": ["solution_steps", "final_answer"],
        "FIX_QUESTION": ["equation"],
        "FIX_QUESTION_AND_ANSWER": ["equation", "solution_steps", "final_answer"],
    },
    "word_problem": {
        "FIX_ANSWER": ["solution_steps", "final_answer"],
        "FIX_OPTIONS": ["solution_steps", "final_answer"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "solution_steps", "final_answer"],
    },
    "essay": {
        "FIX_ANSWER": ["reference_answer", "key_points"],
        "FIX_OPTIONS": ["reference_answer", "key_points"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "reference_answer", "key_points"],
    },
}

_ALL_CANONICAL: Dict[str, Set[str]] = {
    "mcq": {"question", "options", "correct_answer", "solution_certificate"},
    "true_false": {"statement", "answer", "solution_certificate"},
    "fill_in_the_blank": {
        "question", "distractors"
    },
    "definition": {"term", "reference_answer", "solution_certificate"},
    "why": {"question", "reference_answer", "solution_certificate"},
    "equation": {"equation", "solution_steps", "final_answer", "solution_certificate"},
    "word_problem": {"question", "solution_steps", "final_answer", "solution_certificate"},
    "essay": {"question", "reference_answer", "key_points", "solution_certificate"},
}

# Cheap local detectors that never need an LLM call.
_BLANK_MARKERS = ("___", "____", "_____", "________", "…")


def _fitb_term_answer(value: object) -> bool:
    """Reject calculations/formulas while allowing concise vocabulary."""
    text = str(value or "").strip()
    if not text or len(text) > 80 or len(text.split()) > 8 or not is_pure_text_fitb_value(text):
        return False
    if "=" in text or re.fullmatch(
        r"\s*[+\-]?\d+(?:\.\d+)?(?:\s*[+\-*/^??]\s*\d+(?:\.\d+)?)+\s*",
        text,
    ):
        return False
    if re.fullmatch(r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*\S+)?", text):
        return False
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*){2,}", text):
        return False
    return True


def _solve_system_completion_verdict(
    item: Dict[str, Any], plan_item: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if str((plan_item or {}).get("task_operation") or "") not in {
        "solve_system", "solve_system_of_equations", "solve_simultaneous_equations"
    }:
        return None
    task = str(item.get("equation") or item.get("question") or "")
    # Catalog metadata may describe a broad solve-system skill even when the
    # student-facing request asks for only one derived quantity. Completeness
    # must follow the actual request, not expand it from Planner metadata.
    normalized_task = " ".join(task.casefold().split())
    asks_for_complete_system = bool(
        re.search(r"\bsolve\s+(?:the\s+|this\s+)?(?:following\s+)?system\b", normalized_task)
        or (
            re.search(r"^solve\b", normalized_task)
            and normalized_task.count("=") >= 2
            and re.search(r"(?<![a-z])x(?![a-z])", normalized_task)
            and re.search(r"(?<![a-z])y(?![a-z])", normalized_task)
        )
        or re.search(
            r"\bfind\s+(?:the\s+)?(?:values?\s+of\s+)?x\s*(?:,|and)\s*y\b",
            normalized_task,
        )
        or re.search(r"\b(?:solution\s+)?ordered\s+pair\b", normalized_task)
        or any(
            phrase in normalized_task
            for phrase in (
                "حل النظام", "حل نظام", "أوجد حل النظام", "أوجد حل نظام",
                "أوجد قيمتي x و y", "أوجد قيم x و y", "الزوج المرتب",
            )
        )
    )
    if not asks_for_complete_system:
        return None
    answer = " ".join([
        *(str(step) for step in (item.get("solution_steps") or [])),
        str(item.get("final_answer") or ""),
    ])
    final_answer = str(item.get("final_answer") or "")
    final_pair = re.search(
        r"(?:\(\s*x\s*,\s*y\s*\)\s*=\s*)?"
        r"\(\s*([+\-]?\d+(?:\.\d+)?)\s*,\s*([+\-]?\d+(?:\.\d+)?)\s*\)",
        final_answer,
        flags=re.IGNORECASE,
    )
    variables = {
        value.casefold()
        for value in re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", task)
        if value.casefold() not in {"a", "b", "c"}
    }
    non_unique_classification = bool(re.search(
        r"(?i)\bno\s+solutions?\b|\binfinitely\s+many\s+solutions?\b|"
        r"لا\s+يوجد\s+حل|ليس\s+له\s+حل|عدد\s+لا\s+نهائي\s+من\s+الحلول|"
        r"حلول\s+لا\s+نهائية",
        answer,
    ))
    missing = sorted(
        variable for variable in variables
        if not re.search(rf"(?<![A-Za-z]){re.escape(variable)}\s*=", answer, re.IGNORECASE)
    )
    if missing and not non_unique_classification and not (
        final_pair and set(missing) <= {"x", "y"}
    ):
        return _verdict(
            True, False, "FIX_ANSWER", ["solution_steps", "final_answer"],
            "FAIL: incomplete_solution. The answer does not solve every variable requested "
            f"by the system; missing {', '.join(missing)}.",
            "Continue through every variable and provide the complete solution tuple.",
            "incomplete_solution", [f"value for {value}" for value in sorted(variables)],
        )

    # Independently solve the common, safely parseable ax + by = c form.
    def coefficient(raw: str) -> float:
        compact = raw.replace(" ", "")
        if compact in {"", "+"}:
            return 1.0
        if compact == "-":
            return -1.0
        return float(compact)

    matches = re.findall(
        r"([+\-]?\s*\d*(?:\.\d+)?)\s*x\s*"
        r"([+\-]\s*\d*(?:\.\d+)?)\s*y\s*=\s*([+\-]?\s*\d+(?:\.\d+)?)",
        task,
        flags=re.IGNORECASE,
    )
    if len(matches) >= 2:
        try:
            a1, b1, c1 = (coefficient(value) for value in matches[0])
            a2, b2, c2 = (coefficient(value) for value in matches[1])
            determinant = a1 * b2 - a2 * b1
            if abs(determinant) > 1e-12:
                expected_x = (c1 * b2 - c2 * b1) / determinant
                expected_y = (a1 * c2 - a2 * c1) / determinant
                found = {
                    name.casefold(): float(value)
                    for name, value in re.findall(
                        r"(?<![A-Za-z])([xy])\s*=\s*([+\-]?\d+(?:\.\d+)?)",
                        answer,
                        flags=re.IGNORECASE,
                    )
                }
                if final_pair:
                    found["x"] = float(final_pair.group(1))
                    found["y"] = float(final_pair.group(2))
                if (
                    "x" in found and "y" in found
                    and (
                        abs(found["x"] - expected_x) > 1e-6
                        or abs(found["y"] - expected_y) > 1e-6
                    )
                ):
                    expected = [f"x = {expected_x:g}", f"y = {expected_y:g}"]
                    return _verdict(
                        True, False, "FIX_ANSWER", ["solution_steps", "final_answer"],
                        "FAIL: incorrect_result. Deterministic substitution/linear-system "
                        f"checking gives {', '.join(expected)}, which conflicts with the answer.",
                        "Correct the solution steps and give the complete ordered pair.",
                        "incorrect_result", expected,
                    )
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return None


def _normalize(qtype: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize ONE repaired question through the existing normalizers."""
    if qtype == "mcq":
        return _normalize_mcq(item)
    if qtype == "true_false":
        return _normalize_true_false(item)
    if qtype == "fill_in_the_blank":
        return normalize_fitb_item(item)
    if qtype == "definition":
        return _normalize_definition(item)
    if qtype == "why":
        return _normalize_why(item)
    if qtype == "equation":
        return _normalize_equation(item)
    if qtype == "word_problem":
        return _normalize_word_problem(item)
    if qtype == "essay":
        return _normalize_essay(item)
    return None


def _canonical_values(qtype: str, q: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the canonical field values of a (normalized) question dict."""
    if qtype == "mcq":
        return {
            "question": q.get("question"),
            "options": q.get("options"),
            "correct_answer": q.get("correct_answer"),
            "solution_certificate": q.get("solution_certificate"),
        }
    if qtype == "true_false":
        return {"statement": q.get("statement"), "answer": q.get("answer"),
                "solution_certificate": q.get("solution_certificate")}
    if qtype == "fill_in_the_blank":
        return {
            "question": q.get("question"),
            "answers": q.get("answers"),
            "distractors": q.get("distractors"),
            "word_bank": q.get("word_bank"),
            "solution_certificate": q.get("solution_certificate"),
        }
    if qtype == "definition":
        return {"term": q.get("term"), "reference_answer": q.get("reference_answer"),
                "solution_certificate": q.get("solution_certificate")}
    if qtype == "why":
        return {
            "question": q.get("question"),
            "reference_answer": q.get("reference_answer"),
            "solution_certificate": q.get("solution_certificate"),
        }
    if qtype == "equation":
        return {
            "equation": q.get("equation"),
            "solution_steps": q.get("solution_steps"),
            "final_answer": q.get("final_answer"),
            "solution_certificate": q.get("solution_certificate"),
        }
    if qtype == "word_problem":
        return {
            "question": q.get("question"),
            "solution_steps": q.get("solution_steps"),
            "final_answer": q.get("final_answer"),
            "solution_certificate": q.get("solution_certificate"),
        }
    if qtype == "essay":
        return {
            "question": q.get("question"),
            "reference_answer": q.get("reference_answer"),
            "key_points": q.get("key_points"),
            "solution_certificate": q.get("solution_certificate"),
        }
    return {}


def _allowed_canonical(
        qtype: str, fields_to_fix: Optional[List[Any]], action: str
) -> Set[str]:
    """Translate explicit and action-required fields to safe canonical keys.

    Unknown names are ignored. Recognized wrapper paths are normalized, action
    defaults are always added, and answer/options-only actions impose a hard
    ceiling so explicit fields cannot widen their write authority.
    """
    mapping = _LOGICAL_TO_CANONICAL.get(qtype, {})

    def logical_name(raw: Any) -> str:
        name = str(raw).strip().lower()
        # The repair schema wraps the repaired object as `question`, so models
        # sometimes report question.question, question.statement, etc. Strip
        # only that recognized wrapper and only for a known field.
        if name.startswith("question."):
            candidate = name.split(".", 1)[1]
            if candidate in mapping:
                return candidate
        return name

    canon: Set[str] = set()
    requested_fields = list(fields_to_fix or [])
    for name in requested_fields:
        canon |= mapping.get(logical_name(name), set())
    # Defaults recover an omitted or wholly invalid field list. A valid explicit
    # list is authoritative and must not be widened during repair/writeback.
    if not canon and not any(str(name).strip() for name in requested_fields):
        defaults = _ACTION_DEFAULT_FIELDS.get(qtype, {}).get(action, [])
        for name in defaults:
            canon |= mapping.get(name, set())

    # Preserve action boundaries. In particular, a malformed explicit path can
    # never turn FIX_ANSWER into permission to rewrite the stem.
    if action == "FIX_ANSWER":
        answer_fields: Set[str] = set()
        for name in mapping:
            if name not in {"question", "statement", "term", "equation", "options", "word_bank"}:
                answer_fields |= mapping.get(name, set())
        canon &= answer_fields
    elif action == "FIX_OPTIONS" and qtype == "mcq":
        option_fields: Set[str] = set()
        for name in ("options", "correct_answer", "solution_certificate"):
            option_fields |= mapping.get(name, set())
        canon &= option_fields
    # Coupled fields are declared explicitly in the action defaults above.
    # Do not widen authorization here: validator fields, repair permissions,
    # and writeback permissions must remain the same canonical set.
    return canon


def _iter_questions(questions: Dict[str, Any]) -> Iterable[Tuple[str, Any, Dict[str, Any]]]:
    """Yield (qtype, container, item) for every question in an exam.

    For the FITB section the container is the section dict and each item is
    yielded separately so every item is validated/repaired individually.
    """
    for qtype, section in questions.items():
        if qtype == "fill_in_the_blank" and isinstance(section, dict):
            for item in section.get("items") or []:
                if isinstance(item, dict):
                    yield qtype, section, item
        elif isinstance(section, list):
            for item in section:
                if isinstance(item, dict):
                    yield qtype, section, item


def _build_by_id(
        questions: Dict[str, Any],
) -> Dict[str, Tuple[str, Any, Dict[str, Any]]]:
    by_id: Dict[str, Tuple[str, Any, Dict[str, Any]]] = {}
    for qtype, container, item in _iter_questions(questions):
        qid = item.get("question_id")
        if qid:
            by_id[str(qid)] = (qtype, container, item)
    return by_id


def _rebuild_fitb_word_bank(
    section: Dict[str, Any], preferred_question_ids: set[str] | None = None
) -> bool:
    """Rebuild the bank from active answers and exactly two shared distractors."""
    answers: list[str] = []
    answer_keys: set[str] = set()
    for item in section.get("items") or []:
        for value in item.get("answers") or []:
            term = str(value).strip()
            key = normalize_fitb_answer_key(term)
            if term and key not in answer_keys:
                answer_keys.add(key)
                answers.append(term)
    distractors: list[str] = []
    seen = set(answer_keys)
    ordered_items = list(section.get("items") or [])
    if preferred_question_ids:
        ordered_items.sort(key=lambda item: (
            str(item.get("question_id") or "") not in preferred_question_ids
        ))
    values = [
        value for item in ordered_items
        for value in item.get("distractors") or []
    ]
    for value in values:
        term = str(value).strip()
        key = normalize_fitb_answer_key(term)
        if (
            not term or key in seen or not _fitb_term_answer(term)
            or not is_pure_text_fitb_value(term)
        ):
            return False
        seen.add(key)
        distractors.append(term)
        if len(distractors) == 2:
            break
    if len(answers) != len(section.get("items") or []) or len(distractors) != 2:
        return False
    for item in section.get("items") or []:
        item["distractors"] = list(distractors)
    section["word_bank"] = answers + distractors
    return True


def _fitb_word_bank_valid(section: Dict[str, Any]) -> bool:
    bank = [str(value).strip() for value in section.get("word_bank") or [] if str(value).strip()]
    answers = [
        str(answer).strip()
        for item in section.get("items") or []
        for answer in item.get("answers") or []
        if str(answer).strip()
    ]
    distractors = [
        str(value).strip() for item in section.get("items") or []
        for value in item.get("distractors") or [] if str(value).strip()
    ]
    distinct_distractors = list(dict.fromkeys(
        normalize_fitb_answer_key(value) for value in distractors
    ))
    normalized_bank = [normalize_fitb_answer_key(value) for value in bank]
    return (
        bool(answers)
        and all(len(item.get("answers") or []) == 1 for item in section.get("items") or [])
        and all(len(item.get("distractors") or []) == 2 for item in section.get("items") or [])
        and len(normalized_bank) == len(set(normalized_bank))
        and len(bank) == len(answers) + 2
        and len(distinct_distractors) == 2
        and all(value in normalized_bank for value in distinct_distractors)
        and all(value in bank for value in answers)
        and all(is_pure_text_fitb_value(value) for value in [*answers, *distractors])
        and all(
            is_pure_text_fitb_value(str(item.get("question") or ""), allow_blank=True)
            for item in section.get("items") or []
        )
    )


def _fitb_word_bank_structure_valid(section: Dict[str, Any]) -> bool:
    """Validate the source bank independently from generated blank alignment."""
    bank = [str(value).strip() for value in section.get("word_bank") or [] if str(value).strip()]
    normalized = [" ".join(value.casefold().split()) for value in bank]
    return len(bank) >= 3 and len(normalized) == len(set(normalized)) and all(
        _fitb_term_answer(value) for value in bank
    )


def _question_task_text(qtype: str, item: Dict[str, Any]) -> str:
    if qtype == "true_false":
        return str(item.get("statement") or "")
    if qtype == "definition":
        return str(item.get("term") or "")
    if qtype == "equation":
        return str(item.get("equation") or "")
    return str(item.get("question") or "")


def _meaningful_decimal_quantities(text: str) -> tuple[str, ...]:
    """Extract decimal/scientific givens without treating n1/theta2 as data."""
    values: set[str] = set()
    pattern = re.compile(
        r"(?<![\w.])[+\-]?(?:\d+\.\d+|\d+(?:\.\d+)?\s*[×x*]\s*10\s*\^?\s*[+\-]?\d+)",
        re.IGNORECASE,
    )
    for match in pattern.findall(str(text or "")):
        values.add(
            re.sub(r"\s+", "", match).casefold().replace("x", "×").replace("*", "×")
        )
    return tuple(sorted(values))


def _question_result_text(qtype: str, item: Dict[str, Any]) -> str:
    if qtype == "mcq":
        options = item.get("options") or {}
        return str(options.get(str(item.get("correct_answer") or "")) or "")
    if qtype in {"equation", "word_problem"}:
        return str(item.get("final_answer") or "")
    if qtype in {"definition", "why", "essay"}:
        return str(item.get("reference_answer") or "")
    return str(item.get("answer") or item.get("answers") or "")


def _global_duplicate_verdicts(
    questions: Dict[str, Any], plan_by_slot: Optional[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Flag the later task in cross-type plan or generated-question repetition."""
    from app.online.planner import _semantic_plan_conflict

    verdicts: Dict[str, Dict[str, Any]] = {}
    prior: list[dict[str, Any]] = []
    for qtype, _container, item in _iter_questions(questions):
        qid = str(item.get("question_id") or "")
        plan = (plan_by_slot or {}).get(str(item.get("slot_id") or "")) or {}
        basis = plan.get("relationship") or plan.get("concept")
        conditions = tuple(sorted(
            " ".join(normalize_text(value))
            for value in (plan.get("given_conditions") or [])
            if normalize_text(value)
        ))
        core = (
            " ".join(normalize_text(plan.get("knowledge_focus") or "")),
            " ".join(normalize_text(plan.get("student_work_product") or "")),
            " ".join(normalize_text(plan.get("canonical_skill") or "")),
            " ".join(normalize_text(plan.get("task_operation") or "")),
            " ".join(normalize_text(basis or "")),
            " ".join(normalize_text(plan.get("target") or "")),
            " ".join(normalize_text(plan.get("reasoning_mode") or "")),
            conditions,
        )
        family = " ".join(normalize_text(plan.get("scenario_family") or ""))
        scenario = (family, conditions) if family else None
        text = _question_task_text(qtype, item)
        result_text = _question_result_text(qtype, item)
        tokens = set(normalize_text(f"{text} {result_text}"))
        quantities = tuple(re.findall(
            r"[+\-]?\d+(?:\.\d+)?(?:\s*[?x*]\s*10\s*\^?\s*[+\-]?\d+)?", text
        ))
        decimal_givens = _meaningful_decimal_quantities(text)
        result_quantities = _meaningful_decimal_quantities(result_text)
        duplicate_of = None
        for previous in prior:
            prior_core = previous["core"]
            prior_tokens = previous["tokens"]
            prior_quantities = previous["quantities"]
            semantic_plan_repeat = _semantic_plan_conflict(previous["plan"], plan)
            exact_secondary_task = bool(
                core[3] and core[4] and core[5]
                and core[3:7] == prior_core[3:7]
            )
            repeated_scenario = scenario is not None and scenario == previous["scenario"]
            overlap = (
                len(tokens & prior_tokens) / len(tokens | prior_tokens)
                if tokens and prior_tokens else 0.0
            )
            same_skill_target = bool(
                core[2] and core[5]
                and core[2] == prior_core[2] and core[5] == prior_core[5]
            )
            same_specific_givens = bool(
                conditions and conditions == prior_core[7]
                and any(re.search(r"\d|[=<>??]", value) for value in conditions)
            )
            same_quantity_setup = bool(
                len(quantities) >= 3 and quantities == prior_quantities
            )
            same_meaningful_calculation = bool(
                len(decimal_givens) >= 3
                and decimal_givens == previous["decimal_givens"]
                and result_quantities
                and result_quantities == previous["result_quantities"]
            )
            semantic_text_repeat = same_skill_target and (
                overlap >= 0.82
                or (quantities and quantities == prior_quantities and overlap >= 0.55)
            )
            shared_tokens = len(tokens & prior_tokens)
            containment = (
                shared_tokens / min(len(tokens), len(prior_tokens))
                if tokens and prior_tokens else 0.0
            )
            answer_aware_restatement = shared_tokens >= 8 and containment >= 0.60
            hard_bases = [
                name for name, matched in (
                    ("semantic_plan_repeat", semantic_plan_repeat),
                    ("exact_secondary_task", exact_secondary_task),
                    ("repeated_scenario", repeated_scenario),
                    ("same_specific_givens", same_specific_givens),
                    ("same_quantity_setup", same_quantity_setup),
                    ("same_meaningful_calculation", same_meaningful_calculation),
                    ("same_skill_target_text", semantic_text_repeat),
                ) if matched
            ]
            soft_bases = ["answer_aware_restatement"] if answer_aware_restatement else []
            if hard_bases or soft_bases:
                duplicate_of = previous["question_id"]
                duplicate_strength = "hard" if hard_bases else "soft"
                duplicate_bases = [*hard_bases, *soft_bases]
                break
        if duplicate_of and qid:
            verdict = _verdict(
                False, False, "FIX_QUESTION_AND_ANSWER", ["question", "answer"],
                f"FAIL: duplicate_or_repetition. This repeats the underlying task or scenario "
                f"already used by {duplicate_of}, even though its question type or wording may differ.",
                "Replace the plan's knowledge focus or work product, operation, and target; then "
                "regenerate the slot without reusing the earlier setup or important quantities.",
                "duplicate_or_repetition", [],
            )
            verdict["question_id"] = qid
            verdict["conflicting_question_id"] = duplicate_of
            verdict["duplicate_strength"] = duplicate_strength
            verdict["duplicate_bases"] = duplicate_bases
            verdicts[qid] = verdict
        prior.append({
            "question_id": qid,
            "plan": plan,
            "core": core,
            "scenario": scenario,
            "tokens": tokens,
            "quantities": quantities,
            "decimal_givens": decimal_givens,
            "result_quantities": result_quantities,
        })
    return verdicts


# --------------------------------------------------------------------------
# Cheap deterministic checks (no LLM call needed)
# --------------------------------------------------------------------------
def _simple_option_signature(value: str) -> str:
    """Normalize exact text and simple commutative multiplication expressions."""
    compact = re.sub(r"\s+", "", str(value or "").strip().casefold())
    if compact.count("=") != 1:
        return compact
    left, right = compact.split("=", 1)
    factors = re.split(r"[×*·]", right)
    if len(factors) < 2 or any(not factor for factor in factors):
        return compact
    if any(re.search(r"[+\-/]", factor) for factor in factors):
        return compact
    return f"{left}={'×'.join(sorted(factors))}"


_WHY_PROMPT_RE = re.compile(
    r"(?:\bhow\s+does\b|\bexplain\s+(?:the\s+)?(?:cause|mechanism|relationship)\b|"
    r"\bwhy\b|\bgive\s+(?:a|the)\s+reason\b|\bwhat\s+(?:is|was)\s+the\s+"
    r"(?:cause|reason)\b|\bjustify\b|^\s*عل[ّ]?ل\b|\bلماذا\b|\bما\s+سبب\b|"
    r"\bما\s+هو\s+سبب\b|\bفس[ّ]?ر\s+سبب\b)",
    re.IGNORECASE,
)
_DEFINITION_PROMPT_RE = re.compile(
    r"(?:^\s*define\b|^\s*what\s+is\s+meant\s+by\b|^\s*what\s+does\b.+\bmean\b|"
    r"^\s*عر[ّ]?ف\b|^\s*ما\s+(?:المقصود\s+ب|معنى))",
    re.IGNORECASE,
)


def _is_why_prompt(value: object) -> bool:
    return bool(_WHY_PROMPT_RE.search(str(value or "").strip()))


def _is_definition_prompt(value: object) -> bool:
    return bool(_DEFINITION_PROMPT_RE.search(str(value or "").strip()))


def _local_verdict(qtype: str, item: Dict[str, Any], section: Any) -> Optional[Dict[str, Any]]:
    """Return a deterministic repair verdict for obvious structural breaks.

    These are definite, mechanical problems - never send them to the LLM.
    """
    if qtype == "mcq":
        options = item.get("options")
        if not isinstance(options, dict) or len(options) < 4:
            return _verdict(True, False, "FIX_OPTIONS", ["options"],
                            "MCQ has fewer than four options.",
                            "Provide exactly four options (A-D) with exactly one correct answer.")
        values = [str(v).strip() for v in options.values()]
        signatures = [_simple_option_signature(value) for value in values]
        dupes = {
            values[index]
            for index, signature in enumerate(signatures)
            if signatures.count(signature) > 1
        }
        if dupes:
            return _verdict(True, False, "FIX_OPTIONS", ["options"],
                            f"MCQ has duplicate options: {sorted(dupes)}.",
                            "Make all four options clearly distinct.")
        if str(item.get("correct_answer") or "").strip().upper() not in options:
            return _verdict(True, False, "FIX_OPTIONS", ["options", "correct_answer"],
                            "The stored answer points to a nonexistent option.",
                             "Update options so the real correct answer exists and set the answer to it.")
        correct_key = str(item.get("correct_answer") or "").strip().upper()
        correct_text = str(options.get(correct_key) or "").strip()
        stem = str(item.get("question") or "").strip()
        normalized_correct = " ".join(normalize_text(correct_text))
        normalized_stem = " ".join(normalize_text(stem))
        if len(correct_text) >= 12 and normalized_correct in normalized_stem:
            return _verdict(
                False, True, "FIX_QUESTION", ["question"],
                "The MCQ stem contains the complete correct answer and reveals it unnecessarily.",
                "Remove the revealed answer and ask the assigned operation concisely.",
            )
        distractor_lengths = [
            len(str(value).split()) for key, value in options.items() if key != correct_key
        ]
        correct_length = len(correct_text.split())
        if distractor_lengths:
            median = sorted(distractor_lengths)[len(distractor_lengths) // 2]
            if correct_length >= 8 and correct_length > max(median * 2, median + 6):
                return _verdict(
                    True, False, "FIX_OPTIONS", ["options", "correct_answer"],
                    "The correct option visibly stands out because it is much longer and more detailed.",
                    "Make all four options parallel in length, detail, grammar, specificity, and tone.",
                )
        content_stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
            "is", "it", "of", "on", "or", "that", "the", "this", "to", "what",
            "which", "with", "about", "best", "following", "describes",
        }
        token_set = lambda value: {
            token for token in normalize_text(str(value))
            if len(token) > 2 and token not in content_stopwords
        }
        stem_tokens = token_set(stem)
        correct_tokens = token_set(correct_text)
        distractor_tokens = [
            token_set(value) for key, value in options.items() if key != correct_key
        ]
        if (
            stem_tokens & correct_tokens
            and len(distractor_tokens) == 3
            and all(len(tokens) >= 2 and not (tokens & stem_tokens) for tokens in distractor_tokens)
            and not any(
                distractor_tokens[left] & distractor_tokens[right]
                for left in range(3) for right in range(left + 1, 3)
            )
        ):
            return _verdict(
                True, False, "FIX_OPTIONS", ["options", "correct_answer"],
                "The distractors are lexically unrelated to the assessed concept and do not compete with the answer.",
                "Use plausible same-topic misconceptions, reversed relationships, or incomplete reasoning.",
            )
    elif qtype == "true_false":
        statement = str(item.get("statement") or "").strip()
        if not statement:
            return _verdict(False, False, "FIX_QUESTION", ["statement"],
                            "True/False statement is empty.",
                            "Write a complete, single, unambiguous statement.")
        if str(item.get("answer") or "").strip().lower() not in {"true", "false"}:
            return _verdict(True, False, "FIX_ANSWER", ["answer"],
                            "True/False answer is not 'True' or 'False'.",
                            "Set the answer to exactly 'True' or 'False'.")
    elif qtype == "fill_in_the_blank":
        question = str(item.get("question") or "").strip()
        if not is_pure_text_fitb_value(question, allow_blank=True):
            return _verdict(
                False, True, "FIX_QUESTION", ["question"],
                "Fill-in-the-Blank must be pure text with no numbers, units, formulas, "
                "equations, chemical symbols, or calculation.",
                "Write one source-grounded textual clue with exactly one blank.",
                "ambiguous_or_structural",
            )
        blank_count = len(re.findall(r"_{2,}", question))
        if blank_count != 1:
            return _verdict(False, False, "FIX_QUESTION", ["question"],
                            "Fill-in-the-Blank sentence must have exactly one blank marker.",
                            "Use exactly one run of underscores (________) marking the blank.")
        answers = item.get("answers") or []
        if len(answers) != 1 or not all(str(a).strip() for a in answers):
            return _verdict(True, False, "FIX_ANSWER", ["answers"],
                            "Fill-in-the-Blank must have exactly one intended answer.",
                            "Provide exactly one source-grounded answer term.")
        invalid_terms = [str(answer) for answer in answers if not _fitb_term_answer(answer)]
        if invalid_terms:
            return _verdict(
                True, False, "FIX_ANSWER", ["answers"],
                "Fill-in-the-Blank answers must be terms or short phrases, not formulas "
                f"or calculated results: {invalid_terms}.",
                "Choose a grounded vocabulary term or short phrase already in the Word Bank.",
                "ambiguous_or_structural",
            )
        distractors = item.get("distractors") or []
        if (
            len(distractors) != 2
            or any(not _fitb_term_answer(value) for value in distractors)
            or len({
                " ".join(str(value).casefold().split())
                for value in [*answers, *distractors]
            }) != 3
        ):
            return _verdict(
                True, False, "FIX_OPTIONS", ["distractors"],
                "Fill-in-the-Blank must have exactly two distinct pure-text distractors.",
                "Provide two plausible same-topic terms that are definitely wrong for this blank.",
                "ambiguous_or_structural",
            )
        if isinstance(section, dict):
            bank = [str(w).strip() for w in (section.get("word_bank") or [])]
            missing = [
                str(value).strip() for value in [*answers, *distractors]
                if str(value).strip() not in bank
            ]
            if missing:
                return _verdict(
                    True, False, "FIX_QUESTION_AND_ANSWER",
                    ["answers", "distractors", "word_bank"],
                    f"FITB answer/distractor terms are absent from the Word Bank: {missing}.",
                    "Align the answer and two distractors with the shared Word Bank.",
                )
    elif qtype == "definition":
        term = str(item.get("term") or "").strip()
        if not term:
            return _verdict(False, False, "FIX_QUESTION", ["term"],
                            "Definition term is empty.",
                            "Provide the specific source term or concept to define.")
        reference = str(item.get("reference_answer") or "").strip()
        if not reference:
            return _verdict(True, False, "FIX_ANSWER", ["reference_answer"],
                            "Expected definition is empty.",
                            "Write a concise definition grounded in the selected source.")
        if len(reference) > 500 or len(reference.split()) > 80:
            return _verdict(
                True, False, "FIX_ANSWER", ["reference_answer"],
                "Definition answer is not concise.",
                "Give a short, direct, source-supported definition of the requested target.",
                "question_type_operation_mismatch",
            )
    elif qtype in {"equation", "word_problem"}:
        text_field = "equation" if qtype == "equation" else "question"
        task_text = str(item.get(text_field) or "").strip()
        if not task_text:
            return _verdict(False, False, "FIX_QUESTION", [text_field],
                            "Question text is empty.",
                            "Provide a complete question matching its planned operation.")
        steps = item.get("solution_steps")
        if not isinstance(steps, list) or not steps or not all(str(step).strip() for step in steps):
            return _verdict(True, False, "FIX_ANSWER", ["solution_steps", "final_answer"],
                            "Solution steps are missing or malformed.",
                            "Provide ordered solution steps and a consistent final answer.")
        if not str(item.get("final_answer") or "").strip():
            return _verdict(True, False, "FIX_ANSWER", ["solution_steps", "final_answer"],
                            "Final answer is empty.",
                            "Provide a final answer consistent with the solution steps.")
    elif qtype in ("why", "essay"):
        if not str(item.get("question") or "").strip():
            return _verdict(False, False, "FIX_QUESTION", ["question"],
                            "Question text is empty.",
                            "Write a complete, answerable question.")
        if not str(item.get("reference_answer") or "").strip():
            return _verdict(True, False, "FIX_ANSWER", ["reference_answer"],
                            "Reference answer is empty.",
                            "Write a complete reference answer that directly answers the question.")
        question = str(item.get("question") or "").strip()
        if qtype == "why" and not _is_why_prompt(question):
            return _verdict(
                False, True, "FIX_QUESTION", ["question"],
                "Why question does not ask for a cause, reason, or justification.",
                "Preserve the causal task and ask with Why/علّل/لماذا/ما سبب/فسّر سبب.",
                "question_type_operation_mismatch",
            )
    return None


def _verdict(
        qv: bool, av: bool, action: str, fields: List[str], reason: str, expected_fix: str,
        failure_code: str = "ambiguous_or_structural",
        expected_results: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "question_id": "",
        "question_valid": qv,
        "answer_valid": av,
        "action": action,
        "fields_to_fix": fields,
        "failure_code": failure_code,
        "expected_results": expected_results or [],
        "reason": reason,
        "expected_fix": expected_fix,
    }


# --------------------------------------------------------------------------
# Validator LLM prompts
# --------------------------------------------------------------------------
GLOBAL_VALIDATOR_SYSTEM_PROMPT = (
    "You are the final semantic and editorial reviewer for one complete exam. Review the "
    "entire registry as a single assessment, compare every question with every other question, "
    "and return exactly one compact verdict for every question_id. This call does not receive "
    "textbook evidence; bounded validators separately check grounding, calculations, units, "
    "missing givens, visuals, and detailed answer completeness.\n\n"
    "Global review rules:\n"
    "- Judge actual stems, options, correct/reference answers, operations, and knowledge required. "
    "Never treat different metadata strings, wording, question types, stories, or numbers as proof "
    "that tasks are different.\n"
    "- Because source evidence is intentionally absent, do not use outside subject conventions to "
    "overrule a factual result or formula that could depend on the textbook's defined setup. You may "
    "flag an internal contradiction, undefined notation, or disagreement between a stem, answer, and "
    "its own semantic identity; leave source-grounding and independent calculation judgments to the "
    "bounded validators.\n"
    "- Flag conceptual duplicates across every question type: the same fact recalled twice, the "
    "same principle explained after it was selected, or the same equation merely stated in two "
    "formats. Flag repeated scenarios, important givens, targets, formulas, numerical setups, and "
    "answers when they make the student perform substantially the same work.\n"
    "- Reuse of a law or relationship is acceptable only when the student work product, operation, "
    "target, and reasoning are genuinely different, such as deriving a symbolic relationship in "
    "one item and applying it to determine a different unknown in another.\n"
    "- For every duplicate/repetition group, PASS the earliest sound item and fail the later item(s) "
    "with duplicate_or_repetition and all conflicting question IDs. Require a different underlying "
    "task; rewording, changing type, or changing only values is not a fix.\n"
    "- Review overall topic and skill diversity, True/False answer-pattern imbalance, awkward or "
    "circular language (especially Arabic definitions), unclear wording, inconsistent terminology "
    "or symbol notation, Equation versus Word Problem role quality, weak MCQ distractors, answer "
    "patterns, and exam-wide consistency.\n"
    "- A Definition answer must define the term rather than repeat it circularly. Arabic must be "
    "natural, grammatically coherent, and use stable educational terminology and symbols.\n"
    "- Equation and Word Problem items may use the same law only for meaningfully different work. "
    "Do not impose a universal subject rule; judge the planned operation and actual requested work.\n"
    "- Use FIX_OPTIONS for an otherwise sound MCQ with weak distractors. Use FIX_QUESTION for a "
    "wording-only defect. Use FIX_QUESTION_AND_ANSWER when the task/plan must change or both sides "
    "must stay synchronized.\n"
    "- PASS verdicts must have failure_code=pass and empty conflicts, fields, reason, and expected_fix. "
    "Explain only failures. Conflict groups must identify every cross-question repetition found. "
    "Every error-level exam finding and every conflict group must also have at least one affected "
    "question marked for repair; do not hide actionable defects only in the summary.\n"
    "- Return only the structured JSON object requested by the schema."
)


def _global_review_entry(
    qtype: str, item: Dict[str, Any], plan_by_slot: Optional[Dict[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Build the evidence-free registry row used by the exam-level reviewer."""
    plan = (plan_by_slot or {}).get(str(item.get("slot_id") or "")) or {}
    question = deepcopy(item)
    correct_text = ""
    if qtype == "mcq":
        options = question.get("options") or {}
        correct_text = str(options.get(str(question.get("correct_answer") or "")) or "")
    return {
        "question_id": str(item.get("question_id") or ""),
        "slot_id": str(item.get("slot_id") or ""),
        "question_type": qtype,
        "difficulty": plan.get("difficulty"),
        "question": question,
        "correct_option_text": correct_text,
        "reference_or_final_answer": _question_result_text(qtype, item),
        "semantic_identity": {
            "knowledge_focus": plan.get("knowledge_focus"),
            "student_work_product": plan.get("student_work_product"),
            "canonical_skill": plan.get("canonical_skill"),
            "task_operation": plan.get("task_operation"),
            "relationship_or_concept": plan.get("relationship") or plan.get("concept"),
            "target": plan.get("target"),
            "reasoning_mode": plan.get("reasoning_mode"),
            "required_givens": list(plan.get("required_givens") or []),
            "given_conditions": list(plan.get("given_conditions") or []),
            "scenario_family": plan.get("scenario_family"),
        },
    }


def _global_review_candidate_pairs(
    exam: Dict[str, Any], plan_by_slot: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Expose structurally related pairs for explicit AI adjudication."""
    rows = []
    for qtype, _container, item in _iter_questions(exam.get("questions") or {}):
        plan = (plan_by_slot or {}).get(str(item.get("slot_id") or "")) or {}
        norm = lambda value: " ".join(normalize_text(value or ""))
        rows.append({
            "question_id": str(item.get("question_id") or ""),
            "question_type": qtype,
            "knowledge_focus": norm(plan.get("knowledge_focus")),
            "canonical_skill": norm(plan.get("canonical_skill")),
            "task_operation": norm(plan.get("task_operation")),
            "relationship": norm(plan.get("relationship") or plan.get("concept")),
            "target": norm(plan.get("target")),
            "reasoning_mode": norm(plan.get("reasoning_mode")),
            "scenario_family": norm(plan.get("scenario_family")),
        })
    candidates: List[Dict[str, Any]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            signals = []
            for field in ("knowledge_focus", "canonical_skill", "relationship", "scenario_family"):
                if left[field] and left[field] == right[field]:
                    signals.append(f"same_{field}")
            if (
                left["task_operation"] and left["target"]
                and left["task_operation"] == right["task_operation"]
                and left["target"] == right["target"]
            ):
                signals.append("same_operation_and_target")
            if not signals:
                continue
            candidates.append({
                "question_ids": [left["question_id"], right["question_id"]],
                "question_types": [left["question_type"], right["question_type"]],
                "signals": signals,
                "operations": [left["task_operation"], right["task_operation"]],
                "targets": [left["target"], right["target"]],
                "reasoning_modes": [left["reasoning_mode"], right["reasoning_mode"]],
            })
    return candidates


def build_global_review_prompt(
    exam: Dict[str, Any],
    plan_by_slot: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    phase: str = "initial",
) -> str:
    registry = [
        _global_review_entry(qtype, item, plan_by_slot)
        for qtype, _container, item in _iter_questions(exam.get("questions") or {})
        if item.get("question_id")
    ]
    candidate_pairs = _global_review_candidate_pairs(exam, plan_by_slot)
    return (
        f"Review phase: {phase}. This is the complete current exam registry; no question is omitted, "
        "including questions already flagged by deterministic checks.\n\n"
        "## Complete exam registry\n"
        + json.dumps(registry, ensure_ascii=False, indent=2)
        + "\n\n## Candidate pairs requiring explicit semantic comparison\n"
        + json.dumps(candidate_pairs, ensure_ascii=False, indent=2)
        + "\nThese signals are not verdicts. Compare the actual stems and answers for every listed pair. "
          "If the student demonstrates substantially the same knowledge or performs substantially "
          "the same task, report a conflict even when the wording, type, scenario, or values differ."
        + "\n\nReturn one verdict per question_id, conflict_groups, and exam_quality_findings."
    )


def _operational_global_verdict(question_id: str, reason: str) -> Dict[str, Any]:
    return {
        "question_id": question_id,
        "action": "UNVALIDATED",
        "failure_code": "global_review_missing",
        "conflicting_question_ids": [],
        "fields_to_fix": [],
        "reason": reason,
        "expected_fix": "Retry the complete global exam review; do not publish this item meanwhile.",
    }


def review_exam_globally(
    exam: Dict[str, Any],
    plan_by_slot: Optional[Dict[str, Dict[str, Any]]] = None,
    client: Optional[Any] = None,
    *,
    phase: str = "initial",
) -> Dict[str, Any]:
    """Review only embedding-shortlisted duplicate candidates in one AI call."""
    from app.online.duplicate_judge import review_duplicate_candidates

    focused = review_duplicate_candidates(exam, client=client)
    focused.update({"phase": phase, "complete": True, "conflict_groups": []})
    return focused

def _deterministic_exam_verdicts(
    exam: Dict[str, Any],
    plan_by_slot: Optional[Dict[str, Dict[str, Any]]] = None,
    question_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collect cheap structural findings without excluding anything from AI review."""
    selected = (
        {str(value) for value in question_ids if str(value)}
        if question_ids is not None else None
    )
    questions = exam.get("questions") or {}
    verdicts: Dict[str, Dict[str, Any]] = {}
    for qtype, container, item in _iter_questions(questions):
        qid = str(item.get("question_id") or "")
        if not qid or (selected is not None and qid not in selected) or qid in verdicts:
            continue
        plan_item = (plan_by_slot or {}).get(str(item.get("slot_id") or ""))
        verdict = _local_verdict(qtype, item, container)
        if verdict is None and qtype in {"equation", "word_problem"}:
            verdict = _solve_system_completion_verdict(item, plan_item)
        if verdict is not None:
            verdict["question_id"] = qid
            verdicts[qid] = verdict
    return verdicts


def _standardize_global_verdict(verdict: Dict[str, Any]) -> Dict[str, Any]:
    action = str(verdict.get("action") or "UNVALIDATED")
    if action == "UNVALIDATED":
        return dict(verdict)
    result = {
        "question_id": str(verdict.get("question_id") or ""),
        "question_valid": action not in {"FIX_QUESTION", "FIX_QUESTION_AND_ANSWER"},
        "answer_valid": action not in {
            "FIX_ANSWER", "FIX_OPTIONS", "FIX_QUESTION_AND_ANSWER"
        },
        "action": action,
        "fields_to_fix": list(verdict.get("fields_to_fix") or []),
        "failure_code": str(verdict.get("failure_code") or "other"),
        "expected_results": [],
        "reason": str(verdict.get("reason") or ""),
        "expected_fix": str(verdict.get("expected_fix") or ""),
        "conflicting_question_ids": list(verdict.get("conflicting_question_ids") or []),
        "validation_sources": ["global_ai"],
    }
    conflicts = result["conflicting_question_ids"]
    if conflicts:
        result["conflicting_question_id"] = conflicts[0]
    return result


def _merge_global_review(
    bounded_report: Dict[str, Any], global_review: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge independent layers without allowing one PASS to erase another failure."""
    global_by_id = {
        str(item.get("question_id") or ""): item
        for item in global_review.get("verdicts") or []
        if item.get("question_id")
    }
    merged_verdicts: List[Dict[str, Any]] = []
    for bounded in bounded_report.get("verdicts") or []:
        qid = str(bounded.get("question_id") or "")
        global_raw = global_by_id.get(qid)
        if global_raw is None:
            chosen = _operational_global_verdict(
                qid, "No complete-exam verdict was available for this question."
            )
        elif global_raw.get("action") == "PASS":
            soft_similarity_only = (
                bounded.get("action") != "PASS"
                and str(bounded.get("failure_code") or "") == "duplicate_or_repetition"
                and str(bounded.get("duplicate_strength") or "") == "soft"
            )
            if soft_similarity_only:
                chosen = _standardize_global_verdict(global_raw)
                chosen["validation_sources"] = ["deterministic_advisory", "global_ai"]
                chosen["deterministic_advisory"] = {
                    "failure_code": bounded.get("failure_code"),
                    "duplicate_bases": list(bounded.get("duplicate_bases") or []),
                    "conflicting_question_id": bounded.get("conflicting_question_id"),
                    "reason": bounded.get("reason"),
                }
            else:
                chosen = dict(bounded)
                chosen["validation_sources"] = list(dict.fromkeys([
                    *(bounded.get("validation_sources") or ["bounded"]), "global_ai"
                ]))
        elif bounded.get("action") == "PASS" or str(global_raw.get("failure_code")) == "duplicate_or_repetition":
            chosen = _standardize_global_verdict(global_raw)
            if bounded.get("action") != "PASS":
                chosen["additional_bounded_issue"] = {
                    key: bounded.get(key) for key in (
                        "action", "failure_code", "reason", "expected_fix"
                    )
                }
                chosen["validation_sources"].insert(0, "bounded")
        else:
            chosen = dict(bounded)
            chosen["additional_global_issues"] = [{
                key: global_raw.get(key) for key in (
                    "action", "failure_code", "conflicting_question_ids", "reason", "expected_fix"
                )
            }]
            chosen["validation_sources"] = list(dict.fromkeys([
                *(bounded.get("validation_sources") or ["bounded"]), "global_ai"
            ]))
        merged_verdicts.append(chosen)
    merged = dict(bounded_report)
    merged["bounded_verdicts"] = deepcopy(bounded_report.get("verdicts") or [])
    merged["verdicts"] = merged_verdicts
    merged["all_pass"] = all(item.get("action") == "PASS" for item in merged_verdicts)
    merged["global_review"] = deepcopy(global_review)
    merged["warnings"] = [
        *(bounded_report.get("warnings") or []), *(global_review.get("warnings") or [])
    ]
    return merged


VALIDATOR_SYSTEM_PROMPT = """You are the general, non-quantitative exam reviewer. Review each question independently. Do not perform duplicate detection or exam-wide coverage review.

Rules:
- Return one root JSON object with a `verdicts` array containing exactly one verdict per question_id.
- Use assigned_sources for facts, laws, formulas, definitions, terminology, and source-specific values. Neutral numbers explicitly stated as givens in a self-contained exercise are problem data and need not appear verbatim in the source.
- Check clarity, grammar, grounding, planned question type and operation, difficulty fit, assessment guidance, and non-quantitative answer quality. If difficulty is wrong, repair the student-facing question to match the plan; never request changes to plan.difficulty or any Planner field.
- Compare expected_catalog_task with the actual generated question. The question must preserve the assigned concept/knowledge target, student operation, target/result, reasoning mode, required output, required givens, and evidence boundary. Being on the same broad topic is not enough. If the intended task is to interpret a solution region but the question merely asks whether a point satisfies inequalities, return failure_code=task_drift and describe expected versus actual concept, operation, and target in reason.
- Check that required representations are included. Leave missing numerical/symbolic givens and calculations to the STEM quantitative validator.
- For MCQ, verify that the concise stem performs the assigned operation and does not quote a long passage or reveal the answer. Reject difficulty created only by long/confusing wording or irrelevant detail. Verify four parallel options, exactly one defensible best answer, and plausible same-concept distractors based on nearby concepts, misconceptions, reversed relationships, incomplete reasoning, or realistic mistakes. Reject irrelevant/ridiculous distractors, semantic-equivalent options, multiple defensible answers, and a correct option that stands out by length, detail, qualification, grammar, tone, or copied source wording. Reject unsupported facts, numbers, scenarios, and assumptions. Leave numerical/symbolic correctness to the STEM quantitative validator.
- Distinguish a good catalog task with a bad generated MCQ from a fundamentally unsuitable task. For a repairable generated item, request only the exact student-facing fields that need repair and keep the same task. If the assigned task cannot naturally support one clear best answer, three plausible competitors, and meaningful assessment of its target, use failure_code=mcq_task_unsuitable so allocation can select another task; do not repeatedly repair or weaken that task.
- Why must ask for causal reasoning, mechanisms, justification, or relationships and preserve a compatible reasoning task. Reject listing, definition, evidence-significance, or simple retrieval prompts labeled Why.
- Definition stores the bare source term or concept in term; the UI/export supplies the instruction to define it. Reject prompt text embedded in term, another operation, or an answer that does not directly and concisely define that target.
- Check True/False clarity; one-answer/one-blank FITB; relevant Why/Essay depth; and Equation versus meaningful Word Problem form.
- For Equation and Word Problem, inspect task_operation, target, student_work_product, relationship, required_givens, and reasoning_mode together. Necessary premises and givens must be present, but the prompt must not perform the operation assigned to the student. A starting relationship is appropriate for rearrange or apply-a-provided-rule tasks. For derive, construct, formulate, combine, prove, balance, or select-the-governing-relationship tasks, reject a stem that supplies the final relationship, completed equation, exact reasoning step, or result being assessed; use failure_code=task_drift and request a question-field repair.
- Do not impose one universal Equation or Word Problem pattern. Verify that the student work follows the catalog operation rather than defaulting to rearrange-and-isolate or story-then-choose-a-formula. For a Word Problem, providing a rule is valid only when applying a provided rule is the assigned work; omit it when selecting or forming that relationship is assessed.
- Compare quantitative/application questions with assigned_sources. Unless expected_catalog_task explicitly targets the exact source instance through its target and required_givens, reject copying a worked example's same scenario + values + requested result. Reusing its supported method or constraints with a materially different, fully stated, evidence-consistent instance is allowed. Do not require invented facts or contexts.
- Choose the smallest repair: PASS, FIX_ANSWER, FIX_OPTIONS, FIX_QUESTION, or FIX_QUESTION_AND_ANSWER.
- fields_to_fix may name only exact canonical fields: question, statement, term, equation, options, correct_answer, reference_answer, solution_steps, final_answer, key_points, distractors, or solution_certificate. For Fill-in-the-Blank, request only question and/or distractors; its catalog answer and Word Bank are code-owned. Never use a generic answer alias and never name Planner metadata.
- Return only valid JSON using the required verdict schema."""

STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT = """You are a quantitative STEM correctness validator using medium reasoning. Review only numerical and symbolic correctness. Do not judge difficulty, wording, grammar, style, formatting, duplication, coverage, Planner metadata, or whether neutral constructed givens appeared in the source.
For each question, first identify exactly what the student-facing request asks the student to produce. Validate everything quantitatively necessary for that requested output, and nothing beyond it. Do not require an additional quantity the question never asks for, do not automatically require every variable appearing in a system to be solved, do not make optional extra calculations mandatory, and do not redesign or improve the expected answer during validation. A break-even problem asking for the break-even customer count and profit at 60 customers does not also require an unrequested y value.
Then solve the student-facing problem independently without trusting its stored answer or solution_certificate. Record that independent calculation in independent_solution using checkable steps and a final answer. Inspect the stored solution_certificate and compare its givens, mathematical steps, option mapping, units, and final answer with your independent work. If they disagree, identify the first mathematically incorrect step in first_incorrect_step and set certificate_valid=false. Use a second verification method when practical: substitution, reverse calculation, dimensional analysis, constraint testing, recomputation, or checking every feasible candidate. State that check in independent_solution.verification_method.
Check equations, calculations, units, rounding, constraints, missing numerical or symbolic givens required by the actual request, all and only the requested outputs, solution completeness for those outputs, and whether the final answer solves exactly what was asked. For numerical or symbolic MCQ, solve first, verify exactly one option is correct, verify the stored answer points to it, and record matching_correct_option. For optimization, test only feasible candidates. For derivations, verify every step reaches the requested relationship. A missing or malformed certificate is an incomplete quantitative answer and must be repaired. An explicit solution_certificate=null means a previous question repair intentionally invalidated the old certificate; when the student-facing question and answer are otherwise correct, return FIX_ANSWER with fields_to_fix=["solution_certificate"], failure_code="incomplete_solution", and reason="previous solution certificate was invalidated after question repair and must be rebuilt for the current question". This certificate-only case has no mathematically incorrect step, so first_incorrect_step must be null.
Put the independently computed result in expected_results. Request only student-facing quantitative fields plus the internal solution_certificate. Never request a Planner-field edit. Do not report non-quantitative defects. Return one root JSON object with a `verdicts` array containing exactly one verdict per question_id and no other root fields."""

VALIDATOR_OUTPUT_SCHEMA = (
    '{\n'
    '  "verdicts": [\n'
    '  {\n'
    '    "question_id": "...",\n'
    '    "question_valid": true,\n'
    '    "answer_valid": false,\n'
    '    "action": "FIX_ANSWER",\n'
    '    "fields_to_fix": ["reference_answer"],\n'
    '    "failure_code": "incorrect_result",\n'
    '    "expected_results": ["the independently checked result"],\n'
    '    "reason": "...",\n'
    '    "expected_fix": "...",\n'
    '    "independent_solution": {"method": "...", "important_givens": ["..."], '
    '"calculation_steps": ["..."], "final_answer": "...", "units": "", '
    '"matching_correct_option": null, "verification_method": "..."},\n'
    '    "certificate_valid": true,\n'
    '    "first_incorrect_step": null\n'
    '  }\n'
    '  ]\n'
    '}'
)

QUANTITATIVE_VALIDATOR_OUTPUT_SCHEMA = (
    '{\n'
    '  "verdicts": [\n'
    '  {\n'
    '    "question_id": "...",\n'
    '    "question_valid": true,\n'
    '    "answer_valid": true,\n'
    '    "action": "PASS",\n'
    '    "fields_to_fix": [],\n'
    '    "failure_code": "pass",\n'
    '    "expected_results": ["the independently checked result"],\n'
    '    "reason": "...",\n'
    '    "expected_fix": "...",\n'
    '    "independent_solution": {"method": "...", "important_givens": ["..."], '
    '"calculation_steps": ["..."], "final_answer": "...", "units": "", '
    '"matching_correct_option": null, "verification_method": "..."},\n'
    '    "certificate_valid": true,\n'
    '    "first_incorrect_step": null\n'
    '  }\n'
    '  ]\n'
    '}'
)

VALIDATOR_RETRY_SYSTEM_PROMPT = VALIDATOR_SYSTEM_PROMPT + "\nA previous response omitted verdicts. Review every supplied question and never skip an ID."
STEM_QUANTITATIVE_VALIDATOR_RETRY_SYSTEM_PROMPT = STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT + "\nA previous response omitted verdicts. Review every supplied question and never skip an ID."


def build_validator_prompt(
    entries: List[Dict[str, Any]], *, quantitative: bool = False,
    rejection_reason: str = "",
) -> str:
    prompt_entries: List[Dict[str, Any]] = []
    for entry in entries:
        cleaned = dict(entry)
        source_chunks = cleaned.pop("_source_chunks", {}) or {}
        cleaned["assigned_sources"] = {
            str(chunk_id): str(content)
            for chunk_id, content in source_chunks.items()
            if chunk_id and content
        } if isinstance(source_chunks, dict) else {}
        prompt_entries.append(cleaned)
    payload = json.dumps(prompt_entries, ensure_ascii=False, indent=None)
    retry_context = (
        "\n\n## Corrective retry\n"
        "The previous structured response was rejected by code for this exact reason:\n"
        f"{rejection_reason}\n"
        "Return a fresh verdict for only the supplied question ID and preserve the exact schema."
        if rejection_reason else ""
    )
    return (
        "Review every exam question below and return one structured verdict per "
        "question (same number of results as entries).\n"
        "Use the exact question_ids you were given and never invent new ones. Each entry is an "
        "isolated evidence boundary. Use only that entry's assigned_sources; never use another "
        "entry's evidence to validate or rescue it.\n\n"
        "## Questions to review\n"
        f"{payload}\n\n"
        "## Verdict object format (use exactly these field names)\n"
        f"{QUANTITATIVE_VALIDATOR_OUTPUT_SCHEMA if quantitative else VALIDATOR_OUTPUT_SCHEMA}\n\n"
        "Return ONLY one JSON object with a `verdicts` array. No text before or after, "
        "no fences, and never return a bare root array."
        f"{retry_context}"
    )


def _plan_source_ids(plan_item: Optional[Dict[str, Any]]) -> List[str]:
    if not plan_item:
        return []
    values = [
        plan_item.get("source_chunk_id"),
        *(plan_item.get("support_chunk_ids") or []),
    ]
    return list(dict.fromkeys(str(value) for value in values if value))


def _assigned_source_chunks(
    plan_item: Optional[Dict[str, Any]], source_by_chunk: Optional[Dict[str, str]]
) -> Dict[str, str]:
    sources = source_by_chunk or {}
    return {
        chunk_id: sources[chunk_id]
        for chunk_id in _plan_source_ids(plan_item)
        if sources.get(chunk_id)
    }


def _catalog_task_contract(plan_item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Expose the intended assessment task without relying on broad topic fit."""
    plan = plan_item or {}
    contract = {
        "slot_id": plan.get("slot_id"),
        "task_id": plan.get("task_id"),
        "task_family_id": plan.get("task_family_id"),
        "coverage_id": plan.get("coverage_id"),
        "concept_id": plan.get("concept_id"),
        "relationship_id": plan.get("relationship_id"),
        "concept": plan.get("concept") or plan.get("relationship") or plan.get("knowledge_focus"),
        "knowledge_target": plan.get("knowledge_focus"),
        "student_operation": plan.get("task_operation") or plan.get("canonical_skill"),
        "target_or_result": plan.get("target"),
        "relationship": plan.get("relationship"),
        "reasoning_mode": plan.get("reasoning_mode"),
        "required_output": plan.get("student_work_product"),
        "required_givens": list(plan.get("required_givens") or []),
        "scenario_family": plan.get("scenario_family"),
        "difficulty": plan.get("difficulty"),
        "assigned_evidence_ids": _plan_source_ids(plan),
    }
    if plan.get("fitb_entry_id"):
        contract.update({
            "fitb_entry_id": plan.get("fitb_entry_id"),
            "fitb_answer": plan.get("fitb_answer"),
            "clue_basis": plan.get("clue_basis"),
        })
    return contract


REPAIR_SYSTEM_PROMPT = (
    "You are the general, non-quantitative exam-question repairer. For each item, fix ONLY "
    "the wording, clarity, grounding, type-rule, difficulty-fit, or conceptual-quality defect "
    "listed in fields_to_fix. Never repair calculations, equations, units, numerical givens, or "
    "numerical/symbolic answers; those belong to the STEM quantitative repairer. Rules:\n"
    "- Return the EXACT same question_id for each repaired question; never change it.\n"
    "- Do NOT change fields that are not listed in fields_to_fix.\n"
    "- Do NOT rewrite a question that is fine; preserve every valid field verbatim.\n"
    "- Only repair the specific problem described in reason/expected_fix.\n"
    "- Each item is an isolated evidence boundary. When plan and source_chunks are supplied, use "
    "only that item's source_chunks and preserve its assessment guidance, task_operation, "
    "relationship, target, reasoning_mode, and difficulty. Never borrow evidence from another item.\n"
    "- For Fill-in-the-Blank, return only stem, distractor_1, and distractor_2. The selected "
    "fitb_entry_id and answer are authoritative and immutable; code rebuilds the Word Bank.\n"
    "- A Why repair must remain causal and use Why/علّل/لماذا/ما سبب/فسّر سبب wording; never "
    "replace it with listing, definition, evidence-significance, or simple retrieval.\n"
    "- A Definition repair stores only the bare source term or concept in term and must return a "
    "concise direct definition. Never put instruction wording or another operation in term.\n"
    "- Conceptual Multiple Choice (when options are in fields_to_fix): evaluate the stem "
    "and source evidence, then evaluate the option set. There must be EXACTLY one correct "
    "option. If two or more options are correct, keep one and rewrite only the "
    "conflicting distractor(s) so exactly one correct answer remains. If the real "
    "correct answer is missing from the options, replace one incorrect option with "
    "the real correct answer. After changing the options, update the stored answer "
    "to point to the one correct option. Preserve every already-valid option; change "
    "the minimum number of options needed; never regenerate the whole MCQ randomly; "
    "keep the question text unchanged unless the validator explicitly marked the "
    "question itself as broken.\n"
    "- For FIX_OPTIONS, preserve the valid stem and correct-answer meaning unless explicitly "
    "marked broken. Treat every distractor listed in rejected_distractors as rejected content: "
    "discard it and write a NEW replacement; never keep it unchanged or merely move it to a "
    "different A-D label. Generate replacements from the assigned concept, evidence, operation, "
    "and task. Before returning, self-check that every distractor is same-topic, plausible to a "
    "partially informed student, definitely incorrect, and meaningfully competes with the correct "
    "answer, while exactly one answer remains defensible.\n"
    "- MCQ distractors must compete on the assigned concept and remain parallel in length, grammar, "
    "detail, specificity, and tone. Replace irrelevant or implausible distractors with grounded "
    "misconceptions or incomplete/reversed reasoning; never add unsupported facts or scenarios.\n"
    "- For an Equation or Word Problem wording/type repair, preserve all quantitative givens, "
    "solution steps, and final answer verbatim unless a listed non-quantitative field must change.\n"
    "- When repairing an Equation or Word Problem question field, preserve the exact assigned "
    "operation. Give a starting relation for rearrange/apply-provided-rule work, but do not reveal "
    "the final relationship, completed equation, governing rule, reasoning step, or result when "
    "deriving, constructing, formulating, balancing, proving, or selecting it is the assessed work.\n"
    "- Return one root JSON object with a `repairs` array containing one repair object per "
    "input item, in any order. Each repair object is:\n"
    '  {"question_id": "...", "repaired_fields": ["..."], '
    '"question": {the full repaired question object with the same fields}]}\n'
    "- Return ONLY valid JSON. No fences, no extra text."
)

STEM_QUANTITATIVE_REPAIR_SYSTEM_PROMPT = """You are a quantitative STEM question repairer using medium reasoning. Repair only listed student-facing quantitative fields: numbers, equations, calculations, units, missing numerical or symbolic givens, solution_steps, final_answer, and numerical or symbolic MCQ options or correct_answer. Also replace the internal solution_certificate so it exactly documents the repaired mathematics. Preserve the catalog task and every field not authorized by fields_to_fix. Add only premises and givens needed by the assigned operation; never add the final relationship, completed equation, governing rule, reasoning step, or result when producing or selecting it is the student work being assessed.
Independently solve the repaired problem from scratch. Treat expected_results and expected_fix as diagnostic hypotheses, not authority. Verify by a second method when practical, such as substitution, reverse calculation, constraint testing, conservation, dimensional analysis, or independent recomputation. For optimization, use only feasible candidates. For MCQ, verify exactly one option is correct and put its label in solution_certificate.matching_correct_option. Keep solution_steps, final_answer, options, stored answer, and solution_certificate consistent. Never edit Planner metadata or address wording, difficulty, duplication, or coverage.
Return one root JSON object with a `repairs` array containing one full repair object per input question with unchanged question_id. Return only valid JSON."""

REPAIR_OUTPUT_SCHEMA = (
    '{\n'
    '  "repairs": [\n'
    '  {\n'
    '    "question_id": "...",\n'
    '    "repaired_fields": ["..."],\n'
    '    "question": {"question": "...", "options": {"A": "...", "B": "...", '
    '"C": "...", "D": "..."}, "correct_answer": "..."}\n'
    '  }\n'
    '  ]\n'
    '}'
)


def build_repair_prompt(payload: List[Dict[str, Any]]) -> str:
    block = json.dumps(payload, ensure_ascii=False, indent=None)
    return (
        "Repair the following questions. Each item tells you exactly what is broken "
        "and what to fix. Change only the listed fields and preserve the intended assessment task. Always keep "
        "question_id unchanged.\n\n"
        "## Items to repair\n"
        f"{block}\n\n"
        "## Repair object format (use exactly these field names)\n"
        f"{REPAIR_OUTPUT_SCHEMA}\n\n"
        "Return ONLY one JSON object with a `repairs` array. No text before or after, "
        "no fences, and never return a bare root array."
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _sanitize_verdict(raw: Any, known: Set[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    qid = str(raw.get("question_id") or "").strip()
    if not qid or qid not in known:
        return None
    if not isinstance(raw.get("question_valid"), bool) or not isinstance(
        raw.get("answer_valid"), bool
    ):
        return None
    qv = raw["question_valid"]
    av = raw["answer_valid"]
    action = str(raw.get("action") or "").upper()
    if action not in ACTIONS:
        return None
    fields = raw.get("fields_to_fix")
    if not isinstance(fields, list):
        fields = []
    explicit = any(str(f).strip() for f in fields)
    if qv and av:
        # An explicit non-PASS action with fields is the model flagging a real
        # problem (e.g. FIX_OPTIONS for two correct options); do not collapse it.
        if not (explicit and action in ("FIX_OPTIONS", "FIX_ANSWER", "FIX_QUESTION", "FIX_QUESTION_AND_ANSWER")):
            action = "PASS"
    elif not qv and not av:
        action = "FIX_QUESTION_AND_ANSWER"
    elif not qv:
        action = "FIX_QUESTION"
    else:
        if action not in ("FIX_OPTIONS", "FIX_ANSWER"):
            action = "FIX_ANSWER"
    return {
        "question_id": qid,
        "question_valid": qv,
        "answer_valid": av,
        "action": action,
        "fields_to_fix": [str(f) for f in fields],
        "failure_code": str(raw.get("failure_code") or ("pass" if action == "PASS" else "other")),
        "expected_results": [
            str(value) for value in (raw.get("expected_results") or [])
            if str(value).strip()
        ],
        "reason": str(raw.get("reason") or "").strip(),
        "expected_fix": str(raw.get("expected_fix") or "").strip(),
        "solved_answer": raw.get("solved_answer"),
        "independent_solution": raw.get("independent_solution"),
        "certificate_valid": raw.get("certificate_valid"),
        "first_incorrect_step": (
            str(raw.get("first_incorrect_step") or "").strip() or None
        ),
    }


def _reconcile_simple_expected_answer(
    verdict: Dict[str, Any], entry: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve self-contradictory MCQ/TF verdicts from explicit expected answers."""
    if verdict.get("action") == "PASS" or not verdict.get("question_valid"):
        return verdict
    if verdict.get("certificate_valid") is False:
        # A matching answer label cannot erase a broken/missing quantitative
        # solution certificate or an incorrect intermediate calculation.
        return verdict
    qtype = str(entry.get("question_type") or "")
    item = entry.get("question") or {}
    expected = [str(value).strip() for value in verdict.get("expected_results") or []]
    matches = False
    if qtype == "mcq":
        stored = str(item.get("correct_answer") or "").strip().upper()
        matches = bool(stored) and any(
            (match := re.match(r"^\s*([A-D])(?:\b|\s*:)", value.upper()))
            and match.group(1) == stored
            for value in expected
        )
    elif qtype == "true_false":
        stored_raw = str(item.get("answer") or "").strip().casefold()
        truth_map = {
            "true": True, "false": False, "صح": True, "صحيح": True,
            "صواب": True, "خطأ": False, "خطا": False,
        }
        stored = truth_map.get(stored_raw)
        candidates: list[bool] = []
        for value in expected:
            folded = value.casefold()
            if re.search(r"\btrue\b", folded) or any(word in value for word in ("صح", "صحيح", "صواب")):
                candidates.append(True)
            if re.search(r"\bfalse\b", folded) or any(word in value for word in ("خطأ", "خطا")):
                candidates.append(False)
        matches = stored is not None and stored in candidates
    if matches:
        verdict = dict(verdict)
        verdict.update({
            "answer_valid": True,
            "action": "PASS",
            "fields_to_fix": [],
            "failure_code": "pass",
            "expected_fix": "",
        })
    return verdict


def validate_exam(
        exam: Dict[str, Any], client: Optional[Any] = None,
        question_ids: Optional[Iterable[str]] = None,
        plan_by_slot: Optional[Dict[str, Dict[str, Any]]] = None,
        source_by_chunk: Optional[Dict[str, str]] = None,
        validation_role: str = "general",
) -> Dict[str, Any]:
    """Validate every question in bounded batches, retrying missing IDs once.

    Every question with an ID receives either an actual validator/local verdict
    or an ``UNVALIDATED`` operational status. Missing coverage is never converted
    into a content defect.
    """
    model_number = exam.get("model_number") or 1
    quantitative = validation_role == "stem_quantitative"
    active_system_prompt = STEM_QUANTITATIVE_VALIDATOR_SYSTEM_PROMPT if quantitative else VALIDATOR_SYSTEM_PROMPT
    active_retry_prompt = STEM_QUANTITATIVE_VALIDATOR_RETRY_SYSTEM_PROMPT if quantitative else VALIDATOR_RETRY_SYSTEM_PROMPT
    questions = exam.get("questions") or {}
    selected_ids = (
        {str(question_id) for question_id in question_ids if str(question_id)}
        if question_ids is not None
        else None
    )
    warnings: List[str] = []

    # Deterministic structural problems never consume an LLM call.
    local_verdicts = _deterministic_exam_verdicts(
        exam, plan_by_slot, selected_ids
    )
    entries: List[Dict[str, Any]] = []
    for qtype, container, item in _iter_questions(questions):
        qid = item.get("question_id")
        if not qid:
            warnings.append(f"validate: question missing question_id; skipped")
            continue
        if selected_ids is not None and str(qid) not in selected_ids:
            continue
        if str(qid) in local_verdicts:
            continue
        slot = str(item.get("slot_id") or "")
        plan_item = (plan_by_slot or {}).get(slot)
        payload = dict(item)
        if qtype == "fill_in_the_blank" and isinstance(container, dict):
            payload["word_bank"] = container.get("word_bank") or []
        entries.append({
            "question_id": str(qid),
            "question_type": qtype,
            "question": payload,
            "plan": plan_item or {},
            "expected_catalog_task": _catalog_task_contract(plan_item),
            "_source_chunks": _assigned_source_chunks(plan_item, source_by_chunk),
        })

    scope_ids = [*local_verdicts, *(entry["question_id"] for entry in entries)]
    logger.info(
        "Validation scope | model=%d | question_count=%d | question_ids=%s",
        model_number,
        len(scope_ids),
        scope_ids,
    )

    verdicts: Dict[str, Dict[str, Any]] = dict(local_verdicts)
    coverage: List[Dict[str, Any]] = []
    if entries:
        batches = [
            entries[offset: offset + VALIDATOR_BATCH_SIZE]
            for offset in range(0, len(entries), VALIDATOR_BATCH_SIZE)
        ]

        def validate_batch(
            batch_index: int, batch: List[Dict[str, Any]],
        ) -> tuple[int, Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
            batch_warnings: List[str] = []
            batch_verdicts: Dict[str, Dict[str, Any]] = {}
            sent_ids = [entry["question_id"] for entry in batch]
            known_ids = set(sent_ids)
            returned_ids: List[str] = []
            unexpected_ids: List[str] = []
            duplicate_ids: List[str] = []
            entry_by_id = {entry["question_id"]: entry for entry in batch}

            def structural_errors(
                raw_result: Any, expected_ids: List[str], *, salvage: bool,
            ) -> tuple[str, list[dict[str, Any]]]:
                """Validate the exact response envelope and retain only proven rows.

                A malformed multi-question envelope may contain independently valid
                rows. Those rows are safe to retain; every unresolved ID is retried
                alone under the same strict one-verdict schema.
                """
                expected = set(expected_ids)
                if not isinstance(raw_result, dict):
                    return "root must be one JSON object", []
                rows = raw_result.get("verdicts")
                if not isinstance(rows, list):
                    return "root.verdicts must be an array", []
                raw_ids = [
                    str(row.get("question_id") or "").strip()
                    if isinstance(row, dict) else ""
                    for row in rows
                ]
                counts = {qid: raw_ids.count(qid) for qid in set(raw_ids) if qid}
                duplicate = sorted(qid for qid, count in counts.items() if count > 1)
                missing = sorted(expected - set(raw_ids))
                extra = sorted(set(raw_ids) - expected - {""})
                invalid_shape = sum(not isinstance(row, dict) for row in rows)
                problems: list[str] = []
                if len(rows) != len(expected_ids):
                    problems.append(
                        f"expected {len(expected_ids)} verdict(s), received {len(rows)}"
                    )
                if missing:
                    problems.append(f"missing question_ids: {missing}")
                if extra:
                    problems.append(f"extra question_ids: {extra}")
                if duplicate:
                    problems.append(f"duplicate question_ids: {duplicate}")
                if invalid_shape:
                    problems.append(f"{invalid_shape} verdict row(s) were not objects")

                response_model = validator_response_model(
                    expected_ids, quantitative=quantitative
                )
                try:
                    validated = response_model.model_validate(raw_result).model_dump()
                    return "", list(validated["verdicts"])
                except Exception as exc:
                    problems.append(f"schema validation failed: {exc}")

                if not salvage:
                    return "; ".join(problems), []

                valid_rows: list[dict[str, Any]] = []
                for qid in expected_ids:
                    if counts.get(qid) != 1:
                        continue
                    row = next((
                        value for value in rows
                        if isinstance(value, dict)
                        and str(value.get("question_id") or "").strip() == qid
                    ), None)
                    if row is None:
                        continue
                    try:
                        single_model = validator_response_model(
                            [qid], quantitative=quantitative
                        )
                        validated = single_model.model_validate(
                            {"verdicts": [row]}
                        ).model_dump()
                        valid_rows.append(validated["verdicts"][0])
                    except Exception:
                        continue
                return "; ".join(problems), valid_rows

            def consume(
                raw_result: Any,
                allowed_ids: Set[str],
                *,
                retry: bool = False,
                record_returned: bool = True,
            ) -> None:
                results = raw_result.get("verdicts") if isinstance(raw_result, dict) else None
                if not isinstance(results, list):
                    batch_warnings.append(
                        "validate: retry returned no usable verdict array"
                        if retry
                        else "validate: validator returned no usable verdict array"
                    )
                    return
                seen_in_response: Set[str] = set()
                for raw_verdict in results:
                    raw_id = (
                        str(raw_verdict.get("question_id") or "").strip()
                        if isinstance(raw_verdict, dict)
                        else ""
                    )
                    if record_returned:
                        returned_ids.append(raw_id)
                    if raw_id in seen_in_response:
                        if raw_id and raw_id not in duplicate_ids:
                            duplicate_ids.append(raw_id)
                        batch_warnings.append(
                            f"validate: duplicate verdict for {raw_id} ignored"
                        )
                        continue
                    seen_in_response.add(raw_id)
                    if not raw_id or raw_id not in allowed_ids:
                        if raw_id and raw_id not in unexpected_ids:
                            unexpected_ids.append(raw_id)
                        batch_warnings.append(
                            "validate: verdict with unknown/missing question_id rejected"
                        )
                        continue
                    v = _sanitize_verdict(raw_verdict, allowed_ids)
                    if v is None:
                        batch_warnings.append(
                            "validate: verdict with unknown/missing question_id rejected"
                        )
                        continue
                    if quantitative and (
                        not isinstance(v.get("independent_solution"), dict)
                        or not isinstance(v.get("certificate_valid"), bool)
                        or (
                            v.get("certificate_valid") is False
                            and not v.get("first_incorrect_step")
                            and not (
                                v.get("action") == "FIX_ANSWER"
                                and set(v.get("fields_to_fix") or []) == {
                                    "solution_certificate"
                                }
                                and str(v.get("failure_code") or "")
                                == "incomplete_solution"
                            )
                        )
                    ):
                        batch_warnings.append(
                            f"validate: quantitative verdict for {raw_id} omitted the "
                            "independent solution/certificate comparison; rejected"
                        )
                        continue
                    entry = entry_by_id.get(raw_id, {})
                    qtype = str(entry.get("question_type") or "")
                    if v.get("action") == "PASS":
                        v["fields_to_fix"] = []
                    else:
                        requested_fields = list(v.get("fields_to_fix") or [])
                        authorized_fields = sorted(_allowed_canonical(
                            qtype, v.get("fields_to_fix") or [], str(v.get("action") or "")
                        ))
                        if requested_fields and not authorized_fields:
                            batch_warnings.append(
                                f"validate: verdict for {raw_id} used non-canonical fields; rejected"
                            )
                            continue
                        v["fields_to_fix"] = authorized_fields
                    v = _reconcile_simple_expected_answer(v, entry)
                    # A retry never overwrites a verdict already accepted from
                    # the original batch response.
                    if v["question_id"] not in batch_verdicts:
                        batch_verdicts[v["question_id"]] = v

            active_max_tokens = (
                STEM_QUANTITATIVE_VALIDATOR_MAX_TOKENS
                if quantitative else GENERAL_VALIDATOR_MAX_TOKENS
            )
            batch_llm = client or create_llm_client(local_reasoning=VALIDATOR_REASONING)
            initial_rejection_reason = ""
            initial_rows_already_recorded = False
            try:
                raw = request_structured(
                    batch_llm,
                    build_validator_prompt(batch, quantitative=quantitative),
                    response_model=validator_response_model(
                        sent_ids, quantitative=quantitative
                    ),
                    schema_name="validator_verdicts",
                    agent="validator",
                    item_count=len(batch),
                    expected_ids=sent_ids,
                    expected_id_field="question_id",
                    local_json=False,
                    system_prompt=active_system_prompt,
                    temperature=VALIDATOR_TEMPERATURE,
                    max_tokens=active_max_tokens,
                )
                initial_rejection_reason, usable_rows = structural_errors(
                    raw, sent_ids, salvage=True
                )
                if initial_rejection_reason:
                    initial_raw_rows = (
                        raw.get("verdicts") if isinstance(raw, dict) else []
                    )
                    initial_raw_ids = [
                        str(row.get("question_id") or "").strip()
                        if isinstance(row, dict) else ""
                        for row in initial_raw_rows or []
                    ]
                    returned_ids.extend(initial_raw_ids)
                    for raw_id in sorted(set(initial_raw_ids)):
                        if raw_id and initial_raw_ids.count(raw_id) > 1:
                            duplicate_ids.append(raw_id)
                        if raw_id and raw_id not in known_ids:
                            unexpected_ids.append(raw_id)
                    initial_rows_already_recorded = True
                    batch_warnings.append(
                        "validate: validator structured response rejected ("
                        f"{initial_rejection_reason})"
                    )
                raw = {"verdicts": usable_rows}
            except Exception as exc:
                logger.exception(
                    "Validator call failed | role=%s | question_ids=%s",
                    validation_role, sent_ids,
                )
                batch_warnings.append(
                    f"validate: validator LLM call failed ({exc}); no verdicts received"
                )
                initial_rejection_reason = str(exc)
                raw = {"verdicts": []}
            consume(
                raw, known_ids,
                record_returned=not initial_rows_already_recorded,
            )

            missing = [
                entry for entry in batch
                if entry["question_id"] not in batch_verdicts
            ]
            if missing:
                batch_warnings.append(
                    f"validate: {len(missing)} unresolved question(s); retrying each once"
                )
                rejection_reason = initial_rejection_reason or (
                    "the previous response did not provide a schema-valid verdict "
                    "for this question_id"
                )
                for missing_entry in missing:
                    missing_id = missing_entry["question_id"]
                    try:
                        raw2 = request_structured(
                            batch_llm,
                            build_validator_prompt(
                                [missing_entry], quantitative=quantitative,
                                rejection_reason=rejection_reason,
                            ),
                            response_model=validator_response_model(
                                [missing_id], quantitative=quantitative
                            ),
                            schema_name="validator_missing_verdict",
                            agent="validator",
                            item_count=1,
                            expected_ids=[missing_id],
                            expected_id_field="question_id",
                            local_json=False,
                            system_prompt=active_retry_prompt,
                            temperature=VALIDATOR_TEMPERATURE,
                            max_tokens=active_max_tokens,
                        )
                        retry_error, retry_rows = structural_errors(
                            raw2, [missing_id], salvage=False
                        )
                        if retry_error:
                            raise ValueError(retry_error)
                        consume(
                            {"verdicts": retry_rows}, {missing_id}, retry=True
                        )
                    except Exception as exc:
                        logger.exception(
                            "Validator retry failed | role=%s | question_ids=%s",
                            validation_role, [missing_id],
                        )
                        batch_warnings.append(
                            f"validate: validator retry failed for {missing_id} ({exc})"
                        )

            final_missing = [qid for qid in sent_ids if qid not in batch_verdicts]
            batch_coverage = {
                "sent_ids": sent_ids,
                "returned_ids": returned_ids,
                "matched_ids": [qid for qid in sent_ids if qid in batch_verdicts],
                "missing_ids": final_missing,
                "unexpected_ids": unexpected_ids,
                "duplicate_returned_ids": duplicate_ids,
            }
            return (
                batch_index, batch_verdicts, batch_warnings, batch_coverage
            )

        max_workers = min(MAX_PARALLEL_VALIDATOR_CALLS, len(batches))
        logger.info(
            "Parallel validator batches | role=%s | batches=%d | max_workers=%d",
            validation_role, len(batches), max_workers,
        )
        completed: dict[
            int, tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]
        ] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(copy_context().run, validate_batch, index, batch): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                index, batch_verdicts, batch_warnings, batch_coverage = future.result()
                completed[index] = (
                    batch_verdicts, batch_warnings, batch_coverage
                )

        # Completion order never affects the persisted report.
        for index in range(len(batches)):
            batch_verdicts, batch_warnings, batch_coverage = completed[index]
            verdicts.update(batch_verdicts)
            warnings.extend(batch_warnings)
            coverage.append(batch_coverage)

    all_pass = True
    ordered: List[Dict[str, Any]] = []
    for _, _, item in _iter_questions(questions):
        qid = str(item.get("question_id") or "")
        if not qid:
            continue
        if selected_ids is not None and qid not in selected_ids:
            continue
        v = verdicts.get(qid)
        if v is None:
            warnings.append(
                f"validate: no verdict returned for {qid} after retry; marked unvalidated"
            )
            v = {
                "question_id": qid,
                "question_valid": None,
                "answer_valid": None,
                "action": "UNVALIDATED",
                "fields_to_fix": [],
                "reason": "No validator verdict was returned after retry.",
                "expected_fix": "Retry validation; do not modify the question.",
            }
        if v["action"] != "PASS":
            all_pass = False
        v["validation_role"] = validation_role
        ordered.append(v)

    logger.info(
        "Exam validated | model=%d | questions=%d | invalid=%d | unvalidated=%d",
        model_number,
        len(ordered),
        sum(1 for v in ordered if v["action"] in REPAIR_ACTIONS),
        sum(1 for v in ordered if v["action"] == "UNVALIDATED"),
    )
    return {
        "model_number": model_number,
        "all_pass": all_pass,
        "verdicts": ordered,
        "warnings": warnings,
        "coverage": coverage,
    }


def _merge_validation_report(
    previous: Dict[str, Any], update: Dict[str, Any]
) -> Dict[str, Any]:
    """Replace revalidated verdicts by ID and retain every untouched verdict."""
    replacements = {
        str(verdict.get("question_id")): verdict
        for verdict in (update.get("verdicts") or [])
        if verdict.get("question_id")
    }
    merged_verdicts: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for verdict in previous.get("verdicts") or []:
        qid = str(verdict.get("question_id") or "")
        merged_verdicts.append(replacements.get(qid, verdict))
        if qid:
            seen.add(qid)
    for qid, verdict in replacements.items():
        if qid not in seen:
            merged_verdicts.append(verdict)

    merged = dict(previous)
    merged["model_number"] = update.get(
        "model_number", previous.get("model_number")
    )
    merged["verdicts"] = merged_verdicts
    merged["missing_slots"] = list(update.get("missing_slots", previous.get("missing_slots") or []))
    merged["missing_sections"] = list(
        update.get("missing_sections", previous.get("missing_sections") or [])
    )
    merged["all_pass"] = (
        all(verdict.get("action") == "PASS" for verdict in merged_verdicts)
        and not merged["missing_slots"] and not merged["missing_sections"]
    )
    merged["warnings"] = [
        *(previous.get("warnings") or []),
        *(update.get("warnings") or []),
    ]
    merged["coverage"] = [
        *(previous.get("coverage") or []),
        *(update.get("coverage") or []),
    ]
    return merged


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------
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
        payload.append(
            {
                "question_id": qid,
                "action": v.get("action", "FIX_ANSWER"),
                "fields_to_fix": v.get("fields_to_fix") or [],
                "reason": v.get("reason") or "",
                "expected_fix": v.get("expected_fix") or "",
                "expected_results": list(v.get("expected_results") or []),
                "independent_solution": v.get("independent_solution"),
                "first_incorrect_step": v.get("first_incorrect_step"),
                "replacement_mode": _requires_plan_replacement(v),
                "rejected_distractors": rejected_distractors,
                "required_repair": required_repair,
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

    llm = client or create_llm_client(
        local_reasoning=REPAIR_REASONING,
        deepseek_model="deepseek-flash",
    )
    user_prompt = build_repair_prompt(payload)
    try:
        repair_model_items = [
            (item["question_id"], by_id[item["question_id"]][0])
            for item in payload
        ]
        raw = request_structured(
            llm,
            user_prompt,
            response_model=repair_response_model(repair_model_items),
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
        new_item = r.get("question")
        if not isinstance(new_item, dict):
            warnings.append(f"repair: {qid} returned no question payload; rejected")
            continue
        proposed_fitb_bank = None

        stashed_id = item.get("question_id")
        stashed_slot_id = item.get("slot_id")
        stashed_fitb_entry_id = item.get("fitb_entry_id")
        if qtype == "fill_in_the_blank":
            new_item = {
                "slot_id": stashed_slot_id,
                "fitb_entry_id": stashed_fitb_entry_id,
                "question": new_item.get("stem"),
                "answers": list(item.get("answers") or []),
                "distractors": [
                    new_item.get("distractor_1"), new_item.get("distractor_2")
                ],
            }
        normalized = _normalize(qtype, new_item)
        if normalized is None:
            warnings.append(f"repair: {qid} output failed normalization; rejected")
            continue
        # Never copy a certificate from the pre-repair question. It is derived
        # from the old content and becomes stale as soon as an answer-bearing
        # field changes. Quantitative repairs must return a fresh certificate;
        # general repairs retain an explicit null invalidation marker so the
        # quantitative validator must request a fresh certificate.
        if repair_role != "stem_quantitative":
            normalized["solution_certificate"] = None
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

        verdict = verdict_by_id.get(qid) or {}
        request_item = next(
            (value for value in payload if value.get("question_id") == qid), {}
        )
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
        if _requires_plan_replacement(verdict):
            allowed = set(_ALL_CANONICAL.get(qtype, set()))
        else:
            allowed = _allowed_canonical(
                qtype, verdict.get("fields_to_fix") or [], verdict.get("action") or "FIX_ANSWER"
            )
        if repair_role == "stem_quantitative":
            allowed.add("solution_certificate")
        else:
            # General repairs deliberately remove derived certificates before
            # revalidation. That invalidation is mandatory bookkeeping, not an
            # unauthorized student-facing edit.
            allowed.add("solution_certificate")
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
    }


# --------------------------------------------------------------------------
# LangGraph nodes
# --------------------------------------------------------------------------
def _plan_context_for_model(
    state: Dict[str, Any], model_number: int
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    plan_by_slot: Dict[str, Dict[str, Any]] = {}
    assessment_profile = state.get("assessment_profile") or {}
    type_profiles = assessment_profile.get("question_types") or {}
    # Plans are canonical and ordered snapshots. Use only the current snapshot
    # for this model so stale task identities cannot leak into revalidation.
    model_plans = [
        plan for plan in (state.get("plans") or [])
        if int(plan.get("model_number") or 0) == int(model_number)
    ]
    if model_plans:
        plan = model_plans[-1]
        for qtype, values in (plan.get("items") or {}).items():
            for item in values or []:
                sid = str(item.get("slot_id") or "")
                if sid:
                    type_profile = type_profiles.get(qtype) or {}
                    plan_by_slot[sid] = {
                        **item,
                        "assessment_guidance": {
                            "domains": assessment_profile.get("domains") or [],
                            "educational_level": assessment_profile.get("educational_level") or "unspecified",
                            "assessment_style": assessment_profile.get("assessment_style") or "",
                            "question_type": qtype,
                            "support": type_profile.get("support") or "supported",
                            "task_operations": type_profile.get("task_operations") or [],
                            "form_guidance": type_profile.get("form_guidance") or "",
                            "unsupported_visual_tasks": assessment_profile.get("unsupported_visual_tasks") or [],
                        },
                    }
    source_by_chunk = {
        str(chunk.get("child_id")): str(chunk.get("content") or "")
        for chunk in state.get("retrieved_chunks") or []
        if chunk.get("child_id")
    }
    return plan_by_slot, source_by_chunk


def _validate_with_context(
    exam: Dict[str, Any],
    question_ids: Optional[Iterable[str]],
    plan_by_slot: Dict[str, Dict[str, Any]],
    source_by_chunk: Dict[str, str],
    client: Optional[Any] = None,
    validation_role: str = "general",
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if question_ids is not None:
        kwargs["question_ids"] = question_ids
    if plan_by_slot or source_by_chunk:
        kwargs["plan_by_slot"] = plan_by_slot
        kwargs["source_by_chunk"] = source_by_chunk
    if client is not None:
        kwargs["client"] = client
    # Keep the historical call shape for the default role.  This also lets
    # lightweight test/dry-run validators that implement the old signature
    # continue to stand in for the general validator.
    if validation_role != "general":
        kwargs["validation_role"] = validation_role
    return validate_exam(exam, **kwargs)


def _quantitative_question_ids(exam: Dict[str, Any], plan_by_slot: Dict[str, Dict[str, Any]]) -> list[str]:
    modes = {"calculation", "symbolic_manipulation", "derivation_relationship",
             "law_or_theorem_application", "optimization"}
    operation_words = ("calculate", "compute", "solve", "derive", "balance", "optimize",
                       "evaluate", "convert", "determine_value", "write_equation")
    result: list[str] = []
    for qtype, _container, item in _iter_questions(exam.get("questions") or {}):
        if qtype == "fill_in_the_blank":
            continue
        qid = str(item.get("question_id") or "")
        plan = plan_by_slot.get(str(item.get("slot_id") or "")) or {}
        mode = str(plan.get("reasoning_mode") or "").casefold()
        operation = str(plan.get("task_operation") or "").casefold()
        quantitative = (qtype in {"equation", "word_problem"} or mode in modes
                        or any(word in operation for word in operation_words)
                        or (qtype == "mcq" and bool(re.search(r"[=<>≤≥]|\d", _question_task_text(qtype, item)))))
        if qid and quantitative:
            result.append(qid)
    return result


def _merge_role_reports(general: Dict[str, Any], quantitative: Dict[str, Any]) -> Dict[str, Any]:
    quant_by_id = {str(v.get("question_id")): v for v in quantitative.get("verdicts") or []}
    merged = []
    for verdict in general.get("verdicts") or []:
        quant = quant_by_id.get(str(verdict.get("question_id") or ""))
        if quant is None or quant.get("action") == "PASS":
            chosen = dict(verdict)
        elif verdict.get("action") == "PASS":
            chosen = dict(quant)
        else:
            chosen = dict(quant)
            chosen["additional_general_issue"] = {key: verdict.get(key) for key in
                ("action", "failure_code", "fields_to_fix", "reason", "expected_fix")}
        merged.append(chosen)
    return {**general, "verdicts": merged,
            "all_pass": all(v.get("action") == "PASS" for v in merged),
            "coverage": [*(general.get("coverage") or []), *(quantitative.get("coverage") or [])],
            "warnings": [*(general.get("warnings") or []), *(quantitative.get("warnings") or [])],
            "general_review": general,
            "quantitative_review": quantitative}


def _semantic_identity(
    plan: Dict[str, Any], question_type: Optional[str] = None
) -> Dict[str, Any]:
    """Return the compact diagnostic identity persisted for a plan slot."""
    return {
        "slot_id": plan.get("slot_id"),
        "question_type": question_type or plan.get("question_type"),
        "knowledge_focus": plan.get("knowledge_focus"),
        "student_work_product": plan.get("student_work_product"),
        "task_operation": plan.get("task_operation"),
        "relationship": plan.get("relationship") or plan.get("concept"),
        "target": plan.get("target"),
        "reasoning_mode": plan.get("reasoning_mode"),
        "given_conditions": list(plan.get("given_conditions") or []),
        "scenario_family": plan.get("scenario_family"),
    }


def _regenerate_plan_slot(
    state: Dict[str, Any],
    exam: Dict[str, Any],
    model_number: int,
    question_id: str,
    plan_by_slot: Dict[str, Dict[str, Any]],
    source_by_chunk: Dict[str, str],
) -> tuple[bool, list[str]]:
    """Regenerate one replaced plan through its normal type-specific generator."""
    from app.online.exam_builder import (
        _generate_fitb_type,
        _generate_obj_bundle,
        _generate_type_from_plan,
        _infer_question_type,
        _remember_question,
    )

    questions = exam.get("questions") or {}
    entry = _build_by_id(questions).get(question_id)
    if entry is None:
        return False, [f"repair: unknown question_id {question_id}"]
    qtype, _container, current = entry
    sid = str(current.get("slot_id") or "")
    plan = plan_by_slot.get(sid) or {}
    if not plan:
        return False, [f"repair: no plan found for duplicate slot {sid}"]

    sources = _assigned_source_chunks(plan, source_by_chunk)
    if not sources:
        return False, [f"repair: no assigned evidence found for duplicate slot {sid}"]
    context = "\n\n".join(
        f"[source_chunk_id={chunk_id}]\n{text}" for chunk_id, text in sources.items()
    )
    generation_plan = {**plan, "assigned_sources": sources}
    if plan.get("task_reused") and not generation_plan.get("previous_task_questions"):
        related_slots = set(str(value) for value in plan.get("forced_duplicate_slot_ids") or [])
        generation_plan["previous_task_questions"] = [
            {
                "context_role": "previous_question_from_reused_task",
                "question_id": other_item.get("question_id"),
                "slot_id": other_item.get("slot_id"),
                "task_id": (plan_by_slot.get(str(other_item.get("slot_id") or "")) or {}).get(
                    "task_id"
                ),
                "question_text": _question_task_text(other_type, other_item),
                "answer_pattern": _question_result_text(other_type, other_item),
            }
            for other_type, _other_container, other_item in _iter_questions(questions)
            if str(other_item.get("slot_id") or "") in related_slots
            and str(other_item.get("question_id") or "") != question_id
        ]
    within_model = [
        item for _other_type, _other_container, item in _iter_questions(questions)
        if str(item.get("question_id") or "") != question_id
    ]
    seen: set[tuple[str, str]] = set()
    for item in within_model:
        _remember_question(_infer_question_type(item), item, seen)
    difficulty = str(state.get("difficulty") or "mix")
    language = str(state.get("document_language") or "en")
    warnings: list[str] = []

    if qtype in {"mcq", "true_false"}:
        bundle, warnings = _generate_obj_bundle(
            {qtype: [generation_plan]}, context, difficulty, model_number,
            within_model, seen, [], language=language,
            global_plan_by_slot=plan_by_slot,
        )
        candidates = list(bundle.get(qtype) or [])
    elif qtype == "fill_in_the_blank":
        section, warnings = _generate_fitb_type(
            1, [generation_plan], context, difficulty, model_number,
            within_model, [], language=language,
            global_plan_by_slot=plan_by_slot,
        )
        candidates = list((section or {}).get("items") or [])
    else:
        candidates, warnings = _generate_type_from_plan(
            qtype, [generation_plan], context, difficulty, model_number,
            seen, within_model, [], language=language,
            global_plan_by_slot=plan_by_slot,
        )
    if len(candidates) != 1:
        return False, [*warnings, f"repair: normal generator did not replace slot {sid}"]

    replacement = candidates[0]
    replacement["question_id"] = question_id
    replacement["slot_id"] = sid
    current.clear()
    current.update(replacement)
    if qtype == "fill_in_the_blank":
        section = questions.get(qtype)
        if not isinstance(section, dict) or not _rebuild_fitb_word_bank(section):
            return False, [*warnings, f"repair: FITB bank rebuild failed for slot {sid}"]
    return True, warnings


def _question_type_from_slot(slot_id: str, plan_item: Dict[str, Any] | None = None) -> str:
    qtype = str((plan_item or {}).get("question_type") or "")
    if qtype:
        return qtype
    match = re.match(r"m\d+_(.+)_\d+$", slot_id)
    return match.group(1) if match else "unknown"


def _repair_classification(
    failure_code: str, *, validation_role: str = "general",
) -> tuple[str, str]:
    """Map a diagnosis to a mutation strategy without weakening validation."""
    code = str(failure_code or "validation_failure")
    if code == "missing_requested_slot":
        return "missing_slot", "regenerate_missing_slot"
    if code in TASK_REPLACEMENT_FAILURES:
        return "task_or_evidence_invalid", "replace_assignment"
    if validation_role == "stem_quantitative" or code in {
        "incomplete_solution", "incorrect_calculation", "invalid_solution_certificate",
    }:
        return "quantitative_solution_invalid", "repair_question"
    if code in STRUCTURE_FAILURES:
        return "structure_invalid", "repair_question"
    return "question_repairable", "repair_question"


def _manifest_from_report(
    report: Dict[str, Any], exam: Dict[str, Any],
    plan_by_slot: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Create one actionable target for every invalid or absent requested slot."""
    targets: list[Dict[str, Any]] = []
    by_id = _build_by_id(exam.get("questions") or {})
    for verdict in report.get("verdicts") or []:
        if verdict.get("action") == "PASS":
            continue
        qid = str(verdict.get("question_id") or "")
        entry = by_id.get(qid)
        if entry is None:
            continue
        qtype, _container, item = entry
        slot_id = str(item.get("slot_id") or "")
        plan_item = plan_by_slot.get(slot_id) or {}
        category, repair_type = _repair_classification(
            str(verdict.get("failure_code") or ""),
            validation_role=str(verdict.get("validation_role") or "general"),
        )
        if str(verdict.get("failure_code") or "") in {
            "missing_required_given", "missing_required_givens", "invalid_required_givens",
        }:
            diagnosis = " ".join((
                str(verdict.get("reason") or ""), str(verdict.get("expected_fix") or "")
            )).casefold()
            if any(marker in diagnosis for marker in (
                "unsupported", "not supported", "not in evidence", "not grounded", "impossible",
            )):
                category, repair_type = "task_or_evidence_invalid", "replace_assignment"
        targets.append({
            "model_number": report.get("model_number"),
            "slot_id": slot_id,
            "question_id": qid,
            "question_type": qtype,
            "status": "invalid",
            "task_id": plan_item.get("task_id"),
            "fitb_entry_id": plan_item.get("fitb_entry_id"),
            "evidence_ids": list(plan_item.get("evidence_chunk_ids") or []) or [
                value for value in [plan_item.get("evidence_chunk_id") or plan_item.get("source_chunk_id")]
                if value
            ],
            "failure_code": verdict.get("failure_code") or "validation_failure",
            "failure_reason": verdict.get("reason") or verdict.get("expected_fix") or "",
            "repair_category": category,
            "repair_type": repair_type,
            "what_is_wrong": verdict.get("reason") or "",
            "what_must_change": verdict.get("expected_fix") or "",
            "preferred_recovery": repair_type,
            "validation_role": verdict.get("validation_role") or "general",
            "verdict": deepcopy(verdict),
        })
    for missing in report.get("missing_slots") or []:
        slot_id = str(missing.get("slot_id") or "")
        targets.append({
            "model_number": report.get("model_number"),
            "slot_id": slot_id,
            "question_id": None,
            "question_type": missing.get("question_type") or _question_type_from_slot(slot_id),
            "status": "missing",
            "task_id": missing.get("task_id"),
            "fitb_entry_id": missing.get("fitb_entry_id"),
            "evidence_ids": [missing.get("evidence_chunk_id")] if missing.get("evidence_chunk_id") else [],
            "failure_code": "missing_requested_slot",
            "failure_reason": missing.get("recovery_requirement") or "Requested slot was not generated.",
            "repair_category": "missing_slot",
            "repair_type": "regenerate_missing_slot",
            "what_is_wrong": "The requested slot is absent from Generator output.",
            "what_must_change": "Generate and validate the exact assigned slot.",
            "preferred_recovery": "regenerate original assignment; replace it only after a failed candidate",
            "validation_role": "general",
            "verdict": None,
        })
    return targets


def _merge_repair_manifest(
    old: list[Dict[str, Any]], fresh: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Keep unresolved slots until a passing candidate explicitly resolves them."""
    old_by_key = {
        (int(item.get("model_number") or 0), str(item.get("slot_id") or "")): deepcopy(item)
        for item in old if item.get("slot_id")
    }
    fresh_by_key = {
        (int(item.get("model_number") or 0), str(item.get("slot_id") or "")): deepcopy(item)
        for item in fresh if item.get("slot_id")
    }
    merged: list[Dict[str, Any]] = []
    for key in sorted(set(old_by_key) | set(fresh_by_key)):
        if key in fresh_by_key:
            current = fresh_by_key[key]
            previous = old_by_key.get(key) or {}
            current["history"] = list(previous.get("history") or [])
            if current["history"] and current["history"][-1].get("new_failure_reason"):
                current["failure_code"] = previous.get("failure_code", current.get("failure_code"))
                current["failure_reason"] = previous.get("failure_reason", current.get("failure_reason"))
                current["repair_category"] = previous.get("repair_category", current.get("repair_category"))
                current["repair_type"] = previous.get("repair_type", current.get("repair_type"))
                current["verdict"] = previous.get("verdict", current.get("verdict"))
            merged.append(current)
        elif old_by_key[key].get("status") != "resolved":
            # A selective validation cannot silently erase another slot.
            merged.append(old_by_key[key])
    return merged


def _placeholder_question(qtype: str, slot_id: str, question_id: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {"slot_id": slot_id, "question_id": question_id}
    if qtype == "mcq":
        base.update(question="", options={}, correct_answer="")
    elif qtype == "true_false":
        base.update(statement="", answer=False)
    elif qtype == "fill_in_the_blank":
        base.update(question="_____", answers=[], distractors=[])
    elif qtype == "definition":
        base.update(term="", reference_answer="")
    elif qtype in {"why", "essay"}:
        base.update(question="", reference_answer="")
    elif qtype == "equation":
        base.update(equation="", solution_steps=[], final_answer="")
    else:
        base.update(question="", solution_steps=[], final_answer="")
    return base


def _put_question(exam: Dict[str, Any], qtype: str, item: Dict[str, Any]) -> None:
    """Insert/replace a slot while preserving its requested section and slot identity."""
    questions = exam.setdefault("questions", {})
    section = questions.get(qtype)
    if qtype == "fill_in_the_blank":
        if not isinstance(section, dict):
            section = {"items": [], "word_bank": []}
            questions[qtype] = section
        values = section.setdefault("items", [])
    else:
        if not isinstance(section, list):
            section = []
            questions[qtype] = section
        values = section
    slot_id = str(item.get("slot_id") or "")
    for index, current in enumerate(values):
        if str(current.get("slot_id") or "") == slot_id:
            values[index] = deepcopy(item)
            break
    else:
        values.append(deepcopy(item))
    values.sort(key=lambda value: str(value.get("slot_id") or ""))
    if qtype == "fill_in_the_blank":
        _rebuild_fitb_word_bank(section)


def validate_generated_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate correctness only; semantic duplication is a later graph stage."""
    generated = deepcopy(state.get("generated_exams") or [])
    plans = deepcopy(state.get("plans") or [])
    validated = list(state.get("validated_models") or [])
    attempts = state.get("question_repair_attempts") or {}
    pending_revalidation = dict(state.get("pending_revalidation_ids") or {})
    warnings = list(state.get("warnings") or [])
    stored_domain = str((state.get("assessment_profile") or {}).get("domain") or "")
    general_client = None
    quantitative_client = None
    if stored_domain:
        from app.llm.factory import create_review_client
        general_client = create_review_client(domain=stored_domain, stage="general_validator")
        if stored_domain.upper() == "STEM":
            quantitative_client = create_review_client(domain=stored_domain, stage="stem_quantitative_validator")
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
    candidate_updates_by_model: Dict[int, Dict[str, Any]] = {}

    # Repair writes are transactional: validate each staged candidate first and
    # mutate the active exam/plan only after every required validator passes.
    for candidate in state.get("repair_candidates") or []:
        mn = int(candidate.get("model_number") or 1)
        slot_id = str(candidate.get("slot_id") or "")
        qid = str(candidate.get("question_id") or "")
        staged_exam = candidate.get("exam") or {}
        candidate_state = {**state, "plans": candidate.get("plans") or plans}
        candidate_plan, candidate_sources = _plan_context_for_model(candidate_state, mn)
        args = (staged_exam, [qid], candidate_plan, candidate_sources)
        general = (_validate_with_context(*args, general_client, "general")
                   if general_client is not None else _validate_with_context(*args))
        candidate_report = general
        quantitative_ids = set(_quantitative_question_ids(staged_exam, candidate_plan))
        if quantitative_client is not None and qid in quantitative_ids:
            from app.llm.factory import create_review_client
            fresh_quantitative_client = create_review_client(
                domain="STEM", stage="stem_quantitative_validator"
            )
            quantitative = _validate_with_context(
                staged_exam, [qid], candidate_plan, candidate_sources,
                fresh_quantitative_client, "stem_quantitative",
            )
            candidate_report = _merge_role_reports(general, quantitative)
        verdicts = list(candidate_report.get("verdicts") or [])
        if mn in candidate_updates_by_model:
            candidate_updates_by_model[mn] = _merge_validation_report(
                candidate_updates_by_model[mn], candidate_report
            )
        else:
            candidate_updates_by_model[mn] = candidate_report
        passed = bool(verdicts) and all(v.get("action") == "PASS" for v in verdicts)
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
            "candidate_task_id": candidate.get("candidate_task_id"),
            "candidate_fitb_entry_id": candidate.get("candidate_fitb_entry_id"),
            "revalidation_result": "PASS" if passed else "FAIL",
        }
        if passed:
            active_exam = next((e for e in generated if int(e.get("model_number") or 0) == mn), None)
            staged_entry = _build_by_id(staged_exam.get("questions") or {}).get(qid)
            if active_exam is not None and staged_entry is not None:
                _put_question(active_exam, staged_entry[0], staged_entry[2])
                staged_plan = candidate_plan.get(slot_id)
                if staged_plan:
                    from app.online.duplicate_pipeline import _replace_plan_item
                    _replace_plan_item(plans, mn, slot_id, deepcopy(staged_plan))
                if target is not None:
                    target["status"] = "resolved"
                    target["resolved_by_attempt"] = candidate.get("attempt")
                    target["superseded_failure_code"] = target.get("failure_code")
                event["status"] = "accepted"
                retry_seeds.pop(slot_id, None)
        else:
            failure = next((v for v in verdicts if v.get("action") != "PASS"), {})
            new_code = str(failure.get("failure_code") or "candidate_revalidation_failed")
            new_reason = str(failure.get("reason") or failure.get("expected_fix") or
                             "Candidate did not pass revalidation.")
            category, repair_type = _repair_classification(
                new_code, validation_role=str(failure.get("validation_role") or "general")
            )
            if target is not None:
                target.update(
                    status="invalid" if candidate.get("had_original") else "missing",
                    failure_code=new_code,
                    failure_reason=new_reason,
                    repair_category=category,
                    repair_type=repair_type,
                    verdict=deepcopy(failure) if failure else None,
                )
                target.setdefault("history", []).append(deepcopy(event) | {
                    "new_failure_code": new_code, "new_failure_reason": new_reason,
                })
            candidate_id = str(candidate.get("candidate_fitb_entry_id") or
                               candidate.get("candidate_task_id") or "")
            if candidate_id:
                memory = replacement_memory.setdefault(slot_id, {
                    "attempted_candidate_ids": [], "blacklisted_candidate_ids": [],
                    "candidate_failures": [], "replacement_attempt_count": 0,
                })
                if candidate_id not in memory.setdefault("blacklisted_candidate_ids", []):
                    memory["blacklisted_candidate_ids"].append(candidate_id)
                memory.setdefault("candidate_failures", []).append({
                    "candidate_id": candidate_id, "attempt": candidate.get("attempt"),
                    "reason": new_reason, "failure_code": new_code,
                })
            event.update(status="rejected", new_failure_code=new_code,
                         new_failure_reason=new_reason)
            retry_seeds[slot_id] = deepcopy(candidate)
        validation_repair_history.append(event)
    # Candidates are consumed exactly once, whether accepted or rejected.
    for exam in generated:
        mn = exam.get("model_number")
        if mn in validated:
            continue
        previous = reports_by_model.get(mn)
        plan_by_slot, source_by_chunk = _plan_context_for_model(
            {**state, "plans": plans}, int(mn or 1)
        )
        def run_reviews(ids: Optional[Iterable[str]]) -> Dict[str, Any]:
            args = (exam, ids, plan_by_slot, source_by_chunk)
            general = (_validate_with_context(*args, general_client, "general")
                       if general_client is not None else _validate_with_context(*args))
            if quantitative_client is None:
                return general
            quantitative_ids = set(_quantitative_question_ids(exam, plan_by_slot))
            selected = [qid for qid in (ids or quantitative_ids) if qid in quantitative_ids]
            if not selected:
                return general
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
            quantitative = _validate_with_context(
                exam, selected, plan_by_slot, source_by_chunk,
                active_quantitative_client, "stem_quantitative")
            return _merge_role_reports(general, quantitative)
        if int(mn or 0) in candidate_updates_by_model:
            update = candidate_updates_by_model[int(mn or 0)]
            report = _merge_validation_report(previous, update) if previous is not None else update
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
        else:
            report = run_reviews(None)
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
        # unresolved targets from other slots, but retire explicitly resolved ones.
        unresolved_old = [item for item in repair_manifest if item.get("status") != "resolved"]
        repair_manifest = _merge_repair_manifest(unresolved_old, fresh_manifest)
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

    return {
        "generated_exams": generated,
        "plans": plans,
        "validation_reports": list(reports_by_model.values()),
        "repair_manifest": repair_manifest,
        "repair_candidates": [],
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
    }


def repair_invalid_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stage bounded, reason-aware candidates without mutating the active exam."""
    generated = state.get("generated_exams") or []
    plans = deepcopy(state.get("plans") or [])
    manifest = deepcopy(state.get("repair_manifest") or [])
    attempts = dict(state.get("question_repair_attempts") or {})
    memory_by_slot = deepcopy(state.get("repair_replacement_memory") or {})
    history = list(state.get("validation_repair_history") or [])
    warnings = list(state.get("warnings") or [])
    candidates: list[Dict[str, Any]] = []
    retry_seeds = deepcopy(state.get("repair_retry_seeds") or {})
    validated_models = list(state.get("validated_models") or [])
    exam_by_model = {int(exam.get("model_number") or 0): exam for exam in generated}

    for target in manifest:
        if target.get("status") == "resolved":
            continue
        slot_id = str(target.get("slot_id") or "")
        if not slot_id or (attempts.get(slot_id) or 0) >= MAX_QUESTION_REPAIR_ATTEMPTS:
            continue
        mn = int(target.get("model_number") or 1)
        active_exam = exam_by_model.get(mn)
        if active_exam is None:
            continue
        attempt = (attempts.get(slot_id) or 0) + 1
        attempts[slot_id] = attempt
        qtype = str(target.get("question_type") or _question_type_from_slot(slot_id))
        prefix = f"m{mn}_"
        qid = str(target.get("question_id") or f"model{mn}_{slot_id[len(prefix):] if slot_id.startswith(prefix) else slot_id}")
        retry_seed = retry_seeds.get(slot_id) or {}
        staged_exam = deepcopy(retry_seed.get("exam") or active_exam)
        staged_plans = deepcopy(retry_seed.get("plans") or plans)
        staged_state = {**state, "plans": staged_plans}
        plan_by_slot, source_by_chunk = _plan_context_for_model(staged_state, mn)
        if slot_id in plan_by_slot:
            plan_by_slot[slot_id]["recovery_failure_code"] = target.get("failure_code")
            plan_by_slot[slot_id]["recovery_failure_reason"] = target.get("failure_reason")
            plan_by_slot[slot_id]["recovery_attempt"] = attempt
        had_original = _build_by_id(active_exam.get("questions") or {}).get(qid) is not None
        memory = memory_by_slot.setdefault(slot_id, {
            "original_candidate_id": target.get("fitb_entry_id") or target.get("task_id"),
            "attempted_candidate_ids": [], "blacklisted_candidate_ids": [],
            "candidate_failures": [], "replacement_attempt_count": 0,
        })
        repair_action = str(target.get("repair_type") or "repair_question")
        use_replacement = repair_action == "replace_assignment" or (
            repair_action == "regenerate_missing_slot" and attempt > 1
        )
        selected_id = ""
        if use_replacement:
            from app.online.assessment_catalog import catalog_exhaustion_replacement
            excluded = {str(v) for v in [
                target.get("task_id"), target.get("fitb_entry_id"),
                *(memory.get("blacklisted_candidate_ids") or []),
                *(memory.get("attempted_candidate_ids") or []),
            ] if v}
            used_answers = {
                normalize_fitb_answer_key(answer)
                for _kind, _container, item in _iter_questions(staged_exam.get("questions") or {})
                for answer in (item.get("answers") or []) if answer
            }
            replacement = catalog_exhaustion_replacement(
                state.get("assessment_catalog") or {}, staged_plans, slot_id, qtype,
                str(state.get("difficulty") or "mix"), excluded_task_ids=excluded,
                used_fitb_answer_keys=used_answers,
                active_slot_ids={str(item.get("slot_id")) for _k, _c, item in
                                 _iter_questions(staged_exam.get("questions") or {}) if item.get("slot_id")},
            )
            if replacement is None:
                reason = "No unused or reusable grounded replacement remained for this slot."
                target.update(failure_code="replacement_inventory_exhausted", failure_reason=reason)
                history.append({"model_number": mn, "slot_id": slot_id, "question_id": qid,
                                "attempt": attempt, "repair_action": repair_action,
                                "status": "candidate_generation_failed", "new_failure_reason": reason})
                continue
            from app.online.duplicate_pipeline import _replace_plan_item
            selected_id = str(replacement.get("fitb_entry_id") or replacement.get("task_id") or "")
            memory["replacement_attempt_count"] = int(memory.get("replacement_attempt_count") or 0) + 1
            memory["attempted_candidate_ids"].append(selected_id)
            if not _replace_plan_item(staged_plans, mn, slot_id, replacement):
                memory["blacklisted_candidate_ids"].append(selected_id)
                target.update(failure_code="replacement_plan_write_failed",
                              failure_reason="Replacement could not be applied to the plan slot.")
                continue
            staged_state = {**state, "plans": staged_plans}
            plan_by_slot, source_by_chunk = _plan_context_for_model(staged_state, mn)
            if slot_id in plan_by_slot:
                plan_by_slot[slot_id]["recovery_failure_code"] = target.get("failure_code")
                plan_by_slot[slot_id]["recovery_failure_reason"] = target.get("failure_reason")
                plan_by_slot[slot_id]["recovery_attempt"] = attempt
            repair_action = "replace_fitb_entry" if qtype == "fill_in_the_blank" else "replace_task"

        if _build_by_id(staged_exam.get("questions") or {}).get(qid) is None:
            _put_question(staged_exam, qtype, _placeholder_question(qtype, slot_id, qid))
        generated_ok = False
        outcome_warnings: list[str] = []
        if repair_action in {"regenerate_missing_slot", "replace_fitb_entry", "replace_task"}:
            generated_ok, outcome_warnings = _regenerate_plan_slot(
                staged_state, staged_exam, mn, qid, plan_by_slot, source_by_chunk)
        else:
            verdict = deepcopy(target.get("verdict") or {})
            verdict.setdefault("question_id", qid)
            verdict.setdefault("action", "FIX_QUESTION")
            if str(target.get("validation_role") or "general") == "stem_quantitative":
                from app.llm.factory import create_review_client
                client = create_review_client(domain="STEM", stage="stem_quantitative_repairer")
                outcome = repair_exam(staged_exam, [verdict], client=client,
                                      plan_by_slot=plan_by_slot, source_by_chunk=source_by_chunk,
                                      repair_role="stem_quantitative")
            else:
                outcome = (
                    repair_exam(staged_exam, [verdict], plan_by_slot=plan_by_slot,
                                source_by_chunk=source_by_chunk)
                    if plan_by_slot or source_by_chunk
                    else repair_exam(staged_exam, [verdict])
                )
            generated_ok = qid in set(outcome.get("repaired_ids") or [])
            outcome_warnings = list(outcome.get("warnings") or [])
        warnings.extend(outcome_warnings)
        if not generated_ok:
            reason = "; ".join(outcome_warnings) or "Repairer did not produce a candidate."
            if selected_id:
                memory["blacklisted_candidate_ids"].append(selected_id)
                memory["candidate_failures"].append({"candidate_id": selected_id,
                                                      "reason": reason, "attempt": attempt})
            target.update(failure_code="candidate_generation_failed", failure_reason=reason)
            target.setdefault("history", []).append({"attempt": attempt,
                "repair_action": repair_action, "status": "candidate_generation_failed",
                "new_failure_reason": reason})
            history.append({"model_number": mn, "slot_id": slot_id, "question_id": qid,
                            "attempt": attempt, "repair_action": repair_action,
                            "status": "candidate_generation_failed", "new_failure_reason": reason})
            continue
        candidate_plan = plan_by_slot.get(slot_id) or {}
        candidates.append({"model_number": mn, "slot_id": slot_id, "question_id": qid,
            "question_type": qtype, "attempt": attempt, "repair_action": repair_action,
            "had_original": had_original, "previous_failure_code": target.get("failure_code"),
            "previous_failure_reason": target.get("failure_reason"),
            "candidate_task_id": candidate_plan.get("task_id"),
            "candidate_fitb_entry_id": candidate_plan.get("fitb_entry_id"),
            "exam": staged_exam, "plans": staged_plans})
        if mn in validated_models:
            validated_models.remove(mn)
        target["status"] = "candidate_generated"
        target.setdefault("history", []).append({"attempt": attempt, "repair_action": repair_action,
            "candidate_id": selected_id or candidate_plan.get("fitb_entry_id") or candidate_plan.get("task_id"),
            "status": "candidate_generated", "failure_reason_received": target.get("failure_reason")})

    return {"repair_manifest": manifest, "repair_candidates": candidates,
            "question_repair_attempts": attempts,
            "repair_replacement_memory": memory_by_slot,
            "repair_retry_seeds": retry_seeds,
            "validation_repair_history": history, "pending_revalidation_ids": {},
            "validated_models": validated_models, "warnings": warnings,
            "eval_stats": state.get("eval_stats")}
