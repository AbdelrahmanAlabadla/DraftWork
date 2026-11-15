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
        from app.online.models import normalize_true_false_answer
        return {"statement": q.get("statement"),
                "answer": normalize_true_false_answer(q.get("answer")),
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


def _canonical_changes(
        qtype: str, before: Dict[str, Any], after: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return changes to Repair-owned content, excluding orchestrator metadata.

    Generated questions are enriched between graph nodes with assignment
    fingerprints, revisions, evidence bindings, and catalog metadata.  Those
    fields are verified by the assignment/lifecycle checks and are not part of
    a Repair agent's patch contract.
    """
    old = _canonical_values(qtype, before)
    new = _canonical_values(qtype, after)
    return [
        {
            "field": field,
            "previous_value": deepcopy(old.get(field)),
            "new_value": deepcopy(new.get(field)),
        }
        for field in sorted(set(old) | set(new))
        if old.get(field) != new.get(field)
    ]


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


def _is_why_prompt(value: object) -> bool:
    return bool(_WHY_PROMPT_RE.search(str(value or "").strip()))


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
        from app.online.models import normalize_true_false_answer
        statement = str(item.get("statement") or "").strip()
        if not statement:
            return _verdict(False, False, "FIX_QUESTION", ["statement"],
                            "True/False statement is empty.",
                            "Write a complete, single, unambiguous statement.")
        if normalize_true_false_answer(item.get("answer")) is None:
            return _verdict(True, False, "FIX_ANSWER", ["answer"],
                            "True/False answer is not an unambiguous boolean value.",
                            "Set the answer to the boolean true or false.")
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
