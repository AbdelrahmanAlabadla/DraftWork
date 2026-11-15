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


from ..questions import (
    _iter_questions, _local_verdict, _question_result_text, _solve_system_completion_verdict,
)


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
        "difficulty_status": plan.get("difficulty_status"),
        "difficulty_fallback_reason": plan.get("difficulty_fallback_reason"),
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
