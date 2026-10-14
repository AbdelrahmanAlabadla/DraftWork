from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from app import config
from app.logging_conf import get_logger
from app.online.assessment_catalog import (
    catalog_exhaustion_replacement,
    compatible_task_inventory,
    fitb_answer_keys,
    is_authorized_catalog_exhaustion_reuse,
)
from app.online.duplicate_judge import review_duplicate_candidates

logger = get_logger("DUPLICATE_PIPELINE")


class DuplicateReplacementChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_to_replace: str
    conflicts_with: list[str]
    conflict_reason: str
    replacement_task_id: str | None = None
    replacement_fitb_entry_id: str | None = None
    why_this_candidate_resolves_the_conflict: str
    checked_against_active_slots: bool


def _exam_by_model(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(exam.get("model_number") or 1): exam for exam in state.get("generated_exams") or []}


def _correctness_pass_exam(state: dict[str, Any], exam: dict[str, Any]) -> dict[str, Any]:
    model_number = int(exam.get("model_number") or 1)
    report = next((
        value for value in state.get("validation_reports") or []
        if int(value.get("model_number") or 0) == model_number
    ), None)
    filtered = deepcopy(exam)
    if report is not None:
        passed = {
            str(verdict.get("question_id")) for verdict in report.get("verdicts") or []
            if verdict.get("action") == "PASS" and verdict.get("question_id")
        }
        for section in (filtered.get("questions") or {}).values():
            values = section.get("items") if isinstance(section, dict) else section
            if isinstance(values, list):
                values[:] = [
                    item for item in values
                    if str(item.get("question_id") or "") in passed
                ]
    plan_by_slot = _plan_by_slot(state.get("plans") or [], model_number)
    for section in (filtered.get("questions") or {}).values():
        values = section.get("items") if isinstance(section, dict) else section
        for item in values or []:
            plan = plan_by_slot.get(str(item.get("slot_id") or "")) or {}
            if plan.get("plan_review_status") != "intentional_reuse":
                continue
            item["plan_review_reuse_context"] = {
                "status": "intentional_reuse",
                "suggested_assessment_angle": plan.get("suggested_assessment_angle"),
                "reuse_justification": plan.get("reuse_justification"),
                "required_instance_distinctions": deepcopy(
                    plan.get("required_instance_distinctions") or []
                ),
                "reference_slot_ids": list(plan.get("forced_duplicate_slot_ids") or []),
            }
    return filtered


def _plan_by_slot(plans: list[dict[str, Any]], model_number: int) -> dict[str, dict[str, Any]]:
    for plan in plans:
        if int(plan.get("model_number") or 0) == model_number:
            return {
                str(item.get("slot_id")): item
                for values in (plan.get("items") or {}).values()
                for item in values or [] if item.get("slot_id")
            }
    return {}


def _replace_plan_item(
    plans: list[dict[str, Any]], model_number: int, slot_id: str, replacement: dict[str, Any]
) -> bool:
    for plan in plans:
        if int(plan.get("model_number") or 0) != model_number:
            continue
        for values in (plan.get("items") or {}).values():
            for index, item in enumerate(values or []):
                if str(item.get("slot_id") or "") == slot_id:
                    values[index] = replacement
                    return True
    return False


def _remove_question(exam: dict[str, Any], question_id: str) -> dict[str, Any] | None:
    for qtype, section in (exam.get("questions") or {}).items():
        values = section.get("items") if isinstance(section, dict) else section
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if str(item.get("question_id") or "") == question_id:
                removed = values.pop(index)
                return {"question_type": qtype, "slot_id": removed.get("slot_id"), **removed}
    return None


def _question_slot(exam: dict[str, Any], question_id: str) -> str:
    for section in (exam.get("questions") or {}).values():
        values = section.get("items") if isinstance(section, dict) else section
        for item in values or []:
            if str(item.get("question_id") or "") == question_id:
                return str(item.get("slot_id") or "")
    return ""


_NUMERIC_VARIATION_FIELDS: dict[str, list[str]] = {
    "mcq": ["question", "options", "correct_answer", "solution_certificate"],
    "true_false": ["statement", "answer", "solution_certificate"],
    "equation": ["equation", "solution_steps", "final_answer", "solution_certificate"],
    "word_problem": ["question", "solution_steps", "final_answer", "solution_certificate"],
    "fill_in_the_blank": ["question", "distractors"],
    "definition": ["term", "reference_answer"],
    "why": ["question", "reference_answer"],
    "essay": ["question", "reference_answer", "key_points"],
}


_NUMERIC_CHARACTER_RE = re.compile(r"[0-9\u0660-\u0669\u06f0-\u06f9]")


def _numeric_variation_applicable(
    state: dict[str, Any], exam: dict[str, Any], model_number: int,
    question_id: str, qtype: str, item: dict[str, Any],
) -> bool:
    """Require both quantitative classification and mutable numeric content."""
    if qtype not in {"mcq", "true_false", "fill_in_the_blank", "equation", "word_problem"}:
        return False
    from app.online.validator import _plan_context_for_model, _quantitative_question_ids

    plan_by_slot, _source_by_chunk = _plan_context_for_model(state, model_number)
    if question_id not in set(_quantitative_question_ids(exam, plan_by_slot)):
        return False
    relevant = {
        key: item.get(key) for key in (
            "question", "statement", "equation", "options", "correct_answer",
            "answer", "answers", "solution_steps", "final_answer", "distractors",
        ) if key in item
    }
    return bool(_NUMERIC_CHARACTER_RE.search(json.dumps(relevant, ensure_ascii=False)))


def _numeric_variation_fallback(
    state: dict[str, Any], exam: dict[str, Any], model_number: int,
    question_id: str, conflict_question_ids: list[str],
    conflict_edges: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Produce attempt three by changing dependent quantitative content only.

    The normal repair contract owns schema normalization, immutable question IDs,
    source evidence, and answer/certificate consistency.  This wrapper supplies
    both sides of the duplicate conflict and authorizes only question content.
    """
    from app.llm.factory import create_llm_client, create_review_client
    from app.online.validator import (
        _build_by_id, _plan_context_for_model, _quantitative_question_ids, repair_exam,
    )

    isolated = deepcopy(exam)
    by_id = _build_by_id(isolated.get("questions") or {})
    entry = by_id.get(question_id)
    if entry is None:
        return None, [f"numeric fallback: question {question_id} was not found"]
    qtype, _container, item = entry
    conflicts = [
        deepcopy(by_id[qid][2]) for qid in conflict_question_ids if qid in by_id
    ]
    plan_by_slot, source_by_chunk = _plan_context_for_model(state, model_number)
    plan = plan_by_slot.get(str(item.get("slot_id") or "")) or {}
    fields = _NUMERIC_VARIATION_FIELDS.get(qtype, [])
    if not fields or not conflicts:
        return None, ["numeric fallback: no supported fields or conflict question"]
    details = {
        "mode": "numeric_variation_fallback",
        "duplicate_question": deepcopy(item),
        "conflicting_questions": conflicts,
        "duplicate_judge_conflict_details": deepcopy(conflict_edges),
        "preserve": {
            "question_id": question_id,
            "question_type": qtype,
            "difficulty": plan.get("actual_difficulty") or plan.get("difficulty"),
            "idea_to_test": plan.get("idea_to_test"),
            "reasoning_mode": plan.get("reasoning_mode"),
            "concept": plan.get("concept"),
            "task_operation": plan.get("task_operation"),
            "target": plan.get("target"),
            "expected_answer_type": plan.get("expected_work_product")
            or plan.get("student_work_product"),
            "evidence_chunk_ids": plan.get("support_chunk_ids")
            or plan.get("evidence_chunk_ids") or [plan.get("source_chunk_id")],
        },
    }
    verdict = {
        "question_id": question_id,
        "action": "FIX_QUESTION_AND_ANSWER",
        "failure_code": "duplicate_numeric_variation_required",
        "fields_to_fix": fields,
        "reason": json.dumps(details, ensure_ascii=False),
        "expected_fix": (
            "Compare both questions and change the values, coefficients, quantities, "
            "constants, or answer values that cause overlap. Recalculate every dependent "
            "answer, option, distractor, unit, explanation, solution step, and intermediate "
            "result. Preserve the listed semantic metadata and difficulty. Avoid blind "
            "number substitution, invalid or unrealistic values, unintended fractions, "
            "no/infinite-solution cases, and multiple-correct-answer MCQs."
        ),
    }
    quantitative = question_id in set(_quantitative_question_ids(isolated, plan_by_slot))
    domain = str(
        (state.get("assessment_profile") or {}).get("domain")
        or state.get("domain_type") or "STEM"
    )
    client = (
        create_review_client(domain=domain, stage="stem_quantitative_repairer")
        if quantitative else create_llm_client(domain=domain, stage="general_repairer")
    )
    result = repair_exam(
        isolated, [verdict], client=client, plan_by_slot=plan_by_slot,
        source_by_chunk=source_by_chunk,
        repair_role="stem_quantitative" if quantitative else "general",
    )
    warnings = list(result.get("warnings") or [])
    if question_id not in set(result.get("repaired_ids") or []):
        return None, warnings or ["numeric fallback: repair was rejected"]
    repaired = (_build_by_id(isolated.get("questions") or {}).get(question_id) or (None, None, None))[2]
    return deepcopy(repaired) if isinstance(repaired, dict) else None, warnings


def _active_slot_ids(exam: dict[str, Any]) -> set[str]:
    return {
        str(item.get("slot_id"))
        for section in (exam.get("questions") or {}).values()
        for item in (section.get("items") if isinstance(section, dict) else section) or []
        if item.get("slot_id")
    }


def _slot_attempt_key(model_number: int, slot_id: str) -> str:
    return f"{model_number}:{slot_id}"


def _failed_task_exclusions(
    history: list[dict[str, Any]], model_number: int, slot_id: str,
) -> set[str]:
    """Exclude only tasks with an observed generation/precommit failure."""
    return {
        str(event.get("fitb_entry_id") or event.get("task_id") or "")
        for event in history
        if int(event.get("model_number") or 0) == model_number
        and str(event.get("slot_id") or "") == slot_id
        and event.get("status") in {
            "generation_failed", "fallback_precommit_rejected", "validation_failed",
            "quantitative_validation_failed", "duplicate_after_replacement",
        }
        and (event.get("fitb_entry_id") or event.get("task_id"))
    }


def _identity(item: dict[str, Any]) -> str:
    return str(item.get("fitb_entry_id") or item.get("task_id") or "")


def _compact_metadata(item: dict[str, Any]) -> dict[str, Any]:
    compact = {key: item.get(key) for key in (
        "slot_id", "task_id", "fitb_entry_id", "task_family_id", "coverage_id",
        "concept", "concept_id", "task_operation", "operation", "target",
        "reasoning_mode", "expected_work_product", "relationship_id",
        "normalized_answer", "clue_basis", "evidence_chunk_id",
    ) if item.get(key) not in (None, "", [])}
    work_product = item.get("expected_work_product") or item.get("student_work_product")
    if work_product:
        compact["expected_work_product"] = work_product
    return compact


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _obvious_semantic_collision(candidate: dict[str, Any], active: dict[str, Any]) -> bool:
    if candidate.get("fitb_entry_id"):
        return bool(_norm(candidate.get("normalized_answer") or candidate.get("fitb_answer"))
                    and _norm(candidate.get("normalized_answer") or candidate.get("fitb_answer")) ==
                    _norm(active.get("normalized_answer") or active.get("fitb_answer")))
    values = [
        (_norm(candidate.get("concept") or candidate.get("concept_id")),
         _norm(active.get("concept") or active.get("concept_id"))),
        (_norm(candidate.get("task_operation") or candidate.get("operation")),
         _norm(active.get("task_operation") or active.get("operation"))),
        (_norm(candidate.get("target")), _norm(active.get("target"))),
    ]
    return all(left and left == right for left, right in values)


def _candidate_pool(
    catalog: dict[str, Any], plans: list[dict[str, Any]], slot_id: str, qtype: str,
    difficulty: str, active_slots: set[str], excluded: set[str], limit: int = 12,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen = set(excluded)
    for _ in range(limit):
        candidate = catalog_exhaustion_replacement(
            catalog, plans, slot_id, qtype, difficulty,
            excluded_task_ids=seen, active_slot_ids=active_slots,
        )
        if not candidate or not _identity(candidate) or _identity(candidate) in seen:
            break
        candidates.append(candidate)
        seen.add(_identity(candidate))
    if not candidates and difficulty in {"easy", "medium", "hard"}:
        for _ in range(limit):
            candidate = catalog_exhaustion_replacement(
                catalog, plans, slot_id, qtype, "mix",
                excluded_task_ids=seen, active_slot_ids=active_slots,
            )
            if not candidate or not _identity(candidate) or _identity(candidate) in seen:
                break
            candidate["requested_difficulty"] = difficulty
            candidate["actual_difficulty"] = candidate.get("difficulty")
            candidate["selected_replacement_task_id"] = candidate.get("task_id")
            candidates.append(candidate)
            seen.add(_identity(candidate))
    return candidates


def _select_replacement_candidate(
    *, slot_id: str, qtype: str, conflict_slots: list[str], conflict_edges: list[dict[str, Any]],
    candidates: list[dict[str, Any]], active_items: list[dict[str, Any]],
    blacklist: set[str], previous_failures: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    active_without_slot = [item for item in active_items if str(item.get("slot_id")) != slot_id]
    usage: dict[str, int] = {}
    for item in active_without_slot:
        usage[_identity(item)] = usage.get(_identity(item), 0) + 1
    conflicting_ids = {
        _identity(item) for item in active_without_slot
        if str(item.get("slot_id") or "") in conflict_slots
    }
    safe = [candidate for candidate in candidates
            if _identity(candidate) not in blacklist | conflicting_ids
            and not any(_obvious_semantic_collision(candidate, item) for item in active_without_slot)]
    def rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
        collision_count = sum(_obvious_semantic_collision(candidate, item) for item in active_without_slot)
        represented = lambda key: sum(_norm(candidate.get(key)) == _norm(item.get(key))
                                      for item in active_without_slot if candidate.get(key))
        return (collision_count, int(usage.get(_identity(candidate), 0) > 0),
                usage.get(_identity(candidate), 0), represented("coverage_id"),
                represented("concept_id"), represented("task_operation"), represented("target"),
                represented("task_family_id"), represented("reasoning_mode"), _identity(candidate))
    safe.sort(key=rank)
    context = {
        "slot_to_replace": slot_id, "conflicts_with": conflict_slots,
        "conflict": conflict_edges, "active_slots": [_compact_metadata(item) for item in active_without_slot],
        "candidates": [{**_compact_metadata(item), "used_count": usage.get(_identity(item), 0)}
                       for item in safe],
        "blacklisted_candidate_ids": sorted(blacklist),
        "previous_failures": previous_failures,
    }
    if not safe:
        return None, context
    try:
        from app.llm.factory import create_duplicate_replacement_client
        from app.llm.client import request_structured
        raw = request_structured(
            create_duplicate_replacement_client(),
            "Choose the candidate that resolves the stated conflict and is least likely to conflict "
            "with compact active-slot metadata. Prefer unused and less-used only after semantic safety. "
            "Return JSON matching the schema.\n" + json.dumps(context, ensure_ascii=False),
            response_model=DuplicateReplacementChoice, schema_name="duplicate_replacement_choice",
            agent="duplicate_replacement", operation="select_replacement", item_count=len(safe),
            temperature=0.0, max_tokens=1800,
        )
        chosen_id = str((raw or {}).get("replacement_fitb_entry_id") or
                        (raw or {}).get("replacement_task_id") or "")
        chosen = next((item for item in safe if _identity(item) == chosen_id), None)
        if chosen and (raw or {}).get("slot_to_replace") == slot_id and (raw or {}).get("checked_against_active_slots") is True:
            return chosen, {**context, "agent_choice": raw}
    except Exception as exc:
        logger.warning("Duplicate replacement agent failed; deterministic fallback used | %s", exc)
        context["agent_error"] = str(exc)
    return safe[0], {**context, "selection_source": "deterministic_fallback"}


def _used_fitb_answer_keys(
    exam: dict[str, Any], plans: list[dict[str, Any]], model_number: int,
    excluding_slot_id: str,
) -> set[str]:
    plan_by_slot = _plan_by_slot(plans, model_number)
    keys: set[str] = set()
    section = (exam.get("questions") or {}).get("fill_in_the_blank") or {}
    for item in section.get("items") or []:
        slot_id = str(item.get("slot_id") or "")
        if slot_id == excluding_slot_id:
            continue
        keys.update(fitb_answer_keys(item))
        keys.update(fitb_answer_keys(plan_by_slot.get(slot_id) or {}))
    return keys


def _forced_catalog_reuse_pairs(
    exam: dict[str, Any], plans: list[dict[str, Any]], model_number: int
) -> set[frozenset[str]]:
    """Pairs that must reach the Judge even when embeddings do not shortlist them."""
    slot_to_qid: dict[str, str] = {}
    for section in (exam.get("questions") or {}).values():
        values = section.get("items") if isinstance(section, dict) else section
        for item in values or []:
            if item.get("slot_id") and item.get("question_id"):
                slot_to_qid[str(item["slot_id"])] = str(item["question_id"])
    plan_by_slot = _plan_by_slot(plans, model_number)
    forced: set[frozenset[str]] = set()
    for slot_id, item in plan_by_slot.items():
        replacement_stage = str(item.get("catalog_replacement_stage") or "")
        if not (
            is_authorized_catalog_exhaustion_reuse(item)
            or item.get("fitb_entry_reused") is True
            or replacement_stage.startswith("unused_task_")
            or replacement_stage == "unused_fitb_entry"
            or replacement_stage == "plan_review_unused_task"
            or replacement_stage == "regenerated_original_task_after_duplicate"
            or replacement_stage == "regenerated_original_fitb_entry"
        ):
            continue
        qid = slot_to_qid.get(slot_id)
        if not qid:
            continue
        task_id = str(item.get("fitb_entry_id") or item.get("task_id") or "")
        family_id = str(item.get("task_family_id") or "")
        related_slots = {
            other_slot for other_slot, other in plan_by_slot.items()
            if other_slot != slot_id and (
                (task_id and str(
                    other.get("fitb_entry_id") or other.get("task_id") or ""
                ) == task_id)
                or (family_id and str(other.get("task_family_id") or "") == family_id)
            )
        }
        related_slots.update(str(value) for value in item.get("forced_duplicate_slot_ids") or [])
        related_qids = {
            slot_to_qid.get(other_slot) for other_slot in related_slots
            if slot_to_qid.get(other_slot)
        }
        related_qids.update(str(value) for value in item.get("forced_duplicate_question_ids") or [])
        for other_qid in related_qids:
            if other_qid and other_qid != qid:
                forced.add(frozenset((qid, other_qid)))
    return forced


def scan_exam_duplicates(state: dict[str, Any]) -> dict[str, Any]:
    reports = list(state.get("duplicate_reports") or [])
    distributions = list(state.get("duplicate_similarity_distributions") or [])
    pending: list[dict[str, Any]] = []
    invalid = list(state.get("invalid_questions") or [])
    repair_rounds = int(state.get("duplicate_repair_rounds") or 0)
    slot_attempts = dict(state.get("duplicate_replacement_attempts") or {})
    warnings = list(state.get("warnings") or [])
    blacklists = {key: list(value) for key, value in
                  (state.get("duplicate_candidate_blacklist") or {}).items()}
    failure_memory = {key: list(value) for key, value in
                      (state.get("duplicate_replacement_failures") or {}).items()}
    history = list(state.get("duplicate_repair_history") or [])
    scan_index = int(state.get("duplicate_scan_count") or 0) + 1
    fact_cache = dict(state.get("duplicate_fact_cache") or {})
    verification_pending = bool(state.get("duplicate_verification_pending"))
    verification_attempt = int(
        state.get("duplicate_verification_attempt") or repair_rounds + 1
    )
    if repair_rounds >= config.MAX_DUPLICATE_REPAIR_ROUNDS and not verification_pending:
        unresolved = list(state.get("unresolved_duplicate_conflicts") or [])
        unresolved.extend(deepcopy(state.get("pending_duplicate_groups") or []))
        if not unresolved and reports:
            unresolved.extend(deepcopy(reports[-1].get("duplicate_groups") or []))
        warnings.append(
            "Duplicate repair budget exhausted; stopped before another Duplicate Judge call."
        )
        return {
            "duplicate_reports": reports,
            "duplicate_similarity_distributions": distributions,
            "pending_duplicate_groups": [],
            "duplicate_scan_count": int(state.get("duplicate_scan_count") or 0),
            "invalid_questions": invalid, "warnings": warnings,
            "generated_exams": list(_exam_by_model(state).values()),
            "duplicate_fact_cache": fact_cache,
            "unresolved_duplicate_conflicts": unresolved,
            "duplicate_verification_pending": False,
            "duplicate_verification_attempt": 0,
        }
    stored_domain = str((state.get("assessment_profile") or {}).get("domain") or "")
    review_client = None
    if stored_domain:
        from app.llm.factory import create_review_client
        review_client = create_review_client(
            domain=stored_domain, stage="duplicate_judge"
        )

    def run_review(filtered_exam: dict[str, Any], forced_pairs: set[frozenset[str]]):
        kwargs: dict[str, Any] = {
            "fact_cache": fact_cache,
            "forced_question_pairs": forced_pairs,
        }
        if review_client is not None:
            kwargs["client"] = review_client
        try:
            return review_duplicate_candidates(filtered_exam, **kwargs)
        except TypeError as exc:
            # Compatibility for narrow test/dry-run substitutes implementing the
            # historical one-positional-argument call contract.
            if "unexpected keyword argument" not in str(exc):
                raise
            return review_duplicate_candidates(filtered_exam)

    unresolved: list[dict[str, Any]] = []
    final_verification = (
        verification_pending
        and verification_attempt >= config.MAX_DUPLICATE_REPAIR_ROUNDS
    )
    for model_number, exam in _exam_by_model(state).items():
        filtered_exam = _correctness_pass_exam(state, exam)
        # Attempt-three numeric variants have already passed the ordinary /
        # quantitative validator and are accepted without another duplicate
        # judge call by contract. Exclude only those IDs from this judge input.
        exempt_ids = set(state.get("duplicate_judge_exempt_ids") or [])
        if exempt_ids:
            for section in (filtered_exam.get("questions") or {}).values():
                values = section.get("items") if isinstance(section, dict) else section
                if isinstance(values, list):
                    values[:] = [
                        value for value in values
                        if str(value.get("question_id") or "") not in exempt_ids
                    ]
        forced_pairs = _forced_catalog_reuse_pairs(
            filtered_exam, state.get("plans") or [], model_number
        )
        report = run_review(filtered_exam, forced_pairs)
        fact_cache = dict(report.pop("fact_cache", fact_cache))
        report_record = {"model_number": model_number, "scan_index": scan_index, **report}
        reports.append(report_record)
        distributions.append({
            "model_number": model_number, "scan_index": scan_index,
            **report.get("similarity_distribution", {}),
        })
        groups = list(report.get("duplicate_groups") or [])
        if not groups:
            continue
        for group in groups:
            for qid in group.get("question_ids") or []:
                slot_id = _question_slot(exam, qid)
                latest = next((event for event in reversed(history)
                               if int(event.get("model_number") or 0) == model_number
                               and str(event.get("slot_id") or "") == slot_id
                               and event.get("status") == "replaced"), None)
                if latest:
                    failed_id = str(latest.get("fitb_entry_id") or latest.get("task_id") or "")
                    key = _slot_attempt_key(model_number, slot_id)
                    if failed_id:
                        blacklists.setdefault(key, []).append(failed_id)
                        failure_memory.setdefault(key, []).append({
                            "candidate_id": failed_id,
                            "reason": "duplicate_after_replacement",
                            "conflict_edges": deepcopy(group.get("edges") or []),
                        })
        if final_verification:
            unresolved.extend({"model_number": model_number, **deepcopy(group)} for group in groups)
            continue
        replace_ids = list(dict.fromkeys(
            qid for group in groups for qid in group.get("replace_question_ids") or []
        ))
        eligible_ids: set[str] = set()
        exhausted: dict[str, str] = {}
        for qid in replace_ids:
            slot_id = _question_slot(exam, qid)
            qtype = next((
                name for name, section in (exam.get("questions") or {}).items()
                if any(
                    str(value.get("question_id") or "") == qid
                    for value in (section.get("items") if isinstance(section, dict) else section) or []
                )
            ), "")
            exclusions = _failed_task_exclusions(
                list(state.get("duplicate_repair_history") or []), model_number, slot_id
            )
            inventory = compatible_task_inventory(
                state.get("assessment_catalog") or {}, state.get("plans") or [], qtype,
                str(state.get("difficulty") or "mix"), excluding_slot_id=slot_id,
                excluded_task_ids=exclusions,
                active_slot_ids=_active_slot_ids(exam),
            ) if qtype else {}
            current_plan = _plan_by_slot(
                state.get("plans") or [], model_number
            ).get(slot_id) or {}
            current_task_id = str(
                current_plan.get("fitb_entry_id") or current_plan.get("task_id") or ""
            )
            has_eligible_unused = any(
                str(task_id) != current_task_id
                for task_id in inventory.get("eligible_unused_after_exclusions_task_ids") or []
            )
            if int(slot_attempts.get(_slot_attempt_key(model_number, slot_id)) or 0) >= (
                config.MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT
            ):
                exhausted[qid] = "slot_attempt_limit"
            else:
                eligible_ids.add(qid)

        for group in groups:
            eligible = [
                qid for qid in group.get("replace_question_ids") or [] if qid in eligible_ids
            ]
            if eligible:
                pending.append({
                    "model_number": model_number,
                    **group,
                    "replace_question_ids": eligible,
                })

        for qid, exhausted_by in exhausted.items():
            slot_id = _question_slot(exam, qid)
            unresolved.extend(
                {"model_number": model_number, **deepcopy(group)}
                for group in groups if qid in (group.get("question_ids") or [])
            )
            history.append({
                "model_number": model_number, "question_id": qid,
                "slot_id": slot_id, "status": "duplicate_repair_exhausted_kept",
                "exhausted_by": exhausted_by,
            })
        fitb = exam.get("questions", {}).get("fill_in_the_blank")
        if isinstance(fitb, dict):
            from app.online.validator import _fitb_word_bank_valid, _rebuild_fitb_word_bank
            if not _fitb_word_bank_valid(fitb):
                _rebuild_fitb_word_bank(fitb)
        if exhausted:
            warnings.append(
                f"Model {model_number}: retained {len(exhausted)} valid slot(s) with unresolved "
                "similarity after all three duplicate-repair attempts."
            )
    return {
        "duplicate_reports": reports,
        "duplicate_similarity_distributions": distributions,
        "pending_duplicate_groups": pending,
        "duplicate_scan_count": scan_index,
        "invalid_questions": invalid,
        "warnings": warnings,
        "generated_exams": list(_exam_by_model(state).values()),
        "duplicate_fact_cache": fact_cache,
        "duplicate_candidate_blacklist": {
            key: list(dict.fromkeys(values)) for key, values in blacklists.items()
        },
        "duplicate_replacement_failures": failure_memory,
        "duplicate_repair_history": history,
        "duplicate_repair_rounds": (
            verification_attempt if verification_pending else repair_rounds
        ),
        "duplicate_verification_pending": False,
        "duplicate_verification_attempt": 0,
        "unresolved_duplicate_conflicts": unresolved,
    }


def _generate_reserved(
    state: dict[str, Any], plans: list[dict[str, Any]], model_number: int,
    question_id: str, exam: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, str | None, list[str]]:
    from app.online.validator import _build_by_id, _plan_context_for_model, _regenerate_plan_slot

    isolated_exam = deepcopy(exam)
    isolated_state = {**state, "plans": deepcopy(plans)}
    plan_by_slot, source_by_chunk = _plan_context_for_model(isolated_state, model_number)
    success, warnings = _regenerate_plan_slot(
        isolated_state, isolated_exam, model_number, question_id, plan_by_slot, source_by_chunk
    )
    if not success:
        return question_id, None, None, warnings
    entry = _build_by_id(isolated_exam.get("questions") or {}).get(question_id)
    if entry is None:
        return question_id, None, None, [*warnings, "replacement disappeared after generation"]
    qtype, _container, item = entry
    return question_id, deepcopy(item), qtype, warnings


def _precommit_fallback_candidate(
    state: dict[str, Any], plans: list[dict[str, Any]], model_number: int,
    question_id: str, exam: dict[str, Any], generated: dict[str, Any],
    fact_cache: dict[str, dict[str, str]],
) -> tuple[bool, list[str], dict[str, dict[str, str]]]:
    """Correctness-check and duplicate-check a fallback before mutating the live exam."""
    from app.llm.factory import create_review_client
    from app.online.validator import (
        _build_by_id, _merge_role_reports, _plan_context_for_model,
        _quantitative_question_ids, _validate_with_context,
    )

    candidate_exam = deepcopy(exam)
    entry = _build_by_id(candidate_exam.get("questions") or {}).get(question_id)
    if entry is None:
        return False, ["fallback precommit: target question disappeared"], fact_cache
    entry[2].clear()
    entry[2].update(deepcopy(generated))
    context_state = {**state, "plans": plans}
    plan_by_slot, source_by_chunk = _plan_context_for_model(context_state, model_number)
    domain = str((state.get("assessment_profile") or {}).get("domain") or "").upper()
    try:
        general_client = create_review_client(domain=domain, stage="general_validator")
        general = _validate_with_context(
            candidate_exam, [question_id], plan_by_slot, source_by_chunk,
            general_client, "general",
        )
        correctness = general
        if domain == "STEM" and question_id in set(
            _quantitative_question_ids(candidate_exam, plan_by_slot)
        ):
            quantitative_client = create_review_client(
                domain=domain, stage="stem_quantitative_validator"
            )
            quantitative = _validate_with_context(
                candidate_exam, [question_id], plan_by_slot, source_by_chunk,
                quantitative_client, "stem_quantitative",
            )
            correctness = _merge_role_reports(general, quantitative)
    except Exception as exc:
        logger.exception(
            "Fallback correctness precommit failed | model=%d | question_id=%s",
            model_number, question_id,
        )
        return False, [f"fallback precommit correctness check failed: {exc}"], fact_cache
    if not correctness.get("all_pass"):
        reasons = [
            str(verdict.get("reason") or verdict.get("failure_code") or "invalid")
            for verdict in correctness.get("verdicts") or []
            if verdict.get("action") != "PASS"
        ]
        return False, [f"fallback precommit correctness rejected: {'; '.join(reasons)}"], fact_cache

    forced = _forced_catalog_reuse_pairs(candidate_exam, plans, model_number)
    try:
        duplicate_client = create_review_client(domain=domain, stage="duplicate_judge")
        duplicate_report = review_duplicate_candidates(
            candidate_exam, client=duplicate_client, fact_cache=fact_cache,
            forced_question_pairs=forced,
        )
    except Exception as exc:
        logger.exception(
            "Fallback duplicate precommit failed | model=%d | question_id=%s",
            model_number, question_id,
        )
        return False, [f"fallback precommit duplicate check failed: {exc}"], fact_cache
    updated_cache = dict(duplicate_report.get("fact_cache") or fact_cache)
    conflicts = [
        verdict for verdict in duplicate_report.get("pair_verdicts") or []
        if verdict.get("classification") in {"duplicate", "excessive_overlap"}
        and any(
            question_id == pair.get(side, {}).get("question_id")
            for pair in duplicate_report.get("candidate_pairs") or []
            if pair.get("pair_id") == verdict.get("pair_id")
            for side in ("left", "right")
        )
    ]
    if conflicts:
        labels = ", ".join(
            f"{value.get('pair_id')}={value.get('classification')}" for value in conflicts
        )
        return False, [f"fallback precommit duplicate rejected: {labels}"], updated_cache
    return True, [], updated_cache


def repair_duplicate_groups(state: dict[str, Any]) -> dict[str, Any]:
    from app.online.validator import _build_by_id, _fitb_word_bank_valid, _rebuild_fitb_word_bank

    plans = deepcopy(state.get("plans") or [])
    exams = _exam_by_model(state)
    completed_rounds = int(state.get("duplicate_repair_rounds") or 0)
    repair_round = completed_rounds + 1
    slot_attempts = dict(state.get("duplicate_replacement_attempts") or {})
    pending_revalidation = dict(state.get("pending_revalidation_ids") or {})
    validated = list(state.get("validated_models") or [])
    history = list(state.get("duplicate_repair_history") or [])
    warnings = list(state.get("warnings") or [])
    catalog = state.get("assessment_catalog") or {}
    fact_cache = dict(state.get("duplicate_fact_cache") or {})
    diagnostics = list(state.get("developer_diagnostics") or [])
    blacklists = {key: list(value) for key, value in
                  (state.get("duplicate_candidate_blacklist") or {}).items()}
    failure_memory = {key: list(value) for key, value in
                      (state.get("duplicate_replacement_failures") or {}).items()}
    verification_required = False
    numeric_repaired_ids: list[str] = []
    judge_exempt_ids = list(state.get("duplicate_judge_exempt_ids") or [])
    numeric_fallback_attempted = False
    finalization_pending = False
    unresolved_conflicts = list(state.get("unresolved_duplicate_conflicts") or [])
    for report in state.get("validation_reports") or []:
        report_model = int(report.get("model_number") or 0)
        for verdict in report.get("verdicts") or []:
            if verdict.get("action") == "PASS":
                continue
            qid = str(verdict.get("question_id") or "")
            slot_id = _question_slot(exams.get(report_model) or {}, qid)
            latest = next((event for event in reversed(history)
                           if int(event.get("model_number") or 0) == report_model
                           and str(event.get("slot_id") or "") == slot_id
                           and event.get("status") == "replaced"), None)
            if latest:
                candidate_id = str(latest.get("fitb_entry_id") or latest.get("task_id") or "")
                key = _slot_attempt_key(report_model, slot_id)
                if candidate_id:
                    blacklists.setdefault(key, []).append(candidate_id)
                    failure_memory.setdefault(key, []).append({
                        "candidate_id": candidate_id,
                        "reason": str(verdict.get("failure_code") or verdict.get("reason") or "validation_failed"),
                    })
    groups_by_model: dict[int, list[dict[str, Any]]] = {}
    for group in state.get("pending_duplicate_groups") or []:
        groups_by_model.setdefault(int(group.get("model_number") or 1), []).append(group)

    for model_number, groups in groups_by_model.items():
        exam = exams.get(model_number)
        if exam is None:
            continue
        reserved: list[dict[str, Any]] = []
        by_id = _build_by_id(exam.get("questions") or {})
        replace_ids = list(dict.fromkeys(
            qid for group in groups for qid in group.get("replace_question_ids") or []
        ))
        # Reservation is deliberately sequential and atomic. Generation starts only
        # after every candidate has seen the tasks reserved before it.
        for qid in replace_ids:
            entry = by_id.get(qid)
            if entry is None:
                history.append({"model_number": model_number, "question_id": qid,
                                "repair_round": repair_round, "status": "question_not_found"})
                continue
            qtype, _container, item = entry
            slot_id = str(item.get("slot_id") or "")
            attempt_key = _slot_attempt_key(model_number, slot_id)
            if int(slot_attempts.get(attempt_key) or 0) >= config.MAX_DUPLICATE_REPLACEMENT_ATTEMPTS_PER_SLOT:
                history.append({"model_number": model_number, "question_id": qid,
                                "slot_id": slot_id, "repair_round": repair_round,
                                "status": "slot_attempt_limit"})
                continue
            slot_attempt = int(slot_attempts.get(attempt_key) or 0) + 1
            slot_attempts[attempt_key] = slot_attempt
            related_question_ids = list(dict.fromkeys(
                other_qid
                for group in groups if qid in (group.get("replace_question_ids") or [])
                for other_qid in (group.get("question_ids") or [])
                if str(other_qid) and str(other_qid) != qid
            ))
            related_slot_ids = list(dict.fromkeys(
                _question_slot(exam, other_qid) for other_qid in related_question_ids
                if _question_slot(exam, other_qid)
            ))
            old_plan_item = deepcopy(_plan_by_slot(plans, model_number).get(slot_id) or {})
            failed_replacement_task_ids = _failed_task_exclusions(
                history, model_number, slot_id
            )
            blacklist = set(blacklists.get(attempt_key) or []) | failed_replacement_task_ids
            blacklist.add(_identity(old_plan_item))
            active_plan_items = list(_plan_by_slot(plans, model_number).values())
            conflict_edges = [
                edge for group in groups if qid in (group.get("replace_question_ids") or [])
                for edge in group.get("edges") or []
                if qid in {edge.get("left"), edge.get("right")}
            ]
            if slot_attempt == 3:
                if not _numeric_variation_applicable(
                    state, exam, model_number, qid, qtype, item
                ):
                    finalization_pending = True
                    unresolved_conflicts.extend(
                        {"model_number": model_number, **deepcopy(group)}
                        for group in groups if qid in (group.get("question_ids") or [])
                    )
                    history.append({
                        "model_number": model_number, "question_id": qid,
                        "slot_id": slot_id, "repair_round": repair_round,
                        "slot_attempt": slot_attempt,
                        "duplicate_repair_mode": "numeric_variation_fallback",
                        "status": "numeric_variation_not_applicable_kept",
                        "conflict_question_ids": related_question_ids,
                    })
                    warnings.append(
                        f"Model {model_number}: retained {qid}; numeric duplicate variation "
                        "was not applicable after two semantic replacement attempts."
                    )
                    continue
                numeric_fallback_attempted = True
                repaired, numeric_warnings = _numeric_variation_fallback(
                    state, exam, model_number, qid, related_question_ids, conflict_edges
                )
                warnings.extend(numeric_warnings)
                if repaired is None:
                    history.append({
                        "model_number": model_number, "question_id": qid,
                        "slot_id": slot_id, "repair_round": repair_round,
                        "slot_attempt": slot_attempt,
                        "duplicate_repair_mode": "numeric_variation_fallback",
                        "status": "numeric_variation_failed",
                        "failure_reason": "; ".join(numeric_warnings),
                    })
                    continue
                item.clear()
                item.update(repaired)
                item.update({
                    "duplicate_repair_attempt": 3,
                    "duplicate_repair_mode": "numeric_variation_fallback",
                    "duplicate_conflict_with": (
                        related_question_ids[0] if len(related_question_ids) == 1
                        else related_question_ids
                    ),
                    "numeric_variant": True,
                    "duplicate_recheck_required": False,
                })
                numeric_repaired_ids.append(qid)
                judge_exempt_ids.append(qid)
                history.append({
                    "model_number": model_number, "question_id": qid,
                    "slot_id": slot_id, "repair_round": repair_round,
                    "slot_attempt": slot_attempt,
                    "duplicate_repair_mode": "numeric_variation_fallback",
                    "conflict_question_ids": related_question_ids,
                    "status": "numeric_variation_generated",
                })
                continue
            candidates = _candidate_pool(
                catalog, plans, slot_id, qtype, str(state.get("difficulty") or "mix"),
                _active_slot_ids(exam), blacklist,
            )
            replacement, selector_context = _select_replacement_candidate(
                slot_id=slot_id, qtype=qtype, conflict_slots=related_slot_ids,
                conflict_edges=conflict_edges, candidates=candidates,
                active_items=active_plan_items, blacklist=blacklist,
                previous_failures=failure_memory.get(attempt_key) or [],
            )
            if replacement:
                replacement["selected_replacement_task_id"] = replacement.get("task_id")
                replacement["task_was_previously_used"] = bool(
                    replacement.get("task_reused") or replacement.get("fitb_entry_reused")
                )
                replacement["forced_duplicate_slot_ids"] = related_slot_ids
                replacement["forced_duplicate_question_ids"] = related_question_ids
                replacement["duplicate_conflict_context"] = conflict_edges
                replacement["duplicate_replacement_selector_context"] = selector_context
                agent_choice = selector_context.get("agent_choice") or {}
                replacement["suggested_assessment_angle"] = (
                    agent_choice.get("why_this_candidate_resolves_the_conflict")
                    or "Assess the selected task without repeating the conflicting operation, target, or work product."
                )
                required = list(replacement.get("required_instance_distinctions") or [])
                required.extend(
                    distinction for edge in conflict_edges
                    for distinction in edge.get("what_must_be_different") or []
                )
                replacement["required_instance_distinctions"] = list(dict.fromkeys(required))
            if replacement and (
                replacement.get("task_reused") or replacement.get("fitb_entry_reused")
            ):
                reused_task_id = str(
                    replacement.get("fitb_entry_id") or replacement.get("task_id") or ""
                )
                plan_items = _plan_by_slot(plans, model_number)
                previous_task_questions = []

                def add_context(
                    context_qid: str, context_slot: str, context_plan: dict[str, Any],
                    context_item: dict[str, Any], role: str,
                ) -> None:
                    if any(
                        str(existing.get("question_id") or "") == context_qid
                        for existing in previous_task_questions
                    ):
                        return
                    previous_task_questions.append({
                        "context_role": role,
                        "question_id": context_qid,
                        "slot_id": context_slot,
                        "task_id": context_plan.get("task_id"),
                        "fitb_entry_id": context_plan.get("fitb_entry_id"),
                        "question_text": context_item.get("question")
                        or context_item.get("statement") or context_item.get("term")
                        or context_item.get("equation"),
                        "idea_to_test": context_plan.get("idea_to_test"),
                        "task_operation": context_plan.get("task_operation"),
                        "target": context_plan.get("target"),
                        "reasoning_mode": context_plan.get("reasoning_mode"),
                        "scenario_family": context_plan.get("scenario_family"),
                        "answer_pattern": context_item.get("correct_answer")
                        or context_item.get("answers")
                        or context_item.get("reference_answer")
                        or context_item.get("final_answer"),
                    })

                for previous_slot, previous_plan in plan_items.items():
                    if previous_slot == slot_id or str(
                        previous_plan.get("fitb_entry_id")
                        or previous_plan.get("task_id") or ""
                    ) != reused_task_id:
                        continue
                    previous_qid = next((
                        candidate_id for candidate_id, entry in by_id.items()
                        if str(entry[2].get("slot_id") or "") == previous_slot
                    ), "")
                    previous_item = (by_id.get(previous_qid) or (None, None, {}))[2]
                    add_context(
                        previous_qid, previous_slot, previous_plan, previous_item,
                        "previous_question_from_reused_task",
                    )
                if replacement.get("catalog_replacement_stage") == (
                    "regenerated_original_task_after_duplicate"
                ):
                    add_context(
                        qid, slot_id, old_plan_item, item,
                        "rejected_duplicate_question",
                    )
                    for conflict_qid in related_question_ids:
                        conflict_entry = by_id.get(conflict_qid) or (None, None, {})
                        conflict_item = conflict_entry[2]
                        conflict_slot = str(conflict_item.get("slot_id") or "")
                        add_context(
                            conflict_qid, conflict_slot,
                            plan_items.get(conflict_slot) or {}, conflict_item,
                            "duplicate_conflict",
                        )
                replacement["previous_task_questions"] = previous_task_questions
            if replacement is None or not _replace_plan_item(plans, model_number, slot_id, replacement):
                history.append({"model_number": model_number, "question_id": qid,
                                "slot_id": slot_id, "repair_round": repair_round,
                                "slot_attempt": slot_attempt,
                                "status": "no_compatible_catalog_replacement"})
                continue
            reserved.append({"question_id": qid, "slot_id": slot_id, "qtype": qtype,
                             "task_id": replacement.get("task_id"),
                             "fitb_entry_id": replacement.get("fitb_entry_id"),
                             "catalog_replacement_stage": replacement.get("catalog_replacement_stage"),
                             "task_reused": bool(replacement.get("task_reused")),
                             "fitb_entry_reused": bool(replacement.get("fitb_entry_reused")),
                             "reuse_reason": replacement.get("reuse_reason"),
                             "task_inventory": deepcopy(replacement.get("task_inventory") or {}),
                             "fitb_answer_reused": bool(
                                 replacement.get("fitb_answer_reused")
                             ),
                             "previous_task_questions": deepcopy(
                                 replacement.get("previous_task_questions") or []
                             ),
                             "reused_due_to_catalog_exhaustion": bool(
                                 replacement.get("reused_due_to_catalog_exhaustion")
                             ),
                             "old_plan_item": old_plan_item,
                             "repair_round": repair_round, "slot_attempt": slot_attempt})

        repaired_ids: list[str] = []
        if reserved:
            with ThreadPoolExecutor(max_workers=min(4, len(reserved))) as executor:
                futures = {
                    executor.submit(
                        copy_context().run,
                        _generate_reserved,
                        state,
                        deepcopy(plans),
                        model_number,
                        item["question_id"],
                        exam,
                    ): item for item in reserved
                }
                for future in as_completed(futures):
                    reservation = futures[future]
                    history_reservation = {
                        key: value for key, value in reservation.items()
                        if key != "old_plan_item"
                    }
                    try:
                        qid, generated, qtype, generation_warnings = future.result()
                    except Exception as exc:
                        from app.online.lifecycle import developer_diagnostic
                        diagnostics.append(developer_diagnostic(
                            exc, stage="duplicate_replacement",
                            slot_id=reservation.get("slot_id"),
                            question_id=reservation.get("question_id"),
                            task_id=reservation.get("task_id"),
                        ))
                        qid, generated, qtype, generation_warnings = (
                            reservation["question_id"], None, reservation["qtype"], [str(exc)]
                        )
                    warnings.extend(generation_warnings)
                    if generated is None:
                        _replace_plan_item(
                            plans, model_number, reservation["slot_id"],
                            reservation.get("old_plan_item") or {},
                        )
                        history.append({**history_reservation, "model_number": model_number,
                                        "status": "generation_failed"})
                        failed_id = str(reservation.get("fitb_entry_id") or reservation.get("task_id") or "")
                        if failed_id:
                            blacklists.setdefault(_slot_attempt_key(model_number, reservation["slot_id"]), []).append(failed_id)
                            failure_memory.setdefault(_slot_attempt_key(model_number, reservation["slot_id"]), []).append({
                                "candidate_id": failed_id, "reason": "generation_failed",
                            })
                        continue
                    from app.online.difficulty_adjustment import adjust_generated_question
                    active_plan = _plan_by_slot(plans, model_number).get(
                        reservation["slot_id"]
                    ) or {}
                    generated, adjustment = adjust_generated_question(
                        state=state, qtype=qtype or reservation["qtype"],
                        question=generated, plan=active_plan,
                        previous_questions=reservation.get("previous_task_questions") or [],
                    )
                    active_plan["adjustment_attempts"] = int(
                        active_plan.get("adjustment_attempts") or 0
                    ) + (0 if adjustment.get("status") == "not_needed" else 1)
                    if adjustment.get("status") == "succeeded":
                        target = str(active_plan.get("requested_difficulty") or "")
                        active_plan["difficulty"] = target
                        active_plan["actual_difficulty"] = target
                        active_plan["reasoning_depth"] = target
                        active_plan["reasoning_requirements"] = list(
                            adjustment.get("changed_reasoning_requirements") or []
                        )
                        active_plan["difficulty_fallback_reason"] = None
                        active_plan["adjustment_failure_reason"] = None
                        active_plan["difficulty_adaptation"] = {
                            "status": "adapted",
                            "from_difficulty": adjustment.get("from_difficulty")
                            or adjustment.get("direction", "").split("_to_")[0],
                            "to_difficulty": target,
                            "base_task_id": active_plan.get("base_catalog_task_id")
                            or active_plan.get("task_id"),
                            "strategy": "generated_question_adjustment",
                            "failed_strategies": [],
                            "changed_reasoning_requirements": list(
                                adjustment.get("changed_reasoning_requirements") or []
                            ),
                            "evidence_support": list(
                                adjustment.get("evidence_support") or []
                            ),
                            "evidence_justification": adjustment.get(
                                "evidence_justification"
                            ),
                        }
                    elif adjustment.get("status") == "failed":
                        active_plan["adjustment_failure_reason"] = adjustment.get("reason")
                        active_plan["difficulty_fallback_reason"] = adjustment.get("reason")
                    history.append({
                        **history_reservation, "model_number": model_number,
                        "status": "difficulty_adjustment_" + adjustment.get("status", "unknown"),
                        "difficulty_adjustment": adjustment,
                    })
                    current = _build_by_id(exam.get("questions") or {}).get(qid)
                    if current is None:
                        continue
                    current[2].clear(); current[2].update(generated)
                    repaired_ids.append(qid)
                    history.append({**history_reservation, "model_number": model_number,
                                    "status": "replaced"})
            fitb = exam.get("questions", {}).get("fill_in_the_blank")
            if isinstance(fitb, dict):
                if not _fitb_word_bank_valid(fitb):
                    _rebuild_fitb_word_bank(fitb)
        normal_repaired_ids = list(repaired_ids)
        repaired_ids.extend(numeric_repaired_ids)
        numeric_repaired_ids = []
        if repaired_ids:
            verification_required = verification_required or bool(normal_repaired_ids)
            from app.online.exam_builder import balance_mcq_answer_positions
            repaired_ids = list(dict.fromkeys([
                *repaired_ids,
                *balance_mcq_answer_positions(exam.get("questions") or {}, model_number),
            ]))
            pending_revalidation[model_number] = repaired_ids
            if model_number in validated:
                validated.remove(model_number)
        else:
            pending_revalidation.pop(model_number, None)

    return {
        "plans": plans, "generated_exams": list(exams.values()),
        "duplicate_repair_rounds": (
            max(completed_rounds, 3)
            if numeric_fallback_attempted or finalization_pending else completed_rounds
        ),
        "duplicate_verification_pending": verification_required,
        "duplicate_verification_attempt": repair_round if verification_required else 0,
        "duplicate_replacement_attempts": slot_attempts,
        "pending_revalidation_ids": pending_revalidation,
        "pending_duplicate_groups": [], "validated_models": validated,
        "duplicate_repair_history": history, "warnings": warnings,
        "duplicate_fact_cache": fact_cache,
        "duplicate_candidate_blacklist": {
            key: list(dict.fromkeys(values)) for key, values in blacklists.items()
        },
        "duplicate_replacement_failures": failure_memory,
        "duplicate_judge_exempt_ids": list(dict.fromkeys(judge_exempt_ids)),
        "duplicate_numeric_fallback_pending_validation": bool(judge_exempt_ids),
        "duplicate_repair_finalization_pending": (
            finalization_pending or bool(judge_exempt_ids)
        ),
        "unresolved_duplicate_conflicts": unresolved_conflicts,
        "developer_diagnostics": diagnostics,
    }


def has_pending_duplicate_repairs(state: dict[str, Any]) -> bool:
    return bool(state.get("pending_duplicate_groups"))
