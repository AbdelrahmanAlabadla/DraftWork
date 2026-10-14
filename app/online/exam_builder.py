from __future__ import annotations

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from typing import Any, Callable

from app.config import MAX_PARALLEL_GENERATION_CALLS
from app.llm.client import request_structured
from app.llm.schemas import (
    fitb_items_response_model,
    generation_response_model,
    objective_bundle_response_model,
)
from app.logging_conf import get_logger
from app.online.assessment_catalog import (
    catalog_exhaustion_replacement,
    is_pure_text_fitb_value,
    is_authorized_catalog_exhaustion_reuse,
    normalize_fitb_answer_key,
)
from app.online.eval_stats import (
    create_pipeline_eval,
    finalize_pipeline_eval,
    public_eval,
    record_generation_rejection,
    record_initial_generation,
    record_slot_accepted,
    record_shortfall_result,
)
from app.online.graph import get_exam_graph
from app.online.models import (
    TYPE_ORDER,
    contains_forbidden_phrase,
    normalize_fitb_item,
    normalize_text,
    parse_questions_obj,
    question_text,
    render_markdown,
    split_valid_invalid,
)
from app.online.prompts import (
    DEFINITION_SCHEMA,
    EQUATION_SCHEMA,
    ESSAY_SCHEMA,
    MCQ_SCHEMA,
    WHY_SCHEMA,
    TRUE_FALSE_SCHEMA,
    WORD_PROBLEM_SCHEMA,
    build_validation_repair_prompt,
)

logger = get_logger("EXAM_BUILDER")

_MAX_ATTEMPTS = 3

# Upper bound on a single generation request's max_output_tokens. Keeps combined
# prompt + output within the model's context window even for large per-type counts.
_MAX_TOKENS = 4000


def _deepseek_transport_exhausted(exc: Exception) -> bool:
    """True after one DeepSeek call has already used its two network retries.

    Generation attempts are for schema/content recovery. They must not multiply
    the transport client's initial request plus two bounded retries into three
    fresh generation attempts.
    """
    from app.llm.deepseek import (
        DeepSeekNetworkError,
        DeepSeekRateLimitError,
        DeepSeekServerError,
        DeepSeekTimeoutError,
    )

    return isinstance(exc, (
        DeepSeekNetworkError,
        DeepSeekRateLimitError,
        DeepSeekServerError,
        DeepSeekTimeoutError,
    ))

# Words that carry little topical signal and would otherwise inflate the
# similarity of unrelated questions.
_STOPWORDS = frozenset(
    (
        "a an and are as at be been but by can could did do does for from had has have he her "
        "his how i if in into is it its may me more most my no not of on or our out over she "
        "should so some such than that the their them then there these they this those to under "
        "up was we were what when where which who why will with would you your following "
        "best describes describe explain defines define list give giving name"
    ).split()
)

# Jaccard token-overlap above which questions inside one model are treated as
# near-duplicates. Cross-model variants share concepts by design and are checked
# separately for exact wording only.
_NEAR_DUP_THRESHOLD = 0.60

# Stable internal question IDs are stamped by the code in the form
# `model{N}_{type}_{seq}` and never requested from the LLM. They uniquely
# identify the exam model, the question type, and the individual question.
_TYPE_ID_SLUGS = {
    "mcq": "mcq",
    "true_false": "true_false",
    "fill_in_the_blank": "fill_in_the_blank",
    "definition": "definition",
    "why": "why",
    "equation": "equation",
    "word_problem": "word_problem",
    "essay": "essay",
}


def _slot_ids(items: list[dict[str, Any]], model_number: int, qtype: str) -> list[str]:
    return [
        str(item.get("slot_id") or f"m{model_number}_{qtype}_{index}")
        for index, item in enumerate(items, start=1)
    ]


def _record_invalid_candidates(
    eval_stats: dict[str, Any] | None,
    model_number: int,
    qtype: str,
    invalid_items: list[Any],
) -> None:
    """Record structural rejection against its slot whenever the ID survived."""
    for item in invalid_items:
        sid = str(item.get("slot_id") or "") if isinstance(item, dict) else ""
        record_generation_rejection(
            eval_stats,
            model_number,
            qtype,
            "invalid_structure",
            slot_id=sid or None,
        )


def _context_for_items(
    items: list[dict[str, Any]], chunks: list[dict[str, Any]], fallback: str
) -> str:
    """Render each unique source chunk referenced by the requested slots once."""
    wanted = {
        str(chunk_id)
        for item in items
        for chunk_id in [item.get("source_chunk_id"), *(item.get("support_chunk_ids") or [])]
        if chunk_id
    }
    if not wanted:
        return fallback
    selected = [c for c in chunks if str(c.get("child_id")) in wanted]
    if not selected:
        return fallback
    return "\n\n".join(
        "\n".join(
            part
            for part in (
                f"[source_chunk_id={chunk.get('child_id')}]",
                f"## {chunk.get('parent_title') or 'Untitled'}",
                f"### {chunk.get('chunk_title')}" if chunk.get("chunk_title") else "",
                str(chunk.get("content") or ""),
            )
            if part
        )
        for chunk in selected
    )


def _generation_plan_items(
    items: list[dict[str, Any]],
    qtype: str,
    chunks: list[dict[str, Any]],
    assessment_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach only each slot's evidence and its relevant profile slice."""
    source_by_id = {
        str(chunk.get("child_id")): str(chunk.get("content") or "")
        for chunk in chunks
        if chunk.get("child_id")
    }
    type_profile = (
        (assessment_profile.get("question_types") or {}).get(qtype) or {}
    )
    guidance = {
        "domains": assessment_profile.get("domains") or [],
        "educational_level": assessment_profile.get("educational_level") or "unspecified",
        "assessment_style": assessment_profile.get("assessment_style") or "",
        "question_type": qtype,
        "support": type_profile.get("support") or "supported",
        "task_operations": type_profile.get("task_operations") or [],
        "form_guidance": type_profile.get("form_guidance") or "",
        "unsupported_visual_tasks": assessment_profile.get("unsupported_visual_tasks") or [],
    }
    enriched: list[dict[str, Any]] = []
    for item in items:
        source_ids = list(dict.fromkeys([
            str(item.get("source_chunk_id") or ""),
            *(str(value) for value in item.get("support_chunk_ids") or []),
        ]))
        enriched.append({
            **item,
            "assessment_guidance": guidance,
            "assigned_sources": {
                source_id: source_by_id[source_id]
                for source_id in source_ids
                if source_id and source_by_id.get(source_id)
            },
        })
    return enriched


def _attach_plan_review_context(
    plan_items: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Attach compact prior-slot plans without creating another task identity."""
    by_slot = {
        str(item.get("slot_id") or ""): item
        for values in plan_items.values()
        for item in values
        if item.get("slot_id")
    }
    for item in by_slot.values():
        references = []
        for slot_id in item.get("forced_duplicate_slot_ids") or []:
            prior = by_slot.get(str(slot_id))
            if prior is None:
                continue
            references.append({
                "slot_id": prior.get("slot_id"),
                "task_id": prior.get("task_id"),
                "fitb_entry_id": prior.get("fitb_entry_id"),
                "concept": prior.get("concept"),
                "task_operation": prior.get("task_operation"),
                "target": prior.get("target"),
                "reasoning_mode": prior.get("reasoning_mode"),
                "student_work_product": prior.get("student_work_product"),
            })
        if references:
            item["previous_task_plan_items"] = references
            if item.get("task_reused") or item.get("fitb_entry_reused"):
                item["previous_task_questions"] = [
                    {
                        **reference,
                        "question_id": None,
                        "question": None,
                        "context_role": "previous_planned_task_instance",
                        "assessment_angle": (
                            f"{reference.get('task_operation')} -> {reference.get('target')}"
                        ),
                    }
                    for reference in references
                    if (
                        item.get("task_id")
                        and reference.get("task_id") == item.get("task_id")
                    ) or (
                        item.get("fitb_entry_id")
                        and reference.get("fitb_entry_id") == item.get("fitb_entry_id")
                    )
                ]
    return plan_items


def _isolated_item_context(item: dict[str, Any]) -> str:
    sources = item.get("assigned_sources") or {}
    return "\n\n".join(
        f"[source_chunk_id={chunk_id}]\n{content}"
        for chunk_id, content in sources.items()
    )


def _section_items(section: Any) -> list[dict[str, Any]]:
    """Return the flat question list of a section (FITB unwraps its items)."""
    if isinstance(section, dict):
        return list(section.get("items") or [])
    return list(section or [])


_MCQ_POSITION_SEQUENCE = ("B", "D", "A", "C", "C", "A", "D", "B")


def balance_mcq_answer_positions(
    questions: dict[str, Any], model_number: int = 1
) -> list[str]:
    """Rebalance concentrated MCQ answer positions deterministically.

    Only option positions and the stored answer letter change; option content is
    preserved exactly. Any certificate option mapping is recomputed at the same
    time so validation never receives content paired with stale metadata. A
    section is left untouched while every answer position is below 50%.
    """
    mcqs = questions.get("mcq") or []
    labels = ("A", "B", "C", "D")
    eligible: list[tuple[int, dict[str, Any]]] = []
    counts = {label: 0 for label in labels}
    for index, item in enumerate(mcqs):
        options = item.get("options")
        current = str(item.get("correct_answer") or "")
        if not isinstance(options, dict) or current not in labels or any(
            label not in options for label in labels
        ):
            continue
        eligible.append((index, item))
        counts[current] += 1

    total = len(eligible)
    if total <= 1 or not any(count * 2 >= total for count in counts.values()):
        return []

    changed_ids: list[str] = []
    offset = max(0, int(model_number) - 1) % len(_MCQ_POSITION_SEQUENCE)
    while max(counts.values()) - min(counts.values()) > 1:
        source = next(label for label in labels if counts[label] == max(counts.values()))
        target = next(label for label in labels if counts[label] == min(counts.values()))
        matching = [
            (index, item) for index, item in eligible
            if item.get("correct_answer") == source
            and _MCQ_POSITION_SEQUENCE[(index + offset) % len(_MCQ_POSITION_SEQUENCE)] == target
        ]
        index, item = (matching or [
            pair for pair in eligible if pair[1].get("correct_answer") == source
        ])[0]
        options = item["options"]
        options[source], options[target] = options[target], options[source]
        item["correct_answer"] = target
        certificate = item.get("solution_certificate")
        if isinstance(certificate, dict):
            certificate["matching_correct_option"] = target
        if item.get("question_id"):
            changed_ids.append(str(item["question_id"]))
        counts[source] -= 1
        counts[target] += 1
    return changed_ids


def _assign_ids(model_number: int, qtype: str, section: Any) -> None:
    """Stamp a stable `question_id` on every question missing one, in place.

    The per-type sequence continues from the highest existing ID so repair or
    shortfall-fill that appends questions never collides or resets numbering.
    """
    slug = _TYPE_ID_SLUGS.get(qtype, qtype)
    prefix = f"model{model_number}_{slug}_"
    max_seq = 0
    for q in _section_items(section):
        qid = q.get("question_id")
        if qid and str(qid).startswith(prefix):
            try:
                max_seq = max(max_seq, int(str(qid)[len(prefix):]))
            except ValueError:
                pass
    nxt = max_seq + 1
    for q in _section_items(section):
        if not q.get("question_id"):
            q["question_id"] = f"{prefix}{nxt}"
            nxt += 1


def _is_duplicate(qtype: str, question: dict[str, Any], seen: set[tuple[str, str]]) -> bool:
    key = (qtype, " ".join(normalize_text(question_text(qtype, question))))
    return key in seen


def _remember_question(
    qtype: str, question: dict[str, Any], seen: set[tuple[str, str]]
) -> None:
    key = (qtype, " ".join(normalize_text(question_text(qtype, question))))
    seen.add(key)


def _content_tokens(qtype: str, question: dict[str, Any]) -> set[str]:
    """Meaningful tokens of a question, for near-duplicate checks.

    MCQ distractors are excluded because they often inflate unrelated overlap,
    but the correct option text is part of the assessed content and is included.
    For reasoning questions the reference answer is a strong signal too.
    """
    actual_type = _infer_question_type(question, qtype)
    parts = [question_text(actual_type, question)]
    if actual_type == "mcq":
        options = question.get("options") or {}
        parts.append(str(options.get(str(question.get("correct_answer") or "")) or ""))
    elif actual_type in {"definition", "why"}:
        parts.append(str(question.get("reference_answer") or ""))
    elif actual_type in {"equation", "word_problem"}:
        parts.extend(str(step) for step in (question.get("solution_steps") or []))
        parts.append(str(question.get("final_answer") or ""))
    elif actual_type == "essay":
        parts.append(str(question.get("reference_answer") or ""))
        parts.extend(str(kp) for kp in (question.get("key_points") or []))
    elif actual_type == "fill_in_the_blank":
        parts.extend(str(a) for a in (question.get("answers") or []))
    tokens: list[str] = []
    for part in parts:
        tokens.extend(normalize_text(part))
    return {tok for tok in tokens if tok not in _STOPWORDS}


def _infer_question_type(question: dict[str, Any], fallback: str = "") -> str:
    """Keep cross-type comparison tied to the item's actual canonical shape."""
    if "options" in question and "correct_answer" in question:
        return "mcq"
    if "statement" in question:
        return "true_false"
    if "term" in question:
        return "definition"
    if "equation" in question:
        return "equation"
    if "final_answer" in question and "solution_steps" in question:
        return "word_problem"
    if "key_points" in question:
        return "essay"
    if "answers" in question:
        return "fill_in_the_blank"
    if "reference_answer" in question:
        return "why"
    return fallback


def _token_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _is_near_duplicate(
    qtype: str, question: dict[str, Any], accepted: list[dict[str, Any]]
) -> bool:
    """True if the question overlaps enough with an already-accepted one."""
    tokens = _content_tokens(qtype, question)
    for other in accepted:
        if _token_overlap(
            tokens, _content_tokens(_infer_question_type(other, qtype), other)
        ) >= _NEAR_DUP_THRESHOLD:
            return True
    return False


def _is_semantic_plan_duplicate(
    slot_id: str,
    plan_by_slot: dict[str, dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> bool:
    """Prevent unapproved exact task reuse while honoring catalog exhaustion."""
    candidate = plan_by_slot.get(slot_id) or {}
    identity = str(candidate.get("fitb_entry_id") or candidate.get("task_id") or "")
    if not identity:
        return False
    if candidate.get("fitb_entry_id") and candidate.get("fitb_entry_reused"):
        return False
    if is_authorized_catalog_exhaustion_reuse(candidate):
        return False
    for question in accepted:
        other_sid = str(question.get("slot_id") or "")
        other = plan_by_slot.get(other_sid) or {}
        if str(other.get("fitb_entry_id") or other.get("task_id") or "") == identity:
            return True
    return False


# Expected JSON schema per question type, used by structural validation repair.
_SCHEMAS = {
    "mcq": MCQ_SCHEMA,
    "true_false": TRUE_FALSE_SCHEMA,
    "definition": DEFINITION_SCHEMA,
    "why": WHY_SCHEMA,
    "equation": EQUATION_SCHEMA,
    "word_problem": WORD_PROBLEM_SCHEMA,
    "essay": ESSAY_SCHEMA,
}


def _repair_invalid_items(
    qtype: str,
    invalid_raw: list[dict[str, Any]],
    context: str,
    difficulty: str,
    model_number: int,
    language: str = "en",
    max_attempts: int = 2,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Send rejected raw items + the expected schema back for targeted repair.

    The model fixes ONLY the invalid fields/type/structure; it must not invent a
    new question or alter valid content. Returns the newly-valid normalized items.
    """
    from app.llm.factory import create_llm_client

    if not invalid_raw:
        return [], []
    warnings: list[str] = []
    client = create_llm_client()
    schema = _SCHEMAS.get(qtype, "")
    items = list(invalid_raw)
    for attempt in range(1, max_attempts + 1):
        system_prompt, user_prompt = build_validation_repair_prompt(
            qtype, items, schema, context, difficulty=difficulty, model_number=model_number,
            language=language,
        )
        try:
            repair_slot_ids = [
                str(item.get("slot_id"))
                for item in items
                if isinstance(item, dict) and item.get("slot_id")
            ]
            raw = request_structured(
                client,
                user_prompt,
                response_model=generation_response_model(qtype, repair_slot_ids),
                schema_name=f"{qtype}_generation_repair",
                agent="generator",
                item_count=len(items),
                expected_ids=repair_slot_ids or None,
                expected_id_field="slot_id" if repair_slot_ids else None,
                local_json=True,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=min(_MAX_TOKENS, 2048),
            )
        except Exception as exc:
            logger.exception(
                "Question structure repair failed | type=%s | attempt=%d | slots=%s",
                qtype, attempt, repair_slot_ids,
            )
            warnings.append(f"{qtype} repair attempt {attempt}: {exc}")
            if _deepseek_transport_exhausted(exc):
                break
            continue
        valid, still_invalid = split_valid_invalid(qtype, raw)
        if valid:
            return valid, warnings
        items = still_invalid or items
    warnings.append(f"{qtype}: could not repair {len(items)} invalid item(s)")
    return [], warnings


# --------------------------------------------------------------------------
# Generation (plan-driven, one call per model x question type)
# --------------------------------------------------------------------------
def _generate_type_from_plan(
    qtype: str,
    planned_items: list[dict[str, Any]],
    context: str,
    difficulty: str,
    model_number: int,
    seen: set[tuple[str, str]],
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
    language: str = "en",
    eval_stats: dict[str, Any] | None = None,
    global_plan_by_slot: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate and retry exact stable slots instead of inferring gaps by count."""
    from app.llm.factory import create_llm_client
    from app.online.prompts import build_prompt

    ordered_ids = _slot_ids(planned_items, model_number, qtype)
    normalized_plan: list[dict[str, Any]] = []
    for sid, item in zip(ordered_ids, planned_items):
        normalized_plan.append({**item, "slot_id": sid})
    pending = {item["slot_id"]: item for item in normalized_plan}
    accepted_by_slot: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    client = create_llm_client()

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        remaining_plan = list(pending.values())
        if not remaining_plan:
            break

        system_prompt, user_prompt = build_prompt(
            qtype,
            len(remaining_plan),
            context,
            difficulty=difficulty,
            model_number=model_number,
            planned_items=remaining_plan,
            language=language,
        )

        feedback = _build_feedback(
            qtype, list(accepted_by_slot.values()), seen, within_model, []
        )
        if feedback:
            user_prompt += (
                "\n\n## Context from earlier attempts\n"
                "Below are questions already accepted that you must NOT repeat or "
                "reword. Produce the planned questions that are still missing only.\n"
                f"{feedback}"
            )

        try:
            remaining_ids = list(pending)
            raw_obj = request_structured(
                client,
                user_prompt,
                response_model=generation_response_model(qtype, remaining_ids),
                schema_name=f"{qtype}_questions",
                agent="generator",
                item_count=len(remaining_plan),
                expected_ids=remaining_ids,
                expected_id_field="slot_id",
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=min(_MAX_TOKENS, max(2048, len(remaining_plan) * 256)),
            )
        except Exception as exc:
            logger.exception(
                "Question generation failed | type=%s | model=%d | attempt=%d | slots=%s",
                qtype, model_number, attempt, remaining_ids,
            )
            warnings.append(
                f"{qtype} attempt {attempt}: LLM returned no valid questions ({exc})"
            )
            if _deepseek_transport_exhausted(exc):
                break
            continue

        # Split valid vs structurally-invalid raw items.
        candidates, invalid_raw = split_valid_invalid(qtype, raw_obj)
        _record_invalid_candidates(eval_stats, model_number, qtype, invalid_raw)
        # Structural fix FIRST: send rejected output + schema back, repair only
        # the broken fields (never regenerate a fresh random question).
        if invalid_raw and len(candidates) < len(remaining_plan):
            plan_by_slot = {item["slot_id"]: item for item in remaining_plan}
            for index, invalid in enumerate(i for i in invalid_raw if isinstance(i, dict)):
                sid = str(invalid.get("slot_id") or "")
                plan_item = plan_by_slot.get(sid) or (
                    remaining_plan[index] if index < len(remaining_plan) else {}
                )
                repaired, repair_warnings = _repair_invalid_items(
                    qtype,
                    [invalid],
                    _isolated_item_context(plan_item),
                    difficulty,
                    model_number,
                    language=language,
                )
                warnings.extend(repair_warnings)
                candidates.extend(repaired)

        # Legacy/test clients may omit slot_id; positional assignment is safe only
        # for those unconstrained responses. Production schemas require the IDs.
        for index, q in enumerate(candidates[: len(remaining_plan)]):
            if not q.get("slot_id") and index < len(remaining_plan):
                q["slot_id"] = remaining_plan[index]["slot_id"]

        for q in candidates[: len(remaining_plan)]:
            sid = str(q.get("slot_id") or "")
            if sid not in pending or sid in accepted_by_slot:
                warnings.append(f"{qtype} attempt {attempt}: unknown or repeated slot_id {sid!r}")
                continue
            text = question_text(qtype, q)
            if contains_forbidden_phrase(text, language):
                logger.warning("Rejected (forbidden phrase) | type=%s | text=%r", qtype, text[:120])
                record_generation_rejection(
                    eval_stats, model_number, qtype, "forbidden_content", slot_id=sid
                )
                continue
            if _is_duplicate(qtype, q, seen):
                logger.warning("Rejected (duplicate) | type=%s | text=%r", qtype, text[:120])
                record_generation_rejection(
                    eval_stats, model_number, qtype, "duplicate", slot_id=sid
                )
                continue
            if _is_semantic_plan_duplicate(
                sid, global_plan_by_slot or {},
                [*accepted_by_slot.values(), *within_model],
            ):
                warnings.append(
                    f"{qtype} attempt {attempt}: semantic plan duplicate for slot {sid}"
                )
                record_generation_rejection(
                    eval_stats, model_number, qtype, "semantic_duplicate", slot_id=sid
                )
                continue
            if _is_near_duplicate(
                qtype, q, [*accepted_by_slot.values(), *within_model]
            ):
                logger.warning("Rejected (near-duplicate) | type=%s | text=%r", qtype, text[:120])
                record_generation_rejection(
                    eval_stats, model_number, qtype, "near_duplicate", slot_id=sid
                )
                continue
            _remember_question(qtype, q, seen)
            accepted_by_slot[sid] = q
            record_slot_accepted(eval_stats, model_number, qtype, sid)
            pending.pop(sid, None)
        logger.info(
            "Plan type attempt | type=%s | model=%d | attempt=%d/%d | slots=%d | accepted=%d/%d | pending=%s",
            qtype,
            model_number,
            attempt,
            _MAX_ATTEMPTS,
            len(ordered_ids),
            len(accepted_by_slot),
            len(ordered_ids),
            list(pending),
        )

    if pending:
        warnings.append(
            f"{qtype}: {len(accepted_by_slot)}/{len(ordered_ids)} generated after attempts; "
            f"missing slot IDs: {', '.join(pending)}"
        )
        logger.warning(
            "Plan type incomplete | type=%s | accepted=%d/%d | missing_slots=%s",
            qtype,
            len(accepted_by_slot),
            len(ordered_ids),
            list(pending),
        )
    return [accepted_by_slot[sid] for sid in ordered_ids if sid in accepted_by_slot], warnings


def _clean_fitb_terms(raw: object) -> list[str]:
    """Collect distinct vocabulary answers; reject formulas and numeric results."""
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    terms: list[str] = []
    for t in raw:
        s = str(t).strip()
        key = " ".join(normalize_text(s))
        is_formula = bool(
            not is_pure_text_fitb_value(s)
            or len(s) > 80
            or len(s.split()) > 8
            or re.fullmatch(
                r"\s*[+\-]?\d+(?:\.\d+)?(?:\s*[+\-*/^×÷]\s*\d+(?:\.\d+)?)+\s*",
                s,
            )
            or re.fullmatch(
                r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*[×x*]\s*10[⁻−\-]?\d+)?(?:\s*\S+)?",
                s,
            )
        )
        if s and not is_formula and key not in seen:
            seen.add(key)
            terms.append(s)
    return terms


def _autofit_word_bank(bank: list[str], count: int) -> list[str]:
    """Deterministically repair a stage-1 term list to exactly count+2 entries.

    Stage-1 output only has to name terms; this fixes size/duplication in code
    so the LLM does not need to be perfect:
    1. de-duplicate (case-insensitive on normalized tokens);
    2. if oversized, keep the first ``count`` terms plus the LAST 2 entries as
       distractors;
    3. if undersized, pad with generic-but-grounded fallback distractors.
    Returns the fixed bank or an empty list when it cannot reach count terms.
    """
    seen: set[tuple[str, ...]] = set()
    unique: list[str] = []
    for term in bank:
        key = tuple(normalize_text(term))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(term.strip())

    terms = unique[:count]
    rest = unique[count:]
    if len(terms) < count:
        return []
    # Keep exactly two grounded distractors from the model's own extras.
    distractors = rest[-2:]
    return terms + distractors if len(distractors) == 2 else []


def _fitb_errors(
    section: dict[str, Any],
    count: int,
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
    language: str = "en",
) -> list[str]:
    """Validate a generated FITB section against every required invariant."""
    errors: list[str] = []
    word_bank = [str(w).strip() for w in (section.get("word_bank") or []) if str(w).strip()]
    bank_tokens = {tuple(normalize_text(w)) for w in word_bank}
    items = section.get("items") or []
    accepted = list(within_model)

    if len(items) != count:
        errors.append(f"FITB item count {len(items)} != requested {count}")

    used: set[tuple[str, ...]] = set()
    for idx, item in enumerate(items):
        question = str(item.get("question") or "").strip()
        if not question:
            errors.append(f"FITB item {idx + 1}: missing question")
            continue
        answers = [str(a).strip() for a in (item.get("answers") or []) if str(a).strip()]
        if len(answers) != 1:
            errors.append(f"FITB item {idx + 1}: has {len(answers)} answers (must be 1)")
        for a in answers:
            atok = tuple(normalize_text(a))
            if not atok:
                errors.append(f"FITB item {idx + 1}: empty answer")
                continue
            if atok not in bank_tokens:
                errors.append(f"FITB item {idx + 1}: answer '{a}' not in Word Bank")
            used.add(atok)
        if contains_forbidden_phrase(question, language):
            errors.append(f"FITB item {idx + 1}: contains a forbidden phrase")
        if not is_pure_text_fitb_value(question, allow_blank=True):
            errors.append(f"FITB item {idx + 1}: question is not a pure-text single blank")
        item_distractors = _clean_fitb_terms(item.get("distractors"))
        if len(item_distractors) != 2:
            errors.append(f"FITB item {idx + 1}: must have exactly 2 pure-text distractors")
        elif any(tuple(normalize_text(value)) in used for value in item_distractors):
            errors.append(f"FITB item {idx + 1}: distractor duplicates an answer")
        for other in accepted:
            if _token_overlap(
                _content_tokens("fill_in_the_blank", item),
                _content_tokens("fill_in_the_blank", other),
            ) >= _NEAR_DUP_THRESHOLD:
                errors.append(f"FITB item {idx + 1}: near-duplicate of an accepted question")
                break

    expected_bank = {
        tuple(normalize_text(value))
        for item in items
        for value in [*(item.get("answers") or []), *(item.get("distractors") or [])]
        if tuple(normalize_text(value))
    }
    if bank_tokens != expected_bank or len(word_bank) != len(items) + 2:
        errors.append("Word Bank must contain every final answer plus exactly two shared distractors")
    return errors


def _synchronize_fitb_shared_bank(items: list[dict[str, Any]]) -> list[str]:
    """Build the student bank from active answers and two shared distractors only."""
    answers = _clean_fitb_terms([
        answer for item in items for answer in (item.get("answers") or [])
    ])
    answer_keys = {normalize_fitb_answer_key(value) for value in answers}
    shared: list[str] = []
    shared_keys: set[str] = set()
    for item in items:
        for value in _clean_fitb_terms(item.get("distractors")):
            key = normalize_fitb_answer_key(value)
            if not key or key in answer_keys or key in shared_keys:
                continue
            shared.append(value)
            shared_keys.add(key)
            if len(shared) == 2:
                break
        if len(shared) == 2:
            break
    if len(answers) != len(items) or len(shared) != 2:
        return []
    # The same two options are section-level metadata. Keeping them on each
    # item preserves the repair/validator wire shape without creating 2N terms.
    for item in items:
        item["distractors"] = list(shared)
    return [*answers, *shared]


def _generate_fitb_items(
    count: int,
    planned_items: list[dict[str, Any]],
    answer_by_slot: dict[str, str],
    context: str,
    difficulty: str,
    model_number: int,
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
    language: str = "en",
    eval_stats: dict[str, Any] | None = None,
    global_plan_by_slot: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Stage 2: write every blank around its prevalidated per-slot answer."""
    from app.llm.factory import create_llm_client
    from app.online.prompts import build_fitb_items_prompt

    warnings: list[str] = []
    client = create_llm_client()
    ids = _slot_ids(planned_items, model_number, "fill_in_the_blank")
    normalized_plan = [
        {**item, "slot_id": sid} for sid, item in zip(ids, planned_items)
    ]
    pending = {item["slot_id"]: item for item in normalized_plan}
    accepted: dict[str, dict[str, Any]] = {}

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if not pending:
            break
        remaining_plan = list(pending.values())
        system_prompt, user_prompt = build_fitb_items_prompt(
            len(remaining_plan),
            {item["slot_id"]: answer_by_slot[item["slot_id"]] for item in remaining_plan},
            context, difficulty=difficulty,
            model_number=model_number, planned_items=remaining_plan, language=language,
        )
        try:
            raw = request_structured(
                client,
                user_prompt,
                response_model=fitb_items_response_model(list(pending)),
                schema_name="fill_in_the_blank_items",
                agent="generator",
                item_count=len(remaining_plan),
                operation="fitb_items",
                expected_ids=list(pending),
                expected_id_field="slot_id",
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=min(_MAX_TOKENS, max(2048, len(remaining_plan) * 320)),
            )
        except Exception as exc:
            logger.exception(
                "FITB item generation failed | model=%d | attempt=%d | slots=%s",
                model_number, attempt, list(pending),
            )
            warnings.append(f"fill_in_the_blank items attempt {attempt}: {exc}")
            if _deepseek_transport_exhausted(exc):
                break
            continue
        items_raw = raw.get("items") if isinstance(raw, dict) else None
        items_raw = items_raw if isinstance(items_raw, list) else []
        assembled = []
        for value in items_raw:
            if not isinstance(value, dict):
                continue
            sid = str(value.get("slot_id") or "")
            plan = pending.get(sid) or {}
            assembled.append({
                "slot_id": sid,
                "fitb_entry_id": plan.get("fitb_entry_id"),
                "question": value.get("stem"),
                "answers": [answer_by_slot.get(sid)],
                "distractors": [value.get("distractor_1"), value.get("distractor_2")],
            })
        items = [normalize_fitb_item(i) for i in assembled]
        items = [i for i in items if i is not None]
        for index, item in enumerate(items):
            if not item.get("slot_id") and index < len(remaining_plan):
                item["slot_id"] = remaining_plan[index]["slot_id"]
        rejected_count = len(items_raw) - len(items)
        if rejected_count:
            invalid_fitb = [
                item for item in assembled if normalize_fitb_item(item) is None
            ]
            _record_invalid_candidates(
                eval_stats, model_number, "fill_in_the_blank", invalid_fitb
            )

        for item in items[: len(remaining_plan)]:
            sid = str(item.get("slot_id") or "")
            if sid not in pending:
                warnings.append(
                    f"fill_in_the_blank attempt {attempt}: unknown or repeated slot_id {sid!r}"
                )
                continue
            expected_answer = answer_by_slot.get(sid)
            returned_distractors = _clean_fitb_terms(item.get("distractors"))
            answer_key = normalize_fitb_answer_key(expected_answer)
            distractor_keys = [normalize_fitb_answer_key(value) for value in returned_distractors]
            if (
                len(returned_distractors) != 2
                or len(set(distractor_keys)) != 2
                or answer_key in set(distractor_keys)
            ):
                warnings.append(
                    f"fill_in_the_blank attempt {attempt}: slot {sid} returned invalid distractors"
                )
                continue
            if _is_semantic_plan_duplicate(
                sid, global_plan_by_slot or {}, [*within_model, *accepted.values()]
            ):
                warnings.append(
                    f"fill_in_the_blank attempt {attempt}: semantic plan duplicate for slot {sid}"
                )
                record_generation_rejection(
                    eval_stats, model_number, "fill_in_the_blank", "semantic_duplicate",
                    slot_id=sid,
                )
                continue
            # Candidate validation is slot-local. The complete shared bank is
            # assembled and checked after every accepted item is available.
            word_bank = [expected_answer, *returned_distractors]
            errors = _fitb_errors(
                {"word_bank": list(word_bank), "items": [item]},
                1,
                [*within_model, *accepted.values()],
                [],
                language=language,
            )
            content_errors = [
                error for error in errors
                if "count" not in error and "distractors" not in error
            ]
            if content_errors:
                detail = " ".join(content_errors)
                reason = (
                    "forbidden_content" if "forbidden phrase" in detail
                    else "near_duplicate" if "near-duplicate" in detail
                    else "invalid_structure"
                )
                record_generation_rejection(
                    eval_stats, model_number, "fill_in_the_blank", reason,
                    slot_id=sid,
                )
                warnings.append(
                    f"fill_in_the_blank attempt {attempt}: slot {sid} rejected ({reason})"
                )
                continue
            accepted[sid] = item
            pending.pop(sid, None)
            record_slot_accepted(
                eval_stats, model_number, "fill_in_the_blank", sid
            )

    if pending:
        warnings.append(
            "fill_in_the_blank: "
            f"{len(accepted)}/{len(ids)} generated after attempts; missing slot IDs: "
            f"{', '.join(pending)}"
        )
    return [accepted[sid] for sid in ids if sid in accepted], warnings


def _generate_fitb_type(
    count: int,
    planned_items: list[dict[str, Any]],
    context: str,
    difficulty: str,
    model_number: int,
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
    language: str = "en",
    eval_stats: dict[str, Any] | None = None,
    global_plan_by_slot: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Generate stems/distractors around authoritative catalog-bank answers."""
    normalized_plan = [
        {**item, "slot_id": sid}
        for sid, item in zip(
            _slot_ids(planned_items, model_number, "fill_in_the_blank"), planned_items
        )
    ]
    warnings: list[str] = []
    answer_by_slot = {
        str(item["slot_id"]): str(item.get("fitb_answer") or "").strip()
        for item in normalized_plan
        if str(item.get("fitb_answer") or "").strip()
        and item.get("fitb_entry_id")
    }
    valid_answers = _clean_fitb_terms(list(answer_by_slot.values()))
    if len(valid_answers) != len(normalized_plan) or len({
        normalize_fitb_answer_key(value) for value in valid_answers
    }) != len(valid_answers):
        warnings.append("fill_in_the_blank: selected catalog entries have invalid or duplicate answers")
        return None, warnings

    answer_plans = [
        item for item in normalized_plan
        if str(item.get("slot_id") or "") in answer_by_slot
    ]
    items, item_warnings = _generate_fitb_items(
        len(answer_plans), answer_plans, answer_by_slot, context, difficulty,
        model_number, within_model, previous_exams,
        language=language, eval_stats=eval_stats,
        global_plan_by_slot=global_plan_by_slot,
    )
    warnings.extend(item_warnings)
    if not items:
        warnings.append("fill_in_the_blank: no valid items generated")
        return {"word_bank": [], "items": []}, warnings
    bank = _synchronize_fitb_shared_bank(items)
    if not bank:
        # Keep valid slot output. Shortfall/revalidation may repair the shared
        # options; a bank defect must never erase unrelated accepted slots.
        warnings.append("fill_in_the_blank: shared Word Bank could not yet be built")
    random.shuffle(bank)
    return {"word_bank": bank, "items": items}, warnings


def _generate_obj_bundle(
    planned: dict[str, list[dict[str, Any]]],
    context: str,
    difficulty: str,
    model_number: int,
    within_model: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    previous_exams: list[dict[str, Any]],
    language: str = "en",
    eval_stats: dict[str, Any] | None = None,
    global_plan_by_slot: dict[str, dict[str, Any]] | None = None,
    mcq_task_replacer: Callable[[str, dict[str, Any], str], dict[str, Any] | None] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Generate MCQ and True/False together while tracking exact pending slots."""
    from app.llm.factory import create_llm_client
    from app.online.prompts import build_obj_bundled_prompt

    warnings: list[str] = []
    client = create_llm_client()
    normalized: dict[str, list[dict[str, Any]]] = {}
    pending: dict[str, dict[str, dict[str, Any]]] = {}
    accepted: dict[str, dict[str, dict[str, Any]]] = {}
    tf_balance_fallback: dict[str, dict[str, Any]] = {}
    for qtype in ("mcq", "true_false"):
        items = planned.get(qtype) or []
        ids = _slot_ids(items, model_number, qtype)
        normalized[qtype] = [{**item, "slot_id": sid} for sid, item in zip(ids, items)]
        pending[qtype] = {item["slot_id"]: item for item in normalized[qtype]}
        accepted[qtype] = {}

    def incomplete() -> bool:
        return any(pending[qtype] for qtype in ("mcq", "true_false"))

    attempt = 0
    while incomplete() and attempt < _MAX_ATTEMPTS:
        attempt += 1
        rem_planned = {q: list(pending[q].values()) for q in ("mcq", "true_false")}
        bundle_max_tokens = min(
            _MAX_TOKENS,
            max(2048, sum(len(values) for values in rem_planned.values()) * 256),
        )
        feedback = _build_bundle_feedback(
            list(accepted["mcq"].values()),
            list(accepted["true_false"].values()),
            None,
            within_model,
            [],
        )
        system_prompt, user_prompt = build_obj_bundled_prompt(
            rem_planned, context, difficulty=difficulty,
            model_number=model_number, feedback=feedback, language=language,
        )
        try:
            expected_ids = [*pending["mcq"], *pending["true_false"]]
            raw = request_structured(
                client,
                user_prompt,
                response_model=objective_bundle_response_model(
                    list(pending["mcq"]), list(pending["true_false"])
                ),
                schema_name="mcq_true_false_bundle",
                agent="generator",
                item_count=len(expected_ids),
                expected_ids=expected_ids,
                expected_id_field="slot_id",
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=bundle_max_tokens,
            )
        except Exception as exc:
            logger.exception(
                "Objective bundle generation failed | model=%d | attempt=%d | slots=%s",
                model_number, attempt, expected_ids,
            )
            warnings.append(f"obj bundle attempt {attempt}: {exc}")
            if _deepseek_transport_exhausted(exc):
                break
            continue
        if not isinstance(raw, dict):
            warnings.append(f"obj bundle attempt {attempt}: non-object response")
            continue

        for qtype in ("mcq", "true_false"):
            if not pending[qtype]:
                continue
            section_raw = raw.get(qtype) if isinstance(raw, dict) else None
            if qtype == "mcq" and isinstance(section_raw, dict):
                for unsuitable in section_raw.get("unsuitable_tasks") or []:
                    sid = str((unsuitable or {}).get("slot_id") or "")
                    reason = str((unsuitable or {}).get("reason") or "")
                    current_plan = pending["mcq"].get(sid)
                    if current_plan is None:
                        warnings.append(
                            f"obj bundle mcq attempt {attempt}: unsuitable result for unknown slot {sid!r}"
                        )
                        continue
                    replacement = (
                        mcq_task_replacer(sid, current_plan, reason)
                        if mcq_task_replacer is not None else None
                    )
                    if replacement is None:
                        warnings.append(
                            f"mcq_task_unsuitable: no compatible replacement for slot {sid}: {reason}"
                        )
                        continue
                    pending["mcq"][sid] = replacement
                    normalized["mcq"] = [
                        replacement if str(item.get("slot_id") or "") == sid else item
                        for item in normalized["mcq"]
                    ]
                    if global_plan_by_slot is not None:
                        global_plan_by_slot[sid] = replacement
                    warnings.append(
                        f"mcq_task_unsuitable: reassigned slot {sid} from task "
                        f"{current_plan.get('task_id')} to {replacement.get('task_id')}"
                    )
            candidates, invalid_raw = split_valid_invalid(qtype, section_raw)
            _record_invalid_candidates(eval_stats, model_number, qtype, invalid_raw)
            remaining_items = list(pending[qtype].values())
            if invalid_raw and len(candidates) < len(pending[qtype]):
                plan_by_slot = {item["slot_id"]: item for item in remaining_items}
                for index, invalid in enumerate(i for i in invalid_raw if isinstance(i, dict)):
                    sid = str(invalid.get("slot_id") or "")
                    plan_item = plan_by_slot.get(sid) or (
                        remaining_items[index] if index < len(remaining_items) else {}
                    )
                    repaired, rwarn = _repair_invalid_items(
                        qtype, [invalid], _isolated_item_context(plan_item),
                        difficulty, model_number, language=language,
                    )
                    warnings.extend(rwarn)
                    candidates.extend(repaired)
            for index, q in enumerate(candidates[: len(remaining_items)]):
                if not q.get("slot_id") and index < len(remaining_items):
                    q["slot_id"] = remaining_items[index]["slot_id"]
            for q in candidates[: len(remaining_items)]:
                sid = str(q.get("slot_id") or "")
                if sid not in pending[qtype]:
                    warnings.append(
                        f"obj bundle {qtype} attempt {attempt}: unknown slot_id {sid!r}"
                    )
                    continue
                already_accepted = [
                    *accepted["mcq"].values(),
                    *accepted["true_false"].values(),
                    *within_model,
                ]
                if _is_semantic_plan_duplicate(
                    sid, global_plan_by_slot or {}, already_accepted
                ):
                    warnings.append(
                        f"obj bundle {qtype} attempt {attempt}: semantic plan duplicate for slot {sid}"
                    )
                    record_generation_rejection(
                        eval_stats, model_number, qtype, "semantic_duplicate", slot_id=sid,
                    )
                    continue
                if _filter_one(
                    qtype, q, already_accepted,
                    seen, warnings, attempt, language, eval_stats, model_number,
                ):
                    accepted[qtype][sid] = q
                    pending[qtype].pop(sid, None)

        tf_total = len(normalized["true_false"])
        if not pending["true_false"] and tf_total >= 4:
            true_ids = [sid for sid, item in accepted["true_false"].items() if item.get("answer") == "True"]
            false_ids = [sid for sid, item in accepted["true_false"].items() if item.get("answer") == "False"]
            minimum_each = 2
            minority_ids, majority_ids = (
                (true_ids, false_ids) if len(true_ids) < len(false_ids) else (false_ids, true_ids)
            )
            needed = max(0, minimum_each - len(minority_ids))
            plan_by_id = {item["slot_id"]: item for item in normalized["true_false"]}
            for sid in majority_ids[-needed:] if needed else []:
                removed = accepted["true_false"].pop(sid, None)
                if removed is not None:
                    tf_balance_fallback[sid] = removed
                pending["true_false"][sid] = plan_by_id[sid]
            if needed:
                warnings.append(
                    f"obj bundle attempt {attempt}: retrying {needed} True/False slot(s) "
                    "to avoid a one-sided answer pattern"
                )

        warnings.append(
            f"obj bundle attempt {attempt}: mcq {len(accepted['mcq'])}/{len(normalized['mcq'])} | "
            f"tf {len(accepted['true_false'])}/{len(normalized['true_false'])}"
        )

    # Completeness wins if all bounded attempts produced the same polarity.
    # Keep the best previously valid statement rather than silently shrinking
    # the exam; the validation report still exposes the weaker balance.
    for sid in list(pending["true_false"]):
        fallback = tf_balance_fallback.get(sid)
        if fallback is not None:
            accepted["true_false"][sid] = fallback
            pending["true_false"].pop(sid, None)
            warnings.append(
                f"obj bundle: retained valid True/False slot {sid} after balance retries"
            )

    if incomplete():
        missing = [sid for values in pending.values() for sid in values]
        warnings.append(f"obj bundle missing slot IDs after attempts: {', '.join(missing)}")
    return {
        qtype: [
            accepted[qtype][item["slot_id"]]
            for item in normalized[qtype]
            if item["slot_id"] in accepted[qtype]
        ]
        for qtype in ("mcq", "true_false")
    }, warnings


def _filter_one(
    qtype: str,
    q: dict[str, Any],
    accepted: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    warnings: list[str],
    attempt: int,
    language: str = "en",
    eval_stats: dict[str, Any] | None = None,
    model_number: int = 0,
) -> bool:
    """Accept a question iff it passes forbid/dup/near-dup checks."""
    text = question_text(qtype, q)
    if contains_forbidden_phrase(text, language):
        warnings.append(f"obj bundle {qtype} attempt {attempt}: forbidden phrase")
        record_generation_rejection(
            eval_stats, model_number, qtype, "forbidden_content",
            slot_id=str(q.get("slot_id") or "") or None,
        )
        return False
    if _is_duplicate(qtype, q, seen):
        warnings.append(f"obj bundle {qtype} attempt {attempt}: duplicate")
        record_generation_rejection(
            eval_stats, model_number, qtype, "duplicate",
            slot_id=str(q.get("slot_id") or "") or None,
        )
        return False
    if _is_near_duplicate(qtype, q, accepted):
        warnings.append(f"obj bundle {qtype} attempt {attempt}: near-duplicate")
        record_generation_rejection(
            eval_stats, model_number, qtype, "near_duplicate",
            slot_id=str(q.get("slot_id") or "") or None,
        )
        return False
    _remember_question(qtype, q, seen)
    record_slot_accepted(
        eval_stats, model_number, qtype, str(q.get("slot_id") or "")
    )
    return True


def _build_bundle_feedback(
    acc_mcq: list[dict[str, Any]],
    acc_tf: list[dict[str, Any]],
    fitb: dict[str, Any] | None,
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
) -> str:
    """List accepted objective questions so a retry only fills the gap."""
    parts: list[str] = []
    accepted = list(within_model)
    if acc_mcq:
        lines = "\n".join(f"- {question_text('mcq', q)}" for q in acc_mcq)
        parts.append("Already-accepted MCQ (do NOT repeat or reword these):\n" + lines)
    if acc_tf:
        lines = "\n".join(f"- {question_text('true_false', q)}" for q in acc_tf)
        parts.append("Already-accepted True/False (do NOT repeat or reword these):\n" + lines)
        true_count = sum(item.get("answer") == "True" for item in acc_tf)
        false_count = sum(item.get("answer") == "False" for item in acc_tf)
        if true_count > false_count:
            parts.append("The remaining True/False statement must be a source-grounded FALSE statement.")
        elif false_count > true_count:
            parts.append("The remaining True/False statement must be a source-grounded TRUE statement.")
    if fitb is not None:
        lines = "\n".join(f"- {question_text('fill_in_the_blank', q)}" for q in fitb.get("items") or [])
        parts.append("Fill-in-the-Blank already done (do NOT regenerate it):\n" + lines)
    if accepted:
        lines = "\n".join(f"- {question_text('mcq', q)}" for q in accepted if question_text('mcq', q))
        parts.append(
            "Other already-accepted questions across the exam (do NOT test the same "
            "concept):\n" + lines
        )
    if not parts:
        return ""
    return (
        "\n## Already-generated questions (only fill what is still missing; do not "
        "regenerate these)\n" + "\n\n".join(parts) + "\n\n"
    )


def _section_count(qtype: str, section: Any) -> int:
    """Number of questions in a section (FITB counts its items)."""
    if qtype == "fill_in_the_blank":
        return len((section or {}).get("items") or []) if isinstance(section, dict) else 0
    return len(section) if isinstance(section, list) else 0


def _repair_planned_for(
    qtype: str,
    missing: int,
    plan_items: dict[str, list[dict[str, Any]]],
    context: str,
) -> list[dict[str, Any]]:
    """Return ``missing`` grounded plan items for a repair.

    Reuses the original plan concepts for the type when available. If the plan has
    none, we never send a blank concept: the fallback is a grounded topic derived
    from the selected context so the repair stays grounded in the source.
    """
    items = [dict(i) for i in (plan_items.get(qtype) or [])]
    if not items:
        first_line = next(
            (ln.strip() for ln in str(context).splitlines() if ln.strip()), "Selected source content"
        )
        items = [
            {"topic": first_line[:120], "concept_to_test": "a key idea from the selected source"}
            for _ in range(missing)
        ]
    return items[:missing] if len(items) >= missing else items


def _remove_extras(qtype: str, section: Any, extras: int, warnings: list[str]) -> Any:
    """Remove exactly ``extras`` items from the end; valid items stay untouched."""
    if qtype == "fill_in_the_blank" and isinstance(section, dict):
        section = dict(section)
        items = list(section.get("items") or [])[: -extras] if extras else section.get("items") or []
        section["items"] = items
    elif isinstance(section, list):
        section = section[:-extras] if extras else section
    warnings.append(f"{qtype}: removed {extras} extra question(s) (trimmed to target).")
    return section


def _repair_shortfalls(
    questions: dict[str, Any],
    tasks: list[tuple[str, int]],
    plan_items: dict[str, list[dict[str, Any]]],
    context: str,
    difficulty: str,
    model_number: int,
    seen: set[tuple[str, str]],
    within_model: list[dict[str, Any]],
    previous_questions: list[dict[str, Any]],
    language: str = "en",
    max_passes: int = 2,
    appended_by_type: dict[str, int] | None = None,
    eval_stats: dict[str, Any] | None = None,
    retrieved_chunks: list[dict[str, Any]] | None = None,
    mcq_task_replacer: Callable[[str, dict[str, Any], str], dict[str, Any] | None] | None = None,
    fitb_entry_replacer: Callable[[str, dict[str, Any], str], dict[str, Any] | None] | None = None,
) -> list[str]:
    """Retry only the exact planned slot IDs that are still absent."""
    warnings: list[str] = []
    target = {qtype: count for qtype, count in tasks}
    normalized_plan: dict[str, list[dict[str, Any]]] = {}
    for qtype, count in tasks:
        original = [dict(item) for item in (plan_items.get(qtype) or [])]
        ids = _slot_ids(original, model_number, qtype)
        normalized_plan[qtype] = [
            {**item, "slot_id": sid} for sid, item in zip(ids, original[:count])
        ]
        existing = _section_items(questions.get(qtype))
        for index, item in enumerate(existing):
            if not item.get("slot_id") and index < len(ids):
                item["slot_id"] = ids[index]

    def missing_for(qtype: str) -> list[dict[str, Any]]:
        present = {
            str(item.get("slot_id"))
            for item in _section_items(questions.get(qtype))
            if item.get("slot_id")
        }
        return [
            item for item in normalized_plan.get(qtype, [])
            if item["slot_id"] not in present
        ]

    global_plan_by_slot = {
        str(item.get("slot_id")): item
        for values in normalized_plan.values()
        for item in values
        if item.get("slot_id")
    }

    def slot_context(items: list[dict[str, Any]]) -> str:
        return _context_for_items(items, retrieved_chunks or [], context)

    for _pass in range(max_passes):
        # --- Remove extras first (cheap, immediate) ---
        for qtype in list(questions.keys()):
            want = target.get(qtype, 0)
            got = _section_count(qtype, questions[qtype])
            extras = got - want
            if extras > 0:
                questions[qtype] = _remove_extras(qtype, questions[qtype], extras, warnings)
                within_model[:] = _reindex_within(questions)

        missing = {
            qtype: missing_for(qtype)
            for qtype, want in target.items()
            if want > 0
        }
        missing = {qtype: items for qtype, items in missing.items() if items}
        if not missing:
            break

        obj_planned = {
            qtype: missing[qtype]
            for qtype in ("mcq", "true_false")
            if qtype in missing
        }
        if obj_planned:
            objective_items = [item for values in obj_planned.values() for item in values]
            bundle, bwarn = _generate_obj_bundle(
                obj_planned, slot_context(objective_items), difficulty, model_number,
                within_model, seen, previous_questions, language=language,
                eval_stats=eval_stats,
                global_plan_by_slot=global_plan_by_slot,
                mcq_task_replacer=mcq_task_replacer,
            )
            warnings.extend(bwarn)
            for q in obj_planned:
                section = bundle.get(q)
                if not section:
                    continue
                for index, item in enumerate(_section_items(section)):
                    if not item.get("slot_id") and index < len(obj_planned[q]):
                        item["slot_id"] = obj_planned[q][index]["slot_id"]
                before = _section_count(q, questions.get(q))
                new_section = _append_section(questions.get(q), q, section)
                questions[q] = new_section
                if appended_by_type is not None:
                    appended_by_type[q] = appended_by_type.get(q, 0) + max(
                        _section_count(q, new_section) - before, 0
                    )
            within_model[:] = _reindex_within(questions)

        if "fill_in_the_blank" in missing:
            q = "fill_in_the_blank"
            old = questions.get(q)
            new_section, twarn = _generate_fitb_type(
                len(missing[q]), missing[q], slot_context(missing[q]), difficulty, model_number,
                within_model, previous_questions, language=language,
                eval_stats=eval_stats,
                global_plan_by_slot=global_plan_by_slot,
            )
            warnings.extend(twarn)
            if new_section:
                before = _section_count(q, questions.get(q))
                questions[q] = _append_section(old, q, new_section)
                if appended_by_type is not None:
                    appended_by_type[q] = appended_by_type.get(q, 0) + max(
                        _section_count(q, questions[q]) - before, 0
                    )
                within_model[:] = _reindex_within(questions)

        for q in ("definition", "why", "equation", "word_problem", "essay"):
            if q not in missing:
                continue
            planned = missing[q]
            new_questions, twarn = _generate_type_from_plan(
                q, planned, slot_context(planned), difficulty, model_number,
                seen, within_model, previous_questions, language=language,
                eval_stats=eval_stats,
                global_plan_by_slot=global_plan_by_slot,
            )
            warnings.extend(twarn)
            added = new_questions
            for index, item in enumerate(added):
                if not item.get("slot_id") and index < len(planned):
                    item["slot_id"] = planned[index]["slot_id"]
            questions.setdefault(q, [])
            questions[q] = list(questions[q]) + added
            if appended_by_type is not None:
                appended_by_type[q] = appended_by_type.get(q, 0) + len(added)
            within_model[:] = _reindex_within(questions)

    # Exhausted FITB stem repair is slot-local. Replace only the still-missing
    # entry, then give that replacement the normal bounded FITB generation.
    remaining_fitb = missing_for("fill_in_the_blank")
    if remaining_fitb and fitb_entry_replacer is not None:
        valid_items = list(_section_items(questions.get("fill_in_the_blank")))
        for failed_plan in remaining_fitb:
            sid = str(failed_plan.get("slot_id") or "")
            replacement = fitb_entry_replacer(
                sid,
                failed_plan,
                "Assigned FITB entry exhausted bounded stem generation with invalid_structure",
            )
            if replacement is None:
                continue
            normalized_plan["fill_in_the_blank"] = [
                replacement if str(item.get("slot_id") or "") == sid else item
                for item in normalized_plan.get("fill_in_the_blank", [])
            ]
            global_plan_by_slot[sid] = replacement
            section, twarn = _generate_fitb_type(
                1, [replacement], slot_context([replacement]), difficulty, model_number,
                [*within_model, *valid_items], previous_questions, language=language,
                eval_stats=eval_stats, global_plan_by_slot=global_plan_by_slot,
            )
            warnings.extend(twarn)
            if section and _section_items(section):
                questions["fill_in_the_blank"] = _append_section(
                    questions.get("fill_in_the_blank"), "fill_in_the_blank", section
                )
                valid_items = list(_section_items(questions["fill_in_the_blank"]))
                within_model[:] = _reindex_within(questions)

    for qtype, want in target.items():
        present = {
            str(item.get("slot_id"))
            for item in _section_items(questions.get(qtype))
            if item.get("slot_id")
        }
        missing_ids = [
            f"m{model_number}_{qtype}_{index}"
            for index in range(1, want + 1)
            if f"m{model_number}_{qtype}_{index}" not in present
        ]
        if missing_ids:
            warnings.append(
                f"{qtype}: still short ({want - len(missing_ids)}/{want}) after repair; "
                f"missing slot IDs: {', '.join(missing_ids)}"
            )
    return warnings


def _append_section(existing: Any, qtype: str, new_section: Any) -> Any:
    """Concatenate a newly generated objective section onto the existing one."""
    if qtype == "fill_in_the_blank":
        old = existing if isinstance(existing, dict) else {"word_bank": [], "items": []}
        items = list(old.get("items") or []) + list((new_section or {}).get("items") or [])
        bank = _synchronize_fitb_shared_bank(items)
        random.shuffle(bank)
        return {"word_bank": bank, "items": items}
    return list(existing or []) + list(new_section or [])


def _reindex_within(questions: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the flattened within-model list from the current sections."""
    acc: list[dict[str, Any]] = []
    for qtype, section in questions.items():
        if qtype == "fill_in_the_blank":
            acc.extend((section or {}).get("items") or [])
        else:
            acc.extend(section or [])
    return acc


def _build_feedback(
    qtype: str,
    accumulated: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
) -> str | None:
    """Describe accepted questions for the retry so the model only fills the gap."""
    parts: list[str] = []
    accepted = [*accumulated, *within_model]
    if accepted:
        lines = "\n".join(
            f"- {question_text(qtype, q)}" for q in accepted if question_text(qtype, q)
        )
        parts.append(
            "These questions were already accepted in this exam. "
            f"Do NOT repeat or reword them:\n{lines}"
        )
    return "\n\n".join(parts) if parts else None


# --------------------------------------------------------------------------
# Node: generate every exam model (one call per model x type)
# --------------------------------------------------------------------------
def generate_exams_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate each model from grounded plans while retaining slot identity."""

    document_id = state.get("document_id")
    tasks = state.get("tasks") or []
    num_models = state.get("num_models") or 1
    difficulty = state.get("difficulty") or "mix"
    context = state.get("context") or ""
    language = state.get("document_language") or "en"
    plans = state.get("plans") or []
    assessment_profile = state.get("assessment_profile") or {}
    planning_issues = state.get("planning_issues") or []
    retrieved_chunks = state.get("retrieved_chunks") or []
    catalog = state.get("assessment_catalog") or {}
    warnings: list[str] = list(state.get("warnings") or [])
    eval_stats = state.get("eval_stats")

    from app.online.planner import plan_structure_errors, validate_plans
    planning_gate_errors = [
        *(state.get("plan_errors") or []),
        *plan_structure_errors(plans, tasks, int(num_models)),
        *validate_plans(
            plans, tasks, int(num_models),
            {
                str(chunk.get("child_id")) for chunk in retrieved_chunks
                if chunk.get("child_id")
            } or None,
            difficulty, assessment_profile, planning_issues,
        ),
    ]
    if state.get("unresolved_plan_duplicate_conflicts"):
        from app.online.plan_review import _is_duplicate_plan_error
        planning_gate_errors = [
            error for error in planning_gate_errors
            if not _is_duplicate_plan_error(error)
        ]
        warnings.append(
            "Plan duplicate diversification was exhausted; generation continued with "
            "auditable unresolved plan similarity for the final Duplicate Judge."
        )
    if planning_gate_errors:
        raise RuntimeError(
            "Generator refused an invalid reviewed plan: "
            + "; ".join(dict.fromkeys(str(error) for error in planning_gate_errors))
        )

    if planning_issues:
        warnings.extend(
            f"Unsupported request for {issue.get('question_type')}: {issue.get('reason')}"
            for issue in planning_issues
        )

    generated_exams: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    previous_questions: list[dict[str, Any]] = []
    mcq_task_reassignments: list[dict[str, Any]] = list(
        state.get("mcq_task_reassignments") or []
    )
    fitb_recovery_history: list[dict[str, Any]] = list(
        state.get("fitb_recovery_history") or []
    )

    t0 = time.perf_counter()
    for model_number in range(1, num_models + 1):
        plan = plans[model_number - 1] if model_number - 1 < len(plans) else None
        raw_plan_items = (plan or {}).get("items") or {}
        plan_items = _attach_plan_review_context({
            qtype: _generation_plan_items(
                list(items or []), qtype, retrieved_chunks, assessment_profile
            )
            for qtype, items in raw_plan_items.items()
        })
        global_plan_by_slot = {
            str(item.get("slot_id")): item
            for values in plan_items.values()
            for item in values
            if item.get("slot_id")
        }
        model_warnings: list[str] = []
        questions: dict[str, Any] = {}
        within_model: list[dict[str, Any]] = []
        unsuitable_task_ids_by_slot: dict[str, set[str]] = {}
        exhausted_fitb_entries_by_slot: dict[str, set[str]] = {}

        def replace_unsuitable_mcq_task(
            slot_id: str, current_item: dict[str, Any], reason: str
        ) -> dict[str, Any] | None:
            excluded = unsuitable_task_ids_by_slot.setdefault(slot_id, set())
            current_task_id = str(current_item.get("task_id") or "")
            if current_task_id:
                excluded.add(current_task_id)
            replacement = catalog_exhaustion_replacement(
                catalog, plans, slot_id, "mcq", str(difficulty),
                excluded_task_ids=excluded,
                active_slot_ids=set(global_plan_by_slot),
            ) if catalog else None
            event = {
                "model_number": model_number,
                "slot_id": slot_id,
                "failure_code": "mcq_task_unsuitable",
                "reason": reason,
                "old_task_id": current_task_id or None,
                "new_task_id": (replacement or {}).get("task_id"),
                "status": "reassigned" if replacement else "no_compatible_replacement",
            }
            mcq_task_reassignments.append(event)
            if replacement is None:
                return None
            for model_plan in plans:
                if int(model_plan.get("model_number") or 0) != model_number:
                    continue
                for values in (model_plan.get("items") or {}).values():
                    for index, item in enumerate(values or []):
                        if str(item.get("slot_id") or "") == slot_id:
                            values[index] = replacement
            reused_task_id = str(replacement.get("task_id") or "")
            if replacement.get("task_reused") and reused_task_id:
                plan_lookup = {
                    str(item.get("slot_id") or ""): item
                    for values in plan_items.values() for item in values
                }
                previous_task_questions = []
                for question in within_model:
                    previous_plan = plan_lookup.get(str(question.get("slot_id") or "")) or {}
                    if str(previous_plan.get("task_id") or "") != reused_task_id:
                        continue
                    previous_task_questions.append({
                        "question_id": question.get("question_id"),
                        "slot_id": question.get("slot_id"),
                        "question_text": question.get("question")
                        or question.get("statement") or question.get("term")
                        or question.get("equation"),
                        "idea_to_test": previous_plan.get("idea_to_test"),
                        "reasoning_mode": previous_plan.get("reasoning_mode"),
                        "scenario_family": previous_plan.get("scenario_family"),
                        "answer_pattern": question.get("correct_answer")
                        or question.get("answer") or question.get("answers")
                        or question.get("reference_answer") or question.get("final_answer"),
                    })
                replacement["previous_task_questions"] = previous_task_questions
            enriched = _generation_plan_items(
                [replacement], "mcq", retrieved_chunks, assessment_profile
            )[0]
            plan_items["mcq"] = [
                enriched if str(item.get("slot_id") or "") == slot_id else item
                for item in plan_items.get("mcq") or []
            ]
            return enriched

        def replace_exhausted_fitb_entry(
            slot_id: str, current_item: dict[str, Any], reason: str
        ) -> dict[str, Any] | None:
            excluded = exhausted_fitb_entries_by_slot.setdefault(slot_id, set())
            current_entry_id = str(current_item.get("fitb_entry_id") or "")
            if current_entry_id:
                excluded.add(current_entry_id)
            used_answers = {
                normalize_fitb_answer_key(answer)
                for item in _section_items(questions.get("fill_in_the_blank"))
                for answer in (item.get("answers") or [])
                if normalize_fitb_answer_key(answer)
            }
            fitb_recovery_history.append({
                "model_number": model_number,
                "slot_id": slot_id,
                "fitb_entry_id": current_entry_id or None,
                "status": "replacement_pending",
                "failure_code": "invalid_structure",
                "reason": reason,
            })
            replacement = catalog_exhaustion_replacement(
                catalog, plans, slot_id, "fill_in_the_blank", str(difficulty),
                excluded_task_ids=excluded,
                used_fitb_answer_keys=used_answers,
                active_slot_ids=set(global_plan_by_slot),
            ) if catalog else None
            if replacement is None:
                fitb_recovery_history.append({
                    "model_number": model_number,
                    "slot_id": slot_id,
                    "fitb_entry_id": current_entry_id or None,
                    "status": "replacement_exhausted",
                    "failure_code": "fitb_replacement_exhausted",
                })
                return None
            for model_plan in plans:
                if int(model_plan.get("model_number") or 0) != model_number:
                    continue
                for index, item in enumerate(
                    ((model_plan.get("items") or {}).get("fill_in_the_blank") or [])
                ):
                    if str(item.get("slot_id") or "") == slot_id:
                        model_plan["items"]["fill_in_the_blank"][index] = replacement
            enriched = _generation_plan_items(
                [replacement], "fill_in_the_blank", retrieved_chunks, assessment_profile
            )[0]
            plan_items["fill_in_the_blank"] = [
                enriched if str(item.get("slot_id") or "") == slot_id else item
                for item in plan_items.get("fill_in_the_blank") or []
            ]
            fitb_recovery_history.append({
                "model_number": model_number,
                "slot_id": slot_id,
                "fitb_entry_id": enriched.get("fitb_entry_id"),
                "previous_fitb_entry_id": current_entry_id or None,
                "status": "replacement_selected",
                "failure_code": "invalid_structure",
            })
            return enriched

        obj_types = ("mcq", "true_false")
        obj_planned = {q: plan_items.get(q) or [] for q in obj_types}
        if any(obj_planned[q] for q in obj_types):
            obj_items = [item for values in obj_planned.values() for item in values]
            obj_context = _context_for_items(obj_items, retrieved_chunks, context)
            bundle, model_warnings_ = _generate_obj_bundle(
                obj_planned,
                obj_context,
                difficulty,
                model_number,
                within_model,
                seen,
                previous_questions,
                language=language,
                eval_stats=eval_stats,
                global_plan_by_slot=global_plan_by_slot,
                mcq_task_replacer=replace_unsuitable_mcq_task,
            )
            model_warnings.extend(model_warnings_)
            for section_key in obj_types:
                section_val = bundle.get(section_key)
                if not section_val:
                    continue
                questions[section_key] = section_val
                _assign_ids(model_number, section_key, section_val)
                if section_key == "fill_in_the_blank":
                    within_model.extend(section_val.get("items") or [])
                else:
                    within_model.extend(section_val)

        # The catalog planner has already reserved stable slots and tasks. Each
        # remaining question type is therefore independent at generation time.
        # Give every call an immutable snapshot of the already accepted MCQ/TF
        # context, then merge results in request order so IDs and output ordering
        # remain deterministic regardless of completion order.
        base_within_model = deepcopy(within_model)
        base_seen = set(seen)
        base_previous_questions = deepcopy(previous_questions)
        generation_jobs: list[tuple[str, list[dict[str, Any]], str]] = []
        for qtype, _count in tasks:
            if qtype in obj_types:
                continue
            planned = plan_items.get(qtype) or []
            if not planned:
                model_warnings.append(f"{qtype}: no planned concepts; skipped.")
                continue
            generation_jobs.append((
                qtype,
                planned,
                _context_for_items(planned, retrieved_chunks, context),
            ))

        def generate_independent_type(
            qtype: str, planned: list[dict[str, Any]], type_context: str,
        ) -> tuple[Any, list[str]]:
            local_within = deepcopy(base_within_model)
            local_previous = deepcopy(base_previous_questions)
            if qtype == "fill_in_the_blank":
                section, type_warnings = _generate_fitb_type(
                    len(planned), planned, type_context, difficulty, model_number,
                    local_within, local_previous, language=language,
                    eval_stats=eval_stats,
                    global_plan_by_slot=global_plan_by_slot,
                )
                return section or {"word_bank": [], "items": []}, type_warnings
            return _generate_type_from_plan(
                qtype,
                planned,
                type_context,
                difficulty,
                model_number,
                set(base_seen),
                local_within,
                local_previous,
                language=language,
                eval_stats=eval_stats,
                global_plan_by_slot=global_plan_by_slot,
            )

        generated_by_type: dict[str, tuple[Any, list[str]]] = {}
        if generation_jobs:
            worker_count = min(MAX_PARALLEL_GENERATION_CALLS, len(generation_jobs))
            logger.info(
                "Parallel type generation | model=%d | calls=%d | max_workers=%d | types=%s",
                model_number, len(generation_jobs), worker_count,
                [qtype for qtype, _planned, _context in generation_jobs],
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        copy_context().run,
                        generate_independent_type,
                        qtype,
                        planned,
                        type_context,
                    ): qtype
                    for qtype, planned, type_context in generation_jobs
                }
                for future in as_completed(futures):
                    qtype = futures[future]
                    try:
                        generated_by_type[qtype] = future.result()
                    except Exception as exc:
                        logger.exception(
                            "Parallel question generation failed | model=%d | type=%s",
                            model_number, qtype,
                        )
                        empty = ({"word_bank": [], "items": []}
                                 if qtype == "fill_in_the_blank" else [])
                        generated_by_type[qtype] = (
                            empty, [f"{qtype}: generation failed ({exc})"]
                        )

        for qtype, _planned, _type_context in generation_jobs:
            section, type_warnings = generated_by_type[qtype]
            questions[qtype] = section
            _assign_ids(model_number, qtype, section)
            model_warnings.extend(type_warnings)
            section_items = _section_items(section)
            within_model.extend(section_items)
            for item in section_items:
                _remember_question(qtype, item, seen)

        # Snapshot accepted output from the original generation calls before
        # count trimming/filling. Shortfall generation must not improve this.
        if eval_stats is not None:
            record_initial_generation(eval_stats, model_number, questions)

        fitb_present_before_recovery = {
            str(item.get("slot_id") or "")
            for item in _section_items(questions.get("fill_in_the_blank"))
        }
        for item in plan_items.get("fill_in_the_blank") or []:
            sid = str(item.get("slot_id") or "")
            fitb_recovery_history.append({
                "model_number": model_number,
                "slot_id": sid,
                "fitb_entry_id": item.get("fitb_entry_id"),
                "status": "generated" if sid in fitb_present_before_recovery else "invalid",
                "failure_code": None if sid in fitb_present_before_recovery else "invalid_structure",
            })
            if sid not in fitb_present_before_recovery:
                fitb_recovery_history.append({
                    "model_number": model_number,
                    "slot_id": sid,
                    "fitb_entry_id": item.get("fitb_entry_id"),
                    "status": "repair_pending",
                    "failure_code": "invalid_structure",
                })

        # Minimal-diff repair: missing -> generate only the missing amount,
        # extra -> remove only the extra amount, correct -> untouched.
        shortfall_appended = {qtype: 0 for qtype, _count in tasks}
        model_context = _context_for_items(
            [item for values in plan_items.values() for item in values],
            retrieved_chunks,
            context,
        )
        repair_warnings = _repair_shortfalls(
            questions,
            tasks,
            plan_items,
            model_context,
            difficulty,
            model_number,
            seen,
            within_model,
            previous_questions,
            language=language,
            appended_by_type=shortfall_appended,
            eval_stats=eval_stats,
            retrieved_chunks=retrieved_chunks,
            mcq_task_replacer=replace_unsuitable_mcq_task,
            fitb_entry_replacer=replace_exhausted_fitb_entry,
        )
        model_warnings.extend(repair_warnings)
        recovered_fitb_slots = {
            str(item.get("slot_id") or "")
            for item in _section_items(questions.get("fill_in_the_blank"))
        }
        for sid in sorted(recovered_fitb_slots - fitb_present_before_recovery):
            recovered = global_plan_by_slot.get(sid) or {}
            fitb_recovery_history.append({
                "model_number": model_number,
                "slot_id": sid,
                "fitb_entry_id": recovered.get("fitb_entry_id"),
                "status": "generated",
                "failure_code": None,
            })
        for section_key, section_val in questions.items():
            _assign_ids(model_number, section_key, section_val)
        balance_mcq_answer_positions(questions, model_number)
        if eval_stats is not None:
            record_shortfall_result(
                eval_stats, model_number, questions, shortfall_appended
            )

        elapsed = time.perf_counter() - t0
        total = sum(
            len(v["items"]) if qtype == "fill_in_the_blank" else len(v)
            for qtype, v in questions.items()
        )
        logger.info(
            "Exam generated | document_id=%s | model=%d | types=%s | total=%d | time=%.2fs",
            document_id,
            model_number,
            list(questions.keys()),
            total,
            elapsed,
        )
        generated_exams.append(
            {
                "model_number": model_number,
                "questions": questions,
                "warnings": model_warnings,
            }
        )
        previous_questions.extend(within_model)

    warnings.extend(generated_exams[-1]["warnings"] if generated_exams else [])
    return {
        "generated_exams": generated_exams,
        "plans": plans,
        "mcq_task_reassignments": mcq_task_reassignments,
        "fitb_recovery_history": fitb_recovery_history,
        "warnings": warnings,
        "eval_stats": eval_stats,
    }


def assemble_exams_node(state: dict[str, Any]) -> dict[str, Any]:
    """Finalize audit data, then expose only validator-PASS questions."""
    language = state.get("document_language") or "en"
    generated_exams = state.get("generated_exams") or []
    reports = state.get("validation_reports") or []
    eval_stats = state.get("eval_stats")
    if eval_stats is not None:
        finalize_pipeline_eval(eval_stats, generated_exams, reports)

    report_by_model = {
        int(report.get("model_number") or 0): report for report in reports
    }
    invalid_questions: list[dict[str, Any]] = list(state.get("invalid_questions") or [])
    repair_history_by_question: dict[str, list[dict[str, Any]]] = {}
    for event in state.get("validation_repair_history") or []:
        qid = str(event.get("question_id") or "")
        if qid:
            repair_history_by_question.setdefault(qid, []).append(event)
    for exam in generated_exams:
        model_number = int(exam.get("model_number") or 1)
        verdict_by_id = {
            str(verdict.get("question_id")): verdict
            for verdict in (report_by_model.get(model_number) or {}).get("verdicts") or []
        }
        public_questions: dict[str, Any] = {}
        for qtype, section in (exam.get("questions") or {}).items():
            kept: list[dict[str, Any]] = []
            for item in _section_items(section):
                qid = str(item.get("question_id") or "")
                verdict = verdict_by_id.get(qid) or {}
                if verdict.get("action") == "PASS":
                    kept.append(item)
                else:
                    repair_events = repair_history_by_question.get(qid) or []
                    invalid_questions.append({
                        "model_number": model_number,
                        "question_type": qtype,
                        "slot_id": item.get("slot_id"),
                        "question_id": qid,
                        "failure_code": verdict.get("failure_code") or "unvalidated",
                        "failure_stage": "repair" if repair_events else "validation",
                        "reason": verdict.get("reason") or "Question did not receive a PASS verdict.",
                        "question": deepcopy(item),
                    })
            if qtype == "fill_in_the_blank":
                if kept:
                    public_section = {
                        "word_bank": list((section or {}).get("word_bank") or []),
                        "items": kept,
                    }
                    from app.online.validator import (
                        _fitb_word_bank_valid,
                        _rebuild_fitb_word_bank,
                    )
                    bank_aligned = _fitb_word_bank_valid(public_section)
                    # PASS filtering can remove one FITB slot. Rebuild from the
                    # surviving active questions instead of treating a stale,
                    # otherwise well-formed pre-filter bank as a section failure.
                    if not bank_aligned and not _rebuild_fitb_word_bank(public_section):
                        invalid_questions.extend({
                            "model_number": model_number,
                            "question_type": qtype,
                            "slot_id": item.get("slot_id"),
                            "question_id": item.get("question_id"),
                            "failure_code": "ambiguous_or_structural",
                            "failure_stage": "finalization",
                            "reason": "Final FITB Word Bank could not be rebuilt with two distractors.",
                            "question": deepcopy(item),
                        } for item in kept)
                    else:
                        public_questions[qtype] = public_section
            elif kept:
                public_questions[qtype] = kept
        exam["questions"] = public_questions
        exam["markdown"] = assemble_exam(exam["questions"], language=language)
        exam["document_language"] = language
    return {
        "generated_exams": generated_exams,
        "invalid_questions": invalid_questions,
        "eval_stats": eval_stats,
    }


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def _missing_output_slots(
    exams: list[dict[str, Any]],
    tasks: list[tuple[str, int]],
    num_models: int,
) -> list[str]:
    """Return expected stable slot IDs absent from the final exam output."""
    present: set[str] = set()
    for exam in exams:
        for section in (exam.get("questions") or {}).values():
            present.update(
                str(item.get("slot_id"))
                for item in _section_items(section)
                if item.get("slot_id")
            )
    expected = [
        f"m{model_number}_{qtype}_{index}"
        for model_number in range(1, num_models + 1)
        for qtype, count in tasks
        for index in range(1, count + 1)
    ]
    return [sid for sid in expected if sid not in present]


def generate_exams(
    document_id: str,
    tasks: list[tuple[str, int]],
    num_models: int,
    selected_child_ids: list[str] | None = None,
    difficulty: str = "mix",
    *,
    session_id: str | None = None,
    index_job_id: str | None = None,
    processed_document_key: str | None = None,
    domain_type: str | None = None,
    subject: str | None = None,
    progress_callback=None,
    audit_callback=None,
) -> dict[str, Any]:
    """Generate ``num_models`` complete, distinct exam versions.

    Runs the two-phase LangGraph workflow ONCE: retrieval, planning (one call for
    all models), then plan-driven generation per model x question type. Returns
    exams, warnings, detected language, and the JSON-safe pipeline eval summary.
    """
    t0 = time.perf_counter()
    from app.online.lifecycle import initialize_slot_lifecycle
    initial = {
        "document_id": document_id,
        "tasks": list(tasks),
        "num_models": num_models,
        "selected_child_ids": selected_child_ids,
        "session_id": session_id,
        "index_job_id": index_job_id,
        "processed_document_key": processed_document_key,
        "domain_type": str(domain_type or ""),
        "subject": str(subject or ""),
        "difficulty": difficulty,
        "document_language": "en",
        "retrieved_chunks": [],
        "context": "",
        "planner_context": "",
        "assessment_catalog": {},
        "assessment_catalog_key": "",
        "assessment_catalog_cache_hit": False,
        "assessment_compiler_cache_hit": False,
        "plans": [],
        "assessment_profile": {},
        "planning_issues": [],
        "plan_errors": [],
        "plan_attempts": 0,
        "unrepairable_plan_slots": [],
        "plan_review_history": [],
        "plan_review_retry_required": False,
        "plan_review_rechecks": 0,
        "plan_duplicate_repair_attempts": {},
        "unresolved_plan_duplicate_conflicts": [],
        "difficulty_adjustment_history": [],
        "difficulty_adjustment_attempts": {},
        "plan_review_attempts": 0,
        "generated_exams": [],
        "rejection_feedback": None,
        "mcq_task_reassignments": [],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "repair_manifest": [],
        "repair_candidates": [],
        "repair_retry_seeds": {},
        "repair_replacement_memory": {},
        "pending_revalidation_ids": {},
        "invalid_questions": [],
        "duplicate_repair_history": [],
        "validation_repair_history": [],
        "initial_global_reviews": [],
        "final_global_reviews": [],
        "exam_quality_findings": [],
        "duplicate_reports": [],
        "duplicate_similarity_distributions": [],
        "pending_duplicate_groups": [],
        "duplicate_repair_rounds": 0,
        "duplicate_judge_exempt_ids": [],
        "duplicate_numeric_fallback_pending_validation": False,
        "duplicate_repair_finalization_pending": False,
        "duplicate_verification_pending": False,
        "duplicate_verification_attempt": 0,
        "duplicate_replacement_attempts": {},
        "duplicate_candidate_blacklist": {},
        "duplicate_replacement_failures": {},
        "unresolved_duplicate_conflicts": [],
        "duplicate_scan_count": 0,
        "duplicate_fact_cache": {},
        "slot_lifecycle": initialize_slot_lifecycle(tasks, num_models),
        "developer_diagnostics": [],
        "eval_stats": create_pipeline_eval(tasks, num_models),
        "warnings": [],
        "error": None,
        "progress_callback": progress_callback,
        "audit_callback": audit_callback,
    }
    result = get_exam_graph().invoke(initial)

    error = result.get("error")
    if error:
        logger.warning("Exam workflow error | document_id=%s | error=%s", document_id, error)
        return {
            "exams": [],
            "warnings": [str(error)],
            "assessment_profile": result.get("assessment_profile") or {},
            "planning_issues": result.get("planning_issues") or [],
            "eval": public_eval(result.get("eval_stats")),
            "complete": False,
            "status": "failed",
            "missing_slot_ids": [
                f"m{model_number}_{qtype}_{index}"
                for model_number in range(1, num_models + 1)
                for qtype, count in tasks
                for index in range(1, count + 1)
            ],
        }

    exams = result.get("generated_exams") or []
    warnings = result.get("warnings") or []
    invalid_questions = result.get("invalid_questions") or []
    slot_lifecycle = result.get("slot_lifecycle") or {}
    if not slot_lifecycle:
        from app.online.lifecycle import reconcile_slot_lifecycle
        slot_lifecycle = reconcile_slot_lifecycle(
            initial["slot_lifecycle"], result, stage="finalization"
        )
        invalid_ids = {
            str(item.get("slot_id") or "") for item in result.get("invalid_questions") or []
        }
        for exam in result.get("generated_exams") or []:
            for section in (exam.get("questions") or {}).values():
                for item in _section_items(section):
                    slot_id = str(item.get("slot_id") or "")
                    if slot_id in slot_lifecycle and slot_id not in invalid_ids:
                        slot_lifecycle[slot_id].update({
                            "generated": True, "validated": True, "finalized": True,
                            "terminal_status": "finalized", "failure_code": None,
                            "failure_stage": None,
                        })
    unrepairable_plan_slots = result.get("unrepairable_plan_slots") or []
    invalid_slot_ids = sorted(
        slot_id for slot_id, record in slot_lifecycle.items()
        if record.get("terminal_status") not in {None, "finalized"}
        and record.get("generated")
    )
    absent_from_public = _missing_output_slots(exams, tasks, num_models)
    missing_slot_ids = [sid for sid in absent_from_public if sid not in invalid_slot_ids]
    requested_count = sum(count for _qtype, count in tasks) * num_models
    final_question_count = sum(
        1 for exam in exams for section in (exam.get("questions") or {}).values()
        for _item in _section_items(section)
    )
    deficit = max(0, requested_count - final_question_count)
    quality_status = "complete" if deficit == 0 else "incomplete"
    complete = quality_status == "complete"
    if deficit:
        warnings = [
            *warnings,
            f"Exam quality is {quality_status}; omitted {len(invalid_slot_ids)} invalid and "
            f"{len(missing_slot_ids)} missing question(s).",
        ]
    quality_summary = {
        "requested": requested_count,
        "valid": final_question_count,
        "missing": len(missing_slot_ids),
        "invalid": len(invalid_slot_ids),
        "status": quality_status,
    }
    unresolved_slots = [
        deepcopy(item) for item in (result.get("repair_manifest") or [])
        if item.get("status") != "resolved"
    ]
    semantic_identities = [
        {
            "model_number": plan.get("model_number"),
            "question_type": qtype,
            "slot_id": item.get("slot_id"),
            "task_id": item.get("task_id"),
            "fitb_entry_id": item.get("fitb_entry_id"),
            "task_family_id": item.get("task_family_id"),
            "coverage_id": item.get("coverage_id"),
            "concept_id": item.get("concept_id"),
            "relationship_id": item.get("relationship_id"),
            "concept": item.get("concept"),
            "knowledge_focus": item.get("knowledge_focus"),
            "student_work_product": item.get("student_work_product"),
            "task_operation": item.get("task_operation"),
            "relationship": item.get("relationship") or item.get("concept"),
            "target": item.get("target"),
            "reasoning_mode": item.get("reasoning_mode"),
            "difficulty": item.get("difficulty"),
            "requested_difficulty": item.get("requested_difficulty"),
            "actual_difficulty": item.get("actual_difficulty") or item.get("difficulty"),
            "base_catalog_task_id": item.get("base_catalog_task_id"),
            "base_task_family_id": item.get("base_task_family_id"),
            "difficulty_adaptation": item.get("difficulty_adaptation"),
            "difficulty_fallback_reason": item.get("difficulty_fallback_reason"),
            "reasoning_depth": item.get("reasoning_depth"),
            "reasoning_requirements": list(item.get("reasoning_requirements") or []),
            "prerequisites": list(item.get("prerequisites") or []),
            "derived_task_id": item.get("derived_task_id") or item.get("task_id"),
            "selected_replacement_task_id": item.get("selected_replacement_task_id"),
            "task_was_previously_used": bool(item.get("task_was_previously_used")),
            "adjustment_attempts": int(item.get("adjustment_attempts") or 0),
            "adjustment_failure_reason": item.get("adjustment_failure_reason"),
            "source_chunk_id": item.get("source_chunk_id"),
            "support_chunk_ids": list(item.get("support_chunk_ids") or []),
            "given_conditions": list(item.get("given_conditions") or []),
            "scenario_family": item.get("scenario_family"),
            "task_reused": bool(item.get("task_reused")),
            "reuse_reason": item.get("reuse_reason"),
            "catalog_replacement_stage": item.get("catalog_replacement_stage"),
            "fitb_answer_reused": bool(item.get("fitb_answer_reused")),
            "fitb_entry_reused": bool(item.get("fitb_entry_reused")),
            "normalized_fitb_answer": item.get("normalized_fitb_answer"),
            "clue_basis": item.get("clue_basis"),
            "plan_review_status": item.get("plan_review_status"),
            "plan_review_reason": item.get("plan_review_reason"),
            "plan_review_source": item.get("plan_review_source"),
            "suggested_assessment_angle": item.get("suggested_assessment_angle"),
            "reuse_justification": item.get("reuse_justification"),
            "required_instance_distinctions": list(
                item.get("required_instance_distinctions") or []
            ),
            "selection_rank": dict(item.get("selection_rank") or {}),
        }
        for plan in (result.get("plans") or [])
        for qtype, values in (plan.get("items") or {}).items()
        for item in (values or [])
    ]
    language = result.get("document_language") or "en"
    elapsed = time.perf_counter() - t0
    logger.info(
        "Multi-exam generation | document_id=%s | models=%d | difficulty=%s | language=%s | total_time=%.2fs",
        document_id,
        num_models,
        difficulty,
        language,
        elapsed,
    )
    return {
        "exams": exams,
        "warnings": warnings,
        "assessment_profile": result.get("assessment_profile") or {},
        "planning_issues": result.get("planning_issues") or [],
        "document_language": language,
        "eval": public_eval(result.get("eval_stats")),
        "complete": complete,
        "workflow_status": "completed",
        "status": quality_status,
        "quality_status": quality_status,
        "quality_summary": quality_summary,
        "missing_slot_ids": missing_slot_ids,
        "invalid_slot_ids": invalid_slot_ids,
        "invalid_questions": invalid_questions,
        "slot_lifecycle": slot_lifecycle,
        "developer_diagnostics": list(result.get("developer_diagnostics") or []),
        "mcq_task_reassignments": list(result.get("mcq_task_reassignments") or []),
        "unrepairable_plan_slots": unrepairable_plan_slots,
        "semantic_identities": semantic_identities,
        "duplicate_repair_history": list(result.get("duplicate_repair_history") or []),
        "plan_review_history": list(result.get("plan_review_history") or []),
        "difficulty_adjustment_history": list(result.get("difficulty_adjustment_history") or []),
        "difficulty_adjustment_attempts": dict(result.get("difficulty_adjustment_attempts") or {}),
        "plan_review_attempts": int(result.get("plan_review_attempts") or 0),
        "plan_duplicate_repair_attempts": dict(
            result.get("plan_duplicate_repair_attempts") or {}
        ),
        "unresolved_plan_duplicate_conflicts": list(
            result.get("unresolved_plan_duplicate_conflicts") or []
        ),
        "duplicate_repair_rounds": int(result.get("duplicate_repair_rounds") or 0),
        "duplicate_replacement_attempts": dict(
            result.get("duplicate_replacement_attempts") or {}
        ),
        "validation_repair_history": list(result.get("validation_repair_history") or []),
        "repair_manifest": list(result.get("repair_manifest") or []),
        "repair_replacement_memory": dict(result.get("repair_replacement_memory") or {}),
        "final_unresolved_slots": unresolved_slots,
        "initial_global_reviews": list(result.get("initial_global_reviews") or []),
        "final_global_reviews": list(result.get("final_global_reviews") or []),
        "exam_quality_findings": list(result.get("exam_quality_findings") or []),
        "duplicate_reports": list(result.get("duplicate_reports") or []),
        "duplicate_similarity_distributions": list(
            result.get("duplicate_similarity_distributions") or []
        ),
        "assessment_catalog_key": result.get("assessment_catalog_key") or "",
        "assessment_catalog_cache_hit": bool(result.get("assessment_catalog_cache_hit")),
        "assessment_compiler_cache_hit": bool(result.get("assessment_compiler_cache_hit")),
        "assessment_catalog_usage": dict((result.get("assessment_catalog") or {}).get("compiler_usage") or {}),
    }


def generate_exam(
    document_id: str,
    tasks: list[tuple[str, int]],
    selected_child_ids: list[str] | None = None,
    difficulty: str = "mix",
    model_number: int = 1,
) -> dict[str, Any]:
    """Generate one complete exam (single model). Kept for scripts/tests."""
    result = generate_exams(
        document_id, tasks, num_models=1, selected_child_ids=selected_child_ids, difficulty=difficulty
    )
    exam = result["exams"][0] if result["exams"] else None
    return {
        "questions": (exam or {}).get("questions", {}),
        "warnings": result["warnings"],
        "assessment_profile": result.get("assessment_profile") or {},
        "planning_issues": result.get("planning_issues") or [],
        "complete": result.get("complete", False),
        "status": result.get("status", "failed"),
        "quality_status": result.get("quality_status", result.get("status", "failed")),
        "quality_summary": result.get("quality_summary") or {},
        "missing_slot_ids": result.get("missing_slot_ids", []),
        "invalid_slot_ids": result.get("invalid_slot_ids", []),
    }


def assemble_exam(questions: dict[str, list[dict[str, Any]]], language: str = "en") -> str:
    """Render the final exam in canonical order with continuous numbering.

    The application owns structure: ordering, numbering, and markdown.
    """
    sections: list[str] = []
    counter = 1
    for qtype in TYPE_ORDER:
        qs = questions.get(qtype)
        if not qs:
            continue
        sections.append(render_markdown(qtype, qs, start_index=counter, language=language))
        if qtype == "fill_in_the_blank":
            counter += len(qs.get("items") or [])
        else:
            counter += len(qs)
    return "\n\n---\n\n".join(sections)
