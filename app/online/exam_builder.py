from __future__ import annotations

import random
import time
from typing import Any

from app.llm.client import request_structured
from app.llm.schemas import (
    fitb_items_response_model,
    fitb_word_bank_response_model,
    generation_response_model,
    objective_bundle_response_model,
)
from app.logging_conf import get_logger
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
    ESSAY_SCHEMA,
    MCQ_SCHEMA,
    SHORT_ANSWER_SCHEMA,
    TRUE_FALSE_SCHEMA,
    build_validation_repair_prompt,
)

logger = get_logger("EXAM_BUILDER")

_MAX_ATTEMPTS = 3

# Upper bound on a single generation request's max_output_tokens. Keeps combined
# prompt + output within the model's context window even for large per-type counts.
_MAX_TOKENS = 4000

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
    "short_answer": "short_answer",
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
        str(item.get("source_chunk_id"))
        for item in items
        if item.get("source_chunk_id")
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


def _section_items(section: Any) -> list[dict[str, Any]]:
    """Return the flat question list of a section (FITB unwraps its items)."""
    if isinstance(section, dict):
        return list(section.get("items") or [])
    return list(section or [])


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

    The MCQ options are intentionally excluded: distractors are often generic
    and would inflate the overlap of unrelated questions, while the stem alone
    reliably captures a reworded-same-concept MCQ. For short-answer questions
    the reference answer is a strong signal and is included.
    """
    parts = [question_text(qtype, question)]
    if qtype == "short_answer":
        parts.append(str(question.get("reference_answer") or ""))
    elif qtype == "essay":
        parts.append(str(question.get("reference_answer") or ""))
        parts.extend(str(kp) for kp in (question.get("key_points") or []))
    elif qtype == "fill_in_the_blank":
        parts.extend(str(a) for a in (question.get("answers") or []))
    tokens: list[str] = []
    for part in parts:
        tokens.extend(normalize_text(part))
    return {tok for tok in tokens if tok not in _STOPWORDS}


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
        if _token_overlap(tokens, _content_tokens(qtype, other)) >= _NEAR_DUP_THRESHOLD:
            return True
    return False


# Expected JSON schema per question type, used by structural validation repair.
_SCHEMAS = {
    "mcq": MCQ_SCHEMA,
    "true_false": TRUE_FALSE_SCHEMA,
    "short_answer": SHORT_ANSWER_SCHEMA,
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
            warnings.append(f"{qtype} repair attempt {attempt}: {exc}")
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
            warnings.append(
                f"{qtype} attempt {attempt}: LLM returned no valid questions ({exc})"
            )
            continue

        # Split valid vs structurally-invalid raw items.
        candidates, invalid_raw = split_valid_invalid(qtype, raw_obj)
        _record_invalid_candidates(eval_stats, model_number, qtype, invalid_raw)
        # Structural fix FIRST: send rejected output + schema back, repair only
        # the broken fields (never regenerate a fresh random question).
        if invalid_raw and len(candidates) < len(remaining_plan):
            repaired, repair_warnings = _repair_invalid_items(
                qtype,
                [i for i in invalid_raw if isinstance(i, dict)],
                context,
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
    """Collect non-empty, de-duplicated terms from the model output."""
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    terms: list[str] = []
    for t in raw:
        s = str(t).strip()
        key = " ".join(normalize_text(s))
        if s and key not in seen:
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
    # Keep up to 2 distractors from the model's own extras...
    distractors = rest[-2:]
    # ...and pad with safe generic fillers when fewer than 2 survive.
    for filler in ("process", "system", "device"):
        if len(distractors) >= 2:
            break
        if tuple(normalize_text(filler)) not in seen:
            distractors.append(filler)
            seen.add(tuple(normalize_text(filler)))
    return terms + distractors[:2]


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
        if not (1 <= len(answers) <= 2):
            errors.append(f"FITB item {idx + 1}: has {len(answers)} answers (must be 1-2)")
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
        for other in accepted:
            if _token_overlap(
                _content_tokens("fill_in_the_blank", item),
                _content_tokens("fill_in_the_blank", other),
            ) >= _NEAR_DUP_THRESHOLD:
                errors.append(f"FITB item {idx + 1}: near-duplicate of an accepted question")
                break

    # Exactly 2 distractors: unused Word Bank entries that are not answers to any blank.
    distractors = {w: tuple(normalize_text(w)) for w in word_bank}
    n_distractors = sum(1 for t in distractors.values() if t not in used)
    if n_distractors != 2:
        errors.append(f"expected exactly 2 Word Bank distractors, got {n_distractors}")
    return errors


def _generate_fitb_bank(
    count: int,
    planned_items: list[dict[str, Any]],
    context: str,
    difficulty: str,
    model_number: int,
    language: str = "en",
) -> tuple[list[str] | None, list[str]]:
    """Stage 1: choose the answer terms + exactly 2 distractors.

    The bank alone is validated and auto-fixed in code, so the expensive item
    writing only ever runs against a guaranteed-valid bank.
    """
    from app.llm.factory import create_llm_client
    from app.online.prompts import build_fitb_bank_prompt

    warnings: list[str] = []
    client = create_llm_client()

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        system_prompt, user_prompt = build_fitb_bank_prompt(
            count, context, difficulty=difficulty, model_number=model_number,
            planned_items=planned_items, language=language,
        )
        try:
            raw = request_structured(
                client,
                user_prompt,
                response_model=fitb_word_bank_response_model(count),
                schema_name="fill_in_the_blank_bank",
                agent="generator",
                item_count=count,
                operation="fitb_word_bank",
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=min(_MAX_TOKENS, 1024),
            )
        except Exception as exc:
            warnings.append(f"fill_in_the_blank bank attempt {attempt}: {exc}")
            continue
        if not isinstance(raw, dict):
            warnings.append(f"fill_in_the_blank bank attempt {attempt}: non-object response")
            continue

        terms = _clean_fitb_terms(raw.get("correct_terms"))
        distractors = _clean_fitb_terms(raw.get("distractors"))
        if len(terms) < count:
            warnings.append(
                f"fill_in_the_blank bank attempt {attempt}: got {len(terms)}/{count} terms"
            )
            continue
        bank = _autofit_word_bank(terms + distractors, count)
        if len(bank) != count + 2:
            warnings.append(
                f"fill_in_the_blank bank attempt {attempt}: unfittable bank "
                f"({len(terms)} terms)"
            )
            continue
        return bank, warnings

    return None, warnings


def _generate_fitb_items(
    count: int,
    planned_items: list[dict[str, Any]],
    word_bank: list[str],
    context: str,
    difficulty: str,
    model_number: int,
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
    language: str = "en",
    eval_stats: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Stage 2: fill exact FITB slots using only the fixed Word Bank."""
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
            len(remaining_plan), word_bank, context, difficulty=difficulty,
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
            warnings.append(f"fill_in_the_blank items attempt {attempt}: {exc}")
            continue
        items_raw = raw.get("items") if isinstance(raw, dict) else None
        items_raw = items_raw if isinstance(items_raw, list) else []
        items = [normalize_fitb_item(i) for i in items_raw]
        items = [i for i in items if i is not None]
        for index, item in enumerate(items):
            if not item.get("slot_id") and index < len(remaining_plan):
                item["slot_id"] = remaining_plan[index]["slot_id"]
        rejected_count = len(items_raw) - len(items)
        if rejected_count:
            invalid_fitb = [
                item for item in items_raw if normalize_fitb_item(item) is None
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
) -> tuple[dict[str, Any] | None, list[str]]:
    """Two-stage FITB generation: Word Bank FIRST, then blanks against it.

    Stage 1 produces and auto-fixes the bank; stage 2 writes items restricted
    to that fixed bank. This removes the old single-call failure mode where one
    wrong bank size discarded the entire section.
    """
    bank, warnings = _generate_fitb_bank(
        count, planned_items, context, difficulty, model_number, language=language
    )
    if bank is None:
        warnings.append("fill_in_the_blank: could not produce a valid Word Bank")
        return None, warnings

    random.shuffle(bank)
    items, item_warnings = _generate_fitb_items(
        count, planned_items, bank, context, difficulty, model_number, within_model, previous_exams,
        language=language, eval_stats=eval_stats,
    )
    warnings.extend(item_warnings)
    if not items:
        warnings.append("fill_in_the_blank: no valid items generated")
        return None, warnings
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
) -> tuple[dict[str, Any], list[str]]:
    """Generate MCQ and True/False together while tracking exact pending slots."""
    from app.llm.factory import create_llm_client
    from app.online.prompts import build_obj_bundled_prompt

    warnings: list[str] = []
    client = create_llm_client()
    normalized: dict[str, list[dict[str, Any]]] = {}
    pending: dict[str, dict[str, dict[str, Any]]] = {}
    accepted: dict[str, dict[str, dict[str, Any]]] = {}
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
            warnings.append(f"obj bundle attempt {attempt}: {exc}")
            continue
        if not isinstance(raw, dict):
            warnings.append(f"obj bundle attempt {attempt}: non-object response")
            continue

        for qtype in ("mcq", "true_false"):
            if not pending[qtype]:
                continue
            section_raw = raw.get(qtype) if isinstance(raw, dict) else None
            candidates, invalid_raw = split_valid_invalid(qtype, section_raw)
            _record_invalid_candidates(eval_stats, model_number, qtype, invalid_raw)
            if invalid_raw and len(candidates) < len(pending[qtype]):
                repaired, rwarn = _repair_invalid_items(
                    qtype, [i for i in invalid_raw if isinstance(i, dict)],
                    context, difficulty, model_number, language=language,
                )
                warnings.extend(rwarn)
                candidates.extend(repaired)
            remaining_items = list(pending[qtype].values())
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
                if _filter_one(
                    qtype, q, [*accepted[qtype].values(), *within_model],
                    seen, warnings, attempt, language, eval_stats, model_number,
                ):
                    accepted[qtype][sid] = q
                    pending[qtype].pop(sid, None)

        warnings.append(
            f"obj bundle attempt {attempt}: mcq {len(accepted['mcq'])}/{len(normalized['mcq'])} | "
            f"tf {len(accepted['true_false'])}/{len(normalized['true_false'])}"
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
            bank = list(old.get("word_bank") or []) if isinstance(old, dict) else []
            if bank:
                new_items, twarn = _generate_fitb_items(
                    len(missing[q]), missing[q], bank, slot_context(missing[q]), difficulty,
                    model_number, within_model, previous_questions,
                    language=language, eval_stats=eval_stats,
                )
                new_section = {"word_bank": bank, "items": new_items}
            else:
                new_section, twarn = _generate_fitb_type(
                    len(missing[q]), missing[q], slot_context(missing[q]), difficulty, model_number,
                    within_model, previous_questions, language=language,
                    eval_stats=eval_stats,
                )
            warnings.extend(twarn)
            if new_section:
                before = _section_count(q, questions.get(q))
                if isinstance(old, dict) and bank:
                    questions[q] = {
                        "word_bank": bank,
                        "items": list(old.get("items") or [])
                        + list(new_section.get("items") or []),
                    }
                else:
                    questions[q] = new_section
                if appended_by_type is not None:
                    appended_by_type[q] = appended_by_type.get(q, 0) + max(
                        _section_count(q, questions[q]) - before, 0
                    )
                within_model[:] = _reindex_within(questions)

        for q in ("short_answer", "essay"):
            if q not in missing:
                continue
            planned = missing[q]
            new_questions, twarn = _generate_type_from_plan(
                q, planned, slot_context(planned), difficulty, model_number,
                seen, within_model, previous_questions, language=language,
                eval_stats=eval_stats,
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
        bank = list(old.get("word_bank") or []) + list((new_section or {}).get("word_bank") or [])
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
    retrieved_chunks = state.get("retrieved_chunks") or []
    warnings: list[str] = list(state.get("warnings") or [])
    eval_stats = state.get("eval_stats")

    if state.get("plan_errors"):
        warnings.append("Planner could not produce a fully valid plan; using best effort.")

    generated_exams: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    previous_questions: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    for model_number in range(1, num_models + 1):
        plan = plans[model_number - 1] if model_number - 1 < len(plans) else None
        plan_items = (plan or {}).get("items") or {}
        model_warnings: list[str] = []
        questions: dict[str, list[dict[str, Any]]] = {}
        within_model: list[dict[str, Any]] = []

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

        for qtype, _count in tasks:
            if qtype in obj_types:
                continue
            planned = plan_items.get(qtype) or []
            if not planned:
                model_warnings.append(f"{qtype}: no planned concepts; skipped.")
                continue
            type_context = _context_for_items(planned, retrieved_chunks, context)
            if qtype == "fill_in_the_blank":
                section, type_warnings = _generate_fitb_type(
                    len(planned), planned, type_context, difficulty, model_number,
                    within_model, previous_questions, language=language,
                    eval_stats=eval_stats,
                )
                questions[qtype] = section or {"word_bank": [], "items": []}
            else:
                questions[qtype], type_warnings = _generate_type_from_plan(
                    qtype,
                    planned,
                    type_context,
                    difficulty,
                    model_number,
                    seen,
                    within_model,
                    previous_questions,
                    language=language,
                    eval_stats=eval_stats,
                )
            _assign_ids(model_number, qtype, questions[qtype])
            model_warnings.extend(type_warnings)
            within_model.extend(_section_items(questions[qtype]))

        # Snapshot accepted output from the original generation calls before
        # count trimming/filling. Shortfall generation must not improve this.
        if eval_stats is not None:
            record_initial_generation(eval_stats, model_number, questions)

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
        )
        model_warnings.extend(repair_warnings)
        for section_key, section_val in questions.items():
            _assign_ids(model_number, section_key, section_val)
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
        "warnings": warnings,
        "eval_stats": eval_stats,
    }


def assemble_exams_node(state: dict[str, Any]) -> dict[str, Any]:
    """Render each generated exam's questions to markdown."""
    language = state.get("document_language") or "en"
    generated_exams = state.get("generated_exams") or []
    for exam in generated_exams:
        exam["markdown"] = assemble_exam(exam["questions"], language=language)
        exam["document_language"] = language
    eval_stats = state.get("eval_stats")
    if eval_stats is not None:
        finalize_pipeline_eval(
            eval_stats, generated_exams, state.get("validation_reports") or []
        )
    return {"generated_exams": generated_exams, "eval_stats": eval_stats}


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
) -> dict[str, Any]:
    """Generate ``num_models`` complete, distinct exam versions.

    Runs the two-phase LangGraph workflow ONCE: retrieval, planning (one call for
    all models), then plan-driven generation per model x question type. Returns
    exams, warnings, detected language, and the JSON-safe pipeline eval summary.
    """
    t0 = time.perf_counter()
    initial = {
        "document_id": document_id,
        "tasks": list(tasks),
        "num_models": num_models,
        "selected_child_ids": selected_child_ids,
        "difficulty": difficulty,
        "document_language": "en",
        "retrieved_chunks": [],
        "context": "",
        "planner_context": "",
        "plans": [],
        "plan_errors": [],
        "plan_attempts": 0,
        "generated_exams": [],
        "rejection_feedback": None,
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "eval_stats": create_pipeline_eval(tasks, num_models),
        "warnings": [],
        "error": None,
    }
    result = get_exam_graph().invoke(initial)

    error = result.get("error")
    if error:
        logger.warning("Exam workflow error | document_id=%s | error=%s", document_id, error)
        return {
            "exams": [],
            "warnings": [str(error)],
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
    missing_slot_ids = _missing_output_slots(exams, tasks, num_models)
    complete = not missing_slot_ids
    if missing_slot_ids:
        warnings = [
            *warnings,
            "Exam generation is partial; missing slot IDs: "
            + ", ".join(missing_slot_ids),
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
        "document_language": language,
        "eval": public_eval(result.get("eval_stats")),
        "complete": complete,
        "status": "complete" if complete else "partial",
        "missing_slot_ids": missing_slot_ids,
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
        "complete": result.get("complete", False),
        "status": result.get("status", "failed"),
        "missing_slot_ids": result.get("missing_slot_ids", []),
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
