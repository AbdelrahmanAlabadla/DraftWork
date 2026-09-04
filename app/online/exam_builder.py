from __future__ import annotations

import random
import time
from typing import Any

from app.logging_conf import get_logger
from app.online.eval_stats import (
    create_pipeline_eval,
    finalize_pipeline_eval,
    public_eval,
    record_generation_rejection,
    record_initial_generation,
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

# Jaccard token-overlap above which two questions are treated as near-duplicates.
_NEAR_DUP_THRESHOLD = 0.35

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
    if key in seen:
        return True
    seen.add(key)
    return False


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
    from app.llm.client import LMStudioClient

    if not invalid_raw:
        return [], []
    warnings: list[str] = []
    client = LMStudioClient()
    schema = _SCHEMAS.get(qtype, "")
    items = list(invalid_raw)
    for attempt in range(1, max_attempts + 1):
        system_prompt, user_prompt = build_validation_repair_prompt(
            qtype, items, schema, context, difficulty=difficulty, model_number=model_number,
            language=language,
        )
        try:
            raw = client.chat_json(
                user_prompt, system_prompt=system_prompt,
                temperature=0.3, max_tokens=min(_MAX_TOKENS, 2048),
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
    """Generate one question per planned item, retrying only missing concepts.

    Every generated question maps to a planned item; the generator is NEVER asked
    to invent unplanned concepts. If a retry still cannot fill a slot, the exam is
    simply short for that type (a warning), rather than degrading the planning.
    """
    from app.llm.client import LMStudioClient
    from app.online.prompts import build_prompt

    count = len(planned_items)
    accumulated: list[dict[str, Any]] = []
    warnings: list[str] = []
    client = LMStudioClient()

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        # 1:1 mapping by order: the first `len(accumulated)` plan items are done.
        remaining_plan = planned_items[len(accumulated):]
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

        feedback = _build_feedback(qtype, accumulated, seen, within_model, previous_exams)
        if feedback:
            user_prompt += (
                "\n\n## Context from earlier attempts\n"
                "Below are questions already accepted that you must NOT repeat or "
                "reword. Produce the planned questions that are still missing only.\n"
                f"{feedback}"
            )

        try:
            raw_obj = client.chat_json(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=min(_MAX_TOKENS, max(2048, len(remaining_plan) * 256)),
            )
        except Exception as exc:
            warnings.append(
                f"{qtype} attempt {attempt}: LLM returned no valid questions ({exc})"
            )
            continue

        # Split valid vs structurally-invalid raw items.
        candidates, invalid_raw = split_valid_invalid(qtype, raw_obj)
        record_generation_rejection(
            eval_stats, model_number, qtype, "invalid_structure", len(invalid_raw)
        )
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

        valid: list[dict[str, Any]] = []
        for q in candidates[: len(remaining_plan)]:
            text = question_text(qtype, q)
            if contains_forbidden_phrase(text, language):
                logger.warning("Rejected (forbidden phrase) | type=%s | text=%r", qtype, text[:120])
                record_generation_rejection(
                    eval_stats, model_number, qtype, "forbidden_content"
                )
                continue
            if _is_duplicate(qtype, q, seen):
                logger.warning("Rejected (duplicate) | type=%s | text=%r", qtype, text[:120])
                record_generation_rejection(
                    eval_stats, model_number, qtype, "duplicate"
                )
                continue
            if _is_near_duplicate(
                qtype, q, [*valid, *accumulated, *within_model, *previous_exams]
            ):
                logger.warning("Rejected (near-duplicate) | type=%s | text=%r", qtype, text[:120])
                record_generation_rejection(
                    eval_stats, model_number, qtype, "near_duplicate"
                )
                continue
            valid.append(q)

        accumulated.extend(valid)
        logger.info(
            "Plan type attempt | type=%s | model=%d | attempt=%d/%d | planned=%d | accepted=%d/%d",
            qtype,
            model_number,
            attempt,
            _MAX_ATTEMPTS,
            count,
            len(accumulated),
            count,
        )

    if len(accumulated) < count:
        warnings.append(f"{qtype}: {len(accumulated)}/{count} generated after attempts")
        logger.warning("Plan type incomplete | type=%s | got=%d/%d", qtype, len(accumulated), count)
    return accumulated[: count], warnings


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
    accepted = [*within_model, *previous_exams]

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
    from app.llm.client import LMStudioClient
    from app.online.prompts import build_fitb_bank_prompt

    warnings: list[str] = []
    client = LMStudioClient()

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        system_prompt, user_prompt = build_fitb_bank_prompt(
            count, context, difficulty=difficulty, model_number=model_number,
            planned_items=planned_items, language=language,
        )
        try:
            raw = client.chat_json(
                user_prompt, system_prompt=system_prompt, temperature=0.5,
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
    word_bank: list[str],
    context: str,
    difficulty: str,
    model_number: int,
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
    language: str = "en",
    eval_stats: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Stage 2: write numbered items using ONLY the fixed, shuffled Word Bank."""
    from app.llm.client import LMStudioClient
    from app.online.prompts import build_fitb_items_prompt

    warnings: list[str] = []
    client = LMStudioClient()
    best: list[dict[str, Any]] = []

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        system_prompt, user_prompt = build_fitb_items_prompt(
            count, word_bank, context, difficulty=difficulty, model_number=model_number,
            language=language,
        )
        try:
            raw = client.chat_json(
                user_prompt, system_prompt=system_prompt, temperature=0.5,
                max_tokens=min(_MAX_TOKENS, max(2048, count * 320)),
            )
        except Exception as exc:
            warnings.append(f"fill_in_the_blank items attempt {attempt}: {exc}")
            continue
        items_raw = raw.get("items") if isinstance(raw, dict) else None
        items_raw = items_raw if isinstance(items_raw, list) else []
        items = [normalize_fitb_item(i) for i in items_raw]
        items = [i for i in items if i is not None]
        record_generation_rejection(
            eval_stats,
            model_number,
            "fill_in_the_blank",
            "invalid_structure",
            len(items_raw) - len(items),
        )

        candidate: dict[str, Any] = {"word_bank": list(word_bank), "items": items}
        # Validate against the FULL requested count: with a short item set the
        # extra unused bank entries legitimately inflate the distractor count,
        # so both count/distractor messages are completeness artifacts, not
        # content defects.
        errors = _fitb_errors(candidate, count, within_model, previous_exams)
        content_errors = [
            e for e in errors
            if "count" not in e and "distractors" not in e
        ]
        if not content_errors:
            return items, warnings
        # Count only the concrete bad FITB items, assigning one stable reason
        # per raw candidate. Completeness-only errors are shortfalls, not
        # rejected questions.
        rejected_items: dict[str, str] = {}
        for error in content_errors:
            prefix, _, detail = error.partition(":")
            if not prefix.startswith("FITB item "):
                continue
            category = (
                "forbidden_content"
                if "forbidden phrase" in detail
                else "near_duplicate"
                if "near-duplicate" in detail
                else "invalid_structure"
            )
            rejected_items.setdefault(prefix, category)
        for category in rejected_items.values():
            record_generation_rejection(
                eval_stats, model_number, "fill_in_the_blank", category
            )
        best = items if len(items) > len(best) else best
        warnings.append(
            f"fill_in_the_blank items attempt {attempt}: rejected -> "
            f"{'; '.join(content_errors[:2])}"
        )

    if best:
        warnings.append(f"fill_in_the_blank: keeping best partial set ({len(best)} items)")
    return best, warnings


def _generate_fitb_type(
    count: int,
    planned_items: list[dict[str, Any]],
    context: str,
    difficulty: str,
    model_number: int,
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
    language: str = "en",
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
        count, bank, context, difficulty, model_number, within_model, previous_exams,
        language=language,
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
    """Generate MCQ + True/False + Fill-in-the-Blank in ONE LLM call.

    Returns {"mcq": [...], "true_false": [...], "fill_in_the_blank": {...}}.
    Each section is validated with the same rules used by the single-type path.
    On a partial failure we KEEP the sections that already validated and retry
    ONLY the missing items on the next call, so one rejected MCQ never discards
    the valid True/False and Fill-in-the-Blank content.
    """
    from app.llm.client import LMStudioClient
    from app.online.prompts import build_obj_bundled_prompt

    warnings: list[str] = []
    client = LMStudioClient()
    targets = {
        "mcq": len(planned.get("mcq") or []),
        "true_false": len(planned.get("true_false") or []),
        "fill_in_the_blank": len(planned.get("fill_in_the_blank") or []),
    }

    acc_mcq: list[dict[str, Any]] = []
    acc_tf: list[dict[str, Any]] = []
    fitb: dict[str, Any] | None = None

    def remaining(qtype: str) -> int:
        if qtype == "mcq":
            return targets["mcq"] - len(acc_mcq)
        if qtype == "true_false":
            return targets["true_false"] - len(acc_tf)
        return targets["fill_in_the_blank"] if fitb is None else 0

    def incomplete() -> bool:
        return remaining("mcq") > 0 or remaining("true_false") > 0 or remaining("fill_in_the_blank") > 0

    attempt = 0
    while incomplete() and attempt < _MAX_ATTEMPTS:
        attempt += 1
        # Pass only the still-missing plan items (tail) for each type.
        rem_planned = {
            q: (planned.get(q) or [])[-remaining(q):] if remaining(q) > 0 else []
            for q in ("mcq", "true_false", "fill_in_the_blank")
        }
        bundle_max_tokens = min(
            _MAX_TOKENS,
            max(4096, (remaining("mcq") + remaining("true_false")) * 180),
        )
        feedback = _build_bundle_feedback(
            acc_mcq, acc_tf, fitb, within_model, previous_exams
        )
        system_prompt, user_prompt = build_obj_bundled_prompt(
            rem_planned, context, difficulty=difficulty,
            model_number=model_number, feedback=feedback, language=language,
        )
        try:
            raw = client.chat_json(
                user_prompt, system_prompt=system_prompt, temperature=0.5,
                max_tokens=bundle_max_tokens,
            )
        except Exception as exc:
            warnings.append(f"obj bundle attempt {attempt}: {exc}")
            continue
        if not isinstance(raw, dict):
            warnings.append(f"obj bundle attempt {attempt}: non-object response")
            continue

        # --- fill_in_the_blank section: bank FIRST, then items (two-stage) ---
        if remaining("fill_in_the_blank") > 0:
            fitb_count = targets["fill_in_the_blank"]
            bank, bank_warnings = _generate_fitb_bank(
                fitb_count, planned.get("fill_in_the_blank") or [],
                context, difficulty, model_number, language=language,
            )
            warnings.extend(bank_warnings)
            if bank is not None:
                random.shuffle(bank)
                fitb_items, item_warnings = _generate_fitb_items(
                    fitb_count, bank, context, difficulty, model_number,
                    within_model, previous_exams, language=language,
                    eval_stats=eval_stats,
                )
                warnings.extend(item_warnings)
                if fitb_items:
                    fitb = {"word_bank": bank, "items": fitb_items}

        # --- mcq section (incremental, structural-repair first) ---
        need_mcq = remaining("mcq")
        if need_mcq > 0:
            candidates, invalid_raw = split_valid_invalid("mcq", raw.get("mcq"))
            record_generation_rejection(
                eval_stats, model_number, "mcq", "invalid_structure", len(invalid_raw)
            )
            if invalid_raw and len(candidates) < need_mcq:
                repaired, rwarn = _repair_invalid_items(
                    "mcq", [i for i in invalid_raw if isinstance(i, dict)],
                    context, difficulty, model_number, language=language,
                )
                warnings.extend(rwarn)
                candidates.extend(repaired)
            for q in candidates[:need_mcq]:
                if _filter_one(
                    "mcq", q, [*acc_mcq, *within_model, *previous_exams],
                    seen, warnings, attempt, language, eval_stats, model_number,
                ):
                    acc_mcq.append(q)

        # --- true_false section (incremental, structural-repair first) ---
        need_tf = remaining("true_false")
        if need_tf > 0:
            candidates_tf, invalid_tf = split_valid_invalid("true_false", raw.get("true_false"))
            record_generation_rejection(
                eval_stats, model_number, "true_false", "invalid_structure", len(invalid_tf)
            )
            if invalid_tf and len(candidates_tf) < need_tf:
                repaired_tf, rwarn_tf = _repair_invalid_items(
                    "true_false", [i for i in invalid_tf if isinstance(i, dict)],
                    context, difficulty, model_number, language=language,
                )
                warnings.extend(rwarn_tf)
                candidates_tf.extend(repaired_tf)
            for q in candidates_tf[:need_tf]:
                if _filter_one(
                    "true_false", q, [*acc_tf, *within_model, *previous_exams],
                    seen, warnings, attempt, language, eval_stats, model_number,
                ):
                    acc_tf.append(q)

        warnings.append(
            f"obj bundle attempt {attempt}: mcq {len(acc_mcq)}/{targets['mcq']} | "
            f"tf {len(acc_tf)}/{targets['true_false']} | "
            f"fitb {'ok' if fitb is not None else 'pending'}"
        )

    if incomplete():
        warnings.append("obj bundle: not fully generated after attempts")
    return {
        "mcq": acc_mcq,
        "true_false": acc_tf,
        "fill_in_the_blank": fitb,
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
            eval_stats, model_number, qtype, "forbidden_content"
        )
        return False
    if _is_duplicate(qtype, q, seen):
        warnings.append(f"obj bundle {qtype} attempt {attempt}: duplicate")
        record_generation_rejection(eval_stats, model_number, qtype, "duplicate")
        return False
    if _is_near_duplicate(qtype, q, accepted):
        warnings.append(f"obj bundle {qtype} attempt {attempt}: near-duplicate")
        record_generation_rejection(eval_stats, model_number, qtype, "near_duplicate")
        return False
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
    accepted = [*within_model, *previous_exams]
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
) -> list[str]:
    """Minimal-diff count repair of one exam's questions.

    missing -> generate only the missing amount for that type
    extra   -> remove only the extra amount for that type
    correct -> leave untouched

    Never regenerates a whole exam or a whole already-correct section.
    """
    warnings: list[str] = []
    target = {qtype: count for qtype, count in tasks}

    obj_types = ("mcq", "true_false", "fill_in_the_blank")
    free_types = ("short_answer", "essay")

    for _pass in range(max_passes):
        # --- Remove extras first (cheap, immediate) ---
        for qtype in list(questions.keys()):
            want = target.get(qtype, 0)
            got = _section_count(qtype, questions[qtype])
            extras = got - want
            if extras > 0:
                questions[qtype] = _remove_extras(qtype, questions[qtype], extras, warnings)
                within_model[:] = _reindex_within(questions)

        # --- Compute missing amounts ---
        deficits: list[tuple[str,int]] = []
        for qtype, want in target.items():
            if want <= 0:
                continue
            got = _section_count(qtype, questions.get(qtype))
            if got < want:
                deficits.append((qtype, want - got))
        if not deficits:
            break

        # --- Generate ONLY the missing amount per type ---
        obj_missing = {q: m for q, m in deficits if q in obj_types}
        if obj_missing:
            obj_planned = {
                q: _repair_planned_for(q, m, plan_items, context)
                for q, m in obj_missing.items()
            }
            bundle, bwarn = _generate_obj_bundle(
                obj_planned, context, difficulty, model_number,
                within_model, seen, previous_questions, language=language,
                eval_stats=eval_stats,
            )
            warnings.extend(bwarn)
            for q, m in obj_missing.items():
                section = bundle.get(q)
                if not section:
                    continue
                before = _section_count(q, questions.get(q))
                new_section = _append_section(questions.get(q), q, section)
                questions[q] = new_section
                if appended_by_type is not None:
                    appended_by_type[q] = appended_by_type.get(q, 0) + max(
                        _section_count(q, new_section) - before, 0
                    )
            within_model[:] = _reindex_within(questions)

        for q, m in [(qt, mo) for qt, mo in deficits if qt in free_types]:
            planned = _repair_planned_for(q, m, plan_items, context)
            new_questions, twarn = _generate_type_from_plan(
                q, planned, context, difficulty, model_number,
                seen, within_model, previous_questions, language=language,
                eval_stats=eval_stats,
            )
            warnings.extend(twarn)
            added = new_questions[:m]
            questions.setdefault(q, [])
            questions[q] = list(questions[q]) + added
            if appended_by_type is not None:
                appended_by_type[q] = appended_by_type.get(q, 0) + len(added)
            within_model[:] = _reindex_within(questions)

    # Final report for any persistent shortfall.
    for qtype, want in target.items():
        if want <= 0:
            continue
        got = _section_count(qtype, questions.get(qtype))
        if got < want:
            warnings.append(
                f"{qtype}: still short ({got}/{want}) after repair; exam will be short."
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
    accepted = [*accumulated, *within_model, *previous_exams]
    if accepted:
        lines = "\n".join(
            f"- {question_text(qtype, q)}" for q in accepted if question_text(qtype, q)
        )
        parts.append(
            "These questions were already accepted (in this exam or earlier models). "
            f"Do NOT repeat or reword them; do not test the same concept:\n{lines}"
        )
    return "\n\n".join(parts) if parts else None


# --------------------------------------------------------------------------
# Node: generate every exam model (one call per model x type)
# --------------------------------------------------------------------------
def generate_exams_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate `num_models` complete, distinct exams from the shared plan.

    Cross-model uniqueness is secured primarily by the planner distributing
    distinct concepts; the near-duplicate text check runs across all previously
    generated models as a secondary safeguard.
    """
    from app.online.planner import fill_missing_concepts

    document_id = state.get("document_id")
    tasks = state.get("tasks") or []
    num_models = state.get("num_models") or 1
    difficulty = state.get("difficulty") or "mix"
    context = state.get("context") or ""
    language = state.get("document_language") or "en"
    plans = state.get("plans") or []
    warnings: list[str] = list(state.get("warnings") or [])
    eval_stats = state.get("eval_stats")

    # Never free-fill during generation: first repair any remaining plan shortfall.
    if not state.get("plan_errors"):
        plans, fill_warnings = fill_missing_concepts(
            plans, tasks, state.get("planner_context") or ""
        )
        warnings.extend(fill_warnings)
    else:
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

        obj_types = ("mcq", "true_false", "fill_in_the_blank")
        obj_planned = {q: plan_items.get(q) or [] for q in obj_types}
        if any(obj_planned[q] for q in obj_types):
            bundle, model_warnings_ = _generate_obj_bundle(
                obj_planned,
                context,
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
            questions[qtype], type_warnings = _generate_type_from_plan(
                qtype,
                planned,
                context,
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
            within_model.extend(questions[qtype])

        # Snapshot accepted output from the original generation calls before
        # count trimming/filling. Shortfall generation must not improve this.
        if eval_stats is not None:
            record_initial_generation(eval_stats, model_number, questions)

        # Minimal-diff repair: missing -> generate only the missing amount,
        # extra -> remove only the extra amount, correct -> untouched.
        shortfall_appended = {qtype: 0 for qtype, _count in tasks}
        repair_warnings = _repair_shortfalls(
            questions,
            tasks,
            plan_items,
            context,
            difficulty,
            model_number,
            seen,
            within_model,
            previous_questions,
            language=language,
            appended_by_type=shortfall_appended,
            eval_stats=eval_stats,
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
        }

    exams = result.get("generated_exams") or []
    warnings = result.get("warnings") or []
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
