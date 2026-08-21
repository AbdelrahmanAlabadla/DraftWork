from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from app.llm.client import LMStudioClient
from app.logging_conf import get_logger
from app.online.models import (
    _normalize_essay,
    _normalize_mcq,
    _normalize_short_answer,
    _normalize_true_false,
    normalize_fitb_item,
)

logger = get_logger("VALIDATOR")

# Smallest possible repair actions produced by the validator and consumed by the
# repair stage. ``PASS`` means the question is left byte-identical.
ACTIONS = frozenset(
    {"PASS", "FIX_QUESTION", "FIX_ANSWER", "FIX_OPTIONS", "FIX_QUESTION_AND_ANSWER"}
)

MAX_QUESTION_REPAIR_ATTEMPTS = 2

VALIDATOR_TEMPERATURE = 0.1
# Some served models (e.g. qwen/qwen3-8b) only support reasoning 'off'/'on'.
# 'off' keeps the call fast and non-creative while still producing verdicts.
VALIDATOR_REASONING = "on"
VALIDATOR_MAX_TOKENS = 12000

REPAIR_TEMPERATURE = 0.1
REPAIR_REASONING = "on"
REPAIR_MAX_TOKENS = 12000

# Logical field names (as the validator may report them) -> canonical dictionary
# keys produced by the existing normalizers. Used to translate ``fields_to_fix``
# into the field names the field-diff actually compares. Both the logical names
# and the canonical keys themselves are accepted, because the served model may
# echo either spelling in ``fields_to_fix``.
_LOGICAL_TO_CANONICAL: Dict[str, Dict[str, Set[str]]] = {
    "mcq": {
        "question": {"question"},
        "options": {"options"},
        "answer": {"correct_answer"},
        "correct_answer": {"correct_answer"},
    },
    "true_false": {
        "question": {"statement"},
        "statement": {"statement"},
        "answer": {"answer"},
    },
    "fill_in_the_blank": {
        "question": {"question"},
        "answer": {"answers"},
        "answers": {"answers"},
        "word_bank": {"word_bank"},
    },
    "short_answer": {
        "question": {"question"},
        "answer": {"reference_answer"},
        "reference_answer": {"reference_answer"},
    },
    "essay": {
        "question": {"question"},
        "answer": {"reference_answer"},
        "reference_answer": {"reference_answer"},
        "key_points": {"key_points"},
    },
}

# Default allowed canonical fields per (qtype, action) used whenever the
# validator output either omits ``fields_to_fix`` or only lists unknown names.
_ACTION_DEFAULT_FIELDS: Dict[str, Dict[str, List[str]]] = {
    "mcq": {
        "FIX_ANSWER": ["answer"],
        "FIX_OPTIONS": ["options", "answer"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "options", "answer"],
    },
    "true_false": {
        "FIX_ANSWER": ["answer"],
        "FIX_OPTIONS": ["answer"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "answer"],
    },
    "fill_in_the_blank": {
        "FIX_ANSWER": ["answer"],
        "FIX_OPTIONS": ["word_bank", "answer"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "answer"],
    },
    "short_answer": {
        "FIX_ANSWER": ["answer"],
        "FIX_OPTIONS": ["answer"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "answer"],
    },
    "essay": {
        "FIX_ANSWER": ["answer", "key_points"],
        "FIX_OPTIONS": ["answer", "key_points"],
        "FIX_QUESTION": ["question"],
        "FIX_QUESTION_AND_ANSWER": ["question", "answer", "key_points"],
    },
}

_ALL_CANONICAL: Dict[str, Set[str]] = {
    "mcq": {"question", "options", "correct_answer"},
    "true_false": {"statement", "answer"},
    "fill_in_the_blank": {"question", "answers", "word_bank"},
    "short_answer": {"question", "reference_answer"},
    "essay": {"question", "reference_answer", "key_points"},
}

# Cheap local detectors that never need an LLM call.
_BLANK_MARKERS = ("___", "____", "_____", "________", "…")


def _normalize(qtype: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize ONE repaired question through the existing normalizers."""
    if qtype == "mcq":
        return _normalize_mcq(item)
    if qtype == "true_false":
        return _normalize_true_false(item)
    if qtype == "fill_in_the_blank":
        return normalize_fitb_item(item)
    if qtype == "short_answer":
        return _normalize_short_answer(item)
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
        }
    if qtype == "true_false":
        return {"statement": q.get("statement"), "answer": q.get("answer")}
    if qtype == "fill_in_the_blank":
        return {
            "question": q.get("question"),
            "answers": q.get("answers"),
            "word_bank": q.get("word_bank"),
        }
    if qtype == "short_answer":
        return {
            "question": q.get("question"),
            "reference_answer": q.get("reference_answer"),
        }
    if qtype == "essay":
        return {
            "question": q.get("question"),
            "reference_answer": q.get("reference_answer"),
            "key_points": q.get("key_points"),
        }
    return {}


def _allowed_canonical(
        qtype: str, fields_to_fix: Optional[List[Any]], action: str
) -> Set[str]:
    """Translate ``fields_to_fix`` (plus action fallback) to canonical keys.

    Translation is strict: any unknown field name is ignored. If nothing
    translates, the per-action defaults apply; as a last resort the whole
    canonical field set is allowed so a repair never accidentally succeeds while
    wrongly rejecting a valid fix.
    """
    mapping = _LOGICAL_TO_CANONICAL.get(qtype, {})
    canon: Set[str] = set()
    for name in fields_to_fix or []:
        canon |= mapping.get(str(name).strip().lower(), set())
    if not canon:
        defaults = _ACTION_DEFAULT_FIELDS.get(qtype, {}).get(action, [])
        for name in defaults:
            canon |= mapping.get(name, set())
    if not canon:
        canon = set(_ALL_CANONICAL.get(qtype, set()))
    # Essay answers and key points are coupled: rewriting the reference must be
    # allowed to adjust key_points (and vice versa) so they stay consistent.
    if qtype == "essay" and ("reference_answer" in canon or "key_points" in canon):
        canon |= {"reference_answer", "key_points"}
    # MCQ options and the stored answer are coupled too: enforcing "exactly one
    # correct option" may require changing the answer when options change, or
    # fixing the option set when the answer is wrong. Never allow stem edits.
    if qtype == "mcq" and ("options" in canon or "correct_answer" in canon):
        canon |= {"options", "correct_answer"}
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


# --------------------------------------------------------------------------
# Cheap deterministic checks (no LLM call needed)
# --------------------------------------------------------------------------
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
        dupes = {v for v in values if values.count(v) > 1}
        if dupes:
            return _verdict(True, False, "FIX_OPTIONS", ["options"],
                            f"MCQ has duplicate options: {sorted(dupes)}.",
                            "Make all four options clearly distinct.")
        if str(item.get("correct_answer") or "").strip().upper() not in options:
            return _verdict(True, False, "FIX_OPTIONS", ["options", "answer"],
                            "The stored answer points to a nonexistent option.",
                            "Update options so the real correct answer exists and set the answer to it.")
    elif qtype == "true_false":
        statement = str(item.get("statement") or "").strip()
        if not statement:
            return _verdict(False, False, "FIX_QUESTION", ["question"],
                            "True/False statement is empty.",
                            "Write a complete, single, unambiguous statement.")
        if str(item.get("answer") or "").strip().lower() not in {"true", "false"}:
            return _verdict(True, False, "FIX_ANSWER", ["answer"],
                            "True/False answer is not 'True' or 'False'.",
                            "Set the answer to exactly 'True' or 'False'.")
    elif qtype == "fill_in_the_blank":
        question = str(item.get("question") or "").strip()
        if not any(m in question for m in _BLANK_MARKERS):
            return _verdict(False, False, "FIX_QUESTION", ["question"],
                            "Fill-in-the-Blank sentence has no blank marker.",
                            "Add a single run of underscores (________) marking the blank.")
        answers = item.get("answers") or []
        if not answers or not all(str(a).strip() for a in answers):
            return _verdict(True, False, "FIX_ANSWER", ["answer"],
                            "Fill-in-the-Blank item is missing its answer.",
                            "Provide the blank answer(s) as Word Bank terms.")
        if isinstance(section, dict):
            bank = [str(w).strip() for w in (section.get("word_bank") or [])]
            missing = [str(a).strip() for a in answers if str(a).strip() not in bank]
            if missing:
                return _verdict(True, False, "FIX_OPTIONS", ["word_bank", "answer"],
                                f"Answer(s) not present in the Word Bank: {missing}.",
                                "Add the correct term(s) to the Word Bank (or fix the answer).")
    elif qtype in ("short_answer", "essay"):
        if not str(item.get("question") or "").strip():
            return _verdict(False, False, "FIX_QUESTION", ["question"],
                            "Question text is empty.",
                            "Write a complete, answerable question.")
        if not str(item.get("reference_answer") or "").strip():
            return _verdict(True, False, "FIX_ANSWER", ["answer"],
                            "Reference answer is empty.",
                            "Write a complete reference answer that directly answers the question.")
    return None


def _verdict(
        qv: bool, av: bool, action: str, fields: List[str], reason: str, expected_fix: str
) -> Dict[str, Any]:
    return {
        "question_id": "",
        "question_valid": qv,
        "answer_valid": av,
        "action": action,
        "fields_to_fix": fields,
        "reason": reason,
        "expected_fix": expected_fix,
    }


# --------------------------------------------------------------------------
# Validator LLM prompts
# --------------------------------------------------------------------------
VALIDATOR_SYSTEM_PROMPT = (
    "You are a strict, non-creative exam reviewer. Your only job is to review each "
    "exam question independently and report problems with a structured verdict.\n\n"
    "Rules:\n"
    "- Review EVERY question you are given; return exactly one verdict per question_id.\n"
    "- Judge only what is inside the provided question JSON. Never bring in outside "
    "knowledge that is not expressed by the question/info given.\n"
    "- For Multiple Choice: first solve the question independently, then check the "
    "option set and the stored answer. Use FIX_OPTIONS (fields_to_fix "
    '["options","answer"]) when two or more options are correct, when the real '
    "correct answer is absent from the options, or when options are duplicated or "
    "near-duplicates. Use FIX_ANSWER (fields_to_fix [\"answer\"]) only when the "
    "question and option set are sound but the stored answer points to the wrong "
    "option. Flag an unclear stem with FIX_QUESTION.\n"
    "- For True/False: independently judge whether the statement is True or False; "
    "flag a malformed/contradictory statement.\n"
    "- For Fill-in-the-Blank: check the blank has one clear intended answer, the stored "
    "answer fits, other Word Bank entries do not fit equally, and the sentence is "
    "grammatically valid after filling.\n"
    "- For Short Answer / Essay / writing questions: do NOT compare wording. Judge "
    "whether the reference answer actually answers the question, is relevant, logically "
    "correct, complete enough, and free of contradictions or unrelated material.\n"
    "- Choose the SMALLEST repair:\n"
    "    PASS                        -> fully valid\n"
    "    FIX_ANSWER                  -> question fine, answer/reference wrong\n"
    "    FIX_OPTIONS                 -> options / word bank wrong (question fine)\n"
    "    FIX_QUESTION                -> question text broken but answer usable\n"
    "    FIX_QUESTION_AND_ANSWER     -> both broken\n"
    "- A question is NOT invalid just because it could be worded better. If it is "
    "understandable, answerable as intended, and structurally correct, mark PASS.\n"
    "- Return ONLY valid JSON: an array of verdict objects. No fences, no extra text.\n"
)

VALIDATOR_OUTPUT_SCHEMA = (
    '[\n'
    '  {\n'
    '    "question_id": "model1_mcq_2",\n'
    '    "question_valid": true,\n'
    '    "answer_valid": false,\n'
    '    "action": "FIX_ANSWER",\n'
    '    "fields_to_fix": ["answer"],\n'
    '    "reason": "The correct option is C, but the stored answer is B.",\n'
    '    "expected_fix": "Change the answer to C. Keep the question and options unchanged."\n'
    '  }\n'
    ']'
)

VALIDATOR_RETRY_SYSTEM_PROMPT = (
    "You are a strict, non-creative exam reviewer. A previous review failed to "
    "return a verdict for the questions below. Review each one and return exactly "
    "one structured verdict per question_id using the same verdict schema as before "
    "(question_id, question_valid, answer_valid, action, fields_to_fix, reason, "
    "expected_fix).\n"
    "- Review EVERY question; never skip a question_id.\n"
    "- Return ONLY valid JSON: an array of verdict objects. No fences, no extra text."
)


def build_validator_prompt(entries: List[Dict[str, Any]]) -> str:
    payload = json.dumps(entries, ensure_ascii=False, indent=None)
    return (
        "Review every exam question below and return one structured verdict per "
        "question (same number of results as entries).\n"
        "Use the exact question_ids you were given and never invent new ones.\n\n"
        "## Questions to review\n"
        f"{payload}\n\n"
        "## Verdict object format (use exactly these field names)\n"
        f"{VALIDATOR_OUTPUT_SCHEMA}\n\n"
        "Return ONLY the JSON array of verdicts. No text before or after, no fences."
    )


REPAIR_SYSTEM_PROMPT = (
    "You are a precise exam-question repairer. For each item you are given, fix ONLY "
    "the broken field(s) listed in fields_to_fix. Rules:\n"
    "- Return the EXACT same question_id for each repaired question; never change it.\n"
    "- Do NOT change fields that are not listed in fields_to_fix.\n"
    "- Do NOT rewrite a question that is fine; preserve every valid field verbatim.\n"
    "- Only repair the specific problem described in reason/expected_fix.\n"
    "- Multiple Choice (when options are in fields_to_fix): first solve the stem "
    "independently, then evaluate the option set. There must be EXACTLY one correct "
    "option. If two or more options are correct, keep one and rewrite only the "
    "conflicting distractor(s) so exactly one correct answer remains. If the real "
    "correct answer is missing from the options, replace one incorrect option with "
    "the real correct answer. After changing the options, update the stored answer "
    "to point to the one correct option. Preserve every already-valid option; change "
    "the minimum number of options needed; never regenerate the whole MCQ randomly; "
    "keep the question text unchanged unless the validator explicitly marked the "
    "question itself as broken.\n"
    "- Return an array of repair objects, one per input item, in any order, each:\n"
    '  {"question_id": "...", "repaired_fields": ["..."], '
    '"question": {the full repaired question object with the same fields}]}\n'
    "- Return ONLY valid JSON. No fences, no extra text."
)

REPAIR_OUTPUT_SCHEMA = (
    '[\n'
    '  {\n'
    '    "question_id": "example_mcq_1",\n'
    '    "repaired_fields": ["answer"],\n'
    '    "question": {"question": "...", "options": {"A": "...", "B": "...", '
    '"C": "...", "D": "..."}, "correct_answer": "C"}\n'
    '  }\n'
    ']'
)


def build_repair_prompt(payload: List[Dict[str, Any]]) -> str:
    block = json.dumps(payload, ensure_ascii=False, indent=None)
    return (
        "Repair the following questions. Each item tells you exactly what is broken "
        "and what to fix. Change ONLY the broken fields; keep everything else "
        "byte-identical; keep question_id unchanged.\n\n"
        "## Items to repair\n"
        f"{block}\n\n"
        "## Repair object format (use exactly these field names)\n"
        f"{REPAIR_OUTPUT_SCHEMA}\n\n"
        "Return ONLY the JSON array of repaired items. No text before or after, no fences."
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
    qv = bool(raw.get("question_valid", True))
    av = bool(raw.get("answer_valid", True))
    action = str(raw.get("action") or "").upper()
    if action not in ACTIONS:
        action = "PASS"
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
        "reason": str(raw.get("reason") or "").strip(),
        "expected_fix": str(raw.get("expected_fix") or "").strip(),
        "solved_answer": raw.get("solved_answer"),
    }


def validate_exam(
        exam: Dict[str, Any], client: Optional[Any] = None
) -> Dict[str, Any]:
    """Validate every question of one exam model with ONE LLM call.

    Returns ``{"model_number", "all_pass", "verdicts", "warnings"}`` where each
    verdict is keyed by the question's stable ``question_id``. Verdicts for
    questions without a recorded ID are treated as PASS (with a warning).
    """
    model_number = exam.get("model_number") or 1
    questions = exam.get("questions") or {}
    warnings: List[str] = []

    # Deterministic structural problems never consume an LLM call.
    local_verdicts: Dict[str, Dict[str, Any]] = {}
    entries: List[Dict[str, Any]] = []
    for qtype, container, item in _iter_questions(questions):
        qid = item.get("question_id")
        if not qid:
            warnings.append(f"validate: question missing question_id; skipped")
            continue
        verdict = _local_verdict(qtype, item, container)
        if verdict is not None:
            verdict["question_id"] = str(qid)
            local_verdicts[str(qid)] = verdict
            continue
        payload = dict(item)
        if qtype == "fill_in_the_blank" and isinstance(container, dict):
            payload["word_bank"] = container.get("word_bank") or []
        entries.append({"question_id": str(qid), "question_type": qtype, "question": payload})

    verdicts: Dict[str, Dict[str, Any]] = dict(local_verdicts)
    known_ids = {str(item.get("question_id")) for _, _, item in _iter_questions(questions)}
    known_ids.discard("")
    if entries:
        llm = client or LMStudioClient(reasoning=VALIDATOR_REASONING)
        user_prompt = build_validator_prompt(entries)
        try:
            raw = llm.chat_json(
                user_prompt,
                system_prompt=VALIDATOR_SYSTEM_PROMPT,
                temperature=VALIDATOR_TEMPERATURE,
                max_tokens=VALIDATOR_MAX_TOKENS,
            )
        except Exception as exc:
            warnings.append(f"validate: validator LLM call failed ({exc}); no verdicts received")
            raw = []
        results = raw if isinstance(raw, list) else (raw.get("verdicts") if isinstance(raw, dict) else None)
        if not isinstance(results, list):
            warnings.append("validate: validator returned no usable verdict array; questions will be retried/failed")
            results = []
        for r in results:
            v = _sanitize_verdict(r, known_ids)
            if v is None:
                warnings.append(f"validate: verdict with unknown/missing question_id rejected")
                continue
            verdicts[v["question_id"]] = v

        # The validator must not silently clear a question. Give the model one
        # focused retry for the questions it omitted; if they are still missing
        # they fail (non-PASS) into the repair flow instead of being accepted.
        missing = [e for e in entries if e["question_id"] not in verdicts]
        if missing:
            warnings.append(f"validate: {len(missing)} question(s) had no verdict; retrying once")
            try:
                raw2 = llm.chat_json(
                    build_validator_prompt(missing),
                    system_prompt=VALIDATOR_RETRY_SYSTEM_PROMPT,
                    temperature=VALIDATOR_TEMPERATURE,
                    max_tokens=VALIDATOR_MAX_TOKENS,
                )
            except Exception as exc:
                warnings.append(f"validate: validator retry failed ({exc})")
                raw2 = []
            results2 = raw2 if isinstance(raw2, list) else (raw2.get("verdicts") if isinstance(raw2, dict) else None)
            if not isinstance(results2, list):
                warnings.append("validate: retry returned no usable verdict array")
                results2 = []
            for r in results2:
                v = _sanitize_verdict(r, known_ids)
                if v is None:
                    warnings.append(f"validate: retry verdict with unknown/missing question_id rejected")
                    continue
                verdicts[v["question_id"]] = v

    all_pass = True
    ordered: List[Dict[str, Any]] = []
    for _, _, item in _iter_questions(questions):
        qid = str(item.get("question_id") or "")
        if not qid:
            continue
        v = verdicts.get(qid)
        if v is None:
            warnings.append(f"validate: no verdict returned for {qid} after retry; marked failed")
            v = {
                "question_id": qid,
                "question_valid": False,
                "answer_valid": False,
                "action": "FIX_QUESTION_AND_ANSWER",
                "fields_to_fix": ["question", "answer"],
                "reason": "No validator verdict was returned after retry.",
                "expected_fix": "Review the question fully and fix the question and/or answer.",
            }
        if v["action"] != "PASS":
            all_pass = False
        ordered.append(v)

    logger.info(
        "Exam validated | model=%d | questions=%d | invalid=%d",
        model_number,
        len(ordered),
        sum(1 for v in ordered if v["action"] != "PASS"),
    )
    return {"model_number": model_number, "all_pass": all_pass, "verdicts": ordered, "warnings": warnings}


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
        exam: Dict[str, Any], verdicts: List[Dict[str, Any]], client: Optional[Any] = None
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
    invalid = [v for v in verdicts if v.get("action") != "PASS"]
    if not invalid:
        return {"warnings": warnings, "repaired_ids": [], "failed_ids": []}

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
        payload.append(
            {
                "question_id": qid,
                "action": v.get("action", "FIX_ANSWER"),
                "fields_to_fix": v.get("fields_to_fix") or [],
                "reason": v.get("reason") or "",
                "expected_fix": v.get("expected_fix") or "",
                "question": p,
            }
        )

    if not payload:
        return {
            "warnings": warnings,
            "repaired_ids": [],
            "failed_ids": [str(v.get("question_id")) for v in invalid],
        }

    llm = client or LMStudioClient(reasoning=REPAIR_REASONING)
    user_prompt = build_repair_prompt(payload)
    try:
        raw = llm.chat_json(
            user_prompt,
            system_prompt=REPAIR_SYSTEM_PROMPT,
            temperature=REPAIR_TEMPERATURE,
            max_tokens=REPAIR_MAX_TOKENS,
        )
    except Exception as exc:
        warnings.append(f"repair: batched repair LLM call failed ({exc})")
        return {
            "warnings": warnings,
            "repaired_ids": [],
            "failed_ids": [str(v.get("question_id")) for v in invalid],
        }

    repairs = raw if isinstance(raw, list) else (raw.get("repairs") if isinstance(raw, dict) else None)
    if not isinstance(repairs, list):
        warnings.append("repair: LLM returned no usable repair array")
        return {
            "warnings": warnings,
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

        stashed_id = item.get("question_id")
        normalized = _normalize(qtype, new_item)
        if normalized is None:
            warnings.append(f"repair: {qid} output failed normalization; rejected")
            continue
        normalized["question_id"] = stashed_id

        # FITB: word bank is section-level. Only a word_bank-enabled repair may
        # rewrite it; a single item repair never touches other items' content.
        bank_before = None
        if qtype == "fill_in_the_blank" and isinstance(container, dict):
            bank_before = list(container.get("word_bank") or [])
            verdict = verdict_by_id.get(qid) or {}
            allowed = _allowed_canonical(
                qtype, verdict.get("fields_to_fix") or [], verdict.get("action") or "FIX_ANSWER"
            )
            if "word_bank" in allowed:
                bank = new_item.get("word_bank")
                if isinstance(bank, list) and bank:
                    container["word_bank"] = [str(w) for w in bank if str(w).strip()]

        verdict = verdict_by_id.get(qid) or {}
        allowed = _allowed_canonical(
            qtype, verdict.get("fields_to_fix") or [], verdict.get("action") or "FIX_ANSWER"
        )
        violations = _fields_changed_disallowed(qtype, item, normalized, allowed)
        if violations:
            warnings.append(
                f"repair: {qid} modified disallowed field(s) {violations}; rejected"
            )
            continue

        # A repair that leaves every canonical field identical does not fix
        # anything; do not count it as repaired so the loop keeps trying. A FITB
        # repair that only rewrote the section word bank still counts as a fix.
        bank_changed = bank_before is not None and bank_before != list(container.get("word_bank") or [])
        if not bank_changed and _canonical_values(qtype, item) == _canonical_values(qtype, normalized):
            warnings.append(f"repair: {qid} returned no change; rejected")
            continue

        # Replace the question in place. Keeping the same dict object preserves
        # the stable question_id and any external references.
        item.clear()
        item.update(normalized)
        accepted.add(qid)
        logger.info("Question repaired | id=%s | action=%s", qid, verdict.get("action"))

    failed = [str(v.get("question_id")) for v in invalid if str(v.get("question_id")) not in accepted]
    if failed:
        warnings.append(f"repair: {len(failed)} question(s) not repaired: {sorted(failed)}")
    return {
        "warnings": warnings,
        "repaired_ids": sorted(accepted),
        "failed_ids": failed,
    }


# --------------------------------------------------------------------------
# LangGraph nodes
# --------------------------------------------------------------------------
def validate_generated_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: validate every not-yet-passed exam model (one LLM call per model)."""
    generated = state.get("generated_exams") or []
    validated = list(state.get("validated_models") or [])
    attempts = state.get("question_repair_attempts") or {}
    warnings = list(state.get("warnings") or [])
    reports: List[Dict[str, Any]] = []

    for exam in generated:
        mn = exam.get("model_number")
        if mn in validated:
            continue
        report = validate_exam(exam)
        reports.append(report)
        if report["all_pass"]:
            validated.append(mn)
        elif (attempts.get(mn) or 0) >= MAX_QUESTION_REPAIR_ATTEMPTS:
            warnings.append(
                f"Model {mn}: max question repair attempts reached; "
                "keeping best-effort questions."
            )
            validated.append(mn)

    return {
        "validation_reports": reports,
        "validated_models": validated,
        "warnings": warnings,
    }


def repair_invalid_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: batched repair per model, one call per model, ID-keyed write-back."""
    reports = state.get("validation_reports") or []
    generated = state.get("generated_exams") or []
    attempts = dict(state.get("question_repair_attempts") or {})
    validated = list(state.get("validated_models") or [])
    warnings = list(state.get("warnings") or [])
    exam_by_model = {e.get("model_number"): e for e in generated}

    for r in reports:
        mn = r.get("model_number")
        if r.get("all_pass"):
            continue
        if (attempts.get(mn) or 0) >= MAX_QUESTION_REPAIR_ATTEMPTS:
            continue
        exam = exam_by_model.get(mn)
        if exam is None:
            continue
        attempt = (attempts.get(mn) or 0) + 1
        logger.info("Question repair pass | model=%d | attempt=%d/%d", mn, attempt, MAX_QUESTION_REPAIR_ATTEMPTS)
        outcome = repair_exam(exam, r.get("verdicts") or [])
        attempts[mn] = attempt
        warnings.extend(outcome["warnings"])
        if mn in validated:
            validated.remove(mn)

    return {
        "question_repair_attempts": attempts,
        "validated_models": validated,
        "warnings": warnings,
    }
