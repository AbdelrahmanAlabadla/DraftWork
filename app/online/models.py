from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.llm.json_utils import JSONExtractionError, extract_json
from app.logging_conf import get_logger

logger = get_logger("GENERATOR")

TYPE_LABELS = {
    "mcq": "Multiple Choice",
    "true_false": "True / False",
    "fill_in_the_blank": "Fill in the Blank",
    "short_answer": "Short Answer",
    "essay": "Essay",
}

TYPE_ORDER = ["mcq", "true_false", "fill_in_the_blank", "short_answer", "essay"]

# Phrases that reference the source of information; questions containing any of
# these are treated as invalid and re-generated.
FORBIDDEN_PHRASES = (
    "according to the",
    "as shown in",
    "as discussed",
    "the following example",
    "in the passage",
    "in the context",
    "in the document",
    "based on the provided",
    "based on the text",
    "from the given",
    "from the context",
    "according to the analysis",
    "according to the document",
    "according to the context",
)

_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase, strip diacritics/punctuation, collapse whitespace — for dedup."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _PUNCT_RE.sub(" ", text).lower().split()


def contains_forbidden_phrase(text: str) -> bool:
    lowered = str(text).lower()
    return any(phrase in lowered for phrase in FORBIDDEN_PHRASES)


def _questions_from(raw: object) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [q for q in raw if isinstance(q, dict)]
    if isinstance(raw, dict):
        return [q for q in raw.get("questions", []) if isinstance(q, dict)]
    return []


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_mcq(item: dict[str, Any]) -> dict[str, Any] | None:
    question = _clean_str(item.get("question") or item.get("question_text"))
    if not question:
        return None
    options_raw = item.get("options")
    options: dict[str, str] = {}
    if isinstance(options_raw, dict):
        options = {k.strip().upper(): _clean_str(v) for k, v in options_raw.items() if _clean_str(v)}
    elif isinstance(options_raw, list):
        for idx, opt in enumerate(options_raw):
            if _clean_str(opt):
                options[chr(ord("A") + idx)] = _clean_str(opt)

    if len(options) < 4:
        return None

    correct = _clean_str(item.get("correct_answer")).upper()
    if correct not in options:
        # Correct answer letter must actually exist among the options.
        return None

    ordered = {key: options[key] for key in sorted(options, key=lambda k: ("ABCDEF".index(k) if k in "ABCDEF" else 99, k))}
    return {"question": question, "options": ordered, "correct_answer": correct}


def _normalize_true_false(item: dict[str, Any]) -> dict[str, Any] | None:
    statement = _clean_str(item.get("statement") or item.get("question_text") or item.get("question"))
    if not statement:
        return None
    answer = item.get("answer")
    if isinstance(answer, bool):
        answer = "True" if answer else "False"
    answer = _clean_str(answer)
    if answer.lower() not in {"true", "false"}:
        return None
    return {"statement": statement, "answer": "True" if answer.lower() == "true" else "False"}


def _normalize_short_answer(item: dict[str, Any]) -> dict[str, Any] | None:
    question = _clean_str(item.get("question") or item.get("question_text"))
    reference = _clean_str(
        item.get("reference_answer") or item.get("model_answer") or item.get("answer")
    )
    if not question or not reference:
        return None
    return {"question": question, "reference_answer": reference}


def _normalize_essay(item: dict[str, Any]) -> dict[str, Any] | None:
    question = _clean_str(item.get("question") or item.get("question_text"))
    reference = _clean_str(
        item.get("reference_answer") or item.get("model_answer") or item.get("answer")
    )
    if not question or not reference:
        return None
    key_points_raw = item.get("key_points")
    if key_points_raw is not None and not isinstance(key_points_raw, list):
        # Schema violation: key_points must be an array. Reject so the caller can
        # send this item back for targeted structural repair instead of silently
        # dropping the key_points.
        return None
    key_points: list[str] = []
    if isinstance(key_points_raw, list):
        key_points = [_clean_str(k) for k in key_points_raw if _clean_str(k)]
    return {"question": question, "reference_answer": reference, "key_points": key_points}


def normalize_fitb_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a single fill-in-the-blank item (1 or 2 ordered blanks)."""
    question = _clean_str(item.get("question") or item.get("question_text"))
    if not question:
        return None
    raw_answers = item.get("answers")
    if raw_answers is None:
        raw_answers = item.get("answer")
    if isinstance(raw_answers, str):
        raw_answers = [raw_answers]
    answers: list[str] = []
    if isinstance(raw_answers, list):
        answers = [_clean_str(a) for a in raw_answers if _clean_str(a)]
    if not (1 <= len(answers) <= 2):
        return None
    return {"question": question, "answers": answers}


_NORMALIZERS = {
    "mcq": _normalize_mcq,
    "true_false": _normalize_true_false,
    "short_answer": _normalize_short_answer,
    "essay": _normalize_essay,
}


def _parse_obj(question_type: str, raw: object) -> list[dict[str, Any]]:
    """Normalize already-parsed model output into canonical question dicts."""
    normalizer = _NORMALIZERS[question_type]
    questions: list[dict[str, Any]] = []
    for item in _questions_from(raw):
        normalized = normalizer(item)
        if normalized is not None:
            questions.append(normalized)
    return questions


def parse_questions(question_type: str, raw_text: str) -> list[dict[str, Any]]:
    """Parse model JSON text and normalize into canonical question dicts."""
    return _parse_obj(question_type, extract_json(raw_text))


def parse_questions_obj(question_type: str, parsed: object) -> list[dict[str, Any]]:
    """Normalize an already-parsed JSON object into canonical question dicts."""
    return _parse_obj(question_type, parsed)


def split_valid_invalid(
    question_type: str, raw: object
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split raw items into (valid normalized, raw items that failed schema validation).

    The invalid raw items are returned unchanged so a caller can hand them back
    to the model for targeted structural repair (preserving valid content) rather
    than discarding them and regenerating from scratch.
    """
    normalizer = _NORMALIZERS[question_type]
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for item in _questions_from(raw):
        normalized = normalizer(item)
        if normalized is not None:
            valid.append(normalized)
        else:
            invalid.append(item)
    return valid, invalid


def _render_fitb(section: dict[str, Any], start_index: int = 1) -> str:
    """Render the FITB section: boxed Word Bank followed by numbered items."""
    lines: list[str] = [f"## {TYPE_LABELS['fill_in_the_blank']}", ""]
    word_bank = [_clean_str(w) for w in (section.get("word_bank") or []) if _clean_str(w)]
    if word_bank:
        box = (
            '<div class="word-bank"><strong>Word Bank</strong><br>'
            + " · ".join(word_bank)
            + "</div>"
        )
        lines.append(box)
        lines.append("")
    for offset, item in enumerate(section.get("items") or [], start=start_index):
        answers = [_clean_str(a) for a in (item.get("answers") or []) if _clean_str(a)]
        lines.append(f"{offset}. {_clean_str(item.get('question'))}")
        lines.append(f"   **Answer:** {', '.join(answers)}")
        lines.append("")
    return "\n".join(lines).strip()


def render_markdown(
    question_type: str, questions: list[dict[str, Any]] | dict[str, Any], start_index: int = 1
) -> str:
    label = TYPE_LABELS.get(question_type, question_type)
    lines: list[str] = [f"## {label}", ""]

    if question_type == "fill_in_the_blank":
        return _render_fitb(questions, start_index=start_index)

    if not isinstance(questions, list):
        return "\n".join(lines).strip()

    for offset, q in enumerate(questions, start=start_index):
        if question_type == "mcq":
            lines.append(f"{offset}. {q['question']}")
            for letter in sorted(q["options"], key=lambda k: ("ABCDEF".index(k) if k in "ABCDEF" else 99, k)):
                lines.append(f"   {letter}. {q['options'][letter]}")
            lines.append(f"   **Answer: {q['correct_answer']}**")
        elif question_type == "true_false":
            lines.append(f"{offset}. {q['statement']}")
            lines.append(f"   **Answer: {q['answer']}**")
        elif question_type == "essay":
            lines.append(f"{offset}. {q['question']}")
            lines.append(f"   **Reference answer:** {q['reference_answer']}")
            key_points = q.get("key_points") or []
            for kp in key_points:
                lines.append(f"   - {kp}")
        else:
            lines.append(f"{offset}. {q['question']}")
            lines.append(f"   **Answer:** {q['reference_answer']}")
        lines.append("")

    return "\n".join(lines).strip()


def question_text(qtype: str, question: dict[str, Any]) -> str:
    """Return the text used for phrase-filtering/dedup of a question."""
    if qtype == "true_false":
        return str(question.get("statement") or "")
    return str(question.get("question") or "")
