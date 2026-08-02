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
    "short_answer": "Short Answer",
}

TYPE_ORDER = ["mcq", "true_false", "short_answer"]

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


_NORMALIZERS = {
    "mcq": _normalize_mcq,
    "true_false": _normalize_true_false,
    "short_answer": _normalize_short_answer,
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


def render_markdown(
    question_type: str, questions: list[dict[str, Any]], start_index: int = 1
) -> str:
    label = TYPE_LABELS.get(question_type, question_type)
    lines: list[str] = [f"## {label}", ""]

    for offset, q in enumerate(questions, start=start_index):
        if question_type == "mcq":
            lines.append(f"{offset}. {q['question']}")
            for letter in sorted(q["options"], key=lambda k: ("ABCDEF".index(k) if k in "ABCDEF" else 99, k)):
                lines.append(f"   {letter}. {q['options'][letter]}")
            lines.append(f"   **Answer: {q['correct_answer']}**")
        elif question_type == "true_false":
            lines.append(f"{offset}. {q['statement']}")
            lines.append(f"   **Answer: {q['answer']}**")
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
