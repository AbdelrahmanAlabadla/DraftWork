"""Question-type handlers: DraftWork normalized items → Forms batchUpdate requests.

Registry-based so future types (e.g. match-the-columns) plug in without
redesigning the exporter.
"""
from __future__ import annotations

from typing import Any

from app import config

_POINTS = config.GOOGLE_FORMS_POINTS


class UnsupportedQuestionType(Exception):
    pass


def make_page_break(title: str) -> dict[str, Any]:
    return {
        "createItem": {
            "item": {"title": title, "pageBreakItem": {}},
            "location": {"index": 0},
        }
    }


def _question_item(title: str, question: dict[str, Any],
                   grading: dict[str, Any] | None = None) -> dict[str, Any]:
    item_q: dict[str, Any] = {"required": True, **question}
    if grading:
        item_q["grading"] = {"pointValue": _POINTS, **grading}
    return {
        "createItem": {
            "item": {"title": title, "questionItem": {"question": item_q}},
            "location": {"index": 0},
        }
    }


def export_mcq(item: dict[str, Any]) -> list[dict[str, Any]]:
    options = item.get("options") or {}
    values = [str(v) for v in options.values() if str(v).strip()]
    correct_letter = str(item.get("correct_answer") or "").strip().upper()
    correct_text = ""
    if correct_letter and correct_letter in options:
        correct_text = str(options[correct_letter])
    elif item.get("correct_text"):
        correct_text = str(item["correct_text"])
    grading = (
        {"correctAnswers": {"answers": [{"value": correct_text}]}}
        if correct_text else None
    )
    return [_question_item(item["text"], {
        "choiceQuestion": {"type": "RADIO", "options": [{"value": v} for v in values]}
    }, grading)]


def export_true_false(item: dict[str, Any]) -> list[dict[str, Any]]:
    answer = "True" if str(item.get("answer") or "").lower() == "true" else "False"
    return [_question_item(item["text"], {
        "choiceQuestion": {
            "type": "RADIO",
            "options": [{"value": "True"}, {"value": "False"}],
        }
    }, {"correctAnswers": {"answers": [{"value": answer}]}})]


def _answer_variants(answer: str) -> list[str]:
    """Deterministic case variants of a single accepted answer (no LLM)."""
    variants = [answer]
    for v in (answer.lower(), answer.upper(), answer.title()):
        if v not in variants:
            variants.append(v)
    return variants


def export_fitb(item: dict[str, Any]) -> list[dict[str, Any]]:
    answers = [str(a) for a in (item.get("answers") or []) if str(a).strip()]
    graded: list[dict[str, Any]] = []
    for ans in answers:
        graded.extend({"value": v} for v in _answer_variants(ans))
    grading = {"correctAnswers": {"answers": graded}} if graded else None
    return [_question_item(item["text"], {
        "textQuestion": {"paragraph": False}
    }, grading)]


def export_short_answer(item: dict[str, Any]) -> list[dict[str, Any]]:
    # Manual grading; reference_answer is intentionally NOT exported.
    return [_question_item(item["text"], {"textQuestion": {"paragraph": True}})]


export_essay = export_short_answer


def export_unsupported(item: dict[str, Any]) -> list[dict[str, Any]]:
    # Safe fallback: manual paragraph response, warning raised by caller.
    return [_question_item(item["text"], {"textQuestion": {"paragraph": True}})]


QUESTION_HANDLERS = {
    "mcq": export_mcq,
    "true_false": export_true_false,
    "fill_in_the_blank": export_fitb,
    "short_answer": export_short_answer,
    "essay": export_essay,
}


def build_question_requests(item: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    """Return (batchUpdate requests, warning-or-None) for one normalized item."""
    handler = QUESTION_HANDLERS.get(item.get("qtype"))
    if handler is None:
        reqs = export_unsupported(item)
        qid = item.get("question_id") or "?"
        return reqs, f"unsupported question type '{item.get('qtype')}' for {qid}; exported as manual paragraph"
    try:
        return handler(item), None
    except Exception as exc:
        qid = item.get("question_id") or "?"
        return [], f"failed to convert {qid}: {exc}"
