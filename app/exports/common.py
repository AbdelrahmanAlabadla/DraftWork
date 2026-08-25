"""Shared normalized export representation.

Every exporter (PDF / DOCX / Google Forms) consumes the flat item list
produced by :func:`flatten_exam_items` so the structured ``questions`` data —
not the markdown rendering — is the single source of truth for exports.
"""
from __future__ import annotations

from typing import Any

from app.online.models import TYPE_LABELS, TYPE_ORDER

# Printed order follows the supplied exam model while staying limited to the
# question types the application currently supports.  ``TYPE_ORDER`` remains
# unchanged because Google Forms and the generation pipeline rely on it.
EXPORT_TYPE_ORDER = [
    "mcq",
    "fill_in_the_blank",
    "true_false",
    "short_answer",
    "essay",
]

EXPORT_TYPE_LABELS = {
    "mcq": "Multiple Choice Questions",
    "fill_in_the_blank": "Fill in the Blank",
    "true_false": "True / False",
    "short_answer": "Short Answer",
    "essay": "Essay",
}

TEACHER_ONLY_FIELDS = ("reference_answer", "key_points", "correct_answer", "answers")

# One response-space contract shared by PDF, DOCX, and the browser preview.
# Keeping these values here prevents the three representations from drifting.
SHORT_ANSWER_RESPONSE_LINES = 3
ESSAY_RESPONSE_LINES = 22
RESPONSE_LINE_TEXT = "_" * 92


def response_line_count(qtype: str) -> int:
    """Return the fixed number of student writing lines for an open question."""
    return SHORT_ANSWER_RESPONSE_LINES if qtype == "short_answer" else ESSAY_RESPONSE_LINES


def flatten_exam_items(questions: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a model's typed sections into ordered export items.

    Item shape (qtype-specific fields included; teacher-only fields flagged):
    {
      qtype, question_id, label, number, text,
      options / correct_answer        (mcq)
      answer                          (true_false)
      answers / word_bank             (fill_in_the_blank)
      reference_answer / key_points   (short_answer, essay — teacher only)
    }
    """
    items: list[dict[str, Any]] = []
    number = 1
    for qtype in TYPE_ORDER:
        section = questions.get(qtype)
        if not section:
            continue
        entries: list[tuple[str | None, dict[str, Any]]] = []
        word_bank = None
        if qtype == "fill_in_the_blank" and isinstance(section, dict):
            word_bank = [str(w) for w in (section.get("word_bank") or []) if str(w).strip()]
            entries = [(it.get("question_id"), it) for it in (section.get("items") or [])]
        elif isinstance(section, list):
            entries = [(q.get("question_id"), q) for q in section]

        for qid, q in entries:
            if not isinstance(q, dict):
                continue
            item: dict[str, Any] = {
                "qtype": qtype,
                "label": TYPE_LABELS.get(qtype, qtype),
                "question_id": qid,
                "number": number,
            }
            if word_bank is not None:
                item["word_bank"] = word_bank
            if qtype == "mcq":
                item["text"] = str(q.get("question") or "")
                item["options"] = q.get("options") or {}
                item["correct_answer"] = q.get("correct_answer")
            elif qtype == "true_false":
                item["text"] = str(q.get("statement") or "")
                item["answer"] = q.get("answer")
            elif qtype == "fill_in_the_blank":
                item["text"] = str(q.get("question") or "")
                item["answers"] = [str(a) for a in (q.get("answers") or []) if str(a).strip()]
            else:  # short_answer / essay
                item["text"] = str(q.get("question") or "")
                item["reference_answer"] = str(q.get("reference_answer") or "")
                if qtype == "essay":
                    item["key_points"] = [str(kp) for kp in (q.get("key_points") or [])]
            items.append(item)
            number += 1
    return items


def group_exam_sections(questions: dict[str, Any]) -> list[dict[str, Any]]:
    """Return supported questions grouped and locally numbered for printing.

    PDF and DOCX exams restart numbering inside each question-type section,
    matching conventional paper exams.  ``flatten_exam_items`` intentionally
    keeps its existing global numbering contract for other integrations.
    """
    flattened = flatten_exam_items(questions)
    sections: list[dict[str, Any]] = []
    for qtype in EXPORT_TYPE_ORDER:
        typed_items = [item for item in flattened if item["qtype"] == qtype]
        if not typed_items:
            continue
        items: list[dict[str, Any]] = []
        for local_number, item in enumerate(typed_items, start=1):
            local_item = dict(item)
            local_item["number"] = local_number
            items.append(local_item)
        section: dict[str, Any] = {
            "qtype": qtype,
            "label": EXPORT_TYPE_LABELS[qtype],
            "items": items,
        }
        if qtype == "fill_in_the_blank":
            section["word_bank"] = list(dict.fromkeys(items[0].get("word_bank") or []))
        sections.append(section)
    return sections
