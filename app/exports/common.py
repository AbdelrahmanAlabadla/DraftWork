"""Shared normalized export representation.

Every exporter (PDF / DOCX / Google Forms) consumes the flat item list
produced by :func:`flatten_exam_items` so the structured ``questions`` data —
not the markdown rendering — is the single source of truth for exports.
"""
from __future__ import annotations

import re
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

EXPORT_TYPE_LABELS_BY_LANG = {
    "en": EXPORT_TYPE_LABELS,
    "ar": {
        "mcq": "أسئلة الاختيار من متعدد",
        "fill_in_the_blank": "أكمل الفراغ",
        "true_false": "صح / خطأ",
        "short_answer": "سؤال قصير",
        "essay": "مقالي",
    },
}

TRUE_FALSE_CHOICES_BY_LANG = {
    "en": "(   ) True     (   ) False",
    "ar": "(   ) صح     (   ) خطأ",
}

HEADER_LABELS_BY_LANG = {
    "en": {
        "model": "Model {n}",
        "class": "Class",
        "duration": "Duration",
        "date": "Date",
        "teacher": "Teacher",
        "student_name": "Student Name: ",
        "class_suffix": "Class: ",
        "answer_key": "Answer Key - Model {n}",
    },
    "ar": {
        "model": "نسخة {n}",
        "class": "الصف",
        "duration": "المدة",
        "date": "التاريخ",
        "teacher": "المعلم",
        "student_name": "اسم الطالب: ",
        "class_suffix": "الصف: ",
        "answer_key": "مفتاح الإجابة - نسخة {n}",
    },
}


def export_type_label(qtype: str, language: str = "en") -> str:
    return EXPORT_TYPE_LABELS_BY_LANG.get(language, EXPORT_TYPE_LABELS).get(qtype, qtype)

TEACHER_ONLY_FIELDS = ("reference_answer", "key_points", "correct_answer", "answers")

# One response-space contract shared by PDF, DOCX, and the browser preview.
# Keeping these values here prevents the three representations from drifting.
SHORT_ANSWER_RESPONSE_LINES = 3
ESSAY_RESPONSE_LINES = 22
RESPONSE_LINE_TEXT = "_" * 92

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_FILENAME_WHITESPACE = re.compile(r"\s+")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def response_line_count(qtype: str) -> int:
    """Return the fixed number of student writing lines for an open question."""
    return SHORT_ANSWER_RESPONSE_LINES if qtype == "short_answer" else ESSAY_RESPONSE_LINES


def safe_filename_part(value: object, *, max_length: int = 80) -> str:
    """Return a readable filename component safe on Windows, macOS, and Linux."""
    text = _UNSAFE_FILENAME_CHARS.sub("", str(value or "").strip())
    text = _FILENAME_WHITESPACE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip(" ._")
    text = text[:max_length].rstrip(" ._")
    if text.upper() in _WINDOWS_RESERVED_NAMES:
        text += "_"
    return text


def document_export_filenames(
    metadata: dict[str, Any], model_number: int, extension: str
) -> tuple[str, str]:
    """Build matching student/answer filenames from printed exam metadata."""
    parts = [
        part
        for part in (
            safe_filename_part(metadata.get("exam_title")),
            safe_filename_part(metadata.get("class_name")),
        )
        if part
    ]
    prefix = "_".join(parts) if parts else "Exam"
    ext = extension.lower().lstrip(".")
    base = f"{prefix}_Model_{int(model_number)}"
    return f"{base}.{ext}", f"Answers_{base}.{ext}"


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


def group_exam_sections(questions: dict[str, Any], language: str = "en") -> list[dict[str, Any]]:
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
            "label": export_type_label(qtype, language),
            "items": items,
        }
        if qtype == "fill_in_the_blank":
            section["word_bank"] = list(dict.fromkeys(items[0].get("word_bank") or []))
        sections.append(section)
    return sections
