"""Versioned, renderer-independent exam document representation.

Both the Python exporters and the browser preview consume this structure.  A
renderer never reads ``exam["questions"]`` directly; it receives one document
model with explicit paragraphs, inline/display math, native tables, options,
response areas, and teacher-only answer blocks.

The representation is plain JSON so it can be serialized, cached, tested, and
shared with JavaScript without a second transcription layer.
"""
from __future__ import annotations

import re
from typing import Any

from app.exports.common import (
    EXPORT_TYPE_LABELS_BY_LANG,
    HEADER_LABELS_BY_LANG,
    ANSWER_LABELS_BY_LANG,
    expand_fill_blank_text,
    exam_document_language,
    is_complete_equation_prompt,
    response_line_count,
)
from app.exports.content_parser import (
    IR_VERSION,
    normalize_numerals,
    parse_coordinate_table,
    parse_question_content,
)
from app.exports.math_renderer import latex_to_plain_text
from app.online.models import TYPE_ORDER, tf_answer_label

TF_CHOICES = {
    "en": "True (   ) False (   )",
    # The student-facing label is intentionally غلط, while the stored boolean
    # answer remains unchanged.
    "ar": "صح (   ) غلط (   )",
}

_MCQ_LETTERS = ("A", "B", "C", "D")


def _labels(language: str) -> dict[str, str]:
    return HEADER_LABELS_BY_LANG.get(language, HEADER_LABELS_BY_LANG["en"])


def _answer_labels(language: str) -> dict[str, str]:
    return ANSWER_LABELS_BY_LANG.get(language, ANSWER_LABELS_BY_LANG["en"])


def _plain_runs(text: object) -> list[dict[str, Any]]:
    value = normalize_numerals(text)
    return [{"type": "text", "text": value}] if value.strip() else []


def _paragraph(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "paragraph", "runs": runs}


def _text_block(text: object) -> list[dict[str, Any]]:
    value = normalize_numerals(text)
    return [_paragraph(_plain_runs(value))] if value.strip() else []


def _question_value(qtype: str, item: dict[str, Any]) -> str:
    if qtype == "true_false":
        return str(item.get("statement") or "")
    if qtype == "definition":
        return str(item.get("term") or "")
    if qtype == "equation":
        return str(item.get("equation") or "")
    if qtype == "fill_in_the_blank":
        return expand_fill_blank_text(item.get("question") or "")
    return str(item.get("question") or "")


def _split_legacy_solution(value: str) -> tuple[str, str]:
    """Split only explicit line-start solution markers.

    Some generated equation fields contain prompt and worked solution in one
    string.  A conservative split is safe only when a full solution marker
    starts a line and substantive prompt text precedes it.  The solution is
    preserved in the answer key, never discarded.
    """
    match = re.search(
        r"(?m)^\s*(?:الحل|Solution|solution)\s*:\s*",
        value,
    )
    if not match:
        return value, ""
    before = value[:match.start()].strip()
    after = value[match.end():].strip()
    if len(before) < 20 or not after:
        return value, ""
    return before, after


def _answer_blocks(
    qtype: str,
    item: dict[str, Any],
    language: str,
    legacy_solution: str = "",
) -> list[dict[str, Any]]:
    labels = _answer_labels(language)
    if qtype == "mcq":
        return _text_block(str(item.get("correct_answer") or labels["missing"]))
    if qtype == "true_false":
        if item.get("answer") is None:
            return _text_block(labels["missing"])
        label = tf_answer_label(item.get("answer"), language) or labels["missing"]
        if language == "ar" and label == "خطأ":
            label = "غلط"
        return _text_block(label)
    if qtype == "fill_in_the_blank":
        answer = ", ".join(str(a) for a in (item.get("answers") or []) if str(a).strip())
        return _text_block(answer or labels["missing"])
    if qtype in {"equation", "word_problem"}:
        blocks: list[dict[str, Any]] = []
        if legacy_solution:
            solution_blocks, _ = parse_question_content(legacy_solution, language=language)
            blocks.extend(solution_blocks)
        for step in (item.get("solution_steps") or []):
            step = str(step).strip()
            if not step:
                continue
            step_blocks, _ = parse_question_content(step, language=language)
            blocks.extend(step_blocks)
        final = str(item.get("final_answer") or "").strip()
        if final:
            final_blocks, _ = parse_question_content(final, language=language)
            blocks.extend(final_blocks)
        if not blocks:
            blocks = _text_block(labels["missing_final"])
        return blocks
    answer = str(item.get("reference_answer") or labels["missing_reference"])
    blocks = _text_block(answer)
    points = [str(point).strip() for point in (item.get("key_points") or []) if str(point).strip()]
    if points:
        blocks.extend(_text_block(f"{labels['key_points']}: " + "; ".join(points)))
    return blocks


def _option_blocks(value: object, language: str) -> list[dict[str, Any]]:
    text = normalize_numerals(value)
    coordinate = parse_coordinate_table(text, language=language)
    if coordinate and re.match(r"^(?:النقاط|نقاط|Points?)\s*:", text.strip(), re.IGNORECASE):
        return [coordinate]
    blocks, _ = parse_question_content(text, language=language)
    return blocks


def _section_heading(qtype: str, language: str) -> str:
    labels = EXPORT_TYPE_LABELS_BY_LANG.get(language, EXPORT_TYPE_LABELS_BY_LANG["en"])
    return labels.get(qtype, qtype)


def _sections(questions: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]], list[str] | None]]:
    result: list[tuple[str, list[dict[str, Any]], list[str] | None]] = []
    for qtype in TYPE_ORDER:
        section = questions.get(qtype)
        if not section:
            continue
        word_bank: list[str] | None = None
        if qtype == "fill_in_the_blank" and isinstance(section, dict):
            word_bank = [str(w).strip() for w in (section.get("word_bank") or []) if str(w).strip()]
            items = list(section.get("items") or [])
        elif isinstance(section, list):
            items = list(section)
        else:
            items = []
        if items:
            result.append((qtype, items, word_bank))
    return result


def build_document_ir(
    exam: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    kind: str = "student",
) -> dict[str, Any]:
    """Build the versioned document model for one exam model.

    ``kind`` is ``student`` or ``answers``.  Both modes contain the same
    ``answer_blocks`` for teacher use; student renderers must ignore them.
    """
    metadata = dict(metadata or {})
    language = exam_document_language(exam, metadata)
    direction = "rtl" if language == "ar" else "ltr"
    labels = _labels(language)
    model_number = int(exam.get("model_number") or 1)
    title = str(metadata.get("exam_title") or exam.get("title") or "").strip()
    if not title:
        title = "امتحان" if language == "ar" else "Examination"

    document: dict[str, Any] = {
        "version": IR_VERSION,
        "kind": kind,
        "language": language,
        "direction": direction,
        "metadata": {
            "title": title,
            "model_label": labels["model"].format(n=model_number),
            "answer_key_title": labels["answer_key"].format(n=model_number),
            "class_name": metadata.get("class_name") or "",
            "duration": metadata.get("duration") or "",
            "teacher_name": metadata.get("teacher_name") or "",
            "exam_date": metadata.get("exam_date") or "",
            "student_name_label": labels["student_name"].rstrip(": "),
            "class_label": labels["class"],
            "duration_label": labels["duration"],
            "teacher_label": labels["teacher"],
            "date_label": labels["date"],
            "true_false_choices": TF_CHOICES.get(language, TF_CHOICES["en"]),
            "page_label": "صفحة" if language == "ar" else "Page",
            "of_label": "من" if language == "ar" else "of",
        },
        "sections": [],
        "diagnostics": [],
    }

    for qtype, items, word_bank in _sections(exam.get("questions") or {}):
        section: dict[str, Any] = {
            "qtype": qtype,
            "heading": _section_heading(qtype, language),
            "items": [],
        }
        if word_bank:
            section["word_bank"] = word_bank
        for local_number, item in enumerate(items, start=1):
            raw_value = _question_value(qtype, item)
            legacy_solution = ""
            if qtype == "equation":
                raw_value, legacy_solution = _split_legacy_solution(raw_value)
            blocks, diagnostics = parse_question_content(raw_value, language=language)
            if legacy_solution:
                diagnostics.append({
                    "code": "legacy_solution_moved_to_answer_key",
                    "message": "A line-start solution marker was moved to the answer key.",
                    "source": "",
                })
            options = []
            if qtype == "mcq":
                for letter in _MCQ_LETTERS:
                    value = (item.get("options") or {}).get(letter)
                    if value is not None:
                        options.append({
                            "label": letter,
                            "blocks": _option_blocks(value, language),
                        })
            answer_blocks = _answer_blocks(qtype, item, language, legacy_solution)
            question = {
                "number": local_number,
                "question_id": item.get("question_id"),
                "qtype": qtype,
                "blocks": blocks,
                "options": options,
                "answer_blocks": answer_blocks,
                "response_lines": response_line_count(qtype) if kind == "student" else 0,
                "diagnostics": diagnostics,
            }
            if qtype == "true_false":
                question["true_false_choices"] = TF_CHOICES.get(language, TF_CHOICES["en"])
            section["items"].append(question)
            document["diagnostics"].extend(
                {"qtype": qtype, "question_id": item.get("question_id"), **diag}
                for diag in diagnostics
            )
        document["sections"].append(section)

    return document


def document_to_json(document: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy (the IR only uses built-in types)."""
    return document
