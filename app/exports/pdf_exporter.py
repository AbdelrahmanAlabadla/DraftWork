"""Compact, print-ready PDF export for final validated exams (fpdf2).

The student copy is rendered first and never contains answer data.  A grouped
teacher answer key follows on separate pages.  The layout is deliberately
dense: A4 with 0.5-inch margins, row-level page breaking for objective
questions, and flexible ruled response areas for open-ended questions.
"""
from __future__ import annotations

import base64
import binascii
import io
from typing import Any

from fpdf import FPDF
from fpdf.enums import MethodReturnValue

from app.exports.common import RESPONSE_LINE_TEXT, group_exam_sections, response_line_count

PAGE_MARGIN_MM = 12.7  # exactly 0.5 inch
FOOTER_Y_MM = -9.5
BODY_FONT = "helvetica"
BODY_SIZE = 10.5
LINE_HEIGHT = 5.4
ANSWER_LINE_HEIGHT = 5.65  # 16 pt, matching the DOCX response-line spacing

INK = (31, 41, 55)
MUTED = (100, 116, 139)
ACCENT = (51, 65, 85)
SECTION_FILL = (232, 238, 245)
RULE = (180, 190, 202)

_REPLACEMENTS = {
    "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
    "\u2212": "-", "\u2192": "->", "\u00b7": "-",
}


def _latin(text: object) -> str:
    value = str(text or "")
    for src, dst in _REPLACEMENTS.items():
        value = value.replace(src, dst)
    return value.encode("latin-1", "replace").decode("latin-1")


def _clean(value: object, fallback: str = "") -> str:
    text = _latin(value).strip()
    return text or fallback


def _image_stream(data_url: object) -> io.BytesIO | None:
    """Decode an optional browser data URL without trusting its file name."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        return None
    try:
        _, encoded = data_url.split(",", 1)
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return None
    if not raw or len(raw) > 2_500_000:
        return None
    stream = io.BytesIO(raw)
    stream.name = "exam-logo"
    return stream


class _ExamPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(PAGE_MARGIN_MM, PAGE_MARGIN_MM, PAGE_MARGIN_MM)
        self.set_auto_page_break(auto=True, margin=PAGE_MARGIN_MM)
        self._running_title = "Generated Exam"
        self._running_model = ""
        self._copy_label = "Student copy"

    def header(self) -> None:
        # Deliberately blank: the exam title belongs only in the first-page
        # title block, not in a repeated running header.
        self.set_y(self.t_margin)
        self.set_text_color(*INK)

    def footer(self) -> None:
        self.set_y(FOOTER_Y_MM)
        self.set_x(self.l_margin)
        self.set_font(BODY_FONT, "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 4, f"Page {self.page_no()} of {{nb}}", align="C")
        self.set_text_color(*INK)

    def ensure_space(self, height: float) -> bool:
        """Start a page only when the requested minimum block cannot fit."""
        if self.get_y() + height <= self.page_break_trigger:
            return False
        self.add_page()
        self.set_x(self.l_margin)
        return True


def _text_height(pdf: _ExamPDF, text: str, width: float, line_height: float = LINE_HEIGHT) -> float:
    height = pdf.multi_cell(
        width,
        line_height,
        _latin(text),
        dry_run=True,
        output=MethodReturnValue.HEIGHT,
    )
    return max(float(height), line_height)


def _field(value: object, underscore_count: int) -> str:
    return _clean(value) or ("_" * underscore_count)


def _draw_logo(pdf: _ExamPDF, data_url: object, x: float, y: float) -> None:
    stream = _image_stream(data_url)
    if stream is None:
        return
    try:
        pdf.image(stream, x=x, y=y, w=20, h=13, keep_aspect_ratio=True)
    except Exception:
        # A malformed/unsupported optional logo must never prevent export.
        return


def _draw_exam_header(pdf: _ExamPDF, metadata: dict[str, Any], model_number: int) -> None:
    title = _clean(metadata.get("exam_title"), "Examination")
    _draw_logo(pdf, metadata.get("left_logo_data"), pdf.l_margin, pdf.get_y() + 1)
    _draw_logo(pdf, metadata.get("right_logo_data"), pdf.w - pdf.r_margin - 20, pdf.get_y() + 1)

    y = pdf.get_y()
    pdf.set_xy(pdf.l_margin + 22, y + 1)
    pdf.set_font(BODY_FONT, "B", 16)
    pdf.set_text_color(*INK)
    pdf.multi_cell(pdf.epw - 44, 7, title, align="C", new_x="LEFT", new_y="NEXT")
    pdf.set_x(pdf.l_margin + 22)
    pdf.set_font(BODY_FONT, "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(pdf.epw - 44, 5, f"Model {model_number}", align="C")
    pdf.set_y(max(pdf.get_y() + 7, y + 17))

    meta_values = [
        ("Class", _field(metadata.get("class_name"), 9)),
        ("Duration", _field(metadata.get("duration"), 7)),
        ("Date", _field(metadata.get("exam_date"), 9)),
        ("Teacher", _field(metadata.get("teacher_name"), 9)),
    ]
    col_w = pdf.epw / len(meta_values)
    pdf.set_fill_color(246, 248, 251)
    pdf.set_draw_color(*RULE)
    for label, value in meta_values:
        x, cell_y = pdf.get_x(), pdf.get_y()
        pdf.rect(x, cell_y, col_w, 8, style="DF")
        pdf.set_xy(x + 1.5, cell_y + 1.3)
        pdf.set_font(BODY_FONT, "B", 7.2)
        pdf.set_text_color(*MUTED)
        pdf.cell(12.5, 5.2, f"{label}:")
        pdf.set_font(BODY_FONT, "", 8.5)
        pdf.set_text_color(*INK)
        pdf.cell(col_w - 15, 5.2, _latin(value))
        pdf.set_xy(x + col_w, cell_y)
    pdf.set_y(pdf.get_y() + 9.5)

    pdf.set_font(BODY_FONT, "", 9.5)
    student_w = pdf.epw * 0.67
    pdf.cell(student_w, 7, "Student Name: " + "_" * 34)
    pdf.cell(pdf.epw - student_w, 7, "Class: " + "_" * 12)
    pdf.ln(9)


def _draw_section_heading(pdf: _ExamPDF, label: str, minimum_after: float = 12) -> None:
    pdf.ensure_space(7 + minimum_after)
    pdf.set_fill_color(*SECTION_FILL)
    pdf.set_draw_color(*RULE)
    pdf.set_text_color(*ACCENT)
    pdf.set_font(BODY_FONT, "B", 11)
    pdf.cell(0, 7, _latin(label), border="B", fill=True)
    pdf.ln(9)
    pdf.set_text_color(*INK)


def _draw_question_stem(pdf: _ExamPDF, number: int, text: str, minimum_after: float = 0) -> None:
    stem = f"Q{number}. {_clean(text, '(missing question text)')}"
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    height = _text_height(pdf, stem, pdf.epw)
    pdf.ensure_space(height + minimum_after)
    pdf.multi_cell(0, LINE_HEIGHT, stem, new_x="LEFT", new_y="NEXT")


def _draw_mcq_option(pdf: _ExamPDF, letter: str, option: object, width: float) -> float:
    text = f"{letter}. {_clean(option)}"
    pdf.set_font(BODY_FONT, "", 9.6)
    return _text_height(pdf, text, width - 2, 5.0)


def _draw_mcq(pdf: _ExamPDF, item: dict[str, Any]) -> None:
    _draw_question_stem(pdf, item["number"], item["text"], minimum_after=5)
    options = list((item.get("options") or {}).items())
    if not options:
        pdf.ln(2)
        return

    gutter = 5.0
    col_w = (pdf.epw - gutter) / 2
    two_column = all(_draw_mcq_option(pdf, str(letter), option, col_w) <= 10.1 for letter, option in options)
    pdf.set_font(BODY_FONT, "", 9.6)

    if two_column:
        for index in range(0, len(options), 2):
            pair = options[index:index + 2]
            heights = [_draw_mcq_option(pdf, str(letter), option, col_w) for letter, option in pair]
            row_h = max(heights) + 1.2
            pdf.ensure_space(row_h)
            row_y = pdf.get_y()
            for col, (letter, option) in enumerate(pair):
                x = pdf.l_margin + col * (col_w + gutter)
                pdf.set_xy(x + 2, row_y)
                pdf.multi_cell(col_w - 2, 5.0, _latin(f"{letter}. {_clean(option)}"), new_x="LEFT", new_y="NEXT")
            pdf.set_y(row_y + row_h)
    else:
        for letter, option in options:
            text = _latin(f"{letter}. {_clean(option)}")
            height = _draw_mcq_option(pdf, str(letter), option, pdf.epw - 4) + 0.8
            pdf.ensure_space(height)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(pdf.epw - 4, 5.0, text, new_x="LEFT", new_y="NEXT")
            pdf.ln(0.8)
    pdf.ln(2.2)


def _draw_word_bank(pdf: _ExamPDF, words: list[str]) -> None:
    if not words:
        return
    bank_text = "   |   ".join(_latin(word) for word in words)
    pdf.set_font(BODY_FONT, "", 9.4)
    text_h = _text_height(pdf, bank_text, pdf.epw - 8, 5.0)
    box_h = text_h + 9
    pdf.ensure_space(box_h + 5)
    x, y = pdf.l_margin, pdf.get_y()
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(*RULE)
    pdf.rect(x, y, pdf.epw, box_h, style="DF")
    pdf.set_xy(x + 4, y + 2)
    pdf.set_font(BODY_FONT, "B", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(pdf.epw - 8, 4, "WORD BANK", align="C")
    pdf.set_xy(x + 4, y + 6)
    pdf.set_font(BODY_FONT, "", 9.4)
    pdf.set_text_color(*INK)
    pdf.multi_cell(pdf.epw - 8, 5.0, bank_text, align="C", new_x="LEFT", new_y="NEXT")
    pdf.set_y(y + box_h + 4)


def _draw_fill_blank(pdf: _ExamPDF, item: dict[str, Any]) -> None:
    # The blank is part of the question stem.  Do not add a second ruled line
    # between fill-in-the-blank questions.
    _draw_question_stem(pdf, item["number"], item["text"])
    pdf.ln(4)


def _draw_true_false(pdf: _ExamPDF, item: dict[str, Any]) -> None:
    statement = f"Q{item['number']}. {_clean(item['text'], '(missing statement)')}"
    answer_w = 28.0
    text_w = pdf.epw - answer_w - 3
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    height = _text_height(pdf, statement, text_w)
    if height > LINE_HEIGHT + 0.1:
        full_height = _text_height(pdf, statement, pdf.epw)
        pdf.ensure_space(full_height + 7)
        pdf.multi_cell(pdf.epw, LINE_HEIGHT, statement, new_x="LEFT", new_y="NEXT")
        pdf.set_font(BODY_FONT, "", 9.5)
        pdf.cell(0, 5, "(   ) True     (   ) False", align="R")
        pdf.ln(8)
        return
    pdf.ensure_space(height + 2)
    y = pdf.get_y()
    pdf.multi_cell(text_w, LINE_HEIGHT, statement, new_x="LEFT", new_y="NEXT")
    pdf.set_xy(pdf.l_margin + text_w + 3, y)
    pdf.set_font(BODY_FONT, "", 9.5)
    pdf.cell(answer_w, LINE_HEIGHT, "(   ) True   (   ) False", align="R")
    pdf.set_y(y + height + 3)


def _draw_answer_lines(pdf: _ExamPDF, count: int) -> None:
    for _ in range(count):
        pdf.ensure_space(ANSWER_LINE_HEIGHT)
        pdf.set_x(pdf.l_margin + 4)
        pdf.set_font(BODY_FONT, "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(pdf.epw - 4, ANSWER_LINE_HEIGHT, RESPONSE_LINE_TEXT)
        pdf.ln(ANSWER_LINE_HEIGHT)


def _draw_open_ended(
    pdf: _ExamPDF,
    item: dict[str, Any],
    qtype: str,
) -> None:
    text = f"Q{item['number']}. {_clean(item['text'], '(missing question text)')}"
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    stem_h = _text_height(pdf, text, pdf.epw)
    # Keep the stem with the first response line. Remaining lines may continue
    # on the next page, using the same fixed count as DOCX and browser preview.
    pdf.ensure_space(stem_h + ANSWER_LINE_HEIGHT + 2)
    pdf.multi_cell(0, LINE_HEIGHT, text, new_x="LEFT", new_y="NEXT")
    pdf.ln(1)
    _draw_answer_lines(pdf, response_line_count(qtype))
    pdf.ln(2.5)
    pdf.set_text_color(*INK)


def _draw_student_exam(
    pdf: _ExamPDF,
    exam: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    model_number = int(exam.get("model_number") or 1)
    title = _clean(metadata.get("exam_title") or exam.get("title"), "Examination")
    pdf._running_title = title
    pdf._running_model = f"Model {model_number}"
    pdf._copy_label = "Student copy"
    pdf.add_page()
    _draw_exam_header(pdf, metadata, model_number)

    sections = group_exam_sections(exam.get("questions") or {})
    for section_index, section in enumerate(sections):
        if section["qtype"] == "essay" and section_index > 0:
            pdf.add_page()
        minimum_after = 22 if section["qtype"] in {"short_answer", "essay"} else 18
        _draw_section_heading(pdf, section["label"], minimum_after=minimum_after)
        if section["qtype"] == "fill_in_the_blank":
            _draw_word_bank(pdf, section.get("word_bank") or [])

        items = section["items"]
        for item in items:
            qtype = section["qtype"]
            if qtype == "mcq":
                _draw_mcq(pdf, item)
            elif qtype == "fill_in_the_blank":
                _draw_fill_blank(pdf, item)
            elif qtype == "true_false":
                _draw_true_false(pdf, item)
            else:
                _draw_open_ended(pdf, item, qtype)

    return sections


def _answer_text(item: dict[str, Any]) -> str:
    qtype = item["qtype"]
    if qtype == "mcq":
        return _clean(item.get("correct_answer"), "No answer supplied")
    if qtype == "true_false":
        return _clean(item.get("answer"), "No answer supplied")
    if qtype == "fill_in_the_blank":
        return ", ".join(_clean(answer) for answer in (item.get("answers") or [])) or "No answer supplied"
    answer = _clean(item.get("reference_answer"), "No reference answer supplied")
    key_points = [_clean(point) for point in (item.get("key_points") or []) if _clean(point)]
    if key_points:
        answer += " | Key points: " + "; ".join(key_points)
    return answer


def _draw_answer_key(
    pdf: _ExamPDF,
    sections: list[dict[str, Any]],
    metadata: dict[str, Any],
    model_number: int,
) -> None:
    if not sections:
        return
    pdf._copy_label = "Teacher answer key"
    pdf.add_page()
    pdf.set_font(BODY_FONT, "B", 15)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, f"Answer Key - Model {model_number}", align="C")
    pdf.ln(11)

    for section in sections:
        _draw_section_heading(pdf, section["label"], minimum_after=11)
        for item in section["items"]:
            text = f"Q{item['number']}. {_answer_text(item)}"
            pdf.set_font(BODY_FONT, "", 9.5)
            height = _text_height(pdf, text, pdf.epw, 5.0)
            pdf.ensure_space(height + 2)
            pdf.multi_cell(0, 5.0, text, new_x="LEFT", new_y="NEXT")
            pdf.ln(1.5)


def render_exam_pdf(stored_record: dict[str, Any]) -> bytes:
    """Render all stored exam models to a compact student PDF + teacher key."""
    pdf = _ExamPDF()
    pdf.alias_nb_pages()
    metadata = dict(stored_record.get("metadata") or {})

    for exam in stored_record.get("exams") or []:
        if not exam.get("questions"):
            continue
        model_number = int(exam.get("model_number") or 1)
        sections = _draw_student_exam(pdf, exam, metadata)
        _draw_answer_key(pdf, sections, metadata, model_number)

    return bytes(pdf.output())
