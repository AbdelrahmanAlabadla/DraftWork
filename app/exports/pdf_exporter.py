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
import re
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import MethodReturnValue, TextDirection

from app.exports.common import (
    expand_fill_blank_text,
    exam_document_language,
    group_exam_sections,
    response_line_count,
)
from app.language import detect_language
from app.exports.math_renderer import render_math

PAGE_MARGIN_MM = 12.7  # exactly 0.5 inch
FOOTER_Y_MM = -9.5
BODY_FONT = "helvetica"
ARABIC_FONT = "Tajawal"
MATH_FONT = "DejaVuSans"
FONT_DIR = Path(__file__).parent / "fonts"
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

# Set by render_exam_pdf: when a Unicode-capable TTF is registered, text is NOT
# crushed down to latin-1 (which would destroy Arabic) and shaping is enabled.
_UNICODE_OUTPUT = False


def _latin(text: object) -> str:
    value = str(text or "")
    for src, dst in _REPLACEMENTS.items():
        value = value.replace(src, dst)
    if _UNICODE_OUTPUT:
        return value
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
    def __init__(self, language: str = "en") -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(PAGE_MARGIN_MM, PAGE_MARGIN_MM, PAGE_MARGIN_MM)
        self.set_auto_page_break(auto=True, margin=PAGE_MARGIN_MM)
        self.language = language
        self.rtl = language == "ar"
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


def _register_arabic_font() -> bool:
    """Check the bundled Amiri TTFs load correctly. Returns False when
    unavailable so the exporter degrades to Helvetica instead of failing."""
    regular = FONT_DIR / "Tajawal-Regular.ttf"
    bold = FONT_DIR / "Tajawal-Bold.ttf"
    if not (regular.is_file() and bold.is_file()):
        return False

    probe = FPDF()
    try:
        probe.add_font(ARABIC_FONT, "", str(regular))
        probe.add_font(ARABIC_FONT, "B", str(bold))
    except Exception:
        return False
    return True


def _register_math_font() -> bool:
    """Check that the bundled Unicode font can render calculation notation."""
    regular = FONT_DIR / "DejaVuSans.ttf"
    bold = FONT_DIR / "DejaVuSans-Bold.ttf"
    if not (regular.is_file() and bold.is_file()):
        return False
    probe = FPDF()
    try:
        probe.add_font(MATH_FONT, "", str(regular))
        probe.add_font(MATH_FONT, "B", str(bold))
    except Exception:
        return False
    return True


_SCRIPT_LETTER_RE = re.compile(r"[A-Za-z\u00c0-\u024f\u0600-\u06ff]")


def _text_is_rtl(text: object, fallback: bool = False) -> bool:
    value = str(text or "")
    if not _SCRIPT_LETTER_RE.search(value):
        return fallback
    return detect_language(value) == "ar"


def _align(pdf: "_ExamPDF", text: object = "") -> str:
    return "R" if _text_is_rtl(text, pdf.rtl) else "L"


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


def _draw_free_field(
    pdf: _ExamPDF,
    x: float,
    y: float,
    width: float,
    label: str,
    value: object,
    *,
    rtl: bool,
    size: float = 8.5,
) -> None:
    """Draw a label and value/blank line without a box or shaded background."""
    label_text = f"{label}:"
    clean_value = _clean(value)
    pdf.set_font(BODY_FONT, "B", size)
    label_w = min(width * 0.46, pdf.get_string_width(label_text) + 2.5)
    value_w = max(8.0, width - label_w - 1.5)
    pdf.set_text_color(*MUTED)
    if rtl:
        pdf.set_xy(x + width - label_w, y)
        pdf.cell(label_w, 5.5, label_text, align="R")
        value_x = x
    else:
        pdf.set_xy(x, y)
        pdf.cell(label_w, 5.5, label_text, align="L")
        value_x = x + label_w + 1.5

    pdf.set_font(BODY_FONT, "", size)
    pdf.set_text_color(*INK)
    if clean_value:
        pdf.set_xy(value_x, y)
        pdf.cell(value_w, 5.5, clean_value, align="R" if rtl else "L")
    else:
        line_y = y + 4.8
        pdf.set_draw_color(*RULE)
        pdf.line(value_x + 0.8, line_y, value_x + value_w - 0.8, line_y)


def _draw_exam_header(pdf: _ExamPDF, metadata: dict[str, Any], model_number: int) -> None:
    from app.exports.common import HEADER_LABELS_BY_LANG

    labels = HEADER_LABELS_BY_LANG.get(pdf.language, HEADER_LABELS_BY_LANG["en"])
    title = _clean(metadata.get("exam_title"), "Examination")
    if pdf.rtl:
        # Mirror the logo positions for RTL layout.
        _draw_logo(pdf, metadata.get("left_logo_data"), pdf.w - pdf.r_margin - 20, pdf.get_y() + 1)
        _draw_logo(pdf, metadata.get("right_logo_data"), pdf.l_margin, pdf.get_y() + 1)
    else:
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
    pdf.cell(pdf.epw - 44, 5, labels["model"].format(n=model_number), align="C")
    pdf.set_y(max(pdf.get_y() + 7, y + 17))

    meta_values = [
        (labels["duration"], metadata.get("duration")),
        (labels["teacher"], metadata.get("teacher_name")),
        (labels["date"], metadata.get("exam_date")),
    ]
    col_w = pdf.epw / len(meta_values)
    meta_y = pdf.get_y()
    for index, (label, value) in enumerate(meta_values):
        visual_col = len(meta_values) - 1 - index if pdf.rtl else index
        _draw_free_field(
            pdf,
            pdf.l_margin + visual_col * col_w,
            meta_y,
            col_w - 3,
            label,
            value,
            rtl=pdf.rtl,
            size=8.2,
        )
    pdf.set_y(meta_y + 8)

    student_w = pdf.epw * 0.67
    if pdf.rtl:
        _draw_free_field(
            pdf, pdf.l_margin, pdf.get_y(), pdf.epw - student_w - 3,
            labels["class"], metadata.get("class_name"), rtl=True, size=9.5,
        )
        _draw_free_field(
            pdf, pdf.l_margin + pdf.epw - student_w, pdf.get_y(), student_w,
            labels["student_name"].rstrip(": "), None, rtl=True, size=9.5,
        )
    else:
        _draw_free_field(
            pdf, pdf.l_margin, pdf.get_y(), student_w - 3,
            labels["student_name"].rstrip(": "), None, rtl=False, size=9.5,
        )
        _draw_free_field(
            pdf, pdf.l_margin + student_w, pdf.get_y(), pdf.epw - student_w,
            labels["class"], metadata.get("class_name"), rtl=False, size=9.5,
        )
    pdf.ln(9)


def _draw_section_heading(pdf: _ExamPDF, label: str, minimum_after: float = 12) -> None:
    pdf.ensure_space(7 + minimum_after)
    pdf.set_fill_color(*SECTION_FILL)
    pdf.set_draw_color(*RULE)
    pdf.set_text_color(*ACCENT)
    pdf.set_font(BODY_FONT, "B", 11)
    pdf.cell(0, 7, _latin(label), border="B", fill=True, align=_align(pdf, label))
    pdf.ln(9)
    pdf.set_text_color(*INK)


def _draw_question_stem(pdf: _ExamPDF, number: int, text: str, minimum_after: float = 0) -> None:
    rtl = _text_is_rtl(text, pdf.rtl)
    marker = f"{number}. " if rtl else f"Q{number}. "
    stem = marker + _clean(text, "(missing question text)")
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    height = _text_height(pdf, stem, pdf.epw)
    pdf.ensure_space(height + minimum_after)
    pdf.multi_cell(0, LINE_HEIGHT, stem, align=_align(pdf, text), new_x="LEFT", new_y="NEXT")


def _draw_mcq_option(pdf: _ExamPDF, letter: str, option: object, width: float) -> float:
    text = _clean(option)
    pdf.set_font(BODY_FONT, "", 9.6)
    return _text_height(pdf, text, width - 11, 5.0)


def _draw_mcq_option_at(
    pdf: _ExamPDF,
    x: float,
    y: float,
    width: float,
    letter: str,
    option: object,
) -> float:
    """Render the Latin marker separately so bidi cannot move it after Arabic."""
    option_text = _clean(option)
    row_rtl = pdf.rtl
    marker_w = 9.0
    gap = 1.5
    text_w = width - marker_w - gap
    height = _draw_mcq_option(pdf, letter, option_text, width)
    marker_x = x + text_w + gap if row_rtl else x
    text_x = x if row_rtl else x + marker_w + gap

    pdf.set_xy(marker_x, y)
    pdf.set_font(BODY_FONT, "B", 9.6)
    # Arabic text shaping must not reorder a Latin marker from ``A.`` to
    # ``.A``. Keep the marker LTR, then restore automatic shaping for the
    # option text so Arabic and mixed formulas retain their own direction.
    shaping_enabled = pdf.text_shaping is not None
    if shaping_enabled:
        pdf.set_text_shaping(True, direction=TextDirection.LTR)
    try:
        pdf.cell(marker_w, 5.0, f"{letter}.", align="R" if row_rtl else "L")
    finally:
        if shaping_enabled:
            pdf.set_text_shaping(True)
    pdf.set_xy(text_x, y)
    pdf.set_font(BODY_FONT, "", 9.6)
    pdf.multi_cell(
        text_w,
        5.0,
        option_text,
        # The row starts on the exam's logical side. The shaping engine still
        # preserves the option text's own bidi order (for example n = c / v).
        align="R" if row_rtl else "L",
        new_x="LEFT",
        new_y="NEXT",
    )
    return height


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
                visual_col = 1 - col if pdf.rtl else col
                x = pdf.l_margin + visual_col * (col_w + gutter)
                _draw_mcq_option_at(pdf, x + 2, row_y, col_w - 2, str(letter), option)
            pdf.set_y(row_y + row_h)
    else:
        for letter, option in options:
            height = _draw_mcq_option(pdf, str(letter), option, pdf.epw - 4) + 0.8
            pdf.ensure_space(height)
            row_y = pdf.get_y()
            _draw_mcq_option_at(
                pdf, pdf.l_margin + 4, row_y, pdf.epw - 4, str(letter), option
            )
            pdf.set_y(row_y + height)
            pdf.ln(0.8)
    pdf.ln(2.2)


def _draw_word_bank(pdf: _ExamPDF, words: list[str]) -> None:
    if not words:
        return
    from app.online.models import UI_STRINGS_BY_LANG

    strings = UI_STRINGS_BY_LANG.get(pdf.language, UI_STRINGS_BY_LANG["en"])
    bank_label = strings["word_bank"] if pdf.rtl else "WORD BANK"
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
    pdf.cell(pdf.epw - 8, 4, bank_label, align="C")
    pdf.set_xy(x + 4, y + 6)
    pdf.set_font(BODY_FONT, "", 9.4)
    pdf.set_text_color(*INK)
    pdf.multi_cell(pdf.epw - 8, 5.0, bank_text, align="C", new_x="LEFT", new_y="NEXT")
    pdf.set_y(y + box_h + 4)


def _draw_fill_blank(pdf: _ExamPDF, item: dict[str, Any]) -> None:
    # The blank is part of the question stem.  Do not add a second ruled line
    # between fill-in-the-blank questions.
    printed_item = dict(item)
    printed_item["text"] = expand_fill_blank_text(item.get("text"))
    _draw_question_stem(pdf, printed_item["number"], printed_item["text"])
    pdf.ln(4)


def _draw_true_false(pdf: _ExamPDF, item: dict[str, Any]) -> None:
    raw_statement = _clean(item['text'], '(missing statement)')
    statement_rtl = _text_is_rtl(raw_statement, pdf.rtl)
    marker = f"{item['number']}. " if statement_rtl else f"Q{item['number']}. "
    statement = marker + raw_statement
    choices = ("صح", "خطأ") if pdf.rtl else ("True", "False")
    choices_w = 40.0 if pdf.rtl else 43.0
    gutter = 3.0
    max_statement_w = pdf.epw - choices_w - gutter
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    statement_w = min(
        max_statement_w,
        max(42.0, pdf.get_string_width(statement) + 3.0),
    )
    statement_lines = pdf.multi_cell(
        statement_w,
        LINE_HEIGHT,
        statement,
        dry_run=True,
        output=MethodReturnValue.LINES,
    )
    height = _text_height(pdf, statement, statement_w)
    pdf.ensure_space(height + 4)
    row_y = pdf.get_y()
    if pdf.rtl:
        statement_x = pdf.w - pdf.r_margin - statement_w
    else:
        statement_x = pdf.l_margin
    pdf.set_xy(statement_x, row_y)
    pdf.multi_cell(
        statement_w,
        LINE_HEIGHT,
        statement,
        align=_align(pdf, raw_statement),
        new_x="LEFT",
        new_y="NEXT",
    )
    half = choices_w / 2
    choices_y = row_y + max(0.0, height - LINE_HEIGHT)
    last_line = statement_lines[-1] if statement_lines else statement
    last_line_w = min(statement_w, pdf.get_string_width(last_line))
    if pdf.rtl:
        last_line_start = statement_x + statement_w - last_line_w
        choices_x = max(pdf.l_margin, last_line_start - gutter - choices_w)
    else:
        last_line_end = statement_x + last_line_w
        choices_x = min(
            pdf.w - pdf.r_margin - choices_w,
            last_line_end + gutter,
        )
    for index, label in enumerate(choices):
        visual_index = 1 - index if pdf.rtl else index
        pdf.set_xy(choices_x + visual_index * half, choices_y)
        pdf.set_font(BODY_FONT, "", 9.5)
        pdf.cell(half, LINE_HEIGHT, f"{label} (   )", align="R" if pdf.rtl else "L")
    pdf.set_y(row_y + max(height, LINE_HEIGHT) + 3)


def _draw_answer_lines(pdf: _ExamPDF, count: int, *, rtl: bool = False) -> None:
    for _ in range(count):
        pdf.ensure_space(ANSWER_LINE_HEIGHT)
        y = pdf.get_y() + ANSWER_LINE_HEIGHT - 1.2
        start_x = pdf.l_margin if rtl else pdf.l_margin + 4
        end_x = pdf.w - pdf.r_margin - (4 if rtl else 0)
        pdf.set_draw_color(71, 85, 105)
        pdf.set_line_width(0.18)
        pdf.line(start_x, y, end_x, y)
        pdf.ln(ANSWER_LINE_HEIGHT)


def _draw_open_ended(
    pdf: _ExamPDF,
    item: dict[str, Any],
    qtype: str,
) -> None:
    raw_text = _clean(item['text'], '(missing question text)')
    text_rtl = _text_is_rtl(raw_text, pdf.rtl)
    marker = f"{item['number']}. " if text_rtl else f"Q{item['number']}. "
    text = marker + raw_text
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    stem_h = _text_height(pdf, text, pdf.epw)
    line_count = response_line_count(qtype)
    block_height = stem_h + 1 + (line_count * ANSWER_LINE_HEIGHT) + 2.5
    usable_page_height = pdf.page_break_trigger - pdf.t_margin
    # Keep a question and its complete writing area together whenever that
    # block can fit on one page. This prevents orphan response lines on the
    # next page while still allowing an exceptionally large block to flow.
    pdf.ensure_space(
        block_height if block_height <= usable_page_height
        else stem_h + ANSWER_LINE_HEIGHT + 2
    )
    pdf.multi_cell(0, LINE_HEIGHT, text, align=_align(pdf, raw_text), new_x="LEFT", new_y="NEXT")
    pdf.ln(1)
    _draw_answer_lines(pdf, line_count, rtl=text_rtl)
    pdf.ln(2.5)
    pdf.set_text_color(*INK)


def _draw_definition(pdf: _ExamPDF, item: dict[str, Any]) -> None:
    marker = f"{item['number']}. "
    raw_term = _clean(item.get("text"), "(missing term)")
    term = marker + raw_term + ": "
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    term_rtl = _text_is_rtl(raw_term, pdf.rtl)
    term_w = min(pdf.epw * 0.38, max(28.0, pdf.get_string_width(term) + 3.0))
    line_gap = 3.0
    line_w = pdf.epw - term_w - line_gap
    height = _text_height(pdf, term, term_w)
    pdf.ensure_space(height + 4)
    row_y = pdf.get_y()
    if term_rtl:
        term_x = pdf.w - pdf.r_margin - term_w
        line_start = pdf.l_margin
        line_end = term_x - line_gap
    else:
        term_x = pdf.l_margin
        line_start = term_x + term_w + line_gap
        line_end = pdf.w - pdf.r_margin
    pdf.set_xy(term_x, row_y)
    pdf.multi_cell(
        term_w,
        LINE_HEIGHT,
        term,
        align="R" if term_rtl else "L",
        new_x="LEFT",
        new_y="NEXT",
    )
    line_y = row_y + min(height, LINE_HEIGHT) - 1.0
    pdf.set_draw_color(71, 85, 105)
    pdf.set_line_width(0.18)
    pdf.line(line_start, line_y, line_end, line_y)
    pdf.set_y(row_y + height)
    pdf.ln(4)


def _draw_equation(pdf: _ExamPDF, item: dict[str, Any]) -> None:
    equation = _clean(item.get("text"), "(missing equation)")
    number_gutter = 12.0
    equation_width = pdf.epw - (2 * number_gutter)
    rendered = render_math(equation)
    if rendered:
        natural_w = rendered.width_px / 220 * 25.4
        natural_h = rendered.height_px / 220 * 25.4
        draw_w = min(equation_width, natural_w)
        scale = draw_w / natural_w if natural_w else 1.0
        equation_h = max(LINE_HEIGHT, natural_h * scale)
    else:
        pdf.set_font(BODY_FONT, "B", BODY_SIZE)
        equation_h = _text_height(pdf, equation, equation_width)
    pdf.ensure_space(equation_h + ANSWER_LINE_HEIGHT + 4)
    y = pdf.get_y()
    number_x = pdf.w - pdf.r_margin - number_gutter if pdf.rtl else pdf.l_margin
    pdf.set_xy(number_x, y)
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    pdf.cell(number_gutter, LINE_HEIGHT, f"{item['number']}.", align="R" if pdf.rtl else "L")
    if rendered:
        stream = io.BytesIO(rendered.png)
        stream.name = "equation.png"
        image_x = pdf.l_margin + number_gutter + (equation_width - draw_w) / 2
        pdf.image(stream, x=image_x, y=y, w=draw_w, h=equation_h)
        pdf.set_y(y + equation_h)
    else:
        pdf.set_xy(pdf.l_margin + number_gutter, y)
        pdf.multi_cell(
            equation_width,
            LINE_HEIGHT,
            equation,
            align="C",
            new_x="LEFT",
            new_y="NEXT",
        )
    pdf.ln(1)
    _draw_answer_lines(pdf, response_line_count("equation"), rtl=pdf.rtl)
    pdf.ln(2.5)
    pdf.set_text_color(*INK)


def _draw_word_problem(pdf: _ExamPDF, item: dict[str, Any]) -> None:
    raw_text = _clean(item.get("text"), "(missing question text)")
    text = f"{item['number']}. " + raw_text
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    stem_h = _text_height(pdf, text, pdf.epw)
    pdf.ensure_space(stem_h + ANSWER_LINE_HEIGHT + 2)
    pdf.multi_cell(0, LINE_HEIGHT, text, align=_align(pdf, raw_text), new_x="LEFT", new_y="NEXT")
    pdf.ln(1)
    _draw_answer_lines(pdf, response_line_count("word_problem"), rtl=_text_is_rtl(raw_text, pdf.rtl))
    pdf.ln(2.5)
    pdf.set_text_color(*INK)


def _draw_student_exam(
    pdf: _ExamPDF,
    exam: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    model_number = int(exam.get("model_number") or 1)
    title = _clean(metadata.get("exam_title") or exam.get("title"), "Examination")
    from app.exports.common import HEADER_LABELS_BY_LANG

    labels = HEADER_LABELS_BY_LANG.get(pdf.language, HEADER_LABELS_BY_LANG["en"])
    pdf._running_title = title
    pdf._running_model = labels["model"].format(n=model_number)
    pdf._copy_label = "Student copy"
    pdf.add_page()
    _draw_exam_header(pdf, metadata, model_number)

    sections = group_exam_sections(exam.get("questions") or {}, language=pdf.language)
    for section_index, section in enumerate(sections):
        if section["qtype"] == "essay" and section_index > 0:
            pdf.add_page()
        minimum_after = 22 if section["qtype"] in {
            "short_answer", "equation", "word_problem", "essay"
        } else 18
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
            elif qtype == "definition":
                _draw_definition(pdf, item)
            elif qtype == "equation":
                _draw_equation(pdf, item)
            elif qtype == "word_problem":
                _draw_word_problem(pdf, item)
            else:
                _draw_open_ended(pdf, item, qtype)

    return sections


def _answer_text(item: dict[str, Any], language: str = "en") -> str:
    from app.exports.common import ANSWER_LABELS_BY_LANG

    labels = ANSWER_LABELS_BY_LANG.get(language, ANSWER_LABELS_BY_LANG["en"])
    qtype = item["qtype"]
    if qtype == "mcq":
        return _clean(item.get("correct_answer"), labels["missing"])
    if qtype == "true_false":
        from app.online.models import tf_answer_label

        raw = _clean(item.get("answer"), labels["missing"])
        if raw.lower() in {"true", "false"}:
            return tf_answer_label(raw, language)
        return raw
    if qtype == "fill_in_the_blank":
        return ", ".join(_clean(answer) for answer in (item.get("answers") or [])) or labels["missing"]
    if qtype in {"equation", "word_problem"}:
        steps = [_clean(step) for step in (item.get("solution_steps") or []) if _clean(step)]
        final = _clean(item.get("final_answer"), labels["missing_final"])
        return " → ".join([*steps, final])
    answer = _clean(item.get("reference_answer"), labels["missing_reference"])
    key_points = [_clean(point) for point in (item.get("key_points") or []) if _clean(point)]
    if key_points:
        separator = f" | {labels['key_points']}: "
        answer += separator + "; ".join(key_points)
    return answer


def _draw_answer_key(
    pdf: _ExamPDF,
    sections: list[dict[str, Any]],
    metadata: dict[str, Any],
    model_number: int,
) -> None:
    from app.exports.common import HEADER_LABELS_BY_LANG

    if not sections:
        return
    labels = HEADER_LABELS_BY_LANG.get(pdf.language, HEADER_LABELS_BY_LANG["en"])
    pdf._copy_label = "Teacher answer key"
    pdf.add_page()
    pdf.set_font(BODY_FONT, "B", 15)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, labels["answer_key"].format(n=model_number), align="C")
    pdf.ln(11)

    for section in sections:
        _draw_section_heading(pdf, section["label"], minimum_after=11)
        for item in section["items"]:
            answer = _answer_text(item, language=pdf.language)
            answer_rtl = pdf.rtl if item["qtype"] == "mcq" else _text_is_rtl(answer, pdf.rtl)
            marker = f"{item['number']}. " if answer_rtl else f"Q{item['number']}. "
            text = marker + answer
            pdf.set_font(BODY_FONT, "", 9.5)
            height = _text_height(pdf, text, pdf.epw, 5.0)
            pdf.ensure_space(height + 2)
            pdf.multi_cell(0, 5.0, text, align=_align(pdf, answer), new_x="LEFT", new_y="NEXT")
            pdf.ln(1.5)


def _render_model_pdf(
    exam: dict[str, Any], metadata: dict[str, Any], *, answers: bool
) -> bytes:
    """Render one model as either a student exam or a standalone answer key."""
    global _UNICODE_OUTPUT, BODY_FONT
    metadata = dict(metadata or {})
    language = exam_document_language(exam, metadata)
    has_arabic = _register_arabic_font()
    has_math = _register_math_font()
    try:
        pdf = _ExamPDF(language)
        pdf.alias_nb_pages()
        questions = exam.get("questions") or {}
        needs_math_font = any(questions.get(qtype) for qtype in ("equation", "word_problem"))
        content_blob = f"{metadata.get('exam_title', '')} {exam.get('questions', '')}"
        needs_arabic_font = pdf.rtl or detect_language(content_blob) == "ar"
        if needs_arabic_font and has_arabic:
            BODY_FONT = ARABIC_FONT
            _UNICODE_OUTPUT = True
            pdf.add_font(ARABIC_FONT, "", str(FONT_DIR / "Tajawal-Regular.ttf"))
            pdf.add_font(ARABIC_FONT, "B", str(FONT_DIR / "Tajawal-Bold.ttf"))
            if has_math:
                pdf.add_font(MATH_FONT, "", str(FONT_DIR / "DejaVuSans.ttf"))
                pdf.add_font(MATH_FONT, "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
                try:
                    pdf.set_fallback_fonts([MATH_FONT])
                except Exception:
                    pass
            try:
                pdf.set_text_shaping(True)
            except Exception:
                pass
        elif needs_math_font and has_math:
            BODY_FONT = MATH_FONT
            _UNICODE_OUTPUT = True
            pdf.add_font(MATH_FONT, "", str(FONT_DIR / "DejaVuSans.ttf"))
            pdf.add_font(MATH_FONT, "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        else:
            # Latin content keeps the original Helvetica layout.
            BODY_FONT = "helvetica"
            _UNICODE_OUTPUT = False

        if exam.get("questions"):
            model_number = int(exam.get("model_number") or 1)
            if answers:
                sections = group_exam_sections(
                    exam.get("questions") or {}, language=language
                )
                _draw_answer_key(pdf, sections, metadata, model_number)
            else:
                _draw_student_exam(pdf, exam, metadata)

        return bytes(pdf.output())
    finally:
        _UNICODE_OUTPUT = False


def render_exam_pdf(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    """Render one model's student exam without any answer-key pages."""
    return _render_model_pdf(exam, metadata, answers=False)


def render_answers_pdf(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    """Render one model's answer key as a separate PDF."""
    return _render_model_pdf(exam, metadata, answers=True)
