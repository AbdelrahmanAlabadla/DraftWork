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
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import MethodReturnValue

from app.exports.common import RESPONSE_LINE_TEXT, group_exam_sections, response_line_count

PAGE_MARGIN_MM = 12.7  # exactly 0.5 inch
FOOTER_Y_MM = -9.5
BODY_FONT = "helvetica"
ARABIC_FONT = "Amiri"
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
    regular = FONT_DIR / "Amiri-Regular.ttf"
    bold = FONT_DIR / "Amiri-Bold.ttf"
    if not (regular.is_file() and bold.is_file()):
        return False

    probe = FPDF()
    try:
        probe.add_font(ARABIC_FONT, "", str(regular))
        probe.add_font(ARABIC_FONT, "B", str(bold))
    except Exception:
        return False
    return True


def _align(pdf: "_ExamPDF") -> str:
    return "R" if pdf.rtl else "L"


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
        (labels["class"], _field(metadata.get("class_name"), 9)),
        (labels["duration"], _field(metadata.get("duration"), 7)),
        (labels["date"], _field(metadata.get("exam_date"), 9)),
        (labels["teacher"], _field(metadata.get("teacher_name"), 9)),
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
    if pdf.rtl:
        pdf.cell(pdf.epw - student_w, 7, labels["class_suffix"] + "_" * 12, align="R")
        pdf.cell(student_w, 7, "_" * 34 + labels["student_name"], align="R")
    else:
        pdf.cell(student_w, 7, labels["student_name"] + "_" * 34)
        pdf.cell(pdf.epw - student_w, 7, labels["class_suffix"] + "_" * 12)
    pdf.ln(9)


def _draw_section_heading(pdf: _ExamPDF, label: str, minimum_after: float = 12) -> None:
    pdf.ensure_space(7 + minimum_after)
    pdf.set_fill_color(*SECTION_FILL)
    pdf.set_draw_color(*RULE)
    pdf.set_text_color(*ACCENT)
    pdf.set_font(BODY_FONT, "B", 11)
    pdf.cell(0, 7, _latin(label), border="B", fill=True, align=_align(pdf))
    pdf.ln(9)
    pdf.set_text_color(*INK)


def _draw_question_stem(pdf: _ExamPDF, number: int, text: str, minimum_after: float = 0) -> None:
    marker = f"{number}. " if pdf.rtl else f"Q{number}. "
    stem = marker + _clean(text, "(missing question text)")
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    height = _text_height(pdf, stem, pdf.epw)
    pdf.ensure_space(height + minimum_after)
    pdf.multi_cell(0, LINE_HEIGHT, stem, align=_align(pdf), new_x="LEFT", new_y="NEXT")


def _draw_mcq_option(pdf: _ExamPDF, letter: str, option: object, width: float) -> float:
    text = f"{_clean(option)} .{letter}" if pdf.rtl else f"{letter}. {_clean(option)}"
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
                opt_text = f"{_clean(option)} .{letter}" if pdf.rtl else f"{letter}. {_clean(option)}"
                pdf.multi_cell(col_w - 2, 5.0, _latin(opt_text), align=_align(pdf), new_x="LEFT", new_y="NEXT")
            pdf.set_y(row_y + row_h)
    else:
        for letter, option in options:
            opt_text = f"{_clean(option)} .{letter}" if pdf.rtl else f"{letter}. {_clean(option)}"
            height = _draw_mcq_option(pdf, str(letter), option, pdf.epw - 4) + 0.8
            pdf.ensure_space(height)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(pdf.epw - 4, 5.0, _latin(opt_text), align=_align(pdf), new_x="LEFT", new_y="NEXT")
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
    _draw_question_stem(pdf, item["number"], item["text"])
    pdf.ln(4)


def _draw_true_false(pdf: _ExamPDF, item: dict[str, Any]) -> None:
    from app.exports.common import TRUE_FALSE_CHOICES_BY_LANG

    choices_wide = TRUE_FALSE_CHOICES_BY_LANG.get(pdf.language, TRUE_FALSE_CHOICES_BY_LANG["en"])
    choices_compact = (
        "(   ) صح   (   ) خطأ" if pdf.rtl else "(   ) True   (   ) False"
    )
    marker = f"{item['number']}. " if pdf.rtl else f"Q{item['number']}. "
    statement = marker + _clean(item['text'], '(missing statement)')
    answer_w = 28.0
    text_w = pdf.epw - answer_w - 3
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    height = _text_height(pdf, statement, text_w)
    if height > LINE_HEIGHT + 0.1:
        full_height = _text_height(pdf, statement, pdf.epw)
        pdf.ensure_space(full_height + 7)
        pdf.multi_cell(pdf.epw, LINE_HEIGHT, statement, align=_align(pdf), new_x="LEFT", new_y="NEXT")
        pdf.set_font(BODY_FONT, "", 9.5)
        pdf.cell(0, 5, choices_wide, align="L" if pdf.rtl else "R")
        pdf.ln(8)
        return
    pdf.ensure_space(height + 2)
    y = pdf.get_y()
    pdf.multi_cell(text_w, LINE_HEIGHT, statement, align=_align(pdf), new_x="LEFT", new_y="NEXT")
    pdf.set_xy(pdf.l_margin + text_w + 3, y)
    pdf.set_font(BODY_FONT, "", 9.5)
    pdf.cell(answer_w, LINE_HEIGHT, choices_compact, align="R")
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
    marker = f"{item['number']}. " if pdf.rtl else f"Q{item['number']}. "
    text = marker + _clean(item['text'], '(missing question text)')
    pdf.set_font(BODY_FONT, "B", BODY_SIZE)
    stem_h = _text_height(pdf, text, pdf.epw)
    # Keep the stem with the first response line. Remaining lines may continue
    # on the next page, using the same fixed count as DOCX and browser preview.
    pdf.ensure_space(stem_h + ANSWER_LINE_HEIGHT + 2)
    pdf.multi_cell(0, LINE_HEIGHT, text, align=_align(pdf), new_x="LEFT", new_y="NEXT")
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


def _answer_text(item: dict[str, Any], language: str = "en") -> str:
    qtype = item["qtype"]
    if qtype == "mcq":
        return _clean(item.get("correct_answer"), "No answer supplied")
    if qtype == "true_false":
        from app.online.models import tf_answer_label

        raw = _clean(item.get("answer"), "No answer supplied")
        if raw.lower() in {"true", "false"}:
            return tf_answer_label(raw, language)
        return raw
    if qtype == "fill_in_the_blank":
        return ", ".join(_clean(answer) for answer in (item.get("answers") or [])) or "No answer supplied"
    answer = _clean(item.get("reference_answer"), "No reference answer supplied")
    key_points = [_clean(point) for point in (item.get("key_points") or []) if _clean(point)]
    if key_points:
        separator = " | نقاط رئيسية: " if language == "ar" else " | Key points: "
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
            marker = f"{item['number']}. " if pdf.rtl else f"Q{item['number']}. "
            text = marker + _answer_text(item, language=pdf.language)
            pdf.set_font(BODY_FONT, "", 9.5)
            height = _text_height(pdf, text, pdf.epw, 5.0)
            pdf.ensure_space(height + 2)
            pdf.multi_cell(0, 5.0, text, align=_align(pdf), new_x="LEFT", new_y="NEXT")
            pdf.ln(1.5)


def _render_model_pdf(
    exam: dict[str, Any], metadata: dict[str, Any], *, answers: bool
) -> bytes:
    """Render one model as either a student exam or a standalone answer key."""
    global _UNICODE_OUTPUT, BODY_FONT
    metadata = dict(metadata or {})
    language = str(metadata.get("document_language") or "en")
    has_unicode = _register_arabic_font()
    try:
        pdf = _ExamPDF(language)
        pdf.alias_nb_pages()
        if has_unicode and pdf.rtl:
            BODY_FONT = ARABIC_FONT
            _UNICODE_OUTPUT = True
            pdf.add_font(ARABIC_FONT, "", str(FONT_DIR / "Amiri-Regular.ttf"))
            pdf.add_font(ARABIC_FONT, "B", str(FONT_DIR / "Amiri-Bold.ttf"))
            try:
                pdf.set_text_shaping(True)
            except Exception:
                pass
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
