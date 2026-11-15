"""Print-ready A4 PDF export for the redesigned exam document.

The exporter consumes the shared document IR (``app.exports.document_ir``) and
draws black-and-white A4 pages.  It never parses raw question strings and never
turns an inline variable into a standalone image.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import MethodReturnValue, TextDirection

from app.exports.document_ir import build_document_ir
from app.exports.math_renderer import render_math

PAGE_MARGIN_MM = 15.0
CONTENT_WIDTH_MM = 180.0
BODY_SIZE = 10.5
LINE_HEIGHT = 5.6
SMALL_SIZE = 8.8
INK = (0, 0, 0)
RULE = (0, 0, 0)
ARABIC_FONT = "Amiri"
LATIN_FONT = "DejaVu"
FONT_DIR = Path(__file__).parent / "fonts"

_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


def _answer_text(item: dict[str, Any], language: str) -> str:
    """Compatibility helper used by export contract tests and extensions."""
    if item.get("qtype") == "true_false" or "answer" in item:
        answer = item.get("answer")
        if answer is True:
            return "صح" if language == "ar" else "True"
        if answer is False:
            return "غلط" if language == "ar" else "False"
    return str(item.get("correct_answer") or item.get("final_answer") or "")
_NUMBER_RE = re.compile(r"[-+]?\d")


def _has_arabic(value: object) -> bool:
    return bool(_ARABIC_RE.search(str(value or "")))


def _clean(value: object) -> str:
    return re.sub(r"[ \t]+", " ", str(value or "")).strip()


class _ExamPDF(FPDF):
    def __init__(self, language: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.language = language
        self.rtl = language == "ar"
        self.set_margins(PAGE_MARGIN_MM, PAGE_MARGIN_MM, PAGE_MARGIN_MM)
        self.set_auto_page_break(True, margin=13)
        self.set_draw_color(*RULE)
        self.set_text_color(*INK)
        self.body_font = ARABIC_FONT if self.rtl else LATIN_FONT
        self.add_font(ARABIC_FONT, "", str(FONT_DIR / "Amiri-Regular.ttf"))
        self.add_font(ARABIC_FONT, "B", str(FONT_DIR / "Amiri-Bold.ttf"))
        self.add_font(LATIN_FONT, "", str(FONT_DIR / "DejaVuSans.ttf"))
        self.add_font(LATIN_FONT, "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        try:
            self.set_fallback_fonts([LATIN_FONT, ARABIC_FONT])
        except Exception:
            pass

    def footer(self) -> None:
        # Page totals are painted in a final pass after pagination is complete.
        return

    def ensure_space(self, height: float) -> bool:
        if self.get_y() + height <= self.page_break_trigger:
            return False
        self.add_page()
        self.set_x(PAGE_MARGIN_MM)
        return True

    def set_token_font(self, token: dict[str, Any], size: float, bold: bool) -> None:
        font = ARABIC_FONT if token["font"] == "arabic" else LATIN_FONT
        self.set_font(font, "B" if bold else "", size)
        self.set_text_shaping(
            True,
            direction=TextDirection.RTL if token["dir"] == "rtl" else TextDirection.LTR,
        )


def _token(text: str, *, font: str | None = None, direction: str | None = None,
           nowrap: bool = False) -> dict[str, Any]:
    text = normalize_numerals(text)
    if not text:
        return {}
    if direction is None:
        direction = "rtl" if _has_arabic(text) else "ltr"
    if font is None:
        font = "arabic" if direction == "rtl" else "latin"
    return {"text": text, "font": font, "dir": direction, "nowrap": nowrap}


def normalize_numerals(value: object) -> str:
    return str(value or "").translate(
        str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    )


def _run_tokens(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for run in runs:
        if run.get("type") == "text":
            value = normalize_numerals(run.get("text") or "")
            for part in re.split(r"(\s+)", value):
                if not part:
                    continue
                if part.isspace():
                    tokens.append({"space": True})
                else:
                    tokens.append(_token(part))
        elif run.get("type") == "math":
            plain = run.get("plain") or run.get("latex") or ""
            plain = normalize_numerals(plain)
            # Keep the complete expression as one LTR unit.  Splitting at
            # spaces would let RTL paragraph layout reverse the operands.
            tokens.append(_token(plain, font="latin", direction="ltr", nowrap=True))
    return [token for token in tokens if token and (token.get("space") or token.get("text"))]


def _plain_runs_text(runs: list[dict[str, Any]]) -> str:
    pieces = []
    for run in runs:
        if run.get("type") == "text":
            pieces.append(normalize_numerals(run.get("text") or ""))
        else:
            pieces.append(normalize_numerals(run.get("plain") or run.get("latex") or ""))
    return _clean(" ".join(pieces))


def _paragraph_text(runs: list[dict[str, Any]], *, rtl: bool) -> str:
    """Compose one logical paragraph while isolating LTR mathematical runs."""
    pieces: list[str] = []
    for run in runs:
        if run.get("type") == "text":
            value = normalize_numerals(run.get("text") or "")
        else:
            value = normalize_numerals(run.get("plain") or run.get("latex") or "")
            if rtl and value:
                value = f"\u2066{value}\u2069"
        if value:
            pieces.append(value)
    return _clean(" ".join(pieces))


def _draw_logical_paragraph(
    pdf: _ExamPDF, text: str, x: float, width: float, *,
    size: float = BODY_SIZE, bold: bool = False, align: str | None = None,
) -> float:
    """Render a complete bidi paragraph with one wrapping/alignment context."""
    if not text:
        return 0.0
    pdf.set_font(pdf.body_font, "B" if bold else "", size)
    pdf.set_text_shaping(
        True, direction=TextDirection.RTL if pdf.rtl else TextDirection.LTR,
    )
    paragraph_align = align or ("R" if pdf.rtl else "L")
    measured = pdf.multi_cell(
        width, LINE_HEIGHT, text, align=paragraph_align,
        dry_run=True, output=MethodReturnValue.HEIGHT,
    )
    pdf.set_xy(x, pdf.get_y())
    pdf.multi_cell(
        width, LINE_HEIGHT, text, align=paragraph_align,
        new_x="LEFT", new_y="NEXT",
    )
    return float(measured)


def _token_width(pdf: FPDF, token: dict[str, Any], size: float, bold: bool) -> float:
    if token.get("space"):
        return size * 0.18
    pdf.set_token_font(token, size, bold)
    return float(pdf.get_string_width(token["text"]))


def _layout(pdf: FPDF, tokens: list[dict[str, Any]], width: float, size: float,
            bold: bool) -> list[tuple[list[dict[str, Any]], float]]:
    lines: list[tuple[list[dict[str, Any]], float]] = []
    line: list[dict[str, Any]] = []
    line_w = 0.0
    for token in tokens:
        if token.get("space"):
            if line:
                line.append(token)
                line_w += _token_width(pdf, token, size, bold)
            continue
        w = _token_width(pdf, token, size, bold)
        if line and line_w + w > width and not token.get("nowrap"):
            while line and line[-1].get("space"):
                line.pop()
            lines.append((line, line_w))
            line = []
            line_w = 0.0
        if not line and w > width and token.get("nowrap"):
            # A single unbreakable group wider than the column: keep it whole
            # and let the renderer degrade to a smaller size later.
            line.append(token)
            line_w = w
            continue
        line.append(token)
        line_w += w
    while line and line[-1].get("space"):
        line.pop()
    if line:
        lines.append((line, line_w))
    return lines or [([], 0.0)]


def _draw_line(pdf: FPDF, line: list[dict[str, Any]], line_w: float, x: float,
               width: float, y: float, size: float, bold: bool, align: str) -> None:
    if not line:
        return
    gap = size * 0.18
    offset = 0.0
    if align == "center":
        offset = max(0.0, (width - line_w) / 2)
    elif (align == "end" and not pdf.rtl) or (align == "start" and pdf.rtl):
        offset = max(0.0, width - line_w)
    if pdf.rtl:
        cursor = x + width - offset
        for token in line:
            if token.get("space"):
                cursor -= gap
                continue
            tw = _token_width(pdf, token, size, bold)
            cursor -= tw
            pdf.set_token_font(token, size, bold)
            pdf.set_xy(cursor, y)
            pdf.cell(tw, size * 0.53, token["text"], align="L")
    else:
        cursor = x + offset
        for token in line:
            if token.get("space"):
                cursor += gap
                continue
            tw = _token_width(pdf, token, size, bold)
            pdf.set_token_font(token, size, bold)
            pdf.set_xy(cursor, y)
            pdf.cell(tw, size * 0.53, token["text"], align="L")
            cursor += tw


def draw_flow(
    pdf: _ExamPDF,
    runs: list[dict[str, Any]],
    x: float,
    width: float,
    *,
    size: float = BODY_SIZE,
    bold: bool = False,
    align: str = "start",
    line_height: float = LINE_HEIGHT,
    allow_break: bool = True,
) -> float:
    tokens = _run_tokens(runs)
    if not tokens:
        return 0.0
    lines = _layout(pdf, tokens, width, size, bold)
    start_y = pdf.get_y()
    y = start_y
    for line, line_w in lines:
        if allow_break and y + line_height > pdf.page_break_trigger:
            pdf.add_page()
            y = pdf.get_y()
        _draw_line(pdf, line, line_w, x, width, y, size, bold, align)
        y += line_height
    pdf.set_y(y)
    return y - start_y


def draw_paragraph(
    pdf: _ExamPDF, block: dict[str, Any], x: float, width: float,
    *, size: float = BODY_SIZE, bold: bool = False, align: str = "start",
) -> float:
    runs = block.get("runs") or []
    # Preserve extractable words and natural spacing for prose, but let the
    # token composer handle Latin/numeric-only content inside an RTL page.
    if all(run.get("type") == "text" for run in runs):
        text = _clean(" ".join(run.get("text") or "" for run in runs))
        if text and (not pdf.rtl or _has_arabic(text)):
            pdf.set_font(pdf.body_font, "B" if bold else "", size)
            pdf.set_text_shaping(
                True, direction=TextDirection.RTL if pdf.rtl else TextDirection.LTR
            )
            measured = pdf.multi_cell(
                width, LINE_HEIGHT, text, align="R" if pdf.rtl else "L",
                dry_run=True, output=MethodReturnValue.HEIGHT,
            )
            pdf.set_xy(x, pdf.get_y())
            pdf.multi_cell(
                width, LINE_HEIGHT, text, align="R" if pdf.rtl else "L",
                new_x="LEFT", new_y="NEXT",
            )
            return float(measured)
    return draw_flow(pdf, runs, x, width, size=size, bold=bold, align=align)


def measure_flow(pdf: _ExamPDF, runs: list[dict[str, Any]], width: float,
                 *, size: float = BODY_SIZE, bold: bool = False) -> float:
    tokens = _run_tokens(runs)
    if not tokens:
        return 0.0
    return len(_layout(pdf, tokens, width, size, bold)) * LINE_HEIGHT


def draw_display_math(pdf: _ExamPDF, block: dict[str, Any], x: float, width: float) -> float:
    latex = block.get("latex") or ""
    rendered = None if _has_arabic(latex) else render_math(latex, font_size=13, dpi=260)
    if rendered:
        natural_w = rendered.width_px / 260 * 25.4
        natural_h = rendered.height_px / 260 * 25.4
        draw_w = min(width, max(8.0, natural_w))
        draw_h = max(4.0, natural_h * (draw_w / natural_w if natural_w else 1.0))
        pdf.ensure_space(draw_h + 3)
        stream = io.BytesIO(rendered.png)
        stream.name = "display-math.png"
        pdf.image(stream, x=x + (width - draw_w) / 2, y=pdf.get_y(), w=draw_w, h=draw_h)
        pdf.set_y(pdf.get_y() + draw_h + 2.5)
        return draw_h + 2.5
    plain = block.get("plain") or block.get("latex") or ""
    text = _clean(normalize_numerals(plain))
    if text:
        if pdf.rtl:
            # Keep Latin variables, units and operators in logical LTR order,
            # while the complete display remains one centered RTL paragraph.
            text = re.sub(
                r"([A-Za-z0-9]+(?:\s*[+\-*/=<>≤≥.,:]\s*[A-Za-z0-9%]+)*)",
                lambda match: f"\u2066{match.group(1)}\u2069",
                text,
            )
        return _draw_logical_paragraph(
            pdf, text, x, width, size=BODY_SIZE, align="C",
        )
    return 0.0


def _runs_text(runs: list[dict[str, Any]]) -> str:
    return _clean(" ".join(
        normalize_numerals(run.get("text") or run.get("plain") or "")
        for run in runs
    ))


def _table_column_widths(pdf: _ExamPDF, table: dict[str, Any], width: float) -> list[float]:
    headers = table.get("headers") or []
    if not headers:
        return []
    count = len(headers)
    if count == 1:
        return [width]
    pdf.set_font(LATIN_FONT, "", SMALL_SIZE)
    weights = []
    for index in range(count):
        values = [_runs_text(headers[index])]
        for row in table.get("rows") or []:
            if index < len(row):
                values.append(_runs_text(row[index]))
        longest = max((pdf.get_string_width(value) for value in values), default=8.0)
        weights.append(max(12.0, min(80.0, longest + 4)))
    total = sum(weights)
    raw = [width * weight / total for weight in weights]
    minimum = min(18.0, width / count)
    raw = [max(minimum, value) for value in raw]
    factor = width / sum(raw)
    return [value * factor for value in raw]


def draw_table(
    pdf: _ExamPDF, table: dict[str, Any], x: float, width: float,
    *, font_size: float = SMALL_SIZE, repeat_headers: bool = True,
) -> float:
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    if not headers:
        return 0.0
    columns = _table_column_widths(pdf, table, width)
    pad_x = 1.8
    pad_y = 1.7
    line_h = 4.3
    start_y = pdf.get_y()
    caption = _clean(table.get("caption") or "")
    if caption:
        pdf.set_font(pdf.body_font, "B", 9)
        pdf.set_text_shaping(True, direction=TextDirection.RTL if pdf.rtl else TextDirection.LTR)
        pdf.set_xy(x, pdf.get_y())
        pdf.cell(width, 5.2, caption, align="C", new_x="LEFT", new_y="NEXT")
        pdf.set_y(pdf.get_y() + 1)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.35)

    def row_height(cells: list[list[dict[str, Any]]], bold: bool = False) -> float:
        heights = []
        for index, cell in enumerate(cells):
            if index >= len(columns):
                break
            heights.append(measure_flow(pdf, cell, columns[index] - 2 * pad_x,
                                        size=font_size, bold=bold))
        return max([line_h] + heights) + 2 * pad_y

    def cell_text(cell: list[dict[str, Any]], cx: float, cy: float, cw: float,
                  rh: float, align: str, bold: bool) -> None:
        pdf.set_xy(cx + pad_x, cy + pad_y)
        draw_flow(pdf, cell, cx + pad_x, cw - 2 * pad_x, size=font_size,
                  bold=bold, align=align, line_height=line_h, allow_break=False)

    y = pdf.get_y()
    header_h = row_height(headers, bold=True)
    if y + header_h > pdf.page_break_trigger:
        pdf.add_page()
        y = pdf.get_y()
    # Header row: white fill, bold text, thin rules.
    cx = x
    for index, cell in enumerate(headers[:len(columns)]):
        cw = columns[index]
        pdf.rect(cx, y, cw, header_h)
        cell_text(cell, cx, y, cw, header_h,
                  "center" if table.get("align", [])[index:index + 1] == ["center"] else
                  ("end" if (table.get("align") or [])[index:index + 1] == ["end"] else "start"),
                  True)
        cx += cw
    y += header_h
    for row_index, row in enumerate(rows):
        rh = row_height(row)
        if y + rh > pdf.page_break_trigger:
            pdf.add_page()
            y = pdf.get_y()
            if repeat_headers:
                cx = x
                for index, cell in enumerate(headers[:len(columns)]):
                    cw = columns[index]
                    pdf.rect(cx, y, cw, header_h)
                    cell_text(cell, cx, y, cw, header_h, "start", True)
                    cx += cw
                y += header_h
        cx = x
        for index, cell in enumerate(row[:len(columns)]):
            cw = columns[index]
            pdf.rect(cx, y, cw, rh)
            align = (table.get("align") or [])[index] if index < len(table.get("align") or []) else "start"
            if table.get("compact"):
                align = "center"
            cell_text(cell, cx, y, cw, rh, align, False)
            cx += cw
        y += rh
    pdf.set_y(y + 1.5)
    return y - start_y + 1.5


def draw_response_lines(pdf: _ExamPDF, count: int, x: float, width: float) -> float:
    if count <= 0:
        return 0.0
    line_h = 6.5
    # Keep at least two lines together; do not force a whole page of blank
    # lines when only a short answer area remains.
    if pdf.get_y() + min(2, count) * line_h > pdf.page_break_trigger:
        pdf.add_page()
    start = pdf.get_y()
    for _ in range(count):
        if pdf.get_y() + line_h > pdf.page_break_trigger:
            pdf.add_page()
        pdf.set_draw_color(*RULE)
        pdf.set_line_width(0.4)
        pdf.line(x, pdf.get_y(), x + width, pdf.get_y())
        pdf.set_y(pdf.get_y() + line_h)
    return pdf.get_y() - start


def _draw_number(pdf: _ExamPDF, number: int, x: float, y: float, width: float,
                 align: str) -> None:
    pdf.set_font(LATIN_FONT, "B", 10.5)
    pdf.set_text_shaping(True, direction=TextDirection.LTR)
    pdf.set_xy(x, y)
    pdf.cell(width, LINE_HEIGHT, f"{number}.", align=align)


def _draw_question_text(
    pdf: _ExamPDF,
    question: dict[str, Any],
    x: float,
    width: float,
    *,
    include_number: bool = True,
) -> float:
    blocks = question.get("blocks") or []
    start = pdf.get_y()
    content_x = x
    content_w = width
    first = True
    for block in blocks:
        if block.get("type") == "table":
            if first:
                pdf.set_y(start + LINE_HEIGHT)
            draw_table(pdf, block, content_x, content_w)
            first = False
            continue
        pdf.set_x(content_x)
        if first:
            pdf.set_y(start)
        if block.get("type") == "paragraph":
            text = _paragraph_text(block.get("runs") or [], rtl=pdf.rtl)
            if first and include_number:
                marker = f"{int(question['number'])}.\u00a0"
                text = marker + text
            _draw_logical_paragraph(pdf, text, content_x, content_w, bold=True)
        elif block.get("type") == "math" and block.get("display"):
            draw_display_math(pdf, block, content_x, content_w)
        first = False
    used = pdf.get_y() - start
    return max(used, LINE_HEIGHT)


def _option_blocks_height(pdf: _ExamPDF, option: dict[str, Any], width: float) -> float:
    height = 0.0
    for block in option.get("blocks") or []:
        if block.get("type") == "table":
            height += 10.0 + len(block.get("rows") or []) * 7.5
        elif block.get("type") == "paragraph":
            height += measure_flow(pdf, block.get("runs") or [], width)
        elif block.get("type") == "math":
            height += LINE_HEIGHT
    return max(height, 5.5)


def _draw_option(pdf: _ExamPDF, option: dict[str, Any], x: float, y: float,
                 width: float) -> float:
    pdf.set_y(y)
    needed = _option_blocks_height(pdf, option, width)
    if pdf.get_y() + needed + 2 > pdf.page_break_trigger:
        pdf.add_page()
        y = pdf.get_y()
    label = f"{option.get('label', 'A')}."
    content_x = x
    content_w = width
    pdf.set_y(y)
    used = 0.0
    for block in option.get("blocks") or []:
        if block.get("type") == "paragraph":
            text = _paragraph_text(block.get("runs") or [], rtl=pdf.rtl)
            marker = f"{label}\u00a0"
            used = max(used, _draw_logical_paragraph(
                pdf, marker + text, content_x, content_w, size=10.2,
            ))
        elif block.get("type") == "table":
            _draw_logical_paragraph(
                pdf, label, content_x, content_w,
                size=10.2, bold=True,
            )
            used = max(used, draw_table(pdf, block, content_x, content_w, font_size=8.2))
        elif block.get("type") == "math":
            used = max(used, draw_flow(
                pdf, [block], content_x, content_w, size=10.2,
                align="R" if pdf.rtl else "L",
            ))
    return max(used, LINE_HEIGHT)


def _use_two_option_columns(pdf: _ExamPDF, question: dict[str, Any], width: float) -> bool:
    options = question.get("options") or []
    if not options:
        return False
    tables = [
        block for option in options for block in (option.get("blocks") or [])
        if block.get("type") == "table"
    ]
    if tables:
        # Compact numeric option tables remain readable in two columns when
        # their shape and content fit the measured half-page width.  Larger or
        # prose-heavy tables stack automatically.
        return all(
            block.get("compact")
            and len(block.get("headers") or []) <= 3
            and len(block.get("rows") or []) <= 5
            and max(
                (len(_runs_text(cell)) for row in (
                    [block.get("headers") or []] + list(block.get("rows") or [])
                ) for cell in row),
                default=0,
            ) <= 12
            for block in tables
        )
    col_w = (width - 8) / 2
    if tables:
        return True
    # Measure actual wrapped height, not character counts.  Long or multiline
    # options automatically fall back to one column.
    return all(
        _option_blocks_height(pdf, option, col_w) <= LINE_HEIGHT * 2.1
        for option in options
    )


def draw_options(pdf: _ExamPDF, question: dict[str, Any], x: float, width: float) -> float:
    options = question.get("options") or []
    if not options:
        return 0.0
    two_columns = _use_two_option_columns(pdf, question, width)
    if two_columns:
        col_w = (width - 8) / 2
        start = pdf.get_y()
        heights = []
        for index, option in enumerate(options):
            col = index % 2
            visual = 1 - col if pdf.rtl else col
            ox = x + visual * (col_w + 8)
            oy = start + (index // 2) * 0.0
            # Draw each row only after knowing both heights; a simple row
            # pass keeps A/B and C/D aligned.
            heights.append(_option_blocks_height(pdf, option, col_w))
        row_heights = [max(heights[i:i + 2]) + 2 for i in range(0, len(heights), 2)]
        for index, option in enumerate(options):
            row = index // 2
            col = index % 2
            visual = 1 - col if pdf.rtl else col
            ox = x + visual * (col_w + 8)
            oy = start + sum(row_heights[:row])
            _draw_option(pdf, option, ox, oy, col_w)
        pdf.set_y(start + sum(row_heights))
        return sum(row_heights)
    start = pdf.get_y()
    initial = start
    for option in options:
        pdf.set_y(start)
        _draw_option(pdf, option, x, start, width)
        start = pdf.get_y() + 1.0
    pdf.set_y(start)
    return start - initial


def draw_true_false(pdf: _ExamPDF, question: dict[str, Any], x: float, width: float,
                    choices: str) -> float:
    blocks = question.get("blocks") or []
    if not blocks or any(block.get("type") != "paragraph" for block in blocks):
        height = _draw_question_text(pdf, question, x, width)
        pdf.set_font(pdf.body_font, "", 9.8)
        pdf.set_text_shaping(True, direction=TextDirection.RTL if pdf.rtl else TextDirection.LTR)
        pdf.set_xy(x, pdf.get_y())
        pdf.multi_cell(width, LINE_HEIGHT, choices, align="R" if pdf.rtl else "L",
                       new_x="LEFT", new_y="NEXT")
        return height + LINE_HEIGHT
    runs: list[dict[str, Any]] = []
    for block in blocks:
        runs.extend(block.get("runs") or [])
    statement = _paragraph_text(runs, rtl=pdf.rtl)
    marker = f"{int(question.get('number') or 1)}.\u00a0"
    # Keep both labels and checkbox pairs as one unbreakable group.  Avoid
    # directional controls around Arabic here: HarfBuzz already has the
    # paragraph base direction and some PDF extractors drop isolated content.
    group = choices.replace(" ", "\u00a0")
    return _draw_logical_paragraph(
        pdf, f"{marker}{statement} {group}", x, width, bold=True,
    )


def _options_height(pdf: _ExamPDF, question: dict[str, Any], width: float) -> float:
    options = question.get("options") or []
    if not options:
        return 0.0
    two = _use_two_option_columns(pdf, question, width)
    if two:
        col_w = (width - 8) / 2
        heights = [_option_blocks_height(pdf, option, col_w) for option in options]
        return sum(max(heights[i:i + 2]) + 2 for i in range(0, len(heights), 2))
    return sum(_option_blocks_height(pdf, option, width) + 1 for option in options)


def _estimate_question_height(pdf: _ExamPDF, question: dict[str, Any], width: float) -> float:
    blocks = question.get("blocks") or []
    table = next((block for block in blocks if block.get("type") == "table"), None)
    non_table = [block for block in blocks if block.get("type") != "table"]
    height = 0.0
    for block in non_table:
        if block.get("type") == "paragraph":
            height += measure_flow(pdf, block.get("runs") or [], width, bold=True)
        elif block.get("type") == "math" and block.get("display"):
            height += 20.0
    height += _options_height(pdf, question, width)
    height += int(question.get("response_lines") or 0) * 6.5
    if table:
        table_w = min(64.0, width * 0.38)
        question_w = width - table_w - 8.0
        table_height = 12.0 + len(table.get("rows") or []) * 8.0
        if question_w >= 90 and table_height + 20 <= (pdf.page_break_trigger - pdf.get_y()):
            height = max(height, table_height + 12)
        else:
            height += table_height + 8
    return max(height, LINE_HEIGHT)


def _draw_question(pdf: _ExamPDF, question: dict[str, Any], x: float, width: float) -> None:
    estimated = _estimate_question_height(pdf, question, width)
    if pdf.get_y() > PAGE_MARGIN_MM + 4 and pdf.get_y() + estimated > pdf.page_break_trigger:
        pdf.add_page()
    qtype = question.get("qtype")
    if qtype == "definition":
        _draw_question_text(pdf, question, x, width)
        draw_response_lines(pdf, 1, x, width)
        pdf.ln(2.5)
        return
    if qtype == "true_false":
        pdf.ensure_space(14)
        draw_true_false(pdf, question, x, width, question.get("true_false_choices") or "")
        pdf.ln(2.5)
        return

    blocks = question.get("blocks") or []
    table = next((block for block in blocks if block.get("type") == "table"), None)
    non_table = [block for block in blocks if block.get("type") != "table"]
    if table:
        table_w = min(64.0, width * 0.38)
        question_w = width - table_w - 8.0
        table_height = 12.0 + len(table.get("rows") or []) * 8.0
        side_by_side = question_w >= 90.0 and table_height + 20 <= (
            pdf.page_break_trigger - pdf.get_y()
        )
        if side_by_side:
            start = pdf.get_y()
            qx = x + table_w + 8.0 if pdf.rtl else x
            tx = x if pdf.rtl else x + question_w + 8.0
            q_question = dict(question)
            q_question["blocks"] = non_table
            used = _draw_question_text(pdf, q_question, qx, question_w)
            options_used = draw_options(pdf, question, qx, question_w)
            q_end = pdf.get_y()
            pdf.set_y(start)
            draw_table(pdf, table, tx, table_w)
            table_end = pdf.get_y()
            pdf.set_y(max(q_end, table_end) + 3)
            return
        # Stacked layout: question and options first, then the table.
        q_question = dict(question)
        q_question["blocks"] = non_table
        _draw_question_text(pdf, q_question, x, width)
        draw_options(pdf, question, x, width)
        draw_table(pdf, table, x, width)
        return

    _draw_question_text(pdf, question, x, width)
    draw_options(pdf, question, x, width)
    lines = int(question.get("response_lines") or 0)
    if lines:
        draw_response_lines(pdf, lines, x, width)


def _draw_header(pdf: _ExamPDF, document: dict[str, Any]) -> None:
    meta = document["metadata"]
    pdf.set_y(PAGE_MARGIN_MM)
    title = meta.get("title") or ("امتحان" if pdf.rtl else "Examination")
    pdf.set_font(pdf.body_font, "B", 20)
    pdf.set_text_shaping(True, direction=TextDirection.RTL if pdf.rtl else TextDirection.LTR)
    pdf.cell(0, 9, title, align="C", new_x="LEFT", new_y="NEXT")
    pdf.set_font(pdf.body_font, "", 9.5)
    pdf.cell(0, 5, meta.get("model_label") or "", align="C", new_x="LEFT", new_y="NEXT")
    pdf.ln(3)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.7)
    pdf.line(PAGE_MARGIN_MM, pdf.get_y(), PAGE_MARGIN_MM + CONTENT_WIDTH_MM, pdf.get_y())
    pdf.ln(3)
    fields = [
        (meta.get("duration_label"), meta.get("duration")),
        (meta.get("teacher_label"), meta.get("teacher_name")),
        (meta.get("date_label"), meta.get("exam_date")),
    ]
    col_w = CONTENT_WIDTH_MM / 3
    y = pdf.get_y()
    for index, (label, value) in enumerate(fields):
        visual = 2 - index if pdf.rtl else index
        x = PAGE_MARGIN_MM + visual * col_w
        pdf.set_xy(x, y)
        pdf.set_font(pdf.body_font, "B", 9)
        pdf.cell(20, 5, f"{label}:", align="R" if pdf.rtl else "L")
        pdf.set_font(pdf.body_font, "", 9)
        pdf.cell(col_w - 22, 5, _clean(value), align="R" if pdf.rtl else "L")
    pdf.set_y(y + 7)
    pdf.set_line_width(0.4)
    pdf.line(PAGE_MARGIN_MM, pdf.get_y(), PAGE_MARGIN_MM + CONTENT_WIDTH_MM, pdf.get_y())
    pdf.ln(4)
    pdf.set_font(pdf.body_font, "", 9.3)
    class_value = _clean(meta.get("class_name"))
    pdf.cell(100, 6, f"{meta.get('student_name_label')}: ______________________________",
             align="R" if pdf.rtl else "L")
    pdf.cell(80, 6, f"{meta.get('class_label')}: {class_value or '________________'}",
             align="R" if pdf.rtl else "L", new_x="LEFT", new_y="NEXT")
    pdf.ln(3)


def _draw_section_heading(pdf: _ExamPDF, heading: str) -> None:
    pdf.ensure_space(16)
    pdf.ln(2)
    pdf.set_font(pdf.body_font, "B", 11.5)
    pdf.set_text_shaping(True, direction=TextDirection.RTL if pdf.rtl else TextDirection.LTR)
    pdf.set_xy(PAGE_MARGIN_MM, pdf.get_y())
    pdf.cell(CONTENT_WIDTH_MM, 6.5, heading, align="R" if pdf.rtl else "L",
             new_x="LEFT", new_y="NEXT")
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.55)
    pdf.line(PAGE_MARGIN_MM, pdf.get_y(), PAGE_MARGIN_MM + CONTENT_WIDTH_MM, pdf.get_y())
    pdf.ln(4)


def _draw_student(pdf: _ExamPDF, document: dict[str, Any]) -> None:
    _draw_header(pdf, document)
    for section in document.get("sections") or []:
        first_item = (section.get("items") or [None])[0]
        if first_item:
            # Keep the heading with the first question; never leave a heading
            # stranded at the bottom of a page.
            first_height = _estimate_question_height(
                pdf, first_item, CONTENT_WIDTH_MM
            )
            # Reserve the real first-question footprint plus the compact
            # heading gap.  The former +8 mm guard forced a short word problem
            # onto an otherwise empty second page even when it fit safely.
            pdf.ensure_space(max(24.0, min(55.0, first_height) + 4))
        _draw_section_heading(pdf, section.get("heading") or "")
        if section.get("word_bank"):
            pdf.set_font(pdf.body_font, "B", 9.2)
            bank = "  |  ".join(section["word_bank"])
            pdf.set_text_shaping(True, direction=TextDirection.RTL if pdf.rtl else TextDirection.LTR)
            pdf.set_xy(PAGE_MARGIN_MM, pdf.get_y())
            pdf.multi_cell(CONTENT_WIDTH_MM, 5.4, bank, align="R" if pdf.rtl else "L",
                           new_x="LEFT", new_y="NEXT")
            pdf.ln(2)
        for question in section.get("items") or []:
            _draw_question(pdf, question, PAGE_MARGIN_MM, CONTENT_WIDTH_MM)
            pdf.ln(1.5)


def _draw_answers(pdf: _ExamPDF, document: dict[str, Any]) -> None:
    pdf.set_y(PAGE_MARGIN_MM)
    pdf.set_font(pdf.body_font, "B", 16)
    pdf.set_text_shaping(True, direction=TextDirection.RTL if pdf.rtl else TextDirection.LTR)
    pdf.cell(0, 9, document["metadata"].get("answer_key_title") or "", align="C",
             new_x="LEFT", new_y="NEXT")
    pdf.ln(3)
    for section in document.get("sections") or []:
        _draw_section_heading(pdf, section.get("heading") or "")
        for question in section.get("items") or []:
            pdf.ensure_space(12)
            number = int(question.get("number") or 1)
            first_paragraph = True
            for block in question.get("answer_blocks") or []:
                if block.get("type") == "paragraph":
                    text = _paragraph_text(block.get("runs") or [], rtl=pdf.rtl)
                    if first_paragraph:
                        text = f"{number}.\u00a0{text}"
                    _draw_logical_paragraph(
                        pdf, text, PAGE_MARGIN_MM, CONTENT_WIDTH_MM, size=9.5,
                    )
                    first_paragraph = False
                elif block.get("type") == "math" and block.get("display"):
                    draw_display_math(pdf, block, PAGE_MARGIN_MM + 10,
                                      CONTENT_WIDTH_MM - 10)
                elif block.get("type") == "table":
                    draw_table(pdf, block, PAGE_MARGIN_MM + 10,
                               CONTENT_WIDTH_MM - 10, font_size=8.5)
            pdf.ln(1.5)


def _render(exam: dict[str, Any], metadata: dict[str, Any], *, kind: str) -> bytes:
    document = build_document_ir(exam, metadata, kind=kind)
    language = document.get("language") or "en"
    pdf = _ExamPDF(language)
    pdf.alias_nb_pages()
    pdf.set_title(document["metadata"].get("title") or "Exam")
    pdf.add_page()
    if kind == "answers":
        _draw_answers(pdf, document)
    else:
        _draw_student(pdf, document)
    _draw_page_footers(pdf)
    return bytes(pdf.output())


def _draw_page_footers(pdf: _ExamPDF) -> None:
    """Paint stable localized page totals after the last page is known."""
    total = len(pdf.pages)
    current = pdf.page
    auto_break = pdf.auto_page_break
    bottom_margin = pdf.b_margin
    pdf.set_auto_page_break(False)
    for number in range(1, total + 1):
        pdf.page = number
        pdf.set_y(-11)
        pdf.set_font(pdf.body_font, "", 8)
        pdf.set_text_color(*INK)
        text = (
            f"صفحة {number} من {total}"
            if pdf.rtl else f"Page {number} of {total}"
        )
        pdf.set_text_shaping(
            True, direction=TextDirection.RTL if pdf.rtl else TextDirection.LTR,
        )
        pdf.cell(0, 4, text, align="C")
    pdf.page = current
    pdf.set_auto_page_break(auto_break, margin=bottom_margin)


def render_exam_pdf(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    return _render(exam, metadata, kind="student")


def render_answers_pdf(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    return _render(exam, metadata, kind="answers")
