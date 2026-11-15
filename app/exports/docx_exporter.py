"""Editable, black-and-white DOCX export from the shared exam IR.

The document uses native Word tables for every content table, native paragraphs
and runs for prose, and explicit RTL/bidi properties.  Inline math is kept as
editable Unicode text; complex display math is embedded as a high-resolution
image with alt text.  No raw LaTeX or Markdown is written into the document.
"""
from __future__ import annotations

import io
import re
from typing import Any

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

from app.exports.document_ir import build_document_ir
from app.exports.math_renderer import latex_to_plain_text, render_math

BLACK = RGBColor(0, 0, 0)
FONT_NAME = "Arial"
BODY_SIZE = 10.5
SMALL_SIZE = 8.2
PAGE_MARGIN_MM = 15.0
CONTENT_WIDTH_MM = 180.0
CONTENT_WIDTH_IN = CONTENT_WIDTH_MM / 25.4
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_NUMERAL_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _answer_text(item: dict[str, Any], language: str) -> str:
    """Compatibility helper used by export contract tests and extensions."""
    if item.get("qtype") == "true_false" or "answer" in item:
        answer = item.get("answer")
        if answer is True:
            return "صح" if language == "ar" else "True"
        if answer is False:
            return "غلط" if language == "ar" else "False"
    return str(item.get("correct_answer") or item.get("final_answer") or "")


def _rtl(value: object) -> bool:
    return bool(_ARABIC_RE.search(str(value or "")))


def _clean(value: object) -> str:
    return re.sub(r"[ \t]+", " ", str(value or "")).strip()


def _run_text(runs: list[dict[str, Any]]) -> str:
    return " ".join(
        str(run.get("text") or run.get("plain") or "").translate(_NUMERAL_TRANS)
        for run in runs
    ).strip()


def _set_run_font(run, *, size: float = BODY_SIZE, bold: bool = False,
                  color: RGBColor = BLACK, rtl: bool | None = None) -> None:
    run.font.name = FONT_NAME
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        r_fonts.set(qn(attr), FONT_NAME)
    if r_pr.find(qn("w:szCs")) is None:
        sz_cs = OxmlElement("w:szCs")
        r_pr.append(sz_cs)
    r_pr.find(qn("w:szCs")).set(qn("w:val"), str(int(size * 2)))
    if bold:
        b_cs = OxmlElement("w:bCs")
        r_pr.append(b_cs)
    if rtl is True:
        if r_pr.find(qn("w:rtl")) is None:
            r_pr.append(OxmlElement("w:rtl"))
        if r_pr.find(qn("w:lang")) is None:
            r_pr.append(OxmlElement("w:lang"))
        r_pr.find(qn("w:lang")).set(qn("w:bidi"), "ar-SA")
    elif rtl is False:
        if r_pr.find(qn("w:rtl")) is None:
            rtl_node = OxmlElement("w:rtl")
            rtl_node.set(qn("w:val"), "0")
            r_pr.append(rtl_node)
        if r_pr.find(qn("w:lang")) is None:
            r_pr.append(OxmlElement("w:lang"))
        r_pr.find(qn("w:lang")).set(qn("w:val"), "en-US")


def _set_paragraph_bidi(p, rtl: bool) -> None:
    p_pr = p._p.get_or_add_pPr()
    if rtl:
        if p_pr.find(qn("w:bidi")) is None:
            p_pr.append(OxmlElement("w:bidi"))
        # In a bidi paragraph Word mirrors physical left/right alignment.  The
        # OOXML logical `start` value is the stable right reading edge in both
        # Word and LibreOffice; w:jc="right" is physically rendered at the left
        # edge by Word when w:bidi is present.
        jc = p_pr.find(qn("w:jc"))
        if jc is None:
            jc = OxmlElement("w:jc")
            p_pr.append(jc)
        jc.set(qn("w:val"), "start")
    else:
        bidi = p_pr.find(qn("w:bidi"))
        if bidi is not None:
            bidi.set(qn("w:val"), "0")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def _keep_with_next(p) -> None:
    p.paragraph_format.keep_with_next = True


def _add_run(p, text: str, *, size: float = BODY_SIZE, bold: bool = False,
             rtl: bool | None = None, color: RGBColor = BLACK):
    run = p.add_run(text)
    _set_run_font(run, size=size, bold=bold, color=color, rtl=rtl)
    return run


def _add_math_image(p, latex: str, *, width_in: float = 5.0) -> bool:
    rendered = render_math(latex, font_size=13, dpi=300)
    if not rendered:
        return False
    stream = io.BytesIO(rendered.png)
    stream.name = "display-math.png"
    natural_w = min(width_in, max(1.0, rendered.width_px / 300 * 25.4 / 25.4))
    run = p.add_run()
    run.add_picture(stream, width=Mm(natural_w * 25.4))
    # Alt text is useful to screen readers and to users who inspect the doc.
    try:
        inline = run._r.find(qn("w:drawing")).find(qn("wp:inline"))
        doc_pr = inline.find(qn("wp:docPr"))
        # Alt text must remain readable and must not expose raw TeX commands.
        doc_pr.set("descr", (latex_to_plain_text(latex) or "Mathematical equation")[:500])
        doc_pr.set("title", "Equation")
    except Exception:
        pass
    return True


def _add_inline_math_image(p, latex: str) -> bool:
    """Insert complex inline math at its natural size and baseline context."""
    rendered = render_math(latex, font_size=12, dpi=300)
    if not rendered:
        return False
    stream = io.BytesIO(rendered.png)
    stream.name = "inline-math.png"
    natural_in = max(0.18, min(2.2, rendered.width_px / 300))
    run = p.add_run()
    run.add_picture(stream, width=Mm(natural_in * 25.4))
    try:
        inline = run._r.find(qn("w:drawing")).find(qn("wp:inline"))
        doc_pr = inline.find(qn("wp:docPr"))
        doc_pr.set("descr", (latex_to_plain_text(latex) or "Mathematical expression")[:500])
        doc_pr.set("title", "Inline equation")
    except Exception:
        pass
    return True


def _add_native_math(p, latex: str) -> bool:
    """Add editable native Word math for expressions that have a safe linear form.

    Complex TeX remains an image fallback because pretending to convert syntax
    that we do not fully understand is more damaging than a faithful image.
    """
    if not latex or any(token in latex for token in (r"\begin", r"\matrix", r"\cases")):
        return False
    plain = latex_to_plain_text(latex).strip()
    if not plain or "\\" in plain or len(plain) > 240:
        return False
    math = OxmlElement("m:oMath")
    run = OxmlElement("m:r")
    text = OxmlElement("m:t")
    text.text = plain.translate(_NUMERAL_TRANS)
    run.append(text)
    math.append(run)
    p._p.append(math)
    return True


def _add_display_math(p, latex: str, *, width_in: float = 5.0) -> bool:
    # Word's linear OMML import is not interoperable enough for the complex
    # expressions in these exams (Word can stall while repaginating them).
    # Preserve the faithful, labelled image fallback until a full TeX-to-OMML
    # converter is available.
    return _add_math_image(p, latex, width_in=width_in)


def _add_runs(p, runs: list[dict[str, Any]], *, size: float = BODY_SIZE,
              bold: bool = False, rtl: bool | None = None) -> None:
    for run in runs:
        if run.get("type") == "text":
            value = str(run.get("text") or "").translate(_NUMERAL_TRANS)
            if value:
                _add_run(p, value, size=size, bold=bold,
                         rtl=_rtl(value))
        elif run.get("type") == "math":
            if run.get("display"):
                continue
            latex = str(run.get("latex") or "")
            if "\\" in latex and _add_inline_math_image(p, latex):
                continue
            plain = str(run.get("plain") or run.get("latex") or "")
            if not plain or "\\" in plain:
                plain = re.sub(r"\\[A-Za-z]+", " ", plain).replace("{", "").replace("}", "")
            plain = plain.translate(_NUMERAL_TRANS)
            if plain.strip():
                _add_run(p, plain, size=size, bold=bold, rtl=False)


def _add_blocks(doc_or_cell, blocks: list[dict[str, Any]], *, rtl: bool,
                size: float = BODY_SIZE, bold: bool = False,
                keep: bool = False) -> None:
    for block in blocks:
        kind = block.get("type")
        if kind == "paragraph":
            p = doc_or_cell.add_paragraph()
            _set_paragraph_bidi(p, rtl)
            p.paragraph_format.space_after = Pt(2)
            if keep:
                _keep_with_next(p)
            _add_runs(p, block.get("runs") or [], size=size, bold=bold, rtl=rtl)
        elif kind == "math" and block.get("display"):
            p = doc_or_cell.add_paragraph()
            _set_paragraph_bidi(p, rtl)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            latex = str(block.get("latex") or "")
            if not _add_display_math(p, latex, width_in=5.6):
                _add_run(p, str(block.get("plain") or latex),
                         size=size, rtl=False)
        elif kind == "table":
            _add_table(doc_or_cell, block, rtl=rtl, width_mm=CONTENT_WIDTH_MM)
        elif kind == "response_area":
            _add_response_lines(doc_or_cell, int(block.get("lines") or 0), rtl=rtl)


def _add_response_lines(container, count: int, *, rtl: bool) -> None:
    for _ in range(count):
        p = container.add_paragraph()
        _set_paragraph_bidi(p, rtl)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.25
        _add_run(p, "_" * 92, size=9, rtl=rtl)
        p.paragraph_format.keep_together = True


def _normalize_tbl_pr(table) -> None:
    """Put OOXML table-property children in schema order.

    Word and LibreOffice can ignore otherwise-valid border properties when
    they appear after layout/look elements.
    """
    order = [
        "tblStyle", "tblpPr", "tblOverlap", "bidiVisual",
        "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc",
        "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout",
        "tblCellMar", "tblLook", "tblCaption", "tblDescription",
    ]
    tbl_pr = table._tbl.tblPr
    children = list(tbl_pr)
    by_name = {}
    for child in children:
        tag = child.tag.split("}")[-1]
        by_name.setdefault(tag, []).append(child)
    for child in children:
        tbl_pr.remove(child)
    for name in order:
        for child in by_name.pop(name, []):
            tbl_pr.append(child)
    for remaining in by_name.values():
        for child in remaining:
            tbl_pr.append(child)


def _normalize_tc_pr(cell) -> None:
    order = ["tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
             "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign",
             "hideMark"]
    tc_pr = cell._tc.get_or_add_tcPr()
    children = list(tc_pr)
    by_name = {}
    for child in children:
        tag = child.tag.split("}")[-1]
        by_name.setdefault(tag, []).append(child)
    for child in children:
        tc_pr.remove(child)
    for name in order:
        for child in by_name.pop(name, []):
            tc_pr.append(child)
    for remaining in by_name.values():
        for child in remaining:
            tc_pr.append(child)


def _set_table_borders(table, *, size: int = 4) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), str(size))
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), "000000")
    _normalize_tbl_pr(table)


def _remove_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    style = tbl_pr.find(qn("w:tblStyle"))
    if style is not None:
        tbl_pr.remove(style)
    look = tbl_pr.find(qn("w:tblLook"))
    if look is not None:
        tbl_pr.remove(look)
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "8")
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), "FFFFFF")
    # LibreOffice sometimes ignores val="nil" and paints default grid lines.
    # White borders are visually borderless on the white exam page and keep the
    # layout table structural only.
    for row in table.rows:
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_borders = tc_pr.find(qn("w:tcBorders"))
            if tc_borders is None:
                tc_borders = OxmlElement("w:tcBorders")
                tc_pr.append(tc_borders)
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                node = tc_borders.find(qn(f"w:{edge}"))
                if node is None:
                    node = OxmlElement(f"w:{edge}")
                    tc_borders.append(node)
                node.set(qn("w:val"), "single")
                node.set(qn("w:sz"), "8")
                node.set(qn("w:space"), "0")
                node.set(qn("w:color"), "FFFFFF")
            _normalize_tc_pr(cell)
    _normalize_tbl_pr(table)


def _set_table_width(table, widths_mm: list[float]) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(round(sum(widths_mm) / 25.4 * 1440)))
    tbl_w.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width_mm in widths_mm:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(round(width_mm / 25.4 * 1440)))
        grid.append(grid_col)
    for row in table.rows:
        _set_row_cant_split(row)
        for cell, width_mm in zip(row.cells, widths_mm):
            cell.width = Mm(width_mm)
            _set_cell_margins(cell)
    _normalize_tbl_pr(table)


def _set_cell_margins(cell, top: int = 60, start: int = 80,
                      bottom: int = 60, end: int = 80) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
    _normalize_tc_pr(cell)


def _set_row_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def _set_table_bidi(table) -> None:
    tbl_pr = table._tbl.tblPr
    if tbl_pr.find(qn("w:bidiVisual")) is None:
        node = OxmlElement("w:bidiVisual")
        tbl_pr.append(node)
    _normalize_tbl_pr(table)


def _set_header_repeat(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:tblHeader")) is None:
        node = OxmlElement("w:tblHeader")
        tr_pr.append(node)


def _add_table(container, table_block: dict[str, Any], *, rtl: bool,
               width_mm: float = CONTENT_WIDTH_MM) -> None:
    headers = table_block.get("headers") or []
    rows = table_block.get("rows") or []
    if not headers:
        return
    if hasattr(container, "_tc"):
        for paragraph in list(container.paragraphs):
            if not paragraph.text.strip():
                p_pr = paragraph._p.pPr
                if p_pr is None or p_pr.find(qn("w:pBdr")) is None:
                    paragraph._p.getparent().remove(paragraph._p)
    count = len(headers)
    widths = [width_mm / count] * count
    table = container.add_table(rows=1, cols=count)
    _set_table_width(table, widths)
    _set_table_borders(table, size=4)
    if rtl:
        _set_table_bidi(table)
    header = table.rows[0]
    _set_header_repeat(header)
    for index, cell in enumerate(header.cells):
        cell.text = ""
        p = cell.paragraphs[0]
        _set_paragraph_bidi(p, rtl)
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_runs(p, headers[index], size=SMALL_SIZE, bold=True, rtl=rtl)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for row in rows:
        tr = table.add_row()
        _set_row_cant_split(tr)
        for index, cell in enumerate(tr.cells):
            if index >= len(row):
                break
            cell.text = ""
            p = cell.paragraphs[0]
            _set_paragraph_bidi(p, rtl)
            p.paragraph_format.space_after = Pt(0)
            align = (table_block.get("align") or [])[index] if index < len(table_block.get("align") or []) else "start"
            if align == "center":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif not rtl:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _add_runs(p, row[index], size=SMALL_SIZE, rtl=rtl)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if hasattr(container, "_tc"):
        # OOXML requires a cell to end with a paragraph after a nested table.
        p = container.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(1)
    return table


def _ensure_cell_trailing_paragraph(cell) -> None:
    """OOXML requires a table cell to end with a paragraph."""
    tc = cell._tc
    if not len(tc) or tc[-1].tag != qn("w:p"):
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(1)


def _strip_empty_paragraphs(container) -> None:
    paragraphs = list(container.paragraphs)
    for paragraph in paragraphs:
        if len(list(container.paragraphs)) <= 1:
            break
        if paragraph.text.strip():
            continue
        if paragraph._p.findall(".//" + qn("w:drawing")):
            continue
        if paragraph._p.findall(".//" + qn("w:pict")):
            continue
        p_pr = paragraph._p.pPr
        if p_pr is not None and p_pr.find(qn("w:pBdr")) is not None:
            continue
        if paragraph._p.find(qn("w:tbl")) is not None:
            continue
        paragraph._p.getparent().remove(paragraph._p)


def _add_options(doc, question: dict[str, Any], rtl: bool,
                 width_mm: float = CONTENT_WIDTH_MM) -> None:
    options = question.get("options") or []
    if not options:
        return
    has_table = any(
        block.get("type") == "table"
        for option in options for block in (option.get("blocks") or [])
    )
    table_blocks = [
        block for option in options for block in (option.get("blocks") or [])
        if block.get("type") == "table"
    ]
    compact_tables_fit = bool(table_blocks) and all(
        block.get("compact")
        and len(block.get("headers") or []) <= 3
        and len(block.get("rows") or []) <= 5
        and max(
            (len(_run_text(cell)) for row in (
                [block.get("headers") or []] + list(block.get("rows") or [])
            ) for cell in row),
            default=0,
        ) <= 12
        for block in table_blocks
    )
    columns = 2 if not has_table or compact_tables_fit else 1
    rows = (len(options) + columns - 1) // columns
    table = doc.add_table(rows=rows, cols=columns)
    widths = [width_mm / columns] * columns
    _set_table_width(table, widths)
    _remove_table_borders(table)
    for index, option in enumerate(options):
        row = index // columns
        col = index % columns
        visual_col = (columns - 1 - col) if rtl else col
        cell = table.cell(row, visual_col)
        cell.text = ""
        inner_width = widths[col] - 3
        p = cell.paragraphs[0]
        _set_paragraph_bidi(p, rtl)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_together = True
        _add_run(p, f"{option.get('label', 'A')}.", size=10.2, bold=True, rtl=False)
        _add_run(p, "\u00a0", size=10.2, rtl=rtl)
        for block in option.get("blocks") or []:
            if block.get("type") == "paragraph":
                _add_runs(p, block.get("runs") or [], size=10.2, rtl=rtl)
            elif block.get("type") == "table":
                _add_table(cell, block, rtl=rtl, width_mm=max(20, inner_width))
            elif block.get("type") == "math":
                _add_display_math(p, str(block.get("latex") or ""), width_in=1.8)
        _strip_empty_paragraphs(cell)
        _ensure_cell_trailing_paragraph(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    if hasattr(doc, "_tc"):
        _ensure_cell_trailing_paragraph(doc)


def _start_question_paragraph(container, number: int, rtl: bool):
    """Create one bidi paragraph containing the marker and question text."""
    p = container.add_paragraph()
    _set_paragraph_bidi(p, rtl)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.keep_together = True
    # A strong RTL anchor is required when the entire answer is Latin (for
    # example an MCQ key of "1. A"); otherwise Word may physically place a
    # right-aligned bidi paragraph at the left margin.
    if rtl:
        _add_run(p, "\u200f", size=1, rtl=True)
    _add_run(p, f"{number}.", size=10.5, bold=True, rtl=False)
    _add_run(p, "\u00a0", size=10.5, rtl=rtl)
    return p


def _add_question_frame(container, number: int, rtl: bool, width_mm: float = CONTENT_WIDTH_MM):
    """Compatibility wrapper returning the shared inline question paragraph."""
    return container, _start_question_paragraph(container, number, rtl)


def _add_question_number(p, number: int) -> None:
    _add_run(p, f"{number}.", size=10.5, bold=True, rtl=False)
    _add_run(p, " ", size=10.5)


def _question_content(doc, question: dict[str, Any], rtl: bool,
                      include_number: bool = True) -> None:
    blocks = question.get("blocks") or []
    number = int(question.get("number") or 1)
    content = doc
    first_paragraph = _start_question_paragraph(doc, number, rtl) if include_number else None
    for block in blocks:
        if block.get("type") == "paragraph":
            p = first_paragraph or content.add_paragraph()
            if first_paragraph is None:
                _set_paragraph_bidi(p, rtl)
            first_paragraph = None
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            _add_runs(p, block.get("runs") or [], size=BODY_SIZE, bold=True, rtl=rtl)
        elif block.get("type") == "math" and block.get("display"):
            p = content.add_paragraph()
            _set_paragraph_bidi(p, rtl)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            latex = str(block.get("latex") or "")
            if not _add_display_math(p, latex, width_in=5.4):
                _add_run(p, str(block.get("plain") or latex), size=BODY_SIZE, rtl=False)
        elif block.get("type") == "table":
            _add_table(content, block, rtl=rtl)
    _add_options(content, question, rtl, width_mm=CONTENT_WIDTH_MM - 8)
    layout = question.get("response_lines") or 0
    if layout:
        _add_response_lines(content, int(layout), rtl=rtl)
    _strip_empty_paragraphs(content)


def _add_true_false(doc, question: dict[str, Any], rtl: bool) -> None:
    content = doc
    blocks = question.get("blocks") or []
    p = _start_question_paragraph(content, int(question.get("number") or 1), rtl)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_together = True
    for block in blocks:
        if block.get("type") == "paragraph":
            _add_runs(p, block.get("runs") or [], size=BODY_SIZE, bold=True, rtl=rtl)
        elif block.get("type") == "math":
            _add_run(p, str(block.get("plain") or ""), size=BODY_SIZE, rtl=False)
        elif block.get("type") == "table":
            _add_table(content, block, rtl=rtl)
    choices = str(question.get("true_false_choices") or ("صح (   ) غلط (   )" if rtl else "True (   ) False (   )"))
    # Non-breaking spaces keep the complete choice group together; Word moves
    # the whole group to the next line when there is not enough room.
    group = choices.replace(" ", "\u00a0")
    _add_run(p, "\u00a0", size=BODY_SIZE, rtl=rtl)
    _add_run(p, group, size=BODY_SIZE, bold=True, rtl=rtl)
    _strip_empty_paragraphs(content)


def _add_side_by_side(doc, question: dict[str, Any], table: dict[str, Any],
                      rtl: bool) -> None:
    table_w = 78.0
    question_w = CONTENT_WIDTH_MM - table_w - 8.0
    outer = doc.add_table(rows=1, cols=2)
    if rtl:
        widths = [table_w, question_w]
        table_cell = outer.cell(0, 0)
        question_cell = outer.cell(0, 1)
    else:
        widths = [question_w, table_w]
        question_cell = outer.cell(0, 0)
        table_cell = outer.cell(0, 1)
    _set_table_width(outer, widths)
    outer.autofit = True
    _remove_table_borders(outer)
    _set_row_cant_split(outer.rows[0])
    question_cell.text = ""
    table_cell.text = ""
    question_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    table_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    content = question_cell
    first_paragraph = _start_question_paragraph(
        content, int(question.get("number") or 1), rtl
    )
    for block in [b for b in question.get("blocks") or [] if b.get("type") != "table"]:
        if block.get("type") == "paragraph":
            p = first_paragraph or content.add_paragraph()
            if first_paragraph is None:
                _set_paragraph_bidi(p, rtl)
            first_paragraph = None
            p.paragraph_format.space_after = Pt(2)
            _add_runs(p, block.get("runs") or [], size=BODY_SIZE, bold=True, rtl=rtl)
        elif block.get("type") == "math" and block.get("display"):
            p = content.add_paragraph()
            _set_paragraph_bidi(p, rtl)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_math_image(p, str(block.get("latex") or ""), width_in=3.8)
    _add_options(content, question, rtl, width_mm=question_w - 8)
    _strip_empty_paragraphs(content)
    _ensure_cell_trailing_paragraph(content)
    nested = _add_table(table_cell, table, rtl=rtl, width_mm=table_w - 12)
    if nested is not None:
        tc = table_cell._tc
        if nested._tbl.getparent() is not tc:
            tc.insert(0, nested._tbl)
    _strip_empty_paragraphs(table_cell)
    _ensure_cell_trailing_paragraph(table_cell)


def _add_definition(doc, question: dict[str, Any], rtl: bool) -> None:
    content = doc
    p = _start_question_paragraph(content, int(question.get("number") or 1), rtl)
    p.paragraph_format.space_after = Pt(2)
    for block in question.get("blocks") or []:
        if block.get("type") == "paragraph":
            _add_runs(p, block.get("runs") or [], size=BODY_SIZE, bold=True, rtl=rtl)
    _add_run(p, ":", size=BODY_SIZE, bold=True, rtl=rtl)
    line = content.add_paragraph()
    _set_paragraph_bidi(line, rtl)
    line.paragraph_format.space_before = Pt(8)
    line.paragraph_format.space_after = Pt(2)
    _bottom_border(line)
    _strip_empty_paragraphs(content)


def _add_student(doc, document: dict[str, Any]) -> None:
    rtl = document["direction"] == "rtl"
    _add_title(doc, document)
    for section in document.get("sections") or []:
        _add_section_heading(doc, section.get("heading") or "", rtl)
        if section.get("word_bank"):
            p = doc.add_paragraph()
            _set_paragraph_bidi(p, rtl)
            _add_run(p, "  |  ".join(section["word_bank"]), size=9.2, bold=True, rtl=rtl)
        for question in section.get("items") or []:
            if question.get("qtype") == "definition":
                _add_definition(doc, question, rtl)
            elif question.get("qtype") == "true_false":
                _add_true_false(doc, question, rtl)
            else:
                table = next((b for b in question.get("blocks") or [] if b.get("type") == "table"), None)
                if table:
                    _add_side_by_side(doc, question, table, rtl)
                else:
                    _question_content(doc, question, rtl)
            _add_paragraph_spacer(doc)


def _add_answers(doc, document: dict[str, Any]) -> None:
    rtl = document["direction"] == "rtl"
    p = doc.add_paragraph()
    _set_paragraph_bidi(p, rtl)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_run(p, document["metadata"].get("answer_key_title") or "", size=16, bold=True, rtl=rtl)
    for section in document.get("sections") or []:
        _add_section_heading(doc, section.get("heading") or "", rtl)
        for question in section.get("items") or []:
            content_cell = doc
            p = _start_question_paragraph(
                doc, int(question.get("number") or 1), rtl
            )
            first = True
            for block in question.get("answer_blocks") or []:
                if block.get("type") == "paragraph":
                    pp = p if first else content_cell.add_paragraph()
                    first = False
                    if pp is not p:
                        _set_paragraph_bidi(pp, rtl)
                    pp.paragraph_format.space_after = Pt(2)
                    _add_runs(pp, block.get("runs") or [], size=9.5, rtl=rtl)
                elif block.get("type") == "math" and block.get("display"):
                    pp = content_cell.paragraphs[0] if first else content_cell.add_paragraph()
                    first = False
                    _set_paragraph_bidi(pp, rtl)
                    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    _add_display_math(pp, str(block.get("latex") or ""), width_in=4.2)
                elif block.get("type") == "table":
                    _add_table(content_cell, block, rtl=rtl, width_mm=CONTENT_WIDTH_MM - 18)
            _add_paragraph_spacer(doc)


def _add_paragraph_spacer(doc) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)


def _add_title(doc, document: dict[str, Any]) -> None:
    rtl = document["direction"] == "rtl"
    meta = document["metadata"]
    p = doc.add_paragraph()
    _set_paragraph_bidi(p, rtl)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    _add_run(p, meta.get("title") or ("امتحان" if rtl else "Examination"),
             size=20, bold=True, rtl=rtl)
    p2 = doc.add_paragraph()
    _set_paragraph_bidi(p2, rtl)
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(4)
    _add_run(p2, meta.get("model_label") or "", size=9.5, rtl=rtl)
    _bottom_border(p2)
    fields = [
        (meta.get("duration_label"), meta.get("duration")),
        (meta.get("teacher_label"), meta.get("teacher_name")),
        (meta.get("date_label"), meta.get("exam_date")),
    ]
    table = doc.add_table(rows=1, cols=3)
    _set_table_width(table, [CONTENT_WIDTH_MM / 3] * 3)
    _remove_table_borders(table)
    if rtl:
        _set_table_bidi(table)
    for index, (label, value) in enumerate(fields):
        visual = 2 - index if rtl else index
        cell = table.cell(0, visual)
        cell.text = ""
        p = cell.paragraphs[0]
        _set_paragraph_bidi(p, rtl)
        p.paragraph_format.space_after = Pt(0)
        _add_run(p, f"{label}: ", size=9, bold=True, rtl=rtl)
        _add_run(p, str(value or ""), size=9, rtl=rtl)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    _bottom_border(p)
    p = doc.add_paragraph()
    _set_paragraph_bidi(p, rtl)
    p.paragraph_format.space_after = Pt(4)
    _add_run(p, f"{meta.get('student_name_label')}: ______________________________", size=9.3, rtl=rtl)
    _add_run(p, "        ", size=9.3)
    class_value = str(meta.get("class_name") or "").strip()
    _add_run(p, f"{meta.get('class_label')}: {class_value or '________________'}", size=9.3, rtl=rtl)


def _add_section_heading(doc, heading: str, rtl: bool) -> None:
    p = doc.add_paragraph()
    _set_paragraph_bidi(p, rtl)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    _add_run(p, heading, size=11.5, bold=True, rtl=rtl)
    _bottom_border(p)


def _bottom_border(p) -> None:
    p_pr = p._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr")
        p_pr.append(borders)
    bottom = borders.find(qn("w:bottom"))
    if bottom is None:
        bottom = OxmlElement("w:bottom")
        borders.append(bottom)
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "000000")


def _configure(doc: Document, *, rtl: bool) -> None:
    section = doc.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    for attr in ("top_margin", "right_margin", "bottom_margin", "left_margin"):
        setattr(section, attr, Mm(PAGE_MARGIN_MM))
    section.header_distance = Mm(6)
    section.footer_distance = Mm(6)
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_bidi(p, rtl)
    _add_run(p, "صفحة " if rtl else "Page ", size=8, rtl=rtl)
    _add_field(p, "PAGE", "1")
    _add_run(p, " من " if rtl else " of ", size=8, rtl=rtl)
    _add_field(p, "NUMPAGES", "1")
    normal = doc.styles["Normal"]
    normal.font.name = FONT_NAME
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:cs"), FONT_NAME)
    normal.font.size = Pt(BODY_SIZE)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_after = Pt(2)
    normal.paragraph_format.line_spacing = 1.08


def _add_field(p, instruction: str, placeholder: str) -> None:
    run = p.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, instr, separate, text, end):
        run._r.append(node)
    _set_run_font(run, size=8)


def _render(exam: dict[str, Any], metadata: dict[str, Any], *, kind: str) -> bytes:
    document = build_document_ir(exam, metadata, kind=kind)
    doc = Document()
    _configure(doc, rtl=document["direction"] == "rtl")
    doc.core_properties.title = document["metadata"].get("title") or "Exam"
    doc.core_properties.language = document.get("language") or "en"
    if kind == "answers":
        _add_answers(doc, document)
    else:
        _add_student(doc, document)
    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()


def render_exam_docx(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    return _render(exam, metadata, kind="student")


def render_answers_docx(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    return _render(exam, metadata, kind="answers")
