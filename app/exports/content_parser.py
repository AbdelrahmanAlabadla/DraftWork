"""Structured, renderer-independent parsing for exam content.

The generation layer stores question content as rich plain text containing
LaTeX delimiters, Markdown pipe tables, and LaTeX array/tabular tables.  This
module turns that text into JSON-serializable blocks once.  Renderers must not
re-parse raw question strings.

The parser is deliberately conservative:

* a malformed table or expression is preserved in a readable fallback block;
* a diagnostic is emitted with the original source;
* nothing is invented to fill missing rows, cells, or values.
"""
from __future__ import annotations

import re
from typing import Any

from app.exports.math_renderer import latex_to_plain_text

IR_VERSION = "1.0"

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_FILL_BLANK_RE = re.compile(r"_{3,}|-{3,}")


def normalize_numerals(value: object) -> str:
    """Normalize Arabic-Indic and Persian digits without changing values."""
    return str(value or "").translate(_ARABIC_INDIC)


def normalize_math_space(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\\(?:quad|qquad)\b", " ", value)
    value = re.sub(r"\\[,;:!]", " ", value)
    value = re.sub(r"\\\s", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def readable_math_fallback(value: str) -> str:
    """Last-resort readable form; never returns raw backslash commands."""
    plain = latex_to_plain_text(value)
    if plain is not None:
        return normalize_math_space(plain)
    text = str(value or "")
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b", " ", text)
    text = re.sub(r"\\(?:mathrm|mathbf|mathit|text|operatorname)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\s*", " ", text)
    text = text.replace("{", "").replace("}", "").replace("&", " ")
    text = normalize_math_space(text)
    return text or "—"


def _cell_runs(value: str, *, language: str = "en") -> list[dict[str, Any]]:
    """Parse a table cell into text/math runs."""
    runs: list[dict[str, Any]] = []
    text = normalize_numerals(value or "").strip()
    # Array cells commonly contain textual labels rather than math.  Strip
    # text wrappers before looking for delimited math.
    text = re.sub(
        r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}",
        r"\1",
        text,
    )
    text = text.replace(r"\%", "%").replace(r"\&", "&")
    text = normalize_math_space(text)
    pattern = re.compile(
        r"\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|"
        r"\\\(([\s\S]*?)\\\)|(?<!\$)\$([^$\n]+?)\$(?!\$)"
    )
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            runs.append({"type": "text", "text": text[cursor:match.start()]})
        latex = next((group for group in match.groups() if group is not None), "")
        runs.append({
            "type": "math",
            "latex": latex.strip(),
            "display": False,
            "plain": readable_math_fallback(latex),
        })
        cursor = match.end()
    if cursor < len(text):
        runs.append({"type": "text", "text": text[cursor:]})
    if not runs:
        runs.append({"type": "text", "text": text})
    return runs


def _table(
    *,
    headers: list[list[dict[str, Any]]],
    rows: list[list[list[dict[str, Any]]]],
    align: list[str] | None = None,
    caption: str = "",
    compact: bool = False,
    source: str = "",
) -> dict[str, Any]:
    return {
        "type": "table",
        "caption": caption,
        "headers": headers,
        "rows": rows,
        "align": align or ["start"] * len(headers),
        "compact": compact,
        "source": source,
    }


def _parse_array_body(body: str) -> tuple[list[list[str]], list[str]]:
    """Split a LaTeX array/tabular body into cells and alignment hints."""
    value = re.sub(r"\\(?:hline|cline)\b(?:\{[^{}]*\})?", " ", body)
    # A generated table sometimes ends its first row with a single backslash
    # before the newline.  Treat that newline as a row boundary without
    # changing cell text.
    value = re.sub(r"\\\s*\n(?=\s*\\text)", "\\\\\n", value)
    raw_rows = re.split(r"\\\\|\n(?=\s*\\text)", value)
    cells: list[list[str]] = []
    for raw in raw_rows:
        raw = raw.strip()
        if not raw:
            continue
        raw = raw.rstrip("\\").strip()
        parts = [part.strip() for part in raw.split("&")]
        if any(parts):
            cells.append(parts)
    return cells, []


def parse_latex_array(value: str, *, language: str = "en") -> dict[str, Any] | None:
    """Parse a LaTeX array/tabular into a native table block, or None."""
    match = re.search(
        r"\\begin\{(array|tabular)\}(?:\{([^{}]*)\})?([\s\S]*?)\\end\{\1\}",
        value,
    )
    if not match:
        return None
    source = match.group(0)
    spec = match.group(2) or ""
    cells, _ = _parse_array_body(match.group(3))
    if not cells:
        return {
            "type": "table",
            "caption": "",
            "headers": [],
            "rows": [],
            "align": [],
            "compact": False,
            "source": source,
            "diagnostic": "Could not parse any rows from this table.",
        }
    width = max(len(row) for row in cells)
    # Column alignment: c=center, l=start, r=end.  The vertical bar is only a
    # visual rule and must not create a column.
    align = []
    for char in re.sub(r"[^clr]", "", spec):
        align.append({"c": "center", "l": "start", "r": "end"}[char])
    while len(align) < width:
        align.append("start")
    headers = [_cell_runs(cell, language=language) for cell in cells[0]]
    rows = [
        [_cell_runs(cell, language=language) for cell in row]
        for row in cells[1:]
    ]
    return _table(
        headers=headers,
        rows=rows,
        align=align[:width],
        compact=False,
        source=source,
    )


def _split_markdown_row(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    result: list[str] = []
    current: list[str] = []
    escaped = False
    for char in text:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
            current.append(char)
        elif char == "|":
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    result.append("".join(current).strip())
    return result


def _is_markdown_separator(line: str) -> bool:
    text = line.strip()
    if "|" not in text:
        return False
    cells = _split_markdown_row(text)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def parse_markdown_table(lines: list[str], start: int, *, language: str = "en") -> tuple[dict[str, Any], int] | None:
    if start + 1 >= len(lines):
        return None
    if "|" not in lines[start] or not _is_markdown_separator(lines[start + 1]):
        return None
    header_cells = _split_markdown_row(lines[start])
    separator = _split_markdown_row(lines[start + 1])
    align = []
    for cell in separator:
        stripped = cell.strip()
        if stripped.startswith(":") and stripped.endswith(":"):
            align.append("center")
        elif stripped.endswith(":"):
            align.append("end")
        else:
            align.append("start")
    if len(align) < len(header_cells):
        align += ["start"] * (len(header_cells) - len(align))
    align = align[:len(header_cells)]
    rows: list[list[list[dict[str, Any]]]] = []
    index = start + 2
    while index < len(lines) and "|" in lines[index] and lines[index].strip():
        cells = _split_markdown_row(lines[index])
        if len(cells) < len(header_cells):
            cells += [""] * (len(header_cells) - len(cells))
        rows.append([_cell_runs(cell, language=language) for cell in cells[:len(header_cells)]])
        index += 1
    return (
        _table(
            headers=[_cell_runs(cell, language=language) for cell in header_cells],
            rows=rows,
            align=align,
            source="\n".join(lines[start:index]),
        ),
        index,
    )


def parse_coordinate_table(value: str, *, language: str = "en") -> dict[str, Any] | None:
    """Recognize a coordinate list used as an option/table.

    This repairs the common legacy case where a table was flattened into
    ``النقاط: (-4,0)، (0,3)، (4,0)``.  It only uses pairs explicitly present in
    the source.
    """
    text = normalize_numerals(value or "")
    match = re.search(r"(?:النقاط|نقاط|Points?)\s*:\s*(.+)", text, re.IGNORECASE)
    body = match.group(1) if match else text
    pairs = re.findall(
        r"\(\s*([-+]?\d+(?:\.\d+)?)\s*[,،]\s*([-+]?\d+(?:\.\d+)?)\s*\)",
        body,
    )
    if len(pairs) < 2:
        return None
    return _table(
        headers=[_cell_runs("x", language=language), _cell_runs("y", language=language)],
        rows=[
            [_cell_runs(x, language=language), _cell_runs(y, language=language)]
            for x, y in pairs
        ],
        align=["center", "center"],
        compact=True,
        source=value,
    )


_MATH_RE = re.compile(
    r"\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|"
    r"\\\(([\s\S]*?)\\\)|(?<!\$)\$([^$\n]+?)\$(?!\$)"
)


def _math_run(latex: str, *, display: bool) -> dict[str, Any]:
    return {
        "type": "math",
        "latex": latex.strip(),
        "display": display,
        "plain": readable_math_fallback(latex),
    }


def _paragraph(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    cleaned = []
    for run in runs:
        if run["type"] == "text":
            text = run["text"]
            text = re.sub(r"[ \t]+", " ", text)
            if text.strip():
                cleaned.append({"type": "text", "text": text})
        else:
            cleaned.append(run)
    if not any(
        (run.get("text") or "").strip() if run["type"] == "text" else run.get("latex")
        for run in cleaned
    ):
        return None
    return {"type": "paragraph", "runs": cleaned}


def parse_rich_text(value: object, *, language: str = "en") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse prose, inline math, display math, Markdown tables and arrays."""
    text = normalize_numerals(value or "")
    blocks: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    # A flattened coordinate table (legacy MCQ) is a table, not a sentence.
    stripped = text.strip()
    if re.match(r"^(?:النقاط|نقاط|Points?)\s*:", stripped, re.IGNORECASE):
        coordinate = parse_coordinate_table(stripped, language=language)
        if coordinate:
            return [coordinate], []

    # First extract tables; they are independent blocks and must never be
    # treated as prose or as a single math image.
    index = 0
    buffer: list[str] = []
    lines = text.splitlines()
    while index < len(lines):
        markdown = parse_markdown_table(lines, index, language=language)
        if markdown:
            table, index = markdown
            before = "\n".join(buffer).strip()
            if before:
                blocks.extend(_parse_prose(before, language=language, diagnostics=diagnostics))
            blocks.append(table)
            buffer = []
            continue
        # A LaTeX array may be wrapped in \[...\]; extract it as a table.
        array_match = re.search(
            r"\\begin\{(array|tabular)\}(?:\{[^{}]*\})?[\s\S]*?\\end\{\1\}",
            lines[index],
        )
        if array_match:
            before = "\n".join(buffer).strip()
            if before.endswith("\\["):
                before = before[:-2].strip()
            if before:
                blocks.extend(_parse_prose(before, language=language, diagnostics=diagnostics))
            table = parse_latex_array(array_match.group(0), language=language)
            if table:
                if table.get("diagnostic"):
                    diagnostics.append({
                        "code": "table_malformed",
                        "message": table["diagnostic"],
                        "source": table.get("source", ""),
                    })
                blocks.append(table)
            else:
                diagnostics.append({
                    "code": "table_unparsed",
                    "message": "Table could not be parsed; original content preserved.",
                    "source": array_match.group(0),
                })
                blocks.extend(_parse_prose(lines[index], language=language, diagnostics=diagnostics))
            buffer = []
            index += 1
            continue
        buffer.append(lines[index])
        index += 1

    tail = "\n".join(buffer).strip()
    if tail in {r"\]", r"\}"}:
        tail = ""
    if tail:
        blocks.extend(_parse_prose(tail, language=language, diagnostics=diagnostics))

    if not blocks:
        blocks.append({"type": "paragraph", "runs": [{"type": "text", "text": "—"}]})
    return blocks, diagnostics


def _parse_prose(value: str, *, language: str, diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    # Display math environments without delimiters are standalone blocks.
    undelimited = re.search(
        r"(?<!\\)\\begin\{(array|tabular|aligned|align|cases|matrix|bmatrix|pmatrix)\}"
        r"[\s\S]*?\\end\{\1\}",
        value,
    )
    if undelimited:
        before = value[:undelimited.start()].strip()
        after = value[undelimited.end():].strip()
        # The environment can be embedded in ``$A=\begin{bmatrix}...$``.
        # Remove only the delimiter touching the extracted environment; keep
        # the meaningful prefix (``A=``) as prose and never leak a raw dollar.
        before = re.sub(r"(?:\\(?:\(|\[)|\${1,2})\s*$", "", before).strip()
        before = re.sub(r"\$(?=[^$]*$)", "", before).strip()
        after = re.sub(r"^\s*(?:\\(?:\)|\])|\${1,2})", "", after).strip()
        if before:
            blocks.extend(_parse_paragraphs(before, language=language, diagnostics=diagnostics))
        env = undelimited.group(1)
        if env in {"array", "tabular"}:
            table = parse_latex_array(undelimited.group(0), language=language)
            if table:
                blocks.append(table)
        else:
            blocks.append(_math_run(undelimited.group(0), display=True))
        if after:
            blocks.extend(_parse_paragraphs(after, language=language, diagnostics=diagnostics))
        return blocks
    return _parse_paragraphs(value, language=language, diagnostics=diagnostics)


def _parse_paragraphs(value: str, *, language: str, diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    raw_paragraphs = re.split(r"\n\s*\n", value)
    for raw in raw_paragraphs:
        raw = raw.strip()
        if not raw:
            continue
        if raw in {r"\[", r"\]"}:
            continue
        runs: list[dict[str, Any]] = []
        cursor = 0
        for match in _MATH_RE.finditer(raw):
            if match.start() > cursor:
                runs.append({"type": "text", "text": raw[cursor:match.start()]})
            latex = next((group for group in match.groups() if group is not None), "")
            display = match.group(1) is not None or match.group(2) is not None
            runs.append(_math_run(latex, display=display))
            cursor = match.end()
        if cursor < len(raw):
            runs.append({"type": "text", "text": raw[cursor:]})
        # Display math is a block even when it shares a source paragraph with
        # prose or another display expression.
        if any(run["type"] == "math" and run["display"] for run in runs):
            current: list[dict[str, Any]] = []
            for run in runs:
                if run["type"] == "math" and run["display"]:
                    if current:
                        paragraph = _paragraph(current)
                        if paragraph:
                            blocks.append(paragraph)
                        current = []
                    blocks.append(run)
                else:
                    current.append(run)
            if current:
                paragraph = _paragraph(current)
                if paragraph:
                    blocks.append(paragraph)
        else:
            paragraph = _paragraph(runs)
            if paragraph:
                blocks.append(paragraph)
    return blocks


def parse_question_content(value: object, *, language: str = "en") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return parse_rich_text(value, language=language)
