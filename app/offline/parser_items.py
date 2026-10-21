from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any


EXPECTED_ITEM_TYPES = ("heading", "text", "table")
# LlamaParse may emit visual layout records alongside textual JSON items.
# DraftWork does not consume image bytes yet, so these records are deliberately
# removed at the parser boundary while their surrounding text/captions remain.
IGNORED_ITEM_TYPES = ("image",)


class ParserSchemaError(ValueError):
    """Raised when LlamaParse JSON does not match the verified schema."""


def document_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return and validate the ``pages`` array from one LlamaParse result."""
    if not isinstance(payload, dict):
        raise ParserSchemaError("LlamaParse document payload must be an object")
    if "pages" not in payload:
        raise ParserSchemaError("LlamaParse JSON is missing the 'pages' field")

    pages = payload["pages"]
    if not isinstance(pages, list) or not pages:
        raise ParserSchemaError("LlamaParse JSON 'pages' must be a non-empty array")
    return pages


def page_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a page's item array; an empty array is valid for blank pages."""
    if not isinstance(page, dict):
        raise ParserSchemaError("Each LlamaParse page must be an object")
    if "items" not in page:
        raise ParserSchemaError("LlamaParse page is missing the 'items' field")

    items = page["items"]
    if not isinstance(items, list):
        raise ParserSchemaError("LlamaParse page 'items' must be an array")
    if not all(isinstance(item, dict) for item in items):
        raise ParserSchemaError("Every LlamaParse item must be an object")
    return items


def page_number(page: dict[str, Any]) -> int | None:
    value = page.get("page")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParserSchemaError("LlamaParse page 'page' must be an integer or null")
    return value


def item_type(item: dict[str, Any]) -> str:
    value = item.get("type")
    if not isinstance(value, str) or not value:
        raise ParserSchemaError("LlamaParse item is missing a valid 'type'")
    return value


def item_text(item: dict[str, Any]) -> str:
    """Return canonical text while retaining LlamaParse's original fields.

    Tables commonly carry ``value=null`` and place their actual content in
    ``md``, ``csv`` or ``rows``.  Choosing one deterministic representation at
    this boundary prevents downstream cleaners from silently dropping them.
    """
    value = item.get("value")
    if value is not None and not isinstance(value, str):
        raise ParserSchemaError("LlamaParse item 'value' must be a string or null")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if item.get("type") != "table":
        return ""
    return canonical_table_text(item)


def canonical_table_text(item: dict[str, Any]) -> str:
    """Return the richest usable table representation without truncation."""
    md = item.get("md")
    if isinstance(md, str) and md.strip():
        return md.strip()
    csv_text = item.get("csv")
    if isinstance(csv_text, str) and csv_text.strip():
        return csv_text.strip()
    rows = item.get("rows")
    rendered = _render_table_rows(rows)
    if rendered:
        return rendered
    html = item.get("html")
    if isinstance(html, str) and html.strip():
        converted = _table_html_to_markdown(html)
        return converted or html.strip()
    # Preserve any unusual parser representation rather than discarding it.
    for key in ("text", "content", "raw"):
        candidate = item.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    if rows is not None:
        return json.dumps(rows, ensure_ascii=False, default=str)
    return ""


def _render_table_rows(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return ""
    normalized: list[list[str]] = []
    for row in rows:
        if isinstance(row, dict):
            cells = list(row.values())
        elif isinstance(row, (list, tuple)):
            cells = list(row)
        else:
            return ""
        normalized.append([str(cell if cell is not None else "").replace("|", r"\|") for cell in cells])
    width = max((len(row) for row in normalized), default=0)
    if not width:
        return ""
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    lines = ["| " + " | ".join(row) + " |" for row in normalized]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"th", "td"} and self._row is not None and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _table_html_to_markdown(value: str) -> str:
    parser = _TableHTMLParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return ""
    return _render_table_rows(parser.rows)


def validate_pages(pages: list[dict[str, Any]]) -> dict[str, int]:
    """Validate the verified schema while allowing individual blank pages."""
    if not isinstance(pages, list) or not pages:
        raise ParserSchemaError("LlamaParse returned no pages")

    counts = {item_kind: 0 for item_kind in EXPECTED_ITEM_TYPES}
    total_items = 0
    for page in pages:
        page_number(page)
        for item in page_items(page):
            total_items += 1
            kind = item_type(item)
            if kind in IGNORED_ITEM_TYPES:
                continue
            if kind not in counts:
                raise ParserSchemaError(
                    f"Unsupported LlamaParse item type {kind!r}; "
                    f"expected {', '.join(EXPECTED_ITEM_TYPES)}"
                )
            item_text(item)
            counts[kind] += 1

    if total_items == 0:
        raise ParserSchemaError("LlamaParse pages contain no JSON items")
    return counts


def filter_supported_items(
    pages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop known non-text layout records after schema validation."""
    filtered: list[dict[str, Any]] = []
    ignored = {kind: 0 for kind in IGNORED_ITEM_TYPES}
    for page in pages:
        kept: list[dict[str, Any]] = []
        for item in page_items(page):
            kind = item_type(item)
            if kind in ignored:
                ignored[kind] += 1
                continue
            kept.append(item)
        filtered.append({**page, "items": kept})
    return filtered, {kind: count for kind, count in ignored.items() if count}
