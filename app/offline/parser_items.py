from __future__ import annotations

from typing import Any


EXPECTED_ITEM_TYPES = ("heading", "text", "table")


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
    """Return canonical item text from the verified ``value`` field."""
    value = item.get("value")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ParserSchemaError("LlamaParse item 'value' must be a string or null")
    return value.strip()


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
