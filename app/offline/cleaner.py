from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from typing import Any

from app.logging_conf import get_logger
from app.offline.parser_items import item_text, page_items, page_number

logger = get_logger("CLEANER")

# ---------------------------------------------------------------------------
# Normalization helpers (operate on plain strings).
# ---------------------------------------------------------------------------

_UNICODE_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
    "\ufeff": "",
    "\u00ad": "",
}

_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*\d+|\d+\s*|[-|]\s*\d+\s*[-|])\s*$", re.IGNORECASE
)
_COPYRIGHT_RE = re.compile(
    r"\b(copyright|all rights reserved|isbn|published by|publisher|legal disclaimer)\b",
    re.IGNORECASE,
)
_BOILERPLATE_RE = re.compile(
    r"(this page intentionally left blank|printed in the united states|"
    r"no part of this (?:book|publication) may)",
    re.IGNORECASE,
)
_DECORATIVE_RE = re.compile(r"^\s*([*\-=~_]\s*){3,}$")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
_HYPHENATION_RE = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])")
_SENTENCE_END_RE = re.compile(r"[.!?:;)\]\"']")
_HEADING_PREFIX_RE = re.compile(r"^(chapter|section|part|appendix|module|lesson)\b|\d+(\.\d+)*\s+\S", re.IGNORECASE)
_TABLE_OR_CODE_RE = re.compile(r"^(>>>|```| {4,}|def\s+|class\s+)|\t|\|")


def normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    for source, target in _UNICODE_MAP.items():
        normalized = normalized.replace(source, target)
    return normalized


def normalize_whitespace(text: str) -> str:
    text = text.replace("\t", " ")
    text = "\n".join(_MULTISPACE_RE.sub(" ", line).rstrip() for line in text.splitlines())
    text = _EXCESSIVE_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def fix_hyphenation(text: str) -> str:
    return _HYPHENATION_RE.sub("", text)


def fix_line_breaks(text: str) -> str:
    """Merge wrapped lines while preserving paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    fixed: list[str] = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        merged = lines[0]
        for line in lines[1:]:
            if _should_preserve_break(merged, line):
                merged += "\n" + line
            else:
                merged += " " + line
        fixed.append(merged)
    return "\n\n".join(fixed)


def clean_text(text: str) -> str:
    """Full normalization of a plain-text block (used by the chunker)."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = fix_hyphenation(text)
    text = fix_line_breaks(text)
    text = normalize_whitespace(text)
    return text


def _should_preserve_break(previous: str, current: str) -> bool:
    if _HEADING_PREFIX_RE.match(current):
        return True
    if _TABLE_OR_CODE_RE.search(previous) or _TABLE_OR_CODE_RE.search(current):
        return True
    if re.match(r"^(figure|fig\.|table)\s+\d+", current, re.IGNORECASE):
        return True
    if _SENTENCE_END_RE.search(previous):
        return True
    return False


# ---------------------------------------------------------------------------
# Page-level cleaning (headers, footers, page numbers, boilerplate).
# ---------------------------------------------------------------------------


def _line_key(text: str) -> str:
    key = normalize_unicode(text).lower()
    key = re.sub(r"\d+", "#", key)
    key = re.sub(r"\W+", " ", key)
    return key.strip()


def _is_boilerplate(text: str) -> bool:
    return bool(text and (_BOILERPLATE_RE.search(text) or _COPYRIGHT_RE.search(text)))


def _detect_headers_footers(pages: list[dict[str, Any]]) -> set[str]:
    """Find lines repeated at the top/bottom of many pages (headers/footers)."""
    n = len(pages)
    if n < 3:
        return set()
    threshold = max(3, int(n * 0.6))
    top_counts: Counter[str] = Counter()
    bottom_counts: Counter[str] = Counter()

    for page in pages:
        texts = [item_text(i) for i in page_items(page) if item_text(i)]
        if not texts:
            continue
        for line in texts[:3]:
            top_counts[_line_key(line)] += 1
        for line in texts[-3:]:
            bottom_counts[_line_key(line)] += 1

    repeated = {
        key for key, count in top_counts.items() if count >= threshold
    }
    repeated |= {key for key, count in bottom_counts.items() if count >= threshold}
    return repeated


def clean_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean LlamaParse pages: headers/footers, page numbers, boilerplate, noise."""
    t0 = time.perf_counter()
    chars_before = sum(len(item_text(i)) for p in pages for i in page_items(p))

    repeated = _detect_headers_footers(pages)

    cleaned_pages: list[dict[str, Any]] = []
    for page in pages:
        items_out: list[dict[str, Any]] = []
        for item in page_items(page):
            raw = item_text(item)
            if not raw:
                continue
            key = _line_key(raw)
            if key in repeated:
                continue
            if _PAGE_NUMBER_RE.match(raw) or _is_boilerplate(raw) or _DECORATIVE_RE.match(raw):
                continue
            cleaned = clean_text(raw)
            if not cleaned:
                continue
            new_item = dict(item)
            new_item["value"] = cleaned
            if "md" in new_item:
                new_item["md"] = cleaned
            items_out.append(new_item)
        if items_out:
            cleaned_pages.append({"page": page_number(page), "items": items_out})

    chars_after = sum(len(item_text(i)) for p in cleaned_pages for i in page_items(p))
    elapsed = time.perf_counter() - t0
    logger.info(
        "Cleaning completed | pages=%d | chars_before=%d | chars_after=%d | headers_footers=%d | time=%.2fs",
        len(cleaned_pages),
        chars_before,
        chars_after,
        len(repeated),
        elapsed,
    )
    return cleaned_pages
