from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.logging_conf import get_logger
from app.offline.parser_items import item_text, page_items, page_number

logger = get_logger("CLEANER")

# ---------------------------------------------------------------------------
# Normalization helpers (operate on plain strings).
# ---------------------------------------------------------------------------

# \u2010 (hyphen) and \u00ad (soft hyphen) appear when PDF extraction splits a
# word across a line. They must be dropped so ``tech\u2010niques`` -> ``techniques``.
_UNICODE_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    # \u2010 (hyphen) / \u2011 (non-breaking hyphen) are handled by
    # _LETTER_HYPHEN_RE so the space after a mid-word split is also removed.
    "\u00ad": "",
    "\u00a0": " ",
    "\ufeff": "",
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
_WATERMARK_RE = re.compile(
    r"\b(confidential|do not (?:copy|distribute|reproduce)|for (?:internal|"
    r"evaluation|review|research) use(?:\s+only)?|unapproved|unedited copy|"
    r"draft(?:\s+for\s+(?:review|comment))?)\b",
    re.IGNORECASE,
)
_DECORATIVE_RE = re.compile(r"^\s*([*\-=~_#]\s*){3,}$")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
_HYPHENATION_RE = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])")
# Unicode hyphen (possibly + space) between two lowercase letters -> join the word.
_LETTER_HYPHEN_RE = re.compile(r"(?<=[a-z])[\u2010\u2011]+\s*(?=[a-z])")
_SENTENCE_END_RE = re.compile(r"[.!?:;)\]\"']")
_HEADING_PREFIX_RE = re.compile(
    r"^(chapter|section|part|appendix|module|lesson)\b|\d+(\.\d+)*\s+\S",
    re.IGNORECASE,
)
_TABLE_OR_CODE_RE = re.compile(r"^(>>>|```| {4,}|def\s+|class\s+)|\t|\|")

# Figure/table/equation caption that *begins* the paragraph (short => caption only).
_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|table|tab\.?|equation|eq\.?|exhibit|diagram|plate)\b\s*\d+",
    re.IGNORECASE,
)
# Running chapter/section head carrying a page number, e.g. "8 | Chapter 1: ..."
# or "Chapter 1: The Machine Learning Landscape Types of Machine Learning Systems".
_RUNNING_HEAD_LEADING_RE = re.compile(
    r"^\s*\d+\s*[|:[\]\)\-]+\s*(?:chapter|section|part|module|lesson|unit|appendix)\b",
    re.IGNORECASE,
)
_RUNNING_HEAD_TRAILING_RE = re.compile(
    r"^\s*(?:chapter|section|part|module|lesson|unit|appendix)\b"
    r"[^A-Za-z]{0,3}.*\s\d+\s*$",
    re.IGNORECASE,
)
# Table of contents row: a topic line whose dotted leaders run into a page number.
_TOC_RE = re.compile(r"[\.\u00b7\u2026·]{3,}\s*\d+([-,]\d+)*\s*$")
# Footnote-only paragraph: a bare marker followed by a capitalized sentence.
_FOOTNOTE_PARA_RE = re.compile(r"^\s*\d{1,3}\s+[A-Z][a-z]")
# Generic bibliography / references heading.
_BIBLIO_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works cited|further reading)\b[:.]?\s*$",
    re.IGNORECASE,
)
# Inline footnote markers embedded in a paragraph:
#  * after sentence punctuation:  "... spam). 1 Fun fact: ..."
#  * attached right after a period: "(Figure 1-6).1 To train"
#  * glued to the previous word:  "Neural networks2 Some neural..."
_FOOTNOTE_SPACED_RE = re.compile(r"(?<=[.!?)\"'])\s+\d{1,3}\s+(?=[A-Z][a-z])")
_FOOTNOTE_ATTACHED_RE = re.compile(r"(?<=[.!?)\"'])\d{1,3}(?=[A-Z][a-z])")
_FOOTNOTE_GLUED_RE = re.compile(r"(?<=[a-z])\d{1,2}(?=\s+[A-Z][a-z])")

# Padding left over where an image placeholder used to be, e.g. "called .".
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?)\]}\"'])")

# Clear standalone diagram axis labels (very conservative; exact label-shape only).
_DIAGRAM_FRAGMENT_RE = re.compile(
    r"^\s*(?:launch!?|data\b|training\s+set|new\s+instance|feature\s*\d*|"
    r"feature\s+\d+\s+feature|feature\s+\d|class\??|value\s*(?:value|feature|example)\??|"
    r"can\s+be\s+automated|study\s+the\s+problem|analyze\s+errors|"
    r"evaluate(?: solution)?|input|output|update\s+data|feature\s*1\b)\s*\.?\s*$",
    re.IGNORECASE,
)

# A figure/table/equation caption whose body follows the marker with a caption
# separator (":" or "."): e.g. "Figure 2.1: The layered architecture ...". These
# are long captions the short _CAPTION_RE word cap would otherwise keep.
_CAPTION_SEPARATOR_RE = re.compile(
    r"^\s*(figure|fig\.?|table|tab\.?|equation|eq\.?|exhibit|diagram|plate)"
    r"\b\s*\d+(?:[-.]\d+)*\s*([:.])",
    re.IGNORECASE,
)
# A real prose reference to a figure ("Figure 2.1 shows / illustrates ...") is
# body text, NOT a caption; it must be preserved.
_CAPTION_REFERENCE_RE = re.compile(
    r"^\s*(figure|fig\.?|table|tab\.?|equation|eq\.?|exhibit|diagram|plate)"
    r"\b\s*\d+(?:[-.]\d+)*\s+(?:shows?|show|illustrates?|illustrate|depicts?|"
    r"depict|represents?|represent|presents?|present|provides?|provide|"
    r"lists?|list|summarizes?|summarize|gives?|give|displays?|display|"
    r"compares?|compare|demonstrates?|demonstrate)\b",
    re.IGNORECASE,
)

# Standalone credit / source attribution line ("Source: ...", "Image courtesy
# of ...", "Reproduced with permission ..."). Only a short, leading attribution.
_CREDIT_SOURCE_RE = re.compile(
    r"^\s*(?:source|data source|credit|credits|image credit|photo credit|"
    r"courtesy(?: of)?|photo by|image by|adapted from|reprinted(?: from)?|"
    r"reproduced(?:\s+with\s+permission)?|data from|with permission(?: from)?|"
    r"this image is)\b[.:]?\s+",
    re.IGNORECASE,
)

# Bare or dominant URL (http(s)://... or www....).
_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
_URL_ONLY_RE = re.compile(
    r"^\s*(?:(?:https?://[^\s]+|www\.[^\s]+)[\s.,;:!?\u2019\"']*)+$",
    re.IGNORECASE,
)

# Mojibake / extraction artifacts: U+FFFD replacement chars and stray control
# bytes that survive LlamaParse (e.g. "I�" cut off mid-character).
_REPLACEMENT_CHAR_RE = re.compile(r"[\uFFFD\uFFFE\uFFFF]")
_CONTROL_GARBAGE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Multiple-Choice question answer-option markers (inline) and an explicit answer.
_MCQ_OPTION_INLINE_RE = re.compile(
    r"\b(?:[A-Ea-e]\s*[.)]|\([A-Ea-e]\s*\))\s+\S"
)
_MCQ_ANSWER_RE = re.compile(
    r"\b(?:(?:correct\s+)?answer|ans)\s*[:=]?\s*[A-Ea-e](?=[\s.,;)]|$)",
    re.IGNORECASE,
)

# Categories used for the cleaning report.
CATEGORY_TINY = "tiny_fragment"
CATEGORY_EMPTY = "empty"
CATEGORY_DECORATIVE = "decorative"
CATEGORY_PAGE_NUMBER = "page_number"
CATEGORY_COPYRIGHT = "copyright_boilerplate"
CATEGORY_WATERMARK = "watermark"
CATEGORY_FIGURE_CAPTION = "figure_caption"
CATEGORY_DIAGRAM = "diagram_fragment"
CATEGORY_RUNNING_HEAD = "running_head"
CATEGORY_HEADER_FOOTER = "header_footer"
CATEGORY_FOOTNOTE = "footnote"
CATEGORY_TOC = "toc"
CATEGORY_BIBLIOGRAPHY = "bibliography"
CATEGORY_OCR = "ocr_garbage"
CATEGORY_CREDIT = "credit_source"
CATEGORY_URL = "url"
CATEGORY_MCQ = "mcq"
CATEGORY_DUPLICATE = "duplicate"


@dataclass
class CleaningStats:
    """Per-clean totals for reporting how much text was removed and why."""

    paragraphs_processed: int = 0
    paragraphs_removed: int = 0
    inline_footnotes_removed: int = 0
    duplicates_removed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    by_category: Counter[str] = field(default_factory=Counter)

    @property
    def percent_text_removed(self) -> float:
        if self.chars_before <= 0:
            return 0.0
        return round((self.chars_before - self.chars_after) / self.chars_before * 100, 2)


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
    text = _HYPHENATION_RE.sub("", text)
    return _LETTER_HYPHEN_RE.sub("", text)


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
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text


def strip_inline_footnotes(text: str) -> tuple[str, int]:
    """Remove inline footnote markers embedded in a paragraph.

    Returns the cleaned text and how many markers were dropped. Only removes
    numbers that sit right after a sentence boundary and are followed by a
    capitalised word, which is how superscript footnotes survive parsing.
    """
    text, n1 = _FOOTNOTE_SPACED_RE.subn(" ", text)
    text, n2 = _FOOTNOTE_ATTACHED_RE.subn("", text)
    text, n3 = _FOOTNOTE_GLUED_RE.subn("", text)
    return text, n1 + n2 + n3


def strip_inline_garbage(text: str) -> tuple[str, int]:
    """Remove extraction artifacts embedded inside a kept paragraph.

    Drops U+FFFD replacement characters, stray control bytes and bare URLs.
    Returns the cleaned text and how many artifact markers were removed so the
    caller can decide whether the paragraph is too damaged to keep.
    """
    repl = len(_REPLACEMENT_CHAR_RE.findall(text))
    text = _REPLACEMENT_CHAR_RE.sub("", text)
    text = _CONTROL_GARBAGE_RE.sub("", text)
    text = _URL_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip(), repl


def strip_mcq_suffix(text: str) -> tuple[str, bool]:
    """Remove a multiple-choice answer block attached to a normal paragraph.

    When a paragraph carries several answer options (``A) ...`` / ``(A) ...``)
    or an explicit ``Answer: X`` line, the MCQ tail is garbage for retrieval.
    If a real prose sentence leads the block it is preserved and only the MCQ
    tail is stripped; otherwise the caller removes the whole paragraph.

    Returns ``(kept_text, removed)`` where ``removed=True`` means an MCQ block
    was present. ``kept_text`` is ``""`` when the paragraph is nothing but MCQ.
    """
    if not text:
        return text, False
    markers = list(_MCQ_OPTION_INLINE_RE.finditer(text))
    has_answer = _MCQ_ANSWER_RE.search(text)
    has_mcq = len(markers) >= 3 or (len(markers) >= 2 and has_answer)
    if not has_mcq:
        return text, False

    first = markers[0].start() if markers else 0
    if has_answer and markers:
        first = min(markers[0].start(), has_answer.start())
    leading = text[:first].strip()

    # A real sentence leads the MCQ block: keep the prose, drop the MCQ tail.
    if count_alpha_words(leading) >= 3 and _SENTENCE_END_RE.search(leading):
        return leading, True

    # Nothing but the MCQ block (or a hopelessly short lead): drop all of it.
    return "", True


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


def count_alpha_words(text: str) -> int:
    return len([w for w in text.split() if any(c.isalnum() for c in w)])


def _has_real_content(text: str) -> bool:
    return any(c.isalpha() for c in text)


# ---------------------------------------------------------------------------
# Junk classification (paragraph / item level).
# ---------------------------------------------------------------------------


def _repeat_key(text: str) -> str:
    """Canonical key used to recognise headers/footers and exact duplicates.

    Digits become ``#`` and a leading/trailing page number is dropped so a
    running page number does not stop a header being recognised as repeated.
    """
    key = normalize_unicode(text).lower()
    key = re.sub(r"\d+", "#", key)
    key = re.sub(r"\s*[|:·=\-]\s*", " ", key)
    key = re.sub(r"\W+", " ", key)
    key = re.sub(r"\s+#\s*$", "", key)
    key = re.sub(r"^\s*#\s*", "", key)
    return key.strip()


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
            top_counts[_repeat_key(line)] += 1
        for line in texts[-3:]:
            bottom_counts[_repeat_key(line)] += 1

    repeated = {k for k, c in top_counts.items() if c >= threshold}
    repeated |= {k for k, c in bottom_counts.items() if c >= threshold}
    return repeated


def _classify_paragraph(text: str, repeated_keys: set[str]) -> str | None:
    """Return the junk category for a cleaned paragraph, or ``None`` to keep it.

    Deliberately conservative: any doubt about a paragraph being real content
    defaults to keeping it (returns ``None``).
    """
    if not text:
        return CATEGORY_EMPTY

    stripped = text.strip()

    # Header/footer known to repeat across many pages.
    if _repeat_key(text) in repeated_keys:
        return CATEGORY_HEADER_FOOTER

    # Pure page numbers.
    if _PAGE_NUMBER_RE.match(stripped):
        return CATEGORY_PAGE_NUMBER

    # Decorative separators / repeated divider lines / no alphabetic content.
    if _DECORATIVE_RE.match(stripped):
        return CATEGORY_DECORATIVE
    if not _has_real_content(stripped):
        return CATEGORY_DECORATIVE if len(stripped) >= 3 else CATEGORY_EMPTY

    # Copyright / publisher / boilerplate.
    if _BOILERPLATE_RE.search(stripped) or _COPYRIGHT_RE.search(stripped):
        return CATEGORY_COPYRIGHT

    # Watermark / document stamp (only when it is a short standalone phrase).
    if _WATERMARK_RE.search(stripped) and count_alpha_words(stripped) <= 8:
        return CATEGORY_WATERMARK

    # A figure/table/equation caption standing alone.
    if _CAPTION_RE.match(stripped) and count_alpha_words(stripped) <= 14:
        return CATEGORY_FIGURE_CAPTION

    # Long caption: marker + separator (":"/".") whose body does not explain a
    # figure in prose (e.g. "Figure 2.1: The layered architecture ...").
    if (
        _CAPTION_SEPARATOR_RE.match(stripped)
        and not _CAPTION_REFERENCE_RE.match(stripped)
    ):
        return CATEGORY_FIGURE_CAPTION

    # Running chapter/section head that carries a page number.
    if (
        _RUNNING_HEAD_LEADING_RE.match(stripped)
        or _RUNNING_HEAD_TRAILING_RE.match(stripped)
    ) and count_alpha_words(stripped) <= 14:
        return CATEGORY_RUNNING_HEAD

    # Footnote marker + a sentence standing alone.
    if _FOOTNOTE_PARA_RE.match(stripped):
        return CATEGORY_FOOTNOTE

    # Table-of-contents / dotted leaders running to a page number.
    if _TOC_RE.search(stripped):
        return CATEGORY_TOC

    # Reference / bibliography headings are not retrieval content.
    if _BIBLIO_HEADING_RE.match(stripped):
        return CATEGORY_BIBLIOGRAPHY

    # Standalone credit / source attribution ("Source: ACME Corp"). Only when
    # short so a real sentence that merely begins with "Source" is preserved.
    if _CREDIT_SOURCE_RE.match(stripped) and count_alpha_words(stripped) <= 25:
        return CATEGORY_CREDIT

    # A paragraph that is nothing but a URL (or a run of URLs).
    if _URL_ONLY_RE.match(stripped):
        return CATEGORY_URL

    # Clear, exact-shaped diagram label fragments (very specific; safe).
    if _DIAGRAM_FRAGMENT_RE.match(stripped):
        return CATEGORY_DIAGRAM

    return None


def _item_kind(item: dict[str, Any]) -> str:
    kind = item.get("type")
    return kind if isinstance(kind, str) and kind else "text"


# ---------------------------------------------------------------------------
# Page-level cleaning (headers, footers, page numbers, boilerplate, noise).
# ---------------------------------------------------------------------------


def clean_pages_with_stats(
    pages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], CleaningStats]:
    """Clean LlamaParse pages and return a per-category removal report.

    Returns ``(cleaned_pages, stats)``. Keeps only meaningful educational
    content; removes figures/diagram labels, captions, footnotes, headers/
    footers, page numbers, boilerplate, watermarks, TOC/reference rows, OCR
    garbage, near-empty and duplicate paragraphs.
    """
    t0 = time.perf_counter()
    stats = CleaningStats()

    for p in pages:
        for i in page_items(p):
            stats.chars_before += len(item_text(i))

    repeated = _detect_headers_footers(pages)
    cleaned_pages: list[dict[str, Any]] = []
    seen_paragraphs: set[str] = set()

    for page in pages:
        items_out: list[dict[str, Any]] = []
        for item in page_items(page):
            raw = item_text(item)
            if not raw:
                continue
            kind = _item_kind(item)

            base = clean_text(raw)
            stats.paragraphs_processed += 1

            # Structural headings are almost never junk: keep them clean.
            if kind == "heading":
                if base and base not in seen_paragraphs:
                    items_out.append(_fresh_item(item, base))
                    seen_paragraphs.add(base)
                elif base:
                    stats.paragraphs_removed += 1
                    stats.duplicates_removed += 1
                    stats.by_category[CATEGORY_DUPLICATE] += 1
                continue

            category = _classify_paragraph(base, repeated)
            if category is not None:
                stats.paragraphs_removed += 1
                stats.by_category[category] += 1
                continue

            # Multiple-choice answer block attached to a paragraph: strip its
            # tail (or drop the whole block when there is no leading prose).
            stripped, was_mcq = strip_mcq_suffix(base)
            if was_mcq:
                stats.paragraphs_removed += 1
                stats.by_category[CATEGORY_MCQ] += 1
                continue

            # Inline extraction artifacts (replacement chars, control bytes,
            # bare URLs).
            stripped, repl_n = strip_inline_garbage(stripped)
            if not stripped:
                stats.paragraphs_removed += 1
                stats.by_category[CATEGORY_EMPTY] += 1
                continue
            # Paragraph too damaged by mojibake (e.g. "I� APAr") to be useful.
            if repl_n >= 1 and count_alpha_words(stripped) < 6:
                stats.paragraphs_removed += 1
                stats.by_category[CATEGORY_OCR] += 1
                continue

            stripped, n_foot = strip_inline_footnotes(stripped)
            if n_foot:
                stats.inline_footnotes_removed += n_foot
                stripped = stripped.strip()

            if not stripped:
                stats.paragraphs_removed += 1
                stats.by_category[CATEGORY_EMPTY] += 1
                continue

            # Duplicate removal: identical text repeated (PDF extraction artifact).
            if stripped in seen_paragraphs:
                stats.paragraphs_removed += 1
                stats.duplicates_removed += 1
                stats.by_category[CATEGORY_DUPLICATE] += 1
                continue

            items_out.append(_fresh_item(item, stripped))
            seen_paragraphs.add(stripped)

        if items_out:
            cleaned_pages.append({"page": page_number(page), "items": items_out})

    for p in cleaned_pages:
        for i in p["items"]:
            stats.chars_after += len(item_text(i))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Cleaning completed | pages=%d | paragraphs_in=%d | removed=%d "
        "| footnotes=%d | duplicates=%d | removed_pct=%.2f%% | time=%.2fs",
        len(cleaned_pages),
        stats.paragraphs_processed,
        stats.paragraphs_removed,
        stats.inline_footnotes_removed,
        stats.duplicates_removed,
        stats.percent_text_removed,
        elapsed,
    )
    for category, count in stats.by_category.most_common():
        logger.debug("  removed %-22s x%d", category, count)

    return cleaned_pages, stats


def clean_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean LlamaParse pages. Returns only the cleaned page list."""
    cleaned, _ = clean_pages_with_stats(pages)
    return cleaned


def _fresh_item(item: dict[str, Any], value: str) -> dict[str, Any]:
    new_item = dict(item)
    new_item["value"] = value
    if "md" in new_item:
        new_item["md"] = value
    return new_item
