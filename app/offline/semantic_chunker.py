from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from app.config import (
    CHILD_MAX_SIZE,
    CHILD_MIN_TOKENS_DROP,
    CHILD_MIN_TOKENS_MERGE,
    PARENT_MAX_SIZE,
    PARENT_MERGE_TOKENS,
    PARENT_MIN_TOKENS_DROP,
    QUESTION_PARENT_MIN_SHARE,
    SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD_CHILD,
    TITLE_BATCH_SIZE,
    TITLE_CONTEXT_RECENT,
    TITLE_PARALLELISM,
    WORDS_PER_TOKEN,
    FALLBACK_SECTION_MAX_WORDS,
    FALLBACK_SUBSECTION_MAX_WORDS,
)
from app.logging_conf import get_logger
from app.language import arabic_comparison_key, detect_language
from app.offline.embeddings import cosine_sim, dense_vector, device_name
from app.offline.parser_items import item_text, item_type, page_items, page_number
from app.offline.title_generator import (
    descriptive_fallback,
    generate_batch_titles,
    generate_family_batch_titles,
    is_acceptable_title,
    make_titles_unique,
    make_title_client,
    regenerate_title,
)

logger = get_logger("SEMANTIC_CHUNKER")

# Known abbreviations whose period must not start a new sentence.
_ABBREVS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "fig", "vs", "etc", "cm", "mm", "km",
        "kg", "e.g", "i.e", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec", "al", "ed", "ch", "pp", "sec", "approx",
        "cf", "eq", "vol", "no",
    }
)

# Patterns that must not be mistaken for sentence boundaries.
_DECIMAL_RE = re.compile(r"\d+\.\d+")
_FIGURE_REF_RE = re.compile(
    r"\b(Figure|Fig|Table|Equation|Eq)\.?\s*\d+(-?\d+)*\b", re.IGNORECASE
)

# Periods matching these patterns are masked before splitting and restored after.
_MASK = "\x00"


def _resolved_chunk_language(content: str, document_language: str | None) -> str:
    letters = re.findall(rf"[A-Za-z{_ARABIC_LETTER}]", content or "")
    if len(letters) < 12 and document_language in {"ar", "en"}:
        return document_language
    return detect_language(content)


def count_words(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


@dataclass
class Paragraph:
    text: str
    page: int | None = None
    item_type: str = "text"
    heading_level: int | None = None
    order: int = 0
    role: str = "content"
    strong_boundary: bool = False
    parser_item_type: str | None = None
    heading_score: int | None = None
    heading_decision: str | None = None
    heading_reasons: tuple[str, ...] = ()
    atomic_id: str | None = None


@dataclass
class Sentence:
    text: str
    page: int | None = None
    item_type: str = "text"
    heading_level: int | None = None
    atomic_id: str | None = None


@dataclass
class ParentChunk:
    parent_id: str
    document_id: str
    title: str | None
    page_start: int | None
    page_end: int | None
    content: str
    book_heading: str | None = None
    heading_level: int | None = None
    content_role: str = "content"
    source_items: list[Paragraph] | None = None
    protected_boundary: bool = False
    language: str | None = None


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    document_id: str
    title: str | None
    page_start: int | None
    page_end: int | None
    content: str
    book_heading: str | None = None
    heading_level: int | None = None
    content_role: str = "content"
    source_items: list[Paragraph] | None = None
    language: str | None = None


_ARABIC_LETTER = r"\u0621-\u064A\u066E-\u06D3\u06FA-\u06FF"
_SPECIAL_ROLE_RE = re.compile(
    r"^\s*(?:تمرين|تمارين|نشاط|أنشطة|تجربة|اختبر|"
    r"(?:(?:الوحدة|الفصل|القسم)\s*[\d٠-٩]*\s*)?(?:تقويم|تقييم|مراجعة)|"
    r"أسئلة\s+(?:ذات|إجابات|مراجعة)|دليل\s+الدراسة|مراجع|المراجع|"
    r"exercise|activity|review|assessment|evaluation|questions?|"
    r"references?|bibliography)\b",
    re.IGNORECASE,
)

_PARENT_MARKER_RE = re.compile(
    r"^\s*(?:(?:فصل\s*[\d٠-٩]*\s*[:.-]?\s*)?(?:الدرس|درس)\s*[\d٠-٩]+|"
    r"(?:الوحدة|القسم|الفصل|الباب)\s*[\d٠-٩]+|"
    r"[\d٠-٩]+[\s:.-]*(?:الوحدة|القسم|الفصل|الباب|الدرس|درس)|"
    r"(?:chapter|section|unit|lesson|part)\s*\d+)\b",
    re.IGNORECASE,
)
_SECTION_NUMBER_RE = re.compile(
    r"^\s*(?:(?:أولاً|أولًا|ثانياً|ثانيًا|ثالثاً|ثالثًا|رابعاً|رابعًا|خامساً|خامسًا|"
    r"سادساً|سادسًا|سابعاً|سابعًا|ثامناً|ثامنًا|تاسعاً|تاسعًا)\s*[:.)-]|"
    r"\d+(?:\.\d+)*\s*[:.)-]\s*\S)",
    re.IGNORECASE,
)
_QUOTE_INTRO_RE = re.compile(
    r"^\s*(?:قال|يقول|عن\s+\S+|روي\s+عن|رُوي\s+عن|ورد\s+في|نص(?:ت|تّ)|"
    r"according\s+to|said|states?|the\s+(?:law|author|poet)\s+(?:says|states?))\b.*[:：]\s*$",
    re.IGNORECASE,
)
_ATTRIBUTION_RE = re.compile(
    r"^\s*(?:عن\s+.{1,80}(?:قال|رضي\s+الله\s+عنه)|رواه\s+\S+|"
    r"according\s+to\s+.{1,80}|(?:written|said)\s+by\s+.{1,80})\s*[:：]?\s*$",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"^\s*[\[({（]?\s*(?:[\d٠-٩]+\s*)?(?:[\u0600-\u06ffA-Za-z][\u0600-\u06ffA-Za-z\s.-]{0,35})?"
    r"\s*[\d٠-٩:.,-]*\s*[\])}）]?\s*$"
)
_INSTRUCTION_RE = re.compile(
    r"^\s*(?:أبادر|ابادر|أتعلم|اتعلم|نتعلم|أتأمل|اتأمل|أفكر|افكر|أجيب|اجيب|أكمل|اكمل|"
    r"أنظم|انظم|أنشط|انشط|أقيم|اقيم|أقيّم|أضع|اضع|أقرأ|اقرأ|أتلو|اتلو|"
    r"أبحث|ابحث|أستنتج|استنتج|أتعاون|اتعاون|ألاحظ|الاحظ|اقترح|حدد|أحدد|"
    r"ناقش|أناقش|قارن|أقارن|اربط|أربط|اكتب|أكتب|اشرح|أشرح|استخدم|أستخدم|"
    r"أفهم|افهم|إضاءات|اضاءات|أثري|اثري|تقييم\s+ذاتي|التقييم\s+الذاتي|"
    r"read|answer|complete|discuss|compare|explain|identify|choose|activity|exercise)\b",
    re.IGNORECASE,
)
_QUOTE_MARK_RE = re.compile(r"[«»“”„‟\"﴿﴾]")
_HTML_OR_BROKEN_RE = re.compile(r"<[^>]+>|[\[\]{}<>]|�")
_TERMINAL_SENTENCE_RE = re.compile(r"[.!?؟؛]\s*$")


def _heading_level(item: dict[str, Any]) -> int | None:
    value = item.get("lvl", item.get("level"))
    return value if isinstance(value, int) and value > 0 else None


def _content_role(text: str, kind: str) -> str:
    if kind == "table":
        return "table"
    if _SPECIAL_ROLE_RE.search(text):
        return "supplementary"
    return "content"


def _candidate_metadata(item: dict[str, Any]) -> tuple[float | None, bool, float | None]:
    box = item.get("bBox")
    confidence = None
    height = None
    if isinstance(box, dict):
        raw_confidence = box.get("confidence")
        confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else None
        raw_height = box.get("h")
        height = float(raw_height) if isinstance(raw_height, (int, float)) else None
    markdown = str(item.get("md") or "")
    emphasized = "**" in markdown or "<b>" in markdown.lower() or "<u>" in markdown.lower()
    return confidence, emphasized, height


def _looks_like_citation(text: str) -> bool:
    stripped = text.strip()
    if not stripped or count_words(stripped) > 8:
        return False
    if re.fullmatch(r"[\[({（].{1,50}[\])}）]", stripped):
        return True
    if re.search(r"[\d٠-٩]+\s*[:،,.-]\s*[\d٠-٩]+", stripped):
        return True
    # A tiny bracket-damaged reference such as "[6 المائدة".
    return bool(
        _CITATION_RE.fullmatch(stripped)
        and re.search(r"[\[\](){}]|[\d٠-٩]", stripped)
        and count_words(stripped) <= 4
    )


def _looks_ocr_corrupted(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _HTML_OR_BROKEN_RE.search(stripped):
        return True
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return True
    arabic_or_latin = sum(
        bool(re.match(rf"[A-Za-z{_ARABIC_LETTER}]", ch)) for ch in letters
    )
    return arabic_or_latin / len(letters) < 0.9


def _grammatically_continues(text: str, following: str) -> bool:
    stripped = text.strip()
    if not stripped or not following.strip():
        return False
    rule_text = arabic_comparison_key(stripped)
    if _QUOTE_INTRO_RE.match(rule_text) or _ATTRIBUTION_RE.match(rule_text):
        return True
    if stripped.endswith(("،", ",", "؛", ";", "-", "–", "—")):
        return True
    return False


def _score_heading_candidate(
    paragraph: Paragraph,
    previous: Paragraph | None,
    following: Paragraph | None,
    *,
    confidence: float | None,
    emphasized: bool,
    style_count: int,
) -> tuple[int, str, tuple[str, ...]]:
    """Classify a parser-reported heading using deterministic structure only."""
    text = paragraph.text.strip()
    rule_text = arabic_comparison_key(text)
    words = count_words(text)
    score = 1
    reasons: list[str] = ["parser_heading"]

    if _PARENT_MARKER_RE.match(rule_text):
        score += 7
        reasons.append("section_marker")
    if _SECTION_NUMBER_RE.match(rule_text):
        score += 2
        reasons.append("section_numbering")
    if 2 <= words <= 12:
        score += 2
        reasons.append("reasonable_length")
    elif words > 18:
        score -= 3
        reasons.append("long_sentence")
    if emphasized:
        score += 1
        reasons.append("emphasized_style")
    if confidence is not None and confidence >= 0.7:
        score += 1
        reasons.append("layout_confidence")
    elif confidence is not None and confidence < 0.35:
        score -= 1
        reasons.append("low_layout_confidence")
    if style_count >= 2 and (emphasized or confidence is not None):
        score += 1
        reasons.append("consistent_style")
    if following and count_words(following.text) >= 8:
        score += 1
        reasons.append("explanatory_content_follows")
    if previous and _PARENT_MARKER_RE.match(arabic_comparison_key(previous.text)) and not _INSTRUCTION_RE.match(rule_text):
        score += 3
        reasons.append("follows_section_marker")

    if _QUOTE_INTRO_RE.match(rule_text):
        score -= 7
        reasons.append("quotation_introduction")
    elif _ATTRIBUTION_RE.match(rule_text):
        score -= 6
        reasons.append("attribution")
    if _looks_like_citation(rule_text) and not _PARENT_MARKER_RE.match(rule_text):
        score -= 7
        reasons.append("citation")
    if _INSTRUCTION_RE.match(rule_text):
        score -= 5
        reasons.append("instruction_or_exercise")
    if paragraph.role == "supplementary":
        score -= 3
        reasons.append("supplementary_content")
    if _looks_ocr_corrupted(text):
        score -= 3
        reasons.append("ocr_or_markup_noise")
    if _grammatically_continues(text, following.text if following else ""):
        score -= 3
        reasons.append("continues_into_next_block")
    if text.endswith((":", "：")) and ("," in text or "،" in text):
        score -= 3
        reasons.append("activity_prompt_shape")
    if _TERMINAL_SENTENCE_RE.search(text) and words >= 8:
        score -= 2
        reasons.append("complete_sentence")

    if _PARENT_MARKER_RE.match(rule_text) and score >= 5:
        decision = "parent_heading"
    elif score >= 4:
        decision = "subsection_heading"
    else:
        decision = "content"
    return score, decision, tuple(reasons)


def _validate_heading_candidates(
    paragraphs: list[Paragraph], metadata: dict[int, tuple[float | None, bool, float | None]]
) -> None:
    """Turn parser headings into validated structural headings or normal text."""
    style_counts: Counter[tuple[int | None, bool, int | None]] = Counter()
    for paragraph in paragraphs:
        if paragraph.parser_item_type != "heading":
            continue
        confidence, emphasized, height = metadata.get(paragraph.order, (None, False, None))
        height_bucket = round(height / 5) if height else None
        style_counts[(paragraph.heading_level, emphasized, height_bucket)] += 1

    for index, paragraph in enumerate(paragraphs):
        if paragraph.parser_item_type != "heading":
            continue
        following = next(
            (candidate for candidate in paragraphs[index + 1 :] if candidate.text.strip()),
            None,
        )
        previous = next(
            (candidate for candidate in reversed(paragraphs[:index]) if candidate.text.strip()),
            None,
        )
        confidence, emphasized, height = metadata.get(paragraph.order, (None, False, None))
        height_bucket = round(height / 5) if height else None
        score, decision, reasons = _score_heading_candidate(
            paragraph,
            previous,
            following,
            confidence=confidence,
            emphasized=emphasized,
            style_count=style_counts[(paragraph.heading_level, emphasized, height_bucket)],
        )
        paragraph.heading_score = score
        paragraph.heading_decision = decision
        paragraph.heading_reasons = reasons
        paragraph.item_type = "heading" if decision != "content" else "text"
        paragraph.strong_boundary = decision == "parent_heading"


def _assign_atomic_quote_units(paragraphs: list[Paragraph]) -> None:
    """Protect adjacent quote introductions, quotations, citations and explanations."""
    for index, paragraph in enumerate(paragraphs):
        text = paragraph.text.strip()
        rule_text = arabic_comparison_key(text)
        starts_unit = bool(_QUOTE_INTRO_RE.match(rule_text) or _ATTRIBUTION_RE.match(rule_text))
        quote_like = bool(_QUOTE_MARK_RE.search(text))
        if not starts_unit and not (quote_like and index + 1 < len(paragraphs)):
            continue
        atomic_id = paragraph.atomic_id or f"quote-{paragraph.order}"
        paragraph.atomic_id = atomic_id
        word_budget = count_words(paragraph.text)
        # Attach the quotation and one immediate explanatory/citation block.
        for next_index in range(index + 1, min(len(paragraphs), index + 4)):
            candidate = paragraphs[next_index]
            if candidate.strong_boundary:
                break
            if next_index > index + 1 and candidate.item_type == "heading":
                break
            candidate_words = count_words(candidate.text)
            if word_budget + candidate_words > _max_child_words():
                break
            candidate.atomic_id = atomic_id
            word_budget += candidate_words
            if next_index >= index + 2 and not _looks_like_citation(candidate.text):
                break

    # A trailing citation belongs to the immediately preceding quotation unit.
    for index in range(1, len(paragraphs)):
        if _looks_like_citation(paragraphs[index].text) and paragraphs[index - 1].atomic_id:
            paragraphs[index].atomic_id = paragraphs[index - 1].atomic_id


# ---------------------------------------------------------------------------
# Stage 1: Build the document's paragraph units (parser-agnostic).
# ---------------------------------------------------------------------------


def extract_paragraphs(pages: list[dict[str, Any]]) -> list[Paragraph]:
    """Return the document body as ordered, page-tagged paragraphs.

    Every non-empty LlamaParse item (including headings) becomes one paragraph
    unit. Headings are kept so their text remains as a natural section boundary
    and can be reused as the title; the title prompt is told to lean on a
    meaningful heading and only write a new one when it is missing or bad.
    Parent chunking then merges consecutive paragraphs purely by similarity.
    """
    paragraphs: list[Paragraph] = []
    candidate_metadata: dict[int, tuple[float | None, bool, float | None]] = {}
    order = 0
    for page in pages:
        page_num = page_number(page)
        for item in page_items(page):
            text = item_text(item)
            if text:
                kind = item_type(item)
                paragraph = Paragraph(
                    text=text,
                    page=page_num,
                    item_type="heading_candidate" if kind == "heading" else kind,
                    heading_level=_heading_level(item),
                    order=order,
                    role=_content_role(text, kind),
                    strong_boundary=False,
                    parser_item_type=kind,
                )
                paragraphs.append(paragraph)
                if kind == "heading":
                    candidate_metadata[order] = _candidate_metadata(item)
                order += 1
    _validate_heading_candidates(paragraphs, candidate_metadata)
    _assign_atomic_quote_units(paragraphs)
    return paragraphs


def split_sentence_stream(pages: list[dict[str, Any]]) -> list[Sentence]:
    """Document body as ordered, page-tagged sentences (document order)."""
    sentences: list[Sentence] = []
    for page in pages:
        page_num = page_number(page)
        for item in page_items(page):
            text = item_text(item)
            if text:
                for piece in split_sentences(text):
                    if piece:
                        sentences.append(Sentence(text=piece, page=page_num))
    return sentences


def split_sentences(text: str) -> list[str]:
    """Split Arabic, English and mixed text into semantic sentence/list units.

    Guards against known abbreviations (``Dr.``), decimals (``4.91``), and
    figure/equation references (``Figure 1-1.``) that should not end a sentence.
    """
    if not text:
        return []
    text = _mask_non_boundary_periods(text)
    # Preserve parser line boundaries when they introduce a bullet, numbered
    # question, or short heading without terminal punctuation.
    text = re.sub(
        r"\s*\n\s*(?=(?:[-*•–—]\s+|[\(\[]?[0-9٠-٩]{1,3}[.)\]:-]\s+))",
        "\n",
        text,
    )
    text = re.sub(r"[ \t]+", " ", text).strip()
    starter = rf"(?=[A-Z0-9{_ARABIC_LETTER}\"'\(\[])"
    parts = re.split(rf"(?<=[.!?؟؛])\s+{starter}|\n+", text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Re-join if the last token of the previous accumulated sentence is an
        # abbreviation: its period was not a real sentence boundary.
        if out and _last_token(out[-1]).lower() in _ABBREVS:
            out[-1] = f"{out[-1]} {part}"
            continue
        out.append(part)
    return [_restore_mask(s) for s in out]


def _mask_non_boundary_periods(text: str) -> str:
    """Hide periods inside decimals and figure/table references so the splitter
    never treats them as sentence boundaries."""
    masked = _DECIMAL_RE.sub(lambda m: m.group(0).replace(".", _MASK), text)
    return _FIGURE_REF_RE.sub(
        lambda m: m.group(0).replace(".", _MASK), masked
    )


def _restore_mask(sentence: str) -> str:
    return sentence.replace(_MASK, ".")


def _last_token(text: str) -> str:
    tokens = [
        t for t in text.split()
        if t and t not in ('"', "'", "(", ")", "[", "]", "``", "''", "`")
    ]
    if not tokens:
        return ""
    return tokens[-1].rstrip(".")


# ---------------------------------------------------------------------------
# Stage 2: Parent chunks (paragraph embedding similarity walk).
# ---------------------------------------------------------------------------


def split_parents(
    pages: list[dict[str, Any]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[list[Paragraph]]:
    """Group consecutive paragraphs into parents by embedding similarity.

    Each paragraph is embedded exactly once. Walking the stream in order,
    paragraph i+1 joins the current parent while ``cosine_sim(vec[i], vec[i+1])
    >= threshold``; on a drop below threshold the current parent is closed and
    a new one starts. Parents are disjoint (zero overlap). The similarity at
    every boundary decision is logged.
    """
    paragraphs = extract_paragraphs(pages)
    if not paragraphs:
        return []

    t0 = time.perf_counter()
    vecs = dense_vector([p.text for p in paragraphs])
    encode_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    # Only headings that passed deterministic candidate validation may own a
    # parent boundary. Documents without reliable structural markers retain the
    # semantic fallback; parser-reported level alone is never sufficient.
    structured = sum(1 for p in paragraphs if p.strong_boundary) >= 2
    groups: list[list[Paragraph]] = [[paragraphs[0]]]
    for i in range(1, len(paragraphs)):
        sim = cosine_sim(vecs[i - 1], vecs[i])
        current = paragraphs[i]
        previous = paragraphs[i - 1]
        structural = current.strong_boundary
        if structural:
            marker_re = re.compile(
                r"^\s*(الوحدة|القسم|الفصل|الباب|chapter|section|unit|part)\s*([\d٠-٩]+)",
                re.IGNORECASE,
            )
            current_marker = marker_re.match(current.text)
            previous_marker = next(
                (
                    marker_re.match(unit.text)
                    for unit in reversed(groups[-1])
                    if unit.item_type == "heading" and marker_re.match(unit.text)
                ),
                None,
            )
            if (
                current_marker
                and previous_marker
                and current_marker.group(1).lower() == previous_marker.group(1).lower()
                and current_marker.group(2) == previous_marker.group(2)
            ):
                structural = False
        role_boundary = False
        same_atomic_unit = bool(
            current.atomic_id
            and previous.atomic_id
            and current.atomic_id == previous.atomic_id
        )
        split = structural or (not structured and sim < threshold)
        if same_atomic_unit:
            split = False
            structural = False
        reason = (
            "atomic_quote" if same_atomic_unit else "heading" if structural else "content_role" if role_boundary
            else "similarity" if split else "merge"
        )
        logger.info(
            "Parent boundary | i=%d | sim=%.4f | threshold=%.2f | action=%s | reason=%s",
            i,
            sim,
            threshold,
            "merge" if not split else "split",
            reason,
        )
        if split:
            groups.append([paragraphs[i]])
        else:
            groups[-1].append(paragraphs[i])

    elapsed = time.perf_counter() - t0
    logger.info(
        "Parent split | paragraphs=%d | parents=%d | embed_time=%.0fms | sim_time=%.2fs",
        len(paragraphs),
        len(groups),
        encode_ms,
        elapsed,
    )
    return groups


def _paragraph_pages(group: list[Paragraph]) -> tuple[int | None, int | None]:
    pages = [p.page for p in group if p.page is not None]
    return (min(pages) if pages else None, max(pages) if pages else None)


def _group_content(group: list[Paragraph]) -> str:
    return " ".join(p.text for p in group).strip()


def _group_heading(group: list[Paragraph]) -> tuple[str | None, int | None]:
    headings = [unit for unit in group if unit.item_type == "heading"]
    generic = re.compile(
        r"^\s*(?:(?:الوحدة|القسم|الفصل|الباب|الدرس|درس|chapter|section|unit|lesson|part)\s*[\d٠-٩]+|"
        r"[\d٠-٩]+\s*(?:الوحدة|القسم|الفصل|الباب|الدرس|درس))\s*$",
        re.IGNORECASE,
    )
    for unit in headings:
        if (
            not generic.match(unit.text)
            and not _PARENT_MARKER_RE.match(unit.text)
            and not re.match(r"^\s*(?:الشكل|صورة|جدول|figure|table)\b", unit.text, re.IGNORECASE)
            and count_words(unit.text) <= 15
            and unit.role == "content"
        ):
            return unit.text, unit.heading_level
    for unit in headings:
        return unit.text, unit.heading_level
    return None, None


def _group_role(group: list[Paragraph]) -> str:
    if group and group[0].role != "content":
        return group[0].role
    if group and group[0].item_type == "table":
        return "table"
    return "content"


def _can_merge_parents(left: ParentChunk, right: ParentChunk) -> bool:
    """Short chunks may merge only within one unprotected content region."""
    if right.protected_boundary:
        return False
    if left.content_role != right.content_role:
        return False
    if left.content_role in {"supplementary", "table"}:
        return False
    if not left.content or not right.content:
        return False
    # The initial semantic walk already embedded every paragraph. Avoid a
    # second per-parent model call here; require strong lexical continuity as a
    # conservative additional gate for short-fragment merging.
    def terms(value: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(rf"[A-Za-z{_ARABIC_LETTER}]{{3,}}", value)
        }

    a = terms(left.content[-1200:])
    b = terms(right.content[:1200])
    return bool(a and b and len(a & b) / min(len(a), len(b)) >= 0.35)


def _merged_parent(left: ParentChunk, right: ParentChunk) -> ParentChunk:
    units = list(left.source_items or []) + list(right.source_items or [])
    page_start, page_end = _paragraph_pages(units) if units else _page_range(left, right)
    return ParentChunk(
        parent_id=left.parent_id,
        document_id=left.document_id,
        title=left.title,
        page_start=page_start,
        page_end=page_end,
        content=f"{left.content} {right.content}".strip(),
        book_heading=left.book_heading or right.book_heading,
        heading_level=left.heading_level or right.heading_level,
        content_role=left.content_role,
        source_items=units or None,
        protected_boundary=left.protected_boundary,
        language=left.language or right.language,
    )


def _page_range(*parents: ParentChunk) -> tuple[int | None, int | None]:
    pages = [p.page_start for p in parents if p.page_start is not None]
    pages += [p.page_end for p in parents if p.page_end is not None]
    return (min(pages) if pages else None, max(pages) if pages else None)


def _apply_parent_policy(parents: list[ParentChunk]) -> list[ParentChunk]:
    """Enforce parent size bounds (drop / merge / hard ceiling).

    In document order:
    - drop parents at or below PARENT_MIN_TOKENS_DROP,
    - merge parents under PARENT_MERGE_TOKENS into the previous parent (or the
      next one when no previous exists),
    - split any parent above PARENT_MAX_SIZE at sentence boundaries.

    ``count_tokens`` estimates token counts from words via WORDS_PER_TOKEN.
    """
    # 1. Keep structural/supplementary units even when short; only discard
    # genuinely unstructured fragments below the floor.
    kept = [
        p for p in parents
        if count_tokens(p.content) > PARENT_MIN_TOKENS_DROP
        or p.protected_boundary
        or p.content_role in {"supplementary", "table"}
    ]
    if not kept:
        return []

    # 2. Merge short parents into a neighbour (prefer the previous).
    merged: list[ParentChunk] = []
    for parent in kept:
        if (
            merged
            and count_tokens(parent.content) < PARENT_MERGE_TOKENS
            and _can_merge_parents(merged[-1], parent)
        ):
            merged[-1] = _merged_parent(merged[-1], parent)
            continue
        merged.append(parent)

    # 3. Split any parent that exceeds the hard ceiling at sentence boundaries.
    max_parent_words = max(1, int(PARENT_MAX_SIZE / WORDS_PER_TOKEN))
    final: list[ParentChunk] = []
    for parent in merged:
        if count_words(parent.content) <= max_parent_words:
            final.append(parent)
            continue
        units: list[Paragraph] = []
        for unit in parent.source_items or [Paragraph(parent.content, parent.page_start)]:
            if count_words(unit.text) <= max_parent_words:
                units.append(unit)
                continue
            for piece in split_sentences(unit.text):
                # An OCR paragraph can still contain no punctuation. Fall back
                # to a bounded word slice while retaining its exact page.
                words = piece.split()
                for start in range(0, len(words), max_parent_words):
                    units.append(
                        Paragraph(
                            " ".join(words[start : start + max_parent_words]),
                            unit.page,
                            unit.item_type,
                            unit.heading_level,
                            unit.order,
                            unit.role,
                            unit.strong_boundary and start == 0,
                        )
                    )

        packed: list[list[Paragraph]] = []
        current_units: list[Paragraph] = []
        current_words = 0
        for unit in units:
            words = count_words(unit.text)
            if current_units and current_words + words > max_parent_words:
                packed.append(current_units)
                current_units, current_words = [], 0
            current_units.append(unit)
            current_words += words
        if current_units:
            packed.append(current_units)

        # Greedy packing can leave a tiny exercise/citation tail as its own
        # parent (for example 760 words + a final 30-word activity). Rebalance
        # source units from the previous pack so the tail has enough context to
        # remain meaningful, without crossing a structural boundary or cap.
        if len(packed) >= 2:
            tail_words = sum(count_words(unit.text) for unit in packed[-1])
            minimum_tail_words = max(1, int(PARENT_MERGE_TOKENS / WORDS_PER_TOKEN))
            while tail_words < minimum_tail_words and len(packed[-2]) > 1:
                moving = packed[-2][-1]
                if moving.strong_boundary:
                    break
                packed[-2].pop()
                packed[-1].insert(0, moving)
                tail_words += count_words(moving.text)

        for index, part in enumerate(packed):
            page_start, page_end = _paragraph_pages(part)
            final.append(
                ParentChunk(
                    parent_id=str(uuid.uuid4()),
                    document_id=parent.document_id,
                    title=None,
                    page_start=page_start,
                    page_end=page_end,
                    content=_group_content(part),
                    book_heading=parent.book_heading,
                    heading_level=parent.heading_level,
                    content_role=parent.content_role,
                    source_items=part,
                    protected_boundary=parent.protected_boundary and index == 0,
                    language=parent.language,
                )
            )
    return final


def build_parents(
    groups: list[list[Paragraph]], document_id: str
) -> list[ParentChunk]:
    """Turn paragraph groups into parent chunks (no overlap between parents).

    Size policy applied afterwards: drop < PARENT_MIN_TOKENS_DROP, merge
    < PARENT_MERGE_TOKENS, and cap each parent at PARENT_MAX_SIZE.
    """
    parents: list[ParentChunk] = []
    for group in groups:
        if not group:
            continue
        content = _group_content(group)
        if not content:
            continue
        page_start, page_end = _paragraph_pages(group)
        heading, heading_level = _group_heading(group)
        parents.append(
            ParentChunk(
                parent_id=str(uuid.uuid4()),
                document_id=document_id,
                title=None,
                page_start=page_start,
                page_end=page_end,
                content=content,
                book_heading=heading,
                heading_level=heading_level,
                content_role=_group_role(group),
                source_items=list(group),
                protected_boundary=bool(group[0].strong_boundary),
                language=detect_language(content),
            )
        )
    return _apply_parent_policy(parents)


# ---------------------------------------------------------------------------
# LLM titles (delegated to app.offline.title_generator).
# ---------------------------------------------------------------------------


def _run_batches(chunks, *, level: str) -> None:
    """Label sibling chunks in batches so the model sees neighbors.

    Chunks are grouped in document order (``TITLE_BATCH_SIZE`` each) and the
    batches run in parallel. Each batch's prompt includes the titles produced
    by already-finished batches, so sibling awareness is preserved without
    paying the cost of one LLM call per chunk. Results are applied in document
    order regardless of which batch finishes first.
    """
    if not chunks:
        return
    batches = [
        chunks[i : i + TITLE_BATCH_SIZE]
        for i in range(0, len(chunks), TITLE_BATCH_SIZE)
    ]
    assigned: list[tuple] = []
    lock = Lock()

    def do(batch):
        contents = [getattr(c, "content") for c in batch]
        with lock:
            before = [t for _, t in assigned]
        titles = generate_batch_titles(contents, level=level, before=before)
        with lock:
            for child, title in zip(batch, titles):
                child.title = title
                assigned.append((child, title))
        return batch, titles

    with ThreadPoolExecutor(max_workers=TITLE_PARALLELISM) as ex:
        futures = [ex.submit(do, b) for b in batches]
        for batch, titles in [f.result() for f in futures]:
            for child, title in zip(batch, titles):
                child.title = title

    # Global dedup once across the whole run. Parallel batches only see the
    # titles of completed batches, so two batches can still collide on the same
    # concept ("Machine Learning Definition", "Data Preprocessing" twice).
    # Re-apply uniqueness over every title so each is distinct document-wide.
    max_words = (
        FALLBACK_SECTION_MAX_WORDS if level == "section" else FALLBACK_SUBSECTION_MAX_WORDS
    )
    contents = [getattr(c, "content") for c in chunks]
    titles = [c.title for c in chunks]
    final = make_titles_unique(titles, contents, max_words)
    for child, title in zip(chunks, final):
        child.title = title

    for child in chunks:
        cid = getattr(child, "child_id", None) or getattr(child, "parent_id", "")
        logger.info(
            "%s title | id=%s | words=%d | title=%s",
            level.title(),
            cid,
            count_words(child.content),
            child.title,
        )


def generate_section_titles(parents: list[ParentChunk], **kwargs) -> None:
    _run_batches(parents, level="section")


def generate_subsection_titles(children: list[ChildChunk], **kwargs) -> None:
    _run_batches(children, level="subsection")


# ---------------------------------------------------------------------------
# Stage 3: Child chunks within a parent (sentence packing).
# ---------------------------------------------------------------------------


def _max_child_words() -> int:
    return max(1, int(CHILD_MAX_SIZE / WORDS_PER_TOKEN))


def _last_sentence_of(content: str) -> str:
    sents = split_sentences(content)
    return sents[-1] if sents else ""


def count_tokens(text: str) -> int:
    """Approximate token count from the word count via WORDS_PER_TOKEN."""
    return int(count_words(text) * WORDS_PER_TOKEN)


# Stray diagram/axis labels that must never survive as a chunk sentence.
_JUNK_LABEL_RE = re.compile(
    r"^\s*(?:feature\s*(?:\d+\s*)?\??|value\s*(?:value|feature|example)?\??|"
    r"class\??|input\s*\??|output\s*\??|data\s*\??|training\s+set|"
    r"new\s+instance|launch\s*!?\s*|update\s+data|analyze\s+errors|"
    r"can\s+be\s+automated|study\s+the\s+problem|evaluate\s*(?:solution)?\s*|"
    r"this\s+feature\s+based\s*\??)\s*\.?\s*$",
    re.IGNORECASE,
)

# Orphan remainders left when an image/figure label was dropped, e.g.
# "... feature extraction is called." or a lone trailing verb.
_ORPHAN_TAIL_RE = re.compile(
    r"^\s*[\w\s'-]{0,40}\b(?:is|are|was|were|been|has|have|uses|used|called|"
    r"shown|pictured|depicted)\s*\.?\s*$",
    re.IGNORECASE,
)

_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "but",
        "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
        "that", "then", "here", "there", "by", "with", "from", "as", "at",
        "into", "which", "where", "when", "how", "why", "who", "whose", "what",
        "any", "all", "both", "each", "some", "other", "more", "most", "such",
        "than", "also", "very", "just", "only", "often", "usually", "maybe",
    }
)


def _content_words(sentence: str) -> list[str]:
    return [w for w in sentence.split() if _strip_punct_word(w).lower() not in _FUNCTION_WORDS]


def _strip_punct_word(token: str) -> str:
    return re.sub(r"[\W_]+", "", token, flags=re.UNICODE)


def _normalize_sentence(sentence: str) -> str:
    return re.sub(r"[\W_]+", "", sentence.lower(), flags=re.UNICODE)


def _is_junk_sentence(sentence: str) -> bool:
    """True when a sentence is a stray diagram label or a broken fragment."""
    stripped = sentence.strip()
    if not stripped:
        return True
    # Single stray OCR letter glued onto a fragment (e.g. "S In machine ...").
    if re.match(r"^[A-Za-z]\s+[A-Z]", stripped):
        return True
    # Fewer than two words -> too short to be meaningful ("Want", "by").
    words = stripped.split()
    if len(words) < 2:
        return True
    # Axis / diagram labels standing alone.
    if _JUNK_LABEL_RE.match(stripped):
        return True
    if _ORPHAN_TAIL_RE.match(stripped):
        return True
    # Nothing but connecting/leftover words ("feature extraction is called.").
    if not _content_words(stripped):
        return True
    return False


# Structural words that introduce a number that MUST be kept after them
# (e.g. "Chapter 3", "Section 2.1", "Unit 5").
_KEEP_STRUCTURAL_NUMBER_WORDS = frozenset(
    {
        "chapter", "section", "unit", "lesson", "appendix", "part", "module",
        "q", "question", "lesson", "table", "figure", "example", "exercise",
        "step", "row", "column", "item",
    }
)

# Plain numbered-list markers such as "1.", "2)" that carry no meaning and
# should be stripped before embedding. Structural numbers ("Chapter 3",
# "Section 2.1") and year-like numbers ("in 2020.") are protected.
_LIST_MARKER_RE = re.compile(r"(?<![\d٠-٩])([\d٠-٩]{1,3})\s*[.)\]:-]\s+")

# Plain bullet/list markers ("- X", "• X", "– X", "— X", "* X") that carry no
# meaning and should be dropped while keeping the list content that follows.
# A bullet must stand on its own as a leading marker (preceded by text start or
# whitespace, followed by whitespace then non-space) so intra-word hyphens in
# "state-of-the-art" and spaced dashes used inside ordinary prose remain unless
# they clearly separate a list item. Captures the leading boundary so the
# preceding whitespace/paren is preserved, not the bullet symbol.
_BULLET_MARKER_RE = re.compile(r"(^|[\s(])([-•–—*])(?:\s+)(?=\S)")


def strip_list_markers(text: str) -> str:
    """Remove plain numbered-list markers (``1. `` / ``2) ``) and leading bullets.

    A numbered marker is removed only when its number is small (1-2 digits), is
    NOT a structural number ("Chapter 3", "Section 2.1", ...), and is not a
    year-like larger number. The marker's number is dropped but the copied text
    remains. A leading bullet symbol (``-``, ``•``, ``–``, ``—``, ``*``) is
    removed while its content stays.
    """
    if not text:
        return text

    # Strip leading bullet markers, preserving the content and the leading
    # boundary (start of text or the preceding whitespace/paren).
    text = _BULLET_MARKER_RE.sub(lambda m: m.group(1), text)

    # Guard against decimals/versions like "2.1" and "3.5" by requiring the
    # marker not be preceded by a digit or a ".", and not be part of a larger
    # number the structural token list protects.

    def _replace(match: re.Match) -> str:
        # Determine the word right before the match; if structural, keep it.
        prefix = text[: match.start()]
        prev_word = ""
        if prefix.strip():
            toks = [t for t in prefix.split() if t not in ('(', '[')]
            if toks:
                prev_word = toks[-1].rstrip('.')
        if prev_word.lower() in _KEEP_STRUCTURAL_NUMBER_WORDS:
            return match.group(0)  # keep "Chapter 3" intact
        return ""  # strip the bare list marker

    return _LIST_MARKER_RE.sub(_replace, text)


def _is_question_parent(content: str) -> bool:
    """True when most of a parent's sentences look like numbered questions."""
    sents = split_sentences(content)
    if not sents:
        return False
    q = sum(1 for s in sents if _is_question_sentence(s))
    return q / len(sents) >= QUESTION_PARENT_MIN_SHARE


def _is_question_sentence(sent: str) -> bool:
    """A numbered question: starts with a list marker and ends with "?"."""
    stripped = sent.strip()
    if not _LIST_MARKER_RE.match(stripped):
        return False
    return stripped.rstrip().endswith(("?", "؟"))


# Short-but-meaningful units that must never be silently dropped: definitions,
# call-outs, structural headers and questions.
_MEANINGFUL_LABEL_RE = re.compile(
    r"^\s*(?:definition|note|example|key\s+point|important|remember|warning|"
    r"tip|exercise|class\s+exercise|definition\s*\d*)\b[:\-]?",
    re.IGNORECASE,
)
_MEANINGFUL_HEADING_RE = re.compile(
    r"^(chapter|section|part|appendix|module|lesson|unit)\b|\d+(\.\d+)*\s+\S",
    re.IGNORECASE,
)


def _meaningful_fragment(text: str) -> bool:
    """True when a short text carries standalone meaning and should be kept.

    Used to protect legitimate meaningful chunks from the drop/merge floor:
    a definition, call-out, structural heading or question is preserved (merged
    into a neighbour) rather than thrown away as a tiny fragment.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.rstrip().endswith("?"):
        return True
    if _MEANINGFUL_HEADING_RE.match(stripped):
        return True
    if _MEANINGFUL_LABEL_RE.match(stripped):
        return True
    return len(_content_words(stripped)) >= 2


def _apply_min_floor(
    groups: list[str], max_child_words: int, protected_prefixes: set[str] | None = None
) -> list[str]:
    """Enforce the minimum child size on already-packed groups.

    - at or below CHILD_MIN_TOKENS_DROP   -> drop the group entirely, UNLESS it
      is a legitimate meaningful fragment (definition / question / heading),
      which is preserved (merged into the next chunk) instead of discarded;
    - between DROP and CHILD_MIN_TOKENS_MERGE -> merge into the previous (or
      leading) chunk only when the merged chunk stays within ``max_child_words``;
      otherwise the fragment keeps its own chunk so the size cap is never
      exceeded by a merge.
    Returns the filtered, still ordered list of chunk strings.
    """
    if not groups:
        return []
    protected_prefixes = protected_prefixes or set()
    out: list[str] = []
    for g in groups:
        tk = count_tokens(g)
        protected = any(g.startswith(prefix) for prefix in protected_prefixes)
        # Genuine junk at or below the drop floor is discarded; meaningful
        # fragments (definitions / questions / headings) always survive.
        if tk <= CHILD_MIN_TOKENS_DROP and not protected and not _meaningful_fragment(g):
            continue
        # Anything else under the merge floor is folded into the previous chunk,
        # but never past the size ceiling.
        if (
            tk <= CHILD_MIN_TOKENS_MERGE
            and out
            and not protected
            and count_words(out[-1]) + count_words(g) <= max_child_words
        ):
            out[-1] = f"{out[-1]} {g}".strip()
            continue
        out.append(g)
    # Leading tiny single group under the merge floor: fold into next if it fits.
    if (
        len(out) >= 2
        and count_tokens(out[0]) <= CHILD_MIN_TOKENS_MERGE
        and count_words(out[0]) + count_words(out[1]) <= max_child_words
    ):
        out[1] = f"{out[0]} {out[1]}".strip()
        out.pop(0)
    # Any remaining heading-only fragment belongs with the content that
    # follows it, never as a standalone retrieval chunk.
    folded: list[str] = []
    i = 0
    while i < len(out):
        current = out[i]
        protected = any(current.startswith(prefix) for prefix in protected_prefixes)
        if (
            protected
            and count_words(current) <= 12
            and i + 1 < len(out)
            and count_words(current) + count_words(out[i + 1]) <= max_child_words
        ):
            folded.append(f"{current} {out[i + 1]}".strip())
            i += 2
            continue
        folded.append(current)
        i += 1
    out = folded
    return out


def _dispatch_pages(
    bodies: list[str], start: int | None, end: int | None
) -> list[tuple[int, int]]:
    """Spread a parent's page range across its children by word share.

    Children currently inheriting the full parent span (e.g. all ``3-10``) makes
    navigation useless. This assigns each child a proportional slice of the
    parent's real range so adjacent children get distinct, ordered pages.
    """
    if not bodies:
        return []
    if start is None:
        start = 1
    if end is None or end < start:
        end = start
    total = sum(count_words(b) for b in bodies) or 1
    span = end - start
    out: list[tuple[int, int]] = []
    cum = 0.0
    for body in bodies:
        frac = count_words(body) / total
        s_page = start + round(cum)
        cum += frac * span
        e_page = start + round(cum)
        s_page = max(start, min(end, s_page))
        e_page = max(s_page, min(end, e_page))
        out.append((s_page, e_page))
    return out


def _prepare_child_sentences(text: str) -> list[str]:
    """Return the exact sentence stream ``build_children`` will pack.

    Applies the same cleaning the child splitter uses: strips plain numbered-list
    markers, drops junk/broken fragments, and removes sentence duplicates. This is
    factored out so the whole book's sentences can be embedded exactly once,
    up-front, and then aligned to each parent's children by index.
    """
    combined = strip_list_markers(text)
    if not combined:
        return []
    selected: list[str] = []
    for s in split_sentences(combined):
        if _is_junk_sentence(s):
            continue
        norm = _normalize_sentence(s)
        if selected and _normalize_sentence(selected[-1]) == norm:
            continue
        selected.append(s.strip())
    return selected


def _prepare_child_sentence_units(parent: ParentChunk) -> list[Sentence]:
    """Prepare child sentences without losing their parser page provenance."""
    units = parent.source_items or [Paragraph(parent.content, parent.page_start)]
    selected: list[Sentence] = []
    for unit in units:
        cleaned = strip_list_markers(unit.text)
        if not cleaned:
            continue
        for piece in split_sentences(cleaned):
            if _is_junk_sentence(piece):
                continue
            norm = _normalize_sentence(piece)
            if selected and _normalize_sentence(selected[-1].text) == norm:
                continue
            selected.append(
                Sentence(piece.strip(), unit.page, unit.item_type, unit.heading_level, unit.atomic_id)
            )
    return selected


def _is_child_heading(sentence: Sentence) -> bool:
    if sentence.item_type != "heading" or count_words(sentence.text) > 15:
        return False
    return not bool(
        re.match(
            r"^\s*(?:الشكل|صورة|جدول|figure|fig\.?|table)\b",
            sentence.text,
            re.IGNORECASE,
        )
    )


def preembed_sentences(parents: list[ParentChunk]) -> dict[str, list[list[float]]]:
    """Embed every child sentence of the whole book in one batched pass.

    Returns ``{parent_id: [sentence_vec, ...]}`` aligned to the sentence order
    produced by ``_prepare_child_sentences`` for that parent. This replaces the
    per-sentence model calls in ``build_children`` with a single, batched
    forward pass, then the child splitter compares precomputed vectors only.
    """
    pairs: list[tuple[str, str]] = []
    for p in parents:
        if _is_question_parent(p.content):
            continue
        for s in _prepare_child_sentence_units(p):
            pairs.append((p.parent_id, s.text))
    if not pairs:
        return {}
    texts = [s for _, s in pairs]
    t0 = time.perf_counter()
    vecs = dense_vector(texts)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Sentence pre-embed | sentences=%d | vectors=%d | time=%.2fs | model=%s",
        len(texts),
        len(vecs),
        elapsed,
        device_name(),
    )
    result: dict[str, list[list[float]]] = {}
    for (pid, _), vec in zip(pairs, vecs):
        result.setdefault(pid, []).append(vec)
    return result


def build_children(
    parent: ParentChunk, sentence_vecs: list[list[float]] | None = None
) -> list[ChildChunk]:
    """Split one parent's content into focused child chunks (sentence packing).

    The sentence stream is walked in order, building a single current chunk.
    Before any merge the candidate (current chunk + next sentence) is checked
    against the size ceiling; if it exceeds the cap the current chunk is
    finalized and the next sentence starts a new chunk. Otherwise the similarity
    between the current chunk's running sentence centroid and the next sentence
    is compared: it joins while ``cosine_sim(centroid, vec(next)) >=
    SIMILARITY_THRESHOLD_CHILD``; otherwise the current chunk is finalized and
    the next sentence starts a new chunk. ``CHILD_MAX_SIZE`` remains a hard
    ceiling that closes the chunk on size regardless of similarity. A child is
    always cut at a sentence boundary, and between any two adjacent children
    exactly the last sentence of the previous child is carried forward as
    overlap.
    """
    text = parent.content
    if not text:
        return []

    # Question-only parents are kept whole: strip nothing harmful, keep them
    # as one atomic chunk so a short question is never split into a tiny orphan.
    if _is_question_parent(text):
        logger.info(
            "Question parent | parent_id=%s | kept whole (atomic child)",
            parent.parent_id,
        )
        return [
            ChildChunk(
                child_id=str(uuid.uuid4()),
                parent_id=parent.parent_id,
                document_id=parent.document_id,
                title=None,
                page_start=parent.page_start,
                page_end=parent.page_end,
                content=text.strip(),
                book_heading=parent.book_heading,
                heading_level=parent.heading_level,
                content_role=parent.content_role,
                source_items=parent.source_items,
                language=parent.language or detect_language(text),
            )
        ]

    child_sents = _prepare_child_sentence_units(parent)
    if not child_sents:
        return []

    max_child_words = _max_child_words()
    current_chunk = child_sents[0].text
    current_heading_only = _is_child_heading(child_sents[0])
    current_kind = child_sents[0].item_type
    current_atomic_id = child_sents[0].atomic_id
    groups: list[str] = []

    enc = time.perf_counter()
    vecs = sentence_vecs if sentence_vecs is not None else None
    if vecs is not None and len(vecs) == len(child_sents):
        centroid = list(vecs[0])
        count_in_chunk = 1
    else:
        vecs = None
        centroid = None
        count_in_chunk = 0
    for i, next_sent in enumerate(child_sents[1:]):
        candidate = f"{current_chunk} {next_sent.text}"
        same_atomic_unit = bool(
            current_atomic_id
            and next_sent.atomic_id
            and current_atomic_id == next_sent.atomic_id
        )
        if same_atomic_unit and count_words(candidate) <= max_child_words:
            current_chunk = candidate
            current_heading_only = False
            current_kind = "content"
            if vecs is not None:
                c0, c1 = count_in_chunk, count_in_chunk + 1
                centroid = [
                    (a * c0 + b) / c1 for a, b in zip(centroid, vecs[i + 1])
                ]
                count_in_chunk = c1
            continue
        if (
            current_heading_only
            and _is_child_heading(next_sent)
            and count_words(candidate) <= max_child_words
        ):
            current_chunk = candidate
            if vecs is not None:
                c0, c1 = count_in_chunk, count_in_chunk + 1
                centroid = [
                    (a * c0 + b) / c1 for a, b in zip(centroid, vecs[i + 1])
                ]
                count_in_chunk = c1
            continue
        if _is_child_heading(next_sent) or next_sent.item_type == "table" or (
            current_kind == "table" and next_sent.item_type != "table"
        ):
            groups.append(current_chunk)
            current_chunk = next_sent.text
            current_heading_only = _is_child_heading(next_sent)
            current_kind = next_sent.item_type
            current_atomic_id = next_sent.atomic_id
            if vecs is not None:
                centroid = list(vecs[i + 1])
                count_in_chunk = 1
            logger.info(
                "Child boundary | parent_id=%s | action=split | reason=heading",
                parent.parent_id,
            )
            continue
        if current_heading_only and count_words(candidate) <= max_child_words:
            current_chunk = candidate
            current_heading_only = False
            current_kind = "content"
            current_atomic_id = next_sent.atomic_id or current_atomic_id
            if vecs is not None:
                c0, c1 = count_in_chunk, count_in_chunk + 1
                centroid = [
                    (a * c0 + b) / c1 for a, b in zip(centroid, vecs[i + 1])
                ]
                count_in_chunk = c1
            continue
        if count_words(candidate) > max_child_words:
            groups.append(current_chunk)
            current_chunk = next_sent.text
            current_heading_only = False
            current_kind = next_sent.item_type
            current_atomic_id = next_sent.atomic_id
            if vecs is not None:
                centroid = list(vecs[i + 1])
                count_in_chunk = 1
            continue

        if vecs is not None:
            next_embedding = vecs[i + 1]
            sim = cosine_sim(centroid, next_embedding)
        else:
            current_embedding = dense_vector([current_chunk])[0]
            next_embedding = dense_vector([next_sent.text])[0]
            sim = cosine_sim(current_embedding, next_embedding)

        logger.info(
            "Child boundary | parent_id=%s | sim=%.3f | threshold=%.2f | action=%s | reason=%s",
            parent.parent_id,
            sim,
            SIMILARITY_THRESHOLD_CHILD,
            "merge" if sim >= SIMILARITY_THRESHOLD_CHILD else "split",
            "merge" if sim >= SIMILARITY_THRESHOLD_CHILD else "similarity_break",
        )
        if sim >= SIMILARITY_THRESHOLD_CHILD:
            current_chunk = candidate
            current_kind = "content"
            current_atomic_id = next_sent.atomic_id or current_atomic_id
            if vecs is not None:
                c0, c1 = count_in_chunk, count_in_chunk + 1
                centroid = [
                    (a * c0 + b) / c1 for a, b in zip(centroid, next_embedding)
                ]
                count_in_chunk = c1
        else:
            groups.append(current_chunk)
            current_chunk = next_sent.text
            current_heading_only = False
            current_kind = next_sent.item_type
            current_atomic_id = next_sent.atomic_id
            if vecs is not None:
                centroid = list(vecs[i + 1])
                count_in_chunk = 1

    encode_ms = (time.perf_counter() - enc) * 1000
    groups.append(current_chunk)

    # Enforce the minimum child size FIRST: drop / merge tiny fragments while
    # groups are still disjoint. Doing this before the overlap step prevents a
    # carried-forward sentence from being duplicated when a fragment is folded
    # into a chunk that already received it as overlap.
    heading_prefixes = {
        s.text for s in child_sents if s.item_type == "heading"
    }
    floored = _apply_min_floor(groups, max_child_words, heading_prefixes)

    # Overlap: carry exactly the last sentence of the previous child forward
    # into the next child (adds continuity for retrieval). The cap stays strict:
    # if the carried sentence would push the next child over ``max_child_words``
    # it is dropped from the overlap (it still lives in its own chunk), so no
    # chunk is ever created above the size limit.
    overlapped: list[str] = []
    previous_body: str | None = None
    for body in floored:
        body = body.strip()
        if not body:
            continue
        if previous_body:
            last_sent = _last_sentence_of(previous_body)
            if last_sent and last_sent not in body:
                candidate = f"{last_sent} {body}".strip()
                if count_words(candidate) <= max_child_words:
                    body = candidate
                else:
                    logger.warning(
                        "Child overlap skipped | parent_id=%s | would exceed "
                        "max_child_words=%d",
                        parent.parent_id,
                        max_child_words,
                    )
        overlapped.append(body)
        previous_body = body

    filtered = overlapped

    logger.debug(
        "Child split | parent_id=%s | sentences=%d | children=%d | embed_time=%.0fms | max_words=%d",
        parent.parent_id,
        len(child_sents),
        len(filtered),
        encode_ms,
        max_child_words,
    )

    children: list[ChildChunk] = []
    nonempty = [b for b in filtered if b]
    for body in nonempty:
        body_sents = [s for s in child_sents if s.text in body]
        pages = [s.page for s in body_sents if s.page is not None]
        s_page = min(pages) if pages else parent.page_start
        e_page = max(pages) if pages else parent.page_end
        def contributes(unit: Paragraph) -> bool:
            if unit.text in body:
                return True
            cleaned = strip_list_markers(unit.text)
            return any(piece in body for piece in split_sentences(cleaned))

        source_items = [unit for unit in (parent.source_items or []) if contributes(unit)]
        child_heading, child_heading_level = _group_heading(source_items)
        child_heading = child_heading or parent.book_heading
        child_heading_level = child_heading_level or parent.heading_level
        heading_item = next(
            (unit for unit in source_items if unit.text == child_heading), None
        )
        child_role = parent.content_role
        if any(unit.item_type == "table" for unit in source_items):
            child_role = "table"
        elif heading_item and _SPECIAL_ROLE_RE.search(heading_item.text):
            child_role = "supplementary"
        children.append(
            ChildChunk(
                child_id=str(uuid.uuid4()),
                parent_id=parent.parent_id,
                document_id=parent.document_id,
                title=None,
                page_start=s_page,
                page_end=e_page,
                content=body.strip(),
                book_heading=child_heading,
                heading_level=child_heading_level,
                content_role=child_role,
                source_items=source_items or None,
                language=detect_language(body) if body.strip() else parent.language,
            )
        )
    return children


def _dedupe_across_levels(
    parents: list[ParentChunk], children: list[ChildChunk]
) -> None:
    """One document-wide uniqueness pass over section + subsection titles.

    Parallel batches only see completed batches, and sections/subsections are
    titled separately, so the same concept can still win in two places (e.g. a
    section and its subsection both called "Machine Learning Overview"). Re-run
    uniqueness over the union so no two displayed titles collide.
    """
    titled = [c for c in children if c.title]
    items = parents + titled
    if not items:
        return
    contents = [x.content for x in items]
    titles = [x.title for x in items]
    max_words = max(FALLBACK_SECTION_MAX_WORDS, FALLBACK_SUBSECTION_MAX_WORDS)
    final = make_titles_unique(titles, contents, max_words)
    for item, title in zip(items, final):
        item.title = title


# ---------------------------------------------------------------------------
# Overlap verification (spot-check invariants from the pipeline).
# ---------------------------------------------------------------------------


def _word_overlap(a: str, b: str) -> int:
    """Number of words shared by b's start that follow a's end (suffix==prefix)."""
    aw = a.split()
    bw = b.split()
    overlap = 0
    for k in range(1, min(len(aw), len(bw)) + 1):
        if aw[-k:] == bw[:k]:
            overlap = k
    return overlap


def verify_chunk_invariants(chunks: dict[str, Any]) -> list[str]:
    """Spot-check the chunking contract.

    Returns a list of warnings:
    - parent overlap: two adjacent parents share any text (must be zero).
    - child overlap: two adjacent children (within a parent) may share at most
      the last sentence of the previous child.
    """
    warnings: list[str] = []
    parents = chunks.get("parents", [])
    children = chunks.get("children", [])

    for i in range(1, len(parents)):
        ov = _word_overlap(parents[i - 1].get("content", ""), parents[i].get("content", ""))
        if ov > 0:
            warnings.append(
                f"Parent overlap ({ov} words) between parents "
                f"{parents[i - 1]['parent_id']} and {parents[i]['parent_id']}"
            )

    by_parent: dict[str, list] = {}
    for child in children:
        by_parent.setdefault(child.get("parent_id"), []).append(child)
    for pid, siblings in by_parent.items():
        for j in range(1, len(siblings)):
            prev_c, cur_c = siblings[j - 1], siblings[j]
            expected = count_words(_last_sentence_of(prev_c.get("content", "")))
            actual = _word_overlap(prev_c.get("content", ""), cur_c.get("content", ""))
            if actual > expected:
                warnings.append(
                    f"Child overlap exceeds one sentence ({actual}>{expected} words) "
                    f"parent={pid} children={prev_c['child_id']}->{cur_c['child_id']}"
                )
    return warnings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _label_families(
    parents: list[ParentChunk], children: list[ChildChunk], client=None,
    check_cancelled: Callable[[], None] | None = None,
    document_language: str | None = None,
) -> None:
    """Label sections and their subsections, one LLM call per batch of families.

    Families are processed in document order, ``TITLE_BATCH_SIZE`` parents per
    call. Each family entry is the parent SECTION immediately followed by its
    own SUBSECTION children, so parent/child headers are written together and
    stay consistent (a child header never generalizes its section header). The
    most recent accepted headers are threaded across batches so headers stay
    distinct document-wide; any title that fails the format / blocklist /
    exact-dup checks receives exactly one regeneration call. Children whose
    title is ``None`` are single-child subsections that simply mirror their
    parent -- they are never labeled.
    """
    if not parents:
        return
    by_parent: dict[str, list[ChildChunk]] = {p.parent_id: [] for p in parents}
    child_counts = Counter(c.parent_id for c in children)
    for c in children:
        if child_counts[c.parent_id] > 1:
            by_parent.setdefault(c.parent_id, []).append(c)

    client = client or make_title_client()
    used: set[str] = set()
    recent: list[str] = []

    for start in range(0, len(parents), TITLE_BATCH_SIZE):
        if check_cancelled:
            check_cancelled()
        batch = parents[start : start + TITLE_BATCH_SIZE]
        entries: list[tuple[str, str, str | None]] = []
        owners: list[tuple[str, ParentChunk | ChildChunk]] = []
        for p in batch:
            p.language = _resolved_chunk_language(p.content, document_language)
            entries.append(("section", p.content, p.book_heading, p.language))
            owners.append(("section", p))
            for c in by_parent.get(p.parent_id, []):
                c.language = _resolved_chunk_language(c.content, document_language)
                entries.append(("subsection", c.content, c.book_heading, c.language))
                owners.append(("subsection", c))

        titles = generate_family_batch_titles(
            entries, client=client, before=recent[-TITLE_CONTEXT_RECENT:]
        )
        if check_cancelled:
            check_cancelled()
        for (level, item), title in zip(owners, titles):
            if not (
                title
                and is_acceptable_title(
                    title,
                    item.content,
                    level,
                    used,
                    item.book_heading or "",
                    item.language,
                )
            ):
                title = regenerate_title(
                    client=client,
                    content=item.content,
                    level=level,
                    reject=[title] if title else [],
                    used_titles=used,
                    book_heading=item.book_heading or "",
                    expected_language=item.language,
                )
            if title and is_acceptable_title(
                title,
                item.content,
                level,
                used,
                item.book_heading or "",
                item.language,
            ):
                item.title = title
                used.add(title)
                recent.append(title)
                cid = getattr(item, "child_id", getattr(item, "parent_id", ""))
                logger.info(
                    "%s title | id=%s | words=%d | title=%s",
                    level.title(),
                    cid,
                    count_words(item.content),
                    title,
                )



def _enforce_title_invariants(
    parents: list[ParentChunk], children: list[ChildChunk], client=None,
    check_cancelled: Callable[[], None] | None = None,
    document_language: str | None = None,
) -> None:
    """Final document-wide grounding/uniqueness pass after optional review."""
    client = client or make_title_client()
    used: set[str] = set()
    child_counts = Counter(c.parent_id for c in children)
    items: list[tuple[str, ParentChunk | ChildChunk]] = []
    for parent in parents:
        items.append(("section", parent))
        items.extend(
            ("subsection", child)
            for child in children
            if child.parent_id == parent.parent_id and child_counts[child.parent_id] > 1
        )

    for index, (level, item) in enumerate(items, 1):
        if check_cancelled:
            check_cancelled()
        item.language = _resolved_chunk_language(item.content, document_language)
        if item.title and is_acceptable_title(
            item.title,
            item.content,
            level,
            used,
            item.book_heading or "",
            item.language,
        ):
            used.add(item.title)
            continue
        regenerated = regenerate_title(
            client=client,
            content=item.content,
            level=level,
            reject=[item.title] if item.title else [],
            used_titles=used,
            book_heading=item.book_heading or "",
            expected_language=item.language,
        )
        if regenerated and is_acceptable_title(
            regenerated,
            item.content,
            level,
            used,
            item.book_heading or "",
            item.language,
        ):
            item.title = regenerated
            used.add(regenerated)
            continue
        if level == "subsection":
            item.title = None
            continue
        emergency = clean = descriptive_fallback(
            item.content,
            item.book_heading or "",
            index=index,
            max_words=FALLBACK_SECTION_MAX_WORDS,
            language=item.language,
        )
        suffix = 2
        while emergency in used:
            emergency = f"{clean} {suffix}"
            suffix += 1
        item.title = emergency
        used.add(emergency)


def build_semantic_structure(
    pages: list[dict[str, Any]], document_id: str,
    *,
    document_language: str | None = None,
    on_stage: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Generate the semantic section/subsection structure for a document.

    Returns {"parents": [...], "children": [...]}. Only children are the
    retrieval units; parents exist for organization / future selection.
    """
    t0 = time.perf_counter()
    logger.info("Semantic chunking started | document_id=%s", document_id)

    def checkpoint(stage: str | None = None) -> None:
        if is_cancelled and is_cancelled():
            from app.offline.pipeline import PipelineCancelled
            raise PipelineCancelled("Document processing was cancelled")
        if stage and on_stage:
            on_stage(stage)

    # PHASE 1 - chunking only (no LLM). Parents are built first, then each
    # parent is immediately chunked into its child units, in document order.
    checkpoint("detecting_headings")
    parents = build_parents(split_parents(pages), document_id)
    logger.info("Parents generated | count=%d", len(parents))

    # Embed every sentence of the whole book ONCE, then split each parent's
    # children by comparing the precomputed vectors (no per-sentence model calls).
    checkpoint("chunking")
    sentence_vecs = preembed_sentences(parents)
    checkpoint()

    children: list[ChildChunk] = []
    for parent in parents:
        checkpoint()
        children.extend(
            build_children(parent, sentence_vecs=sentence_vecs.get(parent.parent_id))
        )

    child_counts = Counter(c.parent_id for c in children)
    for c in children:
        if child_counts[c.parent_id] == 1:
            c.title = None

    logger.info(
        "CHUNK COUNTS | document_id=%s | parent_chunks=%d | child_chunks=%d",
        document_id,
        len(parents),
        len(children),
    )

    # PHASE 2 - titles only (LLM): one call per batch of parent families.
    checkpoint("generating_titles")
    title_client = make_title_client()
    _label_families(
        parents,
        children,
        client=title_client,
        check_cancelled=checkpoint,
        document_language=document_language,
    )
    _enforce_title_invariants(
        parents,
        children,
        client=title_client,
        check_cancelled=checkpoint,
        document_language=document_language,
    )
    checkpoint()
    if any(not p.title for p in parents):
        raise ValueError("Title generation left a parent without a title")

    elapsed = time.perf_counter() - t0
    logger.info(
        "Semantic chunking completed | parents=%d | children=%d | time=%.2fs",
        len(parents),
        len(children),
        elapsed,
    )

    # Metadata gathers for parent/child persistence. Children are appended in
    # document order inside each parent, so a per-parent running counter yields a
    # stable child_order and the accumulated subsection titles stay aligned with
    # the child order.
    parent_titles = {p.parent_id: p.title for p in parents}
    parent_subtitles: dict[str, list] = {}
    child_order_map: dict[str, int] = {}
    for c in children:
        parent_subtitles.setdefault(c.parent_id, []).append(c.title)
        child_order_map[c.child_id] = len(parent_subtitles[c.parent_id]) - 1

    def source_metadata(items: list[Paragraph] | None) -> list[dict[str, Any]]:
        return [
            {
                "type": item.item_type,
                "page": item.page,
                "heading_level": item.heading_level,
                "order": item.order,
                "role": item.role,
                "parser_type": item.parser_item_type,
                "effective_type": item.item_type,
                "heading_score": item.heading_score,
                "heading_decision": item.heading_decision,
                "heading_reasons": list(item.heading_reasons),
                "atomic_id": item.atomic_id,
                "text": item.text,
            }
            for item in (items or [])
        ]

    return {
        "parents": [
            {
                "parent_id": p.parent_id,
                "document_id": p.document_id,
                "title": p.title,
                "parent_title": p.title,
                "chunk_type": "parent",
                "subsection_titles": parent_subtitles.get(p.parent_id, []),
                "page_start": p.page_start,
                "page_end": p.page_end,
                "content": p.content,
                "book_heading": p.book_heading,
                "heading_level": p.heading_level,
                "content_role": p.content_role,
                "language": p.language,
                "source_items": source_metadata(p.source_items),
            }
            for p in parents
        ],
        "children": [
            {
                "child_id": c.child_id,
                "chunk_id": c.child_id,
                "parent_id": c.parent_id,
                "document_id": c.document_id,
                "title": c.title,
                "heading": c.title,
                "chunk_title": c.title,
                "parent_title": parent_titles.get(c.parent_id),
                "chunk_type": "child",
                "child_order": child_order_map[c.child_id],
                "page_start": c.page_start,
                "page_end": c.page_end,
                "content": c.content,
                "book_heading": c.book_heading,
                "heading_level": c.heading_level,
                "content_role": c.content_role,
                "language": c.language,
                "source_items": source_metadata(c.source_items),
            }
            for c in children
        ],
    }
