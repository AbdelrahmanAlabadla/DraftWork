from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any

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
from app.offline.embeddings import cosine_sim, dense_vector, device_name
from app.offline.parser_items import item_text, page_items, page_number
from app.offline.title_generator import (
    generate_batch_titles,
    generate_family_batch_titles,
    is_acceptable_title,
    make_titles_unique,
    make_title_client,
    regenerate_title,
    review_titles,
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


def count_words(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


@dataclass
class Paragraph:
    text: str
    page: int | None = None


@dataclass
class Sentence:
    text: str
    page: int | None = None


@dataclass
class ParentChunk:
    parent_id: str
    document_id: str
    title: str | None
    page_start: int | None
    page_end: int | None
    content: str


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    document_id: str
    title: str | None
    page_start: int | None
    page_end: int | None
    content: str


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
    for page in pages:
        page_num = page_number(page)
        for item in page_items(page):
            text = item_text(item)
            if text:
                paragraphs.append(Paragraph(text=text, page=page_num))
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
    """Split plain text into sentences on ``.!?`` + whitespace + capital/quote/digit.

    Guards against known abbreviations (``Dr.``), decimals (``4.91``), and
    figure/equation references (``Figure 1-1.``) that should not end a sentence.
    """
    if not text:
        return []
    text = _mask_non_boundary_periods(text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
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

    groups: list[list[Paragraph]] = [[paragraphs[0]]]
    for i in range(1, len(paragraphs)):
        sim = cosine_sim(vecs[i - 1], vecs[i])
        split = sim < threshold
        logger.info(
            "Parent boundary | i=%d | sim=%.4f | threshold=%.2f | action=%s",
            i,
            sim,
            threshold,
            "merge" if not split else "split",
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
    # 1. Drop the tiny ones first.
    kept = [p for p in parents if count_tokens(p.content) > PARENT_MIN_TOKENS_DROP]
    if not kept:
        return []

    # 2. Merge short parents into a neighbour (prefer the previous).
    merged: list[ParentChunk] = []
    for parent in kept:
        if merged and count_tokens(parent.content) < PARENT_MERGE_TOKENS:
            prev = merged[-1]
            page_start, page_end = _page_range(prev, parent)
            merged[-1] = ParentChunk(
                parent_id=prev.parent_id,
                document_id=prev.document_id,
                title=prev.title,
                page_start=page_start,
                page_end=page_end,
                content=f"{prev.content} {parent.content}".strip(),
            )
            continue
        merged.append(parent)
    # First parent may itself be short; fold the second into it if so.
    if (
        len(merged) >= 2
        and count_tokens(merged[0].content) < PARENT_MERGE_TOKENS
    ):
        head = merged[0]
        nxt = merged[1]
        page_start, page_end = _page_range(head, nxt)
        merged[1] = ParentChunk(
            parent_id=head.parent_id,
            document_id=head.document_id,
            title=head.title,
            page_start=page_start,
            page_end=page_end,
            content=f"{head.content} {nxt.content}".strip(),
        )
        merged.pop(0)

    # 3. Split any parent that exceeds the hard ceiling at sentence boundaries.
    max_parent_words = max(1, int(PARENT_MAX_SIZE / WORDS_PER_TOKEN))
    final: list[ParentChunk] = []
    for parent in merged:
        if count_words(parent.content) <= max_parent_words:
            final.append(parent)
            continue
        sents = split_sentences(parent.content)
        current = ""
        for sent in sents:
            candidate = f"{current} {sent}".strip()
            if current and count_words(candidate) > max_parent_words:
                page_start, page_end = parent.page_start, parent.page_end
                final.append(
                    ParentChunk(
                        parent_id=str(uuid.uuid4()),
                        document_id=parent.document_id,
                        title=None,
                        page_start=page_start,
                        page_end=page_end,
                        content=current,
                    )
                )
                current = sent
            else:
                current = candidate
        if current:
            final.append(
                ParentChunk(
                    parent_id=str(uuid.uuid4()),
                    document_id=parent.document_id,
                    title=None,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    content=current,
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
        parents.append(
            ParentChunk(
                parent_id=str(uuid.uuid4()),
                document_id=document_id,
                title=None,
                page_start=page_start,
                page_end=page_end,
                content=content,
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
    return int(count_words(text) / WORDS_PER_TOKEN)


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
_LIST_MARKER_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[.)]\s+")

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
    return stripped.rstrip().endswith("?")


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
    groups: list[str], max_child_words: int
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
    out: list[str] = []
    for g in groups:
        tk = count_tokens(g)
        # Genuine junk at or below the drop floor is discarded; meaningful
        # fragments (definitions / questions / headings) always survive.
        if tk <= CHILD_MIN_TOKENS_DROP and not _meaningful_fragment(g):
            continue
        # Anything else under the merge floor is folded into the previous chunk,
        # but never past the size ceiling.
        if (
            tk <= CHILD_MIN_TOKENS_MERGE
            and out
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
        for s in _prepare_child_sentences(p.content):
            pairs.append((p.parent_id, s))
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
            )
        ]

    selected = _prepare_child_sentences(text)
    if not selected:
        return []

    child_sents = [Sentence(s, page=None) for s in selected]

    max_child_words = _max_child_words()
    current_chunk = child_sents[0].text
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
        if count_words(candidate) > max_child_words:
            groups.append(current_chunk)
            current_chunk = next_sent.text
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
            if vecs is not None:
                c0, c1 = count_in_chunk, count_in_chunk + 1
                centroid = [
                    (a * c0 + b) / c1 for a, b in zip(centroid, next_embedding)
                ]
                count_in_chunk = c1
        else:
            groups.append(current_chunk)
            current_chunk = next_sent.text
            if vecs is not None:
                centroid = list(vecs[i + 1])
                count_in_chunk = 1

    encode_ms = (time.perf_counter() - enc) * 1000
    groups.append(current_chunk)

    # Enforce the minimum child size FIRST: drop / merge tiny fragments while
    # groups are still disjoint. Doing this before the overlap step prevents a
    # carried-forward sentence from being duplicated when a fragment is folded
    # into a chunk that already received it as overlap.
    floored = _apply_min_floor(groups, max_child_words)

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
    body_pages = _dispatch_pages(
        [b for b in filtered if b], parent.page_start, parent.page_end
    )
    for body, (s_page, e_page) in zip(
        [b for b in filtered if b], body_pages
    ):
        children.append(
            ChildChunk(
                child_id=str(uuid.uuid4()),
                parent_id=parent.parent_id,
                document_id=parent.document_id,
                title=None,
                page_start=s_page,
                page_end=e_page,
                content=body.strip(),
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
    parents: list[ParentChunk], children: list[ChildChunk], client=None
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
        batch = parents[start : start + TITLE_BATCH_SIZE]
        entries: list[tuple[str, str]] = []
        owners: list[tuple[str, ParentChunk | ChildChunk]] = []
        for p in batch:
            entries.append(("section", p.content))
            owners.append(("section", p))
            for c in by_parent.get(p.parent_id, []):
                entries.append(("subsection", c.content))
                owners.append(("subsection", c))

        titles = generate_family_batch_titles(
            entries, client=client, before=recent[-TITLE_CONTEXT_RECENT:]
        )
        for (level, item), title in zip(owners, titles):
            if not (title and is_acceptable_title(title, item.content, level, used)):
                title = regenerate_title(
                    client=client,
                    content=item.content,
                    level=level,
                    reject=[title] if title else [],
                    used_titles=used,
                )
            if title and is_acceptable_title(title, item.content, level, used):
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


def build_semantic_structure(
    pages: list[dict[str, Any]], document_id: str
) -> dict[str, list[dict[str, Any]]]:
    """Generate the semantic section/subsection structure for a document.

    Returns {"parents": [...], "children": [...]}. Only children are the
    retrieval units; parents exist for organization / future selection.
    """
    t0 = time.perf_counter()
    logger.info("Semantic chunking started | document_id=%s", document_id)

    # PHASE 1 - chunking only (no LLM). Parents are built first, then each
    # parent is immediately chunked into its child units, in document order.
    parents = build_parents(split_parents(pages), document_id)
    logger.info("Parents generated | count=%d", len(parents))

    # Embed every sentence of the whole book ONCE, then split each parent's
    # children by comparing the precomputed vectors (no per-sentence model calls).
    sentence_vecs = preembed_sentences(parents)

    children: list[ChildChunk] = []
    for parent in parents:
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
    _label_families(parents, children)

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
            }
            for c in children
        ],
    }