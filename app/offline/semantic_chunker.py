from __future__ import annotations

import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
    TITLE_PARALLELISM,
    WORDS_PER_TOKEN,
)
from app.logging_conf import get_logger
from app.offline.embeddings import cosine_sim, dense_vector
from app.offline.parser_items import item_text, page_items, page_number
from app.offline.title_generator import generate_section_title, generate_subsection_title

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

    Every non-empty LlamaParse item (text, table, and headings alike) becomes
    one paragraph unit; headings are treated as ordinary paragraphs. Parent
    chunking then merges consecutive paragraphs purely by embedding similarity.
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


def generate_section_titles(parents: list[ParentChunk], **kwargs) -> None:
    if not parents:
        return
    with ThreadPoolExecutor(max_workers=TITLE_PARALLELISM) as ex:
        titles = list(
            ex.map(
                lambda parent: generate_section_title(
                    content=parent.content, **kwargs
                ),
                parents,
            )
        )
    for parent, title in zip(parents, titles):
        parent.title = title
        logger.info(
            "Section title | parent_id=%s | words=%d | title=%s",
            parent.parent_id,
            count_words(parent.content),
            parent.title,
        )


def generate_subsection_titles(children: list[ChildChunk], **kwargs) -> None:
    if not children:
        return
    with ThreadPoolExecutor(max_workers=TITLE_PARALLELISM) as ex:
        titles = list(
            ex.map(
                lambda child: generate_subsection_title(
                    content=child.content, **kwargs
                ),
                children,
            )
        )
    for child, title in zip(children, titles):
        child.title = title
        logger.info(
            "Subsection title | child_id=%s | words=%d | title=%s",
            child.child_id,
            count_words(child.content),
            child.title,
        )


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


def strip_list_markers(text: str) -> str:
    """Remove plain numbered-list markers (``1. `` / ``2) ````).

    A marker is removed only when its number is small (1-2 digits), is NOT a
    structural number ("Chapter 3", "Section 2.1", ...), and is not a year-like
    larger number. The marker's number is dropped but the copied text remains.
    """
    if not text:
        return text
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


def _apply_min_floor(groups: list[str]) -> list[str]:
    """Enforce the minimum child size on already-packed groups.

    - at or below CHILD_MIN_TOKENS_DROP   -> drop the group entirely
    - between DROP and CHILD_MIN_TOKENS_MERGE -> merge into one/two chunk before
    Returns the filtered, still ordered list of chunk strings.
    """
    if not groups:
        return []
    out: list[str] = []
    for g in groups:
        # fold into previous
        tk = count_tokens(g)
        # Add to the previous group if small
        if tk <= CHILD_MIN_TOKENS_DROP and out:
            # drop
            continue
        if tk <= CHILD_MIN_TOKENS_MERGE and out:
            # merge with previous group
            out[-1] = f"{out[-1]} {g}".strip()
            continue
        out.append(g)
    # Leading tiny single group under the merge floor: fold into next if any.
    if len(out) >= 2 and count_tokens(out[0]) <= CHILD_MIN_TOKENS_MERGE:
        out[1] = f"{out[0]} {out[1]}".strip()
        out.pop(0)
    return out


def build_children(parent: ParentChunk) -> list[ChildChunk]:
    """Split one parent's content into focused child chunks (sentence packing).

    The sentence stream is walked in order, building a single current chunk.
    Before any merge the candidate (current chunk + next sentence) is checked
    against the size ceiling; if it exceeds the cap the current chunk is
    finalized and the next sentence starts a new chunk. Otherwise the complete
    current chunk text is re-embedded and compared against the next sentence:
    it joins while ``cosine_sim(embed(current), embed(next)) >=
    SIMILARITY_THRESHOLD_CHILD``; otherwise the current chunk is finalized and
    the next sentence starts a new chunk. No running centroid or average of
    sentence embeddings is used -- the similarity comparison is always against
    the embedding of the actual accumulated chunk text, re-embedded after every
    successful merge. ``CHILD_MAX_SIZE`` remains a hard ceiling that closes the
    chunk on size regardless of similarity. A child is always cut at a sentence
    boundary, and between any two adjacent children exactly the last sentence
    of the previous child is carried forward as overlap.
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

    # Strip plain numbered-list markers so "1. " / "2) " don't pollute the
    # embedding. Structural numbers (Chapter 3, Section 2.1) are preserved.
    combined = strip_list_markers(text)
    if not combined:
        return []

    child_sents = [Sentence(s, page=parent.page_start) for s in split_sentences(combined)]
    if not child_sents:
        return []

    max_child_words = _max_child_words()
    current_chunk = child_sents[0].text
    groups: list[str] = []

    enc = time.perf_counter()
    for next_sent in child_sents[1:]:
        candidate = f"{current_chunk} {next_sent.text}"
        if count_words(candidate) > max_child_words:
            groups.append(current_chunk)
            current_chunk = next_sent.text
            continue

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
        else:
            groups.append(current_chunk)
            current_chunk = next_sent.text

    encode_ms = (time.perf_counter() - enc) * 1000
    groups.append(current_chunk)

    # Overlap: carry exactly the last sentence of the previous child forward
    # into the next child (adds continuity for retrieval).
    overlapped: list[str] = []
    previous_body: str | None = None
    for body in groups:
        body = body.strip()
        if not body:
            continue
        if previous_body:
            last_sent = _last_sentence_of(previous_body)
            if last_sent and last_sent not in body:
                body = f"{last_sent} {body}".strip()
        overlapped.append(body)
        previous_body = body

    # Enforce the minimum child size: drop / merge tiny fragments.
    filtered = _apply_min_floor(overlapped)

    logger.debug(
        "Child split | parent_id=%s | sentences=%d | children=%d | embed_time=%.0fms | max_words=%d",
        parent.parent_id,
        len(child_sents),
        len(filtered),
        encode_ms,
        max_child_words,
    )

    children: list[ChildChunk] = []
    for body in filtered:
        if not body:
            continue
        children.append(
            ChildChunk(
                child_id=str(uuid.uuid4()),
                parent_id=parent.parent_id,
                document_id=parent.document_id,
                title=None,
                page_start=parent.page_start,
                page_end=parent.page_end,
                content=body.strip(),
            )
        )
    return children


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


def build_semantic_structure(
    pages: list[dict[str, Any]], document_id: str
) -> dict[str, list[dict[str, Any]]]:
    """Generate the semantic section/subsection structure for a document.

    Returns {"parents": [...], "children": [...]}. Only children are the
    retrieval units; parents exist for organization / future selection.
    """
    t0 = time.perf_counter()
    logger.info("Semantic chunking started | document_id=%s", document_id)

    parents = build_parents(split_parents(pages), document_id)
    logger.info("Parents generated | count=%d", len(parents))

    if parents:
        generate_section_titles(parents)

    children: list[ChildChunk] = []
    for parent in parents:
        children.extend(build_children(parent))

    logger.info(
        "CHUNK COUNTS | document_id=%s | parent_chunks=%d | child_chunks=%d",
        document_id,
        len(parents),
        len(children),
    )

    if children:
        generate_subsection_titles(children)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Semantic chunking completed | parents=%d | children=%d | time=%.2fs",
        len(parents),
        len(children),
        elapsed,
    )

    return {
        "parents": [
            {
                "parent_id": p.parent_id,
                "document_id": p.document_id,
                "title": p.title,
                "page_start": p.page_start,
                "page_end": p.page_end,
                "content": p.content,
            }
            for p in parents
        ],
        "children": [
            {
                "child_id": c.child_id,
                "parent_id": c.parent_id,
                "document_id": c.document_id,
                "title": c.title,
                "heading": c.title,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "content": c.content,
            }
            for c in children
        ],
    }