from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.logging_conf import get_logger
from app.offline.parser_items import item_text, item_type, page_items, page_number

logger = get_logger("CHUNKER")

_BODY_TYPES = frozenset({"text", "table"})

_MIN_PARENT_TOKENS = 1500
_MAX_PARENT_TOKENS = 3000
_MIN_CHILD_TOKENS = 300
_MAX_CHILD_TOKENS = 600


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Good enough for chunk sizing."""
    return max(1, len(text) // 4)


@dataclass
class Section:
    heading: str | None
    page: int | None
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class ParentChunk:
    parent_id: str
    document_id: str
    heading: str | None
    page: int | None
    content: str


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    document_id: str
    page: int | None
    heading: str | None
    content: str


def _build_sections(pages: list[dict[str, Any]]) -> list[Section]:
    """Group body items under their nearest preceding heading."""
    sections: list[Section] = []
    current: Section | None = None

    for page in pages:
        page_num = page_number(page)
        for item in page_items(page):
            typ = item_type(item)
            if typ == "heading":
                title = item_text(item)
                if title:
                    if current is not None and current.paragraphs:
                        sections.append(current)
                    current = Section(heading=title, page=page_num)
            elif typ in _BODY_TYPES:
                text = item_text(item)
                if not text:
                    continue
                if current is None:
                    current = Section(heading=None, page=page_num)
                current.paragraphs.append(text)

    if current is not None and current.paragraphs:
        sections.append(current)
    return sections


def _split_paragraph_list(paragraphs: list[str], max_tokens: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        tokens = estimate_tokens(paragraph)
        if current and current_tokens + tokens > max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(paragraph)
        current_tokens += tokens
    if current:
        groups.append(current)
    return groups


def _build_parents(sections: list[Section], document_id: str) -> list[ParentChunk]:
    parents: list[ParentChunk] = []
    for section in sections:
        groups = _split_paragraph_list(section.paragraphs, _MAX_PARENT_TOKENS)
        for group in groups:
            content = "\n\n".join(group)
            if section.heading:
                content = f"{section.heading}\n\n{content}"
            if not content.strip():
                continue
            parents.append(
                ParentChunk(
                    parent_id=str(uuid.uuid4()),
                    document_id=document_id,
                    heading=section.heading,
                    page=section.page,
                    content=content.strip(),
                )
            )
    return parents


def _split_children(parent: ParentChunk) -> list[ChildChunk]:
    """Split a parent into overlapping child chunks of ~300-600 tokens."""
    blocks = [b.strip() for b in parent.content.split("\n\n") if b.strip()]
    if not blocks:
        return []

    children: list[ChildChunk] = []
    current: list[str] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current, current_tokens
        content = "\n\n".join(current).strip()
        if content:
            children.append(
                ChildChunk(
                    child_id=str(uuid.uuid4()),
                    parent_id=parent.parent_id,
                    document_id=parent.document_id,
                    page=parent.page,
                    heading=parent.heading,
                    content=content,
                )
            )
        # Overlap with the previous block for retrieval continuity.
        current = [current[-1]] if current else []
        current_tokens = estimate_tokens(current[0]) if current else 0

    for block in blocks:
        block_tokens = estimate_tokens(block)
        if current_tokens + block_tokens > _MAX_CHILD_TOKENS and current_tokens >= _MIN_CHILD_TOKENS:
            _flush()
        current.append(block)
        current_tokens += block_tokens

    if current:
        content = "\n\n".join(current).strip()
        if content:
            children.append(
                ChildChunk(
                    child_id=str(uuid.uuid4()),
                    parent_id=parent.parent_id,
                    document_id=parent.document_id,
                    page=parent.page,
                    heading=parent.heading,
                    content=content,
                )
            )
    return children


def build_hierarchical_chunks(
    pages: list[dict[str, Any]], document_id: str
) -> dict[str, list[dict[str, Any]]]:
    t0 = time.perf_counter()
    logger.info("Chunking started | document_id=%s", document_id)

    sections = _build_sections(pages)
    parents = _build_parents(sections, document_id)

    children: list[ChildChunk] = []
    for parent in parents:
        children.extend(_split_children(parent))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Chunking completed | parents=%d | children=%d | sections=%d | time=%.2fs",
        len(parents),
        len(children),
        len(sections),
        elapsed,
    )

    return {
        "parents": [
            {
                "parent_id": p.parent_id,
                "document_id": p.document_id,
                "heading": p.heading,
                "page": p.page,
                "content": p.content,
            }
            for p in parents
        ],
        "children": [
            {
                "child_id": c.child_id,
                "parent_id": c.parent_id,
                "document_id": c.document_id,
                "page": c.page,
                "heading": c.heading,
                "content": c.content,
            }
            for c in children
        ],
    }
