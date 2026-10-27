from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.config import STRUCTURES_DIR
from app.logging_conf import get_logger

logger = get_logger("STRUCTURE_STORE")


def _path_for(document_id: str) -> Path:
    return Path(STRUCTURES_DIR) / f"{document_id}.json"


def build_structure_payload(document_id: str, structure: dict[str, Any]) -> dict[str, Any]:
    """Project canonical parent/child chunks into the section-picker contract.

    PostgreSQL stores the complete parent/child records. This JSON remains the
    human-selectable section projection used by the existing API. Each section
    embeds its subsections (with titles) in document order, so the frontend can
    render the textbook hierarchy without a separate hierarchy query.
    """
    children = structure.get("children", [])
    sections = []
    for parent in structure.get("parents", []):
        if not str(parent.get("title") or "").strip():
            raise ValueError(
                f"Refusing to persist parent {parent.get('parent_id')} without a title"
            )
        parent_children = [
            c for c in children if c["parent_id"] == parent["parent_id"]
        ]
        child_ids = [c["child_id"] for c in parent_children]
        child_count = len(child_ids)
        assert child_count == len(child_ids), (
            f"child_count must equal len(child_ids) for parent "
            f"{parent['parent_id']}"
        )
        assert all(child_ids), (
            f"empty child_id found for parent {parent['parent_id']}"
        )
        assert len(set(child_ids)) == len(child_ids), (
            f"duplicate child_ids for parent {parent['parent_id']}"
        )
        # Single-child subsections are suppressed (title is None): they only
        # mirror the parent, so do not render a separate row for them.
        shown = [c for c in parent_children if c.get("title")]
        subsections = [
            {
                "child_id": c["child_id"],
                "title": c.get("title"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "book_heading": c.get("book_heading"),
                "heading_level": c.get("heading_level"),
                "content_role": c.get("content_role"),
                "source_items": c.get("source_items", []),
                "order": i,
            }
            for i, c in enumerate(shown)
        ]
        sections.append(
            {
                "parent_id": parent["parent_id"],
                "document_id": document_id,
                "chunk_type": "parent",
                "title": parent.get("title"),
                "parent_title": parent.get("parent_title", parent.get("title")),
                "subsection_titles": [
                    c.get("title") for c in parent_children
                ],
                "child_ids": child_ids,
                "child_count": child_count,
                "page_start": parent.get("page_start"),
                "page_end": parent.get("page_end"),
                "book_heading": parent.get("book_heading"),
                "heading_level": parent.get("heading_level"),
                "content_role": parent.get("content_role"),
                "source_items": parent.get("source_items", []),
                "subsections": subsections,
            }
        )

    return {
        "document_id": document_id,
        "domain_type": structure.get("domain_type"),
        "subject": structure.get("subject"),
        "section_count": len(sections),
        "child_count": len(structure.get("children", [])),
        "sections": sections,
    }


def save_structure(document_id: str, structure: dict[str, Any]) -> Path:
    """Persist parent structure metadata (sections) to JSON."""
    dir_path = Path(STRUCTURES_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = _path_for(document_id)
    payload = build_structure_payload(document_id, structure)

    t0 = time.perf_counter()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info(
        "Structure saved | document_id=%s | file=%s | sections=%d | children=%d | time=%.2fs",
        document_id,
        path.name,
        len(payload["sections"]),
        payload["child_count"],
        time.perf_counter() - t0,
    )
    return path


def load_structure(document_id: str) -> dict[str, Any]:
    path = _path_for(document_id)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
