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


def save_structure(document_id: str, structure: dict[str, Any]) -> Path:
    """Persist parent structure metadata (sections) to JSON.

    Qdrant stores only child (subsection) embeddings. Parents live here as the
    human-selectable section list for later retrieval scoping. Each section
    embeds its subsections (with titles) in document order, so the frontend can
    render the textbook hierarchy without querying Qdrant.
    """
    dir_path = Path(STRUCTURES_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = _path_for(document_id)

    children = structure.get("children", [])
    sections = []
    for parent in structure.get("parents", []):
        parent_children = [
            c for c in children if c["parent_id"] == parent["parent_id"]
        ]
        # Single-child subsections are suppressed (title is None): they only
        # mirror the parent, so do not render a separate row for them.
        shown = [c for c in parent_children if c.get("title")]
        subsections = [
            {
                "child_id": c["child_id"],
                "title": c.get("title"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "order": i,
            }
            for i, c in enumerate(shown)
        ]
        sections.append(
            {
                "parent_id": parent["parent_id"],
                "title": parent.get("title"),
                "child_ids": [c["child_id"] for c in parent_children],
                "child_count": len(subsections),
                "page_start": parent.get("page_start"),
                "page_end": parent.get("page_end"),
                "subsections": subsections,
            }
        )

    payload = {
        "document_id": document_id,
        "section_count": len(sections),
        "child_count": len(structure.get("children", [])),
        "sections": sections,
    }

    t0 = time.perf_counter()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info(
        "Structure saved | document_id=%s | file=%s | sections=%d | children=%d | time=%.2fs",
        document_id,
        path.name,
        len(sections),
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