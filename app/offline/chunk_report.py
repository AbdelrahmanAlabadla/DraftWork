from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.config import STRUCTURES_DIR, WORDS_PER_TOKEN
from app.logging_conf import get_logger
from app.offline.semantic_chunker import count_words

logger = get_logger("CHUNK_REPORT")


def _est_tokens(words: int) -> int:
    return int(words * WORDS_PER_TOKEN)


def save_chunk_report(document_id: str, chunks: dict[str, Any]) -> Path:
    """Write a human-readable report of chunking + titles for evaluation.

    Pure file output; does not alter the structure metadata used by the API.
    Written in the offline pipeline so the chunking quality can be inspected.
    """
    dir_path = Path(STRUCTURES_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{document_id}.chunks.txt"

    parents = chunks.get("parents", [])
    children = chunks.get("children", [])
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for child in children:
        children_by_parent.setdefault(child.get("parent_id"), []).append(child)

    lines: list[str] = []
    rule = "=" * 72
    thin = "-" * 72
    lines.append(rule)
    lines.append(f"CHUNKING REPORT | document_id={document_id}")
    lines.append(f"parents={len(parents)} | children={len(children)}")
    lines.append(rule)

    for parent in parents:
        pid = parent.get("parent_id")
        psubs = children_by_parent.get(pid, [])
        words = count_words(parent.get("content") or "")
        lines.append("")
        lines.append(rule)
        lines.append(
            f"SECTION {parent.get('title') or 'UNTITLED'} | id={pid} "
            f"| pages={parent.get('page_start')}-{parent.get('page_end')} "
            f"| words={words} | est_tokens={_est_tokens(words)} "
            f"| subsections={len(psubs)}"
        )
        lines.append(thin)
        lines.append(parent.get("content") or "")
        lines.append("")

        for i, child in enumerate(psubs, 1):
            cwords = count_words(child.get("content") or "")
            lines.append(thin)
            lines.append(
                f"  SUBSECTION {i}/{len(psubs)} {child.get('title') or 'UNTITLED'} "
                f"| id={child.get('child_id')} "
                f"| pages={child.get('page_start')}-{child.get('page_end')} "
                f"| words={cwords} | est_tokens={_est_tokens(cwords)}"
            )
            lines.append(thin)
            lines.append(child.get("content") or "")
            lines.append("")

    t0 = time.perf_counter()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info(
        "Chunk report written | document_id=%s | file=%s | parents=%d | children=%d | time=%.3fs",
        document_id,
        path.name,
        len(parents),
        len(children),
        time.perf_counter() - t0,
    )
    return path