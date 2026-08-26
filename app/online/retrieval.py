from __future__ import annotations

import time

from app.config import (
    GENERATION_CONTEXT_TOKENS,
    PLANNER_CONTEXT_TOKENS,
    PLANNER_SNIPPET_TOKENS,
)
from app.language import detect_language
from app.logging_conf import get_logger
from app.offline.vector_store import VectorStore
from app.online.graph import ExamState

logger = get_logger("RETRIEVAL")


def _first_tokens(text: str, max_tokens: int) -> str:
    """Return the first ``max_tokens`` whitespace-delimited tokens of ``text``."""
    if max_tokens <= 0:
        return ""
    tokens = str(text).split()
    return " ".join(tokens[:max_tokens])


def _build_tree_context(children: list[dict]) -> str:
    """Render the selected chunks as a Section -> Subsection tree in order.

    Only the titles available in the project are used: parent_title (##) and
    chunk_title (###). A chunk without a subsection title (single-subsection
    parents where the text lives in that one child) renders only its section
    header. A missing parent title falls back to "Untitled".
    """
    parts: list[str] = []
    current_key: str | None = None
    for child in children:
        section_key = child.get("parent_id") or child.get("parent_title")
        section_label = child.get("parent_title") or "Untitled"
        subtitle = child.get("chunk_title")

        if section_key is None or section_key != current_key:
            current_key = section_key
            parts.append(f"## {section_label}")

        if subtitle:
            parts.append(f"### {subtitle}")

        content = child.get("content", "")
        if content:
            parts.append(content)

    return "\n\n".join(parts)


def build_planner_context(
    children: list[dict], snippet_tokens: int = PLANNER_SNIPPET_TOKENS
) -> str:
    """Return a LIGHTWEIGHT context for planning.

    Per selected child chunk this is just the section title + chunk title plus a
    short snippet of its text (``snippet_tokens`` tokens). The planner only needs
    to know which concepts are available to distribute across models; it never
    writes questions, so it does not need the full chunk text.
    """
    parts: list[str] = []
    current_key: str | None = None
    for child in children:
        section_key = child.get("parent_id") or child.get("parent_title")
        section_label = child.get("parent_title") or "Untitled"
        subtitle = child.get("chunk_title")

        if section_key is None or section_key != current_key:
            current_key = section_key
            parts.append(f"## {section_label}")

        lines = [f"### {subtitle}"] if subtitle else []
        snippet = _first_tokens(child.get("content", ""), snippet_tokens)
        if snippet:
            lines.append(snippet)
        parts.append("\n".join(lines))

    joined = "\n\n".join(parts)
    return _first_tokens(joined, PLANNER_CONTEXT_TOKENS)


def retrieve_selected(
    document_id: str,
    selected_child_ids: list[str],
) -> dict:
    """Pull the exact child chunks the user selected (by id, no search).

    Returns both the FULL tree context (for question generation) and a lightweight
    planner context (titles + short snippets) used only by the planning phase.
    """
    t0 = time.perf_counter()
    children = VectorStore().get_by_child_ids(document_id, selected_child_ids)

    if not children:
        logger.warning(
            "Selection retrieval returned 0 chunks | document_id=%s | selected=%d",
            document_id,
            len(selected_child_ids),
        )
        return {
            "retrieved_chunks": [],
            "context": "",
            "planner_context": "",
            "document_language": "en",
            "error": (
                f"No stored content found for the selected sections of document "
                f"{document_id}. Upload and index it first."
            ),
        }

    context = _first_tokens(_build_tree_context(children), GENERATION_CONTEXT_TOKENS)
    planner_context = build_planner_context(children)
    document_language = detect_language(context or planner_context)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Selection retrieval completed | chunks=%d | selected=%d | time=%.2fs",
        len(children),
        len(selected_child_ids),
        elapsed,
    )
    return {
        "retrieved_chunks": children,
        "context": context,
        "planner_context": planner_context,
        "document_language": document_language,
        "error": None,
    }


def retrieve_context(state: ExamState) -> dict:
    """Route retrieval based on whether the user selected sections.

    V1 requires an explicit topic selection: we PULL the exact selected child
    chunks by id and do NOT run a similarity search (no RRF/reranker here).
    When nothing is selected we do not fall back to searching the whole
    document — we return an actionable error instead.
    """
    document_id = state["document_id"]
    selected_child_ids = state.get("selected_child_ids") or None

    if selected_child_ids:
        return retrieve_selected(document_id, selected_child_ids)

    logger.warning(
        "No sections selected | document_id=%s — refusing to search whole document",
        document_id,
    )
    return {
        "retrieved_chunks": [],
        "context": "",
        "error": (
            "Please choose at least one section topic to generate the exam. "
            "No content was selected."
        ),
    }