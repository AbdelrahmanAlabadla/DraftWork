from __future__ import annotations

import time
from copy import deepcopy

from app.config import GENERATION_CONTEXT_TOKENS, PLANNER_CONTEXT_TOKENS
from app.language import detect_language
from app.logging_conf import get_logger
from app.offline.chunk_store import ChunkStore
from app.offline.processed_cache import load_processed_document_by_key, materialize_for_document
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
            child_id = str(child.get("child_id") or "").strip()
            marker = f"[source_chunk_id={child_id}]" if child_id else ""
            parts.append(f"{marker}\n{content}".strip())

    return "\n\n".join(parts)


def build_planner_context(
    children: list[dict], snippet_tokens: int | None = None
) -> str:
    """Return selected chunk content for the single full-context planning call.

    ``snippet_tokens`` remains accepted for compatibility but no longer truncates
    each selected chunk. The complete rendered selection has one deployment-wide
    cap so a request cannot exceed the configured prompt budget.
    """
    return _first_tokens(_build_tree_context(children), PLANNER_CONTEXT_TOKENS)


def retrieve_selected(
    document_id: str,
    selected_child_ids: list[str],
    *,
    session_id: str | None = None,
    job_id: str | None = None,
    processed_document_key: str | None = None,
) -> dict:
    """Pull the exact child chunks the user selected (by id, no search).

    Returns the selected tree context for question generation and planning.
    """
    t0 = time.perf_counter()
    store = ChunkStore()
    if session_id or job_id:
        children = store.get_by_child_ids(
            document_id,
            selected_child_ids,
            session_id=session_id,
            job_id=job_id,
        )
    else:
        # Preserve the public pipeline contract for local callers and tests.
        children = store.get_by_child_ids(document_id, selected_child_ids)

    returned_ids = {str(child.get("child_id") or "") for child in children}
    missing_ids = [value for value in selected_child_ids if value not in returned_ids]
    if missing_ids and processed_document_key:
        cached = load_processed_document_by_key(processed_document_key)
        if cached:
            cached_structure = cached["structure"]
            cached_document_ids = {
                str(row.get("document_id") or "")
                for row in [
                    *(cached_structure.get("parents") or []),
                    *(cached_structure.get("children") or []),
                ]
                if row.get("document_id")
            }
            structure = (
                deepcopy(cached_structure)
                if cached_document_ids == {document_id}
                else materialize_for_document(cached_structure, document_id)
            )
            cached_by_id = {
                str(child.get("child_id") or ""): child
                for child in structure.get("children") or [] if child.get("child_id")
            }
            recoverable = [cached_by_id[value] for value in missing_ids if value in cached_by_id]
            if recoverable:
                store.replace_document(
                    structure, document_id=document_id, session_id=session_id,
                    job_id=job_id, processed_document_key=processed_document_key,
                    processing_versions=cached.get("processing_versions") or {},
                )
                children = store.get_by_child_ids(
                    document_id, selected_child_ids, session_id=session_id, job_id=job_id,
                )
                logger.info(
                    "Backfilled current cached document chunks into PostgreSQL | "
                    "document_id=%s | recovered=%d", document_id, len(recoverable),
                )

    by_id = {str(child.get("child_id") or ""): child for child in children}
    children = [by_id[value] for value in selected_child_ids if value in by_id]
    missing_ids = [value for value in selected_child_ids if value not in by_id]

    if missing_ids:
        logger.warning(
            "Selection retrieval incomplete | document_id=%s | selected=%d | missing=%s",
            document_id, len(selected_child_ids), missing_ids,
        )
        return {
            "retrieved_chunks": [],
            "context": "",
            "planner_context": "",
            "document_language": "en",
            "error": (
                "Selected source evidence is incomplete for the current processed document. "
                f"Missing chunk IDs: {', '.join(missing_ids)}. Re-index the document before generating."
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
    session_id = state.get("session_id")
    index_job_id = state.get("index_job_id")

    if selected_child_ids:
        return retrieve_selected(
            document_id,
            selected_child_ids,
            session_id=session_id,
            job_id=index_job_id,
            processed_document_key=state.get("processed_document_key"),
        )

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
