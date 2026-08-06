from __future__ import annotations

import time

from app.logging_conf import get_logger
from app.offline.embeddings import embed_texts
from app.offline.vector_store import VectorStore
from app.online.graph import ExamState

logger = get_logger("RETRIEVAL")


def _query_for(question_type: str, count: int) -> str:
    labels = {
        "mcq": "multiple choice questions",
        "true_false": "true or false statements",
        "short_answer": "short answer questions",
    }
    return (
        f"Generate {count} {labels.get(question_type, 'exam')} about the key "
        f"concepts, definitions, facts, and explanations in this document."
    )


def _top_k_for(count: int) -> int:
    return min(12, max(4, (count + 2) // 2))


def retrieve_context(state: ExamState) -> dict:
    t0 = time.perf_counter()
    document_id = state["document_id"]
    qtype = state["question_type"]
    count = state["number_of_questions"]
    top_k = _top_k_for(count)

    logger.info("Retrieval started | document_id=%s | top_k=%d", document_id, top_k)

    query = _query_for(qtype, count)
    embeddings = embed_texts([query])
    dense = embeddings[0]["dense"]
    sparse = embeddings[0]["sparse"]

    selected_child_ids = state.get("selected_child_ids") or None
    chunks = VectorStore().hybrid_search(
        dense=dense,
        sparse=sparse,
        document_id=document_id,
        top_k=top_k,
        selected_child_ids=selected_child_ids,
    )
    if not chunks:
        logger.warning(
            "Retrieval returned 0 chunks | document_id=%s — document may not be indexed",
            document_id,
        )
        return {
            "retrieved_chunks": [],
            "context": "",
            "error": f"No content found for document {document_id}. Upload and index it first.",
        }

    context_parts = []
    for chunk in chunks:
        heading = f"[{chunk.get('heading')}] " if chunk.get("heading") else ""
        context_parts.append(f"{heading}{chunk['content']}")
    context = "\n\n".join(context_parts)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Retrieval completed | chunks=%d | top_k=%d | time=%.2fs",
        len(chunks),
        top_k,
        elapsed,
    )

    return {
        "retrieved_chunks": chunks,
        "context": context,
        "error": None,
    }
