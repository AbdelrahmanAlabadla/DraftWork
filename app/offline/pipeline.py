from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.logging_conf import get_logger, set_request_id
from app.offline.chunker import build_hierarchical_chunks
from app.offline.cleaner import clean_pages
from app.offline.embeddings import embed_texts
from app.offline.parser import LlamaParser
from app.offline.vector_store import VectorStore

logger = get_logger("PIPELINE")


class PipelineError(RuntimeError):
    pass


def run_pipeline(file_path: str | Path, document_id: str) -> dict[str, Any]:
    """Execute PDF parsing, cleaning, chunking, embedding, and indexing."""
    set_request_id(document_id)
    path = Path(file_path)
    if not path.exists():
        raise PipelineError(f"File not found: {path}")

    t_pipeline = time.perf_counter()
    timings: dict[str, float] = {}
    logger.info("=" * 70)
    logger.info("OFFLINE PIPELINE STARTED | document_id=%s | file=%s", document_id, path.name)

    try:
        # --- Stage 1: Parse ---------------------------------------------------
        t0 = time.perf_counter()
        raw_pages = LlamaParser().parse(path)
        timings["parsing"] = time.perf_counter() - t0

        # --- Stage 2: Clean & normalize for chunk generation ------------------
        t0 = time.perf_counter()
        pages = clean_pages(raw_pages)
        timings["cleaning"] = time.perf_counter() - t0
        if not pages:
            raise PipelineError("Cleaning removed all content")

        # --- Stage 3: Hierarchical chunking ------------------------------------
        t0 = time.perf_counter()
        chunks = build_hierarchical_chunks(pages, document_id)
        timings["chunking"] = time.perf_counter() - t0
        parents = chunks["parents"]
        children = chunks["children"]
        if not children:
            raise PipelineError("Chunking produced no children")

        # --- Stage 4: Embeddings ------------------------------------------------
        t0 = time.perf_counter()
        child_texts = [c["content"] for c in children]
        embeddings = embed_texts(child_texts)
        timings["embeddings"] = time.perf_counter() - t0
        if len(embeddings) != len(children):
            raise PipelineError(
                f"Embedding count mismatch: {len(embeddings)} vs {len(children)}"
            )

        # --- Stage 5: Store in Qdrant -------------------------------------------
        t0 = time.perf_counter()
        store = VectorStore()
        store.ensure_collection()
        store.delete_document(document_id)
        vectors_uploaded = store.upsert(children, embeddings)
        timings["qdrant_upload"] = time.perf_counter() - t0

    except Exception as exc:
        total = time.perf_counter() - t_pipeline
        logger.error(
            "OFFLINE PIPELINE FAILED | document_id=%s | stage_elapsed=%.2fs | exc=%s: %s",
            document_id,
            total,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(str(exc)) from exc

    total = time.perf_counter() - t_pipeline
    summary = {
        "document_id": document_id,
        "filename": path.name,
        "total_pages": len(pages),
        "total_parents": len(parents),
        "total_children": len(children),
        "vectors_stored": vectors_uploaded,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "total_time": round(total, 3),
    }

    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY | document_id=%s", document_id)
    logger.info("Upload ............ %-8s", "n/a")
    logger.info("Parsing ........... %.3f sec", timings.get("parsing", 0))
    logger.info("Cleaning .......... %.3f sec", timings.get("cleaning", 0))
    logger.info("Chunking .......... %.3f sec", timings.get("chunking", 0))
    logger.info("Embeddings ........ %.3f sec", timings.get("embeddings", 0))
    logger.info("Qdrant Upload ..... %.3f sec", timings.get("qdrant_upload", 0))
    logger.info("Total Pipeline .... %.3f sec", total)
    logger.info(
        "pages=%d | parents=%d | children=%d | vectors_stored=%d | status=SUCCESS",
        len(pages),
        len(parents),
        len(children),
        vectors_uploaded,
    )
    logger.info("=" * 70)
    return summary
