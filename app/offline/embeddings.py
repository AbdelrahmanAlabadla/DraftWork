from __future__ import annotations

import logging
import threading
import time
from typing import Any

import torch

from app.config import EMBEDDING_MODEL
from app.logging_conf import get_logger

logger = get_logger("EMBEDDING")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_model_lock = threading.Lock()


def device_name() -> str:
    return _DEVICE


def get_model() -> Any:
    """Lazily load the BGE-M3 model once (on GPU with fp16 when available)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from FlagEmbedding import BGEM3FlagModel

                t0 = time.perf_counter()
                logger.info(
                    "Loading embedding model | model=%s | device=%s | fp16=%s",
                    EMBEDDING_MODEL,
                    _DEVICE,
                    _DEVICE == "cuda",
                )
                _model = BGEM3FlagModel(
                    EMBEDDING_MODEL,
                    use_fp16=(_DEVICE == "cuda"),
                    device=_DEVICE,
                )
                logger.info(
                    "Embedding model loaded | time=%.2fs", time.perf_counter() - t0
                )
    return _model


def embed_texts(
    texts: list[str], batch_size: int = 16
) -> list[dict[str, Any]]:
    """Embed a list of texts -> [{"dense": [...], "sparse": {token_id: weight}}]."""
    if not texts:
        return []

    model = get_model()
    t0 = time.perf_counter()
    logger.info(
        "Embedding generation started | texts=%d | batch_size=%d",
        len(texts),
        batch_size,
    )

    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense = out["dense_vecs"]
    sparse = out.get("lexical_weights") or out.get("sparse_vecs")

    results: list[dict[str, Any]] = []
    for i in range(len(texts)):
        results.append(
            {
                "dense": dense[i].tolist(),
                "sparse": dict(sparse[i]) if sparse is not None else {},
            }
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Embedding generation completed | vectors=%d | time=%.2fs", len(results), elapsed
    )
    return results
