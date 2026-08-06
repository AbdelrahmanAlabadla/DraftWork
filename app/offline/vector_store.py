from __future__ import annotations

import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import EMBEDDING_DIM, QDRANT_COLLECTION, QDRANT_URL
from app.logging_conf import get_logger

logger = get_logger("QDRANT")

_DENSE_NAME = ""  # default (unnamed) dense vector space
_SPARSE_NAME = "lexical"


class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection = QDRANT_COLLECTION

    def ensure_collection(self) -> None:
        t0 = time.perf_counter()
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    _DENSE_NAME: qm.VectorParams(
                        size=EMBEDDING_DIM, distance=qm.Distance.COSINE
                    )
                },
                sparse_vectors_config={_SPARSE_NAME: qm.SparseVectorParams()},
            )
            logger.info(
                "Collection created | name=%s | dim=%d | metric=COSINE | time=%.2fs",
                self.collection,
                EMBEDDING_DIM,
                time.perf_counter() - t0,
            )
        else:
            logger.info("Collection exists | name=%s", self.collection)

    def delete_document(self, document_id: str) -> int:
        """Remove all points belonging to a document (idempotent re-indexing)."""
        selector = qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="document_id", match=qm.MatchValue(value=document_id)
                    )
                ]
            )
        )
        try:
            result = self.client.delete(
                collection_name=self.collection,
                points_selector=selector,
            )
        except Exception:
            logger.warning(
                "delete_document | document_id=%s | no prior points to delete",
                document_id,
            )
            return 0
        count = getattr(result, "status", "ok")
        logger.info("Deleted previous vectors | document_id=%s | status=%s", document_id, count)
        return 1

    def upsert(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[dict[str, Any]],
    ) -> int:
        t0 = time.perf_counter()
        points: list[qm.PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            point_id = chunk["child_id"]
            payload = {
                "document_id": chunk["document_id"],
                "parent_id": chunk["parent_id"],
                "child_id": chunk["child_id"],
                "page": chunk.get("page"),
                "heading": chunk.get("heading"),
                "content": chunk["content"],
            }
            sparse = emb.get("sparse") or {}
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector={
                        _DENSE_NAME: emb["dense"],
                        _SPARSE_NAME: qm.SparseVector(
                            indices=sorted(sparse.keys()),
                            values=[sparse[k] for k in sorted(sparse.keys())],
                        ),
                    },
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection, points=points)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Vectors uploaded | collection=%s | count=%d | time=%.2fs",
            self.collection,
            len(points),
            elapsed,
        )
        return len(points)

    def hybrid_search(
        self,
        dense: list[float],
        sparse: dict[int, float],
        document_id: str,
        top_k: int = 6,
        selected_child_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        must = [
            qm.FieldCondition(
                key="document_id", match=qm.MatchValue(value=document_id)
            )
        ]
        if selected_child_ids:
            must.append(
                qm.FieldCondition(
                    key="child_id", match=qm.MatchAny(any=selected_child_ids)
                )
            )
        filter_ = qm.Filter(must=must)
        prefetch_k = top_k * 3
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(
                    query=dense,
                    using=_DENSE_NAME,
                    limit=prefetch_k,
                    filter=filter_,
                ),
                qm.Prefetch(
                    query=qm.SparseVector(
                        indices=sorted(sparse.keys()),
                        values=[sparse[k] for k in sorted(sparse.keys())],
                    ),
                    using=_SPARSE_NAME,
                    limit=prefetch_k,
                    filter=filter_,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        results = [
            {
                "child_id": p.payload.get("child_id"),
                "parent_id": p.payload.get("parent_id"),
                "page": p.payload.get("page"),
                "heading": p.payload.get("heading"),
                "content": p.payload.get("content", ""),
                "score": p.score,
            }
            for p in response.points
        ]
        elapsed = time.perf_counter() - t0
        logger.info(
            "Hybrid search | top_k=%d | document_id=%s | returned=%d | time=%.2fs",
            top_k,
            document_id,
            len(results),
            elapsed,
        )
        return results

    def count_documents(self, document_id: str) -> int:
        result = self.client.count(
            collection_name=self.collection,
            count_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="document_id", match=qm.MatchValue(value=document_id)
                    )
                ]
            ),
            exact=True,
        )
        return result.count
