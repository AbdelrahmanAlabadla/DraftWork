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
            info = self.client.get_collection(self.collection)
            params_vectors = info.config.params.vectors
            if isinstance(params_vectors, dict):
                vparams = params_vectors.get(_DENSE_NAME)
            else:
                vparams = params_vectors
            cur_dim = getattr(vparams, "size", None)
            if cur_dim != EMBEDDING_DIM:
                self.client.delete_collection(self.collection)
                logger.warning(
                    "Collision in dim | collection=%s | cur_dim=%d | expected=%d | "
                    "recreating collection",
                    self.collection,
                    cur_dim,
                    EMBEDDING_DIM,
                )
                self.ensure_collection()
                return
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
                "parent_title": chunk.get("parent_title"),
                "child_id": chunk["child_id"],
                "chunk_id": chunk.get("chunk_id", chunk["child_id"]),
                "chunk_title": chunk.get("chunk_title"),
                "chunk_type": "child",
                "child_order": chunk.get("child_order"),
                "page": chunk.get("page"),
                "heading": chunk.get("heading"),
                "content": chunk["content"],
            }
            # Sparse (lexical) half is optional: only present for models that
            # produce one (e.g. BGE-M3). Dense-only models omit the field.
            vector: dict[str, Any] = {_DENSE_NAME: emb["dense"]}
            sparse = emb.get("sparse") or {}
            if sparse:
                vector[_SPARSE_NAME] = qm.SparseVector(
                    indices=sorted(sparse.keys()),
                    values=[sparse[k] for k in sorted(sparse.keys())],
                )
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
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
                key="chunk_type", match=qm.MatchValue(value="child")
            ),
            qm.FieldCondition(
                key="document_id", match=qm.MatchValue(value=document_id)
            ),
        ]
        if selected_child_ids:
            must.append(
                qm.FieldCondition(
                    key="child_id", match=qm.MatchAny(any=selected_child_ids)
                )
            )
        filter_ = qm.Filter(must=must)

        # Dense-only fallback: when the query has no sparse half (dense-only
        # models like GTE-multilingual), skip the hybrid fusion and search by
        # dense vector alone.
        if not sparse:
            response = self.client.query_points(
                collection_name=self.collection,
                query=dense,
                using=_DENSE_NAME,
                limit=top_k,
                query_filter=filter_,
                with_payload=True,
            )
        else:
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

    def get_by_child_ids(
        self,
        document_id: str,
        child_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Return the exact child (subsection) payloads for the given ids.

        This is a pure lookup — no similarity search, no RRF. It is used when
        the user picks specific sections/subsections from the tree and we want
        to feed exactly those chunks to the LLM in the requested order.
        """
        t0 = time.perf_counter()
        if not child_ids:
            return []

        filter_ = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="chunk_type", match=qm.MatchValue(value="child")
                ),
                qm.FieldCondition(
                    key="document_id", match=qm.MatchValue(value=document_id)
                ),
                qm.FieldCondition(
                    key="child_id", match=qm.MatchAny(any=list(child_ids))
                ),
            ]
        )
        response = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=filter_,
            limit=len(child_ids),
            with_payload=True,
        )
        points = response[0]

        by_id = {}
        for p in points:
            cid = p.payload.get("child_id")
            if cid is not None:
                by_id[cid] = {
                    "child_id": cid,
                    "parent_id": p.payload.get("parent_id"),
                    "parent_title": p.payload.get("parent_title"),
                    "chunk_title": p.payload.get("chunk_title"),
                    "page": p.payload.get("page"),
                    "content": p.payload.get("content", ""),
                }

        # Preserve the caller's document order even though scroll returns
        # points in an unspecified order.
        results = [by_id[cid] for cid in child_ids if cid in by_id]
        elapsed = time.perf_counter() - t0
        logger.info(
            "Fetch by child_ids | document_id=%s | requested=%d | found=%d | time=%.2fs",
            document_id,
            len(child_ids),
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
