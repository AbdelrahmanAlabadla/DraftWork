from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from app import db
from app.logging_conf import get_logger


logger = get_logger("CHUNK_STORE")


class ChunkStore:
    """PostgreSQL-backed parent/child chunk persistence and exact retrieval."""

    def replace_document(
        self,
        structure: dict[str, Any],
        *,
        document_id: str,
        session_id: str | None = None,
        job_id: str | None = None,
        processed_document_key: str | None = None,
        processing_versions: dict[str, Any] | None = None,
    ) -> int:
        parents = list(structure.get("parents") or [])
        children = list(structure.get("children") or [])
        parent_by_id = {str(row.get("parent_id") or ""): row for row in parents}
        with db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                for order, parent in enumerate(parents):
                    parent_id = str(parent.get("parent_id") or "")
                    cursor.execute(
                        """INSERT INTO document_chunks (
                               chunk_id, document_id, session_id, index_job_id,
                               parent_chunk_id, section_id, subsection_id, chunk_type,
                               title, parent_title, content, chunk_order, page_start,
                               page_end, source_reference, processed_document_key,
                               processing_versions, metadata
                           ) VALUES (
                               %s,%s,%s,%s,NULL,%s,NULL,'parent',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                           )""",
                        (
                            parent_id, document_id, session_id, job_id, parent_id,
                            parent.get("title"), parent.get("parent_title") or parent.get("title"),
                            parent.get("content") or "", int(parent.get("parent_order") or order),
                            parent.get("page_start"), parent.get("page_end"),
                            parent.get("source_reference"), processed_document_key,
                            Jsonb(processing_versions or {}), Jsonb(parent),
                        ),
                    )
                for order, child in enumerate(children):
                    child_id = str(child.get("child_id") or child.get("chunk_id") or "")
                    parent_id = str(child.get("parent_id") or "") or None
                    parent = parent_by_id.get(parent_id or "", {})
                    cursor.execute(
                        """INSERT INTO document_chunks (
                               chunk_id, document_id, session_id, index_job_id,
                               parent_chunk_id, section_id, subsection_id, chunk_type,
                               title, parent_title, content, chunk_order, page_start,
                               page_end, source_reference, processed_document_key,
                               processing_versions, metadata
                           ) VALUES (
                               %s,%s,%s,%s,%s,%s,%s,'child',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                           )""",
                        (
                            child_id, document_id, session_id, job_id, parent_id,
                            parent_id, child_id, child.get("title") or child.get("chunk_title"),
                            child.get("parent_title") or parent.get("title"),
                            child.get("content") or "", int(child.get("child_order") or order),
                            child.get("page_start") or child.get("page"), child.get("page_end"),
                            child.get("source_reference"), processed_document_key,
                            Jsonb(processing_versions or {}), Jsonb(child),
                        ),
                    )
            conn.commit()
        logger.info("Document chunks stored | document_id=%s | parents=%d | children=%d", document_id, len(parents), len(children))
        return len(children)

    def get_by_child_ids(
        self,
        document_id: str,
        child_ids: list[str],
        *,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not child_ids:
            return []
        clauses = ["document_id = %s", "chunk_type = 'child'", "chunk_id = ANY(%s)"]
        params: list[Any] = [document_id, child_ids]
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        if job_id:
            clauses.append("index_job_id = %s")
            params.append(job_id)
        with db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""SELECT chunk_id, parent_chunk_id, title, parent_title, content,
                               chunk_order, page_start, page_end, source_reference, metadata
                          FROM document_chunks WHERE {' AND '.join(clauses)}""",
                    tuple(params),
                )
                rows = cursor.fetchall()
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            metadata = dict(row[9] or {})
            by_id[str(row[0])] = {
                **metadata,
                "child_id": str(row[0]), "chunk_id": str(row[0]),
                "parent_id": str(row[1]) if row[1] else None,
                "chunk_title": row[2], "title": row[2], "parent_title": row[3],
                "content": row[4] or "", "child_order": row[5],
                "page": row[6], "page_start": row[6], "page_end": row[7],
                "source_reference": row[8],
            }
        return [by_id[chunk_id] for chunk_id in child_ids if chunk_id in by_id]

    def delete_document(self, document_id: str, **_kwargs: Any) -> int:
        with db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                count = cursor.rowcount
            conn.commit()
        return max(0, count)

    def count_document_chunks(self, document_id: str) -> int:
        with db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM document_chunks WHERE document_id = %s AND chunk_type = 'child'",
                    (document_id,),
                )
                return int(cursor.fetchone()[0])
