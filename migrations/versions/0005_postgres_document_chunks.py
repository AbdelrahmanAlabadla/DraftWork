"""Store processed parent and child chunks in PostgreSQL.

Revision ID: 0005_postgres_document_chunks
Revises: 0004_exam_lifecycle_audit
"""

from alembic import op


revision = "0005_postgres_document_chunks"
down_revision = "0004_exam_lifecycle_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """CREATE TABLE document_chunks (
               chunk_id TEXT PRIMARY KEY,
               document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
               session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
               index_job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
               parent_chunk_id TEXT,
               section_id TEXT,
               subsection_id TEXT,
               chunk_type TEXT NOT NULL CHECK (chunk_type IN ('parent', 'child')),
               title TEXT,
               parent_title TEXT,
               content TEXT NOT NULL DEFAULT '',
               chunk_order INTEGER NOT NULL,
               page_start INTEGER,
               page_end INTEGER,
               source_reference TEXT,
               processed_document_key TEXT,
               processing_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
               metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
               created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
               updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
               UNIQUE (document_id, chunk_id)
           )"""
    )
    op.execute("CREATE INDEX document_chunks_document_order_idx ON document_chunks (document_id, chunk_type, chunk_order)")
    op.execute("CREATE INDEX document_chunks_section_idx ON document_chunks (document_id, section_id, chunk_order)")
    op.execute("CREATE INDEX document_chunks_subsection_idx ON document_chunks (document_id, subsection_id)")
    op.execute("CREATE INDEX document_chunks_parent_idx ON document_chunks (document_id, parent_chunk_id, chunk_order)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS document_chunks")
