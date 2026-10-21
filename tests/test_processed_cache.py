from __future__ import annotations
import numpy as np

from app.offline import processed_cache
from app.offline import pipeline


def test_processed_key_tracks_file_and_every_processing_version(monkeypatch):
    original = processed_cache.processed_document_key("a" * 64)
    assert original != processed_cache.processed_document_key("b" * 64)
    monkeypatch.setattr(processed_cache.config, "PARSER_PIPELINE_VERSION", "parser-next")
    assert original != processed_cache.processed_document_key("a" * 64)


def test_processed_key_tracks_title_model_and_context(monkeypatch):
    original = processed_cache.processed_document_key("a" * 64)
    monkeypatch.setattr(processed_cache.config, "DEEPSEEK_MODEL", "deepseek-next")
    assert original != processed_cache.processed_document_key("a" * 64)

    monkeypatch.undo()
    original = processed_cache.processed_document_key("a" * 64)
    monkeypatch.setattr(processed_cache.config, "TITLE_CLASSIFICATION_CONTEXT_WORDS", 64)
    assert original != processed_cache.processed_document_key("a" * 64)


def test_cached_chunks_materialize_in_a_new_document_scope():
    structure = {
        "parents": [{"parent_id": "p1", "document_id": "old", "title": "Work"}],
        "children": [{
            "child_id": "c1", "chunk_id": "c1", "parent_id": "p1",
            "document_id": "old", "content": "work",
        }],
    }
    first = processed_cache.materialize_for_document(structure, "doc-a")
    second = processed_cache.materialize_for_document(structure, "doc-b")
    assert first["children"][0]["child_id"] != second["children"][0]["child_id"]
    assert first["children"][0]["document_id"] == "doc-a"
    assert first["children"][0]["parent_id"] == first["parents"][0]["parent_id"]


def test_processed_cache_serializes_numpy_embedding_scalars(monkeypatch, tmp_path):
    monkeypatch.setattr(
        processed_cache.config,
        "PROCESSED_DOCUMENT_CACHE_DIR",
        str(tmp_path / "cache"),
    )
    sha256 = "c" * 64

    processed_cache.save_processed_document(
        sha256,
        structure={"parents": [], "children": []},
        embeddings=[{
            "dense": [np.float16(0.25), np.float32(0.5)],
            "sparse": {np.int64(7): np.float16(0.75)},
        }],
        metadata={"total_pages": np.int64(1)},
    )

    cached = processed_cache.load_processed_document(sha256)

    assert cached is not None
    assert cached["embeddings"][0]["dense"] == [0.25, 0.5]
    assert cached["embeddings"][0]["sparse"] == {7: 0.75}
    assert cached["metadata"]["total_pages"] == 1


def test_pipeline_reuses_identical_pdf_content_across_document_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(processed_cache.config, "PROCESSED_DOCUMENT_CACHE_DIR", str(tmp_path / "cache"))
    pdf = tmp_path / "same.pdf"
    pdf.write_bytes(b"%PDF-1.4 identical content")
    calls = {"parse": 0, "store": 0}

    class Parser:
        def parse(self, _path):
            calls["parse"] += 1
            return [{"page_number": 1, "items": [{"type": "text", "text": "Work and energy"}]}]

    class Store:
        def replace_document(self, structure, **_kwargs):
            calls["store"] += 1
            return len(structure["children"])

    def structure(_pages, document_id, **_kwargs):
        return {
            "domain_type": "stem", "subject": "physics",
            "parents": [{"parent_id": "p1", "document_id": document_id, "title": "Work"}],
            "children": [{
                "child_id": "c1", "chunk_id": "c1", "parent_id": "p1",
                "document_id": document_id, "content": "Work and energy",
            }],
        }

    monkeypatch.setattr(pipeline, "LlamaParser", lambda: Parser())
    monkeypatch.setattr(pipeline, "clean_pages", lambda pages: pages)
    monkeypatch.setattr(pipeline, "build_semantic_structure", structure)
    monkeypatch.setattr(pipeline, "verify_chunk_invariants", lambda _chunks: [])
    monkeypatch.setattr(pipeline, "ChunkStore", Store)
    monkeypatch.setattr(pipeline, "save_parse_dump", lambda *_args: tmp_path / "parse.json")
    monkeypatch.setattr(pipeline, "save_chunk_report", lambda *_args: tmp_path / "report.json")
    monkeypatch.setattr(pipeline, "save_chunk_dump", lambda *_args: tmp_path / "chunks.json")
    monkeypatch.setattr(pipeline, "save_structure", lambda document_id, _chunks: tmp_path / f"{document_id}.json")

    first = pipeline.run_pipeline(pdf, "doc-first")
    second = pipeline.run_pipeline(pdf, "doc-later")

    assert first["processed_cache_hit"] is False
    assert second["processed_cache_hit"] is True
    assert first["processed_document_key"] == second["processed_document_key"]
    assert first["domain_type"] == second["domain_type"] == "stem"
    assert first["subject"] == second["subject"] == "physics"
    assert calls == {"parse": 1, "store": 2}
