from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path

from app.jobs import tasks
from app.api import job_routes
from app.offline.structure_store import build_structure_payload


def _canonical_structure() -> dict:
    return {
        "domain_type": "stem",
        "subject": "math",
        "parents": [{"parent_id": "p1", "title": "Linear equations"}],
        "children": [{
            "child_id": "c1", "parent_id": "p1", "title": "Substitution"
        }],
    }


def test_section_projection_matches_frontend_contract():
    payload = build_structure_payload("doc-1", _canonical_structure())

    assert payload["document_id"] == "doc-1"
    assert payload["sections"][0]["child_ids"] == ["c1"]
    assert payload["sections"][0]["subsections"][0]["title"] == "Substitution"


def test_ingestion_publishes_section_projection_not_raw_chunks(monkeypatch, tmp_path):
    captured = {}

    class Storage:
        @contextmanager
        def materialize(self, _key):
            yield tmp_path / "source.pdf"

        def put_json(self, key, value):
            captured["key"] = key
            captured["value"] = value

    (tmp_path / "source.pdf").write_bytes(b"%PDF-smoke")
    monkeypatch.setattr(tasks.repositories, "claim_job", lambda *_args: {
        "claim_generation": 3, "document_id": "doc-1"
    })
    monkeypatch.setattr(tasks.repositories, "get_document", lambda *_args: {
        "id": "doc-1", "session_id": "session-1", "source_storage_key": "source.pdf",
        "sha256": "abc",
    })
    monkeypatch.setattr(tasks.repositories, "update_job_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks.repositories, "job_is_cancelled", lambda *_a, **_k: False)
    monkeypatch.setattr(tasks.repositories, "complete_ingestion_job", lambda *_a, **_k: True)
    monkeypatch.setattr(tasks, "get_file_storage", lambda: Storage())
    monkeypatch.setattr(tasks, "validate_pdf", lambda *_args: None)
    monkeypatch.setattr(tasks, "_heartbeat", lambda *_args: nullcontext())
    monkeypatch.setattr(tasks, "run_pipeline", lambda *_a, **_k: {
        "document_id": "doc-1", "structure_data": _canonical_structure()
    })

    result = tasks.ingest_document.run("job-1")

    assert result == {"job_id": "job-1", "status": "completed"}
    assert "sections" in captured["value"]
    assert "parents" not in captured["value"]
    assert captured["value"]["sections"][0]["child_ids"] == ["c1"]


def test_document_api_adapts_raw_structure_from_affected_existing_uploads(monkeypatch):
    class Storage:
        def read_json(self, _key):
            return _canonical_structure()

    monkeypatch.setattr(job_routes, "get_file_storage", lambda: Storage())

    payload = job_routes._section_structure({"id": "doc-1"}, "structure.json")

    assert payload["sections"][0]["child_ids"] == ["c1"]


def test_frontend_guards_stale_section_responses_and_shows_retry_error():
    root = Path(__file__).parents[1]
    upload = (root / "FrontEnd/js/upload.js").read_text(encoding="utf-8")
    topics = (root / "FrontEnd/js/topics.js").read_text(encoding="utf-8")
    html = (root / "FrontEnd/index.html").read_text(encoding="utf-8")

    assert "version !== documentRequestVersion" in upload
    assert "documentId !== state.currentDocId" in upload
    assert "preserveSelection: true" in upload
    assert "showSectionsError" in topics
    assert 'id="retrySectionsBtn"' in html
