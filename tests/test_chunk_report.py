from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.offline import chunk_report  # noqa: E402


def _sample_chunks() -> dict:
    return {
        "parents": [
            {"parent_id": "p1", "title": "Machine Learning Applications", "page_start": 1, "page_end": 2, "content": "Machine learning powers image classification and speech recognition systems."},
            {"parent_id": "p2", "title": "Regression Tasks", "page_start": 3, "page_end": 4, "content": "Linear regression predicts continuous numeric targets from features."},
        ],
        "children": [
            {"child_id": "c1", "parent_id": "p1", "title": "Image Classification", "page_start": 1, "page_end": 1, "content": "Convolutional networks classify images into discrete categories."},
            {"child_id": "c2", "parent_id": "p1", "title": "Speech Recognition", "page_start": 2, "page_end": 2, "content": "Acoustic models transcribe spoken audio into text sequences."},
            {"child_id": "c3", "parent_id": "p2", "title": "Linear Regression", "page_start": 3, "page_end": 4, "content": "Regression predicts a continuous output from input variables."},
        ],
    }


def test_save_chunk_report_writes_readable_txt(monkeypatch, tmp_path):
    monkeypatch.setattr(chunk_report, "STRUCTURES_DIR", str(tmp_path))

    path = chunk_report.save_chunk_report("doc-1", _sample_chunks())
    assert path.exists()
    assert path.name == "doc-1.chunks.txt"

    text = path.read_text(encoding="utf-8")
    assert "CHUNKING REPORT | document_id=doc-1" in text
    assert "parents=2 | children=3" in text
    assert "SECTION Machine Learning Applications | id=p1" in text
    assert "SUBSECTION 1/2 Image Classification | id=c1" in text
    assert "SUBSECTION 2/2 Speech Recognition | id=c2" in text
    assert "Machine learning powers image classification" in text
    assert "Convolutional networks classify images" in text
    assert "words=" in text and "est_tokens=" in text
    assert text.index("Machine Learning Applications") < text.index("Regression Tasks")
