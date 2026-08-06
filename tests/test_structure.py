from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.offline import structure_store  # noqa: E402


def _sample_structure() -> dict:
    return {
        "parents": [
            {"parent_id": "p1", "title": "Machine Learning Applications", "page_start": 1, "page_end": 4},
            {"parent_id": "p2", "title": "Regression Tasks", "page_start": 5, "page_end": 9},
        ],
        "children": [
            {"child_id": "c1", "parent_id": "p1", "title": "Machine Learning Techniques", "page_start": 1, "page_end": 2},
            {"child_id": "c2", "parent_id": "p1", "title": "Image Classification", "page_start": 3, "page_end": 4},
            {"child_id": "c3", "parent_id": "p2", "title": "Supervised Learning", "page_start": 5, "page_end": 7},
        ],
    }


def test_save_and_load_persists_subsection_titles(monkeypatch, tmp_path):
    monkeypatch.setattr(structure_store, "STRUCTURES_DIR", str(tmp_path))

    path = structure_store.save_structure("doc-1", _sample_structure())
    assert path.exists()

    loaded = structure_store.load_structure("doc-1")
    assert loaded["document_id"] == "doc-1"
    assert loaded["section_count"] == 2
    assert loaded["child_count"] == 3

    first = loaded["sections"][0]
    assert first["parent_id"] == "p1"
    assert first["title"] == "Machine Learning Applications"
    assert first["child_ids"] == ["c1", "c2"]

    subs = first["subsections"]
    assert [s["title"] for s in subs] == [
        "Machine Learning Techniques",
        "Image Classification",
    ]
    assert [s["child_id"] for s in subs] == ["c1", "c2"]
    assert [s["order"] for s in subs] == [0, 1]
    assert subs[0]["page_start"] == 1 and subs[0]["page_end"] == 2

    second = loaded["sections"][1]
    assert [s["child_id"] for s in second["subsections"]] == ["c3"]


def test_load_missing_document_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(structure_store, "STRUCTURES_DIR", str(tmp_path))
    assert structure_store.load_structure("nope") == {}
