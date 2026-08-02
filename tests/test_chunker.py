from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.offline.chunker import build_hierarchical_chunks  # noqa: E402


def _pages() -> list[dict]:
    body = "This is a long enough paragraph that describes the topic in detail. " * 10
    return [
        {
            "page": 1,
            "items": [
                {"type": "heading", "value": "Chapter 1: Intro"},
                {"type": "text", "value": body},
                {"type": "text", "value": body},
            ],
        },
        {
            "page": 2,
            "items": [
                {"type": "heading", "value": "Section 1.1 Details"},
                {"type": "text", "value": body},
                {"type": "text", "value": body},
            ],
        },
    ]


def test_build_hierarchical_chunks_produces_parents_and_children():
    result = build_hierarchical_chunks(_pages(), "doc-1")
    assert result["parents"]
    assert result["children"]
    for parent in result["parents"]:
        assert parent["document_id"] == "doc-1"
        assert parent["parent_id"]
    for child in result["children"]:
        assert child["document_id"] == "doc-1"
        assert child["parent_id"]
        assert child["content"]
        assert child["heading"]


def test_children_reference_existing_parents():
    result = build_hierarchical_chunks(_pages(), "doc-1")
    parent_ids = {p["parent_id"] for p in result["parents"]}
    for child in result["children"]:
        assert child["parent_id"] in parent_ids


def test_headings_are_captured():
    result = build_hierarchical_chunks(_pages(), "doc-1")
    headings = {p["heading"] for p in result["parents"] if p["heading"]}
    assert "Chapter 1: Intro" in headings
    assert "Section 1.1 Details" in headings
