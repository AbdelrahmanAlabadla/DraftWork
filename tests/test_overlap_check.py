from __future__ import annotations

from app.offline.semantic_chunker import verify_chunk_invariants


def _chunks(parents: list[dict], children: list[dict]) -> dict:
    return {"parents": parents, "children": children}


def _parent(pid: str, content: str) -> dict:
    return {
        "parent_id": pid,
        "document_id": "d",
        "title": None,
        "page_start": 1,
        "page_end": 1,
        "content": content,
    }


def _child(cid: str, pid: str, content: str) -> dict:
    return {
        "child_id": cid,
        "parent_id": pid,
        "document_id": "d",
        "title": None,
        "heading": None,
        "page_start": 1,
        "page_end": 1,
        "content": content,
    }


def test_no_warnings_for_disjoint_parents_and_one_sentence_child_overlap():
    chunks = _chunks(
        parents=[
            _parent("p1", "One two three."),
            _parent("p2", "Four five six."),
        ],
        children=[
            _child("c1", "p1", "One two three four."),
            _child("c2", "p1", "One two three four. Five six seven."),
        ],
    )
    assert verify_chunk_invariants(chunks) == []


def test_warns_when_parents_overlap():
    chunks = _chunks(
        parents=[
            _parent("p1", "One two three."),
            _parent("p2", "One two three. Four five six."),
        ],
        children=[],
    )
    warnings = verify_chunk_invariants(chunks)
    assert any("Parent overlap" in w for w in warnings)


def test_warns_when_child_overlap_exceeds_one_sentence():
    chunks = _chunks(
        parents=[_parent("p1", "A B C. D E F. G H I.")],
        children=[
            _child("c1", "p1", "A B C. D E F."),
            _child("c2", "p1", "A B C. D E F. G H I."),
        ],
    )
    warnings = verify_chunk_invariants(chunks)
    assert any("exceeds one sentence" in w for w in warnings)


def test_no_warnings_for_single_sentence_child_overlap():
    chunks = _chunks(
        parents=[_parent("p1", "A B C. D E F. G H I.")],
        children=[
            _child("c1", "p1", "A B C."),
            _child("c2", "p1", "A B C. D E F."),
        ],
    )
    assert verify_chunk_invariants(chunks) == []