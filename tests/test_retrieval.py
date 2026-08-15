from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.online.retrieval import _build_tree_context, retrieve_context, retrieve_selected


def _child(child_id, parent_title, chunk_title, content, parent_id="p"):
    return {
        "child_id": child_id,
        "parent_id": parent_id,
        "parent_title": parent_title,
        "chunk_title": chunk_title,
        "page": 1,
        "content": content,
    }


def test_tree_context_groups_subsections_under_one_section():
    children = [
        _child("c1", "Scientific Method", "Natural Sciences", "Body A."),
        _child("c2", "Scientific Method", "Biology Scope", "Body B."),
    ]
    ctx = _build_tree_context(children)
    assert ctx.count("## Scientific Method") == 1
    assert "### Natural Sciences" in ctx
    assert "Body A." in ctx
    assert "### Biology Scope" in ctx
    assert "Body B." in ctx
    assert ctx.index("## Scientific Method") < ctx.index("### Natural Sciences")
    assert ctx.index("### Natural Sciences") < ctx.index("Body A.")
    assert ctx.index("Body A.") < ctx.index("### Biology Scope")


def test_tree_context_single_child_without_title_renders_section_only():
    children = [
        _child("c1", "Evolutionary History", None, "Only body."),
    ]
    ctx = _build_tree_context(children)
    assert "## Evolutionary History" in ctx
    assert "###" not in ctx
    assert "Only body." in ctx


def test_tree_context_new_section_emits_new_header():
    children = [
        _child("c1", "Section A", "Sub A", "Body A.", parent_id="pa"),
        _child("c2", "Section B", "Sub B", "Body B.", parent_id="pb"),
    ]
    ctx = _build_tree_context(children)
    assert ctx.count("## Section A") == 1
    assert ctx.count("## Section B") == 1
    assert ctx.index("## Section A") < ctx.index("## Section B")


def test_tree_context_missing_parent_title_falls_back():
    children = [
        _child("c1", None, "Some Sub", "Body."),
    ]
    ctx = _build_tree_context(children)
    assert "## Untitled" in ctx
    assert "### Some Sub" in ctx


def test_retrieve_context_needs_selection(monkeypatch):
    # No selection -> error, no store call.
    calls = {"n": 0}

    class Store:
        def get_by_child_ids(self, document_id, child_ids):
            calls["n"] += 1
            return []

    monkeypatch.setattr("app.online.retrieval.VectorStore", Store)
    out = retrieve_context({"document_id": "doc", "selected_child_ids": None})
    assert out["error"]
    assert calls["n"] == 0
    assert out["context"] == ""


def test_retrieve_selected_returns_structured_context(monkeypatch):
    class Store:
        def get_by_child_ids(self, document_id, child_ids):
            assert child_ids == ["c1", "c2"]
            return [
                _child("c1", "Section A", "Sub A", "Body A."),
                _child("c2", "Section A", "Sub B", "Body B."),
            ]

    monkeypatch.setattr("app.online.retrieval.VectorStore", Store)
    out = retrieve_selected("doc", ["c1", "c2"])
    assert out["error"] is None
    assert len(out["retrieved_chunks"]) == 2
    assert "## Section A" in out["context"]
    assert "### Sub A" in out["context"]
    assert "Body A." in out["context"]


def test_retrieve_selected_no_chunks_returns_error(monkeypatch):
    class Store:
        def get_by_child_ids(self, document_id, child_ids):
            return []

    monkeypatch.setattr("app.online.retrieval.VectorStore", Store)
    out = retrieve_selected("doc", ["missing"])
    assert out["error"]
    assert out["context"] == ""


def test_retrieve_context_passes_selected_ids(monkeypatch):
    class Store:
        def get_by_child_ids(self, document_id, child_ids):
            return [_child(cid, "S", None, f"Body {cid}.") for cid in child_ids]

    monkeypatch.setattr("app.online.retrieval.VectorStore", Store)
    out = retrieve_context({
        "document_id": "doc",
        "question_type": "mcq",
        "number_of_questions": 2,
        "selected_child_ids": ["x1", "x2"],
    })
    assert out["error"] is None
    assert out["context"].count("## S") == 1
    assert "## S" in out["context"]