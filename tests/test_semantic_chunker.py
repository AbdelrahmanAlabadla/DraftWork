from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.offline import semantic_chunker as sc  # noqa: E402
from app.offline.semantic_chunker import (  # noqa: E402
    ChildChunk,
    Paragraph,
    ParentChunk,
    Sentence,
    build_children,
    build_parents,
    build_semantic_structure,
    extract_paragraphs,
    split_parents,
    split_sentences,
)

E = [1.0, 0.0]  # concept A
F = [0.0, 1.0]  # concept B
G = [0.0, -1.0]  # concept C (distinct from E and F)


def _page(items: list[dict], page: int = 1) -> dict:
    return {"page": page, "items": items}


def _body(text: str) -> dict:
    return {"type": "text", "value": text}


def _table(text: str) -> dict:
    return {"type": "table", "value": text}


def _heading(text: str) -> dict:
    return {"type": "heading", "value": text}


def _patch_titles(monkeypatch):
    # The family-batch flow labels sections+subsections in one call per batch;
    # keep it deterministic and LLM-free in tests.
    monkeypatch.setattr("app.offline.semantic_chunker.make_title_client", lambda: object())
    monkeypatch.setattr(
        "app.offline.semantic_chunker.generate_family_batch_titles",
        lambda entries, client=None, *, before=None: [
            "ST" if level == "section" else "CT" for level, _ in entries
        ],
    )
    monkeypatch.setattr(
        "app.offline.semantic_chunker.is_acceptable_title",
        lambda title, content, level, used_titles: bool(title),
    )
    monkeypatch.setattr(
        "app.offline.semantic_chunker.regenerate_title",
        lambda client=None, content="", level="section", reject=None, used_titles=None: "RT",
    )


# ---------------------------------------------------------------------------
# paragraph extraction
# ---------------------------------------------------------------------------


def test_extract_paragraphs_includes_headings_preserves_order_and_pages():
    pages = [
        _page([_heading("Chapter 1"), _body("Alpha text."), _table("|a|b|")], page=1),
        _page([_body("Beta text.")], page=2),
    ]
    paras = extract_paragraphs(pages)
    assert [p.text for p in paras] == ["Chapter 1", "Alpha text.", "|a|b|", "Beta text."]
    assert paras[0].page == 1
    assert paras[-1].page == 2


# ---------------------------------------------------------------------------
# sentence splitting guards
# ---------------------------------------------------------------------------


def test_split_sentences_basic_and_abbreviations():
    assert split_sentences("A. B. C.") == ["A.", "B.", "C."]
    assert split_sentences("See Fig. 2. The model works.") == [
        "See Fig. 2.",
        "The model works.",
    ]
    assert split_sentences("Dr. Smith left.") == ["Dr. Smith left."]


def test_split_sentences_guards_decimal():
    assert split_sentences("Theta equals 4.91 exactly. Next sentence.") == [
        "Theta equals 4.91 exactly.",
        "Next sentence.",
    ]


def test_split_sentences_guards_figure_reference():
    parts = split_sentences("See Figure 1-1. for the plot. The result holds.")
    assert parts == ["See Figure 1-1. for the plot.", "The result holds."]


def test_split_sentences_keeps_real_boundary_after_figure():
    # "Fig. 2." followed by a capital is a genuine sentence boundary.
    assert split_sentences("See Fig. 2. The model works.") == [
        "See Fig. 2.",
        "The model works.",
    ]


# ---------------------------------------------------------------------------
# split_parents (deterministic similarity walk; zero parent overlap)
# ---------------------------------------------------------------------------


def _keyword_vec(paragraph_text: str) -> list[float]:
    low = paragraph_text.lower()
    if any(k in low for k in ("alpha", "classification", "supervised")):
        return list(E)
    return list(F)


def test_split_parents_merges_then_splits_on_similarity(monkeypatch):
    pages = [
        _page([_body("Alpha concept one."), _body("Alpha concept two.")]),
        _page(
            [
                _body("Beta concept one."),
                _body("Beta concept two."),
                _body("Alpha concept three."),
            ]
        ),
    ]
    monkeypatch.setattr(sc, "dense_vector", lambda texts: [_keyword_vec(t) for t in texts])
    groups = split_parents(pages, threshold=0.2)
    # alpha,alpha merge; alpha->beta split; beta->alpha split -> 3 parents.
    assert len(groups) == 3
    contents = [_group_content(g) for g in groups]
    assert len(contents) == 3
    assert contents[0] == "Alpha concept one. Alpha concept two."
    assert contents[1].startswith("Beta concept one.")
    assert contents[2] == "Alpha concept three."
    # parents disjoint: no paragraph repeated across groups
    seen = set()
    for g in groups:
        for p in g:
            assert p.text not in seen
            seen.add(p.text)


def test_split_parents_all_similar_single_group(monkeypatch):
    pages = [_page([_body("Alpha one."), _body("Alpha two."), _body("Alpha three.")])]
    monkeypatch.setattr(sc, "dense_vector", lambda texts: [list(E)] * len(texts))
    assert len(split_parents(pages, threshold=0.5)) == 1


def test_split_parents_logs_boundary_decision(monkeypatch, caplog):
    pages = [_page([_body("Alpha one."), _body("Beta two.")])]
    monkeypatch.setattr(sc, "dense_vector", lambda texts: [_keyword_vec(t) for t in texts])
    with caplog.at_level("INFO", logger="SEMANTIC_CHUNKER"):
        split_parents(pages, threshold=0.2)
    assert any("Parent boundary" in r.message for r in caplog.records)


def _group_content(group: list[Paragraph]) -> str:
    return " ".join(p.text for p in group).strip()


# ---------------------------------------------------------------------------
# build_parents
# ---------------------------------------------------------------------------


def test_build_parents_no_overlap_between_adjacent_parents(monkeypatch):
    monkeypatch.setattr(sc, "PARENT_MIN_TOKENS_DROP", 0)
    monkeypatch.setattr(sc, "PARENT_MERGE_TOKENS", 0)
    groups = [
        [Paragraph("one two three")],
        [Paragraph("four five six")],
        [Paragraph("seven eight nine")],
    ]
    parents = build_parents(groups, "d1")
    assert len(parents) == 3
    contents = [p.content for p in parents]
    assert all(("four" not in c) for c in contents[:1] + contents[2:])
    # each word belongs to exactly one parent
    joined = " ".join(contents)
    assert joined.split() == ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


# ---------------------------------------------------------------------------
# build_children (sentence packing + exactly-one-sentence overlap)
# ---------------------------------------------------------------------------


def test_build_children_single_when_all_similar(monkeypatch):
    monkeypatch.setattr(sc, "dense_vector", lambda texts: [list(E)] * len(texts))
    parent = ParentChunk(
        parent_id="p1", document_id="d", title=None, page_start=1, page_end=1,
        content=(
            "Linear regression predicts continuous values. "
            "The parameters are fitted by minimizing squared error. "
            "Gradient descent updates the weights iteratively."
        ),
    )
    children = build_children(parent)
    assert len(children) == 1
    assert children[0].parent_id == "p1"
    assert "Linear regression" in children[0].content
    assert "iteratively" in children[0].content


def test_build_children_splits_on_topic_change_with_last_sentence_overlap(monkeypatch):
    # The two topics are disjoint; the second child carries exactly the last
    # sentence of the previous child as overlap.
    def fake_dense(texts):
        return [list(E) if t.lower().startswith("alpha") else list(F) for t in texts]

    monkeypatch.setattr(sc, "dense_vector", fake_dense)
    monkeypatch.setattr(sc, "CHILD_MIN_TOKENS_DROP", 0)
    monkeypatch.setattr(sc, "CHILD_MIN_TOKENS_MERGE", 0)
    parent = ParentChunk(
        parent_id="p1", document_id="d", title=None, page_start=1, page_end=1,
        content=(
            "Alpha supervised learning uses labeled examples. "
            "Alpha classification predicts discrete categories. "
            "Alpha regression predicts continuous values. "
            "Beta gradient descent optimizes model parameters. "
            "Beta backpropagation updates the network weights."
        ),
    )
    children = build_children(parent)
    assert len(children) == 2
    # the second child starts with the last sentence of the first child (overlap)
    prev_last = sc._last_sentence_of(children[0].content)
    assert children[1].content.startswith(prev_last)
    assert "backpropagation" in children[1].content
    # overlap is only that one sentence, never more
    assert not children[1].content.startswith(children[0].content)


def test_build_children_respects_size_cap_and_cuts_at_sentence_boundary(monkeypatch):
    monkeypatch.setattr(sc, "dense_vector", lambda texts: [list(E)] * len(texts))
    monkeypatch.setattr(sc, "CHILD_MAX_SIZE", 100)
    monkeypatch.setattr(sc, "WORDS_PER_TOKEN", 1.0)
    # Distinct, clearly-over-budget sentences: repeated-sentence dedup must not
    # collapse them, so the CHILD_MAX_SIZE cap alone forces the boundary. Each
    # sentence ~24 words; five of them ~120 words vs. the 100-word cap => 2.
    sentences = [
        "The related model training narrative covers a very long continuous "
        "stream of connected ideas that keep building across many consecutive "
        "clauses and clauses.",
        "A second sentence of the very same broad topic continues the story "
        "with still more words so that together these sentences fully consume "
        "the permitted token budget.",
        "Here the shared subject presses onward through many additional words "
        "adding still more detail and context all on the identical theme for "
        "the reader to see.",
        "This fourth line remains squarely on the one unifying topic while it "
        "piles on generous amounts of extra wording to guarantee the ceiling "
        "is passed very soon.",
        "The concluding statement wraps up the same single theme using the "
        "remaining words and this definitely ensures crossing over the limit.",
    ]
    parent = ParentChunk(
        parent_id="p1", document_id="d", title=None, page_start=1, page_end=1,
        content=" ".join(sentences) + ".",
    )
    children = build_children(parent)
    assert len(children) == 2
    # every boundary is a sentence boundary and no partial sentences appear
    for child in children:
        assert child.content.strip().endswith(".")


def test_build_children_reembeds_accumulated_chunk_against_next(monkeypatch):
    # The complete accumulated chunk text must be re-embedded after every
    # successful merge and compared against the next sentence -- never a running
    # centroid or average of per-sentence embeddings.
    embed_calls: list[str] = []

    def fake_dense(texts):
        embed_calls.extend(texts)
        return [list(E)] * len(texts)  # all similar -> everything merges

    monkeypatch.setattr(sc, "dense_vector", fake_dense)
    parent = ParentChunk(
        parent_id="p1", document_id="d", title=None, page_start=1, page_end=1,
        content="Alpha one. Alpha two. Alpha three. Alpha four.",
    )
    children = build_children(parent)
    assert len(children) == 1
    # the growing accumulated chunk is re-embedded and compared against the next
    # sentence at every step
    assert "Alpha one." in embed_calls
    assert "Alpha one. Alpha two." in embed_calls
    assert "Alpha one. Alpha two. Alpha three." in embed_calls
    assert "Alpha four." in embed_calls


def test_build_children_splits_on_similarity_break_with_full_chunk(monkeypatch, caplog):
    # A sentence that fails similarity against the full accumulated chunk closes
    # the chunk and starts a new one with that sentence (never discarded).
    def fake_dense(texts):
        out = []
        for t in texts:
            low = t.lower()
            out.append(list(E) if "alpha" in low else list(F) if "beta" in low else list(G))
        return out

    A = "Alpha supervised learning uses labeled examples."
    B = "Beta gradient descent optimizes model parameters."
    Cc = "Gamma backpropagation updates the network weights."
    monkeypatch.setattr(sc, "dense_vector", fake_dense)
    monkeypatch.setattr(sc, "CHILD_MIN_TOKENS_DROP", 0)
    monkeypatch.setattr(sc, "CHILD_MIN_TOKENS_MERGE", 0)
    parent = ParentChunk(
        parent_id="p1", document_id="d", title=None, page_start=1, page_end=1,
        content=f"{A} {B} {Cc}",
    )
    with caplog.at_level("INFO", logger="SEMANTIC_CHUNKER"):
        children = build_children(parent)
    # A->B sim 0 (break), B->C sim 0 (break) => 3 children
    assert len(children) == 3
    assert "beta" in children[1].content.lower()
    assert "gamma" in children[2].content.lower()
    # every boundary decision is logged
    child_logs = [r.message for r in caplog.records if r.message.startswith("Child boundary")]
    assert len(child_logs) == 2
    assert all("sim=" in m and "reason=" in m for m in child_logs)
    assert any("action=split" in m and "reason=similarity_break" in m for m in child_logs)


def test_build_children_min_floor_before_overlap_no_duplicate(monkeypatch):
    # Regression: the min-size floor must run BEFORE the overlap step. When a
    # tiny trailing group is folded into the previous chunk, running overlap
    # first used to inject the previous child's last sentence into that small
    # group and then fold it back into the same chunk -> the sentence appeared
    # twice. NOW the floor runs first, so no duplication occurs.
    def fake_dense(texts):
        out = []
        for t in texts:
            low = t.lower()
            out.append(list(E) if "alpha" in low else list(F))
        return out

    A1 = "Alpha supervised learning uses labeled examples."
    A2 = "Alpha classification predicts discrete categories."
    B = "Beta gradient descent optimizes parameters."
    monkeypatch.setattr(sc, "dense_vector", fake_dense)
    monkeypatch.setattr(sc, "CHILD_MIN_TOKENS_DROP", 0)
    monkeypatch.setattr(sc, "CHILD_MIN_TOKENS_MERGE", 100)
    monkeypatch.setattr(sc, "WORDS_PER_TOKEN", 1.0)
    monkeypatch.setattr(sc, "CHILD_MAX_SIZE", 700)
    parent = ParentChunk(
        parent_id="p1", document_id="d", title=None, page_start=1, page_end=1,
        content=f"{A1} {A2} {B}",
    )
    children = build_children(parent)
    assert len(children) == 1
    assert children[0].content == f"{A1} {A2} {B}"
    # the boundary sentence is present exactly once, never duplicated
    assert children[0].content.count(A2) == 1


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


def test_build_semantic_structure_shape(monkeypatch):
    pages = [
        _page([_heading("Chapter 1"), _body("Alpha content one."), _body("Alpha two here.")]),
        _page([_body("Alpha content three.")]),
    ]
    monkeypatch.setattr(sc, "dense_vector", lambda texts: [list(E)] * len(texts))
    monkeypatch.setattr(sc, "PARENT_MIN_TOKENS_DROP", 0)
    monkeypatch.setattr(sc, "PARENT_MERGE_TOKENS", 0)
    _patch_titles(monkeypatch)

    result = build_semantic_structure(pages, "doc-1")
    assert result["parents"]
    assert result["children"]
    parent_ids = {p["parent_id"] for p in result["parents"]}
    for child in result["children"]:
        assert child["parent_id"] in parent_ids
        assert child["child_id"]
        # This parent has exactly one child, so its synthetic subsection title
        # is suppressed (the section itself is the leaf); parent keeps its own.
        assert child["title"] is None
        assert child["heading"] is None
    for p in result["parents"]:
        assert p["title"] == "ST"


def test_label_families_titles_multi_child_children(monkeypatch):
    # A parent with two children must have both labeled; a single-child parent's
    # child stays suppressed (title None) and is not sent to the LLM.
    calls = []

    def fake_batch(entries, client=None, *, before=None):
        calls.append([level for level, _ in entries])
        return ["ST", "CT", "CT", "ST"]

    parents = [
        ParentChunk("p1", "d1", None, 1, 1, "Alpha phrase one."),
        ParentChunk("p2", "d1", None, 2, 2, "Beta phrase one."),
    ]
    children = [
        ChildChunk("c1", "p1", "d1", None, 1, 1, "Alpha child one."),
        ChildChunk("c2", "p1", "d1", None, 1, 1, "Alpha child two."),
        ChildChunk("c3", "p2", "d1", None, 2, 2, "Beta child one."),
    ]
    monkeypatch.setattr("app.offline.semantic_chunker.make_title_client", lambda: object())
    monkeypatch.setattr(
        "app.offline.semantic_chunker.generate_family_batch_titles", fake_batch
    )
    monkeypatch.setattr(
        "app.offline.semantic_chunker.is_acceptable_title",
        lambda title, content, level, used_titles: bool(title),
    )
    monkeypatch.setattr(
        "app.offline.semantic_chunker.regenerate_title",
        lambda client=None, content="", level="section", reject=None, used_titles=None: "RT",
    )

    sc._label_families(parents, children)

    assert calls == [["section", "subsection", "subsection", "section"]]
    assert parents[0].title == "ST"
    assert children[0].title == "CT"
    assert children[1].title == "CT"
    # single-child parent's child is not labeled
    assert children[2].title is None
    assert parents[1].title == "ST"


def test_generate_section_titles_parallel_preserves_order(monkeypatch):
    def fake(contents, *, level="section", before=None):
        return [f"Title {c.split()[0]}" for c in contents]

    monkeypatch.setattr("app.offline.semantic_chunker.generate_batch_titles", fake)
    parents = [
        ParentChunk(
            parent_id=f"p{i}", document_id="d1", title=None,
            content=f"Alpha{i} phrase.", page_start=1, page_end=1,
        )
        for i in range(6)
    ]
    sc.generate_section_titles(parents)
    assert [p.parent_id for p in parents] == [f"p{i}" for i in range(6)]
    assert [p.title for p in parents] == [f"Title Alpha{i}" for i in range(6)]


# ---------------------------------------------------------------------------
# strict chunk-size enforcement
# ---------------------------------------------------------------------------

def _words(prefix: str, n: int) -> str:
    """A single sentence of ``n`` distinct words (plus a trailing period)."""
    toks = [f"{prefix.capitalize()}{i}" for i in range(n)]
    toks[-1] += "Z"  # end on a letter so "N." is not read as a list marker
    return " ".join(toks) + "."


def test_apply_min_floor_merge_respects_child_cap(monkeypatch):
    # Fragment below the merge floor folds into the previous chunk as long as
    # the result stays within the cap...
    monkeypatch.setattr(sc, "WORDS_PER_TOKEN", 1.0)
    near_cap = _words("a", 10)
    groups = [_words("n", 660), _words("f", 35)]
    floored = sc._apply_min_floor(groups, max_child_words=700)
    assert [g for g in floored] == [_words("n", 660) + " " + _words("f", 35)]

    # ...but when the merge would exceed the cap, the fragment keeps its own
    # chunk instead of silently overflowing.
    groups = [_words("n", 690), _words("f", 35)]
    floored = sc._apply_min_floor(groups, max_child_words=700)
    assert floored == [_words("n", 690), _words("f", 35)]


def test_build_children_strict_cap_no_truncation_no_drops(monkeypatch):
    monkeypatch.setattr(sc, "WORDS_PER_TOKEN", 1.0)
    monkeypatch.setattr(sc, "CHILD_MAX_SIZE", 700)
    monkeypatch.setattr(sc, "CHILD_MIN_TOKENS_DROP", 0)

    head = [_words(f"x{i}", 10) for i in range(6)]
    head.append(_words("x6", 40))  # 100 words total, all concept E
    big = _words("big", 690)  # single 690-word sentence, concept E
    frag_a = _words("small", 5)  # 5 words, concept F
    frag_b = _words("tail", 30)  # 30 words, concept F

    content = " ".join(head + [big, frag_a, frag_b])

    prepared = sc._prepare_child_sentences(content)
    assert len(prepared) == 10

    def fake_vecs(texts):
        return [
            list(F) if t == frag_a or t == frag_b else list(E) for t in texts
        ]

    vecs = fake_vecs(prepared)
    parent = ParentChunk(
        parent_id="p1", document_id="d", title=None,
        page_start=1, page_end=1, content=content,
    )
    children = build_children(parent, sentence_vecs=vecs)

    # No child exceeds the cap.
    assert all(len(c.content.split()) <= 700 for c in children)

    # Every prepared sentence survives exactly once across children (the packer
    # never truncates or drops content to fit the cap).
    flat = " ".join(c.content for c in children)
    for s in prepared:
        assert flat.count(s) >= 1

    # Packing is greedy: the 100-word head closes when BIG would overflow it;
    # the 5+30-word fragment cannot merge into the 690-word chunk (725 > 700),
    # so it must keep its own third chunk rather than overflow.
    assert [len(c.content.split()) for c in children] == [100, 690, 35]

    # The overlap step is also cap-bounded: carrying the head's 40-word last
    # sentence into the 690-word chunk (and BIG's sentence into the fragment)
    # would exceed the cap, so both overlaps are skipped and no chunk is
    # duplicated or pushed over the limit.
    assert children[0].content == " ".join(head)
    assert children[1].content == big
    assert children[2].content == f"{frag_a} {frag_b}"