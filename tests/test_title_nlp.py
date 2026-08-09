from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.offline import title_nlp as tnl  # noqa: E402


class _Tok:
    def __init__(self, text, pos, punct=False, space=False):
        self.text = text
        self.pos_ = pos
        self.is_punct = punct
        self.is_space = space


class _Doc(list):
    pass


class _FakeNLP:
    """Deterministic spaCy stand-in: returns a fixed token list per input."""

    def __init__(self, mapping):
        self._mapping = mapping

    def __call__(self, text):
        return self._mapping.get(text, [])


def _pairs(*pairs):
    return [_Tok(t, p) for t, p in pairs]


def test_is_noun_phrase_accepts_real_noun_phrases(monkeypatch):
    mapping = {
        "Linear Regression": _pairs(("Linear", "PROPN"), ("Regression", "NOUN")),
        "Learning Algorithm": _pairs(("Learning", "NOUN"), ("Algorithm", "PROPN")),
        "k-Nearest Neighbors": _pairs(("k", "X"), ("Nearest", "PROPN"), ("Neighbors", "NOUN")),
    }
    monkeypatch.setattr(tnl, "_get_nlp", lambda: _FakeNLP(mapping))
    for title in ("Linear Regression", "Learning Algorithm", "k-Nearest Neighbors"):
        assert tnl.is_noun_phrase(title) is True


def test_is_noun_phrase_rejects_verb_head(monkeypatch):
    mapping = {
        "Stepping": _pairs(("Stepping", "VERB")),
        "Stepping Back": _pairs(("Stepping", "VERB"), ("Back", "ADV")),
    }
    monkeypatch.setattr(tnl, "_get_nlp", lambda: _FakeNLP(mapping))
    assert tnl.is_noun_phrase("Stepping") is False
    assert tnl.is_noun_phrase("Stepping Back") is False


def test_is_noun_phrase_rejects_lone_adverb(monkeypatch):
    monkeypatch.setattr(
        tnl, "_get_nlp", lambda: _FakeNLP({"Fortunately": _pairs(("Fortunately", "ADV"))})
    )
    assert tnl.is_noun_phrase("Fortunately") is False


def test_is_noun_phrase_rejects_dangling_modal_tail(monkeypatch):
    mapping = {
        "Learning Algorithm Would": _pairs(
            ("Learning", "VERB"), ("Algorithm", "NOUN"), ("Would", "AUX")
        )
    }
    monkeypatch.setattr(tnl, "_get_nlp", lambda: _FakeNLP(mapping))
    assert tnl.is_noun_phrase("Learning Algorithm Would") is False


def test_is_noun_phrase_rejects_leading_function_words(monkeypatch):
    mapping = {
        "Introduction to": _pairs(("Introduction", "NOUN"), ("to", "ADP")),
        "Overview of": _pairs(("Overview", "NOUN"), ("of", "ADP")),
    }
    monkeypatch.setattr(tnl, "_get_nlp", lambda: _FakeNLP(mapping))
    assert tnl.is_noun_phrase("Introduction to") is False
    assert tnl.is_noun_phrase("Overview of") is False


def test_is_noun_phrase_fails_open_without_model(monkeypatch):
    monkeypatch.setattr(tnl, "_get_nlp", lambda: None)
    assert tnl.is_noun_phrase("Anything Goes") is True


def test_is_noun_phrase_rejects_empty():
    assert tnl.is_noun_phrase("") is False
    assert tnl.is_noun_phrase(None) is False


class _Chunk:
    def __init__(self, text):
        self.text = text


class _ChunkDoc:
    def __init__(self, chunks):
        self.chunks = chunks

    @property
    def noun_chunks(self):
        yield from self.chunks


class _FakeChunkNLP:
    def __init__(self, doc):
        self._doc = doc

    def __call__(self, text):
        return self._doc


def test_first_noun_chunk_prefers_technical_chunk_within_budget(monkeypatch):
    doc = _ChunkDoc(
        [_Chunk("the relationship"), _Chunk("Linear Regression"), _Chunk("a target")]
    )
    monkeypatch.setattr(tnl, "_get_nlp", lambda: _FakeChunkNLP(doc))
    assert tnl.first_noun_chunk("Linear Regression models a target.", 5) == "Linear Regression"


def test_first_noun_chunk_strips_leading_determiner(monkeypatch):
    doc = _ChunkDoc([_Chunk("the relationship between variables")])
    monkeypatch.setattr(tnl, "_get_nlp", lambda: _FakeChunkNLP(doc))
    assert tnl.first_noun_chunk("the relationship between variables matters.", 3) == (
        "relationship between variables"
    )


def test_first_noun_chunk_caps_to_budget(monkeypatch):
    doc = _ChunkDoc([_Chunk("support vector machine classifier")])
    monkeypatch.setattr(tnl, "_get_nlp", lambda: _FakeChunkNLP(doc))
    assert tnl.first_noun_chunk("support vector machine classifier works well.", 2) == (
        "support vector"
    )


def test_first_noun_chunk_fails_open_without_model(monkeypatch):
    monkeypatch.setattr(tnl, "_get_nlp", lambda: None)
    assert tnl.first_noun_chunk("anything at all", 4) == ""


def test_title_appears_in_text_word_boundaries():
    assert tnl.title_appears_in_text("Supervised Learning", "We study Supervised Learning here.") is True
    assert tnl.title_appears_in_text("Unsupervised Task", "Yet Another Important Unsupervised Task.") is True
    assert tnl.title_appears_in_text("Near", "This is not nearby at all.") is False
    assert tnl.title_appears_in_text("", "anything") is False
    assert tnl.title_appears_in_text("X", "") is False