from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.offline import title_generator as tg  # noqa: E402
from app.offline.title_nlp import title_appears_in_text  # noqa: E402


class _FakeClient:
    """Deterministic fake LLM client for title orchestration tests.

    Accepts either a callable ``responder(prompt) -> str`` or a list of queued
    responses to be returned in order.
    """

    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    def chat(self, prompt, system_prompt=None, temperature=None, max_tokens=None, timeout=None):
        self.calls.append(prompt)
        return self._responder(prompt)


# ---------------------------------------------------------------------------
# clean_title
# ---------------------------------------------------------------------------


def test_clean_title_strips_numbering_markdown_and_quotes():
    assert tg.clean_title("## 3.1 Supervised Learning") == "Supervised Learning"
    assert tg.clean_title('"Linear Regression"') == "Linear Regression"
    assert tg.clean_title("**Training Data**") == "Training Data"


def test_clean_title_preserves_valid_colon_qualifier():
    assert tg.clean_title("Linear Regression: Cost Functions") == "Linear Regression: Cost Functions"


def test_clean_title_strips_generated_explanation_after_colon():
    assert tg.clean_title("Decision Trees: A method that splits on features") == "Decision Trees"


def test_clean_title_respects_max_words():
    assert tg.clean_title("Support Vector Machine Overview", 3) == "Support Vector Machine"


def test_clean_title_strips_bullets_and_trailing_punctuation():
    assert tg.clean_title("- Linear Regression,") == "Linear Regression"
    assert tg.clean_title("* 3.1 Gradient Descent:") == "Gradient Descent"
    assert tg.clean_title("• k-Nearest Neighbors -") == "k-Nearest Neighbors"


def test_clean_title_title_cases():
    assert tg.clean_title("supply and demand") == "Supply and Demand"
    assert tg.clean_title("support vector machine (svm)") == "Support Vector Machine (SVM)"
    assert tg.clean_title("k-Nearest Neighbors explained") == "k-Nearest Neighbors Explained"
    assert tg.clean_title("cost functions in regression") == "Cost Functions in Regression"
    assert tg.clean_title("Online Machine Learning Updates (or) Real-Time") == "Online Machine Learning Updates (or) Real-Time"


# ---------------------------------------------------------------------------
# validate_title
# ---------------------------------------------------------------------------


def test_validate_title_bounds():
    assert tg.validate_title("Linear Regression", 2, 5) is True
    assert tg.validate_title("Linear Regression Algorithm", 2, 5) is True
    assert tg.validate_title("Regression", 2, 5) is False  # too short
    assert tg.validate_title("Supervised Learning For Classification Of Many Things", 2, 5) is False


def test_validate_title_rejects_blank_and_forbidden_prefix():
    assert tg.validate_title("   ", 2, 5) is False
    assert tg.validate_title("Introduction to", 2, 5) is False


def test_validate_title_rejects_question_form():
    assert tg.validate_title("What Is Machine Learning?", 2, 6) is False
    assert tg.validate_title("How Does a Neural Network Work?", 2, 6) is False


def test_validate_title_rejects_comma_list():
    assert tg.validate_title("Data Cleaning, Missing Values, Outliers", 2, 6) is False
    assert tg.validate_title("Regression Tasks & Sequence Modeling", 2, 6) is False


def test_validate_title_rejects_numbered_and_bullets():
    assert tg.validate_title("1. Linear Regression", 2, 6) is False
    assert tg.validate_title("3.1 Supervised Learning", 2, 6) is False
    assert tg.validate_title("- Linear Regression", 2, 6) is False


def test_validate_title_rejects_non_acronym_parens():
    assert tg.validate_title("Linear Regression (analysis)", 2, 6) is False
    assert tg.validate_title("Linear Regression (for prediction)", 2, 6) is False


def test_validate_title_rejects_unbalanced_parens():
    assert tg.validate_title("Supervised Learning vs Unsupervised Learning (Spam", 2, 6) is False


def test_clean_title_trims_trailing_small_words():
    assert tg.clean_title("Online Learning for Continuous Data and", 6) == "Online Learning for Continuous Data"
    assert tg.clean_title("Feature Extraction and Dimensionality Reduction or", 6) == "Feature Extraction and Dimensionality Reduction"


def test_validate_title_accepts_acronym_parens():
    assert tg.validate_title("Support Vector Machine (SVM)", 2, 6) is True


# ---------------------------------------------------------------------------
# Fallback extraction
# ---------------------------------------------------------------------------


def test_fallback_strong_capitalized_phrase():
    label = tg._extract_noun_phrase(
        "Linear Regression models the relationship between a target and predictors.",
        5,
    )
    assert "Regression" in label


def test_fallback_k_prefixed_term():
    label = tg._extract_noun_phrase(
        "k-Nearest Neighbors algorithm classifies points by their closest training examples.",
        4,
    )
    assert label.lower().startswith("k-nearest")


def test_fallback_keeps_acronym():
    assert tg._extract_noun_phrase("Support Vector Machines (SVM) separate data.", 4) == "SVM"


def test_fallback_returns_empty_for_empty_input():
    assert tg._extract_noun_phrase("", 5) == ""


# ---------------------------------------------------------------------------
# Orchestration: valid LLM -> cleaned result, failure -> fallback
# ---------------------------------------------------------------------------


def test_generate_section_title_success():
    client = _FakeClient(lambda p: "Support Vector Machine")
    assert tg.generate_section_title(client, "Some content here that is long enough.") == "Support Vector Machine"


def test_generate_subsection_title_success():
    client = _FakeClient(lambda p: "Neural Networks")
    assert tg.generate_subsection_title(client, "Some content here.") == "Neural Networks"


def test_generate_title_falls_back_when_llm_fails():
    def raise_error(p):
        raise RuntimeError("boom")

    client = _FakeClient(raise_error)
    title = tg.generate_section_title(client, "Linear Regression models continuous targets.")
    assert title


def test_generate_title_falls_back_when_empty():
    client = _FakeClient(lambda p: "")
    title = tg.generate_subsection_title(client, "This is a fill of unordered tokens for a small doc")
    assert isinstance(title, str)


def test_generate_title_rejects_question_form_and_retries():
    client = _FakeClient(lambda p: "What Is Machine Learning?")
    title = tg.generate_section_title(client, "Some content here that is long enough.")
    assert "?" not in title


# ---------------------------------------------------------------------------
# Prompt content (TOC-editor style)
# ---------------------------------------------------------------------------


def test_section_prompt_has_toc_editor_framing():
    p = tg._SECTION_TITLE_PROMPT
    assert "Table of Contents" in p
    assert "ONE main concept" in p
    assert "grouped together in the textbook" in p
    assert "what would this section be called?" in p
    assert "Machine Learning Applications" in p
    assert "chapter" not in p


def test_subsection_prompt_is_focused_on_single_lesson():
    p = tg._SUBSECTION_TITLE_PROMPT
    assert "Table of Contents" in p
    assert "ONE main concept" in p
    assert "single focused lesson" in p
    assert "what would this subsection be called?" in p
    assert "Regression vs Classification" in p
    assert "chapter" not in p


def test_prompts_share_style_abstraction_and_self_check():
    for p in (tg._SECTION_TITLE_PROMPT, tg._SUBSECTION_TITLE_PROMPT):
        assert "2 to 6 words" in p
        assert "Image Classification" in p
        assert "Cardiovascular Diseases" in p
        assert "Email Spam Filtering" in p
        assert "Return ONLY the final title." in p
        assert "Introduction to" in p


def test_fallback_prompt_is_toc_style():
    assert "Table of Contents" in tg._FALLBACK_PROMPT
    assert "{max_words}" in tg._FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# Batch title generation (deduplicates sibling headers)
# ---------------------------------------------------------------------------


def test_parse_title_list_extracts_numbered_headers():
    raw = "1. Linear Regression\n2. Neural Networks\n3) Gradient Descent\n4: Feature Scaling"
    assert tg._parse_title_list(raw) == [
        "Linear Regression",
        "Neural Networks",
        "Gradient Descent",
        "Feature Scaling",
    ]


def test_batch_returns_one_distinct_title_per_chunk():
    client = _FakeClient(
        lambda p: "1. Linear Regression\n2. Neural Networks\n3. Gradient Descent"
    )
    titles = tg.generate_batch_titles(
        ["a" * 50, "b" * 50, "c" * 50], client, level="subsection"
    )
    assert titles == ["Linear Regression", "Neural Networks", "Gradient Descent"]
    assert len(set(titles)) == len(titles)


def test_batch_backfills_missing_or_invalid_header():
    client = _FakeClient(lambda p: "1. What Is Machine Learning?\n2. Neural Networks")
    titles = tg.generate_batch_titles(
        ["Supervised Learning studies labeled data.", "Neural Networks learn."],
        client,
        level="subsection",
    )
    assert titles[1] == "Neural Networks"
    assert titles[0]  # question form rejected -> fallback filled


def test_batch_dedupes_repeated_titles():
    client = _FakeClient(lambda p: "1. Neural Networks\n2. Neural Networks")
    titles = tg.generate_batch_titles(
        ["Neural Networks learn patterns.", "Support Vector Machines separate data."],
        client,
        level="subsection",
    )
    assert titles[0] == "Neural Networks"
    assert titles[1] != "Neural Networks"
    assert len(set(titles)) == len(titles)


def test_batch_keeps_headers_distinct_after_clean():
    client = _FakeClient(lambda p: "1. Neural Networks\n2. neural networks")
    titles = tg.generate_batch_titles(
        ["Neural Networks learn patterns.", "Neural Networks revisit patterns."],
        client,
        level="subsection",
    )
    assert len(set(titles)) == len(titles)


def test_batch_section_level_uses_section_framing():
    client = _FakeClient(lambda p: "1. Machine Learning Applications\n2. Data Preprocessing")
    titles = tg.generate_batch_titles(
        ["Spam filtering and image classification.", "Missing values and outliers."],
        client,
        level="section",
    )
    assert titles[0] == "Machine Learning Applications"
    assert len(set(titles)) == len(titles)


def test_make_titles_unique_never_doubles_roman_suffix():
    # Two chunks can both be titled "Machine Learning Overview II" (e.g. the
    # reviewer rewrote one to match another). The dedup must advance the Roman
    # numeral, not produce "... II II". Empty contents prevent the noun-phrase
    # fallback from masking the suffix path.
    out = tg.make_titles_unique(
        ["Machine Learning Overview II", "Machine Learning Overview II"],
        ["", ""],
        6,
    )
    assert out[0] == "Machine Learning Overview II"
    assert out[1] == "Machine Learning Overview III"
    assert len(set(out)) == len(out)
    assert not any(t.endswith("II II") for t in out)


def test_make_titles_unique_advances_triple_suffix():
    out = tg.make_titles_unique(
        ["Data Mining", "Data Mining", "Data Mining"],
        ["", "", ""],
        6,
    )
    assert out == ["Data Mining", "Data Mining II", "Data Mining III"]


def test_blocklisted_rejects_fill_headers_only():
    assert tg._blocklisted("Overview") is True
    assert tg._blocklisted("Introduction to Data") is True
    assert tg._blocklisted("Key Concepts") is True
    # A full, real heading that merely borrows a blocklist word stays valid.
    assert tg._blocklisted("Data Overview") is False
    assert tg._blocklisted("Regression Analysis") is False
    assert tg._blocklisted("") is True


def test_is_acceptable_title_checks_format_blocklist_and_dup(monkeypatch):
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)
    used = {"Linear Regression"}
    assert tg.is_acceptable_title("Linear Regression", "x", "section", set()) is True
    assert tg.is_acceptable_title("Linear Regression", "x", "section", used) is False
    assert tg.is_acceptable_title("Overview", "x", "section", set()) is False
    assert tg.is_acceptable_title("What Is It?", "x", "section", set()) is False
    assert tg.is_acceptable_title("  ", "x", "section", set()) is False


def test_generate_family_batch_titles_one_call_mixed_levels():
    client = _FakeClient(
        lambda p: "1. Data Preprocessing\n2. Handling Missing Values\n"
        "3. Outlier Detection\n4. Machine Learning Applications"
    )
    entries = [
        ("section", "a" * 60),
        ("subsection", "b" * 60),
        ("subsection", "c" * 60),
        ("section", "d" * 60),
    ]
    titles = tg.generate_family_batch_titles(entries, client)
    assert titles == [
        "Data Preprocessing",
        "Handling Missing Values",
        "Outlier Detection",
        "Machine Learning Applications",
    ]
    assert len(client.calls) == 1
    assert "[SECTION]" in client.calls[0]
    assert "[SUBSECTION]" in client.calls[0]


def test_generate_family_batch_titles_blank_on_invalid(monkeypatch):
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)
    client = _FakeClient(
        lambda p: "1. Data Preprocessing\n2. What Is It?\n3. Outlier Detection"
    )
    titles = tg.generate_family_batch_titles(
        [("section", "a" * 60), ("subsection", "b" * 60), ("subsection", "c" * 60)],
        client,
    )
    assert titles[0] == "Data Preprocessing"
    assert titles[1] == ""  # question form rejected -> left for regeneration
    assert titles[2] == "Outlier Detection"


def test_family_batch_strips_echoed_level_tag():
    client = _FakeClient(
        lambda p: "1. [SECTION] Data Preprocessing\n"
        "2. [SUBSECTION] Handling Missing Values\n"
        "3. [SECTION] Machine Learning Applications"
    )
    titles = tg.generate_family_batch_titles(
        [("section", "a" * 60), ("subsection", "b" * 60), ("section", "c" * 60)],
        client,
    )
    assert titles == [
        "Data Preprocessing",
        "Handling Missing Values",
        "Machine Learning Applications",
    ]


def test_clean_title_removes_level_tag_prefix():
    assert tg.clean_title("[SECTION] Data Preprocessing") == "Data Preprocessing"
    assert tg.clean_title("1. [SUBSECTION] Outlier Detection") == "Outlier Detection"
    assert tg.clean_title("Data Preprocessing") == "Data Preprocessing"


def test_regenerate_title_single_call_then_fallback(monkeypatch):
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)
    client = _FakeClient(lambda p: "Robust Regression")
    out = tg.regenerate_title(
        client,
        content="A robust regression resists outliers in the data.",
        level="section",
        reject=["Stepping"],
    )
    assert out == "Robust Regression"
    assert len(client.calls) == 1


def test_regenerate_title_avoids_used_title(monkeypatch):
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)
    monkeypatch.setattr(tg, "_safe_fallback", lambda content, max_words: "Outlier Handling")
    client = _FakeClient(lambda p: "Robust Regression")
    out = tg.regenerate_title(
        client,
        content="A robust regression resists outliers in the data.",
        level="section",
        reject=["Robust Regression"],
        used_titles={"Robust Regression"},
    )
    assert out == "Outlier Handling"


def test_safe_fallback_rejects_single_word_stub():
    # "option" is a real noun but too thin to be a navigation header; the fallback
    # must widen it to a phrase instead of emitting a bare stub.
    label = tg._safe_fallback(
        "It is also a good option if you have limited computing resources: once an "
        "online learning system has learned about new data instances, it does not "
        "need them anymore, so you can discard them",
        6,
    )
    assert len(label.split()) >= 2


def test_apply_verdict_keep_retains_valid_title(monkeypatch):
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)
    client = _FakeClient(lambda p: "10")

    class _Item:
        title = "Regularization Methods"
        content = "Regularization methods shrink the weight vectors."

    final = tg._apply_verdict(client, _Item(), "subsection", 9, "KEEP", "")
    assert final == "Regularization Methods"


def test_apply_verdict_keep_cannot_retain_invalid_single_word(monkeypatch):
    # A "KEEP" on a one-word stub ("Option") that violates min_words must not be
    # trusted; it falls through to the rewrite path and returns a valid title.
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)
    monkeypatch.setattr(
        tg,
        "_generate_candidates",
        lambda client, content, *, level, header="", n=4: ["Good Option"],
    )
    client = _FakeClient(lambda p: "10")

    class _Item:
        title = "Option"
        content = "Limited computing resources make the option attractive in practice."

    final = tg._apply_verdict(client, _Item(), "subsection", 9, "KEEP", "")
    assert final not in ("", "Option")


def test_apply_verdict_best_of_batch_returns_first_good(monkeypatch):
    # When the best-of-N candidate rescore clears the good threshold it is used.
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)
    monkeypatch.setattr(
        tg,
        "_generate_candidates",
        lambda client, content, *, level, header="", n=4: [
            "Weak Heading",
            "Strong Heading",
        ],
    )
    scores = iter(["9"])  # verify of the first candidate

    class _Fake:
        def chat(self, prompt, system_prompt=None, temperature=None, max_tokens=None, timeout=None):
            return next(scores)

    client = _Fake()

    class _Item:
        title = "Bad Title"
        content = "A passage that teaches something about strong headings here."

    final = tg._apply_verdict(client, _Item(), "subsection", 3, "REPLACE", "")
    assert final == "Weak Heading"


def test_apply_verdict_retries_then_falls_back(monkeypatch):
    # Two failing batches + bad rescore -> deterministic fallback, never the
    # original bad title.
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)
    monkeypatch.setattr(
        tg,
        "_generate_candidates",
        lambda client, content, *, level, header="", n=4: [],
    )
    client = _FakeClient(lambda p: "2")

    class _Item:
        title = "Bad Title"
        content = "Only a short fragment for fallback extraction."

    final = tg._apply_verdict(client, _Item(), "subsection", 1, "REPLACE", "")
    assert final not in ("", "Bad Title")


def test_generate_candidates_rejects_verbatim_lift(monkeypatch):
    # A candidate that appears verbatim in the passage (a proper-noun example,
    # not the section topic) must be dropped by the candidate filter.
    client = _FakeClient(
        lambda p: "1. Learning About Data\n2. Data and Knowledge Concepts\n3. Information Systems"
    )
    content = (
        "Crown Prince Salman was appointed in 2017. This chapter teaches about "
        "data, information and knowledge. Understanding these concepts is central "
        "to the course."
    )
    out = tg._generate_candidates(client, content, level="section")
    assert out
    assert all("Salman" not in c for c in out)
    assert all(title_appears_in_text(c, content) is False for c in out)


# ---------------------------------------------------------------------------
# Title review (post-generation pass)
# ---------------------------------------------------------------------------


class _StubItem:
    """Minimal stand-in for ParentChunk / ChildChunk for review tests."""

    def __init__(self, title, content, parent_id="p1"):
        self.title = title
        self.content = content
        self.parent_id = parent_id


def test_parse_review_builds_order_indexed_rows():
    raw = (
        "1. score=9 KEEP\n"
        "2. score=2 REPLACE: Regularization Techniques\n"
        "3. score=6 REFINE: Cross-Validation\n"
    )
    rows = tg._parse_review(raw)
    assert rows == {
        1: (9, "KEEP", ""),
        2: (2, "REPLACE", "Regularization Techniques"),
        3: (6, "REFINE", "Cross-Validation"),
    }


def test_review_titles_keeps_good_replaces_bad(monkeypatch):
    # Nouns always "grammatical" in this deterministic context.
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)

    def responder(prompt):
        if "Score:" in prompt:
            return "10"
        return "1. score=9 KEEP\n2. score=2 REPLACE: Outlier Handling"

    parent = _StubItem("Data Preprocessing", "Missing values must be handled before modeling.")
    child = _StubItem(
        "Stepping",
        "Outlier handling removes values that skew the training signal.",
    )
    tg.review_titles([parent], [child], client=_FakeClient(responder))
    assert parent.title == "Data Preprocessing"
    assert child.title == "Outlier Handling"


def test_review_titles_refines_polished_title(monkeypatch):
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)

    def responder(prompt):
        if "Score:" in prompt:
            return "9"
        return "1. score=6 REFINE: Regularization Methods"

    parent = _StubItem("Regularization", "L2 penalty shrinks the weight vector.")
    tg.review_titles([parent], [], client=_FakeClient(responder))
    assert parent.title == "Regularization Methods"


def test_review_titles_ignores_low_conf_verification(monkeypatch):
    # Rewrite proposed but a low re-score keeps the better-scoring candidate.
    monkeypatch.setattr(tg, "is_noun_phrase", lambda title: True)
    calls = {"n": 0}

    def responder(prompt):
        if "Score:" in prompt:
            calls["n"] += 1
            return "1"
        return "1. score=2 REPLACE: Robust Regression"

    parent = _StubItem("Stepping", "A robust regression resists outliers in the data.")
    tg.review_titles([parent], [], client=_FakeClient(responder))
    # Verification failed (score=1) so a fresh generation path is attempted;
    # with the fake always returning score=1 the rewrite is still applied as
    # the better of the two candidates.
    assert parent.title == "Robust Regression"
    assert calls["n"] >= 1


def test_review_titles_disabled_no_llm_call(monkeypatch):
    monkeypatch.setattr(tg, "TITLE_REVIEW_ENABLED", False)
    counter = {"n": 0}

    def responder(prompt):  # pragma: no cover - must never be called
        counter["n"] += 1
        return ""

    parent = _StubItem("Anything", "Some content here.")
    tg.review_titles([parent], [], client=_FakeClient(responder))
    assert counter["n"] == 0