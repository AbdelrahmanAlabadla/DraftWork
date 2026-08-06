from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.offline import title_generator as tg  # noqa: E402


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