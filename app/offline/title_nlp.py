from __future__ import annotations

import re

from app.logging_conf import get_logger

logger = get_logger("TITLE_NLP")

# POS tags that make a token a function word rather than content. A title whose
# last content word is one of these is a dangling fragment ("Introduction to",
# "Classification of", "Learning Algorithm Would"), never a noun heading.
_FUNCTIONAL_POS = frozenset(
    {
        "DET", "ADP", "CCONJ", "SCONJ", "PART", "AUX", "PRON",
        "ADV", "INTJ", "SYM", "X",
    }
)

# POS tags acceptable as the grammatical head (last content word) of a title.
_HEAD_POS = frozenset({"NOUN", "PROPN", "NUM", "ADJ"})

_NLP = None


def _get_nlp():
    """Lazily load the spaCy model once; cache the failure so a missing model
    does not retry the (slow) load on every call."""
    global _NLP
    if _NLP is None:
        try:
            import spacy

            _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
        except Exception as exc:  # noqa: BLE001 - NLP is a best-effort helper
            _NLP = False
            logger.warning("spaCy model unavailable, NLP title checks disabled: %s", exc)
    return _NLP or None


def warmup() -> bool:
    """Preload the spaCy model once at server startup.

    Returns True when the model is available; warmup failures are logged by
    ``_get_nlp`` (best-effort helper), so this never raises.
    """
    return _get_nlp() is not None


def is_noun_phrase(title: str) -> bool:
    """True when ``title`` is grammatically a noun phrase, not a fragment.

    Uses spaCy POS tags: rejects titles that start or end on a function word,
    a bare adverb ("Fortunately"), a verb-headed stump ("Stepping Back"), a
    modal tail ("... Would"), or that carry no noun at all. Fails open (returns
    True) when the spaCy model is unavailable so the pipeline never hardens.
    """
    if not title or not title.strip():
        return False
    nlp = _get_nlp()
    if nlp is None:
        return True  # fail-open: absence of the model must not reject titles
    doc = nlp(title)
    toks = [t for t in doc if not t.is_space and not t.is_punct]
    if not toks:
        return False
    last = toks[-1]
    if last.pos_ in _FUNCTIONAL_POS:
        # "Introduction to", "Classification of", "... Would": the heading ends
        # on a dangling function word instead of a noun head.
        return False
    if last.pos_ not in _HEAD_POS:
        return False
    if not any(t.pos_ in {"NOUN", "PROPN", "NUM"} for t in toks):
        # Adverb/adjective-only fragments like "Fortunately".
        return False
    return True


def first_noun_chunk(content: str, max_words: int) -> str:
    """The first suitable spaCy noun chunk of ``content`` as a fallback label.

    Prefers a chunk that fits within ``max_words`` and looks technical
    (starts capitalized or contains an acronym / digit). Falls back to the
    first chunk overall, capped to ``max_words``. Returns "" when the noun
    phrase model is unavailable or nothing usable is found.
    """
    if not content:
        return ""
    nlp = _get_nlp()
    if nlp is None:
        return ""
    try:
        chunks = list(nlp(content[:2000]).noun_chunks)
    except Exception as exc:  # noqa: BLE001 - best-effort helper
        logger.warning("noun chunk extraction failed: %s", exc)
        return ""

    if not chunks:
        return ""

    def cleaned(chunk) -> str:
        return _clean_chunk(chunk.text, max_words)

    # 1) A capitalized multi-word chunk that fits the budget: looks like a real
    #    textbook heading. A bare pronoun/name chunk ("It") never qualifies.
    for chunk in chunks:
        text = cleaned(chunk)
        if text and len(text.split()) >= 2 and text[0].isupper():
            return text
    # 2) A technical-looking chunk (acronym or digit) within the word budget.
    for chunk in chunks:
        text = cleaned(chunk)
        if text and len(text.split()) <= max_words and (
            re.search(r"[A-Z]{2,}", text)
            or re.search(r"(?<![A-Za-z])[\d]", text)
        ):
            return text
    # 3) Any multi-word chunk, capped to the budget.
    for chunk in chunks:
        text = cleaned(chunk)
        if text and len(text.split()) >= 2:
            return text
    # 4) Otherwise the first chunk, trimmed to the word budget.
    return _clean_chunk(chunks[0].text, max_words)


def _clean_chunk(text: str, max_words: int) -> str:
    text = re.sub(r"^(?:the|an?)\s+", "", text.strip(), flags=re.IGNORECASE)
    if not text:
        return ""
    words = [w for w in text.split() if w.strip()]
    return " ".join(words[:max_words]).strip()


def title_appears_in_text(title: str, content: str) -> bool:
    """True when the title reads verbatim inside the passage.

    A title that is printed word-for-word in the body is either a genuine
    reused heading (good) or a phrase lifted from prose (bad); this flag lets
    the reviewer apply that judgement instead of a binary filter. Matching is
    whole-word, so "Near" never matches "nearby".
    """
    if not title or not content:
        return False
    keys = [re.sub(r"[\W_]", "", w, flags=re.UNICODE) for w in re.findall(r"[\w'-]+", title.lower(), flags=re.UNICODE)]
    keys = [k for k in keys if k]
    if not keys:
        return False
    hay = re.findall(r"[\w'-]+", content.lower(), flags=re.UNICODE)
    for i in range(len(hay) - len(keys) + 1):
        if all(
            re.sub(r"[\W_]", "", hay[i + j], flags=re.UNICODE) == keys[j]
            for j in range(len(keys))
        ):
            return True
    return False