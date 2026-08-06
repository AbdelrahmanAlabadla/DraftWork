from __future__ import annotations

import re
from typing import Callable

from app.config import (
    FALLBACK_SECTION_MAX_WORDS,
    FALLBACK_SUBSECTION_MAX_WORDS,
    SECTION_TITLE_CONTEXT_WORDS,
    SECTION_TITLE_MAX_WORDS,
    SECTION_TITLE_MIN_WORDS,
    SUBSECTION_TITLE_CONTEXT_WORDS,
    SUBSECTION_TITLE_MAX_WORDS,
    SUBSECTION_TITLE_MIN_WORDS,
    TITLE_MAX_ATTEMPTS,
    TITLE_MAX_TOKENS,
    TITLE_MODEL,
    TITLE_TEMPERATURE,
)
from app.llm.client import LMStudioClient
from app.logging_conf import get_logger

logger = get_logger("TITLE_GENERATOR")

# --- Prompt templates -------------------------------------------------------

# Core instruction shared by the section (parent) and subsection (child)
# prompts. The model writes a short header/label that sits directly above the
# passage -- not a Table of Contents description. Hierarchy is Section (parent
# chunk) / Subsection (child chunk) only -- there is no chapter layer yet.
_TITLE_CORE = (
    "You are writing the short HEADER for a passage in a university textbook. "
    "The header sits directly above the passage, labeling what it teaches. "
    "Write a concise, scannable header the way a professional textbook editor "
    "would, not a description or summary of the passage.\n"
    "\n"
    "Before writing the header:\n"
    "1. Understand the passage.\n"
    "2. Identify the ONE main concept being taught.\n"
    "3. Ignore supporting information.\n"
    "4. Create a concise header representing that concept.\n"
    "\n"
    "Ignore examples, case studies, figures, tables, exercises, historical "
    "stories, implementation details, code examples, dataset names, specific "
    "instances, and explanations. The header represents the concept, not the "
    "examples used to explain it."
)

# Level-specific instruction: a SECTION groups several related subsections, so
# the heading must name the broader theme that unifies them, not list topics.
_SECTION_LEVEL_BLOCK = (
    "This passage is a SECTION: several related subsections grouped together "
    "in the textbook.\n"
    "\n"
    "Your header must name the broader theme that unifies the subsections, "
    'as if answering "what would this section be called?" It is more general '
    "than any single subsection and must never list the subsections' topics.\n"
    "\n"
    "Good:\n"
    "- Email Spam Filtering, Image Classification, Speech Recognition, "
    "Recommendation Systems  ->  Machine Learning Applications\n"
    "- Missing Values, Outlier Detection, Data Cleaning  ->  Data Preprocessing"
)

# Level-specific instruction: a SUBSECTION is one focused lesson, so the
# heading names that single concept.
_SUBSECTION_LEVEL_BLOCK = (
    "This passage is a SUBSECTION: a single focused lesson inside a section of "
    "the textbook.\n"
    "\n"
    'Your header must name that ONE specific concept, as if answering "what '
    'would this subsection be called?" It is more focused than a section '
    "header and must never join several concepts into one header.\n"
    "\n"
    "Good:\n"
    "- Regression, Classification, Features  ->  Regression vs Classification\n"
    "- SGD, Adam, RMSProp  ->  Optimization Algorithms"
)

# Shared style contract for both levels.
_STYLE_RULES = (
    "Style rules:\n"
    "- 2 to 6 words, Title Case.\n"
    "- Looks like a real Table of Contents entry written by a professional "
    "editor.\n"
    "- Names ONE concept.\n"
    "- Concise and readable on its own, without the passage.\n"
    "- Abstraction level consistent with the sibling headings.\n"
    "\n"
    'Never generate a sentence, an explanation, a summary, a keyword list, '
    'comma-separated concepts, a numbered title, or a heading starting with '
    '"Introduction to", "Overview of", "A Comprehensive", or "Analysis of".'
)

# Abstraction rule: pick the concept being taught, not the most frequent words.
_ABSTRACTION_RULE = (
    "Do NOT choose the most frequent words. Choose the concept being taught:\n"
    "\n"
    "- CNN, ResNet, VGG, ImageNet  ->  Image Classification\n"
    "- Cash, Inventory, Receivables  ->  Current Assets\n"
    "- Heart Attack, Stroke, Hypertension  ->  Cardiovascular Diseases\n"
    "- How Machine Learning Improves Spam Detection  ->  Email Spam Filtering\n"
    "- Data Cleaning, Missing Values, Outliers Handling  ->  Data Preprocessing\n"
    "- Variables, Functions, Scope  ->  Programming Fundamentals\n"
    "- Gears, Torque, Shafts  ->  Gear Systems\n"
    "- Atoms, Electrons, Protons  ->  Atomic Structure\n"
    "- Voltage, Current, Resistance  ->  Electric Circuits\n"
    "- Demand, Supply, Price  ->  Supply and Demand\n"
    "- Assets, Liabilities, Equity  ->  Accounting Equation\n"
    "- Teachers, Students, Classroom  ->  Classroom Management\n"
    "- Napoleon, France, Russia, 1812  ->  Napoleon's Invasion of Russia"
)

# Self-check the model must run before returning; only the title is returned.
_SELF_CHECK = (
    "Before returning, verify:\n"
    "1. It names exactly one concept.\n"
    "2. It looks like a real Table of Contents entry.\n"
    "3. It is 2 to 6 words.\n"
    "4. It is not a sentence, explanation, or summary.\n"
    "5. It is not a keyword list.\n"
    "6. It avoids examples and supporting details.\n"
    "7. The abstraction level is right (the concept, not the most frequent "
    "words).\n"
    "8. Its style matches the sibling headings.\n"
    "9. A student could tell what the passage teaches from the title alone.\n"
    "\n"
    "Return ONLY the final title."
)

_SECTION_TITLE_PROMPT = (
    _TITLE_CORE
    + "\n\n"
    + _SECTION_LEVEL_BLOCK
    + "\n\n"
    + _STYLE_RULES
    + "\n\n"
    + _ABSTRACTION_RULE
    + "\n\n"
    + _SELF_CHECK
    + "\n\n## Content\n{content}"
)

_SUBSECTION_TITLE_PROMPT = (
    _TITLE_CORE
    + "\n\n"
    + _SUBSECTION_LEVEL_BLOCK
    + "\n\n"
    + _STYLE_RULES
    + "\n\n"
    + _ABSTRACTION_RULE
    + "\n\n"
    + _SELF_CHECK
    + "\n\n## Content\n{content}"
)

# Stricter prompt used on the second (validation-fallback) attempt. Told to
# output the single most important concept and nothing else.
_FALLBACK_PROMPT = (
    "Give ONE short heading (2 to {max_words} words) that names the single most "
    "important concept this passage teaches, as it would appear in a textbook "
    "Table of Contents.\n"
    "Rules:\n"
    "- 2 to {max_words} words, Title Case.\n"
    "- A noun heading, not a sentence.\n"
    "- No commas, no lists, no numbering, no question marks, no explanation.\n"
    "- Do not start with Introduction to, Overview of, A study of, "
    "Comprehensive, Analysis of, Framework for.\n"
    "- Return ONLY the heading.\n\n"
    "## Content\n{content}"
)

# Phrases that indicate a summary-style (generic) title. Kept lowercase for the
# case-insensitive prefix check.
_FORBIDDEN_PREFIXES: frozenset[str] = frozenset(
    {
        "a comprehensive",
        "comprehensive analysis",
        "comprehensive overview",
        "an overview",
        "overview of",
        "analysis of",
        "framework for",
        "foundations of",
        "foundation of",
        "introduction to",
        "a study of",
        "exploring",
        "application of",
        "applications of",
        "challenges and opportunities",
        "challenges and",
        "the role of",
    }
)

# Stopword / filler tokens filtered out of extracted phrases. Lowercase.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "but",
        "is", "are", "was", "were", "be", "being", "been", "that", "this",
        "which", "with", "from", "by", "as", "at", "into", "using", "used",
        "use", "via", "such", "can", "may", "should", "will", "we", "you",
        "it", "its", "not", "also", "data", "based", "them", "their", "these",
        "those", "more", "than", "same", "related", "all", "both", "each",
        "usually", "often", "typically", "generally", "about", "between",
        "through", "under",
    }
)

# Words that signal "handling/process" talk; strip these when they trail a
# detected noun phrase (they read like summary verbs, not names).
_TRAILING_VERBS: frozenset[str] = frozenset(
    {
        "predicting", "prediction", "predicted", "using", "uses", "learned",
        "handling", "processing", "comparison", "estimating", "estimation",
        "approaches", "approach", "based", "required", "involves",
        "require", "requires", "reflects", "reflect", "show", "indicates",
        "classifies", "classify", "classifying", "classifier", "classifiers",
        "predicts", "estimates", "computes", "calculates", "calculating",
        "groups", "grouping", "assigns", "assigning", "identifies", "finds",
        "finds", "returns", "outputs", "outputting", "selects", "selecting",
        "works", "achieves", "optimizes", "measures", "suits", "fits",
        "captures", "describes", "illustrates", "demonstrates", "introduces",
        "defines", "explores", "covers", "connects", "touches", "examines",
    }
)

# Functional connectives / main verbs that stop a noun-phrase run dead, so the
# fallback never swallows a full sentence.
_RUN_STOPPERS: frozenset[str] = frozenset(
    {
        "is", "are", "was", "were", "can", "may", "could", "will", "which",
        "that", "where", "when", "because", "while", "with", "and", "or",
        "does", "do", "to", "the",
    }
)


def _first_n_words(text: str, n: int) -> str:
    words = [w for w in text.split() if w.strip()]
    return " ".join(words[:n])


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


# Words kept lowercase in a Title Case heading (except when first).
_SMALL_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "to", "for", "with",
        "vs", "v", "in", "on", "at", "by", "via", "from", "as", "into",
    }
)


def _to_title_case(title: str) -> str:
    """Title-case a heading, preserving acronyms, k-/t- terms, and parens.

    Acronyms like ``SVM`` and k-/t- prefixed terms like ``k-Nearest`` stay as
    written; a parenthesized word is uppercased into an acronym ``(svm)``.
    """
    words = title.split()
    out = []
    for i, w in enumerate(words):
        key = re.sub(r"[^A-Za-z0-9-]", "", w)
        if re.fullmatch(r"[A-Z]{2,}", key):
            out.append(w)  # acronym, unchanged
            continue
        if re.fullmatch(r"[kKtT]-[A-Za-z0-9]+", w):
            out.append(w)  # k-Nearest, t-SNE, unchanged
            continue
        m = re.fullmatch(r"\(([A-Za-z]{2,})\)", w)
        if m:
            inner = m.group(1)
            if inner.lower() in _SMALL_WORDS:
                out.append("(" + inner.lower() + ")")  # "(or)" qualifier, not an acronym
            else:
                out.append("(" + inner.upper() + ")")
            continue
        if i > 0 and w.lower() in _SMALL_WORDS:
            out.append(w.lower())
            continue
        if "-" in w:
            pieces = []
            for pc in w.split("-"):
                if pc and pc[0].isupper():
                    pieces.append(pc)
                elif pc.lower() in _SMALL_WORDS:
                    pieces.append(pc.lower())
                else:
                    pieces.append(pc[:1].upper() + pc[1:])
            out.append("-".join(pieces))
            continue
        out.append(w[:1].upper() + w[1:])
    return " ".join(out)


# ---------------------------------------------------------------------------
# clean_title
# ---------------------------------------------------------------------------


def clean_title(raw: str, max_words: int = 0) -> str:
    """Normalize a raw LLM title into a clean navigation label.

    Strips markdown, quotes, numbering and trims whitespace. Preserves valid
    TOC titles that use ``:`` as a topic qualifier (e.g. "Linear Regression:
    Cost Functions"). Only removes the colon suffix when the part after the
    colon is clearly generated explanation (e.g. "...: A Comprehensive
    Analysis of ...").
    """
    if not raw:
        return ""
    title = raw.strip()
    # Remove markdown emphasis markers anywhere (**title**, __title__).
    title = re.sub(r"(\*\*|__)", "", title)
    # Strip leading markdown heading markers (\# symbols).
    title = re.sub(r"^\s*#+\s*", "", title).strip()
    # Strip leading bullets.
    title = re.sub(r"^\s*[-*•]\s+", "", title).strip()
    # Strip leading numbering like "1. " / "3.1 " / "001: ".
    title = re.sub(r"^\s*\d+(?:[.:-]\d+)*(?:[.:-]\s*)?", "", title).strip()
    # Remove surrounding quote characters.
    title = title.strip('"').strip("'").strip()
    # Trim trailing sentence-like final punctuation (keep intra-title like
    # "k-Nearest Neighbors" unsplit).
    title = title.rstrip(".。")
    # Trim trailing list/code punctuation ("Title,", "Title:", "Title;", "Title -").
    title = title.rstrip(",;:").rstrip("-—").strip()

    if ":" in title:
        before, after = title.split(":", 1)
        before = before.strip()
        after = after.strip()
        # Keep the colon only if the second part is itself a compact, terse
        # qualifier (<=4 words and not starting with a generic filler). Drop it
        # when the suffix is clearly a generated explanation / full sentence.
        if _is_generated_explanation(after):
            title = before
        else:
            title = f"{before}: {after}"

    title = re.sub(r"\s+", " ", title).strip()

    if max_words > 0:
        title = _first_n_words(title, max_words)
    # Drop trailing small words left over from truncation ("... and", "... or").
    words = title.split()
    while len(words) > 1 and words[-1].lower().strip("(") in _SMALL_WORDS:
        words = words[:-1]
    title = " ".join(words).strip()
    title = _to_title_case(title)
    return title


def _is_generated_explanation(clause: str) -> bool:
    """True if the clause after ':' reads like a generated explanation."""
    if not clause:
        return True
    if _word_count(clause) > 4:
        return True
    low = clause.lower()
    # Generic filler prefixes / trailing verbs mark an explanation.
    if low.startswith(tuple(_FORBIDDEN_PREFIXES)):
        return True
    # Is it basically a verb phrase ("...predicts values", "...improves"?)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", low) if w]
    if not words:
        return False
    guess_tok = words[-1].lower()
    if guess_tok in _TRAILING_VERBS:
        return True
    return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_title(title: str, min_words: int, max_words: int) -> bool:
    """A title is valid if non-empty, within bounds, and not generic.

    Enforces strict noun-phrase / Table-of-Contents style: no question marks,
    no comma-separated keyword lists, no numbering, no bullets, and no
    parentheses unless they wrap an uppercase acronym (e.g. ``(SVM)``).
    """
    if not title or not title.strip():
        return False
    n = _word_count(title)
    if n < min_words or n > max_words:
        return False
    low = title.lower().lstrip(" \t")
    if any(low.startswith(term) for term in _FORBIDDEN_PREFIXES):
        return False
    # Reject question-form titles.
    if "?" in title:
        return False
    # Reject comma-separated keyword lists and ampersand joins.
    if "," in title or "&" in title:
        return False
    # Reject numbered markers ("1. ", "3.1 ", "1: ").
    if re.search(r"\b\d+[.:)]", title):
        return False
    # Reject bullets and stray newlines.
    if "\n" in title or re.search(r"(^|\s)[-*•]\s", title):
        return False
    # Reject unbalanced parentheses (e.g. a truncated "(Spam").
    if title.count("(") != title.count(")"):
        return False
    # Parentheses only allowed around an uppercase acronym, e.g. (SVM).
    for match in re.finditer(r"\(([^)]*)\)", title):
        if not re.fullmatch(r"[A-Z]{2,5}", match.group(1)):
            return False
    return True


# ---------------------------------------------------------------------------
# Fallback extraction (conservative noun-phrase label)
# ---------------------------------------------------------------------------


def _extract_noun_phrase(content: str, max_words: int) -> str:
    """Derive a compact label from the first words of content.

    Acts only on a small preview and never reconstructs a full sentence.
    Prefers technical terms, acronyms, and capitalized phrases, and preserves
    their technical capitalization (k-Nearest Neighbors, t-SNE, PCA).
    """
    preview = _first_n_words(content, 250)
    if not preview:
        return ""

    # 1) Acronym in parentheses after a name, e.g. "... (SVM)".
    m = re.search(r"\(([A-Z]{2,})\)", preview)
    if m:
        token = m.group(1).strip()
        if token:
            return token

    # 2) k / t prefixed terms (k-Nearest Neighbors, k-means, t-SNE).
    m = re.search(r"\b([kKtT]-[A-Za-z0-9]+)", preview)
    if m:
        word_map = preview.split()
        # Start the phrase at the token that actually contains the k- term.
        before = preview[:m.start()].split()
        j = len(before)
        if j >= len(word_map) or m.group(0) not in word_map[j]:
            # account for the '-word' being one token
            run = [m.group(1)]
        else:
            run = [word_map[j]]
            j += 1
            while j < len(word_map) and _extends_run(word_map[j]):
                run.append(word_map[j])
                j += 1
        phrase = _trim_phrase(run)
        if phrase:
            return _first_n_words(phrase, max_words)
        return m.group(1)

    # 3) Runs of words headed by a capitalized token: the classic
    #    "Linear Regression", "Training Data", "Support Vector Machine".
    words = preview.split()
    best = ""
    best_len = 0
    i = 0
    while i < len(words):
        if not _STOPS_RUN(words[i]):
            head = _strip_punct(words[i])
            if head and head[0].isupper():
                run = [words[i]]
                j = i + 1
                while j < len(words) and _extends_run(words[j]):
                    run.append(words[j])
                    j += 1
                phrase = _trim_phrase(run)
                pn = len(phrase.split()) if phrase else 0
                if pn and pn > best_len:
                    best = phrase
                    best_len = pn
                i = j
                continue
        i += 1

    if best:
        return _case_presence(best, max_words)

    # 4) Single most salient keyword.
    for tok in words:
        key = _strip_punct(tok)
        if key and key.lower() not in _STOPWORDS and (tok[:1].isupper() or len(key) > 2):
            return _preserve_case(key)
    return ""


def _STOPS_RUN(token: str) -> bool:
    key = _strip_punct(token).lower()
    return key in _RUN_STOPPERS or not key


def _extends_run(token: str) -> bool:
    key = _strip_punct(token).lower()
    if key in _RUN_STOPPERS or key in _TRAILING_VERBS:
        return False
    # Lowercase noun continuation (regression, data, neighbors, trees...).
    if token[:1].islower():
        return len(key) >= 3
    # Another capitalized word: stop (likely a new phrase).
    return False


def _case_presence(phrase: str, max_words: int) -> str:
    """Title-case ordinary words but preserve acronyms / hyphenated tech text."""
    words = _first_n_words(phrase, max_words).split()
    out = []
    for w in words:
        key = _strip_punct(w)
        if re.fullmatch(r"[A-Z]{2,}", key):
            out.append(w)  # acronym, unchanged
        elif "-" in w:
            # Title-case each hyphen piece unless it's a lowercase prefix (k-, t-).
            pieces = []
            for pc in w.split("-"):
                if pc and pc[0].isupper():
                    pieces.append(pc)
                elif pc.lower() in _STOPWORDS:
                    pieces.append(pc)
                else:
                    pieces.append(pc[:1].upper() + pc[1:])
            out.append("-".join(pieces))
        else:
            out.append(w[:1].upper() + w[1:] if w else w)
    return " ".join(out)


def _trim_phrase(run: list[str]) -> str:
    """Trim leading/ trailing filler and trailing verbs from a run."""
    while run and (_strip_punct(run[0]).lower() in _STOPWORDS):
        run = run[1:]
    while run and (_strip_punct(run[-1]).lower() in _TRAILING_VERBS
                   or _strip_punct(run[-1]).lower() in _STOPWORDS):
        run = run[:-1]
    return " ".join(run).strip() if run else ""


def _strip_punct(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", token)


def _preserve_case(key: str) -> str:
    return key


# ---------------------------------------------------------------------------
# LLM call + orchestration
# ---------------------------------------------------------------------------


def _make_client() -> LMStudioClient:
    return LMStudioClient(model=TITLE_MODEL)


def generate_section_title(client=None, content: str = "") -> str:
    return _generate_title(
        client,
        content,
        preview_words=SECTION_TITLE_CONTEXT_WORDS,
        main_prompt=_SECTION_TITLE_PROMPT,
        fallback_prompt=_FALLBACK_PROMPT.replace("{max_words}", str(FALLBACK_SECTION_MAX_WORDS)),
        min_words=SECTION_TITLE_MIN_WORDS,
        max_words=SECTION_TITLE_MAX_WORDS,
        fallback_max_words=FALLBACK_SECTION_MAX_WORDS,
    )


def generate_subsection_title(client=None, content: str = "") -> str:
    return _generate_title(
        client,
        content,
        preview_words=SUBSECTION_TITLE_CONTEXT_WORDS,
        main_prompt=_SUBSECTION_TITLE_PROMPT,
        fallback_prompt=_FALLBACK_PROMPT.replace("{max_words}", str(FALLBACK_SUBSECTION_MAX_WORDS)),
        min_words=SUBSECTION_TITLE_MIN_WORDS,
        max_words=SUBSECTION_TITLE_MAX_WORDS,
        fallback_max_words=FALLBACK_SUBSECTION_MAX_WORDS,
    )


def _generate_title(
    client,
    content: str,
    preview_words: int,
    main_prompt: str,
    fallback_prompt: str,
    min_words: int,
    max_words: int,
    fallback_max_words: int,
) -> str:
    preview = _first_n_words(content, preview_words)
    if not preview:
        return ""

    client = client or _make_client()

    result = _attempt(client, main_prompt.format(content=preview), min_words, max_words)
    if result:
        return _finalize(result)

    # Once more with the stricter fallback prompt, up to the attempt budget.
    for _ in range(max(1, TITLE_MAX_ATTEMPTS - 1)):
        result = _attempt(client, fallback_prompt.format(content=preview), min_words, max_words)
        if result:
            return _finalize(result)

    # Safety net: deterministic noun-phrase label.
    label = _extract_noun_phrase(content, fallback_max_words)
    logger.warning("Title fallback used | min=%d max=%d label=%r", min_words, max_words, label)
    return label


def _attempt(client, prompt: str, min_words: int, max_words: int) -> str:
    try:
        raw = client.chat(
            prompt,
            system_prompt=None,
            temperature=TITLE_TEMPERATURE,
            max_tokens=TITLE_MAX_TOKENS,
            timeout=1800,
        )
    except Exception as exc:  # noqa: BLE001 - title failure must not break pipeline
        logger.warning("Title generation error: %s: %s", type(exc).__name__, exc)
        return None
    cleaned = clean_title(raw, max_words)
    if validate_title(cleaned, min_words, max_words):
        return cleaned
    return None


def _finalize(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()