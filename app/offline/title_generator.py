from __future__ import annotations

import re
from collections import Counter

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
    TITLE_LMS_URL,
    TITLE_MODEL,
    TITLE_BLOCKLIST,
    TITLE_CONTEXT_RECENT,
    TITLE_REVIEW_CONTEXT_WORDS,
    TITLE_REVIEW_ENABLED,
    TITLE_REVIEW_GOOD_SCORE,
    TITLE_REVIEW_POLISH_MIN,
    TITLE_REVIEW_CANDIDATES,
    TITLE_REVIEW_RETRIES,
    TITLE_TEMPERATURE,
)
from app.llm.factory import create_llm_client
from app.language import arabic_comparison_key, detect_language
from app.logging_conf import get_logger
from app.offline.title_nlp import first_noun_chunk, is_noun_phrase, title_appears_in_text

logger = get_logger("TITLE_GENERATOR")

# --- Prompt templates -------------------------------------------------------

# Core instruction shared by the section (parent) and subsection (child)
# prompts. The model writes a short header/label that sits directly above the
# passage -- not a Table of Contents description. Hierarchy is Section (parent
# chunk) / Subsection (child chunk) only -- there is no chapter layer yet.
_TITLE_CORE = (
    "You are writing a standalone descriptive HEADER for a passage in a "
    "university textbook. A student must understand the passage's main subject "
    "and what it teaches about that subject without reading the passage.\n"
    "\n"
    "Before writing the header:\n"
    "1. Understand the passage.\n"
    "2. Identify the dominant subject and the main function, process, relation, "
    "cause, result, or distinction taught about it.\n"
    "3. Check that this meaning is supported across the passage, not by one "
    "small opening sentence.\n"
    "4. Create a concise descriptive header representing that dominant content.\n"
    "\n"
    "Ignore examples, case studies, figures, tables, exercises, historical "
    "stories, implementation details, code examples, dataset names, and isolated "
    "instances. Keep the explanatory relationship that makes the title useful."
)

# The passage often already carries a real textbook heading. Prefer it: only
# invent a header when the heading is missing, noisy, or unrelated.
_HEADING_RULE = (
    "\n"
    "The BOOK HEADING supplied with a passage is topic context from the source. "
    "Use it to stay within the correct topic. Keep it unchanged when it already "
    "describes the chunk's dominant function, process, relation, cause, result, "
    "or distinction. Otherwise expand or refine it using the representative "
    "chunk content. Ignore a "
    "heading only when it is OCR garbage or unrelated.\n"
    "\n"
    "Example: BOOK HEADING: المستقبلات الحسية; content about detecting stimuli "
    "and carrying neural signals -> دور المستقبلات الحسية في اكتشاف المؤثرات "
    "ونقل الإشارات العصبية."
)

# Level-specific instruction: a SECTION groups several related subsections, so
# the heading must name the broader theme that unifies them, not list topics.
_SECTION_LEVEL_BLOCK = (
    "This passage is a SECTION: several related subsections grouped together "
    "in the textbook.\n"
    "\n"
    "Your header must describe the broader lesson that unifies the subsections "
    "while still saying what the section teaches about that subject. It must "
    "not collapse into a vague category label or list subsection topics.\n"
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
    "- Prefer 4 to 12 words; never exceed 15 words.\n"
    "- Looks like a real Table of Contents entry written by a professional "
    "editor.\n"
    "- Names the dominant subject and a useful content-specific relationship.\n"
    "- Concise and readable on its own, without the passage.\n"
    "- Abstraction level consistent with the sibling headings.\n"
    "\n"
    'Never generate a full sentence, an explanation, a vague category, a keyword list, '
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
    "1. It names the dominant subject and what the passage teaches about it.\n"
    "2. It looks like a real Table of Contents entry.\n"
    "3. It is preferably 4 to 12 words and no more than 15.\n"
    "4. It is not a sentence, explanation, or summary.\n"
    "5. It is not a keyword list.\n"
    "6. It avoids examples but keeps the dominant process or relationship.\n"
    "7. The abstraction level is right (the concept, not the most frequent "
    "words).\n"
    "8. Its style matches the sibling headings.\n"
    "9. A student can tell what the passage teaches from the title alone.\n"
    "10. If it repeats the BOOK HEADING, that heading is already specific and "
    "fully describes the dominant content.\n"
    "\n"
    "Return ONLY the final title."
)

_SECTION_TITLE_PROMPT = (
    _TITLE_CORE
    + _HEADING_RULE
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
    + _HEADING_RULE
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

# Multi-chunk batch title prompt. Several sibling passages are labeled in a
# single call; the model sees all of them together so it keeps every header
# distinct (sibling awareness is what the per-call-parallel version lacks,
# which is what produced duplicated titles).
_BATCH_RULE = (
    "You are still writing the same short headers, but now you must label "
    "SEVERAL passages at once. Each numbered entry below is a separate chunk "
    "of the textbook.\n"
    "\n"
    "CRITICAL REQUIREMENTS:\n"
    "- Produce EXACTLY one header for EVERY passage.\n"
    "- Every header must be DIFFERENT from every other header. A real textbook "
    "would never place the same heading over two different passages.\n"
    "- Base each header only on the content of its own numbered passage.\n"
    "- If a passage already contains a meaningful textbook heading, REUSE it "
    "(cleaned up) as that passage's header; only write a new header when the "
    "heading is missing, garbage, or unrelated.\n"
    "\n"
    'Output format (strict): a numbered list, one header per line, keeping '
    "the same order as the passages. Example:\n"
    "1. Header For First Passage\n"
    "2. Header For Second Passage\n"
    "3. Header For Third Passage\n"
)

_BATCH_SELF_CHECK = (
    "Before returning, verify:\n"
    "1. Every passage received exactly one header.\n"
    "2. All headers are distinct from each other.\n"
    "3. Each header is 2 to 6 words, Title Case, one concept, ToC-style.\n"
    "4. Headers appear in the same order as the numbered passages.\n"
    "\n"
    "Return ONLY the numbered list of headers."
)

# Mixed-family batch: a parent SECTION and its SUBSECTION children are labeled
# in the SAME numbered list, so the section header can be written broader than
# (and consistent with) its own subsections in one call.
_FAMILY_BATCH_RULE = (
    "You are still writing the same short headers, but now each numbered entry "
    "is tagged with its level. A SECTION is a group of related subsections; "
    "its header must describe the broader lesson unifying them. A SUBSECTION is a "
    "single focused lesson inside a section; its header must name that ONE "
    "specific concept and must stay NARROWER than its own SECTION header.\n"
    "\n"
    "CRITICAL REQUIREMENTS:\n"
    "- Produce EXACTLY one header for EVERY numbered passage.\n"
    "- Every header must be DIFFERENT from every other header.\n"
    "- A SUBSECTION header must never equal or generalize its SECTION header.\n"
    "- Base each header mainly on the representative content of its own passage.\n"
    "- Write each header in that passage's REQUIRED TITLE LANGUAGE. English "
    "technical abbreviations may remain inside an Arabic header when needed.\n"
    "- Use BOOK HEADING as a topic anchor. Return it unchanged only when it is "
    "already specific and fully describes the dominant content.\n"
    "- Reject a broad category when the content supports a specific process, "
    "function, relation, cause, result, or distinction.\n"
    "\n"
    'Output format (strict): a numbered list, one header per line, keeping '
    "the same order as the numbered passages. Example:\n"
    "1. [SECTION] Data Preprocessing\n"
    "2. [SUBSECTION] Handling Missing Values\n"
    "3. [SUBSECTION] Outlier Detection\n"
    "4. [SECTION] Machine Learning Applications\n"
)

_FAMILY_BATCH_SELF_CHECK = (
    "Before returning, verify:\n"
    "1. Every numbered passage received exactly one header.\n"
    "2. All headers are distinct from each other.\n"
    "3. SUBSECTION headers are narrower than their SECTION header.\n"
    "4. Each header is preferably 4 to 12 words, at most 15, and descriptive.\n"
    "5. Headers appear in the same order as the numbered passages.\n"
    "\n"
    "Return ONLY the numbered list of headers."
)

# One-shot regeneration call: fixes a single rejected header (one LLM call),
# explicitly warned against replicating the rejected header or the recent ones.
_REGENERATE_PROMPT = (
    "A passage in a university textbook needs a short Table-of-Contents style "
    "header. The previous attempt ({header!r}) was rejected and must not be "
    "reused.\n"
    "Rules:\n"
    "- Write the header in {language}. Do not switch to another language.\n"
    "- Prefer 4 to 12 words and use at most {max_words} words.\n"
    "- State the dominant subject plus the function, process, relation, cause, "
    "result, or distinction taught about it.\n"
    "- Use the BOOK HEADING as a topic anchor. Keep it unchanged only when it "
    "already describes the dominant content specifically.\n"
    "- No commas, no lists, no numbering, no question marks, no explanation.\n"
    "- Avoid starting with Introduction to, Overview of, A Comprehensive, "
    "Analysis of, or Summary of.\n"
    "- Do NOT use any of these already-taken headers: {reject}\n"
    "- Return ONLY the header.\n\n"
    "## BOOK HEADING\n{book_heading}\n\n## Representative content\n{content}"
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

def _first_n_words(text: str, n: int) -> str:
    words = [w for w in text.split() if w.strip()]
    return " ".join(words[:n])


def representative_context(text: str, max_words: int) -> str:
    """Sample the beginning, middle and end instead of anchoring on the opening."""
    words = [w for w in text.split() if w.strip()]
    if len(words) <= max_words:
        return " ".join(words)
    first_n = max(1, int(max_words * 0.4))
    middle_n = max(1, int(max_words * 0.3))
    last_n = max(1, max_words - first_n - middle_n)
    middle_start = max(first_n, (len(words) - middle_n) // 2)
    return (
        "[BEGINNING] " + " ".join(words[:first_n])
        + "\n[MIDDLE] " + " ".join(words[middle_start : middle_start + middle_n])
        + "\n[END] " + " ".join(words[-last_n:])
    )


def _normalize_heading(value: str) -> str:
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value or "")
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return re.sub(r"[^\w]+", " ", value.lower(), flags=re.UNICODE).strip()


def titles_too_similar(title: str, used_titles: set[str]) -> bool:
    """Catch normalized and near-duplicate Arabic/English navigation labels."""
    normalized = _normalize_heading(title)
    words = {
        (word[1:] if _is_arabic(word) and word.startswith("و") and len(word) > 3 else word)
        for word in normalized.split()
        if len(word) > 1
    }
    for used in used_titles:
        other_normalized = _normalize_heading(used)
        if normalized == other_normalized:
            return True
        other = {
            (word[1:] if _is_arabic(word) and word.startswith("و") and len(word) > 3 else word)
            for word in other_normalized.split()
            if len(word) > 1
        }
        if not words or not other:
            continue
        overlap = len(words & other)
        union = len(words | other)
        if overlap / union >= 0.8:
            return True
        smaller = min(len(words), len(other))
        if smaller >= 2 and overlap == smaller and abs(len(words) - len(other)) <= 2:
            return True
    return False


def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06ff]", text or ""))


_ARABIC_TITLE_STOP = {
    "في", "من", "عن", "على", "الى", "مع", "بين", "عبر", "دور", "شرح",
    "اثر", "اثار", "اهميه", "مقدمه", "مفهوم", "كيفيه",
}

# Signals that a compact Arabic heading says something specific about its
# subject. These are intentionally stems so common inflections (وظيفة/وظائف,
# تأثير/تأثيرات, etc.) are handled without requiring a full Arabic stemmer.
_ARABIC_DESCRIPTIVE_CUE_STEMS = (
    "دور", "وظيف", "الي", "عمل", "تركيب", "تكوين", "انتقال", "نقل",
    "تنظيم", "استجاب", "تاثير", "علاق", "سبب", "نتيج", "مرح",
    "اكتشاف", "انتاج", "افراز", "حفاظ", "مقارن", "تمييز", "تقسيم",
    "انقسام", "جهد", "توازن", "اتزان",
)

_ARABIC_STRUCTURAL_HEADING_WORDS = {
    "قسم", "القسم", "وحدة", "الوحدة", "فصل", "الفصل", "باب", "الباب",
}


def _has_arabic_descriptive_cue(title: str) -> bool:
    """Whether a compact Arabic title expresses a relation or precise process."""
    words = _normalize_heading(title).split()
    return any(
        word.startswith(stem)
        for word in words
        for stem in _ARABIC_DESCRIPTIVE_CUE_STEMS
    )


def _book_heading_is_descriptive(book_heading: str) -> bool:
    """Conservatively recognize headings that are already useful on their own.

    This is deliberately an adequacy test rather than an equality test. A
    short Arabic heading can pass when it names a function/process precisely;
    longer clean headings can also pass. Numbered structural labels remain
    context, not final UI titles.
    """
    normalized = _normalize_heading(book_heading)
    words = normalized.split()
    if not words or _blocklisted(book_heading):
        return False
    if any(word in _ARABIC_STRUCTURAL_HEADING_WORDS for word in words):
        return False
    if re.search(r"\d", normalized):
        return False
    if not _is_arabic(book_heading):
        return True
    return _has_arabic_descriptive_cue(book_heading) or len(words) >= 4


def _short_arabic_title_is_too_broad(title: str, content: str) -> bool:
    """Flag only terse category labels, not every Arabic title under 4 words."""
    if _word_count(title) >= 4 or _word_count(content) < 80:
        return False
    return not _has_arabic_descriptive_cue(title)


def _arabic_content_terms(text: str) -> set[str]:
    normalized = _normalize_heading(text)
    terms: set[str] = set()
    for raw in normalized.split():
        word = raw
        if word.startswith("و") and len(word) > 4:
            word = word[1:]
        if word.startswith("ال") and len(word) > 4:
            word = word[2:]
        for suffix in ("يات", "ات", "يه", "ية", "ها"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
        if len(word) >= 3 and word not in _ARABIC_TITLE_STOP:
            terms.add(word)
    return terms


def _arabic_title_is_grounded(title: str, content: str, book_heading: str) -> bool:
    title_terms = _arabic_content_terms(title)
    if not title_terms:
        return False
    source_terms = _arabic_content_terms(f"{book_heading} {content}")
    matched = 0
    for term in title_terms:
        if term in source_terms or any(
            min(len(term), len(source)) >= 4
            and (term.startswith(source) or source.startswith(term))
            for source in source_terms
        ):
            matched += 1
    return matched / len(title_terms) >= 0.5


def _merely_rephrases_arabic_heading(title: str, book_heading: str) -> bool:
    title_terms = _arabic_content_terms(title)
    heading_terms = _arabic_content_terms(book_heading) - {"قسم", "وحدة", "وحده", "فصل"}
    if not title_terms or not heading_terms:
        return False
    matched = sum(
        1
        for term in title_terms
        if term in heading_terms
        or any(
            min(len(term), len(source)) >= 4
            and (term.startswith(source) or source.startswith(term))
            for source in heading_terms
        )
    )
    return matched / len(title_terms) >= 0.7


def descriptive_fallback(
    content: str, book_heading: str = "", index: int = 1, max_words: int = 15,
    language: str | None = None,
) -> str:
    """Build a grounded emergency label; never return ``Untitled``/blank."""
    heading = clean_title(re.sub(r"<[^>]+>|[*_#]", "", book_heading), max_words)
    expected = language or detect_language(content)
    if expected == "ar":
        words = [
            word for word in _normalize_heading(content).split()
            if len(word) >= 3 and word not in _ARABIC_TITLE_STOP
        ]
        heading_terms = _arabic_content_terms(heading)
        bigrams = Counter(
            " ".join(words[i : i + 2]) for i in range(max(0, len(words) - 1))
        )
        phrases = [
            phrase for phrase, _ in bigrams.most_common()
            if _arabic_content_terms(phrase) - heading_terms
        ][:2]
        base = heading or f"موضوع القسم {index}"
        if phrases:
            return clean_title(f"{base}: {' و'.join(phrases)}", max_words)
        return clean_title(f"شرح محتوى {base}", max_words)
    base = heading or f"Section Topic {index}"
    return clean_title(f"Content of {base}", max_words)


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
        key = re.sub(r"[^\w-]", "", w, flags=re.UNICODE)
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
    # Strip the level tag the model echoes from the family-batch prompt format
    # (e.g. "1. [SECTION] Data Preprocessing" -> "Data Preprocessing").
    title = re.sub(r"^\s*\[(?:SECTION|SUBSECTION)\]\s*", "", title).strip()
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
    # A terse, noun-phrase qualifier ("Cost Functions") is a real topic
    # qualifier; anything the tagger reads as a verb/adverb-headed fragment
    # ("Predicts Values", "Understanding the approach") is generated prose.
    return not is_noun_phrase(clause)


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
    # Reject immediately repeated words ("Proteins Proteins", "Cell Cell"), so
    # the caller regenerates instead of shipping a duplicated header.
    if re.search(r"\b(\w+)\s+\1\b", title, re.IGNORECASE):
        return False
    # Reject verb/adverb-headed or sentence-fragment titles ("Fortunately",
    # "Stepping Back", "Learning Algorithm Would") via part-of-speech tagging.
    if not is_noun_phrase(title):
        return False
    # Reject question-form titles.
    if "?" in title:
        return False
    # Reject comma-separated keyword lists and ampersand joins.
    if "," in title or "،" in title or "&" in title:
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
    Prefers acronyms, k-/t- prefixed terms, then the first fitting POS noun
    chunk, then a single POS-verified keyword. Preserves technical casing
    (k-Nearest Neighbors, t-SNE, PCA) via ``clean_title`` downstream.
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
        before = preview[:m.start()].split()
        j = len(before)
        if j >= len(word_map) or m.group(0) not in word_map[j]:
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

    # 3) First fitting POS noun chunk ("Linear Regression", "Support Vector
    #    Machine"). PoS tagging already excludes verb/adverb-headed fragments.
    chunk = first_noun_chunk(preview, max_words)
    if chunk:
        return chunk

    # 4) Single most salient POS-verified keyword.
    for tok in preview.split():
        key = _strip_punct(tok)
        if key and key.lower() not in _SMALL_WORDS and (tok[:1].isupper() or len(key) > 3):
            if is_noun_phrase(key):
                return _preserve_case(key)
    return ""


def _extends_run(token: str) -> bool:
    key = _strip_punct(token).lower()
    if not key or key in _SMALL_WORDS:
        return False
    # Lowercase noun continuation (neighbors, regression, trees...).
    if token[:1].islower():
        return len(key) >= 3
    # Another capitalized word: stop (likely a new phrase).
    return False


def _trim_phrase(run: list[str]) -> str:
    """Trim leading/trailing small-word filler from a run."""
    while run and (_strip_punct(run[0]).lower() in _SMALL_WORDS):
        run = run[1:]
    while run and (_strip_punct(run[-1]).lower() in _SMALL_WORDS
                   or not is_noun_phrase(_strip_punct(run[-1]))):
        run = run[:-1]
    return " ".join(run).strip() if run else ""


def _strip_punct(token: str) -> str:
    return re.sub(r"[^\w-]", "", token, flags=re.UNICODE)


def _preserve_case(key: str) -> str:
    return key


def _looks_like_heading(label: str, max_words: int, min_words: int = 2) -> bool:
    """True only if ``label`` reads like a noun heading, not a sentence."""
    if not label:
        return False
    if "," in label or "?" in label or "\n" in label:
        return False
    if re.search(r"\b\d+[.:)]", label):
        return False
    n = len(label.split())
    if n < min_words or n > max_words:
        return False
    return is_noun_phrase(label)


def _extract_keyword(content: str) -> str:
    """The single most salient POS-verified keyword, or "" if none."""
    preview = _first_n_words(content, 250)
    for tok in preview.split():
        key = _strip_punct(tok)
        if not key or key.lower() in _SMALL_WORDS:
            continue
        if (tok[:1].isupper() or len(key) > 3) and is_noun_phrase(key):
            return _preserve_case(key)
    return ""


def _safe_fallback(content: str, max_words: int, min_words: int = 2) -> str:
    """Deterministic last-resort label that is guaranteed to be a noun heading.

    Returns "" when nothing safe can be extracted, so a caller can skip it
    rather than emit a sentence fragment, a bare function word, or a too-short
    single-word stub ("Option", "Data") as a title.
    """
    label = _extract_noun_phrase(content, max_words)
    if label and _looks_like_heading(label, max_words, min_words):
        return _finalize(clean_title(label, max_words))
    keyword = _extract_keyword(content)
    if keyword and _looks_like_heading(keyword, max_words, min_words):
        return _finalize(clean_title(keyword, max_words))
    return ""


# ---------------------------------------------------------------------------
# LLM call + orchestration
# ---------------------------------------------------------------------------


def _make_client():
    # Titles always run with reasoning off: qwen3-class models otherwise burn
    # the small title token budget on thinking and return an empty message.
    return create_llm_client(
        local_url=TITLE_LMS_URL,
        local_model=TITLE_MODEL,
        local_reasoning="off",
    )


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
    preview = representative_context(content, preview_words)
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

    # Safety net: deterministic noun-phrase label (never a sentence fragment).
    label = _safe_fallback(content, fallback_max_words)
    logger.warning("Title fallback used | min=%d max=%d label=%r", min_words, max_words, label)
    return label


_ROMAN_SUFFIX = ("II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")


def _build_batch_prompt(
    level_block: str,
    previews: list[str],
    before: list[str],
) -> str:
    """Assemble a single prompt labeling ``previews``, aware of sibling titles."""
    passages = "\n\n".join(f"{i}. {p}" for i, p in enumerate(previews, 1))
    seen = "\n".join(
        f"- {t}" for t in before if t
    ) or "(none yet)"
    return (
        _TITLE_CORE
        + "\n\n"
        + level_block
        + "\n\n"
        + _STYLE_RULES
        + "\n\n"
        + _ABSTRACTION_RULE
        + "\n\n"
        + "Headers already used by neighboring passages (do NOT reuse any of "
        "these):\n"
        + seen
        + "\n\n"
        + _BATCH_RULE
        + "\n\n"
        + _BATCH_SELF_CHECK
        + "\n\n## Passages\n"
        + passages
    )


def _parse_title_list(raw: str) -> list[str]:
    """Pull one header per line out of a model's numbered-list response."""
    headers: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading list markers: "1.", "3)", "4:", "- ", "• ".
        line = re.sub(r"^\s*(?:[-*•·]\s+|\d+[.):-]?\s+)", "", line)
        line = line.strip()
        if line:
            headers.append(line)
    return headers


def make_titles_unique(
    titles: list[str], contents: list[str], max_words: int
) -> list[str]:
    """Guarantee every title in the batch is distinct (safety net)."""
    used: dict[str, int] = {}
    out: list[str] = []
    for i, t in enumerate(titles):
        if t and t not in used:
            used[t] = 1
            out.append(t)
            continue
        # t is duplicate/blank: try a deterministic label specific to this chunk.
        label = _safe_fallback(contents[i], max_words)
        if label and label not in used:
            used[label] = 1
            out.append(label)
            continue
        # Last resort: append a Roman numeral to force uniqueness. If the base
        # already ends in one (e.g. "Machine Learning Overview II") continue
        # from the NEXT numeral instead of doubling it into "... II II".
        base = t or label or "Key Concept"
        start = 2
        for rank, suffix in enumerate(_ROMAN_SUFFIX, start=2):
            if base.endswith(suffix):
                start = rank + 1
                base = base[: -len(suffix)].rstrip()
                break
        for n in range(start, start + len(_ROMAN_SUFFIX)):
            cand = base if n == 1 else f"{base} {_ROMAN_SUFFIX[n - 2]}"
            if cand not in used:
                used[cand] = 1
                out.append(cand)
                break
    return out


def generate_batch_titles(
    contents: list[str],
    client=None,
    *,
    level: str = "subsection",
    before: list[str] | None = None,
) -> list[str]:
    """Label a group of sibling chunks in ONE call, keeping headers distinct.

    ``level`` selects the section vs subsection framing. ``before`` lists
    headers from earlier-completed batches so the model avoids reusing them.
    """
    contents = [c for c in contents]
    if not contents:
        return []
    if level == "section":
        level_block = _SECTION_LEVEL_BLOCK
        preview_words = SECTION_TITLE_CONTEXT_WORDS
        min_words = SECTION_TITLE_MIN_WORDS
        max_words = SECTION_TITLE_MAX_WORDS
        fallback_max_words = FALLBACK_SECTION_MAX_WORDS
    else:
        level_block = _SUBSECTION_LEVEL_BLOCK
        preview_words = SUBSECTION_TITLE_CONTEXT_WORDS
        min_words = SUBSECTION_TITLE_MIN_WORDS
        max_words = SUBSECTION_TITLE_MAX_WORDS
        fallback_max_words = FALLBACK_SUBSECTION_MAX_WORDS

    previews = [representative_context(c, preview_words) for c in contents]
    client = client or _make_client()

    titles: list[str] = [""] * len(contents)
    prompt = _build_batch_prompt(level_block, previews, before or [])
    try:
        raw = client.chat(
            prompt,
            system_prompt=None,
            temperature=TITLE_TEMPERATURE,
            max_tokens=max(400, len(contents) * 60),
            timeout=1800,
        )
    except Exception as exc:  # noqa: BLE001 - title failure must not break pipeline
        logger.warning("Batch title generation error: %s: %s", type(exc).__name__, exc)
        raw = ""

    headers = _parse_title_list(raw)
    for i, header in enumerate(headers[: len(contents)]):
        cleaned = clean_title(header, max_words)
        if validate_title(cleaned, min_words, max_words):
            titles[i] = _finalize(cleaned)

    # Backfill any chunk the batch model missed or invalidly labeled. A single
    # title-generator call (with its own retries + safe fallback) handles each
    # missing chunk rather than falling straight to keyword extraction.
    for i, t in enumerate(titles):
        if t:
            continue
        single = generate_section_title if level == "section" else generate_subsection_title
        label = single(client=client, content=contents[i])
        logger.warning("Batch title missing for idx=%d | retried=%r", i, label)
        titles[i] = label

    return make_titles_unique(titles, contents, fallback_max_words)


def make_title_client():
    """Public factory for the title-generation LLM client."""
    return _make_client()


def _level_params(level: str) -> tuple[int, int, int, int]:
    """(preview_words, min_words, max_words, fallback_max) for a title level."""
    if level == "section":
        return (
            SECTION_TITLE_CONTEXT_WORDS,
            SECTION_TITLE_MIN_WORDS,
            SECTION_TITLE_MAX_WORDS,
            FALLBACK_SECTION_MAX_WORDS,
        )
    return (
        SUBSECTION_TITLE_CONTEXT_WORDS,
        SUBSECTION_TITLE_MIN_WORDS,
        SUBSECTION_TITLE_MAX_WORDS,
        FALLBACK_SUBSECTION_MAX_WORDS,
    )


# Generic heading openers that are never a real concept ("Introduction to X",
# "Overview of X") -- separate from TITLE_BLOCKLIST's exact-filler matches.
_FILLER_PREFIXES: tuple[str, ...] = (
    "introduction to",
    "overview of",
    "analysis of",
    "summary of",
    "a comprehensive",
    "comprehensive",
    "definitions of",
    "key concepts",
    "key terms",
    "study of",
    "guide to",
)

_WEAK_TITLE_START_RE = re.compile(
    r"^\s*(?:قال|يقول|عن|روي\s+عن|رواه|رضي\s+الله|"
    r"(?:درس|lesson|chapter|section|unit)\s*[\d٠-٩]+|أكمل|اكمل|أجب|اجب|"
    r"أفكر|افكر|أتأمل|اتأمل|أقرأ|اقرأ|أبحث|ابحث|أحدد|احدد|أقيم|اقيم|"
    r"أضع|اضع|أنظم|انظم|أثري|اثري|أقارن|اقارن|أستنتج|استنتج|أناقش|"
    r"اناقش|أكتب|اكتب|اختر|اختار|حدد|read|answer|complete|choose|discuss)\b",
    re.IGNORECASE,
)


def _weak_or_citation_title(title: str) -> bool:
    stripped = title.strip()
    if not stripped:
        return True
    if stripped.startswith((">", "[", "{", "◇", "◆", "🔹", "•")):
        return True
    return bool(_WEAK_TITLE_START_RE.match(arabic_comparison_key(stripped)))


def _blocklisted(title: str) -> bool:
    """True when a title is a bare filler heading or starts with a filler prefix."""
    if not title:
        return True
    low = re.sub(r"\s+", " ", title.lower().strip())
    if low in TITLE_BLOCKLIST:
        return True
    for prefix in _FILLER_PREFIXES:
        if low.startswith(prefix):
            return True
    return False


def title_matches_language(title: str, expected_language: str) -> bool:
    """Require the title's dominant script to match the requested language."""
    arabic = len(re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", title or ""))
    latin = len(re.findall(r"[A-Za-z\u00C0-\u024F]", title or ""))
    if expected_language == "ar":
        return arabic > 0 and arabic >= latin * 2
    return latin > 0 and latin >= arabic * 2


def is_acceptable_title(
    title: str, content: str, level: str, used_titles: set[str],
    book_heading: str = "", expected_language: str | None = None,
) -> bool:
    """A title is acceptable when it passes format, blocklist, and exact-dup checks.

    ``content`` is the passage the title labels; used by callers that also want
    to run deeper checks (e.g. that the header is not lifted verbatim).
    """
    if not title or not title.strip():
        return False
    _, min_words, max_words, _ = _level_params(level)
    if not validate_title(title, min_words, max_words):
        return False
    if _blocklisted(title):
        return False
    if _weak_or_citation_title(title):
        return False
    if titles_too_similar(title, used_titles):
        return False
    expected_language = expected_language or detect_language(content)
    if not title_matches_language(title, expected_language):
        return False
    # A compact Arabic title is valid when it carries a specific process or
    # relation; word count alone is never a rejection reason.
    if _is_arabic(content) and _short_arabic_title_is_too_broad(title, content):
        return False
    repeats_heading = book_heading and (
        _normalize_heading(title) == _normalize_heading(book_heading)
        or (
            _is_arabic(title)
            and _merely_rephrases_arabic_heading(title, book_heading)
        )
    )
    # Reusing the source heading is allowed when that heading already describes
    # the chunk adequately. Similarity itself is not an error.
    if repeats_heading and not _book_heading_is_descriptive(book_heading):
        return False
    if _is_arabic(title) and _word_count(content) >= 10 and not _arabic_title_is_grounded(
        title, content, book_heading
    ):
        return False
    return True


def _unpack_entry(entry) -> tuple[str, str, str, str]:
    level = entry[0]
    content = entry[1]
    heading = (entry[2] or "(none)") if len(entry) >= 3 else "(none)"
    language = (
        entry[3]
        if len(entry) >= 4 and entry[3] in {"ar", "en"}
        else detect_language(content)
    )
    return level, content, heading, language


def _build_family_prompt(entries, before: list[str]) -> str:
    """Assemble a single-label prompt for a mixed batch of parents + children.

    ``entries`` is a list of ``(level, preview)`` in document order where each
    parent entry is immediately followed by its own children's entries. Every
    entry is tagged [SECTION] / [SUBSECTION] so the model can keep subsection
    headers narrower than their section header.
    """
    passages = "\n\n".join(
        f"{i}. [{level.upper()}]\nBOOK HEADING: {heading}\n"
        f"REQUIRED TITLE LANGUAGE: {'Arabic' if language == 'ar' else 'English'}\n"
        f"REPRESENTATIVE CONTENT:\n{preview}"
        for i, entry in enumerate(entries, 1)
        for level, preview, heading, language in [_unpack_entry(entry)]
    )
    seen = "\n".join(f"- {t}" for t in before if t) or "(none yet)"
    return (
        _TITLE_CORE
        + "\n\n"
        + _SECTION_LEVEL_BLOCK
        + "\n\n"
        + _SUBSECTION_LEVEL_BLOCK
        + "\n\n"
        + _STYLE_RULES
        + "\n\n"
        + _ABSTRACTION_RULE
        + "\n\n"
        + "Headers already used by neighboring passages (do NOT reuse any of "
        "these):\n"
        + seen
        + "\n\n"
        + _FAMILY_BATCH_RULE
        + "\n\n"
        + _FAMILY_BATCH_SELF_CHECK
        + "\n\n## Passages\n"
        + passages
    )


def generate_family_batch_titles(
    entries,
    client=None,
    *,
    before: list[str] | None = None,
) -> list[str]:
    """Label a mixed batch of parent + child passages in ONE LLM call.

    ``entries`` is a flat list of ``(level, content)`` in document order, where
    each parent's entry is immediately followed by its children's entries. The
    model sees the whole batch together (section + its subsections tagged) so
    sibling and parent/child headers stay consistent and distinct. Returns one
    title per entry; entries the model missed or invalidly titled come back as
    ``""`` so the caller can regenerate them individually.
    """
    if not entries:
        return []
    unpacked = [_unpack_entry(entry) for entry in entries]
    previews = [
        representative_context(content, _level_params(level)[0])
        for level, content, _, _ in unpacked
    ]
    tagged = [
        (level, preview, heading, language)
        for (level, _, heading, language), preview in zip(unpacked, previews)
    ]

    client = client or _make_client()
    prompt = _build_family_prompt(tagged, before or [])
    try:
        raw = client.chat(
            prompt,
            system_prompt=None,
            temperature=TITLE_TEMPERATURE,
            max_tokens=max(600, len(entries) * 60),
            timeout=1800,
        )
    except Exception as exc:  # noqa: BLE001 - title failure must not break pipeline
        logger.warning("Family batch title error: %s: %s", type(exc).__name__, exc)
        raw = ""

    headers = _parse_title_list(raw)
    titles: list[str] = [""] * len(entries)
    for i, header in enumerate(headers[: len(entries)]):
        level = unpacked[i][0]
        _, min_words, max_words, _ = _level_params(level)
        cleaned = clean_title(header, max_words)
        if validate_title(cleaned, min_words, max_words):
            finalized = _finalize(cleaned)
            if title_matches_language(finalized, unpacked[i][3]):
                titles[i] = finalized
    return titles


def regenerate_title(
    client=None,
    content: str = "",
    level: str = "section",
    reject: list[str] | None = None,
    used_titles: set[str] | None = None,
    book_heading: str = "",
    expected_language: str | None = None,
) -> str:
    """Fix one rejected header with ONE extra LLM call, then a deterministic fallback.

    The model is told the rejected header and the most recent already-taken
    headers so it does not regenerate a duplicate. If the single attempt still
    fails the format / blocklist / exact-dup checks, a deterministic noun-phrase
    label is returned.
    """
    client = client or _make_client()
    preview = representative_context(content, _level_params(level)[0])
    _, min_words, max_words, fallback_max = _level_params(level)
    reject = list(dict.fromkeys(t for t in (reject or []) if t))
    used = used_titles or set()
    recent = list(dict.fromkeys(reject + sorted(used)))[: TITLE_CONTEXT_RECENT]
    reject_str = ", ".join(f"{t!r}" for t in recent) or "(none)"
    header = reject[0] if reject else "the previous attempt"
    expected_language = expected_language or detect_language(content)

    try:
        raw = client.chat(
            _REGENERATE_PROMPT.format(
                header=header,
                max_words=max_words,
                reject=reject_str,
                content=preview,
                book_heading=book_heading or "(none)",
                language="Arabic" if expected_language == "ar" else "English",
            ),
            system_prompt=None,
            temperature=TITLE_TEMPERATURE,
            max_tokens=max(200, int(max_words * 14)),
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001 - regeneration must not break pipeline
        logger.warning("Title regeneration error: %s: %s", type(exc).__name__, exc)
        raw = ""

    cleaned = clean_title(raw, max_words)
    acceptable = bool(cleaned) and is_acceptable_title(
        cleaned, content, level, used, book_heading, expected_language
    )
    if acceptable:
        return _finalize(cleaned)
    label = _safe_fallback(content, fallback_max)
    if not title_matches_language(label, expected_language):
        label = descriptive_fallback(
            content,
            book_heading,
            max_words=fallback_max,
            language=expected_language,
        )
    if label and label not in used:
        return label
    return cleaned


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


# ---------------------------------------------------------------------------
# Title review (post-generation spell-check pass)
# ---------------------------------------------------------------------------

# Reviewer prompt: one call per section carries the section header plus its
# subsection headers, each with the opening of its passage. The model scores
# each header against its passage and sibling context; 8-10 keep, 4-7 refine,
# 1-3 replace. Bad signs are spelled out generically (POS/grammar based), not
# as a hardcoded list.
_REVIEW_PROMPT = (
    "You are a senior copy editor for a university textbook. For every "
    "passage below you see the opening of the passage and its current SECTION "
    "or SUBSECTION header. Judge how well the header would label that passage "
    "in the Table of Contents.\n"
    "\n"
    "Score every header 1 to 10:\n"
    "- 8-10  Excellent header, keep it exactly as-is (KEEP).\n"
    "- 4-7   Correct concept but weak wording; return a tighter header for the "
    "same concept (REFINE).\n"
    "- 1-3   Bad header; write a completely new header for the passage "
    "(REPLACE).\n"
    "\n"
    "A good header is preferably 4 to 12 words (never more than 15), names "
    "the dominant subject and what the passage teaches about it, and is "
    "supported by the representative content. The source book heading guides "
    "the topic and may be kept unchanged when it already describes the "
    "dominant content specifically.\n"
    "\n"
    "A bad header (score 1-3) commonly:\n"
    "- is a phrase lifted verbatim from the passage prose (e.g. \"Yet Another "
    "Important Unsupervised Task\", \"Learning Algorithm Would\"),\n"
    "- starts with an adverb or a bare verb (e.g. \"Fortunately\", "
    "\"Stepping\") or otherwise is not a noun phrase,\n"
    "- ends in a dangling verb or auxiliary (e.g. \"... Would\", "
    "\"... Predicts\"),\n"
    "- carries trailing noise like a Roman numeral or number (e.g. \"Machine "
    "Learning Overview II\"),\n"
    "- is generic filler or only a broad category when a specific process, "
    "function, relation, cause, or result is supported,\n"
    "- is invented, based on an incidental sentence, or does not cover the "
    "dominant passage content.\n"
    "\n"
    "The SECTION header must stay broader than its SUBSECTION headers, and "
    "every header in the group must stay distinct from the others.\n"
    "\n"
    "Reply with ONLY a numbered list, one line per header, in EXACTLY the same "
    "order. Use exactly this format:\n"
    "1. score=9 KEEP\n"
    "2. score=2 REPLACE: Regularization Techniques\n"
    "3. score=6 REFINE: Cross-Validation\n"
)

# Verification prompt: re-score a single rewritten header to confirm the fix
# actually stuck (second pass over replacements).
_VERIFY_PROMPT = (
    "Judge this textbook header against the passage below. Reply with ONLY an "
    "integer score from 1 to 10, where 10 is a perfect Table of Contents "
    "entry, 8-9 excellent, 4-7 usable but weak, and 1-3 bad.\n"
    "Header: {title}\n"
    "Passage (opening):\n{content}\n"
    "Score: "
)

_REVIEW_SCORE_RE = re.compile(
    r"score\s*[=:]\s*(\d{1,2})", re.IGNORECASE
)
_REVIEW_LINE_RE = re.compile(
    r"^\s*(\d+)\s*[.):\-]?\s*score\s*[=:]\s*(\d{1,2})\s+"
    r"(KEEP|REFINE|REPLACE)\s*[):\-]?\s*(.*)$",
    re.IGNORECASE,
)


def _review_level_bounds(level: str) -> tuple[int, int]:
    if level == "section":
        return SECTION_TITLE_MIN_WORDS, SECTION_TITLE_MAX_WORDS
    return SUBSECTION_TITLE_MIN_WORDS, SUBSECTION_TITLE_MAX_WORDS


def _parse_review(raw: str) -> dict[int, tuple[int, str, str]]:
    """Map header index -> (score, verdict, reviewer's proposed title)."""
    rows: dict[int, tuple[int, str, str]] = {}
    for line in raw.splitlines():
        m = _REVIEW_LINE_RE.match(line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        score = int(m.group(2))
        verdict = m.group(3).upper()
        title = m.group(4).strip().strip('"').strip("'").strip()
        rows[idx] = (score, verdict, title)
    return rows


def _verify_title(client, title: str, content: str) -> int | None:
    """Re-score one rewritten header; None when the model reply is unusable."""
    preview = representative_context(content, TITLE_REVIEW_CONTEXT_WORDS)
    try:
        raw = client.chat(
            _VERIFY_PROMPT.format(title=title, content=preview),
            system_prompt=None,
            temperature=TITLE_TEMPERATURE,
            max_tokens=TITLE_MAX_TOKENS,
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001 - review must not break pipeline
        logger.warning("Title verify error: %s: %s", type(exc).__name__, exc)
        return None
    m = _REVIEW_SCORE_RE.search(raw or "")
    if not m:
        m = re.search(r"\b(\d{1,2})\b", raw or "")
        if not m:
            return None
    return min(10, int(m.group(1)))


_CANDIDATES_PROMPT = (
    "A passage in a university textbook currently has the header {header!r}, but "
    "that header does not label what the passage teaches. Write {n} alternative "
    "Table-of-Contents style headers for this passage, one per line, numbered "
    "1..{n}. Output ONLY the numbered list -- no explanations.\n"
    "Passage:\n{preview}\n"
)


def _generate_candidates(
    client, content: str, *, level: str, header: str = "", n: int = 4
) -> list[str]:
    """Propose up to ``n`` candidate headers for one passage in ONE LLM call.

    Candidates are locally filtered (title-style + word bounds) and any
    candidate lifted verbatim from the passage is rejected, since a header that
    already appears in the text is usually a proper-noun example, not the real
    section topic (e.g. "Crown Prince Salman"). Returns candidates in model
    order.
    """
    min_words, max_words = _review_level_bounds(level)
    preview = representative_context(content, TITLE_REVIEW_CONTEXT_WORDS)
    try:
        raw = client.chat(
            _CANDIDATES_PROMPT.format(header=header, n=n, preview=preview),
            system_prompt=None,
            temperature=TITLE_TEMPERATURE,
            max_tokens=max(300, TITLE_MAX_TOKENS * n),
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001 - candidate failure must not break review
        logger.warning("Title candidate error: %s: %s", type(exc).__name__, exc)
        return []

    out: list[str] = []
    for line in _parse_title_list(raw or ""):
        cand = clean_title(line, max_words)
        if not cand or not validate_title(cand, min_words, max_words):
            continue
        if title_appears_in_text(cand, content):
            logger.debug("Title candidate rejected as verbatim | cand=%r", cand)
            continue
        if cand not in out:
            out.append(cand)
    return out[:n]


def _apply_verdict(
    client, item, level: str, score: int, verdict: str, proposed: str
) -> str:
    """Turn one reviewer verdict into the final title for ``item``.

    Rewritten (REPLACE / REFINE) titles are picked best-of-N from a single
    candidate call (``TITLE_REVIEW_CANDIDATES`` alternatives), re-scored once
    by the verifier, and -- when still weak -- retried with another candidate
    batch (``TITLE_REVIEW_RETRIES`` times) before the deterministic fallback.
    """
    min_words, max_words = _review_level_bounds(level)

    if verdict == "KEEP" or score >= TITLE_REVIEW_GOOD_SCORE:
        if validate_title(item.title or "", min_words, max_words):
            return item.title or ""
        # A "KEEP" on a title that fails validation (e.g. a one-word stub like
        # "Option") is treated as a rewrite request instead of being trusted.
        verdict = "REPLACE"
        score = 1

    # Candidate pool: the reviewer's proposal first (already scored by it),
    # then best-of-N from the candidate generation call(s).
    pool: list[str] = []
    if proposed:
        cand = clean_title(proposed, max_words)
        if cand and validate_title(cand, min_words, max_words):
            pool.append(cand)

    best = ""
    best_score = 0
    for _ in range(max(1, TITLE_REVIEW_RETRIES + 1)):
        for cand in _generate_candidates(
            client,
            item.content,
            level=level,
            header=item.title or "",
            n=TITLE_REVIEW_CANDIDATES,
        ):
            if cand not in pool:
                pool.append(cand)
        if not pool:
            continue

        # Best local candidate = the first valid alternative (proposal wins,
        # then model order). Re-scored once per round.
        candidate = pool.pop(0)
        cand_score = _verify_title(client, candidate, item.content)
        if cand_score is None or cand_score < TITLE_REVIEW_GOOD_SCORE:
            if cand_score is not None and cand_score > best_score:
                best, best_score = candidate, cand_score
            continue
        return candidate

    if best and best_score >= TITLE_REVIEW_POLISH_MIN:
        return best
    if score >= TITLE_REVIEW_POLISH_MIN and item.title:
        return item.title or ""
    fallback_max = (
        FALLBACK_SECTION_MAX_WORDS
        if level == "section"
        else FALLBACK_SUBSECTION_MAX_WORDS
    )
    return _safe_fallback(item.content, fallback_max)


def review_titles(parents, children, client=None) -> None:
    """Score every generated header against its own passage; fix the bad ones.

    One reviewer LLM call per SECTION carries the section header plus its
    SUBSECTION headers with the opening of each passage, so each header is
    judged with sibling context. Every rewritten header is then re-scored by a
    second verification call (two-pass), using ``TITLE_REVIEW_GOOD_SCORE`` /
    ``TITLE_REVIEW_POLISH_MIN`` score bands.
    """
    if not TITLE_REVIEW_ENABLED:
        return
    client = client or _make_client()

    by_parent: dict[str, list] = {}
    for c in children:
        if c.title:
            by_parent.setdefault(c.parent_id, []).append(c)

    for parent in parents:
        subs = by_parent.get(parent.parent_id, [])
        items = [(parent, "section")] + [(c, "subsection") for c in subs]
        rows: dict[int, tuple[int, str, str]] = {}
        try:
            entries = []
            for i, (item, level) in enumerate(items, 1):
                label = "SECTION" if level == "section" else "SUBSECTION"
                preview = representative_context(
                    item.content, TITLE_REVIEW_CONTEXT_WORDS
                )
                heading = getattr(item, "book_heading", None) or "(none)"
                verbatim = title_appears_in_text(item.title or "", item.content)
                note = " (NOTE: this header text appears verbatim in the passage)" if verbatim else ""
                entries.append(
                    f"{i}. {label}: {item.title!r}{note}\n"
                    f"   Book heading: {heading}\n"
                    f"   Representative passage: {preview}"
                )
            prompt = _REVIEW_PROMPT + "\n\n## Passages to review\n\n" + "\n\n".join(entries)
            raw = client.chat(
                prompt,
                system_prompt=None,
                temperature=TITLE_TEMPERATURE,
                max_tokens=max(600, len(items) * 120),
                timeout=1800,
            )
            rows = _parse_review(raw)
        except Exception as exc:  # noqa: BLE001 - review must not break pipeline
            logger.warning("Title review error: %s: %s", type(exc).__name__, exc)
            continue

        if len(rows) < len(items):
            logger.warning(
                "Reviewer returned %d rows for %d headers",
                len(rows),
                len(items),
            )
        for i, (item, level) in enumerate(items, 1):
            if i not in rows:
                continue
            score, verdict, proposed = rows[i]
            final = _apply_verdict(client, item, level, score, verdict, proposed)
            if final and final != item.title:
                logger.info(
                    "TITLE REVIEW | %s | %r -> %r", level, item.title, final
                )
                item.title = final
