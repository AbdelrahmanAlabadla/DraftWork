"""Language detection and Arabic-aware text normalization utilities.

Language is detected from the uploaded document content (``document_language``)
and is independent of the UI language. Detection is a simple script-ratio
heuristic: no external dependency, deterministic, works offline.

Normalization here produces *comparison keys only*. Original stored/displayed
text is never modified.
"""

from __future__ import annotations

import re
import unicodedata

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F]")

# Tashkeel (harakat) and Quranic annotation marks — safe to strip for comparison.
_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

# Alef variants normalize to plain alef for comparison only.
_ALEF_VARIANTS = "\u0623\u0625\u0622"

_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


def detect_language(text: str) -> str:
    """Return ``"ar"`` or ``"en"`` based on the dominant script in *text*.

    Arabic wins on ties so mixed documents with any meaningful Arabic share of
    letters are treated as Arabic (English scaffolding words are common even in
    Arabic academic PDFs).
    """
    arabic = len(_ARABIC_RE.findall(text or ""))
    latin = len(_LATIN_RE.findall(text or ""))
    if arabic == 0 and latin == 0:
        return "en"
    return "ar" if arabic * 10 >= latin else "en"


def is_rtl(language: str) -> bool:
    return language == "ar"


def _strip_arabic_diacritics(text: str) -> str:
    return _TASHKEEL_RE.sub("", text)


def _normalize_alef(text: str) -> str:
    for variant in _ALEF_VARIANTS:
        text = text.replace(variant, "\u0627")
    return text


def arabic_comparison_key(text: str) -> str:
    """Comparison form of Arabic text: tashkeel stripped, alef variants folded.

    Deliberately does NOT convert ``\u0629 \u2192 \u0647`` or ``\u0649 \u2192 \u064A``:
    those can change word identity. Used only for dedup/search keys.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = _strip_arabic_diacritics(text)
    text = _normalize_alef(text)
    return text


def normalize_text(text: str) -> list[str]:
    """Lowercase/strip diacritics/punctuation tokens for dedup — script aware.

    For Latin text this behaves exactly as before (NFKD + combining removal).
    For Arabic text it uses the Arabic comparison form instead, preserving
    letter identity without destroying tashkeel-based distinctions elsewhere.
    """
    text = str(text)
    if _ARABIC_RE.search(text):
        text = arabic_comparison_key(text)
        return [tok for tok in _PUNCT_RE.split(text) if tok]
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [tok for tok in _PUNCT_RE.split(text.lower()) if tok]
