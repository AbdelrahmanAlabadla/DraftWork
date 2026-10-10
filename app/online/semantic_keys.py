from __future__ import annotations

import re
import unicodedata
from typing import Any


_SUPERSCRIPT_TRANSLATION = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
})


def formula_semantic_key(value: Any) -> tuple[str, str] | None:
    """Return a label-insensitive key for a simple symbolic equality.

    Descriptive subscripts do not create a new physical relationship. For
    example, ``E_photon = hν`` and ``E_quantum = hν`` normalize to the same
    key while ``ΔE = hν`` remains distinct from ``E = hν``.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).translate(
        _SUPERSCRIPT_TRANSLATION
    ).casefold()
    if text.count("=") != 1:
        return None
    left, right = text.split("=", 1)
    left = re.sub(r"\s+", "", left).split("_", 1)[0]
    right = re.sub(r"\s+", "", right)
    left = re.sub(r"[^\wΔδ]+", "", left, flags=re.UNICODE)
    right = re.sub(r"[^\w+−\-*/^().]+", "", right, flags=re.UNICODE)
    if not left or not right or len(left) > 12:
        return None
    return left, right


def formulas_equivalent(left: Any, right: Any) -> bool:
    first = formula_semantic_key(left)
    second = formula_semantic_key(right)
    return bool(first and second and first == second)
