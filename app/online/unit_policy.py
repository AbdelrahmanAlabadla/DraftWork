"""Subject-aware measurement-unit policy and deterministic unit diagnostics.

DraftWork does not convert units.  This module only decides how strictly the
source units must be preserved and reports contradictory unit systems in the
evidence so the generator/validator can act deterministically.

Policy levels
-------------
``essential``
    The unit is part of the assessed content (exact source data, unit
    conversion, dimensional analysis). The literal value and unit must survive.
``constrained``
    Physics/chemistry/biology and other sciences: prefer the source unit where
    it carries the method or dimensional meaning; a change is allowed only when
    it is dimensionally valid and derivable from stated givens.
``flexible``
    Mathematics and non-STEM: a unit change is acceptable when it preserves the
    mathematical concept and the dimension.
"""

from __future__ import annotations

import re
from typing import Any

_SCIENCE_SUBJECTS = (
    "physics", "chemistry", "biology", "science", "astronomy", "geology",
    "فيزياء", "كيمياء", "أحياء", "علوم",
)
_MATH_SUBJECTS = ("mathematics", "math", "algebra", "geometry", "رياضيات", "جبر", "هندسة")

_UNIT_CONVERSION_RE = re.compile(
    r"convert|conversion|dimensional\s+analysis|unit\s+conversion|"
    r"تحويل\s*(?:الوحدات|وحدات)|التحليل\s*البعدي",
    re.IGNORECASE,
)

_METRIC_RE = re.compile(
    r"\b(?:mm|cm|km|kg|mg|ml|kl|hz|khz|mhz|ghz|kpa|mpa|kj|kw|pa|mol|mmol)\b|"
    r"\d\s*(?:m|g|l|s|n|j|w|v|a|c)\b|"
    r"\(\s*(?:m|cm|mm|km|kg|g|s)\s*\)|"
    r"\bm/s(?:2|²)?\b|\bm/s\^2\b|"
    r"(?:متر|مترًا|مترا|كجم|كيلوجرام|سم|سنتيمتر|كم|كيلومتر|مل|لتر|نيوتن|جول|واط|باسكال)",
    re.IGNORECASE,
)
_IMPERIAL_RE = re.compile(
    r"\b(?:ft|feet|foot|yd|yard|yards|mile|miles|lb|lbs|pound|pounds|oz|ounce|"
    r"gallon|gallons|fahrenheit)\b|"
    r"\d\s*(?:in|inch|inches)\b|"
    r"(?:قدم|أقدام|بوصة|يارد|ميل|رطل|أوقية|جالون|فهرنهايت)",
    re.IGNORECASE,
)


def normalized_subject(subject: Any) -> str:
    return str(subject or "").strip().casefold()


def unit_policy_for(
    subject: Any,
    *,
    exact_source_instance: bool = False,
    operation: str = "",
    target: str = "",
    givens: list[Any] | None = None,
) -> str:
    """Return ``flexible``, ``constrained`` or ``essential`` for one task."""
    text = " ".join([str(operation or ""), str(target or ""),
                     " ".join(map(str, givens or []))])
    has_unit = bool(_METRIC_RE.search(text) or _IMPERIAL_RE.search(text))
    if _UNIT_CONVERSION_RE.search(text) or (exact_source_instance and has_unit):
        return "essential"
    name = normalized_subject(subject)
    if any(token in name for token in _SCIENCE_SUBJECTS):
        return "constrained"
    if any(token in name for token in _MATH_SUBJECTS):
        return "flexible"
    # Unknown discipline: keep the safer scientific behaviour.
    return "constrained"


def detect_unit_systems(text: Any) -> set[str]:
    """Return which measurement systems appear in the supplied text."""
    value = str(text or "")
    systems: set[str] = set()
    if _METRIC_RE.search(value):
        systems.add("metric")
    if _IMPERIAL_RE.search(value):
        systems.add("imperial")
    return systems


def detect_unit_conflict(text: Any) -> list[dict[str, Any]]:
    """Report a same-passage metric/imperial conflict without guessing intent."""
    systems = detect_unit_systems(text)
    if systems != {"metric", "imperial"}:
        return []
    value = str(text or "")
    return [{
        "code": "source_unit_conflict",
        "systems": sorted(systems),
        "metric_examples": sorted(set(_METRIC_RE.findall(value)))[:4],
        "imperial_examples": sorted(set(_IMPERIAL_RE.findall(value)))[:4],
    }]


def unit_policy_instruction(policy: str) -> str:
    """Prompt-ready policy sentence shared by generator, validators and repair."""
    if policy == "essential":
        return (
            "Unit policy: ESSENTIAL. Preserve the exact source value and unit; do not "
            "convert, normalize or round the unit. If the evidence itself uses two unit "
            "systems for the same quantity, keep the student-visible literal used by the "
            "specific item and never invent a conversion factor."
        )
    if policy == "flexible":
        return (
            "Unit policy: FLEXIBLE (Mathematics). New numerical values may use a different "
            "but dimensionally consistent unit when the mathematical concept and requested "
            "result are unchanged. Never invent a conversion that the question does not need."
        )
    return (
        "Unit policy: CONSTRAINED (science). Prefer the source unit where it carries the "
        "method or dimensional meaning. New numbers are allowed. A unit change is allowed "
        "only when it is dimensionally valid and the conversion is derivable from stated "
        "givens or the source; never invent a conversion factor."
    )
