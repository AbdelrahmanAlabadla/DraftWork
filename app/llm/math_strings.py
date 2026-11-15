"""Conservative JSON/LaTeX handling shared by generation and repair agents."""
from __future__ import annotations

import re
from typing import Any

MATH_JSON_CONTRACT = r"""
Mathematical-output contract:
- Keep ordinary prose as prose; use LaTeX only for mathematical expressions.
- In raw JSON, JSON-escape every LaTeX backslash. For example emit
  "\\frac{1}{2}", "\\infty", and "\\leq" in the JSON source.
- Preserve the field's mathematical delimiters. Never ask the student to inspect
  a graph or exhibit that is not included in the generated question.
""".strip()


class MalformedMathStringError(ValueError):
    pass


_INVALID_JSON_LATEX = re.compile(
    r'(?<!\\)\\(infty|sqrt|leq|geq|neq|cdot|left|right|mathrm|mathbf|overline|vec|sum|int|end)(?=[^A-Za-z]|$)'
)


def repair_raw_json_latex_escapes(raw: str) -> str:
    """Double only unmistakable LaTeX slashes that JSON cannot legally parse."""
    return _INVALID_JSON_LATEX.sub(r"\\\\\1", str(raw))


def sanitize_math_string(value: str) -> str:
    """Recover unmistakable parsed escape damage; reject other C0 controls."""
    text = str(value)
    text = text.replace("\f" + "rac", r"\frac")
    text = text.replace("\b" + "egin", r"\begin")
    text = text.replace("\t" + "ext{", r"\text{")
    text = text.replace("\t" + "imes", r"\times")
    text = re.sub(r"(?<!\\)\binfty\b", r"\\infty", text)
    unsafe = [c for c in text if ord(c) < 32 and c not in "\n\r\t"]
    if unsafe:
        points = ", ".join(f"U+{ord(c):04X}" for c in sorted(set(unsafe)))
        raise MalformedMathStringError(f"unsafe control character(s): {points}")
    return text


def sanitize_math_strings(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_math_string(value)
    if isinstance(value, list):
        return [sanitize_math_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_math_strings(item) for item in value)
    if isinstance(value, dict):
        return {key: sanitize_math_strings(item) for key, item in value.items()}
    return value
