"""Convert common generated equation notation into rendered math images.

The generation schema intentionally remains plain text so the web application
does not expose formatting commands. Exporters call this module to normalize
that text into Matplotlib's supported math syntax and rasterize only the final
typeset result.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class RenderedMath:
    png: bytes
    width_px: int
    height_px: int


_SUPERSCRIPTS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
})
_SUBSCRIPTS = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})
_GREEK = {
    "θ": r"\theta", "Θ": r"\Theta", "α": r"\alpha", "β": r"\beta",
    "γ": r"\gamma", "δ": r"\delta", "Δ": r"\Delta", "λ": r"\lambda",
    "μ": r"\mu", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma",
    "φ": r"\phi", "ω": r"\omega", "Ω": r"\Omega",
}
_UNIT_FRACTIONS = {
    "m/s²": r"\;\mathrm{m\,s^{-2}}",
    "m/s^2": r"\;\mathrm{m\,s^{-2}}",
    "km/h": r"\;\mathrm{km\,h^{-1}}",
    "m/s": r"\;\mathrm{m\,s^{-1}}",
}


def _replace_script_runs(text: str, chars: str, translation: dict[int, str], marker: str) -> str:
    pattern = re.compile(f"([{re.escape(chars)}]+)")
    return pattern.sub(lambda match: f"{marker}{{{match.group(1).translate(translation)}}}", text)


def _normalize_clause(clause: str) -> str:
    value = clause.strip()
    if not value:
        return ""

    # Protect conventional units before recognizing mathematical fractions.
    protected: dict[str, str] = {}
    for index, (unit, replacement) in enumerate(_UNIT_FRACTIONS.items()):
        token = f"UNITTOKEN{index}"
        if unit in value:
            value = value.replace(unit, token)
            protected[token] = replacement

    # Unicode scripts and Greek letters commonly returned by the model.
    value = _replace_script_runs(value, "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", _SUPERSCRIPTS, "^")
    value = _replace_script_runs(value, "₀₁₂₃₄₅₆₇₈₉", _SUBSCRIPTS, "_")
    value = re.sub(r"θ\s*([A-Za-z0-9])\b", r"\\theta_{\1}", value)
    for symbol, command in _GREEK.items():
        value = value.replace(symbol, command)

    # Common plain-text index notation (n1, n2, x0) becomes a subscript.
    value = re.sub(r"\b([A-Za-z])([0-9]+)\b", r"\1_{\2}", value)

    # Calculus and operator notation.
    value = re.sub(r"∫\s*_?\s*([^\s^]+)\s*\^\s*([^\s]+)", r"\\int_{\1}^{\2}", value)
    value = value.replace("∫", r"\int ")
    value = value.replace("√", r"\sqrt ")
    value = value.replace("∞", r"\infty")
    value = value.replace("→", r"\to")
    value = value.replace("×", r"\times")
    value = value.replace("÷", r"\div")
    value = value.replace("≤", r"\le")
    value = value.replace("≥", r"\ge")
    value = value.replace("≠", r"\ne")
    value = re.sub(
        r"(?<!\\)\blim\s+([A-Za-z])\s*(?:\\to|→|->)\s*([^\s]+)",
        r"\\lim_{\1\\to \2}",
        value,
    )
    for function in ("sin", "cos", "tan", "log", "ln", "lim"):
        value = re.sub(rf"(?<!\\)\b{function}\b", rf"\\{function}", value)

    # Simple generated fractions such as n2/n1, c/v, dy/dx, and (x+1)/h.
    atom = r"(?:\\[A-Za-z]+(?:_\{[^{}]+\})?|[A-Za-z]{1,3}(?:_\{[^{}]+\})?|\([^()]+\)|-?\d+(?:\.\d+)?)"
    grouped_fraction_re = re.compile(
        rf"(\((?:[^()]|\([^()]*\))+\))\s*/\s*({atom})"
    )
    value = grouped_fraction_re.sub(
        lambda match: rf"\frac{{{match.group(1)[1:-1]}}}{{{match.group(2)}}}",
        value,
    )
    fraction_re = re.compile(rf"(?<!\^)({atom})\s*/\s*({atom})")
    previous = None
    while previous != value:
        previous = value
        value = fraction_re.sub(r"\\frac{\1}{\2}", value)

    value = re.sub(r"(?<![A-Za-z{])d([A-Za-z])\b", r"\\,\\mathrm{d}\1", value)

    # Make ordinary words/units upright while leaving variables italic.
    value = re.sub(
        r"(?<=\d)\s+(kg|cm|mm|km|Hz|Pa|J|N|W|V|A|s|m|h)\b",
        r"\;\\mathrm{\1}",
        value,
    )
    for token, replacement in protected.items():
        value = value.replace(token, replacement)

    value = re.sub(r"\s*=\s*", " = ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def to_mathtext(source: object) -> str:
    """Return normalized math syntax without display delimiters."""
    text = str(source or "").strip()
    if not text:
        return ""
    clauses = [part for part in re.split(r"\s*[؛;]\s*", text) if part.strip()]
    return r"\quad ".join(filter(None, (_normalize_clause(part) for part in clauses)))


def render_math(source: object, *, font_size: float = 13.0, dpi: int = 220) -> RenderedMath | None:
    """Render equation text to a tightly cropped transparent PNG."""
    mathtext = to_mathtext(source)
    if not mathtext:
        return None
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image

        buffer = io.BytesIO()
        math_to_image(
            f"${mathtext}$",
            buffer,
            prop=FontProperties(family="DejaVu Sans", size=font_size),
            dpi=dpi,
            format="png",
            color="#1f2937",
        )
        png = buffer.getvalue()
        with Image.open(io.BytesIO(png)) as image:
            width, height = image.size
        return RenderedMath(png=png, width_px=width, height_px=height)
    except Exception:
        return None
