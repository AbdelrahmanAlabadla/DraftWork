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
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    elif text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()
    if not text:
        return ""
    clauses = [part for part in re.split(r"\s*[؛;]\s*", text) if part.strip()]
    return r"\quad ".join(filter(None, (_normalize_clause(part) for part in clauses)))


def render_math(source: object, *, font_size: float = 13.0, dpi: int = 220) -> RenderedMath | None:
    """Render equation text to a tightly cropped transparent PNG."""
    mathtext = to_mathtext(source)
    if not mathtext:
        return None
    matrix = re.fullmatch(
        r"(.*?)\\begin\{(?:bmatrix|pmatrix|matrix)\}([\s\S]+?)"
        r"\\end\{(?:bmatrix|pmatrix|matrix)\}", mathtext.strip()
    )
    if matrix:
        rows = [row.strip() for row in re.split(r"\\\\", matrix.group(2)) if row.strip()]
        cells = [[cell.strip() for cell in row.split("&")] for row in rows]
        if cells and len({len(row) for row in cells}) == 1:
            rendered_cells = [
                [render_math(cell, font_size=font_size, dpi=dpi) for cell in row]
                for row in cells
            ]
            if all(all(row) for row in rendered_cells):
                try:
                    images = [[Image.open(io.BytesIO(cell.png)).convert("RGBA") for cell in row]
                              for row in rendered_cells]
                    col_widths = [max(row[col].width for row in images)
                                  for col in range(len(images[0]))]
                    row_heights = [max(cell.height for cell in row) for row in images]
                    gap = max(8, int(dpi / 24))
                    bracket = max(8, int(dpi / 28))
                    prefix = render_math(matrix.group(1), font_size=font_size, dpi=dpi) \
                        if matrix.group(1).strip() else None
                    prefix_image = Image.open(io.BytesIO(prefix.png)).convert("RGBA") if prefix else None
                    grid_w = sum(col_widths) + gap * (len(col_widths) - 1)
                    grid_h = sum(row_heights) + gap * (len(row_heights) - 1)
                    prefix_w = (prefix_image.width + gap) if prefix_image else 0
                    canvas = Image.new("RGBA", (prefix_w + grid_w + 2 * bracket, grid_h),
                                       (255, 255, 255, 0))
                    if prefix_image:
                        canvas.alpha_composite(prefix_image, (0, (grid_h - prefix_image.height) // 2))
                    x0 = prefix_w + bracket
                    y = 0
                    for row_index, row in enumerate(images):
                        x = x0
                        for col_index, image in enumerate(row):
                            canvas.alpha_composite(
                                image,
                                (x + (col_widths[col_index] - image.width) // 2,
                                 y + (row_heights[row_index] - image.height) // 2),
                            )
                            x += col_widths[col_index] + gap
                        y += row_heights[row_index] + gap
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(canvas)
                    left, right = prefix_w + 1, prefix_w + bracket + grid_w + bracket - 2
                    line = max(2, int(dpi / 110))
                    for bx, direction in ((left, 1), (right, -1)):
                        draw.line((bx, 0, bx, grid_h - 1), fill="#1f2937", width=line)
                        draw.line((bx, 0, bx + direction * bracket, 0), fill="#1f2937", width=line)
                        draw.line((bx, grid_h - 1, bx + direction * bracket, grid_h - 1),
                                  fill="#1f2937", width=line)
                    output = io.BytesIO()
                    canvas.save(output, format="PNG")
                    return RenderedMath(output.getvalue(), canvas.width, canvas.height)
                except Exception:
                    return None
    cases = re.fullmatch(
        r"\\begin\{cases\}([\s\S]+?)\\end\{cases\}", mathtext.strip()
    )
    if cases:
        rows = [row.strip() for row in re.split(r"\\\\", cases.group(1)) if row.strip()]
        rendered_rows = [render_math(row, font_size=font_size, dpi=dpi) for row in rows]
        if rows and all(rendered_rows):
            try:
                from matplotlib.font_manager import FontProperties, findfont
                from PIL import ImageDraw, ImageFont

                images = [Image.open(io.BytesIO(row.png)).convert("RGBA") for row in rendered_rows]
                gap = max(4, int(dpi / 55))
                brace_width = max(20, int(font_size * dpi / 72 * 0.8))
                width = brace_width + max(image.width for image in images)
                height = sum(image.height for image in images) + gap * (len(images) - 1)
                canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
                y = 0
                for image in images:
                    canvas.alpha_composite(image, (brace_width, y))
                    y += image.height + gap
                draw = ImageDraw.Draw(canvas)
                font_path = findfont(FontProperties(family="DejaVu Sans"))
                brace_font = ImageFont.truetype(font_path, max(18, int(height * 0.92)))
                draw.text((0, height / 2), "{", font=brace_font, fill="#1f2937",
                          anchor="lm")
                output = io.BytesIO()
                canvas.save(output, format="PNG")
                return RenderedMath(png=output.getvalue(), width_px=width, height_px=height)
            except Exception:
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
