from __future__ import annotations

import io

from PIL import Image

from app.exports.math_renderer import render_math, to_mathtext


def test_normalizes_generated_optics_equations():
    rendered = to_mathtext("sin θc = n2/n1؛ n1 = 1.50؛ n2 = 1.00")
    assert r"\sin \theta_{c}" in rendered
    assert r"\frac{n_{2}}{n_{1}}" in rendered
    assert r"\quad n_{1} = 1.50" in rendered
    assert "؛" not in rendered


def test_normalizes_powers_multiplication_and_units():
    rendered = to_mathtext("n = c/v؛ c = 3.00 × 10⁸ m/s؛ n = 1.5")
    assert r"\frac{c}{v}" in rendered
    assert r"\times 10^{8}" in rendered
    assert r"\mathrm{m\,s^{-1}}" in rendered


def test_normalizes_calculus_notation():
    integral = to_mathtext("∫_0^1 (3x^2 + 2x - 1) dx")
    derivative = to_mathtext("dy/dx = lim h→0 ((x+h)^2 - x^2)/h = 2x")
    assert r"\int_{0}^{1}" in integral
    assert r"\,\mathrm{d}x" in integral
    assert r"\frac{dy}{dx}" in derivative
    assert r"\lim_{h\to 0}" in derivative
    assert r"\frac{(x+h)^2 - x^2}{h}" in derivative


def test_renders_math_to_nonempty_png_without_source_text():
    result = render_math("n = c/v؛ c = 3.00 × 10⁸ m/s")
    assert result is not None
    assert result.png.startswith(b"\x89PNG")
    assert result.width_px > result.height_px > 0
    with Image.open(io.BytesIO(result.png)) as image:
        assert image.size == (result.width_px, result.height_px)
