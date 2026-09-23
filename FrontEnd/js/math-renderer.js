const SUPERSCRIPTS = new Map([
  ["⁰", "0"], ["¹", "1"], ["²", "2"], ["³", "3"], ["⁴", "4"],
  ["⁵", "5"], ["⁶", "6"], ["⁷", "7"], ["⁸", "8"], ["⁹", "9"],
  ["⁺", "+"], ["⁻", "-"],
]);
const SUBSCRIPTS = new Map([
  ["₀", "0"], ["₁", "1"], ["₂", "2"], ["₃", "3"], ["₄", "4"],
  ["₅", "5"], ["₆", "6"], ["₇", "7"], ["₈", "8"], ["₉", "9"],
]);
const GREEK = new Map([
  ["θ", "\\theta"], ["Θ", "\\Theta"], ["α", "\\alpha"],
  ["β", "\\beta"], ["γ", "\\gamma"], ["δ", "\\delta"],
  ["Δ", "\\Delta"], ["λ", "\\lambda"], ["μ", "\\mu"],
  ["π", "\\pi"], ["ρ", "\\rho"], ["σ", "\\sigma"],
  ["φ", "\\phi"], ["ω", "\\omega"], ["Ω", "\\Omega"],
]);
const UNIT_FRACTIONS = new Map([
  ["m/s²", String.raw`\;\mathrm{m\,s^{-2}}`],
  ["m/s^2", String.raw`\;\mathrm{m\,s^{-2}}`],
  ["km/h", String.raw`\;\mathrm{km\,h^{-1}}`],
  ["m/s", String.raw`\;\mathrm{m\,s^{-1}}`],
]);

function replaceScriptRuns(value, pattern, mapping, marker) {
  return value.replace(pattern, (run) => {
    const translated = [...run].map((character) => mapping.get(character)).join("");
    return `${marker}{${translated}}`;
  });
}

function normalizeClause(source) {
  let value = source.trim();
  if (!value) return "";

  const protectedUnits = new Map();
  [...UNIT_FRACTIONS.entries()].forEach(([unit, replacement], index) => {
    const token = `UNITTOKEN${index}`;
    if (value.includes(unit)) {
      value = value.split(unit).join(token);
      protectedUnits.set(token, replacement);
    }
  });

  value = replaceScriptRuns(value, /[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+/g, SUPERSCRIPTS, "^");
  value = replaceScriptRuns(value, /[₀₁₂₃₄₅₆₇₈₉]+/g, SUBSCRIPTS, "_");
  value = value.replace(/θ\s*([A-Za-z0-9])\b/g, String.raw`\theta_{$1}`);
  GREEK.forEach((command, symbol) => {
    value = value.split(symbol).join(command);
  });
  value = value.replace(/\b([A-Za-z])([0-9]+)\b/g, "$1_{$2}");

  value = value.replace(/∫\s*_?\s*([^\s^]+)\s*\^\s*([^\s]+)/g, String.raw`\int_{$1}^{$2}`);
  value = value.replaceAll("∫", String.raw`\int `)
    .replaceAll("√", String.raw`\sqrt `)
    .replaceAll("∞", String.raw`\infty`)
    .replaceAll("→", String.raw`\to`)
    .replaceAll("×", String.raw`\times`)
    .replaceAll("÷", String.raw`\div`)
    .replaceAll("≤", String.raw`\le`)
    .replaceAll("≥", String.raw`\ge`)
    .replaceAll("≠", String.raw`\ne`);
  value = value.replace(/\blim\s+([A-Za-z])\s*(?:\\to|->)\s*([^\s]+)/g, String.raw`\lim_{$1\to $2}`);
  ["sin", "cos", "tan", "log", "ln", "lim"].forEach((operator) => {
    value = value.replace(new RegExp(`(?<!\\\\)\\b${operator}\\b`, "g"), `\\${operator}`);
  });

  const atom = String.raw`(?:\\[A-Za-z]+(?:_\{[^{}]+\})?|[A-Za-z]{1,3}(?:_\{[^{}]+\})?|\([^()]+\)|-?\d+(?:\.\d+)?)`;
  const groupedFraction = new RegExp(`(\\((?:[^()]|\\([^()]*\\))+\\))\\s*\\/\\s*(${atom})`, "g");
  value = value.replace(groupedFraction, (_match, numerator, denominator) => (
    `\\frac{${numerator.slice(1, -1)}}{${denominator}}`
  ));
  const simpleFraction = new RegExp(`(?<!\\^)(${atom})\\s*\\/\\s*(${atom})`, "g");
  let previous;
  do {
    previous = value;
    value = value.replace(simpleFraction, String.raw`\frac{$1}{$2}`);
  } while (value !== previous);

  value = value.replace(/(?<![A-Za-z{])d([A-Za-z])\b/g, String.raw`\,\mathrm{d}$1`);
  value = value.replace(
    /(?<=\d)\s+(kg|cm|mm|km|Hz|Pa|J|N|W|V|A|s|m|h)\b/g,
    String.raw`\;\mathrm{$1}`,
  );
  protectedUnits.forEach((replacement, token) => {
    value = value.split(token).join(replacement);
  });
  return value.replace(/\s*=\s*/g, " = ").replace(/\s+/g, " ").trim();
}

export function toMathLatex(source) {
  return String(source ?? "")
    .trim()
    .split(/\s*[؛;]\s*/)
    .filter((part) => part.trim())
    .map(normalizeClause)
    .filter(Boolean)
    .join(String.raw`\quad `);
}

export function renderEquation(target, source) {
  const original = String(source ?? "").trim();
  target.textContent = original;
  target.dir = "ltr";
  target.lang = "en";
  if (!original || !window.katex) return false;
  try {
    window.katex.render(toMathLatex(original), target, {
      displayMode: true,
      throwOnError: true,
      strict: "warn",
      trust: false,
    });
    return true;
  } catch (_error) {
    target.textContent = original;
    return false;
  }
}
