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
    .replace(/^\\\[|\\\]$/g, "")
    .replace(/^\$\$|\$\$$/g, "")
    .split(/\s*[؛;]\s*/)
    .filter((part) => part.trim())
    .map(normalizeClause)
    .filter(Boolean)
    .join(String.raw`\quad `);
}

export function splitEquationContent(source) {
  const text = String(source ?? "").trim();
  const parts = [];
  const display = /\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|(?<!\$)\$([^$\n]+?)\$(?!\$)/g;
  let start = 0;
  let match;
  while ((match = display.exec(text)) !== null) {
    const prose = text.slice(start, match.index).trim();
    if (prose) parts.push({ type: "text", value: prose });
    parts.push({
      type: "math",
      value: (match[1] ?? match[2] ?? match[3] ?? match[4] ?? "").trim(),
      display: match[0].startsWith("$$") || match[0].startsWith("\\["),
    });
    start = display.lastIndex;
  }
  const tail = text.slice(start).trim();
  if (tail) parts.push({ type: parts.length ? "text" : "math", value: tail });
  return parts;
}

export function renderMixedMath(target, source, language = "en") {
  const text = String(source ?? "");
  const parts = splitEquationContent(text);
  const hasDelimitedMath = /\$\$|\\\[|\\\(|(?<!\$)\$[^$\n]+?\$(?!\$)/.test(text);
  target.replaceChildren();
  if (!hasDelimitedMath) {
    target.textContent = text;
    target.dir = language === "ar" ? "rtl" : "ltr";
    target.lang = language;
    return;
  }
  parts.forEach((part) => {
    const node = document.createElement(part.type === "math" && part.display ? "div" : "span");
    if (part.type === "math") {
      node.className = part.display ? "math-display" : "math-inline";
      renderEquation(node, part.value, Boolean(part.display));
    } else {
      node.className = "math-prose";
      node.textContent = part.value;
      node.dir = language === "ar" ? "rtl" : "ltr";
      node.lang = language;
    }
    target.appendChild(node);
  });
}

export function renderEquationContent(target, source, language = "en") {
  const parts = splitEquationContent(source);
  target.replaceChildren();
  parts.forEach((part) => {
    const node = document.createElement("div");
    node.className = part.type === "math" ? "equation-display" : "equation-instruction";
    if (part.type === "math") renderEquation(node, part.value);
    else {
      node.textContent = part.value;
      node.dir = language === "ar" ? "rtl" : "ltr";
      node.lang = language;
    }
    target.appendChild(node);
  });
}

export function latexToPlainText(source) {
  const original = String(source ?? "").trim();
  let value = original
    .replace(/^\$\$|\$\$$/g, "")
    .replace(/^\$|\$$/g, "")
    .replace(/^\\\[|\\\]$/g, "")
    .replace(/^\\\(|\\\)$/g, "")
    .trim();
  const matrix = value.match(
    /^(.*?)\\begin\{(?:bmatrix|pmatrix|matrix|cases)\}([\s\S]+?)\\end\{(?:bmatrix|pmatrix|matrix|cases)\}$/,
  );
  if (matrix) {
    const rows = matrix[2].split(/\\\\/).map((row) => row.split("&").map((cell) => cell.trim()));
    if (!rows.length || rows.some((row) => !row.length || row.some((cell) => !cell))) return null;
    const prefix = matrix[1].trim();
    value = `${prefix}[${rows.map((row) => row.join(", ")).join("; ")}]`;
  }
  let previous;
  do {
    previous = value;
    value = value.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1/$2)");
  } while (value !== previous);
  value = value
    .replace(/\\sqrt\{([^{}]+)\}/g, "sqrt($1)")
    .replace(/\\(?:mathrm|text)\{([^{}]+)\}/g, "$1")
    .replace(/\^\{([^{}]+)\}/g, "^($1)")
    .replace(/_\{([^{}]+)\}/g, "_($1)")
    .replace(/\\times\b/g, "×")
    .replace(/\\div\b/g, "÷")
    .replace(/\\(?:cdot|,|;|!)/g, " ")
    .replace(/\\(?:le|leq)\b/g, "≤")
    .replace(/\\(?:ge|geq)\b/g, "≥")
    .replace(/\\ne\b/g, "≠")
    .replace(/\\to\b/g, "→");
  if (/\\[A-Za-z]+|[{}]/.test(value)) return null;
  return value.trim() || null;
}

export function renderEquation(target, source, displayMode = true) {
  const original = String(source ?? "").trim();
  const showFallback = (error) => {
    const converted = latexToPlainText(original);
    target.textContent = converted ?? original;
    target.classList?.add("math-render-fallback");
    target.dataset.mathRenderFallback = converted === null ? "original" : "plain-text";
    console.error("Math rendering failed", { expression: original, error });
  };
  target.textContent = original;
  target.dir = "ltr";
  target.lang = "en";
  if (!original) return false;
  if (!window.katex) {
    showFallback(new Error("KaTeX is unavailable"));
    return false;
  }
  try {
    window.katex.render(toMathLatex(original), target, {
      displayMode,
      throwOnError: true,
      strict: "warn",
      trust: false,
    });
    return true;
  } catch (_error) {
    showFallback(_error);
    return false;
  }
}
