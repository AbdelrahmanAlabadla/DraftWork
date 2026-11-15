/* Versioned, renderer-independent exam document model.
 *
 * Mirrors app/exports/document_ir.py and app/exports/content_parser.py.  The
 * browser preview consumes this model instead of reinterpreting raw strings in
 * the DOM builder.
 */
import { latexToPlainText } from "./math-renderer.js";

export const IR_VERSION = "1.0";

const TYPE_ORDER = [
  "mcq", "fill_in_the_blank", "true_false", "definition",
  "why", "equation", "word_problem", "essay",
];
const SECTION_LABELS = {
  en: {
    mcq: "Multiple Choice Questions", fill_in_the_blank: "Fill in the Blank",
    true_false: "True / False", definition: "Define", why: "Why Questions",
    equation: "Solve the equations and show your steps.",
    word_problem: "Solve the word problems and show your steps.", essay: "Essay",
  },
  ar: {
    mcq: "أسئلة الاختيار من متعدد", fill_in_the_blank: "أكمل الفراغ",
    true_false: "صح / غلط", definition: "تعريف", why: "علّل",
    equation: "حل المعادلات وأظهر خطواتك.",
    word_problem: "حل المسائل الكلامية وأظهر خطواتك.", essay: "مقال",
  },
};
const HEADER_LABELS = {
  en: {
    model: "Model {n}", class: "Class", duration: "Duration", date: "Date",
    teacher: "Teacher", student_name: "Student Name",
    answer_key: "Answer Key - Model {n}",
  },
  ar: {
    model: "نسخة {n}", class: "الصف", duration: "المدة", date: "التاريخ",
    teacher: "المعلم", student_name: "اسم الطالب",
    answer_key: "مفتاح الإجابة - نسخة {n}",
  },
};
const ANSWER_LABELS = {
  en: { missing: "No answer supplied", missing_final: "No final answer supplied",
        missing_reference: "No reference answer supplied", key_points: "Key points" },
  ar: { missing: "لم تُرفق إجابة", missing_final: "لم تُرفق إجابة نهائية",
        missing_reference: "لم تُرفق إجابة نموذجية", key_points: "النقاط الرئيسية" },
};
const TF_CHOICES = { en: "True (   ) False (   )", ar: "صح (   ) غلط (   )" };

function normalizeNumerals(value) {
  return String(value ?? "").replace(/[٠-٩۰-۹]/g, (digit) => {
    const code = digit.charCodeAt(0);
    return String(code >= 0x06f0 ? code - 0x06f0 : code - 0x0660);
  });
}

function clean(value) {
  return String(value ?? "").replace(/[ \t]+/g, " ").trim();
}

function format(template, params = {}) {
  return Object.entries(params).reduce(
    (result, [key, value]) => result.replace(`{${key}}`, String(value)),
    template,
  );
}

export function readableMath(value) {
  const plain = latexToPlainText(value);
  if (plain) return plain;
  return clean(
    String(value ?? "")
      .replace(/\\(?:begin|end)\{[^{}]*\}/g, " ")
      .replace(/\\(?:left|right|big|Big|bigg|Bigg)\b/g, " ")
      .replace(/\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}/g, "$1")
      .replace(/\\[A-Za-z]+\s*/g, " ")
      .replace(/[{}&]/g, " "),
  ) || "—";
}

function cellRuns(value) {
  const text = normalizeNumerals(value)
    .replace(/\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}/g, "$1")
    .replace(/\\%/g, "%")
    .replace(/\\&/g, "&");
  return splitInline(text);
}

function splitInline(value) {
  const runs = [];
  const pattern = /\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|(?<!\$)\$([^$\n]+?)\$(?!\$)/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(value)) !== null) {
    if (match.index > cursor) {
      runs.push({ type: "text", text: value.slice(cursor, match.index) });
    }
    const latex = match.slice(1).find((part) => part !== undefined) || "";
    runs.push({
      type: "math", latex: latex.trim(), display: match[1] !== undefined || match[2] !== undefined,
      plain: readableMath(latex),
    });
    cursor = match.index + match[0].length;
  }
  if (cursor < value.length) runs.push({ type: "text", text: value.slice(cursor) });
  if (!runs.length) runs.push({ type: "text", text: value });
  return runs;
}

function tableBlock(headers, rows, align = [], compact = false) {
  return {
    type: "table", caption: "", headers, rows,
    align: align.length ? align : ["start"],
    compact, source: "",
  };
}

function parseMarkdownTable(lines, start) {
  if (start + 1 >= lines.length || !lines[start].includes("|")) return null;
  const separator = lines[start + 1].trim();
  if (!separator.includes("|")) return null;
  const sepCells = splitMarkdownRow(separator);
  if (!sepCells.length || !sepCells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s/g, "")))) {
    return null;
  }
  const header = splitMarkdownRow(lines[start]);
  const align = sepCells.map((cell) => cell.trim().endsWith(":") ? "end" : "start");
  const rows = [];
  let index = start + 2;
  while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
    const cells = splitMarkdownRow(lines[index]);
    while (cells.length < header.length) cells.push("");
    rows.push(cells.slice(0, header.length).map(cellRuns));
    index += 1;
  }
  return [
    tableBlock(header.map(cellRuns), rows, align),
    index,
  ];
}

function splitMarkdownRow(line) {
  let text = line.trim();
  if (text.startsWith("|")) text = text.slice(1);
  if (text.endsWith("|")) text = text.slice(0, -1);
  const cells = [];
  let current = "";
  let escaped = false;
  for (const char of text) {
    if (escaped) { current += char; escaped = false; }
    else if (char === "\\") { current += char; escaped = true; }
    else if (char === "|") { cells.push(current.trim()); current = ""; }
    else current += char;
  }
  cells.push(current.trim());
  return cells;
}

function parseArray(value) {
  const match = String(value).match(/\\begin\{(array|tabular)\}(?:\{([^{}]*)\})?([\s\S]*?)\\end\{\1\}/);
  if (!match) return null;
  const spec = match[2] || "";
  const body = match[3].replace(/\\(?:hline|cline)\b(?:\{[^{}]*\})?/g, " ");
  const rawRows = body.split(/\\\\|\n(?=\s*\\text)/).map((row) => row.trim()).filter(Boolean);
  const cells = rawRows.map((row) => row.replace(/\\$/, "").split("&").map((cell) => cell.trim()));
  if (!cells.length) return null;
  const width = Math.max(...cells.map((row) => row.length));
  const align = [...spec.replace(/[^clr]/g, "")].map((char) => (
    char === "c" ? "center" : char === "r" ? "end" : "start"
  ));
  while (align.length < width) align.push("start");
  return tableBlock(
    cells[0].map(cellRuns),
    cells.slice(1).map((row) => row.map(cellRuns)),
    align.slice(0, width),
  );
}

export function parseCoordinateTable(value) {
  const text = normalizeNumerals(value);
  if (!/^(?:النقاط|نقاط|Points?)\s*:/i.test(text.trim())) return null;
  const pairs = [...text.matchAll(/\(\s*([-+]?\d+(?:\.\d+)?)\s*[,،]\s*([-+]?\d+(?:\.\d+)?)\s*\)/g)];
  if (pairs.length < 2) return null;
  return tableBlock(
    [cellRuns("x"), cellRuns("y")],
    pairs.map((match) => [cellRuns(match[1]), cellRuns(match[2])]),
    ["center", "center"], true,
  );
}

export function parseContent(value) {
  const text = normalizeNumerals(value);
  const blocks = [];
  if (/^(?:النقاط|نقاط|Points?)\s*:/i.test(text.trim())) {
    const coordinate = parseCoordinateTable(text);
    if (coordinate) return [coordinate];
  }
  const lines = text.split(/\r?\n/);
  let buffer = [];
  let index = 0;
  const flush = () => {
    if (!buffer.length) return;
    blocks.push(...parseProse(buffer.join("\n")));
    buffer = [];
  };
  while (index < lines.length) {
    const markdown = parseMarkdownTable(lines, index);
    if (markdown) {
      const [table, next] = markdown;
      flush();
      blocks.push(table);
      index = next;
      continue;
    }
    const arrayMatch = lines[index].match(/\\begin\{(array|tabular)\}(?:\{[^{}]*\})?[\s\S]*?\\end\{\1\}/);
    if (arrayMatch) {
      flush();
      const table = parseArray(lines[index]);
      if (table) blocks.push(table);
      else {
        blocks.push({
          type: "paragraph", runs: [{ type: "text", text: readableMath(lines[index]) }],
        });
      }
      index += 1;
      continue;
    }
    buffer.push(lines[index]);
    index += 1;
  }
  flush();
  return blocks.length ? blocks : [{ type: "paragraph", runs: [{ type: "text", text: "—" }] }];
}

function parseProse(value) {
  const blocks = [];
  const undelimited = value.match(
    /(?<!\\)\\begin\{(array|tabular|aligned|align|cases|matrix|bmatrix|pmatrix)\}[\s\S]*?\\end\{\1\}/,
  );
  if (undelimited) {
    const before = value.slice(0, undelimited.index)
      .replace(/\\(?:\(|\[)\s*$/, "").trim();
    const after = value.slice(undelimited.index + undelimited[0].length)
      .replace(/^\s*\\(?:\)|\])/, "").trim();
    if (before) blocks.push(...parseParagraphs(before));
    if (undelimited[1] === "array" || undelimited[1] === "tabular") {
      const table = parseArray(undelimited[0]);
      if (table) blocks.push(table);
    } else {
      blocks.push({
        type: "math", latex: undelimited[0], display: true,
        plain: readableMath(undelimited[0]),
      });
    }
    if (after) blocks.push(...parseParagraphs(after));
    return blocks;
  }
  return parseParagraphs(value);
}

function parseParagraphs(value) {
  const blocks = [];
  value.split(/\n\s*\n/).forEach((raw) => {
    let paragraph = raw.trim();
    if (!paragraph || paragraph === "\\[" || paragraph === "\\]") return;
    const runs = splitInline(paragraph);
    if (runs.some((run) => run.type === "math" && run.display)) {
      let current = [];
      const flush = () => {
        if (current.length) {
          blocks.push({ type: "paragraph", runs: current });
          current = [];
        }
      };
      runs.forEach((run) => {
        if (run.type === "math" && run.display) {
          flush();
          blocks.push(run);
        } else {
          current.push(run);
        }
      });
      flush();
    } else {
      blocks.push({ type: "paragraph", runs });
    }
  });
  return blocks;
}

function questionValue(qtype, item) {
  if (qtype === "true_false") return item.statement || "";
  if (qtype === "definition") return item.term || "";
  if (qtype === "equation") return item.equation || "";
  if (qtype === "fill_in_the_blank") {
    return String(item.question || "").replace(/_{3,}|-{3,}/g, (blank) => blank + blank[0].repeat(4));
  }
  return item.question || "";
}

function splitLegacySolution(value) {
  const match = String(value).match(/(?:^|\n)\s*(?:الحل|Solution|solution)\s*:\s*/);
  if (!match) return [value, ""];
  const before = value.slice(0, match.index).trim();
  const after = value.slice(match.index + match[0].length).trim();
  if (before.length < 20 || !after) return [value, ""];
  return [before, after];
}

function answerBlocks(qtype, item, language, legacySolution = "") {
  const labels = ANSWER_LABELS[language] || ANSWER_LABELS.en;
  if (qtype === "mcq") {
    return [{ type: "paragraph", runs: [{ type: "text", text: clean(item.correct_answer || labels.missing) }] }];
  }
  if (qtype === "true_false") {
    const normalized = String(item.answer ?? "").toLowerCase();
    const text = normalized === "true" ? "صح" : normalized === "false" ? "غلط" : labels.missing;
    return [{ type: "paragraph", runs: [{ type: "text", text }] }];
  }
  if (qtype === "fill_in_the_blank") {
    const answer = (item.answers || []).map(clean).filter(Boolean).join(", ") || labels.missing;
    return [{ type: "paragraph", runs: [{ type: "text", text: answer }] }];
  }
  if (qtype === "equation" || qtype === "word_problem") {
    const blocks = [];
    if (legacySolution) blocks.push(...parseContent(legacySolution));
    (item.solution_steps || []).forEach((step) => {
      if (clean(step)) blocks.push(...parseContent(step));
    });
    if (clean(item.final_answer)) blocks.push(...parseContent(item.final_answer));
    return blocks.length ? blocks : [{ type: "paragraph", runs: [{ type: "text", text: labels.missing_final }] }];
  }
  const blocks = [{ type: "paragraph", runs: [{ type: "text", text: clean(item.reference_answer || labels.missing_reference) }] }];
  const points = (item.key_points || []).map(clean).filter(Boolean);
  if (points.length) {
    blocks.push({ type: "paragraph", runs: [{ type: "text", text: `${labels.key_points}: ${points.join("; ")}` }] });
  }
  return blocks;
}

function optionBlocks(value) {
  const text = normalizeNumerals(value);
  const coordinate = parseCoordinateTable(text);
  if (coordinate) return [coordinate];
  return parseContent(text);
}

export function buildDocumentIR(exam, metadata = {}, kind = "student") {
  const language = clean(exam.document_language || metadata.document_language || "en");
  const direction = language === "ar" ? "rtl" : "ltr";
  const labels = HEADER_LABELS[language] || HEADER_LABELS.en;
  const sectionLabels = SECTION_LABELS[language] || SECTION_LABELS.en;
  const questions = exam.questions || {};
  const modelNumber = Number(exam.model_number || 1);
  const title = clean(metadata.exam_title || exam.title || (language === "ar" ? "امتحان" : "Examination"));
  const document = {
    version: IR_VERSION, kind, language, direction,
    metadata: {
      title, model_label: format(labels.model, { n: modelNumber }),
      answer_key_title: format(labels.answer_key, { n: modelNumber }),
      class_name: metadata.class_name || "", duration: metadata.duration || "",
      teacher_name: metadata.teacher_name || "", exam_date: metadata.exam_date || "",
      student_name_label: labels.student_name, class_label: labels.class,
      duration_label: labels.duration, teacher_label: labels.teacher, date_label: labels.date,
      true_false_choices: TF_CHOICES[language] || TF_CHOICES.en,
      page_label: language === "ar" ? "صفحة" : "Page",
      of_label: language === "ar" ? "من" : "of",
    },
    sections: [], diagnostics: [],
  };
  TYPE_ORDER.forEach((qtype) => {
    let section = questions[qtype];
    if (!section) return;
    let items = [];
    let wordBank = [];
    if (qtype === "fill_in_the_blank") {
      items = section.items || [];
      wordBank = section.word_bank || [];
    } else if (Array.isArray(section)) {
      items = section;
    }
    if (!items.length) return;
    const out = { qtype, heading: sectionLabels[qtype], items: [] };
    if (wordBank.length) out.word_bank = wordBank;
    items.forEach((item, index) => {
      let raw = questionValue(qtype, item);
      let legacySolution = "";
      if (qtype === "equation") [raw, legacySolution] = splitLegacySolution(raw);
      const blocks = parseContent(raw);
      const options = qtype === "mcq"
        ? ["A", "B", "C", "D"].filter((letter) => item.options?.[letter] !== undefined)
          .map((letter) => ({ label: letter, blocks: optionBlocks(item.options[letter]) }))
        : [];
      const question = {
        number: index + 1, question_id: item.question_id || null, qtype, blocks, options,
        answer_blocks: answerBlocks(qtype, item, language, legacySolution),
        response_lines: kind === "student" && ["why", "equation", "word_problem", "essay"].includes(qtype)
          ? ({ why: 3, equation: 5, word_problem: 5, essay: 22 }[qtype] || 0) : 0,
        diagnostics: [],
      };
      if (qtype === "true_false") question.true_false_choices = TF_CHOICES[language] || TF_CHOICES.en;
      out.items.push(question);
    });
    document.sections.push(out);
  });
  return document;
}
