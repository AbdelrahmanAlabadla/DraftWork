import { state } from "./state.js";
import { t } from "./i18n.js";
import { renderEquation } from "./math-renderer.js";
import { buildDocumentIR } from "./document-ir.js";
import { getJSON } from "./api.js";

const RESPONSE_LINE = "_".repeat(92);
const RESPONSE_LINE_COUNTS = { why: 3, equation: 5, word_problem: 5, essay: 22 };
const SECTION_ORDER = [
  "mcq", "fill_in_the_blank", "true_false", "definition",
  "why", "equation", "word_problem", "essay",
];
const EXAM_LABELS = {
  en: {
    examination: "Examination", model: "Model {n}", className: "Class",
    duration: "Duration", date: "Date", teacher: "Teacher",
    studentName: "Student Name", answerKey: "Answer Key - Model {n}",
    wordBank: "WORD BANK", trueFalse: "True (   ) False (   )",
    missingQuestion: "(missing question text)", missingTerm: "(missing term)",
    noAnswer: "No answer supplied", noFinalAnswer: "No final answer supplied",
    noReferenceAnswer: "No reference answer supplied", finalAnswer: "Final answer",
    keyPoints: "Key points",
  },
  ar: {
    examination: "امتحان", model: "نسخة {n}", className: "الصف",
    duration: "المدة", date: "التاريخ", teacher: "المعلم",
    studentName: "اسم الطالب", answerKey: "مفتاح الإجابة - نسخة {n}",
    wordBank: "بنك الكلمات", trueFalse: "صح (   ) غلط (   )",
    missingQuestion: "(نص السؤال غير متاح)", missingTerm: "(المصطلح غير متاح)",
    noAnswer: "لم تُرفق إجابة", noFinalAnswer: "لم تُرفق إجابة نهائية",
    noReferenceAnswer: "لم تُرفق إجابة نموذجية", finalAnswer: "الإجابة النهائية",
    keyPoints: "النقاط الرئيسية",
  },
};

function format(template, params = {}) {
  return Object.entries(params).reduce(
    (result, [key, value]) => result.replace(`{${key}}`, String(value)),
    template,
  );
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function clean(value, fallback = "") {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function isArabic(value) {
  return /[\u0600-\u06ff]/.test(String(value ?? ""));
}

function expandFillBlankText(value) {
  return String(value ?? "").replace(/_{3,}|-{3,}/g, (blank) => (
    blank + blank[0].repeat(4)
  ));
}

function renderRuns(target, runs, language) {
  (runs || []).forEach((run) => {
    if (run.type === "text") {
      const text = String(run.text ?? "");
      if (!text) return;
      const span = element("span", "run-text", text);
      span.dir = isArabic(text) ? "rtl" : "ltr";
      span.lang = isArabic(text) ? "ar" : "en";
      target.appendChild(span);
    } else if (run.type === "math") {
      const math = element("span", "math-inline");
      math.dir = "ltr";
      math.lang = "en";
      target.appendChild(math);
      renderEquation(math, run.latex || run.plain || "", false);
    }
  });
}

function renderTable(tableBlock, language) {
  const table = element("table", "exam-table");
  table.dir = language === "ar" ? "rtl" : "ltr";
  if (tableBlock.caption) {
    const caption = element("caption", "", tableBlock.caption);
    table.appendChild(caption);
  }
  if ((tableBlock.headers || []).length) {
    const thead = element("thead");
    const tr = element("tr");
    tableBlock.headers.forEach((cell) => {
      const th = element("th");
      renderRuns(th, cell, language);
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);
  }
  const tbody = element("tbody");
  (tableBlock.rows || []).forEach((row) => {
    const tr = element("tr");
    row.forEach((cell, index) => {
      const td = element("td");
      const align = (tableBlock.align || [])[index] || "start";
      td.className = align === "center" ? "num" : "";
      renderRuns(td, cell, language);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

function renderMathBlock(block, language) {
  const wrap = element("div", "math-display");
  wrap.dir = "ltr";
  renderEquation(wrap, block.latex || block.plain || "", true);
  return wrap;
}

function renderBlocks(container, blocks, language) {
  (blocks || []).forEach((block) => {
    if (block.type === "table") {
      container.appendChild(renderTable(block, language));
    } else if (block.type === "math" && block.display) {
      container.appendChild(renderMathBlock(block, language));
    } else if (block.type === "paragraph") {
      const p = element("div", "preview-paragraph");
      renderRuns(p, block.runs || [], language);
      container.appendChild(p);
    }
  });
}

function responseLines(count) {
  const lines = element("div", "preview-response-lines");
  for (let index = 0; index < count; index += 1) {
    lines.appendChild(element("div", "preview-response-line", RESPONSE_LINE));
  }
  return lines;
}

function renderOptions(container, options, language) {
  if (!options?.length) return;
  const optionsEl = element("div", "preview-options");
  options.forEach((option) => {
    const item = element("div", "preview-option");
    item.dir = language === "ar" ? "rtl" : "ltr";
    const label = element("span", "preview-option-marker", `${option.label}.`);
    label.dir = "ltr";
    item.appendChild(label);
    const content = element("div", "preview-option-content");
    const table = (option.blocks || []).find((block) => block.type === "table");
    if (table) {
      content.appendChild(renderTable(table, language));
    } else {
      const inline = element("span", "preview-option-inline");
      (option.blocks || []).forEach((block) => {
        if (block.type === "paragraph") renderRuns(inline, block.runs, language);
        if (block.type === "math") {
          const math = element("span", "math-inline");
          math.dir = "ltr";
          inline.appendChild(math);
          renderEquation(math, block.latex || block.plain || "", false);
        }
      });
      content.appendChild(inline);
    }
    item.appendChild(content);
    optionsEl.appendChild(item);
  });
  container.appendChild(optionsEl);
}

function appendTfChoices(stem, choices, language) {
  const group = element("span", "preview-tf-choices", choices);
  group.dir = language === "ar" ? "rtl" : "ltr";
  group.lang = language;
  // The exact group is one unbreakable inline unit; if it does not fit it
  // moves as a whole to the next line.
  group.style.whiteSpace = "nowrap";
  stem.appendChild(group);
}

function renderQuestion(paper, question, language) {
  const section = element("section", `preview-question preview-${question.qtype}`);
  section.dataset.number = String(question.number);
  const blocks = question.blocks || [];
  const table = blocks.find((block) => block.type === "table");
  const textBlocks = blocks.filter((block) => block.type !== "table");

  const buildQuestionPanel = (panel) => {
    const stem = element("div", "preview-question-stem");
    const number = element("strong", "preview-question-number", `${question.number}.`);
    number.dir = "ltr";
    stem.appendChild(number);
    const content = element("div", "preview-question-content");
    renderBlocks(content, textBlocks, language);
    stem.appendChild(content);
    panel.appendChild(stem);
    if (question.qtype === "true_false") {
      const paragraphs = content.querySelectorAll(".preview-paragraph");
      const target = paragraphs.length ? paragraphs[paragraphs.length - 1] : content;
      appendTfChoices(target, question.true_false_choices || "", language);
    }
    if (question.qtype === "mcq") {
      renderOptions(panel, question.options, language);
    }
    if (["why", "equation", "word_problem", "essay"].includes(question.qtype)) {
      const count = question.response_lines || RESPONSE_LINE_COUNTS[question.qtype] || 0;
      panel.appendChild(responseLines(count));
    }
  };

  if (table) {
    const grid = element("div", "preview-with-table");
    const panel = element("div", "preview-question-panel");
    const tablePanel = element("div", "preview-table-panel");
    tablePanel.appendChild(renderTable(table, language));
    buildQuestionPanel(panel);
    // DOM order is question first, table second. CSS grid places the table on
    // the right in English and on the left in Arabic.
    grid.appendChild(panel);
    grid.appendChild(tablePanel);
    section.appendChild(grid);
  } else {
    buildQuestionPanel(section);
  }
  paper.appendChild(section);
}

function addTitleBlock(paper, document) {
  const meta = document.metadata;
  const header = element("header", "preview-title-block");
  const title = element("h1", "preview-exam-title", meta.title);
  title.dir = document.direction;
  header.appendChild(title);
  header.appendChild(element("div", "preview-model", meta.model_label));
  paper.appendChild(header);

  const grid = element("div", "preview-meta-row");
  [
    [meta.duration_label, meta.duration],
    [meta.teacher_label, meta.teacher_name],
    [meta.date_label, meta.exam_date],
  ].forEach(([label, value]) => {
    const field = element("div", "preview-meta-field");
    field.appendChild(element("strong", "", `${label}:`));
    field.appendChild(element("span", value ? "preview-meta-value" : "preview-meta-blank", value || ""));
    grid.appendChild(field);
  });
  paper.appendChild(grid);

  const student = element("div", "preview-student-row");
  const name = element("div", "preview-student-field preview-student-name");
  name.appendChild(element("strong", "", `${meta.student_name_label}:`));
  name.appendChild(element("span", "preview-meta-blank"));
  const klass = element("div", "preview-student-field preview-student-class");
  klass.appendChild(element("strong", "", `${meta.class_label}:`));
  klass.appendChild(element(
    "span",
    meta.class_name ? "preview-meta-value" : "preview-meta-blank",
    meta.class_name || "",
  ));
  student.append(name, klass);
  paper.appendChild(student);
}

function renderStudent(document) {
  const paper = element("article", "exam-paper student-paper");
  paper.dir = document.direction;
  paper.lang = document.language;
  addTitleBlock(paper, document);
  document.sections.forEach((section) => {
    const heading = element("h2", "preview-section-heading", section.heading);
    paper.appendChild(heading);
    if (section.word_bank?.length) {
      const bank = element("div", "preview-word-bank");
      bank.appendChild(element("strong", "", document.language === "ar" ? "بنك الكلمات" : "WORD BANK"));
      bank.appendChild(element("div", "", section.word_bank.join("   |   ")));
      paper.appendChild(bank);
    }
    section.items.forEach((question) => renderQuestion(paper, question, document.language));
  });
  paper.appendChild(element("div", "preview-page-number", `${document.metadata.page_label} 1`));
  return paper;
}

function renderAnswers(document) {
  const paper = element("article", "exam-paper answer-key-paper");
  paper.dir = document.direction;
  paper.lang = document.language;
  paper.appendChild(element("h1", "preview-key-title", document.metadata.answer_key_title));
  document.sections.forEach((section) => {
    paper.appendChild(element("h2", "preview-section-heading", section.heading));
    section.items.forEach((question) => {
      const row = element("div", "preview-key-answer");
      const number = element("strong", "preview-question-number", `${question.number}.`);
      number.dir = "ltr";
      row.appendChild(number);
      const content = element("div", "preview-answer-content");
      renderBlocks(content, question.answer_blocks || [], document.language);
      row.appendChild(content);
      paper.appendChild(row);
    });
  });
  return paper;
}

export function renderPrintDocument(documentIR, target) {
  if (!documentIR || !target) throw new Error("A document and print target are required");
  const language = documentIR.language === "ar" ? "ar" : "en";
  document.documentElement.lang = language;
  document.documentElement.dir = documentIR.direction || (language === "ar" ? "rtl" : "ltr");
  target.replaceChildren(
    documentIR.kind === "answers" ? renderAnswers(documentIR) : renderStudent(documentIR),
  );
}

function diagnosticPanel(document) {
  const diagnostics = (document.diagnostics || []).filter((item) => item && item.message);
  if (!diagnostics.length) return null;
  const panel = element("aside", "document-diagnostics");
  panel.setAttribute("role", "status");
  panel.appendChild(element("strong", "document-diagnostics-title", "Rendering warnings"));
  diagnostics.forEach((item) => {
    const location = [
      document.metadata?.model_label,
      item.section,
      item.question_number ? `Question ${item.question_number}` : "",
      item.component,
    ].filter(Boolean).join(" · ");
    const severity = item.blocking ? "Export blocked" : "Recoverable";
    panel.appendChild(element(
      "div", `document-diagnostic ${item.blocking ? "blocking" : "warning"}`,
      `${severity}: ${location ? `${location} — ` : ""}${item.message}`,
    ));
  });
  return panel;
}

function buildModelPreview(exam, metadata, index, canonical = null) {
  const model = element("div", `exam-model${index === 0 ? " active" : ""}`);
  model.dataset.index = String(index);
  const student = canonical?.student || buildDocumentIR(exam, metadata, "student");
  const answers = canonical?.answers || buildDocumentIR(exam, metadata, "answers");
  const warning = diagnosticPanel(student);
  if (warning) model.appendChild(warning);
  model.appendChild(renderStudent(student));
  model.appendChild(renderAnswers(answers));
  return model;
}

async function hydrateCanonicalDocuments(exams, metadata, contentEl) {
  if (!state.examId) return;
  const documents = await Promise.all(exams.map(async (exam) => {
    const model = encodeURIComponent(exam.model_number || 1);
    const [student, answers] = await Promise.all([
      getJSON(`/exams/${encodeURIComponent(state.examId)}/document-ir?model_number=${model}&kind=student`),
      getJSON(`/exams/${encodeURIComponent(state.examId)}/document-ir?model_number=${model}&kind=answers`),
    ]);
    if (!student.ok || !answers.ok) throw new Error("Canonical document IR is unavailable");
    return { student: student.data, answers: answers.data };
  }));
  contentEl.replaceChildren();
  exams.forEach((exam, index) => {
    contentEl.appendChild(buildModelPreview(exam, metadata, index, documents[index]));
  });
}

function orderedQuestionItems(exam) {
  const items = [];
  SECTION_ORDER.forEach((qtype) => {
    const section = exam.questions?.[qtype];
    const list = qtype === "fill_in_the_blank" ? section?.items || [] : (Array.isArray(section) ? section : []);
    list.forEach((item) => items.push({ qtype, item }));
  });
  return items;
}

function questionText(qtype, item) {
  if (qtype === "true_false") return item.statement;
  if (qtype === "definition") return item.term;
  if (qtype === "equation") return item.equation;
  return item.question;
}

function answerText(qtype, item, labels) {
  if (qtype === "mcq") return clean(item.correct_answer, labels.noAnswer);
  if (qtype === "true_false") {
    const answer = String(item.answer ?? "").toLowerCase();
    if (answer === "true") return labels.trueFalse.includes("صح") ? "صح" : "True";
    if (answer === "false") return labels.trueFalse.includes("صح") ? "غلط" : "False";
    return clean(item.answer, labels.noAnswer);
  }
  if (qtype === "fill_in_the_blank") {
    return (item.answers || []).map(clean).filter(Boolean).join(", ") || labels.noAnswer;
  }
  if (qtype === "equation" || qtype === "word_problem") {
    const steps = (item.solution_steps || []).map(clean).filter(Boolean);
    const finalAnswer = clean(item.final_answer, labels.noFinalAnswer);
    return [...steps, `${labels.finalAnswer}: ${finalAnswer}`].join(" → ");
  }
  let answer = clean(item.reference_answer, labels.noReferenceAnswer);
  const points = (item.key_points || []).map(clean).filter(Boolean);
  if (points.length) answer += ` | ${labels.keyPoints}: ${points.join("; ")}`;
  return answer;
}

function questionMarkdown(qtype, item, labels) {
  const lines = [clean(questionText(qtype, item), labels.missingQuestion)];
  if (qtype === "mcq") {
    Object.keys(item.options || {}).sort().forEach((letter) => {
      lines.push(`- ${letter}. ${clean(item.options[letter])}`);
    });
  } else if (qtype === "true_false") {
    lines.push(labels.trueFalse);
  }
  return lines.join("\n\n");
}

export function buildQuestionsFirstMarkdown(exams, metadata = {}) {
  const multiple = exams.length > 1;
  const questions = ["# Questions"];
  const answers = ["# Answers"];
  exams.forEach((exam, modelIndex) => {
    const language = clean(exam.document_language || metadata.document_language, "en");
    const labels = EXAM_LABELS[language] || EXAM_LABELS.en;
    const items = orderedQuestionItems(exam);
    if (multiple) {
      const model = format(labels.model, { n: exam.model_number || modelIndex + 1 });
      questions.push(`## ${model}`);
      answers.push(`## ${model}`);
    }
    items.forEach(({ qtype, item }, index) => {
      const questionLevel = multiple ? "###" : "##";
      questions.push(`${questionLevel} Question ${index + 1}\n\n${questionMarkdown(qtype, item, labels)}`);
      answers.push(`### Answer ${index + 1}\n\n${answerText(qtype, item, labels)}`);
    });
  });
  return `${questions.join("\n\n")}\n\n${answers.join("\n\n")}\n`;
}

export function renderExamOutput(exams, metadata = {}) {
  state.exams = exams;
  state.rawExam = buildQuestionsFirstMarkdown(exams, metadata);
  const tabsEl = document.getElementById("modelTabs");
  const contentEl = document.getElementById("examContent");
  contentEl.replaceChildren();
  if (exams.length > 1) {
    tabsEl.replaceChildren();
    tabsEl.style.display = "flex";
    exams.forEach((exam, index) => {
      const tab = element(
        "button",
        `model-tab${index === 0 ? " active" : ""}`,
        format(
          (EXAM_LABELS[exam.document_language || metadata.document_language] || EXAM_LABELS.en).model,
          { n: exam.model_number },
        ),
      );
      tab.addEventListener("click", () => activateModelTab(index));
      tabsEl.appendChild(tab);
    });
  } else {
    tabsEl.style.display = "none";
    tabsEl.replaceChildren();
  }
  exams.forEach((exam, index) => {
    contentEl.appendChild(buildModelPreview(exam, metadata, index));
  });
  // The local builder provides an immediate/offline fallback.  The authenticated
  // backend IR is canonical and replaces it as soon as it arrives.
  hydrateCanonicalDocuments(exams, metadata, contentEl).catch(() => {});
  document.getElementById("examOutput").classList.add("show");
}

function activateModelTab(index) {
  document.querySelectorAll(".model-tab").forEach((tab, tabIndex) =>
    tab.classList.toggle("active", tabIndex === index));
  document.querySelectorAll(".exam-model").forEach((model, modelIndex) =>
    model.classList.toggle("active", modelIndex === index));
}

export function copyExam() {
  navigator.clipboard.writeText(state.rawExam).then(() => {
    const toast = document.getElementById("copyToast");
    if (!toast) return;
    toast.textContent = t("toast.copied");
    toast.classList.remove("show");
    window.clearTimeout(copyExam.toastTimer);
    requestAnimationFrame(() => toast.classList.add("show"));
    copyExam.toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3000);
  });
}
