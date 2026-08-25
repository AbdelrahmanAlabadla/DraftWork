import { state } from "./state.js";

// Mirrors app.exports.common so screen, PDF, and DOCX use one visual contract.
const RESPONSE_LINE = "_".repeat(92);
const RESPONSE_LINE_COUNTS = { short_answer: 3, essay: 22 };
const SECTION_ORDER = [
  ["mcq", "Multiple Choice Questions"],
  ["fill_in_the_blank", "Fill in the Blank"],
  ["true_false", "True / False"],
  ["short_answer", "Short Answer"],
  ["essay", "Essay"],
];

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

function sectionItems(questions, qtype) {
  const section = questions?.[qtype];
  if (qtype === "fill_in_the_blank") return section?.items || [];
  return Array.isArray(section) ? section : [];
}

function addTitleBlock(paper, exam, metadata) {
  const header = element("header", "preview-title-block");
  header.appendChild(element(
    "h1",
    "preview-exam-title",
    clean(metadata.exam_title || exam.title, "Examination")
  ));
  header.appendChild(element("div", "preview-model", `Model ${exam.model_number || 1}`));
  paper.appendChild(header);

  const fields = [
    ["Class", metadata.class_name],
    ["Duration", metadata.duration],
    ["Date", metadata.exam_date],
    ["Teacher", metadata.teacher_name],
  ];
  const grid = element("div", "preview-meta-grid");
  fields.forEach(([label, value]) => {
    const field = element("div", "preview-meta-field");
    field.appendChild(element("strong", "", `${label}:`));
    field.appendChild(element("span", "", clean(value, "_________")));
    grid.appendChild(field);
  });
  paper.appendChild(grid);

  const student = element("div", "preview-student-row");
  student.appendChild(element("span", "", `Student Name: ${"_".repeat(34)}`));
  student.appendChild(element("span", "", `Class: ${"_".repeat(12)}`));
  paper.appendChild(student);
}

function addSectionHeading(parent, label) {
  parent.appendChild(element("h2", "preview-section-heading", label));
}

function addQuestionStem(parent, number, text) {
  const stem = element("div", "preview-question-stem");
  stem.appendChild(element("strong", "", `Q${number}. `));
  stem.appendChild(document.createTextNode(clean(text, "(missing question text)")));
  parent.appendChild(stem);
}

function addResponseLines(parent, count) {
  const lines = element("div", "preview-response-lines");
  for (let index = 0; index < count; index += 1) {
    lines.appendChild(element("div", "preview-response-line", RESPONSE_LINE));
  }
  parent.appendChild(lines);
}

function addStudentSection(paper, qtype, label, questions) {
  const items = sectionItems(questions, qtype);
  if (!items.length) return;
  addSectionHeading(paper, label);

  if (qtype === "fill_in_the_blank") {
    const words = questions[qtype]?.word_bank || [];
    if (words.length) {
      const bank = element("div", "preview-word-bank");
      bank.appendChild(element("strong", "", "WORD BANK"));
      bank.appendChild(element("div", "", words.join("   |   ")));
      paper.appendChild(bank);
    }
  }

  items.forEach((item, index) => {
    const question = element("section", `preview-question preview-${qtype}`);
    const text = qtype === "true_false" ? item.statement : item.question;
    addQuestionStem(question, index + 1, text);

    if (qtype === "mcq") {
      const options = element("div", "preview-options");
      Object.keys(item.options || {}).sort().forEach((letter) => {
        options.appendChild(element("div", "preview-option", `${letter}. ${item.options[letter]}`));
      });
      question.appendChild(options);
    } else if (qtype === "true_false") {
      question.appendChild(element("div", "preview-tf-choices", "(   ) True     (   ) False"));
    } else if (qtype === "short_answer" || qtype === "essay") {
      addResponseLines(question, RESPONSE_LINE_COUNTS[qtype]);
    }
    paper.appendChild(question);
  });
}

function answerText(qtype, item) {
  if (qtype === "mcq") return clean(item.correct_answer, "No answer supplied");
  if (qtype === "true_false") return clean(item.answer, "No answer supplied");
  if (qtype === "fill_in_the_blank") {
    return (item.answers || []).map((answer) => clean(answer)).filter(Boolean).join(", ") || "No answer supplied";
  }
  let answer = clean(item.reference_answer, "No reference answer supplied");
  const points = (item.key_points || []).map((point) => clean(point)).filter(Boolean);
  if (points.length) answer += ` | Key points: ${points.join("; ")}`;
  return answer;
}

function buildAnswerKey(exam) {
  const paper = element("article", "exam-paper answer-key-paper");
  paper.appendChild(element("h1", "preview-key-title", `Answer Key - Model ${exam.model_number || 1}`));
  SECTION_ORDER.forEach(([qtype, label]) => {
    const items = sectionItems(exam.questions || {}, qtype);
    if (!items.length) return;
    addSectionHeading(paper, label);
    items.forEach((item, index) => {
      const row = element("div", "preview-key-answer");
      row.appendChild(element("strong", "", `Q${index + 1}. `));
      row.appendChild(document.createTextNode(answerText(qtype, item)));
      paper.appendChild(row);
    });
  });
  return paper;
}

function buildModelPreview(exam, metadata, index) {
  const model = element("div", `exam-model${index === 0 ? " active" : ""}`);
  model.dataset.index = String(index);

  const studentPaper = element("article", "exam-paper student-paper");
  addTitleBlock(studentPaper, exam, metadata);
  SECTION_ORDER.forEach(([qtype, label]) => {
    addStudentSection(studentPaper, qtype, label, exam.questions || {});
  });
  model.appendChild(studentPaper);
  model.appendChild(buildAnswerKey(exam));
  return model;
}

export function renderExamOutput(exams, metadata = {}) {
  state.exams = exams;
  state.rawExam = exams.map(
    (ex) => `## Exam Model ${ex.model_number}\n\n${ex.markdown}`
  ).join("\n\n---\n\n");

  const tabsEl = document.getElementById("modelTabs");
  const contentEl = document.getElementById("examContent");
  contentEl.replaceChildren();

  if (exams.length > 1) {
    tabsEl.replaceChildren();
    tabsEl.style.display = "flex";
    exams.forEach((ex, index) => {
      const tab = element(
        "button",
        `model-tab${index === 0 ? " active" : ""}`,
        `Model ${ex.model_number}`
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
    const btn = document.querySelector(".copy-btn");
    btn.textContent = "Copied!";
    setTimeout(() => (btn.textContent = "Copy Markdown"), 2000);
  });
}
