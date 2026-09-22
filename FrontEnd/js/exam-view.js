import { state } from "./state.js";
import { applyContentDirection, t } from "./i18n.js";

// Mirrors app.exports.common so screen, PDF, and DOCX use one visual contract.
const RESPONSE_LINE = "_".repeat(92);
const RESPONSE_LINE_COUNTS = { short_answer: 3, equation: 3, word_problem: 5, essay: 22 };
const SECTION_ORDER = [
  "mcq", "fill_in_the_blank", "true_false", "definition",
  "short_answer", "equation", "word_problem", "essay",
];
const SECTION_LABELS = {
  en: {
    mcq: "Multiple Choice Questions",
    fill_in_the_blank: "Fill in the Blank",
    true_false: "True / False",
    definition: "Define",
    short_answer: "Why Questions",
    equation: "Solve the equations and show your steps.",
    word_problem: "Solve the word problems and show your steps.",
    essay: "Essay",
  },
  ar: {
    mcq: "أسئلة الاختيار من متعدد",
    fill_in_the_blank: "أكمل الفراغ",
    true_false: "صح / خطأ",
    definition: "تعريف",
    short_answer: "علل",
    equation: "حل المعادلات وأظهر خطواتك.",
    word_problem: "حل المسائل الكلامية وأظهر خطواتك.",
    essay: "مقال",
  },
};
const EXAM_LABELS = {
  en: {
    examination: "Examination", model: "Model {n}", className: "Class",
    duration: "Duration", date: "Date", teacher: "Teacher",
    studentName: "Student Name", answerKey: "Answer Key - Model {n}",
    wordBank: "WORD BANK", trueFalse: "(   ) True     (   ) False",
    missingQuestion: "(missing question text)", missingTerm: "(missing term)",
    noAnswer: "No answer supplied", noFinalAnswer: "No final answer supplied",
    noReferenceAnswer: "No reference answer supplied", finalAnswer: "Final answer",
    keyPoints: "Key points",
  },
  ar: {
    examination: "امتحان", model: "نسخة {n}", className: "الصف",
    duration: "المدة", date: "التاريخ", teacher: "المعلم",
    studentName: "اسم الطالب", answerKey: "مفتاح الإجابة - نسخة {n}",
    wordBank: "بنك الكلمات", trueFalse: "(   ) صح     (   ) خطأ",
    missingQuestion: "(نص السؤال غير متاح)", missingTerm: "(المصطلح غير متاح)",
    noAnswer: "لم تُرفق إجابة", noFinalAnswer: "لم تُرفق إجابة نهائية",
    noReferenceAnswer: "لم تُرفق إجابة نموذجية", finalAnswer: "الإجابة النهائية",
    keyPoints: "النقاط الرئيسية",
  },
};

function format(template, params = {}) {
  let result = template;
  Object.entries(params).forEach(([key, value]) => {
    result = result.replace(`{${key}}`, String(value));
  });
  return result;
}

function contentElement(tag, className, text, fallbackLanguage = "en") {
  return applyContentDirection(element(tag, className, text), text, fallbackLanguage);
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

function sectionItems(questions, qtype) {
  const section = questions?.[qtype];
  if (qtype === "fill_in_the_blank") return section?.items || [];
  return Array.isArray(section) ? section : [];
}

function addTitleBlock(paper, exam, metadata, language, labels) {
  const header = element("header", "preview-title-block");
  const title = clean(metadata.exam_title || exam.title, labels.examination);
  header.appendChild(contentElement(
    "h1",
    "preview-exam-title",
    title,
    language,
  ));
  const modelLabel = format(labels.model, { n: exam.model_number || 1 });
  header.appendChild(contentElement("div", "preview-model", modelLabel, language));
  paper.appendChild(header);

  const fields = [
    [labels.className, metadata.class_name],
    [labels.duration, metadata.duration],
    [labels.date, metadata.exam_date],
    [labels.teacher, metadata.teacher_name],
  ];
  const grid = element("div", "preview-meta-grid");
  fields.forEach(([label, value]) => {
    const field = element("div", "preview-meta-field");
    field.appendChild(element("strong", "", `${label}:`));
    const fieldValue = clean(value, "_________");
    field.appendChild(contentElement("span", "", fieldValue, language));
    grid.appendChild(field);
  });
  paper.appendChild(grid);

  const student = element("div", "preview-student-row");
  student.appendChild(element("span", "", `${labels.studentName}: ${"_".repeat(34)}`));
  student.appendChild(element("span", "", `${labels.className}: ${"_".repeat(12)}`));
  paper.appendChild(student);
}

function addSectionHeading(parent, label, language) {
  parent.appendChild(contentElement("h2", "preview-section-heading", label, language));
}

function addQuestionStem(parent, number, text, language, fallback) {
  const stem = element("div", "preview-question-stem");
  const value = clean(text, fallback);
  applyContentDirection(stem, value, language);
  stem.appendChild(element("strong", "", language === "ar" ? `${number}. ` : `Q${number}. `));
  stem.appendChild(document.createTextNode(value));
  parent.appendChild(stem);
}

function addPlainNumberedStem(parent, number, text, language, fallback) {
  const stem = element("div", "preview-question-stem");
  const value = clean(text, fallback);
  applyContentDirection(stem, value, language);
  stem.appendChild(element("strong", "", `${number}. `));
  stem.appendChild(document.createTextNode(value));
  parent.appendChild(stem);
}

function addResponseLines(parent, count) {
  const lines = element("div", "preview-response-lines");
  for (let index = 0; index < count; index += 1) {
    lines.appendChild(element("div", "preview-response-line", RESPONSE_LINE));
  }
  parent.appendChild(lines);
}

function addStudentSection(paper, qtype, label, questions, language, examLabels) {
  const items = sectionItems(questions, qtype);
  if (!items.length) return;
  addSectionHeading(paper, label, language);

  if (qtype === "fill_in_the_blank") {
    const words = questions[qtype]?.word_bank || [];
    if (words.length) {
      const bank = element("div", "preview-word-bank");
      bank.appendChild(element("strong", "", examLabels.wordBank));
      bank.appendChild(contentElement("div", "", words.join("   |   "), language));
      paper.appendChild(bank);
    }
  }

  items.forEach((item, index) => {
    const question = element("section", `preview-question preview-${qtype}`);
    const text = qtype === "true_false" ? item.statement
      : qtype === "definition" ? item.term
      : qtype === "equation" ? item.equation
      : item.question;

    if (qtype === "definition") {
      const row = element("div", "preview-definition-row");
      const value = clean(text, examLabels.missingTerm);
      applyContentDirection(row, value, language);
      row.appendChild(element("strong", "", `${index + 1}. ${value}:`));
      row.appendChild(element("span", "preview-definition-line"));
      question.appendChild(row);
    } else if (qtype === "equation") {
      const row = element("div", "preview-equation-row");
      row.appendChild(element("strong", "preview-equation-number", `${index + 1}.`));
      const value = clean(text, examLabels.missingQuestion);
      row.appendChild(contentElement("div", "preview-equation-text", value, language));
      question.appendChild(row);
    } else if (qtype === "word_problem") {
      addPlainNumberedStem(question, index + 1, text, language, examLabels.missingQuestion);
    } else {
      addQuestionStem(question, index + 1, text, language, examLabels.missingQuestion);
    }

    if (qtype === "mcq") {
      const options = element("div", "preview-options");
      Object.keys(item.options || {}).sort().forEach((letter) => {
        const optionText = `${letter}. ${item.options[letter]}`;
        options.appendChild(contentElement("div", "preview-option", optionText, language));
      });
      question.appendChild(options);
    } else if (qtype === "true_false") {
      question.appendChild(element("div", "preview-tf-choices", examLabels.trueFalse));
    } else if (["short_answer", "equation", "word_problem", "essay"].includes(qtype)) {
      addResponseLines(question, RESPONSE_LINE_COUNTS[qtype]);
    }
    paper.appendChild(question);
  });
}

function answerText(qtype, item, labels) {
  if (qtype === "mcq") return clean(item.correct_answer, labels.noAnswer);
  if (qtype === "true_false") {
    const answer = clean(item.answer, labels.noAnswer);
    if (answer.toLowerCase() === "true") return labels.trueFalse.includes("صح") ? "صح" : "True";
    if (answer.toLowerCase() === "false") return labels.trueFalse.includes("صح") ? "خطأ" : "False";
    return answer;
  }
  if (qtype === "fill_in_the_blank") {
    return (item.answers || []).map((answer) => clean(answer)).filter(Boolean).join(", ") || labels.noAnswer;
  }
  if (qtype === "equation" || qtype === "word_problem") {
    const steps = (item.solution_steps || []).map((step) => clean(step)).filter(Boolean);
    const finalAnswer = clean(item.final_answer, labels.noFinalAnswer);
    return [...steps, `${labels.finalAnswer}: ${finalAnswer}`].join(" → ");
  }
  let answer = clean(item.reference_answer, labels.noReferenceAnswer);
  const points = (item.key_points || []).map((point) => clean(point)).filter(Boolean);
  if (points.length) answer += ` | ${labels.keyPoints}: ${points.join("; ")}`;
  return answer;
}

function buildAnswerKey(exam, labels, language, examLabels) {
  const paper = element("article", "exam-paper answer-key-paper");
  paper.dir = language === "ar" ? "rtl" : "ltr";
  paper.lang = language;
  const keyTitle = format(examLabels.answerKey, { n: exam.model_number || 1 });
  paper.appendChild(contentElement("h1", "preview-key-title", keyTitle, language));
  SECTION_ORDER.forEach((qtype) => {
    const items = sectionItems(exam.questions || {}, qtype);
    if (!items.length) return;
    addSectionHeading(paper, labels[qtype], language);
    items.forEach((item, index) => {
      const row = element("div", "preview-key-answer");
      const answer = answerText(qtype, item, examLabels);
      if (qtype === "mcq") {
        row.dir = language === "ar" ? "rtl" : "ltr";
        row.lang = language;
      } else {
        applyContentDirection(row, answer, language);
      }
      row.appendChild(element("strong", "", language === "ar" ? `${index + 1}. ` : `Q${index + 1}. `));
      row.appendChild(document.createTextNode(answer));
      paper.appendChild(row);
    });
  });
  return paper;
}

function buildModelPreview(exam, metadata, index) {
  const model = element("div", `exam-model${index === 0 ? " active" : ""}`);
  model.dataset.index = String(index);

  const language = clean(exam.document_language || metadata.document_language, "en");
  const labels = SECTION_LABELS[language] || SECTION_LABELS.en;
  const examLabels = EXAM_LABELS[language] || EXAM_LABELS.en;
  const studentPaper = element("article", "exam-paper student-paper");
  studentPaper.dir = language === "ar" ? "rtl" : "ltr";
  studentPaper.lang = language;
  addTitleBlock(studentPaper, exam, metadata, language, examLabels);
  SECTION_ORDER.forEach((qtype) => {
    addStudentSection(studentPaper, qtype, labels[qtype], exam.questions || {}, language, examLabels);
  });
  model.appendChild(studentPaper);
  model.appendChild(buildAnswerKey(exam, labels, language, examLabels));
  return model;
}

function orderedQuestionItems(exam) {
  const items = [];
  SECTION_ORDER.forEach((qtype) => {
    sectionItems(exam.questions || {}, qtype).forEach((item) => {
      items.push({ qtype, item });
    });
  });
  return items;
}

function questionText(qtype, item) {
  if (qtype === "true_false") return item.statement;
  if (qtype === "definition") return item.term;
  if (qtype === "equation") return item.equation;
  return item.question;
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
    exams.forEach((ex, index) => {
      const tab = element(
        "button",
        `model-tab${index === 0 ? " active" : ""}`,
        format(
          (EXAM_LABELS[ex.document_language || metadata.document_language] || EXAM_LABELS.en).model,
          { n: ex.model_number }
        )
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
    const toast = document.getElementById("copyToast");
    if (!toast) return;
    toast.textContent = t("toast.copied");
    toast.classList.remove("show");
    window.clearTimeout(copyExam.toastTimer);
    requestAnimationFrame(() => toast.classList.add("show"));
    copyExam.toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3000);
  });
}
