import { renderPrintDocument } from "./exam-view.js";

window.renderDraftworkPrintDocument = async (documentIR) => {
  renderPrintDocument(documentIR, document.getElementById("printRoot"));
  await document.fonts.ready;
};

window.getDraftworkPrintMetrics = () => {
  const root = document.getElementById("printRoot");
  return {
    items: root.querySelectorAll(
      ".student-paper .preview-question, .answer-key-paper .preview-key-answer"
    ).length,
    questions: root.querySelectorAll(".student-paper .preview-question").length,
    answers: root.querySelectorAll(".answer-key-paper .preview-key-answer").length,
    tables: root.querySelectorAll("table.exam-table").length,
    math: root.querySelectorAll(".katex").length,
  };
};
