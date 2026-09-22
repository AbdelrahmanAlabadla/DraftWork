import { getJSON } from "./api.js";
import { renderExamOutput } from "./exam-view.js";
import { state } from "./state.js";
import { applyContentDirection, localizedError, t, uiLang } from "./i18n.js";

let cachedExams = null;

function setStatus(message, kind = "") {
  const status = document.getElementById("myExamsStatus");
  if (!status) return;
  status.textContent = message;
  status.dataset.kind = kind;
}

export async function loadExamIntoPreview(examId) {
  const saved = await getJSON(`/exams/${encodeURIComponent(examId)}`);
  if (!saved.ok) {
    throw new Error(localizedError(saved.data.detail, "status.load_exam_failed"));
  }
  state.examId = examId;
  renderExamOutput(saved.data.exams, saved.data.metadata || {});
  const url = new URL(window.location.href);
  url.searchParams.set("exam", examId);
  window.history.replaceState({}, "", url);
  document.getElementById("examOutput")?.scrollIntoView({ behavior: "smooth" });
}

function examCard(exam, dialog) {
  const card = document.createElement("article");
  card.className = "my-exams-dialog-item";

  const copy = document.createElement("div");
  copy.className = "my-exams-dialog-copy";
  const title = document.createElement("h3");
  title.textContent = exam.metadata?.exam_title || t("saved.generated_exam");
  applyContentDirection(title, title.textContent, exam.metadata?.document_language || "en");
  const details = document.createElement("p");
  const created = new Date(exam.created_at).toLocaleString(uiLang() === "ar" ? "ar" : "en");
  const models = t(exam.model_count === 1 ? "saved.model" : "saved.models", { n: exam.model_count });
  details.textContent = `${created} · ${models}`;
  copy.append(title, details);

  const open = document.createElement("button");
  open.className = "auth-link primary my-exams-open";
  open.type = "button";
  open.textContent = t("saved.open");
  open.addEventListener("click", async () => {
    open.disabled = true;
    setStatus(t("saved.opening"));
    try {
      await loadExamIntoPreview(exam.exam_id);
      dialog.close();
      setStatus("");
    } catch (error) {
      setStatus(localizedError(error.message, "status.load_exam_failed"), "error");
    } finally {
      open.disabled = false;
    }
  });

  card.append(copy, open);
  return card;
}

async function showMyExams(dialog, list) {
  list.classList.add("is-empty");
  list.textContent = t("saved.loading");
  setStatus("");
  if (!dialog.open) dialog.showModal();

  const response = await getJSON("/me/exams");
  if (!response.ok) {
    list.textContent = t("saved.load_failed");
    setStatus(localizedError(response.data.detail, "saved.try_again"), "error");
    return;
  }
  if (!response.data.exams.length) {
    list.textContent = t("saved.none");
    return;
  }
  cachedExams = response.data.exams;
  list.classList.remove("is-empty");
  list.replaceChildren(...cachedExams.map((exam) => examCard(exam, dialog)));
}

export function initMyExamsDialog() {
  const dialog = document.getElementById("myExamsDialog");
  const list = document.getElementById("myExamsList");
  const close = document.getElementById("myExamsClose");
  if (!dialog || !list || !close) return;

  document.addEventListener("draftwork:open-my-exams", () => {
    showMyExams(dialog, list).catch((error) => {
      list.classList.add("is-empty");
      list.textContent = t("saved.load_failed");
      setStatus(localizedError(error.message, "saved.try_again"), "error");
      if (!dialog.open) dialog.showModal();
    });
  });
  close.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  document.addEventListener("draftwork:language-changed", () => {
    if (cachedExams?.length && dialog.open) {
      list.replaceChildren(...cachedExams.map((exam) => examCard(exam, dialog)));
    }
  });
}
