import { getJSON } from "./api.js";
import { renderExamOutput } from "./exam-view.js";
import { state } from "./state.js";

function setStatus(message, kind = "") {
  const status = document.getElementById("myExamsStatus");
  if (!status) return;
  status.textContent = message;
  status.dataset.kind = kind;
}

export async function loadExamIntoPreview(examId) {
  const saved = await getJSON(`/exams/${encodeURIComponent(examId)}`);
  if (!saved.ok) {
    throw new Error(saved.data.detail || "Could not load the saved exam.");
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
  title.textContent = exam.metadata?.exam_title || "Generated exam";
  const details = document.createElement("p");
  const created = new Date(exam.created_at).toLocaleString();
  details.textContent = `${created} · ${exam.model_count} model${exam.model_count === 1 ? "" : "s"}`;
  copy.append(title, details);

  const open = document.createElement("button");
  open.className = "auth-link primary my-exams-open";
  open.type = "button";
  open.textContent = "Open exam";
  open.addEventListener("click", async () => {
    open.disabled = true;
    setStatus("Opening exam…");
    try {
      await loadExamIntoPreview(exam.exam_id);
      dialog.close();
      setStatus("");
    } catch (error) {
      setStatus(error.message || "Could not load the saved exam.", "error");
    } finally {
      open.disabled = false;
    }
  });

  card.append(copy, open);
  return card;
}

async function showMyExams(dialog, list) {
  list.classList.add("is-empty");
  list.textContent = "Loading your exams…";
  setStatus("");
  if (!dialog.open) dialog.showModal();

  const response = await getJSON("/me/exams");
  if (!response.ok) {
    list.textContent = "Could not load your exams.";
    setStatus(response.data.detail || "Please try again.", "error");
    return;
  }
  if (!response.data.exams.length) {
    list.textContent = "There is no exam made yet.";
    return;
  }
  list.classList.remove("is-empty");
  list.replaceChildren(...response.data.exams.map((exam) => examCard(exam, dialog)));
}

export function initMyExamsDialog() {
  const dialog = document.getElementById("myExamsDialog");
  const list = document.getElementById("myExamsList");
  const close = document.getElementById("myExamsClose");
  if (!dialog || !list || !close) return;

  document.addEventListener("draftwork:open-my-exams", () => {
    showMyExams(dialog, list).catch((error) => {
      list.classList.add("is-empty");
      list.textContent = "Could not load your exams.";
      setStatus(error.message || "Please try again.", "error");
      if (!dialog.open) dialog.showModal();
    });
  });
  close.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
}
