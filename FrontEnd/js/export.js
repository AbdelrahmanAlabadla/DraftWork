import { state } from "./state.js";
import { BASE } from "./api.js";

const EXPORT_ARCHIVE_NAME = "SmartExam_Export.zip";
let pendingDocumentKind = null;

function setStatus(type, msg) {
  const bar = document.getElementById("exportStatus");
  const spinner = document.getElementById("exportSpinner");
  const msgEl = document.getElementById("exportMsg");
  bar.className = "status-bar";
  if (!type) return;
  bar.classList.add("show", type);
  spinner.style.display = type === "loading" ? "block" : "none";
  msgEl.textContent = msg;
}

export function initExport() {
  const wrap = document.getElementById("exportWrap");
  const btn = document.getElementById("exportBtn");

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!state.examId) return;
    wrap.classList.toggle("open");
  });
  document.addEventListener("click", () => wrap.classList.remove("open"));
  wrap.addEventListener("click", (e) => e.stopPropagation());

  document.querySelectorAll(".export-item").forEach((item) => {
    item.addEventListener("click", () => {
      wrap.classList.remove("open");
      if (!state.examId) {
        setStatus("error", "Generate an exam before exporting.");
        return;
      }
      const kind = item.dataset.export;
      if (kind === "pdf" || kind === "docx") beginDocumentExport(kind);
    });
  });

  document.getElementById("exportModelConfirm").addEventListener("click", () => {
    const selected = [...document.querySelectorAll(
      "#exportModelList input[type='checkbox']:checked"
    )].map((input) => Number(input.value));
    if (!selected.length) {
      document.getElementById("exportModelError").textContent =
        "Select at least one model.";
      return;
    }
    const kind = pendingDocumentKind;
    document.getElementById("exportModelDialog").close();
    if (kind) exportDocumentArchive(kind, selected);
  });
}

function availableModelNumbers() {
  return (state.exams || []).map((exam, index) =>
    Number(exam.model_number || index + 1)
  );
}

function beginDocumentExport(kind) {
  const models = availableModelNumbers();
  if (!models.length) {
    setStatus("error", "Generate an exam before exporting.");
    return;
  }
  if (models.length === 1) {
    exportDocumentArchive(kind, models);
    return;
  }

  pendingDocumentKind = kind;
  document.getElementById("exportModelError").textContent = "";
  document.getElementById("exportModelList").replaceChildren(
    ...models.map((modelNumber) => {
      const label = document.createElement("label");
      label.className = "export-model-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = String(modelNumber);
      const text = document.createElement("span");
      text.textContent = `Model ${modelNumber}`;
      label.append(checkbox, text);
      return label;
    })
  );
  document.getElementById("exportModelDialog").showModal();
}

async function exportDocumentArchive(kind, modelNumbers) {
  const label = kind.toUpperCase();
  setStatus("loading", `Creating ${label} export...`);
  try {
    await triggerDownload(
      `/exams/${state.examId}/export/${kind}`,
      EXPORT_ARCHIVE_NAME,
      { model_numbers: modelNumbers }
    );
    setStatus("success", `${label} ZIP downloaded.`);
  } catch (e) {
    setStatus("error", e.message);
  }
}

async function triggerDownload(path, filename, body = null) {
  const options = { method: "POST" };
  if (body !== null) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  options.credentials = "same-origin";
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    let detail = `Export failed (HTTP ${res.status}).`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch (_) { /* keep default */ }
    throw new Error(detail);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
