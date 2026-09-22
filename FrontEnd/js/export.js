import { state } from "./state.js";
import { BASE } from "./api.js";
import { authHeaders } from "./auth.js";
import { localizedError, t } from "./i18n.js";

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
  const toggleAll = document.getElementById("exportModelToggleAll");

  function updateToggleAll() {
    const checkboxes = [...document.querySelectorAll(
      "#exportModelList input[type='checkbox']"
    )];
    const allSelected = checkboxes.length > 0 && checkboxes.every((item) => item.checked);
    toggleAll.textContent = t(allSelected ? "export.clear_all" : "export.select_all");
  }

  toggleAll.addEventListener("click", () => {
    const checkboxes = [...document.querySelectorAll(
      "#exportModelList input[type='checkbox']"
    )];
    const selectAll = !checkboxes.every((item) => item.checked);
    checkboxes.forEach((item) => { item.checked = selectAll; });
    document.getElementById("exportModelError").textContent = "";
    updateToggleAll();
  });

  document.getElementById("exportModelList").addEventListener("change", updateToggleAll);
  document.addEventListener("draftwork:language-changed", updateToggleAll);

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
        setStatus("error", t("status.export_requires_exam"));
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
        t("status.select_model");
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
    setStatus("error", t("status.export_requires_exam"));
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
      text.textContent = t("export.model", { n: modelNumber });
      label.append(checkbox, text);
      return label;
    })
  );
  updateExportToggleLabel();
  document.getElementById("exportModelDialog").showModal();
}

function updateExportToggleLabel() {
  const checkboxes = [...document.querySelectorAll(
    "#exportModelList input[type='checkbox']"
  )];
  const allSelected = checkboxes.length > 0 && checkboxes.every((item) => item.checked);
  document.getElementById("exportModelToggleAll").textContent =
    t(allSelected ? "export.clear_all" : "export.select_all");
}

async function exportDocumentArchive(kind, modelNumbers) {
  const label = kind.toUpperCase();
  setStatus("loading", t("status.creating_export", { kind: label }));
  try {
    await triggerDownload(
      `/exams/${state.examId}/export/${kind}`,
      EXPORT_ARCHIVE_NAME,
      { model_numbers: modelNumbers }
    );
    setStatus("success", t("status.export_downloaded", { kind: label }));
  } catch (e) {
    setStatus("error", localizedError(e.message, "status.export_failed", { status: "" }));
  }
}

async function triggerDownload(path, filename, body = null) {
  const options = { method: "POST" };
  if (body !== null) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  options.headers = await authHeaders(options.headers || {});
  options.credentials = "same-origin";
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    let detail = t("status.export_failed", { status: res.status });
    try {
      const body = await res.json();
      if (body.detail) detail = localizedError(
        body.detail, "status.export_failed", { status: res.status }
      );
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
