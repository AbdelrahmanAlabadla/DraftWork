import { postJSON } from "./api.js";
import { state } from "./state.js";

const FORMS_LABEL = "Google Forms";
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
  // Keep the menu open while interacting with the email field or menu itself.
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
      else if (kind === "google-forms") exportForms();
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

function parseShareEmails() {
  return document.getElementById("shareEmails")
    .value.split(/[,;\s]+/).map((s) => s.trim()).filter(Boolean);
}

async function exportForms() {
  // Teacher-owned mode: no email field needed; central mode may pass emails.
  const shareRowVisible =
    document.getElementById("shareRow").style.display !== "none";
  const shareWith = shareRowVisible ? parseShareEmails() : [];
  setStatus("loading", "Exporting to Google Forms...");
  try {
    const { ok, status, data } = await postJSON(
      `/exams/${state.examId}/export/google-forms`,
      shareWith.length ? { share_with: shareWith } : {}
    );
    if (!ok) {
      if (status === 401) {
        setStatus("error", "Connect Google Account first (top of the page).");
        return;
      }
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : data.detail?.message || `Export failed (HTTP ${status}).`;
      setStatus("error", detail);
      return;
    }
    renderFormLinks(data.exports || [], data.errors || []);
    const failed = (data.errors || []).length;
    setStatus(
      failed ? "error" : "success",
      failed
        ? `Google Forms export partially failed: ${failed} model(s) had errors.`
        : `Google Forms created for ${(data.exports || []).length} model(s). Links below.`
    );
  } catch (e) {
    setStatus("error", "Google Forms export failed: cannot reach server.");
  }
}

async function triggerDownload(path, filename, body = null) {
  const options = { method: "POST" };
  if (body !== null) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const res = await fetch(`http://127.0.0.1:8000${path}`, options);
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

function renderFormLinks(exports, errors) {
  const container = document.getElementById("formsLinks");
  container.innerHTML = "";
  exports.forEach((exp) => {
    const row = document.createElement("div");
    row.className = "forms-link-row";
    const label = document.createElement("span");
    label.textContent = `Model ${exp.model_number}`;
    row.append(label);

    const viewLink = document.createElement("a");
    viewLink.href = exp.view_url;
    viewLink.target = "_blank";
    viewLink.rel = "noopener";
    viewLink.textContent = "Student view";
    row.appendChild(viewLink);

    const editLink = document.createElement("a");
    editLink.href = exp.edit_url;
    editLink.target = "_blank";
    editLink.rel = "noopener";
    editLink.textContent =
      exp.owner === "teacher" ? "Teacher edit (yours)" : "Teacher edit";
    row.appendChild(editLink);
    if ((exp.shared_with || []).length) {
      const shared = document.createElement("span");
      shared.textContent = `Shared with ${exp.shared_with.join(", ")} \u2713`;
      shared.style.color = "var(--success)";
      row.appendChild(shared);
    }
    (exp.warnings || [])
      .filter((w) => w.toLowerCase().startsWith("share with"))
      .forEach((w) => {
        const warn = document.createElement("span");
        warn.textContent = w;
        warn.style.color = "var(--danger)";
        row.appendChild(warn);
      });
    container.appendChild(row);
  });
  errors.forEach((err) => {
    const row = document.createElement("div");
    row.className = "forms-link-row";
    row.textContent = `Model ${err.model_number ?? "?"}: failed — ${err.error}`;
    row.style.color = "var(--danger)";
    container.appendChild(row);
  });
}
