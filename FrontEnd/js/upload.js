import {
  BASE, JobCancelledError, beginIdempotentOperation, completeIdempotentOperation,
  deleteJSON, getJSON, waitForJob,
} from "./api.js";
import { state } from "./state.js";
import { renderSectionsTree, showSectionsLoading, hideSectionsLoading } from "./topics.js";
import { authHeaders } from "./auth.js";
import { localizedError, stageLabel, t } from "./i18n.js";

let uploadController = null;
let uploadOperation = null;
let cancellationRequested = false;

function clearUploadUi(fileInput) {
  state.uploadDone = false;
  state.currentDocId = null;
  state.uploadJobId = null;
  document.getElementById("sectionsTree").innerHTML = "";
  document.getElementById("sectionsCount").textContent = "";
  document.getElementById("sectionsLoading").classList.remove("show");
  document.getElementById("filePill").classList.remove("show");
  setUploadStatus("", "");
  fileInput.value = "";
}

function setUploadStage(stage, type = "loading") {
  const bar = document.getElementById("uploadStatus");
  bar.dataset.stage = stage;
  setUploadStatus(type, stageLabel(stage));
}

export function initUpload() {
  const zone = document.getElementById("uploadZone");
  const fileInput = document.getElementById("fileInput");
  zone.addEventListener("click", () => fileInput.click());
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault(); zone.classList.remove("drag");
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
  document.addEventListener("draftwork:language-changed", () => {
    const stage = document.getElementById("uploadStatus").dataset.stage;
    if (stage) setUploadStatus(stage === "completed" ? "success" : "loading", stageLabel(stage));
  });
  document.getElementById("removeFile").addEventListener("click", async (event) => {
    event.stopPropagation();
    if (state.uploadDone) {
      clearUploadUi(fileInput);
      return;
    }
    cancellationRequested = true;
    setUploadStage("cancelling");
    uploadController?.abort();
    if (state.uploadJobId) {
      try {
        const result = await deleteJSON(`/jobs/${state.uploadJobId}`);
        if (!result.ok) {
          cancellationRequested = false;
          setUploadStatus("error", localizedError(result.data.detail, "status.cancel_failed"));
          return;
        }
      } catch (error) {
        cancellationRequested = false;
        setUploadStatus("error", localizedError(error.message, "status.cancel_failed"));
        return;
      }
    }
    completeIdempotentOperation(uploadOperation);
    uploadOperation = null;
    clearUploadUi(fileInput);
  });
}

function handleFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (ext !== "pdf") {
    setUploadStatus("error", t("status.unsupported_pdf"));
    return;
  }
  document.getElementById("fileName").textContent = file.name;
  document.getElementById("filePill").classList.add("show");
  uploadFile(file);
}

async function uploadFile(file) {
  cancellationRequested = false;
  setUploadStage("uploading");
  showSectionsLoading();
  const formData = new FormData();
  formData.append("file", file);
  const operation = beginIdempotentOperation(
    `upload:${file.name}:${file.size}:${file.lastModified}`
  );
  uploadOperation = operation;
  uploadController = new AbortController();
  try {
    const res = await fetch(`${BASE}/documents`, {
      method: "POST",
      credentials: "same-origin",
      headers: await authHeaders({ "Idempotency-Key": operation.key }),
      body: formData,
      signal: uploadController.signal,
    });
    uploadController = null;
    const data = await res.json();
    if (res.ok) {
      completeIdempotentOperation(operation);
      uploadOperation = null;
      state.uploadJobId = data.job_id;
      setUploadStage("queued");
      await waitForJob(data.job_id, (job) => {
        setUploadStage(job.stage);
      });
      const documentResponse = await getJSON(`/documents/${data.document_id}`);
      if (!documentResponse.ok) {
        throw new Error(localizedError(
          documentResponse.data.detail, "status.load_document_failed"
        ));
      }
      setUploadStage("completed", "success");
      state.uploadDone = true;
      state.currentDocId = data.document_id;
      renderSectionsTree(documentResponse.data.structure || {});
    } else {
      if (res.status < 500) completeIdempotentOperation(operation);
      setUploadStatus("error", localizedError(data.detail, "status.upload_failed"));
      hideSectionsLoading();
    }
  } catch (e) {
    if (cancellationRequested || e?.name === "AbortError" || e instanceof JobCancelledError) {
      completeIdempotentOperation(operation);
      uploadOperation = null;
      clearUploadUi(document.getElementById("fileInput"));
      return;
    }
    setUploadStatus("error", localizedError(e.message, "status.server_unreachable"));
    hideSectionsLoading();
  } finally {
    uploadController = null;
  }
}

export function setUploadStatus(type, msg) {
  const bar = document.getElementById("uploadStatus");
  const spinner = document.getElementById("uploadSpinner");
  const msgEl = document.getElementById("uploadMsg");
  bar.className = "status-bar";
  if (!type) {
    delete bar.dataset.stage;
    return;
  }
  bar.classList.add("show", type);
  spinner.style.display = type === "loading" ? "block" : "none";
  msgEl.textContent = msg;
}
