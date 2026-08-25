import { BASE } from "./api.js";
import { state } from "./state.js";
import { renderSectionsTree, showSectionsLoading, hideSectionsLoading } from "./topics.js";

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
  document.getElementById("removeFile").addEventListener("click", () => {
    state.uploadDone = false;
    state.currentDocId = null;
    document.getElementById("sectionsTree").innerHTML = "";
    document.getElementById("sectionsCount").textContent = "";
    document.getElementById("sectionsLoading").classList.remove("show");
    document.getElementById("filePill").classList.remove("show");
    setUploadStatus("", "");
    fileInput.value = "";
  });
}

function handleFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["pdf", "txt"].includes(ext)) {
    setUploadStatus("error", "Only .pdf and .txt files are supported.");
    return;
  }
  document.getElementById("fileName").textContent = file.name;
  document.getElementById("filePill").classList.add("show");
  uploadFile(file);
}

async function uploadFile(file) {
  setUploadStatus("loading", "Uploading and indexing...");
  showSectionsLoading();
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await fetch(`${BASE}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    if (res.ok) {
      setUploadStatus("success", data.message || "File indexed successfully.");
      state.uploadDone = true;
      state.currentDocId = data.document_id;
      renderSectionsTree(data.structure || {});
    } else {
      setUploadStatus("error", data.detail || "Upload failed.");
      hideSectionsLoading();
    }
  } catch (e) {
    setUploadStatus("error", "Cannot reach server. Is FastAPI running?");
    hideSectionsLoading();
  }
}

export function setUploadStatus(type, msg) {
  const bar = document.getElementById("uploadStatus");
  const spinner = document.getElementById("uploadSpinner");
  const msgEl = document.getElementById("uploadMsg");
  bar.className = "status-bar";
  if (!type) return;
  bar.classList.add("show", type);
  spinner.style.display = type === "loading" ? "block" : "none";
  msgEl.textContent = msg;
}
