import { postJSON } from "./api.js";
import { state } from "./state.js";
import { initUpload } from "./upload.js";
import { initSettings } from "./settings.js";
import { initTopics } from "./topics.js";
import { step, toggleType, updateTotal } from "./qtypes.js";
import { copyExam, renderExamOutput } from "./exam-view.js";
import { initExport } from "./export.js";
import { initGoogleAuth } from "./google-auth.js";

function inputValue(id) {
  return document.getElementById(id)?.value.trim() || "";
}

function fileToDataUrl(inputId) {
  const file = document.getElementById(inputId)?.files?.[0];
  if (!file) return Promise.resolve("");
  if (file.size > 2_000_000) {
    return Promise.reject(new Error("Each logo must be smaller than 2 MB."));
  }
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    return Promise.reject(new Error("Logos must be PNG, JPEG, or WebP images."));
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("A logo could not be read."));
    reader.readAsDataURL(file);
  });
}

function initGenerate() {
  const btn = document.getElementById("generateBtn");
  const genStatus = document.getElementById("genStatus");
  const examOutput = document.getElementById("examOutput");

  btn.addEventListener("click", async () => {
    if (!state.uploadDone) {
      alert("Please upload and index a file first.");
      return;
    }
    if (state.selectedChildren.size === 0) {
      alert("Please select at least one subsection to include in the exam.");
      return;
    }

    btn.disabled = true;
    genStatus.classList.add("show");
    examOutput.classList.remove("show");

    let leftLogoData = "";
    let rightLogoData = "";
    try {
      [leftLogoData, rightLogoData] = await Promise.all([
        fileToDataUrl("leftLogo"),
        fileToDataUrl("rightLogo"),
      ]);
    } catch (error) {
      alert(error.message);
      btn.disabled = false;
      genStatus.classList.remove("show");
      return;
    }

    const body = {
      document_id: state.currentDocId,
      num_models: state.numModels,
      mcq_count: state.enabled.mcq ? state.counts.mcq : 0,
      why_count: state.enabled.why ? state.counts.why : 0,
      essay_count: state.enabled.essay ? state.counts.essay : 0,
      tf_count: state.enabled.tf ? state.counts.tf : 0,
      fitb_count: state.enabled.fitb ? state.counts.fitb : 0,
      difficulty: state.difficulty,
      child_ids: [...state.selectedChildren],
      exam_title: inputValue("examTitle"),
      class_name: inputValue("examClass"),
      duration: inputValue("examDuration"),
      exam_date: inputValue("examDate"),
      teacher_name: inputValue("teacherName"),
      left_logo_data: leftLogoData,
      right_logo_data: rightLogoData,
    };

    try {
      const { ok, data } = await postJSON("/generate", body);
      if (ok && data.exams?.length) {
        state.examId = data.exam_id || null;
        renderExamOutput(data.exams, data.metadata || {});
      } else {
        alert(`Generation failed: ${data.detail}`);
      }
    } catch (e) {
      alert("Cannot reach server. Is FastAPI running?");
    } finally {
      btn.disabled = false;
      genStatus.classList.remove("show");
    }
  });
}

initUpload();
initSettings();
initTopics();
initExport();
initGenerate();
initGoogleAuth();

// Wire steppers/toggles by their card ids.
["mcq", "tf", "fitb", "why", "essay"].forEach((key) => {
  const card = document.getElementById(`card-${key}`);
  if (!card) return;
  const [minus, plus] = card.querySelectorAll(".stepper button");
  minus.addEventListener("click", () => step(key, -1));
  plus.addEventListener("click", () => step(key, 1));
  card.querySelector(".toggle").addEventListener("click", () => toggleType(key));
});

document.querySelector("#examOutput .copy-btn").addEventListener("click", copyExam);

updateTotal();
window.__dwBooted = true;
