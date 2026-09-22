import {
  JobCancelledError, beginIdempotentOperation, completeIdempotentOperation,
  deleteJSON, getJSON, postJSON, waitForJob,
} from "./api.js";
import { state } from "./state.js";
import { examStageLabel, initI18n, localizedError, t } from "./i18n.js";
import { initUpload } from "./upload.js";
import { initSettings } from "./settings.js";
import { initTopics } from "./topics.js";
import { step, toggleType, updateTotal } from "./qtypes.js";
import { copyExam, renderExamOutput } from "./exam-view.js";
import { initExport } from "./export.js";
import { initAuthNavigation } from "./auth.js";
import { initMyExamsDialog, loadExamIntoPreview } from "./my-exams-dialog.js";

initI18n();
// Account controls load independently so a slow Clerk CDN never blocks the
// anonymous upload and generation interface.
initAuthNavigation().catch(() => {});

function inputValue(id) {
  return document.getElementById(id)?.value.trim() || "";
}

function fileToDataUrl(inputId) {
  const file = document.getElementById(inputId)?.files?.[0];
  if (!file) return Promise.resolve("");
  if (file.size > 2_000_000) {
    return Promise.reject(new Error(t("status.logo_size")));
  }
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    return Promise.reject(new Error(t("status.logo_type")));
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error(t("status.logo_read")));
    reader.readAsDataURL(file);
  });
}

function renderGeneratedExam(data) {
  renderExamOutput(data.exams, data.metadata || {});
}

function initGenerate() {
  const btn = document.getElementById("generateBtn");
  const stopBtn = document.getElementById("stopGenerationBtn");
  const genStatus = document.getElementById("genStatus");
  const examOutput = document.getElementById("examOutput");
  let cancellationRequested = false;
  let generationCompleted = false;

  function renderGenerationProgress(stage, progress) {
    genStatus.dataset.stage = stage;
    genStatus.dataset.progress = String(progress);
    const message = document.getElementById("genStatusMsg");
    if (message) message.textContent = t("status.exam_progress", {
      stage: examStageLabel(stage), progress,
    });
  }

  document.addEventListener("draftwork:language-changed", () => {
    if (genStatus.dataset.stage) {
      renderGenerationProgress(
        genStatus.dataset.stage,
        Number(genStatus.dataset.progress || 0)
      );
    }
  });

  async function cancelGenerationJob() {
    if (!state.generationJobId) return;
    try {
      const result = await deleteJSON(`/jobs/${state.generationJobId}`);
      if (!result.ok) {
        cancellationRequested = false;
        stopBtn.disabled = false;
        stopBtn.textContent = t("gen.stop");
        alert(localizedError(result.data.detail, "status.cancel_failed"));
      }
    } catch (error) {
      cancellationRequested = false;
      stopBtn.disabled = false;
      stopBtn.textContent = t("gen.stop");
      alert(localizedError(error.message, "status.cancel_failed"));
    }
  }

  stopBtn.addEventListener("click", async () => {
    if (cancellationRequested) return;
    cancellationRequested = true;
    stopBtn.disabled = true;
    stopBtn.textContent = t("gen.stopping");
    const message = document.getElementById("genStatusMsg");
    if (message) message.textContent = t("gen.stopping");
    await cancelGenerationJob();
  });

  btn.addEventListener("click", async () => {
    if (!state.uploadDone) {
      alert(t("alert.upload_first"));
      return;
    }
    if (state.selectedChildren.size === 0) {
      alert(t("alert.select_sections"));
      return;
    }

    btn.disabled = true;
    cancellationRequested = false;
    generationCompleted = false;
    state.generationJobId = null;
    genStatus.classList.add("show");
    genStatus.classList.remove("complete");
    renderGenerationProgress("analyzing_request", 5);
    stopBtn.classList.add("show");
    stopBtn.disabled = false;
    stopBtn.textContent = t("gen.stop");
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
      stopBtn.classList.remove("show");
      return;
    }

    const body = {
      document_id: state.currentDocId,
      num_models: state.numModels,
      mcq_count: state.enabled.mcq ? state.counts.mcq : 0,
      fitb_count: state.enabled.fitb ? state.counts.fitb : 0,
      tf_count: state.enabled.tf ? state.counts.tf : 0,
      definition_count: state.enabled.definition ? state.counts.definition : 0,
      why_count: state.enabled.why ? state.counts.why : 0,
      equation_count: state.enabled.equation ? state.counts.equation : 0,
      word_problem_count: state.enabled.word_problem ? state.counts.word_problem : 0,
      essay_count: state.enabled.essay ? state.counts.essay : 0,
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
      const operation = beginIdempotentOperation(
        `generation:${state.currentDocId}`
      );
      const { ok, status, data } = await postJSON(
        `/documents/${state.currentDocId}/exam-jobs`, body, operation.key
      );
      if (ok && data.job_id) {
        completeIdempotentOperation(operation);
        state.generationJobId = data.job_id;
        if (cancellationRequested) await cancelGenerationJob();
        const job = await waitForJob(data.job_id, (current) => {
          renderGenerationProgress(current.stage, current.progress);
        });
        const examId = job.result?.exam_id;
        const examResponse = await getJSON(`/exams/${examId}`);
        if (!examResponse.ok) {
          throw new Error(localizedError(
            examResponse.data.detail, "status.load_exam_failed"
          ));
        }
        state.examId = examId;
        renderGeneratedExam(examResponse.data);
        generationCompleted = true;
        genStatus.classList.add("complete");
        renderGenerationProgress("completed", 100);
        // Let an already-open Eval Dashboard refresh immediately. The
        // dashboard still polls independently, so this is only a fast path.
        try {
          localStorage.setItem("dw:last-generation", String(Date.now()));
        } catch (_) {
          // Storage can be unavailable in private/restricted browser contexts.
        }
      } else {
        if (status < 500) completeIdempotentOperation(operation);
        alert(t("status.generation_failed", {
          detail: localizedError(data.detail, "status.unknown_error"),
        }));
      }
    } catch (e) {
      if (!(cancellationRequested || e instanceof JobCancelledError)) {
        alert(localizedError(e.message, "status.fastapi_unreachable"));
      }
    } finally {
      state.generationJobId = null;
      btn.disabled = false;
      if (!generationCompleted) genStatus.classList.remove("show");
      stopBtn.classList.remove("show");
      stopBtn.disabled = false;
    }
  });
}

initUpload();
initSettings();
initTopics();
initExport();
initGenerate();
initMyExamsDialog();

async function loadSavedExam() {
  const savedExamId = new URLSearchParams(window.location.search).get("exam");
  if (!savedExamId) return;
  await loadExamIntoPreview(savedExamId);
}

loadSavedExam().catch(() => {
  const output = document.getElementById("examOutput");
  if (output) {
    output.dataset.loadError = "true";
  }
});

// Wire steppers/toggles by their card ids.
["mcq", "fitb", "tf", "definition", "why", "equation", "word_problem", "essay"].forEach((key) => {
  const card = document.getElementById(`card-${key}`);
  if (!card) return;
  const [minus, plus] = card.querySelectorAll(".stepper button");
  minus.addEventListener("click", () => step(key, -1));
  plus.addEventListener("click", () => step(key, 1));
  card.querySelector(".toggle").addEventListener("click", () => toggleType(key));
});

document.querySelector("#examOutput .copy-btn").addEventListener("click", copyExam);

window.__dwBooted = true;
updateTotal();
