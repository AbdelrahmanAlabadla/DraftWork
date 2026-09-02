import { BASE } from "./api.js";

const REFRESH_MS = 30000;
const number = (value) => Number(value || 0).toLocaleString();
const pct = (value) => value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
const rateOrDash = (value) => value == null ? "—" : pct(value);

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function renderKpis(summary) {
  const overall = summary.overall || {};
  const rates = overall.rates || summary.rates || {};
  setText("generationRate", rateOrDash(rates.generation_completion_rate));
  setText("generationRaw", `${number(overall.generated_first)} / ${number(overall.requested_questions)} questions`);
  setText("validationRate", rateOrDash(rates.first_validation_pass_rate));
  setText("validationRaw", `${number(overall.validation_passed_first)} / ${number(overall.validation_total_first)} questions`);
  setText("repairRate", rateOrDash(rates.repair_success_rate));
  setText("repairRaw", `${number(overall.repair_succeeded)} / ${number(overall.repair_sent)} repairs`);
  setText("finalRate", rateOrDash(rates.final_success_rate));
  setText("finalRaw", `${number(overall.final_valid)} / ${number(overall.requested_questions)} questions`);
  setText("examRuns", number(summary.total_exam_runs));
  setText("requestedTotal", number(overall.requested_questions));
  setText("finalValidTotal", number(overall.final_valid));
  setText("finalInvalidTotal", number(overall.final_invalid));
  setText("finalUnvalidatedTotal", number(overall.final_unvalidated));
  setText("finalMissingTotal", number(overall.final_missing));
  setText("missingTotal", number(overall.final_missing_or_invalid));
}

function renderPipeline(summary) {
  const o = summary.overall || {};
  const stages = [
    ["Requested", o.requested_questions, null, ""],
    ["Generated First", o.generated_first, o.requested_questions, ""],
    ["Shortfall Recovered", o.shortfall_generated, o.missing_first, ""],
    ["Entered Validation", o.validation_total_first, o.requested_questions, ""],
    ["Passed First", o.validation_passed_first, o.validation_total_first, ""],
    ["Failed First", o.validation_failed_first, o.validation_total_first, "pipeline-error"],
    ["Unvalidated First", o.validation_unvalidated_first, o.validation_total_first, "pipeline-warning"],
    ["Needed Repair", o.repair_sent, null, ""],
    ["Repair Succeeded", o.repair_succeeded, o.repair_sent, ""],
    ["Final Valid", o.final_valid, o.requested_questions, "pipeline-success"],
  ];
  document.getElementById("pipelineStages").innerHTML = stages.map(([label, count, denominator, tone]) => `
    <div class="pipeline-stage ${tone}">
      <div class="pipeline-stage-label">${label}</div>
      <div class="pipeline-stage-count">${number(count)}</div>
      ${denominator ? `<div class="pipeline-stage-rate">${pct(Number(count || 0) / Number(denominator))}</div>` : ""}
    </div>`).join("");
}

function renderReasons(id, reasons, totalId) {
  const entries = Object.entries(reasons || {}).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((sum, [, count]) => sum + Number(count || 0), 0);
  setText(totalId, number(total));
  const max = Math.max(...entries.map(([, count]) => Number(count || 0)), 0);
  document.getElementById(id).innerHTML = entries.map(([reason, count]) => `
    <div class="reason-row">
      <span class="reason-name">${reason.replaceAll("_", " ")}</span>
      <span class="reason-track"><span class="reason-fill" style="width:${max ? (Number(count) / max) * 100 : 0}%"></span></span>
      <span class="reason-count">${number(count)}</span>
    </div>`).join("");
}

function renderModels(summary) {
  const body = document.querySelector("#modelTable tbody");
  const models = Object.entries(summary.models || {}).sort(([a], [b]) => Number(a) - Number(b));
  document.getElementById("modelEmpty").style.display = models.length ? "none" : "block";
  body.innerHTML = models.map(([modelNumber, model]) => {
    const rates = model.rates || {};
    return `<tr><td>Model ${modelNumber}</td>
      <td class="rate-cell">${rateOrDash(rates.generation_completion_rate)}</td>
      <td class="rate-cell">${rateOrDash(rates.first_validation_pass_rate)}</td>
      <td class="rate-cell">${rateOrDash(rates.repair_success_rate)}</td>
      <td class="rate-cell final-rate">${rateOrDash(rates.final_success_rate)}</td></tr>`;
  }).join("");
}

function renderRecent(summary) {
  const rows = summary.recent_exam_runs || [];
  const body = document.querySelector("#recentTable tbody");
  document.getElementById("recentEmpty").style.display = rows.length ? "none" : "block";
  body.innerHTML = rows.map((run) => {
    const statusClass = run.status === "Healthy" ? "" : run.status === "No telemetry" ? " none" : " attention";
    const when = run.generated_at ? new Date(run.generated_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "—";
    return `<tr><td class="technical-id">${run.exam_id || "—"}</td><td>${when}</td><td>${number(run.requested)}</td>
      <td class="rate-cell">${rateOrDash(run.first_pass_rate)}</td><td>${number(run.repairs_sent)}</td>
      <td class="rate-cell final-rate">${rateOrDash(run.final_success_rate)}</td>
      <td><span class="status-pill${statusClass}">${run.status || "—"}</span></td></tr>`;
  }).join("");
}

function render(summary) {
  renderKpis(summary);
  renderPipeline(summary);
  renderReasons("generationReasons", summary.generation_rejection_reasons, "generationRejectTotal");
  renderReasons("validationReasons", summary.validation_failure_reasons, "validationFailTotal");
  renderReasons("validatorReasons", summary.validator_failure_reasons, "validatorFailTotal");
  renderModels(summary);
  renderRecent(summary);
  const updated = summary.updated_at ? new Date(summary.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "just now";
  setText("updatedAt", `Updated ${updated}`);
  document.getElementById("evalState").textContent = "";
}

async function refresh() {
  const state = document.getElementById("evalState");
  try {
    const response = await fetch(`${BASE}/api/eval-summary`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Endpoint returned ${response.status}`);
    render(await response.json());
  } catch (error) {
    state.textContent = "Unable to load telemetry. Retrying automatically…";
    setText("updatedAt", "Connection unavailable");
  }
}

document.getElementById("refreshBtn").addEventListener("click", refresh);
window.addEventListener("storage", (event) => {
  if (event.key === "dw:last-generation") refresh();
});
refresh();
window.setInterval(refresh, REFRESH_MS);
