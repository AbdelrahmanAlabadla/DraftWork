import { ApiError, apiGet, apiPost, appGet } from "./api.js";
import {
  authState,
  badge,
  bars,
  coverageNote,
  emptyState,
  errorState,
  kpi,
  loadingState,
  mountHeader,
  notice,
  panel,
  section,
  setFreshness,
  stat,
  table,
} from "./components.js";
import {
  compact,
  dateTime,
  debounce,
  duration,
  escapeHtml,
  labelize,
  money,
  num,
  pct,
  statusTone,
} from "./utils.js";

const params = new URLSearchParams(window.location.search);
const initialBenchmark = params.get("benchmark_id") || "";
const initialCompare = (params.get("compare") || "").split(",").filter(Boolean);

const DEFAULTS = {
  mcq: 10,
  true_false: 5,
  fill_in_the_blank: 0,
  definition: 0,
  why: 0,
  equation: 0,
  word_problem: 0,
  essay: 0,
};

const state = {
  view: initialBenchmark ? "detail" : initialCompare.length >= 2 ? "compare" : "list",
  options: null,
  sections: [],
  structureFor: "",
  structureLoading: false,
  config: {
    label: "",
    document_id: "",
    child_ids: [],
    tasks: { ...DEFAULTS },
    num_models: 1,
    difficulty: "mix",
    repeats: 1,
    stop_on_failure: true,
  },
  estimate: null,
  estimating: false,
  confirmed: false,
  submitting: false,
  submitKey: null,
  list: null,
  detail: null,
  compare: null,
  compareSelection: [],
  loading: false,
  error: null,
  poll: null,
};

/* --------------------------------------------------------------- helpers */

function limits() {
  return state.options?.limits || {
    max_questions: 150,
    max_models: 4,
    max_repeats: 10,
    max_child_ids: 1000,
  };
}

function selectedDocument() {
  return (state.options?.documents || []).find(
    (document) => document.document_id === state.config.document_id,
  );
}

function totalQuestions() {
  return Object.values(state.config.tasks || {}).reduce(
    (total, count) => total + Number(count || 0), 0,
  );
}

function configValid() {
  return Boolean(
    state.config.document_id
    && state.config.child_ids.length
    && totalQuestions() > 0
    && totalQuestions() <= limits().max_questions
    && state.config.repeats >= 1
    && state.config.repeats <= limits().max_repeats,
  );
}

function configBody() {
  return {
    label: state.config.label || "Benchmark",
    document_id: state.config.document_id,
    child_ids: state.config.child_ids,
    tasks: Object.fromEntries(
      Object.entries(state.config.tasks).filter(([, count]) => Number(count) > 0),
    ),
    num_models: Number(state.config.num_models || 1),
    difficulty: state.config.difficulty,
    repeats: Number(state.config.repeats || 1),
    stop_on_failure: Boolean(state.config.stop_on_failure),
  };
}

function statusBadge(status) {
  const key = String(status || "");
  const label = key === "completed" ? "Completed"
    : key === "running" ? "Running"
      : key === "queued" ? "Queued"
        : key === "cancelled" ? "Cancelled"
          : key === "failed" ? "Failed" : labelize(key);
  return badge(label, statusTone(key));
}

function metricValue(sectionName, metric, value) {
  if (value === null || value === undefined) return "—";
  if (["first_pass_rate", "validation_coverage", "final_acceptance_rate", "repair_success_rate", "recovery_success_rate"].includes(metric)) {
    return pct(value);
  }
  if (metric === "estimated_cost_usd") return money(value);
  if (metric === "duration_ms" || metric === "avg_latency_ms") return duration(value);
  return num(value);
}

function signedDelta(metric, delta) {
  if (delta === null || delta === undefined) return `<span class="delta-flat">—</span>`;
  const tone = delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : "delta-flat";
  const prefix = delta > 0 ? "+" : "";
  let formatted;
  if (["first_pass_rate", "validation_coverage", "final_acceptance_rate", "repair_success_rate", "recovery_success_rate"].includes(metric)) {
    formatted = `${prefix}${pct(delta)}`;
  } else if (metric === "estimated_cost_usd") {
    formatted = `${prefix}${money(delta)}`;
  } else if (metric === "duration_ms" || metric === "avg_latency_ms") {
    formatted = `${prefix}${duration(delta)}`;
  } else {
    formatted = `${prefix}${num(delta)}`;
  }
  return `<span class="${tone}">${formatted}</span>`;
}

/* ------------------------------------------------------------ rendering */

function renderSectionPicker() {
  const host = document.getElementById("sectionHost");
  if (!host) return;
  if (!state.config.document_id) {
    host.innerHTML = `<div class="bars-empty">Choose a ready document to select its sections.</div>`;
    return;
  }
  if (state.structureLoading) {
    host.innerHTML = loadingState("Loading document sections…");
    return;
  }
  if (!state.sections.length) {
    host.innerHTML = `<div class="bars-empty">This document has no selectable sections.</div>`;
    return;
  }
  const selected = new Set(state.config.child_ids);
  host.innerHTML = `
    <div class="row" style="margin-bottom:6px">
      <button class="btn ghost" type="button" data-sections="all">Select all</button>
      <button class="btn ghost" type="button" data-sections="none">Clear</button>
      <span class="muted small">${num(selected.size)} of ${num(state.sections.reduce((total, item) => total + (item.child_ids || []).length, 0))} sections selected</span>
    </div>
    <div class="section-picker">
      ${state.sections.map((item, index) => `
        <label>
          <input type="checkbox" data-section-index="${index}" ${(item.child_ids || []).some((id) => selected.has(id)) ? "checked" : ""} />
          <span>${escapeHtml(item.title || "Untitled section")}
            <span class="muted small">(${num((item.child_ids || []).length)} subsection${(item.child_ids || []).length === 1 ? "" : "s"})</span>
          </span>
        </label>`).join("")}
    </div>`;
  host.querySelectorAll("[data-section-index]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const index = Number(checkbox.dataset.sectionIndex);
      const childIds = state.sections[index]?.child_ids || [];
      const selectedIds = new Set(state.config.child_ids);
      if (checkbox.checked) childIds.forEach((id) => selectedIds.add(id));
      else childIds.forEach((id) => selectedIds.delete(id));
      state.config.child_ids = [...selectedIds];
      state.confirmed = false;
      state.submitKey = null;
      scheduleEstimate();
      updateRunButton();
      const counter = host.querySelector(".muted.small");
      if (counter) counter.textContent = `${num(state.config.child_ids.length)} sections selected`;
    });
  });
  host.querySelectorAll("[data-sections]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.sections === "all") {
        state.config.child_ids = state.sections.flatMap((item) => item.child_ids || []);
      } else {
        state.config.child_ids = [];
      }
      state.confirmed = false;
      state.submitKey = null;
      renderSectionPicker();
      scheduleEstimate();
      updateRunButton();
    });
  });
}

function renderEstimate() {
  const host = document.getElementById("estimateHost");
  if (!host) return;
  if (!configValid()) {
    host.innerHTML = `
      <div class="panel-head"><div><h3>Cost estimate</h3><p>Complete the configuration to estimate cost.</p></div></div>
      <div class="bars-empty">Select a document, at least one section and one question.</div>`;
    updateRunButton();
    return;
  }
  if (state.estimating) {
    host.innerHTML = `<div class="panel-head"><div><h3>Cost estimate</h3></div></div>${loadingState("Estimating from historical telemetry…")}`;
    updateRunButton();
    return;
  }
  const estimate = state.estimate;
  if (!estimate) {
    host.innerHTML = `
      <div class="panel-head"><div><h3>Cost estimate</h3></div></div>
      <div class="bars-empty">No estimate requested yet.</div>
      <div class="mt8"><button class="btn ghost" type="button" id="estimateBtn">Estimate</button></div>`;
    document.getElementById("estimateBtn")?.addEventListener("click", () => loadEstimate());
    updateRunButton();
    return;
  }
  if (!estimate.available) {
    host.innerHTML = `
      <div class="panel-head">
        <div><h3>Cost estimate</h3><p>${escapeHtml(estimate.reason || "No estimate available")}</p></div>
        ${badge("No estimate", "warn")}
      </div>
      <div class="estimate-total">—</div>
      <div class="detail-note">The benchmark can still run, but the cost is unknown from current telemetry.</div>
      ${confirmHtml()}`;
    bindConfirm();
    updateRunButton();
    return;
  }
  const value = estimate.estimate || {};
  host.innerHTML = `
    <div class="panel-head">
      <div><h3>Cost estimate</h3><p>${num(value.runs)} run(s) · ${num(value.total_questions)} questions</p></div>
      ${badge("Estimated", "accent")}
    </div>
    <div class="estimate-total">${money(value.estimated_cost_usd)}</div>
    <div class="muted small">Estimated total for this benchmark</div>
    <div class="mt12">
      <div class="estimate-line"><span>Provider calls</span><strong>${num(value.estimated_calls, 0)}</strong></div>
      <div class="estimate-line"><span>Input tokens</span><strong>${compact(value.estimated_input_tokens)}</strong></div>
      <div class="estimate-line"><span>Output tokens</span><strong>${compact(value.estimated_output_tokens)}</strong></div>
      <div class="estimate-line"><span>Duration</span><strong>${duration(value.estimated_duration_ms)}</strong></div>
      <div class="estimate-line"><span>Basis</span><strong>${num(estimate.basis?.runs)} historical run(s)</strong></div>
    </div>
    <div class="detail-note">${escapeHtml(estimate.note || "")}</div>
    ${confirmHtml()}`;
  bindConfirm();
  updateRunButton();
}

function confirmHtml() {
  return `
    <div class="confirm-box">
      <label class="checkbox">
        <input type="checkbox" id="benchmarkConfirm" ${state.confirmed ? "checked" : ""} />
        <span>I understand this benchmark makes real paid provider calls.</span>
      </label>
      <button class="btn primary" type="button" id="runBenchmark" ${state.confirmed ? "" : "disabled"}>
        Run benchmark
      </button>
    </div>`;
}

function bindConfirm() {
  const checkbox = document.getElementById("benchmarkConfirm");
  const button = document.getElementById("runBenchmark");
  if (checkbox) {
    checkbox.addEventListener("change", () => {
      state.confirmed = checkbox.checked;
      if (button) button.disabled = !state.confirmed || !configValid() || state.submitting;
    });
  }
  if (button) {
    button.disabled = !state.confirmed || !configValid() || state.submitting;
    button.addEventListener("click", submitBenchmark);
  }
}

function updateRunButton() {
  const button = document.getElementById("runBenchmark");
  if (!button) return;
  button.disabled = !state.confirmed || !configValid() || state.submitting || state.estimating;
  button.textContent = state.submitting ? "Starting…" : "Run benchmark";
}

function renderForm() {
  const documents = state.options?.documents || [];
  const questionTypes = state.options?.question_types || Object.keys(DEFAULTS);
  return `
    <div class="bench-layout">
      <div class="card panel">
        <div class="panel-head">
          <div><h3>New benchmark</h3><p>Runs through the production pipeline and is excluded from ordinary telemetry.</p></div>
        </div>
        <div class="bench-fields">
          <label class="field field-wide"><span>Label</span>
            <input class="input" id="benchLabel" maxlength="120" placeholder="e.g. Physics — baseline 28q" value="${escapeHtml(state.config.label)}" />
          </label>
          <label class="field field-wide"><span>Document</span>
            <select class="select" id="benchDocument">
              <option value="">Choose a ready document</option>
              ${documents.map((document) => `
                <option value="${escapeHtml(document.document_id)}" ${document.document_id === state.config.document_id ? "selected" : ""}>
                  ${escapeHtml(document.filename || document.document_id)}
                  ${document.subject ? `— ${escapeHtml(document.subject)}` : ""}
                </option>`).join("")}
            </select>
          </label>
          <label class="field"><span>Model versions</span>
            <select class="select" id="benchModels">
              ${Array.from({ length: limits().max_models }, (_, index) => index + 1).map((value) => `
                <option value="${value}" ${Number(state.config.num_models) === value ? "selected" : ""}>${value}</option>`).join("")}
            </select>
          </label>
          <label class="field"><span>Difficulty</span>
            <select class="select" id="benchDifficulty">
              ${(state.options?.difficulties || ["easy", "medium", "hard", "mix"]).map((value) => `
                <option value="${value}" ${state.config.difficulty === value ? "selected" : ""}>${labelize(value)}</option>`).join("")}
            </select>
          </label>
          <label class="field"><span>Repeated runs</span>
            <input class="input" id="benchRepeats" type="number" min="1" max="${limits().max_repeats}" value="${state.config.repeats}" />
          </label>
          <label class="checkbox" style="align-self:end;padding-bottom:6px">
            <input type="checkbox" id="benchStop" ${state.config.stop_on_failure ? "checked" : ""} />
            <span>Stop after the first failed run</span>
          </label>
        </div>
        <div class="subsection">
          <h4>Sections</h4>
          <div id="sectionHost"></div>
        </div>
        <div class="subsection">
          <h4>Question counts</h4>
          <div class="bench-fields">
            ${questionTypes.map((qtype) => `
              <label class="field"><span>${escapeHtml(labelize(qtype))}</span>
                <input class="input count-input" type="number" min="0" data-count="${escapeHtml(qtype)}"
                  value="${Number(state.config.tasks[qtype] || 0)}" />
              </label>`).join("")}
          </div>
          <div class="detail-note">Questions per model version. Total per run is multiplied by the number of model versions.</div>
        </div>
      </div>
      <aside class="card panel estimate-card" id="estimateHost"></aside>
    </div>`;
}

function benchmarkItem(item) {
  const summary = item.summary || {};
  const quality = summary.quality || {};
  const llm = summary.llm || {};
  const selected = state.compareSelection.includes(item.benchmark_id);
  const runs = summary.runs || {};
  const actualCost = llm.estimated_cost_usd;
  const estimatedCost = item.estimate?.estimated_cost_usd;
  const costLabel = actualCost !== null && actualCost !== undefined
    ? money(actualCost)
    : estimatedCost !== null && estimatedCost !== undefined
      ? `est. ${money(estimatedCost)}`
      : "—";
  return `
    <div class="bench-item ${selected ? "selected" : ""}" data-benchmark="${escapeHtml(item.benchmark_id)}">
      <a class="bench-item-title" href="benchmarks.html?benchmark_id=${encodeURIComponent(item.benchmark_id)}">
        <strong>${escapeHtml(item.label || "Benchmark")}</strong>
        <span class="bench-item-meta">${escapeHtml(dateTime(item.created_at))} · ${num(runs.completed)}/${num(runs.total)} runs · ${escapeHtml(item.config?.document_filename || "")}</span>
      </a>
      <span class="bench-item-num">${statusBadge(item.status)}<span>Status</span></span>
      <span class="bench-item-num">${quality.final_acceptance_rate === null || quality.final_acceptance_rate === undefined ? "—" : pct(quality.final_acceptance_rate)}<span>Final accept</span></span>
      <span class="bench-item-num" title="${actualCost === null || actualCost === undefined ? "Estimated cost; no provider telemetry recorded yet" : "Recorded estimated cost"}">${costLabel}<span>Cost</span></span>
      <span class="bench-item-num">${num(quality.final_valid)}<span>Finalized</span></span>
      <span class="row">
        <label class="checkbox" title="Select for comparison">
          <input type="checkbox" data-compare="${escapeHtml(item.benchmark_id)}" ${selected ? "checked" : ""} />
        </label>
        <a class="btn ghost" href="benchmarks.html?benchmark_id=${encodeURIComponent(item.benchmark_id)}">Open</a>
      </span>
    </div>`;
}

function renderList() {
  const items = state.list?.items || [];
  const noDocuments = state.options && (state.options.documents || []).length === 0;
  return `
    ${noDocuments ? notice("No ready documents are available for benchmarking. Upload and index a PDF in the generator first.", "warn") : ""}
    ${renderForm()}
    ${section(
      "Recorded benchmarks",
      "Select two or more benchmarks to compare equivalent configurations",
      items.length
        ? `<div class="bench-list">${items.map(benchmarkItem).join("")}</div>`
        : emptyState("No benchmarks yet", "Configure a benchmark above to create the first one."),
      `<button class="btn ghost" id="compareSelected" ${state.compareSelection.length >= 2 ? "" : "disabled"}>Compare selected</button>`,
    )}`;
}

function failedSlotsTable(summary) {
  const rows = (summary.failed_slots || []).map((slot) => [
    `<span class="mono">${escapeHtml(slot.slot_id)}</span>`,
    escapeHtml(labelize(slot.question_type || "unknown")),
    escapeHtml(labelize(slot.failure_stage || "unknown")),
    `<span class="badge bad">${escapeHtml(slot.failure_code || "unspecified_failure")}</span>`,
    num(slot.repair_attempts),
  ]);
  return table(
    ["Slot", "Question type", "Stage", "Failure code", "Repair attempts"],
    rows,
    { empty: "No failed slots recorded for this benchmark." },
  );
}

function renderDetail() {
  const detail = state.detail;
  const summary = detail.summary || {};
  const quality = summary.quality || {};
  const llm = summary.llm || {};
  const config = detail.config || {};
  const repro = detail.reproducibility || {};
  const progress = detail.progress || {};
  const active = ["queued", "running"].includes(detail.status);
  const llmKnown = Boolean(llm.telemetry_available ?? Number(llm.calls || 0) > 0);
  return `
    <div class="detail-head">
      <div class="detail-title">
        <h2>${escapeHtml(detail.label || "Benchmark")}</h2>
        <div class="detail-meta">
          <span>${escapeHtml(dateTime(detail.created_at))}</span>
          ${statusBadge(detail.status)}
          <span>${num(progress.finished)} / ${num(progress.total)} runs</span>
          ${progress.current_stage ? `<span>stage: ${escapeHtml(progress.current_stage)}</span>` : ""}
        </div>
      </div>
      <div class="row">
        <a class="btn ghost" href="benchmarks.html">← All benchmarks</a>
        ${active ? `<button class="btn danger" type="button" id="cancelBenchmark">Cancel benchmark</button>` : ""}
      </div>
    </div>
    ${detail.error_message ? notice(escapeHtml(detail.error_message), "bad") : ""}
    ${active ? `
      <div class="card panel mt12">
        <div class="panel-head"><div><h3>Execution progress</h3><p>Run ${num(progress.current_run)} of ${num(progress.total)} · ${escapeHtml(progress.current_status || "queued")}</p></div></div>
        <div class="progress"><span class="bar-fill ok" style="width:${progress.total ? (progress.finished / progress.total) * 100 : 0}%"></span></div>
        <div class="detail-note">This page polls every 5 seconds. Each run executes through the normal generation worker.</div>
      </div>` : ""}
    <div class="grid grid-4 mt12">
      ${kpi({ label: "Final acceptance", value: quality.final_acceptance_rate === null || quality.final_acceptance_rate === undefined ? "—" : pct(quality.final_acceptance_rate) })}
      ${kpi({ label: "First-pass acceptance", value: quality.first_pass_rate === null || quality.first_pass_rate === undefined ? "—" : pct(quality.first_pass_rate) })}
      ${kpi({ label: "Recovery success", value: quality.recovery_success_rate === null || quality.recovery_success_rate === undefined ? "—" : pct(quality.recovery_success_rate) })}
      ${kpi({ label: "Repair success", value: quality.repair_success_rate === null || quality.repair_success_rate === undefined ? "—" : pct(quality.repair_success_rate) })}
      ${kpi({ label: "Estimated cost", value: money(llm.estimated_cost_usd), detail: llmKnown ? "Estimated, not billed" : "No provider telemetry" })}
      ${kpi({ label: "Estimated cost per finalized question", value: money(summary.cost_per_finalized_question_usd), detail: "Attributable estimated cost / finalized questions", hint: "Estimated provider cost for this benchmark divided by its finalized questions." })}
      ${kpi({ label: "Finalized questions", value: num(quality.final_valid), detail: `${num(quality.requested)} requested` })}
      ${kpi({ label: "Runs", value: `${num(summary.runs?.completed)} / ${num(summary.runs?.total)}`, detail: `${num(summary.runs?.failed)} failed · ${num(summary.runs?.cancelled)} cancelled` })}
    </div>
    <div class="grid grid-4 mt12">
      ${kpi({ label: "Configured questions", value: num(config.total_questions), detail: `${num(config.questions_per_model)} per model version` })}
      ${kpi({ label: "Initially generated", value: num(quality.generated), detail: "before validation and repair" })}
      ${kpi({ label: "Finalized", value: num(quality.final_valid), detail: quality.final_acceptance_rate === null ? "" : `${pct(quality.final_acceptance_rate)} acceptance` })}
      ${kpi({ label: "Failed or missing", value: num(quality.final_shortfall), detail: `${num(quality.final_invalid)} invalid · ${num(quality.final_missing)} missing`, tone: quality.final_shortfall ? "bad" : "" })}
    </div>
    <div class="grid grid-4 mt12">
      ${kpi({ label: "Input tokens", value: llmKnown ? compact(llm.input_tokens) : "—", detail: llmKnown ? coverageNote(llm.usage_calls, llm.calls) : "No recorded provider calls" })}
      ${kpi({ label: "Output tokens", value: llmKnown ? compact(llm.output_tokens) : "—" })}
      ${kpi({ label: "Provider calls", value: llmKnown ? num(llm.calls) : "—", detail: llmKnown ? `${num(llm.failed)} failed · ${num(llm.retries)} retries` : "" })}
      ${kpi({ label: "Average / P95 latency", value: llmKnown ? `${duration(llm.avg_latency_ms)} / ${duration(llm.p95_latency_ms)}` : "—" })}
    </div>
    ${section("Runs", "Each run is one production generation job", table(
    ["Run", "Status", "Job stage", "Requested", "Generated", "Final", "First pass", "Cost", "Calls", "Duration", "Error"],
    (detail.runs || []).map((run) => {
      const runCalls = Number(run.llm?.calls || 0);
      return [
        `<span class="strong">${num(run.run_index)}</span>`,
        statusBadge(run.status),
        `<span class="mono">${escapeHtml(run.job_status || "—")}${run.job_stage ? ` / ${escapeHtml(run.job_stage)}` : ""}</span>`,
        num(run.requested),
        num(run.generated),
        num(run.final_valid),
        run.first_pass_rate === null ? "—" : pct(run.first_pass_rate),
        money(run.llm?.estimated_cost_usd),
        runCalls ? `${num(runCalls)} (${num(run.llm?.failed_calls)} failed)` : "—",
        duration(run.duration_ms),
        run.error_message ? `<span class="bad">${escapeHtml(run.error_message)}</span>` : "—",
      ];
    }),
    { empty: "No runs recorded yet.", minWidth: "900px" },
  ))}
    ${section("Failed questions", "Slots that could not be finalized, with their recorded reason", failedSlotsTable(summary))}
    <div class="split mt12">
      ${panel({ title: "Validator rejection reasons", body: bars(Object.entries(summary.validation_rejection_reasons || {}).map(([label, value]) => ({ label: labelize(label), value })), { empty: "No rejections recorded" }) })}
      ${panel({
        title: "Generation rejections",
        body: Object.keys(summary.generation_rejection_reasons || {}).length
          ? bars(Object.entries(summary.generation_rejection_reasons).map(([label, value]) => ({ label: labelize(label), value })), { tone: "warn" })
          : summary.telemetry?.generation_rejections_recorded
            ? `<div class="bars-empty">0 candidates rejected. Generation telemetry was recorded for this benchmark.</div>`
            : `<div class="bars-empty">Generation rejection telemetry was not recorded for this benchmark.</div>`,
      })}
    </div>
    ${section("Configuration", "Saved with the benchmark for reproducibility", `
      <div class="card panel">
        <div class="grid grid-4">
          ${stat({ label: "Document", value: config.document_filename || config.document_id || "—" })}
          ${stat({ label: "Selected child sections", value: num(config.section_count) })}
          ${stat({ label: "Questions per model", value: num(config.questions_per_model) })}
          ${stat({ label: "Model versions", value: num(config.num_models) })}
          ${stat({ label: "Difficulty", value: labelize(config.difficulty) })}
          ${stat({ label: "Repeats", value: num(config.repeats) })}
          ${stat({ label: "Configured questions", value: num(config.total_questions), detail: "requested before generation" })}
          ${stat({ label: "Catalog reuse", value: config.catalog_reuse ? "Automatic" : "—" })}
        </div>
        <div class="subsection"><h4>Reproducibility</h4><div class="repro-grid">
          ${Object.entries(repro).map(([key, value]) => {
            if (key === "versions" && value && typeof value === "object") return "";
            return `<div class="repro-item"><span>${escapeHtml(key.replaceAll("_", " "))}</span><strong>${escapeHtml(String(value ?? "—"))}</strong></div>`;
          }).join("")}
          ${Object.entries(repro.versions || {}).map(([key, value]) => `
            <div class="repro-item"><span>${escapeHtml(key.replaceAll("_", " "))}</span><strong>${escapeHtml(String(value ?? "—"))}</strong></div>`).join("")}
        </div></div>
      </div>`)}`;
}

function renderCompare() {
  const data = state.compare;
  const base = data.base;
  const comparisons = data.comparisons || [];
  const metricRows = (comparisons[0]?.metrics || []).map((metric) => {
    const cells = comparisons.map((comparison) => {
      const value = comparison.metrics.find((item) => item.metric === metric.metric && item.section === metric.section);
      return {
        value: value?.other ?? null,
        delta: value?.delta ?? null,
      };
    });
    return [
      `<span class="muted">${escapeHtml(metric.label)}</span>`,
      metricValue(metric.section, metric.metric, metric.base),
      ...cells.map((cell) => metricValue(metric.section, metric.metric, cell.value)),
      ...cells.map((cell) => signedDelta(metric.metric, cell.delta)),
    ];
  });
  const nonComparable = comparisons.filter((comparison) => !comparison.comparable);
  const headers = ["Metric", base.label || "Base",
    ...comparisons.map((comparison) => comparison.label || comparison.benchmark_id),
    ...comparisons.map((comparison) => `Δ vs ${comparison.label || comparison.benchmark_id}`)];
  return `
    <div class="detail-head">
      <div class="detail-title">
        <h2>Benchmark comparison</h2>
        <div class="detail-meta"><span>${num(comparisons.length + 1)} benchmarks</span></div>
      </div>
      <a class="btn ghost" href="benchmarks.html">← All benchmarks</a>
    </div>
    ${nonComparable.length
      ? nonComparable.map((comparison) => notice(
        `<strong>${escapeHtml(comparison.label)} is not directly comparable:</strong> ${escapeHtml(comparison.differences.join("; "))}. Deltas are still shown for reference.`,
        "warn",
      )).join("")
      : notice("All selected benchmarks use equivalent generation configurations.", "")}
    <div class="card panel mt12">
      ${table(headers, metricRows, { empty: "No comparable metrics yet.", minWidth: "760px" })}
    </div>
    <div class="detail-note">Deltas compare each benchmark with the base benchmark (${escapeHtml(base.label || base.benchmark_id)}). Positive deltas are improvements for acceptance/quality metrics and increases for cost, tokens and duration.</div>`;
}

/* ------------------------------------------------------------------ data */

async function loadOptions() {
  state.options = await apiGet("/benchmarks/options", {});
  if (!state.config.document_id && state.options.documents?.length) {
    // Keep the form empty; the user must choose deliberately.
  }
}

async function loadStructure(documentId) {
  if (!documentId) {
    state.sections = [];
    state.structureFor = "";
    renderSectionPicker();
    return;
  }
  state.structureLoading = true;
  state.structureFor = documentId;
  renderSectionPicker();
  try {
    const data = await appGet(`/documents/${encodeURIComponent(documentId)}`);
    state.sections = data.structure?.sections || [];
    if (!state.config.child_ids.length) {
      state.config.child_ids = state.sections.flatMap((item) => item.child_ids || []);
    }
  } catch (error) {
    state.sections = [];
    state.error = error;
  } finally {
    state.structureLoading = false;
    renderSectionPicker();
    scheduleEstimate();
  }
}

async function loadList() {
  state.list = await apiGet("/benchmarks", {});
  render();
}

async function loadDetail(benchmarkId = state.detail?.benchmark_id || initialBenchmark) {
  state.detail = await apiGet(`/benchmarks/${encodeURIComponent(benchmarkId)}`);
  state.view = "detail";
  render();
  schedulePoll();
}

async function loadCompare(ids) {
  state.compare = await apiGet("/benchmarks/compare", { ids: ids.join(",") });
  state.view = "compare";
  render();
}

async function loadEstimate() {
  if (!configValid()) {
    state.estimate = null;
    renderEstimate();
    return;
  }
  state.estimating = true;
  renderEstimate();
  try {
    state.estimate = await apiPost("/benchmarks/estimate", configBody());
  } catch (error) {
    state.estimate = {
      available: false,
      reason: error.message || "Could not estimate cost",
    };
  } finally {
    state.estimating = false;
    renderEstimate();
  }
}

const scheduleEstimate = debounce(() => {
  if (!configValid()) {
    state.estimate = null;
    renderEstimate();
    return;
  }
  loadEstimate();
}, 550);

function schedulePoll() {
  if (state.poll) window.clearTimeout(state.poll);
  if (state.view === "detail" && state.detail && ["queued", "running"].includes(state.detail.status)) {
    state.poll = window.setTimeout(() => {
      loadDetail().catch(() => {});
    }, 5000);
  }
}

async function submitBenchmark() {
  if (state.submitting || !configValid() || !state.confirmed) return;
  state.submitting = true;
  state.submitKey = state.submitKey || (crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`);
  renderEstimate();
  updateRunButton();
  try {
    const created = await apiPost(
      "/benchmarks",
      configBody(),
      { "Idempotency-Key": state.submitKey },
    );
    window.location.href = `benchmarks.html?benchmark_id=${encodeURIComponent(created.benchmark_id)}`;
  } catch (error) {
    state.submitting = false;
    state.submitKey = null;
    state.error = error;
    render();
  }
}

async function cancelBenchmark() {
  if (!state.detail) return;
  if (!window.confirm("Cancel this benchmark and stop the active run?")) return;
  try {
    state.detail = await apiPost(`/benchmarks/${encodeURIComponent(state.detail.benchmark_id)}/cancel`);
    render();
  } catch (error) {
    state.error = error;
    render();
  }
}

/* --------------------------------------------------------------- render */

function render() {
  const body = document.getElementById("pageBody");
  if (!body) return;
  if (state.loading && !state.list && !state.detail && !state.compare) {
    body.innerHTML = loadingState();
    return;
  }
  if (state.error) {
    body.innerHTML = state.error instanceof ApiError && state.error.isAuth
      ? authState()
      : errorState(state.error.message || "Unknown error");
    return;
  }
  try {
    if (state.view === "detail" && state.detail) {
      body.innerHTML = renderDetail();
      document.getElementById("cancelBenchmark")?.addEventListener("click", cancelBenchmark);
    } else if (state.view === "compare" && state.compare) {
      body.innerHTML = renderCompare();
    } else {
      body.innerHTML = renderList();
      bindFormEvents();
      renderSectionPicker();
      renderEstimate();
      bindListEvents();
    }
    setFreshness({ fetchedAt: new Date().toISOString() });
  } catch (error) {
    console.error(error);
    body.innerHTML = errorState("The page could not be rendered. Reload to try again.");
  }
}

function bindFormEvents() {
  const label = document.getElementById("benchLabel");
  const documentSelect = document.getElementById("benchDocument");
  const models = document.getElementById("benchModels");
  const difficulty = document.getElementById("benchDifficulty");
  const repeats = document.getElementById("benchRepeats");
  const stop = document.getElementById("benchStop");
  label?.addEventListener("input", () => {
    state.config.label = label.value;
    state.submitKey = null;
  });
  documentSelect?.addEventListener("change", () => {
    state.config.document_id = documentSelect.value;
    state.config.child_ids = [];
    state.config.label = state.config.label || (selectedDocument()?.filename || "");
    state.confirmed = false;
    state.estimate = null;
    state.submitKey = null;
    loadStructure(documentSelect.value);
    renderEstimate();
  });
  models?.addEventListener("change", () => {
    state.config.num_models = Number(models.value);
    state.confirmed = false;
    state.submitKey = null;
    scheduleEstimate();
  });
  difficulty?.addEventListener("change", () => {
    state.config.difficulty = difficulty.value;
    state.confirmed = false;
    state.submitKey = null;
    scheduleEstimate();
  });
  repeats?.addEventListener("change", () => {
    state.config.repeats = Math.max(1, Math.min(limits().max_repeats, Number(repeats.value) || 1));
    repeats.value = String(state.config.repeats);
    state.confirmed = false;
    state.submitKey = null;
    scheduleEstimate();
    updateRunButton();
  });
  stop?.addEventListener("change", () => {
    state.config.stop_on_failure = stop.checked;
    state.submitKey = null;
  });
  document.querySelectorAll("[data-count]").forEach((input) => {
    input.addEventListener("input", () => {
      const qtype = input.dataset.count;
      const value = Math.max(0, Number(input.value) || 0);
      state.config.tasks[qtype] = value;
      state.confirmed = false;
      state.submitKey = null;
      scheduleEstimate();
      updateRunButton();
    });
  });
}

function bindListEvents() {
  document.querySelectorAll("[data-compare]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const id = checkbox.dataset.compare;
      const selection = new Set(state.compareSelection);
      if (checkbox.checked) selection.add(id);
      else selection.delete(id);
      state.compareSelection = [...selection];
      const button = document.getElementById("compareSelected");
      if (button) button.disabled = state.compareSelection.length < 2;
      checkbox.closest(".bench-item")?.classList.toggle("selected", checkbox.checked);
    });
  });
  const compareButton = document.getElementById("compareSelected");
  if (compareButton) {
    compareButton.addEventListener("click", () => {
      if (state.compareSelection.length < 2) return;
      window.location.href = `benchmarks.html?compare=${state.compareSelection.map(encodeURIComponent).join(",")}`;
    });
  }
}

/* ----------------------------------------------------------------- boot */

async function boot() {
  state.loading = true;
  render();
  try {
    await loadOptions();
    if (state.view === "detail") {
      await loadDetail(initialBenchmark);
    } else if (state.view === "compare") {
      await loadCompare(initialCompare);
    } else {
      await loadList();
    }
  } catch (error) {
    state.error = error;
  } finally {
    state.loading = false;
    render();
  }
}

mountHeader("benchmarks", {
  onRefresh: () => {
    if (state.view === "detail") loadDetail().catch(() => {});
    else boot();
  },
});
boot();
