import { ApiError, apiGet } from "./api.js";
import {
  authState,
  errorState,
  kpi,
  loadingState,
  mountHeader,
  notice,
  panel,
  section,
  setFreshness,
  stat,
} from "./components.js";
import { COLORS, lineChart } from "./charts.js";
import {
  compact,
  dateDay,
  dateShort,
  duration,
  escapeHtml,
  money,
  num,
  pct,
  statusTone,
} from "./utils.js";

const state = {
  days: 30,
  includeBenchmarks: false,
  data: null,
  loading: false,
  error: null,
};

function toneForRate(rate, good = 0.9, warn = 0.75) {
  if (rate === null || rate === undefined) return "";
  return rate >= good ? "ok" : rate >= warn ? "" : rate >= 0.5 ? "warn" : "bad";
}

function rateValue(rate) {
  return rate === null || rate === undefined ? "—" : pct(rate);
}

function renderKpis(data) {
  const quality = data.quality || {};
  const jobs = data.jobs || {};
  const requests = data.requests || {};
  const llm = data.llm || {};
  const evaluated = llm.scopes?.evaluated || {};
  const unattributed = llm.scopes?.unattributed || {};
  const costPerFinal = quality.cost_per_finalized_question_usd;
  const recoveryKnown = (quality.recovery_entered || 0) > 0;
  return `
    <div class="grid overview-kpis">
      ${kpi({
        label: "Evaluation records",
        value: num(data.runs?.generation_runs),
        detail: `${num(requests.historical_without_job)} historical records without a job link`,
        hint: "Completed generation requests that produced an evaluation record. Older records were written before job tracking existed, so this is not the same as the number of tracked jobs.",
      })}
      ${kpi({
        label: "Known request completion",
        value: rateValue(requests.completion_rate),
        detail: `${num(requests.completed)} completed · ${num(requests.failed)} failed · ${num(requests.cancelled)} cancelled${requests.active ? ` · ${num(requests.active)} active` : ""}`,
        tone: toneForRate(requests.completion_rate, 0.9, 0.75),
        hint: "Completed known generation requests divided by finished known requests (completed + failed + cancelled). The population combines evaluation records with tracked jobs that failed before producing one.",
      })}
      ${kpi({
        label: "Tracked job success",
        value: rateValue(jobs.success_rate),
        detail: `${num(jobs.completed)} of ${num(jobs.finished)} tracked jobs · ${num(jobs.failed)} failed · ${num(jobs.cancelled)} cancelled`,
        tone: toneForRate(jobs.success_rate, 0.9, 0.7),
        hint: "Completed tracked generation jobs divided by finished tracked jobs. Only newer requests have job records; older evaluation records are not part of this denominator.",
      })}
      ${kpi({
        label: "First-pass acceptance",
        value: rateValue(quality.first_pass_rate),
        detail: `${num(quality.validation_passed_first)} / ${num(quality.validation_total_first)} validated · coverage ${pct(quality.validation_coverage)}`,
        tone: toneForRate(quality.first_pass_rate, 0.85, 0.7),
        hint: "Questions that passed their first validation divided by questions that received a validation verdict. Coverage shows how many generated questions actually reached a verdict.",
      })}
      ${kpi({
        label: "Final acceptance",
        value: rateValue(quality.final_acceptance_rate),
        detail: `${num(quality.final_valid)} / ${num(quality.requested)} requested questions`,
        tone: toneForRate(quality.final_acceptance_rate, 0.95, 0.85),
        hint: "Questions that reached the authoritative finalized state divided by all requested question slots in the evaluation records.",
      })}
      ${kpi({
        label: "Repair success",
        value: rateValue(quality.repair_success_rate),
        detail: `${num(quality.repair_succeeded)} / ${num(quality.repair_sent)} questions sent to repair`,
        tone: toneForRate(quality.repair_success_rate, 0.8, 0.6),
        hint: "Questions whose repaired revision was written back and later passed validation, divided by unique questions sent to repair. This is question-level success.",
      })}
      ${kpi({
        label: "Recovery success",
        value: recoveryKnown ? rateValue(quality.recovery_success_rate) : "—",
        detail: recoveryKnown
          ? `${num(quality.recovery_recovered)} / ${num(quality.recovery_entered)} slots entering recovery`
          : "No slot recovery telemetry",
        tone: recoveryKnown ? toneForRate(quality.recovery_success_rate, 0.8, 0.6) : "",
        hint: "Question slots that entered repair or replacement recovery and were eventually finalized, divided by slots that entered recovery. This is slot-level success and is not the same as repair success.",
      })}
      ${kpi({
        label: "Provider request failures",
        value: llm.calls ? pct(llm.failure_rate) : "—",
        detail: llm.calls
          ? `${num(llm.failed)} / ${num(llm.calls)} recorded requests${unattributed.calls ? ` · ${num(unattributed.calls)} unattributed legacy` : ""}`
          : "No provider telemetry",
        tone: llm.failure_rate === null || llm.failure_rate === undefined
          ? ""
          : llm.failure_rate > 0.15 ? "bad" : llm.failure_rate > 0.05 ? "warn" : "ok",
        hint: "Provider requests whose recorded status was not success, divided by all recorded requests. This includes legacy requests without a job link; retries are counted separately.",
      })}
      ${kpi({
        label: "Estimated cost per finalized question",
        value: money(costPerFinal),
        detail: evaluated.estimated_cost_usd === null || evaluated.estimated_cost_usd === undefined
          ? "No attributable cost telemetry"
          : `${money(evaluated.estimated_cost_usd)} over ${num(data.coverage?.final_valid_with_calls)} finalized questions`,
        hint: "Estimated provider cost attributable to evaluated runs, divided by questions those runs finalized. Legacy requests without a job link are excluded. This is an estimate from configured token prices, not billed cost.",
      })}
    </div>`;
}

function renderRequests(data) {
  const requests = data.requests || {};
  const jobs = data.jobs || {};
  return `
    <div class="request-grid">
      ${panel({
        title: "Generation requests",
        subtitle: "De-duplicated known requests",
        body: `<div class="grid grid-2">
          ${stat({ label: "Known requests", value: num(requests.total) })}
          ${stat({ label: "Completed", value: num(requests.completed) })}
          ${stat({ label: "Failed", value: num(requests.failed) })}
          ${stat({ label: "Cancelled", value: num(requests.cancelled) })}
          ${stat({ label: "Active now", value: num(requests.active) })}
          ${stat({ label: "Completion rate", value: requests.completion_rate === null ? "—" : pct(requests.completion_rate) })}
        </div>`,
      })}
      ${panel({
        title: "Tracked jobs",
        subtitle: "Newer requests with queue records",
        body: `<div class="grid grid-2">
          ${stat({ label: "Tracked jobs", value: num(jobs.total) })}
          ${stat({ label: "Completed", value: num(jobs.completed) })}
          ${stat({ label: "Failed", value: num(jobs.failed) })}
          ${stat({ label: "Cancelled", value: num(jobs.cancelled) })}
          ${stat({ label: "Active", value: num(jobs.active) })}
          ${stat({ label: "Job success", value: jobs.success_rate === null ? "—" : pct(jobs.success_rate) })}
        </div>`,
      })}
      ${panel({
        title: "Model exams",
        subtitle: "Model versions across known requests",
        body: `<div class="grid grid-2">
          ${stat({ label: "Requested", value: num(requests.model_exams_requested) })}
          ${stat({ label: "Recorded payloads", value: num(requests.model_exams_produced) })}
          ${stat({ label: "Unavailable payload", value: num(requests.model_exams_unavailable), detail: "legacy records without exam payload" })}
          ${stat({ label: "Not produced", value: num(requests.model_exams_not_produced), detail: "failed or cancelled requests" })}
        </div>`,
      })}
      ${panel({
        title: "Questions",
        subtitle: "Requested versus finalized",
        body: `<div class="grid grid-2">
          ${stat({ label: "Requested", value: num(requests.questions_requested), detail: `${num(requests.questions_requested_evaluated)} evaluated · ${num(requests.questions_requested_failed_jobs)} failed/cancelled jobs` })}
          ${stat({ label: "Finalized", value: num(requests.questions_finalized) })}
          ${stat({ label: "Shortfall", value: num(requests.question_shortfall), detail: "requested minus finalized" })}
          ${stat({ label: "Unavailable final status", value: num(requests.questions_unavailable) })}
        </div>`,
      })}
    </div>
    ${notice(
      `<strong>Why ${num(data.runs?.generation_runs)} evaluation records but ${num(jobs.total)} tracked jobs?</strong>
       ${num(requests.historical_without_job)} evaluation records predate job tracking, and
       ${num(requests.jobs_without_evaluation)} tracked jobs ended (failed or cancelled) before an evaluation record was written.
       The ${num(requests.total)} known requests count each of those once.`,
    )}`;
}

function averageCard(items, denominatorLabel) {
  return `
    <div class="metric-groups">
      ${items.map((item) => stat({
        label: item.label,
        value: item.value,
        detail: item.detail || "",
        hint: item.hint || "",
      })).join("")}
    </div>
    <div class="detail-note">${escapeHtml(denominatorLabel)}</div>`;
}

function renderAverages(data) {
  const llm = data.llm || {};
  const perRun = llm.averages_per_generation_run || {};
  const perExam = llm.averages_per_model_exam || {};
  const coverage = data.coverage || {};
  const runItems = [
    { label: "Estimated cost", value: money(perRun.cost_usd), hint: "Attributable estimated provider cost per evaluated generation run." },
    { label: "LLM requests", value: num(perRun.calls, 1) },
    { label: "Successful", value: num(perRun.successful_calls, 2) },
    { label: "Failed", value: num(perRun.failed_calls, 2) },
    { label: "Retries", value: num(perRun.retries, 2) },
    { label: "Input tokens", value: compact(perRun.input_tokens) },
    { label: "Output tokens", value: compact(perRun.output_tokens) },
    { label: "Reasoning tokens", value: compact(perRun.reasoning_tokens) },
    { label: "Cached tokens", value: compact(perRun.cached_tokens) },
  ];
  const examItems = [
    { label: "Estimated cost", value: money(perExam.cost_usd), hint: "Run-level attributable cost allocated equally across the model exams produced by those runs." },
    { label: "LLM requests", value: num(perExam.calls, 1) },
    { label: "Successful", value: num(perExam.successful_calls, 2) },
    { label: "Failed", value: num(perExam.failed_calls, 2) },
    { label: "Retries", value: num(perExam.retries, 2) },
    { label: "Input tokens", value: compact(perExam.input_tokens) },
    { label: "Output tokens", value: compact(perExam.output_tokens) },
    { label: "Reasoning tokens", value: compact(perExam.reasoning_tokens) },
    { label: "Cached tokens", value: compact(perExam.cached_tokens) },
  ];
  return `
    <div class="averages-grid">
      ${panel({
        title: "Per generation run",
        subtitle: "Only runs with job-linked provider telemetry",
        body: averageCard(
          runItems,
          `Denominator: ${num(perRun.denominator)} evaluated run(s) with recorded provider requests.`,
        ),
      })}
      ${panel({
        title: "Per model exam",
        subtitle: "Allocated equally across model versions",
        body: averageCard(
          examItems,
          `Denominator: ${num(perExam.denominator)} model exam(s) produced by those runs.`,
        ),
      })}
    </div>
    ${notice(
      `Provider telemetry covers ${num(coverage.runs_with_telemetry)} of
       ${num(coverage.evaluation_records)} evaluation records. Legacy requests without a job link
       are excluded from these averages and are shown separately below.`,
    )}`;
}

function renderAttribution(data) {
  const llm = data.llm || {};
  const scopes = llm.scopes || {};
  const jobs = data.jobs || {};
  const scopeCard = (title, scope, description) => panel({
    title,
    subtitle: description,
    body: `<div class="grid grid-2">
      ${stat({ label: "Estimated cost", value: money(scope?.estimated_cost_usd) })}
      ${stat({ label: "Requests", value: num(scope?.calls) })}
      ${stat({ label: "Input tokens", value: compact(scope?.input_tokens) })}
      ${stat({ label: "Output tokens", value: compact(scope?.output_tokens) })}
    </div>`,
  });
  return `
    <div class="split">
      ${scopeCard("Attributed to evaluated runs", scopes.evaluated, "Used for per-run averages")}
      ${scopeCard("Other generation jobs", scopes.other_generation, "Failed or cancelled jobs that still spent tokens")}
    </div>
    <div class="split mt12">
      ${scopeCard("Unattributed legacy requests", scopes.unattributed, "Recorded before job attribution; cannot be assigned to a run")}
      ${scopeCard("Other application requests", scopes.other_jobs, "Document-ingestion or title requests outside exam generation")}
    </div>
    <div class="totals-strip mt12">
      ${stat({ label: "All recorded cost", value: money(llm.estimated_cost_usd), detail: "sum of all four scopes" })}
      ${stat({ label: "All recorded requests", value: num(llm.calls) })}
      ${stat({ label: "All input tokens", value: compact(llm.input_tokens) })}
      ${stat({ label: "All output tokens", value: compact(llm.output_tokens) })}
    </div>
    <div class="mt12">
      ${panel({
        title: "Reliability and duration",
        subtitle: "Recorded worker and provider behaviour",
        body: `<div class="grid grid-4">
          ${stat({ label: "Provider requests", value: num(llm.calls) })}
          ${stat({ label: "Failed requests", value: num(llm.failed) })}
          ${stat({ label: "Provider failure rate", value: llm.failure_rate === null ? "—" : pct(llm.failure_rate) })}
          ${stat({ label: "Retries", value: num(llm.retries) })}
          ${stat({ label: "Avg call latency", value: duration(llm.latency?.avg_ms) })}
          ${stat({ label: "P95 call latency", value: duration(llm.latency?.p95_ms) })}
          ${stat({ label: "Average job duration", value: duration(jobs.duration?.avg_ms) })}
          ${stat({ label: "P95 job duration", value: duration(jobs.duration?.p95_ms) })}
        </div>`,
      })}
    </div>`;
}

function funnel(data) {
  const pipeline = data.pipeline || {};
  const stages = pipeline.question_stages || [];
  const requested = stages[0]?.value || 0;
  const max = Math.max(...stages.map((stage) => Number(stage.value || 0)), 1);
  return stages.map((stage) => {
    const value = Number(stage.value || 0);
    const share = requested ? value / requested : null;
    return `
      <div class="funnel-step">
        <span class="funnel-step-label">${escapeHtml(stage.label)}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${Math.max(1, (value / max) * 100)}%"></span></span>
        <span class="funnel-step-count">${num(value)}</span>
        <span class="funnel-step-share">${share === null ? "" : pct(share, 0)}</span>
      </div>`;
  }).join("");
}

function renderPipeline(data) {
  const terminal = data.recovery?.terminal_status || {};
  const terminalEntries = Object.entries(terminal).sort((a, b) => b[1] - a[1]);
  const slotCards = terminalEntries.length
    ? `<div class="slot-outcomes">
        ${terminalEntries.map(([status, value]) => stat({
          label: status.replaceAll("_", " "),
          value: num(value),
          hint: "Authoritative slot terminal status recorded during finalization.",
        })).join("")}
      </div>`
    : notice("No slot-level outcome telemetry is recorded for this period.");
  return `
    ${panel({
      title: "Question funnel",
      subtitle: "All stages count questions; the share is relative to requested questions.",
      body: `<div class="funnel">${funnel(data)}</div>
        <div class="detail-note">Validation coverage: ${pct(data.quality?.validation_coverage)} of generated questions received a verdict. “Entered repair” counts unique questions sent to repair.</div>
        <div class="subsection"><h4>Final slot outcomes</h4>${slotCards}</div>`,
    })}`;
}

function chartCard({ title, description, series, labels, yFormat, yMax = null, valueFormat, tooltipLabels = null }) {
  const hasData = series.some((item) => (item.values || []).some(
    (value) => value !== null && value !== undefined,
  ));
  return `
    <div class="card chart-card">
      <div class="chart-head"><div><h4>${escapeHtml(title)}</h4><span>${escapeHtml(description)}</span></div></div>
      ${hasData
        ? lineChart({ labels, series, yFormat, yMax, valueFormat, tooltipLabels })
        : `<div class="bars-empty">No recorded data in this period</div>`}
    </div>`;
}

function renderTrends(data) {
  const trends = data.trends || [];
  if (!trends.length) {
    return `<div class="card panel">${loadingState("No trend data in this period.")}</div>`;
  }
  const labels = trends.map((point) => dateShort(point.day));
  const tooltipLabels = trends.map((point) => dateDay(point.day));
  const values = (key) => trends.map((point) => point[key]);
  const charts = [
    chartCard({
      title: "Question acceptance",
      description: "Share of questions accepted on the first validation versus after repair.",
      labels,
      tooltipLabels,
      series: [
        { name: "First-pass acceptance", values: values("first_pass_rate"), color: COLORS.info, sampleCounts: values("runs"), sampleLabel: "runs" },
        { name: "Final acceptance", values: values("final_acceptance_rate"), color: COLORS.success, sampleCounts: values("runs"), sampleLabel: "runs" },
      ],
      yFormat: (value) => pct(value, 0),
      yMax: 1,
      valueFormat: pct,
    }),
    chartCard({
      title: "Repair success",
      description: "Question-level: repaired questions later passing validation.",
      labels,
      tooltipLabels,
      series: [
        { name: "Repair success", values: values("repair_success_rate"), color: COLORS.warning, sampleCounts: values("runs"), sampleLabel: "runs" },
      ],
      yFormat: (value) => pct(value, 0),
      yMax: 1,
      valueFormat: pct,
    }),
    chartCard({
      title: "Recovery success",
      description: "Slot-level: slots finalized after entering repair or replacement recovery.",
      labels,
      tooltipLabels,
      series: [
        { name: "Recovery success", values: values("recovery_success_rate"), color: COLORS.accent, sampleCounts: values("recovery_samples"), sampleLabel: "slots" },
      ],
      yFormat: (value) => pct(value, 0),
      yMax: 1,
      valueFormat: pct,
    }),
    chartCard({
      title: "Average estimated cost per run",
      description: "Per evaluated run with provider telemetry; not total daily spend.",
      labels,
      tooltipLabels,
      series: [
        { name: "Cost per run", values: values("avg_cost_per_run"), color: COLORS.accent, sampleCounts: values("runs_with_telemetry"), sampleLabel: "runs with telemetry" },
      ],
      yFormat: (value) => `$${value.toFixed(2)}`,
      valueFormat: money,
    }),
    chartCard({
      title: "Average exam duration",
      description: "Recorded wall-clock time per generation run (minutes).",
      labels,
      tooltipLabels,
      series: [
        { name: "Exam duration", values: values("avg_duration_ms"), color: COLORS.info, sampleCounts: values("duration_samples"), sampleLabel: "runs" },
      ],
      yFormat: (value) => `${(value / 60000).toFixed(1)}m`,
      valueFormat: duration,
    }),
    chartCard({
      title: "Average LLM call latency",
      description: "Mean provider request duration per day (seconds).",
      labels,
      tooltipLabels,
      series: [
        { name: "Call latency", values: values("avg_call_latency_ms"), color: COLORS.muted, sampleCounts: values("latency_calls"), sampleLabel: "calls" },
      ],
      yFormat: (value) => `${(value / 1000).toFixed(1)}s`,
      valueFormat: duration,
    }),
    chartCard({
      title: "Average input tokens per run",
      description: "Only evaluated runs with provider usage telemetry.",
      labels,
      tooltipLabels,
      series: [
        { name: "Input tokens", values: values("avg_input_tokens_per_run"), color: COLORS.info, sampleCounts: values("runs_with_telemetry"), sampleLabel: "runs" },
      ],
      valueFormat: compact,
    }),
    chartCard({
      title: "Average output tokens per run",
      description: "Output tokens reported by the provider per evaluated run.",
      labels,
      tooltipLabels,
      series: [
        { name: "Output tokens", values: values("avg_output_tokens_per_run"), color: COLORS.success, sampleCounts: values("runs_with_telemetry"), sampleLabel: "runs" },
      ],
      valueFormat: compact,
    }),
    chartCard({
      title: "Provider failure rate",
      description: "Failed requests divided by recorded requests that day. Days with fewer than 5 requests are omitted.",
      labels,
      tooltipLabels,
      series: [
        {
          name: "Failure rate",
          values: trends.map((point) => (Number(point.provider_calls || 0) >= 5
            ? point.provider_failure_rate
            : null)),
          color: COLORS.danger,
          sampleCounts: values("provider_calls"),
          sampleLabel: "requests",
        },
      ],
      yFormat: (value) => pct(value, 0),
      yMax: 1,
      valueFormat: pct,
    }),
  ];
  return `<div class="chart-grid">${charts.join("")}</div>
    <div class="detail-note">Charts use daily cohorts. Gaps mean no recorded data; they are not connected across missing days.
      Hover a point to see its value. Cost and token charts show averages per generation run, with the sample count in the underlying data.</div>`;
}

function renderRecent(data) {
  const rows = data.recent_runs || [];
  if (!rows.length) {
    return `<div class="card panel">${loadingState("No exam runs recorded for this period.")}</div>`;
  }
  const body = rows.map((run) => `
    <a class="recent-row" href="reports.html?exam_id=${encodeURIComponent(run.exam_id)}">
      <span class="recent-id">${escapeHtml(run.exam_id)}</span>
      <span>${escapeHtml(dateShort(run.created_at))}</span>
      <span class="recent-num">${num(run.final_valid)}/${num(run.requested)}</span>
      <span class="recent-num">${run.first_pass_rate === null ? "—" : pct(run.first_pass_rate)}</span>
      <span class="recent-num">${num(run.repairs_sent)}</span>
      <span class="recent-num">${run.recovery_entered ? `${num(run.recovery_recovered)}/${num(run.recovery_entered)}` : "—"}</span>
      <span class="recent-num">${run.llm?.estimated_cost_usd === null ? "—" : money(run.llm?.estimated_cost_usd)}</span>
      <span><span class="badge ${statusTone(run.status_key)}">${escapeHtml(run.status)}</span></span>
    </a>`).join("");
  return `
    <div class="card panel">
      <div class="recent-row recent-head">
        <span>Exam</span><span>Generated</span><span class="recent-num">Final / req.</span>
        <span class="recent-num">First pass</span><span class="recent-num">Repairs</span>
        <span class="recent-num">Recovery</span><span class="recent-num">Cost</span><span>Status</span>
      </div>
      ${body}
    </div>`;
}

function render() {
  const host = document.getElementById("pageBody");
  if (!host) return;
  const { data, loading, error } = state;
  if (loading && !data) {
    host.innerHTML = loadingState();
    return;
  }
  if (error) {
    host.innerHTML = error instanceof ApiError && error.isAuth
      ? authState()
      : errorState(error.message || "Unknown error");
    return;
  }
  if (!data) {
    host.innerHTML = loadingState();
    return;
  }
  host.innerHTML = `
    ${renderKpis(data)}
    ${section("Requests and jobs", "What was submitted, what completed, and what was produced", renderRequests(data))}
    ${section("Average usage", "Attributable provider usage per evaluated run and model exam", renderAverages(data))}
    ${section("Cost and telemetry attribution", "Every recorded request is counted in exactly one scope", renderAttribution(data))}
    ${section("Pipeline overview", "Question flow from request to finalized output", renderPipeline(data))}
    ${section("Performance trends", "Daily averages and rates; gaps are shown rather than connected", renderTrends(data))}
    ${section("Recent runs", "", renderRecent(data), `<a class="btn ghost" href="reports.html">All exam reports →</a>`)}`;
  setFreshness({
    latestRunAt: data.runs?.latest_run_at,
    fetchedAt: data.generated_at,
  });
}

async function load() {
  state.loading = true;
  state.error = null;
  if (!state.data) render();
  else setFreshness({ loading: true });
  try {
    state.data = await apiGet("/overview", {
      days: state.days,
      include_benchmarks: state.includeBenchmarks ? "true" : "false",
    });
  } catch (error) {
    state.error = error;
  } finally {
    state.loading = false;
    render();
  }
}

function bindControls() {
  const windowSelect = document.getElementById("windowSelect");
  const benchmarkToggle = document.getElementById("includeBenchmarks");
  if (windowSelect) {
    windowSelect.value = String(state.days);
    windowSelect.addEventListener("change", () => {
      state.days = Number(windowSelect.value);
      load();
    });
  }
  if (benchmarkToggle) {
    benchmarkToggle.addEventListener("change", () => {
      state.includeBenchmarks = benchmarkToggle.checked;
      load();
    });
  }
}

mountHeader("overview", { onRefresh: () => load() });
bindControls();
load();
