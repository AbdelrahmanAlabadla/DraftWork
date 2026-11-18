import { ApiError, apiGet } from "./api.js";
import {
  authState,
  badge,
  bars,
  bindPagination,
  coverageNote,
  emptyState,
  errorState,
  kpi,
  loadingState,
  meter,
  missingChip,
  mountHeader,
  notice,
  panel,
  pagination,
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

const state = {
  examId: new URLSearchParams(window.location.search).get("exam_id") || "",
  search: "",
  status: "",
  benchmark: "exclude",
  days: 30,
  limit: 20,
  offset: 0,
  list: null,
  detail: null,
  loading: false,
  error: null,
};

const ACTION_LABELS = {
  fix_answer: "Fix answer",
  fix_options: "Fix options",
  fix_question: "Fix question",
  fix_question_and_answer: "Fix question & answer",
  repair_question: "Repair question",
  regenerate_missing_slot: "Regenerate slot",
  switch_variation: "Switch variation",
  replace_task: "Replace task",
  other: "Other",
};

function fmtAction(value) {
  return ACTION_LABELS[String(value || "").toLowerCase()] || labelize(value);
}

function reasonBars(map, { tone = "", empty = "No recorded reasons" } = {}) {
  return bars(
    Object.entries(map || {})
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .map(([label, value]) => ({ label: labelize(label), value })),
    { tone, empty },
  );
}

function rateTone(rate, good = 0.9, warn = 0.75) {
  if (rate === null || rate === undefined) return "";
  return rate >= good ? "ok" : rate >= warn ? "" : rate >= 0.5 ? "warn" : "bad";
}

/* ------------------------------------------------------------------ list */

function controlsHtml() {
  return `
    <div class="report-controls">
      <input class="input" id="reportSearch" type="search" placeholder="Search exam ID" value="${escapeHtml(state.search)}" aria-label="Search exam ID" />
      <select class="select" id="reportStatus" aria-label="Status filter">
        <option value="">All statuses</option>
        <option value="healthy">Healthy</option>
        <option value="needs_attention">Needs attention</option>
        <option value="failed">Failed</option>
        <option value="cancelled">Cancelled</option>
      </select>
      <select class="select" id="reportBenchmark" aria-label="Benchmark filter">
        <option value="exclude">Exclude benchmarks</option>
        <option value="include">Include benchmarks</option>
        <option value="only">Benchmarks only</option>
      </select>
      <select class="select" id="reportDays" aria-label="Time window">
        <option value="7">Last 7 days</option>
        <option value="30">Last 30 days</option>
        <option value="90">Last 90 days</option>
        <option value="0">All recorded</option>
      </select>
    </div>`;
}

function bindControls() {
  const search = document.getElementById("reportSearch");
  const status = document.getElementById("reportStatus");
  const benchmark = document.getElementById("reportBenchmark");
  const days = document.getElementById("reportDays");
  if (search) {
    search.value = state.search;
    search.addEventListener("input", debounce(() => {
      state.search = search.value.trim();
      state.offset = 0;
      load();
    }, 350));
  }
  if (status) {
    status.value = state.status;
    status.addEventListener("change", () => {
      state.status = status.value;
      state.offset = 0;
      load();
    });
  }
  if (benchmark) {
    benchmark.value = state.benchmark;
    benchmark.addEventListener("change", () => {
      state.benchmark = benchmark.value;
      state.offset = 0;
      load();
    });
  }
  if (days) {
    days.value = String(state.days);
    days.addEventListener("change", () => {
      state.days = Number(days.value);
      state.offset = 0;
      load();
    });
  }
}

function examCard(item) {
  const tone = statusTone(item.status);
  const llm = item.llm || {};
  const llmKnown = Boolean(item.has_llm_telemetry);
  const finalRate = item.final_acceptance_rate;
  return `
    <a class="exam-card" href="reports.html?exam_id=${encodeURIComponent(item.exam_id)}">
      <div class="exam-card-head">
        <span class="exam-card-id" title="${escapeHtml(item.exam_id)}">${escapeHtml(item.exam_id)}</span>
        <span class="badge ${tone}">${escapeHtml(item.status_label || item.status)}</span>
      </div>
      <div class="exam-card-date">
        ${escapeHtml(dateTime(item.created_at))}
        ${item.benchmark_id ? " · benchmark" : ""}
        ${item.models_count > 1 ? ` · ${num(item.models_count)} model versions` : ""}
      </div>
      <div class="exam-card-stats">
        <span class="exam-card-stat">Requested <strong>${num(item.requested)}</strong></span>
        <span class="exam-card-stat">Finalized <strong>${num(item.final_valid)}</strong></span>
        <span class="exam-card-stat">First pass <strong>${item.first_pass_rate === null ? "—" : pct(item.first_pass_rate)}</strong></span>
        <span class="exam-card-stat">Cost <strong>${money(llm.estimated_cost_usd)}</strong></span>
        <span class="exam-card-stat" title="Estimated provider cost divided by finalized questions">Est. cost / final Q <strong>${money(item.cost_per_finalized_question_usd)}</strong></span>
        <span class="exam-card-stat">Duration <strong>${duration(item.duration_ms)}</strong></span>
        <span class="exam-card-stat">Calls <strong>${llmKnown ? num(llm.calls) : "—"}</strong></span>
        <span class="exam-card-stat">Failed <strong>${llmKnown ? num(llm.failed_calls) : "—"}</strong></span>
      </div>
      ${finalRate === null ? "" : meter(finalRate, finalRate >= 0.95 ? "ok" : finalRate >= 0.85 ? "warn" : "bad")}
      <div class="small mt8 row">
        ${item.failure_code ? `<span class="badge bad">${escapeHtml(item.failure_code)}</span>` : ""}
        ${llmKnown ? "" : `<span class="chip" title="This run predates provider request telemetry">No provider telemetry</span>`}
        ${item.has_recovery_telemetry ? "" : `<span class="chip" title="This run predates per-run recovery outcome telemetry">No recovery telemetry</span>`}
      </div>
    </a>`;
}

function renderList() {
  const body = document.getElementById("pageBody");
  const controls = document.getElementById("reportsControls");
  if (controls) controls.innerHTML = controlsHtml();
  const data = state.list;
  if (!data) {
    body.innerHTML = loadingState();
    return;
  }
  if (!data.items.length) {
    body.innerHTML = emptyState(
      "No exam reports match these filters",
      "Try widening the period, clearing the search, or including benchmarks.",
    );
    bindControls();
    return;
  }
  body.innerHTML = `
    <div class="report-grid">${data.items.map(examCard).join("")}</div>
    <div id="reportPager">${pagination({ total: data.total, limit: data.limit, offset: data.offset })}</div>`;
  bindControls();
  bindPagination(document.getElementById("reportPager"), {
    limit: data.limit,
    offset: data.offset,
    total: data.total,
    onPage: (offset) => {
      state.offset = offset;
      load();
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
  });
  setFreshness({
    latestRunAt: data.items[0]?.created_at || null,
    fetchedAt: new Date().toISOString(),
  });
}

/* ---------------------------------------------------------------- detail */

function detailNav() {
  return `
    <div class="subnav">
      <a href="#summary">Summary</a>
      <a href="#llm">LLM usage</a>
      <a href="#stages">Stages</a>
      <a href="#quality">Quality & recovery</a>
      <a href="#telemetry">Telemetry gaps</a>
    </div>`;
}

function qtypeTable(report) {
  const types = report.question_types || {};
  const rows = Object.entries(types)
    .sort((a, b) => Number(b[1].requested_questions || 0) - Number(a[1].requested_questions || 0))
    .map(([qtype, bucket]) => {
      const rates = bucket.rates || {};
      return [
        `<span class="strong">${escapeHtml(labelize(qtype))}</span>`,
        `<span class="nowrap">${num(bucket.requested_questions)}</span>`,
        `<span class="nowrap">${num(bucket.generated_first)}</span>`,
        `<span class="nowrap">${rates.first_validation_pass_rate === null ? "—" : pct(rates.first_validation_pass_rate)}</span>`,
        `<span class="nowrap">${rates.validation_coverage_rate === null ? "—" : pct(rates.validation_coverage_rate)}</span>`,
        `<span class="nowrap">${num(bucket.repair_sent)} / ${num(bucket.repair_succeeded)}</span>`,
        `<span class="nowrap">${num(bucket.final_valid)}</span>`,
        `<span class="nowrap">${rates.final_success_rate === null ? "—" : pct(rates.final_success_rate)}</span>`,
        `<span class="nowrap">${num(bucket.generation_rejected)}</span>`,
      ];
    });
  return table(
    ["Type", "Requested", "Generated", "First pass", "Coverage", "Repairs sent / ok", "Final", "Final rate", "Rejected"],
    rows,
    { empty: "No question-type telemetry for this run." },
  );
}

function modelTable(report) {
  const models = report.models || {};
  const rows = Object.entries(models)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([number, bucket]) => {
      const rates = bucket.rates || {};
      return [
        `<span class="strong">Model ${escapeHtml(number)}</span>`,
        `<span class="nowrap">${num(bucket.requested_questions)}</span>`,
        `<span class="nowrap">${rates.first_validation_pass_rate === null ? "—" : pct(rates.first_validation_pass_rate)}</span>`,
        `<span class="nowrap">${rates.repair_success_rate === null ? "—" : pct(rates.repair_success_rate)}</span>`,
        `<span class="nowrap">${num(bucket.final_valid)}</span>`,
        `<span class="nowrap">${rates.final_success_rate === null ? "—" : pct(rates.final_success_rate)}</span>`,
      ];
    });
  return table(["Version", "Requested", "First pass", "Repair", "Final", "Final rate"], rows, {
    empty: "No per-model telemetry.",
  });
}

function stageTable(report) {
  const stages = report.stages || [];
  const totalWall = stages.reduce((total, stage) => total + Number(stage.wall_ms || 0), 0);
  const rows = stages.map((stage) => {
    const llm = stage.llm || {};
    const share = totalWall && stage.wall_ms ? (Number(stage.wall_ms) / totalWall) * 100 : 0;
    return [
      `<span class="strong">${escapeHtml(labelize(stage.stage))}</span>`,
      `<span class="nowrap">${duration(stage.wall_ms)}</span>
       <div class="timeline-bar" title="Share of recorded stage time"><span style="width:${share.toFixed(1)}%"></span></div>`,
      `<span class="nowrap">${duration(llm.call_time_ms)}</span>`,
      `<span class="nowrap">${num(llm.calls)}</span>`,
      `<span class="nowrap">${num(llm.failed)}</span>`,
      `<span class="nowrap">${num(llm.retries)}</span>`,
      `<span class="nowrap">${compact(llm.input_tokens)} / ${compact(llm.output_tokens)} / ${compact(llm.reasoning_tokens)} / ${compact(llm.cached_tokens)}</span>`,
      `<span class="nowrap">${money(llm.estimated_cost_usd)}</span>`,
      `<span class="nowrap">${duration(llm.latency?.avg_ms)} / ${duration(llm.latency?.p95_ms)}</span>`,
    ];
  });
  return table(
    [
      "Stage",
      { label: "Recorded wall time", className: "" },
      { label: "Σ call time", className: "" },
      "Calls",
      "Failed",
      "Retries",
      "In / out / reasoning / cached",
      "Est. cost",
      "Avg / p95 call",
    ],
    rows,
    { empty: "No stage timeline or provider telemetry for this run.", minWidth: "980px" },
  );
}

function llmTotals(report) {
  const totals = report.llm?.totals || {};
  const latency = totals.latency || {};
  return `
    <div class="grid grid-4 detail-kpis">
      ${kpi({ label: "Input tokens", value: compact(totals.input_tokens), detail: coverageNote(totals.usage_calls, totals.calls) })}
      ${kpi({ label: "Output tokens", value: compact(totals.output_tokens) })}
      ${kpi({ label: "Reasoning tokens", value: compact(totals.reasoning_tokens) })}
      ${kpi({ label: "Cached tokens", value: compact(totals.cached_tokens) })}
      ${kpi({ label: "Estimated cost", value: money(totals.estimated_cost_usd), detail: "Estimated, not billed" })}
      ${kpi({ label: "Successful calls", value: num(totals.successful), detail: `${num(totals.calls)} total` })}
      ${kpi({ label: "Failed calls", value: num(totals.failed), tone: totals.failed ? "bad" : "" })}
      ${kpi({ label: "Retries", value: num(totals.retries) })}
      ${kpi({ label: "Average call latency", value: duration(latency.avg_ms) })}
      ${kpi({ label: "P95 call latency", value: duration(latency.p95_ms) })}
    </div>`;
}

function providerTable(report) {
  const rows = (report.llm?.by_provider_model || []).map((group) => [
    `<span class="mono">${escapeHtml(group.key)}</span>`,
    num(group.calls),
    num(group.failed),
    num(group.retries),
    `${compact(group.input_tokens)} / ${compact(group.output_tokens)}`,
    money(group.estimated_cost_usd),
    duration(group.latency?.avg_ms),
  ]);
  return table(["Provider / model", "Calls", "Failed", "Retries", "Input / output", "Est. cost", "Avg latency"], rows, {
    empty: "No provider telemetry for this run.",
  });
}

function recoverySection(report) {
  const recovery = report.recovery || {};
  if (!recovery || !Object.keys(recovery).length) {
    return notice("Recovery outcome telemetry is not recorded for this run (it predates per-run outcome persistence).");
  }
  const strategies = recovery.attempt_strategies || {};
  const outcomes = recovery.attempt_strategy_outcomes || {};
  const strategyRows = Object.entries(strategies)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([strategy, count]) => {
      const byOutcome = outcomes[strategy] || {};
      const accepted = Number(byOutcome.accepted || 0);
      const rate = count ? accepted / count : null;
      const failed = Number(byOutcome.rejected || 0)
        + Number(byOutcome.candidate_generation_failed || 0)
        + Number(byOutcome.strategy_skipped || 0);
      return [
        `<span class="strong">${escapeHtml(strategy)}</span>`,
        num(count),
        num(accepted),
        rate === null ? "—" : pct(rate),
        num(failed),
      ];
    });
  return `
    <div class="grid grid-4">
      ${stat({ label: "Slots", value: num(recovery.slots), detail: "Requested slot slots in this run" })}
      ${stat({ label: "Entered recovery", value: num(recovery.entered_recovery) })}
      ${stat({ label: "Recovered", value: num(recovery.recovered), detail: "finalized after recovery" })}
      ${stat({ label: "Unresolved", value: num(recovery.unresolved), detail: "entered recovery, not finalized" })}
    </div>
    <div class="detail-grid mt12">
      ${panel({
        title: "Recovery V3 strategies",
        subtitle: "Attempt-level outcomes, canonical strategy identifiers",
        body: table(["Strategy", "Attempts", "Accepted", "Rate", "Unsuccessful"], strategyRows, {
          empty: "No repair attempts recorded for this run.",
        }),
      })}
      ${panel({
        title: "Slot terminal status",
        subtitle: "Authoritative lifecycle state at finalization",
        body: bars(Object.entries(recovery.terminal_status || {}).map(([label, value]) => ({ label: labelize(label), value }))),
      })}
    </div>
    <div class="detail-grid mt12">
      ${panel({ title: "Attempt outcomes", body: bars(Object.entries(recovery.attempt_outcomes || {}).map(([label, value]) => ({ label: labelize(label), value }))) })}
      ${panel({ title: "Slot failure codes", body: reasonBars(recovery.slot_failure_codes, { empty: "No unresolved slots" }) })}
    </div>`;
}

function duplicatesSection(report) {
  const duplicates = report.duplicates || {};
  const cache = report.catalog_cache || {};
  if (!Object.keys(duplicates).length && !Object.keys(cache).length) {
    return notice("Duplicate and catalog-cache telemetry is not recorded for this run.");
  }
  const cacheBadges = Object.entries(cache).map(([key, value]) => {
    const name = key.replace(/_hit$/, "").replaceAll("_", " ");
    return badge(`${name}: ${value ? "hit" : "miss"}`, value ? "ok" : "neutral");
  }).join(" ");
  return `
    <div class="grid grid-4">
      ${stat({ label: "Duplicate repair rounds", value: num(duplicates.repair_rounds) })}
      ${stat({ label: "Replacement attempts", value: num(duplicates.replacement_attempts) })}
      ${stat({ label: "Slots replaced", value: num(duplicates.slots_replaced) })}
      ${stat({ label: "Unresolved duplicate groups", value: num(duplicates.unresolved_conflicts), tone: duplicates.unresolved_conflicts ? "bad" : "" })}
    </div>
    <div class="detail-note">Catalog cache: ${Object.keys(cache).length ? cacheBadges : "not recorded for this run"}.</div>`;
}

function missingSection(report) {
  const missing = report.missing || {};
  const labels = {
    llm: "Provider token/cost telemetry",
    stages: "Recorded stage timeline",
    recovery: "Recovery outcomes",
    duplicates: "Duplicate outcomes",
    catalog_cache: "Catalog cache flags",
    duration: "Generation duration",
    job: "Linked job record",
  };
  const gaps = Object.entries(labels).filter(([key]) => missing[key]);
  if (!gaps.length) {
    return notice("All telemetry layers are available for this run.", "");
  }
  return notice(
    `<strong>Unavailable for this run:</strong> ${gaps.map(([, label]) => escapeHtml(label)).join(", ")}.
     Historical runs predate some telemetry; values are shown as “—” rather than estimated.`,
    "warn",
  );
}

function renderDetail() {
  const body = document.getElementById("pageBody");
  const report = state.detail;
  const controls = document.getElementById("reportsControls");
  if (controls) controls.innerHTML = "";
  const title = document.getElementById("reportsTitle");
  if (title) {
    title.innerHTML = `
      <div class="detail-head">
        <div class="detail-title">
          <h2>${escapeHtml(report.exam_id)}</h2>
          <div class="detail-meta">
            <span>${escapeHtml(dateTime(report.created_at))}</span>
            ${report.job_id ? `<span class="mono">job ${escapeHtml(report.job_id)}</span>` : `<span>${missingChip("no job link")}</span>`}
            <span>${num(report.models_count)} model version(s)</span>
            ${report.benchmark_id ? badge("Benchmark", "accent") : ""}
          </div>
        </div>
        <div class="row">
          <a class="btn ghost" href="reports.html">← All reports</a>
          ${badge(report.status_label || report.status, statusTone(report.status))}
        </div>
      </div>
      ${detailNav()}`;
  }
  const summary = report.summary || {};
  const statusToneKey = statusTone(report.status);
  body.innerHTML = `
    ${report.failure_code ? notice(`This run ended as <strong>${escapeHtml(report.status_label)}</strong>: ${escapeHtml(report.failure_code)}${report.failure_message ? ` — ${escapeHtml(report.failure_message)}` : ""}`, "bad") : ""}
    <section id="summary" class="section">
      <div class="grid grid-4 detail-kpis">
        ${kpi({ label: "Requested", value: num(summary.requested), detail: `${num(summary.generated_first)} generated · ${num(summary.final_valid)} finalized` })}
        ${kpi({ label: "Finalized", value: num(summary.final_valid), detail: `${num(summary.final_invalid)} invalid · ${num(summary.final_unvalidated)} unvalidated · ${num(summary.final_missing)} missing`, tone: statusToneKey })}
        ${kpi({ label: "First-pass acceptance", value: summary.first_pass_rate === null ? "—" : pct(summary.first_pass_rate), detail: `Coverage ${pct(summary.validation_coverage)}`, tone: rateTone(summary.first_pass_rate, 0.85, 0.7) })}
        ${kpi({ label: "Final acceptance", value: summary.final_acceptance_rate === null ? "—" : pct(summary.final_acceptance_rate), detail: `${num(summary.final_invalid)} invalid · ${num(summary.final_unvalidated)} unvalidated · ${num(summary.final_missing)} missing`, tone: rateTone(summary.final_acceptance_rate, 0.95, 0.85) })}
        ${kpi({ label: "Repair success", value: summary.repair_success_rate === null ? "—" : pct(summary.repair_success_rate), detail: `${num(summary.repair_succeeded)} / ${num(summary.repair_sent)}`, tone: rateTone(summary.repair_success_rate, 0.8, 0.6) })}
        ${kpi({ label: "Estimated cost", value: money(summary.estimated_cost_usd), detail: "Estimated, not billed" })}
        ${kpi({ label: "Estimated cost per finalized question", value: money(summary.cost_per_finalized_question_usd), detail: "Attributable estimated cost / finalized questions", hint: "Estimated provider cost for this run divided by its finalized questions. It is an estimate from configured token prices, not billed cost." })}
        ${kpi({ label: "Generation duration", value: duration(summary.duration_ms) })}
      </div>
    </section>
    ${section("Question types", "Counts and acceptance by question type", qtypeTable(report))}
    ${section("Model versions", "One generation run can contain several model exam versions", modelTable(report))}
    <section id="llm" class="section">
      <div class="section-head"><div><h2>LLM usage</h2><p>Persisted provider requests for this run</p></div></div>
      ${llmTotals(report)}
      <div class="mt12">${panel({ title: "Provider & model", body: providerTable(report) })}</div>
    </section>
    <section id="stages" class="section">
      <div class="section-head"><div><h2>Pipeline stage breakdown</h2><p>Wall time from the durable audit timeline; call time from provider events</p></div></div>
      ${panel({
        title: "Stages",
        body: stageTable(report),
        subtitle: `Wall time is recorded stage time (segments of the job timeline, including repeated validation/repair cycles).
          “Σ call time” sums individual provider request durations and can exceed wall time when calls run in parallel; it is not stage duration.`,
      })}
    </section>
    <section id="quality" class="section">
      <div class="section-head"><div><h2>Question quality & recovery</h2><p>Validator, repair and duplicate outcomes</p></div></div>
      <div class="detail-grid">
        ${panel({ title: "Validator rejection reasons", body: reasonBars(report.evaluation?.validation_failure_reasons, { empty: "No rejection reasons recorded" }) })}
        ${panel({ title: "Validator repair actions", body: reasonBars(report.evaluation?.validation_failure_actions, { empty: "Not recorded for this historical run" }) })}
      </div>
      ${Object.keys(report.evaluation?.validation_failure_actions || {}).length ? "" : notice("This run predates validator repair-action telemetry; the attempt-level actions and strategies below are reconstructed from its repair history.", "warn")}
      <div class="detail-grid mt12">
        ${panel({ title: "Validator operational failures", body: reasonBars(report.evaluation?.validator_failure_reasons, { tone: "bad", empty: "No operational failures recorded" }) })}
        ${panel({ title: "Generation rejections", body: reasonBars(report.evaluation?.generation_rejection_reasons, { tone: "warn", empty: "No generation rejections recorded" }) })}
      </div>
      <div class="mt12">${panel({ title: "Recovery outcomes", body: recoverySection(report) })}</div>
      <div class="mt12">${panel({ title: "Duplicates & catalog cache", body: duplicatesSection(report) })}</div>
    </section>
    <section id="telemetry" class="section">
      <div class="section-head"><div><h2>Telemetry coverage</h2><p>What was actually recorded for this run</p></div></div>
      ${missingSection(report)}
    </section>`;
  setFreshness({ latestRunAt: report.created_at, fetchedAt: new Date().toISOString() });
}

/* ------------------------------------------------------------------ load */

function render() {
  const body = document.getElementById("pageBody");
  if (state.loading && !state.list && !state.detail) {
    body.innerHTML = loadingState();
    return;
  }
  if (state.error) {
    body.innerHTML = state.error instanceof ApiError && state.error.isAuth
      ? authState()
      : errorState(state.error.message || "Unknown error");
    return;
  }
  if (state.examId) {
    if (state.detail) renderDetail();
  } else {
    renderList();
  }
}

async function load() {
  state.loading = true;
  state.error = null;
  render();
  try {
    if (state.examId) {
      state.detail = await apiGet(`/exams/${encodeURIComponent(state.examId)}`);
    } else {
      state.list = await apiGet("/exams", {
        search: state.search,
        status: state.status,
        benchmark: state.benchmark,
        days: state.days,
        limit: state.limit,
        offset: state.offset,
      });
    }
  } catch (error) {
    state.error = error;
    state.detail = null;
    state.list = null;
  } finally {
    state.loading = false;
    render();
  }
}

mountHeader("reports", { onRefresh: () => load() });
load();
