import { ApiError, apiGet } from "./api.js";
import {
  authState,
  bars,
  coverageNote,
  emptyState,
  errorState,
  kpi,
  loadingState,
  meter,
  mountHeader,
  notice,
  panel,
  section,
  setFreshness,
  stat,
  table,
} from "./components.js";
import { COLORS, stackedBar } from "./charts.js";
import {
  compact,
  duration,
  escapeHtml,
  labelize,
  money,
  num,
  pct,
} from "./utils.js";

const state = { days: 30, includeBenchmarks: false, data: null, loading: false, error: null };

function rateTone(rate, good = 0.9, warn = 0.75) {
  if (rate === null || rate === undefined) return "";
  return rate >= good ? "ok" : rate >= warn ? "" : rate >= 0.5 ? "warn" : "bad";
}

function coverageLineFor(data) {
  const coverage = data.coverage || {};
  return `${num(coverage.runs)} run(s) · ${num(coverage.job_linked_runs)} linked to a job · `
    + `${num(coverage.recovery_runs)} with recovery telemetry · `
    + `${num(coverage.duplicate_runs)} with duplicate telemetry · ${num(coverage.cache_runs)} with cache telemetry`;
}

function entries(map) {
  return Object.entries(map || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([label, value]) => ({ label: labelize(label), value }));
}

function sortedTypes(data) {
  return Object.entries(data.question_types || {}).sort(
    (a, b) => Number(b[1].requested_questions || 0) - Number(a[1].requested_questions || 0),
  );
}

function topReasons(bucket, limit = 2) {
  return Object.entries(bucket.validation_failure_reasons || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, limit)
    .map(([reason, count]) => `${labelize(reason)} (${num(count)})`)
    .join(" · ");
}

function qtypeCards(data) {
  const types = sortedTypes(data);
  if (!types.length) return emptyState("No question-type data for this period");
  return `
    <div class="qtype-grid">
      ${types.map(([qtype, bucket]) => {
        const rates = bucket.rates || {};
        const finalRate = rates.final_success_rate;
        return `
          <article class="qtype-card">
            <div class="qtype-card-head">
              <h4>${escapeHtml(labelize(qtype))}</h4>
              <span class="badge ${rateTone(finalRate, 0.95, 0.85)}">${finalRate === null ? "—" : pct(finalRate)} final</span>
            </div>
            <div class="qtype-counts">
              ${num(bucket.requested_questions)} requested
              · ${num(bucket.generated_first)} generated
              · ${num(bucket.final_valid)} finalized
            </div>
            ${finalRate === null ? "" : meter(finalRate, rateTone(finalRate, 0.95, 0.85) || "ok")}
            <div class="qtype-metrics">
              <div><span>First pass</span><strong>${rates.first_validation_pass_rate === null ? "—" : pct(rates.first_validation_pass_rate)}</strong></div>
              <div><span>Coverage</span><strong>${rates.validation_coverage_rate === null ? "—" : pct(rates.validation_coverage_rate)}</strong></div>
              <div><span>Repair success</span><strong>${rates.repair_success_rate === null ? "—" : pct(rates.repair_success_rate)}</strong></div>
              <div><span>Gen. rejected</span><strong>${num(bucket.generation_rejected)}</strong></div>
            </div>
            <div class="qtype-reasons">
              <span>Top rejection</span>
              ${escapeHtml(topReasons(bucket) || "None recorded")}
            </div>
          </article>`;
      }).join("")}
    </div>`;
}

function qtypeRateCharts(data) {
  const types = sortedTypes(data);
  if (!types.length) return "";
  const firstPass = types
    .map(([qtype, bucket]) => ({
      label: labelize(qtype),
      value: bucket.rates?.first_validation_pass_rate,
    }))
    .filter((item) => item.value !== null && item.value !== undefined);
  const finalPass = types
    .map(([qtype, bucket]) => ({
      label: labelize(qtype),
      value: bucket.rates?.final_success_rate,
    }))
    .filter((item) => item.value !== null && item.value !== undefined);
  return `
    <div class="split mt12">
      ${panel({
        title: "First-pass acceptance by question type",
        subtitle: "Share of validated questions accepted without repair",
        body: bars(firstPass, {
          format: (value) => pct(value),
          max: 1,
          empty: "No first-pass data",
        }),
      })}
      ${panel({
        title: "Final acceptance by question type",
        subtitle: "Share of requested questions finalized",
        body: bars(finalPass, {
          format: (value) => pct(value),
          max: 1,
          empty: "No final acceptance data",
        }),
      })}
    </div>`;
}

function qtypeRows(data) {
  return sortedTypes(data).map(([qtype, bucket]) => {
    const rates = bucket.rates || {};
    return [
      `<span class="strong">${escapeHtml(labelize(qtype))}</span>`,
      num(bucket.requested_questions),
      num(bucket.generated_first),
      num(bucket.final_valid),
      rates.first_validation_pass_rate === null ? "—" : pct(rates.first_validation_pass_rate),
      rates.validation_coverage_rate === null ? "—" : pct(rates.validation_coverage_rate),
      rates.final_success_rate === null ? "—" : pct(rates.final_success_rate),
      rates.repair_success_rate === null ? "—" : pct(rates.repair_success_rate),
      topReasons(bucket) || "—",
    ];
  });
}

function strategyRows(strategies, outcomes) {
  return Object.entries(strategies || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([strategy, count]) => {
      const byOutcome = outcomes?.[strategy] || {};
      const accepted = Number(byOutcome.accepted || 0);
      const unsuccessful = Object.entries(byOutcome)
        .filter(([outcome]) => outcome !== "accepted")
        .reduce((total, [, value]) => total + Number(value || 0), 0);
      const rate = count ? accepted / count : null;
      return [
        `<span class="strong">${escapeHtml(strategy)}</span>`,
        num(count),
        num(accepted),
        rate === null ? "—" : pct(rate),
        num(unsuccessful),
      ];
    });
}

function validationSection(data) {
  const validation = data.validation || {};
  const totalRejections = Object.values(validation.rejection_reasons || {}).reduce(
    (total, value) => total + Number(value || 0), 0,
  );
  const operational = Object.values(validation.operational_failures || {}).reduce(
    (total, value) => total + Number(value || 0), 0,
  );
  const generationRejections = Object.values(validation.generation_rejections || {}).reduce(
    (total, value) => total + Number(value || 0), 0,
  );
  return `
    <div class="stat-strip">
      ${stat({ label: "Validated questions", value: num(validation.validated_questions) })}
      ${stat({ label: "First-pass accepted", value: num(validation.passed_questions), detail: validation.first_pass_rate === null ? "" : pct(validation.first_pass_rate) })}
      ${stat({ label: "Rejected verdicts", value: num(totalRejections), detail: "questions needing repair" })}
      ${stat({ label: "Operational failures", value: num(operational), detail: "blocked, input-invalid or errored" })}
      ${stat({ label: "Generation rejections", value: num(generationRejections), detail: "candidates rejected before validation" })}
    </div>
    <div class="split mt12">
      ${panel({ title: "Validator rejection codes", subtitle: "Why questions failed validation", body: bars(entries(validation.rejection_reasons), { empty: "No rejection codes recorded" }) })}
      ${panel({
        title: "Repair actions requested",
        subtitle: "What the validator asked to fix",
        body: Object.keys(validation.repair_actions || {}).length
          ? bars(entries(validation.repair_actions), {})
          : `<div class="bars-empty">Validator repair-action telemetry was not recorded for these runs. Attempt-level actions are shown in the Repair section.</div>`,
      })}
    </div>
    <div class="split mt12">
      ${panel({ title: "Validator operational failures", subtitle: "Validation could not complete", body: bars(entries(validation.operational_failures), { tone: "bad", empty: "No operational failures recorded" }) })}
      ${panel({ title: "Generation rejections", subtitle: "Candidates rejected before validation", body: bars(entries(validation.generation_rejections), { tone: "warn", empty: "No generation rejections recorded" }) })}
    </div>
    <div class="pipeline-note">Historical runs recorded coarse repair-action categories; runs from this release record diagnostic failure codes and operational classes. A zero here means no rejection was recorded for these runs.</div>`;
}

function repairSection(data) {
  const repair = data.repair || {};
  const attempts = Number(repair.attempts || 0);
  const canonicalRows = strategyRows(repair.strategies, repair.strategy_outcomes);
  const rawRows = strategyRows(repair.strategy_details, repair.strategy_detail_outcomes);
  return `
    <div class="stat-strip">
      ${stat({ label: "Repair submissions", value: num(repair.question_repairs_sent), detail: "unique questions" })}
      ${stat({ label: "Question repair success", value: repair.question_repair_success_rate === null ? "—" : pct(repair.question_repair_success_rate), detail: `${num(repair.question_repairs_succeeded)} repaired` })}
      ${stat({ label: "Repair attempts", value: num(attempts), detail: attempts ? "recorded attempts" : "no attempt telemetry" })}
      ${stat({ label: "Attempt success", value: repair.success_rate === null ? "—" : pct(repair.success_rate), detail: `${num(repair.accepted)} accepted` })}
      ${stat({ label: "Strategies used", value: num(Object.keys(repair.strategies || {}).length) })}
    </div>
    ${attempts ? "" : notice("No attempt-level repair records exist for the selected runs. Older audits may not contain per-attempt events.", "warn")}
    <div class="split mt12">
      ${panel({
        title: "Repair outcomes by strategy",
        subtitle: "Canonical Recovery V3 dispatch strategies",
        body: table(["Strategy", "Attempts", "Accepted", "Acceptance", "Unsuccessful"], canonicalRows, {
          empty: "No repair attempts recorded",
        }),
      })}
      ${panel({
        title: "Repair actions",
        subtitle: "Recorded action per attempt",
        body: bars(entries(repair.actions), { empty: "No attempt telemetry" }),
      })}
    </div>
    <div class="split mt12">
      ${panel({ title: "Attempt outcomes", subtitle: "Top 12 statuses by count", body: bars(entries(repair.outcomes).slice(0, 12), { empty: "No attempt telemetry" }) })}
      ${panel({ title: "Unsuccessful attempt codes", subtitle: "Top 12 codes by count", body: bars(entries(repair.failure_codes).slice(0, 12), { tone: "bad", empty: "No failed attempts recorded" }) })}
    </div>
    <div class="subsection">
      <h4>Raw strategy descriptors</h4>
      ${table(["Recorded strategy", "Attempts", "Accepted", "Acceptance", "Unsuccessful"], rawRows, {
        empty: "No strategy descriptors recorded",
      })}
    </div>`;
}

function strategyTable(data) {
  const repair = data.repair || {};
  return table(
    ["Strategy", "Attempts", "Accepted", "Acceptance", "Unsuccessful"],
    strategyRows(repair.strategies, repair.strategy_outcomes),
    { empty: "No repair attempts recorded" },
  );
}

function recoverySection(data) {
  const recovery = data.recovery || {};
  const known = Number(recovery.slots || 0) > 0;
  return `
    ${known ? "" : notice("No slot-level recovery telemetry is stored for the selected runs.", "warn")}
    <div class="stat-strip">
      ${stat({ label: "Slots observed", value: num(recovery.slots) })}
      ${stat({ label: "Entered recovery", value: num(recovery.entered) })}
      ${stat({ label: "Recovered", value: num(recovery.recovered), detail: "finalized after recovery" })}
      ${stat({ label: "Unresolved", value: num(recovery.unresolved), detail: "entered recovery, not finalized" })}
      ${stat({ label: "Recovery success", value: recovery.success_rate === null ? "—" : pct(recovery.success_rate) })}
    </div>
    <div class="split mt12">
      ${panel({ title: "Recovery V3 strategy outcomes", subtitle: "Canonical dispatch strategies", body: strategyTable(data) })}
      ${panel({ title: "Slot terminal status", body: bars(entries(recovery.terminal_status), { empty: "No slot outcomes recorded" }) })}
    </div>
    <div class="split mt12">
      ${panel({ title: "Failure stages", body: bars(entries(recovery.failure_stages), { tone: "warn" }) })}
      ${panel({ title: "Failure codes", subtitle: "Top 12 codes by count", body: bars(entries(recovery.failure_codes).slice(0, 12), { tone: "bad" }) })}
    </div>`;
}

function lifecycleSection(data) {
  const recovery = data.recovery || {};
  const duplicates = data.duplicates || {};
  const cache = data.catalog_cache || {};
  const coverage = data.coverage || {};
  const terminal = entries(recovery.terminal_status);
  const duplicateRuns = Number(coverage.duplicate_runs || 0);
  const cacheRuns = Number(coverage.cache_runs || 0);
  return `
    <div class="split">
      ${panel({
        title: "Final slot distribution",
        subtitle: `${num(recovery.slots)} slots observed`,
        body: terminal.length
          ? stackedBar(terminal.map((entry, index) => ({
            label: entry.label,
            value: entry.value,
            color: entry.label.toLowerCase() === "finalized"
              ? COLORS.success
              : [COLORS.danger, COLORS.warning, COLORS.info][index % 3],
          })))
          : `<div class="bars-empty">No lifecycle telemetry recorded</div>`,
      })}
      ${panel({
        title: "Duplicates",
        subtitle: duplicateRuns
          ? `Recorded in ${num(duplicateRuns)} run(s)`
          : "No duplicate telemetry for these runs",
        body: `
          <div class="grid grid-2">
            ${stat({ label: "Repair rounds", value: num(duplicates.repair_rounds) })}
            ${stat({ label: "Replacement attempts", value: num(duplicates.replacement_attempts) })}
            ${stat({ label: "Slots replaced", value: num(duplicates.slots_replaced) })}
            ${stat({ label: "Unresolved conflicts", value: num(duplicates.unresolved_conflicts), tone: duplicates.unresolved_conflicts ? "bad" : "" })}
          </div>
          <div class="detail-note">${duplicateRuns
            ? "A zero means no duplicate repair was needed or recorded for these runs."
            : "These runs predate duplicate outcome persistence."}</div>`,
      })}
    </div>
    <div class="mt12">
      ${panel({
        title: "Assessment catalog cache",
        subtitle: cacheRuns
          ? `Across ${num(cacheRuns)} run(s) with cache telemetry`
          : "No cache telemetry for these runs",
        body: `
          <div class="grid grid-3">
            ${stat({ label: "Catalog cache hit rate", value: cache.catalog_hit_rate === null ? "—" : pct(cache.catalog_hit_rate), detail: `${num(cache.catalog_hits)} of ${num(cacheRuns)} hits` })}
            ${stat({ label: "Compiler cache hit rate", value: cache.compiler_hit_rate === null ? "—" : pct(cache.compiler_hit_rate), detail: `${num(cache.compiler_hits)} of ${num(cacheRuns)} hits` })}
            ${stat({ label: "Compatibility cache hit rate", value: cache.compatibility_hit_rate === null ? "—" : pct(cache.compatibility_hit_rate), detail: `${num(cache.compatibility_hits)} of ${num(cacheRuns)} hits` })}
          </div>
          <div class="detail-note">Only runs with recorded cache flags are included in the denominator.</div>`,
      })}
    </div>`;
}

function llmGroupTable(groups, keyLabel) {
  const rows = (groups || []).map((group) => [
    `<span class="mono">${escapeHtml(group.key ?? group.stage ?? "")}</span>`,
    num(group.calls),
    num(group.failed),
    group.failure_rate === null ? "—" : pct(group.failure_rate),
    num(group.retries),
    `${compact(group.input_tokens)} / ${compact(group.output_tokens)} / ${compact(group.reasoning_tokens)} / ${compact(group.cached_tokens)}`,
    money(group.estimated_cost_usd),
    duration(group.latency?.avg_ms),
    duration(group.latency?.p95_ms),
  ]);
  return table(
    [keyLabel, "Calls", "Failed", "Failure rate", "Retries", "In / out / reasoning / cached", "Est. cost", "Avg latency", "P95"],
    rows,
    { empty: "No provider telemetry in this period.", minWidth: "900px" },
  );
}

function llmSection(data) {
  const llm = data.llm || {};
  const totals = llm.totals || {};
  const latency = totals.latency || {};
  return `
    <div class="grid grid-4">
      ${kpi({ label: "Provider calls", value: num(totals.calls), detail: `${num(totals.successful)} successful` })}
      ${kpi({ label: "Failed calls", value: num(totals.failed), detail: totals.failure_rate === null ? "" : pct(totals.failure_rate), tone: totals.failed ? "bad" : "" })}
      ${kpi({ label: "Retries", value: num(totals.retries) })}
      ${kpi({ label: "Estimated cost", value: money(totals.estimated_cost_usd), detail: "Estimated, not billed" })}
      ${kpi({ label: "Input tokens", value: compact(totals.input_tokens), detail: coverageNote(totals.usage_calls, totals.calls) })}
      ${kpi({ label: "Output tokens", value: compact(totals.output_tokens) })}
      ${kpi({ label: "Reasoning tokens", value: compact(totals.reasoning_tokens) })}
      ${kpi({ label: "Cached tokens", value: compact(totals.cached_tokens) })}
      ${kpi({ label: "Average call latency", value: duration(latency.avg_ms) })}
      ${kpi({ label: "P95 call latency", value: duration(latency.p95_ms) })}
    </div>
    <div class="split mt12">
      ${panel({ title: "Cost by pipeline stage", subtitle: "Unknown calls are legacy requests recorded before stage attribution", body: bars((llm.by_stage || []).map((group) => ({
        label: String(group.stage || "").toLowerCase() === "unknown" ? "Unattributed (legacy)" : labelize(group.stage),
        value: group.estimated_cost_usd || 0,
      })), { tone: "accent", format: (value) => money(value) }) })}
      ${panel({ title: "Output tokens by agent", body: bars((llm.by_agent || []).slice(0, 10).map((group) => ({ label: labelize(group.key), value: group.output_tokens })), { tone: "info", format: compact }) })}
    </div>
    <div class="pipeline-note">This section covers all recorded provider requests, including unattributed legacy calls. The Overview page separates attributable, other-generation and unattributed cost.</div>
    <div class="subsection"><h4>By pipeline stage</h4>${llmGroupTable(llm.by_stage, "Stage")}</div>
    <div class="subsection"><h4>By agent role</h4>${llmGroupTable(llm.by_agent, "Agent")}</div>
    <div class="subsection"><h4>By provider / model</h4>${llmGroupTable(llm.by_provider_model, "Provider / model")}</div>`;
}

function render() {
  const body = document.getElementById("pageBody");
  if (!body) return;
  if (state.loading && !state.data) {
    body.innerHTML = loadingState();
    return;
  }
  if (state.error) {
    body.innerHTML = state.error instanceof ApiError && state.error.isAuth
      ? authState()
      : errorState(state.error.message || "Unknown error");
    return;
  }
  if (!state.data) {
    body.innerHTML = loadingState();
    return;
  }
  const data = state.data;
  const coverage = data.coverage || {};
  body.innerHTML = `
    <div class="coverage-line">${escapeHtml(coverageLineFor(data))}</div>
    ${Number(coverage.recovery_runs || 0) < Number(coverage.runs || 0)
      ? notice(`Recovery/duplicate/cache analytics are based on the ${num(coverage.recovery_runs)} runs with outcome telemetry. Runs without it are excluded rather than counted as zero.`, "warn")
      : ""}
    ${section(
      "Question types",
      "Which question types produce reliably, and where they fail",
      `${qtypeCards(data)}
       ${qtypeRateCharts(data)}
       <div class="card panel mt12">
         <div class="panel-head"><div><h3>Detailed comparison</h3><p>Counts and acceptance by question type</p></div></div>
         ${table(
           ["Type", "Requested", "Generated", "Finalized", "First pass", "Coverage", "Final rate", "Repair success", "Top rejection reasons"],
           qtypeRows(data),
           { empty: "No question-type data for this period.", minWidth: "760px" },
         )}
       </div>`,
    )}
    ${section("Validation", "Verdicts, rejection reasons and operational failures", validationSection(data))}
    ${section("Repair", "Question-level and attempt-level repair effectiveness", repairSection(data))}
    ${section("Recovery V3", "Strategy usage, acceptance and unresolved slots", recoverySection(data))}
    ${section("Lifecycle & duplicates", "Where slots end and how duplicates resolve", lifecycleSection(data))}
    ${section("LLM & stage analytics", "Providers, agents and stages", llmSection(data))}`;
  setFreshness({
    latestRunAt: data.latest_run_at || null,
    fetchedAt: new Date().toISOString(),
  });
}

async function load() {
  state.loading = true;
  state.error = null;
  if (!state.data) render();
  try {
    state.data = await apiGet("/pipeline", {
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

mountHeader("pipeline", { onRefresh: () => load() });
bindControls();
load();
