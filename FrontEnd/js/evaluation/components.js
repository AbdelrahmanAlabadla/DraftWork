/** Shared shell and presentational components for the evaluation pages. */
import { dateTime, escapeHtml, num, pct } from "./utils.js";

const PAGES = [
  { id: "overview", href: "overview.html", label: "Overview" },
  { id: "reports", href: "reports.html", label: "Exam Reports" },
  { id: "pipeline", href: "pipeline.html", label: "Pipeline Analytics" },
  { id: "benchmarks", href: "benchmarks.html", label: "Benchmarks" },
];

export function mountHeader(active, { onRefresh } = {}) {
  const host = document.getElementById("dashHeader");
  if (!host) return;
  host.className = "dash-header";
  host.innerHTML = `
    <div class="dash-header-inner">
      <a class="brand" href="overview.html">Smart<span>Exam</span> AI <small>Evaluation</small></a>
      <nav class="dash-nav" aria-label="Evaluation pages">
        ${PAGES.map((page) => `
          <a href="${page.href}" ${page.id === active ? 'aria-current="page"' : ""}>${page.label}</a>
        `).join("")}
      </nav>
      <div class="dash-actions">
        <span class="freshness" id="dashFreshness">Loading telemetry…</span>
        <button class="btn ghost" id="dashRefresh" type="button" title="Reload metrics">Refresh ↻</button>
        <a class="btn ghost" href="/" title="Back to the exam generator">Generator</a>
      </div>
    </div>`;
  const refresh = host.querySelector("#dashRefresh");
  if (refresh && typeof onRefresh === "function") {
    refresh.addEventListener("click", () => onRefresh());
  }
}

export function setFreshness({ latestRunAt, fetchedAt = null, loading = false } = {}) {
  const node = document.getElementById("dashFreshness");
  if (!node) return;
  if (loading) {
    node.textContent = "Refreshing…";
    return;
  }
  const parts = [];
  if (latestRunAt !== undefined) {
    parts.push(latestRunAt ? `Data through ${dateTime(latestRunAt)}` : "No evaluation runs recorded");
  }
  if (fetchedAt) parts.push(`fetched ${dateTime(fetchedAt)}`);
  node.textContent = parts.join(" · ") || "Live telemetry";
}

export function kpi({ label, value, detail = "", tone = "", hint = "" }) {
  return `
    <article class="kpi ${tone}">
      <div class="kpi-label">${escapeHtml(label)}${hint ? tip(hint) : ""}</div>
      <div class="kpi-value">${escapeHtml(value)}</div>
      ${detail ? `<div class="kpi-detail">${detail}</div>` : ""}
    </article>`;
}

export function stat({ label, value, detail = "", hint = "" }) {
  return `
    <div class="stat">
      <div class="stat-label">${escapeHtml(label)}${hint ? tip(hint) : ""}</div>
      <div class="stat-value">${escapeHtml(value)}</div>
      ${detail ? `<div class="stat-detail">${detail}</div>` : ""}
    </div>`;
}

export function tip(text) {
  return `<span class="tip" tabindex="0" role="note" data-tip="${escapeHtml(text)}" aria-label="${escapeHtml(text)}">?</span>`;
}

export function panel({ title, subtitle = "", body, actions = "", className = "" }) {
  return `
    <section class="card panel ${className}">
      <div class="panel-head">
        <div>
          <h3>${escapeHtml(title)}</h3>
          ${subtitle ? `<p>${subtitle}</p>` : ""}
        </div>
        ${actions ? `<div class="panel-actions">${actions}</div>` : ""}
      </div>
      ${body}
    </section>`;
}

export function section(title, subtitle, body, actions = "") {
  return `
    <section class="section">
      <div class="section-head">
        <div>
          <h2>${escapeHtml(title)}</h2>
          ${subtitle ? `<p>${subtitle}</p>` : ""}
        </div>
        ${actions ? `<div class="panel-actions">${actions}</div>` : ""}
      </div>
      ${body}
    </section>`;
}

export function badge(label, tone = "") {
  return `<span class="badge ${tone}">${escapeHtml(label)}</span>`;
}

export function bars(entries, { tone = "", format = num, empty = "No recorded data", max: maxOverride = null } = {}) {
  const items = (entries || []).filter((entry) => Number(entry.value || 0) > 0);
  if (!items.length) return `<div class="bars-empty">${escapeHtml(empty)}</div>`;
  const max = maxOverride ?? Math.max(...items.map((entry) => Number(entry.value || 0)));
  return `<div class="bars">${items.map((entry) => {
    const value = Number(entry.value || 0);
    const width = max ? Math.max(2, (value / max) * 100) : 0;
    return `
      <div class="bar-row">
        <span class="bar-label" title="${escapeHtml(entry.label)}">${escapeHtml(entry.label)}</span>
        <span class="bar-track"><span class="bar-fill ${entry.tone || tone}" style="width:${width}%"></span></span>
        <span class="bar-value">${escapeHtml(format(value))}</span>
      </div>`;
  }).join("")}</div>`;
}

export function table(headers, rows, { empty = "No data recorded", minWidth = "" } = {}) {
  if (!rows || !rows.length) {
    return `<div class="state"><strong>Nothing to show</strong>${escapeHtml(empty)}</div>`;
  }
  return `
    <div class="table-wrap">
      <table class="data-table" ${minWidth ? `style="min-width:${minWidth}"` : ""}>
        <thead><tr>${headers.map((header) => {
          if (typeof header === "object") {
            return `<th class="${header.className || ""}">${escapeHtml(header.label)}</th>`;
          }
          return `<th>${escapeHtml(header)}</th>`;
        }).join("")}</tr></thead>
        <tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>
    </div>`;
}

export function emptyState(title, message = "") {
  return `<div class="state"><strong>${escapeHtml(title)}</strong>${escapeHtml(message)}</div>`;
}

export function loadingState(message = "Loading telemetry…") {
  return `<div class="state"><span class="spinner"></span>${escapeHtml(message)}</div>`;
}

export function errorState(message, { hint = "" } = {}) {
  return `
    <div class="state error">
      <strong>Could not load evaluation data</strong>
      ${escapeHtml(message)}
      ${hint ? `<div class="small mt8">${escapeHtml(hint)}</div>` : ""}
    </div>`;
}

export function authState() {
  return `
    <div class="state">
      <strong>Sign in to view evaluation analytics</strong>
      Detailed pipeline, exam and benchmark reports are protected.
      <div class="mt12"><a class="btn primary" href="/sign-in.html">Sign in</a></div>
    </div>`;
}

export function notice(text, tone = "") {
  return `<div class="notice ${tone}">${text}</div>`;
}

export function missingChip(label = "Not recorded") {
  return `<span class="chip" title="This run predates the telemetry that records this metric">${escapeHtml(label)}</span>`;
}

export function coverageNote(known, total, label = "calls") {
  if (!total) return "No recorded calls";
  return `${num(known)} of ${num(total)} ${label} reported usage`;
}

export function pagination({ total, limit, offset }) {
  const start = total ? offset + 1 : 0;
  const end = Math.min(offset + limit, total);
  return `
    <div class="spread mt16">
      <span class="muted small">Showing ${num(start)}–${num(end)} of ${num(total)}</span>
      <span class="row">
        <button class="btn ghost" type="button" data-page="prev" ${offset <= 0 ? "disabled" : ""}>← Previous</button>
        <button class="btn ghost" type="button" data-page="next" ${end >= total ? "disabled" : ""}>Next →</button>
      </span>
    </div>`;
}

export function bindPagination(host, { limit, offset, total, onPage }) {
  if (!host) return;
  host.querySelectorAll("[data-page]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = button.dataset.page === "next"
        ? offset + limit
        : Math.max(0, offset - limit);
      if (next >= 0 && (next < total || next < offset)) onPage(next);
    });
  });
}

export function meter(value, tone = "ok") {
  const width = Math.max(0, Math.min(100, Number(value || 0) * 100));
  return `<div class="progress"><span class="bar-fill ${tone}" style="width:${width}%"></span></div>`;
}

export function percentCell(value) {
  if (value === null || value === undefined) return `<span class="muted">—</span>`;
  const tone = value >= 0.95 ? "ok" : value >= 0.8 ? "" : value >= 0.6 ? "warn" : "bad";
  return `<span class="${tone}">${pct(value)}</span>`;
}

export function statusBadge(label, tone) {
  return badge(label || "Unknown", tone);
}
