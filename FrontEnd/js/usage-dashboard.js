import { getJSON } from "./api.js";
import { t, uiLang } from "./i18n.js";

function known(value, formatter = String) {
  return value === null || value === undefined ? t("usage.unknown") : formatter(value);
}

function integer(value) {
  return Number(value).toLocaleString(uiLang() === "ar" ? "ar" : "en");
}

function money(value) {
  return `$${Number(value).toFixed(6)}`;
}

function duration(value) {
  const seconds = Number(value) / 1000;
  return seconds < 60 ? `${seconds.toFixed(1)}s` : `${(seconds / 60).toFixed(1)}m`;
}

export function renderUsageDashboard(summary = {}) {
  const root = document.getElementById("examUsageDashboard");
  if (!root) return;
  const totals = summary.totals;
  if (!totals) {
    root.hidden = false;
    root.textContent = t("usage.unknown");
    return;
  }
  const totalTokens = [totals.input_tokens, totals.output_tokens, totals.reasoning_tokens]
    .every((value) => value !== null && value !== undefined)
    ? Number(totals.input_tokens) + Number(totals.output_tokens) + Number(totals.reasoning_tokens)
    : null;
  const heading = document.createElement("h4");
  heading.textContent = t("usage.title");
  const aggregate = document.createElement("p");
  aggregate.className = "usage-dashboard-total";
  aggregate.textContent = t("usage.total", {
    calls: known(totals.calls, integer), tokens: known(totalTokens, integer),
    cost: known(totals.estimated_cost_usd, money),
    latency: known(totals.latency_ms, duration),
    wall: known(summary.stage_wall_time_ms, duration),
  });
  const table = document.createElement("table");
  table.className = "usage-dashboard-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["agent", "calls", "input", "output", "reasoning", "cost", "latency"].forEach((key) => {
    const cell = document.createElement("th");
    cell.textContent = t(`usage.${key}`);
    headRow.appendChild(cell);
  });
  head.appendChild(headRow);
  const body = document.createElement("tbody");
  (summary.agents || []).forEach((agent) => {
    const row = document.createElement("tr");
    const values = [
      agent.agent,
      known(agent.calls, integer),
      known(agent.input_tokens, integer),
      known(agent.output_tokens, integer),
      known(agent.reasoning_tokens, integer),
      known(agent.estimated_cost_usd, money),
      known(agent.latency_ms, duration),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      cell.textContent = String(value);
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  table.append(head, body);
  root.replaceChildren(heading, aggregate, table);
  root.hidden = false;
}

export async function loadUsageDashboard(examId) {
  const response = await getJSON(`/exams/${encodeURIComponent(examId)}/usage`);
  if (!response.ok) return;
  renderUsageDashboard(response.data);
}
