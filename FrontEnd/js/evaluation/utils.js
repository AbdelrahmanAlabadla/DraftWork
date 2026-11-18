/** Formatting and small DOM helpers shared by every evaluation page. */

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function isMissing(value) {
  return value === null || value === undefined || Number.isNaN(value);
}

export function num(value, digits = 0) {
  if (isMissing(value)) return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function compact(value) {
  if (isMissing(value)) return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const abs = Math.abs(number);
  if (abs >= 1e9) return `${(number / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(number / 1e6).toFixed(1)}M`;
  if (abs >= 1e4) return `${(number / 1e3).toFixed(1)}k`;
  return num(number);
}

export function pct(value, digits = 1) {
  if (isMissing(value)) return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${(number * 100).toFixed(digits)}%`;
}

export function money(value, { digits } = {}) {
  if (isMissing(value)) return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const places = digits ?? (Math.abs(number) >= 1 ? 2 : 4);
  return `$${number.toLocaleString(undefined, {
    minimumFractionDigits: places,
    maximumFractionDigits: places,
  })}`;
}

export function duration(ms) {
  if (isMissing(ms)) return "—";
  const value = Number(ms);
  if (!Number.isFinite(value)) return "—";
  if (value < 1000) return `${Math.round(value)} ms`;
  const seconds = value / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${String(rest).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

export function dateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function dateShort(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

export function dateDay(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });
}

export function labelize(value) {
  const text = String(value ?? "").replaceAll("_", " ").trim();
  if (!text) return "Unknown";
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function sortedEntries(map, direction = -1) {
  return Object.entries(map || {}).sort((a, b) => {
    const delta = Number(b[1] || 0) - Number(a[1] || 0);
    return direction < 0 ? delta : -delta;
  });
}

export function safeRate(numerator, denominator) {
  const top = Number(numerator || 0);
  const bottom = Number(denominator || 0);
  if (!bottom) return null;
  return top / bottom;
}

export function sum(values) {
  return (values || []).reduce((total, value) => total + Number(value || 0), 0);
}

export function qs(name, fallback = "") {
  const value = new URLSearchParams(window.location.search).get(name);
  return value === null ? fallback : value;
}

export function setQs(values) {
  const params = new URLSearchParams(window.location.search);
  Object.entries(values).forEach(([key, value]) => {
    if (value === null || value === undefined || value === "") params.delete(key);
    else params.set(key, value);
  });
  const query = params.toString();
  const url = `${window.location.pathname}${query ? `?${query}` : ""}`;
  window.history.replaceState({}, "", url);
}

export function debounce(fn, delay = 300) {
  let handle = null;
  return (...args) => {
    window.clearTimeout(handle);
    handle = window.setTimeout(() => fn(...args), delay);
  };
}

export function statusTone(value) {
  const key = String(value || "").toLowerCase();
  if (["healthy", "completed", "success", "finalized", "accepted", "pass", "clear"].includes(key)) return "ok";
  if (["needs_attention", "running", "queued", "retrying", "warning", "pending", "partial"].includes(key)) return "warn";
  if (["failed", "cancelled", "error", "rejected", "terminal_failure", "exhausted"].includes(key)) return "bad";
  return "neutral";
}

export function prettyJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (_) {
    return String(value);
  }
}

export function coverage(known, total) {
  if (!total) return "No recorded calls";
  return `${num(known)} of ${num(total)} calls reported usage`;
}

export function el(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

export function mount(target, html) {
  const node = typeof target === "string" ? document.getElementById(target) : target;
  if (node) node.innerHTML = html;
  return node;
}
