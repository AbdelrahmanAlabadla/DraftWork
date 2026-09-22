import { authHeaders } from "./auth.js";
import { localizedError } from "./i18n.js";

export const BASE = "/api/v1";

export function beginIdempotentOperation(name) {
  const storageKey = `genexam:idempotency:${name}`;
  try {
    const existing = sessionStorage.getItem(storageKey);
    if (existing) return { key: existing, storageKey };
    const key = globalThis.crypto?.randomUUID?.()
      || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    sessionStorage.setItem(storageKey, key);
    return { key, storageKey };
  } catch (_) {
    return {
      key: globalThis.crypto?.randomUUID?.()
        || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      storageKey: null,
    };
  }
}

export function completeIdempotentOperation(operation) {
  if (!operation?.storageKey) return;
  try { sessionStorage.removeItem(operation.storageKey); } catch (_) { /* no-op */ }
}

export async function postJSON(path, body, idempotencyKey = null) {
  let headers = { "Content-Type": "application/json" };
  if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
  headers = await authHeaders(headers);
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers,
    credentials: "same-origin",
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

export async function getJSON(path) {
  const res = await fetch(`${BASE}${path}`, {
    credentials: "same-origin",
    headers: await authHeaders(),
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

export async function deleteJSON(path) {
  const res = await fetch(`${BASE}${path}`, {
    method: "DELETE",
    credentials: "same-origin",
    headers: await authHeaders(),
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

export class JobCancelledError extends Error {
  constructor(message) {
    super(message);
    this.name = "JobCancelledError";
  }
}

export async function waitForJob(jobId, onProgress = () => {}) {
  while (true) {
    const { ok, data } = await getJSON(`/jobs/${jobId}`);
    if (!ok) throw new Error(localizedError(data.detail, "status.job_read_failed"));
    onProgress(data);
    if (data.status === "completed") return data;
    if (data.status === "failed" || data.status === "cancelled") {
      const key = data.status === "cancelled" ? "status.job_cancelled" : "status.job_failed";
      const message = localizedError(data.error?.message, key);
      if (data.status === "cancelled") throw new JobCancelledError(message);
      throw new Error(message);
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}
