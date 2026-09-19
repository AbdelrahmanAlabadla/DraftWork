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
  const headers = { "Content-Type": "application/json" };
  if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
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
  const res = await fetch(`${BASE}${path}`, { credentials: "same-origin" });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

export async function waitForJob(jobId, onProgress = () => {}) {
  while (true) {
    const { ok, data } = await getJSON(`/jobs/${jobId}`);
    if (!ok) throw new Error(data.detail || "Could not read job status.");
    onProgress(data);
    if (data.status === "completed") return data;
    if (data.status === "failed" || data.status === "cancelled") {
      throw new Error(data.error?.message || `Job ${data.status}.`);
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}
