export const BASE = "/api/v1";

export async function postJSON(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
