/** Authenticated JSON client for the evaluation API. */
import { authHeaders } from "../../js/auth.js";

const BASE = "/api/v1/evaluation";

export class ApiError extends Error {
  constructor(message, { status = 0, code = "", retryable = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }

  get isAuth() {
    return this.status === 401 || this.code === "AUTH";
  }
}

async function request(method, path, { params = {}, body = null, headers = {} } = {}) {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  let payload;
  let finalHeaders = { Accept: "application/json", ...headers };
  if (body !== null && body !== undefined) {
    payload = JSON.stringify(body);
    finalHeaders["Content-Type"] = "application/json";
  }
  finalHeaders = await authHeaders(finalHeaders);
  let response;
  try {
    response = await fetch(url.toString(), {
      method,
      credentials: "same-origin",
      headers: finalHeaders,
      body: payload,
    });
  } catch (error) {
    throw new ApiError("The server could not be reached. Check your connection and try again.", {
      status: 0,
      code: "NETWORK",
      retryable: true,
    });
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail =
      (typeof data.detail === "string" && data.detail) ||
      data.error?.message ||
      `Request failed with status ${response.status}`;
    throw new ApiError(detail, {
      status: response.status,
      code: data.error?.code || (response.status === 401 ? "AUTH" : ""),
      retryable: response.status >= 500,
    });
  }
  return data;
}

export function apiGet(path, params = {}) {
  return request("GET", path, { params });
}

export function apiPost(path, body = null, headers = {}) {
  return request("POST", path, { body, headers });
}

/** GET against the main application API (used by the benchmark section picker). */
export async function appGet(path) {
  const headers = await authHeaders({ Accept: "application/json" });
  const response = await fetch(`/api/v1${path}`, {
    credentials: "same-origin",
    headers,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(
      (typeof data.detail === "string" && data.detail) || `Request failed (${response.status})`,
      { status: response.status, code: data.error?.code || "" },
    );
  }
  return data;
}
