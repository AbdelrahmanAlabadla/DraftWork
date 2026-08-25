import { BASE } from "./api.js";
import { state } from "./state.js";

export function initGoogleAuth() {
  refreshAuthState();
  document.getElementById("signOutBtn").addEventListener("click", async () => {
    await fetch(`${BASE}/auth/google/logout`, { method: "POST" }).catch(() => {});
    refreshAuthState();
  });
}

export async function refreshAuthState() {
  const connectBtn = document.getElementById("connectGoogleBtn");
  const info = document.getElementById("connectedInfo");
  const signOut = document.getElementById("signOutBtn");
  const shareRow = document.getElementById("shareRow");

  let data = { connected: false };
  try {
    const res = await fetch(`${BASE}/auth/google/me`);
    data = await res.json();
  } catch (_) { /* server unreachable; keep logged-out UI */ }

  if (data.connected) {
    state.teacherEmail = data.email;
    connectBtn.style.display = "none";
    info.style.display = "";
    info.textContent = `${data.email} · Google connected ✓`;
    signOut.style.display = "";
    // Teacher-owned mode: no manual email needed.
    shareRow.style.display = "none";
  } else {
    state.teacherEmail = null;
    connectBtn.style.display = "";
    info.style.display = "none";
    signOut.style.display = "none";
    shareRow.style.display = "";
  }
}
