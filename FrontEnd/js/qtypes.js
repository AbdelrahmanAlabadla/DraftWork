import { state } from "./state.js";

export function updateTotal() {
  let t = 0;
  for (const k in state.counts) if (state.enabled[k]) t += state.counts[k];
  document.getElementById("totalBadge").textContent = `Total: ${t} questions`;
}

export function step(key, dir) {
  if (!state.enabled[key]) return;
  state.counts[key] = Math.max(0, state.counts[key] + dir);
  document.getElementById(`count-${key}`).textContent = state.counts[key];
  updateTotal();
}

export function toggleType(key) {
  state.enabled[key] = !state.enabled[key];
  const btn = document.querySelector(`.toggle[data-key="${key}"]`);
  const card = document.getElementById(`card-${key}`);
  btn.classList.toggle("on", state.enabled[key]);
  card.classList.toggle("disabled", !state.enabled[key]);
  updateTotal();
}
