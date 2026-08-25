import { state } from "./state.js";

export function initSettings() {
  document.querySelectorAll(".model-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".model-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.numModels = parseInt(btn.dataset.val);
    });
  });

  document.querySelectorAll(".diff-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".diff-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.difficulty = btn.dataset.val;
    });
  });
}
