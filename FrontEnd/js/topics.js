import { state } from "./state.js";
import { computeSelectAllLabel } from "./select_all.js";

export function initTopics() {
  document.getElementById("selectAllBtn").addEventListener("click", toggleSelectAll);
}

function setSelectAllLabel() {
  const btn = document.getElementById("selectAllBtn");
  btn.textContent = computeSelectAllLabel(state.selectedChildren.size, countAllCheckboxes());
}

function countAllCheckboxes() {
  return document.querySelectorAll(".child-check").length +
    [...document.querySelectorAll(".section-check")].filter(
      (s) => !document.querySelectorAll(`.child-check[data-section="${s.dataset.section}"]`).length
    ).length;
}

export function showSectionsLoading() {
  document.getElementById("sectionsLoading").classList.add("show");
  document.getElementById("sectionsTree").innerHTML = "";
  document.getElementById("sectionsCount").textContent = "";
}

export function hideSectionsLoading() {
  document.getElementById("sectionsLoading").classList.remove("show");
}

export function renderSectionsTree(data) {
  const tree = document.getElementById("sectionsTree");
  tree.innerHTML = "";
  state.selectedChildren.clear();

  const sections = data.sections || [];
  const selectAllBtn = document.getElementById("selectAllBtn");
  if (selectAllBtn) selectAllBtn.style.display = sections.length ? "" : "none";

  if (!sections.length) {
    hideSectionsLoading();
    return;
  }

  const frag = document.createDocumentFragment();
  sections.forEach((section, si) => {
    const subs = section.subsections || [];
    const node = document.createElement("div");
    node.className = "section-node";

    const srow = document.createElement("label");
    srow.className = "section-row";
    const sbox = document.createElement("input");
    sbox.type = "checkbox";
    sbox.className = "section-check";
    sbox.dataset.section = String(si);
    sbox.checked = false;
    sbox.dataset.childIds = (section.child_ids || []).join(",");
    srow.appendChild(sbox);
    const stitle = document.createElement("span");
    stitle.className = "section-title";
    stitle.textContent = section.title || "Untitled";
    srow.appendChild(stitle);
    node.appendChild(srow);

    const slist = document.createElement("div");
    slist.className = "subsection-list";
    subs.forEach((sub) => {
      const lrow = document.createElement("label");
      lrow.className = "subsection-row";
      const cbox = document.createElement("input");
      cbox.type = "checkbox";
      cbox.className = "child-check";
      cbox.dataset.child = sub.child_id;
      cbox.dataset.section = String(si);
      cbox.checked = false;
      lrow.appendChild(cbox);
      const ctitle = document.createElement("span");
      ctitle.className = "subsection-title";
      ctitle.textContent = sub.title || "Untitled";
      lrow.appendChild(ctitle);
      slist.appendChild(lrow);
    });
    node.appendChild(slist);
    frag.appendChild(node);
  });
  tree.appendChild(frag);

  tree.querySelectorAll(".section-check").forEach((box) => {
    box.addEventListener("change", () => {
      const si = box.dataset.section;
      const cbs = tree.querySelectorAll(`.child-check[data-section="${si}"]`);
      if (cbs.length) {
        cbs.forEach((cb) => {
          cb.checked = box.checked;
          if (box.checked) state.selectedChildren.add(cb.dataset.child);
          else state.selectedChildren.delete(cb.dataset.child);
        });
      } else if (box.dataset.childIds) {
        box.dataset.childIds.split(",").forEach((id) => {
          if (!id) return;
          if (box.checked) state.selectedChildren.add(id);
          else state.selectedChildren.delete(id);
        });
      }
      refreshSectionState(tree, si);
      updateSectionsCount();
    });
  });

  tree.querySelectorAll(".child-check").forEach((cb) => {
    cb.addEventListener("change", () => {
      if (cb.checked) state.selectedChildren.add(cb.dataset.child);
      else state.selectedChildren.delete(cb.dataset.child);
      refreshSectionState(tree, cb.dataset.section);
      updateSectionsCount();
    });
  });

  hideSectionsLoading();
  updateSectionsCount();
}

function refreshSectionState(tree, si) {
  const sbox = tree.querySelector(`.section-check[data-section="${si}"]`);
  const cbs = tree.querySelectorAll(`.child-check[data-section="${si}"]`);
  // Sections without visible subsections are selected via child_ids directly.
  if (!cbs.length) return;
  const checked = [...cbs].filter((cb) => cb.checked).length;
  sbox.checked = checked === cbs.length;
  sbox.indeterminate = checked > 0 && checked < cbs.length;
}

function toggleSelectAll() {
  const selectAll = computeSelectAllLabel(
    state.selectedChildren.size, countAllCheckboxes()
  ) === "Unselect All Titles";
  applySelectAll(!selectAll);
}

function applySelectAll(select) {
  const tree = document.getElementById("sectionsTree");
  if (!tree || !tree.children.length) return;
  tree.querySelectorAll(".child-check").forEach((cb) => {
    cb.checked = select;
    if (select) state.selectedChildren.add(cb.dataset.child);
    else state.selectedChildren.delete(cb.dataset.child);
  });
  tree.querySelectorAll(".section-check").forEach((sbox) => {
    sbox.checked = select;
    (sbox.dataset.childIds || "").split(",").forEach((id) => {
      if (!id) return;
      if (select) state.selectedChildren.add(id);
      else state.selectedChildren.delete(id);
    });
    refreshSectionState(tree, sbox.dataset.section);
  });
  updateSectionsCount();
}

function updateSectionsCount() {
  document.getElementById("sectionsCount").textContent =
    `${state.selectedChildren.size} selected`;
  setSelectAllLabel();
}
