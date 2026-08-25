// Pure select-all toggle logic (unit-testable without a DOM).
export function computeSelectAllLabel(selectedCount, totalCount) {
  if (totalCount === 0 || selectedCount < totalCount) return "Select All Titles";
  return "Unselect All Titles";
}
