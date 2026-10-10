import { t } from "./i18n.js";

export function resolveExamId(createResponse = {}, job = {}) {
  const value = createResponse.exam_id || job.exam_id || job.result?.exam_id;
  const examId = String(value || "").trim();
  if (!examId) throw new Error("The server did not return an exam ID.");
  return examId;
}

export function examResourcePath(examId) {
  const value = String(examId || "").trim();
  if (!value || value === "undefined" || value === "null") {
    throw new Error("The server did not return an exam ID.");
  }
  return `/exams/${encodeURIComponent(value)}`;
}

export function partialExamSummary(record = {}) {
  if (record.status !== "partial") return null;
  const quality = record.metadata?.quality_summary || {};
  const shortages = (record.warnings || []).filter(
    (warning) => warning && typeof warning === "object"
      && warning.type === "question_shortage",
  );
  const requested = Number(quality.requested ?? shortages.reduce(
    (total, warning) => total + Number(warning.requested_count ?? warning.requested ?? 0), 0,
  ));
  const generated = Number(quality.valid ?? Math.max(0, requested - shortages.reduce(
    (total, warning) => total + Number(warning.missing_count ?? warning.missing ?? 0), 0,
  )));
  return { requested, generated, shortages };
}

export function renderPartialExamWarning(record = {}) {
  const warning = document.getElementById("examPartialWarning");
  const summary = partialExamSummary(record);
  if (!warning) return;
  if (!summary) {
    warning.hidden = true;
    warning.textContent = "";
    return;
  }
  const details = summary.shortages.map((shortage) => {
    const missing = shortage.missing_count ?? shortage.missing ?? 0;
    const rawType = String(shortage.question_type || "question");
    const typeKey = `warning.qtype.${rawType}`;
    const translatedType = t(typeKey);
    const qtype = translatedType === typeKey ? rawType.replaceAll("_", " ") : translatedType;
    return t("warning.partial.item", { missing, qtype });
  });
  warning.textContent = [
    t("warning.partial.summary", {
      generated: summary.generated,
      requested: summary.requested,
    }),
    t("warning.partial.missing", { details: details.join(", ") }),
  ].join(" ");
  warning.hidden = false;
}
