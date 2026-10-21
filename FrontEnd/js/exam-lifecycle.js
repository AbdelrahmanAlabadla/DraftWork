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
  const requested = Number(record.metadata?.original_request?.total ?? quality.requested ?? shortages.reduce(
    (total, warning) => total + Number(warning.requested_count ?? warning.requested ?? 0), 0,
  ));
  const generated = Number(quality.valid ?? Math.max(0, requested - shortages.reduce(
    (total, warning) => total + Number(warning.missing_count ?? warning.missing ?? 0), 0,
  )));
  return { requested, generated, shortages };
}

export function completionSummaryMessage(record = {}) {
  const quality = record.metadata?.quality_summary || {};
  const requested = Number(record.metadata?.original_request?.total ?? quality.requested ?? 0);
  const generated = Number(quality.valid ?? requested);
  if (record.status === "partial" || generated < requested) {
    return t("status.exam_partial_complete", {
      generated, requested, missing: Math.max(0, requested - generated),
    });
  }
  return t("status.exam_full_complete", { generated, requested });
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
    const slots = (shortage.missing_slot_ids || []).join(", ");
    const base = t("warning.partial.item", { missing, qtype });
    return slots ? `${base} (${slots})` : base;
  });
  const fallbackMessages = (record.warnings || [])
    .filter((item) => item && typeof item === "object" && item.type === "difficulty_fallback")
    .map((item) => {
      const counts = Object.entries(item.fallback_counts || {})
        .map(([level, count]) => `${count} ${t(`difficulty.${level}`)}`)
        .join(", ");
      return t("warning.difficulty_fallback", {
        requested: t(`difficulty.${item.requested_difficulty}`),
        matched: item[`${item.requested_difficulty}_count`] || 0,
        substitutions: counts,
      });
    })
    .filter(Boolean);
  const missingCount = Math.max(0, summary.requested - summary.generated);
  const missingMessage = details.length
    ? t("warning.partial.missing", { details: details.join(", ") })
    : (missingCount ? t("warning.partial.missing", { details: String(missingCount) }) : "");
  warning.textContent = [
    t("warning.partial.summary", {
      generated: summary.generated,
      requested: summary.requested,
    }),
    missingMessage,
    ...fallbackMessages,
  ].filter(Boolean).join(" ");
  warning.hidden = false;
}
