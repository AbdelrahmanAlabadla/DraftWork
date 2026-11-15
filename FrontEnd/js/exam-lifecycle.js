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

export function recoveryProgressRows(events = []) {
  const latest = new Map();
  events.filter((event) => event.event_type === "model_recovery_progress")
    .forEach((event) => {
      const payload = event.payload || {};
      latest.set(Number(payload.model_number || 0), payload);
    });
  return [...latest.entries()].sort(([left], [right]) => left - right)
    .map(([model, payload]) => ({model, ...payload}));
}

export function partialExamSummary(record = {}) {
  const quality = record.metadata?.quality_summary || {};
  const requested = Number(record.metadata?.original_request?.total ?? quality.requested ?? 0);
  const accepted = Number(quality.accepted ?? quality.valid ?? 0);
  const generated = Number(quality.generated ?? accepted);
  const invalid = Number(quality.invalid ?? quality.failed ?? Math.max(0, generated - accepted));
  const missing = Number(quality.missing ?? Math.max(0, requested - generated));
  if (record.status !== "partial" && invalid === 0 && missing === 0 && accepted >= requested) return null;
  const shortages = (record.warnings || []).filter(
    (warning) => warning && typeof warning === "object"
      && warning.type === "question_shortage",
  );
  const audit = record.metadata?.generation_audit || {};
  const unresolved = [
    ...(audit.final_unresolved_slots || []),
    ...(audit.invalid_questions || []),
    ...(audit.unrepairable_plan_slots || []),
  ];
  const seen = new Set();
  const failedSlots = unresolved.filter((item) => {
    const key = String(item?.slot_id || item?.question_id || "");
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  }).map((item) => ({
    slot_id: String(item.slot_id || item.question_id || ""),
    question_type: String(item.question_type || ""),
    model_number: Number(item.model_number || String(item.slot_id || "").match(/^m(\d+)_/)?.[1] || 0),
    reason: String(item.failure_reason || item.reason || item.what_is_wrong
      || t("warning.partial.recovery_exhausted")),
  }));
  return { requested, generated, accepted, invalid, missing, shortages, failedSlots };
}

export function completionSummaryMessage(record = {}) {
  const quality = record.metadata?.quality_summary || {};
  const requested = Number(record.metadata?.original_request?.total ?? quality.requested ?? 0);
  const generated = Number(quality.generated ?? quality.valid ?? requested);
  const accepted = Number(quality.accepted ?? quality.valid ?? generated);
  if (record.status === "partial" || accepted < requested) {
    return t("status.exam_partial_complete", {
      generated: accepted, requested, missing: Math.max(0, requested - accepted),
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
  const details = (summary.shortages || []).map((shortage) => {
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
  const missingCount = summary.missing;
  const missingMessage = details.length
    ? t("warning.partial.missing", { details: details.join(", ") })
    : (missingCount ? t("warning.partial.missing", { details: String(missingCount) }) : "");
  const failed = summary.failedSlots.map((item) => {
    const type = item.question_type || item.slot_id.split("_").slice(1, -1).join("_");
    const translated = t(`warning.qtype.${type}`);
    const label = translated === `warning.qtype.${type}` ? type.replaceAll("_", " ") : translated;
    const index = item.slot_id.match(/_(\d+)$/)?.[1] || "";
    return t("warning.partial.failed_item", {
      model: item.model_number || "?", type: label, index, reason: item.reason,
    });
  });
  warning.textContent = [
    t("warning.partial.heading"),
    t("warning.partial.counts", summary),
    missingMessage,
    failed.length ? `${t("warning.partial.failed_heading")} ${failed.join(" ")}` : "",
    ...fallbackMessages,
  ].filter(Boolean).join("\n");
  warning.dir = record.metadata?.document_language === "ar" ? "rtl" : "auto";
  warning.hidden = false;
}
