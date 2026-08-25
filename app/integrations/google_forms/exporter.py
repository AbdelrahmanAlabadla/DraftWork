"""Deterministic exporter: DraftWork final exam → Google Forms quizzes.

One Form per exam model. Consumes the normalized flat items produced by
app/exports/common.flatten_exam_items so mapping stays schema-driven.
"""
from __future__ import annotations

import time
from typing import Any

from app.integrations.google_forms import client
from app.integrations.google_forms.question_handlers import (
    build_question_requests,
    make_page_break,
)
from app.logging_conf import get_logger

logger = get_logger("GOOGLE_FORMS")


class ExporterDisabled(Exception):
    """Google Forms export is not enabled via configuration."""


class ExportError(Exception):
    """Export could not be completed."""


def export_model(exam: dict[str, Any], title: str | None = None,
                 share_with: list[str] | None = None,
                 creds=None) -> dict[str, Any]:
    """Create one Google Form for one exam model.

    ``creds``: when provided (teacher-owned mode), the form is created with
    those credentials and the teacher IS the owner — no share calls are made.
    When None (central mode), the desktop-account flow is used and explicit
    writer shares are applied to each address in ``share_with``.

    Returns an application-level result dict; per-question failures are
    reported as warnings without aborting the remaining questions. Sharing
    failures are warnings only — the created form is still returned.
    """
    model_number = exam.get("model_number", 1)
    form_title = title or f"Exam — Model {model_number}"
    items = exam.get("_items")
    if items is None:
        from app.exports.common import flatten_exam_items

        items = flatten_exam_items(exam.get("questions") or {})

    teacher_owned = creds is not None
    result: dict[str, Any] = {
        "model_number": model_number,
        "questions_exported": 0,
        "warnings": [],
        "owner": "teacher" if teacher_owned else "central",
    }

    form = client.create_form(form_title, creds=creds)
    result.update({
        "form_id": form["form_id"],
        "edit_url": form["edit_url"],
        "view_url": form["view_url"],
    })

    requests: list[dict[str, Any]] = []
    index = 0
    current_type = None
    questions_per_page = client.QUESTIONS_PER_PAGE
    within_page = 0

    for item in items:
        # One page-break section per question type (titles derive from qtype).
        if item["qtype"] != current_type:
            current_type = item["qtype"]
            label = item.get("label") or current_type.replace("_", " ").title()
            description = ""
            if current_type == "fill_in_the_blank" and item.get("word_bank"):
                description = (
                    "Word Bank: " + " · ".join(dict.fromkeys(item["word_bank"]))
                )
            header = make_page_break(label)
            header["createItem"]["location"]["index"] = index
            index += 1
            requests.append(header)
            if description:
                desc_item = {
                    "createItem": {
                        "item": {"title": description, "textItem": {}},
                        "location": {"index": 0},
                    }
                }
                desc_item["createItem"]["location"]["index"] = index
                index += 1
                requests.append(desc_item)
            within_page = 0

        if questions_per_page > 0 and within_page >= questions_per_page:
            pb = make_page_break(f"{label} (continued)")
            pb["createItem"]["location"]["index"] = index
            index += 1
            requests.append(pb)
            within_page = 0

        reqs, warning = build_question_requests(item)
        if warning:
            result["warnings"].append(warning)
        for req in reqs:
            req["createItem"]["location"]["index"] = index
            index += 1
        requests.extend(reqs)
        result["questions_exported"] += len(reqs)
        within_page += len(reqs)

    t0 = time.perf_counter()
    try:
        client.batch_update(form["form_id"], requests, creds=creds)
    except client.GoogleFormsError as exc:
        # Partial export: the form exists but some/all questions may be missing.
        result["warnings"].append(f"adding questions failed: {exc}")
        result["partial"] = True
        return result
    logger.info(
        "Questions uploaded | model=%d | count=%d | time=%.2fs",
        model_number, result["questions_exported"], time.perf_counter() - t0,
    )

    # Teacher-owned forms need no sharing: owner already has full control.
    if teacher_owned:
        return result

    # Central fallback: explicit user-writer shares only (never `anyone`).
    for email in share_with or []:
        try:
            client.share_form(form["form_id"], email, creds=creds)
            result.setdefault("shared_with", []).append(email)
        except client.GoogleFormsError as exc:
            result["warnings"].append(f"share with {email} failed: {exc}")
    return result


def export_exam(stored_record: dict[str, Any],
                share_with: list[str] | None = None,
                creds=None) -> dict[str, Any]:
    """Export every model of a stored exam record. Returns {exports, errors}."""
    exports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    exams = stored_record.get("exams") or []
    if not isinstance(exams, list) or not exams:
        raise ExportError("stored exam contains no models to export")
    for exam in exams:
        if not isinstance(exam, dict):
            errors.append({"model_number": None, "error": "malformed model entry"})
            continue
        try:
            exports.append(export_model(exam, share_with=share_with, creds=creds))
        except client.GoogleFormsError as exc:
            errors.append({
                "model_number": exam.get("model_number"),
                "error": str(exc),
            })
        except Exception as exc:
            logger.error(
                "Model export crashed | model=%s | exc=%s: %s",
                exam.get("model_number"), type(exc).__name__, exc, exc_info=True,
            )
            errors.append({
                "model_number": exam.get("model_number"),
                "error": f"unexpected error: {exc}",
            })
    return {"exports": exports, "errors": errors}
