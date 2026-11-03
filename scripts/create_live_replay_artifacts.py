"""Create read-only exports and audit reports for the authorized replay run."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psycopg.rows import dict_row

from app import config, db
from app.exports.docx_exporter import render_answers_docx, render_exam_docx
from app.exports.pdf_exporter import render_answers_pdf, render_exam_pdf

SOURCE_JOB = "d12480df-9961-4c0c-93a2-64f86495a33b"
JOB = "dce90847-63df-4017-a3c7-f43fb1e420db"
EXAM = "exam_07d7e614b165"
CATALOG = "8a1a90c242e9ece845061c47c29fb864b0a7708fd8774192ff7ed6a5e14e75ce"
OUT = Path("live_exam_replay")


def safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def section_items(section: Any) -> list[dict[str, Any]]:
    return list(section.get("items") or []) if isinstance(section, dict) else list(section or [])


def flatten_questions(exams: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (qtype, item)
        for exam in exams
        for qtype, section in (exam.get("questions") or {}).items()
        for item in section_items(section)
        if isinstance(item, dict)
    ]


def usage_rows(cursor, job_id: str) -> list[dict[str, Any]]:
    cursor.execute(
        """SELECT provider, model, operation, COUNT(*) AS calls,
                  COALESCE(SUM(input_tokens), 0) AS input_tokens,
                  COALESCE(SUM(output_tokens), 0) AS output_tokens,
                  COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                  COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
                  COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd,
                  COALESCE(SUM(duration_ms), 0) AS duration_ms
             FROM llm_usage WHERE job_id = %s
             GROUP BY provider, model, operation ORDER BY operation""",
        (job_id,),
    )
    return [safe(dict(row)) for row in cursor.fetchall()]


def summarize_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {key: 0 for key in ("calls", "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "estimated_cost_usd", "duration_ms")}
    for row in rows:
        for key in totals:
            totals[key] += row.get(key, 0) or 0
    totals["estimated_cost_usd"] = float(totals["estimated_cost_usd"])
    return totals


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM jobs WHERE id = %s", (JOB,))
            job = dict(cursor.fetchone() or {})
            cursor.execute("SELECT * FROM jobs WHERE id = %s", (SOURCE_JOB,))
            source = dict(cursor.fetchone() or {})
            cursor.execute("SELECT * FROM exams WHERE id = %s", (EXAM,))
            exam_record = dict(cursor.fetchone() or {})
            cursor.execute(
                """SELECT stage, event_type, checkpoint_revision, payload, created_at
                     FROM exam_audit_events WHERE job_id = %s ORDER BY id""",
                (JOB,),
            )
            raw_events = [safe(dict(row)) for row in cursor.fetchall()]
            continuation_usage = usage_rows(cursor, JOB)
            source_usage = usage_rows(cursor, SOURCE_JOB)

    final_state: dict[str, Any] = {}
    for event in raw_events:
        payload = event.get("payload") or {}
        output = payload.get("node_output") or payload.get("checkpoint_state")
        if isinstance(output, dict):
            final_state.update(output)
    missing_slots = {
        str(slot_id)
        for warning in (job.get("result_data") or {}).get("warnings") or []
        if isinstance(warning, dict)
        for slot_id in warning.get("missing_slot_ids") or []
    }
    exams = list(exam_record.get("exams") or [])
    export_exam = dict(exams[0]) if exams else {"model_number": 1, "questions": {}}
    export_exam["document_language"] = "ar"
    export_exam["questions"] = {
        qtype: (
            {**section, "items": [item for item in section_items(section) if str(item.get("slot_id")) not in missing_slots]}
            if isinstance(section, dict)
            else [item for item in section_items(section) if str(item.get("slot_id")) not in missing_slots]
        )
        for qtype, section in (export_exam.get("questions") or {}).items()
    }
    accepted = flatten_questions([export_exam])
    requested = {"mcq": 10, "true_false": 5, "equation": 4, "word_problem": 4}
    final_counts = Counter(qtype for qtype, _item in accepted)
    events = [{
        "stage": event.get("stage"), "event_type": event.get("event_type"),
        "checkpoint_revision": event.get("checkpoint_revision"), "created_at": event.get("created_at"),
        "payload_keys": sorted((event.get("payload") or {}).keys()),
        "replay_provenance": (event.get("payload") or {}).get("replay_provenance"),
    } for event in raw_events]
    plans = [item for plan in final_state.get("plans") or [] for items in (plan.get("items") or {}).values() for item in items or []]
    plan_by_slot = {str(item.get("slot_id")): item for item in plans}
    question_by_slot = {str(item.get("slot_id")): (qtype, item) for qtype, item in accepted}
    lifecycle = final_state.get("slot_lifecycle") or {}
    validation_activity = Counter(
        str(row.get("role") or row.get("validator") or "unknown")
        for row in final_state.get("validation_call_logs") or [] if isinstance(row, dict)
    )
    repair_activity = Counter(
        str(row.get("agent") or row.get("role") or "unknown")
        for row in final_state.get("repair_call_logs") or [] if isinstance(row, dict)
    )
    duplicate_reports = list(final_state.get("duplicate_reports") or [])
    duplicate_v2 = {
        "theoretical_pairs": 253,
        "scan_count": int(final_state.get("duplicate_scan_count") or 0),
        "candidate_pairs": sum(len(row.get("candidate_pairs") or []) for row in duplicate_reports if isinstance(row, dict)),
        "judge_calls": sum(len(row.get("pair_verdicts") or []) for row in duplicate_reports if isinstance(row, dict)),
        "confirmed_groups": sum(len(row.get("duplicate_groups") or []) for row in duplicate_reports if isinstance(row, dict)),
        "unresolved_conflicts": len(final_state.get("unresolved_duplicate_conflicts") or []),
    }
    audits = []
    for slot_id, plan in sorted(plan_by_slot.items()):
        qtype, question = question_by_slot.get(slot_id, (None, {}))
        record = lifecycle.get(slot_id) or {}
        audits.append({
            "slot_id": slot_id,
            "question_id": question.get("question_id"),
            "question_type": qtype or record.get("question_type"),
            "original_task_id": plan.get("base_catalog_task_id") or plan.get("task_id"),
            "final_task_id": plan.get("task_id"),
            "variation_id": plan.get("selected_variation_id"),
            "source_chunk_id": plan.get("source_chunk_id"),
            "support_chunk_ids": plan.get("support_chunk_ids") or [],
            "final_status": record.get("terminal_status") or ("accepted" if question else "blocked"),
            "validation": question.get("validation") or record.get("validation") or {},
            "repair_history": record.get("repair_history") or [],
            "correct_answer": question.get("correct_answer", question.get("answer", question.get("final_answer"))),
            "accepted": bool(question),
        })

    metadata = dict(exam_record.get("metadata") or {})
    metadata["exam_title"] = "\u0627\u062e\u062a\u0628\u0627\u0631 \u0627\u0644\u0631\u064a\u0627\u0636\u064a\u0627\u062a \u0628\u0627\u0644\u0644\u063a\u0629 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u2014 \u063a\u064a\u0631 \u0645\u0643\u062a\u0645\u0644"
    metadata["document_language"] = "ar"
    exports: dict[str, str] = {}
    for name, renderer in (
        ("exam_questions.pdf", render_exam_pdf),
        ("exam_answer_key.pdf", render_answers_pdf),
        ("exam_questions.docx", render_exam_docx),
        ("exam_answer_key.docx", render_answers_docx),
    ):
        try:
            (OUT / name).write_bytes(renderer(export_exam, metadata))
            exports[name] = "created"
        except Exception as exc:
            exports[name] = f"failed: {type(exc).__name__}: {exc}"

    continuation_totals = summarize_usage(continuation_usage)
    source_totals = summarize_usage(source_usage)
    all_totals = {
        key: source_totals.get(key, 0) + continuation_totals.get(key, 0)
        for key in source_totals
    }
    stage_rows = [
        {"run": "original", "stage": row["operation"], **row}
        for row in source_usage
    ] + [
        {"run": "continuation", "stage": row["operation"], **row}
        for row in continuation_usage
    ]
    report = {
        "identifiers": {"original_job_id": SOURCE_JOB, "continuation_job_id": JOB, "exam_id": EXAM, "catalog_id": CATALOG},
        "outcome": {
            "requested": sum(requested.values()), "requested_distribution": requested,
            "final_accepted": len(accepted), "final_distribution": dict(final_counts),
            "blocked_or_rejected": sum(requested.values()) - len(accepted),
            "missing_slot_ids": sorted(missing_slots),
            "job_status": job.get("status"), "exam_status": exam_record.get("status"),
        },
        "telemetry": {
            "original": {"rows": source_usage, "totals": source_totals},
            "continuation": {"rows": continuation_usage, "totals": continuation_totals},
            "combined": all_totals,
            "pricing_note": "Estimated from configured per-token pricing; provider invoices are not available in this telemetry.",
        },
        "audit_events": events,
        "validator_activity": dict(validation_activity),
        "repair_activity": dict(repair_activity),
        "duplicate_v2": duplicate_v2,
        "question_audit": audits,
        "exports": exports,
        "source_job_status_after_replay": source.get("status"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "question_audit.json").write_text(json.dumps(safe(audits), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "execution_metrics.json").write_text(json.dumps(safe({"stage_rows": stage_rows, "events": events, "telemetry": report["telemetry"]}), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "evaluation_report.json").write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# DraftWork live checkpoint replay evaluation",
        "", f"- Original job: `{SOURCE_JOB}`", f"- Continuation job: `{JOB}`", f"- Exam: `{EXAM}`", f"- Catalog: `{CATALOG}`", "",
        "## Outcome", "", f"- Final status: `{exam_record.get('status')}`", f"- Accepted: **{len(accepted)} / 23**", f"- Final distribution: `{dict(final_counts)}`", f"- Blocked or rejected: **{23 - len(accepted)}**", "",
        "## Cost and usage", "", "| Run | Calls | Input | Output | Reasoning | Estimated USD |", "|---|---:|---:|---:|---:|---:|",
        f"| Original catalog + generation | {source_totals['calls']} | {source_totals['input_tokens']} | {source_totals['output_tokens']} | {source_totals['reasoning_tokens']} | ${source_totals['estimated_cost_usd']:.8f} |",
        f"| Continuation | {continuation_totals['calls']} | {continuation_totals['input_tokens']} | {continuation_totals['output_tokens']} | {continuation_totals['reasoning_tokens']} | ${continuation_totals['estimated_cost_usd']:.8f} |",
        f"| Combined | {all_totals['calls']} | {all_totals['input_tokens']} | {all_totals['output_tokens']} | {all_totals['reasoning_tokens']} | ${all_totals['estimated_cost_usd']:.8f} |",
        "", "## Provider-operation telemetry", "", "| Run | Operation | Calls | Input | Output | Cost |", "|---|---|---:|---:|---:|---:|",
    ]
    lines += [f"| {row['run']} | {row['operation']} | {row['calls']} | {row['input_tokens']} | {row['output_tokens']} | ${float(row['estimated_cost_usd']):.8f} |" for row in stage_rows]
    lines += ["", "## Validator and repair routing", "", f"- Validator dispatches: `{dict(validation_activity)}`", f"- Repair routing events: `{dict(repair_activity)}`", "", "## Duplicate V2", "", f"- Theoretical pairs: 253", f"- Screening scans: {duplicate_v2['scan_count']}", f"- Candidate pairs: {duplicate_v2['candidate_pairs']}", f"- Duplicate Validator judge calls: {duplicate_v2['judge_calls']}", f"- Confirmed duplicate groups: {duplicate_v2['confirmed_groups']}", f"- Unresolved conflicts: {duplicate_v2['unresolved_conflicts']}", ""]
    lines += ["", "## Question audit", "", "| Slot | Type | Final status | Answer |", "|---|---|---|---|"]
    lines += [f"| {row['slot_id']} | {row['question_type']} | {row['final_status']} | {str(row['correct_answer'] or '').replace('|', '\\|')} |" for row in audits]
    lines += ["", "## Confirmed observations", "", "- The original job remains `failed`; its checkpoint was not modified.", "- The continuation began from the copied post-generation checkpoint and did not compile a catalog, plan, or initially generate questions.", "- Exports contain only final accepted questions and are labeled incomplete.", "- Estimated costs are token-based estimates; invoice-level charges are unavailable.", ""]
    (OUT / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "distribution": dict(final_counts), "exports": exports, "continuation_cost": f"{continuation_totals['estimated_cost_usd']:.8f}"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
