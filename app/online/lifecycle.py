from __future__ import annotations

import traceback
import hashlib
import json
from copy import deepcopy
from typing import Any


TERMINAL_STATUSES = {
    "finalized", "planning_failed", "generation_failed", "validation_failed",
    "repair_failed", "duplicate_failed", "finalization_failed",
}


ASSIGNMENT_FIELDS = (
    "task_id", "fitb_entry_id", "base_catalog_task_id", "base_task_family_id",
    "task_family_id", "selected_variation_id", "selected_variation_group_id",
    "task_operation", "target", "student_work_product", "effective_givens",
    "required_givens", "source_chunk_id", "support_chunk_ids", "actual_difficulty",
    "difficulty",
)


def stable_fingerprint(value: Any) -> str:
    """Return one JSON-stable identity for revision and stale-result fencing."""
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()


def compact_assignment(item: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(item.get(key)) for key in ASSIGNMENT_FIELDS if item.get(key) is not None}


def assignment_fingerprint(item: dict[str, Any]) -> str:
    return stable_fingerprint(compact_assignment(item))


def question_fingerprint(item: dict[str, Any]) -> str:
    excluded = {
        "assignment_revision", "assignment_fingerprint", "question_revision",
        "question_fingerprint", "validation", "lifecycle_status",
    }
    return stable_fingerprint({
        key: value for key, value in item.items()
        if key not in excluded and not key.startswith("_")
    })


def assessment_instance_fingerprint(
    assignment: dict[str, Any], question: dict[str, Any] | None = None,
) -> str:
    """Identity of realized assessment work, independent of slot/public IDs."""
    question = question or {}
    return stable_fingerprint({
        "task_id": assignment.get("task_id"),
        "variation_id": assignment.get("selected_variation_id"),
        "operation": assignment.get("task_operation"),
        "target": assignment.get("target"),
        "work_product": assignment.get("student_work_product"),
        "effective_givens": assignment.get("effective_givens") or assignment.get("required_givens"),
        "question": question.get("question") or question.get("statement") or question.get("prompt"),
        "options": question.get("options"),
        "answer": question.get("correct_answer") or question.get("answer"),
    })


def initialize_slot_lifecycle(
    tasks: list[tuple[str, int]], num_models: int
) -> dict[str, dict[str, Any]]:
    return {
        f"m{model_number}_{qtype}_{index}": {
            "slot_id": f"m{model_number}_{qtype}_{index}",
            "model_number": model_number,
            "question_type": qtype,
            "requested": True,
            "planned": False,
            "generated": False,
            "validated": False,
            "repaired": False,
            "finalized": False,
            "question_id": None,
            "task_id": None,
            "fitb_entry_id": None,
            "task_reused": False,
            "fitb_entry_reused": False,
            "fitb_recovery_status": None,
            "reuse_reason": None,
            "task_inventory": {},
            "selected_variation_id": None,
            "selected_variation_group_id": None,
            "effective_instance_id": None,
            "suggested_assessment_angle": None,
            "reuse_justification": None,
            "required_instance_distinctions": [],
            "effective_givens": [],
            "failure_code": None,
            "failure_stage": None,
            "fields_to_fix": [],
            "repair_attempts": [],
            "duplicate_status": None,
            "terminal_status": None,
            "developer_diagnostics": [],
            "original_assignment": None,
            "current_assignment": None,
            "assignment_revision": 0,
            "assignment_fingerprint": None,
            "question_revision": 0,
            "question_fingerprint": None,
            "validation": {},
            "history": [],
        }
        for model_number in range(1, int(num_models) + 1)
        for qtype, count in tasks
        for index in range(1, int(count) + 1)
    }


def developer_diagnostic(exc: BaseException, *, stage: str, **context: Any) -> dict[str, Any]:
    return {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        **context,
    }


def reconcile_slot_lifecycle(
    records: dict[str, dict[str, Any]], state: dict[str, Any], *, stage: str
) -> dict[str, dict[str, Any]]:
    result = deepcopy(records)
    for plan in state.get("plans") or []:
        for qtype, items in (plan.get("items") or {}).items():
            for item in items or []:
                slot_id = str(item.get("slot_id") or "")
                if slot_id not in result:
                    continue
                record = result[slot_id]
                assignment = compact_assignment(item)
                fingerprint = assignment_fingerprint(item)
                previous_fingerprint = record.get("assignment_fingerprint")
                if record.get("original_assignment") is None:
                    record["original_assignment"] = deepcopy(assignment)
                if fingerprint != previous_fingerprint:
                    revision = int(record.get("assignment_revision") or 0) + 1
                    if previous_fingerprint:
                        record.setdefault("history", []).append({
                            "revision": revision,
                            "stage": stage,
                            "event": "assignment_changed",
                            "previous_assignment_fingerprint": previous_fingerprint,
                            "new_assignment_fingerprint": fingerprint,
                            "reason": item.get("reuse_reason") or item.get("difficulty_fallback_reason")
                            or item.get("recovery_failure_code") or "pipeline_assignment_update",
                        })
                    record.update({
                        "current_assignment": deepcopy(assignment),
                        "assignment_revision": revision,
                        "assignment_fingerprint": fingerprint,
                    })
                record.update({
                    "planned": True,
                    "question_type": qtype,
                    "task_id": item.get("task_id"),
                    "fitb_entry_id": item.get("fitb_entry_id"),
                    "task_reused": bool(item.get("task_reused")),
                    "fitb_entry_reused": bool(item.get("fitb_entry_reused")),
                    "reuse_reason": item.get("reuse_reason"),
                    "task_inventory": deepcopy(item.get("task_inventory") or {}),
                    "selected_variation_id": item.get("selected_variation_id"),
                    "selected_variation_group_id": item.get("selected_variation_group_id"),
                    "effective_instance_id": item.get("effective_instance_id"),
                    "suggested_assessment_angle": item.get("suggested_assessment_angle"),
                    "reuse_justification": item.get("reuse_justification"),
                    "required_instance_distinctions": deepcopy(
                        item.get("required_instance_distinctions") or []
                    ),
                    "effective_givens": deepcopy(item.get("effective_givens") or []),
                    "requested_difficulty": item.get("requested_difficulty"),
                    "actual_difficulty": item.get("actual_difficulty") or item.get("difficulty"),
                    "difficulty_adjustment": deepcopy(item.get("difficulty_adaptation")),
                    "difficulty_fallback_reason": item.get("difficulty_fallback_reason"),
                    "base_catalog_task_id": item.get("base_catalog_task_id"),
                    "base_task_family_id": item.get("base_task_family_id"),
                    "derived_task_id": item.get("derived_task_id") or item.get("task_id"),
                    "adjustment_attempts": int(item.get("adjustment_attempts") or 0),
                    "adjustment_failure_reason": item.get("adjustment_failure_reason"),
                })

    for exam in state.get("generated_exams") or []:
        for qtype, section in (exam.get("questions") or {}).items():
            values = section.get("items") if isinstance(section, dict) else section
            for item in values or []:
                slot_id = str(item.get("slot_id") or "")
                if slot_id not in result:
                    continue
                fingerprint = question_fingerprint(item)
                record = result[slot_id]
                previous_fingerprint = record.get("question_fingerprint")
                if fingerprint != previous_fingerprint:
                    revision = int(record.get("question_revision") or 0) + 1
                    if previous_fingerprint:
                        record.setdefault("history", []).append({
                            "revision": revision,
                            "stage": stage,
                            "event": "question_changed",
                            "previous_question_fingerprint": previous_fingerprint,
                            "new_question_fingerprint": fingerprint,
                            "reason": "pipeline_question_update",
                        })
                    record.update({
                        "question_revision": revision,
                        "question_fingerprint": fingerprint,
                    })
                result[slot_id].update({
                    "generated": True,
                    "question_id": item.get("question_id"),
                    "question_type": qtype,
                })

    for report in state.get("validation_reports") or []:
        by_question = {
            str(record.get("question_id") or ""): record for record in result.values()
            if record.get("question_id")
        }
        for verdict in report.get("verdicts") or []:
            record = by_question.get(str(verdict.get("question_id") or ""))
            if record is None:
                continue
            passed = verdict.get("action") == "PASS"
            role = str(verdict.get("validation_role") or "general")
            record.setdefault("validation", {})[role] = {
                "verdict": verdict.get("action"),
                "failure_code": verdict.get("failure_code"),
                "question_fingerprint": record.get("question_fingerprint"),
                "assignment_fingerprint": record.get("assignment_fingerprint"),
                "rule_version": verdict.get("validation_fingerprint_version"),
            }
            record.update({
                "validated": passed,
                "failure_code": None if passed else verdict.get("failure_code"),
                "failure_stage": None if passed else "validation",
                "fields_to_fix": list(verdict.get("fields_to_fix") or []),
                "validator_verdict": deepcopy(verdict),
            })

    repair_events_by_slot: dict[str, list[dict[str, Any]]] = {}
    for event in state.get("validation_repair_history") or []:
        slot_id = str(event.get("slot_id") or "")
        if slot_id in result:
            repair_events_by_slot.setdefault(slot_id, []).append(deepcopy(event))
    for slot_id, events in repair_events_by_slot.items():
        result[slot_id]["repair_attempts"] = events
        result[slot_id]["repaired"] = any(
            event.get("status") in {"repaired", "accepted"} for event in events
        )

    manifest_by_slot = {
        str(item.get("slot_id") or ""): item
        for item in state.get("repair_manifest") or [] if item.get("slot_id")
    }
    for slot_id, target in manifest_by_slot.items():
        if slot_id not in result:
            continue
        result[slot_id]["repair_manifest_status"] = target.get("status")
        result[slot_id]["repair_manifest_history"] = deepcopy(target.get("history") or [])
        if target.get("status") == "resolved":
            result[slot_id].update({
                "validated": True, "failure_code": None, "failure_stage": None,
            })
        else:
            result[slot_id].update({
                "failure_code": target.get("failure_code") or result[slot_id].get("failure_code"),
                "failure_stage": "repair",
            })

    for event in state.get("fitb_recovery_history") or []:
        slot_id = str(event.get("slot_id") or "")
        if slot_id not in result:
            continue
        result[slot_id]["fitb_recovery_status"] = event.get("status")
        result[slot_id]["repair_attempts"].append(deepcopy(event))
        if event.get("fitb_entry_id"):
            result[slot_id]["fitb_entry_id"] = event.get("fitb_entry_id")

    for report in state.get("validation_reports") or []:
        for missing in report.get("missing_slots") or []:
            slot_id = str(missing.get("slot_id") or "")
            if slot_id in result and not result[slot_id].get("generated"):
                result[slot_id].update({
                    "failure_stage": "generation",
                    "failure_code": missing.get("failure_code") or "missing_requested_slot",
                })

    for event in state.get("duplicate_repair_history") or []:
        slot_id = str(event.get("slot_id") or "")
        if slot_id in result:
            result[slot_id]["duplicate_status"] = event.get("status")

    for diagnostic in state.get("developer_diagnostics") or []:
        slot_id = str(diagnostic.get("slot_id") or "")
        if slot_id in result:
            diagnostics = result[slot_id]["developer_diagnostics"]
            if diagnostic not in diagnostics:
                diagnostics.append(deepcopy(diagnostic))

    for item in state.get("unrepairable_plan_slots") or []:
        # Pre-fix checkpoints stored bare slot-ID strings. Current producers
        # persist typed records, but resume must remain safe for legacy state.
        if isinstance(item, dict):
            slot_id = str(item.get("slot_id") or "")
            failure_code = item.get("failure_code") or "plan_repair_exhausted"
            reason = item.get("reason")
        else:
            slot_id = str(item or "")
            failure_code = "plan_repair_exhausted"
            reason = "Persisted legacy plan-repair failure."
        if slot_id in result:
            result[slot_id].update({
                "failure_stage": "planning", "failure_code": failure_code,
                "failure_reason": reason,
                "terminal_status": "planning_failed",
            })
    for item in state.get("invalid_questions") or []:
        slot_id = str(item.get("slot_id") or "")
        if slot_id in result:
            if result[slot_id].get("validated") or (
                manifest_by_slot.get(slot_id) or {}
            ).get("status") == "resolved":
                # Historical failures remain in audit history but cannot
                # override the current passing slot version.
                continue
            failure_code = item.get("failure_code") or "invalid"
            failure_stage = str(item.get("failure_stage") or "") or (
                "duplicate" if str(failure_code).startswith("duplicate") else stage
            )
            terminal_by_stage = {
                "planning": "planning_failed",
                "generation": "generation_failed",
                "validation": "validation_failed",
                "repair": "repair_failed",
                "duplicate": "duplicate_failed",
                "finalization": "finalization_failed",
            }
            result[slot_id].update({
                "generated": True,
                "question_id": item.get("question_id") or result[slot_id].get("question_id"),
                "failure_stage": failure_stage,
                "failure_code": failure_code,
                "terminal_status": terminal_by_stage.get(
                    failure_stage, "finalization_failed"
                ),
            })

    if stage == "finalization":
        for record in result.values():
            if record["generated"] and record["validated"] and not record.get("failure_code"):
                record.update({"finalized": True, "terminal_status": "finalized"})
            elif record["terminal_status"] is None:
                if not record["planned"]:
                    status = "planning_failed"
                elif not record["generated"]:
                    status = "generation_failed"
                elif not record["validated"]:
                    status = "validation_failed"
                else:
                    status = "finalization_failed"
                record["terminal_status"] = status
                record["failure_stage"] = record.get("failure_stage") or status.removesuffix("_failed")
    return result


def slot_counts(records: dict[str, dict[str, Any]]) -> dict[str, int]:
    values = list(records.values())
    return {
        "requested": len(values),
        "planned": sum(bool(item.get("planned")) for item in values),
        "generated": sum(bool(item.get("generated")) for item in values),
        "validated": sum(bool(item.get("validated")) for item in values),
        "repaired": sum(bool(item.get("repaired")) for item in values),
        "finalized": sum(bool(item.get("finalized")) for item in values),
    }
