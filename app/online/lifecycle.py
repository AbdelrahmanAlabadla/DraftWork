from __future__ import annotations

import traceback
from copy import deepcopy
from typing import Any


TERMINAL_STATUSES = {
    "finalized", "planning_failed", "generation_failed", "validation_failed",
    "repair_failed", "duplicate_failed", "finalization_failed",
}


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
            "plan_review_status": None,
            "plan_review_reason": None,
            "plan_review_source": None,
            "suggested_assessment_angle": None,
            "reuse_justification": None,
            "required_instance_distinctions": [],
            "plan_review_selection_rank": {},
            "failure_code": None,
            "failure_stage": None,
            "fields_to_fix": [],
            "repair_attempts": [],
            "duplicate_status": None,
            "terminal_status": None,
            "developer_diagnostics": [],
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
                record.update({
                    "planned": True,
                    "question_type": qtype,
                    "task_id": item.get("task_id"),
                    "fitb_entry_id": item.get("fitb_entry_id"),
                    "task_reused": bool(item.get("task_reused")),
                    "fitb_entry_reused": bool(item.get("fitb_entry_reused")),
                    "reuse_reason": item.get("reuse_reason"),
                    "task_inventory": deepcopy(item.get("task_inventory") or {}),
                    "plan_review_status": item.get("plan_review_status"),
                    "plan_review_reason": item.get("plan_review_reason"),
                    "plan_review_source": item.get("plan_review_source"),
                    "suggested_assessment_angle": item.get("suggested_assessment_angle"),
                    "reuse_justification": item.get("reuse_justification"),
                    "required_instance_distinctions": deepcopy(
                        item.get("required_instance_distinctions") or []
                    ),
                    "plan_review_selection_rank": deepcopy(
                        item.get("selection_rank") or {}
                    ),
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
        slot_id = str(item.get("slot_id") or "")
        if slot_id in result:
            result[slot_id].update({
                "failure_stage": "planning", "failure_code": item.get("failure_code"),
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
