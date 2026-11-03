"""Isolated paid replay for Equation 3, True/False 5, and MCQ 9.

Uses production Validation/Repair/Difficulty Adjustment/Generation functions, but
keeps all state and telemetry on the local filesystem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config
from app.api import repositories
from app.online.multi_validator import validate_exam_multi
from app.online.multi_repair import repair_question_specialized
from app.online.validator import (
    _manifest_from_report,
    repair_invalid_questions,
    validate_generated_questions,
)

FIXTURE = ROOT / "data/storage/stage_test_checkpoints/failed_questions_live_regression/failed_questions.json"
SLOTS = ("m1_equation_3", "m1_true_false_5", "m1_mcq_9")
MAX_CALLS = 150
MAX_COST = 0.20


class BudgetStop(RuntimeError):
    pass


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_fixture(slots: tuple[str, ...] = SLOTS) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    records = [row for row in payload.get("records") or [] if row.get("slot_id") in slots]
    by_slot = {row.get("slot_id"): row for row in records}
    missing = sorted(set(slots) - set(by_slot))
    if missing:
        raise RuntimeError(f"Fixture is missing required slots: {missing}")
    for slot, row in by_slot.items():
        if not isinstance(row.get("question"), dict) or not isinstance(row.get("assignment"), dict):
            raise RuntimeError(f"{slot}: question or assignment is unavailable")
        evidence = row.get("source_evidence") or {}
        ids = row.get("evidence_chunk_ids") or []
        absent = [value for value in ids if not str(evidence.get(str(value)) or "").strip()]
        if absent:
            raise RuntimeError(f"{slot}: source evidence unavailable: {absent}")
    return payload, records


def _build_state(payload: dict[str, Any], records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    sections: dict[str, list[dict[str, Any]]] = {}
    plan_items: dict[str, list[dict[str, Any]]] = {}
    plan_by_slot: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for row in records:
        qtype = str(row["question_type"])
        question = deepcopy(row["question"])
        assignment = deepcopy(row["assignment"])
        sections.setdefault(qtype, []).append(question)
        plan_items.setdefault(qtype, []).append(assignment)
        plan_by_slot[str(row["slot_id"])] = assignment
        sources.update({str(k): str(v) for k, v in (row.get("source_evidence") or {}).items()})
    exam = {"model_number": 1, "questions": sections}
    state = {
        "job_id": f"isolated-live-three-{uuid.uuid4().hex}",
        "document_id": payload.get("document_id"),
        "generated_exams": [deepcopy(exam)],
        "plans": [{"model_number": 1, "items": plan_items}],
        "retrieved_chunks": [{"child_id": k, "content": v} for k, v in sources.items()],
        "assessment_profile": {"domain": "STEM", "question_types": {}},
        "difficulty": "mix",
        "document_language": "ar",
        "validated_models": [],
        "question_repair_attempts": {},
        "repair_replacement_memory": {},
        "warnings": [],
        "invalid_questions": [],
    }
    return state, plan_by_slot, sources


@contextmanager
def _capture(
    records: list[dict[str, Any]], *, initial_calls: int = 0, initial_cost: float = 0.0,
):
    from app.api import repositories as repo
    from app.online import difficulty_adjustment, exam_builder, multi_repair, multi_validator

    modules = (multi_validator, multi_repair, difficulty_adjustment, exam_builder)
    originals = {module: module.request_structured for module in modules}
    old_usage, old_event = repo.record_llm_usage, repo.record_llm_request_event
    lock = threading.Lock()
    accounting = {"calls": initial_calls, "actual": initial_cost, "reserved": 0.0}

    def wrapped(original):
        def call(client, prompt, **kwargs):
            input_tokens = max(1, len(str(prompt)) // 4)
            output_tokens = max(1, int(kwargs.get("max_tokens") or 4096))
            reservation = (
                input_tokens * float(config.OPENAI_INPUT_COST_PER_MTOK)
                + output_tokens * float(config.OPENAI_OUTPUT_COST_PER_MTOK)
            ) / 1_000_000
            with lock:
                if accounting["calls"] + 1 > MAX_CALLS:
                    raise BudgetStop("provider request ceiling reached before dispatch")
                if accounting["actual"] + accounting["reserved"] + reservation > MAX_COST:
                    raise BudgetStop("estimated cost ceiling reached before dispatch")
                accounting["calls"] += 1
                accounting["reserved"] += reservation
            began = time.perf_counter()
            record = {
                "provider": getattr(client, "provider_name", None),
                "model": getattr(client, "model", None),
                "agent": kwargs.get("agent"),
                "operation": kwargs.get("operation"),
                "prompt": str(prompt),
                "system_prompt": kwargs.get("system_prompt"),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "reserved_cost_usd": reservation,
            }
            try:
                result = original(client, prompt, **kwargs)
                record.update(status="success", response=deepcopy(result))
                return result
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__, error=str(exc)[:4000])
                raise
            finally:
                usage = deepcopy(getattr(client, "last_usage", None))
                actual = float((usage or {}).get("estimated_cost_usd") or 0.0)
                with lock:
                    accounting["reserved"] -= reservation
                    accounting["actual"] += actual
                    records.append({
                        **record,
                        "duration_seconds": round(time.perf_counter() - began, 6),
                        "usage": usage,
                        "request_id": getattr(client, "last_request_id", None),
                    })
        return call

    try:
        for module, original in originals.items():
            module.request_structured = wrapped(original)
        repo.record_llm_usage = lambda *args, **kwargs: None
        repo.record_llm_request_event = lambda *args, **kwargs: None
        yield accounting
    finally:
        for module, original in originals.items():
            module.request_structured = original
        repo.record_llm_usage, repo.record_llm_request_event = old_usage, old_event


def run(*, preflight: bool = False) -> Path | dict[str, Any]:
    payload, rows = _load_fixture()
    state, plan_by_slot, sources = _build_state(payload, rows)
    summary = {
        "fixture": str(FIXTURE),
        "fixture_sha256": _sha256(FIXTURE),
        "slots": list(SLOTS),
        "evidence_chunks": len(sources),
        "max_calls": MAX_CALLS,
        "max_cost_usd": MAX_COST,
        "production_functions": [
            "validate_exam_multi", "repair_invalid_questions", "validate_generated_questions"
        ],
        "downstream_disabled": ["Duplicate V2", "Final Assembly"],
    }
    if preflight:
        return summary

    out = ROOT / "data/storage/stage_test_checkpoints" / f"three-question-live-regression-{uuid.uuid4().hex}"
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "preflight.json", summary)
    requests: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        with _capture(requests) as accounting:
            exam = deepcopy(state["generated_exams"][0])
            report = validate_exam_multi(
                exam,
                plan_by_slot=plan_by_slot,
                source_by_chunk=sources,
                domain="STEM",
                cache_scope=f"{state['job_id']}-initial",
            )
            state["validation_reports"] = [report]
            state["repair_manifest"] = _manifest_from_report(report, exam, plan_by_slot)
            initial = deepcopy(report)
            rounds = 0
            while rounds < 8:
                rounds += 1
                update = repair_invalid_questions(state)
                state = {**state, **update}
                if not state.get("repair_candidates"):
                    break
                update = validate_generated_questions(state)
                state = {**state, **update}
                if not state.get("repair_candidates") and all(
                    row.get("status") in {"accepted", "terminal_failure", "validation_blocked"}
                    for row in state.get("repair_manifest") or []
                ):
                    break
            result = {
                "status": "complete",
                "rounds": rounds,
                "initial_validation": initial,
                "final_state": state,
                "requests": requests,
                "accounting": accounting,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
    except Exception as exc:
        result = {
            "status": "stopped",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "final_state": state,
            "requests": requests,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    _write(out / "live_results.json", result)
    return out


def run_tf5_mcq9(*, preflight: bool = False) -> Path | dict[str, Any]:
    """Paid two-question regression with an explicitly labeled MCQ9 diagnostic."""
    slots = ("m1_true_false_5", "m1_mcq_9")
    payload, rows = _load_fixture(slots)
    state, plan_by_slot, sources = _build_state(payload, rows)
    summary = {
        "fixture": str(FIXTURE), "fixture_sha256": _sha256(FIXTURE),
        "slots": list(slots), "evidence_chunks": len(sources),
        "max_calls": MAX_CALLS, "max_cost_usd": MAX_COST,
        "production_functions": [
            "validate_exam_multi", "repair_invalid_questions", "validate_generated_questions",
            "repair_question_specialized",
        ],
        "downstream_disabled": ["Duplicate V2", "Final Assembly"],
    }
    if preflight:
        return summary
    out = ROOT / "data/storage/stage_test_checkpoints" / f"tf5-mcq9-live-regression-{uuid.uuid4().hex}"
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "preflight.json", summary)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        with _capture(records) as accounting:
            exam = deepcopy(state["generated_exams"][0])
            initial = validate_exam_multi(
                exam, plan_by_slot=plan_by_slot, source_by_chunk=sources,
                domain="STEM", cache_scope=f"{state['job_id']}-initial",
            )
            state["validation_reports"] = [initial]
            state["repair_manifest"] = _manifest_from_report(initial, exam, plan_by_slot)
            rounds = 0
            while rounds < 8:
                rounds += 1
                before = json.dumps({
                    "manifest": state.get("repair_manifest"),
                    "memory": state.get("repair_replacement_memory"),
                }, sort_keys=True, default=str)
                state = {**state, **repair_invalid_questions(state)}
                if state.get("repair_candidates"):
                    state = {**state, **validate_generated_questions(state)}
                active = [row for row in state.get("repair_manifest") or [] if row.get("status") not in {
                    "accepted", "terminal_failure", "validation_blocked",
                }]
                after = json.dumps({
                    "manifest": state.get("repair_manifest"),
                    "memory": state.get("repair_replacement_memory"),
                }, sort_keys=True, default=str)
                if not active or (before == after and not state.get("repair_candidates")):
                    break

            # Phase B is deliberately separate from independent live detection.
            mcq_row = next(row for row in rows if row["slot_id"] == "m1_mcq_9")
            mcq_exam = {"model_number": 1, "questions": {"mcq": [deepcopy(mcq_row["question"])]}}
            known_verdict = {
                "question_id": mcq_row["question"].get("question_id"),
                "action": "FIX_OPTIONS", "failure_code": "ambiguous_option",
                "fields_to_fix": ["options.D"],
                "reason": "Known historical diagnostic: D duplicates B while contradicting variable definitions.",
                "expected_fix": "Replace only D; preserve A, B, C and correct answer B.",
                "required_fixes": [{
                    "issue_id": "live-diagnostic-mcq9-options-d",
                    "validator": "answer_correctness", "failure_code": "ambiguous_option",
                    "fields_to_fix": ["options.D"],
                    "reason": "D duplicates B with contradictory variable definitions.",
                    "expected_fix": "Replace only D with a distinct incorrect distractor.",
                }],
            }
            mcq_initial = next((row for row in initial.get("verdicts") or []
                                if row.get("question_id") == known_verdict["question_id"]), {})
            diagnostic = {"label": "KNOWN_FINDING_REPAIR_ROUTING_DIAGNOSTIC_NOT_LIVE_DETECTION"}
            # Always test the new staging path separately, even if detection passed.
            outcome = repair_question_specialized(
                exam=mcq_exam, verdict=known_verdict, qtype="mcq",
                plan_item=plan_by_slot["m1_mcq_9"], source_chunks={
                    key: sources[key] for key in mcq_row.get("evidence_chunk_ids") or [] if key in sources
                }, domain="STEM", recovery_strategy="targeted_repair",
            )
            diagnostic["outcome"] = outcome
            diagnostic["original"] = deepcopy(mcq_row["question"])
            diagnostic["candidate"] = deepcopy(mcq_exam["questions"]["mcq"][0])
            diagnostic["initial_live_detection"] = deepcopy(mcq_initial)
            if outcome.get("repaired_ids"):
                diagnostic["full_revalidation"] = validate_exam_multi(
                    mcq_exam, plan_by_slot={"m1_mcq_9": plan_by_slot["m1_mcq_9"]},
                    source_by_chunk=sources, domain="STEM",
                    question_ids=[str(known_verdict["question_id"])],
                    revalidation_context={str(known_verdict["question_id"]): {
                        "original_question": deepcopy(mcq_row["question"]),
                        "original_failure_findings": deepcopy(known_verdict["required_fixes"]),
                        "force_full_revalidation": True,
                    }}, cache_scope=f"{state['job_id']}-mcq9-diagnostic",
                )
            result = {
                "status": "complete", "rounds": rounds, "initial_validation": initial,
                "final_state": state, "mcq9_diagnostic": diagnostic,
                "requests": records, "accounting": accounting,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
    except Exception as exc:
        result = {
            "status": "stopped", "error_type": type(exc).__name__, "error": str(exc),
            "final_state": state, "requests": records,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    _write(out / "live_results.json", result)
    return out


def resume(previous_directory: Path) -> Path:
    previous_path = previous_directory / "live_results.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    state = previous["final_state"]
    previous_requests = list(previous.get("requests") or [])
    previous_cost = sum(
        float((record.get("usage") or {}).get("estimated_cost_usd") or 0.0)
        for record in previous_requests
    )
    new_requests: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        with _capture(
            new_requests,
            initial_calls=len(previous_requests),
            initial_cost=previous_cost,
        ) as accounting:
            rounds = 0
            while rounds < 7:
                rounds += 1
                before = json.dumps(
                    {
                        "manifest": state.get("repair_manifest"),
                        "memory": state.get("repair_replacement_memory"),
                    },
                    sort_keys=True,
                    default=str,
                )
                state = {**state, **repair_invalid_questions(state)}
                if state.get("repair_candidates"):
                    state = {**state, **validate_generated_questions(state)}
                active = [
                    row for row in state.get("repair_manifest") or []
                    if row.get("status") not in {
                        "accepted", "terminal_failure", "validation_blocked"
                    }
                ]
                after = json.dumps(
                    {
                        "manifest": state.get("repair_manifest"),
                        "memory": state.get("repair_replacement_memory"),
                    },
                    sort_keys=True,
                    default=str,
                )
                if not active or (before == after and not state.get("repair_candidates")):
                    break
            result = {
                "status": "complete",
                "continued_from": str(previous_path),
                "rounds": rounds,
                "final_state": state,
                "new_requests": new_requests,
                "all_requests": previous_requests + new_requests,
                "accounting": accounting,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
    except Exception as exc:
        result = {
            "status": "stopped",
            "continued_from": str(previous_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "final_state": state,
            "new_requests": new_requests,
            "all_requests": previous_requests + new_requests,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    output = previous_directory / "continuation_results.json"
    _write(output, result)
    return output


def mcq9_diagnostic(previous_directory: Path) -> Path:
    payload, rows = _load_fixture()
    row = next(item for item in rows if item["slot_id"] == "m1_mcq_9")
    state, plan_by_slot, sources = _build_state(payload, [row])
    qtype = str(row["question_type"])
    exam = deepcopy(state["generated_exams"][0])
    question = deepcopy(row["question"])
    prior = json.loads((previous_directory / "continuation_results.json").read_text(encoding="utf-8"))
    prior_requests = list(prior.get("all_requests") or [])
    prior_cost = sum(
        float((record.get("usage") or {}).get("estimated_cost_usd") or 0.0)
        for record in prior_requests
    )
    requests: list[dict[str, Any]] = []
    verdict = {
        "question_id": question.get("question_id"),
        "action": "FIX_OPTIONS",
        "failure_code": "ambiguous_option",
        "fields_to_fix": ["options.D"],
        "reason": (
            "Known historical diagnostic: option D repeats the correct option B equations "
            "while attaching incompatible variable definitions."
        ),
        "expected_fix": (
            "Replace only option D with one distinct, mathematically incorrect and "
            "unambiguous distractor. Preserve A, B, C and correct_answer B."
        ),
        "required_fixes": [{
            "issue_id": "diagnostic-mcq9-options-d",
            "failure_code": "ambiguous_option",
            "fields_to_fix": ["options.D"],
            "reason": (
                "Option D duplicates B's equations but contradicts the variable definitions."
            ),
            "expected_fix": "Replace only option D; preserve A, B, C and the key B.",
        }],
    }
    began = time.perf_counter()
    try:
        with _capture(
            requests,
            initial_calls=len(prior_requests),
            initial_cost=prior_cost,
        ) as accounting:
            outcome = repair_question_specialized(
                exam=exam,
                verdict=verdict,
                qtype=qtype,
                plan_item=plan_by_slot["m1_mcq_9"],
                source_chunks=sources,
                domain="STEM",
                recovery_strategy="targeted_repair",
            )
            report = None
            if outcome.get("repaired_ids"):
                report = validate_exam_multi(
                    exam,
                    plan_by_slot=plan_by_slot,
                    source_by_chunk=sources,
                    domain="STEM",
                    cache_scope=f"{state['job_id']}-mcq9-diagnostic",
                )
            result = {
                "status": "complete",
                "label": "KNOWN_FINDING_REPAIR_ROUTING_DIAGNOSTIC_NOT_LIVE_DETECTION",
                "known_finding": verdict,
                "original": question,
                "repair_outcome": outcome,
                "candidate": next(iter(exam["questions"][qtype]), None),
                "full_revalidation": report,
                "new_requests": requests,
                "all_requests": prior_requests + requests,
                "accounting": accounting,
                "elapsed_seconds": round(time.perf_counter() - began, 6),
            }
    except Exception as exc:
        result = {
            "status": "stopped",
            "label": "KNOWN_FINDING_REPAIR_ROUTING_DIAGNOSTIC_NOT_LIVE_DETECTION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "new_requests": requests,
            "elapsed_seconds": round(time.perf_counter() - began, 6),
        }
    output = previous_directory / "mcq9_repair_routing_diagnostic.json"
    _write(output, result)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mcq9-diagnostic", type=Path)
    parser.add_argument("--tf5-mcq9", action="store_true")
    args = parser.parse_args()
    if args.tf5_mcq9:
        value = run_tf5_mcq9(preflight=args.preflight)
    elif args.mcq9_diagnostic:
        value = mcq9_diagnostic(args.mcq9_diagnostic)
    elif args.resume:
        value = resume(args.resume)
    else:
        value = run(preflight=args.preflight)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str) if isinstance(value, dict) else value)
