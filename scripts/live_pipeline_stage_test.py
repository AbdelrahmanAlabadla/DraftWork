"""Controlled, stage-by-stage live pipeline runner.

This module deliberately does not use the production LangGraph.  It invokes
the existing production node functions in an explicit order so a paid test can
stop after Generation and retain a filesystem-only checkpoint for a later,
separate Validation run.  It never creates jobs, exams, audit events, or
catalog artifacts in PostgreSQL.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import re
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config, db
from app.api import repositories
from app.online.assessment_catalog import prepare_assessment_catalog
from app.online.difficulty_adjustment import adjust_plan_difficulty
from app.online.exam_builder import (
    generate_exams_node,
    immutable_exam_request,
    request_fingerprint,
)
from app.online.eval_stats import create_pipeline_eval
from app.online.lifecycle import initialize_slot_lifecycle
from app.online.planner import MAX_PLAN_ATTEMPTS, plan_exams, repair_plan
from app.online.retrieval import retrieve_context


DEFAULT_DOCUMENT_ID = "df85570b-700e-42bb-9ae0-6dc36884df4d"
DEFAULT_TASKS = (("mcq", 1), ("true_false", 1), ("equation", 1), ("word_problem", 1))
CHECKPOINT_VERSION = 1
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


class StageTestError(RuntimeError):
    """A controlled-stage precondition failed before downstream work began."""


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_root(value: str | Path | None) -> Path:
    return Path(value) if value else Path(config.LOCAL_STORAGE_ROOT) / "stage_test_checkpoints"


def _run_directory(root: Path, run_id: str) -> Path:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise StageTestError("run_id must contain only letters, numbers, underscores, or hyphens")
    return root / run_id


def _selected_child_ids(document_id: str) -> list[str]:
    with db.connection() as conn:
        rows = conn.execute(
            """SELECT chunk_id FROM document_chunks
               WHERE document_id = %s AND parent_chunk_id IS NOT NULL
               ORDER BY chunk_order, chunk_id""",
            (document_id,),
        ).fetchall()
    return [str(row[0]) for row in rows]


def _document_state(document_id: str, tasks: list[tuple[str, int]], num_models: int, difficulty: str) -> dict[str, Any]:
    document = repositories.get_document(document_id)
    if document is None:
        raise StageTestError(f"Document {document_id} does not exist")
    if str(document.get("status")) != "ready":
        raise StageTestError(f"Document {document_id} is not ready")

    stats = dict(document.get("stats") or {})
    selected_child_ids = _selected_child_ids(document_id)
    if not selected_child_ids:
        raise StageTestError("No stored child chunks are available for the document")

    run_id = f"stage-test-{uuid.uuid4().hex}"
    immutable = immutable_exam_request(
        job_id=None,
        document_id=document_id,
        tasks=tasks,
        num_models=num_models,
        selected_child_ids=selected_child_ids,
        difficulty=difficulty,
        processed_document_key=str(stats.get("processed_document_key") or "") or None,
        domain_type=str(stats.get("domain_type") or "") or None,
        subject=str(stats.get("subject") or "") or None,
        document_language=str(stats.get("document_language") or "") or None,
    )
    return {
        "run_id": run_id,
        "document_id": document_id,
        "tasks": tasks,
        "num_models": num_models,
        "difficulty": difficulty,
        "selected_child_ids": selected_child_ids,
        "session_id": str(document.get("session_id") or "") or None,
        "index_job_id": str(document.get("active_index_job_id") or "") or None,
        "processed_document_key": str(stats.get("processed_document_key") or "") or None,
        "domain_type": str(stats.get("domain_type") or "") or None,
        "subject": str(stats.get("subject") or "") or None,
        "document_language": str(stats.get("document_language") or "") or None,
        "immutable_request": immutable,
        "request_fingerprint": request_fingerprint(immutable),
        "plans": [],
        "assessment_catalog": {},
        "assessment_catalog_key": "",
        "generated_exams": [],
        "warnings": [],
        "plan_errors": [],
        "planning_issues": [],
        "difficulty_adjustment_history": [],
        "difficulty_adjustment_attempts": {},
        "mcq_task_reassignments": [],
        "fitb_recovery_history": [],
        "slot_lifecycle": initialize_slot_lifecycle(tasks, num_models),
        "eval_stats": create_pipeline_eval(tasks, num_models),
        "is_cancelled": lambda: False,
    }


def _update(state: dict[str, Any], result: dict[str, Any], stage: str) -> None:
    if result.get("error"):
        raise StageTestError(f"{stage} failed: {result['error']}")
    state.update(result)


def _checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "document_id", "tasks", "num_models", "difficulty", "selected_child_ids",
        "session_id", "index_job_id", "processed_document_key", "domain_type", "subject",
        "document_language", "immutable_request", "request_fingerprint", "retrieved_chunks",
        "context", "planner_context", "assessment_catalog", "assessment_catalog_key",
        "assessment_catalog_cache_hit", "assessment_compiler_cache_hit", "plans",
        "assessment_profile", "planning_issues", "plan_errors", "plan_attempts",
        "difficulty_adjustment_history", "difficulty_adjustment_attempts", "generated_exams",
        "warnings", "mcq_task_reassignments", "fitb_recovery_history", "slot_lifecycle",
        "eval_stats", "validation_reports", "repair_manifest", "validated_models",
        "validation_call_logs", "repair_candidates", "pending_revalidation_ids",
        "duplicate_reports", "duplicate_scan_count", "pending_duplicate_groups",
    )
    return {field: deepcopy(state.get(field)) for field in fields if field in state}


def _safe_request_value(value: Any) -> Any:
    """Return a JSON-safe copy of public structured request/response data."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        json.dumps(value, ensure_ascii=False, default=_json_default)
        return deepcopy(value)
    except (TypeError, ValueError):
        return {"unserializable_representation": repr(value)}


@contextmanager
def _production_domain(state: dict[str, Any]):
    """Apply the same model-routing context that the production graph applies."""
    from app.llm.factory import reset_online_assessment_domain, set_online_assessment_domain

    domain = (state.get("assessment_profile") or {}).get("domain") or state.get("domain_type")
    token = set_online_assessment_domain(str(domain or ""))
    try:
        yield
    finally:
        reset_online_assessment_domain(token)


@contextmanager
def _isolate_llm_telemetry():
    """Keep standalone paid-test telemetry out of production database tables."""
    from app.api import repositories as repository_module

    originals = {
        "record_llm_usage": repository_module.record_llm_usage,
        "record_llm_request_event": repository_module.record_llm_request_event,
    }

    def discard_persistence(*_args: Any, **_kwargs: Any) -> None:
        return None

    try:
        repository_module.record_llm_usage = discard_persistence  # type: ignore[assignment]
        repository_module.record_llm_request_event = discard_persistence  # type: ignore[assignment]
        yield
    finally:
        repository_module.record_llm_usage = originals["record_llm_usage"]  # type: ignore[assignment]
        repository_module.record_llm_request_event = originals["record_llm_request_event"]  # type: ignore[assignment]


@contextmanager
def _capture_structured_requests(records: list[dict[str, Any]]):
    """Temporarily observe production structured calls without changing their behavior.

    The wrappers live only in this standalone process and call the original
    production helper unchanged.  They capture prompts and parsed structured
    responses into the isolated checkpoint for later generation review.
    """
    from app.online import exam_builder as exam_builder_module
    from app.online import duplicate_judge as duplicate_judge_module
    from app.online import multi_repair as multi_repair_module
    from app.online import multi_validator as multi_validator_module
    from app.online import planner as planner_module

    targets = (
        exam_builder_module, planner_module, multi_validator_module,
        multi_repair_module, duplicate_judge_module,
    )
    originals = {module: module.request_structured for module in targets}

    def make_wrapper(original: Callable[..., Any], component: str) -> Callable[..., Any]:
        def observed(client: Any, prompt: Any, **kwargs: Any) -> Any:
            started = datetime.now(timezone.utc)
            started_clock = time.perf_counter()
            record: dict[str, Any] = {
                "component": component,
                "started_at": started.isoformat(),
                "provider": getattr(client, "provider_name", None) or getattr(client, "provider", None),
                "model": getattr(client, "model", None) or getattr(client, "model_name", None),
                "agent": kwargs.get("agent"),
                "operation": kwargs.get("operation"),
                "schema_name": kwargs.get("schema_name"),
                "system_prompt": _safe_request_value(kwargs.get("system_prompt")),
                "user_prompt": _safe_request_value(prompt),
                "expected_ids": _safe_request_value(kwargs.get("expected_ids")),
                "expected_id_field": kwargs.get("expected_id_field"),
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
                "timeout": kwargs.get("timeout"),
            }
            try:
                response = original(client, prompt, **kwargs)
                record["status"] = "success"
                record["parsed_response"] = _safe_request_value(response)
                return response
            except Exception as exc:
                record["status"] = "error"
                record["error_type"] = type(exc).__name__
                record["error_message"] = str(exc)
                raise
            finally:
                usage = getattr(client, "last_usage", None)
                record["duration_seconds"] = round(time.perf_counter() - started_clock, 6)
                record["usage"] = _safe_request_value(usage) if usage is not None else None
                record["transient_retries"] = getattr(client, "last_transient_retries", None)
                record["structured_attempts"] = getattr(client, "last_structured_attempts", None)
                record["structured_attempt_details"] = _safe_request_value(
                    getattr(client, "last_structured_attempt_details", None)
                )
                record["provider_request_id"] = getattr(client, "last_request_id", None)
                records.append(record)

        return observed

    try:
        for module, original in originals.items():
            module.request_structured = make_wrapper(original, module.__name__)  # type: ignore[assignment]
        yield
    finally:
        for module, original in originals.items():
            module.request_structured = original  # type: ignore[assignment]


def _save_generation_checkpoint(
    state: dict[str, Any], root: Path, audit_events: list[dict[str, Any]], stage_log: list[dict[str, Any]],
    llm_requests: list[dict[str, Any]] | None = None,
) -> Path:
    run_dir = _run_directory(root, str(state["run_id"]))
    checkpoint_path = run_dir / "generation_checkpoint.json"
    checkpoint_state = _checkpoint_state(state)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "mode": "generation_only",
        "status": "generation_complete_awaiting_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": state["run_id"],
        "state_sha256": hashlib.sha256(
            json.dumps(checkpoint_state, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
        ).hexdigest(),
        "state": checkpoint_state,
        "generation_audit_events": audit_events,
        "stage_log": stage_log,
        "llm_requests": llm_requests or [],
        "provider_request_information": {
            "captured_request_count": len(llm_requests or []),
            "note": "Captured structured prompts, parsed responses, public usage telemetry, and available provider request IDs. Hidden model reasoning is never captured.",
            "job_id": None,
        },
    }
    _atomic_json_write(checkpoint_path, payload)
    return checkpoint_path


def load_generation_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise StageTestError("Unsupported stage-test checkpoint version")
    if payload.get("mode") != "generation_only":
        raise StageTestError("Checkpoint is not a generation-only checkpoint")
    if payload.get("status") != "generation_complete_awaiting_validation":
        raise StageTestError("Checkpoint is not waiting for Validation")
    state = dict(payload.get("state") or {})
    if not state.get("generated_exams") or not state.get("plans"):
        raise StageTestError("Checkpoint does not contain complete generated questions and plans")
    expected = str(payload.get("state_sha256") or "")
    actual = hashlib.sha256(
        json.dumps(state, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise StageTestError("Checkpoint integrity digest does not match its state")
    return payload


def run_generation_only(
    *, document_id: str = DEFAULT_DOCUMENT_ID, tasks: list[tuple[str, int]] | None = None,
    num_models: int = 1, difficulty: str = "mix", checkpoint_root: str | Path | None = None,
) -> Path:
    """Run only production retrieval, cache-backed planning, adjustment, and Generation.

    No validator, repair, duplicate, or final-assembly function is imported or called here.
    """
    requested_tasks = list(tasks or DEFAULT_TASKS)
    if num_models != 1:
        raise StageTestError("Controlled stage tests currently require exactly one model")
    if not requested_tasks or any(not qtype or int(count) <= 0 for qtype, count in requested_tasks):
        raise StageTestError("At least one positive question count is required")

    state = _document_state(document_id, requested_tasks, num_models, difficulty)
    audit_events: list[dict[str, Any]] = []
    stage_log: list[dict[str, Any]] = []
    llm_requests: list[dict[str, Any]] = []

    def audit(stage: str, event_type: str, payload: dict[str, Any]) -> None:
        audit_events.append({"stage": stage, "event_type": event_type, "payload": deepcopy(payload)})

    state["audit_callback"] = audit
    state["progress_callback"] = lambda stage, progress: stage_log.append(
        {"stage": stage, "progress": int(progress), "at": datetime.now(timezone.utc).isoformat()}
    )

    _update(state, retrieve_context(state), "retrieval")
    _update(state, prepare_assessment_catalog(state), "assessment catalog")
    if not state.get("assessment_catalog_cache_hit"):
        raise StageTestError("Refusing live test: the verified assessment catalog was not a cache hit")

    with _production_domain(state), _isolate_llm_telemetry(), _capture_structured_requests(llm_requests):
        _update(state, plan_exams(state), "planning")
        while state.get("plan_errors") and int(state.get("plan_attempts") or 0) < MAX_PLAN_ATTEMPTS:
            _update(state, repair_plan(state), "plan repair")
        if state.get("plan_errors"):
            raise StageTestError("Generation was not started because the production plan remained invalid")

        _update(state, adjust_plan_difficulty(state), "difficulty adjustment")
        if state.get("plan_errors"):
            raise StageTestError("Generation was not started because difficulty adjustment invalidated the plan")

        _update(state, generate_exams_node(state), "generation")
    if not any(exam.get("questions") for exam in state.get("generated_exams") or []):
        raise StageTestError("Production generator returned no questions")
    return _save_generation_checkpoint(
        state, _checkpoint_root(checkpoint_root), audit_events, stage_log, llm_requests,
    )


def run_validation_only(checkpoint_path: str | Path, *, checkpoint_root: str | Path | None = None) -> Path:
    """Run production role validation without Duplicate V2, Repair, or Assembly."""
    from app.online.multi_validator import validate_exam_multi
    from app.online.validator import _manifest_from_report, _plan_context_for_model

    source = load_generation_checkpoint(checkpoint_path)
    state = deepcopy(source["state"])
    state["is_cancelled"] = lambda: False
    state["progress_callback"] = lambda _stage, _progress: None
    state["audit_callback"] = lambda _stage, _event, _payload: None
    validation_requests: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    domain = str((state.get("assessment_profile") or {}).get("domain") or state.get("domain_type") or "")
    with _production_domain(state), _isolate_llm_telemetry(), _capture_structured_requests(validation_requests):
        for exam in deepcopy(state.get("generated_exams") or []):
            model_number = int(exam.get("model_number") or 1)
            plan_by_slot, source_by_chunk = _plan_context_for_model(state, model_number)
            report = validate_exam_multi(
                exam,
                plan_by_slot=plan_by_slot,
                source_by_chunk=source_by_chunk,
                domain=domain,
                cache_scope=f"stage-test-validation-{source['run_id']}",
            )
            reports.append(report)
            manifests.extend(_manifest_from_report(report, exam, plan_by_slot))

    state.update({
        "validation_reports": reports,
        "repair_manifest": manifests,
        "validated_models": [
            int(report.get("model_number") or 1)
            for report in reports if report.get("all_pass")
        ],
        "validation_call_logs": [
            trace for report in reports
            for trace in (report.get("validation_call_logs") or [])
        ],
        # This controlled entry point intentionally omits Duplicate V2. It must
        # be run only in the explicitly authorized later duplicate stage.
        "duplicate_reports": [],
        "duplicate_scan_count": 0,
        "pending_duplicate_groups": [],
        "repair_candidates": [],
        "pending_revalidation_ids": {},
    })
    destination = _run_directory(_checkpoint_root(checkpoint_root), str(source["run_id"])) / "validation_checkpoint.json"
    checkpoint_state = _checkpoint_state(state)
    _atomic_json_write(destination, {
        "checkpoint_version": CHECKPOINT_VERSION,
        "mode": "validation_only",
        "status": "validation_complete_awaiting_repair",
        "source_generation_checkpoint": str(Path(checkpoint_path).resolve()),
        "source_generation_state_sha256": source.get("state_sha256"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_sha256": hashlib.sha256(
            json.dumps(checkpoint_state, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
        ).hexdigest(),
        "state": checkpoint_state,
        "validation_requests": validation_requests,
        "provider_request_information": {
            "captured_request_count": len(validation_requests),
            "note": "Captured production Validator prompts, parsed responses, per-attempt diagnostics, and public usage telemetry. No Repair, Revalidation, Duplicate V2, or Final Assembly ran.",
            "job_id": None,
        },
    })
    return destination


def _load_validation_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("mode") != "validation_only":
        raise StageTestError("Checkpoint is not a validation-only checkpoint")
    state = dict(payload.get("state") or {})
    expected = str(payload.get("state_sha256") or "")
    actual = hashlib.sha256(
        json.dumps(state, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise StageTestError("Validation checkpoint integrity digest does not match its state")
    return payload


def run_grounding_only(
    generation_checkpoint_path: str | Path,
    *,
    question_id: str = "model1_mcq_1",
    validation_checkpoint_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Run exactly one production Grounding request for one preserved question."""
    from app.llm.factory import create_review_client
    from app.online.multi_validator import _call_once, _grounding_contract, protect_question
    from app.online.validator import _iter_questions, _plan_context_for_model

    source = load_generation_checkpoint(generation_checkpoint_path)
    validation_path = Path(validation_checkpoint_path) if validation_checkpoint_path else (
        Path(generation_checkpoint_path).with_name("validation_checkpoint.json")
    )
    validation_source = _load_validation_checkpoint(validation_path)
    state = deepcopy(source["state"])
    chosen: tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]] | None = None
    for exam in state.get("generated_exams") or []:
        model_number = int(exam.get("model_number") or 1)
        plan_by_slot, source_by_chunk = _plan_context_for_model(state, model_number)
        for qtype, container, raw in _iter_questions(exam.get("questions") or {}):
            if str(raw.get("question_id") or "") == question_id:
                chosen = (exam, qtype, raw, {"plans": plan_by_slot, "source": source_by_chunk, "container": container})
                break
        if chosen:
            break
    if not chosen:
        raise StageTestError(f"Question {question_id!r} is absent from the Generation checkpoint")

    _exam, qtype, raw, context = chosen
    if qtype != "mcq" or question_id != "model1_mcq_1":
        raise StageTestError("This controlled retest is restricted to the original MCQ slot")
    task = context["plans"].get(str(raw.get("slot_id") or "")) or {}
    item = protect_question(
        qtype, raw, task, context["source"],
        word_bank=(context["container"].get("word_bank") or []) if isinstance(context["container"], dict) else (),
    )
    if item.errors:
        raise StageTestError("Original MCQ cannot be safely grounded: " + "; ".join(item.errors))
    contract = _grounding_contract(item)
    expected_contract = {
        "task_operation": "optimize_linear_function_at_vertices",
        "selected_variation_id": "variation_13007c8931b963a0",
        "numeric_values_may_change": True,
        "exact_source_instance": False,
        "permitted_optimization_direction": "both",
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise StageTestError(f"Original MCQ grounding contract mismatch: {contract}")

    requests: list[dict[str, Any]] = []
    domain = str((state.get("assessment_profile") or {}).get("domain") or state.get("domain_type") or "")
    start = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    with _production_domain(state), _isolate_llm_telemetry(), _capture_structured_requests(requests):
        valid, errors, trace = _call_once(
            "grounding", [item],
            lambda _role: create_review_client(domain=domain, stage="grounding_validator"),
            attempt=1, phase="targeted_grounding_retest",
        )
    completed = datetime.now(timezone.utc)
    if len(requests) != 1:
        raise StageTestError(f"Grounding-only test made {len(requests)} structured requests, expected exactly one")
    destination = _run_directory(_checkpoint_root(checkpoint_root), str(source["run_id"])) / "grounding_retest_checkpoint.json"
    result = valid.get(question_id)
    _atomic_json_write(destination, {
        "checkpoint_version": CHECKPOINT_VERSION,
        "mode": "grounding_only_targeted_retest",
        "status": "grounding_complete_awaiting_review",
        "created_at": completed.isoformat(),
        "source_generation_checkpoint": str(Path(generation_checkpoint_path).resolve()),
        "source_generation_state_sha256": source.get("state_sha256"),
        "source_validation_checkpoint": str(validation_path.resolve()),
        "source_validation_state_sha256": validation_source.get("state_sha256"),
        "question_id": question_id,
        "question_type": qtype,
        "task_id": task.get("task_id"),
        "selected_variation_id": task.get("selected_variation_id"),
        "grounding_contract": contract,
        "question": deepcopy(raw),
        "grounding_verdict": result,
        "grounding_errors": errors,
        "production_trace": trace,
        "captured_requests": requests,
        "execution": {
            "started_at": start.isoformat(), "completed_at": completed.isoformat(),
            "wall_clock_seconds": round(time.perf_counter() - started_clock, 6),
            "scope": "one Grounding Validator request only; no lifecycle status was changed",
        },
    })
    return destination


def _assigned_sources(task: dict[str, Any], source_by_chunk: dict[str, str]) -> dict[str, str]:
    source_ids = list(dict.fromkeys(str(value) for value in (
        task.get("source_evidence_ids") or [
            task.get("source_chunk_id"), *(task.get("support_chunk_ids") or []),
        ]
    ) if value))
    missing = [chunk_id for chunk_id in source_ids if chunk_id not in source_by_chunk]
    if missing:
        raise StageTestError(f"Assigned source chunks are unavailable: {missing}")
    return {chunk_id: source_by_chunk[chunk_id] for chunk_id in source_ids}


def run_repair_revalidate_once(
    generation_checkpoint_path: str | Path,
    *,
    validation_checkpoint_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Repair only the preserved TF once, then invoke production revalidation once.

    This deliberately has no recovery loop.  A failed or blocked revalidation
    is evidence for review, never input to another repair/replacement attempt.
    """
    from app.llm.factory import create_review_client
    from app.online.multi_repair import repair_question_specialized
    from app.online.multi_validator import validate_exam_multi
    from app.online.validator import _iter_questions, _plan_context_for_model

    source = load_generation_checkpoint(generation_checkpoint_path)
    validation_path = Path(validation_checkpoint_path) if validation_checkpoint_path else (
        Path(generation_checkpoint_path).with_name("validation_checkpoint.json")
    )
    validation_source = _load_validation_checkpoint(validation_path)
    state = deepcopy(source["state"])
    original_report = next(iter((validation_source.get("state") or {}).get("validation_reports") or []), None)
    if not isinstance(original_report, dict):
        raise StageTestError("Validation checkpoint has no report to repair from")
    question_id = "model1_true_false_1"
    verdict = next(
        (deepcopy(value) for value in original_report.get("verdicts") or []
         if str(value.get("question_id") or "") == question_id),
        None,
    )
    if not verdict or verdict.get("validation_status") != "FAIL":
        raise StageTestError("Original True/False question is not an eligible failed validation result")
    findings = list(verdict.get("required_fixes") or [])
    if {str(item.get("issue_id") or "") for item in findings} != {
        "model1_true_false_1-I-08d66e25f28d-1", "model1_true_false_1-I-44eb50c40f9c-1",
    }:
        raise StageTestError("Original True/False findings or issue IDs do not match the controlled retest")

    model_number = 1
    plan_by_slot, source_by_chunk = _plan_context_for_model(state, model_number)
    original_question: dict[str, Any] | None = None
    for exam in state.get("generated_exams") or []:
        if int(exam.get("model_number") or 1) != model_number:
            continue
        for qtype, _container, raw in _iter_questions(exam.get("questions") or {}):
            if qtype == "true_false" and str(raw.get("question_id") or "") == question_id:
                original_question = deepcopy(raw)
                break
    if original_question is None:
        raise StageTestError("Original True/False question is absent from the Generation checkpoint")
    task = plan_by_slot.get(str(original_question.get("slot_id") or "")) or {}
    if str(task.get("task_id") or "") != "task_e22b32e17c016bb2":
        raise StageTestError("Original True/False task binding does not match the controlled retest")
    isolated_exam = {"model_number": model_number, "questions": {"true_false": [deepcopy(original_question)]}}
    isolated_plan = {str(original_question["slot_id"]): deepcopy(task)}
    assigned_sources = _assigned_sources(task, source_by_chunk)
    domain = str((state.get("assessment_profile") or {}).get("domain") or state.get("domain_type") or "")
    requests: list[dict[str, Any]] = []
    start = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    with _production_domain(state), _isolate_llm_telemetry(), _capture_structured_requests(requests):
        repair = repair_question_specialized(
            exam=isolated_exam, verdict=verdict, qtype="true_false", plan_item=isolated_plan[str(original_question["slot_id"])],
            source_chunks=assigned_sources, domain=domain,
            client_factory=lambda agent: create_review_client(domain=domain, stage=f"{agent}_repairer"),
        )
        if repair.get("repair_route") != ["task_alignment"]:
            raise StageTestError(f"Unexpected repair route: {repair.get('repair_route')}")
        repair_request_count = len(requests)
        run_revalidation = bool(repair.get("repaired_ids")) and not repair.get("failed_ids")
        revalidation: dict[str, Any] | None = None
        if run_revalidation:
            metadata = deepcopy((repair.get("repair_metadata") or {}).get(question_id) or {})
            revalidation_context = {
                question_id: {
                    "original_question": deepcopy(original_question),
                    "original_failure_findings": deepcopy(findings),
                    "repair_metadata": metadata,
                    "previous_rejections": [],
                    "candidate_id": "stage-test-one-pass-tf-repair",
                    "candidate_fingerprint": None,
                }
            }
            revalidation = validate_exam_multi(
                isolated_exam, plan_by_slot=isolated_plan, source_by_chunk=assigned_sources,
                domain=domain, question_ids=[question_id],
                revalidation_context=revalidation_context,
                cache_scope=f"stage-test-one-pass-revalidation-{source['run_id']}",
            )
    completed = datetime.now(timezone.utc)
    run_dir = _run_directory(_checkpoint_root(checkpoint_root), str(source["run_id"]))
    repair_path = run_dir / "true_false_repair_checkpoint.json"
    repaired_question = next(
        (deepcopy(raw) for qtype, _container, raw in _iter_questions(isolated_exam.get("questions") or {})
         if qtype == "true_false" and str(raw.get("question_id") or "") == question_id),
        None,
    )
    common = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "source_generation_checkpoint": str(Path(generation_checkpoint_path).resolve()),
        "source_generation_state_sha256": source.get("state_sha256"),
        "source_validation_checkpoint": str(validation_path.resolve()),
        "source_validation_state_sha256": validation_source.get("state_sha256"),
        "question_id": question_id,
        "task_id": task.get("task_id"),
        "selected_variation_id": task.get("selected_variation_id"),
        "original_question": original_question,
        "original_findings": findings,
        "repair": repair,
        "repaired_question": repaired_question,
        "requests": requests[:repair_request_count],
        "execution": {"started_at": start.isoformat(), "completed_at": completed.isoformat()},
    }
    _atomic_json_write(repair_path, {
        **common,
        "mode": "one_pass_true_false_repair",
        "status": "repair_complete_awaiting_one_revalidation" if run_revalidation else "repair_failed_no_revalidation",
    })
    if not run_revalidation:
        return repair_path, None

    revalidation_path = run_dir / "true_false_revalidation_checkpoint.json"
    _atomic_json_write(revalidation_path, {
        **common,
        "mode": "one_pass_true_false_revalidation",
        "status": "revalidation_complete_stop_required",
        "revalidation": revalidation,
        "requests": requests[repair_request_count:],
        "execution": {
            "started_at": start.isoformat(), "completed_at": completed.isoformat(),
            "wall_clock_seconds": round(time.perf_counter() - started_clock, 6),
            "repair_request_count": repair_request_count,
            "revalidation_request_count": len(requests) - repair_request_count,
            "hard_stop_after_revalidation": True,
        },
    })
    return repair_path, revalidation_path


def _questions_by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the immutable generated-question payloads keyed by question ID."""
    questions: dict[str, dict[str, Any]] = {}
    for exam in state.get("generated_exams") or []:
        for question_type, entries in (exam.get("questions") or {}).items():
            if isinstance(entries, dict):
                entries = entries.get("items") or []
            for entry in entries or []:
                question_id = str(entry.get("question_id") or "")
                if question_id:
                    questions[question_id] = {
                        "question_type": question_type,
                        "payload": deepcopy(entry),
                    }
    return questions


def _request_totals(requests: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "total_tokens")
    totals = {field: 0 for field in fields}
    total_cost = 0.0
    duration_seconds = 0.0
    missing_usage = 0
    for request in requests:
        usage = request.get("usage")
        if not isinstance(usage, dict):
            missing_usage += 1
            continue
        for field in fields:
            totals[field] += int(usage.get(field) or 0)
        total_cost += float(usage.get("estimated_cost_usd") or 0.0)
        duration_seconds += float(request.get("duration_seconds") or 0.0)
    return {
        "request_count": len(requests),
        **totals,
        "estimated_cost_usd": round(total_cost, 8),
        "observed_request_duration_seconds": round(duration_seconds, 6),
        "requests_missing_usage": missing_usage,
    }


def write_grounding_retest_report(checkpoint_path: str | Path) -> tuple[Path, Path]:
    """Create offline Markdown and JSON reports for one targeted Grounding run."""
    path = Path(checkpoint_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "grounding_only_targeted_retest":
        raise StageTestError("Checkpoint is not a targeted Grounding retest")
    requests = list(payload.get("captured_requests") or [])
    if len(requests) != 1:
        raise StageTestError("Grounding retest checkpoint must contain exactly one request")
    request = requests[0]
    usage = request.get("usage") or {}
    report = {
        "report_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "targeted_retest": True,
        "status": payload.get("status"),
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_generation_checkpoint": payload.get("source_generation_checkpoint"),
        "source_validation_checkpoint": payload.get("source_validation_checkpoint"),
        "question_id": payload.get("question_id"),
        "task_id": payload.get("task_id"),
        "selected_variation_id": payload.get("selected_variation_id"),
        "grounding_contract": payload.get("grounding_contract"),
        "question": payload.get("question"),
        "grounding_verdict": payload.get("grounding_verdict"),
        "grounding_errors": payload.get("grounding_errors"),
        "execution": payload.get("execution"),
        "request": request,
        "telemetry_note": "Costs are estimates from configured pricing and observed usage, not an invoice.",
    }
    json_path = path.with_name("grounding_retest_forensic_report.json")
    _atomic_json_write(json_path, report)
    verdict = payload.get("grounding_verdict") or {}
    lines = [
        "# Targeted Grounding retest forensic report",
        "",
        "- Scope: exactly one production Grounding request for the preserved MCQ. No other Validator, Generation, Repair, Revalidation, Duplicate V2, or Final Assembly ran.",
        f"- Question: `{payload.get('question_id')}`; task: `{payload.get('task_id')}`; variation: `{payload.get('selected_variation_id')}`.",
        f"- Verdict: `{verdict.get('status', 'ERROR')}`; grounding outcome: `{verdict.get('grounding_outcome')}`.",
        f"- Wall-clock duration: `{float((payload.get('execution') or {}).get('wall_clock_seconds') or 0):.6f}` seconds.",
        f"- Provider/model: `{request.get('provider')}` / `{request.get('model')}`.",
        f"- Tokens: input `{usage.get('input_tokens', 'unknown')}`, output `{usage.get('output_tokens', 'unknown')}`, cached `{usage.get('cached_tokens', 'unknown')}`, reasoning `{usage.get('reasoning_tokens', 'unknown')}`.",
        f"- Estimated cost: `${float(usage.get('estimated_cost_usd') or 0):.8f}`.",
        "",
        "## Grounding contract",
        "",
        "```json",
        json.dumps(payload.get("grounding_contract") or {}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Parsed response",
        "",
        "```json",
        json.dumps(verdict if verdict else {"errors": payload.get("grounding_errors")}, ensure_ascii=False, indent=2),
        "```",
        "",
        "The JSON companion preserves the complete system prompt, user payload, structured response, attempt diagnostics, and public usage telemetry. No credentials or headers are included.",
    ]
    markdown_path = path.with_name("grounding_retest_forensic_report.md")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path, json_path


def write_repair_revalidation_report(repair_path: str | Path, revalidation_path: str | Path | None) -> tuple[Path, Path]:
    """Write an offline combined report for the hard-stopped one-pass run."""
    repair = json.loads(Path(repair_path).read_text(encoding="utf-8"))
    revalidation = json.loads(Path(revalidation_path).read_text(encoding="utf-8")) if revalidation_path else None
    if repair.get("mode") != "one_pass_true_false_repair":
        raise StageTestError("Repair artifact is not a controlled one-pass repair")
    all_requests = list(repair.get("requests") or []) + list((revalidation or {}).get("requests") or [])
    totals = _request_totals(all_requests)
    report = {
        "report_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "one Task Alignment Repair pass followed by one production Revalidation pass at most",
        "repair_checkpoint": str(Path(repair_path).resolve()),
        "revalidation_checkpoint": str(Path(revalidation_path).resolve()) if revalidation_path else None,
        "original_question": repair.get("original_question"),
        "repaired_question": repair.get("repaired_question"),
        "original_findings": repair.get("original_findings"),
        "repair": repair.get("repair"),
        "revalidation": (revalidation or {}).get("revalidation"),
        "request_totals": totals,
        "repair_requests": repair.get("requests") or [],
        "revalidation_requests": (revalidation or {}).get("requests") or [],
        "hard_stop_after_revalidation": True,
    }
    json_path = Path(repair_path).with_name("true_false_repair_revalidation_forensic_report.json")
    _atomic_json_write(json_path, report)
    revalidation_verdict = next(iter(((revalidation or {}).get("revalidation") or {}).get("verdicts") or []), {})
    metadata = ((repair.get("repair") or {}).get("repair_metadata") or {}).get("model1_true_false_1") or {}
    lines = [
        "# Controlled True/False Repair and one-time Revalidation report",
        "",
        "- Scope: one Task Alignment Repair pass for `model1_true_false_1`; one production Revalidation pass at most; hard stop afterward.",
        f"- Repair route: `{', '.join((repair.get('repair') or {}).get('repair_route') or [])}`.",
        f"- Repair status: `{(repair.get('repair') or {}).get('route_status')}`.",
        f"- Revalidation status: `{revalidation_verdict.get('validation_status', 'not run')}`; action: `{revalidation_verdict.get('action', 'not run')}`.",
        f"- Calls: `{totals['request_count']}`; estimated cost: `${totals['estimated_cost_usd']:.8f}`; observed cumulative request duration: `{totals['observed_request_duration_seconds']:.6f}s`.",
        "",
        "## Original statement",
        "",
        str((repair.get("original_question") or {}).get("statement") or ""),
        "",
        "## Repaired statement",
        "",
        str((repair.get("repaired_question") or {}).get("statement") or "No valid candidate was produced."),
        "",
        "## Changed fields",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Revalidation verdict",
        "",
        "```json",
        json.dumps(revalidation_verdict or {"not_run": True}, ensure_ascii=False, indent=2),
        "```",
        "",
        "The JSON companion contains full captured prompts, input payloads, parsed responses, raw-response excerpts, attempt diagnostics, and public token telemetry. No additional repair or revalidation was invoked after this result.",
    ]
    markdown_path = Path(repair_path).with_name("true_false_repair_revalidation_forensic_report.md")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path, json_path


def write_validation_forensic_report(checkpoint_path: str | Path) -> tuple[Path, Path]:
    """Create local Markdown and JSON evidence from an existing validation checkpoint.

    This is deliberately offline: it reads an already-completed validation-only
    checkpoint and makes no provider, database, or pipeline calls.
    """
    path = Path(checkpoint_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "validation_only":
        raise StageTestError("Checkpoint is not a validation-only checkpoint")
    if payload.get("status") != "validation_complete_awaiting_repair":
        raise StageTestError("Validation checkpoint is not awaiting Repair")

    state = dict(payload.get("state") or {})
    expected = str(payload.get("state_sha256") or "")
    actual = hashlib.sha256(
        json.dumps(state, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise StageTestError("Validation checkpoint integrity digest does not match its state")

    requests = list(payload.get("validation_requests") or [])
    reports = list(state.get("validation_reports") or [])
    questions = _questions_by_id(state)
    verdicts = [verdict for report in reports for verdict in (report.get("verdicts") or [])]
    totals = _request_totals(requests)
    source_path = Path(str(payload.get("source_generation_checkpoint") or ""))
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else None
    report_json = {
        "report_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "validation_only",
        "status": payload["status"],
        "validation_checkpoint": str(path.resolve()),
        "validation_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validation_state_sha256": expected,
        "source_generation_checkpoint": str(source_path),
        "source_generation_checkpoint_file_sha256": source_hash,
        "source_generation_state_sha256": payload.get("source_generation_state_sha256"),
        "execution_scope": {
            "generation": "not run",
            "repair": "not run",
            "revalidation": "not run",
            "duplicate_v2": "not run",
            "final_assembly": "not run",
            "database_telemetry": "isolated; no standalone-run telemetry was persisted",
        },
        "request_totals": totals,
        "questions": questions,
        "validation_reports": reports,
        "repair_manifest": state.get("repair_manifest") or [],
        "validation_requests": requests,
        "telemetry_notes": {
            "estimated_cost": "Estimated from configured pricing and observed provider usage; it is not an invoice.",
            "reasoning": "Only provider-reported reasoning-token counts are included; hidden reasoning is not captured.",
            "raw_responses": "Parsed responses and size-bounded raw response excerpts are stored per request in structured_attempt_details.",
        },
    }
    json_path = path.with_name("validation_forensic_report.json")
    _atomic_json_write(json_path, report_json)

    lines = [
        "# Controlled live validation forensic report",
        "",
        f"- Status: `{payload['status']}`",
        f"- Validation checkpoint: `{path.resolve()}`",
        f"- Source Generation checkpoint SHA-256: `{source_hash or 'unavailable'}`",
        f"- Validation state SHA-256: `{expected}`",
        "- Scope: Validation only. Generation, Repair, Revalidation, Duplicate V2, and Final Assembly were not run.",
        "- Database telemetry was isolated for this standalone test; no production job, exam, checkpoint, or LLM telemetry record was changed.",
        "",
        "## Request telemetry",
        "",
        "| Metric | Observed value |",
        "|---|---:|",
    ]
    for label, value in (
        ("Structured validator calls", totals["request_count"]),
        ("Input tokens", totals["input_tokens"]),
        ("Output tokens", totals["output_tokens"]),
        ("Cached input tokens", totals["cached_tokens"]),
        ("Reasoning tokens", totals["reasoning_tokens"]),
        ("Total tokens", totals["total_tokens"]),
        ("Observed request duration", f"{totals['observed_request_duration_seconds']:.3f} s"),
        ("Estimated cost", f"${totals['estimated_cost_usd']:.8f}"),
        ("Requests missing usage", totals["requests_missing_usage"]),
    ):
        lines.append(f"| {label} | {value} |")
    lines.extend(["", "## Question results", ""])
    for verdict in verdicts:
        question_id = str(verdict.get("question_id"))
        source = questions.get(question_id, {})
        lines.extend([
            f"### {question_id} ({source.get('question_type', 'unknown')})",
            "",
            f"- Final Validation status: `{verdict.get('validation_status')}`; action: `{verdict.get('action')}`.",
            f"- Role statuses: `{json.dumps(verdict.get('role_statuses') or {}, ensure_ascii=False, sort_keys=True)}`.",
        ])
        if verdict.get("reason"):
            lines.append(f"- Aggregated reason: {verdict['reason']}")
        for finding in verdict.get("required_fixes") or []:
            lines.extend([
                f"- Finding `{finding.get('issue_id')}` from `{finding.get('validator')}`: `{finding.get('failure_code')}`.",
                f"  - Category: `{finding.get('category')}`; fields: `{', '.join(finding.get('fields_to_fix') or [])}`.",
                f"  - Missing information: `{json.dumps(finding.get('missing_information') or [], ensure_ascii=False)}`.",
                f"  - Evidence IDs: `{', '.join(finding.get('source_evidence_ids') or [])}`.",
                f"  - Expected fix: {finding.get('expected_fix')}",
            ])
        lines.append("")
    lines.extend([
        "## Captured request evidence",
        "",
        "The JSON companion includes every captured system prompt, user payload, parsed response, public token telemetry, provider request ID when available, and size-bounded raw structured-response excerpt. It contains no credentials or headers.",
        "",
        "| # | Validator | Operation | Provider/model | Status | Input | Output | Reasoning | Duration | Cost |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for index, request in enumerate(requests, start=1):
        usage = request.get("usage") or {}
        lines.append(
            "| {i} | {agent} | {operation} | {provider}/{model} | {status} | {input} | {output} | {reasoning} | {duration:.3f}s | ${cost:.8f} |".format(
                i=index, agent=request.get("agent") or "", operation=request.get("operation") or "",
                provider=request.get("provider") or "", model=request.get("model") or "",
                status=request.get("status") or "", input=usage.get("input_tokens", "unknown"),
                output=usage.get("output_tokens", "unknown"), reasoning=usage.get("reasoning_tokens", "unknown"),
                duration=float(request.get("duration_seconds") or 0.0), cost=float(usage.get("estimated_cost_usd") or 0.0),
            )
        )
    markdown_path = path.with_name("validation_forensic_report.md")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path, json_path


def _load_json_artifact(path: str | Path, *, expected_mode: str | None = None) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise StageTestError(f"Checkpoint does not exist: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if expected_mode and payload.get("mode") != expected_mode:
        raise StageTestError(f"Unexpected checkpoint mode for {artifact_path.name}")
    return payload


def _replace_question_payload(state: dict[str, Any], question_id: str, replacement: dict[str, Any]) -> None:
    for exam in state.get("generated_exams") or []:
        for section in (exam.get("questions") or {}).values():
            values = section.get("items") if isinstance(section, dict) else section
            for index, item in enumerate(values or []):
                if str(item.get("question_id") or "") == question_id:
                    values[index] = deepcopy(replacement)
                    return
    raise StageTestError(f"Question {question_id} is absent from the generated exam")


def _question_payload(state: dict[str, Any], question_id: str) -> dict[str, Any]:
    for exam in state.get("generated_exams") or []:
        for section in (exam.get("questions") or {}).values():
            values = section.get("items") if isinstance(section, dict) else section
            for item in values or []:
                if str(item.get("question_id") or "") == question_id:
                    return deepcopy(item)
    raise StageTestError(f"Question {question_id} is absent from the generated exam")


def _phase4_reconciled_state(
    source: dict[str, Any], validation: dict[str, Any], grounding: dict[str, Any],
    repair: dict[str, Any], revalidation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a new state from authoritative role evidence without touching sources."""
    from app.online.lifecycle import initialize_slot_lifecycle, reconcile_slot_lifecycle
    from app.online.multi_validator import _compatibility_verdict, protect_question
    from app.online.validator import _iter_questions, _local_verdict, _plan_context_for_model

    source_state = deepcopy(source["state"])
    validation_state = dict(validation.get("state") or {})
    original_report = next(iter(validation_state.get("validation_reports") or []), None)
    if not isinstance(original_report, dict):
        raise StageTestError("Validation checkpoint has no report")
    if grounding.get("source_generation_state_sha256") != source.get("state_sha256"):
        raise StageTestError("Grounding checkpoint is not bound to this Generation checkpoint")
    if revalidation.get("source_generation_state_sha256") != source.get("state_sha256"):
        raise StageTestError("Revalidation checkpoint is not bound to this Generation checkpoint")
    if revalidation.get("source_validation_state_sha256") != validation.get("state_sha256"):
        raise StageTestError("Revalidation checkpoint is not bound to this Validation checkpoint")
    if repair.get("source_generation_state_sha256") != source.get("state_sha256"):
        raise StageTestError("Repair checkpoint is not bound to this Generation checkpoint")

    repaired = deepcopy(revalidation.get("repaired_question") or {})
    if str(repaired.get("question_id") or "") != "model1_true_false_1":
        raise StageTestError("True/False revalidation checkpoint lacks its repaired question")
    _replace_question_payload(source_state, "model1_true_false_1", repaired)

    report = deepcopy(original_report)
    role_results = deepcopy(report.get("role_results") or {})
    role_fingerprints = deepcopy(report.get("role_fingerprints") or {})
    mcq_id, tf_id = "model1_mcq_1", "model1_true_false_1"
    grounding_verdict = deepcopy(grounding.get("grounding_verdict") or {})
    if grounding_verdict.get("status") != "PASS" or grounding_verdict.get("grounding_outcome") != "SUPPORTED":
        raise StageTestError("MCQ Grounding retest is not a supported PASS")
    role_results.setdefault(mcq_id, {})["grounding"] = grounding_verdict
    revalidation_report = revalidation.get("revalidation") or {}
    tf_revalidation = next((deepcopy(item) for item in revalidation_report.get("verdicts") or []
                            if str(item.get("question_id") or "") == tf_id), None)
    if not tf_revalidation or tf_revalidation.get("action") != "PASS":
        raise StageTestError("True/False did not pass its one permitted Revalidation")
    if set((tf_revalidation.get("role_statuses") or {}).values()) != {"PASS"}:
        raise StageTestError("True/False Revalidation did not pass every applicable role")
    role_results[tf_id] = deepcopy((revalidation_report.get("role_results") or {}).get(tf_id) or {})
    if not role_results[tf_id]:
        raise StageTestError("True/False Revalidation lacks per-role results")
    role_fingerprints[tf_id] = deepcopy((revalidation_report.get("role_fingerprints") or {}).get(tf_id) or {})

    plan_by_slot, source_by_chunk = _plan_context_for_model(source_state, 1)
    reconciled_verdicts: list[dict[str, Any]] = []
    for qtype, container, raw in _iter_questions(source_state["generated_exams"][0].get("questions") or {}):
        task = plan_by_slot.get(str(raw.get("slot_id") or "")) or {}
        protected = protect_question(
            qtype, raw, task, source_by_chunk,
            word_bank=(container.get("word_bank") or []) if isinstance(container, dict) else [],
        )
        qid = protected.question_id
        if qid == tf_id:
            verdict = deepcopy(tf_revalidation)
        else:
            verdict = _compatibility_verdict(
                protected, role_results.get(qid) or {},
                _local_verdict(qtype, protected.question,
                               {"word_bank": protected.word_bank} if qtype == "fill_in_the_blank" else None),
            )
        if verdict.get("action") != "PASS":
            raise StageTestError(f"Reconciled lifecycle is not accepted for {qid}: {verdict.get('failure_code')}")
        reconciled_verdicts.append(verdict)
    report.update({
        "verdicts": reconciled_verdicts,
        "role_results": role_results,
        "role_fingerprints": role_fingerprints,
        "all_pass": True,
        "reconciliation_provenance": {
            "mcq_grounding_retest": str(grounding.get("question_id") or ""),
            "true_false_revalidation": tf_id,
            "method": "production_role_aggregation",
        },
    })
    source_state.update({
        "validation_reports": [report],
        "validated_models": [1],
        "validation_repair_history": [{
            "model_number": 1, "question_id": tf_id, "slot_id": "m1_true_false_1",
            "status": "accepted", "repair_route": deepcopy((repair.get("repair") or {}).get("repair_route") or []),
            "issue_ids": [item.get("issue_id") for item in repair.get("original_findings") or []],
            "source": "controlled_one_pass_repair_and_revalidation",
        }],
        "repair_manifest": [{
            "model_number": 1, "question_id": tf_id, "slot_id": "m1_true_false_1", "status": "resolved",
            "failure_code": None, "history": [{"event": "revalidated_pass", "source": "controlled_one_pass"}],
        }],
        "repair_candidates": [], "pending_revalidation_ids": {}, "invalid_questions": [],
        "duplicate_reports": [], "duplicate_scan_count": 0, "pending_duplicate_groups": [],
        "duplicate_screen_fingerprints": {}, "unresolved_duplicate_conflicts": [],
        "duplicate_repair_history": [], "warnings": [],
        "slot_lifecycle": initialize_slot_lifecycle(source_state["tasks"], source_state["num_models"]),
    })
    source_state["slot_lifecycle"] = reconcile_slot_lifecycle(
        source_state["slot_lifecycle"], source_state, stage="phase4_reconciliation",
    )
    return source_state, report


def run_phase4_duplicate_and_assemble(
    generation_checkpoint_path: str | Path, *, validation_checkpoint_path: str | Path,
    grounding_checkpoint_path: str | Path, repair_checkpoint_path: str | Path,
    revalidation_checkpoint_path: str | Path, checkpoint_root: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Run Duplicate V2 once, then production Final Assembly if it is clear.

    The current controlled case has an independently preflighted zero-candidate
    screen. A nonzero candidate set is retained as a durable blocked result;
    it is never allowed to enter the ordinary multi-round duplicate-repair loop.
    """
    from app.online.duplicate_judge import shortlist_duplicate_pairs
    from app.online.duplicate_pipeline import _correctness_pass_exam, _plan_by_slot, scan_exam_duplicates
    from app.online.exam_builder import assemble_exams_node

    source = load_generation_checkpoint(generation_checkpoint_path)
    validation = _load_validation_checkpoint(validation_checkpoint_path)
    grounding = _load_json_artifact(grounding_checkpoint_path, expected_mode="grounding_only_targeted_retest")
    repair = _load_json_artifact(repair_checkpoint_path, expected_mode="one_pass_true_false_repair")
    revalidation = _load_json_artifact(revalidation_checkpoint_path, expected_mode="one_pass_true_false_revalidation")
    state, reconciled_report = _phase4_reconciled_state(source, validation, grounding, repair, revalidation)
    exam = deepcopy((state.get("generated_exams") or [])[0])
    candidates, screening = shortlist_duplicate_pairs(
        _correctness_pass_exam(state, exam), _plan_by_slot(state.get("plans") or [], 1),
    )
    if candidates:
        raise StageTestError(
            "Duplicate preflight found candidate pairs. This isolated runner refuses the production "
            "multi-round replacement loop; the candidates were not sent to a provider."
        )
    requests: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    with _production_domain(state), _isolate_llm_telemetry(), _capture_structured_requests(requests):
        duplicate_result = scan_exam_duplicates(state)
    state.update(duplicate_result)
    completed = datetime.now(timezone.utc)
    run_dir = _run_directory(_checkpoint_root(checkpoint_root), str(source["run_id"]))
    duplicate_path = run_dir / "phase4_duplicate_checkpoint.json"
    common = {
        "checkpoint_version": CHECKPOINT_VERSION, "mode": "phase4_duplicate_v2",
        "source_checkpoints": {
            "generation": str(Path(generation_checkpoint_path).resolve()),
            "validation": str(Path(validation_checkpoint_path).resolve()),
            "grounding": str(Path(grounding_checkpoint_path).resolve()),
            "repair": str(Path(repair_checkpoint_path).resolve()),
            "revalidation": str(Path(revalidation_checkpoint_path).resolve()),
        },
        "reconciled_validation_report": reconciled_report,
        "deterministic_preflight": screening,
        "duplicate_result": duplicate_result,
        "captured_requests": requests,
        "execution": {"started_at": started.isoformat(), "completed_at": completed.isoformat(),
                      "wall_clock_seconds": round(time.perf_counter() - started_clock, 6)},
    }
    if state.get("pending_duplicate_groups") or state.get("unresolved_duplicate_conflicts"):
        _atomic_json_write(duplicate_path, {**common, "status": "duplicate_conflict_blocked_before_repair"})
        return duplicate_path, None
    _atomic_json_write(duplicate_path, {**common, "status": "duplicate_clear_awaiting_final_assembly"})
    assembled = assemble_exams_node(state)
    state.update(assembled)
    accepted = set(state.get("accepted_slot_ids") or [])
    expected_slots = {"m1_mcq_1", "m1_true_false_1", "m1_equation_1", "m1_word_problem_1"}
    if accepted != expected_slots:
        raise StageTestError(f"Final Assembly did not accept exactly the four controlled slots: {sorted(accepted)}")
    final_path = run_dir / "phase4_final_assembly_checkpoint.json"
    _atomic_json_write(final_path, {
        **common, "mode": "phase4_final_assembly", "status": "final_assembly_complete",
        "final_assembly": assembled,
        "accepted_slot_ids": sorted(accepted),
    })
    return duplicate_path, final_path


def write_phase4_forensic_report(
    duplicate_checkpoint_path: str | Path, final_checkpoint_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write a filesystem-only report for the isolated Phase 4 outcome."""
    duplicate_path = Path(duplicate_checkpoint_path)
    duplicate = _load_json_artifact(duplicate_path, expected_mode="phase4_duplicate_v2")
    final_path = Path(final_checkpoint_path) if final_checkpoint_path else duplicate_path.with_name(
        "phase4_final_assembly_checkpoint.json"
    )
    final = _load_json_artifact(final_path, expected_mode="phase4_final_assembly") if final_path.is_file() else None
    requests = list(duplicate.get("captured_requests") or [])
    reports = list((duplicate.get("duplicate_result") or {}).get("duplicate_reports") or [])
    result = {
        "report_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Duplicate V2 and Final Assembly only; no Generation, ordinary Validation, Repair, or Revalidation ran in this phase.",
        "duplicate_checkpoint": str(duplicate_path.resolve()),
        "duplicate_checkpoint_sha256": hashlib.sha256(duplicate_path.read_bytes()).hexdigest(),
        "final_checkpoint": str(final_path.resolve()) if final else None,
        "final_checkpoint_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest() if final else None,
        "source_checkpoints": duplicate.get("source_checkpoints"),
        "duplicate_status": duplicate.get("status"),
        "deterministic_preflight": duplicate.get("deterministic_preflight"),
        "duplicate_reports": reports,
        "duplicate_requests": requests,
        "phase_request_totals": _request_totals(requests),
        "execution": duplicate.get("execution"),
        "reconciled_validation_report": duplicate.get("reconciled_validation_report"),
        "final_assembly": (final or {}).get("final_assembly"),
        "accepted_slot_ids": (final or {}).get("accepted_slot_ids") or [],
        "telemetry_note": "Costs are estimated from provider-reported usage, not an invoice. A zero request count means no Duplicate Judge or repair call was needed.",
    }
    json_path = duplicate_path.with_name("phase4_duplicate_final_assembly_forensic_report.json")
    _atomic_json_write(json_path, result)
    screen = result["deterministic_preflight"] or {}
    totals = result["phase_request_totals"]
    final_lifecycle = ((result.get("final_assembly") or {}).get("slot_lifecycle") or {})
    lines = [
        "# Controlled Phase 4: Duplicate V2 and Final Assembly",
        "",
        "- Scope: one isolated production Duplicate V2 scan followed by production Final Assembly. No Generation, ordinary Validation, Repair, or Revalidation ran in this phase.",
        f"- Duplicate status: `{result['duplicate_status']}`.",
        f"- Deterministic screening: `{screen.get('evaluated_pair_count', 0)}` of `{screen.get('possible_pair_count', 0)}` possible pairs evaluated; `{screen.get('candidate_count', 0)}` candidate pairs; BGE-M3 embedding calls: `{screen.get('embedding_calls', 'unknown')}`.",
        f"- Duplicate Judge calls: `{len(requests)}`; phase estimated cost: `${totals['estimated_cost_usd']:.8f}`.",
        f"- Phase wall-clock duration: `{float((result.get('execution') or {}).get('wall_clock_seconds') or 0):.6f}` seconds.",
        f"- Final Assembly completed: `{bool(final)}`; accepted slots: `{', '.join(result['accepted_slot_ids']) or 'none'}`.",
        "",
        "## Final lifecycle",
        "",
        "| Slot | Current status | Finalized | Terminal status | Failure code |",
        "|---|---|---:|---|---|",
    ]
    for slot_id, lifecycle in sorted(final_lifecycle.items()):
        lines.append(
            f"| {slot_id} | {lifecycle.get('current_status')} | {lifecycle.get('finalized')} | "
            f"{lifecycle.get('terminal_status')} | {lifecycle.get('failure_code') or ''} |"
        )
    lines.extend([
        "",
        "## Duplicate V2 evidence",
        "",
        "No candidate pair means no pair received semantic LLM review; it does not claim that every theoretical pair received a Judge call.",
        "",
        "```json",
        json.dumps(reports, ensure_ascii=False, indent=2),
        "```",
        "",
        "The JSON companion contains the complete reconciled validation evidence, final assembled payload, lifecycle records, all candidate-pair data, and captured request metadata.",
    ])
    markdown_path = duplicate_path.with_name("phase4_duplicate_final_assembly_forensic_report.md")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path, json_path


def _parse_tasks(value: str) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    for raw in value.split(","):
        qtype, separator, count = raw.strip().partition("=")
        if not separator:
            raise argparse.ArgumentTypeError("tasks must use type=count entries")
        try:
            tasks.append((qtype.strip(), int(count)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid count in {raw!r}") from exc
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generation-only")
    generate.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID)
    generate.add_argument("--tasks", type=_parse_tasks, default=list(DEFAULT_TASKS))
    generate.add_argument("--difficulty", default="mix")
    generate.add_argument("--checkpoint-root")
    validate = subparsers.add_parser("validation-only")
    validate.add_argument("--checkpoint", required=True)
    validate.add_argument("--checkpoint-root")
    grounding = subparsers.add_parser("grounding-only")
    grounding.add_argument("--checkpoint", required=True)
    grounding.add_argument("--validation-checkpoint")
    grounding.add_argument("--checkpoint-root")
    report = subparsers.add_parser("validation-report")
    report.add_argument("--checkpoint", required=True)
    grounding_report = subparsers.add_parser("grounding-report")
    grounding_report.add_argument("--checkpoint", required=True)
    repair_revalidate = subparsers.add_parser("repair-revalidate-once")
    repair_revalidate.add_argument("--checkpoint", required=True)
    repair_revalidate.add_argument("--validation-checkpoint")
    repair_revalidate.add_argument("--checkpoint-root")
    repair_report = subparsers.add_parser("repair-revalidation-report")
    repair_report.add_argument("--repair-checkpoint", required=True)
    repair_report.add_argument("--revalidation-checkpoint")
    phase4 = subparsers.add_parser("phase4-duplicate-assemble")
    phase4.add_argument("--checkpoint", required=True)
    phase4.add_argument("--validation-checkpoint", required=True)
    phase4.add_argument("--grounding-checkpoint", required=True)
    phase4.add_argument("--repair-checkpoint", required=True)
    phase4.add_argument("--revalidation-checkpoint", required=True)
    phase4.add_argument("--checkpoint-root")
    phase4_report = subparsers.add_parser("phase4-report")
    phase4_report.add_argument("--duplicate-checkpoint", required=True)
    phase4_report.add_argument("--final-checkpoint")
    args = parser.parse_args(argv)
    if args.command == "generation-only":
        path = run_generation_only(
            document_id=args.document_id, tasks=args.tasks, difficulty=args.difficulty,
            checkpoint_root=args.checkpoint_root,
        )
    elif args.command == "validation-only":
        path = run_validation_only(args.checkpoint, checkpoint_root=args.checkpoint_root)
    elif args.command == "grounding-only":
        path = run_grounding_only(
            args.checkpoint,
            validation_checkpoint_path=args.validation_checkpoint,
            checkpoint_root=args.checkpoint_root,
        )
    elif args.command == "grounding-report":
        markdown_path, json_path = write_grounding_retest_report(args.checkpoint)
        print(markdown_path)
        print(json_path)
        return 0
    elif args.command == "repair-revalidate-once":
        repair_path, revalidation_path = run_repair_revalidate_once(
            args.checkpoint,
            validation_checkpoint_path=args.validation_checkpoint,
            checkpoint_root=args.checkpoint_root,
        )
        print(repair_path)
        if revalidation_path:
            print(revalidation_path)
        return 0
    elif args.command == "repair-revalidation-report":
        markdown_path, json_path = write_repair_revalidation_report(
            args.repair_checkpoint, args.revalidation_checkpoint,
        )
        print(markdown_path)
        print(json_path)
        return 0
    elif args.command == "phase4-duplicate-assemble":
        duplicate_path, final_path = run_phase4_duplicate_and_assemble(
            args.checkpoint,
            validation_checkpoint_path=args.validation_checkpoint,
            grounding_checkpoint_path=args.grounding_checkpoint,
            repair_checkpoint_path=args.repair_checkpoint,
            revalidation_checkpoint_path=args.revalidation_checkpoint,
            checkpoint_root=args.checkpoint_root,
        )
        print(duplicate_path)
        if final_path:
            print(final_path)
        return 0
    elif args.command == "phase4-report":
        markdown_path, json_path = write_phase4_forensic_report(
            args.duplicate_checkpoint, args.final_checkpoint,
        )
        print(markdown_path)
        print(json_path)
        return 0
    else:
        markdown_path, json_path = write_validation_forensic_report(args.checkpoint)
        print(markdown_path)
        print(json_path)
        return 0
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
