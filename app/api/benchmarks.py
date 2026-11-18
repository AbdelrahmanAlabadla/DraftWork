"""Controlled benchmark execution and comparison for the evaluation dashboard.

Benchmarks reuse the production generation pipeline: every run is an ordinary
``exam_generation`` job created through the repository, tagged with an explicit
``benchmark_id`` so it is isolated from normal user telemetry, and executed by
the normal Celery worker.  This module only orchestrates configuration,
sequencing, persistence, and comparison.

The orchestration is polling-driven.  ``advance_benchmark`` is called by the
status endpoint: it takes a PostgreSQL advisory lock for the benchmark, observes
the current run, starts the next run when the previous one finished, and
finalizes the benchmark when all runs are done.  All state transitions are
idempotent so repeated polls cannot enqueue a run twice.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app import config, db
from app.api import evaluation_store, repositories
from app.api.repositories import sanitize_jsonb
from app.jobs import service as job_service

QUESTION_TYPES = (
    "mcq",
    "true_false",
    "fill_in_the_blank",
    "definition",
    "why",
    "equation",
    "word_problem",
    "essay",
)
DIFFICULTIES = frozenset({"easy", "medium", "hard", "mix"})

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class BenchmarkError(RuntimeError):
    """Raised when a benchmark request is invalid or cannot be executed."""


# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------
def _git_commit_from_files(root: Path) -> str | None:
    """Read HEAD without spawning git (works in restricted worker sandboxes)."""
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref: "):
        return head if len(head) >= 7 else None
    ref = head[5:].strip()
    try:
        value = (root / ".git" / ref).read_text(encoding="utf-8").strip()
        if len(value) >= 7:
            return value
    except OSError:
        pass
    try:
        packed = (root / ".git" / "packed-refs").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in packed.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref and len(parts[0]) >= 7:
            return parts[0]
    return None


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        commit = (completed.stdout or "").strip()
        if len(commit) >= 7:
            return commit
    except Exception:
        pass
    return _git_commit_from_files(root)


def _constant_from_source(relative_path: str, name: str) -> str | None:
    """Read a module-level string constant without importing the module.

    Some validation modules pull in the full LLM client stack; importing them in
    the API process just to record a version would be wasteful and fragile.
    """
    try:
        text = (Path(__file__).resolve().parents[2] / relative_path).read_text(
            encoding="utf-8"
        )
    except OSError:
        return None
    match = re.search(rf'^{re.escape(name)}\s*=\s*["\']([^"\']+)', text, re.MULTILINE)
    return match.group(1) if match else None


def reproducibility_metadata() -> dict[str, Any]:
    """Return the configuration identity needed to reproduce a benchmark."""
    return {
        "git_commit": _git_commit(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "app_env": config.APP_ENV,
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": str(getattr(config, "DEEPSEEK_MODEL", None) or ""),
        "versions": {
            "generation": config.GENERATION_PIPELINE_VERSION,
            "validation": config.VALIDATION_PIPELINE_VERSION,
            "duplicate": config.DUPLICATE_PIPELINE_VERSION,
            "assessment_catalog": config.ASSESSMENT_CATALOG_VERSION,
            "assessment_compiler": config.ASSESSMENT_COMPILER_VERSION,
            "parser": config.PARSER_PIPELINE_VERSION,
            "chunker": config.CHUNKER_PIPELINE_VERSION,
            "embedding": config.EMBEDDING_PIPELINE_VERSION,
            "validation_schema": _constant_from_source(
                "app/online/validation/constants.py", "VALIDATION_SCHEMA_VERSION"
            ),
            "general_validation_rule": _constant_from_source(
                "app/online/validation/recovery.py", "GENERAL_VALIDATION_RULE_VERSION"
            ),
            "stem_validation_rule": _constant_from_source(
                "app/online/validation/recovery.py", "STEM_VALIDATION_RULE_VERSION"
            ),
            "catalog_contract": _constant_from_source(
                "app/online/assessment_catalog.py", "CATALOG_CONTRACT_VERSION"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
def _int(value: Any, field: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{field} must be a whole number") from exc
    if not minimum <= parsed <= maximum:
        raise BenchmarkError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def validate_config(
    payload: dict[str, Any], *, session_id: str, user_id: str | None
) -> dict[str, Any]:
    """Validate and normalize one benchmark configuration."""
    from app.api.job_routes import _validate_child_selection

    document_id = str(payload.get("document_id") or "").strip()
    if not document_id:
        raise BenchmarkError("Choose a source document")
    document = repositories.get_document_for_session(document_id, session_id, user_id)
    if document is None:
        raise BenchmarkError("Document not found")
    if document.get("status") != "ready":
        raise BenchmarkError("Document is not ready for generation")

    raw_child_ids = payload.get("child_ids")
    if not isinstance(raw_child_ids, list) or not raw_child_ids:
        raise BenchmarkError("Choose at least one section")
    child_ids = list(dict.fromkeys(str(value) for value in raw_child_ids if value))
    if len(child_ids) > config.MAX_CHILD_IDS_PER_GENERATION:
        raise BenchmarkError(
            f"Choose at most {config.MAX_CHILD_IDS_PER_GENERATION} sections"
        )
    _validate_child_selection(document, child_ids)

    raw_tasks = payload.get("tasks")
    tasks: dict[str, int] = {}
    if isinstance(raw_tasks, dict):
        items = raw_tasks.items()
    elif isinstance(raw_tasks, list):
        items = []
        for item in raw_tasks:
            if isinstance(item, dict) and "question_type" in item:
                items.append((item.get("question_type"), item.get("count")))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                items.append((item[0], item[1]))
    else:
        raise BenchmarkError("Provide question counts")
    for qtype, count in items:
        name = str(qtype or "").strip()
        if name not in QUESTION_TYPES:
            raise BenchmarkError(f"Unsupported question type: {name}")
        value = _int(count, f"{name} count", 0, config.MAX_QUESTIONS_PER_GENERATION)
        if value:
            tasks[name] = value
    if not tasks:
        raise BenchmarkError("Request at least one question")
    if sum(tasks.values()) > config.MAX_QUESTIONS_PER_GENERATION:
        raise BenchmarkError(
            f"A generation job may request at most {config.MAX_QUESTIONS_PER_GENERATION} questions"
        )

    num_models = _int(
        payload.get("num_models", 1),
        "Number of exam models",
        1,
        config.MAX_MODELS_PER_GENERATION,
    )
    difficulty = str(payload.get("difficulty") or "mix").strip().lower()
    if difficulty not in DIFFICULTIES:
        raise BenchmarkError("Unsupported difficulty")
    repeats = _int(
        payload.get("repeats", 1),
        "Repeated runs",
        1,
        max(1, int(getattr(config, "BENCHMARK_MAX_REPEATS", 10))),
    )
    label = str(payload.get("label") or "").strip()[:120] or "Benchmark"
    stop_on_failure = bool(payload.get("stop_on_failure", True))

    return {
        "label": label,
        "document_id": document_id,
        "document_filename": str(document.get("original_filename") or ""),
        "child_ids": child_ids,
        "section_count": len(child_ids),
        "tasks": tasks,
        "num_models": num_models,
        "difficulty": difficulty,
        "repeats": repeats,
        "stop_on_failure": stop_on_failure,
        "questions_per_model": sum(tasks.values()),
        "total_questions": sum(tasks.values()) * num_models * repeats,
        "catalog_reuse": (
            "Reuses the cached assessment catalog when this document and section "
            "selection were already compiled; the first run may compile it."
        ),
    }


def config_fingerprint(cfg: dict[str, Any]) -> str:
    payload = {
        "document_id": cfg.get("document_id"),
        "child_ids": sorted(cfg.get("child_ids") or []),
        "tasks": sorted((cfg.get("tasks") or {}).items()),
        "num_models": cfg.get("num_models"),
        "difficulty": cfg.get("difficulty"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------
def estimate(payload: dict[str, Any], *, session_id: str, user_id: str | None) -> dict[str, Any]:
    """Estimate benchmark cost from actual historical provider telemetry.

    Averages are computed per requested question over completed, non-benchmark
    runs that have provider telemetry, then scaled by the configured question
    volume.  The result is an estimate, never a measurement.
    """
    cfg = validate_config(payload, session_id=session_id, user_id=user_id)
    questions = cfg["questions_per_model"] * cfg["num_models"]
    with db.connection() as conn:
        with conn.cursor() as cursor:
            job_expression = evaluation_store._run_job_expression()
            cursor.execute(
                f"""WITH linked AS (
                        SELECT {job_expression} AS job_id,
                               r.questions_requested,
                               NULLIF(to_jsonb(r)->>'duration_ms', '')::bigint AS duration_ms
                        FROM evaluation_runs r
                        LEFT JOIN exams e ON e.id = r.exam_id
                        WHERE {job_expression} IS NOT NULL
                          AND COALESCE({evaluation_store._benchmark_column()}, NULL) IS NULL
                    )
                    SELECT (SELECT COUNT(*) FROM linked)::bigint AS runs,
                           (SELECT COALESCE(SUM(questions_requested), 0) FROM linked)::bigint
                               AS requested,
                           (SELECT COALESCE(SUM(duration_ms), 0) FROM linked)::bigint
                               AS duration_ms
                    """
            )
            runs, requested, duration_ms = cursor.fetchone()
            cursor.execute(
                f"""WITH linked AS (
                        SELECT {job_expression} AS job_id
                        FROM evaluation_runs r
                        LEFT JOIN exams e ON e.id = r.exam_id
                        WHERE {job_expression} IS NOT NULL
                          AND COALESCE({evaluation_store._benchmark_column()}, NULL) IS NULL
                    )
                    SELECT COUNT(*)::bigint AS calls,
                           COALESCE(SUM(l.input_tokens), 0)::bigint AS input_tokens,
                           COALESCE(SUM(l.output_tokens), 0)::bigint AS output_tokens,
                           COALESCE(SUM(l.estimated_cost_usd), 0) AS cost_usd
                    FROM llm_request_events l
                    JOIN linked ON linked.job_id = l.job_id
                    """
            )
            calls, input_tokens, output_tokens, cost = cursor.fetchone()
    runs = int(runs or 0)
    requested = int(requested or 0)
    calls = int(calls or 0)
    cost_value = evaluation_store._number(cost)
    if runs == 0 or requested == 0 or calls == 0 or cost_value is None:
        return {
            "available": False,
            "estimated": True,
            "config": cfg,
            "reason": (
                "Not enough historical provider telemetry to estimate cost. "
                "The benchmark can still run, but no cost estimate is shown."
            ),
            "estimate": None,
            "basis": {"runs": runs, "requested_questions": requested, "calls": calls},
        }
    per_question_cost = cost_value / requested
    per_question_calls = calls / requested
    per_question_duration = (int(duration_ms or 0) / requested) if duration_ms else None
    return {
        "available": True,
        "estimated": True,
        "config": cfg,
        "basis": {
            "runs": runs,
            "requested_questions": requested,
            "calls": calls,
            "avg_cost_per_question_usd": round(per_question_cost, 8),
            "avg_calls_per_question": round(per_question_calls, 4),
            "avg_duration_per_question_ms": (
                round(per_question_duration, 1) if per_question_duration else None
            ),
        },
        "estimate": {
            "runs": cfg["repeats"],
            "questions_per_run": questions,
            "total_questions": questions * cfg["repeats"],
            "estimated_cost_usd": round(per_question_cost * questions * cfg["repeats"], 6),
            "estimated_calls": round(per_question_calls * questions * cfg["repeats"], 1),
            "estimated_input_tokens": round(
                (int(input_tokens) / requested) * questions * cfg["repeats"], 0
            ),
            "estimated_output_tokens": round(
                (int(output_tokens) / requested) * questions * cfg["repeats"], 0
            ),
            "estimated_duration_ms": (
                round(per_question_duration * questions * cfg["repeats"], 0)
                if per_question_duration else None
            ),
        },
        "note": (
            "Estimated from completed runs with provider telemetry. Actual cost "
            "depends on provider pricing, retries, and repair volume."
        ),
    }


# ---------------------------------------------------------------------------
# Persistence and execution
# ---------------------------------------------------------------------------
def _benchmark_row(cursor: Any, benchmark_id: str) -> dict[str, Any] | None:
    cursor.execute("SELECT * FROM evaluation_benchmarks WHERE id = %s", (benchmark_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def _owned_benchmark(
    cursor: Any, benchmark_id: str, session_id: str, user_id: str
) -> dict[str, Any] | None:
    cursor.execute(
        """SELECT * FROM evaluation_benchmarks
           WHERE id = %s AND (owner_user_id = %s OR
                 (owner_user_id IS NULL AND owner_session_id = %s))""",
        (benchmark_id, user_id, session_id),
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def create_benchmark(
    payload: dict[str, Any],
    *,
    session_id: str,
    user_id: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create and start a benchmark.  Requires an explicit caller action."""
    cfg = validate_config(payload, session_id=session_id, user_id=user_id)
    estimate_payload = estimate(payload, session_id=session_id, user_id=user_id)
    benchmark_id = str(uuid.uuid4())
    key = str(idempotency_key or "").strip()[:100] or None
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"bench:{session_id}",))
            if key:
                cursor.execute(
                    """SELECT * FROM evaluation_benchmarks
                       WHERE owner_session_id = %s AND idempotency_key = %s""",
                    (session_id, key),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    return _benchmark_payload(cursor, dict(existing))
            cursor.execute(
                """INSERT INTO evaluation_benchmarks (
                       id, status, label, config, reproducibility, summary,
                       owner_session_id, owner_user_id, idempotency_key
                   ) VALUES (%s, 'queued', %s, %s, %s, %s, %s, %s, %s)""",
                (
                    benchmark_id,
                    cfg["label"],
                    Jsonb(sanitize_jsonb(cfg)),
                    Jsonb(sanitize_jsonb(reproducibility_metadata())),
                    Jsonb(sanitize_jsonb({"estimate": estimate_payload.get("estimate")})),
                    session_id,
                    user_id,
                    key,
                ),
            )
        conn.commit()
    try:
        _enqueue_next_run(benchmark_id)
    except Exception:
        _mark_benchmark_failed(benchmark_id, "Could not start the benchmark run")
        raise
    return get_benchmark(benchmark_id, session_id=session_id, user_id=user_id) or {}


def _enqueue_next_run(benchmark_id: str) -> dict[str, Any] | None:
    """Start the next run of a benchmark exactly once (advisory-locked)."""
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"benchmark:{benchmark_id}",),
            )
            benchmark = _benchmark_row(cursor, benchmark_id)
            if benchmark is None:
                return None
            status = str(benchmark.get("status") or "")
            if status in TERMINAL_STATUSES:
                return None
            cfg = dict(benchmark.get("config") or {})
            repeats = int(cfg.get("repeats") or 1)
            cursor.execute(
                """SELECT * FROM evaluation_benchmark_runs
                   WHERE benchmark_id = %s ORDER BY run_index DESC LIMIT 1""",
                (benchmark_id,),
            )
            latest = cursor.fetchone()
            if latest is not None:
                latest_status = str(latest.get("status") or "")
                if latest_status not in TERMINAL_STATUSES:
                    return dict(latest)
                run_index = int(latest["run_index"]) + 1
            else:
                run_index = 1
            if run_index > repeats:
                return None
            run_id = str(uuid.uuid4())
            cursor.execute(
                """INSERT INTO evaluation_benchmark_runs (
                       id, benchmark_id, run_index, status
                   ) VALUES (%s, %s, %s, 'queued')""",
                (run_id, benchmark_id, run_index),
            )
            request_data = {
                "tasks": [[qtype, count] for qtype, count in cfg["tasks"].items()],
                "num_models": int(cfg.get("num_models") or 1),
                "difficulty": str(cfg.get("difficulty") or "mix"),
                "child_ids": list(cfg.get("child_ids") or []),
                "metadata": {
                    "benchmark_id": benchmark_id,
                    "benchmark_label": str(cfg.get("label") or "Benchmark"),
                },
                "benchmark_id": benchmark_id,
                "benchmark_run_id": run_id,
            }
            conn.commit()
        job = repositories.create_generation_job(
            session_id=str(benchmark["owner_session_id"]),
            document_id=str(cfg["document_id"]),
            request_data=request_data,
            user_id=str(benchmark.get("owner_user_id") or "") or None,
            benchmark=True,
        )
        job_id = str(job["id"])
        exam_id = str(job.get("exam_id") or "")
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE evaluation_benchmark_runs
                   SET job_id = %s, exam_id = %s, status = 'running'
                   WHERE id = %s""",
                (job_id, exam_id, run_id),
            )
            cursor.execute(
                """UPDATE evaluation_benchmarks
                   SET status = 'running',
                       started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (benchmark_id,),
            )
        conn.commit()
    try:
        job_service.enqueue_generation(job_id)
        repositories.mark_job_dispatched(job_id)
    except Exception:
        repositories.fail_job(
            job_id,
            "QUEUE_UNAVAILABLE",
            "Background job queue is unavailable",
        )
        raise
    return {"id": run_id, "run_index": run_index, "job_id": job_id, "exam_id": exam_id}


def advance_benchmark(benchmark_id: str) -> dict[str, Any] | None:
    """Observe the current run and move the benchmark forward (idempotent).

    Concurrency: the read/enqueue decision happens under a benchmark advisory
    lock, and ``_enqueue_next_run`` re-checks the latest run under the same lock,
    so two concurrent polls cannot create a duplicate paid run.
    """
    action: str | None = None
    latest: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"benchmark:{benchmark_id}",),
            )
            benchmark = _benchmark_row(cursor, benchmark_id)
            if benchmark is None:
                return None
            if str(benchmark.get("status") or "") in TERMINAL_STATUSES:
                return benchmark
            cursor.execute(
                """SELECT * FROM evaluation_benchmark_runs
                   WHERE benchmark_id = %s ORDER BY run_index DESC LIMIT 1""",
                (benchmark_id,),
            )
            latest = cursor.fetchone()
        conn.commit()

    if latest is None or str(latest.get("status") or "") in TERMINAL_STATUSES:
        action = "enqueue"
    elif not latest.get("job_id"):
        created = latest.get("created_at")
        if created is not None:
            age = (datetime.now(timezone.utc) - created).total_seconds()
            if age > 90:
                action = "stuck"
    else:
        action = "observe"

    if action == "enqueue":
        try:
            _enqueue_next_run(benchmark_id)
        except Exception as exc:
            _mark_benchmark_failed(benchmark_id, f"Could not start a benchmark run: {exc}")
            return _benchmark_row_with_cursor(benchmark_id)
        _finalize_if_done(benchmark_id)
        return _benchmark_row_with_cursor(benchmark_id)

    if action == "stuck":
        _finish_run(str(latest["id"]), "failed", error="Benchmark run was never dispatched")
        _mark_benchmark_failed(
            benchmark_id,
            f"Run {latest.get('run_index')} was never dispatched by the worker queue",
        )
        return _benchmark_row_with_cursor(benchmark_id)

    if latest is None:
        return _benchmark_row_with_cursor(benchmark_id)

    job_id = str(latest.get("job_id") or "")
    job = repositories.get_job(job_id)
    if job is None:
        return _benchmark_row_with_cursor(benchmark_id)
    job_status = str(job.get("status") or "")
    if job_status == "completed":
        with db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM evaluation_runs WHERE exam_id = %s",
                    (str(latest.get("exam_id") or ""),),
                )
                evaluated = cursor.fetchone() is not None
        if not evaluated:
            # The worker completed the job but telemetry has not been written yet.
            return _benchmark_row_with_cursor(benchmark_id)
        _finish_run(str(latest["id"]), "completed")
        try:
            _enqueue_next_run(benchmark_id)
        except Exception as exc:
            _mark_benchmark_failed(benchmark_id, f"Could not start a benchmark run: {exc}")
            return _benchmark_row_with_cursor(benchmark_id)
        _finalize_if_done(benchmark_id)
        return _benchmark_row_with_cursor(benchmark_id)

    if job_status in {"failed", "cancelled"}:
        error = job.get("error_message") or job.get("error_code")
        _finish_run(str(latest["id"]), job_status, error=str(error or "")[:1000] or None)
        cfg = dict(benchmark.get("config") or {})
        if job_status == "cancelled" or bool(cfg.get("stop_on_failure", True)):
            _mark_benchmark_failed(
                benchmark_id,
                f"Run {latest.get('run_index')} {job_status}"
                + (f": {error}" if error else ""),
                status="cancelled" if job_status == "cancelled" else "failed",
            )
        else:
            try:
                _enqueue_next_run(benchmark_id)
            except Exception as exc:
                _mark_benchmark_failed(
                    benchmark_id, f"Could not start a benchmark run: {exc}"
                )
                return _benchmark_row_with_cursor(benchmark_id)
            _finalize_if_done(benchmark_id)
        return _benchmark_row_with_cursor(benchmark_id)
    return _benchmark_row_with_cursor(benchmark_id)


def _benchmark_row_with_cursor(benchmark_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            return _benchmark_row(cursor, benchmark_id)


def _finish_run(run_id: str, status: str, *, error: str | None = None) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE evaluation_benchmark_runs
                   SET status = %s, finished_at = COALESCE(finished_at, CURRENT_TIMESTAMP),
                       error_message = COALESCE(%s, error_message)
                   WHERE id = %s AND status NOT IN ('completed', 'failed', 'cancelled')""",
                (status, error, run_id),
            )
        conn.commit()


def _mark_benchmark_failed(
    benchmark_id: str, message: str, *, status: str = "failed"
) -> None:
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE evaluation_benchmarks
                   SET status = %s, error_message = %s,
                       finished_at = COALESCE(finished_at, CURRENT_TIMESTAMP),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s AND status NOT IN ('completed', 'failed', 'cancelled')""",
                (status, str(message)[:1000], benchmark_id),
            )
        conn.commit()


def _finalize_if_done(benchmark_id: str) -> None:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            benchmark = _benchmark_row(cursor, benchmark_id)
            if benchmark is None:
                return
            cfg = dict(benchmark.get("config") or {})
            repeats = int(cfg.get("repeats") or 1)
            cursor.execute(
                """SELECT status, COUNT(*)::bigint AS count
                   FROM evaluation_benchmark_runs WHERE benchmark_id = %s
                   GROUP BY status""",
                (benchmark_id,),
            )
            counts = {str(row["status"]): int(row["count"]) for row in cursor.fetchall()}
            terminal = sum(
                counts.get(status, 0) for status in TERMINAL_STATUSES
            )
            if terminal < repeats:
                return
            summary = _benchmark_summary(benchmark_id)
            status = "completed" if counts.get("completed", 0) else "failed"
            cursor.execute(
                """UPDATE evaluation_benchmarks
                   SET status = %s, summary = %s,
                       finished_at = COALESCE(finished_at, CURRENT_TIMESTAMP),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (status, Jsonb(sanitize_jsonb(summary)), benchmark_id),
            )
        conn.commit()


def cancel_benchmark(
    benchmark_id: str, *, session_id: str, user_id: str
) -> dict[str, Any] | None:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            benchmark = _owned_benchmark(cursor, benchmark_id, session_id, user_id)
            if benchmark is None:
                return None
            if str(benchmark.get("status") or "") in TERMINAL_STATUSES:
                return benchmark
            cursor.execute(
                """SELECT * FROM evaluation_benchmark_runs
                   WHERE benchmark_id = %s AND status NOT IN ('completed','failed','cancelled')
                   ORDER BY run_index DESC LIMIT 1""",
                (benchmark_id,),
            )
            active = cursor.fetchone()
        conn.commit()
    if active is not None and active.get("job_id"):
        job_id = str(active["job_id"])
        try:
            job = repositories.cancel_job(job_id)
            if job and job.get("status") == "cancelled":
                job_service.terminate_job(
                    job_id,
                    document_id=str(job["document_id"]) if job.get("document_id") else None,
                    session_id=str(job["session_id"]),
                    cleanup_document=False,
                )
        except Exception:
            pass
        _finish_run(str(active["id"]), "cancelled", error="Cancelled by operator")
    _mark_benchmark_failed(benchmark_id, "Cancelled by operator", status="cancelled")
    return _benchmark_row_with_cursor(benchmark_id)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _llm_for_benchmark(cursor: Any, benchmark_id: str) -> dict[str, Any]:
    cursor.execute(
        """SELECT COUNT(*)::bigint AS calls,
                  COUNT(*) FILTER (WHERE l.status = 'success')::bigint AS successful,
                  COUNT(*) FILTER (WHERE l.status <> 'success')::bigint AS failed,
                  COALESCE(SUM(l.retry_count), 0)::bigint AS retries,
                  COUNT(l.input_tokens)::bigint AS usage_calls,
                  COUNT(l.estimated_cost_usd)::bigint AS cost_calls,
                  COUNT(l.duration_ms)::bigint AS latency_calls,
                  COALESCE(SUM(l.input_tokens), 0)::bigint AS input_tokens,
                  COALESCE(SUM(l.output_tokens), 0)::bigint AS output_tokens,
                  COALESCE(SUM(l.reasoning_tokens), 0)::bigint AS reasoning_tokens,
                  COALESCE(SUM(NULLIF(l.metadata->>'cached_tokens', '')::bigint), 0)::bigint
                      AS cached_tokens,
                  COALESCE(SUM(l.estimated_cost_usd), 0) AS cost_usd,
                  AVG(l.duration_ms) AS avg_latency_ms,
                  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY l.duration_ms)
                      AS p95_latency_ms
           FROM evaluation_benchmark_runs br
           JOIN llm_request_events l ON l.job_id = br.job_id
           WHERE br.benchmark_id = %s""",
        (benchmark_id,),
    )
    row = cursor.fetchone() or {}
    calls = int(_bench_row(row, "calls", 0) or 0)
    available = calls > 0
    return {
        "calls": calls,
        "telemetry_available": available,
        "successful": int(_bench_row(row, "successful", 1) or 0),
        "failed": int(_bench_row(row, "failed", 2) or 0),
        "retries": int(_bench_row(row, "retries", 3) or 0),
        "usage_calls": int(_bench_row(row, "usage_calls", 4) or 0),
        "cost_calls": int(_bench_row(row, "cost_calls", 5) or 0),
        "latency_calls": int(_bench_row(row, "latency_calls", 6) or 0),
        "input_tokens": int(_bench_row(row, "input_tokens", 7) or 0) if available else None,
        "output_tokens": int(_bench_row(row, "output_tokens", 8) or 0) if available else None,
        "reasoning_tokens": int(_bench_row(row, "reasoning_tokens", 9) or 0) if available else None,
        "cached_tokens": int(_bench_row(row, "cached_tokens", 10) or 0) if available else None,
        "estimated_cost_usd": (
            evaluation_store._number(_bench_row(row, "cost_usd", 11)) if available else None
        ),
        "avg_latency_ms": (
            round(evaluation_store._number(_bench_row(row, "avg_latency_ms", 12)) or 0, 1)
            if available else None
        ),
        "p95_latency_ms": (
            round(evaluation_store._number(_bench_row(row, "p95_latency_ms", 13)) or 0, 1)
            if available else None
        ),
    }


def _bench_row(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return row[index]


def _reason_totals(cursor: Any, column: str, benchmark_id: str) -> dict[str, int]:
    cursor.execute(
        f"""SELECT reason.key, SUM((reason.value)::bigint)::bigint AS count
            FROM evaluation_runs r
            CROSS JOIN LATERAL jsonb_each_text(
                COALESCE({column}, '{{}}'::jsonb)
            ) AS reason(key, value)
            WHERE r.benchmark_id = %s
            GROUP BY reason.key ORDER BY count DESC""",
        (benchmark_id,),
    )
    return {
        str(_bench_row(row, "key", 0)): int(_bench_row(row, "count", 1))
        for row in cursor.fetchall()
    }


def _jsonb_totals(cursor: Any, path: str, benchmark_id: str) -> dict[str, int]:
    cursor.execute(
        f"""SELECT entry.key, SUM((entry.value)::bigint)::bigint AS count
            FROM evaluation_runs r
            CROSS JOIN LATERAL jsonb_each_text(
                COALESCE((to_jsonb(r){path}), '{{}}'::jsonb)
            ) AS entry(key, value)
            WHERE r.benchmark_id = %s
            GROUP BY entry.key ORDER BY count DESC""",
        (benchmark_id,),
    )
    return {
        str(_bench_row(row, "key", 0)): int(_bench_row(row, "count", 1))
        for row in cursor.fetchall()
    }


def _benchmark_summary(benchmark_id: str) -> dict[str, Any]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT COUNT(*)::bigint AS runs,
                          COUNT(*) FILTER (WHERE br.status = 'completed')::bigint AS completed,
                          COUNT(*) FILTER (WHERE br.status = 'failed')::bigint AS failed,
                          COUNT(*) FILTER (WHERE br.status = 'cancelled')::bigint AS cancelled
                   FROM evaluation_benchmark_runs br WHERE br.benchmark_id = %s""",
                (benchmark_id,),
            )
            run_counts = cursor.fetchone()
            cursor.execute(
                """SELECT COALESCE(SUM(r.questions_requested), 0)::bigint AS requested,
                          COALESCE(SUM(r.generated_first), 0)::bigint AS generated,
                          COALESCE(SUM(r.validation_total_first), 0)::bigint AS validation_total,
                          COALESCE(SUM(r.validation_passed_first), 0)::bigint AS validation_passed,
                          COALESCE(SUM(r.generation_rejected), 0)::bigint AS generation_rejected,
                          COALESCE(SUM(r.repair_sent), 0)::bigint AS repair_sent,
                          COALESCE(SUM(r.repair_succeeded), 0)::bigint AS repair_succeeded,
                          COALESCE(SUM(r.final_valid), 0)::bigint AS final_valid,
                          COALESCE(SUM(r.final_invalid), 0)::bigint AS final_invalid,
                          COALESCE(SUM(r.final_unvalidated), 0)::bigint AS final_unvalidated,
                          COALESCE(SUM(r.final_missing), 0)::bigint AS final_missing,
                          COALESCE(SUM(NULLIF(to_jsonb(r)->>'duration_ms', '')::bigint), 0)::bigint
                              AS duration_ms,
                          COUNT(*) FILTER (
                              WHERE r.recovery_outcomes <> '{}'::jsonb
                          )::bigint AS recovery_telemetry_runs,
                          COUNT(*) FILTER (
                              WHERE r.duplicate_outcomes <> '{}'::jsonb
                          )::bigint AS duplicate_telemetry_runs,
                          COUNT(*) FILTER (
                              WHERE r.catalog_cache <> '{}'::jsonb
                          )::bigint AS cache_telemetry_runs
                   FROM evaluation_runs r WHERE r.benchmark_id = %s""",
                (benchmark_id,),
            )
            totals = cursor.fetchone() or {}
            cursor.execute(
                """SELECT COALESCE(SUM(
                              COALESCE((to_jsonb(r)->'recovery_outcomes'->>'entered_recovery')::bigint, 0)
                          ), 0)::bigint AS entered,
                          COALESCE(SUM(
                              COALESCE((to_jsonb(r)->'recovery_outcomes'->>'recovered')::bigint, 0)
                          ), 0)::bigint AS recovered,
                          COALESCE(SUM(
                              COALESCE((to_jsonb(r)->'recovery_outcomes'->>'attempts')::bigint, 0)
                          ), 0)::bigint AS attempts,
                          COALESCE(SUM(
                              COALESCE((to_jsonb(r)->'recovery_outcomes'->>'accepted')::bigint, 0)
                          ), 0)::bigint AS accepted
                   FROM evaluation_runs r WHERE r.benchmark_id = %s""",
                (benchmark_id,),
            )
            recovery = cursor.fetchone() or {}
            validation_reasons = _reason_totals(
                cursor, "r.validation_failure_reasons", benchmark_id
            )
            generation_reasons = _reason_totals(
                cursor, "r.generation_rejection_reasons", benchmark_id
            )
            terminal_status = _jsonb_totals(
                cursor, "->'recovery_outcomes'->'terminal_status'", benchmark_id
            )
            cursor.execute(
                """SELECT br.run_index, r.exam_id,
                          e.metadata->'generation_audit'->'slot_lifecycle' AS lifecycle
                   FROM evaluation_benchmark_runs br
                   JOIN evaluation_runs r ON r.exam_id = br.exam_id
                   LEFT JOIN exams e ON e.id = r.exam_id
                   WHERE br.benchmark_id = %s
                   ORDER BY br.run_index""",
                (benchmark_id,),
            )
            failed_slots: list[dict[str, Any]] = []
            for row in cursor.fetchall():
                lifecycle = row.get("lifecycle") or {}
                if not isinstance(lifecycle, dict):
                    continue
                for slot_id, record in lifecycle.items():
                    if not isinstance(record, dict):
                        continue
                    status = str(record.get("terminal_status") or "")
                    if status in {"", "finalized"}:
                        continue
                    failed_slots.append({
                        "run_index": int(row.get("run_index") or 0),
                        "slot_id": str(slot_id),
                        "question_type": record.get("question_type"),
                        "terminal_status": status,
                        "failure_stage": record.get("failure_stage"),
                        "failure_code": record.get("failure_code")
                        or "unspecified_failure",
                        "failure_reason": record.get("failure_reason"),
                        "repair_manifest_status": record.get("repair_manifest_status"),
                        "repair_attempts": len(record.get("repair_attempts") or []),
                    })
            llm = _llm_for_benchmark(cursor, benchmark_id)
    requested = int(totals.get("requested") or 0)
    final_valid = int(totals.get("final_valid") or 0)
    entered_count = int(recovery.get("entered") or 0)
    recovered_count = int(recovery.get("recovered") or 0)
    attempts = int(recovery.get("attempts") or 0)
    accepted = int(recovery.get("accepted") or 0)
    repair_sent = int(totals.get("repair_sent") or 0)
    generated = int(totals.get("generated") or 0)
    generation_rejected = int(totals.get("generation_rejected") or 0)
    final_invalid = int(totals.get("final_invalid") or 0)
    final_unvalidated = int(totals.get("final_unvalidated") or 0)
    final_missing = int(totals.get("final_missing") or 0)
    final_shortfall = max(requested - final_valid, 0)
    return {
        "runs": {
            "total": int(run_counts.get("runs") or 0),
            "completed": int(run_counts.get("completed") or 0),
            "failed": int(run_counts.get("failed") or 0),
            "cancelled": int(run_counts.get("cancelled") or 0),
        },
        "quality": {
            "requested": requested,
            "generated": generated,
            "validation_total": int(totals.get("validation_total") or 0),
            "validation_passed": int(totals.get("validation_passed") or 0),
            "first_pass_rate": evaluation_store._safe_divide(
                totals.get("validation_passed"), totals.get("validation_total")
            ),
            "validation_coverage": evaluation_store._safe_divide(
                totals.get("validation_total"), requested
            ),
            "generation_rejected": generation_rejected,
            "generation_rejection_rate": evaluation_store._safe_divide(
                generation_rejected, generation_rejected + generated
            ),
            "repair_sent": repair_sent,
            "repair_succeeded": int(totals.get("repair_succeeded") or 0),
            "repair_success_rate": evaluation_store._safe_divide(
                totals.get("repair_succeeded"), repair_sent
            ),
            "recovery_entered": entered_count,
            "recovery_recovered": recovered_count,
            "recovery_unresolved": max(entered_count - recovered_count, 0),
            "recovery_success_rate": evaluation_store._safe_divide(
                recovered_count, entered_count
            ),
            "recovery_attempts": attempts,
            "recovery_attempt_success_rate": evaluation_store._safe_divide(
                accepted, attempts
            ),
            "final_valid": final_valid,
            "final_invalid": final_invalid,
            "final_unvalidated": final_unvalidated,
            "final_missing": final_missing,
            "final_shortfall": final_shortfall,
            "final_acceptance_rate": evaluation_store._safe_divide(final_valid, requested),
            "duration_ms": int(totals.get("duration_ms") or 0) or None,
        },
        "telemetry": {
            "recovery_available": int(totals.get("recovery_telemetry_runs") or 0) > 0,
            "duplicates_available": int(totals.get("duplicate_telemetry_runs") or 0) > 0,
            "cache_available": int(totals.get("cache_telemetry_runs") or 0) > 0,
            "generation_rejections_recorded": int(
                totals.get("recovery_telemetry_runs") or 0
            ) > 0,
        },
        "failed_slots": failed_slots,
        "llm": llm,
        "cost_per_finalized_question_usd": (
            round(llm["estimated_cost_usd"] / final_valid, 6)
            if llm.get("estimated_cost_usd") is not None and final_valid else None
        ),
        "validation_rejection_reasons": validation_reasons,
        "generation_rejection_reasons": generation_reasons,
        "terminal_status": terminal_status,
    }


# ---------------------------------------------------------------------------
# Read APIs
# ---------------------------------------------------------------------------
def _run_rows(cursor: Any, benchmark_id: str) -> list[dict[str, Any]]:
    cursor.execute(
        """SELECT br.id, br.run_index, br.status, br.job_id, br.exam_id,
                  br.created_at, br.finished_at, br.error_message,
                  j.status AS job_status, j.stage AS job_stage,
                  r.questions_requested, r.generated_first, r.validation_total_first,
                  r.validation_passed_first, r.repair_sent, r.repair_succeeded,
                  r.final_valid, r.final_invalid, r.final_unvalidated, r.final_missing,
                  NULLIF(to_jsonb(r)->>'duration_ms', '')::bigint AS duration_ms,
                  l.calls, l.failed_calls, l.input_tokens, l.output_tokens,
                  l.cost_usd, l.avg_latency_ms
           FROM evaluation_benchmark_runs br
           LEFT JOIN jobs j ON j.id = br.job_id
           LEFT JOIN evaluation_runs r ON r.exam_id = br.exam_id
           LEFT JOIN LATERAL (
               SELECT COUNT(*)::bigint AS calls,
                      COUNT(*) FILTER (WHERE e.status <> 'success')::bigint AS failed_calls,
                      COALESCE(SUM(e.input_tokens), 0)::bigint AS input_tokens,
                      COALESCE(SUM(e.output_tokens), 0)::bigint AS output_tokens,
                      SUM(e.estimated_cost_usd) AS cost_usd,
                      AVG(e.duration_ms) AS avg_latency_ms
               FROM llm_request_events e WHERE e.job_id = br.job_id
           ) l ON TRUE
           WHERE br.benchmark_id = %s
           ORDER BY br.run_index""",
        (benchmark_id,),
    )
    rows = []
    for row in cursor.fetchall():
        created = row.get("created_at")
        finished = row.get("finished_at")
        rows.append({
            "run_index": int(row["run_index"]),
            "status": str(row.get("status") or ""),
            "job_id": str(row["job_id"]) if row.get("job_id") else None,
            "exam_id": str(row["exam_id"]) if row.get("exam_id") else None,
            "job_status": str(row.get("job_status") or "") or None,
            "job_stage": str(row.get("job_stage") or "") or None,
            "created_at": created.isoformat() if created else None,
            "finished_at": finished.isoformat() if finished else None,
            "error_message": row.get("error_message"),
            "requested": int(row.get("questions_requested") or 0),
            "generated": int(row.get("generated_first") or 0),
            "final_valid": int(row.get("final_valid") or 0),
            "first_pass_rate": evaluation_store._safe_divide(
                row.get("validation_passed_first"), row.get("validation_total_first")
            ),
            "final_acceptance_rate": evaluation_store._safe_divide(
                row.get("final_valid"), row.get("questions_requested")
            ),
            "repair_sent": int(row.get("repair_sent") or 0),
            "repair_succeeded": int(row.get("repair_succeeded") or 0),
            "duration_ms": (
                int(row["duration_ms"]) if row.get("duration_ms") is not None else None
            ),
            "llm": {
                "calls": int(row.get("calls") or 0),
                "failed_calls": int(row.get("failed_calls") or 0),
                "input_tokens": int(row.get("input_tokens") or 0),
                "output_tokens": int(row.get("output_tokens") or 0),
                "estimated_cost_usd": (
                    evaluation_store._number(row.get("cost_usd"))
                    if row.get("calls") else None
                ),
                "avg_latency_ms": (
                    round(evaluation_store._number(row.get("avg_latency_ms")) or 0, 1)
                    if row.get("avg_latency_ms") is not None else None
                ),
            },
        })
    return rows


def _benchmark_payload(
    cursor: Any, row: dict[str, Any], *, include_runs: bool = True
) -> dict[str, Any]:
    benchmark_id = str(row["id"])
    summary = _benchmark_summary(benchmark_id) if include_runs else {}
    runs = _run_rows(cursor, benchmark_id) if include_runs else []
    config_payload = dict(row.get("config") or {})
    created = row.get("created_at")
    return {
        "benchmark_id": benchmark_id,
        "label": str(row.get("label") or ""),
        "status": str(row.get("status") or ""),
        "created_at": created.isoformat() if created else None,
        "started_at": row["started_at"].isoformat() if row.get("started_at") else None,
        "finished_at": row["finished_at"].isoformat() if row.get("finished_at") else None,
        "error_message": row.get("error_message"),
        "config": config_payload,
        "config_fingerprint": config_fingerprint(config_payload),
        "reproducibility": dict(row.get("reproducibility") or {}),
        "estimate": (row.get("summary") or {}).get("estimate"),
        "summary": summary,
        "runs": runs,
        "progress": {
            "total": int(config_payload.get("repeats") or 0),
            "finished": int(summary.get("runs", {}).get("completed", 0))
            + int(summary.get("runs", {}).get("failed", 0))
            + int(summary.get("runs", {}).get("cancelled", 0)),
            "current_run": runs[-1]["run_index"] if runs else 0,
            "current_status": runs[-1]["status"] if runs else "queued",
            "current_stage": runs[-1].get("job_stage") if runs else None,
        },
    }


def get_benchmark(
    benchmark_id: str, *, session_id: str, user_id: str, advance: bool = True
) -> dict[str, Any] | None:
    if advance:
        advance_benchmark(benchmark_id)
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            row = _owned_benchmark(cursor, benchmark_id, session_id, user_id)
            if row is None:
                return None
            return _benchmark_payload(cursor, row)


def list_benchmarks(
    *, session_id: str, user_id: str, limit: int = 50
) -> dict[str, Any]:
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT id, label, status, config, summary, created_at,
                          started_at, finished_at, error_message
                   FROM evaluation_benchmarks
                   WHERE owner_user_id = %s OR
                         (owner_user_id IS NULL AND owner_session_id = %s)
                   ORDER BY created_at DESC LIMIT %s""",
                (user_id, session_id, int(limit)),
            )
            rows = [dict(row) for row in cursor.fetchall()]
    return {
        "items": [
            {
                "benchmark_id": str(row["id"]),
                "label": str(row.get("label") or ""),
                "status": str(row.get("status") or ""),
                "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
                "finished_at": row["finished_at"].isoformat() if row.get("finished_at") else None,
                "config": dict(row.get("config") or {}),
                "estimate": (row.get("summary") or {}).get("estimate"),
                "summary": _summarize_list_row(row),
                "error_message": row.get("error_message"),
            }
            for row in rows
        ]
    }


def _summarize_list_row(row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("summary") or {}
    return {
        "runs": summary.get("runs") or {},
        "quality": {
            key: (summary.get("quality") or {}).get(key)
            for key in (
                "requested",
                "final_valid",
                "first_pass_rate",
                "final_acceptance_rate",
                "repair_success_rate",
                "recovery_success_rate",
            )
        },
        "llm": summary.get("llm") or {},
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
COMPARISON_METRICS = (
    ("quality", "first_pass_rate", "First-pass acceptance"),
    ("quality", "validation_coverage", "Validation coverage"),
    ("quality", "final_acceptance_rate", "Final acceptance"),
    ("quality", "repair_success_rate", "Repair success"),
    ("quality", "recovery_success_rate", "Recovery success"),
    ("quality", "final_valid", "Finalized questions"),
    ("llm", "estimated_cost_usd", "Estimated cost (USD)"),
    ("llm", "input_tokens", "Input tokens"),
    ("llm", "output_tokens", "Output tokens"),
    ("llm", "calls", "LLM calls"),
    ("llm", "failed", "Failed calls"),
    ("llm", "retries", "Retries"),
    ("llm", "avg_latency_ms", "Average call latency (ms)"),
    ("quality", "duration_ms", "Generation duration (ms)"),
)


def compare_benchmarks(
    benchmark_ids: list[str], *, session_id: str, user_id: str
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    with db.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            for benchmark_id in benchmark_ids:
                row = _owned_benchmark(cursor, benchmark_id, session_id, user_id)
                if row is None:
                    raise BenchmarkError(f"Benchmark not found: {benchmark_id}")
                details.append(_benchmark_payload(cursor, row))
    base = details[0]
    comparisons = []
    for other in details[1:]:
        differences = _comparability_differences(base, other)
        comparisons.append({
            "benchmark_id": other["benchmark_id"],
            "label": other["label"],
            "comparable": not differences,
            "differences": differences,
            "metrics": _metric_deltas(base, other),
        })
    return {"base": base, "comparisons": comparisons}


def _comparability_differences(
    left: dict[str, Any], right: dict[str, Any]
) -> list[str]:
    differences: list[str] = []
    if left.get("config_fingerprint") != right.get("config_fingerprint"):
        differences.append("Generation configuration differs")
    left_repro = left.get("reproducibility") or {}
    right_repro = right.get("reproducibility") or {}
    for field, label in (
        ("llm_provider", "LLM provider"),
        ("llm_model", "LLM model"),
    ):
        if left_repro.get(field) != right_repro.get(field):
            differences.append(f"{label} differs")
    left_versions = left_repro.get("versions") or {}
    right_versions = right_repro.get("versions") or {}
    for key in ("assessment_catalog", "assessment_compiler"):
        if left_versions.get(key) != right_versions.get(key):
            differences.append(f"{key.replace('_', ' ').title()} version differs")
    return differences


def _metric_deltas(base: dict[str, Any], other: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    base_summary = base.get("summary") or {}
    other_summary = other.get("summary") or {}
    for section, field, label in COMPARISON_METRICS:
        base_bucket = base_summary.get(section) or {}
        other_bucket = other_summary.get(section) or {}
        if section == "llm" and not base_bucket.get("telemetry_available", True):
            base_value = None
            other_value = None
        elif section == "llm" and not other_bucket.get("telemetry_available", True):
            base_value = None
            other_value = None
        else:
            base_value = base_bucket.get(field)
            other_value = other_bucket.get(field)
        delta = None
        if base_value is not None and other_value is not None:
            try:
                delta = float(other_value) - float(base_value)
            except (TypeError, ValueError):
                delta = None
        rows.append({
            "section": section,
            "metric": field,
            "label": label,
            "base": base_value,
            "other": other_value,
            "delta": delta,
        })
    return rows
