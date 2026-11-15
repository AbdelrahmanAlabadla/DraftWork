"""Pre-deployment gate for DraftWork worker images.

Runs:
1. static/import checks for the refactored Validator architecture;
2. the fast production-path smoke suite (real orchestration, mocked LLMs);
3. builds a candidate worker image and verifies it contains the tested revision;
4. starts the candidate Celery worker on an isolated queue and confirms it
   becomes healthy without touching the production worker.

Use ``scripts/deploy_worker.py`` for protected deployment.  Plain
``docker compose up`` bypasses this gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MARKER = ROOT / "tmp" / "pre_deploy_candidate.json"
SMOKE_TESTS = [
    "tests/test_production_path_smoke.py",
    "tests/test_full_pipeline_mock.py::test_full_two_model_pipeline_real_validation_assembly_and_export",
    "tests/test_draftwork_40q_readonly_e2e.py::test_40q_mocked_end_to_end_pipeline",
    "tests/test_error_logging_and_checkpoint_safety.py::test_repair_coordinator_telemetry_path_calls_valid_recorder",
    "tests/test_independent_model_recovery.py::test_models_run_isolated_local_lifecycles_and_cross_model_runs_after",
    "tests/test_independent_model_recovery.py::test_single_model_cross_model_recovery_does_not_shadow_model_helper",
    "tests/test_independent_model_recovery.py::test_one_cross_model_target_completes_targeted_recovery_and_final_scan",
    "tests/test_independent_model_recovery.py::test_two_model_cross_model_targets_run_concurrently_and_merge",
    "tests/test_exam_bfe29ab579f1_regressions.py::test_batched_candidate_gate_rejects_only_the_conflicting_candidate",
    "tests/test_multi_validator.py::test_missing_validator_request_record_is_reconstructed_from_authoritative_exam",
    "tests/test_repair_manifest.py::test_missing_slot_manifest_derives_question_type_without_nameerror",
    "tests/test_static_check.py",
    "tests/test_graph_pipeline_integration.py::test_state_graph_preserves_contracts_through_recovery_lifecycles",
    "tests/test_graph_pipeline_integration.py::test_resume_router_keeps_durable_plan_and_partial_questions",
    "tests/test_pipeline_completion_and_versioned_repair.py::test_cancelled_job_schedules_no_validation_or_recovery_work",
]
IMAGE_REPO = "genexam-worker"
NETWORK = os.environ.get("GENEXAM_SMOKE_NETWORK", "genexam_default")


def run(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=timeout,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = sorted(
        [*(ROOT / "app").rglob("*.py"), ROOT / "requirements.txt", ROOT / "Dockerfile"]
    )
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"GATE FAILED: {message}")


def static_checks() -> None:
    result = run([sys.executable, str(ROOT / "scripts" / "static_check.py")])
    require(result.returncode == 0, "static/import checks failed")


def smoke_tests(extra_tests: list[str] | None = None) -> None:
    result = run([
        sys.executable, "-m", "pytest", *SMOKE_TESTS, *(extra_tests or []),
        "-q", "-p", "no:cacheprovider",
    ])
    require(result.returncode == 0, "production-path smoke tests failed")


def build_candidate(fingerprint: str) -> str:
    commit = run(["git", "rev-parse", "HEAD"]).stdout.strip() or "unknown"
    tag = f"{IMAGE_REPO}:smoke-{fingerprint[:12]}"
    result = run([
        "docker", "build",
        "--build-arg", f"GIT_COMMIT={commit}",
        "--build-arg", f"SOURCE_FINGERPRINT={fingerprint}",
        "-t", tag, ".",
    ])
    require(result.returncode == 0, "candidate image build failed")
    return tag


def verify_image(tag: str, fingerprint: str) -> None:
    probe = (
        "import os; "
        "print(os.environ.get('APP_SOURCE_FINGERPRINT','')); "
        "import app.online.validator, app.online.validation.orchestrator, "
        "app.online.validation.recovery, app.online.independent_recovery, "
        "app.online.graph, app.jobs.celery_app"
    )
    result = run(["docker", "run", "--rm", "--entrypoint", "python", tag, "-c", probe])
    require(result.returncode == 0, "candidate image import probe failed")
    require(
        fingerprint in result.stdout.splitlines(),
        "candidate image source fingerprint does not match tested source",
    )


def docker_database_url() -> str | None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("DOCKER_DATABASE_URL="):
            return line.split("=", 1)[1].strip() or None
    return None


def verify_candidate_worker(tag: str, fingerprint: str) -> None:
    name = f"genexam-worker-smoke-{uuid.uuid4().hex[:8]}"
    queue = f"smoke-{uuid.uuid4().hex[:8]}"
    health = (
        "python -c \"import os,sys,psutil; alive=any('celery' in ' '.join(p.info.get('cmdline') or []) "
        "and 'worker' in (p.info.get('cmdline') or []) for p in psutil.process_iter(['cmdline'])); "
        "sys.exit(0 if alive else 1)\""
    )
    command = [
        "docker", "run", "-d", "--name", name,
        "--network", NETWORK,
        "--env-file", str(ROOT / ".env"),
        "-e", "APP_ENV=test",
        "-e", "EMBEDDING_REQUIRE_CUDA=false",
        "-e", "EMBEDDING_DEVICE=cpu",
        "-e", "CELERY_BROKER_URL=redis://redis:6379/0",
        "-e", f"APP_SOURCE_FINGERPRINT={fingerprint}",
    ]
    database_url = docker_database_url()
    if database_url:
        command.extend(["-e", f"DATABASE_URL={database_url}"])
    command.extend([
        "--health-cmd", health,
        "--health-interval", "5s",
        "--health-timeout", "5s",
        "--health-retries", "6",
        "--entrypoint", "celery",
        tag, "-A", "app.jobs.celery_app:celery_app", "worker",
        "--loglevel=INFO", "--pool=solo", "--concurrency=1",
        "-Q", queue, "--hostname", f"{queue}@%h",
    ])
    try:
        started = run(command)
        require(started.returncode == 0, "candidate worker failed to start")
        deadline = time.time() + 75
        state = "starting"
        while time.time() < deadline:
            inspect = run([
                "docker", "inspect", "--format",
                "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
                name,
            ])
            fields = inspect.stdout.strip().split()
            state = fields[1] if len(fields) > 1 else fields[0] if fields else "unknown"
            if state == "healthy":
                break
            if fields and fields[0] == "exited":
                break
            time.sleep(3)
        if state != "healthy":
            logs = run(["docker", "logs", "--tail", "80", name])
            raise SystemExit(f"GATE FAILED: candidate worker health={state}")
        print(f"Candidate worker healthy: {name} (queue {queue})")
    finally:
        run(["docker", "rm", "-f", name])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-only", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument(
        "--extra-test", action="append", default=[],
        help="Additional test node (used only to verify the gate itself).",
    )
    args = parser.parse_args()
    if shutil.which("docker") is None:
        raise SystemExit("GATE FAILED: docker is required")
    static_checks()
    if args.static_only:
        print("GATE PASS (static only)")
        return 0
    smoke_tests(args.extra_test)
    if args.tests_only:
        print("GATE PASS (tests only)")
        return 0
    fingerprint = source_fingerprint()
    if args.skip_image:
        print(f"GATE PASS (image skipped) fingerprint={fingerprint}")
        return 0
    tag = build_candidate(fingerprint)
    verify_image(tag, fingerprint)
    verify_candidate_worker(tag, fingerprint)
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps({
        "image": tag, "fingerprint": fingerprint, "tested_at": time.time(),
    }), encoding="utf-8")
    print(f"GATE PASS image={tag} fingerprint={fingerprint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
