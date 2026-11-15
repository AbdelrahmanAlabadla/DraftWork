"""Protected worker deployment.

Only this command runs the mandatory pre-deployment gate and then promotes the
exact candidate image that passed it.  Plain ``docker compose up`` bypasses the
gate and must not be used for worker releases.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = ROOT / "tmp" / "pre_deploy_candidate.json"


def run(command: list[str]) -> subprocess.CompletedProcess:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def main() -> int:
    run([sys.executable, str(ROOT / "scripts" / "pre_deploy_gate.py")])
    if not MARKER.exists():
        raise SystemExit("DEPLOY FAILED: gate marker missing")
    payload = json.loads(MARKER.read_text(encoding="utf-8"))
    image = str(payload["image"])
    fingerprint = str(payload["fingerprint"])
    run(["docker", "tag", image, "genexam-worker:latest"])
    run([
        "docker", "compose", "up", "-d", "--no-deps", "--no-build",
        "--force-recreate", "worker",
    ])
    inspect = subprocess.run(
        ["docker", "inspect", "genexam-worker", "--format", "{{json .Config.Env}}"],
        cwd=ROOT, text=True, capture_output=True,
    )
    if inspect.returncode != 0:
        raise SystemExit("DEPLOY FAILED: cannot inspect deployed worker")
    env = inspect.stdout.strip()
    if fingerprint not in env:
        raise SystemExit(
            "DEPLOY FAILED: running worker does not match the tested source fingerprint"
        )
    print(f"DEPLOYED worker image={image} fingerprint={fingerprint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
