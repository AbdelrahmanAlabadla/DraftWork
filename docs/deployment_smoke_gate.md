# Pre-Deployment Smoke Gate

## Protected command

```bash
python scripts/deploy_worker.py
```

This is the only supported worker release path. It runs the full gate, then
promotes the exact candidate image and verifies the running worker's source
fingerprint.

## Fast developer checks

```bash
python scripts/static_check.py          # imports + AST undefined/shadow checks
python scripts/pre_deploy_gate.py --tests-only   # static + smoke tests (~20s)
```

## Full gate

```bash
python scripts/pre_deploy_gate.py
```

The full gate:

1. Compiles/imports the production modules and checks for undefined names and
   helper shadowing (`scripts/static_check.py`).
2. Runs the fast production-path smoke suite: real validator orchestration,
   Repair telemetry, single- and two-model cross-model recovery, duplicate
   batching, canonical-request reconstruction, missing-slot manifest, graph
   routing, cancellation, assembly, persistence/export handoff.
3. Computes a SHA-256 fingerprint of `app/**/*.py`, `requirements.txt`, and
   `Dockerfile`.
4. Builds `genexam-worker:smoke-<fingerprint>` with the fingerprint and git
   commit embedded.
5. Starts that candidate image and verifies its `APP_SOURCE_FINGERPRINT`
   matches the tested source.
6. Starts an isolated candidate Celery worker on queue `smoke-<uuid>` (no beat,
   solo pool), waits for its health check, then removes it. The production
   worker is never touched.

## Why the fingerprint matters

Host tests alone are insufficient because the worker runs a baked image. The
candidate image carries `APP_SOURCE_FINGERPRINT`; the gate refuses to continue if
the image and host source differ. `scripts/deploy_worker.py` re-checks the
running container after promotion.

## Candidate isolation

- Queue: unique `smoke-*` queue, so no production job is consumed.
- No `--beat`; no scheduled production work.
- Embeddings forced to CPU and no CUDA requirement.
- Container is removed in a `finally` block.

## Unprotected paths

There is no CI workflow in this repository, so the following still bypass the
gate:

```bash
docker compose build worker
docker compose up -d worker
```

Do not use them for releases. A CI job (when added) should run
`python scripts/pre_deploy_gate.py` and deploy only from its candidate tag.
Ordinary application (`app`) deployments are outside this worker gate.

## Docker build context

`.dockerignore` excludes `.venv`, `data`, `artifacts`, `reports`, `backups`,
forensic folders, `tests`, `tmp`, and archives. Keeping the context small is
required for a fast gate; the ARG/ENV fingerprint block is placed after the pip
install layer so source changes do not invalidate dependency caching.
