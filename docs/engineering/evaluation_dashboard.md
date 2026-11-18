# Evaluation Dashboard

The evaluation dashboard measures DraftWork's multi-agent exam pipeline. It
replaces the single-page `eval.html` telemetry view with four focused pages and
reads only durable, recorded telemetry.

## Pages

| Page | Purpose |
| --- | --- |
| `FrontEnd/evaluation/overview.html` | Headline quality, reliability, cost and latency, daily trends and the question funnel. |
| `FrontEnd/evaluation/reports.html` | Searchable list of generation runs and a full report for one run (quality, cost, stage breakdown, recovery, telemetry gaps). |
| `FrontEnd/evaluation/pipeline.html` | Question-type, validation, repair, Recovery V3, lifecycle, duplicate, cache and provider analytics. |
| `FrontEnd/evaluation/benchmarks.html` | Configure controlled benchmark runs, watch execution, inspect results and compare benchmarks. |

`/eval.html` remains the entry point and redirects to
`evaluation/overview.html`. Historical outcome telemetry is backfilled from
`exams.metadata.generation_audit` by
`scripts/backfill_evaluation_analytics.py` (idempotent; only derived columns are
updated). All pages share `css/evaluation/base.css`,
`js/evaluation/api.js`, `utils.js`, `components.js` and `charts.js`. The pages
are static ES modules served by the same FastAPI application; no frontend
framework or external chart library is used.

## Data flow

```
generation task (Celery)
  ├─ eval_stats counters ──────────────┐
  ├─ slot_lifecycle / repair history ──┤
  ├─ duplicate + cache flags ──────────┤
  └─ measured job duration, job id ────┤
                                       ▼
                         evaluation_runs (one row per generation run)
                                       │
 jobs ───────────────┐                 │
 llm_request_events ─┼── joined by job_id
 exam_audit_events ──┘                 │
                                       ▼
                 /api/v1/evaluation/* (aggregation + analytics)
                                       │
                                       ▼
                          four dashboard pages
```

## API

All endpoints are under `/api/v1/evaluation` and require an authenticated
Clerk user (`require_user_id`). When `EVALUATION_DASHBOARD_TOKEN` is configured,
a request carrying `X-Evaluation-Token` may call **read** endpoints only.
Benchmark creation and cancellation always require a signed-in user and a
browser session.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/overview` | KPIs, totals, averages per run/model exam, trends, funnel, recent runs. `days` (0 = all), `include_benchmarks`. |
| GET | `/exams` | Paginated run list. `search`, `status`, `benchmark=exclude|include|only`, `days`, `limit`, `offset`. |
| GET | `/exams/{exam_id}` | Full per-run report. |
| GET | `/pipeline` | Component analytics for the selected window. |
| GET | `/benchmarks/options` | Ready documents, question types, difficulty levels and limits. |
| POST | `/benchmarks/estimate` | Historical cost/call/duration estimate for a configuration. |
| GET | `/benchmarks` | Benchmarks owned by the caller. |
| POST | `/benchmarks` | Create and start a benchmark. Supports `Idempotency-Key`. |
| GET | `/benchmarks/{id}` | Benchmark status and results; advances orchestration. |
| POST | `/benchmarks/{id}/cancel` | Cancel the active run and the benchmark. |
| GET | `/benchmarks/compare?ids=a,b` | Metric deltas and comparability. |

The legacy public aggregate `/api/v1/eval-summary` (and its cached aliases)
remains available for older consumers and now reports corrected counts.

## Persistence

Migration `0009_evaluation_analytics_benchmarks` adds:

- `evaluation_runs.job_id`, `duration_ms` — links to provider/job telemetry.
- `evaluation_runs.validation_failure_actions` — repair-action distribution.
- `evaluation_runs.recovery_outcomes` — recovery, repair-attempt, terminal slot
  and recovery-failure data derived from the authoritative `slot_lifecycle`.
- `evaluation_runs.duplicate_outcomes` — duplicate rounds, replacements,
  replacement failures and unresolved groups.
- `evaluation_runs.catalog_cache` — catalog/compiler/compatibility cache flags.
- `evaluation_runs.benchmark_id` (FK, `ON DELETE SET NULL`), `failure_code`,
  `failure_message`.
- `status` now permits `cancelled`.
- `evaluation_benchmarks` and `evaluation_benchmark_runs` tables.
- Indexes on `job_id`, `benchmark_id` and benchmark ownership.

`evaluation_runs.exam_id` is the primary key and the write uses
`INSERT … ON CONFLICT (exam_id) DO UPDATE`, so a retried or resumed job updates
its record instead of failing on a duplicate key. Failed and cancelled jobs now
write a terminal row (`status=failed|cancelled`, every requested question counted
as missing) so pipeline reliability is measurable per run. If the same exam later
completes, the row is replaced by the normal evaluation record.

Writes fall back to the legacy column set when migration 0009 has not been
applied; reads use `to_jsonb(r)` so historical schemas still render.

## Metric definitions

All rates are `None` (shown as “—”) when the denominator is zero. No value is
estimated except the explicitly labelled LLM cost estimate.

### Question quality and reliability

| Metric | Definition | Source |
| --- | --- | --- |
| Generation completion | `generated_first / requested_questions` | eval counters |
| Generation rejection rate | `generation_rejected / (generation_rejected + generated_first)` | eval counters |
| Validation coverage | `validation_total_first / generated_first` | eval counters |
| First-pass acceptance | `validation_passed_first / validation_total_first` (questions that received a verdict) | eval counters |
| Repair success | `repair_succeeded / repair_sent` (unique questions whose write-back later passed) | eval counters |
| Final acceptance | `final_valid / requested_questions` | eval counters |
| Recovery success | `recovered / entered_recovery` (slots entering repair/replacement that were finalized) | `recovery_outcomes` from `slot_lifecycle` |
| Recovery attempt success | accepted repair attempts / recorded attempts | `validation_repair_history` |
| Job success | completed jobs / (completed + failed + cancelled) | `jobs` |

### Requests, model exams and questions

| Metric | Definition | Source |
| --- | --- | --- |
| Known generation requests | Evaluation records plus tracked generation jobs that never produced one, de-duplicated by request | `evaluation_runs` ∪ `jobs` |
| Request completion | Completed known requests / finished known requests | derived |
| Tracked job success | Completed jobs / finished tracked jobs (separate, smaller population) | `jobs` |
| Model exams requested | Σ `models_count` + `num_models` of jobs without evaluations | `evaluation_runs`, `jobs` |
| Model exams produced | Σ length of persisted `exams.exams` payloads | `exams` |
| Model exams unavailable | Requested models whose legacy record has no exam payload | `exams` |
| Question shortfall | Requested questions of finished requests minus finalized questions (active requests excluded) | derived |

### Cost, tokens and latency

Provider telemetry is partitioned into four non-overlapping scopes:
**evaluated runs**, **other generation jobs** (failed/cancelled jobs that still
spent tokens), **other application jobs** (ingestion/title), and **unattributed
legacy requests** (`job_id` null). `all recorded` is their sum, so cost is never
counted twice.

| Metric | Definition | Source |
| --- | --- | --- |
| Estimated cost | Σ `estimated_cost_usd` per provider request | `llm_request_events` |
| Cost per finalized question | evaluated-scope cost / questions finalized by runs with provider telemetry | derived |
| Cost per generation run | evaluated-scope cost / evaluated runs with provider requests | derived |
| Cost per model exam | evaluated-scope cost / model exams of those runs (equal allocation) | derived |
| Average cost / input / output tokens per run | daily or window totals divided by runs with provider telemetry (never total daily spend) | derived |
| Tokens | Σ input / output / reasoning / cached tokens over calls that reported usage | `llm_request_events` |
| Token coverage | calls with usage / total calls | `llm_request_events` |
| Call latency | average / p50 / p95 of `duration_ms` | `llm_request_events` |
| Exam duration | stored `duration_ms`, else job timestamps, else the span of `exam_audit_events` | `evaluation_runs`, `jobs`, `exam_audit_events` |
| Job duration | p50 / p95 of `finished_at - started_at` | `jobs` |
| Queue time | average `started_at - created_at` | `jobs` |

Cost is an **estimate** computed from the configured per-million-token prices; it
is not provider billing. Token sums are **partial** when some calls did not report
usage; the UI always shows `N of M calls reported usage`.

### Stage timing

Stage wall time is reconstructed from `exam_audit_events`: a `pipeline_stage`
event starts a stage, and each later event closes the interval attributed to that
stage. Summing intervals yields recorded wall time per stage, including repeated
validation/repair cycles. This is deliberately **not** derived from provider call
durations.

The “Σ call time” column sums individual LLM request durations. Parallel calls
can make it exceed stage wall time; it is labelled as call time, not stage time.

### Benchmark metrics

Benchmark summaries aggregate the evaluation rows whose `benchmark_id` matches the
benchmark and the provider events of their jobs. They report the same quality,
recovery, token, cost, latency and duration metrics as exam reports, plus run
status and reproducibility metadata.

## Benchmarks

- A benchmark configures document, section selection, question counts, model
  versions, difficulty, repeats and stop-on-failure.
- Each run is an ordinary `exam_generation` job created through
  `repositories.create_generation_job(..., benchmark=True)`. It reuses the
  production pipeline and cached assessment catalogs. Benchmark jobs bypass the
  per-minute generation limit but still honour the active-job limit.
- Jobs and evaluation rows carry an explicit `benchmark_id`; ordinary dashboards
  exclude benchmark records by default.
- Orchestration is polling-driven and idempotent: `GET /benchmarks/{id}` takes a
  PostgreSQL advisory lock, observes the active run, starts the next run when
  needed, and finalizes the benchmark when all repeats are terminal. Duplicate
  paid submissions are prevented by the run-state check, the advisory lock and
  the optional `Idempotency-Key`.
- An estimate is shown before creation from actual historical per-question
  averages; unavailable estimates are labelled instead of fabricated.
- Benchmark results distinguish **configured**, **initially generated**,
  **finalized** and **failed or missing** questions. Unresolved slots are listed
  with their recorded failure stage and code, for example
  `no_valid_source_grounded_replacement` or `recovery_v3_plan_exhausted`.
- Token coverage counts calls that reported usage (`185 of 185`), independent of
  the token sums. A zero generation-rejection count is labelled as measured when
  generation telemetry exists, not as missing data.
- Reproducibility metadata stores git commit, provider/model, pipeline versions,
  validation schema and rule versions, document/catalog identifiers and the
  execution timestamp.
- Comparisons require equivalent generation configuration (document, section
  selection, question counts, model versions, difficulty) and flag differences in
  provider/model or catalog/compiler versions. Non-comparable deltas are shown
  only with an explicit warning.

## Historical data and limitations

- Thirty-four of the historical evaluation records predate job tracking, and
  fourteen have no persisted exam payload. Headline evaluation metrics still
  include their recorded question counters; model-exam production is reported as
  unavailable for them rather than estimated.
- Historical recovery, repair-attempt, duplicate and cache telemetry is
  reconstructed from `exams.metadata.generation_audit` and stored by the
  backfill script. Original exam metadata is never modified.
- Durations for runs without stored telemetry are reconstructed from job
  timestamps or the audit-event span. Runs with neither remain unavailable.
- First-pass acceptance excludes questions that never received a verdict;
  validation coverage is shown next to it.
- Provider events recorded before stage attribution appear as
  “Unattributed (legacy)” in the cost-by-stage chart.
- Per-question-type cost is not attributable: provider calls are not tagged with
  question type. It is intentionally omitted.
- Per-node wall time is only available when `exam_audit_events` exist for the run.
- Failed-run scoring uses `failure_code` / `failure_message` from the worker; the
  dashboard displays them but performs no root-cause inference.

## Recommended next steps

1. Persist `performance_telemetry` (per-node timings, queue and scheduler data)
   with each run for richer latency attribution.
2. Attribute provider calls to question types by carrying question-type metadata
   through validator/repair attribution.
3. Add per-validator-role counters (general, quantitative, grounding, duplicate).
4. Record difficulty calibration (requested vs observed) as a first-class metric.
5. Add trend/run drill-down behind authenticated routes and optional SLO alerts.
6. Support a curated “quality rubric” benchmark mode once an independent judge is
   approved; it is intentionally out of scope today.

## Verification

```powershell
.\.venv\Scripts\python.exe scripts\apply_migrations.py
.\.venv\Scripts\python.exe scripts\backfill_evaluation_analytics.py
.\.venv\Scripts\python.exe -m pytest tests\test_eval_stats.py tests\test_evaluation_store.py `
  tests\test_eval_dashboard.py tests\test_benchmarks.py -q
$env:RUN_POSTGRES_TESTS=1
.\.venv\Scripts\python.exe -m pytest tests\test_evaluation_analytics_db.py -q
```

Database tests create labelled synthetic records and delete them afterwards.
