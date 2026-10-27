# Pipeline Optimization Implementation Report

Date: 25 October 2026 (Asia/Dubai)

## Outcome

DraftWork now schedules one generation unit per Planner slot through the existing `generate_exams_node`, with a configurable ceiling of 50 concurrent calls. Results remain keyed by slot and are merged in Planner submission order by the coordinator. The existing LangGraph, Catalog, Planner, difficulty adjustment, validation, duplicate, checkpoint, persistence, export, and partial-exam paths remain in place.

The implementation also compacts Validator evidence, bounds structured-output corrective payloads, separates viable correctness candidates from provider/schema failures in repair accounting, adds assignment-scoped repair budgets and progress classification, strengthens role-specific validation cache identity, and versions duplicate pair facts.

No paid provider call, production job, or PostgreSQL write was made.

## Files and functions changed

- `app/online/exam_builder.py`
  - Added `_dispatch_generation_jobs` as a bounded, work-conserving dispatcher.
  - Changed `generate_exams_node` from objective/type bundles to one immutable job per slot.
  - Added cancellation/ownership checking immediately before queued slot work begins.
  - Kept coordinator-only result merging, stable Planner order, stable public IDs, checkpoint writes, MCQ balancing, and deterministic FITB bank reconstruction.
  - Added deterministic True/False polarity requirements to slot contracts and rejects responses that violate them.
  - Added generation/validation/duplicate semantic versions to `immutable_exam_request` so incompatible resumes fail request-fingerprint matching.
- `app/config.py`, `.env.example`, `.env.production.example`, `compose.yaml`
  - Raised the generation, per-job, and DeepSeek provider ceilings to 50. All remain configurable and bounded.
  - Added generation, validation, and duplicate pipeline version settings.
- `app/online/validator.py`
  - Replaced duplicated `plan` plus `expected_catalog_task` Validator entries with one canonical `task_contract`.
  - Added one request-level evidence registry; questions carry explicit assigned evidence IDs.
  - Added source hashes and rule/schema/model identities to role-specific validation fingerprints.
  - Expanded STEM MCQ dependency closure so a mathematical stem change can repair options, answer binding, and certificate coherently.
  - Changed correctness accounting to increment only after a candidate is produced for validation.
  - Added three-candidate assignment budgets, a six-candidate global slot cap, assignment fingerprints, bounded attempt history, strategy escalation, and `PASSED`/`UNCHANGED`/`LATERAL`/`REGRESSED` outcomes.
- `app/llm/deepseek.py`
  - Bounded corrective task context to 24,000 characters, prior output to 8,000, and failure text to 2,000.
  - Removed duplicated JSON Schema text because the provider receives the authoritative schema through structured-output configuration on every attempt.
  - Honors bounded `Retry-After` values for 429 responses while retaining exponential backoff.
- `app/online/duplicate_judge.py`
  - Added screening and pair-ledger versions to the content/model/prompt cache key. Question/answer, task reuse context, model, and rule changes invalidate affected pair facts; unchanged pairs continue to reuse facts.
- Tests updated or added in `tests/test_pipeline_optimization.py`, `tests/test_exam_builder.py`, `tests/test_deployment_foundation.py`, `tests/test_question_validator.py`, and `tests/test_repair_manifest.py`.

## Implementation by phase

### Independent generation and 50-call scheduling

Every planned MCQ, True/False, FITB, Definition, Why, Equation, Word Problem, and Essay slot is submitted separately. A 23-slot model therefore creates 23 independent scheduler jobs. A 100-slot model creates 100 jobs; the executor starts 50, then starts queued work as individual calls complete. It does not wait for waves.

Each worker receives a deep-copied plan item and slot-only context. Existing dynamic one-slot schemas enforce the expected `slot_id`; variation binding rejects incompatible task/variation output. Existing bounded retries retain successful sibling outputs because each sibling is now a separate job. The coordinator appends results in Planner order, assigns public IDs, reconstructs the FITB bank once, performs MCQ answer-label balancing, and checkpoints the merged model.

DeepSeek and the shared per-job scheduler now both permit 50 requests. Provider 429 responses honor `Retry-After`. Actual live concurrency can still be lower when deployment overrides these settings or the provider delays/rejects admission; that effective value must be observed in controlled live telemetry.

### Feasibility and deterministic preparation

Existing production gates were retained: `validate_plan_catalog_bindings`, Planner hard eligibility, the dynamic 100-point ranking, selected-variation binding, exact evidence assignment, fixed FITB answers, local structural checks, duplicate-option checks, MCQ binding checks, and deterministic equation/system checks. Construction tasks continue to allow authorized variable givens rather than requiring literal source numbers. Unsupported task/evidence findings continue to route to Catalog replacement.

The implementation did not introduce a broad new SymPy template system. Current deterministic solvers apply where the repository already has a safe mathematical representation; ambiguous natural-language mathematics remains under independent General/STEM review.

### Compact complete validation

Initial validation still covers every generated question and waits for all General/STEM batches before repair routing. Strict schemas, missing-ID salvage/single retry, and fail-closed `UNVALIDATED` behavior remain unchanged.

Each Validator batch now carries one canonical task contract per question and one deduplicated evidence registry per request. Per-question evidence IDs preserve hard evidence boundaries. Existing batches remain small and concurrent; no whole-exam mega-prompt was introduced.

Corrective structured-output attempts no longer append an unbounded previous response or duplicate the full JSON Schema in prompt text. The authoritative schema remains attached through the provider's structured-output API.

### Repair intelligence and safety

Provider/transport/schema retries remain bounded by provider settings. They no longer automatically consume a correctness candidate attempt. A slot consumes correctness budget only when a generated candidate is ready for General/STEM evaluation.

Budgets are now:

- At most three validated candidates for one task/variation/evidence assignment fingerprint.
- At most six validated candidates across the slot.
- Assignment replacement receives its own bounded qualification allowance without resetting the global cap.

Repair history records assignment identity, candidate task, strategy, requested fields, revalidation result, new failure, and progress. A failed targeted patch escalates to same-assignment regeneration; repeated failure escalates to assignment replacement. Replacement blacklists and current Catalog selection remain in use.

Candidate exams and plans remain staged. The active question/plan is committed only after required correctness validators pass. Duplicate replacement retains its precommit correctness and duplicate checks and restores the old plan when generation or qualification fails.

### Selective revalidation

Existing selective revalidation remains. The fingerprint now includes question content, answer/options/certificate, plan/effective instance, FITB bank dependency, assigned source text hashes, output schema version, role-specific rule version, and Validator model identity. General and STEM role fingerprints are stored independently. Old ledger entries without these identities become stale and are safely revalidated.

### Incremental duplicate review

The existing content-keyed fact cache already avoided repeated Duplicate Judge calls for unchanged pairs. Its key now explicitly includes screening and ledger versions in addition to Judge prompt version, model, normalized question/answer content, and reuse context. A changed question or answer invalidates only keys involving that content; unchanged pair facts remain reusable. Existing candidate blacklists, family/operation/target/work-product diversity context, reservation, and precommit checks remain active.

## Verification

### Test results

- Focused generation/validation/repair suite: **175 passed**.
- Graph, duplicate, lifecycle, contract, and hardening suite: **124 passed, 1 skipped**.
- Final full repository suite: **852 passed, 11 skipped in 68.23 seconds**.
- Python compilation checks passed for all changed production modules.
- `git diff --check` reported no whitespace errors.

The skipped tests are environment-dependent tests already marked by the suite; no failure was hidden or converted to a skip.

### Offline scheduling benchmark

The production dispatcher was exercised with deterministic sleeping workers and no provider calls:

| Workload | Ceiling | Peak in flight | Completed | Wall time |
|---:|---:|---:|---:|---:|
| 23 slots | 50 | 23 | 23 | 0.0288 s |
| 100 slots | 50 | 50 | 100 | 0.0583 s |

The synthetic worker latency was 0.020 seconds. These numbers verify scheduling, peak concurrency, queue draining, and result collection; they are not live-provider latency estimates.

### Validator payload measurement

An eight-question deterministic fixture shared one 28,000-character source chunk:

| Serialization | Bytes |
|---|---:|
| Previous duplicated per-entry payload only | 209,480 |
| New complete prompt with evidence registry and instructions | 28,781 |
| Absolute reduction | 180,699 |

This fixture proves evidence deduplication. It is not a prediction of live token savings because tokenizer behavior and evidence overlap vary by exam.

The corrective-retry stress test uses a 100,000-character task, 100,000-character prior output, and 100,000-character schema; the resulting retry remains below 35,000 characters and contains no duplicated schema text.

## Baseline comparison

| Metric | Live baseline | Offline result | Interpretation |
|---|---:|---:|---|
| Runtime excluding Catalog | 986.212 s | Not comparable | No paid/live provider run was allowed. |
| Input tokens | 5,696,128 | Not measured as provider tokens | Byte-level prompt reduction was measured instead. |
| Initial generation | 123.904 s | 23 synthetic jobs in 0.0288 s | Scheduling proof only. |
| Revalidation input | 3,728,461 | Not measured as provider tokens | Source deduplication and bounded retries directly remove confirmed amplification. |
| Final valid questions | 21/23 | Deterministic test fixtures pass | Mock fixtures cannot prove live mathematical quality. |
| Unresolved duplicate groups | 1 | Duplicate regressions pass | No new Judge call was made. |

No savings percentage is claimed.

## Features retained

The LangGraph topology, Catalog/compiler format, shared Planner eligibility and ranking, difficulty adjustment, General/STEM independence, strict schemas, lifecycle records, checkpoint revisions, request fingerprints, PostgreSQL fencing, saved-exam shape, Arabic/English behavior, exports, and partial-exam warnings remain intact. Successful generation siblings and the last accepted question survive failed generation, repair, and duplicate candidates.

## Remaining limitations and risks

- Live provider RPM, TPM, account concurrency, p50/p95 latency, and 429 behavior are not available in repository configuration. The ceiling is 50 and 429 backoff is adaptive, but there is no proactive rolling RPM/TPM token bucket because authoritative provider quotas are unknown. Deployments should set the three concurrency variables to verified account limits and perform a controlled live canary.
- The offline benchmark validates scheduling and state behavior, not answer quality, source-grounding quality, or live token counts.
- Deterministic mathematics remains intentionally limited to represented checks. General symbolic certificate verification across all eight types still requires STEM validation.
- Independent generation removes cross-slot shared context. The Planner contract, deterministic answer balancing, and duplicate review mitigate this, but a controlled live A/B test is still required to quantify duplicate rate and question-quality regressions.
- Existing provider schemas permit bounded internal retries. These are separately bounded but are not persisted as individual repair-history candidates because they do not produce an evaluable question.

## Controlled rollout

Use the semantic pipeline versions in the immutable request fingerprint to prevent incompatible checkpoint resumes. Begin with a non-production frozen-plan canary, record effective in-flight calls, RPM/TPM, 429s, p50/p95 latency, generation completeness, Validator disagreement, repair convergence, duplicate rate, and final mathematical audit results. If provider capacity is lower, reduce the provider admission setting while retaining the 50-call generation ceiling and work-conserving queue.
