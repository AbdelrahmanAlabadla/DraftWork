# Post-Live Exam Root Cause Audit

## Executive summary

**CONFIRMED:** the paid job `dc0ae6f9-89be-4d1d-8d97-39f7136e3968` did not run the current workspace implementation. The Docker worker image and the workspace have different hashes for all inspected critical files: `app/online/assessment_catalog.py`, `app/online/exam_builder.py`, `app/online/lifecycle.py`, and `app/jobs/tasks.py`. The worker had no source bind mount; only `data/` was mounted. This is the primary reason the live behavior contradicted the already-tested local fixes.

The failed run had four independent high-impact failures:

1. It calculated Catalog key `71041182…` and did not find it, despite a safe compatible finalized Catalog `339d6526…` with the same 103 child-ID set. The running worker had the compatibility helper but never invoked it. It made 11 paid compiler calls.
2. The first generation pass completed, but a generated value contained a real JSON NUL (`\\u0000`). PostgreSQL rejected its JSONB write. The original offending candidate was never durably checkpointed, so the exact slot/field cannot be reconstructed from remaining records.
3. Celery classified that database error as retriable and restarted the full task. It regenerated all 23 questions because the generation checkpoint write was the failed persistence operation.
4. The deployed independent-generation coordinator assigned IDs one one-item section at a time. Every MCQ became `model1_mcq_1`, every equation became `model1_equation_1`, and so on. Validation and repair then keyed unrelated questions under the same ID. Two real slots (`m1_mcq_10`, `m1_true_false_3`) entered a 177-cycle repair/revalidation loop until the worker was stopped and later marked `WORKER_LOST`.

The job is terminally failed in PostgreSQL: `failed`, `WORKER_LOST`, `attempt_count=2/2`, `claim_generation=2`, `checkpoint_revision=540`, finished `2026-10-26 11:46:53.468070+00`.

## Evidence sources and scope

This was read-only. Evidence came from Docker PostgreSQL (`jobs`, `documents`, `document_chunks`, `exam_audit_events`, `llm_usage`), worker logs, mounted Catalog files, Docker inspect output, and source inspection in both the workspace and live worker image. No paid calls, jobs, database writes, cache changes, image rebuilds, or restarts were made during this audit.

The report distinguishes a value that was not retained from one that was not investigated. The exact first-pass NUL-containing object and rejected FITB evidence ID are **UNKNOWN** because the running system did not persist the required raw outputs.

## 1. Catalog cache recompilation

### Evidence

The document metadata and live request were:

| Field | Value |
|---|---|
| Processed-document key | `772d2717699ca1dc2b015371744fb5d9df8f50751dc7a80b8eb63d716697f84e` |
| Domain | `stem` |
| Subject | `mathematics` |
| Language | `ar` |
| Selected persisted child chunks | 103 |
| Live calculated Catalog key | `71041182e21c363e8db8cb08cbae7be7d9369b0c518398e2f57616bc12cb880c` |

The finalized cache file [339d652657343eec2e1a6f5af7dcee91b9523cdd66d9e86f95bceb47171ae3e7.json](C:/Users/abdul/PycharmProjects/GenExam/data/assessment_catalogs/33/339d652657343eec2e1a6f5af7dcee91b9523cdd66d9e86f95bceb47171ae3e7.json) had:

- the same processed-document key;
- Catalog version `catalog-v12-grouped-type-variations`;
- a 103-ID selected set exactly equal to the live set;
- 95 tasks;
- profile subject `mathematics` and domain `STEM`.

Its stored key was `339d6526…`, so it was not an exact key match. The live identity included a `source_membership_sha256` based on chunk membership/order and the current compiler/finalizer contract. The finalizer contract hash is intentionally source-sensitive; therefore a key mismatch alone is normal after a deployment change.

**CONFIRMED:** the live worker's `prepare_assessment_catalog()` calculated only the exact key then called `_read_cache(key)`. Its `_compatible_compiler_seed()` helper existed but was not called. The workspace version calls that helper after an exact compiler-cache miss and records `assessment_catalog_compatibility_cache_hit`; the live worker did not contain that call. See [assessment_catalog.py](C:/Users/abdul/PycharmProjects/GenExam/app/online/assessment_catalog.py:1894) and [assessment_catalog.py](C:/Users/abdul/PycharmProjects/GenExam/app/online/assessment_catalog.py:1999).

### Was reuse safe?

**LIKELY SAFE, conditional on current contract validation:** the existing file had equal source membership and the expected profile/version. The current compatibility path additionally validates every task with `_cached_task_contract_valid`, so it would have rejected an unsafe legacy task shape. The live worker never performed that validation/migration, so it cannot claim the Catalog was unsafe.

The selected-list order differed from the cached file, but the cached compatibility predicate compares the canonical sorted selection. The observed order difference was not itself a source-content difference.

### Fix

P0: deploy the tested compatibility migration code, ensure it validates full task/variation/evidence contracts, then writes the exact current-key cache atomically. Add an integration test that starts the built image, not only host Python, with the 103 Docker chunks and this Catalog; assert zero assessment-compiler calls.

## 2. FITB evidence outside compiler batch

**CONFIRMED:** compiler batch retries occurred for batches logged as `batch=6` and `batch=3`, each with `invalid completion contract: FITB evidence ID is outside this compiler batch`. The compiler made 11 calls rather than the nominal nine batch calls.

The evidence ID, task/variation, and exact raw model response are **UNKNOWN**. Invalid raw completions were not written to `exam_audit_events` or the compiler cache before `_validate_completed_batch()` rejected them. The log message records the failure category but not the rejected ID.

The error means the returned FITB record cited a chunk ID outside the supplied batch's allowed-ID set. It does not mean the evidence was absent from the full 103-chunk document. The validation correctly enforced batch-local source grounding, but the prompt supplied enough global textbook context for the model to copy an ID from another batch.

P1 fix: include an explicit batch-local evidence registry in the compact prompt and constrain FITB evidence with schema-level enum IDs or a deterministic post-processor that discards only invalid FITB entries before requesting a targeted continuation. Record rejected evidence ID, batch index, candidate task/FITB entry, and raw-response hash. Do not permit cross-batch evidence merely to avoid retries.

## 3. PostgreSQL JSONB NUL crash

### Confirmed failure

At `2026-10-26 11:13:56Z`, after the initial generator fan-out completed, PostgreSQL rejected a JSONB parameter:

```text
psycopg.errors.UntranslatableCharacter: unsupported Unicode escape sequence
DETAIL: \u0000 cannot be converted to text.
CONTEXT: JSON data, line 1: ...frac{y}{2}\\le 20\\\\x+y\\le 45\\\\x\\ge 0,\u0000...
```

This is an actual NUL code point in the Python value serialized for JSONB, not harmless text containing the six printable characters `\\u0000`. PostgreSQL JSONB rejects NUL because PostgreSQL text values cannot contain it.

The failed write occurred while persisting the generation/audit/checkpoint payload, before the first generated exam became durable. The generated output passed Pydantic because the schema validates structure and types, not PostgreSQL's NUL prohibition.

### What cannot be recovered

The exact slot, generator response, field, and candidate object are **UNKNOWN**. The failed payload was only in worker memory and neither the exception record nor `llm_usage` stores a response body or candidate hash. The shown context is mathematical text but does not identify whether the NUL was in a statement, option, certificate, evidence metadata, or serialized checkpoint wrapper.

This exposure applies to every JSONB boundary: audit events, generation checkpoints, `exams`, `metadata`, warnings, and evaluation data. It is a persistence-validity defect, not a performance defect.

### Fix

P0: add one recursive JSON-safe persistence validator at the repository/audit boundary, before every `Jsonb(...)` call. It must return a path such as `generated_exams[0].questions.equation[2].solution_certificate.steps[1]`, classify the value as a persistence-invalid candidate, and prevent the database call.

For generated human text, normalize only NUL code points to a documented safe representation after retaining the raw-response hash and path. For mathematical payloads, reject and regenerate the affected slot if replacement could change formula meaning. Do not indiscriminately strip control characters. Add tests for NUL in every persisted field and assert no full-job retry.

Relevant boundaries: [tasks.py](C:/Users/abdul/PycharmProjects/GenExam/app/jobs/tasks.py:455), [repositories.py](C:/Users/abdul/PycharmProjects/GenExam/app/api/repositories.py:950), [repositories.py](C:/Users/abdul/PycharmProjects/GenExam/app/api/repositories.py:1061), and generation checkpoint emission in [exam_builder.py](C:/Users/abdul/PycharmProjects/GenExam/app/online/exam_builder.py:1883).

## 4. Why Celery regenerated the exam

`generate_exam` is declared with `max_retries=1`. Its broad `except Exception` clause treats the database `UntranslatableCharacter` as retriable, calls `repositories.retry_job(...)`, then calls `self.retry(..., countdown=15)`. See [tasks.py](C:/Users/abdul/PycharmProjects/GenExam/app/jobs/tasks.py:357) and [tasks.py](C:/Users/abdul/PycharmProjects/GenExam/app/jobs/tasks.py:600).

The first pass had generated all 23 questions in memory, but its checkpoint/audit persistence was the rejected JSONB operation. Thus `load_generation_checkpoint()` on the second claim had no valid generated-question state. It resumed the durable plan and regenerated the entire set. The retry produced at least 24 additional generator calls: total generator usage was 65 calls, including the initial fan-out, regenerated job fan-out, and later replacements.

**CONFIRMED:** the checkpoint mechanism can resume durable state, but cannot recover a state that was never saved. **CONFIRMED:** persistence failure is not isolated from generation because the same exception flows through the full-task Celery retry handler.

P0 fix: classify JSON serialization/database-data exceptions as non-transport failures. Persist only a small, validated checkpoint envelope first; if it fails, do not re-run generators. Mark the job failed with an actionable persistence code and retain a redacted in-memory diagnostic hash. For transient database connection failures, retry the exact pending checkpoint write under the same claim generation, then continue from the durable generation checkpoint. If no safe checkpoint exists, fail partial rather than repaying for all slots.

## 5. Generation-path inventory

The live system did launch 23 independent scheduler tasks. However, a one-slot call was passed through old type-bundle functions:

| Path | Live status | Classification |
|---|---|---|
| `generate_exams_node` → `generate_independent_slot` | 23 independent scheduled jobs | Active and required |
| `_generate_obj_bundle({qtype: [item]})` | MCQ/TF initial generation and duplicate replacement | Active, obsolete bundle semantics/schema |
| `_generate_type_from_plan(qtype, [item])` | Equation and Word Problem initial generation | Active, one-slot invocation but old plural schemas |
| `_generate_fitb_type(1, [item])` | FITB | Active, coordinator-owned bank required |
| `validator._regenerate_plan_slot` | duplicate replacement / regeneration | Active, uses old bundle path for MCQ/TF |
| `duplicate_pipeline` → `_regenerate_plan_slot` | duplicate recovery | Active, inherits obsolete path |
| checkpoint resume | active | Re-enters `generate_exams_node`; preserves only durable slots |
| legacy multi-slot direct callers | present in helpers | Compatibility-only until callers are removed |

The schema names `mcq_true_false_bundle`, `equation_questions`, and `word_problem_questions` were not merely cosmetic in this image: warnings show bundle retry/count behavior, and the TF generator repeatedly emitted the wrong required polarity for individual slots. The call carries one planned slot, but the response contract and retry machinery remain batch-oriented.

**Confirmed dead code:** none can be safely declared dead from this run. The cleanup sequence should first migrate all callers and add static call-graph tests; then remove unreachable batch paths and plural-only schemas. Do not delete code until the Docker image has passed the integration test.

## 6. Repair/revalidation loop

### The two persistent targets

| Slot | Corrupted question ID | Initial observed failure | Later recovery outcome |
|---|---|---|---|
| `m1_mcq_10` | `model1_mcq_1` | `task_drift`: certificate used elimination instead of the assigned graphing-calculator operation | Passed on replacement attempt 3 with task `task_2aebbdda8beca821` |
| `m1_true_false_3` | `model1_true_false_1` | `incomplete_solution`: true claim but missing required solution certificate/justification | Remained invalid through attempts 1–4, then reported `replacement_inventory_exhausted` |

`m1_mcq_10` history:

```text
attempt 1: repair_question → task_drift → REGRESSED
attempt 2: replace_task task_0cb1… → invalid_or_extraneous_solution → LATERAL
attempt 3: replace_task task_2aeb… → PASS → accepted
```

`m1_true_false_3` history:

```text
attempt 1: repair_question task_60b6… → incomplete_solution → UNCHANGED
attempt 2: replace_task task_7b16… → incomplete_solution → UNCHANGED
attempt 3: replace_task task_68d2… → incomplete_solution → UNCHANGED
attempt 4: replace_task task_26cd… → incomplete_solution → UNCHANGED
then: replacement_inventory_exhausted
```

The task was a true/false claim that required a checkable geometric certificate. Each candidate retained a bare truth value or otherwise omitted `solution_certificate`. General validation also found a question-type/operation mismatch: the task demanded justification, while the statement/answer only requested/returned True or False.

### Why it became an infinite paid loop

**CONFIRMED:** every independently generated MCQ was assigned `model1_mcq_1`, and every independently generated TF item was assigned a repeated per-type ID. The deployed `_assign_ids()` scans only the one-item `section` passed at each coordinator iteration, sets `max_seq=0`, and assigns sequence 1. See the live function at [exam_builder.py](C:/Users/abdul/PycharmProjects/GenExam/app/online/exam_builder.py:452). The function is called inside the per-job merge loop at [exam_builder.py](C:/Users/abdul/PycharmProjects/GenExam/app/online/exam_builder.py:2154).

Consequences shown in persisted validator coverage:

- batches sent eight copies of `model1_mcq_1`;
- later batches mixed repeated `model1_mcq_1`, `model1_true_false_1`, and `model1_equation_1`;
- returned IDs also repeated;
- `_build_by_id()` can represent only one final item per duplicated ID.

The repair manifest therefore associated validator results with the wrong or ambiguous record. The true/false slot did reach `replacement_inventory_exhausted`, but graph routing repeatedly returned to validation/repair because there was no reliable unique-ID and terminal-slot gate. The event counts prove the loop: 177 `repair_completed`, 177 General validation events, 177 STEM validation events, and 177 `repairing_exam` progress events. The job was eventually manually revoked and marked `WORKER_LOST` after heartbeat expiry.

P0 fixes:

1. assign IDs from stable slot IDs before concurrent generation, or assign after assembling the complete type section; never generate IDs from each isolated section;
2. hard-fail the generation merge on duplicate `question_id` or slot-to-ID non-bijection before validation;
3. key ledger/repair budgets by `slot_id` plus candidate fingerprint, not question ID alone;
4. when `replacement_inventory_exhausted` or global candidate cap is reached, mark the slot terminal and route to final partial assembly;
5. add a graph-level no-progress guard using `(slot_id, assignment_fingerprint, question_fingerprint, normalized failure set)` and maximum repair-round cap.

## 7. Repeated IDs: diagnosis

This was not normal repeated selective revalidation. Repeated stable IDs across separate rounds would be valid, but the same ID appeared multiple times in one validation batch for different slots and distinct source/task assignments. The persisted lifecycle state itself shows `m1_mcq_2`, `m1_mcq_3`, and additional MCQ slots all had `question_id: model1_mcq_1`.

Limited source-topic diversity contributed to semantic duplicate groups, but it did not create this technical identity corruption. The separate semantic issue was handled by duplicate replacement; the repeated technical IDs made that handling unreliable.

## 8. PostgreSQL versus RAM performance

The graph already holds active state in RAM. PostgreSQL is used for job fencing, stage progress, audit events, and monotonic durable checkpoints; it is not the principal latency source in the available telemetry.

Measured LLM time is 2,687,433 ms summed across overlapping calls. The live wall run lasted about 16 minutes before manual revocation, later terminalized by heartbeat recovery. The database has no per-query timing telemetry, so average checkpoint serialization/write time is **UNKNOWN**. It recorded 908 audit events and reached checkpoint revision 540, driven primarily by the repair loop rather than normal design.

| Architecture | Benefit | Risk | Recommendation |
|---|---|---|---|
| Current RAM graph + PostgreSQL checkpoints | Durable resume, fencing, auditability | Excessive full-state event payloads in loops | Retain |
| RAM active state + milestone/interval durable checkpoints | Fewer normal writes | Loses more work on crash; weakens resume if not carefully versioned | Adopt only after correctness fixes and telemetry |

Reducing writes would help only after the loop is removed. It would not fix the NUL crash, duplicated IDs, or cache miss. A justified hybrid design keeps fenced durable checkpoints at plan completion, generation merge completion, each accepted repair batch, duplicate-resolution milestones, and finalization; it avoids writing unchanged complete state. Persist compact deltas or content-addressed snapshots, not the entire graph state on every transition.

## 9. Docker deployment consistency

**CONFIRMED mismatch:** the worker image has no workspace source bind mount. It mounts only `data/` and the Hugging Face cache. Its image ID differs from the workspace code. SHA-256 comparison produced different hashes for each inspected critical file.

The container did have configured ceilings of 50 for provider/job/generation admission, and observed peak generator activity was 23. It used the intended Docker PostgreSQL service and the mounted `data/` cache path. The mismatch was application code, not database identity or concurrency configuration.

The current workspace contains the compatible cache migration and lifecycle work, but the live worker did not. Before another paid test, rebuild the application and worker images from the intended workspace revision, recreate the worker/API containers, then verify in-container source hashes and an endpoint/version fingerprint. This report does not perform that rebuild.

## Prioritized implementation plan

| Priority | Confirmed issue | Fix and regression coverage |
|---|---|---|
| P0 | Docker uses stale code | Build identity endpoint plus CI test comparing image source fingerprint to expected revision; integration test must execute inside built worker image. |
| P0 | Duplicate per-type question IDs | Slot-derived immutable IDs; merge invariant for unique IDs/slot bijection; 23-slot, out-of-order 50-way test. |
| P0 | NUL reaches JSONB | Recursive persistence preflight with path diagnostics; tests for audit/checkpoint/exam JSONB payloads containing NUL. |
| P0 | Whole-job retry after persistence error | Retry only safe checkpoint writes; classify data serialization as non-retriable; test that 23 generated slots are not regenerated. |
| P0 | Repair termination absent under collision/exhaustion | Slot/fingerprint no-progress state, global/per-assignment caps, terminal partial route; regression reproducing `replacement_inventory_exhausted`. |
| P1 | Compatible Catalog not used | Deploy and test `_compatible_compiler_seed` invocation; verify canonical task/variation/evidence validation and zero paid compile calls. |
| P1 | FITB cross-batch evidence | Batch-local evidence enum/registry, rejected-ID telemetry, targeted continuation test. |
| P1 | Old bundled contracts remain active | Introduce strict one-slot response schema for every qtype and replacement path; migrate `validator._regenerate_plan_slot`; retain FITB coordinator bank logic. |
| P2 | Audit/checkpoint volume | Add write/serialization timing and payload-size telemetry, then compact unchanged checkpoints and use milestone deltas. |

## Limitations

The exact first-pass NUL-containing generated field and exact rejected FITB evidence ID cannot be recovered because the live system did not persist rejected raw outputs or structured diagnostics. This is a telemetry gap, not an inference. All other major conclusions above are supported by PostgreSQL records, worker logs, or direct in-container/workspace source comparison.
