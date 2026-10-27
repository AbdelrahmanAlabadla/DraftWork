# DraftWork Docker Pipeline Fix Verification Report

**Verification date:** 2026-10-26 (Asia/Dubai)
**Reference audit:** `POST_LIVE_EXAM_ROOT_CAUSE_AUDIT.md`
**Paid API calls made:** 0
**Real exam jobs started:** 0

## Executive summary

The API and Celery worker are running the rebuilt source and are healthy. PostgreSQL and Redis are healthy, Alembic is at `0006_job_execution_fencing (head)`, and the configured generation, per-job, and provider concurrency ceilings are all 50.

The confirmed production failures from job `dc0ae6f9-89be-4d1d-8d97-39f7136e3968` were fixed:

- Question IDs are now derived from Planner slot IDs and duplicate IDs are rejected before validation.
- Terminal repair failures cannot reenter Repair. Replacement exhaustion terminates immediately, and candidate-generation failures have their own bounded operational budget.
- Every PostgreSQL JSONB write in the exam/checkpoint/audit path recursively normalizes only U+0000 to U+FFFD before serialization.
- Deterministic data-format errors no longer trigger a Celery full-job retry. Final persistence transiently retries its own idempotent transaction.
- A same-version Catalog with the same processed document, evidence membership, and valid task contract can be migrated to the current cache key without compilation. Historical chunk ordering does not cause a miss.
- Compiler batches reject cross-batch task, variation, and FITB evidence IDs, and the prompt explicitly prohibits cross-batch evidence.
- Initial generation, shortfall recovery, repair regeneration, and duplicate replacement use one planned slot per generation request. Legacy bundle names remain only where they are active compatibility wrappers around one-slot requests.

Local tests passed (`864 passed, 12 skipped`). Docker tests passed against the rebuilt worker (`23 passed`) and the real 103 persisted child rows. PostgreSQL integration tests passed in an isolated temporary database, including a real NUL-containing JSONB payload. The temporary database was removed afterward; the authoritative `draftwork` database was not modified.

The deployment is ready for one controlled paid exam test. This verification proves pipeline control, persistence, cache reuse, and mocked convergence; it does not prove live-provider mathematical quality or provider rate-limit behavior.

## 1. Docker deployment verification

### Container health

| Component | Result |
|---|---|
| `genexam-app` | Healthy; `/health/ready` returned PostgreSQL, Redis, and storage `ok` |
| `genexam-worker` | Healthy; Celery beat and worker processes running |
| `genexam-postgres` | Healthy; PostgreSQL 17.11 |
| `genexam-redis` | Healthy |
| `genexam-migrate-1` | Exited 0 |
| Alembic | `0006_job_execution_fencing (head)` |

### Repository/container source identity

The repository, API container, and worker container produced identical SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `app/online/exam_builder.py` | `59f89565417ded72f17ab40a14fd55af50de38bfe97e3d7bbc9b70a746812889` |
| `app/online/graph.py` | `91c1e438a04fbd0b0017b4f7fb8457302005f3c859194d92e348434bc0c83c20` |
| `app/online/validator.py` | `401e7ecc90ff17f7c256ab6e622e933714fccc1fd35990b1ee506d49c3c4f42c` |
| `app/online/assessment_catalog.py` | `37359960cd9b230d7f7bdb3b2a0f3f6df9405be37b70f1997d36affc83fdee9f` |
| `app/api/repositories.py` | `c3ec879ae75ac5ce5a2be83e175ed1909826cd79e508db6aedebd44a0ad8ba44` |
| `app/api/evaluation_store.py` | `d31cb75d61db0fa1ff5a21f42d2598bbd8b79b34098c082d77fd4b66ee97573b` |
| `app/offline/chunk_store.py` | `f3a9a3a1d83d5d43ef86b66042cfa2a46da86a439a34a835c168ba5cede52522` |
| `app/jobs/tasks.py` | `5308ffd5d87c6fe8a93d647532f7969033940a43727bd46ffdebf181e3c0b661` |

This confirms Docker is executing the fixed production source rather than the stale image found in the prior audit.

### Qdrant orphan

`genexam-qdrant` remains healthy and was not deleted. It is not declared in the current `compose.yaml`. No runtime Python or Compose reference to Qdrant was found; only deployment-example variables remain in `.env.production.example`. It appears to be a legacy service from an older vector-storage configuration. It is not needed by the current PostgreSQL chunk retrieval and local embedding path, but its container/volume should be retained until the owner confirms no external workflow uses port 6333.

### Concurrency

Both API and worker report:

```text
MAX_PARALLEL_GENERATION_CALLS=50
LLM_JOB_CONCURRENCY_LIMIT=50
DEEPSEEK_CONCURRENCY_LIMIT=50
```

The work-conserving dispatcher test reached exactly 50 simultaneous workers with 100 queued jobs and completed all jobs in input order.

## 2. Bugs fixed

### A. Stable unique question IDs (P0)

`_assign_ids()` at `app/online/exam_builder.py:452` now requires a canonical Planner slot and derives the public ID from its model, type, and sequence. It overwrites any conflicting generated ID:

```text
m1_mcq_1  -> model1_mcq_1
m1_mcq_2  -> model1_mcq_2
m1_mcq_10 -> model1_mcq_10
```

`_assert_unique_question_ids()` at `app/online/exam_builder.py:472` rejects missing or repeated IDs before answer balancing, validation, repair, or duplicate review. This removes the dictionary overwrite that previously collapsed all MCQs onto `model1_mcq_1`.

### B. Repair termination and no-progress bounds (P0)

Routing at `app/online/graph.py:450` treats `resolved`, `exhausted`, and `terminal_failure` as terminal. Repair staging at `app/online/validator.py:3434` does the same.

When replacement inventory is empty, the target becomes `status=exhausted` with `replacement_inventory_exhausted` (`app/online/validator.py:3537`). It cannot return to Repair merely because it did not spend a validated-candidate attempt.

Provider/schema/no-op failures remain separate from correctness attempts, but they are no longer unbounded. `MAX_CANDIDATE_GENERATION_FAILURES=3` at `app/online/validator.py:50` tracks operational candidate failures per stable slot. The third failure records `candidate_generation_budget_exhausted` (`app/online/validator.py:3642`) and terminates that slot. A genuine candidate that reaches General/STEM evaluation still consumes the existing validated-candidate and assignment budgets.

This directly prevents the previous 177-cycle pattern. An unrecoverable slot proceeds to duplicate review/final assembly and produces a partial exam warning.

### C. PostgreSQL U+0000 protection (P0)

`sanitize_jsonb()` and `_jsonb()` at `app/api/repositories.py:48` recursively process strings, dictionaries, lists, and tuples. Only U+0000 is replaced, with U+FFFD. All other Unicode, backslashes, newlines, Arabic text, and mathematical notation remain unchanged. The replacement marker makes the invalid provider character visible rather than silently joining or deleting neighboring formula characters.

All repository JSONB writes now use this boundary, covering requests, checkpoints, audit events, job results, saved exams, warnings, metadata, and evaluation payloads. Evaluation storage applies the same boundary at `app/api/evaluation_store.py:126`; offline parent/child persistence applies it at `app/offline/chunk_store.py:87` and `:112`.

A real PostgreSQL test persisted `{"formula": "x\u0000+y"}` in an isolated database and read back `{"formula": "x\ufffd+y"}` without an exception.

### D. Celery retry classification and persistence isolation (P0)

`_is_retryable_generation_error()` at `app/jobs/tasks.py:174` permits task-level retries only for provider/network/timeouts and transient PostgreSQL connection/interface errors. `UntranslatableCharacter` and other deterministic data-format failures are terminal and cannot trigger complete paid regeneration.

`_complete_generation_with_transient_retry()` at `app/jobs/tasks.py:186` retries the final fenced, idempotent PostgreSQL transaction up to three times with bounded backoff. It does not re-enter retrieval, planning, generation, or validation. The call site is `app/jobs/tasks.py:567`.

Generation checkpoints remain coordinator-owned and retain successfully merged slot results. If checkpoint persistence itself loses its database connection after bounded retry behavior, the task fails safely rather than pretending the checkpoint is durable.

### E. Compatible Catalog reuse (P0)

The document `175ef217-a754-4e58-8bae-2bbb4841f824` still has 103 authoritative child rows and processed key:

```text
772d2717699ca1dc2b015371744fb5d9df8f50751dc7a80b8eb63d716697f84e
```

The saved `catalog-v12-grouped-type-variations` artifacts use the same 103-ID evidence set. `_compatible_compiler_seed()` at `app/online/assessment_catalog.py:1349` now compares canonical ID sets, not historical list order, while retaining these safety checks:

- same Catalog version;
- same processed-document key;
- identical selected evidence membership;
- valid assessment profile;
- nonempty tasks and valid FITB bank;
- every task satisfies the current finalized task contract.

The actual compatibility loader returned the current reusable Catalog with 97 tasks and 74 FITB entries. The Docker mocked graph also copied the earlier `339d...` artifact into an isolated cache and confirmed `assessment_catalog_cache_hit=True` and `assessment_catalog_compatibility_cache_hit=True`; any provider call was configured to fail the test. No compiler call occurred.

### F. FITB and variation evidence boundaries (P1)

`_validate_completed_batch()` now checks variation-level evidence in addition to top-level task and FITB evidence (`app/online/assessment_catalog.py:478`). The compiler prompt at `app/online/assessment_catalog.py:1561` explicitly requires every task, variation, and FITB ID to come from the current batch, even when another chunk belongs to the same document.

This preserves strict source grounding. An invalid response is rejected; IDs are not silently remapped to unrelated evidence.

### G. Single-slot generation paths (P1)

The active initial dispatcher already creates one immutable job per Planner slot and supports 50 concurrent calls. MCQ/True-False still pass through `_generate_obj_bundle`, but each invocation receives one type and one slot and its dynamic schema permits no more than that slot. Equation, Word Problem, Definition, Why, Essay, and FITB likewise receive one slot per invocation.

Shortfall recovery previously remained genuinely bundled. It now loops over missing slots and invokes the relevant strict generator once per slot (`app/online/exam_builder.py:1637`). Regeneration and duplicate replacement already invoke `_regenerate_plan_slot()` for one stable question/slot.

The following functions were retained because they still have production callers or backward-compatibility tests:

| Function/schema | Classification |
|---|---|
| `_generate_obj_bundle` / `mcq_true_false_bundle` | Active compatibility wrapper; one slot per current call |
| `_generate_type_from_plan` / `*_questions` | Active; one slot per initial/recovery/regeneration call |
| `_generate_fitb_type` | Active; one item per call, with coordinator-owned shared-bank rebuild |
| `_repair_invalid_items` | Active bounded structural correction for one rejected slot |
| `_regenerate_plan_slot` | Active for repair and duplicate replacement; one slot |

No active function was deleted merely because its historical name contains “bundle.” Removing or renaming these wrappers is a cleanup task, not a correctness requirement.

## 3. Regression and integration results

### Complete local suite

```text
864 passed, 12 skipped in 107.02s
```

The skips are environment-gated integration cases; no failure was hidden.

### Rebuilt Docker worker

Tests were mounted read-only into an ephemeral container created from the final worker image. Production code came from the image, while PostgreSQL and the existing `data` volume used the normal Compose configuration.

```text
23 passed in 8.88s
```

This covered:

- authoritative 103 child-row retrieval from Docker PostgreSQL;
- compatible Catalog reuse with compiler calls forbidden;
- production LangGraph routing through planning, difficulty adjustment, generation, validation, duplicate review, and final assembly;
- 23 requested and 23 assembled fixture questions;
- unique stable IDs;
- bounded repair routing;
- malformed/duplicate identity rejection;
- 100 scheduled generation jobs with a measured peak of 50 workers;
- deterministic ordered merge despite concurrent completion.

### Isolated PostgreSQL

An isolated database named `draftwork_pipeline_test_20261026` was migrated to head. Repository ownership/concurrency tests produced `3 passed, 7 skipped`; the dedicated real JSONB NUL test produced `1 passed`. The isolated database was dropped after verification.

### Scenarios covered by focused tests

- Duplicate and missing response IDs are rejected.
- One failed generation slot does not discard successful siblings.
- Provider failures use bounded retry paths.
- Replacement inventory exhaustion terminates cleanly.
- Terminal states do not route back into Repair.
- Catalog chunk order does not force compilation.
- Cross-batch variation evidence is rejected.
- General/STEM selective-validation and repair-manifest behavior remains intact.
- FITB shared banks are rebuilt by the coordinator.
- Checkpoint fencing and resume regression tests pass.

## 4. Mocked 23-question outcome

| Setting | Result |
|---|---:|
| MCQ | 10/10 |
| True/False | 5/5 |
| Equation | 4/4 |
| Word Problem | 4/4 |
| Total | 23/23 |
| Language | Arabic request contract retained |
| Difficulty | Easy |
| Catalog compiler calls | 0 |
| Duplicate groups | 0 in valid fixture |
| Final graph result | Complete |
| Generation concurrency test | Peak 50 with 100 slots |

Malformed questions, duplicate IDs, unsupported/cross-batch evidence, provider errors, and exhausted replacement inventory were exercised through focused deterministic fixtures. No paid endpoint was reachable from these tests.

## 5. Remaining risks

1. **Live provider quality is unverified.** Mocked PASS verdicts prove graph behavior, not Arabic quality or mathematical correctness.
2. **Provider admission may reduce effective concurrency below 50.** The configured and scheduler ceilings are 50, but live RPM/TPM, connection pools, or `Retry-After` responses remain authoritative.
3. **U+0000 normalization preserves availability, not semantic certification.** U+FFFD is deliberately visible. Validators should still reject a formula whose meaning becomes unclear.
4. **Catalog compatibility selects the newest contract-valid artifact.** Several same-version 103-chunk Catalogs exist with different task counts. They are structurally compatible, but the next paid test should record the exact selected cache file/key for reproducibility.
5. **Legacy schema names remain.** They no longer imply multi-slot calls in active paths, but renaming/removing them requires a separate compatibility migration.
6. **Qdrant ownership is unresolved.** Current DraftWork does not reference it, but it was retained as required.

## 6. Readiness decision

**Ready for one controlled paid 23-question test: YES.**

The next run should be monitored for:

- exact Catalog cache key/path and zero compiler calls;
- 23 unique slot-derived question IDs before validation;
- actual HTTP in-flight generation peak and any 429/Retry-After events;
- per-slot candidate/validated attempt counters;
- zero repeated terminal repair states;
- checkpoint revisions around generation completion;
- U+FFFD occurrences in persisted content;
- final 23/23 mathematical correctness and source grounding.

No commit or push was performed. No existing document, Catalog, exam, cache, container, or Docker volume was deleted.
