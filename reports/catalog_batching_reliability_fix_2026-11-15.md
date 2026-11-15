# Assessment Catalog Batching and Reliability Fix

Date: 2026-11-15  
Reference job: `3787defe-1e21-467a-91c8-5bf8382aab30`

## Result

The Catalog compiler now forms deterministic, parent-isolated child-chunk batches; presents an explicit evidence registry; reports exact invalid references on retry; retains independently valid tasks and sibling variations; and fails closed after the existing three-attempt limit if corrections remain invalid.

No Planner, difficulty, Generator, validator strictness, duplicate, Repair, Recovery, assembly, or question-count behavior was changed.

## A. Existing implementation audit

| Requirement | Already implemented | Missing or incorrect | Change made |
|---|---|---|---|
| Targeted retry feedback | Exact membership validation and three bounded attempts existed. | Failure feedback was a broad string, without batch/task/variation/field/invalid-ID/allowed-ID detail. The same broad request was retried. | Added structured diagnostics and included them in the next compiler prompt. |
| Partial preservation | Truncated JSON recovery could salvage complete objects in limited cases. | A schema-valid response with one invalid evidence reference discarded the whole batch. Valid sibling variations were lost. | Validate tasks and variations independently, retain valid fragments, and merge only corrected/replacement output before full revalidation. |
| Evidence registry presentation | Prompts listed valid child IDs and included chunk content. | Parent identity, child identity, and permitted IDs were not presented as a strongly separated registry. | Added a batch-labeled registry with parent context, exact child evidence ID, title, and content; parent IDs are explicitly non-citable. |
| Parent-child grouping | Retrieval retained parent metadata and validation used the batch's child-ID set. | Batching paired parent groups, which deliberately mixed two parents per compiler call. Missing parent metadata had no explicit isolation policy. | Group by actual metadata parent ID, preserve source order, isolate missing-parent children with a warning, and never infer parents from text. |
| Deterministic batch assignment | The prior iteration was deterministic for a fixed input. | It used fixed parent pairs rather than parent-isolated, budget-aware grouping and lacked complete partition invariants. | Added stable ordering, parent isolation, deterministic oversized-parent splitting, duplicate/omission/empty-batch checks, and exact allowed-ID equality checks. |

## B. Root cause of the failed job

Confirmed evidence from the persisted job/audit records:

- The job selected 103 child chunks from 18 parent sections.
- The old batcher produced nine batches by combining two parent groups per batch.
- Catalog batch 2 returned task or variation evidence IDs outside its local child-ID set on all three attempts.
- Strict validation correctly rejected the result with `task evidence IDs are outside this compiler batch`.
- The three failing compiler attempts took approximately 75.0, 80.4, and 78.1 seconds. The job failed after approximately 236.7 seconds.

The failure was a combination of:

1. Compiler output referencing IDs outside the assigned batch (confirmed).
2. Broad retry feedback that did not identify the exact bad fields and IDs (confirmed).
3. Whole-batch retry that discarded valid work and asked for the complete batch again (confirmed).
4. Two-parent batching that enlarged the local evidence namespace and mixed parent sections (confirmed design behavior; not evidence that assignment was random).

There was no evidence that the old batch assignment was random. The prompt already prohibited cross-batch evidence, so the immediate invalid output was a compiler contract violation, not validation weakness. The implementation now reduces ambiguity and makes correction targeted while retaining strict rejection.

## C. Implemented changes

### `app/online/assessment_catalog.py`

- `CatalogBatchValidationError` (line 54): carries structured diagnostics and preserved valid fragments while retaining `ValueError` compatibility.
- `_validate_completed_batch` (line 535): validates profile, task, transformation, variation, and FITB evidence membership; records batch ID, task ID, variation ID, field, bad ID, complete allowed set, and reason. Invalid base tasks are rejected; valid tasks and valid sibling variations are retained.
- `_catalog_evidence_registry` (line 2187): emits the explicit batch/parent/child registry supplied to the compiler.
- `_catalog_retry_feedback` (line 2216): tells the compiler exactly what must be corrected and forbids arbitrary ID substitution.
- `_merge_catalog_batch_fragments` (line 2261): deterministically merges preserved fragments with corrected output without duplicating task/variation identities.
- `_catalog_prompt` (line 2320): uses the registry and, on retries, supplies diagnostics plus preserved valid content.
- `_balanced_catalog_batches` (line 2975): groups by real parent metadata, preserves source order, splits oversized parents deterministically, and enforces partition invariants.
- `prepare_assessment_catalog.compile_batch` (line 3200): carries diagnostics/preserved content between attempts, normalizes and merges corrected output, and fully revalidates before cache/acceptance.
- Catalog batching fingerprint updated to `parent-isolated-budget-v2` so incompatible old batch caches are not reused.

### `app/config.py`

- `CATALOG_BATCH_MAX_CHILDREN` (line 83): configurable deterministic per-call child limit, default 24 and bounded to 1–128.

### Tests

- Added `tests/test_catalog_batch_reliability.py` covering deterministic grouping, oversized splits, partition integrity, registry presentation, exact diagnostics, profile isolation, partial task/variation preservation, failed-job behavior, and bounded unrecoverable failure.
- Updated prior batching tests to reflect parent-isolated rather than two-parent batches. No semantic downstream expectations were relaxed.

## Retry and preservation behavior

For every invalid reference, Python performs only exact ID-membership validation. It does not choose replacement evidence or interpret semantic support.

- Invalid task-level or transformation evidence: reject that task; preserve unrelated valid tasks.
- Invalid variation evidence: preserve the valid task and valid sibling variations; request only a corrected/replacement variation.
- Invalid FITB evidence: reject that entry; preserve valid entries.
- Invalid first-batch assessment-profile evidence: reject the profile portion and request correction; preserve otherwise valid tasks.
- Corrected content is merged with preserved content and passed through the complete schema and evidence validation again.
- If the compiler cannot correct the missing/rejected portions within three attempts, the batch fails explicitly. It is never accepted as complete merely because some content was preserved.

## D. Verification

All tests were offline and compiler/provider calls were mocked.

| Test run | Result |
|---|---:|
| Focused Catalog and compatibility suite (final) | 130 passed |
| Final targeted retry/legacy compatibility run | 22 passed |
| Full repository suite | 1408 passed, 21 skipped, 3 failed |

The three full-suite failures were reviewed:

1. `test_saved_103_chunk_catalog_mocked_graph_reaches_final_assembly`: environment fixture failure; the expected PostgreSQL document returned zero children instead of 103.
2. `test_catalog_batch_rejects_cross_batch_variation_evidence`: initially failed only because the new exception text no longer contained the legacy phrase. Compatibility wording was restored and the test passes in the final targeted run.
3. `test_job_scheduler_allows_fifty_recovery_requests`: unrelated scheduler/concurrency configuration failure outside Catalog batching.

Therefore the final known state is: all Catalog-focused tests pass; two unrelated full-suite failures remain environment/scheduler issues, and the one Catalog message-compatibility failure was fixed and reverified.

The regression test for job 3787 simulates an invalid task ID and invalid sibling variation ID from another batch, verifies exact retry feedback, verifies valid task/sibling preservation, supplies corrected fragments, and confirms successful full compilation without identity collision.

## E. Performance impact

| Area | Impact |
|---|---|
| Compiler calls | May increase because unrelated parents are no longer paired. For the failed job's 18 parents, parent isolation would normally produce 18 calls if every parent fits the 24-child limit, rather than the old nine two-parent calls. Oversized parents add deterministic split calls. |
| Parallelism | Unchanged; existing `MAX_PARALLEL_CATALOG_CALLS` behavior is preserved. |
| Batch size | Now bounded by parent identity and `CATALOG_BATCH_MAX_CHILDREN` (default 24). |
| Retry limit | Unchanged at three attempts. |
| Retry frequency | Expected to decrease from clearer evidence boundaries and targeted correction, but not measured; no performance claim is made. |
| Token consumption | Smaller parent-local initial prompts can reduce per-call size, while more calls and preserved-content retry context can increase total tokens. Net change was not measured. |
| Completeness | Still fail-closed. Preserved fragments do not make an incomplete batch complete; corrected/replacement portions must validate. |
| Cache behavior | Batching-version change invalidates incompatible old compiler-batch cache keys once; subsequent identical inputs remain deterministic and cacheable. |

## Remaining risks

- An LLM can still repeatedly ignore allowed IDs or select a locally valid but semantically unsupported child. Exact membership is deterministic; semantic grounding remains an LLM/validator responsibility.
- Missing or inconsistent ingestion parent metadata cannot be reconstructed safely by text similarity. Such children are explicitly isolated and warned about, which preserves evidence boundaries but may increase calls.
- Partial preservation reduces lost work but retry prompts may still be large when a batch produced many valid fragments.
- Runtime/token improvements require production telemetry from new jobs; offline mocks prove contract behavior, not provider latency or semantic correction quality.

## Post-deployment incident correction

After the initial deployment, jobs `4b628189-3f40-46a5-aa09-ce377d61d5cb` and `e1df88dd-866c-4e88-b1b8-a49cc9259422` failed before the first Catalog compiler call with:

`CatalogBatchAssignmentError: catalog batch assignment changed source order or omitted selected children`

The completeness invariant incorrectly compared the flattened parent-isolated batches with the globally interleaved retrieval order. Parent isolation legitimately changes that flattened global order whenever rows from different parents are interleaved.

The invariant now verifies the intended properties independently:

- the assigned child set and count exactly match the selected children;
- no child is duplicated;
- no batch is empty;
- every batch contains exactly one parent identity;
- source order is unchanged within each parent.

A regression test covers interleaved parent rows. The exact failed document was also checked read-only against its persisted 103 children: all 103 were assigned exactly once into 18 parent-isolated batches, with 103 unique assigned IDs. The rebuilt app and worker containers were healthy after deployment.

