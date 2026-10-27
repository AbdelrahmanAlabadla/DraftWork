# Docker v13 Final Verification

## Final decision

**PASS — ready for a controlled paid test.**

The rebuilt API and worker are healthy, contain the exact repository implementation, expose Catalog v13 and generation concurrency 50, and pass the Docker-based mocked production-graph and targeted failure regressions. No paid provider call, real exam job, Docker rebuild, source edit, or write to the `draftwork` database occurred during this verification.

## 1. Runtime health and readiness

Verified services:

| Service | Result |
|---|---|
| `genexam-app` | Healthy; HTTP `/health` returned 200 |
| `genexam-worker` | Healthy |
| `genexam-postgres` | Healthy; `pg_isready` reports accepting connections to `draftwork` |
| `genexam-redis` | Healthy; `redis-cli PING` returned `PONG` |
| `genexam-qdrant` | Healthy; no mutation performed |

The API had completed initialization by the time final verification began.

## 2. Source hash verification

SHA-256 hashes matched across the repository, API container, and worker container for every production file modified by the implementation.

| File | SHA-256 | Repository/API/Worker |
|---|---|---|
| `app/config.py` | `94384b0648ffa01ffc09440f419447e8265543cf3b92b7936433c96067339077` | MATCH |
| `app/online/lifecycle.py` | `9ad24d1d2eb86a107d1283116a83262238409090230e3b0c436fea386f648b29` | MATCH |
| `app/online/assessment_catalog.py` | `6ad9184d47cd039cc2b3f5c62aff2a9a2421b3c7db769d5b82488aba4e8f0e8b` | MATCH |
| `app/online/duplicate_pipeline.py` | `49f43b7c9c86b384a5d416d0cecb488edc5ac2aaf93b8d79a33c66c163a0446e` | MATCH |
| `app/online/validator.py` | `de761ddd076995e0370a8f93be548ae33a4e1baf944f2aecbdc7635664b5263d` | MATCH |
| `app/online/exam_builder.py` | `7200b271d713ebb28f509c00a99e0559c317acf251590328607cf6add16b7746` | MATCH |
| `app/online/graph.py` | `07061cedf2272e88927fdd4da95e4a7affcd37b91aa64cce673f6cfdf7c604fd` | MATCH |

This confirms both running services use the rebuilt source rather than the previous image.

## 3. Catalog v13 and cache migration

The worker reports:

```text
ASSESSMENT_CATALOG_VERSION = catalog-v13-numerical-givens-contract
CATALOG_CONTRACT_VERSION   = catalog-v13-numerical-givens-contract
```

Docker tests confirmed:

- Contradictory required/variable numerical metadata is detected.
- The dinner-tip task safely moves 60 AED and 17% to `source_example_givens` while preserving evidence and variable parameter names.
- `exact_source_instance` becomes false only for the mechanically proven contradiction.
- Legitimate exact numerical targets remain unchanged.
- Exact-source tasks cannot authorize altered numerical instances.
- Same-version cache compatibility remains insensitive to historical chunk ordering.
- The saved v12 103-child-chunk Catalog is inspected and migrated deterministically without a compiler provider call.

No Catalog artifact or processed document was rewritten.

## 4. Saved 103-child-chunk production graph

Executed inside `genexam-worker` using `/app` production imports and a temporary copy of the test fixtures:

```text
test_saved_103_chunk_catalog_mocked_graph_reaches_final_assembly: PASS
```

The test verified:

- Exactly 103 authoritative PostgreSQL child chunks.
- Selected chunk IDs match the saved Catalog.
- Catalog reuse does not call the compiler provider.
- Production graph, Planner and Difficulty Adjustment execute with mocked model responses.
- The requested configuration reaches 23 assigned/generated slots.
- Final assembly is reached.

The test only read PostgreSQL rows. It did not create a job or modify `draftwork`.

## 5. Dinner-tip and coordinated-replacement verification

### Identical 60 AED/17% instance

The Docker regression verifies that identical assessment-instance fingerprints match, while a source-authorized distinct instance has a different fingerprint.

A direct in-worker concurrent General Repair simulation proposed the same 60 AED/17% question for two slots. Result:

```text
concurrent_general_repair_identical_instance=REJECTED
```

Only one staged candidate remained. The sibling was rejected with:

```text
assessment_instance_reservation_conflict
```

### Legitimate variations

The tests confirm that changed, authorized effective givens produce a distinct realized-instance fingerprint. No per-task reuse cap is imposed. Catalog authorization and subsequent correctness/duplicate qualification remain required before commit.

### Duplicate Replacement

The complete `test_duplicate_pipeline_controlled.py` suite ran inside the worker. It verifies:

- Reused tasks are forced into duplicate comparison.
- Replacement generation is transactional.
- Failed precommit qualification restores/preserves the active valid question.
- Numeric variation requires explicit task authorization and is forbidden for exact-source tasks.
- Third-attempt variation requires another duplicate check.
- Exhausted attempts terminate without extra Judge loops.
- Final verification preserves exact unresolved conflicts and clears stale ones only when current evidence supports that result.

## 6. Authoritative state and checkpoint/resume

Docker tests and a JSON checkpoint round-trip confirmed preservation of:

- Immutable `original_assignment`.
- Current assignment and `assignment_revision`.
- Assignment fingerprint.
- Current `question_revision` and question fingerprint.
- General validation reference bound to current fingerprints.
- Append-only assignment/question history.
- Existing lifecycle terminal/failure behavior.
- Legacy lifecycle/checkpoint compatibility.
- Resume routing with durable plans and partial questions.
- Legacy same-job resume without unnecessary replanning.

The explicit round-trip result was:

```text
checkpoint_roundtrip=PASS
```

PostgreSQL monotonic checkpoint fencing was not mutated or integration-written during this verification; its repository tests had already passed in the implementation suite.

## 7. Partial duplicate-group resolution

The production failure shape was reproduced with a three-question group:

```text
q1 — q2
q1 — q8
q2 — q8
```

After q1 changed:

- q1/q2 was invalidated.
- q1/q8 was invalidated.
- The unchanged q2/q8 duplicate edge was retained.
- The resulting current group contained only q2 and q8.
- Routing required a verification scan when replacement changed content.
- Routing could finalize from existing fingerprint-bound evidence when no verification remained necessary.

This prevents the stale warning that previously continued naming MCQ 1 after its content changed, while preserving the actual MCQ 2/MCQ 8 conflict.

## 8. Generation concurrency

The worker reports:

```text
MAX_PARALLEL_GENERATION_CALLS = 50
```

The rebuilt `exam_builder.py` hash matches the repository implementation containing the work-conserving independent-slot dispatcher. No provider call was made, so actual live HTTP concurrency was intentionally not measured here.

## 9. Docker test results

Primary in-container run:

```text
47 passed
```

This included the saved 103-chunk graph, the new shared-state/Catalog regressions, the complete controlled duplicate suite, and performance/parallel-repair tests.

State, lifecycle, Catalog and audit run:

```text
70 passed, 5 skipped, 1 fixture-packaging failure
```

The failure was not a product defect. `test_lifecycle_migration_keeps_failed_exam_and_audit_rows` reads a migration source file, and the temporary test bundle initially lacked `migrations/`. After copying that directory into the temporary container-only verification folder, the exact failed test produced:

```text
1 passed
```

Additional resume/state selection:

```text
5 passed
checkpoint_roundtrip=PASS
```

### Skipped checks

Five existing optional/environment-dependent tests were skipped by pytest in the lifecycle/Catalog selection. No skipped check was converted into a pass claim.

### Initial invocation issue

The Docker image intentionally does not package repository tests. The first pytest command reported “file or directory not found” and ran zero tests. Tests and the saved Catalog were then copied to `/tmp/draftwork-v13-verification`; all imports continued to resolve against `/app`, ensuring the rebuilt image code was exercised. No image or repository file was modified.

## 10. Remaining limitations

- Mocked LLM responses validate control flow, identity, state, and commit behavior; they do not prove live-provider mathematical quality.
- Live RPM/TPM behavior, real HTTP peak concurrency, latency, and 429 handling were not exercised because paid calls were prohibited.
- Exact assessment-instance fingerprints reject identical realized content. Semantically near-duplicate questions still require embedding screening and Duplicate Judge.
- The deterministic Catalog repair handles explicit lexical required/variable overlap. Subtle semantic contradictions remain fail-closed and may require controlled recompilation.
- This verification did not write a real PostgreSQL checkpoint. It tested checkpoint JSON survival and graph resume behavior without modifying `draftwork`.

## 11. Readiness decision

**PASS.** The rebuilt Docker deployment contains the verified v13 implementation and is ready for one controlled paid exam test, subject to the existing provider budgets, retry limits, and monitoring requirements. The next paid run should still be treated as live quality/performance validation rather than proof already supplied by these mocked tests.
