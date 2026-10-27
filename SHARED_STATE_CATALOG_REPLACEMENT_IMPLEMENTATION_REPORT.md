# Shared State, Catalog Contract, and Coordinated Replacement Implementation Report

## 1. Outcome

The implementation extends DraftWork's existing LangGraph state, slot lifecycle, Catalog finalizer/cache, repair staging, and duplicate pipeline. It does not add another memory system or merge agents.

Implemented outcomes:

- Every slot now retains its immutable original assignment, revisioned current assignment, revisioned current question fingerprint, compact validator references, and append-only transition entries.
- Asynchronous generation and repair candidates carry assignment/content preconditions; stale results are rejected before publication.
- Concurrent General/STEM repair proposals are merged through one coordinator check that rejects an assessment instance already active or reserved by a sibling.
- Catalog v13 enforces cross-field numerical-givens consistency, safely normalizes mechanically provable required/variable overlaps, preserves source examples, and quarantines unresolved contradictions.
- Compatible v12 Catalogs can be deterministically revalidated and migrated without a provider call; incompatible tasks fail closed.
- Duplicate reports now bind pairs, verdicts, and groups to question fingerprints and rule versions.
- Partial duplicate replacement invalidates changed endpoints while preserving unchanged conflict edges.
- Final routing performs required post-replacement duplicate verification, and final assembly derives warnings from current fingerprints.
- Generation concurrency remains configured at 50.

No paid LLM call or real exam job was run.

## 2. Files and functions changed

Only the following implementation areas were intentionally changed for this task. These files already contained unrelated working-tree changes, which were preserved.

| File | Relevant changes |
|---|---|
| `app/online/lifecycle.py` | Added stable assignment/question/assessment-instance fingerprints and revisioned lifecycle reconciliation |
| `app/online/assessment_catalog.py` | Added numerical-contract diagnostics, deterministic repair/quarantine, v12 migration checks, and v13 metadata |
| `app/config.py` | Default Catalog contract/version advanced to `catalog-v13-numerical-givens-contract` |
| `app/online/exam_builder.py` | Generation jobs/results carry assignment revisions; stale results are rejected; final duplicate groups are fingerprint-filtered |
| `app/online/validator.py` | Repair candidates carry revision preconditions; stale candidates are rejected; concurrent proposals receive global realized-instance reservation checks |
| `app/online/duplicate_pipeline.py` | Added pair/group endpoint fingerprints, versions, and current-conflict derivation |
| `app/online/graph.py` | Finalization-pending state routes through duplicate verification when verification is required |
| `tests/test_shared_state_catalog_coordination.py` | Added focused state, Catalog, reuse, and duplicate lifecycle regressions |

Key source locations after implementation:

- Lifecycle fingerprints: `app/online/lifecycle.py:25-67`
- Lifecycle revision reconciliation: `app/online/lifecycle.py:109-183`
- Catalog diagnostics/normalization: `app/online/assessment_catalog.py:937-1025`
- Catalog finalization enforcement: `_finalize_catalog()` beginning at `app/online/assessment_catalog.py:1651`
- Compatible cache migration: `_compatible_compiler_seed()` in `app/online/assessment_catalog.py`
- Duplicate fingerprint ledger helpers: `app/online/duplicate_pipeline.py:27-76`
- Stale repair rejection: `app/online/validator.py:2991-3018`
- Coordinated repair reservations: `app/online/validator.py:3828-3882`
- Stale generation rejection: `app/online/exam_builder.py:2136-2148`
- Final duplicate routing: `app/online/graph.py:446-466`

## 3. Authoritative shared-state design

The existing `slot_lifecycle` remains the canonical compact per-slot view. `plans`, `generated_exams`, validation ledgers, repair memory, duplicate reports, and audit events retain their existing responsibilities.

Each lifecycle slot now contains:

- `original_assignment`: written once from the first Planner-visible assignment.
- `current_assignment`: compact current task/variation/operation/target/work-product/givens/evidence/difficulty data.
- `assignment_revision` and `assignment_fingerprint`.
- `question_revision` and `question_fingerprint`.
- `validation`: compact role-keyed verdict references bound to assignment/question fingerprints.
- `history`: append-only compact assignment/question transitions.
- Existing attempts, failure state, duplicate status, and terminal status.

Large prompts, source text, and repeated full plans are not copied into each record. Assignments reference evidence by IDs.

The graph wrapper already reconciles lifecycle after each node and persists node output under monotonic PostgreSQL checkpoint revisions. The new fields therefore participate in existing checkpoint replay without a new persistence subsystem.

## 4. Data retained from each stage

| Stage | Durable/current information |
|---|---|
| Retrieval | Existing selected chunk IDs, processed key, retrieved chunks and evidence scope |
| Catalog | Existing tasks/evidence plus Catalog contract version, source-example givens, and structured contract diagnostics |
| Planner | First materialized assignment becomes immutable `original_assignment`; score/rationale remain in the plan/audit event |
| Difficulty Adjustment | Changed plan is reflected as a new assignment revision and compact history event |
| Plan Repair | Same revision mechanism; existing failed candidates and repair history remain intact |
| Generator | Result carries assignment revision/fingerprint; current question receives revision/fingerprint |
| General/STEM Validation | Lifecycle stores verdict/failure reference bound to current assignment/question fingerprint; complete verdict stays in reports/ledger |
| General/STEM Repair | Candidate records expected revisions, strategy, assignment identity and realized-instance fingerprint; existing history records disposition |
| Duplicate Screening/Judge | Pair payloads and verdicts now carry endpoint fingerprints and screening/Judge versions |
| Duplicate Replacement | Existing sequential reservation, candidate history and precommit validation retained; unresolved groups become fingerprint-bound |
| Final Assembly | Unresolved groups are filtered/rebuilt against current question fingerprints before warnings are emitted |

## 5. Revision and commit rules

Assignments and questions use canonical JSON SHA-256 fingerprints. Assignment fingerprints exclude large source payloads and include identity, variation, operation, target, work product, givens, evidence IDs and difficulty. Question fingerprints exclude lifecycle metadata so revision stamps do not recursively alter identity.

Rules implemented:

1. Original assignment is immutable after first observation.
2. Assignment revision increments only when the compact assignment fingerprint changes.
3. Question revision increments only when student/answer content fingerprint changes.
4. Generation jobs carry the assignment revision/fingerprint used to launch them.
5. Generation merge rejects a result when the authoritative assignment fingerprint changed.
6. Repair candidates carry expected assignment and question revisions/fingerprints.
7. Candidate validation rejects stale preconditions before making validator calls or committing content.
8. Legacy checkpoint/test candidates without revision metadata remain accepted for backward compatibility.
9. PostgreSQL claim generation, request fingerprints, and monotonic checkpoint revisions remain the outer ownership fence.

## 6. Catalog numerical-value contract enforcement

The new `catalog_numerical_contract_diagnostics()` detects:

- A numeric required given that names a declared variable parameter.
- An exact-source task that simultaneously authorizes numerical variation.

`_repair_catalog_numerical_contract()` performs only mechanically safe corrections:

- Overlapping concrete values move from `required_givens` to `source_example_givens`.
- Variable parameter names and evidence IDs remain unchanged.
- If all concrete numeric required values were overlaps, `exact_source_instance` becomes false.
- Exact-source variations have `numeric_values_may_change` disabled while preserving the exact task.
- Every correction receives a structured diagnostic code and disposition.
- Any contradiction remaining after safe normalization is quarantined at task level; it does not discard an otherwise usable batch.

For `task_b76c4d06b6beb27a`, 60 AED and 17% are preserved as source-example givens, while bill amount and tip rate remain variable parameters. No alternate values are invented.

Legitimate exact values remain unchanged when no required/variable contradiction exists. The regression suite includes an exact area-20 construction case.

## 7. Cache compatibility and versioning

The default public and internal contract is now `catalog-v13-numerical-givens-contract`. The cache key already incorporates version, finalizer source, prompt source, and schema hashes.

Behavior:

- Native v13 artifacts must satisfy the strengthened task contract.
- Same-version compatible cache discovery remains order-independent for selected chunk IDs.
- v12 finalized artifacts with the identical document key/evidence set are inspected task by task.
- Safely repairable tasks are normalized in memory and re-finalized under v13 without an LLM call.
- A task with a remaining contradiction causes that compatibility candidate to fail closed.
- Processed chunks and retrieval caches are untouched.

The saved 103-child-chunk mathematics Catalog passed deterministic migration/reuse in the mocked production graph with zero compiler provider calls.

## 8. Global candidate coordination

Existing responsibilities remain separate:

- Difficulty Adjustment continues sequential mutation of the shared plan and existing ranking.
- Duplicate Replacement continues sequential reservation across a conflict round and existing candidate ranking/precommit checks.
- General/STEM Repair still generates independent candidates concurrently.

The new repair coordinator derives realized assessment-instance fingerprints from task, variation, operation, target, work product, effective givens, question/options and answer. After concurrent proposal generation, it checks each proposal against active sibling instances and earlier same-round reservations. An exact collision is rejected with `assessment_instance_reservation_conflict`; valid distinct instances proceed concurrently to validators.

This introduces no per-task reuse cap. The same canonical task remains reusable when effective givens and realized assessment work are distinct and authorized.

## 9. Duplicate findings and finalization

Each scan now records:

- Left/right question fingerprints on candidate pairs.
- Screening version.
- Endpoint fingerprints and Judge version on verdicts.
- Resolution status.
- A question-fingerprint map on each conflict group.

`_current_conflict_groups()` removes stale endpoints and edges and preserves edges whose two endpoint fingerprints still match. In the production failure shape:

- Changing MCQ 1 invalidates MCQ 1/MCQ 2 and MCQ 1/MCQ 8.
- The unchanged MCQ 2/MCQ 8 edge remains unresolved.
- Final warnings contain MCQ 2 and MCQ 8 only.

When replacement marks finalization pending but a replacement changed content, routing now performs the required verification scan. If no endpoint changed and budgets are exhausted, finalization can reuse the current fingerprint-bound verdict without an unnecessary Judge call.

## 10. Tests and offline verification

### Focused verification

After implementation:

```text
115 passed
```

This covered the new tests plus existing Catalog, lifecycle, duplicate, repair, and performance tests.

Additional production-graph/generation integration coverage:

```text
56 passed
```

### Full repository suite

Final run after all changes:

```text
871 passed, 12 skipped in 50.28s
```

The skipped tests are existing environment/optional tests; no failures were hidden.

### Existing 103-chunk offline production-graph test

`test_saved_103_chunk_catalog_mocked_graph_reaches_final_assembly` passed. It verifies:

- Exactly 103 authoritative child chunks.
- Compatible Catalog reuse with no provider call.
- Production Planner/Difficulty/graph routing.
- 23 requested slots reaching mocked final assembly.

Existing tests also retain the configured 50-call generation ceiling and work-conserving dispatcher behavior. No paid provider was contacted.

## 11. Docker verification

Docker services were healthy when checked: API, worker, PostgreSQL, Redis and Qdrant all reported healthy.

The running worker was built before this implementation. SHA-256 hashes for all seven changed production files differ between the repository and `/app` in `genexam-worker`. Therefore:

- Local code and tests are verified.
- The running Docker deployment **does not yet contain this implementation**.
- No rebuild/restart was performed as part of this task.
- Docker must be rebuilt/recreated before deployment-level or paid verification.

## 12. Remaining risks and limitations

- The deterministic Catalog normalizer only repairs explicit lexical required/variable overlaps. More subtle semantic contradictions are quarantined or require compiler regeneration; it does not guess.
- Assessment-instance fingerprints detect exact realized collisions. Near-duplicate pedagogy remains the responsibility of BGE screening and Duplicate Judge.
- Legacy checkpoints hydrate lifecycle revisions when reconciled; they cannot recreate historical intermediate assignments that were never persisted.
- Candidate scoring telemetry remains richer in some replacement paths than others. The implementation records new reservation/staleness reasons but does not retroactively reconstruct missing historic score vectors.
- The full offline suite validates control flow and fixtures, not live-provider mathematical quality or provider concurrency behavior.

## 13. Paid-test readiness

Another controlled paid test is technically justified only after rebuilding and recreating the API/worker containers and confirming their hashes match the repository. Before spending, run the same mocked Docker graph test inside the rebuilt worker. The local implementation and full suite are ready for that deployment verification; the currently running worker is not.
