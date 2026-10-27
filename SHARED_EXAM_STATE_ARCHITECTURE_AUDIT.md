# Shared Exam State Architecture Audit

## Scope and conclusion

This read-only audit covers job `cc84370b-a450-432e-bf05-0bf130dbb973`, the current LangGraph implementation, PostgreSQL checkpoint reconstruction, and `exam_forensic_cc84370b/`.

**CONFIRMED:** DraftWork does not need a separate memory system. `ExamState` already contains most necessary information (`app/online/graph.py:12-95`), node outputs are appended as checkpointed audit events (`app/online/graph.py:98-160`), and resume reconstructs state by monotonically applying node outputs (`app/api/repositories.py:756-806`). The architectural weakness is that mutable plans and generated exams serve as both current truth and history, while assignment/content revisions, same-round reservations, and fingerprint-bound duplicate facts are incomplete.

The minimum correction is an authoritative per-slot record with revisioned assignment and question identities, derived exam-wide indexes, append-only transition events, and short-lived coordinator reservations. Agents should continue to receive compact role-specific views rather than the entire graph state.

## Existing state and durability

`ExamState` currently contains:

- Request identity: immutable request, request fingerprint, schema version, selected child IDs, subject, language, difficulty, and requested counts.
- Retrieval/Catalog: retrieved chunks, compact context, Planner context, Catalog, cache key, and cache-hit flags.
- Planning: current plans, profile, issues/errors, attempts, failed candidates, unrepairable slots, plan duplicate state, and difficulty history.
- Generation: current generated exams, rejection feedback, and MCQ reassignments.
- Validation/repair: reports, validation ledger, validated models, per-slot attempts, repair manifest/candidates/seeds/memory, pending revalidation, invalid questions, and repair histories.
- Duplicate review: reports, score distributions, pending groups, attempts, blacklists, failures, unresolved conflicts, scan count, and fact cache.
- Cross-stage: `slot_lifecycle`, diagnostics, telemetry, warnings, and error.

### Durable mechanism

`_audit_node_output()` deep-copies each node result and adds request/Catalog/retrieval identity when absent (`app/online/graph.py:113-148`). For generation onward it also persists plans and generated exams after each stage. PostgreSQL writes each event atomically with job ownership and monotonic checkpoint revision checks (`app/api/repositories.py:1082-1129`). Resume accepts only the same request fingerprint and folds checkpoint outputs in revision order (`app/api/repositories.py:756-806`).

### Temporary fields

Callbacks and cancellation functions are process-only. Repair candidates are staging objects, but they are included in node output when returned. Thread-local isolated repair snapshots are temporary. Duplicate candidate reservations exist as local variables inside a repair round and are not a first-class durable reservation ledger.

### Overwritten or lossy information

- `plans` is the latest mutable assignment set. Original assignments survive only in prior audit events or separately exported snapshots.
- `generated_exams` is the latest active content. Rejected intermediate candidates are inconsistently represented in histories.
- `validation_reports` describes the current/last validation set; the ledger/history retains some prior facts, but verdict-to-content dependency is not uniformly authoritative.
- `pending_duplicate_groups` is replaced by each scan. `unresolved_duplicate_conflicts` can carry old group membership without endpoint fingerprint validation.
- Candidate score vectors and eligibility rejection reasons are not consistently persisted.
- `slot_lifecycle` reconciles summaries after each node (`app/online/graph.py:212-219`) but is not a substitute for immutable assignment and content revisions.

## Stage-by-stage state matrix

| Stage | Reads | Produces/writes | Visibility and recovery | Missing or weak information |
|---|---|---|---|---|
| Retrieval | request, selected child IDs | retrieved chunks, contexts, processed key | Global; checkpoint event | Retrieval query/ranking details are not primary slot state |
| Catalog Compiler/Finalizer | retrieved chunks, domain/language | Catalog, profile, key, usage/cache metadata | Global and durable | Per-field finalizer corrections and compiler provenance are limited |
| Planner | Catalog, profile, request counts | plans, issues, errors, scores | Global and durable | Original assignment lacks a permanent per-slot authoritative field after mutation |
| Difficulty Adjustment | current plans, Catalog, history | mutated plans, attempts/history | Sequential, globally visible, durable | No explicit assignment revision/precondition |
| Plan Review/Repair | current plans/errors/Catalog | repaired plans, failed candidates | Global and durable | Candidate rejection score vectors incomplete |
| Independent Generation | plans, assigned evidence/chunks | generated exams and generation checkpoints | Calls parallel; coordinator merges; durable checkpoints | Content revision and assignment revision used by each response are not explicit everywhere |
| General Validation | current questions, plans, evidence | reports, ledger, manifest | Batch parallel; coordinator merges; durable | Verdict dependency/version contract is distributed rather than one canonical record |
| STEM Validation | same quantitative subset | quantitative review within reports | Parallel and durable through validation output | Separate role fingerprint should be explicit and versioned |
| General/STEM Repair | manifest, reports, plans, active questions, per-slot memory | staged candidates, updated attempts/memory/history | Per-slot isolated snapshots; merged later | Sibling assignment reservations absent; stale precondition checks incomplete |
| Selective Revalidation | pending IDs, candidate exams/plans, ledger | accepted/rejected candidate state, new verdicts | Coordinator validation and merge | Need atomic compare-and-swap on assignment/content revisions |
| Duplicate Screening | active exam, plans, fact cache | candidate pairs, similarities, reports | Exam-wide and durable | Pair cache key/version is not exposed as an authoritative pair ledger |
| Duplicate Judge | shortlisted/forced pairs and metadata | classifications/reasons | Results embedded in reports | Endpoint fingerprints and resolution status should be durable fields |
| Duplicate Replacement | groups, histories, plans, active questions, Catalog | sequential reservations, staged generation, histories | Reservations sequential in-node; output durable | Reservation plan is temporary; partial component decisions can become stale |
| Final Validation | changed IDs and staged candidates | final correctness verdicts | Durable | Must require current content/assignment revision match |
| Assembly/Persistence | latest plans/exams/manifests/unresolved groups | saved exam, warnings, lifecycle metadata | Durable saved exam | It trusts unresolved group snapshots instead of rebuilding from current pair evidence |

## What each agent actually sees

- Planner and Difficulty Adjustment work from global Catalog/plan state.
- Independent generators receive one slot assignment and evidence snapshot; parallelism does not mutate graph state directly.
- Validator batches receive compact question/assignment/evidence payloads, not arbitrary shared memory.
- General/STEM Repair receives isolated per-slot state for non-FITB targets. `repair_invalid_questions()` creates isolated jobs at `app/online/validator.py:3735-3753`, runs them concurrently at `3755-3763`, and merges by slot at `3765-3808`.
- Duplicate Replacement sees the active exam and all current plans and reserves candidates sequentially within one round (`app/online/duplicate_pipeline.py:1031-1038`).

This role-specific context is appropriate. The defect is not limited context itself; it is the absence of a coordinator-owned revision and reservation contract shared by all selection paths.

## Minimum necessary state model

### Per-slot fields

| Information | Treatment | Reason |
|---|---|---|
| Immutable `slot_id`, model, type, order | **Primary authoritative** | Stable identity across all stages |
| Original Planner assignment | **Append-only event plus compact primary reference** | Must never be reconstructed from final plan |
| Current effective assignment | **Primary authoritative** | One source for every downstream payload |
| Assignment revision/fingerprint | **Primary authoritative** | Detect stale generation, validation, and repair results |
| Assignment transition history/reason/actor | **Durable append-only event** | Avoid copying every full plan version into slot state |
| Task/variation/family/operation/target/effective givens/evidence IDs | **Part of current assignment** | Required contract; avoid parallel copies |
| Difficulty and adjustment history | Current difficulty **primary**; history **event** | Current routing plus forensic trace |
| Current question and content revision/fingerprint | **Primary authoritative** | Bind verdicts and duplicate facts to exact content |
| Full prior question payloads | **Append-only only when candidate/active transition matters** | Avoid large redundant state |
| General/STEM verdicts and rule/model/schema versions | **Primary latest verdict references + events** | Safe selective reuse |
| Repair attempts, strategy, rejected candidate identity | Counters/memory **primary**; outcomes **events** | Routing needs compact counters; audit needs history |
| Terminal/exhaustion status | **Primary authoritative** | Prevent re-entry loops |
| Source chunk text | **Reference by chunk ID/hash** | Text already exists in retrieved/Catalog storage |

### Exam-wide fields

| Information | Treatment | Reason |
|---|---|---|
| Task/variation/scenario/operation usage | **Derived index** from current slot assignments | Prevent divergence |
| Realized mathematical instances | **Derived index** from active question fingerprints/certificates | Needed after generation |
| Active candidate reservations | **Temporary coordinator reservation**, checkpoint only at milestone | Short-lived consistency mechanism |
| Pairwise screening/Judge facts | **Durable keyed ledger** | Reuse unchanged pair evidence safely |
| Pair endpoint fingerprints and rule/model versions | **Part of pair-ledger key** | Automatic invalidation on change |
| Unresolved groups | **Derived** from current pair ledger | Never preserve stale memberships as truth |
| Global replacement plan and outcomes | Plan **primary while active**; decisions/outcomes **events** | Coordinate a component atomically |
| Checkpoint/graph revision | **Already available** | Monotonic persistence fencing exists |

Do not store repeated prompts, schemas, source text, or complete exam copies in every slot. Store hashes, versions, IDs, and event references.

## Coordination across replacement stages

Difficulty Adjustment, General/STEM Repair, and Duplicate Replacement should remain separate:

- Difficulty Adjustment changes difficulty-compatible assignments before content exists.
- General/STEM Repair resolves correctness/grounding failures.
- Duplicate Replacement resolves cross-question conflicts after correctness.

They can share one deterministic coordinator API built on the existing eligibility and ranking functions:

1. Read a current assignment snapshot and its graph revision.
2. Derive global usage and realized-instance indexes.
3. Build candidates with the common hard filter.
4. Rank against active assignments, questions, rejection history, and existing reservations.
5. Reserve candidate assignment/instance identities sequentially under the coordinator.
6. Dispatch generation/repair calls concurrently after reservations are fixed.
7. Commit each qualified result only if expected assignment/content revisions still match.

This keeps agents independent and permits task reuse when the realized assessment instance is materially distinct.

## Concurrency and consistency

### Current behavior

Independent generation is safe structurally: immutable slot snapshots go to workers and the coordinator merges by stable slot ID. General Repair differs because selection itself happens inside concurrent isolated snapshots. Two branches can choose equivalent tasks or mathematical instances without seeing each other's choices.

### Required consistency boundary

Global assignment decisions should be serialized or atomically reserved; paid generation can remain up to 50-way parallel. An optimistic protocol is sufficient:

- Each reservation records `slot_id`, expected assignment revision, candidate assignment fingerprint, intended instance constraints, and reservation revision.
- A result returns those revisions and its realized question fingerprint.
- Coordinator commit uses compare-and-swap semantics. A stale result is not published and does not overwrite newer state.
- Before commit, check the candidate against active questions and all successful staged siblings.
- Database checkpoint ownership and monotonic revisions remain the outer fencing mechanism.

No shared mutable graph state should be exposed to worker threads.

## Duplicate findings lifecycle capability

| Required step | Current support | Correction |
|---|---|---|
| Collect all pair findings first | **Implemented** by full scan/Judge report | Retain |
| Persist scores, reasons, metadata, verdicts | **Mostly implemented** in duplicate reports | Add endpoint/rule fingerprints as required keys |
| One coordinated replacement plan | **Partial**: groups and sequential replace IDs | Persist component-level plan before candidate generation |
| Stage replacements | **Implemented** | Make staging identity/revision explicit |
| Validate before commit | **Implemented in substantial part** | Require atomic revision checks |
| Compare staged candidates with active and siblings | **Incomplete** | Add one coordinator precommit pass |
| Invalidate changed endpoint pairs only | **Partial fact cache** | Replace opaque cache behavior with versioned pair ledger |
| Reuse unchanged findings | **Partial** | Reuse only exact fingerprint/version matches |
| Recompute unresolved groups | **Missing at finalization** | Always derive from current pair ledger |

The reference execution demonstrates the gap: scan 3 found a three-node group, one endpoint changed, the old group was retained, and routing skipped a final scan.

## Prioritized improvements

### P0

1. Add authoritative assignment and content revisions/fingerprints per slot; require them in every asynchronous result and commit.
2. Replace persisted unresolved-group snapshots as final truth with groups derived from a fingerprint-bound pair ledger.
3. Coordinate all same-round assignment reservations before concurrent General/STEM Repair generation.

### P1

4. Persist compact assignment/content transition events with actor, reason, predecessor revision, and candidate disposition.
5. Persist candidate score vectors and deterministic rejection reasons for Planner, Difficulty Adjustment, Repair, and Duplicate Replacement.
6. Add staged-sibling duplicate checks before atomic replacement commit.

### P2

7. Normalize role-specific General/STEM verdict fingerprints and expose why a prior verdict was reused or invalidated.
8. Reduce checkpoint duplication by referencing immutable chunks/Catalogs by key and retaining milestone snapshots plus append-only deltas.

## Verification limitations

The forensic export provides stage snapshots and event histories but not every temporary thread-local object or candidate score vector. The repository describes current behavior; the export does not embed a source revision for every event. Conclusions about the reference duplicate lifecycle are supported by recorded events and matching state contracts.
