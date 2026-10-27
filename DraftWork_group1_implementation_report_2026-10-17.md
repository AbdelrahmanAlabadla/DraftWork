# DraftWork Group 1 Implementation Report

Date: 2026-10-17
Reference job: `4e7a3194-3eea-49cc-8bde-8ff7708b91fd`

## 1. Confirmed root causes

The benchmark's 1/4 Word Problem result came from one shared contract problem rather than an arbitrary reuse limit. The catalog represented the tennis exercise as a narrow instance whose illustrative values were stored as required givens. That narrow contract was then propagated to generation and validation. Replacements that changed the values or assessed another valid angle could therefore be classified as task drift.

The Planner's deterministic allocation order was not the root cause. The benchmark catalog exposed only one compatible Word Problem task, so reuse was necessary. The existing allocator already prefers globally unused compatible tasks, then compatible tasks used by another question type, then same-type reuse.

The Generator received reuse context, but the shared task contract did not explicitly distinguish invariant facts from variable exercise parameters or list the source-supported variation space. A deterministic acceptance gate also compared semantic plan identity and could reject a generated instance because its task ID matched an earlier question, before content-level duplicate checks made the decision.

Duplicate Recovery could select alternatives, but its final same-task option was effectively excluded by identity bookkeeping. Its third-attempt numerical fallback also had no explicit catalog authorization check. This created the contradictory outcomes seen in the benchmark: changing values could be rejected as drift, while retaining them produced another duplicate.

These failures shared one root cause: the pipeline lacked a durable, cross-agent distinction between invariant source requirements and permitted question-instance variation.

## 2. Implementation summary

### P1 — reusable Catalog contract

`CatalogTaskDraft` and `PlanItem` now carry four backward-compatible fields:

- `allowed_variations`: source-supported target, operation, representation, reasoning, application, or numerical-instance changes.
- `variable_givens`: illustrative quantities or parameters that may change.
- `complexity_boundaries`: unsupported concepts or difficulty increases that must not be introduced.
- `exact_source_instance`: an explicit exception for tasks that must preserve the original worked instance.

The Catalog compiler prompt now states that `required_givens` contains invariants only. Ordinary worked-example numbers belong in `variable_givens`. The finalizer, materializer, replacement paths, plan fingerprints, diagnostic context, validator context, and final semantic identity output preserve the new fields.

The catalog cache version is `catalog-v9-reusable-task-contract`. Existing files are retained, while new work uses the new contract and cannot silently consume an old catalog as if it contained variation metadata. No database schema or Docker setting changed.

Files: `app/online/assessment_catalog.py`, `app/llm/schemas.py`, `app/config.py`, `app/online/planner.py`, `app/online/plan_review.py`, `app/online/difficulty_adjustment.py`, and `app/online/exam_builder.py`.

### P2 — allocation preserved

The deterministic allocation policy was left intact. The only Planner changes preserve the four new catalog fields when plans are normalized or replaced. The Planner still does not invent question text, numerical values, or variation details.

Files: `app/online/planner.py` and `app/online/plan_review.py`.

### P3 — diverse same-task generation

The Generator prompt now uses the shared variation contract and applies this order when a task is reused: change target/output, operation or reasoning, application/structure, given/unknown combination, then use a valid recalculated numerical instance. It compares same-task items inside one response and receives prior accepted questions, the rejected question, conflict questions, and prior plan angles during recovery.

The task-ID semantic gate no longer treats shared plan identity as a duplicate. Exact-text and normalized-content checks remain active, as do same-batch checks and the final Duplicate Judge. A wording-only duplicate is therefore still rejected; a distinct question with the same task ID is not rejected merely because of that ID.

Files: `app/online/prompts.py` and `app/online/exam_builder.py`.

### P4 — recovery with the same task

Recovery still prefers a useful different compatible task, but the original task ID is now available as the final catalog-backed choice. That regenerated candidate receives the rejected question, the conflicting accepted questions, the judge conflict information, and failed-attempt context.

The final same-task candidate is validated transactionally before it mutates the live exam. A failed precommit check rolls the plan back, records `precommit_rejected`, remembers the failed approach, and leaves the original question intact. Successful replacements still enter normal General/Quantitative revalidation and Duplicate Judge verification.

Third-attempt numerical variation now requires explicit `variable_givens` or a numerical variation authorization. It is disabled when `exact_source_instance` is true. The existing three-attempt limit is unchanged.

File: `app/online/duplicate_pipeline.py`.

### P5 — cross-agent consistency

The General Validator now accepts authorized target, operation, reasoning, representation, and numerical-instance changes while continuing to reject changes outside `allowed_variations`, crossings of `complexity_boundaries`, missing invariants, unsupported source claims, and incorrect work. It explicitly distinguishes neutral constructed exercise data from source facts.

Difficulty Adjustment sees the same invariant, variable, variation, and boundary fields. Plan Review includes them in task fingerprints and replacement records. Duplicate Recovery includes them in candidate and conflict context. Final semantic identities retain them for auditability.

Files: `app/online/validator.py`, `app/online/difficulty_adjustment.py`, `app/online/plan_review.py`, `app/online/duplicate_pipeline.py`, and `app/online/exam_builder.py`.

## 3. Before and after behavior

### One task, multiple questions

Before, four uses of the tennis task inherited the same target and values, and changing the values risked `task_drift`.

After, one linear-systems task can authorize:

1. Solve a new totals problem for two unknown counts.
2. Verify whether proposed counts satisfy both equations.
3. Construct the two equations from a supported context.
4. Solve another valid numerical instance after recalculating its answer.

All four retain the same source-grounded concept and educational boundary. Their task ID may be identical because duplicate status is decided from the actual assessment work and question content.

For constant speed, the same task can authorize distance from speed/time, time from distance/speed, speed from distance/time, and a new numerical instance when the distinct targets are exhausted. Acceleration remains invalid when listed outside the complexity boundary.

### Numerical changes

The values `13` matches and `38` player participations may now be cataloged as variable givens. Nonnegative integral counts and the singles/doubles participation structure remain invariant constraints. A generated replacement can use different consistent totals only when the catalog authorizes that variation, and it must recalculate the equations and answer.

### Duplicate handling

Changing only “car” to “vehicle” while retaining the same values, unknown, method, and answer remains a content duplicate. Changing the unknown from distance to time, with a coherent new instance and solution, is eligible when the catalog lists that target variation.

### Recovery

When no different compatible task exists, Recovery may regenerate the original task with a different authorized assessment angle. It validates that candidate before commit, then schedules the normal validators and Duplicate Judge. If the precommit validation fails, the live exam and original plan remain unchanged and the failed approach is recorded.

## 4. Cross-agent conflicts resolved

| Conflict | Resolution |
|---|---|
| Catalog treated illustrative numbers as immutable | Split `required_givens` from `variable_givens`; added exact-instance opt-in. |
| Generator allowed constructed data but Validator could call it drift | Both now consume `allowed_variations`, `variable_givens`, boundaries, and exact-instance status. |
| Same task ID could trigger deterministic rejection | Removed identity-only rejection; content checks and Duplicate Judge remain. |
| Recovery could not return to the original task | Original task is a valid final candidate after alternatives. |
| Numeric fallback could mutate an instance without task authorization | Fallback now requires explicit numeric authorization and rejects exact-source tasks. |
| Same-task fallback could affect live state before validation | Added transactional precommit validation and rollback. |
| Variation metadata disappeared in later stages | Preserved it through planning, review, adjustment, duplicate context, validation, and final semantic identities. |

No demonstrated cross-agent contradiction remains in the offline paths covered by the suite.

## 5. Test results

Compilation:

- `python -m compileall -q app/online app/llm/schemas.py` — passed.

Full offline suite:

- `pytest -q` — **776 passed, 4 skipped, 0 failed** in 327.47 seconds.
- The four skips are existing environment-dependent tests.

Focused coverage includes:

- Four distinct same-batch questions sharing one task ID.
- Same-task identity no longer causing automatic rejection.
- Exact wording/content duplicates still rejected by deterministic checks.
- Previous accepted and rejected/conflicting questions reaching regeneration context.
- Same-task recovery with transactional rollback on failed validation.
- Authorized numerical fallback and mandatory duplicate recheck.
- Numerical fallback denied without authorization and for exact-source instances.
- Reusable catalog metadata surviving materialization and validator serialization.
- Existing plan review, repair limits, question types, quantitative validation, and duplicate lifecycle behavior.

`git diff --check` passed. It reported only existing Windows line-ending notices.

## 6. Remaining risks

- No paid benchmark was run, as required. Offline tests verify prompts, metadata, deterministic gates, retries, and lifecycle behavior, but cannot measure how consistently a production model follows the new diversity order.
- Old cached catalogs are preserved but deliberately not treated as the new contract. The next use of an old selection may require normal catalog recompilation. That is necessary to avoid guessing which historical values were illustrative.
- `allowed_variations` quality depends on the Catalog model accurately identifying what the selected source supports. Complexity boundaries and validators limit unsupported expansion, but a real benchmark is still needed to measure model adherence.
- The architecture keeps an atomic base task plus permitted variations rather than introducing a new task-family/question-instance storage model. This minimizes compatibility risk, but broad source concepts still depend on well-formed `allowed_variations`.
- No runtime latency or question-count improvement is claimed without a new real exam.

## 7. Final assessment

| Group | Status | Evidence |
|---|---|---|
| P1 | **Fixed offline** | Reusable contract added, propagated, cache-versioned, and tested. |
| P2 | **Fixed / preserved** | Existing deterministic allocation remains; metadata survives its paths. |
| P3 | **Fixed offline** | Same-ID gate corrected; same-batch/prior context and diversity contract tested. |
| P4 | **Fixed offline** | Same-task recovery, authorization, precommit validation, rollback, and recheck tested. |
| P5 | **Fixed offline** | Shared rules reach all requested stages; full suite passes. |

Production-model effectiveness remains unverified until an explicitly authorized paid benchmark is run.
