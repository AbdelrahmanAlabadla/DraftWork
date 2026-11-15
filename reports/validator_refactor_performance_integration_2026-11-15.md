# Validator Refactor, Performance Optimization, and Pipeline Integration

Date: 2026-11-15

## A. Architecture Changes

### Line counts

| Item | Lines |
|---|---:|
| Original `app/online/validator.py` (baseline) | **6,833** |
| Final `app/online/validator.py` (compatibility facade) | **191** |
| `validation/orchestrator.py` | 1,598 |
| `validation/recovery.py` | 2,252 |
| `validation/questions.py` | 930 |
| `validation/legacy/validation.py` | 654 |
| `validation/legacy/repair.py` | 505 |
| `validation/contracts.py` | 371 |
| `validation/legacy/global_review.py` | 334 |
| `validation/context.py` | 332 |
| `validation/manifest.py` | 273 |
| `validation/duplicate_coordinator.py` | 208 |
| `validation/constants.py` | 99 |
| `validation/runtime.py` | 49 |
| `validation/state.py` | 9 |

`validator.py` is now a compatibility facade: imports/re-exports plus thin wrappers
for `validate_generated_questions`, `repair_invalid_questions`,
`_repair_invalid_questions_serial`, `_ensure_recovery_v3_decisions`, and
`_safe_record_repair_outcome`.

### Responsibilities moved

| Responsibility | New module |
|---|---|
| Recovery eligibility, dispatch, variation switching, difficulty fallback, repair staging state machine, recovery coordinator | `validation/recovery.py` |
| Specialized validation orchestration, candidate group validation, candidate duplicate gate, transactional commit | `validation/orchestrator.py` |
| Normalization, canonical fields, iterators, FITB word bank, deterministic local verdicts, duplicate heuristics, system-completion check | `validation/questions.py` |
| Legacy `validate_exam` and report merge | `validation/legacy/validation.py` |
| Legacy `repair_exam` | `validation/legacy/repair.py` |
| Dead global-review stack | `validation/legacy/global_review.py` |
| Prompts, output schemas, task contract, repair prompts | `validation/contracts.py` |
| Plan context, difficulty observations, regeneration adapter, `_validate_with_context` | `validation/context.py` |
| Repair manifest/lifecycle/placement | `validation/manifest.py` |
| Duplicate integration | `validation/duplicate_coordinator.py` |
| Budgets/actions/version constants | `validation/constants.py` |
| Cancellation helper | `validation/state.py` |
| Patch-compatible provider/coordination bridge | `validation/runtime.py` |
| Compatibility facade | `validator.py` |

### Public API compatibility

All historical imports continue to resolve through `validator.py`, including
`validate_generated_questions`, `repair_invalid_questions`, `validate_exam`,
`repair_exam`, `persist_difficulty_observations`,
`_integrate_duplicate_validation`, `_local_verdict`, `_iter_questions`,
`_build_by_id`, `_plan_context_for_model`, `_regenerate_plan_slot`,
`_manifest_from_report`, `_merge_repair_manifest`, `_canonical_values`,
`_canonical_changes`, `_fields_changed_disallowed`, `_normalize`,
`_LOGICAL_TO_CANONICAL`, `_ACTION_DEFAULT_FIELDS`, `_ALL_CANONICAL`,
`_solve_system_completion_verdict`, `recovery_eligibility`,
`recovery_v3_dispatch_action`, `record_repair_outcome`, and the budget constants.

Tests and runtime monkeypatches that target `app.online.validator.<name>` still
work: `validation/runtime.py` resolves provider, regeneration, legacy-repair, and
telemetry entry points through the facade at call time.

### Legacy paths isolated

- Legacy validation/repair and the unused global-review stack live under
  `app/online/validation/legacy/`.
- `_solve_system_completion_verdict` was preserved but isolated inside
  `validation/questions.py`; it is Python mathematical checking and should be
  retired in favor of the Quantitative Validator in a future change. It was not
  removed because it is still exercised by existing regression tests.
- `pending_revalidation_ids` remains a compatibility state key (still unused by
  the specialized path).

## B. Performance Changes

### Confirmed bottlenecks

- Validation/Recovery: 458.6 s of a 655 s job (70%), average concurrency 4.22 of
  30, only 5.9 s with no LLM call active.
- Candidate duplicate gates ran serially in the commit loop: 18 single-pair
  calls, 92.2 s summed; 4-pair calls cost 2.4 s/pair vs 5.1 s/pair.
- Revalidation payloads were 3–7× larger per question than initial validation
  because they carried original questions, full repair patches, and rejected
  candidates.
- `_role_cache_fingerprint` regenerated the Pydantic response schema on every
  role/question: 677 ms per 23 questions × 5 roles.

### Implemented

1. **Batched, parallel candidate duplicate gate** (`orchestrator.py`,
   `duplicate_judge.py`).
   - All ready candidates are inserted once into a synthetic model snapshot with
     unique synthetic question/slot identities.
   - Candidate-vs-active and candidate-vs-candidate pairs are screened together,
     reviewed in provider-sized batches concurrently, and cached through the
     shared fact ledger.
   - Own-slot replacement pairs are excluded explicitly.
   - Conflicts are resolved deterministically: candidate-vs-active rejects the
     candidate; candidate-vs-candidate keeps the earlier attempt.
   - Only clear candidates commit; the final authoritative duplicate scan still
     runs.
   - `review_duplicate_candidates` now accepts `excluded_question_pairs`.
2. **Revalidation prompt trimming** (`multi_validator.py`).
   - Revalidation context now carries only the role's original failure findings.
   - Removed original question, repair patch metadata, previous rejections, and
     candidate bookkeeping.
3. **Schema/static-contract caching** (`multi_validator.py`).
   - Response schema is hashed once per process.
   - Role instructions, field contract, provider identities, and schema hash are
     hashed once per role; per-question fingerprints only hash the dynamic input
     and assignment identity.
4. **Recovery concurrency** (`independent_recovery.py`).
   - Cross-model targeted recovery runs affected models concurrently; the
     duplicate scan/merge remains deterministic and serial.
5. **Numerical contract deduplication** (`multi_validator.py`).
   - Removed `given_conditions`, `effective_givens`, `source_example_givens`, and
     `source_variation_givens` duplicates from validator contracts.
   - `authoritative_givens`, `value_policy`, `illustrative_values`, and
     `immutable_values` remain the single authority; the illustrative-vs-immutable
     fix is intact.

### Still sequential by design

- Atomic Repair steps for one question (each patch depends on the previous).
- Transactional candidate commit loop.
- Model-level duplicate scan and cross-model barrier after both models are ready.
- Final authoritative duplicate scan.
- Provider concurrency limits were not changed.

## C. Benchmarks

Controlled offline benchmark: 2 × 23-question synthetic exam, mocked provider at
120 ms/call, identical decisions. These are relative measurements, not provider
wall-clock predictions.

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| Initial validation (46 q, 70 calls) | 6.58 s | **4.11 s** | **−37.5%** wall (same calls) |
| Revalidate after 1 changed question | 35 calls / 0.94 s | **5 calls / 0.15 s** | **−86% calls**, −84% mock wall |
| Revalidate 8 changed q: per-candidate vs batched | 40 calls / 1.12 s | **20 calls / 0.17 s** | **−50% calls**, −85% mock wall |
| Five repeated duplicate scans | 85 calls / 1.25 s | **17 calls / 0.26 s** | **−80% calls**, −79% mock wall |
| Eight candidate gates, reset vs shared cache | 136 calls / 1.99 s | **24 calls / 1.17 s** | **−82% calls**, −41% mock wall |
| Duplicate cache hit rate | 0% | **67–100%** | n/a |
| Fingerprint CPU, 23 q × 5 roles | 677 ms | **12.6 ms** | **−98%** |
| Revalidation payload, task alignment | 5,011 B | **2,982 B** | **−40.5%** |
| Revalidation payload, grounding | 3,871 B | **1,842 B** | **−52.4%** |
| Revalidation payload, answer correctness | 2,517 B | **488 B** | **−80.6%** |
| Revalidation payload, difficulty | 2,732 B | **703 B** | **−74.3%** |
| Complete mocked exam runtime | not controlled | not controlled | unavailable |

Production observations remain the 458.6 s Validation/Recovery window from
`exam_bfe29ab579f1`. No live exam was run; the optimized production wall time is
not claimed.

## D. Full Pipeline Integration

| Pipeline Component | Old Import / Route | New Import / Route | Verification |
|---|---|---|---|
| FastAPI / Celery entry | `app.api.*`, `app.jobs.tasks` → graph | unchanged; graph imports facade | full suite, API tests |
| Validation stage | `graph` → `validator.validate_generated_questions` | facade → `validation.orchestrator` | `test_question_validator`, `test_40q`, `test_60q` |
| Repair | `graph`/`independent_recovery` → `validator.repair_invalid_questions` | facade → `validation.recovery` | `test_multi_repair`, `test_performance_optimizations` |
| Recovery | `independent_recovery` → facade | facade → `validation.recovery` | `test_independent_model_recovery`, `test_recovery_v3` |
| Regeneration | recovery → `_regenerate_plan_slot` | recovery → `runtime` → facade → context | `test_regeneration_revalidation_integration` |
| Revalidation | orchestrator → `multi_validator` | unchanged (specialized pipeline) | `test_validation_repair_difficulty_recovery` |
| Duplicate checking | orchestrator → `duplicate_judge`/`duplicate_pipeline` | batched `review_duplicate_candidates` + final scan | `test_duplicate_*`, new batch gate tests |
| Cross-model checking | `independent_recovery` → `cross_model_duplicates` | concurrent affected models, serial merge | `test_cross_model_duplicates` |
| Final assembly | graph → `exam_builder` | unchanged | `test_40q`, `test_graph_pipeline_integration` |
| Persistence / Export | repositories, exporters | unchanged | `test_exports_pdf_docx`, `test_exam_store` |

Internal callers updated to import shared helpers directly:
`duplicate_pipeline.py`, `multi_repair.py`, and `multi_validator.py` now use
`validation.questions`, `validation.context`, and `validation.legacy.repair`
instead of importing helpers from `validator.py`. Public/patch-sensitive calls
still go through the facade.

## E. Regression Results

- Full offline suite: **1,431 passed, 21 skipped, 2 failed**.
- Focused validation/duplicate/repair suite: **226 passed, 1 skipped**.
- Mocked end-to-end/integration set (`40q`, `60q`, independent recovery,
  recovery v3, exam builder, regeneration/revalidation): **79 passed, 2 skipped**.

Remaining failures, both verified unrelated to the refactor:

1. `test_graph_pipeline_integration::test_saved_103_chunk_catalog_mocked_graph_reaches_final_assembly` —
   the fixture document `175ef217-a754-4e58-8bae-2bbb4841f824` has **0 child
   rows** in the configured PostgreSQL database (queried directly). The test
   requires 103 saved chunk rows.
2. `test_recovery_v3::test_job_scheduler_allows_fifty_recovery_requests` —
   `peak == 50` in-flight assertion under full-suite load. It passes when run
   alone; the failure is thread-overlap timing, and no scheduler code was
   changed.

## F. Remaining Limitations

- `validation/orchestrator.py` (1,598) and `validation/recovery.py` (2,252) are
  still larger than the 500-line target. Their nested closures and the single
  1,300-line recovery state machine require behavior-preserving extraction of
  commit/candidate-validation and staging sub-phases; that is the next
  maintainability step.
- Production wall-clock improvement must be verified with one controlled live
  exam. Offline benchmarks isolate orchestration/caching; they do not model
  provider prefill or network variance.
- Candidate-vs-candidate conflicts are resolved by keeping the earlier attempt;
  this is deterministic and safe but may reject a useful later candidate.
- `_solve_system_completion_verdict` remains Python mathematical validation for
  legacy compatibility; it should be retired in favor of the Quantitative
  Validator once its tests are migrated.
- The batched gate adds one synthetic model assembly per validation round; this
  is CPU-only and measured as negligible relative to the eliminated serial calls.

## G. Repair-route follow-up: redundant Answer Repair

The failed 20:29 exam's `model2_equation_2` was sent to the Answer Repair agent
even though its answer had already been corrected by the Solution step. Root
cause: `task_alignment/wrong_variation` declares dependencies
`("solution", "answer")`; for Equation, Solution already owns
`equation/solution_steps/final_answer/solution_certificate`, so the dependent
Answer step targeted `final_answer` redundantly and returned the same value.

Fix:
- `build_repair_route()` tracks directly requested agents and removes a
  dependency-only agent when every field it owns is already covered by another
  step.
- `REPAIR_AGENTS` now places Difficulty before Solution so a difficulty rewrite
  cannot invalidate a solution that was just rebuilt (this caused the second
  Solution call in the same failed exam).

Tests: `test_equation_task_alignment_route_skips_redundant_answer_step`,
`test_direct_answer_defect_still_routes_to_answer`, and the existing repair suite
(51 passed).

## H. Cross-model recovery follow-up: `_model_number` shadowing

The 20:54 exam (`exam_24f231a77f76`) failed inside the cross-model phase with:

```text
TypeError: 'int' object is not callable
  independent_recovery.py, _targeted_cross_model_recovery
  reports = { _model_number(report.get("model_number"), 0): report ... }
```

Root cause: the single-affected-model branch introduced during the concurrency
change used:

```python
_model_number, worker = recover_model(...)
```

which bound the module-level `_model_number()` helper to an `int` for the rest of
the function. The following report comprehension then called that integer.

Fix: use a distinct local name (`_resolved_model`). Added regressions:
- `test_single_model_cross_model_recovery_does_not_shadow_model_helper`
- `test_one_cross_model_target_completes_targeted_recovery_and_final_scan`

The second test exercises the full path: local model workers → cross-model scan
→ one targeted cross-model recovery → second/final cross-model scan → returned
state ready for assembly.

### Cross-model conflicts in that exam

The cross-model scan found four Model 1 vs Model 2 overlaps:

| Pair | Same knowledge | Same operation | Same target | New reasoning | Answer give-away |
|---|---|---|---|---|---|
| `m1_mcq_1` / `m2_true_false_3` | yes | yes | yes | no | yes |
| `m1_mcq_3` / `m2_true_false_4` | yes | yes | no | no | yes |
| `m1_true_false_1` / `m2_true_false_2` | yes | yes | no | no | yes |
| `m1_true_false_2` / `m2_true_false_5` | yes | yes | yes | no | yes |

Targeted cross-model recovery ran (repairs/revalidation at 20:58–20:59), and the
subsequent duplicate call reported no remaining conflict; the crash then occurred
before the post-recovery cross-model integrate/assembly could complete.


