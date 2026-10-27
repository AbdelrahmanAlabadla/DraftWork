# DraftWork Task-Family Grouping and Source-Grounded Variations

## Implementation summary

The Assessment Catalog now treats a task as a coherent educational objective or task family. The compiler is instructed to inspect every selected source chunk, identify objectives, enumerate all supported targets, operations, reasoning directions, representations, and applications, and then decide whether each opportunity is a variation of one objective or a separately taught objective or method. It has no task or variation quota and must not invent opportunities, use filler labels, or remove coverage to save tokens.

The existing top-level operation, target, work product, and reasoning mode remain as the primary angle so the 18-field compiler task contract and older downstream assumptions remain usable. Alternative angles are stored in `allowed_variations`. During planning, code exposes the primary angle and alternatives together as `permitted_variations`; the Planner still allocates task IDs only.

Each normalized variation now carries:

- a deterministic `variation_id`;
- target, operation, and givens;
- compatible `question_types`;
- optional `reasoning_mode` and `student_work_product`;
- optional representation and application;
- optional `evidence_chunk_ids`;
- the existing bounded numerical-change permission.

There is no minimum or maximum variation count. Zero-variation tasks remain valid. Duplicate variation objects are merged by semantic identity, with question-type and evidence coverage combined.

## Why the previous Catalog produced mostly one variation per task

The previous compiler prompt said that one task was exactly a concept plus a narrow target, student operation, and expected work product. It explicitly directed the model to create separate tasks when operation, target, or work product differed. Canonical `task_id` and `task_family_id` hashes then included those same fields. The compiler therefore had both an instruction and a deterministic identity incentive to split nearby assessment opportunities. Its variation guidance described bounded reuse but did not require a complete search for variations within each educational objective. This made a variation that repeated the primary angle a safe output while other supported targets became separate tasks.

The new prompt reverses that framing: objective discovery happens first, variation discovery happens within each objective, and independently taught methods remain separate.

## Canonical identity and duplicate grouping

`task_id` now hashes the knowledge anchor, concept, core family operation or method, semantic relationship, invariant givens, material scenario, and complexity boundaries. It no longer hashes one primary target, work product, or reasoning mode. `task_family_id` uses the anchor, concept, relationship, and material scenario. This makes harmless changes to the selected primary angle stable while preserving method, relationship, restriction, and context distinctions that prevent broad collisions.

When multiple compiler rows canonicalize to the same task objective, finalization merges evidence and question-type coverage and converts every distinct later primary angle into a normalized variation. No assessment target is silently discarded.

Variation IDs hash the target, operation, givens, reasoning mode, work product, representation, application, and numerical-change permission. Evidence location and compatible output form can expand without changing the semantic variation ID.

## Evidence behavior

A variation may name a narrower `evidence_chunk_ids` list. Omission inherits the parent task evidence. Python verifies explicit IDs against both the selected source and the parent task evidence scope. An empty or out-of-scope explicit list causes that variation to be discarded while the valid parent task and its other variations remain available. Evidence references are treated as boundaries, not semantic proof; the existing validators still judge the generated educational content.

The Generator receives the parent source plus each variation's evidence declaration. After it reports its selection, code records `selected_variation`, `selected_variation_id`, and `effective_evidence_chunk_ids`. Validators receive only the effective evidence chunks for that question.

## Question-type compatibility and generation

Task-level `allowed_question_types` is finalized as the union of the primary angle and all variation-level `question_types`. Each variation retains its narrower compatibility. The Planner continues allocating a task ID using the task-level union. The Generator must select one entry in `permitted_variations` compatible with the assigned question type and return its ID. Code rejects an unknown ID or a variation whose type does not match the slot before accepting the generated question.

The selected angle is included in `expected_catalog_task` for both General and Quantitative Validators, together with effective evidence. Validator instructions require semantic consistency among the selected variation, question, answer, solution, difficulty, type, and source without exact wording matching. Existing bounded question repair handles violations.

Repeated allocation remains permitted through the existing Catalog-exhaustion/reuse contract. Prior selected variation IDs and metadata are included in reuse context where available. They guide diversity but do not impose a one-use rule. Duplicate Judge and its existing recovery limits remain authoritative for realized overlap.

## Saved Catalog evaluation

The saved Luna artifact contains 116 tasks and was inspected offline. No compiler or provider call was made, so the examples below are architectural projections rather than claims about improved live extraction.

### Account-balance family: appropriate grouping

Before, three real Catalog tasks share the anchor `Book-club activity account balance` and evidence `3f88cabc-b39e-4913-b059-4208f86e4eaa`:

| Existing task ID | Operation and target | Types |
|---|---|---|
| `task_378742792bd4b8db` | `model_linear_balance` → construct the balance equation | Equation |
| `task_523aafb35a7b1069` | `find_initial_balance` → original account amount | MCQ, Word Problem |
| `task_2e754b2dfda6ce06` | `find_zero_balance_time` → meetings until no money remains | MCQ, Word Problem |

Under the new contract, a source-grounded compiler response can represent these as one objective. Applying the implemented canonical functions offline to that grouped representation gives projected task ID `task_d387fbb43da502a0`, with variations:

| Variation ID | Target | Compatible types | Effective evidence |
|---|---|---|---|
| `variation_1f050e1b3c63b438` | remaining-balance equation | Equation | inherits `3f88cabc-b39e-4913-b059-4208f86e4eaa` |
| `variation_94305b6a77cfbe15` | initial balance | MCQ, Word Problem | inherits the same parent evidence |
| `variation_bc4c809ff3b9cc74` | meeting count until zero balance | MCQ, Word Problem | inherits the same parent evidence |

This removes redundant IDs while preserving all three operations and the union of their real question-type coverage. A future compiler run may choose different concise family wording and therefore a different deterministic task ID; no live-extraction claim is made here.

### Elimination method: appropriate separation

The saved artifact has five tasks under the Arabic anchor `طريقة الحذف`, but they do not all represent one interchangeable objective. For example:

- `task_05c1c2dfe2d59143` identifies the suitable method from a coefficient condition using evidence `f38b47e0-d943-4a96-9ebe-0f474926d138` and supports MCQ.
- `task_235d9c6e1897ec86` performs elimination, including creating opposite coefficients, using evidence `f38b47e0-d943-4a96-9ebe-0f474926d138` and `02ea5612-4853-446e-a967-20fc1295a8ff`, and supports Equation and MCQ.
- `task_2eb975613d23bca5` explains why cancellation works, using those same two chunks, and supports Why.

Identification, execution of the taught elimination method, and explanation of its invariant may be separate curriculum objectives. The new identity retains the core operation or method and restrictions, so these are not collapsed merely because they share an anchor. A compiler may group only angles the source presents as one coherent learning objective.

Other candidate groups require the same judgment. The four Arabic definition tasks for consistent, inconsistent, independent, and dependent systems share one classification anchor and evidence `0927ab12-24a9-4658-b238-d04b54419a91`, but their distinct terms may need to remain separate when the curriculum assesses each definition independently. The implementation does not merge on chapter or anchor alone.

## Cache and persistence

The Catalog and compiler contract version is now `catalog-v11-task-family-variations`. Cache keys include this version, compiler prompt source, schemas, canonical identity source, variation normalization source, and finalizer source. A v10 Catalog cannot pass as v11. Cache files are not mass-deleted, so unrelated entries remain intact.

Completed saved exams are not rewritten. Existing persisted plans/questions remain readable through the compatibility path; new v11 generation records variation selection. No database migration is required because the added metadata is stored inside existing JSON payloads. Partial Catalog compilation still cannot reach planning, and completed zero-task batches, selective retry, completed-batch caching, `CATALOG_BATCH_MAX_ATTEMPTS`, and global provider concurrency behavior are unchanged.

## Tests

Focused tests cover objective grouping, method separation, identity stability across primary-angle changes, collision resistance through method identity, multiple/single/zero variations, deduplication and stable IDs, bounded numerical permissions, explicit and inherited evidence, invalid evidence isolation, Arabic metadata, task-level type unions, generator selection, incompatible-type rejection, effective evidence, Planner-level allocation, repeated task reuse, validator payloads, duplicate recovery, cache invalidation, persistence compatibility, and incomplete-Catalog gates.

- Catalog-focused tests: **38 passed**.
- Expanded Catalog/generator/validator/planner regression set: **42 passed** after contract updates.
- Compact Catalog and planner-focused tests: **75 passed**.
- Full offline suite: **829 passed, 4 skipped** in 93.22 seconds.

No paid model, live provider, benchmark, document processing, or textbook reprocessing was run.

## Files modified for this change

- `app/config.py`
- `app/llm/schemas.py`
- `app/online/assessment_catalog.py`
- `app/online/prompts.py`
- `app/online/exam_builder.py`
- `app/online/validator.py`
- `tests/test_assessment_catalog.py`
- `tests/test_compact_catalog.py`
- `tests/test_plan_review.py`
- `tests/test_planner_v2.py`
- `tests/test_question_validator.py`
- this report

The worktree already contained substantial unrelated uncommitted changes in several of these files. They were preserved; no commit, reset, checkout, push, deployment, infrastructure change, or model-routing change was performed.

## Remaining risks and limitations

Task grouping remains an educational judgment made by the compiler. Deterministic code can preserve and validate the declared boundaries, but it cannot prove that two differently worded objectives are pedagogically equivalent. Conservative core-method, relationship, restriction, and context identity fields reduce over-grouping; the prompt explicitly preserves independently taught methods.

Variation evidence IDs prove only that the references are in scope. Existing Validators continue to assess semantic grounding. A bad but in-scope evidence choice can still require normal generation validation and repair.

The saved 116-task Catalog demonstrates where the new representation can help, but measuring actual extraction quality requires a separately authorized compiler run. This implementation intentionally did not run one.
