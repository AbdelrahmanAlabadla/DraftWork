# DraftWork task reuse, diversity, and missing-question investigation

**Scope:** read-only review of the current local, uncommitted implementation and persisted benchmark `4e7a3194-3eea-49cc-8bde-8ff7708b91fd`. No provider calls, database writes, source edits, migrations, or test runs were performed.

## A. Confirmed root causes

The `1/4` Word Problem result requires three conditions, all confirmed by code and the saved audit.

1. **The benchmark had exactly one Word Problem-compatible catalog task.** The 48-task catalog had 32 MCQ-compatible, 24 equation-compatible, 17 definition-compatible, 17 true/false-compatible, 4 why-compatible, and **1 Word Problem-compatible** task. Its native difficulty was medium; the planner legitimately used its documented difficulty fallback for Easy. The persisted inventory for every Word Problem slot contains only `task_fedb2995e754f86a`, so no unused compatible task was overlooked.
2. **That one task is over-specified as a fixed textbook instance.** Its plan says `target=Find the numbers of singles and doubles matches in the tennis scenario`, `task_operation=solve_contextual_system_for_counts`, and `required_givens=[38 players, 13 matches, 4-player doubles, 2-player singles]`. `materialize_plan_item` copies `required_givens` verbatim to both `given_conditions` and `required_givens`. The Generator and General Validator then enforce them as mandatory. This contradicts the compiler prompt, which says worked-example values should not be required unless the exact instance is the narrow target.
3. **Exact-task reuse contains contradictory instructions.** Reuse is explicitly authorized when the catalog is exhausted, and the generator prompt asks for a different target, method, representation, direction, application, or numerical instance. But the same plan still fixes its target, work product, operation, and numerical givens. In this benchmark there is no remaining degree of freedom that satisfies both rules. The generation gate rejected Word Problem slots 2 and 3 before semantic content was considered because the same task ID was already accepted; the later recovery path repeatedly returned to the only task and ultimately used numeric fallback. General validation correctly rejected 16/44 and 17/50 as `task_drift`, because they violated that task's persisted required givens.

The missing-question failure is therefore not an LLM inability to solve the tennis math. It is a contract-design conflict: the code permits task reuse but represents a task as one immutable assessment instance.

## B. Benchmark inventory and actual allocation

| Question type | Catalog-compatible tasks | Native Easy | Requested | Allocation |
|---|---:|---:|---:|---|
| MCQ | 32 | 17 | 10 | 10 distinct task IDs |
| True/False | 17 | 13 | 5 | 5 distinct task IDs |
| Equation | 24 | 3 | 4 | 4 distinct task IDs |
| Word Problem | **1** | **0** | 4 | all four use `task_fedb2995e754f86a` |

The persisted `task_inventory` for Word Problem slots 1–4 reports `compatible_total_count=1`, `compatible_task_ids=[task_fedb2995e754f86a]`, and zero unused compatible IDs after the first allocation. So:

- Tasks already used by MCQ or Equation were not incorrectly excluded. None of them advertised Word Problem compatibility.
- The tennis task was not selected through an arbitrary ranking preference. It was the sole eligible task.
- The planner does prefer unused task IDs and coverage IDs first. In `assessment_catalog.allocate_catalog_plan` it uses bipartite matching for coverage uniqueness, then unused task IDs, and only then explicit reuse. Reuse is expected behavior when inventory is exhausted.

The selected task is materially narrower than a reusable task family should be:

| Stored field | Benchmark value | Reuse impact |
|---|---|---|
| Concept | `Solving a count model` | Appropriate family-level knowledge. |
| Relationship | totals jointly determine single/double match counts | Appropriate family-level relationship. |
| Operation | `solve_contextual_system_for_counts` | Potentially reusable. |
| Target / work product | numbers of singles and doubles in **the tennis scenario** | Ties every reuse to one target/output. |
| Required givens | 13, 38, 2, 4 | Makes illustrative totals immutable. |
| Reasoning requirements | use both totals; check both totals | Appropriate invariant method requirements. |

## C. Conflicting rules and locations

| Location | Current rule | Observed effect |
|---|---|---|
| `app/online/assessment_catalog.py:779-789` | Compiler prompt says a worked example should become a transferable task and `required_givens` should contain only source-required constants or constraints. | Correct intended policy; the catalog response violated it for tennis and finalization accepted it. |
| `app/online/assessment_catalog.py:451-489` | `task_id` includes operation, work product, target, reasoning mode, relationship, **required_givens**, and scenario family. `task_family_id` still includes operation, work product, target, relationship, and reasoning mode. | IDs describe a narrow task/instance, not a broad family. Changing target or required values changes identity. |
| `app/online/assessment_catalog.py:1235-1280` | `materialize_plan_item` copies every `required_givens` value as both required and given conditions. | No field distinguishes immutable laws from variable exercise parameters. |
| `app/online/assessment_catalog.py:1330-1604` | Allocator prefers unique coverage/task IDs but permits explicit task reuse once exhausted; reuse metadata demands different target/method/direction. | Correctly prevents a shortage at planning time, but creates an impossible plan for a fully fixed task. |
| `app/online/prompts.py:10-42, 877-985, 1054-1078` | Generator may use neutral constructed numbers but must preserve required/fixed givens, task operation, target, work product, and reasoning mode. Reused tasks must differ materially; numeric variation is an allowed final fallback. | Direct contradiction for tennis: `13/38` must both change to diversify and remain unchanged to satisfy the plan. |
| `app/online/exam_builder.py:455-474` | Same task ID is rejected only unless `is_authorized_intentional_practice_reuse` returns true. | The benchmark's persisted pre-generation plans contain the fields that the current predicate requires, and the current predicate returns `True` for that shape. Yet the benchmark logged repeated `semantic plan duplicate` rejections. This is a confirmed audit/code inconsistency, requiring a deterministic snapshot regression test before behavior is changed; it may reflect an execution/version/state transition not preserved in the final plan snapshot. |
| `app/online/exam_builder.py:556-710` | One type-specific call receives all pending slots; accepted outputs are fed into later attempts. It checks exact duplicate, semantic-plan duplicate, then near-duplicate. | The four Word Problems were in one call, but accepted slot 1 became a hard same-task gate for subsequent slots. The LLM did receive plan-level negative context, yet code rejected slots before evaluating meaningful instance diversity. |
| `app/online/validator.py:1298-1310` | General validator requires the planned concept, operation, target, work product, required givens, and evidence boundary. It does permit a different instance only unless task target and required givens explicitly demand the exact source instance. | Correct validator rule in isolation. Given the bad plan contract, its `task_drift` verdicts for 16/44 and 17/50 were correct. |
| `app/online/duplicate_pipeline.py:150-245, 870-1323` | After two replacement attempts, numeric fallback can rewrite question fields, then revalidate. | Fallback changes the question but does not derive a compatible **instance contract** nor update the plan. It produced mathematically valid 16/44 and 17/50 questions that necessarily failed the authoritative 13/38 plan. |
| `app/online/validator.py:2722-2814, 3318-3525` | `task_drift` becomes `replace_assignment`; a missing slot first regenerates the original assignment and only later tries a replacement candidate. | Appropriate role separation, but with a single compatible task the replacement inventory is structurally empty. It cannot invent an authorized instance variant. |
| `app/online/plan_review.py:203-246, 1474-1639` | Plan review treats shared task/family/coverage identity as a conflict signal and accepts a replacement only if at least two semantic fields differ. | It handles distinct catalog tasks. It has no way to create different assessment angles within one task family without changing the task identity itself. |

## D. Generator behavior and why the retries failed

### What it receives

`generate_exams_node` calls the Word Problem generator once with all four planned slots. `_generation_plan_items` attaches each slot's assigned evidence and plan metadata. `_attach_plan_review_context` adds compact prior reused-slot plans; the prompt formatter includes `task_reused`, `required_instance_distinctions`, `previous_task_plan_items`, and, when available, `previous_task_questions`.

This is a good basis for diversity, but the actual benchmark plans have identical target, operation, work product, scenario, and required values. The Generator is told both to follow these fields exactly and to produce a different target/method. It cannot satisfy both truthfully.

### What rejects output

For Word Problem slots, `_generate_type_from_plan` has three distinct gates:

1. Exact normalized text duplicate.
2. `_is_semantic_plan_duplicate`: a task-ID-level rejection for an unapproved same-task reuse.
3. Token-overlap near duplicate over stem, steps, and final answer.

The saved job warnings show slots 2 and 3 were rejected at gate 2 on all three initial attempts, then in two further two-slot recovery batches. This occurred before semantic comparison of differing unknowns, steps, or valid values. The benchmark audit therefore proves unnecessary repeated generation calls: the same planned slots were known to be blocked by plan identity and were retried unchanged. There is one important inconsistency: the final pre-generation audit snapshot has complete exhaustion-reuse fields, which the current authorization predicate accepts. Capture the exact per-attempt plan passed to this gate in an offline regression test; do not assume the final audit snapshot is the exact in-memory object used at rejection time.

### Recovery trace

1. Initial generation accepted two Word Problems and rejected slots 2 and 3 three times as `semantic plan duplicate`.
2. Missing-slot repair attempted normal regeneration of the exact same assignment; it retried three times and remained blocked.
3. One later recovery generated Word Problem slot 3 (the retained 13/38 instance); slot 2 stayed missing.
4. Duplicate recovery found no different Word Problem-compatible catalog task for slot 4. It did find the same tennis task for slot 1 as an authorized exhausted-catalog reuse and generated it.
5. On the next recovery round, no compatible replacement remained for slots 1 and 4. At third attempt, numeric fallback created 16/44 and 17/50 variants.
6. General validation marked both `task_drift`: their stored task demanded the tennis source instance's 13/38 givens. The retry limit was reached, so they were omitted.

The replacement selector does receive conflict edges, failed question context, active exam question context, prior candidate failures, blacklist, and candidate pool. It did not select a different task because the pool had none. The audit's `duplicate_repair_history` records two `no_compatible_catalog_replacement` outcomes and two `numeric_variation_generated` outcomes. This is inventory/contract failure, not lost selector context.

## E. Responsibility boundaries

| Component | Current responsibility | Not responsible for |
|---|---|---|
| Catalog compiler | Extracts atomic tasks and their contract from source chunks. | Creating final question wording or repairing calculations. |
| Planner / allocator | Matches requested slots to catalog tasks, tracks use, marks exhausted reuse. | Creating a new legal assessment instance inside a task. |
| Plan Review | Detects duplicate planned semantic contracts and substitutes another existing catalog task. | Relaxing immutable givens or inventing a new instance family. |
| Generator | Realizes a plan into a question, answer, steps, and certificate; receives reuse context. | Changing fixed target/givens without a plan-level authorization. |
| General Validator | Checks source grounding, task/operation/target/givens alignment, clarity, and task drift. | Authorizing a change from a fixed task into a new task. |
| Quantitative Validator | Checks mathematics, solution certificate, options, units, and calculations. | Establishing source/task legitimacy. |
| Field-level Repairer | Repairs fields of an existing question using a validator verdict. | Selecting a new catalog task or defining variation policy. |
| Catalog task replacement / duplicate recovery | Chooses another catalog item, regenerates it, then revalidates. | Creating a fresh task-family instance when catalog has no distinct task. |
| Duplicate Judge | Determines whether actual questions overlap after generation. | Choosing content or validating source/task adherence. |

## F. Required design: family, assessment angle, and instance

Keep source grounding, but represent three levels explicitly.

1. **Concept / task family:** source-backed knowledge and invariant constraints. For tennis: solve a two-equation count model with 2 participants per singles event and 4 per doubles event. For physics: distance/speed/time relationship and units.
2. **Assessment angle:** authorized target, unknown, operation, representation, and reasoning route. Examples: solve for counts; construct the system; verify a proposed count pair; derive time given distance and speed; derive distance given speed and time.
3. **Question instance:** self-contained scenario values, variables, computed answer, options, steps, and certificate.

`task_id` should continue to identify the source-backed atomic task/angle. `task_family_id` should be broadened to family-level identity and must not include target, work product, or variable example values. Add an explicit instance policy to the catalog/plan rather than infer it from strings:

- `invariant_givens`: laws, definitions, constants, units, real source facts, and hard physical/domain constraints.
- `variable_parameters`: values/scenario quantities that may change, including admissible ranges/relationships and integrality/non-negativity constraints.
- `allowed_assessment_angles`: permitted targets/operations/representations supported by the source.
- `reuse_policy`: `distinct_angle_required`, then `distinct_instance_allowed` if an angle is exhausted.

The tennis source would be modeled as:

- Invariants: singles uses 2, doubles uses 4; counts are nonnegative integers.
- Variable parameters: total matches and total players, constrained to admit an integer nonnegative solution.
- Angles: solve counts from totals; formulate the system; verify whether a proposed pair fits totals; compare/eliminate equations.

Then four valid questions can share a family without false source drift:

1. 13 matches/38 players; solve for singles and doubles (7, 6).
2. 17 matches/50 players; solve for singles and doubles (9, 8).
3. Given 15 matches/42 players, decide whether `(9 singles, 6 doubles)` satisfies both totals; show checks.
4. Given 12 matches/36 players, formulate and solve the system, using a table representation (6, 6).

For speed–distance–time, one family can authorize `d=vt` with variable speed/time/distance and angles `find_distance`, `find_time`, `find_speed`, `compare_rates`, or `verify_claim`. A 55 km/h for 5-minute distance question and a 20 km at 80 km/h time question share a relationship but are distinct instances and targets.

## G. Ordered implementation plan (do not implement before approval)

1. **Define the contract.** Extend the catalog task and plan schemas with `invariant_givens`, `variable_parameters`, `allowed_assessment_angles`, and `reuse_policy`. Preserve existing `required_givens` as a backward-compatible alias for invariants until catalog migration is complete. Risk: schema compatibility for old cached catalogs; invalidate or normalize old entries by catalog version.
2. **Repair compiler semantics.** Update compiler instructions and finalization validation to reject a general reusable task whose `required_givens` merely copy worked-example values. Require an explicit `exact_source_instance=true` reason if values must remain fixed. Regression tests: tennis-style examples, physical constants, real historical facts, worked examples, and formulas.
3. **Allocate an angle plus task/family.** Planner should reserve unused compatible task/angle first, then another permitted angle from an already-used family, then a distinct numeric instance from the same angle. It must not label a reused task as a semantic plan duplicate solely by task ID. Keep type compatibility and source boundaries.
4. **Make reuse generation deterministic and visible.** Add a plan-owned `instance_brief`: requested unknown, allowed variable values/constraints, selected angle, required distinctions, and prior instance summaries. Generator uses it and recalculates dependent answer/option/solution fields. Regression test two and four tennis instances generated in one call and in recovery.
5. **Align validation with the contract.** General Validator checks invariants and selected angle; it accepts legal variable-parameter changes. Quantitative Validator verifies the generated instance independently. `task_drift` remains correct for changing an invariant or unauthorized angle. Regression test accepts 17/50 only when parameter policy permits it and rejects it under exact-source policy.
6. **Change duplicate gates.** Replace task-ID-level early rejection with instance/angle comparison: reject same angle plus same givens/result/reasoning, not an authorized shared family. Retain semantic duplicate judge as the final cross-question safety check. Regression tests should prove wording-only changes fail and different unknown/steps/numbers pass where policy allows.
7. **Unify recovery.** Missing-slot recovery should request a new legal instance brief from the same family before declaring catalog replacement exhausted. Numeric fallback must generate and attach a new authorized plan instance before generation, never mutate question fields beneath an unchanged fixed plan. Persist attempted instance fingerprints and outcomes to prevent repeated calls.
8. **Measure with offline tests, then one authorized benchmark.** Add tests for no repeat call when the plan gate will deterministically reject; cache/recovery compatibility tests; task-family diversity tests; validator/duplicate contract tests. Run offline tests first. Only after approval, run one paid benchmark and compare 23/23 completion, Word Problem diversity, calls, cost, and latency.

## Compatibility and safeguards

- Existing exact-source tasks remain supported through `exact_source_instance` / invariant values; no scientific constants, source facts, or physical constraints become variable by default.
- Existing task IDs remain usable. New fields can be normalized when absent to a conservative exact-instance policy, then improved catalogs opt into parameterized reuse.
- Question type permissions and isolated evidence boundaries remain enforced.
- The final Duplicate Judge remains necessary because authorization of a family/instance does not prove that two generated questions are educationally distinct.

## Bottom-line assessment

The system already contains much of the intended reuse vocabulary—explicit exhaustion metadata, previous-question context, required distinctions, numeric-variation language, selector context, and revalidation. The benchmark failed because those mechanisms sit below an atomic task contract that encodes illustrative values and a specific output as immutable. The correct solution is to make instance variation an explicit, source-grounded plan artifact, then have planner, generator, validators, and recovery all enforce that same artifact.
