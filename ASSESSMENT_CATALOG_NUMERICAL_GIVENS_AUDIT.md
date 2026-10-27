# Assessment Catalog Numerical Givens Audit

## Executive conclusion

Catalog task `task_b76c4d06b6beb27a` violates the explicit current Compiler contract and is internally contradictory.

Applicable classifications:

- **INVALID_FIXED_NUMERICAL_GIVENS** — 60 AED and 17% are ordinary exercise parameters but were placed in `required_givens`.
- **INCORRECT_EXACT_SOURCE_FLAG** — the source is a worked/exercise instance of a transferable percentage operation; the stored object marks `exact_source_instance=true` without evidence that reproducing that exact exercise is itself the assessment target.
- **MISSING_NUMERICAL_VARIATION_METADATA** at the finalized task level — the raw semantics recognize variable parameters and derived type variations say numbers may change, but finalization clears `allowed_variations` because of the incorrect exact-instance flag.

This is not merely a downstream Generator failure. The contradictory Catalog contract made exact repetition the safest grounded behavior, while replacement metadata simultaneously demanded a different numerical instance.

## Existing Compiler contract

The live compiler prompt in `app/online/assessment_catalog.py:1519-1564` is explicit:

> “Unless the exact instance is itself the assessment target, express its transferable operation and target.”

It further states:

> “required_givens contains only invariant laws … constraints, or source-specific values that every valid instance must preserve”

> “variable_givens describes illustrative quantities or parameters that a future question may choose and change”

> “exact_source_instance is true only when assessing the exact worked example is itself the task.”

Most decisively, line 1556 says:

> “Ordinary textbook exercise numbers belong in variable_givens, not required_givens.”

The prompt permits `vary_numbers=true` only when changing the named givens preserves the relationship and constraints (`1532-1538`). It also prohibits unsupported numerical freedom and requires all fixed numerical givens to be present in cited evidence (`1553-1564`). These are balanced rules; they do not make every number variable.

## Schema and finalizer behavior

`TaskVariation.numeric_values_may_change` defaults to false (`app/llm/schemas.py:54`). Catalog tasks separately carry `variable_givens` and `exact_source_instance` (`app/llm/schemas.py:144-146`).

The finalizer at `app/online/assessment_catalog.py:1678-1698` trusts the compiler's exact-instance flag. If true, it sets `allowed_variations` to an empty list. It preserves `variable_givens` unchanged. It does not reject the contradictory combination:

- fixed numeric strings in `required_givens`;
- matching parameter names in `variable_givens`;
- `exact_source_instance=true`; and
- a variation that says the numbers may change.

For task merging, exactness becomes true only if both merged forms are true (`app/online/assessment_catalog.py:1641-1644`), but that does not repair a single incorrectly classified task.

The existing test `test_reusable_task_contract_separates_invariants_from_exercise_parameters` at `tests/test_assessment_catalog.py:535-561` demonstrates the intended model: structural constraints stay required, numerical totals are variable, and `exact_source_instance` is false. There is no equivalent negative test rejecting the dinner-tip contradiction.

## Stored dinner-tip task

The compatible cached Catalog is `catalog-v12-grouped-type-variations`, cache key `835f51022a2653ddeab87a5eafe26435ae7a55fd5806776142cf73348be6b8dc`. Its recorded compiler usage is `source=compatible_finalized_catalog`, `provider_calls=0` for the reference execution.

The stored object contains:

| Field | Value |
|---|---|
| `task_id` | `task_b76c4d06b6beb27a` |
| concept | percentage and total cost |
| operation | `calculate_total_with_tip` |
| target | calculate dinner cost after adding percentage tip |
| `required_givens` | bill value 60 AED; tip 17% |
| `variable_givens` | bill value; tip percentage |
| `exact_source_instance` | `true` |
| `allowed_variations` | empty after finalization |
| evidence | `c752dc57-845c-5042-9dd7-96b47e78e2eb` |
| allowed types | word problem, MCQ |

The plan's deterministic `permitted_variations` contains one word-problem and one MCQ representation. Both repeat 60/17 in `givens`, yet both set `numeric_values_may_change=true`. This comes from primary-variation synthesis: `numeric_values_may_change` is derived from whether `variable_givens` is nonempty (`app/online/assessment_catalog.py:688`).

Thus the same object simultaneously says:

1. preserve 60 and 17 as required;
2. those two quantities are variable;
3. the exact source instance is the task;
4. numerical values may change; and
5. no explicit variations are allowed.

## Source comparison

The cited chunk is titled “حساب التكلفة الإجمالية شاملة الإكرامية” and contains a short-answer textbook exercise: a family of four has a 60 AED dinner bill, leaves a 17% tip, and must calculate the total. The chunk supports the operation, context, and original example values.

Nothing in the retrieved text says that 60 or 17 is a mathematical constant, factual truth, mandatory curriculum value, or invariant constraint. The educational operation remains valid for other reasonable bill amounts and tip percentages. The source therefore supports treating these as illustrative parameters, subject to values that produce an answer appropriate for the requested difficulty.

**CONFIRMED:** the Compiler, rather than Planner, introduced the contradiction. Planner materialization copies `required_givens`, `variable_givens`, and `exact_source_instance` from the finalized task (`app/online/assessment_catalog.py:2265-2295`). It did not convert variable values into fixed values.

## Why the task became fixed

The most supported reconstruction is:

1. Compiler correctly noticed that bill value and tip rate are variable categories.
2. It also copied the concrete exercise values into `required_givens`.
3. It incorrectly treated the presence of a concrete textbook exercise as evidence that the exact instance itself was the task.
4. Finalizer trusted that flag and removed `allowed_variations`.
5. Primary type variations were later synthesized for MCQ/word-problem compatibility and inherited `numeric_values_may_change=true` from nonempty `variable_givens`, creating contradictory downstream metadata.

The raw pre-finalizer model response is not separately available in the forensic package, so the exact model rationale is **UNKNOWN**. The finalized contradiction and responsible deterministic transformations are confirmed.

## Contract verdict

### INVALID_FIXED_NUMERICAL_GIVENS — confirmed

The values are ordinary textbook exercise numbers, directly covered by the rule at `assessment_catalog.py:1556`. They are not invariants that every valid tip-calculation task must preserve.

### INCORRECT_EXACT_SOURCE_FLAG — confirmed

The target is a transferable operation, and the object itself identifies both numeric quantities as variable. No source evidence makes reproduction of the exact exercise the assessment target.

### MISSING_NUMERICAL_VARIATION_METADATA — confirmed at finalized-task level

The task has no `allowed_variations` because finalization clears them for exact instances. Although synthesized permitted representations state that numbers may change, there is no coherent bounded numerical contract describing permissible values, constraints, or difficulty range.

### CONTRACT_AMBIGUITY — not the primary finding

The prompt contract is sufficiently clear. There is a smaller schema ambiguity because contradictory fields are structurally legal and the finalizer does not enforce cross-field consistency.

## Sample of other numerical tasks

The same Catalog contains both correct and problematic patterns.

| Area/task | Required givens | Variable givens | Exact | Assessment |
|---|---|---|---:|---|
| Linear system by table, `task_342e3f526523862e` | two linear equations | coefficients/constants and table x-values | false | Correct reusable abstraction |
| Three-variable system, `task_5fc251153209bb23` | three linear equations | coefficients/constants | false | Correct reusable abstraction |
| Graph inequalities, `task_68d239e77d07c94e` | boundary semantics | coefficients/bounds | false | Correct; supports varied instances |
| Study-time inequalities, `task_af86e4a774884039` | nonnegative hours | subject limits and totals | false | Correct structural split |
| Vote percentages, `task_38b603086b9902e5` | 55%, 1541 votes, 40% | B votes and percentages | true | Same contradiction pattern |
| Job-hours equations, `task_525b76b99cd8d847` | 350 hours, wage rates, total income | only hours per job | true | Likely exact exercise; needs semantic review |
| Painting constraints, `task_8039807f1bd1f07c` | 20 days, rates, 45 buildings | output counts | false | Fixed example values despite reusable flag; contract inconsistency |
| Photo cost equations, `task_645060a9e20b31be` | 3 AED, 20 AED shipping, 5 AED | photo count | false | Fixed pricing may be scenario facts; valid only if task intentionally preserves pricing |
| Area-20 construction, `task_76365722fef7977f` | area 20 and fourth quadrant | none | true | Plausibly valid exact construction target |

This is **not isolated**. Several tasks mix fixed exercise numbers with reusable/exact metadata inconsistently. It is also not universal: many linear-system and inequality tasks correctly separate structural requirements from changeable coefficients.

The sample shows why automatic numeric stripping would be wrong. “Area exactly 20” may define the requested construction; store prices may be scenario facts for a particular modeling task; mathematical constants and feasibility constraints must remain fixed. Classification requires the relationship between value and assessment target.

## Enforcement defects

The current prompt is adequate, but deterministic enforcement is insufficient:

1. No finalizer check detects the same semantic parameter in both required and variable givens.
2. No rule rejects ordinary numeric literals in required givens when the task is reusable and the source does not make them invariant.
3. No consistency rule rejects `exact_source_instance=true` together with changeable numerical metadata.
4. Clearing variations for exact instances hides, rather than reports, conflicting compiler output.
5. Synthesized primary variations infer numeric freedom solely from nonempty `variable_givens`, even when exactness says the opposite.

## Recommended contract and enforcement

Retain the existing distinction and make it machine-checkable:

- **Canonical task:** operation, target, work product, reasoning mode, invariant constraints, and evidence scope.
- **Source example:** original values and scenario as evidence, stored separately from invariants.
- **Required structural givens:** laws, constants, factual scenario constraints, exact requested result constraints, and feasibility conditions.
- **Variable parameters:** named quantities that may be instantiated, with type/range/integrality/unit constraints where known.
- **Exact source instance:** true only when the exact instance is the pedagogical target, not merely because the source contains an exercise.
- **Permitted variation:** evidence-backed changes to representation, target, method, context, or numerical parameters.

### Required tests

1. Tip example: 60/17 become source-example values and variable parameters; exact flag false.
2. Contradiction: reject or quarantine a task whose same quantities are required and variable.
3. Exactness: reject exact=true plus `numeric_values_may_change=true` unless an explicit field explains which unrelated parameters may change.
4. Exact construction: preserve “area 20 in quadrant IV” when that exact target is what the exercise assesses.
5. Physical/factual constants: preserve values whose alteration changes the supported fact or governing relationship.
6. Finalizer: emit a diagnostic instead of silently clearing contradictory variations.
7. Resume/cache: version the strengthened contract so incompatible Catalogs are not silently reused.

## Combined recommendation

Fix Catalog invariant/parameter enforcement first because it determines whether later agents can produce distinct grounded instances at all. In parallel, add authoritative assignment/content revisions and coordinated reservations described in `SHARED_EXAM_STATE_ARCHITECTURE_AUDIT.md`. Upstream Catalog correction prevents impossible variation requests; shared state then ensures Planner, Repair, and Duplicate Replacement coordinate valid reusable tasks consistently.

No fixed task-use cap is needed. A reusable task may appear multiple times when each active question has a supported, materially distinct assessment instance.

## Evidence limitations

The cached finalized Catalog, plan materializations, source chunk, and downstream transitions are available. The raw pre-finalizer compiler response and model rationale for setting `exact_source_instance` are not separately persisted, so that subjective rationale cannot be recovered. The broader sample is diagnostic rather than an exhaustive semantic audit of every Catalog task.
