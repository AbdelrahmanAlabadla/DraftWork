# Numerical Instance, Revalidation, Duplicate-Cache, and Grounding Fixes

Date: 2026-11-15
Scope: `exam_bfe29ab579f1` failure classes (8 first-pass failures, 54 revalidation calls, 40 duplicate-validator calls, false `contradicted` on `m1_mcq_1`).

## 1. Confirmed root causes

### 1.1 Validators treated illustrative numbers as authoritative

- `authoritative_givens()` returned the selected variation / `effective_givens` as
  "the numerical instance used by Generator and all Validators". The task-alignment
  instruction told the model to check the generated question against those givens,
  but did not say that an ordinary example instance may differ. A curriculum-style
  example (`x + y = 5`, `x - y = 1`) therefore read as a required value.
- The Catalog's per-variation `numeric_values_may_change` flag was exposed to
  validators even though the task-level authority (`numeric_variation_allowed`)
  could disagree. Repair instructions likewise said "satisfies … givens" without
  distinguishing invariant values from example values.
- Python did not compare numbers itself, but every LLM prompt framed the example
  as the reference instance. That is why `wrong_givens`, `wrong_variation`, and
  `unsupported_given` appeared for mathematically correct new instances.

### 1.2 Revalidation was not scoped to the changed revision

- Candidate revalidation set `force_full_revalidation=True` for every quantitative
  candidate. `validate_exam_multi` honored it by bypassing the job verdict cache for
  **all five roles**, even when a role's own payload (for example task alignment,
  which excludes answer fields) had not changed. Every repair cycle therefore
  reopened the full validator closure.
- The explicit `pending_revalidation_ids` path was never populated, so the only
  working reuse was the specialized role cache. That cache is correct, but the
  blanket bypass defeated it for the exact case the user reported: one answer/solution
  edit re-ran grounding, alignment, difficulty, and quantitative validators.

### 1.3 Candidate duplicate checks discarded their fact cache

- `candidate_duplicate_conflict()` built a fresh `duplicate_state` from the node's
  original state, called `scan_exam_duplicates`, used the returned groups, and then
  threw away `duplicate_fact_cache` and `duplicate_screen_fingerprints`.
- Each staged candidate therefore started from the same baseline. Every unchanged
  pair was re-judged once per candidate attempt. This is the direct mechanism behind
  the report's 94 verdicts on 35 unique pairs and one pair judged 15 times.

### 1.4 Grounding failures had no evidence obligation

- The grounding schema required only `reason`/`expected_fix`. A model could return
  `contradicted` for an invented ambiguity — exactly the `m1_mcq_1` case where a
  correct strict-inequality boundary description (dashed line) was rejected.
- There was no requirement to name the challenged claim or cite the assigned chunk,
  so an invented contradiction was indistinguishable from a real one.

## 2. Numerical equality rules removed or corrected

- Removed the implied "generated numbers must match the Catalog/example" contract
  from grounding, task alignment, quantitative, the legacy general validator, and
  the task-alignment repair instruction.
- Added one task-level contract (`numerical_instance_contract`) that decides
  `value_policy` only from explicit signals:
  - `exact_source_instance = true`
  - `numeric_values_fixed`
  - `fixed_numerical_data`
  - `protected_source_data`
  - dataset/measurement operations
- `effective_givens` are no longer automatically immutable. They are the planned
  instance and are labelled illustrative when the explicit contract allows variation.
- Immutable source values (exact instance, fixed data, protected data) remain
  authoritative and are listed in `immutable_values`; changing them still fails.

## 3. How validators now distinguish illustrative and immutable numbers

Every relevant payload carries the same three fields:

```json
{
  "value_policy": "illustrative",
  "illustrative_values": ["x + y = 5", "x - y = 1"],
  "immutable_values": []
}
```

- `illustrative`: different permitted values are valid; the validator checks the
  relationship, operation, target, and constraints instead of equality.
- `immutable`: the listed values (and `exact_source_instance`) must be preserved.
- The system prompts for grounding, task alignment, and quantitative now state that
  a different permitted instance is not by itself `wrong_givens`,
  `unsupported_given`, `missing_required_given`, `contradicted`, or `wrong_variation`.
- Generator and Repair prompts consume the same distinction, so a repair will not
  force the source example's numbers back into a valid new instance.

## 4. Files and functions changed

| File | Change |
|---|---|
| `app/online/models.py` | Added `numerical_instance_contract()` and value helper; separates illustrative/immutable values. |
| `app/online/multi_validator.py` | Added `NUMERICAL_VALUE_POLICY`, value contract in grounding/task-alignment/quantitative payloads; updated role instructions; removed blanket `force_full_revalidation`; added `_validate_grounding_evidence()`; bumped cache contract version. |
| `app/online/validator.py` | Added value fields to `_catalog_task_contract`; updated general validator prompt; shared candidate duplicate fact cache and screen fingerprint baseline; bumped validation ledger versions. |
| `app/online/prompts.py` | `given_conditions` and `effective_givens` explicitly illustrative unless immutable. |
| `app/online/multi_repair.py` | Task-alignment repair may not restore example numbers when `value_policy` is illustrative. |
| `app/llm/schemas.py` | `ValidationIssue.challenged_claim` for concrete grounding claims. |
| `tests/test_exam_bfe29ab579f1_regressions.py` | New offline regressions for all required cases. |
| `tests/test_validation_repair_difficulty_recovery.py` | Replaced blanket full-revalidation test with proportional reuse test. |
| `tests/test_draftwork_40q_readonly_e2e.py` | Mock grounding failures now provide `challenged_claim` and assigned evidence. |

## 5. How revalidation identifies changed revisions

- Each role has a fingerprint over exactly the inputs that can change its verdict:
  question/answer fields, assignment identity (`task_id`, `assignment_fingerprint`,
  `assignment_revision`, selected variation), contract/model version, and role
  instructions.
- Verdicts are cached per `(job, question, role, fingerprint)`. `revalidation_context`
  is excluded from the fingerprint, so history alone cannot force a re-run.
- A cached verdict is reused only when the role's fingerprint matches **and** every
  original issue owned by that role is covered by a resolved `original_issue_check`.
- Removing `force_full_revalidation` means an answer-only edit reruns quantitative,
  answer correctness, difficulty, and grounding (which consume answer fields) while
  task alignment reuses its still-valid verdict. An unchanged revision makes zero
  validator calls.
- Untouched verdicts are still retained by `_merge_validation_report`, and the
  validation ledger continues to cover legacy/non-specialized domains.

## 6. How duplicate verdicts are persisted and reused

- Duplicate facts remain keyed by normalized pair content plus model and prompt
  version (`_fact_cache_key`), never by pair ID alone.
- `validate_generated_questions` now keeps one `candidate_duplicate_fact_cache`
  for the whole validation node and passes it into every candidate gate.
- The candidate gate also receives the real `duplicate_screen_fingerprints`
  baseline, so a staged candidate only screens pairs that involve it (or a
  just-committed revision).
- Each gate merges the facts it learned back into the shared ledger, and the final
  `_integrate_duplicate_validation` scan inherits that ledger. An unchanged pair is
  therefore judged once per content revision, not once per repair attempt.
- Cross-model checks already had a persisted content fingerprint and fact cache;
  they continue to use the same source of truth.

## 7. How grounding failures are now supported by evidence

- `challenged_claim` names the exact student-visible claim.
- `source_evidence_ids` must cite assigned chunks for `contradicted`, `unsupported`,
  and `copied_source_example`.
- `insufficient_evidence` requires the challenged claim but not a citation (the
  finding is that support is absent).
- Code rejects a malformed grounding FAIL and retries only that row with a
  corrective instruction. If it stays malformed, the question remains blocked, not
  passed; missing evidence never becomes PASS.
- Prompt calibration now states that a correct consequence of the source concept is
  grounded even if not verbatim (the dashed-boundary case), and that an ambiguity or
  contradiction absent from the question must not be invented.

## 8. Tests and remaining limitations

### Test results

- New regression file: **14 passed**.
- Core affected suites (multi-validator, repair, grounding, question validator,
  graph variations, duplicate judge/pipeline, cross-model, performance, pipeline
  efficiency): **257 passed, 1 skipped**.
- Full offline suite: **1427 passed, 21 skipped, 4 failed**. The four failures are
  environment/timing tests, not behavior regressions:
  - `test_graph_pipeline_integration`: requires live `genexam-postgres`/`draftwork` rows.
  - `test_catalog_incomplete_recovery`: `threading.Barrier` timeout under full-suite load.
  - `test_recovery_v3` (two tests): peak in-flight count/timing under load.
  Each concurrency test passes or fails independently of these changes; the data-dependent
  test has no PostgreSQL data in this environment.

### Expected impact

- First-pass numerical false positives should drop; the target remains 0–2 repairs per
  46-question exam, but this needs a controlled live run to measure.
- Revalidation is now proportional: unchanged questions cost zero calls; a changed
  answer reruns only the roles that consume it.
- Candidate duplicate judging is paid once per content revision instead of once per
  attempt, directly addressing the 15-time pair.

### Remaining limitations

- No paid LLM exam was generated during implementation, per instruction. Prompt-level
  improvements still depend on model compliance and need one controlled live run.
- A provider that repeatedly omits `challenged_claim` will block the question rather
  than accept it. This is intentional (never invent evidence), but it is a new
  operational failure mode to monitor.
- Difficulty-designed `effective_givens` are formally illustrative unless the explicit
  immutable contract is set. That follows the requested rule but means the adjuster's
  numeric instance is no longer enforced by equality.
- The duplicate fact cache is process-local for a job; a worker restart starts cold
  (safe, just more calls).
