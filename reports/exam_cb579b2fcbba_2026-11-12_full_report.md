# DraftWork Exam Report — 12 November 2026, After 19:00

**Exam:** `exam_cb579b2fcbba`  
**Job:** `ba782f06-b76f-46ac-98f5-0a0a17e7cfa0`  
**Run window:** 19:00:12–19:09:12 Asia/Dubai  
**Method:** Read-only PostgreSQL artifact audit. No provider calls, writes, fixes, or cache changes.

## Executive summary

This is the correct post-19:00 run. It completed **partially: 43/46 accepted**, with three missing Model 2 questions:

- `m2_mcq_9`
- `m2_true_false_2`
- `m2_true_false_3`

The run did **not** run forever. It finished in approximately **9m 00.7s**. However, the recovery phase repeatedly cycled through duplicate checks, repair dispatch, revalidation, and fallback decisions. The cycle was bounded, but it was inefficient and exposed a lifecycle-state inconsistency.

All three missing slots were initially generated, then failed final recovery. Their recorded blocking findings were cross-model duplicate conflicts. The repair pipeline then reported `ROUTING_BLOCKED`; for `m2_true_false_2`, a regenerated candidate actually passed revalidation at attempt 4, but a later repair failure record left the slot terminally failed. That is a state/merge ordering problem in the persisted lifecycle, not a provider outage.

## Final exam outcome

| Metric | Result |
|---|---:|
| Requested | 46 |
| Generated | 46 |
| Accepted/finalized | 43 |
| Invalid/failed | 3 |
| Missing generated questions | 0 at generation level |
| Missing accepted questions | 3 |
| Final status | Partial/incomplete |
| Model 1 accepted | 23/23 |
| Model 2 accepted | 20/23 |
| Requested composition | 20 MCQ, 10 True/False, 8 Equation, 8 Word Problem |
| Recorded actual difficulty | 31 Easy, 12 Medium |

The shortage records were:

| Type | Requested | Generated | Missing slots |
|---|---:|---:|---|
| MCQ | 20 | 19 | `m2_mcq_9` |
| True/False | 10 | 8 | `m2_true_false_2`, `m2_true_false_3` |

The other question types reached their requested accepted counts.

## Time by stage

| Stage/operation | Provider calls | Provider duration sum | Recorded cost | Wall-clock position |
|---|---:|---:|---:|---|
| Difficulty adjustment | 13 | 5m 23s | $0.018912 | 19:00:32–19:06:48 calls overlapped other work |
| Generation | 65 | 12m 32s | $0.063847 | 19:01:30–19:08:32 |
| Standard validator batches | 143 | 21m 05s | $0.125233 | 19:02:10–19:08:40 |
| Duplicate V2 | 31 | 5m 07s | $0.014623 | 19:02:44–19:09:00 |
| Recovery Decision | 6 | 59s | $0.011777 | 19:02:58–19:09:10 |
| Repair agents | 20 | 3m 07s | $0.035610 | 19:03:12–19:07:28 |
| Catalog replacement selector | 4 | 1m 04s | $0.011614 | 19:06:01–19:08:24 |
| **Total** | **282** | **49m 16s** | **$0.281616** | **9m 00.7s wall time** |

The provider-duration sum is much larger than wall time because calls ran concurrently. It does not mean the user waited 49 minutes.

## Calls, failures, and cost

| Metric | Count/value |
|---|---:|
| Total recorded provider attempts | 282 |
| Successful provider attempts | 281 |
| Failed provider attempts | 1 |
| Failed-call category | `SCHEMA_VALIDATION_FAILED` |
| Logged retries | No retry event recorded for the failed provider attempt |
| Input tokens | 1,882,743 |
| Output tokens | 221,497 |
| Reasoning tokens | 127,046 |
| Recorded estimated cost | **$0.281615914** |

The one failed provider request was an Answer Repair request. This was not a network outage or rate-limit failure; it was a schema-validation failure in the returned/processed repair response.

### Agent call totals

| Agent/role | Calls | Success | Failed | Cost |
|---|---:|---:|---:|---:|
| Generator | 65 | 65 | 0 | $0.063847 |
| Task Alignment validator | 31 | 31 | 0 | $0.036292 |
| Grounding validator | 32 | 32 | 0 | $0.031228 |
| Difficulty validator | 29 | 29 | 0 | $0.025059 |
| Answer Correctness validator | 29 | 29 | 0 | $0.018233 |
| Quantitative validator | 22 | 22 | 0 | $0.014420 |
| Duplicate V2 validator | 31 | 31 | 0 | $0.014623 |
| Difficulty Adjuster | 13 | 13 | 0 | $0.018912 |
| Recovery Decision | 6 | 6 | 0 | $0.011777 |
| Catalog replacement selector | 4 | 4 | 0 | $0.011614 |
| Answer Repair | 5 | 4 | 1 | $0.013884 |
| Solution Repair | 5 | 5 | 0 | $0.007667 |
| Task Alignment Repair | 3 | 3 | 0 | $0.004129 |
| Difficulty Repair | 2 | 2 | 0 | $0.002825 |
| Duplicate Repair | 2 | 2 | 0 | $0.002301 |
| Question Repair | 2 | 2 | 0 | $0.002342 |
| Options Repair | 1 | 1 | 0 | $0.002463 |

## Recovery counts

These are slot-level categories; a slot can pass through more than one category.

| Recovery action | Count/slots | Outcome |
|---|---|---|
| Targeted Repair attempted | 4 failed Model 2 slots plus other repair activity | Several repair agents were invoked; three terminal slots ended `ROUTING_BLOCKED`. |
| Regeneration attempted | At least `m2_true_false_2`; recovery histories also contain regeneration paths | `m2_true_false_2` recorded a PASS candidate at attempt 4 before later state corruption/overwrite. |
| Different task/assignment switching | Recovery histories contain replacement/switch actions | Used to handle duplicate and copied/unsupported assignments. |
| Replacement selector calls | 4 | Expensive but did not produce accepted candidates for the three missing slots. |
| Terminal failures | 3 | `m2_mcq_9`, `m2_true_false_2`, `m2_true_false_3`. |

## What happened between the agents

```text
Difficulty adjustment
  → Two-model generation
  → Grounding / Task Alignment / Answer / Difficulty / Quantitative validation
  → Within/cross-model Duplicate V2
  → Recovery Decision
  → specialized Repair or regeneration/replacement
  → revalidation
  → Duplicate V2 again
  → local model readiness
  → cross-model duplicate completion
  → final assembly
```

The audit contains **41 model recovery-progress events**. Model 1 and Model 2 validated independently, then repeatedly emitted `duplicate_check`, `repairing`, and `revalidating` events. This confirms concurrent model activity, not a global deadlock.

The recovery phase lasted from the first validation/recovery activity around 19:02:05 until cross-model completion at 19:09:11. That is roughly **7m 06s**, explaining most of the 9-minute run.

## Missing slot 1 — `m2_mcq_9`

- Initial generated MCQ was a direct substitution question.
- Duplicate V2 found a cross-model conflict with `m1_mcq_4`.
- Conflict signals included shared task, same operation/concept, repeated numerical structure, and matching normalized formula/equation.
- Recovery Decision selected `REPAIR`, with fallbacks `REGENERATE → SWITCH_TASK`.
- Specialized Repair immediately returned `ROUTING_BLOCKED`.
- The final lifecycle state was `FAILED_TO_REPAIR` / `repair_failed`.

This was not an answer-correctness failure. The question was rejected because it duplicated another model’s assessment, and the selected Repair route could not perform the required substantive change.

## Missing slot 2 — `m2_true_false_2`

- Initial True/False candidate conflicted with `m1_mcq_9`.
- Duplicate V2 classified it as `cross_model_duplicate`; the conflict shared task identity and substantially overlapped the tested knowledge/operation.
- Three Repair attempts were recorded as `VALIDATION_BLOCKED` / `repair_provider_blocked`.
- A no-progress guard skipped an identical fourth Repair input.
- A regeneration using `task_6e2e41fd1c2332fd` then recorded **`revalidation_result: PASS`** at attempt 4.
- Later, another Repair record at attempt 5 marked the slot `ROUTING_BLOCKED` and finalization stored `FAILED_TO_REPAIR`.

This is the clearest lifecycle inconsistency in the run: a valid regenerated candidate was recorded, but a later failure update won in the final merged slot state. The persisted final question/validation objects show PASS-like data while `slot_lifecycle` is terminally failed. This indicates stale or out-of-order recovery-state merging rather than a bad provider response.

## Missing slot 3 — `m2_true_false_3`

- Initial True/False candidate conflicted with `m1_mcq_7`.
- Duplicate V2 found shared task/variation-group and the same conceptual target/operation.
- Recovery Decision selected `REPAIR`, with `REGENERATE → SWITCH_TASK` fallbacks.
- Specialized Repair returned `ROUTING_BLOCKED` on the first attempt.
- The slot ended `FAILED_TO_REPAIR` and was excluded from final assembly.

Again, this was primarily a duplicate/recovery-routing failure, not a provider availability failure.

## Was it an infinite loop?

No. Evidence against an infinite loop:

- The job terminated at 19:09:12.
- There were 41 recovery-progress events, not an unbounded stream.
- The lifecycle contains bounded attempt numbers and a no-progress guard.
- The finalizer explicitly emitted a shortage and saved the partial exam.

However, it behaved like an expensive cycle because the same unresolved duplicate class repeatedly passed through:

```text
Duplicate V2 → Repair decision → Repair blocked → Revalidation/fallback → Duplicate V2
```

The main inefficiencies were:

1. Duplicate conflicts were discovered after normal validation and sometimes again after replacement/revalidation.
2. Recovery Decision selected targeted Repair for conflicts that required substantive regeneration or task switching.
3. Repair routing returned `ROUTING_BLOCKED`, consuming recovery state without resolving the conflict.
4. `m2_true_false_2` continued after a recorded PASS candidate, indicating stale terminal-state merging.
5. Replacement selection and revalidation were repeated while the candidate remained in a conflict-prone assignment family.

## Overall diagnosis

The three-question shortfall was caused by a combination of:

- Cross-model duplicate conflicts in Model 2.
- Repair routes that could not make the required substantive changes.
- One schema-validation failure in an Answer Repair call.
- Excessive recovery/revalidation churn.
- A likely stale/out-of-order lifecycle merge for `m2_true_false_2`, where a later failure overwrote an earlier successful regeneration.

The models did generate 46 candidates, and most content validation succeeded. The dominant failure was not “the models cannot generate questions”; it was that the generated questions were not sufficiently distinct across models and the recovery execution/state handling did not consistently commit a valid fallback.

## Final conclusion

This 12 November run produced **43 accepted questions**, cost approximately **$0.281616**, took **9m 00.7s**, and made **282 provider attempts**. Three Model 2 slots were lost. Two were directly blocked by duplicate/recovery routing; one (`m2_true_false_2`) has evidence of a successful regenerated candidate that was later overwritten by a terminal failure state.

The most important evidence-backed issue is therefore **recovery state consistency after a candidate PASS**, followed by **using Repair for conflicts that require regeneration or task switching**. The run was bounded, but the recovery loop was too expensive and failed to preserve at least one apparently valid fallback.

