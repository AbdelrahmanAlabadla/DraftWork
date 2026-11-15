# DraftWork Exam Runtime, Cost, Validation and Recovery Report

## Exam identified

| Item | Value |
|---|---|
| Exam ID | `exam_bfe29ab579f1` |
| Job ID | `9e1b8b92-bb1d-4877-8a20-4cfa316156ef` |
| Date/time | 15 November 2026, 17:34:16–17:45:11 Asia/Dubai |
| Job status | Completed |
| Requested output | 2 models × 23 questions = 46 questions |
| Final output | 46/46 finalized, valid questions |
| Requested difficulty | Easy |
| Selected child chunks | 103 |
| Assessment Catalog cache hit | No |

## Executive summary

The exam completed successfully, but it was slow because it compiled a new 103-chunk Catalog, adjusted a large number of assignments for an easy exam, then performed eight first-pass validation corrections plus cross-model duplicate resolution.

The job ran for **10 minutes 54.9 seconds**. It made **284 recorded LLM requests** at an estimated total cost of **$0.335771636**.

There was one failed application-level request, but it was not a provider outage: the difficulty-adjuster returned a completed response that failed the local structured schema because it claimed a successful adjustment without changing an allowed field. There were **no confirmed provider/transport failures**.

The recovery cycle ultimately worked:

- All 46 requested slots reached finalized PASS state.
- `final_unresolved_slots` is empty.
- The repair manifest is empty.
- Eight first-pass failures were resolved.
- Cross-model duplicate handling added three more recovery targets.
- One direct replacement attempt remained rejected, but its original question became valid after another cross-model conflict member was regenerated; therefore it did not leave a missing or invalid final slot.

## Cost, calls and tokens

### Total

| Metric | Value |
|---|---:|
| Recorded LLM requests | 284 |
| Successful requests | 283 |
| Failed requests | 1 |
| Application/provider retry count | 1 |
| Input tokens | 1,829,484 |
| Output tokens | 351,346 |
| Reasoning tokens | 189,987 |
| Estimated cost | **$0.335771636** |
| Sum of individual request durations | 3,797.136 s / 63m 17s |
| Actual elapsed job time | **654.934 s / 10m 54.9s** |

The summed request duration is much greater than elapsed time because Catalog compilation, generation, validation, and revalidation use parallel calls.

### Provider/model split

| Provider / model | Calls | Input | Output | Reasoning | Estimated cost |
|---|---:|---:|---:|---:|---:|
| OpenAI `gpt-6-luna` | 282 | 1,812,893 | 350,757 | 189,987 | $0.330162800 |
| DeepSeek `deepseek-flash` | 2 | 16,591 | 589 | 0 | $0.005608836 |

### Major call groups

| Stage / role | Calls | Estimated cost | Sum request time | Notes |
|---|---:|---:|---:|---|
| Assessment Catalog compiler | 18 | $0.066801480 | 18m 14.6s | 18 parent-isolated batches for 103 selected chunks; parallel wall time 85.6s |
| Generator | 62 | $0.063312530 | 10m 52.2s | Includes initial generation and recovery/replacement generation |
| Difficulty adjuster, successful | 22 | $0.028961450 | 7m 7.0s | Initial planning plus later adjustment/recovery work |
| Difficulty adjuster, schema-failed | 1 | $0.000741750 | 10.6s | Completed provider response but invalid local schema result |
| Duplicate validator | 40 | $0.020808900 | 4m 57.3s | Within-model candidate gates and cross-model checks |
| Grounding validator | 27 | $0.029992300 | 4m 55.9s | Initial validation and revalidation |
| Task-alignment validator | 26 | $0.033865400 | 4m 31.3s | Highest validator cost because assigned evidence/contracts are large |
| Difficulty validator | 26 | $0.025107600 | 4m 27.7s | Initial validation and revalidation |
| Answer-correctness validator | 26 | $0.018975450 | 2m 48.8s | Initial validation and revalidation |
| Quantitative validator | 22 | $0.017480840 | 2m 11.9s | Quantitative slots and revalidation |
| Recovery-decision calls | 5 | $0.012371700 | 54.1s | Task-alignment recovery routing |
| Specialized repair agents | 8 | $0.017352236 | 2m 5.8s | Question, solution, options, answer, task-alignment and difficulty repair |

## Wall-clock timeline

| Window | Wall time | What happened |
|---|---:|---|
| 17:34:16–17:35:45 | ~89s | Request start, retrieval, Catalog compilation and finalization |
| 17:35:47–17:37:02 | 74.5s | Difficulty adjustment across eligible slots |
| 17:37:03 onward | — | Initial question generation began |
| 17:37:32–17:45:10 | 458.6s / 7m 38.6s | Validation, repairs, replacements, revalidations and duplicate checks overlap in this window |
| 17:39:10–17:44:52 | 341.5s / 5m 41.5s | Revalidation calls alone |
| 17:45:11 | — | Finalization completed |

The Catalog itself was not the primary wall-clock bottleneck despite 18 calls: those parent-section batches ran in parallel. The long tail was validation/recovery.

## Failure accounting

### Provider/application failures

| Type | Count | Detail |
|---|---:|---|
| Provider/transport failures | 0 | All provider responses completed successfully |
| Local structured-output failure | 1 | Difficulty adjuster returned `adjustment_succeeded=true` but did not alter requirements, transformations or givens, violating `PlanDifficultyAdjustmentBatch` validation |
| Generator structural warnings | 3 | Two objective-bundle slot warnings and one incompatible selected variation for `m2_equation_1`; recovery later filled/validated the affected slot |

### First-pass validation

| Measure | Count |
|---|---:|
| First-pass generated questions | 46 |
| First-pass PASS | 38 |
| First-pass failures | 8 |
| First-pass pass rate | 82.6% |
| First-pass failure rate | 17.4% |
| Final PASS | 46 |

First-pass failures by recorded evaluation category:

| Category | Count |
|---|---:|
| Unclear/invalid question | 6 |
| Wrong answer | 1 |
| Question and answer mismatch | 1 |

Model 1 had 3 initial failures; Model 2 had 5. The telemetry merge fix worked in this run: both models have nonzero first-pass measurements.

## Recovery cycle: did it work?

Yes. The system did not silently accept invalid output. It staged candidates, ran revalidation, rejected failed candidates, changed strategy where necessary, and finalized only complete, valid slots.

### Recovery totals

| Measure | Count |
|---|---:|
| Unique slots sent into content/recovery telemetry | 11 |
| Revalidation/recovery history entries | 18 |
| Accepted candidate outcomes | 10 |
| Rejected candidate outcomes | 8 |
| Final unresolved slots | 0 |
| Final missing questions | 0 |
| Final invalid questions | 0 |

The 11 recovery targets consist of the eight initial validation failures plus three cross-model duplicate-driven targets. This is why the recovery count is higher than the eight first-pass failures.

### Main recovery paths

| Slot(s) | First attempted strategy | What happened | Final outcome |
|---|---|---|---|
| `m1_mcq_1` | Regenerate | New candidate contradicted strict-inequality boundary representation | Switched task on attempt 2; PASS |
| `m1_mcq_7` | Switch task | Replacement used wrong system givens and remained medium | Different task on attempt 2; PASS |
| `m1_mcq_8` | Switch task | Candidate asked maximum while selected variation authorized minimization | Different task on attempt 2; PASS |
| `m1_word_problem_3` | Switch task | Candidate copied the source investment example and remained medium | Different task on attempt 2; PASS |
| `m2_equation_1` | Targeted repair | Still failed selected-variation operation and easy difficulty | Regenerated on attempt 2; PASS |
| `m2_equation_2`, `m2_mcq_7` | Targeted repair | Corrected candidate passed | PASS on attempt 1 |
| `m2_word_problem_4` | Switch task | Candidate passed | PASS on attempt 1 |
| `m2_mcq_1` | Switch task | Two replacement candidates failed the pre-commit duplicate gate | Third replacement passed |
| `m2_true_false_2` | Cross-model regeneration | Regenerated distinct instance passed | PASS |
| `m2_true_false_4` | Cross-model switch task | Replacement candidate still duplicated another question | That candidate stayed rejected; conflict was resolved when the related cross-model item changed, and this slot finalized PASS |

### Important telemetry detail

Some history rows show `revalidation_result=PASS` together with `new_failure_code=duplicate_question`. This means the ordinary content validators passed, but the subsequent authoritative candidate duplicate gate rejected the staged candidate. It is not a contradiction in the final result; it records two separate checks.

## Why validation and repair were still slow

### 1. The request is large and strict

The job requested 46 questions across two models using 103 source chunks. Every candidate is checked for grounding, task alignment, answer correctness, difficulty, and quantitative correctness when applicable. Those roles carry the assigned evidence and selected variation contract, so their prompts are large.

For example, the slowest revalidation task-alignment call took 29.2 seconds and carried 33,856 input tokens for four repaired slots.

### 2. Eight initial quality failures were not superficial

The recorded defects included wrong givens, wrong optimization direction, source-example copying, contradiction with source evidence, wrong variation operation, answer mismatch, and difficulty mismatch. These require different recovery actions; a simple wording patch would have been unsafe.

### 3. Revalidation repeats the full relevant validator closure

Recovery is deliberately transactional. A changed candidate is not committed after one repair call: it needs the applicable validators again, then the duplicate gate. The job recorded 54 revalidation calls across 11 revalidation groups.

### 4. Duplicate checking remains expensive during recovery

There were 40 duplicate-validator calls. The repaired-candidate duplicate gate and cross-model comparison prevented duplicate questions from entering the final exam, but `m2_mcq_1` needed three task-replacement attempts because its first two alternatives were still duplicates.

The duplicate-cache preservation fix was present, but it cannot eliminate checks involving changed candidates or newly staged replacement content. The report cannot prove cache hit/miss counts because that metric was not persisted in this job's audit record.

### 5. Easy-difficulty demand caused substantial adjustment and fallback work

The request demanded easy difficulty, while the selected source had many medium-native tasks:

- 36 final questions were easy.
- 10 final questions were medium fallback.

The job made 32 recorded difficulty-adjustment attempts/history entries. Several source-supported tasks could not safely be simplified to easy without changing source constraints, so the system used accepted medium fallbacks rather than inventing unsupported simplifications.

## What improved compared with the earlier run

- Generator events now carry slot/task/variation/attempt attribution for later calls, which makes recovery cost traceable.
- Cross-model replacements are labeled separately in repair telemetry.
- Both models retained first-pass validation statistics; Model 1 is no longer incorrectly reported as zero.
- The final result was complete, with no unresolved slots and no partial-exam acceptance.

## Remaining concerns

1. **First-pass quality is still below the target.** Eight of 46 questions failed first validation, and the desired 0–2 repair target was not met.
2. **Task/variation suitability for easy difficulty is still constrained.** Ten medium fallbacks were accepted.
3. **Recovery successfully protects quality but adds latency.** This run spent about 7m 39s in the validation/recovery window.
4. **Duplicate alternatives remain weak for some heavily used types.** `m2_mcq_1` needed three assignment attempts, and a cross-model True/False replacement initially duplicated another candidate.
5. **Audit evidence does not persist duplicate cache hit/miss statistics**, so the actual performance benefit of the new cache-preservation implementation cannot yet be quantified from this run.

## Bottom line

The cycle worked correctly in the safety sense: it rejected invalid candidates, escalated strategies when needed, resolved cross-model duplication, and produced a complete 46-question exam. It was slow because the job had a large new Catalog, a strict easy-difficulty request, eight genuine first-pass failures, three duplicate-driven recoveries, repeated validator closure for every candidate, and 40 duplicate-validation calls.

The strongest next performance opportunity is not relaxing validation; it is improving first-pass task/variation suitability—especially easy-level MCQs, equation tasks, word problems, and duplicate-distinct replacement options—so fewer candidates enter the recovery loop.
