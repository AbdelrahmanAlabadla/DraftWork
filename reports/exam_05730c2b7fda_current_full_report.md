# Current Generation Exam — Full Read-Only Report

**Exam:** `exam_05730c2b7fda`  
**Job:** `0dadcf77-4f55-4c8f-8ef1-d37942500d7f`  
**Completed:** 10 November 2026, 21:35:39 Asia/Dubai  
**Scope:** Read-only artifact review. No provider calls, database writes, code changes, or cache changes were made.

## Executive result

The run did **not** enter an unbounded/infinite loop. Recovery was bounded and eventually terminated the unresolved slot. The exam completed **partially: 45 of 46 questions**.

The missing slot was `m1_true_false_1`. Its content validators passed; the blocking failure was a within-model Duplicate V2 finding against `m1_true_false_5`. Replacement and regeneration attempts either remained too similar or exhausted the eligible grounded True/False assignment pool. The system correctly refused to publish a duplicate question.

There were no provider-call failures, no logged retries, and no evidence of a network/provider outage. The long wall time came from recovery and revalidation waves, not a literal infinite loop.

## Exam totals

| Metric | Result |
|---|---:|
| Requested | 46 |
| Generated candidates | 46 |
| Accepted/finalized | 45 |
| Missing/terminal | 1 |
| Completion | Partial (45/46) |
| Models | 2 |
| Requested composition | 20 MCQ, 10 True/False, 8 Equation, 8 Word Problem |
| Delivered difficulty | 29 Easy, 15 Medium, 1 Hard |
| Missing slot | `m1_true_false_1` |

## Time and stages

| Stage | Time | Calls/events | Notes |
|---|---:|---:|---|
| Catalog compilation | 2m 59s | 11 calls | Largest setup stage by wall time. |
| Planning | <1s | Python planning | Two-model plan completed. |
| Difficulty adjustment | 1m 18s | 19 calls | Many Easy requests needed source-constrained fallback handling. |
| Initial generation | 36s | 55 generator calls | Both model bundles generated. |
| Initial validation | 1m 11s | Validator/Duplicate calls | Five validator passes plus Duplicate V2 participation. |
| Recovery/revalidation | 5m 19s | 4 repair/recovery stages; 5 validation passes overall | Main source of perceived cycle time. |
| Final assembly | <1s | Finalizer | Assembled 45 accepted questions and retained shortage. |
| **Total wall time** | **11m 26s** | **230 provider requests** | 21:24:13.949–21:35:39.817 Dubai time. |

The sum of provider durations was **59m 20s**, versus 11m 26s wall time. This is concurrent work, not 59 minutes of sequential waiting.

## Cost and provider calls

Recorded application estimate: **US$0.278740**. This is not an invoice reconciliation.

| Agent/stage | Calls | Input tokens | Output tokens | Reasoning tokens | Provider-time sum | Estimated cost |
|---|---:|---:|---:|---:|---:|---:|
| Generator | 55 | 336,012 | 59,725 | 34,213 | 12m 28s | $0.061702 |
| Catalog Compiler | 11 | 91,291 | 80,372 | 18,281 | 13m 56s | $0.047750 |
| Replacement selector | 8 | 218,752 | 11,060 | 9,268 | 2m 24s | $0.027405 |
| Task Alignment validator | 25 | 223,180 | 17,075 | 10,711 | 6m 56s | $0.027150 |
| Difficulty Adjuster | 19 | 87,674 | 35,959 | 24,714 | 6m 34s | $0.025038 |
| Grounding validator | 25 | 169,429 | 18,801 | 12,080 | 5m 09s | $0.023028 |
| Difficulty validator | 24 | 111,859 | 23,124 | 12,889 | 4m 34s | $0.020526 |
| Answer Correctness validator | 24 | 106,199 | 11,374 | 5,727 | 2m 43s | $0.013855 |
| Quantitative validator | 17 | 85,280 | 8,772 | 3,717 | 1m 56s | $0.010989 |
| Duplicate V2 validator | 11 | 30,450 | 6,597 | 4,963 | 1m 34s | $0.006344 |
| Answer Repair | 3 | 18,786 | 531 | 0 | 5s | $0.006048 |
| Task Alignment Repair | 4 | 26,511 | 2,923 | 667 | 33s | $0.004112 |
| Solution Repair | 3 | 16,719 | 1,983 | 463 | 24s | $0.002663 |
| Options Repair | 1 | 5,492 | 433 | 0 | 3s | $0.002130 |
| **Total** | **230** | **1,527,634** | **278,729** | **137,693** | **59m 20s** | **$0.278740** |

### Failed calls

| Failure metric | Count |
|---|---:|
| Provider request failures | 0 |
| Logged retries | 0 |
| Operational validator/provider failures | 0 recorded |
| Terminal question failures | 1 |

The one terminal question failure was a correctness-protection outcome, not a failed provider request.

## Recovery counts by question/slot

These are slot-level recovery categories; one slot can appear in more than one category when it moved through a fallback sequence.

| Recovery category | Slots | Count | Detail |
|---|---|---:|---|
| Targeted content Repair | `m1_mcq_10`, `m1_true_false_5`, `m2_word_problem_4` | 3 | Local defects were repaired and accepted. |
| Regeneration on same assignment | `m1_word_problem_3`, `m2_mcq_2`, `m1_true_false_1` | 3 | Regeneration was used when a new instance was needed. |
| Different task/catalog replacement | `m1_mcq_2`, `m1_mcq_6`, `m2_mcq_2`, `m1_true_false_1` | 4 | Replacement candidates were evaluated under grounding/type/duplicate constraints. |
| Recovery requiring multiple rounds | `m2_mcq_2`, `m1_true_false_1` | 2 | These explain most of the late recovery time. |
| Terminal unresolved | `m1_true_false_1` | 1 | No eligible distinct grounded True/False assignment remained. |

The provider-level Repair calls were 11 total across Answer, Task Alignment, Solution, and Options Repair. They should not be interpreted as 11 different questions; several belonged to the same recovery paths.

## What happened between agents

```text
Catalog Compiler
  → Planner
  → Difficulty Adjuster
  → Question Generators (two models)
  → Grounding / Task Alignment / Answer / Difficulty / Quantitative validators
  → Duplicate V2
  → targeted Repair, regeneration, or Catalog replacement
  → revalidation
  → final Duplicate V2 checks
  → final assembly
```

Observed recovery paths:

- `m1_mcq_10`: targeted Repair failed once with `wrong_givens`, then a follow-up Repair corrected it and the slot passed.
- `m1_mcq_2`: two Catalog replacements, then accepted.
- `m1_mcq_6`: Catalog replacement, then accepted.
- `m1_true_false_5`: replacement produced `wrong_givens`; Task Alignment Repair restored the authoritative inequality and point, then it passed.
- `m1_word_problem_3`: regenerated from its original assignment, then accepted.
- `m2_word_problem_4`: targeted Repair, then accepted.
- `m2_mcq_2`: unsupported-graph attempt, regeneration with `ambiguous_option`, then two Catalog replacements; accepted on the fourth recovery attempt.
- `m1_true_false_1`: replacement initially passed immediate validation, later failed Duplicate V2, regeneration remained too similar, and no eligible replacement remained.

## Why the run felt stuck

It was not an infinite loop. The expensive pattern was:

1. A candidate could pass the five content validators but still fail a later Duplicate V2 comparison.
2. Recovery then had to regenerate or select another assignment.
3. Each changed candidate required revalidation and duplicate checking again.
4. The True/False candidate pool was small, so the selector had few legitimate alternatives.
5. Replacement selection itself consumed 8 calls and 218,752 input tokens.

The longest stage was Recovery/revalidation at **5m 19s**, after generation and initial validation had already completed. The repeated work was bounded, but the late duplicate discovery caused the appearance of a repair/revalidate cycle.

## Why `m1_true_false_1` was missing

Final retained findings:

| Validator | Final result |
|---|---|
| Grounding | Pass |
| Task Alignment | Pass |
| Answer Correctness | Pass |
| Difficulty | Pass |
| Quantitative | Pass |
| Duplicate V2 | Fail: `duplicate_question` |

Duplicate V2 identified excessive overlap with `m1_true_false_5` in the same model. The replacement candidate initially passed immediate validation but was later rejected when the broader within-model comparison ran.

The final source-grounded True/False inventory had no unused compatible assignment: the remaining compatible tasks were already used or excluded by the duplicate conflict. The terminal code was `no_valid_source_grounded_replacement`. Publishing the duplicate would have produced a nominal 46/46 exam with a quality violation, so the system retained a documented shortage instead.

## Is the model producing too many bad questions?

There is evidence of elevated recovery pressure, but not of provider instability:

- 46 candidates were generated and 45 were ultimately accepted.
- 0 provider calls failed.
- Several candidates had correctable defects (`wrong_givens`, `ambiguous_option`, unsupported graph assignment).
- The final missing slot was not rejected for bad arithmetic, grounding, or answer correctness; it was rejected for duplicate overlap and lack of an eligible alternative.

So the dominant problem in this run was **assignment diversity/candidate capacity plus late duplicate detection**, not a general inability of the models to generate valid questions.

## Comparison with preceding run

| Metric | Previous run | Current run | Change |
|---|---:|---:|---:|
| Accepted | 43/46 | 45/46 | +2 |
| Missing True/False slots | 3 | 1 | -2 |
| Provider calls | 247 | 230 | -17 |
| Failed provider calls | 2 | 0 | -2 |
| Estimated cost | $0.262268 | $0.278740 | +$0.016472 |
| Wall time | 7m 40s | 11m 26s | +3m 46s |

Completion improved, but the added recovery work increased elapsed time and cost.

## Remaining weaknesses

1. The True/False catalog pool is too small to guarantee distinct questions at the requested count.
2. Replacement screening does not always prevent a late Duplicate V2 collision.
3. Replacement selector prompts are expensive relative to their call count.
4. Easy difficulty fidelity remains imperfect: 15 Medium and 1 Hard question were delivered under an Easy request through source-constrained policy.
5. Unsupported graph assignments can consume recovery attempts before a compatible candidate is selected.

## Bottom line

The exam failed to reach 46/46 because one True/False slot exhausted its valid, distinct, source-grounded alternatives. It did not fail because of an infinite loop, provider outages, or failed API calls. The run produced **45 valid accepted questions**, incurred an estimated **$0.278740**, took **11m 26s wall time**, and used **230 successful provider calls**. The main time sink was bounded recovery/revalidation after late duplicate conflicts.

