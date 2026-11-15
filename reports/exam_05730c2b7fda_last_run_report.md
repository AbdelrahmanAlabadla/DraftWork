# DraftWork last-run report — `exam_05730c2b7da`

**Job:** `0dadcf77-4f55-4c8f-8ef1-d37942500d7f`  
**Exam:** `exam_05730c2b7fda`  
**Completed:** 10 November 2026, 21:35:39 Asia/Dubai  
**Method:** Read-only analysis of PostgreSQL job, exam, usage, request-event, audit-event, lifecycle, validation, and recovery artifacts. No provider calls, database writes, or cache changes were made.

## Executive summary

The latest run completed without a top-level error but produced a **partial exam: 45 of 46 requested questions**. The only omitted slot was **`m1_true_false_1`**.

The omission was intentional safety behavior. All content validators except Duplicate V2 passed for that slot; Duplicate V2 found it excessively overlapped `m1_true_false_5`. Recovery first produced a replacement that passed, but a later duplicate pass still found a conflict. The final replacement search had no remaining eligible, grounded True/False task or authorized reusable assignment. The slot was therefore marked `FAILED_TO_REPAIR` rather than admitting a duplicate question.

This was **not** a provider outage:

- **230 provider requests**, all recorded as `success`.
- **0 provider-request failures** and **0 logged retries**.
- Estimated ledger cost: **US$0.278740**.
- Wall-clock job runtime: **11m 26s**.
- Sum of individual provider durations: **59m 20s**, which confirms substantial overlap/concurrency rather than 59 minutes of user-visible waiting.

## Request and outcome

| Metric | Result |
|---|---:|
| Models | 2 |
| Requested questions | 46 |
| Requested composition | 20 MCQ, 10 True/False, 8 Equation, 8 Word Problem |
| Requested difficulty | Easy |
| Generated candidates | 46 |
| Accepted/finalized questions | 45 |
| Failed terminal slots | 1 |
| Final exam status | Partial |
| Missing questions | 1 True/False |
| Missing slot | `m1_true_false_1` |
| Difficulty actually delivered | 29 Easy, 15 Medium, 1 Hard |

The system explicitly recorded a shortage warning rather than hiding it:

```json
{
  "question_type": "true_false",
  "requested_count": 10,
  "generated_count": 9,
  "missing_slot_ids": ["m1_true_false_1"]
}
```

## Time line

| Time (Dubai) | Stage | Elapsed since previous stage | What occurred |
|---|---|---:|---|
| 21:24:13 | Job began | — | Generation worker started. |
| 21:24:15–21:27:14 | Catalog compilation | 2m 59s | Assessment Catalog compiled; 11 compiler requests. |
| 21:27:15 | Planning | <1s | Two-model plan completed. |
| 21:27:15–21:28:33 | Difficulty adjustment | 1m 18s | 19 adjustment calls; many requested-Easy slots required constrained fallback handling. |
| 21:28:33–21:29:09 | Generation | 36s | Both model bundles generated. |
| 21:29:09–21:30:20 | Initial validation | 1m 11s | Multi-validator evaluation, including Duplicate V2. |
| 21:30:20–21:35:39 | Recovery and revalidation | 5m 19s | Four bounded Repair/Replacement waves and five total validation passes. |
| 21:35:39 | Final assembly | <1s | 45 accepted questions assembled; shortage persisted. |

**Wall time:** 11m 26s, from worker start at 21:24:13.949 to completion at 21:35:39.817.

## Cost, tokens, and calls

The cost below is the application’s recorded estimate, not an invoice reconciliation. Provider billing can differ because of provider-side pricing, cached-token treatment, or later billing adjustments.

| Operation / agent | Calls | Input tokens | Output tokens | Reasoning tokens | Provider-duration sum | Estimated cost |
|---|---:|---:|---:|---:|---:|---:|
| Generator | 55 | 336,012 | 59,725 | 34,213 | 12m 28s | $0.061702 |
| Assessment Catalog Compiler | 11 | 91,291 | 80,372 | 18,281 | 13m 56s | $0.047750 |
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

### Provider call reliability

| Metric | Result |
|---|---:|
| Request events | 230 |
| Successful requests | 230 |
| Failed requests | 0 |
| Logged retries | 0 |
| Operational-validator failures | 0 recorded |

There is no evidence that the missing question was caused by a timeout, rate limit, invalid JSON response, or provider error.

## What happened between agents

The observed pipeline was:

```text
Assessment Catalog Compiler
  → Python Planner
  → Difficulty Adjuster
  → Generator (two models)
  → Grounding / Task Alignment / Answer / Difficulty / Quantitative validators
  → Duplicate V2
  → bounded Repair, regeneration, or catalog Replacement
  → revalidation
  → final assembly
```

The job ran five `validation_completed` stages and four `repair_completed` stages. Recovery did not spin indefinitely: each unresolved slot followed bounded recovery actions, then the finalizer either accepted the new candidate or marked the slot terminal.

### Recovery ledger

| Slot | Recovery path | Result |
|---|---|---|
| `m1_mcq_10` | Targeted Repair failed once with `wrong_givens`; targeted follow-up Repair then passed. | Accepted on attempt 2. |
| `m1_mcq_2` | Catalog replacements on two attempts. | Accepted. |
| `m1_mcq_6` | Catalog replacement. | Accepted. |
| `m1_true_false_5` | Replacement failed `wrong_givens`; Task Alignment follow-up Repair restored the authoritative inequality and point. | Accepted on attempt 2. |
| `m1_word_problem_3` | Regenerated from original assignment. | Accepted. |
| `m2_word_problem_4` | Targeted Repair. | Accepted. |
| `m2_mcq_2` | An unsupported-graph attempt failed, a regeneration then failed `ambiguous_option`, followed by two catalog replacements. | Accepted on attempt 4. |
| `m1_true_false_1` | Catalog replacement initially passed; later Duplicate V2 still found overlap; no eligible replacement remained. | Terminal failure; omitted. |

## Why `m1_true_false_1` is missing

### Final validator outcome

The final retained finding was from **Duplicate V2**:

| Validator | Verdict |
|---|---|
| Grounding | Pass |
| Task Alignment | Pass |
| Answer Correctness | Pass |
| Difficulty | Pass |
| Quantitative | Pass |
| Duplicate V2 | Fail: `duplicate_question` |

Duplicate V2 paired `model1_true_false_1` with `model1_true_false_5` in `group_1`. Its prescribed remediation was an authorized distinct variation or a different Catalog task followed by regeneration.

### Exact recovery sequence

1. The original slot was an Easy True/False item from `task_a2c7cde59c169ef9`.
2. Recovery used a catalog replacement based on that task; it passed immediate revalidation.
3. A later Duplicate V2 pass found the replacement still overlapped `m1_true_false_5`.
4. The system advanced the assignment revision and regenerated the original task after the duplicate finding.
5. That regenerated candidate also remained duplicate.
6. Final inventory showed only two compatible True/False tasks for the relevant pool; both were already used. The alternate compatible task was excluded for this duplicate conflict. There were **zero unused, eligible, grounded replacements**.
7. The slot became `FAILED_TO_REPAIR` with terminal code `no_valid_source_grounded_replacement`.

This is a **candidate-capacity and duplicate-diversity problem**, not evidence that the Boolean answer was invalid or that the source was ungrounded. The system correctly refused to publish a duplicate merely to fill the count.

## What improved versus the preceding run

The immediately preceding completed run was job `0404eb80-8ab7-464a-9bfc-f5b02d550623`. This is an observational comparison, not a controlled A/B test: task selection and provider outputs differed.

| Metric | Previous run | Latest run | Change |
|---|---:|---:|---:|
| Accepted questions | 43/46 | 45/46 | +2 |
| Missing True/False slots | 3 | 1 | -2 |
| Provider calls | 247 | 230 | -17 |
| Failed provider requests | 2 | 0 | -2 |
| Estimated cost | $0.262268 | $0.278740 | +$0.016472 |
| Wall time | 7m 40s | 11m 26s | +3m 46s |
| Provider-duration sum | 43m 48s | 59m 20s | +15m 32s |

### Improvements observed

- Completion improved materially: 45 accepted rather than 43.
- The prior run had three missing True/False slots; this run had one.
- No request failed. In the prior run, two provider calls failed.
- Multiple difficult slots were recovered successfully, including a True/False `wrong_givens` failure and a multi-round MCQ ambiguity failure.

### Trade-offs observed

- The higher completion rate cost more time and about $0.0165 more estimated spend.
- Replacement selection was expensive: 8 calls consumed 218,752 input tokens and $0.027405.
- The job still delivered 16 questions above requested Easy difficulty: 15 Medium and 1 Hard. The fallback is surfaced in metadata, but this remains a product-quality limitation.

## What is still weak

1. **True/False inventory is too small for the requested diversity.** The terminal slot had only two compatible tasks in its final candidate inventory, both already used. That makes Duplicate V2 recovery brittle.
2. **Duplicate recovery can create a late collision.** `m1_true_false_1` passed an intermediate replacement validation, then failed a later duplicate comparison. This is safe, but inefficient and explains the apparent acceptance-then-failure transition.
3. **Replacement-selector context remains costly.** It was the third-highest input-token consumer despite only eight calls.
4. **Difficulty fidelity remains weak.** The request was Easy; the final mix included 15 Medium and 1 Hard questions. These may be allowed by source-constrained policy, but they should be prominently disclosed to the requester.
5. **Unsupported-graph selection still appeared during recovery.** `m2_mcq_2` first attempted an unsupported supplied-graph assignment before later recovery succeeded. This did not cause the final shortfall but spent a recovery attempt.
6. **Historical-versus-fresh verdict presentation needs care.** The terminal True/False finding is retained in its lifecycle for diagnostics. It must not be described as a new provider evaluation after the terminal state.

## Bottom line

The run was safer and more complete than the preceding run: **45 valid questions, no provider failures, and one explicitly documented shortage**. The remaining missing True/False question is explained by Duplicate V2 plus an exhausted eligible candidate pool—not bad provider output and not a generic Repair loop.

The highest-value remaining work is to increase or reuse distinct grounded True/False capacity under explicit diversity rules, prevent late duplicate collisions earlier in Replacement selection, and make the Easy-to-Medium/Hard fallback visible as a first-class exam outcome.
