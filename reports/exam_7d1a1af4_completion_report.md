# DraftWork exam completion report

**Job:** `7d1a1af4-bebc-4090-a529-aa3238537336`  
**Completed:** 7 November 2026, 18:41:03 (Asia/Dubai)  
**Scope:** Read-only analysis of the persisted job, audit events, final assembly state, and LLM usage/request records.

## Result at a glance

| Metric | Result |
| --- | ---: |
| Requested models | 2 |
| Questions requested per model | 23 |
| Questions requested total | 46 |
| Questions accepted | 44 |
| Questions omitted | 2 |
| Wall-clock runtime | 12m 12s |
| Logged provider-request events | 213 |
| Successful provider-request events | 211 |
| Failed provider-request events | 2 |
| Persisted LLM-usage records | 214 |
| Estimated LLM cost (usage ledger) | **US$0.262193** |
| Cumulative provider duration | 48m 17s |
| Catalog cache hit | No |
| Catalog compiler batches | 9 |

The exam completed successfully, but it is incomplete: it contains **44 of 46** requested questions. The omitted slots are `m1_mcq_2` and `m2_true_false_3`.

## Time line

| Time | Stage | What happened |
| --- | --- | --- |
| 18:28:51 | Job started | Two easy mathematics models, 23 slots each. |
| 18:28:52–18:30:29 | Catalog compilation | Catalog cache missed; nine Compiler calls rebuilt the Catalog. |
| 18:30:29 | Planning | Planner completed. |
| 18:30:30–18:31:59 | Difficulty adjustment | One plan issue was found and repaired: `m1_equation_1` needed the derivation/relationship reasoning mode. |
| 18:31:59–18:32:27 | Generation | Both models generated concurrently; model 1 completed in 20.5s and model 2 in 27.8s. |
| 18:32:27–18:34:59 | Initial validation | Six initial slot failures were identified across the two models. |
| 18:34:59–18:41:02 | Recovery | Six Repair → Revalidation cycles ran for unresolved slots. Independent slot recovery work overlapped, but every changed candidate returned to validation. |
| 18:41:02 | Assembly | 44 accepted questions assembled. |
| 18:41:03 | Shortage recorded | The two unrecoverable slots were explicitly recorded as a question shortage. |

## What happened between agents

The actual flow was:

```text
Catalog Compiler (9 calls)
  → Python Planner
  → Difficulty Adjuster / Validator
  → Generator (two models in parallel)
  → specialized Validators
      grounding, task alignment, answer correctness,
      quantitative, difficulty, Duplicate V2
  → Repair / replacement selection for failed slots
  → targeted revalidation
  → final assembly
```

Initial validation found **six** slot-level failures:

| Model | Type | Initial failures | Main recorded reasons |
| --- | --- | ---: | --- |
| 1 | MCQ | 1 | wrong answer |
| 1 | Equation | 1 | unclear/invalid question |
| 2 | MCQ | 1 | question and answer defect |
| 2 | True/False | 1 | unclear/invalid question |
| 2 | Word problem | 2 | unclear/invalid question |

Four of the six initially failing slots eventually produced accepted questions. Two did not.

## Why two questions are missing

### 1. `m1_mcq_2` — source-grounding failure

This MCQ concerned whether a strict-inequality boundary is included and whether it should be drawn dashed or solid.

The grounding validator found that the assigned source supports exclusion of the boundary, but **does not clearly support the claimed dashed-line convention**. The required correction was to use a fully source-supported boundary description or provide evidence for the dashed convention.

Recovery history:

1. Targeted repair was attempted three times.
2. Each attempt returned `VALIDATION_BLOCKED` rather than a valid revised candidate.
3. The fallback replacement route then ran out of unused/reusable source-grounded candidates.
4. The slot ended as `terminal_failure` with `no_valid_source_grounded_replacement`.

This omission was therefore a safety decision: the system did **not** insert a weakly grounded question just to reach 46.

### 2. `m2_true_false_3` — unsupported variation plus difficulty conflict

The source only supported evaluating a general statement about an unspecified curve. The recovery candidates changed that into narrower or more technical claims, including a nonlinear/polynomial example and an isolated point at the origin.

Those candidates failed because they:

- added mathematical givens not supported by the assigned source;
- narrowed the original claim, creating a wrong variation;
- produced an ambiguous truth value because “curve” was not defined sufficiently; and
- exceeded the requested easy difficulty while the assignment was marked hard.

Recovery history:

1. Regenerated original assignment: rejected as unsupported and mismatched difficulty.
2. Another regeneration: rejected as a wrong/narrowed variation with an ambiguous answer.
3. Targeted follow-up repair: rejected as unsupported and too difficult.
4. Replacement fallback found no unused/reusable grounded task.

It ended as `terminal_failure`, so assembly omitted it rather than accepting an unsupported or misleading true/false question.

## Why the Repair/Validation cycle was visible

The repeated cycle was not Catalog regeneration and not an unbounded loop. The graph intentionally does:

```text
failed validation → bounded repair/replacement → targeted revalidation →
either accept, try another permitted recovery, or mark terminal
```

This job performed seven validation passes and six Repair stages. The longest recovery work was caused by repeated attempts on the two slots above, particularly source-grounding and difficulty conflicts. Recovery stopped once its independent bounded routes were exhausted.

## Provider calls and estimated cost

The durable usage ledger recorded **214 usage records**, totalling **US$0.262193 estimated cost**. The request-event ledger recorded **213 request events**: 211 successful and two failed. The one-record difference is an observability mismatch between the usage and request-event tables, not evidence of an additional visible provider request.

| Agent / operation | Usage records | Estimated cost |
| --- | ---: | ---: |
| Generator | 55 | $0.056890 |
| Assessment Catalog Compiler | 9 | $0.041359 |
| Catalog replacement selector | 14 | $0.037135 |
| Difficulty Adjuster | 19 | $0.027109 |
| Difficulty Validator | 22 | $0.023005 |
| Grounding Validator | 22 | $0.021500 |
| Task Alignment Validator | 22 | $0.020816 |
| Answer Correctness Validator | 22 | $0.011316 |
| Quantitative Validator | 16 | $0.011795 |
| Duplicate Validator | 7 | $0.005108 |
| Repair agents combined | 6 | $0.006160 |
| **Total** | **214** | **$0.262193** |

The two provider-request failures were both from `catalog_replacement_selector`. They were structured-output responses that hit `max_output_tokens` while returning long replacement explanations. They were not network/provider outages, and the selector later made successful calls.

## Other findings

1. **Catalog compilation was expensive because of the cache miss.** It took about 1m 37s and made nine Compiler calls. This was the source-manifest/order cache miss previously identified, not repeated compilation inside this job.
2. **Initial validation was the largest wall-clock stage** at about 2m 23s. It ran model validation concurrently, but the overall stage waited for final exam-level validation/duplicate processing.
3. **Recovery used about 6m 03s of wall time.** Several independent slot recoveries did overlap; the stage duration was determined by the slowest unresolved slot in each wave.
4. **The final output is internally safe but some telemetry is misleading.** In the persisted evaluation summary, individual slot outcomes still show the two omitted slots as “accepted,” while the authoritative `accepted_slot_ids` list has 44 entries and correctly excludes `m1_mcq_2` and `m2_true_false_3`. The shortage record and final assembly are authoritative.
5. **No Catalog retry loop occurred.** Catalog compilation happened once in this job. The visible repetition was six bounded validation/recovery rounds.

## Final status

The job is marked `completed` with no top-level error. It produced two model exams with **44 safe, validated questions** and intentionally omitted two questions whose recovery candidates could not meet source-grounding, answer clarity, and requested-difficulty constraints.

