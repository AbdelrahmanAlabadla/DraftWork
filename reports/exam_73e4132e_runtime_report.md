# DraftWork exam runtime report

**Job:** `73e4132e-8be4-4f93-98e6-00e7bc4a12a6`  
**Exam request:** two models, 23 questions per model (46 requested), Easy  
**Report basis:** persisted job/checkpoint/audit/request telemetry; no new provider requests were made.

## Executive result

The job ran from **17:58:59.222** to **18:15:34.974** (Asia/Dubai): **16 minutes 35.75 seconds** wall-clock time.

It completed as a **partial** exam: **44 accepted of 46 requested**. Model 1 produced 23/23; Model 2 produced 21/23. The omitted slots were `m2_mcq_7` and `m2_true_false_1` after bounded duplicate recovery was exhausted.

| Measure | Result |
|---|---:|
| Provider requests | 283 |
| Provider-request failures | 0 |
| Provider-reported errors | 0 |
| Input tokens | 1,579,174 |
| Output tokens | 314,090 |
| Estimated provider cost | **$0.29690** |
| Sum of individual request durations | 57m 43s |
| Actual elapsed job time | **16m 36s** |

The difference between cumulative request time and wall-clock time is expected: many requests ran concurrently.

## What happened between the agents

| Time | Stage | Result |
|---|---|---|
| 17:58:59 | Retrieval | Retrieved the selected source chunks and request context. |
| 17:59:00–18:01:21 | Assessment Catalog + planning | Catalog compiled, then Planner created both model plans. |
| 18:01:21–18:02:55 | Difficulty adjustment + plan repair | The pipeline assessed whether requested Easy difficulty could be reached; some assignments retained source-constrained Medium/Hard fallbacks. |
| 18:02:55–18:03:17 | Generation | Both model pipelines generated their initial candidates concurrently. |
| 18:03:17–18:05:48 | Initial validation | Five validators plus Duplicate V2 evaluated candidates. 39/46 passed first validation; 7 failed. |
| 18:05:48–18:15:34 | Targeted recovery | Only affected slots were repaired, regenerated, replaced, and revalidated in bounded rounds. |
| 18:15:34 | Final assembly | 44 valid questions were assembled; two invalid slots were omitted and recorded as a question shortage. |

### Initial quality outcome

| Initial outcome | Count |
|---|---:|
| Passed first validation | 39 |
| Failed first validation | 7 |
| Final accepted | 44 |
| Final omitted/invalid | 2 |

Initial validation failures recorded by the final telemetry were primarily **unclear or invalid question** findings (6) and one **wrong answer** finding. The later recovery work also detected source-evidence, duplicate, and difficulty constraints on changed candidates.

## Why recovery took time

Ten slots entered recovery at one point. The expensive part was not an unbounded fourth replacement; it was the valid combination of separate bounded routes:

- local content Repair for answer, question, options, task-alignment, or solution defects;
- same-task regeneration where the assignment remained valid;
- Catalog replacement where a distinct compatible task was available;
- Difficulty Adjustment for a changed assignment; and
- applicable targeted validation plus incremental Duplicate V2 after every material candidate change.

The most problematic source assignments required interpreting a graph/vertices while the bound source evidence had no usable graph or coordinate data. The contracts also prohibited inventing replacement numeric data. That makes a self-contained MCQ impossible without adding unsupported evidence, so the pipeline correctly rejected those candidates.

### Terminal slots

| Slot | Final status | Why it was omitted |
|---|---|---|
| `m2_mcq_7` | `duplicate_repair_budget_exhausted` | It duplicated another MCQ. Its replacement required graph vertices but the assigned evidence contained neither the graph nor coordinates; a local repair also changed the required student operation. |
| `m2_true_false_1` | `duplicate_repair_budget_exhausted` | Replacement/difficulty recovery could not produce a distinct, valid Easy-level instance within the permitted bounded routes. |

The replacement limit did apply: neither terminal slot received a fourth Catalog replacement. Some later actions were different allowed routes (for example targeted repair or difficulty processing), which is why the audit trail contains several repair/revalidation transitions.

## Calls, time, and cost by agent role

| Agent role | Calls | Input tokens | Output tokens | Request-duration sum | Estimated cost |
|---|---:|---:|---:|---:|---:|
| Assessment Catalog compiler | 10 | 72,388 | 66,073 | 9m 44s | $0.03961 |
| Generator | 73 | 403,623 | 73,237 | 13m 53s | $0.07037 |
| Difficulty Adjuster | 30 | 106,274 | 49,567 | 8m 39s | $0.03518 |
| Difficulty Validator | 30 | 152,362 | 39,920 | 7m 44s | $0.03520 |
| Grounding Validator | 30 | 202,790 | 19,591 | 3m 59s | $0.02714 |
| Task Alignment Validator | 29 | 228,712 | 14,066 | 3m 13s | $0.02700 |
| Answer Correctness Validator | 30 | 145,637 | 14,288 | 2m 59s | $0.01888 |
| Quantitative Validator | 19 | 89,919 | 12,165 | 2m 22s | $0.01318 |
| Catalog replacement selector | 4 | 79,705 | 5,441 | 1m 05s | $0.01069 |
| Duplicate V2 Judge | 14 | 38,730 | 8,323 | 1m 41s | $0.00803 |
| Repair agents combined | 13 | 56,445 | 11,113 | 2m 20s | $0.01120 |

All 283 persisted provider requests have status `success`. Therefore **failed provider calls = 0**. Question-quality rejections are different: they are Validator decisions on otherwise successfully completed provider calls.

## Catalog cache: why this run missed and what happens next

This job was a genuine cache miss:

| Cache field at Catalog preparation | Value |
|---|---|
| Final Catalog cache hit | `false` |
| Compiler cache hit | `false` |
| Compatible-seed cache hit | `false` |
| Effective Catalog contract | `catalog-v14-numerical-instance-policy` |

The miss is expected because the effective Catalog policy is now v14, while earlier work used a different v13 contract. A finalized v13 Catalog must not be treated as a final v14 Catalog: the policy changes task finalization and numerical-instance permissions.

The run compiled the v14 Catalog and wrote the finalized entry under cache key:

`6d238ce6b5d06e460d7197ca34e9885a991c4519afe5ae72deda7767d49377eb`

That cache file now exists locally. A later job can reuse it **without the ten compiler calls** only if all cache-relevant inputs match:

- same processed document identity and selected source-chunk content/membership;
- same subject and domain;
- same effective Catalog/Compiler versions, prompts, schemas, batching and finalization identity; and
- evidence remapping still succeeds.

So, yes: from now on, a request using the same matching document/retrieval context should obtain a Catalog cache hit. A different upload, selected-chunk set, policy/prompt/schema version, or incompatible evidence mapping should correctly miss rather than reuse unsafe evidence.

## Important conclusions

1. The first v14 run paid the Catalog compilation cost; that is why it made ten compiler calls and spent about $0.03961 there.
2. Recovery—not provider failure—drove the remaining cost and time. Every provider call completed successfully.
3. The job did not loop forever. It reached a bounded terminal partial result once two slots could not be safely recovered from the available source evidence and duplicate constraints.
4. The final exam contains 44 validated questions, but the requested 46-question count was not achieved. The shortage is explicit in the final job result and should be treated as a source/task-availability issue, not hidden or silently filled.

