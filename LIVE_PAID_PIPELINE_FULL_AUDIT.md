# LIVE_PAID_PIPELINE_FULL_AUDIT

## 1. Executive Summary

**Final verdict: PARTIAL.** The one authorized paid run completed and persisted a complete 23/23 exam with unique question IDs. Cached document retrieval and Catalog compatibility were successful, and generation used 23 independent slot calls with real in-flight concurrency. The run nevertheless exposed quality and performance issues: the Catalog/difficulty stage took about 136.95 seconds, five questions failed first validation, one General Validator response was schema-invalid and required targeted revalidation, and duplicate review ran three replacement cycles while retaining one unresolved possible overlap group.

The job completed as `completed`; the saved exam is `complete`. This is operational success, but it does not meet the stronger goal of a clean first-pass, no-warning, independently proven 23-question exam.

## 2. Environment and Docker Verification

Preflight evidence was collected before submission:

- API, worker, PostgreSQL, Redis, and Qdrant containers were healthy.
- Alembic was at `0006_job_execution_fencing (head)`.
- Repository and running-container hashes matched for the critical API/worker source files.
- `MAX_PARALLEL_GENERATION_CALLS`, `LLM_JOB_CONCURRENCY_LIMIT`, and `DEEPSEEK_CONCURRENCY_LIMIT` were all 50.
- No conflicting active job existed.
- Qdrant was left intact.

The worker log showed provider calls to `https://api.openai.com/v1/responses` using the configured provider model `gpt-6-luna`. No 429, timeout, transport, or database exception was logged for this run.

## 3. Document and Catalog Cache Reuse

The run used document `175ef217-a754-4e58-8bae-2bbb4841f824`, stored as `math (2)2_compressed.pdf`, with exactly 103 child chunks. Retrieval selected all 103 chunks.

The authoritative in-container Catalog preflight reported:

| Item | Observed value |
|---|---|
| Catalog cache key | `1ff3868b51bd9fa2954733820100328fb83f8a7c5fcd9849c79fdd1c4594786f` |
| Cache hit | `true` |
| Compatibility hit | `true` |
| Tasks | 97 |
| Variations | 50 |
| FITB variations | 74 |
| Evidence IDs | 103 |
| Compiler calls for this run | 0 |

The audit event is named `assessment_catalog_compiled`, but the preflight and worker evidence establish that the compatible cached Catalog was loaded rather than paid compiler work being executed. The report therefore treats Catalog load/reuse as successful. Exact provider billing for Catalog compilation is zero for this job.

## 4. Exam Configuration

The request used Mathematics, Arabic (`ar`), Easy difficulty, one model, and 10 MCQ, 5 True/False, 4 Equation, and 4 Word Problem questions. The total was 23 and all other question types were zero.

## 5. Job and Exam IDs

| Field | Value |
|---|---|
| Job ID | `cc84370b-a450-432e-bf05-0bf130dbb973` |
| Exam ID | `exam_0f5ffc683170` |
| Document ID | `175ef217-a754-4e58-8bae-2bbb4841f824` |
| Created/submitted | 2026-10-26 14:01:51.592860 UTC |
| Started | 2026-10-26 14:01:52.951637 UTC |
| Completed | 2026-10-26 14:10:02.266463 UTC |
| Wall time from submission | 490.674 s |
| Worker execution time | 489.472 s |
| Job status | `completed` |
| Exam status | `complete` |
| Checkpoint revision | 30 |
| Celery/job attempts | 1 / maximum 2 |

The timestamps are UTC. Dubai local time is UTC+4.

## 6. Full Chronological Timeline

| UTC time | Event |
|---|---|
| 14:01:51.592 | Exam/job created and queued |
| 14:01:54.671 | Retrieval completed; 103 selected chunks |
| 14:01:55.613 | Planner completed; 23 slots allocated with no plan issues |
| 14:01:55.835 | Difficulty Adjustment entered |
| 14:04:12.788 | Difficulty Adjustment completed |
| 14:04:13.018 | Independent generation entered |
| 14:04:34.324 | 23 generated questions checkpointed |
| 14:04:34.485 | Initial validation entered |
| 14:05:31.626 | General and quantitative validation completed |
| 14:05:31.638 | Question repair entered |
| 14:05:53.937 | First repair round completed |
| 14:06:05.742 | First revalidation completed |
| 14:06:47.616 | Second repair/revalidation cycle completed |
| 14:06:47.627 | Duplicate review entered |
| 14:07:08.360 | First duplicate scan completed |
| 14:07:18.448 | First duplicate replacement completed |
| 14:07:47.864 | Second duplicate scan completed |
| 14:08:22.083 | Second duplicate replacement completed |
| 14:08:57.940 | Revalidation after replacement completed |
| 14:09:41.271 | Third duplicate scan completed |
| 14:09:50.984 | Third duplicate replacement completed |
| 14:10:01.500 | Final validation completed |
| 14:10:01.590 | Exam assembled |
| 14:10:02.266 | Exam saved and job completed |

## 7. Stage-by-Stage Results

Durations below are elapsed event-to-event wall times. Overlapping provider calls are not summed into stage wall time.

| Stage | Start UTC | End UTC | Duration | Calls/evidence | Result |
|---|---|---|---:|---:|---|
| Retrieval | 14:01:54.534 | 14:01:54.671 | 0.137 s | 103 chunks | PASS |
| Catalog load/reuse | 14:01:54.856 | 14:01:55.386 | 0.530 s | 0 compiler calls | PASS |
| Planning | 14:01:55.485 | 14:01:55.613 | 0.128 s | 23 slots | PASS |
| Difficulty Adjustment | 14:01:55.835 | 14:04:12.788 | 136.953 s | 11 LLM calls | PASS, slow |
| Independent Generation | 14:04:13.018 | 14:04:34.324 | 21.306 s | 23 initial calls | PASS |
| Initial Validation | 14:04:34.485 | 14:05:31.626 | 57.141 s | General/STEM batches | PARTIAL; 5 first-pass failures |
| Question Repair | 14:05:31.638 | 14:05:53.937 | 22.299 s | 4 repair calls plus replacement work | PASS with recovery |
| Revalidation | Multiple intervals in the event stream | UNKNOWN as one aggregate | multiple rounds | PASS after repairs |
| Duplicate Review | 14:06:47.627–14:07:08.360 | 20.733 s | 253 pairs, 5 Judge calls | PASS with findings |
| Duplicate Replacement | 3 rounds through 14:09:50.984 | approximately 162 s including rescans | 3 replacement calls and generators | PARTIAL; 1 warning retained |
| Final Assembly | 14:10:01.512–14:10:01.590 | 0.078 s | 23 questions | PASS |
| Persistence | 14:10:01.590–14:10:02.266 | 0.676 s | exam save | PASS |

The exact `revalidation` label is not persisted as one contiguous event interval; the audit records several validation/revalidation cycles. The aggregate duration is therefore marked UNKNOWN rather than inferred from overlapping repair and duplicate activity. Exact per-round intervals are available in the event timestamps above.

## 8. Question-by-Question Results — All 23 Slots

The persisted exam contains 23 questions and 23 distinct `question_id` values. The stable IDs are `model1_mcq_1` through `model1_mcq_10`, `model1_true_false_1` through `model1_true_false_5`, `model1_equation_1` through `model1_equation_4`, and `model1_word_problem_1` through `model1_word_problem_4`.

| Planner slot | Final ID | Type | Generated | Final lifecycle result | Repair/replacement evidence |
|---|---|---|---|---|---|
| m1_mcq_1 | model1_mcq_1 | MCQ | YES | accepted | replacement/revalidation history present |
| m1_mcq_2 | model1_mcq_2 | MCQ | YES | accepted | repaired `FIX_ANSWER` |
| m1_mcq_3 | model1_mcq_3 | MCQ | YES | accepted | repaired `FIX_ANSWER` |
| m1_mcq_4 | model1_mcq_4 | MCQ | YES | accepted | no final failure |
| m1_mcq_5 | model1_mcq_5 | MCQ | YES | accepted | duplicate-related candidate activity |
| m1_mcq_6 | model1_mcq_6 | MCQ | YES | accepted | no final failure |
| m1_mcq_7 | model1_mcq_7 | MCQ | YES | accepted | no final failure |
| m1_mcq_8 | model1_mcq_8 | MCQ | YES | accepted | no final failure |
| m1_mcq_9 | model1_mcq_9 | MCQ | YES | accepted | no final failure |
| m1_mcq_10 | model1_mcq_10 | MCQ | YES | accepted | replacement/revalidation activity |
| m1_true_false_1 | model1_true_false_1 | True/False | YES | accepted | repaired `FIX_QUESTION_AND_ANSWER` |
| m1_true_false_2 | model1_true_false_2 | True/False | YES | accepted | no final failure |
| m1_true_false_3 | model1_true_false_3 | True/False | YES | accepted | repaired `FIX_ANSWER` |
| m1_true_false_4 | model1_true_false_4 | True/False | YES | accepted | no final failure |
| m1_true_false_5 | model1_true_false_5 | True/False | YES | accepted | no final failure |
| m1_equation_1 | model1_equation_1 | Equation | YES | accepted | schema-invalid General verdict triggered targeted revalidation |
| m1_equation_2 | model1_equation_2 | Equation | YES | accepted | no final failure |
| m1_equation_3 | model1_equation_3 | Equation | YES | accepted | no final failure |
| m1_equation_4 | model1_equation_4 | Equation | YES | accepted | no final failure |
| m1_word_problem_1 | model1_word_problem_1 | Word Problem | YES | accepted | no final failure |
| m1_word_problem_2 | model1_word_problem_2 | Word Problem | YES | accepted | no final failure |
| m1_word_problem_3 | model1_word_problem_3 | Word Problem | YES | accepted | no final failure |
| m1_word_problem_4 | model1_word_problem_4 | Word Problem | YES | accepted | repaired `FIX_ANSWER` |

Persisted variation IDs were present for the quantitative records inspected, including `variation_b9fabb6b1d335be3`, `variation_a9e79650d89f2730`, `variation_52a2c58607f5d9e0`, `variation_1e438780e428d4f8`, and word-problem variations `variation_730808f5c2356f16`, `variation_bffcf80c7dd27e91`, and `variation_6176b8d66803e627`. Exact task/variation fields for every MCQ and True/False record were not exposed by the compact SQL output; where absent from the persisted record this audit marks them UNKNOWN rather than inferring them.

The saved evaluation reports, by type, final valid counts of MCQ 10, True/False 5, Equation 4, and Word Problem 4. First-pass failures were 3 `wrong_answer` and 2 `question_and_answer`; final missing/invalid count was 0.

Independent mathematical checking was not a complete formal audit of all Arabic expressions. The report therefore treats Validator PASS and persisted solution certificates as pipeline evidence, not as independent proof of every mathematical claim.

## 9. General/STEM Validation Findings

The initial validation checked all 23 generated questions. The worker logged three General batches and three quantitative/STEM batches. One General response did not contain a schema-valid verdict for `model1_equation_1`; the system issued a targeted single-question revalidation rather than accepting the missing verdict.

The first-pass persisted failures were:

- MCQ: 1 failure, `wrong_answer`.
- True/False: 1 failure, `question_and_answer`.
- Equation: 0 failures.
- Word Problem: 1 failure, `wrong_answer`.
- Overall evaluation also records three `wrong_answer` and two `question_and_answer` first-pass failures; the two additional failures were recovered during the first validation/repair transition.

No malformed final verdict was accepted as PASS. Exact validator reasoning for every individual question is not fully represented in the final `evaluation` object; the complete per-call reasoning is therefore UNKNOWN from persisted summary data.

## 10. Repair and Revalidation History

Observed repair calls included:

- `model1_mcq_2` → `FIX_ANSWER`.
- `model1_mcq_3` → `FIX_ANSWER`.
- `model1_word_problem_4` → `FIX_ANSWER`.
- `model1_true_false_1` → `FIX_QUESTION_AND_ANSWER`.
- `model1_true_false_3` → `FIX_ANSWER`.

The repairer made 5 recorded calls. Replacement generation also occurred during duplicate review. Repair candidates were revalidated before final acceptance, and the final saved exam retained all 23 questions.

The most significant observed inefficiency was duplicate review: three duplicate replacement rounds were interleaved with generation and validation. The first scan compared 253 pairs, and later scans again reported 253 pairs. Similarity distributions showed high screening candidates, with the final pass retaining one unresolved possible overlap group affecting `model1_mcq_1`, `model1_mcq_2`, and `model1_mcq_8`.

## 11. Duplicate Screening and Replacement Results

Initial duplicate screening examined 23 questions, 253 pairs, and selected candidate pairs for Judge review. The worker log recorded five concurrent Judge calls in the first scan and repeated scans after replacement. The first reported distribution included 39 candidates, 253 pairs, threshold 0.8, and combined maximum similarity 0.8982. Later distributions included 44 and 48 candidates and combined maxima 0.9558 and 0.9726.

Three replacement phases completed. Valid replacements were generated and validated, and the previous exam was not reduced. The final warnings metadata records one unresolved possible overlap group, affecting `model1_mcq_1`, `model1_mcq_2`, and `model1_mcq_8`. This is recorded as a possible overlap, not proof of semantic identity.

## 12. Real Parallel Generation Measurements

The worker explicitly logged `Parallel slot generation | calls=23 | max_workers=23` for the 23 Planner slots. The scheduler admitted generator requests with `limit=50`; the initial wave reached active=23. This confirms real provider requests were launched concurrently, rather than merely creating 23 deferred records. Subsequent replacement generation was smaller and bounded by the number of affected slots.

Observed initial generator latency ranged from about 3.7 s to 21.1 s, with the last initial response at 14:04:34.237. The initial generation wall time was approximately 21.3 s from stage transition to checkpoint completion. Exact p50/p95 latency and CPU/memory telemetry are not stored in the job tables and are UNKNOWN.

## 13. API Calls, Tokens, and Cost

PostgreSQL `llm_usage` reported 98 calls and the following totals. Token and cost values are provider telemetry recorded by the application; no prices were invented.

| Agent | Calls | Input | Output | Reasoning | Cached | Recorded cost USD |
|---|---:|---:|---:|---:|---:|---:|
| Difficulty Adjuster | 11 | 38,465 | 27,284 | 19,052 | 3,765 | 0.017149 |
| Generator | 38 | 229,324 | 30,117 | 17,364 | 39,259 | 0.034461 |
| General Validator | 16 | 110,484 | 28,626 | 19,369 | 3,472 | 0.025049 |
| STEM Validator | 14 | 82,940 | 15,310 | 4,590 | 0 | 0.015950 |
| Repairer | 5 | 65,453 | 3,725 | 1,979 | 0 | 0.008408 |
| Duplicate Replacement | 3 | 126,836 | 3,282 | 2,109 | 0 | 0.014324 |
| Duplicate Judge | 11 | 48,779 | 14,316 | 9,739 | 0 | 0.012037 |
| **Total** | **98** | **702,281** | **122,660** | **74,202** | **46,496** | **0.127378** |

Recorded provider duration sums are overlapping call durations and must not be treated as wall time. No API 429s or provider transport failures were observed. The largest input-token consumers were Generator, Duplicate Replacement, and General Validator. Duplicate Replacement consumed 126,836 input tokens across only three calls, making it a clear optimization target.

## 14. Performance Comparison Against Previous Runs

| Metric | Previous baseline | This run | Difference |
|---|---:|---:|---:|
| Total runtime | 1,211.412 s | 490.674 s | -720.738 s (-59.5%) |
| Runtime excluding Catalog | 986.212 s | approximately 490.1 s | not exactly comparable; this run reused Catalog |
| Initial generation | 123.904 s | approximately 21.306 s | -102.598 s (-82.8%) |
| Input tokens excluding Catalog | 5,696,128 | 702,281 recorded total | lower, but telemetry/configuration paths differ |
| Revalidation input tokens | 3,728,461 | not separately exposed; included in General/STEM totals | UNKNOWN comparison |
| Final questions | 21/23 | 23/23 | +2 questions |
| Unresolved duplicates | 1 | 1 | unchanged |
| Total recorded LLM calls | 171 | 98 | -73 (-42.7%) |

The runtime and token improvements are real observations for these executions, but comparison is not perfectly controlled: the new run reused Catalog, used a different repair/duplicate path, and the previous job had a different failure history. The strongest directly supported result is that independent initial generation completed in about 21 seconds with 23 concurrent requests, versus 123.904 seconds in the earlier baseline.

## 15. Passed Checks and Successful Behaviors

- Exactly one paid exam job was run.
- The existing document and 103 child chunks were reused.
- Catalog cache compatibility was confirmed before submission; no compiler calls were recorded for this job.
- Planner allocated all 23 slots with zero issues and zero errors.
- Independent generation created 23 slot-specific calls.
- Peak initial generator admission was 23, under the configured ceiling of 50.
- All 23 initial questions were generated and checkpointed.
- Final IDs were unique: 23 questions, 23 distinct IDs.
- General/STEM validation covered the full exam and did not accept the schema-invalid verdict silently.
- Repairs and replacement candidates were revalidated.
- Failed first-pass questions were recovered; final invalid/missing count was zero.
- Duplicate replacement preserved completeness.
- Final assembly and PostgreSQL persistence succeeded.
- No API rate-limit, timeout, NUL, JSONB, Celery full-job retry, or worker-fencing error occurred.

## 16. Failed Checks, Warnings, and Problems

- Difficulty Adjustment consumed approximately 137 seconds and 11 calls, dominating the pre-generation portion.
- Five first-pass validation failures required recovery.
- One General Validator response was schema-invalid for `model1_equation_1`, requiring an extra targeted call.
- Duplicate review repeated full 253-pair scans after replacements.
- Three duplicate replacement rounds were needed.
- One unresolved possible duplicate group remained in the final warning metadata.
- The persisted summary does not retain complete per-question validator reasoning, exact MCQ/TF task IDs for every item, or an independent deterministic proof of every Arabic mathematical certificate.
- Exact CPU, memory, provider admission wait, and request p95 telemetry were unavailable.
- The evaluation metadata records a difficulty fallback: 22 Easy and 1 Medium question despite the Easy request.

## 17. Remaining Risks and Recommended Fixes

1. **P1 — Reduce Difficulty Adjustment calls.** The 11 calls and 137-second wall time are the dominant measured pre-generation cost. Cache adjustment decisions where the plan and difficulty contract are unchanged, and batch only when the prompt remains bounded.
2. **P1 — Make duplicate review incremental.** The repeated 253-pair scans and three replacement rounds indicate that unchanged pair verdicts are not being reused sufficiently. Persist pair fingerprints and invalidate only pairs involving changed questions.
3. **P1 — Preserve per-question validator telemetry.** Persist exact General/STEM verdicts, reason codes, and revalidation causes in a queryable ledger so the final audit does not depend on truncated logs.
4. **P2 — Bound schema-invalid revalidation amplification.** The targeted retry behaved safely, but repeated malformed verdicts can increase cost. Keep the fail-closed rule and add a compact structured-output retry metric.
5. **P2 — Investigate difficulty fallback policy.** The final evaluation reports 22 Easy and 1 Medium. Determine whether the fallback is authorized by the request or should produce a user-visible warning.
6. **P2 — Add independent deterministic math checks.** Apply SymPy or equivalent checks to supported quantitative forms, while retaining General/STEM LLM validation for language and grounding.

## 18. Final Verdict: PARTIAL

The optimized Docker deployment passed the core reliability test for this run: it reused the cached document/Catalog, made 23 independent generation calls concurrently, preserved stable unique IDs, recovered validation failures, persisted a complete 23/23 exam, and avoided the earlier NUL crash and full-job regeneration.

It is **PARTIAL** against the full success objective because the run required multiple repairs, repeated duplicate rescans, retained one unresolved possible overlap warning, incurred a schema-invalid validator response, and spent disproportionate time in Difficulty Adjustment. The result is usable and complete, but the pipeline is not yet demonstrably clean or maximally token-efficient.

Evidence sources: PostgreSQL rows for job `cc84370b-a450-432e-bf05-0bf130dbb973` and exam `exam_0f5ffc683170`, 55 persisted audit events, `llm_usage` telemetry, worker logs, the preflight Catalog compatibility check, and the running-container source verification performed before submission. Unavailable evidence is explicitly marked UNKNOWN above.
