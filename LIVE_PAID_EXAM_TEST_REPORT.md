# Live Paid Exam Test Report

## Result

The single authorized paid run did not produce a saved exam. It was stopped after repeated repair/revalidation cycles failed to converge. No second paid exam was started.

- Job ID: `dc0ae6f9-89be-4d1d-8d97-39f7136e3968`
- Exam ID: `exam_4d75fc56142a`
- Document ID: `175ef217-a754-4e58-8bae-2bbb4841f824`
- Document: `math (2)2_compressed.pdf`
- Requested: 10 MCQ, 5 True/False, 4 Equation, 4 Word Problem; total 23
- Final persisted exam: none
- Final job state at stop: `running`, stage `validating_exam`/`repairing_exam`, progress 95%, with `GENERATION_FAILED` retained from an earlier retry
- Test timezone: timestamps are UTC (`+00`); local Dubai time is UTC+04

## Source and cache reuse

The authoritative Docker PostgreSQL database contained 103 child chunks for the document. The request used all 103 persisted child IDs. Retrieval completed with `selected=103`.

The expected saved Catalog file was present in the mounted application data directory and contained the matching selected chunk set. However, the running Docker pipeline did not reuse it. The audit entered `compiling_assessment_catalog` at `11:08:17.114931Z`. Worker logs show 11 Catalog compiler usage records, including corrective retries. Therefore the cache-reuse requirement was not met in the live container.

The compilation also produced contract retries: `FITB evidence ID is outside this compiler batch`. This consumed paid calls before generation began.

## Timing

The job was created at `11:08:03.999839Z` and started at `11:08:15.345551Z`. The run was stopped at approximately `11:24:10Z`; PostgreSQL had not recorded `finished_at`.

| Stage | Observed interval / wall time | Result |
|---|---:|---|
| Queue wait | 11.35 s | Queued to worker start |
| Retrieval | about 1.45 s | 103 chunks selected |
| Catalog compilation | about 176.7 s | Unexpectedly executed; 11 paid calls |
| Planning | about 0.35 s after catalog completion | Completed |
| Difficulty Adjustment | about 107.7 s | 9 paid calls; one transient retry |
| Plan repair | about 5.5 s | Completed; one paid call |
| Initial independent generation | about 42.4 s | 23 generator admissions; persistence then failed |
| Built-in job retry generation | about 32.7 s | Recovered from persistence failure |
| Initial General/STEM validation | about 102 s | Completed; 23-question scope initially validated |
| Repair | about 7.3 s | 2 repair calls |
| Revalidation and duplicate replacement | continued for several rounds | Two replacement questions remained invalid; stopped |
| Final assembly/persistence | not reached | No saved exam |

The exact end-to-end wall duration is not available because the job had no completion timestamp and was manually revoked after it stopped making progress. From creation to stop it was approximately 16 minutes.

## Generation concurrency

The first generation pass created 23 independent scheduler admissions. Worker telemetry reached `active=23`, under the configured limit of 50. The calls were launched concurrently and completed out of order; the scheduler immediately admitted additional work when a call finished.

The optimized independent path was incomplete in the running container. Initial calls used type-specific schemas:

- `mcq_true_false_bundle`
- `equation_questions`
- `word_problem_questions`

Replacement generation also used `mcq_true_false_bundle`. Thus the observed behavior confirms 23 concurrent requests, but does not confirm one-question-per-call semantics for all supported types.

## LLM usage

Usage persisted for this job was:

| Operation | Calls | Input tokens | Output tokens | Reasoning tokens | Cached input |
|---|---:|---:|---:|---:|---:|
| Assessment compiler | 11 | 77,318 | 76,536 | 23,446 | 14,118 |
| Difficulty adjustment | 9 | 31,247 | 19,667 | 13,195 | 0 |
| Plan repairer | 1 | 2,256 | 271 | 88 | 0 |
| Generator | 65 | 360,524 | 48,277 | 27,519 | 104,598 |
| STEM validator | 14 | 132,727 | 39,587 | 11,001 | 49,476 |
| General validator | 19 | 161,326 | 53,149 | 38,528 | 74,553 |
| Repairer | 2 | 8,902 | 747 | 386 | 0 |
| Duplicate replacement | 5 | 358,744 | 4,941 | 2,908 | 0 |
| **Total** | **126** | **1,133,044** | **243,175** | **117,071** | **242,745** |

The previous live baseline had 171 calls and 5,696,128 input tokens excluding Catalog. This run's persisted usage is lower in raw input tokens, but the comparison is not valid as a quality improvement because this run did not finish, compiled Catalog unexpectedly, and used different execution paths.

## Validation, repair, and duplicate behavior

The first complete validation reached 23 generated questions. Subsequent logs show validation scopes of two replacement questions. Both remained invalid after repair/revalidation. Duplicate replacement was attempted with five recorded calls, followed by replacement generation and validation.

Observed provider behavior included one transient network failure during Difficulty Adjustment and one validator structured-output retry using `validator_missing_verdict`. No 429 response was observed in the captured worker logs.

The most serious persistence failure was:

```text
psycopg.errors.UntranslatableCharacter:
unsupported Unicode escape sequence
DETAIL: \\u0000 cannot be converted to text
```

The offending generated JSON contained a NUL escape in a mathematical payload. PostgreSQL rejected the JSONB write after generation had completed. The Celery task automatically retried the job once, which regenerated the exam and incurred another generation pass.

The initial validation log reported `questions=10` for one model scope and `invalid=0`; later replacement scopes reported `questions=2` and `invalid=2`. Because final assembly was never reached, there is no authoritative 23-question final output and no valid basis for independent mathematical verification of all questions.

## Comparison with the previous live baseline

| Metric | Previous baseline | This paid run | Interpretation |
|---|---:|---:|---|
| Runtime excluding Catalog | 986.212 s | Not complete | No improvement claim possible |
| Input tokens excluding Catalog | 5,696,128 | 1,133,044 persisted so far | Lower, but incomplete run and different path |
| Initial generation time | 123.904 s | about 42.4 s first pass | Faster first pass; persistence failure followed |
| Final questions | 21/23 | 0 persisted | Regression in outcome |
| Duplicate groups | 1 unresolved | No final result | Review did not reach assembly |
| Generation concurrency | Previous baseline unavailable | Peak 23 | Confirmed independent scheduler fan-out |

## Confirmed remaining bugs and bottlenecks

1. The live Docker worker did not find or accept the compatible Catalog cache, causing 11 paid compiler calls and roughly 177 seconds of avoidable work.
2. Catalog compiler batches can produce cross-batch FITB evidence IDs, causing corrective retries.
3. Generated content can contain `\\u0000`, which reaches JSONB persistence and aborts the job. Sanitization or fail-closed validation must happen before persistence.
4. The Celery task retry policy retries the entire exam after a persistence error, regenerating all questions.
5. The running container still uses bundled generation schemas for initial and replacement generation, despite the intended independent generation architecture.
6. Repair/revalidation did not converge for two replacement questions. The job remained at 95% and required manual stopping.
7. Validator logs showed repeated question IDs within batch scopes, which should be audited for stable slot/question identity mapping.

## Conclusion

The live test verified that the Docker scheduler can launch 23 concurrent generation requests and that real provider calls reach generation, validation, repair, and duplicate replacement. It did not meet the success criteria: the Catalog was not reused, the job did not persist an exam, and 23/23 correctness cannot be claimed.

No production code, document chunks, Catalog files, or prior exams were modified. The only repository artifact created for this test is this report.
