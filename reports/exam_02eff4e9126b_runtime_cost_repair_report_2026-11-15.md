# Latest Exam Generation Runtime, Cost, and Repair Report

Date: 2026-11-15  
Job: `b03ca729-03c7-4d88-a7f6-521164934848`  
Exam: `exam_02eff4e9126b`  
Document: `a51b75e6-373a-4476-a399-2dbf7c97c15f`

## Executive summary

The latest generation completed successfully and produced all 46 requested questions across two models. It took **10 minutes 18.50 seconds of worker runtime**. The job waited approximately **4.74 seconds** between creation and worker start, so creation-to-finish time was approximately **10 minutes 23.24 seconds**.

The run was slow primarily because it executed **238 recorded LLM requests**, including 19 Catalog calls, 111 validator calls, 57 generator calls, 17 duplicate-judge calls, 12 repair calls, four recovery-decision calls, and two corrective small-batch validations. Validation/recovery occupied most of the wall-clock runtime. Model 2 repeatedly cycled through repair, regeneration, revalidation, and duplicate checks. The final cross-model duplicate phase alone took approximately two minutes.

Recorded estimated LLM cost was **$0.286182**. This is the application's telemetry estimate, not a provider invoice.

## Job result

| Item | Result |
|---|---:|
| Status | Completed |
| Worker attempts | 1 |
| Worker runtime | 618.50 seconds |
| Celery task runtime | 618.89 seconds |
| Queue delay before start | 4.74 seconds |
| Requested questions | 46 |
| Final valid questions | 46 |
| Final missing questions | 0 |
| Final invalid questions | 0 |
| Models | 2 |
| Language | Arabic |
| Requested difficulty | Easy |
| Actual difficulty | 35 easy, 11 medium fallback |

Requested distribution: 20 MCQ, 10 True/False, 8 Equation, and 8 Word Problem questions. The final exam retained the complete requested count.

## Wall-clock timeline

Times below come from persisted job and audit timestamps. They are wall-clock intervals, so concurrent calls are counted once.

| Phase | Timestamp | Approximate wall time | Finding |
|---|---:|---:|---|
| Job created | 03:33:05.590 | — | Queued |
| Worker started | 03:33:10.331 | 4.74 s queue delay | One worker attempt |
| Retrieval complete | 03:33:11.285 | 0.95 s | 103/103 chunks selected |
| Catalog complete | 03:34:47.914 | 96.63 s | 18 batches plus one targeted retry; 128 tasks compiled |
| Initial plan complete | 03:34:48.370 | 0.46 s | 46 slots; one error |
| Plan repair complete | 03:34:48.993 | 0.62 s | One targeted plan repair |
| Difficulty adjustment complete | 03:35:41.981 | 52.99 s | 16 batch calls; 11 questions ultimately remained medium |
| Generation complete | 03:36:16.134 | 34.15 s | Both 23-question models initially generated |
| Validation/recovery begins | 03:36:16.859 | 0.73 s | Both models began concurrently |
| Model 1 locally ready | 03:37:40.751 | 83.89 s after validation began | Included one validation-stage regeneration call |
| Model 2 locally ready | 03:41:27.445 | 310.49 s after validation began | Multiple repair/revalidation cycles |
| Cross-model duplicate phase | 03:41:27.539–03:43:27.563 | 120.02 s | Dominant final serial phase |
| Assembly | 03:43:27.798 | 0.24 s | All 46 questions assembled |
| Job finished | 03:43:28.834 | 1.04 s | Persisted as complete |

The validation/recovery interval from generated questions to completed cross-model duplicate checking was approximately **431.43 seconds (7 minutes 11 seconds)**, about **69.8% of worker runtime**.

## LLM calls, tokens, and cost

### Totals

| Provider | Successful calls | Input tokens | Output tokens | Reasoning tokens | Recorded cost | Sum of call durations |
|---|---:|---:|---:|---:|---:|---:|
| OpenAI | 234 | 1,388,414 | 309,240 | 164,000 | $0.275400 | 3,219.12 s |
| DeepSeek | 4 | 31,710 | 1,183 | 0 | $0.010782 | 7.91 s |
| **Total** | **238** | **1,420,124** | **310,423** | **164,000** | **$0.286182** | **3,227.03 s** |

The summed call duration is much greater than job runtime because many requests ran concurrently. It must not be interpreted as elapsed wall time.

### Calls by operation

| Operation | Calls | Serial call time | Average | Maximum |
|---|---:|---:|---:|---:|
| Catalog compilation | 19 | 1,220.67 s | 64.25 s | 92.72 s |
| Difficulty adjustment | 16 | 285.06 s | 17.82 s | 26.68 s |
| Generation/regeneration | 57 | 529.63 s | 9.29 s | 22.51 s |
| Validator batches | 111 | 903.64 s | 8.14 s | 29.68 s |
| Duplicate validation | 17 | 110.99 s | 6.53 s | 13.12 s |
| Recovery decisions | 4 | 36.08 s | 9.02 s | 9.91 s |
| Repair agents | 12 | 110.65 s | 9.22 s | 22.17 s |
| Corrective smaller-batch validation | 2 | 30.32 s | 15.16 s | 25.17 s |

### Largest call groups by agent

| Agent | Calls | Serial call time | Recorded cost |
|---|---:|---:|---:|
| Assessment compiler | 19 | 1,220.67 s | $0.074390 |
| Generator | 57 | 529.63 s | $0.056817 |
| Task-alignment validator/decision | 29 | 306.84 s | $0.038493 |
| Difficulty adjuster | 16 | 285.06 s | $0.021274 |
| Difficulty validator | 24 | 229.44 s | $0.018864 |
| Grounding validator | 24 | 203.27 s | $0.022155 |
| Answer-correctness validator | 24 | 137.33 s | $0.013318 |
| Duplicate validator | 17 | 110.99 s | $0.009214 |
| Quantitative validator | 16 | 93.17 s | $0.010030 |

## Failed and rejected calls

### Provider/API failures

- Recorded non-success LLM requests: **0**.
- Requests with transport/provider retries: **0**.
- Recorded retry-after delay: **0 seconds**.
- Job-level failure: **none**.

### Logical failures despite successful API responses

Provider success does not mean that the returned content passed application validation.

1. **One Catalog response was logically rejected.** Batch 3 referenced a child evidence ID outside its assigned batch. The provider call returned HTTP success, but strict Catalog validation rejected it and issued one targeted retry. This explains 19 Catalog calls for 18 deterministic parent batches.
2. **One initial plan slot failed planning checks.** Slot `m1_equation_3` triggered a targeted plan repair, which succeeded.
3. **Difficulty adjustment could not reach easy for 11 questions.** They remained medium under the existing permitted fallback policy. These are model-declined/contract-limited adjustments, not provider call failures.
4. **Four recorded first-pass validation failures occurred in Model 2:** one wrong answer, two combined question-and-answer defects, and one unclear/invalid question.
5. Some generated questions were changed for repair, recovery, duplicate handling, or balancing and therefore required revalidation.

## Generation and regeneration

The initial generation produced 46 questions using 46 generator calls: 23 for Model 1 and 23 for Model 2.

There were **11 additional generator calls**:

- Model 1 validation-stage regeneration: 1.
- Model 2 validation-stage regeneration: 8.
- Later generation-stage calls without a model number in telemetry: 2, associated temporally with cross-model correction/balancing.

Therefore, generator activity was 46 initial calls plus 11 corrective/regeneration calls, for 57 total.

## Repair and recovery cycle

### Recorded repair outcomes

The final evaluation reports:

| Metric | Count |
|---|---:|
| Questions sent to repair | 6 |
| Repair succeeded | 6 |
| Repair failed | 0 |
| Final invalid | 0 |
| Final missing | 0 |

Repair by type:

| Question type | Sent | Succeeded | Recorded first-pass reason |
|---|---:|---:|---|
| MCQ | 2 | 2 | One question-and-answer defect; one unclear/invalid question |
| Equation | 1 | 1 | Wrong answer |
| True/False | 2 | 2 | No first-pass reason persisted in the summary |
| Word Problem | 1 | 1 | Question-and-answer defect |

### Repair-agent calls

Six affected questions required **12 repair-agent calls** because multiple specialized agents can operate on one question:

| Repair agent | Calls | Serial time |
|---|---:|---:|
| Difficulty repairer | 3 | 41.10 s |
| Solution repairer | 3 | 25.62 s |
| Answer repairer | 3 | 4.88 s |
| Question repairer | 1 | 22.17 s |
| Task-alignment repairer | 1 | 13.84 s |
| Options repairer | 1 | 3.04 s |

All 12 repair calls returned successful provider responses. The six repaired questions ultimately passed.

### Model recovery sequence

- Model 1: validation → duplicate check → repair/recovery → revalidation → duplicate check → locally ready. It reached local readiness at 03:37:40.751.
- Model 2: validation → duplicate check → repair/recovery → revalidation, followed by three more recorded repair/revalidation/duplicate-check cycles before reaching local readiness at 03:41:27.445.
- Cross-model processing then ran for another two minutes and included duplicate judgments, two generator corrections, difficulty adjustments, validator calls, and two corrective smaller-batch task-alignment validations.

This repeated Model 2 cycle and final cross-model phase are the largest reasons the job continued long after initial generation completed.

## Duplicate-check workload

There were **17 LLM duplicate-judgment calls**. Persisted duplicate-screening summaries show repeated scans:

- Model 1: at least two within-model scans.
- Model 2: at least six within-model scans as questions changed during recovery.
- Final forced cross-model scan: 46 questions, 1,035 possible/evaluated pairs, with three candidates sent for judgment.

The final duplicate result was clear: zero unresolved duplicate conflicts. However, rescanning after material changes added significant latency, especially because cross-model checking was effectively a final serial barrier.

## Why the run took so long

Ranked by confirmed impact:

1. **Validation and recovery dominated wall time.** Approximately 431 seconds elapsed from generation completion through cross-model duplicate completion.
2. **Model 2 required repeated correction cycles.** It took about 310 seconds from validation start to local readiness, compared with about 84 seconds for Model 1.
3. **Cross-model duplicate/correction processing took about 120 seconds.** This occurred after both models were locally ready and directly extended final completion time.
4. **Catalog compilation used 18 parallel batches plus one retry.** Catalog wall time was about 97 seconds. Individual calls were slow—64.25 seconds on average and up to 92.72 seconds—but parallelism prevented the 1,220 seconds of aggregate call time from becoming 1,220 seconds of wall time.
5. **The pipeline made many calls.** 238 requests and 1.42 million input tokens create substantial provider latency even with concurrency.
6. **Eleven corrective generator calls followed the 46 initial generation calls.** Corrective generation also triggers downstream revalidation and duplicate rescanning, multiplying its cost.
7. **Difficulty processing was substantial.** Initial adjustment took about 53 wall-clock seconds, and further difficulty calls appeared during later correction. Eleven tasks still required medium fallback because easy transformations could not preserve their assessment contracts.

This was not slow because of provider outages, rate-limit retries, scheduler queue contention, or a failed worker attempt. Scheduler acquisition logs showed essentially zero queue time for the observed calls, and all recorded requests completed successfully.

## Data-quality and telemetry limitations

1. The final evaluation reports Model 1 `validation_total_first = 0`, even though logs and request events show Model 1 validator and recovery activity. First-pass statistics for Model 1 are therefore incomplete or incorrectly aggregated.
2. Two late generator calls have no model number in request metadata. Their exact target questions cannot be proven from `llm_request_events` alone.
3. The audit records phase transitions, not every internal question-level decision. The exact question-to-agent mapping for all 12 repair calls is not fully persisted.
4. Recorded cost is an application estimate. Provider billing may differ.
5. Serial call-duration sums overlap because of concurrency and cannot be added to obtain wall time.

## Practical conclusions

- The exam completed correctly with the full requested count.
- No provider calls failed, but one Catalog result and several question outputs failed application-level checks.
- The current performance bottleneck is no longer the earlier Catalog crash. It is the volume of validation plus repeated Model 2 recovery and final cross-model duplicate/correction work.
- The most useful next optimization investigation would measure why Model 2 needed four repair/revalidation passes and why duplicate/cross-model correction repeatedly caused new generation and validation work.

