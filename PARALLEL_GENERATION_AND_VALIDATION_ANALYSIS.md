# Parallel Generation and Validation Analysis

Investigation date: 25 October 2026 (Asia/Dubai). This is a read-only analysis of the current production source, tests, Docker/Celery configuration, PostgreSQL audit records, retained LLM-usage rows, and worker logs for job `6f32fc76-1ce8-4492-837d-812aa687abdf` / exam `exam_1d3236e97400`. No provider call, exam run, database write, cache deletion, code change, or test change was performed.

## 1. Executive conclusion

**CONFIRMED:** DraftWork does not currently generate one question per call. For the 23-question baseline, it created three logical initial generation units: one combined 10-MCQ/5-True-False bundle, one four-Equation batch, and one four-Word-Problem batch. Structured-output recovery expanded those logical units into seven measured provider requests. The generation stage took 123.904 seconds and used 125,600 input and 17,412 output tokens.

**CONFIRMED:** All eight supported types can be generated one slot per call without replacing the graph or final-assembly architecture. The existing plan item, source isolation, strict per-type schemas, stable `slot_id`, lifecycle, checkpoint, and merge concepts are sufficient. FITB needs special handling because its shared word bank is a section dependency; the item calls can be independent, but the bank must still be rebuilt deterministically after collection.

**CONFIRMED:** A literal 50-call launch is not safe under the current runtime. Generation is capped at 5 by the worker's Docker setting, non-Catalog work shares a per-job gate of 10, and the inherited provider semaphore is effectively 5 in this deployment. Provider RPM, TPM, daily-token, and account-level concurrent-request limits are not stored in the repository or audit data. Fifty must remain a configurable hard ceiling behind an adaptive limiter, not the initial operating value.

**RECOMMENDATION:** Implement one-slot generation, but ship with an effective default of 5 and benchmark 5, 10, 20, then 50. Use a work-conserving queue: start a new slot whenever one finishes, while enforcing provider-specific request and token budgets. Do not raise the ceiling until controlled live verification establishes the provider allowance.

**CONFIRMED:** The requested validation/repair phase behavior mostly exists already. The first validation pass covers every generated question before graph routing. General and STEM validation run concurrently in bounded batches, all verdicts are merged into one manifest, non-FITB repairs are launched independently in the same repair phase, candidates are staged, and only candidates that pass revalidation are committed. Fingerprints already avoid revalidating unchanged questions. The design should retain and tighten these mechanisms, not build a parallel validation architecture.

**RECOMMENDATION:** Keep small validation batches, normally 4-8 questions, with independent evidence boundaries and deduplicated source text within each request. One huge call has poor failure isolation and structured-output reliability; one call per question repeats too much instruction and schema overhead. The current batch-of-8 shape is reasonable, but its payload is bloated and its corrective retries can become enormous.

## 2. Current generation architecture

### 2.1 Production path

The online path is `app/jobs/tasks.py:363` (`generate_exam`) -> `app/online/exam_builder.py:2373` (`generate_exams`) -> the LangGraph assembled at `app/online/graph.py:224` (`build_exam_graph`) -> `app/online/exam_builder.py:1759` (`generate_exams_node`). The referenced `app/offline/pipeline.py` performs document processing and is not the question-generation scheduler.

The graph validates the reviewed plan again at generator entry. `generate_exams_node` calls `plan_structure_errors`, `validate_plans`, and `validate_plan_catalog_bindings`; it refuses generation if the plan is invalid (`app/online/exam_builder.py:1775`). This gate must remain.

### 2.2 Which types are bundled

| Question type | Current logical generation unit | Concurrency behavior |
|---|---|---|
| MCQ | All pending MCQ slots are sent with all pending True/False slots in one objective bundle | Objective bundle runs before the other types |
| True/False | Bundled with MCQ | Same call as MCQ; bounded retries may resend only pending slots |
| Fill-in-the-Blank | One type-level item-generation call for all pending FITB slots, followed by deterministic shared-bank construction | Runs as one type job; FITB repairs remain grouped because of the bank |
| Definition | All pending Definition slots in one type call | Different type jobs can run concurrently |
| Why | All pending Why slots in one type call | Different type jobs can run concurrently |
| Equation | All pending Equation slots in one type call | Concurrent with other non-objective types |
| Word Problem | All pending Word-Problem slots in one type call | Concurrent with other non-objective types |
| Essay | All pending Essay slots in one type call | Concurrent with other non-objective types |

`_generate_obj_bundle` at `app/online/exam_builder.py:1180` owns MCQ/True-False bundling. `_generate_type_from_plan` at line 629 owns ordinary type batches. FITB has its own path at lines 960 and 1124. At lines 2022-2056 the objective bundle is completed and checkpointed before remaining types are scheduled. At lines 2066-2139 one job per remaining question type is submitted to a `ThreadPoolExecutor`; results are collected with `as_completed` but merged in original task order.

For the requested 10 MCQ, 5 True/False, 4 Word Problems, and 4 Equations, the intended initial call topology is:

1. One logical MCQ+True/False request containing 15 slots.
2. One logical Equation request containing 4 slots.
3. One logical Word-Problem request containing 4 slots.
4. Equation and Word Problem run concurrently after the objective bundle finishes.

In the live run, provider structured-output retry behavior made seven provider requests: three for the objective phase and two each for Equation and Word Problem. Worker logs show the objective phase from 20:36:04.678 to 20:37:29.932 UTC, then two concurrent type jobs from 20:37:29.962 to 20:38:08.343 UTC.

### 2.3 Prompt and schema contract

Each type batch receives only its planned items plus assigned source content selected by `_context_for_items` (`app/online/exam_builder.py:175`). `_generation_plan_items` at line 205 enriches each slot with the exact source registry and assessment profile. `build_prompt` (`app/online/prompts.py:788`) renders the type instructions, evidence registry, assessment guidance, plan contract, difficulty contract, and notation rules. `build_obj_bundled_prompt` at line 1203 renders the analogous combined objective prompt.

The model does not receive the complete exam plan. It receives all pending slots for that generation unit. A slot entry carries task/family identity, selected variation, operation, target, givens, reasoning mode, work product, difficulty adaptation, assigned evidence IDs/text, and reuse context when applicable.

Strict schemas are dynamically narrowed to the expected slots. `generation_response_model` (`app/llm/schemas.py:539`) constrains `slot_id` to the requested literals. MCQ uniquely permits a question list shorter than the slot count plus explicit `unsuitable_tasks`; the other types require the exact count. `objective_bundle_response_model` at line 558 combines MCQ and True/False response models.

### 2.4 Slot identity, ordering, and answer balancing

Planner IDs have the stable form `m{model}_{type}_{index}`. Generation responses return this `slot_id`; `_bind_selected_variation` (`app/online/exam_builder.py:317`) rejects a response whose selected variation is incompatible with the plan. Results are stored by slot and emitted in Planner order, then `_assign_ids` at line 430 stamps public IDs such as `model1_mcq_3`.

MCQ answer-label balancing is deterministic and post-generation. `balance_mcq_answer_positions` at line 376 swaps option positions and updates `correct_answer` and `solution_certificate.matching_correct_option`; it does not alter option content. Therefore independent generation is compatible with deterministic balancing. Any MCQ changed by balancing after a repair is added to selective revalidation (`app/online/validator.py:3200`). True/False has a different generation-time balance rule: with at least four items, `_generate_obj_bundle` retries enough majority-polarity slots to obtain at least two True and two False answers, retaining a previously valid fallback rather than shrinking the exam.

### 2.5 Failures and retries today

Generation has three layers of recovery:

- A type/objective loop retries only still-pending slot IDs up to `_MAX_ATTEMPTS` (`app/online/exam_builder.py:655` and 1214).
- Structurally invalid rows can be sent to `_repair_invalid_items` at line 562 for a field/schema repair, isolated to one invalid item.
- The provider client may make up to three structured-output attempts (`app/llm/deepseek.py:381`). A corrective attempt resends the original prompt, previous output, failure reason, expected IDs, and the full JSON Schema (`app/llm/deepseek.py:330`).

Valid rows from a successful batch are retained by slot and later outer attempts request only pending slots. A transport or envelope failure can still fail the whole call and cause every slot in that call to be resent. A malformed individual row does not necessarily destroy valid siblings because row-level splitting and repair exist. Unknown, duplicate, missing, forbidden, near-duplicate, or variation-incompatible slot responses are rejected. After the bounded attempts, missing slots remain explicit; they are not silently fabricated.

## 3. One-question-per-call feasibility

### 3.1 Type compatibility

**CONFIRMED:** MCQ, True/False, Definition, Why, Equation, Word Problem, and Essay are directly compatible with one-slot response models. The current dynamic schema already supports a one-element slot list.

**CONFIRMED WITH A SECTION DEPENDENCY:** FITB item generation can be one call per slot, but the final Word Bank is shared. Collect all accepted FITB items, then run the existing deterministic bank synchronization/rebuild. A changed FITB stem/distractor can invalidate the bank fingerprint for the section, so all affected FITB items may require revalidation even when only one item changed.

### 3.2 Required call payload

Each call should contain one immutable `GenerationSlotRequest` equivalent to the existing enriched plan item:

- slot ID, model number, and question type;
- Catalog task, base task/family, selected variation/group, and effective instance IDs;
- operation, target, required output, reasoning mode/depth/requirements, and difficulty;
- required/effective/variable givens and complexity boundaries;
- only assigned evidence chunks and their IDs;
- assessment guidance for that type;
- reuse or duplicate-avoidance context only when this task was reused or the slot is a replacement;
- deterministic answer-position target for MCQ only if balancing is moved before generation; otherwise keep the current post-generation balancer.

The complete document, unrelated Catalog tasks, unrelated questions, and unrelated source chunks should not be sent.

### 3.3 Quality and duplicate trade-offs

Independent calls improve failure isolation and make per-slot retry exact. They also reduce schema size, output size, and the risk that one malformed row triggers a large corrective retry. Variation and evidence identity remain enforceable because they are already checked after response parsing.

The main quality risk is loss of within-batch awareness. The current objective/type prompt can see siblings and generation filters compare accepted content. Independent calls launched simultaneously cannot see each other's realized text, so lexical and semantic duplicates may increase. The Planner's task/variation diversity remains available, but it cannot eliminate coincident wording or answer patterns. Mitigations are deterministic plan diversity, compact per-slot `avoid_signatures` derived from already reserved task/variation metadata, post-collection local duplicate checks, and the existing BGE-M3/Judge phase. Do not serialize generation merely to share previous generated text; that would erase the wall-time benefit.

Instruction and schema overhead will be repeated 23 times. Shared source text also repeats when multiple slots use the same chunk. Consequently one-slot calls are not guaranteed to use fewer total input tokens. They are likely to reduce tokens only if the request is compact and provider corrective retries fall materially. This must be benchmarked.

Question order remains deterministic by collecting into `results_by_slot` and rebuilding each section by the original Planner slot sequence. Successful slots remain immutable while only failed slots are retried. Duplicate or missing response IDs are rejected before merge.

## 4. Fifty-way concurrency assessment

### 4.1 Current limit stack

There are three relevant ceilings:

1. `MAX_PARALLEL_GENERATION_CALLS` is capped at 10 in `app/config.py:105`; Docker currently supplies a default of 5 to the worker.
2. `LLM_JOB_CONCURRENCY_LIMIT` is a shared non-Catalog per-job gate, default 10 and capped at 64 (`app/config.py:120`; `app/llm/scheduler.py:39`). General validation, STEM validation, generation, repair, and duplicate calls compete for this same job gate.
3. `DeepSeekClient._concurrency` is a process-wide provider semaphore (`app/llm/deepseek.py:105`). `OpenAIClient` inherits this implementation. Docker currently sets `DEEPSEEK_CONCURRENCY_LIMIT=5`, so the effective process-wide provider ceiling is 5 even if the per-job scheduler admits more.

The scheduler is per job, not global across jobs, except for the inherited process-wide provider semaphore. Catalog calls use a separate gate. The worker runs Celery prefork concurrency 1, but each job creates threads for LLM I/O; one active Celery task can therefore have several calls in flight.

### 4.2 Provider and resource constraints

**UNKNOWN:** The repository and retained telemetry do not contain the OpenAI account/project RPM, TPM, daily-token, or concurrent-request allowance. No claim that 50 is provider-safe can be made from current evidence. The live job records no 429, timeout, or provider transport failure, but it operated near 5 effective provider calls, not 50.

Each client request creates an `AsyncOpenAI` context in `_request_with_backoff`; there is no shared long-lived HTTP connection pool. Fifty calls would create substantial connection/TLS churn. Python memory for 50 compact prompts is manageable in principle, but 50 large prompts, schemas, previous outputs, and response objects can be hundreds of megabytes. CPU cost is mainly JSON/Pydantic work and BGE later; network/provider latency dominates generation.

The 1,300-second job deadline is only 88.6 seconds above the live run's 1,211.4 seconds. A 50-way implementation that triggers rate-limit backoff or long tail retries could exceed it. The timeout must not be raised as a substitute for controlling load.

### 4.3 Cancellation and fencing

Database writes and checkpoints are fenced by `claim_generation`, checkpoint revision, and request fingerprint (`app/api/repositories.py:678`, 733, and 1060). A heartbeat runs every 30 seconds and final persistence rechecks ownership. These mechanisms prevent a stale worker from publishing.

**GAP:** The generic generation loops do not check `is_cancelled` before each individual provider launch. Catalog compilation does, but a thread pool that already queued 50 generation jobs could continue making paid calls after cancellation or ownership loss until the node returns. The proposed dispatcher must check cancellation/ownership immediately before acquiring a provider permit and again before committing a result. Pending futures should be cancelled where possible. In-flight HTTP requests cannot be assumed cancellable.

Shared graph state must not be mutated by worker threads. The existing type-generation pattern already follows the right approach: each worker gets an immutable/deep-copied snapshot and returns a value; one coordinator merges deterministically. Extend that pattern to slots.

### 4.4 Safe operating recommendation

Use 50 as a configuration maximum and queue capacity. Start production at 5 effective in-flight generation calls, matching observed deployment behavior. Benchmark 10 next. Permit 20 and 50 only after live tests show acceptable 429 rate, TPM headroom, p95 latency, memory, and cancellation behavior.

The limiter should be provider-aware and adaptive:

- a global per-provider semaphore across jobs;
- a per-job fairness ceiling;
- token-weighted admission using estimated prompt tokens, not request count alone;
- RPM/TPM token buckets when authoritative limits are configured;
- retry-after handling with jitter and a temporary concurrency decrease;
- gradual recovery after a quiet interval;
- work-conserving scheduling through a fixed worker pool or queue, so a completed call immediately releases a permit for the next slot.

For a 23-question exam, a ceiling above 23 provides no benefit. With the current provider ceiling of 5, the 23 slots require at least five waves. For 100 questions, a 50-worker queue can exist while provider admission remains lower.

## 5. Result collection and exam combination

This should remain inside `generate_exams_node`.

1. Build an ordered list of `GenerationSlotRequest` objects from the authoritative plan.
2. Skip slots already present in a resumed generation checkpoint with a matching generation fingerprint.
3. Submit one immutable request per missing slot to a bounded dispatcher.
4. Return `GenerationSlotResult(slot_id, question_type, model_number, status, question, attempts, usage, error)`; workers never mutate shared state.
5. Reject an unknown slot, duplicate slot result, mismatched type/model, invalid selected variation, or invalid Pydantic payload.
6. Preserve the first successful result for a slot; retry only failed slots.
7. Merge by Planner order, run existing local forbidden/exact/near-duplicate checks, rebuild FITB banks, assign stable public IDs, and run deterministic MCQ balancing.
8. Persist periodic generation checkpoints after a bounded number of completed slots or elapsed seconds, using the existing checkpoint writer and revision fencing.
9. If retries exhaust, retain an explicit missing lifecycle entry. Never adjust the requested count.

The result schema should require exactly one question or one explicit unsuitable result. A single-slot response cannot accidentally claim another slot because the literal schema contains one allowed ID.

## 6. Validation architecture comparison

### 6.1 Current behavior

`validate_generated_questions` at `app/online/validator.py:2945` validates the entire current model on its first pass. General and STEM reviews run concurrently (`run_reviews`, line 3171). `validate_exam` at line 1668 divides entries into batches of `VALIDATOR_BATCH_SIZE=8`, runs up to `MAX_PARALLEL_VALIDATOR_CALLS`, salvages valid rows from malformed envelopes, and retries missing IDs individually once. Graph routing occurs only after the node returns all reports, so repair does not begin after the first failed batch.

The live first pass sent 23 questions to General validation in batches 8/8/7 and 21 quantitative questions to STEM in batches 8/8/5. It used 6 provider calls, 238,298 input tokens, 23,264 output tokens, and about 51.6 seconds of wall time. This is measured.

### 6.2 Why validation input became enormous

The 2.07M General and 1.89M STEM figures cover all nine validation/revalidation cycles, including replacement candidates, not one initial pass. The largest General call had 292,516 input tokens.

Confirmed prompt contributors are:

- the full current question, options/answer/steps/certificate;
- the full plan item;
- `expected_catalog_task`, which repeats much of that plan in a second representation (`app/online/validator.py:1431`);
- full assigned source text embedded separately inside every entry (`build_validator_prompt`, line 1371), so two entries using the same chunk repeat it in one request;
- the detailed validator system prompt and output-format example;
- the provider JSON Schema;
- on structured-output retry, the complete original prompt, previous output, failure text, expected IDs, and full schema again (`app/llm/deepseek.py:330`).

The live call ledger also shows repeated changed candidates and duplicate replacements: 32 General calls and 31 STEM calls. Thus both payload duplication and repeated validation cycles drove the total.

### 6.3 Option comparison

| Option | Tokens | Wall time | Accuracy/reliability | Failure isolation | Recommendation |
|---|---|---|---|---|---|
| A. One huge call | Lowest repeated instruction overhead in theory; very large prompt/output and corrective retry | One long tail; a retry repeats everything | Attention dilution and structured-output omission risk increase; provider context/output limits become material | Poor: one envelope failure affects all questions | Reject |
| B. One call per question | Smallest evidence boundary and strongest isolation; repeats system/schema/contract 23-44 times | Fast only with high permitted concurrency; sensitive to RPM/TPM | Strong local focus; loses batch amortization | Excellent | Use for missing-ID retry, high-risk candidates, and benchmarks; not default for all validation |
| C. Small concurrent batches | Amortizes instructions/schema; payload remains bounded | Good parallel throughput | Current evidence shows complete coverage and robust salvage | Good; unresolved IDs retry alone | Preferred |

Use batches of 4-8, chosen by estimated tokens rather than a fixed count alone. Within a batch, emit a single evidence registry keyed by chunk ID and have each entry refer to IDs. Replace the duplicated full plan plus `expected_catalog_task` with one canonical compact validation contract. Preserve every General/STEM rule and independent evidence boundary.

## 7. Coordinated repair and iterative revalidation

### 7.1 What already exists

The current pipeline already satisfies most requested semantics:

- `validation_reports` preserve every verdict and role-specific findings.
- `_merge_role_reports` (`app/online/validator.py:2627`) combines General and STEM defects into `required_fixes` without discarding either.
- `repair_manifest` records one target per slot, category, fields, reason, strategy, status, and history.
- Graph routing at `app/online/graph.py:444` waits until validation completes, then routes unresolved manifest entries to repair.
- `repair_invalid_questions` at `app/online/validator.py:3617` launches one isolated job per non-FITB target concurrently; FITB targets are grouped because of the shared bank.
- Repairs are staged in `repair_candidates`. The next validation node validates candidates in parallel and commits them only if every required validator passes (`app/online/validator.py:2992`).
- Repair attempts are bounded per slot at three (`app/online/validator.py:48`).
- Unsupported evidence/task failures route through plan replacement rather than weakening the verdict (`_requires_plan_replacement`, line 83, and `_regenerate_plan_slot`, line 2667).
- The graph loops repair -> validation until budgets exhaust, then duplicate review.

One repair call per invalid question is the best default. It matches current non-FITB behavior, keeps failures isolated, and permits parallelism. Small grouped repairs are useful only for a real shared dependency such as a FITB bank. One request containing the entire failed list repeats unrelated evidence and lets one malformed patch affect all failures.

### 7.2 Fingerprint reuse

The current validation ledger is stronger than the proposal assumes. At `app/online/validator.py:3139`, the fingerprint includes the question, current plan, and FITB section dependency. At lines 3229-3248, unchanged questions with all required validator roles reuse prior verdicts; stale IDs alone are rechecked. Selective repaired IDs are handled at lines 3209-3220. Tests at `tests/test_performance_optimizations.py:174` verify unchanged failed-verdict reuse, and line 236 verifies overlapping independent repair jobs and deterministic merge.

Required hardening:

- version the fingerprint with validator rules, prompt version, schema version, model identity, and source-content hashes;
- store General and STEM fingerprints separately when their dependencies differ;
- invalidate an MCQ after answer-position balancing, a FITB after relevant bank change, and any question after task/evidence replacement;
- retain PASS only when every dependency hash and validator version is unchanged;
- keep prior failed strategies/candidates in repair memory and forbid exact strategy repetition.

## 8. Incremental duplicate review

Initial review currently embeds/scores all pairs in memory, shortlists by BGE-M3 similarity and deterministic signals, then sends candidate batches to the Duplicate Judge (`app/online/duplicate_judge.py:207` and 477). Replacement reservation is sequential to avoid selecting the same Catalog resource, while generation of reserved replacements can run concurrently (`app/online/duplicate_pipeline.py:978`). Replacements return to correctness validation before another duplicate scan through the existing graph.

There is already partial incremental behavior. `duplicate_fact_cache` stores content-keyed Judge semantic facts; unchanged pairs reuse those facts, and a changed answer invalidates affected cache entries. This is covered at `tests/test_duplicate_judge.py:269` and 310. Candidate blacklists and failure history also prevent some repeated task selections.

The missing part is incremental BGE screening and pair-set persistence. `scan_exam_duplicates` at `app/online/duplicate_pipeline.py:659` rebuilds the filtered exam and rescans it. Implement a persisted pair ledger keyed by:

`hash(question A content+answer+task metadata, question B content+answer+task metadata, embedding model/version, screening rules, Judge prompt/model version)`.

After replacement, invalidate only pairs touching changed question IDs. Re-embed the changed question, compare it with every unchanged active question, Judge only newly shortlisted/forced pairs, then rebuild conflict groups from active cached verdicts. Preserve unresolved conflict history separately from the current active pair state. Continue using family/operation/target/work-product exclusions in candidate ranking.

## 9. Measured baseline

All values below are measured from PostgreSQL and worker logs for the specified live job.

| Metric | Measured value |
|---|---:|
| Submission-to-completion wall time | 1,211.412 s |
| Catalog compilation stage | 225.200 s |
| Runtime with only Catalog stage duration subtracted | 986.212 s |
| Runtime from Catalog completion to job completion | 984.951 s |
| All LLM calls | 171 |
| All input / output tokens | 5,780,634 / 277,845 |
| Input / output excluding Catalog calls | 5,696,128 / 183,222 |
| Total input+output excluding Catalog | 5,879,350 |
| Initial generation logical units / provider requests | 3 / 7 |
| Initial generation wall time | 123.904 s |
| Initial generation input / output tokens | 125,600 / 17,412 |
| Initial validation calls | 6 (3 General + 3 STEM) |
| Initial validation input / output tokens | 238,298 / 23,264 |
| Initial validation wall time | about 51.6 s |
| All General / STEM calls | 32 / 31 |
| All General / STEM input tokens | 2,072,332 / 1,894,427 |
| Question repair calls | 9 |
| Duplicate replacement / Judge calls | 17 / 22 |
| Final result | 21/23; one unresolved duplicate group |

The 171 calls include provider structured retries, candidate generation/replacement, validation, repair, and Judge calls. They are not 171 sequential graph nodes.

## 10. Modelled performance

These are engineering models, not measurements. Ranges use the live call-duration distribution and explicit assumptions; only an A/B benchmark can replace them.

For generator calls, the live median provider duration was 8.107 seconds, p25 6.598 seconds, p75 10.970 seconds, and maximum 42.648 seconds. With 23 independent calls and an effective ceiling of 5, no-retry service requires five waves: approximately 33 seconds at p25, 41 seconds at median, or 55 seconds at p75, plus scheduling/merge time. Tail calls and structured retries can raise this to 90-210 seconds. At a verified ceiling of 10, the no-retry median-wave model is about 24 seconds; at 23 true concurrent calls, wall time is controlled by the slowest call, not the median.

Independent-call input tokens are unknown because repeated instructions/schema/evidence may offset smaller payloads. A benchmark must measure them. A reasonable planning envelope is 70k-160k input tokens for 23 first-attempt calls if prompts are compact; it is explicitly unverified. The measured current initial generation value is 125.6k including retries.

| Metric | Current architecture (measured) | Proposed best case (modelled) | Proposed typical (modelled) | Proposed worst case / unknown |
|---|---:|---:|---:|---:|
| Initial generation logical/provider calls | 3 / 7 | 23 / 23 | 23 / 25-30 | 23 / up to 69 with three provider attempts each |
| Generation wall time | 123.9 s | 25-40 s with >=10 admitted | 40-90 s at 5-10 admitted | 210+ s under tail/retry pressure |
| Generation input tokens | 125.6k | 70-100k | 100-160k | Could exceed current; benchmark required |
| Initial validation calls | 6 | 6 compact batches | 6-12 token-bounded batches | 44 per-question role calls, or more with retries |
| Initial validation input tokens | 238.3k | 80-130k | 120-190k | >=238k if compaction fails/retries persist |
| Revalidation calls | 57 after initial across both roles | 0-8 | 8-24 | Similar to or above current if many candidates change |
| Revalidation tokens | Included in 3.97M General+STEM total | 0.05-0.25M | 0.25-1.0M | Up to current or higher |
| Repair calls | 9 | 0-4 | 4-9 | Up to per-slot limits; 69 slot attempts theoretical for 23 slots |
| Duplicate Judge calls | 22 | 1-6 | 4-12 | Similar to current if duplicate replacements churn |
| Runtime excluding Catalog | 986.2 s | 120-240 s | 300-540 s | 960-1,300+ s with rate limits/tails |
| Input tokens excluding Catalog | 5.696M | 0.45-0.9M | 1.2-2.8M | 5.7-7.5M if overhead/retries are not controlled |
| Probability of losing a question | Observed 2/23 in one run; not a probability estimate | Unknown | Expected lower through slot isolation, but unquantified | Must be measured over repeated paired runs |

The proposed typical runtime improvement is therefore approximately 45-70% excluding Catalog, and typical input-token reduction approximately 50-80%, **if** compact validation payloads, ledger reuse, and incremental duplicate review work as designed. One-slot generation alone does not justify those totals; most measured waste was in repeated validation and duplicate replacement. No statistically valid loss probability can be inferred from one exam.

## 11. Offline and controlled-live benchmark plan

### Offline deterministic A/B

Use the exact saved Catalog, authoritative 23-slot plan, assigned source chunks, and recorded generation/validator fixtures. Do not create a new production exam.

Compare:

- A: current objective/type batching;
- B: one-slot generation with concurrency 1, 5, 10, 20, and simulated 50;
- validation batches 1, 4, 8, and all-in-one;
- current payload versus compact canonical contract/evidence registry;
- current full duplicate rescan versus incremental pair ledger.

Measure call count, prompt/schema characters and tokenizer estimates, peak queued/in-flight calls, deterministic merge/order, slot completeness, schema rejection isolation, checkpoint/resume equivalence, cancellation launch count, fingerprint invalidation, BGE pair count, Judge cache hits, and repair candidate commit correctness. Inject missing IDs, duplicate IDs, malformed rows, 429/retry-after, timeouts, stale ownership, cancellation, and worker restart.

Offline fixtures can prove orchestration, token estimates, deterministic state, isolation, and cache invalidation. They cannot measure provider latency, model quality, real structured-output failure rate, RPM/TPM behavior, or mathematical accuracy.

### Controlled live verification

After offline success, use a non-production controlled provider benchmark with the same frozen plan/evidence and fixed seeds where supported. Run enough paired samples to report medians and p95, not one anecdote. Compare slot completion, independent human/validator quality, duplicates, retries, prompt/cache tokens, cost, 429s, latency, and cancellation. Increase concurrency one step at a time. Do not enable 50 in production until the provider limit and p95 behavior are known.

## 12. Ordered implementation plan

1. **Add feature flags and versions.** In `app/config.py`, add `GENERATION_MODE=type_batch|per_slot`, `GENERATION_MAX_IN_FLIGHT`, provider/global ceilings, token-weighted limits, compact validation payload flag, validation batch token budget, and incremental duplicate flag. Include behavior-changing flags/versions in `immutable_exam_request` and `request_fingerprint` (`app/online/exam_builder.py:52`) so an old checkpoint cannot resume under a different contract.

2. **Add single-slot schemas.** In `app/llm/schemas.py`, add strict `GenerationSlotResult` models per type or wrap the existing one-item model. Require one literal slot ID and exactly one of `question` or an authorized explicit unsuitability result. Do not weaken existing question schemas.

3. **Build a compact slot prompt.** In `app/online/prompts.py`, add `build_slot_generation_prompt`. Reuse current type rules, plan block semantics, difficulty, notation, and evidence isolation. Include only one enriched plan item and its assigned chunks. Keep the existing batch prompts for fallback and A/B.

4. **Refactor generation without a second assembler.** In `app/online/exam_builder.py`, add a slot dispatcher used inside `generate_exams_node`. Reuse `_generation_plan_items`, `_bind_selected_variation`, local rejection checks, `_assign_ids`, FITB rebuild, MCQ balancing, lifecycle, and final assembly. Return immutable slot results; merge on the coordinator thread in Planner order.

5. **Make retries slot-local.** Retry only the failed slot. Preserve successful results and failed candidate history. For `mcq_task_unsuitable`, reserve replacement tasks sequentially before launching replacement generation. Keep bounded attempts and explicit missing status.

6. **Strengthen checkpoints.** Persist completed slot results, attempts, generation fingerprints, and pending slot IDs periodically through the current audit callback. Resume only slots whose fingerprint is absent/stale. Preserve checkpoint revision, claim generation, and request fingerprint fencing.

7. **Replace static admission with layered adaptive scheduling.** Modify `app/llm/scheduler.py` and provider clients to enforce process/global provider ceilings, per-job fairness, RPM/TPM buckets, retry-after, and cancellation checks. Reuse connections where the SDK safely supports it. Keep 50 as the configurable hard maximum; default to 5.

8. **Compact validator payloads.** In `app/online/validator.py`, serialize one canonical task contract instead of both full `plan` and duplicated `expected_catalog_task`. Deduplicate assigned source text into a request-level registry while retaining per-entry evidence IDs. Batch by estimated tokens, normally 4-8 questions. Preserve both validator system prompts and schemas.

9. **Prevent corrective retry amplification.** In `app/llm/deepseek.py`, design a bounded corrective retry for validation/generation that does not blindly repeat a huge previous output and full schema when the provider already receives the schema separately. Preserve enough rejected-row context to repair the exact failure. Add strict size telemetry and fall back to per-question retry for unresolved IDs.

10. **Version validation fingerprints.** Extend the ledger fingerprint with content, plan, source hashes, section dependencies, validator prompt/schema/model versions, and role. Preserve unchanged PASS and FAIL verdicts only when dependencies match.

11. **Keep coordinated repair.** Retain `repair_manifest`, parallel per-question jobs, staged candidates, transactional commit, and the repair-validation graph loop. Ensure every repair prompt carries the exact current question, answer/certificate, merged role findings, task/variation, givens, assigned evidence, and prior failed strategies. Group only true shared dependencies such as FITB.

12. **Add incremental duplicate pair state.** Extend `duplicate_fact_cache` into a versioned screening/Judge pair ledger. Re-embed and rejudge only pairs touching changed questions; preserve active unaffected verdicts and unresolved history. Keep candidate reservation, blacklists, and correctness validation before replacement commit.

13. **Tests.** Add unit tests for all eight single-slot types, unknown/duplicate/missing IDs, deterministic ordering, MCQ balancing and revalidation, FITB bank dependencies, one-slot retry isolation, provider 429/adaptive reduction, cancellation before launch, stale ownership, checkpoint resume, ledger version invalidation, incremental pair invalidation, and old-mode fallback. Add an integration fixture for 23 and 100 slots with completion-order randomization.

14. **Rollout.** Ship dark telemetry first, then offline A/B, controlled live 5-way, 10-way, and higher only if justified. Keep `type_batch` as a runtime fallback. Gate compact validation and incremental duplicate state independently so regressions can be isolated. Promote based on p95 latency, token use, complete-slot rate, validator agreement, duplicate rate, and cost.

## 13. Risks and final recommendation

The largest risks are provider throttling, repeated instruction overhead, simultaneous near-duplicate generation, stale-worker paid calls, and invalid reuse of cached PASS verdicts. The mitigations are adaptive provider admission, compact slot prompts, post-collection duplicate checks, launch-time ownership checks, and versioned dependency fingerprints.

Proceed with one-question-per-call generation as an opt-in architecture because it materially improves isolation and can shorten the measured generation stage. Do not define success as “50 calls.” Define it as the highest work-conserving concurrency that stays inside provider token/request limits and improves p95 completion without increasing defects. Keep small-batch whole-exam validation, existing coordinated per-question repair, selective fingerprint revalidation, and the existing graph/final assembly. The strongest expected savings come from combining slot isolation with compact validator payloads and incremental duplicate review.

## 14. Evidence limitations

- Provider account limits were unavailable; 50-way safety is **UNKNOWN** until controlled live verification.
- The proposed token and runtime ranges are **MODELLED**, not measured.
- One observed 21/23 exam cannot establish a probability of question loss.
- No new mathematical quality comparison was run, and no paid provider was called.
- Source locations refer to the working tree inspected on 25 October 2026; unrelated local changes were preserved.
