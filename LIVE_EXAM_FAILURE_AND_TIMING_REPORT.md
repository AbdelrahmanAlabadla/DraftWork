# Live Exam Failure and Timing Report

Investigation date: 25 October 2026 (Asia/Dubai). This was a read-only investigation of persisted PostgreSQL records, the job audit trail, available worker logs, and production source. No database records, code, tests, caches, or jobs were changed.

## 1. Executive Summary

**CONFIRMED:** The examined live execution is job `6f32fc76-1ce8-4492-837d-812aa687abdf`, producing exam `exam_1d3236e97400`. It requested 23 questions and persisted 21: 8 MCQ, 5 True/False, 4 Word Problem, and 4 Equation. Its public exam status is `partial`; the underlying job status is `completed`.

**CONFIRMED:** `m1_mcq_3` and `m1_mcq_10` were not lost in planning, generation, or duplicate review. Both reached generation and then failed validation/repair. Each exhausted three repair attempts, acquired terminal status `repair_failed`, and was deliberately omitted by final assembly as an invalid slot.

**CONFIRMED:** The two MCQs failed for different reasons:

* `m1_mcq_3` ended with `invalid_or_extraneous_solution`: its answer choice was mathematically plausible, but the recorded solution certificate claimed that adding the equations produced the option, which is false.
* `m1_mcq_10` ended with `missing_required_given`: the final question referred to team-membership conditions without stating them, making the asserted correct answer unanswerable from the prompt.

**CONFIRMED:** The one unresolved duplicate group is separate: `model1_mcq_5` and `model1_true_false_4`. Duplicate review classified their shared knowledge, operation, target, work product, and answer giveaway as a duplicate. It retained the MCQ and sought to replace the True/False question. The persisted final warning is consistent with that result.

**CONFIRMED:** Total submission-to-completion wall time was **1,211.412 seconds (20m 11.412s)**. The largest measured stage was Catalog compilation (225.200s). Repeated validation, repair, duplicate replacement, and rescans also consumed a substantial part of the run. LLM call durations sum to 2,927.603s, which must not be interpreted as wall time because calls overlapped.

## 2. Exam/Job Identification

| Item | Value |
|---|---|
| Job ID | `6f32fc76-1ce8-4492-837d-812aa687abdf` |
| Exam ID | `exam_1d3236e97400` |
| Document ID | `175ef217-a754-4e58-8bae-2bbb4841f824` |
| Request | MCQ 10, True/False 5, Word Problem 4, Equation 4; model 1; easy; Arabic |
| Job created (UTC) | 2026-10-24 20:30:44.308807+00:00 |
| Execution started (UTC) | 2026-10-24 20:30:44.360486+00:00 |
| Completed (UTC) | 2026-10-24 20:50:55.720750+00:00 |
| Dubai time | 2026-10-25 00:30:44.308807 to 00:50:55.720750 (+04:00) |
| Queue wait | 0.051679s |
| Submission-to-completion | 1,211.412s |
| Active execution | 1,211.360s |
| Job status / exam status | `completed` / `partial` |
| Final quality summary | `valid=21`, `invalid=2`, `missing=0`, `status=incomplete` |

The final shortage warning records exactly two missing MCQ slots: `m1_mcq_3` and `m1_mcq_10`. This identifies the live execution and distinguishes it from earlier test or offline records.

## 3. Root Cause of Missing `m1_mcq_3`

### Lifecycle

| Evidence | Finding |
|---|---|
| Initial planned assignment | Task `task_3746f8c5acd853c2`; base task `task_0f3215e1a5fd1900`; family `family_86efe3e2a0fb51f0`; variation `variation_3c3f8cdd63171e02`; group `variation_group_7b50f7e26750475d` |
| Initial operation / target | `graph_inequalities_find_feasible_region_vertices`; identify the feasible region and its vertices from a system of inequalities |
| Initial evidence | Chunk `dd5e5bb5-c1c8-5d2d-b738-125db608811e` |
| Difficulty Adjustment | Recorded as not needed for this slot |
| Final effective assignment | Task `task_91e26bc5cba54279`; variation `variation_d7dff0db0e2516a9`; group `variation_group_2573ab238e5c5062` |
| Final question ID | `model1_mcq_3` |
| Repair attempts | 3 persisted attempts |
| Final lifecycle | `planned=true`, `generated=true`, `repaired=true`, `validated=false`, `finalized=false`, `terminal_status=repair_failed` |
| Final failure | `invalid_or_extraneous_solution` |

**CONFIRMED root cause:** The final MCQ described hours paid at AED 6.85 and AED 11.00, with total hours and earnings, and selected an option corresponding to the intended system. Its solution certificate nevertheless asserted that *adding the equations* produced that option. The validation evidence says that inference is mathematically false: adding the two equations does not yield the proposed system. Therefore the certificate was invalid/extraneous even though the selected option itself was plausible.

**CONFIRMED downstream handling:** The slot underwent three repair attempts. The last recorded STEM repair requested updates to `question`, `options`, `correct_answer`, and `solution_certificate`. Its revalidation also recorded `unsupported_given`: the assigned source evidence concerned an undefined-curve claim and did not support constructing the wages/hours system. No valid repaired candidate remained, so final assembly excluded it.

**Conclusion:** The first demonstrated failure for the final candidate was validation of its mathematical certificate; the final repair also revealed an evidence-grounding problem. The evidence does not establish whether the original planner assignment or a later replacement first introduced the unsupported scenario, so that earlier attribution is **UNKNOWN**.

Relevant implementation locations:

* `app/online/validator.py:2795-2835` converts non-PASS verdicts into repair targets and treats unsupported required-given diagnoses as task/evidence-invalid replacement work.
* `app/online/exam_builder.py:2577-2604` classifies invalid slots and marks the assembled exam incomplete when there is a deficit.
* `app/jobs/tasks.py:528-545` logs invalid omissions and persists the result as `partial`.

## 4. Root Cause of Missing `m1_mcq_10`

### Lifecycle

| Evidence | Finding |
|---|---|
| Initial planned assignment | Task `task_d4a19d94cf1ace4a`; family `family_02aade85bbbcf1ff`; variation `variation_a2a41f25ac268e87`; group `variation_group_26fb3f89691340ed` |
| Initial operation / target | `construct_linear_cost_equation_from_context`; write a truck rental/drive cost equation |
| Initial effective givens | AED 100 fixed charge; AED 0.10 per kilometre |
| Initial evidence | Chunk `e733ea1d-969b-5bd4-8083-aba8c8391956` |
| Difficulty Adjustment | The initial adjustment history did not require adaptation for this slot |
| Final effective assignment | Task `task_d205b5526997275e`; variation `variation_2ed142b4894103e2`; group `variation_group_d32f2207e396bdd1` |
| Final question ID | `model1_mcq_10` |
| Repair attempts | 3 persisted attempts |
| Final lifecycle | `planned=true`, `generated=true`, `repaired=true`, `validated=false`, `finalized=false`, `terminal_status=repair_failed` |
| Final failure | `missing_required_given` |

**CONFIRMED root cause:** The final MCQ asked for the system representing the conditions for joining a team, but did not state the two membership conditions in its prompt. It nevertheless marked option D as correct, based on thresholds `x >= 5` and `y >= 2.0`. The General Validator rejected it as unanswerable because the required conditions were absent.

**CONFIRMED repair evidence:** At least one repair candidate subsequently preserved a structural error: the expected thresholds were `x >= 5` and `y >= 2.0`, while an option reversed them, leaving no correct answer. The third repair requested an options/answer/certificate correction. It did not produce a validated replacement. The worker logged the final reason at 20:50:55.720Z: the question referenced the conditions without stating them, while answer A/D asserted those conditions.

**Conclusion:** This MCQ was lost because the final generated/repaired prompt omitted indispensable givens. It was not removed by duplicate review. The exact point at which the final replacement task replaced the initial truck-cost assignment is visible as a changed final task identity, but the available audit payload does not timestamp every per-slot reassignment; the first stage responsible for that change is therefore **UNKNOWN**.

Relevant implementation locations are the same validation-to-repair and omission paths listed above, especially `app/online/validator.py:2812-2821`, which maps unsupported/missing-given diagnoses into replacement-oriented repair where the evidence indicates it is needed.

## 5. Unresolved Duplicate Group

**CONFIRMED:** Persisted `unresolved_duplicate_conflicts` contains one group:

| Field | Value |
|---|---|
| Group | `group_1` |
| Questions | `model1_mcq_5`, `model1_true_false_4` |
| Keep | `model1_mcq_5` |
| Replace | `model1_true_false_4` |
| MCQ task / variation | `task_5ffb6999da55dddc` / `variation_813aa8c2bf0e704d` |
| True/False task / variation | `task_5ffb6999da55dddc` / `variation_5f3e83f21235d1e0` |
| Classification | `duplicate` |
| Judge rationale | Same knowledge, operation, target, work product, answer giveaway; new reasoning false |
| Required distinction | operation, target, work product, reasoning |

No numeric similarity score was persisted in the unresolved-conflict object examined; it is **UNKNOWN** whether a score existed only in a transient screening record. The persisted Judge result is sufficient to establish that this is an adjudicated duplicate conflict, not merely an unreviewed embedding candidate.

**CONFIRMED:** The audit history shows repeated replacement activity involving `m1_true_false_4`, and the final unresolved object shows it remained the replacement target. The final warning records one unresolved possible duplicate group with the two correct question IDs, matching the persisted review metadata.

**CONFIRMED:** Neither missing MCQ belongs to this group. The duplicate conflict did not directly remove `m1_mcq_3` or `m1_mcq_10`.

Implementation evidence: `app/online/exam_builder.py:2609-2628` derives duplicate-quality metadata and adds the non-blocking unresolved-duplicate warning; `app/online/duplicate_pipeline.py:1451-1479` returns the unresolved conflicts and repair state.

## 6. Other Issues Found

| ID | Confidence | Finding | Impact |
|---|---|---|---|
| OI-01 | CONFIRMED | Two final MCQ slots had changed task/variation identities from their initial plan. The stored end-state does not retain a complete per-slot reassignment timeline. | Limits root-cause attribution for replacement selection and makes debugging harder. |
| OI-02 | CONFIRMED | Six total repair attempts were spent on only two slots, and duplicate repair ran in three scan/repair cycles while one group remained unresolved. | Materially increased time without restoring the missing questions or resolving the duplicate. |
| OI-03 | CONFIRMED | The final output correctly stayed available as a partial exam, but quality is incomplete (`21/23`) and one duplicate warning remains. | User receives usable output with visible limitations. |
| OI-04 | LIKELY | The initial plan/evidence contracts were insufficiently protected through late replacement/repair for these slots, since final failure evidence diverged from the initial planned scenario. | Candidate replacement may need stronger evidence and required-given checks before costly validation. |
| OI-05 | UNKNOWN | No persisted worker-fencing rejection, stale-checkpoint conflict, provider timeout, rate-limit event, or malformed-provider response was found in the reviewed audit trail. | Cannot prove absence outside retained logs/telemetry. |

No evidence shows either missing question had previously passed final validation and was subsequently removed by duplicate replacement. Both terminal lifecycle records say `repair_failed`.

## 7. Total Job Execution Time

All timestamps are UTC in database/audit records; Dubai is UTC+04:00.

* Submission to completion: **1,211.412 seconds** (20m 11.412s).
* Queue wait: **0.052 seconds**.
* Active processing: **1,211.360 seconds** (20m 11.360s).
* The worker log independently reports **1,211.42 seconds**, consistent with the database timestamps.

There is no evidence of a Celery takeover/restart during this execution. This conclusion is **CONFIRMED only for the reviewed audit and worker-log records**.

## 8. Stage-by-Stage Timing Table

Durations are elapsed intervals between recorded stage entry/completion audit events. They do not add to the job duration because model calls and stage bookkeeping overlap. `UNKNOWN` means no independently bounded timestamp pair was retained.

| Stage | Start UTC | End UTC | Wall seconds | Calls/attempts | Result |
|---|---|---:|---:|---:|---|
| Retrieval | 20:30:45.352 | 20:30:45.429 | 0.077 | 1 | completed |
| Catalog load/compile | 20:30:45.569 | 20:34:30.769 | 225.200 | 12 compiler calls | completed |
| Catalog finalization | — | — | UNKNOWN | included in compile checkpoint | completed |
| Planner/allocation | 20:34:30.950 | 20:34:31.098 | 0.148 | 1 | 23 slots planned |
| Plan validation/repair | — | — | UNKNOWN | no repair checkpoint | no persisted unresolved plan error |
| Difficulty adjustment | 20:34:31.357 | 20:36:04.460 | 93.103 | 12 calls | completed |
| Initial generation | 20:36:04.614 | 20:38:08.518 | 123.904 | initial generator batch | completed |
| Initial general validation | 20:38:08.786 | 20:39:00.371 | 51.585 | concurrent model calls | failures routed to repair |
| Initial STEM validation | 20:39:00.371 | 20:39:00.619 | 0.248 | concurrent/overlapped | completed checkpoint |
| Question repair round 1 | 20:39:00.631 | 20:39:32.369 | 31.738 | repair | completed |
| Revalidation round 1 | 20:39:33.810 | 20:40:10.562 | 36.752 | general + STEM | completed |
| Question repair round 2 | 20:40:10.580 | 20:40:31.655 | 21.075 | repair | completed |
| Revalidation round 2 | 20:40:32.095 | 20:40:40.638 | 8.543 | general + STEM | completed |
| Duplicate screen/Judge 1 | 20:40:40.655 | 20:41:09.496 | 28.841 | screening + Judge | conflict found |
| Duplicate replacement 1 | 20:41:10.058 | 20:43:09.682 | 119.624 | replacement | completed |
| Revalidation round 3 | 20:43:10.970 | 20:43:30.990 | 20.020 | general + STEM | completed |
| Question repair round 3 | 20:43:31.018 | 20:43:56.930 | 25.912 | repair | completed |
| Revalidation round 4 | 20:43:57.772 | 20:44:11.806 | 14.034 | general + STEM | completed |
| Duplicate screen/Judge 2 | 20:44:11.823 | 20:44:24.098 | 12.275 | screening + Judge | conflict persisted |
| Duplicate replacement 2 | 20:44:24.497 | 20:45:52.226 | 87.729 | replacement | completed |
| Revalidation round 5 | 20:45:52.860 | 20:47:35.176 | 102.316 | general + STEM | completed |
| Question repair round 4 | 20:47:35.192 | 20:48:02.105 | 26.913 | repair | completed |
| Revalidation round 6 | 20:48:04.833 | 20:48:34.156 | 29.323 | general + STEM | completed |
| Question repair round 5 | 20:48:34.183 | 20:48:57.251 | 23.068 | repair | completed |
| Revalidation round 7 | 20:48:59.345 | 20:49:13.328 | 13.983 | general + STEM | completed |
| Duplicate screen/Judge 3 | 20:49:13.354 | 20:49:33.284 | 19.930 | screening + Judge | conflict persisted |
| Duplicate replacement 3 | 20:49:34.798 | 20:49:47.226 | 12.428 | replacement | completed |
| Revalidation round 8 | 20:49:47.988 | 20:50:08.913 | 20.925 | general + STEM | completed |
| Question repair round 6 | 20:50:08.936 | 20:50:18.325 | 9.389 | repair | completed |
| Revalidation round 9 | 20:50:20.570 | 20:50:36.954 | 16.384 | general + STEM | completed |
| Final duplicate scan | 20:50:36.982 | 20:50:54.181 | 17.199 | screening + Judge | 1 unresolved group |
| Final assembly/persistence | 20:50:55.016 | 20:50:55.720 | 0.704 | 1 | partial exam persisted |

Stage accounting clarification: Catalog Finalization, Plan Validation, Planner Repair, and a standalone BGE-M3 Screening interval were not persisted with separate start/end events. Their durations are **UNKNOWN** (or **NOT EXECUTED** where the audit has no event or attempt), rather than zero. Duplicate Judge is included in the duplicate scan bounds and is separately represented by 22 ledger calls; Duplicate Rescan is represented by the subsequent scan events. Final assembly and persistence are both bounded by the `exam_assembled` event at 20:50:55.016-20:50:55.194 UTC and the `exam_saved` event at 20:50:55.194-20:50:55.720 UTC.

## 9. Detailed LLM Call Timings

The retained call ledger has **171 LLM call records**. Each row records `gpt-6-luna` through the `openai` provider, operation, timestamp, duration, token counts, and success/error state. The following is the complete persisted-operation accounting; individual-call start timestamps are not consistently stored separately from ledger timestamps, so exact per-call start/end pairs are **UNKNOWN**. Durations below are exact recorded call durations; call totals overlap and are not wall-clock totals.

| Operation | Calls | First recorded UTC | Last recorded UTC | Summed call seconds | Average / max seconds | Input / output tokens | Result |
|---|---:|---|---|---:|---|---:|---|
| Assessment compiler | 12 | 20:31:41.839 | 20:34:21.224 | 815.603 | 67.967 / 88.353 | 84,506 / 94,623 | success |
| Difficulty adjustment | 12 | 20:34:50.451 | 20:45:52.121 | 255.832 | 21.319 / 40.528 | 40,884 / 22,851 | success |
| Generator | 36 | 20:36:47.334 | 20:49:47.059 | 390.857 | 10.857 / 42.648 | 310,740 / 38,209 | success |
| STEM validator | 31 | 20:38:30.182 | 20:50:26.267 | 368.430 | 11.885 / 38.425 | 1,894,427 / 25,579 | success |
| General validator | 32 | 20:38:45.115 | 20:50:35.534 | 598.963 | 18.718 / 101.534 | 2,072,332 / 52,895 | success |
| Question repair | 9 | 20:39:04.366 | 20:50:18.067 | 55.152 | 6.128 / 11.383 | 219,611 / 4,648 | success |
| Duplicate Judge | 22 | 20:40:57.556 | 20:50:53.613 | 260.217 | 11.828 / 20.239 | 77,237 / 24,175 | success |
| Duplicate replacement | 17 | 20:39:10.623 | 20:48:48.090 | 182.549 | 10.738 / 18.304 | 1,080,897 / 14,865 | success |
| **All recorded calls** | **171** | 20:31:41.839 | 20:50:53.613 | **2,927.603** | — | **5,780,634 / 277,845** | success at transport layer |

The slowest single recorded call was a General Validator call: **101.534s**, recorded at 20:47:34.554 UTC, with 292,516 input and 10,886 output tokens. The transport-level success result does not mean the validated question passed; failures described above are semantic/contract verdicts, not provider failures.

## 10. Chronological Execution Timeline

* **20:30:44.309 UTC** — exam/job created.
* **20:30:44.360** — worker begins execution.
* **20:30:45.352–20:30:45.429** — retrieval completes.
* **20:30:45.569–20:34:30.769** — Assessment Catalog compiles.
* **20:34:30.950–20:34:31.098** — deterministic plan completes with 23 slots.
* **20:34:31.357–20:36:04.460** — difficulty adjustment completes.
* **20:36:04.614–20:38:08.518** — initial questions generate.
* **20:38:08.786–20:40:40.638** — initial validation, first repair, and two revalidation passes.
* **20:40:40.655–20:43:09.682** — duplicate scan/Judge and first duplicate replacement.
* **20:43:10.970–20:49:13.328** — five alternating revalidation/question-repair rounds.
* **20:44:11.823–20:49:47.226** — two further duplicate scans and two duplicate replacement cycles interleaved with repairs.
* **20:50:08.936–20:50:36.954** — final question repair and revalidation.
* **20:50:36.982–20:50:54.181** — final duplicate scan retains one unresolved group.
* **20:50:55.016–20:50:55.720** — final assembly excludes the two repair-failed MCQs and persists a partial exam.

Small intervals between checkpoints are normal orchestration/checkpoint intervals. The audit trail does not include enough process-level instrumentation to prove whether every larger gap was CPU work, queue scheduling, or unrecorded serialization; those details are **UNKNOWN**.

## 11. Performance Bottlenecks

1. **CONFIRMED: Catalog compilation** is the largest single stage at 225.200s and 12 compiler calls.
2. **CONFIRMED: repeated quality loops** dominate the remainder: three duplicate replacement cycles, six question-repair rounds, and nine revalidation intervals occurred after initial generation.
3. **CONFIRMED: General Validation** used the largest model input volume (2.07M tokens) and includes the slowest individual call (101.534s).
4. **LIKELY:** Earlier deterministic rejection of unsupported/missing-given candidates would avoid expensive generator, validator, repair, and duplicate work for unusable replacements. This is supported by the two terminal failures, but the trace cannot prove how many later calls it would save.
5. **UNKNOWN:** Whether stages were unnecessarily serial. The ledger shows overlapping operations but lacks queue/concurrency instrumentation for a conclusive throughput analysis.

## 12. Recommended Fixes (ranked by priority)

1. **High — validate required givens and evidence after every replacement candidate, before generation/repair.** Require all conditions necessary to select an answer to appear in the prompt or in an explicitly permitted grounded construction contract. This addresses `m1_mcq_10`.
2. **High — validate solution certificates with the same rigor as answers before accepting a repair candidate.** Reject certificates that assert a false derivation, even if the answer option happens to be right. This addresses `m1_mcq_3`.
3. **High — persist a per-slot reassignment history with timestamp, reason, prior/new task/variation/evidence, and validation result.** It will make the first invalid assignment detectable rather than only inferable from initial and terminal identities.
4. **Medium — stop or change strategy when repeated repair/replacement has no progress.** The trace shows multiple loops with no recovery for the two slots and one duplicate group. Preserve the current partial-exam behavior, but avoid costly repetitions of blacklisted failure patterns.
5. **Medium — add timing telemetry at node and per-call boundaries.** Store unambiguous call start/end, queue delay, concurrency identifier, and outcome class so future performance reports do not have to infer boundaries from checkpoint events.
6. **Medium — make duplicate replacement prefer a task family different from the retained question when the Judge requires operation/target/work-product/reasoning differentiation.** Both unresolved questions share the same base task identity, making the recorded Judge conflict structurally unsurprising.

## 13. Evidence and Verification Limitations

* **CONFIRMED:** Findings marked confirmed come from the authoritative PostgreSQL exam/job/audit records, the retained worker messages, and source locations inspected during this investigation.
* **LIKELY:** Findings marked likely are evidence-supported inferences about where earlier prevention would have occurred.
* **UNKNOWN:** No standalone per-slot event stream captured every reassignment, and LLM ledger timestamps do not uniformly provide distinct start and end values. Therefore this report does not claim an exact timestamp for every individual replacement or exact serialized model-call intervals.
* No new model call, duplicate adjudication, exam, test run, database write, or data restoration was performed. Mathematical observations are limited to the recorded validator evidence; this is not a fresh independent audit of every retained question.
