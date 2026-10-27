# DraftWork v13 Paid Exam Audit

## Executive Summary

**Verdict: FAIL.** Exactly one paid job was started: job `0156d649-64e7-4c79-825b-6a3833a0dd72`, exam `exam_00bb0f90334c`. It reused the 103 stored chunks and migrated the compatible Catalog v12 artifact to v13 without compiler calls. Generation produced all 23 questions, but the job entered repeated repair and selective-validation cycles and was cancelled at the safety boundary before final assembly.

The run spent 104 recorded provider calls, 1,120,394 input tokens, 111,721 output tokens, 64,465 reasoning tokens, 86,238 cached tokens, and `$0.160143` recorded cost. The persisted exam has zero final questions because assembly had not run when cancellation was recorded. No second paid run was launched.

## Configuration and Environment

- Document: `math (2)2_compressed.pdf`, ID `175ef217-a754-4e58-8bae-2bbb4841f824`.
- Processed-document key: `772d2717699ca1dc2b015371744fb5d9df8f50751dc7a80b8eb63d716697f84e`.
- Source scope: 103 authoritative child chunks; retrieval selected 103.
- Requested distribution: 10 MCQ, 5 True/False, 4 equations, 4 word problems.
- Language: Arabic; subject: mathematics; difficulty: easy; models: 1.
- Docker API, worker, PostgreSQL, Redis, and Qdrant were healthy before launch.
- Repository/API/worker hashes matched for the modified production files.
- Generation scheduler ceiling: 50.

## Catalog Reuse

Worker logs recorded `Assessment catalog compatible cache migrated`, key `c5adfb5717dd25687349d687b4dcd82657236a333a63363207e5b57d582ea57a`, followed by `Assessment catalog migrated ... tasks=97`. Retrieval and Catalog loading took place without paid Catalog compilation. This confirms deterministic v12-to-v13 migration and reuse for this run. The effective Catalog was allocated to 23 slots with zero allocation issues or errors.

## Timing and Timeline

| Event | Timestamp |
|---|---|
| Job created | 2026-10-26 17:54:23.396862 UTC |
| Worker started | 2026-10-26 17:54:30.396921 UTC |
| Retrieval completed | 17:54:33.819 UTC |
| Catalog migration completed | 17:54:34.624 UTC |
| Difficulty adjustment began | 17:54:35.072 UTC |
| Difficulty adjustment completed | 17:56:34.202296 UTC |
| Generation completed | 17:57:03.052728 UTC |
| Initial validation began | 17:57:03.526 UTC |
| First repair cycle | 17:57:58–17:58:23 UTC |
| Later repair/validation cycles | 17:58:23–18:03:47 UTC |
| Job cancelled for safety | 18:03:42.268255 UTC |

Submission-to-cancellation wall time was approximately **558.871 seconds**; worker-start-to-cancellation was approximately **551.871 seconds**. This is not a successful end-to-end runtime because final assembly and persistence never completed.

Observed stage behavior:

- Retrieval: selected 103/103 chunks.
- Catalog: compatible migration, 97 tasks.
- Planning: 23 slots allocated, zero issues.
- Difficulty Adjustment: multiple concurrent calls; v13 scheduler limit was 50.
- Generation: 23 questions produced; worker logged `total=23`.
- Initial validation: all 23 entered validation using parallel General/STEM batches.
- Repair: repeated targeted repairs and replacement generation.
- Final assembly: NOT EXECUTED.
- Final persistence: NOT EXECUTED; the job was cancelled.

## Generation and Concurrency

The worker logged concurrent scheduler admissions for Difficulty Adjustment, Generator, Validators, Repairer, and Duplicate Replacement with `limit=50`. Generation completed 23 questions in approximately 28.43 seconds after the generation phase began. However, the live logs still show schemas named `mcq_true_false_bundle` and `equation_questions`; the logs establish successful one-slot acceptance records, but do not by themselves prove that every internal compatibility path is free of bundled schema semantics.

## Validation and Repair Findings

Initial validation covered all 23 generated questions. The first persisted summary recorded 17 passes and 6 failures: MCQ failures were one `bad_options` and two `unclear_or_invalid_question`; equation failures were two `unclear_or_invalid_question`; the True/False failure was one `wrong_answer`.

The critical live failure was `m1_true_false_1` / `model1_true_false_1`:

1. A repair candidate was generated with `repair_question`.
2. Revalidation returned no schema-valid verdict, classified as lateral progress.
3. A task replacement was selected (`task_60b654b7aaefa187`); validation reported a NUL/control-character problem in the statement.
4. Another replacement (`task_26cd3df651f64aef`) again produced no schema-valid verdict.
5. A further replacement (`task_7b16c0cc2843b240`) was rejected for an incomplete solution certificate and task/operation mismatch.
6. The persisted repair manifest marked the slot `exhausted` at attempt 4 with `replacement_inventory_exhausted`.
7. Despite that terminal manifest state, the graph continued to schedule additional validation/replacement work. The job was still running at 95% when safety cancellation occurred.

This is a confirmed lifecycle defect: an exhausted slot remained reachable through later validation-update routing. It also confirms that the NUL-character issue remains possible in real provider output even after the v13 changes; it was detected by validation in this run rather than causing a JSONB crash.

Other observed repair actions included `FIX_OPTIONS` for `model1_mcq_6` and `FIX_ANSWER` for `model1_true_false_1`. The audit folder contains the complete persisted manifests and node outputs available before cancellation.

## Catalog Numerical-Givens Regression

The job log proves that the migrated v13 Catalog was used, but the cancelled run never reached final exam export. The effective Catalog object and all available assignment/lifecycle records are in `exam_forensic_v13_0156d649-64e7-4c79-825b-6a3833a0dd72/04_planner_and_assignments.json` and `06_slot_lifecycle.json`. A final student-facing dinner-tip comparison is **UNKNOWN** because no final exam was assembled.

## Final Question Quality and Diversity

All 23 slots had generated candidates before validation. There are no final student-facing questions to independently audit, and no valid display-number-to-question mapping because `exams.exams` remained an empty array. Therefore the dinner-tip duplicate regression, final source grounding, final difficulty distribution, and final unresolved duplicate groups are **UNKNOWN**, rather than being treated as passed.

## Comparison With Reference Job

| Metric | v12 reference | v13 run | Finding |
|---|---:|---:|---|
| Requested | 23 | 23 | same |
| Final questions | 23 | 0 persisted | v13 failed before assembly |
| Runtime | 490.674 s | 558.871 s to cancellation | not comparable as successful runtime; v13 exceeded reference before stopping |
| LLM calls | 98 | 104 | v13 exceeded reference while incomplete |
| Input tokens | 702,281 | 1,120,394 | higher |
| Output tokens | 122,660 | 111,721 | lower, but incomplete |
| Reasoning tokens | 74,202 | 64,465 | lower, but incomplete |
| Recorded cost | $0.127378 | $0.160143 | higher |
| Catalog | compatible v12 reuse | v12→v13 migration | reuse path worked |
| Exact duplicate result | Q3/Q9 | UNKNOWN | no final assembly |

The comparison is limited because the v13 job was cancelled before finalization and used a different live state path with multiple repair cycles.

## Successful Checks

- Docker health and source parity passed before launch.
- Correct document identity and 103-child scope were used.
- Compatible Catalog reuse/migration succeeded without compiler calls.
- Planner allocated all 23 requested slots.
- Generation produced 23 candidates.
- Initial validation included every generated question.
- LLM admission remained bounded by the configured 50 ceiling.
- Audit events, checkpoints, slot lifecycle, and LLM usage were persisted.
- No PostgreSQL JSONB NUL crash occurred in this run.

## Failed Checks and Remaining Risks

- **P0:** Exhausted repair manifest did not terminate graph routing; the job continued paid validation and replacement calls.
- **P0:** A real provider candidate contained NUL/control characters; persistence and validator sanitization must be enforced before candidate acceptance.
- **P1:** Several validator responses lacked schema-valid verdicts, causing corrective single-question calls and repeated revalidation.
- **P1:** Replacement and repair work continued after the affected slot was already terminal.
- **UNKNOWN:** Final duplicate screening, final duplicate warnings, final question quality, and 23/23 persisted output because final assembly was never reached.

## Forensic Artifacts

Raw and derived evidence is in [exam_forensic_v13_0156d649-64e7-4c79-825b-6a3833a0dd72](C:/Users/abdul/PycharmProjects/GenExam/exam_forensic_v13_0156d649-64e7-4c79-825b-6a3833a0dd72/), including the job row, empty final exam, 30 audit events, planning/difficulty state, generation and validation node outputs, 23 slot lifecycle records, 104 LLM-usage rows, repair manifests, and availability limitations.

## Highest-Priority Follow-Up

1. Make routing derive terminal repair state from the authoritative manifest and remove exhausted slots from all validation-update and replacement queues.
2. Reject or safely normalize NUL/control characters before validator calls and JSONB persistence, while preserving mathematical content.
3. Treat missing validator verdicts as a bounded terminal validation failure for the current candidate, then escalate or finalize partial rather than re-entering the same repair path.

No paid retry was run after cancellation.
