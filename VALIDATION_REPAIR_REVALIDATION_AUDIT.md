# Validation, Repair, and Revalidation Intelligence Audit

Investigation date: 25 October 2026 (Asia/Dubai). Scope: production source, prompts, Pydantic schemas, tests, PostgreSQL audit/checkpoint records, LLM usage rows, and retained worker logs for job `6f32fc76-1ce8-4492-837d-812aa687abdf` / exam `exam_1d3236e97400`. Catalog compilation is excluded from performance conclusions. This was read-only: no model call, exam run, database write, cache/checkpoint change, code/test change, commit, or push was performed.

Confidence labels:

- **CONFIRMED**: directly supported by source and/or live execution evidence.
- **LIKELY**: strongly supported but missing one historical detail.
- **UNKNOWN**: retained evidence cannot establish the claim.

## 1. Executive Summary

DraftWork's pipeline is substantially more intelligent than a simple “validate, ask an LLM to rewrite, retry three times” loop. The first pass validates the complete generated exam; General and STEM checks run concurrently; all failures are collected before repair routing; role-specific findings are merged; repair is field-authorized; incompatible task/evidence failures route to Catalog replacement; candidates are staged outside the active exam; only passing candidates are committed; unchanged questions reuse fingerprinted verdicts; and independent repairs run concurrently.

The 21/23 result came from weaknesses in strategy state, attempt accounting, and prompt cost rather than absence of these mechanisms.

The most consequential findings are:

1. **CONFIRMED — attempt budgets are per slot and survive task changes.** An attempt is consumed as soon as a repair target is processed, before candidate generation or revalidation. Provider failure, no-op/invalid patch, replacement-inventory failure, and candidate validation failure can all consume the same three-attempt budget. Replacing the task does not reset or partition the budget.

2. **CONFIRMED — `m1_mcq_3` had a passing replacement, then duplicate processing changed it again.** Attempt 2 produced task `task_5ffb6999da55dddc` and passed correctness validation. A later duplicate-replacement round changed the slot; the final active task became `task_91e26bc5cba54279`. The slot still had only attempt 3 available. That candidate's certificate was repaired, but revalidation found its wages/hours scenario unsupported by the assigned curve evidence. The slot exhausted its shared budget and was omitted.

3. **CONFIRMED — `m1_mcq_10` showed serial defect substitution.** Attempt 1 replaced the original assignment with `task_d205b5526997275e` and passed candidate validation. Later validation found that the prompt omitted membership conditions. Attempt 2 repaired the question but left/revealed an option-answer mismatch: the thresholds were reversed and no option was correct. Attempt 3 repaired options/answer/certificate but produced a stem that disclosed the complete answer. The defects changed, so a simple “same failure code” detector would not have caught this; a quality-progress model is required.

4. **CONFIRMED — no malformed or missing Validator verdict is silently accepted.** Strict schemas, exact ID checking, salvage, one-ID retry, and explicit `UNVALIDATED` operational outcomes fail closed. A timeout or failed call does not become PASS.

5. **CONFIRMED — selective revalidation works, but candidate validation is extremely expensive.** The live run had one initial validation plus nine graph revalidation rounds. Unchanged questions were generally skipped, yet one-question candidate checks sometimes used 177k-493k input tokens per round because each request repeats a large question/plan/evidence payload and structured retry can append the original prompt, previous output, failure, IDs, and full schema.

6. **CONFIRMED — live validation/revalidation cost dominated repair generation.** General + STEM used 3,966,759 input tokens. Initial validation used 238,298; the nine revalidation rounds used 3,728,461. Repairer calls used 219,611 input tokens. Graph validation wall time was 304.974 seconds (51.585 initial + 253.389 revalidation), while repair-node wall time was 138.095 seconds.

7. **IMPLEMENTED BUT INEFFECTIVE — failure memory is incomplete.** Later patch prompts receive the latest rejection string, and replacement selection receives candidate blacklists/failures. There is no per-attempt content fingerprint, patch fingerprint, normalized root-cause progression, or deterministic “strategy produced no net improvement” score. Only exact no-change patches are rejected.

The recommended design preserves the current graph and agents. Add deterministic root-cause/strategy state, partition attempt budgets by assignment generation and strategy class, fingerprint candidates and failures, compare each result with its predecessor, escalate patch -> regenerate same assignment -> replace assignment, and compact validation/repair payloads. No extra LLM agent is justified.

## 2. Current Validation Architecture

### 2.1 Execution path and coverage

The worker enters `generate_exam` at `app/jobs/tasks.py:363`, invokes `generate_exams` at `app/online/exam_builder.py:2373`, and executes the graph built at `app/online/graph.py:224`. Generation flows directly to `validate_generated_questions` (`app/online/validator.py:2946`). Graph routing does not inspect a partial batch result; it waits for the validation node to finish and then examines the complete `repair_manifest` (`app/online/graph.py:444`).

**CONFIRMED:** The first General pass checked all 23 generated questions. The STEM pass checked the 21 questions selected as quantitative by `_quantitative_question_ids` (`app/online/validator.py:2599`); the two non-quantitative True/False questions were General-only. Worker logs at 20:38:08.817/821 UTC list the exact scopes.

`validate_exam` (`app/online/validator.py:1668`) first performs deterministic local checks. Remaining entries are split into batches of `VALIDATOR_BATCH_SIZE`, currently 8 (`app/config.py:102`). The live initial General batches were 8/8/7 and STEM batches 8/8/5. Batches run concurrently up to `MAX_PARALLEL_VALIDATOR_CALLS`.

Every batch completes before the report is merged. `validate_generated_questions` then builds one actionable manifest target for every non-PASS verdict and every absent planned slot through `_manifest_from_report` (`app/online/validator.py:2792`). Repair begins only after the node returns. This is **already implemented and working**.

### 2.2 Information sent to each Validator

For each question, `validate_exam` sends:

- stable `question_id`, inferred question type, and complete current question payload;
- the current authoritative plan item for its slot;
- `expected_catalog_task`, built by `_catalog_task_contract` (`app/online/validator.py:1431`), containing task/family/coverage/concept identity, operation, target, work product, required/variable/effective givens, reasoning mode, difficulty/adaptation, selected variation/group, effective instance, and effective evidence IDs;
- assigned source text limited to the plan/effective evidence IDs via `_assigned_source_chunks` at line 1419;
- the FITB word bank when applicable.

**CONFIRMED:** The exact selected variation and effective instance are present when they exist. Validation does not receive unrelated source chunks as authorized evidence, although the same source text may be repeated once per entry in a batch.

`build_validator_prompt` (`app/online/validator.py:1371`) explicitly declares each entry an isolated evidence boundary. The General system prompt judges answerability, grounding, task/operation fit, wording, structure, options, and conceptual correctness. The STEM system prompt independently solves quantitative questions, checks results, units, every calculation step, MCQ uniqueness/binding, and the internal solution certificate.

### 2.3 PASS/FIX contract

The strict schema is `ValidatorVerdict` / `QuantitativeValidatorVerdict` (`app/llm/schemas.py:309`). Supported actions are:

- `PASS`
- `FIX_QUESTION`
- `FIX_ANSWER`
- `FIX_OPTIONS`
- `FIX_QUESTION_AND_ANSWER`

The provider-facing failure-code enum is:

- `pass`
- `incorrect_result`
- `incomplete_solution`
- `missing_units`
- `invalid_or_extraneous_solution`
- `unsupported_given`
- `missing_required_given`
- `visual_dependency`
- `question_type_operation_mismatch`
- `task_drift`
- `mcq_task_unsuitable`
- `ambiguous_or_structural`
- `other`

Deterministic/local pipeline findings also use internal codes such as `invalid_structure`, `missing_required_field`, `invalid_options`, `invalid_word_bank`, `malformed_question`, `evidence_mismatch`, `bad_evidence_assignment`, `wrong_task_identity`, `bad_fitb_answer_evidence`, and `missing_requested_slot`. These are consumed by repair classification even though they are not all provider output literals.

`_sanitize_verdict` at `app/online/validator.py:1567` reconciles `question_valid`, `answer_valid`, action, and fields. A claimed PASS with explicit valid FIX fields is retained as a repair request; inconsistent boolean/action combinations are normalized to a FIX action. STEM quantitative verdicts must include an independent solution and certificate-valid flag.

### 2.4 General/STEM disagreement

**CONFIRMED:** They can disagree. A question may PASS General and fail STEM, or vice versa. `_merge_role_reports` (`app/online/validator.py:2627`) stores every non-PASS role finding in `required_fixes`.

Selection rules:

- both PASS -> merged PASS;
- General FAIL, STEM PASS/absent -> General verdict;
- General PASS, STEM FAIL -> STEM verdict;
- both FAIL -> STEM verdict is the primary object, General is retained in `additional_general_issue`, `required_fixes` contains both, `validation_role="combined"`, and `fields_to_fix` is the union.

This is **already implemented and working**. The repair prompt receives `required_fixes`, so one role's finding is not discarded.

### 2.5 Malformed responses and failures

`validate_exam` validates the exact envelope and IDs. A malformed multi-question response may have independently valid rows salvaged. Missing/invalid IDs are retried individually once with a one-verdict schema. Remaining missing IDs become explicit operational `UNVALIDATED` results/coverage warnings; they are not converted to PASS. Unexpected or duplicate IDs are rejected.

Provider `request_structured` may itself retry up to three times. If it times out, fails, or remains schema-invalid, the batch path records warnings and unresolved coverage. No code path marks the question correct merely because the call failed. This is **already implemented and working**, though failure telemetry should make `UNVALIDATED` easier to distinguish from a content FAIL.

### 2.6 Persistence

Validator reports, role-specific reviews, exact reasons, expected fixes, repair manifests, candidate histories, and validation ledgers are checkpointed in `exam_audit_events`. The graph wrapper also emits `quantitative_validation_completed` events (`app/online/graph.py:154`). Final metadata persists repair and validation history through `app/jobs/tasks.py:484` onward.

## 3. Repair Decision Tree

The deterministic strategy classifier is `_repair_classification` (`app/online/validator.py:2772`). `_manifest_from_report` adds a special diagnostic rule: a `missing_required_given` becomes assignment replacement only when the reason/expected-fix text indicates unsupported/not-grounded/impossible evidence.

```text
Non-PASS or missing planned slot
|
+-- missing_requested_slot
|   +-- attempt 1: regenerate original assignment
|   +-- later attempt: replace assignment, then regenerate
|
+-- task/evidence-invalid code
|   task_drift, operation mismatch, visual dependency,
|   MCQ unsuitable, unsupported_given, evidence mismatch,
|   bad evidence/task identity/FITB evidence
|   +-- rebuild candidate pool through catalog_exhaustion_replacement
|   +-- apply hard compatibility/exclusion checks
|   +-- rank/select replacement; write staged plan
|   +-- regenerate exact slot from replacement
|
+-- missing required given
|   +-- diagnosis says unsupported/impossible -> replace assignment
|   +-- otherwise -> patch authorized question/given fields
|
+-- quantitative role or quantitative failure
|   incomplete/incorrect result, calculation, units,
|   invalid/extraneous certificate
|   +-- STEM repair patch
|   +-- expand dependent fields (e.g. MCQ options -> answer + certificate)
|
+-- structure/content failure
    +-- General or STEM repair selected by role/code/fields
    +-- exact fields_to_fix become the only schema fields

Every branch
|
+-- build candidate in a deep-copied exam/plan
+-- deterministic normalization/field-authority checks
+-- stage candidate outside active state
+-- General + STEM revalidation as required
    +-- all PASS: commit candidate and staged plan
    +-- any FAIL: keep active question/plan, record new reason,
        blacklist candidate identity where applicable, retry if budget remains
```

### 3.1 Actual strategy mapping

| Failure | Current behavior | Assessment |
|---|---|---|
| Wrong numerical answer/result | STEM patch; answer dependencies/certificate expanded | Already implemented |
| Wrong mathematical explanation/certificate | STEM patch with independent solve and certificate replacement | Implemented; live success not guaranteed |
| Missing required givens | Patch unless diagnosis text says evidence cannot support them; then replace | Intelligent but relies on wording heuristics |
| Unsupported evidence | Replace assignment | Already implemented |
| Wrong MCQ options/binding | Authorize options; dependency expansion adds correct answer and certificate | Already implemented |
| Invalid wording | General patch of question/statement only | Already implemented |
| Task/variation mismatch | Replace assignment | Already implemented |
| Duplicate | Separate Duplicate Replacement stage after correctness | Already implemented |
| Missing generated slot | Regenerate same assignment first; replacement after failure | Already implemented |

### 3.2 Replacement quality

Task replacement does reuse the Catalog's current compatibility/hard-filter path. `_candidate_pool` (`app/online/duplicate_pipeline.py:430`) repeatedly calls `catalog_exhaustion_replacement`, whose ranking uses the shared 0-100 candidate score in `app/online/assessment_catalog.py:842` and replacement ranking near lines 2879-2907. `_select_replacement_candidate` (`app/online/duplicate_pipeline.py:461`) then excludes current/attempted/blacklisted identities, checks question-type compatibility, considers active-exam collisions and previous failures, and selects from a bounded pool.

This is not a blind random replacement. However, the selector's secondary ranking and optional LLM choice can override the simple top candidate, and the live telemetry does not preserve every candidate score/rejection in an easy per-attempt table. Whether the globally best remaining candidate was always selected is **UNKNOWN**.

### 3.3 Repair prompt inputs and write safety

`repair_exam` (`app/online/validator.py:2232`) receives the exact current question, full plan item, assigned source chunks, reason, expected fix, all `required_fixes`, expected results, independent solution, first incorrect step, rejected distractors, authorized fields, and latest `previous_repair_rejection` when present.

The dynamic `repair_response_model` (`app/llm/schemas.py:678`) exposes only authorized fields. `_repair_fields_with_dependencies` (`app/online/validator.py:2186`) expands mathematical dependencies: changing MCQ options adds `correct_answer` and `solution_certificate`; changing equation/word-problem mathematical fields requires steps, final answer, and certificate. Returned patches are rejected for unknown/duplicate IDs, missing/extra fields, normalization failure, disallowed modifications, invalid FITB bank, retained rejected distractors, or byte-equivalent canonical content. The original active question is preserved.

Separate non-FITB repairs operate on isolated deep copies and cannot mutate one another. Coordinator merge is keyed by slot. FITB remains grouped because of its bank. Candidates are committed only after revalidation. This is **already implemented and working**.

## 4. Repair Attempt Accounting

### 4.1 What consumes an attempt

`MAX_QUESTION_REPAIR_ATTEMPTS = 3` at `app/online/validator.py:48`. `_repair_invalid_questions_serial` increments `question_repair_attempts[slot_id]` immediately after selecting an unresolved target (`app/online/validator.py:3393`), before candidate-pool selection, regeneration, patch generation, deterministic acceptance, or candidate revalidation.

Therefore one slot attempt is a graph-level repair opportunity, not exactly one LLM call or one completed cycle.

| Event | Consumes slot attempt? | Notes |
|---|---|---|
| Valid patch generated and passes revalidation | Yes | Slot resolves |
| Patch candidate fails revalidation | Yes | New failure saved |
| Replacement candidate fails revalidation | Yes | Identity blacklisted |
| Provider call fails/times out | Yes | Increment occurred before call |
| Provider internal schema retry | No additional slot attempt | Up to three provider requests remain inside one slot attempt |
| All provider retries fail schema | Yes | Candidate generation failed |
| Deterministic no-op/disallowed patch rejection | Yes | No staged candidate |
| Replacement inventory exhausted | Yes | Increment occurred before pool result |
| Replacement plan write fails | Yes | Attempt remains spent |
| Duplicate-replacement attempt | Separate duplicate budget | But subsequent correctness repair uses the existing slot correctness count |

Attempts are keyed only by `slot_id`, not question content, assignment generation, failure code, validator role, strategy, or task identity. Changing the task does not reset the count. This avoids infinite loops, but it conflates materially different recovery problems.

### 4.2 Use of prior failures

Patch attempts receive only the most recent rejection string through `previous_repair_rejection`, not a structured history of all attempts, content hashes, strategies, and mathematical deltas. Replacement selection receives richer `repair_replacement_memory`: attempted IDs, blacklisted IDs, candidate failures, and replacement count.

The candidate from a failed revalidation is saved in `repair_retry_seeds`, so the next attempt starts from that staged exam/plan rather than always reverting to the original. This permits progressive repair. It also means a chain can drift through successive candidate defects.

The same candidate task should not be selected again once blacklisted, but the same ineffective patch strategy can be retried because there is no patch/content fingerprint blacklist. Exact no-change patches are rejected, yet semantically equivalent wrong patches are not recognized.

### 4.3 Exhaustion behavior

At three attempts, graph routing no longer sends the target to repair (`app/online/graph.py:444`). The unresolved manifest/lifecycle becomes terminal `repair_failed`; final assembly excludes invalid questions. Partial-exam policy persists a shortage rather than fabricating a PASS.

Remaining Catalog tasks are not always tried because:

- a failure classified as patchable remains on patch strategy;
- every failed patch consumes the same budget that replacement would use;
- assignment replacement may occur late;
- replacement inventory may contain no compatible unused candidate after exclusions;
- duplicate replacement and correctness repair have separate decisions but share downstream slot state;
- there is no rule requiring exhaustion of all compatible Catalog candidates before terminal failure.

## 5. MCQ 3 Failure Reconstruction

Slot `m1_mcq_3`, public ID `model1_mcq_3`.

### Confirmed chronology

1. **Initial assignment:** task `task_3746f8c5acd853c2`, variation `variation_3c3f8cdd63171e02`, inequality-graph operation.
2. **Attempt 1 — replacement:** a correctness/evidence failure routed to `replace_task`. Candidate `task_91e26bc5cba54279` was generated but rejected as `unsupported_given`. The wages/hours system was internally consistent, but assigned evidence concerned an unbounded curve/intercepts and did not support that assessment concept.
3. **Attempt 2 — replacement:** candidate `task_5ffb6999da55dddc` passed correctness revalidation at 20:49:12.241 UTC. This is recorded as `replace_task`, status `accepted`.
4. **Duplicate phase changed the slot again:** the 20:49:13-20:49:47 duplicate scan/replacement cycle selected `model1_mcq_3` among changed questions. The subsequent selective revalidation at 20:49:48 checked `model1_mcq_3` and `model1_true_false_4`. The final effective assignment for MCQ 3 became `task_91e26bc5cba54279`, variation `variation_d7dff0db0e2516a9`.
5. **Attempt 3 — patch:** STEM diagnosed `invalid_or_extraneous_solution`: answer option A represented the intended system, but the certificate falsely said adding the equations produced it. The repair was authorized to change question, options, answer, and certificate. Candidate generation succeeded.
6. **Final candidate revalidation:** General/source validation returned `unsupported_given`: the repaired wages/hours question still was not grounded in its assigned curve evidence. With the slot count already at three, no fourth replacement was attempted. Final status: invalid, not finalized, `repair_failed`.

### Why it failed

The budget was not scoped to the current assignment. A passing attempt-2 correctness candidate was superseded by duplicate replacement, but the new task inherited two consumed correctness attempts. Only one repair remained for a fresh assignment. The final patch addressed the certificate symptom but could not repair an underlying task/evidence mismatch by editing mathematical fields.

### Was a better recovery available?

**CONFIRMED:** A correctness-valid candidate existed at attempt 2. It was later changed for duplicate reasons, so simply restoring it might reintroduce the duplicate conflict. **UNKNOWN:** The persisted trace does not prove that another non-duplicate, evidence-valid candidate remained after all exclusions. The system should have treated duplicate replacement as a new assignment generation with its own bounded correctness qualification budget, or rejected it before replacing the last valid active candidate.

## 6. MCQ 10 Failure Reconstruction

Slot `m1_mcq_10`, public ID `model1_mcq_10`.

### Confirmed chronology

1. **Initial assignment:** task `task_d4a19d94cf1ace4a`, variation `variation_a2a41f25ac268e87`, truck cost equation.
2. **Attempt 1 — replacement:** the system chose `task_d205b5526997275e`, variation `variation_2ed142b4894103e2`, and candidate revalidation accepted it at 20:40:09.115 UTC. The active question later exposed a new defect.
3. **Diagnosis before attempt 2:** General validation reported `missing_required_given`: the prompt referred to two team-membership conditions without stating them, while an answer relied on thresholds `x >= 5` and `y >= 2.0`. The repair strategy was `repair_question`, authorized field `question`; source/plan context evidently contained the conditions, so the system attempted restoration instead of task replacement.
4. **Attempt 2 candidate:** question patch generation succeeded. Revalidation returned `ambiguous_or_structural`: the intended conditions required `x >= 5` and `y >= 2.0`, but option D reversed the thresholds and no listed option was correct. The certificate contained the correct inequalities but pointed to D. The candidate stayed staged and the last validated active version was preserved.
5. **Attempt 3 candidate:** STEM repair was authorized to change `options`, `correct_answer`, and `solution_certificate`. Revalidation returned another `ambiguous_or_structural` failure: the new stem contained the complete correct answer and disclosed it unnecessarily.
6. **Exhaustion:** attempt count reached three. Final lifecycle retained `missing_required_given` as the headline failure in some merged metadata, while manifest history records the subsequent option and answer-giveaway failures. The slot was excluded.

### Why three attempts were insufficient

Each attempt did meaningful work, but correctness did not converge across the complete MCQ contract. Attempt 2 repaired answerability while leaving answer/options inconsistent. Attempt 3 repaired the answer binding but overcorrected the stem. The dynamic field schema followed each latest verdict, so it did not authorize a holistic self-check/edit of every coupled MCQ field until later. The STEM prompt requires consistency, yet the candidate still violated a General quality constraint.

The same exact mistake was not repeated. This is defect substitution. A no-progress mechanism must compare overall validity and root-cause sets, not only failure-code equality.

The assigned task/evidence apparently contained the membership thresholds because the Validator could identify them and the repair was not classified unsupported. Exact source text is retained in the checkpoint, but the audit does not expose a concise “given X found at source span Y” proof. Source-span telemetry is **missing**.

## 7. Revalidation and Fingerprint Behavior

### 7.1 Triggers and selected IDs

Revalidation is triggered by staged repair candidates or `pending_revalidation_ids` from duplicate replacement/balancing. Candidate validation always checks the candidate question in isolation before commit. Duplicate replacements return to the same validation node. Changed MCQ labels from post-repair balancing are explicitly revalidated.

After candidate handling, the active exam uses a ledger. The fingerprint at `app/online/validator.py:3139` hashes:

- the complete question object, including options, answer, steps, certificate, selected variation/evidence metadata;
- the complete current plan item, including task, variation, givens, difficulty, and evidence IDs;
- FITB `word_bank` as a section dependency.

Previously passed questions are excluded when the fingerprint matches and the ledger contains every required role. A previous FAIL may also be reused when unchanged; `tests/test_performance_optimizations.py:174` explicitly verifies this. Selective report merge retains unresolved slots so another slot's revalidation cannot erase them (`_merge_repair_manifest`, line 2870; `tests/test_repair_manifest.py:176`).

### 7.2 Role reuse and invalidation

General and STEM verdicts are stored separately under one fingerprint. A quantitative question requires both role entries; a non-quantitative question requires General. Any question or plan mutation changes the common fingerprint and invalidates both roles, even if only one role's dependency changed.

PASS becomes stale after question/answer/option/certificate/task/variation/evidence/plan/FITB-bank change. MCQ balancing changes option order, answer letter, and certificate mapping, so changed IDs are revalidated. FITB bank rebuild adds all affected FITB IDs. Duplicate replacement marks changed IDs pending. A global plan object change does not invalidate unrelated questions because each fingerprint uses only that slot's plan item.

Checkpoint resume reloads the ledger only under the same request fingerprint and ordered checkpoint revision. This is safe against cross-request reuse. **WEAKNESS:** the validation fingerprint does not explicitly include validator prompt version, schema version, model/reasoning configuration, or source text hash separate from plan IDs. If source content or validator rules changed while IDs/question/plan remained byte-identical across a resume, stale reuse is possible. Current job-scoped immutable checkpoints make this unlikely, but versioned role fingerprints are still needed.

## 8. Nine-Round Revalidation Analysis

The audit contains ten `validation_completed` events: initial validation plus nine revalidation nodes. “Questions checked” below is the union of logged IDs for the graph node; General/STEM scopes can differ. “Skipped” is relative to 23 generated questions. Calls and tokens are measured from `llm_usage` inside each audit interval. Summed call seconds overlap and are not shown as wall time.

| Round | Questions checked | Questions skipped | General calls | STEM calls | Wall time | Input tokens | Reason / assessment |
|---|---|---:|---:|---:|---:|---:|---|
| Initial | All 23 General; 21 quantitative STEM | 0 General / 2 STEM | 3 | 3 | 51.585s | 238,298 | Necessary complete first pass |
| 1 | 10 candidate IDs: MCQ 2/4/5/10, TF 3/4, Equation 3, WP 1/2/3 | 13 | 10 | 9 | 35.305s | 487,943 | Necessary candidate qualification; expensive one-question calls |
| 2 | Equation 3, MCQ 4 | 21 | 2 | 2 | 7.609s | 287,632 | Second repair candidates; necessary |
| 3 | MCQ 4/7, TF 4 | 20 | 1 | 1 | 18.892s | 265,606 | Duplicate replacements; necessary changed IDs only |
| 4 | MCQ 7 | 22 | 1 | 1 | 13.569s | 177,596 | Repair candidate; necessary |
| 5 | MCQ 3/7/10, TF 5, WP 1/2/3 (STEM checked 5) | 16 | 1 | 1 | 101.822s | 492,376 | Duplicate replacements; scope selective, request exceptionally slow/large |
| 6 | MCQ 3/7/10, WP 1/2 | 18 | 5 | 5 | 28.145s | 781,594 | Isolated repair candidates; necessary but payload cost excessive |
| 7 | MCQ 3/10, WP 1 | 20 | 2 | 2 | 12.895s | 494,988 | Isolated next-attempt candidates; necessary |
| 8 | MCQ 3, TF 4 | 21 | 1 | 1 | 19.842s | 188,014 | Third duplicate replacements; necessary changed IDs only |
| 9 | MCQ 3, TF 4 | 21 | 2 | 2 | 15.310s | 191,604 | Final repair candidates; necessary |

The scopes were selective; the nine rounds were not nine full-exam rechecks. Most were logically necessary because new candidates or replacements had to be validated. The waste was primarily payload amplification, repeated candidate churn, and allowing new assignments to inherit nearly exhausted budgets.

Worker log scope counts across both roles/candidate calls show the most repeatedly checked IDs: MCQ 3 (12 role-scopes), WP 1 (12), TF 4 (10), WP 2 (10), MCQ 7 (9), MCQ 10 (8), MCQ 4 (8), and WP 3 (8). Counts include separate General/STEM checks and are not unique graph rounds.

## 9. No-Progress and Strategy Escalation

### 9.1 Existing mechanisms

Already implemented:

- exact canonical no-change patch rejection;
- field-authority and dependency checks;
- per-slot attempt bound;
- failed replacement candidate IDs and reasons;
- blacklist of replacement identities;
- retry seeds that continue from the latest failed candidate;
- failure-code reclassification after candidate revalidation;
- assignment replacement for task/evidence-invalid diagnoses;
- transactional candidate staging.

### 9.2 Missing or ineffective mechanisms

There is no persisted fingerprint for each attempted question/patch and no comparison of normalized defects across attempts. The code does not track:

- same semantic question with cosmetic changes;
- same mathematical error under different prose;
- failure-set severity or whether it shrank;
- root cause versus downstream symptom;
- strategy class and whether that strategy already failed on the same assignment;
- assignment generation/version for budget purposes;
- total invalid-question count trend across a repair round.

Strategy escalation is only failure-code driven. A patchable code can consume all remaining attempts even when two patches merely exchange defects. A duplicate replacement can create a new assignment without a fresh correctness qualification allowance. Fixing one question generally cannot mutate another because candidates are isolated, except for intentional cross-question dependencies: FITB bank, MCQ balancing, and duplicate relationships.

### 9.3 Recommended no-progress mechanism

Persist a per-slot `RepairAttemptRecord`:

- assignment generation ID and task/variation/evidence fingerprint;
- question content fingerprint and canonical mathematical fingerprint;
- incoming root-cause set, role findings, severity, and authorized fields;
- strategy (`patch`, `regenerate_same_assignment`, `replace_assignment`);
- patch/candidate fingerprint;
- deterministic acceptance failures;
- outgoing root-cause set and whether it improved, changed laterally, or regressed;
- provider/schema attempt information and token/time usage.

Use deterministic policy:

1. Reject identical candidate fingerprint without spending a content attempt; count it as provider/strategy failure separately.
2. If the same normalized root cause remains after a patch, do not repeat the same field/strategy; regenerate the same assignment or replace it according to evidence feasibility.
3. If the root cause changes but severity does not decrease (MCQ 10), allow one coupled-field repair with a complete deterministic MCQ consistency precheck; then escalate.
4. If task/evidence is unsupported, skip patching and replace immediately.
5. When duplicate replacement changes assignment fingerprint, create a new bounded qualification budget while retaining a global slot cap to prevent infinite loops.
6. Before consuming a replacement attempt, run deterministic source/givens feasibility and candidate hard filtering.
7. Stop only when strategy-specific budgets and compatible candidate inventory are exhausted; never weaken validation.

This needs no new LLM agent.

## 10. Repair Prompt and Schema Analysis

### 10.1 Strengths

The General and STEM repair system prompts are explicit. They state what fields may change, what must remain unchanged, source isolation, type contracts, MCQ uniqueness, distractor rules, equation/word-problem boundaries, independent mathematical verification, and certificate consistency. The user prompt includes exact failure reason, expected fix, required findings, plan, evidence, and current question. The schema permits only the authorized patch keys.

The STEM prompt explicitly says to treat expected results as hypotheses, solve independently, verify by a second method where practical, ensure one MCQ option, and synchronize answer/certificate. This is intelligent and quality-preserving.

### 10.2 Weaknesses and contradictions

1. **Field authorization can lag coupled defects.** The schema is generated from the latest Validator's fields. Dependency expansion covers many quantitative fields, but a General-only `question` patch can make options/answer stale while the schema forbids updating them. MCQ 10 attempt 2 demonstrates this sequence.

2. **The repairer sees one latest rejection string, not structured full history.** It cannot reliably detect that it is trading one defect for another or repeating a semantic strategy.

3. **Replacement and patch context are asymmetric.** Replacement selection gets prior candidate failures/blacklists; patch repair gets much less historical structure.

4. **Prompt/schema retry amplification.** Provider corrective retry appends the full original prompt, prior output, error, expected IDs, and full JSON Schema (`app/llm/deepseek.py:330`). Input can grow drastically even for one question.

5. **Metadata duplication.** Validator payload includes both full `plan` and a largely duplicated `expected_catalog_task`, plus source text in each entry. Repair includes full plan and source again. This is unnecessary repetition, not stronger validation.

6. **No deterministic preflight for common MCQ failures.** Before paid revalidation, code could check that the stored answer exists, certificate mapping matches it, options are distinct, and simple threshold/equation consistency holds where deterministically parseable. It must not replace STEM validation, but it can reject obvious regressions.

The schema does not generally prevent necessary quantitative coupled changes because `_repair_fields_with_dependencies` expands them. The demonstrated gap is cross-role/cross-field diagnosis: a narrow General repair may need a subsequent quantitative consistency update that was not authorized in the same attempt.

## 11. Performance and Token Bottlenecks

### 11.1 Measured totals

| Category | Calls | Input tokens | Output tokens | Summed call duration | Graph wall time |
|---|---:|---:|---:|---:|---:|
| General Validator, all passes | 32 | 2,072,332 | 52,895 | 598.963s | overlaps STEM |
| STEM Validator, all passes | 31 | 1,894,427 | 25,579 | 368.430s | overlaps General |
| Initial validation | 6 | 238,298 | 23,264 | 223.931s | 51.585s |
| Nine revalidation nodes | 57 | 3,728,461 | 55,210 | 743.462s | 253.389s |
| Repairer | 9 | 219,611 | 4,648 | 55.152s | included in six repair nodes |
| Six repair nodes | n/a | includes repairer plus replacement generation | n/a | n/a | 138.095s |

“Total Validation time” as graph wall time is 304.974 seconds for the initial plus nine validation nodes. General and STEM summed call durations must not be added as wall time because they overlap. Repair-node wall time includes candidate selection/regeneration and orchestration, not only the nine `repairer` calls.

At least 23 unique questions were checked by General; 21 were checked by STEM. Ten questions failed the initial validation according to persisted eval statistics. The final two missing slots each consumed three correctness repair attempts. The live metadata also records 17 duplicate-replacement model calls and multiple failed candidate replacements; duplicate work is separate from the nine repairer calls.

### 11.2 Measurable waste

Unchanged whole-exam revalidation was largely avoided. Exact “tokens wasted on unchanged questions” cannot be computed because provider usage is per request and the retained prompt text/token attribution per component is absent. The measurable inefficiency is that isolated one-question rounds consumed extraordinary input totals: 177,596 to 494,988 tokens in several rounds, and round 6 used 781,594 for five candidate IDs checked by both roles.

Confirmed repeated payload components are full question/certificate, full plan, duplicated expected task contract, full assigned source chunks, system prompt, output examples, and JSON Schema. Corrective structured retries add the complete prior output and schema again. Retry input therefore can become larger on later provider attempts.

The most expensive unnecessary operations were:

1. corrective retries with amplified validator payloads;
2. revalidating replacement candidates that should have failed a deterministic evidence/givens preflight;
3. spending a scarce slot attempt before candidate feasibility/generation succeeds;
4. duplicate replacement introducing a new correctness problem after a prior candidate had passed;
5. serial defect substitution without a whole-MCQ deterministic consistency gate.

## 12. Confirmed Bugs and Design Weaknesses

| Status | Severity / priority | Current vs expected | Root cause and evidence | Correctness/time impact | Recommended change |
|---|---|---|---|---|---|
| Implemented but ineffective | Critical / P0 | One per-slot budget spans multiple task identities; expected budget tied to assignment/strategy | Increment at `validator.py:3393`; MCQ 3 attempt-2 pass then duplicate reassignment then attempt-3 terminal failure | Valid recovery can be abandoned after one chance on a new task; extra loops | Partition by assignment fingerprint plus global hard cap |
| Missing | High / P0 | No semantic no-progress detection; expected root-cause and candidate comparison | Only exact canonical no-op rejected in `repair_exam`; no attempt fingerprint ledger | Defect substitution/repeated strategy wastes attempts/tokens | Add structured attempt record and progress classification |
| Implemented but ineffective | High / P0 | Latest field list can be too narrow for coupled MCQ repair | Dynamic schema at `schemas.py:678`; MCQ 10 question-only patch created option inconsistency | New errors and extra attempts | Deterministic dependency closure across General+STEM MCQ contract |
| Missing | High / P0 | Attempts consumed before feasibility/candidate success | Increment precedes pool/generation in `_repair_invalid_questions_serial` | Timeouts/schema/no-candidate spend quality budget | Separate provider attempts, strategy attempts, and validated-candidate attempts |
| Implemented but ineffective | High / P1 | Replacement is ranked/blacklisted but post-duplicate candidates can displace valid questions before correctness qualification | Duplicate graph returns replacements to validation; MCQ 3 chronology | Previously valid question lost | Transactionally keep prior candidate until replacement passes correctness and duplicate checks |
| Missing | High / P1 | Validation fingerprints omit explicit validator/prompt/schema/model/source-content versions | Fingerprint at `validator.py:3139` | Potential stale reuse after behavior change/resume | Versioned per-role dependency fingerprints |
| Implemented but ineffective | High / P1 | Selective revalidation scopes are correct but payloads are huge | Live nine rounds: 3.728M input tokens | Dominant cost and latency | Canonical compact task contract, shared evidence registry, bounded retry payload |
| Already working | — | All initial failures collected before repair | Graph/node behavior and logs | Preserves coordinated decisions | Keep |
| Already working | — | Failed candidates are staged and active version preserved | `validator.py:2979-3117`; `test_repair_manifest.py:120` | Prevents bad patch publication | Keep |
| Already working | — | General/STEM disagreement retained | `_merge_role_reports` | Preserves mathematical and grounding findings | Keep |
| Cannot verify | Medium / P2 | Whether all remaining eligible Catalog candidates were genuinely exhausted | Candidate-score history incomplete in audit | Could hide avoidable terminal failures | Persist ranked candidate inventory and rejection reasons |

## 13. Proposed Improvements

Preserve the current graph and agents:

1. Add a deterministic `RepairStrategy` enum and complete mapping from normalized failure/root-cause combinations.
2. Store assignment generation/fingerprint separately from question fingerprint.
3. Count three different budgets: provider/schema calls, generated candidates, and validated strategy attempts. Only a candidate that reaches deterministic acceptance/revalidation should consume the main quality budget.
4. Add per-attempt question, mathematical, patch, plan, evidence, and failure fingerprints.
5. Compare incoming/outgoing defect sets and severity. Mark `improved`, `lateral`, `regressed`, or `unchanged`.
6. Escalate deterministically: targeted patch -> full same-assignment regeneration -> replacement assignment. Jump directly to replacement for unsupported task/evidence.
7. Run source/givens feasibility before an LLM repair or replacement commit.
8. Expand authorized fields using the union of General/STEM dependencies and a question-type consistency closure.
9. Add cheap deterministic MCQ/equation/certificate preflight checks before paid revalidation, while retaining full Validators.
10. Preserve the old valid active question and duplicate state until a replacement passes correctness and affected duplicate checks.
11. Compact prompts: one canonical plan contract, one request-level evidence registry, source hashes/spans, and bounded corrective retries.
12. Version General and STEM fingerprints independently with prompt/schema/model/source-content versions.
13. Persist ranked replacement inventory, 100-point score components, exclusions, and selector outcome for every attempt.
14. Add round-level telemetry: invalid count before/after, changed IDs, strategy transitions, candidate fingerprint, prompt tokens, retry count, and no-progress reason.

## 14. Prioritized Implementation Plan

### P0 — correctness and recovery

1. Refactor attempt accounting in `app/online/validator.py` so failed transport/schema/no-candidate work does not consume a validated-candidate attempt.
2. Add assignment-scoped budgets and a global slot cap; reset/allocate bounded qualification attempts after task/variation/evidence replacement, including Duplicate Replacement.
3. Add `RepairAttemptRecord` and deterministic progress comparison.
4. Add cross-role field dependency closure and deterministic MCQ consistency preflight.
5. Make replacement commit transactional across correctness and affected duplicate checks.

### P1 — performance and safe reuse

6. Compact `build_validator_prompt` and `build_repair_prompt` payloads; remove duplicated plan/task metadata and deduplicate source chunks.
7. Bound structured corrective retries in `app/llm/deepseek.py`; retry unresolved IDs with compact one-question prompts rather than appending huge prior envelopes.
8. Version role-specific validation fingerprints and persist source-content hashes.
9. Persist complete candidate ranking/exclusion telemetry.

### P2 — observability and deterministic checks

10. Add deterministic checks for answer-option-certificate binding, option uniqueness, stated required givens, parseable arithmetic/equation consistency, and FITB bank invariants.
11. Add tests for MCQ 3's valid-then-duplicate-replaced lifecycle and MCQ 10's defect-substitution sequence.
12. Add round-level tests proving all failures are collected, independent repairs overlap, unchanged verdicts reuse safely, strategy escalates on no progress, and exhaustion occurs only after applicable strategies/candidates are tried.

Feature-flag attempt-policy and prompt-compaction changes independently. Keep the current behavior as rollback while comparing complete-slot rate, validator agreement, token use, p95 time, and regression rate on frozen fixtures and controlled provider runs.

## 15. Evidence and Verification Limitations

- PostgreSQL checkpoints preserve end-state and accumulated histories, but not a minimal immutable snapshot of every prompt, patch diff, candidate score, and source span. Some per-attempt field deltas are therefore **UNKNOWN**.
- The MCQ 3 chronology proves a correctness-valid attempt-2 candidate was later superseded; it does not prove that candidate was duplicate-safe or should have been restored.
- Exact token attribution to question text, plan metadata, evidence, system prompt, schema, and retry additions is unavailable. Token waste by component is inferred from confirmed prompt construction and measured request totals.
- Worker logs provide question scopes and call timing, but no durable role label on every `Validation scope` line. Round unions and call counts were reconciled with `llm_usage` and audit intervals.
- No fresh mathematical audit, Validator call, repair call, duplicate Judge call, or benchmark was run.
