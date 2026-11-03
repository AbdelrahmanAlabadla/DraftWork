# DraftWork live-exam remediation review

## Scope and evidence

This review is an investigation and implementation plan. It does not change production code, PostgreSQL data, checkpoints, model routing, or provider configuration, and it made no LLM requests.

The conclusions below are based on the current production source, the 20 continuation checkpoints, and the forensic artifacts for jobs `d12480df-9961-4c0c-93a2-64f86495a33b` and `dce90847-63df-4017-a3c7-f43fb1e420db`. The prior Markdown reports are currently under `all_markdown_files/`; the structured evidence remains in `live_exam_replay/`.

The review also ran the existing provider-contract, repair, duplicate, lifecycle, and contract-regression tests. The selected set passed: **102 passed, 0 failed, 0 skipped**. A direct schema probe confirmed that `SpecializedValidatorResponse` has no empty schema nodes, while `SpecializedRepairResponse` still normalizes to an untyped `repairs.items = {}` node.

## Review matrix

| Problem | Current behavior | Status | Proposed change | Files affected | Tests required | Priority |
|---|---|---|---|---|---|---|
| 1. Final Assembly Integrity | Assembly filters by the latest Validator verdict, while shortage reporting uses lifecycle terminal state. Persistence and exports consume the assembled arrays. | **Broken** | Compute one authoritative accepted-slot set, filter once, assert persisted/exported IDs match it, and preserve excluded-slot audit data. | `app/online/exam_builder.py`, `app/online/lifecycle.py`, `app/jobs/tasks.py`, `app/api/export_routes.py`, export helpers | Graph, persistence, API, PDF/DOCX, legacy-checkpoint tests | **P0 Critical** |
| 2. JSON Schema Compatibility | Most production contracts are typed. Specialized batch Repair sends `list[object]`, which becomes an OpenAI-invalid empty item schema; exact validation occurs only after the call. | **Broken; audit incomplete** | Replace the generic repair envelope with the exact production patch model before dispatch. Add a registry-style offline audit covering every structured call and both provider normalizers. | `app/llm/schemas.py`, `app/online/multi_repair.py`, provider-client contract tests, stage schema modules | Full schema inventory, mocked OpenAI/DeepSeek, malformed output and provider error tests | **P0 High** |
| 3. Current Status and stale failures | Lifecycle records, Validator reports, repair manifest, ledger, invalid list, and candidate state can disagree. Candidate commit is transactional, but assembly does not consume the lifecycle acceptance result. | **Partially fixed** | Make lifecycle/current verdict the canonical current state; update it atomically at validation, staging, revalidation, commit, exhaustion, and operational block boundaries. Keep history append-only. | `app/online/lifecycle.py`, `app/online/validator.py`, `app/online/multi_repair.py`, `app/online/exam_builder.py`, graph state contract | State-transition matrix, stale PASS/FAIL, checkpoint restart, candidate isolation | **P0 High** |
| 4. Quota, rate limits, provider availability | SDK 429s become one generic rate-limit exception. Only `Retry-After` is retained transiently. Scheduler is a per-job concurrency gate and has no provider/account adaptive state. | **Broken classification; basic bounded retry exists** | Normalize provider errors, stop an affected provider/account on confirmed insufficient quota, adapt cooldown to response guidance, and preserve diagnostic metadata. | `app/llm/deepseek.py`, `app/llm/openai_client.py`, `app/llm/scheduler.py`, provider factory/config, graph error handling | Provider-specific 429 fixtures, cooldown/circuit tests, concurrent scheduling, checkpoint block tests | **P1 High** |
| 5. Catalog Replacement context | Deterministic filtering exists, but the Selector receives duplicated full exam context and recursively retained prior contexts. Candidate limit is 12, and telemetry is mislabeled `duplicate_replacement`. | **Broken** | Remove recursive fields, compact exam/failure metadata, rank by compatibility and failure relevance, send top-five windows, and validate the choice against the full catalog. | `app/online/duplicate_pipeline.py`, `app/online/validator.py`, catalog-binding utilities, usage labels | Token ceiling, no-recursion, ranking/windowing, no-repeat, full-binding verification | **P1 High** |
| 6. Retry accounting | Revalidation has separate operational counters and multi-repair has tests protecting content attempts. The serial recovery path still counts failed candidate production in `candidate_generation_failures` and can exhaust recovery after integration/provider errors. | **Partially fixed** | Define one operational classification shared with Problem 4; increment content attempts only after a valid candidate fails revalidation; terminally block after bounded operational retries. | `app/online/validator.py`, `app/online/multi_repair.py`, lifecycle/error contracts | Error matrix across schema/JSON/timeout/429/unavailable versus content failure | **P1 High** |
| 7. Incremental Duplicate V2 | Pair shortlisting supports changed IDs and the live run used incremental checks in six of nine scans. Missing or cleared changed IDs cause a full scan; unchanged-state skipping is not authoritative. | **Mostly implemented** | Persist duplicate input fingerprints and dirty question IDs, skip when unchanged, invalidate only touched pairs, and retain a conditional final full verification. | `app/online/duplicate_judge.py`, `app/online/duplicate_pipeline.py`, `app/online/validator.py`, graph state | Dirty-pair, newly eligible, cross-task duplicate, restart/cache isolation, final verification tests | **P2 Medium** |
| 8. LLM diagnostics | Structured logs and `llm_usage` capture successful usage, but failed requests lack durable provider codes, request IDs, rate headers, slot attribution, and replacement component sizes. | **Partial** | Extend existing request/usage telemetry with compact optional diagnostic fields and failure events; avoid prompt bodies and secrets. | `app/logging_conf.py`, provider clients, `app/api/repositories.py`, migration, replacement selector | Redaction, failed-request persistence, aggregation compatibility, request correlation | **P2 Medium** |

## A. Confirmed findings

### 1. Final Assembly Integrity

The production pipeline currently has two acceptance sources.

`app/online/exam_builder.py::assemble_exams_node` builds a latest-verdict map and removes questions whose latest Validator action is not `PASS`. Later, `generate_exams` reconciles `slot_lifecycle` and computes `invalid_slot_ids` from lifecycle terminal states. The returned `exams` array is not re-filtered against that lifecycle set. `app/jobs/tasks.py::_complete_generation_with_transient_retry` persists the returned exam arrays, while shortage warnings are derived independently from lifecycle counts.

This exactly explains the live discrepancy. The four extra persisted objects had a current `PASS` verdict but a non-final lifecycle state. They passed the assembly filter and failed the later authoritative count:

- `m1_mcq_10`
- `m1_true_false_1`
- `m1_true_false_3`
- `m1_word_problem_3`

The export routes load the persisted exam payload and render its arrays. They do not independently reconcile lifecycle acceptance. Therefore the defect reaches PDF, DOCX, and My Exams if an inconsistent payload is saved.

Existing protection is limited. `tests/test_exam_builder.py::test_assembly_excludes_non_pass_questions_from_student_output` checks a simple PASS/non-PASS case. It does not model a latest PASS combined with a terminal repair or lifecycle failure. Lifecycle shortage tests verify counts, but do not assert that payload IDs, accepted IDs, and export IDs are identical.

### 2. JSON Schema Compatibility

`app/llm/schemas.py::strict_json_schema` performs provider-oriented normalization: it resolves references, removes unsupported annotation keys, converts constants, makes object properties required, and sets `additionalProperties: false`.

The confirmed failing contract is still present:

```python
class SpecializedRepairRawItem(StrictOutputModel):
    question_id: object
    patch: object
    repair_metadata: object

class SpecializedRepairResponse(StrictOutputModel):
    repairs: list[object]
```

The normalized schema contains `$.properties.repairs.items = {}`. OpenAI strict structured output rejects that before model generation. `app/online/multi_repair.py::_execute_batch` sends this generic envelope, then validates returned rows against `repair_response_model(...)`. The correct exact model is therefore applied too late.

The direct offline probe found no empty nodes in `SpecializedValidatorResponse` and found the expected empty item node in `SpecializedRepairResponse`. This agrees with the live schema failures and explains why ordinary Validator contracts succeeded while specialized Repair contracts failed.

The repository already has strong pieces to reuse:

- `repair_response_model` builds typed, question-specific patch contracts.
- `tests/test_repair_patch_contract.py` verifies allowed patch fields.
- `tests/test_provider_structured_contracts.py` verifies provider identity and the Validator contract.
- Difficulty, Planner, Generator, Duplicate Judge, and catalog modules already expose their production Pydantic models or schema builders.

What is missing is one complete inventory test that exercises the actual production model for every structured call. A test over only simplified representative schemas would not detect this defect.

### 3. Question status and stale failure state

`app/online/lifecycle.py::reconcile_slot_lifecycle` is the closest existing component to an authoritative current-state model. It records assignment and question fingerprints, current validation state, failure stage, terminal state, and append-only history. Current code also contains an explicit guard so a historical invalid record cannot override a currently validated slot.

Candidate handling in `app/online/validator.py::validate_generated_questions` is materially safer than the live summary alone suggests:

- A staged candidate is fingerprinted.
- Assignment and question revisions are checked for staleness.
- All required candidate validators run before commit.
- The active question and plan are mutated only after revalidation PASS.
- Operationally blocked revalidation is retained for bounded retry rather than committed.

The remaining problem is multiple mutable sources of current truth: `validation_reports`, `repair_manifest`, `validation_ledger`, `repair_candidates`, `invalid_questions`, `revalidation_progress`, and `slot_lifecycle`. They are updated at different boundaries. A latest PASS can coexist with a terminal lifecycle failure; conversely, historical invalid entries can survive after successful candidate commit. Temporary states are represented indirectly rather than through one current status.

The agreed lifecycle can be represented with existing contracts. A new agent or parallel state machine is unnecessary. `slot_lifecycle` should carry the canonical current status, while all other collections serve as evidence, work queues, or append-only history.

### 4. Insufficient quota, rate limiting, and provider availability

`app/llm/deepseek.py::_request_with_backoff` catches the SDK `RateLimitError` and maps every instance to a generic provider rate-limit exception. `OpenAIClient` reuses this request/backoff implementation with OpenAI-specific exception classes. The code reads `Retry-After` to calculate a bounded delay, but does not preserve the provider response body, normalized provider error code, request ID, or remaining/reset headers in durable state.

The scheduler in `app/llm/scheduler.py` is a process-local, per-job semaphore. It has separate default and catalog gates and records queue/run time. It does not coordinate by provider/account across jobs, reserve estimated tokens, or adapt its gate after a provider response.

The forensic record therefore supports only these conclusions:

- At least 18 responses explicitly indicated insufficient credits.
- Sixty-eight terminal Validator errors were reduced to a generic rate-limit exception.
- The remaining 429 causes cannot be reconstructed from durable telemetry.
- Treating every 429 as RPM/TPM throttling would be incorrect.

Existing bounded transient retry and `Retry-After` handling should remain. The missing behavior is precise classification and a provider/account circuit for confirmed quota exhaustion. Guessed fixed RPM/TPM values are not justified by the evidence.

### 5. Catalog Replacement context explosion

`app/online/duplicate_pipeline.py::_select_replacement_candidate` is used for ordinary catalog recovery as well as duplicate-related recovery. It serializes active slots, the failed slot, complete existing questions, candidate metadata, previous uses, and prior failures. `app/online/validator.py` stores the resulting `replacement_selector_context` on a replacement plan. A later recovery deep-copies that plan and embeds the old context inside the next request.

The recursion is confirmed, not hypothetical. The saved contexts average approximately 135,580 tokens in the checkpoint artifacts. Removing nested selector and agent-choice contexts reduces the measured average to roughly 50,219 tokens. The first clean request is still large because complete questions and duplicated active-slot metadata account for most of it. Sending only five candidates saves roughly 4,000 tokens from a clean request, so top-five shortlisting alone does not solve the problem.

The code already performs useful deterministic filtering in `_candidate_pool`: type and variation compatibility, attempted/blacklisted assignments, reuse/collision signals, concept, target, operation, family, and reasoning metadata. It currently limits the pool to 12. It does not explicitly score how well a candidate addresses the current failure code.

The Selector genuinely needs compatibility, evidence identity, givens/constraints, and enough current-exam metadata to avoid overlap. It does not need full question bodies, old selector prompts, complete repair transcripts, or prior agent-choice payloads. Provenance can be stored separately from the next LLM input.

The `duplicate_replacement` operation label is misleading for ordinary catalog replacement. It caused the 47 calls to appear connected to Duplicate V2 even though Duplicate V2 found zero candidates and made zero Judge calls.

### 6. Operational retry accounting

This area is partially corrected in current code.

`validate_generated_questions` classifies candidate revalidation verdicts with `VALIDATION_BLOCKED` or `PENDING_RETRY` as operational and tracks `operational_retry_counts` separately. `tests/test_multi_repair.py::test_operational_repair_failure_does_not_spend_content_attempt` and regeneration integration tests assert that some operational failures do not increment content attempts.

The serial recovery path still tracks `candidate_generation_failures` when no valid candidate is produced and can terminally exhaust at `MAX_CANDIDATE_GENERATION_FAILURES`. In the live run, schema rejection and provider failures contributed to candidate-generation exhaustion. Although this counter is separate from `question_repair_attempts`, its terminal effect still converts an integration failure into exhausted recovery and can trigger another recovery route.

The required distinction should be based on the existence and evaluation of a valid candidate:

- No schema-valid candidate exists because transport, schema, parsing, availability, quota, or contract failed: operational retry.
- A valid candidate exists and fails required revalidation for content reasons: content attempt.

This classification must be shared with Problem 4. Duplicating error rules in Validator, Repair, and provider modules would recreate inconsistent behavior.

### 7. Incremental Duplicate V2

The core pair-screening implementation is already incremental. `app/online/duplicate_judge.py::shortlist_duplicate_pairs` accepts `changed_question_ids` and retains only pairs touching those IDs. It still evaluates cross-task signals such as concepts, operations, targets, formulas, numerical values, givens, text similarity, and reuse metadata. It is not restricted to identical Task IDs.

The continuation's nine scans were not nine full scans. The checkpoint evidence shows:

- Full scans at eligible populations of 8 and 9.
- Six incremental scans while repaired or newly eligible questions changed.
- A final full scan over 13 eligible questions.

The current gap is how `scan_exam_duplicates` discovers changes. It infers them from `pending_revalidation_ids` and current repair candidates. Once those transient fields are cleared, `changed_question_ids` becomes `None`, which means full scan. There is no durable last-scanned question/assignment fingerprint registry that proves nothing relevant changed.

The existing shortlist and classification logic should remain. The implementation change should be limited to dirty-state derivation, cache invalidation, and skip/final-verification rules.

### 8. LLM diagnostics

The repository already has a JSON logging formatter, job context, scheduler timing, per-call usage capture, and `llm_usage` persistence. The `llm_usage` table records job, provider, model, operation, token categories, duration, cost, and timestamp. It records successful requests after usage is available.

It does not durably record failed provider attempts, question/slot attribution, agent role, request/attempt IDs, normalized error category, sanitized provider code/message, provider request ID, retry guidance, rate-limit headers, or recovery decision. Replacement context component sizes are also absent. These omissions made the 68 terminal 429s and per-slot replacement costs impossible to reconstruct precisely.

The smallest coherent extension is to build on the current request logging and usage repository. It should not store complete prompts or credentials. Existing cost aggregation must continue to count successful usage only; failed-attempt rows with unknown usage must not silently appear as zero-cost successful calls.

## B. Proposed implementation

### 1. One authoritative acceptance set

1. Add a small lifecycle helper in `app/online/lifecycle.py`, for example `accepted_slot_ids(lifecycle)`, which returns slots whose current terminal state is finalized and whose required current validation is PASS.
2. In `assemble_exams_node`, reconcile current state before public assembly and filter every question section by this set. Rebuild section counts and rendered text after filtering.
3. Return the accepted ID set and excluded-slot audit records in graph state. Do not truncate arrays to force count equality.
4. In `generate_exams`, assert that flattened exam question IDs exactly equal the authoritative accepted set. A mismatch should block persistence as an integrity error.
5. In `app/jobs/tasks.py`, persist the accepted set and final status with the same transaction as the exam payload. Status is `INCOMPLETE` whenever accepted count is below requested count.
6. In export and My Exams read paths, treat the persisted accepted payload as authoritative. Add a defensive integrity check against accepted IDs when that metadata exists. Legacy records without accepted IDs continue to render under the existing contract and are marked as legacy in internal diagnostics.
7. Preserve all excluded slots and reasons in audit/checkpoint data; do not place them in student-facing arrays.

### 2. Complete structured-schema audit and repair contract correction

1. Change `multi_repair._execute_batch` to construct and send an exact `repair_response_model` for the actual question type, IDs, and allowed fields before the provider call.
2. Remove runtime use of `SpecializedRepairResponse` and `SpecializedRepairRawItem` as provider contracts. They may be deleted after all call sites and serialized checkpoints are checked, or retained only as non-provider internal types during a compatibility period.
3. Create a test-only registry of actual production schema factories. Enumerate Catalog Compiler, Planner, difficulty adjustment, Plan Review, every generator type, all six Validator roles, all seven repair routes, Duplicate Judge V2, Replacement Selector, regeneration, and any title/export structured call.
4. For every produced schema, recursively assert typed array items, typed object properties, closed-object behavior where strict mode requires it, required/nullable consistency, resolvable unions, supported enums/literals, and absence of unsupported defaults/annotations after normalization.
5. Pass each actual schema through mocked OpenAI and DeepSeek boundaries. Test valid output, invalid schema, invalid JSON, missing required fields, wrong types, timeout, unavailable provider, retryable errors, and non-retryable errors.
6. Verify that response parsing uses the same Pydantic model that generated the provider schema. A provider/schema failure must return an operational result and preserve the original candidate/plan.

### 3. Canonical current question status

1. Reuse `slot_lifecycle` as the canonical current state and add or standardize a `current_status` field mapped to existing enums/constants.
2. Define one transition function in `lifecycle.py`; call it at full validation, repair start, candidate staging, revalidation start, candidate commit, content exhaustion, and operational exhaustion.
3. On successful candidate commit, atomically update the active question/plan fingerprints, current validation PASS, manifest resolution, and lifecycle status. Append the previous finding to history.
4. On revalidation content failure, replace current findings with the latest findings and invalidate only affected role verdicts. Historical findings remain in history.
5. On operational failure, preserve the original active question and its last valid content verdict, set current status to validation blocked or pending operational retry, and do not manufacture a content FAIL.
6. Make `validation_reports`, `repair_manifest`, and `invalid_questions` projections or evidence collections. They must not override lifecycle based only on list order.
7. Add checkpoint upgrade logic that derives current status from legacy state once. Do not rewrite historical checkpoints.

### 4. Provider error classification and adaptive scheduling

1. Introduce a compact normalized provider error value shared by OpenAI and DeepSeek clients: category, provider code, sanitized message, HTTP status, request ID, retry-after, selected reset/remaining headers, retryable flag, and usage-availability flag.
2. Parse provider-specific response bodies before mapping SDK errors. Explicit account-credit/quota exhaustion becomes `INSUFFICIENT_QUOTA`; temporary throttling becomes `RATE_LIMITED`; 5xx/network outages become `PROVIDER_UNAVAILABLE`. An ambiguous 429 remains an operational 429 with its evidence; it must not be called insufficient quota without proof.
3. Keep bounded exponential backoff and honor provider retry guidance.
4. Extend the existing scheduler with provider/account state. Confirmed quota exhaustion opens a circuit for that provider/account and blocks new paid calls until an operator or configured refresh clears it. Temporary limits set adaptive cooldown and may lower current concurrency based on observed headers. Existing configured concurrency remains the upper bound.
5. Propagate normalized operational failures to graph state so affected questions become `VALIDATION_BLOCKED`, checkpoints remain resumable, and content budgets are unchanged.
6. Do not add a USD budget or guessed fixed RPM/TPM policy.

### 5. Compact catalog replacement

1. Stop writing `replacement_selector_context` and full previous agent choices onto plans. Store compact provenance in audit history: selected candidate IDs, window index, failure codes, prompt fingerprint, and decision summary.
2. Replace `existing_questions` plus duplicated active-slot payloads with one compact exam registry containing slot ID, type, current task/variation, concept, operation, target, formula/result/givens signatures, difficulty, and reuse state.
3. Build compact failure context from slot/question ID, type, current failure codes and concise reasons, relevant constraints, attempted/blacklisted assignments, and attempt counters.
4. Keep candidate task/variation, compatibility, concept, operation, target, difficulty, reasoning mode, required givens, evidence chunk IDs, and reuse/collision data.
5. Extend deterministic ranking with explicit failure-resolution features. Filter invalid evidence and bindings first, score all eligible candidates, and send the best five.
6. If no candidate succeeds, send the next non-overlapping window of five. Persist the window cursor and exhausted IDs. Respect the existing content-recovery limit.
7. Validate the LLM-selected assignment with the production full-catalog binding validator before regeneration. A selector cannot authorize an assignment absent from the authoritative catalog.
8. Rename ordinary usage to `catalog_replacement_selector`; reserve duplicate-specific operation labels for actual duplicate repair/Judge work.

### 6. Separate content attempts from operational retries

1. Replace ad hoc exception checks with the normalized classification from Problem 4.
2. Increment `question_repair_attempts` only after a schema-valid candidate is staged and fails applicable revalidation for content.
3. Track bounded operational attempts by stage, role, provider, and question. Schema rejection, invalid JSON, timeout, quota/rate limit, provider unavailability, and incomplete response stay in this counter.
4. When operational retries are exhausted, transition to `VALIDATION_BLOCKED`; when valid candidates exhaust content attempts, transition to `FAILED_TO_REPAIR`.
5. Do not route an operational failure to a different content Repair Agent or regeneration. Resume the same stage when the operational condition clears.
6. Preserve valid rows from partially successful batches and retry failed items independently where the schema contract permits.

### 7. Finish incremental Duplicate V2 state handling

1. Persist a job-local fingerprint for each eligible question's content, assignment, and duplicate-relevant metadata after every completed scan.
2. Derive dirty IDs by comparing current fingerprints with the last scan, plus newly eligible questions. Do not rely only on transient repair queues.
3. Skip screening entirely when the dirty set and eligibility set are unchanged.
4. When dirty IDs exist, compare each against every other eligible question using the existing deterministic rules. Invalidate only pair-cache entries touching a dirty or removed ID.
5. Retain verified pair verdicts for unchanged pairs and keep cache scope keyed by continuation job.
6. Run final full deterministic verification only when no complete baseline exists or the acceptance set changed after the last scan. LLM Judge calls remain limited to shortlisted pairs.

### 8. Extend existing diagnostics

1. Add optional correlation and diagnostic columns to the existing request/usage persistence path, or add a tightly coupled request-event table if preserving `llm_usage` aggregation semantics requires it. Keep current successful-usage rows backward compatible.
2. Record job, slot/question, stage, role, provider/model, operation, request/attempt ID, timing, retry count, normalized outcome/error, sanitized provider identifiers, retry guidance, selected rate headers, and recovery decision.
3. Record token/cost values only when usage is observed; use an explicit `usage_available` flag for rejected requests.
4. For replacement selection, record candidate count, window, component character/token estimates, prompt fingerprint, and recovery reason.
5. Apply current secret redaction and add tests proving credentials and full prompts are absent.

## C. Dependencies

Problems 1 and 3 must be designed together. Final assembly needs the canonical current-status source, while the status design needs a consumer that enforces it. The first deployable safety patch can still add a conservative assembly intersection using existing lifecycle terminal state and current PASS, followed by the full lifecycle cleanup.

Problems 2, 4, and 6 share the operational-error contract. The exact repair schema can be corrected independently, but its failure handling must use the same classification and counters as rate limits and provider outages. Otherwise a future schema defect could again consume content recovery.

Problem 5 depends on status and retry semantics for window exhaustion and on the existing catalog-binding validator. It does not depend on Duplicate V2 changes. Ordinary catalog replacement and duplicate repair should share candidate utilities but retain distinct telemetry and routing reasons.

Problem 7 is mostly isolated. It depends on reliable question/assignment fingerprints from Problem 3, so its dirty-set implementation should follow the lifecycle fingerprint decision.

Problem 8 should be added alongside Problems 2, 4, 5, and 6 so their new outcomes are observable. A small first slice—request correlation and normalized errors—should accompany those fixes; broader reporting can follow.

## D. Testing strategy

### Production graph and state

- Run the actual graph with mocked providers for a 23-slot replay fixture.
- Assert Catalog, Planning, and initial Generation are not invoked for checkpoint replay.
- Exercise PASS, content FAIL, repair, candidate staging, revalidation PASS/FAIL, operational block, content exhaustion, and restart.
- Assert active question/plan changes are atomic and candidates cannot mutate accepted state before PASS.
- Assert historical events remain while exactly one current status exists.

### Final assembly, persistence, and export

- Reproduce the live four-extra-object pattern: latest Validator PASS plus terminal lifecycle failure.
- Assert the public payload, persisted payload, accepted ID set, API response, PDF, DOCX, and My Exams contain the same IDs.
- Assert 9/23 produces exactly nine objects and `INCOMPLETE`.
- Assert all 14 excluded slots and reasons remain in audit data.
- Add legacy saved-exam/checkpoint compatibility fixtures.

### Structured schemas

- Generate every actual production schema for representative question types and all roles/routes.
- Validate both raw Pydantic and normalized provider schemas recursively.
- Mock valid responses and every requested malformed/error condition at the SDK boundary.
- Confirm provider/model/role identity survives error mapping.
- Confirm all five ordinary Validator roles and Duplicate V2 use their production contracts.
- Confirm all seven Repair routes send exact typed contracts.

### Provider failures and retry accounting

- Feed captured-style OpenAI and DeepSeek responses for insufficient quota, RPM/TPM throttling with and without `Retry-After`, timeout, connection failure, and 5xx.
- Assert quota opens the affected provider/account circuit and does not retry repeatedly.
- Assert temporary throttling uses bounded adaptive delay.
- Assert operational errors do not alter content attempts, plans, or questions.
- Assert valid candidate plus content revalidation failure increments exactly one content attempt.
- Assert partial batch success is preserved.

### Catalog Replacement

- Use the live-size catalog/checkpoint fixture offline.
- Assert no prompt contains `replacement_selector_context`, prior complete selector request, or full repair transcript.
- Assert deterministic top-five ordering, compatibility, evidence validity, failure relevance, and no repeated exhausted candidates.
- Assert subsequent windows work and stop at the existing content limit.
- Assert the selected binding is checked against the full catalog.
- Compare selection outcomes against the current full-context fixture and investigate any quality regression.

### Duplicate V2

- Verify unchanged state makes zero screening and zero LLM calls.
- Verify one changed question is compared with every other eligible question, including cross-task pairs, and unrelated pairs are reused.
- Verify newly eligible and removed questions update the cache correctly.
- Verify known formula/text/result duplicates reach the Judge and zero-candidate scans make no Judge request.
- Verify restart and cross-job cache isolation.

### Diagnostics

- Assert success and failure events correlate to one request/attempt ID.
- Assert unknown usage is not counted as observed zero usage.
- Assert existing cost aggregations remain stable.
- Assert API keys, authorization headers, source text, and full prompts are redacted or absent.

Before any paid canary, run the focused suites, the existing 40-question mocked evaluation, the 60-question stress suite, the full offline test suite, Python compilation, and `git diff --check`.

## E. Performance targets

These targets are verification thresholds, not promises of live provider behavior.

| Area | Target |
|---|---|
| Replacement recursion | Zero historical selector contexts or full previous agent-choice payloads in a new prompt |
| Candidate shortlist | At most 5 candidates per Selector call; no exhausted candidate resent in later windows |
| Replacement prompt | Median at or below 10,000 input tokens on the captured live fixture; hard offline regression threshold at 15,000 unless a documented evidence payload requires more |
| Replacement reduction | At least 85% reduction from the measured 133,553-token live average; investigate against the modeled ~6,788-token payload |
| Replacement attribution | 100% of calls mapped to job, slot, failure reason, candidate window, request ID, and operation |
| Status consistency | Payload IDs, persisted accepted IDs, lifecycle finalized IDs, and export IDs are exactly equal |
| Stale verdicts | Zero cases where an older PASS/FAIL overrides a newer authoritative verdict for the same fingerprint |
| Operational accounting | Zero content attempts consumed without a schema-valid candidate that underwent content revalidation |
| Duplicate screening | Zero full rescans when eligible-set and duplicate-relevant fingerprints are unchanged |
| Duplicate Judge | Zero LLM calls when deterministic screening returns no candidates |
| Quota handling | No second paid request to a provider/account after confirmed insufficient quota within the same run |

## F. Risks and compatibility

### Existing checkpoints

New lifecycle fields and duplicate fingerprints will be absent from historical checkpoints. Loaders need a deterministic compatibility path that derives state without mutating old checkpoint records. A resumed legacy job should perform one conservative full duplicate screen before establishing its baseline.

### Saved exams and API contracts

Historical exam payloads may already contain inconsistent objects. This change should prevent new inconsistent writes. Retrofitting or deleting historical questions is a separate data-migration decision and is outside this plan. Adding accepted/excluded metadata should be backward compatible and optional in API serializers until clients are updated.

### Schema batching

Exact question-specific repair models can increase the number of distinct schemas. Batching must group compatible contracts or build a typed combined model. Falling back to untyped objects is not acceptable. Provider schema-name length and cache behavior should be tested.

### Provider scheduling

Provider/account circuit state must be shared across worker concurrency in a deployment. A process-local-only circuit would still allow another worker to continue after confirmed quota exhaustion. Redis is already present and is a likely coordination mechanism, but the implementation should reuse existing deployment infrastructure and degrade safely if coordination is unavailable.

### Replacement quality

Compaction can remove context that helped avoid subtle overlap. Preserve all binding, evidence, givens, target, formula/result signature, and reuse signals, and compare offline decisions against captured fixtures. Subsequent candidate windows protect coverage without restoring recursive payloads.

### Source evidence

Evidence IDs and minimum supporting text must remain sufficient for the Selector and regeneration contract. Full source documents are unnecessary, but eliminating all evidence text may reduce grounding. Measure the exact required excerpt size using production contract tests.

### Telemetry migration

Existing cost reports assume every `llm_usage` row represents observed successful usage. Failure-event support must not inflate request counts or treat unknown usage as zero-cost observed usage. Keep estimated cost separate from provider invoices.

## G. Proposed execution order

### Stage 0 — Freeze contracts and capture regression fixtures

Capture sanitized versions of the four inconsistent assembly states, the failing Repair schema, representative 429 bodies, recursive Selector contexts, and duplicate scan fingerprints. Add tests that fail on current behavior. Verification checkpoint: failures reproduce without provider calls.

### Stage 1 — Integrity boundary

Implement the authoritative accepted-slot helper, assembly filter, persistence assertion, and export consistency checks. Verification checkpoint: the live 9/23 fixture persists and exports exactly nine questions and remains `INCOMPLETE` with 14 audit exclusions.

### Stage 2 — Structured-output contracts

Add the full production-schema registry tests and switch specialized Repair to exact typed contracts. Verification checkpoint: every production schema passes OpenAI and DeepSeek normalization and mocked request/response cases; malformed contracts are operationally blocked.

### Stage 3 — Current status lifecycle

Centralize current-state transitions and make all later stages consume them. Add legacy-checkpoint derivation. Verification checkpoint: exhaustive transition tests and replay restart tests show one current state and append-only history.

### Stage 4 — Operational errors and retry accounting

Implement provider-specific normalized errors, provider/account quota circuit, adaptive cooldown, and shared operational/content counters. Verification checkpoint: captured 429 fixtures classify correctly; no operational error spends a content attempt or triggers unrelated regeneration.

### Stage 5 — Catalog Replacement cost control

Remove recursive context, create compact exam/failure payloads, add deterministic failure-aware top-five windows, full-catalog verification, and corrected telemetry names. Verification checkpoint: captured fixture meets prompt-size and no-repeat targets without reducing successful replacement coverage in offline scenarios.

### Stage 6 — Duplicate dirty-state completion

Persist duplicate fingerprints and dirty IDs and add unchanged-state suppression. Verification checkpoint: only changed pairs are screened, cross-task duplicates remain detectable, and final verification is conditional and complete.

### Stage 7 — Diagnostic completion and full regression

Finish compact request diagnostics, migration compatibility, redaction, and reporting. Run all offline suites, mocked evaluations, compilation, and diff checks. Verification checkpoint: no new regression, all request outcomes are attributable, and no secret or full prompt is retained.

The recommended order is therefore close to the proposed order, with two adjustments: current-state work must be designed with the final-assembly fix from the start, and the minimum diagnostic/error-correlation fields should ship alongside schema and provider-error work rather than waiting entirely until the end.
