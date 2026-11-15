# Comprehensive Investigation: Why Generated Exams Can Contain Fewer Questions Than Requested

**Date:** 2026-11-14  
**Scope:** Current repository runtime code, tests, and existing local forensic artifacts  
**Method:** Read-only static control-flow analysis plus inspection of pre-existing evidence. No generation job, provider call, database write, or code change was performed. This report is the only file created.

## Executive conclusion

The architecture **does not guarantee the requested question count**. It creates an exact slot ledger, tries to fill and validate those slots, and accurately detects deficits at the end, but it deliberately permits persistence of any non-empty subset. The decisive policy is in `app/jobs/tasks.py:562-568,570-585,659-702`: zero accepted questions fails the job; one or more accepted questions may be saved as an exam with status `partial`, while the Celery task itself returns `status: completed`.

Thus, for a request of 40 resulting in 32, the missing eight are not padded or enforced at assembly. Each is either:

1. **missing**: no public question ever survived for its requested stable slot (planning/generation/recovery did not produce one), or
2. **invalid**: a question was generated for the slot but did not reach authoritative `finalized` status.

Only lifecycle records satisfying all conditions in `app/online/lifecycle.py:465-480` are public: finalized terminal status, finalized flag, validated flag, PASS/current status, no failure code. Assembly removes everything else (`app/online/exam_builder.py:2724-2865`). It checks consistency, not completeness (`app/online/exam_builder.py:3122-3162`).

The strongest confirmed count-loss mechanisms are:

- bounded generation leaves slots pending after unsupported assignments, provider/schema failures, invalid variation binding, forbidden content, or duplicate screening;
- validators permanently exclude content after repair/replacement budgets are exhausted;
- recovery can run out of compatible catalog candidates, fail generation/revalidation, hit reservation conflicts, or exhaust independent budgets;
- finalization filters every slot without authoritative finalized state, and can remove an entire FITB section if its word bank cannot be rebuilt;
- partial-exam persistence is explicitly accepted.

## Pipeline and count invariants

Requested slots are initialized exactly from `tasks × num_models` in `initialize_slot_lifecycle()` (`app/online/lifecycle.py:76-124`). Stable IDs have the form `m{model}_{type}_{index}`. This is the strongest count invariant in the system, but it is an **audit ledger**, not a completion guarantee.

The main route is defined in `app/online/graph.py:346-456`:

`receive → retrieval → catalog → planning/plan repair → generation → validation → repair/revalidation → independent model recovery/cross-model duplicate recovery → assembly`

The final count is computed from actual public items (`app/online/exam_builder.py:3122-3126`). A deficit changes `quality_summary.status` to `incomplete` and `complete` to false, but does not cause failure (`app/online/exam_builder.py:3144-3162`).

## A. Complete rejection and failure reasons

The table groups conditions only when they share the same control flow. Validator failure codes are listed explicitly so no validator rejection class is hidden.

| Stage | Rejection / failure reason | Exact trigger | Retry / recovery behavior | Can reduce final count? | Code location |
|---|---|---|---|---:|---|
| Request | Missing document ID, tasks, normalized tasks, or selected chunks | Required request fields empty/invalid | Raises; no exam produced | No partial exam; whole job fails | `app/online/receive.py:18-78` |
| Retrieval | Selected chunk absent from index and processed cache | Any requested child ID remains missing after cache recovery | Raises with re-index instruction | Whole job fails, rather than short exam | `app/online/retrieval.py:67-146` |
| Catalog compilation | Incomplete batch; invalid compiler schema; evidence IDs outside batch; `batch_complete != true` | Compiler output violates batch contract | Bounded catalog batch attempts; then `CatalogBatchIncompleteError` | Usually whole job failure; can indirectly leave no plan | `app/online/assessment_catalog.py:515-526,2840-2948`; limit `app/config.py:78-80` |
| Catalog eligibility | Variation uses unsupported supplied graph; type/form mismatch; disallowed difficulty/constraints/evidence | Candidate fails `eligible_task_variations()`/candidate filters | Candidate skipped; allocator tries another | Yes if pool ultimately empty | `app/online/assessment_catalog.py:915-948,1012-1042,1101-1134` |
| Catalog adequacy | No compatible grounded task for requested type/difficulty | Ranked candidate pool empty after eligibility/exclusions | Reuse may be authorized after exhaustion; otherwise planning issue | Yes | `app/online/assessment_catalog.py:3294-3596` |
| Catalog diversity | Unique tasks/FITB entries exhausted | No unused compatible candidate remains | Explicit task/FITB reuse can be authorized | Normally prevents loss, but reused content is more likely to fail duplicate validation | `app/online/assessment_catalog.py:3378-3580,3700-3783` |
| Planning | Incomplete catalog object | `catalog_compilation_complete is not True` on a versioned catalog | Raises | Whole job fails | `app/online/planner.py:1267-1278` |
| Planning | Missing/extra slots; duplicate IDs; wrong type/model/count; invalid task/evidence/variation/difficulty/profile | `validate_plans()` or profile validation returns errors | Targeted repair, bounded by `MAX_PLAN_ATTEMPTS` | Yes if unresolved slot becomes unrepairable; otherwise whole workflow returns error | `app/online/planner.py:458-650,1094-1146,1257-1305` |
| Planning provider/schema | Planner call, structured validation, or JSON/schema output fails | Provider exception or incomplete/invalid structured response | One controlled full retry if profile truncated; checkpoint continuation for missing slots; then targeted repair | Yes if continuation/repair cannot preserve slots | `app/online/planner.py:1322-1427` |
| Plan repair | Error has no parseable slot ID | `_invalid_slot_ids(errors)` empty | Cannot target; errors retained until graph budget ends | Can prevent generation/whole job | `app/online/planner.py:2222-2263` |
| Plan repair exhaustion | No progress or max attempts reached | Plans/errors unchanged, or unresolved slots at limit | Records `plan_repair_exhausted` in `unrepairable_plan_slots` | Yes; terminal `planning_failed` | `app/online/planner.py:2265-2339`; lifecycle `app/online/lifecycle.py:326-342` |
| Generation assignment | Unsupported supplied-graph assignment with no supported alternative | `_reassign_supported_variation()` returns `None` | Slot skipped before provider call; later missing-slot recovery may act | Yes | `app/online/exam_builder.py:908-923,1502-1516` |
| Generation budget | Per-slot provider request budget exhausted | `request_budget.remaining(slot) == 0` | Slot removed from pending for that generation loop | Yes | `app/online/exam_builder.py:926-941,1543-1552`; default/cap `app/config.py:69-71` |
| Generation provider/JSON/schema | Provider error, timeout, malformed JSON, structured schema validation, incomplete output | `request_structured()` raises | Generation loop retries up to `_MAX_ATTEMPTS`; transport exhaustion breaks early | Yes | `app/online/exam_builder.py:964-990`; provider errors `app/llm/deepseek.py:45-88`, `app/llm/openai_client.py:15-58` |
| Structural normalization | Required fields absent/invalid | MCQ invalid stem/options/answer; TF empty statement/invalid bool; definition/why/essay missing prompt/answer; equation/word problem missing text, steps, or final answer; essay `key_points` not array; FITB not exactly one answer/two distractors | Targeted structural repair, max 2 at this layer; then remains missing | Yes | `app/online/models.py:275-396,410-449`; repair `app/online/exam_builder.py:818-885` |
| Generation identity | Unknown, repeated, missing, or malformed stable slot ID / duplicate question ID | Candidate cannot bind uniquely to pending slot | Candidate ignored; pending slot retried | Yes after attempts | `app/online/exam_builder.py:682-706,1016-1026` |
| Variation binding | Selected variation absent or incompatible with planned task/type | `_bind_selected_variation()` false | Candidate rejected; slot stays pending | Yes | `app/online/exam_builder.py:1027-1034` |
| Forbidden content | Question text contains forbidden phrase | `contains_forbidden_phrase()` true | Candidate rejected; slot stays pending | Yes | `app/online/exam_builder.py:1035-1041` |
| Early duplicates | Exact normalized duplicate, semantic plan duplicate, or near duplicate | `_is_duplicate`, `_is_semantic_plan_duplicate`, or `_is_near_duplicate` true | Candidate rejected; retry generates pending slots | Yes after attempts | `app/online/exam_builder.py:1042-1066` |
| Generation exhaustion | Pending slots remain after bounded attempts | Not every ordered slot enters `accepted_by_slot` | Warning only; downstream missing-slot recovery may try replacements | Yes | `app/online/exam_builder.py:1071-1095` |
| Objective bundle | Same failures as above plus True/False polarity mismatch | Candidate polarity differs from planned requirement | Retries with polarity feedback; bounded loop | Yes | `app/online/exam_builder.py:1500-1759` |
| FITB generation | Invalid/duplicate/non-term answer, invalid blank, bad distractors or bank | FITB-specific normalization and bank rules fail | Bounded generation/repair and later FITB recovery | Yes, and finalization may drop whole section | `app/online/exam_builder.py:1098-1471`; `app/online/validator.py:1024-1109` |
| Deterministic validation | Local structural/content rule fails | `_local_verdict()` returns non-PASS | Added to repair manifest | Yes if repair fails | `app/online/validator.py:1298-1496`; integration `app/online/multi_validator.py:1151-1157` |
| Grounding validator | `contradicted`, `unsupported`, `insufficient_evidence`, `copied_source_example` | Question/answer exceeds, conflicts with, or copies assigned evidence | Repair or assignment replacement; insufficient evidence is content failure, not outage | Yes | `app/online/multi_validator.py:118-125,1163-1185` |
| Task-alignment validator | `task_drift`, `wrong_givens`, `missing_required_given`, `constraint_violation`, `wrong_variation`, `question_type_operation_mismatch`, `visual_dependency`, `copied_source_example` | Actual assessment work violates planned catalog contract | Usually repair; fundamental assignment failures route toward replacement | Yes | `app/online/multi_validator.py:118-125`; routing `app/online/validator.py:82-93,560-578` |
| Deterministic alignment | Unauthorized transformation or transformation evidence not assigned | Selected cognitive transform not authorized, or its evidence chunk unavailable | Converts role PASS to FAIL (`wrong_variation`/`constraint_violation`) | Yes | `app/online/multi_validator.py:1094-1150` |
| Answer validator | `incorrect_answer`, `answer_inconsistent`, `ambiguous_or_structural`, `ambiguous_option`, `multiple_correct_options`, `no_correct_option`, `weak_distractors`, `incomplete_answer`, `unsupported_claim`, `invalid_word_bank` | Answer/options/reference response is wrong, incomplete, ambiguous, or malformed | Specialized question/options/answer repair then full revalidation | Yes | `app/online/multi_validator.py:118-125`; repair routing `app/online/multi_repair.py:375-711` |
| Difficulty validator | `difficulty_mismatch`, `reasoning_complexity_mismatch` | Observed reasoning/complexity does not match requested/declared difficulty | Targeted difficulty adjustment; after bounded failure may accept source-constrained fallback in specified paths | Yes if neither adjustment nor fallback resolves | `app/online/multi_validator.py:118-125`; `app/online/validator.py:5740-5930` |
| Quantitative validator | `incorrect_calculation`, `incorrect_result`, `invalid_equation`, `incomplete_solution`, `missing_units`, `unit_inconsistency`, `rounding_error`, `significant_figures_error`, `unsolvable`, `invalid_or_extraneous_solution`, `incorrect_option_mapping`, `unsupported_given`, `constraint_violation`, `missing_required_given` | Mathematical certificate/result/options/givens/units violate contract | Answer/solution/question repair or assignment replacement; revalidation required | Yes | `app/online/multi_validator.py:118-125` |
| Validation prerequisite | Assigned child chunk unavailable or task has no assigned child chunk | Protected-question evidence errors consist only of missing assignment evidence | `UNVALIDATED`, `validation_prerequisite_unavailable`; replacement path | Yes | `app/online/multi_validator.py:1203-1224` |
| Validator input | Invalid protected input | Other protection errors | `UNVALIDATED`, `input_invalid` | Yes | `app/online/multi_validator.py:1225-1232` |
| Validator operational | Role provider/schema/batch call fails or returns unusable/missing rows | Role status `ERROR` / `VALIDATION_BLOCKED` | Operational errors are not cached; affected validation/recovery is bounded | Yes after operational retry/generation budget | `app/online/multi_validator.py:955-1048,1168-1169,1233-1255,1424-1450` |
| Validator response contract | Unsupported code, missing required issue detail, bad field paths/conflicting requirements | `_parse_rows()` rejects malformed specialized verdict | Corrective/bounded retry; otherwise role blocked | Yes | `app/online/multi_validator.py:647-765,849-1009` |
| Duplicate validator | `duplicate_question` or `excessive_overlap` within model | Duplicate V2/Judge confirms conflict | Adds repair/replacement target; affected question is not accepted until resolved | Yes | `app/online/multi_validator.py:124`; `app/online/validator.py:3731-3867` |
| Duplicate operational | Duplicate recheck candidate missing, duplicate recheck failed, duplicate validation provider error | Recheck cannot run/finish | `VALIDATION_BLOCKED`; bounded recovery | Yes | `app/online/validator.py:3764-3773,4246-4278` |
| Cross-model duplicate | Question overlaps another model | Cross-model integration confirms duplicate | Up to three semantic recovery plan steps; if only overlap remains, current code restores/keeps last non-duplicate-validator-pass question as nonblocking fallback | Historical count loss; current fallback is designed to prevent it | `app/online/independent_recovery.py:298-358,396-445,488-528` |
| Repair route | No compatible repair route or router/provider blocked | Conflict reconciliation or specialized repair cannot select/execute authorized patch | Route may switch to regenerate/replace; otherwise terminal after budgets | Yes | `app/online/validator.py:5275-5291,6149-6170` |
| Repair proposal integrity | No-op, unauthorized fields, stale dependent fields/certificate, invalid normalized patch, changed immutable assignment | `_apply_step()`/authorization/dependency checks reject proposal | Counts as failed content repair unless operational | Yes | `app/online/multi_repair.py:913-1059,1061-1409` |
| Repair conflicts | Multiple validator requirements genuinely conflict | Conflict reconciliation cannot merge them | `repair_conflict`; tries next recovery strategy/budget | Yes | `app/online/multi_repair.py:468-709`; `app/online/validator.py:6073-6144` |
| Repair/revalidation | Candidate fails full revalidation or produces new failure | Any validator returns failure for new candidate | Candidate rejected; history retained; next strategy if budget remains | Yes | `app/online/validator.py:4354-4633` |
| Stale async result | Assignment/question revision changed while concurrent work ran | Candidate fingerprint/revision no longer matches current slot | Stale result rejected, not merged | Potentially, if no later candidate | `app/online/exam_builder.py:2445-2477`; `app/online/validator.py:4100-4138` |
| Content repair budget | Failed content repairs or repair calls reach limits | failures ≥ configured limit or calls ≥ call ceiling | Content editing disabled; independent replacement may remain | Yes | `app/online/validator.py:168-215,5303-5316,6114-6127`; config `app/config.py:376-380` |
| Completion/replacement budget | Assignment replacements reach limit | completion attempts ≥ `MAX_COMPLETION_ATTEMPTS_PER_SLOT` | Terminal `replacement_budget_exhausted`/candidate budget exhausted | Yes | `app/online/validator.py:168-215,5303-5316,6211-6232` |
| Candidate operational budget | Three candidate-generation operational failures | `operational_retry_failures >= MAX_CANDIDATE_GENERATION_FAILURES` | `validation_blocked`/budget exhausted | Yes | `app/online/validator.py:78-80,6089-6103` |
| Assignment repair budget | Same assignment validated three times unsuccessfully | assignment fingerprint count reaches 3 | `assignment_repair_budget_exhausted` | Yes | `app/online/validator.py:6264-6277` |
| Recovery loop prevention | Same repair input/candidate repeats | Fingerprint already attempted/rejected | Strategy skipped or candidate rejected to prevent loops | Yes if alternatives exhausted | `app/online/validator.py:5340-5360,6284-6315` |
| Replacement selection | No valid source-grounded replacement | Ranked catalog candidates exhausted/blacklisted/incompatible | `no_valid_source_grounded_replacement`, then terminal/next V3 strategy | Yes | `app/online/validator.py:5110-5231` |
| Replacement reservation | Concurrent model reserves same task/assessment instance | Global reservation coordinator rejects candidate | Try another candidate while bounded; failed candidates release reservation | Yes | `app/online/validator.py:5597-5622,6710-6757` |
| Replacement integration | Plan write fails, catalog binding invalid, or evidence mismatch | Replacement cannot be written/bound to active evidence | Reject replacement and continue while budget remains | Yes | `app/online/validator.py:5647-5731` |
| Replacement difficulty | Adjusted replacement cannot meet/finalize requested difficulty | Adjustment fails or remains unresolved | Retry/authorized source fallback in narrow cases; else terminal | Yes | `app/online/validator.py:5740-5929` |
| Recovery V3 | Decision blocked, unsupported/empty plan, or all saved strategies exhausted | Recovery decision unavailable or cursor reaches end | `recovery_v3_plan_exhausted` | Yes | `app/online/validator.py:5220-5231,6428-6568` |
| Independent local recovery | Four local validate/repair rounds end without model ready | Pending/terminal manifest persists through `_MAX_LOCAL_ROUNDS` | Stops local lifecycle; terminal slots proceed to assembly as excluded | Yes | `app/online/independent_recovery.py:273-295` |
| Finalization lifecycle | Slot not planned, generated, validated, or otherwise has failure | Reconciliation cannot mark `generated && validated && !failure_code` | Terminalized by stage; no further recovery after assembly | Yes | `app/online/lifecycle.py:374-450` |
| Assembly filter | Slot not in authoritative accepted set or verdict not PASS | Fails lifecycle acceptance predicate | Question copied to `invalid_questions`, omitted from public section | Yes | `app/online/lifecycle.py:465-480`; `app/online/exam_builder.py:2747-2825` |
| FITB assembly | Surviving FITB word bank cannot be aligned/rebuilt with two distractors | `_fitb_word_bank_valid` false and `_rebuild_fitb_word_bank` false | All kept FITB items recorded invalid; section not added | Yes, potentially multiple at once | `app/online/exam_builder.py:2826-2854` |
| Count classification | Public slot absent | Expected stable ID not present after assembly | Classified missing unless lifecycle says generated terminal invalid | Yes | `app/online/exam_builder.py:2871-2888,3114-3126` |
| Persistence policy | Non-empty incomplete result | accepted count > 0 but shortage exists or `complete` false | Saves `partial`; emits shortage audit and warnings; task returns completed | **Yes—direct architectural cause** | `app/jobs/tasks.py:562-585,659-702` |
| Empty result | No accepted questions | `quality_summary.accepted/valid == 0` | Raises; generation job may retry once only for classified retryable error, otherwise fails | Does not create short persisted exam | `app/jobs/tasks.py:563-568,719-731` |
| Timeout/cancellation | Global deadline, Celery soft limit, cancellation | Deadline/cancel signal fires | Job fails/cancels; generation task has at most one retry for retryable exceptions | No persisted partial from this path unless a prior checkpoint/result was already finalized | `app/jobs/tasks.py:526-558,703-731`; timeout `app/config.py:367-384` |
| Export | Model selection only; no count padding | Export uses saved exam models/questions | Exports partial payload as stored | Preserves, does not cause, shortage | `app/api/export_routes.py:33-85`; `app/exports/document_ir.py` |

## B. Question-loss flow

For every requested question, the system creates a slot. A slot can then follow one of these terminal paths:

1. **Never planned.** Catalog/profile/plan validation cannot allocate a compatible task. Targeted plan repair exhausts and records `planning_failed`.
2. **Planned but never generated.** The assignment is unsupported, provider budget ends, structured output is invalid, slot identity is wrong, variation binding fails, or every candidate is rejected by early duplicate/content screens. The validation report creates a `missing_requested_slot` target (`app/online/validator.py:3510-3625`).
3. **Generated but validator-failed.** One or more deterministic/specialized roles rejects it. It enters the repair manifest.
4. **Repair succeeds.** The candidate must still pass fresh revalidation. Only then is the manifest resolved and lifecycle validation restored.
5. **Repair fails.** The pipeline may regenerate the assignment or switch catalog task/variation. These are independent bounded budgets, not an unlimited “fill until count reached” loop.
6. **Replacement fails.** There may be no compatible unused candidate; evidence/difficulty/binding may fail; a concurrent reservation may block it; generation or revalidation may fail; or a repeated fingerprint is rejected. Eventually the slot becomes terminal.
7. **Duplicate recovery.** Within-model duplicate failure remains blocking. Cross-model duplicate currently has a nonblocking last-valid fallback, but historical artifacts show earlier/current-run state interactions have lost slots here.
8. **Assembly.** Lifecycle is reconciled. Only authoritative finalized slots are copied. A question that exists in `generated_exams` can still disappear because it is invalid, stale, pending revalidation, terminally failed, or in an unrebuildable FITB section.
9. **Persistence.** Exact count is measured and warned about. If at least one question remains, the partial exam is saved.

There is no “missing eight” ambiguity when lifecycle/audit data is intact: `slot_lifecycle`, `invalid_questions`, `unrepairable_plan_slots`, `repair_manifest`, `validation_repair_history`, duplicate histories, `missing_slot_ids`, and `invalid_slot_ids` provide the per-slot chain. The shortage event itself contains only aggregate/type counts and optional IDs (`app/jobs/tasks.py:54-94`), so the detailed diagnosis requires the generation audit metadata.

## C. Root causes of short exams

### Confirmed architectural causes (ranked by directness, not frequency)

1. **Partial persistence is intentional.** There is no final exact-count assertion. Any accepted count above zero can be saved as `partial`; the job reports completed.
2. **Recovery is bounded and terminal.** Content repair, repair calls, assignment validation, candidate operational failures, slot completion, and duplicate replacement have separate ceilings. Once all applicable routes are terminal, assembly excludes the slot.
3. **Assembly is exclusionary, not count-restoring.** It faithfully filters to finalized lifecycle slots and never backfills.
4. **Generation itself is best-effort.** Bounded loops return accepted subsets and warnings.
5. **Catalog compatibility can be narrower than raw task count.** Type, response form, evidence, graph/visual dependency, variation, uniqueness, and requested difficulty shrink the eligible replacement pool.
6. **A valid-looking question can be lost later.** Fresh revalidation, duplicate checks, lifecycle merge, or FITB section rebuilding can remove it.

### Observed historical/artifact evidence

- `reports/exam_cb579b2fcbba_2026-11-12_full_report.md:18-43,137-169,199-201` documents a 46-slot run saved with **43 accepted questions**. Three Model 2 slots ended in duplicate/recovery routing failure. One slot had a regenerated candidate pass revalidation, after which a later failure record left the lifecycle terminal. This is direct evidence of a historical state/merge ordering loss.
- `reports/draftwork_deep_investigation_73e4132e.md:9,85-114` documents two omitted slots caused by unsupported graph/evidence/task-operation routing and bounded distinct-instance recovery—not a provider outage.
- `live_exam_replay/forensic_report.json:27-55` and `live_exam_replay/question_failure_matrix.json` record repeated terminal repair failures and specialized-repair schema failures (notably normalized JSON Schema rejection). This is replay/historical evidence, not proof that the current version has identical frequencies.
- `reports/exam_0c71ca002413_forensic_investigation.md:16-40,115-117` records difficulty-specific eligible-supply pressure and recovery candidates that introduced unsupported/contradictory givens.

### Frequency limitations

No read-only aggregate production dataset in the repository establishes population-wide frequencies by validator, repair success, language, domain, or question type. The evaluation schema can store rejection/repair summaries (`app/api/evaluation_store.py:57-252`), but querying the live database was prohibited and not performed. Therefore it would be unsound to claim that one validator is globally “most frequent.”

Based only on the available case artifacts, duplicate/recovery routing, task alignment/grounding, and repair schema failures recur. This is a **small, selected forensic sample**, not a rate estimate. Arabic and English examples both appear in the artifacts, and STEM/quantitative failures are prominent, but there is insufficient denominator data for comparative language/domain rates.

## D. Recovery system analysis

### What recovery does well

- Separates content-repair and assignment-replacement budgets (`app/online/validator.py:168-215`).
- Uses stable slot IDs, assignment/question revisions, fingerprints, blacklists, and stale-result fencing.
- Revalidates repaired candidates with the applicable validators; repair output is never accepted solely because a repair agent returned it.
- Preserves detailed histories and terminal failure codes.
- Coordinates cross-model reservations and releases permanently failed candidates (`app/online/validator.py:6697-6757`).
- Current cross-model duplicate code includes a nonblocking restoration fallback (`app/online/independent_recovery.py:396-445`).

### Why replacement can still fail

1. The apparent catalog size overstates the compatible pool. Eligibility is per type, evidence, operation, variation, visual requirements, response form, difficulty, exclusions, blacklists, and already-reserved assessment identity.
2. A candidate can be selectable but not materializable: generation can fail schema, bind the wrong variation, introduce unsupported givens, or fail requested difficulty.
3. Every replacement is subject to the full validation stack and duplicate checks.
4. Repeated assignments/candidates are deliberately blocked to prevent loops.
5. Independent budgets can terminate a path before another theoretically possible candidate is tried.
6. Concurrency does not silently drop completed futures—`future.result()` propagates unexpected exceptions—but global task/instance reservations can reject otherwise eligible choices. Those conflicts are recorded and bounded.
7. Recovery-decision failure can yield `decision_blocked`; a saved three-strategy V3 plan can be exhausted.
8. Missing evidence is not treated as a transient validator PASS; it becomes blocked/replacement-required.

### Missed or fragile opportunities

- **No final global fill loop:** after assembly identifies a deficit, the system does not return to catalog selection with the exact missing slots.
- **No transactional “PASS wins” lifecycle rule:** historical evidence shows a later terminal record could overwrite a freshly passing candidate. Current lifecycle reconciliation has safeguards for historical invalid records (`app/online/lifecycle.py:343-351`), but a formal monotonic, revision-gated state transition is still the safer invariant.
- **Dropped unsupported assignment before initial generation:** `_reassign_supported_variation() is None` removes a slot from the local pending map, relying on later missing-slot recovery instead of immediately invoking catalog replacement.
- **FITB section atomicity:** one unrebuildable final bank can suppress every surviving FITB item in that section.
- **Operational and content budgets interact:** an outage/schema problem can consume the bounded opportunity to produce a candidate even though the educational assignment itself is sound.
- **Partial completion is accepted before a deficit-directed recovery pass.**

## E. Question-count guarantees

| Question | Finding |
|---|---|
| Does Planner always allocate exact slots? | It is required to and validation detects deviations, but provider/catalog/repair failures can leave invalid/unrepairable plans. Exact lifecycle slots exist regardless. |
| Can Catalog be insufficient? | Yes, at the compatibility level even when total tasks are numerous. Reuse is allowed in some exhaustion cases, but not every unsupported type/difficulty/evidence combination can be reused safely. |
| Can task selection leave slots unfilled? | Yes: empty eligible pool, unsupported graph, exclusions/blacklists/reservations, or bounded plan repair. |
| One generated question per allocated slot? | No. Generation returns an accepted subset after bounded attempts. |
| Can validators permanently reject? | Yes, after applicable recovery budgets end. |
| Does every rejection trigger Recovery? | Content failures enter the manifest; early generation misses become `missing_requested_slot`. Whole-job errors/cancellation do not. A terminal or malformed state may be ineligible. |
| Can Recovery find no replacement? | Yes; this is explicitly represented by `no_valid_source_grounded_replacement` and related terminal codes. |
| Can concurrency lose slots? | Stale results are intentionally rejected and reservations can block candidates. Unexpected future exceptions generally propagate, rather than silently disappear. Historical artifacts show merge/state ordering could nevertheless overwrite a PASS. |
| Can a question disappear after validation? | Yes: duplicate verification, stale revision/lifecycle status, later repair record, authoritative finalization filter, or FITB bank failure. |
| Does assembly enforce count? | No. It enforces public/lifecycle set equality only. |
| Can completed job contain fewer questions? | Yes. Exam status is `partial`; task/job return is `completed`. |
| Minimum completion threshold? | Exactly one accepted question. Zero is refused; there is no percentage or per-type minimum. |
| Is partial intentional? | Yes, explicitly encoded and audited. |

The current architecture therefore offers **accurate deficit accounting**, not exact-count delivery.

## F. Recommended fixes (not implemented)

### Critical fixes

1. **Make exact count a persistence invariant unless the user explicitly opted into partial output.** Before `complete_generation_job`, require every requested lifecycle slot to be authoritative finalized. Otherwise keep the job recoverable/failed, never silently successful. If partials remain a product feature, expose a separate explicit policy and return job status `partial`, not `completed`.
2. **Add a final deficit-directed completion phase.** Feed `missing_slot_ids + invalid_slot_ids` back through replacement selection, generation, full revalidation, and duplicate checks. Stop only at exact count or an explicit, typed terminal reason for each slot.
3. **Make lifecycle transitions revision-gated and monotonic.** A PASS for the current assignment/question revision must not be overwritten by stale terminal failure history. Persist compare-and-set revision/fingerprint with every validation/repair result. Add regression coverage for the historically observed “candidate passed, later failure won” case.
4. **Separate operational retry budgets from content budgets.** Provider/schema/timeouts must not consume educational repair/replacement attempts. Persist an operational state that can resume without terminalizing the slot.
5. **Require per-slot terminal accounting before persistence.** Assert `requested_slots == finalized_slots ∪ typed_terminal_slots`, with no unknown/pending slots, and assert public slots equal finalized slots. The second assertion exists; the first should be added.

### High-priority improvements

1. Replace unsupported supplied-graph assignments immediately at generation entry instead of merely skipping them.
2. Precompute and persist a compatibility matrix by question type, response form, difficulty, evidence/visual requirements, and variation; reject impossible requests before generation or negotiate counts with the user.
3. Add a deterministic candidate-exhaustion ledger: every catalog candidate should have a reason (`ineligible`, `reserved`, `blacklisted`, generation failed, validation code, duplicate) so “no replacement” is fully explainable.
4. Make FITB finalization per-slot resilient. Rebuild the bank from survivors and, if impossible, recover only the missing distractor/affected slot instead of dropping the whole section.
5. Add a recovery scheduler that tries every applicable route under independent, explicit budgets and records why an apparently available route was skipped.
6. Treat reservation conflict as a selection retry that does not consume content/replacement quality budget; use deterministic locking/order across models.
7. Strengthen repair schema compatibility tests against each configured provider/model, especially normalized JSON Schema behavior seen in replay evidence.
8. Promote shortage events to a first-class API result containing every slot's final stage, code, last candidate fingerprint, attempts, and exhausted candidate inventory.

### Optional improvements

1. Add dashboards using existing evaluation records: loss by stage/code/type/language/domain, repair success by agent, replacement exhaustion, provider/schema blocks, and partial-exam rate.
2. Add property-based tests asserting exact slot conservation across planner, concurrent recovery merges, resume/checkpoint replay, and assembly.
3. Add adversarial tests for invalid IDs, duplicated IDs, stale futures, partial validator batches, conflicting repairs, exhausted catalogs, FITB survivor banks, Arabic math content, and mixed STEM/non-STEM exams.
4. Expose a preflight “maximum safely supported count” and warn before accepting a request whose compatibility pool cannot satisfy diversity/difficulty constraints.
5. Distinguish user-visible warnings from developer diagnostics; optional audit/progress callback failures are currently logged and allowed to continue (`app/online/graph.py:21-51,250-332`), which is correct for availability but can hide observability loss.

## Logging, audit, and silent-failure assessment

Most permanent question-loss paths are recorded in one or more of: lifecycle, validation reports, repair manifest/history, duplicate histories, invalid/missing IDs, warnings, generation audit metadata, and `question_shortage` checkpoint event (`app/jobs/tasks.py:579-648`). Invalid and unrepairable slots are also logged (`app/jobs/tasks.py:649-657`).

The important observability gaps are:

- progress/audit/checkpoint callbacks can fail without stopping generation (`app/online/graph.py:21-51,250-332`; `app/jobs/tasks.py:303-325`), so the exam state may be correct while the external audit trail is incomplete;
- evaluation telemetry save failure is only warned and does not affect completion (`app/jobs/tasks.py:676-684`);
- generation warnings are strings in several paths, making aggregate failure analysis weaker than typed per-slot events;
- early generator rejection counts exist in `eval_stats`, but not every warning has a durable raw candidate/fingerprint and precise rejection layer;
- the final shortage warning aggregates by type and may omit individual IDs if upstream lists are incomplete.

No current asynchronous generation/validation future was found that is simply ignored: futures are consumed and exceptions generally propagate or become explicit operational verdicts. The higher risk is **state reconciliation**, not an un-awaited task.

## Tests versus production evidence

Tests cover stable slot lifecycle, targeted repair, revalidation, concurrency limits, missing-information validation, duplicate handling, and 24-question repaired revalidation. They demonstrate intended behavior, not production frequency. Particularly relevant files include:

- `tests/test_revalidation_24_question_e2e.py`
- `tests/test_validation_repair_difficulty_recovery.py`
- `tests/test_shared_state_catalog_coordination.py`
- `tests/test_validator_concurrency_limits.py`
- `tests/test_validator_missing_information.py`

Existing forensic reports and replay JSON are stronger evidence of past real executions, but may describe code predating the current safeguards. This report does not convert them into population statistics.

## Bottom line

If 40 were requested and 32 were delivered, the implementation can explain the missing eight slot by slot, but it currently considers that an acceptable terminal product state. The fundamental reason is not merely that validators reject questions; it is that **bounded recovery plus exclusionary assembly is followed by an explicit partial-persistence policy**. Exact count will remain unguaranteed until successful persistence requires all requested slots—or the user explicitly authorizes a partial exam—and until a final deficit-driven recovery pass has exhausted a fully typed candidate inventory.
