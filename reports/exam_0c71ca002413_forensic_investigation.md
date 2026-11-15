# DraftWork forensic investigation — exam `exam_0c71ca002413`

**Job:** `0e06b6b9-5b9d-4e66-b5bc-c27451f8ff18`
**Session:** `d3b96f1b-53bf-4089-8ba7-379f9133a9c6`
**Document:** `a9e081d6-bd9d-4e59-92b7-6283a3fa1892`
**Request:** 46 questions (23 × 2 models): MCQ 10, True/False 5, Equation 4, Word Problem 4; difficulty `easy`
**Outcome:** 43/46 delivered — `m1_mcq_3`, `m1_mcq_9`, `m2_mcq_7` are `FAILED_TO_REPAIR`
**Wall time:** 631.1 s (10m31.1s) · **LLM calls:** 227 · **Cost:** $0.2604

**Scope and method.** Read-only investigation of retained runtime artifacts (`exam_metadata.json`, `exam_evaluation.json`, `warnings`, `evaluation_runs`, 40 `exam_audit_events`, and the full `validation_completed` node output), the persisted assessment catalog, PostgreSQL SELECTs, and the production code paths. No provider calls, code edits, database writes, cache changes, Docker changes, or commits were made. Findings are marked **Confirmed**, **Design weakness**, **Source limitation**, or **Unverified** where appropriate.

---

## A. Executive diagnosis

The job did not fail because of a provider outage, cache failure, model mis-routing, or a global catalog task-ID shortage. It failed because **the requested exam could not be met at its requested difficulty without substantial reuse or adaptation, and the machinery that was supposed to absorb that gap selected task types the source could not support**.

Three independent-looking terminal failures share one dominant defect: **the supplied-graph safeguard returns false for Arabic and for synthesized variations, so graph-reading tasks leak into every candidate pool** (planning, difficulty fallback, and replacement selection).

| Slot | Terminal code | Proximate cause | Shared systemic cause |
|---|---|---|---|
| `m1_mcq_3` | `duplicate_repair_budget_exhausted` | Confirmed false-positive duplicate vs `m1_equation_2`; replacement cascade ended on graph task `task_d4e09228b3dd3a60` | Duplicate judge false positive + graph guard false negative |
| `m1_mcq_9` | `no_valid_source_grounded_replacement` | Catalog/source unit conflict: source says **48 ft**, table header says **m**; 3 repairs converted units instead of restoring the source; final replacement was another graph-reading MCQ | Unit-consistency gap + graph guard false negative + repair loop without progress detection |
| `m2_mcq_7` | `no_valid_source_grounded_replacement` | Difficulty fallback swapped a probability task for graph-vertex task `task_20cfebbfb887270b`; generator fabricated an ASCII grid; 3 attempts failed; replacement also picked graph tasks | Difficulty infeasibility + fallback drift + graph guard false negative |

### The capacity problem (new, quantified)

The request asks for **46 easy questions**, but the catalog contains only **12 distinct native-easy tasks usable for any of the requested types**:

| Type | Requested slots | Native-easy tasks eligible | Deficit |
|---|---:|---:|---:|
| MCQ | 20 | 9 | 11 |
| Equation | 8 | 2 | 6 |
| True/False | 10 | 2 | 8 |
| Word Problem | 8 | 1 | 7 |
| **Total** | **46** | **12 unique** | **34** |

*Type counts overlap: two easy tasks are eligible for both MCQ and True/False, so the per-type deficits sum to 32 while the unique-task deficit is 34.*

There is **no global task-ID shortage**: the catalog has 120 tasks, the plan used 45 distinct base tasks across 46 slots, and only one task was repeated. The shortage is **level-specific**: under the system's diversity-first reuse policy, native-easy supply cannot cover an all-easy 46-question exam from this document.

That mismatch is why planning produced **36 medium / 9 easy / 1 hard** and marked 37 slots `difficulty_adaptation.status = "pending"` (`app/online/assessment_catalog.py:3094-3104`), and why `planner.validate_plans` explicitly exempts pending slots from the requested-difficulty check (`app/online/planner.py:694-700`). The design intends to "plan nearest-native, adapt later". In this run, adaptation was only partly possible, and the fallback then chose unsupported graph tasks.

### Hypotheses tested and rejected

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Catalog task-ID exhaustion / excessive reuse | **Rejected** | 120 tasks; 45 distinct base tasks for 46 slots; 46 distinct variations; one task reused twice |
| Cache miss caused the failure | **Rejected** | `assessment_catalog_cache_hit=False`, compiler cache false, compatibility cache false; this was the **first** catalog compile for this document; ingestion `processed_cache_hit=true`; 9 calls are concurrent batches of one compile |
| Wrong provider routing (Luna instead of DeepSeek) | **Rejected** | `.env` routes STEM generators/validators to `gpt-6-luna` by design (`app/llm/factory.py`); all 227 calls were Luna as configured |
| Provider outage/rate limit | **Rejected** | 227 calls accounted for; no operational validator failures; `validator_operational_failures: {}` |
| The three questions were simply "too hard" | **Rejected** | Two of the three were valid easy tasks; failure came from duplicate judgement, unit corruption, and graph-task leakage |

### Top root causes, prioritized

1. **Graph safeguard is effectively disabled for Arabic and synthesized variations** — **Confirmed**.
   `_GRAPH_*_RE` are English-only (`assessment_catalog.py:824-831`); an explicit variation flag always wins (`:842-843`); `TaskVariation.requires_supplied_graph` defaults to `False` (`app/llm/schemas.py:149`), and `_primary_variations` builds variations without that key (`assessment_catalog.py:676-700`), so the default `False` is materialized and masks the task-level heuristic. Verified by executing the real function:
   - `task_20cfebbfb887270b` → `False` despite Arabic `تمثيل بياني` ("graph representation").
   - `task_d4e09228b3dd3a60` → `True` at task level, but its synthesized primary variation carries explicit `False`, so the guard returns `False`.
2. **Difficulty contract is deferred, then violated by fallback** — **Design weakness / infeasibility**.
   Planner emits 36 medium/1 hard for an easy request with `pending` adaptations; `_apply_catalog_fallbacks` prefers an **unused medium** task over an **occupied easy** task (`difficulty_adjustment.py:677-706`), so a 20-easy-MCQ request with only 9 native-easy MCQs necessarily drifts to medium/hard.
3. **Duplicate judge accepts weak numeric repetition as duplication** — **Confirmed false positive**.
   For `m1_mcq_3` the only shortlist signal was `important_repeated_numbers`; the judge returned `same_core_knowledge=yes`, `same_target_or_result=yes`, `same_student_operation_or_method=yes`, `genuinely_new_reasoning=no`, despite the two questions having different targets and different answers.
4. **Unit/source inconsistency is not preflighted** — **Confirmed source defect + missing guard**.
   Source chunk `6c6a9777-0be1-5184-a421-eaaa82c650d9` has a table header `الارتفاع (m)` but item (b) says `48 ft`; the catalog recorded meters; the generator used 48 m; validator correctly failed `contradicted`; three repairs added `1 ft = 0.3048 m` instead of restoring `48 ft`.
5. **Repair loops repeat without progress detection or global blacklisting** — **Design weakness**.
   `m1_mcq_9` regenerated four times and `m2_mcq_7` three times along the same failure axis; `task_d4e...` was rejected as `mcq_task_unsuitable` for `m1_mcq_4` and later selected for `m1_mcq_3`; the duplicate verdict itself never got re-adjudicated.

---

## B. Per-question forensics

### B.1 `m1_mcq_3` — duplicate false positive leads to an unsupported graph task

**Original assignment (valid):** `task_cff1ed005775cbd8`, native easy, "evaluate `f(-3)`" with answer **−1**.

**Duplicate verdict.**
- Duplicate group `group_1`, pair `model1_equation_2::model1_mcq_3`.
- The shortlist had one weak signal: `important_repeated_numbers` (no shared target, no shared answer, no shared stem).
- Judge output: `same_core_knowledge=yes`, `same_target_or_result=yes`, `same_student_operation_or_method=yes`, `genuinely_new_reasoning=no`.
- Content comparison proves the verdict wrong:
  - `m1_mcq_3`: evaluate a function at `x = -3`; answer `-1`.
  - `m1_equation_2` (`task_278bf1dab2a2a7d1`): evaluate six three-variable expressions at `x=-1, y=3, z=7`; answers `9, 9, -8, -5, 20, 15`.
  - Neither target nor result matches. "Same core knowledge" (substitution) is too coarse to make the items duplicates.
- **Classification: Confirmed false positive**, caused by weak shortlist signal + judge prompt that permits coarse "same operation" reasoning.

**Repair cascade** (`repair_manifest` for `m1_mcq_3`):
1. `replace_assignment` / `catalog_replacement`: four medium candidates rejected on difficulty (requested easy).
2. Preflight rejected a selected replacement because it remained `adaptation_pending` (candidate id not retained), setting the slot to `completion_pending` (`app/online/validator.py:5030`).
3. Replacement selector chose `task_d4e09228b3dd3a60` — easy but **requires reading a supplied graph**; the guard should have excluded it.
4. Generator refused three bundle attempts: `mcq_task_unsuitable: ... لا يظهر التمثيل البياني المطلوب في الأدلة ...` ("the required graph is not in the evidence; creating one would require inventing data").
5. Task blacklisted; terminal `duplicate_repair_budget_exhausted`.

**Aggravating fact:** `task_d4e...` had already been rejected for `m1_mcq_4` in the same run (`mcq_task_unsuitable`, reason graph not in evidence) and was reassigned to `task_fb95fe46cfae2e2b`. It remained eligible for later slots, so the same bad task was selected again. Replacement blacklisting is per-slot/per-attempt, not run-global.

**Minimal fix that would have saved the slot:** if either (a) the duplicate verdict had been re-adjudicated against answer/target, or (b) the graph guard had excluded `task_d4e...`, `m1_mcq_3` would have remained the valid `task_cff1...` assignment or moved to a valid easy candidate.

---

### B.2 `m1_mcq_9` — source unit inconsistency, repaired in the wrong direction

**Assignment:** `task_d281bd2d404ed464`, native easy, exact source instance, tree-height interval-classification table.

**Source evidence.** Chunk `6c6a9777-0be1-5184-a421-eaaa82c650d9`:
- Table header: `الارتفاع (m)` ("height (m)").
- Item (b): tree height **48 ft**.

So the source itself is internally inconsistent. The catalog recorded the unit as `مترًا` ("meters"); the generator wrote `48 m`; the validator correctly failed with `contradicted`:
> "The assigned source asks about a tree with height 48 ft, but the question changes that source-specific given to 48 m. The contract marks this as an exact source instance and does not permit numeric changes."

**Repair history (four content attempts, then a final replacement lookup; all rejected):**

| Attempt | Action | Strategy | Result |
|---:|---|---|---|
| 1 | `regenerate_missing_slot` | `regenerate_original_assignment` | FAIL `unsupported`: introduced `1 ft = 0.3048 m`, not in the chunk |
| 2 | `repair_question` | `targeted_followup_repair` | FAIL `contradicted`: stem changed to `30 m`, certificate still converted `48 ft` |
| 3 | `repair_question` | `targeted_followup_repair` | FAIL `unsupported`: still used the conversion factor; validator also flagged extra work beyond easy |
| 4 | `replace_task` | `on_demand_completion_replacement` | FAIL `mcq_task_unsuitable`: selected another **graph-reading** MCQ; graph not in evidence |
| 5 | `replace_assignment` | — | No unused or reusable grounded replacement remained |

The repairer optimized for the validator's complaint ("conversion unsupported") without restoring the exact source token (`48 ft`). The correct minimal repair was to keep `48 ft` and remove the conversion. The validator was right; the repair strategy was wrong.

**Classification:** **Confirmed source defect** (ft/m conflict) + **design weakness** (no unit-consistency preflight; repair never restores exact source values; replacement pool again contaminated by graph tasks).

---

### B.3 `m2_mcq_7` — difficulty fallback substitutes an unsupported graph task

**Planned assignment:** `task_3825dbeb0809abef`, independent-events probability, native **medium**, requested easy, `difficulty_adaptation.status = pending`.

**Adjustment decision.** The adjuster declined to reduce it to easy:
> "لا توجد تنويعات مسموح بها لهذه المهمة المصدرية المحددة. خفضها إلى سهلة يتطلب تغيير معطياتها أو إزالة جزء أساسي من حساب احتمال الحدثين المستقلين..."
> ("No permitted variations; reducing it to easy would require changing its givens or removing an essential part of the independent-events probability calculation.")

Status became `failed`. `_apply_catalog_fallbacks` then replaced it with **`task_20cfebbfb887270b`** — "identify the coordinates of the vertices of the feasible region" — a graph task requiring `تمثيل بياني` ("graph representation").

**Why the fallback chose medium, not easy.** This is fully explained by the algorithm, not an anomaly:
- 20 MCQ slots requested easy, but only **9 native-easy MCQ tasks** exist.
- At fallback time all nine were already occupied by other slots (`occupied_task_ids` includes every other slot's `base_catalog_task_id` or `task_id`).
- The loop tries `require_unused=True` first and only then `require_unused=False`. In the first pass, easy has zero unused tasks, so it immediately takes the first unused **medium** task. It never reaches the second pass that would have allowed reusing an easy task.
- Comment at `difficulty_adjustment.py:677-680` states the policy explicitly: "Prefer an unused nearby-difficulty task over an exact-level task already occupied by another slot."

So the policy deliberately trades difficulty fidelity for diversity. With this catalog, that trade is guaranteed to drift.

**Downstream failure.**
- Generator fabricated an ASCII grid to satisfy the graph requirement; grounding validator rejected it (graph not in evidence).
- Three repair attempts reused the same graph task; `replace_task` also selected a graph-reading MCQ.
- Terminal: `no_valid_source_grounded_replacement`.

**Stale metadata bug (confirmed).** The replacement item for `m2_mcq_7` carries:
- `adjustment_failure_reason` and `failed_strategies[0]` still describe **independent-events probability**, the original `task_3825...`, not the graph task now assigned.
- The replacement code copies `item.get("adjustment_failure_reason")` and `adaptation.get("failed_strategies")` into the new item (`difficulty_adjustment.py:729-733`).

This is misleading audit evidence: a later reader would think the graph task was declined for probability reasons.

---

### B.4 Other observations

- `m1_mcq_4` initially hit the same graph-task problem (`task_d4e...` → `task_fb95fe46cfae2e2b`, `mcq_task_unsuitable`), then a targeted difficulty repair brought it back to easy. This shows the graph defect was detected at generation but never removed from the catalog/replacement pool.
- `exam_evaluation.json` slot outcomes mark `m1_mcq_3` and `m1_mcq_9` as `"accepted"` while `exam_metadata.json` marks them `FAILED_TO_REPAIR`. The evaluation artifact is stale/derived inconsistently. `evaluation_run.json` also reports `repair_sent=0`, `repair_failed=0`, `repair_succeeded=0` despite 20+ repair attempts and 3 terminal failures. **Confirmed telemetry inconsistency**, not a generation cause.
- `cross_model_duplicate_scan_complete` is `false` at final validation. Cross-model duplicate findings are empty. This did not cause the three failures, but it means cross-model duplicate coverage was incomplete for this exam.

---

## C. Difficulty system review

### C.1 Contract as implemented

| Component | Behavior | Evidence |
|---|---|---|
| Planner difficulty eligibility | Accepts a plan where a slot is not the requested level if `difficulty_adaptation.status == "pending"` | `planner.py:694-700` |
| Pending marking | When native level ≠ requested, planner writes `status=pending`, `from_difficulty`, `to_difficulty`, `strategy=None` | `assessment_catalog.py:3094-3104` |
| `_difficulty_for(task, requested)` | Returns native only if equal; otherwise requires adaptation | `assessment_catalog.py:2575` |
| Difficulty validator | Can pass a medium task on an easy request as `accepted_source_constraint` when adaptation floor is reached | validator acceptance path; m1/m2 events |
| Catalog fallback | Iterates `failed`/`completed` items, chooses `easy → medium → hard` with unused-first, then occupied | `difficulty_adjustment.py:635-760` |
| Fallback metadata | Sets `difficulty_status=accepted_source_constraint`, `difficulty_floor`, `actual_difficulty=chosen_level` | `apply_difficulty_fallback_metadata` (`assessment_catalog.py:3364`) |

### C.2 What happened in this job

| Stage | Easy | Medium | Hard |
|---|---:|---:|---:|
| Requested | 46 | 0 | 0 |
| Plan (`0857`) | 9 | 36 | 1 |
| Difficulty stage (`0859`) | **32** | **14** | **0** |
| Final accepted (43) | 29 | 12 | 2 |

Generation-time task reassignments and repairs later moved the mix again: across all 46 slots the final state is 31 easy / 13 medium / 2 hard, and among the 43 accepted questions it is 29 / 12 / 2.

- 35 of 46 slots changed `task_id` and/or difficulty between plan and difficulty stage.
- The final exam contains **12 medium and 2 hard questions** on an explicit `easy` request. This is surfaced only indirectly via `difficulty_fallbacks`; it is not treated as a request violation.
- `_apply_catalog_fallbacks` is the main churn engine: it considers every item whose adaptation is `failed` **or `completed`** (`difficulty_adjustment.py:652`). A `completed` item already achieved its target, yet it is still eligible for replacement. The condition should be `failed` only, or the code should verify the achieved level before replacing.
- Because `occupied_task_ids` excludes only the current slot (`:664-671`), tasks freed by earlier slots' replacements become available to later slots. Example: `m2_mcq_4` moved to `task_5fe677c114470c18`, freeing `task_20cf...`, which `m2_mcq_7` then took. This is why graph tasks propagate through the plan.
- Replacement items inherit stale `failed_strategies` and `adjustment_failure_reason` from the discarded assignment (`:729-733`).

### C.3 Verdict

**Design weakness + source limitation.** The "plan nearest-native, adapt later" policy is reasonable only if (a) adaptation is usually possible, and (b) failure is bounded by a feasibility check. Neither holds here:
- Only 12 native-easy tasks exist for 46 easy slots.
- The adjuster cannot lower many medium/hard tasks without changing their core operation.
- The fallback then silently changes difficulty and, worse, changes **task family** (probability → graph vertices).

The request should either be rejected/qualified up front ("this document can supply at most N native-easy questions; reuse or difficulty relaxation required") or fulfilled by deliberate reuse of easy tasks. Silently delivering a "different" exam is the worst option.

---

## D. Catalog coverage and unsupported graph tasks

### D.1 Capacity is not the problem for total task count

| Metric | Value |
|---|---:|
| Catalog tasks | 120 |
| Selected chunks | 103 |
| Distinct base tasks used across 46 slots | 45 |
| Distinct variations used | 46 |
| Tasks reused more than once | 1 (`task_e0602033ba73d527`, twice) |
| `assessment_profile` | `{}` (empty) |
| Catalog version | `catalog-v14-numerical-instance-policy` |
| Recovered batch count | 0 |

A task-ID shortage or "all tasks consumed" explanation is contradicted by this evidence.

### D.2 The graph safeguard

`requires_unsupported_supplied_graph(task, variation)` (`assessment_catalog.py:834-861`):

```python
explicit = variation.get("requires_supplied_graph") if variation else None
if explicit is not None:
    return explicit is True                       # :842-843
explicit = task.get("requires_supplied_graph") if task else None
if explicit is not None:
    return explicit is True                       # :845-846
text = ...                                        # English-only regexes below
if not _GRAPH_EXHIBIT_RE.search(text):
    return False
...
return bool(_GRAPH_INSPECTION_RE.search(text))
```

Three independent failure modes:

1. **English-only detection.** `_GRAPH_EXHIBIT_RE`, `_GRAPH_CONSTRUCTION_RE`, `_GRAPH_INSPECTION_RE` (`:824-831`) match `graph|chart|plot|figure`, `draw|sketch|...`, `read|identify|... from the graph`. Arabic `تمثيل بياني` / `الرسم البياني` never matches.
2. **Explicit-`False` masking.** `TaskVariation.requires_supplied_graph` is `StrictBool = False` (`app/llm/schemas.py:149`). `_primary_variations` (`:676-700`) constructs a variation dict **without** that key, then `_normalize_variations` calls `TaskVariation.model_validate`, which fills in `False`. That synthesized `False` is then treated as an explicit author statement and overrides the task-level heuristic.
3. **Task-only materialization.** `_expand_compact_task` / `materialize_plan_item` sets `"requires_supplied_graph": requires_unsupported_supplied_graph(task)` (`:2657`) using the task only, so the item-level flag can disagree with the selected variation.

**Executed verification:**

| Task | Task-level explicit | Variation-level explicit | Function result | Truth |
|---|---|---|---|---|
| `task_20cfebbfb887270b` | absent | absent (Arabic, `text_answerable:true`) | **False** | **True** (requires graph) |
| `task_d4e09228b3dd3a60` | `True` | `False` (synthesized primary variation) | **False** | **True** (requires graph) |

The compiler prompt (`assessment_catalog.py:1824`) instructs the model to set the flag only when a student must inspect/read a supplied graph. The compiler omitted it or set it false for these tasks; the runtime then treats the omission/default as authoritative.

**Consumers that skip graph tasks:**
- `duplicate_pipeline.py:58` skips only when the flag `is True`.
- Planner/eligibility does not use the flag as a hard filter for MCQ.
- Difficulty fallback does not use it.
- Replacement selector does not use it.

So even when a task is correctly flagged at one level, the default `False` hides it. This is the single most consequential bug in this run.

### D.3 Working safeguard (counterexample)

During generation, `m1_mcq_4` was reassigned from `task_d4e09228b3dd3a60` to `task_fb95fe46cfae2e2b` with `failure_code: mcq_task_unsuitable`, because the generator could not ground a graph not present in the evidence. The generator's runtime refusal is currently the **only** reliable graph filter. It is too late and too expensive: by then the task has been planned, adjusted, and billed.

---

## E. Catalog cache investigative finding

| Flag | Value |
|---|---|
| `assessment_catalog_cache_hit` | `False` |
| `assessment_compiler_cache_hit` | `False` |
| `assessment_catalog_compatibility_cache_hit` | `False` |
| Catalog key | `edd2a703d9b6712ab2cdce92c593657042a43d704fa2cdeb845fe71d6249a439` |
| Source document hash | `098abe6...` (matches document sha256) |
| Compiler batch calls | 9 concurrent calls, each `cached_tokens: 0` |
| Compiler tokens | 65,823 input / 72,863 output total |
| Compiler cost | $0.0430 |
| Aggregate compiler compute | 617.8 s (wall 87.0 s, ~7.1× concurrency) |

This was the **first and only** exam-generation job for this document; the ingestion job `989632d9...` had `processed_cache_hit: true`. There was no prior catalog to hit. The 9 calls are concurrent batches of a single compile, not retries. A zero cached-token count is expected for a first compile.

**Verdict: not a failure cause.** It is an optimization gap: the catalog is persisted after compilation, but the same document's catalog is not precomputed/attached at ingestion, so the first exam pays the full compile cost. It also represents 16.5% of total job cost for a one-time artifact.

---

## F. Recovery and architecture defects

### F.1 Duplicate pipeline
- Shortlist admits a pair on `important_repeated_numbers` alone. Number repetition is the weakest possible signal, especially in a math exam where values repeat naturally.
- The judge can say `same_target_or_result=yes` without deterministically comparing answers/targets. In `m1_mcq_3` the answers were `-1` vs `9,9,-8,-5,20,15`.
- Once judged duplicate, the verdict is never re-opened even when repair repeatedly fails. The repair loop tries replacements, not re-adjudication.

### F.2 Replacement preflight deadlock
- `validator.py:5020-5031`: if a replacement still has `difficulty_status == "adaptation_pending"` or `difficulty_adaptation.status == "pending"` after preflight, the slot is set to `completion_pending` with `failure_code=replacement_difficulty_unresolved` and the loop `continue`s.
- This happened during `m1_mcq_3` replacement selection (the specific candidate id was not retained): the replacement was never generated or rejected by a model; it was discarded by an unresolved planner state. The preflight should force the same deterministic fallback resolution used elsewhere (`_finalize_replacement_difficulty_fallback`) or treat pending as a known fallback with a recorded level, not as an infinite deferral.

### F.3 Repair loop
- `m1_mcq_9`: four attempts, each moving laterally (`LATERAL` progress explicitly recorded); attempts 1-3 all fought the same unit issue; attempt 4 changed the task to another graph task.
- `m2_mcq_7`: three attempts; attempts 1-2 regenerated the same graph task; attempt 3 replaced it with another graph task.
- No rule stops after two identical `failure_code`s or after two `LATERAL` results. The loop spends calls without changing the information available.
- Blacklisting is not global: `task_d4e...` failed for `m1_mcq_4` and was reused for `m1_mcq_3`.

### F.4 Fallback metadata
- `_apply_catalog_fallbacks` reads `adaptation.get("failed_strategies")` from the **old** item and copies it into the **new** item (`:729-733`), along with `adjustment_failure_reason`.
- It also replaces items whose adaptation status is `completed`, not just `failed` (`:652`). A completed adaptation may already satisfy the request, so replacing it is unnecessary churn.

### F.5 Telemetry/count consistency
- `evaluation_run.json`: `repair_sent=0`, `repair_failed=0`, `repair_succeeded=0` while `repair_manifest` contains 3 if terminal failures and `validation_repair_history` has 17 entries.
- `exam_evaluation.json` slot outcomes mark some final failures as accepted.
- `final_missing=0` is technically true (failed slots are separate), but a reader can mistake it for "nothing missing".
- These are reporting defects, not generation defects. They matter because they hide recurrence and make cost/quality attribution wrong.

---

## G. Performance, latency, and token waste

### G.1 Timeline (audit events `0850`–`0889`)

| Phase | Duration | Share of wall |
|---|---:|---:|
| Queue + retrieval | 1.0 s | 0.2% |
| Catalog compile | 87.0 s | 13.8% |
| Planning | 0.3 s | 0.05% |
| Difficulty adjustment | 88.6 s | 14.0% |
| Generation | 24.0 s | 3.8% |
| Initial validation | 131.0 s | 20.8% |
| Repair/revalidate cycles (5) | **291.7 s** | **46.2%** |
| Finalization | 1.6 s | 0.3% |
| **Total** | **631.1 s** | 100% |

Repair cycle detail: 46.2 / 69.4 / 24.5 / 17.8 / 26.3 / 24.8 / 38.8 / 1.9 / 34.7 / 1.5 s (repair, revalidate alternating). Nearly half the job was spent trying to recover three slots that ultimately failed, plus incidental revalidation of the other 43.

### G.2 LLM usage (227 calls, all `openai` / `gpt-6-luna`)

| Metric | Total |
|---|---:|
| Input tokens | 1,429,821 |
| Output tokens | 266,489 |
| Cached input tokens | 175,822 |
| Reasoning tokens | 133,552 |
| Aggregate call duration | 2,798,563 ms |
| Cost | **$0.2604** |

Top operations by cost:

| Operation | Calls | Cost | Notes |
|---|---:|---:|---|
| Generator | 67 | $0.0613 | 23.5% of cost; includes retries/bundles |
| Assessment compiler | 9 | $0.0430 | 16.5%; first compile, 0 cached |
| Difficulty adjuster | 17 | $0.0257 | |
| Task alignment validator | 23 | $0.0234 | |
| Difficulty validator | 23 | $0.0223 | |
| Grounding validator | 23 | $0.0214 | |
| Answer correctness | 23 | $0.0136 | |
| Quantitative validator | 15 | $0.0113 | |
| Catalog replacement selector | 10 | $0.0257 | 206K input = 14.4% of all input for 10 calls |
| Duplicate validator | 9 | $0.0045 | |
| Question repairer | 3 | — | |
| Solution repairer | 2 | — | |
| Difficulty repairer | 2 | — | |
| Task alignment repairer | 1 | — | |

**Waste signals:**
- Repair/revalidation consumed ~46% of wall time and a large share of validator calls for zero net gain on the three failed slots.
- The replacement selector is disproportionately expensive per call (206K input for 10 calls), which suggests very large candidate/context payloads; deterministic filters should prune before the LLM.
- The catalog compile's 617.8 s aggregate compute is hidden behind 87 s wall due to concurrency, but it is still 16.5% of cost for a reusable artifact.
- Cost per delivered question: $0.00605; if the three failed slots are counted as pure waste, ~$0.018 plus ~291.7 s were spent to produce nothing for them.

---

## H. Prioritized remediation

### P0 — correctness and recurrence prevention

#### H.1 Make the supplied-graph safeguard tri-state, language-aware, and non-maskable
- **Evidence:** `assessment_catalog.py:824-861` (English regex, explicit-first), `:676-700` (synthesized variations), `:2657` (task-only materialization), `app/llm/schemas.py:149` (default `False`), `duplicate_pipeline.py:58` (`is True`).
- **Root cause:** default `False` is indistinguishable from an author's explicit `False`, and Arabic graph terms are not detected.
- **Impact:** all three missing questions; recurrence in prior job `73e4132e` (`m2_mcq_7` graph-vertex omission).
- **Fix:**
  1. Change schema to `requires_supplied_graph: bool | None = None` so omission is preserved.
  2. In `requires_unsupported_supplied_graph`, consult task-level `True` before variation-level `False`; only an explicit `False` on a variation whose text is genuinely graph-free may override.
  3. Add Arabic/mixed-language patterns: `تمثيل بياني`, `الرسم البياني`, `الشكل`, `المنحنى`, `المستوى الإحداثي`, `حدد من الرسم`, `اقرأ من الرسم`.
  4. When synthesizing primary variations, inherit the task's computed flag; never write `False` by default.
  5. Use the flag as a hard eligibility filter in planner, difficulty fallback, duplicate replacement, and on-demand completion.
- **Regression tests:**
  - `task_20cf...` and `task_d4e...` must return `True`.
  - Arabic MCQ "حدد رؤوس المنطقة من التمثيل البياني" must be `True` even with a synthesized primary variation.
  - A text-only Arabic task with no graph terms must remain `False`.
  - End-to-end: an easy MCQ request with this catalog must not assign `task_20cf...` or `task_d4e...`.
- **Complexity:** M.

#### H.2 Add up-front feasibility check for requested difficulty
- **Evidence:** 12 unique native-easy tasks vs 46 easy slots (Section A); `assessment_catalog.py:3094-3104`; `planner.py:694-700`.
- **Root cause:** the system accepts an infeasible request and defers the problem to adaptation.
- **Impact:** 36/46 planned as medium/hard, 35 task/difficulty changes, 12 medium/2 hard delivered.
- **Fix:** before planning, compute per-type native-level inventory and compare with requested counts. If insufficient:
  - default: allow controlled reuse of exact-level tasks (best quality), or
  - ask/flag the user for permission to relax difficulty, or
  - fail the request early with an explicit capacity message.
  Do not silently plan non-requested levels.
- **Regression test:** request 20 easy MCQs from a catalog with 9 → either reuse or explicit warning; never a silent medium plan.
- **Complexity:** M.

#### H.3 Stop difficulty fallback from changing task family / level silently
- **Evidence:** `difficulty_adjustment.py:652` (replaces `completed` too), `:664-671` (occupied excludes only current), `:677-706` (unused-first policy), `:729-733` (stale failure text).
- **Root cause:** the fallback optimizes diversity before difficulty fidelity and does not restrict replacement to a compatible task family.
- **Impact:** `m2_mcq_7` moved from independent-events probability to graph vertices; `m1_mcq_9` replacement moved to graph reading; 35/46 slots churned.
- **Fix:**
  - Only replace `failed` items; leave `completed` items alone.
  - Rank candidates by a lexicographic objective: level distance, task-family distance, then reuse penalty; do not choose a different family while an exact-level, same-family task exists (even if occupied).
  - Persist a hard `difficulty_floor`/`difficulty_ceiling` and fail the slot if the replacement violates it.
  - Do not copy `failed_strategies`/`adjustment_failure_reason` from the discarded item.
- **Regression test:** easy request + 9 easy MCQs → fallback must reuse an easy MCQ, not choose medium; replacement metadata must not mention the old task's concept.
- **Complexity:** M/L.

#### H.4 Fix duplicate shortlist/judge false positives
- **Evidence:** `duplicate_judge.py:249` (`shortlist_duplicate_pairs`), `:352` (`_prompt`), `:434` (`deterministic_pair_classification`); `m1_mcq_3` history.
- **Root cause:** `important_repeated_numbers` alone admits a pair, and the judge can declare `same_target_or_result` without comparing the actual answers/targets.
- **Impact:** valid `m1_mcq_3` was destroyed and cascaded into a graph-task failure.
- **Fix:**
  - Require at least one strong signal (shared target, shared answer, or near-identical stem) before LLM adjudication; treat number repetition as a tiebreaker only.
  - Deterministically compare normalized answers/targets; if they differ, force `same_target_or_result=no`.
  - Add explicit negative examples (same operation, different function/inputs) to the judge prompt.
  - Add a "duplicate verdict was wrong" escape hatch when replacement repeatedly fails.
- **Regression test:** `m1_mcq_3` vs `m1_equation_2` must be **not duplicate**; a true duplicate pair must still be caught.
- **Complexity:** M.

### P1 — robustness and data quality

#### H.5 Unit-consistency preflight and exact-instance fidelity
- **Evidence:** chunk `6c6a9777...` (`m` header vs `48 ft` item); `catalog_numerical_contract_diagnostics` exists but does not check units; `m1_mcq_9` repair history.
- **Root cause:** catalog compiler normalized units; no deterministic unit check; repairer preferred conversion over restoring source.
- **Impact:** `m1_mcq_9` terminal after 4 repairs.
- **Fix:**
  - Add a unit-consistency diagnostic over table headers, item text, required givens, and variable givens; flag conflicting units before generation.
  - For `exact_source_instance`, copy the source's literal value/unit into the contract and disallow unit conversion.
  - Repair instructions for `contradicted` exact instances must restore source tokens, not introduce conversion factors.
- **Regression test:** chunk with `m` header and `48 ft` item must produce a preflight finding, and the generator must not silently choose either unit.
- **Complexity:** M.

#### H.6 Resolve replacement preflight deadlock
- **Evidence:** `validator.py:5020-5031`.
- **Root cause:** a replacement that remains `adaptation_pending` is deferred indefinitely rather than deterministically finalized.
- **Impact:** discarded a selected replacement for `m1_mcq_3` before generation (candidate id not retained); `task_6dcbde914b8800a4` had already been rejected separately as medium.
- **Fix:** call the deterministic fallback finalizer on pending replacements; if no level can be resolved, record `difficulty_unresolved` immediately instead of `completion_pending`.
- **Regression test:** replacement returning pending must either finalize with a recorded level or fail deterministically in the same attempt.
- **Complexity:** S/M.

#### H.7 Add no-progress detection and run-global blacklist to repair loops
- **Evidence:** `m1_mcq_9` 4 attempts / `m2_mcq_7` 3 attempts; `m1_mcq_4` and `m1_mcq_3` both hit `task_d4e...`.
- **Root cause:** repeated `LATERAL` results and same failure codes do not change strategy; blacklist is local.
- **Impact:** ~46% of wall time, zero net gain, repeated graph-task selection.
- **Fix:**
  - Stop after two `LATERAL`/same-code failures and escalate to a different action (replacement/fallback/terminal).
  - Blacklist `mcq_task_unsuitable` task IDs for the whole exam run, including replacement pools.
  - Persist attempted strategies and reject a duplicate strategy automatically.
- **Regression test:** a slot failing twice with `contradicted` must not issue a third identical regeneration; a blacklisted graph task cannot be selected for another slot.
- **Complexity:** M.

### P2 — efficiency and reporting

#### H.8 Precompute/reuse catalog at ingestion
- **Evidence:** Section E; catalog persisted but not attached at ingestion.
- **Fix:** compile the catalog during ingestion (or eagerly on document upload) and key it by document hash; first exam then gets a cache hit. Also trim compiler output requirements to reduce 72.9K output tokens per compile.
- **Impact:** saves ~87 s wall and ~$0.043 per first exam per document; more with reuse across models/jobs.
- **Complexity:** M.

#### H.9 Prune replacement-selector context deterministically
- **Evidence:** 10 calls / 206K input / $0.0257 (14.4% of all input).
- **Fix:** filter candidates by question type, level, graph flag, and family before sending; send compact candidate summaries only. Cache candidate pools.
- **Impact:** likely large token reduction; fewer hallucinated graph selections.
- **Complexity:** S/M.

#### H.10 Fix evaluation/telemetry consistency
- **Evidence:** `evaluation_run` `repair_sent=0`/`repair_failed=0`; `exam_evaluation` accepted for failed slots; `cross_model_duplicate_scan_complete=false`.
- **Fix:** derive counts from `repair_manifest`/`validation_repair_history` and final slot statuses; assert `requested = valid + invalid + missing`; never mark a `FAILED_TO_REPAIR` slot accepted; complete or explicitly skip cross-model scan with a recorded reason.
- **Impact:** restores trust in dashboards and prevents recurrence from being hidden.
- **Complexity:** S.

#### H.11 Surface difficulty and shortfall honestly in customer output
- **Evidence:** final 12 medium + 2 hard on easy request; only `question_shortages` surfaced.
- **Fix:** add a `difficulty_shortfall` summary to exam metadata/UI; state how many questions are above/below requested and why.
- **Impact:** product honesty; enables support to set expectations.
- **Complexity:** S.

---

## Appendix — Evidence inventory and confidence

| Artifact | Location / key |
|---|---|
| Job audit events `0850`–`0889` | `C:\Users\abdul\AppData\Local\Temp\opencode\forensic\` (dumped from `exam_audit_events`) |
| Plan event | `0857_planning_plan_completed.json` |
| Difficulty event | `0859_difficulty_adjustment_difficulty_adjustment_completed.json` |
| Final validation event | `0885_general_validation_validation_completed.json` (`repair_manifest`, `validation_repair_history`, `slot_lifecycle`, `repair_replacement_memory`) |
| Exam metadata | `exam_metadata.json` (`generation_audit`, `difficulty_fallbacks`, `question_shortages`, `quality_summary`) |
| Evaluation | `exam_evaluation.json`, `evaluation_run.json` |
| Catalog | `data/assessment_catalogs/ed/edd2a703d9b6712ab2cdce92c593657042a43d704fa2cdeb845fe71d6249a439.json` |
| Source chunk | `document_chunks` row `6c6a9777-0be1-5184-a421-eaaa82c650d9` |
| Runtime code | `app/online/assessment_catalog.py`, `difficulty_adjustment.py`, `planner.py`, `duplicate_judge.py`, `duplicate_pipeline.py`, `validator.py`, `multi_repair.py`; `app/llm/schemas.py`, `app/llm/factory.py` |
| Prior recurrence | `reports/draftwork_deep_investigation_73e4132e.md` (same graph-vertex `m2_mcq_7` failure mode) |

**Confidence labels:**
- Graph-guard false negatives, duplicate false positive, unit conflict, cache-first-compile, capacity numbers, terminal codes, timing, token/cost totals: **Confirmed** by code execution and/or retained artifacts.
- Difficulty infeasibility and fallback drift: **Confirmed** mechanism + **source limitation** (12 easy tasks). The claim "the request could not be fully satisfied at easy level" follows arithmetically from 12 < 46 and the catalog's per-type eligibility.
- Exact cause of every one of the 35 plan→difficulty task changes: **Partially confirmed**; the fallback mechanism and representative examples are proven, but reconstructing each intermediate state is not possible from the retained event alone.
- Evaluation/stale-slot inconsistency: **Confirmed artifact inconsistency**; the precise write path that produced it was not traced.
