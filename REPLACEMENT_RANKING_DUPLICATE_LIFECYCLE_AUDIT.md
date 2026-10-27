# Replacement Ranking, Global Allocation, and Duplicate Findings Lifecycle Audit

## Scope and evidence

This is a read-only audit of job `cc84370b-a450-432e-bf05-0bf130dbb973`, exam `exam_0f5ffc683170`. It uses the current repository and the raw exports in `exam_forensic_cc84370b/`, especially `02_plan_transitions.json`, `06_validation_repair_history.json`, and `07_duplicate_review_history.json`. No model call, exam job, database write, or code change was made.

Status terms in this report are **CONFIRMED** (direct code and execution evidence), **LIKELY** (strong inference), and **UNKNOWN** (not persisted).

## Executive summary

**CONFIRMED — the final repeated dinner-tip questions were produced by a lifecycle failure, not by a fixed task-reuse policy.** The Catalog had only three compatible MCQ tasks available to Duplicate Replacement, so controlled reuse was legitimate. The system nevertheless reused `task_b76c4d06b6beb27a` with its exact source instance—60 AED and 17%—for multiple slots. Its ranking represented task, concept, operation, target, family, and reasoning usage, but did not rank or prohibit an identical effective-givens/scenario instance.

**CONFIRMED — replacement components do not share one transactional global allocator.** Difficulty Adjustment updates a shared plan sequentially. Duplicate Replacement reserves choices sequentially inside a round. General Repair creates isolated per-slot snapshots and processes unrelated repair targets concurrently, so independently repaired slots cannot see sibling selections made during that same round. All three use common Catalog eligibility primitives, but their ranking inputs and semantic safeguards differ.

**CONFIRMED — the last duplicate group was only partly acted on.** Scan 3 at event 2138 classified every pair among `model1_mcq_1`, `model1_mcq_2`, and `model1_mcq_8` as `duplicate`, keeping MCQ 8 and requesting replacement of MCQ 1 and MCQ 2. Event 2140 replaced MCQ 1, while MCQ 2 had reached its third duplicate-replacement attempt and was retained under `numeric_variation_not_applicable_kept`. The graph then routed directly to final assembly because `duplicate_repair_finalization_pending` was set. It did not rescan after MCQ 1 changed. Final persistence therefore retained the old three-question unresolved group, although the final content conflict was the two-question pair MCQ 2/MCQ 8.

**CONFIRMED — the Duplicate Judge found the problem correctly.** For scan 3, all three relevant edges were judged duplicates. The failure occurred after judgment: candidate qualification, attempt exhaustion, findings invalidation, rescan routing, and final warning construction.

The P0 remedy is a coordinated replacement transaction: freeze current fingerprints, persist all pair findings, solve all requested replacements against one shared reservation state, qualify candidates without overwriting active questions, rescan every pair incident to changed candidates, and derive unresolved warnings only from findings whose endpoint fingerprints still match the final exam.

## 1. Eligibility and ranking call paths

### Planner

The Planner uses the shared hard eligibility function `eligible_task_variation_candidates()` at `app/online/assessment_catalog.py:881`. That function requires a text-answerable task, a variation compatible with the requested question type, complete operation/target/work-product/reasoning/evidence metadata, an operation permitted by the assessment profile, and evidence IDs inside the selected chunk set. Candidate scoring is centralized in `score_task_variation_candidate()` at `app/online/assessment_catalog.py:846`; Planner calls it at `app/online/planner.py:1882`.

The 100-point score rewards lower task reuse, difficulty fit, scenario diversity, target diversity, and variation-group diversity. It is an allocation heuristic. It does not encode the generated mathematical instance, exact effective givens, question fingerprint, or Judge conflict edges.

### Difficulty Adjustment

`adjust_plan_difficulty()` begins at `app/online/difficulty_adjustment.py:635` and applies deterministic Catalog fallbacks through `_apply_catalog_fallbacks()` at line 531. It obtains shared eligible candidates, separates unused from used tasks, prefers exact/near difficulty, and ranks using `replacement_candidate_score()` (`app/online/assessment_catalog.py:2823`). It mutates the plan serially; a later slot sees earlier mutations in the same pass.

**CONFIRMED:** this path has current-plan visibility but no generated-question visibility because it runs before generation. It can diversify task/variation metadata, but cannot compare realized numerical instances or wording.

### General Validation Repair

General/STEM repair stages assignment replacements in `app/online/validator.py:3420-3712`. It calls Duplicate Replacement's `_candidate_pool()` and `_select_replacement_candidate()`, but passes empty `conflict_slots` and `conflict_edges` and explicitly sets `allow_semantic_overlap_candidates=True` at `app/online/validator.py:3515-3532`. As a result, its selection does not apply the ordinary obvious-semantic-collision exclusion.

`repair_invalid_questions()` at `app/online/validator.py:3715` runs non-FITB repair targets concurrently in isolated state copies (`app/online/validator.py:3735-3763`) and merges candidates afterward (`3765-3808`). Per-slot blacklist and failure memory are preserved, but sibling assignment selections made concurrently are absent from each isolated snapshot.

**CONFIRMED weakness:** simultaneous General Repair replacements can reserve the same task/instance because there is no global reservation/merge conflict check. This is a structural race in allocation semantics even when thread safety is maintained.

### Duplicate Replacement

`repair_duplicate_groups()` starts at `app/online/duplicate_pipeline.py:978`. It collects all requested replacement IDs and reserves candidates sequentially before generation (`1031-1038`), so later reservations in one duplicate round can see earlier reservations. `_candidate_pool()` (`430-458`) repeatedly calls `catalog_exhaustion_replacement()` to form up to 12 alternatives. `_select_replacement_candidate()` (`461-590`) filters blacklist, previous failures, question-type mismatch, conflict reuse, and obvious semantic collisions, then ranks candidates lexicographically.

The selector's live rank at `app/online/duplicate_pipeline.py:513-520` counts exact task reuse and representation of coverage, concept, operation, target, family, and reasoning. `_obvious_semantic_collision()` at `379-391` only detects a collision when concept, operation, and target all match. It does not compare required/effective givens, source-instance identity, scenario fingerprint, mathematical fingerprint, or generated question text.

When no deterministic conflict resolution is available, an LLM selector receives the active slots, previous failures, previous uses, blacklist, and ranked candidates (`539-586`). If that call fails, `safe[0]` is selected (`587-590`). Exact candidate score components beyond ordinal `ranking` are not persisted.

### Are the rules and global state the same?

| Component | Shared hard eligibility | Shared 100-point score | Sees generated questions | Same-round global reservation | Judge conflict edges |
|---|---|---|---|---|---|
| Planner | Yes | Yes | No | Planner-owned allocation | No |
| Difficulty Adjustment | Yes | Via replacement score | No | Sequential plan mutation | No |
| General Repair | Yes, through replacement pool | Pool uses Catalog replacement ranking; final selector is lexicographic/agent | Yes, before the round | **No; isolated concurrent snapshots** | **No; explicitly empty** |
| Duplicate Replacement | Yes, through replacement pool | Pool uses Catalog replacement ranking; final selector is lexicographic/agent | Yes | Yes, sequential reservation | Yes |

Thus the code shares eligibility primitives, but not a single allocation transaction, ranking objective, or conflict state.

## 2. Reconstructed candidate decisions

### Data that was persisted

Duplicate replacement events 2121, 2128, and 2140 persist plans after the stage, generated questions, attempt counters, failure memory, unresolved conflicts, task inventory, previous uses, and portions of selector context. They support the following sequence:

| Event | Slot | Prior role | Selected task/result | Evidence |
|---:|---|---|---|---|
| 2121 | `m1_mcq_2` | Duplicate of original MCQ 10 tip question | `task_60b654b7aaefa187`, system classification | replacement round 1 |
| 2121 | `m1_mcq_8` | Duplicate of MCQ 7 | `task_b76c4d06b6beb27a`, 60 AED/17% tip | replacement round 1 |
| 2128 | `m1_mcq_1` | Excessive overlap with Equation 2 | `task_b76c4d06b6beb27a`, 60 AED/17% tip | replacement round 2 |
| 2128 | `m1_mcq_10` | Duplicate of MCQ 8 tip | `task_0cb1d5ab85cc34b9`, graph inequality | replacement round 2 |
| 2128 | `m1_mcq_2` | Excessive overlap with system-classification TFs | `task_b76c4d06b6beb27a`, 60 AED/17% tip | replacement round 2 |
| 2140 | `m1_mcq_1` | Duplicate of MCQ 2 and MCQ 8 | `task_60b654b7aaefa187`, system classification | replacement round 3 |
| 2140 | `m1_mcq_2` | Duplicate of MCQ 1 and MCQ 8 | No replacement; `numeric_variation_not_applicable_kept` at attempt 3 | replacement round 3 |

The replacement inventory recorded only three compatible MCQ task IDs: `task_0cb1d5ab85cc34b9` (graph inequality), `task_60b654b7aaefa187` (system classification), and `task_b76c4d06b6beb27a` (tip calculation). At the relevant rounds all three were already used, so `reuse_reason=no_unused_compatible_task` and `catalog_replacement_stage=reused_task_after_catalog_exhaustion` are supported by persisted evidence.

### MCQ 2 and MCQ 8

`m1_mcq_8` acquired the tip task at event 2121 after the first duplicate scan. The persisted `previous_task_questions` already included `m1_mcq_10` with the exact same 60 AED/17% statement, answer, and certificate. The replacement was nevertheless generated and committed. `m1_mcq_2` acquired the same task and exact givens at event 2128, after MCQ 8 already held it. By scan 3, MCQ 1, 2, and 8 were the same assessment instance.

**CONFIRMED root cause:** task reuse was authorized as an intentional practice variation, but neither selection nor generation enforced an actual variation. `allowed_variations` was empty for the tip task and `exact_source_instance=true`; the only grounded instance was therefore reproduced. The metadata saying a different instance was intended was aspirational, not a satisfiable contract.

**CONFIRMED:** this was not a stale-state or parallel-reservation conflict in Duplicate Replacement. Its reservations are sequential. Each choice had current previous-use context. The defect was that the fallback and selector allowed an already conflicting task and the qualification step did not reject the identical realized instance before commit.

### Graph-inequality MCQs

The graph-inequality task `task_0cb1d5ab85cc34b9` was reused across several slots, including MCQ 3, 4, 5, and eventually MCQ 10, with different inequalities. Those questions shared operation and work product but had different coefficients, boundaries, and shading results. The record therefore supports **NEAR-DUPLICATE/template reuse**, not identity, for those realized questions. It also shows why a fixed per-task cap would be harmful: this task supports legitimate numerical variation.

The available export records chosen tasks and generated outcomes but not a complete scored list for every Difficulty Adjustment or General Repair decision. Exact candidate-pool scores, every rejected candidate, and every LLM selector response are **UNKNOWN** where `replacement_selector_context` was not retained in the exported event. Ordinal candidate rankings and task inventories are evidence; reconstructing missing 100-point values from final state would not be reliable and was not done.

## 3. Why identical tip instances were selected twice

1. **CONFIRMED Catalog exhaustion:** all three compatible MCQ tasks were in use.
2. **CONFIRMED reuse fallback:** `catalog_exhaustion_replacement()` at `app/online/assessment_catalog.py:2950` permits least-used compatible reuse after unused inventory is exhausted.
3. **CONFIRMED incomplete identity model:** rankings represent task/metadata use, not the exact source-backed assessment instance.
4. **CONFIRMED weak collision test:** `_obvious_semantic_collision()` compares only concept + operation + target. Authorized reuse bypasses that protection in `_select_replacement_candidate()` lines 499-512.
5. **CONFIRMED unsatisfiable variation:** the tip task had no allowed variations, fixed 60/17 givens, and `exact_source_instance=true`, yet was tagged for intentional variation.
6. **CONFIRMED missing precommit guarantee:** the realized question was not rejected for matching an active question's task, givens, operation, target, answer, and mathematical fingerprint.
7. **Not the cause here:** no evidence shows stale plan state or simultaneous Duplicate Replacement selection. The relevant duplicate replacements were sequential and recorded previous uses.

## 4. State carried between ranking decisions

| State | Difficulty Adjustment | General Repair | Duplicate Replacement |
|---|---|---|---|
| Current used tasks/families | Yes | Yes at snapshot start | Yes, including sequential reservations |
| Variations/groups | In score metadata | Candidate metadata | Candidate metadata/rank |
| Active generated scenarios | No | Yes, pre-round | Yes |
| Previously rejected candidates | No repair history | Per-slot memory | Per-slot blacklist/failure memory |
| Accepted sibling replacements in same round | Sequentially visible | **Not visible** | Sequentially visible |
| Judge edges/reasons | No | **No** | Yes |
| Exact givens/mathematical instance collision | No hard check | No global hard check | No hard check for authorized reuse |

**CONFIRMED:** blacklists and failures are scoped per replacement slot. A candidate rejected for one slot is not inherently globally unsuitable; that is appropriate. What is missing is a global reservation ledger and a pair-specific conflict constraint that follows the candidate into qualification.

## 5. Duplicate findings lifecycle

`scan_exam_duplicates()` at `app/online/duplicate_pipeline.py:659` runs BGE-style candidate discovery plus forced Catalog-reuse pairs, calls the Judge, builds groups, and stores pending groups and fact-cache data. The forensic export confirms complete candidate pairs, component similarities, pair verdicts, and groups for all three scans.

The live chronology is:

1. **Event 2119 / scan 1:** 39 candidate pairs, 39 Judge verdicts, four groups. It found MCQ 10/MCQ 2 as duplicates and MCQ 7/MCQ 8 as duplicates.
2. **Event 2121 / replacement 1:** MCQ 2 and MCQ 8 were replaced, among other affected slots.
3. **Event 2126 / scan 2:** 44 pairs, 44 verdicts, three groups. It found MCQ 10/MCQ 8 as duplicates, plus other overlap groups.
4. **Event 2128 / replacement 2:** MCQ 10 moved to graph inequality; MCQ 1 and MCQ 2 moved to the tip task, while MCQ 8 remained tip.
5. **Event 2138 / scan 3:** 48 pairs, 48 verdicts, one group containing MCQ 1, 2, and 8. All three pair edges were `duplicate`.
6. **Event 2140 / replacement 3:** MCQ 1 changed to system classification. MCQ 2, on attempt 3, was kept because numeric variation was considered not applicable.
7. **Events 2142/2145:** correctness validation/finalization occurred without a fourth duplicate scan.

For scan 3 the recorded scores were:

| Pair | Stem | Answer | Combined | Judge |
|---|---:|---:|---:|---|
| MCQ 1 / MCQ 2 | 0.9218 | 0.9844 | 0.9463 | duplicate |
| MCQ 1 / MCQ 8 | 0.9365 | 0.9756 | 0.9440 | duplicate |
| MCQ 2 / MCQ 8 | 0.9645 | 0.9671 | 0.9726 | duplicate |

**CONFIRMED:** every scan retained the evaluated pair payloads, scores, Judge verdicts, and group decisions in audit evidence. **CONFIRMED gap:** the durable unresolved-conflict object was copied forward as a group-level snapshot; it was not versioned by endpoint question fingerprint or recomputed after partial replacement.

## 6. Why only MCQ 1 changed after scan 3

The scan requested two replacements: MCQ 1 and MCQ 2. In `repair_duplicate_groups()`, per-slot attempts are checked and incremented at `app/online/duplicate_pipeline.py:1047-1053`. At attempt 3, the code enters a numeric-variation fallback branch (`1075` onward). For MCQ 2, the live event records `slot_attempt=3`, `numeric_variation_not_applicable_kept`, and conflict IDs MCQ 1/MCQ 8. It marked finalization pending and retained the unresolved group. MCQ 1 still had a viable attempt and was replaced.

This behavior explains the asymmetry. It does not resolve the group: replacing MCQ 1 removes two edges, but the MCQ 2/MCQ 8 edge remains.

## 7. Stale findings and finalization

**CONFIRMED code defect:** `_route_after_validation()` routes directly to `assemble_exams` when `duplicate_repair_finalization_pending` is true (`app/online/graph.py:462-464`). It therefore skips `scan_exam_duplicates` even after a replacement changed MCQ 1.

**CONFIRMED stale warning:** finalization reads `unresolved_duplicate_conflicts` verbatim and derives affected IDs from those stored groups (`app/online/exam_builder.py:2605-2624`). It does not verify endpoint fingerprints against final questions. The final warning consequently named MCQ 1, 2, and 8 although MCQ 1 had changed. The warning remained directionally correct that one unresolved group existed, but its membership was stale.

**CONFIRMED:** the last valid Judge evidence still proves MCQ 2/MCQ 8 were duplicates because neither endpoint changed after scan 3. No new Judge call was required for that unchanged edge. The MCQ 1 edges should have been invalidated when MCQ 1 changed.

## 8. Coordinated replacement lifecycle

A safe design can preserve task reuse and the current graph:

1. Persist each screening/Judge fact under `(left_question_id, left_fingerprint, right_question_id, right_fingerprint, screening_version, judge_version)`.
2. Freeze all current conflict edges before replacement. Build connected components from current fingerprints.
3. Select keep/replace decisions for the whole component, then construct one global replacement plan. Candidate scoring must consider active questions and all candidates already reserved in that plan.
4. Treat task reuse as eligible when evidence supports it, but require an explicit instance-diversity contract: different effective givens, scenario, representation, target, procedure, or reasoning as appropriate. If an exact-source task has no supported variation, mark it unavailable for a conflict that requires differentiation.
5. Generate candidates into staging objects. Run deterministic assignment, givens, answer, and mathematical-fingerprint checks before paid validation.
6. Run General/STEM validation and duplicate precommit checks against active questions plus all other staged replacements. This catches conflicts between replacements.
7. Commit successful candidates atomically through the coordinator. Keep prior valid questions when a candidate fails.
8. Invalidate only pair facts incident to changed fingerprints. Reuse unchanged pair facts, including the scan-3 MCQ 2/MCQ 8 verdict.
9. Re-screen and Judge only invalidated/new candidate pairs. Rebuild connected components from current facts.
10. Finalize only when every current conflict is either resolved or explicitly unresolved under current endpoint fingerprints. Construct warnings from that current ledger.

## 9. Improving ranking without reuse caps

Keep hard eligibility separate from ranking. Add evidence-backed terms rather than a maximum-use rule:

- **Instance novelty:** compare normalized effective givens, exact-source-instance identity, scenario values, operation, target, work product, and a canonical mathematical fingerprint.
- **Conflict-directed distance:** weight dimensions named by Judge's `what_must_be_different`; reject candidates incapable of satisfying them.
- **Feasibility:** sharply penalize a task whose allowed variations cannot produce a distinct grounded instance.
- **Marginal coverage:** score candidate contribution against the current exam plus globally reserved replacements, not only the pre-round plan.
- **Realized-question check:** after generation, recompute diversity from the actual question; Catalog metadata alone is insufficient.
- **History-aware penalties:** include per-slot failed candidates/strategies and pair-specific conflict history, while allowing a task that failed one slot to remain available elsewhere.
- **Deterministic tie-breaking:** retain stable task/variation IDs only as the final tie-breaker, after quality and diversity dimensions.

This permits graph-inequality reuse with genuinely different inequalities while rejecting repeated 60 AED/17% tip instances.

## 10. Confirmed issues and priorities

| Priority | Finding | Impact | Recommended regression test |
|---|---|---|---|
| P0 | Final duplicate replacement can route directly to assembly without rescanning changed fingerprints | Stale warnings and unresolved final duplicates | Three-node group; replace one node at exhaustion; assert unchanged conflict remains and changed edges are invalidated |
| P0 | Authorized task reuse does not require a distinct grounded instance | Identical questions can be committed | Exact-source task with fixed givens reused twice must be rejected unless a supported variation exists |
| P0 | No global qualification of all staged replacements against each other | Two replacements can introduce a new mutual duplicate | Stage two individually valid candidates with identical fingerprints; atomic commit must reject/replan one |
| P1 | General Repair selects assignments in isolated concurrent snapshots | Same-round allocation conflicts | Two repair targets prefer the same candidate; coordinator must reserve globally before generation |
| P1 | Unresolved groups lack endpoint fingerprint/version checks | Stale group membership | Change one endpoint and verify final warning uses current pair ledger only |
| P1 | Selector collision model omits effective givens/scenario/math identity | Ranking mistakes exact instances for acceptable practice variation | Same operation/target with identical versus different givens must rank distinctly |
| P2 | Candidate scores/rejections are incompletely persisted | Root-cause analysis cannot reproduce every decision | Persist score vector, eligibility rejection reasons, selector result, fallback cause, and reservation snapshot |

## 11. Missing telemetry and limits

- Exact numeric ranking vectors for every live candidate were not retained. The export often contains candidate order and task inventory, but not every 100-point component.
- Not every General Repair and Difficulty Adjustment decision retained its complete candidate pool and rejection list.
- The forensic export proves the duplicate lifecycle and plan/question changes, but it does not contain an immutable code revision identifier for every event. Source locations describe the current repository and align with the recorded state fields and behavior.
- A final post-event-2140 scan does not exist. The final MCQ 2/MCQ 8 duplicate is established from unchanged endpoint content and scan-3 Judge evidence; any claim about how a hypothetical new Judge call would phrase its rationale would be speculation.

## Final conclusion

The Planner did not need an arbitrary task-use cap, and the Duplicate Judge did not miss the identical dinner-tip pair. The pipeline exhausted a narrow compatible MCQ inventory, authorized reuse without proving a distinct instance, partially executed a three-question replacement plan, then finalized from stale group metadata without rescanning changed fingerprints. Global reservation and fingerprint-versioned pair facts are the central fixes. They preserve legitimate repeated practice while preventing an identical mathematical assessment instance from surviving as a supposed variation.
