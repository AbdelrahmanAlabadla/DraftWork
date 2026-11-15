# Repair-failure investigation — job `7d1a1af4-bebc-4090-a529-aa3238537336`

**Date:** 7 November 2026  
**Method:** Read-only inspection of persisted audit checkpoints, final assembly state, source chunks, repair traces, Catalog/plan data, and recovery code.  
**Scope:** Missing slots `m1_mcq_2` and `m2_true_false_3`.

## Executive conclusion

Repair did not fail because Validators gave vague feedback and then forgot it. The evidence shows two different failures:

1. **`m1_mcq_2`: routing blocked all local Repair calls by design.** The Grounding verdict was `insufficient_evidence`; the specialized Repair router converts that exact code into `VALIDATION_BLOCKED` and sends no repair-agent request. The system nevertheless repeated that blocked route three times before attempting replacement. This is a confirmed recovery-routing defect.
2. **`m2_true_false_3`: the assignment was unsuitable for the requested easy exam before generation.** The selected Catalog task is intrinsically hard, depends on an undefined/underspecified mathematical claim, and had already failed difficulty adaptation. Generation and subsequent Repair invented narrower curve/polynomial examples that the source did not support. This is a combination of an upstream task-selection/difficulty defect, Repair-agent drift, and a genuine source-evidence limitation.

Both final omissions were safe: DraftWork correctly refused to include unsupported questions. The waste was in trying recovery routes that had no viable path to acceptance.

## What was persisted versus what is unavailable

The audit retains the original questions, assigned plan item, exact source chunk IDs/content, Validator findings, repair history, selected repair route, and revalidation result. It proves which information Python placed in the specialized Repair payload.

It does **not** retain the full plaintext prompt/response for every repair call in this job; repair tracing is configured to store fingerprints/limited diagnostic traces rather than all content. Therefore, this report does not claim to quote raw LLM repair prompts that were not persisted. The current code proves the payload contract; the audit proves that prior failure text was supplied to the later attempts.

For a specialized repair call, Python constructs a payload containing:

```text
current_question
findings (Validator reason, expected fix, fields and source IDs)
editable_fields / precise_fields_to_fix
immutable_task_contract (the plan item)
assigned_child_chunks (the assigned source text)
previous_repair_rejection
previous_attempts (last three)
recovery_strategy and selected route
```

That contract is implemented in `app/online/multi_repair.py`. The history stored for `m2_true_false_3` also explicitly records `failure_reason_received` for attempts 1–3.

## Slot 1 — `m1_mcq_2`

### Original assignment and question

The Catalog assignment required the student to determine whether a strict-inequality boundary is included **and** whether it should be rendered solid or dashed. The generated Arabic MCQ asked about:

```latex
y < 3x + 2
```

with the keyed answer “not included, dashed.”

The assigned evidence chunk clearly supports the first part: a boundary arising from `<` or `>` is not included. Its sentence concerning the visual representation is linguistically inconsistent/ambiguous in the stored extraction: it includes language that can be read as denying that the line is dashed, while the Catalog task relationship asserts that it is dashed. The Grounding Validator correctly treated the asserted drawing convention as not clearly supported.

### Validator result

| Field | Persisted result |
| --- | --- |
| Action | `FIX_ANSWER` |
| Failure code | `insufficient_evidence` |
| Validator | Grounding |
| Fields requested | `correct_answer`, `options`, `solution_certificate` |
| Required correction | Use a boundary-line description clearly supported by the chunk, or provide evidence for the dashed-line convention. |
| Other roles | Task alignment, answer correctness, and difficulty passed. |

The Validator gave a specific, actionable diagnosis. It did **not** say the algebraic fact “strict boundary is excluded” was wrong.

### What Repair actually received and did

The important fact is that no specialized repair sub-agent was selected. In `build_repair_route`, the branch:

```text
grounding + insufficient_evidence → VALIDATION_BLOCKED
```

returns an empty route before an LLM is called. The persisted outcome is exactly:

```text
specialized repair blocked: VALIDATION_BLOCKED
```

This happened three times. The durable counters confirm no content repair call was made for this slot. Therefore:

- Python did preserve the Validator explanation in the repair manifest.
- Python did **not** submit the original question, plan, source chunk, or that explanation to a Repair LLM, because the route rejected it first.
- There is no Repair-agent output to blame for those three attempts.

### Why revalidation and recovery failed

| Recovery attempt | Outcome | Why |
| --- | --- | --- |
| 1 | `VALIDATION_BLOCKED` | `insufficient_evidence` cannot enter local Repair. |
| 2 | `VALIDATION_BLOCKED` | Same unchanged input and same blocked route. |
| 3 | `VALIDATION_BLOCKED` | Same unchanged input and same blocked route. |
| Fallback replacement | terminal failure | “No unused or reusable grounded replacement remained for this slot.” |

### Could a simpler correction have worked?

Possibly, but it needed a different policy route. The minimal academically safe correction would have removed the unsupported drawing-style claim and asked only whether the strict boundary is included. However, the immutable assigned task/variation explicitly required both inclusion and solid/dashed representation. The Validator also marked only options/answer/certificate editable, not the stem. Thus the local-repair field contract was too narrow for the likely correction.

The right conclusion is not “force an answer repair.” The Catalog task itself should not have required a drawing convention unless its evidence unambiguously supported it. A variant limited to boundary inclusion could have been a separate authorized Catalog variation.

### Classification

- **Confirmed Catalog/source-contract defect:** Catalog relationship claims “dashed,” while source evidence is ambiguous/inconsistent on that point.
- **Confirmed Python routing defect:** `VALIDATION_BLOCKED` was retried three times even though no agent could act and the input did not change.
- **Not a Repair-agent mistake:** no content Repair agent was called.
- **Replacement limitation:** recorded candidate inventory found no eligible grounded replacement under all active constraints; raw Filter & Rank candidate window evidence was not retained sufficiently to determine whether the inventory was truly empty or over-filtered.

## Slot 2 — `m2_true_false_3`

### Original assignment was already incompatible with the request

The job requested **easy** questions. The assigned task was Catalog-rated **hard**, with operation `evaluate_universal_curve_claim` and reasoning mode `logical_proof_reasoning`. Its plan includes:

```text
difficulty: hard
requested_difficulty: easy
difficulty_fallback: true
difficulty_adaptation.status: accepted_source_constraint
further_adjustment_possible: false
```

The source chunk asks the student to judge and explain a universal claim about an unspecified curve and an isolated point on an axis. It supplies no definition of “curve,” no counterexample, and no enough information to establish one. The plan’s own difficulty adjustment history says the task could not be safely simplified without changing its required reasoning.

This task should have been rejected before Generation for an easy exam, rather than retained as a “closest supported hard task.”

### Original Validator failure

The initial true/false statement was effectively: a universal claim about curves is not always correct because a curve may pass through a point on an axis. It was keyed `true`.

The Grounding Validator found that the source asks to assess the general claim but does not provide an example/evidence supporting the added assertion that a curve may pass through a point on either axis. The initial recovery type was correctly `regenerate_missing_slot`, not merely an answer edit.

### Attempt-by-attempt trace

| Attempt | Python route | What was produced | Revalidation result |
| --- | --- | --- | --- |
| 1 | `regenerate_original_assignment` | Changed the claim to a **nonlinear** curve. | Unsupported: the source says an unspecified curve, so this narrows/changes the assigned claim. The task remained hard despite the easy request. |
| 2 | `regenerate_original_assignment` | Narrowed the issue to an isolated point specifically at the **origin**. | `wrong_variation`: it changed “any point on either axis” to only the origin; answer remained ambiguous because “curve” was undefined; still too difficult. |
| 3 | `targeted_followup_repair` with route `task_alignment → question → solution → difficulty` | Introduced a real-polynomial/product-equation example and accompanying geometric reasoning. | Unsupported: the equation, definitions, and analysis were absent from the source. It also remained beyond easy difficulty. |
| 4/5 | replacement path | No new candidate was accepted. | Terminal reason: no unused/reusable grounded replacement remained. |

### Did later attempts receive prior feedback?

Yes. The stored history records `failure_reason_received` for each new attempt. Before the specialized third attempt, Python supplied the latest rejection plus the last three attempt summaries. The third repair route included four specialized agents because the current verdict had Task Alignment, question, solution, and difficulty failures.

The failures were not caused by Python discarding prior feedback. The repair model changed strategy, but each change violated a constraint the feedback had explicitly identified: do not narrow the universal claim, do not introduce unsupported givens, and do not exceed easy difficulty.

### Why local Repair could not solve it

The contracts conflict:

- The task demands evaluation/justification of a hard universal mathematical claim.
- The source does not supply a definition or counterexample sufficient for an unambiguous true/false item.
- The job requested easy difficulty.
- The exact-source rule prohibits inventing a polynomial counterexample or definitions.
- The task binding prevents a local Repair from changing the underlying operation/variation freely.

A local wording/answer patch cannot reconcile all five. This was an **unrepairable assignment**, not a repairable question defect.

### Classification

- **Confirmed upstream planning/difficulty defect:** a hard task with `further_adjustment_possible: false` was retained for an easy slot.
- **Confirmed Repair-agent mistake:** attempts 1–3 invented/narrowed mathematical content despite explicit source-grounding feedback.
- **Confirmed conflicting contract:** exact-source grounding forbids the examples needed to make the task answerable, while the task requires justification.
- **Genuine source limitation:** the evidence alone does not establish a well-defined, easy, unambiguous answer.

## Why replacement was too late

`m1_mcq_2` reached replacement only after three no-op `VALIDATION_BLOCKED` cycles. That delay is a confirmed bug in recovery classification: no local route existed, so the first blocked result should have moved directly to replacement or terminal assignment rejection.

`m2_true_false_3` first used two regeneration attempts and one specialized multi-agent repair before replacement. Given the plan already said the task was hard and could not be simplified, it should have moved to Filter & Rank before initial Generation or immediately after the first validation established that source grounding and easy difficulty were jointly impossible.

The audit records “No unused or reusable grounded replacement remained,” but does not preserve the full final candidate-window and eligibility ledger for these two terminal searches. That prevents a conclusive claim that a valid alternative did not exist in the Catalog. It does prove the system arrived at replacement after wasting recoveries that could not satisfy the contracts.

## Smallest recommended fixes (not implemented)

1. **Classify grounding `insufficient_evidence` before recovery.**
   - If the required correction is removal of a source-unsupported claim and the task contract cannot permit that removal, mark the assignment unrepairable and send it directly to Filter & Rank.
   - Do not schedule a second `VALIDATION_BLOCKED` attempt with the same assignment/input fingerprint.
   - Regression: `m1_mcq_2` must make zero content-repair calls and at most one blocked-routing decision before replacement.

2. **Require Catalog evidence support for every atomic task operation.**
   - A task requiring “included/excluded + solid/dashed” must cite unambiguous evidence for both parts.
   - Otherwise compile an inclusion-only task/variation or mark it non-text-answerable for that question type.
   - Regression: the strict-inequality source can yield an inclusion-only question, but cannot yield a dashed-line question without explicit support.

3. **Disallow an unmet hard fallback in an easy slot.**
   - `accepted_source_constraint` with achieved difficulty different from requested difficulty must be ineligible for an easy exam unless the product explicitly permits a labeled exception.
   - Regression: the curve task is excluded before Generation from an easy slot; a valid easy replacement is selected or the plan reports shortfall before provider generation.

4. **Escalate jointly incompatible source/difficulty findings.**
   - When Grounding says the required counterexample is unsupported and Difficulty says preserving the task is too hard, skip local content Repair and use replacement.
   - Regression: `m2_true_false_3` reaches no more than one failed validation before Filter & Rank.

5. **Persist terminal replacement evidence.**
   - Save candidate IDs, Python eligibility reasons, selector choice/no-candidate decision, and blacklist state for every final shortage.
   - Regression: a terminal shortage report can prove whether no candidate existed or whether candidates were excluded by a rule.

6. **Strengthen Repair prompt guardrails for source-bound examples.**
   - Explicitly state that no new curve class, polynomial, equation, definition, or counterexample may be introduced unless it occurs in `assigned_child_chunks` or is explicitly authorized variable data.
   - Regression: a repair attempt offered the unspecified-curve source must return a source-preserving patch or explicitly decline; it must not invent a polynomial example.

## Final assessment

The Validators worked: they caught unsupported claims, ambiguity, task drift, and difficulty mismatch. The final omission decision also worked: DraftWork did not accept bad questions.

The failure lies earlier: Catalog/Planner allowed two fragile assignments into the exam, recovery treated one known-unrepairable grounding failure as three retries, and Repair tried to escape the other assignment by inventing unsupported mathematics. The two shortages were therefore **not unavoidable** in the sense that they could have been detected and routed sooner, but accepting either missing question would have violated the system’s grounding and correctness guarantees.
