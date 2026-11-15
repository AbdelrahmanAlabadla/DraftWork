# DraftWork Targeted Quality Fixes and Catalog Blueprint Report

Date: 2026-11-15

## Scope

Implemented the six confirmed orchestration/telemetry fixes and added concise Catalog variation blueprints. Planner ranking, difficulty allocation, validator strictness, retry limits, duplicate thresholds, and recovery architecture were not changed.

## Implementation audit

| Feature | Already implemented | Defect found | Fix applied |
|---|---|---|---|
| Repair dependency scheduling | Atomic candidate staging, dependent agents, stale-field rejection | Scheduler treated authorized fields as if they had actually been recomputed | Added actual-value comparison after material stem changes and schedules/widens the existing dependent owner step for unchanged derived fields |
| Duplicate cache preservation | Content fingerprints, changed IDs, content-keyed pair facts | Candidate pre-commit gate cleared the entire fact cache | Preserves the cache; changed candidate pairs naturally receive new content keys while unchanged pairs reuse facts |
| Validator request reconstruction | Local request protection and malformed-response retries | Missing/mismatched input records were not reconstructed | Added authoritative-exam indexing, local reconstruction, corrected validation dispatch, and separate request-event telemetry |
| Concurrent statistics merging | Per-model counters and runtime ID bookkeeping | Generic nested-dict merge could overwrite the other model's statistics | Added model-scoped eval merge, runtime-ID union, repair-ID merge, and aggregate recomputation |
| Repair telemetry separation | Repair outcome counters already counted actual content repair submissions | Cross-model duplicate replacements appeared in ordinary history without a category | Added explicit `telemetry_category`, including `cross_model_duplicate_replacement`, `content_repair`, `regeneration`, `assignment_replacement`, and `revalidation` |
| Generator event identifiers | Durable request-event persistence already captures usage, latency and cost | Generation calls lacked slot/task/variation/attempt/strategy attribution | Added generation attribution for normal, bundled, FITB and structural-repair Generator calls; job/exam IDs continue to come from job context |
| Catalog variation blueprints | Task concepts, relationships, operations, selected variation propagation and effective-variation precedence existed | Primary variations often inherited prose relationships and lacked explicit variables/validity constraints | Added task and variation concept/equation/variables/constraints contracts, prompt instructions, inheritance for old catalogs, propagation, and compiler-version invalidation |

## Key code changes

- `app/online/multi_repair.py`: actual dependent-field scheduling after material changes.
- `app/online/validator.py`: preserve content-keyed duplicate facts and categorize recovery telemetry.
- `app/online/multi_validator.py`: authoritative request reconstruction and request assembly events.
- `app/online/independent_recovery.py`: model-scoped evaluation-statistics merge.
- `app/online/exam_builder.py`: Generator request attribution.
- `app/llm/schemas.py`: backward-compatible variation blueprint fields.
- `app/online/assessment_catalog.py`: compact compiler contract, primary blueprint fields, propagation, structural comparison, compiler prompt and legacy inheritance.
- `app/online/prompts.py`: Generator instruction to use the blueprint and avoid copying example coefficients.
- `app/config.py`: compiler fingerprint/version changed to `compiler-v7-variation-blueprint-contract`.
- Focused regression coverage was added to the relevant existing test modules.

## Actual Catalog test

Source: `C:\Users\abdul\Downloads\math (2)2_compressed.pdf`

The existing ingestion artifact and production `retrieve_selected` path were used. The paid test was limited to one representative parent section containing 11 ordered child chunks about inequalities, linear programming, cost/revenue, probability and word problems. This was one Catalog batch; the complete 30-page document was not compiled.

### First inspection

The first run produced 12 tasks. One response invented a malformed FITB evidence ID and was rejected; targeted Catalog retry corrected it. Inspection found that synthesized primary variations still inherited prose `relationship` text as `equation_structure`. The schema/prompt was then corrected so the primary task angle supplies its own explicit blueprint.

### Verification output

The second run completed in one call with 10 tasks and no rejection diagnostics. Representative blueprints:

| Concept | Equation structure | Variables | Constraints | Inspection |
|---|---|---|---|---|
| Total bill with tip | `T = B(1 + p)` | `B`, `p`, `T` with roles | Tip is calculated from the bill | Pass |
| Two-job hours and wages | `x + y = H; r_x x + r_y y = W` | Work hours, total hours, wage rates, total wage | Each rate stays paired with its job | Pass |
| Independent compound probability | `P(A ∩ B) = P(A)P(B)` | Letter/name and die outcome parameters | One card, one die roll, joint event | Pass |
| Package revenue optimization | `R = 5s + 8l; 25s + 50l ≤ 4200; 3s + 5l ≤ 480` | Small/large package counts and revenue | Nonnegative integers and capacity limits | Pass; exact source quantities correctly preserved |
| Plastic processing profit | `P = 64f + 73d; f + d ≤ 1200; f ≥ 300; d ≥ 450` | Two processing quantities and profit | Capacity/minimum/price constraints | Pass; exact source constraints retained |
| Multiple maximizing vertices | `null` | Objective and feasible-region vertices | Bounded maximization context | Pass; no artificial equation invented |

One remaining quality limitation was observed: a graphical inequality-system task used a descriptive equation structure rather than a fully symbolic general system. It remains source-grounded and usable, but prompt compliance is not perfectly uniform.

## Paid LLM usage

Configured provider/model: OpenAI `gpt-6-luna`.

| Metric | Total |
|---|---:|
| Paid provider calls | 3 |
| Catalog batches tested | 2 runs of the same one-batch representative selection |
| Catalog retry calls | 1 |
| Provider failures | 0 |
| Validation-rejected responses | 1 (invalid FITB evidence ID) |
| Input tokens | 31,552 |
| Output tokens | 21,050 |
| Reasoning tokens | 8,199 |
| Estimated cost | **$0.01353467** |
| Budget | $0.20 |
| Remaining budget | $0.18646533 |

No paid Generator, Planner, Validator, Repair, Recovery or Duplicate Judge calls were made.

## Verification

- Focused quality/contract suite: **85 passed**.
- Planner, difficulty, generation, duplicate and evaluation suite: **165 passed, 1 skipped**.
- Full repository suite: **1,416 passed, 21 skipped, 1 failed**.
- The one full-suite failure is an external historical-fixture issue: `test_saved_103_chunk_catalog_mocked_graph_reaches_final_assembly` expected 103 database child rows for document `175ef217-a754-4e58-8bae-2bbb4841f824`, but the current database returned zero. It is not a code assertion failure introduced by these changes.
- Python compilation completed successfully.

## Remaining limitations

1. The paid quality test covered one representative parent section, not the entire PDF.
2. Prompt/schema improvements cannot guarantee perfect semantic equation abstraction on every LLM response.
3. Generator-attribution `question_id` is necessarily absent before a new question exists; slot/task/variation identifiers provide the authoritative pre-generation trace.
4. Validator reconstruction requires an authoritative exam record to be supplied when the assembled request itself lacks the question.
5. The target of 0–2 repairs per 46-question exam cannot be proven without a full paid generation/validation run, which was explicitly outside the authorized paid scope.
