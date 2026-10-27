# DraftWork Phase 1 — Compact Assessment Catalog Metadata

Date: 2026-10-17
Scope: Compiler output contract only (`app/online/assessment_catalog.py`)
Constraint compliance: no model routing change, no book-upload workflow, no paid LLM calls, no new E2E exam, no chunk-size/concurrency change, no new arbitrary task limits, no Docker/deployment/DB changes, no commits/pushes. `CATALOG_MAX_TASKS=48` was **not** reintroduced. `CatalogTaskDraft` remains the full contract and is unchanged.

Architecture implemented: **LLM → Compact Task Draft → Deterministic Normalization/Expansion → Existing Full Catalog Task**. Downstream consumers (Planner, Generator, Validators, Duplicate Judge, Repair, Replacement) still receive the complete metadata; none of them interpret the new codes.

---

## 1. Field-by-field schema audit

Two schemas now exist. `CatalogTaskDraft` (internal/test contract, unchanged, 21 fields) and `CompactCatalogTask` (LLM-facing, 19 fields). Every compact field maps 1:1 to a full field; two full fields are reconstructed.

| # | Full `CatalogTaskDraft` field | Compact field | Handling |
|---|---|---|---|
| 1 | `knowledge_anchor` | ✅ present | verbatim |
| 2 | `topic` | ✅ present | verbatim |
| 3 | `concept` | ✅ present | verbatim |
| 4 | `relationship` | ✅ present | verbatim |
| 5 | `task_operation` | ✅ present | verbatim |
| 6 | `student_work_product` | ✅ present | verbatim |
| 7 | `target` | ✅ present | verbatim |
| 8 | `reasoning_mode` | ✅ present | verbatim |
| 9 | `allowed_question_types` | ✅ present | verbatim |
| 10 | `evidence_chunk_ids` | ✅ present | verbatim |
| 11 | `required_givens` | ✅ present | verbatim |
| 12 | `allowed_variations` | ✅ present | **compacted** (hybrid code/text) |
| 13 | `variable_givens` | ✅ present | verbatim |
| 14 | `complexity_boundaries` | ✅ present | **compacted** (hybrid code/text) |
| 15 | `exact_source_instance` | ✅ present | verbatim |
| 16 | `scenario_family` | ✅ present | verbatim |
| 17 | `native_difficulty` | ✅ present | verbatim |
| 18 | `reasoning_depth` | ❌ **removed** | reconstructed `= native_difficulty` |
| 19 | `reasoning_requirements` | ✅ present | verbatim |
| 20 | `prerequisites` | ✅ present | verbatim |
| 21 | `text_answerable` | ❌ **removed** | reconstructed `= True` |

Batch wrappers:

| Full | Compact |
|---|---|
| `AssessmentCatalogResponse` (`assessment_profile` + `tasks` + `fitb_bank`) | batch 0 → `CompactCatalogBatchWithProfileResponse`; batches 1..n → `CompactCatalogBatchResponse` (no profile) |

---

## 2. Fields compacted / removed / preserved / reconstructed

### Removed from the LLM-facing schema (reconstructed deterministically)
- **`reasoning_depth`** — the schema validator already enforced `reasoning_depth == native_difficulty`; 124/124 tasks were equal and contradictions were dropped. The compact prompt instructs one classification (`native_difficulty`); `_expand_compact_task` sets `reasoning_depth = native_difficulty`. If a full task arrives with an explicit contradictory `reasoning_depth`, it is **preserved**, and the existing finalizer safeguard still drops the task.
- **`text_answerable`** — the compiler always emitted `True` (124/124); `_expand_compact_task` sets it **only when the key is absent**, so an explicit `False` on any input is preserved (no silent permissive default).

### Compacted (hybrid code vocabulary, fail-safe)
- **`allowed_variations`** and **`complexity_boundaries`** now accept a short standard code (expanded to a language-independent sentence) **or** explicit source-specific text. Codes: 6 variation codes and 8 boundary codes (see `CATALOG_VARIATION_CODES` / `CATALOG_BOUNDARY_CODES`). Expansion is case-insensitive. **Unknown tokens are preserved verbatim**, never converted to a default.

### Preserved unchanged
- `assessment_profile` structure, all identity/`evidence_chunk_ids`/question-type semantics, FITB bank, and every verbatim field listed in §1.

### Deterministic expansion
- `_expand_catalog_codes(values, vocabulary)` — code→sentence; passthrough otherwise.
- `_expand_compact_task(task)` — fills `reasoning_depth`/`text_answerable` and expands the two list fields. **Idempotent**: a full task or an already-expanded task passes through unchanged.
- Called at the top of the `_finalize_catalog` per-task loop, so every downstream artifact (cache payload, plan items, generation) is full-contract.

---

## 3. Before / after JSON (same real task)

Same task `task_ec1ab33d25ea12e5` (equation/mcq/true_false, medium). Fields unchanged except the two removed keys and the two list fields.

**Full (finalized, as persisted) — excerpt:**
```json
{
  "task_id": "task_ec1ab33d25ea12e5",
  "native_difficulty": "medium",
  "reasoning_depth": "medium",
  "text_answerable": true,
  "allowed_variations": [
    "Use different coefficients and constants while preserving the stated proportionality relationship.",
    "Present the classification as a selected answer or a symbolic conclusion."
  ],
  "complexity_boundaries": [
    "Do not extend to nonlinear systems or systems with more than two variables.",
    "Do not require a full graphical analysis."
  ]
}
```

**Compact (what the LLM now returns for this task):**
```json
{
  "native_difficulty": "medium",
  "allowed_variations": [
    "Use different coefficients and constants while preserving the stated proportionality relationship.",
    "Present the classification as a selected answer or a symbolic conclusion."
  ],
  "complexity_boundaries": [
    "Do not extend to nonlinear systems or systems with more than two variables.",
    "Do not require a full graphical analysis."
  ]
}
```
(`reasoning_depth` and `text_answerable` omitted; these two source-specific values are not standard codes, so they pass through verbatim. Expansion reproduces the full JSON exactly.)

**Compact when a standard code applies (synthetic illustration):**
```json
{ "native_difficulty": "medium",
  "allowed_variations": ["numeric_givens", "different_target"],
  "complexity_boundaries": ["no_nonlinear"] }
```
expands to:
```json
{ "native_difficulty": "medium", "reasoning_depth": "medium", "text_answerable": true,
  "allowed_variations": [
    "Change numerical givens or the illustrative instance while preserving the stated relationship and all required constraints.",
    "Assess a different source-supported target or unknown."
  ],
  "complexity_boundaries": [
    "Do not extend to nonlinear equations, inequalities, or systems."
  ] }
```

---

## 4. Measured serialized size + estimated token savings

Measured on the real 12-batch compiler output for `math (2).pdf` (`data/assessment_catalogs/compiler/**`, merged `b7d085…`, catalog `04cb9e99…`), same 124 tasks / 108 FITB entries.

| Measure | Bytes |
|---|---|
| 12 batch responses, full | 271,069 |
| 12 batch responses, compact (profile only in batch 0; derivable keys dropped) | 223,484 |
| **Reduction** | **47,585 (17.6%)** |
| `assessment_profile` across 12 batches (full) | 44,182 |
| Profile bytes eliminated (batches 1..11) | 40,701 |
| 124 finalized tasks, full | 235,723 |
| 124 finalized tasks, keys stripped (`reasoning_depth`+`text_answerable`) | 229,103 |
| Task-key reduction | 6,620 (2.81%) |

Estimated output-token savings (approximation; ~4 bytes/token on English JSON):
- Full compile ≈ 271,069 / 4 ≈ **67,767 output tokens** → compact ≈ 55,871 → **≈ 11,896 output tokens saved**.
- At the benchmark's output rate ($0.060994 for 110,099 output tokens ≈ $0.554 / 1M), ≈ **$0.0066 saved per catalog compile** (~11% of the `$0.060994` compiler cost).

**Code-vocabulary honesty note:** on this document **0 of 376** variation/boundary values were exactly a standard code, so the code vocabulary saved ~0 bytes here. Its value is bounded to future documents that use standard phrasing; the substantial, measured win is **profile de-dup (40.7 kB) + dropped derivable keys (6.6 kB)**. JSON byte reduction ≠ equal token/latency reduction: latency in the original run was dominated by provider-side variance and queueing, not these bytes.

---

## 5. Evidence: task count and educational information intact

- `_finalize_catalog` is unchanged in behavior toward tasks: it still validates evidence IDs against the selection, canonicalizes identity, enforces the difficulty equality contract, and requires a category-valid `allowed_question_types`. The only structural change is that raw tasks pass through `_expand_compact_task` first.
- No task-count cap exists (`CATALOG_MAX_TASKS` is neither used nor reintroduced in the compiler path); the batching/coverage prompt language is untouched by Phase 1.
- The full 21-field contract is reproduced for every task (verified in `test_finalize_accepts_compact_tasks_and_emits_full_contract` and by re-expanding the real baseline: compact→expand is byte-identical to the full contract for the representative task).
- Group 1 variation safeguards (`allowed_variations`, `variable_givens`, `complexity_boundaries`, `exact_source_instance`) are preserved and materialized into plan items (`materialize_plan_item`).

---

## 6. Downstream compatibility

- Planner (`materialize_plan_item`, `allocate_catalog_plan`) consumes the **finalized** dict; unchanged, still receives all fields.
- Generator/Validator/Duplicate Judge/Repair/Replacement consume plan items/catalog built from the finalized dict; they never see compact codes (verified: expansion happens exclusively in `_finalize_catalog`).
- `tasks_by_question_type`, `semantic_overlap_group_id`, and FITB handling are untouched.
- `CatalogTaskDraft` and `AssessmentCatalogResponse` remain exported and unchanged, so direct-schema tests (`tests/test_assessment_catalog.py:228`) keep passing.
- Recovery (`_recover_catalog_checkpoint`) now takes `require_profile`; per-batch recovery for batches 1..n no longer requires a profile.

---

## 7. Cache / version behavior

- `compiler_cache_key` now hashes `_catalog_prompt` source + both code dicts + both compact schemas (replacing the single full schema). Any prompt, code-vocabulary, or compact-schema change invalidates the compiler cache.
- `catalog_cache_key` contract now additionally hashes `_expand_catalog_codes`, `_expand_compact_task`, and both code dicts (alongside the existing finalizer/identity/schema hashes).
- `ASSESSMENT_CATALOG_VERSION` is **not** bumped: the persisted/finalized structure is unchanged, so old finalized catalogs remain valid. Compiler-batch caches written under the previous contract are naturally superseded by the new keys.
- No chunk-level or cross-chunk Catalog caching was added.

---

## 8. Focused + full offline test results

- New: `tests/test_compact_catalog.py` — **16 passed** (reconstruction, code expansion, unknown-code verbatim fail-safe, idempotency, empty-list preservation, full-contract emission, unrecoverable-metadata drop, planner allocation, batch merge profile-from-first, prompt scoping, cache-key sensitivity, compact≤full serialization).
- Focused regression: `test_assessment_catalog.py`, `test_prompt_role_regressions.py`, `test_online_contract_regressions.py`, `test_performance_optimizations.py` — **77 passed** (one prompt-phrase assertion updated to the new difficulty wording).
- Full offline suite: **803 passed, 4 skipped** (`python -m pytest tests -q`).
- `git diff --check`: no whitespace errors (only pre-existing CRLF warnings).

---

## 9. Remaining risks

- **Token estimate is an approximation** (bytes/4); actual provider tokenization differs. No paid call was made to confirm.
- **Code vocabulary is low-yield on bespoke sources** (0/376 here). It is a bounded future benefit, not the primary saving. It is fail-safe (unknown → verbatim) but unused codes won't shrink this document.
- **Structural/field-name compaction (~30% of task bytes) is deferred** — renaming keys is risky for downstream readers and was intentionally avoided ("smallest safe changes").
- **Profile de-dup assumes the model completes batch 0's profile**; if batch 0 fails and only later batches succeed, `require_profile=False` recovery for those batches yields no profile and the merge has none. This mirrors the pre-existing single-profile dependency; no new failure mode was added because a missing profile was already fatal to `_recover_catalog_checkpoint`.
- Per-batch recovery's `len(tasks) < 28` guard remains as before (pre-existing; not modified).
- Latency impact is **not** measured (no paid run). Byte reduction does not guarantee proportional latency reduction.

---

## 10. Files changed

- `app/online/assessment_catalog.py` — added `CompactCatalogTask`, `CompactCatalogBatchResponse`, `CompactCatalogBatchWithProfileResponse`, `CATALOG_VARIATION_CODES`, `CATALOG_BOUNDARY_CODES`, `_expand_catalog_codes`, `_expand_compact_task`; `_finalize_catalog` expansion call; `_catalog_prompt(include_profile=)`; `compile_batch` per-batch response model + `include_profile`; `_recover_catalog_checkpoint(require_profile=)`; `compiler_cache_key` / `catalog_cache_key` contracts.
- `tests/test_assessment_catalog.py` — updated one difficulty prompt-phrase assertion.
- `tests/test_compact_catalog.py` — new offline coverage.

---

PHASE 1 COMPLETE — READY FOR MODEL COMPARISON
