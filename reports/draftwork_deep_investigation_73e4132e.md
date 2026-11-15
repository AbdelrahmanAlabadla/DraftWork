# DraftWork deep investigation — job `73e4132e-8be4-4f93-98e6-00e7bc4a12a6`

**Scope.** Read-only code and retained-job-artifact investigation. No provider calls, test runs, database writes, cache changes, or application-code changes were made. This report deliberately distinguishes a code-proven conclusion from a job-specific conclusion. The retained runtime report is the only artifact for this job; raw provider payloads, persisted candidate revisions, preview/export files, and per-slot structured-attempt records were not available in the workspace.

## A. Executive diagnosis

The job completed **44/46** questions in **16m35.75s**, with **283 successful provider requests**. It did not suffer provider-request failures. It was expensive because a first-time v14 Catalog compilation used 10 calls, then seven initial validation failures initiated bounded, multi-agent recovery: repair/regeneration, assignment replacement, difficulty adjustment, targeted revalidation, and incremental Duplicate V2. The retained telemetry assigns 73 calls to Generator, 30 each to Difficulty Adjuster/Grounding Validator, 29 Task Alignment calls, 30 Answer Correctness calls, 19 Quantitative calls, 14 Duplicate V2 Judge calls, four replacement-selector calls, and 13 repair calls.

`m2_mcq_7` was omitted because a duplicate-resolution path reached a graph-vertex task whose evidence did not contain a displayable graph or coordinates; a local edit also changed the required student operation. `m2_true_false_1` was omitted after bounded recovery could not make a distinct, valid Easy instance. This is a real source/contract and routing failure, not a provider outage.

`m1_mcq_7` made 11 Generator requests according to the supplied job summary. The code makes that count plausible through nested retry layers, but the retained data cannot attribute all 11 precisely to structured-output retries versus bundle attempts versus later regeneration. In particular, raw outputs and `structured_output_attempt` telemetry for this job are absent; claiming an exact 3×3 pattern would be speculation.

## B. LaTeX / JSON diagnosis

### What the contract currently does

Generation uses provider strict JSON Schema plus Pydantic (`app/llm/client.py`, `app/llm/deepseek.py`). Question text fields are ordinary JSON strings; schema does not prohibit LaTeX. Thus LaTeX is supported *provided that the raw JSON uses JSON escaping*.

The Generator global prompt says infinity must be Unicode `∞` or valid `\infty`, and forbids bare `infty`, malformed escapes, and controls (`app/online/prompts.py`, `SYSTEM_PROMPT`). Equation-specific instructions also say: “Because this is JSON, escape every LaTeX backslash correctly.” This is good but incomplete: it appears in Equation rules, not as a clear all-field contract covering MCQ stem/options, True/False statements, Word Problems, explanations, and certificates. It gives no concrete raw-JSON examples for `\frac`, `\text`, `\times`, or `\begin`.

The Pydantic model parses the raw provider JSON in `DeepSeekClient.achat_structured` with `response_model.model_validate_json(previous_output)`. Invalid raw JSON such as `{"answer":"(-\infty, 2)"}` (one literal JSON backslash before `i`) fails JSON parsing because `\i` is not a JSON escape. Correct raw JSON needs two bytes `\\` in that position and parses into one in-memory backslash.

There is a subtler dangerous case: raw JSON `"\\frac"` is correct and parses to the six-character in-memory string `\frac`; raw JSON `"\frac"` is syntactically valid but parses `\f` as U+000C FORM FEED followed by `rac`. The normalizer (`app/online/models.py:_clean_str`) rejects controls below U+0020 except TAB/LF/CR, so the field becomes empty and its question fails structural normalization. The same applies to `\text` (`\t` → TAB, which is currently allowed), `\begin` (`\b` → BACKSPACE, rejected), and some malformed control sequences. Pydantic is not the component that rejects the parsed form-feed in this path; `_clean_str` is. Allowing TAB means a malformed `\text` can survive JSON parsing/normalization as a tab plus `ext`, which is a confirmed normalization gap.

`_clean_str` runs **after** parsing. It safely repairs only bare word `infty` to `\infty`; it does not broadly rewrite LaTeX. That limited repair cannot recover `\frac` after JSON has already converted `\f` to U+000C. It is not evidence that normalization caused the original escaping loss. It can, however, change ordinary prose containing the standalone word `infty`; that is a small false-positive risk.

### Required representation trace

| Stage | Correct `\\frac{1}{2}` example | Single-backslash `\frac` raw JSON example | Evidence |
|---|---|---|---|
| Raw LLM structured text | JSON bytes contain `"\\\\frac{1}{2}"` | JSON bytes contain `"\\frac{1}{2}"` | Provider output is held as `previous_output`; raw job output unavailable. |
| JSON/Pydantic parse | Python string is `"\\frac{1}{2}"` | Python string is U+000C + `rac{1}{2}` | `model_validate_json` follows JSON rules. |
| Question normalization | Preserved, trimmed | `_clean_str` returns empty because U+000C is forbidden | `app/online/models.py`. |
| Candidate storage | Stored only after normalization/binding | Candidate omitted or structurally rejected | `split_valid_invalid` / generation loop. |
| Browser preview | Delimited math is sent to KaTeX as `\frac{1}{2}` | No valid candidate should reach preview | `FrontEnd/js/math-renderer.js`. |
| PDF/DOCX | Simple math can become Unicode/plain text; complex math becomes raster image | No valid candidate should reach export | exporters use `split_equation_content`, `latex_to_plain_text`, `render_math`. |

For `(-\infty,2)`, the correct raw JSON has `(-\\infty,2)` and the parsed/stored text has a single backslash. A raw single backslash before `i` is invalid JSON, unlike `\frac` which is syntactically valid-but-corrupted because `\f` is legal JSON. This distinction is essential.

### What is and is not proven for this job

The job summary identifies repeated malformed infinity notation and 11 Generator calls, but it supplies neither raw response bytes nor structured-attempt failure categories. Therefore **invalid JSON escaping is a credible, code-supported mechanism, not a confirmed cause of this job’s retries**. The code proves malformed `\frac`/`\begin` can trigger a full item rejection after parsing. It does not prove that occurred for `m1_mcq_7`.

The full provider strict-output path preserves a correctly escaped slash: it stores provider text, validates it as JSON/Pydantic, and passes the resulting Python string forward. There is no post-parse “double unescape” in the inspected generator path. The local legacy `extract_json` path also uses `json.loads`, with the same semantics.

### Output contract and regression matrix

Adopt one mathematical-output contract for **mathematical spans only**: prose stays prose; mathematical content is Unicode notation or TeX delimited by `\\(...\\)` / `\\[...\\]`; every raw JSON backslash is doubled. Validate controls before persistence and reject/retry the individual invalid slot, not the entire bundle. Required offline cases:

| Case | Expected parsed/stored result |
|---|---|
| `\infty`, `\frac{1}{2}`, `\sqrt{x}`, `\text{speed}`, `\times`, `\leq`, `\geq` | one literal backslash before each TeX command; renderer gets the same text |
| `\begin{aligned}x&=1\\y&=2\end{aligned}` | preserved delimiter/environment and rendered as a multiline image where necessary |
| Arabic prose + `\\(x\\leq 2\\)` | Arabic text remains RTL, math LTR, no bidi corruption |
| Unicode `∞ ≤ ≥ × √` | preserved; renderer may normalize to TeX internally |
| raw invalid `\infty` JSON escape | structured parse failure, classified as invalid JSON |
| raw `\frac`, `\begin`, `\text` | parsed controls are rejected consistently; specifically reject TAB too when it is an escaped-command corruption |
| U+0000, U+0010, U+001C | rejected in every question/certificate string before candidate storage |

Tests must cover MCQ stem/options/certificate, Equation, Word Problem, and True/False statement/explanation/certificate. They should explicitly assert raw bytes, parsed Python `repr`, stored payload, browser DOM fallback state, PDF extracted/rendered output, and DOCX runs/images.

## C. Export diagnosis

No exported PDF/DOCX or browser-preview capture for this job was retained. Thus no expression can be assigned a proven first defective stage for this job.

| Expression | Raw LLM | Parsed / stored | Preview | PDF | DOCX | First defective stage |
|---|---|---|---|---|---|---|
| `(-\infty,2)` attributed to `m1_mcq_7` | unavailable | unavailable | unavailable | unavailable | unavailable | **Unconfirmed** |
| Valid `\frac{1}{2}` representative | unavailable | unavailable | unverified for job | unverified for job | unverified for job | **Unconfirmed** |
| Parsed form-feed + `rac` representative | unavailable | normalizer rejects it by code | cannot normally reach | cannot normally reach | cannot normally reach | JSON parse/normalization boundary (code-proven mechanism, not job event) |

The renderer paths are not identical. Browser preview renders only recognized delimiters with KaTeX; undelimited ordinary text remains literal text. PDF/DOCX split the same delimiter forms, convert simple TeX to editable Unicode/plain text, and rasterize complex TeX through Matplotlib. If rendering fails, browser preserves original text with `math-render-fallback`; PDF/DOCX can print the raw value. Therefore valid persisted but visibly raw `\frac` would be an export/preview delimiter-or-renderer defect, not a Generator defect. Conversely, a stored control character or bare malformed token is already a Generation/parsing defect.

Arabic is treated differently: preview gives prose RTL and math LTR; PDF/DOCX choose paragraph direction and isolate mathematical spans. The implementation has tests for Arabic inline math and complex displays, but no retained artifact proves this job’s Arabic output.

## D. Retry breakdown for `m1_mcq_7`

**Confirmed:** job telemetry records 11 Generator calls for `m1_mcq_7`; all persisted provider requests were successful at transport/provider level. Rejected quality/structure is not a failed provider call.

**Code-proven retry budgets (current source):**

- Provider structured-output loop: up to 3 attempts (`DEEPSEEK_MAX_STRUCTURED_ATTEMPTS`, `app/config.py`; `app/llm/deepseek.py`). It retries invalid JSON, schema validation, incomplete/failed statuses, and passes previous output plus failure reason to a corrective prompt.
- Generation bundle loop: up to 3 attempts (`_MAX_ATTEMPTS`, `app/online/exam_builder.py:_generate_type_from_plan`). It retries pending slots and adds only accepted-question anti-duplication context; it does not pass the malformed field or structured parser reason as dedicated feedback.
- Structural item repair: up to 2 calls for invalid normalized items (`_repair_invalid_items`).
- Downstream content repair, same-plan regeneration, replacement, difficulty adjustment, targeted revalidation, and duplicate repair each have their own bounded counters. Duplicate replacement is capped at 3 per slot and duplicate rounds at 3; ordinary failed content repairs default to 3, with a separate call ceiling of 8.

These independent budgets can multiply. A structured parse failure can consume as many as three provider requests before one bundle attempt is recorded as failed; the bundle can then repeat. A full `3 × 3 = 9` sequence is possible before structural-repair or downstream recovery. Eleven is consistent with such nesting plus two additional calls, but exact allocation is **unconfirmed** without the missing per-attempt telemetry. The model’s corrective structured prompt does receive previous output/failure reason; the bundle prompt does not reliably receive the specific escape/control diagnosis.

Smallest remedy: capture a slot-level attempt ledger (slot, raw-output hash, failure category, layer, input fingerprint, question-field paths); retry only invalid slots; for safe lexical defects repair the parsed field deterministically before regeneration; stop when the same failed fingerprint and failure class recur. Do not increase retry counts.

## E. Duplicate V2 policy diagnosis

Duplicate V2 is **not task-ID-only** in the current inspected code. `exam_builder._is_semantic_plan_duplicate` explicitly states task identity alone never makes realized questions duplicates. The Judge receives actual stem, answer/solution, task/variation IDs, effective instance ID, operation, target, and givens. Its classifier accepts an authorized practice variation where the core knowledge/operation/target are the same only when it is authorized and neither answer substantially gives away the other (`app/online/duplicate_judge.py`). This matches the requested 2x+4 versus 3x+6 policy in principle.

However, the policy remains under-specified at the contract boundary:

- `numeric_values_may_change` is passed in selected variation metadata, but there is no durable normalized *instance fingerprint* made from mathematical givens, unknown, operation, and result before Judge selection.
- The Judge prompt says changing only numbers is not genuinely new reasoning; the classifier then permits it only under the separate authorization flag. This is reasonable, but dependent on correct permission propagation.
- Same-task reuse is blocked earlier unless authorization metadata is present. Prior audited evidence documents a plan/snapshot inconsistency in that authorization gate. It needs a deterministic regression fixture before policy changes.
- Duplicate candidate screening/windowing can miss or over-prioritize based on compact metadata and a five-candidate selector window. It cannot establish that task ID by itself caused either job-specific conflict because detailed pairs are unavailable.

For `m2_mcq_6` and `m2_mcq_7`, the retained job summary confirms a duplicate conflict involving the same task/variation, but does not retain both student-facing candidates, effective instance IDs, numeric-permission flags, or Judge facts. It is therefore **unconfirmed** whether they were cosmetic paraphrases (properly rejected) or authorized distinct numeric instances (wrongly rejected).

## F. Duplicate Repair and Filter & Rank diagnosis

### What the code actually routes

Replacement candidate discovery uses `compatible_task_inventory` / catalog eligibility, filters blacklisted and previously failed identities, type compatibility, source availability, obvious collisions, and reservations. It ranks unused/least-used identity, coverage/concept/operation/target/family/mode diversity, then exposes only the first five candidates to the selector. The selector sees compact conflict edges, failed-slot context, relevant active slots/questions, prior failures, blacklist, and numerical/reuse metadata; it may return a valid explicit no-candidate, which Python respects. An invalid selector response or operational failure falls back to the first Python-ranked candidate.

Reservations are sequential and atomic in Duplicate Repair, so same-round candidates see prior reservations. Failed replacement task IDs are blacklisted. Candidate/assignment fingerprints and prior failure memory exist in ordinary validation recovery. These are positive safeguards.

### Confirmed weaknesses

1. **Same-task numeric repair is not a first-class pre-replacement route.** The code has a numeric variation fallback only at duplicate slot attempt 3, after two semantic replacement attempts. It mutates a question candidate but does not create/attach a fresh authoritative instance contract first. This is the exact incompatibility identified in the earlier persisted task-reuse audit: a numerically sound new instance can subsequently fail task alignment against unchanged fixed `required_givens`.
2. **Task/variation ID blacklisting is too coarse for authorized instances.** A failed task identity prevents reconsidering it even where a different authorized numerical instance could be valid. The implementation lacks an instance-level blacklist keyed to canonical givens/unknown/result/reasoning.
3. **Selector window is only five candidates.** Better candidates beyond the compact window can be overlooked. The code contains no proof that they were for this job, so that specific claim remains unconfirmed.
4. **No explicit graph-exhibit capability filter in replacement eligibility.** `text_answerable` and source evidence are checked, but a task can carry graph-related givens and still be selected without a student-visible graph.

`m2_true_false_1` repeatedly returned to task/variation according to the job summary, but its replacement-attempt ledger is unavailable. We cannot truthfully provide the requested candidate-by-candidate table. The evidence limitation itself is a telemetry defect: the record must persist, for each attempt, candidate task/variation/instance policy, source/exhibit availability, Python eligibility/rejection reasons, selector response, final selection, generated candidate fingerprint, validator findings, and avoidability.

## G. Graph-capability audit

The intended policy is partly expressed but not enforced strongly enough. Catalog/prompt metadata has `text_answerable`, `required_givens`, `unsupported_visual_tasks`, and visual-cue checks. Planner rejects tasks marked not text-answerable or visual dependencies without sufficient givens. Generator requires a self-contained exhibit. These are useful, but they conflate “source has information about a graph” with “DraftWork can present a graph.”

Current leak paths:

1. Compiler/LLM can mark a graph-interpretation task `text_answerable=true` if it has some givens, despite requiring supplied visual geometry/vertices.
2. `_has_visual_dependency` gates only visual cue + absent givens; it does not require an explicit `student_visible_exhibit` available in the product.
3. Planner/replacement eligibility trusts that metadata; it has no product capability flag for supplied graph images.
4. Generator is told an inline figure may be printed, and past artifacts show ASCII graph construction. That is incompatible with the stated product constraint if the task requires students to inspect a supplied graph; it is not proof that a graph image can be delivered.
5. Validator can detect missing visual context only after expensive generation; it cannot authorize changing interpretation into construction, and Repair must not do so absent an allowed variation.

Supported categories must be represented structurally, not by the word “graph”:

| Operation | Current product decision | Required contract |
|---|---|---|
| Interpret supplied graph (read vertices/intersection/shape) | Reject unless complete student-visible non-graph data makes it equivalent | `requires_exhibit=true`, `exhibit_kind=graph`, `exhibit_available=false` → ineligible before planning |
| Construct graph from equations/data | Allow | `student_work_product=construct_graph`; equations/data visible |
| Graph concept | Allow if source-supported | `requires_exhibit=false`; concept/operation stated |

The terminal `m2_mcq_7` is confirmed evidence of the first leak: a graph-vertices replacement reached recovery with neither graph nor coordinates. It should have been ineligible before Generation.

## H. Root-cause classification

| Issue | Classification | Basis |
|---|---|---|
| Correctly escaped TeX supported by structured JSON | Expected behavior | JSON/Pydantic path preserves a parsed backslash |
| `\frac`/`\begin` single-backslash JSON turns into controls then rejects item | Confirmed JSON/schema/normalization defect | JSON semantics plus `_clean_str` behavior |
| TAB-producing malformed `\text` can survive control filter | Confirmed JSON/schema/normalization defect | `_clean_str` permits TAB |
| Job-specific malformed LaTeX caused 11 calls | Unconfirmed due to missing raw attempt data | Summary signals issue; raw events absent |
| Nested retries can amplify one bad field | Confirmed Python routing/retry defect | independent bounded provider/bundle/repair routes |
| Export caused this job’s bad PDF math | Unconfirmed due to missing artifacts | renderer paths inspected, job artifact absent |
| Valid persisted raw TeX can display literally if delimiter/render fails | Export rendering risk, not job-confirmed | preview/export fallbacks |
| Duplicate V2 rejects solely due to task ID | Not confirmed; current code contradicts it | Judge/classifier uses content and authorization |
| Same-task valid numeric instance lacks first-class repair contract | Duplicate-policy and Python routing defect | numeric fallback late and plan unchanged |
| m2 pair incorrectly rejected despite valid variation | Unconfirmed due to missing pair records | no candidates/Judge facts retained |
| Five-candidate window can overlook alternatives | Confirmed Filter & Rank design limitation; job impact unconfirmed | selector code |
| Graph interpretation can reach Generation without displayable exhibit | Confirmed Filter & Rank/catalog capability defect | metadata gate lacks exhibit availability; terminal evidence |
| Two omissions | Source-evidence limitation plus recovery-routing weakness | job runtime report |

## I. Prioritized minimal implementation plan

1. **P0 — Establish a shared math-string boundary.** Add a pure local validator/normalizer before candidate acceptance: reject all C0 controls (including TAB when arising in TeX-bearing fields), preserve correct strings byte-for-byte, recognize only narrowly safe repairs such as bare `infty`, and return field paths/reasons. Use it for every student-facing math field and certificate. Regression: the matrix in section B, plus `repr`/byte assertions.
2. **P0 — Make retries slot- and failure-aware.** Persist structured attempt records and hash raw output; classify invalid JSON versus parsed control versus Pydantic schema versus content failure. Run deterministic field repair for safe recoverable math text; regenerate only affected slots. Stop exact no-progress repeats. Regression: one malformed option yields bounded slot-only retry and keeps valid bundle siblings.
3. **P0 — Enforce graph capability at catalog and eligibility time.** Add `student_operation_kind`, `requires_student_visible_exhibit`, `exhibit_kind`, and `exhibit_available`. Reject unavailable graph interpretation before planning/replacement; allow construction/concept. Regression: vertices-from-graph rejected; equations-to-graph and slope-concept accepted; no Generator call for rejected task.
4. **P1 — Make numerical instances first-class.** Catalog/plan must separate invariant givens from variable parameters, selected assessment angle, and `effective_instance_id`. Create an authorized same-family/same-variation instance brief before catalog replacement; recompute answer/options/certificate and attach the plan revision. Regression: 2x+4/12 vs 3x+6/21 accepted only with permission; paraphrase rejected; exact-source instance remains immutable.
5. **P1 — Align Duplicate V2 and Filter & Rank.** Duplicate decision should compare operation, target, canonical mathematical givens/unknown/result, and reasoning—not task ID. Blacklist instance fingerprints, not an entire task when additional instances are permitted. Feed failed instance fingerprints and full conflict group into ranking; expand/evaluate the candidate window deterministically before the LLM and preserve a true no-candidate. Regression: same task + different authorized instance, same task + wording-only clone, exhausted source, reservation collision, and selector no-candidate.
6. **P1 — Preserve bounded recovery and prove routing.** Keep independent ceilings, but give same-task instance repair its own one-per-fingerprint budget before replacement; retain targeted revalidation and incremental Duplicate V2 after any content change. Regression: a duplicate repaired by an authorized numeric instance succeeds without catalog replacement; repeated fingerprint terminates; failed replacement cannot loop.
7. **P2 — Repair rendering only where evidence proves it.** Add cross-format golden artifacts for valid stored TeX, Arabic/mixed spans, MCQ options, answers, and certificates. Browser must mark fallback; PDF/DOCX must not silently print raw TeX where rendering is expected. Do not change exporter behavior until an actual persisted-valid/job-export mismatch is captured.

## J. Evidence gaps that must be closed

For a future forensic report to answer the requested per-attempt questions exactly, retain (with safe redaction): raw output or encrypted artifact hash/reference; parsed canonical candidate revision; normalization failures by JSON path and codepoint; provider structured-attempt category/count; bundle/repair/replacement route; task/variation/effective-instance ID; numerical permission; exhibit capability; selector candidate table and decision; validator verdicts; preview/PDF/DOCX artifact IDs and renderer fallback events. This is observability, not a request to rerun this job.

