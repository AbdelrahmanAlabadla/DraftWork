# DraftWork `math (2).pdf` production exam run

## Overall result

- **Job:** `4e7a3194-3eea-49cc-8bde-8ff7708b91fd`; **exam:** `exam_dea1114316a0`.
- **Document:** `8928f477-3481-47ed-ac0d-1bb855f389b2`, restored from historical processed-cache key `9b7c21316e78ec49cb5cf54b48b5e687183729aff44330392167772ad9e12115`.
- The normal retrieval path wrote **20 parent and 89 child** cached chunks to Docker PostgreSQL at 18:22:14 UTC. Retrieval selected all 89 children in 0.41 s. No LlamaParse, embedding regeneration, original Windows PostgreSQL (`:1966`), document reprocessing, deletion, or overwrite was used.
- The job was queued for **5.68 s**, ran from 18:22:12.964 to 18:38:47.287 UTC, and completed in **16m 34.3s of worker execution** (16m 40.0s from job creation). The job record is `completed`; the saved exam is deliberately **partial**.
- Requested 23; finalized **20**: MCQ **10/10**, true/false **5/5**, equation **4/4**, word problem **1/4**. Quality summary: valid 20, invalid 2, missing 1.
- Database-estimated API cost: **$0.17872**. All 102 logged LLM calls used `openai / gpt-6-luna`. Provider call durations sum to 2,099.7 s; this is deliberately not total runtime because many calls were concurrent.

## Timeline and handoffs

| UTC interval | Wall time | Stage and handoff |
|---|---:|---|
| 18:22:07.280–18:22:12.964 | 5.68 s | Job persisted and waited in the Celery queue. |
| 18:22:12.964–18:22:14.303 | 1.34 s | Job setup, cache restoration, PostgreSQL retrieval. 89/89 selected child chunks supplied to catalog compilation. |
| 18:22:14.303–18:24:44.397 | 2m30.1s | Assessment Catalog: ten parallel compiler calls produced 48 tasks. No catalog cache hit. |
| 18:24:44.407–18:26:14.141 | 1m29.7s | Planner allocated 23 slots. Plan Review evaluated 32 shortlisted pairs out of 253 possible; four sequential review passes occurred. |
| 18:26:14.154–18:27:24.136 | 1m10.0s | Difficulty Adjustment made three concurrent batch calls, then Plan Review rechecked the adjusted plan. |
| 18:27:24.150–18:30:32.549 | 3m08.4s | Generator produced 21 questions: all 19 MCQ/TF/equation slots and 2/4 word-problem slots. Two word-problem slots failed the semantic-plan-duplicate gate through three normal attempts and two recovery batches. |
| 18:30:32.564–18:31:05.286 | 32.7s | Initial general validation (21 questions) plus quantitative validation (9 quantitative questions), five calls concurrently. General rejected 2; quantitative rejected 3. |
| 18:31:05.288–18:34:47.332 | 3m42.0s | Repair and revalidation cycles. Five MCQs were repaired; a recovered word problem was generated and repaired. The pipeline serially revalidated individual repairs, then reached 22 working questions. |
| 18:34:47.335–18:38:47.175 | 3m59.8s | Duplicate scans/recovery. Scan 1 assessed 32 candidates and found two duplicate groups; scan 2 found one; scan 3 found one; scan 4 found none. Replacement, difficulty adjustment, validation, and repair introduced repeated work. |
| 18:38:47.228–18:38:47.287 | 0.06 s | Final assembly and PostgreSQL persistence. Three word-problem slots remained unresolved, so the exam was saved partial. |

Calls had zero recorded scheduler wait time. Concurrency was active: catalog (10), initial validators (5), and several replacement/validator groups ran in parallel. The main waits were sequential review/repair loops and slow provider responses, not Celery or scheduler queuing.

## Agent evidence and cost

| Agent / operation | Calls | Input / output / cached / reasoning tokens | Provider duration sum | Cost |
|---|---:|---:|---:|---:|
| Assessment compiler | 10 | 62,124 / 67,420 / 9,594 / 26,782 | 1,055.4 s | $0.03906 |
| Plan reviewer | 5 | 144,820 / 7,356 / 0 / 3,669 | 104.8 s | $0.01816 |
| Difficulty adjuster | 9 | 23,186 / 8,699 / 1,719 / 4,696 | 108.8 s | $0.00651 |
| Generator | 22 | 131,432 / 31,978 / 50,720 / 15,630 | 326.4 s | $0.02457 |
| Validator | 36 | 614,662 / 30,124 / 183,761 / 13,453 | 398.6 s | $0.05999 |
| Repairer | 9 | 75,826 / 4,375 / 0 / 1,816 | 58.7 s | $0.00977 |
| Duplicate replacement selector | 5 | 131,275 / 3,469 / 0 / 2,059 | 41.6 s | $0.01486 |
| Duplicate judge | 6 | 17,715 / 8,056 / 0 / 5,863 | 105.4 s | $0.00580 |

The compiler was the slowest stage by elapsed wall time (its longest parallel call took 149.9 s). Validation was the most expensive operation ($0.05999) and sent the most input tokens. The repeated post-generation validations are the clearest avoidable runtime and cost concentration.

## Task flow and question result

The catalog supplied 48 tasks. The planner assigned all 23 requested slots; the ten MCQs, five true/false questions, and four equations received distinct catalog task IDs. All four word-problem slots were assigned the same authoritative tennis task (`task_fedb2995e754f86a`): 13 matches, 38 players, 2 players in singles, 4 in doubles, requiring 7 singles and 6 doubles.

| Slots | Result | Validation / repair evidence |
|---|---|---|
| MCQ 1–5 | retained | No final invalid finding. |
| MCQ 6–8 | retained | Quantitative validator initially rejected answer fields; repairer applied `FIX_ANSWER`; general and quantitative revalidation passed. |
| MCQ 9–10 | retained | General validator initially rejected options; repairer applied `FIX_OPTIONS`; revalidation passed. |
| TF 1–5 | retained | No repair required. |
| Equation 1–4 | retained | No final invalid finding. |
| Word problem 1 | omitted invalid | Numeric duplicate fallback changed source givens to 16 matches/44 players. General validation correctly flagged `task_drift`; max repair attempts were exhausted. |
| Word problem 2 | missing | Never emitted for its requested slot: the generator rejected it as a semantic plan duplicate in all normal and recovery attempts. |
| Word problem 3 | retained | Uses the required 13/38 tennis givens and correctly derives 7 singles and 6 doubles. |
| Word problem 4 | omitted invalid | Numeric duplicate fallback changed source givens to 17 matches/50 players. General validation correctly flagged `task_drift`; max repair attempts were exhausted. |

The final retained questions independently check out mathematically. MCQ answer keys are single-answer and correct; the Arabic true/false claims are correctly keyed; equations give correct results: `y=3x-8`, `(2,3)`, `(1,3)`, and `(2,1)` is not a system solution. The retained tennis problem correctly solves `s+d=13`, `2s+4d=38` to `s=7`, `d=6`. Arabic is suitable for the material, while Latin variables and Western digits are preserved. LaTex is valid in the persisted question objects. The large shortfall is completeness, not a detected error among the 20 retained questions.

## Duplicate and repair findings

- Initial plan screening had 253 possible slot pairs, shortlisted 32 using semantic similarity, and sent reviewed outcomes into the planner/difficulty handoff. It recorded six issues and 61 plan-review history records; diversification was eventually exhausted, with an auditable warning rather than a hidden substitution.
- Actual-question duplicate scans considered 32, 31, 31, and 30 semantic candidates. Fact-cache results across scans were 0/32, 26/5, 31/0, and 28/2 hits/misses. Four judge calls ran in the first scan and two in the final scan; the audit retains the judge verdicts and duplicate groups.
- Duplicate recovery attempted catalog replacement, then reused the tennis task after catalog exhaustion, then fell back to numeric variations. That fallback directly caused the two source-drift failures. The final scan had zero duplicate groups, but the affected slots were invalid because their new numbers no longer matched the assigned source task.
- General validation passed those variations in places where quantitative validation later rejected them. This shows that `PASS` was not treated as conclusive in this review.

## Bottlenecks and recommendations (not implemented)

| Priority | Root cause and evidence | Location to inspect | Recommended change | Benefit / risk |
|---|---|---|---|---|
| P0 | Four planned word-problem slots shared one fixed-givens task. Semantic duplicate filtering rejected normal generation repeatedly; numeric fallback then drifted from 13/38 source facts. | `app/online` planner, duplicate recovery, and word-problem generation paths; audit task `task_fedb2995e754f86a`. | Reserve distinct compatible catalog tasks before planning four word-problem slots; disallow numeric variation when the task has immutable source givens. | Prevents the 1/4 word-problem result. Risk: more explicit shortage reporting when the catalog lacks distinct source-backed tasks. |
| P1 | Validation/repair repeated 36 validator calls and 9 repairer calls. Several final cycles revalidated one or two questions. | `app/jobs/tasks.py` generation/repair orchestration and validator batching. | Batch compatible repaired questions and stop a slot once its authoritative task cannot be preserved. | Lower latency/cost; risk: batching must keep general and STEM verdict provenance per question. |
| P1 | Ten catalog calls consumed the first 2.5 minutes; the tail was 149.9 s. | Assessment catalog compiler invocation and cache-key selection. | Reuse a validated catalog keyed to immutable document/chunk selection, or bound/fan out catalog batches only when needed. | Faster first run; risk: cache invalidation must include source/chunk selection. |
| P2 | Five serial Plan Review calls consumed 104.8 provider seconds and processed 144,820 input tokens. | Plan-review loop and candidate shortlist policy. | Review only changed/conflicting pairs after a repair; preserve prior verdicts. | Reduced repeated context; risk: stale verdicts if task content changes materially. |
| P2 | Historical cache key differs from the newly computed key, though restoration succeeded via the stored historical key. | Processed-cache key computation and retrieval backfill. | Preserve legacy-key aliases or migrate cache metadata explicitly. | More reliable restoration; risk: key aliasing must validate source hash and parser version. |

## Assessment against the earlier benchmark

This run is **20/23** in 16m40.0s from job creation, compared with the earlier **19/23** in 16m43s. MCQ improved from 8/10 to 10/10; true/false stayed 5/5; equations improved from 3/4 to 4/4; word problems fell from 3/4 to 1/4. The runtime is 3.0 s shorter. This is indicative only, not a controlled comparison: the original benchmark chunk selection was not recovered, and this run used the restored 89-child cache.

## Migration and cache assessment

The PostgreSQL migration and runtime retrieval path worked for this run. The cache was materialized into `document_chunks`, then normal PostgreSQL retrieval supplied all selected context. The migration did not need a Qdrant fallback, and no old Windows database was contacted. The processed cache is a source for restoration, not the PostgreSQL chunk store. The restored document is therefore recoverable from its existing cache and is now present in Docker PostgreSQL.
