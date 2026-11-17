# DraftWork HTML-to-PDF production migration

## Status

The production PDF export code now uses the shared Exam Review HTML renderer and headless Chromium. The 915-line custom FPDF renderer was deleted. Existing API routes, ZIP filenames, model selection, student PDFs, answer-key PDFs, multiple models, DOCX export, Document IR, and the frontend download flow remain intact.

One deployment verification remains incomplete: Docker Desktop/the Docker engine is unavailable on this workstation, so the updated image could not be built and executed locally. The Dockerfile installs the version-matched Playwright Chromium runtime and Linux dependencies, but CI or a Docker-capable host must execute the final image smoke test before release.

## New architecture

```text
Saved exam
  -> canonical backend Document IR
  -> FrontEnd/print.html
  -> the same exam-view.js renderStudent/renderAnswers functions as Exam Review
  -> the same styles.css + local KaTeX assets
  -> print.css pagination rules
  -> reused headless Chromium browser
  -> PDF bytes
  -> unchanged SmartExam_Export.zip API response
```

Python constructs canonical Document IR, as it already does for the browser Preview, but performs no question layout, Arabic shaping, line breaking, equation drawing, table drawing, or page composition.

Private exam content is injected directly into an isolated browser context. The print page loads from the repository through `file://`; every non-local request is blocked. No private exam URL or externally accessible print endpoint is created.

## Production converter

- `app/exports/html_pdf.py:54-198`
  - Lazy, reusable Playwright/Chromium process.
  - New browser context and page per PDF.
  - Configurable bounded concurrency; default `2`, maximum `4`.
  - Configurable timeout; default `90` seconds.
  - Waits for the print bridge, fonts, images, and synchronous KaTeX output.
  - Compares expected Document IR item count with rendered DOM item count before accepting output.
  - Validates the `%PDF` signature.
  - Closes each context in `finally` and closes Chromium during ASGI shutdown.
  - Blocks all non-`file:` network requests.
- `FrontEnd/print.html` is a data-free print shell using local CSS and KaTeX.
- `FrontEnd/js/print-view.js` is a small bridge into the shared renderer.
- `FrontEnd/js/exam-view.js:314-323` exposes `renderPrintDocument` and continues to use the existing `renderStudent` and `renderAnswers` implementations.
- `app/api/export_routes.py:88-112` renders selected model/student/answer pairs concurrently and preserves the existing archive contract.
- `app/api/job_routes.py:549-558` uses the same archive builder for authenticated PostgreSQL exams.

## Security and reliability

- Saved records are read but never modified during export.
- No public print-data route exists.
- Browser contexts isolate concurrent exports.
- Only repository-local files are allowed to load.
- A render timeout returns a controlled export failure instead of leaving the request running indefinitely.
- Context cleanup occurs on success, validation failure, timeout, or Chromium exception.
- The reusable browser is closed in FastAPI lifespan shutdown.
- TestClient/CLI event-loop changes are detected so stale Playwright channels are not reused.

## Pagination changes

`FrontEnd/css/print.css` adds print-only behavior while preserving the Preview DOM and base styling:

- A4 page margins reserve safe top and footer regions on every physical page.
- White backgrounds and auto-height papers prevent dark overflow regions.
- Section headings stay with following content.
- Student questions, stems, options, question/table grids, tables, and answer rows avoid internal breaks when they fit on a page.
- Answer rows use a stable two-column print grid for inline number/content alignment.
- Response lines use a print-specific 6 mm height.
- Orphan/widow protection is enabled for question and answer prose.
- The Preview's literal page marker is hidden during export.
- Chromium footer templates produce `Page X of Y` and `صفحة X من Y` using Western digits.

The saved 23-question Arabic Model 2 produces five student pages. The fifth page contains the final word problem and its working space. It is intentionally sparse because moving it back would either split the question/response area or remove useful student writing space.

## Shared Preview correction

The migration exposed an existing Preview defect: the class label was shown but the saved `class_name` value was omitted. `FrontEnd/js/exam-view.js:261-270` now renders the value. This fixes both browser Preview and PDF because they share the same DOM implementation.

## Removed legacy implementation

- Deleted `app/exports/pdf_exporter.py` (915 lines).
- Removed `fpdf2`.
- Removed `uharfbuzz`, which had no remaining consumer after FPDF deletion.
- Removed all imports and references to `pdf_exporter` from application code, tests, manual smoke tools, and artifact scripts.
- Retained `app/exports/math_renderer.py` because DOCX and Document IR still use `latex_to_plain_text` and DOCX image fallback.
- Retained DOCX exporters, shared content parsing, common export helpers, and Document IR.

## Docker/runtime changes

- `requirements.txt` adds `playwright>=1.50,<2` and removes FPDF/HarfBuzz.
- `Dockerfile` runs `python -m playwright install --with-deps chromium` before dropping to the non-root application user.
- `compose.yaml` exposes `PDF_EXPORT_CONCURRENCY` and `PDF_EXPORT_TIMEOUT_SECONDS` for the API service.
- `app/config.py:34-41` validates concurrency, timeout, and optional local Chromium executable configuration.

The worker image also contains Chromium because the repository currently uses one shared Dockerfile. It does not launch Chromium unless PDF export code is called.

## Tests

### Full suite

```text
1447 passed
21 skipped
1 failed
1 warning
149.43 seconds
```

The only failure is the pre-existing optional historical-database fixture:

```text
tests/test_graph_pipeline_integration.py::
test_saved_103_chunk_catalog_mocked_graph_reaches_final_assembly
```

It expected 103 persisted child chunks for document `175ef217-a754-4e58-8bae-2bbb4841f824`; the configured PostgreSQL database returned zero. This is unrelated to PDF export and matches the previously observed environment failure.

### Final focused suite after pagination correction

```text
15 passed in 24.00 seconds
```

Covered:

- Legacy and authenticated archive contracts.
- Student/answer separation.
- Model selection and multiple archive entries.
- English all-question-type rendering.
- Arabic/mixed-direction rendering.
- KaTeX fractions and systems.
- Tables.
- Correct class metadata.
- A 40+ question PDF.
- Local print-shell asset reuse.

### Real saved exam

PostgreSQL contained six completed exams, all Arabic. The final visual regression used:

- Exam `exam_1b866d011562`.
- Saved `2026-11-15 22:18:21 +04:00`.
- Model 2.
- 10 MCQ, 5 True/False, 4 Equation, 4 Word Problem.
- 23 student questions, 23 answer rows, mathematical systems/fractions, mixed Latin/Arabic runs, and two tables.

All five student pages and two answer pages were rasterized and visually inspected. No overlap, clipping, dark overflow area, disconnected Arabic shaping, missing table, missing question, missing answer section, or footer collision remained. Bounding-box inspection found content within the reserved print region on every page.

No completed English exam exists in the current PostgreSQL database. English was therefore verified with a deterministic all-types regression fixture, not falsely reported as a saved PostgreSQL exam.

## Benchmark

Fixture: saved Arabic `exam_1b866d011562`, Model 2. Four sequential outputs alternated student and answer PDFs.

| Measurement | Legacy FPDF | Chromium HTML-to-PDF |
|---|---:|---:|
| Average per PDF | 2.516 s | 0.950 s |
| Cold start | Not isolated | 1.645 s |
| Warm average | Not isolated | 0.718 s |
| Student PDF size | 115,415 bytes | 171,596 bytes |
| Answer PDF size | 61,640 bytes | 102,995 bytes |
| Observed process-tree RSS | 147,738,624 bytes | 676,425,728 bytes |
| Concurrent student+answer wall time | Not measured | 1.099 s |

RSS values are observed process-tree snapshots rather than instrumented high-water marks. Chromium is faster on this fixture and produces the preferred visual result, but uses roughly 4.6x the observed process memory and larger files. The default concurrency of 2 is intentionally conservative.

Raw benchmark data: `reports/html_pdf_migration_benchmark_2026-11-16.json`.

## API and frontend compatibility

- `POST /exams/{exam_id}/export/pdf` remains available for the compatibility API.
- `POST /api/v1/exams/{exam_id}/export/pdf` remains available for authenticated saved exams.
- Request body `model_numbers` is unchanged.
- Response remains `SmartExam_Export.zip`.
- Per-model student and answer filenames are unchanged.
- Frontend export buttons require no contract change.
- DOCX endpoints and output remain unchanged.

## Remaining limitations

1. Docker image execution is not verified on this workstation because no Docker engine is available. This must be completed before deployment.
2. No saved English PostgreSQL exam was available; the English sample is deterministic test data.
3. Chromium uses substantially more memory than FPDF. Concurrency should be raised only after load testing on the deployment host.
4. KaTeX is visually correct, but PDF text extraction can expose mathematical glyphs separately or Arabic footer text as presentation forms. Search/accessibility quality for mathematics remains a separate concern.
5. A final question with substantial response space may occupy a sparse last student page by design.

## Release recommendation

The application migration and local verification are complete. Do not deploy until the updated Docker image is built and the authenticated export endpoint is smoke-tested inside that image on a Docker-capable CI/host. No fallback to the deleted FPDF renderer should be added; deployment failure should be fixed in the Chromium packaging/runtime configuration.

