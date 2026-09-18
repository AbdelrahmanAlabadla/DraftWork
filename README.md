# DraftWork — AI Exam Generator
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LLM](https://img.shields.io/badge/LLM-Powered-purple)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-black)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)
![Document AI](https://img.shields.io/badge/Document_AI-PDF_Processing-blueviolet)
![Reliability](https://img.shields.io/badge/Reliability-Validation_%26_Repair-orange)
An AI-powered exam generation system that transforms educational PDF documents into structured, configurable exams.

The system combines document parsing, semantic chunking, embeddings, selected-section loading, LLM-based planning, question generation, validation, and targeted repair to keep questions grounded in the selected source material.

## Features

* Upload and process educational PDF documents
* Select specific sections to include in the exam
* Choose exam difficulty
* Generate multiple exam models
* Configure question counts by type
* Supports:

  * Multiple Choice
  * True / False
  * Fill in the Blank
  * Short Answer
  * Essay
* Load only selected child chunks for generation
* Build a structured question plan with stable plan-slot IDs
* Generate each question from its assigned concept, difficulty, and source chunk
* Retry only missing plan items when generation falls short
* Automatically validate generated questions
* Repair only invalid questions instead of regenerating the entire exam
* Revalidate only the question IDs changed by each repair pass
* Track generation, validation, repair, and final outcome telemetry
* Review overall and per-model performance in a read-only Eval Dashboard
* Separate question-quality failures from validator operational failures
* Export each exam model and its answer key as separate PDF or DOCX files

## Application

### 01 — Upload Document

Upload the educational PDF that will be used as the exam source.

![Upload Document](doc/images/01-upload.png)

### 02 — Exam Settings

Choose the number of exam models and difficulty level.

Available difficulty options:

* Easy
* Medium
* Hard
* Mix

![Exam Settings](doc/images/02-exam-settings.png)

### 03 — Choose Sections

Select the exact document sections that should be included in the exam.

This keeps the generation context focused only on the material selected by the user.

![Choose Sections](doc/images/03-section-selection.png)

### 04 — Question Types

Configure the number of questions required for each type.

![Question Types](doc/images/04-question-types.png)

Supported question types include:

| Question Type     | Description                          |
| ----------------- | ------------------------------------ |
| Multiple Choice   | Four options with one correct answer |
| True / False      | Quick factual recall                 |
| Fill in the Blank | Key-term recall                      |
| Short Answer      | Concise reasoning questions           |
| Essay             | Open-ended responses                 |

### 05 — Eval Dashboard

![Eval Dashboard](doc/images/eval_dashboard.png)

DraftWork includes a read-only Eval Dashboard for monitoring the quality and behavior of the exam-generation pipeline.

The dashboard tracks:

* exam runs, counted as the total number of generated models
* questions requested
* questions generated on the first attempt
* missing questions and shortfall recovery
* first-pass validation results
* validation failure reasons
* validator operational failures such as missing verdicts
* questions sent to repair
* repair success and failure
* final valid, invalid, unvalidated, and missing questions
* overall and per-model performance
* recent exam runs

This makes it easier to see where the pipeline is spending time and where failures occur.

A final exam can have a high success rate while still requiring retries, extra validation calls, or repair steps. The telemetry helps show whether the main bottleneck is generation, validation, repair, or model reliability.

It also helps compare whether the model used for generation, validation, or repair is actually performing well for that task or relying heavily on retries and repair.

Open the dashboard at:

```text
http://localhost:8000/eval.html
```

Telemetry is also available through:

```text
GET /api/eval-summary
```

The dashboard refreshes automatically every 30 seconds and can also be refreshed manually.

Evaluation history is stored persistently in PostgreSQL. Each database row represents one generation request, which may contain more than one exam model. For that reason, **Exam Runs** is the total number of generated models across all stored requests, rather than simply the number of database rows. Dashboard totals and rates are calculated from the saved raw counters.

The versioned `/api/v1` workflow stores sessions, documents, jobs, and generated exams in PostgreSQL. Upload parsing and exam generation run in a Celery worker through Redis, so API restarts do not lose job or exam state. The old synchronous endpoints can be enabled temporarily with `ENABLE_LEGACY_SYNC_API=true`; they are disabled by default.

Saving evaluation telemetry is best-effort. If PostgreSQL is temporarily unavailable, the generated exam is still returned to the teacher and remains available for export, but that generation run may not appear in the Eval Dashboard.

### 06 — Export Exams

PDF and DOCX exports are downloaded as `SmartExam_Export.zip`. Every selected model produces two separate documents: one student exam file and one teacher answer file. Answers are never appended to the exam document, and different exam models are never merged into one document.

When one model was generated, DraftWork starts the export without showing a model-selection dialog and packages the exam and answer file together in the ZIP. When several models were generated, the teacher selects which models to export. The matching answer file is included automatically for every selected model.

Export filenames use the **Exam Title** and **Class** entered under Printed Exam Details. For example:

```text
Biology_Midterm_Grade_10_Model_1.pdf
Answers_Biology_Midterm_Grade_10_Model_1.pdf
```

If only one of those details is present, the filename uses that value. If neither is present, DraftWork falls back to `Exam_Model_1.pdf` and `Answers_Exam_Model_1.pdf`. The same naming rules apply to DOCX exports.

## Sample Generated Exam

Below is a sample exam generated by DraftWork from the **Cambridge International AS and A Level IT Coursebook**.

The generated exam includes:

* Multiple Choice questions
* Fill in the Blank questions with a word bank
* True / False questions
* Short Answer questions
* Essay questions
* A separate teacher answer file

You can view examples of the generated document formatting in either format:

* [View Generated IT Exam — PDF](output/pdf/exam_exam_430306ff4a9f_matched.pdf)
* [View Generated IT Exam — DOCX](output/docx/exam_exam_430306ff4a9f_matched.docx)

### Source Material

The sample exam was generated from:

**Paul Long, Sarah Lawrey, and Victoria Ellis. _Cambridge International AS and A Level IT Coursebook_. Cambridge University Press, 2016. ISBN: 978-1-107-57724-4.**

The source textbook is used only as input material to demonstrate DraftWork's document-processing and exam-generation workflow. The textbook itself is not included in this repository.

## Architecture

```mermaid id="mz5m42"
flowchart TD
    A[PDF Upload] --> B[Parsing & Cleaning]
    B --> C[Semantic Chunking]
    C --> D[Embeddings]
    D --> E[(Qdrant Vector Store)]

    E --> F[Selected Child Chunk Loading]
    F --> G[Planner: Structured Plan Items]
    G --> H[Generator: Questions by Plan Slot]
    H --> I[Validator: Structured Verdicts]

    I -->|Valid| J[Final Exam]
    I -->|Invalid| K[Repairer: Targeted Fields]
    K -->|Repaired IDs only| I

    G -.-> L[Shared LLM Factory]
    H -.-> L
    I -.-> L
    K -.-> L
    L --> M[LM Studio Client]
    L --> N[DeepSeek Client]
```

Planner, Generator, Validator, and Repairer all use the shared LLM factory. The
factory selects LM Studio or DeepSeek from `LLM_PROVIDER`, so agent code does not
need provider-specific branches. Agent responses are checked against Pydantic
models before entering the pipeline. DeepSeek sends the corresponding JSON Schema
through the Responses API for native structured output and then applies the same
local Pydantic validation.

## Planning & Generation

Planner creates one structured plan item for every requested question. Each item
has a stable plan-slot ID, a source chunk ID, a focused concept, a question type,
and a difficulty. If a plan item fails validation, its corrective retry keeps the
same slot ID and fixes that item instead of creating a replacement slot.

Generator follows these plan items rather than choosing questions loosely from the
combined source context. A generated question must return the expected slot ID,
which keeps it connected to the Planner's concept and source chunk. If generation
falls short, the next attempt receives only the missing plan items. Existing
questions and completed slots remain unchanged.

## Validation & Repair

Generated questions are validated before they are accepted as final output.

Validation is processed in configurable batches, and each verdict is matched to the stable question ID supplied to the validator.

If the validator does not return a verdict after the allowed retry, the question is marked as `UNVALIDATED`. An unvalidated question is treated as a validator failure, not a question-quality failure, and is not sent to repair.

Only questions with an identified content defect are eligible for targeted repair.

The validator checks for issues such as:

* unclear or malformed questions
* duplicate MCQ options
* multiple potentially correct answers
* missing or incorrect answers
* invalid question structure
* Fill-in-the-Blank answer and word-bank inconsistencies

Each question receives a stable ID generated by the application.

Example:

```text id="p9u964"
model1_mcq_1
model1_true_false_1
model1_fill_in_the_blank_1
```

If a question fails validation, the system repairs only that question instead of regenerating the whole exam. Initial validation covers every generated question. After a repair pass, Validator receives only the IDs successfully repaired in that pass. The new verdicts are merged into the existing report by `question_id`, preserving PASS verdicts for untouched questions. If a second repair attempt is needed, only the questions repaired during that second attempt are revalidated.

```text id="4igx1e"
Generate
   ↓
Validate
   ↓
PASS ───────→ Final Exam
   ↓
FIX
   ↓
Repair Invalid Question
   ↓
Revalidate Repaired ID Only
```

This keeps already-valid questions unchanged and avoids sending them through Validator again.

## Tech Stack

| Component              | Technology                   |
| ---------------------- | ---------------------------- |
| Language               | Python 3.11                  |
| Backend API            | FastAPI                      |
| Workflow Orchestration | LangGraph                    |
| Embeddings             | FlagEmbedding / Transformers |
| Vector/Document Retrieval | Qdrant                    |
| Application State      | PostgreSQL                   |
| Background Jobs        | Celery + Redis               |
| File Storage           | Replaceable local backend    |
| ML Runtime             | PyTorch                      |
| PDF Parsing            | LlamaParse                   |
| LLM Providers          | LM Studio + DeepSeek V4.1 Flash |
| DeepSeek API Client    | OpenAI Python SDK / Responses API |
| Structured Output      | Pydantic v2 + JSON Schema    |
| Validation             | Pydantic + custom validation |
| NLP                    | spaCy                        |
| Testing                | Pytest                       |
| Frontend               | HTML / JavaScript            |
| Containers             | Docker Compose               |

## Setup

Clone the repository:

```bash id="w6fc3n"
git clone https://github.com/AbdelrahmanAlabadla/DraftWork.git
cd DraftWork
```

Create a virtual environment:

```bash id="54mvaa"
python -m venv .venv
```

Activate it on Windows:

```bash id="ec7xpl"
.venv\Scripts\activate
```

Install dependencies:

```bash id="6h31ob"
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and configure the required model and API settings. Set the PostgreSQL connection with your own local credentials:

```text
DATABASE_URL=postgresql://postgres:your_password@localhost:1966/draftwork
```

Choose the provider with `LLM_PROVIDER`. The existing local LM Studio backend
remains the default and does not require any DeepSeek settings:

```text
LLM_PROVIDER=local
LMS_URL=http://127.0.0.1:1234
LMS_MODEL=mistralai/mistral-7b-instruct-v0.3
LMS_API_KEY=
LMS_REASONING=off
TITLE_LMS_URL=http://127.0.0.1:1234/v1
TITLE_MODEL=mistralai/mistral-7b-instruct-v0.3
```

To use DeepSeek V4.1 Flash instead, store the key only in the gitignored `.env`
file and select the DeepSeek provider:

```text
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_real_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_CONCURRENCY_LIMIT=4
DEEPSEEK_MAX_STRUCTURED_ATTEMPTS=3
DEEPSEEK_MAX_TRANSIENT_RETRIES=2
DEEPSEEK_RETRY_BASE_SECONDS=1.0
```

At startup, `python-dotenv` loads `.env` into the process environment. The
DeepSeek client reads `DEEPSEEK_API_KEY` from that environment and sends it in
the OpenAI Python SDK. The SDK sends it as a Bearer token to DeepSeek's
Responses API. Docker Compose also reads the project `.env` file and passes
these variables to the `app` container; the image never contains the key.

Provider environment variables:

| Variable | Purpose |
| -------- | ------- |
| `LLM_PROVIDER` | Selects `local` or `deepseek`; defaults to `local` |
| `LMS_URL` | LM Studio server URL |
| `LMS_MODEL` | Model served by LM Studio |
| `LMS_API_KEY` | Optional bearer token for compatible local endpoints |
| `LMS_REASONING` | Default local-model reasoning setting |
| `TITLE_LMS_URL` | Optional LM Studio endpoint used for local title generation |
| `TITLE_MODEL` | Optional local model used for title generation |
| `DEEPSEEK_API_KEY` | DeepSeek credential; required only for the DeepSeek provider |
| `DEEPSEEK_BASE_URL` | DeepSeek API base URL |
| `DEEPSEEK_MODEL` | DeepSeek model identifier; currently `deepseek-flash` |
| `DEEPSEEK_CONCURRENCY_LIMIT` | Maximum simultaneous DeepSeek requests |
| `DEEPSEEK_MAX_STRUCTURED_ATTEMPTS` | Maximum attempts for one structured operation |
| `DEEPSEEK_MAX_TRANSIENT_RETRIES` | Retries for rate limits, network failures, timeouts, and server errors |
| `DEEPSEEK_RETRY_BASE_SECONDS` | Initial delay used by exponential backoff |

Create the `draftwork` database if needed, then apply all migrations:

```bash
.venv\Scripts\python.exe scripts\apply_migrations.py
```

Start Qdrant:

```bash id="mjzm19"
docker run -p 6333:6333 qdrant/qdrant
```

Run the application:

```bash id="fb7ho3"
.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

### Docker Compose quick-start

Copy the example environment file and add your own credentials:

```powershell
Copy-Item .env.example .env
```

Set `POSTGRES_PASSWORD` and `LLAMA_PARSE_API` in `.env`. Keep
`LLM_PROVIDER=local` for LM Studio, or set `LLM_PROVIDER=deepseek` and add
`DEEPSEEK_API_KEY` for DeepSeek. Then build and start the API, Celery worker,
Redis, PostgreSQL, and Qdrant:

```powershell
docker compose up -d --build
```

Open DraftWork at `http://localhost:8000`. To follow application logs:

```powershell
docker compose logs -f app
docker compose logs -f worker
```

The worker expects an NVIDIA-compatible Docker GPU runtime for the embedding
model. The API does not request a GPU. Local application files, Redis,
PostgreSQL, Qdrant, and the Hugging Face cache use mounted directories or named
volumes and survive a container rebuild. The migration service applies every
SQL migration before the API and worker start.

## Tests

Run the test suite with:

```bash id="mu4ygr"
pytest -q
```

The tests cover core components including:

* document processing
* semantic chunking
* selected content loading
* structured planning and plan-slot preservation
* schema-based exam generation and targeted shortfall recovery
* validation, targeted repair, and selective revalidation
* DeepSeek client retries, errors, and token accounting
* API behavior
* PostgreSQL evaluation persistence and aggregation
* best-effort telemetry failure handling
* validator batching and verdict coverage
* Eval Dashboard API and frontend behavior
* per-model ZIP exports with separate exam and answer files
* export model selection and safe metadata-based filenames

## Author

**Abdelrahman Alabadla**

AI / Software Engineer
