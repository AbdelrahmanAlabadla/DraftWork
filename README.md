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
  * Why Questions
  * Essay
* Load only selected child chunks for generation
* Automatically validate generated questions
* Repair only invalid questions instead of regenerating the entire exam
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
| Why Questions     | Short reasoning questions            |
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

Generated exam content is handled separately. It remains in the application's temporary in-memory store for up to one hour so PDF, DOCX, and Google Forms exports can use it. Restarting the server clears this temporary exam content, but it does not clear the PostgreSQL evaluation history.

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

Google Forms export continues to create a separate form for each exam model and is not packaged into the downloadable ZIP.

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
    F --> G[Exam Planner]
    G --> H[Question Generation]
    H --> I[Validation]

    I -->|Valid| J[Final Exam]
    I -->|Invalid| K[Targeted Repair]

    K --> I
```

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

If a question fails validation, the system repairs only that question instead of regenerating the whole exam.

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
Revalidate
```

This keeps already-valid questions unchanged and reduces unnecessary LLM regeneration.

## Tech Stack

| Component              | Technology                   |
| ---------------------- | ---------------------------- |
| Language               | Python                       |
| Backend API            | FastAPI                      |
| Workflow Orchestration | LangGraph                    |
| Embeddings             | FlagEmbedding / Transformers |
| Vector/Document Retrieval | Qdrant                    |
| Evaluation History     | PostgreSQL                   |
| ML Runtime             | PyTorch                      |
| PDF Parsing            | LlamaParse                   |
| Validation             | Pydantic + custom validation |
| NLP                    | spaCy                        |
| Testing                | Pytest                       |
| Frontend               | HTML / JavaScript            |

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

Create the `draftwork` database if needed, then apply the evaluation-history migration:

```bash
psql "postgresql://postgres:your_password@localhost:1966/draftwork" -f migrations/001_create_evaluation_runs.sql
```

Start Qdrant:

```bash id="mjzm19"
docker run -p 6333:6333 qdrant/qdrant
```

Run the application:

```bash id="fb7ho3"
.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

## Tests

Run the test suite with:

```bash id="mu4ygr"
pytest -q
```

The tests cover core components including:

* document processing
* semantic chunking
* selected content loading
* exam generation
* validation and repair
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
