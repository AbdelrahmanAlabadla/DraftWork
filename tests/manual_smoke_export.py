"""End-to-end smoke: ExamStore -> all three exporters (Forms mocked)."""
import io

from app.api import exam_store
from app.exports.common import flatten_exam_items
from app.exports.pdf_exporter import render_answers_pdf, render_exam_pdf
from app.exports.docx_exporter import render_answers_docx, render_exam_docx
from app.integrations.google_forms import client
from app.integrations.google_forms.exporter import export_exam

exam = {
    "model_number": 1,
    "questions": {
        "mcq": [{"question_id": "model1_mcq_1", "question": "What is 2+2?",
                 "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
                 "correct_answer": "B"}],
        "true_false": [{"question_id": "model1_true_false_1",
                        "statement": "The sky is green.", "answer": "False"}],
        "fill_in_the_blank": {"word_bank": ["CPU", "RAM", "ROM", "GPU"],
                              "items": [{"question_id": "model1_fill_in_the_blank_1",
                                         "question": "The ___ computes.",
                                         "answers": ["CPU"]}]},
        "short_answer": [{"question_id": "model1_short_answer_1",
                          "question": "Why do we cache?",
                          "reference_answer": "TEACHER ONLY"}],
        "essay": [{"question_id": "model1_essay_1", "question": "Discuss OS types.",
                   "reference_answer": "TEACHER ONLY ESSAY",
                   "key_points": ["batch", "time-sharing"]}],
    },
    "markdown": "", "warnings": [],
}
record = {"exams": [exam], "warnings": []}

# 1. Store roundtrip
eid = exam_store.save_exam(exam["questions"] and [exam], warnings=[], document_id="doc-smoke")
loaded = exam_store.get_exam(eid)
assert loaded["exams"][0]["model_number"] == 1
print("store OK ->", eid)

# 2. PDF
stored_model = loaded["exams"][0]
stored_metadata = loaded.get("metadata") or {}
pdf = render_exam_pdf(stored_model, stored_metadata)
assert pdf.startswith(b"%PDF") and len(pdf) > 1000
open("data/smoke_exam.pdf", "wb").write(pdf)
pdf_answers = render_answers_pdf(stored_model, stored_metadata)
assert pdf_answers.startswith(b"%PDF")
open("data/smoke_answers.pdf", "wb").write(pdf_answers)
print(f"PDF OK ({len(pdf)} + {len(pdf_answers)} bytes)")

# 3. DOCX
docx = render_exam_docx(stored_model, stored_metadata)
assert docx[:2] == b"PK"
open("data/smoke_exam.docx", "wb").write(docx)
docx_answers = render_answers_docx(stored_model, stored_metadata)
assert docx_answers[:2] == b"PK"
open("data/smoke_answers.docx", "wb").write(docx_answers)
print(f"DOCX OK ({len(docx)} + {len(docx_answers)} bytes)")

# 4. Google Forms (mocked API)
created, updated = [], {}

def fake_create(title):
    created.append(title)
    return {"form_id": "smoke_form_1", "title": title,
            "edit_url": "https://edit", "view_url": "https://view"}

def fake_batch(form_id, reqs):
    updated[form_id] = list(reqs)
    return len(reqs)

client.create_form, client.batch_update = fake_create, fake_batch
result = export_exam(loaded)
assert result["errors"] == [] and len(result["exports"]) == 1
exp = result["exports"][0]
titles = [r["createItem"]["item"].get("title", "") for r in updated["smoke_form_1"]]
payload = repr(updated["smoke_form_1"])
assert "TEACHER ONLY" not in payload, "reference answer leaked!"
assert any("Word Bank" in t for t in titles)
assert any(t == "CPU" for r in updated["smoke_form_1"]
           for q in [r["createItem"]["item"].get("questionItem", {}).get("question", {})]
           for a in q.get("grading", {}).get("correctAnswers", {}).get("answers", [])
           for t in [a["value"]])
print("Google Forms OK ->", exp["form_id"], "| questions:", exp["questions_exported"])

exam_store.clear()
print("ALL SMOKE CHECKS PASSED")
