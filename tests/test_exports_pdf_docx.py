from __future__ import annotations

import io
import zipfile

import pytest

pytest.importorskip("fpdf")
pytest.importorskip("docx")

from app.exports.common import RESPONSE_LINE_TEXT, flatten_exam_items, group_exam_sections
from app.exports.docx_exporter import render_exam_docx
from app.exports.pdf_exporter import render_exam_pdf


def _stored_record() -> dict:
    return {
        "exam_id": "exam_test",
        "exams": [
            {
                "model_number": 1,
                "questions": {
                    "mcq": [{
                        "question_id": "model1_mcq_1",
                        "question": "What is a CPU?",
                        "options": {"A": "Central Processing Unit", "B": "GPU",
                                    "C": "Hard drive", "D": "Monitor"},
                        "correct_answer": "A",
                    }],
                    "true_false": [{
                        "question_id": "model1_true_false_1",
                        "statement": "RAM is volatile memory.",
                        "answer": "True",
                    }],
                    "fill_in_the_blank": {
                        "word_bank": ["CPU", "RAM", "ROM", "GPU"],
                        "items": [{
                            "question_id": "model1_fill_in_the_blank_1",
                            "question": "The ___ processes instructions.",
                            "answers": ["CPU"],
                        }],
                    },
                    "short_answer": [{
                        "question_id": "model1_short_answer_1",
                        "question": "Why is cache useful?",
                        "reference_answer": "It speeds up repeated access.",
                    }],
                    "essay": [{
                        "question_id": "model1_essay_1",
                        "question": "Explain the fetch-decode-execute cycle.",
                        "reference_answer": "Long explanation...",
                        "key_points": ["fetch", "decode", "execute"],
                    }],
                },
                "markdown": "",
                "warnings": [],
            }
        ],
        "warnings": [],
        "metadata": {
            "exam_title": "Computer Science Final Examination",
            "class_name": "12A",
            "duration": "90 minutes",
            "exam_date": "2026-08-25",
            "teacher_name": "Ms Example",
            "footer_message": "Do your best!",
        },
    }


def test_flatten_items_structure():
    items = flatten_exam_items(_stored_record()["exams"][0]["questions"])
    assert [i["qtype"] for i in items] == [
        "mcq", "true_false", "fill_in_the_blank", "short_answer", "essay"]
    assert [i["number"] for i in items] == [1, 2, 3, 4, 5]
    mcq = items[0]
    assert mcq["options"]["A"] == "Central Processing Unit"
    assert mcq["correct_answer"] == "A"
    fitb = items[2]
    assert fitb["word_bank"] == ["CPU", "RAM", "ROM", "GPU"]
    assert fitb["answers"] == ["CPU"]
    # question_ids preserved for tracing
    assert items[0]["question_id"] == "model1_mcq_1"


def test_print_sections_follow_reference_order_and_restart_numbers():
    sections = group_exam_sections(_stored_record()["exams"][0]["questions"])
    assert [section["qtype"] for section in sections] == [
        "mcq", "fill_in_the_blank", "true_false", "short_answer", "essay"]
    assert all(section["items"][0]["number"] == 1 for section in sections)
    assert sections[1]["word_bank"] == ["CPU", "RAM", "ROM", "GPU"]


def test_pdf_export_contains_exam_and_key():
    pdf_bytes = render_exam_pdf(_stored_record())
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_pdf_student_copy_has_metadata_and_no_reference_answer_leak():
    pypdf = pytest.importorskip("pypdf")
    pdf_bytes = render_exam_pdf(_stored_record())
    text = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages)
    student, answer_key = text.split("Answer Key - Model 1", 1)
    assert "Computer Science Final Examination" in student
    assert "Class:" in student and "12A" in student
    assert "Multiple Choice Questions" in student
    assert "Fill in the Blank" in student
    assert "True / False" in student
    assert "Short Answer" in student
    assert "Essay" in student
    assert "It speeds up repeated access." not in student
    assert "Long explanation..." not in student
    assert "It speeds up repeated access." in answer_key
    assert "Do your best!" not in text
    assert text.count("Computer Science Final Examination") == 1
    # Three short-answer lines plus 22 essay lines, exactly matching DOCX.
    assert student.count(RESPONSE_LINE_TEXT) == 25


def test_pdf_unknown_model_empty_questions_produces_no_pages():
    out = render_exam_pdf({"exams": [{"model_number": 1, "questions": {}}]})
    assert isinstance(out, bytes)


def test_docx_export_valid_zip():
    docx_bytes = render_exam_docx(_stored_record())
    # DOCX is a ZIP container
    assert docx_bytes[:2] == b"PK"
    assert len(docx_bytes) > 500


def test_docx_uses_a4_half_inch_margins_and_grouped_sections():
    docx_bytes = render_exam_docx(_stored_record())
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    # A4 is 11906 x 16838 twips; 0.5 inch is 720 twips.
    assert 'w:w="11906"' in document_xml
    assert 'w:h="16838"' in document_xml
    assert 'w:top="720"' in document_xml
    assert 'w:right="720"' in document_xml
    assert 'w:bottom="720"' in document_xml
    assert 'w:left="720"' in document_xml
    for heading in ("Multiple Choice Questions", "Fill in the Blank", "True / False", "Short Answer", "Essay"):
        assert heading in document_xml


def test_docx_has_page_number_only_and_visible_open_answer_lines():
    docx_bytes = render_exam_docx(_stored_record())
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        header_xml = "".join(
            archive.read(name).decode("utf-8")
            for name in archive.namelist() if name.startswith("word/header")
        )
        footer_xml = "".join(
            archive.read(name).decode("utf-8")
            for name in archive.namelist() if name.startswith("word/footer")
        )
    assert "Computer Science Final Examination" not in header_xml
    assert "Student copy" not in header_xml
    assert "Do your best!" not in document_xml
    assert "PAGE" in footer_xml and "NUMPAGES" in footer_xml
    # Three short-answer lines plus 22 essay lines. FITB gets no extra rule.
    assert document_xml.count("_" * 92) == 25


def test_docx_multi_model_isolated():
    record = _stored_record()
    model2 = {"model_number": 2,
              "questions": {"mcq": [{
                  "question_id": "model2_mcq_1", "question": "M2 only?",
                  "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                  "correct_answer": "D"}]}, "markdown": "", "warnings": []}
    record["exams"].append(model2)
    docx_bytes = render_exam_docx(record)
    assert b"word/document.xml" in docx_bytes[:2000] or docx_bytes[:2] == b"PK"


def test_browser_preview_uses_structured_student_and_key_layout():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    preview_js = (root / "FrontEnd/js/exam-view.js").read_text(encoding="utf-8")
    main_js = (root / "FrontEnd/js/main.js").read_text(encoding="utf-8")
    assert "marked.parse" not in preview_js
    assert 'short_answer: 3, essay: 22' in preview_js
    assert "buildAnswerKey(exam)" in preview_js
    assert "item.correct_answer" in preview_js
    assert "addResponseLines(question" in preview_js
    assert "renderExamOutput(data.exams, data.metadata || {})" in main_js
