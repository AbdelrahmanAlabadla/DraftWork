from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import exam_store
from app.api.main import app
from app.exports.common import document_export_filenames


client = TestClient(app)


def _model(number: int) -> dict:
    return {
        "model_number": number,
        "questions": {
            "mcq": [{
                "question_id": f"model{number}_mcq_1",
                "question": f"Model {number} question?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "correct_answer": "A",
            }]
        },
        "markdown": "",
        "warnings": [],
    }


@pytest.fixture(autouse=True)
def _clear_exam_store():
    exam_store.clear()
    yield
    exam_store.clear()


@pytest.mark.parametrize(
    ("metadata", "exam_name", "answer_name"),
    [
        (
            {"exam_title": "Biology Midterm", "class_name": "Grade 10"},
            "Biology_Midterm_Grade_10_Model_1.pdf",
            "Answers_Biology_Midterm_Grade_10_Model_1.pdf",
        ),
        (
            {"exam_title": "Biology Midterm"},
            "Biology_Midterm_Model_1.pdf",
            "Answers_Biology_Midterm_Model_1.pdf",
        ),
        (
            {"class_name": "Grade 10"},
            "Grade_10_Model_1.pdf",
            "Answers_Grade_10_Model_1.pdf",
        ),
        ({}, "Exam_Model_1.pdf", "Answers_Exam_Model_1.pdf"),
    ],
)
def test_document_export_filename_fallbacks(metadata, exam_name, answer_name):
    assert document_export_filenames(metadata, 1, "pdf") == (
        exam_name,
        answer_name,
    )


def test_document_export_filename_sanitizes_metadata_without_ids():
    names = document_export_filenames(
        {"exam_title": " Biology: Mid/term? ", "class_name": "Grade 10*"},
        2,
        ".docx",
    )
    assert names == (
        "Biology_Midterm_Grade_10_Model_2.docx",
        "Answers_Biology_Midterm_Grade_10_Model_2.docx",
    )
    assert all("exam_" not in name.lower() for name in names)
    assert all(not any(char in name for char in '<>:"/\\|?*') for name in names)


def test_single_model_pdf_export_is_zip_with_exam_and_answers():
    exam_id = exam_store.save_exam([_model(1)], metadata={})
    response = client.post(f"/exams/{exam_id}/export/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "SmartExam_Export.zip" in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == ["Exam_Model_1.pdf", "Answers_Exam_Model_1.pdf"]
        assert all(archive.read(name).startswith(b"%PDF") for name in archive.namelist())


def test_multi_model_docx_export_contains_only_selected_models():
    exam_id = exam_store.save_exam(
        [_model(1), _model(2), _model(3)],
        metadata={"exam_title": "Biology Midterm", "class_name": "Grade 10"},
    )
    response = client.post(
        f"/exams/{exam_id}/export/docx", json={"model_numbers": [1, 2]}
    )

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == [
            "Biology_Midterm_Grade_10_Model_1.docx",
            "Answers_Biology_Midterm_Grade_10_Model_1.docx",
            "Biology_Midterm_Grade_10_Model_2.docx",
            "Answers_Biology_Midterm_Grade_10_Model_2.docx",
        ]
        assert all(archive.read(name).startswith(b"PK") for name in archive.namelist())


def test_selecting_only_model_two_still_returns_exam_and_answer_zip():
    exam_id = exam_store.save_exam([_model(1), _model(2), _model(3)], metadata={})
    response = client.post(
        f"/exams/{exam_id}/export/pdf", json={"model_numbers": [2]}
    )

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == [
            "Exam_Model_2.pdf",
            "Answers_Exam_Model_2.pdf",
        ]


def test_document_export_rejects_empty_duplicate_and_unknown_selections():
    exam_id = exam_store.save_exam([_model(1), _model(2)])
    assert client.post(
        f"/exams/{exam_id}/export/pdf", json={"model_numbers": []}
    ).status_code == 400
    assert client.post(
        f"/exams/{exam_id}/export/pdf", json={"model_numbers": [1, 1]}
    ).status_code == 400
    assert client.post(
        f"/exams/{exam_id}/export/pdf", json={"model_numbers": [3]}
    ).status_code == 400


def test_frontend_uses_model_dialog_only_for_multiple_document_models():
    root = Path(__file__).resolve().parents[1]
    html = (root / "FrontEnd/index.html").read_text(encoding="utf-8")
    script = (root / "FrontEnd/js/export.js").read_text(encoding="utf-8")
    assert 'id="exportModelDialog"' in html
    assert 'id="exportModelList"' in html
    assert "models.length === 1" in script
    assert "showModal()" in script
    assert "model_numbers: modelNumbers" in script
    assert 'kind === "pdf" || kind === "docx"' in script
    assert "google-forms" in script
