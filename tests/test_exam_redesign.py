from __future__ import annotations

import io
import zipfile

import pytest

from app.exports.common import TRUE_FALSE_CHOICES_BY_LANG
from app.exports.content_parser import (
    normalize_numerals,
    parse_coordinate_table,
    parse_latex_array,
    parse_rich_text,
)
from app.exports.document_ir import build_document_ir
from app.exports.docx_exporter import render_answers_docx, render_exam_docx
from app.exports.pdf_exporter import render_exam_pdf


def _arabic_exam() -> dict:
    return {
        "model_number": 1,
        "document_language": "ar",
        "questions": {
            "mcq": [{
                "question_id": "m1",
                "question": "يبين الجدول عدد الساعات. أي عبارة صحيحة؟\n\n"
                            "| اليوم | الرياضيات | الفيزياء |\n"
                            "|---|---|---|\n"
                            "| الأول | 2 | 3 |\n"
                            "| الثاني | 3 | 4 |",
                "options": {"A": "\\(S=2d+3\\)", "B": "\\(S=5d\\)",
                            "C": "\\(S=3d+2\\)", "D": "\\(S=6d\\)"},
                "correct_answer": "B",
            }],
            "true_false": [{
                "question_id": "m2",
                "statement": "الشمس تشرق من الشرق.",
                "answer": True,
            }],
            "equation": [{
                "question_id": "m3",
                "equation": "لتكن x إجمالي المبيعات.\n"
                            "\\[A=50\\%\\text{ من }x\\]\n"
                            "الحل:\nSOLUTION_SECRET",
                "final_answer": "x=5",
            }],
        },
    }


def _english_exam() -> dict:
    return {
        "model_number": 1,
        "document_language": "en",
        "questions": {
            "mcq": [{
                "question_id": "e1",
                "question": "The data is shown below.\n\n"
                            "| x | y |\n|---|---|\n| 1 | 2 |\n| 2 | 4 |",
                "options": {"A": "2", "B": "4", "C": "6", "D": "8"},
                "correct_answer": "B",
            }],
            "true_false": [{
                "question_id": "e2",
                "statement": "The Earth revolves around the Sun.",
                "answer": True,
            }],
        },
    }


def test_ir_is_versioned_and_preserves_question_type_order():
    document = build_document_ir(_arabic_exam(), {"exam_title": "امتحان"})
    assert document["version"] == "1.0"
    assert document["direction"] == "rtl"
    assert [section["qtype"] for section in document["sections"]] == [
        "mcq", "true_false", "equation",
    ]
    assert document["sections"][0]["items"][0]["number"] == 1


def test_markdown_table_becomes_structured_table_without_raw_pipes():
    blocks, diagnostics = parse_rich_text(
        "Look at this.\n\n| x | y |\n|---|---|\n| 1 | 2 |\n| 2 | 4 |",
        language="en",
    )
    assert diagnostics == []
    table = next(block for block in blocks if block["type"] == "table")
    assert len(table["headers"]) == 2
    assert len(table["rows"]) == 2
    assert table["headers"][0][0]["text"] == "x"
    assert table["rows"][1][1][0]["text"] == "4"


def test_latex_array_becomes_native_table_with_arabic_cells():
    table = parse_latex_array(
        r"\begin{array}{c|cc}"
        r"&\text{الراتب الثابت}&\text{العمولة}\\"
        r"\text{العرض الجديد}&800&2\%\\"
        r"\text{الوظيفة الحالية}&1200&1.5\%"
        r"\end{array}",
        language="ar",
    )
    assert table is not None
    assert table["type"] == "table"
    assert table["headers"][1][0]["text"] == "الراتب الثابت"
    assert table["rows"][0][1][0]["text"] == "800"
    assert table["rows"][0][2][0]["text"] == "2%"


def test_coordinate_list_becomes_compact_table():
    table = parse_coordinate_table(
        r"النقاط: \((-4,0)، (0,3)، (4,0)\)", language="ar",
    )
    assert table is not None
    assert table["compact"] is True
    assert len(table["rows"]) == 3
    assert table["rows"][0][0][0]["text"] == "-4"


def test_arabic_indic_and_persian_numerals_are_normalized():
    assert normalize_numerals("١٢٣٤٥٦٧٨٩٠") == "1234567890"
    assert normalize_numerals("۱۲۳۴۵۶۷۸۹۰") == "1234567890"


def test_true_false_choice_labels_are_exact_in_arabic_and_english():
    assert TRUE_FALSE_CHOICES_BY_LANG["ar"] == "صح (   ) غلط (   )"
    assert TRUE_FALSE_CHOICES_BY_LANG["en"] == "True (   ) False (   )"
    arabic = build_document_ir(_arabic_exam(), {"exam_title": "امتحان"})
    english = build_document_ir(_english_exam(), {"exam_title": "Exam"})
    assert arabic["metadata"]["true_false_choices"] == "صح (   ) غلط (   )"
    assert english["metadata"]["true_false_choices"] == "True (   ) False (   )"


def test_legacy_solution_is_moved_out_of_student_copy():
    student = build_document_ir(_arabic_exam(), {"exam_title": "امتحان"}, kind="student")
    answers = build_document_ir(_arabic_exam(), {"exam_title": "امتحان"}, kind="answers")
    student_text = str(student["sections"][2]["items"][0]["blocks"])
    answer_text = str(answers["sections"][2]["items"][0]["answer_blocks"])
    assert "SOLUTION_SECRET" not in student_text
    assert "SOLUTION_SECRET" in answer_text


def test_pdf_has_no_raw_markdown_or_latex_and_contains_table_cells():
    pypdf = pytest.importorskip("pypdf")
    pdf = render_exam_pdf(_arabic_exam(), {"exam_title": "امتحان"})
    text = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(io.BytesIO(pdf)).pages)
    assert "اليوم" in text or "اليوم"[::-1] in text
    assert "الرياضيات" in text or "الرياضيات"[::-1] in text
    assert "|---" not in text
    assert "\\begin" not in text
    assert "\\frac" not in text
    assert "SOLUTION_SECRET" not in text


def test_docx_uses_native_table_and_no_raw_table_markup():
    document = render_exam_docx(_english_exam(), {"exam_title": "Exam"})
    with zipfile.ZipFile(io.BytesIO(document)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "<w:tbl>" in xml
    assert "| x |" not in xml
    assert "|---" not in xml
    assert "The Earth revolves around the Sun." in xml
    assert "True (   ) False (   )" in xml.replace("\u00a0", " ")


def test_docx_arabic_marker_is_A_not_reversed_dot_A():
    document = render_exam_docx(_arabic_exam(), {"exam_title": "امتحان"})
    with zipfile.ZipFile(io.BytesIO(document)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "<w:t>A.</w:t>" in xml
    assert "<w:t>.A</w:t>" not in xml
    assert "صح (   ) غلط (   )" in xml.replace("\u00a0", " ")


def test_answer_key_contains_secret_and_student_does_not():
    student = render_exam_docx(_arabic_exam(), {"exam_title": "امتحان"})
    answers = render_answers_docx(_arabic_exam(), {"exam_title": "امتحان"})
    with zipfile.ZipFile(io.BytesIO(student)) as archive:
        student_xml = archive.read("word/document.xml").decode("utf-8")
    with zipfile.ZipFile(io.BytesIO(answers)) as archive:
        answer_xml = archive.read("word/document.xml").decode("utf-8")
    assert "SOLUTION_SECRET" not in student_xml
    assert "SOLUTION_SECRET" in answer_xml


def test_pdf_true_false_uses_exact_labels():
    pymupdf = pytest.importorskip("pymupdf")
    pdf = render_exam_pdf(_arabic_exam(), {"exam_title": "امتحان"})
    page = pymupdf.open(stream=pdf, filetype="pdf")[0]
    text = page.get_text()
    words = page.get_text("words")
    true_word = next(word for word in words if word[4] == "صح")
    false_word = next(word for word in words if word[4] == "غلط")
    # The nonbreaking options group is one visual line and remains inside the
    # paragraph margins. RTL extraction order is intentionally not asserted.
    assert abs(true_word[1] - false_word[1]) < 2.0
    assert 25 < min(true_word[0], false_word[0])
    assert max(true_word[2], false_word[2]) < page.rect.width - 25
    assert "خطأ" not in text


def test_docx_section_headings_have_no_background_shading():
    document = render_exam_docx(_arabic_exam(), {"exam_title": "امتحان"})
    with zipfile.ZipFile(io.BytesIO(document)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    # A section heading must not carry a gray/colored fill.
    assert 'w:fill="E8EEF5"' not in xml
    assert 'w:fill="F6F8FB"' not in xml

