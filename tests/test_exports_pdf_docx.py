from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("fpdf")
pytest.importorskip("docx")

from app.exports.common import (
    RESPONSE_LINE_TEXT,
    expand_fill_blank_text,
    flatten_exam_items,
    group_exam_sections,
)
from app.exports.docx_exporter import render_answers_docx, render_exam_docx
from app.exports.pdf_exporter import render_answers_pdf, render_exam_pdf


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
                    "definition": [{
                        "question_id": "model1_definition_1",
                        "term": "Operating system",
                        "reference_answer": "Software that manages computer resources.",
                    }],
                    "equation": [{
                        "question_id": "model1_equation_1",
                        "equation": "2x + 5 = 15",
                        "solution_steps": ["2x = 10", "x = 5"],
                        "final_answer": "x = 5",
                    }],
                    "word_problem": [{
                        "question_id": "model1_word_problem_1",
                        "question": "A car travels 120 km in 2 hours. Calculate its average speed.",
                        "solution_steps": ["speed = distance ÷ time", "120 ÷ 2 = 60"],
                        "final_answer": "60 km/h",
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
        "mcq", "fill_in_the_blank", "true_false", "definition",
        "short_answer", "equation", "word_problem", "essay"]
    assert [i["number"] for i in items] == list(range(1, 9))
    mcq = items[0]
    assert mcq["options"]["A"] == "Central Processing Unit"
    assert mcq["correct_answer"] == "A"
    fitb = items[1]
    assert fitb["word_bank"] == ["CPU", "RAM", "ROM", "GPU"]
    assert fitb["answers"] == ["CPU"]
    # question_ids preserved for tracing
    assert items[0]["question_id"] == "model1_mcq_1"


def test_print_sections_follow_reference_order_and_restart_numbers():
    sections = group_exam_sections(_stored_record()["exams"][0]["questions"])
    assert [section["qtype"] for section in sections] == [
        "mcq", "fill_in_the_blank", "true_false", "definition",
        "short_answer", "equation", "word_problem", "essay"]
    assert all(section["items"][0]["number"] == 1 for section in sections)
    assert sections[1]["word_bank"] == ["CPU", "RAM", "ROM", "GPU"]


def _model_and_metadata() -> tuple[dict, dict]:
    record = _stored_record()
    return record["exams"][0], record["metadata"]


def _arabic_model_and_metadata() -> tuple[dict, dict]:
    return (
        {
            "model_number": 1,
            "document_language": "ar",
            "questions": {
                "mcq": [{
                    "question_id": "ar_mcq_1",
                    "question": "ما دور المستقبلات الحسية في اكتشاف المؤثرات؟",
                    "options": {
                        "A": "استقبال المؤثرات",
                        "B": "هضم الغذاء",
                        "C": "إنتاج الطاقة فقط",
                        "D": "تخزين الماء",
                    },
                    "correct_answer": "A",
                }],
                "short_answer": [{
                    "question_id": "ar_short_1",
                    "question": "فسر انتقال السيال العصبي إلى الجهاز العصبي المركزي.",
                    "reference_answer": "ينتقل السيال عبر الخلايا العصبية الحسية.",
                }],
            },
        },
        {"exam_title": "اختبار الجهاز العصبي", "class_name": "الصف الثاني عشر"},
    )


def test_pdf_export_produces_separate_exam_and_answer_documents():
    exam, metadata = _model_and_metadata()
    exam_bytes = render_exam_pdf(exam, metadata)
    answer_bytes = render_answers_pdf(exam, metadata)
    assert exam_bytes.startswith(b"%PDF")
    assert answer_bytes.startswith(b"%PDF")
    assert len(exam_bytes) > 1000
    assert len(answer_bytes) > 500


def test_pdf_student_copy_has_metadata_and_no_reference_answer_leak():
    pypdf = pytest.importorskip("pypdf")
    exam, metadata = _model_and_metadata()
    pdf_bytes = render_exam_pdf(exam, metadata)
    student = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages)
    assert "Computer Science Final Examination" in student
    assert "Class:" in student and "12A" in student
    assert "Multiple Choice Questions" in student
    assert "Fill in the Blank" in student
    assert "True / False" in student
    assert "Define" in student
    assert "Why Questions" in student
    assert "Solve the equations and show your steps." in student
    assert "Solve the word problems and show your steps." in student
    assert "Essay" in student
    assert "Answer Key" not in student
    assert "It speeds up repeated access." not in student
    assert "Long explanation..." not in student
    assert "Software that manages computer resources." not in student
    assert "speed = distance" not in student
    assert "Do your best!" not in student
    assert student.count("Computer Science Final Examination") == 1
    # Open questions remain present; writing lines are drawn as PDF rules so
    # they are direction-neutral and do not appear as extractable underscores.
    assert "Why is cache useful?" in student


def test_pdf_answer_file_contains_answers_without_student_exam():
    pypdf = pytest.importorskip("pypdf")
    exam, metadata = _model_and_metadata()
    pdf_bytes = render_answers_pdf(exam, metadata)
    text = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages)
    assert "Answer Key - Model 1" in text
    assert "It speeds up repeated access." in text
    assert "Long explanation..." in text
    assert "Software that manages computer resources." in text
    assert "120" in text and "60 km/h" in text
    assert RESPONSE_LINE_TEXT not in text


def test_pdf_keeps_each_essay_writing_area_with_its_question():
    pypdf = pytest.importorskip("pypdf")
    exam = {
        "model_number": 1,
        "questions": {
            "essay": [
                {"question_id": "model1_essay_1", "question": "Explain the first topic.", "reference_answer": "A"},
                {"question_id": "model1_essay_2", "question": "Explain the second topic.", "reference_answer": "B"},
            ]
        },
    }
    pages = pypdf.PdfReader(io.BytesIO(render_exam_pdf(exam, {"exam_title": "Essay test"}))).pages
    page_texts = [page.extract_text() or "" for page in pages]
    second_page = next(text for text in page_texts if "Explain the second topic." in text)
    assert len(pages) == 2
    assert "Explain the second topic." in second_page


def test_pdf_unknown_model_empty_questions_produces_no_pages():
    out = render_exam_pdf({"model_number": 1, "questions": {}}, {})
    assert isinstance(out, bytes)


def test_docx_export_valid_zip():
    exam, metadata = _model_and_metadata()
    docx_bytes = render_exam_docx(exam, metadata)
    # DOCX is a ZIP container
    assert docx_bytes[:2] == b"PK"
    assert len(docx_bytes) > 500


def test_docx_uses_a4_half_inch_margins_and_grouped_sections():
    exam, metadata = _model_and_metadata()
    docx_bytes = render_exam_docx(exam, metadata)
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    # A4 is 11906 x 16838 twips; 0.5 inch is 720 twips.
    assert 'w:w="11906"' in document_xml
    assert 'w:h="16838"' in document_xml
    assert 'w:top="720"' in document_xml
    assert 'w:right="720"' in document_xml
    assert 'w:bottom="720"' in document_xml
    assert 'w:left="720"' in document_xml
    for heading in (
        "Multiple Choice Questions", "Fill in the Blank", "True / False", "Define",
        "Why Questions", "Solve the equations and show your steps.",
        "Solve the word problems and show your steps.", "Essay",
    ):
        assert heading in document_xml


def test_docx_has_page_number_only_and_visible_open_answer_lines():
    exam, metadata = _model_and_metadata()
    docx_bytes = render_exam_docx(exam, metadata)
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
    assert "Answer Key" not in document_xml
    assert "It speeds up repeated access." not in document_xml
    assert "Do your best!" not in document_xml
    assert "PAGE" in footer_xml and "NUMPAGES" in footer_xml
    # 3 Why + 3 Equation + 5 Word Problem + 22 Essay lines.
    assert document_xml.count("_" * 92) == 33


def test_docx_answer_file_contains_answers_without_student_sections():
    exam, metadata = _model_and_metadata()
    docx_bytes = render_answers_docx(exam, metadata)
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "Answer Key - Model 1" in document_xml
    assert "It speeds up repeated access." in document_xml
    assert "Long explanation..." in document_xml
    assert "Software that manages computer resources." in document_xml
    assert "60 km/h" in document_xml
    assert RESPONSE_LINE_TEXT not in document_xml


def test_arabic_pdf_uses_exam_language_when_metadata_has_no_language():
    pypdf = pytest.importorskip("pypdf")
    exam, metadata = _arabic_model_and_metadata()
    pdf_bytes = render_exam_pdf(exam, metadata)
    text = "\n".join(
        page.extract_text() or ""
        for page in pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages
    )
    # Depending on the installed PDF text extractor, shaped RTL runs may be
    # returned in either logical or visual character order.
    title = "اختبار الجهاز العصبي"
    section = "أسئلة الاختيار من متعدد"
    assert title in text or title[::-1] in text
    assert section in text or section[::-1] in text
    assert "صح" not in text  # no English/Arabic true-false scaffold was added


def test_arabic_docx_marks_paragraphs_and_runs_rtl():
    exam, metadata = _arabic_model_and_metadata()
    docx_bytes = render_exam_docx(exam, metadata)
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "اختبار الجهاز العصبي" in document_xml
    assert "أسئلة الاختيار من متعدد" in document_xml
    assert "w:bidi" in document_xml
    assert "w:rtl" in document_xml
    assert 'w:val="right"' in document_xml


def test_export_fill_blank_adds_four_writing_characters():
    assert expand_fill_blank_text("Complete ____ now") == "Complete ________ now"
    assert expand_fill_blank_text("أكمل ----- هنا") == "أكمل --------- هنا"


def test_arabic_docx_separates_mcq_markers_and_uses_tajawal():
    exam, metadata = _arabic_model_and_metadata()
    exam["questions"]["mcq"][0]["options"]["B"] = "n = c / v"
    docx_bytes = render_exam_docx(exam, metadata)
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert '<w:t>A.</w:t>' in document_xml
    assert "استقبال المؤثرات .A" not in document_xml
    formula = document_xml.index("n = c / v")
    marker = document_xml.index("<w:t>B.</w:t>")
    assert formula < marker
    marker_properties = document_xml[marker - 400:marker]
    assert '<w:bidi w:val="0"/>' in marker_properties
    assert '<w:rtl w:val="0"/>' in marker_properties
    assert 'w:jc w:val="right"' in document_xml[formula - 500:formula]
    assert 'w:ascii="Tajawal"' in document_xml
    assert "F6F8FB" not in document_xml
    response_line = document_xml.index(RESPONSE_LINE_TEXT)
    assert '<w:bidi/>' in document_xml[response_line - 500:response_line]
    assert 'w:right="230"' in document_xml[response_line - 500:response_line]


def test_docx_definition_uses_a_long_flexible_answer_line():
    exam, metadata = _model_and_metadata()
    docx_bytes = render_exam_docx(exam, metadata)
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    definition_table = next(
        table for table in root.findall(".//w:tbl", ns)
        if "Operating system" in "".join(table.itertext())
    )
    cells = definition_table.findall("./w:tr/w:tc", ns)
    assert len(cells) == 2
    widths = [int(cell.find("./w:tcPr/w:tcW", ns).get(f"{{{ns['w']}}}w")) for cell in cells]
    assert widths[1] > widths[0] * 2
    assert cells[1].find(".//w:pBdr/w:bottom", ns) is not None


def test_arabic_docx_places_true_false_choices_after_statement():
    exam, metadata = _arabic_model_and_metadata()
    statement = "يعلم الله تعالى كل شيء."
    exam["questions"]["true_false"] = [{
        "question_id": "ar_tf_1",
        "statement": statement,
        "answer": "True",
    }]
    docx_bytes = render_exam_docx(exam, metadata)
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    tf_table = next(
        table for table in root.findall(".//w:tbl", ns)
        if statement in "".join(table.itertext())
    )
    cells = tf_table.findall("./w:tr/w:tc", ns)
    assert "صح" in "".join(cells[0].itertext())
    assert statement in "".join(cells[1].itertext())


def test_english_docx_remains_ltr():
    exam, metadata = _model_and_metadata()
    document = render_exam_docx(exam, metadata)
    with zipfile.ZipFile(io.BytesIO(document)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "w:bidi" not in document_xml
    assert "w:rtl" not in document_xml


def test_equations_are_rendered_as_math_images_in_pdf_and_docx():
    exam = {
        "model_number": 1,
        "questions": {
            "equation": [{
                "question_id": "model1_equation_math",
                "equation": "sin θc = n2/n1؛ n1 = 1.50؛ n2 = 1.00",
                "solution_steps": ["sin θc = 1.00 / 1.50"],
                "final_answer": "θc = 41.8°",
            }],
        },
    }
    source = exam["questions"]["equation"][0]["equation"]
    pdf_bytes = render_exam_pdf(exam, {"exam_title": "Math test"})
    pypdf = pytest.importorskip("pypdf")
    pdf_text = "\n".join(
        page.extract_text() or ""
        for page in pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages
    )
    assert source not in pdf_text

    docx_bytes = render_exam_docx(exam, {"exam_title": "Math test"})
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
    assert source not in document_xml
    assert media


def test_browser_preview_uses_structured_student_and_key_layout():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    preview_js = (root / "FrontEnd/js/exam-view.js").read_text(encoding="utf-8")
    main_js = (root / "FrontEnd/js/main.js").read_text(encoding="utf-8")
    assert "marked.parse" not in preview_js
    assert 'short_answer: 3, equation: 3, word_problem: 5, essay: 22' in preview_js
    assert "buildAnswerKey(exam, labels, language, examLabels)" in preview_js
    assert "item.correct_answer" in preview_js
    assert "addResponseLines(question" in preview_js
    assert "renderExamOutput(data.exams, data.metadata || {})" in main_js


def test_browser_preview_matches_export_header_tf_blanks_and_math_layout():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    preview_js = (root / "FrontEnd/js/exam-view.js").read_text(encoding="utf-8")
    math_js = (root / "FrontEnd/js/math-renderer.js").read_text(encoding="utf-8")
    html = (root / "FrontEnd/index.html").read_text(encoding="utf-8")
    css = (root / "FrontEnd/css/styles.css").read_text(encoding="utf-8")

    assert 'const grid = element("div", "preview-meta-row")' in preview_js
    assert "[labels.className, metadata.class_name]" not in preview_js
    assert '"preview-student-name"' in preview_js
    assert '"preview-student-class"' in preview_js
    assert 'blank + blank[0].repeat(4)' in preview_js
    assert 'stem.appendChild(element("span", "preview-tf-choices"' in preview_js
    assert "renderEquation(equation, value)" in preview_js
    assert "window.katex.render" in math_js
    assert "toMathLatex" in math_js
    assert "katex@0.18.7/dist/katex.min.js" in html
    assert ".preview-equation-text .katex-display" in css


def test_frontend_markdown_copy_is_questions_first_and_uses_toast():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    preview_js = (root / "FrontEnd/js/exam-view.js").read_text(encoding="utf-8")
    html = (root / "FrontEnd/index.html").read_text(encoding="utf-8")
    css = (root / "FrontEnd/css/styles.css").read_text(encoding="utf-8")
    assert 'const questions = ["# Questions"]' in preview_js
    assert 'const answers = ["# Answers"]' in preview_js
    assert preview_js.index('const questions = ["# Questions"]') < preview_js.index(
        'const answers = ["# Answers"]'
    )
    assert "buildQuestionsFirstMarkdown(exams, metadata)" in preview_js
    assert 'id="copyToast"' in html
    assert "3000" in preview_js
    assert ".copy-toast.show" in css


def test_frontend_has_content_direction_and_localized_dynamic_statuses():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    preview_js = (root / "FrontEnd/js/exam-view.js").read_text(encoding="utf-8")
    topics_js = (root / "FrontEnd/js/topics.js").read_text(encoding="utf-8")
    i18n = (root / "FrontEnd/js/i18n.js").read_text(encoding="utf-8")
    upload = (root / "FrontEnd/js/upload.js").read_text(encoding="utf-8")
    export = (root / "FrontEnd/js/export.js").read_text(encoding="utf-8")
    assert "applyContentDirection" in preview_js
    assert "applyContentDirection(stitle" in topics_js
    for key in (
        "status.uploading", "stage.generating_titles", "status.generation_failed",
        "status.creating_export", "status.export_downloaded", "saved.loading",
    ):
        assert i18n.count(f'"{key}"') == 2
    assert 'setUploadStage("uploading")' in upload
    assert "status.processing_pdf" not in i18n
    assert "parsing_and_indexing" not in upload
    assert 't("status.creating_export"' in export


def test_frontend_question_type_order_and_word_problem_naming():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    html = (root / "FrontEnd/index.html").read_text(encoding="utf-8")
    i18n = (root / "FrontEnd/js/i18n.js").read_text(encoding="utf-8")
    cards = [
        "card-mcq", "card-fitb", "card-tf", "card-definition",
        "card-why", "card-equation", "card-word_problem", "card-essay",
    ]
    positions = [html.index(f'id="{card}"') for card in cards]
    assert positions == sorted(positions)
    assert '"qtype.word_problem": "Word Problem"' in i18n
    assert '"qtype.word_problem": "سؤال كلامي"' in i18n
    assert "verbal_equation" not in html + i18n
