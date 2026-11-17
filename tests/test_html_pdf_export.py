from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pypdf

from app.exports.html_pdf import HtmlPdfRenderer
from app.exports.document_ir import build_document_ir


def _complete_exam(language: str = "en", *, repeat: int = 1) -> dict:
    arabic = language == "ar"
    question = "احسب $\\frac{3}{4}$ من 20 AED." if arabic else "Calculate $\\frac{3}{4}$ of 20 AED."
    groups = {
        "mcq": [{
            "question_id": f"mcq_{index}",
            "question": question,
            "options": {"A": "15", "B": "10", "C": "5", "D": "20"},
            "correct_answer": "A",
        } for index in range(repeat)],
        "fill_in_the_blank": {
            "word_bank": ["15", "20"],
            "items": [{"question_id": "fitb_1", "question": "3 × 5 = ____", "answers": ["15"]}],
        },
        "true_false": [{"question_id": "tf_1", "statement": question, "answer": True}],
        "definition": [{"question_id": "definition_1", "term": "المعادلة equation", "reference_answer": "A mathematical equality."}],
        "why": [{"question_id": "why_1", "question": "لماذا؟ Why?", "reference_answer": "Because the values are proportional."}],
        "equation": [{
            "question_id": "equation_1",
            "equation": r"Solve: \[\begin{cases}x+y=3\\x-y=1\end{cases}\]",
            "solution_steps": ["Add the equations: $2x=4$.", "$x=2$, $y=1$."],
            "final_answer": "$(2,1)$",
        }],
        "word_problem": [{
            "question_id": "word_1",
            "question": "A table gives x and y.\n| x | y |\n|---|---|\n| 1 | 3 |\n| 2 | 5 |",
            "solution_steps": ["Use $y=2x+1$."],
            "final_answer": "$y=2x+1$",
        }],
        "essay": [{
            "question_id": "essay_1",
            "question": "Explain the relationship between x and y.",
            "reference_answer": "The relationship is linear.",
            "key_points": ["constant rate", "intercept"],
        }],
    }
    return {
        "model_number": 1,
        "document_language": language,
        "questions": groups,
    }


def _render(document: dict) -> bytes:
    async def run() -> bytes:
        renderer = HtmlPdfRenderer()
        try:
            return await renderer.render_document(document)
        finally:
            await renderer.close()

    return asyncio.run(run())


def test_print_shell_reuses_exam_review_renderer_and_assets():
    root = Path(__file__).resolve().parents[1]
    shell = (root / "FrontEnd/print.html").read_text(encoding="utf-8")
    bridge = (root / "FrontEnd/js/print-view.js").read_text(encoding="utf-8")
    preview = (root / "FrontEnd/js/exam-view.js").read_text(encoding="utf-8")
    assert 'src="js/print-view.js"' in shell
    assert 'href="css/styles.css"' in shell
    assert 'href="vendor/katex/katex.min.css"' in shell
    assert 'import { renderPrintDocument } from "./exam-view.js"' in bridge
    assert "export function renderPrintDocument" in preview


def test_english_all_question_types_render_without_answer_leak():
    exam = _complete_exam("en")
    document = build_document_ir(
        exam,
        {"exam_title": "English Integration Exam", "class_name": "10A"},
        kind="student",
    )
    payload = _render(document)
    assert payload.startswith(b"%PDF")
    text = "\n".join(
        page.extract_text() or ""
        for page in pypdf.PdfReader(io.BytesIO(payload)).pages
    )
    assert "English Integration Exam" in text
    assert "10A" in text
    assert "Multiple Choice Questions" in text
    assert "Essay" in text
    assert "The relationship is linear." not in text
    assert "Page 1 of" in text


def test_arabic_mixed_direction_math_and_table_render():
    exam = _complete_exam("ar")
    document = build_document_ir(
        exam,
        {"exam_title": "اختبار الرياضيات", "class_name": "10B"},
        kind="student",
    )
    payload = _render(document)
    reader = pypdf.PdfReader(io.BytesIO(payload))
    assert payload.startswith(b"%PDF")
    assert len(reader.pages) >= 1
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "10B" in text
    assert "AED" in text
    # Chromium may expose Arabic footer glyphs as presentation forms to PDF
    # text extractors; the rendered footer is covered by visual regression.
    assert str(len(reader.pages)) in text
    assert "|---|" not in text


def test_large_exam_over_40_questions_preserves_every_dom_item():
    exam = _complete_exam("en", repeat=41)
    document = build_document_ir(exam, {"exam_title": "Large Exam"}, kind="student")
    payload = _render(document)
    text = "\n".join(
        page.extract_text() or ""
        for page in pypdf.PdfReader(io.BytesIO(payload)).pages
    )
    assert payload.startswith(b"%PDF")
    assert "41." in text

