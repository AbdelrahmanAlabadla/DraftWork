"""Create deterministic Arabic/math export fixtures for visual regression review."""
from __future__ import annotations

import json
from pathlib import Path

from app.exports.docx_exporter import render_answers_docx, render_exam_docx
from app.exports.pdf_exporter import render_answers_pdf, render_exam_pdf


OUTPUT = Path("artifacts/rendering_regression_23467d6b")


def fixture() -> tuple[dict, dict]:
    exam = {
        "model_number": 1,
        "document_language": "ar",
        "questions": {
            "mcq": [{
                "question_id": "reg_mcq_1",
                "question": "أي نظام يعبّر عن القيود $x+y\\leq 25$ و$72x+96y\\geq 925$؟",
                "options": {
                    "A": "$x+y\\leq25,\\;72x+96y\\geq925$",
                    "B": "$x+y\\geq25,\\;72x+96y\\leq925$",
                    "C": "$x-y=25$",
                    "D": "$72x+96y=25$",
                },
                "correct_answer": "A",
            }],
            "true_false": [{
                "question_id": "reg_tf_1",
                "statement": "إذا كان $7\\leq c\\leq14$ و$8\\leq p\\leq12$ و$c+p\\leq23$، فإن $c=12$ و$p=10$ يحققان القيود.",
                "answer": True,
            }],
            "equation": [{
                "question_id": "reg_eq_1",
                "equation": "حل النظام الآتي وحدد نقطة التقاطع:\n$$\\begin{aligned}2x+y&=7\\\\x-y&=2\\end{aligned}$$",
                "solution_steps": ["بجمع المعادلتين: $3x=9$", "إذن $x=3$ و$y=1$"],
                "final_answer": "$(x,y)=(3,1)$",
            }],
            "word_problem": [{
                "question_id": "reg_wp_1",
                "question": "استثمرت سارة مبلغاً في ثلاثة صناديق. كوّن نظاماً من ثلاث معادلات يصف رأس المال والعائد، ثم احسب التوزيع. استخدم المتغيرات $x$ و$y$ و$z$.",
                "solution_steps": [
                    "$x+y+z=350$",
                    "$0.06x+0.085y+0.11z=29.785$",
                    "$z=x+50$",
                ],
                "final_answer": "$x=100,\\;y=100,\\;z=150$",
            }],
        },
    }
    metadata = {
        "exam_title": "اختبار رياضيات — نموذج مراجعة التنسيق",
        "class_name": "الصف الثاني عشر",
        "duration": "45 دقيقة",
        "teacher_name": "DraftWork",
        "footer_message": "تحقق من خطوات الحل بوضوح.",
        "document_language": "ar",
    }
    return exam, metadata


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    exam, metadata = fixture()
    outputs = {
        "corrected_exam.pdf": render_exam_pdf(exam, metadata),
        "corrected_answer_key.pdf": render_answers_pdf(exam, metadata),
        "corrected_exam.docx": render_exam_docx(exam, metadata),
        "corrected_answer_key.docx": render_answers_docx(exam, metadata),
    }
    for name, payload in outputs.items():
        (OUTPUT / name).write_bytes(payload)
    (OUTPUT / "fixture.json").write_text(
        json.dumps({"exam": exam, "metadata": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
