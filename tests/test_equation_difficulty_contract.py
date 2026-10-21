from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest
import pypdf

from app.exports.common import split_equation_content
from app.exports.docx_exporter import render_exam_docx
from app.exports.pdf_exporter import render_exam_pdf
from app.exports.math_renderer import render_math
from app.online import assessment_catalog, planner

NODE = shutil.which("node")


def _task(task_id: str, difficulties: list[str], *, operation: str = "solve_equation") -> dict:
    return {
        "task_id": task_id,
        "task_family_id": f"family_{task_id}",
        "coverage_id": f"coverage_{task_id}",
        "concept_id": "concept_linear",
        "relationship_id": "relationship_linear",
        "knowledge_anchor": "linear equations",
        "topic": "linear equations",
        "concept": "linear equations",
        "relationship": "linear equality",
        "task_operation": operation,
        "student_work_product": "solve equation",
        "target": "value of x",
        "reasoning_mode": "symbolic_manipulation",
        "allowed_question_types": ["equation"],
        "evidence_chunk_ids": ["c1"],
        "required_givens": [],
        "scenario_family": None,
        "text_answerable": True,
        "supported_difficulties": difficulties,
    }


def _profile() -> dict:
    types = {}
    for name in planner.VALID_TYPES:
        types[name] = {
            "support": "supported",
            "task_operations": ["solve_equation", "solve_three_variable_system", "calculate_optimum"],
            "form_guidance": "grounded",
            "evidence_chunk_ids": ["c1"],
        }
    return {"question_types": types, "unsupported_visual_tasks": []}


def test_equation_content_separates_arabic_instruction_and_multiline_latex():
    value = "حل النظام الآتي:\n\\[\\begin{cases}3x+2y=12\\\\5x-y=7\\end{cases}\\]"
    assert split_equation_content(value) == [
        ("text", "حل النظام الآتي:"),
        ("math", r"\begin{cases}3x+2y=12\\5x-y=7\end{cases}"),
    ]


def test_math_renderer_accepts_latex_fraction_without_showing_commands():
    rendered = render_math(r"\frac{3}{4}x^2=6")
    assert rendered is not None
    assert rendered.png.startswith(b"\x89PNG")


def test_math_renderer_renders_multiline_latex_system():
    rendered = render_math(r"\begin{cases}3x+2y=12\\5x-y=7\end{cases}")
    assert rendered is not None
    assert rendered.height_px > 40


@pytest.mark.parametrize("delimiter,value", [
    ("$x+y+z=3$", "x+y+z=3"),
    (r"\(a_{2,3}\)", r"a_{2,3}"),
    (r"\[\frac{3}{4}\]", r"\frac{3}{4}"),
    ("$$x^2+y^2$$", "x^2+y^2"),
])
def test_mixed_math_parser_removes_all_supported_delimiters(delimiter, value):
    assert split_equation_content(
        f"Ø§Ø­Ø³Ø¨ {delimiter}", undelimited_as_math=False
    ) == [("text", "Ø§Ø­Ø³Ø¨"), ("math", value)]


def test_docx_renders_inline_arabic_math_and_matrix_without_raw_latex():
    expressions = [
        "$x+y+z=3$", "$80x+20z=456{,}000$", "$a_{2,3}$",
        r"$\frac{3}{4}$", "$x^2+y^2$",
        r"$A=\begin{bmatrix}4&-1&6\\8&3&-5\end{bmatrix}$",
    ]
    exam = {"model_number": 1, "document_language": "ar", "questions": {"mcq": [{
        "question_id": "q1", "slot_id": "m1_mcq_1",
        "question": "Ø§Ø­Ø³Ø¨ " + " ØŒ ".join(expressions),
        "options": {"A": "$x^2+y^2$", "B": "$x+y+z=3$", "C": "1", "D": "2"},
        "correct_answer": "A",
    }]}}
    payload = render_exam_docx(exam, {"exam_title": "Ø§Ø®ØªØ¨Ø§Ø±"})
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
        assert "$" not in xml
        assert r"\begin" not in xml and r"\frac" not in xml
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
        assert len(media) >= len(expressions)


def test_pdf_renders_inline_arabic_math_without_raw_latex_text():
    exam = {"model_number": 1, "document_language": "ar", "questions": {"word_problem": [{
        "question_id": "q1", "slot_id": "m1_word_problem_1",
        "question": r"Ø§Ø­Ø³Ø¨ $\frac{3}{4}$ Ù…Ù† $80x+20z=456{,}000$",
        "solution_steps": [r"\frac{3}{4}"], "final_answer": "$x^2+y^2$",
    }]}}
    payload = render_exam_pdf(exam, {"exam_title": "Ø§Ø®ØªØ¨Ø§Ø±"})
    text = "\n".join(
        page.extract_text() or "" for page in pypdf.PdfReader(io.BytesIO(payload)).pages
    )
    assert "$" not in text
    assert r"\frac" not in text and r"\begin" not in text


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_failed_katex_rendering_preserves_mathematical_meaning(tmp_path):
    source = Path(__file__).parents[1] / "FrontEnd" / "js" / "math-renderer.js"
    module = tmp_path / "math-renderer.mjs"
    module.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    expressions = [
        r"\frac{3}{4}", r"x^{2}", r"a_{2,3}",
        r"A=\begin{bmatrix}4&-1&6\\8&3&-5\end{bmatrix}",
        r"Ø§Ø­Ø³Ø¨ \frac{3}{4}", r"\unsupported{3}{4}",
    ]
    script = """
globalThis.window = {katex: {render: () => { throw new Error('forced failure'); }}};
const errors = [];
console.error = (...args) => errors.push(args);
const math = await import(process.env.MODULE_URL);
const outputs = JSON.parse(process.env.EXPRESSIONS).map((expression) => {
  const classes = [];
  const target = {textContent: '', dataset: {}, classList: {add: (name) => classes.push(name)}};
  const rendered = math.renderEquation(target, expression, false);
  return {expression, rendered, text: target.textContent, dataset: target.dataset, classes};
});
console.log(JSON.stringify({outputs, errors: errors.length}));
"""
    process = subprocess.run(
        [NODE, "--input-type=module", "-e", script], capture_output=True,
        text=True, encoding="utf-8", timeout=30,
        env={
            **os.environ, "MODULE_URL": module.resolve().as_uri(),
            "EXPRESSIONS": json.dumps(expressions, ensure_ascii=False),
        },
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    outputs = result["outputs"]
    assert [item["text"] for item in outputs[:4]] == [
        "(3/4)", "x^(2)", "a_(2,3)", "A=[4, -1, 6; 8, 3, -5]",
    ]
    assert "Ø§Ø­Ø³Ø¨" in outputs[4]["text"] and "(3/4)" in outputs[4]["text"]
    assert outputs[5]["text"] == expressions[5]
    assert outputs[5]["dataset"]["mathRenderFallback"] == "original"
    assert all("math-render-fallback" in item["classes"] for item in outputs)
    assert result["errors"] == len(expressions)


def test_docx_renders_display_latex_as_image_not_raw_commands():
    exam = {"model_number": 1, "document_language": "ar", "questions": {"equation": [{
        "question_id": "q1", "slot_id": "m1_equation_1",
        "equation": "حل المعادلة:\n\\[\\frac{x}{2}=3\\]",
        "solution_steps": [r"\frac{x}{2}=3", "x=6"], "final_answer": "x=6",
    }]}}
    payload = render_exam_docx(exam, {"exam_title": "اختبار"})
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
        assert r"\frac" not in xml
        assert "word/media/" in "\n".join(archive.namelist())


def test_hard_allocation_prefers_native_hard_before_fallback_candidate():
    catalog = {"tasks": [_task("medium", ["easy", "medium"]), _task("hard", ["hard"])]}
    plans, issues = assessment_catalog.allocate_catalog_plan(
        catalog, [("equation", 1)], 1, "hard"
    )
    item = plans[0]["items"]["equation"][0]
    assert issues == []
    assert item["base_catalog_task_id"] == "hard"
    assert item["actual_difficulty"] == "hard"
    assert item["difficulty_adaptation"] is None


def test_hard_shortage_keeps_medium_as_pending_grounded_adaptation():
    catalog = {"tasks": [_task("medium", ["easy", "medium"])]}
    plans, issues = assessment_catalog.allocate_catalog_plan(
        catalog, [("equation", 1)], 1, "hard"
    )
    item = plans[0]["items"]["equation"][0]
    assert issues == []
    assert item["requested_difficulty"] == "hard"
    assert item["actual_difficulty"] == "medium"
    assert item["difficulty_adaptation"]["status"] == "pending"
    assert item["base_catalog_task_id"] == "medium"


def test_exhausted_hard_adaptation_records_honest_medium_fallback(monkeypatch):
    catalog = {"tasks": [_task("medium", ["easy", "medium"])]}
    plans, _ = assessment_catalog.allocate_catalog_plan(catalog, [("equation", 1)], 1, "hard")
    from app.online import difficulty_adjustment
    monkeypatch.setattr(difficulty_adjustment, "request_structured", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no adaptation")))
    result = difficulty_adjustment.adjust_plan_difficulty({
        "document_id": "doc", "tasks": [("equation", 1)], "num_models": 1,
        "difficulty": "hard", "plans": plans,
        "assessment_catalog": catalog, "assessment_profile": _profile(),
        "planning_issues": [], "retrieved_chunks": [{"child_id": "c1", "content": "linear"}],
        "difficulty_adjustment_history": [], "difficulty_adjustment_attempts": {},
    })
    item = result["plans"][0]["items"]["equation"][0]
    assert result["plan_errors"] == []
    assert item["requested_difficulty"] == "hard"
    assert item["actual_difficulty"] == item["difficulty"] == "medium"
    assert item["difficulty_adaptation"]["status"] == "retained"
    assert item["adjustment_attempts"] == 3
    assert item["difficulty_fallback_reason"]


def test_difficulty_agent_is_not_called_when_level_already_matches(monkeypatch):
    from app.online import difficulty_adjustment
    catalog = {"tasks": [_task("easy", ["easy"])]}
    plans, _ = assessment_catalog.allocate_catalog_plan(catalog, [("equation", 1)], 1, "easy")
    monkeypatch.setattr(
        difficulty_adjustment, "request_structured",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected LLM call")),
    )
    result = difficulty_adjustment.adjust_plan_difficulty({
        "tasks": [("equation", 1)], "num_models": 1, "difficulty": "easy",
        "plans": plans, "retrieved_chunks": [{"child_id": "c1", "content": "linear"}],
        "assessment_profile": _profile(), "planning_issues": [],
    })
    assert result["difficulty_adjustment_history"] == []


def test_generated_adjustment_failure_retains_candidate_unchanged(monkeypatch):
    from app.online import difficulty_adjustment
    question = {"slot_id": "m1_equation_1", "question_id": "q1",
                "equation": r"Solve:\n\[2x+5=15\]",
                "solution_steps": ["2x=10", "x=5"], "final_answer": "x=5",
                "solution_certificate": None}
    monkeypatch.setattr(
        difficulty_adjustment, "request_structured",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("unsupported")),
    )
    result, audit = difficulty_adjustment.adjust_generated_question(
        state={"difficulty": "hard", "assessment_profile": {"domain": "STEM"}},
        qtype="equation", question=question,
        plan={"requested_difficulty": "hard", "actual_difficulty": "medium"},
        previous_questions=[],
    )
    assert result == question
    assert result is not question
    assert audit["status"] == "failed"


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_all_difficulty_levels_require_a_consistent_reasoning_profile(level):
    task = _task(f"task-{level}", [level])
    task.update({
        "reasoning_depth": level,
        "reasoning_requirements": ["Solve the assigned linear relationship."],
        "prerequisites": ["linear equality"],
    })
    plans, _ = assessment_catalog.allocate_catalog_plan(
        {"tasks": [task]}, [("equation", 1)], 1, level
    )
    assert planner.validate_plans(
        plans, [("equation", 1)], 1, {"c1"}, level, _profile(), []
    ) == []


def test_failed_math_equation_slots_pass_without_hard_keyword_cues():
    tasks = []
    for task_id, operation, target, requirements in (
        (
            "task_79a38f52823336c6", "solve_three_variable_system",
            "number of seats in each auditorium area",
            ["Use the relation to reduce the system.", "Solve for all three quantities.",
             "Check that the ordered triple satisfies the original constraints."],
        ),
        (
            "task_6723d53a53787ebc", "calculate_optimum",
            "maximum profit and processing quantities",
            ["Form the feasible constraints and objective function.",
             "Evaluate the feasible extreme allocations.",
             "Interpret the maximizing quantities and profit in context."],
        ),
    ):
        task = _task(task_id, ["hard"], operation=operation)
        task.update({
            "target": target, "student_work_product": target,
            "reasoning_mode": "calculation", "reasoning_depth": "hard",
            "reasoning_requirements": requirements,
            "prerequisites": ["systems of linear equations"],
        })
        tasks.append(task)
    plans, _ = assessment_catalog.allocate_catalog_plan(
        {"tasks": tasks}, [("equation", 2)], 1, "hard"
    )
    errors = planner.validate_plans(
        plans, [("equation", 2)], 1, {"c1"}, "hard", _profile(), []
    )
    assert not any("m1_equation_1" in error or "m1_equation_2" in error for error in errors)


def test_removed_catalog_reasoning_metadata_cannot_override_plan_difficulty():
    task = _task("stuffed", ["hard"])
    task.update({
        "reasoning_depth": "easy",
        "reasoning_requirements": ["Perform one direct substitution."],
        "prerequisites": [],
    })
    plans, _ = assessment_catalog.allocate_catalog_plan(
        {"tasks": [task]}, [("equation", 1)], 1, "hard"
    )
    item = plans[0]["items"]["equation"][0]
    item["difficulty"] = "hard"
    item["actual_difficulty"] = "hard"
    item["difficulty_adaptation"] = None
    item["idea_to_test"] = "derive, prove, compare alternatives, and use several meaningful steps"
    errors = planner.validate_plans(
        plans, [("equation", 1)], 1, {"c1"}, "hard", _profile(), []
    )
    assert item["reasoning_depth"] == "hard"
    assert item["reasoning_requirements"] == [
        f"{task['task_operation']}: {task['target']}"
    ]
    assert not any("does not match its reasoning_depth" in error for error in errors)


def test_adjusted_task_rejects_unsupported_reasoning_evidence():
    task = _task("adapted", ["hard"])
    task.update({
        "reasoning_depth": "hard",
        "reasoning_requirements": ["Combine three supported transformations."],
        "prerequisites": ["linear equality"],
    })
    plans, _ = assessment_catalog.allocate_catalog_plan(
        {"tasks": [task]}, [("equation", 1)], 1, "hard"
    )
    item = plans[0]["items"]["equation"][0]
    item["difficulty_adaptation"] = {
        "status": "adapted", "from_difficulty": "medium", "to_difficulty": "hard",
        "base_task_id": "adapted", "strategy": "more transformations",
        "failed_strategies": [],
        "changed_reasoning_requirements": ["Combine three supported transformations."],
        "evidence_support": ["outside-assigned-evidence"],
    }
    errors = planner.validate_plans(
        plans, [("equation", 1)], 1, {"c1"}, "hard", _profile(), []
    )
    assert any("grounded in its assigned evidence" in error for error in errors)
