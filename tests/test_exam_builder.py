from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.online import exam_builder  # noqa: E402
from app.online.models import (  # noqa: E402
    contains_forbidden_phrase,
    normalize_text,
    parse_questions,
)


class FakeGraph:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def invoke(self, state: dict) -> dict:
        self.calls.append(state)
        if self.responses:
            return self.responses.pop(0)
        return {"questions": [], "error": "no more responses"}


def _install(monkeypatch, responses: list[dict]) -> FakeGraph:
    graph = FakeGraph(responses)
    monkeypatch.setattr("app.online.exam_builder.get_exam_graph", lambda: graph)
    return graph


def _mcq(text: str, correct: str = "A") -> dict:
    return {
        "question": text,
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "correct_answer": correct,
    }


def test_retry_fills_shortfall(monkeypatch):
    graph = _install(
        monkeypatch,
        [
            {"questions": [_mcq("q1")]},
            {"questions": [_mcq("q2"), _mcq("q3")]},
        ],
    )
    result = exam_builder.generate_exam("doc-x", [("mcq", 3)])
    assert len(result["questions"]["mcq"]) == 3
    assert result["warnings"] == []
    # Second attempt requested the remaining count.
    assert graph.calls[1]["number_of_questions"] == 2


def test_forbidden_phrase_rejected_and_regenerated(monkeypatch):
    _install(
        monkeypatch,
        [
            {"questions": [_mcq("What is AI according to the analysis?")]},
            {"questions": [_mcq("What is AI?")]},
        ],
    )
    result = exam_builder.generate_exam("doc-x", [("mcq", 1)])
    questions = result["questions"]["mcq"]
    assert len(questions) == 1
    assert questions[0]["question"] == "What is AI?"
    assert not contains_forbidden_phrase(questions[0]["question"])


def test_duplicate_questions_deduped(monkeypatch):
    _install(
        monkeypatch,
        [
            {"questions": [_mcq("What is AI?"), _mcq("What is AI?")]},
            {"questions": [_mcq("What is ML?")]},
        ],
    )
    result = exam_builder.generate_exam("doc-x", [("mcq", 2)])
    questions = result["questions"]["mcq"]
    assert len(questions) == 2
    assert {q["question"] for q in questions} == {"What is AI?", "What is ML?"}


def test_type_incomplete_after_attempts_warns(monkeypatch):
    _install(
        monkeypatch,
        [{"questions": [_mcq("q1")]}, {"questions": []}, {"questions": []}],
    )
    result = exam_builder.generate_exam("doc-x", [("mcq", 5)])
    assert len(result["questions"]["mcq"]) == 1
    assert any("mcq: 1/5" in w for w in result["warnings"])


def test_assemble_exam_continuous_numbering():
    questions = {
        "mcq": [_mcq("mcq1"), _mcq("mcq2")],
        "true_false": [{"statement": "tf1", "answer": "True"}],
        "short_answer": [{"question": "sa1", "reference_answer": "ans"}],
    }
    md = exam_builder.assemble_exam(questions)
    assert "## Multiple Choice" in md
    assert "## True / False" in md
    assert "## Short Answer" in md
    # Continuous numbering across sections.
    assert md.index("1. mcq1") < md.index("2. mcq2") < md.index("3. tf1") < md.index("4. sa1")


def test_assemble_exam_skips_empty_types():
    md = exam_builder.assemble_exam({"mcq": [_mcq("only")], "true_false": [], "short_answer": []})
    assert "## Multiple Choice" in md
    assert "## True / False" not in md
    assert "## Short Answer" not in md


def test_parse_questions_rejects_structural_badness():
    raw = (
        '{"questions":['
        '{"question":"ok","options":{"A":"a","B":"b","C":"c","D":"d"},"correct_answer":"B"},'
        '{"question":"missing option","options":{"A":"a","B":"b","C":"c"},"correct_answer":"A"},'
        '{"question":"bad answer letter","options":{"A":"a","B":"b","C":"c","D":"d"},"correct_answer":"E"},'
        '{"question":"","options":{"A":"a","B":"b","C":"c","D":"d"},"correct_answer":"A"}'
        "]}"
    )
    mcqs = parse_questions("mcq", raw)
    assert len(mcqs) == 1
    assert mcqs[0]["question"] == "ok"


def test_normalize_text_and_forbidden_phrases():
    assert normalize_text("What is AI?") == normalize_text("what is ai")
    assert contains_forbidden_phrase("according to the analysis of the data")
    assert contains_forbidden_phrase("as shown in the figure")
    assert not contains_forbidden_phrase("Which digit is often confused?")
