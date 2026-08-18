from __future__ import annotations

import re
from typing import Any

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.online import exam_builder  # noqa: E402
from app.online.models import (  # noqa: E402
    contains_forbidden_phrase,
    normalize_text,
    parse_questions,
)


def _mcq(text: str, correct: str = "A") -> dict:
    return {
        "question": text,
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "correct_answer": correct,
    }


_COUNT_RE = re.compile(r"Create exactly (\d+)")


def _qtype(prompt: str) -> str:
    if "Multiple Choice (MCQ)" in prompt:
        return "mcq"
    if "True/False exam question" in prompt:
        return "true_false"
    return "short_answer"


def _count(prompt: str) -> int:
    m = _COUNT_RE.search(prompt)
    return int(m.group(1)) if m else 1


class SequencedClient:
    """Returns a fixed ordered list of question-batches across calls."""

    def __init__(self, batches: list[dict | list[dict]]):
        self.batches = list(batches)
        self.calls: list[dict[str, Any]] = []

    def chat_json(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096,
                  timeout=600, max_repair_attempts=2):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.batches:
            item = self.batches.pop(0)
            return item if isinstance(item, list) else {"questions": item.get("questions", item)}
        return {"questions": []}


def _qtype_client(prompt) -> str:
    return _qtype(prompt)


def _patch_client(monkeypatch, client):
    monkeypatch.setattr("app.llm.client.LMStudioClient", lambda *a, **k: client)
    return client


def test_plan_followed_fills_each_planned_slot(monkeypatch):
    planned = [
        {"topic": "T1", "concept_to_test": "concept one"},
        {"topic": "T2", "concept_to_test": "concept two"},
    ]
    client = SequencedClient([[_mcq("What is concept one?"), _mcq("What is concept two?")]])
    _patch_client(monkeypatch, client)

    questions, warnings = exam_builder._generate_type_from_plan(
        "mcq", planned, "FULL SOURCE CONTENT HERE", "mix", 1, set(), [], []
    )
    assert len(questions) == 2
    assert warnings == []
    # Full source must appear before the Question Plan; planned concepts included.
    p = client.calls[0]["prompt"]
    assert p.index("FULL SOURCE CONTENT HERE") < p.index("## Question Plan")
    assert "concept one" in p


def test_missing_concepts_not_free_filled(monkeypatch):
    # The generator retries missing planned items but never invents unplanned ones;
    # if still short it warns rather than fabricating a concept.
    planned = [
        {"topic": "T", "concept_to_test": "c1"},
        {"topic": "T", "concept_to_test": "c2"},
    ]
    client = SequencedClient([[], [], []])  # every attempt returns nothing
    _patch_client(monkeypatch, client)
    questions, warnings = exam_builder._generate_type_from_plan(
        "mcq", planned, "ctx", "mix", 1, set(), [], []
    )
    assert questions == []
    assert len(client.calls) == 3  # bounded retries
    assert any("mcq: 0/2" in w for w in warnings)


def test_cross_model_near_duplicate_rejected(monkeypatch):
    # A question already accepted in an earlier model must be rejected as a
    # near-duplicate, so nothing is accepted in the later model.
    prior_exam = [_mcq("What is supervised learning?")]
    planned = [{"topic": "S", "concept_to_test": "supervised learning"}]
    client = SequencedClient([[_mcq("What is supervised learning?")]])
    _patch_client(monkeypatch, client)
    questions, _ = exam_builder._generate_type_from_plan(
        "mcq", planned, "ctx", "mix", 2, set(), [], prior_exam
    )
    assert questions == []


def test_exact_duplicate_deduped_within_batch(monkeypatch):
    planned = [
        {"topic": "T", "concept_to_test": "a"},
        {"topic": "T", "concept_to_test": "b"},
    ]
    client = SequencedClient([
        [_mcq("What is AI?"), _mcq("What is AI?")],  # duplicate in batch
        [_mcq("What is ML?")],
    ])
    _patch_client(monkeypatch, client)
    questions, warnings = exam_builder._generate_type_from_plan(
        "mcq", planned, "ctx", "mix", 1, set(), [], []
    )
    assert len(questions) == 2
    assert {q["question"] for q in questions} == {"What is AI?", "What is ML?"}


def test_forbidden_phrase_rejected(monkeypatch):
    planned = [{"topic": "A", "concept_to_test": "ai"}]
    client = SequencedClient([[_mcq("What is AI according to the analysis?")]])
    _patch_client(monkeypatch, client)
    questions, _ = exam_builder._generate_type_from_plan(
        "mcq", planned, "ctx", "mix", 1, set(), [], []
    )
    assert questions == []


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


def _near_dup_mcq():
    return {
        "question": "Which of the following best describes the requirement for training data "
        "in a supervised learning system?",
        "options": {
            "A": "The training data must be entirely unlabeled",
            "B": "Only features and predictions, without corresponding labels",
            "C": "The training set must include desired solutions, known as labels",
            "D": "Detecting patterns without needing explicit guidance",
        },
        "correct_answer": "C",
    }


def _near_dup_mcq_twin():
    return {
        "question": "Which characteristic defines the training data for a supervised learning "
        "system?",
        "options": {
            "A": "Learns without any external guidance or teacher",
            "B": "The training set includes desired solutions, known as labels",
            "C": "Attempts to detect patterns without explicit answers",
            "D": "Relies solely on detecting patterns within the input data",
        },
        "correct_answer": "B",
    }


def _distinct_mcq():
    return {
        "question": "What is the difference between a parametric and a nonparametric model?",
        "options": {
            "A": "Fixed versus a growing number of parameters",
            "B": "Learning on the fly",
            "C": "Online versus batch updates",
            "D": "Supervised versus unsupervised",
        },
        "correct_answer": "A",
    }


def test_is_near_duplicate_detects_reworded_duplicates():
    base = _near_dup_mcq()
    twin = _near_dup_mcq_twin()
    distinct = _distinct_mcq()
    assert exam_builder._is_near_duplicate("mcq", twin, [base]) is True
    assert exam_builder._is_near_duplicate("mcq", distinct, [base]) is False


def test_assemble_exam_fitb_section_and_essay():
    questions = {
        "mcq": [_mcq("mcq1")],
        "true_false": [{"statement": "tf1", "answer": "True"}],
        "fill_in_the_blank": {
            "word_bank": ["area", "perimeter", "length", "width", "diagonal"],
            "items": [
                {"question": "To calculate the ________ multiply length and width.", "answers": ["area"]},
                {"question": "A rectangle's ________ is its total distance around.", "answers": ["perimeter"]},
            ],
        },
        "short_answer": [{"question": "sa1", "reference_answer": "ans"}],
        "essay": [{"question": "es1", "reference_answer": "ref", "key_points": ["kp1", "kp2"]}],
    }
    md = exam_builder.assemble_exam(questions)
    order = [
        md.index("## Multiple Choice"),
        md.index("## True / False"),
        md.index("## Fill in the Blank"),
        md.index("## Short Answer"),
        md.index("## Essay"),
    ]
    assert order == sorted(order)
    assert 'class="word-bank"' in md
    assert "area · perimeter · length · width · diagonal" in md
    assert "1. mcq1" in md and "2. tf1" in md
    assert "3. To calculate the ________" in md and "4. A rectangle's ________" in md
    assert "5. sa1" in md and "6. es1" in md
    assert "**Reference answer:** ref" in md
    assert "kp1" in md and "kp2" in md


def test_parse_questions_essay():
    raw = (
        '{"questions":[{"question":"q","reference_answer":"ref",'
        '"key_points":["a","b","c"]}]}'
    )
    parsed = parse_questions("essay", raw)
    assert len(parsed) == 1
    assert parsed[0]["question"] == "q"
    assert parsed[0]["key_points"] == ["a", "b", "c"]


def test_fitb_validation_enforces_word_bank_membership_and_distractors():
    good = {
        "word_bank": ["area", "perimeter", "dist1", "dist2"],
        "items": [
            {"question": "blank one ________ here.", "answers": ["area"]},
            {"question": "blank two ________ here.", "answers": ["perimeter"]},
        ],
    }
    assert exam_builder._fitb_errors(good, count=2, within_model=[], previous_exams=[]) == []

    # Answer not in the Word Bank.
    bad = {
        "word_bank": ["area", "perimeter", "length", "width", "dist1", "dist2"],
        "items": [
            {"question": "blank one ________ here.", "answers": ["notinbank"]},
            {"question": "blank two ________ here.", "answers": ["perimeter"]},
        ],
    }
    errors = exam_builder._fitb_errors(bad, count=2, within_model=[], previous_exams=[])
    assert any("not in Word Bank" in e for e in errors)

    # More than 2 blanks in an item.
    too_many = {
        "word_bank": ["area", "perimeter", "length", "width", "dist1", "dist2"],
        "items": [
            {"question": "a ________ b ________ c ________ here.", "answers": ["area", "perimeter", "length"]},
            {"question": "blank two ________ here.", "answers": ["width"]},
        ],
    }
    errors = exam_builder._fitb_errors(too_many, count=2, within_model=[], previous_exams=[])
    assert any("must be 1-2" in e for e in errors)

    # Exactly two distractors required.
    wrong_d = {
        "word_bank": ["area", "perimeter", "length", "width", "diagonal"],
        "items": [
            {"question": "blank one ________ here.", "answers": ["area"]},
            {"question": "blank two ________ here.", "answers": ["perimeter"]},
        ],
    }
    errors = exam_builder._fitb_errors(wrong_d, count=2, within_model=[], previous_exams=[])
    assert any("exactly 2 Word Bank distractors" in e for e in errors)