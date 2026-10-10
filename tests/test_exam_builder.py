from __future__ import annotations

import re
import threading
from typing import Any

import pytest

from app.logging_conf import configure_logging
from app.llm.deepseek import DeepSeekNetworkError

configure_logging("WARNING")

from app.online import exam_builder  # noqa: E402
from app.online.models import (  # noqa: E402
    contains_forbidden_phrase,
    normalize_text,
    parse_questions,
)
from app.online.eval_stats import create_pipeline_eval, public_eval
from app.online.assessment_catalog import apply_canonical_task_ids


def _mcq(text: str, correct: str = "A") -> dict:
    return {
        "question": text,
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "correct_answer": correct,
    }


def _reviewed_plan_item(qtype: str, slot_id: str, *, chunk: str = "c1", index: int = 1) -> dict:
    """Build the minimum complete item accepted by the reviewed-plan gate."""
    item = {
        "slot_id": slot_id,
        "source_chunk_id": chunk,
        "support_chunk_ids": [],
        "topic": "Topic",
        "concept": f"Concept {qtype} {index}",
        "canonical_skill": f"Assess concept {qtype} {index}",
        "knowledge_focus": f"Knowledge focus {qtype} {index}",
        "student_work_product": f"Provide a {qtype} response for concept {index}",
        "relationship": None,
        "target": f"Target {qtype} {index}",
        "task_operation": "recall_term" if qtype == "fill_in_the_blank" else "apply_concept",
        "given_conditions": [],
        "required_givens": [],
        "text_answerable": True,
        "reasoning_mode": "conceptual_recall" if qtype == "fill_in_the_blank" else "conceptual_reasoning",
        "difficulty": "easy",
        "idea_to_test": f"Identify and explain concept {qtype} {index}",
    }
    if qtype == "fill_in_the_blank":
        item.update({
            "fitb_entry_id": f"fitb_entry_{index}",
            "answer": f"Target {qtype} {index}",
            "evidence_chunk_id": chunk,
            "clue_basis": f"The source supports target {qtype} {index}.",
        })
    else:
        apply_canonical_task_ids(item)
    return item


def test_context_for_items_includes_main_and_support_chunks_only():
    chunks = [
        {"child_id": "main", "content": "main evidence"},
        {"child_id": "support", "content": "support evidence"},
        {"child_id": "other", "content": "unassigned evidence"},
    ]
    context = exam_builder._context_for_items(
        [{"source_chunk_id": "main", "support_chunk_ids": ["support"]}],
        chunks,
        "fallback",
    )
    assert "main evidence" in context
    assert "support evidence" in context
    assert "unassigned evidence" not in context


def test_mcq_answer_positions_are_balanced_and_keep_correct_content():
    questions = {"mcq": []}
    for index in range(8):
        questions["mcq"].append({
            "options": {"A": f"correct-{index}", "B": "b", "C": "c", "D": "d"},
            "correct_answer": "A",
        })
    exam_builder.balance_mcq_answer_positions(questions)
    answers = [item["correct_answer"] for item in questions["mcq"]]
    assert answers == ["B", "D", "A", "C", "C", "A", "D", "B"]
    assert all(
        item["options"][item["correct_answer"]] == f"correct-{index}"
        for index, item in enumerate(questions["mcq"])
    )
    snapshot = [dict(item["options"]) for item in questions["mcq"]]
    exam_builder.balance_mcq_answer_positions(questions)
    assert [item["options"] for item in questions["mcq"]] == snapshot


_COUNT_RE = re.compile(r"Create exactly (\d+)")


def _qtype(prompt: str) -> str:
    if "Multiple Choice (MCQ)" in prompt:
        return "mcq"
    if "True/False exam question" in prompt:
        return "true_false"
    return "why"


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
    monkeypatch.setattr("app.llm.factory.create_llm_client", lambda: client)
    return client


def test_plan_followed_fills_each_planned_slot(monkeypatch):
    planned = [
        {
            "topic": "T1", "concept": "Forces", "source_chunk_id": "chunk-1",
            "relationship": "Newton's second law", "target": "acceleration",
            "reasoning_mode": "calculation", "difficulty": "medium",
            "idea_to_test": "concept one", "concept_to_test": "concept one",
        },
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
    assert "relationship=Newton's second law" in p
    assert "target=acceleration" in p
    assert "reasoning_mode=calculation" in p
    assert "difficulty=medium" in p
    assert "slot_id=m1_mcq_1" in p


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


def test_cross_model_near_duplicate_is_allowed(monkeypatch):
    # Semantic overlap across models is allowed; the plan assigns distinct ideas.
    prior_exam = [_mcq("What is supervised learning?")]
    planned = [{"topic": "S", "concept_to_test": "supervised learning"}]
    client = SequencedClient([[_mcq("Which learning method uses labeled examples?")]])
    _patch_client(monkeypatch, client)
    questions, _ = exam_builder._generate_type_from_plan(
        "mcq", planned, "ctx", "mix", 2, set(), [], prior_exam
    )
    assert len(questions) == 1


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


def test_generation_rejection_hook_counts_existing_filter(monkeypatch):
    planned = [
        {"topic": "T", "concept_to_test": "a"},
        {"topic": "T", "concept_to_test": "b"},
    ]
    client = SequencedClient([
        [_mcq("What is AI?"), _mcq("What is AI?")],
        [_mcq("What is ML?")],
    ])
    _patch_client(monkeypatch, client)
    stats = create_pipeline_eval([("mcq", 2)], 1)
    questions, _ = exam_builder._generate_type_from_plan(
        "mcq", planned, "ctx", "mix", 1, set(), [], [], eval_stats=stats
    )
    assert len(questions) == 2
    model = public_eval(stats)["models"]["1"]
    assert model["generation_rejected"] == 1
    assert model["generation_rejection_reasons"] == {"duplicate": 1}


def test_exhausted_transport_retries_do_not_multiply_generation_attempts(monkeypatch):
    calls = []

    def failed_request(*_args, **_kwargs):
        calls.append("request")
        raise DeepSeekNetworkError("network retries exhausted")

    monkeypatch.setattr(
        "app.llm.factory.create_llm_client", lambda: object()
    )
    monkeypatch.setattr(exam_builder, "request_structured", failed_request)

    questions, warnings = exam_builder._generate_type_from_plan(
        "why",
        [{"slot_id": "m1_why_1", "topic": "T", "concept": "C"}],
        "ctx", "mix", 1, set(), [], [],
    )

    assert questions == []
    assert calls == ["request"]
    assert any("attempt 1" in warning for warning in warnings)


def test_generation_node_snapshots_before_shortfall(monkeypatch):

    def fake_bundle(*args, **kwargs):
        return {"mcq": [_mcq("initial")], "true_false": [], "fill_in_the_blank": None}, []

    def fake_shortfall(questions, *args, **kwargs):
        questions["mcq"].append(_mcq("shortfall"))
        kwargs["appended_by_type"]["mcq"] += 1
        return []

    monkeypatch.setattr(exam_builder, "_generate_obj_bundle", fake_bundle)
    monkeypatch.setattr(exam_builder, "_repair_shortfalls", fake_shortfall)
    stats = create_pipeline_eval([("mcq", 2)], 1)
    out = exam_builder.generate_exams_node(
        {
            "document_id": "doc",
            "tasks": [("mcq", 2)],
            "num_models": 1,
            "difficulty": "easy",
            "context": "ctx",
            "planner_context": "ctx",
            "document_language": "en",
            "plans": [{"model_number": 1, "items": {
                "mcq": [
                    _reviewed_plan_item("mcq", "m1_mcq_1", index=1),
                    _reviewed_plan_item("mcq", "m1_mcq_2", index=2),
                ],
            }}],
            "plan_errors": [],
            "warnings": [],
            "eval_stats": stats,
        }
    )
    model = public_eval(out["eval_stats"])["models"]["1"]
    assert model["generated_first"] == 1
    assert model["missing_first"] == 1
    assert model["shortfall_generated"] == 1
    assert model["shortfall_still_missing"] == 0


def test_generation_node_records_complete_twenty_question_mixed_shape(monkeypatch):
    """The live 10/3/3/3/1 shape is counted before shortfall repair."""
    calls = {"bundle": 0, "free": []}

    def fake_bundle(*args, **kwargs):
        calls["bundle"] += 1
        return {
            "mcq": [_mcq(f"MCQ {i}") for i in range(1, 11)],
            "true_false": [
                {"statement": f"Statement {i}", "answer": "True"}
                for i in range(1, 4)
            ],
            "fill_in_the_blank": {
                "word_bank": ["one", "two", "three", "four", "five"],
                "items": [
                    {"question": f"Complete ________ {i}", "answers": ["one"]}
                    for i in range(1, 4)
                ],
            },
        }, []

    def fake_free(qtype, planned, *args, **kwargs):
        calls["free"].append((qtype, len(planned)))
        if qtype == "essay":
            return [
                {
                    "question": "Essay question",
                    "reference_answer": "Essay answer",
                    "key_points": ["point"],
                }
            ], []
        return [
            {"question": f"Why does effect {i} occur?", "reference_answer": "Answer"}
            for i in range(1, len(planned) + 1)
        ], []

    def fake_fitb(*args, **kwargs):
        return {
            "word_bank": ["one", "two", "three", "four", "five"],
            "items": [
                {"question": f"Complete ________ {i}", "answers": ["one"]}
                for i in range(1, 4)
            ],
        }, []

    monkeypatch.setattr(exam_builder, "_generate_obj_bundle", fake_bundle)
    monkeypatch.setattr(exam_builder, "_generate_fitb_type", fake_fitb)
    monkeypatch.setattr(exam_builder, "_generate_type_from_plan", fake_free)

    tasks = [
        ("mcq", 10),
        ("true_false", 3),
        ("fill_in_the_blank", 3),
        ("why", 3),
        ("essay", 1),
    ]
    plan_items = {
        qtype: [
            _reviewed_plan_item(qtype, f"m1_{qtype}_{i}", index=i)
            for i in range(1, count + 1)
        ]
        for qtype, count in tasks
    }
    stats = create_pipeline_eval(tasks, 1)
    out = exam_builder.generate_exams_node(
        {
            "document_id": "doc",
            "tasks": tasks,
            "num_models": 1,
            "difficulty": "easy",
            "context": "ctx",
            "planner_context": "ctx",
            "document_language": "en",
            "plans": [{"model_number": 1, "items": plan_items}],
            "plan_errors": [],
            "warnings": [],
            "eval_stats": stats,
        }
    )

    model = public_eval(out["eval_stats"])["models"]["1"]
    assert calls == {
        "bundle": 1,
        "free": [("why", 3), ("essay", 1)],
    }
    assert model["requested_questions"] == 20
    assert model["generated_first"] == 20
    assert model["missing_first"] == 0
    assert model["shortfall_generated"] == 0
    assert model["shortfall_still_missing"] == 0


def test_independent_question_types_generate_in_parallel_and_merge_in_task_order(monkeypatch):
    active = 0
    peak = 0
    lock = threading.Lock()
    release = threading.Event()
    # Three generation workers plus this coordinating test thread.
    entered = threading.Barrier(4)

    def fake_free(qtype, planned, *args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        entered.wait(timeout=2)
        release.wait(timeout=2)
        with lock:
            active -= 1
        return [{
            "slot_id": planned[0]["slot_id"],
            "question": f"{qtype} question",
            "reference_answer": f"{qtype} answer",
            **({"key_points": ["point"]} if qtype == "essay" else {}),
        }], []

    monkeypatch.setattr(exam_builder, "MAX_PARALLEL_GENERATION_CALLS", 5)
    monkeypatch.setattr(exam_builder, "_generate_type_from_plan", fake_free)
    monkeypatch.setattr(exam_builder, "_repair_shortfalls", lambda *args, **kwargs: [])

    tasks = [("definition", 1), ("why", 1), ("essay", 1)]
    state = {
        "document_id": "doc",
        "tasks": tasks,
        "num_models": 1,
        "difficulty": "easy",
        "context": "evidence",
        "document_language": "en",
        "plans": [{
            "model_number": 1,
            "items": {
                qtype: [_reviewed_plan_item(qtype, f"m1_{qtype}_1")]
                for qtype, _count in tasks
            },
        }],
        "retrieved_chunks": [{"child_id": "c1", "content": "evidence"}],
        "plan_errors": [],
        "warnings": [],
    }

    result_holder = {}
    thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", exam_builder.generate_exams_node(state)
        )
    )
    thread.start()
    entered.wait(timeout=2)
    release.set()
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert peak == 3
    questions = result_holder["result"]["generated_exams"][0]["questions"]
    assert list(questions) == ["definition", "why", "essay"]


def test_assemble_exam_continuous_numbering():
    questions = {
        "mcq": [_mcq("mcq1"), _mcq("mcq2")],
        "true_false": [{"statement": "tf1", "answer": "True"}],
        "why": [{"question": "sa1", "reference_answer": "ans"}],
    }
    md = exam_builder.assemble_exam(questions)
    assert "## Multiple Choice" in md
    assert "## True / False" in md
    assert "## Why Questions" in md
    assert md.index("1. mcq1") < md.index("2. mcq2") < md.index("3. tf1") < md.index("4. sa1")


def test_assemble_exam_skips_empty_types():
    md = exam_builder.assemble_exam({"mcq": [_mcq("only")], "true_false": [], "why": []})
    assert "## Multiple Choice" in md
    assert "## True / False" not in md
    assert "## Why Questions" not in md


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
    assert not contains_forbidden_phrase("Calculate the yield according to the balanced equation.")
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


def test_authorized_catalog_exhaustion_reuse_passes_generator_task_guard():
    existing = {"slot_id": "s1", "question": "Existing question"}
    plan_by_slot = {
        "s1": {"slot_id": "s1", "task_id": "task_equation"},
        "s2": {
            "slot_id": "s2", "task_id": "task_equation",
            "catalog_replacement_stage": "reused_task_after_catalog_exhaustion",
            "reused_due_to_catalog_exhaustion": True,
            "task_reused": True,
            "reuse_reason": "no_unused_compatible_task",
            "reused_task_id": "task_equation",
            "required_instance_distinctions": ["different_direction_of_reasoning"],
            "forced_duplicate_slot_ids": ["s1"],
        },
    }
    assert exam_builder._is_semantic_plan_duplicate("s2", plan_by_slot, [existing]) is False


def test_unauthorized_catalog_task_reuse_stays_blocked_by_generator_guard():
    existing = {"slot_id": "s1", "question": "Existing question"}
    plan_by_slot = {
        "s1": {"slot_id": "s1", "task_id": "task_equation"},
        "s2": {
            "slot_id": "s2", "task_id": "task_equation",
            "reused_due_to_catalog_exhaustion": True,
            # Missing the allocator's full authorization contract.
        },
    }
    assert exam_builder._is_semantic_plan_duplicate("s2", plan_by_slot, [existing]) is True


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
        "why": [{"question": "sa1", "reference_answer": "ans"}],
        "definition": [{"term": "CPU", "reference_answer": "Central processing unit."}],
        "equation": [{"equation": "2x + 5 = 15", "solution_steps": ["2x = 10", "x = 5"], "final_answer": "x = 5"}],
        "word_problem": [{"question": "A car travels 120 km in 2 hours. Find its speed.", "solution_steps": ["speed = distance ÷ time", "120 ÷ 2 = 60"], "final_answer": "60 km/h"}],
        "essay": [{"question": "es1", "reference_answer": "ref", "key_points": ["kp1", "kp2"]}],
    }
    md = exam_builder.assemble_exam(questions)
    order = [
        md.index("## Multiple Choice"),
        md.index("## Fill in the Blank"),
        md.index("## True / False"),
        md.index("## Define"),
        md.index("## Why Questions"),
        md.index("## Solve the equations"),
        md.index("## Solve the word problems"),
        md.index("## Essay"),
    ]
    assert order == sorted(order)
    assert 'class="word-bank"' in md
    assert "area · perimeter · length · width · diagonal" in md
    assert "1. mcq1" in md
    assert "2. To calculate the ________" in md and "3. A rectangle's ________" in md
    assert "4. tf1" in md and "5. CPU:" in md
    assert "6. sa1" in md and "7. 2x + 5 = 15" in md
    assert "8. A car travels" in md and "9. es1" in md
    assert "**Reference answer:** ref" in md
    assert "kp1" in md and "kp2" in md


def test_assembly_excludes_non_pass_questions_from_student_output():
    first = _mcq("first")
    second = _mcq("second")
    first.update({"question_id": "model1_mcq_1", "slot_id": "m1_mcq_1"})
    second.update({"question_id": "model1_mcq_2", "slot_id": "m1_mcq_2", "question": "Broken question"})
    state = {
        "document_language": "en",
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [first, second]}}],
        "validation_reports": [{
            "model_number": 1,
            "verdicts": [
                {"question_id": "model1_mcq_1", "action": "PASS", "reason": ""},
                {
                    "question_id": "model1_mcq_2", "action": "FIX_QUESTION",
                    "failure_code": "unsupported_given", "reason": "missing wage values",
                },
            ],
        }],
        "eval_stats": None,
    }
    result = exam_builder.assemble_exams_node(state)
    public = result["generated_exams"][0]
    assert [item["question_id"] for item in public["questions"]["mcq"]] == ["model1_mcq_1"]
    assert "Broken question" not in public["markdown"]
    assert result["invalid_questions"][0]["slot_id"] == "m1_mcq_2"


def test_assembly_preserves_four_valid_fitb_when_one_slot_fails():
    items = []
    verdicts = []
    answers = ["alpha", "beta", "gamma", "delta", "epsilon"]
    for index, answer in enumerate(answers, start=1):
        qid = f"model1_fill_in_the_blank_{index}"
        items.append({
            "question_id": qid,
            "slot_id": f"m1_fill_in_the_blank_{index}",
            "fitb_entry_id": f"fitb_{answer}",
            "question": f"The grounded term is ________ for {answer}.",
            "answers": [answer],
            "distractors": ["theta", "lambda"],
        })
        verdicts.append({
            "question_id": qid,
            "action": "PASS" if index < 5 else "FIX_QUESTION",
            "failure_code": None if index < 5 else "ambiguous_or_structural",
            "reason": "" if index < 5 else "invalid_structure",
        })
    state = {
        "document_language": "en",
        "generated_exams": [{"model_number": 1, "questions": {
            "fill_in_the_blank": {
                "items": items,
                "word_bank": [*answers, "theta", "lambda"],
            }
        }}],
        "validation_reports": [{"model_number": 1, "verdicts": verdicts}],
        "eval_stats": None,
    }

    result = exam_builder.assemble_exams_node(state)
    section = result["generated_exams"][0]["questions"]["fill_in_the_blank"]

    assert len(section["items"]) == 4
    assert len(section["word_bank"]) == 6
    assert "epsilon" not in section["word_bank"]
    assert len(result["invalid_questions"]) == 1


def test_generation_reports_separate_quality_status_and_counts(monkeypatch):
    valid = _mcq("Valid question")
    valid.update({"question_id": "model1_mcq_1", "slot_id": "m1_mcq_1"})

    class FakeGraph:
        def invoke(self, _state):
            return {
                "generated_exams": [{
                    "model_number": 1, "questions": {"mcq": [valid]},
                    "markdown": "Valid question", "document_language": "en",
                }],
                "invalid_questions": [{"slot_id": "m1_mcq_2", "reason": "invalid"}],
                "warnings": [], "document_language": "en", "eval_stats": None,
                "assessment_profile": {}, "planning_issues": [],
                "plans": [{"model_number": 1, "items": {"mcq": [{
                    "slot_id": "m1_mcq_1", "task_id": "task-1",
                    "task_family_id": "family-1", "coverage_id": "coverage-1",
                    "concept_id": "concept-1", "relationship_id": "relationship-1",
                    "concept": "waves", "task_operation": "distinguish",
                    "target": "wave types", "reasoning_mode": "classification",
                    "student_work_product": "select the matching wave type",
                    "difficulty": "medium", "source_chunk_id": "chunk-1",
                    "support_chunk_ids": ["chunk-2"], "task_reused": True,
                    "reuse_reason": "no_unused_compatible_task",
                    "catalog_replacement_stage": "reused_task_after_catalog_exhaustion",
                    "fitb_answer_reused": False,
                }]}}],
            }

    monkeypatch.setattr(exam_builder, "get_exam_graph", lambda: FakeGraph())
    result = exam_builder.generate_exams("doc", [("mcq", 3)], 1)
    assert result["workflow_status"] == "completed"
    assert result["quality_status"] == "incomplete"
    assert result["quality_summary"] == {
        "requested": 3, "valid": 1, "missing": 1, "invalid": 1,
        "status": "incomplete",
    }
    identity = result["semantic_identities"][0]
    assert identity["task_id"] == "task-1"
    assert identity["task_family_id"] == "family-1"
    assert identity["coverage_id"] == "coverage-1"
    assert identity["concept_id"] == "concept-1"
    assert identity["relationship_id"] == "relationship-1"
    assert identity["source_chunk_id"] == "chunk-1"
    assert identity["support_chunk_ids"] == ["chunk-2"]
    assert identity["catalog_replacement_stage"] == "reused_task_after_catalog_exhaustion"


def test_four_invalid_or_missing_slots_need_attention(monkeypatch):
    class FakeGraph:
        def invoke(self, _state):
            return {
                "generated_exams": [{"model_number": 1, "questions": {}, "markdown": ""}],
                "invalid_questions": [], "warnings": [], "document_language": "en",
                "eval_stats": None, "assessment_profile": {}, "planning_issues": [],
            }

    monkeypatch.setattr(exam_builder, "get_exam_graph", lambda: FakeGraph())
    result = exam_builder.generate_exams("doc", [("mcq", 4)], 1)
    assert result["quality_status"] == "incomplete"
    assert result["quality_summary"]["missing"] == 4


def test_parse_questions_essay():
    raw = (
        '{"questions":[{"question":"q","reference_answer":"ref",'
        '"key_points":["a","b","c"]}]}'
    )
    parsed = parse_questions("essay", raw)
    assert len(parsed) == 1
    assert parsed[0]["question"] == "q"
    assert parsed[0]["key_points"] == ["a", "b", "c"]


def test_parse_new_question_types_keep_separate_shapes():
    definition = parse_questions(
        "definition", '{"questions":[{"term":"Force","reference_answer":"A push or pull."}]}'
    )
    equation = parse_questions(
        "equation", '{"questions":[{"equation":"2x = 8","solution_steps":["x = 4"],"final_answer":"x = 4"}]}'
    )
    word_problem = parse_questions(
        "word_problem", '{"questions":[{"question":"A car travels 60 km in 2 h. Find its speed.","solution_steps":["60 ÷ 2 = 30"],"final_answer":"30 km/h"}]}'
    )
    assert definition[0]["term"] == "Force"
    assert equation[0]["equation"] == "2x = 8"
    assert word_problem[0]["question"].startswith("A car")


def test_fitb_validation_enforces_word_bank_membership_and_distractors():
    good = {
        "word_bank": ["area", "perimeter", "volume", "length"],
        "items": [
            {"question": "blank one ________ here.", "answers": ["area"],
             "distractors": ["volume", "length"]},
            {"question": "blank two ________ here.", "answers": ["perimeter"],
             "distractors": ["volume", "length"]},
        ],
    }
    assert exam_builder._fitb_errors(good, count=2, within_model=[], previous_exams=[]) == []

    # Answer not in the Word Bank.
    bad = {
        "word_bank": ["area", "perimeter", "length", "width", "dist1", "dist2"],
        "items": [
            {"question": "blank one ________ here.", "answers": ["notinbank"],
             "distractors": ["length", "width"]},
            {"question": "blank two ________ here.", "answers": ["perimeter"],
             "distractors": ["distone", "disttwo"]},
        ],
    }
    errors = exam_builder._fitb_errors(bad, count=2, within_model=[], previous_exams=[])
    assert any("not in Word Bank" in e for e in errors)

    # More than one blank in an item.
    too_many = {
        "word_bank": ["area", "perimeter", "length", "width", "dist1", "dist2"],
        "items": [
            {"question": "a ________ b ________ here.", "answers": ["area"],
             "distractors": ["perimeter", "length"]},
            {"question": "blank two ________ here.", "answers": ["width"],
             "distractors": ["distone", "disttwo"]},
        ],
    }
    errors = exam_builder._fitb_errors(too_many, count=2, within_model=[], previous_exams=[])
    assert any("pure-text single blank" in e for e in errors)

    # Exactly two distractors required.
    wrong_d = {
        "word_bank": ["area", "perimeter", "length", "width", "diagonal"],
        "items": [
            {"question": "blank one ________ here.", "answers": ["area"],
             "distractors": ["length"]},
            {"question": "blank two ________ here.", "answers": ["perimeter"],
             "distractors": ["width", "diagonal"]},
        ],
    }
    errors = exam_builder._fitb_errors(wrong_d, count=2, within_model=[], previous_exams=[])
    assert any("exactly 2 pure-text distractors" in e for e in errors)


def test_generation_entries_keep_sources_and_profile_guidance_isolated():
    profile = {
        "domains": ["chemistry"],
        "educational_level": "secondary",
        "assessment_style": "Reaction-focused",
        "question_types": {
            "equation": {
                "support": "supported",
                "task_operations": ["balance_equation"],
                "form_guidance": "Balance source-supported reactions.",
            }
        },
        "unsupported_visual_tasks": [],
    }
    enriched = exam_builder._generation_plan_items(
        [
            {"source_chunk_id": "a", "support_chunk_ids": []},
            {"source_chunk_id": "b", "support_chunk_ids": []},
        ],
        "equation",
        [{"child_id": "a", "content": "A only"}, {"child_id": "b", "content": "B only"}],
        profile,
    )
    assert enriched[0]["assigned_sources"] == {"a": "A only"}
    assert enriched[1]["assigned_sources"] == {"b": "B only"}
    assert enriched[0]["assessment_guidance"]["task_operations"] == ["balance_equation"]


def test_generation_preserves_supported_work_on_structured_planning_issue():
    with pytest.raises(RuntimeError, match="Generator refused an invalid reviewed plan"):
        exam_builder.generate_exams_node({
            "document_id": "doc",
            "tasks": [("essay", 1)],
            "num_models": 1,
            "plans": [{"model_number": 1, "items": {"essay": []}}],
            "planning_issues": [{
                "question_type": "essay",
                "reason": "The selected material does not support an essay.",
            }],
            "warnings": [],
        })
