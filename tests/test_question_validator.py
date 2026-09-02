"""Tests for stable question IDs + the post-generation validator/repair stage.

The LLM is mocked (FakeClient), so no server call is made. Focus:
- stable, collision-free, model-isolated question IDs
- per-question validation (one call per exam) with PASS / FIX_ANSWER /
  FIX_OPTIONS / FIX_QUESTION / FIX_QUESTION_AND_ANSWER
- minimal-difference repair, enforced by code via fields_to_fix
- ID-keyed write-back (never by position)
- bounded, per-model repair attempts
"""
from __future__ import annotations

import copy
import json

import app.online.exam_builder as eb
from app.online import validator as val
from app.online.validator import (
    MAX_QUESTION_REPAIR_ATTEMPTS,
    repair_exam,
    repair_invalid_questions,
    validate_exam,
    validate_generated_questions,
    REPAIR_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_RETRY_SYSTEM_PROMPT,
)
from app.online.graph import _route_after_validation


class FakeClient:
    """chat_json mock; routes by system prompt and records calls."""

    def __init__(self, validator_result=None, repair_result=None, validator_retry_result=None):
        self.validator_result = validator_result
        self.validator_retry_result = validator_retry_result
        self.repair_result = repair_result
        self.validate_calls = 0
        self.retry_calls = 0
        self.repair_calls = 0
        self.last_validator_prompt = None
        self.last_repair_prompt = None

    def chat_json(self, prompt, system_prompt=None, temperature=0.7,
                  max_tokens=4096, timeout=600, max_repair_attempts=2):
        if system_prompt == VALIDATOR_SYSTEM_PROMPT:
            self.validate_calls += 1
            self.last_validator_prompt = prompt
            return self.validator_result
        if system_prompt == VALIDATOR_RETRY_SYSTEM_PROMPT:
            self.retry_calls += 1
            return self.validator_retry_result
        if system_prompt == REPAIR_SYSTEM_PROMPT:
            self.repair_calls += 1
            self.last_repair_prompt = prompt
            return self.repair_result
        raise AssertionError(f"unexpected system prompt: {system_prompt!r}")


def _validator_entries(prompt):
    block = prompt.split("## Questions to review\n", 1)[1].split(
        "\n\n## Verdict object format", 1
    )[0]
    return json.loads(block)


class EchoValidatorClient:
    """Return PASS for every ID and retain exact per-call batch membership."""

    def __init__(self):
        self.primary_batches = []
        self.retry_batches = []

    def chat_json(self, prompt, system_prompt=None, **kwargs):
        entries = _validator_entries(prompt)
        ids = [entry["question_id"] for entry in entries]
        if system_prompt == VALIDATOR_SYSTEM_PROMPT:
            self.primary_batches.append(ids)
        elif system_prompt == VALIDATOR_RETRY_SYSTEM_PROMPT:
            self.retry_batches.append(ids)
        else:
            raise AssertionError(f"unexpected system prompt: {system_prompt!r}")
        return [_pass_verdict(qid) for qid in ids]


def _mcq(qid, question="Which metric is the harmonic mean of precision and recall?",
         answer="B", options=None):
    return {
        "question_id": qid,
        "question": question,
        "options": options or {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
        "correct_answer": answer,
    }


def _tf(qid, statement="Cyanobacteria released oxygen through photosynthesis.", answer="True"):
    return {"question_id": qid, "statement": statement, "answer": answer}


def _sa(qid, question="Why is peer review important?",
        ref="Peer review allows other experts to evaluate the quality of research and identify errors."):
    return {"question_id": qid, "question": question, "reference_answer": ref}


def _essay(qid, question="Compare precision and recall.", ref="A full reference answer.",
           key_points=None):
    return {"question_id": qid, "question": question, "reference_answer": ref,
            "key_points": key_points or ["a", "b"]}


def _fitb_item(qid, question="To calculate the ________ of a rectangle, multiply length by width.",
               answers=None):
    return {"question_id": qid, "question": question, "answers": answers or ["area"]}


def _fitb(items, bank=None):
    return {"word_bank": bank or ["area", "perimeter", "length", "width", "square"], "items": items}


def _exam(questions, model_number=1):
    return {"model_number": model_number, "questions": questions}


def _verdict(qid, action="PASS", qv=True, av=True, fields=None, reason="", expected_fix=""):
    return {
        "question_id": qid,
        "question_valid": qv,
        "answer_valid": av,
        "action": action,
        "fields_to_fix": fields or [],
        "reason": reason,
        "expected_fix": expected_fix,
    }


def _pass_verdict(qid):
    return _verdict(qid)


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------
def test_assign_ids_unique_per_type():
    mcq1, mcq2 = _mcq("x1"), _mcq("x2")
    del mcq1["question_id"], mcq2["question_id"]
    eb._assign_ids(1, "mcq", [mcq1, mcq2])
    assert [q["question_id"] for q in (mcq1, mcq2)] == ["model1_mcq_1", "model1_mcq_2"]


def test_assign_ids_different_types_no_collision():
    mcq = _mcq("x")
    del mcq["question_id"]
    tf = _tf("x")
    del tf["question_id"]
    eb._assign_ids(1, "mcq", [mcq])
    eb._assign_ids(1, "true_false", [tf])
    ids = {mcq["question_id"], tf["question_id"]}
    assert ids == {"model1_mcq_1", "model1_true_false_1"}


def test_assign_ids_model_isolation():
    mcq1, mcq2 = _mcq("x"), _mcq("y")
    del mcq1["question_id"], mcq2["question_id"]
    eb._assign_ids(1, "mcq", [mcq1])
    eb._assign_ids(2, "mcq", [mcq2])
    assert mcq1["question_id"] == "model1_mcq_1"
    assert mcq2["question_id"] == "model2_mcq_1"
    assert mcq1["question_id"] != mcq2["question_id"]


def test_assign_ids_continues_sequence():
    q1 = _mcq("x")
    del q1["question_id"]
    eb._assign_ids(1, "mcq", [q1])
    q2 = _mcq("y")
    del q2["question_id"]
    eb._assign_ids(1, "mcq", [q1, q2])  # q2 has no id -> continues
    assert q1["question_id"] == "model1_mcq_1"
    assert q2["question_id"] == "model1_mcq_2"


def test_assign_ids_fitb_items():
    item = _fitb_item("x")
    del item["question_id"]
    eb._assign_ids(1, "fill_in_the_blank", _fitb([item]))
    assert item["question_id"] == "model1_fill_in_the_blank_1"


def test_id_survives_repair():
    exam = _exam({"true_false": [_tf("model1_true_false_1", answer="False")]})
    fake = FakeClient(
        repair_result=[
            {"question_id": "model1_true_false_1", "repaired_fields": ["answer"],
             "question": {"statement": "Cyanobacteria released oxygen through photosynthesis.", "answer": "True"}}
        ]
    )
    verdicts = [_verdict("model1_true_false_1", "FIX_ANSWER", fields=["answer"],
                         reason="should be True")]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["true_false"][0]
    assert q["question_id"] == "model1_true_false_1"
    assert q["answer"] == "True"


def test_display_numbering_independent_of_id():
    questions = {
        "true_false": [_tf("model1_true_false_1")],
        "mcq": [_mcq("model1_mcq_1")],
    }
    _ = eb.assemble_exam(questions)
    assert questions["true_false"][0]["question_id"] == "model1_true_false_1"
    assert questions["mcq"][0]["question_id"] == "model1_mcq_1"


# ---------------------------------------------------------------------------
# Validation: objective types
# ---------------------------------------------------------------------------
def test_valid_mcq_remains_unchanged():
    q = _mcq("model1_mcq_1")
    exam = _exam({"mcq": [q]})
    fake = FakeClient(validator_result=[_pass_verdict("model1_mcq_1")])
    report = validate_exam(exam, client=fake)
    assert report["all_pass"] is True
    assert zero_repair(fake)  # no repair call needed
    assert exam["questions"]["mcq"][0] is q  # same object, untouched


def test_wrong_mcq_answer_repaired_answer_only():
    exam = _exam({"mcq": [_mcq("model1_mcq_2", answer="B")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_mcq_2", "FIX_ANSWER", av=False, fields=["answer"])],
        repair_result=[
            {"question_id": "model1_mcq_2", "repaired_fields": ["answer"],
             "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                          "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
                          "correct_answer": "C"}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["mcq"][0]
    assert q["question_id"] == "model1_mcq_2"
    assert q["correct_answer"] == "C"
    assert q["question"] == "Which metric is the harmonic mean of precision and recall?"
    assert q["options"] == {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"}


def test_mcq_missing_correct_option_detected_locally():
    bad = _mcq("model1_mcq_3", answer="E")
    exam = _exam({"mcq": [bad]})
    report = validate_exam(exam, client=FakeClient())
    v = report["verdicts"][0]
    assert v["action"] == "FIX_OPTIONS"
    assert set(v["fields_to_fix"]) & {"options", "answer"}


def test_mcq_duplicate_option_detected_locally():
    bad = _mcq("model1_mcq_4", options={"A": "same", "B": "same", "C": "c", "D": "d"})
    exam = _exam({"mcq": [bad]})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_OPTIONS"


def test_mcq_two_valid_options_repairs_options():
    exam = _exam({"mcq": [_mcq("model1_mcq_5", answer="B")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_mcq_5", "FIX_OPTIONS", av=False,
                                   fields=["options", "answer"])],
        repair_result=[
            {"question_id": "model1_mcq_5", "repaired_fields": ["options", "answer"],
             "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                          "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                          "correct_answer": "A"}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["mcq"][0]
    assert q["options"] == {"A": "a", "B": "b", "C": "c", "D": "d"}
    assert q["correct_answer"] == "A"


def test_valid_tf_unchanged():
    q = _tf("model1_true_false_1")
    exam = _exam({"true_false": [q]})
    fake = FakeClient(validator_result=[_pass_verdict("model1_true_false_1")])
    report = validate_exam(exam, client=fake)
    assert report["all_pass"]
    assert exam["questions"]["true_false"][0] is q


def test_tf_wrong_answer_repaired_answer_only():
    exam = _exam({"true_false": [_tf("model1_true_false_2", answer="False")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_true_false_2", "FIX_ANSWER", av=False, fields=["answer"])],
        repair_result=[
            {"question_id": "model1_true_false_2", "repaired_fields": ["answer"],
             "question": {"statement": "Cyanobacteria released oxygen through photosynthesis.", "answer": "True"}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["true_false"][0]
    assert q["answer"] == "True"
    assert q["statement"] == "Cyanobacteria released oxygen through photosynthesis."
    assert q["question_id"] == "model1_true_false_2"


def test_malformed_tf_statement_repaired():
    exam = _exam({"true_false": [{"question_id": "model1_true_false_3", "statement": "", "answer": "True"}]})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_QUESTION"


def test_fitb_wrong_answer_repaired():
    exam = _exam({"fill_in_the_blank": _fitb([_fitb_item("model1_fill_in_the_blank_1", answers=["perimeter"])])})
    fake = FakeClient(
        validator_result=[_verdict("model1_fill_in_the_blank_1", "FIX_ANSWER", av=False, fields=["answer"])],
        repair_result=[
            {"question_id": "model1_fill_in_the_blank_1", "repaired_fields": ["answer"],
             "question": {"question": "To calculate the ________ of a rectangle, multiply length by width.",
                          "answers": ["area"]}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    items = exam["questions"]["fill_in_the_blank"]["items"]
    assert items[0]["answers"] == ["area"]
    assert len(items) == 1


def test_fitb_answer_missing_from_bank_detected_locally():
    exam = _exam({"fill_in_the_blank": _fitb([_fitb_item("model1_fill_in_the_blank_2", answers=["nowhere"])])})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_OPTIONS"


def test_fitb_ambiguous_bank_repair_scoped():
    item1 = _fitb_item("model1_fill_in_the_blank_3", answers=["area"])
    item2 = _fitb_item("model1_fill_in_the_blank_4", answers=["perimeter"])
    exam = _exam({"fill_in_the_blank": _fitb([item1, item2])})
    fake = FakeClient(
        validator_result=[
            _verdict("model1_fill_in_the_blank_3", "FIX_OPTIONS", av=False, fields=["word_bank", "answer"]),
            _pass_verdict("model1_fill_in_the_blank_4"),
        ],
        repair_result=[
            {"question_id": "model1_fill_in_the_blank_3", "repaired_fields": ["word_bank", "answer"],
             "question": {"question": "To calculate the ________ of a rectangle, multiply length by width.",
                          "answers": ["area"], "word_bank": ["area", "perimeter", "length", "width", "square"]}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    section = exam["questions"]["fill_in_the_blank"]
    assert section["items"][0]["answers"] == ["area"]  # item repaired
    assert section["items"][1] is item2  # unrelated item untouched
    assert section["items"][1]["answers"] == ["perimeter"]


# ---------------------------------------------------------------------------
# Validation: writing types
# ---------------------------------------------------------------------------
def test_short_answer_irrelevant_reference_repaired_answer_only():
    exam = _exam({"short_answer": [_sa("model1_short_answer_1")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_short_answer_1", "FIX_ANSWER", av=False, fields=["answer"])],
        repair_result=[
            {"question_id": "model1_short_answer_1", "repaired_fields": ["answer"],
             "question": {"question": "Why is peer review important?",
                          "reference_answer": "Peer review lets experts catch errors."}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["short_answer"][0]
    assert q["reference_answer"] == "Peer review lets experts catch errors."
    assert q["question"] == "Why is peer review important?"


def test_essay_unrelated_answer_repaired():
    exam = _exam({"essay": [_essay("model1_essay_1")]})
    fake = FakeClient(
        validator_result=[_verdict("model1_essay_1", "FIX_ANSWER", av=False, fields=["answer"])],
        repair_result=[
            {"question_id": "model1_essay_1", "repaired_fields": ["answer"],
             "question": {"question": "Compare precision and recall.",
                          "reference_answer": "A fixed answer.", "key_points": ["a", "b"]}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    assert exam["questions"]["essay"][0]["reference_answer"] == "A fixed answer."


def test_essay_repair_keypoints_coupled_to_answer():
    exam = _exam({"essay": [_essay("model1_essay_1",
                                   ref="A wrong answer.",
                                   key_points=["old", "stale"])]})
    fake = FakeClient(
        validator_result=[_verdict("model1_essay_1", "FIX_ANSWER", av=False,
                                   fields=["reference_answer"])],
        repair_result=[
            {"question_id": "model1_essay_1", "repaired_fields": ["reference_answer", "key_points"],
             "question": {"question": "Compare precision and recall.",
                          "reference_answer": "The right answer.",
                          "key_points": ["new", "consistent"]}}
        ],
    )
    verdicts = validate_exam(exam, client=fake)["verdicts"]
    repair_exam(exam, verdicts, client=fake)
    q = exam["questions"]["essay"][0]
    assert q["reference_answer"] == "The right answer."
    assert q["key_points"] == ["new", "consistent"]


def test_valid_writing_question_unchanged():
    q = _sa("model1_short_answer_2")
    exam = _exam({"short_answer": [q]})
    fake = FakeClient(validator_result=[_pass_verdict("model1_short_answer_2")])
    report = validate_exam(exam, client=fake)
    assert report["all_pass"]
    assert exam["questions"]["short_answer"][0] is q


def test_missing_answer_detected_locally():
    exam = _exam({"essay": [{"question_id": "model1_essay_2", "question": "Q?", "reference_answer": ""}]})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_ANSWER"


def test_empty_question_detected_locally():
    exam = _exam({"essay": [{"question_id": "model1_essay_3", "question": "  ", "reference_answer": "ans"}]})
    report = validate_exam(exam, client=FakeClient())
    assert report["verdicts"][0]["action"] == "FIX_QUESTION"


def test_question_answer_mismatch_flagged_by_llm():
    exam = _exam({"short_answer": [_sa("model1_short_answer_3")]})
    fake = FakeClient(validator_result=[_verdict("model1_short_answer_3", "FIX_ANSWER", av=False,
                                                 fields=["answer"], reason="answer belongs to another question")])
    report = validate_exam(exam, client=fake)
    assert report["verdicts"][0]["action"] == "FIX_ANSWER"


# ---------------------------------------------------------------------------
# Missing verdicts: retry once, then remain operationally unvalidated
# ---------------------------------------------------------------------------
def test_missing_verdict_retried_once():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(validator_result=[],
                      validator_retry_result=[_pass_verdict("model1_mcq_1")])
    report = validate_exam(exam, client=fake)
    assert fake.retry_calls == 1
    assert report["all_pass"]
    v = {x["question_id"]: x for x in report["verdicts"]}["model1_mcq_1"]
    assert v["action"] == "PASS"


def test_missing_verdict_never_passed():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(validator_result=[], validator_retry_result=[])
    report = validate_exam(exam, client=fake)
    assert fake.retry_calls == 1
    assert not report["all_pass"]
    v = {x["question_id"]: x for x in report["verdicts"]}["model1_mcq_1"]
    assert v["action"] == "UNVALIDATED"
    assert v["fields_to_fix"] == []
    assert "no validator verdict" in v["reason"].lower()


def test_validator_splits_large_input_into_configured_exact_id_batches(monkeypatch):
    monkeypatch.setattr(val, "VALIDATOR_BATCH_SIZE", 8)
    ids = [f"model1_mcq_{i}" for i in range(1, 35)]
    exam = _exam({"mcq": [_mcq(qid, question=f"Question {qid}?") for qid in ids]})
    fake = EchoValidatorClient()

    report = validate_exam(exam, client=fake)

    assert [len(batch) for batch in fake.primary_batches] == [8, 8, 8, 8, 2]
    assert [qid for batch in fake.primary_batches for qid in batch] == ids
    assert fake.retry_batches == []
    assert [v["question_id"] for v in report["verdicts"]] == ids
    assert all(v["action"] == "PASS" for v in report["verdicts"])


def test_batch_coverage_retries_only_missing_and_tracks_protocol_issues(monkeypatch):
    monkeypatch.setattr(val, "VALIDATOR_BATCH_SIZE", 3)
    ids = ["model1_mcq_1", "model1_mcq_2", "model1_mcq_3"]
    exam = _exam({"mcq": [_mcq(qid, question=f"Question {qid}?") for qid in ids]})

    class PartialClient:
        def __init__(self):
            self.retry_ids = []

        def chat_json(self, prompt, system_prompt=None, **kwargs):
            sent = [entry["question_id"] for entry in _validator_entries(prompt)]
            if system_prompt == VALIDATOR_SYSTEM_PROMPT:
                first = _pass_verdict(sent[0])
                return [first, dict(first), _pass_verdict("unexpected_id")]
            self.retry_ids.append(sent)
            return [_pass_verdict(sent[0])]

    fake = PartialClient()
    report = validate_exam(exam, client=fake)
    actions = {v["question_id"]: v["action"] for v in report["verdicts"]}

    assert fake.retry_ids == [["model1_mcq_2", "model1_mcq_3"]]
    assert actions == {
        "model1_mcq_1": "PASS",
        "model1_mcq_2": "PASS",
        "model1_mcq_3": "UNVALIDATED",
    }
    assert report["coverage"] == [
        {
            "sent_ids": ids,
            "returned_ids": [
                "model1_mcq_1",
                "model1_mcq_1",
                "unexpected_id",
                "model1_mcq_2",
            ],
            "matched_ids": ["model1_mcq_1", "model1_mcq_2"],
            "missing_ids": ["model1_mcq_3"],
            "unexpected_ids": ["unexpected_id"],
            "duplicate_returned_ids": ["model1_mcq_1"],
        }
    ]


# ---------------------------------------------------------------------------
# Repair: strict ID-keyed mapping + field enforcement
# ---------------------------------------------------------------------------
def test_repair_batches_only_invalid_questions():
    exam = _exam({
        "mcq": [_mcq("model1_mcq_1", answer="B"), _mcq("model1_mcq_2", answer="B")],
        "true_false": [_tf("model1_true_false_1")],
    })
    fake = FakeClient(repair_result=[])
    verdicts = [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"]),
                _pass_verdict("model1_mcq_2"),
                _pass_verdict("model1_true_false_1")]
    repair_exam(exam, verdicts, client=fake)
    ids = json_ids(fake.last_repair_prompt)
    assert "model1_mcq_1" in ids  # the invalid question IS batched
    assert "model1_mcq_2" not in ids  # valid questions excluded
    assert "model1_true_false_1" not in ids


def test_unvalidated_is_never_sent_to_repair():
    exam = _exam({"mcq": [_mcq("missing"), _mcq("bad")]})
    fake = FakeClient(repair_result=[])
    repair_exam(
        exam,
        [
            _verdict("missing", "UNVALIDATED", qv=False, av=False),
            _verdict("bad", "FIX_ANSWER", av=False, fields=["answer"]),
        ],
        client=fake,
    )
    ids = json_ids(fake.last_repair_prompt)
    assert "missing" not in ids
    assert "bad" in ids


def test_unvalidated_only_report_does_not_enter_repair_node(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("repair_exam must not run for UNVALIDATED")

    monkeypatch.setattr(val, "repair_exam", should_not_run)
    state = {
        "generated_exams": [_exam({"mcq": [_mcq("missing")]})],
        "validation_reports": [
            {
                "model_number": 1,
                "all_pass": False,
                "verdicts": [_verdict("missing", "UNVALIDATED", qv=False, av=False)],
                "warnings": [],
            }
        ],
        "validated_models": [],
        "question_repair_attempts": {},
        "warnings": [],
    }

    out = repair_invalid_questions(state)
    assert out["question_repair_attempts"] == {}
    assert _route_after_validation(state) == "assemble_exams"


def test_real_content_failure_still_routes_to_repair():
    state = {
        "validation_reports": [
            {
                "model_number": 1,
                "all_pass": False,
                "verdicts": [_verdict("bad", "FIX_ANSWER", av=False)],
            }
        ],
        "question_repair_attempts": {},
    }
    assert _route_after_validation(state) == "repair_invalid_questions"


def test_missing_id_in_repair_response_rejected():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(repair_result=[{"repaired_fields": ["answer"],
                                      "question": {"question": "Q", "options": {}, "correct_answer": "A"}}])
    outcome = repair_exam(exam, [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert outcome["failed_ids"] == ["model1_mcq_1"]
    assert exam["questions"]["mcq"][0]["correct_answer"] == "B"  # original kept


def test_unknown_id_in_repair_response_rejected():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model9_mcq_9", "repaired_fields": ["answer"],
         "question": {"question": "Q", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct_answer": "A"}}
    ])
    outcome = repair_exam(exam, [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert "model1_mcq_1" in outcome["failed_ids"]
    assert exam["questions"]["mcq"][0]["correct_answer"] == "B"


def test_duplicate_id_in_repair_response_rejected():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    good = {"question_id": "model1_mcq_1", "repaired_fields": ["answer"],
            "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                         "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
                         "correct_answer": "A"}}
    fake = FakeClient(repair_result=[good, good])
    outcome = repair_exam(exam, [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert "model1_mcq_1" in outcome["repaired_ids"]  # first accepted
    assert not outcome["failed_ids"]


def test_disallowed_field_change_rejected():
    exam = _exam({"mcq": [_mcq("model1_mcq_2", answer="B", question="Original stem")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_2", "repaired_fields": ["answer"],
         "question": {"question": "CHANGED STEM", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                      "correct_answer": "C"}}
    ])
    outcome = repair_exam(exam, [_verdict("model1_mcq_2", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert outcome["failed_ids"] == ["model1_mcq_2"]  # stem touched -> rejected
    q = exam["questions"]["mcq"][0]
    assert q["question"] == "Original stem"
    assert q["correct_answer"] == "B"


def test_noop_repair_rejected():
    opts = {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"}
    exam = _exam({"mcq": [_mcq("model1_mcq_2", answer="B", question="Original stem", options=opts)]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_2", "repaired_fields": ["answer"],
         "question": {"question": "Original stem", "options": dict(opts),
                      "correct_answer": "B"}}  # byte-identical to the original
    ])
    outcome = repair_exam(exam, [_verdict("model1_mcq_2", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert outcome["failed_ids"] == ["model1_mcq_2"]
    assert any("no change" in w for w in outcome["warnings"])
    assert exam["questions"]["mcq"][0]["correct_answer"] == "B"


def test_sanitize_preserves_explicit_fix_options():
    raw = {"question_id": "model1_mcq_2", "question_valid": True, "answer_valid": True,
           "action": "FIX_OPTIONS", "fields_to_fix": ["options", "answer"],
           "reason": "Two options are correct.", "expected_fix": "Rewrite one distractor."}
    v = val._sanitize_verdict(raw, {"model1_mcq_2"})
    assert v["action"] == "FIX_OPTIONS"
    assert v["fields_to_fix"] == ["options", "answer"]


def test_sanitize_rejects_unknown_action_instead_of_turning_it_into_pass():
    raw = {
        "question_id": "model1_mcq_2",
        "question_valid": True,
        "answer_valid": True,
        "action": "NO_ACTION",
        "fields_to_fix": [],
    }
    assert val._sanitize_verdict(raw, {"model1_mcq_2"}) is None


def test_mcq_ambiguous_options_repair_accepted():
    exam = _exam({"mcq": [_mcq("model1_mcq_2",
                               question="Which of the following numbers is prime?",
                               options={"A": "2", "B": "4", "C": "7", "D": "9"},
                               answer="B")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_2", "repaired_fields": ["options", "answer"],
         "question": {"question": "Which of the following numbers is prime?",
                      "options": {"A": "2", "B": "4", "C": "6", "D": "9"},
                      "correct_answer": "A"}}  # exactly one correct option remains
    ])
    verdicts = [_verdict("model1_mcq_2", "FIX_OPTIONS", av=False, fields=["options", "answer"])]
    outcome = repair_exam(exam, verdicts, client=fake)
    assert outcome["repaired_ids"] == ["model1_mcq_2"]
    q = exam["questions"]["mcq"][0]
    assert q["options"]["C"] == "6"
    assert q["correct_answer"] == "A"


def test_mcq_missing_answer_option_repair_accepted():
    exam = _exam({"mcq": [_mcq("model1_mcq_4",
                               question="Which organelle produces most of a cell's ATP?",
                               options={"A": "Nucleus", "B": "Ribosome",
                                        "C": "Golgi apparatus", "D": "Lysosome"},
                               answer="A")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_4", "repaired_fields": ["options", "answer"],
         "question": {"question": "Which organelle produces most of a cell's ATP?",
                      "options": {"A": "Nucleus", "B": "Mitochondria",
                                  "C": "Golgi apparatus", "D": "Lysosome"},
                      "correct_answer": "B"}}  # real answer added to the set
    ])
    verdicts = [_verdict("model1_mcq_4", "FIX_OPTIONS", av=False, fields=["options", "answer"])]
    outcome = repair_exam(exam, verdicts, client=fake)
    assert outcome["repaired_ids"] == ["model1_mcq_4"]
    q = exam["questions"]["mcq"][0]
    assert q["options"]["B"] == "Mitochondria"
    assert q["correct_answer"] == "B"


def test_mcq_answer_verdict_still_allows_option_rewrite():
    exam = _exam({"mcq": [_mcq("model1_mcq_4",
                               question="Which organelle produces most of a cell's ATP?",
                               options={"A": "Nucleus", "B": "Ribosome",
                                        "C": "Golgi apparatus", "D": "Lysosome"},
                               answer="A")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_4", "repaired_fields": ["answer"],
         "question": {"question": "Which organelle produces most of a cell's ATP?",
                      "options": {"A": "Nucleus", "B": "Mitochondria",
                                  "C": "Golgi apparatus", "D": "Lysosome"},
                      "correct_answer": "B"}}  # options+answer coupled, stem untouched
    ])
    verdicts = [_verdict("model1_mcq_4", "FIX_ANSWER", av=False, fields=["answer"])]
    outcome = repair_exam(exam, verdicts, client=fake)
    assert outcome["repaired_ids"] == ["model1_mcq_4"]
    q = exam["questions"]["mcq"][0]
    assert q["options"]["B"] == "Mitochondria"
    assert q["correct_answer"] == "B"
    assert q["question"] == "Which organelle produces most of a cell's ATP?"


def test_repair_normalizes_before_write_back():
    exam = _exam({"true_false": [_tf("model1_true_false_1", answer="False")]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_true_false_1", "repaired_fields": ["answer"],
         "question": {"statement": "Cyanobacteria released oxygen through photosynthesis.", "answer": "true"}}
    ])
    repair_exam(exam, [_verdict("model1_true_false_1", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert exam["questions"]["true_false"][0]["answer"] == "True"  # normalized via normalizer


def test_valid_questions_not_touched_when_sibling_repairs():
    q_valid = _mcq("model1_mcq_3", question="Keep me")
    exam = _exam({"mcq": [_mcq("model1_mcq_2", answer="B"), q_valid]})
    fake = FakeClient(repair_result=[
        {"question_id": "model1_mcq_2", "repaired_fields": ["answer"],
         "question": {"question": "Q2", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                      "correct_answer": "C"}}
    ])
    before = copy.deepcopy(q_valid)
    repair_exam(exam, [_verdict("model1_mcq_2", "FIX_ANSWER", av=False, fields=["answer"]),
                       _pass_verdict("model1_mcq_3")], client=fake)
    assert exam["questions"]["mcq"][1] is q_valid
    assert q_valid == before


def test_repair_failure_keeps_original_for_revalidation():
    exam = _exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})
    fake = FakeClient(repair_result=None)  # unusable array
    outcome = repair_exam(exam, [_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"])], client=fake)
    assert outcome["failed_ids"] == ["model1_mcq_1"]
    assert exam["questions"]["mcq"][0]["correct_answer"] == "B"


# ---------------------------------------------------------------------------
# Graph nodes: revalidation, bounded and per-model attempts, isolation
# ---------------------------------------------------------------------------
def test_repair_flow_revalidates_repaired_model(monkeypatch):
    fake = FakeClient(
        validator_result=[_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"])],
        repair_result=[
            {"question_id": "model1_mcq_1", "repaired_fields": ["answer"],
             "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                          "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
                          "correct_answer": "C"}}
        ],
    )
    validate_calls = {"n": 0}

    def _chat_json(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, **kw):
        if system_prompt == REPAIR_SYSTEM_PROMPT:
            return fake.repair_result
        validate_calls["n"] += 1
        if validate_calls["n"] >= 2:
            return [_pass_verdict("model1_mcq_1")]
        return fake.validator_result

    monkeypatch.setattr(val.LMStudioClient, "chat_json", _chat_json)

    state = {
        "generated_exams": [_exam({"mcq": [_mcq("model1_mcq_1", answer="B")]})],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "warnings": [],
    }
    state = {**state, **validate_generated_questions(state)}
    assert not state["validated_models"]  # invalid -> not validated yet
    state = {**state, **repair_invalid_questions(state)}
    assert state["question_repair_attempts"] == {1: 1}
    state = {**state, **validate_generated_questions(state)}
    assert 1 in state["validated_models"]
    assert state["validation_reports"][0]["all_pass"]


def test_repair_attempts_bounded_per_model(monkeypatch):
    fake = FakeClient(validator_result=[_verdict("model1_tf_1", "FIX_ANSWER", av=False, fields=["answer"])],
                      repair_result=[
                          {"question_id": "model1_tf_1", "repaired_fields": ["answer"],
                           "question": {"statement": "Stmt", "answer": "True"}}
                      ])
    monkeypatch.setattr(val, "LMStudioClient", lambda *a, **k: fake)

    state = {
        "generated_exams": [_exam({"true_false": [_tf("model1_tf_1", answer="False")]})],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {1: MAX_QUESTION_REPAIR_ATTEMPTS},
        "warnings": [],
    }
    out = validate_generated_questions(state)
    assert any("max question repair attempts reached" in w for w in out["warnings"])
    assert 1 in out["validated_models"]  # best-effort: never repaired again


def test_model_repairs_never_mix(monkeypatch):
    q1 = _mcq("model1_mcq_1", answer="B")
    q2 = _mcq("model2_mcq_1", answer="B")
    fake = FakeClient(
        validator_result=[_verdict("model1_mcq_1", "FIX_ANSWER", av=False, fields=["answer"])],
        validator_retry_result=[_pass_verdict("model2_mcq_1")],
        repair_result=[
            {"question_id": "model1_mcq_1", "repaired_fields": ["answer"],
             "question": {"question": "Which metric is the harmonic mean of precision and recall?",
                          "options": {"A": "Accuracy", "B": "F1 Score", "C": "Specificity", "D": "Log Loss"},
                          "correct_answer": "A"}}
        ],
    )
    monkeypatch.setattr(val, "LMStudioClient", lambda *a, **k: fake)
    state = {
        "generated_exams": [_exam({"mcq": [q1]}, model_number=1),
                            _exam({"mcq": [q2]}, model_number=2)],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "warnings": [],
    }
    state = {**state, **validate_generated_questions(state)}
    state = {**state, **repair_invalid_questions(state)}
    assert q1["correct_answer"] == "A"  # model1 repaired
    assert q2["correct_answer"] == "B"  # model2 untouched
    assert q2["question_id"] == "model2_mcq_1"
    assert state["question_repair_attempts"] == {1: 1}


def test_validated_model_not_revalidated(monkeypatch):
    fake = FakeClient(validator_result=[_pass_verdict("model1_mcq_1")])
    monkeypatch.setattr(val, "LMStudioClient", lambda *a, **k: fake)
    state = {
        "generated_exams": [_exam({"mcq": [_mcq("model1_mcq_1")]})],
        "validation_reports": [],
        "validated_models": [],
        "question_repair_attempts": {},
        "warnings": [],
    }
    out = validate_generated_questions(state)
    assert 1 in out["validated_models"]
    assert fake.validate_calls == 1


def test_latest_report_for_already_passed_model_is_preserved(monkeypatch):
    old_report = {
        "model_number": 1,
        "all_pass": True,
        "verdicts": [_pass_verdict("model1_mcq_1")],
        "warnings": [],
    }

    monkeypatch.setattr(
        val,
        "validate_exam",
        lambda exam: {
            "model_number": 2,
            "all_pass": True,
            "verdicts": [_pass_verdict("model2_mcq_1")],
            "warnings": [],
        },
    )
    state = {
        "generated_exams": [
            _exam({"mcq": [_mcq("model1_mcq_1")]}, model_number=1),
            _exam({"mcq": [_mcq("model2_mcq_1")]}, model_number=2),
        ],
        "validation_reports": [old_report],
        "validated_models": [1],
        "question_repair_attempts": {},
        "warnings": [],
    }
    out = validate_generated_questions(state)
    assert {report["model_number"] for report in out["validation_reports"]} == {1, 2}
    assert out["validation_reports"][0] is old_report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def json_ids(prompt):
    """Parse the repair payload JSON embedded in a prompt and return IDs order."""
    import re
    return re.findall(r'"question_id"\s*:\s*"([^"]+)"', prompt)


def zero_repair(fake):
    return fake.repair_calls == 0
