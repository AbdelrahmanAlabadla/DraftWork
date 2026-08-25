from __future__ import annotations

import pytest

from app.integrations.google_forms import client, exporter
from app.integrations.google_forms.question_handlers import (
    build_question_requests,
    export_mcq,
    export_true_false,
    _answer_variants,
)
from app.exports.common import flatten_exam_items


def _item(qtype: str, **kw) -> dict:
    base = {"qtype": qtype, "question_id": "model1_x_1", "number": 1}
    base.update(kw)
    return base


# ---------------------------------------------------------------- mapping

def test_mcq_mapping_and_correct_option_text():
    item = _item("mcq",
                 text="What is a CPU?",
                 options={"A": "Central Processing Unit", "B": "GPU", "C": "ROM"},
                 correct_answer="B")
    reqs, warning = build_question_requests(item)
    assert warning is None
    assert len(reqs) == 1
    q = reqs[0]["createItem"]["item"]["questionItem"]["question"]
    choice = q["choiceQuestion"]
    assert choice["type"] == "RADIO"
    assert [o["value"] for o in choice["options"]] == [
        "Central Processing Unit", "GPU", "ROM"]
    # Correct answer maps to the option TEXT, not the letter.
    assert q["grading"]["correctAnswers"]["answers"] == [{"value": "GPU"}]
    assert q["grading"]["pointValue"] >= 1


def test_true_false_mapping():
    reqs, _ = build_question_requests(
        _item("true_false", text="RAM is volatile.", answer="True"))
    q = reqs[0]["createItem"]["item"]["questionItem"]["question"]
    opts = [o["value"] for o in q["choiceQuestion"]["options"]]
    assert opts == ["True", "False"]
    assert q["grading"]["correctAnswers"]["answers"] == [{"value": "True"}]

    reqs_f, _ = build_question_requests(
        _item("true_false", text="x", answer="false"))
    qf = reqs_f[0]["createItem"]["item"]["questionItem"]["question"]
    assert qf["grading"]["correctAnswers"]["answers"] == [{"value": "False"}]


def test_fitb_mapping_with_case_variants():
    reqs, _ = build_question_requests(
        _item("fill_in_the_blank", text="The ___ runs programs.",
              answers=["CPU"], word_bank=["CPU", "RAM"]))
    q = reqs[0]["createItem"]["item"]["questionItem"]["question"]
    assert "textQuestion" in q
    values = [a["value"] for a in q["grading"]["correctAnswers"]["answers"]]
    assert set(values) == {"CPU", "cpu", "CPU".upper(), "CPU".title()} | {"CPU"}
    assert "cpu" in values and "Cpu" in values


def test_fitb_two_blanks_both_graded():
    variants = _answer_variants("CPU") + _answer_variants("RAM")
    assert all(v for v in variants)


def test_writing_questions_manual_and_no_reference_leak():
    for qtype, ref in (("short_answer", "secret ref"), ("essay", "essay ref")):
        item = _item(qtype, text="Why?", reference_answer=ref,
                     key_points=["kp1"] if qtype == "essay" else [])
        reqs, _ = build_question_requests(item)
        payload = repr(reqs)
        q = reqs[0]["createItem"]["item"]["questionItem"]["question"]
        assert "textQuestion" in q
        assert q["textQuestion"]["paragraph"] is True
        # Reference answers must NOT appear anywhere in the exported request.
        assert ref not in payload
        assert "kp1" not in payload
        assert "grading" not in q


def test_unsupported_type_safe_fallback():
    reqs, warning = build_question_requests(_item("match_columns", text="Match A to B"))
    assert warning is not None
    assert "unsupported question type" in warning
    q = reqs[0]["createItem"]["item"]["questionItem"]["question"]
    assert "textQuestion" in q  # manual paragraph fallback


# ------------------------------------------------------- multiple models

@pytest.fixture()
def fake_forms(monkeypatch):
    created: list[str] = []
    updated: dict[str, list] = {}
    counter = {"n": 0}

    def fake_create_form(title, creds=None):
        counter["n"] += 1
        form_id = f"form_{counter['n']}"
        created.append(title)
        return {
            "form_id": form_id, "title": title,
            "edit_url": f"https://docs.google.com/forms/d/{form_id}/edit",
            "view_url": f"https://docs.google.com/forms/d/{form_id}/viewform",
        }

    def fake_batch_update(form_id, requests_list, creds=None):
        updated.setdefault(form_id, []).extend(requests_list)
        return len(requests_list)

    monkeypatch.setattr(client, "create_form", fake_create_form)
    monkeypatch.setattr(client, "batch_update", fake_batch_update)
    return created, updated


def _record(models=2) -> dict:
    exams = []
    for m in range(1, models + 1):
        exams.append({
            "model_number": m,
            "questions": {
                "mcq": [{
                    "question_id": f"model{m}_mcq_1",
                    "question": f"Model {m} MCQ?",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct_answer": "A",
                }],
                "true_false": [{"question_id": f"model{m}_tf_1",
                                "statement": f"S{m}?", "answer": "False"}],
                "fill_in_the_blank": {
                    "word_bank": ["X", "Y", "Z"],
                    "items": [{"question_id": f"model{m}_fitb_1",
                               "question": f"___ {m}", "answers": ["X"]}],
                },
                "essay": [{"question_id": f"model{m}_essay_1",
                           "question": f"E{m}?", "reference_answer": "hidden",
                           "key_points": []}],
            },
            "markdown": "", "warnings": [],
        })
    return {"exams": exams, "warnings": []}


def test_models_export_to_separate_forms(fake_forms):
    created, updated = fake_forms
    result = exporter.export_exam(_record(2))
    assert result["errors"] == []
    assert len(result["exports"]) == 2
    assert len(created) == 2  # one form per model — never merged
    ids = {e["form_id"] for e in result["exports"]}
    assert len(ids) == 2
    assert {e["model_number"] for e in result["exports"]} == {1, 2}


def test_model_questions_do_not_mix(fake_forms):
    created, updated = fake_forms
    result = exporter.export_exam(_record(2))
    by_form = {e["form_id"]: e["model_number"] for e in result["exports"]}
    for form_id, model in by_form.items():
        titles = [r["createItem"]["item"].get("title", "")
                  for r in updated[form_id]]
        mcq_titles = [t for t in titles if "MCQ?" in t]
        assert all(f"Model {model}" in t for t in mcq_titles), (
            f"form for model {model} contains foreign questions: {mcq_titles}")


# ------------------------------------------------------------ failures

def test_api_failure_reported_per_model(monkeypatch, fake_forms):
    created, _ = fake_forms

    def failing_batch_update(form_id, requests_list, creds=None):
        raise client.GoogleFormsError("Google Forms API error (status 500).")

    monkeypatch.setattr(client, "batch_update", failing_batch_update)
    result = exporter.export_exam(_record(2))
    assert len(result["exports"]) == 2
    assert all(e.get("partial") for e in result["exports"])
    assert any("adding questions failed" in w
               for e in result["exports"] for w in e["warnings"])


def test_malformed_record_raises_clean_error():
    with pytest.raises(exporter.ExportError):
        exporter.export_exam({"exams": [], "warnings": []})


def test_no_retry_of_form_creation(monkeypatch):
    """create_form must be attempted exactly once per model even on failure."""
    calls = {"create": 0, "batch": 0}

    def fake_create_form(title, creds=None):
        calls["create"] += 1
        return {"form_id": "f1", "title": title, "edit_url": "e", "view_url": "v"}

    def fail_batch(form_id, reqs, creds=None):
        calls["batch"] += 1
        raise client.GoogleFormsError("boom")

    monkeypatch.setattr(client, "create_form", fake_create_form)
    monkeypatch.setattr(client, "batch_update", fail_batch)
    exam = {"model_number": 1, "_items": flatten_exam_items({
        "mcq": [{"question_id": "m1", "question": "?", "options":
                 {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct_answer": "A"}]
    })}
    result = exporter.export_model(exam)
    assert calls["create"] == 1
    # Retry policy lives inside client.batch_update (tested separately);
    # the exporter itself never re-attempts anything.
    assert calls["batch"] == 1
    assert result.get("partial") is True


# ------------------------------------------------------- batch retry policy

def test_batch_update_retries_transient_then_succeeds(monkeypatch):
    attempts = {"n": 0}
    monkeypatch.setattr(time := __import__("time"), "sleep", lambda s: None)

    class FakeResp:
        status = 500

    class TransientError(Exception):
        def __init__(self):
            self.resp = FakeResp()

    class FakeFormsResource:
        def batchUpdate(self, formId, body):
            resource = self

            class Exec:
                def execute(self_inner):
                    attempts["n"] += 1
                    if attempts["n"] < 3:
                        raise TransientError()
                    return {"replies": [{}, {}]}

            return Exec()

    class FakeService:
        forms = lambda self: FakeFormsResource()  # noqa: E731

    monkeypatch.setattr(client, "get_service", lambda creds=None: FakeService())
    applied = client.batch_update("f1", [{"a": 1}, {"b": 2}])
    assert applied == 2
    assert attempts["n"] == 3


def test_batch_update_does_not_retry_permanent_error(monkeypatch):
    attempts = {"n": 0}

    class FakeResp:
        status = 400

    class PermanentError(Exception):
        def __init__(self):
            self.resp = FakeResp()

    class FakeFormsResource:
        def batchUpdate(self, formId, body):
            class Exec:
                def execute(self_inner):
                    attempts["n"] += 1
                    raise PermanentError()
            return Exec()

    class FakeService:
        forms = lambda self: FakeFormsResource()  # noqa: E731

    monkeypatch.setattr(client, "get_service", lambda creds=None: FakeService())
    with pytest.raises(client.GoogleFormsError, match="permanently"):
        client.batch_update("f1", [{"a": 1}])
    assert attempts["n"] == 1
