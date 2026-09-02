from __future__ import annotations

import time

import pytest

from app.api import exam_store


@pytest.fixture(autouse=True)
def _clean_store():
    exam_store.clear()
    yield
    exam_store.clear()


def _exam(model: int = 1) -> dict:
    return {
        "model_number": model,
        "questions": {
            "mcq": [{"question_id": f"model{model}_mcq_1", "question": f"Q{model}?",
                     "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
                     "correct_answer": "B"}]
        },
        "markdown": "## x",
        "warnings": [],
    }


def test_save_and_get_roundtrip():
    eval_stats = {"overall": {"requested_questions": 1}, "models": {"1": {}}}
    exam_id = exam_store.save_exam(
        [_exam()],
        warnings=["w1"],
        document_id="doc1",
        metadata={"exam_title": "Final Examination", "class_name": "12A"},
        eval_stats=eval_stats,
    )
    assert exam_id.startswith("exam_")
    record = exam_store.get_exam(exam_id)
    assert record["document_id"] == "doc1"
    assert record["warnings"] == ["w1"]
    assert record["metadata"] == {"exam_title": "Final Examination", "class_name": "12A"}
    assert record["eval"] == eval_stats
    assert record["exams"][0]["questions"]["mcq"][0]["correct_answer"] == "B"


def test_unknown_id_raises_clear_error():
    with pytest.raises(exam_store.ExamNotFound, match="Unknown or expired"):
        exam_store.get_exam("exam_nope")


def test_models_stay_isolated():
    id1 = exam_store.save_exam([_exam(1)])
    id2 = exam_store.save_exam([_exam(2)])
    r1, r2 = exam_store.get_exam(id1), exam_store.get_exam(id2)
    assert r1["exams"][0]["model_number"] == 1
    assert r2["exams"][0]["model_number"] == 2
    assert (r1["exams"][0]["questions"]["mcq"][0]["question_id"]
            != r2["exams"][0]["questions"]["mcq"][0]["question_id"])


def test_max_cap_evicts_oldest(monkeypatch):
    monkeypatch.setattr(exam_store, "MAX_STORED_EXAMS", 2)
    ids = [exam_store.save_exam([_exam(m)]) for m in range(3)]
    with pytest.raises(exam_store.ExamNotFound):
        exam_store.get_exam(ids[0])
    exam_store.get_exam(ids[1])
    exam_store.get_exam(ids[2])


def test_ttl_expires(monkeypatch):
    monkeypatch.setattr(exam_store, "EXAM_TTL_SECONDS", 10)
    exam_id = exam_store.save_exam([_exam()])
    # Simulate aging by rewriting created_at directly.
    entry = exam_store._exams[exam_id]
    entry["created_at"] -= 11
    with pytest.raises(exam_store.ExamNotFound):
        exam_store.get_exam(exam_id)


def test_in_memory_note():
    """Documented behavior: store module states it is in-memory."""
    doc = exam_store.__doc__ or ""
    assert "in-memory" in doc.lower()
