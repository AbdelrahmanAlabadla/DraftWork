"""FITB generation from authoritative catalog-bank entries."""
from __future__ import annotations

import pytest

from app.online import exam_builder as eb


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def chat_json(self, user_prompt, **kwargs):
        self.calls.append(user_prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture()
def fake_llm(monkeypatch):
    def install(responses):
        llm = FakeLLM(responses)
        monkeypatch.setattr("app.llm.factory.create_llm_client", lambda: llm)
        return llm
    return install


def _plans(answers):
    return [
        {
            "slot_id": f"m1_fill_in_the_blank_{index}",
            "fitb_entry_id": f"fitb_{index}",
            "fitb_answer": answer,
            "clue_basis": f"clue for {answer}",
        }
        for index, answer in enumerate(answers, start=1)
    ]


def _generated(stems, distractors):
    return {"items": [
        {
            "slot_id": f"m1_fill_in_the_blank_{index}",
            "stem": stem,
            "distractor_1": distractors[index - 1][0],
            "distractor_2": distractors[index - 1][1],
        }
        for index, stem in enumerate(stems, start=1)
    ]}


def test_autofit_rejects_short_grounded_bank():
    assert eb._autofit_word_bank(["CPU", "RAM", "ROM", "GPU", "cache"], 4) == []


def test_autofit_trims_oversized_bank():
    terms = ["a1", "a2", "a3", "a4"]
    bank = eb._autofit_word_bank([*terms, "d1", "d2", "d3", "junk"], 4)
    assert bank[:4] == terms
    assert set(bank[4:]) == {"d3", "junk"}


def test_generator_returns_only_stem_and_distractors_and_code_assembles_answers(fake_llm):
    answers = ["processor", "memory"]
    distractors = [["controller", "register"], ["cache", "storage"]]
    llm = fake_llm([_generated([
        "The component that executes instructions is the ________.",
        "Temporary working data is held in ________.",
    ], distractors)])

    section, warnings = eb._generate_fitb_type(
        2, _plans(answers), "hardware evidence", "easy", 1, [], []
    )

    assert warnings == []
    assert len(llm.calls) == 1
    assert "Never return or replace the answer" in llm.calls[0]
    assert [item["answers"] for item in section["items"]] == [["processor"], ["memory"]]
    assert [item["fitb_entry_id"] for item in section["items"]] == ["fitb_1", "fitb_2"]
    assert len(section["word_bank"]) == 4
    assert set(section["word_bank"]) == {
        "processor", "memory", "controller", "register"
    }
    assert all(item["distractors"] == ["controller", "register"] for item in section["items"])


def test_one_bad_fitb_slot_does_not_discard_four_valid_slots(fake_llm):
    plans = _plans(["alpha", "beta", "gamma", "delta", "epsilon"])
    valid_stems = [
        "First grounded clue ________.",
        "Second grounded clue ________.",
        "Third grounded clue ________.",
        "Fourth grounded clue ________.",
    ]
    first = _generated(
        [*valid_stems, "Broken numeric clue 123 ________."],
        [["zeta", "eta"] for _ in range(5)],
    )
    bad_retry = {"items": [{
        "slot_id": "m1_fill_in_the_blank_5",
        "stem": "Still broken 456 ________.",
        "distractor_1": "zeta",
        "distractor_2": "eta",
    }]}
    fake_llm([first, bad_retry, bad_retry])

    section, warnings = eb._generate_fitb_type(
        5, plans, "grounded evidence", "easy", 1, [], []
    )

    assert [item["slot_id"] for item in section["items"]] == [
        f"m1_fill_in_the_blank_{index}" for index in range(1, 5)
    ]
    assert len(section["word_bank"]) == 6
    assert any("4/5 generated" in warning for warning in warnings)


@pytest.mark.parametrize("count", [1, 3, 5])
def test_shared_word_bank_is_always_n_plus_two(count):
    answer_terms = ["alpha", "beta", "gamma", "delta", "epsilon"]
    items = [
        {"answers": [answer_terms[index]], "distractors": ["theta", "lambda"]}
        for index in range(count)
    ]

    bank = eb._synchronize_fitb_shared_bank(items)

    assert len(bank) == count + 2
    assert all(item["distractors"] == ["theta", "lambda"] for item in items)


def test_retry_keeps_authoritative_answer_and_retries_only_stem_and_distractors(fake_llm):
    llm = fake_llm([
        _generated(["The processor is 123 ________."], [["controller", "register"]]),
        _generated(["The component that executes instructions is the ________."],
                   [["controller", "register"]]),
    ])

    section, warnings = eb._generate_fitb_type(
        1, _plans(["processor"]), "hardware evidence", "easy", 1, [], []
    )

    assert len(llm.calls) == 2
    assert section["items"][0]["answers"] == ["processor"]
    assert any("invalid_structure" in warning for warning in warnings)


def test_alias_or_duplicate_distractors_are_rejected_then_retried(fake_llm):
    llm = fake_llm([
        _generated(["The component is the ________."], [["Processor", "controller"]]),
        _generated(["The component is the ________."], [["controller", "register"]]),
    ])

    section, warnings = eb._generate_fitb_type(
        1, _plans(["processor"]), "hardware evidence", "easy", 1, [], []
    )

    assert section["items"][0]["distractors"] == ["controller", "register"]
    assert any("invalid distractors" in warning for warning in warnings)


def test_invalid_catalog_answer_never_reaches_generator(fake_llm):
    llm = fake_llm([])
    section, warnings = eb._generate_fitb_type(
        1, _plans(["E = hv"]), "physics evidence", "easy", 1, [], []
    )
    assert section is None
    assert llm.calls == []
    assert any("invalid or duplicate answers" in warning for warning in warnings)
