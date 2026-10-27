from __future__ import annotations

import threading
import time

from app.llm.deepseek import DeepSeekClient
from app.online import exam_builder
from app.online.validator import build_validator_prompt


def test_generation_dispatcher_is_work_conserving_and_capped_at_fifty():
    lock = threading.Lock()
    fifty_started = threading.Event()
    active = 0
    peak = 0
    started = []

    def worker(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            started.append(value)
            if active == 50:
                fifty_started.set()
        assert fifty_started.wait(timeout=5)
        with lock:
            active -= 1
        return value * 2

    result = exam_builder._dispatch_generation_jobs(
        list(range(100)), worker, max_workers=50,
    )
    assert peak == 50
    assert started[:50] == list(range(50))
    assert [result[index] for index in range(100)] == [n * 2 for n in range(100)]


def test_compact_validator_prompt_stores_shared_evidence_once():
    prompt = build_validator_prompt([
        {"question_id": "q1", "question_type": "why", "question": {},
         "task_contract": {"task_id": "t1"},
         "_source_chunks": {"c1": "shared source text"}},
        {"question_id": "q2", "question_type": "definition", "question": {},
         "task_contract": {"task_id": "t2"},
         "_source_chunks": {"c1": "shared source text"}},
    ])
    assert prompt.count("shared source text") == 1
    assert prompt.count('"c1"') >= 3  # registry key plus two explicit boundaries


def test_corrective_retry_is_bounded_and_does_not_repeat_schema():
    prompt = "P" * 100_000
    previous = "O" * 100_000
    result = DeepSeekClient._corrective_input(
        prompt, previous, "bad field", {"schema": "S" * 100_000}, ["q1"],
    )
    assert len(result) < 35_000
    assert "Expected JSON Schema" not in result
    assert "q1" in result
