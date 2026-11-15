from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from app.llm import scheduler


@pytest.mark.parametrize("limit", [5, 20, 30])
def test_mocked_validator_scheduler_limit_mapping_and_failure_isolation(
    monkeypatch, limit,
):
    monkeypatch.setattr(scheduler.config, "LLM_JOB_CONCURRENCY_LIMIT", limit)
    scheduler.reset_scheduler_for_tests()
    lock = threading.Lock()
    active = 0
    peak = 0

    def request(index: int):
        nonlocal active, peak
        with scheduler.llm_request_slot(
            agent="task_alignment_validator", operation="mock_validation",
            provider="offline",
        ):
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02 + (index % 5) * 0.001)
                if index in {17, 43}:
                    raise TimeoutError("mock timeout")
                return index, f"q{index}"
            finally:
                with lock:
                    active -= 1

    results = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=65) as executor:
        futures = {executor.submit(request, index): index for index in range(65)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                result_index, question_id = future.result()
                results[result_index] = question_id
            except Exception as exc:  # the coordinator isolates one request failure
                errors[index] = type(exc).__name__

    # Worker threads intentionally have no request context and therefore share
    # the scheduler's documented process-local unscoped gate.
    snapshot = scheduler.scheduler_snapshot(job_id="__unscoped__")
    assert peak == limit
    assert snapshot["peak"] == limit
    assert snapshot["active"] == 0
    assert snapshot["completed"] == 65
    assert all(question_id == f"q{index}" for index, question_id in results.items())
    assert errors == {17: "TimeoutError", 43: "TimeoutError"}


def test_validator_stays_thirty_while_recovery_job_ceiling_is_fifty():
    assert scheduler.config.MAX_PARALLEL_VALIDATOR_CALLS == 30
    assert scheduler.config.LLM_JOB_CONCURRENCY_LIMIT == 50
    assert scheduler.config.MAX_PARALLEL_RECOVERY_DECISION_CALLS == 50
    assert scheduler.config.VALIDATOR_BATCH_SIZE == 4

