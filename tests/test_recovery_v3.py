from __future__ import annotations

import threading
import time

import pytest

from app import config
from app.llm import scheduler
from app.online import recovery_decision, validator


def _contexts(count: int):
    return [{"question_id": f"q{index}", "question": {"question": str(index)}}
            for index in range(count)]


@pytest.mark.parametrize("count,expected", [(1, [1]), (2, [2]), (3, [3]), (4, [3, 1]), (7, [3, 3, 1])])
def test_recovery_decision_batches_are_three(monkeypatch, count, expected):
    def fake_request(_client, _prompt, **kwargs):
        return {"decisions": [{
            "question_id": qid, "decision": "REPAIR", "reason": "localized",
            "fallback_sequence": ["REGENERATE", "SWITCH_TASK"],
            "switch_preference": None,
        } for qid in kwargs["expected_ids"]]}

    monkeypatch.setattr(recovery_decision, "request_structured", fake_request)
    decisions, audit = recovery_decision.decide_recovery(
        _contexts(count), domain="STEM", client_factory=lambda: object(),
    )
    assert len(decisions) == count
    assert audit["batch_sizes"] == expected
    assert all(value["plan"] == ["REPAIR", "REGENERATE", "SWITCH_TASK"]
               for value in decisions.values())


def test_recovery_decision_supports_fifty_real_inflight_calls(monkeypatch):
    lock = threading.Lock()
    current = peak = 0
    all_inflight = threading.Barrier(50, timeout=5)

    def fake_request(_client, _prompt, **kwargs):
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
        all_inflight.wait()
        with lock:
            current -= 1
        return {"decisions": [{
            "question_id": qid, "decision": "REGENERATE", "reason": "broken output",
            "fallback_sequence": ["SWITCH_TASK"], "switch_preference": None,
        } for qid in kwargs["expected_ids"]]}

    monkeypatch.setattr(recovery_decision, "request_structured", fake_request)
    decisions, audit = recovery_decision.decide_recovery(
        _contexts(150), domain="STEM", client_factory=lambda: object(),
    )
    assert len(decisions) == 150
    assert audit["calls"] == 50
    assert peak == 50


def test_job_scheduler_allows_fifty_recovery_requests():
    scheduler.reset_scheduler_for_tests()
    lock = threading.Lock()
    current = peak = 0
    all_inflight = threading.Barrier(50, timeout=5)

    def worker(index: int):
        nonlocal current, peak
        with scheduler.llm_request_slot(
            agent="task_alignment_validator", operation="recovery_decision",
            provider="mock",
        ):
            with lock:
                current += 1
                peak = max(peak, current)
            if index < 50:
                all_inflight.wait()
            time.sleep(0.01)
            with lock:
                current -= 1

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=55) as executor:
        list(executor.map(worker, range(55)))
    snapshot = scheduler.scheduler_snapshot("__unscoped__")
    assert config.LLM_JOB_CONCURRENCY_LIMIT == 50
    assert peak == 50
    assert snapshot["peak"] == 50
    assert snapshot["completed"] == 55
    assert snapshot["active"] == 0
    assert snapshot["waiting"] == 0


def test_invalid_batch_is_persisted_as_resumable_block(monkeypatch):
    monkeypatch.setattr(
        recovery_decision, "request_structured",
        lambda *_args, **_kwargs: {"decisions": []},
    )
    decisions, _audit = recovery_decision.decide_recovery(
        _contexts(2), domain="STEM", client_factory=lambda: object(),
    )
    assert {value["status"] for value in decisions.values()} == {"decision_blocked"}


def test_manifest_merge_preserves_one_time_decision():
    old = [{
        "model_number": 1, "slot_id": "m1_mcq_1", "status": "invalid",
        "history": [], "recovery_v3": {
            "status": "ready", "decision": "REPAIR",
            "plan": ["REPAIR", "REGENERATE", "SWITCH_TASK"], "next_index": 1,
            "decision_calls": 1,
        },
    }]
    fresh = [{"model_number": 1, "slot_id": "m1_mcq_1", "status": "invalid",
              "history": [], "failure_code": "incorrect_answer"}]
    merged = validator._merge_repair_manifest(old, fresh)
    assert merged[0]["recovery_v3"]["next_index"] == 1
    assert merged[0]["recovery_v3"]["decision_calls"] == 1


def test_recovery_eligibility_does_not_select_strategy():
    result = validator.recovery_eligibility(
        {"status": "invalid", "repair_type": "replace_assignment"},
        failures=0, calls=0, completion_attempts=0, completion_enabled=True,
    )
    assert result["eligible"] is True
    assert result["selected_route"] is None


@pytest.mark.parametrize("strategy,action", [
    ("REPAIR", "repair_question"),
    ("SWITCH_VARIATION", "switch_variation"),
    ("REGENERATE", "regenerate_missing_slot"),
    ("SWITCH_TASK", "replace_assignment"),
])
def test_python_router_only_dispatches_saved_agent_choice(strategy, action):
    assert validator.recovery_v3_dispatch_action(strategy) == action


def test_decision_is_not_called_again_after_checkpoint_resume(monkeypatch):
    calls = []

    def decide(contexts, **_kwargs):
        calls.append([row["question_id"] for row in contexts])
        return {"q1": {
            "version": "recovery-v3-task-alignment", "status": "ready",
            "decision": "REPAIR", "reason": "localized",
            "plan": ["REPAIR", "REGENERATE", "SWITCH_TASK"],
            "fallback_sequence": ["REGENERATE", "SWITCH_TASK"],
            "next_index": 0, "attempted_strategies": [], "decision_calls": 1,
        }}, {"calls": 1}

    monkeypatch.setattr(recovery_decision, "decide_recovery", decide)
    state = {
        "assessment_profile": {"domain": "STEM"},
        "generated_exams": [{"model_number": 1, "questions": {"mcq": [{
            "question_id": "q1", "slot_id": "m1_mcq_1", "question": "2+2?",
            "options": {"A": "4", "B": "5", "C": "6", "D": "7"},
            "correct_answer": "A",
        }]}}],
        "plans": [{"model_number": 1, "items": {"mcq": [{
            "slot_id": "m1_mcq_1", "task_id": "t1", "source_chunk_id": "c1",
        }]}}],
        "retrieved_chunks": [{"child_id": "c1", "content": "2+2=4"}],
    }
    manifest = [{
        "model_number": 1, "slot_id": "m1_mcq_1", "question_id": "q1",
        "question_type": "mcq", "status": "invalid", "failure_code": "incorrect_answer",
    }]
    first, _ = validator._ensure_recovery_v3_decisions(state, manifest)
    resumed, audit = validator._ensure_recovery_v3_decisions(state, first)
    assert audit is None
    assert resumed[0]["recovery_v3"]["decision_calls"] == 1
    assert calls == [["q1"]]


def test_recovery_strategy_schema_accepts_variation_and_rejects_invented_strategy(monkeypatch):
    from pydantic import ValidationError
    from app.llm.schemas import RecoveryDecision

    accepted = RecoveryDecision.model_validate({
        "question_id": "q1", "strategy": "SWITCH_VARIATION",
        "reason": "an eligible narrower variation exists",
        "fallback_sequence": ["SWITCH_TASK"],
    })
    assert accepted.strategy == "SWITCH_VARIATION"
    with pytest.raises(ValidationError):
        RecoveryDecision.model_validate({
            "question_id": "q1", "strategy": "MAKE_NEW_QUESTION",
            "reason": "invented", "fallback_sequence": [],
        })

    def fake_request(_client, _prompt, **kwargs):
        return {"decisions": [{
            "question_id": qid, "strategy": "SWITCH_VARIATION",
            "reason": "evidence is variation-specific",
            "fallback_sequence": ["SWITCH_TASK"],
            "switch_preference": None,
        } for qid in kwargs["expected_ids"]]}

    monkeypatch.setattr(recovery_decision, "request_structured", fake_request)
    decisions, _audit = recovery_decision.decide_recovery(
        _contexts(1), domain="STEM", client_factory=lambda: object(),
    )
    assert decisions["q0"]["plan"] == ["SWITCH_VARIATION", "SWITCH_TASK"]


def test_recovery_decision_does_not_invent_fixed_fallbacks(monkeypatch):
    def fake_request(_client, _prompt, **kwargs):
        return {"decisions": [{
            "question_id": qid, "strategy": "REPAIR",
            "reason": "localized repair", "fallback_sequence": [],
            "switch_preference": None,
        } for qid in kwargs["expected_ids"]]}

    monkeypatch.setattr(recovery_decision, "request_structured", fake_request)
    decisions, _audit = recovery_decision.decide_recovery(
        _contexts(1), domain="STEM", client_factory=lambda: object(),
    )
    assert decisions["q0"]["plan"] == ["REPAIR"]
