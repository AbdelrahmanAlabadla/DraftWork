from __future__ import annotations

import json
import re
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from app.llm.deepseek import DeepSeekIncompleteError, DeepSeekTimeoutError
from app.online import assessment_catalog as catalog


def _profile(chunk_id: str) -> dict:
    return {
        "subject": "mathematics", "domain": "STEM", "domains": ["mathematics"],
        "educational_level": "secondary", "assessment_style": "grounded",
        "question_types": {
            qtype: {
                "support": "supported", "task_operations": ["calculate"],
                "form_guidance": "Use source evidence.", "evidence_chunk_ids": [chunk_id],
            }
            for qtype in (
                "mcq", "fill_in_the_blank", "true_false", "definition", "why",
                "equation", "word_problem", "essay",
            )
        },
        "unsupported_visual_tasks": [],
    }


def _task(chunk_id: str, *, arabic: bool = False) -> dict:
    return {
        "anchor": "السرعة الثابتة" if arabic else f"knowledge {chunk_id}",
        "concept": "السرعة الثابتة" if arabic else f"concept {chunk_id}",
        "relation": "d = v * t", "operation": "calculate",
        "work_product": "قيمة مع وحدة" if arabic else "value with unit",
        "target": "المسافة" if arabic else "distance", "mode": "calculation",
        "question_types": ["word_problem"], "evidence_ids": [chunk_id],
        "required_givens": ["v constant"],
        "variations": [{
            "target": "الزمن" if arabic else "time", "operation": "calculate",
            "givens": ["distance", "speed"], "form": None,
            "context": "مسألة حركة" if arabic else "motion problem",
            "vary_numbers": True,
        }],
        "variable_givens": ["distance", "speed", "time"],
        "boundaries": ["دون تسارع" if arabic else "no acceleration"],
        "exact_source_instance": False, "scenario_family": "constant_speed",
        "answerable": True, "difficulty": "easy", "prereqs": ["arithmetic"],
    }


def _state(count: int = 9, *, cancelled=None) -> dict:
    return {
        "processed_document_key": "retry-document", "document_id": "doc-retry",
        "retrieved_chunks": [
            {
                "child_id": f"c{index}", "parent_id": f"p{index // 3}",
                "parent_title": f"section {index // 3}",
                "content": f"source material {index}",
            }
            for index in range(count)
        ],
        "document_language": "ar", "domain_type": "stem", "subject": "mathematics",
        "is_cancelled": cancelled,
    }


def _ids_from_prompt(prompt: str) -> tuple[str, ...]:
    match = re.search(r"Valid source chunk IDs: (\[[^\n]+\])", prompt)
    assert match
    return tuple(json.loads(match.group(1)))


def _complete_response(prompt: str, response_model, *, zero: bool = False) -> dict:
    ids = _ids_from_prompt(prompt)
    value = {
        "tasks": [] if zero else [_task(ids[0], arabic=True)],
        "fitb_bank": [], "batch_complete": True,
    }
    if "assessment_profile" in response_model.model_fields:
        value["assessment_profile"] = _profile(ids[0])
    return value


def _install(monkeypatch, tmp_path, responder):
    monkeypatch.setattr(catalog.config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(catalog.config, "CATALOG_BATCH_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(
        catalog, "create_assessment_compiler_client",
        lambda **_kwargs: SimpleNamespace(last_usage={}),
    )
    monkeypatch.setattr(catalog, "request_structured", responder)


def test_all_batches_complete_without_retry_and_preserve_arabic_variations(monkeypatch, tmp_path):
    calls = Counter()

    def responder(_client, prompt, **kwargs):
        calls[_ids_from_prompt(prompt)] += 1
        return _complete_response(prompt, kwargs["response_model"])

    _install(monkeypatch, tmp_path, responder)
    result = catalog.prepare_assessment_catalog(_state())
    assert sorted(calls.values()) == [1, 1]
    compiled = result["assessment_catalog"]
    assert compiled["catalog_compilation_complete"] is True
    assert len(compiled["tasks"]) == 1
    assert len(compiled["tasks"][0]["evidence_chunk_ids"]) == 2
    assert all(task["allowed_variations"][0]["target"] == "الزمن" for task in compiled["tasks"])


def test_one_hundred_parents_can_fill_fifty_catalog_workers(monkeypatch, tmp_path):
    state = _state(100)
    state["retrieved_chunks"] = [
        {
            "child_id": f"c{index}", "parent_id": f"p{index}",
            "parent_title": f"section {index}", "content": f"source {index}",
        }
        for index in range(100)
    ]
    active = 0
    peak = 0
    lock = threading.Lock()
    all_started = threading.Barrier(50)

    def responder(_client, prompt, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        all_started.wait(timeout=5)
        with lock:
            active -= 1
        return _complete_response(prompt, kwargs["response_model"], zero=True)

    _install(monkeypatch, tmp_path, responder)
    monkeypatch.setattr(catalog.config, "MAX_PARALLEL_CATALOG_CALLS", 50)
    result = catalog.prepare_assessment_catalog(state)
    assert peak == 50
    assert result["assessment_catalog_insufficient_content"] is True


def test_one_incomplete_batch_is_the_only_batch_retried(monkeypatch, tmp_path):
    state = _state()
    batches = catalog._balanced_catalog_batches(state["retrieved_chunks"], 2)
    target = tuple(str(row["child_id"]) for row in batches[1])
    calls = Counter()
    lock = threading.Lock()

    def responder(_client, prompt, **kwargs):
        ids = _ids_from_prompt(prompt)
        with lock:
            calls[ids] += 1
            attempt = calls[ids]
        if ids == target and attempt == 1:
            raise DeepSeekIncompleteError(
                "truncated", partial_output='{"tasks":[', reason="max_output_tokens")
        return _complete_response(prompt, kwargs["response_model"])

    _install(monkeypatch, tmp_path, responder)
    result = catalog.prepare_assessment_catalog(state)
    assert calls[target] == 2
    assert sorted(calls.values()) == [1, 2]
    assert result["assessment_catalog"]["catalog_compilation_complete"] is True


def test_several_incomplete_batches_retry_independently(monkeypatch, tmp_path):
    calls = Counter(); lock = threading.Lock()

    def responder(_client, prompt, **kwargs):
        ids = _ids_from_prompt(prompt)
        with lock:
            calls[ids] += 1; attempt = calls[ids]
        if attempt == 1:
            raise DeepSeekIncompleteError("truncated", partial_output='{"tasks":[', reason="length")
        return _complete_response(prompt, kwargs["response_model"])

    _install(monkeypatch, tmp_path, responder)
    result = catalog.prepare_assessment_catalog(_state(17))
    assert sorted(calls.values()) == [2, 2, 2]
    assert result["assessment_catalog"]["catalog_compilation_complete"] is True


def test_repeated_incomplete_batch_fails_bounded_and_keeps_completed_batch_cache(
    monkeypatch, tmp_path,
):
    state = _state(); batches = catalog._balanced_catalog_batches(state["retrieved_chunks"], 2)
    target = tuple(str(row["child_id"]) for row in batches[1]); calls = Counter(); lock = threading.Lock()

    def failing(_client, prompt, **kwargs):
        ids = _ids_from_prompt(prompt)
        with lock: calls[ids] += 1
        if ids == target:
            raise DeepSeekIncompleteError("truncated", partial_output='{"tasks":[', reason="length")
        return _complete_response(prompt, kwargs["response_model"])

    _install(monkeypatch, tmp_path, failing)
    with pytest.raises(catalog.CatalogBatchIncompleteError, match="batch 2.*3 attempts"):
        catalog.prepare_assessment_catalog(state)
    assert calls[target] == 3
    successful = next(ids for ids in calls if ids != target)
    assert calls[successful] == 1
    cached_after_failure = list(tmp_path.rglob("*.json"))
    assert len(cached_after_failure) == 1
    assert "compiler" in cached_after_failure[0].parts
    # A restart reuses the completed batch file and recompiles only the failed batch.
    def succeeding(_client, prompt, **kwargs):
        calls[_ids_from_prompt(prompt)] += 1
        return _complete_response(prompt, kwargs["response_model"])
    monkeypatch.setattr(catalog, "request_structured", succeeding)
    result = catalog.prepare_assessment_catalog(state)
    assert calls[successful] == 1
    assert calls[target] == 4
    assert result["assessment_catalog"]["catalog_compilation_complete"] is True


def test_catalog_outer_retry_uses_provider_neutral_config(monkeypatch, tmp_path):
    """Changing a DeepSeek SDK retry setting cannot change Catalog recovery."""
    calls = 0

    def responder(_client, prompt, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 2:
            raise DeepSeekIncompleteError(
                "truncated", partial_output='{"tasks":[', reason="length"
            )
        return _complete_response(prompt, kwargs["response_model"])

    _install(monkeypatch, tmp_path, responder)
    monkeypatch.setattr(catalog.config, "CATALOG_BATCH_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(catalog.config, "DEEPSEEK_MAX_STRUCTURED_ATTEMPTS", 1)
    result = catalog.prepare_assessment_catalog(_state(1))
    assert calls == 2
    assert result["assessment_catalog"]["catalog_compilation_complete"] is True


def test_catalog_outer_retry_honors_custom_configured_limit(monkeypatch, tmp_path):
    calls = 0

    def responder(_client, _prompt, **_kwargs):
        nonlocal calls
        calls += 1
        raise DeepSeekIncompleteError(
            "truncated", partial_output='{"tasks":[', reason="length"
        )

    _install(monkeypatch, tmp_path, responder)
    monkeypatch.setattr(catalog.config, "CATALOG_BATCH_MAX_ATTEMPTS", 2)
    with pytest.raises(catalog.CatalogBatchIncompleteError, match="2 attempts"):
        catalog.prepare_assessment_catalog(_state(1))
    assert calls == 2


@pytest.mark.parametrize("marker", [None, False, "true"])
def test_missing_false_or_invalid_completion_marker_is_retried_then_rejected(
    monkeypatch, tmp_path, marker,
):
    calls = 0
    def responder(_client, prompt, **kwargs):
        nonlocal calls; calls += 1
        value = _complete_response(prompt, kwargs["response_model"])
        if marker is None: value.pop("batch_complete")
        else: value["batch_complete"] = marker
        return value
    _install(monkeypatch, tmp_path, responder)
    with pytest.raises(catalog.CatalogBatchIncompleteError):
        catalog.prepare_assessment_catalog(_state(1))
    assert calls == 3
    assert not list(tmp_path.rglob("*.json"))


def test_completed_zero_task_batch_needs_no_retry(monkeypatch, tmp_path):
    calls = 0
    def responder(_client, prompt, **kwargs):
        nonlocal calls; calls += 1
        return _complete_response(prompt, kwargs["response_model"], zero=True)
    _install(monkeypatch, tmp_path, responder)
    result = catalog.prepare_assessment_catalog(_state(1))
    assert calls == 1
    assert result["assessment_catalog_insufficient_content"] is True


def test_cancellation_after_incomplete_response_prevents_retry(monkeypatch, tmp_path):
    cancelled = False; calls = 0
    def is_cancelled(): return cancelled
    def responder(_client, _prompt, **_kwargs):
        nonlocal cancelled, calls; calls += 1; cancelled = True
        raise DeepSeekIncompleteError("truncated", partial_output='{"tasks":[', reason="length")
    _install(monkeypatch, tmp_path, responder)
    with pytest.raises(catalog.CatalogCompilationCancelled):
        catalog.prepare_assessment_catalog(_state(1, cancelled=is_cancelled))
    assert calls == 1


def test_provider_timeout_keeps_existing_transport_semantics(monkeypatch, tmp_path):
    calls = 0
    def responder(*_args, **_kwargs):
        nonlocal calls; calls += 1
        raise DeepSeekTimeoutError("provider timeout")
    _install(monkeypatch, tmp_path, responder)
    with pytest.raises(DeepSeekTimeoutError):
        catalog.prepare_assessment_catalog(_state(1))
    assert calls == 1


def test_planner_guards_incomplete_catalog_and_accepts_complete_catalog():
    incomplete = {"tasks": [], "catalog_version": "v10", "catalog_compilation_complete": False}
    with pytest.raises(RuntimeError, match="incomplete"):
        catalog.allocate_catalog_plan(incomplete, [("word_problem", 1)], 1, "easy")
    complete = {"tasks": [], "catalog_version": "v10", "catalog_compilation_complete": True}
    plans, issues = catalog.allocate_catalog_plan(complete, [("word_problem", 1)], 1, "easy")
    assert plans and issues
