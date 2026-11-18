from __future__ import annotations

from contextlib import AbstractContextManager

import pytest

from app import config
from app.api import benchmarks


class _CursorContext(AbstractContextManager):
    def __init__(self, cursor):
        self.cursor = cursor

    def __enter__(self):
        return self.cursor

    def __exit__(self, *exc_info):
        return False


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return self.rows.pop(0)


class _FakeConnection(AbstractContextManager):
    def __init__(self, rows):
        self.rows = list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self, **_kwargs):
        return _CursorContext(_FakeCursor(self.rows))


def _config(**overrides):
    cfg = {
        "label": "Physics baseline",
        "document_id": "doc-1",
        "document_filename": "physics.pdf",
        "child_ids": ["c1", "c2"],
        "section_count": 2,
        "tasks": {"mcq": 4, "true_false": 2},
        "num_models": 1,
        "difficulty": "easy",
        "repeats": 2,
        "stop_on_failure": True,
        "questions_per_model": 6,
        "total_questions": 12,
    }
    cfg.update(overrides)
    return cfg


def test_config_fingerprint_ignores_child_order_but_tracks_tasks():
    left = benchmarks.config_fingerprint(_config())
    right = benchmarks.config_fingerprint(_config(child_ids=["c2", "c1"]))
    assert left == right
    changed = benchmarks.config_fingerprint(_config(tasks={"mcq": 5}))
    assert changed != left


def test_validate_config_normalizes_question_counts(monkeypatch):
    class _Document(dict):
        pass

    document = _Document(status="ready", original_filename="physics.pdf")
    monkeypatch.setattr(
        benchmarks.repositories,
        "get_document_for_session",
        lambda *a, **k: document,
    )
    monkeypatch.setattr(
        "app.api.job_routes._validate_child_selection", lambda *a, **k: None
    )
    cfg = benchmarks.validate_config(
        {
            "label": "  Baseline  ",
            "document_id": "doc-1",
            "child_ids": ["c1", "c1", "c2"],
            "tasks": {"mcq": 4, "true_false": 0, "essay": 1},
            "num_models": 2,
            "difficulty": "MEDIUM",
            "repeats": 3,
        },
        session_id="session-1",
        user_id=None,
    )
    assert cfg["label"] == "Baseline"
    assert cfg["child_ids"] == ["c1", "c2"]
    assert cfg["tasks"] == {"mcq": 4, "essay": 1}
    assert cfg["difficulty"] == "medium"
    assert cfg["questions_per_model"] == 5
    assert cfg["total_questions"] == 30
    assert cfg["catalog_reuse"]


def test_validate_config_rejects_invalid_requests(monkeypatch):
    class _Document(dict):
        pass

    document = _Document(status="ready", original_filename="physics.pdf")
    monkeypatch.setattr(
        benchmarks.repositories,
        "get_document_for_session",
        lambda *a, **k: document,
    )
    monkeypatch.setattr(
        "app.api.job_routes._validate_child_selection", lambda *a, **k: None
    )

    base = {
        "document_id": "doc-1",
        "child_ids": ["c1"],
        "tasks": {"mcq": 2},
        "num_models": 1,
        "difficulty": "easy",
        "repeats": 1,
    }
    with pytest.raises(benchmarks.BenchmarkError):
        benchmarks.validate_config(
            {**base, "child_ids": []}, session_id="s", user_id=None
        )
    with pytest.raises(benchmarks.BenchmarkError):
        benchmarks.validate_config(
            {**base, "tasks": {}}, session_id="s", user_id=None
        )
    with pytest.raises(benchmarks.BenchmarkError):
        benchmarks.validate_config(
            {**base, "tasks": {"not_a_type": 1}}, session_id="s", user_id=None
        )
    with pytest.raises(benchmarks.BenchmarkError):
        benchmarks.validate_config(
            {**base, "num_models": config.MAX_MODELS_PER_GENERATION + 1},
            session_id="s",
            user_id=None,
        )
    with pytest.raises(benchmarks.BenchmarkError):
        benchmarks.validate_config(
            {**base, "repeats": config.BENCHMARK_MAX_REPEATS + 1},
            session_id="s",
            user_id=None,
        )
    with pytest.raises(benchmarks.BenchmarkError):
        benchmarks.validate_config(
            {**base, "difficulty": "extreme"}, session_id="s", user_id=None
        )


def test_estimate_scales_historical_per_question_averages(monkeypatch):
    monkeypatch.setattr(
        benchmarks, "validate_config", lambda payload, **kwargs: _config()
    )
    # runs, requested questions, summed duration / calls, input, output, cost
    rows = [
        (5, 100, 500_000),
        (500, 1_000, 200, 1.0),
    ]
    monkeypatch.setattr(
        benchmarks.db, "connection", lambda: _FakeConnection(rows)
    )
    result = benchmarks.estimate({}, session_id="s", user_id=None)
    assert result["available"] is True
    assert result["estimated"] is True
    assert result["basis"]["runs"] == 5
    assert result["basis"]["avg_cost_per_question_usd"] == 0.01
    assert result["estimate"]["total_questions"] == 12
    assert result["estimate"]["estimated_cost_usd"] == 0.12
    assert result["estimate"]["estimated_calls"] == 60.0


def test_estimate_reports_missing_telemetry_instead_of_zero(monkeypatch):
    monkeypatch.setattr(
        benchmarks, "validate_config", lambda payload, **kwargs: _config()
    )
    rows = [
        (0, 0, 0),
        (0, 0, 0, None),
    ]
    monkeypatch.setattr(
        benchmarks.db, "connection", lambda: _FakeConnection(rows)
    )
    result = benchmarks.estimate({}, session_id="s", user_id=None)
    assert result["available"] is False
    assert result["estimate"] is None
    assert "Not enough historical provider telemetry" in result["reason"]


def test_comparability_requires_equivalent_generation_configuration():
    base = {
        "config_fingerprint": "abc",
        "reproducibility": {
            "llm_provider": "deepseek",
            "llm_model": "deepseek-flash",
            "versions": {"assessment_catalog": "v1", "assessment_compiler": "c1"},
        },
    }
    same = {
        "config_fingerprint": "abc",
        "reproducibility": {
            "llm_provider": "deepseek",
            "llm_model": "deepseek-flash",
            "versions": {"assessment_catalog": "v1", "assessment_compiler": "c1"},
        },
    }
    assert benchmarks._comparability_differences(base, same) == []

    different = {
        "config_fingerprint": "xyz",
        "reproducibility": {
            "llm_provider": "openai",
            "llm_model": "gpt-6-luna",
            "versions": {"assessment_catalog": "v2", "assessment_compiler": "c1"},
        },
    }
    differences = benchmarks._comparability_differences(base, different)
    assert "Generation configuration differs" in differences
    assert "LLM provider differs" in differences
    assert "LLM model differs" in differences
    assert "Assessment Catalog version differs" in differences


def test_metric_deltas_are_computed_only_for_recorded_values():
    base = {
        "summary": {
            "quality": {"final_acceptance_rate": 0.8, "requested": 10},
            "llm": {"estimated_cost_usd": 1.0},
        },
    }
    other = {
        "summary": {
            "quality": {"final_acceptance_rate": 0.9, "requested": None},
            "llm": {"estimated_cost_usd": 1.5},
        },
    }
    rows = {
        row["metric"]: row for row in benchmarks._metric_deltas(base, other)
    }
    assert rows["final_acceptance_rate"]["delta"] == pytest.approx(0.1)
    assert rows["estimated_cost_usd"]["delta"] == pytest.approx(0.5)
    assert rows["final_valid"]["delta"] is None


def test_reproducibility_metadata_records_versions_and_commit():
    metadata = benchmarks.reproducibility_metadata()
    assert metadata["llm_provider"] == config.LLM_PROVIDER
    assert metadata["versions"]["assessment_catalog"] == config.ASSESSMENT_CATALOG_VERSION
    assert metadata["versions"]["validation_schema"] == "validator-verdict-v3"
    # The repository is a git checkout in this environment.
    assert metadata["git_commit"] is None or len(metadata["git_commit"]) >= 7

def test_benchmark_llm_usage_coverage_counts_usage_rows():
    row = {
        "calls": 185, "successful": 185, "failed": 0, "retries": 0,
        "usage_calls": 185, "cost_calls": 185, "latency_calls": 185,
        "input_tokens": 931363, "output_tokens": 120542,
        "reasoning_tokens": 67234, "cached_tokens": 265158,
        "cost_usd": 0.148084448, "avg_latency_ms": 9505.3,
        "p95_latency_ms": 19453.8,
    }

    class _Cursor:
        def execute(self, *_args, **_kwargs):
            return None

        def fetchone(self):
            return row

    payload = benchmarks._llm_for_benchmark(_Cursor(), "benchmark-id")
    assert payload["usage_calls"] == 185
    assert payload["cost_calls"] == 185
    assert payload["latency_calls"] == 185
    assert payload["telemetry_available"] is True
    assert payload["input_tokens"] == 931363

def test_git_commit_fallback_reads_repository_files(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    refs = git_dir / "refs" / "heads"
    refs.mkdir(parents=True)
    commit = "a" * 40
    (refs / "main").write_text(commit, encoding="utf-8")
    assert benchmarks._git_commit_from_files(tmp_path) == commit
