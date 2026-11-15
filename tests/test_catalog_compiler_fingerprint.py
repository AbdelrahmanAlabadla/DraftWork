from __future__ import annotations

import json
from pathlib import Path

from app import config
from app.online import assessment_catalog as catalog


def _profile(chunk_id: str = "c1") -> dict:
    return {
        "subject": "mathematics", "domain": "STEM", "domains": ["mathematics"],
        "educational_level": "secondary", "assessment_style": "grounded",
        "question_types": {
            qtype: {
                "support": "supported", "task_operations": ["solve"],
                "form_guidance": "grounded", "evidence_chunk_ids": [chunk_id],
            }
            for qtype in (
                "mcq", "fill_in_the_blank", "true_false", "definition", "why",
                "equation", "word_problem", "essay",
            )
        },
        "unsupported_visual_tasks": [],
    }


def _task(chunk_id: str = "c1") -> dict:
    return {
        "knowledge_anchor": "linear equation", "concept": "linear equation",
        "relationship": "ax+b=c", "task_operation": "solve_equation",
        "student_work_product": "solution", "target": "value of x",
        "reasoning_mode": "symbolic_manipulation", "allowed_question_types": ["mcq"],
        "evidence_chunk_ids": [chunk_id], "required_givens": [],
        "allowed_variations": [], "variable_givens": [], "complexity_boundaries": [],
        "exact_source_instance": False, "scenario_family": None,
        "text_answerable": True, "native_difficulty": "easy", "prerequisites": [],
        "allowed_transformations": [],
    }


def test_fingerprint_is_deterministic_and_unrelated_recovery_code_does_not_change_it():
    first = catalog.catalog_compiler_fingerprint()
    second = catalog.catalog_compiler_fingerprint()
    assert first == second
    assert len(first) == 64


def test_compiler_prompt_change_changes_fingerprint(monkeypatch):
    before = catalog.catalog_compiler_fingerprint()
    original = catalog._catalog_prompt

    def changed_prompt(*args, **kwargs):
        return original(*args, **kwargs) + "\nNew compiler instruction."

    monkeypatch.setattr(catalog, "_catalog_prompt", changed_prompt)
    assert catalog.catalog_compiler_fingerprint() != before


def test_compiler_config_change_changes_fingerprint(monkeypatch):
    before = catalog.catalog_compiler_fingerprint()
    monkeypatch.setattr(config, "ASSESSMENT_COMPILER_VERSION", "compiler-v99-test")
    assert catalog.catalog_compiler_fingerprint() != before


def test_compiler_schema_change_changes_fingerprint(monkeypatch):
    before = catalog.catalog_compiler_fingerprint()

    class ChangedVariation(catalog.CompactTaskVariation):
        added_contract_field: str | None = None

    monkeypatch.setattr(catalog, "CompactTaskVariation", ChangedVariation)
    assert catalog.catalog_compiler_fingerprint() != before


def test_same_fingerprint_produces_same_cache_key():
    first = catalog.catalog_cache_key("doc", ["c1"], {"provider": "openai"})
    second = catalog.catalog_cache_key("doc", ["c1"], {"provider": "openai"})
    assert first == second
    assert first != catalog.catalog_cache_key("doc", ["c1"], {"provider": "deepseek"})


def test_legacy_final_catalog_cache_without_fingerprint_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    key = "legacy-key"
    path = Path(tmp_path) / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "cache_key": key,
        "catalog_version": config.ASSESSMENT_CATALOG_VERSION,
        "tasks": [], "fitb_bank": [],
    }), encoding="utf-8")
    assert catalog._read_cache(key) is None


def test_legacy_final_catalog_cannot_be_migrated_without_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    path = Path(tmp_path) / "aa" / "legacy.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "catalog_version": config.ASSESSMENT_CATALOG_VERSION,
        "processed_document_key": "doc",
        "source_document_hash": "doc",
        "selected_chunk_ids": ["c1"],
        "assessment_profile": _profile(),
        "tasks": [_task()],
        "fitb_bank": [],
    }), encoding="utf-8")
    assert catalog._compatible_compiler_seed("doc", ["c1"]) is None


def test_legacy_compiler_batch_cache_without_fingerprint_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    key = "batch-key"
    path = Path(tmp_path) / "compiler" / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "compiler_cache_key": key,
        "raw": {"tasks": [], "fitb_bank": [], "batch_complete": True},
    }), encoding="utf-8")
    assert catalog._read_compiler_cache(key) is None
