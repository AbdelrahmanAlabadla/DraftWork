from __future__ import annotations

import json
import re
from collections import Counter
from types import SimpleNamespace

import pytest

from app.online import assessment_catalog as catalog


def _variation(target: str, evidence: str) -> dict:
    return {
        "target": target, "op": "calculate", "type": "word_problem",
        "givens": ["supported values"], "ev": [evidence],
        "mode": "calculation", "wp": "value", "response": ["worked_solution"],
        "form": None, "context": "grounded context", "vary": True, "graph": False,
    }


def _task(anchor: str, evidence: str, variations: list[dict] | None = None) -> dict:
    return {
        "anchor": anchor, "concept": f"concept {anchor}", "relation": "d = vt",
        "operation": "calculate", "work_product": "value with unit",
        "target": f"target {anchor}", "mode": "calculation",
        "question_types": ["word_problem"], "evidence_ids": [evidence],
        "required_givens": ["constant rate"], "variations": variations or [],
        "variable_givens": ["distance"], "boundaries": ["no acceleration"],
        "exact_source_instance": False, "scenario_family": anchor,
        "answerable": True, "difficulty": "easy", "prereqs": ["arithmetic"],
        "transformations": [],
    }


def _profile(evidence: str) -> dict:
    return {
        "subject": "mathematics", "domain": "STEM", "domains": ["mathematics"],
        "educational_level": "secondary", "assessment_style": "grounded",
        "question_types": {
            qtype: {
                "support": "supported", "task_operations": ["calculate"],
                "form_guidance": "Use source evidence.",
                "evidence_chunk_ids": [evidence],
            }
            for qtype in (
                "mcq", "fill_in_the_blank", "true_false", "definition", "why",
                "equation", "word_problem", "essay",
            )
        },
        "unsupported_visual_tasks": [],
    }


def _prompt_ids(prompt: str) -> tuple[str, ...]:
    match = re.search(r"Valid source chunk IDs: (\[[^\n]+\])", prompt)
    assert match
    return tuple(json.loads(match.group(1)))


def test_parent_batches_are_deterministic_and_oversized_parent_splits_in_order():
    chunks = [
        {
            "child_id": f"a{index}", "parent_id": "parent-a",
            "parent_title": "Arabic section", "child_order": index,
            "content": f"محتوى {index}",
        }
        for index in range(5)
    ] + [
        {
            "child_id": f"b{index}", "parent_id": "parent-b",
            "parent_title": "English section", "child_order": 5 + index,
            "content": f"content {index}",
        }
        for index in range(2)
    ]
    expected = [["a0", "a1"], ["a2", "a3"], ["a4"], ["b0", "b1"]]
    for value in (chunks, list(reversed(chunks))):
        assert [
            [row["child_id"] for row in batch]
            for batch in catalog._balanced_catalog_batches(value, 2)
        ] == expected


def test_interleaved_parent_rows_group_without_false_global_order_failure():
    chunks = [
        {"child_id": "a1", "parent_id": "a", "child_order": 1},
        {"child_id": "b1", "parent_id": "b", "child_order": 2},
        {"child_id": "a2", "parent_id": "a", "child_order": 3},
        {"child_id": "b2", "parent_id": "b", "child_order": 4},
    ]
    batches = catalog._balanced_catalog_batches(chunks, 24)
    assert [
        [row["child_id"] for row in batch] for batch in batches
    ] == [["a1", "a2"], ["b1", "b2"]]


def test_batch_assignment_rejects_duplicate_children_and_isolates_missing_parent():
    with pytest.raises(catalog.CatalogBatchAssignmentError, match="more than once"):
        catalog._balanced_catalog_batches([
            {"child_id": "c1", "parent_id": "p1"},
            {"child_id": "c1", "parent_id": "p1"},
        ])
    batches = catalog._balanced_catalog_batches([
        {"child_id": "c1", "content": "one"},
        {"child_id": "c2", "content": "two"},
    ])
    assert [[row["child_id"] for row in batch] for batch in batches] == [["c1"], ["c2"]]


def test_registry_separates_parent_context_from_allowed_child_evidence():
    registry = catalog._catalog_evidence_registry([{
        "child_id": "child_12", "parent_id": "parent_3",
        "parent_title": "Chapter 3 — Linear Equations",
        "chunk_title": "Substitution", "content": "Solve by substitution.",
    }], "batch_2")
    assert "CATALOG BATCH: batch_2" in registry
    assert "Evidence ID: child_12" in registry
    assert "Parent title: Chapter 3 — Linear Equations" in registry
    assert "Parent IDs are context only" in registry


def test_evidence_diagnostics_preserve_valid_task_and_sibling_variation():
    response = {
        "tasks": [
            _task("valid", "c1"),
            _task("invalid-task", "outside"),
            _task("mixed", "c1", [
                _variation("valid variation", "c1"),
                _variation("invalid variation", "outside"),
            ]),
        ],
        "fitb_bank": [], "batch_complete": True,
    }
    with pytest.raises(catalog.CatalogBatchValidationError) as caught:
        catalog._validate_completed_batch(
            response, response_model=catalog.CompactCatalogBatchResponse,
            valid_chunk_ids={"c1", "c2"}, batch_id="batch_2",
        )
    diagnostics = caught.value.diagnostics
    assert {item["invalid_evidence_id"] for item in diagnostics} == {"outside"}
    assert {item["field"] for item in diagnostics} == {
        "task.evidence_chunk_ids", "variation.evidence_chunk_ids",
    }
    assert all(item["batch_id"] == "batch_2" for item in diagnostics)
    assert all(item["allowed_evidence_ids"] == ["c1", "c2"] for item in diagnostics)
    preserved = caught.value.preserved["tasks"]
    assert {item["knowledge_anchor"] for item in preserved} == {"valid", "mixed"}
    mixed = next(item for item in preserved if item["knowledge_anchor"] == "mixed")
    assert [item["target"] for item in mixed["allowed_variations"]] == ["valid variation"]


def test_profile_evidence_is_bound_to_the_first_compiler_batch():
    response = {
        "assessment_profile": _profile("outside"),
        "tasks": [_task("valid", "c1")],
        "fitb_bank": [], "batch_complete": True,
    }
    with pytest.raises(catalog.CatalogBatchValidationError) as caught:
        catalog._validate_completed_batch(
            response, response_model=catalog.CompactCatalogBatchWithProfileResponse,
            valid_chunk_ids={"c1"}, batch_id="batch_1",
        )
    assert "assessment_profile" not in caught.value.preserved
    assert all(
        item["field"].startswith("assessment_profile.question_types.")
        for item in caught.value.diagnostics
    )
    assert {item["invalid_evidence_id"] for item in caught.value.diagnostics} == {
        "outside"
    }


def test_job_3787_failure_pattern_gets_targeted_feedback_and_partial_preservation(
    monkeypatch, tmp_path,
):
    chunks = [
        {"child_id": "c1", "parent_id": "p1", "child_order": 0, "content": "one"},
        {"child_id": "c2", "parent_id": "p1", "child_order": 1, "content": "two"},
        {"child_id": "c3", "parent_id": "p2", "child_order": 2, "content": "three"},
        {"child_id": "c4", "parent_id": "p2", "child_order": 3, "content": "four"},
    ]
    calls = Counter()
    retry_prompt = ""

    def responder(_client, prompt, **kwargs):
        nonlocal retry_prompt
        ids = _prompt_ids(prompt)
        calls[ids] += 1
        value = {"tasks": [], "fitb_bank": [], "batch_complete": True}
        if "assessment_profile" in kwargs["response_model"].model_fields:
            value["assessment_profile"] = _profile(ids[0])
        if ids == ("c1", "c2"):
            value["tasks"] = [_task("first-parent", "c1")]
            return value
        if calls[ids] == 1:
            value["tasks"] = [
                _task("preserved", "c3"),
                _task("repair-task", "c1"),
                _task("repair-variation", "c4", [
                    _variation("keep sibling", "c4"),
                    _variation("repair sibling", "c2"),
                ]),
            ]
            return value
        retry_prompt = prompt
        value["tasks"] = [
            _task("repair-task", "c3"),
            _task("repair-variation", "c4", [
                _variation("repair sibling", "c4"),
            ]),
        ]
        return value

    monkeypatch.setattr(catalog.config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(catalog.config, "CATALOG_BATCH_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(catalog.config, "CATALOG_BATCH_MAX_CHILDREN", 24)
    monkeypatch.setattr(
        catalog, "create_assessment_compiler_client",
        lambda **_kwargs: SimpleNamespace(last_usage={}),
    )
    monkeypatch.setattr(catalog, "request_structured", responder)
    result = catalog.prepare_assessment_catalog({
        "processed_document_key": "job-3787-regression", "document_id": "doc",
        "retrieved_chunks": chunks, "document_language": "en",
        "domain_type": "stem", "subject": "mathematics",
    })
    assert calls[("c3", "c4")] == 2
    assert '"invalid_evidence_id": "c1"' in retry_prompt
    assert '"invalid_evidence_id": "c2"' in retry_prompt
    assert "Preserved valid compiler content" in retry_prompt
    assert "preserved" in retry_prompt and "keep sibling" in retry_prompt
    compiled = result["assessment_catalog"]
    anchors = {task["knowledge_anchor"] for task in compiled["tasks"]}
    assert {"first-parent", "preserved", "repair-task", "repair-variation"} <= anchors
    repaired = next(
        task for task in compiled["tasks"]
        if task["knowledge_anchor"] == "repair-variation"
    )
    assert {item["target"] for item in repaired["allowed_variations"]} == {
        "keep sibling", "repair sibling",
    }


def test_unrecoverable_cross_batch_reference_fails_explicitly(monkeypatch, tmp_path):
    calls = 0

    def responder(_client, prompt, **kwargs):
        nonlocal calls
        calls += 1
        ids = _prompt_ids(prompt)
        value = {
            "tasks": [_task("bad", "outside")],
            "fitb_bank": [], "batch_complete": True,
        }
        if "assessment_profile" in kwargs["response_model"].model_fields:
            value["assessment_profile"] = _profile(ids[0])
        return value

    monkeypatch.setattr(catalog.config, "ASSESSMENT_CATALOG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(catalog.config, "CATALOG_BATCH_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(
        catalog, "create_assessment_compiler_client",
        lambda **_kwargs: SimpleNamespace(last_usage={}),
    )
    monkeypatch.setattr(catalog, "request_structured", responder)
    with pytest.raises(catalog.CatalogBatchIncompleteError, match="invalid evidence IDs: outside"):
        catalog.prepare_assessment_catalog({
            "processed_document_key": "unrecoverable", "document_id": "doc",
            "retrieved_chunks": [{
                "child_id": "c1", "parent_id": "p1", "content": "source",
            }],
            "document_language": "en", "domain_type": "stem", "subject": "mathematics",
        })
    assert calls == 3
