from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.online import assessment_catalog as catalog


def _profile(chunk_id: str) -> dict:
    question_types = {
        qtype: {"support": "supported", "task_operations": ["solve_equation"],
                "form_guidance": "Use source evidence.", "evidence_chunk_ids": [chunk_id]}
        for qtype in ("mcq", "fill_in_the_blank", "true_false", "definition", "why",
                      "equation", "word_problem", "essay")
    }
    return {"subject": "mathematics", "domain": "STEM", "domains": ["mathematics"],
            "educational_level": "secondary", "assessment_style": "grounded",
            "question_types": question_types, "unsupported_visual_tasks": []}


def _variation(**overrides) -> dict:
    value = {"target": "value of x", "operation": "solve_equation",
             "givens": ["linear equation"], "representation": "symbolic equation",
             "application": None, "numeric_values_may_change": True}
    value.update(overrides)
    return catalog._normalize_variations([value])[0]


def _compact_task(*, aliases: bool = False, **overrides) -> dict:
    task = {
        "knowledge_anchor": "linear equations", "concept": "linear equality",
        "relationship": "equal operations preserve equality", "task_operation": "solve_equation",
        "student_work_product": "solution value", "target": "value of x",
        "reasoning_mode": "symbolic_manipulation", "allowed_question_types": ["equation"],
        "evidence_chunk_ids": ["c1"], "required_givens": [],
        "allowed_variations": [_variation()], "variable_givens": ["coefficients"],
        "complexity_boundaries": ["linear only"], "exact_source_instance": False,
        "scenario_family": None, "text_answerable": True, "native_difficulty": "medium",
        "prerequisites": ["equality"],
    }
    task.update(overrides)
    if not aliases:
        return task
    alias_map = {"knowledge_anchor": "anchor", "relationship": "relation",
                 "task_operation": "operation", "student_work_product": "work_product",
                 "reasoning_mode": "mode", "allowed_question_types": "question_types",
                 "evidence_chunk_ids": "evidence_ids", "allowed_variations": "variations",
                 "complexity_boundaries": "boundaries", "text_answerable": "answerable",
                 "native_difficulty": "difficulty", "prerequisites": "prereqs"}
    return {alias_map.get(key, key): value for key, value in task.items()}


def _finalize(tasks: list[dict]) -> dict:
    return catalog._finalize_catalog(
        {"assessment_profile": _profile("c1"), "tasks": tasks, "fitb_bank": []},
        key="catalog", processed_document_key="processed", selected_chunk_ids=["c1"], usage=None)


def test_final_task_schema_has_exact_18_semantic_fields():
    fields = set(catalog.CatalogTaskDraft.model_fields)
    assert len(fields) == 18
    assert {"text_answerable", "native_difficulty", "allowed_variations"} <= fields
    assert not {"topic", "reasoning_depth", "reasoning_requirements"} & fields


def test_compact_schema_requires_explicit_answerability_and_rejects_invalid_values():
    catalog.CompactCatalogTask.model_validate(_compact_task())
    for value in (None, "true", 1):
        task = _compact_task()
        if value is None:
            task.pop("text_answerable")
        else:
            task["text_answerable"] = value
        with pytest.raises(ValidationError):
            catalog.CompactCatalogTask.model_validate(task)


def test_answerability_true_is_kept_and_false_or_missing_is_filtered():
    assert len(_finalize([_compact_task()])["tasks"]) == 1
    assert _finalize([_compact_task(text_answerable=False)])["tasks"] == []
    missing = _compact_task(); missing.pop("text_answerable")
    assert _finalize([missing])["tasks"] == []


def test_aliases_normalize_to_internal_contract():
    normalized = catalog.CompactCatalogTask.model_validate(_compact_task(aliases=True)).model_dump()
    assert normalized["knowledge_anchor"] == "linear equations"
    assert normalized["text_answerable"] is True and "anchor" not in normalized
    expanded = catalog._expand_compact_task(normalized)
    assert not {"topic", "reasoning_depth", "reasoning_requirements"} & expanded.keys()


def test_variations_are_precise_and_broad_legacy_codes_are_not_authorized():
    assert catalog._expand_compact_task(_compact_task())["allowed_variations"] == [_variation()]
    legacy = catalog._expand_compact_task(
        _compact_task(allowed_variations=["different_target", "different_operation"]))
    assert legacy["allowed_variations"] == []


def test_variation_requires_target_operation_and_valid_boolean():
    for invalid in ({"operation": "calculate", "givens": []},
                    {"target": "distance", "givens": []},
                    {"target": "distance", "operation": "calculate", "givens": [],
                     "numeric_values_may_change": "yes"}):
        with pytest.raises(ValidationError):
            catalog.CompactCatalogTask.model_validate(
                _compact_task(allowed_variations=[invalid]))


def test_finalize_emits_18_field_contract_plus_derived_identity_and_support():
    task = _finalize([_compact_task()])["tasks"][0]
    assert task["text_answerable"] is True
    assert task["allowed_variations"][0]["target"] == "value of x"
    assert task["allowed_variations"][0]["variation_group_id"].startswith("variation_group_")
    assert task["allowed_variations"][0]["variation_id"].startswith("variation_")
    assert task["supported_difficulties"] == ["medium"]
    assert not {"topic", "reasoning_depth", "reasoning_requirements"} & task.keys()
    assert all(task.get(field) for field in
               ("task_id", "task_family_id", "coverage_id", "concept_id", "relationship_id"))


def test_exact_source_instance_cannot_authorize_variable_instances():
    task = _finalize([_compact_task(exact_source_instance=True)])["tasks"][0]
    assert task["exact_source_instance"] is True
    assert task["allowed_variations"] == []


def test_compact_task_flows_through_planner_with_derived_plan_reasoning():
    finalized = _finalize([_compact_task()])
    plans, issues = catalog.allocate_catalog_plan(finalized, [("equation", 1)], 1, "medium")
    assert issues == []
    item = plans[0]["items"]["equation"][0]
    assert item["topic"] == "linear equality"
    assert item["reasoning_depth"] == "medium"
    assert item["reasoning_requirements"] == ["solve_equation: value of x"]
    assert item["allowed_variations"] == finalized["tasks"][0]["allowed_variations"]


def test_batch_schemas_accept_explicit_completed_zero_task_batches():
    value = catalog.CompactCatalogBatchResponse.model_validate(
        {"tasks": [], "fitb_bank": [], "batch_complete": True})
    assert value.tasks == [] and value.batch_complete is True
    with pytest.raises(ValidationError):
        catalog.CompactCatalogBatchResponse.model_validate({"tasks": [], "fitb_bank": []})


def test_merge_tracks_complete_and_recovered_batches():
    merged = catalog._merge_catalog_batches([
        {"assessment_profile": _profile("c1"), "tasks": [], "fitb_bank": [],
         "batch_complete": True},
        {"tasks": [{"b": 2}], "fitb_bank": [], "batch_complete": False,
         "recovered_truncated_checkpoint": True}])
    assert merged["tasks"] == [{"b": 2}]
    assert merged["catalog_compilation_complete"] is False
    assert merged["recovered_batch_count"] == 1


def test_prompt_requests_precise_variations_concision_and_completion_marker():
    prompt = catalog._catalog_prompt(
        "source", "English", ["c1"], domain_type="stem", subject="mathematics")
    assert "Never emit broad labels such as different_target" in prompt
    assert "Use information-dense values" in prompt
    assert "batch_complete: true" in prompt
    assert "A batch may legitimately return empty tasks" in prompt


def test_profile_is_only_required_by_first_batch():
    first = catalog.CompactCatalogBatchWithProfileResponse.model_fields
    later = catalog.CompactCatalogBatchResponse.model_fields
    assert "assessment_profile" in first and "assessment_profile" not in later


def test_compact_alias_serialization_is_smaller_than_internal_names():
    model = catalog.CompactCatalogTask.model_validate(_compact_task())
    assert len(json.dumps(model.model_dump(by_alias=True), ensure_ascii=False)) < len(
        json.dumps(model.model_dump(), ensure_ascii=False))


def test_cache_keys_are_stable_and_contract_sensitive(monkeypatch):
    identity = {"provider": "openai", "model": "x", "prompt_version": "p"}
    baseline = catalog.compiler_cache_key("processed", ["c1"], identity)
    assert baseline == catalog.compiler_cache_key("processed", ["c1"], identity)
    monkeypatch.setattr(catalog, "CATALOG_CONTRACT_VERSION", "next")
    assert catalog.compiler_cache_key("processed", ["c1"], identity) != baseline


@pytest.mark.parametrize("count", [1, 3, 12, 25])
def test_checkpoint_recovery_accepts_any_number_of_complete_valid_tasks(count):
    objects = []
    for index in range(count):
        task = _compact_task(aliases=True)
        task["target"] = f"value of x {index}"
        objects.append(json.dumps(task, ensure_ascii=False))
    recovered = catalog._recover_catalog_checkpoint(
        '{"tasks":[' + ",".join(objects) + ",", require_profile=False)
    assert recovered is not None
    assert len(recovered["tasks"]) == count
    assert recovered["batch_complete"] is False
    assert recovered["recovered_truncated_checkpoint"] is True


def test_checkpoint_recovery_rejects_empty_or_only_invalid_partial_output():
    assert catalog._recover_catalog_checkpoint('{"tasks":[', require_profile=False) is None
    invalid = _compact_task(aliases=True)
    invalid.pop("answerable")
    assert catalog._recover_catalog_checkpoint(
        '{"tasks":[' + json.dumps(invalid) + ",", require_profile=False
    ) is None


def test_legacy_unique_method_constraint_migrates_but_generic_steps_do_not():
    task = _compact_task(
        reasoning_requirements=[
            "Use elimination, not matrix inversion",
            "Substitute values",
        ]
    )
    expanded = catalog._expand_compact_task(task)
    assert "Use elimination, not matrix inversion" in expanded["complexity_boundaries"]
    assert "Substitute values" not in expanded["complexity_boundaries"]
    assert "reasoning_requirements" not in expanded


def test_arabic_source_specific_metadata_survives_alias_normalization():
    task = _compact_task(
        aliases=True,
        concept="السرعة الثابتة",
        target="المسافة",
        boundaries=["سرعة ثابتة فقط", "دون تسارع"],
        variations=[{
            "target": "الزمن", "operation": "calculate", "givens": ["المسافة", "السرعة"],
            "representation": None, "application": "مسألة حركة",
            "numeric_values_may_change": True,
        }],
    )
    normalized = catalog.CompactCatalogTask.model_validate(task).model_dump()
    assert normalized["concept"] == "السرعة الثابتة"
    assert normalized["complexity_boundaries"] == ["سرعة ثابتة فقط", "دون تسارع"]
    assert normalized["allowed_variations"][0]["target"] == "الزمن"
