from __future__ import annotations

import pytest

from app.online import assessment_catalog, exam_builder, planner
from app.llm.deepseek import DeepSeekIncompleteError


def _item(
    chunk: str,
    idea: str,
    *,
    relationship: str | None = None,
    target: str | None = None,
    task_operation: str = "apply_concept",
    reasoning_mode: str = "conceptual_reasoning",
    difficulty: str = "medium",
) -> dict:
    resolved_target = target or idea
    return {
        "source_chunk_id": chunk,
        "support_chunk_ids": [],
        "topic": "Machine learning",
        "concept": "Supervised learning",
        "canonical_skill": idea,
        "knowledge_focus": relationship or resolved_target,
        "student_work_product": f"{task_operation}: {resolved_target}",
        "relationship": relationship,
        "target": resolved_target,
        "task_operation": task_operation,
        "given_conditions": [],
        "scenario_family": None,
        "required_givens": [],
        "text_answerable": True,
        "reasoning_mode": reasoning_mode,
        "difficulty": difficulty,
        "reasoning_depth": difficulty,
        "reasoning_requirements": [f"{task_operation}: {resolved_target}"],
        "prerequisites": [],
        "idea_to_test": idea,
        "concept_to_test": idea,
    }


def _profile(chunk: str = "chunk-a") -> dict:
    return {
        "domains": ["science"],
        "educational_level": "secondary",
        "assessment_style": "Source-grounded mixed assessment",
        "question_types": {
            qtype: {
                "support": "supported",
                "task_operations": ["apply_concept"],
                "form_guidance": f"Use source-supported {qtype} form.",
                "evidence_chunk_ids": [chunk],
            }
            for qtype in planner.VALID_TYPES
        },
        "unsupported_visual_tasks": [],
    }


def test_plan_schema_enforces_exact_counts_and_selected_chunks():
    schema = planner.build_plan_schema(
        [("mcq", 3), ("essay", 1)], 2, ["chunk-a", "chunk-b"]
    )
    exams = schema["properties"]["exams"]
    assert "assessment_profile" in schema["properties"]
    assert "planning_issues" in schema["properties"]
    assert exams["minItems"] == exams["maxItems"] == 2
    items = exams["items"]["properties"]["items"]["properties"]
    assert items["mcq"]["maxItems"] == 3
    assert items["essay"]["maxItems"] == 1
    chunk_rule = items["mcq"]["items"]["properties"]["source_chunk_id"]
    assert chunk_rule["enum"] == ["chunk-a", "chunk-b"]
    support_rule = items["mcq"]["items"]["properties"]["support_chunk_ids"]
    assert support_rule["items"]["enum"] == ["chunk-a", "chunk-b"]
    plan_properties = items["mcq"]["items"]["properties"]
    assert {
        "knowledge_focus", "student_work_product", "relationship", "target",
        "task_operation", "reasoning_mode", "difficulty",
    } <= set(plan_properties)
    assert {
        "slot_id", "canonical_skill", "given_conditions", "scenario_family",
        "required_givens", "text_answerable",
    } <= set(plan_properties)
    assert {"string", "null"} == {
        branch["type"] for branch in plan_properties["relationship"]["anyOf"]
    }


def test_plan_schema_supports_new_question_types():
    schema = planner.build_plan_schema(
        [("definition", 1), ("equation", 2), ("word_problem", 1)],
        1,
        ["chunk-a"],
    )
    items = schema["properties"]["exams"]["items"]["properties"]["items"]["properties"]
    assert set(items) == {"definition", "equation", "word_problem"}
    assert items["definition"]["maxItems"] == 1
    assert items["equation"]["maxItems"] == 2


def test_finalize_plans_rebuilds_missing_canonical_task_ids():
    item = _item("chunk-a", "apply conservation of energy", target="final energy")
    item["slot_id"] = "m1_word_problem_1"

    plans = planner.finalize_plans(
        [{"model_number": 1, "items": {"word_problem": [item]}}],
        [("word_problem", 1)],
    )

    finalized = plans[0]["items"]["word_problem"][0]
    assert {
        field: finalized.get(field)
        for field in assessment_catalog.CANONICAL_TASK_ID_FIELDS
    } == assessment_catalog.canonical_task_ids(finalized)
    assert finalized["task_id"]
    assert finalized["task_family_id"]
    assert finalized["coverage_id"]
    assert finalized["concept_id"]


def test_incomplete_identity_regenerates_only_affected_plan_slot(monkeypatch):
    good = _item("chunk-a", "identify training data", target="training data")
    broken = _item("chunk-a", "identify model output", target="model output")
    good["slot_id"] = "m1_mcq_1"
    broken["slot_id"] = "m1_mcq_2"
    broken["knowledge_focus"] = ""
    plans = planner.finalize_plans(
        [{"model_number": 1, "items": {"mcq": [good, broken]}}],
        [("mcq", 2)],
    )
    good_task_id = plans[0]["items"]["mcq"][0]["task_id"]
    assert plans[0]["items"]["mcq"][1].get("task_id") is None
    errors = planner.validate_plans(
        plans, [("mcq", 2)], 1, {"chunk-a"}, "medium", _profile(), [],
    )
    assert any(
        "m1_mcq_2" in error and "canonical task identity" in error
        for error in errors
    )

    replacement = _item(
        "chunk-a", "identify prediction output", target="prediction output"
    )
    replacement.update({
        "slot_id": "m1_mcq_2",
        "task_id": "stale_task_id",
        "task_family_id": "stale_family_id",
        "coverage_id": "stale_coverage_id",
        "concept_id": "stale_concept_id",
        "relationship_id": "stale_relationship_id",
    })
    requested_slots = []
    monkeypatch.setattr(planner, "create_llm_client", lambda: object())

    def fake_request(*_args, **kwargs):
        requested_slots.extend(kwargs.get("expected_ids") or [])
        return {"replacements": [replacement]}

    monkeypatch.setattr(planner, "request_structured", fake_request)
    result = planner.repair_plan({
        "document_id": "doc",
        "tasks": [("mcq", 2)],
        "num_models": 1,
        "difficulty": "medium",
        "plans": plans,
        "plan_errors": errors,
        "plan_attempts": 1,
        "assessment_profile": _profile(),
        "planning_issues": [],
        "retrieved_chunks": [{"child_id": "chunk-a", "content": "source"}],
        "unrepairable_plan_slots": [],
    })

    repaired = result["plans"][0]["items"]["mcq"]
    assert requested_slots == ["m1_mcq_2"]
    assert repaired[0]["task_id"] == good_task_id
    assert repaired[1]["slot_id"] == "m1_mcq_2"
    assert repaired[1]["task_id"] != "stale_task_id"
    assert repaired[1]["task_id"] == assessment_catalog.canonical_task_ids(
        repaired[1]
    )["task_id"]
    assert result["plan_errors"] == []


def test_same_concept_with_different_ideas_is_valid():
    plans = [
        {"model_number": 1, "items": {"mcq": [_item("chunk-a", "uses labeled examples")]}},
        {"model_number": 2, "items": {"mcq": [_item("chunk-a", "predicts target labels")]}},
    ]
    assert planner.validate_plans(
        plans, [("mcq", 1)], 2, {"chunk-a"}
    ) == []


def test_targeted_plan_repair_changes_only_bad_slot(monkeypatch):
    plans = [
        {"model_number": 1, "items": {"mcq": [_item("chunk-a", "uses labeled examples")]}},
        {"model_number": 2, "items": {"mcq": [_item("chunk-a", "uses labeled examples")]}},
    ]
    plans = planner.finalize_plans(plans, [("mcq", 1)])
    errors = planner.validate_plans(plans, [("mcq", 1)], 2, {"chunk-a"})
    assert any("m2_mcq_1" in error for error in errors)

    class FakeClient:
        def chat_structured(self, *args, **kwargs):
            return {
                "replacements": [
                    {
                        "slot_id": "m2_mcq_1",
                        "source_chunk_id": "chunk-a",
                        "support_chunk_ids": [],
                            "topic": "Machine learning",
                            "concept": "Supervised learning",
                            "canonical_skill": "predicts target labels",
                            "knowledge_focus": "predicted target labels",
                            "student_work_product": "apply_concept: predicted target labels",
                            "relationship": None,
                        "target": "predicted target labels",
                            "task_operation": "apply_concept",
                            "given_conditions": [],
                            "scenario_family": None,
                            "required_givens": [],
                            "text_answerable": True,
                        "reasoning_mode": "conceptual_reasoning",
                        "difficulty": "medium",
                        "idea_to_test": "predicts target labels",
                    }
                ]
            }

    monkeypatch.setattr(planner, "create_llm_client", FakeClient)
    result = planner.repair_plan(
        {
            "document_id": "doc",
            "tasks": [("mcq", 1)],
            "num_models": 2,
            "plans": plans,
            "plan_errors": errors,
            "plan_attempts": 1,
            "retrieved_chunks": [{"child_id": "chunk-a", "content": "source"}],
        }
    )
    assert result["plan_errors"] == []
    assert result["plans"][0] == plans[0]
    assert result["plans"][1]["items"]["mcq"][0]["idea_to_test"] == "predicts target labels"


def test_duplicate_task_key_crosses_question_types_and_ignores_difficulty():
    shared = {
        "relationship": "Snell's law",
        "target": "refractive index",
        "reasoning_mode": "calculation",
    }
    plans = [{
        "model_number": 1,
        "items": {
            "mcq": [_item("chunk-a", "calculate index", difficulty="easy", **shared)],
            "word_problem": [
                _item("chunk-a", "calculate index from a situation", difficulty="hard", **shared)
            ],
        },
    }]
    errors = planner.validate_plans(
        plans, [("mcq", 1), ("word_problem", 1)], 1, {"chunk-a"}
    )
    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_same_task_family_targets_require_explicit_reuse_metadata():
    plans = [{
        "model_number": 1,
        "items": {
            "mcq": [_item(
                "chunk-a", "calculate index", relationship="Snell's law",
                target="refractive index", reasoning_mode="calculation",
            )],
            "word_problem": [_item(
                "chunk-a", "calculate angle", relationship="Snell’s law",
                target="refraction angle", reasoning_mode="calculation",
            )],
        },
    }]
    errors = planner.validate_plans(
        plans, [("mcq", 1), ("word_problem", 1)], 1, {"chunk-a"}
    )
    assert any("duplicates catalog task" in error for error in errors)


def test_symbolic_formula_aliases_conflict_before_generation():
    plans = [{
        "model_number": 1,
        "items": {
            "mcq": [_item(
                "chunk-a", "select photon energy formula",
                relationship="E_photon = hν", target="photon energy formula",
                task_operation="apply_equation", reasoning_mode="symbolic_manipulation",
            )],
            "equation": [_item(
                "chunk-a", "write quantum energy formula",
                relationship="E_كم = hν", target="quantum energy formula",
                task_operation="apply_equation", reasoning_mode="symbolic_manipulation",
            )],
        },
    }]

    errors = planner.validate_plans(
        plans, [("mcq", 1), ("equation", 1)], 1, {"chunk-a"}
    )

    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_global_duplicate_detects_same_scenario_across_types_even_with_different_operation():
    equation = _item(
        "chunk-a", "write the break-even equation", target="break-even point",
        task_operation="formulate_equation",
    )
    word_problem = _item(
        "chunk-a", "calculate the break-even point", target="break-even point",
        task_operation="calculate_from_scenario",
    )
    for item in (equation, word_problem):
        item["scenario_family"] = "company break-even"
        item["given_conditions"] = ["cost 48x + 500", "income 65x - 145"]
        item["required_givens"] = list(item["given_conditions"])
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"equation": [equation], "word_problem": [word_problem]}}],
        [("equation", 1), ("word_problem", 1)], 1, {"chunk-a"},
    )
    assert any("same scenario/setup" in error for error in errors)


def test_plan_rejects_visual_dependency_without_reproduced_text_givens():
    item = _item("chunk-a", "interpret the wage table")
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"word_problem": [item]}}],
        [("word_problem", 1)], 1, {"chunk-a"},
    )
    assert any("visual representation" in error for error in errors)


def test_arabic_energy_form_is_not_misread_as_a_visual_dependency():
    item = _item(
        "chunk-a",
        "مصطلح: اسم شكل الطاقة الذي يسلك سلوكًا شبيهًا بالموجات أثناء انتقاله في الفراغ",
        target="اسم شكل الطاقة",
        task_operation="identify_term_from_definition",
        reasoning_mode="conceptual_recall",
    )
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"mcq": [item]}}],
        [("mcq", 1)], 1, {"chunk-a"},
    )
    assert not any("visual representation" in error for error in errors)


def test_arabic_explicit_figure_reference_is_a_visual_dependency():
    item = _item("chunk-a", "فسر البيانات الموضحة في الشكل التالي")
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"word_problem": [item]}}],
        [("word_problem", 1)], 1, {"chunk-a"},
    )
    assert any("visual representation" in error for error in errors)


def test_arabic_derivation_cue_is_recognized():
    item = _item(
        "chunk-a", "استنتاج العلاقة الرياضية بين الطاقة والتردد",
        relationship="E = hν", target="علاقة الطاقة والتردد",
        task_operation="derive_relationship", reasoning_mode="derivation_relationship",
    )
    assert planner.validate_plans(
        [{"model_number": 1, "items": {"equation": [item]}}],
        [("equation", 1)], 1, {"chunk-a"},
    ) == []


def test_truncated_planner_checkpoint_keeps_only_complete_profile_and_slots():
    profile = _profile()
    item = _item("chunk-a", "first complete skill")
    item["slot_id"] = "m1_mcq_1"
    partial = (
        '{"assessment_profile":' + __import__("json").dumps(profile) +
        ',"planning_issues":[],"exams":[{"model_number":1,"items":{"mcq":[' +
        __import__("json").dumps(item) +
        ',{"slot_id":"m1_mcq_2","source_chunk_id":"chunk-a"'
    )
    recovered_profile, plans, issues = planner.recover_planner_checkpoint(
        partial, [("mcq", 2)], 1
    )
    assert recovered_profile == profile
    assert issues == []
    assert [value["slot_id"] for value in plans[0]["items"]["mcq"]] == ["m1_mcq_1"]


def test_initial_planner_call_uses_25000_output_tokens(monkeypatch):
    captured = {}
    item = _item("chunk-a", "grounded skill")
    item["slot_id"] = "m1_mcq_1"

    def fake_call(_prompt, **kwargs):
        captured.update(kwargs)
        return {
            "assessment_profile": _profile(),
            "planning_issues": [],
            "exams": [{"model_number": 1, "items": {"mcq": [item]}}],
        }

    monkeypatch.setattr(planner, "_call_planner", fake_call)
    result = planner.plan_exams({
        "document_id": "doc", "tasks": [("mcq", 1)], "num_models": 1,
        "planner_context": "source", "difficulty": "medium", "document_language": "en",
        "retrieved_chunks": [{"child_id": "chunk-a", "content": "source"}],
    })
    assert captured["max_tokens"] == 25000
    assert captured["operation"] == "initial_planner"
    assert result["plan_attempts"] == 1


def test_truncation_continues_only_missing_slots_when_profile_is_complete(monkeypatch):
    import json

    profile = _profile()
    first = _item("chunk-a", "first skill", target="first target")
    first["slot_id"] = "m1_mcq_1"
    partial = (
        '{"assessment_profile":' + json.dumps(profile) +
        ',"planning_issues":[],"exams":[{"model_number":1,"items":{"mcq":[' +
        json.dumps(first) + ',{"slot_id":"m1_mcq_2"'
    )
    initial_calls = []

    def truncated_call(_prompt, **kwargs):
        initial_calls.append(kwargs)
        raise DeepSeekIncompleteError(
            "response incomplete: max_output_tokens",
            partial_output=partial, reason="max_output_tokens",
        )

    captured_missing = []

    def continuation(_state, _plans, _profile_value, _issues, missing):
        captured_missing.extend(missing)
        second = _item("chunk-a", "second skill", target="second target")
        second["slot_id"] = "m1_mcq_2"
        return [second]

    monkeypatch.setattr(planner, "_call_planner", truncated_call)
    monkeypatch.setattr(planner, "_continue_plan_slots", continuation)
    result = planner.plan_exams({
        "document_id": "doc", "tasks": [("mcq", 2)], "num_models": 1,
        "planner_context": "full source", "difficulty": "medium", "document_language": "en",
        "retrieved_chunks": [{"child_id": "chunk-a", "content": "source"}],
    })
    assert len(initial_calls) == 1
    assert captured_missing == ["m1_mcq_2"]
    assert not result["plan_errors"]


def test_truncated_profile_allows_only_one_controlled_full_context_retry(monkeypatch):
    calls = []
    item = _item("chunk-a", "grounded skill")
    item["slot_id"] = "m1_mcq_1"

    def fake_call(_prompt, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise DeepSeekIncompleteError(
                "response incomplete", partial_output='{"assessment_profile":',
                reason="max_output_tokens",
            )
        return {
            "assessment_profile": _profile(), "planning_issues": [],
            "exams": [{"model_number": 1, "items": {"mcq": [item]}}],
        }

    monkeypatch.setattr(planner, "_call_planner", fake_call)
    result = planner.plan_exams({
        "document_id": "doc", "tasks": [("mcq", 1)], "num_models": 1,
        "planner_context": "full source", "difficulty": "medium", "document_language": "en",
        "retrieved_chunks": [{"child_id": "chunk-a", "content": "source"}],
    })
    assert len(calls) == 2
    assert all(call["max_tokens"] == 25000 for call in calls)
    assert result["plan_attempts"] == 2


def test_continuation_uses_bounded_low_reasoning_agent(monkeypatch):
    captured = {}
    replacement = _item("chunk-a", "second skill", target="second target")
    replacement["slot_id"] = "m1_mcq_2"

    def fake_call(_prompt, **kwargs):
        captured.update(kwargs)
        return {"replacements": [replacement]}

    monkeypatch.setattr(planner, "_call_planner", fake_call)
    first = _item("chunk-a", "first skill", target="first target")
    first["slot_id"] = "m1_mcq_1"
    result = planner._continue_plan_slots(
        {
            "retrieved_chunks": [{"child_id": "chunk-a", "content": "source"}],
            "difficulty": "medium",
        },
        [{"model_number": 1, "items": {"mcq": [first]}}],
        _profile(),
        [],
        ["m1_mcq_2"],
    )
    assert result == [replacement]
    assert captured["agent"] == "planner_continuation"
    assert captured["operation"] == "planner_continuation"
    assert captured["max_tokens"] < 25000


def test_fitb_plan_rejects_formula_or_calculated_answer_shape():
    formula = _item(
        "chunk-a",
        "Recall the equation c = lambda nu",
        target="wave equation",
        task_operation="recall_equation",
        reasoning_mode="conceptual_recall",
    )
    plans = [{"model_number": 1, "items": {"fill_in_the_blank": [formula]}}]
    errors = planner.validate_plans(
        plans, [("fill_in_the_blank", 1)], 1, {"chunk-a"}
    )
    assert any("Fill-in-the-Blank require an equation" in error for error in errors)


def test_fitb_plan_allows_grounded_short_term():
    term = _item(
        "chunk-a",
        "Recall the term electromagnetic radiation",
        target="electromagnetic radiation",
        task_operation="recall_term",
        reasoning_mode="conceptual_recall",
    )
    term.update({
        "fitb_entry_id": "fitb_entry_test_electromagnetic_radiation",
        "answer": "electromagnetic radiation",
        "evidence_chunk_id": "chunk-a",
        "clue_basis": "The source identifies electromagnetic radiation as the term.",
    })
    plans = [{"model_number": 1, "items": {"fill_in_the_blank": [term]}}]
    assert planner.validate_plans(
        plans, [("fill_in_the_blank", 1)], 1, {"chunk-a"}
    ) == []


def test_fitb_relationship_notation_is_not_by_itself_a_formula_answer():
    term = _item(
        "chunk-a",
        "Recall the term photon.",
        target="photon",
        relationship="The source writes the transition as n = 2.",
        task_operation="identify_term",
        reasoning_mode="conceptual_recall",
    )
    assert planner._fitb_plan_requires_non_term_answer(term) is False


def test_fitb_operation_label_does_not_override_word_answer_shape():
    item = _item(
        "chunk-a", "Identify the keyword used in the source.",
        target="electromagnetic radiation",
        task_operation="calculate_related_term",
        reasoning_mode="conceptual_recall",
    )
    assert planner._fitb_plan_requires_non_term_answer(item) is False


def test_fitb_quantitative_endpoint_with_relationship_notation_is_rejected():
    item = _item(
        "chunk-a",
        "Identify the final energy level.",
        target="final energy level",
        relationship="Transitions end at n = 2.",
        task_operation="identify_level",
        reasoning_mode="conceptual_recall",
    )
    assert planner._fitb_plan_requires_non_term_answer(item) is True


def test_distinct_word_problem_tasks_may_share_broad_coverage():
    first = _item(
        "chunk-a", "Calculate radio wavelength", target="radio wavelength",
        task_operation="calculate_radio_wavelength", reasoning_mode="calculation",
    )
    second = _item(
        "chunk-a", "Calculate microwave wavelength", target="microwave wavelength",
        task_operation="calculate_microwave_wavelength", reasoning_mode="calculation",
    )
    for item, task_id, family in ((first, "task-radio", "family-radio"), (second, "task-microwave", "family-microwave")):
        item.update({"task_id": task_id, "task_family_id": family, "coverage_id": "shared-knowledge"})
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"word_problem": [first, second]}}],
        [("word_problem", 2)], 1, {"chunk-a"}, "mix",
    )
    assert not any("repeats catalog knowledge coverage" in error for error in errors)


def test_semantic_plan_repair_selects_candidate_that_resolves_named_conflict(monkeypatch):
    def task(name: str, operation: str, target: str, product: str) -> dict:
        value = {
            "knowledge_anchor": f"anchor {name}", "topic": "waves", "concept": name,
            "relationship": "c = wavelength * frequency" if "frequency" in target else None,
            "task_operation": operation, "student_work_product": product,
            "target": target, "reasoning_mode": "calculation" if operation.startswith("calculate") else "conceptual_reasoning",
            "allowed_question_types": ["word_problem"], "evidence_chunk_ids": ["chunk-a"],
            "required_givens": [], "scenario_family": name, "text_answerable": True,
            "supported_difficulties": ["medium"],
        }
        value.update(assessment_catalog.canonical_task_ids(value) or {})
        return value

    first = task("radio", "calculate_frequency", "frequency from wavelength", "numeric frequency")
    duplicate = task("xray", "calculate_frequency", "frequency from wavelength", "numeric frequency")
    distinct = task("spectrum", "interpret_spectrum", "identify element from line pattern", "evidence-based identification")
    plans = [{"model_number": 1, "items": {"word_problem": [
        assessment_catalog.materialize_plan_item(first, "m1_word_problem_1", "medium"),
        assessment_catalog.materialize_plan_item(duplicate, "m1_word_problem_2", "medium"),
    ]}}]
    error = "Slot m1_word_problem_2 duplicates the planned reasoning task in m1_word_problem_1; change the target."
    monkeypatch.setattr(planner, "create_llm_client", lambda: object())
    monkeypatch.setattr(planner, "request_structured", lambda *_args, **_kwargs: {
        "repair": {
            "slot_to_replace": "m1_word_problem_2",
            "conflicts_with": ["m1_word_problem_1"],
            "conflict_reason": "Both calculate frequency from wavelength using the same relationship.",
            "what_must_be_different": ["Use a different operation and student work product."],
            "replacement_task_id": distinct["task_id"],
            "why_this_replacement_resolves_the_conflict": "It interprets a spectrum instead of repeating the calculation.",
        }
    })
    state = {
        "assessment_catalog": {"tasks": [first, duplicate, distinct]},
        "tasks": [("word_problem", 2)], "num_models": 1, "difficulty": "medium",
        "retrieved_chunks": [{"child_id": "chunk-a", "content": "grounded"}],
        "assessment_profile": {}, "planning_issues": [],
    }

    revised, errors, accepted = planner._repair_plan_slot(
        state, plans, "m1_word_problem_2", [error]
    )
    repaired = revised[0]["items"]["word_problem"][1]
    assert accepted is True
    assert errors == []
    assert repaired["task_id"] == distinct["task_id"]
    assert repaired["plan_repair_conflicts_with"] == ["m1_word_problem_1"]
    assert repaired["plan_repair_required_differences"]


def test_exhausted_plan_repair_never_prunes_requested_slot(monkeypatch):
    bad = _item("chunk-a", "bad visual task", target="missing table value")
    bad["slot_id"] = "m1_word_problem_1"
    good = _item("chunk-a", "good concept", target="supported concept")
    good["slot_id"] = "m1_mcq_1"
    plans = [{
        "model_number": 1,
        "items": {"mcq": [good], "word_problem": [bad]},
    }]
    errors = ["Slot m1_word_problem_1 has visual_dependency and missing data."]

    monkeypatch.setattr(
        planner,
        "_repair_plan_slot",
        lambda _state, current, _sid, current_errors: (current, current_errors, False),
    )
    with pytest.raises(RuntimeError, match="could not preserve every requested slot"):
        planner.repair_plan({
            "document_id": "doc",
            "tasks": [("mcq", 1), ("word_problem", 1)],
            "num_models": 1,
            "plans": plans,
            "plan_errors": errors,
            "plan_attempts": 2,
            "retrieved_chunks": [{"child_id": "chunk-a", "content": "source"}],
        })
    assert plans[0]["items"]["word_problem"] == [bad]


def test_prior_missing_slot_does_not_block_unrelated_duplicate_plan_repair(monkeypatch):
    original = _item("chunk-a", "identify refraction", target="refraction")
    original["slot_id"] = "m1_mcq_1"
    plans = planner.finalize_plans(
        [{"model_number": 1, "items": {"mcq": [original], "fill_in_the_blank": []}}],
        [("mcq", 1), ("fill_in_the_blank", 1)],
    )
    replacement = _item("chunk-a", "identify concave lens behavior", target="lens behavior")
    replacement["slot_id"] = "m1_mcq_1"
    monkeypatch.setattr(planner, "create_llm_client", lambda: object())
    monkeypatch.setattr(
        planner,
        "request_structured",
        lambda *_args, **_kwargs: {"replacements": [replacement]},
    )
    state = {
        "tasks": [("mcq", 1), ("fill_in_the_blank", 1)],
        "num_models": 1,
        "difficulty": "medium",
        "assessment_profile": _profile(),
        "planning_issues": [],
        "retrieved_chunks": [{"child_id": "chunk-a", "content": "source"}],
        "unrepairable_plan_slots": [{"slot_id": "m1_fill_in_the_blank_1"}],
    }

    revised, errors, accepted = planner._repair_plan_slot(
        state,
        plans,
        "m1_mcq_1",
        ["Slot m1_mcq_1 has duplicate_or_repetition."],
    )

    assert accepted is True
    assert errors == []
    assert revised[0]["items"]["mcq"][0]["knowledge_focus"] == "lens behavior"


def test_null_relationship_duplicate_key_falls_back_to_concept():
    plans = [{
        "model_number": 1,
        "items": {
            "mcq": [_item("chunk-a", "recognize labeled examples", target="training data")],
            "why": [_item("chunk-a", "explain labeled examples", target="training data")],
        },
    }]
    errors = planner.validate_plans(
        plans, [("mcq", 1), ("why", 1)], 1, {"chunk-a"}
    )
    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_support_chunks_must_be_selected_unique_and_different_from_main():
    item = _item("chunk-a", "derive a relationship")
    item["support_chunk_ids"] = ["chunk-a", "chunk-b", "chunk-b", "unknown"]
    plans = [{"model_number": 1, "items": {"essay": [item]}}]
    errors = planner.validate_plans(
        plans, [("essay", 1)], 1, {"chunk-a", "chunk-b"}
    )
    assert any("main chunk" in error for error in errors)
    assert any("repeats a support" in error for error in errors)
    assert any("unselected support" in error for error in errors)


def test_mix_does_not_force_unsupported_difficulty_quota():
    plans = [{
        "model_number": 1,
        "items": {
            "mcq": [
                _item("chunk-a", f"direct idea {index}", target=f"target {index}", difficulty="easy")
                for index in range(3)
            ]
        },
    }]
    assert planner.validate_plans(
        plans, [("mcq", 3)], 1, {"chunk-a"}, "mix"
    ) == []


def test_symbolic_mode_rejects_a_derivation_task():
    plans = [{"model_number": 1, "items": {"equation": [_item(
        "chunk-a",
        "Derive the relationship y = mx + b from the supplied equations.",
        relationship="linear relationship",
        target="y = mx + b",
        reasoning_mode="symbolic_manipulation",
        difficulty="hard",
    )]}}]
    errors = planner.validate_plans(
        plans, [("equation", 1)], 1, {"chunk-a"}, "mix"
    )
    assert any("must be derivation_relationship" in error for error in errors)


def test_direct_single_relationship_calculation_cannot_be_labeled_hard():
    plans = [{"model_number": 1, "items": {"mcq": [_item(
        "chunk-a",
        "Calculate the critical angle using sin theta_c = n2/n1 and choose the value.",
        relationship="critical-angle relationship",
        target="critical angle",
        reasoning_mode="calculation",
        difficulty="hard",
    )]}}]
    plans[0]["items"]["mcq"][0]["reasoning_depth"] = "easy"
    errors = planner.validate_plans(
        plans, [("mcq", 1)], 1, {"chunk-a"}, "mix"
    )
    assert any("does not match its reasoning_depth" in error for error in errors)


def test_scenario_suffix_does_not_hide_duplicate_quantitative_target():
    plans = [{"model_number": 1, "items": {
        "mcq": [_item(
            "chunk-a", "calculate refraction angle", relationship="Snell's law",
            target="refraction angle", reasoning_mode="calculation",
        )],
        "word_problem": [_item(
            "chunk-a", "calculate refraction angle in ethanol", relationship="Snell's law",
            target="refraction angle in ethanol", reasoning_mode="calculation",
        )],
    }}]
    errors = planner.validate_plans(
        plans, [("mcq", 1), ("word_problem", 1)], 1, {"chunk-a"}, "mix"
    )
    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_changed_values_and_material_do_not_hide_same_quantitative_task():
    first = _item(
        "chunk-a", "apply Snell law", relationship="Snell's law",
        target="angle of refraction in glass", task_operation="apply_snell_law",
        reasoning_mode="calculation",
    )
    first["given_conditions"] = ["n1=1.00", "n2=1.52", "theta1=30 degrees"]
    second = _item(
        "chunk-a", "apply Snell law", relationship="Snell's law",
        target="angle of refraction in polyethylene", task_operation="apply_snell_law",
        reasoning_mode="calculation",
    )
    second["given_conditions"] = ["n1=1.33", "n2=1.50", "theta1=57.5 degrees"]
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"equation": [first], "word_problem": [second]}}],
        [("equation", 1), ("word_problem", 1)], 1, {"chunk-a"}, "mix",
    )
    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_changed_values_do_not_hide_repeated_scenario_family():
    first = _item("chunk-a", "write break-even equation", task_operation="formulate_equation")
    second = _item("chunk-a", "explain break-even result", task_operation="explain_result")
    first.update({"scenario_family": "company break-even", "given_conditions": ["cost 48x+500"]})
    second.update({"scenario_family": "company break-even", "given_conditions": ["cost 30x+200"]})
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"equation": [first], "why": [second]}}],
        [("equation", 1), ("why", 1)], 1, {"chunk-a"}, "mix",
    )
    assert any("same scenario/setup" in error for error in errors)


def test_same_specific_system_is_duplicate_even_when_operation_labels_differ():
    first = _item(
        "chunk-a", "determine whether the system has a solution",
        relationship="elimination", target="number of solutions",
        task_operation="determine_number_of_solutions",
    )
    second = _item(
        "chunk-a", "classify the inconsistent system",
        relationship="coefficient ratios", target="system classification",
        task_operation="classify_linear_system",
    )
    givens = ["5x+3y=52", "15x+9y=54"]
    first["given_conditions"] = givens
    second["given_conditions"] = givens
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"mcq": [first], "equation": [second]}}],
        [("mcq", 1), ("equation", 1)], 1, {"chunk-a"}, "mix",
    )
    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_same_numeric_setup_is_duplicate_despite_different_condition_wording():
    first = _item(
        "chunk-a", "calculate angle", relationship="Snell law",
        target="refraction angle", task_operation="calculate_angle",
    )
    second = _item(
        "chunk-a", "calculate sine", relationship="Snell law",
        target="sine of refraction angle", task_operation="calculate_sine",
    )
    first["given_conditions"] = ["n1=1.00", "n2=1.52", "theta1=30.0 degrees"]
    second["given_conditions"] = [
        "index of first medium 1.00", "index of second medium 1.52",
        "incidence angle 30.0 degrees",
    ]
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"equation": [first, second]}}],
        [("equation", 2)], 1, {"chunk-a"}, "mix",
    )
    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_arabic_definite_article_does_not_hide_duplicate_target():
    first = _item("chunk-a", "calculate refractive index")
    first.update({
        "relationship": "قانون سنل للانكسار",
        "target": "معامل الانكسار",
        "reasoning_mode": "calculation",
    })
    second = _item("chunk-b", "calculate refractive index of an unknown material")
    second.update({
        "relationship": "قانون سنل للانكسار",
        "target": "معامل انكسار مادة مجهولة",
        "reasoning_mode": "calculation",
    })
    plan = {"model_number": 1, "items": {"equation": [first], "word_problem": [second]}}

    errors = planner.validate_plans([plan], [("equation", 1), ("word_problem", 1)], 1)

    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_hard_conceptual_reasoning_requires_real_depth_cue():
    item = _item("chunk-a", "identify what happens when an object is at the focal point")
    item.update({
        "reasoning_mode": "conceptual_reasoning",
        "difficulty": "hard",
        "reasoning_depth": "easy",
    })
    plan = {"model_number": 1, "items": {"true_false": [item]}}

    errors = planner.validate_plans([plan], [("true_false", 1)], 1)

    assert any("does not match its reasoning_depth" in error for error in errors)


def test_why_rejects_numerical_calculation_mode():
    plans = [{"model_number": 1, "items": {"why": [_item(
        "chunk-a", "Calculate the incidence angle.", relationship="Snell's law",
        target="incidence angle", reasoning_mode="calculation",
    )]}}]
    errors = planner.validate_plans(
        plans, [("why", 1)], 1, {"chunk-a"}, "mix"
    )
    assert any("assigns calculation to why" in error for error in errors)


def test_why_semantic_operation_is_not_rejected_by_narrow_operation_allowlist():
    item = _item(
        "chunk-a", "Explain how the mechanism produces the observed effect.",
        relationship="mechanism relationship", target="causal mechanism",
        task_operation="explain", reasoning_mode="conceptual_reasoning",
    )
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"why": [item]}}],
        [("why", 1)], 1, {"chunk-a"}, "mix",
    )
    assert not any("non-causal operation" in error for error in errors)


def test_direct_comparison_cannot_be_labeled_hard():
    plans = [{"model_number": 1, "items": {"mcq": [_item(
        "chunk-a", "Compare wavelength in the medium with wavelength in vacuum.",
        relationship="wavelength-index relationship", target="wavelength in medium",
        reasoning_mode="comparison_prediction", difficulty="hard",
    )]}}]
    plans[0]["items"]["mcq"][0]["reasoning_depth"] = "medium"
    errors = planner.validate_plans(
        plans, [("mcq", 1)], 1, {"chunk-a"}, "mix"
    )
    assert any("does not match its reasoning_depth" in error for error in errors)


def test_missing_output_slots_are_reported_by_id():
    exams = [
        {
            "model_number": 1,
            "questions": {
                "mcq": [{"slot_id": "m1_mcq_1"}],
                "essay": [],
            },
        }
    ]
    assert exam_builder._missing_output_slots(
        exams, [("mcq", 2), ("essay", 1)], 1
    ) == ["m1_mcq_2", "m1_essay_1"]


def test_profile_requires_guidance_for_all_draftwork_types():
    schema = planner.build_plan_schema([("mcq", 1)], 1, ["chunk-a"])
    question_types = schema["properties"]["assessment_profile"]["properties"][
        "question_types"
    ]["properties"]
    assert set(question_types) == planner.VALID_TYPES


def test_different_task_operations_are_not_cross_type_duplicates():
    plans = [{"model_number": 1, "items": {
        "equation": [_item(
            "chunk-a", "derive the relationship", relationship="rate law",
            target="rate expression", task_operation="derive_relationship",
            reasoning_mode="derivation_relationship",
        )],
        "word_problem": [_item(
            "chunk-a", "apply the relationship in a scenario", relationship="rate law",
            target="rate expression", task_operation="calculate_from_scenario",
            reasoning_mode="calculation",
        )],
    }}]
    assert planner.validate_plans(
        plans, [("equation", 1), ("word_problem", 1)], 1, {"chunk-a"}
    ) == []


def test_refraction_recognition_and_definition_are_distinct_atomic_tasks():
    mcq = _item(
        "chunk-a", "recognize the meaning of refraction",
        task_operation="identify_definition", reasoning_mode="conceptual_recall",
    )
    definition = _item(
        "chunk-a", "define refraction",
        task_operation="define_term", reasoning_mode="conceptual_recall",
    )
    for item in (mcq, definition):
        item["knowledge_focus"] = "definition of refraction"
    mcq["student_work_product"] = "select the definition of refraction"
    definition["student_work_product"] = "state the definition of refraction"

    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"mcq": [mcq], "definition": [definition]}}],
        [("mcq", 1), ("definition", 1)], 1, {"chunk-a"},
    )

    assert not any("duplicates the planned reasoning task" in error for error in errors)


def test_air_to_glass_direction_fact_cannot_repeat_as_mcq_and_true_false():
    mcq = _item(
        "chunk-a", "choose the direction of bending from air to glass",
        task_operation="predict_direction", reasoning_mode="conceptual_recall",
    )
    true_false = _item(
        "chunk-a", "judge the direction of bending from air to glass",
        task_operation="judge_statement", reasoning_mode="conceptual_recall",
    )
    for item in (mcq, true_false):
        item["knowledge_focus"] = "light bends toward the normal from air to glass"
    mcq["student_work_product"] = "select toward or away from the normal"
    true_false["student_work_product"] = "judge the same direction fact"

    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"mcq": [mcq], "true_false": [true_false]}}],
        [("mcq", 1), ("true_false", 1)], 1, {"chunk-a"},
    )

    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_equation_and_word_problem_same_snell_calculation_are_duplicates():
    equation = _item(
        "chunk-a", "calculate the refracted angle",
        relationship="Snell's law", target="refracted angle",
        task_operation="calculate_refraction_angle", reasoning_mode="calculation",
    )
    word_problem = _item(
        "chunk-a", "calculate the refracted angle for a ray entering glass",
        relationship="Snell's law", target="refracted angle",
        task_operation="calculate_from_scenario", reasoning_mode="calculation",
    )
    for item in (equation, word_problem):
        item.update({
            "knowledge_focus": "apply Snell's law to determine refracted angle",
            "student_work_product": "calculate theta2 from n1 n2 and theta1",
            "given_conditions": ["n1=1.00", "n2=1.52", "theta1=30 degrees"],
            "required_givens": ["n1", "n2", "theta1"],
            "scenario_family": "air to glass Snell calculation",
        })

    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"equation": [equation], "word_problem": [word_problem]}}],
        [("equation", 1), ("word_problem", 1)], 1, {"chunk-a"},
    )

    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_snell_relationship_is_reusable_for_genuinely_different_work():
    derivation = _item(
        "chunk-a", "derive the symbolic Snell relationship",
        relationship="Snell's law", target="symbolic relationship",
        task_operation="derive_relationship", reasoning_mode="derivation_relationship",
    )
    calculation = _item(
        "chunk-a", "calculate an unknown refractive index",
        relationship="Snell's law", target="unknown refractive index",
        task_operation="calculate_refractive_index", reasoning_mode="calculation",
    )
    derivation.update({
        "knowledge_focus": "derivation of Snell's law",
        "student_work_product": "derive a symbolic relationship",
    })
    calculation.update({
        "knowledge_focus": "use Snell's law to determine material index",
        "student_work_product": "calculate an unknown refractive index from two angles",
        "given_conditions": ["n1=1.00", "theta1=30 degrees", "theta2=20 degrees"],
        "required_givens": ["n1", "theta1", "theta2"],
    })

    assert planner.validate_plans(
        [{"model_number": 1, "items": {"equation": [derivation], "word_problem": [calculation]}}],
        [("equation", 1), ("word_problem", 1)], 1, {"chunk-a"},
    ) == []


def test_shared_wave_constant_does_not_duplicate_distinct_student_work():
    recognition = _item(
        "chunk-a", "recognize c = lambda nu", relationship="c = lambda nu",
        target="truth of the wave relationship", task_operation="judge_relationship",
        reasoning_mode="conceptual_recall",
    )
    wavelength = _item(
        "chunk-a", "calculate wavelength from frequency", relationship="c = lambda nu",
        target="wavelength", task_operation="calculate_wavelength",
        reasoning_mode="calculation",
    )
    frequency = _item(
        "chunk-a", "calculate frequency from wavelength", relationship="c = lambda nu",
        target="frequency", task_operation="calculate_frequency",
        reasoning_mode="calculation",
    )
    recognition["given_conditions"] = ["c = 3.00 x 10^8 m/s"]
    wavelength["given_conditions"] = ["c = 3.00 x 10^8 m/s", "frequency = 6.0 x 10^14 Hz"]
    frequency["given_conditions"] = ["c = 3.00 x 10^8 m/s", "wavelength = 500 nm"]

    errors = planner.validate_plans(
        [{"model_number": 1, "items": {
            "true_false": [recognition], "word_problem": [wavelength, frequency],
        }}],
        [("true_false", 1), ("word_problem", 2)], 1, {"chunk-a"}, "mix",
    )

    assert not any("duplicates the planned reasoning task" in error for error in errors)


def test_same_quantitative_setup_operation_and_target_remains_duplicate():
    first = _item(
        "chunk-a", "calculate wavelength", relationship="c = lambda nu",
        target="wavelength", task_operation="calculate_wavelength",
        reasoning_mode="calculation",
    )
    second = _item(
        "chunk-a", "find the wavelength", relationship="c = lambda nu",
        target="wavelength", task_operation="calculate_wavelength",
        reasoning_mode="calculation",
    )
    for item in (first, second):
        item["given_conditions"] = ["c = 3.00 x 10^8 m/s", "frequency = 6.0 x 10^14 Hz"]
    errors = planner.validate_plans(
        [{"model_number": 1, "items": {"word_problem": [first, second]}}],
        [("word_problem", 2)], 1, {"chunk-a"}, "mix",
    )
    assert any("duplicates the planned reasoning task" in error for error in errors)


def test_unsupported_requested_type_returns_structured_issue_without_missing_slot_error():
    profile = _profile()
    profile["question_types"]["essay"] = {
        "support": "unsupported",
        "task_operations": [],
        "form_guidance": "The selection does not support an evidence-grounded essay.",
        "evidence_chunk_ids": [],
    }
    issues = [{
        "question_type": "essay",
        "requested_count": 1,
        "supported_count": 0,
        "reason": "No extended writing task is supported.",
        "suggested_question_types": ["why"],
    }]
    assert planner.validate_assessment_profile(
        profile, [("essay", 1)], {"chunk-a"}, issues
    ) == []
    assert planner.validate_plans(
        [{"model_number": 1, "items": {"essay": []}}],
        [("essay", 1)], 1, {"chunk-a"}, "mix", profile, issues,
    ) == []
