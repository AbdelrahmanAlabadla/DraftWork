from __future__ import annotations

from app.online import exam_builder, planner


def _item(chunk: str, idea: str) -> dict:
    return {
        "source_chunk_id": chunk,
        "topic": "Machine learning",
        "concept": "Supervised learning",
        "idea_to_test": idea,
        "concept_to_test": idea,
    }


def test_plan_schema_enforces_exact_counts_and_selected_chunks():
    schema = planner.build_plan_schema(
        [("mcq", 3), ("essay", 1)], 2, ["chunk-a", "chunk-b"]
    )
    exams = schema["properties"]["exams"]
    assert exams["minItems"] == exams["maxItems"] == 2
    items = exams["items"]["properties"]["items"]["properties"]
    assert items["mcq"]["minItems"] == items["mcq"]["maxItems"] == 3
    assert items["essay"]["minItems"] == items["essay"]["maxItems"] == 1
    chunk_rule = items["mcq"]["items"]["properties"]["source_chunk_id"]
    assert chunk_rule["enum"] == ["chunk-a", "chunk-b"]


def test_plan_schema_supports_new_question_types():
    schema = planner.build_plan_schema(
        [("definition", 1), ("equation", 2), ("word_problem", 1)],
        1,
        ["chunk-a"],
    )
    items = schema["properties"]["exams"]["items"]["properties"]["items"]["properties"]
    assert set(items) == {"definition", "equation", "word_problem"}
    assert items["definition"]["minItems"] == 1
    assert items["equation"]["maxItems"] == 2


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
                        "topic": "Machine learning",
                        "concept": "Supervised learning",
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
