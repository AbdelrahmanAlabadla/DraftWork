from __future__ import annotations

import copy
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

NonEmptyString = Annotated[str, Field(min_length=1)]


class StrictOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Return an inline strict schema accepted by both configured providers."""
    root = copy.deepcopy(model.model_json_schema())
    definitions = root.get("$defs", {})

    def visit(node, stack: tuple[str, ...] = ()):
        if isinstance(node, list):
            return [visit(item, stack) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            prefix = "#/$defs/"
            ref = str(node["$ref"])
            if not ref.startswith(prefix):
                raise ValueError(f"Unsupported external JSON Schema ref: {ref}")
            name = ref[len(prefix):]
            if name in stack or name not in definitions:
                raise ValueError(f"Invalid recursive JSON Schema ref: {ref}")
            merged = copy.deepcopy(definitions[name])
            merged.update({key: value for key, value in node.items() if key != "$ref"})
            return visit(merged, (*stack, name))

        normalized = {}
        for key, value in node.items():
            if key in {"$defs", "$schema", "title", "default", "examples"}:
                continue
            if key == "const":
                normalized["enum"] = [value]
                continue
            normalized[key] = visit(value, stack)
        if normalized.get("type") == "object" or "properties" in normalized:
            properties = normalized.get("properties") or {}
            normalized["properties"] = properties
            normalized["required"] = list(properties)
            normalized["additionalProperties"] = False
        return normalized

    schema = visit(root)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Structured output requires an object schema")
    return schema


class PlanItem(StrictOutputModel):
    source_chunk_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    idea_to_test: str = Field(min_length=1)


class MCQOptions(StrictOutputModel):
    A: str = Field(min_length=1)
    B: str = Field(min_length=1)
    C: str = Field(min_length=1)
    D: str = Field(min_length=1)


class MCQQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: MCQOptions
    correct_answer: Literal["A", "B", "C", "D"]


class TrueFalseQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    answer: Literal["True", "False"]


class FITBQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answers: list[NonEmptyString] = Field(min_length=1, max_length=2)


class ShortAnswerQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)


class EssayQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    key_points: list[NonEmptyString] = Field(min_length=1)


class FITBWordBank(StrictOutputModel):
    correct_terms: list[NonEmptyString]
    distractors: list[NonEmptyString] = Field(min_length=2, max_length=2)


class ValidatorVerdict(StrictOutputModel):
    question_id: str = Field(min_length=1)
    question_valid: bool
    answer_valid: bool
    action: Literal[
        "PASS",
        "FIX_QUESTION",
        "FIX_ANSWER",
        "FIX_OPTIONS",
        "FIX_QUESTION_AND_ANSWER",
    ]
    fields_to_fix: list[str]
    reason: str
    expected_fix: str


class RepairMCQQuestion(StrictOutputModel):
    question: str = Field(min_length=1)
    options: MCQOptions
    correct_answer: Literal["A", "B", "C", "D"]


class RepairTrueFalseQuestion(StrictOutputModel):
    statement: str = Field(min_length=1)
    answer: Literal["True", "False"]


class RepairFITBQuestion(StrictOutputModel):
    question: str = Field(min_length=1)
    answers: list[NonEmptyString] = Field(min_length=1, max_length=2)
    word_bank: list[NonEmptyString]


class RepairShortAnswerQuestion(StrictOutputModel):
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)


class RepairEssayQuestion(StrictOutputModel):
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    key_points: list[NonEmptyString]


_GENERATION_ITEMS: dict[str, type[StrictOutputModel]] = {
    "mcq": MCQQuestion,
    "true_false": TrueFalseQuestion,
    "fill_in_the_blank": FITBQuestion,
    "short_answer": ShortAnswerQuestion,
    "essay": EssayQuestion,
}

_REPAIR_QUESTIONS: dict[str, type[StrictOutputModel]] = {
    "mcq": RepairMCQQuestion,
    "true_false": RepairTrueFalseQuestion,
    "fill_in_the_blank": RepairFITBQuestion,
    "short_answer": RepairShortAnswerQuestion,
    "essay": RepairEssayQuestion,
}


def _literal(values: list[str]):
    unique = tuple(dict.fromkeys(str(value) for value in values if str(value)))
    return Literal.__getitem__(unique) if unique else str


def _bounded_list(item_type, count: int):
    return Annotated[list[item_type], Field(min_length=count, max_length=count)]


def _generation_item_model(qtype: str, slot_ids: list[str]):
    base = _GENERATION_ITEMS[qtype]
    return create_model(
        f"{base.__name__}ForSlots",
        __base__=base,
        slot_id=(_literal(slot_ids), ...),
    )


def generation_response_model(qtype: str, slot_ids: list[str]):
    item_model = _generation_item_model(qtype, slot_ids)
    return create_model(
        f"{item_model.__name__}Response",
        __base__=StrictOutputModel,
        questions=(_bounded_list(item_model, len(slot_ids)), ...),
    )


def objective_bundle_response_model(
    mcq_slot_ids: list[str], true_false_slot_ids: list[str]
):
    fields = {}
    if mcq_slot_ids:
        fields["mcq"] = (generation_response_model("mcq", mcq_slot_ids), ...)
    if true_false_slot_ids:
        fields["true_false"] = (
            generation_response_model("true_false", true_false_slot_ids),
            ...,
        )
    return create_model("ObjectiveBundleResponse", __base__=StrictOutputModel, **fields)


def fitb_word_bank_response_model(count: int):
    return create_model(
        "FITBWordBankResponse",
        __base__=FITBWordBank,
        correct_terms=(_bounded_list(NonEmptyString, count), ...),
    )


def fitb_items_response_model(slot_ids: list[str]):
    item_model = _generation_item_model("fill_in_the_blank", slot_ids)
    return create_model(
        "FITBItemsResponse",
        __base__=StrictOutputModel,
        items=(_bounded_list(item_model, len(slot_ids)), ...),
    )


def planner_response_model(
    tasks: list[tuple[str, int]], num_models: int, chunk_ids: list[str]
):
    plan_item = create_model(
        "GroundedPlanItem",
        __base__=PlanItem,
        source_chunk_id=(_literal(chunk_ids), ...),
    )
    item_fields = {
        qtype: (_bounded_list(plan_item, count), ...) for qtype, count in tasks
    }
    plan_items = create_model(
        "GroundedPlanItems", __base__=StrictOutputModel, **item_fields
    )
    plan = create_model(
        "GroundedExamPlan",
        __base__=StrictOutputModel,
        model_number=(int, Field(ge=1, le=num_models)),
        items=(plan_items, ...),
    )
    return create_model(
        "GroundedExamPlanResponse",
        __base__=StrictOutputModel,
        exams=(_bounded_list(plan, num_models), ...),
    )


def plan_repair_response_model(slot_ids: list[str], chunk_ids: list[str]):
    item = create_model(
        "PlanSlotReplacement",
        __base__=PlanItem,
        slot_id=(_literal(slot_ids), ...),
        source_chunk_id=(_literal(chunk_ids), ...),
    )
    return create_model(
        "PlanSlotRepairResponse",
        __base__=StrictOutputModel,
        replacements=(_bounded_list(item, len(slot_ids)), ...),
    )


def plan_fill_response_model(count: int):
    return create_model(
        "PlanFillResponse",
        __base__=StrictOutputModel,
        questions=(_bounded_list(PlanItem, count), ...),
    )


def validator_response_model(question_ids: list[str]):
    verdict = create_model(
        "ValidatorVerdictForQuestions",
        __base__=ValidatorVerdict,
        question_id=(_literal(question_ids), ...),
    )
    return create_model(
        "ValidatorResponse",
        __base__=StrictOutputModel,
        verdicts=(_bounded_list(verdict, len(question_ids)), ...),
    )


def repair_response_model(items: list[tuple[str, str]]):
    repair_models = []
    for index, (question_id, qtype) in enumerate(items, start=1):
        repair_models.append(
            create_model(
                f"RepairItem{index}",
                __base__=StrictOutputModel,
                question_id=(_literal([question_id]), ...),
                repaired_fields=(list[NonEmptyString], ...),
                question=(_REPAIR_QUESTIONS[qtype], ...),
            )
        )
    item_type = Union[tuple(repair_models)] if len(repair_models) > 1 else repair_models[0]
    return create_model(
        "QuestionRepairResponse",
        __base__=StrictOutputModel,
        repairs=(_bounded_list(item_type, len(items)), ...),
    )
