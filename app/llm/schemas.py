from __future__ import annotations

import copy
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, create_model

NonEmptyString = Annotated[str, Field(min_length=1)]

ReasoningMode = Literal[
    "conceptual_recall",
    "conceptual_reasoning",
    "calculation",
    "symbolic_manipulation",
    "law_or_theorem_application",
    "derivation_relationship",
    "logical_proof_reasoning",
    "representation_interpretation",
    "comparison_prediction",
    "error_analysis",
    "estimation_reasonableness",
]

QuestionType = Literal[
    "mcq",
    "fill_in_the_blank",
    "true_false",
    "definition",
    "why",
    "equation",
    "word_problem",
    "essay",
]


class StrictOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskVariation(StrictOutputModel):
    """One exact, source-supported assessment angle for a reusable task."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    target: NonEmptyString
    operation: NonEmptyString
    givens: list[NonEmptyString] = Field(default_factory=list, max_length=12)
    question_types: list[QuestionType] = Field(default_factory=list)
    evidence_chunk_ids: list[NonEmptyString] | None = Field(default=None, max_length=4)
    reasoning_mode: ReasoningMode | None = None
    student_work_product: NonEmptyString | None = None
    representation: NonEmptyString | None = Field(default=None, alias="form")
    application: NonEmptyString | None = Field(default=None, alias="context")
    numeric_values_may_change: StrictBool = Field(default=False, alias="vary_numbers")
    variation_group_id: str | None = None
    variation_id: str | None = None


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


class DifficultyAdaptation(StrictOutputModel):
    status: Literal["pending", "adapted", "failed", "fallback", "retained"]
    from_difficulty: Literal["easy", "medium", "hard"]
    to_difficulty: Literal["easy", "medium", "hard"]
    base_task_id: str = Field(min_length=1)
    strategy: str | None = None
    failed_strategies: list[str] = Field(default_factory=list)
    changed_reasoning_requirements: list[NonEmptyString] = Field(default_factory=list)
    evidence_support: list[NonEmptyString] = Field(default_factory=list)
    evidence_justification: str | None = None
    base_variation_id: str | None = None
    effective_instance_id: str | None = None
    effective_givens: list[NonEmptyString] = Field(default_factory=list, max_length=12)


class ReasoningProfile(StrictOutputModel):
    depth: Literal["easy", "medium", "hard"]
    requirements: list[NonEmptyString] = Field(min_length=1, max_length=8)
    prerequisites: list[NonEmptyString] = Field(default_factory=list, max_length=8)
    evidence_chunk_ids: list[NonEmptyString] = Field(min_length=1, max_length=4)


class PlanItem(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    source_chunk_id: str = Field(min_length=1)
    support_chunk_ids: list[str]
    topic: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    canonical_skill: str = Field(min_length=1, max_length=160)
    knowledge_focus: str = Field(min_length=1, max_length=200)
    student_work_product: str = Field(min_length=1, max_length=160)
    relationship: str | None
    target: str = Field(min_length=1)
    task_operation: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    given_conditions: list[NonEmptyString] = Field(max_length=12)
    scenario_family: str | None = Field(max_length=160)
    required_givens: list[NonEmptyString] = Field(max_length=12)
    allowed_variations: list[TaskVariation] = Field(default_factory=list)
    permitted_variations: list[TaskVariation] = Field(default_factory=list)
    selected_variation_id: str | None = None
    selected_variation_group_id: str | None = None
    primary_question_types: list[QuestionType] = Field(default_factory=list)
    variable_givens: list[NonEmptyString] = Field(default_factory=list, max_length=12)
    complexity_boundaries: list[NonEmptyString] = Field(default_factory=list, max_length=8)
    exact_source_instance: bool = False
    text_answerable: bool
    reasoning_mode: ReasoningMode
    difficulty: Literal["easy", "medium", "hard"]
    idea_to_test: str = Field(min_length=1)
    requested_difficulty: Literal["easy", "medium", "hard", "mix"] | None = None
    actual_difficulty: Literal["easy", "medium", "hard"] | None = None
    base_catalog_task_id: str | None = None
    base_task_family_id: str | None = None
    difficulty_adaptation: DifficultyAdaptation | None = None
    difficulty_fallback_reason: str | None = None
    reasoning_depth: Literal["easy", "medium", "hard"] | None = None
    reasoning_requirements: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    derived_task_id: str | None = None
    effective_instance_id: str | None = None
    effective_givens: list[NonEmptyString] = Field(default_factory=list, max_length=12)
    selected_replacement_task_id: str | None = None
    task_was_previously_used: bool = False
    adjustment_attempts: int = 0
    adjustment_failure_reason: str | None = None


class QuestionTypeAssessment(StrictOutputModel):
    support: Literal["supported", "limited", "unsupported"]
    task_operations: list[str] = Field(max_length=16)
    form_guidance: str = Field(min_length=1, max_length=320)
    evidence_chunk_ids: list[str] = Field(max_length=8)


class QuestionTypeAssessments(StrictOutputModel):
    mcq: QuestionTypeAssessment
    fill_in_the_blank: QuestionTypeAssessment
    true_false: QuestionTypeAssessment
    definition: QuestionTypeAssessment
    why: QuestionTypeAssessment
    equation: QuestionTypeAssessment
    word_problem: QuestionTypeAssessment
    essay: QuestionTypeAssessment


class AssessmentProfile(StrictOutputModel):
    subject: str = Field(min_length=1, max_length=120)
    domain: Literal["STEM", "NON_STEM"]
    domains: list[str] = Field(min_length=1, max_length=4)
    educational_level: str = Field(min_length=1, max_length=160)
    assessment_style: str = Field(min_length=1, max_length=320)
    question_types: QuestionTypeAssessments
    unsupported_visual_tasks: list[str] = Field(max_length=8)


class PlanningIssue(StrictOutputModel):
    question_type: QuestionType
    requested_count: int = Field(ge=1)
    supported_count: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=400)
    suggested_question_types: list[QuestionType] = Field(max_length=3)


class MCQOptions(StrictOutputModel):
    A: str = Field(min_length=1)
    B: str = Field(min_length=1)
    C: str = Field(min_length=1)
    D: str = Field(min_length=1)


class SolutionCertificate(StrictOutputModel):
    """Internal, non-rendered certificate for a quantitative STEM answer."""

    method: str = Field(min_length=1)
    important_givens: list[NonEmptyString]
    calculation_steps: list[NonEmptyString] = Field(min_length=1)
    final_answer: str = Field(min_length=1)
    units: str = ""
    matching_correct_option: Literal["A", "B", "C", "D"] | None = None
    verification_method: str = Field(min_length=1)


class MCQQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    selected_variation_id: str | None = None
    question: str = Field(min_length=1)
    options: MCQOptions
    correct_answer: Literal["A", "B", "C", "D"]
    solution_certificate: SolutionCertificate | None = None


class TrueFalseQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    selected_variation_id: str | None = None
    statement: str = Field(min_length=1)
    answer: Literal["True", "False"]
    solution_certificate: SolutionCertificate | None = None


class FITBQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    selected_variation_id: str | None = None
    stem: str = Field(min_length=1)
    distractor_1: str = Field(min_length=1)
    distractor_2: str = Field(min_length=1)


class WhyQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    selected_variation_id: str | None = None
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    solution_certificate: SolutionCertificate | None = None


class DefinitionQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    selected_variation_id: str | None = None
    term: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    solution_certificate: SolutionCertificate | None = None


class EquationQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    selected_variation_id: str | None = None
    equation: str = Field(min_length=1)
    solution_steps: list[NonEmptyString] = Field(min_length=1)
    final_answer: str = Field(min_length=1)
    solution_certificate: SolutionCertificate | None = None


class WordProblemQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    selected_variation_id: str | None = None
    question: str = Field(min_length=1)
    solution_steps: list[NonEmptyString] = Field(min_length=1)
    final_answer: str = Field(min_length=1)
    solution_certificate: SolutionCertificate | None = None


class EssayQuestion(StrictOutputModel):
    slot_id: str = Field(min_length=1)
    selected_variation_id: str | None = None
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    key_points: list[NonEmptyString] = Field(min_length=1)
    solution_certificate: SolutionCertificate | None = None


class MCQTaskUnsuitable(StrictOutputModel):
    """A planned task that cannot honestly support a strong four-option MCQ."""

    slot_id: str = Field(min_length=1)
    failure_code: Literal["mcq_task_unsuitable"]
    reason: str = Field(min_length=1)


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
    failure_code: Literal[
        "pass",
        "incorrect_result",
        "incomplete_solution",
        "missing_units",
        "invalid_or_extraneous_solution",
        "unsupported_given",
        "missing_required_given",
        "visual_dependency",
        "question_type_operation_mismatch",
        "task_drift",
        "mcq_task_unsuitable",
        "ambiguous_or_structural",
        "other",
    ]
    expected_results: list[str]
    reason: str
    expected_fix: str
    independent_solution: SolutionCertificate | None = None
    certificate_valid: bool | None = None
    first_incorrect_step: str | None = None


class QuantitativeValidatorVerdict(ValidatorVerdict):
    """Quantitative verdicts always include independently checked work."""

    independent_solution: SolutionCertificate
    certificate_valid: bool


class GlobalExamVerdict(StrictOutputModel):
    """Compact exam-level verdict; PASS rows intentionally carry no explanation."""

    question_id: str = Field(min_length=1)
    action: Literal[
        "PASS",
        "FIX_QUESTION",
        "FIX_ANSWER",
        "FIX_OPTIONS",
        "FIX_QUESTION_AND_ANSWER",
    ]
    failure_code: Literal[
        "pass",
        "duplicate_or_repetition",
        "answer_pattern_imbalance",
        "language_quality",
        "terminology_inconsistency",
        "question_type_operation_mismatch",
        "weak_distractors",
        "ambiguous_or_structural",
        "other",
    ]
    conflicting_question_ids: list[str]
    fields_to_fix: list[str]
    reason: str
    expected_fix: str


class GlobalConflictGroup(StrictOutputModel):
    question_ids: list[NonEmptyString] = Field(min_length=2)
    issue_type: Literal[
        "conceptual_duplicate",
        "scenario_repetition",
        "givens_target_repetition",
        "formula_target_repetition",
        "other_repetition",
    ]
    reason: str = Field(min_length=1)


class ExamQualityFinding(StrictOutputModel):
    code: Literal[
        "topic_skill_diversity",
        "true_false_answer_pattern",
        "arabic_language_quality",
        "terminology_notation_consistency",
        "equation_word_problem_roles",
        "distractor_quality",
        "exam_consistency",
        "other",
    ]
    severity: Literal["warning", "error"]
    question_ids: list[str]
    reason: str = Field(min_length=1)
    expected_fix: str


class RepairMCQQuestion(StrictOutputModel):
    question: str = Field(min_length=1)
    options: MCQOptions
    correct_answer: Literal["A", "B", "C", "D"]
    solution_certificate: SolutionCertificate | None = None


class RepairTrueFalseQuestion(StrictOutputModel):
    statement: str = Field(min_length=1)
    answer: Literal["True", "False"]
    solution_certificate: SolutionCertificate | None = None


class RepairFITBQuestion(StrictOutputModel):
    stem: str = Field(min_length=1)
    distractor_1: str = Field(min_length=1)
    distractor_2: str = Field(min_length=1)


class RepairWhyQuestion(StrictOutputModel):
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    solution_certificate: SolutionCertificate | None = None


class RepairDefinitionQuestion(StrictOutputModel):
    term: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    solution_certificate: SolutionCertificate | None = None


class RepairEquationQuestion(StrictOutputModel):
    equation: str = Field(min_length=1)
    solution_steps: list[NonEmptyString] = Field(min_length=1)
    final_answer: str = Field(min_length=1)
    solution_certificate: SolutionCertificate | None = None


class RepairWordProblemQuestion(StrictOutputModel):
    question: str = Field(min_length=1)
    solution_steps: list[NonEmptyString] = Field(min_length=1)
    final_answer: str = Field(min_length=1)
    solution_certificate: SolutionCertificate | None = None


class RepairEssayQuestion(StrictOutputModel):
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    key_points: list[NonEmptyString]
    solution_certificate: SolutionCertificate | None = None


_GENERATION_ITEMS: dict[str, type[StrictOutputModel]] = {
    "mcq": MCQQuestion,
    "true_false": TrueFalseQuestion,
    "fill_in_the_blank": FITBQuestion,
    "definition": DefinitionQuestion,
    "why": WhyQuestion,
    "equation": EquationQuestion,
    "word_problem": WordProblemQuestion,
    "essay": EssayQuestion,
}

_REPAIR_QUESTIONS: dict[str, type[StrictOutputModel]] = {
    "mcq": RepairMCQQuestion,
    "true_false": RepairTrueFalseQuestion,
    "fill_in_the_blank": RepairFITBQuestion,
    "definition": RepairDefinitionQuestion,
    "why": RepairWhyQuestion,
    "equation": RepairEquationQuestion,
    "word_problem": RepairWordProblemQuestion,
    "essay": RepairEssayQuestion,
}

_REPAIR_PATCH_FIELDS: dict[str, dict[str, object]] = {
    "mcq": {
        "question": NonEmptyString, "options": MCQOptions,
        "correct_answer": Literal["A", "B", "C", "D"],
        "solution_certificate": SolutionCertificate,
    },
    "true_false": {
        "statement": NonEmptyString, "answer": Literal["True", "False"],
        "solution_certificate": SolutionCertificate,
    },
    "fill_in_the_blank": {
        "question": NonEmptyString,
        "distractors": Annotated[list[NonEmptyString], Field(min_length=2, max_length=2)],
    },
    "definition": {
        "term": NonEmptyString, "reference_answer": NonEmptyString,
        "solution_certificate": SolutionCertificate,
    },
    "why": {
        "question": NonEmptyString, "reference_answer": NonEmptyString,
        "solution_certificate": SolutionCertificate,
    },
    "equation": {
        "equation": NonEmptyString,
        "solution_steps": Annotated[list[NonEmptyString], Field(min_length=1)],
        "final_answer": NonEmptyString,
        "solution_certificate": SolutionCertificate,
    },
    "word_problem": {
        "question": NonEmptyString,
        "solution_steps": Annotated[list[NonEmptyString], Field(min_length=1)],
        "final_answer": NonEmptyString,
        "solution_certificate": SolutionCertificate,
    },
    "essay": {
        "question": NonEmptyString, "reference_answer": NonEmptyString,
        "key_points": Annotated[list[NonEmptyString], Field(min_length=1)],
        "solution_certificate": SolutionCertificate,
    },
}


def _literal(values: list[str]):
    unique = tuple(dict.fromkeys(str(value) for value in values if str(value)))
    return Literal.__getitem__(unique) if unique else str


def _bounded_list(item_type, count: int):
    return Annotated[list[item_type], Field(min_length=count, max_length=count)]


def _up_to_list(item_type, count: int):
    return Annotated[list[item_type], Field(max_length=count)]


def _generation_item_model(qtype: str, slot_ids: list[str]):
    base = _GENERATION_ITEMS[qtype]
    return create_model(
        f"{base.__name__}ForSlots",
        __base__=base,
        slot_id=(_literal(slot_ids), ...),
    )


def generation_response_model(qtype: str, slot_ids: list[str]):
    item_model = _generation_item_model(qtype, slot_ids)
    if qtype == "mcq":
        unsuitable_model = create_model(
            "MCQTaskUnsuitableForSlots",
            __base__=MCQTaskUnsuitable,
            slot_id=(_literal(slot_ids), ...),
        )
        return create_model(
            f"{item_model.__name__}Response",
            __base__=StrictOutputModel,
            questions=(_up_to_list(item_model, len(slot_ids)), ...),
            unsuitable_tasks=(_up_to_list(unsuitable_model, len(slot_ids)), ...),
        )
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
    slot_ids = [
        f"m{model_number}_{qtype}_{index}"
        for model_number in range(1, num_models + 1)
        for qtype, count in tasks
        for index in range(1, count + 1)
    ]
    plan_item = create_model(
        "GroundedPlanItem",
        __base__=PlanItem,
        slot_id=(_literal(slot_ids), ...),
        source_chunk_id=(_literal(chunk_ids), ...),
        support_chunk_ids=(list[_literal(chunk_ids)], ...),
    )
    item_fields = {qtype: (_up_to_list(plan_item, count), ...) for qtype, count in tasks}
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
        assessment_profile=(AssessmentProfile, ...),
        planning_issues=(list[PlanningIssue], ...),
        exams=(_bounded_list(plan, num_models), ...),
    )


def plan_repair_response_model(slot_ids: list[str], chunk_ids: list[str]):
    item = create_model(
        "PlanSlotReplacement",
        __base__=PlanItem,
        slot_id=(_literal(slot_ids), ...),
        source_chunk_id=(_literal(chunk_ids), ...),
        support_chunk_ids=(list[_literal(chunk_ids)], ...),
    )
    return create_model(
        "PlanSlotRepairResponse",
        __base__=StrictOutputModel,
        replacements=(_bounded_list(item, len(slot_ids)), ...),
    )


def validator_response_model(question_ids: list[str], *, quantitative: bool = False):
    base = QuantitativeValidatorVerdict if quantitative else ValidatorVerdict
    verdict = create_model(
        "ValidatorVerdictForQuestions",
        __base__=base,
        question_id=(_literal(question_ids), ...),
    )
    return create_model(
        "ValidatorResponse",
        __base__=StrictOutputModel,
        verdicts=(_bounded_list(verdict, len(question_ids)), ...),
    )


def global_exam_review_response_model(question_ids: list[str]):
    verdict = create_model(
        "GlobalExamVerdictForQuestions",
        __base__=GlobalExamVerdict,
        question_id=(_literal(question_ids), ...),
        conflicting_question_ids=(list[_literal(question_ids)], ...),
    )
    return create_model(
        "GlobalExamReviewResponse",
        __base__=StrictOutputModel,
        verdicts=(_bounded_list(verdict, len(question_ids)), ...),
        conflict_groups=(list[GlobalConflictGroup], ...),
        exam_quality_findings=(list[ExamQualityFinding], ...),
    )


def repair_response_model(items: list[tuple[str, str, list[str]]]):
    """Build an exact, per-question patch contract from authorized fields."""
    repair_models = []
    for index, (question_id, qtype, fields) in enumerate(items, start=1):
        definitions = _REPAIR_PATCH_FIELDS[qtype]
        patch_fields = {
            field: (definitions[field], ...)
            for field in fields
            if field in definitions
        }
        if not patch_fields:
            raise ValueError(f"No authorized repair fields for {question_id}")
        patch_model = create_model(
            f"RepairPatch{index}", __base__=StrictOutputModel, **patch_fields,
        )
        repair_models.append(
            create_model(
                f"RepairItem{index}",
                __base__=StrictOutputModel,
                question_id=(_literal([question_id]), ...),
                patch=(patch_model, ...),
            )
        )
    item_type = Union[tuple(repair_models)] if len(repair_models) > 1 else repair_models[0]
    return create_model(
        "QuestionRepairResponse",
        __base__=StrictOutputModel,
        repairs=(_bounded_list(item_type, len(items)), ...),
    )
