from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api import repositories
from app.llm import deepseek
from app.llm.schemas import generation_response_model
from app.online import exam_builder, prompts
from app.online.models import contains_forbidden_phrase


def _variation(question_type: str, variation_id: str = "variation-selected") -> dict:
    return {
        "variation_id": variation_id,
        "variation_group_id": "group-1",
        "question_types": [question_type],
        "operation": "evaluate_constraints",
        "target": "feasibility",
        "givens": ["7 <= c <= 14", "8 <= p <= 12", "c + p <= 23"],
        "reasoning_mode": "constraint_testing",
        "student_work_product": "true_or_false_decision",
        "evidence_chunk_ids": ["chunk-1"],
    }


def _plan(question_type: str) -> dict:
    variation = _variation(question_type)
    return {
        "task_id": "task-1",
        "task_family_id": "family-1",
        "concept_id": "constraints",
        "task_operation": "evaluate_constraints",
        "target": "feasibility",
        "reasoning_mode": "constraint_testing",
        "difficulty": "medium",
        "required_givens": variation["givens"],
        "variable_givens": ["c", "p"],
        "source_chunk_id": "chunk-1",
        "selected_variation_id": variation["variation_id"],
        "selected_variation_group_id": variation["variation_group_id"],
        "permitted_variations": [variation],
        "allowed_variations": [variation, _variation(question_type, "variation-unused")],
    }


@pytest.mark.parametrize("question_type", ["mcq", "true_false"])
def test_objective_types_bind_matching_selected_variation(question_type):
    plan = _plan(question_type)
    question = {"selected_variation_id": "variation-selected"}

    assert exam_builder._bind_selected_variation(question, plan, question_type) is True
    assert question["task_id"] == "task-1"
    assert question["selected_variation_id"] == "variation-selected"
    assert question["task_operation"] == "evaluate_constraints"
    assert question["target"] == "feasibility"
    assert question["required_givens"] == plan["required_givens"]
    assert question["effective_evidence_chunk_ids"] == ["chunk-1"]


@pytest.mark.parametrize("question_type", ["mcq", "true_false"])
def test_objective_types_reject_mismatching_selected_variation(question_type):
    plan = _plan(question_type)
    question = {"selected_variation_id": "variation-unused"}

    assert exam_builder._bind_selected_variation(question, plan, question_type) is False


def test_objective_bundle_retries_existing_call_after_variation_mismatch(monkeypatch):
    mcq_plan = {**_plan("mcq"), "slot_id": "m1_mcq_1", "assigned_sources": {"chunk-1": "evidence"}}
    tf_plan = {**_plan("true_false"), "slot_id": "m1_true_false_1", "assigned_sources": {"chunk-1": "evidence"}}
    calls = []

    def response(variation_id):
        return {
            "mcq": {"questions": [{
                "slot_id": "m1_mcq_1", "selected_variation_id": variation_id,
                "question": "Which pair satisfies all three stated inequalities?",
                "options": {"A": "(12, 10)", "B": "(15, 10)", "C": "(12, 13)", "D": "(6, 8)"},
                "correct_answer": "A", "solution_certificate": None,
            }], "unsuitable_tasks": []},
            "true_false": {"questions": [{
                "slot_id": "m1_true_false_1", "selected_variation_id": variation_id,
                "statement": "If 7 <= c <= 14, 8 <= p <= 12, and c+p <= 23, then c=12 and p=10 is feasible.",
                "answer": "True", "solution_certificate": None,
            }]},
        }

    def fake_request(*_args, **_kwargs):
        calls.append(1)
        return response("variation-unused" if len(calls) == 1 else "variation-selected")

    monkeypatch.setattr("app.llm.factory.create_llm_client", lambda: object())
    monkeypatch.setattr(exam_builder, "request_structured", fake_request)
    bundle, warnings = exam_builder._generate_obj_bundle(
        {"mcq": [mcq_plan], "true_false": [tf_plan]}, "", "medium", 1,
        [], set(), [], language="en",
        global_plan_by_slot={"m1_mcq_1": mcq_plan, "m1_true_false_1": tf_plan},
    )

    assert len(calls) == 2
    assert len(bundle["mcq"]) == 1
    assert len(bundle["true_false"]) == 1
    assert bundle["mcq"][0]["task_id"] == "task-1"
    assert bundle["true_false"][0]["selected_variation_id"] == "variation-selected"
    assert sum("invalid or incompatible selected variation" in warning for warning in warnings) == 2


def test_true_false_prompt_requires_student_facing_completeness():
    rules = prompts.TRUE_FALSE_RULES
    assert "Every condition needed" in rules
    assert "student-facing statement" in rules
    assert "numerical constraints, inequalities" in rules
    assert "solution certificate" in rules
    assert "without the source document" in rules


def test_planned_item_renders_only_selected_variation_contract():
    plan = {
        **_plan("true_false"), "slot_id": "m1_true_false_1",
        "required_answer_polarity": "True",
    }
    rendered = prompts._format_planned_item(1, plan)

    assert "selected_variation_id=variation-selected" in rendered
    assert "selected_variation=" in rendered
    assert "variation-unused" not in rendered
    assert "allowed_variations=" not in rendered
    assert "permitted_variations=" not in rendered
    for required in (
        "task_id=task-1", "task_operation=evaluate_constraints",
        "target=feasibility", "difficulty=medium", "required_givens=",
        "reasoning_mode=constraint_testing",
    ):
        assert required in rendered
    assert "required_answer_polarity=True" in rendered


def test_true_false_polarity_mismatch_retries_with_explicit_feedback(monkeypatch):
    plan = {
        **_plan("true_false"), "slot_id": "m1_true_false_3",
        "required_answer_polarity": "True",
        "assigned_sources": {"chunk-1": "evidence"},
    }
    prompts_seen: list[str] = []

    def fake_request(_client, user, **_kwargs):
        prompts_seen.append(user)
        answer = False if len(prompts_seen) == 1 else True
        return {
            "mcq": {"questions": [], "unsuitable_tasks": []},
            "true_false": {"questions": [{
                "slot_id": "m1_true_false_3",
                "selected_variation_id": "variation-selected",
                "statement": (
                    "The selected values satisfy all stated inequalities."
                    if answer else "The selected values violate a stated inequality."
                ),
                "answer": answer,
                "solution_certificate": None,
            }]},
        }

    monkeypatch.setattr("app.llm.factory.create_llm_client", lambda: object())
    monkeypatch.setattr(exam_builder, "request_structured", fake_request)
    bundle, warnings = exam_builder._generate_obj_bundle(
        {"mcq": [], "true_false": [plan]}, "", "medium", 1,
        [], set(), [], language="en", global_plan_by_slot={"m1_true_false_3": plan},
    )

    assert len(prompts_seen) == 2
    assert "required_answer_polarity=True" in prompts_seen[0]
    assert "previous candidate was rejected" in prompts_seen[1]
    assert bundle["true_false"][0]["answer"] is True
    assert any("required deterministic polarity True" in warning for warning in warnings)


@pytest.mark.parametrize(
    "text",
    [
        "وفق القيود، فإن تخصيص 12 ساعة للتفاضل و10 ساعات للفيزياء يحقق جميع القيود.",
        "حسب المعطيات، تكون الإجابة صحيحة.",
        "كما ورد في النص، تزداد القيمة.",
        "بالاعتماد على المعلومات السابقة احسب الناتج.",
    ],
)
def test_arabic_external_context_references_are_rejected(text):
    assert contains_forbidden_phrase(text, "ar") is True


@pytest.mark.parametrize(
    "text",
    [
        "وفق القيود التالية: 7 ≤ c ≤ 14، و8 ≤ p ≤ 12، وc + p ≤ 23، فإن c=12 وp=10 حل ممكن.",
        "حسب المعطيات التالية: الكتلة 5 كغ والسرعة 2 م/ث، احسب الزخم.",
        "إذا كان 7 ≤ c ≤ 14 و8 ≤ p ≤ 12، فهل c=12 وp=10 يحققان الشرطين؟",
    ],
)
def test_arabic_inline_context_is_not_rejected(text):
    assert contains_forbidden_phrase(text, "ar") is False


def _response(output_text: str, *, response_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        status="completed",
        output_text=output_text,
        incomplete_details=None,
        error=None,
        usage=SimpleNamespace(
            input_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
            output_tokens=5,
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
            total_tokens=15,
        ),
    )


def test_structured_retry_records_failed_and_successful_attempts(monkeypatch):
    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    client = deepseek.DeepSeekClient()
    responses = iter([
        _response('{"questions":[', response_id="provider-failed"),
        _response(
            '{"questions":[{"slot_id":"m1_true_false_1",'
            '"statement":"1 + 1 = 2","answer":"True"}]}',
            response_id="provider-success",
        ),
    ])

    async def fake_request_with_backoff(**_kwargs):
        return next(responses)

    events = []
    usages = []
    monkeypatch.setattr(client, "_request_with_backoff", fake_request_with_backoff)
    monkeypatch.setattr(client, "_record_request_event", lambda **values: events.append(values))
    monkeypatch.setattr(repositories, "record_llm_usage", lambda **values: usages.append(values))
    model = generation_response_model("true_false", ["m1_true_false_1"])

    result = client.chat_structured(
        "Generate one statement.", response_model=model,
        json_schema=model.model_json_schema(), schema_name="true_false_questions",
        agent="generator", item_count=1, expected_ids=["m1_true_false_1"],
        expected_id_field="slot_id",
    )

    assert result["questions"][0]["slot_id"] == "m1_true_false_1"
    assert len(client.last_structured_attempt_details) == 2
    failed, succeeded = client.last_structured_attempt_details
    assert failed["status"] == "failed"
    assert failed["failure_classification"] == "INVALID_JSON"
    assert failed["provider_request_id"] == "provider-failed"
    assert failed["input_tokens"] == 10
    assert failed["cached_tokens"] == 3
    assert failed["reasoning_tokens"] == 2
    assert succeeded["status"] == "success"
    assert succeeded["provider_request_id"] == "provider-success"
    assert len(usages) == 2  # both paid provider responses retain usage/cost
    assert [event["status"] for event in events] == ["failed", "success"]
    assert events[0]["metadata"]["raw_response_truncated"] is False


def test_structured_attempt_marks_missing_usage_unknown(monkeypatch):
    monkeypatch.setattr(deepseek, "DEEPSEEK_API_KEY", "test-key")
    client = deepseek.DeepSeekClient()
    response = _response("not json", response_id="provider-no-usage")
    response.usage = None

    async def fake_request_with_backoff(**_kwargs):
        return response

    events = []
    monkeypatch.setattr(client, "_request_with_backoff", fake_request_with_backoff)
    monkeypatch.setattr(client, "_record_request_event", lambda **values: events.append(values))
    model = generation_response_model("true_false", ["m1_true_false_1"])

    with pytest.raises(deepseek.DeepSeekValidationError):
        client.chat_structured(
            "Generate one statement.", response_model=model,
            json_schema=model.model_json_schema(), schema_name="true_false_questions",
            agent="assessment_compiler", item_count=1,
            expected_ids=["m1_true_false_1"], expected_id_field="slot_id",
        )

    assert events[0]["usage_available"] is False
    assert events[0]["input_tokens"] is None
    assert events[0]["estimated_cost_usd"] is None
