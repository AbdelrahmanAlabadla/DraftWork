from __future__ import annotations

import time
from typing import Any

from app.logging_conf import get_logger
from app.online.graph import get_exam_graph
from app.online.models import (
    TYPE_ORDER,
    contains_forbidden_phrase,
    normalize_text,
    parse_questions_obj,
    question_text,
    render_markdown,
)

logger = get_logger("EXAM_BUILDER")

_MAX_ATTEMPTS = 3

# Words that carry little topical signal and would otherwise inflate the
# similarity of unrelated questions.
_STOPWORDS = frozenset(
    (
        "a an and are as at be been but by can could did do does for from had has have he her "
        "his how i if in into is it its may me more most my no not of on or our out over she "
        "should so some such than that the their them then there these they this those to under "
        "up was we were what when where which who why will with would you your following "
        "best describes describe explain defines define list give giving name"
    ).split()
)

# Jaccard token-overlap above which two questions are treated as near-duplicates.
_NEAR_DUP_THRESHOLD = 0.35


def _is_duplicate(qtype: str, question: dict[str, Any], seen: set[tuple[str, str]]) -> bool:
    key = (qtype, " ".join(normalize_text(question_text(qtype, question))))
    if key in seen:
        return True
    seen.add(key)
    return False


def _content_tokens(qtype: str, question: dict[str, Any]) -> set[str]:
    """Meaningful tokens of a question, for near-duplicate checks.

    The MCQ options are intentionally excluded: distractors are often generic
    and would inflate the overlap of unrelated questions, while the stem alone
    reliably captures a reworded-same-concept MCQ. For short-answer questions
    the reference answer is a strong signal and is included.
    """
    parts = [question_text(qtype, question)]
    if qtype == "short_answer":
        parts.append(str(question.get("reference_answer") or ""))
    tokens: list[str] = []
    for part in parts:
        tokens.extend(normalize_text(part))
    return {tok for tok in tokens if tok not in _STOPWORDS}


def _token_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _is_near_duplicate(
    qtype: str, question: dict[str, Any], accepted: list[dict[str, Any]]
) -> bool:
    """True if the question overlaps enough with an already-accepted one."""
    tokens = _content_tokens(qtype, question)
    for other in accepted:
        if _token_overlap(tokens, _content_tokens(qtype, other)) >= _NEAR_DUP_THRESHOLD:
            return True
    return False


# --------------------------------------------------------------------------
# Generation (plan-driven, one call per model x question type)
# --------------------------------------------------------------------------
def _generate_type_from_plan(
    qtype: str,
    planned_items: list[dict[str, Any]],
    context: str,
    difficulty: str,
    model_number: int,
    seen: set[tuple[str, str]],
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate one question per planned item, retrying only missing concepts.

    Every generated question maps to a planned item; the generator is NEVER asked
    to invent unplanned concepts. If a retry still cannot fill a slot, the exam is
    simply short for that type (a warning), rather than degrading the planning.
    """
    from app.llm.client import LMStudioClient
    from app.online.prompts import build_prompt

    count = len(planned_items)
    accumulated: list[dict[str, Any]] = []
    warnings: list[str] = []
    client = LMStudioClient()

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        # 1:1 mapping by order: the first `len(accumulated)` plan items are done.
        remaining_plan = planned_items[len(accumulated):]
        if not remaining_plan:
            break

        system_prompt, user_prompt = build_prompt(
            qtype,
            len(remaining_plan),
            context,
            difficulty=difficulty,
            model_number=model_number,
            planned_items=remaining_plan,
        )

        feedback = _build_feedback(qtype, accumulated, seen, within_model, previous_exams)
        if feedback:
            user_prompt += (
                "\n\n## Context from earlier attempts\n"
                "Below are questions already accepted that you must NOT repeat or "
                "reword. Produce the planned questions that are still missing only.\n"
                f"{feedback}"
            )

        try:
            batch = parse_questions_obj(
                qtype,
                client.chat_json(
                    user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.5,
                    max_tokens=max(2048, len(remaining_plan) * 256),
                ),
            )
        except Exception as exc:
            warnings.append(
                f"{qtype} attempt {attempt}: LLM returned no valid questions ({exc})"
            )
            continue

        valid: list[dict[str, Any]] = []
        for q in batch[: len(remaining_plan)]:
            text = question_text(qtype, q)
            if contains_forbidden_phrase(text):
                logger.warning("Rejected (forbidden phrase) | type=%s | text=%r", qtype, text[:120])
                continue
            if _is_duplicate(qtype, q, seen):
                logger.warning("Rejected (duplicate) | type=%s | text=%r", qtype, text[:120])
                continue
            if _is_near_duplicate(
                qtype, q, [*valid, *accumulated, *within_model, *previous_exams]
            ):
                logger.warning("Rejected (near-duplicate) | type=%s | text=%r", qtype, text[:120])
                continue
            valid.append(q)

        accumulated.extend(valid)
        logger.info(
            "Plan type attempt | type=%s | model=%d | attempt=%d/%d | planned=%d | accepted=%d/%d",
            qtype,
            model_number,
            attempt,
            _MAX_ATTEMPTS,
            count,
            len(accumulated),
            count,
        )

    if len(accumulated) < count:
        warnings.append(f"{qtype}: {len(accumulated)}/{count} generated after attempts")
        logger.warning("Plan type incomplete | type=%s | got=%d/%d", qtype, len(accumulated), count)
    return accumulated[: count], warnings


def _build_feedback(
    qtype: str,
    accumulated: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    within_model: list[dict[str, Any]],
    previous_exams: list[dict[str, Any]],
) -> str | None:
    """Describe accepted questions for the retry so the model only fills the gap."""
    parts: list[str] = []
    accepted = [*accumulated, *within_model, *previous_exams]
    if accepted:
        lines = "\n".join(
            f"- {question_text(qtype, q)}" for q in accepted if question_text(qtype, q)
        )
        parts.append(
            "These questions were already accepted (in this exam or earlier models). "
            f"Do NOT repeat or reword them; do not test the same concept:\n{lines}"
        )
    return "\n\n".join(parts) if parts else None


# --------------------------------------------------------------------------
# Node: generate every exam model (one call per model x type)
# --------------------------------------------------------------------------
def generate_exams_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate `num_models` complete, distinct exams from the shared plan.

    Cross-model uniqueness is secured primarily by the planner distributing
    distinct concepts; the near-duplicate text check runs across all previously
    generated models as a secondary safeguard.
    """
    from app.online.planner import fill_missing_concepts

    document_id = state.get("document_id")
    tasks = state.get("tasks") or []
    num_models = state.get("num_models") or 1
    difficulty = state.get("difficulty") or "mix"
    context = state.get("context") or ""
    plans = state.get("plans") or []
    warnings: list[str] = list(state.get("warnings") or [])

    # Never free-fill during generation: first repair any remaining plan shortfall.
    if not state.get("plan_errors"):
        plans, fill_warnings = fill_missing_concepts(
            plans, tasks, state.get("planner_context") or ""
        )
        warnings.extend(fill_warnings)
    else:
        warnings.append("Planner could not produce a fully valid plan; using best effort.")

    generated_exams: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    previous_questions: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    for model_number in range(1, num_models + 1):
        plan = plans[model_number - 1] if model_number - 1 < len(plans) else None
        plan_items = (plan or {}).get("items") or {}
        model_warnings: list[str] = []
        questions: dict[str, list[dict[str, Any]]] = {}
        within_model: list[dict[str, Any]] = []

        for qtype, _count in tasks:
            planned = plan_items.get(qtype) or []
            if not planned:
                model_warnings.append(f"{qtype}: no planned concepts; skipped.")
                continue
            questions[qtype], type_warnings = _generate_type_from_plan(
                qtype,
                planned,
                context,
                difficulty,
                model_number,
                seen,
                within_model,
                previous_questions,
            )
            model_warnings.extend(type_warnings)
            within_model.extend(questions[qtype])

        elapsed = time.perf_counter() - t0
        total = sum(len(v) for v in questions.values())
        logger.info(
            "Exam generated | document_id=%s | model=%d | types=%s | total=%d | time=%.2fs",
            document_id,
            model_number,
            list(questions.keys()),
            total,
            elapsed,
        )
        generated_exams.append(
            {
                "model_number": model_number,
                "questions": questions,
                "warnings": model_warnings,
            }
        )
        previous_questions.extend(within_model)

    warnings.extend(generated_exams[-1]["warnings"] if generated_exams else [])
    return {"generated_exams": generated_exams, "warnings": warnings}


def assemble_exams_node(state: dict[str, Any]) -> dict[str, Any]:
    """Render each generated exam's questions to markdown."""
    generated_exams = state.get("generated_exams") or []
    for exam in generated_exams:
        exam["markdown"] = assemble_exam(exam["questions"])
    return {"generated_exams": generated_exams}


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def generate_exams(
    document_id: str,
    tasks: list[tuple[str, int]],
    num_models: int,
    selected_child_ids: list[str] | None = None,
    difficulty: str = "mix",
) -> dict[str, Any]:
    """Generate ``num_models`` complete, distinct exam versions.

    Runs the two-phase LangGraph workflow ONCE: retrieval, planning (one call for
    all models), then plan-driven generation per model x question type. Returns
    {"exams": [{"model_number", "questions", "markdown", "warnings"}], "warnings"}.
    """
    t0 = time.perf_counter()
    initial = {
        "document_id": document_id,
        "tasks": list(tasks),
        "num_models": num_models,
        "selected_child_ids": selected_child_ids,
        "difficulty": difficulty,
        "retrieved_chunks": [],
        "context": "",
        "planner_context": "",
        "plans": [],
        "plan_errors": [],
        "plan_attempts": 0,
        "generated_exams": [],
        "rejection_feedback": None,
        "warnings": [],
        "error": None,
    }
    result = get_exam_graph().invoke(initial)

    error = result.get("error")
    if error:
        logger.warning("Exam workflow error | document_id=%s | error=%s", document_id, error)
        return {"exams": [], "warnings": [str(error)]}

    exams = result.get("generated_exams") or []
    warnings = result.get("warnings") or []
    elapsed = time.perf_counter() - t0
    logger.info(
        "Multi-exam generation | document_id=%s | models=%d | difficulty=%s | total_time=%.2fs",
        document_id,
        num_models,
        difficulty,
        elapsed,
    )
    return {"exams": exams, "warnings": warnings}


def generate_exam(
    document_id: str,
    tasks: list[tuple[str, int]],
    selected_child_ids: list[str] | None = None,
    difficulty: str = "mix",
    model_number: int = 1,
) -> dict[str, Any]:
    """Generate one complete exam (single model). Kept for scripts/tests."""
    result = generate_exams(
        document_id, tasks, num_models=1, selected_child_ids=selected_child_ids, difficulty=difficulty
    )
    exam = result["exams"][0] if result["exams"] else None
    return {
        "questions": (exam or {}).get("questions", {}),
        "warnings": result["warnings"],
    }


def assemble_exam(questions: dict[str, list[dict[str, Any]]]) -> str:
    """Render the final exam in canonical order with continuous numbering.

    The application owns structure: ordering, numbering, and markdown.
    """
    sections: list[str] = []
    counter = 1
    for qtype in TYPE_ORDER:
        qs = questions.get(qtype) or []
        if not qs:
            continue
        sections.append(render_markdown(qtype, qs, start_index=counter))
        counter += len(qs)
    return "\n\n---\n\n".join(sections)