from __future__ import annotations

import time

from app.config import LMS_MODEL
from app.llm.client import LMStudioClient
from app.llm.json_utils import JSONExtractionError
from app.logging_conf import get_logger
from app.online.graph import ExamState
from app.online.models import parse_questions_obj
from app.online.prompts import build_prompt

logger = get_logger("GENERATOR")


def generate_questions(state: ExamState) -> dict:
    qtype = state["question_type"]
    count = state["number_of_questions"]
    context = state.get("context", "")

    if state.get("error") or not context:
        logger.warning("Skipping generation | document has no context")
        return {"questions": [], "error": state.get("error") or "No context available"}

    t0 = time.perf_counter()
    logger.info(
        "Generation started | model=%s | type=%s | count=%d",
        LMS_MODEL,
        qtype,
        count,
    )

    system_prompt, user_prompt = build_prompt(qtype, count, context)
    feedback = state.get("rejection_feedback") or ""
    if feedback:
        user_prompt += (
            "\n\n## Feedback from previous attempts\n"
            "Your earlier questions were rejected for the reasons below. "
            "Do NOT repeat these mistakes. Fix the issues and produce valid questions.\n"
            f"{feedback}"
        )
    client = LMStudioClient()

    try:
        parsed = client.chat_json(
            user_prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=max(2048, count * 256),
        )
        questions = parse_questions_obj(qtype, parsed)
    except (JSONExtractionError, ValueError) as exc:
        logger.error(
            "Generation failed | type=%s | count=%d | exc=%s: %s",
            qtype,
            count,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return {
            "questions": [],
            "error": f"LLM returned no valid {qtype} questions",
        }

    if not questions:
        logger.error("Generation failed | type=%s | count=%d | empty result", qtype, count)
        return {
            "questions": [],
            "error": f"LLM returned no valid {qtype} questions",
        }

    # Truncate to the requested count if the model over-generated.
    questions = questions[:count]

    elapsed = time.perf_counter() - t0
    logger.info(
        "Generation completed | type=%s | generated=%d | time=%.2fs",
        qtype,
        len(questions),
        elapsed,
    )
    return {"questions": questions, "error": None}
