from __future__ import annotations

from app.logging_conf import get_logger
from app.online.graph import ExamState

logger = get_logger("RECEIVE")

VALID_TYPES = {"mcq", "true_false", "short_answer"}
MAX_PER_TYPE = 100


def receive_request(state: ExamState) -> dict:
    logger.info(
        "Request received | document_id=%s | question_type=%s | count=%d",
        state.get("document_id"),
        state.get("question_type"),
        state.get("number_of_questions"),
    )

    document_id = state.get("document_id")
    qtype = state.get("question_type")
    count = state.get("number_of_questions")

    if not document_id:
        return {"error": "document_id is required", "questions": []}
    if qtype not in VALID_TYPES:
        return {
            "error": f"Unsupported question_type '{qtype}'. Supported: {sorted(VALID_TYPES)}",
            "questions": [],
        }
    if not isinstance(count, int) or count < 1 or count > MAX_PER_TYPE:
        return {
            "error": f"number_of_questions must be an integer between 1 and {MAX_PER_TYPE}",
            "questions": [],
        }

    return {}
