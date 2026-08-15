from __future__ import annotations

from app.logging_conf import get_logger
from app.online.graph import ExamState

logger = get_logger("RECEIVE")

VALID_TYPES = {"mcq", "true_false", "short_answer"}
MAX_PER_TYPE = 100
_NUM_MODELS_MIN = 1
_NUM_MODELS_MAX = 4
_VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard", "mix"})


def receive_request(state: ExamState) -> dict:
    document_id = state.get("document_id")
    tasks = state.get("tasks") or []
    num_models = state.get("num_models")
    difficulty = state.get("difficulty")
    selected_child_ids = state.get("selected_child_ids")

    logger.info(
        "Request received | document_id=%s | tasks=%s | models=%s",
        document_id,
        tasks,
        num_models,
    )

    if not document_id:
        return {"error": "document_id is required", "questions": []}

    if not tasks:
        return {"error": "At least one question type/count must be requested.", "questions": []}
    for qtype, count in tasks:
        if qtype not in VALID_TYPES:
            return {
                "error": f"Unsupported question_type '{qtype}'. Supported: {sorted(VALID_TYPES)}",
                "questions": [],
            }
        if not isinstance(count, int) or count < 1 or count > MAX_PER_TYPE:
            return {
                "error": f"question count must be an integer between 1 and {MAX_PER_TYPE}",
                "questions": [],
            }

    if not isinstance(num_models, int) or not (
        _NUM_MODELS_MIN <= num_models <= _NUM_MODELS_MAX
    ):
        return {
            "error": f"num_models must be an integer between {_NUM_MODELS_MIN} and {_NUM_MODELS_MAX}",
            "questions": [],
        }

    if difficulty not in _VALID_DIFFICULTIES:
        return {
            "error": f"Unsupported difficulty '{difficulty}'. Supported: {sorted(_VALID_DIFFICULTIES)}",
            "questions": [],
        }

    if not selected_child_ids:
        return {
            "error": (
                "Please choose at least one section topic to generate the exam. "
                "No content was selected."
            ),
            "questions": [],
        }

    return {}
