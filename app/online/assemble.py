from __future__ import annotations

import time

from app.logging_conf import get_logger
from app.online.graph import ExamState
from app.online.models import render_markdown

logger = get_logger("RESPONSE")


def return_result(state: ExamState) -> dict:
    t0 = time.perf_counter()
    qtype = state["question_type"]
    questions = state.get("questions") or []

    markdown = ""
    if questions:
        markdown = render_markdown(qtype, questions)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Exam generation completed | type=%s | questions=%d | total_time=%.2fs | success=%s",
        qtype,
        len(questions),
        elapsed,
        bool(questions),
    )
    return {"exam_markdown": markdown, "error": None}
