from __future__ import annotations

import time
from typing import Any

from app.logging_conf import get_logger
from app.online.graph import get_exam_graph
from app.online.models import (
    TYPE_ORDER,
    contains_forbidden_phrase,
    normalize_text,
    question_text,
    render_markdown,
)

logger = get_logger("EXAM_BUILDER")

_MAX_ATTEMPTS = 3


def _is_duplicate(qtype: str, question: dict[str, Any], seen: set[tuple[str, str]]) -> bool:
    key = (qtype, " ".join(normalize_text(question_text(qtype, question))))
    if key in seen:
        return True
    seen.add(key)
    return False


def _generate_type(
    document_id: str,
    qtype: str,
    count: int,
    seen: set[tuple[str, str]],
    selected_child_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate up to `count` valid questions for one type, retrying as needed."""
    graph = get_exam_graph()
    accumulated: list[dict[str, Any]] = []
    warnings: list[str] = []
    rejected_phrases = 0
    rejected_dups = 0

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        remaining = count - len(accumulated)
        if remaining <= 0:
            break

        state = graph.invoke(
            {
                "document_id": document_id,
                "question_type": qtype,
                "number_of_questions": remaining,
                "selected_child_ids": selected_child_ids,
            }
        )
        if state.get("error"):
            warnings.append(f"{qtype} attempt {attempt}: {state['error']}")
            continue

        batch = state.get("questions") or []
        valid: list[dict[str, Any]] = []
        for q in batch:
            text = question_text(qtype, q)
            if contains_forbidden_phrase(text):
                rejected_phrases += 1
                logger.warning(
                    "Rejected question (forbidden phrase) | type=%s | text=%r",
                    qtype,
                    text[:120],
                )
                continue
            if _is_duplicate(qtype, q, seen):
                rejected_dups += 1
                logger.warning(
                    "Rejected question (duplicate) | type=%s | text=%r",
                    qtype,
                    text[:120],
                )
                continue
            valid.append(q)

        accumulated.extend(valid)
        logger.info(
            "Type attempt | type=%s | attempt=%d/%d | returned=%d | accepted=%d | accumulated=%d/%d | rejected_phrases=%d | rejected_dups=%d",
            qtype,
            attempt,
            _MAX_ATTEMPTS,
            len(batch),
            len(valid),
            len(accumulated),
            count,
            rejected_phrases,
            rejected_dups,
        )

    if len(accumulated) < count:
        warnings.append(
            f"{qtype}: {len(accumulated)}/{count} generated after {_MAX_ATTEMPTS} attempts"
        )
        logger.warning(
            "Type incomplete | type=%s | got=%d/%d | attempts=%d",
            qtype,
            len(accumulated),
            count,
            _MAX_ATTEMPTS,
        )
    return accumulated[:count], warnings


def generate_exam(
    document_id: str,
    tasks: list[tuple[str, int]],
    selected_child_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Generate all requested question types with validation/retries.

    ``selected_child_ids`` scopes retrieval to the chosen subsections; when
    empty/None the whole document is used. Returns
    {"questions": {qtype: [...]}, "warnings": [...]}.
    """
    t0 = time.perf_counter()
    questions: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()

    for qtype, count in tasks:
        questions[qtype], type_warnings = _generate_type(
            document_id, qtype, count, seen, selected_child_ids
        )
        warnings.extend(type_warnings)

    elapsed = time.perf_counter() - t0
    total = sum(len(v) for v in questions.values())
    logger.info(
        "Exam generation | document_id=%s | types=%s | total_questions=%d | total_time=%.2fs",
        document_id,
        list(questions.keys()),
        total,
        elapsed,
    )
    return {"questions": questions, "warnings": warnings}


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
