from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END


class ExamState(TypedDict):
    # --- Request -------------------------------------------------------
    document_id: str
    question_type: str          # "mcq" | "true_false" | "short_answer"
    number_of_questions: int
    # --- Retrieval ------------------------------------------------------
    retrieved_chunks: list[dict[str, Any]]
    context: str
    # --- Generation -----------------------------------------------------
    questions: list[dict[str, Any]]
    rejection_feedback: Optional[str]   # reasons earlier attempts were rejected
    # --- Result ----------------------------------------------------------
    exam_markdown: str
    error: Optional[str]


def build_exam_graph() -> StateGraph:
    from app.online.retrieval import retrieve_context
    from app.online.generator import generate_questions
    from app.online.assemble import return_result
    from app.online.receive import receive_request

    builder = StateGraph(ExamState)

    builder.add_node("receive_request", receive_request)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("return_result", return_result)

    builder.set_entry_point("receive_request")
    builder.add_edge("receive_request", "retrieve_context")
    builder.add_edge("retrieve_context", "generate_questions")
    builder.add_edge("generate_questions", "return_result")
    builder.add_edge("return_result", END)

    return builder.compile()


_exam_graph = None


def get_exam_graph() -> StateGraph:
    global _exam_graph
    if _exam_graph is None:
        _exam_graph = build_exam_graph()
    return _exam_graph
