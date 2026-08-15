from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END


class ExamState(TypedDict):
    # --- Request -------------------------------------------------------
    document_id: str
    tasks: list[tuple[str, int]]        # [(question_type, count), ...]
    num_models: int
    selected_child_ids: Optional[list[str]]
    difficulty: Optional[str]
    # --- Retrieval (once) ----------------------------------------------
    retrieved_chunks: list[dict[str, Any]]
    context: str                        # FULL selected child content (generation)
    planner_context: str                # titles + short snippets (planning)
    # --- Planning -------------------------------------------------------
    plans: list[dict[str, Any]]         # one plan per model
    plan_errors: list[str]              # validation failures from the last attempt
    plan_attempts: int
    # --- Generation -----------------------------------------------------
    generated_exams: list[dict[str, Any]]  # one result per model, accumulates cross-model memory
    rejection_feedback: Optional[str]
    # --- Warnings / Result ----------------------------------------------
    warnings: list[str]
    error: Optional[str]


def build_exam_graph() -> StateGraph:
    from app.online.retrieval import retrieve_context
    from app.online.planner import plan_exams, repair_plan
    from app.online.exam_builder import generate_exams_node, assemble_exams_node
    from app.online.receive import receive_request

    builder = StateGraph(ExamState)

    builder.add_node("receive_request", receive_request)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("plan_exams", plan_exams)
    builder.add_node("repair_plan", repair_plan)
    builder.add_node("generate_exams", generate_exams_node)
    builder.add_node("assemble_exams", assemble_exams_node)

    builder.set_entry_point("receive_request")
    builder.add_edge("receive_request", "retrieve_context")
    builder.add_edge("retrieve_context", "plan_exams")
    builder.add_conditional_edges(
        "plan_exams",
        _route_after_plan,
        {"repair_plan": "repair_plan", "generate_exams": "generate_exams"},
    )
    builder.add_conditional_edges(
        "repair_plan",
        _route_after_plan,
        {"repair_plan": "repair_plan", "generate_exams": "generate_exams"},
    )
    builder.add_edge("generate_exams", "assemble_exams")
    builder.add_edge("assemble_exams", END)

    return builder.compile()


def _route_after_plan(state: ExamState) -> str:
    """Cycle to repair while a plan is invalid, up to 3 total planner attempts."""
    if state.get("plan_errors") and (state.get("plan_attempts") or 0) < 3:
        return "repair_plan"
    return "generate_exams"


exam_graph = None


def get_exam_graph() -> StateGraph:
    global exam_graph
    if exam_graph is None:
        exam_graph = build_exam_graph()
    return exam_graph