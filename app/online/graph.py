from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.online.validator import MAX_QUESTION_REPAIR_ATTEMPTS, REPAIR_ACTIONS
from app.online.eval_stats import PipelineEvalStats


class ExamState(TypedDict):
    # --- Request -------------------------------------------------------
    document_id: str
    tasks: list[tuple[str, int]]  # [(question_type, count), ...]
    num_models: int
    selected_child_ids: Optional[list[str]]
    session_id: Optional[str]
    index_job_id: Optional[str]
    processed_document_key: Optional[str]
    domain_type: str
    subject: str
    difficulty: Optional[str]
    document_language: str  # language detected from the uploaded content ("en" | "ar")
    # --- Retrieval (once) ----------------------------------------------
    retrieved_chunks: list[dict[str, Any]]
    context: str  # selected-content fallback, capped by deployment prompt budget
    planner_context: str  # selected content for one planner call, capped at 120k
    assessment_catalog: dict[str, Any]
    assessment_catalog_key: str
    assessment_catalog_cache_hit: bool
    assessment_compiler_cache_hit: bool
    # --- Planning -------------------------------------------------------
    plans: list[dict[str, Any]]  # one plan per model
    assessment_profile: dict[str, Any]  # one compact analysis shared by all models
    planning_issues: list[dict[str, Any]]  # unsupported requested forms/counts
    plan_errors: list[str]  # validation failures from the last attempt
    plan_attempts: int
    unrepairable_plan_slots: list[dict[str, Any]]  # omitted slots with repair reasons
    plan_review_history: list[dict[str, Any]]
    plan_review_attempts: int
    # --- Generation -----------------------------------------------------
    generated_exams: list[dict[str, Any]]  # one result per model, accumulates cross-model memory
    rejection_feedback: Optional[str]
    mcq_task_reassignments: list[dict[str, Any]]
    # --- Post-generation validation / repair ----------------------------
    validation_reports: list[dict[str, Any]]  # one report per validated model
    validated_models: list[int]  # models that fully passed (skip re-validation)
    question_repair_attempts: dict[str, int]  # per-slot repair budget already used
    repair_manifest: list[dict[str, Any]]
    repair_candidates: list[dict[str, Any]]
    repair_retry_seeds: dict[str, dict[str, Any]]
    repair_replacement_memory: dict[str, dict[str, Any]]
    fitb_recovery_history: list[dict[str, Any]]
    pending_revalidation_ids: dict[int, list[str]]  # repaired IDs awaiting validation
    invalid_questions: list[dict[str, Any]]  # internal audit only; never rendered
    duplicate_repair_history: list[dict[str, Any]]  # semantic plan replacements and outcomes
    validation_repair_history: list[dict[str, Any]]  # all post-generation repair outcomes
    initial_global_reviews: list[dict[str, Any]]  # complete-exam AI reviews before repair
    final_global_reviews: list[dict[str, Any]]  # complete-exam AI reviews after changes
    exam_quality_findings: list[dict[str, Any]]  # persisted exam-level issues/warnings
    duplicate_reports: list[dict[str, Any]]  # actual-question pair adjudications by scan
    duplicate_similarity_distributions: list[dict[str, Any]]
    pending_duplicate_groups: list[dict[str, Any]]
    duplicate_repair_rounds: int  # complete exam-wide scan/repair cycles already used
    duplicate_replacement_attempts: dict[str, int]  # "model:slot" -> replacements tried
    duplicate_candidate_blacklist: dict[str, list[str]]
    duplicate_replacement_failures: dict[str, list[dict[str, Any]]]
    unresolved_duplicate_conflicts: list[dict[str, Any]]
    duplicate_scan_count: int
    duplicate_fact_cache: dict[str, dict[str, str]]  # content-keyed Luna facts for unchanged pairs
    slot_lifecycle: dict[str, dict[str, Any]]
    developer_diagnostics: list[dict[str, Any]]
    eval_stats: PipelineEvalStats  # JSON-safe V1 pipeline telemetry
    # --- Warnings / Result ----------------------------------------------
    warnings: list[str]
    error: Optional[str]
    progress_callback: Optional[Callable[[str, int], None]]
    audit_callback: Optional[Callable[[str, str, dict[str, Any]], None]]


_AUDIT_NODE_EVENTS = {
    "prepare_assessment_catalog": ("planning", "assessment_catalog_compiled"),
    "plan_exams": ("planning", "plan_completed"),
    "repair_plan": ("planning", "plan_repaired"),
    "review_plan": ("plan_review", "plan_review_completed"),
    "generate_exams_node": ("generation", "questions_generated"),
    "validate_generated_questions": ("general_validation", "validation_completed"),
    "repair_invalid_questions": ("repair", "repair_completed"),
    "scan_exam_duplicates": ("duplicate_scan", "duplicate_scan_completed"),
    "repair_duplicate_groups": ("duplicate_replacement", "duplicate_replacement_completed"),
    "assemble_exams_node": ("finalization", "exam_assembled"),
}


def _audit_node_output(
    state: ExamState, node_name: str, result: dict[str, Any],
    before: dict[str, Any] | None = None,
) -> None:
    callback = state.get("audit_callback")
    event = _AUDIT_NODE_EVENTS.get(node_name)
    if not callback or not event:
        return
    stage, event_type = event
    payload: dict[str, Any] = {"node_output": result}
    if before:
        payload["stage_input"] = before
    if node_name in {
        "review_plan", "generate_exams_node", "validate_generated_questions",
        "repair_invalid_questions", "scan_exam_duplicates",
        "repair_duplicate_groups", "assemble_exams_node",
    }:
        payload["plans_after_stage"] = deepcopy(
            result.get("plans", state.get("plans") or [])
        )
        payload["generated_exams_after_stage"] = deepcopy(
            result.get("generated_exams", state.get("generated_exams") or [])
        )
    if node_name == "repair_invalid_questions":
        payload["repair_input_verdicts"] = state.get("validation_reports") or []
    callback(stage, event_type, payload)
    if node_name == "validate_generated_questions":
        quantitative = [
            report.get("quantitative_review")
            for report in (result.get("validation_reports") or [])
            if report.get("quantitative_review")
        ]
        if quantitative:
            callback(
                "quantitative_validation",
                "quantitative_validation_completed",
                {"reports": quantitative},
            )


def _with_progress(node, stage: str, progress: int, *, after_repair: bool = False):
    """Report the real graph step before running a node."""
    def wrapped(state: ExamState):
        callback = state.get("progress_callback")
        if callback:
            current_stage = stage
            current_progress = progress
            if after_repair and state.get("pending_revalidation_ids"):
                current_stage = "validating_updates"
                current_progress = 95
            callback(current_stage, current_progress)
        before = None
        if state.get("audit_callback") and node.__name__ in _AUDIT_NODE_EVENTS:
            before = deepcopy({
                "plans": state.get("plans") or [],
                "generated_exams": state.get("generated_exams") or [],
                "validation_reports": state.get("validation_reports") or [],
                "planning_issues": state.get("planning_issues") or [],
                "duplicate_reports": state.get("duplicate_reports") or [],
                "duplicate_fact_cache": state.get("duplicate_fact_cache") or {},
            })
        try:
            result = node(state)
        except Exception as exc:
            from app.online.lifecycle import developer_diagnostic
            diagnostic = developer_diagnostic(
                exc,
                stage=_AUDIT_NODE_EVENTS.get(node.__name__, (stage, ""))[0],
                node=node.__name__,
            )
            audit_callback = state.get("audit_callback")
            if audit_callback:
                audit_callback(
                    diagnostic["stage"], "stage_exception", diagnostic
                )
            raise
        from app.online.lifecycle import reconcile_slot_lifecycle
        merged_state = {**state, **result}
        result["slot_lifecycle"] = reconcile_slot_lifecycle(
            state.get("slot_lifecycle") or {},
            merged_state,
            stage=_AUDIT_NODE_EVENTS.get(node.__name__, (stage, ""))[0],
        )
        _audit_node_output(state, node.__name__, result, before)
        return result

    return wrapped


def build_exam_graph() -> StateGraph:
    from app.online.retrieval import retrieve_context
    from app.online.assessment_catalog import prepare_assessment_catalog
    from app.online.planner import plan_exams, repair_plan
    from app.online.plan_review import review_plan
    from app.online.exam_builder import generate_exams_node, assemble_exams_node
    from app.online.receive import receive_request
    from app.online.validator import (
        validate_generated_questions,
        repair_invalid_questions,
    )
    from app.online.duplicate_pipeline import scan_exam_duplicates, repair_duplicate_groups

    builder = StateGraph(ExamState)

    builder.add_node("receive_request", _with_progress(receive_request, "analyzing_request", 5))
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node(
        "prepare_assessment_catalog",
        _with_progress(prepare_assessment_catalog, "compiling_assessment_catalog", 20),
    )
    builder.add_node("plan_exams", _with_progress(plan_exams, "planning_exam", 25))
    builder.add_node("repair_plan", _with_progress(repair_plan, "planning_exam", 25))
    builder.add_node("review_plan", _with_progress(review_plan, "reviewing_plan", 35))
    builder.add_node("generate_exams", _with_progress(generate_exams_node, "generating_exam", 50))
    builder.add_node(
        "validate_generated_questions",
        _with_progress(
            validate_generated_questions, "validating_exam", 75, after_repair=True
        ),
    )
    builder.add_node(
        "repair_invalid_questions",
        _with_progress(repair_invalid_questions, "repairing_exam", 90),
    )
    builder.add_node(
        "scan_exam_duplicates",
        _with_progress(scan_exam_duplicates, "checking_duplicates", 96),
    )
    builder.add_node(
        "repair_duplicate_groups",
        _with_progress(repair_duplicate_groups, "repairing_duplicates", 97),
    )
    builder.add_node(
        "assemble_exams",
        _with_progress(assemble_exams_node, "finalizing_exam", 99),
    )

    builder.set_entry_point("receive_request")
    builder.add_edge("receive_request", "retrieve_context")
    builder.add_edge("retrieve_context", "prepare_assessment_catalog")
    builder.add_edge("prepare_assessment_catalog", "plan_exams")
    builder.add_conditional_edges(
        "plan_exams",
        _route_after_plan,
        {"repair_plan": "repair_plan", "review_plan": "review_plan"},
    )
    builder.add_conditional_edges(
        "repair_plan",
        _route_after_plan,
        {"repair_plan": "repair_plan", "review_plan": "review_plan"},
    )
    builder.add_edge("review_plan", "generate_exams")
    builder.add_edge("generate_exams", "validate_generated_questions")
    builder.add_conditional_edges(
        "validate_generated_questions",
        _route_after_validation,
        {
            "repair_invalid_questions": "repair_invalid_questions",
            "scan_exam_duplicates": "scan_exam_duplicates",
        },
    )
    builder.add_edge("repair_invalid_questions", "validate_generated_questions")
    builder.add_conditional_edges(
        "scan_exam_duplicates",
        _route_after_duplicate_scan,
        {
            "repair_duplicate_groups": "repair_duplicate_groups",
            "assemble_exams": "assemble_exams",
        },
    )
    builder.add_edge("repair_duplicate_groups", "validate_generated_questions")
    builder.add_edge("assemble_exams", END)

    return builder.compile()


def _route_after_plan(state: ExamState) -> str:
    """Cycle to repair while a plan is invalid within the bounded attempt budget."""
    from app.online.planner import MAX_PLAN_ATTEMPTS

    if (
        state.get("plan_errors")
        and not state.get("planning_issues")
        and (state.get("plan_attempts") or 0) < MAX_PLAN_ATTEMPTS
    ):
        return "repair_plan"
    return "review_plan"


def _route_after_validation(state: ExamState) -> str:
    """Route unresolved requested slots through their own bounded budgets."""
    attempts = state.get("question_repair_attempts") or {}
    for target in state.get("repair_manifest") or []:
        if target.get("status") == "resolved":
            continue
        slot_id = str(target.get("slot_id") or "")
        if slot_id and (attempts.get(slot_id) or 0) < MAX_QUESTION_REPAIR_ATTEMPTS:
            return "repair_invalid_questions"
    # Compatibility for an in-flight/pre-manifest state. New validations always
    # populate the manifest; this prevents persisted older jobs from skipping a
    # repair solely because their state predates the contract.
    if not state.get("repair_manifest"):
        for report in state.get("validation_reports") or []:
            if any(v.get("action") in REPAIR_ACTIONS for v in report.get("verdicts") or []):
                return "repair_invalid_questions"
    return "scan_exam_duplicates"


def _route_after_duplicate_scan(state: ExamState) -> str:
    if state.get("pending_duplicate_groups"):
        return "repair_duplicate_groups"
    return "assemble_exams"


exam_graph = None


def get_exam_graph() -> StateGraph:
    global exam_graph
    if exam_graph is None:
        exam_graph = build_exam_graph()
    return exam_graph
