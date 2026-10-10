from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm import deepseek
from app.logging_conf import configure_logging
from app.offline.processed_cache import file_sha256, processed_document_key
from app.offline.vector_store import VectorStore
from app.online import assessment_catalog, duplicate_judge, exam_builder, planner, validator
from app.online.eval_stats import create_pipeline_eval
from app.online.graph import get_exam_graph


def safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return safe(dump())
    return repr(value)


def main() -> None:
    source_state_path = ROOT / "data/e2e_arch_validation/physics_work_energy_28_arabic/final_state.json"
    pdfs = list((ROOT / "data/storage").rglob("760cabfe-cf68-4f15-86f4-b9537ae13a3a/source.pdf"))
    if not pdfs:
        raise RuntimeError("Saved Physics PDF is unavailable")
    source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
    chunks = source_state["retrieved_chunks"]
    child_ids = source_state["selected_child_ids"]
    document_id = source_state["document_id"]
    processed_key = processed_document_key(file_sha256(pdfs[0]))
    output = ROOT / "data/e2e_arch_validation/physics_catalog_28_arabic"
    output.mkdir(parents=True, exist_ok=True)

    def write(name: str, value: Any) -> None:
        (output / name).write_text(
            json.dumps(safe(value), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    selected_by_id = {str(chunk["child_id"]): chunk for chunk in chunks}
    original_get = VectorStore.get_by_child_ids
    VectorStore.get_by_child_ids = lambda self, _document_id, ids, **kwargs: [
        copy.deepcopy(selected_by_id[value]) for value in ids if value in selected_by_id
    ]

    configure_logging("INFO")
    handler = logging.FileHandler(output / "generation.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    network_calls: list[dict[str, Any]] = []
    structured_calls: list[dict[str, Any]] = []
    original_network = deepseek.DeepSeekClient._request_with_backoff

    async def observed_network(self, *, timeout: int, **kwargs):
        started = time.perf_counter()
        record = {
            "index": len(network_calls) + 1,
            "agent_max_output_tokens": kwargs.get("max_output_tokens"),
            "input_chars": len(str(kwargs.get("input") or "")),
        }
        try:
            response = await original_network(self, timeout=timeout, **kwargs)
            record.update({"status": str(getattr(response, "status", "")), "usage": safe(getattr(response, "usage", None))})
            return response
        except Exception as exc:
            record["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["duration_seconds"] = round(time.perf_counter() - started, 3)
            network_calls.append(record)
            write("network_calls.json", network_calls)

    deepseek.DeepSeekClient._request_with_backoff = observed_network

    def observe(module_name: str, original):
        def wrapped(client, prompt, **kwargs):
            started = time.perf_counter()
            record = {
                "index": len(structured_calls) + 1, "module": module_name,
                "schema_name": kwargs.get("schema_name"), "agent": kwargs.get("agent"),
                "operation": kwargs.get("operation"), "item_count": kwargs.get("item_count"),
                "prompt_chars": len(prompt),
            }
            try:
                result = original(client, prompt, **kwargs)
                record["usage"] = safe(getattr(client, "last_usage", None))
                return result
            except Exception as exc:
                record["exception"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["duration_seconds"] = round(time.perf_counter() - started, 3)
                structured_calls.append(record)
                write("structured_calls.json", structured_calls)
        return wrapped

    assessment_catalog.request_structured = observe("assessment_catalog", assessment_catalog.request_structured)
    planner.request_structured = observe("planner", planner.request_structured)
    exam_builder.request_structured = observe("exam_builder", exam_builder.request_structured)
    validator.request_structured = observe("validator", validator.request_structured)
    duplicate_judge.request_structured = observe("duplicate_judge", duplicate_judge.request_structured)

    tasks = [
        ("mcq", 8), ("true_false", 5), ("definition", 4),
        ("why", 4), ("equation", 3), ("word_problem", 4),
    ]
    initial = {
        "document_id": document_id, "processed_document_key": processed_key,
        "tasks": tasks, "num_models": 1, "selected_child_ids": child_ids,
        "session_id": None, "index_job_id": None, "difficulty": "easy",
        "document_language": "en", "retrieved_chunks": [], "context": "", "planner_context": "",
        "assessment_catalog": {}, "assessment_catalog_key": "", "assessment_catalog_cache_hit": False,
        "assessment_compiler_cache_hit": False,
        "plans": [], "assessment_profile": {}, "planning_issues": [], "plan_errors": [],
        "plan_attempts": 0, "unrepairable_plan_slots": [], "generated_exams": [],
        "rejection_feedback": None, "validation_reports": [], "validated_models": [],
        "question_repair_attempts": {}, "pending_revalidation_ids": {}, "invalid_questions": [],
        "duplicate_repair_history": [], "validation_repair_history": [],
        "initial_global_reviews": [], "final_global_reviews": [], "exam_quality_findings": [],
        "duplicate_reports": [], "duplicate_similarity_distributions": [],
        "pending_duplicate_groups": [], "duplicate_repair_rounds": 0,
        "duplicate_replacement_attempts": {}, "duplicate_scan_count": 0,
        "eval_stats": create_pipeline_eval(tasks, 1), "warnings": [], "error": None,
        "progress_callback": lambda stage, progress: print(json.dumps({"stage": stage, "progress": progress}), flush=True),
    }
    state = copy.deepcopy({key: value for key, value in initial.items() if key != "progress_callback"})
    state["progress_callback"] = initial["progress_callback"]
    started = time.perf_counter()
    try:
        for update in get_exam_graph().stream(initial, stream_mode="updates"):
            for node_update in update.values():
                if isinstance(node_update, dict):
                    state.update(node_update)
    finally:
        VectorStore.get_by_child_ids = original_get
        state.pop("progress_callback", None)
        state["duration_seconds"] = round(time.perf_counter() - started, 3)
        write("final_state.json", state)
        write("network_calls.json", network_calls)
        write("structured_calls.json", structured_calls)
        exams = state.get("generated_exams") or []
        if exams:
            (output / "exam.md").write_text(
                exam_builder.assemble_exam(
                    exams[0].get("questions") or {},
                    language=state.get("document_language") or "ar",
                ),
                encoding="utf-8",
            )
    print(json.dumps({
        "duration_seconds": state["duration_seconds"],
        "catalog_cache_hit": state.get("assessment_catalog_cache_hit"),
        "compiler_cache_hit": state.get("assessment_compiler_cache_hit"),
        "catalog_tasks": len((state.get("assessment_catalog") or {}).get("tasks") or []),
        "plan_errors": state.get("plan_errors"), "planning_issues": state.get("planning_issues"),
        "questions": sum(
            len(section.get("items") or []) if isinstance(section, dict) else len(section or [])
            for exam in state.get("generated_exams") or [] for section in (exam.get("questions") or {}).values()
        ),
        "structured_calls": len(structured_calls), "network_calls": len(network_calls),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
