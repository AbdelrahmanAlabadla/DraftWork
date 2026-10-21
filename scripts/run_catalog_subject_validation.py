from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm import deepseek
from app.logging_conf import configure_logging
from app.offline.chunk_store import ChunkStore
from app.online import assessment_catalog, duplicate_judge, exam_builder, planner, validator
from app.online.eval_stats import create_pipeline_eval
from app.online.graph import get_exam_graph


SUBJECTS = {
    "math": {
        "source": "data/e2e_arch_validation/math/final_state.json",
        "tasks": [("mcq", 8), ("true_false", 4), ("why", 5),
                  ("equation", 7), ("word_problem", 4)],
    },
    "chemistry": {
        "source": "data/e2e_arch_validation/chemistry/final_state.json",
        "tasks": [("mcq", 8), ("true_false", 4), ("fill_in_the_blank", 2),
                  ("definition", 6), ("why", 3), ("equation", 2),
                  ("word_problem", 3)],
    },
}


def safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe(item) for item in value]
    dump = getattr(value, "model_dump", None)
    return safe(dump()) if callable(dump) else repr(value)


def snapshot_key(chunks: list[dict[str, Any]]) -> str:
    canonical = [{"child_id": str(c.get("child_id") or ""), "content": str(c.get("content") or "")}
                 for c in chunks]
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "legacy-processed-snapshot-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_report(log_path: Path) -> dict[str, Any]:
    segment = log_path.read_text(encoding="utf-8").rsplit("Request received |", 1)[-1]
    pattern = re.compile(
        r"DeepSeek usage \| agent=([^ |]+) \| input_tokens=(\d+) \| output_tokens=(\d+) "
        r"\| reasoning_tokens=(\d+) \| cached_tokens=(\d+) \| total_tokens=(\d+)"
    )
    totals = defaultdict(lambda: {"responses": 0, "input_tokens": 0, "output_tokens": 0,
                                  "reasoning_tokens": 0, "cached_tokens": 0, "total_tokens": 0})
    for match in pattern.finditer(segment):
        agent = match.group(1)
        values = list(map(int, match.groups()[1:]))
        row = totals[agent]
        row["responses"] += 1
        for key, value in zip(("input_tokens", "output_tokens", "reasoning_tokens",
                               "cached_tokens", "total_tokens"), values):
            row[key] += value
    return {"stages": dict(totals), "total_tokens": sum(v["total_tokens"] for v in totals.values())}


def run(subject: str, *, output_name: str | None = None) -> None:
    spec = SUBJECTS[subject]
    source = json.loads((ROOT / spec["source"]).read_text(encoding="utf-8"))
    chunks = source["retrieved_chunks"]
    child_ids = [str(value) for value in source["selected_child_ids"]]
    document_id = str(source["document_id"])
    tasks = spec["tasks"]
    output = ROOT / "data/e2e_arch_validation" / (
        output_name or f"{subject}_catalog_28_arabic"
    )
    output.mkdir(parents=True, exist_ok=True)

    def write(name: str, value: Any) -> None:
        (output / name).write_text(json.dumps(safe(value), ensure_ascii=False, indent=2), encoding="utf-8")

    by_id = {str(chunk["child_id"]): chunk for chunk in chunks}
    original_get = ChunkStore.get_by_child_ids
    ChunkStore.get_by_child_ids = lambda self, _doc, ids, **kwargs: [
        copy.deepcopy(by_id[value]) for value in ids if value in by_id
    ]
    configure_logging("INFO")
    handler = logging.FileHandler(output / "generation.log", mode="w", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    network_calls: list[dict[str, Any]] = []
    structured_calls: list[dict[str, Any]] = []
    original_network = deepseek.DeepSeekClient._request_with_backoff

    async def observed_network(self, *, timeout: int, **kwargs):
        started = time.perf_counter()
        record = {"index": len(network_calls) + 1, "max_output_tokens": kwargs.get("max_output_tokens")}
        try:
            response = await original_network(self, timeout=timeout, **kwargs)
            record["status"] = str(getattr(response, "status", ""))
            return response
        except Exception as exc:
            record["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["duration_seconds"] = round(time.perf_counter() - started, 3)
            network_calls.append(record)

    deepseek.DeepSeekClient._request_with_backoff = observed_network

    def observe(module_name: str, original):
        def wrapped(client, prompt, **kwargs):
            started = time.perf_counter()
            record = {"index": len(structured_calls) + 1, "module": module_name,
                      "agent": kwargs.get("agent"), "schema_name": kwargs.get("schema_name"),
                      "item_count": kwargs.get("item_count"), "prompt_chars": len(prompt),
                      "provider": getattr(client, "provider_name", None),
                      "model": getattr(client, "model", None)}
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
        return wrapped

    assessment_catalog.request_structured = observe("assessment_catalog", assessment_catalog.request_structured)
    planner.request_structured = observe("planner", planner.request_structured)
    exam_builder.request_structured = observe("exam_builder", exam_builder.request_structured)
    validator.request_structured = observe("validator", validator.request_structured)
    duplicate_judge.request_structured = observe("duplicate_judge", duplicate_judge.request_structured)

    initial = {
        "document_id": document_id, "processed_document_key": snapshot_key(chunks),
        "tasks": tasks, "num_models": 1, "selected_child_ids": child_ids,
        "session_id": None, "index_job_id": None, "difficulty": "mix",
        "document_language": "ar", "retrieved_chunks": [], "context": "", "planner_context": "",
        "assessment_catalog": {}, "assessment_catalog_key": "", "assessment_catalog_cache_hit": False,
        "assessment_compiler_cache_hit": False, "plans": [], "assessment_profile": {},
        "planning_issues": [], "plan_errors": [], "plan_attempts": 0,
        "unrepairable_plan_slots": [], "generated_exams": [], "rejection_feedback": None,
        "validation_reports": [], "validated_models": [], "question_repair_attempts": {},
        "pending_revalidation_ids": {}, "invalid_questions": [], "duplicate_repair_history": [],
        "validation_repair_history": [], "initial_global_reviews": [], "final_global_reviews": [],
        "duplicate_reports": [], "duplicate_similarity_distributions": [],
        "pending_duplicate_groups": [], "duplicate_repair_rounds": 0,
        "duplicate_replacement_attempts": {}, "duplicate_scan_count": 0,
        "exam_quality_findings": [], "eval_stats": create_pipeline_eval(tasks, 1),
        "warnings": [], "error": None,
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
        ChunkStore.get_by_child_ids = original_get
        deepseek.DeepSeekClient._request_with_backoff = original_network
        logging.getLogger().removeHandler(handler)
        handler.close()
        state.pop("progress_callback", None)
        state["duration_seconds"] = round(time.perf_counter() - started, 3)
        write("final_state.json", state)
        write("structured_calls.json", structured_calls)
        write("network_calls.json", network_calls)
        exams = state.get("generated_exams") or []
        if exams:
            (output / "exam.md").write_text(
                exam_builder.assemble_exam(exams[0].get("questions") or {}, language="ar"), encoding="utf-8"
            )
        if (output / "generation.log").exists():
            write("token_usage_by_stage.json", token_report(output / "generation.log"))
    print(json.dumps({"subject": subject, "duration_seconds": state["duration_seconds"],
                      "catalog_cache_hit": state.get("assessment_catalog_cache_hit"),
                      "compiler_cache_hit": state.get("assessment_compiler_cache_hit"),
                      "catalog_tasks": len((state.get("assessment_catalog") or {}).get("tasks") or []),
                      "plan_errors": state.get("plan_errors"), "planning_issues": state.get("planning_issues"),
                      "questions": sum(len(s.get("items") or []) if isinstance(s, dict) else len(s or [])
                                       for e in state.get("generated_exams") or []
                                       for s in (e.get("questions") or {}).values())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", choices=sorted(SUBJECTS))
    parser.add_argument("--output-name")
    args = parser.parse_args()
    run(args.subject, output_name=args.output_name)
