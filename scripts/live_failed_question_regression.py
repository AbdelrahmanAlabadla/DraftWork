"""Isolated, budget-capped replay of historic failed questions.

Reads checkpoints and chunks only; all resulting state is filesystem-local.
"""
from __future__ import annotations

import hashlib, json, sys, time, uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from app import config, db
from app.api import repositories
from app.logging_conf import get_logger
from app.online.multi_validator import validate_exam_multi
from app.online.multi_repair import repair_question_specialized
from app.online.validator import _plan_context_for_model

MAIN = "23467d6b-61b1-4073-89b2-74020ac1b3fa"
OLD = "c04575f1-e372-47e2-9c1d-b296b86c2ac1"
SLOTS = {"m1_mcq_1", "m1_mcq_4", "m1_mcq_9", "m1_true_false_5", "m1_equation_3", "m1_word_problem_2", "m1_word_problem_3"}
TF_SLOTS = {"m1_true_false_3", "m1_true_false_5"}
MAX_CALLS, MAX_COST = 150, 0.20

class BudgetStop(RuntimeError): pass

def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def _items(exam: dict[str, Any]):
    for qtype, section in (exam.get("questions") or {}).items():
        values = section.get("items") if isinstance(section, dict) else section
        for question in values or []: yield qtype, question

def _event_state(job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the earliest complete generated state and latest checkpoint payload."""
    with db.connection() as conn:
        rows = conn.execute("SELECT checkpoint_revision,event_type,payload FROM exam_audit_events WHERE job_id=%s ORDER BY checkpoint_revision,id", (job_id,)).fetchall()
    generated = latest = None
    for row in rows:
        payload = row[2] or {}; out = payload.get("node_output") or payload.get("checkpoint_state") or {}
        if out: latest = out
        if generated is None and out.get("generated_exams") and out.get("plans"):
            generated = out
    if not generated: raise RuntimeError(f"No complete generated checkpoint for {job_id}")
    return deepcopy(generated), deepcopy(latest or {})

def _hydrate_sources(state: dict[str, Any], document_id: str) -> None:
    ids = {str(v) for plan in state.get("plans") or [] for vals in (plan.get("items") or {}).values() for item in vals or [] for v in (item.get("source_evidence_ids") or [item.get("source_chunk_id"), *(item.get("support_chunk_ids") or [])]) if v}
    with db.connection() as conn:
        rows = conn.execute("SELECT chunk_id, content FROM document_chunks WHERE document_id=%s AND chunk_id = ANY(%s)", (document_id, list(ids))).fetchall()
    source = {str(row[0]): str(row[1] or "") for row in rows}
    missing = sorted(ids - set(source))
    if missing: raise RuntimeError(f"Original assignment evidence unavailable: {missing}")
    state["retrieved_chunks"] = [{"child_id": key, "content": value} for key, value in source.items()]

@contextmanager
def _capture_and_cap(records: list[dict[str, Any]]):
    from app.online import multi_validator, multi_repair, duplicate_judge, exam_builder
    from app.api import repositories as repo
    modules = (multi_validator, multi_repair, duplicate_judge, exam_builder)
    originals = {m: m.request_structured for m in modules}
    old_usage, old_event = repo.record_llm_usage, repo.record_llm_request_event
    spent = [0.0]
    def wrapped(original):
        def call(client, prompt, **kwargs):
            in_tokens = max(1, len(str(prompt)) // 4); out_tokens = int(kwargs.get("max_tokens") or 0)
            projected = (in_tokens * config.OPENAI_INPUT_COST_PER_MTOK + out_tokens * config.OPENAI_OUTPUT_COST_PER_MTOK) / 1_000_000
            if len(records) >= MAX_CALLS or spent[0] + projected > MAX_COST: raise BudgetStop("hard request/cost ceiling reached before dispatch")
            began=time.perf_counter(); record={"provider":getattr(client,"provider_name",None),"model":getattr(client,"model",None),"agent":kwargs.get("agent"),"operation":kwargs.get("operation"),"prompt":str(prompt),"system_prompt":kwargs.get("system_prompt"),"max_tokens":out_tokens,"started_at":datetime.now(timezone.utc).isoformat()}
            try:
                result=original(client,prompt,**kwargs); record["status"]="success"; record["response"]=result; return result
            except Exception as exc:
                record.update(status="error",error_type=type(exc).__name__,error=str(exc)); raise
            finally:
                usage=deepcopy(getattr(client,"last_usage",None)); record.update(duration_seconds=round(time.perf_counter()-began,6),usage=usage,request_id=getattr(client,"last_request_id",None)); spent[0]+=float((usage or {}).get("estimated_cost_usd") or 0); records.append(record)
        return call
    try:
        for m, original in originals.items(): m.request_structured=wrapped(original)
        repo.record_llm_usage=lambda *a,**k: None; repo.record_llm_request_event=lambda *a,**k: None
        yield
    finally:
        for m, original in originals.items(): m.request_structured=original
        repo.record_llm_usage,repo.record_llm_request_event=old_usage,old_event

def run() -> Path:
    run = ROOT / "data" / "storage" / "stage_test_checkpoints" / f"failed-question-regression-{uuid.uuid4().hex}"
    main, _ = _event_state(MAIN); old, _ = _event_state(OLD)
    main_job = repositories.get_job(MAIN) or {}; old_job = repositories.get_job(OLD) or {}
    _hydrate_sources(main, str(main_job.get("document_id") or "")); _hydrate_sources(old, str(old_job.get("document_id") or ""))
    main_exam = deepcopy(main["generated_exams"][0]); old_exam = deepcopy(old["generated_exams"][0])
    selected = {q.get("slot_id") for _,q in _items(main_exam)}
    if not SLOTS.issubset(selected): raise RuntimeError(f"Main slots missing: {sorted(SLOTS-selected)}")
    if not TF_SLOTS.issubset({q.get("slot_id") for _,q in _items(old_exam)}): raise RuntimeError("Historic TF slots missing")
    records: list[dict[str,Any]]=[]; results={}; started=datetime.now(timezone.utc).isoformat()
    plan, sources = _plan_context_for_model(main, 1); domain=str((main.get("assessment_profile") or {}).get("domain") or "STEM")
    with _capture_and_cap(records):
        for slot in sorted(SLOTS):
            qtype, question=next((t,q) for t,q in _items(main_exam) if q.get("slot_id")==slot)
            isolated={"model_number":1,"questions":{qtype:[deepcopy(question)]}}
            report=validate_exam_multi(isolated,plan_by_slot={slot:plan[slot]},source_by_chunk=sources,domain=domain,question_ids=[str(question.get("question_id"))],cache_scope=f"live-regression-{run.name}-{slot}")
            verdict=(report.get("verdicts") or [{}])[0]; attempts=[]
            for number in range(1,4):
                if verdict.get("action")=="PASS": break
                outcome=repair_question_specialized(exam=isolated,verdict=deepcopy(verdict),qtype=qtype,plan_item=plan[slot],source_chunks={k:sources[k] for k in (plan[slot].get("source_evidence_ids") or [plan[slot].get("source_chunk_id")]) if k in sources},domain=domain,recovery_strategy="targeted_repair" if number==1 else "reconciled_targeted_repair")
                attempts.append(outcome)
                if not outcome.get("repaired_ids"): break
                report=validate_exam_multi(isolated,plan_by_slot={slot:plan[slot]},source_by_chunk=sources,domain=domain,question_ids=[str(question.get("question_id"))],cache_scope=f"live-regression-revalidate-{run.name}-{slot}-{number}")
                verdict=(report.get("verdicts") or [{}])[0]
            results[slot]={"question_type":qtype,"original":question,"validation":verdict,"repair_attempts":attempts,"candidate":next(q for _,q in _items(isolated))}
    _write(run/"live_regression_checkpoint.json", {"started_at":started,"main_job":MAIN,"old_job":OLD,"results":results,"requests":records,"limits":{"max_calls":MAX_CALLS,"max_cost":MAX_COST},"totals":{"calls":len(records),"cost":sum(float((r.get("usage") or {}).get("estimated_cost_usd") or 0) for r in records)}})
    return run

if __name__=="__main__": print(run())
