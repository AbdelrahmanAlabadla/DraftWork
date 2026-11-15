from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from app.api import repositories
from app.logging_conf import set_job_context, set_llm_attribution
from app.online import independent_recovery, validator


def _candidate_state() -> dict:
    active = {"model_number": 1, "questions": {"equation": [
        {"slot_id": "m1_equation_1", "question_id": "q1", "question": "old", "answer": "1"},
        {"slot_id": "m1_equation_2", "question_id": "q2", "question": "other", "answer": "2"},
    ]}}
    plan = {"model_number": 1, "items": {"equation": [
        {"slot_id": "m1_equation_1", "question_type": "equation", "task_id": "backup_1",
         "selected_variation_id": "v2", "assignment_revision": 2},
        {"slot_id": "m1_equation_2", "question_type": "equation", "task_id": "other",
         "selected_variation_id": "vo"},
    ]}}
    staged = {"model_number": 1, "questions": {"equation": [
        {"slot_id": "m1_equation_1", "question_id": "q1", "question": "new", "answer": "3"},
    ]}}
    return {
        "generated_exams": [active], "plans": [plan],
        "repair_candidates": [{
            "model_number": 1, "slot_id": "m1_equation_1", "question_id": "q1",
            "question_type": "equation", "exam": staged, "plans": [deepcopy(plan)],
            "attempt": 1, "repair_action": "replace_task", "repair_strategy": "preplanned_backup",
            "candidate_task_id": "backup_1", "previous_failure_code": "duplicate_question",
            "had_original": True,
        }],
        "repair_manifest": [{
            "model_number": 1, "slot_id": "m1_equation_1", "question_id": "q1",
            "question_type": "equation", "status": "invalid",
            "failure_code": "duplicate_question",
        }],
        "validation_reports": [{"model_number": 1, "verdicts": [{
            "question_id": "q1", "action": "FIX_QUESTION_AND_ANSWER",
            "failure_code": "duplicate_question", "validation_role": "duplicate",
        }]}],
        "validated_models": [], "warnings": [], "validation_repair_history": [],
    }


def _pass_candidate(exam, question_ids, *_args):
    return {
        "model_number": 1, "all_pass": True, "warnings": [], "coverage": [],
        "verdicts": [{"question_id": qid, "action": "PASS",
                      "validation_role": "general", "fields_to_fix": []}
                     for qid in question_ids],
    }


def test_replace_task_full_validation_and_duplicate_pass_commits(monkeypatch):
    observed = {}

    def duplicate_gate(exam, **kwargs):
        observed["questions"] = [
            item.get("question")
            for values in (exam.get("questions") or {}).values()
            for item in (values.get("items") if isinstance(values, dict) else values)
        ]
        return {
            "candidate_pairs": [], "pair_verdicts": [], "duplicate_groups": [],
            "similarity_distribution": {}, "usage": {},
            "fact_cache": kwargs.get("fact_cache") or {},
            "fact_cache_hits": 0, "fact_cache_misses": 0,
        }

    monkeypatch.setattr(validator, "_validate_with_context", _pass_candidate)
    monkeypatch.setattr(
        "app.online.duplicate_judge.review_duplicate_candidates", duplicate_gate,
    )
    result = validator.validate_generated_questions(_candidate_state())
    assert "new" in observed["questions"] and "other" in observed["questions"]
    monkeypatch.setattr(
        "app.online.duplicate_judge.review_duplicate_candidates",
        lambda _exam, **kwargs: {
            "candidate_pairs": [], "pair_verdicts": [], "duplicate_groups": [],
            "similarity_distribution": {}, "usage": {},
            "fact_cache": kwargs.get("fact_cache") or {},
            "fact_cache_hits": 0, "fact_cache_misses": 0,
        },
    )
    result = validator.validate_generated_questions(_candidate_state())
    assert {"new", "other"} <= set(observed["questions"])
    assert result["generated_exams"][0]["questions"]["equation"][0]["question"] == "new"
    assert result["repair_manifest"] == []
    assert result["validation_repair_history"][-1]["status"] == "accepted"
    verdict = result["validation_reports"][0]["verdicts"][0]
    assert verdict["action"] == "PASS"
    assert verdict["role_statuses"]["duplicate"] == "PASS"


def test_replace_task_validator_pass_duplicate_fail_rejects_without_stale_commit(monkeypatch):
    monkeypatch.setattr(validator, "_validate_with_context", _pass_candidate)
    monkeypatch.setattr(
        "app.online.duplicate_pipeline.scan_exam_duplicates",
        lambda _state: {"duplicate_reports": [{"duplicate_groups": [{
            "group_id": "g1", "question_ids": ["q1", "q2"], "pair_ids": ["q1::q2"],
        }]}]},
    )

    def duplicate_conflict(exam, **_kwargs):
        ids = [
            item["question_id"]
            for section in (exam.get("questions") or {}).values()
            for item in (section.get("items") if isinstance(section, dict) else section)
        ]
        synthetic = next(value for value in ids if value.startswith("__candidate_"))
        active = next(
            value for value in ids
            if not value.startswith("__candidate_") and value == "q2"
        )
        pair_id = "::".join(sorted((synthetic, active)))
        return {
            "candidate_pairs": [{
                "pair_id": pair_id, "left": {"question_id": synthetic},
                "right": {"question_id": active},
            }],
            "pair_verdicts": [{"pair_id": pair_id, "classification": "duplicate"}],
            "duplicate_groups": [], "similarity_distribution": {}, "usage": {},
            "fact_cache": {}, "fact_cache_hits": 0, "fact_cache_misses": 1,
        }

    monkeypatch.setattr(
        "app.online.duplicate_judge.review_duplicate_candidates", duplicate_conflict,
    )
    result = validator.validate_generated_questions(_candidate_state())
    assert result["generated_exams"][0]["questions"]["equation"][0]["question"] == "old"
    assert result["repair_manifest"][0]["status"] == "invalid"
    assert result["repair_manifest"][0]["failure_code"] == "duplicate_question"


def test_cross_model_unresolved_duplicate_keeps_latest_valid_and_is_not_user_visible():
    state = {
        "num_models": 2,
        "generated_exams": [{"model_number": 1, "questions": {}}, {"model_number": 2, "questions": {}}],
        "plans": [{"model_number": 1, "items": {}}, {"model_number": 2, "items": {}}],
        "validation_reports": [{"model_number": 2, "verdicts": [{
            "question_id": "q2", "action": "PASS", "evaluation_origin": "fresh",
        }]}],
        "repair_manifest": [{
            "model_number": 2, "slot_id": "m2_equation_1", "question_id": "q2",
            "failure_code": "cross_model_duplicate", "status": "invalid",
        }],
        "validated_models": [1],
        "warnings": [{"type": "cross_model_duplicates", "count": 1}],
        "cross_model_duplicate_findings": [{"recovery_status": "pending"}],
    }
    result = independent_recovery._apply_cross_model_nonblocking_fallback(
        state, deepcopy(state),
    )
    assert result["validated_models"] == [1, 2]
    assert result["repair_manifest"] == []
    assert result["warnings"] == []
    assert result["cross_model_duplicate_fallbacks"][0]["resolution"] == "kept_latest_valid"
    assert result["cross_model_duplicate_findings"][0]["recovery_status"].startswith("accepted_")


def test_cross_model_invalid_latest_restores_previous_fully_validated_candidate():
    previous = {
        "num_models": 2,
        "generated_exams": [{"model_number": 1, "questions": {}}, {"model_number": 2,
            "questions": {"equation": [{"question_id": "q2", "slot_id": "m2_equation_1", "question": "valid"}]}}],
        "plans": [{"model_number": 1, "items": {}}, {"model_number": 2, "items": {}}],
        "validation_reports": [{"model_number": 2, "verdicts": [{"question_id": "q2", "action": "PASS"}]}],
    }
    latest = deepcopy(previous)
    latest["generated_exams"][1]["questions"]["equation"][0]["question"] = "invalid repair"
    latest["validation_reports"][0]["verdicts"][0]["action"] = "FIX_ANSWER"
    latest.update({
        "repair_manifest": [{"model_number": 2, "slot_id": "m2_equation_1", "question_id": "q2",
                             "failure_code": "cross_model_duplicate"}],
        "validated_models": [1], "warnings": [],
    })
    result = independent_recovery._apply_cross_model_nonblocking_fallback(latest, previous)
    question = result["generated_exams"][1]["questions"]["equation"][0]
    assert question["question"] == "valid"
    assert result["cross_model_duplicate_fallbacks"][0]["resolution"] == "restored_previous_valid"


def test_request_attempt_persistence_carries_exam_slot_and_is_idempotent(monkeypatch):
    captured = {}

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def cursor(self): return Cursor()
        def commit(self): pass

    monkeypatch.setattr(repositories.db, "connection", lambda: Connection())
    set_job_context("job-1", "doc-1", exam_id="exam-1")
    set_llm_attribution(model_number=1, slot_ids=["m1_equation_1"], stage="revalidation")
    repositories.record_llm_request_event(
        job_id="job-1", provider="mock", model="mock-model", operation="validate_batch",
        agent_role="task_alignment_validator", request_id="attempt-1", status="success",
        usage_available=True, input_tokens=10, output_tokens=3, reasoning_tokens=1,
        estimated_cost_usd=0.01, duration_ms=25,
    )
    assert "ON CONFLICT DO NOTHING" in captured["sql"]
    metadata = captured["params"][-1].obj
    assert metadata == {
        "exam_id": "exam-1", "model_number": 1,
        "slot_ids": ["m1_equation_1"], "stage": "revalidation",
    }


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_usage_dashboard_displays_persisted_totals_and_unknown(tmp_path):
    source = (Path(__file__).parents[1] / "FrontEnd/js/usage-dashboard.js").read_text(encoding="utf-8")
    source = source.replace('import { getJSON } from "./api.js";\n', "const getJSON = async () => ({});\n")
    source = source.replace(
        'import { t, uiLang } from "./i18n.js";\n',
        "const uiLang=()=>\"en\"; const t=(k,v={})=>k+JSON.stringify(v);\n",
    )
    module = tmp_path / "usage-dashboard.mjs"
    module.write_text(source, encoding="utf-8")
    script = r'''
class Element {
  constructor(tag='div') { this.tag=tag; this.children=[]; this.hidden=true; this.textContent=''; this.className=''; }
  appendChild(value) { this.children.push(value); return value; }
  append(...values) { this.children.push(...values); }
  replaceChildren(...values) { this.children=values; }
}
const root = new Element();
globalThis.document = {getElementById: () => root, createElement: (tag) => new Element(tag)};
const module = await import(process.env.MODULE_URL);
module.renderUsageDashboard({stage_wall_time_ms: 100,
 totals:{calls:2,input_tokens:null,output_tokens:5,reasoning_tokens:1,estimated_cost_usd:0.1,latency_ms:50},
 agents:[{agent:'validator',calls:2,input_tokens:null,output_tokens:5,reasoning_tokens:1,estimated_cost_usd:0.1,latency_ms:50}]});
const texts=[]; const walk=(node)=>{texts.push(node.textContent); (node.children||[]).forEach(walk)}; walk(root);
console.log(JSON.stringify({hidden:root.hidden,texts}));
'''
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", script], capture_output=True,
        text=True, encoding="utf-8", timeout=30,
        env={**os.environ, "MODULE_URL": module.resolve().as_uri()},
    )
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["hidden"] is False
    assert any("usage.unknown" in value for value in rendered["texts"])
    assert any("$0.100000" in value for value in rendered["texts"])
