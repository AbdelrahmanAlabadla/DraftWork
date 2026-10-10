from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import repositories
from app.api.main import app
from app.jobs import tasks as task_module


EXAM_ID = "exam_canonical123"
JOB_ID = "22222222-2222-4222-8222-222222222222"
DOCUMENT_ID = "33333333-3333-4333-8333-333333333333"
SESSION_ID = "11111111-1111-4111-8111-111111111111"


def _session(monkeypatch):
    value = {"id": SESSION_ID}
    monkeypatch.setattr(
        "app.api.session_middleware.repositories.resolve_session", lambda _token: value
    )
    monkeypatch.setattr(
        "app.api.session_middleware.repositories.create_session", lambda _token: value
    )


def test_generate_request_returns_the_precreated_exam_id(monkeypatch):
    _session(monkeypatch)

    class Storage:
        def read_json(self, _key):
            return {"sections": [{"child_ids": ["c1"]}]}

    monkeypatch.setattr(
        repositories,
        "get_document_for_session",
        lambda *_args, **_kwargs: {
            "id": DOCUMENT_ID,
            "status": "ready",
            "structure_storage_key": "structure.json",
        },
    )
    monkeypatch.setattr("app.api.job_routes.get_file_storage", lambda: Storage())
    monkeypatch.setattr(
        repositories,
        "create_generation_job",
        lambda **_kwargs: {
            "id": JOB_ID,
            "exam_id": EXAM_ID,
            "status": "queued",
            "_idempotency_reused": False,
        },
    )
    monkeypatch.setattr("app.api.job_routes.job_service.enqueue_generation", lambda _job: None)
    monkeypatch.setattr(repositories, "mark_job_dispatched", lambda _job: None)

    response = TestClient(app).post(
        f"/api/v1/documents/{DOCUMENT_ID}/exam-jobs",
        json={"mcq_count": 1, "child_ids": ["c1"]},
    )

    assert response.status_code == 202
    assert response.json()["exam_id"] == EXAM_ID


def _questions(qtype: str, count: int):
    if qtype == "fill_in_the_blank":
        return {"word_bank": [], "items": []}
    return [
        {"question_id": f"model1_{qtype}_{index}", "slot_id": f"m1_{qtype}_{index}"}
        for index in range(1, count + 1)
    ]


def test_partial_exam_is_saved_with_shortage_and_same_id(monkeypatch):
    request = {
        "exam_id": EXAM_ID,
        "tasks": [
            ["mcq", 7], ["true_false", 5], ["fill_in_the_blank", 5],
            ["definition", 3], ["why", 3],
        ],
        "num_models": 1,
        "child_ids": ["c1"],
        "difficulty": "easy",
        "metadata": {},
    }
    job = {"document_id": DOCUMENT_ID, "session_id": SESSION_ID, "request_data": request}
    document = {
        "id": DOCUMENT_ID,
        "active_index_job_id": "index-job",
        "stats": {"processed_document_key": "cache", "domain_type": "stem", "subject": "math"},
    }
    saved = {}
    audit_events = []
    monkeypatch.setattr(task_module.repositories, "claim_job", lambda *_args: job)
    monkeypatch.setattr(task_module.repositories, "get_document", lambda *_args: document)
    monkeypatch.setattr(task_module.repositories, "get_exam_id_for_job", lambda *_args: EXAM_ID)
    monkeypatch.setattr(task_module.repositories, "update_job_progress", lambda *_args: None)
    monkeypatch.setattr(task_module.repositories, "job_is_cancelled", lambda *_args: False)
    monkeypatch.setattr(task_module.repositories, "complete_generation_job", lambda *args, **kwargs: saved.update(kwargs) or True)
    monkeypatch.setattr(
        task_module.repositories,
        "append_exam_audit_event",
        lambda exam_id, **kwargs: audit_events.append({"exam_id": exam_id, **kwargs}),
    )
    monkeypatch.setattr(task_module.evaluation_store, "save_evaluation", lambda **_kwargs: None)
    monkeypatch.setattr(task_module, "_heartbeat", lambda *_args: nullcontext())

    def generated(*_args, **kwargs):
        questions = {
            "mcq": _questions("mcq", 7),
            "true_false": _questions("true_false", 5),
            "fill_in_the_blank": _questions("fill_in_the_blank", 0),
            "definition": _questions("definition", 3),
            "why": _questions("why", 3),
        }
        kwargs["audit_callback"](
            "generation", "questions_generated", {"generated_exams": [{"questions": questions}]}
        )
        kwargs["audit_callback"](
            "general_validation", "validation_completed", {"verdicts": [{"action": "PASS"}]}
        )
        kwargs["audit_callback"](
            "repair", "repair_completed", {"validation_repair_history": []}
        )
        kwargs["audit_callback"](
            "duplicate_scan", "duplicate_scan_completed", {
                "candidate_pairs": [], "pair_verdicts": [], "fact_cache_hits": 0,
            }
        )
        return {
            "exams": [{"model_number": 1, "questions": questions, "markdown": "exam"}],
            "warnings": [],
            "planning_issues": [{
                "question_type": "fill_in_the_blank",
                "requested_count": 5,
                "supported_count": 0,
                "reason": "The unique catalog has too few distinct grounded tasks.",
            }],
            "assessment_profile": {"domain": "STEM", "subject": "math"},
            "quality_summary": {"requested": 23, "valid": 18, "missing": 5, "invalid": 0},
            "document_language": "en",
            "complete": False,
            "status": "needs_attention",
            "eval": {},
        }

    monkeypatch.setattr(task_module, "generate_exams", generated)

    response = task_module.generate_exam.run(JOB_ID)

    assert response["exam_id"] == EXAM_ID
    assert response["exam_status"] == "partial"
    assert saved["exam_id"] == EXAM_ID
    assert saved["status"] == "partial"
    assert len(saved["exams"][0]["questions"]["mcq"]) == 7
    shortage = next(item for item in saved["warnings"] if isinstance(item, dict))
    assert shortage == {
        "type": "question_shortage",
        "question_type": "fill_in_the_blank",
        "requested_count": 5,
        "generated_count": 0,
        "missing_count": 5,
        "requested": 5,
        "generated": 0,
        "missing": 5,
        "reason": "missing_questions",
    }
    assert saved["metadata"]["question_shortages"] == [shortage]
    assert {event["stage"] for event in audit_events} >= {
        "generation", "general_validation", "repair", "duplicate_scan"
    }
    generated_event = next(event for event in audit_events if event["stage"] == "generation")
    assert generated_event["payload"]["generated_exams"][0]["questions"]["mcq"]


def test_lifecycle_migration_keeps_failed_exam_and_audit_rows():
    migration = (
        Path(__file__).parents[1]
        / "migrations" / "versions" / "0004_exam_lifecycle_audit.py"
    ).read_text(encoding="utf-8")
    assert "'failed'" in migration
    assert "CREATE TABLE exam_audit_events" in migration
    assert "REFERENCES exams(id)" in migration


def test_completed_full_exam_has_no_shortage_warning():
    result = {
        "exams": [{
            "model_number": 1,
            "questions": {"mcq": _questions("mcq", 2)},
        }],
        "planning_issues": [],
        "complete": True,
    }
    assert task_module._question_shortage_warnings(result, [("mcq", 2)], 1) == []


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_frontend_opens_partial_exam_and_never_builds_undefined_path(tmp_path):
    source = Path(__file__).parents[1] / "FrontEnd" / "js" / "exam-lifecycle.js"
    module = tmp_path / "exam-lifecycle.mjs"
    module.write_text(
        source.read_text(encoding="utf-8").replace('./i18n.js', './i18n.mjs'),
        encoding="utf-8",
    )
    i18n_source = Path(__file__).parents[1] / "FrontEnd" / "js" / "i18n.js"
    (tmp_path / "i18n.mjs").write_text(i18n_source.read_text(encoding="utf-8"), encoding="utf-8")
    script = """
globalThis.localStorage = {getItem: () => 'en', setItem: () => {}};
const lifecycle = await import(process.env.MODULE_URL);
const examId = lifecycle.resolveExamId(
  {exam_id: 'exam_fixed'},
  {status: 'completed', result: {status: 'partial', exam_id: 'exam_fixed'}}
);
const summary = lifecycle.partialExamSummary({
  status: 'partial',
  metadata: {quality_summary: {requested: 23, valid: 18}},
  warnings: [{type: 'question_shortage', question_type: 'fill_in_the_blank', missing_count: 5}]
});
const fullSummary = lifecycle.partialExamSummary({status: 'complete', warnings: []});
let undefinedRejected = false;
try { lifecycle.examResourcePath(undefined); } catch (_) { undefinedRejected = true; }
console.log(JSON.stringify({examId, path: lifecycle.examResourcePath(examId), summary, fullSummary, undefinedRejected}));
"""
    module_url = module.resolve().as_uri()
    process = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "MODULE_URL": module_url},
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["path"] == "/exams/exam_fixed"
    assert result["summary"]["generated"] == 18
    assert result["fullSummary"] is None
    assert result["undefinedRejected"] is True


@pytest.mark.skipif(NODE is None, reason="node not available")
@pytest.mark.parametrize("language", ["en", "ar"])
def test_partial_exam_warning_uses_ui_language_and_dynamic_values(tmp_path, language):
    frontend = Path(__file__).parents[1] / "FrontEnd" / "js"
    module = tmp_path / "exam-lifecycle.mjs"
    module.write_text(
        (frontend / "exam-lifecycle.js").read_text(encoding="utf-8")
        .replace('./i18n.js', './i18n.mjs'),
        encoding="utf-8",
    )
    (tmp_path / "i18n.mjs").write_text(
        (frontend / "i18n.js").read_text(encoding="utf-8"), encoding="utf-8"
    )
    script = """
globalThis.localStorage = {getItem: () => process.env.UI_LANG, setItem: () => {}};
const warning = {hidden: true, textContent: ''};
globalThis.document = {getElementById: () => warning};
const lifecycle = await import(process.env.MODULE_URL);
lifecycle.renderPartialExamWarning({
  status: 'partial',
  metadata: {quality_summary: {requested: 23, valid: 18}},
  warnings: [{
    type: 'question_shortage', question_type: 'fill_in_the_blank',
    requested_count: 5, generated_count: 0, missing_count: 5,
    reason: 'No compatible finalized catalog tasks were available', stage: 'planning'
  }]
});
console.log(JSON.stringify(warning));
"""
    process = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        env={**os.environ, "MODULE_URL": module.resolve().as_uri(), "UI_LANG": language},
    )
    assert process.returncode == 0, process.stderr
    rendered = json.loads(process.stdout)["textContent"]
    assert "18" in rendered and "23" in rendered and "5" in rendered
    if language == "en":
        assert "Fill-in-the-Blank" in rendered
        assert "Missing:" in rendered
        assert not any("\u0600" <= char <= "\u06ff" for char in rendered)
    else:
        assert "ملء الفراغ" in rendered
        assert "المفقود" in rendered
        assert "Your exam" not in rendered and "Fill-in" not in rendered
