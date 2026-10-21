from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_generation_progress_has_real_stage_labels_in_english_and_arabic():
    source = (ROOT / "Frontend/js/i18n.js").read_text(encoding="utf-8")
    for key in (
        "retrieving_context", "compiling_assessment_catalog", "planning_exam",
        "adjusting_difficulty", "reviewing_plan", "generating_exam",
        "validating_exam", "repairing_exam", "checking_duplicates",
        "repairing_duplicates", "finalizing_exam", "completed",
    ):
        assert source.count(f'"exam_stage.{key}"') == 2
    assert 'return normalized.replaceAll("_", " ");' in source


def test_generation_progress_uses_persisted_events_and_reuses_loop_stages():
    source = (ROOT / "Frontend/js/main.js").read_text(encoding="utf-8")
    html = (ROOT / "Frontend/index.html").read_text(encoding="utf-8")
    assert 'current.pipeline_events || []' in source
    assert '["validating_exam", "validating_updates"]' in source
    assert '["repairing_exam"]' in source
    assert 'jobStatus === "retrying"' in source
    assert 'id="generationWorkflow"' in html


def test_generation_progress_styles_active_completed_failed_and_skipped_states():
    css = (ROOT / "Frontend/css/styles.css").read_text(encoding="utf-8")
    for state in ("active", "complete", "failed", "skipped"):
        assert f".generation-stage.{state}" in css
    assert "animation: pulse-stage" in css
