"""Reduced offline replay of the 46-question mathematics exam (Group 2).

The retained forensic event stream lives outside the repository. When it is not
present the test skips explicitly rather than inventing data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.online import assessment_catalog
from app.online.difficulty_adjustment import (
    _apply_catalog_fallbacks,
    normalize_difficulty_designs,
)

FORENSIC = Path(r"C:\Users\abdul\AppData\Local\Temp\opencode\forensic")
CATALOG = Path(
    r"C:\Users\abdul\PycharmProjects\GenExam\data\assessment_catalogs\ed"
    r"\edd2a703d9b6712ab2cdce92c593657042a43d704fa2cdeb845fe71d6249a439.json"
)


def _load(name: str) -> dict:
    return json.loads((FORENSIC / name).read_text(encoding="utf-8"))


def _find(value, key, depth=0):
    if depth > 6:
        return None
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find(child, key, depth + 1)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find(child, key, depth + 1)
            if found is not None:
                return found
    return None


@pytest.mark.skipif(not FORENSIC.exists(), reason="46-question forensic stream unavailable")
def test_46_slot_plan_normalizes_and_keeps_every_assignment():
    plan_event = _load("0857_planning_plan_completed.json")
    plans = _find(plan_event, "plans")
    assert isinstance(plans, list)
    items = [item for plan in plans for values in (plan.get("items") or {}).values()
             for item in values]
    assert len(items) == 46, len(items)
    before = {
        str(item.get("slot_id")): str(item.get("base_catalog_task_id") or item.get("task_id"))
        for item in items
    }
    # These are pre-Group-2 plans: no design contract yet.
    assert not any(item.get("difficulty_design") for item in items)
    normalize_difficulty_designs(plans, "easy")
    after = {
        str(item.get("slot_id")): str(item.get("base_catalog_task_id") or item.get("task_id"))
        for item in items
    }
    assert before == after
    assert all(item.get("difficulty_design") for item in items)
    assert all(
        item["difficulty_design"]["status"] in {"pending", "not_required",
                                                "accepted_source_constraint"}
        for item in items
    )


@pytest.mark.skipif(not FORENSIC.exists(), reason="46-question forensic stream unavailable")
def test_46_slot_final_assembly_still_has_43_questions():
    assembled = _load("0887_finalization_exam_assembled.json")
    exams = _find(assembled, "generated_exams")
    assert isinstance(exams, list)
    questions = [
        question
        for exam in exams
        for section in (exam.get("questions") or {}).values()
        for question in (section.get("items") if isinstance(section, dict) else section) or []
    ]
    assert len(questions) == 43, len(questions)
    assert all(question.get("question_id") for question in questions)


@pytest.mark.skipif(not FORENSIC.exists(), reason="46-question forensic stream unavailable")
def test_difficulty_fallback_is_noop_for_46_slot_plans():
    plan_event = _load("0857_planning_plan_completed.json")
    plans = _find(plan_event, "plans")
    items = [item for plan in plans for values in (plan.get("items") or {}).values()
             for item in values]
    before = [str(item.get("base_catalog_task_id") or item.get("task_id")) for item in items]
    normalize_difficulty_designs(plans, "easy")
    _apply_catalog_fallbacks(plans, {"tasks": []}, "easy")
    after = [str(item.get("base_catalog_task_id") or item.get("task_id")) for item in items]
    assert before == after


def test_group1_graph_guard_blocks_only_explicitly_unavailable_visuals():
    if not CATALOG.exists():
        pytest.skip("persisted 120-task catalog unavailable")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for task_id in ("task_d4e09228b3dd3a60", "task_20cfebbfb887270b"):
        task = next(t for t in catalog["tasks"] if t["task_id"] == task_id)
        assert not assessment_catalog.requires_unsupported_supplied_graph(task)
        explicit = {**task, "requires_supplied_graph": True}
        assert assessment_catalog.requires_unsupported_supplied_graph(explicit)
