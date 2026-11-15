from __future__ import annotations

import json

import pytest

from app.llm.client import ProviderRequestBudget, ProviderRequestBudgetExceeded
from app.llm.math_strings import (
    MalformedMathStringError,
    repair_raw_json_latex_escapes,
    sanitize_math_strings,
)
from app.online.assessment_catalog import requires_unsupported_supplied_graph


def test_raw_json_latex_recovery_preserves_math_commands():
    raw = r'{"answer":"(-\infty, 2)","root":"\sqrt{x}"}'
    value = json.loads(repair_raw_json_latex_escapes(raw))
    assert value == {"answer": r"(-\infty, 2)", "root": r"\sqrt{x}"}


def test_parsed_json_escape_damage_is_repaired_without_regeneration():
    parsed = json.loads(r'{"fraction":"\frac{1}{2}","label":"\text{x}","product":"a\times b"}')
    assert sanitize_math_strings(parsed) == {
        "fraction": r"\frac{1}{2}", "label": r"\text{x}", "product": r"a\times b",
    }


@pytest.mark.parametrize("control", ["\x00", "\x10", "\x1c"])
def test_ambiguous_control_characters_fail_closed(control):
    with pytest.raises(MalformedMathStringError):
        sanitize_math_strings({"answer": f"x{control}y"})


def test_provider_budget_counts_bundle_request_for_each_slot():
    budget = ProviderRequestBudget(2)
    budget.consume(["m1_mcq_7", "m1_mcq_8"])
    budget.consume(["m1_mcq_7"])
    with pytest.raises(ProviderRequestBudgetExceeded):
        budget.consume(["m1_mcq_7"])
    assert budget.remaining("m1_mcq_8") == 1


def test_graph_capability_classification_distinguishes_operations():
    assert requires_unsupported_supplied_graph({
        "task_operation": "identify_vertices_from_graph",
        "student_work_product": "identified vertices",
        "required_givens": ["the graph shown below"],
        "requires_supplied_graph": True,
    })
    assert not requires_unsupported_supplied_graph({
        "task_operation": "graph_linear_equation",
        "student_work_product": "draw a graph",
        "required_givens": ["y=2x+3"],
    })
    assert not requires_unsupported_supplied_graph({
        "task_operation": "explain_slope_meaning",
        "student_work_product": "conceptual explanation",
        "target": "meaning of slope on a graph",
    })


def test_replacement_selector_pages_after_valid_no_candidate(monkeypatch):
    from app.online import duplicate_pipeline
    import app.llm.client as llm_client
    import app.llm.factory as llm_factory

    calls = []

    def choose(_client, prompt, **_kwargs):
        calls.append(prompt)
        context = json.loads(prompt.split("Return JSON matching the schema.\n", 1)[1])
        if len(calls) == 1:
            return {"slot_to_replace": "s", "replacement_task_id": None,
                    "replacement_fitb_entry_id": None,
                    "checked_against_active_slots": True, "reason": "unsuitable"}
        return {"slot_to_replace": "s", "replacement_task_id": "t6",
                "replacement_fitb_entry_id": None,
                "checked_against_active_slots": True, "reason": "suitable"}

    monkeypatch.setattr(llm_factory, "create_duplicate_replacement_client", lambda: object())
    monkeypatch.setattr(llm_client, "request_structured", choose)
    candidates = [{"task_id": f"t{i}", "concept": f"concept {i}"} for i in range(1, 8)]
    chosen, context = duplicate_pipeline._select_replacement_candidate(
        slot_id="s", qtype="mcq", conflict_slots=[], conflict_edges=[],
        candidates=candidates, active_items=[], blacklist=set(), previous_failures=[],
    )
    assert chosen["task_id"] == "t6"
    assert len(calls) == 2
    assert context["candidate_page"] == 2
