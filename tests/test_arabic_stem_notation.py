from __future__ import annotations

from app.language import source_notation
from app.online import prompts


def _arabic_planned_item(evidence: str) -> dict:
    return {
        "slot_id": "m1_equation_1",
        "source_chunk_id": "c1",
        "support_chunk_ids": [],
        "concept": "حل معادلة خطية",
        "task_operation": "solve_equation",
        "relationship": "linear equality",
        "target": "unknown value",
        "reasoning_mode": "calculation",
        "difficulty": "medium",
        "required_givens": [],
        "assigned_sources": {"c1": evidence},
    }


def test_source_notation_detects_latin_variables_and_western_digits():
    notation = source_notation("Solve 2x + 3y = 12 for x.")
    assert notation["digits"] == "western"
    assert notation["latin_variables"] == "present"


def test_source_notation_detects_arabic_indic_digits():
    notation = source_notation("حل المعادلة ٢س + ٣ص = ١٢")
    assert notation["digits"] == "arabic_indic"


def test_source_notation_unspecified_for_plain_text():
    notation = source_notation("energy is conserved")
    assert notation["digits"] == "unspecified"


def test_arabic_prompt_preserves_latin_source_variables_and_western_digits():
    item = _arabic_planned_item("حل المعادلة التالية: 2x + 3y = 12")
    _system, user = prompts.build_prompt(
        "equation", 1, "fallback", planned_items=[item], language="ar"
    )
    folded = user.casefold()
    assert "## mathematical notation (arabic output)" in folded
    assert "keep latin letters" in folded
    assert "never convert them to arabic letters" in folded
    assert "keep western digits" in folded
    # The source expression itself is preserved verbatim.
    assert "2x + 3y = 12" in user


def test_arabic_prompt_preserves_arabic_indic_when_source_uses_them():
    item = _arabic_planned_item("حل المعادلة ٢س + ٣ص = ١٢")
    _system, user = prompts.build_prompt(
        "equation", 1, "fallback", planned_items=[item], language="ar"
    )
    folded = user.casefold()
    assert "arabic-indic numerals" in folded
    assert "keep western digits" not in folded


def test_english_prompt_has_no_arabic_notation_block():
    item = _arabic_planned_item("Solve 2x + 3y = 12")
    _system, user = prompts.build_prompt(
        "equation", 1, "fallback", planned_items=[item], language="en"
    )
    assert "Mathematical Notation (Arabic output)" not in user


def test_bundled_arabic_prompt_includes_notation_policy():
    item = _arabic_planned_item("Energy is conserved as 2 J is transferred.")
    item["slot_id"] = "m1_mcq_1"
    _system, user = prompts.build_obj_bundled_prompt(
        {"mcq": [item], "true_false": []}, "fallback", language="ar",
        model_number=1,
    )
    assert "Mathematical Notation (Arabic output)" in user


def test_fitb_arabic_prompt_includes_notation_policy():
    item = _arabic_planned_item("الطاقة محفوظة")
    item["slot_id"] = "m1_fill_in_the_blank_1"
    item["fitb_answer"] = "energy"
    _system, user = prompts.build_fitb_items_prompt(
        1, {"m1_fill_in_the_blank_1": "energy"}, "fallback",
        planned_items=[item], language="ar",
    )
    assert "Mathematical Notation (Arabic output)" in user
