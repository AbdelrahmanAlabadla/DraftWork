from __future__ import annotations

import gzip
import json

from app.offline.cleaner import clean_pages, clean_text
from app.offline.parser_items import item_text
from app.offline import processed_cache
from app.offline.semantic_chunker import (
    Paragraph,
    ParentChunk,
    build_parents,
    build_children,
    split_sentences,
    strip_list_markers,
)


def _parent(text: str, *, kind: str = "text", atomic_id: str | None = None) -> ParentChunk:
    item = Paragraph(
        text=text, page=1, item_type=kind, parser_item_type=kind,
        order=0, atomic_id=atomic_id,
    )
    return ParentChunk(
        parent_id="parent", document_id="doc", title=None, page_start=1,
        page_end=1, content=text, source_items=[item],
    )


def test_line_aware_marker_cleanup_never_deletes_math_operators():
    value = "4y - 6x ≥ -32\nx − y = 2\n-3 ≤ x < 8\n• First item\n  - Nested prose"
    cleaned = strip_list_markers(value)
    assert "4y - 6x ≥ -32" in cleaned
    assert "x − y = 2" in cleaned
    assert "-3 ≤ x < 8" in cleaned
    assert "•" not in cleaned
    assert "Nested prose" in cleaned


def test_equation_and_malformed_math_are_atomic_and_exact():
    system = "Intro.\n$$\n\\begin{cases}\nx-y=2 \\\\\nx+y=8\n\\end{cases}\n$$\nExplain."
    pieces = split_sentences(system)
    math = next(piece for piece in pieces if piece.startswith("$$"))
    assert math == "$$\n\\begin{cases}\nx-y=2 \\\\\nx+y=8\n\\end{cases}\n$$"
    assert split_sentences("Before. $$ x - y = 4")[-1] == "$$ x - y = 4"


def test_oversized_table_is_one_child_and_is_not_truncated():
    table = "| A | B |\n|---|---|\n" + "\n".join(
        f"| row {index} | {index} - 1 |" for index in range(1200)
    )
    children = build_children(_parent(table, kind="table"), sentence_vecs=[[1.0, 0.0]])
    assert len(children) == 1
    assert children[0].content == table
    parents = build_parents([[Paragraph(
        table, page=1, item_type="table", parser_item_type="table", order=0,
        role="table", atomic_id="table-0",
    )]], "doc")
    assert len(parents) == 1
    assert parents[0].content == table


def test_complete_oversized_exercise_remains_atomic():
    exercise = "7. A tank contains 120 liters.\nIt loses 8 liters per minute.\n" + "\n".join(
        f"{letter}) Explain consequence {letter} with all relevant givens."
        for letter in "abcdefghijklmnopqrstuvwxyz"
    )
    children = build_children(_parent(exercise, atomic_id="exercise-7"))
    assert len(children) == 1
    assert "7. A tank" in children[0].content
    assert "z) Explain" in children[0].content


def test_table_canonical_content_fallbacks_and_cleaner_preserves_original_fields():
    markdown = {"type": "table", "value": None, "md": "| س | ص |\n|---|---|\n| 4 | x - 2 |"}
    assert item_text(markdown) == markdown["md"]
    csv_item = {"type": "table", "value": None, "csv": 'name,value\n"a,b",2'}
    assert item_text(csv_item) == csv_item["csv"]
    rows = {"type": "table", "value": None, "rows": [["A", "B"], ["x | y", "4"]]}
    assert r"x \| y" in item_text(rows)
    cleaned = clean_pages([{"page": 1, "items": [markdown]}])[0]["items"][0]
    assert cleaned["value"] == markdown["md"]
    assert cleaned["md"] == markdown["md"]


def test_known_html_is_unwrapped_without_damaging_comparison_operators():
    assert clean_text("Find <u>x</u> when x < 7.<br>Explain.") == "Find x when x < 7.\nExplain."


def test_multipart_exercise_is_not_removed_as_mcq_choices():
    exercise = (
        "75. Explore the system.\n"
        "$$\\begin{aligned}x+y&=3 \\\\ x-y&=1\\end{aligned}$$\n"
        "a. Make a table.\nb. Explain the intersection.\n"
        "c. Graph the equations.\nd. State when no solution exists."
    )
    cleaned = clean_pages([{"page": 1, "items": [{"type": "text", "value": exercise}]}])
    value = item_text(cleaned[0]["items"][0])
    assert value.startswith("75. Explore")
    assert "\\begin{aligned}" in value
    assert "d. State when no solution exists." in value


def test_explicit_cache_key_rejects_obsolete_processing_versions(monkeypatch, tmp_path):
    monkeypatch.setattr(processed_cache.config, "PROCESSED_DOCUMENT_CACHE_DIR", str(tmp_path))
    sha = "d" * 64
    processed_cache.save_processed_document(
        sha, structure={"parents": [], "children": []}, metadata={}
    )
    key = processed_cache.processed_document_key(sha)
    path = tmp_path / key[:2] / f"{key}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload["processing_versions"]["chunker"] = "obsolete"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    assert processed_cache.load_processed_document_by_key(key) is None
    assert processed_cache.load_processed_document_by_key(
        key, require_current_versions=False
    ) is not None
