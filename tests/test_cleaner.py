from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.offline.cleaner import (  # noqa: E402
    clean_pages,
    clean_pages_with_stats,
    clean_text,
    strip_inline_footnotes,
)


def _page(items: list[dict], page: int = 1) -> dict:
    return {"page": page, "items": items}


def test_clean_text_normalizes_whitespace():
    raw = "Hello   world.\n\n\n\nThis\tis a test.\n"
    result = clean_text(raw)
    assert "  " not in result
    assert "\t" not in result
    assert "\n\n\n" not in result


def test_clean_text_fixes_hyphenation():
    raw = "This is an exam- \nple of hyphenated text."
    assert clean_text(raw) == "This is an example of hyphenated text."


def test_clean_text_joins_soft_hyphen_between_letters():
    # U+2010 hyphen with a space after it (PDF word split) must be joined.
    assert clean_text("tech\u2010 niques and bet\u2010ter") == "techniques and better"
    assert clean_text("prob\u2010 lem") == "problem"


def test_clean_text_unicode_quotes():
    raw = "It\u2019s called \u201csmart\u201d."
    result = clean_text(raw)
    assert "\u2019" not in result
    assert "\u201c" not in result


def test_clean_pages_removes_repeated_headers():
    pages = [
        _page(
            [
                {"type": "text", "value": "RUNNING HEADER"},
                {"type": "text", "value": f"content page {i}"},
            ],
            page=i,
        )
        for i in range(5)
    ]
    cleaned = clean_pages(pages)
    for page in cleaned:
        texts = [i.get("value") for i in page["items"]]
        assert "RUNNING HEADER" not in texts
        assert any("content page" in t for t in texts)


def test_clean_pages_removes_page_numbers():
    pages = [_page([{"type": "text", "value": "3"}, {"type": "text", "value": "body"}])]
    cleaned = clean_pages(pages)
    texts = [i.get("value") for i in cleaned[0]["items"]]
    assert "3" not in texts
    assert "body" in texts


def test_clean_pages_removes_boilerplate():
    pages = [
        _page(
            [
                {"type": "text", "value": "All rights reserved."},
                {"type": "text", "value": "real content"},
            ]
        )
    ]
    cleaned = clean_pages(pages)
    texts = [i.get("value") for i in cleaned[0]["items"]]
    assert "All rights reserved." not in texts
    assert "real content" in texts


def test_clean_pages_removes_figure_caption():
    pages = [_page([{"type": "text", "value": "Figure 1-8."}])]
    cleaned = clean_pages(pages)
    assert cleaned == []


def test_clean_pages_removes_diagram_fragment():
    pages = [_page([{"type": "text", "value": "Feature 2 Feature"}])]
    cleaned = clean_pages(pages)
    assert cleaned == []


def test_clean_pages_removes_running_head_with_page_number():
    pages = [_page([{"type": "text", "value": "8 | Chapter 1: The Machine Learning Landscape"}])]
    cleaned = clean_pages(pages)
    assert cleaned == []


def test_clean_pages_removes_toc_row():
    pages = [_page([{"type": "text", "value": "Introduction ............ 3"}])]
    cleaned = clean_pages(pages)
    assert cleaned == []


def test_clean_pages_strips_inline_footnotes():
    pages = [
        _page(
            [
                {
                    "type": "text",
                    "value": "It is a system (e.g., 20% chance of being spam). "
                    "1 Fun fact: this odd-sounding name came from Galton.",
                }
            ]
        )
    ]
    cleaned = clean_pages(pages)
    text = cleaned[0]["items"][0]["value"]
    assert "1 Fun fact" not in text
    assert "Galton" in text


def test_clean_pages_removes_duplicate_paragraph():
    pages = [
        _page(
            [
                {"type": "text", "value": "Repeated content here."},
                {"type": "text", "value": "Repeated content here."},
            ]
        )
    ]
    cleaned = clean_pages(pages)
    texts = [i.get("value") for i in cleaned[0]["items"]]
    assert texts.count("Repeated content here.") == 1


def test_clean_pages_with_stats_reports_categories():
    pages = [
        _page(
            [
                {"type": "text", "value": "Figure 1-8."},
                {"type": "text", "value": "Feature 2 Feature"},
                {"type": "text", "value": "Real content paragraph."},
            ]
        )
    ]
    cleaned, stats = clean_pages_with_stats(pages)
    assert len(cleaned[0]["items"]) == 1
    assert stats.by_category["figure_caption"] == 1
    assert stats.by_category["diagram_fragment"] == 1
    assert stats.paragraphs_removed == 2
    assert stats.percent_text_removed > 0


def test_clean_pages_removes_value_label():
    # "Value Value?" is a clear diagram axis label, not prose.
    pages = [_page([{"type": "text", "value": "Value Value?"}])]
    cleaned = clean_pages(pages)
    assert cleaned == []


def test_clean_pages_keeps_ambiguous_short_text():
    # Conservative: a real, sentence-like short item must be kept.
    pages = [_page([{"type": "text", "value": "See Figure 1-1 for details."}])]
    cleaned = clean_pages(pages)
    texts = [i.get("value") for i in cleaned[0]["items"]]
    assert texts == ["See Figure 1-1 for details."]


def test_strip_inline_footnotes_counts():
    text = "End. 1 Note A. 2 Note B."
    out, n = strip_inline_footnotes(text)
    assert n == 2
    assert "Note A" in out and "Note B" in out
