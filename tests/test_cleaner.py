from __future__ import annotations

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.offline.cleaner import clean_pages, clean_text  # noqa: E402


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


def test_clean_text_unicode_quotes():
    raw = "It\u2019s called \u201csmart\u201d."
    result = clean_text(raw)
    assert "\u2019" not in result
    assert "\u201c" not in result


def test_clean_pages_removes_repeated_headers():
    pages = [
        _page([{"type": "text", "value": "Chapter 1"}], page=1),
        _page([{"type": "text", "value": "Real content one"}], page=1),
    ]
    # Add 5 pages all repeating the same header at the top.
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
