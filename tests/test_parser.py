from __future__ import annotations

import sys
import types

import pytest

from app.offline import parser as parser_module
from app.offline.parser import LlamaParser, ParserError
from app.offline.parser_items import validate_pages


class FakeResult:
    def __init__(self, payload: dict):
        self.payload = payload

    def model_dump(self, mode: str) -> dict:
        assert mode == "json"
        return self.payload


def _install_llama_parse(monkeypatch, payload: dict) -> dict:
    captured: dict = {}

    class FakeLlamaParse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def parse(self, file_path: str):
            captured["file_path"] = file_path
            return FakeResult(payload)

    monkeypatch.setitem(
        sys.modules,
        "llama_parse",
        types.SimpleNamespace(LlamaParse=FakeLlamaParse),
    )
    monkeypatch.setattr(parser_module, "LLAMA_PARSE_API", "test-key")
    return captured


def test_parser_returns_json_pages(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"pdf")
    pages = [
        {
            "page": 1,
            "items": [
                {"type": "heading", "value": "Chapter"},
                {"type": "text", "lvl": None, "value": "Body"},
                {"type": "table", "lvl": None, "value": "A | B"},
            ],
        },
        {"page": 2, "items": []},
    ]
    captured = _install_llama_parse(
        monkeypatch,
        {"pages": pages, "error": None},
    )

    result = LlamaParser().parse(pdf)

    assert result == pages
    assert captured["result_type"] == "json"
    assert captured["ignore_errors"] is False
    assert validate_pages(result) == {"heading": 1, "text": 1, "table": 1}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"pages": []},
        {"pages": [{"page": 1}]},
        {"pages": [{"page": 1, "items": "invalid"}]},
        {"pages": [{"page": 1, "items": []}]},
    ],
)
def test_parser_rejects_missing_or_malformed_json_items(
    monkeypatch, tmp_path, payload
):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"pdf")
    _install_llama_parse(monkeypatch, payload)

    with pytest.raises(ParserError, match="Invalid LlamaParse JSON"):
        LlamaParser().parse(pdf)
