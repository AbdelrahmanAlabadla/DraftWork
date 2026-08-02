from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.config import LLAMA_PARSE_API
from app.logging_conf import get_logger
from app.offline.parser_items import (
    ParserSchemaError,
    document_pages,
    validate_pages,
)

logger = get_logger("PARSER")

class ParserError(RuntimeError):
    pass


class LlamaParser:
    """PDF parsing via the LlamaParse API. Returns LlamaParse page dicts."""

    def __init__(self) -> None:
        if not LLAMA_PARSE_API:
            raise ParserError("Missing LLAMA_PARSE_API in environment/.env")
        from llama_parse import LlamaParse  # deferred import keeps CLI light

        self.parser = LlamaParse(
            api_key=LLAMA_PARSE_API,
            result_type="json",
            ignore_errors=False,
        )

    def parse(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"File not found: {path}")

        t0 = time.perf_counter()
        logger.info("Parsing started | parser=LlamaParse | file=%s", path.name)

        try:
            result = self.parser.parse(str(path))
        except Exception as exc:
            logger.error(
                "Parsing failed | parser=LlamaParse | file=%s | exc=%s: %s",
                path.name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            raise ParserError(f"LlamaParse failed: {exc}") from exc

        try:
            documents = result if isinstance(result, list) else [result]
            pages: list[dict[str, Any]] = []
            for doc in documents:
                if not hasattr(doc, "model_dump"):
                    raise ParserSchemaError(
                        "LlamaParse result does not support JSON serialization"
                    )
                payload = doc.model_dump(mode="json")
                error = payload.get("error")
                if error:
                    raise ParserSchemaError(f"LlamaParse returned an error: {error}")
                pages.extend(document_pages(payload))

            validate_pages(pages)
        except ParserSchemaError as exc:
            logger.error(
                "Parser schema validation failed | file=%s | reason=%s",
                path.name,
                exc,
                exc_info=True,
            )
            raise ParserError(f"Invalid LlamaParse JSON: {exc}") from exc

        elapsed = time.perf_counter() - t0
        logger.info(
            "Parsing completed | parser=LlamaParse | file=%s | pages=%d | time=%.2fs",
            path.name,
            len(pages),
            elapsed,
        )
        return pages
