from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from app.config import PARSED_OUTPUT_DIR
from app.logging_conf import get_logger

logger = get_logger("DUMP_OUTPUTS")


def _output_dir() -> Path:
    dir_path = Path(PARSED_OUTPUT_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_parse_dump(document_id: str, raw_pages: list[dict[str, Any]]) -> Path:
    """Write the EXACT raw LlamaParse output (unmodified) as JSON."""
    path = _output_dir() / f"{document_id}.parse.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw_pages, fh, ensure_ascii=False, indent=2)
    logger.info(
        "Raw parse dump written | document_id=%s | file=%s | pages=%d",
        document_id,
        path.name,
        len(raw_pages),
    )
    return path


def save_chunk_dump(document_id: str, chunk_report_path: Path) -> Path:
    """Copy the chunking report into the parsed-output directory."""
    src = Path(chunk_report_path)
    dst = _output_dir() / f"{document_id}.chunks.txt"
    if src.exists():
        shutil.copyfile(src, dst)
    else:
        dst.write_text("", encoding="utf-8")
    logger.info("Chunk dump written | document_id=%s | file=%s", document_id, dst.name)
    return dst


def save_all_dumps(
    document_id: str,
    raw_pages: list[dict[str, Any]] | None = None,
    chunk_report_path: Path | None = None,
) -> dict[str, str]:
    """Convenience wrapper: dump parse output and/or chunks."""
    t0 = time.perf_counter()
    result: dict[str, str] = {}
    if raw_pages is not None:
        result["parse"] = str(save_parse_dump(document_id, raw_pages))
    if chunk_report_path is not None:
        result["chunks"] = str(save_chunk_dump(document_id, chunk_report_path))
    logger.info(
        "Dump outputs done | document_id=%s | time=%.3fs",
        document_id,
        time.perf_counter() - t0,
    )
    return result
