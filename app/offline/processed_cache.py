from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from app import config


def _json_native(value: Any) -> Any:
    """Recursively convert array-library containers and scalars for JSON."""
    if isinstance(value, dict):
        return {_json_native(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _json_native(converted)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_native(tolist())
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def processing_versions() -> dict[str, str]:
    app_root = Path(__file__).resolve().parents[1]
    def code_hash(relative: str) -> str:
        return hashlib.sha256((app_root / relative).read_bytes()).hexdigest()

    return {
        "parser": config.PARSER_PIPELINE_VERSION,
        "parser_code_sha256": code_hash("offline/parser.py"),
        "parser_items_code_sha256": code_hash("offline/parser_items.py"),
        "cleaner_code_sha256": code_hash("offline/cleaner.py"),
        "chunker": config.CHUNKER_PIPELINE_VERSION,
        "chunker_code_sha256": code_hash("offline/semantic_chunker.py"),
        "title_generator_code_sha256": code_hash("offline/title_generator.py"),
        "title_provider": "deepseek",
        "title_model": config.DEEPSEEK_MODEL,
        "title_classification_context_words": str(
            config.TITLE_CLASSIFICATION_CONTEXT_WORDS
        ),
        "embedding": config.EMBEDDING_PIPELINE_VERSION,
        "embedding_code_sha256": code_hash("offline/embeddings.py"),
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": str(config.EMBEDDING_DIM),
    }


def processed_document_key(sha256: str) -> str:
    material = json.dumps(
        {"file_sha256": sha256.lower(), "versions": processing_versions()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _cache_path(key: str) -> Path:
    return Path(config.PROCESSED_DOCUMENT_CACHE_DIR) / key[:2] / f"{key}.json.gz"


def load_processed_document(sha256: str) -> dict[str, Any] | None:
    key = processed_document_key(sha256)
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    if (
        payload.get("processed_document_key") != key
        or payload.get("file_sha256") != sha256.lower()
        or payload.get("processing_versions") != processing_versions()
        or not isinstance(payload.get("structure"), dict)
    ):
        return None
    for embedding in payload.get("embeddings") or []:
        sparse = embedding.get("sparse") if isinstance(embedding, dict) else None
        if isinstance(sparse, dict):
            embedding["sparse"] = {int(key): value for key, value in sparse.items()}
    return payload


def load_processed_document_by_key(
    key: str, *, require_current_versions: bool = True
) -> dict[str, Any] | None:
    """Load an already validated cache artifact for PostgreSQL backfill."""
    path = _cache_path(str(key))
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    if (
        payload.get("processed_document_key") != str(key)
        or not isinstance(payload.get("structure"), dict)
        or (require_current_versions and payload.get("processing_versions") != processing_versions())
    ):
        return None
    return payload


def save_processed_document(
    sha256: str,
    *,
    structure: dict[str, Any],
    embeddings: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any],
) -> Path:
    key = processed_document_key(sha256)
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_document_key": key,
        "file_sha256": sha256.lower(),
        "processing_versions": processing_versions(),
        "structure": structure,
        # Retained as an empty compatibility field so old cache readers fail
        # safely while new restores never require vector-index embeddings.
        "embeddings": embeddings or [],
        "metadata": metadata,
    }
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{key}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(
                _json_native(payload),
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def materialize_for_document(
    cached_structure: dict[str, Any], document_id: str
) -> dict[str, Any]:
    """Clone cached chunks with deterministic IDs in a new document scope."""
    structure = deepcopy(cached_structure)
    parent_ids: dict[str, str] = {}
    for parent in structure.get("parents") or []:
        old = str(parent.get("parent_id") or "")
        new = str(uuid.uuid5(uuid.NAMESPACE_URL, f"genexam:{document_id}:parent:{old}"))
        parent_ids[old] = new
        parent["parent_id"] = new
        parent["document_id"] = document_id
    for child in structure.get("children") or []:
        old = str(child.get("child_id") or "")
        child_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"genexam:{document_id}:child:{old}"))
        child["child_id"] = child_id
        child["chunk_id"] = child_id
        child["parent_id"] = parent_ids.get(str(child.get("parent_id") or ""), str(child.get("parent_id") or ""))
        child["document_id"] = document_id
    return structure
