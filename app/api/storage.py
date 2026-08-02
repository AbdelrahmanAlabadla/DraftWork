from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.config import REGISTRY_FILE

_lock = threading.Lock()
_current_document: str | None = None


def _load_registry() -> dict[str, Any]:
    path = Path(REGISTRY_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(registry: dict[str, Any]) -> None:
    path = Path(REGISTRY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def register_document(document_id: str, metadata: dict[str, Any]) -> None:
    global _current_document
    with _lock:
        registry = _load_registry()
        registry[document_id] = metadata
        _save_registry(registry)
        _current_document = document_id


def list_documents() -> dict[str, Any]:
    with _lock:
        return _load_registry()


def get_document(document_id: str) -> dict[str, Any] | None:
    with _lock:
        return _load_registry().get(document_id)


def get_current_document() -> str | None:
    global _current_document
    if _current_document:
        return _current_document
    with _lock:
        registry = _load_registry()
        if registry:
            _current_document = max(registry, key=lambda k: registry[k].get("uploaded_at", ""))
    return _current_document
