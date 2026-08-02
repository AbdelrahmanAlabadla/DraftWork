from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


LLAMA_PARSE_API: str | None = _env("LLAMA_PARSE_API")
QDRANT_URL: str = _env("QDRANT_URL", "http://localhost:6333") or "http://localhost:6333"
QDRANT_COLLECTION: str = _env("QDRANT_COLLECTION", "genexam") or "genexam"

LMS_URL: str = _env("LMS_URL", "http://127.0.0.1:1234") or "http://127.0.0.1:1234"
LMS_MODEL: str = _env("LMS_MODEL", "google/gemma-4-e2b") or "google/gemma-4-e2b"

EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "BAAI/bge-m3") or "BAAI/bge-m3"
EMBEDDING_DIM: int = int(_env("EMBEDDING_DIM", "1024") or "1024")

UPLOAD_DIR: str = _env("UPLOAD_DIR", "data/uploads") or "data/uploads"
REGISTRY_FILE: str = _env("REGISTRY_FILE", "data/documents.json") or "data/documents.json"

LOG_LEVEL: str = _env("LOG_LEVEL", "INFO") or "INFO"
