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
TITLE_MODEL: str = _env("TITLE_MODEL", "mistralai/mistral-7b-instruct-v0.3") or "mistralai/mistral-7b-instruct-v0.3"

EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "BAAI/bge-m3") or "BAAI/bge-m3"
EMBEDDING_DIM: int = int(_env("EMBEDDING_DIM", "1024") or "1024")

UPLOAD_DIR: str = _env("UPLOAD_DIR", "data/uploads") or "data/uploads"
REGISTRY_FILE: str = _env("REGISTRY_FILE", "data/documents.json") or "data/documents.json"

LOG_LEVEL: str = _env("LOG_LEVEL", "INFO") or "INFO"

# --- Semantic structure generation (offline) ------------------------------
STRUCTURES_DIR: str = _env("STRUCTURES_DIR", "data/structures") or "data/structures"

# Dense cosine-similarity thresholds used by the chunking pipeline.
# - Parent chunks: consecutive paragraphs are merged while their embedding
#   cosine similarity stays at/above SIMILARITY_THRESHOLD.
# - Child chunks: a sentence joins the current child while its cosine
#   similarity with the child's running centroid stays at/above
#   SIMILARITY_THRESHOLD_CHILD.
SIMILARITY_THRESHOLD: float = float(_env("SIMILARITY_THRESHOLD", "0.50") or "0.50")
SIMILARITY_THRESHOLD_CHILD: float = float(
    _env("SIMILARITY_THRESHOLD_CHILD", "0.62") or "0.62"
)

# Average token-per-word ratio used to derive word caps from token limits.
WORDS_PER_TOKEN: float = float(_env("WORDS_PER_TOKEN", "1.3") or "1.3")

# Hard ceiling on a single child chunk, expressed in TOKENS. Sentences are
# packed greedily up to this ceiling (derived word cap via WORDS_PER_TOKEN),
# always cutting only at sentence boundaries.
CHILD_MAX_SIZE: int = int(_env("CHILD_MAX_SIZE", "800") or "800")

# Minimum child chunk sizes, expressed in TOKENS (converted to words via
# WORDS_PER_TOKEN). Keeps tiny fragments out of the vector store:
# - at or below CHILD_MIN_TOKENS_DROP  -> discarded entirely
# - between DROP and CHILD_MIN_TOKENS_MERGE -> merged into the previous child
CHILD_MIN_TOKENS_DROP: int = int(_env("CHILD_MIN_TOKENS_DROP", "6") or "6")
CHILD_MIN_TOKENS_MERGE: int = int(_env("CHILD_MIN_TOKENS_MERGE", "10") or "10")

# A parent is treated as an atomic "questions" unit when this share of its
# sentences are numbered questions (e.g. "1. How would you define ML?").
# Such parents are kept whole instead of being split into dozens of tiny
# overlapping child chunks.
QUESTION_PARENT_MIN_SHARE: float = float(
    _env("QUESTION_PARENT_MIN_SHARE", "0.6") or "0.6"
)

# Parent chunk size bounds, expressed in TOKENS (converted to words via
# WORDS_PER_TOKEN):
# - at or below PARENT_MIN_TOKENS_DROP -> discarded entirely
# - under PARENT_MERGE_TOKENS -> merged into the previous parent (or next)
# - PARENT_MAX_SIZE is a hard ceiling; any single parent above it is split.
PARENT_MIN_TOKENS_DROP: int = int(_env("PARENT_MIN_TOKENS_DROP", "30") or "30")
PARENT_MERGE_TOKENS: int = int(_env("PARENT_MERGE_TOKENS", "100") or "100")
PARENT_MAX_SIZE: int = int(_env("PARENT_MAX_SIZE", "1200") or "1200")

# --- Title generation (naming stage; does not affect chunk boundaries) -----
# LLM only receives a preview of the chunk content, not the whole chunk.
SECTION_TITLE_CONTEXT_WORDS: int = int(_env("SECTION_TITLE_CONTEXT_WORDS", "250") or "250")
SUBSECTION_TITLE_CONTEXT_WORDS: int = int(
    _env("SUBSECTION_TITLE_CONTEXT_WORDS", "175") or "175"
)

TITLE_TEMPERATURE: float = float(_env("TITLE_TEMPERATURE", "0.15") or "0.15")
TITLE_MAX_TOKENS: int = int(_env("TITLE_MAX_TOKENS", "25") or "25")
TITLE_MAX_ATTEMPTS: int = int(_env("TITLE_MAX_ATTEMPTS", "2") or "2")

# How many title-generation workers run concurrently (each makes its own LLM call).
TITLE_PARALLELISM: int = int(_env("TITLE_PARALLELISM", "4") or "4")

# Chunk titles are generated in document order, in batches of this many chunks.
# Within a batch the titles run in parallel using the same recent-titles
# snapshot (from previously completed batches only).
TITLE_BATCH_SIZE: int = int(_env("TITLE_BATCH_SIZE", "4") or "4")

# Navigation-label word bounds: sections and subsections 2-5 words.
SECTION_TITLE_MIN_WORDS: int = int(_env("SECTION_TITLE_MIN_WORDS", "2") or "2")
SECTION_TITLE_MAX_WORDS: int = int(_env("SECTION_TITLE_MAX_WORDS", "5") or "5")
SUBSECTION_TITLE_MIN_WORDS: int = int(_env("SUBSECTION_TITLE_MIN_WORDS", "2") or "2")
SUBSECTION_TITLE_MAX_WORDS: int = int(_env("SUBSECTION_TITLE_MAX_WORDS", "5") or "5")

# Fallback (safety-net) label caps.
FALLBACK_SECTION_MAX_WORDS: int = int(_env("FALLBACK_SECTION_MAX_WORDS", "6") or "6")
FALLBACK_SUBSECTION_MAX_WORDS: int = int(
    _env("FALLBACK_SUBSECTION_MAX_WORDS", "6") or "6"
)
