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

# Base URL of the LM Studio server (native REST API /api/v1/chat is used).
# A trailing /v1 (OpenAI-compatible form) is tolerated and stripped.
LMS_URL: str = _env("LMS_URL", "http://127.0.0.1:1234") or "http://127.0.0.1:1234"
LMS_MODEL: str = _env("LMS_MODEL", "qwen/qwen3-8b") or "qwen/qwen3-8b"
# Bearer token sent as Authorization header for OpenAI-compatible endpoints
# (OpenCode Zen). Empty means no auth header (local servers).
LMS_API_KEY: str | None = _env("LMS_API_KEY") or None
# Reasoning setting for chat calls ("off"|"low"|"medium"|"high"|"on"). We default
# to "off" so reasoning-capable models don't burn output tokens on hidden
# reasoning content that this pipeline discards anyway.
LMS_REASONING: str = _env("LMS_REASONING", "off") or "off"
# Keep validator requests small enough that local models reliably return one
# verdict per question.  Missing IDs are retried once in an even smaller request.
VALIDATOR_BATCH_SIZE: int = max(
    1, int(_env("VALIDATOR_BATCH_SIZE", "8") or "8")
)
TITLE_MODEL: str = _env("TITLE_MODEL", "mistralai/mistral-7b-instruct-v0.3") or "mistralai/mistral-7b-instruct-v0.3"
# Base URL used ONLY by the title-generation client (section/subsection
# naming). Defaults to the local LM Studio OpenAI-compatible endpoint so
# titles can use a different served model than the main pipeline.
TITLE_LMS_URL: str = _env("TITLE_LMS_URL", "http://127.0.0.1:1234/v1") or "http://127.0.0.1:1234/v1"

EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "BAAI/bge-m3") or "BAAI/bge-m3"
EMBEDDING_DIM: int = int(_env("EMBEDDING_DIM", "1024") or "1024")

UPLOAD_DIR: str = _env("UPLOAD_DIR", "data/uploads") or "data/uploads"
REGISTRY_FILE: str = _env("REGISTRY_FILE", "data/documents.json") or "data/documents.json"

LOG_LEVEL: str = _env("LOG_LEVEL", "INFO") or "INFO"

# --- Google Forms export (optional, downstream of generation) ---------------
# Master switch; when false/absent the export endpoints return 503.
GOOGLE_FORMS_ENABLED: bool = str(_env("GOOGLE_FORMS_ENABLED", "false")).lower() in (
    "1", "true", "yes", "on"
)
# OAuth desktop-client secrets file (NEVER committed) and cached user token.
# Both live under gitignored data/ by default.
GOOGLE_OAUTH_CLIENT_FILE: str = _env(
    "GOOGLE_OAUTH_CLIENT_FILE", "data/google_forms/oauth_client.json"
) or "data/google_forms/oauth_client.json"
GOOGLE_OAUTH_TOKEN_FILE: str = _env(
    "GOOGLE_OAUTH_TOKEN_FILE", "data/google_forms/token.json"
) or "data/google_forms/token.json"
# 0 disables page breaks inside a question-type section.
GOOGLE_FORMS_QUESTIONS_PER_PAGE: int = int(
    _env("GOOGLE_FORMS_QUESTIONS_PER_PAGE", "0") or "0"
)
# Points assigned to each auto-graded question in Google Forms quizzes.
GOOGLE_FORMS_POINTS: int = int(_env("GOOGLE_FORMS_POINTS", "1") or "1")

# --- Google Forms ownership mode --------------------------------------------
# teacher: Forms are created with the signed-in teacher's OAuth credentials
#          (teacher owns them). Requires the Web OAuth client config below.
# central: legacy fallback — Forms created under the desktop-client account,
#          shared as writer with one explicit email. Never automatic.
# disabled: Google Forms export fully off.
GOOGLE_FORMS_MODE: str = (_env("GOOGLE_FORMS_MODE", "teacher") or "teacher").lower()
if GOOGLE_FORMS_MODE not in ("teacher", "central", "disabled"):
    GOOGLE_FORMS_MODE = "teacher"

# Web application OAuth client (teacher sign-in + teacher-owned Forms).
GOOGLE_WEB_OAUTH_CLIENT_ID: str | None = _env("GOOGLE_WEB_OAUTH_CLIENT_ID")
GOOGLE_WEB_OAUTH_CLIENT_SECRET: str | None = _env("GOOGLE_WEB_OAUTH_CLIENT_SECRET")
GOOGLE_WEB_OAUTH_REDIRECT_URI: str = _env(
    "GOOGLE_WEB_OAUTH_REDIRECT_URI", "http://127.0.0.1:8000/auth/google/callback"
) or "http://127.0.0.1:8000/auth/google/callback"

# Server-side signing key for the session cookie. If unset, a random key is
# generated PER BOOT: every server restart invalidates all sessions (and, in
# teacher mode, connections are lost too). Set it for stable sessions.
SESSION_SECRET: str | None = _env("SESSION_SECRET")
SESSION_MAX_AGE_SECONDS: int = int(_env("SESSION_MAX_AGE_SECONDS", "604800") or "604800")


def web_oauth_configured() -> bool:
    return bool(
        GOOGLE_WEB_OAUTH_CLIENT_ID
        and GOOGLE_WEB_OAUTH_CLIENT_SECRET
        and GOOGLE_WEB_OAUTH_REDIRECT_URI
    )

# --- Online exam planning --------------------------------------------------
# The planner LLM receives a LIGHTWEIGHT context per selected child chunk:
# the chunk title plus only this many leading tokens of its text. It decides
# WHAT each question should test (question_type / topic / concept_to_test) but
# never writes the question. Actual question generation keeps using the FULL
# selected child-chunk content.
PLANNER_SNIPPET_TOKENS: int = int(_env("PLANNER_SNIPPET_TOKENS", "100") or "100")

# Total cap on the whole planner context (all section/chunk titles + snippets) in
# TOKENS. The planner request must stay inside the model context window; without
# this cap many chunks quickly overflow it and the server returns a 500.
PLANNER_CONTEXT_TOKENS: int = int(
    _env("PLANNER_CONTEXT_TOKENS", "4000") or "4000"
)

# Cap on the FULL generation context (selected child content) in TOKENS. The
# question generator receives this whole context, so keeping it bounded prevents
# oversized prompts that overflow the model's context window (500s).
GENERATION_CONTEXT_TOKENS: int = int(
    _env("GENERATION_CONTEXT_TOKENS", "3000") or "3000"
)

# --- Semantic structure generation (offline) ------------------------------
STRUCTURES_DIR: str = _env("STRUCTURES_DIR", "data/structures") or "data/structures"

# Exact dumps of the offline pipeline for inspection: the RAW LlamaParse output
# and the chunking report are copied here for every uploaded document.
PARSED_OUTPUT_DIR: str = _env("PARSED_OUTPUT_DIR", "data/parsed output") or "data/parsed output"

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
CHILD_MIN_TOKENS_MERGE: int = int(_env("CHILD_MIN_TOKENS_MERGE", "45") or "45")

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

# How many of the most recent accepted titles are shown to the LLM in each
# family-batch call so it avoids reusing a heading seen just before.
TITLE_CONTEXT_RECENT: int = int(_env("TITLE_CONTEXT_RECENT", "4") or "4")

# Navigation-label word bounds: sections and subsections 2-5 words.
SECTION_TITLE_MIN_WORDS: int = int(_env("SECTION_TITLE_MIN_WORDS", "2") or "2")
SECTION_TITLE_MAX_WORDS: int = int(_env("SECTION_TITLE_MAX_WORDS", "15") or "15")
SUBSECTION_TITLE_MIN_WORDS: int = int(_env("SUBSECTION_TITLE_MIN_WORDS", "2") or "2")
SUBSECTION_TITLE_MAX_WORDS: int = int(_env("SUBSECTION_TITLE_MAX_WORDS", "15") or "15")

# Fallback (safety-net) label caps.
FALLBACK_SECTION_MAX_WORDS: int = int(_env("FALLBACK_SECTION_MAX_WORDS", "6") or "6")
FALLBACK_SUBSECTION_MAX_WORDS: int = int(
    _env("FALLBACK_SUBSECTION_MAX_WORDS", "6") or "6"
)

# Generic filler headings that never make a usable navigation header. A title
# is rejected only when it EQUALS one of these ("Data Overview" stays valid;
# bare "Overview", "Key Concepts" do not) or when it starts with one of the
# filler prefixes in title_generator ("Introduction to X", "Overview of X").
TITLE_BLOCKLIST: frozenset[str] = frozenset(
    w.strip().lower()
    for w in _env(
        "TITLE_BLOCKLIST",
        "overview,introduction,summary,conclusion,key concepts,key terms,"
        "discussion,task,activity,exercises,questions,notes,basics,"
        "fundamentals,review,reading,objectives,aims,outline,definitions",
    ).split(",")
    if w.strip()
)

# --- Title review (spell-check pass over generated headers) -----------------
# After all titles are generated a reviewer LLM call per section (section +
# its subsections, one call each) scores every header against its passage and
# rewrites the bad ones. Rewritten headers are re-scored afterwards to confirm
# the fix stuck.
# Disabled by default: the family-batch generator now titles sections and their
# subsections together in one call with local blocklist/format checks, so the
# separate reviewer pass is an opt-in extra.
TITLE_REVIEW_ENABLED: bool = str(_env("TITLE_REVIEW_ENABLED", "false")).lower() in (
    "1", "true", "yes", "on"
)
# Preview length seen by the reviewer / verifier.
TITLE_REVIEW_CONTEXT_WORDS: int = int(
    _env("TITLE_REVIEW_CONTEXT_WORDS", "150") or "150"
)
# Score bands: >= GOOD keep; between POLISH_MIN and GOOD refine; below replace.
TITLE_REVIEW_GOOD_SCORE: int = int(_env("TITLE_REVIEW_GOOD_SCORE", "8") or "8")
TITLE_REVIEW_POLISH_MIN: int = int(_env("TITLE_REVIEW_POLISH_MIN", "4") or "4")
# When a rewritten title is needed, the model proposes this many candidates in
# ONE call; the best locally-valid one is picked. If it fails the rescore, this
# many extra candidate batches are tried before the deterministic fallback.
TITLE_REVIEW_CANDIDATES: int = int(_env("TITLE_REVIEW_CANDIDATES", "4") or "4")
TITLE_REVIEW_RETRIES: int = int(_env("TITLE_REVIEW_RETRIES", "1") or "1")
