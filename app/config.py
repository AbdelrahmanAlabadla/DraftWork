from __future__ import annotations

import os
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

if os.getenv("APP_ENV", "local").strip().lower() != "production":
    load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} must be a boolean")
    return normalized in {"1", "true", "yes", "on"}


APP_ENV: str = (_env("APP_ENV", "local") or "local").lower()
if APP_ENV not in {"local", "test", "production"}:
    raise ValueError("APP_ENV must be local, test, or production")
IS_PRODUCTION: bool = APP_ENV == "production"


LLAMA_PARSE_API: str | None = _env("LLAMA_PARSE_API")
QDRANT_URL: str = _env("QDRANT_URL", "http://localhost:6333") or "http://localhost:6333"
QDRANT_COLLECTION: str = _env("QDRANT_COLLECTION", "genexam") or "genexam"
QDRANT_API_KEY: str | None = _env("QDRANT_API_KEY")
QDRANT_VERIFY_TLS: bool = _env_bool("QDRANT_VERIFY_TLS", True)
QDRANT_TIMEOUT_SECONDS: int = max(
    1, int(_env("QDRANT_TIMEOUT_SECONDS", "30") or "30")
)

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

# LLM backend selection. "local" preserves the existing LM Studio behavior;
# "deepseek" uses the separate DeepSeek API client.
LLM_PROVIDER: str = (_env("LLM_PROVIDER", "local") or "local").lower()
if LLM_PROVIDER not in {"local", "deepseek"}:
    raise ValueError("LLM_PROVIDER must be either 'local' or 'deepseek'")

DEEPSEEK_API_KEY: str | None = _env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL: str = (
    _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    or "https://api.deepseek.com"
)
# V4.1 Flash's official API identifier is the only accepted DeepSeek model.
DEEPSEEK_MODEL: str = _env("DEEPSEEK_MODEL", "deepseek-flash") or "deepseek-flash"
if LLM_PROVIDER == "deepseek" and DEEPSEEK_MODEL != "deepseek-flash":
    raise ValueError("DEEPSEEK_MODEL must be 'deepseek-flash'")
DEEPSEEK_CONCURRENCY_LIMIT: int = max(
    1, int(_env("DEEPSEEK_CONCURRENCY_LIMIT", "4") or "4")
)
DEEPSEEK_MAX_STRUCTURED_ATTEMPTS: int = max(
    1, min(3, int(_env("DEEPSEEK_MAX_STRUCTURED_ATTEMPTS", "3") or "3"))
)
DEEPSEEK_MAX_TRANSIENT_RETRIES: int = max(
    0, int(_env("DEEPSEEK_MAX_TRANSIENT_RETRIES", "2") or "2")
)
DEEPSEEK_RETRY_BASE_SECONDS: float = max(
    0.0, float(_env("DEEPSEEK_RETRY_BASE_SECONDS", "1.0") or "1.0")
)
ACTIVE_LLM_MODEL: str = DEEPSEEK_MODEL if LLM_PROVIDER == "deepseek" else LMS_MODEL
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
LEGACY_EXAM_DIR: str = _env("LEGACY_EXAM_DIR", "data/legacy_exams") or "data/legacy_exams"

# Durable application state and replaceable file storage.  The local backend is
# deliberately addressed by opaque keys so Azure Blob can implement the same
# interface later without changing API or worker code.
STORAGE_BACKEND: str = (_env("STORAGE_BACKEND", "local") or "local").lower()
LOCAL_STORAGE_ROOT: str = _env("LOCAL_STORAGE_ROOT", "data/storage") or "data/storage"

# Anonymous browser sessions.
SESSION_COOKIE_NAME: str = _env("SESSION_COOKIE_NAME", "genexam_session") or "genexam_session"
SESSION_COOKIE_SECURE: bool = _env_bool("SESSION_COOKIE_SECURE", False)
SESSION_COOKIE_SAMESITE: str = (_env("SESSION_COOKIE_SAMESITE", "lax") or "lax").lower()
if SESSION_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    raise ValueError("SESSION_COOKIE_SAMESITE must be lax, strict, or none")
SESSION_TTL_SECONDS: int = max(300, int(_env("SESSION_TTL_SECONDS", "604800") or "604800"))
SESSION_TOKEN_BYTES: int = max(32, int(_env("SESSION_TOKEN_BYTES", "32") or "32"))
SESSION_HASH_PEPPER: str = _env("SESSION_HASH_PEPPER", "") or ""

# Clerk verifies signed browser session tokens on the API.  The publishable key
# is intentionally safe for browser delivery; the secret key is server-only.
CLERK_PUBLISHABLE_KEY: str | None = _env("CLERK_PUBLISHABLE_KEY")
CLERK_SECRET_KEY: str | None = _env("CLERK_SECRET_KEY")
CLERK_JWT_KEY: str | None = _env("CLERK_JWT_KEY")
CLERK_FRONTEND_API_URL: str | None = _env("CLERK_FRONTEND_API_URL")
CLERK_AUTHORIZED_PARTIES: tuple[str, ...] = tuple(
    origin.strip()
    for origin in (_env(
        "CLERK_AUTHORIZED_PARTIES",
        "http://127.0.0.1:8000,http://localhost:8000",
    ) or "").split(",")
    if origin.strip()
)

# Celery/Redis local job queue.  Azure Service Bus will replace this transport
# at deployment time; application messages contain only a job id.
CELERY_BROKER_URL: str = _env("CELERY_BROKER_URL", "redis://localhost:6379/0") or "redis://localhost:6379/0"
CELERY_RESULT_BACKEND: str = _env("CELERY_RESULT_BACKEND", "redis://localhost:6379/1") or "redis://localhost:6379/1"
CELERY_TASK_ALWAYS_EAGER: bool = _env_bool("CELERY_TASK_ALWAYS_EAGER", False)
ENABLE_LEGACY_SYNC_API: bool = _env_bool("ENABLE_LEGACY_SYNC_API", False)

# Agreed server-side limits.  Token-accounting limits intentionally remain TBD.
MAX_UPLOAD_BYTES: int = max(1, int(_env("MAX_UPLOAD_BYTES", "104857600") or "104857600"))
MAX_QUESTIONS_PER_GENERATION: int = max(
    1, int(_env("MAX_QUESTIONS_PER_GENERATION", "150") or "150")
)
MAX_MODELS_PER_GENERATION: int = max(
    1, min(4, int(_env("MAX_MODELS_PER_GENERATION", "4") or "4"))
)
MAX_ACTIVE_GENERATION_JOBS_PER_SESSION: int = max(
    1, int(_env("MAX_ACTIVE_GENERATION_JOBS_PER_SESSION", "2") or "2")
)
DEEPSEEK_INPUT_COST_PER_MTOK: float = max(
    0.0, float(_env("DEEPSEEK_INPUT_COST_PER_MTOK", "0") or "0")
)
DEEPSEEK_CACHED_INPUT_COST_PER_MTOK: float = max(
    0.0, float(_env("DEEPSEEK_CACHED_INPUT_COST_PER_MTOK", "0") or "0")
)
DEEPSEEK_OUTPUT_COST_PER_MTOK: float = max(
    0.0, float(_env("DEEPSEEK_OUTPUT_COST_PER_MTOK", "0") or "0")
)
MAX_CHILD_IDS_PER_GENERATION: int = max(
    1, int(_env("MAX_CHILD_IDS_PER_GENERATION", "1000") or "1000")
)
MAX_LOGO_BYTES: int = max(1, int(_env("MAX_LOGO_BYTES", "2097152") or "2097152"))
IDEMPOTENCY_KEY_MAX_LENGTH: int = max(
    16, int(_env("IDEMPOTENCY_KEY_MAX_LENGTH", "200") or "200")
)
IDEMPOTENCY_TTL_SECONDS: int = max(
    3600, int(_env("IDEMPOTENCY_TTL_SECONDS", "86400") or "86400")
)
REQUIRE_IDEMPOTENCY_KEY: bool = _env_bool(
    "REQUIRE_IDEMPOTENCY_KEY", IS_PRODUCTION
)
GENERATION_REQUESTS_PER_MINUTE: int = max(
    1, int(_env("GENERATION_REQUESTS_PER_MINUTE", "2") or "2")
)
PARSING_TIMEOUT_SECONDS: int = max(
    1, int(_env("PARSING_TIMEOUT_SECONDS", "120") or "120")
)
GENERATION_TIMEOUT_SECONDS: int = max(
    1, int(_env("GENERATION_TIMEOUT_SECONDS", "420") or "420")
)
JOB_HEARTBEAT_SECONDS: int = max(5, int(_env("JOB_HEARTBEAT_SECONDS", "30") or "30"))
JOB_STALE_AFTER_SECONDS: int = max(
    GENERATION_TIMEOUT_SECONDS + 30,
    int(_env("JOB_STALE_AFTER_SECONDS", "480") or "480"),
)
ABANDONED_UPLOAD_AFTER_SECONDS: int = max(
    JOB_STALE_AFTER_SECONDS,
    int(_env("ABANDONED_UPLOAD_AFTER_SECONDS", "86400") or "86400"),
)
TERMINAL_JOB_RETENTION_DAYS: int = max(
    1, int(_env("TERMINAL_JOB_RETENTION_DAYS", "30") or "30")
)
TEMP_EXPORT_RETENTION_SECONDS: int = max(
    300, int(_env("TEMP_EXPORT_RETENTION_SECONDS", "86400") or "86400")
)
CLEANUP_BATCH_SIZE: int = max(
    1, min(1000, int(_env("CLEANUP_BATCH_SIZE", "100") or "100"))
)
ORPHAN_SCAN_MAX_POINTS: int = max(
    100, int(_env("ORPHAN_SCAN_MAX_POINTS", "10000") or "10000")
)

# PostgreSQL is the durable source for evaluation history.  It is optional at
# runtime so a telemetry outage never prevents the core exam-generation flow.
DATABASE_URL: str | None = _env("DATABASE_URL")

LOG_LEVEL: str = _env("LOG_LEVEL", "INFO") or "INFO"
METRICS_TOKEN: str | None = _env("METRICS_TOKEN")
CORS_ALLOW_ORIGINS: tuple[str, ...] = tuple(
    origin.strip()
    for origin in (_env(
        "CORS_ALLOW_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ) or "").split(",")
    if origin.strip()
)
CORS_ALLOW_METHODS: tuple[str, ...] = tuple(
    method.strip().upper()
    for method in (_env("CORS_ALLOW_METHODS", "GET,POST,OPTIONS") or "").split(",")
    if method.strip()
)
CORS_ALLOW_HEADERS: tuple[str, ...] = tuple(
    header.strip()
    for header in (
        _env(
            "CORS_ALLOW_HEADERS",
            "Content-Type,Idempotency-Key,X-Request-ID,Authorization",
        )
        or ""
    ).split(",")
    if header.strip()
)
ALLOWED_HOSTS: tuple[str, ...] = tuple(
    host.strip()
    for host in (_env("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver") or "").split(",")
    if host.strip()
)


def validate_runtime_config() -> None:
    """Reject unsafe production settings before accepting traffic."""
    if not IS_PRODUCTION:
        return
    problems: list[str] = []
    if not SESSION_COOKIE_SECURE:
        problems.append("SESSION_COOKIE_SECURE must be true")
    def placeholder(value: str | None) -> bool:
        lowered = (value or "").lower()
        return not lowered or "replace" in lowered or "change_me" in lowered or "your_" in lowered

    if placeholder(SESSION_HASH_PEPPER):
        problems.append("SESSION_HASH_PEPPER must be a non-placeholder secret")
    if not DATABASE_URL:
        problems.append("DATABASE_URL is required")
    if ENABLE_LEGACY_SYNC_API:
        problems.append("ENABLE_LEGACY_SYNC_API must be false")
    if not REQUIRE_IDEMPOTENCY_KEY:
        problems.append("REQUIRE_IDEMPOTENCY_KEY must be true")
    if not CORS_ALLOW_ORIGINS:
        problems.append("CORS_ALLOW_ORIGINS must contain the production frontend origin")
    for origin in CORS_ALLOW_ORIGINS:
        lowered = origin.lower()
        if origin == "*" or "localhost" in lowered or "127.0.0.1" in lowered:
            problems.append(f"CORS origin is not allowed in production: {origin}")
        elif not lowered.startswith("https://"):
            problems.append(f"CORS origin must use HTTPS in production: {origin}")
    if not ALLOWED_HOSTS or "*" in ALLOWED_HOSTS:
        problems.append("ALLOWED_HOSTS must contain explicit production hostnames")
    if not SESSION_COOKIE_NAME.startswith("__Host-"):
        problems.append("SESSION_COOKIE_NAME must use the __Host- prefix")
    if SESSION_COOKIE_SAMESITE == "none":
        problems.append("SameSite=None is disabled until explicit CSRF protection is configured")
    if LLM_PROVIDER == "deepseek" and placeholder(DEEPSEEK_API_KEY):
        problems.append("DEEPSEEK_API_KEY is required for the DeepSeek provider")
    if placeholder(LLAMA_PARSE_API):
        problems.append("LLAMA_PARSE_API is required")
    if placeholder(METRICS_TOKEN) or len(METRICS_TOKEN or "") < 24:
        problems.append("METRICS_TOKEN must contain at least 24 characters")
    if placeholder(CLERK_PUBLISHABLE_KEY):
        problems.append("CLERK_PUBLISHABLE_KEY is required")
    if placeholder(CLERK_SECRET_KEY):
        problems.append("CLERK_SECRET_KEY is required")
    if not CLERK_AUTHORIZED_PARTIES:
        problems.append("CLERK_AUTHORIZED_PARTIES must contain the production frontend origin")
    for origin in CLERK_AUTHORIZED_PARTIES:
        lowered = origin.lower()
        if "localhost" in lowered or "127.0.0.1" in lowered:
            problems.append(f"Clerk authorized party is not allowed in production: {origin}")
        elif not lowered.startswith("https://"):
            problems.append(f"Clerk authorized party must use HTTPS in production: {origin}")
    if not DATABASE_URL or "replace" in DATABASE_URL.lower() or "your_" in DATABASE_URL.lower():
        problems.append("DATABASE_URL must not contain placeholder credentials")
    qdrant_url = urlparse(QDRANT_URL)
    qdrant_is_local = qdrant_url.hostname in {"localhost", "127.0.0.1", "qdrant"}
    if not qdrant_is_local and placeholder(QDRANT_API_KEY):
        problems.append("QDRANT_API_KEY is required for remote Qdrant")
    if not qdrant_is_local and qdrant_url.scheme != "https":
        problems.append("QDRANT_URL must use HTTPS outside the local container network")
    if DATABASE_URL:
        database_url = urlparse(DATABASE_URL)
        database_is_local = database_url.hostname in {
            "localhost", "127.0.0.1", "postgres"
        }
        ssl_mode = parse_qs(database_url.query).get("sslmode", [""])[0]
        if not database_is_local and ssl_mode not in {"require", "verify-ca", "verify-full"}:
            problems.append("Remote DATABASE_URL must require TLS with sslmode")
    if problems:
        raise RuntimeError("Unsafe production configuration: " + "; ".join(problems))

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
TITLE_CONTEXT_RECENT: int = int(_env("TITLE_CONTEXT_RECENT", "12") or "12")

# Descriptive navigation-label bounds. Prompts prefer 4-12 words while 15 is
# retained as an absolute guard against sentence-length output.
SECTION_TITLE_MIN_WORDS: int = int(_env("SECTION_TITLE_MIN_WORDS", "2") or "2")
SECTION_TITLE_MAX_WORDS: int = int(_env("SECTION_TITLE_MAX_WORDS", "15") or "15")
SUBSECTION_TITLE_MIN_WORDS: int = int(_env("SUBSECTION_TITLE_MIN_WORDS", "2") or "2")
SUBSECTION_TITLE_MAX_WORDS: int = int(_env("SUBSECTION_TITLE_MAX_WORDS", "15") or "15")

# Fallback (safety-net) label caps.
FALLBACK_SECTION_MAX_WORDS: int = int(_env("FALLBACK_SECTION_MAX_WORDS", "12") or "12")
FALLBACK_SUBSECTION_MAX_WORDS: int = int(
    _env("FALLBACK_SUBSECTION_MAX_WORDS", "12") or "12"
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
