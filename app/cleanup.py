from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app import config
from app.api import repositories
from app.file_storage import get_file_storage
from app.logging_conf import get_logger
from app.metrics import increment, observe
from app.offline.vector_store import VectorStore


logger = get_logger("CLEANUP")


def _delete_pipeline_artifacts(document_id: str) -> int:
    deleted = 0
    candidates = (
        config.STRUCTURES_DIR,
        config.PARSED_OUTPUT_DIR,
    )
    for directory in candidates:
        root = Path(directory)
        if not root.exists():
            continue
        for path in root.glob(f"{document_id}.*"):
            if path.is_file():
                path.unlink(missing_ok=True)
                deleted += 1
    return deleted


def run_cleanup() -> dict[str, Any]:
    """Delete abandoned/regenerable assets while retaining generated exams."""
    cleanup_id = repositories.start_cleanup_run()
    started = datetime.now(timezone.utc)
    stats: dict[str, int] = {
        "documents": 0,
        "storage_objects": 0,
        "qdrant_documents": 0,
        "temporary_exports": 0,
        "orphan_storage_prefixes": 0,
        "orphan_qdrant_points": 0,
    }
    try:
        storage = get_file_storage()
        candidates = repositories.cleanup_document_candidates(
            abandoned_after_seconds=config.ABANDONED_UPLOAD_AFTER_SECONDS,
            limit=config.CLEANUP_BATCH_SIZE,
        )
        vector_store: VectorStore | None = None
        for document in candidates:
            document_id = str(document["id"])
            session_id = str(document["session_id"])
            if vector_store is None:
                vector_store = VectorStore()
            vector_store.delete_document(
                document_id, session_id=session_id, raise_errors=True
            )
            stats["qdrant_documents"] += 1
            stats["storage_objects"] += storage.delete_tree(
                f"sessions/{session_id}/documents/{document_id}"
            )
            stats["storage_objects"] += _delete_pipeline_artifacts(document_id)
            # Exam JSON remains in PostgreSQL. Only its regenerable exports are
            # stored under the session prefix and handled by the expiry pass.
            repositories.mark_document_assets_deleted(document_id)
            stats["documents"] += 1

        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=config.TEMP_EXPORT_RETENTION_SECONDS
        )
        stats["temporary_exports"] = storage.delete_older_than(
            "sessions", cutoff, path_component="exports"
        )
        for session_id in repositories.expired_session_ids(
            limit=config.CLEANUP_BATCH_SIZE
        ):
            stats["storage_objects"] += storage.delete_tree(
                f"sessions/{session_id}"
            )
        valid_scopes = repositories.active_document_scopes()
        for prefix in storage.list_prefixes("sessions", levels=3):
            parts = prefix.split("/")
            if len(parts) == 4 and parts[2] == "documents":
                scope = (parts[1], parts[3])
                if scope not in valid_scopes:
                    stats["storage_objects"] += storage.delete_tree(prefix)
                    stats["orphan_storage_prefixes"] += 1
        if vector_store is None:
            vector_store = VectorStore()
        stats["orphan_qdrant_points"] = vector_store.delete_orphaned_session_points(
            valid_scopes,
            max_points=config.ORPHAN_SCAN_MAX_POINTS,
        )
        stats.update(
            repositories.cleanup_database_records(
                terminal_job_days=config.TERMINAL_JOB_RETENTION_DAYS
            )
        )
        repositories.finish_cleanup_run(cleanup_id, stats=stats)
        duration = (datetime.now(timezone.utc) - started).total_seconds()
        increment("genexam_cleanup_runs", status="completed")
        observe("genexam_cleanup_duration_seconds", duration)
        logger.info(
            "Cleanup completed | cleanup_id=%s | stats=%s | duration=%.3fs",
            cleanup_id,
            stats,
            duration,
            extra={"event": "cleanup_completed", "duration_ms": int(duration * 1000)},
        )
        return stats
    except Exception as exc:
        repositories.finish_cleanup_run(cleanup_id, stats=stats, error=str(exc))
        increment("genexam_cleanup_runs", status="failed")
        logger.exception(
            "Cleanup failed | cleanup_id=%s", cleanup_id,
            extra={"event": "cleanup_failed"},
        )
        raise
