"""Backfill evaluation outcome columns from persisted generation audits.

Historical evaluation records predate the per-run outcome columns added by
migration 0009, but their authoritative `slot_lifecycle`, repair history,
duplicate bookkeeping and catalog-cache flags were already persisted in
``exams.metadata.generation_audit``.  This script derives the same summaries the
worker now writes, and only fills columns that are still empty.

Usage:
    python scripts/backfill_evaluation_analytics.py --dry-run
    python scripts/backfill_evaluation_analytics.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from app import db  # noqa: E402
from app.api.benchmarks import _git_commit  # noqa: E402
from app.online.eval_stats import summarize_pipeline_outcomes  # noqa: E402


def _audit_payload(audit: dict) -> dict:
    """Project the persisted audit into the worker's outcome-summary input."""
    return {
        "slot_lifecycle": audit.get("slot_lifecycle") or {},
        "validation_repair_history": audit.get("validation_repair_history") or [],
        "duplicate_repair_rounds": audit.get("duplicate_repair_rounds") or 0,
        "duplicate_replacement_attempts": audit.get("duplicate_replacement_attempts") or {},
        "duplicate_replacement_failures": audit.get("duplicate_replacement_failures") or {},
        "unresolved_duplicate_conflicts": audit.get("unresolved_duplicate_conflicts") or [],
        "assessment_catalog_cache_hit": bool(audit.get("assessment_catalog_cache_hit")),
        "assessment_compiler_cache_hit": bool(audit.get("assessment_compiler_cache_hit")),
        "assessment_catalog_compatibility_cache_hit": bool(
            audit.get("assessment_catalog_compatibility_cache_hit")
        ),
    }


def backfill(*, dry_run: bool = False, force: bool = False) -> dict[str, int]:
    stats = {"candidates": 0, "updated": 0, "no_audit": 0, "skipped": 0}
    with db.connection() as conn:
        with conn.cursor() as cursor:
            if force:
                cursor.execute(
                    """SELECT r.exam_id, e.metadata
                         FROM evaluation_runs r
                         JOIN exams e ON e.id = r.exam_id"""
                )
            else:
                cursor.execute(
                    """SELECT r.exam_id, e.metadata
                         FROM evaluation_runs r
                         JOIN exams e ON e.id = r.exam_id
                        WHERE (
                            r.recovery_outcomes = '{}'::jsonb
                            OR r.duplicate_outcomes = '{}'::jsonb
                            OR r.catalog_cache = '{}'::jsonb
                            OR NOT (r.recovery_outcomes ? 'attempt_strategy_detail_outcomes')
                        )"""
                )
            rows = cursor.fetchall()
            stats["candidates"] = len(rows)
            for exam_id, metadata in rows:
                audit = ((metadata or {}).get("generation_audit") or {})
                if not isinstance(audit, dict) or not audit:
                    stats["no_audit"] += 1
                    continue
                outcomes = summarize_pipeline_outcomes(_audit_payload(audit))
                if not outcomes:
                    stats["skipped"] += 1
                    continue
                if dry_run:
                    stats["updated"] += 1
                    continue
                cursor.execute(
                    """UPDATE evaluation_runs
                          SET recovery_outcomes = %s,
                              duplicate_outcomes = CASE
                                  WHEN duplicate_outcomes = '{}'::jsonb
                                  THEN %s ELSE duplicate_outcomes END,
                              catalog_cache = CASE
                                  WHEN catalog_cache = '{}'::jsonb
                                  THEN %s ELSE catalog_cache END
                        WHERE exam_id = %s""",
                    (
                        Jsonb(outcomes.get("recovery") or {}),
                        Jsonb(outcomes.get("duplicates") or {}),
                        Jsonb(outcomes.get("catalog_cache") or {}),
                        str(exam_id),
                    ),
                )
                stats["updated"] += 1
            if dry_run:
                conn.rollback()
            else:
                conn.commit()
    return stats


def backfill_benchmark_commits(*, dry_run: bool = False) -> int:
    """Fill a missing git commit on benchmark reproducibility metadata."""
    commit = _git_commit()
    if not commit:
        return 0
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT COUNT(*) FROM evaluation_benchmarks
                    WHERE reproducibility->>'git_commit' IS NULL"""
            )
            count = int(cursor.fetchone()[0])
            if dry_run or not count:
                if dry_run:
                    conn.rollback()
                return count
            cursor.execute(
                """UPDATE evaluation_benchmarks
                      SET reproducibility = jsonb_set(
                          reproducibility, '{git_commit}', to_jsonb(%s::text), true
                      )
                    WHERE reproducibility->>'git_commit' IS NULL""",
                (commit,),
            )
        conn.commit()
    return count


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute every audit-linked record")
    args = parser.parse_args()
    stats = backfill(dry_run=args.dry_run, force=args.force)
    commits = backfill_benchmark_commits(dry_run=args.dry_run)
    print(("DRY RUN " if args.dry_run else "") + f"backfill: {stats}, benchmark_commits: {commits}")


if __name__ == "__main__":
    main()
