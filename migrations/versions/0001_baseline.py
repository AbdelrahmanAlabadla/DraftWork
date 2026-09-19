"""Baseline the durable GenExam schema.

Revision ID: 0001_baseline
Revises: None
"""
from __future__ import annotations

from pathlib import Path

from alembic import context, op


revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    migration_root = Path(__file__).resolve().parents[1]
    scripts = [
        (migration_root / name).read_text(encoding="utf-8")
        for name in (
            "001_create_evaluation_runs.sql",
            "002_create_core_tables.sql",
            "003_deployment_hardening.sql",
        )
    ]
    if context.is_offline_mode():
        for script in scripts:
            op.execute(script)
        return
    raw_connection = op.get_bind().connection.driver_connection
    with raw_connection.cursor() as cursor:
        for script in scripts:
            cursor.execute(script, prepare=False)


def downgrade() -> None:
    # The baseline is intentionally irreversible. Production records, most
    # notably generated exams, must never be dropped by an automated rollback.
    pass
