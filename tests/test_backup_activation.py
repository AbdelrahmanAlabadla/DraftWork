from __future__ import annotations

from app.online import duplicate_pipeline


def test_preplanned_backup_cannot_bypass_shared_task_ranking():
    chosen, context = duplicate_pipeline.select_preplanned_backup(
        slot_id="m1_mcq_1",
        qtype="mcq",
        candidates=[{"task_id": "task_b", "selected_variation_id": "v2"}],
        backups=[{"task_id": "task_b", "variation_id": "v2"}],
    )
    assert chosen is None
    assert context["selection_source"] == "preplanned_backup_disabled_shared_rank_required"
