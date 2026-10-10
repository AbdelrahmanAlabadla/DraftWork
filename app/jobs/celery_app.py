from __future__ import annotations

from celery import Celery
from celery.signals import worker_ready

from app import config


celery_app = Celery(
    "genexam",
    broker=config.CELERY_BROKER_URL,
    # Durable job state and results live in PostgreSQL; disabling Celery's
    # result backend also avoids serializing the SystemExit raised by a hard
    # task revocation.
    backend=None,
    include=["app.jobs.tasks"],
)

celery_app.conf.update(
    result_backend=None,
    task_always_eager=config.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=True,
    task_serializer="json",
    result_serializer="json",
    task_ignore_result=True,
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    result_expires=3600,
    beat_schedule={
        "recover-stale-jobs": {
            "task": "genexam.recover_stale_jobs",
            "schedule": 60.0,
        },
        "dispatch-undispatched-jobs": {
            "task": "genexam.dispatch_undispatched_jobs",
            "schedule": 30.0,
        },
        "cleanup-expired-data": {
            "task": "genexam.cleanup_expired_data",
            "schedule": 3600.0,
        },
    },
)


@worker_ready.connect(weak=False)
def schedule_embedding_warmup(sender=None, **_kwargs) -> None:
    """Queue CUDA warmup only after the worker can accept normal tasks."""
    if config.CELERY_TASK_ALWAYS_EAGER:
        return
    celery_app.send_task("genexam.warmup_embeddings")
