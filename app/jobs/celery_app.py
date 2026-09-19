from __future__ import annotations

from celery import Celery

from app import config


celery_app = Celery(
    "genexam",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
    include=["app.jobs.tasks"],
)

celery_app.conf.update(
    task_always_eager=config.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=True,
    task_serializer="json",
    result_serializer="json",
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
