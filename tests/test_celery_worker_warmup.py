from __future__ import annotations

import inspect
import threading
import time

from app.jobs import celery_app
from app.jobs.tasks import generate_exam, warmup_embeddings
from app.offline import embeddings


def test_worker_does_not_load_cuda_embeddings_during_prefork_startup():
    source = inspect.getsource(celery_app)

    assert "worker_process_init" not in source
    assert "warmup()" not in source


def test_worker_ready_queues_process_local_embedding_warmup(monkeypatch):
    queued = []
    monkeypatch.setattr(celery_app.config, "CELERY_TASK_ALWAYS_EAGER", False)
    monkeypatch.setattr(
        celery_app.celery_app, "send_task", lambda name: queued.append(name)
    )

    celery_app.schedule_embedding_warmup()

    assert queued == ["genexam.warmup_embeddings"]


def test_embedding_warmup_is_single_flight_per_process(monkeypatch):
    calls = []

    class FakeModel:
        def encode(self, *_args, **_kwargs):
            calls.append("encode")
            time.sleep(0.03)
            return {"dense_vecs": [[0.0]], "lexical_weights": [{}]}

    monkeypatch.setattr(embeddings, "_warmed", False)
    monkeypatch.setattr(embeddings, "get_model", lambda: FakeModel())
    monkeypatch.setattr(embeddings, "_get_device", lambda: "cuda")
    threads = [threading.Thread(target=embeddings.warmup) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == ["encode"]


def test_warmup_task_loads_in_the_task_process(monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings, "warmup", lambda: calls.append("warmup"))
    monkeypatch.setattr(embeddings, "device_name", lambda: "cuda")

    assert warmup_embeddings.run() == {"status": "ready", "device": "cuda"}
    assert calls == ["warmup"]


def test_exam_task_timeout_allows_the_initial_cuda_model_load():
    # BGE-M3 cold loading is expected to take roughly 40-70 seconds. It now
    # occurs inside the first real embedding call rather than during fork.
    assert generate_exam.soft_time_limit >= 90
    assert generate_exam.time_limit > generate_exam.soft_time_limit
