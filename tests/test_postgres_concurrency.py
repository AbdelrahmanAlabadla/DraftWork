from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from psycopg.types.json import Jsonb

from app import db
from app.api import repositories
from app.offline.chunk_store import ChunkStore


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_POSTGRES_CONCURRENCY_TESTS") != "1",
    reason="requires an isolated migrated PostgreSQL database",
)


def _ids():
    return tuple(str(uuid.uuid4()) for _ in range(5))


def _seed_job(*, kind="exam_generation", document_id=None):
    session_id, generated_document, job_id, exam_uuid, _ = _ids()
    document_id = document_id or generated_document
    exam_id = f"exam_{exam_uuid.replace('-', '')[:12]}"
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO sessions (id, token_hash, expires_at) VALUES (%s,%s,CURRENT_TIMESTAMP + INTERVAL '1 day')",
                (session_id, uuid.uuid4().hex),
            )
            cursor.execute(
                """INSERT INTO documents (
                       id, session_id, status, original_filename, media_type,
                       size_bytes, sha256, source_storage_key
                   ) VALUES (%s,%s,'ready','math.pdf','application/pdf',1,%s,%s)""",
                (document_id, session_id, uuid.uuid4().hex * 2, f"source/{document_id}"),
            )
            cursor.execute(
                """INSERT INTO jobs (
                       id, session_id, document_id, type, status, stage,
                       request_data, max_attempts
                   ) VALUES (%s,%s,%s,%s,'queued','queued',%s,10)""",
                (job_id, session_id, document_id, kind, Jsonb({"exam_id": exam_id})),
            )
            if kind == "exam_generation":
                cursor.execute(
                    """INSERT INTO exams (
                           id, session_id, document_id, job_id, status, stage,
                           metadata, exams, warnings, evaluation
                       ) VALUES (%s,%s,%s,%s,'queued','queued','{}','[]','[]','{}')""",
                    (exam_id, session_id, document_id, job_id),
                )
        conn.commit()
    return session_id, document_id, job_id, exam_id


def _structure(document_id: str, suffix: str):
    parent = f"parent-{suffix}"
    child = f"child-{suffix}"
    return {
        "parents": [{"parent_id": parent, "document_id": document_id, "title": suffix}],
        "children": [{
            "child_id": child, "document_id": document_id, "parent_id": parent,
            "title": suffix, "content": suffix, "child_order": 0,
        }],
    }


def test_stale_worker_is_fenced_after_repeated_takeovers():
    _session, _document, job_id, exam_id = _seed_job()
    current = repositories.claim_job(job_id, "worker-a")
    assert current
    generations = [int(current["claim_generation"])]
    for number in range(3):
        with db.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE jobs SET status='retrying', worker_id=NULL WHERE id=%s",
                    (job_id,),
                )
            conn.commit()
        current = repositories.claim_job(job_id, f"worker-{number + 2}")
        assert current
        generations.append(int(current["claim_generation"]))
    assert generations == sorted(set(generations))
    stale = generations[-2]
    active = generations[-1]
    for operation in (
        lambda: repositories.update_job_progress(
            job_id, "old", 10, claim_generation=stale
        ),
        lambda: repositories.heartbeat_job(job_id, claim_generation=stale),
        lambda: repositories.fail_job(
            job_id, "OLD", "stale", claim_generation=stale
        ),
        lambda: repositories.retry_job(
            job_id, "OLD", "stale", claim_generation=stale
        ),
        lambda: repositories.append_exam_audit_event(
            exam_id, job_id=job_id, claim_generation=stale,
            checkpoint_revision=1, request_fingerprint="request",
            stage="generation", event_type="generation_checkpoint",
            payload={"checkpoint_state": {"value": "old"}},
        ),
    ):
        with pytest.raises(repositories.StaleJobExecution):
            operation()
    assert repositories.complete_generation_job(
        job_id, exam_id=exam_id, document_id=_document,
        session_id=_session, claim_generation=stale, status="complete",
        exams=[], warnings=[], metadata={}, evaluation={},
    ) is False
    repositories.update_job_progress(
        job_id, "current", 50, claim_generation=active
    )


def test_checkpoint_revisions_reject_replay_and_restore_latest():
    _session, _document, job_id, exam_id = _seed_job()
    job = repositories.claim_job(job_id, "worker")
    generation = int(job["claim_generation"])
    for revision in range(1, 7):
        value = 10 if revision < 6 else 18
        repositories.append_exam_audit_event(
            exam_id, job_id=job_id, claim_generation=generation,
            checkpoint_revision=revision, request_fingerprint="request",
            stage="generation", event_type="generation_checkpoint",
            payload={"checkpoint_state": {"completed_questions": value}},
        )
    with pytest.raises(repositories.StaleCheckpoint):
        repositories.append_exam_audit_event(
            exam_id, job_id=job_id, claim_generation=generation,
            checkpoint_revision=5, request_fingerprint="request",
            stage="generation", event_type="generation_checkpoint",
            payload={"checkpoint_state": {"completed_questions": 10}},
        )
    restored = repositories.load_generation_checkpoint(
        job_id, claim_generation=generation, request_fingerprint="request"
    )
    assert restored["completed_questions"] == 18
    assert restored["checkpoint_revision"] == 6


def test_same_document_replacement_is_atomic_and_old_publisher_is_fenced():
    session_id, document_id, job_id, _exam = _seed_job(kind="document_ingestion")
    job = repositories.claim_job(job_id, "index-a")
    generation = int(job["claim_generation"])
    store = ChunkStore()
    store.replace_document(
        _structure(document_id, "original"), document_id=document_id,
        session_id=session_id, job_id=job_id, claim_generation=generation,
    )
    broken = _structure(document_id, "broken")
    broken["children"].append(dict(broken["children"][0]))
    with pytest.raises(Exception):
        store.replace_document(
            broken, document_id=document_id, session_id=session_id,
            job_id=job_id, claim_generation=generation,
        )
    assert store.get_by_child_ids(document_id, ["child-original"])

    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE jobs SET status='retrying' WHERE id=%s", (job_id,))
        conn.commit()
    replacement = repositories.claim_job(job_id, "index-b")
    with pytest.raises(repositories.StaleJobExecution):
        store.replace_document(
            _structure(document_id, "stale"), document_id=document_id,
            session_id=session_id, job_id=job_id, claim_generation=generation,
        )
    store.replace_document(
        _structure(document_id, "new"), document_id=document_id,
        session_id=session_id, job_id=job_id,
        claim_generation=int(replacement["claim_generation"]),
    )
    assert store.get_by_child_ids(document_id, ["child-new"])
    assert not store.get_by_child_ids(document_id, ["child-original"])


def test_simultaneous_same_document_replacements_serialize_cleanly():
    session_id, document_id, job_id, _exam = _seed_job(kind="document_ingestion")
    job = repositories.claim_job(job_id, "index-worker")
    generation = int(job["claim_generation"])
    barrier = threading.Barrier(2)

    def replace(label: str):
        barrier.wait(timeout=5)
        return ChunkStore().replace_document(
            _structure(document_id, label), document_id=document_id,
            session_id=session_id, job_id=job_id, claim_generation=generation,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(replace, ("one", "two"))) == [1, 1]
    present = sum(bool(ChunkStore().get_by_child_ids(document_id, [f"child-{label}"]))
                  for label in ("one", "two"))
    assert present == 1


def test_checkpoint_write_failure_rolls_back_revision():
    _session, _document, job_id, exam_id = _seed_job()
    job = repositories.claim_job(job_id, "worker")
    generation = int(job["claim_generation"])
    with pytest.raises(Exception):
        repositories.append_exam_audit_event(
            exam_id, job_id=job_id, claim_generation=generation,
            checkpoint_revision=1, request_fingerprint="request",
            stage="generation", event_type="generation_checkpoint",
            payload={"checkpoint_state": {"not_json": object()}},
        )
    current = repositories.get_job(job_id)
    assert int(current["checkpoint_revision"]) == 0


def test_different_documents_replace_concurrently_without_cross_locking():
    first = _seed_job(kind="document_ingestion")
    second = _seed_job(kind="document_ingestion")
    jobs = []
    for label, seeded in zip(("a", "b"), (first, second)):
        session_id, document_id, job_id, _exam = seeded
        job = repositories.claim_job(job_id, f"worker-{label}")
        jobs.append((session_id, document_id, job_id, int(job["claim_generation"]), label))
    barrier = threading.Barrier(2)

    def replace(values):
        session_id, document_id, job_id, generation, label = values
        barrier.wait(timeout=5)
        return ChunkStore().replace_document(
            _structure(document_id, label), document_id=document_id,
            session_id=session_id, job_id=job_id, claim_generation=generation,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(replace, jobs)) == [1, 1]


def test_two_exam_jobs_have_independent_fences_and_checkpoints():
    first = _seed_job()
    second = _seed_job()
    barrier = threading.Barrier(2)

    def run(values):
        fingerprint, seeded = values
        _session, _document, job_id, exam_id = seeded
        barrier.wait(timeout=5)
        job = repositories.claim_job(job_id, fingerprint)
        generation = int(job["claim_generation"])
        repositories.append_exam_audit_event(
            exam_id, job_id=job_id, claim_generation=generation,
            checkpoint_revision=1, request_fingerprint=fingerprint,
            stage="planning", event_type="plan_completed",
            payload={"node_output": {"owner": fingerprint}},
        )
        restored = repositories.load_generation_checkpoint(
            job_id, claim_generation=generation, request_fingerprint=fingerprint
        )
        assert restored["owner"] == fingerprint
        return job_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(run, zip(("request-a", "request-b"), (first, second))))
    assert len(set(completed)) == 2
