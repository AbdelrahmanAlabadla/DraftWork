from __future__ import annotations

from contextlib import contextmanager

from app.offline.chunk_store import ChunkStore
from app.online import retrieval


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.executed = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        self.executed.append((" ".join(sql.split()), params))
        self.last_sql = sql
        if sql.lstrip().startswith("DELETE"):
            self.rowcount = 2

    def fetchall(self):
        return self.rows

    def fetchone(self):
        if "FROM jobs j JOIN documents d" in getattr(self, "last_sql", ""):
            return (
                "running", 1, "33333333-3333-4333-8333-333333333333",
            )
        return (len(self.rows),)


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


def _structure(document_id="11111111-1111-4111-8111-111111111111"):
    return {
        "parents": [{
            "parent_id": "parent-a", "document_id": document_id,
            "title": "Linear equations", "page_start": 2, "page_end": 4,
        }],
        "children": [
            {
                "child_id": "child-a", "document_id": document_id,
                "parent_id": "parent-a", "title": "Solving", "content": "First",
                "child_order": 0, "page_start": 2, "page_end": 2,
            },
            {
                "child_id": "child-b", "document_id": document_id,
                "parent_id": "parent-a", "title": "Checking", "content": "Second",
                "child_order": 1, "page_start": 3, "page_end": 4,
            },
        ],
    }


def test_replace_document_persists_parent_child_relationships(monkeypatch):
    cursor = FakeCursor()
    connection = FakeConnection(cursor)

    @contextmanager
    def connected():
        yield connection

    monkeypatch.setattr("app.offline.chunk_store.db.connection", connected)
    count = ChunkStore().replace_document(
        _structure(), document_id="11111111-1111-4111-8111-111111111111",
        session_id="22222222-2222-4222-8222-222222222222",
        job_id="33333333-3333-4333-8333-333333333333",
        claim_generation=1,
        processed_document_key="cache-key", processing_versions={"chunker": "v1"},
    )

    assert count == 2
    assert connection.committed is True
    assert any(sql.startswith("DELETE FROM document_chunks") for sql, _ in cursor.executed)
    inserts = [params for sql, params in cursor.executed if sql.startswith("INSERT")]
    assert len(inserts) == 3
    assert inserts[0][4] == "parent-a"
    assert inserts[1][4:7] == ("parent-a", "parent-a", "child-a")


def test_exact_child_lookup_preserves_requested_order_and_document_scope(monkeypatch):
    rows = [
        ("child-b", "parent-a", "Checking", "Linear equations", "Second", 1, 3, 4, None, {}),
        ("child-a", "parent-a", "Solving", "Linear equations", "First", 0, 2, 2, None, {}),
    ]
    cursor = FakeCursor(rows)
    connection = FakeConnection(cursor)

    @contextmanager
    def connected():
        yield connection

    monkeypatch.setattr("app.offline.chunk_store.db.connection", connected)
    result = ChunkStore().get_by_child_ids(
        "11111111-1111-4111-8111-111111111111", ["child-a", "child-b"],
        session_id="22222222-2222-4222-8222-222222222222",
    )

    assert [row["child_id"] for row in result] == ["child-a", "child-b"]
    sql, params = cursor.executed[0]
    assert "document_id = %s" in sql and "chunk_id = ANY(%s)" in sql
    assert params[0] == "11111111-1111-4111-8111-111111111111"
    assert result[0]["parent_id"] == "parent-a"


def test_retrieval_backfills_cached_structure_when_postgres_is_empty(monkeypatch):
    calls = []

    class Store:
        def get_by_child_ids(self, document_id, child_ids, **_kwargs):
            calls.append(("get", document_id, tuple(child_ids)))
            if len(calls) == 1:
                return []
            return [{
                "child_id": child_ids[0], "parent_id": "parent-a",
                "parent_title": "Linear equations", "chunk_title": "Solving",
                "content": "Solve the equation.",
            }]

        def replace_document(self, structure, **kwargs):
            calls.append(("replace", kwargs["document_id"], len(structure["children"])))

    monkeypatch.setattr(retrieval, "ChunkStore", Store)
    monkeypatch.setattr(retrieval, "load_processed_document_by_key", lambda _key: {
        "structure": _structure("11111111-1111-4111-8111-111111111111"),
        "processing_versions": {"chunker": "v1"},
    })
    result = retrieval.retrieve_selected(
        "11111111-1111-4111-8111-111111111111", ["child-a"],
        processed_document_key="cache-key",
    )

    assert result["error"] is None
    assert calls[1][0] == "replace"
    assert result["retrieved_chunks"][0]["content"] == "Solve the equation."
