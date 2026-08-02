from __future__ import annotations

from fastapi.testclient import TestClient

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.api.main import app  # noqa: E402
from app.llm.json_utils import extract_json  # noqa: E402

client = TestClient(app)


def _install_fakes(monkeypatch, document_id: str = "doc-x") -> None:
    # Registry fakes (avoid writing data/documents.json).
    monkeypatch.setattr("app.api.storage.get_current_document", lambda: document_id)
    monkeypatch.setattr(
        "app.api.storage.get_document",
        lambda doc_id: {"document_id": doc_id, "filename": "x.pdf"},
    )

    # Embeddings fake.
    monkeypatch.setattr(
        "app.online.retrieval.embed_texts",
        lambda texts, batch_size=16: [
            {"dense": [0.1, 0.2, 0.3, 0.4], "sparse": {1: 0.5}} for _ in texts
        ],
    )

    # Vector store fake.
    class FakeStore:
        def hybrid_search(self, dense, sparse, document_id, top_k=6):
            return [
                {
                    "child_id": f"c{i}",
                    "parent_id": "p1",
                    "page": 1,
                    "heading": "Heading",
                    "content": f"Context paragraph number {i} with useful facts.",
                    "score": 1.0,
                }
                for i in range(top_k)
            ]

    monkeypatch.setattr("app.online.retrieval.VectorStore", FakeStore)

    # LLM fake: return JSON based on the requested type in the prompt, with
    # unique question text per call so the dedup layer never collapses them.
    counter = {"n": 0}

    def fake_chat(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, timeout=600):
        counter["n"] += 1
        if "Multiple Choice" in prompt:
            return (
                f'{{"questions":[{{"question":"Which layer collects data? (v{counter["n"]})",'
                f'"options":{{"A":"Perception","B":"Network","C":"Application","D":"Middleware"}},'
                f'"correct_answer":"A"}}]}}'
            )
        if "True/False" in prompt:
            return f'{{"questions":[{{"statement":"MQTT is a network protocol. (v{counter["n"]})","answer":"True"}}]}}'
        return f'{{"questions":[{{"question":"Why is load balancing used? (v{counter["n"]})","reference_answer":"To avoid overload."}}]}}'

    class FakeLLM:
        def chat(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, timeout=600):
            return fake_chat(
                self, prompt, system_prompt=system_prompt, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout,
            )

        def chat_json(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096,
                      timeout=600, max_repair_attempts=2):
            raw = fake_chat(
                self, prompt, system_prompt=system_prompt, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout,
            )
            return extract_json(raw)

    monkeypatch.setattr("app.online.generator.LMStudioClient", FakeLLM)


def test_generate_single_type_mcq(monkeypatch):
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate", json={"document_id": "doc-x", "mcq_count": 1}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "Multiple Choice" in data["exam"]
    assert "Answer: A" in data["exam"]
    assert data["questions"]["mcq"][0]["correct_answer"] == "A"


def test_generate_html_payload_multiple_types(monkeypatch):
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate",
        json={
            "mcq_count": 1,
            "tf_count": 1,
            "why_count": 1,
            "fitb_count": 3,  # ignored in V1
            "essay_count": 2,  # ignored in V1
            "num_models": 2,   # ignored in V1
            "difficulty": "hard",  # ignored in V1
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["questions"].keys()) == {"mcq", "true_false", "short_answer"}
    assert "Multiple Choice" in data["exam"]
    assert "True / False" in data["exam"]
    assert "Short Answer" in data["exam"]


def test_generate_no_supported_types_returns_400(monkeypatch):
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate", json={"fitb_count": 3, "essay_count": 2}
    )
    assert resp.status_code == 400
    assert "No supported question types" in resp.json()["detail"]


def test_generate_unknown_document_returns_404(monkeypatch):
    _install_fakes(monkeypatch, document_id="doc-other")
    monkeypatch.setattr(
        "app.api.storage.get_document", lambda doc_id: None
    )
    resp = client.post(
        "/generate", json={"document_id": "nope", "mcq_count": 1}
    )
    assert resp.status_code == 404


def test_generate_fills_requested_counts_via_retry(monkeypatch):
    # The fake LLM returns exactly 1 question per call; requesting 2 must retry
    # until the count is met, then render continuously-numbered sections.
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate", json={"document_id": "doc-x", "mcq_count": 2, "tf_count": 2}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["questions"]["mcq"]) == 2
    assert len(data["questions"]["true_false"]) == 2
    md = data["exam"]
    assert md.index("1. Which layer collects data?") < md.index("2. Which layer collects data?")
    assert md.index("## True / False") > md.index("2. Which layer collects data?")


def test_generate_repairs_malformed_json(monkeypatch):
    # Fake LLM returns broken JSON on the generation call and valid JSON on the
    # repair call; the request must still succeed via chat_json's repair loop.
    from app.llm.json_utils import REPAIR_SYSTEM_PROMPT, build_repair_prompt, extract_json

    calls = {"n": 0}

    def fake_chat(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, timeout=600):
        calls["n"] += 1
        if system_prompt == REPAIR_SYSTEM_PROMPT:
            return (
                '{"questions":[{"question":"Repaired question? (v%d)",'
                '"options":{"A":"a","B":"b","C":"c","D":"d"},"correct_answer":"B"}]}' % calls["n"]
            )
        return '{"questions":[{"question": "broken",}'  # malformed JSON

    class FakeLLM:
        def chat(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, timeout=600):
            return fake_chat(
                self, prompt, system_prompt=system_prompt, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout,
            )

        def chat_json(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096,
                      timeout=600, max_repair_attempts=2):
            raw = self.chat(
                prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
            )
            for attempt in range(max_repair_attempts + 1):
                try:
                    return extract_json(raw)
                except Exception as exc:
                    if attempt == max_repair_attempts:
                        raise
                    repair_prompt = build_repair_prompt(raw, str(exc))
                    raw = self.chat(
                        repair_prompt,
                        system_prompt=REPAIR_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_tokens=max_tokens * 2,
                    )

    monkeypatch.setattr("app.api.storage.get_current_document", lambda: "doc-x")
    monkeypatch.setattr(
        "app.api.storage.get_document",
        lambda doc_id: {"document_id": doc_id, "filename": "x.pdf"},
    )
    monkeypatch.setattr(
        "app.online.retrieval.embed_texts",
        lambda texts, batch_size=16: [
            {"dense": [0.1, 0.2, 0.3, 0.4], "sparse": {1: 0.5}} for _ in texts
        ],
    )

    class FakeStore:
        def hybrid_search(self, dense, sparse, document_id, top_k=6):
            return [
                {
                    "child_id": f"c{i}", "parent_id": "p1", "page": 1,
                    "heading": "Heading", "content": f"Context {i}.", "score": 1.0,
                }
                for i in range(top_k)
            ]

    monkeypatch.setattr("app.online.retrieval.VectorStore", FakeStore)
    monkeypatch.setattr("app.online.generator.LMStudioClient", FakeLLM)

    resp = client.post("/generate", json={"document_id": "doc-x", "mcq_count": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["questions"]["mcq"][0]["question"].startswith("Repaired question?")
