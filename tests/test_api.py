from __future__ import annotations

import json
import re

from fastapi.testclient import TestClient

from app.logging_conf import configure_logging

configure_logging("WARNING")

from app.api.main import app  # noqa: E402
from app.llm.json_utils import extract_json  # noqa: E402

client = TestClient(app)

_COUNT_RE = re.compile(r"Create exactly (\d+)")
_PLANNER_MODELS_RE = re.compile(r"Plan exam questions for (\d+)")
_PLANNER_COUNT_RE = re.compile(
    r"^\s*-\s*(\d+)\s+(mcq|true_false|fill_in_the_blank|short_answer|essay)\s*$", re.M
)
_MODEL_RE = re.compile(r"Exam Model #(\d+)")


def _model_number(prompt: str) -> int:
    m = _MODEL_RE.search(prompt)
    return int(m.group(1)) if m else 1


_fitb_counter = [0]


def _fitb_terms(model: int, count: int) -> list[str]:
    """Return `count` lexically-unique answer terms for a FITB section."""
    return [f"term-{model}-{i}" for i in range(count)]


_stem_counter = [0]


def _distinct_stem(type_name: str, model: int, idx: int) -> str:
    """Return a lexically-unique stem for model>1.

    Unique hex tokens carry the content; every surrounding word is a stopword the
    near-duplicate filter strips, so surviving vocabulary (the hex tokens + model
    number) barely overlaps between any two stems.
    """
    c = _stem_counter[0] + 1 + model * 1000 + idx * 53
    _stem_counter[0] = c
    return (
        f"the {c:x} and {c * 3:b} of {c * 7:x} with {model} which are "
        f"{c * 5:d} and {c * 11:o}"
    )


def _fake_planner(prompt: str, num_models: int) -> object:
    """Return a valid, cross-model-distinct plan for every model/type count."""
    counts = {qtype: int(n) for n, qtype in _PLANNER_COUNT_RE.findall(prompt)}
    if not counts:
        counts = {"mcq": 1}
    # Distinct topical vocabulary per model so cross-model concept-overlap is low.
    model_topic = {
        1: "cell metabolism",
        2: "electron transport",
        3: "gene expression",
    }
    exams = []
    for m in range(1, num_models + 1):
        topic = model_topic.get(m, f"domain {m}")
        questions = [{"question_type": qtype, "topic": topic, "concept_to_test": f"distinct {qtype} concept {m}-{idx}"} for qtype, count in counts.items() for idx in range(count)]
        exams.append({"model_number": m, "questions": questions})
    return {"exams": exams}


def _make_fake_llm(captured_prompts: list[str] | None = None):
    """Return a FakeLLM serving BOTH roles parsed from the prompt.

    Planner prompts (containing "concept_to_test") return a valid concept plan
    covering every model/type; generator prompts return the requested count of
    distinct questions (count parsed from "Create exactly N ..."). ``captured_prompts``
    (optional) receives the raw text of every non-repair LLM call.
    """
    mcq_pool = [
        "carbohydrate monomers link via glycosidic bonds",
        "analysing a histogram reveals skewed distributions",
        "the Krebs cycle occurs within mitochondria matrix",
        "Newton's third law pairs equal opposing forces",
        "a certificate validates a public key owner",
        "osmosis transports water across a semipermeable membrane",
        "the heap stores dynamically allocated objects",
        "a ciphertext is produced through cipher operations",
        "drought reduces agricultural crop yields",
        "an interrupt triggers immediate processor handling",
    ]
    tf_pool = [
        "glucagon elevates hepatic glucose output",
        "fiscal stimulus expands aggregate demand",
        "entropy measures system disorder",
        "magnets possess paired poles",
        "convection transfers heat through fluid",
    ]
    sa_pool = [
        ("justify refrigeration reducing bacterial growth", "cooler slows division"),
        ("explain corrosion requiring oxygen plus moisture", "oxidation degrades metals"),
        ("describe absorption occurring along gut walls", "nutrients enter bloodstream"),
    ]

    def fake_chat(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, timeout=600):
        if captured_prompts is not None:
            captured_prompts.append(prompt)

        if "Plan exam questions for" in prompt:
            num_models_m = _PLANNER_MODELS_RE.search(prompt)
            num_models = int(num_models_m.group(1)) if num_models_m else 1
            return _fake_planner(prompt, num_models)

        m = _COUNT_RE.search(prompt)
        count = int(m.group(1)) if m else 1
        model = _model_number(prompt)
        options = {"A": "Perception", "B": "Network", "C": "Application", "D": "Middleware"}

        if "correct answer terms for a Fill-in-the-Blank" in prompt:
            return {
                "correct_terms": _fitb_terms(model, count),
                "distractors": [f"distractor {model} a", f"distractor {model} b"],
            }
        if "numbered Fill-in-the-Blank items" in prompt:
            terms = _fitb_terms(model, count)
            return {"items": [
                {"question": _distinct_stem("fitb", model, i), "answers": [terms[i]]}
                for i in range(count)
            ]}
        if "Essay exam question" in prompt:
            stems = [_distinct_stem("essay", model, i) for i in range(count)]
            return {"questions": [
                {
                    "question": stems[i],
                    "reference_answer": f"reference for {i} in {model}",
                    "key_points": [f"point {i}a", f"point {i}b"],
                }
                for i in range(count)
            ]}
        if "Multiple Choice (MCQ) exam question" in prompt:
            texts = (
                mcq_pool
                if model == 1
                else [_distinct_stem("mcq", model, i) for i in range(count)]
            )
            return {"questions": [
                {"question": texts[i], "options": options, "correct_answer": "A"}
                for i in range(min(count, len(texts)))
            ]}
        if "True/False exam question" in prompt:
            texts = (
                tf_pool
                if model == 1
                else [_distinct_stem("true_false", model, i) for i in range(count)]
            )
            return {"questions": [
                {"statement": texts[i], "answer": "True"}
                for i in range(min(count, len(texts)))
            ]}
        if model == 1:
            return {"questions": [
                {"question": text, "reference_answer": ref}
                for text, ref in sa_pool[:count]
            ]}
        stems = [_distinct_stem("short_answer", model, i) for i in range(count)]
        return {"questions": [
            {
                "question": stems[i],
                "reference_answer": f"this is the {i} outcome {model}",
            }
            for i in range(count)
        ]}

    class FakeLLM:
        def chat(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, timeout=600):
            return fake_chat(
                self, prompt, system_prompt=system_prompt, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout,
            )

        def chat_json(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096,
                      timeout=600, max_repair_attempts=2):
            return fake_chat(
                self, prompt, system_prompt=system_prompt, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout,
            )

    return FakeLLM


def _install_registry_and_store(monkeypatch, document_id: str = "doc-x") -> None:
    # Registry fakes (avoid writing data/documents.json).
    monkeypatch.setattr("app.api.storage.get_current_document", lambda: document_id)
    monkeypatch.setattr(
        "app.api.storage.get_document",
        lambda doc_id: {"document_id": doc_id, "filename": "x.pdf"},
    )

    # Vector store fake.
    class FakeStore:
        def get_by_child_ids(self, document_id, child_ids):
            return [
                {
                    "child_id": cid,
                    "parent_id": "p1",
                    "parent_title": "Parent",
                    "chunk_title": f"Sub {i}",
                    "page": 1,
                    "content": f"Context paragraph number {i} with useful facts.",
                }
                for i, cid in enumerate(child_ids)
            ]

    monkeypatch.setattr("app.online.retrieval.VectorStore", FakeStore)


def _install_fakes(monkeypatch, document_id: str = "doc-x") -> None:
    _install_registry_and_store(monkeypatch, document_id)
    # LLM fake: serves both the planner role and the generator role.
    fake = _make_fake_llm()
    monkeypatch.setattr("app.online.planner.LMStudioClient", fake)
    monkeypatch.setattr("app.llm.client.LMStudioClient", fake)


def test_generate_single_type_mcq(monkeypatch):
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate", json={"document_id": "doc-x", "mcq_count": 1, "child_ids": ["c1"]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["exams"]) == 1
    exam = data["exams"][0]
    assert exam["model_number"] == 1
    assert "Multiple Choice" in exam["markdown"]
    assert "Answer: A" in exam["markdown"]
    assert exam["questions"]["mcq"][0]["correct_answer"] == "A"


def test_generate_html_payload_multiple_types(monkeypatch):
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate",
        json={
            "mcq_count": 1,
            "tf_count": 1,
            "fitb_count": 3,
            "why_count": 1,
            "essay_count": 2,
            "num_models": 2,
            "difficulty": "hard",
            "child_ids": ["c1", "c2"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_models"] == 2
    assert data["difficulty"] == "hard"
    assert len(data["exams"]) == 2
    for exam in data["exams"]:
        assert set(exam["questions"].keys()) == {
            "mcq", "true_false", "fill_in_the_blank", "short_answer", "essay",
        }
        assert "Multiple Choice" in exam["markdown"]
        assert "True / False" in exam["markdown"]
        assert "Fill in the Blank" in exam["markdown"]
        assert "Short Answer" in exam["markdown"]
        assert "Essay" in exam["markdown"]
        # FITB: one shared Word Bank + exactly the requested number of items.
        fitb = exam["questions"]["fill_in_the_blank"]
        assert len(fitb["items"]) == 3
        assert len(fitb["word_bank"]) == 3 + 2  # 3 correct + exactly 2 distractors
        assert any("Word Bank" in exam["markdown"] for _ in [0])


def test_generate_num_models_returns_separate_full_exams(monkeypatch):
    # The specific requirement: 3 models, each with its own full 10/5/3 counts.
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate",
        json={
            "document_id": "doc-x",
            "mcq_count": 10,
            "tf_count": 5,
            "why_count": 3,
            "num_models": 3,
            "difficulty": "medium",
            "child_ids": ["c1"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_models"] == 3
    assert len(data["exams"]) == 3
    for exam in data["exams"]:
        assert len(exam["questions"]["mcq"]) == 10
        assert len(exam["questions"]["true_false"]) == 5
        assert len(exam["questions"]["short_answer"]) == 3


def test_difficulty_directive_in_prompt(monkeypatch):
    expected = {
        "easy": "Generate EASY questions",
        "medium": "Generate MEDIUM questions",
        "hard": "Generate HARD questions",
        "mix": "EASY, MEDIUM, and HARD",
    }
    for difficulty, needle in expected.items():
        captured: list[str] = []
        _install_registry_and_store(monkeypatch)
        fake = _make_fake_llm(captured)
        monkeypatch.setattr("app.online.planner.LMStudioClient", fake)
        monkeypatch.setattr("app.llm.client.LMStudioClient", fake)
        resp = client.post(
            "/generate",
            json={
                "document_id": "doc-x",
                "mcq_count": 1,
                "difficulty": difficulty,
                "child_ids": ["c1"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["difficulty"] == difficulty
        assert captured, f"no prompts captured for difficulty={difficulty}"
        # The difficulty directive lives in the generator prompt, not the planner.
        gen_prompts = [p for p in captured if "Create exactly" in p]
        assert gen_prompts, "no generator prompt captured"
        assert needle in gen_prompts[0]


def test_generate_invalid_num_models_returns_400(monkeypatch):
    _install_fakes(monkeypatch)
    for bad in (0, 5, 7):
        resp = client.post(
            "/generate",
            json={"document_id": "doc-x", "mcq_count": 1, "num_models": bad, "child_ids": ["c1"]},
        )
        assert resp.status_code == 400
        assert "num_models" in resp.json()["detail"]


def test_generate_invalid_difficulty_returns_400(monkeypatch):
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate",
        json={"document_id": "doc-x", "mcq_count": 1, "difficulty": "ultra", "child_ids": ["c1"]},
    )
    assert resp.status_code == 400
    assert "difficulty" in resp.json()["detail"]


def test_generate_no_supported_types_returns_400(monkeypatch):
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate",
        json={
            "mcq_count": 0,
            "tf_count": 0,
            "fitb_count": 0,
            "why_count": 0,
            "essay_count": 0,
            "child_ids": ["c1"],
        },
    )
    assert resp.status_code == 400
    assert "No supported question types" in resp.json()["detail"]


def test_generate_no_selection_returns_400(monkeypatch):
    _install_fakes(monkeypatch)
    resp = client.post("/generate", json={"document_id": "doc-x", "mcq_count": 1})
    assert resp.status_code == 400
    assert "choose at least one section" in resp.json()["detail"]


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
    # The fake LLM returns exactly the requested number per call; requesting 2
    # must fill the count and render continuously-numbered sections.
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate", json={"document_id": "doc-x", "mcq_count": 2, "tf_count": 2, "child_ids": ["c1"]}
    )
    assert resp.status_code == 200
    data = resp.json()
    exam = data["exams"][0]
    assert len(exam["questions"]["mcq"]) == 2
    assert len(exam["questions"]["true_false"]) == 2
    md = exam["markdown"]
    assert md.index("1. carbohydrate monomers link via glycosidic bonds") < md.index("## True / False")


def test_generate_repairs_malformed_json(monkeypatch):
    # Fake LLM returns broken JSON on the generation call and valid JSON on the
    # repair call; the request must still succeed via chat_json's repair loop.
    from app.llm.json_utils import REPAIR_SYSTEM_PROMPT, build_repair_prompt, extract_json

    calls = {"n": 0}

    def fake_chat(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, timeout=600):
        calls["n"] += 1
        if "Plan exam questions for" in prompt:
            return json.dumps(_fake_planner(prompt, 1))  # valid planner output as JSON text
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

    class FakeStore:
        def get_by_child_ids(self, document_id, child_ids):
            return [
                {
                    "child_id": cid, "parent_id": "p1", "parent_title": "P",
                    "chunk_title": "Sub", "page": 1, "content": f"Context {i}.",
                }
                for i, cid in enumerate(child_ids)
            ]

    monkeypatch.setattr("app.online.retrieval.VectorStore", FakeStore)
    monkeypatch.setattr("app.online.planner.LMStudioClient", FakeLLM)
    monkeypatch.setattr("app.llm.client.LMStudioClient", FakeLLM)

    resp = client.post("/generate", json={"document_id": "doc-x", "mcq_count": 1, "child_ids": ["c1"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["exams"][0]["questions"]["mcq"][0]["question"].startswith("Repaired question?")


def test_generate_empty_selection_returns_400(monkeypatch):
    _install_fakes(monkeypatch)
    resp = client.post(
        "/generate", json={"document_id": "doc-x", "mcq_count": 1, "child_ids": []}
    )
    assert resp.status_code == 400
    assert "choose at least one section topic" in resp.json()["detail"]


def test_generate_passes_child_ids_to_retrieval(monkeypatch):
    _install_fakes(monkeypatch)

    captured = {}

    class CapturingStore:
        def get_by_child_ids(self, document_id, child_ids):
            captured["child_ids"] = child_ids
            return [
                {
                    "child_id": cid,
                    "parent_id": "p1",
                    "parent_title": "Parent",
                    "chunk_title": "Sub",
                    "page": 1,
                    "content": f"Selected context for {cid}.",
                }
                for cid in child_ids
            ]

    monkeypatch.setattr("app.online.retrieval.VectorStore", CapturingStore)

    resp = client.post(
        "/generate",
        json={
            "document_id": "doc-x",
            "mcq_count": 1,
            "child_ids": ["c1", "c2", "c3"],
        },
    )
    assert resp.status_code == 200
    assert captured["child_ids"] == ["c1", "c2", "c3"]
