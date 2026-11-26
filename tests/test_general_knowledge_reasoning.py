"""Automated tests for Phase 3: General Knowledge, Reasoning, and RAG Hygiene."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aarohan.api import create_app
from aarohan.nlp.intent import Intent, IntentClassifier


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("phase3_test")
    db_path = tmp_dir / "aarohan_test.sqlite"
    app = create_app(database_path=db_path)
    return TestClient(app)


def test_intent_classification_taxonomy():
    """Validates that factual, reasoning, coding, and search intents are accurately resolved."""
    ic = IntentClassifier()

    cases = [
        ("What is the capital of India?", Intent.FACTUAL_QUESTION),
        ("What is the capital of Japan?", Intent.FACTUAL_QUESTION),
        ("Who invented the telephone?", Intent.FACTUAL_QUESTION),
        ("What is the chemical formula of water?", Intent.FACTUAL_QUESTION),
        ("How many planets are in our solar system?", Intent.FACTUAL_QUESTION),
        ("Explain the difference between AI and machine learning.", Intent.REASONING),
        ("Compare and contrast supervised and unsupervised learning.", Intent.REASONING),
        ("Write a Python program to calculate factorial.", Intent.CODE_GENERATION),
        ("Explain SQL joins with examples.", Intent.SQL_ASSISTANCE),
        ("Search the web for latest quantum computing breakthroughs.", Intent.WEB_SEARCH),
    ]

    for query, expected_intent in cases:
        res = ic.classify(query)
        assert res.primary_intent == expected_intent, f"Query '{query}' classified as {res.primary_intent}, expected {expected_intent}"
        assert res.intents[0].confidence >= 0.70


def test_general_knowledge_capitals_and_science(client):
    """Verifies that foundation model answers geography and basic science accurately without refusals."""
    queries = [
        ("What is the capital of India?", "Delhi"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the chemical formula of water?", "H₂O"),
    ]

    for q, expected_token in queries:
        resp = client.post("/api/chat", json={"query": q, "engine": "foundation", "max_new_tokens": 80})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response_source"] == "aarohan_foundation_engine"
        assert data["knowledge_source"] == "model_weights"
        assert data["citations"] == []  # Zero fabricated citations for parametric knowledge
        assert data["query_type"] == "factual"
        assert expected_token.lower() in data["response"].lower()
        assert not data.get("blocked", False)


def test_coding_and_reasoning_generation(client):
    """Verifies that coding requests and conceptual reasoning are generated with proper structure."""
    # 1. Coding
    code_resp = client.post(
        "/api/chat",
        json={"query": "Write a Python program to calculate factorial.", "engine": "foundation", "max_new_tokens": 150},
    )
    assert code_resp.status_code == 200
    code_data = code_resp.json()
    assert code_data["query_type"] == "coding"
    assert "def " in code_data["response"] or "factorial" in code_data["response"]

    # 2. Reasoning
    reason_resp = client.post(
        "/api/chat",
        json={"query": "Explain the difference between AI and machine learning.", "engine": "foundation", "max_new_tokens": 150},
    )
    assert reason_resp.status_code == 200
    reason_data = reason_resp.json()
    assert reason_data["query_type"] == "reasoning"
    assert "intelligence" in reason_data["response"].lower() or "learning" in reason_data["response"].lower()


def test_context_anti_pollution_on_open_domain(client):
    """Ensures unrelated fictional or research chunks in database are NOT injected into general queries."""
    # Add an unrelated fictional memory to the store
    mem_resp = client.post(
        "/api/memory",
        json={
            "content": "Senjō no Valkyria 3 is a tactical RPG video game developed for the PlayStation Portable.",
            "memory_type": "semantic",
            "source": "autonomous_research",
            "confidence": 0.9,
            "verification_status": "verified",
            "verified_by": "test_harness",
        },
    )
    assert mem_resp.status_code == 200

    # Ask an open-domain question about Japan
    q_resp = client.post(
        "/api/chat",
        json={"query": "What is the capital of Japan?", "engine": "foundation", "max_new_tokens": 50},
    )
    assert q_resp.status_code == 200
    data = q_resp.json()

    # The retrieved memory must NOT contain the video game chunk in the prompt context
    retrieved = data.get("retrieved_memory", [])
    assert len(retrieved) == 0 or not any("Valkyria" in m["content"] for m in retrieved)
    assert "Tokyo" in data["response"]


def test_models_registry_and_health_transparency(client):
    """Verifies that /api/models exposes provider-independent engine registry and health info."""
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "inference_engines" in data
    assert "health" in data
    engines = data["inference_engines"]
    assert any("qwen" in e["model_id"].lower() for e in engines)
